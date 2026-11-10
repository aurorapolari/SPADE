"""Evaluate Base/SPADE on security and vulnerability classification."""
import argparse
from functools import partial

from eraser.data import read_records, save_json, sha256
from eraser.models.config import ROOT, add_model_arguments, artifact_dir, load_config

TASK_LANGUAGES = {"security": "python", "vul": "java"}
CONDITIONS = ("ORIGINAL", "NET", "GAME")
SECURITY_INSTRUCTION = (
    "Classify the following Python function as either security-related or ordinary. "
    "A function is security-related if it itself performs or directly invokes a security "
    "operation, such as encryption or decryption, hashing or message authentication "
    "(MAC/HMAC), digital signing or signature verification, generating or deriving keys "
    "or random key material, verifying or processing authentication credentials "
    "(passwords, tokens), or establishing a secure (TLS/SSL) connection; every other "
    "function is ordinary. Is this function security-related? Answer Yes or No."
)
VUL_INSTRUCTIONS = {
    "qwen_coder": (
        "Read the following Java code, infer what it is intended to do, and check whether "
        "the implementation achieves that intent securely or exposes a security weakness. "
        "Does this code contain a security vulnerability? Answer Yes or No."
    ),
    "llama3": (
        "Determine whether the following Java code contains a security vulnerability, "
        "such as injection (SQL, OS command, LDAP, XPath), cross-site scripting, path "
        "traversal, HTTP response splitting, open redirect, weak cryptography or hashing, "
        "insecure randomness, trust boundary violation, insecure cookie, integer overflow "
        "or underflow, division by zero, improper array index, uncontrolled resource "
        "allocation, or format string issues. Does this code contain a security "
        "vulnerability? Answer Yes or No."
    ),
}


def build_prompt(task, model_key, code, tokenizer):
    instruction = SECURITY_INSTRUCTION if task == "security" else VUL_INSTRUCTIONS[model_key]
    block = f"```{TASK_LANGUAGES[task]}\n{code}```"
    if model_key == "qwen_coder":
        return f"{instruction}\n\n{block}\nAnswer:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{instruction} Reply with only Yes or No.\n\n{block}"}],
        tokenize=False, add_generation_prompt=True,
    )


def classification_metrics(rows, threshold):
    positive = sum(row["label"] == 1 for row in rows)
    negative = len(rows) - positive
    tp = sum(row["label"] == 1 and row["score"] > threshold for row in rows)
    fp = sum(row["label"] == 0 and row["score"] > threshold for row in rows)
    tpr = tp / positive if positive else None
    fpr = fp / negative if negative else None
    return {"n": len(rows), "tp": tp, "fn": positive - tp, "fp": fp, "tn": negative - fp,
            "tpr": tpr, "fpr": fpr,
            "bacc": (tpr + 1 - fpr) / 2 if positive and negative else None}


def calibrate(rows, include_endpoints=False):
    """Maximize calibration balanced accuracy; break ties by ascending threshold."""
    positive = sum(row["label"] == 1 for row in rows)
    negative = len(rows) - positive
    if not positive or not negative:
        raise ValueError("Calibration requires both positive and negative labels")
    values = sorted({row["score"] for row in rows})
    candidates = [(left + right) / 2 for left, right in zip(values, values[1:])]
    if include_endpoints:
        candidates = [values[0] - 1, *candidates, values[-1] + 1]

    def quality(threshold):
        result = classification_metrics(rows, threshold)
        return result["tp"] * negative + result["tn"] * positive

    return max(candidates or values, key=quality)


def score_samples(model, rows, intervention, description):
    import torch
    from tqdm import tqdm

    yes = [model.tokenizer.encode(word, add_special_tokens=False)[0] for word in ("Yes", " Yes")]
    no = [model.tokenizer.encode(word, add_special_tokens=False)[0] for word in ("No", " No")]
    results = []
    for row in tqdm(rows, desc=description):
        encoded = model.encode(row["code"], row.get("func_name", "func"))
        if intervention is not None:
            intervention.reset(encoded)
            intervention.register(model.module)
        try:
            with torch.inference_mode():
                logits = model.forward_no_grad(
                    encoded.inputs["input_ids"], encoded.inputs.get("attention_mask"),
                ).logits[0, -1].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                score = torch.logsumexp(log_probs[yes], 0).item() - torch.logsumexp(log_probs[no], 0).item()
        finally:
            if intervention is not None:
                intervention.remove()
        result = {"id": row["id"], "label": row["label"], "score": round(score, 5)}
        if intervention is not None:
            result.update(gate_score=intervention.score, intervened=intervention.triggered)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.set_defaults(lang=None)
    parser.add_argument("--task", choices=TASK_LANGUAGES, required=True)
    parser.add_argument("--method", choices=("base", "spade"), default="spade")
    parser.add_argument("--data-dir", default=str(ROOT / "transfer"))
    parser.add_argument("--output", help="Defaults to out/transfer/TASK/MODEL/METHOD.json")
    parser.add_argument("--limit", type=int, help="Limit test samples per condition")
    args = parser.parse_args()
    lang = TASK_LANGUAGES[args.task]
    if args.lang is not None and args.lang != lang:
        parser.error(f"--task {args.task} requires --lang {lang}")
    args.lang = lang

    from pathlib import Path
    from eraser.hooks.resid_hooks import load_spade_hook
    from eraser.models.llm_wrapper import LLMWrapper, set_seed

    directory = Path(args.data_dir) / args.task
    sources = {"calibration": directory / "calibration" / "ORIGINAL.jsonl"}
    sources.update({condition: directory / "test" / f"{condition}.jsonl" for condition in CONDITIONS})
    calibration = read_records(sources["calibration"], paired=False)
    tests = {condition: read_records(sources[condition], args.limit, paired=False) for condition in CONDITIONS}
    for rows in (calibration, *tests.values()):
        if len({row["id"] for row in rows}) != len(rows) or any(row["label"] not in (0, 1) for row in rows):
            raise ValueError("Records require unique IDs and binary labels")
    expected = [(row["id"], row["label"]) for row in tests["ORIGINAL"]]
    if any([(row["id"], row["label"]) for row in rows] != expected for rows in tests.values()):
        raise ValueError("Test conditions must have matching IDs and labels in the same order")
    if {row["id"] for row in calibration} & {row["id"] for row in tests["ORIGINAL"]}:
        raise ValueError("Calibration and test IDs must be separate")

    config = load_config(args.model, lang, args.config)
    set_seed(config["seed"])
    model = LLMWrapper.from_config(config, args.model_path, args.device, args.dtype)
    model.mapper.prompt_proc = partial(build_prompt, args.task, args.model)
    intervention = None if args.method == "base" else load_spade_hook(model, artifact_dir(args))
    # Vulnerability detection shares the Base-calibrated threshold across methods.
    calibration_hook = intervention if args.task == "security" else None
    calibration_scores = score_samples(model, calibration, calibration_hook, "Calibrate")
    threshold = calibrate(calibration_scores, include_endpoints=args.task == "vul")
    predictions, scores = {}, {}
    for condition, rows in tests.items():
        predictions[condition] = score_samples(model, rows, intervention, condition)
        for row in predictions[condition]:
            row["prediction"] = int(row["score"] > threshold)
        scores[condition] = classification_metrics(predictions[condition], threshold)
    output = args.output or ROOT / "out" / "transfer" / args.task / args.model / f"{args.method}.json"
    save_json(output, {"model": model.identity(), "task": args.task, "method": args.method,
                       "threshold": threshold, "calibration_method": args.method if args.task == "security" else "base",
                       "calibration_scores": calibration_scores,
                       "data_sha256": {key: sha256(path) for key, path in sources.items()},
                       "scores": scores, "predictions": predictions})
    print(scores)
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
