"""Compare Base/SPADE on paired clean and renamed summary inputs."""
import argparse

from eraser.models.config import ROOT, add_model_arguments, artifact_dir, load_config
from eraser.data import read_records, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--method", choices=("base", "spade", "spade_no_gate"), default="spade")
    parser.add_argument("--input", help="Paired evaluation JSON/JSONL[.gz] with reference summaries")
    parser.add_argument("--output", help="Defaults to out/LANG/MODEL/METHOD.json")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    from eraser.hooks.resid_hooks import load_spade_hook
    from eraser.metrics import check_resources, score_summaries
    from eraser.models.llm_wrapper import LLMWrapper, set_seed
    from tqdm import tqdm

    config = load_config(args.model, args.lang, args.config)
    source = args.input or ROOT / "data" / "evaluation" / "summary" / f"{args.lang}.jsonl.gz"
    rows = read_records(source, args.limit, references=True)
    check_resources()
    model = LLMWrapper.from_config(config, args.model_path, args.device, args.dtype)
    intervention = None if args.method == "base" else load_spade_hook(model, artifact_dir(args), args.method == "spade")
    predictions = []
    for index, row in enumerate(tqdm(rows, desc=f"Evaluate {args.method}")):
        result = {"id": row.get("id", index)}
        for condition, field in (("clean", "code"), ("renamed", "vari_code")):
            set_seed(config["seed"] + index)
            encoded = model.encode(row[field], row.get("func_name", ""))
            prediction = model.generate(encoded, intervention, args.temperature, args.max_new_tokens)
            result[condition] = {"prediction": prediction}
            if intervention is not None:
                result[condition].update(gate_score=intervention.score, intervened=intervention.triggered)
        predictions.append(result)
    scores = {condition: score_summaries([row[condition]["prediction"] for row in predictions], rows)
              for condition in ("clean", "renamed")}
    output = args.output or ROOT / "out" / args.lang / args.model / f"{args.method}.json"
    save_json(output, {"model": model.identity(), "method": args.method, "config": config,
                       "temperature": args.temperature if args.temperature is not None else config["temperature"],
                       "max_new_tokens": args.max_new_tokens or config["max_new_tokens"],
                       "data_sha256": sha256(source), "samples": len(rows), "scores": scores, "predictions": predictions})
    print(scores)
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
