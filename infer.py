"""Generate summaries with locally constructed SPADE artifacts."""
import argparse

from eraser.models.config import ROOT, add_model_arguments, artifact_dir, load_config
from eraser.data import read_records, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--input", default=str(ROOT / "examples" / "input.jsonl"))
    parser.add_argument("--field", choices=("code", "vari_code"), default="code")
    parser.add_argument("--output", default=str(ROOT / "out" / "predictions.json"))
    parser.add_argument("--no-gate", action="store_true", help="Always apply identifier projection")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    from eraser.hooks.resid_hooks import load_spade_hook
    from eraser.models.llm_wrapper import LLMWrapper, set_seed

    config = load_config(args.model, args.lang, args.config)
    rows = read_records(args.input, args.limit, paired=args.field == "vari_code")
    model = LLMWrapper.from_config(config, args.model_path, args.device, args.dtype)
    intervention = load_spade_hook(model, artifact_dir(args), use_gate=not args.no_gate)
    predictions = []
    for index, row in enumerate(rows):
        set_seed(config["seed"] + index)
        encoded = model.encode(row[args.field], row.get("func_name", ""))
        prediction = model.generate(encoded, intervention, args.temperature, args.max_new_tokens)
        predictions.append({"id": row.get("id", index), "prediction": prediction,
                            "gate_score": intervention.score, "intervened": intervention.triggered})
    save_json(args.output, {"model": model.identity(), "field": args.field, "predictions": predictions})
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
