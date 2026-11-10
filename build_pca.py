"""Build the configured PCA basis from released paired construction data."""
import argparse

from eraser.models.config import add_model_arguments, artifact_dir, construction_path, load_config
from eraser.data import read_records, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--data", help="Paired JSON/JSONL[.gz]; defaults to the selected model example")
    parser.add_argument("--limit", type=int, help="Limit pairs for a smaller construction run")
    args = parser.parse_args()
    from eraser.probe.pca import fit_pca
    from eraser.models.llm_wrapper import LLMWrapper, set_seed
    from eraser.erasers.erasers import PCAEraser

    config = load_config(args.model, args.lang, args.config)
    source = args.data or construction_path(args.lang, f"pca_{args.model}.jsonl.gz")
    rows = read_records(source, args.limit)
    set_seed(config["seed"])
    model = LLMWrapper.from_config(config, args.model_path, args.device, args.dtype)
    if not 0 < config["k"] <= model.dimension:
        raise ValueError("Configured k exceeds the selected hook dimension")
    basis, occurrences, used = fit_pca(model, rows)
    output = artifact_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    PCAEraser(basis).save(output / "pca.pt")
    save_json(output / "pca.json", {
        **model.identity(), "k": config["k"], "input_pairs": len(rows),
        "pairs": used, "occurrences": occurrences,
        "data_sha256": sha256(source), "seed": config["seed"],
        "estimator": "centered PCA on paired identifier-occurrence mean differences",
    })
    print(f"Saved PCA basis to {output}")


if __name__ == "__main__":
    main()
