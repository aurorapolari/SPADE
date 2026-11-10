"""Train and calibrate a selective gate from the released construction pairs."""
import argparse

from eraser.models.config import add_model_arguments, artifact_dir, construction_path, load_config
from eraser.data import assert_disjoint, read_records, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--train-data", help="Defaults to shared gate_train.jsonl.gz for the selected language")
    parser.add_argument("--valid-data", help="Defaults to shared gate_valid.jsonl.gz for the selected language")
    parser.add_argument("--limit", type=int, help="Limit pairs in each split for a smaller construction run")
    args = parser.parse_args()
    from eraser.gate.train import fit_gate
    from eraser.models.llm_wrapper import LLMWrapper
    import torch

    config = load_config(args.model, args.lang, args.config)
    train_path = args.train_data or construction_path(args.lang, "gate_train.jsonl.gz")
    valid_path = args.valid_data or construction_path(args.lang, "gate_valid.jsonl.gz")
    train, valid = read_records(train_path, args.limit), read_records(valid_path, args.limit)
    assert_disjoint(train, valid)
    model = LLMWrapper.from_config(config, args.model_path, args.device, args.dtype)
    best, history = fit_gate(model, train, valid)
    output = artifact_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best.pop("state"), output / "gate.pt")
    save_json(output / "gate.json", {
        **model.identity(), **config["gate"], **best, "seed": config["seed"],
        "input_training_pairs": len(train), "input_validation_pairs": len(valid),
        "train_data_sha256": sha256(train_path), "valid_data_sha256": sha256(valid_path),
        "feature": "ident_mean,ident_max,ctx_mean,ctx_max,abs_diff,product,cosine",
        "history": history,
    })
    print(f"Saved gate and calibrated threshold to {output}")


if __name__ == "__main__":
    main()
