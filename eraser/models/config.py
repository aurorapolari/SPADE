"""Paths and explicit configuration for the released examples."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen_coder", "llama3")
LANGUAGES = ("python", "java")
Models_Path = {
    "qwen_coder": str(ROOT / "models" / "Qwen2.5-Coder-7B"),
    "llama3": str(ROOT / "models" / "Meta-Llama-3.1-8B-Instruct"),
}


def load_config(model, lang, path=None):
    if model not in MODELS or lang not in LANGUAGES:
        raise ValueError(f"Released examples: {MODELS}, languages: {LANGUAGES}")
    path = Path(path) if path else ROOT / "configs" / f"{model}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["model_key"] != model:
        raise ValueError("Configuration model_key does not match --model")
    config["lang"] = lang
    config["k"] = config["k_by_language"][lang]
    return config


def add_model_arguments(parser):
    parser.add_argument("--model", choices=MODELS, default="qwen_coder")
    parser.add_argument("--lang", choices=LANGUAGES, default="python")
    parser.add_argument("--model-path", help="Local Hugging Face model directory or repository ID")
    parser.add_argument("--config", help="Optional JSON configuration, including k_by_language")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--artifact-dir", help="Locally built artifacts; defaults to artifacts/LANG/MODEL")


def artifact_dir(args):
    return Path(args.artifact_dir) if args.artifact_dir else ROOT / "artifacts" / args.lang / args.model


def construction_path(lang, filename):
    return ROOT / "data" / "construction" / lang / filename


def get_best_pca_pos(model_name, lang):
    config = load_config(model_name, lang)
    return {"pos": [config["position"], "forward"], "k": config["k"]}
