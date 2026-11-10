# SPADE: Selective Projection via Approximate Direction Estimation

This repository provides the core SPADE implementation, paired construction data,
and scripts for construction, inference and evaluation. Python and Java data are
included. PCA bases and selective gates are built locally.

The entrypoints cover PCA construction, gate training, inference and evaluation.
Transfer evaluation data is available in `transfer/`.

## Installation

Use Python 3.10 or later, with a PyTorch build appropriate for your CUDA version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

The NLTK download is used for METEOR scoring.

Download the base model from its official distribution. Pass `--model-path`
to use a local model directory. If omitted, the configured
Hugging Face ID is used. Use the same model source for construction and inference.
Large base model weights are not part of this repository.

## Construct SPADE locally

From this directory, build the PCA basis and train the selective gate.
The following commands use Qwen2.5-Coder as an example:

```bash
python build_pca.py --model qwen_coder --lang python --model-path /path/to/Qwen2.5-Coder-7B --device cuda:0
python train_gate.py --model qwen_coder --lang python --model-path /path/to/Qwen2.5-Coder-7B --device cuda:0
```

Use `--model` to select a model configuration and `--model-path` to specify its
location. The commands also accept `--lang java`. Each model/language
combination needs its own artifacts, written to `artifacts/LANG/MODEL/` by default:

```text
pca.pt       PCA basis
pca.json     PCA metadata
gate.pt      Selective gate
gate.json    Gate metadata
```

Use `--artifact-dir` to change the output directory. Use `--data` for custom PCA pairs, or `--train-data` and
`--valid-data` for custom gate pairs. Plain JSON lists, JSONL and `.jsonl.gz` inputs
are supported. Use `--limit` for a subset of the data and `--config` for a custom
configuration.

Model and runtime settings are provided in `configs/`.

## Inference

```bash
python infer.py --model qwen_coder --lang python --model-path /path/to/Qwen2.5-Coder-7B --input examples/input.jsonl --field vari_code
```

For your own input, each record needs `code`; `id` and `func_name` are optional.
Use `--field vari_code` to summarize the renamed member of a paired record.
Reference summaries are not required for inference. `--no-gate` applies projection
unconditionally. Outputs include the summary, gate score and intervention flag.
Build the required artifacts with the construction commands before inference.

Use `--device` and `--dtype` to select the execution device and precision.

## Summary evaluation

```bash
python evaluate.py --model qwen_coder --lang python --model-path /path/to/Qwen2.5-Coder-7B --method base
python evaluate.py --model qwen_coder --lang python --model-path /path/to/Qwen2.5-Coder-7B --method spade
```

The optional `--method spade_no_gate` evaluates unconditional projection. Each
run evaluates clean and renamed code. Use `--limit` for a short run.
Results and predictions are
written to `out/LANG/MODEL/METHOD.json`, or the path specified by `--output`.

Evaluation reports BLEU, METEOR and ROUGE-L.

The prepared ACCENT examples below use the same evaluation entrypoint:

```bash
python evaluate.py --model qwen_coder --lang python --model-path /path/to/Qwen2.5-Coder-7B --input data/evaluation/accent/eval_accent.jsonl.gz --output out/accent_native.json
python evaluate.py --model qwen_coder --lang python --model-path /path/to/Qwen2.5-Coder-7B --input data/evaluation/accent/eval_accent_crossdomain.jsonl.gz --output out/accent_misleading.json
```

## Transfer evaluation data

The `transfer/` directory contains security-sensitive function identification and
vulnerability detection data, including original code, renamed variants and
calibration splits. Run `evaluate_transfer.py` to evaluate Base or SPADE.

## Data and release scope

[data/manifest.json](data/manifest.json) records provenance and file hashes.

```text
configs/                 Model configurations
eraser/
  models/                LLMWrapper, model configuration and prompts
  ast/                   IdentiTokenMapper and parser utilities
  captures/              ActCaptures and shared activation collection
  erasers/               PCAEraser
  gate/                  GateModel, features and policy training
  hooks/                 CaptureHook, EraseHook and GateEraseHook
  probe/                 PCA direction estimation
build_pca.py             Construct the selected PCA basis
train_gate.py            Train/calibrate the selective gate
infer.py                 Summarize input code with SPADE
evaluate.py              Base / SPADE / unconditional projection evaluation
evaluate_transfer.py     Transfer classification evaluation
data/construction/       Paired construction data
data/evaluation/         Summary and prepared ACCENT evaluation inputs
examples/                Small input-format example
transfer/                Transfer evaluation data
```

This repository provides the core implementation of SPADE and the data and scripts
for its main evaluations.

