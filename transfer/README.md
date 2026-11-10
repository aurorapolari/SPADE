# Transfer evaluation

| Directory | Task | Language | Source |
|---|---|---|---|
| `security/` | Security-sensitive function identification | Python | [ManyTypes4Py](https://doi.org/10.1109/MSR52588.2021.00079), with task-specific labels |
| `vul/` | Vulnerability detection | Java | [OWASP Benchmark](https://github.com/OWASP-Benchmark/BenchmarkJava) |

Each task contains:

```text
test/ORIGINAL.jsonl        Original test code
test/NET.jsonl             Network-themed renamed code
test/GAME.jsonl            Game-themed renamed code
calibration/ORIGINAL.jsonl Calibration code
```

Files use UTF-8 JSONL. Records contain `id`, `code`, `label`, `language`,
`input_sha256` and task metadata. Renamed records also contain `variant_id`;
vulnerability variants include the identifier `mapping`.

Test conditions are paired by `id` and share labels. The same inputs can be used
across models and Base/SPADE. A label of `1` denotes a security-sensitive function
for `security`, or vulnerable code for `vul`; `0` denotes the negative class.
Calibration data is separate from the test data.

File counts and checksums are recorded in [data/manifest.json](../data/manifest.json).

## Evaluation

Run from the repository root after constructing the SPADE artifacts:

```bash
python evaluate_transfer.py --task security --model qwen_coder --model-path /path/to/Qwen2.5-Coder-7B --method base
python evaluate_transfer.py --task security --model qwen_coder --model-path /path/to/Qwen2.5-Coder-7B --method spade
```

Use `--task vul` for vulnerability detection, or `--model llama3` with the
corresponding model path. The task selects the language and artifact directory.
The script calibrates the classification threshold, evaluates the test conditions,
and reports TPR, FPR and balanced accuracy. Results and predictions are saved to
`out/transfer/TASK/MODEL/METHOD.json`. Use `--limit` for a short test run.
