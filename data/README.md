# Data

Files use UTF-8 JSONL compressed with gzip and can be read directly by the scripts.
`manifest.json` contains record counts, provenance and SHA-256 checksums. Counts
refer to pairs; each clean function can have multiple renamed variants.

## Construction

| Language | Qwen PCA pairs | Llama PCA pairs | Gate training pairs | Gate validation pairs |
|---|---:|---:|---:|---:|
| Python | 8,000 | 8,000 | 20,000 | 4,000 |
| Java | 8,000 | 8,000 | 20,000 | 6,000 |

PCA data is specific to the selected model. Gate training and validation data is
shared by both models within each language. The splits are disjoint by clean
function and separate from the main evaluation data.

```json
{"id":"...","func_name":"...","code":"original code","vari_code":"renamed code","source_index":123}
```

`id` identifies a released record. `source_index`, when present, is the original
source-list index. Construction does not require reference summaries.

## Evaluation

| File under `evaluation/` | Pairs | Scope |
|---|---:|---|
| `summary/python.jsonl.gz` | 8,000 | Python summarization |
| `summary/java.jsonl.gz` | 10,000 | Java summarization |
| `accent/eval_accent.jsonl.gz` | 100 | ACCENT-native, Qwen/Python |
| `accent/eval_accent_crossdomain.jsonl.gz` | 100 | ACCENT with misleading vocabulary, Qwen/Python |

Evaluation records use the same paired schema, with a `references` list containing
three reference summaries. Each reference set is scored separately and the three
scores are averaged. The same references are used for clean and renamed code.
Python/Java summary sets are shared across models.

The programs derive from CodeXGLUE's Python/Java summarization data; references
are LLM-generated summaries. Original program licenses and attribution apply.

