"""Read plain JSON/JSONL or gzip-compressed JSONL without a download step."""
import gzip
import hashlib
import json
from pathlib import Path


def read_records(path, limit=None, paired=True, references=False):
    path = Path(path)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if path.name.endswith(".json"):
            raw = stream.read()
            if raw.startswith("version https://git-lfs.github.com/spec/"):
                raise ValueError(f"{path} is a Git LFS pointer, not data")
            rows = json.loads(raw)
        else:
            rows = []
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty list of records")
    rows = rows[:limit] if limit else rows
    for index, row in enumerate(rows):
        required = ("code", "vari_code") if paired else ("code",)
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
            raise ValueError(f"{path}: record {index} requires nonempty {required}")
        if references:
            refs = row.get("references")
            if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or not x.strip() for x in refs):
                raise ValueError(f"{path}: record {index} requires nonempty references")
    return rows


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def assert_disjoint(train, valid):
    # Group by clean code: different renamings of one function belong to one split.
    if {x["code"] for x in train} & {x["code"] for x in valid}:
        raise ValueError("Training and validation contain the same clean function")
