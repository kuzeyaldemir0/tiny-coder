from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator
from urllib.request import urlopen

from .constants import SWH_CONTENT_URL


def load_dataset_stream(dataset: str, config: str, split: str = "train"):
    from datasets import load_dataset

    return load_dataset(dataset, config, split=split, streaming=True)


def load_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_split_key(source: str, row_id: str, text: str) -> int:
    raw = f"{source}\0{row_id}\0{text_hash(text)}".encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def one_line(text: str, max_chars: int = 180) -> str:
    return " ".join(text.split())[:max_chars]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def rough_token_count(text: str) -> int:
    return max(1, round(len(text.encode("utf-8", errors="ignore")) / 4))


def download_python_contents(blob_id: str, timeout: int = 15) -> str:
    with urlopen(f"{SWH_CONTENT_URL}/{blob_id}", timeout=timeout) as response:
        return gzip.decompress(response.read()).decode("utf-8", errors="ignore")
