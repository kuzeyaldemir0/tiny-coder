from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .constants import (
    DEFAULT_PYTHON_DEDUP_BUFFER_DOCS,
    MATH_CONFIG,
    MATH_DATASET,
    MIN_TEXT_CHARS,
    PYTHON_CONFIG,
    PYTHON_DATASET,
    SOURCE_MATH,
    SOURCE_PYTHON,
    SOURCE_WEB,
    WEB_CONFIG,
    WEB_DATASET,
)
from .dedup import exact_dedup_hash, near_dedup_records
from .io_utils import download_python_contents, load_dataset_stream, load_jsonl, normalize_text, rough_token_count, text_hash
from .python_cleaning import clean_python_row


@dataclass
class FilterStats:
    source: str
    seen: int = 0
    accepted: int = 0
    chars: int = 0
    estimated_tokens: int = 0
    drop_reasons: Counter = field(default_factory=Counter)
    boilerplate_changed: int = 0
    boilerplate_chars_removed: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    near_clusters: int = 0

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "seen": self.seen,
            "accepted": self.accepted,
            "chars": self.chars,
            "estimated_tokens": self.estimated_tokens,
            "drop_reasons": dict(self.drop_reasons),
            "boilerplate_changed": self.boilerplate_changed,
            "boilerplate_chars_removed": self.boilerplate_chars_removed,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "near_clusters": self.near_clusters,
        }


def clean_plain_text_row(row: dict, source: str) -> tuple[dict | None, str | None]:
    text = normalize_text(row.get("text") or "")
    if len(text) < MIN_TEXT_CHARS:
        return None, "too_short"
    row_id = row.get("id") or row.get("url") or row.get("metadata") or text_hash(text)
    return {"id": str(row_id), "source": source, "text": text}, None


def hydrate_python_row(row: dict, download_timeout: int) -> dict:
    row = dict(row)
    if "text" not in row:
        try:
            row["text"] = download_python_contents(row["blob_id"], timeout=download_timeout)
        except Exception as exc:
            row["download_error"] = repr(exc)
            row["text"] = ""
    return row


def iter_python_rows(
    limit: int | None = None,
    sample_dir: Path | None = None,
    download_workers: int = 16,
    download_timeout: int = 15,
) -> Iterator[dict]:
    sample_path = sample_dir / "python_edu.jsonl" if sample_dir else None
    if sample_path and sample_path.exists():
        for i, row in enumerate(load_jsonl(sample_path)):
            if limit is not None and i >= limit:
                return
            yield row
        return

    stream = load_dataset_stream(PYTHON_DATASET, PYTHON_CONFIG)
    batch_size = max(32, download_workers * 4)
    batch = []
    seen = 0

    def flush_batch(rows: list[dict]) -> Iterator[dict]:
        if not rows:
            return
        if download_workers <= 1:
            for item in rows:
                yield hydrate_python_row(item, download_timeout)
            return
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = [executor.submit(hydrate_python_row, row, download_timeout) for row in rows]
            for future in as_completed(futures):
                yield future.result()

    for row in stream:
        if limit is not None and seen >= limit:
            break
        batch.append(row)
        seen += 1
        if len(batch) >= batch_size:
            yield from flush_batch(batch)
            batch = []

    yield from flush_batch(batch)


def iter_plain_rows(source: str, limit: int | None = None, sample_dir: Path | None = None) -> Iterator[dict]:
    sample_path = sample_dir / f"{source}.jsonl" if sample_dir else None
    if sample_path and sample_path.exists():
        for i, row in enumerate(load_jsonl(sample_path)):
            if limit is not None and i >= limit:
                return
            yield row
        return

    if source == SOURCE_WEB:
        stream = load_dataset_stream(WEB_DATASET, WEB_CONFIG)
    elif source == SOURCE_MATH:
        stream = load_dataset_stream(MATH_DATASET, MATH_CONFIG)
    else:
        raise ValueError(f"unknown plain text source: {source}")

    for i, row in enumerate(stream):
        if limit is not None and i >= limit:
            return
        yield row


def iter_clean_python(
    limit: int | None,
    stats: FilterStats,
    sample_dir: Path | None,
    dedup_buffer_docs: int = DEFAULT_PYTHON_DEDUP_BUFFER_DOCS,
    apply_near_dedup: bool = True,
    require_fasttext: bool = True,
    download_workers: int = 16,
    download_timeout: int = 15,
) -> Iterator[dict]:
    seen_hashes: set[str] = set()
    buffer: list[dict] = []

    def flush_buffer() -> Iterator[dict]:
        nonlocal buffer
        if not buffer:
            return
        kept = buffer
        decisions: list[dict] = []
        if apply_near_dedup:
            kept, decisions = near_dedup_records(buffer)
            stats.near_clusters += len(decisions)
            stats.near_duplicates += len(buffer) - len(kept)
        for rec in kept:
            yield rec
        buffer = []

    for row in iter_python_rows(
        limit=limit,
        sample_dir=sample_dir,
        download_workers=download_workers,
        download_timeout=download_timeout,
    ):
        stats.seen += 1
        if row.get("download_error"):
            stats.drop_reasons["download_error"] += 1
            continue
        rec, reason, meta = clean_python_row(row, require_fasttext=require_fasttext)
        if meta["boilerplate_changed"]:
            stats.boilerplate_changed += 1
            stats.boilerplate_chars_removed += meta["boilerplate_chars_removed"]
        if rec is None:
            stats.drop_reasons[reason or "unknown"] += 1
            continue
        h = exact_dedup_hash(rec["text"])
        if h in seen_hashes:
            stats.exact_duplicates += 1
            stats.drop_reasons["exact_duplicate"] += 1
            continue
        seen_hashes.add(h)
        if not apply_near_dedup:
            yield rec
            continue
        buffer.append(rec)
        if len(buffer) >= dedup_buffer_docs:
            yield from flush_buffer()

    yield from flush_buffer()


def iter_clean_plain(source: str, limit: int | None, stats: FilterStats, sample_dir: Path | None) -> Iterator[dict]:
    for row in iter_plain_rows(source, limit=limit, sample_dir=sample_dir):
        stats.seen += 1
        rec, reason = clean_plain_text_row(row, source)
        if rec is None:
            stats.drop_reasons[reason or "unknown"] += 1
            continue
        yield rec


def update_accept_stats(stats: FilterStats, rec: dict, token_count: int | None = None) -> None:
    stats.accepted += 1
    stats.chars += len(rec["text"])
    stats.estimated_tokens += token_count if token_count is not None else rough_token_count(rec["text"])


def iter_clean_source(
    source: str,
    limit: int | None,
    stats: FilterStats,
    sample_dir: Path | None = None,
    dedup_buffer_docs: int = DEFAULT_PYTHON_DEDUP_BUFFER_DOCS,
    apply_near_dedup: bool = True,
    require_fasttext: bool = True,
    download_workers: int = 16,
    download_timeout: int = 15,
) -> Iterator[dict]:
    if source == SOURCE_PYTHON:
        yield from iter_clean_python(
            limit=limit,
            stats=stats,
            sample_dir=sample_dir,
            dedup_buffer_docs=dedup_buffer_docs,
            apply_near_dedup=apply_near_dedup,
            require_fasttext=require_fasttext,
            download_workers=download_workers,
            download_timeout=download_timeout,
        )
    elif source in {SOURCE_WEB, SOURCE_MATH}:
        yield from iter_clean_plain(source, limit=limit, stats=stats, sample_dir=sample_dir)
    else:
        raise ValueError(f"unknown source: {source}")
