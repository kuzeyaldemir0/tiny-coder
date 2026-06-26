from __future__ import annotations

import statistics
from pathlib import Path
from typing import Iterator

from .constants import SOURCES, SPECIAL_TOKENS
from .io_utils import load_jsonl


def sample_texts_for_tokenizer(sample_dir: Path, max_docs_per_source: int | None = None) -> Iterator[str]:
    for source in SOURCES:
        path = sample_dir / f"{source}.jsonl"
        if not path.exists():
            continue
        for i, row in enumerate(load_jsonl(path)):
            if max_docs_per_source is not None and i >= max_docs_per_source:
                break
            text = row.get("text") or ""
            if text:
                yield text


def train_one_tokenizer(vocab_size: int, texts: list[str], output_path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <eos>",
        special_tokens=[("<eos>", eos_id)],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def tokenizer_report(sample_dir: Path, tokenizer_paths: list[Path], max_docs_per_source: int) -> dict:
    rows_by_source = {
        source: list(load_jsonl(sample_dir / f"{source}.jsonl"))
        for source in SOURCES
        if (sample_dir / f"{source}.jsonl").exists()
    }
    report = {"tokenizers": {}}
    for tokenizer_path in tokenizer_paths:
        tokenizer = load_tokenizer(tokenizer_path)
        one_report = {}
        for source, rows in rows_by_source.items():
            texts = [row.get("text", "") for row in rows[:max_docs_per_source]]
            token_counts = [len(tokenizer.encode(text).ids) for text in texts if text]
            byte_counts = [len(text.encode("utf-8", errors="ignore")) for text in texts if text]
            if not token_counts:
                continue
            one_report[source] = {
                "docs": len(token_counts),
                "tokens": sum(token_counts),
                "bytes": sum(byte_counts),
                "bytes_per_token": sum(byte_counts) / sum(token_counts),
                "mean_tokens": statistics.mean(token_counts),
                "median_tokens": statistics.median(token_counts),
            }
        report["tokenizers"][str(tokenizer_path)] = one_report
    return report
