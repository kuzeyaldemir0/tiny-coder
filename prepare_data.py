from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from scripts.constants import (
    DEFAULT_PYTHON_DEDUP_BUFFER_DOCS,
    DEFAULT_SHARD_TOKENS,
    DEFAULT_VAL_FRACTION,
    SOURCE_MATH,
    SOURCE_PYTHON,
    SOURCE_WEB,
    SOURCES,
)
from scripts.io_utils import stable_split_key, write_jsonl
from scripts.shards import BinShardWriter
from scripts.sources import FilterStats, iter_clean_source, update_accept_stats
from scripts.tokenizers import load_tokenizer, sample_texts_for_tokenizer, tokenizer_report, train_one_tokenizer


def command_sample(args: argparse.Namespace) -> None:
    sample_dir = Path(args.output_dir)
    accepted_targets = {
        SOURCE_PYTHON: args.python_docs,
        SOURCE_WEB: args.web_docs,
        SOURCE_MATH: args.math_docs,
    }
    raw_limits = {
        SOURCE_PYTHON: args.python_max_docs,
        SOURCE_WEB: args.web_max_docs,
        SOURCE_MATH: args.math_max_docs,
    }

    for source in SOURCES:
        stats = FilterStats(source=source)
        rows = []
        for rec in iter_clean_source(
            source,
            limit=raw_limits[source],
            stats=stats,
            sample_dir=None,
            dedup_buffer_docs=args.python_dedup_buffer_docs,
            apply_near_dedup=not args.no_near_dedup,
            require_fasttext=not args.no_fasttext,
            download_workers=args.python_download_workers,
            download_timeout=args.python_download_timeout,
        ):
            rows.append(rec)
            update_accept_stats(stats, rec)
            if args.progress_every and len(rows) % args.progress_every == 0:
                print(f"{source}: accepted {len(rows)} / {accepted_targets[source]} rows after scanning {stats.seen}")
            if len(rows) >= accepted_targets[source]:
                break
        count = write_jsonl(sample_dir / f"{source}.jsonl", rows)
        print(f"{source}: wrote {count} cleaned rows to {sample_dir / f'{source}.jsonl'}")
        print(json.dumps(stats.as_dict(), indent=2, sort_keys=True))


def command_audit(args: argparse.Namespace) -> None:
    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    limits = {
        SOURCE_PYTHON: args.python_max_docs,
        SOURCE_WEB: args.web_max_docs,
        SOURCE_MATH: args.math_max_docs,
    }
    target_tokens = {
        SOURCE_PYTHON: args.python_target_tokens,
        SOURCE_WEB: args.web_target_tokens,
        SOURCE_MATH: args.math_target_tokens,
    }
    report = {"sources": {}, "targets": target_tokens}
    for source in SOURCES:
        stats = FilterStats(source=source)
        for rec in iter_clean_source(
            source,
            limit=limits[source],
            stats=stats,
            sample_dir=sample_dir,
            dedup_buffer_docs=args.python_dedup_buffer_docs,
            apply_near_dedup=not args.no_near_dedup,
            require_fasttext=not args.no_fasttext,
            download_workers=args.python_download_workers,
            download_timeout=args.python_download_timeout,
        ):
            update_accept_stats(stats, rec)
            if target_tokens[source] and stats.estimated_tokens >= target_tokens[source]:
                break
        report["sources"][source] = stats.as_dict()
        print(f"\n{source}")
        print(json.dumps(stats.as_dict(), indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote audit report: {output}")


def command_train_tokenizer(args: argparse.Namespace) -> None:
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    texts = list(sample_texts_for_tokenizer(sample_dir, max_docs_per_source=args.max_docs_per_source))
    if not texts:
        raise SystemExit(f"no sample texts found in {sample_dir}")
    for vocab_size in args.vocab_sizes:
        output_path = output_dir / f"tokenizer-{vocab_size}.json"
        print(f"training {output_path} on {len(texts)} sample documents")
        train_one_tokenizer(vocab_size, texts, output_path)


def command_tokenizer_report(args: argparse.Namespace) -> None:
    report = tokenizer_report(
        sample_dir=Path(args.sample_dir),
        tokenizer_paths=[Path(path) for path in args.tokenizers],
        max_docs_per_source=args.max_docs_per_source,
    )
    for tokenizer_path, one_report in report["tokenizers"].items():
        print(f"\n{tokenizer_path}")
        print(json.dumps(one_report, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote tokenizer report: {output}")


def command_build(args: argparse.Namespace) -> None:
    tokenizer_path = Path(args.tokenizer)
    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise SystemExit("tokenizer must include <eos>")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise SystemExit("tokenizer vocab exceeds uint16 range")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")

    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    target_tokens = {
        SOURCE_PYTHON: args.python_target_tokens,
        SOURCE_WEB: args.web_target_tokens,
        SOURCE_MATH: args.math_target_tokens,
    }
    source_order = args.source_order
    unknown_sources = sorted(set(source_order) - set(SOURCES))
    if unknown_sources:
        raise SystemExit(f"unknown sources in --source-order: {', '.join(unknown_sources)}")
    limits = {
        SOURCE_PYTHON: args.python_max_docs,
        SOURCE_WEB: args.web_max_docs,
        SOURCE_MATH: args.math_max_docs,
    }
    source_tokens = Counter()
    source_docs = Counter()
    source_stats = {}
    started_at = time.time()
    next_progress_tokens = {source: args.progress_every_tokens for source in SOURCES}
    writers = {
        "train": BinShardWriter(output_dir, "train", args.shard_tokens),
        "val": BinShardWriter(output_dir, "val", args.shard_tokens),
    }

    def write_progress(current_source: str | None = None) -> None:
        progress = {
            "status": "running",
            "elapsed_seconds": round(time.time() - started_at, 1),
            "current_source": current_source,
            "target_tokens": target_tokens,
            "actual_tokens": dict(source_tokens),
            "actual_docs": dict(source_docs),
            "shards": {split: writer.files for split, writer in writers.items()},
        }
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")

    def enforce_local_limit() -> None:
        used = sum(writer.written_bytes + writer.buffered_bytes for writer in writers.values())
        if used > args.max_local_gb * 1024**3:
            raise SystemExit(
                f"local build exceeded --max-local-gb={args.max_local_gb}; "
                "upload or clean completed shards before continuing"
            )

    for source in source_order:
        print(f"starting source={source} target_tokens={target_tokens[source]:,}", flush=True)
        stats = FilterStats(source=source)
        for rec in iter_clean_source(
            source,
            limit=limits[source],
            stats=stats,
            sample_dir=sample_dir,
            dedup_buffer_docs=args.python_dedup_buffer_docs,
            apply_near_dedup=not args.no_near_dedup,
            require_fasttext=not args.no_fasttext,
            download_workers=args.python_download_workers,
            download_timeout=args.python_download_timeout,
        ):
            ids = tokenizer.encode(rec["text"]).ids
            if not ids or ids[-1] != eos_id:
                ids.append(eos_id)
            split_key = stable_split_key(source, rec["id"], rec["text"]) % 1_000_000
            split = "val" if split_key < int(args.val_fraction * 1_000_000) else "train"
            writers[split].add(ids)
            update_accept_stats(stats, rec, token_count=len(ids))
            source_tokens[source] += len(ids)
            source_docs[source] += 1
            enforce_local_limit()
            if args.progress_every_tokens and source_tokens[source] >= next_progress_tokens[source]:
                rate = sum(source_tokens.values()) / max(time.time() - started_at, 1)
                print(
                    f"progress source={source} "
                    f"source_tokens={source_tokens[source]:,}/{target_tokens[source]:,} "
                    f"total_tokens={sum(source_tokens.values()):,} "
                    f"docs={source_docs[source]:,} "
                    f"rate={rate:,.0f} tok/s",
                    flush=True,
                )
                write_progress(current_source=source)
                next_progress_tokens[source] += args.progress_every_tokens
            if target_tokens[source] and source_tokens[source] >= target_tokens[source]:
                break
        source_stats[source] = stats.as_dict()
        print(
            f"finished source={source} tokens={source_tokens[source]:,} "
            f"docs={source_docs[source]:,} seen={stats.seen:,}",
            flush=True,
        )
        write_progress(current_source=source)

    for writer in writers.values():
        writer.close()

    meta = {
        "tokenizer": "tokenizer.json",
        "dtype": "uint16",
        "sources": source_stats,
        "target_tokens": target_tokens,
        "actual_tokens": dict(source_tokens),
        "actual_docs": dict(source_docs),
        "val_fraction": args.val_fraction,
        "shards": {split: writer.files for split, writer in writers.items()},
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "audit_report.json").write_text(json.dumps({"sources": source_stats}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "progress.json").write_text(
        json.dumps({"status": "complete", **meta}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


def command_upload(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, create_repo

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"missing input dir: {input_dir}")
    create_repo(args.hf_repo, repo_type="dataset", private=args.private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        repo_id=args.hf_repo,
        repo_type="dataset",
        folder_path=str(input_dir),
        commit_message=args.commit_message,
    )
    print(f"uploaded {input_dir} to dataset repo {args.hf_repo}")


def command_clean_local(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"nothing to clean: {input_dir}")
        return
    removed = 0
    for path in input_dir.glob("*.bin"):
        path.unlink()
        removed += 1
    if args.remove_reports:
        for name in ("meta.json", "audit_report.json", "tokenizer_report.json", "tokenizer.json"):
            path = input_dir / name
            if path.exists():
                path.unlink()
                removed += 1
    print(f"removed {removed} files from {input_dir}")


def add_common_stream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--python-max-docs", type=int, default=None)
    parser.add_argument("--web-max-docs", type=int, default=None)
    parser.add_argument("--math-max-docs", type=int, default=None)
    parser.add_argument("--python-dedup-buffer-docs", type=int, default=DEFAULT_PYTHON_DEDUP_BUFFER_DOCS)
    parser.add_argument("--python-download-workers", type=int, default=16)
    parser.add_argument("--python-download-timeout", type=int, default=15)
    parser.add_argument("--no-near-dedup", action="store_true")
    parser.add_argument("--no-fasttext", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare tiny-coder pretraining data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="write small cleaned local samples")
    sample.add_argument("--output-dir", default="data/samples")
    sample.add_argument("--python-docs", type=int, default=1000)
    sample.add_argument("--web-docs", type=int, default=1000)
    sample.add_argument("--math-docs", type=int, default=1000)
    sample.add_argument("--python-max-docs", type=int, default=None)
    sample.add_argument("--web-max-docs", type=int, default=None)
    sample.add_argument("--math-max-docs", type=int, default=None)
    sample.add_argument("--python-dedup-buffer-docs", type=int, default=DEFAULT_PYTHON_DEDUP_BUFFER_DOCS)
    sample.add_argument("--python-download-workers", type=int, default=16)
    sample.add_argument("--python-download-timeout", type=int, default=15)
    sample.add_argument("--progress-every", type=int, default=100)
    sample.add_argument("--no-near-dedup", action="store_true")
    sample.add_argument("--no-fasttext", action="store_true")
    sample.set_defaults(func=command_sample)

    audit = subparsers.add_parser("audit", help="audit accepted docs/chars/tokens")
    add_common_stream_args(audit)
    audit.add_argument("--python-target-tokens", type=int, default=0)
    audit.add_argument("--web-target-tokens", type=int, default=0)
    audit.add_argument("--math-target-tokens", type=int, default=0)
    audit.add_argument("--output", default="data/audit_report.json")
    audit.set_defaults(func=command_audit)

    train_tok = subparsers.add_parser("train-tokenizer", help="train byte-level BPE tokenizer candidates")
    train_tok.add_argument("--sample-dir", default="data/samples")
    train_tok.add_argument("--vocab-sizes", type=int, nargs="+", default=[16_000, 24_000, 32_000, 48_000])
    train_tok.add_argument("--output-dir", default="data/tokenizers")
    train_tok.add_argument("--max-docs-per-source", type=int, default=None)
    train_tok.set_defaults(func=command_train_tokenizer)

    tok_report = subparsers.add_parser("tokenizer-report", help="compare tokenizer compression")
    tok_report.add_argument("--sample-dir", default="data/samples")
    tok_report.add_argument("--tokenizers", nargs="+", required=True)
    tok_report.add_argument("--max-docs-per-source", type=int, default=2000)
    tok_report.add_argument("--output", default="data/tokenizer_report.json")
    tok_report.set_defaults(func=command_tokenizer_report)

    build = subparsers.add_parser("build", help="stream, clean, tokenize, and write binary shards")
    add_common_stream_args(build)
    build.add_argument("--tokenizer", required=True)
    build.add_argument("--output-dir", default="data/build")
    build.add_argument("--python-target-tokens", type=int, default=2_450_000_000)
    build.add_argument("--web-target-tokens", type=int, default=3_150_000_000)
    build.add_argument("--math-target-tokens", type=int, default=1_400_000_000)
    build.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    build.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    build.add_argument("--max-local-gb", type=float, default=40.0)
    build.add_argument("--progress-every-tokens", type=int, default=5_000_000)
    build.add_argument("--source-order", nargs="+", default=list(SOURCES))
    build.set_defaults(func=command_build)

    upload = subparsers.add_parser("upload", help="upload build artifacts to a Hugging Face dataset repo")
    upload.add_argument("--input-dir", default="data/build")
    upload.add_argument("--hf-repo", required=True)
    upload.add_argument("--private", action="store_true")
    upload.add_argument("--commit-message", default="Upload tiny-coder data artifacts")
    upload.set_defaults(func=command_upload)

    clean = subparsers.add_parser("clean-local", help="remove local build artifacts after upload")
    clean.add_argument("--input-dir", default="data/build")
    clean.add_argument("--remove-reports", action="store_true")
    clean.set_defaults(func=command_clean_local)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
