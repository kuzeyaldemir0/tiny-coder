from __future__ import annotations

import hashlib
import io
import re
import tokenize
from collections import defaultdict

from .constants import MINHASH_NUM_PERM, NEAR_DEDUP_THRESHOLD, SHINGLE_SIZE


def normalize_code_for_exact_dedup(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in code.split("\n")]
    return "\n".join(lines).strip()


def exact_dedup_hash(code: str) -> str:
    normalized = normalize_code_for_exact_dedup(code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedup_quality_key(rec: dict) -> tuple[float, int, int]:
    return rec.get("lang_conf", 0.0), rec.get("mixed_chars", 0), len(rec.get("text", ""))


def normalize_token_text(text: str) -> str:
    return " ".join(text.lower().split())


def token_fingerprint_for_near_dedup(code: str) -> list[str]:
    tokens: list[str] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in stream:
            if tok.type in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }:
                continue
            if tok.type == tokenize.COMMENT:
                text = normalize_token_text(tok.string.lstrip("#"))
                if text:
                    tokens.extend(["<COMMENT>"] + text.split())
            elif tok.type == tokenize.STRING:
                text = normalize_token_text(tok.string)
                if text:
                    tokens.extend(["<STRING>"] + text.split())
            elif tok.type == tokenize.NUMBER:
                tokens.append("<NUM>")
            elif tok.type == tokenize.NAME:
                tokens.append(tok.string.lower())
            elif tok.string.strip():
                tokens.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        tokens = re.findall(r"\w+|[^\w\s]", code.lower())
    return tokens


def token_shingles(tokens: list[str], size: int = SHINGLE_SIZE) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) <= size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def make_minhash(shingles: set[str]):
    from datasketch import MinHash

    mh = MinHash(num_perm=MINHASH_NUM_PERM)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh


def near_dedup_records(records: list[dict], threshold: float = NEAR_DEDUP_THRESHOLD) -> tuple[list[dict], list[dict]]:
    if len(records) < 2:
        return records, []

    from datasketch import MinHashLSH

    minhashes = []
    for rec in records:
        rec_shingles = token_shingles(token_fingerprint_for_near_dedup(rec["text"]))
        minhashes.append(make_minhash(rec_shingles))

    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
    for i, mh in enumerate(minhashes):
        lsh.insert(str(i), mh)

    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, mh in enumerate(minhashes):
        for candidate in lsh.query(mh):
            j = int(candidate)
            if j > i:
                union(i, j)

    cluster_map: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        cluster_map[find(i)].append(i)

    drop_indices: set[int] = set()
    decisions: list[dict] = []
    for cluster in cluster_map.values():
        if len(cluster) <= 1:
            continue
        representative_idx = max(cluster, key=lambda idx: dedup_quality_key(records[idx]))
        dropped = [idx for idx in cluster if idx != representative_idx]
        drop_indices.update(dropped)
        decisions.append(
            {
                "representative": records[representative_idx].get("id"),
                "dropped": [records[idx].get("id") for idx in dropped],
                "size": len(cluster),
            }
        )

    kept = [rec for idx, rec in enumerate(records) if idx not in drop_indices]
    return kept, decisions
