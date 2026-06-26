from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_meta(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"missing metadata file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_shards(data_dir: str | Path, split: str) -> list[Path]:
    data_dir = Path(data_dir)
    shards = sorted(data_dir.glob(f"{split}_*.bin"))
    if not shards:
        raise FileNotFoundError(f"no {split!r} shards found in {data_dir}")
    return shards


class ShardedTokenDataset(Dataset):
    """Read fixed-length language-model chunks from uint16/uint32 .bin shards."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        context_length: int,
        dtype: str | np.dtype | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.context_length = context_length
        meta = load_meta(self.data_dir)
        self.dtype = np.dtype(dtype or meta.get("dtype", "uint16"))
        self.shards = find_shards(self.data_dir, split)
        self.arrays = [np.memmap(path, dtype=self.dtype, mode="r") for path in self.shards]

        lengths = [len(arr) for arr in self.arrays]
        self.cumulative_lengths: list[int] = []
        total = 0
        for length in lengths:
            total += length
            self.cumulative_lengths.append(total)

        self.total_tokens = total
        self.num_chunks = max(0, (self.total_tokens - 1) // self.context_length)

    def __len__(self) -> int:
        return self.num_chunks

    def _slice_tokens(self, start: int, length: int) -> np.ndarray:
        remaining = length
        offset = start
        chunks = []

        while remaining > 0:
            shard_index = bisect.bisect_right(self.cumulative_lengths, offset)
            shard_start = 0 if shard_index == 0 else self.cumulative_lengths[shard_index - 1]
            local_offset = offset - shard_start
            shard = self.arrays[shard_index]
            take = min(remaining, len(shard) - local_offset)
            chunks.append(np.asarray(shard[local_offset : local_offset + take], dtype=np.int64))
            offset += take
            remaining -= take

        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.context_length
        tokens = self._slice_tokens(start, self.context_length + 1)
        chunk = torch.from_numpy(tokens.astype(np.int64, copy=False))
        return chunk[:-1], chunk[1:]


def create_token_dataset(data_dir: str | Path, split: str, context_length: int) -> ShardedTokenDataset:
    return ShardedTokenDataset(data_dir=data_dir, split=split, context_length=context_length)
