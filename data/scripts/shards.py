from __future__ import annotations

from array import array
from pathlib import Path

import numpy as np


class BinShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        shard_tokens: int,
        dtype: str = "uint16",
        append: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.shard_tokens = shard_tokens
        self.dtype = dtype
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = array("H" if dtype == "uint16" else "I")
        self.shard_index = 0
        self.total_tokens = 0
        self.files: list[dict] = []
        if append:
            self._load_existing_shards()

    def _load_existing_shards(self) -> None:
        shards = sorted(self.output_dir.glob(f"{self.split}_*.bin"))
        for path in shards:
            tokens = path.stat().st_size // np.dtype(self.dtype).itemsize
            self.files.append({"path": path.name, "tokens": tokens, "bytes": path.stat().st_size})
            self.total_tokens += tokens
            try:
                index = int(path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            self.shard_index = max(self.shard_index, index + 1)

    def add(self, token_ids: list[int]) -> None:
        if self.dtype == "uint16" and token_ids and max(token_ids) > np.iinfo(np.uint16).max:
            raise ValueError("token id exceeds uint16 range; choose a vocab <= 65,535 or use uint32")
        self.buffer.extend(token_ids)
        self.total_tokens += len(token_ids)
        while len(self.buffer) >= self.shard_tokens:
            self.flush(self.shard_tokens)

    def flush(self, token_count: int | None = None) -> None:
        if not self.buffer:
            return
        if token_count is None:
            token_count = len(self.buffer)
        chunk = self.buffer[:token_count]
        del self.buffer[:token_count]
        path = self.output_dir / f"{self.split}_{self.shard_index:05d}.bin"
        np.asarray(chunk, dtype=self.dtype).tofile(path)
        self.files.append({"path": path.name, "tokens": len(chunk), "bytes": path.stat().st_size})
        self.shard_index += 1

    def close(self) -> None:
        self.flush()

    @property
    def buffered_bytes(self) -> int:
        return len(self.buffer) * np.dtype(self.dtype).itemsize

    @property
    def written_bytes(self) -> int:
        return sum(item["bytes"] for item in self.files)
