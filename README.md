# tiny-coder

Small local language-model experiments for Python/code-focused training.

## Current State

- Data pipeline is script-based: `prepare_data.py` plus `data/scripts/`.
- Active training notebook: `tiny_coder_notebook.ipynb`.
- Tokenizer: custom byte-level BPE, current candidate `32k`.
- Local completed test dataset: `data/build-1b`.
- `data/build-1b` contains about `1.0B` tokens:
  - FineWeb-Edu-Dedup: `450M`
  - FineMath-4+: `200M`
  - Python-Edu: `350M`

Generated data is intentionally ignored by git.

## Next Stage

Run small, iterative experiments on the `1B` token build:

- decide model size and architecture;
- test context length;
- tune batch size, learning rate, warmup, and decay;
- compare short training runs before scaling;
- keep experiments small, measured, and reproducible.
