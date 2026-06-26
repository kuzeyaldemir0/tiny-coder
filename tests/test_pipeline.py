from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data import ShardedTokenDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PARENT = ROOT_DIR / "data"
if str(SCRIPT_PARENT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PARENT))

from scripts.dedup import near_dedup_records
from scripts.python_cleaning import clean_python_row, extract_identifier_text, parse_python3, strip_known_boilerplate
from scripts.tokenizers import load_tokenizer, train_one_tokenizer


class PythonCleaningTests(unittest.TestCase):
    def test_cjk_file_is_dropped(self) -> None:
        row = {"id": "cjk", "text": "# 这是中文注释\nprint('hello')\n"}
        cleaned, reason, _ = clean_python_row(row, require_fasttext=False)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "cjk")

    def test_python2_syntax_is_dropped(self) -> None:
        row = {"id": "py2", "text": "print 'hello'\n"}
        cleaned, reason, _ = clean_python_row(row, require_fasttext=False)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "not_python3")

    def test_boilerplate_is_stripped_but_useful_comment_remains(self) -> None:
        code = (
            "# -*- coding: utf-8 -*-\n"
            "# author: Example Person\n"
            "# This function explains the important algorithm.\n"
            "def calculate_total(invoice_amount):\n"
            "    return invoice_amount + 1\n"
        )
        cleaned, info = strip_known_boilerplate(code)
        self.assertTrue(info["changed"])
        self.assertNotIn("coding", cleaned)
        self.assertNotIn("author", cleaned.lower())
        self.assertIn("important algorithm", cleaned)

    def test_identifier_extraction_splits_user_names(self) -> None:
        tree = parse_python3(
            "class InvoiceCalculator:\n"
            "    def calculateGrandTotal(self, invoice_amount):\n"
            "        running_total = invoice_amount\n"
            "        return running_total\n"
        )
        identifiers = extract_identifier_text(tree)
        self.assertIn("invoice", identifiers)
        self.assertIn("calculator", identifiers)
        self.assertIn("grand", identifiers)
        self.assertIn("total", identifiers)

    def test_exact_and_near_dedup_keep_quality_representative(self) -> None:
        low = {
            "id": "low",
            "text": "def add_numbers(first_number, second_number):\n    return first_number + second_number\n",
            "lang_conf": 0.6,
            "mixed_chars": 10,
        }
        high = {
            "id": "high",
            "text": "# Add two numbers together.\ndef add_numbers(first_number, second_number):\n    return first_number + second_number\n",
            "lang_conf": 0.8,
            "mixed_chars": 40,
        }
        kept, decisions = near_dedup_records([low, high], threshold=0.60)
        self.assertEqual([row["id"] for row in kept], ["high"])
        self.assertEqual(decisions[0]["representative"], "high")


class BinaryDatasetTests(unittest.TestCase):
    def test_sharded_token_dataset_reads_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1, 2, 3], dtype=np.uint16).tofile(root / "train_00000.bin")
            np.asarray([4, 5, 6], dtype=np.uint16).tofile(root / "train_00001.bin")
            (root / "meta.json").write_text(json.dumps({"dtype": "uint16"}), encoding="utf-8")

            dataset = ShardedTokenDataset(root, "train", context_length=4)
            self.assertEqual(len(dataset), 1)
            x, y = dataset[0]
            self.assertEqual(x.tolist(), [1, 2, 3, 4])
            self.assertEqual(y.tolist(), [2, 3, 4, 5])


class TokenizerSmokeTests(unittest.TestCase):
    def test_train_tokenizer_writes_tokenizer_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tokenizer.json"
            texts = [
                "def add_numbers(first_number, second_number): return first_number + second_number",
                "Mathematics uses variables, equations, proofs, and examples.",
                "Educational text should include clear explanations and useful context.",
            ]
            train_one_tokenizer(256, texts, output)
            tokenizer = load_tokenizer(output)
            encoded = tokenizer.encode("def add_numbers(x, y): return x + y").ids
            self.assertGreater(len(encoded), 0)
            self.assertEqual(tokenizer.token_to_id("<eos>"), encoded[-1])


if __name__ == "__main__":
    unittest.main()
