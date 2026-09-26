"""Real (unmocked) tests for the ICL retrieval leakage guard.

The original Evaluate.py built its retrieval corpus out of the test split, so
every query retrieved itself and its own gold answer was pasted into the prompt.
These tests pin the guard that makes that configuration impossible.
"""

import sys
import unittest
from unittest.mock import MagicMock

for _mod in ['torch', 'transformers', 'evaluate', 'wandb', 'tqdm',
             'numpy', 'qlora_icl.splits', 'peft', 'sentence_transformers']:
    sys.modules.setdefault(_mod, MagicMock())

from qlora_icl.evaluation import assert_no_leakage  # noqa: E402


class TestLeakageGuard(unittest.TestCase):

    def test_disjoint_corpus_passes(self):
        """Retrieving from train while scoring test is the correct setup."""
        train = ["train question a", "train question b"]
        test = ["test question a", "test question b"]

        assert_no_leakage(train, test)  # must not raise

    def test_identical_corpus_raises(self):
        """The original bug: corpus and eval set are the same rows."""
        rows = ["question a", "question b"]

        with self.assertRaises(ValueError) as ctx:
            assert_no_leakage(rows, rows)

        self.assertIn("overlaps", str(ctx.exception))

    def test_partial_overlap_raises(self):
        """Even one shared row can leak a gold answer into a prompt."""
        train = ["shared question", "train only"]
        test = ["shared question", "test only"]

        with self.assertRaises(ValueError) as ctx:
            assert_no_leakage(train, test)

        self.assertIn("1 example", str(ctx.exception))

    def test_empty_corpus_passes(self):
        assert_no_leakage([], ["test question"])


if __name__ == '__main__':
    unittest.main()
