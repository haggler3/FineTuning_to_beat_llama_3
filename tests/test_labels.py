"""Real (unmocked) tests for causal-LM label construction.

The rest of the suite mocks transformers wholesale, which means it passes
regardless of whether the loss is masked correctly. build_labels is pure, so it
can be tested for real.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Fine_Tune imports heavy libraries at module scope; stub them out. build_labels
# itself touches none of them, so the logic under test stays real.
for _mod in ['transformers', 'peft', 'huggingface_hub', 'qlora_icl.splits', 'torch']:
    sys.modules.setdefault(_mod, MagicMock())

from qlora_icl.train import build_labels  # noqa: E402


class TestBuildLabels(unittest.TestCase):

    def test_prompt_tokens_are_masked(self):
        """Tokens belonging to the prompt must not contribute to the loss."""
        input_ids = [10, 11, 12, 20, 21]
        attention = [1, 1, 1, 1, 1]
        labels = build_labels(input_ids, attention, n_prompt_tokens=3)

        self.assertEqual(labels[:3], [-100, -100, -100])
        self.assertEqual(labels[3:], [20, 21])

    def test_padding_is_masked(self):
        """Pad positions (attention 0) must not contribute to the loss.

        This is the defect that mattered most: pad_token == eos_token, so an
        unmasked run trains the model to emit end-of-sequence forever.
        """
        input_ids = [10, 20, 21, 0, 0]
        attention = [1, 1, 1, 0, 0]
        labels = build_labels(input_ids, attention, n_prompt_tokens=1)

        self.assertEqual(labels, [-100, 20, 21, -100, -100])

    def test_no_prompt_scores_whole_sequence(self):
        """Raw-text rows have no prompt boundary, so everything is scored."""
        input_ids = [5, 6, 7]
        attention = [1, 1, 1]
        labels = build_labels(input_ids, attention, n_prompt_tokens=0)

        self.assertEqual(labels, [5, 6, 7])

    def test_fully_masked_row_falls_back(self):
        """An all -100 row makes the loss NaN, so it must fall back."""
        input_ids = [10, 11, 12]
        attention = [1, 1, 1]
        labels = build_labels(input_ids, attention, n_prompt_tokens=3)

        self.assertEqual(labels, [10, 11, 12])
        self.assertNotIn(-100, labels)

    def test_labels_align_with_input_length(self):
        input_ids = list(range(512))
        attention = [1] * 400 + [0] * 112
        labels = build_labels(input_ids, attention, n_prompt_tokens=50)

        self.assertEqual(len(labels), len(input_ids))
        self.assertEqual(labels.count(-100), 50 + 112)


if __name__ == '__main__':
    unittest.main()
