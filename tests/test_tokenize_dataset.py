"""Integration test for the standalone dataset tokenizer.

Uses a real fake Dataset so the split chain (shuffle -> select -> split -> split)
is exercised with correct semantics instead of being swallowed by a MagicMock.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from fakes import FakeTokenizer, make_qa_dataset
from qlora_icl.tokenize_dataset import main_function


class TestEndToEndIntegration(unittest.TestCase):

    def _patched(self, mock_load_dataset, mock_tokenizer_class, mock_sbert_class):
        mock_load_dataset.return_value = make_qa_dataset(4000)
        mock_tokenizer_class.from_pretrained.return_value = FakeTokenizer()

        encoder = MagicMock()
        encoder.encode.return_value = MagicMock()
        mock_sbert_class.return_value = encoder
        return encoder

    @patch('qlora_icl.tokenize_dataset.torch')
    @patch('qlora_icl.tokenize_dataset.SentenceTransformer')
    @patch('qlora_icl.tokenize_dataset.AutoTokenizer')
    @patch('qlora_icl.tokenize_dataset.load_dataset')
    @patch.dict(os.environ, {"HUGGINGFACE_HUB_TOKEN": "dummy_token"})
    def test_end_to_end_standard_pipeline(
        self, mock_load_dataset, mock_tokenizer_class, mock_sbert_class, mock_torch
    ):
        self._patched(mock_load_dataset, mock_tokenizer_class, mock_sbert_class)
        mock_torch.load.return_value = {
            'embeddings': MagicMock(),
            'patient_inputs': ['q'],
            'doctor_outputs': ['a'],
        }

        result = main_function(cli_args=[])

        for key in ("tokenized_train_dataset", "tokenized_val_dataset",
                    "tokenized_test_dataset", "train_dataset",
                    "val_dataset", "test_dataset"):
            self.assertIn(key, result)

        # 4000 rows -> 80/10/10
        self.assertEqual(len(result["train_dataset"]), 3200)
        self.assertEqual(len(result["val_dataset"]), 400)
        self.assertEqual(len(result["test_dataset"]), 400)

    @patch('qlora_icl.tokenize_dataset.torch')
    @patch('qlora_icl.tokenize_dataset.SentenceTransformer')
    @patch('qlora_icl.tokenize_dataset.AutoTokenizer')
    @patch('qlora_icl.tokenize_dataset.load_dataset')
    @patch.dict(os.environ, {"HUGGINGFACE_HUB_TOKEN": "dummy_token"})
    def test_tokenizer_matches_the_target_model(
        self, mock_load_dataset, mock_tokenizer_class, mock_sbert_class, mock_torch
    ):
        """The tokenizer must be the target model's, not a hardcoded BERT one.

        This previously loaded bert-base-uncased, a masked-LM tokenizer whose
        ids mean nothing to the causal LM they were fed to.
        """
        self._patched(mock_load_dataset, mock_tokenizer_class, mock_sbert_class)
        mock_torch.load.return_value = {
            'embeddings': MagicMock(),
            'patient_inputs': ['q'],
            'doctor_outputs': ['a'],
        }

        main_function(cli_args=['--model_id', 'EleutherAI/pythia-1b'])

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            'EleutherAI/pythia-1b', trust_remote_code=True
        )


if __name__ == "__main__":
    unittest.main()
