"""Integration test for the evaluation pipeline.

The dataset is a real fake with correct HuggingFace Dataset semantics, so the
batching bug the original version shipped (slicing a Dataset yields a dict of
columns, not a list of examples) would fail here rather than pass.
"""

import json
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fakes import FakeTokenizer, make_qa_dataset
from qlora_icl.evaluation import main_function

logging.basicConfig(level=logging.DEBUG)


class TestEvaluationIntegration(unittest.TestCase):

    def _common_mocks(self, mock_tokenizer_class, mock_pipeline, mock_load_metric):
        mock_tokenizer_class.return_value = FakeTokenizer()

        # A batched text-generation pipeline returns one list of candidates per
        # prompt. Returning bare dicts, as the old test did, hid a real bug.
        def fake_generate(prompts, **kwargs):
            return [[{"generated_text": f"generated for {p[:20]}"}] for p in prompts]

        mock_pipeline.return_value = fake_generate

        mock_load_metric.return_value.compute = MagicMock(return_value={
            "precision": [0.9, 0.85],
            "recall": [0.8, 0.82],
            "f1": [0.85, 0.83],
        })

    @patch('qlora_icl.evaluation.build_splits')
    @patch('qlora_icl.evaluation.AutoTokenizer.from_pretrained')
    @patch('qlora_icl.evaluation.AutoModelForCausalLM.from_pretrained')
    @patch('qlora_icl.evaluation.pipeline')
    @patch('qlora_icl.evaluation.load_metric')
    @patch('qlora_icl.evaluation.torch.cuda.is_available')
    def test_end_to_end_without_icl(
        self, mock_cuda, mock_load_metric, mock_pipeline,
        mock_model_class, mock_tokenizer_class, mock_build_splits
    ):
        mock_cuda.return_value = False
        mock_build_splits.return_value = (
            make_qa_dataset(6, "train"),
            make_qa_dataset(2, "val"),
            make_qa_dataset(2, "test"),
        )
        self._common_mocks(mock_tokenizer_class, mock_pipeline, mock_load_metric)
        mock_model_class.return_value = MagicMock()

        ret = main_function([
            '--test_dataset', 'fake/dataset',
            '--model_id', 'fake/model',
            '--batch_size', '2',
            '--no_wandb',
            '--hf_token', 'dummy_token',
        ])

        self.assertEqual(ret, 0)
        mock_build_splits.assert_called_once()
        mock_load_metric.assert_called_once_with('bertscore')

        # The generation pipeline must not echo the prompt back, or BERTScore
        # compares prompt+answer against answer and reports inflated scores.
        _, pipe_kwargs = mock_pipeline.call_args
        self.assertFalse(pipe_kwargs['return_full_text'])
        self.assertFalse(pipe_kwargs['do_sample'])

    @patch('qlora_icl.evaluation.build_splits')
    @patch('qlora_icl.evaluation.AutoTokenizer.from_pretrained')
    @patch('qlora_icl.evaluation.AutoModelForCausalLM.from_pretrained')
    @patch('qlora_icl.evaluation.pipeline')
    @patch('qlora_icl.evaluation.load_metric')
    @patch('qlora_icl.evaluation.torch.cuda.is_available')
    def test_icl_retrieves_from_train_not_test(
        self, mock_cuda, mock_load_metric, mock_pipeline,
        mock_model_class, mock_tokenizer_class, mock_build_splits
    ):
        """ICL demos must come from train, never from the split being scored."""
        mock_cuda.return_value = False
        train = make_qa_dataset(6, "train")
        test = make_qa_dataset(2, "test")
        mock_build_splits.return_value = (train, make_qa_dataset(2, "val"), test)
        self._common_mocks(mock_tokenizer_class, mock_pipeline, mock_load_metric)
        mock_model_class.return_value = MagicMock()

        import sys
        st = MagicMock()
        encoder = MagicMock()
        encoder.encode.return_value = MagicMock()
        st.SentenceTransformer.return_value = encoder

        # cos_sim returns one similarity row per query over the train corpus.
        class Row:
            def __getitem__(self, idx):
                return MagicMock(item=lambda: 0.9)
        st.util.cos_sim.return_value = [Row(), Row()]

        with patch.dict(sys.modules, {'sentence_transformers': st}):
            with patch('qlora_icl.evaluation.torch.topk') as mock_topk:
                mock_topk.return_value.indices.tolist.return_value = [0, 1]
                ret = main_function([
                    '--test_dataset', 'fake/dataset',
                    '--model_id', 'fake/model',
                    '--batch_size', '2',
                    '--use_icl',
                    '--icl_top_k', '2',
                    '--no_wandb',
                    '--hf_token', 'dummy_token',
                ])

        self.assertEqual(ret, 0)
        # The corpus encoded for retrieval must be the train inputs.
        encoded = encoder.encode.call_args_list[0][0][0]
        self.assertEqual(encoded, train['input'])
        for item in encoded:
            self.assertNotIn(item, test['input'])

    @patch('qlora_icl.evaluation.build_splits')
    @patch('qlora_icl.evaluation.AutoTokenizer.from_pretrained')
    @patch('qlora_icl.evaluation.AutoModelForCausalLM.from_pretrained')
    @patch('qlora_icl.evaluation.pipeline')
    @patch('qlora_icl.evaluation.load_metric')
    @patch('qlora_icl.evaluation.torch.cuda.is_available')
    def test_adapter_is_loaded_when_requested(
        self, mock_cuda, mock_load_metric, mock_pipeline,
        mock_model_class, mock_tokenizer_class, mock_build_splits
    ):
        """The original version ignored adapters and silently scored the base model."""
        mock_cuda.return_value = False
        mock_build_splits.return_value = (
            make_qa_dataset(4, "train"), make_qa_dataset(2, "val"), make_qa_dataset(2, "test")
        )
        self._common_mocks(mock_tokenizer_class, mock_pipeline, mock_load_metric)
        base_model = MagicMock()
        mock_model_class.return_value = base_model

        import sys
        peft = MagicMock()
        with patch.dict(sys.modules, {'peft': peft}):
            ret = main_function([
                '--test_dataset', 'fake/dataset',
                '--model_id', 'fake/model',
                '--adapter_id', 'testuser/some-adapter',
                '--batch_size', '2',
                '--no_wandb',
                '--hf_token', 'dummy_token',
            ])

        self.assertEqual(ret, 0)
        peft.PeftModel.from_pretrained.assert_called_once()
        args, _ = peft.PeftModel.from_pretrained.call_args
        self.assertIs(args[0], base_model)
        self.assertEqual(args[1], 'testuser/some-adapter')

    @patch('qlora_icl.evaluation.build_splits')
    @patch('qlora_icl.evaluation.AutoTokenizer.from_pretrained')
    @patch('qlora_icl.evaluation.AutoModelForCausalLM.from_pretrained')
    @patch('qlora_icl.evaluation.pipeline')
    @patch('qlora_icl.evaluation.load_metric')
    @patch('qlora_icl.evaluation.torch.cuda.is_available')
    def test_adaptive_k_spends_a_different_budget_per_query(
        self, mock_cuda, mock_load_metric, mock_pipeline,
        mock_model_class, mock_tokenizer_class, mock_build_splits
    ):
        """A query with one clearly-best match should spend fewer demonstrations
        than a query with several equally good matches - and --output_json
        should report the resulting average, not just the ceiling.
        """
        mock_cuda.return_value = False
        train = make_qa_dataset(6, "train")
        test = make_qa_dataset(2, "test")
        mock_build_splits.return_value = (train, make_qa_dataset(2, "val"), test)
        self._common_mocks(mock_tokenizer_class, mock_pipeline, mock_load_metric)
        mock_model_class.return_value = MagicMock()

        import sys
        st = MagicMock()
        encoder = MagicMock()
        encoder.encode.return_value = MagicMock()
        st.SentenceTransformer.return_value = encoder

        # Query 0: three clustered scores -> adaptive keeps all three.
        # Query 1: one clear best match, then a cliff -> adaptive keeps one.
        clustered = {0: 0.95, 1: 0.94, 2: 0.93}
        cliff = {0: 0.95, 1: 0.30, 2: 0.20}

        def make_row(score_by_idx):
            class Row:
                def __getitem__(self, idx):
                    return MagicMock(item=lambda: score_by_idx[idx])
            return Row()

        st.util.cos_sim.return_value = [make_row(clustered), make_row(cliff)]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "results.json")
            with patch.dict(sys.modules, {'sentence_transformers': st}):
                with patch('qlora_icl.evaluation.torch.topk') as mock_topk:
                    mock_topk.return_value.indices.tolist.return_value = [0, 1, 2]
                    ret = main_function([
                        '--test_dataset', 'fake/dataset',
                        '--model_id', 'fake/model',
                        '--batch_size', '16',
                        '--use_icl',
                        '--icl_top_k', '3',
                        '--adaptive_k',
                        '--adaptive_min_k', '1',
                        '--adaptive_relative_drop', '0.85',
                        '--no_wandb',
                        '--hf_token', 'dummy_token',
                        '--output_json', out_path,
                    ])

            self.assertEqual(ret, 0)
            with open(out_path, encoding='utf-8') as fh:
                payload = json.load(fh)

        self.assertTrue(payload['adaptive_k'])
        self.assertEqual(payload['icl_top_k'], 3)
        # (3 + 1) / 2 queries = 2.0 - strictly less than the ceiling of 3,
        # which is the entire point: not every query pays full price.
        self.assertEqual(payload['mean_demonstrations_per_query'], 2.0)
        self.assertLess(payload['mean_demonstrations_per_query'], payload['icl_top_k'])


if __name__ == "__main__":
    unittest.main()
