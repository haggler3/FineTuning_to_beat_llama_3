"""Integration test for the QLoRA training pipeline.

Heavy libraries are stubbed, but the dataset and tokenizer are real fakes with
correct semantics (see tests/fakes.py), so a misuse of the Dataset API or a
regression in label masking fails here instead of passing silently.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from fakes import FakeTokenizer, make_qa_dataset
from qlora_icl.train import main_function

logging.basicConfig(level=logging.DEBUG)


class TestQLoRAIntegration(unittest.TestCase):

    @patch('qlora_icl.train.HfApi')
    @patch('qlora_icl.train.AutoTokenizer.from_pretrained')
    @patch('qlora_icl.train.AutoModelForCausalLM.from_pretrained')
    @patch('qlora_icl.train.build_splits')
    @patch('qlora_icl.train.get_peft_model')
    @patch('qlora_icl.train.prepare_model_for_kbit_training')
    @patch('qlora_icl.train.LoraConfig')
    @patch('qlora_icl.train.Trainer')
    @patch('qlora_icl.train.hf_logging.set_verbosity_error')
    @patch('qlora_icl.train.trainer_utils.get_last_checkpoint')
    @patch('qlora_icl.train.torch.cuda.is_available')
    @patch('qlora_icl.train.torch.cuda.is_bf16_supported')
    @patch('qlora_icl.train.torch.cuda.get_device_properties')
    def test_end_to_end_training_pipeline(
        self, mock_props, mock_bf16, mock_cuda, mock_get_ckpt, mock_set_verbosity,
        mock_trainer_class, mock_lora_config, mock_prepare_kbit, mock_get_peft,
        mock_build_splits, mock_model_pretrained, mock_tokenizer_pretrained,
        mock_hfapi
    ):
        logging.info("[TEST] Starting end-to-end QLoRA integration test")

        # The training path asserts a GPU is present, since 4-bit has no CPU path.
        mock_cuda.return_value = True
        mock_bf16.return_value = True
        mock_props.return_value.total_memory = 16 * 10**9

        mock_build_splits.return_value = (
            make_qa_dataset(10, "train"),
            make_qa_dataset(3, "val"),
            make_qa_dataset(2, "test"),
        )

        mock_tokenizer_pretrained.return_value = FakeTokenizer()

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model_pretrained.return_value = mock_model
        mock_prepare_kbit.return_value = mock_model
        mock_get_peft.return_value = mock_model

        # LoRA needs at least one trainable parameter or main_function asserts.
        trainable = MagicMock()
        trainable.numel.return_value = 100
        trainable.requires_grad = True
        mock_model.parameters.return_value = [trainable]
        mock_model.named_modules.return_value = [("layer.q_proj", MagicMock())]

        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        mock_get_ckpt.return_value = None

        cli_args = [
            '--torch_dataset_url', 'fake/dataset',
            '--model_id', 'fake/model',
            '--project', 'testproject',
            '--user_id', 'testuser',
            '--hf_token', 'dummy_token',
            '--no_wandb',
        ]

        ret_code = main_function(cli_args)

        self.assertEqual(ret_code, 0)

        # Splits must be derived through the shared helper, so that Evaluate.py
        # given the same seed reproduces the same held-out set.
        mock_build_splits.assert_called_once()
        _, kwargs = mock_build_splits.call_args
        self.assertEqual(kwargs['seed'], 42)

        mock_model_pretrained.assert_called_once_with(
            'fake/model',
            quantization_config=unittest.mock.ANY,
            device_map='auto',
            trust_remote_code=True
        )
        mock_tokenizer_pretrained.assert_called_once_with('fake/model', trust_remote_code=True)
        mock_lora_config.assert_called_once()
        mock_prepare_kbit.assert_called_once_with(mock_model)
        mock_get_peft.assert_called_once_with(mock_model, unittest.mock.ANY)
        mock_trainer_class.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once()
        # create_repo is called on the HfApi instance, not the class.
        mock_hfapi.return_value.create_repo.assert_called_once_with(
            repo_id='testuser/model-testproject', exist_ok=True
        )

        logging.info("[TEST] End-to-end integration test passed")

    @patch('qlora_icl.train.AutoTokenizer.from_pretrained')
    @patch('qlora_icl.train.build_splits')
    @patch('qlora_icl.train.torch.cuda.is_available')
    def test_cpu_host_is_rejected(self, mock_cuda, mock_build_splits, mock_tok):
        """bitsandbytes has no usable CPU path, so this must fail clearly."""
        mock_cuda.return_value = False
        mock_build_splits.return_value = (
            make_qa_dataset(4), make_qa_dataset(2), make_qa_dataset(2)
        )
        mock_tok.return_value = FakeTokenizer()

        with self.assertRaises(RuntimeError) as ctx:
            main_function([
                '--torch_dataset_url', 'fake/dataset',
                '--hf_token', 'dummy_token', '--no_wandb',
            ])

        self.assertIn("CUDA", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
