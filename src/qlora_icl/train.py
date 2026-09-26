import argparse
import gc
import os
from datetime import datetime

import torch
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    trainer_utils,
)
from transformers import logging as hf_logging

from qlora_icl.splits import DEFAULT_MAX_SAMPLES, DEFAULT_SEED, build_splits, split_fingerprint

MAX_SEQ_LENGTH = 512


def build_labels(input_ids, attention_mask, n_prompt_tokens):
    """Build causal-LM labels that score only the completion.

    -100 marks a position as ignored by the loss. Both padding and prompt
    tokens are masked: pad_token is eos_token and every row is padded out to
    max_length, so without this the model is trained to reproduce the prompt
    and then emit padding.
    """
    labels = [
        -100 if (not attends or idx < n_prompt_tokens) else token_id
        for idx, (token_id, attends) in enumerate(zip(input_ids, attention_mask, strict=True))
    ]

    # A row truncated hard enough to leave no completion would be all -100,
    # which makes the loss NaN. Fall back to scoring the whole sequence.
    if all(label == -100 for label in labels):
        return list(input_ids)

    return labels


def main_function(cli_args):
    # ------------------------------
    # Argument Parsing and Setup
    # ------------------------------
    parser = argparse.ArgumentParser(description='Train a model via QLoRA with specified options.')

    parser.add_argument(
        '--torch_dataset_url',
        type=str,
        default="lavita/ChatDoctor-HealthCareMagic-100k",
        help="HuggingFace Hub or local directory identifier for the dataset to load."
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default="EleutherAI/pythia-1b",
        help="Model name or path (e.g., EleutherAI/pythia-1b)"
    )
    parser.add_argument(
        '--project',
        type=str,
        default="qlora-run",
        help="Project name suffix for experiment tracking and output folders."
    )
    parser.add_argument(
        '--user_id',
        type=str,
        default="YOUR_HF_USER",  # <-- CHANGE THIS TO YOUR HF USERNAME
        help="Your Hugging Face username for uploading/checkpointing."
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help="(Optional) HF Hub token. If not set, reads from HUGGINGFACE_HUB_TOKEN env variable."
    )
    parser.add_argument(
        '--lora_target_modules',
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules (e.g., 'q_proj,v_proj' for Mistral or 'q_proj,k_proj,v_proj' for Pythia)"
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help="Disable Weights & Biases logging (enabled by default)"
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help="Number of optimizer steps to train for."
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2.5e-5,
        help="Learning rate."
    )
    parser.add_argument(
        '--no_push',
        action='store_true',
        help="Skip creating and pushing to a Hugging Face Hub repo."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help="Seed controlling the train/val/test partition. Evaluate.py must be given the same value."
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Downsample the raw dataset to this many rows before splitting. Evaluate.py must match."
    )

    args = parser.parse_args(cli_args)

    HF_TOKEN = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if HF_TOKEN is None:
        raise ValueError("No Hugging Face token provided for dataset or pushing to hub.")

    dataset_path = args.torch_dataset_url
    model_id = args.model_id
    project = args.project
    user_id = args.user_id
    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    # Handle wandb flag (default: enabled, can be disabled with --no_wandb)
    use_wandb = not args.no_wandb
    wandb_report = "wandb" if use_wandb else "none"

    # ------------------------------
    # Load and Prepare Datasets
    # ------------------------------

    # These can be customized if you want to use different splits/datasets
    print("\n[STEP 1/6] Downloading & loading datasets...")
    print(f"[INFO] Dataset: {dataset_path}")

    # Splits come from splits.build_splits so that Evaluate.py, given the same
    # --seed and --max_samples, scores on exactly the held-out rows.
    try:
        masked_train, masked_val, masked_test = build_splits(
            dataset_path, token=HF_TOKEN, seed=args.seed, max_samples=args.max_samples
        )
        print("[SUCCESS] Datasets created:")
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_path}': {e}") from e

    split_sizes = [len(masked_train), len(masked_val), len(masked_test)]
    fingerprint = split_fingerprint(dataset_path, args.seed, args.max_samples, split_sizes)
    print(f"  - Training samples: {len(masked_train)}")
    print(f"  - Validation samples: {len(masked_val)}")
    print(f"  - Test samples: {len(masked_test)}")
    print(f"  - Split fingerprint: {fingerprint} (pass --seed {args.seed} --max_samples {args.max_samples} to Evaluate.py)")

    # Tokenize datasets
    print("\n[TOKENIZING] Preparing tokenizer...")
    print(f"[INFO] Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[SUCCESS] Tokenizer loaded. Vocab size: {len(tokenizer)}")
    def split_prompt_completion(example):
        """Return (prompt, completion) for an example.

        Instruction-style rows have a natural boundary: everything up to the
        answer is context the model is given, not something it should be scored
        on. Raw-text rows have no such boundary, so the whole row is completion.
        """
        if 'instruction' in example and 'output' in example:
            instr = example.get('instruction', '') or ''
            inp = example.get('input', '') or ''
            output = example.get('output', '') or ''
            return f"{instr}\n{inp}\n", str(output)
        if 'text' in example and example['text']:
            return "", str(example['text'])
        if 'content' in example:
            return "", str(example['content'])
        # Fallback: concatenate all string fields.
        return "", " ".join(str(v) for k, v in example.items() if isinstance(v, str) and v)

    def tokenize_function(example):
        """Tokenize one example, masking the loss to the completion only."""
        prompt, completion = split_prompt_completion(example)

        tokenized = tokenizer(
            prompt + completion,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None,
            return_attention_mask=True
        )

        # How many tokens the prompt occupies, so they can be left out of the loss.
        n_prompt_tokens = len(
            tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
        ) if prompt else 0

        tokenized["labels"] = build_labels(
            tokenized["input_ids"], tokenized["attention_mask"], n_prompt_tokens
        )
        return tokenized

    print("[TOKENIZING] Processing training set...")
    masked_train = masked_train.map(tokenize_function, remove_columns=masked_train.column_names)
    print("[TOKENIZING] Processing validation set...")
    masked_val = masked_val.map(tokenize_function, remove_columns=masked_val.column_names)
    print("[TOKENIZING] Processing test set...")
    masked_test = masked_test.map(tokenize_function, remove_columns=masked_test.column_names)
    print("[SUCCESS] Tokenization complete")

    # ------------------------------
    # Model Setup and Quantization
    # ------------------------------
    print("\n[STEP 2/6] Setting up model and quantization...")
    hf_logging.set_verbosity_error()
    gc.collect()
    device = 0 if torch.cuda.is_available() else -1
    device_name = 'CUDA' if device == 0 else 'CPU'
    print(f"[INFO] Device: {device_name}")
    if device == 0:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if device != 0:
        raise RuntimeError(
            "4-bit quantization requires a CUDA GPU; bitsandbytes has no usable CPU path. "
            "Run this on a GPU host (see infra/terraform for a spot T4)."
        )

    # bf16 needs Ampere or newer. Older CUDA cards fall back to fp16.
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[INFO] Compute dtype: {compute_dtype}")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[INFO] Model: {model_id}")

    print(f"[INFO] Loading quantized model '{model_id}' with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    print("[SUCCESS] Model loaded")

    print("[INFO] Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print("[SUCCESS] Model prepared for k-bit training")

    # ------------------------------
    # LoRA Configuration and Application
    # ------------------------------
    print("\n[STEP 3/6] Applying LoRA adapters...")

    # Auto-detect target modules if default ones not found
    def find_target_modules(model, preferred_modules):
        """Find available target modules in the model"""
        # First try the preferred modules
        module_names = set()
        for name, _module in model.named_modules():
            module_names.add(name.split(".")[-1])

        # Check if preferred modules exist
        found_modules = [m for m in preferred_modules if m in module_names]
        if found_modules:
            return found_modules

        # Architecture-specific attention projections, most specific first.
        # GPTNeoX (Pythia) fuses q/k/v into a single query_key_value matrix, so
        # without it LoRA silently attaches to "dense" instead, which is the
        # MLP/output projection rather than attention.
        for family in (
            ["q_proj", "v_proj"],              # Llama / Mistral
            ["query_key_value"],               # GPTNeoX / Pythia / Falcon
            ["c_attn"],                        # GPT-2
            ["qkv_proj"],                      # Phi-3
        ):
            found_modules = [m for m in family if m in module_names]
            if found_modules:
                print(f"[WARNING] Preferred modules not found. Using fallback: {found_modules}")
                return found_modules

        raise ValueError(
            "Could not find a known attention projection module in this model. "
            "Pass --lora_target_modules explicitly. Candidate leaf modules: "
            + str(sorted(m for m in module_names
                         if "proj" in m or "attn" in m or "dense" in m)[:20])
        )

    target_modules = find_target_modules(model, lora_target_modules)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(f"[INFO] LoRA config: r=8, alpha=32, target_modules={target_modules}")
    model = get_peft_model(model, lora_config)
    print("[SUCCESS] LoRA adapters applied")

    # Assert trainable params > 0
    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[METRICS] Total parameters: {total_params:,}")
    print(f"[METRICS] Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    assert trainable_params > 0, "No trainable parameters found with LoRA applied!"

    # ------------------------------
    # Experiment Output & Tracking Setup
    # ------------------------------
    print("\n[STEP 4/6] Setting up experiment tracking...")
    base_model_name = model_id.split("/")[-1]
    run_name = f"{base_model_name}-{project}"
    output_dir = os.path.join(".", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Run name: {run_name}")
    print(f"[INFO] Output directory: {output_dir}")

    # ------------------------------
    # Training Arguments and Trainer
    # ------------------------------
    print("[STEP 5/6] Initializing Trainer...")
    print("[INFO] Training configuration:")
    print(f"  - Max steps: {args.max_steps}")
    print(f"  - Learning rate: {args.learning_rate}")
    print("  - Batch size: 2 (per device)")
    print("  - Eval/Save every: 50 steps")
    print(f"  - Weights & Biases: {'[OK] Enabled' if use_wandb else '[OFF] Disabled'}")
    trainer = Trainer(
        model=model,
        train_dataset=masked_train,
        eval_dataset=masked_val,
        args=TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=50,
            bf16=use_bf16,
            fp16=not use_bf16,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=50,
            eval_strategy="steps",
            eval_steps=50,
            do_eval=True,
            report_to=wandb_report,
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence warnings; enable for inference

    # ------------------------------
    # Training and Checkpointing
    # ------------------------------
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    last_ckpt = trainer_utils.get_last_checkpoint(output_dir)
    if last_ckpt is not None:
        print(f"[INFO] Resuming training from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("[INFO] Starting fresh training")
        print(f"[INFO] Checkpoints will be saved to: {output_dir}")
        trainer.train()
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)

    # ------------------------------
    # Saving Model and Tokenizer
    # ------------------------------
    print("[INFO] Saving trained model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ------------------------------
    # Uploading to Hugging Face Hub
    # ------------------------------
    if args.no_push:
        print("\n[STEP 6/6] Skipping Hub upload (--no_push)")
        print("\n" + "="*60)
        print("[OK] FINE-TUNING PIPELINE COMPLETE")
        print("="*60)
        print(f"Model saved locally at: {output_dir}")
        return 0

    print("\n[STEP 6/6] Uploading model to Hugging Face Hub...")
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    api = HfApi()
    repo_id = f"{user_id}/{run_name}"
    print(f"[INFO] Creating repo: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        print("[SUCCESS] Repository ready")
    except Exception as e:
        print(f"[WARNING] Could not create repo: {e}")

    print("[INFO] Pushing model to hub...")
    try:
        trainer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"QLoRA fine-tuned on {dataset_name}"
        )
        print(f"[SUCCESS] Model pushed to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[WARNING] Could not push to hub: {e}")

    print("\n" + "="*60)
    print("[OK] FINE-TUNING PIPELINE COMPLETE")
    print("="*60)
    print(f"Model saved locally at: {output_dir}")
    print(f"Model on Hub: https://huggingface.co/{repo_id}")
    return 0

def main(argv=None):
    """Console-script entry point."""
    import sys
    return main_function(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
