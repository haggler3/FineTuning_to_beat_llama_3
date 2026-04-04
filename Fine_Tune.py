import argparse
import gc
from datetime import datetime
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    logging as hf_logging,
    trainer_utils
)
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from huggingface_hub import HfApi

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

    args = parser.parse_args(cli_args)

    HF_TOKEN = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if HF_TOKEN is None:
        raise ValueError("No Hugging Face token provided for dataset or pushing to hub.")

    dataset_path = args.torch_dataset_url
    model_id = args.model_id
    project = args.project
    user_id = args.user_id

    # ------------------------------
    # Load and Prepare Datasets
    # ------------------------------

    # These can be customized if you want to use different splits/datasets
    print("\\n[STEP 1/6] Downloading & loading datasets...")
    print(f"[INFO] Dataset: {dataset_path}")
    
    # Try to load existing splits, or create them if only 'train' exists
    try:
        # Load the full dataset
        full_dataset = load_dataset(dataset_path, token=HF_TOKEN, split="train")
        print(f"[INFO] Loaded raw dataset with {len(full_dataset)} samples")
        
        # Downsample to 4000 for faster training
        if len(full_dataset) > 4000:
            print(f"[INFO] Downsampling to 4,000 samples...")
            downsampled = full_dataset.shuffle(seed=42).select(range(4000))
        else:
            downsampled = full_dataset
        
        # Create train/val/test splits (80/10/10)
        print(f"[INFO] Creating train/val/test splits...")
        train_temp_split = downsampled.train_test_split(test_size=0.2, seed=42)
        masked_train = train_temp_split['train']      # 80%
        temp = train_temp_split['test']                # 20%
        
        val_test_split = temp.train_test_split(test_size=0.5, seed=42)
        masked_val = val_test_split['train']           # 10% (50% of 20%)
        masked_test = val_test_split['test']           # 10% (50% of 20%)
        
        print(f"[SUCCESS] Datasets created:")
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_path}': {e}")
    
    print(f"  - Training samples: {len(masked_train)}")
    print(f"  - Validation samples: {len(masked_val)}")
    print(f"  - Test samples: {len(masked_test)}")

    # Tokenize datasets
    print("\n[TOKENIZING] Preparing tokenizer...")
    print(f"[INFO] Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[SUCCESS] Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    def tokenize_function(examples):
        """Tokenize text examples for language modeling"""
        return tokenizer(
            examples.get('text') or examples.get('instruction', [''])[0],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
    
    print("[TOKENIZING] Processing training set...")
    masked_train = masked_train.map(tokenize_function, batched=True, remove_columns=masked_train.column_names)
    print("[TOKENIZING] Processing validation set...")
    masked_val = masked_val.map(tokenize_function, batched=True, remove_columns=masked_val.column_names)
    print("[TOKENIZING] Processing test set...")
    masked_test = masked_test.map(tokenize_function, batched=True, remove_columns=masked_test.column_names)
    print("[SUCCESS] Tokenization complete")

    # ------------------------------
    # Model Setup and Quantization
    # ------------------------------
    print("\\n[STEP 2/6] Setting up model and quantization...")
    hf_logging.set_verbosity_error()
    gc.collect()
    device = 0 if torch.cuda.is_available() else -1
    device_name = 'CUDA' if device == 0 else 'CPU'
    print(f"[INFO] Device: {device_name}")
    if device == 0:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
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
    print(f"[SUCCESS] Model loaded")

    print(f"[INFO] Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print(f"[SUCCESS] Model prepared for k-bit training")

    # ------------------------------
    # LoRA Configuration and Application
    # ------------------------------
    print("\\n[STEP 3/6] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("[INFO] LoRA config: r=8, alpha=32, target_modules=['q_proj', 'v_proj']")
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
    print(f"  - Max steps: 1000")
    print(f"  - Learning rate: 2.5e-5")
    print(f"  - Batch size: 2 (per device)")
    print(f"  - Eval/Save every: 50 steps")
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
            max_steps=1000,
            learning_rate=2.5e-5, # 10x smaller than the mistral learning rate change for the model 
            logging_steps=50,
            bf16=False,
            optim="paged_adamw_8bit",
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
            do_eval=True,
            report_to="wandb",
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
        trainer.train(resume_from_checkpoint=output_dir)
    else:
        print(f"[INFO] Starting fresh training")
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
    print("\n[STEP 6/6] Uploading model to Hugging Face Hub...")
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    api = HfApi()
    repo_id = f"{user_id}/{run_name}"
    print(f"[INFO] Creating repo: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        print(f"[SUCCESS] Repository ready")
    except Exception as e:
        print(f"[WARNING] Could not create repo: {e}")

    print(f"[INFO] Pushing model to hub...")
    try:
        trainer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"QLoRA fine-tuned on {dataset_name}"
        )
        print(f"[SUCCESS] Model pushed to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[WARNING] Could not push to hub: {e}")

    print("\n" + "="*60)
    print("✓ FINE-TUNING PIPELINE COMPLETE")
    print("="*60)
    print(f"Model saved locally at: {output_dir}")
    print(f"Model on Hub: https://huggingface.co/{repo_id}")
    return 0

if __name__ == "__main__":
    import sys
    main_function(sys.argv[1:])
