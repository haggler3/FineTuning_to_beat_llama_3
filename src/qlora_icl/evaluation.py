import argparse
import json
import os

import numpy as np
import torch
from evaluate import load as load_metric
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from qlora_icl.adaptive_retrieval import AdaptiveKConfig, select_k
from qlora_icl.splits import DEFAULT_MAX_SAMPLES, DEFAULT_SEED, build_splits, split_fingerprint

MAX_SCORED_TOKENS = 512


# --- Batched Retrieval for In-Context Learning ---
def retrieve_top_k(corpus_inputs, corpus_embeddings, corpus_outputs,
                   model_GTR, util, query_inputs, top_k=5):
    """Retrieve top-k similar demonstrations for each query.

    The corpus must come from a split the queries are NOT drawn from, or each
    query retrieves itself and its own gold answer lands in the prompt.
    """
    query_embeddings = model_GTR.encode(query_inputs, convert_to_tensor=True)
    similarities = util.cos_sim(query_embeddings, corpus_embeddings)  # [batch, N]

    all_retrieved = []
    for sim_vector in similarities:
        top_indices = torch.topk(sim_vector, k=min(top_k, len(corpus_inputs))).indices.tolist()
        all_retrieved.append([
            {
                "input": corpus_inputs[idx],
                "label": corpus_outputs[idx],
                "score": sim_vector[idx].item(),
            }
            for idx in top_indices
        ])
    return all_retrieved


def build_icl_context(retrieved):
    """Render a list of {input, label, score} demonstrations as prompt text."""
    return "".join(
        f"Input: {r['input']}\nOutput: {r['label']}\n\n" for r in retrieved
    )


def icl_prompts_batch(inputs, corpus_inputs, corpus_embeddings, corpus_outputs,
                      model_GTR, util, top_k=5, adaptive_config=None):
    """Build an ICL context string for each input in the batch.

    In fixed mode (adaptive_config=None) every query gets exactly top_k
    demonstrations, retrieved best-first. In adaptive mode, a pool of up to
    adaptive_config.max_k candidates is retrieved per query and select_k()
    decides how many of them to actually spend - an easy query (one clearly
    best match) may use just one; a query with several close matches may use
    the full pool.

    Returns (prompts, k_used) - k_used[i] is how many demonstrations ended up
    in prompt i's context, so callers can report what the adaptive budget
    actually spent.
    """
    pool_size = max(top_k, adaptive_config.max_k) if adaptive_config else top_k
    retrieved_batches = retrieve_top_k(
        corpus_inputs, corpus_embeddings, corpus_outputs,
        model_GTR, util, inputs, top_k=pool_size,
    )

    prompt_contexts, k_used = [], []
    for retrieved in retrieved_batches:
        # retrieve_top_k comes from torch.topk, which is sorted descending, so
        # retrieved is already best-first - slicing the first k keeps the best.
        if adaptive_config is not None:
            k = select_k([r["score"] for r in retrieved], adaptive_config)
        else:
            k = min(top_k, len(retrieved))
        prompt_contexts.append(build_icl_context(retrieved[:k]))
        k_used.append(k)
    return prompt_contexts, k_used


def assert_no_leakage(corpus_inputs, eval_inputs):
    """Fail loudly if the retrieval corpus overlaps the evaluation set.

    An overlap means retrieved demonstrations can carry the gold answer for the
    very question being scored, which inflates every metric downstream.
    """
    overlap = set(corpus_inputs) & set(eval_inputs)
    if overlap:
        raise ValueError(
            f"Retrieval corpus overlaps the eval set on {len(overlap)} example(s). "
            "Scores computed this way are not meaningful. Retrieve from the train split."
        )


def main_function(cli_args):
    parser = argparse.ArgumentParser(description='Evaluate a tuned model via unbiased LLM with optional ICL retrieval.')
    parser.add_argument('--model_id', type=str, default="mistralai/Mistral-7B-Instruct-v0.1",
                        help="Base model name or path for text generation.")
    parser.add_argument('--adapter_id', type=str, default=None,
                        help="Path or Hub id of a LoRA adapter to load on top of --model_id. "
                             "Omit to evaluate the base model.")
    parser.add_argument('--test_dataset', type=str, required=True,
                        help="Path or HuggingFace repo for the dataset (e.g., lavita/ChatDoctor-HealthCareMagic-100k)")
    parser.add_argument('--use_icl', action='store_true',
                        help="Enable in-context learning retrieval for prompt construction.")
    parser.add_argument('--icl_input_field', type=str, default="input",
                        help="Which field to use for retrieval/prompt in ICL mode (default: input)")
    parser.add_argument('--icl_label_field', type=str, default="output",
                        help="Which field to use for label/reference in ICL mode (default: output)")
    parser.add_argument('--judge_model', type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                        help="Model name for BERTScore metric judge.")
    parser.add_argument('--project', default="medical-qa-evaluation", help="wandb project name")
    parser.add_argument('--no_wandb', action='store_true', help="Disable Weights & Biases logging.")
    parser.add_argument('--hf_token', default=None, help="Optional: HF_TOKEN (or from env)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for text generation and evaluation.")
    parser.add_argument('--icl_top_k', type=int, default=5,
                        help="Fixed number of demonstrations per query. Also the ceiling on "
                             "how many a query can use in --adaptive_k mode.")
    parser.add_argument('--adaptive_k', action='store_true',
                        help="Choose how many demonstrations to use per query instead of "
                             "always using --icl_top_k. An easy query (one clearly best match "
                             "in the corpus) spends fewer tokens; a query with several equally "
                             "good matches spends up to --icl_top_k. No effect without --use_icl.")
    parser.add_argument('--adaptive_min_k', type=int, default=1,
                        help="Minimum demonstrations per query in --adaptive_k mode.")
    parser.add_argument('--adaptive_relative_drop', type=float, default=0.85,
                        help="In --adaptive_k mode, keep a demonstration only while its "
                             "similarity score is >= this fraction of the query's best match.")
    parser.add_argument('--adaptive_absolute_floor', type=float, default=0.0,
                        help="In --adaptive_k mode, never keep a demonstration below this "
                             "absolute similarity score, regardless of --adaptive_relative_drop.")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help="Must match the --seed used by Fine_Tune.py, or the test split will differ.")
    parser.add_argument('--max_samples', type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Must match the --max_samples used by Fine_Tune.py.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Evaluate only the first N test examples (for smoke tests).")
    parser.add_argument('--output_json', type=str, default=None,
                        help="Write final metrics as JSON to this path, for CI/CD or a dashboard "
                             "to consume without scraping stdout.")
    args = parser.parse_args(cli_args)

    adaptive_config = None
    if args.use_icl and args.adaptive_k:
        adaptive_config = AdaptiveKConfig(
            min_k=args.adaptive_min_k,
            max_k=args.icl_top_k,
            relative_drop=args.adaptive_relative_drop,
            absolute_floor=args.adaptive_absolute_floor,
        )

    HF_TOKEN = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if HF_TOKEN is None:
        raise ValueError("No Hugging Face token provided for dataset or model.")

    # --- Dataset Loading ---
    # The same partition Fine_Tune.py trained on, so 'test' really is held out.
    train_dataset, _val_dataset, test_dataset = build_splits(
        args.test_dataset, token=HF_TOKEN, seed=args.seed, max_samples=args.max_samples
    )
    if args.limit:
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))

    fingerprint = split_fingerprint(
        args.test_dataset, args.seed, args.max_samples,
        [len(train_dataset), len(_val_dataset), len(test_dataset)],
    )
    print(f"[INFO] Split fingerprint: {fingerprint}")
    print(f"[INFO] Test examples: {len(test_dataset)}")

    eval_inputs = test_dataset[args.icl_input_field]
    eval_labels = test_dataset[args.icl_label_field]
    eval_instructions = (
        test_dataset["instruction"] if "instruction" in test_dataset.column_names
        else [""] * len(test_dataset)
    )

    # --- wandb Setup ---
    use_wandb = not args.no_wandb
    if use_wandb:
        # Imported lazily so --no_wandb runs without the dependency installed.
        import wandb
        wandb.init(
            project=args.project,
            config={
                "model": args.model_id,
                "adapter": args.adapter_id,
                "eval_metric": "BERTScore",
                "test_set_size": len(test_dataset),
                "use_icl": args.use_icl,
                "icl_top_k": args.icl_top_k,
                "adaptive_k": args.adaptive_k,
                "adaptive_min_k": args.adaptive_min_k if adaptive_config else None,
                "adaptive_relative_drop": args.adaptive_relative_drop if adaptive_config else None,
                "judge_model": args.judge_model,
                "split_fingerprint": fingerprint,
                "seed": args.seed,
            }
        )

    # --- Model & Tokenizer Loading ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Generation needs left padding so the continuation follows the real prompt.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if args.adapter_id:
        # Without this the script silently scores the base model no matter what
        # was fine-tuned, which is what the original version did.
        from peft import PeftModel
        print(f"[INFO] Loading LoRA adapter: {args.adapter_id}")
        model = PeftModel.from_pretrained(model, args.adapter_id, token=HF_TOKEN)
        model = model.merge_and_unload()
        print("[SUCCESS] Adapter merged into base model")
    else:
        print("[INFO] No --adapter_id given; evaluating the BASE model")

    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,          # greedy; temperature/top_k do not apply
        max_new_tokens=256,
        return_full_text=False,   # score the continuation, not prompt + continuation
        pad_token_id=tokenizer.pad_token_id,
    )

    bertscore = load_metric("bertscore")

    # --- Optional: ICL retrieval corpus (train split only) ---
    if args.use_icl:
        from sentence_transformers import SentenceTransformer, util
        corpus_inputs = train_dataset[args.icl_input_field]
        corpus_outputs = train_dataset[args.icl_label_field]

        assert_no_leakage(corpus_inputs, eval_inputs)

        model_GTR = SentenceTransformer('sentence-transformers/gtr-t5-base')
        corpus_embeddings = model_GTR.encode(
            corpus_inputs, convert_to_tensor=True, show_progress_bar=True
        )
        print(f"[INFO] ICL corpus: {len(corpus_inputs)} train examples (disjoint from test)")
    else:
        model_GTR = util = None
        corpus_inputs = corpus_outputs = corpus_embeddings = None

    # --- Batched Prediction Generation and Evaluation ---
    batch_size = args.batch_size
    test_f1, test_recall, test_precision = [], [], []
    all_k_used = []

    num_examples = len(test_dataset)
    print("[INFO] Starting batch evaluation...")
    for i in tqdm(range(0, num_examples, batch_size), desc="Batch Evaluating"):
        # Slicing a Dataset yields a dict of columns, so index the lists directly.
        batch_inputs = eval_inputs[i: i + batch_size]
        batch_instructions = eval_instructions[i: i + batch_size]
        batch_references = eval_labels[i: i + batch_size]

        if args.use_icl:
            batch_icl_contexts, batch_k_used = icl_prompts_batch(
                batch_inputs, corpus_inputs, corpus_embeddings, corpus_outputs,
                model_GTR, util, top_k=args.icl_top_k, adaptive_config=adaptive_config,
            )
            all_k_used.extend(batch_k_used)
            batch_prompts = [
                icl_ctx + f"Instruction: {ins}\nInput: {inp}\nResponse:"
                for icl_ctx, ins, inp in zip(batch_icl_contexts, batch_instructions, batch_inputs, strict=True)
            ]
        else:
            batch_prompts = [
                f"Instruction: {ins}\nInput: {inp}\nResponse:"
                for ins, inp in zip(batch_instructions, batch_inputs, strict=True)
            ]

        # Let each batch generate about as much as its references need.
        batch_target_lengths = [len(tokenizer(out)["input_ids"]) for out in batch_references]
        max_new_tokens = max(1, min(max(batch_target_lengths), 256))

        batch_outputs_gen = generator(batch_prompts, max_new_tokens=max_new_tokens)

        # A batched pipeline call returns one list of candidates per prompt.
        batch_generated_texts = [
            (out[0]["generated_text"] if isinstance(out, list) else out["generated_text"])
            for out in batch_outputs_gen
        ]

        # Truncate by tokens, matching what the comment always claimed.
        def truncate_tokens(text):
            ids = tokenizer(text, truncation=True, max_length=MAX_SCORED_TOKENS,
                            add_special_tokens=False)["input_ids"]
            return tokenizer.decode(ids, skip_special_tokens=True)

        predictions_trunc = [truncate_tokens(s) for s in batch_generated_texts]
        references_trunc = [truncate_tokens(r) for r in batch_references]

        # BERTScore rejects empty strings; a greedy model can emit nothing.
        predictions_trunc = [p if p.strip() else " " for p in predictions_trunc]

        out = bertscore.compute(
            predictions=predictions_trunc,
            references=references_trunc,
            model_type=args.judge_model,
            num_layers=12,
            batch_size=32,
            lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        test_precision.extend(out["precision"])
        test_recall.extend(out["recall"])
        test_f1.extend(out["f1"])

        if use_wandb:
            wandb.log({
                "batch/precision": float(np.mean(out["precision"])),
                "batch/recall": float(np.mean(out["recall"])),
                "batch/f1": float(np.mean(out["f1"])),
                "batch_start_idx": i
            })

    # --- Final aggregate metrics ---
    mean_precision = float(np.mean(test_precision))
    mean_recall = float(np.mean(test_recall))
    mean_f1 = float(np.mean(test_f1))
    mean_k = float(np.mean(all_k_used)) if all_k_used else None

    if use_wandb:
        wandb_log = {
            "eval/precision": mean_precision,
            "eval/recall": mean_recall,
            "eval/f1": mean_f1,
        }
        if mean_k is not None:
            wandb_log["eval/mean_demos_per_query"] = mean_k
        wandb.log(wandb_log)

    print(f'[RESULT] Model averages - f1: {mean_f1:.4f} | recall: {mean_recall:.4f} | precision: {mean_precision:.4f}')
    if mean_k is not None:
        print(f'[RESULT] Mean demonstrations per query: {mean_k:.2f} '
              f'(ceiling={args.icl_top_k}, adaptive={args.adaptive_k})')
    print(f'[RESULT] split={fingerprint} n={num_examples} icl={args.use_icl} adapter={args.adapter_id}')

    if args.output_json:
        # Machine-readable results, for a CI gate or a dashboard to read without
        # scraping stdout. Every number needed to judge whether this run is
        # comparable to another one is included, not just the headline score.
        payload = {
            "model_id": args.model_id,
            "adapter_id": args.adapter_id,
            "test_dataset": args.test_dataset,
            "split_fingerprint": fingerprint,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "num_examples": num_examples,
            "use_icl": args.use_icl,
            "icl_top_k": args.icl_top_k if args.use_icl else None,
            "adaptive_k": args.adaptive_k if args.use_icl else False,
            "mean_demonstrations_per_query": mean_k,
            "judge_model": args.judge_model,
            "metrics": {
                "bertscore_f1": mean_f1,
                "bertscore_precision": mean_precision,
                "bertscore_recall": mean_recall,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[INFO] Wrote results JSON to {args.output_json}")

    if use_wandb:
        wandb.finish()
    return 0


def main(argv=None):
    """Console-script entry point."""
    import sys
    return main_function(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
