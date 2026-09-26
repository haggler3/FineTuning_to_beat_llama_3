"""Measure the cost/quality tradeoff of adaptive-k retrieval against a fixed
budget, on the same model and the same examples.

Fixed-k spends the same number of demonstration tokens on every query,
whether the query needed them or not. Adaptive-k (qlora_icl.adaptive_retrieval)
spends a variable budget decided from each query's own retrieval scores. This
script is the honest way to find out whether that trade is worth it: run both
strategies over a sweep of settings, on the same examples, and report quality
against actual cost - not the ceiling either strategy was configured with.

Cost is measured in prompt tokens actually sent to the model (context tokens
from the demonstrations, not padding), which is what a per-token API bill or a
GPU-second budget is proportional to.

Usage:
    python scripts/pareto_curve.py \
        --model_id EleutherAI/pythia-410m --adapter_id ./pythia-410m-smoke \
        --limit 16
"""

import argparse
import csv
import os

import numpy as np
import torch
from evaluate import load as load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from qlora_icl.adaptive_retrieval import AdaptiveKConfig, select_k
from qlora_icl.splits import build_splits

MAX_SCORED_TOKENS = 512


def build_context_and_cost(retrieved, tokenizer):
    """Render demonstrations as prompt text and count their tokens."""
    context = "".join(f"Input: {r['input']}\nOutput: {r['label']}\n\n" for r in retrieved)
    n_tokens = len(tokenizer(context, add_special_tokens=False)["input_ids"]) if context else 0
    return context, n_tokens


def retrieve_pool(queries, corpus_inputs, corpus_outputs, encoder, util, corpus_embeddings, pool_size):
    q_emb = encoder.encode(queries, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, corpus_embeddings)
    pools = []
    for sim_vector in sims:
        idxs = torch.topk(sim_vector, k=min(pool_size, len(corpus_inputs))).indices.tolist()
        pools.append([
            {"input": corpus_inputs[i], "label": corpus_outputs[i], "score": sim_vector[i].item()}
            for i in idxs
        ])
    return pools


def run_condition(label, k_for_pool, pools, instructions, queries, references,
                  tokenizer, model, bertscore, judge_model, adaptive_config=None):
    """Build prompts for one condition (fixed-k or adaptive), generate, score.

    k_for_pool caps how much of each pool is available to spend; adaptive_config,
    when given, decides how much of that budget each query actually uses.
    """
    prompts, k_used, demo_tokens = [], [], []
    for pool, instr, query in zip(pools, instructions, queries, strict=True):
        available = pool[:k_for_pool]
        if adaptive_config is not None:
            k = select_k([r["score"] for r in available], adaptive_config)
        else:
            k = len(available)
        context, n_tok = build_context_and_cost(available[:k], tokenizer)
        k_used.append(k)
        demo_tokens.append(n_tok)
        prompts.append(context + f"Instruction: {instr}\nInput: {query}\nResponse:")

    gen = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        do_sample=False, max_new_tokens=64, return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    outs = gen(prompts, batch_size=4)
    preds = [(o[0] if isinstance(o, list) else o)["generated_text"] for o in outs]
    preds = [p if p.strip() else " " for p in preds]

    judge_tok = AutoTokenizer.from_pretrained(judge_model)

    def _truncate(text, limit=500):
        ids = judge_tok(text, truncation=True, max_length=limit, add_special_tokens=False)["input_ids"]
        return judge_tok.decode(ids, skip_special_tokens=True)

    preds_t = [_truncate(p) for p in preds]
    refs_t = [_truncate(r) for r in references]

    res = bertscore.compute(
        predictions=preds_t, references=refs_t, model_type=judge_model,
        num_layers=12, lang="en", device="cuda" if torch.cuda.is_available() else "cpu",
    )

    return {
        "label": label,
        "f1": float(np.mean(res["f1"])),
        "mean_k": float(np.mean(k_used)),
        "mean_demo_tokens": float(np.mean(demo_tokens)),
        "total_demo_tokens": int(np.sum(demo_tokens)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', default="EleutherAI/pythia-410m")
    ap.add_argument('--adapter_id', default=None)
    ap.add_argument('--dataset', default="lavita/ChatDoctor-HealthCareMagic-100k")
    ap.add_argument('--max_samples', type=int, default=200)
    ap.add_argument('--limit', type=int, default=16)
    ap.add_argument('--max_k', type=int, default=5, help="Ceiling both fixed-k and adaptive-k budgets share.")
    ap.add_argument('--judge_model', default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    ap.add_argument('--output_csv', default=None)
    args = ap.parse_args()

    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    train, _val, test = build_splits(args.dataset, token=token, max_samples=args.max_samples)
    test = test.select(range(min(args.limit, len(test))))

    queries = test["input"]
    references = test["output"]
    instructions = test["instruction"]

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if args.adapter_id:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_id).merge_and_unload()
    model.eval()

    from sentence_transformers import SentenceTransformer, util
    encoder = SentenceTransformer('sentence-transformers/gtr-t5-base')
    corpus_inputs, corpus_outputs = train["input"], train["output"]
    corpus_embeddings = encoder.encode(corpus_inputs, convert_to_tensor=True, show_progress_bar=False)

    pools = retrieve_pool(queries, corpus_inputs, corpus_outputs, encoder, util,
                          corpus_embeddings, args.max_k)

    bertscore = load_metric("bertscore")

    conditions = [
        ("fixed k=1", 1, None),
        ("fixed k=3", 3, None),
        (f"fixed k={args.max_k}", args.max_k, None),
        ("adaptive (drop=0.95, tight)", args.max_k,
         AdaptiveKConfig(min_k=1, max_k=args.max_k, relative_drop=0.95)),
        ("adaptive (drop=0.85, default)", args.max_k,
         AdaptiveKConfig(min_k=1, max_k=args.max_k, relative_drop=0.85)),
        ("adaptive (drop=0.60, loose)", args.max_k,
         AdaptiveKConfig(min_k=1, max_k=args.max_k, relative_drop=0.60)),
    ]

    print(f"\nCost/quality tradeoff on {len(test)} examples "
          f"(model={args.model_id}, max_k={args.max_k}):\n")
    header = f"  {'condition':<32} {'mean k':>8} {'mean tok':>10} {'F1':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for label, k_for_pool, adaptive_config in conditions:
        r = run_condition(label, k_for_pool, pools, instructions, queries, references,
                          tok, model, bertscore, args.judge_model, adaptive_config)
        results.append(r)
        print(f"  {r['label']:<32} {r['mean_k']:>8.2f} {r['mean_demo_tokens']:>10.1f} {r['f1']:>8.4f}")

    fixed_full = next(r for r in results if r['label'] == f"fixed k={args.max_k}")
    best_adaptive = max((r for r in results if r['label'].startswith('adaptive')),
                        key=lambda r: r['f1'] - 0.0001 * r['mean_demo_tokens'])
    token_savings = 1 - best_adaptive['mean_demo_tokens'] / max(fixed_full['mean_demo_tokens'], 1)
    f1_gap = fixed_full['f1'] - best_adaptive['f1']
    print(f"\n  Best adaptive setting vs fixed k={args.max_k}: "
          f"{token_savings:+.1%} demo tokens, {-f1_gap:+.4f} F1")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["label", "mean_k", "mean_demo_tokens",
                                                     "total_demo_tokens", "f1"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
