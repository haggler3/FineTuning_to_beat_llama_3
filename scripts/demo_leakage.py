"""Quantify what the original evaluation bugs were worth, in BERTScore points.

Scores the same model on the same examples four ways:

  1. fixed            - demos from train, prediction is the continuation only
  2. echo only        - demos from train, prediction includes the prompt
  3. leak only        - demos retrieved from the eval split itself
  4. leak + echo      - both, i.e. exactly what the original Evaluate.py did

Run this to see how much of a reported score came from the model and how much
came from the gold answer being visible in its own prompt.
"""

import argparse
import os

import numpy as np
import torch
from evaluate import load as load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from qlora_icl.splits import build_splits


def build_prompts(queries, instructions, corpus_inputs, corpus_outputs,
                  encoder, util, corpus_embeddings, top_k):
    """Standard Dr.ICL prompt: k retrieved demonstrations, then the question."""
    q_emb = encoder.encode(queries, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, corpus_embeddings)

    prompts = []
    for sim_vector, instr, query in zip(sims, instructions, queries, strict=True):
        idxs = torch.topk(sim_vector, k=min(top_k, len(corpus_inputs))).indices.tolist()
        ctx = "".join(
            f"Input: {corpus_inputs[i]}\nOutput: {corpus_outputs[i]}\n\n" for i in idxs
        )
        prompts.append(ctx + f"Instruction: {instr}\nInput: {query}\nResponse:")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', default="EleutherAI/pythia-410m")
    ap.add_argument('--adapter_id', default=None)
    ap.add_argument('--dataset', default="lavita/ChatDoctor-HealthCareMagic-100k")
    ap.add_argument('--max_samples', type=int, default=200)
    ap.add_argument('--limit', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=3)
    ap.add_argument('--judge_model', default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    args = ap.parse_args()

    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    train, _val, test = build_splits(args.dataset, token=token,
                                     max_samples=args.max_samples)
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

    # Two corpora: the correct one (train) and the leaky one (the eval split).
    corpora = {
        "train": (train["input"], train["output"]),
        "leak": (queries, references),
    }
    embeddings = {
        name: encoder.encode(inp, convert_to_tensor=True, show_progress_bar=False)
        for name, (inp, _out) in corpora.items()
    }

    bertscore = load_metric("bertscore")
    judge_tok = AutoTokenizer.from_pretrained(args.judge_model)

    def _truncate(text, limit=500):
        ids = judge_tok(text, truncation=True, max_length=limit,
                        add_special_tokens=False)["input_ids"]
        return judge_tok.decode(ids, skip_special_tokens=True)

    def score(corpus_name, echo_prompt):
        corpus_inputs, corpus_outputs = corpora[corpus_name]
        prompts = build_prompts(
            queries, instructions, corpus_inputs, corpus_outputs,
            encoder, util, embeddings[corpus_name], args.top_k,
        )
        gen = pipeline(
            "text-generation", model=model, tokenizer=tok,
            do_sample=False, max_new_tokens=64,
            return_full_text=echo_prompt,
            pad_token_id=tok.pad_token_id,
        )
        outs = gen(prompts, batch_size=4)
        preds = [(o[0] if isinstance(o, list) else o)["generated_text"] for o in outs]
        preds = [p if p.strip() else " " for p in preds]

        # The judge is a BERT model with a hard 512-token limit, and an echoed
        # prompt easily exceeds it. Truncate every condition the same way so the
        # comparison isolates leakage and echo, not truncation differences.
        preds = [_truncate(p) for p in preds]
        refs = [_truncate(r) for r in references]

        res = bertscore.compute(
            predictions=preds, references=refs,
            model_type=args.judge_model, num_layers=12, lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return float(np.mean(res["f1"]))

    print(f"\nScoring the same model on the same {len(test)} examples, four ways:\n")
    conditions = [
        ("1. fixed        (train demos, continuation only)", "train", False),
        ("2. echo only    (train demos, prompt echoed)    ", "train", True),
        ("3. leak only    (test demos, continuation only) ", "leak", False),
        ("4. leak + echo  (the original Evaluate.py)      ", "leak", True),
    ]

    results = {}
    for label, corpus, echo in conditions:
        f1 = score(corpus, echo)
        results[label] = f1
        print(f"  {label}  BERTScore F1 = {f1:.4f}")

    baseline = results[conditions[0][0]]
    worst = results[conditions[3][0]]
    print(f"\n  Inflation from the two bugs: {worst - baseline:+.4f} F1 "
          f"({baseline:.4f} -> {worst:.4f})")


if __name__ == "__main__":
    main()
