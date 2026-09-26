# GPU verification run — 2026-09-20

First end-to-end execution of the repaired pipeline on real hardware.

## Environment

| | |
|---|---|
| Host | GCP spot VM `n1-standard-4`, `us-central1-a` |
| GPU | NVIDIA Tesla T4, 15360 MiB, driver 580.178.04 |
| Image | `deeplearning-platform-release/pytorch-2-9-cu129-ubuntu-2204-nvidia-580` |
| Python | 3.10.12 |
| Key libs | torch 2.9.1+cu129, transformers 4.46.3, peft 0.21.0, bitsandbytes 0.50.2 |

Exact pinned versions are in [`requirements.txt`](../requirements.txt).

## What ran

Fine-tuning, `EleutherAI/pythia-410m`, 4-bit NF4 + LoRA, on
`lavita/ChatDoctor-HealthCareMagic-100k` downsampled to 200 rows (160/20/20),
20 optimizer steps:

```
trainable params: 786,432 || all params: 406,120,448 || trainable%: 0.1936
LoRA target modules: ['query_key_value']
eval_loss  3.729
train_loss 3.680
train_runtime 87.6s
```

Loss is finite and decreasing, which is the check that matters for the label-masking
repair: the previous code copied `input_ids` into `labels` wholesale, training the
model to reproduce its own prompt and then emit padding forever.

Evaluation, adapter merged onto the base model, 8 held-out examples:

| condition | BERTScore F1 |
|---|---|
| adapter, no ICL | 0.8486 |
| adapter, ICL k=3 | 0.8568 |

The ICL run logs `ICL corpus: 160 train examples (disjoint from test)` — the
leakage guard is active and passing.

## The measurement that matters

`demo_leakage.py` scores one model on one set of 16 examples four ways,
isolating each of the two evaluation bugs the original code shipped:

| # | condition | BERTScore F1 |
|---|---|---|
| 1 | fixed — train demos, continuation only | **0.8810** |
| 2 | echo only — train demos, prompt echoed back | 0.9042 |
| 3 | leak only — demos retrieved from the eval split | 0.9132 |
| 4 | leak + echo — **exactly what the original `Evaluate.py` did** | **0.9450** |

**Inflation from the two bugs together: +0.0640 F1 (0.8810 → 0.9450).**

Two things follow.

First, the artifact is *larger than the entire effect the project claimed to
measure*. The original README reported a spread of 0.83 (base) → 0.89
(ICL+QLoRA), a difference of 0.06. The bugs alone are worth 0.064.

Second, a 20-step LoRA on a **410M** model scores 0.881 once measured correctly —
at or above the 0.87–0.89 the README attributes to Mistral-7B and Llama-3-70B.
That is not evidence the small model is competitive; it is evidence that
BERTScore against this reference set has a floor high enough to swamp the
differences being reported. Any future claim needs a metric with real dynamic
range on this task, plus confidence intervals.

## Caveat

These numbers come from a 410M model trained for 20 steps on 200 rows and scored
on 8–16 examples. They are a *mechanism* demonstration — that the bugs inflate
scores, and by roughly how much — not a restatement of the project's results at
the original 7B scale. Re-running the real experiments is still outstanding.

## Reproducing

```bash
python Fine_Tune.py --model_id EleutherAI/pythia-410m \
  --torch_dataset_url lavita/ChatDoctor-HealthCareMagic-100k \
  --max_samples 200 --max_steps 20 --project smoke --no_wandb --no_push

python Evaluate.py --model_id EleutherAI/pythia-410m \
  --adapter_id ./pythia-410m-smoke \
  --test_dataset lavita/ChatDoctor-HealthCareMagic-100k \
  --max_samples 200 --limit 8 --batch_size 4 --no_wandb

python demo_leakage.py --model_id EleutherAI/pythia-410m \
  --adapter_id ./pythia-410m-smoke --limit 16
```

`Fine_Tune.py` prints a split fingerprint; pass the same `--seed` and
`--max_samples` to `Evaluate.py` and confirm the fingerprints match, or the two
scripts are not talking about the same held-out set.
