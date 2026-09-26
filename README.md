# QLoRA + Dr.ICL

[![CI](https://github.com/CMUZrz/FineTuning_to_beat_llama_3/actions/workflows/ci.yml/badge.svg)](https://github.com/CMUZrz/FineTuning_to_beat_llama_3/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Changelog](https://img.shields.io/badge/changelog-Unreleased-informational.svg)](CHANGELOG.md)

Parameter-efficient fine-tuning of small language models with
demonstration-retrieved in-context learning — packaged, tested, and with an
evaluation harness built so that a leaked demonstration fails the run instead
of inflating the score.

If you're evaluating this for production use: `pip install`-able package with
pinned dependencies, CI across two OSes and two Python versions, machine-readable
(`--output_json`) evaluation output for a CI gate or dashboard, and an
adaptive-retrieval mode built to cut inference cost without a corresponding
quality drop (below). None of that substitutes for running it against your own
data before you rely on it.

Originally the course project for **10-623 Generative AI** at Carnegie Mellon
University, by Dan Jung, Dhruva Byrapatna and Zachary Zdobinski.

---

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <img alt="A single build_splits call partitions the dataset into train, validation and test and emits a fingerprint that both entry points print. The training lane tokenizes the train split with completion-only labels, attaches LoRA adapters to a 4-bit quantized base, and emits an adapter. The evaluation lane retrieves demonstrations from the train split only, guarded by an assert_no_leakage check, builds an in-context prompt around the test split, generates from the merged base plus adapter, and scores the continuation alone." src="docs/architecture.svg">
</picture>

Two properties the picture is drawn to make obvious, because they are what the
measurements depend on:

- **One partition feeds both lanes.** `build_splits()` is a pure function of
  `(dataset, seed, max_samples)`, so training and evaluation agree on what "held
  out" means. Both entry points print a **split fingerprint**; if the two runs
  print different fingerprints, they are not scoring the same examples.
- **Demonstrations come from `train`, never from the split being scored.**
  `assert_no_leakage()` sits on that edge and raises if the retrieval corpus and
  the evaluation set intersect at all.

The diagram is generated — edit [`docs/make_architecture_svg.py`](docs/make_architecture_svg.py)
and re-run it rather than hand-editing the SVGs, so the light and dark variants
cannot drift apart.

---

## Quick start

```bash
git clone https://github.com/CMUZrz/FineTuning_to_beat_llama_3.git
cd FineTuning_to_beat_llama_3
pip install -e .
```

Fine-tune, then evaluate the resulting adapter:

```bash
export HUGGINGFACE_HUB_TOKEN="hf_..."

qlora-train \
  --model_id EleutherAI/pythia-410m \
  --torch_dataset_url lavita/ChatDoctor-HealthCareMagic-100k \
  --max_samples 4000 --max_steps 1000 \
  --project healthcare --no_push
```

```bash
qlora-eval \
  --model_id EleutherAI/pythia-410m \
  --adapter_id ./pythia-410m-healthcare \
  --test_dataset lavita/ChatDoctor-HealthCareMagic-100k \
  --max_samples 4000 --use_icl --icl_top_k 5
```

> Pass the **same** `--seed` and `--max_samples` to both commands and confirm they
> print the same split fingerprint. Different fingerprints mean the evaluation set
> is not the one that was held out.

A CUDA GPU is required: 4-bit quantization goes through `bitsandbytes`, which has
no usable CPU path. `qlora-train` fails with a clear message rather than limping
along on CPU. See [Infrastructure](#infrastructure) for a one-command spot T4.

---

## Method

**QLoRA.** The base model is loaded in 4-bit NormalFloat (NF4) with double
quantization, gradient checkpointing, and paged 8-bit AdamW. LoRA adapters
(`r=8`, `alpha=32`) attach to the attention projections, leaving roughly 0.2% of
parameters trainable. Target modules are detected per architecture — Llama and
Mistral expose `q_proj`/`v_proj`, while GPTNeoX models such as Pythia fuse them
into a single `query_key_value`, and attaching to the wrong one silently trains
the MLP instead of attention.

**Completion-only loss.** Labels are built by
[`build_labels()`](src/qlora_icl/train.py): padding and prompt positions are set
to `-100` so neither contributes to the loss. Because `pad_token` is
`eos_token`, copying `input_ids` into `labels` wholesale — the obvious
implementation — trains the model to echo its prompt and then emit
end-of-sequence forever.

**Dr.ICL.** A GTR-T5 dual encoder embeds the training split; each evaluation
query retrieves its top-k nearest demonstrations, which are prepended to the
prompt. Retrieval is restricted to `train` and asserted disjoint from the
evaluation split.

**Scoring.** Generation runs greedily with `return_full_text=False`, so the
BERTScore judge sees the model's continuation rather than the prompt it was
given. Predictions and references are truncated by *token* count against the
judge's own tokenizer.

---

## Adaptive-k retrieval

Fixed-k Dr.ICL spends the same number of demonstration tokens on every query,
whether the query needed them or not: an easy question pays for k demos it
didn't need, and a question with no good match in the corpus gets k irrelevant
ones anyway. [`select_k()`](src/qlora_icl/adaptive_retrieval.py) replaces the
fixed count with a per-query budget decided from that query's own retrieval
scores — keep demonstrations while they stay close to the best match, stop at
the first one that falls off a cliff:

```bash
qlora-eval \
  --model_id EleutherAI/pythia-410m --adapter_id ./pythia-410m-healthcare \
  --test_dataset lavita/ChatDoctor-HealthCareMagic-100k --max_samples 4000 \
  --use_icl --icl_top_k 5 --adaptive_k \
  --output_json results/eval.json
```

`--icl_top_k` becomes the ceiling; each query spends up to that many
demonstrations, and `--output_json` reports `mean_demonstrations_per_query`
alongside the score, so you can see what was actually spent, not just what was
configured. [`scripts/pareto_curve.py`](scripts/pareto_curve.py) runs fixed-k
and several adaptive-k settings on the same model and the same examples and
reports mean demonstration tokens against BERTScore F1 — the honest way to find
out whether the trade is worth it on your data, rather than assume it.

**Status:** the selection logic has 15 unit tests covering clustered scores,
steep dropoffs, the min/max-k floor and ceiling, and order-independence, plus
an integration test proving two queries with different score shapes really do
spend a different number of demonstrations end-to-end (all in
[`test_adaptive_retrieval.py`](tests/test_adaptive_retrieval.py) and
[`test_evaluation.py`](tests/test_evaluation.py)). It has **not** yet had the
GPU measurement `demo_leakage.py` got for the leakage fix below — that's what
`pareto_curve.py` is for, and it hasn't been run yet. Treat the mechanism as
verified and the cost/quality tradeoff as unmeasured until that changes.

---

## Evaluation integrity

This repository treats a plausible-looking number as a bug until it is shown not
to be. Three mechanisms enforce that, each with a test that fails if it regresses:

| Mechanism | What it prevents | Test |
|---|---|---|
| `split_fingerprint()` | Scoring on examples the model trained on | [`test_train.py`](tests/test_train.py) |
| `assert_no_leakage()` | Retrieved demonstrations carrying the gold answer | [`test_leakage.py`](tests/test_leakage.py) |
| `return_full_text=False` | The prompt being scored as if it were the answer | [`test_evaluation.py`](tests/test_evaluation.py) |
| `build_labels()` | Training on padding and prompt tokens | [`test_labels.py`](tests/test_labels.py) |

[`scripts/demo_leakage.py`](scripts/demo_leakage.py) exists to show why this is
not paranoia. It scores one model on one set of examples four ways, isolating
each failure mode:

| condition | BERTScore F1 |
|---|---|
| demos from train, continuation only | **0.8810** |
| demos from train, prompt echoed back | 0.9042 |
| demos from the eval split, continuation only | 0.9132 |
| demos from the eval split **and** prompt echoed | **0.9450** |

**+0.0640 F1 from the two bugs alone.** For calibration, a typical reported
spread between a base model and a tuned one on this task is around 0.06 — so the
artifact can be the same size as the effect being measured.

A related caution: in that run a **410M** model trained for **20 steps** scored
0.881 when measured correctly. BERTScore against this reference set has a floor
high enough to compress real differences, so any comparison drawn from it needs
confidence intervals before it means much.

Full run details: [`results/gpu_verification_2026-09-20.md`](results/gpu_verification_2026-09-20.md).

---

## Results

The pipeline is verified end-to-end on an NVIDIA T4 (see the link above).
Headline accuracy numbers from the original coursework are **not yet restored**
to this repository — they predate the evaluation fixes above and are being
re-measured rather than copied forward. This section will carry re-run numbers,
with confidence intervals and a split fingerprint per row, as they land.

---

## Repository layout

```
src/qlora_icl/
  splits.py               deterministic partition + fingerprint (shared)
  train.py                QLoRA fine-tuning              -> qlora-train
  evaluation.py           Dr.ICL evaluation harness       -> qlora-eval
  tokenize_dataset.py     standalone tokenization         -> qlora-tokenize
  adaptive_retrieval.py   per-query demonstration budget (select_k)
scripts/
  demo_leakage.py       quantifies the leakage and prompt-echo artifacts
  pareto_curve.py       fixed-k vs adaptive-k cost/quality measurement
  run_tests_light.py    zero-install test runner (what CI executes)
tests/
  fakes.py              Dataset/Tokenizer doubles with real semantics
docs/
  make_architecture_svg.py   regenerates both diagram themes
infra/
  terraform/           GCP spot T4 + GCS checkpoint bucket
results/               measured runs, committed
Dockerfile            CUDA runtime image (not build-verified, see below)
CHANGELOG.md          what changed and why, most recent first
```

---

## Deployment

**Package.** `pip install -e .` for development, or `pip install .` for a
pinned, reproducible install — every ML dependency in
[`requirements.txt`](requirements.txt) is version-pinned, with the reason
noted for the two that are pinned *below* their latest major version
(`transformers`, `sentence-transformers`) rather than left floating.

**CI.** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the test
suite on Ubuntu and Windows across Python 3.10 and 3.12, lints with `ruff`,
and builds + validates the package metadata (`twine check`) on every push and
PR — a red badge above means don't merge, not "check it manually."

**Machine-readable results.** `qlora-eval --output_json path.json` writes
BERTScore, the split fingerprint, and (in adaptive mode) mean demonstrations
per query as JSON, for a pipeline to gate on or a dashboard to plot, rather
than scraping stdout.

**Docker.** A [`Dockerfile`](Dockerfile) is provided for `qlora-train` /
`qlora-eval` / `qlora-tokenize` on a CUDA 12.4 runtime base. **It has not been
build-verified in this pass or in CI** — pulling and building a full CUDA base
image is heavier than this repo's other checks support, so validate it against
your own registry and host before depending on it in production.

**What's still missing for a hardened production deployment:** dependency
vulnerability scanning, a signed/pinned base image digest rather than a
floating tag, and load-bearing metrics/alerting beyond the JSON output above.
None of that is present yet — said plainly rather than implied by the presence
of a Dockerfile.

---

## Testing

```bash
python scripts/run_tests_light.py    # no GPU, no ML install, runs in seconds
```

The runner stubs `torch`, `transformers`, `peft` and friends, so the suite runs
anywhere. What it does **not** stub is the code under test: `tests/fakes.py`
provides a `Dataset` double with correct indexing semantics (slicing yields a
dict of columns, not a list of rows) and a whitespace tokenizer that tracks real
character offsets. A `MagicMock` accepts any misuse silently; these doubles do
not.

The runner also **fails when it discovers zero tests**. Test files are matched as
`test_*.py`; an earlier `Test_*.py` naming matched only on case-insensitive
filesystems, so the suite collected nothing on Linux and still reported success.
CI now runs on Ubuntu and Windows across Python 3.10 and 3.12 for that reason.

---

## Infrastructure

[`infra/terraform/`](infra/terraform) provisions a preemptible **NVIDIA T4**
(`n1-standard-4`, ~$0.11/hr) from a prebaked PyTorch + CUDA 12.9 deep learning
image, with a versioned GCS bucket for preemption checkpoints and a
least-privilege service account.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # set your project_id
terraform init && terraform apply
```

---

## Related work

- **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al., 2023. The
  4-bit NF4 + paged optimizer + LoRA recipe this repository implements.
- **Dr.ICL: Demonstration-Retrieved In-Context Learning** — Luo et al., 2023. The
  retrieval-augmented prompting strategy evaluated here.
- **Large Dual Encoders Are Generalizable Retrievers** — Ni et al., 2021. Basis
  for using GTR-T5 as the dense retriever.
- **AnyTaskTune: Advanced Domain-Specific Solutions through Task-Fine-Tuning** —
  Cui et al., 2024. Domain-specific dataset construction.
- **Textbooks Are All You Need** — Gunasekar et al., 2023. Small models with
  high-quality data.

---

## License

MIT — see [LICENSE](LICENSE).
