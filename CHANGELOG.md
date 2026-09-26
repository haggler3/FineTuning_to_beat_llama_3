# Changelog

## Unreleased

### Added
- **Adaptive-k retrieval** (`qlora_icl.adaptive_retrieval`): instead of
  retrieving a fixed number of demonstrations for every query, `select_k()`
  picks a per-query budget from the shape of that query's own retrieval
  scores — spend little on an easy query with one clearly-best match, spend up
  to the ceiling on a query with several close matches. Wired into
  `qlora-eval` as `--adaptive_k` (with `--adaptive_min_k`,
  `--adaptive_relative_drop`, `--adaptive_absolute_floor`); `--icl_top_k`
  doubles as the ceiling. 15 unit tests on the pure selection logic, plus an
  integration test proving two queries with different score shapes really do
  spend a different number of demonstrations end-to-end.
- `--output_json` on `qlora-eval`: writes final metrics (BERTScore, split
  fingerprint, mean demonstrations per query, config) as JSON, for a CI gate
  or dashboard to consume without scraping stdout.
- `scripts/pareto_curve.py`: measures fixed-k vs. adaptive-k cost/quality
  on the same model and the same examples — mean demonstration tokens spent
  against BERTScore F1, not just the ceiling either mode was configured with.
- `Dockerfile` / `.dockerignore`: CUDA 12.4 runtime image for `qlora-train` /
  `qlora-eval` / `qlora-tokenize`. **Not build-verified in CI or this pass** —
  a full CUDA base image pull is heavier than this repo's other checks;
  validate it against your own registry/host before relying on it.

### Changed
- Restructured into an installable package (`src/qlora_icl/`, `pyproject.toml`),
  three console scripts (`qlora-train`, `qlora-eval`, `qlora-tokenize`) in
  place of bare scripts.
- `splits.py`: single `build_splits()` / `split_fingerprint()` shared by
  training and evaluation, so both sides agree on what "held out" means.

### Fixed
- `Evaluate.py` never loaded a LoRA adapter — it always scored the base model
  regardless of `--model_id`. Now loads and merges via `PeftModel`.
- ICL retrieval corpus was built from the split being scored, so a query
  could retrieve itself and its own gold answer. `assert_no_leakage()` now
  refuses to run if the retrieval corpus and evaluation set overlap at all.
- Generation echoed the prompt back into the scored text (no
  `return_full_text=False`); in ICL mode this could make BERTScore compare
  the gold answer to itself.
- `Fine_Tune.py` copied `input_ids` into `labels` wholesale; since
  `pad_token` is `eos_token`, this trained the model to reproduce its prompt
  and then emit padding forever. `build_labels()` masks padding and prompt
  tokens with `-100` — completion-only loss.
- LoRA's target-module fallback had no GPTNeoX entry, so Pythia models
  silently attached to `dense` (MLP) instead of attention.
- `dataset_tokenizer.py` tokenized with a hardcoded `bert-base-uncased`
  (masked-LM) tokenizer regardless of the causal LM being trained.
- Test discovery used `unittest`'s default `test*.py` pattern against files
  named `Test_*.py`; matched by accident on Windows' case-insensitive
  filesystem, matched nothing on Linux, and reported success either way — the
  suite ran zero tests on any Linux CI. Renamed to `test_*.py`; the runner now
  refuses to report success if it discovers zero tests.

**Verification:** the repair above was run end-to-end on a GCP spot NVIDIA T4
(see [`results/gpu_verification_2026-09-20.md`](results/gpu_verification_2026-09-20.md)).
Training converges, the adapter loads and merges, both ICL and non-ICL
evaluation paths run. Measuring what the two eval bugs above were worth: they
inflate BERTScore F1 by +0.064 on identical inputs, comparable to the entire
effect size this class of project usually reports. Adaptive-k retrieval is
covered by unit and integration tests but has not yet had an equivalent GPU
measurement — `scripts/pareto_curve.py` is written for exactly that and is
ready to run.
