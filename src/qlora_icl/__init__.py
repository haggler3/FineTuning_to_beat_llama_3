"""QLoRA fine-tuning with demonstration-retrieved in-context learning.

Public surface:

    build_splits        deterministic train/val/test partition shared by the
                        training and evaluation entry points
    split_fingerprint   short hash identifying a partition, logged by both
    build_labels        causal-LM label construction with completion-only loss
    assert_no_leakage   guard that the ICL retrieval corpus and the evaluation
                        split do not overlap
"""

__version__ = "0.2.0"

from qlora_icl.splits import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SEED,
    build_splits,
    split_fingerprint,
)

__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_SEED",
    "build_splits",
    "split_fingerprint",
    "__version__",
]
