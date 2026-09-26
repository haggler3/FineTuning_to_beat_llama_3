"""Deterministic dataset splitting shared by Fine_Tune.py and Evaluate.py.

Both scripts must derive the *same* train/val/test partition from the same raw
dataset, otherwise evaluation scores are measured on examples the model may have
trained on. Keeping the logic here (rather than duplicated per script) is what
makes that guarantee checkable.
"""

import hashlib
import json

from datasets import load_dataset

DEFAULT_SEED = 42
DEFAULT_MAX_SAMPLES = 4000


def build_splits(dataset_path, token=None, seed=DEFAULT_SEED,
                 max_samples=DEFAULT_MAX_SAMPLES):
    """Load a dataset and carve it into an 80/10/10 train/val/test partition.

    Returns (train, val, test). The partition is a pure function of
    (dataset_path, seed, max_samples), so any caller passing the same three
    values gets byte-identical splits.
    """
    full = load_dataset(dataset_path, token=token, split="train")

    if max_samples and len(full) > max_samples:
        full = full.shuffle(seed=seed).select(range(max_samples))

    train_temp = full.train_test_split(test_size=0.2, seed=seed)
    train = train_temp["train"]                                  # 80%
    val_test = train_temp["test"].train_test_split(test_size=0.5, seed=seed)
    val = val_test["train"]                                      # 10%
    test = val_test["test"]                                      # 10%

    return train, val, test


def split_fingerprint(dataset_path, seed, max_samples, split_sizes):
    """Short hash identifying a partition, for logging alongside results.

    Two runs that report the same fingerprint were scored on the same examples.
    """
    payload = json.dumps(
        {
            "dataset": dataset_path,
            "seed": seed,
            "max_samples": max_samples,
            "sizes": split_sizes,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
