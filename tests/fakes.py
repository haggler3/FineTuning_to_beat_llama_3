"""A minimal stand-in for a HuggingFace Dataset.

The suite mocks the heavy libraries, but a bare MagicMock silently accepts any
call and returns another MagicMock, so tests written against one cannot catch a
misuse of the Dataset API. This implements the handful of operations the
pipeline actually uses, with the same semantics, so misuse surfaces as a real
failure instead.
"""


class FakeDataset:

    def __init__(self, rows):
        self.rows = list(rows)

    # --- sizing and columns -------------------------------------------------
    def __len__(self):
        return len(self.rows)

    @property
    def column_names(self):
        return list(self.rows[0].keys()) if self.rows else []

    def __getitem__(self, key):
        # Indexing by column name yields a list; slicing yields a dict of
        # columns. Matching this is the point: the original Evaluate.py assumed
        # a slice gave a list of examples, and iterated column names instead.
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        if isinstance(key, slice):
            chunk = self.rows[key]
            return {col: [row[col] for row in chunk] for col in self.column_names}
        return self.rows[key]

    # --- transformations ----------------------------------------------------
    def map(self, fn, remove_columns=None, batched=False):
        if batched:
            columns = {col: [r[col] for r in self.rows] for col in self.column_names}
            return FakeDataset(_unbatch(fn(columns)))
        return FakeDataset([fn(row) for row in self.rows])

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices])

    def shuffle(self, seed=None):
        return self

    def train_test_split(self, test_size, seed=None):
        cut = int(len(self.rows) * (1 - test_size))
        return {
            'train': FakeDataset(self.rows[:cut]),
            'test': FakeDataset(self.rows[cut:]),
        }

    def push_to_hub(self, *args, **kwargs):
        return True


def _unbatch(columns):
    keys = list(columns.keys())
    n = len(columns[keys[0]]) if keys else 0
    return [{k: columns[k][i] for k in keys} for i in range(n)]


def make_qa_dataset(n, prefix="ex"):
    """An instruction/input/output dataset shaped like ChatDoctor."""
    return FakeDataset([
        {
            "instruction": f"Answer the {prefix} question {i}",
            "input": f"{prefix} question {i}",
            "output": f"{prefix} answer {i}",
        }
        for i in range(n)
    ])


class FakeTokenizer:
    """Whitespace tokenizer with the interface the pipeline relies on."""

    def __init__(self, pad_token="<pad>", eos_token="<eos>"):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.padding_side = "right"

    def __len__(self):
        return 1000

    def __call__(self, text, truncation=False, padding=None, max_length=None,
                 return_tensors=None, return_attention_mask=True,
                 add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        if isinstance(text, list):
            encoded = [
                self(t, truncation=truncation, padding=padding, max_length=max_length,
                     return_offsets_mapping=return_offsets_mapping)
                for t in text
            ]
            out = {
                "input_ids": [e["input_ids"] for e in encoded],
                "attention_mask": [e["attention_mask"] for e in encoded],
            }
            if return_offsets_mapping:
                out["offset_mapping"] = [e["offset_mapping"] for e in encoded]
            return out

        # Character offsets are tracked per whitespace token so that
        # offset-mapping based label masking can be exercised for real.
        ids, offsets, cursor = [], [], 0
        for tok in text.split():
            start = text.index(tok, cursor)
            end = start + len(tok)
            ids.append((hash(tok) % 900) + 10)
            offsets.append((start, end))
            cursor = end

        if truncation and max_length:
            ids = ids[:max_length]
            offsets = offsets[:max_length]

        attention = [1] * len(ids)
        if padding == "max_length" and max_length:
            pad_n = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_n
            attention = attention + [0] * pad_n
            offsets = offsets + [(0, 0)] * pad_n

        out = {"input_ids": ids, "attention_mask": attention}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def save_pretrained(self, path):
        return None
