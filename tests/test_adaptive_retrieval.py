"""Real (unmocked) tests for adaptive-k demonstration selection.

select_k has no dependency on the heavy libraries, so these run the actual
algorithm rather than a stand-in for it.
"""

import unittest

from qlora_icl.adaptive_retrieval import AdaptiveKConfig, select_indices, select_k


class TestSelectK(unittest.TestCase):

    def test_clustered_scores_keep_many(self):
        """All candidates close to the best match -> spend the full budget."""
        scores = [0.91, 0.90, 0.89, 0.88, 0.87]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.85)

        self.assertEqual(select_k(scores, cfg), 5)

    def test_steep_dropoff_keeps_few(self):
        """One clearly best match, the rest irrelevant -> spend little."""
        scores = [0.95, 0.40, 0.38, 0.35, 0.30]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.85)

        self.assertEqual(select_k(scores, cfg), 1)

    def test_gradual_dropoff_keeps_some(self):
        """A middle case: keeps demos until the cliff, not all five."""
        scores = [0.90, 0.87, 0.60, 0.55, 0.50]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.85)

        # 0.87 / 0.90 = 0.9667 >= 0.85 -> kept. 0.60 / 0.90 = 0.667 < 0.85 -> cut.
        self.assertEqual(select_k(scores, cfg), 2)

    def test_min_k_is_a_floor(self):
        """Even a total cliff still returns at least min_k."""
        scores = [0.99, 0.01, 0.01, 0.01, 0.01]
        cfg = AdaptiveKConfig(min_k=2, max_k=5, relative_drop=0.85)

        self.assertEqual(select_k(scores, cfg), 2)

    def test_max_k_is_a_ceiling(self):
        """Identical scores never exceed max_k, however generous the config."""
        scores = [0.9] * 20
        cfg = AdaptiveKConfig(min_k=1, max_k=3, relative_drop=0.5)

        self.assertEqual(select_k(scores, cfg), 3)

    def test_fewer_candidates_than_max_k(self):
        """max_k is a ceiling, not a promise; can't keep more than exist."""
        scores = [0.9, 0.85]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.5)

        self.assertEqual(select_k(scores, cfg), 2)

    def test_min_k_never_exceeds_available_candidates(self):
        """A high min_k against a short candidate list is clamped, not an error."""
        scores = [0.9]
        cfg = AdaptiveKConfig(min_k=5, max_k=5, relative_drop=0.85)

        self.assertEqual(select_k(scores, cfg), 1)

    def test_absolute_floor_rejects_a_uniformly_weak_corpus(self):
        """Scores can be relatively close yet all be bad matches; the absolute
        floor is what catches that, since relative_drop alone would not.
        """
        scores = [0.20, 0.19, 0.18]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.5, absolute_floor=0.3)

        # min_k still guarantees one demo even though it's below the floor -
        # the floor cuts the second and third, not the first.
        self.assertEqual(select_k(scores, cfg), 1)

    def test_empty_scores(self):
        self.assertEqual(select_k([], AdaptiveKConfig()), 0)

    def test_unsorted_input_gives_same_result_as_sorted(self):
        scores = [0.87, 0.90, 0.50, 0.88, 0.55]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.85)

        self.assertEqual(select_k(scores, cfg), select_k(sorted(scores), cfg))

    def test_order_independence_with_duplicates(self):
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.85)
        self.assertEqual(select_k([0.5, 0.5, 0.5], cfg), select_k([0.5, 0.5, 0.5][::-1], cfg))


class TestSelectIndices(unittest.TestCase):

    def test_returns_indices_not_scores(self):
        scores = [0.40, 0.95, 0.38]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.85)

        # index 1 has the best score (0.95); a steep dropoff keeps only it.
        self.assertEqual(select_indices(scores, cfg), [1])

    def test_best_first_ordering(self):
        scores = [0.10, 0.95, 0.90, 0.20]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.85)

        # indices 1 (0.95) and 2 (0.90) both clear the 0.85-of-top threshold.
        self.assertEqual(select_indices(scores, cfg), [1, 2])

    def test_matches_select_k_count(self):
        scores = [0.9, 0.88, 0.5, 0.3, 0.1]
        cfg = AdaptiveKConfig(min_k=1, max_k=5, relative_drop=0.85)

        self.assertEqual(len(select_indices(scores, cfg)), select_k(scores, cfg))

    def test_empty(self):
        self.assertEqual(select_indices([], AdaptiveKConfig()), [])


if __name__ == "__main__":
    unittest.main()
