"""Adaptive-k demonstration selection for Dr.ICL.

The original pipeline retrieves a fixed number of demonstrations for every
query. That is wasteful two ways at once: an easy query pays for k demos it
didn't need, and a query with no good match in the corpus gets k irrelevant
ones anyway, which can crowd out the model's own knowledge rather than help
it.

select_k() replaces the fixed count with a per-query budget decided from the
shape of that query's own similarity scores: keep demonstrations while they
are still close to the best match, stop once the next one falls off a cliff.
Two queries that retrieve the same top score can end up with different k -
one because its next-best match is nearly as good, the other because nothing
else in the corpus is close.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveKConfig:
    """Tunables for select_k(). Defaults are conservative: prefer keeping a
    demonstration over dropping one that might help.
    """

    min_k: int = 1
    max_k: int = 5
    # A demonstration is kept only while its score is still within this
    # fraction of the top score for that query. 0.85 means: cut a demo once
    # it is more than 15% worse (in similarity) than the best match.
    relative_drop: float = 0.85
    # A demonstration below this absolute similarity is never kept, even if it
    # is within relative_drop of the top score - guards the case where a query
    # has no good match anywhere in the corpus, so every score is mediocre and
    # "relatively close to the best of a bad set" is not a reason to keep it.
    absolute_floor: float = 0.0


def select_k(scores, config=AdaptiveKConfig()):
    """Return how many of the top-scored demonstrations to keep.

    `scores` is the similarity score of each retrieved candidate, in any
    order - this function sorts them. Returns an int in
    [0, min(config.max_k, len(scores))].

    Greedy walk down the sorted scores: keep going while the current score is
    >= relative_drop * top_score AND >= absolute_floor, stop at the first
    candidate that fails either test. min_k is then applied as a floor, but
    never above what is actually available (len(scores)) or config.max_k.
    """
    if not scores:
        return 0

    ranked = sorted(scores, reverse=True)
    n = min(config.max_k, len(ranked))
    if n <= 0:
        return 0

    top = ranked[0]
    threshold = config.relative_drop * top

    kept = 0
    for score in ranked[:n]:
        if score < threshold or score < config.absolute_floor:
            break
        kept += 1

    floor = min(config.min_k, n)
    return max(floor, kept)


def select_indices(scores, config=AdaptiveKConfig()):
    """Like select_k, but returns the indices to keep (best-first) instead of
    just the count - what callers actually need to slice a parallel list of
    demonstrations.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    k = select_k(scores, config=config)
    return order[:k]
