"""Sticky sets and the initial active set (Algorithm 1 line 2; spec C5, C6)."""

import numpy as np


def sample_sticky_sets(K: int, n: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform S-subsets of {1..K} per agent (bool mask, arm 1 = index 0).

    Spec C6: the whole collection is resampled until arm 1 lies in the union,
    which is the assumption the experiments run under (1 in some sticky set).
    """
    assert 1 <= S and K - S >= 2, "active set needs two non-sticky arms"
    while True:
        sticky = np.zeros((n, K), dtype=np.bool_)
        for i in range(n):
            sticky[i, rng.choice(K, size=S, replace=False)] = True
        if sticky[:, 0].any():
            return sticky


def sample_ul_pair(sticky_row: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Spec C5: U and L start as a uniform distinct pair of non-sticky arms."""
    u, l = rng.choice(np.flatnonzero(~sticky_row), size=2, replace=False)
    return int(u), int(l)
