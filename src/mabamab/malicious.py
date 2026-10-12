"""Malicious recommendation strategies (spec §7) — Algorithm 2 line 5.

A malicious agent pulls no arms and contributes no regret (C9); it only answers
`Get-Rec`. The strategy is indexed by the *receiving* honest agent, so each
malicious agent answers each honest agent separately: in the kernel the answer is
drawn from the receiver's own stream, which keeps a trial reproducible.
"""

import numpy as np
from numba import njit

from .rng import uniform
from .sticky import in_active

STRATEGY_UNIFORM = 0
STRATEGY_OMNISCIENT = 1


@njit(cache=True)
def recommend_uniform(state, K):
    """`nu` = uniform over `{1..K}`: may name the best arm or an already-active
    arm, in which case Algorithm 1 line 9 leaves the active set unchanged."""
    state, r = uniform(state)
    arm = np.int32(r * K)
    if arm >= K:  # r == 1.0 cannot happen with a 53-bit mantissa; guard anyway
        arm = np.int32(K - 1)
    return state, arm


@njit(cache=True)
def recommend_omniscient(counts, active):
    """`argmin_{k' in {2..K} minus S_j^(i)} T_{k'}^(i)(A_j)` — the least-played
    inactive suboptimal arm of the *receiving* agent, from its cumulative counts.

    Ties go to the lowest arm index (C3). Empty candidate set cannot happen for
    the experiment configs (`S + 2 < K`); the fallback keeps the return valid.
    """
    best_arm = -1
    best_val = -1
    for k in range(1, counts.shape[0]):  # arm 1 (index 0) is excluded by {2..K}
        if in_active(active, k):
            continue
        c = counts[k]
        if best_arm < 0 or c < best_val:
            best_val = c
            best_arm = k
    return np.int32(best_arm if best_arm >= 0 else 1)
