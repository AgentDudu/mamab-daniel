"""Algorithm 3 `Update-Blocklists` + the blocklist-aware `Get-Rec` sampling.

Spec §4 (Alg. 3) and §5 C7/C8. Blocking at epoch `j` punishes the agent
contacted at epoch `j-1`, and the block covers epochs `j .. ceil(j^eta)`.
Blocklists are stored as one `until_epoch` per (honest agent, agent) pair
(Remark 6), never as a set of epochs; `d[a] >= j` means `a` is blocked at `j`.
"""

from math import ceil

import numpy as np
from numba import njit


@njit(cache=True)
def ceil_pow(j, p):
    """`ceil(j^p)` with a closeness guard so exactly-representable powers do
    not round up through float error. Used for `A_j` (p = beta, Alg. 1 line 1)
    and for the blocking horizon (p = eta, Alg. 3 line 2)."""
    v = j**p
    r = int(v + 0.5)
    if r >= 1 and abs(v - r) <= 1e-9 * v:
        return r
    return int(ceil(v))


@njit(cache=True)
def last_epoch(T, beta):
    """Largest `j` with `A_j <= T` — the cap on every blocking horizon (C8)."""
    j = 0
    while ceil_pow(j + 1, beta) <= T:
        j += 1
    return j


@njit(cache=True)
def update_blocklist(d_row, j, eta, j_max, prev_h, prev_r, b_j):
    """Alg. 3 lines 1-4, in place on one honest agent's `d` row.

    `eta <= 0` selects the Chawla baseline (Alg. 1 line 7 is a no-op, every
    `P_j` stays empty). `prev_h < 0` means epoch `j-1` had no recommender
    (spec C7), so there is nobody to block.
    """
    if eta > 0.0 and j > 1 and prev_h >= 0 and b_j != prev_r:
        until = ceil_pow(j, eta)
        if until > j_max:  # C8: blocks never outlive the run
            until = j_max
        if until > d_row[prev_h]:
            d_row[prev_h] = until


@njit(cache=True)
def pick_recommender(d_row, j, i, total, z):
    """Alg. 2 line 1: uniform over all agents except `i` and the blocked ones, given a 64-bit draw.

    Returns `-1` when every other agent is blocked — spec C7 (no communication
    that epoch; the caller falls back to `R_j = B_j`).
    """
    cnt = 0
    for a in range(total):
        if a != i and d_row[a] < j:
            cnt += 1
    if cnt == 0:
        return -1
    c = np.int64(z % np.uint64(cnt))
    for a in range(total):
        if a != i and d_row[a] < j:
            if c == 0:
                return a
            c -= 1
    return -1

