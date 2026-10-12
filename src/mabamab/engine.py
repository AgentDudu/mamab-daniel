"""Engine core: Algorithm 1 for `n` honest agents, with the Algorithm 2 `Get-Rec`
communication path over honest *and* malicious agents.

One numba kernel covers the four compared algorithms (spec §8):

| mode | how |
|---|---|
| `gossip` (Chawla) | `eta = 0` → Algorithm 1 line 7 is a no-op, every `P_j` stays empty |
| `proposed` | `eta > 1` → Algorithm 3 blocking |
| `oracle` | `eta > 1` plus malicious agents permanently blocklisted (C14) |
| `no_comm` | not here — see `baselines.run_no_comm` (C15) |

Conventions implemented (specs/paper_spec.md): C1 unplayed arms have index +inf,
C2 `A_0 = 0`, C3 lowest-arm-index tie-breaks, C4 synchronous epochs (every agent
computes `B_j` at `t = A_j` before any recommendation is read), C5/C6 (`sticky.py`),
C7 no recommender available ⇒ `R_j = B_j` (counted in `diag[0]`), C8 blocking
horizon capped at the last epoch, C9 malicious agents pull no arms, C10 regret
averaged over honest agents, C14 oracle blocklist, C16 Bernoulli rewards.
"""

from __future__ import annotations

from math import log

import numpy as np
from numba import njit

from .blocking import ceil_pow, last_epoch, pick_recommender, update_blocklist
from .malicious import recommend_omniscient, recommend_uniform
from .rng import next_u64, uniform
from .sticky import in_active, sample_sticky_sets, sample_ul_pair
from .ucb import phase_argmax, ucb_argmax


@njit(cache=True)
def _rebuild_active(sticky_row, active_row, U, L):
    """S_j = sticky set + {U, L}, written in ascending arm order (C3)."""
    a = 0
    for k in range(sticky_row.shape[0]):
        if sticky_row[k] or k == U or k == L:
            active_row[a] = k
            a += 1


@njit(cache=True)
def run_alg1(mu, alpha, beta, eta, m, strategy, oracle, sticky, active, U, L,
             rng_state, T, checkpoints, curve, agent_regret, diag):
    """Simulate `T` pulls for `n` honest agents and `m` malicious recommenders.

    Writes the per-checkpoint mean honest-agent regret into `curve` and the
    per-agent final regret into `agent_regret`. `active`, `U`, `L` are mutated.
    `diag` counts `[no-recommender epochs (C7), malicious recommendations]`.
    """
    n, K = sticky.shape
    total = n + m
    counts = np.zeros((n, K), dtype=np.int64)
    phase = np.zeros((n, K), dtype=np.int64)  # T_k(A_j) - T_k(A_{j-1}), spec C2
    sums = np.zeros((n, K), dtype=np.float64)
    B = np.empty(n, dtype=np.int32)
    regret = np.zeros(n, dtype=np.float64)

    j_max = last_epoch(T, beta)
    d = np.full((n, total), -1, dtype=np.int64)  # d[i, a] >= j  <=>  a in P_j^(i)
    if oracle:  # C14: honest agents know the malicious set a priori
        for i in range(n):
            for a in range(n, total):
                d[i, a] = j_max
    prev_h = np.full(n, -1, dtype=np.int32)  # H_{j-1}^(i), -1 = none (C7)
    prev_r = np.full(n, -1, dtype=np.int32)  # R_{j-1}^(i)

    j = 1
    A_j = 1
    c = 0
    for t in range(1, T + 1):
        lt = log(t)
        for i in range(n):  # Alg. 1 line 4: UCB restricted to the active set
            arm = ucb_argmax(counts[i], sums[i], active[i], alpha, lt)
            rng_state[i], u = uniform(rng_state[i])
            if u < mu[arm]:  # C16
                sums[i, arm] += 1.0
            counts[i, arm] += 1
            phase[i, arm] += 1
            regret[i] += mu[0] - mu[arm]

        if t == A_j:
            for i in range(n):  # line 6, computed for everyone first (C4)
                B[i] = phase_argmax(phase[i], active[i])
            for i in range(n):
                update_blocklist(d[i], j, eta, j_max, prev_h[i], prev_r[i], B[i])  # line 7
                rng_state[i], z = next_u64(rng_state[i])
                h = pick_recommender(d[i], j, i, total, z)  # Alg. 2 line 1
                if h < 0:
                    diag[0] += 1
                    rec = B[i]  # C7: nobody reachable, keep the active set
                elif h < n:
                    rec = B[h]  # Alg. 2 line 3
                else:  # Alg. 2 line 5: a malicious recommender
                    diag[1] += 1
                    if strategy == 1:
                        rec = recommend_omniscient(counts[i], active[i])
                    else:
                        rng_state[i], rec = recommend_uniform(rng_state[i], K)
                prev_h[i] = h
                prev_r[i] = rec
                if not in_active(active[i], rec):  # line 9
                    du = phase[i, U[i]]
                    dl = phase[i, L[i]]
                    if dl > du or (dl == du and L[i] < U[i]):  # line 12, C3
                        U[i] = L[i]
                    L[i] = rec  # line 13
                    _rebuild_active(sticky[i], active[i], U[i], L[i])  # line 14
            for i in range(n):
                for k in range(K):
                    phase[i, k] = 0
            j += 1
            A_j = ceil_pow(j, beta)

        while c < checkpoints.shape[0] and t >= checkpoints[c]:
            total_regret = 0.0
            for i in range(n):
                total_regret += regret[i]
            curve[c] = total_regret / n  # C10
            c += 1

    for i in range(n):
        agent_regret[i] = regret[i]


def agent_seeds(base_seed: int, n: int) -> np.ndarray:
    """One reproducible 64-bit stream per (trial, agent) — spec: SeedSequence seeding."""
    return np.asarray(
        [ss.generate_state(1, dtype=np.uint32)[0] for ss in np.random.SeedSequence(base_seed).spawn(n)],
        dtype=np.uint64,
    )


def synthetic_means(K: int, rng: np.random.Generator) -> np.ndarray:
    """Section 6.1: mu1 = 0.95, mu2 = 0.85, mu3..K ~ U[0, 0.85]."""
    mu = np.empty(K, dtype=np.float64)
    mu[0] = 0.95
    mu[1] = 0.85
    if K > 2:
        mu[2:] = rng.uniform(0.0, 0.85, K - 2)
    return mu


def make_instance(K: int, n: int, S: int, rng: np.random.Generator):
    """Draw one trial instance: arm means, sticky sets, initial U/L and active sets."""
    mu = synthetic_means(K, rng)
    sticky = sample_sticky_sets(K, n, S, rng)
    U = np.empty(n, dtype=np.int32)
    L = np.empty(n, dtype=np.int32)
    active = np.empty((n, S + 2), dtype=np.int32)
    for i in range(n):
        U[i], L[i] = sample_ul_pair(sticky[i], rng)
        _rebuild_active(sticky[i], active[i], U[i], L[i])
    return mu, sticky, active, U, L


def run_trial(mu, sticky, active, U, L, alpha, beta, T, checkpoints, base_seed,
              m=0, eta=0.0, strategy=0, oracle=False, diag=None):
    """Run one Algorithm-1 trial on one instance; the instance arrays are left
    untouched so paired algorithms can share them (spec C11)."""
    n = sticky.shape[0]
    curve = np.zeros(len(checkpoints), dtype=np.float64)
    agent_regret = np.zeros(n, dtype=np.float64)
    if diag is None:
        diag = np.zeros(2, dtype=np.int64)
    run_alg1(
        np.ascontiguousarray(mu, dtype=np.float64),
        float(alpha),
        float(beta),
        float(eta),
        int(m),
        int(strategy),
        bool(oracle),
        np.ascontiguousarray(sticky),
        np.ascontiguousarray(active, dtype=np.int32).copy(),
        np.ascontiguousarray(U, dtype=np.int32).copy(),
        np.ascontiguousarray(L, dtype=np.int32).copy(),
        agent_seeds(base_seed, n),
        int(T),
        np.ascontiguousarray(checkpoints, dtype=np.int64),
        curve,
        agent_regret,
        diag,
    )
    return curve, agent_regret
