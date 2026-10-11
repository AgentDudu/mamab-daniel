"""Engine core: Algorithm 1 for `n` honest agents with the Chawla (no-blocklist)
communication path — Algorithm 2 `Get-Rec` over honest recommenders only.

Blocking (Algorithm 3), malicious agents and the baselines are added on top of
this kernel later; the epoch-update seam is marked below.

Conventions implemented (specs/paper_spec.md): C1 unplayed arms have index +inf,
C2 `A_0 = 0`, C3 lowest-arm-index tie-breaks, C4 synchronous epochs (every agent
computes `B_j` at `t = A_j` before any recommendation is read), C7/C8 not
reachable without blocklists, C9/C15/C14 out of scope here, C10 regret averaged
over honest agents, C16 Bernoulli rewards.
"""

from __future__ import annotations

from math import ceil, log

import numpy as np
from numba import njit

from .sticky import sample_sticky_sets, sample_ul_pair
from .ucb import phase_argmax, ucb_argmax

_U64 = np.uint64
_TWO53_INV = 1.1102230246251565e-16  # 2**-53


@njit(cache=True)
def _next_u64(state):
    """splitmix64: one 64-bit draw, returns (new_state, draw).

    Per-agent streams (seeded from a SeedSequence in `agent_seeds`) keep runs
    reproducible and let a single agent's stream be resumed.
    """
    state += _U64(0x9E3779B97F4A7C15)
    z = state
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return state, z ^ (z >> _U64(31))


@njit(cache=True)
def _uniform(state):
    state, z = _next_u64(state)
    return state, (z >> _U64(11)) * _TWO53_INV


@njit(cache=True)
def _epoch_start(j, beta):
    """A_j = ceil(j^beta) (Alg. 1 line 1), with a closeness guard so that
    exactly-representable powers do not round up through float error."""
    v = j**beta
    r = int(v + 0.5)
    if r >= 1 and abs(v - r) <= 1e-9 * v:
        return r
    return int(ceil(v))


@njit(cache=True)
def _rebuild_active(sticky_row, active_row, U, L):
    """S_j = sticky set + {U, L}, written in ascending arm order (C3)."""
    a = 0
    for k in range(sticky_row.shape[0]):
        if sticky_row[k] or k == U or k == L:
            active_row[a] = k
            a += 1


@njit(cache=True)
def _in_active(active_row, arm):
    for a in range(active_row.shape[0]):
        if active_row[a] == arm:
            return True
    return False


@njit(cache=True)
def run_gossip(mu, alpha, beta, sticky, active, U, L, rng_state, T, checkpoints, curve, agent_regret):
    """Simulate `T` pulls for `n` honest agents; `m = 0`, no blocklists.

    Writes the per-checkpoint mean honest-agent regret into `curve` and the
    per-agent final regret into `agent_regret`. `active`, `U`, `L` are mutated.
    """
    n, K = sticky.shape
    counts = np.zeros((n, K), dtype=np.int64)
    phase = np.zeros((n, K), dtype=np.int64)  # T_k(A_j) - T_k(A_{j-1}), spec C2
    sums = np.zeros((n, K), dtype=np.float64)
    B = np.empty(n, dtype=np.int32)
    regret = np.zeros(n, dtype=np.float64)

    j = 1
    A_j = 1
    c = 0
    for t in range(1, T + 1):
        lt = log(t)
        for i in range(n):  # Alg. 1 line 4: UCB restricted to the active set
            arm = ucb_argmax(counts[i], sums[i], active[i], alpha, lt)
            rng_state[i], u = _uniform(rng_state[i])
            if u < mu[arm]:  # C16
                sums[i, arm] += 1.0
            counts[i, arm] += 1
            phase[i, arm] += 1
            regret[i] += mu[0] - mu[arm]

        if t == A_j:
            for i in range(n):  # line 6, computed for everyone first (C4)
                B[i] = phase_argmax(phase[i], active[i])
            for i in range(n):
                rng_state[i], z = _next_u64(rng_state[i])
                h = np.int64(z % _U64(n - 1))  # Alg. 2 line 1: uniform over agents != i
                if h >= i:
                    h += 1
                rec = B[h]
                if not _in_active(active[i], rec):  # line 9
                    du = phase[i, U[i]]
                    dl = phase[i, L[i]]
                    if dl > du or (dl == du and L[i] < U[i]):  # line 12, C3
                        U[i] = L[i]
                    L[i] = rec  # line 13
                    _rebuild_active(sticky[i], active[i], U[i], L[i])  # line 14
                # Alg. 1 line 7 (Update-Blocklists) is a no-op here: the Chawla
                # baseline keeps every P_j empty.
            for i in range(n):
                for k in range(K):
                    phase[i, k] = 0
            j += 1
            A_j = _epoch_start(j, beta)

        while c < checkpoints.shape[0] and t >= checkpoints[c]:
            total = 0.0
            for i in range(n):
                total += regret[i]
            curve[c] = total / n  # C10
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


def run_trial(mu, sticky, active, U, L, alpha, beta, T, checkpoints, base_seed):
    """Run one trial on one instance; the instance arrays are left untouched so
    paired algorithms can share them (spec C11)."""
    n = sticky.shape[0]
    curve = np.zeros(len(checkpoints), dtype=np.float64)
    agent_regret = np.zeros(n, dtype=np.float64)
    run_gossip(
        np.ascontiguousarray(mu, dtype=np.float64),
        float(alpha),
        float(beta),
        np.ascontiguousarray(sticky),
        np.ascontiguousarray(active, dtype=np.int32).copy(),
        np.ascontiguousarray(U, dtype=np.int32).copy(),
        np.ascontiguousarray(L, dtype=np.int32).copy(),
        agent_seeds(base_seed, n),
        int(T),
        np.ascontiguousarray(checkpoints, dtype=np.int64),
        curve,
        agent_regret,
    )
    return curve, agent_regret
