"""The baselines of spec §8: the no-communication run (C15) and the dispatcher
that runs any of the four compared algorithms on one shared instance (C11).

`no_comm` has no pseudocode in the paper; C15 fixes it as `UCB(alpha)` over all
`K` arms, isolated — no sticky set, no active set, no epochs, no communication.
"""

from __future__ import annotations

from math import log

import numpy as np
from numba import njit

from .engine import agent_seeds, run_trial
from .rng import uniform
from .ucb import ucb_argmax

ALGORITHMS = ("proposed", "gossip", "oracle", "no_comm", "no_comm_alg1")


@njit(cache=True)
def run_no_comm(mu, alpha, all_arms, T, checkpoints, rng_state, curve, agent_regret):
    """`n` isolated UCB(alpha) agents over all `K` arms (C15)."""
    n = rng_state.shape[0]
    K = mu.shape[0]
    counts = np.zeros((n, K), dtype=np.int64)
    sums = np.zeros((n, K), dtype=np.float64)
    regret = np.zeros(n, dtype=np.float64)
    c = 0
    for t in range(1, T + 1):
        lt = log(t)
        for i in range(n):
            arm = ucb_argmax(counts[i], sums[i], all_arms, alpha, lt)
            rng_state[i], u = uniform(rng_state[i])
            if u < mu[arm]:  # C16
                sums[i, arm] += 1.0
            counts[i, arm] += 1
            regret[i] += mu[0] - mu[arm]
        while c < checkpoints.shape[0] and t >= checkpoints[c]:
            total = 0.0
            for i in range(n):
                total += regret[i]
            curve[c] = total / n  # C10
            c += 1
    for i in range(n):
        agent_regret[i] = regret[i]


def run_trial_no_comm(mu, alpha, n, T, checkpoints, base_seed):
    """One `no_comm` trial; same instance and same per-agent seed scheme as the
    Algorithm-1 runners, so the comparison is paired (C11)."""
    K = mu.shape[0]
    curve = np.zeros(len(checkpoints), dtype=np.float64)
    agent_regret = np.zeros(n, dtype=np.float64)
    run_no_comm(
        np.ascontiguousarray(mu, dtype=np.float64),
        float(alpha),
        np.arange(K, dtype=np.int32),
        int(T),
        np.ascontiguousarray(checkpoints, dtype=np.int64),
        agent_seeds(base_seed, n),
        curve,
        agent_regret,
    )
    return curve, agent_regret


def run_trial_algo(name, mu, sticky, active, U, L, alpha, beta, T, checkpoints,
                   base_seed, m=0, eta=0.0, strategy=0, diag=None):
    """One trial of one compared algorithm (spec §8) on one shared instance."""
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}, expected one of {ALGORITHMS}")
    if name == "no_comm":
        return run_trial_no_comm(mu, alpha, sticky.shape[0], T, checkpoints, base_seed)
    if name == "no_comm_alg1":  # C15b: Algorithm 1 minus the Get-Rec path
        return run_trial(mu, sticky, active, U, L, alpha, beta, T, checkpoints, base_seed,
                         m=m, eta=0.0, isolated=True, diag=diag)
    if name != "gossip":
        assert eta > 1.0, f"{name} needs the Algorithm 3 parameter eta > 1"
    return run_trial(mu, sticky, active, U, L, alpha, beta, T, checkpoints, base_seed,
                     m=m, eta=0.0 if name == "gossip" else eta, strategy=strategy,
                     oracle=(name == "oracle"), diag=diag)
