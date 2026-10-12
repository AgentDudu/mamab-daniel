"""Reference port of the upstream `gosine` gossip simulator.

Source: runtime-BQ1qeL/Synch.py (commit a9eec637191487280ba5228f588aa93f4c02c7eb),
class `Synch`, methods `getarm` (lines 105-117), `update` (119-137) and
`update_delta_T` (139-155). No third-party dependency; numpy only.

State layout is upstream's: `arms[:, i]` holds agent `i`'s active set as
`S` sticky rows followed by the two replaceable rows (upstream calls them
`arms[S]` = U and `arms[S+1]` = L). Per-arm reward means are kept as a running
average and the UCB index of every arm is recomputed after each pull.

Two conventions differ from `mabamab.engine` and are switchable here, because
they are the only source of divergence between the two implementations:

* `tie` — order used when indices tie (including which never-pulled arm goes
  first). `"row"` is upstream: the `arms` row order, i.e. sticky arms first in
  their sampled order, then U, then L. `"index"` is spec C3: lowest arm index.
* `swap_tie` — which of U/L survives an epoch when both were pulled equally
  often. Upstream keeps the current U label (`False`); spec C3 keeps the lower
  arm index (`True`).

`exact_score=True` recomputes the index as `sums[k]/T_k + sqrt(alpha*log(t)/T_k)`
at pull time instead of the running-average form, which is arithmetically
identical to the engine and lets a parity run be exact rather than float-noisy.
"""

from __future__ import annotations

import math

import numpy as np


def arms_from_instance(sticky: np.ndarray, U: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Build the upstream `arms` array from our instance arrays."""
    n = sticky.shape[0]
    sizes = sticky.sum(axis=1)
    if not np.all(sizes == sizes[0]):
        raise ValueError("sticky sets must all have the same size")
    arms = np.empty((int(sizes[0]) + 2, n), dtype=np.int64)
    for i in range(n):
        s = np.flatnonzero(sticky[i])
        arms[: s.size, i] = s
        arms[s.size, i] = U[i]
        arms[s.size + 1, i] = L[i]
    return arms


def _pick(cand: np.ndarray, vals: np.ndarray, tie: str) -> int:
    """Row of `vals` to take: first maximum (`row`) or lowest arm index (`index`)."""
    if tie == "row":
        return int(np.argmax(vals))
    if tie == "index":
        return int(np.lexsort((cand, -vals))[0])
    raise ValueError(f"unknown tie order {tie!r}")


def run_gosine(
    mu: np.ndarray,
    arms: np.ndarray,
    alpha: float,
    beta: float,
    T: int,
    rng: np.random.Generator | None = None,
    draws: tuple[np.ndarray, np.ndarray] | None = None,
    tie: str = "row",
    swap_tie: bool = False,
    exact_score: bool = False,
):
    """Simulate `T` pulls for `n` honest agents, m = 0, no blocklists.

    `draws=(uni, rec)` feeds pre-generated randomness: `uni[t, i]` is agent
    `i`'s reward uniform at pull `t` and `rec[t, i]` is the recommender agent
    index (already mapped to `!= i`) used at time `t`. Feeding the engine's own
    stream makes the two runs directly comparable. Otherwise `rng` is used,
    which consumes one uniform per pull and one integer per (agent, epoch).

    Returns `(agent_regret, pulls, arms)` with `pulls` shaped (n, K).
    """
    if rng is None and draws is None:
        raise ValueError("pass rng= or draws=")
    n = arms.shape[1]
    K = mu.size
    arms = np.array(arms, dtype=np.int64, copy=True)
    repl = arms.shape[0] - 2  # rows `repl` (U) and `repl+1` (L) are replaceable

    pulls = np.zeros((K, n), dtype=np.int64)
    prev = np.zeros((K, n), dtype=np.int64)
    sums = np.zeros((K, n), dtype=np.float64)
    mu_bar = np.zeros((K, n), dtype=np.float64)
    score = np.full((K, n), np.inf, dtype=np.float64)
    regret = np.zeros(n, dtype=np.float64)

    j = 1
    for t in range(1, T + 1):
        for ag in range(n):
            cand = arms[:, ag]
            if exact_score:
                lt = math.log(t)
                vals = np.empty(cand.shape[0], dtype=np.float64)
                for r in range(cand.shape[0]):
                    k = int(cand[r])
                    c = pulls[k, ag]
                    vals[r] = math.inf if c == 0 else sums[k, ag] / c + math.sqrt(alpha * lt / c)
            else:
                vals = score[cand, ag]
            arm = int(cand[_pick(cand, vals, tie)])

            pulls[arm, ag] += 1
            u = draws[0][t, ag] if draws is not None else rng.random()
            hit = u < mu[arm]
            if hit:
                sums[arm, ag] += 1.0
            if not exact_score:  # upstream running average
                c = pulls[arm, ag]
                mu_bar[arm, ag] = ((c - 1) * mu_bar[arm, ag] + (1.0 if hit else 0.0)) / c
            regret[ag] += mu[0] - mu[arm]

            if not exact_score:
                with np.errstate(divide="ignore"):
                    score[:, ag] = mu_bar[:, ag] + np.sqrt(alpha * math.log(t + 1) / pulls[:, ag])

        if t == j**beta:  # upstream epoch boundary (A_j = j^beta)
            delta = pulls - prev
            for ag in range(n):
                du = delta[arms[repl, ag], ag]
                dl = delta[arms[repl + 1, ag], ag]
                if dl > du or (swap_tie and dl == du and arms[repl + 1, ag] < arms[repl, ag]):
                    arms[repl, ag], arms[repl + 1, ag] = arms[repl + 1, ag], arms[repl, ag]
                if draws is not None:
                    h = int(draws[1][t, ag])
                else:
                    h = int(rng.integers(n - 1))
                    if h >= ag:
                        h += 1
                cand = arms[:, h]
                rec = int(cand[_pick(cand, delta[cand, h], tie)])
                if not np.any(arms[:, ag] == rec):
                    arms[repl + 1, ag] = rec
            prev = pulls.copy()
            j += 1

    return regret, pulls.T.copy(), arms
