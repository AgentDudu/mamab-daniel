"""Engine-core unit tests: UCB rules, epochs, sticky/active sets, determinism, throughput gate."""

import time

import numpy as np
import pytest

from mabamab.blocking import ceil_pow
from mabamab.engine import (
    _rebuild_active,
    agent_seeds,
    make_instance,
    run_alg1,
    run_trial,
)
from mabamab.sticky import in_active, sample_sticky_sets, sample_ul_pair
from mabamab.ucb import phase_argmax, ucb_argmax

ALPHA, BETA = 4.0, 2.0


def test_ceil_pow_is_ceil_j_p():
    assert [ceil_pow(j, 2.0) for j in range(1, 6)] == [1, 4, 9, 16, 25]
    # beta = 1.5: 2.83 -> 3, 5.20 -> 6, 4**1.5 = 8 exactly (float-error guard), 11.62 -> 12
    assert [ceil_pow(j, 1.5) for j in range(1, 6)] == [1, 3, 6, 8, 12]


def test_ucb_pulls_unplayed_arms_first_in_arm_order():
    counts = np.zeros(6, dtype=np.int64)
    sums = np.zeros(6)
    active = np.array([1, 3, 5], dtype=np.int32)
    assert ucb_argmax(counts, sums, active, ALPHA, 1.0) == 1
    counts[1] = 1
    assert ucb_argmax(counts, sums, active, ALPHA, 4.0) == 3
    counts[3] = 1
    assert ucb_argmax(counts, sums, active, ALPHA, 9.0) == 5


def test_ucb_ties_go_to_lowest_arm_index():
    counts = np.full(6, 10, dtype=np.int64)
    sums = np.full(6, 5.0)
    active = np.array([2, 4], dtype=np.int32)
    assert ucb_argmax(counts, sums, active, ALPHA, 100.0) == 2


def test_ucb_prefers_higher_index_when_counts_equal():
    counts = np.full(6, 100, dtype=np.int64)
    sums = np.zeros(6)
    sums[4] = 90.0
    active = np.array([2, 4], dtype=np.int32)
    assert ucb_argmax(counts, sums, active, ALPHA, 100.0) == 4


def test_phase_argmax_most_played_then_lowest_index():
    phase = np.zeros(6, dtype=np.int64)
    phase[4] = 3
    phase[2] = 3
    assert phase_argmax(phase, np.array([2, 4], dtype=np.int32)) == 2
    phase[4] = 4
    assert phase_argmax(phase, np.array([2, 4], dtype=np.int32)) == 4


def test_sticky_sets_size_and_union_assumption():
    sticky = sample_sticky_sets(100, 25, 4, np.random.default_rng(7))
    assert sticky.shape == (25, 100)
    assert (sticky.sum(axis=1) == 4).all()
    assert sticky[:, 0].any()  # C6: arm 1 in at least one sticky set
    again = sample_sticky_sets(100, 25, 4, np.random.default_rng(7))
    assert (again == sticky).all()


def test_ul_pair_is_distinct_and_non_sticky():
    rng = np.random.default_rng(3)
    sticky = sample_sticky_sets(75, 25, 3, rng)
    for i in range(25):
        u, l = sample_ul_pair(sticky[i], rng)
        assert u != l and not sticky[i, u] and not sticky[i, l]


def test_rebuild_active_is_sorted_sticky_plus_ul():
    sticky = np.zeros(8, dtype=np.bool_)
    sticky[[1, 5]] = True
    row = np.empty(4, dtype=np.int32)
    _rebuild_active(sticky, row, 6, 0)
    assert row.tolist() == [0, 1, 5, 6]
    assert in_active(row, 6) and not in_active(row, 2)


def _instance(K=100, n=25, S=4, seed=11):
    return make_instance(K, n, S, np.random.default_rng(seed))


def test_run_state_keeps_active_set_sticky_plus_ul():
    mu, sticky, active, U, L = _instance()
    n, K = sticky.shape
    active, U, L = active.copy(), U.copy(), L.copy()
    curve = np.zeros(1)
    agent_regret = np.zeros(n)
    run_alg1(mu, ALPHA, BETA, 0.0, 0, 0, False, False, sticky, active, U, L,
             agent_seeds(1, n), 5000, np.array([5000], dtype=np.int64), curve,
             agent_regret, np.zeros(2, dtype=np.int64))
    for i in range(n):
        assert np.all(np.diff(active[i]) > 0)  # ascending, distinct (C3 order)
        want = np.flatnonzero(sticky[i] | (np.arange(K) == U[i]) | (np.arange(K) == L[i]))
        assert want.size == active.shape[1]  # active set stayed sticky + {U, L}
        assert active[i].tolist() == want.tolist()
        assert U[i] != L[i] and not sticky[i, U[i]] and not sticky[i, L[i]]


def test_regret_curve_non_negative_and_nondecreasing():
    mu, sticky, active, U, L = _instance()
    cps = np.array([500, 2000, 8000], dtype=np.int64)
    curve, agent_regret = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 8000, cps, 5)
    assert (curve >= 0).all()
    assert np.all(np.diff(curve) >= 0)
    assert np.allclose(curve[-1], agent_regret.mean())


def test_same_seed_reproduces_and_instance_is_not_mutated():
    mu, sticky, active, U, L = _instance()
    frozen = (mu.copy(), sticky.copy(), active.copy(), U.copy(), L.copy())
    cps = np.array([3000], dtype=np.int64)
    a, _ = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 3000, cps, 42)
    b, _ = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 3000, cps, 42)
    c, _ = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 3000, cps, 43)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert np.array_equal(mu, frozen[0]) and np.array_equal(sticky, frozen[1])
    assert np.array_equal(active, frozen[2]) and np.array_equal(U, frozen[3])


def test_regret_grows_like_log_t():
    """Rough sanity for the gossip path: regret is sublinear and log-like in T."""
    mu, sticky, active, U, L = _instance()
    lo, _ = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 20000, np.array([20000]), 9)
    hi, _ = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 40000, np.array([40000]), 9)
    assert 0 < lo[0] < hi[0]
    assert hi[0] / lo[0] < 1.6  # log(4e4)/log(2e4) ~ 1.03; generous margin


def test_throughput_gate_one_million_pulls_per_second():
    n, K, S, T = 25, 100, 4, 20000
    mu, sticky, active, U, L = _instance(K, n, S)
    cps = np.array([T], dtype=np.int64)
    run_trial(mu, sticky, active, U, L, ALPHA, BETA, T, cps, 0)  # JIT warm-up
    t0 = time.perf_counter()
    run_trial(mu, sticky, active, U, L, ALPHA, BETA, T, cps, 1)
    pulls_per_s = n * T / (time.perf_counter() - t0)
    assert pulls_per_s >= 1e6, f"{pulls_per_s:.3g} pulls/s"
