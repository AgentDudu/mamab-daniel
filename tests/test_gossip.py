"""Gossip-baseline parity tests: our engine vs the ported upstream `gosine` simulator.

The parity claim is about semantics, so the tests feed both implementations the
engine's own random stream: with the tie-break conventions aligned the two runs
must agree exactly, and with upstream's conventions they must not.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from gosine_ref import arms_from_instance, run_gosine  # noqa: E402
from mabamab.engine import make_instance, run_trial  # noqa: E402
from p3_gossip import engine_draws  # noqa: E402

ALPHA, BETA = 4.0, 2
TINY = dict(K=8, n=4, S=2)


def tiny_instance(seed=7):
    return make_instance(TINY["K"], TINY["n"], TINY["S"], np.random.default_rng(seed))


def test_parity_is_exact_with_aligned_tie_breaks():
    mu, sticky, active, U, L = tiny_instance()
    arms = arms_from_instance(sticky, U, L)
    T = 400
    uni, rec = engine_draws(TINY["n"], BETA, T, 1000)
    ours = run_trial(mu, sticky, active, U, L, ALPHA, BETA, T, np.array([T], np.int64), 1000)[1]
    port = run_gosine(mu, arms, ALPHA, BETA, T, draws=(uni, rec),
                      tie="index", swap_tie=True, exact_score=True)[0]
    np.testing.assert_allclose(port, ours, rtol=0, atol=1e-12)


def test_upstream_tie_breaks_diverge_from_spec_c3():
    """Row-order ties (sticky arms first, U label kept) are a different algorithm."""
    mu, sticky, active, U, L = tiny_instance()
    arms = arms_from_instance(sticky, U, L)
    T = 400
    uni, rec = engine_draws(TINY["n"], BETA, T, 1000)
    ours = run_trial(mu, sticky, active, U, L, ALPHA, BETA, T, np.array([T], np.int64), 1000)[1]
    port = run_gosine(mu, arms, ALPHA, BETA, T, draws=(uni, rec), tie="row")[0]
    assert not np.allclose(port, ours, atol=1e-9)


def test_sticky_arms_are_never_dropped():
    mu, sticky, active, U, L = tiny_instance(seed=11)
    arms = arms_from_instance(sticky, U, L)
    _, _, final = run_gosine(mu, arms, ALPHA, BETA, 300, rng=np.random.default_rng(3))
    for i in range(TINY["n"]):
        col = set(int(a) for a in final[:, i])
        assert len(col) == final.shape[0], "active set lost an arm"
        assert set(np.flatnonzero(sticky[i])) <= col


def test_regret_is_sublinear_in_T():
    mu, sticky, active, U, L = tiny_instance(seed=13)
    r_small = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 500, np.array([500], np.int64), 1000)[1].mean()
    r_big = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 4000, np.array([4000], np.int64), 1000)[1].mean()
    assert r_big / r_small < 6  # an 8x longer run, so linear growth would give 8x
