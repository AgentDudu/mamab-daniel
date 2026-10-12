"""Algorithm 3 blocking, malicious strategies (Alg. 2 line 5), the oracle (C14),
the C7 no-recommender path, and the no-comm baseline (C15)."""

import numpy as np

from mabamab.baselines import run_trial_algo, run_trial_no_comm
from mabamab.blocking import ceil_pow, last_epoch, pick_recommender, update_blocklist
from mabamab.engine import make_instance, run_trial
from mabamab.malicious import (
    STRATEGY_OMNISCIENT,
    STRATEGY_UNIFORM,
    recommend_omniscient,
    recommend_uniform,
)

ALPHA, BETA = 4.0, 2.0


def test_ceil_pow_and_last_epoch_cap_the_horizon():
    assert [ceil_pow(j, 2.0) for j in (1, 2, 5, 9)] == [1, 4, 25, 81]
    assert ceil_pow(10, 2.0) == 100  # float-error guard: 10**2 == 99.999... must not round up
    assert ceil_pow(3, 2.5) == 16  # 15.59 -> 16
    assert last_epoch(100, 2.0) == 10  # A_10 = 100 <= T < A_11 = 121
    assert last_epoch(99, 2.0) == 9


def test_blocking_punishes_the_previous_recommender_until_ceiling():
    d = np.full(5, -1, dtype=np.int64)
    update_blocklist(d, 5, 2.0, 1000, 1, 3, 4)  # B_5 = 4 != R_4 = 3 -> block agent 1
    assert d[1] == 25  # ceil(5^2)
    assert (d[[0, 2, 3, 4]] == -1).all()  # only the recommender of epoch j-1 is blocked


def test_blocking_needs_mismatch_epoch_gt_1_and_eta():
    d = np.full(4, -1, dtype=np.int64)
    update_blocklist(d, 5, 2.0, 1000, 1, 4, 4)  # B_5 == R_4: nothing to punish
    update_blocklist(d, 1, 2.0, 1000, 2, 3, 0)  # j = 1: no epoch 0 to punish
    update_blocklist(d, 6, 2.0, 1000, -1, 3, 0)  # epoch j-1 had no recommender (C7)
    update_blocklist(d, 6, 0.0, 1000, 2, 3, 0)  # eta = 0 -> Chawla no-op
    assert (d == -1).all()


def test_blocking_is_monotone_and_capped_at_the_last_epoch():
    d = np.full(4, -1, dtype=np.int64)
    update_blocklist(d, 5, 2.0, 1000, 1, 3, 4)
    update_blocklist(d, 4, 2.0, 1000, 1, 3, 4)  # shorter horizon must not shorten it
    assert d[1] == 25
    d = np.full(4, -1, dtype=np.int64)
    update_blocklist(d, 9, 2.0, last_epoch(100, 2.0), 1, 3, 4)  # ceil(9^2)=81 > j_max=10
    assert d[1] == 10


def test_pick_recommender_skips_self_and_blocked_agents():
    d = np.full(4, -1, dtype=np.int64)
    hits = {a: 0 for a in range(4)}
    for z in range(3000):
        a = pick_recommender(d, 1, 1, 4, np.uint64(z))
        hits[a] += 1
    assert hits[1] == 0  # never itself
    assert hits[0] == hits[2] == hits[3] == 1000  # uniform over the rest

    d[2] = 5  # agent 2 blocked through epoch 5 (inclusive)
    assert all(pick_recommender(d, 5, 0, 4, np.uint64(z)) != 2 for z in range(200))
    assert all(pick_recommender(d, 4, 0, 4, np.uint64(z)) != 2 for z in range(200))
    assert any(pick_recommender(d, 6, 0, 4, np.uint64(z)) == 2 for z in range(200))

    d[:] = 5
    assert pick_recommender(d, 5, 0, 4, np.uint64(0)) == -1  # C7: nobody left


def test_malicious_uniform_recommendation_covers_all_arms():
    state = np.uint64(12345)
    seen = set()
    for _ in range(2000):
        state, arm = recommend_uniform(state, 6)
        # numba on Windows boxes uint64 > 2**63 as a negative Python int; re-box or
        # the next call specialises on int64 and the shifts go arithmetic.
        state = np.uint64(state)
        assert 0 <= arm < 6
        seen.add(int(arm))
    assert seen == set(range(6))


def test_omniscient_names_the_least_played_inactive_suboptimal_arm():
    counts = np.array([5, 7, 1, 9, 3], dtype=np.int64)
    assert recommend_omniscient(counts, np.array([0, 3], np.int32)) == 2
    assert recommend_omniscient(counts, np.array([0, 2], np.int32)) == 4  # arms 1,4 left
    # arm 1 (index 0) is never named, even when least played; ties -> lowest index
    assert recommend_omniscient(np.array([0, 4, 4, 9], np.int64), np.array([0, 3], np.int32)) == 1


def _instance(K=100, n=25, S=4, seed=11):
    return make_instance(K, n, S, np.random.default_rng(seed))


def _mean(name, m, eta, strategy, T, trials, K=100, n=25, S=4, seed=11):
    ck = np.array([T], dtype=np.int64)
    vals = []
    for r in range(trials):
        mu, sticky, active, U, L = make_instance(K, n, S, np.random.default_rng(seed + r))
        vals.append(run_trial_algo(name, mu, sticky, active, U, L, ALPHA, BETA, T, ck,
                                   7000 + 13 * r, m=m, eta=eta, strategy=strategy)[-1].mean())
    return float(np.mean(vals))


def test_oracle_never_asks_a_malicious_agent():
    mu, sticky, active, U, L = _instance(K=25, n=10, S=3)
    diag = np.zeros(2, dtype=np.int64)
    run_trial_algo("oracle", mu, sticky, active, U, L, ALPHA, BETA, 5000,
                   np.array([5000], np.int64), 3, m=5, eta=2.0,
                   strategy=STRATEGY_OMNISCIENT, diag=diag)
    assert diag[1] == 0  # C14: {n+1..n+m} excluded from Get-Rec
    assert diag[0] >= 0


def test_oracle_with_only_malicious_agents_equals_no_malicious_agents():
    """C14 exactly: an oracle run with m malicious agents is bit-identical to the
    same run with m = 0 (the malicious slots are never sampled)."""
    mu, sticky, active, U, L = _instance(K=25, n=10, S=3)
    ck = np.array([2000], np.int64)
    a = run_trial_algo("oracle", mu, sticky, active, U, L, ALPHA, BETA, 2000, ck, 3,
                       m=0, eta=2.0, strategy=STRATEGY_OMNISCIENT)[1]
    b = run_trial_algo("oracle", mu, sticky, active, U, L, ALPHA, BETA, 2000, ck, 3,
                       m=4, eta=2.0, strategy=STRATEGY_OMNISCIENT)[1]
    assert np.array_equal(a, b)


def test_c7_no_recommender_epoch_keeps_the_active_set():
    """n = 2 with eta = 4 blocks the only other agent for most epochs, so C7 fires."""
    mu, sticky, active, U, L = _instance(K=25, n=2, S=10)
    diag = np.zeros(2, dtype=np.int64)
    curve, agent_regret = run_trial_algo("proposed", mu, sticky, active, U, L, ALPHA, BETA,
                                         5000, np.array([5000], np.int64), 3, m=0, eta=4.0,
                                         strategy=STRATEGY_UNIFORM, diag=diag)
    assert diag[0] > 0
    assert np.isfinite(curve).all() and (agent_regret >= 0).all()


def test_gossip_row_matches_the_phase_3_kernel():
    """eta = 0 (Alg. 3 no-op) with no malicious agents == the Phase 3 gossip kernel."""
    mu, sticky, active, U, L = _instance(K=25, n=10, S=3)
    ck = np.array([3000], np.int64)
    a = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 3000, ck, 17)
    b = run_trial_algo("gossip", mu, sticky, active, U, L, ALPHA, BETA, 3000, ck, 17, m=0)
    assert np.array_equal(a[1], b[1])


def test_blocking_lowers_regret_under_omniscient_attacks():
    T, trials = 20000, 5
    g = _mean("gossip", 10, 0.0, STRATEGY_OMNISCIENT, T, trials)
    p = _mean("proposed", 10, 2.0, STRATEGY_OMNISCIENT, T, trials)
    o = _mean("oracle", 10, 2.0, STRATEGY_OMNISCIENT, T, trials)
    assert p < 0.8 * g, (p, g)
    assert o < p, (o, p)


def test_theorem1_smoke_gossip_is_not_worse_than_no_comm():
    """Plan Phase 4: m = 1, uniform, K = 25, T = 1e4 -> gossip/no-comm <= ~1.3.

    The measured ratio is ~0.43, not ~1: C15 gives the no-comm baseline all K arms
    while Algorithm 1 agents keep sticky + {U,L} sets, so communication also removes
    exploration cost here. Theorem 1 only bounds gossip from above.
    """
    T, trials, m = 10000, 15, 1
    nc = _mean("no_comm", m, 0.0, STRATEGY_UNIFORM, T, trials, K=25, n=25, S=1)
    g = _mean("gossip", m, 0.0, STRATEGY_UNIFORM, T, trials, K=25, n=25, S=1)
    assert 0.2 * nc <= g <= 1.3 * nc, (g, nc)


def test_no_comm_regret_is_sublinear_and_paired():
    mu, sticky, active, U, L = _instance(K=50, n=10, S=5)
    small = run_trial_no_comm(mu, ALPHA, 10, 10000, np.array([10000], np.int64), 5)[1].mean()
    big = run_trial_no_comm(mu, ALPHA, 10, 40000, np.array([40000], np.int64), 5)[1].mean()
    assert 0 < small < big < 2 * small  # ~log T growth
    assert np.array_equal(small, run_trial_no_comm(mu, ALPHA, 10, 10000, np.array([10000], np.int64), 5)[1].mean())


def test_no_comm_alg1_is_the_isolated_algorithm_1():
    """C15b: Get-Rec removed -> malicious agents cannot influence the run."""
    mu, sticky, active, U, L = _instance(K=25, n=10, S=3)
    ck = np.array([3000], np.int64)
    a = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 3000, ck, 17, m=0, isolated=True)[1]
    b = run_trial(mu, sticky, active, U, L, ALPHA, BETA, 3000, ck, 17,
                  m=7, strategy=STRATEGY_OMNISCIENT, isolated=True)[1]
    assert np.array_equal(a, b)
    assert a.mean() > run_trial(mu, sticky, active, U, L, ALPHA, BETA, 3000, ck, 17, m=0)[1].mean()
