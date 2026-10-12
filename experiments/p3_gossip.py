"""Section 6.1 gossip baseline: parity against the upstream reference port, plus
the regret growth table.

Instance seed is frozen (20260101) so the instance is identical across runs;
trial seeds are 1000..1000+trials-1, matching experiments/bench_engine.py.

Parity is checked the strong way first: the port is fed the engine's own
per-agent random stream, so the two runs see identical draws and any semantic
difference shows up as an exact regret difference rather than Monte-Carlo noise.
The port's tie-break conventions are then switched back to upstream's to
measure how much those conventions move the metric.

Writes results/p3_parity.csv, results/p3_tiebreak.csv, results/p3_growth.csv,
results/p3_scale.csv and the `gossip` section of results/metrics.json.

Usage: python experiments/p3_gossip.py [--trials 25] [--t 20000]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gosine_ref import arms_from_instance, run_gosine  # noqa: E402
from mabamab.engine import (  # noqa: E402
    _epoch_start,
    _next_u64,
    _uniform,
    agent_seeds,
    make_instance,
    run_trial,
)

INSTANCE_SEED = 20260101
K, N, S = 100, 25, 4
ALPHA, BETA = 4.0, 2
GROWTH_T = [1250, 2500, 5000, 10000, 20000, 40000, 80000]


def engine_draws(n: int, beta: float, T: int, base_seed: int):
    """Replay the engine's per-agent splitmix64 stream into (uni, rec) arrays.

    Order matches `run_gossip`: one reward uniform per (agent, pull), plus one
    64-bit draw per (agent, epoch) for the recommender choice.
    """
    state = agent_seeds(base_seed, n)
    uni = np.zeros((T + 1, n), dtype=np.float64)
    rec = np.zeros((T + 1, n), dtype=np.int64)
    j, A = 1, 1
    for t in range(1, T + 1):
        epoch = t == A
        for i in range(n):
            state[i], uni[t, i] = _uniform(state[i])
            if epoch:
                state[i], z = _next_u64(state[i])
                h = int(z % (n - 1))
                rec[t, i] = h + 1 if h >= i else h
        if epoch:
            j += 1
            A = _epoch_start(j, beta)
    return uni, rec


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ",".join(header) + "\n" + "".join(",".join(str(x) for x in r) + "\n" for r in rows)
    path.write_text(body, encoding="utf-8")


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    se = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0
    return float(v.mean()), float(se)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--t", type=int, default=20000)
    ap.add_argument("--parity-trials", type=int, default=5)
    ap.add_argument("--scale-t", type=int, default=100000)
    ap.add_argument("--scale-trials", type=int, default=5)
    ap.add_argument("--growth-trials", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    args = ap.parse_args()

    mu, sticky, active, U, L = make_instance(K, N, S, np.random.default_rng(INSTANCE_SEED))
    arms0 = arms_from_instance(sticky, U, L)
    warm = np.array([10], dtype=np.int64)
    run_trial(mu, sticky, active, U, L, ALPHA, BETA, 10, warm, 0)  # jit warmup

    print(f"instance seed {INSTANCE_SEED}  K={K} n={N} S={S} alpha={ALPHA} beta={BETA}")

    # --- parity on identical draws (aligned tie-breaks) -----------------------
    print(f"\nparity, identical draws, T={args.t}, {args.parity_trials} trials")
    ck = np.array([args.t], dtype=np.int64)
    parity_rows = []
    for k in range(args.parity_trials):
        uni, rec = engine_draws(N, BETA, args.t, 1000 + k)
        r_ours = run_trial(mu, sticky, active, U, L, ALPHA, BETA, args.t, ck, 1000 + k)[1]
        r_port = run_gosine(
            mu, arms0, ALPHA, BETA, args.t, draws=(uni, rec),
            tie="index", swap_tie=True, exact_score=True,
        )[0]
        d = float(np.max(np.abs(r_ours - r_port)))
        parity_rows.append([k, 1000 + k, r_ours.mean(), r_port.mean(), d])
        print(f"  trial {k}: ours={r_ours.mean():9.2f} gosine={r_port.mean():9.2f} max|diff|={d:.3e}")
    parity_max_diff = max(r[4] for r in parity_rows)
    write_csv(Path("results/p3_parity.csv"),
              ["trial", "seed", "ours_mean_regret", "gosine_mean_regret", "max_abs_agent_diff"],
              parity_rows)

    # --- how much do the upstream tie-break conventions move the metric? -----
    print(f"\ntie-break sensitivity, independent streams, T={args.t}, {args.trials} trials")
    ours = [
        run_trial(mu, sticky, active, U, L, ALPHA, BETA, args.t, ck, 1000 + k)[1].mean()
        for k in range(args.trials)
    ]
    variants = {"upstream": ("row", False), "spec C3": ("index", True)}
    sens = {}
    ours_mean, ours_se = stats(ours)
    sens_rows = [["ours", "", "", ours_mean, ours_se, 0.0, 0.0]]
    print(f"  ours      mean={ours_mean:9.2f} se={ours_se:6.2f}")
    for name, (tie, swap) in variants.items():
        t0 = time.perf_counter()
        v = [
            run_gosine(mu, arms0, ALPHA, BETA, args.t, rng=np.random.default_rng(500000 + k),
                       tie=tie, swap_tie=swap)[0].mean()
            for k in range(args.trials)
        ]
        m, se = stats(v)
        den = math.sqrt(se**2 + ours_se**2)
        z = (m - ours_mean) / den if den else float("nan")
        sens[name] = {"tie": tie, "swap_tie": swap, "mean": m, "se": se,
                      "delta_vs_ours": m - ours_mean, "z_vs_ours": z}
        sens_rows.append([name, tie, swap, m, se, m - ours_mean, z])
        print(f"  {name:9s} mean={m:9.2f} se={se:6.2f} delta={m - ours_mean:+8.2f} z={z:+6.2f}"
              f"  [{time.perf_counter() - t0:.0f}s]")
    write_csv(Path("results/p3_tiebreak.csv"),
              ["variant", "tie", "swap_tie", "mean_regret", "se", "delta_vs_ours", "z_vs_ours"],
              sens_rows)

    # --- regret growth table (engine) ----------------------------------------
    print(f"\ngrowth table, {args.growth_trials} trials")
    growth_rows = []
    for T in GROWTH_T:
        c = np.array([T], dtype=np.int64)
        v = [run_trial(mu, sticky, active, U, L, ALPHA, BETA, T, c, 1000 + k)[1].mean()
             for k in range(args.growth_trials)]
        m, se = stats(v)
        growth_rows.append([T, m, se])
        print(f"  T={T:6d}  mean={m:9.2f}  se={se:6.2f}")
    write_csv(Path("results/p3_growth.csv"), ["T", "mean_regret", "se"], growth_rows)
    ts = np.log([r[0] for r in growth_rows], dtype=float)
    ys = np.log([r[1] for r in growth_rows], dtype=float)
    loglog_slope = float(np.polyfit(ts, ys, 1)[0])
    # regret ~ a + b*ln(T): b is the empirical constant multiplying log T.
    b, a = (float(x) for x in np.polyfit(np.log([r[0] for r in growth_rows]),
                                         [r[1] for r in growth_rows], 1))
    delta = float(mu[0] - mu[1])
    theo_gossip = (K / N) / delta   # Theorem 2 constant, min{m+K/n,K}/Delta with m=0
    theo_single = K / delta         # single-agent UCB baseline constant
    upper = [r for r in growth_rows if r[0] >= 10000]
    b_hi, a_hi = (float(x) for x in np.polyfit(np.log([r[0] for r in upper]),
                                               [r[1] for r in upper], 1))
    print(f"  log-log slope = {loglog_slope:.3f}   (pure log T -> ~0.1 here)")
    print(f"  regret ~ {a:.1f} + {b:.1f}*ln(T):  b / ((K/n)/Delta) = {b / theo_gossip:.2f}"
          f"   b / (K/Delta) = {b / theo_single:.3f}")
    print(f"  fit on T>=10000 only: regret ~ {a_hi:.1f} + {b_hi:.1f}*ln(T)"
          f"  (b / ((K/n)/Delta) = {b_hi / theo_gossip:.2f})")

    # --- scale check: port at large T ----------------------------------------
    print(f"\nscale check, T={args.scale_t}, {args.scale_trials} trials")
    big = np.array([args.scale_t], dtype=np.int64)
    o = [run_trial(mu, sticky, active, U, L, ALPHA, BETA, args.scale_t, big, 1000 + k)[1].mean()
         for k in range(args.scale_trials)]
    t0 = time.perf_counter()
    p = [run_gosine(mu, arms0, ALPHA, BETA, args.scale_t, rng=np.random.default_rng(500000 + k),
                    tie="index", swap_tie=True)[0].mean() for k in range(args.scale_trials)]
    om, ose = stats(o)
    pm, pse = stats(p)
    den = math.sqrt(ose**2 + pse**2)
    z = (pm - om) / den if den else float("nan")
    print(f"  ours   mean={om:9.2f} se={ose:6.2f}")
    print(f"  gosine mean={pm:9.2f} se={pse:6.2f} delta={pm - om:+8.2f} z={z:+6.2f}"
          f"  [{time.perf_counter() - t0:.0f}s]")
    write_csv(Path("results/p3_scale.csv"),
              ["T", "trials", "ours_mean", "ours_se", "gosine_mean", "gosine_se", "z"],
              [[args.scale_t, args.scale_trials, om, ose, pm, pse, z]])

    if args.out.exists():
        metrics = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        metrics = {}
    metrics["gossip"] = {
        "instance_seed": INSTANCE_SEED,
        "K": K, "n": N, "S": S, "alpha": ALPHA, "beta": BETA,
        "reference": "runtime-BQ1qeL/Synch.py @ a9eec637191487280ba5228f588aa93f4c02c7eb",
        "parity": {
            "T": args.t, "trials": args.parity_trials, "identical_draws": True,
            "tie_break": "index", "swap_tie": True, "max_abs_agent_regret_diff": parity_max_diff,
            "ours_mean": float(np.mean([r[2] for r in parity_rows])),
            "port_mean": float(np.mean([r[3] for r in parity_rows])),
        },
        "tie_break_sensitivity": {"T": args.t, "trials": args.trials, "ours_mean": ours_mean,
                                  "ours_se": ours_se, "variants": sens},
        "growth": {"trials": args.growth_trials, "points": growth_rows,
                   "log_log_slope": loglog_slope, "log_fit": {"intercept": a, "slope_per_efold": b,
                                                              "T_ge_10000": {"intercept": a_hi,
                                                                             "slope_per_efold": b_hi},
                                                              "delta": delta,
                                                              "theory_gossip_const": theo_gossip,
                                                              "theory_single_const": theo_single,
                                                              "b_over_theory_gossip": b / theo_gossip,
                                                              "b_over_theory_single": b / theo_single}},
        "scale": {"T": args.scale_t, "trials": args.scale_trials, "ours_mean": om, "ours_se": ose,
                  "port_mean": pm, "port_se": pse, "z": z},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
