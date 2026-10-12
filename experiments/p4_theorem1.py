"""Phase-4 exit checks: Theorem-1 smoke test and the blocking effect at scale.

A) Theorem 1 (m = 1, uniform attacks, K = 25, n = 25, S = 1): regret of the gossip
   baseline vs the no-comm baseline, as T grows.
B) Paper-scale sanity (K = 100, n = 25, S = 4, m = 10, omniscient attacks, eta = 2):
   all four algorithms on paired seeds, plus the C7 / oracle diagnostics.

Usage: python experiments/p4_theorem1.py [--trials 25] [--sanity-trials 10]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mabamab.baselines import ALGORITHMS, run_trial_algo  # noqa: E402
from mabamab.engine import make_instance  # noqa: E402
from mabamab.malicious import STRATEGY_OMNISCIENT, STRATEGY_UNIFORM  # noqa: E402

INSTANCE_SEED = 20260101
ALPHA, BETA, ETA = 4.0, 2.0, 2.0
GROWTH_T = [1250, 2500, 5000, 10000, 20000]


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ",".join(header) + "\n" + "".join(",".join(str(x) for x in r) + "\n" for r in rows)
    path.write_text(body, encoding="utf-8")


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    se = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0
    return float(v.mean()), float(se)


def run(name, K, n, S, T, seed, m, eta, strategy, diag=None):
    """One trial on paired seeds: instance seed INSTANCE_SEED+seed, engine seed 7000+13*seed."""
    mu, sticky, active, U, L = make_instance(K, n, S, np.random.default_rng(INSTANCE_SEED + seed))
    return run_trial_algo(name, mu, sticky, active, U, L, ALPHA, BETA, T,
                          np.array([T], dtype=np.int64), 7000 + 13 * seed,
                          m=m, eta=eta, strategy=strategy, diag=diag)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25, help="Theorem-1 trials per T")
    ap.add_argument("--sanity-t", type=int, default=100000)
    ap.add_argument("--sanity-trials", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    args = ap.parse_args()

    warm = np.array([10], dtype=np.int64)
    mu, sticky, active, U, L = make_instance(25, 5, 5, np.random.default_rng(1))
    run_trial_algo("proposed", mu, sticky, active, U, L, ALPHA, BETA, 10, warm, 0,
                   m=1, eta=ETA, strategy=STRATEGY_OMNISCIENT)  # jit warmup

    # --- A) Theorem 1: gossip vs no-comm, m = 1 uniform attacks, K = 25 -------
    print(f"Theorem 1: K=25 n=25 S=1 m=1 uniform, {args.trials} paired trials")
    growth_rows, per_algo = [], {}
    for T in GROWTH_T:
        row = [T]
        for name in ("no_comm", "gossip"):
            v = [run(name, 25, 25, 1, T, r, 1, 0.0, STRATEGY_UNIFORM).mean()
                 for r in range(args.trials)]
            m_, se = stats(v)
            per_algo.setdefault(name, []).append(m_)
            row += [m_, se]
        growth_rows.append(row)
        print(f"  T={T:6d}  no_comm={row[1]:8.2f} (se {row[2]:5.2f})   "
              f"gossip={row[3]:8.2f} (se {row[4]:5.2f})   ratio={row[3] / row[1]:.3f}")
    write_csv(Path("results/p4_theorem1.csv"),
              ["T", "no_comm_mean", "no_comm_se", "gossip_mean", "gossip_se"], growth_rows)
    slopes = {}
    for name, ys in per_algo.items():
        xs = np.log(GROWTH_T, dtype=float)
        slopes[name] = float(np.polyfit(xs, np.log(ys), 1)[0])
        print(f"  {name}: log-log slope {slopes[name]:.3f} (pure log T -> ~0.1)")
    last = growth_rows[-1]
    ratio_t1 = last[3] / last[1]
    print(f"  gossip/no_comm at T={last[0]}: {ratio_t1:.3f} (plan expected ~1, upper bound 1.3)")

    # --- B) paper scale: does Alg. 3 recover the oracle? ----------------------
    K, n, S, m = 100, 25, 4, 10
    print(f"\nsanity: K={K} n={n} S={S} m={m} omniscient eta={ETA} T={args.sanity_t}, "
          f"{args.sanity_trials} paired trials")
    sanity_rows, means, diags = [], {}, {}
    for name in ALGORITHMS:
        t0 = time.perf_counter()
        v, diag = [], np.zeros(2, dtype=np.int64)
        for r in range(args.sanity_trials):
            v.append(run(name, K, n, S, args.sanity_t, r, m, ETA, STRATEGY_OMNISCIENT, diag).mean())
        m_, se = stats(v)
        means[name], diags[name] = m_, diag.copy()
        sanity_rows.append([name, m_, se, int(diag[0]), int(diag[1])])
        print(f"  {name:8s} mean={m_:9.2f} se={se:6.2f}  c7_epochs={int(diag[0]):7d} "
              f"malicious_rec={int(diag[1]):7d}  [{time.perf_counter() - t0:.0f}s]")
    write_csv(Path("results/p4_sanity.csv"),
              ["algorithm", "mean_regret", "se", "c7_epochs", "malicious_rec_epochs"], sanity_rows)
    r_pg = means["proposed"] / means["gossip"]
    r_og = means["oracle"] / means["gossip"]
    r_gn = means["gossip"] / means["no_comm"]
    print(f"  proposed/gossip = {r_pg:.3f}   oracle/gossip = {r_og:.3f}   gossip/no_comm = {r_gn:.3f}")

    # --- merge into metrics.json ---------------------------------------------
    metrics = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    metrics["blocking"] = {
        "instance_seed": INSTANCE_SEED,
        "alpha": ALPHA,
        "beta": BETA,
        "eta": ETA,
        "theorem1": {
            "K": 25, "n": 25, "S": 1, "m": 1, "strategy": "uniform", "trials": args.trials,
            "growth": [{"T": r[0], "no_comm": r[1], "no_comm_se": r[2],
                        "gossip": r[3], "gossip_se": r[4]} for r in growth_rows],
            "log_log_slope": slopes,
            "gossip_over_no_comm": ratio_t1,
            "note": "ratio < 1 because the C15 no-comm baseline explores all K arms while "
                    "Algorithm 1 agents keep sticky + {U,L} active sets; Theorem 1 only "
                    "bounds gossip regret from above.",
        },
        "sanity": {
            "K": K, "n": n, "S": S, "m": m, "strategy": "omniscient",
            "T": args.sanity_t, "trials": args.sanity_trials,
            "algorithms": {r[0]: {"mean_regret": r[1], "se": r[2],
                                  "c7_epochs": r[3], "malicious_rec_epochs": r[4]}
                           for r in sanity_rows},
            "proposed_over_gossip": r_pg,
            "oracle_over_gossip": r_og,
            "gossip_over_no_comm": r_gn,
        },
    }
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
