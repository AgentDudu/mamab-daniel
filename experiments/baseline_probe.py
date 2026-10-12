"""Which no-communication baseline does the paper mean, and does Theorem 1 hold?

C15 has two readings and the paper gives no pseudocode:
  C15a `no_comm`      — plain UCB(alpha) over all K arms (no sticky/active sets)
  C15b `no_comm_alg1` — Algorithm 1 with the Get-Rec path removed (sticky sets,
                        epochs and active-set updates kept, R_j := B_j)

A) K=25 n=25 S=1 m=1 uniform, T up to 1e6: gossip vs both readings, plus the
   Theorem-1 scale alpha(1-1/sqrt(alpha))^2 * sum_k 1/Delta_k * log T (eq. 3).
B) Paper scale K=100 n=25 S=4 m=10 omniscient eta=2 T=1e5: which reading
   reproduces the Figure-1 ordering (landmark: gossip >> no-comm).

Usage: python experiments/baseline_probe.py [--trials 10] [--big-t 1000000]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mabamab.baselines import run_trial_algo  # noqa: E402
from mabamab.engine import make_instance  # noqa: E402
from mabamab.malicious import STRATEGY_OMNISCIENT, STRATEGY_UNIFORM  # noqa: E402

INSTANCE_SEED = 20260101
ALPHA, BETA = 4.0, 2.0
PROBE_T = [10000, 20000, 50000, 100000, 200000, 500000]
NAMES = ("gossip", "no_comm", "no_comm_alg1")


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ",".join(header) + "\n" + "".join(",".join(str(x) for x in r) + "\n" for r in rows)
    path.write_text(body, encoding="utf-8")


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    se = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0
    return float(v.mean()), se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--big-t", type=int, default=1000000)
    ap.add_argument("--scale-trials", type=int, default=10)
    ap.add_argument("--scale-t", type=int, default=100000)
    ap.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    args = ap.parse_args()

    # --- A) Theorem-1 growth at K=25, m=1 uniform attacks ---------------------
    K, n, S, m = 25, 25, 1, 1
    ts = PROBE_T + [args.big_t]
    print(f"A) K={K} n={n} S={S} m={m} uniform, {args.trials} paired trials")
    rows, curves = [], {name: [] for name in NAMES}
    for T in ts:
        ck = np.array([T], dtype=np.int64)
        row = [T]
        for name in NAMES:
            v = [run_trial_algo(name, *make_instance(K, n, S, np.random.default_rng(INSTANCE_SEED + r)),
                                ALPHA, BETA, T, ck, 7000 + 13 * r, m=m, eta=0.0,
                                strategy=STRATEGY_UNIFORM)[1].mean()
                 for r in range(args.trials)]
            mean, se = stats(v)
            curves[name].append(mean)
            row += [mean, se]
        rows.append(row)
        print(f"  T={T:8d}  gossip={row[1]:9.1f} (se {row[2]:5.1f})  "
              f"no_comm={row[3]:9.1f} (se {row[4]:5.1f})  no_comm_alg1={row[5]:9.1f} (se {row[6]:5.1f})")
    write_csv(Path("results/baseline_probe_growth.csv"),
              ["T", "gossip", "gossip_se", "no_comm", "no_comm_se", "no_comm_alg1", "no_comm_alg1_se"], rows)

    # Theorem-1 scale: eq. 3 constant on the mean instance gaps.
    sums = [float(np.sum(1.0 / (mu[0] - mu[1:])))
            for mu in (make_instance(K, n, S, np.random.default_rng(INSTANCE_SEED + r))[0]
                       for r in range(args.trials))]
    sum_inv_gap = float(np.mean(sums))
    c3 = ALPHA * (1.0 - 1.0 / math.sqrt(ALPHA)) ** 2 * sum_inv_gap
    print(f"  sum_k 1/Delta_k = {sum_inv_gap:.2f}  ->  eq.(3) constant = {c3:.2f} per log T")
    theory = [c3 * math.log(T) for T in ts]
    ratios = {name: [curves[name][i] / theory[i] for i in range(len(ts))] for name in NAMES}
    for name in NAMES:
        print(f"  {name:12s} regret / (eq.3 * log T): " + " ".join(f"{x:.2f}" for x in ratios[name]))
    slopes = {name: float(np.polyfit(np.log(ts), np.log(curves[name]), 1)[0]) for name in NAMES}
    print("  log-log slopes: " + "  ".join(f"{name}={slopes[name]:.3f}" for name in NAMES))
    print("  gossip/no_comm:      " + " ".join(f"{curves['gossip'][i] / curves['no_comm'][i]:.3f}"
                                               for i in range(len(ts))))
    print("  gossip/no_comm_alg1: " + " ".join(f"{curves['gossip'][i] / curves['no_comm_alg1'][i]:.3f}"
                                               for i in range(len(ts))))

    # --- B) paper scale: which reading reproduces Figure 1 -------------------
    K2, n2, S2, m2, eta = 100, 25, 4, 10, 2.0
    print(f"\nB) K={K2} n={n2} S={S2} m={m2} omniscient eta={eta} T={args.scale_t}, "
          f"{args.scale_trials} paired trials")
    ck = np.array([args.scale_t], dtype=np.int64)
    scale_rows, means = [], {}
    for name in NAMES + ("proposed", "oracle"):
        t0 = time.perf_counter()
        v = [run_trial_algo(name, *make_instance(K2, n2, S2, np.random.default_rng(INSTANCE_SEED + r)),
                            ALPHA, BETA, args.scale_t, ck, 7000 + 13 * r, m=m2, eta=eta,
                            strategy=STRATEGY_OMNISCIENT)[1].mean()
             for r in range(args.scale_trials)]
        mean, se = stats(v)
        means[name] = mean
        scale_rows.append([name, mean, se])
        print(f"  {name:12s} mean={mean:9.1f} se={se:6.1f}  [{time.perf_counter() - t0:.0f}s]")
    write_csv(Path("results/baseline_probe_scale.csv"), ["algorithm", "mean_regret", "se"], scale_rows)
    print(f"  gossip/no_comm = {means['gossip'] / means['no_comm']:.3f}   "
          f"gossip/no_comm_alg1 = {means['gossip'] / means['no_comm_alg1']:.3f}   "
          f"proposed/gossip = {means['proposed'] / means['gossip']:.3f}")

    metrics = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    metrics["no_comm_baseline"] = {
        "instance_seed": INSTANCE_SEED, "alpha": ALPHA, "beta": BETA,
        "growth": {"K": K, "n": n, "S": S, "m": m, "strategy": "uniform",
                   "trials": args.trials,
                   "T": ts, "regret": curves,
                   "eq3_constant": c3, "sum_inv_gap": sum_inv_gap,
                   "regret_over_eq3_logT": ratios,
                   "log_log_slope": slopes,
                   "gossip_over_no_comm": [curves["gossip"][i] / curves["no_comm"][i] for i in range(len(ts))],
                   "gossip_over_no_comm_alg1": [curves["gossip"][i] / curves["no_comm_alg1"][i]
                                                for i in range(len(ts))]},
        "scale": {"K": K2, "n": n2, "S": S2, "m": m2, "eta": eta, "strategy": "omniscient",
                  "T": args.scale_t, "trials": args.scale_trials,
                  "regret": means,
                  "gossip_over_no_comm": means["gossip"] / means["no_comm"],
                  "gossip_over_no_comm_alg1": means["gossip"] / means["no_comm_alg1"]},
    }
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
