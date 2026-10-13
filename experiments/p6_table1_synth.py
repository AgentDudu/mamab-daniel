"""Phase 6 — Table 1 synthetic columns (spec §10, Appendix D lines 3149–3260).

Grid: m ∈ {10,15} × K ∈ {75,100} × η ∈ {2,3,4} × {uniform, omniscient}, at
T = 1e5 with n = 25, α = 4, β = 2, S = ceil(K/n) (spec §6). 50 trials per cell
(C12). A cell's metric is the **paired per-trial ratio** `R_proposed / R_gossip`
(C11): both algorithms see the same instance seed and the same per-agent streams
(C13, same seed scheme as experiments/p5_fig1.py so the m=10 K=100 η=2 row is
comparable to results/fig1.csv).

A config is one (m, K, η, strategy) tuple. Each finished config writes its
per-trial rows to `results/table1_raw_m{m}_K{K}_eta{eta}_{strategy}.csv`, so an
interrupted run resumes by skipping finished configs, and configs can be run as
separate processes (`--m/--K/--eta` filters) writing disjoint files.

Aggregation folds the raw files into `results/table1_synth.csv` (one row per
config × strategy, with the paper value, delta and verdict), the `table1_synth`
section of `results/metrics.json`, and prints the exit checks:
match if |ours − paper| ≤ 0.10, within-noise if inside the paper's ±std,
else mismatch (plan §4). Plan §7: ≥ 4 mismatching cells ⇒ stop and report.

Usage: python experiments/p6_table1_synth.py [--trials 50] [--t 100000]
           [--m 10,15] [--K 75,100] [--eta 2,3,4] [--strategy both|uniform|omniscient]
           [--aggregate-only]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mabamab.baselines import run_trial_algo  # noqa: E402
from mabamab.engine import make_instance  # noqa: E402
from mabamab.malicious import STRATEGY_OMNISCIENT, STRATEGY_UNIFORM  # noqa: E402

# spec §6 fixed parameters + §10 grid.
N, ALPHA, BETA, T, TRIALS = 25, 4.0, 2.0, 100000, 50
M_GRID, K_GRID, ETA_GRID = (10, 15), (75, 100), (2, 3, 4)
STRATEGIES = {"uniform": STRATEGY_UNIFORM, "omniscient": STRATEGY_OMNISCIENT}
# Same seed scheme as p4/p5 (C13: one seed list reused for every row).
INSTANCE_SEED, TRIAL_SEED = 20260101, 7000
RESULTS = Path("results")
SPEC = Path(__file__).resolve().parents[1] / "specs" / "paper_spec.md"
MATCH_TOL, MISMATCH_STOP = 0.10, 4

# spec §10 row: | m | K | eta | Synth-unif | Synth-omni | Real-unif | Real-omni |
SPEC_ROW = re.compile(
    r"^\| (\d+) \| (\d+) \| (\d+) \| " + r"([\d.]+) ± ([\d.]+) \| " * 3
    + r"([\d.]+) ± ([\d.]+) \|$"
)


def configs(m, K, eta):
    return [(mi, Ki, ei) for mi in m for Ki in K for ei in eta]


def sticky_size(K: int) -> int:
    return -(-K // N)  # S = ceil(K/n), spec §6


def raw_path(cfg, strategy: str) -> Path:
    m, K, eta = cfg
    return RESULTS / f"table1_raw_m{m}_K{K}_eta{eta}_{strategy}.csv"


def run_config(cfg, strategy: str, trials: int, t: int) -> None:
    """One cell: `trials` paired proposed/gossip runs at T = t."""
    m, K, eta = cfg
    S = sticky_size(K)
    cps = np.array([t], dtype=np.int64)
    rows = []
    for r in range(trials):
        mu, sticky, active, U, L = make_instance(K, N, S, np.random.default_rng(INSTANCE_SEED + r))
        seed = TRIAL_SEED + 13 * r
        p, _ = run_trial_algo("proposed", mu, sticky, active, U, L, ALPHA, BETA, t, cps, seed,
                              m=m, eta=float(eta), strategy=STRATEGIES[strategy])
        g, _ = run_trial_algo("gossip", mu, sticky, active, U, L, ALPHA, BETA, t, cps, seed,
                              m=m, strategy=STRATEGIES[strategy])
        rows.append(f"{r},{p[-1]:.6f},{g[-1]:.6f},{p[-1] / g[-1]:.6f}")
    raw_path(cfg, strategy).write_text("trial,proposed_regret,gossip_regret,ratio\n"
                                       + "\n".join(rows) + "\n", encoding="utf-8")


def load_raw(cfg, strategy: str, trials: int):
    """(proposed, gossip, ratio) per-trial vectors, or None if not runnable."""
    path = raw_path(cfg, strategy)
    if not path.exists():
        return None
    data = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1))
    if data.shape != (trials, 4):
        return None  # partial or a different trial count -> re-run
    return data[:, 1], data[:, 2], data[:, 3]


def paper_cells():
    """(m, K, eta) -> {strategy: (mean, std)} for the synthetic columns of spec §10."""
    out = {}
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        mt = SPEC_ROW.match(line)
        if mt:
            g = mt.groups()
            out[tuple(int(x) for x in g[:3])] = {
                "uniform": (float(g[3]), float(g[4])),
                "omniscient": (float(g[5]), float(g[6])),
            }
    return out


def verdict(ours: float, paper: float, paper_std: float) -> str:
    if abs(ours - paper) <= MATCH_TOL:
        return "match"
    return "within-noise" if abs(ours - paper) <= paper_std else "mismatch"


def aggregate(cfgs, strategies, trials: int, t: int, out: Path):
    """Fold raw per-trial ratios into table1_synth.csv + metrics + exit checks."""
    paper = paper_cells()
    rows, cells, missing = [], {}, []
    for cfg in cfgs:
        for strategy in strategies:
            loaded = load_raw(cfg, strategy, trials)
            if loaded is None:
                missing.append(cfg + (strategy,))
                continue
            p, g, ratio = loaded
            pm, ps = paper[cfg][strategy]
            mean, sd = float(ratio.mean()), float(ratio.std(ddof=1))
            delta = mean - pm
            vd = verdict(mean, pm, ps)
            cells[cfg + (strategy,)] = {"mean": mean, "std": sd, "paper": pm,
                                        "paper_std": ps, "delta": delta, "verdict": vd}
            rows.append([cfg[0], cfg[1], cfg[2], sticky_size(cfg[1]), strategy, trials,
                         pm, ps, mean, sd, float(p.mean()), float(g.mean()),
                         float(p.mean() / g.mean()), delta, vd])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "table1_synth.csv").write_text(
        "m,K,eta,S,strategy,trials,paper_mean,paper_std,ours_mean,ours_std,"
        "proposed_mean_regret,gossip_mean_regret,unpaired_ratio,delta,verdict\n"
        + "".join(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n"
                  for r in rows), encoding="utf-8")

    tally = {v: sum(1 for c in cells.values() if c["verdict"] == v)
             for v in ("match", "within-noise", "mismatch")}
    metrics = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    metrics["table1_synth"] = {
        "n": N, "alpha": ALPHA, "beta": BETA, "T": t, "trials": trials,
        "instance_seed": INSTANCE_SEED, "grid": {"m": list(M_GRID), "K": list(K_GRID),
                                                 "eta": list(ETA_GRID)},
        "cells": {f"m{m}_K{K}_eta{e}_{s}": c for (m, K, e, s), c in cells.items()},
        "tally": tally,
        "mean_abs_delta": float(np.mean([abs(c["delta"]) for c in cells.values()])) if cells else None,
        "missing": [list(x) for x in missing],
    }
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"\nTable 1 synthetic (T={t}, {trials} paired trials, n={N} alpha={ALPHA:g} beta={BETA:g})")
    print(f"{'m':>3} {'K':>4} {'eta':>4} {'strategy':>11} {'paper':>16} {'ours':>16} "
          f"{'delta':>7}  verdict")
    for r in rows:
        print(f"{r[0]:>3} {r[1]:>4} {r[2]:>4} {r[4]:>11} {r[6]:>7.3f} ± {r[7]:<6.3f} "
              f"{r[8]:>7.3f} ± {r[9]:<6.3f} {r[13]:>+7.3f}  {r[14]}")
    for cfg in missing:
        print(f"MISSING cell: m={cfg[0]} K={cfg[1]} eta={cfg[2]} {cfg[3]}")
    print(f"\ntally: {tally}  mean|delta| = {metrics['table1_synth']['mean_abs_delta']}")
    ok = not missing and tally["mismatch"] < MISMATCH_STOP
    if tally["mismatch"] >= MISMATCH_STOP:
        print(f"STOP CONDITION (plan §7): {tally['mismatch']} mismatching cells >= {MISMATCH_STOP}")
    print(f"Table 1 synthetic: {'PASS' if ok else 'FAIL — see above'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--t", type=int, default=T)
    ap.add_argument("--m", default=",".join(map(str, M_GRID)))
    ap.add_argument("--K", default=",".join(map(str, K_GRID)))
    ap.add_argument("--eta", default=",".join(map(str, ETA_GRID)))
    ap.add_argument("--strategy", default="both", choices=["both", *STRATEGIES])
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS / "metrics.json")
    args = ap.parse_args()

    cfgs = configs([int(x) for x in args.m.split(",") if x],
                   [int(x) for x in args.K.split(",") if x],
                   [int(x) for x in args.eta.split(",") if x])
    strategies = list(STRATEGIES) if args.strategy == "both" else [args.strategy]

    if not args.aggregate_only:
        for cfg in cfgs:
            for strategy in strategies:
                if load_raw(cfg, strategy, args.trials) is not None:
                    print(f"m{cfg[0]} K{cfg[1]} eta{cfg[2]} {strategy}: cached")
                    continue
                t0 = time.perf_counter()
                run_config(cfg, strategy, args.trials, args.t)
                print(f"m{cfg[0]} K{cfg[1]} eta{cfg[2]} {strategy}: {args.trials} trials in "
                      f"{time.perf_counter() - t0:.0f}s -> {raw_path(cfg, strategy)}")

    return 0 if aggregate(cfgs, strategies, args.trials, args.t, args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
