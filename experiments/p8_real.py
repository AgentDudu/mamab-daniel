"""Phase 8 — Figure 2 and the Table 1 real columns (spec §12 + §10).

The Phase 5/6 harness with the arm means swapped for the Phase-7 MovieLens
instance (`results/arm_means.csv`, conventions C17–C20):

* **Figure 2** (spec §6 caption): `n = 25` honest + `m = 15` malicious, `K = 100`,
  `eta = 2`, `beta = 2`, `alpha = 4`, `S = ceil(K/n) = 4`, 50 paired trials,
  regret vs `T` on log-x, four algorithms × two malicious strategies.
* **Table 1 real columns**: `m ∈ {10,15} × K ∈ {75,100} × η ∈ {2,3,4}` ×
  {uniform, omniscient}, paired per-trial `proposed/gossip` ratio at `T = 1e5` (C11).

`K = 75` takes the top-75 arm means of the C20 list — frozen as **C21**.
Seed scheme is the p4/p5/p6 one (`INSTANCE_SEED + r`, `TRIAL_SEED + 13 r`) so
rows are comparable across phases, and proposed/gossip/oracle/no_comm share one
instance and one stream set per trial (C11).

Each finished config writes its raw per-trial CSV, so an interrupted run resumes
by skipping finished configs and configs can run as separate processes
(`--m/--K/--eta` filters write disjoint files). Aggregation folds them into
`results/fig2.csv`, `results/fig2.png`, `results/table1_real.csv`,
`results/fig2_cohort_sensitivity.csv` and the `fig2` / `table1_real` sections of
`results/metrics.json`, and prints the exit checks (plan §4 tolerances, plan §7
stop condition on ≥ 4 mismatching cells).

Usage: python experiments/p8_real.py [--trials 50] [--t 100000]
           [--m 10,15] [--K 75,100] [--eta 2,3,4]
           [--strategy both|uniform|omniscient]
           [--algorithms proposed,gossip,oracle,no_comm]
           [--arm-means results/arm_means.csv] [--aggregate-only]
           [--no-cohort-sensitivity]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from mabamab.baselines import run_trial_algo  # noqa: E402
from mabamab.engine import make_instance_from_means  # noqa: E402
from mabamab.malicious import STRATEGY_OMNISCIENT, STRATEGY_UNIFORM  # noqa: E402
from p5_fig1 import checkpoints, discard_rate, slope_profile, stats  # noqa: E402
from p6_table1_synth import (  # noqa: E402
    MATCH_TOL,
    MISMATCH_STOP,
    SPEC_ROW,
    sticky_size,
    verdict,
)

# spec §6 fixed parameters; Figure 2 caption = m=15, K=100, eta=2 (spec §12.5).
N, ALPHA, BETA, T, TRIALS = 25, 4.0, 2.0, 100000, 50
M_GRID, K_GRID, ETA_GRID = (10, 15), (75, 100), (2, 3, 4)
M_FIG, K_FIG, ETA_FIG = 15, 100, 2.0
BUMP_TARGET = 4e4  # paper's Figure 2 best-arm-discard / regret bump (spec §11.4)
STRATEGIES = {"uniform": STRATEGY_UNIFORM, "omniscient": STRATEGY_OMNISCIENT}
ALGORITHMS = ("proposed", "gossip", "oracle", "no_comm")
# Same seed scheme as p4/p5/p6 (C13).
INSTANCE_SEED, TRIAL_SEED = 20260101, 7000
RESULTS = Path("results")
SPEC = ROOT / "specs" / "paper_spec.md"
_MEANS: dict[tuple[int, str], np.ndarray] = {}


def arm_means(K: int, path: Path = RESULTS / "arm_means.csv") -> np.ndarray:
    """The Phase-7 arm-mean vector truncated to its top `K` entries (C20/C21)."""
    key = (K, str(path))
    if key not in _MEANS:
        col = pd.read_csv(path).arm_mean.to_numpy(dtype=np.float64)
        if col.size < K:
            raise ValueError(f"{path} has {col.size} arms, need {K}")
        _MEANS[key] = np.ascontiguousarray(col[:K])
    return _MEANS[key]


def trial_instance(K: int, r: int, means_path: Path = RESULTS / "arm_means.csv"):
    """Trial `r`'s instance: fixed real means, sticky sets and U/L redrawn (C6)."""
    return make_instance_from_means(arm_means(K, means_path), N, sticky_size(K),
                                    np.random.default_rng(INSTANCE_SEED + r))


def paper_cells():
    """(m, K, eta) -> {strategy: (mean, std)} for the **real** columns of spec §10."""
    out = {}
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        mt = SPEC_ROW.match(line)
        if mt:
            g = mt.groups()  # 3 ints, then 4 (mean, std) pairs: synth×2 then real×2
            out[tuple(int(x) for x in g[:3])] = {
                "uniform": (float(g[7]), float(g[8])),
                "omniscient": (float(g[9]), float(g[10])),
            }
    return out


# --- Table 1 real columns -----------------------------------------------------

def table_raw_path(cfg, strategy: str) -> Path:
    m, K, eta = cfg
    return RESULTS / f"table1_real_raw_m{m}_K{K}_eta{eta}_{strategy}.csv"


def run_table_config(cfg, strategy: str, trials: int, t: int) -> None:
    """One cell: `trials` paired proposed/gossip runs at T = t."""
    m, K, eta = cfg
    cps = np.array([t], dtype=np.int64)
    rows = []
    for r in range(trials):
        inst = trial_instance(K, r)
        seed = TRIAL_SEED + 13 * r
        p, _ = run_trial_algo("proposed", *inst, ALPHA, BETA, t, cps, seed,
                              m=m, eta=float(eta), strategy=STRATEGIES[strategy])
        g, _ = run_trial_algo("gossip", *inst, ALPHA, BETA, t, cps, seed,
                              m=m, strategy=STRATEGIES[strategy])
        rows.append(f"{r},{p[-1]:.6f},{g[-1]:.6f},{p[-1] / g[-1]:.6f}")
    table_raw_path(cfg, strategy).write_text(
        "trial,proposed_regret,gossip_regret,ratio\n" + "\n".join(rows) + "\n",
        encoding="utf-8")


def load_table_raw(cfg, strategy: str, trials: int):
    """(proposed, gossip, ratio) per-trial vectors, or None if not runnable."""
    path = table_raw_path(cfg, strategy)
    if not path.exists():
        return None
    data = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1))
    if data.shape != (trials, 4):
        return None  # partial or a different trial count -> re-run
    return data[:, 1], data[:, 2], data[:, 3]


def aggregate_table(cfgs, strategies, trials: int, t: int, metrics: dict) -> bool:
    """Fold raw per-trial ratios into table1_real.csv + metrics + exit checks."""
    paper = paper_cells()
    rows, cells, missing = [], {}, []
    for cfg in cfgs:
        for strategy in strategies:
            loaded = load_table_raw(cfg, strategy, trials)
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
    (RESULTS / "table1_real.csv").write_text(
        "m,K,eta,S,strategy,trials,paper_mean,paper_std,ours_mean,ours_std,"
        "proposed_mean_regret,gossip_mean_regret,unpaired_ratio,delta,verdict\n"
        + "".join(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n"
                  for r in rows), encoding="utf-8")

    tally = {v: sum(1 for c in cells.values() if c["verdict"] == v)
             for v in ("match", "within-noise", "mismatch")}
    metrics["table1_real"] = {
        "n": N, "alpha": ALPHA, "beta": BETA, "T": t, "trials": trials,
        "instance_seed": INSTANCE_SEED, "arm_means": str(RESULTS / "arm_means.csv"),
        "K75_convention": "top-75 arm means of the C20 list (C21)",
        "grid": {"m": list(M_GRID), "K": list(K_GRID), "eta": list(ETA_GRID)},
        "cells": {f"m{m}_K{K}_eta{e}_{s}": c for (m, K, e, s), c in cells.items()},
        "tally": tally,
        "mean_abs_delta": float(np.mean([abs(c["delta"]) for c in cells.values()])) if cells else None,
        "missing": [list(x) for x in missing],
    }

    print(f"\nTable 1 real (T={t}, {trials} paired trials, n={N} alpha={ALPHA:g} beta={BETA:g})")
    print(f"{'m':>3} {'K':>4} {'eta':>4} {'strategy':>11} {'paper':>16} {'ours':>16} "
          f"{'delta':>7}  verdict")
    for r in rows:
        print(f"{r[0]:>3} {r[1]:>4} {r[2]:>4} {r[4]:>11} {r[6]:>7.3f} ± {r[7]:<6.3f} "
              f"{r[8]:>7.3f} ± {r[9]:<6.3f} {r[13]:>+7.3f}  {r[14]}")
    for cfg in missing:
        print(f"MISSING cell: m={cfg[0]} K={cfg[1]} eta={cfg[2]} {cfg[3]}")
    print(f"\ntally: {tally}  mean|delta| = {metrics['table1_real']['mean_abs_delta']}")
    if tally["mismatch"] >= MISMATCH_STOP:
        print(f"STOP CONDITION (plan §7): {tally['mismatch']} mismatching cells >= {MISMATCH_STOP}")
    return not missing and tally["mismatch"] < MISMATCH_STOP


# --- Figure 2 ----------------------------------------------------------------

def fig_raw_path(strategy: str, algorithm: str) -> Path:
    return RESULTS / f"fig2_raw_{strategy}_{algorithm}.csv"


def run_fig_config(strategy: str, algorithm: str, trials: int, t: int, cps: np.ndarray) -> None:
    """One Figure-2 config: `trials` paired curves (instance + streams shared)."""
    regret = np.zeros((trials, cps.size))
    dropped = np.full((trials, cps.size), np.nan)
    for r in range(trials):
        inst = trial_instance(K_FIG, r)
        d = np.zeros(cps.size, dtype=np.int64)
        curve, _ = run_trial_algo(
            algorithm, *inst, ALPHA, BETA, t, cps, TRIAL_SEED + 13 * r,
            m=M_FIG, eta=ETA_FIG, strategy=STRATEGIES[strategy], dropped=d,
        )
        regret[r] = curve
        if algorithm != "no_comm":  # no active set in C15a, so no discard to count
            dropped[r] = d
    rows = [f"{strategy},{algorithm},{r},{cps[c]},{regret[r, c]:.6f},{dropped[r, c]:.1f}"
            for r in range(trials) for c in range(cps.size)]
    fig_raw_path(strategy, algorithm).write_text(
        "strategy,algorithm,trial,T,mean_regret,best_arm_discards\n" + "\n".join(rows) + "\n",
        encoding="utf-8")


def load_fig_raw(strategy: str, algorithm: str, trials: int, cps: np.ndarray):
    """(regret, dropped) trial × checkpoint matrices, or None if not runnable."""
    path = fig_raw_path(strategy, algorithm)
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.shape != (trials * cps.size, 6):
        return None
    regret = data[:, 4].reshape(trials, cps.size)
    dropped = data[:, 5].reshape(trials, cps.size)
    if not np.array_equal(data[:, 3].astype(np.int64).reshape(trials, cps.size)[0], cps):
        return None
    return regret, dropped


def plot_fig2(curves, drops, cps, trials: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, strategy in enumerate(STRATEGIES):
        for row, (kind, ylabel) in enumerate((("regret", "mean honest-agent regret"),
                                              ("dropped", "arm-1 discards per agent\nper epoch"))):
            ax = axes[row][col]
            for algorithm in ALGORITHMS:
                if kind == "regret":
                    m = curves.get((strategy, algorithm))
                    if m is None:
                        continue
                    mean, sd = m.mean(axis=0), m.std(axis=0, ddof=1)
                    ax.plot(cps, mean, label=algorithm)
                    ax.fill_between(cps, mean - sd, mean + sd, alpha=0.2)
                else:
                    d = drops.get((strategy, algorithm))
                    if d is None or not np.isfinite(d).all():
                        continue
                    ax.plot(cps, discard_rate(d.mean(axis=0), cps), label=algorithm)
            ax.set_xscale("log")
            ax.set_title(f"{strategy} attacks — {['regret', 'best-arm discard'][row]}")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("T")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Figure 2 replication (MovieLens {N} honest + {M_FIG} malicious, K={K_FIG}, "
                 f"eta={ETA_FIG:g}, beta={BETA:g}, alpha={ALPHA:g}, {trials} paired trials)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2.png", dpi=140)


def aggregate_fig(trials: int, t: int, cps: np.ndarray, metrics: dict) -> bool:
    """Fold raw per-trial curves into fig2.csv + fig2.png + metrics + exit checks."""
    curves, drops, rows = {}, {}, []
    for strategy in STRATEGIES:
        for algorithm in ALGORITHMS:
            loaded = load_fig_raw(strategy, algorithm, trials, cps)
            if loaded is None:
                print(f"MISSING raw curves for {strategy}/{algorithm}")
                continue
            regret, dropped = loaded
            curves[(strategy, algorithm)] = regret
            drops[(strategy, algorithm)] = dropped
            for c, T in enumerate(cps):
                mean, se = stats(regret[:, c])
                dmean, dse = stats(dropped[:, c]) if np.isfinite(dropped[:, c]).all() else (float("nan"),) * 2
                rows.append([strategy, algorithm, int(T), mean, se, float(regret[:, c].std(ddof=1)),
                             dmean, dse, trials])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fig2.csv").write_text(
        "strategy,algorithm,T,mean_regret,se_across_trials,std_across_trials,"
        "best_arm_discards_cumulative,discard_se,trials\n"
        + "".join(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n" for r in rows),
        encoding="utf-8")
    plot_fig2(curves, drops, cps, trials)

    checks, ratios = {}, {}
    for strategy in STRATEGIES:
        g, p, o, nc = (curves.get((strategy, a)) for a in ("gossip", "proposed", "oracle", "no_comm"))
        entry = {}
        if g is not None and p is not None:
            ratio = p[:, -1] / g[:, -1]  # paired per-trial ratio at T (C11)
            mean, se = stats(ratio)
            ratios[strategy] = {"mean": mean, "se": se, "std": float(ratio.std(ddof=1)),
                                "mean_over_mean": float(p[:, -1].mean() / g[:, -1].mean())}
            entry["proposed_under_half_gossip"] = {"value": mean, "pass": bool(mean < 0.5)}
        if g is not None and o is not None and nc is not None and p is not None:
            entry["oracle_below_proposed"] = {"oracle": float(o[:, -1].mean()),
                                              "proposed": float(p[:, -1].mean()),
                                              "pass": bool(o[:, -1].mean() < p[:, -1].mean())}
            entry["gossip_vs_no_comm"] = {"gossip": float(g[:, -1].mean()),
                                          "no_comm": float(nc[:, -1].mean()),
                                          "ratio": float(g[:, -1].mean() / nc[:, -1].mean()),
                                          "pass": bool(g[:, -1].mean() > nc[:, -1].mean()),
                                          "gate": False,
                                          "note": "known C15a deviation (spec §11.1), not a gate"}
        if g is not None:
            prof = slope_profile(g.mean(axis=0), cps)
            entry["gossip_bump_T"] = dict(prof, target_T=BUMP_TARGET,
                                          pass_=bool(0.5 * BUMP_TARGET <= prof["slope_peak_T"]
                                                     <= 2 * BUMP_TARGET))
            if strategy == "omniscient":  # the paper sees the S-curve only here (§6.1)
                entry["s_curve"] = {
                    "slope_first_third": prof["slope_first_third"], "slope_peak": prof["slope_peak"],
                    "slope_last_third": prof["slope_last_third"],
                    "pass": bool(prof["slope_peak"] > 2 * max(prof["slope_first_third"], 1e-9)
                                 and prof["slope_last_third"] < prof["slope_peak"])}
            d = drops[(strategy, "gossip")].mean(axis=0)
            rate = discard_rate(d, cps)
            entry["discard_rate"] = {"peak_T": int(cps[1:][int(rate[1:].argmax()) + 1]),
                                     "peak": float(rate[1:].max()),
                                     "cumulative_per_agent_at_T": float(d[-1] / N),
                                     "target_T": BUMP_TARGET, "pass": False, "gate": False,
                                     "note": "Phase-5 deviation re-checked on real data"}
            gp = drops[(strategy, "proposed")].mean(axis=0)
            if gp.size:
                rg, rp = rate[-8:], discard_rate(gp, cps)[-8:]
                entry["blocking_stops_discards"] = {"gossip_rate_last_third": float(rg.mean()),
                                                    "proposed_rate_last_third": float(rp.mean()),
                                                    "pass": bool(rp.mean() < 0.2 * rg.mean())}
        checks[strategy] = entry

    metrics["fig2"] = {
        "K": K_FIG, "n": N, "S": sticky_size(K_FIG), "m": M_FIG, "eta": ETA_FIG,
        "beta": BETA, "alpha": ALPHA, "trials": trials, "T": t,
        "instance_seed": INSTANCE_SEED, "arm_means": str(RESULTS / "arm_means.csv"),
        "checkpoints": cps.tolist(),
        "final_regret": {f"{s}/{a}": {"mean": stats(curves[(s, a)][:, -1])[0],
                                      "se": stats(curves[(s, a)][:, -1])[1]}
                         for s in STRATEGIES for a in ALGORITHMS if (s, a) in curves},
        "best_arm_discards_per_agent": {f"{s}/{a}": float(np.nanmean(drops[(s, a)][:, -1]) / N)
                                        for s in STRATEGIES for a in ALGORITHMS
                                        if (s, a) in drops and np.isfinite(drops[(s, a)]).all()},
        "proposed_over_gossip_paired": ratios,
        "checks": checks,
    }

    print(f"\nFigure 2 exit checks (T={t}, {trials} trials)")
    ok = True
    for strategy, entry in checks.items():
        print(f"  {strategy}:")
        for name, res in entry.items():
            passed = res.pop("pass_", None) if "pass_" in res else res.get("pass")
            if passed is not None and res.get("gate", True):
                ok = ok and passed
            if passed is None:
                print(f"    {name:26s} {res}")
            else:
                print(f"    {name:26s} {res} {'PASS' if passed else 'FAIL'}"
                      + ("" if res.get("gate", True) else " (informational)"))
    print(f"Figure 2: {'PASS' if ok else 'CHECK FAILURES — see above'}")
    return ok


# --- cohort sensitivity (plan §6 risk 1) -------------------------------------

def run_cohort_sensitivity(means_path: Path, trials: int, t: int, metrics: dict) -> None:
    """Figure-2 config re-run on the runner-up cohort, final regrets only."""
    cps = np.array([t], dtype=np.int64)
    rows, finals = [], {}
    for strategy in STRATEGIES:
        for algorithm in ALGORITHMS:
            final = np.array([run_trial_algo(
                algorithm, *trial_instance(K_FIG, r, means_path), ALPHA, BETA, t, cps,
                TRIAL_SEED + 13 * r, m=M_FIG, eta=ETA_FIG, strategy=STRATEGIES[strategy]
            )[0][-1] for r in range(trials)])
            finals[(strategy, algorithm)] = final
            mean, se = stats(final)
            rows.append([strategy, algorithm, mean, se, float(final.std(ddof=1))])
    (RESULTS / "fig2_cohort_sensitivity.csv").write_text(
        "strategy,algorithm,mean_regret,se_across_trials,std_across_trials\n"
        + "".join(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n"
                  for r in rows), encoding="utf-8")
    metrics["fig2"]["cohort_sensitivity"] = {
        "arm_means": str(means_path),
        "proposed_over_gossip_paired": {
            s: {"mean": float((finals[(s, "proposed")] / finals[(s, "gossip")]).mean()),
                "std": float((finals[(s, "proposed")] / finals[(s, "gossip")]).std(ddof=1))}
            for s in STRATEGIES},
        "final_regret": {f"{s}/{a}": stats(finals[(s, a)])[0] for s in STRATEGIES for a in ALGORITHMS},
    }
    print(f"\ncohort sensitivity ({means_path}): "
          + ", ".join(f"{s} ratio {metrics['fig2']['cohort_sensitivity']['proposed_over_gossip_paired'][s]['mean']:.3f}"
                      for s in STRATEGIES))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--t", type=int, default=T)
    ap.add_argument("--m", default=",".join(map(str, M_GRID)))
    ap.add_argument("--K", default=",".join(map(str, K_GRID)))
    ap.add_argument("--eta", default=",".join(map(str, ETA_GRID)))
    ap.add_argument("--strategy", default="both", choices=["both", *STRATEGIES])
    ap.add_argument("--algorithms", default=",".join(ALGORITHMS))
    ap.add_argument("--arm-means", type=Path, default=RESULTS / "arm_means.csv")
    ap.add_argument("--sensitivity-means", type=Path, default=RESULTS / "arm_means_25-M-0.csv")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--no-table", action="store_true", help="Figure 2 configs only")
    ap.add_argument("--no-fig", action="store_true", help="Table 1 cells only")
    ap.add_argument("--no-cohort-sensitivity", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS / "metrics.json")
    args = ap.parse_args()

    cps = checkpoints(args.t)
    strategies = list(STRATEGIES) if args.strategy == "both" else [args.strategy]
    algorithms = [a for a in args.algorithms.split(",") if a]
    cfgs = [(mi, Ki, ei) for mi in (int(x) for x in args.m.split(",") if x)
            for Ki in (int(x) for x in args.K.split(",") if x)
            for ei in (int(x) for x in args.eta.split(",") if x)]

    if not args.aggregate_only:
        warm = make_instance_from_means(arm_means(K_FIG)[:25], 5, 5, np.random.default_rng(1))
        run_trial_algo("proposed", *warm, ALPHA, BETA, 10, np.array([10], np.int64), 0,
                       m=1, eta=ETA_FIG, strategy=STRATEGY_OMNISCIENT)  # jit warmup
        if not args.no_table:
            for cfg in cfgs:
                for strategy in strategies:
                    if load_table_raw(cfg, strategy, args.trials) is not None:
                        print(f"m{cfg[0]} K{cfg[1]} eta{cfg[2]} {strategy}: cached")
                        continue
                    t0 = time.perf_counter()
                    run_table_config(cfg, strategy, args.trials, args.t)
                    print(f"m{cfg[0]} K{cfg[1]} eta{cfg[2]} {strategy}: {args.trials} trials in "
                          f"{time.perf_counter() - t0:.0f}s -> {table_raw_path(cfg, strategy)}")
        if not args.no_fig:
            for strategy in strategies:
                for algorithm in algorithms:
                    if load_fig_raw(strategy, algorithm, args.trials, cps) is not None:
                        print(f"fig2 {strategy}/{algorithm}: cached")
                        continue
                    t0 = time.perf_counter()
                    run_fig_config(strategy, algorithm, args.trials, args.t, cps)
                    print(f"fig2 {strategy}/{algorithm}: {args.trials} trials in "
                          f"{time.perf_counter() - t0:.0f}s -> {fig_raw_path(strategy, algorithm)}")

    metrics = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    ok_fig = aggregate_fig(args.trials, args.t, cps, metrics)
    ok_table = aggregate_table(cfgs, strategies, args.trials, args.t, metrics)
    if not args.aggregate_only and not args.no_fig and not args.no_cohort_sensitivity \
            and args.sensitivity_means.exists():
        run_cohort_sensitivity(args.sensitivity_means, args.trials, args.t, metrics)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nPhase 8 real data: {'PASS' if (ok_fig and ok_table) else 'SEE CHECKS ABOVE'}")
    return 0 if (ok_fig and ok_table) else 1


if __name__ == "__main__":
    raise SystemExit(main())
