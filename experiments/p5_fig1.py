"""Phase 5 — Figure 1 (spec §6.1 synthetic setup, §11 qualitative landmarks).

Config (Figure 1 caption): n = 25 honest + m = 10 malicious agents, K = 100 arms,
eta = 2, beta = 2, alpha = 4, S = ceil(K/n) = 4, 50 trials, regret vs T on log-x
up to 1e5, for both malicious strategies (uniform, omniscient) and the four
compared algorithms (spec §8). Trials are paired across algorithms: the same
instance seed and the same per-agent streams (C11/C13).

A config is one (strategy, algorithm) pair. Each config writes its per-trial
curves to `results/fig1_raw_<strategy>_<algorithm>.csv` as soon as it finishes,
so an interrupted run resumes by skipping finished configs. Aggregation then
folds the raw files into `results/fig1.csv` (one row per strategy/algorithm/T),
`results/fig1.png` (regret + best-arm-discard panels) and the `fig1` section of
`results/metrics.json`, and prints the exit checks.

Usage: python experiments/p5_fig1.py [--trials 50] [--t 100000]
                                     [--strategy both|uniform|omniscient]
                                     [--algorithms proposed,gossip,oracle,no_comm]
                                     [--aggregate-only]
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

from mabamab.baselines import run_trial_algo  # noqa: E402
from mabamab.blocking import last_epoch  # noqa: E402
from mabamab.engine import make_instance  # noqa: E402
from mabamab.malicious import STRATEGY_OMNISCIENT, STRATEGY_UNIFORM  # noqa: E402

# Figure 1 caption + §6.1 fixed parameters.
K, N, S, M = 100, 25, 4, 10
ALPHA, BETA, ETA = 4.0, 2.0, 2.0
STRATEGIES = {"uniform": STRATEGY_UNIFORM, "omniscient": STRATEGY_OMNISCIENT}
ALGORITHMS = ("proposed", "gossip", "oracle", "no_comm")
# Same seed scheme as experiments/p4_theorem1.py so rows are comparable.
INSTANCE_SEED, TRIAL_SEED = 20260101, 7000
RESULTS = Path("results")
BUMP_TARGET = 6e4  # paper's Figure 1 regret bump / best-arm discard (spec §11.4)


def checkpoints(t: int) -> np.ndarray:
    """Log-spaced evaluation points 100 .. t (Figure 1's x-axis is log-scaled)."""
    return np.unique(np.round(np.logspace(2, math.log10(t), 25)).astype(np.int64))


def raw_path(strategy: str, algorithm: str) -> Path:
    return RESULTS / f"fig1_raw_{strategy}_{algorithm}.csv"


def run_config(strategy: str, algorithm: str, trials: int, t: int, cps: np.ndarray) -> None:
    """Run one config and write its per-trial curves (paired instance + streams)."""
    regret = np.zeros((trials, cps.size))
    dropped = np.full((trials, cps.size), np.nan)
    for r in range(trials):
        mu, sticky, active, U, L = make_instance(K, N, S, np.random.default_rng(INSTANCE_SEED + r))
        d = np.zeros(cps.size, dtype=np.int64)
        curve, _ = run_trial_algo(
            algorithm, mu, sticky, active, U, L, ALPHA, BETA, t, cps, TRIAL_SEED + 13 * r,
            m=M, eta=ETA, strategy=STRATEGIES[strategy], dropped=d,
        )
        regret[r] = curve
        if algorithm != "no_comm":  # no active set in C15a, so no discard to count
            dropped[r] = d  # cumulative best-arm discard events
    rows = [f"{strategy},{algorithm},{r},{cps[c]},{regret[r, c]:.6f},{dropped[r, c]:.1f}"
            for r in range(trials) for c in range(cps.size)]
    raw_path(strategy, algorithm).write_text(
        "strategy,algorithm,trial,T,mean_regret,best_arm_discards\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def load_raw(strategy: str, algorithm: str, trials: int, cps: np.ndarray):
    """Return (regret, dropped) trial x checkpoint matrices, or None if not runnable."""
    path = raw_path(strategy, algorithm)
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.shape != (trials * cps.size, 6):
        return None  # partial or different grid -> re-run
    regret = data[:, 4].reshape(trials, cps.size)
    dropped = data[:, 5].reshape(trials, cps.size)
    if not np.array_equal(data[:, 3].astype(np.int64).reshape(trials, cps.size)[0], cps):
        return None
    return regret, dropped


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    se = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0
    return float(v.mean()), float(se)


def slope_profile(curve, cps):
    """Local regret slope d R / d ln T between checkpoints (S-curve detector)."""
    x = np.log(cps, dtype=float)
    s = np.diff(curve) / np.diff(x)
    third = max(1, s.size // 3)
    return {
        "slope_first_third": float(s[:third].mean()),
        "slope_last_third": float(s[-third:].mean()),
        "slope_peak": float(s.max()),
        "slope_peak_T": int(cps[1:][int(s.argmax())]),
        "slope_min": float(s.min()),
    }


def discard_rate(cum, cps):
    """Best-arm discards per honest agent per epoch between checkpoints.

    `cum[c]` is the mean cumulative event count, so the difference over a
    checkpoint window divided by the epochs in that window is the rate.
    """
    epochs = np.array([last_epoch(int(T), BETA) for T in cps], dtype=np.float64)
    events, epochs_in = np.diff(cum, prepend=0.0), np.diff(epochs, prepend=0.0)
    return np.where(epochs_in > 0, events / (N * np.maximum(epochs_in, 1.0)), 0.0)


def aggregate(trials: int, t: int, cps: np.ndarray, out: Path):
    """Fold raw per-trial curves into fig1.csv + fig1.png + metrics + exit checks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves, drops, rows = {}, {}, []
    for strategy in STRATEGIES:
        for algorithm in ALGORITHMS:
            loaded = load_raw(strategy, algorithm, trials, cps)
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
    (RESULTS / "fig1.csv").write_text(
        "strategy,algorithm,T,mean_regret,se_across_trials,std_across_trials,"
        "best_arm_discards_cumulative,discard_se,trials\n"
        + "".join(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n" for r in rows),
        encoding="utf-8",
    )

    # ---- exit checks ---------------------------------------------------------
    checks, ratios = {}, {}
    for strategy in STRATEGIES:
        g = curves.get((strategy, "gossip"))
        p = curves.get((strategy, "proposed"))
        o = curves.get((strategy, "oracle"))
        nc = curves.get((strategy, "no_comm"))
        entry = {}
        if g is not None and p is not None:
            ratio = p[:, -1] / g[:, -1]  # paired per-trial ratio at T (C11)
            mean, se = stats(ratio)
            ratios[strategy] = {"mean": mean, "se": se, "std": float(ratio.std(ddof=1)),
                                "mean_over_mean": float(p[:, -1].mean() / g[:, -1].mean())}
            entry["proposed_under_half_gossip"] = {"value": mean, "pass": bool(mean < 0.5)}
        if g is not None and o is not None and nc is not None:
            entry["oracle_below_proposed"] = {"oracle": float(o[:, -1].mean()),
                                              "proposed": float(p[:, -1].mean()),
                                              "pass": bool(o[:, -1].mean() < p[:, -1].mean())}
            entry["gossip_vs_no_comm"] = {"gossip": float(g[:, -1].mean()),
                                          "no_comm": float(nc[:, -1].mean()),
                                          "ratio": float(g[:, -1].mean() / nc[:, -1].mean()),
                                          "pass": bool(g[:, -1].mean() > nc[:, -1].mean()),
                                          "gate": False,
                                          "note": "known C15a deviation (spec §11.1), not a "
                                                  "Phase-5 gate"}
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
            peak_T = int(cps[1:][int(rate[1:].argmax()) + 1])
            entry["discard_rate"] = {"peak_T": peak_T, "peak": float(rate[1:].max()),
                                     "cumulative_per_agent_at_T": float(d[-1] / N),
                                     "target_T": BUMP_TARGET, "pass": False, "gate": False,
                                     "note": "deviation: discards run at ~0.07/agent/epoch from "
                                             "T~5e2 on, they are not localised at 6e4"}
            gp = drops[(strategy, "proposed")].mean(axis=0)
            if gp.size:
                rg, rp = rate[-8:], discard_rate(gp, cps)[-8:]
                entry["blocking_stops_discards"] = {"gossip_rate_last_third": float(rg.mean()),
                                                    "proposed_rate_last_third": float(rp.mean()),
                                                    "pass": bool(rp.mean() < 0.2 * rg.mean())}
        checks[strategy] = entry

    # ---- figure --------------------------------------------------------------
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
    fig.suptitle(f"Figure 1 replication: n={N} honest + m={M} malicious, K={K}, "
                 f"eta={ETA:g}, beta={BETA:g}, alpha={ALPHA:g}, {trials} paired trials")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1.png", dpi=140)

    # ---- metrics.json --------------------------------------------------------
    metrics = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    metrics["fig1"] = {
        "K": K, "n": N, "S": S, "m": M, "eta": ETA, "beta": BETA, "alpha": ALPHA,
        "trials": trials, "T": t, "instance_seed": INSTANCE_SEED,
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
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # ---- report --------------------------------------------------------------
    print(f"\nexit checks (T={t}, {trials} trials)")
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
    print(f"Figure 1: {'PASS' if ok else 'CHECK FAILURES — see above'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--t", type=int, default=100000)
    ap.add_argument("--strategy", default="both", choices=["both", *STRATEGIES])
    ap.add_argument("--algorithms", default=",".join(ALGORITHMS))
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS / "metrics.json")
    args = ap.parse_args()

    cps = checkpoints(args.t)
    strategies = list(STRATEGIES) if args.strategy == "both" else [args.strategy]
    algorithms = [a for a in args.algorithms.split(",") if a]

    if not args.aggregate_only:
        mu, sticky, active, U, L = make_instance(25, 5, 5, np.random.default_rng(1))
        run_trial_algo("proposed", mu, sticky, active, U, L, ALPHA, BETA, 10,
                       np.array([10], np.int64), 0, m=1, eta=ETA,
                       strategy=STRATEGY_OMNISCIENT)  # jit warmup
        for strategy in strategies:
            for algorithm in algorithms:
                if load_raw(strategy, algorithm, args.trials, cps) is not None:
                    print(f"{strategy}/{algorithm}: cached")
                    continue
                t0 = time.perf_counter()
                run_config(strategy, algorithm, args.trials, args.t, cps)
                print(f"{strategy}/{algorithm}: {args.trials} trials in "
                      f"{time.perf_counter() - t0:.0f}s -> {raw_path(strategy, algorithm)}")

    return 0 if aggregate(args.trials, args.t, cps, args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
