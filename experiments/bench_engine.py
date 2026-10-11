"""Engine throughput gate: T=1e5 x n=25 x 50 trials = 1.25e8 arm pulls.

Must finish in minutes (>= 1e6 pulls/s), per REPLICATION_PLAN.md. One shared
instance across the 50 trials — this measures the kernel, not instance sampling.

Run: .venv/Scripts/python experiments/bench_engine.py
"""

import json
import time
from pathlib import Path

import numpy as np

from mabamab.engine import make_instance, run_trial

K, N, S, ALPHA, BETA, T, TRIALS = 100, 25, 4, 4.0, 2.0, 100_000, 50
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    mu, sticky, active, U, L = make_instance(K, N, S, np.random.default_rng(20260101))
    cps = np.array([T], dtype=np.int64)
    run_trial(mu, sticky, active, U, L, ALPHA, BETA, T, cps, 0)  # JIT warm-up, not counted

    regrets = []
    t0 = time.perf_counter()
    for trial in range(TRIALS):
        _, agent_regret = run_trial(mu, sticky, active, U, L, ALPHA, BETA, T, cps, trial)
        regrets.append(agent_regret.mean())
    seconds = time.perf_counter() - t0

    pulls = N * T * TRIALS
    row = {
        "K": K, "n": N, "S": S, "alpha": ALPHA, "beta": BETA, "T": T, "trials": TRIALS,
        "pulls": pulls, "seconds": round(seconds, 3), "pulls_per_s": round(pulls / seconds, 1),
        "mean_regret": round(float(np.mean(regrets)), 4),
        "std_regret": round(float(np.std(regrets)), 4),
    }
    out = ROOT / "results" / "bench_engine.csv"
    out.write_text(
        ",".join(row) + "\n" + ",".join(str(v) for v in row.values()) + "\n", encoding="utf-8"
    )
    metrics = ROOT / "results" / "metrics.json"
    data = json.loads(metrics.read_text(encoding="utf-8")) if metrics.exists() else {}
    data["engine_throughput"] = row
    metrics.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
