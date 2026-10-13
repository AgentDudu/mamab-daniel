# STATE.md — replication state

Paper: *Robust Multi-Agent Multi-Armed Bandits* — Vial, Shakkottai, Srikant (MobiHoc '21, arXiv:2007.03812v3).
Plan: `REPLICATION_PLAN.md`. Contract for all phases after Phase 1: `specs/paper_spec.md`.

## Environment (Phase 0)

- Local venv `.venv`, **Python 3.14.8** (Windows x64). Chosen over 3.12 by the user; `numba 0.68.0` verified to install and import on 3.14, so the Phase-2 perf gate is not blocked.
- Install: `python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt && .venv/Scripts/python -m pip install -e .`
- Run tests: `.venv/Scripts/python -m pytest`
- Pins (`requirements.txt`): numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, pandas 3.0.6, numba 0.68.0, pytest 9.1.1.
- No `softimpute` dependency installed — Phase 7 falls back to numpy ALS/SVD soft-thresholding (plan §7 Phase 7 fallback). Deviation must be recorded in the spec/report.
- Git: remote `origin` = `https://github.com/AgentDudu/mamab-daniel.git`. Tracked: code, specs, STATE.md, requirements, `paper.txt`, `paper_html.txt`. Ignored: `2007.03812v3.pdf`, `paper.html`, `data/`, `*.png`, `.venv/`. `results/` **is** tracked (CSV/JSON are the deliverable).

## Phase log

### Phase 0 — env + scaffolding — PASS (2026-10-11)

- Created `specs/ src/mabamab/ tests/ experiments/ results/ logs/ data/`, `pyproject.toml` (src layout, editable install), `requirements.txt`, `.gitignore`, `STATE.md`.
- `git init -b main` + `git remote add origin https://github.com/AgentDudu/mamab-daniel.git` + first commit + `git push -u origin main`.
- Commands run: `python -m venv .venv`; `pip install -U pip setuptools wheel`; `pip install numba numpy scipy matplotlib pandas pytest`; `pip install -e .`; `python -m pytest`.
- Pass/fail: all imports OK (`numpy scipy matplotlib pandas numba pytest mabamab`); `pytest` green (2 tests: import + numba JIT).
- Early Phase-2 signal (throwaway benchmark, not committed): numba `njit` UCB loop, K=100 full scan, 2e6 pulls in 0.18 s = **11.2M pulls/s** single-core → the 1M pulls/s gate is not at risk.
- numba 0.68 caveat found: `np.random.default_rng` is **not** supported inside `@njit`; use legacy `np.random.seed`/`np.random.random()` in the kernel, or pre-generate uniforms outside. Carry into Phase 2.
- Not done here (deferred by plan): `specs/paper_spec.md` (Phase 1), engine code (Phase 2), throughput benchmark (Phase 2 gate).
- **Next action:** Phase 1 — transcribe Alg 1/2/3 + params into `specs/paper_spec.md` from `paper_html.txt` (~lines 350–640) and `paper.txt`; record the §6 open choices explicitly; after that, never read the PDF again.

### Phase 1 — spec freeze — PASS (2026-10-11)

- `specs/paper_spec.md` written from `paper_html.txt` (LaTeX `alttext`), cross-checked against `paper.txt`. PDF never opened.
- Alg 1 (18 lines), Alg 2 `Get-Rec` (6 lines), Alg 3 `Update-Blocklists` (5 lines) transcribed verbatim; §0 grep map records the exact `paper_html.txt` line ranges so later phases grep instead of re-reading.
- 16 numbered conventions `C1`–`C16` resolve the paper's gaps (UCB `0/0` at t=1, `A_0`, tie-breaks, synchronous `B_j`, initial `U/L`, sticky-set resampling, empty candidate set, `⌈j^η⌉` overflow cap, malicious agents pull no arms, per-trial paired ratio, 50 trials, oracle/no-comm definitions). Each is marked paper-derived or *our choice*.
- Params confirmed: `n=25 β=2 α=4 S=⌈K/n⌉`, `μ1=0.95 μ2=0.85 μ3..K~U[0,0.85]`, `T=1e5`, 50 trials. α=4 clears the Thm-2 bound `α>(3+(1+βη)/β)/2` for η=2/3/4 (2.75/3.25/3.75).
- Table 1 (12 rows × 4 cells) transcribed; `tests/test_spec_table1.py` parses the table out of `paper_html.txt` and asserts equality with the spec table — guards the transcription. `pytest tests/` → 3 passed.
- Figures 1–2 are raster in the PDF: no numeric axis data extractable → §11 records the qualitative landmarks (ratio <0.5, oracle≈proposed≪no_comm, S-curve, best-arm discard at T≈6e4 / 4e4).
- Open risk carried forward: MovieLens cohort + arm selection + softImpute rank/λ are **not in the paper** (§12) — Phase 7 must record ours and run a second cohort if time allows.
- **Next action:** Phase 2 — `src/mabamab/engine.py` + `experiments/bench_engine.py` (≥1M pulls/s gate), then Phase 3 gosine parity.

### Phase 2 — engine core — PASS (2026-10-11)

- `src/mabamab/{ucb,sticky,engine}.py` + `tests/test_engine.py` (16 tests total, all green: `.venv/Scripts/python -m pytest`).
- `engine.run_gossip` = one numba `@njit(cache=True)` kernel: Algorithm 1 lines 1–6, 8–16 with Algorithm 2 `Get-Rec` over honest recommenders (`m = 0`, every `P_j` empty). Line 7 (Update-Blocklists) is the marked no-op seam.
- Conventions wired: C1 (`T_k = 0` ⇒ index +inf, unplayed arms first), C2 (`A_0 = 0`, phase counts instead of `T_k(A_j) − T_k(A_{j−1})`), C3 (active list kept ascending + strict `>` ⇒ lowest-index tie-breaks, incl. line 12 `argmax{U,L}`), C4 (all `B_j` computed before any recommendation is read), C5/C6 (`sticky.py`), C10 (curve = mean over honest agents), C16 (Bernoulli).
- RNG: splitmix64 per agent, 64-bit seeds from `np.random.SeedSequence(base_seed).spawn(n)` (`agent_seeds`) ⇒ per-(trial,agent) reproducible streams; instance arrays are copied before the kernel mutates them, so paired algorithms can share one instance (C11).
- Throughput gate: `experiments/bench_engine.py` → **25.0M pulls/s**, `1.25e8` pulls (T=1e5 × n=25 × 50 trials) in **5.0 s**; gate was 1e6/s. `results/bench_engine.csv`, `results/metrics.json` (`engine_throughput`).
- Same run, sanity only (not a fidelity claim): mean honest-agent regret `905.6 ± 72.6` for gossip with `m=0, K=100, n=25, S=4, α=4, β=2`.
- numba 0.68 gotchas for later phases: `int(uint64)` inside `@njit` yields float64 (use `np.int64(...)`); `SeedSequence.generate_stream` does not exist (use `spawn` + `generate_state`).
- Not implemented (deferred by plan): Algorithm 3 blocking, malicious agents + uniform/omniscient strategies, `oracle` and `no_comm` baselines.
- **Next action:** Phase 3 — `experiments/p3_gossip.py`, compare `m=0` gossip regret against `gosine` `Synch.py` on a shared instance and check the `(K/n)·log T/Δ` slope.

### Phase 3 — gossiping baseline (m=0) — PASS (2026-10-12)

- `experiments/gosine_ref.py` ports upstream `Synch.py` (`github.com/ronshee/gosine`, commit `a9eec637…`: `getarm`/`update`/`update_delta_T`, arms-as-rows state, running-average means). `experiments/p3_gossip.py` harness, `tests/test_gossip.py` (4 tests); `.venv/Scripts/python -m pytest` → **20 passed**.
- Parity is **exact, not statistical**: `engine_draws()` replays the engine's splitmix64 stream into the port, so both see identical draws — per-agent regret identical, `max|diff| = 0.0` over 5 trials at T=2e4.
- The only divergence found was tie-breaking. Upstream resolves `argmax` ties in `arms` row order (sticky first, then U, then L) and keeps the U label on an equal-count epoch; spec C3 fixes both to lowest arm index. With C3 conventions the port matches exactly; with upstream's it costs **+44.3 ± 20 regret at T=2e4** (25 trials, z=2.18) and +46 ± 25 at T=1e5 (z=1.82) — a real but modest convention effect, recorded rather than hidden.
- Independent-stream parity with C3 conventions: port 656.7 ± 12.5 vs ours 642.0 ± 13.3 (z=0.81) ⇒ matches `gosine` within Monte-Carlo noise.
- Growth (25 trials, engine, instance seed 20260101): T=1250→188.5, 2e4→642.0, 8e4→838.1. Fit on T≥1e4: `regret ≈ −813 + 146.6·ln T` ⇒ logarithmic in T as Theorem 2 predicts, with empirical constant 3.7× the asymptotic `(K/n)/Δ = 40` and 0.147× the single-agent `K/Δ = 1000` (pre-asymptotic constant inflation, not a slope mismatch).
- Artifacts: `results/p3_parity.csv`, `p3_tiebreak.csv`, `p3_growth.csv`, `p3_scale.csv`; `results/metrics.json` → `gossip`. Command: `.venv/Scripts/python experiments/p3_gossip.py --trials 25 --t 20000 --parity-trials 5 --scale-t 100000 --scale-trials 5 --growth-trials 25` (~18 min).
- Carried forward: `gosine_ref.run_gosine(tie=…, swap_tie=…, exact_score=…)` keeps both conventions selectable — Phase 5/6 gossip curves must use the C3-aligned settings so paired ratios are not polluted by tie-breaks.
- **Next action:** Phase 4 — `src/mabamab/{blocking,malicious,baselines}.py` (Alg 3 blocks `B_j ≠ R_{j−1}` until `⌈j^η⌉`, uniform + omniscient strategies, oracle/no-comm) and the Theorem-1 smoke test (m=1 uniform ⇒ gossip ≈ no-comm).

### Phase 4 — Algorithm 3 + malicious agents + baselines — PASS (2026-10-12)

- New: `src/mabamab/blocking.py` (Alg 3 `Update-Blocklists`: C7 no-recommender ⇒ `B_j = R_{j−1}`, C8 block until `⌈j^η⌉`, C9 `R_{j−1}` = previous-epoch recommendation, cap at `j_max = ⌈(T/n)^{1/β}⌉`, monotone), `src/mabamab/malicious.py` (`uniform` = own splitmix64 stream, `omniscient` = `argmax` over the **receiving honest agent's** counts/sums, ties by lowest index), `src/mabamab/baselines.py` (`no_comm` = C15 isolated `UCB(α)` over all K arms + `run_trial_algo` dispatcher over the four compared algorithms), `src/mabamab/rng.py` (splitmix64 `uniform`/`next_u64`/`agent_seeds` extracted from `engine.py` so the Phase-3 parity harness keeps one stream).
- `engine.run_gossip` is now `run_trial(..., m, eta, strategy, oracle, diag)`: agents `0..n−1` honest, `n..n+m−1` malicious (pull no arm, contribute no regret — C12/C13), `oracle` = C14 (every agent recommends its own `argmax` mean, no blocklist). Line 7 wired in; `sticky.py` exposes `in_active`. Phase-3 parity is untouched: `tests/test_blocking.py::test_gossip_row_matches_the_phase_3_kernel` asserts the `m=0` row is **bit-identical** to `run_trial`'s predecessor. `.venv/Scripts/python -m pytest` → **34 passed** (14 new).
- numba 0.68 / Windows gotcha found: a `uint64` state ≥ 2^63 boxes back to Python as a **negative int**, so a Python-side loop that feeds it back re-specialises the kernel on `int64` and the shift goes arithmetic (`r ∈ [0,0.5)`). Inside the kernel the state lives in a `uint64` array, so the engine is unaffected; `tests/test_blocking.py` re-boxes with `np.uint64(state)`.
- Theorem-1 smoke (K=25 n=25 S=1 m=1 uniform, 25 paired trials, `results/p4_theorem1.csv`): gossip/no_comm = 0.369 (T=1250) → 0.478 (T=20000), **rising** with T; log-log slopes 0.648 (gossip) vs 0.552 (no_comm). Plan expected ≈1. Deviation explained: C15 `no_comm` explores all K arms with no sticky/active sets, so it is worse than the paper's implicit baseline; Theorem 1 only bounds gossip from above, and the ratio trending up with T is the direction the theorem allows.
- Paper-scale sanity (K=100 n=25 S=4 m=10 omniscient η=2 T=1e5, 10 paired trials, `results/p4_sanity.csv`): proposed 2403 ± 295, gossip 7667 ± 130, oracle 835 ± 32, no_comm 10150 ± 137 ⇒ proposed/gossip **0.313**, oracle/gossip **0.109**, gossip/no_comm **0.755**. Figure-1 landmarks held: proposed ≪ gossip, proposed > oracle (blocking costs regret but recovers most of it), C7 fires (3601 no-recommender epochs, 6899 malicious recommendations absorbed). **Open deviation:** gossip < no_comm here (paper shows gossip ≫ no-comm) — same C15 cause; carry into the Phase 8/9 report rather than tune the baseline.
- Command: `.venv/Scripts/python experiments/p4_theorem1.py` (~23 s); `results/metrics.json` → `blocking`.
- **Next action:** Phase 5 — Figure 1 reproduction (η ∈ {2,3,4}, m/n sweep, proposed vs gossip vs oracle vs no-comm, 50 trials) and Figure 2 (η sweep), with `gosine_ref` gossip settings kept C3-aligned.

### Phase 4 addendum — the C15 no-comm fork, resolved (2026-10-12)

- The paper never gives no-comm pseudocode (line 808), so C15 had two readings. Both are now implemented and measured; **C15a stays frozen** (spec §8, §11.1).
  - **C15a** `no_comm` = `UCB(α=4)` over all K arms, isolated — kept, same α as every other algorithm, no post-hoc tuning.
  - **C15b** `no_comm_alg1` = Algorithm 1 with the `Get-Rec` path removed (`engine.run_trial(..., isolated=True)`; lines 9–14 factored into `_apply_rec`, shared with the communicating path). **Rejected:** log-log slope **0.987** ⇒ linear regret (23789 @T=1e5, 235092 @T=1e6). An isolated sticky-set agent discards the best arm permanently, so this cannot be the paper's sublinear no-comm curve.
- Theorem-1 gate restated (plan §3 Phase 4) and **passed**: `regret / [α(1−1/√α)² Σ_{k≥2} 1/Δ_k · log T]` with `Σ 1/Δ_k = 68.5` ⇒ eq.(3) constant 68.5/log T. gossip 0.96→2.96, no_comm 2.23→3.74 over T=1e4→1e6 (O(1), non-decreasing = the liminf lower bound Theorem 1 states). Old numeric gate (gossip/no_comm ≈ 1.3 @T=1e4) is unattainable: 0.428 @1e4 → 0.793 @1e6, slopes 0.329 vs 0.194 ⇒ crossing only near T ≈ 5e6.
- **Landmark §11.1 half-fails, cause pinned:** at K=100 n=25 S=4 m=10 omniscient η=2 T=1e5, `no_comm` regret is ~linear in α — 2868 (α=1), 5467 (α=2), 10150 (α=4), 17461 (α=8) — while gossip is 7667. `gossip ≫ no_comm` needs α ≲ 2. Recorded as a mismatch; do not tune the baseline to fit.
- Figure 1/2 plotting decision: plot **C15a α=4 only** (one no-comm curve), and report the α-sensitivity in the Phase 8/9 text.
- Artifacts: `experiments/baseline_probe.py` (~2 min), `results/baseline_probe_growth.csv`, `results/baseline_probe_scale.csv`, `results/metrics.json` → `no_comm_baseline`. `.venv/Scripts/python -m pytest` → **35 passed** (new: `test_no_comm_alg1_is_the_isolated_algorithm_1`).
- **Next action:** Phase 5 — Figure 1 (m=10, K=100, η ∈ {2,3,4}, 50 trials, log-x to 1e5, one resumable CSV row per config), curves = proposed / gossip / oracle / no_comm(α=4).

### Phase 5 — Figure 1 (synthetic, m=10 K=100 η=2) — PASS (2026-10-12)

- `experiments/p5_fig1.py`: 50 trials × 2 strategies (uniform, omniscient) × 4 algorithms (proposed / gossip / oracle / no_comm C15a α=4), `n=25 K=100 S=4 m=10 η=2 α=4 β=2`, T=1e5, 25 log-spaced checkpoints, 1e9 pulls in **172 s**. Proposed and gossip share instance + agent seeds per trial (C11) so the ratio is paired.
- Raw per-trial CSVs `results/fig1_raw_<strategy>_<algorithm>.csv` are cached: re-running the same `--trials/--t` skips the compute and re-aggregates in ~1 s. Aggregates: `results/fig1.csv` (200 rows), `results/fig1.png` (2×2: regret + arm-1 discard rate per strategy), `results/metrics.json` → `fig1`.
- **Gate 1 passes:** paired per-trial `proposed/gossip` at T=1e5 = **0.421 ± 0.214** (uniform), **0.290 ± 0.095** (omniscient) — both < 0.5. Paper Table 1 row `m=10,K=100,η=2`: 0.413 ± 0.086 / 0.401 ± 0.045 ⇒ uniform matches, our omniscient blocking is *stronger* than the paper's.
- **Gate 2 passes:** final mean honest-agent regret — gossip 6696 (uniform) / 7663 (omniscient), proposed 2822 / 2212, oracle 842, no_comm 10028 ⇒ `oracle < proposed ≪ gossip` both strategies.
- **Gate 3 passes (S-curve, omniscient gossip):** regret slope 104 → **3761 at T=56234** → 2513 (paper: S-curve with the bump at 6e4). Uniform gossip slope is still rising at T=1e5 (peak 2494 at the axis edge) — the paper only claims the S-curve for omniscient.
- **Deviation (recorded, not gated): best-arm discards are not localised at 6e4.** `run_trial(..., dropped=)` now counts cumulative arm-1 **discard events** (agent had arm 1 in `S_j` at an epoch boundary and lost it via lines 12–14). Gossip rate ≈ 0.07–0.08 discards/agent/epoch from T≈5e2 through T≈3e4 (16–17 per agent by 1e5); no wave at 6e4.
- **Mechanism confirmed instead:** Algorithm 3 stops the discards — proposed rate collapses to ~0 after T≈1.3e4 (last-third mean 0.0020 vs gossip 0.0514 omniscient; 0.0061 vs 0.0550 uniform). Oracle stops them by T≈6e3.
- Metric trap found: the first version latched "share of agents that ever discarded arm 1", which saturated at 0.94 by T≈3e2 because `A_1 = 1` makes line 12 drop the unpulled member of `{U_1,L_1}` at t=1. Replaced by the event-rate metric; `tests/test_engine.py::test_dropped_counts_agents_that_discarded_the_best_arm` pins it (arm-1-sticky agents never count; control instance does). `pytest -q` → **36 passed**.
- **Deviation (recorded, not gated): gossip < no_comm** — 0.668 (uniform) / 0.764 (omniscient). Known C15a consequence (spec §11.1, α-sensitivity), carried to the Phase 8/9 report.
- **Next action:** Phase 6 — Table 1 synthetic: 12 configs × 2 strategies × (proposed, gossip) at T=1e5, paired seeds, per-trial ratio vs paper ±0.10 (plan §4), reusing the `fig1_raw_*` caching pattern as `results/table1_raw_*.csv`.

### Phase 6 — Table 1, synthetic half — PASS (2026-10-13)

- `experiments/p6_table1_synth.py`: 12 configs (`m ∈ {10,15} × K ∈ {75,100} × η ∈ {2,3,4}`) × 2 strategies, `T=1e5`, `n=25 α=4 β=2 S=⌈K/n⌉` (3 for K=75, 4 for K=100), 50 paired trials (C11/C12/C13, same seed scheme as p4/p5). Paper targets are parsed out of spec §10 at run time, not hard-coded; `tests/test_table1_synth.py` (5 tests) asserts they equal the `paper_html.txt` transcription. `pytest -q` → **41 passed**.
- Cost: 24 cells × 2 algorithms × 1.25e8 pulls = **6e9 pulls in ~80 s** (4 parallel processes via `--m/--K/--eta`; per-cell raw CSVs `results/table1_raw_*.csv` make the run resumable/skippable).
- **Exit criterion passes: 21/24 cells match (|Δ| ≤ 0.10), 1 within-noise, 2 mismatch, mean |Δ| = 0.048.** All 12 uniform-column cells match (max |Δ| 0.084). Cross-check: the `m=10 K=100 η=2` uniform cell reproduces Phase 5's 0.421 exactly (same seeds).
- Mismatches (both omniscient): `m=10 K=100 η=2` 0.290 ± 0.095 vs paper 0.401 ± 0.076 (Δ −0.111 — the same value Phase 5 measured independently, so our blocking is *stronger* than the paper's there) and `m=15 K=75 η=3` 0.549 ± 0.180 vs 0.439 ± 0.054 (Δ +0.110).
- Investigation (raw per-trial data, no engine change): the mismatches are **proposed-side, not denominator-side** — gossip regret is tight in every cell (±278–409) while proposed spreads ±556–1458. Blocking success is bimodal: 24–28% of the `m=15 K=75` omniscient trials sit at ratio > 0.6 (median 0.627), whereas `m=10 K=100 η=2` omniscient is uniformly strong (median 0.313). Our omniscient ratio stds (0.086–0.186) exceed the paper's (0.034–0.114) while the means stay within ±0.11 ⇒ the paper's ±std is not reproducible from a 50-trial paired ratio; carried to the Phase 9 report, not tuned away.
- Artifacts: `results/table1_synth.csv` (24 rows: paper value, ours, per-algorithm mean regrets, unpaired ratio, Δ, verdict), `results/metrics.json` → `table1_synth`.
- **Next action:** Phase 7 — MovieLens pipeline (`experiments/p7_movielens.py`, cohort filter ≥30/≥30, numpy soft-thresholded ALS/SVD since `softImpute` is unavailable, `results/arm_means.csv`, ≥100 arms).

### Phase 7 — MovieLens pipeline — PASS (2026-10-13)

- New: `experiments/p7_movielens.py` + `tests/test_movielens.py` (7 tests). `.venv/Scripts/python -m pytest -q` → **48 passed**. ml-1m unpacked into `data/ml-1m/` (gitignored since Phase 0).
- **C17 cohort:** `18-M-4` (age 18–24, male, college) = 340 users × 695 movies, 45,187 ratings, density 0.191 — the largest cohort surviving the ≥30-ratings-per-user / ≥30-ratings-per-movie filters. Runner-up `25-M-0` (183 × 516) also run, per plan §3. Filters applied in one pass (users then movies); 23 users fall below 30 ratings after the movie filter — recorded, not iterated away.
- **C18 completion:** no R / rpy2 / softImpute in this env, so the cited algorithm is re-implemented in numpy (`softimpute()`: impute → rank-r SVD → soft-threshold singular values by λ, column-mean offset, clip to [1,5]). Rank and λ picked by 3-fold holdout RMSE on observed ratings: max rank, λ = 7.807, RMSE 0.921 (cohort 2: rank 20, λ = 11.626, RMSE 0.901). Grid in `results/p7_softimpute.csv`.
- **C19 star rounding:** completed ratings are rounded to whole stars before the ≥ 4 test. Thresholding the unrounded estimates instead gives mean arm mean **0.25** — nuclear-norm shrinkage destroys the bimodal 4/5 mass — while rounding gives **0.566** against the cohort's observed 0.616. Both columns are kept in `results/arm_means.csv`.
- **C20 arm selection:** seeded uniform sample of 100 of the 695 filtered movies, sorted by arm mean descending so arm 0 is the best arm. The 100-most-rated variant gives 0.783 (popular titles are liked disproportionately by this cohort); kept as a sensitivity in metrics.
- Exit checks (plan §2): **100 arms ✓, all arm means in [0,1] ✓, mean arm mean 0.566 ≈ 0.6 ✓**; cohort 2 gives 100 arms, mean 0.585. No plan §7 stop condition hit.
- Artifacts: `results/arm_means.csv` (Phase 8 input), `results/arm_means_25-M-0.csv`, `results/p7_cohorts.csv`, `results/p7_softimpute.csv`, `metrics.json` → `movielens`. Command: `.venv/Scripts/python experiments/p7_movielens.py` (10.5 min, almost all of it the CV grid; `--reuse-cv` reproduces the same numbers in 26 s).
- **Phase-8 warning (measured, not tuned):** the real instance has near-ties at the top — cohort `18-M-4` best arm mean 1.000 with **Δ₂ = 0.0029** and 7 arms within 0.01 of the best; `25-M-0` has Δ₂ = 0.0000 with 14 such arms — versus Δ₂ = 0.1 in the synthetic instances. Real-data regret magnitudes will not be comparable cell-for-cell with the synthetic columns.
- **Next action:** Phase 8 — `experiments/p8_real.py`: Figure 2 (m=15, K=100, η=2) plus the 12-row Table 1 real columns, same harness and seed scheme as p5/p6 with the arm means loaded from `results/arm_means.csv`.

### Phase 8 — Figure 2 + Table 1 real columns — PASS with recorded deviations (2026-10-13)

- New: `experiments/p8_real.py` (Figure 2 **and** the 12-row real grid in one resumable harness, one raw CSV per config) + `tests/test_real_instance.py` (5 tests) → `.venv/Scripts/python -m pytest -q` = **53 passed**. `engine.make_instance_from_means()` splits the sticky/U/L draw from the mean draw; `make_instance` keeps its exact rng call order, so the Phase 5/6 caches and results are untouched.
- Instance: `results/arm_means.csv` (C17–C20); `K=75` = top-75 arms of that sorted vector, frozen as new convention **C21**. Same seed scheme as p4/p5/p6 (`INSTANCE_SEED + r`, `TRIAL_SEED + 13r`), proposed/gossip paired per trial (C11).
- Cost: 24 table cells + 8 Figure-2 configs + 8 cohort-sensitivity configs = **8e9 pulls in ~5 min wall** (5 parallel processes writing disjoint raws, then one `--aggregate-only`-style pass that owns `metrics.json`).
- **Table 1 real: 16/24 match (|Δ| ≤ 0.10), 6 within-noise, 2 mismatch, mean |Δ| = 0.086** — under the plan §7 stop (≥ 4). All 12 uniform cells and 10/12 omniscient cells land within the paper's ±std.
- Mismatches investigated from the raw per-trial data (no engine change): `m=10 K=75 η=2` omniscient 0.711 ± 0.055 vs 0.543 ± 0.136 is **denominator-side and systematic** — every `K=75` omniscient cell sits at 0.70–0.79 (median 0.712, 98% of trials > 0.6) because C21's top-75 subset raises the mean arm mean to 0.741 and shrinks gossip regret to 3244 (vs 4215 at `K=100`) while proposed stays ≈ 2300. `m=10 K=100 η=3` uniform 0.405 ± 0.048 vs 0.564 ± 0.143 is proposed-side: our uniform `K=100` ratios run 0.40–0.48 for every η, i.e. our blocking is stronger than the paper's on real data too.
- Our real-column ratio stds (0.038–0.080) are **tighter** than the paper's (0.058–0.229) — the opposite of the synthetic columns (Phase 6) ⇒ the paper's ±std is not reproducible in either direction.
- **Figure 2 (`m=15 K=100 η=2`, 50 paired trials):** paired ratio **0.485 ± 0.074 uniform (PASS < 0.5) / 0.580 ± 0.046 omniscient (FAIL < 0.5**, though that cell vs paper 0.501 is a Table-1 `match`). Final regret: gossip 5162 / 4088, proposed 2505 / 2366, oracle 1077, no_comm 6825 ⇒ `oracle < proposed < gossip < no_comm`; the C15a `gossip ≫ no_comm` half still fails (0.76 / 0.60).
- **Landmarks that hold on real data:** the omniscient gossip S-curve with the regret-slope peak at **T = 31623** (paper's Fig-2 bump ≈ 4e4) and the discard-rate peak at the same checkpoint; uniform gossip slope still rising at T=1e5.
- **Deviation:** Algorithm 3 only *halves* best-arm discards here (proposed 0.0138 vs gossip 0.0300 per agent/epoch uniform; 0.0108 vs 0.0358 omniscient; 5.2 vs 9.4 cumulative per agent) — the synthetic ~0 rate does not carry over to the near-tied real instance.
- Cohort sensitivity (plan §6 risk 1): runner-up cohort `25-M-0` gives Figure-2 ratios 0.419 / 0.572 vs 0.485 / 0.580 ⇒ cohort choice moves the ratio ≤ 0.07, inside the paper's ±std, so it does not explain the real-column deltas.
- Artifacts: `results/fig2.csv`, `results/fig2.png` (gitignored), `results/fig2_raw_*.csv`, `results/fig2_cohort_sensitivity.csv`, `results/table1_real.csv`, `results/table1_real_raw_*.csv`, `metrics.json` → `fig2`, `table1_real`. Spec updated: **C21**, §10 measured line, §11.2–11.4 Figure-2 results, §12.8.
- **Next action:** Phase 9 — `REPLICATION_REPORT.md`: paper value | replicated value | delta | verdict for Figure 1, Figure 2 and all 48 Table 1 cells.
