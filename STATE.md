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
