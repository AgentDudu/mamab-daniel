# Replication Plan — "Robust Multi-Agent Multi-Armed Bandits" (arXiv:2007.03812v3)

**Paper:** Daniel Vial, Sanjay Shakkottai, R. Srikant. *Robust Multi-Agent Multi-Armed Bandits.* MobiHoc '21, DOI 10.1145/3466772.3467045. Local copy: `2007.03812v3.pdf` (22 pages).
**Baseline paper replicated as a component:** Chawla, Sankararaman, Ganesh, Shakkottai. *The Gossiping Insert-Eliminate Algorithm for Multi-Agent Bandits* (AISTATS 2020) — official code `https://github.com/ronshee/gosine` (Python: `Synch.py`, `Asynch.py`).
**Long version with extra experiments:** arXiv:2007.03812 (this PDF) — Appendix D = Table 1, the main quantitative target.

---

## 0. Why phases / context discipline

The paper is 22 pages of proofs plus 2 figures and 1 table. Only ~1 page is empirical. Re-reading the PDF costs context and adds nothing after Phase 1.

**Hard rules for every phase:**

1. **Never read `2007.03812v3.pdf` after Phase 1.** All needed facts live in `specs/paper_spec.md`.
2. Each phase reads **only** the files listed in its "Inputs" row, plus `STATE.md`.
3. Each phase ends by (a) writing its outputs to disk, (b) appending a ≤15-line entry to `STATE.md` (what was done, exact commands, pass/fail, next action).
4. Numeric results go to `results/*.csv` + `results/metrics.json`, never into chat prose. Chat reports only pass/fail + deltas.
5. A phase is **done** when its Exit criteria pass. If a phase fails twice, stop and report; do not silently re-scope.
6. Keep `paper.txt` (pdftotext dump) and `paper_html.txt` (HTML→text, math alttext preserved) on disk as grep targets instead of re-reading the PDF.

**Repo layout created in Phase 0:**

```
mamab-daniel/
  2007.03812v3.pdf  paper.txt  paper_html.txt  REPLICATION_PLAN.md  STATE.md
  specs/paper_spec.md
  src/mabamab/        # engine: agents, algos, malicious strategies, datasets
  tests/
  experiments/        # one script per figure/table row set
  results/            # csv + json + figures
  logs/
```

---

## 1. Paper facts to replicate (extracted; the contract for all phases)

### Setting
- `n` honest agents + `m` malicious agents, complete graph, each agent faces its own instance of the same `K`-armed Bernoulli bandit.
- Time discrete. Phases delimited by communication epochs `A_j = ceil(j^β)`, `β > 1` (shared across agents).
- Agent `i` keeps a **sticky set** `Ŝ^(i)`, `|Ŝ^(i)| = S`, and an **active set** `S_j^(i) = Ŝ^(i) ∪ {U_j^(i), L_j^(i)}` (size `S+2`).
- Assumption used in experiments: `1 ∈ ∪_i Ŝ^(i)` (best arm in at least one agent's sticky set).

### Algorithm 1 — `Multi-Agent-MAB-With-Blocking(α, β, Ŝ^(i))`
1. `A_j' = ceil(j'^β)`, `P_j'^(i) = ∅` ∀ j'.
2. `j = 1`; `U_1^(i), L_1^(i)` = two distinct arms of `{1..K} \ Ŝ^(i)`; `S_1^(i) = Ŝ^(i) ∪ {U_1^(i), L_1^(i)}`.
3. for `t ∈ ℕ`:
   4. pull `I_t^(i) ∈ argmax_{k ∈ S_j^(i)} μ̂_k^(i)(t-1) + sqrt(α log t / T_k^(i)(t-1))` — **UCB(α) restricted to the active set**.
   5. if `t == A_j`:
      6. `B_j^(i) = argmax_{k ∈ S_j^(i)} T_k^(i)(A_j) − T_k^(i)(A_{j-1})` (most-played active arm this phase).
      7. `{P_j'} ← Update-Blocklists(...)` — **no-op for the Chawla baseline; Algorithm 3 for the proposed algorithm**.
      8. `(H_j^(i), R_j^(i)) = Get-Rec(i, j, P_j^(i))`.
      9. if `R_j^(i) ∈ S_j^(i)`: `S_{j+1}^(i) = S_j^(i)`.
      12. else: `U_{j+1}^(i) = argmax_{k ∈ {U_j, L_j}} T_k(A_j) − T_k(A_{j-1})`; `L_{j+1}^(i) = R_j^(i)`; `S_{j+1}^(i) = Ŝ^(i) ∪ {U_{j+1}, L_{j+1}}`.
      16. `j ← j+1`.

### Algorithm 2 — `Get-Rec(i, j, P_j^(i))` (black box to honest agents)
1. `H_j^(i)` uniform over `{1..n+m} \ (P_j^(i) ∪ {i})`.
2. if `H_j^(i) ≤ n` (honest): `R_j^(i) = B_j^(H_j^(i))` (its most-played arm in phase j).
3. else (malicious `i'`): `R_j^(i) ~ ν_{j,i}^(i')` — arbitrary strategy.

### Algorithm 3 — `Update-Blocklists`
1. if `j > 1` and `B_j^(i) ≠ R_{j-1}^(i)` (previous recommendation was not this phase's most-played arm):
2. for `j' ∈ {j, ..., ceil(j^η)}`: `P_j'^(i) ← P_j'^(i) ∪ {H_{j-1}^(i)}`.
- Parameter `η > 1` controls blocking length. Blocklists are infinite sequences; implement as `dict[agent] = until_epoch` (Remark 6).

### Theorem 1 (lower bound, empirical check)
`m = 1`, malicious strategy uniform over `{1..K}`, no blocking (Chawla algo) ⇒ honest regret `Ω(K log T / Δ)` — i.e. collaboration benefit destroyed, matches single-agent baseline.

### Theorem 2 (upper bound, empirical check)
`β > 1`, `α > (3 + (1+βη)/β)/2`, `η > 1` ⇒ regret `O(min{m + K/n, K} · log T / Δ)` (Remark 7, multiplicative constant `min{m + K/n, K}`). Improvement over the `O(K log T/Δ)` single-agent baseline whenever **`m < K(1 − 1/n)`**. General theory choice `S = ⌈(K/n) log(1/ε)⌉` (Remark 3); **experiments use `S = ⌈K/n⌉`** (= 4 for `K=100, n=25`).

### Experiment setup (Section 6, verbatim values)
| Item | Value |
|---|---|
| Synthetic arm means | `μ1 = 0.95`, `μ2 = 0.85` (so `Δ2 = 0.1`), `μ3..μK ~ U[0, 0.85]` |
| Fixed across experiments | `n = 25`, `β = 2`, `α = 4`, `S = ⌈K/n⌉` (so active-set size `S+2`) |
| Sticky sets | uniform random, resampled until `1 ∈ ∪_i Ŝ^(i)` |
| Trials | 50, report mean ± std of regret |
| **Figure 1** (synthetic) | `m = 10`, `K = 100`, `η = 2` |
| **Figure 2** (real) | `m = 15`, `K = 100`, `η = 2` |
| Compared algorithms | (a) proposed (blocking), (b) Chawla gossiping (no blocking), (c) no-communication baseline, (d) oracle (honest agents know & block malicious agents a priori) |
| Malicious strategies | **uniform**: `ν` uniform over `{1..K}`. **omniscient**: `argmin_{k' ∈ {2..K} \ S_j^(i)} T_{k'}^(i)(A_j)` — least-played inactive suboptimal arm |
| Qualitative landmarks | gossiping "S-curve"; best arm discarded by honest agents at `T ≈ 6e4` (Fig 1) and `T ≈ 4e4` (Fig 2); proposed algo < ½ the regret of gossiping |

### Table 1 (Appendix D) — primary quantitative target
Average regret `E[R^(i)(T)]` **ratio relative to the Chawla algorithm** at `T = 1e5`, mean ± std, same datasets/strategies, `n = 25, β = 2, α = 4, S = ⌈K/n⌉`:

| m | K | η | Synth-uniform | Synth-omniscient | Real-uniform | Real-omniscient |
|---|---|---|---|---|---|---|
| 10 | 75 | 2 | 0.450 ± 0.160 | 0.413 ± 0.052 | 0.582 ± 0.157 | 0.543 ± 0.136 |
| 10 | 75 | 3 | 0.415 ± 0.172 | 0.415 ± 0.046 | 0.578 ± 0.168 | 0.567 ± 0.204 |
| 10 | 75 | 4 | 0.435 ± 0.124 | 0.395 ± 0.034 | 0.525 ± 0.143 | 0.593 ± 0.227 |
| 10 | 100 | 2 | 0.413 ± 0.086 | 0.401 ± 0.076 | 0.560 ± 0.142 | 0.483 ± 0.080 |
| 10 | 100 | 3 | 0.464 ± 0.235 | 0.412 ± 0.114 | 0.564 ± 0.143 | 0.504 ± 0.108 |
| 10 | 100 | 4 | 0.418 ± 0.107 | 0.404 ± 0.070 | 0.535 ± 0.139 | 0.521 ± 0.119 |
| 15 | 75 | 2 | 0.418 ± 0.088 | 0.433 ± 0.047 | 0.547 ± 0.119 | 0.603 ± 0.217 |
| 15 | 75 | 3 | 0.411 ± 0.081 | 0.439 ± 0.054 | 0.551 ± 0.138 | 0.651 ± 0.229 |
| 15 | 75 | 4 | 0.423 ± 0.105 | 0.451 ± 0.062 | 0.557 ± 0.109 | 0.645 ± 0.220 |
| 15 | 100 | 2 | 0.430 ± 0.113 | 0.408 ± 0.040 | 0.507 ± 0.120 | 0.501 ± 0.058 |
| 15 | 100 | 3 | 0.429 ± 0.133 | 0.414 ± 0.058 | 0.494 ± 0.120 | 0.514 ± 0.089 |
| 15 | 100 | 4 | 0.420 ± 0.085 | 0.412 ± 0.058 | 0.514 ± 0.110 | 0.511 ± 0.078 |

### Real data pipeline (MovieLens, Section 6.2)
1. MovieLens (`ml-1m`, Harper & Konstan 2015). Movies = arms.
2. Extract rating matrix restricted to users sharing the **same age, gender, and occupation**; keep users with **≥ 30 ratings** and movies rated **≥ 30 times** by that user set.
3. **Matrix completion** (Hastie et al. 2015, `softImpute`-style) to fill missing entries.
4. Arm mean = fraction of ratings **≥ 4** on the 1–5 scale (per movie, over the estimated matrix). Bernoulli rewards.
5. Run the same four algorithms, `K = 100` arms, `m = 15`, `η = 2`.

---

## 2. Phase plan

| # | Phase | Reads | Writes | Exit criterion | Est. wall time |
|---|---|---|---|---|---|
| 0 | Env + scaffolding | this file | repo skeleton, `requirements.txt`, `STATE.md` | `pytest` runs (0 tests), import works | 10 min |
| 1 | Spec freeze | PDF, `paper.txt`, `paper_html.txt`, arXiv HTML | `specs/paper_spec.md` | pseudocode of Alg 1/2/3 transcribed line-by-line, params table filled, PDF never read again | 20 min |
| 2 | Engine core | `specs/paper_spec.md` | `src/mabamab/{engine,ucb,sticky}.py`, `tests/test_engine.py` | unit tests pass; throughput ≥ 1e6 arm pulls/s | 1–2 h |
| 3 | Gossiping baseline (m=0) | spec, engine | `experiments/p3_gossip.py`, `results/p3_*.csv` | `m=0` regret ≈ `O((K/n) log T / Δ)`; matches `gosine` on a shared seed/instance | 1 h |
| 4 | Blocking + malicious + baselines | spec, engine | `src/mabamab/{blocking,malicious,baselines}.py`, `tests/test_blocking.py` | Alg 3 blocks on `B_j ≠ R_{j-1}` until `ceil(j^η)`; oracle & no-comm baselines implemented; Theorem-1 smoke test (m=1 uniform ⇒ gossip ≈ no-comm) | 2–3 h |
| 5 | Figure 1 (synthetic) | spec, engine | `experiments/p5_fig1.py`, `results/fig1.{csv,png}` | 4 curves × 2 strategies, `n=25,K=100,m=10,η=2,β=2,α=4`, 50 trials; proposed < 0.5 × gossip; gossip S-curve + best-arm discard near `T≈6e4` | 2–6 h compute |
| 6 | Table 1, synthetic half | spec, engine | `experiments/p6_table1_synth.py`, `results/table1_synth.csv` | 12 rows × 2 strategies within tolerance of paper (see §4) | 4–12 h compute |
| 7 | MovieLens pipeline | spec §real-data | `experiments/p7_movielens.py`, `data/ml-1m/`, `results/arm_means.csv` | cohort filter + matrix completion yields ≥ 100 arms; arm means in [0,1]; sanity: mean ≈ 0.6 | 1–2 h |
| 8 | Figure 2 + Table 1 real half | spec, engine, Phase-7 outputs | `experiments/p8_real.py`, `results/fig2.*`, `results/table1_real.csv` | same criteria as 5/6 on real means; discard bump near `T≈4e4` | 4–12 h compute |
| 9 | Verification + report | `results/metrics.json`, spec | `REPLICATION_REPORT.md` | every figure/table cell mapped to paper value with delta + verdict | 1 h |

Phases 2→4 are strictly sequential. 5 and 6 can run in parallel once 4 passes. 7 is independent of 2–6 and can run early (good warm-up). 8 needs 4 and 7.

---

## 3. Phase details

### Phase 0 — environment
- Ask the user for execution environment (local / venv / Docker / cloud / plan-only) **before running anything**. Default: local venv.
- Pin: `python>=3.11`, `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`, `numba` (or `cython`) for the hot loop, `softimpute`-equivalent (see Phase 7 fallback).
- Create `STATE.md` with a "Phase log" heading.

### Phase 1 — spec freeze
- Transcribe Alg 1/2/3 exactly (already drafted in §1 above — verify against `paper_html.txt` lines ~350–640, which preserve LaTeX `alttext`).
- Record ambiguities explicitly rather than guessing: sticky-set sampling with/without replacement across agents, tie-breaking in `argmax`/`argmin`, whether `B_j^(i')` is read at the same epoch boundary (assume yes — synchronous epochs), how the no-comm and oracle baselines are defined (paper gives no pseudocode; define and document).
- Output must be self-sufficient: a later phase must be able to implement from `specs/paper_spec.md` alone.

### Phase 2 — engine core (the performance gate)
- Per-agent state: counts `T_k`, sums `μ̂_k`, sticky set, `U`, `L`, blocklist dict.
- Vectorize over agents (and trials) with numpy; the inner loop is `K`-sized argmax over the active set only (`S+2` arms, not `K`).
- Deterministic seeding: `SeedSequence` per (trial, agent) so runs are reproducible and resumable.
- **Benchmark required**: `T=1e5 × n=25 × 50 trials = 1.25e8 pulls` must finish in minutes, not hours. If pure Python, Phase 2 fails — switch to numba before continuing.

### Phase 3 — baseline fidelity
- Port/compare against `github.com/ronshee/gosine` (`Synch.py`) on one shared instance; expected agreement within Monte-Carlo noise.
- Sanity: with `m=0`, gossiping regret slope ≈ `(K/n)·log T/Δ` and clearly below single-agent `K log T/Δ`.

### Phase 4 — the contribution
- Blocking rule exactly as Alg 3 (note: it compares `B_j^(i)` to `R_{j-1}^(i)`, i.e. the block decision at epoch `j` punishes the recommender chosen at epoch `j−1`).
- Malicious strategies: uniform; omniscient (needs the *receiving* honest agent's per-arm counts — that is the "private information" the name refers to).
- Oracle: honest agents permanently exclude `{n+1..n+m}` from `Get-Rec` sampling.
- No-comm: each honest agent runs UCB(α) over all `K` arms (document this choice; paper does not specify).
- Theorem-1 smoke test: `m=1`, uniform strategy, `K=25`, `T=1e4` → gossip regret ≈ no-comm regret (ratio within ~1.3).

### Phase 5–6 — synthetic figures/table
- Regret definition: per-agent cumulative regret `E[R^(i)(T)] = Σ_t Σ_{k≠1} Δ_k · 1{I_t^(i)=k}`, averaged over honest agents; report mean ± std across the 50 trials.
- Curves: regret vs `T` on log-x, up to `T ≈ 1e5` (paper's x-axis reaches ~1e5; the discard bump is at 6e4).
- Table 1 metric: `mean_trials( regret_ours(T=1e5) / regret_gossip(T=1e5) )` ± std of the per-trial ratio. Compute both algorithms on the **same** instance/trial seeds (paired) — this is what makes the ±0.1 std reproducible.
- Run configs as separate processes; write one CSV row per config as it completes (resumable, cheap to re-enter context).

### Phase 7 — MovieLens
- Download `ml-1m.zip` (GroupLens). Pick a cohort by (age, gender, occupation) — the paper does not say which; **choose the largest cohort satisfying the ≥30/≥30 filters and record the choice**. This is the main non-reproducibility risk; document it and, if time allows, repeat with a second cohort.
- Matrix completion: prefer R `softImpute` (the cited method) via `rpy2`, else implement ALS/SVD soft-thresholding in numpy and note the deviation.
- Arm means = fraction of ratings ≥ 4 per movie from the completed matrix; keep top/first `K = 100` arms (paper does not state selection; document).

### Phase 8–9 — real data + report
- Same harness as Phase 5/6, arm means swapped.
- Report table: paper value | replicated value | delta | verdict (match / within-noise / mismatch), for Figure 1, Figure 2, and all 48 Table 1 cells.

---

## 4. Acceptance tolerances

- Table 1 ratios: **match** if |ours − paper| ≤ 0.10; **within-noise** if within paper's reported ±std; else **mismatch** (investigate before reporting).
- Figure 1/2 qualitative claims (all must hold): proposed ≈ oracle ≪ no-comm; gossip ≫ no-comm; proposed < 0.5 × gossip; S-curve for gossip under omniscient; best-arm discard bump at `T ≈ 4–6e4`.
- Scaling claims: slope of regret vs log T roughly linear in log T; `m` small vs `K` ⇒ improvement, `m` comparable to `K` ⇒ no improvement (Remark 7).

## 5. Compute budget

Pulls per config = `T × n × trials`. Figure 1: `1e5 × 25 × 50 = 1.25e8` per algorithm per strategy; 4 algorithms × 2 strategies ≈ 1e9. Table 1 synthetic: 12 configs × 2 strategies × 2 algorithms ≈ 3e9. Total across Phases 5–8 ≈ 8e9 pulls.
At 1e6 pulls/s (numba, single core) ≈ 2.2 CPU-hours per 1e9 → ~20 CPU-hours total; parallelize across configs. If throughput is < 1e5 pulls/s, cut trials to 20 and flag the std inflation in the report.

## 6. Known risks / open choices (must be documented, not silently decided)

1. **MovieLens cohort choice** — unspecified in paper; biggest source of deviation in Figure 2 and real Table 1 columns.
2. **Matrix completion implementation** — `softImpute` vs ALS; rank and λ not stated.
3. **Arm selection for K=100** from the completed matrix — not stated.
4. **No-communication and oracle baselines** — no pseudocode in paper; our definitions must be written down.
5. **Tie-breaking** in argmax/argmin — affects the discard-bump location slightly.
6. **Sticky-set resampling** — per-agent independent uniform subsets of size `S = ⌈K/n⌉`; resample the whole collection until `1 ∈ ∪_i Ŝ^(i)`.
7. Paper's Figure 1/2 are images; only Table 1 gives hard numbers. Figure comparison is qualitative + landmark-based.

## 7. Stop conditions

- Stop and ask the user if: Phase 2 throughput gate fails after numba attempt; Phase 3 disagrees with `gosine` beyond noise; Table 1 has ≥ 4 mismatching cells after investigation; MovieLens filters yield < 100 usable arms.
