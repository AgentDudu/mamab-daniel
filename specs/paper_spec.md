# `specs/paper_spec.md` — frozen paper contract

Paper: Vial, Shakkottai, Srikant. *Robust Multi-Agent Multi-Armed Bandits.* MobiHoc '21, arXiv:2007.03812v3 (local `2007.03812v3.pdf`).
Baseline component: Chawla, Sankararaman, Ganesh, Shakkottai, *Gossiping Insert-Eliminate* (AISTATS 2020), official code `https://github.com/ronshee/gosine` (`Synch.py`).

**This file is the only paper source later phases may read.** `2007.03812v3.pdf` is never read again. If a fact below is marked *not in paper*, it is our documented choice, not the paper's.

## 0. Provenance (grep map, `paper_html.txt` line numbers; LaTeX preserved as `alttext`)

| Fact | `paper_html.txt` lines |
|---|---|
| §2 Preliminaries (Bernoulli, gaps, regret def (1)) | 300–325 |
| §3 General algorithm prose (epochs, phases, blocklists) | 328–350 |
| **Algorithm 1** (lines 1–18) | 351–441 |
| **Algorithm 2** `Get-Rec` | 443–479 |
| §4 Chawla = no-blocking special case + footnote 3 (synchronous, sample from `{1..n+m}\{i}`) | 481–492 |
| Theorem 1 (statement) | 495–517 |
| Remark 2 (`Ω(K log T/Δ)` with m=1) | 518–521 |
| Remark 3 (sticky-set assumption, `S=⌈(K/n)log(1/ε)⌉`) | 525–527 |
| §5 + **Algorithm 3** `Update-Blocklists` | 592–626 |
| Remark 4 (growing blocking period) | 628–631 |
| Remark 6 (blocklist = `d^{(i)}(i')` counter, memory `(m+n)log T`) | 641–643 |
| Theorem 2 (upper bound, α condition) | 650–678 |
| Remark 7 (`min{m+K/n,K}`, improvement iff m small vs K) | 680–682 |
| Remark 8 (m=0 recovers Chawla) / Remark 9 (m≫n) / Remark 10 (η,β roles) | 686–700 |
| §6.1 Synthetic setup + strategies + Fig-1 discussion | 796–808 |
| Figure 1 caption (`m=10,K=100,η=2`) | 812 |
| §6.2 Real data (MovieLens pipeline) | 816–821 |
| Figure 2 caption (`m=15,K=100,η=2`) | 822 |
| Appendix D intro (Table 1 metric definition) | 3149–3152 |
| **Table 1** (12 rows × 4 columns) | 3155–3260 |

Figures 1 and 2 are raster images in the PDF: **no numeric axis data is extractable**. Figure comparison is qualitative + landmark-based (§11 below).

## 1. Model (§2)

- `K` arms labelled so `μ1 > μ2 ≥ … ≥ μK`; arm `k` gives `Bernoulli(μ_k)` rewards, independent across agents and pulls. Arm `1` = unique best arm. Gap `Δ_k = μ1 − μ_k ∈ (0,1)`.
- `n+m` agents on a **complete graph**; agents `1..n` honest, `n+1..n+m` malicious. Honest `i` faces its own instance of the same bandit and minimises
  `E R_T^{(i)} = Σ_{t=1..T} E Δ_{I_t^{(i)}}`  (eq. 1).
- Malicious agents need not run the algorithm; they only answer recommendations. They pull no arms and contribute no regret (our implementation choice; regret is defined only for honest agents).
- Honest agents do not know who is honest/malicious.

## 2. Algorithm 1 — `Multi-Agent-MAB-With-Blocking(α, β, Ŝ^(i))` (verbatim, lines 351–441)

Input: UCB parameter `α > 0`, phase-duration parameter `β > 1`, sticky set `Ŝ^(i) ⊆ {1..K}`, `|Ŝ^(i)| = S`.

```
1  Initialize A_{j'} = ⌈(j')^β⌉ ,  P_{j'}^{(i)} = ∅   ∀ j' ∈ ℕ
2  Set j = 1 ; let U_j^{(i)}, L_j^{(i)} be distinct elements of {1..K} \ Ŝ^{(i)} ;
   set S_j^{(i)} = Ŝ^{(i)} ∪ {U_j^{(i)}, L_j^{(i)}}
3  for t ∈ ℕ do
4      Pull I_t^{(i)} ∈ argmax_{k ∈ S_j^{(i)}}  μ̂_k^{(i)}(t−1) + sqrt( α log(t) / T_k^{(i)}(t−1) )
5      if t = A_j then
6          B_j^{(i)} = argmax_{k ∈ S_j^{(i)}}  T_k^{(i)}(A_j) − T_k^{(i)}(A_{j−1})   (most played active arm in this phase)
7          {P_{j'}^{(i)}}_{j'=j..∞} ← Update-Blocklist({P_{j'}^{(i)}}_{j'=j..∞})
             (algorithm from Chawla et al. performs no update; we propose using Algorithm 3)
8          (H_j^{(i)}, R_j^{(i)}) = Get-Rec(i, j, P_j^{(i)})                       (see Algorithm 2)
9          if R_j^{(i)} ∈ S_j^{(i)} (recommendation already active) then
10             S_{j+1}^{(i)} = S_j^{(i)}                                            (same active set)
11         else
12             U_{j+1}^{(i)} = argmax_{k ∈ {U_j^{(i)}, L_j^{(i)}}} T_k^{(i)}(A_j) − T_k^{(i)}(A_{j−1})   (most played non-sticky active arm)
13             L_{j+1}^{(i)} = R_j^{(i)}                                            (replace least played non-sticky arm with recommendation)
14             S_{j+1}^{(i)} = Ŝ^{(i)} ∪ {U_{j+1}^{(i)}, L_{j+1}^{(i)}}              (new active set)
15         end if
16         j ← j+1                                                                 (increment phase)
17     end if
18 end for
```

Notes: line 7 is printed `Update-Blocklist` in Alg. 1 and `Update-Blocklists` in Alg. 3 — same subroutine. `|S_j^{(i)}| = S + 2` always. The pull at line 4 happens **before** the epoch update, so the new active set takes effect at `t = A_j + 1`.

## 3. Algorithm 2 — `(H_j^{(i)}, R_j^{(i)}) = Get-Rec(i, j, P_j^{(i)})` (verbatim, lines 443–479)

Black box to honest agents (i does not execute it; i only observes its output).

```
1  Choose H_j^{(i)} from {1..n+m} \ (P_j^{(i)} ∪ {i}) uniformly at random (non-blocked agents), set i' = H_j^{(i)}
2  if i' ≤ n then
3      R_j^{(i)} = B_j^{(i')}                                (honest recommender: its most played arm this phase)
4  else
5      Sample R_j^{(i)} from ν_{j,i}^{(i')}                  (any F-measurable map Ω → distributions over {1..K})
6  end if
   Return: (H_j^{(i)}, R_j^{(i)})
```

Malicious strategy `{ν_{j,i}^{(i')}}_{j∈ℕ, i∈{1..n}}` may depend on the whole history and is indexed by the *receiving* honest agent `i` — i.e. a malicious agent may answer different honest agents differently.

## 4. Algorithm 3 — `Update-Blocklists` (verbatim, lines 598–626)

```
1  if j > 1 , B_j^{(i)} ≠ R_{j−1}^{(i)}   (previous recommendation not most played) then
2      for j' ∈ {j, ..., ⌈j^η⌉} do
3          P_{j'}^{(i)} ← P_{j'}^{(i)} ∪ {H_{j−1}^{(i)}}      (block the recommender)
4      end for
5  end if
   Return: {P_{j'}^{(i)}}_{j'=j..∞}
```

Blocking at epoch `j` punishes the agent contacted at epoch `j−1` (not the one contacted at `j`). Parameter `η > 1` sets blocking length. Remark 6: store as `d^{(i)}(i')` (initialised 0, overwritten with `⌈j^η⌉` when `i'` is blocked at phase `j`), then `P_j^{(i)} = {i' : d^{(i)}(i') ≥ j}`.

## 5. Implementation conventions (ambiguities resolved, not guessed silently)

| # | Ambiguity | Convention we implement | Basis |
|---|---|---|---|
| C1 | Line 4 at `t=1`: `log(1)=0`, `T_k(0)=0` → `0/0` | Any arm with `T_k(t−1)=0` gets index `+∞` (pulled first). Ties among unplayed arms → lowest index. | standard UCB; *not in paper* |
| C2 | `A_0` for the `j=1` phase counts | `A_0 := 0`, so `T_k(A_0)=0`. With `β=2`: `A_j = j²`, epochs at perfect squares; phase `j` = `(A_{j−1}, A_j]`. | implied by line 1; *boundary not in paper* |
| C3 | `argmax`/`argmin` ties | Deterministic lowest arm index (`numpy.argmax`/`argmin` first occurrence). Affects discard-bump location slightly. | *not in paper* |
| C4 | Is `B_j^{(i')}` available to `i` at the same epoch? | Yes — synchronous epochs: all agents share `A_j`, so `B_j^{(i')}` is computed at `t=A_j` and read by whoever contacts `i'` at that same epoch. | paper footnote 3 (line 489): "the synchronous algorithm in Chawla et al."; the asynchronous variant is out of scope |
| C5 | `U_j, L_j` initial choice (line 2) | Uniform random distinct pair from `{1..K} \ Ŝ^{(i)}`, drawn per trial per agent. | *not in paper* |
| C6 | Sticky sets | Per agent, independent uniform `S`-subsets of `{1..K}`; **resample the whole collection** until `1 ∈ ∪_i Ŝ^{(i)}`. Redrawn per trial (they are part of the trial instance). | §6.1 line 799 states resampling; per-trial redraw is our choice |
| C7 | Empty candidate set in `Get-Rec` line 1 (all others blocked) | Cannot be excluded a priori. Convention: no communication that epoch — set `R_j^{(i)} := B_j^{(i)}` (line 9 keeps the active set unchanged) and `H_j^{(i)}` unused, so Alg. 3 cannot block. Count occurrences in diagnostics. | *not in paper* |
| C8 | Blocking horizon overflow | `⌈j^η⌉` is capped at the last epoch `j_max = ⌈T^{1/β}⌉`. With `β=2, η=4`, any block at `j ≥ 18` is permanent within `T=1e5`; with `η=2` blocks expire. Store `until_epoch` (Remark 6), never a set of epochs. | Remark 6 + arithmetic |
| C9 | Malicious agents' own arm pulls / regret | Not simulated; malicious agents only answer `Get-Rec`. | regret defined only for honest agents (eq. 1) |
| C10 | Regret reported per agent or averaged | Per-trial average over honest agents: `(1/n) Σ_{i=1..n} R_T^{(i)}`; mean ± std across trials. | Appendix D line 3152 |
| C11 | Table 1 ratio: paired or unpaired | **Paired per-trial ratio**: run ours and Chawla on the same instance/trial seed, take `ratio_trial = R_ours/R_chawla`, report `mean ± std` of the 50 ratios. (Unpaired `mean(R_ours)/mean(R_chawla)` is also logged for reference; it has no std.) | Appendix D wording is ambiguous; pairing is what makes a ±0.1 std meaningful. *our choice* |
| C12 | Number of trials for Table 1 | 50 (stated for Figure 1 in §6.1; Appendix D does not restate it). | *assumption* |
| C13 | Instance seeds across Table 1 rows | Same trial seeds for the two algorithms in a row (required by C11). Same seed list reused across `(m,K,η)` rows so row-to-row differences are not seed noise. | *our choice* |
| C14 | Does the oracle also block honest agents? | Primary oracle: Algorithm 1 + Algorithm 3 **plus** permanent blocklist of `{n+1..n+m}` (i.e. `d(i')=∞` for malicious). Sensitivity variant (logged separately if cheap): honest-only sampling with **no** Algorithm 3 blocking. | paper gives no pseudocode (line 808); *our choice* |
| C15 | No-communication baseline | Each honest agent runs `UCB(α)` over **all K arms** (no sticky set, no active set, no communication), with the **same `α = 4`** as every other algorithm (C15a). | paper gives no pseudocode (line 808); *our choice, frozen 2026-10-12* — the alternative reading C15b was tested and rejected (§11.1) |
| C16 | Reward sampling | Bernoulli per pull from the arm mean (synthetic and real). | §2 + §6.2 |

## 6. Experiment parameters

Fixed for every experiment (§6.1 line 799, Appendix D line 3152):

| Param | Value |
|---|---|
| `n` | 25 |
| `β` | 2 |
| `α` | 4 |
| `S` | `⌈K/n⌉` (active set size `S+2`) → `K=75 ⇒ S=3`, `K=100 ⇒ S=4` |
| trials | 50, report mean ± std |
| `T` (Table 1) | `1e5` |
| `T` (Figures) | up to ≈ `1e5` |

Theorem-2 α condition `α > (3 + (1+βη)/β)/2`: `β=2` ⇒ `η=2 → 2.75`; `η=3 → 3.25`; `η=4 → 3.75`. **`α=4` satisfies it for all Table-1 rows** (tightest at `η=4`).

Synthetic arm means (§6.1): `μ1 = 0.95`, `μ2 = 0.85` (`Δ2 = 0.1`), `μ3..μK ~ U[0, 0.85]` i.i.d. (resampled per trial).

Figure configs: Fig 1 = `m=10, K=100, η=2`; Fig 2 = `m=15, K=100, η=2`.
Table 1 grid: `m ∈ {10,15} × K ∈ {75,100} × η ∈ {2,3,4}` × {synthetic, real} × {uniform, omniscient}.

## 7. Malicious strategies (§6.1 line 799, footnote 4 line 802)

- **uniform**: `ν_{j,i}^{(i')}` = uniform over `{1..K}` (may recommend arm 1 or an already-active arm → no active-set change).
- **omniscient**: `ν_{j,i}^{(i')}` = `argmin_{k' ∈ {2..K} \ S_j^{(i)}} T_{k'}^{(i)}(A_j)` — the **least-played inactive suboptimal arm of the receiving honest agent `i`**, using `i`'s *cumulative* counts at epoch `A_j` (not phase counts). "Omniscient" = it exploits `i`'s private information. Deterministic given `i`'s state; ties → lowest index (C3). It depends on `i`, so each malicious agent answers each honest agent separately.

## 8. The four compared algorithms

| Label | Definition |
|---|---|
| `proposed` | Algorithm 1 + Algorithm 3 (`η` per config) |
| `gossip` (Chawla) | Algorithm 1 with line 7 as a no-op (`P_j^{(i)} = ∅ ∀ j`) |
| `no_comm` | C15a: `UCB(α=4)` over all `K` arms, isolated (`baselines.run_no_comm`) |
| `no_comm_alg1` | C15b (**tested, rejected**): Algorithm 1 with the `Get-Rec` path removed — sticky sets, epochs and active-set updates kept, `R_j := B_j` (`engine.run_trial(..., isolated=True)`). Linear regret ⇒ not the paper's baseline (§11.1) |
| `oracle` | C14: Algorithm 1 + Algorithm 3 + malicious agents permanently blocked |

## 9. Metrics

- Per-agent cumulative regret: `R_T^{(i)} = Σ_{t=1..T} Δ_{I_t^{(i)}}` (expected regret is the same sum with `Δ` fixed by the true means; in simulation we accumulate `Δ_k` per pull, i.e. the *deterministic-Δ* regret, not sampled reward regret).
- Per-trial statistic: `(1/n) Σ_{i=1..n} R_T^{(i)}`.
- Figure curves: mean ± std over 50 trials vs `T` (log-x).
- Table 1 cell: paired per-trial ratio `proposed/gossip` at `T=1e5`, `mean ± std` (C11).

## 10. Table 1 — primary quantitative target (Appendix D, lines 3155–3260, transcribed)

Average regret at `T=1e5` relative to Chawla. Columns: Synth-uniform, Synth-omniscient, Real-uniform, Real-omniscient.

| m | K | η | Synth-unif | Synth-omni | Real-unif | Real-omni |
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

Tolerances (plan §4): match if `|ours − paper| ≤ 0.10`; within-noise if inside the paper's ±std; else mismatch → investigate.

## 11. Qualitative landmarks (Figures 1–2; §6.1 lines 808, 812; §6.2 lines 821–822)

1. `proposed` ≈ `oracle` ≪ `no_comm`; `gossip` ≫ `no_comm`. **Deviation (2026-10-12, `experiments/baseline_probe.py`, `results/baseline_probe_*.csv`):** we get `proposed` (2403) ≪ `no_comm` (10150) ✓ but `gossip` (7667) < `no_comm` at `K=100 n=25 S=4 m=10 η=2 T=1e5`, i.e. the second half of the landmark does not hold. Cause is the C15a constant, not the engine: `no_comm` regret is ~linear in `α` (2868 @α=1, 5467 @α=2, 10150 @α=4, 17461 @α=8), so the paper's ordering needs `α ≲ 2`. C15a stays frozen at `α = 4` (no post-hoc tuning) and the landmark is reported as a mismatch. The other reading, C15b, is ruled out by measurement: isolated Algorithm 1 has log-log slope **0.987** (23789 @T=1e5, 235092 @T=1e6) — an isolated sticky-set agent discards the best arm permanently, so its regret is linear and cannot be the paper's sublinear no-comm curve.
2. `proposed` < 0.5 × `gossip` regret (this is the Table 1 ratio ≈ 0.4–0.6).
3. "S-curve" for `gossip` under the **omniscient** strategy.
4. Honest agents **discard the best arm** at `T ≈ 6e4` (Fig 1, synthetic) and `T ≈ 4e4` (Fig 2, real) → regret bump near those `T`.
5. Uniform vs omniscient results "roughly similar" except the S-curve/bump.

## 12. Real-data pipeline (§6.2 lines 816–821)

1. MovieLens `ml-1m` (Harper & Konstan 2015). Movies = arms.
2. Extract the rating matrix restricted to users with the **same age, gender, and occupation**; keep users with **≥ 30 ratings**; keep movies rated **≥ 30 times** by that user set.
3. **Matrix completion** (Hastie et al. 2015, `softImpute`) to estimate missing entries. *Deviation:* no R/`rpy2`/`softImpute` in this env (STATE.md Phase 0) → implement soft-thresholded ALS/SVD in numpy; rank and λ are **not stated in the paper** → record ours.
4. Arm mean = fraction of (estimated) ratings `≥ 4` on the 1–5 scale; Bernoulli rewards.
5. Run the four algorithms with `K=100`, `m=15`, `η=2` (Fig 2); Table 1 real columns use the same grid as §6.
6. **Not in paper:** which cohort, and which 100 of the completed movies. Our choice: largest cohort passing the ≥30/≥30 filters, and arm selection recorded in `results/arm_means.csv`; repeat with a second cohort if time allows.

## 13. Theorems as empirical checks

- **Theorem 1** (lines 495–517): `m=1`, uniform strategy, Chawla algorithm, `1 ∈ ∪ Ŝ^{(i)}` ⇒ `liminf E R_T^{(i)}/log T ≥ α(1−1/√α)² Σ_{k=2..K} 1/Δ_k` (eq. 3); with equal gaps `Ω(K log T/Δ)` = single-agent baseline. Smoke test (plan §4 Phase 4): `m=1, K=25, T=1e4` ⇒ `gossip/no_comm` ratio within ~1.3.
- **Theorem 2** (lines 650–678): `β>1`, `α > (3+(1+βη)/β)/2`, `η>1`, `1 ∈ ∪ Ŝ^{(i)}` ⇒ `E R_T^{(i)} ≤ 4α min{ (2η−1)/(η−1) Σ_{k=2..m+3} 1/Δ_k + Σ_{k=m+4..S+m+4} 1/Δ_k , Σ_{k=2..K} 1/Δ_k } log T + C*`, with `Δ_k := 1` for `k > K`.
- **Remark 7** (line 682): with `S=⌈(K/n)log(1/ε)⌉` and equal gaps, regret `O(min{m+K/n, K} log T/Δ)`; collaboration helps iff `m` small vs `K`. Experiments use `S=⌈K/n⌉`, not the theory value.
- **Remark 8/9/10** (686–700): `m=0` recovers Chawla's `O((K/n)log T/Δ)` up to `(2η−1)/(η−1)`; robust even when `m ≫ n`; long-term `log T` coefficient decreases with `η`, independent of `β`.

## 14. Engine state required by Phase 2 (derived from §2–§4, for implementation only)

Per honest agent `i`: `T_k` (counts, `K`), `sum_k` (reward sums → `μ̂`), sticky mask (`K`), active list (`S+2` arms), `U`, `L`, `B_j`, `H_{j−1}`, `R_{j−1}`, `d[i']` (block-until-epoch, `n+m` entries).
Per epoch `A_j` (all agents, synchronous): compute `B_j^{(i)}` from phase-count deltas → apply Alg. 3 → sample `H_j^{(i)}` from non-blocked → resolve `R_j^{(i)}` (honest: `B_j^{(H)}`; malicious: strategy, omniscient needs the receiver's `T_k(A_j)`) → update `U/L/active`.
Epochs to `T=1e5, β=2`: `j_max = 317`. Inner loop argmax is over `S+2` arms, not `K`.
Numba caveat from Phase 0: `np.random.default_rng` unsupported in `@njit` — use legacy `np.random.seed` per kernel or pre-generate uniforms outside.
