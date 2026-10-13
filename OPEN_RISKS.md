# Open risks / follow-up work

Paper: *Robust Multi-Agent Multi-Armed Bandits* (Vial, Shakkottai, Srikant, MobiHoc '21, arXiv:2007.03812v3).
Written 2026-10-13 after the real-data run (`experiments/p8_real.py`, merged in `4aaf3b5`).
Contract for all phases: `specs/paper_spec.md` (conventions C1–C21). Phase log: `STATE.md`.

Rule that still applies: report deviations, never tune a convention to make a cell match.

---

## Already closed

| Risk | Status |
|---|---|
| Engine refactor (`make_instance_from_means`) changing earlier synthetic results | **Closed 2026-10-13.** Re-ran cell `m=10 K=75 η=2 uniform` (50 trials, T=1e5) into a temp dir after the refactor → byte-identical to `results/table1_raw_m10_K75_eta2_uniform.csv`. `results/metrics.json` diff was additions-only. |
| MovieLens filters yielding < 100 usable arms (plan §7 stop) | **Closed** — cohort `18-M-4` gives 100 arms, mean arm mean 0.566 (spec C17). |

---

## 1. C21 — the `K = 75` real arm subset moves the numbers most

- **What:** `K=75` real configs take the **top-75** entries of the C20 sorted arm-mean vector. The paper never states the subset.
- **Effect:** the whole `K=75` omniscient family sits at ratio 0.70–0.79 (median 0.712, 98% of trials > 0.6) vs paper 0.543–0.645. Cause is the denominator: top-75 raises mean arm mean to 0.741 and shrinks `gossip` regret to 3244 (vs 4215 at `K=100`) while `proposed` stays ≈ 2300. It is the driver of 1 of the 2 real mismatches and of 4 "within-noise" cells.
- **Check:** rerun `--K 75` with a different subset rule — uniform 75-of-100 sample, movie-id order, or bottom-75. `p8_real.arm_means()` only truncates, so this needs a loader flag (e.g. `--subset top|sample|id`).
- **Effort:** small (loader flag + one 1.5e9-pull rerun).

## 2. Best-arm mean is exactly 1.000

- **What:** C19 (round to whole stars, then `≥ 4`) + C20 (top selection) give `μ₀ = 1.000` exactly, so the best arm is Bernoulli(1) — zero reward noise on the arm every agent must identify. The synthetic instances use `μ₁ = 0.95`.
- **Effect (suspected):** UCB exploration and the sticky/active-set churn behave differently than in the paper's instance; may contribute to the real-vs-synthetic std asymmetry (risk 9) and to the discard behaviour in risk 6.
- **Check:** count arms with mean 1.0 in `results/arm_means.csv`; rerun the Figure-2 config with means capped at 0.99 and compare ratio + discard rate.
- **Effort:** small.

## 3. Matrix completion is our numpy re-implementation, not `softImpute`

- **What:** no R / `rpy2` / `softImpute` in this env (spec §12.3, C18). Ours: impute → rank-r SVD → soft-threshold singular values by λ, column-mean offset, clip [1,5]; max rank, λ = 7.807 by 3-fold holdout RMSE (0.921).
- **Effect:** arm means are the entire real-data input. Sensitivity already seen: thresholding the *unrounded* estimates gives mean arm mean 0.25 instead of 0.566 (C19). Paired ratios are partially robust; absolute regrets are not.
- **Check:** run R `softImpute` on cohort `18-M-4` and diff the arm-mean vector; or perturb rank/λ (grid already in `results/p7_softimpute.csv`) and rerun one real cell per setting.
- **Effort:** medium (needs R or a second completion implementation).

## 4. Cohort choice probed at n = 2 only

- **What:** plan §6 risk 1 calls this the biggest non-reproducibility source. We ran `18-M-4` (primary) and `25-M-0` (runner-up): Figure-2 ratios 0.485 / 0.580 vs 0.419 / 0.572 → cohort moves the ratio ≤ 0.07.
- **Effect:** ml-1m has ~84 (age, gender, occupation) cells; the paper's real-column ±std is 0.058–0.229, so a third cohort could push cells across the 0.10 tolerance in either direction.
- **Check:** `experiments/p7_movielens.py --cohorts <n>` (arm means only, ~10 min with `--reuse-cv`), then `p8_real.py --arm-means <file> --no-table` for the Figure-2 config per cohort.
- **Effort:** medium; mostly compute.

## 5. Figure 2 omniscient gate fails structurally, not by noise

- **What:** paired `proposed/gossip` = **0.580 ± 0.046** omniscient vs the `< 0.5` landmark (uniform 0.485 ± 0.074 passes). η = 3 and η = 4 give 0.576 and 0.580 — the η sweep does not fix it.
- **Related:** the synthetic side shows the same signature in the opposite direction (`m=10 K=100 η=2` omniscient 0.290 vs paper 0.401, `m=15 K=75 η=3` 0.549 vs 0.439). Our blocking behaves differently from the paper's specifically under the **omniscient** strategy.
- **Check:** per-trial bimodality + `diag` counters (risk 6) for the omniscient cells; test the omniscient strategy's tie-breaking and its use of *cumulative* vs *phase* counts (spec §7 — the paper says cumulative; a phase-count reading would change the attack's strength).
- **Effort:** medium; this is the most interesting open scientific question.

## 6. Blocking mechanism does not reproduce on real data — and `diag` is not collected

- **What:** Algorithm 3 stops best-arm discards on synthetic data (rate → ~0 after `T ≈ 1.3e4`) but only **halves** them on real data (proposed 0.0138 vs gossip 0.0300 per agent/epoch uniform; 0.0108 vs 0.0358 omniscient; cumulative 5.2 vs 9.4 per agent).
- **Gap:** `p8_real.run_fig_config` does not pass `diag=`, so the C7 counters (no-recommender epochs, malicious recommendations absorbed) are missing for every real-data config. Phase 4 collected them (`m=10 K=100 η=2` synthetic: 3601 / 6899).
- **Check:** add a `diag` column to `fig2_raw_*` (or a side CSV) and compare real vs synthetic blocking activity; also check whether the near-tied top arms (Δ₂ = 0.0029, 7 arms within 0.01) make `B_j ≠ R_{j−1}` fire spuriously.
- **Effort:** small (one kwarg + a column), needs a rerun of the 8 figure configs (~1e9 pulls).

## 7. Harness footgun: subset runs clobber `results/metrics.json`

- **What:** any filtered run (`--m 10 --no-fig`) with the default `--out` rewrites the `fig2` / `table1_real` metrics sections with `missing` entries, and parallel processes race on the same file.
- **Workaround used:** parallel compute wrote `logs/p8_*.json` (deleted afterwards), then one final unfiltered run owned `results/metrics.json`. Nothing enforces this.
- **Check/fix:** make `--out` default to a per-run file, or merge metrics sections instead of replacing them; document the two-step parallel recipe in the script docstring.
- **Effort:** small.

## 8. `results/fig2.png` is gitignored

- **What:** `.gitignore` excludes `*.png` (figures are regenerable). `results/fig2.csv` + `results/fig2_raw_*.csv` are committed, the PNG is not.
- **Effect:** the Phase 9 report will cite a figure that is not in the repo; reproducing it needs matplotlib + the raw CSVs (`p8_real.py --aggregate-only`).
- **Check/fix:** decide before Phase 9 — either force-add the PNGs (`git add -f results/fig2.png results/fig1.png`) or state in the report that figures are regenerated by command.
- **Effort:** trivial.

## 9. The paper's ±std is not reproducible in either direction

- **What:** synthetic ratio stds — ours 0.086–0.186 vs paper 0.034–0.114 (ours ~2× larger). Real ratio stds — ours 0.038–0.080 vs paper 0.058–0.229 (ours ~2× smaller). Same harness, opposite sign.
- **Implication:** C11 (paired per-trial ratio) and C12 (50 trials) are our assumptions, not the paper's; the paper's ±std may come from a different trial count, an unpaired ratio, or per-agent rather than per-trial aggregation.
- **Check:** the `unpaired_ratio` column already in `results/table1_real.csv` / `table1_synth.csv`; also try 25 and 100 trials on two cells and see which std band the paper's numbers sit in.
- **Effort:** small–medium (compute only, no code).

## 10. Pre-existing, still open: the `gossip ≫ no_comm` landmark

- **What:** `no_comm` is C15a (`UCB(α=4)` over all K arms, isolated) because the paper gives no pseudocode. Measured `gossip/no_comm` = 0.67–0.76 synthetic, 0.60–0.76 real — the paper shows the opposite ordering. `no_comm` regret is ~linear in α (2868 @α=1 … 17461 @α=8), so the paper's ordering needs α ≲ 2. C15b (Algorithm 1 minus `Get-Rec`) is ruled out: log-log slope 0.987 ⇒ linear regret.
- **Decision frozen:** keep α = 4, report as a mismatch, do not tune the baseline to fit.
- **Check (optional):** ask the paper's authors which no-comm baseline they used; otherwise it stays a documented mismatch in Phase 9.

## 11. Not yet done at all

- **Phase 9:** `REPLICATION_REPORT.md` — paper value | replicated value | delta | verdict for Figure 1, Figure 2 and all 48 Table 1 cells (24 synthetic in `results/table1_synth.csv`, 24 real in `results/table1_real.csv`), plus the landmark verdicts and the α-sensitivity text.
- **Remark 7 scaling check** (plan §4): regret-vs-log-T slope behaviour, and `m` small vs `K` ⇒ improvement, `m` comparable to `K` ⇒ none. Only the `m ∈ {10,15}` pair has been run; no `m` sweep at fixed `K`.
