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
