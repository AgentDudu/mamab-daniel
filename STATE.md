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
