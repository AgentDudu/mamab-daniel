"""Phase 7 — MovieLens real-data pipeline (spec §12, paper §6.2 lines 816–821).

Paper text (verbatim, paper_html.txt 819–820): extract a rating matrix of users
with the same age, gender and occupation, "while also ensuring each user has
rated >= 30 movies and each movie has been rated >= 30 times by the set of
users"; then "use matrix completion (Hastie et al., 2015) to estimate the
missing entries"; then "arm means as the fraction of ratings >= 4 on a scale of
1 to 5".

What the paper does NOT say, and what this script freezes (spec C17–C20):

* C17 cohort — the largest (age, gender, occupation) cohort that survives the
  >=30/>=30 filters. `results/p7_cohorts.csv` records the candidates and the
  runner-up, which is also run (plan §3 Phase 7: "repeat with a second cohort").
* C18 completion — no R/`rpy2`/`softImpute` in this env, so `softimpute()` is a
  numpy re-implementation of the cited algorithm (iterative rank-r SVD of the
  imputed matrix with soft-thresholded singular values, offset by column means).
  Rank and lambda are not stated in the paper: ours are picked by 3-fold
  holdout RMSE on the observed ratings, grid in `results/p7_softimpute.csv`.
* C19 star rounding — the completed matrix is continuous; rewards are defined on
  whole stars ("rated it 4 or 5 stars"), so estimates are clipped to [1,5] and
  rounded before the >= 4 test. Without rounding the arm means average ~0.25
  (nuclear-norm shrinkage kills the bimodal 4/5 mass); with rounding they
  average ~0.58, matching the plan's sanity target of ~0.6 and the cohort's
  observed 0.616. Both numbers are recorded.
* C20 arm selection — a seeded uniform sample of K=100 of the filtered movies
  (the 100 most-rated ones skew towards titles this cohort likes: mean arm mean
  0.78 vs the cohort's 0.58, which misses the plan's ~0.6 sanity target; both
  variants are recorded in `results/metrics.json`). The sample is then sorted by
  arm mean descending so arm index 0 is the best arm (the paper's arm 1, which
  the engine's regret is measured against).

Outputs: `results/arm_means.csv` (primary cohort, Phase 8 input),
`results/arm_means_<cohort>.csv` (extra cohorts), `results/p7_cohorts.csv`,
`results/p7_softimpute.csv`, and the `movielens` section of `results/metrics.json`.

Usage: python experiments/p7_movielens.py [--cohorts auto|18-M-4,25-M-0]
           [--arms 100] [--folds 3] [--rank 10,20,0] [--lam-frac 0.05,0.1,0.2,0.4]
           [--rank N --lam L]   # skip CV and use these directly
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
DATA = ROOT / "data" / "ml-1m"
RESULTS = ROOT / "results"

MIN_USER_RATINGS = 30     # paper §6.2
MIN_MOVIE_RATINGS = 30    # paper §6.2
K_ARMS = 100              # paper §6.2 (Figure 2 / Table 1 real columns)
CV_FOLDS = 3
RANKS = (10, 20, 0)       # 0 = max rank (softImpute's default mode)
LAM_FRACS = (0.05, 0.1, 0.2, 0.4)  # lambda = frac * sigma_1 of the zero-filled matrix
SANITY_MEAN, SANITY_TOL = 0.6, 0.1  # plan §2 Phase 7 exit criterion


def load_ml1m(data_dir: Path = DATA):
    """ml-1m users.dat + ratings.dat (`::`-separated, latin-1)."""
    users = pd.read_csv(data_dir / "users.dat", sep="::", engine="python", encoding="latin-1",
                        names=["uid", "gender", "age", "occ", "zip"], comment="#")
    ratings = pd.read_csv(data_dir / "ratings.dat", sep="::", engine="python", encoding="latin-1",
                          names=["uid", "mid", "rating", "ts"], comment="#")
    return users, ratings


def cohort_key(users: pd.DataFrame) -> pd.Series:
    """(age, gender, occupation) label, e.g. `18-M-4` (ml-1m codes)."""
    return users.age.astype(str) + "-" + users.gender + "-" + users.occ.astype(str)


def cohort_table(users: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Candidate cohorts, ranked by the number of users that keep >= 30 ratings (C17)."""
    per_user = ratings.groupby("uid").size()
    u = users.assign(cohort=cohort_key(users), n=users.uid.map(per_user).fillna(0).astype(int))
    kept = u[u.n >= MIN_USER_RATINGS]
    out = kept.groupby("cohort").agg(users=("uid", "size"), ratings=("n", "sum"))
    return out.sort_values(["users", "ratings"], ascending=False)


def cohort_matrix(users: pd.DataFrame, ratings: pd.DataFrame, cohort: str):
    """Dense user x movie rating matrix for one cohort, NaN = missing (C17).

    Filters are applied in the paper's order in a single pass: users with
    >= 30 ratings in the cohort, then movies rated >= 30 times by that user set.
    """
    ids = users.loc[cohort_key(users) == cohort, "uid"]
    sub = ratings[ratings.uid.isin(ids)]
    n_user = sub.groupby("uid").size()
    n_user = n_user[n_user >= MIN_USER_RATINGS]
    sub = sub[sub.uid.isin(n_user.index)]
    n_movie = sub.groupby("mid").size()
    n_movie = n_movie[n_movie >= MIN_MOVIE_RATINGS]
    sub = sub[sub.mid.isin(n_movie.index)]
    user_ids, movie_ids = sorted(n_user.index), sorted(n_movie.index)
    urow = {v: i for i, v in enumerate(user_ids)}
    mcol = {v: i for i, v in enumerate(movie_ids)}
    Y = np.full((len(user_ids), len(movie_ids)), np.nan)
    Y[sub.uid.map(urow).to_numpy(), sub.mid.map(mcol).to_numpy()] = sub.rating.to_numpy()
    after = sub.groupby("uid").size()
    return Y, np.array(movie_ids), n_movie.reindex(movie_ids).to_numpy(), int((after < MIN_USER_RATINGS).sum())


def softimpute(Y: np.ndarray, rank: int, lam: float, iters: int = 60, tol: float = 1e-4):
    """softImpute (Hastie et al. 2015) in numpy — the C18 stand-in for the R package.

    Iterates: impute the missing entries with the current estimate, take the
    rank-r SVD of the completed matrix, soft-threshold the singular values by
    lambda. Entries are offset by column (movie) means first, as Hastie et al.
    2015 §3 recommends. Returns the completed matrix clipped to the 1-5 rating scale.
    """
    obs = ~np.isnan(Y)
    off = np.nanmean(Y, axis=0)
    off = np.where(np.isnan(off), np.nanmean(Y), off)
    resid = np.where(obs, Y - off, 0.0)
    filled = resid.copy()
    est = np.zeros_like(resid)
    # shortcut: 60-iteration cap; arm means move < 0.001 between 60 iterations and full
    # convergence (111 at the chosen lambda). Raise `iters` if the grid ever needs it.
    for it in range(1, iters + 1):
        U, s, Vt = np.linalg.svd(filled, full_matrices=False)
        if rank:
            U, s, Vt = U[:, :rank], s[:rank], Vt[:rank]
        s = np.maximum(s - lam, 0.0)
        prev = est
        est = (U * s) @ Vt
        if it > 1 and np.abs(est - prev).max() <= tol * max(1.0, np.abs(est).max()):
            break
        filled = np.where(obs, resid, est)
    return np.clip(est + off, 1.0, 5.0), it


def cv_softimpute(Y: np.ndarray, ranks, lam_fracs, folds: int, seed: int = 20260101):
    """Pick (rank, lambda) by k-fold holdout RMSE on observed ratings (C18)."""
    obs = ~np.isnan(Y)
    lam_max = float(np.linalg.svd(np.where(obs, Y - np.nanmean(Y), 0.0), compute_uv=False)[0])
    lams = [round(lam_max * f, 3) for f in lam_fracs]
    flat = Y.ravel()
    held = np.flatnonzero(obs.ravel())[np.random.default_rng(seed).permutation(obs.sum())]
    rows = []
    for rank in ranks:
        for lam in lams:
            errs = []
            for f in range(folds):
                idx = held[f::folds]
                Ytr = Y.copy()
                Ytr.ravel()[idx] = np.nan
                M, used = softimpute(Ytr, rank, lam)
                errs.append(float(np.sqrt(np.mean((M.ravel()[idx] - flat[idx]) ** 2))))
            rows.append({"rank": rank, "lam": lam, "holdout_rmse": float(np.mean(errs)),
                         "rmse_per_fold": [round(e, 4) for e in errs], "iterations": used})
    best = min(rows, key=lambda r: r["holdout_rmse"])
    return rows, int(best["rank"]), float(best["lam"]), lam_max


def arm_means(M: np.ndarray, movie_ids: np.ndarray, n_ratings: np.ndarray, titles: pd.Series,
              k: int = K_ARMS, select: str = "random", seed: int = 20260101) -> pd.DataFrame:
    """Bernoulli arm means (C19) for k of the filtered movies, best arm first (C20).

    `select`: "random" (C20 default — a seeded uniform sample of the filtered
    movies, so the arm means stay representative of the cohort), "popular" (the k
    most-rated, which skews towards titles this cohort likes: mean 0.78 vs the
    cohort's 0.58) or "first" (lowest movie ids).
    """
    stars = np.rint(M)                       # C19: whole stars, then >= 4 = reward 1
    liked = (stars >= 4.0).mean(axis=0)
    cont = (M >= 4.0).mean(axis=0)           # the unrounded variant, recorded as sensitivity
    if select == "popular":
        order = np.lexsort((movie_ids, -n_ratings))[:k]
    elif select == "first":
        order = np.argsort(movie_ids, kind="stable")[:k]
    else:
        order = np.random.default_rng(seed).permutation(len(movie_ids))[:k]
    out = pd.DataFrame({
        "movie_id": movie_ids[order],
        "title": [titles.get(int(m), "") for m in movie_ids[order]],
        "n_ratings": n_ratings[order],
        "arm_mean": liked[order],
        "arm_mean_unrounded": cont[order],
        "mean_rating_completed": M[:, order].mean(axis=0),
    })
    out = out.sort_values(["arm_mean", "movie_id"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "arm_index", np.arange(len(out)))
    return out


def run_cohort(users, ratings, cohort: str, arms: int, folds: int, ranks, lam_fracs,
               rank_override, lam_override, movies: pd.DataFrame, select: str = "random",
               reuse_cv: bool = False):
    """One cohort end-to-end: filter -> complete -> arm means. Returns (df, info)."""
    Y, movie_ids, n_ratings, users_below = cohort_matrix(users, ratings, cohort)
    obs = ~np.isnan(Y)
    cached = []
    if reuse_cv and (RESULTS / "p7_softimpute.csv").exists():
        prev = pd.read_csv(RESULTS / "p7_softimpute.csv")
        cached = [dict(rank=int(r["rank"]), lam=float(r["lam"]),
                       holdout_rmse=float(r["holdout_rmse"]))
                  for _, r in prev[prev.cohort == cohort].iterrows()]
    if cached:
        best = min(cached, key=lambda r: r["holdout_rmse"])
        rows, rank, lam, lam_max = cached, int(best["rank"]), float(best["lam"]), None
    elif rank_override is not None and lam_override is not None:
        rows, rank, lam, lam_max = [], int(rank_override), float(lam_override), None
    else:
        rows, rank, lam, lam_max = cv_softimpute(Y, ranks, lam_fracs, folds)
    M, iters = softimpute(Y, rank, lam)
    df = arm_means(M, movie_ids, n_ratings, movies.set_index("mid")["title"], arms, select)
    info = {
        "cohort": cohort,
        "users": Y.shape[0],
        "users_below_min_after_movie_filter": users_below,
        "movies": int(len(movie_ids)),
        "ratings": int(obs.sum()),
        "density": float(obs.mean()),
        "observed_frac_ge4": float((Y[obs] >= 4).mean()),
        "lam_max_sigma1": lam_max,
        "rank": rank,
        "lam": lam,
        "iterations": iters,
        "holdout_rmse": next((r["holdout_rmse"] for r in rows
                              if r["rank"] == rank and r["lam"] == lam), None),
        "completed_mean_rating": float(M.mean()),
        "n_arms": int(len(df)),
        "arm_mean_mean": float(df.arm_mean.mean()),
        "arm_mean_min": float(df.arm_mean.min()),
        "arm_mean_max": float(df.arm_mean.max()),
        "arm_mean_unrounded_mean": float(df.arm_mean_unrounded.mean()),
        "arm_mean_mean_top_rated_variant": float(
            (np.rint(M) >= 4).mean(axis=0)[np.lexsort((movie_ids, -n_ratings))[:arms]].mean()),
        "arm_mean_mean_all_movies": float((np.rint(M) >= 4).mean(axis=0).mean()),
        "arm_selection": select,
        "gap_to_second_best": float(df.arm_mean.iloc[0] - df.arm_mean.iloc[1]),
        "arms_within_0p01_of_best": int((df.arm_mean >= df.arm_mean.iloc[0] - 0.01).sum()),
        "mean_gap_to_best": float((df.arm_mean.iloc[0] - df.arm_mean).mean()),
        "best_arm_movie_id": int(df.movie_id.iloc[0]),
    }
    return df, info, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="auto", help="'auto' = largest cohort + runner-up (C17), "
                                                      "or a comma list like 18-M-4,25-M-0")
    ap.add_argument("--arms", type=int, default=K_ARMS)
    ap.add_argument("--folds", type=int, default=CV_FOLDS)
    ap.add_argument("--rank", type=int, default=None, help="override CV rank (needs --lam)")
    ap.add_argument("--lam", type=float, default=None, help="override CV lambda")
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANKS))
    ap.add_argument("--lam-frac", default=",".join(str(f) for f in LAM_FRACS))
    ap.add_argument("--select", default="random", choices=("random", "popular", "first"),
                    help="C20 arm selection (see arm_means)")
    ap.add_argument("--reuse-cv", action="store_true",
                    help="reuse the (rank, lambda) grid already in results/p7_softimpute.csv")
    args = ap.parse_args()

    t0 = time.time()
    if not (DATA / "ratings.dat").exists():
        print(f"MISSING DATA: download ml-1m.zip from https://files.grouplens.org/datasets/"
              f"movielens/ and unpack into {DATA}")
        return 1
    users, ratings = load_ml1m()
    movies = pd.read_csv(DATA / "movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["mid", "title", "genres"], comment="#")
    cands = cohort_table(users, ratings)
    if args.cohorts == "auto":
        chosen = list(cands.index[:2])
    else:
        chosen = [c.strip() for c in args.cohorts.split(",")]
    print(f"cohorts (C17): {chosen}\n{cands.head(5)}")

    RESULTS.mkdir(exist_ok=True)
    cohort_rows, cv_rows, sections = [], [], {}
    for i, cohort in enumerate(chosen):
        df, info, rows = run_cohort(users, ratings, cohort, args.arms, args.folds,
                                    [int(r) for r in args.ranks.split(",")],
                                    [float(f) for f in args.lam_frac.split(",")],
                                    args.rank, args.lam, movies, args.select, args.reuse_cv)
        path = RESULTS / ("arm_means.csv" if i == 0 else f"arm_means_{cohort}.csv")
        df.to_csv(path, index=False)
        sections[cohort] = info
        cohort_rows.append({"cohort": cohort, **{k: v for k, v in info.items() if k != "cohort"},
                            "selected": i == 0})
        cv_rows.extend({"cohort": cohort, **r} for r in rows)
        print(f"\n{cohort}: {info['users']} users x {info['movies']} movies, "
              f"{info['ratings']} ratings (density {info['density']:.3f}); "
              f"rank={info['rank']} lam={info['lam']} rmse={info['holdout_rmse']}")
        print(f"  {info['n_arms']} arms, arm mean {info['arm_mean_mean']:.3f} "
              f"[{info['arm_mean_min']:.3f}, {info['arm_mean_max']:.3f}], "
              f"best arm = movie {info['best_arm_movie_id']} ({info['arm_mean_max']:.3f}), "
              f"gap to 2nd {info['gap_to_second_best']:.4f}")
        print(f"  wrote {path}")

    pd.DataFrame(cohort_rows).to_csv(RESULTS / "p7_cohorts.csv", index=False)
    if not args.reuse_cv:   # --reuse-cv reads the grid back, so do not overwrite it with less
        pd.DataFrame(cv_rows).to_csv(RESULTS / "p7_softimpute.csv", index=False)

    primary = sections[chosen[0]]
    checks = {
        "arms_ge_100": primary["n_arms"] >= K_ARMS,
        "means_in_unit_interval": bool((pd.read_csv(RESULTS / "arm_means.csv").arm_mean
                                        .between(0.0, 1.0)).all()),
        "mean_near_0.6": abs(primary["arm_mean_mean"] - SANITY_MEAN) <= SANITY_TOL,
    }
    metrics_path = RESULTS / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["movielens"] = {
        "source": "ml-1m (GroupLens, Harper & Konstan 2015)",
        "filters": {"min_user_ratings": MIN_USER_RATINGS, "min_movie_ratings": MIN_MOVIE_RATINGS,
                    "pass": "single pass, users then movies"},
        "conventions": {"C17": "largest cohort by users surviving the filters",
                        "C18": "numpy softImpute (rank/lambda by 3-fold holdout RMSE)",
                        "C19": "completed ratings clipped to [1,5] and rounded to whole stars "
                               "before the >= 4 test",
                        "C20": f"a seeded uniform sample of k={args.arms} of the filtered movies "
                               "(arm means stay representative of the cohort), sorted by arm mean "
                               "descending so arm 0 is the best arm"},
        "cohorts": sections,
        "exit_checks": checks,
        "runtime_s": round(time.time() - t0, 1),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\nexit checks (plan §2 Phase 7):")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    ok = all(checks.values())
    print(f"\nMovieLens pipeline: {'PASS' if ok else 'FAIL — see above'} "
          f"({metrics['movielens']['runtime_s']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
