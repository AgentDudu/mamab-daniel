"""Phase 7 guards: the >=30/>=30 cohort filters, softImpute, and the arm-mean mapping.

The synthetic fixtures pin the conventions the paper leaves open (spec C18–C20);
the last test runs the real ml-1m cohort end-to-end when the dataset is present.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import p7_movielens as p7  # noqa: E402


def _fixture():
    """Cohort 25-M-0: 40 users x 30 movies (all pass), u41 with 29 ratings, m31 with 5."""
    users, rows = [], []
    for uid in range(1, 41):
        users.append((uid, "M", 25, 0, "00000"))
        rows += [(uid, mid, 4, 0) for mid in range(1, 31)]
    users.append((41, "M", 25, 0, "00000"))
    rows += [(41, mid, 4, 0) for mid in range(1, 30)]          # 29 ratings -> dropped
    users.append((42, "F", 25, 0, "00000"))                    # other cohort
    rows += [(42, mid, 5, 0) for mid in range(1, 31)]
    rows += [(uid, 31, 5, 0) for uid in range(1, 6)]           # 5 ratings -> dropped
    users = pd.DataFrame(users, columns=["uid", "gender", "age", "occ", "zip"])
    ratings = pd.DataFrame(rows, columns=["uid", "mid", "rating", "ts"])
    return users, ratings


def test_cohort_table_ranks_by_users_with_30_ratings():
    users, ratings = _fixture()
    table = p7.cohort_table(users, ratings)
    assert list(table.index) == ["25-M-0", "25-F-0"]
    assert table.users.tolist() == [40, 1]  # u41 fails the >=30 user filter


def test_cohort_matrix_applies_the_30_and_30_filters():
    users, ratings = _fixture()
    Y, movie_ids, n_ratings, users_below = p7.cohort_matrix(users, ratings, "25-M-0")
    assert Y.shape == (40, 30)
    assert movie_ids.tolist() == list(range(1, 31))   # movie 31 fails the >=30 movie filter
    assert n_ratings.tolist() == [40] * 30
    assert users_below == 0
    assert np.isnan(Y).sum() == 0


def test_softimpute_recovers_a_low_rank_matrix():
    rng = np.random.default_rng(7)
    truth = np.clip(rng.normal(size=(40, 2)) @ rng.normal(size=(2, 30)) * 0.8 + 3.5, 1, 5)
    Y = truth.copy()
    Y[rng.random(Y.shape) < 0.3] = np.nan
    M, _ = p7.softimpute(Y, rank=2, lam=0.5)
    hidden = np.isnan(Y)
    assert np.sqrt(np.mean((M[hidden] - truth[hidden]) ** 2)) < 0.5
    assert M.min() >= 1.0 and M.max() <= 5.0


def test_softimpute_above_lambda_max_falls_back_to_column_means():
    truth = np.tile(np.arange(1.0, 6.0), (20, 1))
    Y = truth.copy()
    Y[::2, ::2] = np.nan
    M, _ = p7.softimpute(Y, rank=0, lam=1e6)
    assert np.allclose(M, truth)  # every column's observed mean is that column's value


def test_arm_means_round_stars_then_threshold():
    """C19: 3.6 estimated stars counts as a 4-star reward; 3.4 does not."""
    M = np.array([[3.6, 3.4, 4.4, 3.4]] * 10)
    ids = np.array([5, 6, 7, 8])
    titles = pd.Series(["a", "b", "c", "d"], index=ids)
    df = p7.arm_means(M, ids, np.array([50, 40, 30, 20]), titles, k=4)
    assert df.arm_mean.tolist() == [1.0, 1.0, 0.0, 0.0]        # rounded to whole stars
    assert df.arm_mean_unrounded.tolist() == [0.0, 1.0, 0.0, 0.0]  # unrounded, in the same order
    assert df.movie_id.tolist() == [5, 7, 6, 8]  # best arm first, ties by movie id


def test_arm_selection_variants():
    M = np.full((10, 5), 4.5)
    ids = np.array([10, 11, 12, 13, 14])
    titles = pd.Series(index=ids, dtype=object)
    n = np.array([5, 9, 30, 30, 4])
    popular = p7.arm_means(M, ids, n, titles, k=3, select="popular")
    assert sorted(popular.movie_id.tolist()) == [11, 12, 13]  # top-3 by rating count
    first = p7.arm_means(M, ids, n, titles, k=3, select="first")
    assert sorted(first.movie_id.tolist()) == [10, 11, 12]    # lowest movie ids
    sample = p7.arm_means(M, ids, n, titles, k=3)
    assert sorted(sample.movie_id.tolist()) == sorted(p7.arm_means(M, ids, n, titles, k=3).movie_id)
    assert len(set(sample.movie_id)) == 3                     # a sample, no repeats


DATA = Path(__file__).resolve().parents[1] / "data" / "ml-1m"


@pytest.mark.skipif(not (DATA / "ratings.dat").exists(), reason="ml-1m not downloaded")
def test_real_cohort_yields_100_arms_with_means_near_0_6():
    users, ratings = p7.load_ml1m()
    cohort = p7.cohort_table(users, ratings).index[0]
    Y, movie_ids, n_ratings, _ = p7.cohort_matrix(users, ratings, cohort)
    M, _ = p7.softimpute(Y, rank=20, lam=20.0)
    movies = pd.read_csv(DATA / "movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["mid", "title", "genres"], comment="#")
    df = p7.arm_means(M, movie_ids, n_ratings, movies.set_index("mid")["title"], k=100)
    assert len(df) >= 100                              # plan §7 stop condition
    assert df.arm_mean.between(0.0, 1.0).all()
    assert 0.45 <= df.arm_mean.mean() <= 0.75          # plan §2: sanity mean ~0.6
    assert df.arm_mean.is_monotonic_decreasing         # arm 0 = best arm (C20)
