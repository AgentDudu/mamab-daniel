"""Phase 8 guards: real arm-mean loading, real instance shape, the paper's real
Table-1 columns, and the raw-CSV cache round-trip that makes the long run resumable."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import p8_real as p8  # noqa: E402
from test_spec_table1 import _paper_rows  # noqa: E402


def test_paper_real_columns_match_the_paper_html_transcription():
    """The script targets the paper's *real* columns, read from the spec table."""
    cells = p8.paper_cells()
    paper = _paper_rows()
    assert len(cells) == len(paper) == 12
    for key, four in paper:
        assert cells[key]["uniform"] == four[2]
        assert cells[key]["omniscient"] == four[3]


def test_arm_means_are_the_sorted_phase_7_vector_truncated_to_k():
    """C20/C21: arm 0 is the best arm; K=75 is the top-75 prefix of K=100."""
    m100, m75 = p8.arm_means(100), p8.arm_means(75)
    assert m100.shape == (100,) and m75.shape == (75,)
    assert np.array_equal(m75, m100[:75])
    assert np.all((m100 >= 0.0) & (m100 <= 1.0))
    assert np.all(np.diff(m100) <= 0) and m100[0] == m100.max()
    with pytest.raises(ValueError):
        p8.arm_means(101)


def test_trial_instance_wraps_the_real_means():
    """Real means stay fixed; sticky sets, U/L and active sets follow C5/C6."""
    mu, sticky, active, U, L = p8.trial_instance(75, 0)
    S = p8.sticky_size(75)
    assert mu is p8.arm_means(75)
    assert mu[0] == mu.max()
    assert sticky.shape == (p8.N, 75) and active.shape == (p8.N, S + 2)
    assert sticky[:, 0].any()  # C6: the best arm is in some sticky set
    for i in range(p8.N):
        sticky_arms, u, l = set(np.flatnonzero(sticky[i])), int(U[i]), int(L[i])
        assert u != l and u not in sticky_arms and l not in sticky_arms  # C5
        assert set(active[i].tolist()) == sticky_arms | {u, l}  # active = sticky + {U, L}


def test_table_raw_csv_roundtrip(tmp_path, monkeypatch):
    """A cell writes paired per-trial rows that reload with the ratio intact."""
    monkeypatch.setattr(p8, "RESULTS", tmp_path)
    cfg = (10, 75, 2)
    p8.run_table_config(cfg, "uniform", trials=1, t=2000)
    loaded = p8.load_table_raw(cfg, "uniform", 1)
    assert loaded is not None
    proposed, gossip, ratio = loaded
    assert proposed.shape == gossip.shape == ratio.shape == (1,)
    assert pytest.approx(ratio[0], rel=1e-5) == proposed[0] / gossip[0]
    assert p8.load_table_raw(cfg, "uniform", 2) is None  # a different trial count re-runs


def test_fig_raw_csv_roundtrip(tmp_path, monkeypatch):
    """Curves + discard counts reload with the checkpoint grid intact."""
    monkeypatch.setattr(p8, "RESULTS", tmp_path)
    cps = np.unique(np.round(np.logspace(2, np.log10(4000), 5)).astype(np.int64))
    p8.run_fig_config("omniscient", "proposed", trials=1, t=4000, cps=cps)
    loaded = p8.load_fig_raw("omniscient", "proposed", 1, cps)
    assert loaded is not None
    regret, dropped = loaded
    assert regret.shape == dropped.shape == (1, cps.size)
    assert np.all(np.diff(regret[0]) >= 0)  # regret is non-decreasing in T
    assert p8.load_fig_raw("omniscient", "proposed", 1, cps[:-1]) is None
