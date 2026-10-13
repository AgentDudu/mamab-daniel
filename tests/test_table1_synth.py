"""Phase 6 harness guards: paper targets, tolerance verdicts, raw-CSV round-trip."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import p6_table1_synth as p6  # noqa: E402
from test_spec_table1 import _paper_rows  # noqa: E402


def test_paper_targets_match_the_paper_html_transcription():
    """The script's Table-1 targets are the paper's synthetic columns, not a copy."""
    cells = p6.paper_cells()
    paper = _paper_rows()
    assert len(cells) == len(paper) == 12
    for key, cells4 in paper:
        assert cells[key]["uniform"] == cells4[0]
        assert cells[key]["omniscient"] == cells4[1]


def test_grid_is_the_spec_grid():
    cfgs = p6.configs(p6.M_GRID, p6.K_GRID, p6.ETA_GRID)
    assert len(cfgs) == 12
    assert set(cfgs) == set(p6.paper_cells())


def test_sticky_size_is_ceil_K_over_n():
    assert p6.sticky_size(75) == 3 and p6.sticky_size(100) == 4


def test_verdict_bands():
    assert p6.verdict(0.50, 0.45, 0.05) == "match"          # |delta| = 0.05
    assert p6.verdict(0.54, 0.45, 0.20) == "match"          # inside the 0.10 tolerance
    assert p6.verdict(0.56, 0.45, 0.20) == "within-noise"   # outside tol, inside std
    assert p6.verdict(0.70, 0.45, 0.20) == "mismatch"


def test_raw_csv_roundtrip(tmp_path, monkeypatch):
    """A cell writes paired per-trial rows that reload with the ratio intact."""
    monkeypatch.setattr(p6, "RESULTS", tmp_path)
    cfg = (10, 75, 2)
    p6.run_config(cfg, "uniform", trials=1, t=2000)
    loaded = p6.load_raw(cfg, "uniform", 1)
    assert loaded is not None
    proposed, gossip, ratio = loaded
    assert proposed.shape == gossip.shape == ratio.shape == (1,)
    assert pytest.approx(ratio[0], rel=1e-5) == proposed[0] / gossip[0]
    assert p6.load_raw(cfg, "uniform", 2) is None  # a different trial count re-runs
