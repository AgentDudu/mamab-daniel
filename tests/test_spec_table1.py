"""Guard: specs/paper_spec.md Table 1 must match the transcription source paper_html.txt."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PM = re.compile(r"([0-9.]+)\s*\\pm\s*([0-9.]+)")
SPEC_ROW = re.compile(
    r"^\| (\d+) \| (\d+) \| (\d+) \| "
    + r"([\d.]+) ± ([\d.]+) \| "
    + r"([\d.]+) ± ([\d.]+) \| "
    + r"([\d.]+) ± ([\d.]+) \| "
    + r"([\d.]+) ± ([\d.]+) \|$"
)


def _paper_rows():
    text = (ROOT / "paper_html.txt").read_text(encoding="utf-8")
    region = text[text.index("Table 1. Average regret") : text.index("Experimental support")]
    tokens = []
    for line in region.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d+", line):
            tokens.append(("i", int(line)))
        elif PM.fullmatch(line):
            a, b = PM.fullmatch(line).groups()
            tokens.append(("c", (float(a), float(b))))
    rows, k = [], 0
    while k + 7 <= len(tokens):
        if all(t == "i" for t, _ in tokens[k : k + 3]) and all(
            t == "c" for t, _ in tokens[k + 3 : k + 7]
        ):
            rows.append((tuple(v for _, v in tokens[k : k + 3]), [v for _, v in tokens[k + 3 : k + 7]]))
            k += 7
        else:
            k += 1
    return rows


def _spec_rows():
    spec = (ROOT / "specs" / "paper_spec.md").read_text(encoding="utf-8")
    out = []
    for line in spec.splitlines():
        mt = SPEC_ROW.match(line)
        if mt:
            g = mt.groups()
            key = tuple(int(x) for x in g[:3])
            cells = [(float(g[3 + 2 * i]), float(g[4 + 2 * i])) for i in range(4)]
            out.append((key, cells))
    return out


def test_table1_transcription():
    paper, spec = _paper_rows(), _spec_rows()
    assert len(paper) == 12 and len(spec) == 12, (len(paper), len(spec))
    assert paper == spec
