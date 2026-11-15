"""Swiss cross-check helpers (BRAINSTORM #5); no dataset needed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bp.sources import swiss as S


def test_letter_gap():
    assert S.letter_gap(["A", "B", "C", "D", "E", "G", "H"]) == ["F"]
    assert S.letter_gap(["A", "B", "C"]) is None
    assert S.letter_gap(["B", "C"]) is None  # not starting at A: not judged
    assert S.letter_gap(["1", "2", "4"]) is None  # numbered units: not judged


def test_fill_area_cuts_holes():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(2, 2), (2, 4), (4, 4), (4, 2)]  # orientation does not matter
    other = [(20, 0), (25, 0), (25, 5), (20, 5)]  # a second, disjoint part adds
    assert S._fill_area([outer]) == 100
    assert S._fill_area([outer, hole]) == 96
    assert S._fill_area([outer, hole, other]) == 121


def test_area_check_flags_disagreement(monkeypatch):
    monkeypatch.setattr(S, "xcheck", lambda stem: {"pdf_area": {"1.01": 10.0, "1.02": 20.0}})
    f = {"stem": "x"}
    rooms = [{"number": "1.01", "area": 10.0}, {"number": "1.02", "area": 26.0}]
    pd, rel, bad = S.area_check(f, rooms)
    assert pd == 30.0 and rel > S.AREA_TOL and [r["number"] for r in bad] == ["1.02"]
    pd, rel, bad = S.area_check(f, rooms[:1])
    assert rel == 0 and not bad
    assert S.area_check(f, [{"number": "9.99", "area": 1.0}])[0] is None  # no PDF area -> unchecked, never abstain
