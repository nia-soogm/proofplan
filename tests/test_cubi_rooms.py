"""Room-label lexicon / summary parser / de-duplication (no dataset needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bp.sources import cubi_rooms as R


def kinds(t):
    return [k for k, _ in R.classify(t)]


def test_lexicon_basic_and_swedish():
    assert kinds("MH") == ["bedroom"] and kinds("MH 2") == ["bedroom"] and kinds("MH 12,5 m²") == ["bedroom"]
    assert kinds("SOVRUM") == ["bedroom"] and kinds("KÖK") == ["kitchen"] and kinds("BADRUM") == ["bathroom"]
    assert kinds("PSH") == ["bathroom"] and kinds("TUPAK") == ["kitchen"] and kinds("PARV.") == ["balcony"]
    assert kinds("KHH") == [] and kinds("TK") == []  # utility room / vestibule are not counted


def test_combo_labels_and_junk():
    assert kinds("KH/WC") == ["bathroom", "wc"] and kinds("S/PH") == ["sauna", "bathroom"] and kinds("OH+KK") == ["kitchen"]
    assert kinds("OH+KK 26.0 m2") == []  # area list repeats the room label
    assert kinds("H+KK") == []  # summary without its digit
    assert kinds("0 9k21") == [] and kinds("k") == []  # single letters must be clean capitals
    assert kinds("KUISTI/XYZ") == []  # unknown word -> not a label
    assert R.classify("HW") == [("bedroom", True)]  # 'MH' read upside down


def test_summary():
    s = R.parse_summary("3H+K+S")
    assert s["h"] == 3 and {"K", "S"} <= s["parts"]
    assert R.summary_value("bedroom", s) == 2 and R.summary_value("sauna", s) == 1 and R.summary_value("wc", s) == 0
    assert R.parse_summary("4h+kt+s 98,5 m2")["h"] == 4 and R.parse_summary("2 H + KK")["h"] == 2
    assert R.parse_summary("3H-K-S")["parts"] >= {"K", "S"}
    assert R.parse_summary("MH 12") is None and R.parse_summary("1200+300") is None
    assert R.classify("3H+K+S") == []


def test_dedupe_two_line_label():
    a = {"text": "LASITETTU", "conf": 0.8, "bbox": [100, 100, 160, 112]}
    b = {"text": "PARVEKE", "conf": 0.9, "bbox": [105, 114, 155, 126]}
    c = {"text": "PARVEKE", "conf": 0.9, "bbox": [600, 400, 650, 412]}
    ex = R.extract([a, b, c])
    assert len(ex["hits"]["balcony"]) == 2


def test_candidates_carry_evidence():
    toks = [{"text": "MH", "conf": 0.9, "bbox": [10, 10, 30, 22]}, {"text": "MH", "conf": 0.9, "bbox": [300, 10, 320, 22]},
            {"text": "3H+K+S", "conf": 0.9, "bbox": [10, 500, 80, 512]}, {"text": "OH", "conf": 0.9, "bbox": [200, 200, 220, 212]}]
    ex = R.extract(toks)
    cs = {v: (src, ev) for v, src, f, ev in R.candidates(ex, "bedroom", (800, 600))}
    assert 2 in cs and "label" in cs[2][0] and "summary" in cs[2][0] and len(cs[2][1]) == 3
    cs = {v: (src, ev) for v, src, f, ev in R.candidates(ex, "wc", (800, 600))}
    assert 0 in cs and all(ev for _, ev in cs.values())


def test_washroom_implies_sauna_candidate():
    toks = [{"text": "PH", "conf": 0.9, "bbox": [10, 10, 30, 22]}, {"text": "MH", "conf": 0.9, "bbox": [300, 10, 320, 22]},
            {"text": "KHH", "conf": 0.9, "bbox": [500, 10, 530, 22]}]
    ex = R.extract(toks)
    cf, ph = R.code_feats(ex)
    assert cf["n_ph"] == 1 and cf["n_kh"] == 0 and cf["n_khh"] == 1 and len(ph) == 1
    cs = {v: (src, f, ev) for v, src, f, ev in R.candidates(ex, "sauna", (800, 600))}
    assert 1 in cs and "washroom" in cs[1][0] and cs[1][1]["src_wash"] == 1 and cs[1][2][0][1].startswith("washroom")
    old = {v: src for v, src, f, ev in R.candidates(ex, "sauna", (800, 600), use_new=False)}
    assert 1 not in old  # the old gate had no such candidate
