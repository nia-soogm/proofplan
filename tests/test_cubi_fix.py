"""Clopper-Pearson threshold rule and fixture-from-label formulas (no dataset needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bp.sources import cubi_rooms as R, cubi_fix as F
from bp.sources.calib import cp_lower, choose_tau


def test_cp_lower_bound():
    assert cp_lower(0, 10) == 0.0
    assert 0.74 < cp_lower(10, 10) < 0.75  # 0.05 ** (1/10)
    assert cp_lower(92, 100) < 0.92 and cp_lower(920, 1000) > cp_lower(92, 100)


def test_choose_tau_prefers_bound_over_point_estimate():
    # 10 answers all right (point 100 %) is not enough for a 90 % lower bound; 100 answers at 97 % is
    p = [0.99] * 10 + [0.8] * 90 + [0.5] * 50
    y = [1] * 10 + [1] * 87 + [0] * 3 + [0] * 50
    ch = choose_tau(p, y, 150)
    assert ch["tau"] <= 0.8 and ch["n"] == 100 and ch["lower"] >= 0.9
    assert choose_tau([0.99] * 5, [1] * 5, 5)["tau"] == 1.01


def _tok(text, x, y=10, conf=0.9):
    return {"text": text, "conf": conf, "bbox": [x, y, x + 20, y + 12]}


def test_label_info_and_formulas():
    ex = R.extract([_tok("KH/WC", 10), _tok("PH", 200), _tok("S", 260), _tok("K", 400), _tok("MH", 600), _tok("OH", 800)])
    L = F.label_info(ex)
    assert L["nb"] == 2 and L["nw"] == 1 and L["combo_bw"] == 1 and L["n_kh"] == 1 and L["n_ph"] == 1 and L["nk"] == 1
    assert 2 in F.label_cands("shower_bath", L)  # one per bathroom/washroom
    assert 3 in F.label_cands("sink", L)  # kitchen + 2 bathrooms + WC, combo label counted once
    assert 2 in F.label_cands("wc_seat", L)
    L0 = F.label_info(R.extract([_tok("MH", 10), _tok("OH", 200)]))
    assert F.label_cands("shower_bath", L0)[1] >= {"at_least_one"}  # prior when no bathroom label is read


def test_strict_mode_thresholds():
    from bp.sources import calib
    M = {"tau": {"a": 0.8, "b": 0.6}, "tau_strict": {"a": 0.95}}
    assert calib.tau_for(M, "a") == 0.8 and not calib.strict()
    tok = calib.set_strict(True)
    try:
        assert calib.tau_for(M, "a") == 0.95
        assert calib.tau_for(M, "b") == 1.01  # no strict threshold for the kind -> never answer
        assert calib.tau_for({"tau": 0.5}) == 1.01  # model without strict thresholds -> never answer
        assert "strict" in calib.mode_text()
    finally:
        calib.reset_strict(tok)
    assert calib.tau_for({"tau": 0.5}) == 0.5 and not calib.strict()


def test_choose_strict_never_looser_than_normal():
    from bp.sources.calib import choose_strict
    p = [0.99] * 300 + [0.7] * 300
    y = [1] * 300 + [1] * 270 + [0] * 30
    c = choose_strict(p, y, 600, tau_normal=0.75)
    assert c["tau"] >= 0.75 and c["n"] == 300
    assert choose_strict([0.6] * 1000, [1] * 1000, 1000, tau_normal=0.8)["tau"] >= 0.8
