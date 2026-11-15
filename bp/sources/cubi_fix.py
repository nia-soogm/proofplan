"""CubiCasa fixture counts (showers/bathtubs, sinks, WC seats) inferred from room labels, fused with the WC-symbol detector.

There is no CPU shower/tub detector, and the sink detector is too weak. Room labels predict fixtures well, though. A
bathroom ('KH', 'PH', 'PESUH') almost always holds one shower or tub. Sinks sit in the kitchen, the bathrooms and the WCs. A WC
seat sits in each WC and in most bathrooms.
1. Room-label hits come from cubi_rooms.extract on OCR v2, with the code identity kept: KH-type (bathroom, often a tub) vs
   PH-type (washroom, shower), combo labels ('KH/WC', 'S/PH'), plus the apartment-type summary and how many labels were read.
2. Candidate counts per kind are simple label formulas. Showers: n_bath, max(1, n_bath), n_bath plus a sauna washroom. Sinks:
   kitchen + bathrooms + WCs. WC seats: bathrooms + WCs, and the detector's symbol count.
3. A gradient-boosted gate trained out-of-fold on DEV images (scripts/cubi/train_fix.py) scores P(candidate == gold). We answer
   only when p >= tau, where tau is chosen on dev so the Clopper-Pearson 95 % lower bound on precision is >= 90 %
   (bp/sources/calib.py).
4. Evidence: the room-label boxes the count was inferred from, plus the detector boxes when the WC-symbol count is the answer.
   The answer text states plainly when a count is inferred from labels and the fixture symbol itself was not detected.
"""
import functools, pickle
from pathlib import Path
import numpy as np
from . import cubi_rooms as R

KINDS = ("shower_bath", "sink", "wc_seat")
KH = {"KH", "KPH", "KYLPYH", "KYLPYHUONE", "BADRUM", "BDR"}  # bathroom proper (tub or shower)
PH = {"PH", "PESUH", "PESUHUONE", "PSH", "PESUTILA", "PESUTILAT", "SH", "SUIHKU", "PESU", "DUSCH", "TVÄTTRUM"}  # washroom (shower)
TYP = {"colorful": 0, "high_quality": 1, "high_quality_architectural": 2}
LABS = {"shower_bath": ("bathroom", "sauna"), "sink": ("kitchen", "bathroom", "wc"), "wc_seat": ("bathroom", "wc")}
PRIOR_TAGS = {"at_least_one", "kit1_bath_wc", "bath_wc_min1"}  # formulas that assume >= 1 even with no matching label
NAMES = {"shower_bath": ("shower/bathtub", "showers/bathtubs"), "sink": ("sink/basin", "sinks/basins"), "wc_seat": ("WC seat", "WC seats")}


def _codes(t):
    c = R.core(t["text"])
    parts = {p.rstrip(".") for p in c.replace("/", " ").replace("+", " ").replace(",", " ").split()} | {c.replace(" ", "")}
    return parts


def label_info(ex):
    """Counts with code identity from cubi_rooms.extract output."""
    hb, hw, hk, hs = ex["hits"]["bathroom"], ex["hits"]["wc"], ex["hits"]["kitchen"], ex["hits"]["sauna"]
    inn = lambda t, L: any(t is u for u in L)
    combo_bw = sum(1 for t in hb if inn(t, hw))
    combo_sb = sum(1 for t in hb if inn(t, hs))
    n_kh = sum(1 for t in hb if _codes(t) & KH)
    n_ph = sum(1 for t in hb if (_codes(t) & PH) and not (_codes(t) & KH))
    khh = sum(1 for t in ex["others"] if _codes(t) & {"KHH", "KODINH", "KODINHOITOH"})
    sv = [s for _, s in ex["summ"]]
    s = sv[0] if sv else None
    bconf = [t["conf"] for t in hb + hw] or [0.0]
    return {"nb": len(hb), "nb_st": len(ex["strong"]["bathroom"]), "nw": len(hw), "nw_st": len(ex["strong"]["wc"]),
            "nk": len(hk), "nk_st": len(ex["strong"]["kitchen"]), "ns": len(hs), "ns_st": len(ex["strong"]["sauna"]),
            "n_kh": n_kh, "n_ph": n_ph, "combo_bw": combo_bw, "combo_sb": combo_sb, "n_khh": khh,
            "n_bed": len(ex["strong"]["bedroom"]), "n_bal": len(ex["strong"]["balcony"]),
            "n_other": ex["n_other"], "n_lab": ex["n_lab"], "n_tok": ex["n_tok"], "has_sum": int(s is not None),
            "sum_h": s["h"] if s else -1, "sum_s": int(bool(s and s["parts"] & R.SUM_PART["sauna"])) if s else -1,
            "sum_wc": int(bool(s and s["parts"] & R.SUM_PART["wc"])) if s else -1,
            "sum_bath": int(bool(s and s["parts"] & R.SUM_PART["bathroom"])) if s else -1,
            "bw_minconf": min(bconf), "bw_meanconf": float(np.mean(bconf))}


def label_cands(kind, L):
    """-> {value: set(formula tags)} from room-label formulas."""
    nb, nw, nk, cb = L["nb"], L["nw"], min(1, L["nk"]), L["combo_bw"]
    c = {}
    add = lambda v, tag: c.setdefault(int(v), set()).add(tag) if v >= 0 else None
    if kind == "shower_bath":
        add(nb, "per_bath"); add(L["nb_st"], "per_bath_strong"); add(max(1, nb), "at_least_one")
        add(nb + (1 if L["ns"] and L["n_ph"] == 0 and nb == 0 else 0), "sauna_wash")
    elif kind == "sink":
        base = nk + nb + nw - cb
        add(base, "kit_bath_wc"); add(base + 1, "kit_bath_wc_plus1"); add(nk + nb, "kit_bath"); add(max(1, nk) + nb + nw - cb, "kit1_bath_wc")
    elif kind == "wc_seat":
        add(nb + nw - cb, "bath_wc"); add(max(1, nb + nw - cb), "bath_wc_min1"); add(nw + (1 if nw == 0 and nb else 0), "wc_or_bath")
    return c


def det_feats(path, P, bx, s, nlab, prm_wc):
    """WC-symbol detector count + its gate features, and a sink detector count (weak, used only as a feature)."""
    from .cubi_sym import gate_feats, nms_count
    if len(P) == 0:
        return 0, [], [0] * 19, 0
    n, kept, ff = gate_feats(path, bx, P, 1, s, nlab, prm_wc["thr"], prm_wc["merge"])
    nsink = len(nms_count(bx, P[:, 2], 0.5, 1.6))
    return n, kept, list(ff), nsink


def candidates(kind, ex, size, path, det):
    """-> list of (value, tags, feature vector, evidence [(token, label)], uses_detector)."""
    L = label_info(ex)
    dn, kept, dff, nsink = det
    W, H = size
    cs = label_cands(kind, L)
    if kind == "wc_seat":
        cs.setdefault(int(dn), set()).add("detector")
    base = [L[k] for k in sorted(L)] + [ex["lab_h"] / max(1, max(W, H)) * 100, W * H / 1e6, TYP.get(Path(path).stem.rsplit("-", 1)[0], 1),
                                       dn, nsink] + dff
    ref = {"shower_bath": L["nb"], "sink": min(1, L["nk"]) + L["nb"] + L["nw"] - L["combo_bw"], "wc_seat": L["nb"] + L["nw"] - L["combo_bw"]}[kind]
    out = []
    all_tags = ["per_bath", "per_bath_strong", "at_least_one", "sauna_wash", "kit_bath_wc", "kit_bath_wc_plus1", "kit_bath",
                "kit1_bath_wc", "bath_wc", "bath_wc_min1", "wc_or_bath", "detector"]
    labs = LABS[kind]
    for v, tags in sorted(cs.items()):
        f = base + [v, v - ref, v - dn, int(v == dn), len(tags)] + [int(t in tags) for t in all_tags]
        ev = []
        for k in labs:
            for t in ex["hits"][k]:
                if all(t is not u for u, _ in ev):
                    ev.append((t, k))
        if not ev and (v == 0 or (v == 1 and tags & PRIOR_TAGS)):
            # 0: no matching label among the labels read; 1 from a 'min 1' formula: no matching label, but a dwelling plan
            # (these room labels were read) is assumed to hold at least one. The answer text says so.
            ev = [(t, "other room label") for t in ex["others"][:12]] + [(t, k) for k in R.KINDS for t in ex["strong"][k]][:12]
        if not ev and not ("detector" in tags and kept):
            continue  # no box to show for this count -> never an answer
        out.append((v, tags, f, ev, "detector" in tags))
    return out


@functools.lru_cache(maxsize=1)
def model():
    f = Path(__file__).resolve().parent.parent / "models" / "cubi_fix.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def detector_inputs(path):
    """Detector boxes/probabilities at answer time: the dev-trained stage-1 candidate classifier from cubi_sym.pkl."""
    from . import cubi_sym
    M = cubi_sym._model()
    bx, X, s, nlab = cubi_sym.cached_desc(path)
    P = M["cand"].predict_proba(X) if len(bx) else np.zeros((0, len(M["classes"])))
    return bx, P, s, nlab, M["params"]["wc"]


def answer(path, kind, tokens, size):
    """-> None (no model / kind disabled) or dict with status, value, message parts and evidence."""
    M = model()
    if M is None or kind not in M["models"] or not M["enabled"].get(kind):
        return None
    ex = R.extract(tokens)
    bx, P, s, nlab, prm = detector_inputs(path)
    det = det_feats(path, P, bx, s, nlab, prm)
    cs = candidates(kind, ex, size, path, det)
    if not cs:
        return {"status": "abstain", "why": "no room labels readable to infer from" + (" and no WC-seat symbol detected" if kind == "wc_seat" else "")}
    X = np.array([c[2] for c in cs], float)
    p = M["models"][kind].predict_proba(X)[:, 1]
    i = int(p.argmax())
    v, tags, f, ev, used_det = cs[i]
    det_boxes = [bx[j] for j in det[1]] if used_det else []
    from .calib import tau_for
    tau = float(tau_for(M, kind))
    return {"status": "answered" if p[i] >= tau else "abstain", "value": int(v), "tags": sorted(tags), "evidence": ev,
            "p": float(p[i]), "tau": tau, "det_boxes": det_boxes, "det_n": int(det[0]), "info": label_info(ex),
            "used_det": used_det}
