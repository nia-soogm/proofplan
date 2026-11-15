"""CubiCasa5K (raster Finnish real-estate plans): key-free, CPU only.

Room-type counts (bedroom, bathroom, separate WC, sauna, kitchen, balcony) come from bp/sources/cubi_rooms.py.
It runs on OCR v2 (`ocr2`: upright + 90°/270° passes, low-confidence single letters kept), with a Finnish/Swedish label
lexicon and the apartment-type summary ('3H+K+S'). A dev-trained gate decides whether to answer, and each counted
label is a box on the image. The older rule set further down is only a fallback for when bp/models/cubi_rooms.pkl is missing.
Storeys: sheet layout split + gate (cubi_cv). WC seats, sinks, showers/baths: room-label formulas fused with the WC-symbol
detector behind a dev-trained gate (cubi_fix; answers say when a count is inferred from labels rather than detected).
Doors and windows abstain (bp/sources/vlm.py is key-pluggable; nothing is sent without a key).
"""
import functools, json, re
from pathlib import Path
from . import JDATA, CACHE
from .calib import mode_text, strict

# label -> room kind. Matching is on the whole OCR token after removing digits/punctuation ('MH 2', 'MH2.', 'S 1').
LEX = {
    "bedroom": {"MH", "MAKUUHUONE", "MAKUUH", "MAKH"},
    "sauna": {"S", "SAUNA", "SAUN"},
    "bathroom": {"KH", "KPH", "PH", "PESUH", "SH", "PESUHUONE", "KYLPYH", "KYLPYHUONE", "PESUTILA", "PHH", "KHH/PH"},
    "wc": {"WC", "W.C", "W C"},
    "kitchen": {"K", "KK", "KT", "AVOK", "AVOKEITTIO", "AVOKEITTIÖ", "KEITTIO", "KEITTIÖ", "KEITTIÖTILA", "K-TILA", "KEITTOKOMERO", "AVO-K"},
    "balcony": {"PARVEKE", "PARV", "LASIT.PARV", "LASITETTU PARVEKE", "LASITETTUPARVEKE", "LASIPARVEKE", "LPARV", "LASIT PARV", "RANSKALAINEN PARVEKE"},
}
FUZZY = {"NH": "bedroom", "MHE": "bedroom", "HM": "bedroom", "MN": "bedroom", "KPh": "bathroom"}  # frequent OCR misreads of MH
SUMMARY_RE = re.compile(r"(\d)\s*H\s*(\+\s*[A-ZÄÖ]+\s*)+", re.I)  # apartment type '4H+K+S', '3h+kt+s'
STOREY_RE = re.compile(r"^(\d)\s*\.?\s*(KRS|KERROS)\b|^(KELLARI|ULLAKKO|KELLARIKERROS|POHJAKERROS|PARVI)\b", re.I)

QKIND = [  # question (EN or FI) -> kind; symbol questions first (they mention kitchens/bathrooms in passing)
    ("wc_seat", r"toilet bowl|wc seat|wc-istuin"), ("sink", r"sinks?\b|basin|allas|altaita|lavuaari"),
    ("shower_bath", r"shower|bathtub|suihku|kylpyamme"), ("door", r"\bdoors?\b|\bovea\b|oviaukko"), ("window", r"window|ikkuna"),
    ("bedroom", r"bedroom|makuuhuone"), ("sauna", r"sauna"), ("bathroom", r"bathroom|washroom|kylpyhuone|pesuhuone|pesutil"),
    ("wc", r"separate wc|wc rooms|wc-tilo|erillis"), ("kitchen", r"kitchen|keitti"), ("balcony", r"balcon|parvek"),
    ("storey", r"floors \(storeys\)|kerrosta"),
]
SYMBOL_KINDS = {"door", "window", "wc_seat", "sink", "shower_bath"}


@functools.lru_cache(maxsize=1)
def _ocr():
    import os
    from rapidocr_onnxruntime import RapidOCR
    n = int(os.environ.get("BP_ORT_THREADS", "0"))
    if n > 0:  # onnxruntime ignores OMP_NUM_THREADS; cap threads so parallel OCR shards don't oversubscribe the CPU
        from rapidocr_onnxruntime import utils as _u
        _SO = _u.SessionOptions

        def _so():
            o = _SO(); o.intra_op_num_threads = n; o.inter_op_num_threads = 1
            return o
        _u.SessionOptions = _so
    return RapidOCR()


def ocr(path):
    stem = Path(path).stem
    cf = CACHE / "cubi" / f"{stem}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    from PIL import Image
    import numpy as np
    im = Image.open(JDATA / path).convert("RGB")
    s = 2.0 if max(im.size) < 1600 else 1.0
    if s != 1.0:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    res, _ = _ocr()(np.array(im))
    toks = []
    for box, text, conf in res or []:
        xs = [p[0] / s for p in box]; ys = [p[1] / s for p in box]
        toks.append({"text": text, "conf": float(conf), "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    out = {"stem": stem, "size": list(Image.open(JDATA / path).size), "tokens": toks}
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out))
    return out


def ocr2(path):
    """OCR v2 for room labels: the upright pass plus 90° / 270° rotated passes (Finnish plans often write labels
    vertically), keeping low-confidence tokens down to 0.1 (RapidOCR scores single letters like 'K' or 'S' ~0.2-0.5).
    Boxes are mapped back to the original image; overlapping tokens from different passes keep the higher score."""
    stem = Path(path).stem
    cf = CACHE / "cubi_ocr2" / f"{stem}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    from PIL import Image
    import numpy as np
    im0 = Image.open(JDATA / path).convert("RGB")
    W, H = im0.size
    s = 2.0 if max(im0.size) < 1600 else 1.0
    im = im0.resize((int(W * s), int(H * s)), Image.LANCZOS) if s != 1.0 else im0
    w, h = im.size
    toks = []
    for rot in (0, 90, 270):
        x = im.rotate(rot, expand=True) if rot else im
        res, _ = _ocr()(np.array(x), text_score=0.1)
        for box, text, conf in res or []:
            pts = []
            for px, py in box:  # PIL rotate(+angle) is counter-clockwise; invert it
                if rot == 90:
                    ox, oy = w - py, px
                elif rot == 270:
                    ox, oy = py, h - px
                else:
                    ox, oy = px, py
                pts.append((ox / s, oy / s))
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            n = len(text.strip())
            # orientation check: a multi-letter word read in a rotated pass must sit in a tall box, and vice versa;
            # otherwise it is a stacked-letter misread ('OH' upright read as 'HO' in the 90° pass)
            consistent = n <= 1 or ((bh > bw) if rot else (bw >= bh * 0.8))
            toks.append({"text": text, "conf": float(conf), "bbox": [min(xs), min(ys), max(xs), max(ys)], "rot": rot,
                         "orient_ok": bool(consistent)})
    toks.sort(key=lambda t: -(t["conf"] + (0.5 if t["orient_ok"] else 0.0)))
    keep = []
    for t in toks:
        if not t["text"].strip():
            continue
        a = t["bbox"]; aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        dup = False
        for u in keep:
            b = u["bbox"]
            iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
            if iw > 0 and ih > 0:
                inter = iw * ih; bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
                if inter / min(aa, bb) > 0.5:
                    dup = True; break
        if not dup:
            keep.append(t)
    out = {"stem": stem, "size": [W, H], "tokens": keep, "v": 2}
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out))
    return out


def core(t):
    t = t.upper().replace("Ä", "Ä").strip()
    t = re.sub(r"[\d]+", " ", t)
    t = re.sub(r"[^\wÄÖÅ./ -]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .-/")
    return t


def classify(text):
    if "+" in text or re.search(r"m2|m²", text, re.I):  # apartment summary '2h+kk 50 m2', not a room label
        return None
    c = core(text)
    if not c:
        return None
    for k, lex in LEX.items():
        if c in lex or c.replace(" ", "") in lex or c.replace(".", "") in lex:
            return k
    # labels glued to an area ('MH 12,5') or a list 'MH/VH'
    if c in FUZZY:
        return FUZZY[c]
    parts = re.split(r"[/ ]+", c)
    hits = {k for p in parts for k, lex in LEX.items() if p in lex and len(p) >= 2}
    return hits.pop() if len(hits) == 1 else None


def qkind(q):
    ql = q.lower()
    for k, rx in QKIND:
        if re.search(rx, ql):
            return k
    return None


def answer(question, path, min_labels=3, min_conf=0.5):
    k = qkind(question)
    if k is None:
        return _abst("question not understood")
    if k in SYMBOL_KINDS:
        from . import vlm
        if vlm.available():
            return vlm.count_symbols(path, k)
        r = fixture_answer(path, k)
        if r is not None:
            return r
        return symbol_answer(path, k)
    if k == "storey":
        return storey_answer(path)
    r = room_answer(path, k)
    if r is not None:
        return r
    o = ocr(path)
    toks = [t for t in o["tokens"] if t["conf"] >= min_conf]
    known = [(t, classify(t["text"])) for t in toks]
    known = [(t, c) for t, c in known if c]
    if k == "storey":
        return storey_answer(path)
    hits = [t for t, c in known if c == k]
    summ = [t for t in toks if SUMMARY_RE.search(t["text"].replace(" ", ""))]
    parts = {p.strip().upper() for t in summ for p in re.findall(r"\+\s*([A-ZÄÖa-zäö]+)", t["text"])}
    if not hits and summ and k in ("sauna", "kitchen"):
        flag = {"sauna": {"S", "SAUNA"}, "kitchen": {"K", "KK", "KT", "AVOK", "KEITTIÖ", "KEITTIO", "TK"}}[k] & parts
        if flag:
            t = summ[0]
            return _ok(1, f"1 {k}: the apartment-type summary '{t['text']}' lists '+{sorted(flag)[0]}' (no room label was readable).", [_ev(t, k + " (summary)")], weak=True)
        if k == "sauna" and parts:
            return _ok(0, f"0 saunas: the apartment-type summary '{summ[0]['text']}' has no '+S'.", [_ev(summ[0], "summary")], weak=True)
    if k == "bedroom":
        hs = [(t, int(SUMMARY_RE.search(t["text"].replace(" ", "")).group(1))) for t in summ]
        if hs:
            t, h = hs[0]
            ev = [_ev(t, "summary")] + [dict(_ev(x, k), verified=(len(hits) == h - 1)) for x in hits]
            ev[0]["verified"] = len(hits) == h - 1 if hits else None
            note = (f"; {len(hits)} MH label(s) read — agrees." if len(hits) == h - 1 else f"; only {len(hits)} MH label(s) readable (OCR) — not confirmed.") if hits else "; no MH labels readable."
            return _ok(h - 1, f"{h - 1} bedrooms: apartment type '{t['text']}' = {h} rooms incl. the living room{note}", ev)
        return _abst(f"no apartment-type summary ('3H+K+S'); {len(hits)} MH label(s) read but OCR recall is too low to trust a label count")
    if hits:
        return _ok(len(hits), f"{len(hits)} {k} label(s) read on the plan: " + ", ".join(t["text"] for t in hits) + ".", [_ev(t, k) for t in hits])
    if len(known) >= min_labels and k in ("balcony",):  # dev: absence was reliable only for balconies
        return _ok(0, f"0 — no {k} label among {len(known)} readable room labels ({', '.join(sorted({t['text'] for t, _ in known})[:8])}).",
                   [_ev(t, c) for t, c in known], weak=True)
    return _abst(f"no {k} label found ({len(known)} readable room labels)")


ROOM_NAMES = {"bedroom": ("bedroom", "bedrooms"), "bathroom": ("bathroom", "bathrooms"), "wc": ("separate WC", "separate WCs"),
              "sauna": ("sauna", "saunas"), "kitchen": ("kitchen", "kitchens"), "balcony": ("balcony", "balconies")}
SRC_TEXT = {"label": "room label(s) read", "label_all": "room label(s) read incl. low-confidence single letters",
            "presence": "a room label is present", "summary": "the apartment-type summary",
            "washroom": "inferred from a PH/PESUH washroom label (no sauna label read; saunas usually adjoin the washroom, sauna bench not detected)"}



def room_answer(path, k):
    """Room-kind counts via bp/sources/cubi_rooms.py (lexicon + summary + dev-trained gate on OCR v2). None if no model."""
    from . import cubi_rooms
    M = cubi_rooms.model()
    if M is None or k not in M["models"]:
        return None
    o = ocr2(path) if M.get("ocr") == "cubi_ocr2" else ocr(path)
    r = cubi_rooms.answer(path, k, o["tokens"], o["size"])
    if r is None:
        return None
    if r[0] == "abstain" and len(r) == 4:
        return _abst(r[2])
    status, v, src, ev, p, tau, ex = r
    one, many = ROOM_NAMES[k]
    why = " + ".join(SRC_TEXT[s] for s in src.split("+") if s in SRC_TEXT)
    labs = ", ".join(f"'{t['text']}'" for t, lab in ev if lab == k or lab.startswith("washroom"))
    summ = [t for t, lab in ev if lab == "apartment-type summary"]
    detail = (f"; labels: {labs}" if labs else "") + (f"; summary '{summ[0]['text'].strip()}'" if summ else "")
    if summ and k == "bedroom":
        detail += f" = {v + 1} rooms incl. the living room"
    if v == 0 and not labs:
        detail += f"; {len(ex['others'])} other room labels were read and none is a {one}"
    evid = [_ev(t, lab) for t, lab in ev]
    srcs = set(src.split("+"))
    if summ and srcs & {"label", "label_all"}:  # two independent sources agree: labels and summary are cross-checked (green)
        for e in evid:
            e["verified"] = True
    if status != "answered":
        return {"status": "abstain", "value": None, "evidence": [dict(e, verified=False) for e in evid],
                "answer": f"NOT SURE: best guess {v} {one if v == 1 else many} ({why}{detail}), but the gate is p={p:.2f} < {tau:.2f} (dev-calibrated; {mode_text()})."}
    return _ok(v, f"{v} {one if v == 1 else many}: {why}{detail} (dev-calibrated gate p={p:.2f}; {mode_text()}).", evid)


@functools.lru_cache(maxsize=1)
def _storey_model():
    import pickle
    f = Path(__file__).resolve().parent.parent / "models" / "cubi_storey.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def storey_answer(path):
    """Storeys = separate plan drawings on the sheet (XY-cut of the ink, several gap thresholds) + storey words
    ('1. KRS', 'YLÄKERTA', 'KELLARI'); a gradient-boosted gate trained on dev decides; the count must equal the number
    of drawings actually found (each one is an evidence box), otherwise NOT SURE."""
    from . import cubi_cv
    import numpy as np
    M = _storey_model()
    if M is None:
        return _abst("storey model missing (run scripts/cubi/train_storey.py)")
    f = cubi_cv.storey_all_features(path)
    pr = M["model"].predict_proba(np.array([[f[k] for k in M["keys"]]], float))[0]
    n = int(M["model"].classes_[pr.argmax()]); conf = float(pr.max())
    vs = [v for v in cubi_cv.VALLEYS if f[f"v{int(v * 100)}_n_big"] == n]
    if not vs:
        return _abst(f"layout suggests {n} storey(s) (p={conf:.2f}) but no split of the sheet into {n} drawings was found")
    from .calib import tau_for, mode_text
    tau = tau_for(M)
    if conf < tau:
        return _abst(f"{n} storey(s) is the best guess but confidence {conf:.2f} is below the dev-calibrated gate {tau:.2f} ({mode_text()})")
    v = max(vs, key=lambda v: f[f"v{int(v * 100)}_parts_lab"])
    big = cubi_cv.big_parts(path, v)[0]
    ev = [{"bbox": [int(x), int(y), int(x + w), int(y + h)], "page": 0, "kind": "storey",
           "text": f"plan drawing {i + 1} of {len(big)}", "verified": None} for i, (x, y, w, h, _) in enumerate(sorted(big, key=lambda b: (b[0], b[1])))]
    _, words = cubi_cv.storey_ocr_features(path, {})
    wtoks = [t for t in ocr(path)["tokens"] if re.sub(r"\s+", "", t["text"].upper()) in words]
    ev += [_ev(t, "storey label") for t in wtoks]
    msg = (f"{n} storey{'s' if n > 1 else ''}: the sheet holds {len(big)} separate plan drawing{'s' if len(big) > 1 else ''}"
           + (f"; storey labels read: {', '.join(sorted(words))}" if words else "") + f" (layout gate p={conf:.2f}; {mode_text()}).")
    return _ok(n, msg, ev)


FIX_RULE = {"shower_bath": "about one shower or bathtub per bathroom/washroom", "sink": "about one sink per kitchen, bathroom and WC",
            "wc_seat": "about one WC seat per WC and bathroom"}


def fixture_answer(path, k):
    """Showers/baths, sinks, WC seats via bp/sources/cubi_fix.py: room-label formulas (+ the WC-symbol detector for WC seats)
    behind a dev-trained gate. None when there is no model or the kind is disabled (falls back to symbol_answer)."""
    from . import cubi_fix
    M = cubi_fix.model()
    if M is None or not M["enabled"].get(k):
        return None
    o = ocr2(path)
    r = cubi_fix.answer(path, k, o["tokens"], o["size"])
    if r is None:
        return None
    if "value" not in r:
        return _abst(r["why"])
    one, many = cubi_fix.NAMES[k]
    v, L = r["value"], r["info"]
    read = []
    for key, name in (("nk", "kitchen"), ("nb", "bathroom/washroom"), ("nw", "WC"), ("ns", "sauna")):
        if L[key]:
            read.append(f"{L[key]} {name}")
    labs = ", ".join(f"'{t['text']}'" for t, lab in r["evidence"] if lab != "other room label")
    read_txt = (f"room labels read: {', '.join(read)} ({labs})" if read else f"no matching room label among {L['n_lab']} room labels read")
    evid = [_ev(t, lab) for t, lab in r["evidence"]]
    det_txt = ""
    if r["used_det"] and r["det_boxes"]:
        for i, b in enumerate(r["det_boxes"]):
            x0, y0, x1, y1 = (int(c) for c in b); pad = int(0.25 * max(x1 - x0, y1 - y0)) + 1
            evid.append({"bbox": [x0 - pad, y0 - pad, x1 + pad, y1 + pad], "page": 0, "kind": k, "text": f"WC seat symbol #{i + 1} (detector)", "verified": None})
        has_lab = any(lab in cubi_fix.LABS[k] for _, lab in r["evidence"])
        label_only = has_lab and any(t not in ("detector",) and t not in cubi_fix.PRIOR_TAGS for t in r["tags"])
        how = (f"{r['det_n']} WC-seat symbol(s) detected (boxed)" + (" and the room labels give the same count" if label_only else "")
               + (f"; {read_txt}" if has_lab else ""))
        if label_only:
            for e in evid:
                e["verified"] = True  # detector and labels agree: two independent sources
    elif not any(lab in cubi_fix.LABS[k] for _, lab in r["evidence"]):
        need = {"shower_bath": "bathroom/washroom", "sink": "kitchen, bathroom or WC", "wc_seat": "bathroom or WC"}[k]
        how = (f"no {need} label was read, so this is a prior: the plan is a dwelling ({L['n_lab']} room labels read, boxed) "
               f"and dwellings like it hold at least one; the fixture symbols themselves were not detected"
               + (f" (the WC-symbol detector found {r['det_n']})" if k == "wc_seat" else ""))
    else:
        how = (f"inferred from room labels, {FIX_RULE[k]}; the fixture symbols themselves were not detected"
               + (f" (the WC-symbol detector found {r['det_n']}, so the count rests on the labels)" if k == "wc_seat" else "") + f"; {read_txt}")
    noun = one if v == 1 else many
    if r["status"] != "answered":
        return {"status": "abstain", "value": None, "evidence": [dict(e, verified=False) for e in evid],
                "answer": f"NOT SURE: best guess {v} {noun} ({how}), but the gate is p={r['p']:.2f} < {r['tau']:.2f} (dev-calibrated; {mode_text()})."}
    return _ok(v, f"{v} {noun}: {how} (dev-calibrated gate p={r['p']:.2f}; {mode_text()}).", evid)


def symbol_answer(path, k):
    from . import cubi_sym
    r = cubi_sym.count(path, k)
    one, many = cubi_sym.NAMES.get(k, (k, k))
    if r is None:
        why = {"sink": "the CPU sink detector reached only ~14 % exact counts on dev and no confidence gate reached 90 % precision",
               "door": "the CPU door detectors (swing-arc template and wall-gap swing test) reach only ~10-18 % exact counts on dev and no confidence gate reached the required precision",
               "window": "the CPU window detector (thin glass lines in wall gaps) reaches only ~10-17 % exact counts on dev and no confidence gate reached the required precision",
               }.get(k, f"no CPU detector for {many} yet")
        return _abst(f"{why} (needs a trained detector or a grounded VLM)")
    n, conf, tau, ev = r
    if conf < tau:
        return {"status": "abstain", "answer": f"NOT SURE: found {n} {one} candidate(s) but the count gate is {conf:.2f} < {tau:.2f} (dev-calibrated{'; strict mode' if strict() else ''}).",
                "value": None, "evidence": [dict(e, verified=False) for e in ev]}
    return _ok(n, f"{n} {one if n == 1 else many} detected on the plan (closed-shape symbol detector + dev-calibrated count gate p={conf:.2f}); each is boxed.", ev)


def _ev(t, kind):
    return {"bbox": [round(v, 1) for v in t["bbox"]], "page": 0, "text": f"OCR '{t['text']}' ({t['conf']:.2f}) → {kind}", "kind": kind, "verified": None}


def _ok(v, msg, ev, weak=False):
    return {"status": "answered", "answer": msg + (" (absence of a label — weaker evidence)" if weak else ""), "value": v, "evidence": ev}


def _abst(why):
    return {"status": "abstain", "answer": f"NOT SURE: {why}.", "value": None, "evidence": []}
