"""swiss_dwellings: DXF (structured layers) is the measuring source, the PDF is the checking source.

DXF layers: ROOM_OUTLINE (polygons), ROOM_TEXT (name + number), UNIT_OUTLINE, DOOR, WINDOW, FIXTURE, WALL.
  rooms     = ROOM_OUTLINE polygon containing the room-number text; area = shoelace (metres)
  apartments= UNIT_OUTLINE polygons, letter from the 'Whg. A'/'App. A' text inside
  doors     = door-leaf rectangles (5-pt polylines on DOOR)
  windows   = closed 4-pt polylines on WINDOW
  fixtures  = FIXTURE primitives: 48-pt ellipse with drain circle = washbasin, without = WC,
              69-pt rounded tub = bathtub, 5-pt tray with r~0.045 drain = shower
DXF->PDF transform is fitted per file from the room-number texts (x'=a x+b, y'=-a y+c; residual reported),
so every item gets a box/polygon on the PDF, and we check the PDF has the same label text / vector ink there.
"""
from __future__ import annotations
import json, re, functools, collections, math
from pathlib import Path
from . import JDATA, CACHE

SW = JDATA / "data" / "swiss_dwellings"
NUM_RE = re.compile(r"^[A-Z]?\d+\.\d+$")
UNIT_RE = re.compile(r"^(?:Whg|App|Appt|Appart|Wohnung|Apt)\.?\s*([A-Z0-9]+)$", re.I)
OUTDOOR = {"balkon", "loggia", "terrasse", "dachterrasse", "sitzplatz", "balcon", "balcone", "terrazza", "logge", "veranda",
           "wintergarten", "jardin d'hiver", "giardino d'inverno", "garten", "jardin", "giardino", "patio", "aussenraum"}
VOID = {"schacht", "luftraum", "gaine", "vide", "cavedio", "vuoto", "lichtschacht"}


def poly_area(p):
    return abs(sum(p[i][0] * p[i - 1][1] - p[i - 1][0] * p[i][1] for i in range(len(p)))) / 2


def pip(pt, p):
    x, y = pt; c = False
    for i in range(len(p)):
        x1, y1 = p[i - 1]; x2, y2 = p[i]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1:
            c = not c
    return c


def bbox(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower().replace("’", "'"))


@functools.lru_cache(maxsize=256)
def facts(stem: str) -> dict:
    cf = CACHE / "swiss" / f"{stem}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    import ezdxf, pymupdf, numpy as np
    msp = ezdxf.readfile(str(SW / f"{stem}.dxf")).modelspace()
    texts = []
    for e in msp.query('TEXT[layer=="ROOM_TEXT"]'):
        pt = e.dxf.align_point if (e.dxf.get("halign") or e.dxf.get("valign")) and e.dxf.hasattr("align_point") else e.dxf.insert
        texts.append((e.dxf.text.strip(), (pt[0], pt[1])))
    outlines = [[(p[0], p[1]) for p in e.get_points("xy")] for e in msp.query('LWPOLYLINE[layer=="ROOM_OUTLINE"]')]
    outlines = [o for o in outlines if len(o) >= 3 and poly_area(o) > 1e-6]
    # pair each number with the name text right above it (same x, next line up)
    rooms = []
    for t, pt in texts:
        if not NUM_RE.match(t):
            continue
        above = [(t2, p2) for t2, p2 in texts if not NUM_RE.match(t2) and not UNIT_RE.match(t2) and abs(p2[0] - pt[0]) < 0.05 and 0 < p2[1] - pt[1] < 0.6]
        name = min(above, key=lambda a: a[1][1] - pt[1])[0] if above else None
        polys = [o for o in outlines if pip(pt, o)]
        poly = min(polys, key=poly_area) if polys else None
        holes = [o for o in outlines if poly and o is not poly and poly_area(o) < poly_area(poly) and all(pip(q, poly) for q in o)
                 and not any(NUM_RE.match(t2) and pip(p2, o) for t2, p2 in texts)]  # a numbered room inside is a room, not a column
        area = poly_area(poly) - sum(poly_area(h) for h in holes) if poly else None  # net: minus columns/shafts drawn inside
        rooms.append({"number": t, "name": name, "pt": pt, "poly": poly, "area": area, "gross": poly_area(poly) if poly else None, "holes": holes})
    units = []
    for e in msp.query('LWPOLYLINE[layer=="UNIT_OUTLINE"]'):
        poly = [(p[0], p[1]) for p in e.get_points("xy")]
        lab = None
        for t, pt in texts:
            m = UNIT_RE.match(t)
            if m and pip(pt, poly):
                lab = m.group(1)
        units.append({"label": lab, "poly": poly, "area": poly_area(poly)})
    for r in rooms:
        r["unit"] = next((u["label"] for u in units if pip(r["pt"], u["poly"])), None)
    doors = [bbox([(p[0], p[1]) for p in e.get_points("xy")]) for e in msp.query('LWPOLYLINE[layer=="DOOR"]') if len(list(e.get_points())) == 5]
    windows = [bbox([(p[0], p[1]) for p in e.get_points("xy")]) for e in msp.query('LWPOLYLINE[layer=="WINDOW"]') if e.closed and len(list(e.get_points())) == 4]
    fpolys, circs = [], []
    for e in msp.query('*[layer=="FIXTURE"]'):
        if e.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points("xy")]
            fpolys.append((len(pts), bbox(pts)))
        elif e.dxftype() == "CIRCLE":
            c = e.dxf.center; circs.append((c[0], c[1], e.dxf.radius))
    fixtures = collections.defaultdict(list)
    inside = lambda b, c: b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]
    for n, b in fpolys:
        cs = [c for c in circs if inside(b, c)]
        if n == 69:
            fixtures["bathtub"].append(b)
        elif n == 48:
            fixtures["washbasin" if cs else "wc"].append(b)
        elif n == 5 and any(0.03 <= c[2] <= 0.06 for c in cs) and max(b[2] - b[0], b[3] - b[1]) < 2.5:
            fixtures["shower"].append(b)
    # DXF -> PDF transform from room numbers
    doc = pymupdf.open(str(SW / f"{stem}.pdf")); page = doc[0]
    words = page.get_text("words")
    W = collections.defaultdict(list)
    for w in words:
        W[w[4]].append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, w[:4]))
    X, Y = [], []
    for r in rooms:
        if len(W.get(r["number"], [])) == 1:
            X.append(r["pt"]); Y.append(W[r["number"]][0][:2])
    T, resid = None, None
    if len(X) >= 2:
        X = np.array(X); Y = np.array(Y)
        A = np.zeros((2 * len(X), 3)); B = np.zeros(2 * len(X))
        A[0::2, 0] = X[:, 0]; A[0::2, 1] = 1; B[0::2] = Y[:, 0]
        A[1::2, 0] = -X[:, 1]; A[1::2, 2] = 1; B[1::2] = Y[:, 1]
        sol = np.linalg.lstsq(A, B, rcond=None)[0]
        T = [float(v) for v in sol]; resid = float(np.abs(A @ sol - B).max())
    f = {"stem": stem, "rooms": rooms, "units": units, "doors": doors, "windows": windows, "fixtures": dict(fixtures),
         "T": T, "T_resid_pt": resid, "page": [page.rect.width, page.rect.height],
         "pdf_words": [[w[4], [round(v, 1) for v in w[:4]]] for w in words]}
    # PDF check: ink present at each symbol's mapped bbox (vector drawings overlap)
    if T:
        drawings = page.get_drawings()
        dr = [d["rect"] for d in drawings]
        # room fill on the PDF: the filled path whose bbox matches the room outline. Grey = outdoor (balcony/loggia...) convention.
        for r in rooms:
            if not r["poly"]:
                r["fill"] = None; continue
            pb = bbox([to_pdf(T, q) for q in r["poly"]])
            m = [d for d in drawings if d.get("fill") is not None and abs(d["rect"].x0 - pb[0]) < 1.5 and abs(d["rect"].y0 - pb[1]) < 1.5
                 and abs(d["rect"].x1 - pb[2]) < 1.5 and abs(d["rect"].y1 - pb[3]) < 1.5]
            r["fill"] = round(sum(m[-1]["fill"]) / 3, 3) if m else None
        def ink(b):
            pb = to_pdf_bbox(T, b)
            return any(r.x0 <= pb[2] + 0.5 and r.x1 >= pb[0] - 0.5 and r.y0 <= pb[3] + 0.5 and r.y1 >= pb[1] - 0.5 for r in dr)
        f["ink"] = {"doors": [ink(b) for b in doors], "windows": [ink(b) for b in windows],
                    **{k: [ink(b) for b in v] for k, v in fixtures.items()}}
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(f))
    return f


def is_grey(r):
    """PDF renders outdoor rooms with a light-grey fill (0.894) and indoor rooms white."""
    return r.get("fill") is not None and 0.8 < r["fill"] < 0.95


def is_common(r):
    """Very light grey (0.953) = shared circulation (stairwell) drawn inside a unit outline; not part of the flat's own area."""
    return r.get("fill") is not None and 0.95 <= r["fill"] < 0.99


def to_pdf(T, p):
    a, b, c = T
    return (a * p[0] + b, -a * p[1] + c)


def to_pdf_bbox(T, bb):
    x0, y0 = to_pdf(T, (bb[0], bb[3])); x1, y1 = to_pdf(T, (bb[2], bb[1]))
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def label_in_pdf(f, r):
    """Second source: the PDF must carry the same number word, and the name word right above it."""
    if not f["T"]:
        return False
    x, y = to_pdf(f["T"], r["pt"])
    nums = [w for w in f["pdf_words"] if w[0] == r["number"] and w[1][0] - 2 <= x <= w[1][2] + 2 and w[1][1] - 2 <= y <= w[1][3] + 2]
    if not nums:
        return False
    if not r["name"]:
        return True
    first = r["name"].split()[0]
    nb = nums[0][1]
    return any(w[0] == first and nb[1] - 16 <= w[1][1] <= nb[1] for w in f["pdf_words"])


def room_ev(f, r, extra=""):
    it = {"source": "dxf+pdf", "kind": "room", "text": f"{r['number']} {r['name'] or ''} · {r['area']:.2f} m² (DXF polygon){extra}" if r["area"] else f"{r['number']} {r['name']}",
          "file": f["stem"] + ".pdf", "page": 0}
    if f["T"] and r["poly"]:
        pp = [list(map(lambda v: round(v, 1), to_pdf(f["T"], p))) for p in r["poly"]]
        it["poly"] = pp; it["bbox"] = bbox(pp)
    it["verified"] = label_in_pdf(f, r)
    return it


def sym_ev(f, kind, i, b):
    it = {"source": "dxf+pdf", "kind": kind, "text": f"{kind} #{i + 1} (DXF {kind} layer)", "file": f["stem"] + ".pdf", "page": 0}
    if f["T"]:
        it["bbox"] = to_pdf_bbox(f["T"], b)
        key = {"door": "doors", "window": "windows"}.get(kind, kind)
        it["verified"] = bool(f.get("ink", {}).get(key, [True] * (i + 1))[i])
    return it


# ------------------------------------------------------------------ question parsing
COUNT_KINDS = [("apartment", r"apartment|wohnungen|appartement|appartamenti|unità abitative"),
               ("door", r"\bdoors?\b|türen|portes\b|porte\b"),
               ("window", r"windows?|fenster|fenêtres|finestre"),
               ("wc", r"toilet|\bwcs?\b"),
               ("washbasin", r"washbasin|lavabo|lavabi"),
               ("bathtub", r"bathtub|badewanne|baignoire|vasche|vasca"),
               ("shower", r"shower|dusche|douche|docce|doccia")]


def label_of(q):
    for rx in [r"«\s*(.+?)\s*»", r"„(.+?)[“”]", r"\('(.+)'\)", r"labelled '(.+)' (?:are|on)\b", r"“(.+?)”", r"\"(.+?)\""]:
        m = re.search(rx, q)
        if m:
            return m.group(1).strip()
    return None


def parse(q):
    ql = q.lower()
    is_area = "m²" in q or "m2" in ql
    num = re.search(r"\b([A-Z]?\d+\.\d+)\b", q)
    apt = re.search(r"(?:apartment|wohnung|appartement|appartamento|dell'appartamento|l'appartement)\s+([A-Z0-9]+)\b", q, re.I)
    lab = label_of(q)
    if is_area:
        if num:
            return ("room_area", num.group(1), lab)
        if apt and not re.search(r"all numbered|nummerierten|numérotées|numerati", ql):
            return ("apt_area", apt.group(1), None)
        if re.search(r"numbered|nummerierten|numérotées|numerati", ql):
            return ("indoor_total", None, None)
        if lab:
            return ("label_area", lab, None)
        return ("unknown", None, None)
    if lab and re.search(r"labelled|bezeichnung|désignation|denominazione", ql):
        return ("label_count", lab, None)
    for k, rx in COUNT_KINDS:
        if re.search(rx, ql):
            return ("count", k, None)
    return ("unknown", None, None)


def answer(question, path):
    stem = Path(path).stem
    try:
        f = facts(stem)
    except Exception as e:
        return {"status": "abstain", "answer": f"NOT SURE: could not read {stem}.dxf ({type(e).__name__})", "value": None, "evidence": []}
    op, a1, a2 = parse(question)
    rooms = f["rooms"]
    if op == "count":
        if a1 == "apartment":
            ev = []
            for u in [u for u in f["units"] if u["label"]]:  # unlabelled outlines = stair cores / common areas
                it = {"source": "dxf+pdf", "kind": "apartment", "text": f"apartment {u['label'] or '?'} (UNIT_OUTLINE)", "file": stem + ".pdf", "page": 0}
                if f["T"]:
                    pp = [list(map(lambda v: round(v, 1), to_pdf(f["T"], p))) for p in u["poly"]]
                    it["poly"] = pp; it["bbox"] = bbox(pp)
                it["verified"] = bool(u["label"]) and any(w[0] == u["label"] for w in f["pdf_words"])
                ev.append(it)
            n = sum(1 for u in f["units"] if u["label"]); labs = sorted(u["label"] for u in f["units"] if u["label"])
            return _ok(n, f"{n} apartments (unit outlines {', '.join(labs)}).", ev)
        items = f["doors"] if a1 == "door" else f["windows"] if a1 == "window" else f["fixtures"].get(a1, [])
        ev = [sym_ev(f, a1, i, b) for i, b in enumerate(items)]
        ok = sum(1 for e in ev if e.get("verified"))
        return _ok(len(items), f"{len(items)} {a1}s drawn (DXF symbols); {ok}/{len(items)} confirmed by vector ink at the same spot on the PDF.", ev)
    if op == "label_count":
        sel = [r for r in rooms if norm(r["name"]) == norm(a1)]
        ev = [room_ev(f, r) for r in sel]
        pdf_n = sum(1 for w in f["pdf_words"] if norm(w[0]) == norm(a1)) if " " not in a1 else None
        msg = f"{len(sel)} rooms labelled {a1!r} ({', '.join(r['number'] for r in sel)})."
        if pdf_n is not None:
            msg += f" PDF text has {pdf_n} '{a1}' labels — " + ("agrees." if pdf_n == len(sel) else "differs.")
        return _ok(len(sel), msg, ev)
    if op == "room_area":
        r = next((r for r in rooms if r["number"] == a1), None)
        if not r or r["area"] is None:
            return _abst(f"room {a1} not found")
        if a2 and r["name"] and norm(a2) != norm(r["name"]):
            return {"status": "conflict", "answer": f"CONFLICT: room {a1} is labelled {r['name']!r} on the drawing, not {a2!r}. Area {r['area']:.2f} m².",
                    "value": round(r["area"], 2), "evidence": [room_ev(f, r)]}
        return _ok(round(r["area"], 2), f"Room {a1} ({r['name']}): {r['area']:.2f} m² (polygon area from DXF ROOM_OUTLINE).", [room_ev(f, r)])
    if op == "label_area":
        sel = [r for r in rooms if norm(r["name"]) == norm(a1) and r["area"]]
        if not sel:
            return _abst(f"no rooms labelled {a1!r}")
        v = sum(r["area"] for r in sel)
        return _ok(round(v, 2), f"{v:.2f} m² = " + " + ".join(f"{r['number']} {r['area']:.2f}" for r in sel) + ".", [room_ev(f, r) for r in sel])
    if op in ("apt_area", "indoor_total"):
        sel = [r for r in rooms if r["area"] and norm(r["name"]) not in OUTDOOR and norm(r["name"]) not in VOID and not is_grey(r)
               and not (op == "apt_area" and is_common(r)) and (op == "indoor_total" or r["unit"] == a1)]
        if op == "apt_area" and not any(u["label"] == a1 for u in f["units"]):
            return _abst(f"no apartment {a1} on this plan")
        v = sum(r["area"] for r in sel)
        excl = [r for r in rooms if r["area"] and (op == "indoor_total" or r["unit"] == a1) and r not in sel]
        msg = f"{v:.2f} m² = sum of {len(sel)} rooms" + (f" of apartment {a1}" if op == "apt_area" else "") + \
              (f", excluding {', '.join(r['number'] + ' ' + (r['name'] or '(unnamed)') + (' [grey=outdoor on PDF]' if is_grey(r) else ' [light grey=common on PDF]' if is_common(r) else '') for r in excl)}" if excl else "") + "."
        return _ok(round(v, 2), msg, [room_ev(f, r) for r in sel])
    return _abst("question type not recognised for swiss_dwellings")


def _ok(v, msg, ev):
    return {"status": "answered", "answer": msg, "value": v, "evidence": ev}


def _abst(why):
    return {"status": "abstain", "answer": f"NOT SURE: {why}.", "value": None, "evidence": []}
