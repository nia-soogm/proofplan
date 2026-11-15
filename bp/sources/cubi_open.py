"""Doors and windows on CubiCasa rasters from classical CV (CPU only, no network weights, no annotations at answer time).

1. Wall mask: walls are the thickest long strokes on the sheet. The wall thickness T is the mode of the ink's medial-axis
   width (distance transform ridges); the mask is the ink closed by 3 px (fills hatching) and opened by ~0.6 T (drops text,
   fixtures and thin lines).
2. Openings: wall-mask pixels in horizontal (vertical) runs >= 2 T are closed along the run with a line kernel up to 14 T.
   What the closing adds is a gap between two wall ends on one wall line, i.e. a candidate door or window opening.
   A gap must be about one wall thick and 1.5-14 T long.
3. Each opening is described by the thin ink inside it and around it:
   - window: thin lines running the whole length of the gap, parallel to the wall (glass / frame lines)
   - door: a quarter-circle swing arc of radius ~ the gap width hinged at one end of the gap (or two half-width arcs = a
     double door) plus the straight leaf, on either side of the wall.
4. Counts at a few strictness levels feed a dev-trained gate (scripts/cubi/train_open.py). Every counted opening is a box.

Status (Phase E, dev only): NOT SHIPPED. On dev OOF no threshold reaches the required Clopper-Pearson 95 % lower bound
>= 95 % precision (best: windows 46 % right at 1.2 % answered, doors 36 % at 0.7 %). Even where this detector and the
older arc-template detector (cubi_door) agree on a door count, the count matches the gold only ~30 % of the time.
No model file is written, so answer() returns None and doors/windows keep abstaining. See eval/cubicasa/train_open_dev_oof.txt.
"""
import functools, json
from pathlib import Path
import numpy as np
import cv2
from . import cubi_cv, CACHE

VERSION = 2
MAX_GAPS = 150


def wall_thickness(m):
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-3) & (dt >= 1.4)
    v = np.round(2 * dt[ridge]).astype(int)
    if len(v) < 20:
        return 3.0, 0.0
    h = np.bincount(v, minlength=64).astype(float)
    h[:3] = 0
    w = h * np.arange(len(h))  # favour thick strokes (walls are few but long)
    T = int(np.argmax(w))
    share = float(h[max(0, T - 1):T + 2].sum() / max(1.0, h.sum()))
    return float(max(3, T)), share


@functools.lru_cache(maxsize=4)
def walls(path):
    m = cubi_cv._ink(path)
    g = cv2.cvtColor(cubi_cv.load(path), cv2.COLOR_BGR2GRAY)
    dark = (m & (g < 120)).astype(np.uint8)  # walls are drawn dark (black / navy / green); window glass lines are lighter
    if dark.sum() < 0.3 * m.sum():
        dark = m
    T, share = wall_thickness(dark)
    c = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k = max(2, int(round(0.25 * T)))
    W = cv2.morphologyEx(c, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return m, W, T, share


def _arc_cov(near, cx, cy, r, a0, n=24):
    """Fraction of a quarter arc (centre cx, cy, radius r, angles a0..a0+90°) lying on ink, ignoring the end points."""
    th = np.linspace(a0 + 0.3, a0 + np.pi / 2 - 0.3, n)  # the middle of the arc: far from the wall line and its frame lines
    xs = np.round(cx + r * np.cos(th)).astype(int); ys = np.round(cy + r * np.sin(th)).astype(int)
    H, W = near.shape
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if ok.sum() < n * 0.7:
        return 0.0
    return float(near[ys[ok], xs[ok]].mean())


def _line_cov(near, x0, y0, x1, y1, n=16):
    t = np.linspace(0.3, 0.95, n)
    xs = np.round(x0 + (x1 - x0) * t).astype(int); ys = np.round(y0 + (y1 - y0) * t).astype(int)
    H, W = near.shape
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if ok.sum() < n * 0.7:
        return 0.0
    return float(near[ys[ok], xs[ok]].mean())


RF = (0.8, 0.87, 0.94, 1.0, 1.07, 1.14)  # swing radius / gap length (the gap box is only roughly the door width)


def _swing(near, inner_ink, hinge, sgn, side, c, th, w, horiz, T):
    """Best min(arc, leaf) - interior over a small search of radius, hinge shift and wall-face offset; also arc-only."""
    H, W = inner_ink.shape
    best, best_arc = -1.0, -1.0
    for off in (0.5, 0.0, 1.0):
        for d in (0.0, 0.05):
            for f in RF:
                r = f * w
                if r < 1.5 * T:
                    continue  # too small for a door leaf at this wall scale (window frames would match)
                h = hinge + sgn * d * w
                if horiz:
                    hx, hy = h, c + side * off * th
                    leaf = _line_cov(near, hx, hy, hx, hy + side * r)
                    a0 = (0.0 if side > 0 else -np.pi / 2) if sgn > 0 else (np.pi / 2 if side > 0 else np.pi)
                    ix, iy = hx + sgn * 0.45 * r, hy + side * 0.45 * r
                else:
                    hx, hy = c + side * off * th, h
                    leaf = _line_cov(near, hx, hy, hx + side * r, hy)
                    a0 = (0.0 if side > 0 else np.pi / 2) if sgn > 0 else (-np.pi / 2 if side > 0 else np.pi)
                    ix, iy = hx + side * 0.45 * r, hy + sgn * 0.45 * r
                arc = _arc_cov(near, hx, hy, r, a0)
                if arc < 0.5:
                    continue
                rr = max(1, int(0.2 * r))
                y0, y1, x0, x1 = int(max(0, iy - rr)), int(min(H, iy + rr + 1)), int(max(0, ix - rr)), int(min(W, ix + rr + 1))
                inner = float(inner_ink[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 1.0
                best = max(best, min(arc, leaf) - inner)
                best_arc = max(best_arc, arc - inner)
    return best, best_arc


def _door_score(near, inner_ink, a, b, c, th, horiz, T):
    """Swing test for a gap from a to b (along the wall axis), wall centre line c, wall thickness th (px).
    Hinge at either end, on either side of the wall; leaf perpendicular to the wall, arc from the leaf tip to the far end
    of the opening. Returns (single-leaf score, double-leaf score, arc-only score); score = min(arc, leaf) - interior ink."""
    w = b - a
    s1, s2, sa = 0.0, 0.0, 0.0
    for side in (-1, 1):
        half = []
        for hinge, sgn in ((a, 1), (b, -1)):
            x, xa = _swing(near, inner_ink, hinge, sgn, side, c, th, w, horiz, T)
            s1 = max(s1, x); sa = max(sa, xa)
            half.append(_swing(near, inner_ink, hinge, sgn, side, c, th, w / 2, horiz, T)[0])
        s2 = max(s2, min(half))
    return s1, s2, sa


@functools.lru_cache(maxsize=4)
def openings(path):
    """-> (list of opening dicts, T, share). Each: bbox [x0,y0,x1,y1], horiz, w/T, window features, door scores."""
    m, W, T, share = walls(path)
    H, Wd = m.shape
    near = (cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3) <= 2.0).astype(np.float32)
    thin = (m & (1 - cv2.dilate(W, np.ones((3, 3), np.uint8)))).astype(np.uint8)  # ink that is not wall
    Ti = int(round(T))
    thinf = thin.astype(np.float32)
    im = cubi_cv.load(path)
    out = []
    for horiz in (True, False):
        run = max(4, 2 * Ti)
        kr = np.ones((1, run), np.uint8) if horiz else np.ones((run, 1), np.uint8)
        Wa = cv2.morphologyEx(W, cv2.MORPH_OPEN, kr)
        L = max(8, 25 * Ti)
        kc = np.ones((1, L), np.uint8) if horiz else np.ones((L, 1), np.uint8)
        gap = (cv2.morphologyEx(Wa, cv2.MORPH_CLOSE, kc) & (1 - W)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(gap, connectivity=4)
        if n > MAX_GAPS:  # outline / hatched walls: the wall mask is not solid and gaps explode -> no opening analysis
            return [], T, share, True
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in st[i])
            ln, th = (w, h) if horiz else (h, w)
            if not (0.2 * T <= th <= 1.8 * T and ln >= 1.5 * T and ln <= 25.5 * T) or x == 0 or y == 0 or x + w >= Wd or y + h >= H:
                continue
            if area < 0.6 * w * h:
                continue
            # window: thin ink lines spanning the gap along the wall
            sub = thin[y:y + h, x:x + w] if horiz else thin[y:y + h, x:x + w].T  # rows = across the wall, cols = along it
            ln_cols = sub.shape[1]
            inner = sub[:, max(0, int(0.08 * ln_cols)):max(1, int(0.92 * ln_cols))]
            rowfrac = inner.mean(axis=1) if inner.size else np.zeros(1)
            n_lines = int(((rowfrac >= 0.85).astype(np.int8)[1:] > (rowfrac >= 0.85).astype(np.int8)[:-1]).sum()
                          + int(rowfrac[0] >= 0.85)) if len(rowfrac) else 0
            colspan = float((inner.max(axis=0) > 0).mean()) if inner.size else 0.0
            ink_frac = float(inner.mean()) if inner.size else 0.0
            # colour of the thin ink in the gap (blue glass on coloured plans)
            px = im[y:y + h, x:x + w][thin[y:y + h, x:x + w] > 0]
            blue = float(((px[:, 0].astype(int) - px[:, 2].astype(int)) > 40).mean()) if len(px) else 0.0
            a, b = (x, x + w) if horiz else (y, y + h)
            c = y + h / 2 if horiz else x + w / 2
            d1, d2, da = _door_score(near, thinf, a, b, c, th, horiz, T)
            out.append({"bbox": [x, y, x + w, y + h], "horiz": horiz, "len": ln / T, "th": th / T, "n_lines": n_lines,
                        "colspan": colspan, "ink": ink_frac, "maxrow": float(rowfrac.max()) if len(rowfrac) else 0.0,
                        "blue": blue, "door1": d1, "door2": d2, "door_arc": da})
    return out, T, share, False


WIN_LEVELS = ((1, 0.85), (2, 0.85), (1, 0.95), (2, 0.95))  # (min lines, min colspan)
DOOR_LEVELS = (0.45, 0.6, 0.75)


def is_window(o, nl, cs):
    return o["n_lines"] >= nl and o["colspan"] >= cs and max(o["door1"], o["door2"]) < 0.45


def is_door(o, t):
    return max(o["door1"], o["door2"]) >= t


def features(path):
    """Image-level counts and summary stats (cached). Pure function of the pixels."""
    cf = CACHE / "cubi_open" / (Path(path).stem + ".json")
    if cf.exists():
        d = json.loads(cf.read_text())
        if d.get("v") == VERSION:
            return d
    ops, T, share, exploded = openings(path)
    H, W = cubi_cv._ink(path).shape
    f = {"v": VERSION, "T": T, "T_share": share, "T_rel": T / max(H, W) * 1000, "n_gap": len(ops), "W": W, "H": H,
         "n_gap_h": sum(o["horiz"] for o in ops), "exploded": int(exploded)}
    for nl, cs in WIN_LEVELS:
        f[f"win_{nl}_{int(cs * 100)}"] = sum(is_window(o, nl, cs) for o in ops)
    for t in DOOR_LEVELS:
        f[f"door_{int(t * 100)}"] = sum(is_door(o, t) for o in ops)
    f["win_blue"] = sum(o["blue"] > 0.5 and o["colspan"] >= 0.85 for o in ops)
    f["empty_gap"] = sum(o["ink"] < 0.02 for o in ops)
    f["ops"] = [[int(v) for v in o["bbox"]] + [round(o[k], 3) for k in ("len", "th", "colspan", "ink", "maxrow", "blue", "door1", "door2", "door_arc")]
                + [o["n_lines"], int(o["horiz"])] for o in ops]
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(f))
    return f


# ---- candidates for the dev-trained count gate -------------------------------------------------------------------------
KINDS = ("door", "window")
DOOR_T = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
WIN_T = ((1, 0.85, 0.45), (1, 0.95, 0.45), (2, 0.85, 0.45), (1, 0.85, 0.6), (1, 0.7, 0.45), (1, 0.85, 0.3))  # lines, colspan, max door score
TYP = {"colorful": 0, "high_quality": 1, "high_quality_architectural": 2}
# per-opening record layout in features()["ops"]
I_LEN, I_TH, I_CS, I_INK, I_MAXROW, I_BLUE, I_D1, I_D2, I_DA, I_NL, I_H = range(4, 15)


def sources(f):
    """-> {(kind, source name): list of opening indices counted by that source}."""
    ops = f["ops"]
    S = {}
    for t in DOOR_T:
        S[("door", f"d{int(t * 100)}")] = [i for i, o in enumerate(ops) if max(o[I_D1], o[I_D2]) >= t]
        S[("door", f"s{int(t * 100)}")] = [i for i, o in enumerate(ops) if o[I_D1] >= t]
    for nl, cs, dm in WIN_T:
        S[("window", f"w{nl}_{int(cs * 100)}_{int(dm * 100)}")] = [i for i, o in enumerate(ops)
                                                                  if o[I_NL] >= nl and o[I_CS] >= cs and max(o[I_D1], o[I_D2]) < dm]
    S[("window", "blue")] = [i for i, o in enumerate(ops) if o[I_BLUE] > 0.5 and o[I_CS] >= 0.85]
    return S


def label_feats(path):
    from . import cubicasa, cubi_rooms as R, cubi_fix as F
    o = cubicasa.ocr2(path)
    L = F.label_info(R.extract(o["tokens"]))
    return [L[k] for k in sorted(L)]


def candidates(kind, path):
    """-> list of (value, source tags, feature vector, opening indices for the evidence boxes)."""
    f = features(path)
    if f["exploded"] or not f["ops"]:
        return []
    S = sources(f)
    mine = {n: ix for (k, n), ix in S.items() if k == kind}
    names = sorted(mine)
    counts = [len(mine[n]) for n in names]
    other = [len(ix) for (k, n), ix in sorted(S.items()) if k != kind]
    stem = Path(path).stem.rsplit("-", 1)[0]
    base = [f["T"], f["T_share"], f["T_rel"], f["n_gap"], f["n_gap_h"], f["W"], f["H"], TYP.get(stem, 1), f["empty_gap"]] \
        + counts + other + label_feats(path)
    by_v = {}
    for n, c in zip(names, counts):
        by_v.setdefault(c, []).append(n)
    out = []
    sv = sorted(by_v)
    for v, tags in sorted(by_v.items()):
        fv = base + [v, len(tags), sv.index(v), len(sv), v - np.median(counts), v / max(1, f["n_gap"])] + [int(n in tags) for n in names]
        out.append((v, tags, fv, mine[tags[0]]))
    return out


@functools.lru_cache(maxsize=1)
def model():
    import pickle
    fm = Path(__file__).resolve().parent.parent / "models" / "cubi_open.pkl"
    return pickle.load(open(fm, "rb")) if fm.exists() else None


def answer(path, kind):
    """-> None (no model / kind disabled) or dict(status, value, p, tau, boxes, tags)."""
    M = model()
    if M is None or not M["enabled"].get(kind):
        return None
    cs = candidates(kind, path)
    if not cs:
        f = features(path)
        return {"status": "abstain", "why": "the wall outline could not be traced cleanly (outline/hatched walls)" if f["exploded"]
                else "no wall openings found"}
    X = np.array([c[2] for c in cs], float)
    p = M["models"][kind].predict_proba(X)[:, 1]
    i = int(p.argmax())
    v, tags, fv, ix = cs[i]
    from .calib import tau_for
    tau = float(tau_for(M, kind))
    ops = features(path)["ops"]
    boxes = [ops[j][:4] for j in ix]
    return {"status": "answered" if p[i] >= tau else "abstain", "value": int(v), "p": float(p[i]), "tau": tau,
            "boxes": boxes, "tags": tags, "n_gap": features(path)["n_gap"]}
