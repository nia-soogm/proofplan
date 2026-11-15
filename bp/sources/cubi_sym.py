"""Fixture symbols on CubiCasa rasters (WC seats, sinks, showers/baths, sauna stoves) without a neural net.

1. Scale: the median height of OCR'd room labels ("MH", "KH", ...) ≈ 0.3 m on the plan; fallback from the image size.
2. Candidates: enclosed white regions (a WC bowl, the basin of a sink, a tub) whose size is plausible for that scale.
3. Each candidate gets a descriptor (normalised ink patch, rotation-pooled HOG + shape numbers) and a small
   classifier (trained on dev images only, see scripts/cubi/train_sym.py) scores it as wc / sink / bath / other.
4. Count = candidates above the class threshold after NMS; every one is returned as an evidence box.
"""
import collections, functools, json, pickle
from pathlib import Path
import numpy as np
import cv2
from . import cubi_cv, cubicasa, CACHE

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
PS = 32  # patch size


def text_h(path):
    o = cubicasa.ocr(path)
    hs = [min(t["bbox"][3] - t["bbox"][1], t["bbox"][2] - t["bbox"][0]) for t in o["tokens"]
          if t["conf"] >= 0.5 and cubicasa.classify(t["text"]) and len(t["text"].strip()) <= 6]
    if len(hs) >= 2:
        return float(np.median(hs)), len(hs)
    H, W = cubi_cv.load(path).shape[:2]
    return max(H, W) / 60.0, 0


@functools.lru_cache(maxsize=16)
def candidates(path):
    """List of dicts {bbox, area, fill, aspect, rel, nholes, patch} for enclosed white regions of plausible size."""
    im = cubi_cv.load(path)
    m = cubi_cv._ink(path)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    s, nlab = text_h(path)
    bg = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    H, W = m.shape
    tb = np.array([t["bbox"] for t in cubicasa.ocr(path)["tokens"] if t["conf"] >= 0.3 and len(t["text"].strip()) >= 1]).reshape(-1, 4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if not (0.6 * s <= max(w, h) <= 6 * s and a >= 0.25 * s * s) or x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        cx, cy = x + w / 2, y + h / 2
        if len(tb) and ((tb[:, 0] <= cx) & (cx <= tb[:, 2]) & (tb[:, 1] <= cy) & (cy <= tb[:, 3])).any():
            continue  # a letter/digit hole ('O', '6', '0') inside OCR'd text
        fill = a / (w * h)
        if fill < 0.35 or max(w, h) > 3 * min(w, h) or min(w, h) < 0.35 * s:
            continue
        pad = int(0.35 * max(w, h)) + 2
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        reg = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c); ha = cv2.contourArea(hull) or 1
        ell = 0.0
        if len(c) >= 5:
            (_, _), (ea, eb), _ = cv2.fitEllipse(c)
            ell = a / (np.pi * ea * eb / 4 + 1e-6)
        peri = cv2.arcLength(c, True)
        sub_ink = m[y0:y1, x0:x1]
        out.append({"bbox": [x, y, x + w, y + h], "area": a, "fill": fill, "aspect": max(w, h) / max(1, min(w, h)),
                    "rel": max(w, h) / s, "rel_a": a / (s * s), "solid": a / ha, "ell": ell, "circ": 4 * np.pi * a / (peri * peri + 1e-6),
                    "ink_around": float(sub_ink.mean()), "crop": (x0, y0, x1, y1), "region": reg})
    return out, s, nlab


def _hog(p, cell=8, nb=9):
    """Plain HOG: unsigned gradient orientation histograms on cell x cell blocks, L2-normalised (OpenCV 5 dropped HOGDescriptor)."""
    f = p.astype(np.float32) / 255
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy); ang = (np.arctan2(gy, gx) % np.pi) / np.pi * nb
    b = np.minimum(ang.astype(int), nb - 1)
    n = p.shape[0] // cell
    h = np.zeros((n, n, nb), np.float32)
    for k in range(nb):
        mk = (mag * (b == k)).reshape(n, cell, n, cell).sum(axis=(1, 3))
        h[:, :, k] = mk
    h /= np.linalg.norm(h) + 1e-6
    return h.ravel()


def descriptor(path, c):
    m = cubi_cv._ink(path)
    x0, y0, x1, y1 = c["crop"]
    # square crop centred on the candidate
    cx, cy = (c["bbox"][0] + c["bbox"][2]) / 2, (c["bbox"][1] + c["bbox"][3]) / 2
    r = 0.5 * max(x1 - x0, y1 - y0)
    H, W = m.shape
    X0, Y0, X1, Y1 = int(cx - r), int(cy - r), int(cx + r), int(cy + r)
    pad = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, X0), max(0, Y0), min(W, X1), min(H, Y1)
    pad[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = m[sy0:sy1, sx0:sx1] * 255
    p = cv2.resize(pad, (PS, PS), interpolation=cv2.INTER_AREA)
    # rotation pooling: HOG of the 4 right-angle rotations, summed (symbols appear in any orientation)
    hs = [_hog(np.ascontiguousarray(np.rot90(p, k))) for k in range(4)]
    hog = np.sort(np.stack(hs), axis=0).ravel()  # sorted per-dimension -> rotation-invariant but keeps detail
    small = cv2.resize(pad, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    rad = np.array([small[3:9, 3:9].mean(), small.mean(), small[0].mean() + small[-1].mean() + small[:, 0].mean() + small[:, -1].mean()])
    shape = np.array([c["fill"], c["aspect"], c["rel"], c["rel_a"], c["solid"], c["ell"], c["circ"], c["ink_around"]])
    return np.concatenate([shape, rad, hog]).astype(np.float32), p


def cached_desc(path):
    """(boxes, descriptors, scale, n_labels) for an image, cached on disk (pure function of the pixels + OCR)."""
    f = CACHE / "cubi_desc" / (Path(path).stem + ".npz")
    if f.exists():
        z = np.load(f)
        return z["boxes"], z["X"], float(z["s"]), int(z["nlab"])
    cs, X, s, nlab = all_desc(path)
    boxes = np.array([c["bbox"] for c in cs], np.int32).reshape(-1, 4)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, boxes=boxes, X=X, s=s, nlab=nlab)
    return boxes, X, s, nlab


def cached_doors(path, s):
    f = CACHE / "cubi_door" / (Path(path).stem + ".json")
    if f.exists():
        return json.loads(f.read_text())
    from . import cubi_door
    out = {"s": s, "strict": cubi_door.detect(path, s, *cubi_door.STRICT), "loose": cubi_door.detect(path, s, *cubi_door.LOOSE)}
    f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out))
    return out


def all_desc(path):
    cs, s, nlab = candidates(path)
    if not cs:
        return cs, np.zeros((0, 1), np.float32), s, nlab
    D = np.stack([descriptor(path, c)[0] for c in cs])
    return cs, D, s, nlab


def nms_count(bx, sc, thr, merge=1.0):
    """Greedy grouping: candidates above thr; candidates closer than merge*size to a kept one join it (double bowls)."""
    idx = [i for i in np.argsort(-sc) if sc[i] >= thr]
    kept = []
    for i in idx:
        x0, y0, x1, y1 = bx[i]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = max(x1 - x0, y1 - y0)
        dup = False
        for j in kept:
            a0, b0, a1, b1 = bx[j]
            if abs(cx - (a0 + a1) / 2) < merge * max(r, a1 - a0, b1 - b0) and abs(cy - (b0 + b1) / 2) < merge * max(r, a1 - a0, b1 - b0):
                dup = True; break
        if not dup:
            kept.append(i)
    return kept


def image_feats(bx, P, ci, s, nlab, thr, merge, p):
    sc = P[:, ci] if len(P) else np.zeros(0)
    kept = nms_count(bx, sc, thr, merge) if len(sc) else []
    n = len(kept)
    srt = np.sort(sc)[::-1] if len(sc) else np.zeros(0)
    inside = sc[kept] if kept else np.array([1.0])
    out = [v for v in srt if v < thr]
    typ = Path(p).stem.rsplit("-", 1)[0]
    f = [n, inside.min(), inside.mean(), out[0] if out else 0, sum(1 for v in sc if 0.2 < v < thr), s, nlab, len(sc),
         {"colorful": 0, "high_quality": 1, "high_quality_architectural": 2}[typ]]
    return n, kept, f


def context(path):
    """OCR room-label context for the count gate: bathrooms/WCs/kitchens/saunas read, apartment type 'nH'."""
    import re
    toks = [t for t in cubicasa.ocr(path)["tokens"] if t["conf"] >= 0.5]
    cl = collections.Counter(cubicasa.classify(t["text"]) for t in toks)
    h = [int(m.group(1)) for t in toks for m in [cubicasa.SUMMARY_RE.search(t["text"].replace(" ", ""))] if m]
    return [cl["bathroom"], cl["wc"], cl["kitchen"], cl["sauna"], cl["bedroom"], h[0] if h else -1, sum(v for k, v in cl.items() if k)]


def gate_feats(path, bx, P, ci, s, nlab, thr, merge):
    n, kept, f = image_feats(bx, P, ci, s, nlab, thr, merge, path)
    other = [len(nms_count(bx, P[:, j], 0.5, 1.0)) if len(P) else 0 for j in range(1, P.shape[1])] if len(P) else [0, 0, 0]
    return n, kept, f + context(path) + other


def door_feats(path, det, which, bx, P, s, nlab):
    """det = {'strict': [...], 'loose': [...]} arc detections; answer value = len(det[which])."""
    n = len(det[which])
    ci = 3
    closed = len(nms_count(bx, P[:, ci], 0.5, 0.6)) if len(P) else 0
    typ = Path(path).stem.rsplit("-", 1)[0]
    sc = [d[4] for d in det[which]] or [0]
    return n, [n, len(det["strict"]), len(det["loose"]), len(det["loose"]) - len(det["strict"]), closed, min(sc), float(np.mean(sc)), s, nlab,
               {"colorful": 0, "high_quality": 1, "high_quality_architectural": 2}[typ]] + context(path)


SYM_CLASS = {"wc_seat": "wc", "sink": "sink", "door": "door"}
NAMES = {"wc_seat": ("WC seat", "WC seats"), "sink": ("sink/basin", "sinks/basins"), "door": ("door", "doors"),
         "window": ("window", "windows"), "shower_bath": ("shower/bathtub", "showers/bathtubs")}


@functools.lru_cache(maxsize=1)
def _model():
    f = MODEL_DIR / "cubi_sym.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def _tau(prm):
    """Point-rule threshold of the detector gate (superseded by cubi_fix for WC seats); it has no strict threshold -> never answers in strict mode."""
    from .calib import strict
    return 1.01 if strict() else prm["tau"]


def count(path, kind):
    """-> (n, confidence, gate threshold, evidence boxes) or None when there is no model for the kind."""
    M = _model()
    cname = SYM_CLASS.get(kind)
    if not M or cname not in M["gates"] or not M["params"].get(cname):
        return None
    prm = M["params"][cname]; ci = M["classes"].index(cname)
    bx, X, s, nlab = cached_desc(path)
    P = M["cand"].predict_proba(X) if len(bx) else np.zeros((0, len(M["classes"])))
    if kind == "door":
        det = cached_doors(path, s)
        n, f = door_feats(path, det, prm["which"], bx, P, s, nlab)
        conf = float(M["gates"][cname].predict_proba(np.array([f], float))[0, 1])
        ev = [{"bbox": [x0, y0, x1, y1], "page": 0, "kind": kind, "text": f"door swing #{i + 1} (arc + leaf, score {sc:.2f})", "verified": None}
              for i, (x0, y0, x1, y1, sc) in enumerate(det[prm["which"]])]
        return n, conf, _tau(prm), ev
    n, kept, f = gate_feats(path, bx, P, ci, s, nlab, prm["thr"], prm["merge"])
    conf = float(M["gates"][cname].predict_proba(np.array([f], float))[0, 1])
    ev = []
    for i in kept:
        x0, y0, x1, y1 = (int(v) for v in bx[i])
        pad = int(0.25 * max(x1 - x0, y1 - y0)) + 1
        ev.append({"bbox": [x0 - pad, y0 - pad, x1 + pad, y1 + pad], "page": 0, "kind": kind,
                   "text": f"{NAMES[kind][0]} symbol #{len(ev) + 1} (score {P[i, ci]:.2f})", "verified": None})
    return n, conf, _tau(prm), ev
