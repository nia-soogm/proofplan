"""Classical CV on CubiCasa rasters (CPU, no model weights, no annotations at answer time).

- plan_parts(): splits the sheet into separate plan drawings (recursive XY-cut on the ink mask). Two drawings side by
  side on one sheet are almost always two storeys of the same dwelling; each part becomes an evidence box.
"""
import functools
from pathlib import Path
import numpy as np
import cv2
from . import JDATA
VALLEYS = (0.0, 0.08, 0.15, 0.25)  # a column/row is 'blank' if its ink is below this fraction of the plan's median (dimension lines cross gaps)


@functools.lru_cache(maxsize=8)
def load(path):
    p = Path(path)
    p = p if p.is_absolute() else JDATA / p
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:  # transparent background -> white
        a = im[:, :, 3:4].astype(np.float32) / 255
        im = (im[:, :, :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    return im


def ink_mask(im, thr=None):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if thr is None:
        thr = min(200, int(np.percentile(g, 50)) - 40) if np.percentile(g, 50) > 120 else 100
    m = (g < thr).astype(np.uint8)
    # drop tiny specks (scan noise); keep text and lines
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, np.uint8); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(4, m.size // 400000)
    return keep[lab]


def _cut(m, x0, y0, depth, out, gap_frac, min_ink, valley):
    h, w = m.shape
    tot = m.sum()
    if depth > 3 or tot == 0:
        out.append((x0, y0, w, h, int(tot))); return
    best = None
    for axis in (1, 0):  # 1: split along x (vertical gap) first, 0: along y
        prof = m.sum(axis=0 if axis == 1 else 1).astype(float)
        L = len(prof)
        nz = np.nonzero(prof)[0]
        if len(nz) == 0:
            continue
        lo, hi = nz[0], nz[-1]
        med = np.median(prof[lo:hi + 1][prof[lo:hi + 1] > 0])
        blank = prof <= max(1.0, valley * med, 0.004 * (h if axis == 1 else w))
        g_min = max(6, int(gap_frac * L))
        i = lo
        while i <= hi:
            if blank[i]:
                j = i
                while j <= hi and blank[j]:
                    j += 1
                if j - i >= g_min:
                    c = (i + j) // 2
                    a_ink, b_ink = prof[:c].sum(), prof[c:].sum()
                    if min(a_ink, b_ink) >= min_ink:
                        score = (j - i) / L
                        if best is None or score > best[0]:
                            best = (score, axis, c)
                i = j
            else:
                i += 1
    if best is None:
        nz_y, nz_x = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        out.append((x0 + nz_x[0], y0 + nz_y[0], nz_x[-1] - nz_x[0] + 1, nz_y[-1] - nz_y[0] + 1, int(tot))); return
    _, axis, c = best
    if axis == 1:
        _cut(m[:, :c], x0, y0, depth + 1, out, gap_frac, min_ink, valley); _cut(m[:, c:], x0 + c, y0, depth + 1, out, gap_frac, min_ink, valley)
    else:
        _cut(m[:c], x0, y0, depth + 1, out, gap_frac, min_ink, valley); _cut(m[c:], x0, y0 + c, depth + 1, out, gap_frac, min_ink, valley)


@functools.lru_cache(maxsize=8)
def _ink(path):
    return ink_mask(load(path))


@functools.lru_cache(maxsize=8)
def _ink_layout(path):
    """Ink without bright saturated strokes (green sheet frames, highlighter) that would bridge two drawings."""
    im = load(path)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    bright_col = (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 150)
    return (_ink(path) & ~bright_col).astype(np.uint8)


@functools.lru_cache(maxsize=64)
def plan_parts(path, valley=0.08, gap_frac=0.015, min_ink_frac=0.08):
    """Boxes (x, y, w, h, ink) of separate drawings on the sheet, largest first."""
    m = _ink_layout(path)
    out = []
    _cut(m, 0, 0, 0, out, gap_frac, min_ink_frac * m.sum(), valley)
    return sorted(out, key=lambda b: -b[4])


def big_parts(path, valley):
    parts = plan_parts(path, valley)
    tot = sum(p[4] for p in parts) or 1
    return [p for p in parts if p[4] >= 0.15 * tot], parts, tot


def storey_features(path):
    im = load(path); H, W = im.shape[:2]
    f = {"W": W, "H": H, "aspect": W / H}
    for v in VALLEYS:
        big, parts, tot = big_parts(path, v)
        t = f"v{int(v * 100)}_"
        areas = sorted([p[2] * p[3] for p in big], reverse=True)
        f[t + "n_parts"] = len(parts); f[t + "n_big"] = len(big)
        f[t + "ink1"] = parts[0][4] / tot
        f[t + "ink2"] = parts[1][4] / tot if len(parts) > 1 else 0
        f[t + "area_ratio"] = areas[1] / areas[0] if len(areas) > 1 else 0
        if len(big) > 1:  # footprint similarity of the two largest drawings (storeys of one house share the outline)
            a, b = sorted(big, key=lambda p: -p[2] * p[3])[:2]
            f[t + "wsim"] = min(a[2], b[2]) / max(a[2], b[2]); f[t + "hsim"] = min(a[3], b[3]) / max(a[3], b[3])
        else:
            f[t + "wsim"] = f[t + "hsim"] = 0
    return f


STOREY_WORDS = ("KRS", "KERROS", "ALAKERTA", "YLÄKERTA", "YLAKERTA", "KELLARI", "ULLAKKO", "PARVI", "POHJAKERROS", "VÄLIKERROS", "1.KRS", "2.KRS")


def storey_ocr_features(path, parts_by_valley):
    """OCR cues: storey names ('1. KRS', 'YLÄKERTA', 'KELLARI'), and how many separate drawings carry room labels."""
    from . import cubicasa
    import re
    toks = [t for t in cubicasa.ocr(path)["tokens"] if t["conf"] >= 0.5]
    rooms = [(t, cubicasa.classify(t["text"])) for t in toks]
    rooms = [(t, c) for t, c in rooms if c]
    up = [re.sub(r"\s+", "", t["text"].upper()) for t in toks]
    st = sorted({u for u in up if any(w in u for w in STOREY_WORDS) and len(u) <= 14})
    oh = sum(1 for u in up if re.fullmatch(r"OH|OLOH|OLOHUONE|TUPA", re.sub(r"[\d.,]", "", u)))
    f = {"n_storey_words": len(st), "n_room_labels": len(rooms), "n_kitchen": sum(c == "kitchen" for _, c in rooms),
         "n_oh": oh, "n_sauna": sum(c == "sauna" for _, c in rooms), "n_tokens": len(toks),
         "has_upper": int(any("YLÄ" in u or "YLA" in u or "2.KRS" in u or "2KRS" in u for u in up)),
         "has_lower": int(any("ALAK" in u or "1.KRS" in u or "1KRS" in u or "KELLARI" in u for u in up))}
    for v, big in parts_by_valley.items():
        k = 0
        for (x, y, w, h, _) in big:
            n = sum(1 for t, _ in rooms if x <= (t["bbox"][0] + t["bbox"][2]) / 2 <= x + w and y <= (t["bbox"][1] + t["bbox"][3]) / 2 <= y + h)
            k += n >= 2
        f[f"v{int(v * 100)}_parts_lab"] = k
    return f, st


def storey_all_features(path):
    import json
    from . import CACHE
    cf = CACHE / "cubi_storey" / (Path(path).stem + ".json")
    if cf.exists():
        return json.loads(cf.read_text())
    f = storey_features(path)
    pv = {v: big_parts(path, v)[0] for v in VALLEYS}
    f2, _ = storey_ocr_features(path, pv)
    f.update(f2)
    f = {k: float(v) for k, v in f.items()}
    cf.parent.mkdir(parents=True, exist_ok=True); cf.write_text(json.dumps(f))
    return f
