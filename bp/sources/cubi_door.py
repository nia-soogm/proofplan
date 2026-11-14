"""Door swings on CubiCasa rasters: quarter-circle arc + straight leaf from the hinge (classical template coverage).

The sheet is rescaled so the OCR'd room-label height is ~10 px (door leaf ≈ 2–6 label heights). For each radius and
each of the 8 arc/leaf orientations a sparse template (arc points, leaf points) is correlated with the binary map
"within 1.5 px of ink"; a swing is a hinge position where both arc and leaf are mostly covered while the quarter-disc
interior is mostly empty. Overlapping hits are merged (NMS). Each hit is an evidence box.
"""
import functools
import numpy as np
import cv2
from . import cubi_cv

S_NORM = 8.0


def _templates(r):
    """8 (arc kernel, leaf kernel, interior kernel, bbox offsets) for radius r around the hinge at the kernel centre."""
    R = int(np.ceil(1.2 * r)) + 2
    size = 2 * R + 1
    out = []
    for q in range(4):
        a0 = q * np.pi / 2
        th = np.linspace(a0 + 0.08, a0 + np.pi / 2 - 0.08, max(12, int(r * 1.2)))
        arc = np.zeros((size, size), np.float32)
        for t in th:
            arc[int(round(R + r * np.sin(t))), int(round(R + r * np.cos(t)))] = 1
        ring = np.zeros((size, size), np.float32)  # just inside/outside the arc: must be empty for a thin curved stroke
        for rr in (0.8 * r, 1.2 * r):
            for t in th[2:-2]:
                yy, xx = int(round(R + rr * np.sin(t))), int(round(R + rr * np.cos(t)))
                if 0 <= yy < size and 0 <= xx < size:
                    ring[yy, xx] = 1
        inner = np.zeros((size, size), np.float32)
        for rr in np.linspace(0.25 * r, 0.8 * r, 5):
            for t in th[::2]:
                inner[int(round(R + rr * np.sin(t))), int(round(R + rr * np.cos(t)))] = 1
        for side in (0, 1):
            t = a0 if side == 0 else a0 + np.pi / 2
            leaf = np.zeros((size, size), np.float32)
            for rr in np.linspace(0.15 * r, 0.95 * r, max(8, int(r))):
                leaf[int(round(R + rr * np.sin(t))), int(round(R + rr * np.cos(t)))] = 1
            # bbox of the quarter disc relative to the hinge
            xs = [0, r * np.cos(a0), r * np.cos(a0 + np.pi / 2), r * np.cos(a0 + np.pi / 4)]
            ys = [0, r * np.sin(a0), r * np.sin(a0 + np.pi / 2), r * np.sin(a0 + np.pi / 4)]
            out.append((arc / arc.sum(), leaf / leaf.sum(), inner / inner.sum(), ring / ring.sum(), (min(xs), min(ys), max(xs), max(ys)), q, side))
    return out


@functools.lru_cache(maxsize=8)
def _raw_hits(path, s):
    """All hinge positions passing the LOOSE thresholds, with their component scores (one pass; strict = a filter)."""
    m = cubi_cv._ink(path)
    f = S_NORM / max(s, 1e-3)
    f = min(2.0, max(0.25, f))
    H, W = m.shape
    ms = cv2.resize(m * 255, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA) > 60
    near = (cv2.distanceTransform((~ms).astype(np.uint8), cv2.DIST_L2, 3) <= 1.5).astype(np.float32)
    inkf = ms.astype(np.float32)
    ca_min, cl_min, ci_max, cr_max = LOOSE
    hits = []
    for r in S_NORM * np.geomspace(1.8, 6.5, 9):
        for arc, leaf, inner, ring, (bx0, by0, bx1, by1), q, side in _templates(r):
            ca = cv2.filter2D(near, -1, arc, borderType=cv2.BORDER_CONSTANT)
            ys, xs = np.nonzero(ca >= ca_min)
            if not len(ys):
                continue
            def at(K):  # sparse correlation only at the surviving hinge positions
                R = K.shape[0] // 2
                ky, kx = np.nonzero(K)
                wv = K[ky, kx]
                pad = np.pad(near if K is not inner else inkf, R)
                return (pad[ys[:, None] + ky[None, :], xs[:, None] + kx[None, :]] * wv[None, :]).sum(1)
            cl = at(leaf); ok = cl >= cl_min
            ys, xs, cl = ys[ok], xs[ok], cl[ok]
            if not len(ys):
                continue
            ci = at(inner); ok = ci <= ci_max
            ys, xs, cl, ci = ys[ok], xs[ok], cl[ok], ci[ok]
            if not len(ys):
                continue
            cr = at(ring); ok = cr <= cr_max
            for y, x, l, i, rr in zip(ys[ok], xs[ok], cl[ok], ci[ok], cr[ok]):
                a_ = float(ca[y, x])
                hits.append((float(a_ + 0.5 * l - i - rr), (x + bx0) / f, (y + by0) / f, (x + bx1) / f, (y + by1) / f, a_, float(l), float(i), float(rr)))
    hits.sort(key=lambda h: -h[0])
    return hits


def detect(path, s, cov_arc=0.8, cov_leaf=0.75, max_inner=0.1, max_ring=0.3):
    """-> list of (x0, y0, x1, y1, score) in original pixel coordinates, after NMS."""
    hits = [h for h in _raw_hits(path, s) if h[5] >= cov_arc and h[6] >= cov_leaf and h[7] <= max_inner and h[8] <= max_ring]
    kept = []
    for h in hits:
        x0, y0, x1, y1 = h[1:5]
        good = True
        for k in kept:
            ix = max(0, min(x1, k[3]) - max(x0, k[1])); iy = max(0, min(y1, k[4]) - max(y0, k[2]))
            if ix * iy > 0.3 * min((x1 - x0) * (y1 - y0), (k[3] - k[1]) * (k[4] - k[2])):
                good = False; break
        if good:
            kept.append(h)
    return [(int(h[1]), int(h[2]), int(h[3]), int(h[4]), round(h[0], 3)) for h in kept]


STRICT = (0.9, 0.85, 0.06, 0.2)
LOOSE = (0.8, 0.75, 0.1, 0.3)
