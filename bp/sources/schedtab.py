"""Generic ruled-table (schedule) extractor for vector CAD/Revit sheets, any 90° text rotation.

Works in a 'reading frame' where the table's text is horizontal: words + ruling lines are rotated into it,
rows/columns come from the ruling lines, and every cell keeps its bbox in ORIGINAL page coordinates for evidence.
"""
import collections, functools, re
import pymupdf

ROT = {  # text direction -> map original (x, y) to reading frame (X, Y)
    (1, 0): lambda x, y, W, H: (x, y),
    (0, -1): lambda x, y, W, H: (H - y, x),
    (0, 1): lambda x, y, W, H: (y, W - x),
}


def _rect_to(f, b, W, H):
    pts = [f(b[0], b[1], W, H), f(b[2], b[3], W, H), f(b[0], b[3], W, H), f(b[2], b[1], W, H)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


@functools.lru_cache(maxsize=8)
def page_frames(pdf_path, pno=0):
    """-> {dir: {"words": [(X0,Y0,X1,Y1,text,orig_bbox)], "h": [(y, x0, x1)], "v": [(x, y0, y1)]}}"""
    doc = pymupdf.open(pdf_path); page = doc[pno]
    W, H = page.rect.width, page.rect.height
    d = page.get_text("dict")
    linedirs = []
    for b in d["blocks"]:
        for l in b.get("lines", []):
            dr = (round(l["dir"][0]), round(l["dir"][1]))
            linedirs.append((pymupdf.Rect(l["bbox"]), dr))
    words = page.get_text("words")
    out = {k: {"words": [], "h": [], "v": []} for k in ROT}
    for w in words:
        c = pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
        dr = next((dr for r, dr in linedirs if r.contains(c)), (1, 0))
        if dr not in ROT:
            continue
        f = ROT[dr]
        X0, Y0, X1, Y1 = _rect_to(f, w[:4], W, H)
        out[dr]["words"].append((X0, Y0, X1, Y1, w[4], tuple(round(v, 1) for v in w[:4])))
    segs = []
    for dw in page.get_drawings():
        for it in dw["items"]:
            if it[0] == "l":
                segs.append((it[1].x, it[1].y, it[2].x, it[2].y))
            elif it[0] == "re":
                r = it[1]
                segs += [(r.x0, r.y0, r.x1, r.y0), (r.x0, r.y1, r.x1, r.y1), (r.x0, r.y0, r.x0, r.y1), (r.x1, r.y0, r.x1, r.y1)]
    for dr, f in ROT.items():
        for x0, y0, x1, y1 in segs:
            a = f(x0, y0, W, H); b = f(x1, y1, W, H)
            if abs(a[1] - b[1]) < 0.5 and abs(a[0] - b[0]) > 2:
                out[dr]["h"].append((a[1], min(a[0], b[0]), max(a[0], b[0])))
            elif abs(a[0] - b[0]) < 0.5 and abs(a[1] - b[1]) > 2:
                out[dr]["v"].append((a[0], min(a[1], b[1]), max(a[1], b[1])))
    for dr in ROT:
        out[dr]["h"] = _merge(out[dr]["h"]); out[dr]["v"] = _merge(out[dr]["v"])
    out["_size"] = (W, H)
    out["_inv"] = None
    return out


def _merge(segs, tol=0.6, gap=2.5):
    """Join collinear touching segments (CAD tables draw each cell edge separately)."""
    by = collections.defaultdict(list)
    for c, a, b in segs:
        by[round(c / tol)].append((c, a, b))
    out = []
    for k, ss in by.items():
        ss.sort(key=lambda s: s[1])
        cur = list(ss[0])
        for c, a, b in ss[1:]:
            if a <= cur[2] + gap:
                cur[2] = max(cur[2], b)
            else:
                out.append(tuple(cur)); cur = [c, a, b]
        out.append(tuple(cur))
    return out


def _cluster(vals, tol=1.5):
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(c) / len(c) for c in out]


def find_titles(pdf_path, rx=r"SCHEDULE"):
    fr = page_frames(pdf_path)
    res = []
    for dr in ROT:
        ws = fr[dr]["words"]
        rows = collections.defaultdict(list)
        for w in ws:
            rows[round((w[1] + w[3]) / 2)].append(w)
        for y, rw in rows.items():
            rw.sort(key=lambda w: w[0])
            # phrases on the same baseline
            ph = []
            for w in rw:
                if ph and w[0] - ph[-1][-1][2] < 8:
                    ph[-1].append(w)
                else:
                    ph.append([w])
            for p in ph:
                t = " ".join(w[4] for w in p)
                if re.search(rx, t, re.I) and len(t) < 60:
                    res.append({"dir": dr, "text": t, "bbox": [p[0][0], min(w[1] for w in p), p[-1][2], max(w[3] for w in p)]})
    return res


def extract(pdf_path, title):
    """Grid under a title -> {"title","dir","cols":[...], "rows":[{"cells":[text], "boxes":[orig bbox]}], "grid_rows":[...]}"""
    fr = page_frames(pdf_path); W, H = fr["_size"]
    F = fr[title["dir"]]; tb = title["bbox"]
    tx = (tb[0] + tb[2]) / 2
    # the table frame: long horizontal rules right around/below the title that span the title centre
    hs = [h for h in F["h"] if h[1] - 2 <= tx <= h[2] + 2 and tb[1] - 30 <= h[0] <= tb[3] + 40 and h[2] - h[1] > 2 * (tb[2] - tb[0])]
    if not hs:
        return None
    top = min(hs, key=lambda h: abs(h[0] - tb[1]))
    X0, X1 = top[1], top[2]
    full = lambda h: h[1] <= X0 + 3 and h[2] >= X1 - 3
    ys = _cluster([h[0] for h in F["h"] if full(h) and h[0] >= top[0] - 1])
    # keep the contiguous run of row rules (stop at a big gap)
    run = [ys[0]]
    for y in ys[1:]:
        if y - run[-1] > 80:
            break
        run.append(y)
    Y0, Y1 = run[0], run[-1]
    # column rules: vertical segments inside the frame; columns = x where rules cover the lower body
    vx = _cluster([v[0] for v in F["v"] if X0 - 2 <= v[0] <= X1 + 2 and v[2] > Y0 and v[1] < Y1 and v[2] - v[1] > 3])
    words = [w for w in F["words"] if X0 - 1 <= (w[0] + w[2]) / 2 <= X1 + 1 and Y0 - 1 <= (w[1] + w[3]) / 2 <= Y1 + 1]
    rows = []
    for ya, yb in zip(run, run[1:]):
        # per-row column bounds: vertical rules that cross this row band
        xs = _cluster([v[0] for v in F["v"] if X0 - 2 <= v[0] <= X1 + 2 and v[1] <= ya + 1.5 and v[2] >= yb - 1.5])
        xs = sorted(set([X0] + [x for x in xs if X0 + 1 < x < X1 - 1] + [X1]))
        cells = []
        for xa, xb in zip(xs, xs[1:]):
            ws = sorted([w for w in words if xa <= (w[0] + w[2]) / 2 < xb and ya <= (w[1] + w[3]) / 2 < yb], key=lambda w: (round((w[1] + w[3]) / 4), w[0]))
            ob = [w[5] for w in ws]
            box = [min(b[0] for b in ob), min(b[1] for b in ob), max(b[2] for b in ob), max(b[3] for b in ob)] if ob else None
            cells.append({"text": " ".join(w[4] for w in ws), "x": (xa, xb), "box": box, "words": [((w[0] + w[2]) / 2, w[4], (w[1] + w[3]) / 2) for w in ws]})
        if any(c["text"] for c in cells):
            rows.append({"y": (ya, yb), "cells": cells})
    return {"title": title["text"], "dir": title["dir"], "frame": [X0, Y0, X1, Y1], "rows": rows}


def to_records(tab, min_cols=3):
    """Header = rows above the first row with the modal column count; header texts stacked per column."""
    rows = tab["rows"]
    cnt = collections.Counter(len(r["cells"]) for r in rows)
    ncol = max((n for n, c in cnt.items() if c >= 2), key=lambda n: (cnt[n], n), default=None)
    if not ncol or ncol < min_cols:
        return None
    body_i = [i for i, r in enumerate(rows) if len(r["cells"]) == ncol]
    first = body_i[0]
    colx = [c["x"] for c in rows[first]["cells"]]
    names = [[] for _ in colx]
    for r in rows[:first]:
        for c in r["cells"]:
            if not c["text"] or c["text"].strip() == tab["title"].strip():
                continue
            spans = [j for j, (a, b) in enumerate(colx) if c["x"][0] - 1 <= (a + b) / 2 <= c["x"][1] + 1]
            if len(spans) == 1:
                names[spans[0]].append(c["text"])
            else:  # a spanning cell: group header for all, or sub-headers placed word by word
                per = collections.defaultdict(list)
                lines = _cluster([w[2] for w in c["words"]], tol=3)
                top = [w for w in c["words"] if abs(w[2] - lines[0]) <= 3] if len(lines) > 1 else []
                for xc, t, yc in c["words"]:
                    if top and abs(yc - lines[0]) <= 3:
                        continue
                    j = next((j for j, (a, b) in enumerate(colx) if a <= xc < b), None)
                    if j is not None:
                        per[j].append(t)
                if top:
                    for j in spans:
                        names[j].append(" ".join(w[1] for w in sorted(top)))
                if len(per) > 1 or top:
                    for j, ts in per.items():
                        names[j].append(" ".join(ts))
                else:
                    for j in spans:
                        names[j].append(c["text"])
    # the first body rows may still be column headers (e.g. 'Width' 'Height' under a group header)
    recs, hdr_extra = [], 0
    for i in body_i:
        r = rows[i]
        texts = [c["text"] for c in r["cells"]]
        if not recs and not any(re.search(r"\d", t) for t in texts):
            for j, t in enumerate(texts):
                if t:
                    names[j].append(t)
            continue
        recs.append({"cells": texts, "boxes": [c["box"] for c in r["cells"]],
                     "bbox": _union([c["box"] for c in r["cells"] if c["box"]])})
    cols = [" ".join(n).strip() or f"col{j}" for j, n in enumerate(names)]
    return {"title": tab["title"], "cols": cols, "records": recs}


def _union(bs):
    bs = [b for b in bs if b]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def schedules(pdf_path):
    out = []
    seen = set()
    for t in find_titles(pdf_path):
        tab = extract(pdf_path, t)
        if not tab:
            continue
        key = tuple(round(v) for v in tab["frame"])
        if key in seen:
            continue
        seen.add(key)
        rec = to_records(tab)
        if rec and rec["records"]:
            rec["frame"] = tab["frame"]; rec["dir"] = t["dir"]
            out.append(rec)
    return out
