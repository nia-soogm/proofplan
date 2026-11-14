"""PDF drawing set -> per-project 'sheet graph' JSON (+ page PNGs).

Everything here comes from vector PDF text coordinates (PyMuPDF), never from a
vision model, so every highlight is a real parsed bbox in PDF points.

Usage: python -m bp.ingest path/to/drawings.pdf --name myproject   (writes data/projects/myproject/)
"""
from __future__ import annotations
import argparse, json, re, collections
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "data" / "projects"
RENDER_ZOOM = 1.0  # 1 px == 1 pt, so bboxes map 1:1 onto the PNG

SHEET_ID_RE = re.compile(r"^(?:[A-Z]{1,2}-?\d{1,3}(?:\.\d{1,2})?[A-Z]?|\d{1,3})$")
TITLE_KW = re.compile(r"PLAN|SCHEDULE|ELEVATION|SECTION|DETAIL|NOTES|SUMMARY|SHEET|INDEX|RENDERING|FRAMING|TYPES", re.I)
MARK_HDR_RE = re.compile(r"^(MARK|NUMBER|NO\.?|DOOR\s*#|DR\s*NO|DOOR\s*NO\.?)$", re.I)
MARK_VAL_RE = re.compile(r"^[A-Z]{0,3}-?\d{1,4}[A-Z]?$")
CALLOUT_RE = re.compile(r"^(\d{1,2}[A-Z]?)/([A-Z]{1,2}-?\d{1,3}(?:\.\d{1,2})?)$")
LEVELS = [
    (re.compile(r"\b(BASEMENT|CELLAR)\b", re.I), "B"),
    (re.compile(r"\b(FIRST|1ST|MAIN|GROUND)\s+FLOOR\b|\bLEVEL\s*1\b|\b100 LEVEL\b", re.I), "1"),
    (re.compile(r"\b(SECOND|2ND)\s+FLOOR\b|\bLEVEL\s*2\b|\b200 LEVEL\b", re.I), "2"),
    (re.compile(r"\b(THIRD|3RD)\s+FLOOR\b|\bLEVEL\s*3\b|\b300 LEVEL\b", re.I), "3"),
]


def rbox(b):
    return [round(float(v), 1) for v in b]


def level_of(text: str):
    for rx, lv in LEVELS:
        if rx.search(text or ""):
            return lv
    return None


def page_lines(page):
    """Lines from get_text('dict') with font info; works for rotated text too."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            out.append({
                "text": " ".join(s["text"].strip() for s in spans),
                "bbox": rbox(l["bbox"]),
                "size": round(max(s["size"] for s in spans), 1),
                "font": spans[0]["font"],
                "dir": [round(l["dir"][0]), round(l["dir"][1])],
            })
    return out


def page_words(page):
    """Words with bbox + font signature, de-duplicated (CAD exports often double-draw text)."""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    spans.append((pymupdf.Rect(s["bbox"]), s["font"], round(s["size"], 1)))
    seen, out = set(), []
    for x0, y0, x1, y1, t, *_ in page.get_text("words"):
        key = (t, round(x0), round(y0))
        if key in seen:
            continue
        seen.add(key)
        c = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        font, size = "", 0.0
        for r, f, sz in spans:
            if c in r:
                font, size = f, sz
                break
        out.append({"text": t, "bbox": rbox((x0, y0, x1, y1)), "font": font, "size": size})
    return out


def detect_sheet_meta(lines, W, H):
    tb = [l for l in lines if l["bbox"][0] > 0.85 * W or l["bbox"][1] > 0.88 * H]
    ids = [l for l in tb if SHEET_ID_RE.match(l["text"].strip()) and l["size"] >= 14]
    sheet_id = max(ids, key=lambda l: l["size"])["text"].strip() if ids else None
    titles = [l for l in tb if l["bbox"][0] > 0.85 * W and TITLE_KW.search(l["text"]) and len(l["text"]) < 60]
    title = None
    if titles:
        t = max(titles, key=lambda l: l["size"])
        sib = [l for l in tb if l["dir"] == t["dir"] and abs(l["size"] - t["size"]) < 0.5 and l is not t
               and abs(l["bbox"][0] - t["bbox"][0]) < 3 * t["size"] and abs(l["bbox"][1] - t["bbox"][1]) < 3 * t["size"]]
        parts = sorted([t] + sib, key=lambda l: (l["bbox"][0], l["bbox"][1]) if t["dir"] == [0, -1] else (l["bbox"][1], l["bbox"][0]))
        title = " ".join(l["text"].strip() for l in parts)
    return sheet_id, title, (ids and max(ids, key=lambda l: l["size"])["bbox"])


def group_rows(words, tol=3.0):
    rows = []
    for w in sorted(words, key=lambda w: ((w["bbox"][1] + w["bbox"][3]) / 2, w["bbox"][0])):
        yc = (w["bbox"][1] + w["bbox"][3]) / 2
        if rows and abs(rows[-1]["yc"] - yc) <= tol:
            rows[-1]["words"].append(w)
        else:
            rows.append({"yc": yc, "words": [w]})
    for r in rows:
        r["words"].sort(key=lambda w: w["bbox"][0])
    return rows


def phrases(ws, gap=6.0, bounds=()):
    out = []
    for w in ws:
        crosses = out and any(out[-1]["bbox"][2] - 1 <= b <= w["bbox"][0] + 1 for b in bounds)
        if out and not crosses and w["bbox"][0] - out[-1]["bbox"][2] <= gap:
            p = out[-1]
            p["text"] += " " + w["text"]
            p["bbox"] = [min(p["bbox"][0], w["bbox"][0]), min(p["bbox"][1], w["bbox"][1]),
                         max(p["bbox"][2], w["bbox"][2]), max(p["bbox"][3], w["bbox"][3])]
        else:
            out.append({"text": w["text"], "bbox": list(w["bbox"])})
    return out


def vertical_rules(page, y, x0, x1):
    """x positions of vertical ruling lines crossing height y (CAD tables are drawn with lines)."""
    xs = set()
    clip = pymupdf.Rect(x0 - 5, y - 40, x1 + 5, y + 40)
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None or r.x1 < clip.x0 or r.x0 > clip.x1 or r.y1 < clip.y0 or r.y0 > clip.y1:
            continue
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.x - b.x) < 0.6 and min(a.y, b.y) - 1 <= y <= max(a.y, b.y) + 1 and x0 - 5 <= a.x <= x1 + 5:
                    xs.add(round(a.x, 1))
            elif it[0] == "re":
                rr = it[1]
                if rr.y0 - 1 <= y <= rr.y1 + 1:
                    for x in (rr.x0, rr.x1):
                        if x0 - 5 <= x <= x1 + 5:
                            xs.add(round(x, 1))
    out = []
    for x in sorted(xs):
        if not out or x - out[-1] > 3:
            out.append(x)
    return out


def wc(w):
    return ((w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2)


def detect_schedules(page, words, lines, page_no):
    """Schedule parser: 'DOOR ... SCHEDULE' title -> header row with a MARK-like column ->
    column boundaries from the table's vertical ruling lines (fallback: header word gaps) -> rows."""
    scheds = []
    titles = [l for l in lines if l["dir"] == [1, 0] and re.search(r"\bDOOR\b", l["text"], re.I)
              and re.search(r"SCHEDULE", l["text"], re.I) and len(l["text"]) < 40]
    for t in titles:
        tx0, ty0, tx1, ty1 = t["bbox"]
        cands = [w for w in words if MARK_HDR_RE.match(w["text"]) and ty1 - 2 <= w["bbox"][1] <= ty1 + 130
                 and tx0 - 600 <= w["bbox"][0] <= tx1 + 200]
        if not cands:
            continue
        mk = min(cands, key=lambda w: (w["bbox"][1], abs(w["bbox"][0] - tx0)))
        myc = wc(mk)[1]
        hdr = sorted([w for w in words if abs(wc(w)[1] - myc) < 3 and w["bbox"][0] >= mk["bbox"][0] - 2],
                     key=lambda w: w["bbox"][0])
        keep, last = [], None
        for w in hdr:
            if last is not None and w["bbox"][0] - last > 200:
                break
            keep.append(w)
            last = w["bbox"][2]
        x_lo, x_hi = mk["bbox"][0] - 25, keep[-1]["bbox"][2] + 60
        rules = vertical_rules(page, myc, x_lo - 20, x_hi + 600)
        rules = [x for x in rules if x <= x_hi + 600]
        if len(rules) >= 3:
            bounds = [x for x in rules if x >= mk["bbox"][0] - 30]
            if bounds and bounds[0] > mk["bbox"][0]:
                bounds = [mk["bbox"][0] - 2] + bounds
            right = keep[-1]["bbox"][2]
            after = [x for x in bounds if x >= right - 1]
            bounds = [x for x in bounds if x < right - 1] + ([after[0]] if after else [right + 200])
        else:  # fallback: midpoints between header word groups
            ph = phrases(keep, gap=4)
            cs = [(p["bbox"][0] + p["bbox"][2]) / 2 for p in ph]
            bounds = [x_lo] + [(a + b) / 2 for a, b in zip(cs, cs[1:])] + [x_hi]
        x_lo, x_hi = bounds[0], bounds[-1]
        ncol = len(bounds) - 1
        colidx = lambda x: next((i for i in range(ncol) if bounds[i] <= x < bounds[i + 1]), None)
        # header names: all words between title and bottom of mark header line, per column
        hw = [w for w in words if ty1 - 1 <= w["bbox"][1] and w["bbox"][3] <= mk["bbox"][3] + 1 and x_lo <= wc(w)[0] <= x_hi]
        names = [[] for _ in range(ncol)]
        for w in sorted(hw, key=lambda w: (round(w["bbox"][1]), w["bbox"][0])):
            i = colidx(wc(w)[0])
            if i is not None:
                names[i].append(w["text"])
        cols, seen = [], collections.Counter()
        for i, n in enumerate(names):
            nm = " ".join(n).strip() or f"col{i}"
            seen[nm] += 1
            cols.append(nm if seen[nm] == 1 else f"{nm} ({seen[nm]})")
        size0 = mk["size"] or 0
        region = [w for w in words if x_lo <= wc(w)[0] <= x_hi and w["bbox"][1] > mk["bbox"][3]
                  and (not size0 or not w["size"] or 0.6 * size0 <= w["size"] <= 1.6 * size0)]
        marks_found, last_y = [], mk["bbox"][3]
        for w in sorted(region, key=lambda w: w["bbox"][1]):
            yc = wc(w)[1]
            if colidx(wc(w)[0]) != 0 or not MARK_VAL_RE.match(w["text"]):
                continue
            if yc - last_y > 260:
                break
            marks_found.append((yc, w))
            last_y = yc
        if not marks_found:
            continue
        ys = [y for y, _ in marks_found]
        pitch = sorted(b - a for a, b in zip(ys, ys[1:]))[max(0, len(ys) // 2 - 1)] if len(ys) > 2 else 20
        tol = max(3.0, min(14.0, 0.45 * pitch))
        buckets = {i: [] for i in range(len(ys))}
        for w in region:
            yc = wc(w)[1]
            i = min(range(len(ys)), key=lambda i: abs(ys[i] - yc))
            if abs(ys[i] - yc) <= tol and w is not marks_found[i][1]:
                buckets[i].append(w)
        rows = []
        for i, (yc, first) in enumerate(marks_found):
            per = collections.defaultdict(list)
            for vl in group_rows(buckets[i], tol=2.0):
                for ph in phrases(vl["words"], gap=4.5, bounds=bounds):
                    j = colidx(ph["bbox"][0] + 2)
                    if j is not None and j != 0:
                        per[j].append(ph)
            cells = {c: "" for c in cols}
            cell_boxes = {cols[0]: first["bbox"]}
            cells[cols[0]] = first["text"]
            for j, ws in per.items():
                cells[cols[j]] = " ".join(w["text"] for w in ws)
                cell_boxes[cols[j]] = [min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws),
                                       max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)]
            bx = list(cell_boxes.values())
            bbox = [min(b[0] for b in bx), min(b[1] for b in bx), max(b[2] for b in bx), max(b[3] for b in bx)]
            rows.append({"mark": first["text"], "cells": cells, "bbox": rbox(bbox),
                         "cell_bboxes": {k: rbox(v) for k, v in cell_boxes.items()}})
        allb = [mk["bbox"]] + [r["bbox"] for r in rows]
        scheds.append({"kind": "door", "page": page_no, "title": t["text"], "columns": cols,
                       "header_bbox": rbox(mk["bbox"]),
                       "bbox": rbox([min(tx0, x_lo), ty0, max(x_hi, max(b[2] for b in allb)), max(b[3] for b in allb)]),
                       "rows": rows})
    best = {}
    for sc in scheds:
        k = tuple(sc["header_bbox"])
        d = sc["header_bbox"][1] - sc["bbox"][1]
        if k not in best or d < best[k][0]:
            best[k] = (d, sc)
    return [v[1] for v in best.values()]


def inside(b, r, pad=2):
    return b[0] >= r[0] - pad and b[1] >= r[1] - pad and b[2] <= r[2] + pad and b[3] <= r[3] + pad


def ingest(pdf: Path, name: str, render=True):
    doc = pymupdf.open(pdf)
    out_dir = PROJECTS / name
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    sheets, schedules = [], []
    for i, page in enumerate(doc):
        W, H = page.rect.width, page.rect.height
        words, lines = page_words(page), page_lines(page)
        sid, title, sid_box = detect_sheet_meta(lines, W, H)
        callouts = [{"text": w["text"], "detail": m.group(1), "target_sheet": m.group(2), "bbox": w["bbox"]}
                    for w in words for m in [CALLOUT_RE.match(w["text"])] if m]
        sc = detect_schedules(page, words, lines, i)
        schedules += sc
        sheets.append({"page": i, "sheet_id": sid or f"p{i+1}", "title": title, "level": level_of(title),
                       "width": W, "height": H, "is_vector": len(words) > 20, "n_words": len(words),
                       "words": words, "lines": lines, "callouts": callouts,
                       "schedule_bboxes": [s["bbox"] for s in sc]})
        if render:
            png = out_dir / "pages" / f"{i}.png"
            if not png.exists():
                page.get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM)).save(png)
    # sheet id fallback: drop duplicated ids (keep first) so lookups stay unambiguous
    marks = {r["mark"] for s in schedules for r in s["rows"]}
    tags = detect_tags(sheets, marks)
    graph = {"project": name, "source_pdf": str(pdf.relative_to(ROOT)) if pdf.is_absolute() else str(pdf),
             "render_zoom": RENDER_ZOOM, "sheets": sheets, "schedules": schedules, "tags": tags}
    (out_dir / "graph.json").write_text(json.dumps(graph))
    return graph


def detect_tags(sheets, marks):
    """Door tags = words on drawing sheets equal to a schedule mark, filtered to the dominant
    (font, size) signature of such matches. Schedule-driven, so it generalises across naming schemes."""
    cands = []
    for s in sheets:
        tb_x = 0.85 * s["width"]
        for w in s["words"]:
            if w["text"] in marks and w["bbox"][0] < tb_x and not any(inside(w["bbox"], b) for b in s["schedule_bboxes"]):
                cands.append((s, w))
    sig = collections.Counter((w["font"], w["size"]) for _, w in cands)
    if not sig:
        return []
    distinct = collections.defaultdict(set)
    for _, w in cands:
        distinct[(w["font"], w["size"])].add(w["text"])
    best = max(distinct, key=lambda k: (len(distinct[k]), sig[k]))
    keep = {best}
    return [{"mark": w["text"], "page": s["page"], "sheet_id": s["sheet_id"], "level": s["level"],
             "bbox": w["bbox"], "font": w["font"], "size": w["size"]}
            for s, w in cands if (w["font"], w["size"]) in keep]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    g = ingest(Path(a.pdf).resolve(), a.name, render=not a.no_render)
    print(f"{a.name}: {len(g['sheets'])} sheets, {len(g['schedules'])} door schedules "
          f"({sum(len(s['rows']) for s in g['schedules'])} rows), {len(g['tags'])} tags on plans")
    for s in g["sheets"]:
        print(f"  p{s['page']+1:>2} {s['sheet_id']:<8} L={s['level'] or '-'} {s['title']}")
