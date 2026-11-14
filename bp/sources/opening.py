"""opening_design: answers from the ruled schedules (door / finish / window / type schedules) on a sheet.

Generic, not per-question: the question is matched against the VALUES actually present in the schedule
(codes like A1, WD-1, U01, SS, 118, 3'-0") and against the COLUMN names (width, height, area, floor finish ...).
  - filters: every question token that equals a cell value of some column; same column -> OR, different -> AND
  - count: number of matching rows, or the sum of a 'Count' column when the schedule has one (type schedules)
  - area: sum of the 'Area' column over the matching rows
  - lookup (text): row by its id (door number / room number / type mark) + the column named in the question
Evidence = the schedule cells used, as boxes on the sheet (rotated sheets handled).
"""
import functools, re
from pathlib import Path
import pymupdf
from . import JDATA
from .schedtab import schedules

FEET = re.compile(r"(\d+)\s*'\s*-?\s*(\d+(?:\s+\d+/\d+)?)\s*\"")


def nv(s):
    """normalise a cell value / question token."""
    s = (s or "").upper().replace("’", "'").replace("”", '"').replace("″", '"')
    m = FEET.fullmatch(s.strip())
    if m:
        return f"{int(m.group(1))}'-{m.group(2).strip()}\""
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"(?<=[A-Z])\s*-\s*(?=\d)", "-", s)  # 'STL - 1' -> 'STL-1' (wrapped cells)
    return s.replace(" SF", "") if re.fullmatch(r"[\d,]+ SF", s) else s


@functools.lru_cache(maxsize=16)
def sheet(path):
    pdf = str(JDATA / path)
    sc = schedules(pdf)
    page = pymupdf.open(pdf)[0]
    M = page.rotation_matrix
    for s in sc:
        s["kind"] = ("finish" if re.search(r"FINISH|ROOM", s["title"], re.I) else "window" if re.search(r"WINDOW", s["title"], re.I)
                     else "keynote" if re.search(r"KEYNOTE", s["title"], re.I) else "door")
        for r in s["records"]:
            r["n"] = [nv(c) for c in r["cells"]]
            r["boxes"] = [list(pymupdf.Rect(b) * M) if b else None for b in r["boxes"]]
            r["boxes"] = [[min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])] if b else None for b in r["boxes"]]
    return sc


def colidx(s, *pats, avoid=None):
    for p in pats:
        for j, c in enumerate(s["cols"]):
            if re.search(p, c, re.I) and not (avoid and re.search(avoid, c, re.I)):
                return j
    return None


def num(t):
    m = re.search(r"-?[\d,]*\.?\d+", (t or "").replace(",", ""))
    return float(m.group()) if m else None


def pick_schedule(sc, q):
    ql = q.lower()
    want = ("finish" if re.search(r"finish|room schedule|area|rooms? (?:have|named|in the)|locker|wall finish|ceiling", ql) and "door" not in ql.split("schedule")[0][-30:]
            else "window" if "window" in ql else "door")
    cands = [s for s in sc if s["kind"] == want]
    if want == "door" and len(cands) > 1:  # prefer the type schedule ('scheduled' / type mark questions) when it has a Count column
        typ = [s for s in cands if colidx(s, r"^count$") is not None]
        num_ = [s for s in cands if colidx(s, r"^count$") is None]
        cands = (typ if re.search(r"\bscheduled\b|type schedule|in total", ql) else num_ or typ) or cands
    return cands[0] if cands else None


def q_tokens(q):
    q2 = q.replace("’", "'").replace("”", '"')
    toks = set()
    for m in FEET.finditer(q2):
        toks.add(f"{int(m.group(1))}'-{m.group(2).strip()}\"")
    for t in re.findall(r"[A-Za-z]*-?\d[\w./-]*|\b[A-Z]{2,3}\b(?=\)|,| and| or|\s*\()", q2):
        t = t.strip(".,").upper(); toks.add(t)
        if re.fullmatch(r"\d+\.\w+-\d+\.\w+", t):  # id range '118.A1-118.B2' -> both ends
            toks.update(t.split("-"))
    for t in re.findall(r"\b[A-Z][A-Z .&'-]{2,}[A-Z]\b", q2):  # upper-case names, e.g. OFFICE, RESTROOM
        toks.add(t.strip())
    return toks


def filters(s, q, skip_cols=()):
    """{col: set(values)} for question tokens that occur as whole cell values (or as one item of 'W1 & W2')."""
    toks = q_tokens(q)
    feet_parts = {p for m in FEET.finditer(q.replace("’", "'").replace("”", '"')) for p in re.findall(r"\d+", m.group())}
    toks = {t for t in toks if not (t.isdigit() and t in feet_parts)}
    skip_cols = set(skip_cols) | {j for j, c in enumerate(s["cols"]) if re.fullmatch(r"count", c, re.I)}
    note = re.search(r"\bnote ([A-Z])\b", q)
    out = {}
    for j, c in enumerate(s["cols"]):
        if j in skip_cols:
            continue
        vals = {v for r in s["records"] for v in [r["n"][j]] if v}
        parts = {p for v in vals for p in re.split(r"\s*(?:&|,|/)\s*", v)}
        hit = {t for t in toks if t in vals or (t in parts and len(t) >= 2)}
        if note and re.search(r"note", c, re.I):
            hit.add(note.group(1))
        if hit:
            out[j] = hit
    # the same code found in several columns (AL-1 is a door AND a frame material): keep the column the question names
    ql = q.lower()
    for j in list(out):
        for k in list(out):
            if j < k and j in out and k in out and out[j] & out[k]:
                named = [x for x in (j, k) if any(w in ql for w in re.findall(r"[a-z]{4,}", s["cols"][x].lower()) if w not in ("door", "type"))]
                drop = [x for x in (j, k) if x not in named] if len(named) == 1 else []
                for x in drop:
                    out.pop(x, None)
    return out


STOP = set("how many are is the a an of in on to and or with for at be by scheduled schedule listed total door doors window windows items types type sheet drawings what which whose there does do".split())


def text_filter(s, q):
    """Rows whose free-text columns contain the question's descriptive words (e.g. '<brand> sliding patio',
    'description starts with SWING', 'on the 3rd floor'). Only words that occur in the schedule at all are used."""
    qq = re.sub(r"^.*?drawings(?: \([^)]*\))?,?", "", q)
    txtcols = [j for j, c in enumerate(s["cols"]) if sum(1 for r in s["records"] if len(r["cells"][j]) > 12) >= 2]
    if not txtcols:
        return None
    vocab = {w for r in s["records"] for j in txtcols for w in re.findall(r"[A-Z0-9]+", r["n"][j])}
    starts = re.search(r"starts? with '([^']+)'", qq)
    if starts:
        pre = starts.group(1).upper()
        return (pre, lambda r: any(r["n"][j].startswith(pre) for j in txtcols))
    words = [w for w in re.findall(r"[A-Za-z0-9]+", qq) if w.lower() not in STOP and w.upper() in vocab]
    words = [w.upper() for w in words]
    if not words:
        return None
    return (" ".join(words), lambda r: all(any(re.search(rf"\b{re.escape(w)}\b", r["n"][j]) for j in txtcols) for w in words))


def row_match(r, flt):
    for j, vals in flt.items():
        v = r["n"][j]
        items = set(re.split(r"\s*(?:&|,)\s*", v)) | {v}
        if not (vals & items):
            return False
    return True


def ev(s, r, js, tag=""):
    out = []
    for j in js:
        b = r["boxes"][j] if j is not None and j < len(r["boxes"]) else None
        if b:
            out.append({"bbox": [round(v, 1) for v in b], "page": 0, "text": f"{s['title']} · {r['cells'][0]} · {s['cols'][j]} = {r['cells'][j]}{tag}",
                        "kind": "schedule cell", "verified": None})
    return out


def answer(question, path):
    try:
        sc = sheet(path)
    except Exception as e:
        return _abst(f"could not read schedules ({type(e).__name__})")
    if not sc:
        return _abst("no ruled schedule found on this sheet (plan-view counts need symbol detection)")
    q = question
    ql = q.lower()
    s = pick_schedule(sc, q)
    if s is None:
        return _abst("no matching schedule on this sheet")
    id_col = 0 if s["kind"] != "door" or colidx(s, r"^mark$") is None else colidx(s, r"^mark$")
    if s["kind"] == "finish":
        id_col = colidx(s, r"room number") or 0
    if colidx(s, r"type mark") is not None and colidx(s, r"^count$") is not None:
        id_col = colidx(s, r"type mark")
    cnt_col = colidx(s, r"^count$")
    area_col = colidx(s, r"^area$")
    flt = filters(s, q)
    # 'room 118 (Fitness Center)' in a door schedule -> the room-number column only
    if s["kind"] == "door":
        rn = colidx(s, r"room number")
        if rn is not None and re.search(r"for room (\d+)", q):
            flt = {rn: {re.search(r"for room (\d+)", q).group(1)}}
    nm_col = colidx(s, r"room name")
    if not flt and nm_col is not None and not re.search(r"\ball rooms\b|listed|how many rooms are", ql):
        qw = {re.sub(r"'s$|s$", "", w) for w in re.findall(r"[a-z']+", ql)}
        byname = [r for r in s["records"] if r["n"][nm_col] and all(re.sub(r"'s$|s$", "", w) in qw for w in re.findall(r"[a-z']+", r["n"][nm_col].lower()))]
        if byname:
            flt = {nm_col: {r["n"][nm_col] for r in byname}}
    if not flt and not re.search(r"in total|\ball\b|listed", ql):
        flt_txt = text_filter(s, q)
    else:
        flt_txt = None
    rows = [r for r in s["records"] if row_match(r, flt)] if flt else list(s["records"])
    if flt_txt:
        rows = [r for r in rows if flt_txt[1](r)]
    if re.search(r"\bif every\b|\bassum", ql):
        return _abst("hypothetical / arithmetic question beyond the schedule rows")
    meas = "area" if re.search(r"\barea\b", ql) and "how many" not in ql else "count" if "how many" in ql else "lookup"
    if "how many" in ql and re.search(r"total area", ql) and area_col is not None:
        rr = [r for r in rows if num(r["cells"][area_col]) is not None]
        v = sum(num(r["cells"][area_col]) for r in rr)
        return {"status": "answered", "answer": f"{len(rows)} rooms, {v:g} SF" + _fdesc(s, flt) + ".", "value": f"{len(rows)} rooms, {v:g} SF",
                "evidence": [x for r in rows for x in ev(s, r, [id_col, area_col] + list(flt))]}
    if meas == "count":
        if cnt_col is not None:
            vals = [num(r["cells"][cnt_col]) or 0 for r in rows]
            v = int(sum(vals))
            e = [x for r in rows for x in ev(s, r, [id_col, cnt_col])]
            return _ok(v, f"{v} = sum of the Count column over {len(rows)} row(s) of '{s['title']}'" + _fdesc(s, flt) + ".", e)
        if re.search(r"all four walls", ql) and flt:
            wall = [j for j, c in enumerate(s["cols"]) if re.search(r"wall", c, re.I)]
            vals = set().union(*flt.values())
            rows = [r for r in s["records"] if all(r["n"][j] in vals for j in wall)]
            e = [x for r in rows for x in ev(s, r, [id_col] + wall)]
            return _ok(len(rows), f"{len(rows)} rooms with {'/'.join(sorted(vals))} on all four walls in '{s['title']}'.", e)
        e = [x for r in rows for x in ev(s, r, [id_col] + list(flt))]
        return _ok(len(rows), f"{len(rows)} row(s) in '{s['title']}'" + _fdesc(s, flt) + ".", e)
    if re.search(r"largest|biggest|smallest", ql):
        meas = "lookup"
    if meas == "area":
        if area_col is None:
            return _abst("schedule has no Area column")
        rows = [r for r in rows if num(r["cells"][area_col]) is not None]
        v = sum(num(r["cells"][area_col]) for r in rows)
        e = [x for r in rows for x in ev(s, r, [id_col, area_col])]
        return _ok(round(v, 2), f"{v:g} SF = sum of Area over {len(rows)} room(s)" + _fdesc(s, flt) + ".", e)
    # lookup: the row by id, the column named in the question
    if re.search(r"largest|biggest|smallest", ql) and area_col is not None:
        rs = [r for r in s["records"] if num(r["cells"][area_col]) is not None]
        r = (min if "smallest" in ql else max)(rs, key=lambda r: num(r["cells"][area_col]))
        nm = colidx(s, r"room name")
        return {"status": "answered", "answer": f"Room {r['cells'][id_col]} {r['cells'][nm] if nm is not None else ''} ({r['cells'][area_col]}) — {'smallest' if 'smallest' in ql else 'largest'} of {len(rs)} rooms with an area.",
                "value": f"Room {r['cells'][id_col]} {r['cells'][nm] if nm is not None else ''} ({r['cells'][area_col]})", "evidence": ev(s, r, [id_col, nm, area_col])}
    ids = [t for t in q_tokens(q) if any(r["n"][id_col] == t for r in s["records"])]
    if len(ids) == 1:
        rws = [next(r for r in s["records"] if r["n"][id_col] == ids[0])]
    elif flt:
        rws = [r for r in s["records"] if row_match(r, {j: v for j, v in flt.items() if j != id_col})] if any(j != id_col for j in flt) else []
        if ids:
            rws = [r for r in rws if r["n"][id_col] in ids] or rws
    else:
        rws = list(s["records"])  # no id: answer only if every row agrees (checked below)
    if not rws:
        return _abst("could not find the row the question is about")
    r = rws[0]
    want = [(r"width", r"width"), (r"height", r"height"), (r"hardware", r"hardware"), (r"door type|what door type", r"door type"),
            (r"(?<!frame )material", r"matl|material"), (r"floor finish", r"floor"), (r"ceiling", r"ceiling"), (r"north wall", r"north"),
            (r"south wall", r"south"), (r"east wall", r"east"), (r"west wall", r"west"), (r"fire rating", r"fire"), (r"frame material", r"frame mat")]
    js = []
    for qp, cp in want:
        if re.search(qp, ql):
            j = colidx(s, cp, avoid=r"frame" if cp in ("matl|material", "door type") and "frame" not in ql else None)
            if j is not None and j not in js:
                js.append(j)
    if not js:
        return _abst("could not tell which column is asked")
    if len(rws) > 1 and len(ids) != 1 and not flt and len({tuple(x["cells"][j] for j in js) for x in rws}) > 1:
        return _abst("the question names no row and the rows differ")
    if len({tuple(x["cells"][j] for j in js) for x in rws}) > 1:
        return {"status": "conflict", "answer": f"CONFLICT: the {len(rws)} matching rows differ in {', '.join(s['cols'][j] for j in js)}.", "value": None,
                "evidence": [e for x in rws for e in ev(s, x, [id_col] + js)]}
    vals = [r["cells"][j] for j in js]
    leg = [legend(path, v) if re.search(r"mean|what (?:kind|floor finishes)|finish", ql) else None for v in vals]
    if [s["cols"][j].split()[-1].lower() for j in js] == ["width", "height"]:
        w_, h_ = (nv(v) for v in vals)
        return {"status": "answered", "answer": f"{r['cells'][id_col]} — {w_} wide x {h_} high (from '{s['title']}').", "value": f"{w_} wide x {h_} high",
                "evidence": [e for x in rws for e in ev(s, x, [id_col] + js)]}
    parts = [f"{s['cols'][j]}: {v}" + (f" ({l})" if l else "") for j, v, l in zip(js, vals, leg)]
    who = ids[0] if len(rws) == 1 else f"{len(rws)} rows ({', '.join(x['cells'][id_col] for x in rws)})"
    return {"status": "answered", "answer": f"{who} — " + "; ".join(parts) + f" (from '{s['title']}').",
            "value": " / ".join(v + (f" - {l}" if l else "") for v, l in zip(vals, leg)),
            "evidence": [e for x in rws for e in ev(s, x, [id_col] + js)]}


@functools.lru_cache(maxsize=4096)
def legend(path, code):
    """'F4 - RESILIENT FLOORING (T.B.D.)' style legend lines on the same sheet."""
    if not code or len(code) > 8:
        return None
    for p in re.split(r"\s*(?:&)\s*", code):
        pass
    txt = _lines(path)
    hits = []
    for c in re.split(r"\s*&\s*", code):
        m = next((re.match(rf"^{re.escape(c)}\s*[-–=:]\s*(.+)$", l) for l in txt if re.match(rf"^{re.escape(c)}\s*[-–=:]\s*\S", l)), None)
        if m:
            hits.append(m.group(1).strip()); continue
        # legend tables: the code on its own line, its description on the next text line
        for i, l in enumerate(txt[:-1]):
            nx = txt[i + 1]
            if l == c and len(re.findall(r"[A-Za-z]{3,}", nx)) >= 2 and not re.fullmatch(r"[A-Z]{1,3}\d?(\s*&\s*[A-Z]{1,3}\d?)*", nx):
                hits.append(nx.strip()); break
    return " & ".join(hits) or None


@functools.lru_cache(maxsize=16)
def _lines(path):
    page = pymupdf.open(str(JDATA / path))[0]
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            out.append(" ".join(sp["text"] for sp in l["spans"]).strip())
    return out


def _fdesc(s, flt):
    return (" where " + " and ".join(f"{s['cols'][j]} ∈ {{{', '.join(sorted(v))}}}" for j, v in flt.items())) if flt else ""


def _ok(v, msg, ev_):
    return {"status": "answered", "answer": msg, "value": v, "evidence": ev_}


def _abst(why):
    return {"status": "abstain", "answer": f"NOT SURE: {why}.", "value": None, "evidence": []}
