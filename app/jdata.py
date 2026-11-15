"""Junction real-data integration for the UI: curated files, page rendering, licence credits, citation formatting."""
import collections, functools, json, re
from pathlib import Path
from bp.jsolve import solve
from bp.sources import JDATA, CACHE
from bp.sources.ifcsrc import model_of

SCALE = 2.0  # PNG px per PDF pt
SOURCES = {"swiss_dwellings": "Swiss Dwellings — vector PDF + DXF",
           "ifc_models": "IFC models — BIM + rendered plans",
           "opening_design": "OpeningDesign — door / finish schedules (rotated sheets)",
           "cubicasa5k": "CubiCasa5K — Finnish raster floor plans (OCR + CPU symbol detector)",
           "ifc_bench": "ifc_bench — free-text questions about IFC models"}
LICENCE = {"swiss_dwellings": "Swiss Dwellings v3.0.0, Standfest et al. (Archilyse AG) · CC BY 4.0",
           "ifc_models": "Licence per model (see data/ifc_models/LICENSE.md); CC BY-ND models are not rendered",
           "opening_design": "OpeningDesign projects · CC BY-SA 4.0",
           "cubicasa5k": "CubiCasa5K · CC BY-NC-SA 4.0 (non-commercial)",
           "ifc_bench": "Licence per model (see data/ifc_bench/LICENSE.md)"}
BLURB = {"swiss_dwellings": "Apartment / room areas from DXF polygons, cross-checked against PDF fills.",
         "ifc_models": "Element counts, areas, lengths, volumes from the IFC model, cross-checked on the rendered plan.",
         "opening_design": "Door, window and finish schedules read from rotated ruled tables; every cell is a box.",
         "cubicasa5k": "Room counts from OCR of Finnish/Swedish labels (MH, KH, WC, S…, rotated too) + the '3H+K+S' summary, gated on dev; storeys from the sheet layout; WC seats from a CPU symbol detector. Other symbols abstain.",
         "ifc_bench": "Template answers from the IFC model (IfcOpenShell); every entity is cited by GUID and boxed on the rendered plan when one exists."}
FEATURED = {"high_quality_architectural-9005", "high_quality_architectural-13313", "high_quality_architectural-524"}  # demo picks (test split)
BIG = 2000  # pt; huge sheets (36x24in) render at 1x


@functools.lru_cache(maxsize=None)
def _credits():
    out = {}
    for src in SOURCES:
        t = (JDATA / "data" / src / "LICENSE.md").read_text()
        m = re.search(r"^All files: (\S+)\. Credit: (.+)$", t, re.M)
        if m:
            out[src] = {"*": f"{m.group(2)} [{m.group(1)}]"}
        else:
            d = {}
            for l in t.splitlines():
                c = [x.strip() for x in l.split("|")]
                if len(c) >= 6 and c[1].endswith((".ifc", ".pdf", ".svg", ".png")):
                    d[c[1]] = f"{c[4]} [{c[3]}]"; d.setdefault(Path(c[1]).name, d[c[1]])
            out[src] = d
    return out


def credit(src, fname):
    c = _credits().get(src, {})
    return c.get("*") or c.get(fname) or c.get(model_of(fname) + ".ifc") or "see data/%s/LICENSE.md" % src


@functools.lru_cache(maxsize=1)
def catalog():
    """Per source: a handful of held-out (test split) drawings with real questions + gold answers.
    Cached on disk (data/jcache/app_catalog.json); rebuilt when a solver model or this file is newer than the cache."""
    cf = CACHE / "app_catalog.json"
    deps = [Path(__file__)] + list((Path(__file__).resolve().parent.parent / "bp" / "models").glob("*.pkl"))
    if cf.exists() and cf.stat().st_mtime > max(d.stat().st_mtime for d in deps):
        return json.loads(cf.read_text())
    out = _build_catalog()
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out))
    return out


def _build_catalog():
    from eval.jeval import split_of
    byfile = collections.defaultdict(list)
    for l in open(JDATA / "questions.jsonl"):
        q = json.loads(l)
        src = q["path"].split("/")[1]
        if src == "ifc_bench" and (JDATA / q["path"]).exists():
            byfile[q["path"]].append(q)
        elif src in SOURCES and q["path"].endswith((".pdf", ".png")) and (JDATA / q["path"]).exists() and (split_of(q["path"]) == "test" or src == "opening_design"):
            byfile[q["path"]].append(q)
    out = {}
    for src in SOURCES:
        if src == "ifc_bench":
            out[src] = _ifcb_catalog({p: qs for p, qs in byfile.items() if p.split("/")[1] == src})
            continue
        if src == "cubicasa5k":
            files = sorted([p for p in byfile if p.split("/")[1] == src])
            from bp.sources import cubicasa
            ranked = []
            for p in files:
                good = [q for q in byfile[p] if cubicasa.answer(q["question"], p)["status"] == "answered" and q["question"].isascii()]
                ranked.append((-len(good) - (100 if Path(p).stem in FEATURED else 0), p, good))
            ranked.sort()
            out[src] = [{"path": p, "name": Path(p).stem, "split": "test", "credit": credit(src, Path(p).name),
                         "samples": [{"q": q["question"], "gold": q["answer"], "type": q["type"]} for q in good[:5]] +
                                    [{"q": q["question"], "gold": q["answer"], "type": q["type"]} for q in byfile[p] if q not in good and q["question"].isascii()][:2]}
                        for _, p, good in ranked[:14]]
            continue
        files = sorted([p for p in byfile if p.split("/")[1] == src], key=lambda p: -len({q["type"] for q in byfile[p]}) * 1000 - len(byfile[p]))
        sel = []
        for p in files:
            if src == "ifc_models" and sum(1 for s in sel if model_of(s["path"]) == model_of(p)) >= 2:
                continue
            qs = byfile[p]
            # one of each flavour first, then fill up
            seen, picks = set(), []
            for q in qs:
                key = (q["type"], re.sub(r"[\d.]+|'[^']*'|„[^“]*“|«[^»]*»", "", q["question"])[:40])
                if key not in seen and re.match(r"[A-Z][a-z]+ ", q["question"]) and q["question"].replace("²", "").replace("³", "").isascii():
                    seen.add(key); picks.append({"q": q["question"], "gold": q["answer"], "type": q["type"]})
            picks = [x for x in picks if x["type"] != "count"][:4] + [x for x in picks if x["type"] == "count"][:5]
            if src == "opening_design":
                picks = [{"q": q["question"], "gold": q["answer"], "type": q["type"]} for q in qs if q["question"].isascii()][:9]
            sel.append({"path": p, "name": Path(p).stem, "split": split_of(p), "credit": credit(src, Path(p).name), "samples": picks})
            if len(sel) >= 14:
                break
        out[src] = sel
    return out


def _ifcb_catalog(byfile):
    """ifc_bench models that have a rendered plan in data/ifc_models (so GUID evidence can be boxed); test split first.
    Each entry carries `view` = the plan to show, and samples = questions the solver answers (plus one it abstains on)."""
    from eval.jeval import split_of
    from bp.sources import ifcbench
    try:
        idx = ifcbench._plan_index()
    except Exception:
        return []
    out = []
    for p in sorted(byfile, key=lambda p: (split_of(p) != "test", p)):
        try:
            m = ifcbench.M(Path(p).stem)
        except Exception:
            continue
        pdfs = collections.Counter(idx[g["guid"]][0] for g in m.F.get("els", []) if isinstance(g, dict) and g.get("guid") in idx)
        if not pdfs:
            continue
        view = pdfs.most_common(1)[0][0]
        good, bad = [], []
        for q in byfile[p]:
            r = ifcbench.answer(q["question"], p)
            (good if r["status"] == "answered" else bad).append({"q": q["question"], "gold": str(q["answer"])[:80], "type": q["type"]})
        out.append({"path": p, "name": Path(p).stem, "split": split_of(p), "credit": credit("ifc_bench", Path(p).name),
                    "view": {"src": "ifc_models", "rel": view}, "samples": good[:6] + bad[:1]})
    return out


def _safe(src, rel):
    base = (JDATA / "data" / src).resolve()
    f = (base / rel).resolve()
    if base not in f.parents or not f.exists() or f.suffix not in (".pdf", ".png"):
        return None
    return f


@functools.lru_cache(maxsize=None)
def _scale(src, rel):
    import pymupdf
    f = _safe(src, rel)
    if f.suffix == ".png":
        return 1.0
    r = pymupdf.open(str(f))[0].rect
    return 1.0 if max(r.width, r.height) > BIG else SCALE


def page_png(src, rel):
    import pymupdf, hashlib
    f = _safe(src, rel)
    if f is None:
        return None
    if f.suffix == ".png":
        return f
    out = CACHE / "png" / src / (f.stem if "/" not in rel else hashlib.md5(rel.encode()).hexdigest()[:10] + "_" + f.stem[:40]).replace(" ", "_")
    out = out.with_suffix(".png")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        s = _scale(src, rel)
        pymupdf.open(str(f))[0].get_pixmap(matrix=pymupdf.Matrix(s, s), alpha=False).save(str(out))
    return out


def _size(src, rel):
    import pymupdf
    f = _safe(src, rel)
    if f.suffix == ".png":
        from PIL import Image
        return Image.open(f).size
    r = pymupdf.open(str(f))[0].rect
    s = _scale(src, rel)
    return r.width * s, r.height * s


def ask(path, question, strict=False):
    src = path.split("/")[1]
    r = solve(question, path, strict=strict)
    from urllib.parse import quote
    pdfs = []  # (source folder, relative pdf path); ifc_bench evidence may point at a plan rendered in ifc_models (pdf_src)
    if path.endswith((".pdf", ".png")) and src in ("opening_design", "cubicasa5k"):
        pdfs.append((src, path.split("/", 2)[2]))
    elif path.endswith(".pdf"):
        pdfs.append((src, Path(path).name))
    for e in r.get("evidence", []):
        pf = e.get("pdf") or (e.get("file") if str(e.get("file", "")).endswith(".pdf") else None)
        key = (e.get("pdf_src") or src, pf)
        if e.get("bbox") and pf and key not in pdfs:
            pdfs.append(key)
    sheets = []
    for i, (sp, p) in enumerate(pdfs):
        w, h = _size(sp, p)
        sheets.append({"page": i, "sheet": Path(p).stem.split("_plan-")[-1][:44], "title": "", "level": True, "width": w, "height": h,
                       "url": f"/jpages/{sp}/{quote(p)}", "credit": credit(sp, p)})
    cites = []
    for e in r.get("evidence", []):
        pf = e.get("pdf") or (e.get("file") if str(e.get("file", "")).endswith(".pdf") else None)
        c = {"ref": len(cites) + 1, "text": e.get("text", ""), "kind": e.get("kind", ""), "verified": e.get("verified"),
             "guid": e.get("guid"), "page": None, "sheet": e.get("file", "")}
        if e.get("bbox") and pf is None and len(pdfs) == 1 and src in ("opening_design", "cubicasa5k"):
            pf = pdfs[0][1]
        key = (e.get("pdf_src") or src, pf)
        if e.get("bbox") and key in pdfs:
            SC = _scale(key[0], pf)
            c["page"] = pdfs.index(key); c["sheet"] = sheets[c["page"]]["sheet"]
            c["bbox"] = [v * SC for v in e["bbox"]]
            if e.get("poly"):
                c["poly"] = [[x * SC, y * SC] for x, y in e["poly"]]
        cites.append(c)
    n = len(cites)
    refs = " ".join(f"[{i}]" for i in range(1, min(n, 40) + 1)) + (f" … [{n}]" if n > 40 else "")
    nv = sum(1 for c in cites if c["verified"] is True); nb = sum(1 for c in cites if c["verified"] is False)
    status = r["status"] if r["status"] in ("answered", "abstain", "conflict") else "abstain"
    text = r["answer"] + (f"\nEvidence: {refs}" if n else "")
    if n:
        text += f"\nCross-check: {nv} green (second source agrees), {nb} red (disagrees / not found), {n - nv - nb} unchecked."
    return {"status": status, "answer": text, "value": r.get("value"), "citations": cites, "sheets": sheets, "engine": "deterministic solver (" + src + (", strict mode" if strict else "") + ")",
            "trace": {"solver": src, "check": r.get("check"), "path": path, "mode": r.get("mode")}, "credit": credit(src, Path(path).name)}
