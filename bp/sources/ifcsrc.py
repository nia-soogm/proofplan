"""ifc_models: IFC model (IfcOpenShell) + the plan SVG/PDF rendered from it.

Two independent sources per answer:
  * model  - IfcOpenShell entity counts / base quantities (evidence: GUIDs)
  * plan   - the rendered floor-plan SVG groups (class + data-guid + paths), mapped 1:1 onto the PDF
            (PDF pt = (svg_mm - viewBox_origin) * 72/25.4), so every GUID has a polygon on the PDF.
Cross-check: counts on plan vs model containment; space area from Qto vs polygon area on the 1:100 plan.
"""
from __future__ import annotations
import json, re, math, collections, functools, glob, os
from pathlib import Path
from . import JDATA, CACHE
from .qparse import quotes, measure, kind, scope

IFC_DIR = JDATA / "data" / "ifc_models"
MM2PT = 72 / 25.4
TYPES = {"door": ["IfcDoor"], "window": ["IfcWindow"], "space": ["IfcSpace"], "wall": ["IfcWall"], "column": ["IfcColumn"],
         "beam": ["IfcBeam"], "stair": ["IfcStair"], "slab": ["IfcSlab"], "sanitary": ["IfcSanitaryTerminal"],
         "light": ["IfcLightFixture"]}
SVG_CLASS = {"door": {"IfcDoor"}, "window": {"IfcWindow"}, "space": {"IfcSpace"}, "wall": {"IfcWall", "IfcWallStandardCase", "IfcWallElementedCase"},
             "column": {"IfcColumn"}, "beam": {"IfcBeam"}, "stair": {"IfcStair"}, "slab": {"IfcSlab"}}
FALLBACK_TYPES = {"sanitary": ["IfcFlowTerminal"], "light": ["IfcFlowTerminal"]}


def model_of(path: str) -> str:
    n = os.path.basename(path)
    n = re.sub(r"\.(ifc|pdf|svg|png)$", "", n)
    return re.sub(r"_(plan|elevation|section)-.*$", "", n)


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _qto(el, keys):
    import ifcopenshell.util.element as ue
    q = ue.get_psets(el, qtos_only=True)
    for pset in sorted(q, key=lambda k: (not k.startswith("Qto_"), k != "BaseQuantities")):
        for k in keys:
            v = _num(q[pset].get(k))
            if v is not None:
                return v, f"{pset}.{k}"
    return None, None


_GS = None


def _storey_via_containment(cs):
    """Storey reached from the spatial container (storey itself, or a space/zone aggregated under it)."""
    import ifcopenshell.util.element as ue
    for c in cs:
        for _ in range(5):
            if c is None:
                break
            if c.is_a("IfcBuildingStorey"):
                return c.Name
            if c.is_a("IfcBuilding") or c.is_a("IfcSite"):
                break
            c = ue.get_aggregate(c)
    return None


def _geom_area(e):
    """Plan footprint area of an IfcSpace's body (m², world coords) - used only when the file has no quantity."""
    global _GS
    try:
        import ifcopenshell.geom as g, ifcopenshell.util.shape as us
        if _GS is None:
            _GS = g.settings(); _GS.set("use-world-coords", True)
        geo = g.create_shape(_GS, e).geometry
        a = float(us.get_footprint_area(geo))
        if a <= 1e-6:  # open surface models: footprint fails -> area of downward faces
            import numpy as np
            v = np.array(geo.verts).reshape(-1, 3); F = np.array(geo.faces).reshape(-1, 3)
            n = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
            a = float(np.abs(n[:, 2]).sum() / 4)  # projected area of a closed shell = (top + bottom) / 2
        return round(a, 4) if a > 1e-6 else None
    except Exception:
        return None


@functools.lru_cache(maxsize=64)
def facts(model: str) -> dict:
    cf = CACHE / "ifc" / f"{model}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    import ifcopenshell, ifcopenshell.util.element as ue
    m = ifcopenshell.open(str(IFC_DIR / f"{model}.ifc"))
    proj = (m.by_type("IfcProject") or [None])[0]
    bld = (m.by_type("IfcBuilding") or [None])[0]
    storeys = [{"guid": s.GlobalId, "name": s.Name, "long": getattr(s, "LongName", None), "elev": _num(s.Elevation)} for s in m.by_type("IfcBuildingStorey")]

    def storey_of(el):
        c = ue.get_container(el)
        if c is None:
            a = ue.get_aggregate(el)
            while a is not None and not a.is_a("IfcBuildingStorey"):
                a = ue.get_aggregate(a) if not a.is_a("IfcBuilding") else None
            c = a
        if c is not None and not c.is_a("IfcBuildingStorey"):
            c = ue.get_aggregate(c) if c.is_a("IfcSpace") else None
        return c.Name if c is not None and c.is_a("IfcBuildingStorey") else None

    import ifcopenshell.util.unit as uu
    SC = {"l": uu.calculate_unit_scale(m, "LENGTHUNIT"), "a": uu.calculate_unit_scale(m, "AREAUNIT"), "v": uu.calculate_unit_scale(m, "VOLUMEUNIT")}
    els = {}
    for k, ts in {**TYPES, **{f"{k}_fallback": v for k, v in FALLBACK_TYPES.items()}}.items():
        seen, out = set(), []
        for t in ts:
            try:
                objs = m.by_type(t)
            except RuntimeError:
                objs = []
            for e in objs:
                if e.GlobalId in seen:
                    continue
                seen.add(e.GlobalId)
                cs = [r.RelatingStructure for r in (getattr(e, "ContainedInStructure", None) or [])]
                ag = [r.RelatingObject for r in (getattr(e, "Decomposes", None) or [])]
                if not cs and ag and ag[0].is_a("IfcSpatialStructureElement"):
                    cs = ag  # spaces are aggregated, not contained
                elif not cs and ag and ag[0].is_a() != e.is_a():
                    cs = [r.RelatingStructure for r in (getattr(ag[0], "ContainedInStructure", None) or [])]  # e.g. door inside a curtain wall
                d = {"guid": e.GlobalId, "type": e.is_a(), "name": e.Name, "storey": storey_of(e),
                     "contained": not (ag and ag[0].is_a() == e.is_a()),  # nested sub-parts of the same class (stair-in-stair) are not separate objects
                     "direct": _storey_via_containment(cs)}
                parts = [c.GlobalId for r in (getattr(e, "IsDecomposedBy", None) or []) for c in r.RelatedObjects]
                if parts:
                    d["parts"] = parts  # e.g. a stair drawn only through its IfcBuildingElementPart children
                if k == "space":
                    d["long"] = e.LongName
                    d["area"], d["area_src"] = _qto(e, ["NetFloorArea", "GrossFloorArea"])
                    if d["area"] is None:
                        d["area"], d["area_src"] = _geom_area(e), "footprint of IfcSpace body geometry (no Qto in file)"
                        if d["area"] is None:
                            d["area_src"] = None
                    elif SC["a"] != 1:
                        d["area"] *= SC["a"]
                elif k == "wall":
                    d["length"], d["length_src"] = _qto(e, ["Length"])
                    d["side"], d["side_src"] = _qto(e, ["NetSideArea", "GrossSideArea"])
                    d["vol"], d["vol_src"] = _qto(e, ["NetVolume", "GrossVolume"])
                    for a_, f_ in (("length", "l"), ("side", "a"), ("vol", "v")):
                        if d[a_] is not None:
                            d[a_] *= SC[f_]
                elif k == "slab":
                    d["vol"], d["vol_src"] = _qto(e, ["NetVolume", "GrossVolume"])
                    if d["vol"] is not None:
                        d["vol"] *= SC["v"]
                elif k.endswith("_fallback"):
                    t_ = ue.get_type(e)
                    d["predefined"] = str(getattr(t_, "PredefinedType", "") or getattr(e, "PredefinedType", "") or "")
                    d["typename"] = (t_.Name if t_ else "") or ""
                    d["typeclass"] = t_.is_a() if t_ else None
                out.append(d)
        els[k] = out
    f = {"model": model, "project": proj.Name if proj else None, "project_long": getattr(proj, "LongName", None) if proj else None,
         "building": bld.Name if bld else None, "schema": m.schema, "storeys": storeys, "elements": els, "plans": plans(model)}
    import ifcopenshell.guid as ig
    by_gid = {s_["guid"]: s_["name"] for s_ in storeys}
    for pl in f["plans"]:
        names = [by_gid.get(ig.compress(u.replace("-", ""))) for u in pl["storey_uuids"]]
        names = [n for n in names if n]
        if names:
            pl["storeys"] = sorted(set(names))
            if len(pl["storeys"]) == 1:
                pl["plan_title_storey"] = pl["storeys"][0]
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(f))
    return f


def parse_path_d(d):
    """Absolute M/L/A (+Z) paths as written by IfcConvert. Arcs contribute their end point and mid-sweep control (bbox only)."""
    toks = re.findall(r"[MLAZmlaz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    pts, i, cmd = [], 0, None
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t.upper(); i += 1
            continue
        if cmd in ("M", "L"):
            pts.append((float(toks[i]), float(toks[i + 1]))); i += 2
        elif cmd == "A":
            x, y = float(toks[i + 5]), float(toks[i + 6])
            if pts:
                (px, py), r = pts[-1], float(toks[i])
                # approximate the swing: add the arc's bulge point (good enough for a highlight box)
                mx, my = (px + x) / 2, (py + y) / 2; dx, dy = x - px, y - py; L = (dx * dx + dy * dy) ** .5 or 1
                h = max(0.0, r - (max(r * r - (L / 2) ** 2, 0)) ** .5)
                sweep = toks[i + 4] == "1"
                nx, ny = (-dy / L, dx / L) if sweep else (dy / L, -dx / L)
                pts.append((mx - nx * h, my - ny * h))
            pts.append((x, y)); i += 7
        else:
            i += 1
    return pts


def plans(model):
    out = []
    for svg in sorted(glob.glob(str(IFC_DIR / f"{model}_plan-*.svg"))):
        s = open(svg, encoding="utf-8", errors="ignore").read()
        vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
        storeys = re.findall(r'class="IfcBuildingStorey" data-name="([^"]*)"', s)
        st_uuids = sorted(set(re.findall(r'<g id="storey-([0-9a-f-]{36})-product', s)))
        groups = []
        for m in re.finditer(r'<g id="[^"]*" class="(?:cut )?(Ifc\w+)" data-name="([^"]*)" data-guid="([^"]*)"[^>]*>(.*?)</g>', s, re.S):
            cls, name, guid, body = m.groups()
            polys = []
            for d in re.findall(r'<path d="([^"]+)"', body):
                pts = [((x - vb[0]) * MM2PT, (y - vb[1]) * MM2PT) for x, y in parse_path_d(d)]
                if pts:
                    polys.append([[round(x, 2), round(y, 2)] for x, y in pts])
            if not polys:
                continue
            xs = [p[0] for pl in polys for p in pl]; ys = [p[1] for pl in polys for p in pl]
            groups.append({"cls": cls, "name": name, "guid": guid, "polys": polys, "bbox": [min(xs), min(ys), max(xs), max(ys)]})
        scale = 100
        try:
            import pymupdf
            m_ = re.search(r"1\s*:\s*(\d+)", pymupdf.open(svg[:-4] + ".pdf")[0].get_text())
            scale = int(m_.group(1)) if m_ else 100
        except Exception:
            pass
        out.append({"svg": os.path.basename(svg), "pdf": os.path.basename(svg)[:-4] + ".pdf", "storeys": storeys, "groups": groups, "scale": scale,
                    "plan_title_storey": storeys[0] if len(storeys) == 1 else None, "storey_uuids": st_uuids})
    return out


def poly_area(p):
    return abs(sum(p[i][0] * p[i - 1][1] - p[i - 1][0] * p[i][1] for i in range(len(p)))) / 2


def plan_area_m2(group, scale=100):
    """Area of the drawn polygon(s) in m². Paper pt -> mm = 25.4/72; mm -> m = scale/1000 (scale read from the title block)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    ps = [Polygon(p).buffer(0) for p in group["polys"] if len(p) >= 3]
    a_pt2 = unary_union(ps).area if ps else 0
    return a_pt2 * (25.4 / 72) ** 2 * (scale / 1000) ** 2


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def find_storey(f, name):
    if name is None:
        return None
    n = norm(name)
    for s in f["storeys"]:
        if norm(s["name"]) == n or norm(s.get("long")) == n:
            return s["name"]
    return None


def plan_for(f, storey, pdf_name=None):
    for p in f["plans"]:
        if pdf_name and p["pdf"] == pdf_name:
            return p
    for p in f["plans"]:
        if storey and storey in p["storeys"] and len(p["storeys"]) == 1:
            return p
    return None


def ev_guid(model, e, plan=None, verified=None, extra=""):
    """Evidence item: IFC GUID + (if drawn) the polygon/bbox on the rendered PDF plan."""
    item = {"source": "ifc", "guid": e["guid"], "kind": e.get("type", ""), "text": f"{e.get('type','')} {e.get('long') or e.get('name') or ''} {extra}".strip(),
            "file": f"{model}.ifc"}
    if plan:
        g = next((g for g in plan["groups"] if g["guid"] == e["guid"]), None)
        if g is None and e.get("parts"):
            ps = set(e["parts"]); pg = [g_ for g_ in plan["groups"] if g_["guid"] in ps]
            if pg:
                bb = [min(x["bbox"][0] for x in pg), min(x["bbox"][1] for x in pg), max(x["bbox"][2] for x in pg), max(x["bbox"][3] for x in pg)]
                g = {"guid": e["guid"], "cls": e.get("type"), "bbox": bb, "polys": [], "via_parts": len(pg)}
                item["text"] += f" (drawn via {len(pg)} part(s))"
        if g:
            item.update({"pdf": plan["pdf"], "page": 0, "bbox": [round(v, 1) for v in g["bbox"]], "poly": max(g["polys"], key=poly_area) if g["cls"] == "IfcSpace" else None,
                         "verified": True if verified is None else verified})
        else:
            item["verified"] = False
    return item


def answer(question, path):
    model = model_of(path)
    try:
        f = facts(model)
    except Exception as e:
        return {"status": "abstain", "answer": f"NOT SURE: could not open {model}.ifc ({type(e).__name__})", "value": None, "evidence": []}
    qs, meas, k, sc = quotes(question), measure(question), kind(question), scope(question)
    pdf_name = os.path.basename(path) if path.endswith(".pdf") else None
    storey_q = qs[-1] if qs else None
    storey = find_storey(f, storey_q) if sc == "plan" or meas != "count" else None
    plan = plan_for(f, storey, pdf_name if sc == "plan" else None)
    if storey is None and plan and plan["plan_title_storey"] and sc == "plan":
        storey = plan["plan_title_storey"]
    els = f["elements"]
    ev = []
    if meas == "count":
        if k is None:
            return _abst("unknown element kind")
        items = list(els.get(k, []))
        note = ""
        if not items and k in ("sanitary", "light"):
            fb = els.get(f"{k}_fallback", [])
            rx = "sanit|wc|toilet|sink|basin|lavat|shower|bath|urinal" if k == "sanitary" else "light|lamp|lumin"
            tc = "IfcSanitaryTerminalType" if k == "sanitary" else "IfcLightFixtureType"
            items = [e for e in fb if e.get("typeclass") == tc]
            note = f" (IFC2x3 IfcFlowTerminal typed by {tc})"
            if not items:  # untyped terminals: fall back to the type/instance name
                items = [e for e in fb if not e.get("typeclass") and re.search(rx, (e["typename"] + " " + (e["name"] or "") + " " + e["predefined"]).lower())]
                note = " (IfcFlowTerminal by name - weak)"
        if sc == "model":
            items = [e for e in items if e.get("contained")]  # spatially contained objects (parts of aggregates are not counted)
            val = len(items)
            ev = [ev_guid(model, e, plan_for(f, e["storey"])) for e in items]
            on_plan = sum(1 for e in ev if e.get("bbox"))
            return _ok(val, f"{val} {k}s in the IFC model{note}; {on_plan} of them are drawn on a rendered plan.", ev,
                       check={"model": val, "drawn_on_plans": on_plan})
        # plan scope: what the rendered plan shows (SVG groups), cross-checked with model containment
        if storey is None:
            return _abst(f"storey {storey_q!r} not found in model")
        classes = SVG_CLASS.get(k, {f"Ifc{k.title()}"})
        drawn = {}
        for g in (plan["groups"] if plan else []):
            if g["cls"] in classes:
                drawn.setdefault(g["guid"], g)
        model_on_storey = {e["guid"]: e for e in items if e.get("direct") == storey}
        val = len(model_on_storey)  # the storey's own contents = what the plan of that storey shows
        byguid = {e["guid"]: e for e in items}
        for gid, g in drawn.items():
            e = byguid.get(gid, {"guid": gid, "type": g["cls"], "name": g["name"]})
            if gid in model_on_storey:
                ev.append({**ev_guid(model, e, plan), "verified": True})
        gids = {g["guid"] for g in (plan["groups"] if plan else [])}
        for gid, e in model_on_storey.items():
            if gid not in drawn and set(e.get("parts", [])) & gids:
                drawn[gid] = True; ev.append({**ev_guid(model, e, plan), "verified": True})
        missing = [e for gid, e in model_on_storey.items() if gid not in drawn]
        if not plan:
            ev = [{**ev_guid(model, e), "verified": None} for e in model_on_storey.values()]
            return _ok(val, f"{val} {k}s on {storey!r} (IFC storey containment). No rendered plan for this storey — model only, not cross-checked.", ev,
                       check={"model": val, "agree": None})
        ev += [{**ev_guid(model, e, plan), "verified": False} for e in missing]
        agree = not missing
        msg = f"{val} {k}s on {storey or storey_q!r} (IFC storey containment). Plan drawing: {len(model_on_storey) - len(missing)} of them found as GUID-tagged geometry" + \
              (" — agrees." if agree else f" — {len(missing)} not drawn (red).")
        return _ok(val, msg, ev, check={"plan_drawn": len(model_on_storey) - len(missing), "model": val, "agree": agree})
    if meas == "area" and len(qs) >= 2 and re.search(r"\broom|raum|pièce|rum\b|ruimte|pokoj|pomieszcz|stanza|habitaci|sala|ruum", question.lower()):
        k = "space"
    if meas == "area" and k == "space" and len(qs) >= 2:
        room = norm(qs[0])
        cands = []
        for e in els["space"]:
            labels = {norm(f"{e['long']} ({e['name']})"), norm(f"{e['name']} ({e['long']})"), norm(e["name"]), norm(e["long"])}
            if room in labels and (storey is None or e["storey"] == storey):
                cands.append(e)
        if not cands:
            return _abst(f"no IfcSpace named {qs[0]!r} on {storey_q!r}")
        if len(cands) > 1:
            vals = {round(c["area"] or -1, 2) for c in cands}
            if len(vals) > 1:
                return {"status": "conflict", "answer": f"CONFLICT: {len(cands)} spaces match {qs[0]!r} with different areas {sorted(vals)}", "value": None,
                        "evidence": [ev_guid(model, c, plan) for c in cands]}
        e = cands[0]
        it = ev_guid(model, e, plan)
        pa = None
        if plan:
            g = next((g for g in plan["groups"] if g["guid"] == e["guid"]), None)
            pa = plan_area_m2(g, plan.get("scale", 100)) if g else None
        val = e["area"]
        if val is None:
            if pa is None:
                return _abst("space has no quantity and is not drawn")
            val, src = round(pa, 2), "plan polygon (no IFC quantity)"
        else:
            src = e["area_src"]
        agree = (abs(pa - val) <= 0.02 * max(val, 1e-9)) if pa is not None else None
        it["verified"] = agree if pa is not None else None
        it["text"] += f" · IFC {src}={round(e['area'],2) if e['area'] is not None else '—'} m² · plan polygon={round(pa,2) if pa else '—'} m²"
        return _ok(round(val, 2), f"{round(val, 2)} m² ({src})." + (f" Plan polygon: {pa:.2f} m² — " + ("agrees within 2%." if agree else "differs >2%.") if pa else " Not drawn on plan."),
                   [it], check={"ifc": val, "plan": pa, "agree": agree})
    # storey totals
    if storey is None:
        return _abst(f"storey {storey_q!r} not found in model")
    key = {"area": ("space", "area") if k == "space" else ("wall", "side"), "length": ("wall", "length"), "volume": ((k if k in ("wall", "slab") else "slab"), "vol")}[meas]
    if meas == "area" and k not in ("space", "wall"):
        return _abst("area of what?")
    kk, attr = key
    sel = [e for e in els[kk] if e["storey"] == storey]
    vals = [e[attr] for e in sel if e.get(attr) is not None]
    if not sel:
        return _abst(f"no {kk}s on {storey}")
    if len(vals) < len(sel):
        return {"status": "abstain", "answer": f"NOT SURE: {len(sel) - len(vals)} of {len(sel)} {kk}s on {storey} have no {attr} quantity.", "value": round(sum(vals), 3),
                "evidence": [ev_guid(model, e, plan) for e in sel]}
    val = sum(vals)
    ev = [ev_guid(model, e, plan, extra=f"{attr}={round(e[attr], 3)}") for e in sel]
    msg = f"{round(val, 3)} {'m²' if meas == 'area' else 'm³' if meas == 'volume' else 'm'} = sum of {len(sel)} {kk}s' {attr} on {storey} (IFC base quantities)."
    if kk == "space" and plan:
        pa = sum(plan_area_m2(g, plan.get("scale", 100)) for g in plan["groups"] if g["cls"] == "IfcSpace")
        msg += f" Plan polygons sum: {pa:.2f} m²."
    return _ok(round(val, 3), msg, ev)


def _ok(val, msg, ev, check=None):
    return {"status": "answered", "answer": msg, "value": val, "evidence": ev, "check": check}


def _abst(why):
    return {"status": "abstain", "answer": f"NOT SURE: {why}.", "value": None, "evidence": []}
