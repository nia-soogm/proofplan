"""ifc_bench: deterministic answers to the IFC-Bench v2 free-text questions from the IFC model (IfcOpenShell facts).

Each question is routed by intent (regex) to a template handler that reads only the cached model facts
(bp.sources.ifcb_facts). Handlers return the answer text plus the IFC entities (GUIDs) that support it; when the same
model has a rendered plan in data/ifc_models, the entity is also boxed on that plan. Anything not mapped -> NOT SURE.
No LLM: questions asking for estimates, judgement, adjacency, routes or design intent abstain.
"""
from __future__ import annotations
import re, collections, math, functools
from .ifcb_facts import load

# ---------------------------------------------------------------- helpers

WALL = ("IfcWall", "IfcWallStandardCase", "IfcWallElementedCase")
FURN = ("IfcFurniture", "IfcFurnishingElement", "IfcSystemFurnitureElement")


def _f(x, d=2):
    return f"{x:.{d}f}"


def _pct(n, t):
    return f"{100.0 * n / t:.1f}%" if t else "0%"


def label(e):
    """Type label as the authoring tool shows it: type object name, else ObjectType, else the instance name.
    Revit writes 'Family:Type' into ObjectType and only 'Type' into the type object: the longer form is used then."""
    ot, tn = (e.get("objtype") or "").strip(), (e.get("tname") or "").strip()
    mm = re.match(r"^(.+?:.+):\d+$", (e.get("name") or "").strip())
    if mm and ":" in ot:  # Revit instance name 'Family:Type:ElementId' (ObjectType can be stale on some exports)
        return mm.group(1)
    if ot and tn and ot != tn and ot.endswith(tn) and ":" in ot:
        return ot
    for k in ("tname", "objtype", "name"):
        v = e.get(k)
        if v and str(v).strip() and str(v).strip().upper() not in ("NOTDEFINED", "USERDEFINED", "$"):
            return str(v).strip()
    return "(untyped)"


def prop(e, *keys):
    """First property value whose short or full name matches one of keys (instance first, then type)."""
    for src in (e.get("props") or {}, e.get("ttag") or {}):
        for k in keys:
            for pk, v in src.items():
                if (pk == k or pk.split(".", 1)[-1] == k) and v not in (None, ""):
                    return v
    return None


def qty(e, *keys):
    for src in (e.get("props") or {},):
        for pref in ("Qto_", "BaseQuantities", ""):
            for k in keys:
                for pk, v in src.items():
                    if pk.startswith(pref) and pk.split(".", 1)[-1] == k and isinstance(v, (int, float)) and not isinstance(v, bool):
                        return float(v)
    return None


_VOL = re.compile(r"volume|volumen", re.I)
_AREA = re.compile(r"area|fläche", re.I)
_LEN = re.compile(r"(length|width|height|depth|thickness|perimeter|diameter|elevation|höhe|breite|länge|dicke|umfang)$", re.I)


def _norm_units(F, model):
    """Quantities / measure properties are stored in project units (mm, ft ...): convert once to m, m2, m3."""
    if F.get("_scaled"):
        return
    from .ifcb_facts import units
    try:
        u = units(model)
    except Exception:
        u = {"l": F.get("unit_l", 1.0), "a": F.get("unit_l", 1.0) ** 2, "v": F.get("unit_l", 1.0) ** 3}

    def sc(k):
        k = k.split(".", 1)[-1]
        return u["v"] if _VOL.search(k) else u["a"] if _AREA.search(k) else u["l"] if _LEN.search(k) else 1.0

    if any(abs(x - 1.0) > 1e-12 for x in u.values()):
        for e in F["els"]:
            for d in (e.get("props") or {}, e.get("ttag") or {}):
                for k, v in d.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        d[k] = v * sc(k)
        for s_ in F["spaces"]:
            for vs in s_["psets"].values():
                for k, v in vs.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vs[k] = v * sc(k)
    F["_scaled"] = True


class M:
    """Model view: indexes over the cached facts."""

    def __init__(self, model):
        self.model = model
        self.F = load(model)
        _norm_units(self.F, model)
        F = self.F
        self.els = F["els"]
        self.by_cls = collections.defaultdict(list)
        for e in self.els:
            self.by_cls[e["cls"]].append(e)
        st = [s for s in F["storeys"]]
        st.sort(key=lambda s: (s.get("elev_m") if s.get("elev_m") is not None else 0))
        self.storeys = st
        self.sname = {s["guid"]: (s["name"] or s.get("long") or "?") for s in st}

    def cls(self, *names):
        out = []
        for n in names:
            out += self.by_cls.get(n, [])
        return out

    def walls(self):
        return [e for e in self.cls(*WALL) if not e.get("parent") or not any(p["guid"] == e.get("parent") and p["cls"] in WALL for p in [])]

    def ext(self, e):
        v = prop(e, "IsExternal")
        if isinstance(v, bool):
            return v
        lay = str(prop(e, "Ebene", "Layer") or "").lower()
        if re.search(r"au(ß|ss)en|exter|fassad|facade", lay):
            return True
        if re.search(r"innen|inter|trenn", lay):
            return False
        return None

    def spaces(self):
        return self.F["spaces"]

    def space_area(self, s):
        q = s["psets"]
        for pset in sorted(q, key=lambda k: (not (k or "").startswith("Qto_"), k != "BaseQuantities")):
            for k in ("NetFloorArea", "GrossFloorArea"):
                v = q[pset].get(k)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        for pset in q:
            for k in ("Area", "Fläche"):
                v = q[pset].get(k)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        return None

    def space_height(self, s):
        q = s["psets"]
        for pset in sorted(q, key=lambda k: (not (k or "").startswith("Qto_"), k != "BaseQuantities")):
            for k in ("FinishCeilingHeight", "Height", "ClearHeight", "NetHeight"):
                v = q[pset].get(k)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        return None

    def space_gross(self, s):
        q = s["psets"]
        for pset in sorted(q, key=lambda k: (not (k or "").startswith("Qto_"), k != "BaseQuantities")):
            v = q[pset].get("GrossFloorArea")
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        return None

    def space_volume(self, s):
        q = s["psets"]
        for pset in sorted(q, key=lambda k: (not (k or "").startswith("Qto_"), k != "BaseQuantities")):
            for k in ("NetVolume", "GrossVolume"):
                v = q[pset].get(k)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        return None

    def area_txt(self, s):
        a, g = self.space_area(s), self.space_gross(s)
        if a is None:
            return "area n/a"
        if g and abs(g - a) > 0.005:
            return f"{_f(a)} sq m net / {_f(g)} sq m gross"
        return f"{_f(a)} sq m"

    def space_label(self, s):
        n, l = (s.get("name") or "").strip(), (s.get("long") or "").strip()
        return f"{n}: {l}" if l and l != n else n or l

    def storey_order(self):
        return [s for s in self.storeys]

    def ground_idx(self):
        st = self.storeys
        for i, s in enumerate(st):
            if re.search(r"\b(EG|Erdgeschoss|ground|level 0|00|rez|begane grond|parterre)\b", (s["name"] or "") + " " + (s.get("long") or ""), re.I):
                return i
        el = [abs(s.get("elev_m") or 0) for s in st]
        return el.index(min(el)) if el else 0


def find_storey(m: M, q: str):
    """Storey named in the question (ground floor / level 2 / basement / attic / storey name)."""
    st = m.storeys
    if not st:
        return None
    ql = q.lower()
    for s in sorted(st, key=lambda s: -len(s["name"] or "")):
        nm = (s["name"] or "").lower()
        if len(nm) >= 3 and re.search(r"(?<!\w)" + re.escape(nm) + r"(?!\w)", ql):
            return s
    g = m.ground_idx()
    if re.search(r"ground|\beg\b|erdgeschoss", ql):
        return st[g]
    if re.search(r"basement|cellar|keller", ql):
        return st[g - 1] if g > 0 else None
    if re.search(r"attic|roof level|dach", ql):
        return st[-1]
    n = None
    mm = re.search(r"\b(?:level|floor|storey|story)\s*(\d+)\b", ql) or re.search(r"\b(\d+)(?:st|nd|rd|th)\s+(?:floor|level|storey)", ql)
    if mm:
        n = int(mm.group(1))
    else:
        for w, k in (("first", 1), ("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5)):
            if re.search(rf"\b{w}\s+(floor|level|storey)", ql):
                n = k
    if n is not None and 0 <= g + n < len(st):
        return st[g + n]
    return None


def ev(m: M, items, extra=None, limit=400):
    out = []
    for e in items[:limit]:
        nm = e.get("long") or e.get("name") or ""
        out.append({"source": "ifc", "guid": e["guid"], "kind": e.get("cls", ""), "file": f"{m.model}.ifc",
                    "text": f"{e.get('cls', '')} {e.get('name') or ''} {('- ' + label(e)) if e.get('cls') != 'IfcSpace' else ('- ' + nm)}".strip()[:160]})
    return out


def grouped(items, key=label):
    c = collections.Counter(key(e) for e in items)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def type_table(items, noun, title):
    n = len(items)
    lines = [f"{title}:", f"   Total {noun}: {n}", ""]
    for k, v in grouped(items):
        lines.append(f"   - {k}: {v} {noun} ({_pct(v, n)})")
    return "\n".join(lines)


def neg(text, items=()):
    return {"text": text, "ev": list(items), "neg": True}


def ok(text, items=()):
    return {"text": text, "ev": list(items)}


# ---------------------------------------------------------------- handlers (each: (m, q) -> dict | None)

def _raw_schema(model):
    from .ifcb_facts import BENCH_DIR
    try:
        with open(BENCH_DIR / f"{model}.ifc", "rb") as fh:
            head = fh.read(20000).decode("latin-1")
        mm = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", head)
        return mm.group(1) if mm else None
    except Exception:
        return None


def h_software(m, q):
    h = m.F["head"]
    apps = h.get("apps") or []
    a = apps[0] if apps else {}
    parts = []
    if a:
        parts.append(f"Application: {a.get('name') or ''} (version {a.get('version') or 'n/a'}; developer {a.get('dev') or 'n/a'})")
    if h.get("originating_system"):
        parts.append(f"Originating system (file header): {h['originating_system']}")
    parts.append(f"IFC schema: {_raw_schema(m.model) or h.get('schema_id') or h.get('schema')}")
    return ok("Software and standards:\n   - " + "\n   - ".join(parts), [])


def _dms(v):
    if not v:
        return "n/a"
    d, mi, s, us = (list(v) + [0, 0, 0, 0])[:4]
    dec = d + math.copysign(abs(mi) / 60 + abs(s) / 3600 + abs(us) / 3.6e9, d if d else 1)
    sec = abs(s) + abs(us) / 1e6
    return f"{d}° {abs(mi)}' {sec:.4f}\" ({dec:.6f}°; IfcCompoundPlaneAngleMeasure {tuple(v)})"


def _addr(a):
    out = []
    for v in a.values():
        if isinstance(v, str) and v.startswith("("):
            v = v.strip("(),").replace("'", "")
        if v and str(v) not in out:
            out.append(str(v))
    return ", ".join(out)


def h_project(m, q):
    F = m.F
    p = (F["project"] or [{}])[0]
    lines = ["Project Information:", f"   - Project Name: {p.get('name') or 'N/A'}", f"   - Long Name: {p.get('long') or 'N/A'}",
             f"   - Description: {p.get('desc') or 'N/A'}", f"   - Phase: {p.get('Phase') or 'N/A'}"]
    for s in F["sites"][:1]:
        lines += ["", "   Site Location:", f"   - Site Name: {s.get('name') or 'N/A'}", f"   - Latitude: {_dms(s.get('lat'))}",
                  f"   - Longitude: {_dms(s.get('lon'))}", f"   - Reference elevation: {s.get('RefElevation')}"]
        if s.get("address"):
            lines.append(f"   - Address: {_addr(s['address'])}")
    for b in F["buildings"][:1]:
        lines += ["", "   Building:", f"   - Name: {b.get('name') or 'N/A'}", f"   - Long Name: {b.get('long') or 'N/A'}",
                  f"   - Description: {b.get('desc') or 'N/A'}"]
        if b.get("address"):
            lines.append(f"   - Address: {_addr(b['address'])}")
    return ok("\n".join(lines), [])


def h_levels(m, q):
    st = m.storeys
    if not st:
        return None
    lines = ["Building Level Information:"]
    for i, s in enumerate(st):
        e = s.get("elev_m")
        nxt = st[i + 1].get("elev_m") if i + 1 < len(st) else None
        hgt = (nxt - e) if (nxt is not None and e is not None) else None
        lines.append(f"   - {s['name']}: Elevation {_f(e, 2) if e is not None else 'n/a'} m" + (f", floor-to-floor height {_f(hgt, 2)} m" if hgt is not None else ""))
    lines.append(f"   Total levels: {len(st)}")
    return ok("\n".join(lines), st)


def _spaces_by_storey(m):
    by = collections.OrderedDict((s["guid"], []) for s in m.storeys)
    for sp in m.spaces():
        by.setdefault(sp.get("storey"), []).append(sp)
    return by


def _skey(sp):
    n = sp.get("name") or ""
    return (int(n) if n.isdigit() else 10 ** 9, n)


def h_space_breakdown(m, q):
    sps = m.spaces()
    if not sps or any(m.space_area(s) is None for s in sps):
        return None
    by = _spaces_by_storey(m)
    tot = sum(m.space_area(s) for s in sps)
    lines = ["Space Breakdown by Level:", f"   Total Building: {len(sps)} spaces, {_f(tot)} sq m", ""]
    for sg, L in by.items():
        a = sum(m.space_area(s) for s in L)
        gs = [m.space_gross(s) for s in L]
        gtxt = f", {_f(sum(gs))} sq m gross" if L and all(gs) and abs(sum(gs) - a) > 0.01 else ""
        lines += [f"   {m.sname.get(sg, 'No level')}:", f"   - Number of spaces: {len(L)}", f"   - Total area: {_f(a)} sq m{gtxt} ({_pct(a, tot)})"]
        if not re.search(r"percent|function", q, re.I) or re.search(r"individual|room area|function", q, re.I):
            for s in sorted(L, key=_skey):
                lines.append(f"     • {m.space_label(s)} - {m.area_txt(s)}")
        lines.append("")
    if re.search(r"function|percent", q, re.I):
        c = collections.defaultdict(lambda: [0, 0.0])
        for s in sps:
            k = (s.get("long") or s.get("name") or "?")
            c[k][0] += 1; c[k][1] += m.space_area(s)
        lines.append("   Space functions (by space name):")
        for k, (n, a) in sorted(c.items(), key=lambda kv: -kv[1][1]):
            lines.append(f"   - {k}: {n} spaces, {_f(a)} sq m ({_pct(a, tot)})")
    return ok("\n".join(lines), sps)


def h_storey_spaces(m, q):
    s = find_storey(m, q)
    if s is None:
        return None
    L = _spaces_by_storey(m).get(s["guid"], [])
    if not L:
        return neg(f"No spaces are defined on level {s['name']} in the model.", [s])
    if any(m.space_area(x) is None for x in L):
        return None
    a = sum(m.space_area(x) for x in L)
    gs = [m.space_gross(x) for x in L]
    gtxt = f" net / {_f(sum(gs))} sq m gross" if all(gs) and abs(sum(gs) - a) > 0.01 else ""
    lines = [f"Level {s['name']} Spaces:", f"   Total: {len(L)} spaces", f"   Total area: {_f(a)} sq m{gtxt}", ""]
    if re.search(r"function|group", q, re.I):
        g = collections.defaultdict(list)
        for x in sorted(L, key=_skey):
            g[x.get("long") or x.get("name")].append(x)
        for k in sorted(g):
            lines.append(f"   {k} ({len(g[k])} spaces):")
            for x in g[k]:
                lines.append(f"     - {x.get('name')}: {m.area_txt(x)}")
    else:
        for x in sorted(L, key=_skey):
            hh = m.space_height(x)
            lines.append(f"   - {m.space_label(x)} - {m.area_txt(x)}" + (f", height {_f(hh)} m" if hh and re.search(r"height", q, re.I) else ""))
    return ok("\n".join(lines), L)


def h_area_by_level(m, q):
    sps = m.spaces()
    if not sps or any(m.space_area(s) is None for s in sps):
        return None
    by = _spaces_by_storey(m)
    tot = sum(m.space_area(s) for s in sps)
    lines = ["Floor Area by Level (sum of IfcSpace floor areas):"]
    for sg, L in by.items():
        a = sum(m.space_area(s) for s in L)
        gs = [m.space_gross(s) for s in L]
        gtxt = f" net / {_f(sum(gs))} sq m gross" if L and all(gs) and abs(sum(gs) - a) > 0.01 else ""
        lines.append(f"   - {m.sname.get(sg, 'No level')}: {_f(a)} sq m{gtxt} ({len(L)} spaces, {_pct(a, tot)})")
    lines.append(f"   Total: {_f(tot)} sq m ({len(sps)} spaces)")
    return ok("\n".join(lines), sps)


def h_largest_room(m, q):
    sps = [s for s in m.spaces() if m.space_area(s)]
    if not sps:
        return None
    s = max(sps, key=m.space_area)
    hh = m.space_height(s)
    v = m.space_volume(s)
    gv = next((vs.get("GrossVolume") for vs in s["psets"].values() if isinstance(vs.get("GrossVolume"), (int, float))), None)
    if v and gv and abs(gv - v) > 0.01:
        v = f"{_f(v)} cu m net / {_f(gv)}"
    t = (f"Largest Room: {s.get('name')}\n   Function: {s.get('long') or 'n/a'}\n   Area: {m.area_txt(s)}\n"
         f"   Level: {m.sname.get(s.get('storey'), 'n/a')}" + (f"\n   Height: {_f(hh)} m" if hh else "") + (f"\n   Volume: {v if isinstance(v, str) else _f(v)} cu m" if v else ""))
    return ok(t, [s])


def h_tallest(m, q):
    sps = [s for s in m.spaces() if m.space_height(s)]
    if not sps:
        return None
    mx = max(m.space_height(s) for s in sps)
    top = [s for s in sps if abs(m.space_height(s) - mx) < 0.005]
    if len(top) == len(sps):
        return ok(f"All rooms have a ceiling height of {_f(mx)} m.", top)
    t = f"Highest ceiling height: {_f(mx)} m\n   Rooms: " + ", ".join(f"{m.space_label(s)} ({m.sname.get(s.get('storey'), 'n/a')}, {m.area_txt(s)})" for s in sorted(top, key=_skey))
    if len(sps) <= 25:
        t += "\n\n   All rooms by ceiling height:\n" + "\n".join(f"   - {m.space_label(s)}: {_f(m.space_height(s))} m" for s in sorted(sps, key=lambda s: -m.space_height(s)))
    return ok(t, top)


def _named_spaces(m, rx):
    return [s for s in m.spaces() if re.search(rx, f"{s.get('long') or ''} {s.get('name') or ''}", re.I)]


ROOMKIND = {"toilet|restroom|wc|lavator|bathroom": r"\bwc\b|toilet|restroom|lavator|bad\b|bath|dusche|shower|sanit|herren|damen",
            "meeting|conference": r"meeting|conference|besprech|konferenz",
            "working space": r"working space|arbeitsplatz",
            "open-plan": r"open-plan|open plan|großraum",
            "office": r"office|büro|buero",
            "kitchen": r"kitchen|küche|kueche",
            "laundry|utility": r"laundry|utility|wasch|hauswirtschaft",
            "balcon": r"balcon|balkon|loggia",
            "stor": r"storage|stor\b|lager|abstell|store",
            "x-ray": r"x-?ray|röntgen"}


def h_room_kind(m, q):
    ql = q.lower()
    rx = None
    for k, v in ROOMKIND.items():
        if re.search(k, ql):
            rx = v
            break
    if rx is None:
        return None
    s = find_storey(m, q) if re.search(r"floor|level|storey|on each", ql) and not re.search(r"each floor|every floor|per floor|across|by floor|distributed", ql) else None
    L = _named_spaces(m, rx)
    if s is not None:
        L = [x for x in L if x.get("storey") == s["guid"]]
    if not L:
        return None  # a model without such named spaces: we cannot tell absence from naming
    by = collections.OrderedDict()
    for st in m.storeys:
        g = [x for x in L if x.get("storey") == st["guid"]]
        if g:
            by[st["name"]] = g
    lines = [f"Matching spaces{(' on ' + s['name']) if s else ''}: {len(L)}"]
    for k, g in by.items():
        ga = [m.space_area(x) for x in g]
        lines.append(f"   {k}: {len(g)} spaces" + (f", total {_f(sum(ga))} sq m" if all(ga) else ""))
        for x in sorted(g, key=_skey):
            a = m.space_area(x)
            lines.append(f"   - {m.space_label(x)}" + (f" ({m.area_txt(x)})" if a else ""))
    if all(m.space_area(x) for x in L):
        lines.append(f"   Total area: {_f(sum(m.space_area(x) for x in L))} sq m")
    return ok("\n".join(lines), L)


def _kind_items(m, kind):
    return {"door": m.cls("IfcDoor"), "window": m.cls("IfcWindow"), "wall": m.cls(*WALL), "column": m.cls("IfcColumn"),
            "beam": m.cls("IfcBeam"), "slab": m.cls("IfcSlab"), "railing": m.cls("IfcRailing"), "stair": m.cls("IfcStair"),
            "curtain wall": m.cls("IfcCurtainWall"), "furniture": m.cls(*FURN), "roof": m.cls("IfcRoof"),
            "covering": m.cls("IfcCovering"), "member": m.cls("IfcMember"), "plate": m.cls("IfcPlate"),
            "ramp": m.cls("IfcRamp"), "footing": m.cls("IfcFooting"), "pile": m.cls("IfcPile")}.get(kind, [])


def h_types(m, q):
    ql = q.lower()
    kind = None
    for k, rx in (("curtain wall", r"curtain"), ("door", r"\bdoor"), ("window", r"window"), ("column", r"column"), ("beam", r"\bbeam"),
                  ("railing", r"railing"), ("stair", r"\bstair"), ("furniture", r"furni"), ("wall", r"\bwall"), ("slab", r"\bslab"),
                  ("covering", r"covering"), ("roof", r"\broof")):
        if re.search(rx, ql):
            kind = k
            break
    if kind is None:
        return None
    if re.search(r"thickness|material|length|volume|\barea\b|height|layer", ql) and kind not in ("door", "window", "column", "beam", "roof", "furniture"):
        return None  # the question asks for a property this handler does not report
    items = _kind_items(m, kind)
    noun = {"furniture": "items", "stair": "stairs", "curtain wall": "curtain walls"}.get(kind, kind + "s")
    if kind == "wall" and re.search(r"exterior|outer|outside|external|interior|inside|internal|partition", ql):
        want = bool(re.search(r"exterior|outer|outside|external", ql))
        xs = [m.ext(e) for e in items]
        if all(x is None for x in xs):
            return None
        items = [e for e, x in zip(items, xs) if x is want]
        noun = ("exterior " if want else "interior ") + noun
    if not items:
        return neg(f"No {kind} elements found in the model (0 {noun}).")
    if kind == "furniture":
        st_ = find_storey(m, q) if re.search(r"floor|level", ql) and not re.search(r"by floor|per floor|distribution", ql) else None
        if st_:
            items = [e for e in items if e.get("storey") == st_["guid"]]
    t = type_table(items, noun, f"{kind.title()} Types and Quantities")
    if kind in ("window", "door"):
        t += "\n\n   Typical dimensions (OverallWidth x OverallHeight):"
        for k, _ in grouped(items):
            ws = [e for e in items if label(e) == k and e.get("OverallWidth")]
            if ws:
                w = collections.Counter((round(e["OverallWidth"], 3), round(e.get("OverallHeight") or 0, 3)) for e in ws).most_common(1)[0][0]
                t += f"\n   - {k}: {_f(w[0])} m x {_f(w[1])} m ({w[0] * 1000:.0f} mm x {w[1] * 1000:.0f} mm)"
    if len(items) <= 12:
        t += "\n\n   Individual elements:"
        for i, e in enumerate(items, 1):
            d = []
            if e.get("OverallWidth"):
                d.append(f"{e['OverallWidth'] * 1000:.0f} mm x {(e.get('OverallHeight') or 0) * 1000:.0f} mm")
            for qk in ("Length", "Width", "Height", "Depth", "CrossSectionArea", "GrossArea", "NetVolume", "GrossVolume"):
                v = qty(e, qk)
                if v is not None and not e.get("OverallWidth"):
                    d.append(f"{qk} {v:g}")
            pt = e.get("pt") if e.get("pt") not in (None, "NOTDEFINED", "USERDEFINED") else None
            t += f"\n   - {e.get('name')} ({label(e)}; {m.sname.get(e.get('storey'), 'no level')}" + (f"; {pt}" if pt else "") + (f"; {', '.join(e['mats'])}" if e.get("mats") else "") + ")" + (f": {', '.join(d)}" if d else "")
    if kind == "column" and re.search(r"siz|dimension|spec", ql):
        t += "\n" + _column_sizes(items)
    if kind in ("furniture",) and re.search(r"floor|level", ql) and re.search(r"distribut|by floor|per floor", ql):
        t += "\n\n   By level:"
        for st in m.storeys:
            g = [e for e in items if e.get("storey") == st["guid"]]
            t += f"\n   - {st['name']}: {len(g)} items"
    if kind == "stair":
        t += f"\n\n   Related circulation elements:\n   - Railings: {len(m.cls('IfcRailing'))}\n   - Ramps: {len(m.cls('IfcRamp'))}"
    return ok(t, items)


def _column_sizes(items):
    out = ["   Column dimensions by type:"]
    for k, n in grouped(items):
        g = [e for e in items if label(e) == k]
        dims = []
        for e in g[:1]:
            for key in ("Width", "Depth", "Diameter", "Breite", "Höhe"):
                v = qty(e, key)
                if v:
                    dims.append(f"{key} {v * (1000 if v < 5 else 1):.0f} mm")
        out.append(f"   - {k}: {', '.join(dims) if dims else 'dimensions in type name'} ({n} columns)")
    return "\n".join(out)


def _thick(e):
    L = e.get("layers")
    if L and all(isinstance(t, (int, float)) for _, t in L):
        return sum(t for _, t in L), "layers"
    w = qty(e, "Width", "Thickness", "Depth")
    return (w, "qto") if w else (None, None)


def _mm(v, unit_l):
    """Layer thicknesses are in model length units; convert to mm."""
    return v * unit_l * 1000.0


def h_wall_thick(m, q):
    W = m.cls(*WALL)
    if not W:
        return None
    rows = []
    for k, n in grouped(W):
        g = [e for e in W if label(e) == k]
        t, src = _thick(g[0])
        if t is None:
            return None
        mm = _mm(t, m.F["unit_l"]) if src == "layers" else t * 1000.0
        rows.append((k, mm, n))
    by_t = collections.Counter()
    for k, mm, n in rows:
        by_t[round(mm)] += n
    lines = ["Wall Thicknesses by Type:"] + [f"   - {k}: {mm:.0f} mm ({mm / 10:.1f} cm) ({n} walls)" for k, mm, n in rows]
    lines += ["", "   Walls by thickness:"] + [f"   - {t}mm: {n} walls ({_pct(n, len(W))})" for t, n in sorted(by_t.items())]
    return ok("\n".join(lines), W)


def h_wall_materials(m, q):
    ql = q.lower()
    if re.search(r"\broom\b|finish", ql):
        return None
    W = m.cls(*WALL)
    if re.search(r"exterior|outer|outside|external|interior|inside|internal|partition", ql):
        want = bool(re.search(r"exterior|outer|outside|external", ql))
        xs = [m.ext(e) for e in W]
        if all(x is None for x in xs):
            return None
        W = [e for e, x in zip(W, xs) if x is want]
    if not W:
        return None
    if not any(e.get("mats") for e in W):
        return neg("Wall material information is not available in this model (no material associations on the walls).", W)
    layers = bool(re.search(r"layer|thickness", ql))
    lines = ["Wall materials:"]
    mc = collections.Counter()
    for e in W:
        for mt in set(e.get("mats") or []):
            mc[mt] += 1
    lines.append(f"   Distinct materials: {len(mc)}")
    for mt, n in mc.most_common():
        lines.append(f"   - {mt}: {n} walls ({_pct(n, len(W))})")
    lines.append("")
    lines.append("   By wall type:")
    for k, n in grouped(W):
        g = [e for e in W if label(e) == k]
        e = g[0]
        if layers and e.get("layers"):
            tot = _mm(sum(t for _, t in e["layers"] if isinstance(t, (int, float))), m.F["unit_l"])
            lines.append(f"   {k} ({n} walls), total thickness {tot:.0f} mm:")
            for nm, t in e["layers"]:
                mm = _mm(t, m.F["unit_l"]) if isinstance(t, (int, float)) else None
                lines.append(f"      - {nm}: {mm:.1f} mm ({mm / 10:g} cm)" if mm is not None else f"      - {nm}")
        else:
            lines.append(f"   - {k} ({n} walls): {', '.join(e.get('mats') or ['n/a'])}")
    return ok("\n".join(lines), W)


def h_loadbearing(m, q):
    if re.search(r"volume|material|\broom\b|thickness|length|area", q, re.I) or (re.search(r"types", q, re.I) and not re.search(r"by type", q, re.I)):
        return None
    W = m.cls(*WALL)
    if not W:
        return None
    vals = [prop(e, "LoadBearing") for e in W]
    if not any(isinstance(v, bool) for v in vals):
        return neg("Load-bearing classification data is not available in this model (Pset_WallCommon.LoadBearing is not set).", W)
    lb = [e for e, v in zip(W, vals) if v is True]
    nl = [e for e, v in zip(W, vals) if v is False]
    lines = ["Load-Bearing Wall Analysis:", f"   - Total walls: {len(W)}", f"   - Load-bearing walls: {len(lb)} ({_pct(len(lb), len(W))})",
             f"   - Non-load-bearing walls: {len(nl)} ({_pct(len(nl), len(W))})"]
    if len(lb) + len(nl) < len(W):
        lines.append(f"   - Unspecified: {len(W) - len(lb) - len(nl)}")
    if lb:
        lines.append("   Load-bearing walls by type:")
        for k, n in grouped(lb):
            lines.append(f"   - {k}: {n}")
    if re.search(r"\binterior\b", q, re.I) and re.search(r"^are|^is", q.strip(), re.I):
        xs = [(m.ext(e), v) for e, v in zip(W, vals)]
        ilb = sum(1 for x, v in xs if x is False and v is True)
        lines.insert(0, ("Yes, some interior walls are load-bearing" if ilb else "No, the interior walls are not load bearing") + f" ({ilb} load-bearing interior walls).")
    return ok("\n".join(lines), lb or W)


def h_wall_length(m, q):
    ql = q.lower()
    W = m.cls(*WALL)
    want = None
    if re.search(r"exterior|outer|outside|external", ql):
        want = True
    elif re.search(r"interior|inside|internal|partition", ql):
        want = False
    if want is not None:
        xs = [m.ext(e) for e in W]
        if all(x is None for x in xs):
            return None
        W = [e for e, x in zip(W, xs) if x is want]
    if not W:
        return None
    Ls = [qty(e, "Length") for e in W]
    if any(v is None for v in Ls):
        if all(v is None for v in Ls):
            return neg("Wall length values are not available in this model (no Length quantities on the walls).", W)
        return None
    kind = {True: "exterior", False: "interior", None: "all"}[want]
    return ok(f"Total {kind} wall length: {_f(sum(Ls))} m ({len(W)} {kind} walls, sum of wall Length quantities).", W)


def h_firerating(m, q):
    ql = q.lower()
    kind = "door" if "door" in ql else "window" if "window" in ql else "wall" if "wall" in ql else None
    if kind is None:
        return None
    items = _kind_items(m, kind)
    fr = [(e, prop(e, "FireRating", "Fire Rating", "Feuerwiderstandsklasse")) for e in items]
    fr = [(e, v) for e, v in fr if v not in (None, "", "-", "--")]
    if not fr:
        return neg(f"Fire rating information for {kind}s is not available in this model (no FireRating property on any {kind}).", items)
    by = collections.defaultdict(list)
    for e, v in fr:
        by[(label(e), str(v))].append(e)
    lines = [f"{kind.title()}s with fire rating: {len(fr)} of {len(items)}"]
    for (k, v), g in sorted(by.items()):
        lines.append(f"   - {k}: {v} ({len(g)} {kind}s)")
    return ok("\n".join(lines), [e for e, _ in fr])


def h_extreme_width(m, q):
    ql = q.lower()
    kind = "door" if "door" in ql else "window" if "window" in ql else None
    if kind is None:
        return None
    items = [e for e in _kind_items(m, kind) if e.get("OverallWidth")]
    if not items:
        return None
    big = bool(re.search(r"widest|largest|greatest|maximum|max\b|biggest", ql))
    nm = re.search(r"\b(two|three|four|five|2|3|4|5)\b", ql)
    k = {"two": 2, "three": 3, "four": 4, "five": 5}.get(nm.group(1), int(nm.group(1)) if nm and nm.group(1).isdigit() else 1) if nm else 1
    items.sort(key=lambda e: e["OverallWidth"], reverse=big)
    word = "Largest" if big else "Smallest"
    if k == 1:
        w = items[0]["OverallWidth"]
        same = [e for e in items if abs(e["OverallWidth"] - w) < 1e-4]
        lines = [f"{word} {kind.title()} Width: {_f(w)} m ({w * 1000:.0f} mm)"]
        for e in same:
            lines.append(f"   - {e.get('name')} ({label(e)})")
        return ok("\n".join(lines), same)
    lines = [f"The {k} {kind}s with the {'greatest' if big else 'smallest'} widths:"]
    for i, e in enumerate(items[:k], 1):
        lines.append(f"   {i}. {e.get('name')} ({label(e)}) - Width: {_f(e['OverallWidth'])} m ({e['OverallWidth'] * 1000:.0f} mm)")
    return ok("\n".join(lines), items[:k])


def h_count(m, q):
    ql = q.lower()
    if not re.fullmatch(r"\s*how many (windows|doors|rooms|spaces|buildings|walls|columns|beams|slabs|stairs)( are there| exist| are)?"
                        r"( (in|inside) (this|the) (building|model|clinic|house|project|ifc file))?( (are )?modell?ed in (this|the) ifc file)?\s*\??\s*", ql):
        return None  # any qualifier (facade, house A, access to ...) changes the population: abstain
    for rx, kind in ((r"\bwindows?\b", "window"), (r"\bdoors?\b", "door"), (r"\bwalls?\b", "wall"), (r"\bcolumns?\b", "column"),
                     (r"\bbeams?\b", "beam"), (r"\bslabs?\b", "slab"), (r"\bstairs?\b", "stair")):
        if re.search(rx, ql):
            items = _kind_items(m, kind)
            return ok(f"There are {len(items)} {kind}s in this building (Ifc{kind.title()} entities).", items)
    if re.search(r"\b(rooms?|spaces?)\b", ql):
        sps = m.spaces()
        return ok(f"There are {len(sps)} rooms (IfcSpace entities) in the model.", sps)
    if re.search(r"\bbuildings?\b", ql):
        b = m.F["buildings"]
        return ok(f"There are {len(b)} buildings modeled in this IFC file (IfcBuilding entities).", b)
    return None


def h_absent(m, q):
    """'Which X are installed?' questions where the model has no such elements."""
    ql = q.lower()
    checks = [(r"light fixture|lighting", ("IfcLightFixture",), "light fixtures"),
              (r"sanitary", ("IfcSanitaryTerminal",), "sanitary fixtures"),
              (r"\bpip(e|ing)", ("IfcPipeSegment", "IfcPipeFitting"), "pipe segments or pipe fittings"),
              (r"\bmep systems?\b|\bmep components?\b", ("IfcDistributionSystem", "IfcSystem"), "MEP systems"),
              (r"heating", ("IfcSpaceHeater", "IfcBoiler", "IfcUnitaryEquipment"), "heating system components"),
              (r"cooling", ("IfcChiller", "IfcCoolingTower", "IfcCoil"), "cooling system components"),
              (r"electrical", ("IfcElectricAppliance", "IfcOutlet", "IfcSwitchingDevice", "IfcCableSegment", "IfcElectricDistributionBoard", "IfcLightFixture",
                               "IfcCableCarrierSegment", "IfcCableCarrierFitting", "IfcJunctionBox", "IfcElectricDistributionPoint", "IfcLamp"), "electrical components"),
              (r"furniture", FURN, "furniture elements"),
              (r"covering", ("IfcCovering",), "covering elements"),
              (r"\bbeams?\b", ("IfcBeam",), "beam elements"),
              (r"\bcolumns?\b", ("IfcColumn",), "columns")]
    for rx, classes, noun in checks:
        if re.search(rx, ql):
            n = sum(1 for e in m.els if kcls(e) in classes) + (len(m.F["systems"]) if "IfcSystem" in classes else 0)
            if n == 0 and not (classes[0] == "IfcSanitaryTerminal" and any("sanit" in (label(e) + " " + str(e.get("tcls"))).lower() for e in m.cls("IfcFlowTerminal"))) \
                    and not (classes[0] == "IfcLightFixture" and any("light" in str(e.get("tcls")).lower() for e in m.cls("IfcFlowTerminal"))):
                return neg(f"No {noun} were found in this model (0 elements).")
            return None
    return None


def h_struct_summary(m, q):
    if re.search(r"system|material|assessment|efficien|main structural|concrete|volume|walls", q, re.I) or \
            (re.search(r"roof", q, re.I) and not re.search(r"slabs, members, and beams", q, re.I)):
        return None
    sl = m.cls("IfcSlab")
    cat = collections.defaultdict(list)
    for e in sl:
        cat[_slab_cat(e)].append(e)
    rows = [("Beams", m.cls("IfcBeam")), ("Columns", m.cls("IfcColumn")), ("Slabs", sl), ("Foundations", m.cls("IfcFooting", "IfcPile")),
            ("Structural members", m.cls("IfcMember")), ("Load-bearing walls", [e for e in m.cls(*WALL) if prop(e, "LoadBearing") is True])]
    lines = ["Structural Elements Summary:"] + [f"   - {k}: {len(v)} elements" for k, v in rows]
    nm = {"FLOOR": "Floor slabs", "ROOF": "Roof slabs", "LANDING": "Landing slabs", "BASESLAB": "Base/foundation slabs", "OTHER": "Other slabs"}
    lines.insert(4, "     (" + ", ".join(f"{nm[k]}: {len(v)}" for k, v in cat.items()) + ")")
    lines.append(f"   Total structural elements (beams, columns, slabs, footings, members): {len(m.cls('IfcBeam')) + len(m.cls('IfcColumn')) + len(sl) + len(m.cls('IfcFooting', 'IfcPile')) + len(m.cls('IfcMember'))}")
    if re.search(r"roof", q, re.I):
        rs = cat.get("ROOF", [])
        lines.append("   Roof slabs:")
        for e in rs:
            th, src = _thick(e)
            mm = (_mm(th, m.F["unit_l"]) if src == "layers" else th * 1000) if th else None
            a, v = qty(e, "GrossArea", "NetArea"), qty(e, "NetVolume", "GrossVolume")
            lines.append(f"   - {e.get('name')} ({label(e)})" + (f": {_f(a)} sq m" if a else "") + (f", {mm:.0f} mm thick" if mm else "") + (f", {_f(v)} cu m" if v else ""))
    return ok("\n".join(lines), [e for _, v in rows for e in v])


def h_slab_types(m, q):
    sl = m.cls("IfcSlab")
    if not sl:
        return neg("No slab elements found in the model (0 slabs).")
    lines = [f"Slab Types and Thicknesses:", f"   Total slabs: {len(sl)}", ""]
    by_pt = collections.Counter(str(e.get("pt") or e.get("tpt") or "NOTDEFINED") for e in sl)
    for k, v in by_pt.most_common():
        lines.append(f"   - {k} slabs: {v}")
    lines.append("")
    for k, n in grouped(sl):
        g = [e for e in sl if label(e) == k]
        t, src = _thick(g[0])
        mm = (_mm(t, m.F["unit_l"]) if src == "layers" else t * 1000) if t else None
        lines.append(f"   - {k}: {n} slab(s)" + (f", thickness {mm:.0f} mm ({mm / 10:g} cm)" if mm else ""))
    return ok("\n".join(lines), sl)


def h_slab_volume(m, q):
    sl = [e for e in m.cls("IfcSlab") if str(e.get("pt") or e.get("tpt") or "").upper() != "ROOF"]
    if not sl:
        return None
    vs = [qty(e, "NetVolume", "GrossVolume") for e in sl]
    if any(v is None for v in vs):
        if all(v is None for v in vs):
            return neg("Floor slab volume values are not available in this model (no volume quantities on the slabs).", sl)
        return None
    lines = ["Floor Slab Volume:", f"   - Number of floor slabs: {len(sl)}", f"   - Total volume: {_f(sum(vs))} cu m", "", "   By slab type:"]
    for k, n in grouped(sl):
        v = sum(x for e, x in zip(sl, vs) if label(e) == k)
        lines.append(f"   - {k}: {_f(v)} cu m ({n} slabs)")
    return ok("\n".join(lines), sl)


def h_guid_dims(m, q):
    g = re.search(r"\b([0-9A-Za-z_$]{22})\b", q)
    if not g:
        return None
    e = next((x for x in m.els if x["guid"] == g.group(1)), None)
    if e is None:
        return None
    w, h = e.get("OverallWidth"), e.get("OverallHeight")
    if not w and not h:
        return None
    return ok(f"The {e['cls'][3:].lower()} with GUID {e['guid']} ({e.get('name')}, {label(e)}) has a height of {_f(h, 3)} m and a width of {_f(w, 3)} m.", [e])


def h_phases(m, q):
    c = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    for e in m.els:
        if e["cls"] == "IfcOpeningElement":
            continue
        v = prop(e, "Umbau-Status", "Renovation Status", "Phase Created", "Status", "Phase")
        if v is not None:
            c[str(v)] += 1
            by[str(v)][e["cls"]] += 1
    if not c:
        return neg("No construction phase / renovation status information is available in this model.")
    n = sum(c.values())
    lines = ["Construction phases / renovation status found:"] + [f"   - {k}: {v} elements ({_pct(v, n)})" for k, v in c.most_common()]
    for k, _ in c.most_common():
        lines.append(f"   {k} by element type: " + ", ".join(f"{cl[3:]} {v}" for cl, v in by[k].most_common()))
    return ok("\n".join(lines), [])


def _slab_cat(e):
    pt = str(e.get("pt") or "").upper()
    if pt in ("", "NOTDEFINED", "USERDEFINED", "NONE"):
        pt = str(e.get("tpt") or "").upper()
    return pt if pt in ("FLOOR", "ROOF", "LANDING", "BASESLAB") else "OTHER"


def h_slab_types2(m, q):
    sl = m.cls("IfcSlab")
    if not sl:
        return neg("No slab elements found in the model (0 slabs).")
    cats = collections.OrderedDict((k, [e for e in sl if _slab_cat(e) == k]) for k in ("FLOOR", "ROOF", "LANDING", "BASESLAB", "OTHER"))
    nm = {"FLOOR": "Floor slabs", "ROOF": "Roof slabs", "LANDING": "Landing slabs", "BASESLAB": "Base slabs", "OTHER": "Other/undefined slabs"}
    lines = ["Slab Types and Thicknesses:", f"   Total slabs: {len(sl)}", "", "   Slabs by category:"]
    lines += [f"   - {nm[k]}: {len(v)}" for k, v in cats.items() if v]
    for k, v in cats.items():
        if not v:
            continue
        lines += ["", f"   {nm[k]}:"]
        for t, n in grouped(v):
            g = [e for e in v if label(e) == t]
            ths = []
            for e in g:
                th, src = _thick(e)
                if th:
                    ths.append(_mm(th, m.F["unit_l"]) if src == "layers" else th * 1000)
            tt = ""
            if ths:
                lo, hi = min(ths), max(ths)
                tt = f", thickness {lo:.0f} mm ({lo / 10:g} cm)" if hi - lo < 1 else f", thickness {lo:.0f} - {hi:.0f} mm (average {sum(ths) / len(ths):.0f} mm)"
            ar = [qty(e, "GrossArea", "NetArea") for e in g]
            vo = [qty(e, "NetVolume", "GrossVolume") for e in g]
            extra = (f", total area {_f(sum(ar))} sq m" if all(ar) else "") + (f", total volume {_f(sum(vo))} cu m" if all(vo) else "")
            lines.append(f"   - {t}: {n} slab(s){tt}{extra}")
    return ok("\n".join(lines), sl)


def h_slab_volume2(m, q):
    sl = [e for e in m.cls("IfcSlab") if _slab_cat(e) == "FLOOR"]
    if not sl:
        return None
    nv = [qty(e, "NetVolume") for e in sl]
    gv = [qty(e, "GrossVolume") for e in sl]
    if not all(v is not None for v in nv) and not all(v is not None for v in gv):
        if all(v is None for v in nv + gv):
            return neg("Floor slab volume values are not available in this model (no volume quantities on the floor slabs).", sl)
        return None
    main = nv if all(v is not None for v in nv) else gv
    lines = ["Floor Slab Volume (slabs with PredefinedType FLOOR):", f"   - Number of floor slabs: {len(sl)}", f"   - Total volume: {_f(sum(main))} cu m"]
    if all(v is not None for v in nv) and all(v is not None for v in gv) and abs(sum(nv) - sum(gv)) > 0.01:
        lines.append(f"   - Total net volume: {_f(sum(nv))} cu m; total gross volume: {_f(sum(gv))} cu m")
    lines += ["", "   By slab type:"]
    for k, n in grouped(sl):
        v = sum(x for e, x in zip(sl, main) if label(e) == k)
        lines.append(f"   - {k}: {_f(v)} cu m ({n} slabs)")
    if len(sl) <= 10:
        lines += ["", "   Individual slabs:"]
        for e, v in zip(sl, main):
            a = qty(e, "GrossArea", "NetArea")
            th, src = _thick(e)
            mm = (_mm(th, m.F["unit_l"]) if src == "layers" else th * 1000) if th else None
            lines.append(f"   - {e.get('name')} ({m.sname.get(e.get('storey'), 'n/a')}): volume {_f(v)} cu m" + (f", area {_f(a)} sq m" if a else "") + (f", thickness {mm:.0f} mm" if mm else ""))
    return ok("\n".join(lines), sl)


def h_footprint(m, q):
    g = m.ground_idx()
    if not m.storeys:
        return None
    gs = m.storeys[g]["guid"]
    sl = [e for e in m.cls("IfcSlab") if e.get("storey") == gs and _slab_cat(e) in ("FLOOR", "BASESLAB")]
    if len(sl) != 1:
        return None
    a = qty(sl[0], "GrossArea")
    if not a:
        return None
    return ok(f"Building footprint area: {_f(a)} sq m (gross area of the ground-level floor slab {sl[0].get('name')} on {m.storeys[g]['name']}).", sl)


def h_wall_envelope(m, q):
    ql = q.lower()
    want = bool(re.search(r"exterior|outer|outside|external|envelope", ql))
    W = m.cls(*WALL)
    xs = [m.ext(e) for e in W]
    if all(x is None for x in xs):
        return None
    W = [e for e, x in zip(W, xs) if x is want]
    if not W:
        return None
    kind = "Exterior" if want else "Interior"
    Ls = [qty(e, "Length") for e in W]
    lines = [f"{kind} Walls: {len(W)} walls", "", f"   {kind} wall types:"]
    for k, n in grouped(W):
        g = [e for e in W if label(e) == k]
        th, src = _thick(g[0])
        mm = (_mm(th, m.F["unit_l"]) if src == "layers" else th * 1000) if th else None
        lines.append(f"   - {k}: {n} walls" + (f", thickness {mm:.0f} mm" if mm else "") + (f", material {', '.join(g[0].get('mats') or [])}" if g[0].get("mats") else ""))
    if all(v is not None for v in Ls):
        lines.append(f"   Total length: {_f(sum(Ls))} m")
    return ok("\n".join(lines), W)


def _base_mat(nm):
    """ArchiCAD appends the building-material ID to the name ('Kalksandstein 2816491304'): group by the base name."""
    return re.sub(r"\s+\d{5,}$", "", nm or "").strip() or nm


def h_wall_materials2(m, q):
    r = h_wall_materials(m, q)
    if not r or r.get("neg"):
        return r
    W = r["ev"]
    bases = collections.Counter()
    for e in W:
        for b in {_base_mat(x) for x in (e.get("mats") or [])}:
            bases[b] += 1
    full = {x for e in W for x in (e.get("mats") or [])}
    if len(bases) < len(full):
        lines = [f"   Base materials (ArchiCAD material IDs removed): {len(bases)}"]
        for b, n in bases.most_common():
            lines.append(f"   - {b}: {n} walls ({_pct(n, len(W))})")
            for k, c in grouped([e for e in W if b in {_base_mat(x) for x in (e.get('mats') or [])}]):
                lines.append(f"     {k}: {c} walls")
        r["text"] = r["text"] + "\n\n" + "\n".join(lines)
    return r


def h_ceiling_mat(m, q):
    cv = [e for e in m.cls("IfcCovering") if str(e.get("pt") or e.get("tpt") or "").upper() == "CEILING"]
    if not cv:
        return neg("Ceiling material information is not available in this model: no IfcCovering elements of type CEILING were found.")
    mats = collections.Counter(x for e in cv for x in (e.get("mats") or ["(no material)"]))
    return ok("Ceiling coverings: %d\n" % len(cv) + "\n".join(f"   - {k}: {v} coverings" for k, v in mats.most_common()), cv)


def h_coverings(m, q):
    cv = m.cls("IfcCovering")
    if not cv:
        return neg("Total Coverings: 0\n   No covering elements (IfcCovering) were found in the model.")
    lines = [f"Total Coverings: {len(cv)}"]
    for k, n in grouped(cv, key=lambda e: f"{label(e)} [{e.get('pt') or e.get('tpt')}]"):
        lines.append(f"   - {k}: {n}")
    return ok("\n".join(lines), cv)


def h_vertical(m, q):
    st, fl, rp, tr = m.cls("IfcStair"), m.cls("IfcStairFlight"), m.cls("IfcRamp"), m.cls("IfcTransportElement")
    shafts = _named_spaces(m, r"elevator|lift|aufzug|fahrstuhl|elevater")
    lines = ["Vertical Circulation Elements:", f"   - Stairs (IfcStair): {len(st)}", f"   - Stair flights (IfcStairFlight): {len(fl)}",
             f"   - Ramps: {len(rp)}", f"   - Elevators/escalators (IfcTransportElement): {len(tr)}"]
    if shafts:
        lines.append(f"   - Elevator/lift spaces (by space name): {len(shafts)}")
    if st:
        lines.append("   Stair types:")
        for k, n in grouped(st):
            lines.append(f"   - {k}: {n}")
    return ok("\n".join(lines), st + tr)


def h_furniture_in_space(m, q):
    from .ifcb_facts import space_containment
    ql = q.lower()
    sps = m.spaces()
    target = None
    mm = re.search(r"(?:space|room|office)\s+([0-9][\w.-]*)", ql)
    if mm:
        c = [s for s in sps if (s.get("name") or "").lower() == mm.group(1)]
        c = c or [s for s in sps if re.search(r"\b" + re.escape(mm.group(1)) + r"$", (s.get("long") or "").lower())]
        if len(c) == 1:
            target = c[0]
    if target is None:
        st = find_storey(m, q)
        for s in sorted(sps, key=lambda s: -len(s.get("long") or "")):
            lg = (s.get("long") or "").lower()
            if lg and len(lg) > 3 and lg in ql and (st is None or s.get("storey") == st["guid"]):
                if target is not None and (target.get("long") or "").lower() == lg:
                    return None  # ambiguous
                target = target or s
    if target is None:
        return None
    cont = space_containment(m.model)
    items = [e for e in m.cls(*FURN) if cont.get(e["guid"]) == target["guid"]]
    if not items:
        return None  # furniture not contained in spaces in this model: we cannot place it without geometry
    lines = [f"Furniture in space {m.space_label(target)} ({m.sname.get(target.get('storey'), 'n/a')}, {m.area_txt(target)}):", f"   Total items: {len(items)}"]
    for k, n in grouped(items):
        lines.append(f"   - {k}: {n} items")
    return ok("\n".join(lines), [target] + items)


def h_room_types(m, q):
    sps = m.spaces()
    if not sps:
        return None
    c = collections.Counter((s.get("long") or s.get("name") or "?") for s in sps)
    lines = [f"Room types (IfcSpace names): {len(c)} distinct, {len(sps)} spaces"]
    for i, (k, n) in enumerate(sorted(c.items()), 1):
        lines.append(f"   - {k} ({n} spaces)")
    return ok("\n".join(lines), sps)


def h_room_lookup(m, q):
    """Area / height / level of one room named by its number (e.g. 'room 2A12', 'treatment room 1D34')."""
    toks = re.findall(r"\b([0-9]+[A-Za-z]+[0-9]*[A-Za-z]?|[A-Za-z]?[0-9]{2,}[A-Za-z]?)\b", q)
    sps = m.spaces()
    hit = [s for s in sps if (s.get("name") or "") in toks]
    if len(hit) != 1:
        return None
    s = hit[0]
    ql = q.lower()
    hh, a = m.space_height(s), m.space_area(s)
    if re.search(r"height", ql) and hh:
        return ok(f"The ceiling height in room {s.get('name')} ({s.get('long') or ''}) is {_f(hh)} m.", [s])
    if re.search(r"area|size|how big", ql) and a:
        return ok(f"Room {s.get('name')} ({s.get('long') or ''}, {m.sname.get(s.get('storey'), 'n/a')}) has an area of {m.area_txt(s)}.", [s])
    return None


def h_named_room(m, q):
    """'size/area of the <room name> (on the <floor>)' -> the single space whose name contains all words of <room name>."""
    mm = re.search(r"(?:size|area|ceiling height|height) of (?:the )?(.+?)(?: on the (.+?))?\s*\??$", q, re.I)
    if not mm:
        return None
    words = [w for w in re.findall(r"[a-z0-9]+", mm.group(1).lower()) if w not in ("room", "rooms", "space")]
    if not words or len(words) > 4:
        return None
    st = find_storey(m, q) if mm.group(2) else None
    if mm.group(2) and st is None:
        return None
    hit = [s_ for s_ in m.spaces() if all(re.search(r"\b" + re.escape(w), (s_.get("long") or "").lower()) for w in words)
           and (st is None or s_.get("storey") == st["guid"])]
    if len(hit) != 1:
        return None
    x = hit[0]
    if re.search(r"height", q, re.I):
        hh = m.space_height(x)
        return ok(f"The ceiling height of {x.get('long')} (room {x.get('name')}) is {_f(hh)} m.", [x]) if hh else None
    a = m.space_area(x)
    return ok(f"The {x.get('long')} on {m.sname.get(x.get('storey'), 'n/a')} (room {x.get('name')}) has an area of {m.area_txt(x)}.", [x]) if a else None


def h_doors_windows(m, q):
    d, w = m.cls("IfcDoor"), m.cls("IfcWindow")
    return ok(f"There are {len(d) + len(w)} doors and windows in total: {len(w)} windows and {len(d)} doors (IfcWindow / IfcDoor entities).", d + w)


def h_building(m, q):
    b = (m.F["buildings"] or [{}])[0]
    occ = None
    for p, vs in (b.get("psets") or {}).items():
        for k, v in vs.items():
            if re.search(r"occupancy|nutzung|usage|buildingtype|gebäudetyp", k, re.I) and v:
                occ = v
    lines = ["Building:", f"   - Building Name: {b.get('name') or 'N/A'}", f"   - Long Name: {b.get('long') or 'N/A'}",
             f"   - Description: {b.get('desc') or 'N/A'}", f"   - Object type: {b.get('objtype') or 'N/A'}", f"   - Occupancy type (Pset): {occ or 'N/A'}",
             f"   - Total Levels: {len(m.storeys)}", f"   - Total Spaces: {len(m.spaces())}"]
    return ok("\n".join(lines), [])


def h_metrics(m, q):
    sps = m.spaces()
    ar = [m.space_area(s) for s in sps]
    lines = ["Building Scale:", f"   - Number of Levels: {len(m.storeys)}", f"   - Total Rooms/Spaces: {len(sps)}"]
    if sps and all(ar):
        gs = [m.space_gross(s) for s in sps]
        lines.append(f"   - Total Floor Area (sum of spaces): {_f(sum(ar))} sq m net" + (f" / {_f(sum(gs))} sq m gross" if all(gs) else ""))
    fp = h_footprint(m, q)
    if fp:
        lines.append("   - " + fp["text"])
    rows = [("Walls", m.cls(*WALL)), ("Doors", m.cls("IfcDoor")), ("Windows", m.cls("IfcWindow")), ("Slabs", m.cls("IfcSlab")),
            ("Columns", m.cls("IfcColumn")), ("Beams", m.cls("IfcBeam")), ("Members", m.cls("IfcMember")), ("Stairs", m.cls("IfcStair")),
            ("Railings", m.cls("IfcRailing")), ("Roofs", m.cls("IfcRoof"))]
    lines.append("   - Element counts: " + ", ".join(f"{k} {len(v)}" for k, v in rows))
    st = len(m.cls("IfcBeam")) + len(m.cls("IfcColumn")) + len(m.cls("IfcMember")) + len(m.cls("IfcSlab")) + len(m.cls("IfcFooting"))
    lines.append(f"   - Structural elements (beams + columns + members + slabs + footings): {st}")
    return ok("\n".join(lines), [])


def h_sanitary_types(m, q):
    items = m.cls("IfcSanitaryTerminal") or [e for e in m.cls("IfcFlowTerminal") if "sanit" in str(e.get("tcls") or "").lower()]
    if not items:
        return h_absent(m, "sanitary")
    return ok(type_table(items, "items", "Sanitary Terminal Types"), items)


GENERIC = {"IfcFlowSegment", "IfcFlowFitting", "IfcFlowTerminal", "IfcFlowController", "IfcFlowMovingDevice", "IfcEnergyConversionDevice",
           "IfcFlowStorageDevice", "IfcFlowTreatmentDevice", "IfcDistributionControlElement", "IfcDistributionFlowElement", "IfcDistributionElement",
           "IfcBuildingElementProxy"}


def kcls(e):
    """Specific IFC class: IFC2x3 generic flow elements are resolved through their type (IfcPipeSegmentType -> IfcPipeSegment)."""
    c = e["cls"]
    if c in GENERIC and e.get("tcls") and e["tcls"].endswith("Type") and e["tcls"][:-4] not in GENERIC and e["tcls"] != "IfcBuildingElementProxyType":
        return e["tcls"][:-4]
    return c


# question words -> (IFC classes, noun)
MEP = [(r"pipe fittings?|fitting types", ("IfcPipeFitting",), "pipe fittings"),
       (r"pipe accessor|valves?|flow control", ("IfcValve",), "valves / pipe accessories"),
       (r"\bpipes?\b|pipe segments?|piping", ("IfcPipeSegment",), "pipe segments"),
       (r"duct fittings?", ("IfcDuctFitting",), "duct fittings"),
       (r"duct accessor|dampers?", ("IfcDamper",), "dampers / duct accessories"),
       (r"\bducts?\b|ductwork|duct segments?", ("IfcDuctSegment",), "duct segments"),
       (r"air terminals?", ("IfcAirTerminal",), "air terminals"),
       (r"sanitary terminals?", ("IfcSanitaryTerminal",), "sanitary terminals"),
       (r"waste terminals?", ("IfcWasteTerminal",), "waste terminals"),
       (r"pumps?", ("IfcPump",), "pumps"),
       (r"\bfans?\b", ("IfcFan",), "fans"),
       (r"coils?", ("IfcCoil",), "coils"),
       (r"heat recovery", ("IfcAirToAirHeatRecovery",), "heat recovery units"),
       (r"measuring|meters?\b|flow meters?|sensors?", ("IfcFlowMeter", "IfcSensor"), "measuring devices"),
       (r"water heat|boilers?", ("IfcBoiler",), "boilers / water heaters"),
       (r"mechanical equipment", ("IfcUnitaryEquipment", "IfcAirToAirHeatRecovery", "IfcFan", "IfcPump", "IfcBoiler", "IfcChiller", "IfcCompressor", "IfcTank"), "mechanical equipment"),
       (r"controllers?", ("IfcController", "IfcFlowController"), "controllers"),
       (r"radiators?|space heaters?|heat emitters?", ("IfcSpaceHeater",), "space heaters / radiators"),
       (r"insulation", ("IfcCovering",), "insulation coverings"),
       (r"junction box", ("IfcJunctionBox",), "junction boxes"),
       (r"cable tray|cable carrier|conduits?|cable routing", ("IfcCableCarrierSegment", "IfcCableCarrierFitting"), "cable carrier segments / fittings"),
       (r"cables?\b", ("IfcCableSegment", "IfcCableFitting"), "cable segments / fittings"),
       (r"switch", ("IfcSwitchingDevice",), "switching devices"),
       (r"receptacle|outlets?|sockets?", ("IfcOutlet",), "outlets"),
       (r"distribution (equipment|boards?)|panels?\b|switchboards?", ("IfcElectricDistributionBoard",), "distribution boards"),
       (r"light fixtures?|lighting|luminaires?|lamps?", ("IfcLightFixture", "IfcLamp"), "light fixtures / lamps"),
       (r"data and communication|communication devices|data devices", ("IfcCommunicationsAppliance",), "communications appliances"),
       (r"tanks?", ("IfcTank",), "tanks")]


# question words -> Revit categories (used first when the model carries a Category property)
MEP_CAT = [(r"pipe fittings?", ("Pipe Fittings",)), (r"pipe accessor|valves?", ("Pipe Accessories",)), (r"flex(ible)? pipes?", ("Flex Pipes",)),
           (r"\bpipes?\b|pipe segments?|piping", ("Pipes",)), (r"duct fittings?", ("Duct Fittings",)), (r"duct accessor", ("Duct Accessories",)),
           (r"flex(ible)? ducts?", ("Flex Ducts",)), (r"\bducts?\b|ductwork|duct segments?", ("Ducts",)), (r"air terminals?", ("Air Terminals",)),
           (r"mechanical equipment", ("Mechanical Equipment",)), (r"plumbing fixtures?|sanitary (fixtures|equipment)", ("Plumbing Fixtures",)),
           (r"cable trays?", ("Cable Trays", "Cable Tray Fittings")), (r"conduit fittings?", ("Conduit Fittings",)), (r"conduits?", ("Conduits",)),
           (r"lighting (fixtures|components)|light fixtures?", ("Lighting Fixtures", "Lighting Devices")), (r"lighting devices?", ("Lighting Devices",)),
           (r"distribution (equipment|boards?)|electrical equipment", ("Electrical Equipment",)), (r"data and communication|data devices", ("Data Devices", "Communication Devices", "Telephone Devices")),
           (r"communication devices", ("Communication Devices",)), (r"fire alarm", ("Fire Alarm Devices",)), (r"sprinklers?", ("Sprinklers",))]


def _mep_select(m, ql):
    """(items, description) for the component kind named in the question: Revit category first, IFC class otherwise."""
    has_cat = sum(1 for e in m.els[:2000] if category(e)) > 0.5 * min(len(m.els), 2000)
    if has_cat:
        for rx, cats in MEP_CAT:
            if re.search(rx, ql):
                items = [e for e in m.els if category(e) in cats]
                if items:
                    return items, " + ".join(cats), f"Category {' / '.join(cats)}"
                break
    for rx, classes, noun in MEP:
        if re.search(rx, ql):
            items = [e for e in m.els if kcls(e) in classes]
            return items, noun, ", ".join(classes)
    return None, None, None


def h_mep_types(m, q):
    ql = q.lower()
    if re.search(r"length|distribution by (building )?level|by floor|per floor|material|connect|percentage distribution of each|by shape|thickness|\bwalls?\b|"
                 r"application|density|subsystem|sub-system|system types|routing|junction components", ql):
        return None
    items, noun, src = _mep_select(m, ql)
    if not items:
        return None
    t = type_table(items, noun.lower(), f"{noun} in the model ({src})")
    cats = collections.Counter(category(e) for e in items)
    if len(cats) > 1 and all(cats):
        t += "\n\n   By category:\n" + "\n".join(f"   - {k}: {v} units" for k, v in cats.most_common())
    return ok(t, items)


def h_mep_types_old(m, q):
    ql = q.lower()
    hit = None
    for rx, classes, noun in MEP:
        if re.search(rx, ql):
            hit = (classes, noun)
            break
    if hit is None:
        return None
    classes, noun = hit
    if re.search(r"length|distribution by (building )?level|by floor|per floor|material|connect|percentage distribution of each|by shape|by specification|thickness|\bwalls?\b|"
                 r"application|density|overview|comprehensive|subsystem|sub-system|system types", ql):
        return None  # needs a measure / grouping this handler does not compute
    items = [e for e in m.els if kcls(e) in classes]
    if not items:
        return None  # absence is reported only by the explicit absence templates
    t = type_table(items, noun, f"{noun[0].upper() + noun[1:]} in the model ({', '.join(classes)})")
    return ok(t, items)


def h_mep_length(m, q):
    ql = q.lower()
    kw = "flexible pipes" if re.search(r"flex", ql) else "pipes" if re.search(r"pipe", ql) else "ducts" if re.search(r"duct", ql) else \
        "conduits" if re.search(r"conduit", ql) else "cable trays" if re.search(r"cable tray", ql) else None
    if kw is None:
        return None
    items, noun, src = _mep_select(m, kw)
    if not items:
        return None
    items = [e for e in items if not re.search(r"fitting", (category(e) or e["cls"]).lower())]
    classes = (src,)
    Ls = [qty(e, "Length") for e in items]
    have = [v for v in Ls if v is not None]
    if len(have) < len(items):
        return None
    tot = sum(have)
    lines = [f"Length of {', '.join(classes)}:", f"   - Total segments: {len(items)}", f"   - Segments with length data: {len(have)}",
             f"   - Total length: {_f(tot)} m", f"   - Average segment length: {_f(tot / len(have))} m",
             f"   - Average per building level ({len(m.storeys)} levels): {_f(tot / max(1, len(m.storeys)))} m"]
    if re.search(r"level|floor|storey|distribution", ql):
        lines.append("   By level:")
        for st in m.storeys:
            g = [v for e, v in zip(items, Ls) if e.get("storey") == st["guid"]]
            if g:
                lines.append(f"   - {st['name']}: {len(g)} segments, {_f(sum(g))} m")
    return ok("\n".join(lines), items)


def category(e):
    v = prop(e, "Category")
    return str(v) if v else None


def h_categories(m, q):
    """MEP / Revit models: components grouped by the authoring category (Pset_ProductRequirements.Category / 'Category')."""
    items = [e for e in m.els if e["cls"] != "IfcOpeningElement"]
    cats = [category(e) for e in items]
    if sum(1 for c in cats if c) < 0.8 * max(1, len(items)):
        return None
    st = find_storey(m, q) if re.search(r"\blevel \d|\bfloor \d|\d(st|nd|rd|th) floor|ground", q, re.I) else None
    by_level = bool(re.search(r"by (building )?level|per (level|floor)|by floor|across (the )?floors", q, re.I))
    if st is not None:
        sel = [(e, c) for e, c in zip(items, cats) if e.get("storey") == st["guid"] and c]
        n = len(sel)
        cc = collections.Counter(c for _, c in sel)
        lines = [f"Components on {st['name']}: {n} ({_pct(n, len([c for c in cats if c]))} of the model)"]
        lines += [f"   - {k}: {v} ({_pct(v, n)})" for k, v in cc.most_common()]
        return ok("\n".join(lines), [e for e, _ in sel])
    sel = [(e, c) for e, c in zip(items, cats) if c]
    n = len(sel)
    cc = collections.Counter(c for _, c in sel)
    lines = [f"Components by category (authoring-tool Category property): {n} elements"]
    lines += [f"   - {k}: {v} ({_pct(v, n)})" for k, v in cc.most_common()]
    if by_level:
        lines.append("")
        lines.append("   By level:")
        for s_ in m.storeys:
            g = [c for e, c in sel if e.get("storey") == s_["guid"]]
            if g:
                lines.append(f"   {s_['name']}: {len(g)} components ({_pct(len(g), n)})")
                lines += [f"     - {k}: {v}" for k, v in collections.Counter(g).most_common()]
    return ok("\n".join(lines), [e for e, _ in sel])


# ---------------------------------------------------------------- routing

ROUTES = [
    # things we deliberately do not answer (estimates, judgement, routes, adjacency, intent)
    (r"estimat|typical|likely|suggest|strategy|histor|nearest|closest|distance|route|path taken|adjacent|connected to|ratio of clinical|"
     r"possible|could be|capacit|\bwhy\b|reinforced|\bstc\b|sound|glazed|interior layout|what town|does this model represent|density|connectivity|"
     r"integration|suggest|ratio of clinical|typical room configuration|residential units", None),
    (r"\bguid\b", h_guid_dims),
    (r"(software|program|application|tool).*(ifc|standard|schema|version)|(ifc|standard|schema).*(software|program)|modell?ing (software|program|standards?)", h_software),
    (r"project information|location (details|for)|site location", h_project),
    (r"fire ?rating", h_firerating),
    (r"load.?bearing", h_loadbearing),
    (r"(widest|narrowest|largest|smallest|greatest|maximum|minimum|biggest).*(door|window).*width|(door|window).*(widest|narrowest|greatest width|largest width)|"
     r"(narrowest|widest) (door|window)|(door|window) (model )?(has|with) the (largest|greatest|smallest) width|"
     r"(door|window)s? (have|has) the (greatest|largest|smallest) widths?|(maximum|minimum) (door|window) width", h_extreme_width),
    (r"walls (form|are used on) the (exterior|interior)|walls form the exterior envelope", h_wall_envelope),
    (r"(total|combined).*length.*(wall)|wall.*total.*length|length of (exterior|interior|inside|outside|outer|inner) walls", h_wall_length),
    (r"thickness(es)? of (the )?(various|different)? ?wall|wall (type )?thickness", h_wall_thick),
    (r"material.*wall|wall.*material", h_wall_materials2),
    (r"slab.*volume|volume.*slab", h_slab_volume2),
    (r"slab types?|types? of slabs?", h_slab_types2),
    (r"key building metrics|number of levels, building footprint", h_metrics),
    (r"footprint", h_footprint),
    (r"ceiling.*material|material.*ceiling", h_ceiling_mat),
    (r"which coverings|coverings are applied|covering elements", h_coverings),
    (r"vertical circulation", h_vertical),
    (r"furniture.*(in|available in) (working space|space|office|room|open-plan|the )|(space|office) \d+.*furniture", h_furniture_in_space),
    (r"list of all room types|all room types", h_room_types),
    (r"\broom [0-9]+[A-Za-z]+[0-9]*|treatment room|room \w*\d\w*", h_room_lookup),
    (r"openings.*doors.*windows|doors versus windows", h_doors_windows),
    (r"^what is the (size|area|ceiling height) of the ", h_named_room),
    (r"building'?s type|type of building|building type and|function of the building", h_building),
    (r"key building metrics|number of levels, building footprint", h_metrics),
    (r"inventory of sanitary|sanitary equipment|sanitary terminal", h_sanitary_types),
    (r"structural (components|elements|system)|main structural", h_struct_summary),
    (r"largest room|biggest room|room .*is the largest", h_largest_room),
    (r"tallest ceiling|highest ceiling|ceiling heights?\b.*highest", h_tallest),
    (r"construction phase|renovation", h_phases),
    (r"(heights?|elevations?) of (each|the|building)? ?(building )?levels?|level heights|floor.to.floor|elevation, floor height", h_levels),
    (r"(gross|net|usable)? ?floor area.*(each|by|per) (floor|level)|(gross|total).*area of each floor|total net.*floor area", h_area_by_level),
    (r"space breakdown by level|space distribution by level|space breakdown by (building )?level", h_space_breakdown),
    (r"(spaces|rooms) (on|in) (the )?(ground|level|attic|basement|first|second|third|fourth|\d)|space breakdown of the|rooms on the ground|rooms in the attic", h_storey_spaces),
    (r"toilet|restroom|\bwc\b|bathroom|meeting room|x-?ray room|laundry|utility room|balcon|working spaces|offices by floor|office rooms|"
     r"distribution of offices", h_room_kind),
    (r"how many (windows|doors|rooms|buildings|spaces|walls|columns|beams|slabs|stairs)", h_count),
    (r"which (sanitary|heating|cooling|mep|electrical)|what (types of )?(pip|electrical comp)|light fixtures included|furniture items are present|"
     r"which (beams|coverings)|which column types|which (sanitary fixtures|stair)|are light fixtures", h_absent),
    (r"(total|pipe|duct|conduit).*length|length.*(pipe|duct|conduit)", h_mep_length),
    (r"components.*(by category|grouped by type|by building level|category)|(electrical|equipment|system components?) distribution (by|in) (building )?level|"
     r"components of the hvac|breakdown of the plumbing system components", h_categories),
    (r"pipe|duct|valve|terminal|pump|\bfans?\b|coil|damper|heat recovery|measuring|meter|boiler|water heat|mechanical equipment|controller|radiator|"
     r"insulation|junction|cable|conduit|switch|receptacle|outlet|distribution (equipment|board)|light|lamp|communication|tank", h_mep_types),
    (r"(types?|inventory|quantities|how many of each|with their counts)", h_types),
]


NARROW = re.compile(r"\b(north|south|east|west)(ern)?\b|facade|façade|house [a-z]\b|unit [a-z0-9]\b|apartment|\broom [a-z]?\d|\bin room\b|"
                    r"\bu-?value|acoustic|cost|price|weight|accessib|access to|daylight|orientation|adjacent|between", re.I)
NARROW_OK = {"h_room_lookup", "h_furniture_in_space", "h_guid_dims", "h_named_room", "h_room_kind_ok"}


def routes(q):
    """Matching handlers in order; a blocking route (handler None) ends the list. Questions narrowed to one facade / unit /
    room / special property only go to handlers that understand that narrowing (others would answer a broader question)."""
    out = []
    if NARROW.search(q):
        return [h for h in _routes(q) if h.__name__ in NARROW_OK]
    return _routes(q)


def _routes(q):
    out = []
    for rx, h in ROUTES:
        if re.search(rx, q, re.I):
            if h is None:
                break
            out.append(h)
    return out


# Dev-tuned gate (fixed before the first test run): templates whose answers matched the gold in < 60 % of >= 3 dev answers
# abstain instead of answering. Values = (correct, answered) on dev; regenerate with eval/ifcb_eval.py --split dev.
GATED = {"h_slab_types2": (1, 6), "h_categories": (0, 5), "h_mep_types": (6, 21), "h_space_breakdown": (3, 6), "h_ceiling_mat": (3, 6),
         "h_slab_volume2": (4, 7)}


def route(q):
    r = routes(q)
    return r[0] if r else None


# ---------------------------------------------------------------- plan boxes (same GUIDs drawn in a data/ifc_models plan)

@functools.lru_cache(maxsize=1)
def _plan_index():
    """GUID -> (pdf, bbox) over the rendered ifc_models plans (cached once; cheap regex over the SVGs)."""
    import glob, os, json
    from . import JDATA, CACHE
    cf = CACHE / "ifcb" / "plan_guid_index.json"
    if cf.exists():
        return json.loads(cf.read_text())
    from .ifcsrc import parse_path_d, MM2PT
    idx = {}
    for svg in sorted(glob.glob(str(JDATA / "data" / "ifc_models" / "*_plan-*.svg"))):
        try:
            s = open(svg, encoding="utf-8", errors="ignore").read()
            vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
        except Exception:
            continue
        pdf = os.path.basename(svg)[:-4] + ".pdf"
        for mt in re.finditer(r'<g id="[^"]*" class="(?:cut )?(Ifc\w+)" data-name="([^"]*)" data-guid="([^"]*)"[^>]*>(.*?)</g>', s, re.S):
            cls, name, guid, body = mt.groups()
            pts = []
            for d in re.findall(r'<path d="([^"]+)"', body):
                pts += [((x - vb[0]) * MM2PT, (y - vb[1]) * MM2PT) for x, y in parse_path_d(d)]
            if pts and guid not in idx:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                idx[guid] = [pdf, [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]]
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(idx))
    return idx


def _boxes(evs):
    try:
        idx = _plan_index()
    except Exception:
        return evs
    for e in evs:
        hit = idx.get(e.get("guid"))
        if hit:
            e.update({"pdf": hit[0], "pdf_src": "ifc_models", "sheet": hit[0], "page": 0, "bbox": hit[1], "verified": True})
    return evs


# ---------------------------------------------------------------- entry point

def answer(question, path):
    import os
    model = re.sub(r"\.ifc$", "", os.path.basename(path))
    hs = routes(question)
    if not hs:
        return {"status": "abstain", "value": None, "evidence": [], "answer": "NOT SURE: question does not map to a model query template."}
    try:
        m = M(model)
    except Exception as e:
        return {"status": "abstain", "value": None, "evidence": [], "answer": f"NOT SURE: could not read {model}.ifc ({type(e).__name__})."}
    r = None
    for h in hs:
        try:
            r = h(m, question)
        except Exception:
            r = None
        if r:
            break
    if not r:
        return {"status": "abstain", "value": None, "evidence": [], "answer": "NOT SURE: the model does not hold the data this template needs."}
    if h.__name__ in GATED and not os.environ.get("IFCB_NO_GATE"):
        c, n = GATED[h.__name__]
        return {"status": "abstain", "value": None, "evidence": _boxes(ev(m, r["ev"]))[:50], "handler": h.__name__,
                "answer": f"NOT SURE: the model gives an answer for this template, but it matched the reference only {c}/{n} times on dev, "
                          f"so it is withheld.\nDraft (unverified):\n{r['text'][:1500]}"}
    evs = _boxes(ev(m, r["ev"]))
    if not evs:  # header / project facts: cite the file-level entities
        evs = [{"source": "ifc", "guid": (m.F["project"] or [{}])[0].get("guid"), "kind": "IfcProject", "file": f"{model}.ifc",
                "text": f"IfcProject / file header of {model}.ifc"}]
    text = r["text"] + f"\n\n(Source: {model}.ifc via IfcOpenShell; {len(r['ev'])} IFC entities cited by GUID.)"
    return {"status": "answered", "value": text, "answer": text, "evidence": evs, "handler": h.__name__}
