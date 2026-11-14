"""ifc_bench: one-pass fact extraction from an IFC model (IfcOpenShell) into a cached pickle.

Everything the free-text templates need is pulled once per model: header (schema, authoring app), project/site/building,
storeys, spaces (name, long name, storey, quantities), every IfcProduct (class, GUID, name, type name, storey, predefined
type, selected property-set values, base quantities, materials / layer sets), systems and their members.
Large MEP files (100-350 MB) take minutes to open, so answering never touches the .ifc again after this.
Usage: python -m bp.sources.ifcb_facts [model ...]   (no args = every model in data/ifc_bench)
"""
from __future__ import annotations
import pickle, re, sys, time, functools
from pathlib import Path
from . import JDATA, CACHE

BENCH_DIR = JDATA / "data" / "ifc_bench"
OUT = CACHE / "ifcb"
VERSION = 4
# property names kept for every element (MEP files have ~100k elements: keep it compact)
KEEP_PROPS = re.compile(r"^(IsExternal|LoadBearing|FireRating|Reference|Status|Phase|AcousticRating|ThermalTransmittance|"
                        r"Width|Height|Thickness|Length|Diameter|NominalDiameter|Size|OverallWidth|OverallHeight|Material|"
                        r"Category|Family|Type|Family and Type|System Name|System Type|System Classification|Phase Created|"
                        r"Phase Demolished|Renovation Status|Bauteilstatus|Tragende Funktion|Volume|Area|Perimeter|Unbounded Height|"
                        r"Limit Offset|Base Offset|Elevation|Computed Height|Height Offset From Level|Raumhöhe|Lichte Höhe|"
                        r"Nominal Diameter|Outside Diameter|Inside Diameter|Pipe Segment|Segment|Insulation Type|Insulation Thickness|"
                        r"Lining Type|Lining Thickness|Shape|Overall Size|Free Size|Cross-Sectional Area|Hydraulic Diameter|"
                        r"Number|Name|Description|Comments|Mark|Function|Assembly Code|Assembly Description|Keynote|Model|Manufacturer|"
                        r"OmniClass Number|OmniClass Title|Occupancy|Department|Base Finish|Ceiling Finish|Floor Finish|Wall Finish|"
                        r"FinishCeilingMaterial|FinishFloorMaterial|FinishWallMaterial|Finish Ceiling|Finish Floor|Finish Wall|"
                        r"Classification|ObjectTypeOverride|Roof|Slope|Pitch|PitchAngle|Span|ProjectedArea|Fire Rating|"
                        r"Feuerwiderstandsklasse|Brandschutzklasse|Feuerschutz|NetFloorArea|GrossFloorArea|NetArea|GrossArea|"
                        r"NetVolume|GrossVolume|NetSideArea|GrossSideArea|Depth|CrossSectionArea|OuterSurfaceArea|"
                        r"FinishCeilingHeight|FinishFloorHeight|NetHeight|GrossHeight|ClearHeight|Höhe|Breite|Länge|Dicke|Ebene|Layer|Umbau-Status|"
                        r"Renovation Status|Phase Created|Status|Raumname|Raumnummer|Zone|Zonenkategorie|Raumkategorie|Element-Klassifizierung)$", re.I)


def _scalar(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def _mat_info(m, el, ue):
    """(material names, layers [(name, thickness_m)], source) from the element or, failing that, its type."""
    try:
        mat = ue.get_material(el, should_inherit=True)
    except Exception:
        mat = None
    if mat is None:
        return [], [], None
    names, layers = [], []
    k = mat.is_a()
    if k == "IfcMaterial":
        names = [mat.Name]
    elif k in ("IfcMaterialLayerSetUsage", "IfcMaterialLayerSet"):
        ls = mat.ForLayerSet if k == "IfcMaterialLayerSetUsage" else mat
        for L in ls.MaterialLayers or []:
            nm = L.Material.Name if L.Material else None
            layers.append((nm, _scalar(L.LayerThickness)))
            if nm and nm not in names:
                names.append(nm)
    elif k in ("IfcMaterialConstituentSet",):
        for c in mat.MaterialConstituents or []:
            if c.Material and c.Material.Name not in names:
                names.append(c.Material.Name)
    elif k in ("IfcMaterialProfileSetUsage", "IfcMaterialProfileSet"):
        ps = mat.ForProfileSet if k == "IfcMaterialProfileSetUsage" else mat
        for p in ps.MaterialProfiles or []:
            if p.Material and p.Material.Name not in names:
                names.append(p.Material.Name)
    elif k == "IfcMaterialList":
        names = [x.Name for x in mat.Materials or []]
    else:
        try:
            names = [x.Name for x in ue.get_materials(el)]
        except Exception:
            pass
    return names, layers, k


def extract(model: str) -> dict:
    import ifcopenshell, ifcopenshell.util.element as ue, ifcopenshell.util.unit as uu
    t0 = time.time()
    f = ifcopenshell.open(str(BENCH_DIR / f"{model}.ifc"))
    S = {"l": uu.calculate_unit_scale(f, "LENGTHUNIT")}
    hdr = f.header
    fn = hdr.file_name
    head = {"schema": f.schema, "schema_id": getattr(f, "schema_identifier", f.schema),
            "originating_system": fn.originating_system, "preprocessor": fn.preprocessor_version,
            "author": list(fn.author or []), "organization": list(fn.organization or []), "timestamp": fn.time_stamp,
            "file_description": list(hdr.file_description.description or [])}
    apps = []
    for a in f.by_type("IfcApplication"):
        apps.append({"name": a.ApplicationFullName, "version": a.Version, "id": a.ApplicationIdentifier,
                     "dev": a.ApplicationDeveloper.Name if a.ApplicationDeveloper and hasattr(a.ApplicationDeveloper, "Name") else None})
    head["apps"] = apps

    def obj(o, extra=()):
        d = {"guid": o.GlobalId, "cls": o.is_a(), "name": o.Name, "desc": getattr(o, "Description", None),
             "objtype": getattr(o, "ObjectType", None), "long": getattr(o, "LongName", None)}
        for a in extra:
            d[a] = _scalar(getattr(o, a, None))
        try:
            d["psets"] = {p: {k: _scalar(v) for k, v in vs.items() if k != "id"} for p, vs in ue.get_psets(o).items()}
        except Exception:
            d["psets"] = {}
        return d

    proj = [obj(p, ("Phase",)) for p in f.by_type("IfcProject")]
    sites = []
    for s in f.by_type("IfcSite"):
        d = obj(s, ("RefElevation", "LandTitleNumber"))
        d["lat"], d["lon"] = (list(s.RefLatitude) if s.RefLatitude else None), (list(s.RefLongitude) if s.RefLongitude else None)
        try:
            a = s.SiteAddress
            d["address"] = {k: _scalar(getattr(a, k, None)) for k in ("AddressLines", "Town", "Region", "PostalCode", "Country")} if a else None
        except Exception:
            d["address"] = None
        sites.append(d)
    blds = []
    for b in f.by_type("IfcBuilding"):
        d = obj(b, ("ElevationOfRefHeight", "ElevationOfTerrain"))
        try:
            a = b.BuildingAddress
            d["address"] = {k: _scalar(getattr(a, k, None)) for k in ("AddressLines", "Town", "Region", "PostalCode", "Country")} if a else None
        except Exception:
            d["address"] = None
        blds.append(d)
    storeys = []
    for s in f.by_type("IfcBuildingStorey"):
        d = obj(s, ("Elevation",))
        if isinstance(d.get("Elevation"), (int, float)):
            d["elev_m"] = d["Elevation"] * S["l"]
        try:
            ag = ue.get_aggregate(s)
            d["parent"] = ag.GlobalId if ag else None
        except Exception:
            d["parent"] = None
        storeys.append(d)

    @functools.lru_cache(maxsize=None)
    def storey_of_id(oid):
        o = f.by_id(oid)
        c = ue.get_container(o)
        if c is None:
            c = ue.get_aggregate(o)
        for _ in range(8):
            if c is None:
                return None
            if c.is_a("IfcBuildingStorey"):
                return c.GlobalId
            if c.is_a("IfcBuilding") or c.is_a("IfcSite") or c.is_a("IfcProject"):
                return None
            nxt = ue.get_container(c) if not c.is_a("IfcSpatialStructureElement") else None
            c = nxt or ue.get_aggregate(c)
        return None

    spaces = []
    for s in f.by_type("IfcSpace"):
        d = obj(s, ("PredefinedType",))
        d["storey"] = storey_of_id(s.id())
        spaces.append(d)

    # products
    els = []
    skip = ("IfcSpatialStructureElement", "IfcSpatialElement", "IfcAnnotation", "IfcGrid", "IfcVirtualElement", "IfcPort")
    for e in f.by_type("IfcProduct"):
        if any(e.is_a(s_) for s_ in skip if _has(f, s_)):
            continue
        t = None
        try:
            t = ue.get_type(e)
        except Exception:
            pass
        d = {"guid": e.GlobalId, "cls": e.is_a(), "name": e.Name, "objtype": getattr(e, "ObjectType", None),
             "tag": getattr(e, "Tag", None), "pt": _scalar(getattr(e, "PredefinedType", None)),
             "tname": t.Name if t else None, "tcls": t.is_a() if t else None, "tpt": _scalar(getattr(t, "PredefinedType", None)) if t else None,
             "storey": storey_of_id(e.id()), "desc": getattr(e, "Description", None)}
        for a in ("OverallWidth", "OverallHeight"):
            v = getattr(e, a, None)
            if isinstance(v, (int, float)):
                d[a] = v * S["l"]
        try:
            ag = ue.get_aggregate(e)
            if ag is not None and not ag.is_a("IfcSpatialElement" if _has(f, "IfcSpatialElement") else "IfcSpatialStructureElement"):
                d["parent"] = ag.GlobalId
        except Exception:
            pass
        props, tprops = {}, {}
        try:
            for p, vs in ue.get_psets(e).items():
                for k, v in vs.items():
                    if k != "id" and KEEP_PROPS.match(k):
                        props[f"{p}.{k}"] = _scalar(v)
        except Exception:
            pass
        d["props"] = props
        if t is not None:
            d["ttag"] = _type_props(t, ue)
        names, layers, mk = _mat_info(f, e, ue)
        if names:
            d["mats"] = names
        if layers:
            d["layers"] = layers
        if mk:
            d["matkind"] = mk
        try:
            d["sys"] = sorted({g.Name for r in (e.HasAssignments or []) if r.is_a("IfcRelAssignsToGroup") for g in [r.RelatingGroup] if g.Name})
        except Exception:
            d["sys"] = []
        if hasattr(e, "FillsVoids") and e.FillsVoids:
            try:
                d["host"] = e.FillsVoids[0].RelatingOpeningElement.VoidsElements[0].RelatingBuildingElement.GlobalId
            except Exception:
                pass
        els.append(d)
    systems = []
    for g in f.by_type("IfcSystem"):
        if g.is_a("IfcStructuralAnalysisModel") or g.is_a("IfcZone"):
            continue
        n = 0
        for r in g.IsGroupedBy or []:
            n += len(r.RelatedObjects)
        systems.append({"guid": g.GlobalId, "cls": g.is_a(), "name": g.Name, "objtype": getattr(g, "ObjectType", None),
                        "pt": _scalar(getattr(g, "PredefinedType", None)), "n": n, "long": getattr(g, "LongName", None)})
    zones = [{"guid": z.GlobalId, "name": z.Name, "objtype": z.ObjectType, "members": [o.GlobalId for r in z.IsGroupedBy or [] for o in r.RelatedObjects]} for z in f.by_type("IfcZone")]
    types = []
    for t in f.by_type("IfcTypeObject"):
        types.append({"guid": t.GlobalId, "cls": t.is_a(), "name": t.Name, "n": sum(len(r.RelatedObjects) for r in (getattr(t, "Types", None) or getattr(t, "ObjectTypeOf", None) or []))})
    phases = {}
    return {"version": VERSION, "model": model, "head": head, "project": proj, "sites": sites, "buildings": blds, "storeys": storeys,
            "spaces": spaces, "els": els, "systems": systems, "zones": zones, "types": types, "unit_l": S["l"],
            "seconds": round(time.time() - t0, 1)}


def _has(f, cls):
    try:
        f.by_type(cls)
        return True
    except Exception:
        return False


def _type_props(t, ue):
    out = {}
    try:
        for p, vs in ue.get_psets(t).items():
            for k, v in vs.items():
                if k != "id" and KEEP_PROPS.match(k):
                    out[f"{p}.{k}"] = _scalar(v)
    except Exception:
        pass
    return out


@functools.lru_cache(maxsize=4)
def load(model: str) -> dict:
    p = OUT / f"{model}.pkl"
    if p.exists():
        d = pickle.loads(p.read_bytes())
        if d.get("version") == VERSION:
            return d
    d = extract(model)
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL))
    return d


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "--units"):
    ms = sys.argv[1:] or sorted(p.stem for p in BENCH_DIR.glob("*.ifc"))
    for m in ms:
        t0 = time.time()
        try:
            d = load(m)
            print(m, len(d["els"]), "els", len(d["spaces"]), "spaces", f"{time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(m, "FAILED", type(e).__name__, e, flush=True)


@functools.lru_cache(maxsize=4)
def space_containment(model: str) -> dict:
    """element GUID -> IfcSpace GUID for elements spatially contained in (or aggregated under) a space. Cached as JSON."""
    import json
    p = OUT / f"{model}.spacecont.json"
    if p.exists():
        return json.loads(p.read_text())
    import ifcopenshell
    f = ifcopenshell.open(str(BENCH_DIR / f"{model}.ifc"))
    out = {}
    for r in f.by_type("IfcRelContainedInSpatialStructure"):
        if r.RelatingStructure.is_a("IfcSpace"):
            for o in r.RelatedElements:
                out[o.GlobalId] = r.RelatingStructure.GlobalId
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


@functools.lru_cache(maxsize=None)
def units(model: str) -> dict:
    """Project unit scales to SI (length -> m, area -> m2, volume -> m3). Cached as JSON (opening big files is slow)."""
    import json
    p = OUT / f"{model}.units.json"
    if p.exists():
        return json.loads(p.read_text())
    import ifcopenshell, ifcopenshell.util.unit as uu
    f = ifcopenshell.open(str(BENCH_DIR / f"{model}.ifc"))
    u = {k: uu.calculate_unit_scale(f, n) for k, n in (("l", "LENGTHUNIT"), ("a", "AREAUNIT"), ("v", "VOLUMEUNIT"))}
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(u))
    return u


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--units":
    for m in sorted(p.stem for p in BENCH_DIR.glob("*.ifc")):
        try:
            print(m, units(m), flush=True)
        except Exception as e:
            print(m, "FAILED", e, flush=True)
