"""Tessellate an IFC model for the 3D evidence view.

Cached per file (mtime + size) under data/jcache/ifc3d/. CPU only, no network.
Does not touch answers, gating or accuracy.
"""
from __future__ import annotations

import gzip, json, os, time
from pathlib import Path

from fastapi import HTTPException

from bp.sources import JDATA, CACHE
from bp.sources.ifcsrc import model_of

CACHE_DIR = CACHE / "ifc3d"
CACHE_VER = "pp-ifc3d-v1"
# Coarser than IfcOpenShell's 1 mm default: a building-scale view, much faster / smaller.
LINEAR_DEFLECTION = 0.03  # metres
CIRCLE_SEGMENTS = 12
# Spatial / void / annotation types are not useful as solid evidence meshes.
SKIP = {
    "IfcOpeningElement", "IfcOpeningStandardCase",
    "IfcSpace", "IfcSpatialZone", "IfcSpatialElement",
    "IfcAnnotation", "IfcGrid", "IfcGridAxis",
    "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcProject",
    "IfcVirtualElement", "IfcExternalSpatialElement",
    "IfcDistributionPort", "IfcPort",
}

# In-process cache of decoded payloads (geometry is the heavy part).
_MEM: dict[str, dict] = {}


def resolve_ifc(path: str) -> tuple[Path, str, str]:
    """Map a Junction data path (plan PDF or .ifc) to the IFC file on disk."""
    if not path.startswith("data/") or ".." in path or path.startswith("data/../"):
        raise HTTPException(400, "unsupported path")
    parts = path.split("/")
    if len(parts) < 3:
        raise HTTPException(400, "unsupported path")
    src = parts[1]
    if src not in ("ifc_models", "ifc_bench"):
        raise HTTPException(400, "3D view is only available for IFC models")
    if src == "ifc_models":
        model = model_of(path)
        f = (JDATA / "data" / "ifc_models" / f"{model}.ifc")
    else:
        f = JDATA / path
        model = Path(path).stem
    base = (JDATA / "data" / src).resolve()
    try:
        f = f.resolve()
    except OSError:
        raise HTTPException(404, "IFC not found")
    if base not in f.parents and f.parent != base:
        raise HTTPException(400, "unsupported path")
    if not f.exists() or f.suffix.lower() != ".ifc":
        raise HTTPException(404, f"IFC not found for {model}")
    return f, model, src


def _cache_path(model: str) -> Path:
    return CACHE_DIR / f"{model}.json.gz"


def _unit_scales(ifc):
    import ifcopenshell.util.unit as uu
    return {
        "l": uu.calculate_unit_scale(ifc, "LENGTHUNIT") or 1.0,
        "a": uu.calculate_unit_scale(ifc, "AREAUNIT") or 1.0,
        "v": uu.calculate_unit_scale(ifc, "VOLUMEUNIT") or 1.0,
    }


def _qty(el, sc) -> dict:
    """Base quantities + Overall* dims, converted to SI (m, m², m³)."""
    out = {}
    for attr in ("OverallWidth", "OverallHeight", "OverallDepth"):
        if hasattr(el, attr):
            v = getattr(el, attr)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[attr] = round(float(v) * sc["l"], 4)
    try:
        import ifcopenshell.util.element as ue
        qtos = ue.get_psets(el, qtos_only=True) or {}
    except Exception:
        qtos = {}
    for _pset, vals in qtos.items():
        for k, v in vals.items():
            if k == "id" or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            key = k
            kl = k.lower()
            if "area" in kl:
                v = float(v) * sc["a"]
            elif "volume" in kl:
                v = float(v) * sc["v"]
            elif any(s in kl for s in ("length", "width", "height", "depth", "thickness", "perimeter", "offset")):
                v = float(v) * sc["l"]
            else:
                v = float(v)
            out[key] = round(v, 4)
    return out


def _storey_name(el) -> str | None:
    try:
        import ifcopenshell.util.element as ue
        c = ue.get_container(el)
        for _ in range(6):
            if c is None:
                break
            if c.is_a("IfcBuildingStorey"):
                return c.Name or getattr(c, "LongName", None)
            if c.is_a("IfcBuilding") or c.is_a("IfcSite"):
                break
            c = ue.get_aggregate(c)
    except Exception:
        return None
    return None


def tessellate(ifc_path: Path, model: str) -> dict:
    """Walk IfcOpenShell geometry and return a compact, JSON-serialisable mesh."""
    import ifcopenshell
    import ifcopenshell.geom as geom

    t0 = time.time()
    ifc = ifcopenshell.open(str(ifc_path))
    sc = _unit_scales(ifc)
    settings = geom.settings()
    settings.set("use-world-coords", True)
    settings.set("apply-default-materials", False)
    settings.set("weld-vertices", True)
    settings.set("no-normals", True)
    settings.set("mesher-linear-deflection", LINEAR_DEFLECTION)
    settings.set("circle-segments", CIRCLE_SEGMENTS)

    n_threads = max(1, min(8, os.cpu_count() or 1))
    it = geom.iterator(settings, ifc, n_threads)
    if not it.initialize():
        return {
            "ver": CACHE_VER,
            "model": model,
            "schema": getattr(ifc, "schema", None),
            "origin": [0, 0, 0],
            "scale": 0.001,
            "bounds": {"min": [0, 0, 0], "max": [0, 0, 0]},
            "elements": [],
            "tessellate_ms": int((time.time() - t0) * 1000),
            "n_triangles": 0,
            "n_elements": 0,
        }

    # guid -> accumulating mesh (an element may yield several representations)
    acc: dict[str, dict] = {}
    minv = [1e18, 1e18, 1e18]
    maxv = [-1e18, -1e18, -1e18]
    ok = True
    while ok:
        sh = it.get()
        el = getattr(sh, "product", None)
        typ = el.is_a() if el is not None else (getattr(sh, "type", None) or "?")
        guid = getattr(sh, "guid", None) or (el.GlobalId if el is not None else None)
        if guid and typ not in SKIP and not typ.startswith("IfcOpening"):
            verts = list(sh.geometry.verts)
            faces = list(sh.geometry.faces)
            if verts and faces:
                for i in range(0, len(verts), 3):
                    x, y, z = verts[i], verts[i + 1], verts[i + 2]
                    if x < minv[0]:
                        minv[0] = x
                    if y < minv[1]:
                        minv[1] = y
                    if z < minv[2]:
                        minv[2] = z
                    if x > maxv[0]:
                        maxv[0] = x
                    if y > maxv[1]:
                        maxv[1] = y
                    if z > maxv[2]:
                        maxv[2] = z
                if guid not in acc:
                    acc[guid] = {
                        "guid": guid,
                        "cls": typ,
                        "name": (el.Name if el is not None else None) or getattr(sh, "name", None) or "",
                        "storey": _storey_name(el) if el is not None else None,
                        "qty": _qty(el, sc) if el is not None else {},
                        "v": verts,
                        "f": faces,
                    }
                else:
                    off = len(acc[guid]["v"]) // 3
                    acc[guid]["v"].extend(verts)
                    acc[guid]["f"].extend(i + off for i in faces)
                    # keep first class/name; merge qty
                    for k, v in (_qty(el, sc) if el is not None else {}).items():
                        acc[guid]["qty"].setdefault(k, v)
        ok = it.next()

    if maxv[0] < minv[0]:
        origin = [0.0, 0.0, 0.0]
        minv = maxv = [0.0, 0.0, 0.0]
    else:
        origin = [round((minv[i] + maxv[i]) / 2, 4) for i in range(3)]

    # Quantise verts to millimetres relative to origin (compact JSON, plenty for a view).
    elements = []
    n_tri = 0
    for e in acc.values():
        v_mm = []
        ox, oy, oz = origin
        vs = e["v"]
        for i in range(0, len(vs), 3):
            v_mm.extend((
                int(round((vs[i] - ox) * 1000)),
                int(round((vs[i + 1] - oy) * 1000)),
                int(round((vs[i + 2] - oz) * 1000)),
            ))
        e["v"] = v_mm
        n_tri += len(e["f"]) // 3
        elements.append(e)

    # Stable order: class then name then guid (easier to test).
    elements.sort(key=lambda e: (e["cls"], e.get("name") or "", e["guid"]))
    return {
        "ver": CACHE_VER,
        "model": model,
        "schema": getattr(ifc, "schema", None),
        "origin": origin,
        "scale": 0.001,
        "bounds": {"min": [round(x, 4) for x in minv], "max": [round(x, 4) for x in maxv]},
        "elements": elements,
        "tessellate_ms": int((time.time() - t0) * 1000),
        "n_triangles": n_tri,
        "n_elements": len(elements),
        "unit": "m",
        "src_mtime": ifc_path.stat().st_mtime,
        "src_size": ifc_path.stat().st_size,
    }


def _load_cache(ifc_path: Path, model: str) -> dict:
    key = str(ifc_path)
    st = ifc_path.stat()
    mem = _MEM.get(key)
    if mem and mem.get("src_mtime") == st.st_mtime and mem.get("src_size") == st.st_size and mem.get("ver") == CACHE_VER:
        return mem
    cf = _cache_path(model)
    if cf.exists():
        try:
            with gzip.open(cf, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            if (data.get("ver") == CACHE_VER and data.get("src_mtime") == st.st_mtime
                    and data.get("src_size") == st.st_size and data.get("model") == model):
                _MEM[key] = data
                return data
        except Exception:
            pass
    data = tessellate(ifc_path, model)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cf.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    os.replace(tmp, cf)
    _MEM[key] = data
    return data



def _bbox_m(e, origin, scale):
    vs = e.get("v") or []
    if len(vs) < 9:
        return None
    ox, oy, oz = origin
    xs, ys, zs = [], [], []
    for i in range(0, len(vs), 3):
        xs.append(vs[i] * scale + ox)
        ys.append(vs[i + 1] * scale + oy)
        zs.append(vs[i + 2] * scale + oz)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _is_site_surface(e, origin, scale) -> bool:
    """Drop campus-scale ground planes (thin in Z, far larger than a building)."""
    b = _bbox_m(e, origin, scale)
    if not b:
        return False
    dx, dy, dz = b[3] - b[0], b[4] - b[1], b[5] - b[2]
    # Campus / site slabs are wide and shallow (even 20–40 cm "terrain" fills).
    if max(dx, dy) > 90:
        return True
    if dz < 0.45 and (dx * dy > 2000 or max(dx, dy) > 70):
        return True
    return False



CORE_TYPES = {
    "IfcWall", "IfcWallStandardCase", "IfcWallElementedCase",
    "IfcDoor", "IfcWindow", "IfcColumn", "IfcBeam", "IfcSlab", "IfcRoof",
    "IfcStair", "IfcStairFlight", "IfcRailing", "IfcMember", "IfcPlate",
    "IfcCovering", "IfcFooting", "IfcRamp", "IfcCurtainWall", "IfcBuildingElementPart",
}

def _clip_to_building(elements, origin, scale, pad=8.0):
    """Keep meshes that overlap the building (walls/doors/slabs/…) plus a short pad.

    IFC site models often include car parks, trees and road markings tens of metres
    away; those dwarf the building in a default frame without helping evidence.
    """
    cores = []
    for e in elements:
        if e.get("cls") not in CORE_TYPES:
            continue
        if _is_site_surface(e, origin, scale):
            continue
        b = _bbox_m(e, origin, scale)
        if b:
            cores.append(b)
    if not cores:
        return elements
    cmin = [min(b[i] for b in cores) - pad for i in range(3)]
    cmax = [max(b[i+3] for b in cores) + pad for i in range(3)]
    kept = []
    for e in elements:
        b = _bbox_m(e, origin, scale)
        if not b:
            continue
        if b[3] < cmin[0] or b[0] > cmax[0] or b[4] < cmin[1] or b[1] > cmax[1]:
            continue
        kept.append(e)
    return kept or elements


def _norm_highlights(highlight) -> list[str]:
    out = []
    seen = set()
    for x in highlight or []:
        for p in str(x).split(","):
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def payload(path: str, highlight=None) -> dict:
    """Build the /api/ifc3d response: geometry + highlight ids that exist in the model."""
    t0 = time.time()
    ifc_path, model, src = resolve_ifc(path)
    data = _load_cache(ifc_path, model)
    origin, scale = data["origin"], data["scale"]
    elements = [e for e in data["elements"] if not _is_site_surface(e, origin, scale)]
    elements = _clip_to_building(elements, origin, scale)
    if len(elements) != len(data["elements"]):
        n_tri = sum(len(e.get("f") or []) // 3 for e in elements)
        # Tight bounds for the kept building geometry (better camera framing).
        mins, maxs = [1e18]*3, [-1e18]*3
        for e in elements:
            b = _bbox_m(e, origin, scale)
            if not b:
                continue
            for i in range(3):
                if b[i] < mins[i]:
                    mins[i] = b[i]
                if b[i+3] > maxs[i]:
                    maxs[i] = b[i+3]
        bounds = data["bounds"] if mins[0] > maxs[0] else {"min": [round(x, 4) for x in mins], "max": [round(x, 4) for x in maxs]}
    else:
        n_tri = data["n_triangles"]
        bounds = data["bounds"]
    want = _norm_highlights(highlight)
    have = {e["guid"] for e in elements}
    # Cited GUIDs with no mesh (e.g. IfcSpace, or a filtered site surface) are listed in missing.
    hl = [g for g in want if g in have]
    missing = [g for g in want if g not in have]
    load_ms = int((time.time() - t0) * 1000)
    return {
        "ok": True,
        "model": model,
        "src": src,
        "file": ifc_path.name,
        "path": path,
        "schema": data.get("schema"),
        "origin": origin,
        "scale": scale,
        "bounds": bounds,
        "unit": "m",
        "elements": elements,
        "highlight": hl,
        "missing": missing,
        "n_elements": len(elements),
        "n_triangles": n_tri,
        "tessellate_ms": data.get("tessellate_ms"),
        "load_ms": load_ms,
        "cached": load_ms < max(50, (data.get("tessellate_ms") or 0) / 4),
    }
