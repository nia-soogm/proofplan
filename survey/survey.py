"""Survey swiss_dwellings + ifc_models: vector/text in PDFs, label kinds, DXF layers, SVG classes."""
import glob, json, collections, re, os, sys
import pymupdf, ezdxf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bp.sources import JDATA  # noqa: E402
J = str(JDATA / "data")
out = {}
# swiss PDFs
st = collections.Counter(); words_n = []; labels = collections.Counter(); langs = collections.Counter()
for f in sorted(glob.glob(f"{J}/swiss_dwellings/*.pdf")):
    p = pymupdf.open(f)[0]
    w = p.get_text("words"); words_n.append(len(w))
    st["vector_text" if len(w) > 10 else "no_text"] += 1
    st["has_images"] += bool(p.get_images())
    txt = p.get_text()
    langs["de" if "Grundriss" in txt else "fr" if "Plan d" in txt or "Étage" in txt or "Niveau" in txt else "it" if "Pianta" in txt else "other"] += 1
    st["has_area_labels(m²)"] += bool(re.search(r"m²|m2", txt))
    st["has_room_numbers"] += bool(re.search(r"\b\d+\.\d{2}\b", txt))
out["swiss_pdf"] = {"files": len(words_n), **st, "median_words": sorted(words_n)[len(words_n)//2], "lang": dict(langs)}
layers = collections.Counter(); ents = collections.Counter()
for f in sorted(glob.glob(f"{J}/swiss_dwellings/*.dxf"))[:200]:
    msp = ezdxf.readfile(f).modelspace()
    for e in msp:
        layers[e.dxf.layer] += 1; ents[(e.dxf.layer, e.dxftype())] += 1
out["swiss_dxf_200files"] = {f"{k[0]}:{k[1]}": v for k, v in sorted(ents.items())}
# ifc PDFs
st = collections.Counter(); cls = collections.Counter()
for f in sorted(glob.glob(f"{J}/ifc_models/*.pdf")):
    p = pymupdf.open(f)[0]; w = p.get_text("words")
    kind = "plan" if "_plan-" in f else "elev/section"
    st[(kind, "vector_text" if len(w) > 5 else "few_text")] += 1
    st[(kind, "has_images")] += bool(p.get_images())
for f in glob.glob(f"{J}/ifc_models/*_plan-*.svg"):
    for c in re.findall(r'class="(?:cut )?(Ifc\w+)"', open(f, errors="ignore").read()):
        cls[c] += 1
out["ifc_pdf"] = {f"{a}:{b}": v for (a, b), v in st.items()}
out["ifc_plan_svg_classes"] = dict(cls.most_common(25))
out["ifc_files"] = {"ifc": len(glob.glob(f"{J}/ifc_models/*.ifc")), "models_with_plans": len({os.path.basename(f).split("_plan-")[0] for f in glob.glob(f"{J}/ifc_models/*_plan-*.pdf")})}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "survey.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
