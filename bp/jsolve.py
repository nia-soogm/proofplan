"""Dispatch a Junction-data question to the right source solver."""
from .sources import swiss, ifcsrc, opening, cubicasa, ifcbench


def solve(question, path):
    src = path.split("/")[1] if path.startswith("data/") else ""
    if src == "swiss_dwellings":
        r = swiss.answer(question, path)
    elif src == "ifc_models":
        r = ifcsrc.answer(question, path)
    elif src == "cubicasa5k":
        r = cubicasa.answer(question, path)
    elif src == "ifc_bench":
        r = ifcbench.answer(question, path)
    elif src == "opening_design":
        r = opening.answer(question, path)
    else:
        r = {"status": "abstain", "value": None, "evidence": [],
             "answer": f"NOT SURE: no deterministic solver for {src or 'this source'} yet (needs a vision model / IFC file)."}
    r["source"] = src
    return r
