"""Dev-side helper: per-image gold counts {path: {kind: answer}} for one split (calibration only; never used at answer time)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bp.sources import JDATA, cubicasa
from eval.jeval import split_of


def gold(split):
    g = {}
    for l in open(JDATA / "questions.jsonl"):
        if '"data/cubicasa5k/' not in l:
            continue
        q = json.loads(l)
        if split_of(q["path"]) != split:
            continue
        k = cubicasa.qkind(q["question"])
        if k:
            g.setdefault(q["path"], {})[k] = int(q["answer"])
    return g
