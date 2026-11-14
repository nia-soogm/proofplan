"""Warm the per-image answer caches (OCR must exist): symbol candidates/descriptors + storey layout features.
Pure functions of the pixels; no gold is read. usage: precompute.py SPLIT k/n"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bp.sources import JDATA, cubi_sym, cubi_cv
from eval.jeval import split_of
split, sh = sys.argv[1], sys.argv[2]; k, n = map(int, sh.split("/"))
paths = sorted({json.loads(l)["path"] for l in open(JDATA / "questions.jsonl") if '"data/cubicasa5k/' in l})
paths = [p for p in paths if split_of(p) == split][k::n]
for i, p in enumerate(paths):
    try:
        cubi_sym.cached_desc(p); cubi_cv.storey_all_features(p)
    except Exception as e:
        print(p, repr(e), flush=True)
    if i % 200 == 0:
        print(i, len(paths), flush=True)
