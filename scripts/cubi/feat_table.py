"""Dev-side: build (and cache) feature tables for one split: storey features. usage: feat_table.py dev"""
import sys, json, pickle
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from bp.sources import cubi_cv, CACHE


def one(p):
    try:
        return p, cubi_cv.storey_all_features(p)
    except Exception as e:
        return p, repr(e)


if __name__ == "__main__":
    split = sys.argv[1]; g = gold(split)
    out = CACHE / f"cubi_storey_feat_{split}.pkl"
    have = pickle.load(open(out, "rb")) if out.exists() else {}
    ps = [p for p in g if p not in have and (CACHE / "cubi" / (Path(p).stem + ".json")).exists()]
    print(len(ps), flush=True)
    with Pool(4) as pool:
        for p, f in pool.imap_unordered(one, ps, chunksize=8):
            if isinstance(f, dict): have[p] = f
            else: print(p, f)
    pickle.dump(have, open(out, "wb")); print(len(have))
