"""Dev-side: cache candidate boxes + descriptors per image (data/jcache/cubi_desc/<stem>.npz)."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bp.sources import cubi_sym, CACHE
D = CACHE / "cubi_desc"


def get(p):
    f = D / (Path(p).stem + ".npz")
    if f.exists():
        z = np.load(f)
        return z["boxes"], z["X"], float(z["s"]), int(z["nlab"])
    cs, X, s, nlab = cubi_sym.all_desc(p)
    boxes = np.array([c["bbox"] for c in cs], np.int32).reshape(-1, 4)
    D.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, boxes=boxes, X=X, s=s, nlab=nlab)
    return boxes, X, s, nlab


def _one(p):
    try:
        get(p); return None
    except Exception as e:
        return p, repr(e)


if __name__ == "__main__":
    from multiprocessing import Pool
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gold import gold
    split = sys.argv[1]
    ps = [p for p in gold(split) if (CACHE / "cubi" / (Path(p).stem + ".json")).exists() and not (D / (Path(p).stem + ".npz")).exists()]
    print(len(ps), flush=True)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as pool:
        for r in pool.imap_unordered(_one, ps, chunksize=4):
            if r: print(r, flush=True)
