"""Pre-compute (cache) RapidOCR tokens for every CubiCasa image that has questions. CPU, parallel.
Usage: python scripts/cubi/ocr_all.py [--split dev|test|all] [--shard k/n] [--v2]"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from bp.sources import JDATA, CACHE


V2 = False


def one(p):
    from bp.sources import cubicasa
    try:
        (cubicasa.ocr2 if V2 else cubicasa.ocr)(p)
        return p, None
    except Exception as e:  # noqa
        return p, repr(e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="all"); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--v2", action="store_true", help="rotated passes + low-confidence tokens (cache: cubi_ocr2)")
    a = ap.parse_args()
    V2 = a.v2
    from eval.jeval import split_of
    paths = sorted({json.loads(l)["path"] for l in open(JDATA / "questions.jsonl") if '"data/cubicasa5k/' in l})
    paths = [p for p in paths if a.split == "all" or split_of(p) == a.split]
    paths = [p for p in paths if not (CACHE / ("cubi_ocr2" if a.v2 else "cubi") / f"{Path(p).stem}.json").exists()]
    k, n = map(int, a.shard.split("/"))
    paths = paths[k::n]  # independent shard processes (a multiprocessing Pool hung on macOS with onnxruntime)
    print(len(paths), "to OCR", flush=True)
    for i, p in enumerate(paths):
        p, err = one(p)
        if err or i % 100 == 0:
            print(i, p, err or "", flush=True)
