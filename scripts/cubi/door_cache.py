"""Dev-side: cache door-arc detections at two strictness levels per image. usage: door_cache.py SPLIT k/n"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from bp.sources import cubi_door, cubi_sym, CACHE
D = CACHE / "cubi_door"


def get(p):
    return cubi_sym.cached_doors(p, cubi_sym.text_h(p)[0])


if __name__ == "__main__":
    split, sh = sys.argv[1], sys.argv[2]; k, n = map(int, sh.split("/"))
    G = gold(split); ps = sorted(p for p in G if "door" in G[p])[k::n]
    for i, p in enumerate(ps):
        get(p)
