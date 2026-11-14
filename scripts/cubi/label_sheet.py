"""Dev-only helper: montage of symbol candidates (numbered) for manual seed labelling. Writes sheet PNG + index JSON."""
import sys, json, glob, random
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from bp.sources import cubi_cv, cubi_sym, cubicasa, CACHE

def tile(path, c, idx, T=150):
    im = cubi_cv.load(path)
    x0, y0, x1, y1 = c["bbox"]; r = int(1.6 * max(x1 - x0, y1 - y0)); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    H, W = im.shape[:2]
    crop = im[max(0, cy - r):min(H, cy + r), max(0, cx - r):min(W, cx + r)].copy()
    ox, oy = max(0, cx - r), max(0, cy - r)
    cv2.rectangle(crop, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), (0, 0, 255), 1)
    t = np.full((T, T, 3), 255, np.uint8); s = (T - 14) / max(crop.shape[:2]); crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    t[:crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(t, str(idx), (2, T - 2), 0, 0.45, (255, 0, 0), 1)
    return t

if __name__ == "__main__":
    seed, out, n_img = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    near = len(sys.argv) > 4 and sys.argv[4] == "near"
    g = gold("dev"); fs = sorted(glob.glob(str(CACHE / "cubi" / "*.json"))); random.seed(seed); random.shuffle(fs)
    items = []
    for f in fs:
        o = json.load(open(f)); p = "data/cubicasa5k/" + o["stem"] + ".png"
        if p not in g: continue
        cs, s, _ = cubi_sym.candidates(p)
        labs = [t for t in o["tokens"] if cubicasa.classify(t["text"]) in ("bathroom", "wc", "sauna", "kitchen")]
        for c in cs:
            cx, cy = (c["bbox"][0] + c["bbox"][2]) / 2, (c["bbox"][1] + c["bbox"][3]) / 2
            if near and not any(abs(cx - (t["bbox"][0] + t["bbox"][2]) / 2) < 7 * s and abs(cy - (t["bbox"][1] + t["bbox"][3]) / 2) < 7 * s for t in labs):
                continue
            items.append((p, c["bbox"]))
        n_img -= 1
        if n_img == 0: break
    random.shuffle(items); items = items[:96]
    tiles = []
    for i, (p, bb) in enumerate(items):
        c = next(c for c in cubi_sym.candidates(p)[0] if c["bbox"] == bb)
        tiles.append(tile(p, c, i))
    while len(tiles) % 12: tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]
    cv2.imwrite(out, np.vstack(rows))
    json.dump(items, open(out + ".json", "w"))
    print(len(items))
