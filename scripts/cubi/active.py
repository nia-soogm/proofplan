"""Dev-only active learning: train on seed labels, montage the top-scored unlabelled candidates of one class.
usage: active.py CLASS OUT.png [n_per_img]"""
import sys, json, glob, random
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from bp.sources import CACHE, cubi_sym
from desc_cache import D
from label_sheet import tile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

LAB = json.load(open(Path(__file__).resolve().parent / "seed_labels.json"))


def load_all():
    data = {}
    for f in sorted(D.glob("*.npz")):
        z = np.load(f); p = "data/cubicasa5k/" + f.stem + ".png"
        data[p] = (z["boxes"], z["X"])
    return data


def train(data, cls):
    X, y = [], []
    for p, b, c in LAB:
        if c == "skip" or p not in data: continue
        boxes, F = data[p]
        k = np.nonzero((boxes == np.array(b)).all(1))[0]
        if len(k): X.append(F[k[0]]); y.append(int(c == cls))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000))
    clf.fit(np.array(X), np.array(y))
    return clf


if __name__ == "__main__":
    cls, out = sys.argv[1], sys.argv[2]; per = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    data = load_all()
    if "--heur" in sys.argv:  # ellipse-like closed region: fill, ellipse fit, aspect (shape columns of the descriptor)
        class H:
            def predict_proba(self, F):
                fill, asp, rel, rel_a, solid, ell, circ = (F[:, i] for i in range(7))
                if cls == "door":  # quarter disc between leaf, swing arc and wall
                    sc = -abs(fill - 0.79) * 6 - abs(asp - 1.05) * 2 - abs(solid - 0.97) * 3 - abs(rel - 3.0) * 0.2
                else:
                    sc = -abs(ell - 0.97) * 4 - abs(asp - 1.4) - abs(solid - 0.97) * 4 - abs(rel - 2.0) * 0.3 + (fill < 0.9) * 0.5
                return np.stack([-sc, sc], 1)
        clf = H()
    else:
        clf = train(data, cls)
    labeled = {(p, tuple(b)) for p, b, _ in LAB}
    cand = []
    for p, (boxes, F) in data.items():
        if len(F) == 0: continue
        pr = clf.predict_proba(F)[:, 1]
        order = np.argsort(-pr)
        k = 0
        for i in order:
            if (p, tuple(boxes[i].tolist())) in labeled: continue
            cand.append((pr[i], p, boxes[i].tolist())); k += 1
            if k >= per: break
    cand.sort(key=lambda t: -t[0]); random.seed(len(LAB)); top = cand[:300]; random.shuffle(top); top = top[:96]
    items = [(p, b) for _, p, b in top]
    tiles = []
    for i, (p, b) in enumerate(items):
        x0, y0, x1, y1 = b; c = {"bbox": b}
        tiles.append(tile(p, c, i))
    while len(tiles) % 12: tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]))
    json.dump(items, open(out + ".json", "w")); print(len(items), "min p", min(t[0] for t in top[:96]))
