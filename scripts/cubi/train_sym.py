"""Train symbol counters on DEV only.

Stage 1 (candidate classifier, classes other/wc/sink): seed labels from manual dev label sheets
(scripts/cubi/seed_labels.json) + count-consistent pseudo-labels: on a dev image whose gold WC-seat count is c,
if the current model's top-c candidates are confident and the (c+1)-th is not, those become positives and the
rest negatives. Gold 0 => every candidate is a negative for that class.
Stage 2 (answer gate): image-level features (count, score margins, scale, label context) -> P(count is right),
learnt out-of-fold (5 folds by image) so the threshold is chosen on held-out dev images.
Saves bp/models/cubi_sym.pkl.
"""
import sys, json, pickle, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from desc_cache import D
from bp.sources import cubi_sym
from bp.sources.cubi_sym import nms_count, image_feats, gate_feats
from bp.sources import CACHE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

CLASSES = ["other", "wc", "sink", "door"]
KIND = {"wc": "wc_seat", "sink": "sink", "door": "door"}
TARGET = 0.90
LAB = json.load(open(Path(__file__).resolve().parent / "seed_labels.json"))
g = gold("dev")
data = {}
for f in sorted(D.glob("*.npz")):
    p = "data/cubicasa5k/" + f.stem + ".png"
    if p in g:
        z = np.load(f); data[p] = (z["boxes"], z["X"], float(z["s"]), int(z["nlab"]))
paths = sorted(data)
seed = collections.defaultdict(dict)
for p, b, c in LAB:
    if p in data and c in CLASSES:
        bx = data[p][0]
        k = np.nonzero((bx == np.array(b)).all(1))[0] if len(bx) else []
        if len(k):
            seed[p][int(k[0])] = CLASSES.index(c)


def mk():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def fit(train_paths, pseudo):
    X, y = [], []
    for p in train_paths:
        bx, F, _, _ = data[p]
        lab = dict(pseudo.get(p, {})); lab.update(seed.get(p, {}))
        for i, c in lab.items():
            X.append(F[i]); y.append(c)
    X, y = np.array(X), np.array(y)
    # rotation-pooled descriptors are already orientation-free; class balance via weights
    return mk().fit(X, y)


def pseudo_labels(model, train_paths):
    ps = {}
    for p in train_paths:
        bx, F, _, _ = data[p]
        if len(F) == 0:
            continue
        P = model.predict_proba(F)
        lab = {}
        ok = True
        for ci, cname in enumerate(CLASSES[1:], start=1):
            gk = g[p].get(KIND[cname])
            if gk is None:
                continue
            sc = P[:, ci]; order = np.argsort(-sc)
            if gk == 0:
                for i in np.nonzero(sc > 0.2)[0]:
                    lab[int(i)] = 0  # hard negatives: confident detections on a plan whose gold count is 0
                continue
            if cname == "sink":
                continue  # double-bowl sinks count once: per-candidate pseudo-labels would be ambiguous
            top = order[:gk]
            nxt = sc[order[gk]] if len(order) > gk else 0
            if len(top) == gk and sc[top].min() > 0.6 and nxt < 0.3:
                for i in top:
                    lab[int(i)] = ci
                for i in order[gk:]:
                    if sc[i] > 0.1:
                        lab[int(i)] = 0
        ps[p] = lab
    return ps


def stage1():
    """Out-of-fold candidate probabilities (5 folds by image) + the final candidate model on all dev; cached."""
    f = CACHE / "cubi_sym_stage1.pkl"
    if f.exists() and "--redo1" not in sys.argv:
        return pickle.load(open(f, "rb"))
    P_oof = {}
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(paths)):
        trp = [paths[i] for i in tr]
        m = fit(trp, {})
        for it in range(2):
            m = fit(trp, pseudo_labels(m, trp))
        for i in te:
            p = paths[i]; F = data[p][1]
            P_oof[p] = m.predict_proba(F) if len(F) else np.zeros((0, len(CLASSES)))
        print("fold", fold, flush=True)
    m = fit(paths, {})
    for it in range(2):
        m = fit(paths, pseudo_labels(m, paths))
    out = {"P_oof": P_oof, "cand": m}
    pickle.dump(out, open(f, "wb"))
    return out


def gate_model():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, random_state=0)


def door_gate(P_oof):
    """Doors: arc detector counts (strict / loose) + closed-region door candidates + label context -> P(strict or loose count is exact)."""
    from door_cache import get as door_get
    from bp.sources.cubi_sym import door_feats
    best = None
    for which in ("strict", "loose"):
        rows, ys = [], []
        for p in paths:
            if "door" not in g[p] or not (CACHE / "cubi_door" / (Path(p).stem + ".json")).exists():
                continue
            bx, F, s, nlab = data[p]
            n, f = door_feats(p, door_get(p), which, bx, P_oof[p], s, nlab)
            rows.append(f); ys.append(int(n == g[p]["door"]))
        rows, ys = np.array(rows, float), np.array(ys)
        conf = np.zeros(len(ys))
        for tr, te in KFold(5, shuffle=True, random_state=1).split(rows):
            conf[te] = gate_model().fit(rows[tr], ys[tr]).predict_proba(rows[te])[:, 1]
        for tau in np.arange(0.5, 0.99, 0.02):
            sel = conf >= tau
            if sel.sum() < 20:
                continue
            pr = ys[sel].mean()
            if pr >= TARGET and (best is None or sel.mean() > best["cov"]):
                best = {"which": which, "thr": 0.5, "merge": 1.0, "tau": float(tau), "cov": float(sel.mean()), "prec": float(pr), "raw_acc": float(ys.mean()), "n": int(len(ys)), "_rows": rows, "_ys": ys}
        print("door", which, "raw", round(ys.mean(), 3), {k: v for k, v in (best or {}).items() if not k.startswith("_")}, flush=True)
    if not best:
        return None, None
    return best, gate_model().fit(best.pop("_rows"), best.pop("_ys"))


def main():
    S1 = stage1(); P_oof = S1["P_oof"]
    res, gates = {}, {}
    for cname in ("wc", "sink"):
        ci = CLASSES.index(cname); kind = KIND[cname]
        best = None
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for merge in ((1.0,) if cname == "wc" else (1.0, 1.6) if cname == "sink" else (0.6, 1.1)):
                rows, ys = [], []
                for p in paths:
                    if kind not in g[p]:
                        continue
                    bx, F, s, nlab = data[p]
                    n, kept, f = gate_feats(p, bx, P_oof[p], ci, s, nlab, thr, merge)
                    rows.append(f); ys.append(int(n == g[p][kind]))
                rows, ys = np.array(rows, float), np.array(ys)
                raw = ys.mean()
                conf = np.zeros(len(ys))
                for tr, te in KFold(5, shuffle=True, random_state=1).split(rows):
                    conf[te] = gate_model().fit(rows[tr], ys[tr]).predict_proba(rows[te])[:, 1]
                for tau in np.arange(0.5, 0.99, 0.02):
                    sel = conf >= tau
                    if sel.sum() < 20:
                        continue
                    pr = ys[sel].mean()
                    if pr >= TARGET and (best is None or sel.mean() > best["cov"]):
                        best = {"thr": thr, "merge": merge, "tau": float(tau), "cov": float(sel.mean()), "prec": float(pr), "raw_acc": float(raw), "n": int(len(ys)), "_rows": rows, "_ys": ys}
                print(cname, thr, merge, "raw acc", round(raw, 3), "best so far", {k: v for k, v in (best or {}).items() if not k.startswith("_")}, flush=True)
        if best:
            gates[cname] = gate_model().fit(best.pop("_rows"), best.pop("_ys"))
        res[cname] = best
    res["door"], gates["door"] = door_gate(P_oof)
    out = Path(__file__).resolve().parents[2] / "bp/models/cubi_sym.pkl"
    pickle.dump({"cand": S1["cand"], "classes": CLASSES, "gates": gates, "params": res}, open(out, "wb"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
