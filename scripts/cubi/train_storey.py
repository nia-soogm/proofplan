"""Train the storey-count gate on DEV only (5-fold out-of-fold to choose the confidence threshold), save bp/models/cubi_storey.pkl.
tau (BRAINSTORM #6): max coverage with the Clopper-Pearson 95 % lower bound on OOF precision >= 0.90; tau_strict: >= 0.95.
The old rule (point precision >= TARGET) is printed for comparison (--point keeps it as the shipped tau)."""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from bp.sources import CACHE, cubi_cv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from bp.sources.calib import choose_tau, choose_strict, TARGET as CP_TARGET

TARGET = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 0.93
feat = pickle.load(open(CACHE / "cubi_storey_feat_dev.pkl", "rb")); g = gold("dev")
ps = sorted(p for p in feat if "storey" in g.get(p, {}))
keys = sorted(feat[ps[0]])
X = np.array([[feat[p][k] for k in keys] for p in ps], float); y = np.array([min(3, g[p]["storey"]) for p in ps])
mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
oof = np.zeros((len(y), 3))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    m = mk().fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])
pred = oof.argmax(1) + 1; conf = oof.max(1)
agree = np.array([any(feat[p][f"v{int(v * 100)}_n_big"] == k for v in cubi_cv.VALLEYS) for p, k in zip(ps, pred)])
print("n", len(y), "base acc", (pred == y).mean(), "majority", (y == 1).mean())
best = None
for t in np.arange(0.5, 0.99, 0.02):
    sel = (conf >= t) & agree
    if sel.sum() == 0: continue
    pr = (pred[sel] == y[sel]).mean(); cov = sel.mean()
    print(f"tau={t:.2f} cov={cov:.3f} prec={pr:.3f} n={sel.sum()}  by class:", {k: (int((pred[sel] == k).sum()), round(float((y[sel][pred[sel] == k] == k).mean()), 3) if (pred[sel] == k).any() else None) for k in (1, 2, 3)})
    if pr >= TARGET and (best is None or cov > best[1]): best = (t, cov, pr)
print("old point rule: chosen", best)
pg = np.where(agree, conf, 0.0); yg = (pred == y).astype(int)  # answers need a matching split of the sheet (agree)
cp = choose_tau(pg, yg, len(y), target=CP_TARGET, grid=np.arange(0.5, 0.99, 0.02))
cs = choose_strict(pg, yg, len(y), cp["tau"])
print("CP rule (LB >= 90 %):", cp)
print("strict  (LB >= 95 %):", cs)
tau = float(best[0]) if "--point" in sys.argv else cp["tau"]
model = mk().fit(X, y)
if "--nosave" not in sys.argv:
    pickle.dump({"model": model, "keys": keys, "tau": tau, "tau_strict": max(cs["tau"], tau), "rule": "point" if "--point" in sys.argv else "cp90",
                 "dev_oof": {"cov": cp["answered"], "prec": cp["precision"], "lower": cp["lower"], "n": cp["n"]},
                 "dev_oof_point": {"tau": float(best[0]), "cov": float(best[1]), "prec": float(best[2])},
                 "dev_oof_strict": {"cov": cs["answered"], "prec": cs["precision"], "lower": cs["lower"], "n": cs["n"]}},
                open(Path(__file__).resolve().parents[2] / "bp/models/cubi_storey.pkl", "wb"))
