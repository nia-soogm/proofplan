"""Train the storey-count gate on DEV only (5-fold out-of-fold to choose the confidence threshold), save bp/models/cubi_storey.pkl."""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from bp.sources import CACHE, cubi_cv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.93
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
print("chosen", best)
model = mk().fit(X, y)
pickle.dump({"model": model, "keys": keys, "tau": float(best[0]), "dev_oof": {"cov": best[1], "prec": best[2]}}, open(Path(__file__).resolve().parents[2] / "bp/models/cubi_storey.pkl", "wb"))
