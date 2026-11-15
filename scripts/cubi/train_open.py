"""Train the door / window count gates (bp/sources/cubi_open.py) on DEV images only.

Per (image, kind) the candidates are opening counts at several strictness levels from the classical-CV opening detector
(wall gaps classified as door swing / window lines). A gradient-boosted classifier learns P(candidate == gold) from the
count, its agreement with the other levels, detector quality cues (wall thickness, gap count, exploded mask) and room-label
counts. 5-fold out-of-fold predictions grouped by image give the per-image top candidate and its score.
tau: the max-coverage threshold whose one-sided Clopper-Pearson 95 % lower bound on dev OOF precision is >= 0.95
(Phase E rule; stricter than the 0.90 used for rooms/fixtures). If no tau qualifies, the kind is DISABLED (never answers).
Usage: python scripts/cubi/train_open.py [--report] [--nosave]
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from bp.sources import CACHE, cubi_open as O
from bp.sources.calib import choose_tau, cp_lower
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
TARGET = 0.95
GRID = np.round(np.arange(0.5, 0.9995, 0.005), 3)


def mk():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)


def table(split="dev"):
    g = gold(split)
    rows = {k: [] for k in O.KINDS}; asked = {k: 0 for k in O.KINDS}
    for p, gk in sorted(g.items()):
        for k in O.KINDS:
            if k not in gk:
                continue
            asked[k] += 1
            for v, tags, fv, ev in O.candidates(k, p):
                rows[k].append((p, v, fv, int(v == gk[k])))
    return rows, asked


def oof_top(rk):
    X = np.array([r[2] for r in rk], float); y = np.array([r[3] for r in rk]); grp = np.array([r[0] for r in rk])
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    best = {}
    for i, r in enumerate(rk):
        if r[0] not in best or oof[i] > oof[best[r[0]]]:
            best[r[0]] = i
    idx = np.array(list(best.values()))
    return X, y, oof[idx], y[idx], idx


def main():
    rows, asked = table()
    out = {"models": {}, "tau": {}, "tau_strict": {}, "dev_oof": {}, "enabled": {},
           "rule": f"Clopper-Pearson 95% lower bound on dev OOF precision >= {TARGET}", "version": O.VERSION}
    for k in O.KINDS:
        rk = rows[k]
        X, y, pb, yb, idx = oof_top(rk)
        print(f"== {k}: asked {asked[k]}, with candidates {len(pb)}, any candidate right {len({r[0] for r in rk if r[3]}) / asked[k]:.3f}, "
              f"top-1 acc {yb.mean():.3f}")
        if "--report" in sys.argv:
            for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97):
                s = pb >= t
                if s.sum():
                    print(f"   tau {t:.2f}: answered {s.sum() / asked[k]:.3f} prec {yb[s].mean():.3f} n={s.sum()} LB {cp_lower(yb[s].sum(), s.sum()):.3f}")
        ch = choose_tau(pb, yb, asked[k], target=TARGET, grid=GRID)
        ok = ch["n"] > 0
        print(f"   CP rule (LB >= {TARGET:.0%}): tau={ch['tau']} answered {ch['answered']:.2%} precision {ch['precision']:.1%} "
              f"(LB {ch['lower']:.1%}, n={ch['n']}) -> {'SHIP' if ok else 'DISABLED (no tau reaches the bound)'}")
        out["models"][k] = mk().fit(X, y); out["tau"][k] = ch["tau"]; out["tau_strict"][k] = ch["tau"]
        out["dev_oof"][k] = dict(ch, asked=asked[k]); out["enabled"][k] = bool(ok)
    if "--nosave" not in sys.argv:
        pickle.dump(out, open(ROOT / "bp/models/cubi_open.pkl", "wb"))
        print("saved bp/models/cubi_open.pkl", out["enabled"])


if __name__ == "__main__":
    main()
