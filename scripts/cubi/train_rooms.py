"""Train the room-count gates (bp/sources/cubi_rooms.py) on DEV images only.

For each (image, kind) we build the candidates (label count, summary value) and learn P(candidate == gold) with a
gradient-boosted classifier. Thresholds are chosen per kind from 5-fold out-of-fold predictions grouped by image
(so tau is calibrated on images the gate never saw). Saves bp/models/cubi_rooms.pkl.
Usage: python scripts/cubi/train_rooms.py [target_precision=0.92] [--report] [--ocr1 (old OCR cache, for comparison)]
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from bp.sources import CACHE, cubi_rooms as R
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from bp.sources.calib import choose_tau, TARGET as CP_TARGET, TARGET_STRICT, choose_strict

TARGET = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 0.92
MIN_TAU = 0.6
OCRDIR = "cubi" if "--ocr1" in sys.argv else "cubi_ocr2"
NEW = "--oldfeat" not in sys.argv  # BRAINSTORM #4: code-identity features + washroom->sauna candidate
CP = "--point" not in sys.argv  # BRAINSTORM #6: tau by Clopper-Pearson 95 % lower bound >= 0.90 (default); --point = old 92 % point rule


def table(split):
    g = gold(split)
    rows = []  # (img, kind, val, feats, gold)
    for p, gk in sorted(g.items()):
        f = CACHE / OCRDIR / f"{Path(p).stem}.json"
        if not f.exists():
            continue
        o = json.loads(f.read_text())
        ex = R.extract(o["tokens"])
        for k in R.KINDS:
            if k not in gk:
                continue
            for v, src, f, ev in R.candidates(ex, k, o["size"], use_new=NEW):
                rows.append((p, k, v, f, gk[k]))
    return g, rows


def mk():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)


def main():
    g, rows = table("dev")
    out = {"models": {}, "tau": {}, "tau_strict": {}, "dev_oof": {}, "dev_oof_strict": {}, "target": TARGET}
    for k in R.KINDS:
        rk = [r for r in rows if r[1] == k]
        n_q = len({r[0] for r in rows if r[1] == k} | {p for p in g if k in g[p] and (CACHE / OCRDIR / f"{Path(p).stem}.json").exists()})
        X = np.array([R.fvec(r[3]) for r in rk], float); y = np.array([int(r[2] == r[4]) for r in rk])
        grp = np.array([r[0] for r in rk])
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        best = {}
        for i, r in enumerate(rk):
            if r[0] not in best or oof[i] > oof[best[r[0]]]:
                best[r[0]] = i
        idx = np.array(list(best.values())); pb = oof[idx]; yb = y[idx]
        choice = None
        print(f"== {k}: {n_q} dev images asked, {len(idx)} with a candidate; raw top-candidate acc {yb.mean():.3f}")
        for t in np.arange(0.5, 0.995, 0.02):
            sel = pb >= t
            if sel.sum() < 10:
                continue
            pr = yb[sel].mean(); cov = sel.sum() / n_q
            if t >= MIN_TAU and pr >= TARGET and (choice is None or cov > choice[1]):
                choice = (float(t), float(cov), float(pr), int(sel.sum()))
            if "--report" in sys.argv:
                print(f"   tau={t:.2f} answered={cov:.3f} prec={pr:.3f} n={sel.sum()}")
        if choice is None:
            choice = (1.01, 0.0, 0.0, 0)
        if CP:
            c = choose_tau(pb, yb, n_q, target=CP_TARGET, min_tau=MIN_TAU)
            choice = (c["tau"], c["answered"], c["precision"], c["n"])
            print(f"   CP lower bound {c['lower']:.3f}")
        print(f"   chosen tau={choice[0]:.2f}: answered {choice[1]:.1%} of asked, precision {choice[2]:.1%} (n={choice[3]})")
        cs = choose_strict(pb, yb, n_q, choice[0], min_tau=MIN_TAU)  # strict mode (BRAINSTORM #6)
        print(f"   strict (LB >= {TARGET_STRICT:.0%}): tau={cs['tau']:.2f} answered {cs['answered']:.1%} precision {cs['precision']:.1%} "
              f"(LB {cs['lower']:.3f}, n={cs['n']})")
        out["tau_strict"][k] = cs["tau"]
        out["dev_oof_strict"][k] = {"answered": cs["answered"], "precision": cs["precision"], "n": cs["n"], "asked": n_q, "lower": cs["lower"]}
        out["models"][k] = mk().fit(X, y); out["tau"][k] = choice[0]
        out["dev_oof"][k] = {"answered": choice[1], "precision": choice[2], "n": choice[3], "asked": n_q}
    out["keys"] = sorted(rows[0][3])
    out["ocr"] = OCRDIR
    out["new_feats"] = NEW
    out["rule"] = "cp90" if CP else f"point{TARGET}"
    if "--nosave" not in sys.argv:
        pickle.dump(out, open(Path(__file__).resolve().parents[2] / "bp/models/cubi_rooms.pkl", "wb"))
    tot = sum(v["n"] for v in out["dev_oof"].values()); ask = sum(v["asked"] for v in out["dev_oof"].values())
    ok = sum(v["n"] * v["precision"] for v in out["dev_oof"].values())
    print(f"ALL room kinds dev OOF: answered {tot / ask:.1%}, precision {ok / max(1, tot):.1%}")
    tot = sum(v["n"] for v in out["dev_oof_strict"].values()); ok = sum(v["n"] * v["precision"] for v in out["dev_oof_strict"].values())
    print(f"ALL room kinds dev OOF, strict: answered {tot / ask:.1%}, precision {ok / max(1, tot):.1%}")


if __name__ == "__main__":
    main()
