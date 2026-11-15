"""Train the fixture-from-labels gates (bp/sources/cubi_fix.py) on DEV images only.

Per (image, kind) the candidates are room-label formulas (and the WC-symbol detector count for wc_seat). A gradient-boosted
classifier learns P(candidate == gold); 5-fold out-of-fold predictions grouped by image give the per-image top candidate and
its score. tau is the max-coverage threshold whose Clopper-Pearson 95 % lower bound on precision is >= 0.90 (bp/sources/calib.py).
The detector features use the stage-1 OUT-OF-FOLD candidate probabilities (data/jcache/cubi_sym_stage1.pkl), so the gate never
sees in-sample detector scores. Compared against the shipped WC-seat detector gate under the same rule.
A kind ships (enabled) only if its dev result beats the dev target given on the command line / in TARGETS.
Usage: python scripts/cubi/train_fix.py [--report] [--nosave]
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold import gold
from bp.sources import CACHE, cubi_rooms as R, cubi_fix as F
from bp.sources.calib import choose_tau, cp_lower, TARGET_STRICT, choose_strict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[2]
# dev targets (BRAINSTORM #1-#3, restated under the lower-bound rule): ship only if coverage beats these at LB >= 90 %
TARGETS = {"shower_bath": 0.25, "sink": 0.08, "wc_seat": None}  # wc_seat: must beat the shipped detector gate under the same rule


def mk():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)


def table():
    g = gold("dev")
    S1 = pickle.load(open(CACHE / "cubi_sym_stage1.pkl", "rb"))["P_oof"]
    prm = pickle.load(open(ROOT / "bp/models/cubi_sym.pkl", "rb"))["params"]["wc"]
    rows = {k: [] for k in F.KINDS}; asked = {k: 0 for k in F.KINDS}; old = []
    for p, gk in sorted(g.items()):
        if not any(k in gk for k in F.KINDS):
            continue
        for k in F.KINDS:
            asked[k] += k in gk
        f = CACHE / "cubi_ocr2" / f"{Path(p).stem}.json"
        d = CACHE / "cubi_desc" / f"{Path(p).stem}.npz"
        if not f.exists() or not d.exists() or p not in S1:
            continue
        o = json.loads(f.read_text()); ex = R.extract(o["tokens"])
        z = np.load(d)
        det = F.det_feats(p, S1[p], z["boxes"], float(z["s"]), int(z["nlab"]), prm)
        if "wc_seat" in gk:
            old.append((p, det[2], int(det[0] == gk["wc_seat"])))
        for k in F.KINDS:
            if k not in gk:
                continue
            for v, tags, fv, ev, ud in F.candidates(k, ex, o["size"], p, det):
                rows[k].append((p, v, fv, int(v == gk[k]), int(ud), int(any(l in F.LABS[k] for _, l in ev))))
    return rows, asked, old


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
    cf = Path("/tmp/pp_fix_table.pkl")
    if cf.exists() and "--fresh" not in sys.argv:
        rows, asked, old = pickle.load(open(cf, "rb"))
    else:
        rows, asked, old = table(); pickle.dump((rows, asked, old), open(cf, "wb"))
    out = {"models": {}, "tau": {}, "tau_strict": {}, "dev_oof": {}, "dev_oof_strict": {}, "enabled": {},
           "rule": "Clopper-Pearson 95% lower bound >= 0.90 (strict: >= 0.97)"}
    # baseline: the shipped WC-seat detector gate, re-scored OOF under the same lower-bound rule
    Xo = np.array([r[1] for r in old], float); yo = np.array([r[2] for r in old]); co = np.zeros(len(yo))
    for tr, te in KFold(5, shuffle=True, random_state=1).split(Xo):
        co[te] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, random_state=0).fit(Xo[tr], yo[tr]).predict_proba(Xo[te])[:, 1]
    base = choose_tau(co, yo, asked["wc_seat"])
    s76 = co >= 0.76
    print(f"baseline wc_seat detector gate (OOF): raw exact {yo.mean():.3f}; at shipped tau 0.76: answered {s76.sum() / asked['wc_seat']:.3f} "
          f"prec {yo[s76].mean():.3f} LB {cp_lower(yo[s76].sum(), s76.sum()):.3f}; CP rule: {base}")
    TARGETS["wc_seat"] = base["answered"]
    for k in F.KINDS:
        rk = rows[k]
        X, y, pb, yb, idx = oof_top(rk)
        ch = choose_tau(pb, yb, asked[k])
        print(f"== {k}: asked {asked[k]}, with candidates {len(pb)}, top-1 acc {yb.mean():.3f} (of asked {yb.sum() / asked[k]:.3f})")
        sel = pb >= ch["tau"]
        for name, m in (("matching room label(s) in evidence", lambda r: r[5]), ("detector count", lambda r: r[4]),
                        ("prior only (no matching label, no detector)", lambda r: not r[5] and not r[4])):
            mm = np.array([bool(m(rk[i])) for i in idx]) & sel
            if mm.sum():
                print(f"   answered via {name}: {mm.sum()} ({yb[mm].mean():.1%} right)")
        if "--report" in sys.argv:
            for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
                s = pb >= t
                if s.sum():
                    print(f"   tau {t:.2f}: answered {s.sum() / asked[k]:.3f} prec {yb[s].mean():.3f} n={s.sum()} LB {cp_lower(yb[s].sum(), s.sum()):.3f}")
        ok = ch["answered"] > TARGETS[k]
        print(f"   CP rule: tau={ch['tau']} answered {ch['answered']:.1%} precision {ch['precision']:.1%} (LB {ch['lower']:.1%}, n={ch['n']}); "
              f"target answered > {TARGETS[k]:.1%} -> {'SHIP' if ok else 'DISABLED'}")
        cs = choose_strict(pb, yb, asked[k], ch["tau"])  # strict mode (BRAINSTORM #6)
        print(f"   strict (LB >= {TARGET_STRICT:.0%}): tau={cs['tau']} answered {cs['answered']:.1%} precision {cs['precision']:.1%} (LB {cs['lower']:.1%}, n={cs['n']})")
        out["tau_strict"][k] = cs["tau"]; out["dev_oof_strict"][k] = dict(cs, asked=asked[k])
        out["models"][k] = mk().fit(X, y); out["tau"][k] = ch["tau"]; out["dev_oof"][k] = dict(ch, asked=asked[k], target=TARGETS[k])
        out["enabled"][k] = bool(ok)
    out["baseline_wc_seat"] = base
    if "--nosave" not in sys.argv:
        pickle.dump(out, open(ROOT / "bp/models/cubi_fix.pkl", "wb"))
        print("saved bp/models/cubi_fix.pkl", {k: v for k, v in out["enabled"].items()})


if __name__ == "__main__":
    main()
