"""ifc_bench leave-one-model-out (LOMO) template gate (BRAINSTORM #7). Dev split only.

1. Answer every dev question with the gate off (the strict parser stays on) and score it (eval.ifcb_score).
2. Release rule for a template, on a set of dev models (phase E, raised from >= 75 % of >= 3): >= 90 % of its answers match
   the reference, over >= 5 answers, and the same holds with any single dev model AND any single near-duplicate group
   (fantasy_office 1-3, sixty5) left out. Answers are counted after the phase E self-checks (bp.sources.ifcbench.self_check),
   so a template with no self-check has no released dev answers and is withheld.
3. Out-of-fold estimate: for each dev model, the rule is fitted on the other 10 models and applied to the held-out one.
   The 3 fantasy_office models and the 3 sixty5 models are near-duplicates, so a stricter leave-one-group-out estimate is
   reported too.
4. The rule fitted on all 11 dev models is written to bp/sources/ifcb_gate.json and used by bp.sources.ifcbench.answer.
Usage: python -m eval.ifcb_lomo [--write]
"""
import argparse, json, os, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["IFCB_NO_GATE"] = "1"
from bp.sources import JDATA
from bp.sources import ifcbench
from eval.jeval import split_of
from eval.ifcb_score import score
from bp.sources.calib import cp_lower

PREC, MIN_N = 0.90, 5  # phase C was 0.75, 3
GATE_FILE = Path(ifcbench.__file__).with_name("ifcb_gate.json")


def group(m):
    return "fantasy_office" if m.startswith("fantasy_office") else "sixty5" if m.startswith("sixty5") else m


def stats(rows, h, keep):
    c = n = 0
    for r in rows:
        if r["h"] == h and keep(r):
            n += 1; c += r["ok"]
    return c, n


def release(rows, h, unit):
    """(release?, correct, answered, why) for template h on rows; unit(r) gives the fold key for the robustness check."""
    c, n = stats(rows, h, lambda r: True)
    if n < MIN_N or c / n < PREC:
        return False, c, n, f"{c}/{n} < {PREC:.0%} of >= {MIN_N}" if n >= MIN_N else f"only {n} dev answers"
    for un in (unit, lambda r: group(unit(r))):
        for u in sorted({un(r) for r in rows if r["h"] == h}):
            cc, nn = stats(rows, h, lambda r: un(r) != u)
            if nn < MIN_N or cc / nn < PREC:
                return False, c, n, f"fails without {u} ({cc}/{nn})"
    return True, c, n, f"{c}/{n}, >= {PREC:.0%} with any one dev model or model group left out"


def oof(rows, fold, unit):
    """Out-of-fold: rule fitted without the fold, applied to it. Returns (correct, answered, n)."""
    C = A = 0
    hs = {r["h"] for r in rows if r["h"]}
    for f in sorted({fold(r) for r in rows}):
        tr = [r for r in rows if fold(r) != f]
        rel = {h for h in hs if release(tr, h, unit)[0]}
        for r in rows:
            if fold(r) == f and r["h"] in rel:
                A += 1; C += r["ok"]
    return C, A, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="eval/ifcb/lomo_dev.txt")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(JDATA / "questions.jsonl")]
    qs = [q for q in qs if q["path"].startswith("data/ifc_bench/") and split_of(q["path"]) == "dev"]
    rows = []
    for q in qs:
        r = ifcbench.answer(q["question"], q["path"])
        ans = r["status"] != "abstain"
        rows.append({"model": Path(q["path"]).stem, "h": r.get("handler") if ans else None, "sc": r.get("handler") if r.get("selfcheck") else None,
                     "ok": bool(ans and score(q["answer"], r.get("value")))})
    model = lambda r: r["model"]  # noqa: E731
    grp = lambda r: group(r["model"])  # noqa: E731
    L = [f"ifc_bench LOMO gate (dev only, n={len(rows)}; rule: >= {PREC:.0%} of >= {MIN_N} answers, robust to dropping any one model or model group; "
         f"answers counted after the self-checks)"]
    c, n_ans = sum(r["ok"] for r in rows), sum(1 for r in rows if r["h"])
    L.append(f"no gate (strict parser on): acc {c / len(rows):.1%}  answered {n_ans} ({n_ans / len(rows):.1%})  acc|ans {c / max(1, n_ans):.1%}")
    for name, fold, unit in (("leave-one-model-out", model, model), ("leave-one-group-out", grp, grp)):
        C, A, N = oof(rows, fold, unit)
        L.append(f"OOF {name:<20}: acc {C / N:.1%}  answered {A} ({A / N:.1%})  acc|ans {C / max(1, A):.1%}  (CP 95% lower bound {cp_lower(C, A):.1%})")
    hs = sorted({h.__name__ for _, h in ifcbench.ROUTES if h is not None} | {r["h"] for r in rows if r["h"]})
    gate = {}
    L.append("")
    L.append("template gate fitted on all 11 dev models:")
    for h in sorted(hs, key=lambda h: (-stats(rows, h, lambda r: True)[1], h)):
        rel, cc, nn, why = release(rows, h, model)
        gate[h] = {"release": rel, "correct": cc, "answered": nn, "why": why}
        nsc = sum(1 for r in rows if r["sc"] == h)
        if not rel and h not in ifcbench.SELF_CHECKS:
            why = "no self-check (not re-derivable independently)"
        if nn or nsc:
            L.append(f"   {'RELEASE ' if rel else 'withhold'} {h:<22} {why}" + (f"; self-check stopped {nsc} dev draft(s)" if nsc else ""))
        gate[h]["why"] = why
    rel_n = sum(1 for r in rows if r["h"] and gate[r["h"]]["release"])
    rel_c = sum(1 for r in rows if r["h"] and gate[r["h"]]["release"] and r["ok"])
    L.append(f"in-sample with gate: acc {rel_c / len(rows):.1%}  answered {rel_n} ({rel_n / len(rows):.1%})  acc|ans {rel_c / max(1, rel_n):.1%}")
    print("\n".join(L))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    if a.write:
        GATE_FILE.write_text(json.dumps({"rule": f">= {PREC:.0%} of >= {MIN_N} dev answers (after self-checks), robust to leaving out any one dev model or model group",
                                         "fitted_on": "dev split, 11 models", "templates": gate}, indent=1) + "\n")
        print("wrote", GATE_FILE)


if __name__ == "__main__":
    main()
