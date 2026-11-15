"""Eval on the real Junction questions.

Split: by DRAWING (md5 of the model/plan id) -> dev (tune on) / test (held out, report). A drawing's questions
never appear in both. Stratified sample: N per (source, type) family, separately from dev and test.
Scoring (official): count exact; area/length/volume within 2%; text: lenient token check (ifc_bench: strict eval.ifcb_score).
Usage: python -m eval.jeval --split test --n 300 [--out eval/results_test.json] [--strict] [--only cubicasa5k]
"""
import argparse, hashlib, json, random, sys, time, collections, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bp.jsolve import solve
from bp.sources.ifcsrc import model_of
from eval.ifcb_score import score as ifcb_score

from bp.sources import JDATA
Q = JDATA / "questions.jsonl"


def drawing_id(path):
    src = path.split("/")[1]
    return f"{src}/{model_of(path)}" if src == "ifc_models" else path.rsplit(".", 1)[0]


def split_of(path):
    if path.startswith("data/opening_design/"):  # only 3 sheets: split by PROJECT (Sports Complex = dev, Cannery Trail = test)
        return "test" if "/CTR/" in path else "dev"
    return "dev" if int(hashlib.md5(drawing_id(path).encode()).hexdigest(), 16) % 2 == 0 else "test"


def correct(q, v):
    if v is None:
        return False
    try:
        if q["type"] == "count":
            return int(round(float(v))) == int(q["answer"]) and abs(float(v) - round(float(v))) < 1e-6
        if q["type"] in ("area", "length", "volume"):
            a = float(q["answer"])
            return abs(float(v) - a) <= 0.02 * abs(a) + 1e-9
    except (TypeError, ValueError):
        return False
    return False


def text_correct(q, v):
    """Lenient free-form check (README: text answers are free-form): every code/number token of the gold
    (A1, WD-1, 19, 5'-10 3/8", ...) must appear in our answer; reported separately, never mixed with the official score."""
    if v is None:
        return False
    norm = lambda t: re.sub(r"[\s'\"]", "", str(t).upper())
    toks = re.findall(r"[A-Z]*-?\d[\w./-]*", str(q["answer"]).upper()) or str(q["answer"]).upper().split()[:1]
    nv = norm(v)
    return bool(toks) and all(norm(t) in nv for t in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    ap.add_argument("--show-errors", type=int, default=0)
    ap.add_argument("--strict", action="store_true", help="strict thresholds for the dev-calibrated gates (BRAINSTORM #6)")
    ap.add_argument("--only", help="comma-separated sources to run, e.g. cubicasa5k,swiss_dwellings")
    a = ap.parse_args()
    fam = collections.defaultdict(list)
    for l in open(Q):
        q = json.loads(l)
        src = q["path"].split("/")[1]
        if src == "cubicasa5k" and not (JDATA / q["path"]).exists():
            continue  # skip images that are not on this box
        if a.only and src not in a.only.split(","):
            continue
        if split_of(q["path"]) == a.split:
            fam[(src, q["type"])].append(q)
    rnd = random.Random(a.seed)
    rows = []
    t0 = time.time()
    for k in sorted(fam):
        sample = rnd.sample(fam[k], min(a.n, len(fam[k])))
        src = k[0]
        for q in sample:
            try:
                r = solve(q["question"], q["path"], strict=a.strict)
            except Exception as e:
                r = {"status": "error", "value": None, "answer": f"{type(e).__name__}: {e}"}
            rows.append({"family": f"{k[0]}/{k[1]}", "q": q, "status": r["status"], "value": r.get("value"), "ok": ((ifcb_score(q["answer"], r.get("value")) if src == "ifc_bench" else text_correct(q, r.get("value"))) if q["type"] == "text" else correct(q, r.get("value"))) if r["status"] not in ("abstain",) else False,
                         "answer": r["answer"][:200]})
    print(f"split={a.split}  n_per_family<={a.n}  mode={'strict' if a.strict else 'normal'}  seconds={time.time() - t0:.0f}")
    print(f"{'family':<28}{'n':>5}{'acc':>8}{'answered':>10}{'acc|ans':>9}{'abstain':>9}")
    tot = collections.Counter()
    for f in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == f]
        n = len(rs); ok = sum(r["ok"] for r in rs); ab = sum(r["status"] == "abstain" for r in rs); an = n - ab
        tot["n"] += n; tot["ok"] += ok; tot["ab"] += ab
        print(f"{f:<28}{n:>5}{ok / n:>8.1%}{an:>10}{(ok / an if an else 0):>9.1%}{ab / n:>9.1%}")
    print(f"{'ALL (sampled)':<28}{tot['n']:>5}{tot['ok'] / tot['n']:>8.1%}{tot['n'] - tot['ab']:>10}{tot['ok'] / max(1, tot['n'] - tot['ab']):>9.1%}{tot['ab'] / tot['n']:>9.1%}")
    if a.show_errors:
        errs = [r for r in rows if not r["ok"] and r["status"] != "abstain" and not r["family"].startswith(("cubicasa", "ifc_bench", "opening"))]
        for r in errs[:a.show_errors]:
            print("ERR", r["family"], r["q"]["path"].split("/")[-1][:50], "|", r["q"]["question"][:90], "| gold", r["q"]["answer"], "| got", r["value"], "|", r["answer"][:100])
        ab = [r for r in rows if r["status"] == "abstain" and not r["family"].startswith(("cubicasa", "ifc_bench", "opening"))]
        for r in ab[:a.show_errors]:
            print("ABS", r["family"], r["q"]["path"].split("/")[-1][:50], "|", r["q"]["question"][:90], "| gold", r["q"]["answer"], "|", r["answer"][:100])
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=0, default=str)


if __name__ == "__main__":
    main()
