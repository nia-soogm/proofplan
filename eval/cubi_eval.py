"""Per-kind accuracy/abstention for the CubiCasa baseline on the 300 on-box images (split by image: dev/test)."""
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bp.sources import cubicasa, JDATA
from eval.jeval import split_of, correct

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="dev"); ap.add_argument("--errors", type=int, default=0)
ap.add_argument("--min-labels", type=int, default=3); ap.add_argument("--min-conf", type=float, default=0.5)
ap.add_argument("--strict", action="store_true", help="strict thresholds for the dev-calibrated gates (BRAINSTORM #6)")
a = ap.parse_args()
from bp.sources import calib
calib.set_strict(a.strict)
st = collections.defaultdict(collections.Counter); errs = []
for l in open(JDATA / "questions.jsonl"):
    q = json.loads(l)
    if not q["path"].startswith("data/cubicasa5k/") or not (JDATA / q["path"]).exists() or split_of(q["path"]) != a.split:
        continue
    k = cubicasa.qkind(q["question"])
    r = cubicasa.answer(q["question"], q["path"], a.min_labels, a.min_conf)
    s = st[k]; s["n"] += 1
    if r["status"] == "abstain":
        s["abst"] += 1
    else:
        ok = correct(q, r["value"]); s["ok"] += ok; s["ans"] += 1
        if not ok:
            errs.append((k, q["path"].split("/")[-1], q["answer"], r["value"], r["answer"][:120]))
print(f"split={a.split}  mode={'strict' if a.strict else 'normal'}")
print(f"{'kind':<12}{'n':>6}{'acc':>8}{'answered':>10}{'acc|ans':>9}{'abstain':>9}")
T = collections.Counter()
for k, s in sorted(st.items(), key=lambda x: str(x[0])):
    T.update(s)
    print(f"{str(k):<12}{s['n']:>6}{s['ok']/s['n']:>8.1%}{s['ans']:>10}{(s['ok']/s['ans'] if s['ans'] else 0):>9.1%}{s['abst']/s['n']:>9.1%}")
print(f"{'ALL':<12}{T['n']:>6}{T['ok']/T['n']:>8.1%}{T['ans']:>10}{(T['ok']/T['ans'] if T['ans'] else 0):>9.1%}{T['abst']/T['n']:>9.1%}")
for e in errs[:a.errors]:
    print("ERR", *e)
