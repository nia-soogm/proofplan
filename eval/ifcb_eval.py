"""ifc_bench free-text eval (split by model, same md5 split as eval.jeval). Scored with eval.ifcb_score (strict).
Usage: python -m eval.ifcb_eval --split dev [--show 50] [--out eval/ifcb/dev.txt]
"""
import argparse, json, sys, collections, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bp.sources import JDATA
from bp.sources.ifcbench import answer, GATED
from eval.jeval import split_of
from eval.ifcb_score import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--model")
    ap.add_argument("--out")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(JDATA / "questions.jsonl")]
    qs = [q for q in qs if q["path"].startswith("data/ifc_bench/") and split_of(q["path"]) == a.split]
    if a.model:
        qs = [q for q in qs if a.model in q["path"]]
    rows, t0 = [], time.time()
    for q in qs:
        r = answer(q["question"], q["path"])
        ok = r["status"] != "abstain" and score(q["answer"], r.get("value"))
        rows.append((q, r, ok))
    n = len(rows); ans = [x for x in rows if x[1]["status"] != "abstain"]; okn = sum(x[2] for x in rows)
    lines = [f"ifc_bench/text split={a.split} n={n} seconds={time.time() - t0:.0f}",
             f"acc {okn / n:.1%}  answered {len(ans)} ({len(ans) / n:.1%})  acc|ans {okn / max(1, len(ans)):.1%}  abstain {1 - len(ans) / n:.1%}"]
    per = collections.defaultdict(lambda: [0, 0, 0])
    for q, r, ok in rows:
        k = r.get("handler", "abstain")
        per[k][0] += 1; per[k][1] += r["status"] != "abstain"; per[k][2] += ok
    for k, (c, an, o) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"   {k:<22} n={c:<4} correct={o:<4} " + (f"({o / an:.0%} of answered)" if an else ("(gated: withheld)" if k in GATED else "")))
    print("\n".join(lines))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n")
    if a.show:
        for q, r, ok in rows:
            if r["status"] != "abstain" and not ok:
                print("=" * 100); print(q["path"].split("/")[-1], "|", q["question"]); print("GOLD:", q["answer"][:900]); print("OURS:", r["value"][:900])
                a.show -= 1
                if a.show <= 0:
                    break


if __name__ == "__main__":
    main()
