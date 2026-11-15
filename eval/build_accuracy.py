"""Collect the honest accuracy table shown in the UI (eval/accuracy.json) from the eval result files.
'first run' = the first time the solver touched the held-out split (before any fix informed by it).
'current'  = latest held-out run; any gap vs first run means test errors informed fixes (optimistic)."""
import json, re
from pathlib import Path
E_ = Path(__file__).parent

def table(f):
    out = {}
    for l in (E_ / f).read_text().splitlines():
        m = re.match(r"(\S+(?: \(sampled\))?)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%", l)
        if m:
            out[m.group(1)] = dict(n=int(m.group(2)), acc=float(m.group(3)), answered=int(m.group(4)), acc_ans=float(m.group(5)), abstain=float(m.group(6)))
    return out

cur = table("results_test_all_current.txt")
before = table("results_test_all_before_overnight.txt")  # state before the overnight phases A-D (commit a0ffb88), full test split
first = table("results_test_300_firstrun.txt")      # swiss + ifc first run (n<=300 per family)
cubi_first = table("cubicasa/test_firstrun_phaseB.txt")  # phase B: storey gate on the CP rule (only change vs phase A), first and only test run (commit 93a0afe)
cubi_strict = table("cubicasa/test_firstrun_phaseB_strict.txt")  # phase B strict mode (CP lower bound >= 95 %), first and only test run
cubi_phaseA = table("cubicasa/test_firstrun_phaseA.txt")  # phase A (fixtures from labels, WC-seat fusion, room gate v3), first and only test run (commit 6f44962)
cubi_rooms_v2 = table("cubicasa/test_firstrun_rooms_v2.txt")  # phase 3 (room gate on OCR v2), first run, kept for the record
cubi_phase1 = table("cubicasa/test_firstrun_full.txt")      # phase 1 first run, kept for the record


def ifcb(f):
    """eval/ifcb/*.txt header line: 'acc X%  answered N (Y%)  acc|ans Z%  abstain W%'"""
    t = (E_ / f).read_text()
    n = int(re.search(r"\bn=(\d+)", t).group(1))
    m = re.search(r"acc ([\d.]+)%\s+answered (\d+) \(([\d.]+)%\)\s+acc\|ans ([\d.]+)%\s+abstain ([\d.]+)%", t)
    return dict(n=n, acc=float(m.group(1)), answered=int(m.group(2)), acc_ans=float(m.group(4)), abstain=float(m.group(5)))


ifcb_first = ifcb("ifcb/test_firstrun.txt")  # ifc_bench first (and only pre-fix) run on the full test split (477 q)  # cubicasa first (and only) run on the full test split (2,495 images)
# opening_design first test run, recorded by hand in eval/opening/NOTES.md before fixes
op_first = {"opening_design/count": dict(n=17, acc=64.7, abstain=0.0), "opening_design/text": dict(n=3, acc=33.3, abstain=33.3)}
NOTES = {
    "swiss_dwellings/area": "DXF polygons; NOT SURE when the PDF fill area differs > 1.5 % (phase B first run: 99.97 % when answering, 0.55 % abstain)", "swiss_dwellings/count": "DXF blocks/rooms",
    "ifc_models/area": "IfcSpace Qto / geometry; plan cross-check", "ifc_models/count": "IFC elements; first run 90% → fixes informed by test",
    "ifc_models/length": "", "ifc_models/volume": "",
    "opening_design/count": "rotated schedule tables; fixes informed by test", "opening_design/text": "lenient text match; n=3",
    "cubicasa5k/count": "room labels (OCR v2 incl. rotated text, FI/SV lexicon, summary) + dev-trained gate; storeys by layout split + dev-trained gate; WC seats = symbol detector fused with bathroom/WC labels; showers/baths and sinks inferred from room labels (said so in the answer); doors, windows abstain",
    "ifc_bench/text": "IfcOpenShell template answers (no LLM), strict scorer; phase E: every answer re-derived by an independent IFC code path (else NOT SURE) + template gate >= 90 % of >= 5 dev answers robust to leaving out any dev model/group + strict parser; phase C first run 13.8 % acc / 58.9 % when answering; phase E current (optimistic, test scored before) 7.3 % / 72.9 %",
}
rows = []
for fam, c in cur.items():
    if fam.startswith("ALL"):
        continue
    f = first.get(fam) if fam.split("/")[0] in ("swiss_dwellings", "ifc_models") else op_first.get(fam)
    if fam == "cubicasa5k/count":
        f = cubi_first["ALL"]
    if fam == "ifc_bench/text":
        f = ifcb_first
    rows.append({"family": fam, "n": c["n"], "acc": c["acc"], "acc_ans": c["acc_ans"], "abstain": c["abstain"],
                 "first_acc": f["acc"] if f else None, "first_n": f["n"] if f else None,
                 "before_overnight": before.get(fam), "note": NOTES.get(fam, "")})
cubi = [{"kind": k, **v, "strict": cubi_strict.get(k)} for k, v in cubi_first.items() if k != "ALL"]
ROOMS = ("balcony", "bathroom", "bedroom", "kitchen", "sauna", "wc")


def rooms(t):
    n = sum(t[k]["n"] for k in ROOMS); a = sum(t[k]["answered"] for k in ROOMS); ok = sum(t[k]["acc"] * t[k]["n"] / 100 for k in ROOMS)
    return dict(n=n, answered=a, acc=round(100 * ok / n, 1), acc_ans=round(100 * ok / a, 1), abstain=round(100 * (n - a) / n, 1))


json.dump({"split": "held-out test (by drawing / by project)", "rows": rows, "all": cur.get("ALL (sampled)"),
           "all_before_overnight": before.get("ALL (sampled)"),
           "cubicasa_kinds_before_overnight": {k: v for k, v in cubi_rooms_v2.items()},
           "cubicasa_kinds": cubi,
           "cubicasa_rooms": {"phase1_first_run": rooms(cubi_phase1), "phase3_first_run": rooms(cubi_rooms_v2), "phaseA_first_run": rooms(cubi_phaseA), "first_run": rooms(cubi_first),
                              "strict_first_run": rooms(cubi_strict)},
           "cubicasa_strict": cubi_strict.get("ALL"),
           "ifc_bench": {"first_run": ifcb_first, "phase2_current_optimistic": ifcb("ifcb/test_current.txt"), "first_run_phaseC": ifcb("ifcb/test_firstrun_C.txt"), "phaseE_current_optimistic": ifcb("ifcb/test_current_E.txt"), "current": ifcb("ifcb/test_current_E.txt"), "dev": ifcb("ifcb/dev.txt")},
           "phaseE": {"cubicasa_doors_windows": "not shipped: on dev OOF no gate threshold reached a Clopper-Pearson 95 % lower bound >= 95 % (doors 37 % right at 3.7 % answered, windows 46 % at 1.2 %); still abstain, so no test run",
                      "ifc_bench": "current (optimistic, test scored in phases 2 and C before): 7.3 % acc, 72.9 % when answering, 10.1 % answered; dev LOMO OOF 98.0 % when answering (LB 90.9 %) at 14.6 % answered"}},
          open(E_ / "accuracy.json", "w"), indent=1)
print(json.dumps(rows, indent=0)[:1500])
