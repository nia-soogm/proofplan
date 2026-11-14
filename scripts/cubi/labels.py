"""Dev-only: record manual seed labels from a label sheet. usage: labels.py SHEET.json cls=1,2,3 cls2=... [--rest other]"""
import json, sys
from pathlib import Path
LAB = Path(__file__).resolve().parent / "seed_labels.json"
items = json.load(open(sys.argv[1]))
lab = json.loads(LAB.read_text()) if LAB.exists() else []
have = {(p, tuple(b)) for p, b, _ in lab}
assigned = {}
for a in sys.argv[2:]:
    if a == "--rest":
        continue
    c, ids = a.split("=")
    for i in ids.split(","):
        if i:
            assigned[int(i)] = c
for i, (p, b) in enumerate(items):
    c = assigned.get(i, "other" if "--rest" in sys.argv else None)
    if c and (p, tuple(b)) not in have:
        lab.append([p, b, c])
LAB.write_text(json.dumps(lab))
import collections; print(collections.Counter(c for _, _, c in lab))
