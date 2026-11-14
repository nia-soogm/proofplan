"""Deterministic tools over a project's sheet graph. Every tool returns
{"result": <json>, "evidence": [{sheet, page, bbox, text, kind}]} so answers can cite real coordinates."""
from __future__ import annotations
import json, re, collections, functools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "data" / "projects"

LEVEL_WORDS = [
    (r"\b(basement|cellar|level\s*-?\s*b|kellari)\b", "B"),
    (r"\b(main|first|1st|ground|level\s*1|l1|floor\s*1|1\.?\s*krs|ensimm\w*)\b", "1"),
    (r"\b(second|2nd|upper|level\s*2|l2|floor\s*2|2\.?\s*krs|toi\w*)\b", "2"),
    (r"\b(third|3rd|level\s*3|l3|floor\s*3|3\.?\s*krs)\b", "3"),
]
ATTR_ALIASES = {
    "fire": ["FIRE", "RATING", "LABEL"], "rating": ["FIRE", "RATING", "LABEL"],
    "width": ["WIDTH"], "wide": ["WIDTH"], "height": ["HEIGHT"], "tall": ["HEIGHT"],
    "hardware": ["HDWE", "HDW", "HARDWARE"], "thick": ["THICK"], "material": ["MATERIAL", "MATL", "MAT"],
    "frame": ["FRAME"], "type": ["TYPE"], "qty": ["QTY", "QUANTITY"], "quantity": ["QTY"],
    "description": ["DESCRIPTION", "DESC"], "floor": ["FLOOR", "LEVEL"], "comment": ["COMMENT", "REMARK", "NOTE"],
    "remark": ["REMARK", "COMMENT"], "glass": ["LITE", "GLAZ"], "lite": ["LITE"],
}


def parse_level(text):
    if text is None:
        return None
    t = str(text).strip().lower()
    if t in {"1", "2", "3", "b"}:
        return t.upper()
    for rx, lv in LEVEL_WORDS:
        if re.search(rx, t):
            return lv
    return None


def rating_minutes(s):
    """'20 MIN.' -> 20, 'EI60' -> 60, '1 HR' -> 60, '90' -> 90, '' -> None."""
    if not s:
        return None
    t = str(s).upper().replace(" ", "")
    m = re.search(r"(?:EI|E|B|A)?(\d+(?:\.\d+)?)(MIN|M|HR|H|HOUR)?", t)
    if not m:
        return None
    v = float(m.group(1))
    if m.group(2) in ("HR", "H", "HOUR") or (v <= 3 and m.group(2) is None and "HR" in t):
        v *= 60
    return int(v)


def norm_mark(s):
    return re.sub(r"[\s\-_.]", "", str(s)).upper()


@functools.lru_cache(maxsize=8)
def _load(name):
    return json.loads((PROJECTS / name / "graph.json").read_text())


def list_projects():
    return sorted(p.name for p in PROJECTS.iterdir() if (p / "graph.json").exists())


class Project:
    def __init__(self, name):
        self.name = name
        self.g = _load(name)
        self.sheets = {s["page"]: s for s in self.g["sheets"]}
        ids = [s["sheet_id"] for s in self.g["sheets"]]
        self.has_disciplines = sum(bool(re.match(r"^[A-Z]{1,2}-?\d", i)) for i in ids) > len(ids) / 2
        self.rows = []  # (schedule, row)
        for sc in self.g["schedules"]:
            for r in sc["rows"]:
                self.rows.append((sc, r))
        self.by_mark = collections.defaultdict(list)
        for sc, r in self.rows:
            self.by_mark[norm_mark(r["mark"])].append((sc, r))

    # ---------- helpers
    def ev(self, page, bbox, text, kind, **extra):
        s = self.sheets[page]
        e = {"sheet": s["sheet_id"], "page": page, "bbox": bbox, "text": text, "kind": kind}
        e.update(extra)
        return e

    def is_plan(self, s):
        if not s["level"]:
            return False
        if self.has_disciplines:
            sid = s["sheet_id"].upper()
            return sid.startswith("A") and not sid.startswith("AD")
        return True

    def find_col(self, sc, attr):
        attr = attr.lower()
        keys = ATTR_ALIASES.get(attr) or ATTR_ALIASES.get(attr.rstrip("s")) or [attr.upper()]
        for k in keys:
            for c in sc["columns"]:
                if k in c.upper():
                    return c
        return None

    def sheet_lookup(self, sheet):
        if sheet is None:
            return None
        sheet = str(sheet).strip().upper()
        for s in self.g["sheets"]:
            if s["sheet_id"].upper() == sheet or norm_mark(s["sheet_id"]) == norm_mark(sheet):
                return s
        return None

    # ---------- tools
    def list_sheets(self):
        res = [{"sheet": s["sheet_id"], "page": s["page"], "title": s["title"], "level": s["level"],
                "is_plan": self.is_plan(s), "n_tags": sum(1 for t in self.g["tags"] if t["page"] == s["page"])}
               for s in self.g["sheets"]]
        sch = [{"title": sc["title"], "sheet": self.sheets[sc["page"]]["sheet_id"], "columns": sc["columns"],
                "n_rows": len(sc["rows"])} for sc in self.g["schedules"]]
        return {"result": {"sheets": res, "door_schedules": sch}, "evidence": []}

    def find_text(self, query, sheet=None, limit=20):
        q = query.lower().strip()
        out, ev = [], []
        target = self.sheet_lookup(sheet) if sheet else None
        for s in self.g["sheets"]:
            if target and s["page"] != target["page"]:
                continue
            for l in s["lines"]:
                if q in l["text"].lower():
                    out.append({"sheet": s["sheet_id"], "text": l["text"]})
                    ev.append(self.ev(s["page"], l["bbox"], l["text"], "text"))
                    if len(ev) >= limit:
                        return {"result": {"matches": out, "truncated": True}, "evidence": ev}
        return {"result": {"matches": out, "truncated": False}, "evidence": ev}

    def tag_locations(self, mark, plans_only=True):
        m = norm_mark(mark)
        return [t for t in self.g["tags"] if norm_mark(t["mark"]) == m
                and (not plans_only or self.is_plan(self.sheets[t["page"]]))]

    def schedule_row(self, tag, attribute=None):
        hits = self.by_mark.get(norm_mark(tag), [])
        if not hits:
            near = [r["mark"] for _, r in self.rows if norm_mark(tag) in norm_mark(r["mark"])][:5]
            return {"result": {"found": False, "tag": tag, "similar_marks": near}, "evidence": []}
        ev, rows = [], []
        for sc, r in hits:
            col = self.find_col(sc, attribute) if attribute else None
            item = {"schedule": sc["title"], "sheet": self.sheets[sc["page"]]["sheet_id"], "cells": r["cells"]}
            if attribute:
                item["attribute_column"] = col
                item["attribute_value"] = r["cells"].get(col) if col else None
            rows.append(item)
            if col and col in r["cell_bboxes"]:
                ev.append(self.ev(sc["page"], r["cell_bboxes"][col], f"{col}: {r['cells'][col]}", "schedule_cell", mark=r["mark"]))
            ev.append(self.ev(sc["page"], r["bbox"], " | ".join(f"{k}={v}" for k, v in r["cells"].items() if v),
                              "schedule_row", mark=r["mark"]))
        locs = self.tag_locations(tag)
        for t in locs:
            ev.append(self.ev(t["page"], t["bbox"], f"tag {t['mark']} on {t['sheet_id']}", "tag", mark=t["mark"]))
        values = {it.get("attribute_value") for it in rows if attribute}
        return {"result": {"found": True, "rows": rows, "plan_locations": [{"sheet": t["sheet_id"], "level": t["level"]} for t in locs],
                           "conflict": attribute is not None and len(values) > 1},
                "evidence": ev}

    def _row_matches(self, sc, r, where):
        for attr, want in (where or {}).items():
            if attr in ("*", "any"):
                w = str(want).lower()
                if not any(w in str(v).lower() for v in r["cells"].values()):
                    return False
                continue
            col = self.find_col(sc, attr)
            if not col:
                return False
            have = r["cells"].get(col, "")
            if "fire" in attr.lower() or "rating" in attr.lower():
                hm, wm = rating_minutes(have), rating_minutes(want)
                if str(want).lower() in ("any", "rated", "yes"):
                    if not hm:
                        return False
                elif hm != wm:
                    return False
            elif str(want).lower() not in str(have).lower():
                return False
        return True

    def count_tags(self, level=None, sheet=None, mark=None, where=None):
        """Count door tags on plan sheets, optionally filtered by level / sheet / mark / schedule attributes.
        Cross-checks against the schedule (QTY or FLOOR columns) and reports conflicts."""
        lv = parse_level(level) if level else None
        target = self.sheet_lookup(sheet) if sheet else None
        marks = set()
        matched_rows = []
        for sc, r in self.rows:
            if mark and norm_mark(r["mark"]) != norm_mark(mark) and not re.fullmatch(str(mark).replace("*", ".*"), r["mark"], re.I):
                continue
            if self._row_matches(sc, r, where):
                marks.add(norm_mark(r["mark"]))
                matched_rows.append((sc, r))
        tags = []
        for t in self.g["tags"]:
            s = self.sheets[t["page"]]
            if norm_mark(t["mark"]) not in marks:
                continue
            if target:
                if t["page"] != target["page"]:
                    continue
            elif not self.is_plan(s):
                continue
            if lv and s["level"] != lv:
                continue
            tags.append(t)
        per = collections.Counter(t["mark"] for t in tags)
        ev = [self.ev(t["page"], t["bbox"], f"tag {t['mark']} on {t['sheet_id']}", "tag", mark=t["mark"]) for t in tags]
        conflicts, sched_expected = [], 0
        has_qty = False
        for sc, r in matched_rows:
            if where and r["mark"] in per:
                attrs = [a for a in where if a not in ("*", "any")] or ["description", "type"]
                for attr in attrs[:1]:
                    col = self.find_col(sc, attr)
                    if col and col in r["cell_bboxes"]:
                        ev.append(self.ev(sc["page"], r["cell_bboxes"][col], f"{r['mark']} {col}: {r['cells'][col]}", "schedule_cell", mark=r["mark"]))
            qcol, fcol = self.find_col(sc, "qty"), self.find_col(sc, "floor")
            rl = parse_level(r["cells"].get(fcol)) if fcol else None
            if lv and fcol and rl and rl != lv:
                continue
            if lv and not fcol:
                continue  # schedule has no floor column: cannot say what this level should contain
            exp = None
            if qcol and re.fullmatch(r"\d+", r["cells"].get(qcol, "").strip()):
                has_qty = True
                exp = int(r["cells"][qcol])
            elif not target:
                exp = 1  # one row per opening is the common convention
            if exp is not None:
                sched_expected += exp
                got = per.get(r["mark"], 0)
                if got != exp and not target:
                    conflicts.append({"mark": r["mark"], "schedule_expects": exp, "found_on_plans": got})
                    ev.append(self.ev(sc["page"], r["bbox"], f"schedule row {r['mark']} (expects {exp}, plans show {got})",
                                      "schedule_row", mark=r["mark"], conflict=True))
        res = {"count": len(tags), "per_mark": dict(sorted(per.items())),
               "sheets": sorted({t["sheet_id"] for t in tags}), "filters": {"level": lv, "sheet": sheet, "mark": mark, "where": where},
               "schedule_rows_matching": len(matched_rows),
               "schedule_expected_count": sched_expected if not target else None,
               "schedule_has_qty_column": has_qty, "conflicts": conflicts}
        if where and not matched_rows:
            res["note"] = "no schedule rows match the attribute filter (check column names via list_sheets)"
        return {"result": res, "evidence": ev}

    def follow_callout(self, ref):
        m = re.match(r"^\s*(\w+)\s*/\s*([A-Z]{1,2}-?\d+(?:\.\d+)?)\s*$", ref.upper())
        if not m:
            return {"result": {"found": False, "error": "expected DETAIL/SHEET like 3/A-601"}, "evidence": []}
        det, sh = m.groups()
        s = self.sheet_lookup(sh)
        if not s:
            return {"result": {"found": False, "error": f"sheet {sh} not in set"}, "evidence": []}
        cands = [w for w in s["words"] if w["text"] == det]
        if not cands:
            return {"result": {"found": False, "sheet": s["sheet_id"]}, "evidence": [self.ev(s["page"], [0, 0, s["width"], s["height"]], s["title"] or s["sheet_id"], "sheet")]}
        w = max(cands, key=lambda w: w["size"])
        title = None
        for l in s["lines"]:  # detail title = nearest line right of the bubble
            if 0 <= l["bbox"][0] - w["bbox"][2] < 60 and abs(l["bbox"][1] - w["bbox"][1]) < 25 and re.search("[A-Z]{3}", l["text"]) and not l["text"].upper().startswith("SCALE"):
                title = l["text"]
                break
        bb = [w["bbox"][0] - 4, w["bbox"][1] - 4, w["bbox"][2] + 4, w["bbox"][3] + 4]
        return {"result": {"found": True, "sheet": s["sheet_id"], "detail": det, "detail_title": title},
                "evidence": [self.ev(s["page"], bb, f"detail {det}/{s['sheet_id']}" + (f": {title}" if title else ""), "callout_target")]}

    def consistency_report(self):
        """Schedule vs plan: marks never tagged, QTY mismatches, schedule floor vs tag level."""
        issues, ev = [], []
        for sc, r in self.rows:
            locs = self.tag_locations(r["mark"])
            qcol, fcol = self.find_col(sc, "qty"), self.find_col(sc, "floor")
            q = r["cells"].get(qcol, "") if qcol else ""
            exp = int(q) if re.fullmatch(r"\d+", q.strip() or "x") else 1
            if len(locs) != exp:
                issues.append({"mark": r["mark"], "issue": "count_mismatch", "schedule": exp, "plans": len(locs)})
                ev.append(self.ev(sc["page"], r["bbox"], f"{r['mark']}: schedule {exp} vs plans {len(locs)}", "schedule_row", mark=r["mark"], conflict=True))
            if fcol:
                rl = parse_level(r["cells"].get(fcol))
                bad = [t for t in locs if rl and t["level"] != rl]
                for t in bad:
                    issues.append({"mark": r["mark"], "issue": "level_mismatch", "schedule_floor": rl, "tag_level": t["level"], "sheet": t["sheet_id"]})
                    ev.append(self.ev(t["page"], t["bbox"], f"{r['mark']} tagged on level {t['level']} but schedule says {rl}", "tag", conflict=True))
        if not issues:
            ev = [self.ev(sc["page"], sc["bbox"], f"{sc['title']} ({len(sc['rows'])} rows)", "schedule") for sc in self.g["schedules"]]
        return {"result": {"n_schedule_rows": len(self.rows), "issues": issues, "consistent": not issues}, "evidence": ev}


TOOL_SPECS = [
    {"name": "list_sheets", "description": "List sheets (id, title, level, whether it is a floor plan) and the door schedules with their column names.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "find_text", "description": "Full-text search over all text on the drawings; returns matching lines with locations.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "sheet": {"type": "string"}}, "required": ["query"]}},
    {"name": "schedule_row", "description": "Look up a door tag/mark in the door schedule; returns all columns, the row location, and where the tag appears on plans. Pass attribute (e.g. 'fire rating', 'width', 'hardware') to get one value.",
     "parameters": {"type": "object", "properties": {"tag": {"type": "string"}, "attribute": {"type": "string"}}, "required": ["tag"]}},
    {"name": "count_tags", "description": "Count door tags on floor plans. Filters: level ('1','2','main','second'...), sheet id, mark (supports * wildcard), where = schedule attribute filter like {\"fire rating\": \"20 min\"} or {\"description\": \"pocket\"}. Cross-checks with the schedule and returns conflicts.",
     "parameters": {"type": "object", "properties": {"level": {"type": "string"}, "sheet": {"type": "string"}, "mark": {"type": "string"},
                                                     "where": {"type": "object", "additionalProperties": {"type": "string"}}}}},
    {"name": "follow_callout", "description": "Resolve a detail callout like '3/A-601' to the detail on its target sheet.",
     "parameters": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}},
    {"name": "consistency_report", "description": "Check door schedule vs floor plans for count and level mismatches.",
     "parameters": {"type": "object", "properties": {}}},
]


def call_tool(project: Project, name: str, args: dict):
    fn = getattr(project, name, None)
    if fn is None or name.startswith("_") or name not in {t["name"] for t in TOOL_SPECS}:
        return {"result": {"error": f"unknown tool {name}"}, "evidence": []}
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return {"result": {"error": f"bad arguments: {e}"}, "evidence": []}
