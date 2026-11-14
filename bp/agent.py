"""Question -> tool calls -> answer with numbered citations [n] to parsed-PDF evidence.

Engines:
  * LLM tool-calling loop (Anthropic / OpenAI / xAI) if an API key is in the env.
  * Rule-based router fallback (no key needed) so the demo always runs.
Both share the same tools and the same abstain / conflict policy.
"""
from __future__ import annotations
import json, os, re
import requests
from .tools import Project, TOOL_SPECS, call_tool, parse_level, rating_minutes, norm_mark

SYSTEM = """You answer questions about a construction drawing set using ONLY the provided tools.
Rules:
- Every factual claim must cite evidence refs like [3] taken from tool results (field evidence_refs). Never invent refs.
- Counts come from count_tags; attributes of a specific door come from schedule_row. Do not estimate.
- If tools return no evidence, or results conflict (e.g. schedule quantity vs plan tag count, or two schedule rows disagree),
  say so explicitly. Start the answer with 'NOT SURE:' when evidence is missing, or 'CONFLICT:' when sources disagree.
- Be brief: one or two sentences, then the citations. Answer in the language of the question."""

STOP = set("""a an the of on in at for to is are was were how many what which where door doors floor level plan plans sheet
there do does have has with and or me show find count number total all any rated rating fire minute minutes min hour hours
main first second third ground upper 1st 2nd 3rd basement this that give list tell about please type""".split())
MATERIAL_SYNONYMS = {"wood": "WD", "wooden": "WD", "hollow metal": "HM", "steel": "HM", "aluminum": "Alum", "aluminium": "Alum",
                     "glass": "Glass", "frp": "FRP"}


class Evidence:
    def __init__(self):
        self.items = []

    def add(self, evs):
        refs = []
        for e in evs:
            e = dict(e)
            e["ref"] = len(self.items) + 1
            self.items.append(e)
            refs.append(e["ref"])
        return refs


def refs_str(refs, cap=40):
    r = "".join(f"[{n}]" for n in refs[:cap])
    return r + (f"…(+{len(refs) - cap})" if len(refs) > cap else "")


def finalize(answer, ev: Evidence, engine, trace, status=None):
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(ev.items)})
    if status is None:
        up = answer.strip().upper()
        status = "abstain" if up.startswith("NOT SURE") else "conflict" if up.startswith("CONFLICT") else "answered"
        if status == "answered" and not cited:
            status = "abstain"
            answer = "NOT SURE: the model gave an answer without citing drawing evidence, so it is withheld. (" + answer[:200] + ")"
    # renumber citations 1..k in order of first appearance, so the UI shows compact numbers
    order = []
    for n in re.findall(r"\[(\d+)\]", answer):
        n = int(n)
        if 0 < n <= len(ev.items) and n not in order:
            order.append(n)
    mp = {old: i + 1 for i, old in enumerate(order)}
    answer = re.sub(r"\[(\d+)\]", lambda m: f"[{mp[int(m.group(1))]}]" if int(m.group(1)) in mp else "", answer)
    citations = []
    for old in order:
        e = dict(ev.items[old - 1])
        e["ref"] = mp[old]
        citations.append(e)
    return {"answer": answer, "status": status, "citations": citations, "engine": engine, "trace": trace}


# ---------------------------------------------------------------- rule-based router
def find_mark(q, proj: Project):
    marks = {norm_mark(r["mark"]): r["mark"] for _, r in proj.rows}
    # explicit alphanumeric tags first: D-214, D02, 101A
    for tok in re.findall(r"\b[A-Za-z]{1,3}[-\s]?\d{1,4}[A-Za-z]?\b", q):
        if norm_mark(tok) in marks and not re.match(r"(?i)^ei", tok):
            return marks[norm_mark(tok)]
    m = re.search(r"(?i)\b(?:door|ovi|mark|tag)\s*(?:no\.?|number|#|nro)?\s*([A-Za-z]{0,3}-?\d{1,4}[A-Za-z]?)\b", q)
    if m:
        return marks.get(norm_mark(m.group(1)), m.group(1))
    return None


def parse_rating(q):
    m = re.search(r"(?i)\b(EI|E)\s?-?(\d{2,3})\b", q) or re.search(r"(?i)\b(\d{1,3}(?:\.\d)?)\s*-?\s*(min|minute|minutes|hr|hour|hours)\b", q)
    if m:
        return rating_minutes("".join(g for g in m.groups() if g))
    if re.search(r"(?i)fire[- ]?rated|rated door|palo", q):
        return "any"
    return None


def rules_answer(question, proj: Project):
    q = question.strip()
    ev, trace = Evidence(), []

    def run(name, **args):
        out = call_tool(proj, name, args)
        refs = ev.add(out["evidence"])
        trace.append({"tool": name, "args": args, "result": out["result"], "refs": refs})
        return out["result"], refs

    ql = q.lower()
    if re.search(r"consisten|mismatch|conflict|disagree|discrepan|ristiriit", ql):
        res, refs = run("consistency_report")
        if res["consistent"]:
            return finalize(f"No conflicts found: all {res['n_schedule_rows']} door-schedule rows {refs_str(refs)} match the plan tags (count and level).",
                            ev, "rules", trace, status="answered")
        txt = "; ".join(f"{i['mark']}: {i['issue']}" for i in res["issues"][:10])
        return finalize(f"CONFLICT: {len(res['issues'])} schedule/plan disagreements — {txt} {refs_str(refs)}", ev, "rules", trace)

    m = re.search(r"\b(\d{1,2}[A-Z]?)\s*/\s*([A-Z]{1,2}-?\d{1,3}(?:\.\d+)?)\b", q.upper())
    if m and re.search(r"detail|callout|refer|see|what is", ql):
        res, refs = run("follow_callout", ref=f"{m.group(1)}/{m.group(2)}")
        if not res.get("found"):
            return finalize(f"NOT SURE: could not resolve callout {m.group(0)}.", ev, "rules", trace, status="abstain")
        return finalize(f"Callout {m.group(0)} points to detail {res['detail']} on sheet {res['sheet']}"
                        + (f" — '{res['detail_title']}'" if res.get("detail_title") else "") + f" {refs_str(refs)}.", ev, "rules", trace)

    is_count = re.search(r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal\b|kuinka monta|montako", ql)
    mark = find_mark(q, proj)
    level = parse_level(q)
    if is_count:
        where = {}
        r = parse_rating(q)
        if r is not None:
            where["fire rating"] = "any" if r == "any" else f"{r} min"
        for k, v in MATERIAL_SYNONYMS.items():
            if re.search(rf"\b{k}\b", ql):
                where["*"] = v
        if "*" not in where:
            vocab = {}
            for _, row in proj.rows:
                for col, val in row["cells"].items():
                    if re.search("DESC|TYPE|MATERIAL", col, re.I):
                        for w in re.findall(r"[a-z]{4,}", str(val).lower()):
                            vocab[w] = w
            for w in re.findall(r"[a-z]{4,}", ql):
                if w not in STOP and (w in vocab or w.rstrip("s") in vocab):
                    where["*"] = w if w in vocab else w.rstrip("s")
                    break
        res, refs = run("count_tags", level=level, mark=None if mark is None else mark, where=where or None)
        tag_refs = refs[:res["count"]]
        other = refs[res["count"]:]
        desc = []
        if "fire rating" in where:
            desc.append("fire-rated" if where["fire rating"] == "any" else f"{where['fire rating']} fire-rated")
        if "*" in where:
            desc.append(f"'{where['*']}'")
        what = (" ".join(desc) + " doors").strip() if desc else (f"{mark} doors" if mark else "doors")
        where_txt = f" on level {level}" if level else " on the floor plans"
        if where and res["schedule_rows_matching"] == 0:
            return finalize(f"NOT SURE: no door-schedule row matches {what.replace(' doors', '')}, so I can't tell which tags qualify. "
                            f"Check the schedule columns (e.g. fire rating) for this project.", ev, "rules", trace, status="abstain")
        if res["count"] == 0 and not proj.rows:
            return finalize("NOT SURE: no door schedule or door tags were parsed in this set.", ev, "rules", trace, status="abstain")
        per = ", ".join(f"{k}×{v}" for k, v in sorted(res["per_mark"].items(), key=lambda kv: (len(kv[0]), kv[0])))
        ans = f"{res['count']} {what}{where_txt} (sheet{'s' if len(res['sheets']) > 1 else ''} {', '.join(res['sheets']) or '—'}): {per} {refs_str(tag_refs)}."
        if res["conflicts"]:
            c = "; ".join(f"{x['mark']} schedule {x['schedule_expects']} vs plans {x['found_on_plans']}" for x in res["conflicts"])
            crefs = [n for n in other if ev.items[n - 1].get("conflict")]
            ans = f"CONFLICT: plans show {ans} But the schedule disagrees — {c} {refs_str(crefs)}."
        elif res["schedule_expected_count"]:
            srefs = [n for n in other if ev.items[n - 1]["kind"] == "schedule_cell"][:40]
            basis = "QTY column" if res["schedule_has_qty_column"] else "one row per door"
            ans += f" Cross-check: the door schedule ({basis}) also gives {res['schedule_expected_count']}" + (f" {refs_str(srefs)}" if srefs else "") + "."
        elif where:
            srefs = [n for n in other if ev.items[n - 1]["kind"] == "schedule_cell"][:40]
            ans += f" Qualifying marks from the schedule {refs_str(srefs)}."
        return finalize(ans, ev, "rules", trace)

    if mark:
        attr = None
        for k in ["fire", "rating", "width", "wide", "height", "tall", "hardware", "thick", "material", "frame", "qty", "quantity",
                  "description", "floor", "comment", "remark", "glass", "lite", "type"]:
            if re.search(rf"\b{k}", ql):
                attr = k
                break
        res, refs = run("schedule_row", tag=mark, attribute=attr)
        if not res.get("found"):
            sim = f" Similar marks: {', '.join(res['similar_marks'])}." if res.get("similar_marks") else ""
            return finalize(f"NOT SURE: door {mark} is not in the door schedule of this set.{sim}", ev, "rules", trace, status="abstain")
        row = res["rows"][0]
        kinds = [ev.items[n - 1]["kind"] for n in refs]
        cell = [n for n, k in zip(refs, kinds) if k == "schedule_cell"]
        rowr = [n for n, k in zip(refs, kinds) if k == "schedule_row"]
        tagr = [n for n, k in zip(refs, kinds) if k == "tag"]
        loc = (f" Tagged on sheet {', '.join(sorted({l['sheet'] for l in res['plan_locations']}))} {refs_str(tagr)}." if tagr
               else " (Not found as a tag on any floor plan.)")
        if res.get("conflict"):
            vals = "; ".join(f"{r['attribute_value']} ({r['schedule']})" for r in res["rows"])
            return finalize(f"CONFLICT: schedules disagree on {attr} for {mark}: {vals} {refs_str(cell + rowr)}.", ev, "rules", trace)
        if attr and row.get("attribute_column"):
            v = row["attribute_value"]
            if v:
                return finalize(f"Door {mark} — {row['attribute_column']}: {v} {refs_str(cell)} (schedule row {refs_str(rowr)}, sheet {row['sheet']}).{loc}", ev, "rules", trace)
            return finalize(f"The {row['attribute_column']} cell for door {mark} is blank in the schedule {refs_str(rowr)} — no value is given.{loc}",
                            ev, "rules", trace)
        if attr and not row.get("attribute_column"):
            return finalize(f"NOT SURE: the schedule has no '{attr}' column. Row for {mark}: " +
                            ", ".join(f"{k}={v}" for k, v in row["cells"].items() if v) + f" {refs_str(rowr)}.", ev, "rules", trace, status="abstain")
        return finalize(f"Door {mark}: " + ", ".join(f"{k}={v}" for k, v in row["cells"].items() if v) + f" {refs_str(rowr)}.{loc}", ev, "rules", trace)

    m = re.search(r'"([^"]+)"|\b(?:where|find|show|locate)\b(?: is| are)?(?: the)? (.+?)\??$', q, re.I)
    if m:
        term = (m.group(1) or m.group(2)).strip()
        res, refs = run("find_text", query=term)
        if refs:
            return finalize(f"Found '{term}' {len(refs)}× on sheets {', '.join(sorted({x['sheet'] for x in res['matches']}))} {refs_str(refs)}.",
                            ev, "rules", trace)
        return finalize(f"NOT SURE: '{term}' does not appear in the drawing text.", ev, "rules", trace, status="abstain")
    return finalize("NOT SURE: the offline rule router only handles counts, door-tag lookups, callouts, text search and consistency checks. "
                    "Set an LLM API key for free-form questions.", ev, "rules", trace, status="abstain")


# ---------------------------------------------------------------- LLM tool loop
PROVIDERS = [  # (env key, kind, base url, default model)
    ("ANTHROPIC_API_KEY", "anthropic", "https://api.anthropic.com/v1", "claude-sonnet-4-5"),
    ("OPENAI_API_KEY", "openai", "https://api.openai.com/v1", "gpt-4.1"),
    ("XAI_API_KEY", "openai", "https://api.x.ai/v1", "grok-4"),
]


def available_engine():
    forced = os.environ.get("BP_ENGINE")
    if forced == "rules":
        return None
    for key, kind, base, model in PROVIDERS:
        if os.environ.get(key):
            return {"key": os.environ[key], "kind": kind, "base": os.environ.get("BP_BASE_URL", base),
                    "model": os.environ.get("BP_MODEL", model), "env": key}
    return None


def _http(url, headers, body):
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def _tool_payload(out, refs):
    return json.dumps({"result": out["result"],
                       "evidence_refs": [{"ref": n, "sheet": e["sheet"], "kind": e["kind"], "text": e["text"][:120]}
                                         for n, e in zip(refs, out["evidence"])][:80]})[:12000]


def llm_answer(question, proj: Project, eng, max_steps=8, http=_http):
    ev, trace = Evidence(), []
    tag = f"{eng['env'].split('_')[0].lower()}:{eng['model']}"
    sysmsg = SYSTEM + f"\nProject: {proj.name}. Door schedules: " + json.dumps(
        [{"title": s["title"], "columns": s["columns"]} for s in proj.g["schedules"]])
    if eng["kind"] == "anthropic":
        tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOL_SPECS]
        msgs = [{"role": "user", "content": question}]
        for _ in range(max_steps):
            data = http(eng["base"] + "/messages", {"x-api-key": eng["key"], "anthropic-version": "2023-06-01",
                                                   "content-type": "application/json"},
                        {"model": eng["model"], "max_tokens": 1200, "system": sysmsg, "tools": tools, "messages": msgs})
            msgs.append({"role": "assistant", "content": data["content"]})
            uses = [c for c in data["content"] if c["type"] == "tool_use"]
            if not uses:
                text = "".join(c.get("text", "") for c in data["content"] if c["type"] == "text")
                return finalize(text, ev, tag, trace)
            results = []
            for u in uses:
                out = call_tool(proj, u["name"], u.get("input") or {})
                refs = ev.add(out["evidence"])
                trace.append({"tool": u["name"], "args": u.get("input"), "result": out["result"], "refs": refs})
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": _tool_payload(out, refs)})
            msgs.append({"role": "user", "content": results})
    else:
        tools = [{"type": "function", "function": t} for t in TOOL_SPECS]
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": question}]
        for _ in range(max_steps):
            data = http(eng["base"] + "/chat/completions", {"Authorization": f"Bearer {eng['key']}", "content-type": "application/json"},
                        {"model": eng["model"], "messages": msgs, "tools": tools})
            msg = data["choices"][0]["message"]
            msgs.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            calls = msg.get("tool_calls") or []
            if not calls:
                return finalize(msg.get("content") or "", ev, tag, trace)
            for c in calls:
                args = json.loads(c["function"].get("arguments") or "{}")
                out = call_tool(proj, c["function"]["name"], args)
                refs = ev.add(out["evidence"])
                trace.append({"tool": c["function"]["name"], "args": args, "result": out["result"], "refs": refs})
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": _tool_payload(out, refs)})
    return finalize("NOT SURE: ran out of tool steps.", ev, tag, trace, status="abstain")


def answer(question, project, engine="auto"):
    proj = Project(project)
    eng = available_engine() if engine in ("auto", "llm") else None
    if eng:
        try:
            return llm_answer(question, proj, eng)
        except Exception as e:  # network / auth problems -> keep the demo alive
            out = rules_answer(question, proj)
            out["engine"] = f"rules (LLM failed: {type(e).__name__}: {str(e)[:120]})"
            return out
    return rules_answer(question, proj)


if __name__ == "__main__":
    import sys
    print(json.dumps(answer(sys.argv[2], sys.argv[1]), indent=1)[:4000])
