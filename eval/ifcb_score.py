"""Scorer for ifc_bench free-text answers (gold answers are long LLM-written reports).

Fixed BEFORE the first test run; stricter than eval.jeval.text_correct (which strips whitespace and substring-matches,
so any long answer full of digits can 'contain' a short gold number by accident).

  * every NUMBER in the gold must appear in our answer as a number that rounds to it at the gold's precision
    (gold 537 matches 537.2; gold 87.0 matches 87.0 / 87.04, not 870 or 1870); thousands separators are removed
  * every CODE token in the gold (letters+digits such as 2A12, IFC2X3, D200, 0otfaO0qPDAhynjJ6DmgH8) must appear as a whole token
  * gold with no numbers/codes: if the gold is a negative ("not available", "no ... found") our answer must also be
    negative; otherwise >= 60 % of the gold's content words (len >= 4) must appear in ours
Abstentions are never correct.
"""
import re

NEG = re.compile(r"\b(not available|not (?:included|found|present|defined|populated|modeled|modelled)|no [\w /-]*(?:found|information|elements?|data|present|defined)|"
                 r"none (?:found|present)|are not|is not|n/a)\b", re.I)
STOP = set("this that with from have been were they their there which what model building ifc information available found also each "
           "only into used uses using than then them these those such where when will would could should about other".split())
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")
# numbers glued to a unit or code are split: 300mm -> 300, 2.70m -> 2.70; pure codes (2A12, IFC2X3) are kept whole
_TOK = re.compile(r"[A-Za-z0-9_][\w.-]*")


def _numbers(s):
    out = []
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 19,495 -> 19495
    for t in re.findall(r"-?\d+(?:\.\d+)?", s):
        out.append(t)
    return out


def _codes(s):
    """Alphanumeric identifiers mixing letters and digits (not plain numbers with a unit suffix like 300mm / 2.5m / 87.0%)."""
    out = []
    for t in _TOK.findall(s):
        t = t.strip(".-")
        if re.search(r"\d", t) and re.search(r"[A-Za-z]", t) and not re.fullmatch(r"-?\d+(?:\.\d+)?(?:mm|cm|m|m2|m3|sqm|kg|kw|w|l|s|x|st|nd|rd|th|°)?", t, re.I):
            out.append(t.upper())
    return out


def _num_ok(g, ours):
    dec = len(g.split(".")[1]) if "." in g else 0
    gv = float(g)
    tol = 0.5 * 10 ** (-dec) + 1e-9
    return any(abs(float(o) - gv) <= tol for o in ours)


def _split(s):
    """(codes, numbers) of a text: code tokens kept whole, numbers taken from every other token (300mm -> 300)."""
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 19,495 -> 19495
    codes, nums = [], []
    for t in re.findall(r"[^\s()\[\]{}:;,=/×*\"'<>]+", s):
        c = _codes(t)
        if c and len(c) == 1 and re.fullmatch(r"[A-Za-z0-9_.-]+", t.strip(".-")) and not re.fullmatch(r"-?\d+(?:\.\d+)*", t):
            codes += c
        else:
            nums += re.findall(r"-?\d+(?:\.\d+)?", t)
    return codes, nums


def score(gold, ours):
    if ours is None:
        return False
    gold, ours = str(gold), str(ours)
    ours = re.sub(r"(?m)^(\s*)\d{1,3}\.\s", r"\1- ", ours)  # list indices in OUR text are not facts
    gcodes, gnums = _split(gold)
    ocodes, _ = _split(ours)
    onums = _numbers(ours)
    otoks = {t.upper().strip(".-") for t in _TOK.findall(ours)}
    if gcodes or gnums:
        return all(c in ocodes or c in otoks for c in gcodes) and all(_num_ok(g, onums) for g in gnums)
    if NEG.search(gold):
        return bool(NEG.search(ours))
    words = [w.lower() for w in re.findall(r"[^\W\d_]{4,}", gold) if w.lower() not in STOP]
    if not words:
        return False
    ol = ours.lower()
    return sum(w in ol for w in words) / len(words) >= 0.6
