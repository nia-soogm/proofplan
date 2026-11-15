"""CubiCasa room-kind counts (bedroom, bathroom, separate WC, sauna, kitchen, balcony) from OCR'd room labels.

Pipeline (CPU, no key, no LLM):
1. Label lexicon. Finnish and Swedish room abbreviations, plus a few frequent RapidOCR misreads of 'MH' (HM, HW, HV, NH).
   Combo labels ('KH/WC', 'OH+KK', 'S/PH', 'KHH+PH') count once for each target kind they name, but only when every part
   is a known room code or a unit. The lexicon is kept apart from the 'other room' codes (OH, ET, VH, KHH, ...). Those
   are not counted. They show that OCR read the small labels, which is what makes "no label → 0" believable.
2. Spatial de-duplication. Same-kind hits closer than ~2.5 label heights are one room ('LASITETTU' over 'PARVEKE').
3. Apartment-type summary ('3H+K+S', '4h+kt+s', '3H-K-S', '2 H + KK'). Bedrooms = H − 1, and the '+S' / '+K' / '+WC'
   flags give 1 / 0 candidates for the other kinds.
4. Candidates per kind: the label count, plus the summary value when one is present. A gradient-boosted gate trained
   out-of-fold on DEV images (scripts/cubi/train_rooms.py) scores P(candidate == truth) from count, source, OCR
   confidence, label height, how many other room labels were read, and summary agreement. We answer only when
   p >= tau_kind, where tau_kind was chosen on dev out-of-fold for precision. Each answer carries the boxes it used:
   the counted labels, or for a 0 the summary and/or the other room labels that were read.
"""
import functools, pickle, re
from pathlib import Path

KINDS = ("bedroom", "bathroom", "wc", "sauna", "kitchen", "balcony")

LEX = {
    "bedroom": {"MH", "MAKUUHUONE", "MAKUUH", "MAKH", "MAKUU", "MAKUUTILA", "SOVRUM", "SOVR", "SOV", "LASTENH", "LASTENHUONE"},
    "sauna": {"S", "SAUNA", "SAUN", "BASTU", "LOYLYH", "LÖYLYH"},
    "bathroom": {"KH", "KPH", "PH", "PESUH", "PESUHUONE", "PSH", "KYLPYH", "KYLPYHUONE", "PESUTILA", "PESUTILAT", "SH",
                 "SUIHKU", "PESU", "BADRUM", "BDR", "DUSCH", "TVÄTTRUM"},
    "wc": {"WC", "W.C", "W.C.", "VC", "TOA"},
    "kitchen": {"K", "KK", "KT", "TPK", "AVOK", "AVOKEITTIO", "AVOKEITTIÖ", "KEITTIO", "KEITTIÖ", "KEITTIÖTILA", "KEITTIOTILA",
                "K-TILA", "KEITTOKOMERO", "AVO-K", "TUPAK", "TUPAKEITTIO", "TUPAKEITTIÖ", "KEITTO", "KEITTI", "KEITT", "KEIT",
                "KEITTIÖ/RUOK", "KÖK", "KOK", "KOKETT", "KÖKSVRÅ"},
    "balcony": {"PARVEKE", "PARV", "PARY", "LASIT.PARV", "LASITETTU PARVEKE", "LASITETTUPARVEKE", "LASIPARVEKE", "LPARV",
                "LASIT PARV", "LASIT", "LASITETTU", "BALKONG", "BALK", "RANSKALAINEN PARVEKE", "RANSK.PARV"},
}
MISREAD = {"bedroom": {"HM", "HW", "HV", "NH", "MN", "MHE", "MHH"}}  # RapidOCR misreads of 'MH' (dev: they explain the gap to gold)
OTHER = {  # room codes that are not counted, used for coverage features and to accept combo labels
    "OH", "OLOH", "OLOHUONE", "ET", "ETEINEN", "VH", "VAATEH", "VAATEHUONE", "KHH", "KODINH", "KODINHOITOH", "VAR", "VARASTO",
    "AULA", "HALL", "HALLI", "RT", "RUOK", "RUOKAILU", "RUOKAILUTILA", "US", "SK", "PKH", "PUKUH", "PUKUHUONE", "TEKN", "TKH",
    "TERASSI", "TERASS", "TER", "KUISTI", "KUIST", "AUTOTALLI", "AT", "AK", "AUTOKATOS", "KATOS", "TYÖH", "TYOH", "TYÖHUONE",
    "TYOHUONE", "ASKARTELU", "TAKKAH", "TAKKAHUONE", "ALK", "ALKOVI", "PARVI", "KELLARI", "KATTILAH", "LVK", "IVH", "VS", "HUONE",
    "TUPA", "LH", "APK", "PPK", "APUK", "JK", "LAMM", "VAJA", "KOMERO", "PORRAS", "HISSI", "OLESKELU", "VARDAGSRUM", "VR",
    "FÖRRÅD", "FRD", "KLK", "KLÄDKAMMARE", "GROVK", "TVÄTT", "ENTRÉ", "HOBBY", "AS", "ASH", "AS.H", "HH", "IK", "TH", "PE",
    "VK", "VARAUS", "ULLAKKO", "AUTOVAJA", "PATIO", "KYLMÄ", "KYLMA", "TILA", "R", "TK", "TUULIK",
}
KH = {"KH", "KPH", "KYLPYH", "KYLPYHUONE", "BADRUM", "BDR"}  # bathroom proper (tub or shower)
PH = {"PH", "PESUH", "PESUHUONE", "PSH", "PESUTILA", "PESUTILAT", "SH", "SUIHKU", "PESU", "DUSCH", "TVÄTTRUM"}  # washroom next to a sauna
SAUNA_WORD = {"SAUNA", "SAUN", "BASTU", "LOYLYH", "LÖYLYH"}
NEW_KEYS = ("n_kh", "n_ph", "n_ph_only", "combo_bw", "combo_sb", "n_khh", "sauna_word", "sauna_letter", "src_wash", "eq_wash")
UNITS = {"M²", "M2", "M", "MZ", "M?", "", "H", "HUONE"}
SUMMARY_RE = re.compile(r"(?<![\d,.])([1-9])\s*[HhRr]\s*[+\-,.]\s*([A-ZÄÖÅa-zäöå/ +\-,.]+)")
SUM_PART = {"sauna": {"S", "SA", "SAUNA"}, "kitchen": {"K", "KK", "KT", "TPK", "AVOK", "TUPAK", "KEITTIÖ", "KEITTIO", "KÖK", "KOK", "KV", "KS"},
            "wc": {"WC"}, "bathroom": {"KH", "KPH", "PH", "PESUH", "PSH"}, "balcony": {"PARV", "PARVEKE", "LASIT", "P", "PARV."}}
# for the Swedish 'r+k' form only kitchen/sauna make sense; bedrooms = rooms - 1 is the same convention


def core(t):
    t = t.upper().strip()
    t = re.sub(r"\d+([,.]\d+)?", " ", t)
    t = re.sub(r"[^\wÄÖÅ²./+ -]", " ", t)
    return re.sub(r"\s+", " ", t).strip(" .-/+")


def _kind_of_code(c):
    c2 = c.replace(" ", "").replace(".", "")
    for k, lex in LEX.items():
        if c in lex or c2 in lex or c.rstrip(".") in lex:
            return k, False
    for k, lex in MISREAD.items():
        if c in lex or c2 in lex:
            return k, True
    return None, False


def classify(text):
    """-> list of (kind, misread) for one OCR token; [] if it is not a counted room label. Summaries are handled elsewhere."""
    if SUMMARY_RE.search(text):
        return []
    if "+" in text and re.search(r"m2|m²|M2|M²|\d[,.]\d", text):  # area list 'OH+KK 26.0 m²' repeats a room label
        return []
    if re.match(r"\s*\d*\s*[Hh]\s*\+", text):  # summary whose digit was lost ('H+KK')
        return []
    c = core(text)
    if not c or len(c) > 24:
        return []
    k, mis = _kind_of_code(c)
    if k:
        if len(c) == 1:  # single-letter label: must be an upper-case letter, not glued to digits on its left ('0 9k21')
            if not re.search(r"(^|[^\w])" + c + r"(?![A-Za-zÄÖÅäöå])", text) or re.search(r"\d\s*\d.*" + c, text):
                return []
        return [(k, mis)]
    parts = [p for p in re.split(r"[/+ ,]+", c) if p]
    if len(parts) < 2 or len(parts) > 4:
        return []
    out = []
    for p in parts:
        k, mis = _kind_of_code(p)
        if k:
            out.append((k, mis))
        elif p.rstrip(".") not in OTHER and p not in UNITS:
            return []  # an unknown word in the token: not a room label
    return list(dict.fromkeys(out))


def is_other(text):
    c = core(text)
    return bool(c) and (c in OTHER or c.rstrip(".") in OTHER)


def parse_summary(text):
    """'3H+K+S' -> {'h': 3, 'parts': {'K','S'}}; also '4h+kt+s', '3H-K-S', '2 H + KK', '3r+k' (Swedish)."""
    m = SUMMARY_RE.search(text)
    if not m:
        return None
    rest = m.group(2).upper()
    parts = {p.strip(" .") for p in re.split(r"[+\-,/ ]+", rest) if p.strip(" .")}
    parts = {re.sub(r"M²|M2$", "", p) for p in parts} - {""}
    if not parts:
        return None
    known = set().union(*SUM_PART.values()) | OTHER | {"S", "VH", "KHH", "AUTOTALLI", "AT", "PARVI", "SAUNA"}
    if not (parts & known):
        return None
    return {"h": int(m.group(1)), "parts": parts}


def codes(t):
    """Set of room codes in one OCR token ('KH/WC' -> {'KH', 'WC', 'KHWC'})."""
    c = core(t["text"])
    return {p.rstrip(".") for p in re.split(r"[/+ ,]+", c) if p} | {c.replace(" ", "")}


def code_feats(ex):
    """Code identity of what was read: KH-type vs PH-type washrooms, combo labels, utility room, sauna word vs single 'S'."""
    hb, hw, hs = ex["hits"]["bathroom"], ex["hits"]["wc"], ex["hits"]["sauna"]
    inn = lambda t, L: any(t is u for u in L)
    ph = [t for t in hb if (codes(t) & PH) and not (codes(t) & KH)]
    return {"n_kh": sum(1 for t in hb if codes(t) & KH), "n_ph": len(ph), "n_ph_only": len([t for t in ph if not inn(t, hs)]),
            "combo_bw": sum(1 for t in hb if inn(t, hw)), "combo_sb": sum(1 for t in hb if inn(t, hs)),
            "n_khh": sum(1 for t in ex["others"] if codes(t) & {"KHH", "KODINH", "KODINHOITOH"}),
            "sauna_word": sum(1 for t in hs if codes(t) & SAUNA_WORD), "sauna_letter": sum(1 for t in hs if core(t["text"]).replace(" ", "") == "S")}, ph


def _h(t):
    return max(1.0, t["bbox"][3] - t["bbox"][1])


def _dedupe(hits):
    """Same-kind hits closer than 2.5 label heights are one room (two-line labels, OCR splitting 'LASITETTU' / 'PARVEKE')."""
    out = []
    for t in sorted(hits, key=lambda t: -t["conf"]):
        cx, cy = (t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2
        dup = False
        for u in out:
            ux, uy = (u["bbox"][0] + u["bbox"][2]) / 2, (u["bbox"][1] + u["bbox"][3]) / 2
            hh = 2.5 * max(_h(t), _h(u))
            if abs(cx - ux) < hh * 1.6 and abs(cy - uy) < hh:
                dup = True; break
        if not dup:
            out.append(t)
    return out


def extract(tokens, min_conf=0.1, strong=0.5):
    """Room-label hits per kind. OCR v2 keeps tokens down to conf 0.1. Hits at or above `strong` are 'strong'. Weaker
    ones (mostly single letters like 'S' or 'K', which RapidOCR scores at 0.2-0.5) form a separate candidate, and
    the gate learns how far to trust them."""
    toks = [t for t in tokens if t["conf"] >= min_conf and t.get("orient_ok", True)]
    hits = {k: [] for k in KINDS}; mis = {k: 0 for k in KINDS}
    others, summ = [], []
    for t in toks:
        s = parse_summary(t["text"])
        if s and t["conf"] >= 0.3:
            summ.append((t, s)); continue
        ks = classify(t["text"])
        for k, m in ks:
            hits[k].append(t); mis[k] += m
        if not ks and t["conf"] >= strong and is_other(t["text"]):
            others.append(t)
    raw = {k: len(v) for k, v in hits.items()}
    hits = {k: _dedupe(v) for k, v in hits.items()}
    strong_hits = {k: [t for t in v if t["conf"] >= strong] for k, v in hits.items()}
    labs = [t for v in strong_hits.values() for t in v] + others
    hs = sorted(_h(t) for t in labs)
    return {"hits": hits, "strong": strong_hits, "raw": raw, "mis": mis, "others": others, "summ": summ, "n_tok": len(toks),
            "lab_h": hs[len(hs) // 2] if hs else 0.0, "n_lab": len(labs), "n_other": len(others)}


def summary_value(kind, s):
    if kind == "bedroom":
        return s["h"] - 1
    return 1 if s["parts"] & SUM_PART[kind] else 0


def candidates(ex, kind, size, use_new=True):
    """-> list of (value, source, features, evidence) where evidence = [(token, label)]. Candidate values: strong label
    count, all-label count (incl. weak single letters), presence (1 if any label: two labels of one kitchen are common),
    and the apartment-type summary value."""
    allh = ex["hits"][kind]; st = ex["strong"][kind]
    n_all, n_st = len(allh), len(st)
    sv = [summary_value(kind, s) for _, s in ex["summ"]]
    s_val = sv[0] if sv else -1
    s_agree = len(set(sv)) <= 1
    kinds_seen = sum(1 for k in KINDS if ex["strong"][k])
    confs = [t["conf"] for t in allh] or [0.0]
    W, H = size
    base = {
        "n": n_st, "n_all": n_all, "n_weak": n_all - n_st, "raw": ex["raw"][kind], "mis": ex["mis"][kind],
        "n_other": ex["n_other"], "n_lab": ex["n_lab"], "kinds_seen": kinds_seen, "n_tok": ex["n_tok"],
        "lab_h_rel": ex["lab_h"] / max(1, max(W, H)) * 100, "minconf": min(confs), "maxconf": max(confs),
        "meanconf": sum(confs) / len(confs), "has_sum": int(bool(sv)), "sum_val": s_val, "n_sum": len(sv),
        "sum_agree": int(s_agree), "img_mp": W * H / 1e6,
        "single_letter": int(any(len(core(t["text"]).replace(" ", "")) == 1 for t in allh)),
        "n_bed": len(ex["strong"]["bedroom"]), "n_bath": len(ex["strong"]["bathroom"]), "n_kit": len(ex["strong"]["kitchen"]),
        "n_sauna": len(ex["strong"]["sauna"]), "n_wc": len(ex["strong"]["wc"]), "n_bal": len(ex["strong"]["balcony"]),
    }
    cf, ph = code_feats(ex)
    if use_new:
        base.update(cf)
    vals = {}  # value -> (source tags, evidence tokens)
    def add(v, tag, toks):
        tags, ev = vals.setdefault(v, (set(), []))
        tags.add(tag)
        for t in toks:
            if all(t is not u for u in ev):
                ev.append(t)
    if n_st > 0 or ex["n_lab"] >= 1 or sv:
        add(n_st, "label", st)
    if n_all != n_st:
        add(n_all, "label_all", allh)
    if n_all >= 1 and kind in ("kitchen", "bathroom", "sauna", "balcony"):
        add(1, "presence", sorted(allh, key=lambda t: -t["conf"])[:1])
    if sv and s_agree and s_val >= 0:
        add(s_val, "summary", [])
    if use_new and kind == "sauna" and n_all == 0 and ph:
        add(1, "washroom", ph[:2])  # a PH / PESUH washroom with no sauna label read: saunas usually adjoin it (dev: 79 %)
    out = []
    for v, (tags, toks) in vals.items():
        f = dict(base, val=v, src_label=int("label" in tags), src_all=int("label_all" in tags), src_pres=int("presence" in tags),
                 src_sum=int("summary" in tags), eq_sum=int(v == s_val), eq_lab=int(v == n_st), eq_all=int(v == n_all),
                 diff_sum=(v - s_val) if sv else -9, diff_lab=v - n_st, diff_all=v - n_all)
        if use_new:
            f.update(src_wash=int("washroom" in tags), eq_wash=int(kind == "sauna" and n_all == 0 and bool(ph) and v == 1))
        ev = [(t, "washroom (sauna inferred)" if "washroom" in tags and any(t is u for u in ph) else kind) for t in toks]
        if sv and v == s_val:
            ev.append((ex["summ"][0][0], "apartment-type summary"))
        if v == 0 and not ev:
            ev = [(t, "other room label") for t in ex["others"][:12]] + [(t, k) for k in KINDS for t in ex["strong"][k]][:12]
        if ev:
            out.append((v, "+".join(sorted(tags)), f, ev))
    return out


FKEYS = None


def fvec(f):
    return [f[k] for k in sorted(f)]


@functools.lru_cache(maxsize=1)
def model():
    f = Path(__file__).resolve().parent.parent / "models" / "cubi_rooms.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def answer(path, kind, tokens, size):
    """-> (status, value, message, evidence) or None if no model."""
    M = model()
    if M is None or kind not in M["models"]:
        return None
    ex = extract(tokens)
    cs = candidates(ex, kind, size, use_new=M.get("new_feats", False))
    if not cs:
        return ("abstain", None, f"no room labels or apartment-type summary readable for {kind}", [])
    import numpy as np
    m = M["models"][kind]
    X = np.array([fvec(f) for _, _, f, _ in cs], float)
    p = m.predict_proba(X)[:, 1]
    i = int(p.argmax()); v, src, f, ev = cs[i]
    from .calib import tau_for
    tau = tau_for(M, kind)
    return ("answered" if p[i] >= tau else "abstain", v, src, ev, float(p[i]), float(tau), ex)
