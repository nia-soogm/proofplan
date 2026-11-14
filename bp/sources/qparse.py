"""Language-light question parsing: quoted strings, element kind, measure, scope."""
import re

QUOTE_RE = re.compile(r"(?<!\w)'(.+?)'(?!\w)|„([^“”]*)[“”]|«\s*([^»]*?)\s*»|\"([^\"]*)\"|“([^”]*)”|‘([^’]*)’|»([^«]*)«")

KINDS = [  # (kind, regex) checked in order; first hit wins
    ("light", r"light fixture|iluminat|leuchte|verlichting"),
    ("sanitary", r"sanitar|sanitär|sanitair|sanitaire"),
    ("door", r"\bdoors?\b|türen|\btür\b|deuren|\bdøre|\bust\b|puertas|porte\b|portes|drzwi|uși"),
    ("window", r"windows?|fenster|ramen\b|vinduer|akent|ventanas|finestre|fenêtres|okien|ferestre"),
    ("stair", r"stairs?|treppe|trappe|treppi|escaleras|scale\b|schod|scări"),
    ("column", r"columns?|stützen|kolommen|søjler|posti|pilares|pilastri|stâlpi|słup"),
    ("beam", r"beams?|träger|balken|bjælker|tala\b|vigas|travi|belek|grinzi"),
    ("slab", r"slabs?|decken|bodenplatten|vloeren|dæk|plader|losas|solai"),
    ("wall", r"walls?|wände|wanden|vægge|seina|muros|muri\b|ścian|pereți|murs"),
    ("space", r"rooms?|spaces?|räume|raums?|ruimtes?|\brum\b|rummet|ruumi|locales?|local\b|pomieszcz|locali|pièces?|stanze"),
]
MODEL_RE = r"\bmodel\b|modell|modelo|modello|modelu|modellen|mudel|modèle|modelul"


def quotes(q):
    return [next(g for g in m.groups() if g is not None).strip() for m in QUOTE_RE.finditer(q)]


def measure(q):
    ql = q.lower()
    if "m³" in q or "m3" in ql or "volum" in ql:
        return "volume"
    if "m²" in q or "m2" in ql:
        return "area"
    if re.search(r"\b(length|länge|lang\b|lengte|længde|longitud|pikkus|metres|meter|metern|metros|meetrit)", ql):
        return "length"
    return "count"


def kind(q):
    ql = q.lower()
    for k, rx in KINDS:
        if re.search(rx, ql):
            return k
    return None


def scope(q):
    return "model" if re.search(MODEL_RE, q.lower()) else "plan"
