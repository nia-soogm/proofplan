"""ifc_bench strict question parser: every content word of the question must be consumed by the template that answers it.

A regex route only says a handler *could* be relevant. Questions on unseen models often add a qualifier the handler ignores
("volume of load-bearing walls", "doors with a width over 1 m", "furniture per apartment"); the handler would then answer a
broader question with a whole-building figure. Here each handler declares the words it understands (CONSUMES). A word of the
question is consumed when it is
  * a filler / answer-shape word (FILLER: question words, "provide", "including", "quantities", "breakdown" ...),
  * a word the handler declares (prefix match on a stem, so "thicknesses" matches "thick"),
  * a number, or a code token (GUID, room number), for handlers that look those up,
  * part of a storey name or space name of this model, for handlers that resolve storeys / spaces by name.
Any other word -> the handler is skipped (the next matching route may still answer; otherwise NOT SURE).

The vocabularies were written from the dev questions only (the test questions were not read for this).
"""
from __future__ import annotations
import re

FILLER = set("""
what which how many much who whose where when is are was were be been being do does did done the a an of in on to and or for with by from
this that these those there here it its as at all each every any some both your you me my our we i can could would will should please
provide give list show tell identify specify specifying describe detail details detailed including include includes included along also
total overall complete full building buildings model models ifc file project present installed used use using exist exists existing
available found located applied contained contain contains respective their they them throughout across within inside per
quantity quantities count counts number numbers breakdown broken down distribution distributed grouped group summary overview
type types kind kinds different various variety specific individual information info data value values respectively
percentage percentages proportion proportions share
have has had side showing specified relevant about comprehensive analysis usage
""".split())

S = lambda s: tuple(s.split())  # noqa: E731

# stems (prefix match on the lower-cased word). Keep to what the handler really reports.
CONSUMES = {
    "h_software": S("software program applicat tool standard schema version modell modeling modelling creat author export applied"),
    "h_project": S("project location site detail address latitude longitude name"),
    "h_levels": S("height elevation level storey story floor"),
    "h_space_breakdown": S("space room level floor area function individual primary percent storey"),
    "h_storey_spaces": S("space room area level floor storey story ground basement attic function ceiling height first second third fourth fifth"),
    "h_area_by_level": S("gross net usable floor area level storey story"),
    "h_largest_room": S("room space largest biggest area"),
    "h_tallest": S("room space tallest highest ceiling height"),
    "h_room_kind": S("toilet restroom wc lavator bathroom bath meeting conference x ray xray laundry utility balcon working space office room "
                     "floor level storey ground first second third fourth area kitchen storage located open plan"),
    "h_types": S("door window wall column beam railing stair furniture furnish curtain slab covering roof exterior interior outer outside "
                 "external internal inside partition inventory item sizing size dimension specification spec structur propert floor level storey"),
    "h_wall_thick": S("thick wall"),
    "h_wall_materials2": S("material wall construct outer exterior external outside interior internal inside partition layer thick"),
    "h_loadbearing": S("load bearing wall non versus compared status interior"),
    "h_wall_length": S("length wall combined exterior interior inside outside outer inner external internal partition"),
    "h_firerating": S("fire rating door window wall"),
    "h_extreme_width": S("door window width widest narrowest largest smallest greatest maximum minimum biggest two three four five"),
    "h_count": S("window door room space building wall column beam slab stair clinic house modeled modelled"),
    "h_absent": S("sanitary fixture equipment beam mep system pipe piping heating cooling electrical component furniture item column light"),
    "h_struct_summary": S("structural component element roof slab member beam employed"),
    "h_slab_types2": S("slab thick"),
    "h_slab_volume2": S("slab volume floor combined"),
    "h_guid_dims": S("dimension window door guid width height size"),
    "h_phases": S("construction phase renovation existing element correspond represented work versus"),
    "h_wall_envelope": S("wall exterior interior envelope partition form propert thickness length"),
    "h_ceiling_mat": S("ceiling material"),
    "h_coverings": S("covering"),
    "h_vertical": S("vertical circulation element stair elevator ramp escalator sub component"),
    "h_furniture_in_space": S("furniture item space room office kitchen working basic floor ground level"),
    "h_room_types": S("room"),
    "h_room_lookup": S("room ceiling height area size treatment big"),
    "h_named_room": S("size area ceiling height room space"),
    "h_doors_windows": S("opening door window versus"),
    "h_building": S("type function"),
    "h_metrics": S("key metric element level footprint area floor room space"),
    "h_sanitary_types": S("sanitary terminal"),
    "h_mep_types": S("pipe fitting accessor valve flex duct air terminal waste pump fan coil damper heat recovery measuring meter boiler "
                     "water mechanical equipment controller radiator insulation junction box cable tray conduit switch receptacle outlet "
                     "distribution board light lamp fixture device communication tank plumbing ventilation system unit"),
    "h_mep_length": S("pipe duct ductwork conduit cable length segment hydronic network plumbing ventilation system"),
    "h_categories": S("component category electrical plumbing hvac ventilation system equipment level"),
}

# handlers that resolve storeys / spaces / codes from the question (their names and numbers are consumed)
USES_STOREY = {"h_storey_spaces", "h_room_kind", "h_types", "h_furniture_in_space", "h_named_room", "h_categories", "h_mep_types"}
USES_SPACE = {"h_furniture_in_space", "h_named_room", "h_room_lookup", "h_room_kind"}
USES_NUM = {"h_storey_spaces", "h_room_kind", "h_furniture_in_space", "h_room_lookup", "h_guid_dims", "h_named_room", "h_extreme_width",
            "h_categories", "h_mep_types"}


def words(q):
    return re.findall(r"[a-zäöüß]+", q.lower())


def _ok(w, stems):
    return any(w.startswith(s) for s in stems)


def unconsumed(q, hname, m=None):
    """Words of q that handler hname does not consume (empty list = the parse is complete)."""
    stems = CONSUMES.get(hname)
    if stems is None:
        return []  # handler without a declared vocabulary: no check
    names = set()
    if m is not None and hname in USES_STOREY:
        for s in m.storeys:
            names |= set(words(f"{s.get('name') or ''} {s.get('long') or ''}"))
    if m is not None and hname in USES_SPACE:
        for s in m.spaces():
            names |= set(words(f"{s.get('name') or ''} {s.get('long') or ''}"))
    out = []
    for raw in re.findall(r"[\wäöüß$-]+", q):
        if re.fullmatch(r"[\d.,-]+", raw):
            if hname not in USES_NUM:
                out.append(raw)
            continue
        if re.search(r"\d", raw):  # code token (GUID, room number 2A12, 'level1')
            if hname not in USES_NUM:
                out.append(raw)
            continue
        for w in words(raw):
            if w in FILLER or len(w) <= 1 or _ok(w, stems) or w in names:
                continue
            if w.endswith("s") and (w[:-1] in FILLER or w[:-1] in names):
                continue
            out.append(w)
    return out
