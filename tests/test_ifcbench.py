"""ifc_bench: strict text scorer + deterministic IFC answers (model tests skip when the dataset is not on this box)."""
import os, sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ifcb_score import score

from bp.sources import JDATA
FZK = JDATA / "data" / "ifc_bench" / "fzk_house_arc.ifc"


def test_score_numbers_and_codes():
    assert score("Room 2A12 is 2.80 m high", "room 2A12: 2.8 m")
    assert score("Total: 19,495 components (38.9%)", "19495 components, 38.9 %")
    assert not score("537 sq m", "5370 sq m")                     # no substring matches inside other numbers
    assert not score("There are 29 room types", "   29. ADMIN\n   30. LAB")  # list indices are not facts
    assert not score("2 doors", None)


def test_score_negative_gold():
    assert score("Fire rating information is not available in this model.", "No fire rating information found (0 doors).")
    assert not score("Fire rating information is not available in this model.", "Doors with fire rating: 5")


@pytest.mark.skipif(not FZK.exists(), reason="dataset not available")
def test_fzk_answers_with_guid_evidence():
    from bp.jsolve import solve
    r = solve("Which door types are installed in the building, with their quantities?", "data/ifc_bench/fzk_house_arc.ifc")
    assert r["status"] == "answered" and "Total doors: 5" in r["answer"]
    assert len(r["evidence"]) == 5 and all(len(e["guid"]) == 22 for e in r["evidence"])
    assert any(e.get("bbox") for e in r["evidence"])  # drawn on the ifc_models render of the same model
    r = solve("What is the total length of exterior walls?", "data/ifc_bench/fzk_house_arc.ifc")
    assert "85.60 m" in r["answer"]
    r = solve("What is the estimated distance to the nearest emergency exit?", "data/ifc_bench/fzk_house_arc.ifc")
    assert r["status"] == "abstain"


@pytest.mark.skipif(not FZK.exists(), reason="dataset not available")
def test_lomo_gate_and_strict_parser():
    from bp.jsolve import solve
    # h_count is right on dev only on one model -> withheld by the leave-one-model-out gate, draft still shown
    r = solve("How many doors are there in this building?", "data/ifc_bench/fzk_house_arc.ifc")
    assert r["status"] == "abstain" and "withheld" in r["answer"] and "5 doors" in r["answer"]
    # a qualifier the template does not understand -> NOT SURE naming the word
    r = solve("What is the total length of exterior walls with a height above 3 m?", "data/ifc_bench/fzk_house_arc.ifc")
    assert r["status"] == "abstain" and "'above'" in r["answer"]


def test_parser_consumes_words():
    from bp.sources.ifcb_parse import unconsumed
    assert unconsumed("What is the total length of exterior walls?", "h_wall_length") == []
    assert unconsumed("What is the total volume of load-bearing walls?", "h_loadbearing") == ["volume"]


def test_storey_number_parse():
    from bp.sources.ifcbench import _storey_num
    assert _storey_num("ground floor") == 0 and _storey_num("Erdgeschoss") == 0
    assert _storey_num("3rd Floor") == 3 and _storey_num("level 2") == 2 and _storey_num("first floor") == 1
    assert _storey_num("Dachgeschoss") is None


@pytest.mark.skipif(not FZK.exists(), reason="dataset not available")
def test_self_checks():
    from bp.jsolve import solve
    p = "data/ifc_bench/fzk_house_arc.ifc"
    # wall 'types' group differently by type object vs instance name in this model -> NOT SURE with the reason
    r = solve("What wall types are used throughout the building, with their respective quantities?", p)
    assert r["status"] == "abstain" and "ambiguous" in r["answer"]
    # spaces + areas re-derived from the IFC decomposition and IfcElementQuantity -> answered
    r = solve("What are all the rooms on the ground level, including their areas and ceiling heights?", p)
    assert r["status"] == "answered"
    # templates without an independent re-derivation are not released
    r = solve("What is the total length of exterior walls?", p)
    assert r["status"] == "abstain"
