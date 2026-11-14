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
    r = solve("How many doors are there in this building?", "data/ifc_bench/fzk_house_arc.ifc")
    assert r["status"] == "answered" and "5 doors" in r["answer"]
    assert len(r["evidence"]) == 5 and all(len(e["guid"]) == 22 for e in r["evidence"])
    assert any(e.get("bbox") for e in r["evidence"])  # drawn on the ifc_models render of the same model
    r = solve("What is the total length of exterior walls?", "data/ifc_bench/fzk_house_arc.ifc")
    assert "85.60 m" in r["answer"]
    r = solve("What is the estimated distance to the nearest emergency exit?", "data/ifc_bench/fzk_house_arc.ifc")
    assert r["status"] == "abstain"
