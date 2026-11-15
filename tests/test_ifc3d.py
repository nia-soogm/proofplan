"""3D evidence view: geometry endpoint + highlight ids match answer citations.

Does not change answers, gating or accuracy. Skips if the Junction IFC files
are not on this box.
"""
import os, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

J = Path(os.environ.get("BP_JDATA", "/workspace/jdata"))
DEMO = "data/ifc_models/rwth-digitalhub_digitalhub-fm-arc-v2_plan-02-e00-okrd.pdf"
DEMO_IFC = J / "data/ifc_models/rwth-digitalhub_digitalhub-fm-arc-v2.ifc"
FZK = "data/ifc_models/kit-fzk-haus_ac20-fzk-haus_plan-01-erdgeschoss.pdf"
FZK_IFC = J / "data/ifc_models/kit-fzk-haus_ac20-fzk-haus.ifc"
Q_DOORS = "How many doors are shown on the floor plan of 'E00_OKRD'?"

pytestmark = pytest.mark.skipif(not DEMO_IFC.exists() and not FZK_IFC.exists(), reason="dataset not available")


@pytest.fixture(scope="module")
def client():
    from app.server import app
    return TestClient(app)


def test_ifc3d_returns_geometry(client):
    path = FZK if FZK_IFC.exists() else DEMO
    r = client.get("/api/ifc3d", params={"path": path})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and d["n_elements"] > 0 and d["n_triangles"] > 0
    assert d["elements"] and "v" in d["elements"][0] and "f" in d["elements"][0]
    assert len(d["elements"][0]["v"]) >= 9 and len(d["elements"][0]["f"]) >= 3
    assert len(d["origin"]) == 3 and d["scale"] > 0
    doors = [e for e in d["elements"] if e["cls"] == "IfcDoor"]
    assert doors and all(len(e["guid"]) == 22 for e in doors)
    assert all(e.get("name") is not None and "qty" in e for e in doors)


def test_ifc3d_highlights_match_citations(client):
    ask = client.post("/api/jask", json={"path": DEMO, "question": Q_DOORS}).json()
    assert ask["status"] == "answered" and ask["value"] == 26
    guids = [c["guid"] for c in ask["citations"] if c.get("guid")]
    assert len(guids) == 26 and all(len(g) == 22 for g in guids)
    r = client.get("/api/ifc3d", params={"path": DEMO, "highlight": guids})
    assert r.status_code == 200
    d = r.json()
    assert set(d["highlight"]) == set(guids)
    assert d["missing"] == []
    by = {e["guid"]: e for e in d["elements"]}
    for g in d["highlight"]:
        assert by[g]["cls"] == "IfcDoor" and len(by[g]["f"]) >= 3


def test_ifc3d_rejects_non_ifc(client):
    r = client.get("/api/ifc3d", params={"path": "data/swiss_dwellings/foo.pdf"})
    assert r.status_code == 400


def test_door_answer_unchanged():
    """Guard: adding the 3D view must not change the 26-door demo answer."""
    from bp.jsolve import solve
    r = solve(Q_DOORS, DEMO)
    assert r["status"] == "answered" and r["value"] == 26
    assert len(r["evidence"]) == 26
