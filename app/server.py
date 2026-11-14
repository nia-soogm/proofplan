"""ProofPlan demo server. Run: uvicorn app.server:app --port 8765"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # tiny per-request sklearn predicts: OpenMP pools across request threads oversubscribe the CPU
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from bp import agent
from bp.tools import Project, list_projects as _all_projects, PROJECTS
from app import jdata
import os


def list_projects():
    """Public list: only the Junction datasets. Your own PDFs ingested with `python -m bp.ingest` show with BP_SHOW_DEMO=1."""
    if os.environ.get("BP_SHOW_DEMO") != "1":
        return []
    return _all_projects()

app = FastAPI(title="ProofPlan")
STATIC = Path(__file__).parent / "static"



class Ask(BaseModel):
    project: str
    question: str
    engine: str = "auto"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/projects")
def projects():
    eng = agent.available_engine()
    return {"projects": [{"name": f"jd:{k}", "label": v, "samples": [], "licence": jdata.LICENCE.get(k, ""), "blurb": jdata.BLURB.get(k, "")}
                         for k, v in jdata.SOURCES.items()] +
                        [{"name": p, "label": p, "samples": []} for p in list_projects()],
            "engine": f"{eng['env']} / {eng['model']}" if eng else "rules (no LLM key set)"}


@app.get("/api/accuracy")
def accuracy():
    f = Path(__file__).parent.parent / "eval" / "accuracy.json"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/project/{name}/sheets")
def sheets(name: str):
    if name not in list_projects():
        raise HTTPException(404)
    p = Project(name)
    return [{"page": s["page"], "sheet": s["sheet_id"], "title": s["title"], "level": s["level"],
             "width": s["width"], "height": s["height"]} for s in p.g["sheets"]]


@app.get("/pages/{name}/{page}.png")
def page_png(name: str, page: int):
    f = PROJECTS / name / "pages" / f"{page}.png"
    if name not in list_projects() or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "max-age=3600"})


@app.post("/api/ask")
def ask(a: Ask):
    if a.project not in list_projects():
        raise HTTPException(404, "unknown project")
    return agent.answer(a.question, a.project, engine=a.engine)


class JAsk(BaseModel):
    path: str
    question: str


@app.get("/api/jdata/{src}/files")
def jfiles(src: str):
    if src not in jdata.SOURCES:
        raise HTTPException(404)
    return jdata.catalog()[src]


@app.get("/jpages/{src}/{pdf:path}")
def jpage(src: str, pdf: str):
    if src not in jdata.SOURCES:
        raise HTTPException(404)
    f = jdata.page_png(src, pdf)
    if not f:
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "max-age=3600"})


@app.post("/api/jask")
def jask(a: JAsk):
    if not a.path.startswith("data/") or a.path.split("/")[1] not in jdata.SOURCES or ".." in a.path:
        raise HTTPException(400, "unsupported path")
    return jdata.ask(a.path, a.question)
