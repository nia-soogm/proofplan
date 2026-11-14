"""Offline test of the LLM tool-calling loop with a scripted fake provider (no API key needed)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bp.agent import llm_answer
from bp.tools import Project, PROJECTS
import pytest
import os
# Needs a PDF project with a door schedule that lists door "2" (python -m bp.ingest your.pdf --name NAME; BP_TEST_PROJECT=NAME).
PROJ = os.environ.get("BP_TEST_PROJECT", "")
pytestmark = pytest.mark.skipif(not PROJ or not (PROJECTS / PROJ).exists(), reason="set BP_TEST_PROJECT to an ingested PDF project")


def fake_openai():
    step = {"n": 0}
    def http(url, headers, body):
        step["n"] += 1
        if step["n"] == 1:
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "schedule_row", "arguments": json.dumps({"tag": "2", "attribute": "fire rating"})}}]}}]}
        refs = json.loads(body["messages"][-1]["content"])["evidence_refs"]
        return {"choices": [{"message": {"role": "assistant", "content": f'Door 2 is rated 20 MIN [{refs[0]["ref"]}] [99].'}}]}
    return http


def fake_anthropic_no_cite():
    def http(url, headers, body):
        return {"content": [{"type": "text", "text": "There are probably 20 doors."}]}
    return http


def test_openai_loop_cites_and_drops_bad_refs():
    r = llm_answer("fire rating of door 2?", Project(PROJ), {"kind": "openai", "base": "x", "key": "k", "model": "m", "env": "OPENAI_API_KEY"}, http=fake_openai())
    assert r["status"] == "answered" and "[1]" in r["answer"] and "[99]" not in r["answer"]
    assert r["citations"][0]["kind"] == "schedule_cell" and r["citations"][0]["bbox"]


def test_uncited_answer_is_withheld():
    r = llm_answer("how many doors?", Project(PROJ), {"kind": "anthropic", "base": "x", "key": "k", "model": "m", "env": "ANTHROPIC_API_KEY"}, http=fake_anthropic_no_cite())
    assert r["status"] == "abstain"


if __name__ == "__main__":
    test_openai_loop_cites_and_drops_bad_refs(); test_uncited_answer_is_withheld(); print("ok")
