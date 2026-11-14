"""Key-pluggable grounded-VLM backend for raster symbol counts (CubiCasa doors/windows/WC seats/sinks/showers).

Contract: the model must return JSON boxes [{"label": ..., "bbox": [x0,y0,x1,y1]}]; the count is the number of
boxes, never a free-text number, so every answer is backed by highlights. Two passes (full image + 2x2 tiles) must
agree or we abstain. Set BP_VLM=anthropic|openai|xai and the matching API key to enable. Nothing runs without a key.
"""
import base64, json, os, re
from pathlib import Path
from . import JDATA

PROMPT = ("You are counting symbols on a Finnish real-estate floor plan. Return ONLY JSON: a list of objects "
          "{{\"label\": \"{label}\", \"bbox\": [x0, y0, x1, y1]}} in pixel coordinates of the image, one per {what}. "
          "Do not count anything else.")
WHAT = {"door": "door leaf (a doorway with a door drawn; not open passages)", "window": "window",
        "wc_seat": "toilet bowl (WC seat)", "sink": "sink or wash basin (a double-bowl sink counts once)",
        "shower_bath": "shower or bathtub"}


def available():
    b = os.environ.get("BP_VLM")
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "xai": "XAI_API_KEY"}.get(b or "")
    return bool(key and os.environ.get(key))


def _call(img_b64, prompt):  # pragma: no cover - needs a key
    import requests
    b = os.environ["BP_VLM"]
    if b == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.environ.get("BP_VLM_MODEL", "claude-sonnet-4-5"), "max_tokens": 2000, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}, {"type": "text", "text": prompt}]}]})
        txt = r.json()["content"][0]["text"]
    else:
        url = {"openai": "https://api.openai.com/v1/chat/completions", "xai": "https://api.x.ai/v1/chat/completions"}[b]
        key = os.environ["OPENAI_API_KEY" if b == "openai" else "XAI_API_KEY"]
        r = requests.post(url, timeout=120, headers={"authorization": f"Bearer {key}"}, json={
            "model": os.environ.get("BP_VLM_MODEL", "gpt-4o" if b == "openai" else "grok-2-vision"),
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                                                      {"type": "text", "text": prompt}]}]})
        txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", txt, re.S)
    return json.loads(m.group()) if m else []


def count_symbols(path, kind):  # pragma: no cover - needs a key
    img = base64.b64encode((JDATA / path).read_bytes()).decode()
    boxes = [b for b in _call(img, PROMPT.format(label=kind, what=WHAT[kind])) if isinstance(b, dict) and len(b.get("bbox", [])) == 4]
    boxes2 = [b for b in _call(img, PROMPT.format(label=kind, what=WHAT[kind]) + " Be exhaustive; scan the plan region by region.")
              if isinstance(b, dict) and len(b.get("bbox", [])) == 4]
    if len(boxes) != len(boxes2):
        return {"status": "abstain", "answer": f"NOT SURE: two VLM passes disagree ({len(boxes)} vs {len(boxes2)}).", "value": None,
                "evidence": [{"bbox": b["bbox"], "page": 0, "text": f"VLM {kind}", "kind": kind, "verified": False} for b in boxes]}
    return {"status": "answered", "answer": f"{len(boxes)} {kind.replace('_', ' ')} symbols boxed by the VLM (two passes agree).", "value": len(boxes),
            "evidence": [{"bbox": b["bbox"], "page": 0, "text": f"VLM {kind}", "kind": kind, "verified": True} for b in boxes]}
