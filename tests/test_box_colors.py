"""Evidence-box colours in the web UI match the legend: green = 2nd source agrees,
red = disagrees / not found, blue = unchecked. Renders app/static/index.html in headless
Chromium with stubbed API calls; skipped if Playwright or its browser is missing."""
import json
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
HTML = (Path(__file__).resolve().parent.parent / "app" / "static" / "index.html").read_text()
GREEN, RED, BLUE = "rgb(22, 163, 74)", "rgb(220, 38, 38)", "rgb(31, 94, 255)"


def _border_colors(project, sel="null"):
    stubs = {
        "/api/projects": {"engine": "test", "projects": [{"name": project, "label": project, "samples": []}]},
        "/api/accuracy": {"rows": [], "cubicasa_kinds": []},
    }
    try:
        p = pw.sync_playwright().start()
        b = p.chromium.launch()
    except Exception as e:  # browser not installed
        pytest.skip(f"no Chromium: {e}")
    try:
        pg = b.new_page()

        def route(r):
            path = "/" + r.request.url.split("/", 3)[3].split("?")[0]
            if path == "/":
                return r.fulfill(body=HTML, content_type="text/html")
            body = stubs.get(path, [] if path.startswith("/api/jdata/") else {})
            r.fulfill(body=json.dumps(body), content_type="application/json")

        pg.route("http://t/**", route)
        pg.goto("http://t/")
        pg.wait_for_function("document.querySelector('#project').options.length>0")
        return pg.evaluate("""sel0 => {
          sheets=[{page:0,width:200,height:200}]; curPage=0; sel=sel0;
          cites=[{ref:1,page:0,bbox:[10,10,20,20],verified:null},
                 {ref:2,page:0,bbox:[30,30,40,40],verified:true},
                 {ref:3,page:0,bbox:[50,50,60,60],verified:false},
                 {ref:4,page:0,bbox:[70,70,80,80]}];
          drawBoxes();
          return [1,2,3,4].map(i => getComputedStyle(document.getElementById('box'+i)).borderTopColor);
        }""", None if sel == "null" else sel)
    finally:
        b.close(); p.stop()


def test_dataset_boxes_follow_legend():
    assert _border_colors("jd:opening_design") == [BLUE, GREEN, RED, BLUE]


def test_selected_unchecked_box_uses_selection_colour():
    assert _border_colors("jd:opening_design", sel=1)[0] == "rgb(0, 179, 255)"
