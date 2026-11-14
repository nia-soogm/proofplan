"""Headless screenshots of the running app -> shots/. Usage: python scripts/shot_proofplan.py [base_url]

Needs the server running with BP_JDATA set. No dataset question text is stored here: each case picks the sample
question from the UI chips (loaded from your local questions.jsonl) by keywords, then hides the chip list and the
gold line so the screenshot shows only the one question that was asked."""
import sys, asyncio, os
from pathlib import Path
from playwright.async_api import async_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = Path(__file__).resolve().parent.parent / "shots"
SW, IFC, OD, CC, IB = "jd:swiss_dwellings", "jd:ifc_models", "jd:opening_design", "jd:cubicasa5k", "jd:ifc_bench"
CHROME = "/usr/bin/google-chrome"  # used when present; otherwise Playwright's bundled Chromium
# (source, file, keywords that pick the chip, evidence ref to click, output name)
CASES = [
    (SW, "site3411_bldg5284_floor23591", ["net floor area of apartment A"], None, "07_swiss_apartment_area_polygons"),
    (SW, "site3387_bldg5238_floor23498", ["How many doors"], None, "08_swiss_door_count_symbols"),
    (IFC, "tuwien-escape-route_real-world-escape-route-test-model-zdb_plan-02-ug1", ["'Garage'"], None, "09_ifc_room_area_model_vs_plan_green"),
    (IFC, "rwth-digitalhub_digitalhub-fm-arc-v2_plan-02-e00-okrd", ["doors"], 5, "10_ifc_plan_door_count_guid_boxes"),
    (IFC, "rwth-digitalhub_digitalhub-fm-arc-v2_plan-02-e00-okrd", ["Eingangsbereich"], None, "11_ifc_red_model_plan_disagree"),
    (OD, "A700 - SCHEDULES", ["U02"], 2, "13_opening_schedule_cell_evidence"),
    (CC, "high_quality_architectural-9005", ["bedrooms"], None, "14_cubicasa_ocr_labels_green"),
    (CC, "high_quality_architectural-9005", ["windows"], None, "15_cubicasa_symbols_abstain"),
    (CC, "high_quality_architectural-13313", ["WC rooms"], None, "16_cubicasa_room_gate_wc"),
    (CC, "high_quality_architectural-524", ["saunas"], None, "17_cubicasa_sauna_label_and_summary"),
    (CC, "high_quality_architectural-13313", ["storeys"], None, "18_cubicasa_storey_layout"),
    (CC, "high_quality_architectural-13313", ["WC seats"], None, "19_cubicasa_wc_seat_symbol"),
    (IB, "smiley_west_arc", ["doors", "by type"], None, "20_ifc_bench_guid_boxes"),
]
PICK = """kw => { const c = [...document.querySelectorAll('.chip')].find(c => kw.every(k => (c.dataset.q || '').includes(k)));
                  return c ? c.dataset.q : null; }"""


async def main():
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        kw = {"executable_path": CHROME} if os.path.exists(CHROME) else {}
        b = await p.chromium.launch(headless=True, args=["--no-sandbox"], **kw)
        pg = await b.new_page(viewport={"width": 1600, "height": 950})
        await pg.goto(BASE + "/")
        await pg.wait_for_selector(".card")
        await pg.wait_for_timeout(800)
        await pg.screenshot(path=str(OUT / "12_landing_accuracy.png"), full_page=True)
        print("saved landing")
        for proj, f, keys, cite, name in CASES:
            await pg.goto(f"{BASE}/?p={proj}&f={f}")
            await pg.wait_for_selector("#file option", state="attached")
            await pg.wait_for_timeout(2500)
            q = await pg.evaluate(PICK, keys)
            if not q:
                print("skip", name, "(no sample question matches", keys, ")"); continue
            await pg.evaluate("document.querySelector('#chips').innerHTML = ''")  # no dataset questions / gold in the shot
            await pg.fill("#q", q)
            await pg.click("#go")
            await pg.wait_for_function("document.querySelector('#answer .badge')", timeout=120000)
            await pg.wait_for_timeout(2500)
            if cite:
                await pg.click(f"a.cite[data-r='{cite}']")
                await pg.wait_for_timeout(1500)
            await pg.screenshot(path=str(OUT / f"{name}.png"))
            print("saved", name)
        await b.close()
asyncio.run(main())
