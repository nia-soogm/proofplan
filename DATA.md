# Data

**Nothing from the challenge dataset is redistributed in this repository**: no drawings, models, images, questions or gold answers.
Only aggregate scores (`eval/accuracy.json`, `eval/**/*.txt`) are published. The code, tests and docs work without the data. The app
and the evaluation scripts need it.

## Getting the data

ProofPlan was built on the **Junction 2026 "Make Blueprints Talk" challenge dataset**. The challenge organisers distribute it to
participants *for non-commercial use in the Junction challenge*. Ask the organisers for access. We cannot pass it on.

Unpack it and point `BP_JDATA` at the folder:

```bash
export BP_JDATA=/path/to/junction_data    # default if unset: ./data/jdata (git-ignored)
```

Expected layout (as shipped by the organisers):

```
$BP_JDATA/
  questions.jsonl            # one question per line: path, question, answer, ...
  data/swiss_dwellings/      # vector PDF + DXF per floor, LICENSE.md
  data/ifc_models/           # IFC models + rendered plan PDFs/SVGs, LICENSE.md (per file)
  data/ifc_bench/            # IFC models for free-text questions, LICENSE.md (per file)
  data/opening_design/       # rotated schedule sheets (PDF), LICENSE.md
  data/cubicasa5k/           # raster floor plans (PNG), LICENSE.md
```

The app reads each source's `LICENSE.md` and prints its credit line under every drawing.

## Licences per source

These are the licences of the underlying sources as listed in the dataset's `LICENSE.md` files. They apply on top of the
challenge terms.

| source | licence | upstream |
|---|---|---|
| Swiss Dwellings v3.0.0 | CC BY 4.0 | Archilyse AG, doi:10.5281/zenodo.7788422 (plans rendered by the dataset builders from that geometry) |
| IFC models / ifc_bench | per model: CC BY 4.0, MIT, KIT "unrestricted use", CC BY-ND 3.0 (buildingSMART Duplex and Medical-Dental Clinic: used unchanged, never rendered) | BIM-SPEED (TU Berlin), TU Wien escape-route model, RWTH DigitalHub, ResBIM, buildingSMART sample files, KIT IFC examples |
| OpeningDesign | CC BY-SA 4.0 (some projects MIT) | github.com/OpeningDesign |
| CubiCasa5K | CC BY-NC-SA 4.0 (non-commercial) | © 2019 CubiCasa, Kalervo et al., SCIA 2019 |

Full credits are in the README under "Data & licences".

## What is derived from the data and kept in the repo

- `bp/models/*.pkl` and `scripts/cubi/seed_labels.json` are CubiCasa-derived gates and hand labels, under CC BY-NC-SA 4.0 (see
  `bp/models/LICENSE.md`). They contain no images.
- `shots/*.png` are screenshots of our app showing excerpts of drawings with their credit line, under the licence of the drawing
  shown. The app's list of dataset questions and gold answers is blanked out in every screenshot.
- `eval/accuracy.json`, `eval/results_*.txt`, `eval/cubicasa/*.txt`, `eval/ifcb/*.txt` hold aggregate scores only.

Caches that ProofPlan writes (`data/jcache/`: OCR tokens, IFC facts, rendered pages, the app's question catalogue) contain dataset
content. They are git-ignored and must not be committed.
