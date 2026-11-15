# CubiCasa5K — key-free OCR baseline (bp/sources/cubicasa.py)

300 sampled images in the Junction subset, split by image (md5): 139 dev / 161 test, 2574 dev / 2876 test questions.

Method (CPU only, no key):
- RapidOCR (ONNX) on the raster, 2x upscale for small images, tokens cached in data/jcache/cubi/.
- Room labels classified with a Finnish lexicon (MH → bedroom, KH/KPH/PH/PESUH/SH → bathroom, WC, S/SAUNA, K/KK/KT/AVOK → kitchen, PARVEKE/PARV/LASIT.PARV → balcony),
  plus a few frequent OCR misreads of MH (NH, MHE, HM, MN).
- The apartment-type summary ("4H+K+S") is a second source: bedrooms = H − 1 (living room included in H); "+S" ⇒ sauna; "+K/KK/KT" ⇒ kitchen.
  Labels that agree with the summary are shown green.
- Every answer carries the OCR boxes it used. Rules that were unreliable on **dev** abstain:
  storey counts (36–76 % on dev), "no label ⇒ 0" for sauna/WC (31 % / 52 % on dev), bedroom counts from labels alone (57 % on dev).
- Symbol counts (doors, windows, WC seats, sinks, showers/baths) always abstain: they need a symbol detector or a grounded VLM.
  `bp/sources/vlm.py` is key-pluggable (BP_VLM=anthropic|openai|xai + key); it asks for JSON boxes twice and abstains unless both passes agree. **Not run: no key.**

## Results

Dev (tuned on): see dev.txt — ALL acc|answered 93.0 %, abstain 86.6 %.

Held-out test, **first and only run** (test_firstrun.txt):

| kind | n | acc | answered | acc \| answered | abstain |
|---|---|---|---|---|---|
| balcony | 110 | 43.6 % | 54 | 88.9 % | 50.9 % |
| bathroom | 154 | 35.1 % | 54 | 100.0 % | 64.9 % |
| bedroom | 202 | 19.8 % | 50 | 80.0 % | 75.2 % |
| kitchen | 220 | 40.9 % | 94 | 95.7 % | 57.3 % |
| sauna | 230 | 26.1 % | 74 | 81.1 % | 67.8 % |
| wc | 146 | 2.7 % | 8 | 50.0 % | 94.5 % |
| room kinds total | 1062 | 27.9 % | 334 | 88.6 % | 68.5 % |
| storey | 322 | 0 % | 0 | — | 100 % |
| door / window / wc_seat / sink / shower_bath | 1492 | 0 % | 0 | — | 100 % |
| **ALL** | 2876 | 10.3 % | 334 | 88.6 % | 88.4 % |

Precision drops from 93 % (dev) to 88.6 % (test): bedroom 90 → 80 %, sauna 85 → 81 %, wc 100 → 50 % (n=8).
Coverage on CubiCasa needs a VLM or a trained symbol detector (CubiCasa5K ships SVG annotations, but they are not in this subset).

---

## Phase 1 (2026-11-14): storeys + symbol counts on the full image set (CPU, no key, no model weights downloaded)

The full CubiCasa image set is now local: 4,787 images, split by image (md5), **2,292 dev / 2,495 test**. There are no SVG annotations
in the dataset, so dev-side calibration used (a) the dev gold *counts* and (b) ~700 symbol crops I hand-labelled on dev label sheets
(`scripts/cubi/seed_labels.json`, `scripts/cubi/label_sheet.py`, `active.py`). Nothing gold-derived is read at answer time; the
answer path uses only pixels + OCR + the two dev-trained models in `bp/models/`.

**Storeys** (`bp/sources/cubi_cv.py`, `cubicasa.storey_answer`)
- Recursive XY-cut of the ink mask splits the sheet into separate drawings. A row or column counts as a gap when its ink is below
  {0, 8, 15, 25} % of the plan's median density, because dimension lines cross the gaps. Bright saturated strokes (green sheet
  frames, highlighter) are ignored.
- Features: drawing count per gap threshold, footprint similarity of the two largest drawings, ink shares, and how many drawings carry
  ≥ 2 OCR room labels. OCR storey words are added too (`1. KRS`, `YLÄKERTA`, `ALAKERTA`, `KELLARI`, …).
- A gradient-boosted classifier is trained on dev. It answers only when (i) its probability is ≥ τ = 0.80 (chosen out-of-fold on dev
  for ≥ 95 % precision) and (ii) the predicted count equals the number of drawings actually found. Each drawing is an evidence box.
- Dev, 5-fold out-of-fold: 91.7 % raw (majority "1" = 80.2 %); gated **95.1 % precision at 87.8 % answered**.

**WC seats** (`bp/sources/cubi_sym.py`)
- Scale = median height of OCR'd room labels.
- Candidates = enclosed white regions of plausible size (the bowl), dropping holes inside OCR'd text ('O', '6').
- Descriptor = shape numbers plus rotation-pooled HOG of the ink patch. A HistGradientBoosting classifier (other / WC / sink /
  door-region) is trained on the seed labels plus count-consistent pseudo-labels: on a dev plan with gold c, a confident top-c
  becomes positive, and gold 0 gives hard negatives.
- Count = detections after NMS. A second, image-level gate gets the count, score margins, scale, and OCR context (bathroom / WC /
  kitchen labels, apartment type). It is trained on out-of-fold stage-1 scores, so τ is calibrated on held-out dev images.
- Dev out-of-fold: raw exact count 45 %; gated **90.4 % precision at 29 % answered**. Every counted seat is a box.

**Tried, not good enough → still abstain**
- Sinks: same pipeline; raw exact count ≤ 14 % on dev, mostly under-counting (kitchen sinks inside counters, tiny basins on
  low-resolution scans). No gate reached 90 %.
- Doors:
  - Closed-region candidates miss most swings, because the arcs leave gaps.
  - A dedicated arc + leaf template detector (`bp/sources/cubi_door.py`: chamfer-style coverage, empty interior, thin-stroke ring
    test, 8 orientations × 9 radii) reaches 17 % exact / 35 % within ±1 on dev.
  - No gate reached 90 %, so doors abstain. The detector and its synthetic unit test stay in the repo for phase 3 or a future
    learned verifier.
- Windows, showers/bathtubs: not attempted beyond inspection.

### Results

Dev (models fit on all dev, so in-sample; the honest dev numbers are the out-of-fold ones above): `dev_full.txt`.

Held-out test, **first and only run** (`test_firstrun_full.txt`; nothing was changed after it):

| kind | n | acc | answered | acc \| answered | abstain |
|---|---|---|---|---|---|
| storey | 4990 | 85.2 % | 4456 | **95.4 %** | 10.7 % |
| wc_seat | 4984 | 28.0 % | 1526 | **91.3 %** | 69.4 % |
| room kinds (balcony, bathroom, bedroom, kitchen, sauna, wc) | 16186 | 28.5 % | 5138 | 89.6 % | 68.3 % |
| door / window / sink / shower_bath | 18190 | 0 % | 0 | — | 100 % |
| **ALL** | 44350 | 23.1 % | 11120 | 92.2 % | 74.9 % |

Per room kind on the full test split: balcony 94.1 %, bathroom 92.2 %, bedroom 87.4 %, kitchen 95.6 %, sauna 82.6 %, wc 70.2 % when
answering. These are the old OCR rules scored for the first time on the 2,195 images that were not in the 300-image subset.

Test precision matches the dev out-of-fold estimates (storey 95.1 → 95.4 %, WC seats 90.4 → 91.3 %).

Reproduce:
1. `scripts/cubi/ocr_all.py --split dev|test --shard k/n`
2. `scripts/cubi/precompute.py`
3. `scripts/cubi/feat_table.py dev` + `train_storey.py 0.95`
4. `desc_cache.py dev` + `door_cache.py dev` + `train_sym.py`
5. `eval/cubi_eval.py --split dev|test`

---

## Phase 3 (2026-11-14): room-kind counts (bedroom, bathroom, WC, sauna, kitchen, balcony)

Code: `bp/sources/cubi_rooms.py`, `cubicasa.ocr2` / `room_answer`, `scripts/cubi/train_rooms.py`, model `bp/models/cubi_rooms.pkl`.

**What changed**
- **OCR v2.** An upright pass plus 90° and 270° rotated passes, with boxes mapped back to the image. Finnish plans often write labels
  vertically: `MH1/MH2/MH3`, `PH`, `WC`. Tokens down to RapidOCR score 0.1 are kept, because single letters like `K` and `S` are read
  correctly but scored 0.2–0.5, below the default 0.5 cutoff. An orientation check rejects stacked-letter misreads (`OH` upright →
  `HO` in a rotated pass). Two images crash RapidOCR on the rotated passes and keep the v1 tokens.
- **Lexicon.** Built from dev token/gold statistics, keeping only codes whose meaning is clear:
  - added `PSH`, `PESU`, `KYLPYH`, `TUPAK`, `TPK`, `KEITT…`, the Swedish `SOVRUM`/`KÖK`/`BADRUM`/`BASTU`/`BALKONG`, and `MH` misreads `HM`/`HW`/`HV`.
    `HW` is `MH` read upside down.
  - combo labels (`KH/WC`, `S/PH`, `OH+KK`, `KHH+PH`) count once per target kind when every part is a known room code.
  - `TK` is not a kitchen: it is often *tuulikaappi* (vestibule).
  - area lists (`OH+KK 26.0 m²`) and junk (`0 9k21`) are dropped.
- **De-duplication.** Same-kind hits within ~2.5 label heights count once.
- **Summary parser.** `3H+K+S`, `3h+kt+s`, `3H-K-S`, `2 H + KK`, `3r+k`.
- **Gate.** Candidates per (image, kind) are the strong label count, the all-label count (incl. weak letters), "presence" = 1, and the
  summary value. A HistGradientBoosting classifier scores P(candidate = gold) from 36 count, confidence, layout and summary-agreement
  features. τ per kind is chosen on 5-fold out-of-fold dev predictions grouped by image, targeting 92 % precision (τ ≥ 0.6).
  Every answer needs evidence: labels, the summary, or for a 0 the other room labels that were read.

**Dev, out-of-fold** (`train_rooms_dev_oof.txt`):
- OCR v1 with the same lexicon and gate: 33.8 % answered, 94.9 % precision.
- **OCR v2: 51.6 % answered, 94.9 % precision.** OCR preprocessing was the main lever.

**Test, first and only run** (`test_firstrun_rooms_v2.txt`, scored once with commit 64d5b5b; nothing changed after it except
display text and green cross-check colouring):

| kind | n | acc | answered | acc \| answered | phase 1 acc \| answered (answered) |
|---|---|---|---|---|---|
| balcony | 1492 | 40.2 % | 622 | 96.5 % | 94.1 % (648) |
| bathroom | 2436 | 51.1 % | 1324 | 94.1 % | 92.2 % (902) |
| bedroom | 3098 | 27.1 % | 886 | 94.8 % | 87.4 % (858) |
| kitchen | 3386 | 77.1 % | 2670 | 97.8 % | 95.6 % (1418) |
| sauna | 3434 | 33.7 % | 1262 | 91.8 % | 82.6 % (1124) |
| wc | 2340 | 41.2 % | 1066 | 90.4 % | 70.2 % (188) |
| **room kinds** | 16186 | **45.8 %** | 7830 (48.4 %) | **94.7 %** | 89.6 % (31.7 %) |

**Caveats**
- Kitchen gold is 1 on 98 % of plans, so its high precision is easy.
- Answers of 0 rest on absence of a label.
- Saunas without a text label (bench symbol only) are missed. A brown-pixel bench cue was tried on dev and was too weak.
- Bedrooms remain the hardest: 29 % answered (886 of 3,098). The gate needs labels or the summary, and the two often disagree when OCR misses an `MH`.

---

## Phase A (2026-11-14): fixtures from room labels, WC-seat fusion, room gate v3 (BRAINSTORM #1–#4, τ rule #6)

Code: `bp/sources/cubi_fix.py` + `scripts/cubi/train_fix.py` (model `bp/models/cubi_fix.pkl`), `bp/sources/calib.py`,
`cubi_rooms.code_feats` and the washroom → sauna candidate, `scripts/cubi/train_rooms.py` (CP rule is the default; `--point` gives the old rule).

**τ rule.** For every gate trained in this phase, τ is the max-coverage threshold where the one-sided Clopper–Pearson 95 % lower bound on dev
out-of-fold precision is ≥ 90 %.

**Fixtures.** Candidates are room-label formulas, plus the detector count for WC seats:
- showers/baths: n_bath, max(1, n_bath), n_bath with a sauna washroom
- sinks: kitchen + bathrooms + WCs (combo labels once), +1, kitchen + bathrooms
- WC seats: bathrooms + WCs, max(1, ·), the WC-symbol detector count

Features add the code identity (KH vs PH, combos, KHH), the summary flags, and the detector's gate features. The detector features come
from stage-1 out-of-fold scores. The gate is a HistGradientBoosting classifier with 5-fold out-of-fold training grouped by image.
A candidate with no box to show is dropped. "≥ 1" priors with no matching label are kept only with the read room labels as boxes and an
answer text that says it is a prior.

| kind | dev OOF answered @ precision (LB) | test first run answered @ precision | before |
|---|---|---|---|
| shower_bath | 45.4 % @ 91.7 % (90.1 %) — 626 via a bathroom label @ 92.0 %, 345 prior-only @ 91.0 % | 45.3 % @ 94.2 % | 0 % |
| sink | 23.4 % @ 92.2 % (90.1 %) | 20.8 % @ 91.8 % | 0 % |
| wc_seat | 49.7 % @ 91.7 % (90.3 %); shipped detector gate under the same rule: 26.2 % @ 92.2 % | 50.0 % @ 92.4 % | 30.6 % @ 91.3 % |

Observed: the shower gate only ever answers 1 and the sink gate only answers 2, so in practice they recognise "simple one-bathroom flat"
situations. Robust to a different split (10-fold, other seed): 46.4 % / 19.1 % / 46.7 % answered at 91.6 / 92.8 / 92.1 %.

**Room gate v3 (#4).** Code-identity features (KH-type vs PH-type washrooms, combos, KHH, 'SAUNA' word vs single 'S') and a sauna candidate
"1 when a PH/PESUH washroom is read and no sauna label". Dev OOF under the CP rule:
- sauna 39.9 % @ 92.2 % → 51.9 % @ 91.9 % (122 washroom-sourced answers @ 96.7 %)
- all room kinds 51.6 % @ 94.9 % → 55.3 % @ 94.6 % (old features under the same rule: 51.6 % @ 94.9 %)

**Test, first and only run** (`test_firstrun_phaseA.txt`, commit 6f44962): CubiCasa ALL **39.3 %** accuracy, 94.2 % when answering
(was 29.5 % / 94.6 %). Room kinds 49.2 % / 94.3 % (was 45.8 % / 94.7 %). Sauna 91.7 % at 49.0 % answered (was 36.8 %). The separate-WC room
kind came out at 89.7 % (was 90.4 %): τ moved 0.88 → 0.87 under the CP rule.
