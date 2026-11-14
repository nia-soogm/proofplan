# Licence of the files in this folder (and `scripts/cubi/seed_labels.json`)

`cubi_rooms.pkl`, `cubi_storey.pkl` and `cubi_sym.pkl` are small scikit-learn gates (gradient-boosted classifiers + thresholds).
They were fitted on dev-split CubiCasa5K images and counts from the Junction 2026 challenge data. `scripts/cubi/seed_labels.json`
holds about 700 symbol boxes we labelled by hand on dev images. It stores image file names, box coordinates and class names, but
no pixels.

These files are derived from CubiCasa5K, so they are **not** covered by the repository's AGPL-3.0 licence. They are shared under
**CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/), non-commercial use only.

Credit: CubiCasa5K © 2019 CubiCasa. Kalervo, Ylioinas, Häikiö, Karhu, Kannala: "CubiCasa5K: A Dataset and an Improved Multi-Task
Model for Floorplan Image Analysis", SCIA 2019. Changes: models trained on the images and counts; no images are included.

If you delete these files, the CubiCasa solver falls back to its older OCR rules, which answer fewer questions.
To retrain them on your own copy of the data, run `scripts/cubi/train_rooms.py`, `train_storey.py` and `train_sym.py` (see `eval/cubicasa/NOTES.md`).

The `.pkl` files are Python pickles. Load only copies you trust.
