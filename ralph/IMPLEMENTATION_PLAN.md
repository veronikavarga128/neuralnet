<!-- NAPLÓ-RÉTEG: a RALPH PLAN módja írja, a BUILD mód pipálja.
     Formátum: KIZÁRÓLAG checkbox-lista, mert a loop.sh a "- [ ]" sorokat számolja.
     Ez egy kézzel írt KEZDŐ terv; a `./ralph/loop.sh plan` felülírhatja, utána nézd át. -->

# IMPLEMENTATION_PLAN

- [x] Projekt-váz: `unet_pet/` csomag üres modulokkal, `tests/`, `configs/default.yaml`, `pyproject.toml` (ruff + pytest), `requirements.txt` és `requirements-dev.txt` pontos verziókkal, kiegészített `.gitignore` (spec: 01, 06) — kész, ha `pytest -q` (0 teszt is OK) és `ruff check .` lefut
- [x] `config.py` + `utils.py`: `load_config`, `set_seed`, `seed_worker`, `get_device` (spec: 01, modul: config-utils) — kész, ha a seed- és konfig-tesztek zöldek
- [x] `data.py`: dataset wrapper, szinkron transzformációk, címke-leképezés, seedelt split, `make_dataloaders` (spec: 02, modul: data) — kész, ha a letöltés nélküli adat-tesztek zöldek
- [x] `sample_data/`: 6-8 kép + maszk kiválasztása a datasetből és `load_sample_data` (spec: 02) — kész, ha < 2 MB és a betöltő teszt zöld
- [x] `model.py`: `DoubleConv`, `Down`, `Up`, `UNet`, `count_parameters` (spec: 03, modul: model) — kész, ha az alak- és overfit-tesztek zöldek
- [x] `metrics.py`: konfúziós mátrix alapú IoU/mIoU/Dice/pixel acc., `SegmentationMeter` (spec: 05, modul: evaluation) — kész, ha a kézzel számolt tesztek zöldek
- [x] `losses.py`: `DiceLoss`, `CombinedLoss` (spec: 04, modul: training) — kész, ha a loss-tesztek zöldek (tökéletes predikció ≈ 0 Dice loss)
- [x] `visualize.py`: `denormalize`-ra épülő ábrák, overlay, TensorBoard képrács, konfúziós mátrix ábra (spec: 05, modul: evaluation) — kész, ha a függvények hamis adaton hiba nélkül ábrát adnak
- [x] `train.py`: `train_one_epoch`, `validate`, `fit`, checkpoint, history JSON, TensorBoard, CLI (spec: 04, modul: training) — kész, ha a smoke-teszt (1 epoch, 2 batch, tmp mappa) zöld
- [x] `evaluate.py`: teszt kiértékelés, baseline, legjobb/legrosszabb képek, JSON mentés, CLI (spec: 05, modul: evaluation) — kész, ha a CLI hamis súllyal `--limit 2` mellett lefut
- [x] EMBERI LÉPÉS (nem RALPH) — KÉSZ: fő futás (CE+Dice, 25 epoch, CPU, val mIoU 0,7264, teszt 0,7371) exportálva; CE-only összehasonlítás (`20261125-094758_ce_only`, val 0,7318, teszt 0,7350) a `results/*_ce_only.json` fájlokban. Eredeti szöveg: teljes tanítás GPU-n / Colabon, `models/unet_pet_best.pt` és a TensorBoard eredmények mentése; 1-2 összehasonlító kísérlet (pl. csak CE vs. CE+Dice)
- [x] Notebook — KÉSZ (CE-only összevetés a 8–9. fejezetben; végleges `nbconvert --execute` futás kimenetekkel hibamentes, 2026-11-25): `unet_pet_segmentation.ipynb` a `ai-context/modules/notebook.md` szerkezetével, `TRAIN = False` alapértékkel, magyar magyarázatokkal (spec: 06, modul: notebook) — kész, ha `nbconvert --execute` hibamentes
- [x] README: leírás, dataset link, telepítés, futtatás, eredménytábla, mappaszerkezet (spec: 06) — kész, ha minden parancs a README-ből bemásolva működik
- [ ] Beadás előtti ellenőrzés: friss venv-ben telepítés + notebook futtatás, internet nélküli futás, fájlméretek, nincs TODO a notebookban (spec: 06) — kész, ha a spec összes elfogadási kritériuma teljesül
