<!-- NAPLÓ-RÉTEG: a RALPH PLAN módja írja, a BUILD mód pipálja.
     Formátum: KIZÁRÓLAG checkbox-lista, mert a loop.sh a "- [ ]" sorokat számolja.
     Ez egy kézzel írt KEZDŐ terv; a `./ralph/loop.sh plan` felülírhatja, utána nézd át. -->

# IMPLEMENTATION_PLAN

- [ ] Projekt-váz: `unet_pet/` csomag üres modulokkal, `tests/`, `configs/default.yaml`, `pyproject.toml` (ruff + pytest), `requirements.txt` és `requirements-dev.txt` pontos verziókkal, kiegészített `.gitignore` (spec: 01, 06) — kész, ha `pytest -q` (0 teszt is OK) és `ruff check .` lefut
- [ ] `config.py` + `utils.py`: `load_config`, `set_seed`, `seed_worker`, `get_device` (spec: 01, modul: config-utils) — kész, ha a seed- és konfig-tesztek zöldek
- [ ] `data.py`: dataset wrapper, szinkron transzformációk, címke-leképezés, seedelt split, `make_dataloaders` (spec: 02, modul: data) — kész, ha a letöltés nélküli adat-tesztek zöldek
- [ ] `sample_data/`: 6-8 kép + maszk kiválasztása a datasetből és `load_sample_data` (spec: 02) — kész, ha < 2 MB és a betöltő teszt zöld
- [ ] `model.py`: `DoubleConv`, `Down`, `Up`, `UNet`, `count_parameters` (spec: 03, modul: model) — kész, ha az alak- és overfit-tesztek zöldek
- [ ] `metrics.py`: konfúziós mátrix alapú IoU/mIoU/Dice/pixel acc., `SegmentationMeter` (spec: 05, modul: evaluation) — kész, ha a kézzel számolt tesztek zöldek
- [ ] `losses.py`: `DiceLoss`, `CombinedLoss` (spec: 04, modul: training) — kész, ha a loss-tesztek zöldek (tökéletes predikció ≈ 0 Dice loss)
- [ ] `visualize.py`: `denormalize`-ra épülő ábrák, overlay, TensorBoard képrács, konfúziós mátrix ábra (spec: 05, modul: evaluation) — kész, ha a függvények hamis adaton hiba nélkül ábrát adnak
- [ ] `train.py`: `train_one_epoch`, `validate`, `fit`, checkpoint, history JSON, TensorBoard, CLI (spec: 04, modul: training) — kész, ha a smoke-teszt (1 epoch, 2 batch, tmp mappa) zöld
- [ ] `evaluate.py`: teszt kiértékelés, baseline, legjobb/legrosszabb képek, JSON mentés, CLI (spec: 05, modul: evaluation) — kész, ha a CLI hamis súllyal `--limit 2` mellett lefut
- [ ] EMBERI LÉPÉS (nem RALPH): teljes tanítás GPU-n / Colabon, `models/unet_pet_best.pt` és a TensorBoard eredmények mentése; 1-2 összehasonlító kísérlet (pl. csak CE vs. CE+Dice)
- [ ] Notebook: `unet_pet_segmentation.ipynb` a `ai-context/modules/notebook.md` szerkezetével, `TRAIN = False` alapértékkel, magyar magyarázatokkal (spec: 06, modul: notebook) — kész, ha `nbconvert --execute` hibamentes
- [ ] README: leírás, dataset link, telepítés, futtatás, eredménytábla, mappaszerkezet (spec: 06) — kész, ha minden parancs a README-ből bemásolva működik
- [ ] Beadás előtti ellenőrzés: friss venv-ben telepítés + notebook futtatás, internet nélküli futás, fájlméretek, nincs TODO a notebookban (spec: 06) — kész, ha a spec összes elfogadási kritériuma teljesül
