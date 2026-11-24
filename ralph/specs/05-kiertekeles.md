# Kiértékelés és vizualizáció

## Cél
Objektív eredmény a teszt spliten és szemléletes ábrák a notebookhoz és a bemutatóhoz.

## Viselkedés
- Metrikák a teljes split konfúziós mátrixából: mIoU (fő), Dice, pixel accuracy, osztályonkénti IoU.
- Baseline: „mindig háttér” predikció ugyanezekkel a metrikákkal.
- Ábrák: kép | valódi maszk | predikció | overlay; legjobb/legrosszabb 3 kép képenkénti IoU alapján; normalizált konfúziós mátrix.
- CLI: `python -m unet_pet.evaluate --weights models/unet_pet_best.pt [--split test] [--limit N]`.

## Elfogadási kritériumok
- [ ] `pytest tests/test_metrics.py` zöld (kézzel számolt esetek).
- [ ] Az evaluate CLI kiírja a metrikatáblát és JSON-ba menti (`results/test_metrics.json`).
- [ ] A notebook eredménytáblája egyezik a JSON-nal.

## Nem cél
Hausdorff-távolság és egyéb haladó metrikák.
