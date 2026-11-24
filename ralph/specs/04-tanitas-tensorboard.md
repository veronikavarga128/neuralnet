# Tanítás és TensorBoard kísérletkövetés

## Cél
Stabil tanítás, és a tárgy által elvárt kísérletkövetés TensorBoarddal.

## Viselkedés
- Loss: CrossEntropy + Dice (súly a konfigból). Adam, ReduceLROnPlateau a val mIoU-n.
- Epochonként train → val; a legjobb val mIoU modell `models/unet_pet_best.pt`-be (state_dict).
- `checkpoints/last.pt` a folytatáshoz; a history JSON-ba is mentődik (a notebook ebből rajzol görbét).
- TensorBoard: loss/train, loss/val, val metrikák (osztályonként is), lr, predikció-képrács, hparams.
- CLI: `python -m unet_pet.train --config ... [--epochs] [--limit-batches] [--run-name]`.

## Elfogadási kritériumok
- [ ] `pytest tests/test_losses.py tests/test_train.py` zöld (hamis mini-datasettel, tmp mappában).
- [ ] `--epochs 1 --limit-batches 2` CPU-n < 1 perc alatt lefut, és létrejön a `runs/...` event fájl.
- [ ] TensorBoardban látszanak a skalárok és a képek.
- [ ] A teljes tanítás után a val mIoU ésszerű (célérték: > 0.70; csak tájékoztató).

## Nem cél
Elosztott tanítás, mixed precision kötelezően (opcionálisan bekapcsolható cudán), nagy hiperparaméter-keresés.
