<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: TERVEZETT — a kód még nem létezik. Amikor elkészül, a doc-keeper
  ügynök frissíti a valós fájlnevekre és függvényekre.
-->

# training — loss, tanító ciklus, TensorBoard

## Cél

A U-Net tanítása a train spliten, validálás minden epoch után, a legjobb
(val mIoU) modell mentése és minden fontos érték naplózása TensorBoardba.

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `unet_pet/losses.py` | `DiceLoss` (soft, multiclass), `CombinedLoss` (CE + w·Dice) |
| `unet_pet/train.py` | `train_one_epoch`, `validate`, `fit(cfg)`, CLI `main()` |

## Belépési pontok

- `fit(cfg, model=None, loaders=None) -> dict` — teljes tanítás, visszaadja a history-t
- `train_one_epoch(model, loader, loss_fn, optimizer, device) -> dict`
- `validate(model, loader, loss_fn, device) -> dict`
- CLI: `python -m unet_pet.train --config configs/default.yaml [--epochs N] [--limit-batches K] [--run-name NAME]`

## TensorBoard naplózás (`runs/<YYYYmmdd-HHMMSS>_<run-name>/`)

- `loss/train`, `loss/val`
- `metrics/val_mIoU`, `metrics/val_dice`, `metrics/val_pixel_acc`, `metrics/val_IoU_<osztály>`
- `lr`
- `images/val_predictions` — képrács: kép | valódi maszk | predikció, fix val mintákon
- `add_hparams` a futás végén (konfig + legjobb val mIoU)
- a felhasznált konfig mentése a run mappába (`config.yaml`)

## Checkpointolás

- `checkpoints/last.pt`: modell + optimizer + epoch (folytatáshoz)
- `models/unet_pet_best.pt`: **csak `state_dict`**, a legjobb val mIoU-nál

## Függőségek

Támaszkodik: data, model, metrics (evaluation), config-utils, `tensorboard`, `tqdm`.
Rá támaszkodik: notebook.

## Buktatók

- `model.train()` / `model.eval()` + `torch.no_grad()` a validálásnál.
- Dice loss a **softmax** valószínűségeken számolódjon, one-hot célmaszkkal.
- A `--limit-batches` kapcsoló kell a gyors smoke-teszthez (RALPH, tesztek).
- A legjobb súlyt a **val** alapján válaszd, ne a test alapján.
- A `SummaryWriter`-t zárd le (`close()`), különben hiányozhatnak logok.

## Hogyan teszteld

```
pytest tests/test_losses.py tests/test_train.py -q
python -m unet_pet.train --epochs 1 --limit-batches 2 --run-name smoke
```
Smoke-teszt: hamis mini-datasettel 1 epoch lefut, keletkezik TensorBoard event
fájl és checkpoint (ideiglenes mappában).

<!--
KARBANTARTÁS: ha ezt a modult módosítod a kódban, frissítsd ezt a docot
UGYANABBAN a commitban (lásd AGENTS.md → Git és commit).
-->
