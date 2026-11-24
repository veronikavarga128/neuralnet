<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: RÉSZBEN KÉSZ — losses.py implementálva, tesztekkel; train.py TERVEZETT.
-->

# training — loss, tanító ciklus, TensorBoard

## Cél

A U-Net tanítása a train spliten, validálás minden epoch után, a legjobb
(val mIoU) modell mentése és minden fontos érték naplózása TensorBoardba.

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `unet_pet/losses.py` | `DiceLoss` (soft, multiclass, batch-Dice), `CombinedLoss` (CE + w·Dice), `build_loss(cfg)` — KÉSZ |
| `tests/test_losses.py` | loss-tesztek (tökéletes → 0, rossz → ≈1, kézzel számolt eset) |
| `unet_pet/train.py` | `train_one_epoch`, `validate`, `fit(cfg)`, CLI `main()` — TERVEZETT |

## Belépési pontok

- `fit(cfg, model=None, loaders=None) -> dict` — teljes tanítás, visszaadja a history-t
- `train_one_epoch(model, loader, loss_fn, optimizer, device) -> dict`
- `validate(model, loader, loss_fn, device) -> dict`
- CLI: `python -m unet_pet.train --config configs/default.yaml [--epochs N] [--limit-batches K] [--run-name NAME]`

## Loss (losses.py)

- `build_loss(cfg)` → `CombinedLoss(dice_weight=cfg.train.dice_weight)`; `dice_weight=0` → tiszta CE
  (a „csak CE vs. CE+Dice” kísérlethez).
- `CombinedLoss.forward` a teljes loss tenzort adja; a részek `loss_fn.last_parts =
  {"ce": float, "dice": float}`-ban (logoláshoz, pl. `loss/train_ce`, `loss/train_dice`).
- `DiceLoss(smooth=1.0)`: softmax valószínűség × one-hot cél, összegzés a batch + pixelek
  mentén osztályonként (batch-Dice; a vékony határ osztály képenként zajos lenne), majd
  `1 - mean_c Dice_c`. A `smooth` miatt hiányzó osztályra 0, nem NaN.
- Indoklás (a modul docstringjében is): CE stabil pixelenkénti gradiens, de a háttér
  dominál; Dice osztályonként egyenlő súllyal méri az átfedést → kiegyensúlyozza.

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
