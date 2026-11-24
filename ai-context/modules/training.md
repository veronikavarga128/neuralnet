<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: KÉSZ — losses.py és train.py implementálva, tesztekkel.
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
| `unet_pet/train.py` | `train_one_epoch`, `validate`, `fit(cfg, ...)`, CLI `main()`, `HISTORY_KEYS` — KÉSZ |
| `tests/test_train.py` | smoke-tesztek: kis modell, hamis adat, minden `tmp_path`-ban |

## Belépési pontok

- `fit(cfg, *, model=None, loaders=None, device=None, run_name="", limit_batches=None,
  export=True, resume=None, progress=True) -> dict` — visszaad: `history`,
  `best_val_mIoU`, `best_epoch`, `run_dir`, `checkpoint_dir`. A modell / loaderek /
  eszköz beinjektálható (tesztek).
- `train_one_epoch(model, loader, loss_fn, optimizer, device, *, scaler, amp, limit_batches)`
  → `{"loss", "ce", "dice"}` (mintaszámmal súlyozott átlag)
- `validate(model, loader, loss_fn, device, num_classes, *, amp, limit_batches)` →
  `{"loss", "ce", "dice_loss", "mIoU", "dice", "pixel_acc", "iou_per_class", ...}`
- CLI: `python -m unet_pet.train [--config F] [--epochs N] [--limit-batches K] [--run-name NAME]
  [--resume checkpoints/<run>/last.pt] [--no-export] [--device cpu] [--set key=value ...]`
  - `--set train.dice_weight=0` → tetszőleges konfig-felülírás (pl. a „csak CE” kísérlet)
  - `--limit-batches` (CLI-ből és a `fit()` közvetlen hívásakor is) → **nincs export**, így
    egy gyors próba sosem írhatja felül a beadandó súlyokat; `--limit-batches 0` → hiba

## Kimenetek (run azonosító: `<YYYYmmdd-HHMMSS>_<run-name>`)

- `runs/<run>/`: TensorBoard eventek, `config.yaml`, `history.json`
- `checkpoints/<run>/last.pt`: **önálló** folytatási pont — model, optimizer, scheduler,
  scaler, epoch, history, a legjobb súlyok (`best_model`), `limit_batches`, `config`.
  `checkpoints/<run>/best.pt`: a legjobb epoch `state_dict`-je (kényelmi másolat).
- Folytatás (`--resume <út>/last.pt`):
  - a `train.epochs` az **összes** epoch száma; ugyanabba a `runs/<run>`-ba ír tovább,
    a checkpointokat pedig a megadott `last.pt` mellé (áthelyezett fájlból is működik);
  - epochonkénti újraseedelés (`seed + epoch`, a train loader generátora is) → a
    folytatott futás **azonos** a megszakítás nélkülivel (CPU-n tesztelve);
  - ha a checkpoint `--limit-batches` futásból jön → nincs export; konfig-eltérésnél
    figyelmeztet; ha nincs hátralévő epoch → üzenet, és semmit nem ír;
  - új konfig a run mappában: `config_resume_epoch<N>.yaml`.
- Export (ha `export=True` és nincs `--limit-batches`): `models/unet_pet_best.pt`
  (**csak `state_dict`**) és `models/unet_pet_history.json` (+ `best_epoch`, `run_id`) —
  mindkettő gitbe kerül, a notebook `TRAIN = False` módban ezekből dolgozik.
- History formátum: epochonkénti listák, kulcsok `HISTORY_KEYS`: `train_loss`, `train_ce`,
  `train_dice`, `val_loss`, `val_ce`, `val_dice_loss`, `val_mIoU`, `val_dice`,
  `val_pixel_acc`, `lr` (a `visualize.plot_history` ezt rajzolja).

## Tanítási részletek

- Adam (`lr`, `weight_decay`), ReduceLROnPlateau `mode="max"` a val mIoU-n
  (`scheduler_factor`, `scheduler_patience`); `scheduler: none` → kikapcsolva.
- AMP (`train.amp`) csak CUDA-n kapcsol be (float16 autocast + GradScaler).
- Képrács: a val split első `num_log_images` mintája, minden `log_images_every` epochban
  (ugyanazok a képek → látszik a fejlődés).
- Mért sebesség (CPU, 128×128, batch 16): ~1,4 s / train batch → ~5 perc / epoch;
  a teljes 25 epoch CPU-n ~2+ óra, ezért GPU / Colab ajánlott.

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

- `loss/train`, `loss/val`, `loss/train_ce`, `loss/train_dice`
- `metrics/val_mIoU`, `metrics/val_dice`, `metrics/val_pixel_acc`, `metrics/val_IoU_<osztály>`
- `lr`
- `images/val_predictions` — képrács: kép | valódi maszk | predikció, fix val mintákon
- `add_hparams` a futás végén (konfig + legjobb val mIoU)
- a felhasznált konfig mentése a run mappába (`config.yaml`)

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
