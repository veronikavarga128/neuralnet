<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: RÉSZBEN KÉSZ — metrics.py és visualize.py implementálva, tesztekkel;
  evaluate.py TERVEZETT.
-->

# evaluation — metrikák, kiértékelés, vizualizáció

## Cél

A betanított modell objektív mérése a hivatalos teszt spliten és az
eredmények szemléltetése ábrákkal (a notebook és a bemutató számára).

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `unet_pet/metrics.py` | `confusion_matrix`, `iou_per_class`, `mean_iou`, `dice_per_class`, `mean_dice`, `pixel_accuracy`, `normalize_confusion`, `per_image_iou`, `SegmentationMeter` (akkumulál batch-enként) — KÉSZ |
| `tests/test_metrics.py` | kézzel számolt példák (a levezetés a fájl docstringjében) |
| `unet_pet/evaluate.py` | `evaluate(model, loader, device) -> dict`, CLI — TERVEZETT |
| `unet_pet/visualize.py` | `mask_to_rgb`, `overlay_mask`, `show_predictions`, `make_tb_grid`, `plot_confusion_matrix`, `plot_history`, `CLASS_COLORS` — KÉSZ |
| `tests/test_visualize.py` | ábra-tesztek hamis adaton (Agg backend a `tests/conftest.py`-ban) |

## Belépési pontok

- `evaluate(model, loader, device)` → `{"mIoU", "dice", "pixel_acc", "iou_per_class", "confusion_matrix", "per_image_iou"}`
- CLI: `python -m unet_pet.evaluate --weights models/unet_pet_best.pt [--split test]`
- `show_predictions(images, masks, preds=None, *, mean, std, n=6, titles=None)` → `Figure`;
  soronként kép | valódi maszk | predikció | overlay (preds nélkül: kép | maszk | overlay).
  `titles` a sor bal oldalára (pl. név + képenkénti IoU a legjobb/legrosszabb képekhez).
- `make_tb_grid(images, masks, preds, *, mean, std, n=4)` → `(3, H', W')` tenzor a
  `SummaryWriter.add_image`-hez, ugyanazzal a 4 oszloppal
- `plot_confusion_matrix(cm, normalize=True)` → `Figure` (soronként normalizált, cellákban a számok)
- `plot_history(history)` → `Figure`; a history **epochonkénti listák dictje**: `train_loss`,
  `val_loss`, `val_mIoU`, `val_dice`, `lr` (a hiányzó kulcsokat kihagyja) — a `train.py`
  ezt a formátumot írja

## metrics.py — konvenciók

- Konfúziós mátrix: `cm[t, p]` = t valódi osztályú, p-nek jósolt pixelek száma (sor = valódi).
- Minden függvény elfogad logitot `(B, C, H, W)` vagy címkét `(B, H, W)`.
- `SegmentationMeter.update(preds, targets)` → `compute()` dict, sima Python értékekkel
  (JSON-ba írható): `mIoU`, `dice` (macro), `pixel_acc`, `iou_per_class`,
  `dice_per_class`, `confusion_matrix`, `per_image_iou`.
- 0/0 (az osztály se nem fordul elő, se nem jósolták) → NaN, az átlagok `nanmean`-nel
  kihagyják. Jósolt, de nem létező osztály → IoU = 0 (ez valódi hiba).
- `normalize_confusion`: soronként normalizál (recall-nézet), üres sor → 0, nem NaN.
- Baseline („mindig háttér”) a `sample_data/`-n: pixel acc 0,659, mIoU 0,220 → ezért a
  pixel accuracy félrevezető, a fő metrika a mIoU.

## visualize.py — konvenciók

- A képbemenet mindig a **normalizált** modellbemenet; a függvények `mean`/`std`
  alapján denormalizálnak. Predikció lehet logit `(B,C,H,W)` vagy címke `(B,H,W)`.
- Színek (`CLASS_COLORS`): kisállat narancs, háttér sötétszürke, határ sárga. Az overlay
  a háttérpixeleket érintetlenül hagyja.

## Mit kell kiértékelni (beadandó-követelmény)

- Teszt mIoU (fő metrika), Dice, pixel accuracy, osztályonkénti IoU.
- Konfúziós mátrix (normalizált) ábra.
- **Baseline** összevetés: „mindig háttér” predikció pixel accuracy-je és mIoU-ja,
  hogy látsszon, mennyit ad hozzá a modell.
- Kvalitatív: véletlen, legjobb 3 és legrosszabb 3 kép (képenkénti IoU alapján).
- Rövid hibaelemzés (pl. vékony határ osztály, hasonló színű háttér, kis állat a képen).

## Függőségek

Támaszkodik: `torch`, `numpy`, `matplotlib`, data (`denormalize`), model.
Rá támaszkodik: training (val metrikák), notebook.

## Buktatók

- A metrikát a **teljes split konfúziós mátrixából** számold, ne batch-átlagok átlagából.
- Ha egy osztály egy képen nem fordul elő és nem is jósolták, az IoU-ja
  definiálatlan (0/0) → képenkénti átlagnál hagyd ki (NaN-kezelés).
- Predikció: `logits.argmax(dim=1)`.

## Hogyan teszteld

```
pytest tests/test_metrics.py -q
```
Kézzel számolt kis tenzorokon: tökéletes predikció → IoU = 1; teljesen rossz →
0; ismert konfúziós mátrixra ismert mIoU.

<!--
KARBANTARTÁS: ha ezt a modult módosítod a kódban, frissítsd ezt a docot
UGYANABBAN a commitban (lásd AGENTS.md → Git és commit).
-->
