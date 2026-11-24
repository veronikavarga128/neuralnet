<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: RÉSZBEN KÉSZ — metrics.py implementálva, tesztekkel; evaluate.py és
  visualize.py TERVEZETT.
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
| `unet_pet/visualize.py` | `show_predictions`, `overlay_mask`, `plot_confusion_matrix`, `make_tb_grid` — TERVEZETT |

## Belépési pontok

- `evaluate(model, loader, device)` → `{"mIoU", "dice", "pixel_acc", "iou_per_class", "confusion_matrix", "per_image_iou"}`
- CLI: `python -m unet_pet.evaluate --weights models/unet_pet_best.pt [--split test]`
- `show_predictions(images, masks, preds, n=6)` → matplotlib ábra

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
