<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: KÉSZ — metrics.py, visualize.py és evaluate.py implementálva, tesztekkel.
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
| `unet_pet/evaluate.py` | `load_trained_model`, `make_eval_dataset`, `image_names`, `evaluate`, `select_examples`, `build_report`, `format_report`, `save_report`, `examples_figure`, `save_figures`, `run_evaluation`, CLI — KÉSZ |
| `tests/test_evaluate.py` | kézzel számolt baseline, egyezés a `SegmentationMeter`-rel, best/worst NaN-kezelés, szigorú JSON, rossz architektúra hibaüzenete, CLI `--split sample --limit 2` hamis súllyal |
| `unet_pet/visualize.py` | `mask_to_rgb`, `overlay_mask`, `show_predictions`, `make_tb_grid`, `plot_confusion_matrix`, `plot_history`, `CLASS_COLORS` — KÉSZ |
| `tests/test_visualize.py` | ábra-tesztek hamis adaton (Agg backend a `tests/conftest.py`-ban) |

## Belépési pontok

- CLI: `python -m unet_pet.evaluate [--weights models/unet_pet_best.pt] [--split test|val|sample] [--limit N] [--figures] [--device cpu] [--set KEY=VALUE ...]`
  → metrikatábla (modell vs. baseline, osztályonként) + best/worst képek a konzolra.
- `run_evaluation(cfg, weights=None, *, split="test", limit=None, device=None, figures=False)`
  → `{"report", "json_path", "figure_paths"}` — a notebook ezt hívja.
- `evaluate(model, loader, device, num_classes)` → `{"model": ..., "baseline": ...}`, mindkettő
  `SegmentationMeter.compute()` dict; a modellé tartalmazza a `per_image_iou`-t (a loader
  sorrendjében), a baseline-é nem.
- `select_examples(per_image_iou, k=3)` → `{"best": [...], "worst": [...]}` indexek (NaN kihagyva).
- `examples_figure(model, dataset, report, cfg, device)` → best, majd worst sorok ábrája
  (`show_predictions`, sorcímben név + IoU).
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

## evaluate.py — viselkedés, döntések

- **Egy menet:** a modell és a „mindig háttér” baseline metrikáit ugyanabban a ciklusban
  számolja (a teszt split 3669 képét csak egyszer kell betölteni).
- **A képeket nem tartja memóriában** (teljes test split 128×128-on ~700 MB lenne): csak a
  képenkénti IoU-t; a best/worst képeket utólag index alapján tölti be a datasetből.
- **Split-ek:** `test` (alap, hivatalos teszt), `val` (a seedelt trainval-felosztás val része,
  ugyanaz, mint tanításkor), `sample` (`sample_data/`, letöltés nélkül — offline demó).
  Ha a dataset nem tölthető be, a hiba üzenete a `--split sample`-t javasolja.
- **Kimenet:** `paths.results/<split>_metrics.json` (`results/test_metrics.json` a gitbe
  kerülő, README-ben és notebookban idézett eredmény). Tartalma: `split`, `n_images`,
  `weights`, `class_names`, `model` (metrikák + konfúziós mátrix), `baseline`, `examples`
  (`best`/`worst`: index, név, IoU), `per_image_iou` (`names`, `iou`). Szigorú JSON: NaN → `null`.
- **`--limit N`** = az első N kép (nem batch). Limitált futás **nem ír JSON-t** — ugyanaz a
  védelem, mint a `train.py` exportjánál: próbafuttatás nem írhatja felül az eredményt.
- **`--figures`:** `results/<split>_confusion_matrix.png`, `results/<split>_best_worst.png`
  (a `.gitignore` szerint PNG nem kerül gitbe).
- Súlybetöltés `weights_only=True`-val; ha a súlyfájl nem illik a `model.*` konfighoz,
  érthető `RuntimeError`, hiányzó fájlnál `FileNotFoundError` tanítási tippel.
- Mérés a betanított modellel (val mIoU 0,7264, 23. epoch): `--split test --limit 160` →
  mIoU 0,703 (baseline 0,187); a `sample`-n mIoU 0,668 (baseline 0,220). A teljes test
  split CPU-n kb. 4 perc.

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
pytest tests/test_metrics.py tests/test_evaluate.py -q
python -m unet_pet.evaluate --split sample --limit 2   # gyors, letöltés nélkül
```
Kézzel számolt kis tenzorokon: tökéletes predikció → IoU = 1; teljesen rossz →
0; ismert konfúziós mátrixra ismert mIoU.

<!--
KARBANTARTÁS: ha ezt a modult módosítod a kódban, frissítsd ezt a docot
UGYANABBAN a commitban (lásd AGENTS.md → Git és commit).
-->
