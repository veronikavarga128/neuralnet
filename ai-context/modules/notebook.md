<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: RÉSZBEN KÉSZ — a notebook mind a 0–10. fejezettel elkészült, limitált és offline
  módban hibátlanul fut. Hátra: a CE-only futás eredményei (8. fejezet) és a végleges,
  kimenetekkel mentett futtatás.
-->

# notebook — a beadandó notebook

## Cél

A `unet_pet_segmentation.ipynb` a beadandó fő eleme: tartalmazza a projekt
fő logikáját (a `unet_pet` csomag függvényeit lépésről lépésre hívja), és
**magyarul, részletesen elmagyarázza** a munkafolyamatot.

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `unet_pet_segmentation.ipynb` | a notebook (repo gyökér) |

## Kötelező szerkezet (fejezetek)

0. **Cím, szerző, összefoglaló** — feladat, adathalmaz, modell, fő eredmény egy bekezdésben.
1. **Beállítások** — importok, `TRAIN = False` kapcsoló, seed, eszköz, konfig betöltése és kiírása.
2. **Az adathalmaz** — forrás + link, splitek mérete, címkék jelentése, példaképek maszkkal,
   osztályeloszlás (pixelarányok) ábra.
3. **Előfeldolgozás és augmentáció** — mit és miért (méret, normalizálás, NEAREST, szinkron flip),
   augmentált példák.
4. **A modell** — U-Net elv (encoder–decoder, skip kapcsolatok), ábra/szöveges séma,
   `model` kiírása, paraméterszám, miért ez a választás (és miért nem órai modell).
5. **Tanítás** — loss (CE + Dice) indoklása, optimizer, scheduler, hiperparaméterek táblázata;
   `TRAIN = True` esetén tanítás, egyébként a mentett súlyok betöltése; tanulási görbék
   (a history-ból vagy a mentett CSV/JSON-ból), TensorBoard megnyitásának leírása
   (`%load_ext tensorboard` / `%tensorboard --logdir runs`).
6. **Kiértékelés** — teszt metrikák táblázata, baseline összevetés, konfúziós mátrix,
   osztályonkénti IoU.
7. **Vizuális eredmények** — véletlen, legjobb és legrosszabb példák; predikció a `sample_data/` képeire.
8. **Kísérletek összevetése** — 2-3 futás (pl. csak CE vs. CE+Dice, vagy base_channels 16 vs 32) táblázatban.
9. **Összegzés és korlátok** — mit tanultunk, hibák, továbbfejlesztési lehetőségek.
10. **Reprodukálás** — pontos parancsok, verziók, seed, futásidő.

## Megvalósítás

- **46 cella**, a fejezetek a fenti szerkezetet követik; a kód a `unet_pet` függvényeit hívja
  (`load_config`, `make_datasets`, `build_model`, `fit`, `load_trained_model`,
  `run_evaluation`, `examples_figure`, `show_predictions`, `plot_history`, ...).
- **Kapcsolók (1. fejezet):** `TRAIN = False`, `RUN_TEST_EVAL = True` (a felhasználó döntése:
  alapból élő teszt-kiértékelés, a torchvision letölti az adatot), `EVAL_LIMIT = None`.
- **`DATA_AVAILABLE`:** a 2. fejezet `load_base_datasets`-t try/except-be teszi. Ha nem sikerül,
  egyértelmű üzenet; a példák és az osztályeloszlás a `sample_data/`-ból, a 6. fejezet a mentett
  `results/test_metrics.json`-t mutatja, a 7. fejezet tesztkép-ábrái kimaradnak (üzenettel),
  a `sample_data/` demó mindig fut.
- **`LIVE_EVAL`** = `RUN_TEST_EVAL and DATA_AVAILABLE`; élő futásnál a `run_evaluation`
  (`EVAL_LIMIT = None` esetén) újraírja a `results/test_metrics.json`-t — determinisztikus,
  ezért a tartalma nem változik, és a notebook táblája garantáltan egyezik a JSON-nal.
- **8. fejezet:** a `models/unet_pet_history.json` + `results/test_metrics.json` (fő modell) és a
  `results/history_ce_only.json` + `results/test_metrics_ce_only.json` (CE-only) fájlokat
  olvassa; a hiányzó futást üzenettel jelzi, a meglévőket táblázatban és görbén mutatja.
- **Abszolút útvonal** a kimenetekben sincs: a `rel()` segéd a repó gyökeréhez képest írja ki.
- **Táblázatok** `IPython.display.Markdown`-nal (a pandas nem függőség).
- Ábráknál a cella `plt.show()`-val zárul (különben a visszaadott `Figure` kétszer jelenne meg);
  `progress=False` a kiértékelésnél (a tqdm widget nem jelenik meg jól a mentett notebookban).
- A notebook kódcelláit is ellenőrzi a `ruff check .` / `ruff format`.

## Futásidők (CPU, Ryzen 7 8845HS)

- `TRAIN = False`, teljes teszt-kiértékelés: kb. 5 perc (ebből a kiértékelés kb. 4 perc).
- `EVAL_LIMIT = 200`: kb. 20 s. Offline (adathalmaz nélkül): kb. 12 s.

## Belépési pontok

Fentről lefelé futtatva (`Kernel → Restart & Run All`) hibamentesen végig kell érnie.

## Függőségek

Támaszkodik: az összes `unet_pet` modul, `models/unet_pet_best.pt`, `sample_data/`.

## Buktatók

- `TRAIN = False` alapérték; a súlyfájl hiányában érthető hibaüzenet + teendő.
- Letöltési hiba esetén a teszt-kiértékelés cellái lépjenek át (jelezve), a
  `sample_data/` demó fusson.
- Ne legyen abszolút útvonal, Colab-specifikus kód csak `if IN_COLAB:` ágban.
- A futásidő `TRAIN = False` esetén CPU-n is maradjon pár perc alatt
  (a teszt-kiértékeléshez opcionális `EVAL_LIMIT` kapcsoló).
- A markdown magyarázat a pontozás része: ne legyen „TODO”, a számok egyezzenek a kimenetekkel.

## Hogyan teszteld

```
jupyter nbconvert --to notebook --execute unet_pet_segmentation.ipynb --output nb_check.ipynb --output-dir <temp mappa>
```
Gyors ellenőrzés: egy másolatban `EVAL_LIMIT = 200`; offline szimuláció: a másolatban a
`load_base_datasets(...)` sort `raise OSError(...)`-ra cserélve. Végül friss venv-ben, csak
`pip install -r requirements.txt` után is.

## Ha új szöveg/eredmény kerül bele

A markdown-cellákban néhány szám szerepel (teszt mIoU 0,737, val mIoU 0,726, 23. epoch, 7 hibás
maszk stb.). Újratanítás után ezeket a kimenetekhez kell igazítani. A 8. fejezet értelmezése a
CE-only futás után egészítendő ki a tényleges számokkal.

<!--
KARBANTARTÁS: ha ezt a modult módosítod a kódban, frissítsd ezt a docot
UGYANABBAN a commitban (lásd AGENTS.md → Git és commit).
-->
