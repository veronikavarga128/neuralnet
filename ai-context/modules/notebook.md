<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: TERVEZETT — a kód még nem létezik. Amikor elkészül, a doc-keeper
  ügynök frissíti a valós fájlnevekre és függvényekre.
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
jupyter nbconvert --to notebook --execute unet_pet_segmentation.ipynb --output /tmp/nb_check.ipynb
```
Majd friss venv-ben, csak `pip install -r requirements.txt` után is.

<!--
KARBANTARTÁS: ha ezt a modult módosítod a kódban, frissítsd ezt a docot
UGYANABBAN a commitban (lásd AGENTS.md → Git és commit).
-->
