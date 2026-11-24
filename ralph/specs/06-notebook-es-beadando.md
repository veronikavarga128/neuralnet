# Notebook és beadandó csomag

## Cél
A tárgy beadási követelményei: a notebook tartalmazza a fő logikát és részletes
magyarázatot; az oktató extra beállítás nélkül tudja futtatni.

## Viselkedés
- `unet_pet_segmentation.ipynb` a gyökérben, a szerkezet: `ai-context/modules/notebook.md`.
- Magyar nyelvű markdown magyarázat minden lépés előtt (adat, előkészítés, modell, tanítás, kiértékelés).
- `TRAIN = False` alapértelmezés: súlybetöltés + kiértékelés + ábrák; `TRAIN = True`: teljes tanítás.
- `requirements.txt`: minden futáshoz szükséges csomag pontos (`==`) verzióval; `requirements-dev.txt`: pytest, ruff.
- `models/unet_pet_best.pt` gitben (< 50 MB). `sample_data/` gitben (< 2 MB).
- README: leírás, dataset link, telepítés (Windows/macOS/Linux), futtatás, eredmények táblázata, mappaszerkezet.
- Opcionális: `make_submission` script / leírás, ami zip-et készít a beadandóból (data/, runs/, .venv/ nélkül).

## Elfogadási kritériumok
- [ ] Friss venv + `pip install -r requirements.txt` után `jupyter nbconvert --execute` hibamentesen lefut.
- [ ] Internet nélkül is lefut (a teszt-kiértékelést kihagyja, jelzi; a sample_data demó működik).
- [ ] A notebookban nincs „TODO”, üres markdown vagy abszolút útvonal.
- [ ] A beadandó zip < 100 MB, és tartalmaz mindent a fenti listából.

## Nem cél
Webes demó, Docker image.
