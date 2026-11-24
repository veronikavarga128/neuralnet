<!--
================================================================
  AI-NAK SZÁNT FÁJL — projekt-szabályok (a szabályok FORRÁSA)
  Ez egy önálló, egy-repós projekt: nincs szülő-mappa
  (../AGENTS.md, ../PROJECT.md, ../rules/). Minden szabály ITT van.
================================================================
-->

# AGENTS.md — neuralnet (U-Net képszegmentálás, Oxford-IIIT Pet)

Féléves projekt a „Neurális hálók” tárgyhoz (PyTorch). A cél egy
**U-Net** szemantikus szegmentáló háló tanítása és kiértékelése az
**Oxford-IIIT Pet** adathalmazon, teljes, reprodukálható pipeline-nal.

## Kötelező olvasás, mielőtt kódot írsz

1. Ez a fájl
2. [`ai-context/OVERVIEW.md`](ai-context/OVERVIEW.md) — architektúra, modulok, buktatók
3. a témához tartozó modul-doc az [`ai-context/modules/`](ai-context/modules/) alól
4. a feladathoz tartozó spec a [`ralph/specs/`](ralph/specs/) alól

## A tárgy kötelező követelményei (NE sértsd meg)

- **Modell × feladat szabály:** órán NEM szerepelt szegmentálás és U-Net, ezért
  választottuk. Ne cseréld a modellt órán vett modell+feladat párra (pl. ResNet
  osztályozás, autoencoder MNIST-en). Előtanított encodert (pl. ResNet backbone)
  csak akkor használj, ha a felhasználó kifejezetten kéri — alapból **saját,
  nulláról épített U-Net**.
- **Teljes megoldás:** adat-előkészítés → modell → tanítás → kiértékelés.
- **Reprodukálhatóság:** fix seed (Python, NumPy, PyTorch, DataLoader worker,
  train/val split), egyetlen konfigurációs fájl, README futtatási leírással.
- **Kísérletkövetés TensorBoarddal:** loss, metrikák, learning rate és
  predikció-képek naplózása.

## A beadandó (ezt a végén ellenőrizni kell)

| Elem | Hol | Megjegyzés |
|------|-----|------------|
| Notebook | `unet_pet_segmentation.ipynb` (repo gyökér) | fő logika + **részletes magyar nyelvű magyarázat** minden lépéshez; fentről lefelé hibamentesen fusson |
| `requirements.txt` | gyökér | **minden** csomag **pontos verzióval** (`==`) |
| Betanított súlyok | `models/unet_pet_best.pt` | `state_dict`, < 50 MB, gitbe kerül |
| Mintadataset | `sample_data/` | néhány kép + maszk (össz. < 2 MB), a notebook demójához |
| Külső dataset link | README + notebook | az Oxford-IIIT Pet hivatalos oldala; a torchvision automatikusan letölti |
| README | `README.md` | telepítés, futtatás, eredmények, mappaszerkezet |

**Az oktató a saját gépén, extra beállítás nélkül futtatja.** Ezért:
- a notebook alapból **`TRAIN = False`** módban fut: betölti a mentett súlyokat,
  és csak kiértékel / vizualizál; `TRAIN = True` esetén tanít újra;
- semmi ne függjön `.env`-től, abszolút úttól, GPU-tól vagy `pip install -e`-től;
- CPU-n is fusson (lassabban), az eszközt automatikusan válassza: cuda → mps → cpu;
- ha nincs internet / nem sikerül a letöltés, a notebook a `sample_data/`-n
  mutassa be a predikciót, és ezt egyértelmű üzenettel jelezze.

## Stack és konvenciók

- **Nyelv:** Python 3.11
- **Fő csomagok:** `torch`, `torchvision`, `tensorboard`, `numpy`, `matplotlib`,
  `pyyaml`, `tqdm`, `jupyter`; fejlesztéshez `pytest`, `ruff`.
- **Kód:** azonosítók és docstringek angolul, típusannotációkkal. A notebook
  markdown-cellái és a README **magyarul**.
- **Csomag:** a kód a gyökérben lévő `unet_pet/` csomagban van, így a notebook
  és a tesztek útvonal-trükk nélkül importálják (`from unet_pet.model import UNet`).
- **Konfiguráció:** `configs/default.yaml` → `unet_pet/config.py` dataclass.
  Hiperparamétert ne égess be a kódba.
- **Útvonalak:** `pathlib.Path`, mindig a repo gyökeréhez képest.
- **Új függőség:** csak indokolt esetben; ha felveszed, rögtön kerüljön a
  `requirements.txt`-be pontos verzióval.

## Belépési pontok

- `unet_pet_segmentation.ipynb` — a beadandó, ezt futtatja az oktató
- `python -m unet_pet.train --config configs/default.yaml` — tanítás parancssorból
- `python -m unet_pet.evaluate --weights models/unet_pet_best.pt` — kiértékelés
- `tensorboard --logdir runs` — kísérletek megtekintése

## Parancsok (build / teszt / lint)

A RALPH ciklus és az ügynökök ezekkel validálnak.

| Mire | Parancs |
|------|---------|
| Telepítés | `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt` |
| Teszt | `pytest -q` |
| Lint | `ruff check .` |
| Formázás | `ruff format .` |
| Notebook futtatás ellenőrzése | `jupyter nbconvert --to notebook --execute unet_pet_segmentation.ipynb --output /tmp/nb_check.ipynb` |

A tesztek **CPU-n, internet és letöltött dataset nélkül** fussanak, pár
másodperc alatt (kis, véletlen tenzorokkal / a `sample_data/` mappával).

## Buktatók (amit egy AI könnyen elront itt)

- **Maszk átméretezése:** a maszkot mindig `NEAREST` interpolációval méretezd,
  különben köztes, nem létező osztálycímkék keletkeznek.
- **Trimap címkék:** az Oxford-IIIT Pet maszk értékei `1=kisállat, 2=háttér,
  3=határ`. Ezt `0..2`-re kell eltolni (`mask - 1`), `torch.long` típussal.
- **Augmentáció:** a képre és a maszkra **ugyanaz** a véletlen geometriai
  transzformáció menjen (pl. flip); színtranszformáció csak a képre.
- **Adatszivárgás:** a hivatalos `test` splitet csak a végső kiértékelésre
  használd. Validáció = a `trainval` split seedelt felosztása.
- **Súlyok betöltése:** `torch.load(..., map_location=device, weights_only=True)`.
- **Notebook Windows-on:** `num_workers=0` legyen az alapérték a notebookban.
- **Nagy fájlok:** `data/`, `runs/`, `checkpoints/` NEM kerül gitbe; egyedül a
  végső `models/unet_pet_best.pt` igen.
- **Determinizmus:** a seed beállítása egy helyen (`unet_pet/utils.py`) történjen,
  és a notebook legelején hívódjon meg.

## Git és commit

- Branch: `feat/<téma>`, `fix/<téma>`, `docs/<téma>`; RALPH: `ralph/YYYYMMDD-<téma>`.
  A RALPH soha nem fut `main`-en.
- Commit üzenet: Conventional Commits, magyarul is lehet
  (pl. `feat(model): U-Net encoder blokkok`).
- Ha egy modult módosítasz, a modul-docját **ugyanabban a commitban** frissítsd.
- Push és merge csak ember által.

## NE

- Ne commitolj datasetet, TensorBoard logot, `.env`-et.
- Ne kapcsolj ki és ne törölj tesztet azért, hogy zöld legyen.
- Ne írd át a notebook magyarázó szövegét üresre vagy „TODO”-ra — a magyarázat
  a beadandó része és értékelt.
- Ne válts modellt vagy feladatot a felhasználó kérése nélkül (lásd a tárgy szabályát).

## Ügynökök (ai-agents/)

A szerepek forrása az [`ai-agents/`](ai-agents/) mappa, a `.claude/agents/` csak adapter.

| Szerep | Mikor |
|--------|-------|
| `reviewer` | commit előtt |
| `test-writer` | új feature vagy bugfix után |
| `doc-keeper` | feladat végén: modul-doc és NOTES |

## AI-fájlok karbantartása

Modul-docot frissíthetsz és létrehozhatsz; az alapot (`AGENTS.md`,
`CLAUDE.md`, `ai-context/OVERVIEW.md`, `ai-agents/`, `ralph/PROMPT_*.md`) nem
írod át magadtól — javaslatot írsz az [`ai-context/NOTES.md`](ai-context/NOTES.md)-be,
és a projekt gazdája dönt róla.

## RALPH

Autonóm ciklus: [`ralph/README.md`](ralph/README.md).

## Modul-térkép

| Modul | Doc |
|-------|-----|
| Konfiguráció, seed, eszköz | [`ai-context/modules/config-utils.md`](ai-context/modules/config-utils.md) |
| Adat (dataset, transzformációk, split) | [`ai-context/modules/data.md`](ai-context/modules/data.md) |
| Modell (U-Net) | [`ai-context/modules/model.md`](ai-context/modules/model.md) |
| Tanítás (loss, loop, TensorBoard) | [`ai-context/modules/training.md`](ai-context/modules/training.md) |
| Kiértékelés (metrikák, vizualizáció) | [`ai-context/modules/evaluation.md`](ai-context/modules/evaluation.md) |
| Notebook (beadandó) | [`ai-context/modules/notebook.md`](ai-context/modules/notebook.md) |
