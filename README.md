# neuralnet — Kisállat-szegmentálás U-Nettel (Oxford-IIIT Pet)

Féléves projekt a **Neurális hálók** tárgyhoz.
Egy nulláról épített **U-Net** háló pixelenként szétválasztja a fotókon a
kisállatot, a hátteret és a körvonalat (3 osztályos szemantikus szegmentálás).

> A projekt fejlesztés alatt áll. A lenti parancsok és fájlok a **tervezett**
> állapotot írják le; amelyik még nem létezik, azt a fejlesztés során hozzuk létre.

## Gyors áttekintés

| | |
|---|---|
| Feladat | szemantikus szegmentálás (kisállat / háttér / határ) |
| Adathalmaz | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) — torchvision automatikusan letölti |
| Modell | U-Net (saját implementáció, PyTorch) |
| Loss | CrossEntropy + Dice |
| Fő metrika | mIoU (mellette Dice, pixel accuracy) |
| Kísérletkövetés | TensorBoard |
| Reprodukálhatóság | fix seed (42), `configs/default.yaml` |

## Telepítés

Python 3.11 ajánlott.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

## Futtatás

**A notebook (beadandó):**

```bash
jupyter notebook unet_pet_segmentation.ipynb
```

Alapból `TRAIN = False`: a notebook betölti a mentett súlyokat
(`models/unet_pet_best.pt`), kiértékel és ábrákat készít. Újratanításhoz állítsd
`TRAIN = True`-ra (GPU ajánlott).

**Parancssorból:**

```bash
python -m unet_pet.train --config configs/default.yaml     # tanítás
python -m unet_pet.evaluate --weights models/unet_pet_best.pt   # kiértékelés
tensorboard --logdir runs                                   # kísérletek
```

Az adathalmaz első futáskor a `data/` mappába töltődik le (~800 MB). Internet
nélkül a notebook a `sample_data/` mintaképeken mutatja be a modellt.

## Eredmények

<!-- TODO: a tanítás után kitöltendő -->

| Metrika (teszt) | U-Net | Baseline („mindig háttér”) |
|---|---|---|
| mIoU | – | – |
| Dice | – | – |
| Pixel accuracy | – | – |

## Mappaszerkezet

```
.
├── unet_pet_segmentation.ipynb   # a beadandó notebook
├── unet_pet/                     # Python csomag (adat, modell, tanítás, kiértékelés)
├── configs/default.yaml          # hiperparaméterek
├── models/unet_pet_best.pt       # betanított súlyok
├── sample_data/                  # pár mintakép + maszk
├── tests/                        # pytest tesztek
├── requirements.txt              # függőségek pontos verzióval
├── data/  runs/  checkpoints/    # helyben keletkezik, nincs gitben
│
├── AGENTS.md, CLAUDE.md          # szabályok az AI-kódoló asszisztensnek
├── ai-context/                   # AI-nak szánt projektleírás: OVERVIEW, modules/, NOTES
├── ai-agents/, .claude/agents/   # AI-ügynökök (reviewer, test-writer, doc-keeper)
└── ralph/                        # RALPH autonóm ciklus: specs/, terv, promptok
```

## Fejlesztés (AI-asszisztenssel)

A repo Claude Code-dal (VS Code) való fejlesztésre van előkészítve. Az AI a
[`CLAUDE.md`](CLAUDE.md) → [`AGENTS.md`](AGENTS.md) →
[`ai-context/OVERVIEW.md`](ai-context/OVERVIEW.md) sorrendben olvassa a
szabályokat, a feladatokat a [`ralph/specs/`](ralph/specs/) és a
[`ralph/IMPLEMENTATION_PLAN.md`](ralph/IMPLEMENTATION_PLAN.md) írja le.

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

## Hivatkozások

- O. Ronneberger, P. Fischer, T. Brox: *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar: *Cats and Dogs*, CVPR 2012 (Oxford-IIIT Pet).
