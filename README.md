# neuralnet — Kisállat-szegmentálás U-Nettel (Oxford-IIIT Pet)

Féléves projekt a **Neurális hálók** tárgyhoz.
Egy nulláról épített **U-Net** háló pixelenként szétválasztja a fotókon a
kisállatot, a hátteret és a körvonalat (3 osztályos szemantikus szegmentálás).

A beadandó a [`unet_pet_segmentation.ipynb`](unet_pet_segmentation.ipynb) notebook:
lépésről lépésre, magyar magyarázattal végigvezet az adat-előkészítésen, a modellen,
a tanításon és a kiértékelésen. A kimenetek el vannak mentve benne, így futtatás nélkül
is olvasható.

## Gyors áttekintés

| | |
|---|---|
| Feladat | szemantikus szegmentálás (kisállat / háttér / határ) |
| Adathalmaz | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) — a torchvision automatikusan letölti |
| Modell | U-Net, saját implementáció (PyTorch), kb. 7,8 M paraméter, előtanítás nélkül |
| Bemenet | 128 x 128 px RGB kép |
| Loss | CrossEntropy + Dice |
| Fő metrika | mIoU (mellette Dice és pixelpontosság) |
| Kísérletkövetés | TensorBoard |
| Reprodukálhatóság | fix seed (42), minden hiperparaméter a `configs/default.yaml`-ban |

## Eredmények

A hivatalos **teszt spliten** (3669 kép), a legjobb validációs mIoU-jú epoch
(25-ből a 23.) súlyaival:

| Metrika (teszt) | U-Net | Baseline („mindig háttér”) |
|---|---|---|
| **mIoU** | **0,737** | 0,193 |
| Dice (átlag) | 0,837 | 0,244 |
| Pixelpontosság | 0,900 | 0,578 |
| IoU — kisállat | 0,813 | 0,000 |
| IoU — háttér | 0,897 | 0,578 |
| IoU — határ | 0,501 | 0,000 |

A validációs mIoU 0,726 volt, a teszten mért érték ezzel összhangban van. A gyenge pont
a vékony határ osztály. A legrosszabb tesztképek egy részén a címke hibás: 7 tesztkép
valódi maszkjában egyetlen kisállat-pixel sincs. Részletek a notebook 6. és 7. fejezetében.

**Összehasonlító kísérlet: CE + Dice vs. csak CE.** Ugyanazzal a konfigurációval és
seeddel tanítottunk egy második modellt csak CrossEntropy loss-szal:

| Futás | Legjobb val mIoU | Teszt mIoU | Teszt IoU kisállat / háttér / határ |
|---|---|---|---|
| CE + Dice (beadott modell) | 0,726 | 0,737 | 0,813 / 0,897 / 0,501 |
| csak CE | 0,732 | 0,735 | 0,819 / 0,901 / 0,485 |

A két modell összesítve gyakorlatilag egyformán jó. A Dice-tag a ritka határ osztályon
hoz kis javulást. A két futás tanulásiráta-ütemezése eltérően alakult, ezért egy
futáspárból a különbség nem írható egyértelműen a loss számlájára. Részletek a
notebook 8. fejezetében.

## Telepítés

Python **3.11** kell. GPU nem szükséges, minden CPU-n is fut.

```bash
git clone https://github.com/veronikavarga128/neuralnet.git
cd neuralnet
python -m venv .venv
```

A virtuális környezet aktiválása:

| Rendszer | Parancs |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Ha a PowerShell nem engedi a szkript futtatását, előtte:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Utána:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

A `requirements.txt` minden csomagot pontos verzióval rögzít. Az `--extra-index-url` a
PyTorch CPU-s változatát telepíti, ami minden rendszeren elég a notebookhoz. Linuxon e
nélkül a pip a CUDA-s buildet töltené le, ami több GB-nyi NVIDIA-csomagot jelent.
Windowson és macOS-en a kapcsoló nem változtat semmin, de nem is árt. Ha van NVIDIA GPU-d
és azon akarsz tanítani, hagyd el a kapcsolót.

## Futtatás

### A notebook (beadandó)

```bash
jupyter notebook unet_pet_segmentation.ipynb
```

Majd `Kernel → Restart & Run All`. A notebook elején három kapcsoló van:

| Kapcsoló | Alapérték | Jelentés | Futásidő (CPU) |
|---|---|---|---|
| `TRAIN` | `False` | `False`: betölti a mentett súlyokat (`models/unet_pet_best.pt`); `True`: újratanít | `True` esetén kb. 2 óra |
| `RUN_TEST_EVAL` | `True` | élő kiértékelés a teljes teszt spliten; `False`: a mentett `results/test_metrics.json` jelenik meg | kb. 4 perc |
| `EVAL_LIMIT` | `None` | ha egy szám, csak az első ennyi tesztképen értékel (gyors próba) | pl. 200 képnél kb. 20 s |

Az alapbeállítással a teljes notebook CPU-n néhány perc alatt végigfut. Ehhez nem
számít bele az adathalmaz első letöltése (kb. 800 MB a `data/` mappába).

**Internet nélkül** a letöltés nem sikerül. A notebook ezt üzenetben jelzi, a teszt-
kiértékelésnél a mentett eredményeket mutatja, a predikciót pedig a repóban mellékelt
`sample_data/` 8 képén (4 macska, 4 kutya) mutatja be.

### Parancssorból

Minden parancs a repó gyökeréből futtatandó.

```bash
# tanítás (a legjobb súlyokat a models/ mappába exportálja)
python -m unet_pet.train --config configs/default.yaml --run-name ce_dice

# megszakadt tanítás folytatása az utolsó mentett epochtól
python -m unet_pet.train --resume checkpoints/<run>/last.pt

# gyors próbafuttatás (1 epoch, 2 batch, nem ír a models/ mappába)
python -m unet_pet.train --epochs 1 --limit-batches 2 --run-name smoke

# kiértékelés a teszt spliten -> results/test_metrics.json, --figures: PNG ábrák is
python -m unet_pet.evaluate --weights models/unet_pet_best.pt --figures

# gyors kiértékelés az első 2 tesztképen (nem ment JSON-t)
python -m unet_pet.evaluate --limit 2

# kísérletek megtekintése (http://localhost:6006)
tensorboard --logdir runs
```

A **csak CE-s összehasonlító kísérlet** reprodukálása:

```bash
python -m unet_pet.train --run-name ce_only --no-export --set train.dice_weight=0
python -m unet_pet.evaluate --weights checkpoints/<run>_ce_only/best.pt --tag ce_only
```

A `--no-export` miatt ez nem írja felül a beadott súlyokat, a `--tag` pedig külön fájlba
menti az eredményt (`results/test_metrics_ce_only.json`).

**Folytatásnál a `--set` és a `--no-export` kapcsolókat újra meg kell adni**, mert a
konfiguráció a parancssorból jön, nem a checkpointból.

## Adathalmaz

**Oxford-IIIT Pet Dataset**: <https://www.robots.ox.ac.uk/~vgg/data/pets/>

37 macska- és kutyafajta, kb. 7400 kép, pixelszintű *trimap* maszkokkal
(1 = kisállat, 2 = háttér, 3 = határ; a kódban 0..2-re eltolva). A hivatalos `trainval`
splitet seedelten 85% tanító és 15% validációs részre bontjuk. A hivatalos `test`
splitet csak a végső kiértékeléshez használjuk.

Az adathalmazt nem tartalmazza a repó: a `torchvision.datasets.OxfordIIITPet` első
futáskor letölti. Licenc: Creative Commons Attribution-ShareAlike 4.0 International
(CC BY-SA 4.0).

## Mappaszerkezet

```
.
├── unet_pet_segmentation.ipynb   # a beadandó notebook (kimenetekkel)
├── unet_pet/                     # Python csomag
│   ├── config.py, utils.py       #   konfiguráció, seed, eszközválasztás
│   ├── data.py                   #   dataset, transzformációk, split
│   ├── model.py                  #   U-Net
│   ├── losses.py, metrics.py     #   Dice/CE loss, IoU/Dice metrikák
│   ├── train.py, evaluate.py     #   tanítás és kiértékelés (CLI is)
│   └── visualize.py              #   ábrák
├── configs/default.yaml          # minden hiperparaméter
├── models/
│   ├── unet_pet_best.pt          # betanított súlyok (state_dict, ~30 MB)
│   └── unet_pet_history.json     # a tanítás epochonkénti görbéi
├── results/                      # kiértékelési eredmények (JSON; a PNG ábrák nincsenek gitben)
├── sample_data/                  # 8 tesztkép + maszk az offline demóhoz
├── tests/                        # pytest tesztek (CPU, internet nélkül, pár mp)
├── requirements.txt              # futtatási függőségek pontos verzióval
├── requirements-dev.txt          # pytest, ruff
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
