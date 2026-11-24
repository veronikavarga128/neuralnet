<!--
================================================================
  AI-NAK SZÁNT FÁJL — projekt áttekintés
  Az `ai-context/` mappa kifejezetten AI-kódoló eszközöknek készült.
  Ez az OVERVIEW a projekt belépési dokumentuma az AI számára.
================================================================
-->

# neuralnet — Áttekintés (AI-kontextus)

## Egy mondatban

Egy nulláról épített **U-Net** háló, ami az **Oxford-IIIT Pet** fotóin
pixelenként szétválasztja a kisállatot, a hátteret és a körvonalat
(3 osztályos szemantikus szegmentálás), TensorBoarddal követett, reprodukálható
tanítással és kiértékeléssel.

## Felelősség / hatókör

**Benne van:**
- az adathalmaz letöltése (torchvision), előfeldolgozása, augmentálása, felosztása;
- a U-Net modell definíciója;
- tanítás (CE + Dice loss), checkpointolás, TensorBoard naplózás;
- kiértékelés a teszt spliten (mIoU, Dice, pixel accuracy, osztályonkénti IoU,
  konfúziós mátrix), vizuális példák, hibaelemzés;
- a beadandó notebook, ami mindezt magyarázza és futtatja.

**Nincs benne (nem cél):**
- osztályozás (kutya/macska, fajta) — az órán vett feladattípus, nem kell;
- előtanított backbone, külső szegmentáló könyvtár (pl. `segmentation_models_pytorch`);
- webes demo, deploy, API;
- hiperparaméter-keresés nagy léptékben (legfeljebb 2-3 kézi kísérlet összevetése).

## Stack

- Python 3.11, PyTorch + torchvision (pontos verziók: `requirements.txt`)
- TensorBoard (`torch.utils.tensorboard.SummaryWriter`)
- NumPy, Matplotlib, PyYAML, tqdm, Jupyter
- Fejlesztés: pytest, ruff (`requirements-dev.txt`)
- Futtatás: helyi gép (CPU vagy GPU) vagy Google Colab; a tanítás GPU-n ajánlott

## Adathalmaz

- **Oxford-IIIT Pet** — 37 kutya- és macskafajta, ~7 400 kép, pixelszintű trimap maszkkal.
  Hivatalos oldal: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Betöltés: `torchvision.datasets.OxfordIIITPet(root="data", split=..., target_types="segmentation", download=True)`
  (~800 MB, első futáskor töltődik le a `data/` mappába).
- Splitek: hivatalos `trainval` (3 680 kép) → seedelt **85% train / 15% val**;
  hivatalos `test` (3 669 kép) → csak végső kiértékelés.
- Címkék: trimap `1=kisállat, 2=háttér, 3=határ` → `0=kisállat, 1=háttér, 2=határ`.
- Bemenet: 128×128 RGB, ImageNet-átlaggal/szórással normalizálva (konfigból állítható).

## Magas szintű architektúra

```
configs/default.yaml
        │  load_config()
        ▼
 data.py ──► OxfordIIITPet → közös (kép, maszk) transzformáció → DataLoader (train/val/test)
        │
        ▼
 model.py ──► UNet(in_ch=3, num_classes=3, base_channels=32, depth=4)
        │        encoder: [Conv-BN-ReLU ×2 + MaxPool] ×4
        │        bottleneck
        │        decoder: [ConvTranspose (vagy upsample) + skip concat + Conv-BN-ReLU ×2] ×4
        │        1×1 conv → logits (B, 3, H, W)
        ▼
 train.py ──► loss = CrossEntropy + Dice, Adam, LR scheduler
        │     epochonként: train → val → metrikák → TensorBoard → legjobb val mIoU checkpoint
        ▼
 models/unet_pet_best.pt  +  runs/<időbélyeg>_<név>/ (TensorBoard)
        │
        ▼
 evaluate.py ──► teszt split: mIoU, Dice, pixel acc., osztályonkénti IoU, konfúziós mátrix
        │
        ▼
 visualize.py ──► kép | valódi maszk | predikció | overlay ábrák, legjobb/legrosszabb példák
```

A notebook (`unet_pet_segmentation.ipynb`) ezeket a modulokat hívja lépésről
lépésre, és minden lépés előtt magyarázza, mi történik és miért.

## Tervezett mappaszerkezet

```
.
├── unet_pet_segmentation.ipynb   # BEADANDÓ: fő notebook
├── unet_pet/                     # Python csomag
│   ├── __init__.py
│   ├── config.py                 # YAML → dataclass
│   ├── utils.py                  # seed, eszköz, útvonalak
│   ├── data.py                   # dataset, transzformációk, split, DataLoader
│   ├── model.py                  # UNet
│   ├── losses.py                 # DiceLoss, kombinált loss
│   ├── metrics.py                # IoU, Dice, pixel accuracy, konfúziós mátrix
│   ├── train.py                  # tanító ciklus + CLI
│   ├── evaluate.py               # teszt kiértékelés + CLI
│   └── visualize.py              # ábrák, TensorBoard képrács
├── configs/default.yaml
├── models/unet_pet_best.pt       # betanított súlyok (gitbe kerül)
├── sample_data/                  # pár kép + maszk (images/, masks/)
├── tests/                        # pytest
├── requirements.txt              # pontos verziók
├── requirements-dev.txt          # pytest, ruff
├── pyproject.toml                # ruff + pytest beállítás
├── data/        (gitignore)      # letöltött dataset
├── runs/        (gitignore)      # TensorBoard logok
└── checkpoints/ (gitignore)      # köztes checkpointok
```

## Modulok

Minden modulnak külön doksija van a [`modules/`](modules/) mappában, a
[`modules/_MODULE_TEMPLATE.md`](modules/_MODULE_TEMPLATE.md) sablon szerint.

| Modul | Rövid leírás | Doc |
|-------|--------------|-----|
| config-utils | konfiguráció, seed, eszközválasztás | [`modules/config-utils.md`](modules/config-utils.md) |
| data | dataset, transzformációk, split, DataLoader | [`modules/data.md`](modules/data.md) |
| model | U-Net architektúra | [`modules/model.md`](modules/model.md) |
| training | loss, tanító ciklus, checkpoint, TensorBoard | [`modules/training.md`](modules/training.md) |
| evaluation | metrikák, kiértékelés, vizualizáció | [`modules/evaluation.md`](modules/evaluation.md) |
| notebook | a beadandó notebook felépítése | [`modules/notebook.md`](modules/notebook.md) |

## Külső kapcsolatok

- Egyetlen külső függés: az Oxford-IIIT Pet letöltése a torchvisionön keresztül
  (VGG Oxford szerver). Ha nem érhető el, a notebook a `sample_data/`-ra esik vissza.
- Nincs adatbázis, API, titok. `.env` nem szükséges.

## Ismert buktatók

- Maszk átméretezése csak `NEAREST`-tel; címkék `mask - 1`, `torch.long`.
- A kép és a maszk geometriai augmentációja szinkronban.
- A teszt split érintetlen a tanítás alatt (a modellválasztás a val mIoU alapján).
- A határ (`2`) osztály vékony és kevés pixel → alacsony IoU várható rá; ez nem
  hiba, a kiértékelésben külön kell róla írni.
- Pixel accuracy félrevezető lehet (sok háttér) → a fő metrika a **mIoU**.
- `torch.load(..., weights_only=True, map_location=device)`.
- CPU-n a tanítás lassú: a notebook alapból `TRAIN = False`.

## Hogyan futtasd / teszteld

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q && ruff check .
python -m unet_pet.train --config configs/default.yaml
tensorboard --logdir runs
jupyter notebook unet_pet_segmentation.ipynb
```
