# Adat: Oxford-IIIT Pet szegmentálás

## Cél
Tiszta, ellenőrzött `(kép, maszk)` párok a tanításhoz és a kiértékeléshez.

## Viselkedés
- A torchvision `OxfordIIITPet` (`target_types="segmentation"`) letölti az adatot a `data/` mappába.
- Kép: 128×128, normalizált `float32 (3,H,W)`; maszk: `int64 (H,W)`, értékek `{0,1,2}`.
- Train: szinkron véletlen vízszintes tükrözés + enyhe ColorJitter (csak kép).
- Val: a `trainval` 15%-a, seedelt; Test: hivatalos `test` split. Val/test augmentáció nélkül.
- `sample_data/`: 6-8 kép + maszk (össz. < 2 MB) offline demóhoz, betöltő függvénnyel.

## Elfogadási kritériumok
- [ ] `pytest tests/test_data.py` zöld, letöltés nélkül.
- [ ] Maszk egyedi értékei ⊆ {0,1,2}; tükrözés után kép és maszk egyezik.
- [ ] Split determinisztikus és diszjunkt.
- [ ] A notebookban megjelenik pár példakép a maszkjával és az osztályok pixelaránya.

## Nem cél
Más dataset, saját annotálás, fajta-osztályozás.
