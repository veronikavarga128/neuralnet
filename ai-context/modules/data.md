<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: RÉSZBEN KÉSZ — data.py implementálva, tesztekkel. A sample_data/ és a
  load_sample_data a következő tervtételben készül.
-->

# data — adathalmaz, transzformációk, split

## Cél

Az Oxford-IIIT Pet letöltése és `(kép, maszk)` párok előállítása a tanításhoz:
átméretezés, normalizálás, címke-leképezés, szinkronizált augmentáció,
seedelt train/val felosztás és DataLoaderek.

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `unet_pet/data.py` | `JointTransform`, `build_transform`, `trimap_to_labels`, `PetSegmentationDataset`, `split_indices`, `load_base_datasets`, `make_datasets`, `make_dataloaders`, `denormalize`, `class_pixel_fractions`, `CLASS_NAMES` |
| `tests/test_data.py` | tesztek hamis alapdatasettel (`FakePetBase`), letöltés nélkül |
| `sample_data/images/*.jpg`, `sample_data/masks/*.png` | kis mintakészlet (TERVEZETT) |

## Belépési pontok

- `make_dataloaders(cfg, trainval_base=None, test_base=None, download=True) -> dict[str, DataLoader]`
  — kulcsok: `train`, `val`, `test`. Az alapdataset beinjektálható (tesztek); ha nincs
  megadva, `OxfordIIITPet`-et tölt be a `cfg.data.root`-ból.
- `make_datasets(cfg, trainval_base, test_base)` — ugyanez DataLoader nélkül
- `denormalize(img, mean, std)` — vizualizációhoz, `[0,1]`-re vágva
- `class_pixel_fractions(dataset, max_items=None)` — osztályonkénti pixelarány (notebook)
- `load_sample_data(path, img_size)` — TERVEZETT, a `sample_data/` offline betöltése

## Felépítés, döntések

- Az alapdataset PIL `(kép, trimap maszk)` párokat ad; a `PetSegmentationDataset` egy
  indexlistával és saját transzformációval csomagolja be. Így a train és a val
  ugyanabból a `trainval` példányból jön, de a val augmentáció nélkül.
- `JointTransform`: `transforms.v2` pipeline, a maszk `tv_tensors.Mask`-ban →
  NEAREST átméretezés és szinkron tükrözés automatikusan; ColorJitter / ToDtype /
  Normalize csak a képre hat.
- `trimap_to_labels`: előbb `long`, aztán `- 1` (uint8-on a 0 → 255-re fordulna), és
  hibát dob, ha a címkék nincsenek `{0,1,2}`-ben.
- `split_indices`: `torch.randperm` seedelt generátorral, diszjunkt, rendezett indexek;
  3680 képből 552 val.
- Train DataLoader: `shuffle`, seedelt `generator`, `worker_init_fn=seed_worker`,
  `drop_last` (az 1-es méretű utolsó batch elrontaná a BatchNormot).
- Augmentáció erőssége konfigból: `data.hflip_prob`, `data.color_jitter`.

## Adat-specifikáció

- Kimenet: kép `float32 (3, H, W)` normalizálva, maszk `int64 (H, W)` értékei `{0,1,2}`.
- Címkék: `0=kisállat, 1=háttér, 2=határ` (= eredeti trimap − 1).
- Train augmentáció: véletlen vízszintes tükrözés (kép+maszk), enyhe
  ColorJitter (csak kép). Val/test: csak átméretezés + normalizálás.
- Split: `trainval` → `random_split` / indexkeverés fix seeddel (85/15).

## Függőségek

Támaszkodik: `torchvision.datasets.OxfordIIITPet`, `torchvision.transforms.v2`, config-utils.
Rá támaszkodik: training, evaluation, notebook.

## Buktatók

- Maszkot **NEAREST** interpolációval méretezz.
- `transforms.v2`-nél a maszkot `tv_tensors.Mask`-ba csomagolva lehet a
  képpel közösen transzformálni — ez a legegyszerűbb szinkronizálás.
- A val részhalmaz NE kapjon augmentációt, holott ugyanabból a `trainval`
  datasetből jön (két külön dataset-példány kell, vagy wrapper).
- A tesztek ne töltsenek le semmit: hamis datasettel / `sample_data/`-val dolgozz.

## Hogyan teszteld

```
pytest tests/test_data.py -q
```
Ellenőrizd: alak, dtype, a maszk egyedi értékei ⊆ {0,1,2}, a split mérete és
determinisztikussága (két futás = ugyanazok az indexek), tükrözésnél a kép és
a maszk együtt fordul.

<!--
KARBANTARTÁS: ha ezt a modult módosítod a kódban, frissítsd ezt a docot
UGYANABBAN a commitban (lásd AGENTS.md → Git és commit).
-->
