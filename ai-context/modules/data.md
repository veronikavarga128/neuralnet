<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: TERVEZETT — a kód még nem létezik. Amikor elkészül, a doc-keeper
  ügynök frissíti a valós fájlnevekre és függvényekre.
-->

# data — adathalmaz, transzformációk, split

## Cél

Az Oxford-IIIT Pet letöltése és `(kép, maszk)` párok előállítása a tanításhoz:
átméretezés, normalizálás, címke-leképezés, szinkronizált augmentáció,
seedelt train/val felosztás és DataLoaderek.

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `unet_pet/data.py` | `PetSegmentationDataset`, `JointTransform`, `make_dataloaders(cfg)` |
| `sample_data/images/*.jpg`, `sample_data/masks/*.png` | kis mintakészlet (pár kép) |

## Belépési pontok

- `make_dataloaders(cfg) -> dict[str, DataLoader]` — kulcsok: `train`, `val`, `test`
- `load_sample_data(path, img_size)` — a `sample_data/` betöltése offline demóhoz
- `denormalize(img_tensor)` — vizualizációhoz

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
