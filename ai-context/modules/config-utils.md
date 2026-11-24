<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: TERVEZETT — a kód még nem létezik. Amikor elkészül, a doc-keeper
  ügynök frissíti a valós fájlnevekre és függvényekre.
-->

# config-utils — konfiguráció, seed, eszköz

## Cél

Egy helyen kezeli a hiperparamétereket (YAML → típusos dataclass), a
reprodukálhatóságot (seed) és az eszközválasztást (cuda / mps / cpu).

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `configs/default.yaml` | az összes hiperparaméter és útvonal |
| `unet_pet/config.py` | `Config` dataclass, `load_config(path, overrides=None)` |
| `unet_pet/utils.py` | `set_seed(seed)`, `get_device()`, `project_root()`, `seed_worker` |

## Belépési pontok

- `load_config("configs/default.yaml")` → `Config`
- `set_seed(cfg.seed)` — a notebook és a `train.py` legelején
- `get_device()` → `torch.device`

## Konfig-kulcsok (tervezett alapértékek)

| Kulcs | Érték | Megjegyzés |
|-------|-------|------------|
| `seed` | 42 | |
| `data.root` | `data` | torchvision letöltés helye |
| `data.img_size` | 128 | |
| `data.val_fraction` | 0.15 | a `trainval` splitből |
| `data.num_workers` | 0 | Windows / notebook miatt |
| `data.mean` | `[0.485, 0.456, 0.406]` | ImageNet-normalizálás |
| `data.std` | `[0.229, 0.224, 0.225]` | ImageNet-normalizálás |
| `model.in_channels` | 3 | RGB |
| `model.base_channels` | 32 | |
| `model.depth` | 4 | |
| `model.num_classes` | 3 | |
| `train.epochs` | 25 | |
| `train.batch_size` | 16 | |
| `train.lr` | 1e-3 | Adam |
| `train.weight_decay` | 1e-4 | |
| `train.scheduler` | `reduce_on_plateau` | val mIoU alapján |
| `train.dice_weight` | 1.0 | loss = CE + w·Dice |
| `train.log_images_every` | 1 | epoch |
| `paths.runs` | `runs` | |
| `paths.checkpoints` | `checkpoints` | |
| `paths.best_weights` | `models/unet_pet_best.pt` | |

## Függőségek

Támaszkodik: `pyyaml`, `numpy`, `torch`.
Rá támaszkodik: minden más modul és a notebook.

## Buktatók

- A seed a Python `random`, a `numpy`, a `torch` (és cuda) generátorait is
  állítsa; a DataLoadernek `generator` és `worker_init_fn` kell.
- `torch.backends.cudnn.deterministic = True`, `benchmark = False`.
- Útvonalakat a repo gyökeréhez képest oldj fel, ne a futtatás helyéhez.

## Hogyan teszteld

```
pytest tests/test_config.py tests/test_utils.py -q
```
Ellenőrzés: kétszer `set_seed(42)` után ugyanaz a `torch.rand(3)`.

<!--
KARBANTARTÁS: ha ezt a modult módosítod a kódban, frissítsd ezt a docot
UGYANABBAN a commitban (lásd AGENTS.md → Git és commit).
-->
