<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: KÉSZ — a config.py és a utils.py implementálva, tesztekkel.
-->

# config-utils — konfiguráció, seed, eszköz

## Cél

Egy helyen kezeli a hiperparamétereket (YAML → típusos dataclass), a
reprodukálhatóságot (seed) és az eszközválasztást (cuda / mps / cpu).

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `configs/default.yaml` | az összes hiperparaméter és útvonal |
| `unet_pet/config.py` | `Config` (+ `DataConfig`, `ModelConfig`, `TrainConfig`, `PathsConfig`), `load_config(path, overrides=None)`, `Config.to_dict()` |
| `unet_pet/utils.py` | `project_root()`, `resolve_path(p)`, `set_seed(seed)`, `seed_worker`, `make_generator(seed)`, `get_device(preferred=None)` |
| `tests/test_config.py`, `tests/test_utils.py` | tesztek |

## Belépési pontok

- `load_config()` → `Config` (alapból `configs/default.yaml`; relatív út a repo gyökeréhez képest)
- `load_config(overrides={"train.epochs": "1"})` — pontozott kulcsos felülírás; a CLI-ből
  jövő szöveget a mező típusára konvertálja (`"5e-4"` → `float`)
- `cfg.to_dict()` — sima dict (útvonalak stringként), TensorBoard hparams / JSON mentéshez
- `set_seed(cfg.seed)` — a notebook és a `train.py` legelején
- DataLoader: `generator=make_generator(cfg.seed), worker_init_fn=seed_worker`
- split: `random_split(..., generator=make_generator(cfg.seed))`
- `get_device()` → `torch.device` (cuda → mps → cpu); `get_device("cpu")` kényszerít

## Viselkedés, döntések

- A dataclass-mezőknek **nincs alapértéke**: a YAML az egyetlen forrás. Hiányzó vagy
  ismeretlen (elgépelt) kulcs → `KeyError`, rossz típus / tartomány → `ValueError`.
- Validáció: `val_fraction ∈ (0,1)`, `img_size` osztható `2**depth`-szel,
  `len(mean) == len(std) == in_channels`, `scheduler ∈ {reduce_on_plateau, none}` stb.
- A `Path` mezők betöltéskor abszolút útra oldódnak a repo gyökeréhez képest.
- `set_seed` nem hívja a `torch.use_deterministic_algorithms(True)`-t, mert egyes CUDA
  műveleteknek (pl. 2D cross-entropy backward) nincs determinisztikus változata.

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
