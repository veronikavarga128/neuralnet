# Reprodukálhatóság és konfiguráció

## Cél
A tárgy kötelezően elvárja a fix seedet és a konfigurációs fájlt; az oktatónak
ugyanazt kell kapnia, amit mi.

## Viselkedés
- Minden hiperparaméter és útvonal a `configs/default.yaml`-ban van.
- `load_config()` típusos `Config` dataclasst ad; CLI-ből egyes értékek felülírhatók.
- `set_seed(seed)` minden véletlenforrást beállít; a DataLoader is seedelt.
- `get_device()` automatikusan választ: cuda → mps → cpu.

## Elfogadási kritériumok
- [ ] `pytest tests/test_config.py tests/test_utils.py` zöld.
- [ ] Két `set_seed(42)` után azonos véletlenszámok és azonos train/val split.
- [ ] Nincs beégetett hiperparaméter a `unet_pet/` kódban (review ellenőrzi).

## Nem cél
Hydra, MLflow vagy más konfig-/kísérletkezelő keretrendszer.
