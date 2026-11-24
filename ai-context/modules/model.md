<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: KÉSZ — model.py implementálva, tesztekkel.
-->

# model — U-Net

## Cél

Saját, nulláról megírt U-Net (Ronneberger et al., 2015) szemantikus
szegmentáláshoz: `(B, 3, H, W)` képből `(B, num_classes, H, W)` logitot ad.

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `unet_pet/model.py` | `DoubleConv`, `Down`, `Up`, `UNet`, `build_model`, `count_parameters` |
| `tests/test_model.py` | alak-, gradiens-, paraméterszám- és overfit-tesztek |

## Belépési pontok

- `UNet(in_channels=3, num_classes=3, base_channels=32, depth=4, bilinear=False)`
- `build_model(cfg) -> UNet` — a `model` konfigszekcióból (`model.bilinear` is innen jön)
- `count_parameters(model, trainable_only=True) -> int`

## Architektúra

- `DoubleConv`: (Conv3×3 → BatchNorm → ReLU) × 2, `padding=1` (méretmegtartó); a conv
  `bias=False`, mert a BatchNorm úgyis kivonja az átlagot.
- Encoder: `depth` db `Down` = MaxPool2 + DoubleConv; csatornák 32→64→128→256→512.
- Decoder: `Up` = ConvTranspose2d (vagy bilineáris upsample + 1×1 conv) — mindkettő
  felezi a csatornaszámot → ha kell, `F.pad` a skip méretére → concat `[skip, x]` → DoubleConv.
- Kimenet: 1×1 conv → logitok (softmax a loss-ban / predikciónál).
- Paraméterszám `base_channels=32`, `depth=4` mellett: **7 763 107** (ConvTranspose, ~31 MB
  float32), bilineárissal 7 240 867.
- Bemenet min. `2**depth` px oldalú, különben `ValueError`.

## Függőségek

Támaszkodik: `torch.nn`.
Rá támaszkodik: training, evaluation, notebook.

## Buktatók

- Nem `2**depth`-nel osztható H, W esetén a MaxPool lefelé kerekít; az `Up` ezt
  paddinggel kezeli (tesztelve 100×100 és 97×131 bemenettel).
- A kimenet **logit**, ne tegyél bele softmaxot (a `CrossEntropyLoss` maga csinálja).
- Ne használj előtanított súlyt (a tárgy szabálya, lásd AGENTS.md).

## Hogyan teszteld

```
pytest tests/test_model.py -q
```
Ellenőrizd: kimeneti alak `(2, 3, 128, 128)` véletlen bemenetre; nem 2^depth-nel
osztható bemenetre (pl. 100×100) is helyes alak; egy kis batch-et pár lépésben
túl tud tanulni (a loss csökken).

<!--
KARBANTARTÁS: ha ezt a modult módosítod a kódban, frissítsd ezt a docot
UGYANABBAN a commitban (lásd AGENTS.md → Git és commit).
-->
