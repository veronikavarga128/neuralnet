<!--
  AI-NAK SZÁNT FÁJL — modul-doc (a _MODULE_TEMPLATE.md alapján)
  ÁLLAPOT: TERVEZETT — a kód még nem létezik. Amikor elkészül, a doc-keeper
  ügynök frissíti a valós fájlnevekre és függvényekre.
-->

# model — U-Net

## Cél

Saját, nulláról megírt U-Net (Ronneberger et al., 2015) szemantikus
szegmentáláshoz: `(B, 3, H, W)` képből `(B, num_classes, H, W)` logitot ad.

## Kulcsfájlok

| Fájl | Szerep |
|------|--------|
| `unet_pet/model.py` | `DoubleConv`, `Down`, `Up`, `UNet` |

## Belépési pontok

- `UNet(in_channels=3, num_classes=3, base_channels=32, depth=4, bilinear=False)`
- `count_parameters(model) -> int`

## Architektúra

- `DoubleConv`: (Conv3×3 → BatchNorm → ReLU) × 2, `padding=1` (méretmegtartó).
- Encoder: `depth` db `Down` = MaxPool2 + DoubleConv; csatornák 32→64→128→256→512.
- Decoder: `Up` = ConvTranspose2d (vagy bilineáris upsample + 1×1 conv) →
  concat a megfelelő skip-pel → DoubleConv.
- Kimenet: 1×1 conv → logitok (softmax a loss-ban / predikciónál).
- Paraméterszám `base_channels=32`, `depth=4` mellett ≈ 7,8 M (~31 MB float32).

## Függőségek

Támaszkodik: `torch.nn`.
Rá támaszkodik: training, evaluation, notebook.

## Buktatók

- A bemeneti H, W legyen `2**depth`-nel osztható (128 jó); ha nem, a `Up`
  blokk paddingelje a felsamplingolt tenzort a skip méretére.
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
