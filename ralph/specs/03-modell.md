# Modell: U-Net

## Cél
Saját, nulláról megírt U-Net — a tárgy szabálya miatt (órán nem volt
szegmentálás és U-Net), és mert ez a szegmentálás klasszikus architektúrája.

## Viselkedés
- `UNet(in_channels=3, num_classes=3, base_channels=32, depth=4, bilinear=False)`.
- Encoder–bottleneck–decoder, skip kapcsolatokkal; kimenet logit `(B, C, H, W)`.
- Bármilyen bemeneti méretre a kimenet mérete = bemenet mérete.

## Elfogadási kritériumok
- [ ] `pytest tests/test_model.py` zöld (alakok, nem osztható méret, overfit-egy-batch).
- [ ] Paraméterszám kiírható; alapkonfigon ≈ 7–8 M.
- [ ] Nincs előtanított súly és külső szegmentáló könyvtár.

## Nem cél
Attention U-Net, U-Net++, pretrained encoder (esetleg későbbi kísérletként, a project owner döntése alapján).
