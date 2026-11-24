# specs/ — követelmények a RALPH-hoz

Egy fájl = egy téma (pl. `auth.md`, `export.md`). A spec a **mit és miért**,
nem a hogyan — az utóbbi az ügynök dolga.

Ajánlott szerkezet:

```
# <téma>
## Cél          — miért kell, kinek
## Viselkedés   — mit csináljon (felhasználói szemmel)
## Elfogadási kritériumok — ellenőrizhető pontok, lehetőleg teszttel
## Nem cél      — mit NE csináljon most
```

A specet ember írja (akár egy tervező chattel közösen),
és a projekt gazdája hagyja jóvá.

## A projekt specjei

| Fájl | Téma |
|------|------|
| [`01-reprodukalhatosag-es-konfig.md`](01-reprodukalhatosag-es-konfig.md) | seed, konfig, eszköz |
| [`02-adat.md`](02-adat.md) | Oxford-IIIT Pet betöltés, transzformáció, split |
| [`03-modell.md`](03-modell.md) | U-Net |
| [`04-tanitas-tensorboard.md`](04-tanitas-tensorboard.md) | tanítás, loss, TensorBoard |
| [`05-kiertekeles.md`](05-kiertekeles.md) | metrikák, ábrák |
| [`06-notebook-es-beadando.md`](06-notebook-es-beadando.md) | notebook, requirements, súlyok, README |
