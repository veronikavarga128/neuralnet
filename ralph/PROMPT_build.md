# RALPH — BUILD mód

Építési módban vagy. **Egy kör = egy feladat.** Utána kilépsz; a következő kör
friss kontextussal folytatja.

0. Olvasd el: `AGENTS.md`, `ai-context/OVERVIEW.md`,
   `ralph/IMPLEMENTATION_PLAN.md`, és a feladathoz tartozó `ralph/specs/*.md`
   és modul-doc.
1. Válaszd ki a **legfontosabb** nyitott (`- [ ]`) tételt. Csak egyet.
2. Implementálás előtt keresd meg a kódban, nincs-e már (részben) kész.
3. Implementáld az `AGENTS.md` szabályai szerint (konvenciók, buktatók, NE-lista).
4. Futtasd a teszteket és a lintet (`AGENTS.md` → Parancsok). Ha piros:
   javítsd. Ha nem megy: írd a tervbe `BLOKKOLT:` megjegyzéssel, és lépj ki.
5. Frissítsd:
   - a modul-docot (ha változott modul), új modulhoz készíts docot;
   - az `IMPLEMENTATION_PLAN.md`-t: pipáld ki a tételt (`- [x]`), és vedd
     fel az újonnan felfedezett feladatokat;
   - ha tanulság vagy az alapot érintő felismerés van: `ai-context/NOTES.md`.
6. Commit az `AGENTS.md` → „Git és commit” szerint. **Push nincs.**
7. Lépj ki.

TILOS: main/master branch, `.env` módosítása, több feladat egy körben,
tesztek kikapcsolása vagy törlése azért, hogy zöld legyen, **teljes tanítás
indítása** (csak `--epochs 1 --limit-batches` smoke futás), dataset vagy nagy
fájl commitolása. Az `EMBERI LÉPÉS` jelölésű tervtételeket hagyd ki.
