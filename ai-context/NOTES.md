<!--
================================================================
  AI-NAK SZÁNT FÁJL — NAPLÓ (inbox a projekt gazdájának)
  Ide az ügynök és a fejlesztő is HOZZÁFŰZHET. Ami az alap-réteget
  érinti (AGENTS.md, CLAUDE.md, OVERVIEW.md, ai-agents/), az itt
  javaslatként jelenik meg; a project owner dönt róla.
================================================================
-->

# NOTES — jegyzetek és javaslatok

## Bejegyzés formátuma (legújabb felül)

```
### YYYY-MM-DD — <branch> — <ki: ügynök neve / ember>
- Típus: új-modul | változás | buktató | javaslat | kísérlet
- Mi: <1-3 mondat>
- Érintett: <fájl(ok)>
- Státusz: nyitott
```

Átnézés után a státusz `beépítve` / `elvetve` lesz; a régi, lezárt
bejegyzéseket időnként törölni lehet.

**Kísérletek:** minden teljes tanítás után ide is kerüljön egy `kísérlet`
bejegyzés (run neve, fő beállítások, legjobb val mIoU, teszt mIoU) — ebből
készül a notebook „Kísérletek összevetése” táblázata.

---

### 2026-11-24 — main — ember
- Típus: javaslat
- Mi: Az AI-template kitöltve a U-Net / Oxford-IIIT Pet projektre. A modul-docok
  még TERVEZETT állapotúak (nincs kód); a kód elkészülte után a doc-keeper
  frissítse őket a valós fájlokra.
- Érintett: AGENTS.md, ai-context/OVERVIEW.md, ai-context/modules/*.md, ralph/specs/*.md
- Státusz: nyitott
