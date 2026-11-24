<!--
================================================================
  AI-NAK SZÁNT FÁJL — CLAUDE BELÉPŐ
  A szabályok forrása az AGENTS.md; ezt az alábbi @import tölti be.
================================================================
-->
@AGENTS.md

# CLAUDE.md — neuralnet

A fenti `@AGENTS.md` sor betölti a projekt szabályait. Ez egy önálló repo,
nincs szülő-mappa: `../AGENTS.md`, `../PROJECT.md`, `../rules/` **nem létezik**,
ne keresd.

Olvasási sorrend egy új feladatnál: [`AGENTS.md`](AGENTS.md) →
[`ai-context/OVERVIEW.md`](ai-context/OVERVIEW.md) → a témához tartozó modul-doc
→ a feladat spec-je a [`ralph/specs/`](ralph/specs/) alól.

## Claude-specifikus kiegészítések

- **Válasz nyelve:** a felhasználóval magyarul kommunikálj.
- **Kis lépések:** egyszerre egy modult / egy tervtételt valósíts meg, utána
  futtasd a `pytest -q` és `ruff check .` parancsot, és csak zöld állapotban lépj tovább.
- **Előbb terv, aztán kód:** nagyobb feladatnál először mondd el röviden, mit
  fogsz csinálni és mely fájlokat érinti, és várd meg a jóváhagyást.
- **Hosszú futások:** teljes tanítást (több perc / óra) ne indíts el magadtól.
  Ellenőrzéshez használj 1 epoch-os, kis részhalmazos futást
  (`--epochs 1 --limit-batches 5` jellegű kapcsolók a `train.py`-ban).
- **Notebook szerkesztése:** a `.ipynb` JSON; cellát a NotebookEdit eszközzel
  szerkessz, ne kézi JSON-szerkesztéssel. A kimeneteket (outputs) commit előtt
  csak akkor töröld, ha a project owner kéri — a beadandóban hasznos, ha látszanak az eredmények.
- **Tanulás a cél:** Nem triviális döntésnél (loss, architektúra, metrika) írj 2-3 mondatos
  indoklást a kódkommentbe vagy a notebook markdown-cellájába.
- Ügynökök: [`.claude/agents/`](.claude/agents/) — vékony adapterek, a szerepek
  forrása az [`ai-agents/`](ai-agents/) mappa.
