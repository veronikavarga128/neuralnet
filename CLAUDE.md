<!--
================================================================
  AI-NAK SZÁNT FÁJL — CLAUDE BELÉPŐ (service szint)
  Két változat: A) import (aktív), B) pointer — lásd lent és a
  gyökér TEMPLATE_GUIDE.md-t. A szabályok forrása az AGENTS.md.
================================================================
-->
@AGENTS.md

# CLAUDE.md — projekt_git_repo_1

**A) Import — aktív.** A fenti `@AGENTS.md` sor betölti ennek a service-nek az
`AGENTS.md`-jét. A gyökér `../CLAUDE.md`-t (és vele a gyökér `AGENTS.md`-t) a
Claude Code magától betölti, ha a session a projekt-mappán belül indul, mert a
szülőkönyvtárak CLAUDE.md-it is beolvassa.

**B) Pointer — ha törlöd az import sort.** Olvasási sorrend:
`../PROJECT.md` → `../AGENTS.md` → [`AGENTS.md`](AGENTS.md) →
[`ai-context/OVERVIEW.md`](ai-context/OVERVIEW.md) → a témához tartozó modul-doc.

## Claude-specifikus kiegészítések

- Ügynökök: [`.claude/agents/`](.claude/agents/) — vékony adapterek, a szerepek
  forrása az [`ai-agents/`](ai-agents/) mappa.
- <!-- TODO: csak Claude-ra igaz service-szabályok -->
