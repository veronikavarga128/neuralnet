<!--
================================================================
  AI-NAK SZÁNT FÁJL — service-szintű szabályok
  Ez a service AGENTS.md-je. A projekt-szintű szabályok forrása a
  gyökér ../AGENTS.md — ez a fájl csak a service-specifikus
  kiegészítéseket tartalmazza.
================================================================
-->

# AGENTS.md — projekt_git_repo_1 (service szint)

Ez a fájl a **projekt-szintű** [`../AGENTS.md`](../AGENTS.md)-re épül. Először azt
olvasd, itt csak az ehhez a service-hez tartozó extra szabályok vannak.

## Kötelező olvasás e service-en dolgozva

1. [`../PROJECT.md`](../PROJECT.md) és [`../AGENTS.md`](../AGENTS.md)
2. Ez a fájl
3. [`ai-context/OVERVIEW.md`](ai-context/OVERVIEW.md)
4. a témához tartozó modul-doc az [`ai-context/modules/`](ai-context/modules/) alól

## Service-specifikus szabályok

<!-- TODO: ide azok a szabályok, amik CSAK erre a service-re igazak. Példák: -->

- **Stack:** <!-- TODO: nyelv / keretrendszer / verzió -->
- **Belépési pont:** <!-- TODO: hol indul a service (fájl / parancs) -->
- **Adatbázis / migráció:** <!-- TODO: van-e, hogyan kezeld -->
- **Buktatók:** <!-- TODO: amit egy AI könnyen elront ezen a service-en -->

## Parancsok (build / teszt / lint)

<!-- TODO: pontos parancsok — a RALPH ciklus és az ügynökök ezekkel validálnak -->

| Mire | Parancs |
|------|---------|
| Telepítés | `<!-- TODO -->` |
| Teszt | `<!-- TODO -->` |
| Lint / típus | `<!-- TODO -->` |

## Ügynökök (ai-agents/)

A service saját ügynökei. A szerepek forrása az [`ai-agents/`](ai-agents/)
mappa, a `.claude/agents/` és `.codex/agents/` csak adapter.

| Szerep | Mikor |
|--------|-------|
| `reviewer` | commit / PR előtt |
| `test-writer` | új feature vagy bugfix után |
| `doc-keeper` | feladat végén: modul-doc és NOTES |

## AI-fájlok karbantartása

Modul-docot frissíthetsz és létrehozhatsz; az alapot (`AGENTS.md`,
`ai-context/OVERVIEW.md`, `ai-agents/`) nem írod át — javaslat az
[`ai-context/NOTES.md`](ai-context/NOTES.md)-be. Lásd
[`../rules/ai-docs-ownership.md`](../rules/ai-docs-ownership.md).

## RALPH

Autonóm ciklus: [`ralph/README.md`](ralph/README.md).

## Modul-térkép

<!-- TODO: ahogy készülnek a modul-docok, vezesd fel ide -->

| Modul | Doc |
|-------|-----|
| <!-- TODO --> | `ai-context/modules/<!-- TODO -->.md` |
