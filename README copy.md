# projekt_git_repo_1

<!-- TODO: egy mondatos leírás mit csinál ez a service -->

Ez egy önálló service, saját git-repóval. A multi-service kontextushoz lásd a
gyökér [`PROJECT.md`](../PROJECT.md) fájlt.

## AI-kontextus

Ha AI-eszközzel dolgozol ezen a service-en, olvasd el:

1. [`../PROJECT.md`](../PROJECT.md) és [`../AGENTS.md`](../AGENTS.md) — projekt-szintű áttekintés és szabályok
2. [`AGENTS.md`](AGENTS.md) — ennek a service-nek a szabályai
3. [`ai-context/OVERVIEW.md`](ai-context/OVERVIEW.md) — a service részletes áttekintése
4. a témához tartozó modul-doc az [`ai-context/modules/`](ai-context/modules/) alól

## Mappák

```
.
├── AGENTS.md            # service AI-szabályok (forrás)
├── CLAUDE.md            # Claude belépő: @AGENTS.md import (vagy pointer)
├── ai-context/          # AI-nak szánt doksi: OVERVIEW, modules/, NOTES.md
├── ai-agents/           # a service ügynökei (szerep-források)
├── .claude/agents/      # Claude Code adapterek
├── .codex/agents/       # Codex adapterek
├── ralph/               # RALPH ciklus: specs/, terv, promptok, loop.sh
└── .github/CODEOWNERS   # alap-réteg → szakmai vezető
```

## Telepítés / futtatás

<!-- TODO: telepítési és futtatási lépések -->

```
# TODO: pl. függőségek telepítése, indítás
```

## Környezeti változók

Másold a [`.env.example`](.env.example) fájlt `.env`-re és töltsd ki.
A `.env` **soha nem kerül gitbe**.
