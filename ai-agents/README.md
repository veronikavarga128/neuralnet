<!-- AI-NAK SZÁNT FÁJL — a service ügynök-mappája -->

# ai-agents/ — projekt_git_repo_1 ügynökei

Ez a mappa a **service saját ügynök-csapata**. Minden repónak (pl. FE és BE)
saját `ai-agents/` mappája van, mert a jó reviewer vagy tesztíró
**stack-függő**.

## Ugyanaz a minta, mint a CLAUDE.md → AGENTS.md

```
ai-agents/<szerep>.md          ← A FORRÁS: a szerep teljes leírása (eszközfüggetlen)
.claude/agents/<szerep>.md     ← Claude Code adapter (Markdown + YAML frontmatter)
.codex/agents/<szerep>.toml    ← Codex adapter (TOML)
```

Az adapterek csak nevet, leírást, jogosultságot adnak meg, és a forrásfájlra
mutatnak. **Szerepet mindig az `ai-agents/`-ben módosíts**, az adaptert csak
akkor, ha a név, leírás vagy eszközkészlet változik.

## Szerepek

| Szerep | Fájl | Ír kódot? | Röviden |
|--------|------|-----------|---------|
| reviewer | [`reviewer.md`](reviewer.md) | nem (csak olvas) | review commit / PR előtt |
| test-writer | [`test-writer.md`](test-writer.md) | csak tesztet | teszt a változtatáshoz vagy bughoz |
| doc-keeper | [`doc-keeper.md`](doc-keeper.md) | csak modul-docot és NOTES-t | ai-context naprakészen tartása |

## Hívás

- **Claude Code:** „Futtasd a reviewer ügynököt a branch változásaira.”
  (A Claude a leírás alapján magától is delegálhat.)
- **Codex:** a custom agentet kérésre indítja: „Spawn the reviewer agent on
  the current diff.”

## Új szerep felvétele

1. Írd meg az `ai-agents/<szerep>.md`-t (a meglévők szerkezetével).
2. Másold az egyik adapter-párt, és írd át a nevet és a leírást.
3. Vedd fel a fenti táblába és a service `AGENTS.md` „Ügynökök” részébe.

Az ügynök-mappa az **alap-réteg** része: a szakmai vezető tartja karban
(lásd `../rules/ai-docs-ownership.md`).
