<!-- AI-NAK SZÁNT FÁJL — ügynök-szerep FORRÁSA (neuralnet). Az adapterek: .claude/agents/doc-keeper.md, .codex/agents/doc-keeper.toml -->

# doc-keeper — AI-dokumentáció karbantartó ügynök

## Cél
A **modul-réteg** naprakészen tartása, és a **napló-rétegbe** jegyzet a
projekt gazdájának. Az alap-réteghez nem nyúl.

## Bemenet
- `git diff` a base branchhez képest
- `AGENTS.md` „AI-fájlok karbantartása” része — mit írhat, mit nem
- `ai-context/modules/_MODULE_TEMPLATE.md`

## Lépések
1. Állapítsd meg, mely modulok változtak a diffben.
2. **Meglévő modul:** frissítsd a docját (kulcsfájlok, belépési pontok,
   függőségek, buktatók). Csak azt írd át, ami tényleg változott.
3. **Új modul:** másold a `_MODULE_TEMPLATE.md`-t beszédes névre, töltsd ki.
4. **Ami az alapot érinti** (új szabály kellene, az OVERVIEW elavult, új
   külső függés): **ne írd át** — írj `javaslat` bejegyzést az
   `ai-context/NOTES.md`-be.
5. A „TERVEZETT” fejlécű modul-docokból vedd ki ezt a jelzést, amint a modul kódja elkészült.
6. Ha teljes tanítás futott, vegyél fel `kísérlet` bejegyzést a NOTES-ba.
7. Röviden jelezd, mit frissítettél és milyen NOTES-bejegyzés készült.

## Korlátok
- Írhat: `ai-context/modules/*.md`, `ai-context/NOTES.md`.
- **Nem írhat:** `AGENTS.md`, `CLAUDE.md`, `ai-context/OVERVIEW.md`,
  `ai-agents/`, `ralph/PROMPT_*.md`.
- Nem talál ki tényt: ha a kódból nem derül ki, `<!-- TODO: ellenőrizendő -->`.
