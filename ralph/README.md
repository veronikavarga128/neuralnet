<!-- AI-NAK ÉS EMBERNEK — RALPH ciklus ehhez a service-hez -->

# ralph/ — autonóm fejlesztési ciklus (projekt_git_repo_1)

A RALPH (Geoffrey Huntley technikája) egy **egyszerű ciklus**: ugyanazt a
promptot adja újra és újra az ügynöknek, **minden körben friss kontextussal**.
Az állapot nem a beszélgetésben, hanem a **fájlokban** él.

```
while :; do cat PROMPT.md | claude -p ; done      # a lényeg egy sorban
```

## Fájlok

| Fájl | Szerep | Ki írja |
|------|--------|---------|
| `specs/*.md` | A követelmények: egy fájl egy téma | ember (szakmai vezető / fejlesztő) |
| `PROMPT_plan.md` | PLAN mód: hiányelemzés, terv írása, kód nélkül | szakmai vezető (alap-réteg) |
| `PROMPT_build.md` | BUILD mód: egy feladat körönként | szakmai vezető (alap-réteg) |
| `IMPLEMENTATION_PLAN.md` | A kör-közi állapot: priorizált `- [ ]` lista | az ügynök (napló-réteg) |
| `loop.sh` | A ciklus (Claude vagy Codex) | — |

## Futtatás

```bash
git switch -c ralph/YYYYMMDD-projekt_git_repo_1-<téma>     # SOHA ne main-en
./ralph/loop.sh plan 3                      # terv elkészítése (max 3 kör)
# → nézd át és javítsd kézzel az IMPLEMENTATION_PLAN.md-t!
./ralph/loop.sh build 20                    # építés (max 20 kör)
AGENT=codex ./ralph/loop.sh build 20        # ugyanez Codexszel
```

## A backpressure a lényeg

A ciklust a **tesztek, a lint és a típusellenőrzés** tartják kordában. Ha
nincs gyors, megbízható teszt (`AGENTS.md` → Parancsok), a RALPH gyorsan
gyártott hulladékot termel. Előbb legyen tesztparancs, utána ciklus.

## Biztonság

- Konténerben vagy sandboxban futtasd. A jogosultság-kihagyó kapcsolókat
  (`--dangerously-skip-permissions` és társai) csak izolált környezetben.
- Mindig iteráció-limittel (`loop.sh` 2. paramétere).
- Push nincs a ciklusban: a branch-et ember nézi át és mergeli.
