#!/usr/bin/env bash
# RALPH ciklus — Claude Code vagy Codex.
# Használat:  ./ralph/loop.sh [plan|build] [max_iteráció]
#             AGENT=codex ./ralph/loop.sh build 20
# A CLAUDE_FLAGS / CODEX_FLAGS felülírható. Ellenőrizd a kapcsolókat a saját
# verziódon:  claude --help   /   codex exec --help
set -euo pipefail

MODE="${1:-build}"
MAX="${2:-10}"
AGENT="${AGENT:-claude}"

cd "$(dirname "$0")/.."                       # a repo gyökere
PROMPT="ralph/PROMPT_${MODE}.md"
PLAN="ralph/IMPLEMENTATION_PLAN.md"
[[ -f "$PROMPT" ]] || { echo "Nincs ilyen prompt: $PROMPT"; exit 1; }

BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" == "main" || "$BRANCH" == "master" ]]; then
  echo "Main/master branch-en nem futtatom. Előbb: git switch -c ralph/<téma>"; exit 1
fi

# Engedélyezett parancsok: git, pytest, ruff, python (tesztek, smoke-tanítás).
# A mintákban ne legyen szóköz: a $CLAUDE_FLAGS szavakra bomlik.
CLAUDE_FLAGS="${CLAUDE_FLAGS:---permission-mode acceptEdits --allowedTools Bash(git:*) Bash(pytest:*) Bash(ruff:*) Bash(python:*)}"
CODEX_FLAGS="${CODEX_FLAGS:---sandbox workspace-write}"

for ((i = 1; i <= MAX; i++)); do
  echo "================ RALPH ${MODE} #${i}/${MAX}  (${AGENT}, ${BRANCH})"
  case "$AGENT" in
    claude) claude -p "$(cat "$PROMPT")" $CLAUDE_FLAGS ;;
    codex)  codex exec $CODEX_FLAGS "$(cat "$PROMPT")" ;;
    *) echo "Ismeretlen AGENT: $AGENT"; exit 1 ;;
  esac

  if [[ "$MODE" == "build" ]] && ! grep -q '^- \[ \]' "$PLAN"; then
    echo "Nincs több nyitott feladat a tervben — kész."; break
  fi
done
