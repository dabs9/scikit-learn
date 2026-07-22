#!/usr/bin/env bash
# One agent seat: non-interactive claude invocation with retry wrapper.
# usage: seat.sh NAME PROMPT_FILE OUT_FILE CWD [readonly|write]
# Success = exit 0 AND non-empty output file (checked BEFORE any content
# scanning — a reviewer's own text may legitimately mention e.g. rate limits).
set -uo pipefail

NAME=$1; PROMPT=$2; OUT=$3; SEATCWD=$4; MODE=${5:-readonly}
RETRIES=${SEAT_RETRIES:-3}
MODEL=${REVIEW_MODEL:?REVIEW_MODEL not set}
TIMEOUT_S=${SEAT_TIMEOUT:-4500}
LOG=${OUT}.log

READONLY_TOOLS="Read,Grep,Glob,Bash(git:*),Bash(rg:*),Bash(grep:*),Bash(ls:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(sed:*),Bash(awk:*),Bash(find:*),Bash(python3:*)"
WRITE_TOOLS="Read,Grep,Glob,Write,Edit,Bash(git:*),Bash(ls:*),Bash(cat:*),Bash(python3:*)"

run_once() {
  local attempt=$1 tools flags=()
  if [[ $MODE == write ]]; then
    tools=$WRITE_TOOLS
    flags+=(--permission-mode acceptEdits)
  else
    tools=$READONLY_TOOLS
  fi
  local timer=()
  command -v timeout >/dev/null 2>&1 && timer=(timeout "$TIMEOUT_S")
  if (( attempt % 2 == 1 )); then
    (cd "$SEATCWD" && "${timer[@]}" claude -p --model "$MODEL" \
        --allowedTools "$tools" "${flags[@]}" <"$PROMPT") >"$OUT.tmp" 2>>"$LOG"
  else
    # alternate permission style in case the worker CLI rejects allowedTools
    (cd "$SEATCWD" && "${timer[@]}" claude -p --model "$MODEL" \
        --dangerously-skip-permissions <"$PROMPT") >"$OUT.tmp" 2>>"$LOG"
  fi
}

seat_t0=$(date +%s)
attempt=1
while (( attempt <= RETRIES )); do
  echo "[seat:$NAME] attempt $attempt $(date -u +%FT%TZ)" >>"$LOG"
  if run_once "$attempt" && [[ -s "$OUT.tmp" ]]; then
    mv "$OUT.tmp" "$OUT"
    echo "[seat:$NAME] ok in $(( $(date +%s) - seat_t0 ))s" >>"$LOG"
    exit 0
  fi
  echo "[seat:$NAME] attempt $attempt failed" >>"$LOG"
  sleep $(( attempt * 20 ))
  attempt=$(( attempt + 1 ))
done
echo "[seat:$NAME] FAILED after $RETRIES attempts (see $LOG)" >&2
exit 1
