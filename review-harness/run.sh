#!/usr/bin/env bash
# Fan-out review-harness pipeline runner (see REVIEW_HARNESS_MISSION.md build spec).
# Resumable: every completed seat/stage leaves a marker in runs/<label>/markers/;
# rerunning skips completed work. All intermediate artifacts stay on disk.
set -uo pipefail

RH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(cd "$RH/.." && pwd -P)
# shellcheck disable=SC1091
source "$RH/config.env"
export REVIEW_MODEL
# only force a thinking budget when one is set; empty = model adaptive default
if [[ -n "${MAX_THINKING_TOKENS:-}" ]]; then export MAX_THINKING_TOKENS; else unset MAX_THINKING_TOKENS; fi

RUN=$RH/runs/$RUN_LABEL
MARK=$RUN/markers
SEAT=$RH/scripts/seat.sh
LINT=$RH/scripts/contract_lint.py
SNAP=$RH/scripts/snapshot_check.py
AUDIT=$RH/scripts/fidelity_audit.py
TIMINGS=$RUN/timings.tsv
mkdir -p "$RUN" "$MARK" "$RUN/draws"

note() { printf '[harness %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { note "FATAL: $*"; exit 1; }
done_marker() { [[ -f "$MARK/$1.done" ]]; }
mark_done() { touch "$MARK/$1.done"; }

STAGE_T0=0
t_start() { STAGE_T0=$(date +%s); }
t_end() { printf '%s\t%s\t%s\n' "$1" "$(( $(date +%s) - STAGE_T0 ))" "$(date -u +%FT%TZ)" >> "$TIMINGS"; }

count_findings() { grep -c '^### F' "$1" 2>/dev/null || true; }

# ---------------------------------------------------------------- objects + worktree
ensure_objects() {
  local sha
  for sha in "$HEAD_SHA" "$BASE_SHA"; do
    if ! git -C "$ROOT" rev-parse --quiet --verify "$sha^{commit}" >/dev/null; then
      note "fetching missing commit $sha"
      git -C "$ROOT" fetch --unshallow origin 2>/dev/null || true
      git -C "$ROOT" fetch origin "refs/heads/pr-$PR_NUMBER:refs/remotes/origin/pr-$PR_NUMBER" 2>/dev/null || true
      git -C "$ROOT" fetch origin "$sha" 2>/dev/null || true
      git -C "$ROOT" fetch "$UPSTREAM_URL" "refs/pull/$PR_NUMBER/head" 2>/dev/null || true
      git -C "$ROOT" fetch "$UPSTREAM_URL" "$sha" 2>/dev/null || true
    fi
    git -C "$ROOT" rev-parse --quiet --verify "$sha^{commit}" >/dev/null \
      || die "commit $sha unavailable after fetch attempts"
  done
}

WT=${REVIEW_WT:-$(dirname "$ROOT")/rh-wt-$RUN_LABEL}
ensure_worktree() {
  if [[ ! -d "$WT/.git" && ! -f "$WT/.git" ]]; then
    git -C "$ROOT" worktree add --detach "$WT" "$HEAD_SHA" || die "worktree add failed"
  fi
  [[ "$(git -C "$WT" rev-parse HEAD)" == "$HEAD_SHA" ]] || die "worktree not at HEAD_SHA"
}

# ---------------------------------------------------------------- prompt templating
render_prompt() { # render_prompt <template> <outfile>
  python3 - "$1" "$2" <<PY
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
mapping = {
    "PLAN": "$RUN/PLAN.md", "MANIFEST": "$RUN/DIFF_MANIFEST.md",
    "HUNKS": "$WT/hunks", "WORKTREE": "$WT", "RUNDIR": "$RUN",
    "BASE_SHA": "$BASE_SHA", "HEAD_SHA": "$HEAD_SHA",
    "PR_NUMBER": "$PR_NUMBER", "K": "$K_DRAWS",
}
for key, val in mapping.items():
    text = text.replace("{{%s}}" % key, val)
Path(sys.argv[2]).write_text(text)
PY
}

# ------------------------------------------------- draft -> lint -> revise loop
lint_loop() { # lint_loop <findings_file> <base_prompt> <seat_name>
  local file=$1 base_prompt=$2 name=$3 round=1 rp
  while true; do
    if python3 "$LINT" "$file" > "$file.lint" 2>&1; then
      return 0
    fi
    (( round > 3 )) && { note "lint still dirty after 3 revisions: $file"; cat "$file.lint"; return 1; }
    note "lint violations ($name, revision $round)"
    rp="$file.revise$round.prompt"
    {
      cat "$RH/prompts/revise_header.md"
      printf '\n## Lint violations\n'; cat "$file.lint"
      printf '\n## Your previous output (revise this)\n-----BEGIN PREVIOUS OUTPUT-----\n'
      cat "$file"
      printf '\n-----END PREVIOUS OUTPUT-----\n\n## Original task prompt\n'
      cat "$base_prompt"
    } > "$rp"
    bash "$SEAT" "$name-rev$round" "$rp" "$file" "$WT" readonly || return 1
    round=$(( round + 1 ))
  done
}

run_seat_linted() { # run_seat_linted <seat_name> <prompt> <out>
  local name=$1 prompt=$2 out=$3
  if done_marker "$name"; then note "skip $name (marker)"; return 0; fi
  bash "$SEAT" "$name" "$prompt" "$out" "$WT" readonly || return 1
  lint_loop "$out" "$prompt" "$name" || return 1
  mark_done "$name"
}

# ================================================================ stage 0: inputs
if ! done_marker stage0; then
  t_start; note "stage 0: inputs"
  ensure_objects
  ensure_worktree
  [[ -s "$RUN/PLAN.md" ]] || die "PLAN.md missing at $RUN/PLAN.md (must be committed with the harness)"
  python3 "$RH/scripts/diff_manifest.py" "$BASE_SHA" "$HEAD_SHA" --repo "$ROOT" \
    --manifest "$RUN/DIFF_MANIFEST.md" --hunks-dir "$WT/hunks" || die "diff_manifest failed"
  for p in reviewer merge depth closure inverse synthesis; do
    render_prompt "$RH/prompts/$p.md" "$RUN/$p.prompt.md"
  done
  t_end stage0_inputs; mark_done stage0
else
  ensure_objects; ensure_worktree
  note "skip stage 0 (marker)"
fi

[[ "${STOP_AFTER:-}" == "stage0" ]] && { note "STOP_AFTER=stage0 — exiting after inputs"; exit 0; }

# ================================================================ stage 1: draws
if ! done_marker stage1; then
  t_start; note "stage 1: $K_DRAWS independent draws"
  pids=()
  for i in $(seq 1 "$K_DRAWS"); do
    run_seat_linted "draw_$i" "$RUN/reviewer.prompt.md" "$RUN/draws/draw_$i.md" &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  (( fail )) && die "stage 1: at least one draw seat failed (no silent partial results)"
  t_end stage1_draws; mark_done stage1
else note "skip stage 1 (marker)"; fi

# ================================================================ stage 2: merge
if ! done_marker stage2; then
  t_start; note "stage 2: union merge"
  max_draw=0
  for f in "$RUN"/draws/draw_*.md; do
    c=$(count_findings "$f"); (( c > max_draw )) && max_draw=$c
  done
  merge_try=1
  while true; do
    run_seat_linted "merge_try$merge_try" "$RUN/merge.prompt.md" "$RUN/merged.md" \
      || die "stage 2: merge seat failed"
    merged_count=$(count_findings "$RUN/merged.md")
    if (( merged_count >= max_draw )); then break; fi
    note "MERGE SHRINKAGE: merged=$merged_count < max draw=$max_draw (invariant: union can never shrink)"
    echo "merge_shrinkage try$merge_try merged=$merged_count max_draw=$max_draw" >> "$RUN/harness_incidents.log"
    (( merge_try >= 2 )) && die "stage 2: merge output smaller than a single draw twice — merge broke"
    rm -f "$MARK/merge_try$merge_try.done"
    {
      cat "$RUN/merge.prompt.md"
      printf '\n\n# RETRY NOTICE\nYour previous merge produced %s findings but one draw alone has %s. A union can never have fewer findings than any single input. Re-merge, dropping nothing.\n' "$merged_count" "$max_draw"
    } > "$RUN/merge.retry.prompt.md"
    bash "$SEAT" "merge_retry" "$RUN/merge.retry.prompt.md" "$RUN/merged.md" "$WT" readonly || die "merge retry failed"
    lint_loop "$RUN/merged.md" "$RUN/merge.retry.prompt.md" "merge_retry" || die "merge retry lint failed"
    break
  done
  note "merged findings: $(count_findings "$RUN/merged.md") (max single draw: $max_draw)"
  t_end stage2_merge; mark_done stage2
else note "skip stage 2 (marker)"; fi

# ================================================================ stage 3: depth (add-only)
if ! done_marker stage3; then
  t_start; note "stage 3: depth (add-only strengthening)"
  try=1
  while true; do
    if (( try == 1 )); then
      bash "$SEAT" "depth_try$try" "$RUN/depth.prompt.md" "$RUN/depth.md" "$WT" readonly || die "depth seat failed"
    fi
    lint_ok=0; snap_ok=0
    python3 "$LINT" "$RUN/depth.md" > "$RUN/depth.md.lint" 2>&1 && lint_ok=1
    python3 "$SNAP" "$RUN/merged.md" "$RUN/depth.md" > "$RUN/depth.md.snap" 2>&1 && snap_ok=1
    (( lint_ok && snap_ok )) && break
    (( try >= 3 )) && {
      echo "depth_failed_after_3 falling back to merged.md" >> "$RUN/harness_incidents.log"
      note "stage 3 could not produce a valid add-only doc in 3 tries; falling back to merged.md verbatim"
      cp "$RUN/merged.md" "$RUN/depth.md"
      break
    }
    note "depth violations (try $try)"
    {
      cat "$RUN/depth.prompt.md"
      printf '\n\n# RETRY NOTICE — your previous output failed deterministic checks\n## Contract lint\n'
      cat "$RUN/depth.md.lint"
      printf '\n## Snapshot (add-only) check\n'
      cat "$RUN/depth.md.snap"
      printf '\nReproduce the ORIGINAL merged.md content byte-identical and only APPEND clean blocks.\n'
    } > "$RUN/depth.retry$try.prompt.md"
    try=$(( try + 1 ))
    bash "$SEAT" "depth_try$try" "$RUN/depth.retry$((try-1)).prompt.md" "$RUN/depth.md" "$WT" readonly || die "depth retry failed"
  done
  note "depth findings: $(count_findings "$RUN/depth.md")"
  t_end stage3_depth; mark_done stage3
else note "skip stage 3 (marker)"; fi

# ================================================================ stage 4: closure
if ! done_marker stage4; then
  t_start; note "stage 4: rule closure sweep"
  run_seat_linted "closure" "$RUN/closure.prompt.md" "$RUN/closure.md" || die "stage 4 failed"
  t_end stage4_closure
  mark_done stage4
else note "skip stage 4 (marker)"; fi

# ================================================================ stage 5: inverse
if ! done_marker stage5; then
  t_start; note "stage 5: inverse conformance"
  run_seat_linted "inverse" "$RUN/inverse.prompt.md" "$RUN/inverse.md" || die "stage 5 failed"
  t_end stage5_inverse
  mark_done stage5
else note "skip stage 5 (marker)"; fi

# ================================================ stage 6+7: synthesis + fidelity loop
check_synthesis_files() {
  python3 - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
try:
    findings = json.loads((run / "findings.json").read_text())
    json.loads((run / "synthesis_ledger.json").read_text())
    assert (run / "report.md").stat().st_size > 0
    assert isinstance(findings, list)
    required = {"id", "severity", "title", "file", "line", "scenario", "contract", "instances", "body"}
    for f in findings:
        missing = required - set(f)
        assert not missing, f"finding {f.get('id')} missing keys {missing}"
except Exception as exc:
    print(f"synthesis outputs invalid: {exc}")
    sys.exit(1)
PY
}

if ! done_marker stage7; then
  t_start; note "stage 6+7: synthesis + fidelity audit"
  try=1
  prompt="$RUN/synthesis.prompt.md"
  while true; do
    if ! done_marker "synthesis_try$try"; then
      bash "$SEAT" "synthesis_try$try" "$prompt" "$RUN/synthesis_reply_$try.txt" "$WT" write \
        || die "synthesis seat failed"
      mark_done "synthesis_try$try"
    fi
    files_err=$(check_synthesis_files 2>&1) || {
      note "synthesis files invalid: $files_err"
      (( try >= 3 )) && die "stage 6: synthesis files invalid after 3 tries"
      prompt="$RUN/synthesis.retry$try.prompt.md"
      { cat "$RUN/synthesis.prompt.md"; printf '\n\n# RETRY NOTICE\nPrevious attempt outputs were invalid: %s\nRewrite ALL THREE files correctly.\n' "$files_err"; } > "$prompt"
      try=$(( try + 1 )); continue
    }
    if python3 "$AUDIT" "$RUN" > "$RUN/fidelity_audit.out" 2>&1; then
      note "fidelity audit clean"
      break
    fi
    note "fidelity audit violations (try $try)"
    cat "$RUN/fidelity_audit.out"
    echo "fidelity_violation try$try" >> "$RUN/harness_incidents.log"
    (( try >= 3 )) && die "stage 7: fidelity loop exhausted after 3 tries"
    prompt="$RUN/synthesis.retry$try.prompt.md"
    {
      cat "$RUN/synthesis.prompt.md"
      printf '\n\n# RETRY NOTICE — the deterministic auditors rejected your output\n'
      cat "$RUN/fidelity_audit.out"
      printf '\nRestore every missing/mishandled input finding and rewrite ALL THREE files.\n'
    } > "$prompt"
    try=$(( try + 1 ))
  done
  t_end stage67_synthesis_fidelity; mark_done stage7
else note "skip stages 6+7 (marker)"; fi

# ================================================================ stage 8: deliver
t_start
note "stage 8: deliver (local only)"
python3 - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
findings = json.loads((run / "findings.json").read_text())
counts = {}
for f in findings:
    counts[f["severity"]] = counts.get(f["severity"], 0) + 1
print("severity counts:", json.dumps(counts))
print("deliverables:")
for name in ("report.md", "findings.json", "synthesis_ledger.json", "timings.tsv"):
    p = run / name
    print(f"  {p} ({p.stat().st_size if p.exists() else 'MISSING'} bytes)")
PY
t_end stage8_deliver
note "run complete: $RUN"
