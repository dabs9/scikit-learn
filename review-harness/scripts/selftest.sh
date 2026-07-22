#!/usr/bin/env bash
# Deterministic-machinery selftest: no agents, no network. Exercises contract_lint,
# snapshot_check, and fidelity_audit on fixtures — including the "hand-delete a finding
# from synthesis output and confirm the auditors force it back" invariant.
set -uo pipefail
RH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()   { pass=$((pass+1)); echo "  ok: $1"; }
bad()  { fail=$((fail+1)); echo "  FAIL: $1"; }
expect() { # expect <desc> <want_exit> cmd...
  local desc=$1 want=$2; shift 2
  "$@" >/dev/null 2>&1
  local got=$?
  [[ $got -eq $want ]] && ok "$desc" || bad "$desc (exit $got, wanted $want)"
}

echo "== contract_lint =="
cat > "$TMP/good.md" <<'EOF'
### F1 — off-by-one in tree condense loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:142 — loop bound uses n_samples where n_samples-1 is required
scenario: "cluster tree with a single leaf → IndexError during condense"
contract: change the loop bound to n_samples - 1
instances: [sklearn/cluster/_hdbscan/_tree.pyx:142, sklearn/cluster/_hdbscan/_tree.pyx:198]

### F2 — docstring promises default that code does not set
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:310 — docstring says default=5, signature has 10
scenario: "user relies on documented default → different clustering than documented"
contract: update the docstring default to 10
instances: single-instance
EOF
expect "clean file passes" 0 python3 "$RH/scripts/contract_lint.py" "$TMP/good.md"

cat > "$TMP/nofind.md" <<'EOF'
NO FINDINGS
EOF
expect "NO FINDINGS sentinel passes" 0 python3 "$RH/scripts/contract_lint.py" "$TMP/nofind.md"

cat > "$TMP/bad.md" <<'EOF'
### F1 — vague worry
severity: critical
evidence: something seems off in the tree module
contract: fix it, or alternatively document it, whichever is easier
EOF
expect "bad file rejected" 1 python3 "$RH/scripts/contract_lint.py" "$TMP/bad.md"
v=$(python3 "$RH/scripts/contract_lint.py" "$TMP/bad.md")
for want in "severity" "anchor" "scenario" "instances" "hedged"; do
  grep -qi "$want" <<<"$v" && ok "violation names $want" || bad "violation missing $want in: $v"
done

echo "== snapshot_check =="
cp "$TMP/good.md" "$TMP/old.md"
{ cat "$TMP/old.md"; printf '\n### F3 — appended sibling defect\nseverity: low\nevidence: sklearn/cluster/_hdbscan/hdbscan.py:12 — x\nscenario: "a → b"\ncontract: do the one thing\ninstances: single-instance\n'; } > "$TMP/new_ok.md"
expect "add-only append passes" 0 python3 "$RH/scripts/snapshot_check.py" "$TMP/old.md" "$TMP/new_ok.md"
sed 's/off-by-one/off_by_one/' "$TMP/new_ok.md" > "$TMP/new_bad.md"
expect "mutated original rejected" 1 python3 "$RH/scripts/snapshot_check.py" "$TMP/old.md" "$TMP/new_bad.md"

echo "== fidelity_audit =="
RUNFIX="$TMP/runfix"; mkdir -p "$RUNFIX"
cp "$TMP/good.md" "$RUNFIX/depth.md"
printf 'NO FINDINGS\n' > "$RUNFIX/closure.md"
cat > "$RUNFIX/inverse.md" <<'EOF'
### F1 — plan requirement without implementing change
severity: medium
evidence: review-harness/runs/26385/PLAN.md:20 — plan promises X; no manifest file implements it
scenario: "release ships → documented capability X absent"
contract: implement X or amend the plan of record
instances: single-instance
EOF
python3 - "$RUNFIX" <<'PY'
import json, sys
run = sys.argv[1]
findings = [
 {"id":"U1","severity":"medium","title":"off-by-one in tree condense loop","file":"sklearn/cluster/_hdbscan/_tree.pyx","line":142,
  "scenario":"cluster tree with a single leaf → IndexError during condense","contract":"change the loop bound to n_samples - 1",
  "instances":["sklearn/cluster/_hdbscan/_tree.pyx:142"],"body":"### F1 ... sklearn/cluster/_hdbscan/_tree.pyx:142 verbatim"},
 {"id":"U2","severity":"low","title":"docstring default mismatch","file":"sklearn/cluster/_hdbscan/hdbscan.py","line":310,
  "scenario":"user relies on documented default → different clustering","contract":"update the docstring default to 10",
  "instances":["sklearn/cluster/_hdbscan/hdbscan.py:310"],"body":"### F2 ... sklearn/cluster/_hdbscan/hdbscan.py:310 verbatim"},
 {"id":"U3","severity":"medium","title":"plan requirement without implementing change","file":"review-harness/runs/26385/PLAN.md","line":20,
  "scenario":"release ships → documented capability X absent","contract":"implement X or amend the plan of record",
  "instances":["review-harness/runs/26385/PLAN.md:20"],"body":"### F1 ... review-harness/runs/26385/PLAN.md:20 verbatim"},
]
ledger = {"D:F1":{"disposition":"kept-as","unit":"U1"},
          "D:F2":{"disposition":"kept-as","unit":"U2"},
          "I:F1":{"disposition":"kept-as","unit":"U3"}}
open(f"{run}/findings.json","w").write(json.dumps(findings))
open(f"{run}/synthesis_ledger.json","w").write(json.dumps(ledger))
open(f"{run}/report.md","w").write("# report fixture\n")
PY
expect "complete synthesis passes" 0 python3 "$RH/scripts/fidelity_audit.py" "$RUNFIX"

# the load-bearing invariant: hand-delete a delivered finding -> auditor must force it back
python3 - "$RUNFIX" <<'PY'
import json, sys
run = sys.argv[1]
findings = json.loads(open(f"{run}/findings.json").read())
ledger = json.loads(open(f"{run}/synthesis_ledger.json").read())
findings = [f for f in findings if f["id"] != "U3"]   # synthesizer "loses" a finding
del ledger["I:F1"]
open(f"{run}/findings.json","w").write(json.dumps(findings))
open(f"{run}/synthesis_ledger.json","w").write(json.dumps(ledger))
PY
expect "hand-deleted finding rejected" 1 python3 "$RH/scripts/fidelity_audit.py" "$RUNFIX"
audit_out=$(python3 "$RH/scripts/fidelity_audit.py" "$RUNFIX" 2>&1 || true)
grep -q "missing_ids: I:F1" <<<"$audit_out" \
  && ok "auditor names the missing id for re-prompt" || bad "missing_ids not reported"

# kept-ratio floor: everything merged into one unit
python3 - "$RUNFIX" <<'PY'
import json, sys
run = sys.argv[1]
findings = [{"id":"U1","severity":"medium","title":"t","file":"f.py","line":1,"scenario":"a → b","contract":"c",
 "instances":["f.py:1"],
 "body":"sklearn/cluster/_hdbscan/_tree.pyx:142 sklearn/cluster/_hdbscan/hdbscan.py:310 review-harness/runs/26385/PLAN.md:20"}]
ledger = {"D:F1":{"disposition":"merged-into","unit":"U1"},
          "D:F2":{"disposition":"merged-into","unit":"U1"},
          "I:F1":{"disposition":"merged-into","unit":"U1"}}
open(f"{run}/findings.json","w").write(json.dumps(findings))
open(f"{run}/synthesis_ledger.json","w").write(json.dumps(ledger))
PY
expect "kept-ratio floor rejected (1 unit / 3 inputs)" 1 python3 "$RH/scripts/fidelity_audit.py" "$RUNFIX"

echo
echo "selftest: $pass passed, $fail failed"
exit $(( fail > 0 ))
