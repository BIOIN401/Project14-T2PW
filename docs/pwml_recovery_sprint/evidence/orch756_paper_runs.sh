#!/usr/bin/env bash
# ORCH-756 / ORCH-759 (D-105 § 3): the cohort paper legs, one bounded leg at a time.
# Copy of orch755_paper_runs.sh with these changes:
#   - the tree the legs run in is a parameter (ORCH-756: its own worktree, whose .env sets 1 draw;
#     ORCH-759: the primary checkout, whose .env sets 3 draws). The tree's .env is never edited here;
#   - modes are a parameter (default strict only);
#   - pair timeout 5,400 s, wrapper 6,000 s (D-105 budgets make calls longer than ORCH-755's 3,600 s);
#   - G11 reports, the run log and the REVIEW folder always land in the PRIMARY checkout, so the
#     product owner finds them in one place; the leg directories stay in the tree that ran them;
#   - the review pack run is the TREE's copy (its ROOT is its own location: it reads that tree's
#     tmp/extraction_diagnostics), and it copies every *.pwml and research_pathway_* file;
#   - before each leg the runner also waits while <primary>/runs_papers/<TASK>/PAUSE exists, so the
#     orchestrator can take the heavy lock between legs for a gate without racing the next leg.
#
#   bash orch756_paper_runs.sh <tree-root> <TASK-ID> ["strict" | "strict research"] ["<conditions>"] [<pair-timeout-s>]
# Pair timeout default 5,400 s; the wrapper gets +600 s. ORCH-756's retry pass and ORCH-759 use 10,800 s
# (orchestrator operating ruling under D-105's intent, approved by main 2026-10-21; see ORCH-756-RESULT.md).
set -u
ROOT=${1:?tree root}
TASK=${2:?task id, e.g. ORCH-756}
MODES=${3:-strict}
CONDS=${4:-"enterobactin menaquinone chlorophyll ara_operon sour_taste wnt_pcp wnt_ca"}
PAIR_TIMEOUT=${5:-5400}
WRAP_TIMEOUT=$((PAIR_TIMEOUT + 600))
DEADLINE_H=$(awk "BEGIN{printf \"%.2f\", ($PAIR_TIMEOUT + 720) / 3600}")
PRIMARY="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
cd "$ROOT" || exit 2
PY="$PRIMARY/.venv/Scripts/python.exe"
EV="$ROOT/docs/pwml_recovery_sprint/evidence"
G11="$PRIMARY/docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py"
TOPICS="$PRIMARY/docs/evaluation_v2/paper_runs_orch755/topics"
LEGS="$ROOT/runs_papers/$TASK/legs"
OUT="$PRIMARY/runs_papers/$TASK"
REVIEW="$OUT/REVIEW"
NIGHT="$OUT/night.log"
mkdir -p "$OUT/logs" "$REVIEW" "$LEGS"
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$ROOT/src"
say() { echo "$(date '+%F %T') $*" | tee -a "$NIGHT"; }
lbl() { echo "$1" | tr 'A-Z_' 'a-z-'; }

wait_lock() {  # up to 3 h
  local n=0
  until { [ ! -f C:/t/heavylock/holder.json ] && [ ! -f "$OUT/PAUSE" ]; } || [ $n -ge 360 ]; do sleep 30; n=$((n+1)); done
  [ ! -f C:/t/heavylock/holder.json ] && [ ! -f "$OUT/PAUSE" ]
}

say "$TASK start; tree=$ROOT head=$(git rev-parse --short HEAD) modes=[$MODES] conds=[$CONDS] pair_timeout=$PAIR_TIMEOUT draws=$(grep -E '^T2PW_EXTRACTION_DRAWS=' .env | cut -d= -f2) budgets: $(grep -E '^(LLM_MAX_TOKENS|OPENROUTER_[A-Z]*_MAX_TOKENS)=' .env | tr '\n' ' ')"
for mode in $MODES; do
  for c in $CONDS; do
    rc=""
    L="leg-$(lbl "$c")-$mode"
    for attempt in 1 2; do
      if ! wait_lock; then rc="skipped-lock-busy-3h"; break; fi
      P=$("$PY" "$G11" next --task "$TASK" --label "$L" 2>&1 | tail -1)
      case "$P" in *rror*|"") rc="skipped-invalid-report-path"; break;; esac
      case "$P" in *"$L.json") : ;; *) rc="skipped-invalid-report-path"; break;; esac
      [ -d "$(dirname "$P")" ] || { rc="skipped-invalid-report-dir"; break; }
      N=$(basename "$P" .json)
      start=$(date +%s)
      say "START $c/$mode ($N, attempt $attempt)"
      "$PY" "$EV/bounded_run.py" --label "$L" --timeout "$WRAP_TIMEOUT" --heavy-lock "$TASK" --json "$P" -- \
        "$PY" -u scripts/batch_run.py --topics "$TOPICS/$c.txt" --modes "$mode" --out "$LEGS/$c-$mode" \
        --fresh --timeout "$PAIR_TIMEOUT" --deadline "$DEADLINE_H" > "$OUT/logs/$N.log" 2>&1
      rc=$?
      end=$(date +%s)
      say "END   $c/$mode exit=$rc $(( (end-start)/60 )) min | $(grep 'FINAL SURVIVING' "$OUT/logs/$N.log" | tail -1)"
      [ "$rc" = "95" ] && { say "heavy lock lost the race; retrying"; sleep 60; continue; }  # acquire failed, child never ran
      break
    done
    case "$rc" in skipped-*) start=$(date +%s); end=$start; say "SKIP  $c/$mode: $rc";; esac
    "$PY" "$EV/orch756_review_pack.py" --task "$TASK" --leg-out "$LEGS/$c-$mode" --cond "$c" --mode "$mode" \
      --started "$start" --finished "$end" --exit "$rc" --review "$REVIEW" >> "$NIGHT" 2>&1
  done
done
say "$TASK done. Review: $REVIEW/INDEX.txt"
