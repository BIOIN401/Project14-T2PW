#!/usr/bin/env bash
# ORCH-755: overnight paper runs in the D-104 production configuration (.env: 3 draws, 64,000
# extraction tokens; prompt v03; live curator). 7 conditions x {strict, research} = 14 legs.
# All strict legs run first, so every strict PWML exists by morning even if research runs long.
#
# One leg per bounded_run (G11): the heavy lock is held per leg, not all night, so the parallel
# ORCH-760 session can take it between legs. After every leg, orch755_review_pack.py renders the
# leg into runs_papers/ORCH-755/REVIEW/, so a night cut short still leaves what finished.
set -u
cd "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW" || exit 2
PY="$PWD/.venv/Scripts/python.exe"
EV=docs/pwml_recovery_sprint/evidence
TOPICS=docs/evaluation_v2/paper_runs_orch755/topics
BASE=runs_papers/ORCH-755
REVIEW=$BASE/REVIEW
NIGHT=$BASE/night.log
mkdir -p "$BASE/logs" "$REVIEW"
export PYTHONIOENCODING=utf-8
CONDS="enterobactin menaquinone chlorophyll ara_operon sour_taste wnt_pcp wnt_ca"
say() { echo "$(date '+%F %T') $*" | tee -a "$NIGHT"; }

wait_lock() {  # up to 3 h; the other session's jobs are short
  local n=0
  until [ ! -f C:/t/heavylock/holder.json ] || [ $n -ge 360 ]; do sleep 30; n=$((n+1)); done
  [ ! -f C:/t/heavylock/holder.json ]
}

say "ORCH-755 night start; draws=$(grep -E '^T2PW_EXTRACTION_DRAWS=' .env | cut -d= -f2) extraction_tokens=$(grep -E '^OPENROUTER_EXTRACTION_MAX_TOKENS=' .env | cut -d= -f2) head=$(git rev-parse --short HEAD)"
for mode in strict research; do
  for c in $CONDS; do
    rc=""
    for attempt in 1 2; do
      if ! wait_lock; then rc="skipped-lock-busy-3h"; break; fi
      P=$($PY $EV/g11/g11_evidence.py next --task ORCH-755 --label "leg-$c-$mode" 2>&1 | tail -1)
      case "$P" in *rror*) rc="skipped-invalid-report-path"; break;; esac
      [ -d "$(dirname "$P")" ] || { rc="skipped-invalid-report-dir"; break; }
      N=$(basename "$P" .json)
      start=$(date +%s)
      say "START $c/$mode ($N, attempt $attempt)"
      $PY $EV/bounded_run.py --label "leg-$c-$mode" --timeout 4200 --heavy-lock ORCH-755 --json "$P" -- \
        $PY -u scripts/batch_run.py --topics "$TOPICS/$c.txt" --modes "$mode" --out "$BASE/legs/$c-$mode" \
        --fresh --timeout 3600 --deadline 1.2 > "$BASE/logs/$N.log" 2>&1
      rc=$?
      end=$(date +%s)
      say "END   $c/$mode exit=$rc $(( (end-start)/60 )) min | $(grep 'FINAL SURVIVING' "$BASE/logs/$N.log" | tail -1)"
      [ "$rc" = "95" ] && { say "heavy lock lost the race; retrying"; continue; }  # bounded_run: acquire failed, child never ran
      break
    done
    case "$rc" in skipped-*) start=$(date +%s); end=$start; say "SKIP  $c/$mode: $rc";; esac
    $PY $EV/orch755_review_pack.py --leg-out "$BASE/legs/$c-$mode" --cond "$c" --mode "$mode" \
      --started "$start" --finished "$end" --exit "$rc" --review "$REVIEW" >> "$NIGHT" 2>&1
  done
done
say "ORCH-755 night done. Review: $REVIEW/INDEX.txt"
