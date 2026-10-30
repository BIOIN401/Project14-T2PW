#!/usr/bin/env bash
# ORCH-798 (D-120 § 6): five NEW papers, strict, 3 draws, 10,800 s per leg, from a pinned worktree.
# Copy of orch770_night.sh with the five ORCH-798 conditions and no gold scoring (none of them has gold).
# Each leg runs through orch756_paper_runs.sh (heavy lock per leg, PAUSE file honoured between legs, review pack into
# the PRIMARY runs_papers/<TASK>/REVIEW/, G11 under evidence/g11/<TASK>/). The runner writes MORNING_SUMMARY.md itself.
#   bash orch798_night.sh <pinned worktree> [task id, default ORCH-798]
set -u
WT=${1:?pinned worktree}
TASK=${2:-ORCH-798}
P="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
PY="$P/.venv/Scripts/python.exe"
EV="$P/docs/pwml_recovery_sprint/evidence"
OUT="$P/runs_papers/$TASK"
mkdir -p "$OUT/logs"
note() { echo "$(date '+%F %T') $*" >> "$OUT/night.log"; }

note "$TASK NIGHT START worktree=$WT head=$(git -C "$WT" rev-parse --short HEAD)"
for c in serine ergosterol staphyloxanthin glucosinolate cgas_sting; do
  bash "$WT/docs/pwml_recovery_sprint/evidence/orch756_paper_runs.sh" "$WT" "$TASK" strict "$c" 10800
done
"$PY" "$EV/orch770_morning_summary.py" --task "$TASK" --legs "$WT/runs_papers/$TASK/legs" --out-dir "$OUT" \
  --src "$WT/src" --gold "$OUT/no_gold_for_new_papers.txt" > "$OUT/logs/morning_summary.log" 2>&1
note "$TASK NIGHT DONE summary_exit=$? -> $OUT/MORNING_SUMMARY.md"
