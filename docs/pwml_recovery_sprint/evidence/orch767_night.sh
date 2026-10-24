#!/usr/bin/env bash
# ORCH-767 (D-111): the overnight run. Six strict legs at 3 draws, 10,800 s each, from a worktree pinned at main
# (.claude/worktrees/orch767, .env copied from the primary: draws 3, budgets as set), so merges into the primary cannot
# change code under a running leg. Each leg goes through orch756_paper_runs.sh (heavy lock per leg, review pack into the
# PRIMARY runs_papers/ORCH-767/REVIEW/, G11 under evidence/g11/ORCH-767/). After enterobactin and after menaquinone
# the gold scorer runs automatically (orch756_gold_accuracy.py, bounded). If runs_papers/ORCH-767/RUN_767B holds a
# worktree path when the six legs are done, one more menaquinone leg runs there as ORCH-767b ("with 766/765").
# At the end orch767_morning_summary.py drafts runs_papers/ORCH-767/MORNING_SUMMARY.md.
set -u
P="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
WT="$P/.claude/worktrees/orch767"
PY="$P/.venv/Scripts/python.exe"
EV="$P/docs/pwml_recovery_sprint/evidence"
OUT="$P/runs_papers/ORCH-767"
mkdir -p "$OUT/logs"
note() { echo "$(date '+%F %T') $*" >> "$OUT/night.log"; }

score() {  # task legs review papers out label
  local R N
  R=$("$PY" "$EV/g11/g11_evidence.py" next --task "$1" --label "$6" 2>&1 | tail -1)
  case "$R" in *"$6.json") : ;; *) note "SCORE $4: INVALID REPORT PATH ($R)"; return;; esac
  N=$(basename "$R" .json)
  ( cd "$P" && PYTHONPATH="$P/src" PYTHONIOENCODING=utf-8 "$PY" "$EV/bounded_run.py" --label "$6" --timeout 500 --json "$R" -- \
      "$PY" -u "$EV/orch756_gold_accuracy.py" --task "$1" --legs "$2" --review "$3" --papers "$4" --out "$5" \
      > "$OUT/logs/$N.log" 2>&1 )
  note "SCORE $1 $4 -> $5 exit=$? ($N)"
}

note "ORCH-767 NIGHT START worktree head=$(git -C "$WT" rev-parse --short HEAD)"
for c in enterobactin chlorophyll sour_taste_v2 wnt_pcp wnt_ca menaquinone; do
  bash "$WT/docs/pwml_recovery_sprint/evidence/orch756_paper_runs.sh" "$WT" ORCH-767 strict "$c" 10800
  case "$c" in
    enterobactin) score ORCH-767 "$WT/runs_papers/ORCH-767/legs" "$OUT/REVIEW" enterobactin "$EV/orch767_gold_accuracy.txt" score-enterobactin ;;
    menaquinone)  score ORCH-767 "$WT/runs_papers/ORCH-767/legs" "$OUT/REVIEW" enterobactin,menaquinone "$EV/orch767_gold_accuracy.txt" score-enterobactin-menaquinone ;;
  esac
done
if [ -f "$OUT/RUN_767B" ]; then
  WB=$(head -1 "$OUT/RUN_767B" | tr -d '\r')
  note "ORCH-767b: menaquinone from $WB head=$(git -C "$WB" rev-parse --short HEAD)"
  bash "$WB/docs/pwml_recovery_sprint/evidence/orch756_paper_runs.sh" "$WB" ORCH-767b strict menaquinone 10800
  score ORCH-767b "$WB/runs_papers/ORCH-767b/legs" "$P/runs_papers/ORCH-767b/REVIEW" menaquinone "$EV/orch767b_gold_accuracy.txt" score-menaquinone
fi
"$PY" "$EV/orch767_morning_summary.py" > "$OUT/logs/morning_summary_gen.log" 2>&1
note "ORCH-767 NIGHT DONE summary_exit=$?"
