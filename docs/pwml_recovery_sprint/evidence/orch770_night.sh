#!/usr/bin/env bash
# ORCH-770 (D-112 § 4): one live overnight confirmation run on the merged main, from a pinned worktree.
# Same six papers, strict, 3 draws, 10,800 s per leg, as ORCH-767; each leg through orch756_paper_runs.sh (heavy lock per leg,
# review pack into the PRIMARY runs_papers/ORCH-770/REVIEW/, G11 under evidence/g11/ORCH-770/). Gold scoring after
# enterobactin and menaquinone. The runner writes MORNING_SUMMARY.md ITSELF as its last step (orch770_morning_summary.py,
# which exists before launch: the ORCH-767 lesson).
#   bash orch770_night.sh <pinned worktree> [task id, default ORCH-770]
# ORCH-772: the task id names the run directory, the G11 reports and the gold file (ORCH-774 -> runs_papers/ORCH-774,
# evidence/orch774_gold_accuracy.txt); without the second argument every path is exactly what it was.
set -u
WT=${1:?pinned worktree}
TASK=${2:-ORCH-770}
P="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
PY="$P/.venv/Scripts/python.exe"
EV="$P/docs/pwml_recovery_sprint/evidence"
OUT="$P/runs_papers/$TASK"
GOLD="$EV/$(echo "$TASK" | tr 'A-Z' 'a-z' | tr -d '-')_gold_accuracy.txt"
mkdir -p "$OUT/logs"
note() { echo "$(date '+%F %T') $*" >> "$OUT/night.log"; }

score() {  # papers label
  local R N
  R=$("$PY" "$EV/g11/g11_evidence.py" next --task "$TASK" --label "$2" 2>&1 | tail -1)
  case "$R" in *"$2.json") : ;; *) note "SCORE $1: INVALID REPORT PATH ($R)"; return;; esac
  N=$(basename "$R" .json)
  ( cd "$P" && PYTHONPATH="$P/src" PYTHONIOENCODING=utf-8 "$PY" "$EV/bounded_run.py" --label "$2" --timeout 500 --json "$R" -- \
      "$PY" -u "$EV/orch756_gold_accuracy.py" --task "$TASK" --legs "$WT/runs_papers/$TASK/legs" --review "$OUT/REVIEW" \
      --papers "$1" --out "$GOLD" > "$OUT/logs/$N.log" 2>&1 )
  note "SCORE $1 exit=$? ($N)"
}

note "$TASK NIGHT START worktree=$WT head=$(git -C "$WT" rev-parse --short HEAD)"
for c in enterobactin chlorophyll sour_taste_v2 wnt_pcp wnt_ca menaquinone; do
  bash "$WT/docs/pwml_recovery_sprint/evidence/orch756_paper_runs.sh" "$WT" "$TASK" strict "$c" 10800
  case "$c" in
    enterobactin) score enterobactin score-enterobactin ;;
    menaquinone)  score enterobactin,menaquinone score-enterobactin-menaquinone ;;
  esac
done
"$PY" "$EV/orch770_morning_summary.py" --task "$TASK" --legs "$WT/runs_papers/$TASK/legs" --out-dir "$OUT" \
  --src "$WT/src" --gold "$GOLD" --compare "ORCH-767=$EV/orch767_gold_accuracy.txt" \
  --compare "ORCH-759p=$EV/orch759p_gold_accuracy.txt" --compare "ORCH-756=$EV/orch756_gold_accuracy.txt" \
  > "$OUT/logs/morning_summary.log" 2>&1
note "$TASK NIGHT DONE summary_exit=$? -> $OUT/MORNING_SUMMARY.md"
