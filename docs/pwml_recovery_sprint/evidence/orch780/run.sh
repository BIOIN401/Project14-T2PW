#!/usr/bin/env bash
# ORCH-780 bounded job runner. usage: run.sh <label> <timeout_s> <cmd...>
# Checks available RAM (lead ruling: >= 3072 MB) and the heavy lock, takes a fresh G11 path, runs under
# bounded_run with the heavy lock, logs to evidence/orch780/<NN-label>.log (first line: the RAM figure).
set -u
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
T=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch780
EV=$T/docs/pwml_recovery_sprint/evidence
label=$1; timeout=$2; shift 2
free=$(powershell -NoProfile -Command "[int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue")
if [ "$free" -lt 3072 ]; then echo "RAM below floor: $free MB"; exit 96; fi
if [ -e C:/t/heavylock/holder.json ]; then echo "heavy lock held"; exit 95; fi
P=$($PY $EV/g11/g11_evidence.py next --task ORCH-780 --label "$label" | tail -1)
case "$P" in *"$label.json") ;; *) echo "G11 next failed: $P"; exit 97;; esac
mkdir -p $EV/orch780
log=$EV/orch780/$(basename "$P" .json).log
echo "available RAM MB at job start: $free (lead ruling 2026-10-26: >= 3072, was 6144)" | tee "$log"
echo "G11: $P" | tee -a "$log"
env -u PATHBANK_DB_HOST -u PATHBANK_DB_USER -u PATHBANK_DB_PASSWORD -u PATHBANK_DB_NAME -u PATHBANK_DB_PORT \
  PYTHONPATH="${PYTHONPATH:-}" \
  $PY $EV/bounded_run.py --label "$label" --timeout "$timeout" --cwd C:/t/ORCH-780/cwd --heavy-lock ORCH-780 --json "$P" -- "$@" 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
echo "exit: $rc" | tee -a "$log"
exit $rc
