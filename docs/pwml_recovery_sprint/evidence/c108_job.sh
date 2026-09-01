# C-108 job launcher. Source this, then: c108run <label> <timeout-seconds> <logfile> -- <cmd...>
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
BR=C:/t/c108/docs/pwml_recovery_sprint/evidence/bounded_run.py
G11=C:/t/c108/docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py

c108run() {
  local label="$1"; shift
  local tmo="$1"; shift
  local log="$1"; shift
  [ "$1" = "--" ] && shift
  local P
  P=$("$PY" "$G11" next --task C-108 --label "$label" 2>/dev/null | tail -1)
  if [ -z "$P" ]; then echo "G11 EMPTY for label=$label"; return 9; fi
  case "$P" in *C-108*"$label".json) : ;; *) echo "G11 INVALID: $P"; return 9;; esac
  echo "G11: $P"
  T2PW_OFFLINE_CURATOR=1 PYTHONIOENCODING=utf-8 \
    "$PY" "$BR" --timeout "$tmo" --label "$label" --heavy-lock C-108 --json "$P" \
    -- "$@" > "$log" 2>&1
  local rc=$?
  echo "wrapper rc=$rc  log=$log"
  grep -E "FINAL SURVIVING COUNT|^cleanup |exit code \(real\)|exit reason" "$log"
  return $rc
}
