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
  # F-160. Purge bytecode caches before every job. A same-length edit landing in
  # the same whole second as the write before it leaves a timestamp-keyed .pyc
  # valid, so the OLD module runs and the job reports a stale result. Measured on
  # this tree: a plain import DOES serve the stale bytecode
  # (c108_r2_f160_demo.log ARM 0), and every probe here imports the guard module
  # directly. Milliseconds against a false green.
  #
  # SCOPED TO src/t2pw AND tests, AND THAT SCOPE IS LOAD-BEARING. THIS REPO
  # TRACKS __pycache__ AT FOUR PATHS -- __pycache__/, scripts/__pycache__/,
  # src/__pycache__/ and src/tools/__pycache__/ -- and an unscoped
  # "find $PWD -name __pycache__ -exec rm -rf" DELETES 56 TRACKED FILES. It did,
  # once, before this comment existed. Neither directory below is tracked:
  #   git ls-files | grep -E "^(src/t2pw|tests)/.*__pycache__"  ->  0
  for d in "$PWD/src/t2pw" "$PWD/tests"; do
    [ -d "$d" ] && find "$d" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  done
  T2PW_OFFLINE_CURATOR=1 PYTHONIOENCODING=utf-8 \
    "$PY" "$BR" --timeout "$tmo" --label "$label" --heavy-lock C-108 --json "$P" \
    -- "$@" > "$log" 2>&1
  local rc=$?
  echo "wrapper rc=$rc  log=$log"
  grep -E "FINAL SURVIVING COUNT|^cleanup |exit code \(real\)|exit reason" "$log"
  return $rc
}
