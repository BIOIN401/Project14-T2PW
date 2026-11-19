#!/usr/bin/env bash
# W9-local: start the T2PW web server (uvicorn on 0.0.0.0:$PORT) and the job worker.
# Run it from Git bash in the foreground (or a background Bash tool call). It stays attached
# until both wrappers exit: stop.sh, the 24 h bounded_run timeout, or a crash.
# Each wrapper is bounded_run.py WITHOUT the heavy lock; each pipeline job takes the heavy
# lock itself (t2pw_deploy.run_pathway).
set -u
R=C:/t/t2pw-server
D=${T2PW_DATA:-C:/t/t2pw-server-data}
PY=C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe
BR=$R/docs/pwml_recovery_sprint/evidence/bounded_run.py
PORT=${PORT:-8800}

if [ -s "$D/pids" ]; then
  echo "already started (pids: $(tr '\n' ' ' < "$D/pids")); run stop.sh first" >&2
  exit 1
fi
mkdir -p "$D/logs" "$D/work"
[ -s "$D/rate_key_secret" ] || "$PY" -c "import secrets; print(secrets.token_hex(32))" > "$D/rate_key_secret"

export PYTHONPATH="$R/src;$R/deploy/local" PYTHONUTF8=1
export T2PW_DB="$D/jobs.sqlite" T2PW_WORK_ROOT="$D/work"
export T2PW_BASE_URL="${T2PW_BASE_URL:-http://192.168.1.232:$PORT}"
export T2PW_RATE_KEY_SECRET="$(cat "$D/rate_key_secret")"
export T2PW_RATE_LIMIT=${T2PW_RATE_LIMIT:-2} T2PW_MAX_QUEUED=${T2PW_MAX_QUEUED:-5}
# PO ruling 2026-11-19: the server runs 3 Stage-1 extraction draws (code default is 1).
export T2PW_EXTRACTION_DRAWS=${T2PW_EXTRACTION_DRAWS:-3}
cd "$R"
stamp=$(date +%Y%m%d-%H%M%S)

"$PY" "$BR" --timeout 86400 --label t2pw-web-uvicorn --json "$D/logs/uvicorn-$stamp.bounded.json" -- \
  "$PY" -m uvicorn --factory t2pw_deploy:create_app --host 0.0.0.0 --port "$PORT" \
  > "$D/logs/uvicorn-$stamp.log" 2>&1 &
cat /proc/$!/winpid > "$D/pids"

"$PY" "$BR" --timeout 86400 --label t2pw-web-worker --json "$D/logs/worker-$stamp.bounded.json" -- \
  "$PY" -m t2pw.web.worker --db "$T2PW_DB" --work-root "$T2PW_WORK_ROOT" \
  --run-fn t2pw_deploy:run_pathway --poll 5 \
  > "$D/logs/worker-$stamp.log" 2>&1 &
cat /proc/$!/winpid >> "$D/pids"

echo "started: http://localhost:$PORT  $T2PW_BASE_URL  pids $(tr '\n' ' ' < "$D/pids") logs $D/logs/*-$stamp.*"
wait
echo "stopped at $(date)"
rm -f "$D/pids"
