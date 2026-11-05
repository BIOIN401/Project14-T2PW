"""Wait until no pytest/bounded_run python from another worktree is alive (R2 brief addendum).
Polls every 60 s, up to 20 min. Appends every observation to C:/t/wsR2tmp/waits.log.
Exit 0 = free, 1 = still busy after 20 min."""
import json
import subprocess
import sys
import time

OTHERS = ("c:/t/ws-r25", "c:/t/ws-int", "-tr")
LOG = "C:/t/wsR2tmp/waits.log"
PS = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
      "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress")


def busy():
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    rows = json.loads(out) if out else []
    rows = rows if isinstance(rows, list) else [rows]
    hits = []
    for r in rows:
        cmd = (r.get("CommandLine") or "").replace("\\", "/").lower()
        if ("pytest" in cmd or "bounded_run" in cmd) and "c:/t/ws-r2/" not in cmd \
                and "c:/t/ws-r2-base" not in cmd and any(o in cmd for o in OTHERS):
            hits.append((r["ProcessId"], cmd[:200]))
    return hits


label = sys.argv[1] if len(sys.argv) > 1 else "?"
for attempt in range(9):
    hits = busy()
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "before": label,
                             "attempt": attempt, "busy": hits}) + "\n")
    if not hits:
        print(f"free ({label}, attempt {attempt})")
        sys.exit(0)
    print(f"busy ({label}): {hits}")
    if attempt < 8:
        time.sleep(60)
sys.exit(1)
