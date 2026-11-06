"""R3b heavy-run gate: wait until no pytest/bounded_run python names ANOTHER worktree.

Paths are compared exactly: every ``c:/t/<dir>`` named in a command line, plus the
primary checkout (its ``.venv`` interpreter path does not count), must be one of ours.
Pylance's ``get_pytest_options.py`` is ignored. Polls 3 times, 55 s apart (under the harness 180 s auto-background), so it
fits one foreground Bash call; exit 0 = free, 1 = still busy (call again).
Every observation is appended to C:/t/wsR3btmp/waits.log.
"""
import json
import re
import subprocess
import sys
import time

OURS = {"c:/t/ws-r3b", "c:/t/ws-r3b-base", "c:/t/wsr3btmp"}
PRIMARY = "c:/users/angad/desktop/summerbioin/project14-t2pw"
LOG = "C:/t/wsR3btmp/waits.log"
PS = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
      "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress")


def trees(cmd):
    found = set(re.findall(r"c:/t/[^/\s\"']+", cmd))
    if PRIMARY in cmd.replace(PRIMARY + "/.venv/", ""):
        found.add(PRIMARY)
    return found


def busy():
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    rows = json.loads(out) if out else []
    rows = rows if isinstance(rows, list) else [rows]
    hits = []
    for r in rows:
        cmd = (r.get("CommandLine") or "").replace("\\", "/").lower()
        if "get_pytest_options.py" in cmd or not ("pytest" in cmd or "bounded_run" in cmd):
            continue
        others = sorted(trees(cmd) - OURS)
        if others:
            hits.append((r["ProcessId"], others, cmd[:160]))
    return hits


label = sys.argv[1] if len(sys.argv) > 1 else "?"
for attempt in range(3):
    hits = busy()
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "before": label,
                             "attempt": attempt, "busy": hits}) + "\n")
    if not hits:
        print(f"free ({label}, attempt {attempt})")
        sys.exit(0)
    print(f"busy ({label}): {[(h[0], h[1]) for h in hits]}")
    if attempt < 2:
        time.sleep(55)
sys.exit(1)
