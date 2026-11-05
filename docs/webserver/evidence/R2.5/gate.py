"""Scratch gate for R2.5 record segments. Every check is appended to C:/t/wsR25tmp/gate.log.

    gate.py wait <seconds>   poll every 10 s until clear (exit 0) or the budget runs out (exit 1)
    gate.py check            one check: exit 0 if clear, 1 if not

Clear = no C:/t/heavylock/holder.json AND no python process whose command line runs pytest or
bounded_run and names a worktree other than ws-R25 (e.g. C:/t/ws-int, C:/t/ws-R2-tip).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("C:/t/wsR25tmp/gate.log")
MINE = {"c:/t/ws-r25", "c:/t/wsr25tmp"}


def check() -> bool:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
        capture_output=True, text=True).stdout
    others = []
    for line in out.splitlines():
        low = line.lower().replace("\\", "/")
        if "pytest" in low or "bounded_run" in low:
            trees = sorted(set(re.findall(r"c:/t/[\w.-]+", low)))
            # Exact worktree names: C:/t/ws-R25-trial is NOT this worktree (fix: a substring test missed it).
            if not trees or not set(trees) <= MINE:
                others.append(f"{line.split(chr(9))[0]} {' '.join(trees) or '(no worktree path)'}")
    holder = Path("C:/t/heavylock/holder.json")
    lock = None
    if holder.exists():
        try:
            lock = json.loads(holder.read_text(encoding="utf-8")).get("holder")
        except Exception:  # noqa: BLE001
            lock = "(unreadable)"
    clear = not others and lock is None
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} clear={clear} lock={lock} other_jobs={others}\n")
    return clear


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(0 if check() else 1)
    deadline = time.time() + float(sys.argv[2])
    while time.time() < deadline:
        if check():
            print("CLEAR", time.strftime("%H:%M:%S"))
            sys.exit(0)
        time.sleep(10)
    print("NOT CLEAR", time.strftime("%H:%M:%S"))
    sys.exit(1)
