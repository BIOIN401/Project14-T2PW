"""Gate for R26 record segments and test runs. Every check is appended to C:/t/wsR26tmp/gate.log.

    gate.py wait <seconds>   poll every 15 s until clear (exit 0) or the budget runs out (exit 1)
    gate.py check            one check: exit 0 if clear, 1 if not

Clear = no C:/t/heavylock/holder.json, at least 6 GB of free RAM, AND no python process whose command line
runs pytest or bounded_run and names a tree other than ws-R26 / wsR26tmp.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("C:/t/wsR26tmp/gate.log")
MINE = {"c:/t/ws-r26", "c:/t/wsr26tmp"}
MIN_FREE_GB = 6.0


def powershell(cmd: str) -> str:
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout


def check() -> bool:
    out = powershell("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    others = []
    for line in out.splitlines():
        low = line.lower().replace("\\", "/")
        if "pytest" in low or "bounded_run" in low:
            trees = sorted(set(re.findall(r"c:/t/[\w.-]+", low)))
            if not trees or not set(trees) <= MINE:
                others.append(f"{line.split(chr(9))[0]} {' '.join(trees) or '(no worktree path)'}")
    free_kb = powershell("(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory").strip()
    free_gb = round(int(free_kb) / 1048576, 1) if free_kb.isdigit() else 0.0
    holder = Path("C:/t/heavylock/holder.json")
    lock = None
    if holder.exists():
        try:
            lock = json.loads(holder.read_text(encoding="utf-8")).get("holder")
        except Exception:  # noqa: BLE001
            lock = "(unreadable)"
    clear = not others and lock is None and free_gb >= MIN_FREE_GB
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} clear={clear} lock={lock} free_gb={free_gb} other_jobs={others}\n")
    return clear


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(0 if check() else 1)
    deadline = time.time() + float(sys.argv[2])
    while time.time() < deadline:
        if check():
            print("CLEAR", time.strftime("%H:%M:%S"))
            sys.exit(0)
        time.sleep(15)
    print("NOT CLEAR", time.strftime("%H:%M:%S"))
    sys.exit(1)
