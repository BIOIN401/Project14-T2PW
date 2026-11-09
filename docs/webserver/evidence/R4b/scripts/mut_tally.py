"""Last result per mutation in summary.txt (survivors were re-run after the test tightening)."""
import re
import sys
from pathlib import Path

rows = {}
for line in Path(sys.argv[1], "summary.txt").read_text(encoding="utf-8").splitlines():
    if line.startswith("killed "):
        continue
    name = line.split()[0]
    rows[name] = line
killed = [n for n, l in rows.items() if re.search(r"\d+ failed|ERROR|error", l)]
survived = sorted(set(rows) - set(killed))
print(f"killed {len(killed)}/{len(rows)}; survived: {survived}")
