"""Gate before the docs commit: secret shapes and configured secret values in the evidence tree. Names and counts only."""
import re
import sys
from pathlib import Path

from dotenv import dotenv_values

root = Path(sys.argv[1])
values = {k: v for k, v in dotenv_values(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.env").items()
          if v and len(v) >= 4 and any(s in k for s in ("KEY", "PASSWORD", "USER", "HOST", "EMAIL"))}
shape = re.compile(r"sk-or-[A-Za-z0-9-]{8,}|(?i:\bbearer\s+[A-Za-z0-9._-]{12,})")
email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
hits = 0
for p in sorted(x for x in root.rglob("*") if x.is_file()):
    text = p.read_text(encoding="utf-8", errors="replace")
    found = [k for k, v in values.items() if v in text]
    n_shape, n_mail = len(shape.findall(text)), len(email.findall(text))
    if found or n_shape or n_mail:
        hits += 1
        print(p.name, "values:", found, "shapes:", n_shape, "emails:", n_mail)
print("files with hits:", hits)
