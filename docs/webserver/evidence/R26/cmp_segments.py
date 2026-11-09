"""Where does segment B stop following segment A? Compare per-kind key order and report the first difference."""
import json
import sys
from pathlib import Path


def rows(path):
    out = {"llm": [], "http": [], "db": []}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        out[row["kind"]].append(row)
    return out


a, b = rows(sys.argv[1]), rows(sys.argv[2])
for kind in ("llm", "http", "db"):
    ka, kb = [r["key"] for r in a[kind]], [r["key"] for r in b[kind]]
    n = next((i for i, (x, y) in enumerate(zip(ka, kb)) if x != y), min(len(ka), len(kb)))
    print(kind, "A", len(ka), "B", len(kb), "common prefix", n, "keys in A not B", len(set(ka) - set(kb)), "keys in B not A", len(set(kb) - set(ka)))
    if n < min(len(ka), len(kb)):
        for side, r in (("A", a[kind][n]), ("B", b[kind][n])):
            print(" ", side, json.dumps(r["request"], ensure_ascii=False)[:300])
    for r in [r for r in a[kind] if r["key"] not in set(kb)][:4]:
        print("  only-A:", json.dumps(r["request"], ensure_ascii=False)[:260])
