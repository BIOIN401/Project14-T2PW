"""Show where two transcripts' LLM request N differ (first difference in the last user message and system)."""
import difflib
import json
import sys
from pathlib import Path


def llm(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row["kind"] == "llm":
            out.append(row)
    return out


a, b = llm(sys.argv[1]), llm(sys.argv[2])
n = int(sys.argv[3])
ra, rb = a[n]["request"], b[n]["request"]
print("same keys:", a[n]["key"] == b[n]["key"], {k: (ra.get(k) == rb.get(k)) for k in ra})
for i, (ma, mb) in enumerate(zip(ra["messages"], rb["messages"])):
    ta, tb = str(ma["content"]), str(mb["content"])
    if ta == tb:
        continue
    print("message", i, ma["role"], len(ta), len(tb))
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    print(len(ops), "diff ops")
    for tag, i1, i2, j1, j2 in ops[:8]:
        print(tag, i1, i2, j1, j2)
        print("  A:", repr(ta[max(0, i1 - 120):i2 + 120])[:420])
        print("  B:", repr(tb[max(0, j1 - 120):j2 + 120])[:420])
