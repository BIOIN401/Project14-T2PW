"""A4: print short context windows around regex hits in a sample paper text. usage: a4_grep.py IDX REGEX [WIDTH] [MAXHITS]"""
import json, re, sys
from pathlib import Path

sample = {s["idx"]: s for s in json.loads(Path("C:/t/corr-assess/sample.json").read_text(encoding="utf-8"))}
idx, pat = int(sys.argv[1]), sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 140
mx = int(sys.argv[4]) if len(sys.argv) > 4 else 8
t = Path(sample[idx]["paper_text"]).read_text(encoding="utf-8", errors="replace")
t = re.sub(r"\s+", " ", t)
n = 0
for m in re.finditer(pat, t, re.I):
    print(f"[{m.start()}] ...{t[max(0, m.start() - w):m.end() + w]}...")
    n += 1
    if n >= mx:
        break
print(f"-- {len(re.findall(pat, t, re.I))} hits, text len {len(t)}")
