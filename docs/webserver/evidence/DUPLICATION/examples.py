import json, sys, re, os
sys.path.insert(0, "C:/t/ev-dedup")
import scan_runs as S
rows = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl", encoding="utf-8")]
def date(r):
    m = re.search(r"(20\d\d-\d\d-\d\d)_\d{4}", r["path"])
    if m: return m.group(1)
    return __import__("time").strftime("%Y-%m-%d", __import__("time").localtime(os.path.getmtime(r["path"])))
for r in rows: r["date"] = date(r)
json.dump([{ "path": r["path"], "date": r["date"], "q": r["quarantine_ok"]} for r in rows], open("C:/t/ev-dedup/out/run_dates.json", "w"))
print("date range", min(r["date"] for r in rows), max(r["date"] for r in rows))
from collections import Counter
print(Counter(r["date"][:7] for r in rows))
def show(key, n=6):
    k = 0
    for r in rows:
        if r.get(key) and r["quarantine_ok"] is True:
            print(key, r["date"], r["path"][-95:], r[key]); k += 1
            if k >= n: break
show("a_rx_exact"); show("a_tr_exact", 3)
for r in rows:
    if r["b_examples"] and r["quarantine_ok"] is True:
        print("b2", r["date"], r["path"][-80:], r["b_examples"][:3])
