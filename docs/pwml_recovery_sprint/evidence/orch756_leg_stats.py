"""ORCH-756/759: per-leg LLM attempt stats from LEG_TRACE.jsonl (stdlib only, no t2pw import).

For each trace: per stage, attempts, empty finish_reason=length completions (the ORCH-755 measure,
same rule as orch755_trace_stats.py), other empty/error attempts, and the wall time attributed to the
stage (elapsed since the previous model_attempt record, a lower bound on time spent waiting for it).

    python orch756_leg_stats.py "<glob of LEG_TRACE.jsonl>" [...]
"""
import collections
import glob
import json
import sys

for pattern in sys.argv[1:]:
    for path in sorted(glob.glob(pattern)):
        stats = collections.OrderedDict()
        prev = 0.0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("kind") != "model_attempt":
                continue
            s = stats.setdefault(r.get("stage"), {"n": 0, "len_empty": 0, "other_empty": 0, "secs": 0.0})
            el = float(r.get("elapsed_seconds") or prev)
            s["n"] += 1
            s["secs"] += max(0.0, el - prev)
            prev = el
            empty = not r.get("content_chars")
            if empty and r.get("finish_reason") == "length":
                s["len_empty"] += 1
            elif empty:
                s["other_empty"] += 1
        print(path)
        print(f"  last model attempt at {prev:.0f} s")
        for stage, s in stats.items():
            print(f"  {stage:<34} attempts {s['n']:>3}  empty-length {s['len_empty']:>3}  "
                  f"other-empty {s['other_empty']:>3}  ~{s['secs']:>6.0f} s")
