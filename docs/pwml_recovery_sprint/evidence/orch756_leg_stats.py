"""ORCH-756/759: per-leg LLM attempt stats from LEG_TRACE.jsonl (stdlib only, no t2pw import).

For each trace: per stage, attempts, empty finish_reason=length completions (the ORCH-755 measure,
same rule as orch755_trace_stats.py), other empty attempts, and the wall time attributed to the stage.
A trace records each model attempt when it ENDS (``elapsed_seconds``) and no start time, so a call's
duration is taken as the time since the previous model attempt ended: an UPPER bound on the call
itself (it also holds any non-LLM work in between).

Then every LONG call (>= 1,200 s by that measure) and every provider ``finish_reason=error`` attempt,
one line each: stage, attempt, finish_reason, content chars, seconds.

    python orch756_leg_stats.py "<glob of LEG_TRACE.jsonl>" [...]
"""
import collections
import glob
import json
import sys

LONG = 1200.0
totals = collections.Counter()
for pattern in sys.argv[1:]:
    for path in sorted(glob.glob(pattern)):
        stats = collections.OrderedDict()
        flagged = []
        prev = 0.0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("kind") != "model_attempt":
                continue
            s = stats.setdefault(r.get("stage"), {"n": 0, "len_empty": 0, "other_empty": 0, "secs": 0.0})
            el = float(r.get("elapsed_seconds") or prev)
            dur = max(0.0, el - prev)
            prev = el
            s["n"] += 1
            s["secs"] += dur
            empty = not r.get("content_chars")
            if empty and r.get("finish_reason") == "length":
                s["len_empty"] += 1
            elif empty:
                s["other_empty"] += 1
            err = r.get("finish_reason") == "error"
            if dur >= LONG or err:
                flagged.append((r.get("stage"), r.get("attempt"), r.get("finish_reason"), r.get("content_chars"), dur))
                totals["long"] += dur >= LONG
                totals["error"] += err
        totals["calls"] += sum(s["n"] for s in stats.values())
        print(path)
        print(f"  last model attempt at {prev:.0f} s")
        for stage, s in stats.items():
            print(f"  {stage:<34} attempts {s['n']:>3}  empty-length {s['len_empty']:>3}  "
                  f"other-empty {s['other_empty']:>3}  ~{s['secs']:>6.0f} s")
        for stage, att, fr, chars, dur in flagged:
            print(f"    LONG/ERROR  {stage} attempt {att}: finish_reason={fr} chars={chars} ~{dur:.0f} s")
print(f"TOTAL model attempts {totals['calls']}, long (>= {LONG:.0f} s) {totals['long']}, "
      f"finish_reason=error {totals['error']}")
