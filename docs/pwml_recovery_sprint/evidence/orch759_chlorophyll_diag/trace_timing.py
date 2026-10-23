"""Stdlib-only: per-stage wall time and per-call gaps from LEG_TRACE.jsonl.

A model_attempt row carries the elapsed time at which the call RETURNED, so a call's
duration is bounded by the gap to the previous row (an upper bound: local work in
between is included).
"""
import json
from collections import Counter, OrderedDict
from pathlib import Path

LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
rows = [json.loads(line) for line in (LEG / "LEG_TRACE.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
prev = 0.0
per_stage = OrderedDict()
first_last = OrderedDict()
long_calls = []
for r in rows:
    t = float(r.get("elapsed_seconds") or 0.0)
    gap = t - prev
    if r["kind"] == "model_attempt":
        st = r["stage"]
        per_stage.setdefault(st, [0, 0.0, Counter()])
        per_stage[st][0] += 1
        per_stage[st][1] += gap
        per_stage[st][2][(r["status"], r["finish_reason"])] += 1
        first_last.setdefault(st, [t - gap, t])[1] = t
        if gap >= 600:
            long_calls.append((r["seq"], st, round(gap, 1), r["status"], r["finish_reason"], r["content_chars"]))
        if r["finish_reason"] not in ("stop",) or r["status"] != "ok":
            print("NON-STOP/NON-OK:", r["seq"], st, r["status"], r["finish_reason"], r.get("reason"), round(gap, 1), r["content_chars"])
    else:
        print("EVENT:", json.dumps(r)[:300])
    prev = t
print("\nstage | calls | sum(gap) s | span start-end s | outcomes")
for st, (n, s, c) in per_stage.items():
    a, b = first_last[st]
    print(f"{st} | {n} | {s:.0f} | {a:.0f}-{b:.0f} | {dict(c)}")
print("\ncalls with gap >= 600 s:", long_calls)
print("max gap:", max(((float(rows[i]['elapsed_seconds']) - float(rows[i-1]['elapsed_seconds'])), rows[i]['seq'], rows[i].get('stage')) for i in range(1, len(rows))))
print("end elapsed:", rows[-1]["elapsed_seconds"])
