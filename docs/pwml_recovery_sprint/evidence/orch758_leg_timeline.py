"""ORCH-758: per-leg wall-clock timeline from each ORCH-755 LEG_TRACE.jsonl. Stdlib only, reads only.

Usage: python orch758_leg_timeline.py <ORCH-755 legs dir>

A model_attempt event's elapsed_seconds is stamped when the attempt ENDS, so an attempt's
duration is the gap from the previous event. Consecutive attempts of one stage form a phase.
Per leg it prints every phase (start, end, wall-clock, attempts, empty-length attempts, the
seconds spent on empty attempts), then per-stage totals, then the time an empty-free leg
would have saved (the seconds spent on attempts that came back empty).
"""
import collections
import glob
import json
import os
import sys

root = sys.argv[1]
summary = {}
for trace in sorted(glob.glob(os.path.join(root, "*", "*", "papers", "*", "*", "LEG_TRACE.jsonl"))):
    leg = trace.split(os.sep)[-6]
    rows = [json.loads(line) for line in open(trace, encoding="utf-8") if line.strip()]
    term = json.load(open(os.path.join(os.path.dirname(trace), "LEG_TERMINAL.json"), encoding="utf-8"))
    state = term.get("terminal_state_before_cleanup") or {}
    print(f"\n=== {leg}  status={state.get('status')} reason={state.get('termination_reason')} "
          f"elapsed={state.get('elapsed_seconds')} child_deadline={rows[0].get('child_deadline_seconds')}")
    phases = []
    prev = 0.0
    per_stage = collections.defaultdict(lambda: {"n": 0, "empty": 0, "secs": 0.0, "empty_secs": 0.0})
    for r in rows:
        if r.get("kind") != "model_attempt":
            if r.get("kind") not in ("leg_begin", "stage_begin"):
                print("   event", r)
            continue
        t = float(r["elapsed_seconds"])
        dur = t - prev
        prev = t
        empty = r.get("finish_reason") == "length" and not r.get("content_chars")
        st = per_stage[r["stage"]]
        st["n"] += 1
        st["secs"] += dur
        if empty:
            st["empty"] += 1
            st["empty_secs"] += dur
        if phases and phases[-1]["stage"] == r["stage"]:
            p = phases[-1]
        else:
            p = {"stage": r["stage"], "start": t - dur, "n": 0, "empty": 0, "empty_secs": 0.0,
                 "attempts": []}
            phases.append(p)
        p["end"] = t
        p["n"] += 1
        p["empty"] += empty
        p["empty_secs"] += dur if empty else 0.0
        p["attempts"].append(f"{r.get('attempt')}{'E' if empty else ('L' if r.get('finish_reason') == 'length' else '')}"
                             f"({dur:.0f}s,{r.get('content_chars')}c,{r.get('request_hash', '')[-6:]})")
    for p in phases:
        print(f"  {p['start']:7.0f}-{p['end']:7.0f}s  {p['end'] - p['start']:6.0f}s  {p['stage']:<24} "
              f"n={p['n']:<3} empty={p['empty']:<3} empty_s={p['empty_secs']:5.0f}")
        if p["stage"] in ("Stage 1 extraction", "Stage 2 inference", "audit loop", "chat", "gap resolver",
                          "extraction json repair", "inference json repair"):
            print("        ", " ".join(p["attempts"]))
    last = prev
    total_empty = sum(s["empty_secs"] for s in per_stage.values())
    print(f"  last model event at {last:.0f}s; tail after it to kill: {float(state.get('elapsed_seconds') or 0) - last:.0f}s")
    print("  per stage: " + " | ".join(f"{k}: {v['n']} att, {v['empty']} empty, {v['secs']:.0f}s "
                                       f"({v['empty_secs']:.0f}s empty)" for k, v in per_stage.items()))
    print(f"  seconds on empty-length attempts: {total_empty:.0f}")
    summary[leg] = {"status": state.get("status"), "elapsed": state.get("elapsed_seconds"),
                    "last_model_event": round(last), "empty_seconds": round(total_empty),
                    "per_stage": {k: {kk: round(vv) for kk, vv in v.items()} for k, v in per_stage.items()},
                    "phases": [{k: (round(v) if isinstance(v, float) else v) for k, v in p.items() if k != "attempts"}
                               for p in phases]}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orch758_leg_timeline.json")
json.dump(summary, open(out, "w", encoding="utf-8"), indent=1)
print("\nwrote", out)
