"""ORCH-758: Stage-1 vs downstream wall-clock per leg, for a reference run and ORCH-755. Stdlib only, reads only.

Usage: python orch758_reference_timing.py <glob of LEG_TRACE.jsonl> [<glob> ...]

Per leg: source-text chars, Stage-1 wall-clock (end of the last attempt before the first
non-Stage-1 attempt that follows Stage 1), downstream wall-clock (Stage-1 end to leg end or kill),
seconds on empty finish_reason=length attempts downstream, downstream seconds net of them,
and the leg's status. "downstream" = everything after Stage 1: RAG, Stage 2, audit, curator,
gap resolver, mapping, gates, export.
"""
import glob
import json
import os
import sys

for pattern in sys.argv[1:]:
    print(f"\n### {pattern}")
    print(f"{'leg':<44} {'chars':>7} {'s1_s':>6} {'down_s':>6} {'down_empty_s':>12} {'down_net_s':>10} {'s2_calls':>8}  status")
    for trace in sorted(glob.glob(pattern)):
        leg_dir = os.path.dirname(trace)
        rows = [json.loads(line) for line in open(trace, encoding="utf-8") if line.strip()]
        src = os.path.join(os.path.dirname(leg_dir), "01_source_text.txt")
        chars = os.path.getsize(src) if os.path.exists(src) else -1
        term_path = os.path.join(leg_dir, "LEG_TERMINAL.json")
        status, end = "?", None
        if os.path.exists(term_path):
            st = json.load(open(term_path, encoding="utf-8")).get("terminal_state_before_cleanup") or {}
            status, end = st.get("status"), st.get("elapsed_seconds")
        ends = [r for r in rows if r.get("kind") == "leg_end"]
        if ends:
            status = status if status != "?" else ends[-1].get("status")
            end = end or ends[-1].get("elapsed_seconds")
        attempts = [r for r in rows if r.get("kind") == "model_attempt"]
        prev, s1_end, seen_s1, down_empty = 0.0, 0.0, False, 0.0
        s2_hashes = set()
        for r in attempts:
            t = float(r["elapsed_seconds"])
            if r["stage"] == "Stage 1 extraction":
                seen_s1, s1_end = True, t
            elif seen_s1 and r["stage"] != "preprocessor":
                seen_s1 = None  # Stage 1 is over
            if seen_s1 is None and r.get("finish_reason") == "length" and not r.get("content_chars"):
                down_empty += t - prev
            if r["stage"] == "Stage 2 inference":
                s2_hashes.add(r.get("request_hash"))
            prev = t
        end = float(end if end is not None else prev)
        down = end - s1_end
        leg = os.sep.join(trace.split(os.sep)[-6:-1]) if "legs" in trace else os.sep.join(trace.split(os.sep)[-3:-1])
        print(f"{leg[-44:]:<44} {chars:>7} {s1_end:>6.0f} {down:>6.0f} {down_empty:>12.0f} {down - down_empty:>10.0f} {len(s2_hashes):>8}  {status}")
