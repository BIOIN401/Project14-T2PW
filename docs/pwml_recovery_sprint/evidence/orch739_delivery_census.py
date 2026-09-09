#!/usr/bin/env python3
"""ORCH-739 - leg-level provider/model delivery census. READ ONLY.

Reconstructs a call-level timeline from every LEG_TRACE.jsonl in the recent
cohorts and reports delivery cost at the unit that matters: the leg.

Measurement conditions, stated so the numbers are falsifiable:

  * object            : LEG_TRACE.jsonl `model_attempt` rows, which are
                        CLIENT-level calls, not extraction-ladder attempts.
                        One ladder attempt can be three rows here.
  * duration          : gap to the previous traced event. This is an UPPER
                        BOUND on call duration - non-LLM work between two
                        events is charged to the later call. For a run of
                        consecutive retries carrying one request_hash the
                        bound is tight, which is the case this tool is for.
  * leg runtime       : `wall time` in RESULT.txt when present, else the
                        last elapsed_seconds in the trace.
  * classification    : from status/finish_reason/content_chars only. The
                        trace cannot see downstream JSON parse outcome, so
                        `malformed` is NOT decidable here and is never
                        claimed. See ORCH-733 section 1 for the parse-level view.

Nothing is written outside the report path. No network, no LLM, no re-run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Derived, not invented: ORCH-733 section 1 fixed 200 chars as the tiny/degenerate
# threshold from observed completions (2 chars, 84-1267 chars) against genuine
# truncations at 9,501 and 10,895. Reused unchanged so the two censuses compare.
TINY_CHARS = 200

COHORTS = {
    "ORCH-732": "runs_smoke/2026-09-07_2323",
    "ORCH-734": "runs_smoke/2026-09-08_1528",
    "ORCH-730": "runs_validation/2026-09-07_1929",
    "C-120-validation": "runs_validation/c120/2026-09-08_1240",
    "C-121-live-validation": "runs_validation/c121/2026-09-09_0028",
}

WALL_RE = re.compile(r"wall time\s*:\s*([0-9.]+)s")
STATUS_RE = re.compile(r"^status\s*:\s*(\S+)", re.M)
RESULT_RE = re.compile(r"^RESULT:\s*(\S+)", re.M)
STAGE_RE = re.compile(r"^stage\s*:\s*(\S+)", re.M)


def classify(row: dict) -> str:
    """One call -> one delivery class. Content and finish_reason only."""
    status = row.get("status")
    chars = row.get("content_chars") or 0
    finish = row.get("finish_reason") or ""
    if status not in ("ok", "empty"):
        # No such row exists in the current corpus; kept so a future
        # provider exception is classified rather than silently dropped.
        return "provider_failure"
    if chars == 0:
        return "empty"
    if chars <= TINY_CHARS:
        return "degenerate"
    if finish == "length":
        return "truncated"
    return "normal"


def is_stage1(stage: str) -> bool:
    return "Stage 1" in (stage or "")


def read_leg(leg_dir: Path) -> dict:
    trace = leg_dir / "LEG_TRACE.jsonl"
    rows = []
    if trace.exists():
        for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    calls = []
    prev_elapsed = 0.0
    for row in rows:
        elapsed = float(row.get("elapsed_seconds") or 0.0)
        if row.get("kind") == "model_attempt":
            calls.append(
                {
                    "seq": row.get("seq"),
                    "stage": row.get("stage"),
                    "model": row.get("model"),
                    "attempt": row.get("attempt"),
                    "start_after": prev_elapsed,
                    "elapsed_at": elapsed,
                    "gap_seconds": round(max(0.0, elapsed - prev_elapsed), 3),
                    "content_chars": row.get("content_chars"),
                    "finish_reason": row.get("finish_reason"),
                    "status": row.get("status"),
                    "request_hash": row.get("request_hash"),
                    "reason": row.get("reason"),
                    "klass": classify(row),
                }
            )
        prev_elapsed = elapsed

    # Leg runtime: RESULT.txt is authoritative, trace is the fallback.
    result_txt = leg_dir / "RESULT.txt"
    wall = status = verdict = stage = None
    if result_txt.exists():
        text = result_txt.read_text(encoding="utf-8", errors="replace")
        m = WALL_RE.search(text)
        if m:
            wall = float(m.group(1))
        m = STATUS_RE.search(text)
        if m:
            status = m.group(1)
        m = RESULT_RE.search(text)
        if m:
            verdict = m.group(1)
        m = STAGE_RE.search(text)
        if m:
            stage = m.group(1)
    runtime_source = "RESULT.txt"
    if wall is None:
        wall = round(prev_elapsed, 1)
        runtime_source = "trace_last_elapsed"

    counts = {k: 0 for k in ("normal", "empty", "degenerate", "truncated", "provider_failure")}
    time_by_class = {k: 0.0 for k in counts}
    for c in calls:
        counts[c["klass"]] += 1
        time_by_class[c["klass"]] += c["gap_seconds"]

    s1 = [c for c in calls if is_stage1(c["stage"])]
    s1_counts = {k: 0 for k in counts}
    for c in s1:
        s1_counts[c["klass"]] += 1

    # Longest unbroken run of non-productive completions, and its cost.
    worst_chain = 0
    worst_chain_secs = 0.0
    cur = 0
    cur_secs = 0.0
    for c in calls:
        if c["klass"] in ("empty", "degenerate"):
            cur += 1
            cur_secs += c["gap_seconds"]
            if cur > worst_chain:
                worst_chain, worst_chain_secs = cur, cur_secs
        else:
            cur, cur_secs = 0, 0.0

    wasted = time_by_class["empty"] + time_by_class["degenerate"]

    pwml = sorted(p.name for p in leg_dir.glob("*.pwml"))
    models = sorted({c["model"] for c in calls if c.get("model")})

    by_stage = {}
    for c in calls:
        st = by_stage.setdefault(c["stage"] or "?", {k: 0 for k in counts})
        st[c["klass"]] += 1

    return {
        "paper": leg_dir.parent.name,
        "mode": leg_dir.name,
        "leg_dir": leg_dir.as_posix(),
        "verdict": verdict,
        "status": status,
        "stage_reported": stage,
        "runtime_seconds": wall,
        "runtime_source": runtime_source,
        "has_trace": trace.exists(),
        "models_used": models,
        "calls_total": len(calls),
        "counts": counts,
        "time_by_class_seconds": {k: round(v, 1) for k, v in time_by_class.items()},
        "wasted_seconds_upper_bound": round(wasted, 1),
        "wasted_fraction": round(wasted / wall, 4) if wall else None,
        "worst_nonproductive_chain": worst_chain,
        "worst_chain_seconds": round(worst_chain_secs, 1),
        "stage1_calls": len(s1),
        "stage1_counts": s1_counts,
        "by_stage": by_stage,
        "pwml_files": pwml,
        "pwml": bool(pwml),
        "calls": calls,
    }


def discover(root: Path, rel: str) -> list[Path]:
    base = root / rel
    if not base.exists():
        return []
    # rglob, never a fixed-depth glob: ORCH-733 section 8 recorded a fixed-depth
    # pattern silently undercounting the one nested run family.
    legs = {p.parent for p in base.rglob("RESULT.txt")}
    legs |= {p.parent for p in base.rglob("LEG_TRACE.jsonl")}
    return sorted(legs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()

    cohorts = {}
    all_legs = []
    for name, rel in COHORTS.items():
        legs = discover(root, rel)
        entries = [read_leg(p) for p in legs]
        for e in entries:
            e["cohort"] = name
        cohorts[name] = {"path": rel, "legs_found": len(entries), "legs": entries}
        all_legs.extend(entries)

    traced = [e for e in all_legs if e["calls_total"] > 0]
    with_pwml = [e for e in traced if e["pwml"]]

    def med(xs):
        xs = sorted(xs)
        if not xs:
            return None
        n = len(xs)
        return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 2)

    totals = {k: 0 for k in ("normal", "empty", "degenerate", "truncated", "provider_failure")}
    for e in traced:
        for k, v in e["counts"].items():
            totals[k] += v

    summary = {
        "legs_discovered": len(all_legs),
        "legs_with_trace": len(traced),
        "calls_total": sum(e["calls_total"] for e in traced),
        "class_totals": totals,
        "empty_rate_all_calls": round(
            totals["empty"] / sum(e["calls_total"] for e in traced), 4
        )
        if traced
        else None,
        "models_seen_corpus_wide": sorted({m for e in traced for m in e["models_used"]}),
        "median_empty_per_successful_leg": med([e["counts"]["empty"] for e in with_pwml]),
        "worst_empty_count_any_leg": max((e["counts"]["empty"] for e in traced), default=0),
        "worst_empty_leg": max(traced, key=lambda e: e["counts"]["empty"])["leg_dir"]
        if traced
        else None,
        "total_wasted_seconds_upper_bound": round(
            sum(e["wasted_seconds_upper_bound"] for e in traced), 1
        ),
        "total_runtime_seconds": round(sum(e["runtime_seconds"] or 0 for e in traced), 1),
        "legs_wasting_over_25pct": [
            {
                "leg": e["leg_dir"],
                "fraction": e["wasted_fraction"],
                "wasted_s": e["wasted_seconds_upper_bound"],
                "runtime_s": e["runtime_seconds"],
                "pwml": e["pwml"],
            }
            for e in traced
            if (e["wasted_fraction"] or 0) > 0.25
        ],
        "timeout_legs": [
            {
                "leg": e["leg_dir"],
                "runtime_s": e["runtime_seconds"],
                "empty": e["counts"]["empty"],
                "wasted_s": e["wasted_seconds_upper_bound"],
                "fraction": e["wasted_fraction"],
            }
            for e in traced
            if e["status"] == "timeout"
        ],
        "fallback_model_calls": sum(
            1
            for e in traced
            for c in e["calls"]
            if c.get("model") and "deepseek" not in (c["model"] or "")
        ),
    }

    report = {"tool": "orch739_delivery_census.py", "summary": summary, "cohorts": cohorts}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
