"""Stage-1 terminal-failure census — read-only, over archived run artifacts.

Nothing is re-run, no LLM is called, no network is touched, no run directory is
written, no production code is imported.

**Discovery is by SHAPE, not depth**, and is reconciled against an independent
filesystem count before any number is interpreted. The `F-192` census used a
fixed-depth ``*/papers/*/*`` glob and silently missed a nested run family; this
one matches ``…/papers/<paper>/<mode>`` at any depth and refuses to report if the
reconciliation fails.

Two populations, deliberately kept apart:

* **leg-level** — the unit that decides whether a PWML can exist. Source of
  truth is ``RESULT.txt`` (present for nearly every leg).
* **attempt-level** — individual Stage-1 model calls. Source is
  ``LEG_TRACE.jsonl``, which exists for only a subset of legs; its coverage is
  reported alongside every attempt number so it is never mistaken for the whole
  corpus.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[3]
FAMILIES = ("runs", "runs_verify", "runs_smoke", "runs_validation")

#: Mechanism classes (§ 4 of the charter).
CLASS_EMPTY_LENGTH = "A_empty_completion_length"
CLASS_TINY_STOP = "B_tiny_completion_stop"
CLASS_TRUNCATED_JSON = "C_truncated_json_length"
CLASS_MALFORMED_JSON = "D_malformed_json_not_truncated"
CLASS_SEMANTIC_GUARD = "E_semantic_guard_after_recovery"
CLASS_PROVIDER_EXC = "F_provider_exception"
CLASS_OTHER_TERMINAL = "G_other_terminal"
#: Not provider-DELIVERY failures. Stage 1 returned parseable output and the
#: pipeline rejected it, or a guard declined to extract at all. Kept apart from
#: A-F because a retry/alternate-route policy cannot help these and counting them
#: together would inflate the case for one.
CLASS_MISSING_CONTAINER = "H_valid_json_missing_required_container"
CLASS_ZERO_PROCESSES = "I_valid_extraction_zero_processes"
CLASS_GUARD_SKIPPED = "J_extraction_skipped_by_guard"
CLASS_UNCLASSIFIABLE = "Z_no_attempt_evidence_retained"
#: The leg's wall-clock budget expired. Recorded at stage1 by the driver but the
#: message names a later stage, so it is NOT a Stage-1 delivery failure.
CLASS_LEG_TIMEOUT = "T_leg_timeout_not_stage1_delivery"

#: Classes where the provider failed to DELIVER usable output. Only these are
#: candidates for a retry / alternate-route resilience policy.
DELIVERY_CLASSES = frozenset({
    CLASS_EMPTY_LENGTH, CLASS_TINY_STOP, CLASS_TRUNCATED_JSON,
    CLASS_MALFORMED_JSON, CLASS_PROVIDER_EXC,
})

#: "Tiny" completion threshold. Chosen from the corpus rather than invented: the
#: two genuine Stage-1 truncations carry 9,501 and 10,895 content chars, the
#: smallest VALID Stage-1 extraction retained in the archive is far above this,
#: and the degenerate completions this class is for measure 2 chars (PMC7615680)
#: and 84-1,267 chars (PMC8510960's repair rungs). 200 sits above every observed
#: degenerate completion and two orders of magnitude below any real extraction.
_TINY_CHARS = 200

#: Substrings the *production* code uses to recognise a truncated-JSON failure
#: (``pipeline.py::_looks_truncated_json_failure``). Reused verbatim so the census
#: classifies by the same rule the code does, not by a rule invented here. They
#: are applied ONLY when a sizeable completion exists -- see ``looks_truncated``.
_TRUNCATION_MARKERS = (
    "unterminated string",
    "expecting value",
    "unexpected end",
    "eof",
)


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _load(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001
        return None


def discover_legs() -> Tuple[List[Path], Dict[str, Any]]:
    """Every ``…/papers/<paper>/<mode>`` directory, at any depth, plus a reconciliation."""
    legs: List[Path] = []
    for family in FAMILIES:
        root = _REPO / family
        if not root.is_dir():
            continue
        for leg in root.rglob("papers/*/*"):
            if leg.is_dir() and leg.name in {"strict", "research"}:
                legs.append(leg)
    legs = sorted(set(legs))

    # Independent counts, derived a different way, to catch a discovery bug.
    independent: Dict[str, int] = {}
    for artifact in ("RESULT.txt", "stage1_payload.json", "LEG_TRACE.jsonl",
                     "extraction_boundary_report.json", "final_mapped.json"):
        found = 0
        for family in FAMILIES:
            root = _REPO / family
            if root.is_dir():
                found += sum(1 for _ in root.rglob(artifact))
        independent[artifact] = found

    from_legs = {
        artifact: sum(1 for leg in legs if (leg / artifact).exists())
        for artifact in independent
    }
    mismatches = {
        artifact: {"independent": independent[artifact], "via_discovered_legs": from_legs[artifact]}
        for artifact in independent
        if independent[artifact] != from_legs[artifact]
    }
    return legs, {
        "legs_discovered": len(legs),
        "independent_artifact_counts": independent,
        "same_counts_via_discovered_legs": from_legs,
        "mismatches": mismatches,
        "reconciled": not mismatches,
    }


def _result_facts(leg: Path) -> Dict[str, Any]:
    text = _text(leg / "RESULT.txt")
    if not text:
        return {"present": False}
    out: Dict[str, Any] = {"present": True}
    for key, pattern in (
        ("status", r"^status\s*:\s*(.+)$"),
        ("stage", r"^stage\s*:\s*(.+)$"),
        ("message", r"^message\s*:\s*(.+)$"),
        ("failure_kind", r"^failure_kind\s*:\s*(.+)$"),
        ("issue_codes", r"^issue codes\s*:\s*(.+)$"),
        ("termination", r"^termination_reason\s*:\s*(.+)$"),
    ):
        match = re.search(pattern, text, re.MULTILINE)
        out[key] = match.group(1).strip() if match else ""
    for key in ("reactions", "blocking_issues", "gate_errors"):
        match = re.search(rf"^\s*{key}\s*=\s*(\d+)\s*$", text, re.MULTILINE)
        out[key] = int(match.group(1)) if match else None
    out["has_pwml"] = bool(list(leg.glob("*.pwml")))
    return out


def _stage1_attempts(leg: Path) -> List[Dict[str, Any]]:
    """Stage-1 model attempts recorded in ``LEG_TRACE.jsonl`` (subset of legs)."""
    path = leg / "LEG_TRACE.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in _text(path).splitlines():
        line = line.strip()
        if not line or '"model_attempt"' not in line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if row.get("kind") != "model_attempt":
            continue
        rows.append(row)
    return rows


def _is_stage1(stage: str) -> bool:
    text = str(stage or "").casefold()
    return "stage 1" in text or "stage1" in text


def classify(leg: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one terminal Stage-1 failure by mechanism, from artifacts only."""
    boundary = _load(leg / "extraction_boundary_report.json")
    attempts = [row for row in _stage1_attempts(leg) if _is_stage1(row.get("stage"))]

    outcomes: List[str] = []
    boundaries: List[Dict[str, Any]] = []
    if isinstance(boundary, dict):
        outcomes = [str(v) for v in (boundary.get("outcomes") or [])]
        boundaries = [b for b in (boundary.get("boundaries") or []) if isinstance(b, dict)]

    # Attempt-log rows carried inside the boundary report (richer than the trace
    # for older legs, and present when the trace is not).
    attempt_log: List[Dict[str, Any]] = []
    for entry in boundaries:
        for row in entry.get("attempt_log") or []:
            if isinstance(row, dict):
                attempt_log.append(row)

    pool = attempts or attempt_log
    finishes = Counter(str(row.get("finish_reason") or "") for row in pool)
    statuses = Counter(str(row.get("status") or "") for row in pool)
    chars = [int(row.get("content_chars") or 0) for row in pool if row.get("content_chars") is not None]

    # Judge on STRUCTURED fields and on the leg's own message. The previous
    # version substring-matched a JSON blob that carries sha256 hex, so "429"
    # matched inside a hash and mislabelled the F-193 exemplar as a provider
    # exception. Blob matching is now confined to the message text.
    message = str(result.get("message") or "").casefold()
    errors = " ".join(
        str(entry.get("error") or "") for entry in boundaries
    ).casefold() + " " + " ".join(str(row.get("error") or "") for row in pool).casefold()

    empty_length = sum(
        1
        for row in pool
        if int(row.get("content_chars") or 0) == 0
        and str(row.get("finish_reason") or "") == "length"
    )
    tiny_stop = sum(
        1
        for row in pool
        if 0 < int(row.get("content_chars") or 0) <= _TINY_CHARS
        and str(row.get("finish_reason") or "") == "stop"
    )
    big_length = sum(
        1
        for row in pool
        if int(row.get("content_chars") or 0) > _TINY_CHARS
        and str(row.get("finish_reason") or "") == "length"
    )
    # A truncation marker only means truncation when there was CONTENT to cut.
    # "Expecting value: line 1 column 1 (char 0)" on an empty completion is the
    # absence of output, not a truncated object.
    looks_truncated = big_length > 0 and any(m in errors for m in _TRUNCATION_MARKERS)

    exception_words = ("timeout", "connection error", "rate limit", "apierror", "readtimeout")
    provider_exception = (
        any(word in errors for word in exception_words)
        or "exception" in statuses
        or "error" in statuses
    )

    # Content / guard outcomes first -- these are NOT delivery failures.
    if str(result.get("status") or "").split()[:1] == ["timeout"]:
        mechanism = CLASS_LEG_TIMEOUT
    elif "extraction skipped" in message or "no selected_example" in message:
        mechanism = CLASS_GUARD_SKIPPED
    elif "must include a processes object" in message or "must include an entities object" in message             or "processes_required" in message or "entities_required" in message:
        mechanism = CLASS_MISSING_CONTAINER
    elif "no reactions and no transports" in message:
        mechanism = CLASS_ZERO_PROCESSES
    elif "semantic_guard_failed" in errors or "semantic_guard" in outcomes:
        mechanism = CLASS_SEMANTIC_GUARD
    elif not pool and not isinstance(boundary, dict):
        mechanism = CLASS_UNCLASSIFIABLE
    elif provider_exception:
        mechanism = CLASS_PROVIDER_EXC
    elif empty_length and empty_length >= max(1, len(pool) - 1):
        mechanism = CLASS_EMPTY_LENGTH
    elif looks_truncated:
        mechanism = CLASS_TRUNCATED_JSON
    elif tiny_stop and not big_length:
        mechanism = CLASS_TINY_STOP
    elif (
        "invalid_json" in outcomes
        or "invalid json" in message
        or "failed to produce valid json" in message
    ):
        # Sizeable content that finished cleanly and still would not parse is
        # MALFORMED, not truncated -- the provider stopped of its own accord.
        mechanism = (
            CLASS_MALFORMED_JSON
            if not looks_truncated
            else CLASS_TRUNCATED_JSON
        )
    elif empty_length:
        mechanism = CLASS_EMPTY_LENGTH
    elif not pool:
        mechanism = CLASS_UNCLASSIFIABLE
    else:
        mechanism = CLASS_OTHER_TERMINAL

    return {
        "mechanism": mechanism,
        "boundary_report_present": isinstance(boundary, dict),
        "boundary_outcomes": outcomes,
        "attempt_source": "LEG_TRACE" if attempts else ("boundary_attempt_log" if attempt_log else "none"),
        "stage1_attempts_seen": len(pool),
        "finish_reasons": dict(finishes),
        "statuses": dict(statuses),
        "content_chars": sorted(chars)[:12],
        "empty_length_attempts": empty_length,
        "tiny_stop_attempts": tiny_stop,
        "sizeable_length_attempts": big_length,
        "looks_truncated_json": looks_truncated,
        "terminal_reason": sorted(
            {str(entry.get("terminal_reason") or "") for entry in boundaries if entry.get("terminal_reason")}
        ),
    }


def main() -> int:
    legs, reconciliation = discover_legs()
    if not reconciliation["reconciled"]:
        sys.stdout.write(json.dumps({"RECONCILIATION_FAILED": reconciliation}, indent=1) + "\n")
        return 3

    leg_rows: List[Dict[str, Any]] = []
    for leg in legs:
        result = _result_facts(leg)
        parts = leg.parts
        row: Dict[str, Any] = {
            "path": str(leg.relative_to(_REPO)).replace("\\", "/"),
            "run": "/".join(parts[parts.index("papers") - 2 : parts.index("papers")])
            if "papers" in parts
            else "",
            "family": str(leg.relative_to(_REPO)).replace("\\", "/").split("/")[0],
            "paper": parts[-2],
            "mode": parts[-1],
            "result": result,
        }
        leg_rows.append(row)

    # ── leg-level ────────────────────────────────────────────────────────────
    with_result = [row for row in leg_rows if row["result"].get("present")]
    reached_stage1 = [
        row
        for row in with_result
        if (Path(_REPO / row["path"]) / "stage1_payload.json").exists()
        or _is_stage1(row["result"].get("stage"))
        or str(row["result"].get("stage") or "") not in {"", "stage0", "input", "acquisition"}
    ]
    terminal_stage1 = [
        row
        for row in with_result
        if str(row["result"].get("status") or "").split()[0] in {"fail", "timeout", "error"}
        and _is_stage1(row["result"].get("stage"))
    ]

    for row in terminal_stage1:
        row["classification"] = classify(Path(_REPO / row["path"]), row["result"])

    by_mechanism: Counter = Counter(row["classification"]["mechanism"] for row in terminal_stage1)
    papers_by_mechanism: Dict[str, set] = defaultdict(set)
    for row in terminal_stage1:
        papers_by_mechanism[row["classification"]["mechanism"]].add(row["paper"])

    # ── attempt-level, over the legs that carry a trace ───────────────────────
    trace_legs = [row for row in leg_rows if (Path(_REPO / row["path"]) / "LEG_TRACE.jsonl").exists()]
    all_attempts: List[Dict[str, Any]] = []
    for row in trace_legs:
        all_attempts.extend(_stage1_attempts(Path(_REPO / row["path"])))
    stage1_attempts = [row for row in all_attempts if _is_stage1(row.get("stage"))]

    def attempt_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        chars = [int(r.get("content_chars") or 0) for r in rows]
        return {
            "attempts": len(rows),
            "finish_stop": sum(1 for r in rows if r.get("finish_reason") == "stop"),
            "finish_length": sum(1 for r in rows if r.get("finish_reason") == "length"),
            "finish_other": sum(
                1 for r in rows if str(r.get("finish_reason") or "") not in {"stop", "length"}
            ),
            "status_counts": dict(Counter(str(r.get("status") or "") for r in rows)),
            "empty_content": sum(1 for c in chars if c == 0),
            "tiny_content_1_200": sum(1 for c in chars if 0 < c <= 200),
            "empty_and_length": sum(
                1
                for r in rows
                if int(r.get("content_chars") or 0) == 0 and r.get("finish_reason") == "length"
            ),
            "sizeable_and_length": sum(
                1
                for r in rows
                if int(r.get("content_chars") or 0) > 200 and r.get("finish_reason") == "length"
            ),
        }

    summary = {
        "reconciliation": reconciliation,
        "leg_level": {
            "legs_discovered": len(legs),
            "legs_with_RESULT_txt": len(with_result),
            "legs_without_RESULT_txt": [
                row["path"] for row in leg_rows if not row["result"].get("present")
            ],
            "legs_reaching_or_past_stage1": len(reached_stage1),
            "terminal_stage1_failures": len(terminal_stage1),
            "distinct_papers_affected": len({row["paper"] for row in terminal_stage1}),
            "by_family": dict(Counter(row["family"] for row in terminal_stage1)),
            "status_stage_matrix": dict(
                Counter(
                    f"{row['result'].get('status')} | {row['result'].get('stage')}"
                    for row in with_result
                ).most_common()
            ),
        },
        "mechanism_table": {
            mechanism: {
                "terminal_legs": count,
                "papers": sorted(papers_by_mechanism[mechanism]),
                "paper_count": len(papers_by_mechanism[mechanism]),
                "stage1_attempts_in_those_legs": sum(
                    row["classification"]["stage1_attempts_seen"]
                    for row in terminal_stage1
                    if row["classification"]["mechanism"] == mechanism
                ),
            }
            for mechanism, count in by_mechanism.most_common()
        },
        "attempt_level": {
            "coverage_note": "LEG_TRACE.jsonl exists for a SUBSET of legs; this is not the whole corpus",
            "legs_with_trace": len(trace_legs),
            "legs_total": len(legs),
            "all_stages": attempt_stats(all_attempts),
            "stage1_only": attempt_stats(stage1_attempts),
            "stage1_content_char_buckets": dict(
                Counter(
                    (
                        "0"
                        if int(r.get("content_chars") or 0) == 0
                        else "1-200"
                        if int(r.get("content_chars") or 0) <= 200
                        else "201-2000"
                        if int(r.get("content_chars") or 0) <= 2000
                        else "2001-10000"
                        if int(r.get("content_chars") or 0) <= 10000
                        else ">10000"
                    )
                    for r in stage1_attempts
                ).most_common()
            ),
        },
    }

    detail = [
        {
            "path": row["path"],
            "paper": row["paper"],
            "mode": row["mode"],
            "family": row["family"],
            "message": row["result"].get("message"),
            "failure_kind": row["result"].get("failure_kind"),
            "classification": row["classification"],
        }
        for row in terminal_stage1
    ]

    sys.stdout.write(
        json.dumps({"summary": summary, "terminal_stage1_legs": detail}, indent=1, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
