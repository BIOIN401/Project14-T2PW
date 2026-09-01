"""C-110 probe -- what a leg record CAN and CANNOT distinguish, and the RESULT.txt gap.

Three questions, answered offline from constructed rows and from shipped code.
**No committed run directory is opened and T-107 is not read, re-scored or
re-interpreted anywhere in this file.**

1. **INVENTORY.** Which fields does a manifest row carry that could separate an
   empty leg that DECLINED from one that was KILLED -- and which of them does the
   acceptance scorer actually reach? Named gaps are the point: they belong to
   F-148.

2. **DISCRIMINATION.** Run the four canonical row shapes through the new rule and
   print what each one earns and why. This is the measurement behind condition 3.

3. **THE `RESULT: FAIL` STOP CONDITION.** ``batch.runner.result_text(row, paper=)``
   receives a manifest row and a paper dict -- **no GoldCase**. Print what it
   emits for a correct decline and for a timeout, and show that the two rows
   differ in ways the file never prints, so a reader of ``RESULT.txt`` cannot
   tell them apart on the page even though the row could.

Usage::

    <venv-python> c110_leg_record_inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench import acceptance  # noqa: E402
from t2pw.bench.goldset import load_gold_set  # noqa: E402
from t2pw.bench.metrics import classify_strict_boundary  # noqa: E402
from t2pw.pipeline.deadline import (  # noqa: E402
    BUDGET_EXHAUSTED,
    OPERATIONAL_TERMINATION_REASONS,
    TERMINATION_REASONS,
)

RULE = "=" * 78
THIN = "-" * 78


DECLINED = {
    "paper_id": "PMC13231680",
    "slug": "PMC13231680",
    "mode": "strict",
    "status": "fail",
    "stage": "stage1",
    "failure_kind": "no_reactions",
    "message": (
        "extraction produced no reactions: nothing lipid-A-related is present in this "
        "paper at any level of partiality, so no pathway was exported"
    ),
    "detail": "(stage 1 returned an empty process set)",
    "issue_codes": [],
    "counts": {"reactions": 0, "transports": 0, "entities": 0},
    "files": [{"name": "extraction_diagnostics.json", "bytes": 812}],
}

TIMED_OUT = {
    "paper_id": "PMC13231680",
    "slug": "PMC13231680",
    "mode": "strict",
    "status": "timeout",
    "stage": "unknown",
    "failure_kind": "timeout",
    "termination_reason": BUDGET_EXHAUSTED,
    "operational_failure": True,
    "message": (
        "the child process was still running after 1800s and was killed, so this "
        f"paper+mode produced nothing ({BUDGET_EXHAUSTED})"
    ),
    "detail": "(the child printed nothing before it was killed)",
    "issue_codes": [],
    "counts": {},
    "files": [],
}

SILENT = dict(DECLINED, failure_kind="", message="", detail="")
NO_ARTIFACTS = dict(DECLINED, files=[], counts={})

# REV-110 round 1's three rows. Each EARNED the status before the correction,
# through an issue code that both satisfied condition 2 and cancelled the
# indeterminate refusal. Kept here so the shapes stay measurable.
CONTRACT_RELABELLED = dict(
    DECLINED,
    failure_kind="contract",
    issue_codes=["gate.protein_x_is_missing_a_unipro"],
    message="the model call failed and the gate could not be evaluated",
)
UNKNOWN_WITH_CODE = dict(
    DECLINED,
    failure_kind="unknown",
    issue_codes=["processes_required"],
    # driver.py:2565, verbatim. A message whose CONTENT is that no reason was
    # given was being scored as a stated reason.
    message="no research report was produced and no reason was given",
)
AMBIGUOUS_WITH_CODES = dict(
    DECLINED,
    failure_kind="ambiguous_review_scope",
    issue_codes=["scope.ambiguous"],
    message="the requested scope was ambiguous and the run stopped",
)


def _leg(row):
    """Build the ModeResult exactly as ``score_run`` does, without a run dir."""

    leg = acceptance.ModeResult(
        paper_id=row["paper_id"], mode=row["mode"], attempted=True
    )
    leg.status = row.get("status", "")
    leg.stage = row.get("stage", "")
    leg.failure_kind = row.get("failure_kind", "")
    leg.message = row.get("message", "")
    leg.issue_codes = list(row.get("issue_codes") or ())
    leg.termination_reason = str(row.get("termination_reason") or "")
    leg.operational_failure = row.get("operational_failure") is True
    leg.artifacts_recorded = len([f for f in (row.get("files") or ()) if f])
    leg.recorded_counts = dict(row.get("counts") or {})
    leg.boundary, leg.boundary_evidence = classify_strict_boundary(row)
    return leg


def main() -> int:
    gold = load_gold_set()
    case = next(c for c in gold if c.paper_id == "PMC13231680")
    arm2 = next(c for c in gold if c.paper_id == "PMC12180156")

    print(RULE)
    print("1. INVENTORY -- what the leg record carries about declined-vs-killed")
    print(RULE)
    print(
        "Written by batch/driver.py::RunOutcome.to_dict and batch/runner.py::"
        "_timeout_row / _crash_row."
    )
    print()
    rows = [
        ("status", "closed: pass|fail|error|timeout|scope_conflict", "YES", "runner writes error/timeout; driver writes pass/fail/scope_conflict"),
        ("failure_kind", "closed: " + ", ".join(sorted({"contract", "llm", "network", "ambiguous_review_scope", "no_reactions", "crash", "timeout", "unknown"})), "YES", "no_reactions is the declared decline; timeout/crash are casualties"),
        ("stage", "free-ish; 'unknown' on the OUTER kill path", "YES", "F-148: 'unknown' means the PARENT could not see, not a pipeline stage"),
        ("message", "free text", "YES", "the stated reason. EMPTY is a silence"),
        ("issue_codes", "list of guard/gate codes", "YES", "an explicit named stop"),
        ("termination_reason", "closed, 7 values: " + ", ".join(TERMINATION_REASONS), "NEW in C-110", "conditional key; ABSENT on every leg that did not time out"),
        ("operational_failure", "bool", "NEW in C-110", "is_operational(termination_reason); written by BOTH kill paths under this name"),
        ("counts", "dict; {} on the outer-kill path", "NEW in C-110", "{} = NOT MEASURED; {'reactions': 0} = affirmatively none"),
        ("files", "list; [] on the outer-kill path", "NEW in C-110", "F-148 section 3: [] is the signature of a kill with the reserve spent"),
        ("budget", "dict, timeout rows only", "NO", "GAP -- elapsed/remaining never reach the acceptance leg"),
        ("budget_unrecorded", "str, inner timeout only", "NO", "GAP"),
        ("detail", "free text, capped", "NO", "GAP -- the traceback tail never reaches the acceptance leg"),
        ("warnings", "list", "NO", "GAP"),
        ("attempts / retries", "DOES NOT EXIST", "N/A", "F-148 section 5: the artifact needed to exclude retry amplification is the one the kill destroyed"),
    ]
    print(f"{'row field':<22}{'shape':<58}{'reached by scorer':<20}note")
    for name, shape, reached, note in rows:
        print(f"{name:<22}{shape[:56]:<58}{reached:<20}{note}")
    print()
    print("GAPS, NAMED (they belong to F-148, not to this card):")
    print("  * `budget` / `budget_unrecorded` / `detail` / `warnings` are on the row")
    print("    and are NOT carried onto the acceptance leg. C-110 did not add them:")
    print("    none of them separates declined from killed better than the four it did.")
    print("  * There is NO attempt or retry record of any kind on ANY row shape. A leg")
    print("    that was killed cannot be shown to have retried or not retried.")
    print("  * `stage` is 'unknown' on the outer-kill path BY DESIGN. A reader treating")
    print("    it as a pipeline stage will mis-diagnose. C-110 does not read `stage`.")
    print("  * A row written before termination_reason/operational_failure existed has")
    print("    NEITHER key. CORRECTED (round 1): that is NOT treated as indeterminate.")
    print("    An old-shape DECLINE (no_reactions + message + files) passes cleanly,")
    print("    which is right. What absence buys is only that those two readings do")
    print("    not fire; an old-shape TIMEOUT is still caught by status, failure_kind")
    print("    and boundary -- which is why the casualty test has FIVE readings.")
    print()

    print(RULE)
    print("2. DISCRIMINATION -- the four canonical shapes through the new rule")
    print(RULE)
    for name, row in (
        ("declined (the ruling's case)", DECLINED),
        ("timed out (outer kill)", TIMED_OUT),
        ("empty and silent", SILENT),
        ("empty, no artifacts preserved", NO_ARTIFACTS),
        ("REV-110 #1 provider-as-contract", CONTRACT_RELABELLED),
        ("REV-110 #2 unknown + one code", UNKNOWN_WITH_CODE),
        ("REV-110 #3 ambiguous + codes", AMBIGUOUS_WITH_CODES),
    ):
        record = acceptance.negative_control_outcome(case, _leg(row))
        print(f"{name:<32} -> {record['status']}")
        print(f"{'':<32}    conditions : {record['conditions']}")
        print(f"{'':<32}    blocked_by : {record['blocked_by']}")
    print()
    print("BOTH ARMS of _empty_is_correct, on the pinned gold:")
    for c in (case, arm2):
        record = acceptance.negative_control_outcome(c, _leg(DECLINED))
        print(
            f"  {c.paper_id}  is_negative_control={c.is_negative_control}  "
            f"relevance={c.mechanistic_relevance}  min_connected={c.min_connected_reactions}"
            f"  -> arm={record['arm']}  status={record['status']}"
        )
    print()
    print("A relevant paper is not judged by this rule at all:")
    for pid in ("PMC12096016", "PMC12452463"):
        c = next(x for x in gold if x.paper_id == pid)
        print(
            f"  {pid}  _empty_is_correct={acceptance._empty_is_correct(c)}  "
            f"record={acceptance.negative_control_outcome(c, _leg(DECLINED))}"
        )
    print()
    print(f"OPERATIONAL_TERMINATION_REASONS (D-005, imported not restated): "
          f"{sorted(OPERATIONAL_TERMINATION_REASONS)}")
    print()

    print(RULE)
    print("3. THE batch/runner.py STOP CONDITION -- measured, not argued")
    print(RULE)
    from t2pw.batch.runner import result_text

    print("result_text(row, paper=...) signature carries NO GoldCase:")
    import inspect

    print(f"  {inspect.signature(result_text)}")
    print()
    for name, row in (("DECLINED CORRECTLY", DECLINED), ("KILLED BY THE CLOCK", TIMED_OUT)):
        text = result_text(row, paper={"title": "a paper"})
        print(THIN)
        print(f"row shape: {name}")
        print(THIN)
        print(text.rstrip())
        print()
    print(THIN)
    print("WHAT THIS SHOWS")
    print(THIN)
    print("* Both rows render `RESULT: FAIL`. On a negative control the first one is")
    print("  the outcome the gold DEMANDS, and the page says the opposite.")
    print("* The two rows DO differ, and `result_text` prints SOME of the difference")
    print("  and not the rest. It prints `counts` and `files` -- so the empty COUNTS")
    print("  and FILES blocks on the timeout row are visible. It does NOT print")
    print("  `termination_reason` or `operational_failure`, which are the two fields")
    print("  that say IN TERMS that the leg was an operational casualty and that")
    print("  D-005 forbids reading as a biological verdict. That omission is a")
    print("  gold-free honesty gap, and it is REGISTERED here, not fixed: it belongs")
    print("  to F-148's deadline seam, and RESULT.txt is a live pipeline artifact.")
    print("* Even with that gap closed, `RESULT: FAIL` would still be wrong on the")
    print("  DECLINE row, because nothing in the row says the decline was correct.")
    print("* Making `RESULT: FAIL` non-misleading for the DECLINE case requires knowing")
    print("  the paper is a negative control. That fact lives only in the gold set, and")
    print("  `result_text` has no access to it. STOP CONDITION TAKEN: batch/runner.py is")
    print("  NOT modified by C-110.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
