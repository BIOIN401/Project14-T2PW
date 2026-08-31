"""C-102: run the 22-file gold-readers selection ONE FILE PER PROCESS.

A combined total hides a shift between two files: one file losing a test while
another gains one leaves `453 passed` unchanged and unremarkable. This runs each
file alone, in a fresh process, and prints its own counts, so a per-file move is
visible even when the total is not.

Usage::

    <python> c102_goldreaders_split.py <worktree-root> <basetemp-parent>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASETEMP = Path(sys.argv[2])
# The parent MUST exist before pytest is handed a --basetemp under it. Without
# it every test errors in setup and the file reports 0 passed with exit 1 -- an
# infrastructure failure that reads exactly like a wiped test file. The first
# run of this driver did precisely that; its log is kept beside this fix.
BASETEMP.mkdir(parents=True, exist_ok=True)
PY = sys.executable
FILES = """
tests/test_protein_export_policy.py tests/test_c056b_semantic_denominators.py
tests/test_batch_pwml_artifact_naming.py tests/test_c074_strict_core_floor.py
tests/test_c076_alias_holo_apo_identity.py tests/test_c077_stage0_conflict_disposition.py
tests/test_c081_cofactor_role_identity.py tests/test_c085_priority2_honesty.py
tests/test_c088_extracted_not_serialized_disposition.py tests/test_cofactor_policy.py
tests/test_rag_multi_relation_spans.py tests/test_release_status_classification.py
tests/test_semantic_production_no_gold.py tests/test_strict_failure_replay.py
tests/test_batch_driver_quarantine_artifacts.py tests/test_c056c_semantic_evaluability.py
tests/test_c071_actor_evidence_gate.py tests/test_c080_production_gate_kind_aware.py
tests/test_c089_participant_schema.py tests/test_c090_wrapper_identity_actor_evidence.py
tests/test_c097_semantic_name_keys.py tests/test_semantic_release_gating.py
""".split()

# F-152 -- the count parse is scoped to pytest's OWN SUMMARY LINE.
#
# It used to be `re.findall(r"(\d+) (passed|failed|skipped|errors|error)", out)`
# over the ENTIRE combined stdout+stderr. A green file whose output merely
# CONTAINED the text "3 errors" -- a warning, a captured log line, a failure
# message -- was recorded as `errors=3`. Before C-104 that was inert (a spurious
# `errors` only suppressed an abort); after C-104 it is FATAL, because `errors`
# now aborts on its own. REV-104 measured the difference:
#
#     SCENARIO green_with_warning_text
#       BASE exit=0 aborted=False files_reported=22
#       TIP  exit=1 aborted=True  files_reported=1
#
# pytest's summary line starts with "<n> <outcome>" and ENDS with the duration,
# optionally wrapped in the '=' rule pytest draws outside -q and optionally
# carrying a parenthesised elapsed time for long runs. Anchoring on BOTH ends is
# what keeps prose out: a log line reading "found 3 errors in the payload" does
# not start with a count, and "3 errors in 12.5s of work" does not end with one.
#
# The LAST matching line wins -- pytest prints its summary last, and a rerun
# banner or a captured earlier summary must not outvote it.
#
# This narrows ONLY the parse. The abort predicate below is C-104's, was reviewed
# and merged under D-083, and is deliberately left exactly as it was: a genuine
# `1 failed` must still be counted and still fold into the totals. A parse that
# killed the false positive by counting nothing would be a worse defect than the
# one it fixed.
_SUMMARY_LINE = re.compile(
    r"^=*\s*\d+ [a-z]+.*\bin \d+(?:\.\d+)?s(?: \(\d+:\d\d:\d\d\))?\s*=*\s*$"
)
_COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors|error)\b")


def summary_counts(out: str) -> dict[str, int]:
    """Counts from pytest's terminal summary line, and from nowhere else."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "errors": 0}
    summary = ""
    for line in out.splitlines():
        stripped = line.strip()
        if _SUMMARY_LINE.match(stripped):
            summary = stripped
    for value, key in _COUNT.findall(summary):
        counts[key] = int(value)
    return counts


totals = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "errors": 0}
reds: list[str] = []
for index, rel in enumerate(FILES, 1):
    proc = subprocess.run(
        [PY, "-m", "pytest", rel, "-q", "--no-header", "-rf",
         f"--basetemp={BASETEMP / ('g%02d' % index)}"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    counts = summary_counts(out)
    for key in totals:
        totals[key] += counts[key]
    reds.extend(re.findall(r"FAILED (\S+)", out))
    errors = counts["error"] + counts["errors"]
    print(
        f"{index:>2}. {rel:62s} passed={counts['passed']:>3} "
        f"failed={counts['failed']} skipped={counts['skipped']} errors={errors}"
        f"  exit={proc.returncode}"
    )
    # A nonzero exit with nothing failed and nothing errored is not a test
    # result. Abort rather than fold it into a total.
    #
    # D-083 follow-on 2: NEITHER IS A SETUP ERROR, at any exit code. The two
    # conditions above were specified as "nonzero exit with nothing failed AND
    # NOTHING ERRORED", so on the original F-114 condition -- a missing
    # `--basetemp` parent, which errors tests in setup -- the second disjunct is
    # false and nothing fires. That run lost 71 tests, reported them as simply
    # absent, and still exited 0
    # (`c102_goldreaders_split_r1.attempt1-missing-basetemp-parent.log`). Error
    # counting made the loss visible but not fatal.
    #
    # So `errors` now aborts ON ITS OWN, independent of the exit code and of
    # `failed`: a setup error is an INFRASTRUCTURE FAILURE in the same class as
    # a surviving process, never a legitimate outcome of this gate. The two
    # original conditions are kept exactly as they were -- a planted bad import
    # still fires on its exit code, and a genuine red test still folds into the
    # totals rather than aborting the gate.
    if errors or proc.returncode not in (0, 1) or (
        proc.returncode == 1 and not counts["failed"] and not errors
    ):
        print(out[-3000:])
        raise SystemExit(
            f"INFRASTRUCTURE FAILURE on {rel}: exit={proc.returncode} errors={errors} "
            f"failed={counts['failed']} passed={counts['passed']} -- a setup error or an "
            f"unexpected exit is not a test result"
        )

print()
print(f"files run    : {len(FILES)}")
print(f"split totals : {totals['passed']} passed, {totals['failed']} failed, "
      f"{totals['skipped']} skipped, {totals['error'] + totals['errors']} errors")
print(f"reds ({len(reds)}):")
for red in reds:
    print(f"    {red}")
raise SystemExit(0)
