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

totals = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "errors": 0}
reds: list[str] = []
for index, rel in enumerate(FILES, 1):
    proc = subprocess.run(
        [PY, "-m", "pytest", rel, "-q", "--no-header", "-rf",
         f"--basetemp={BASETEMP / ('g%02d' % index)}"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    counts = {key: 0 for key in totals}
    for value, key in re.findall(r"(\d+) (passed|failed|skipped|errors|error)", out):
        counts[key] = int(value)
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
    if proc.returncode not in (0, 1) or (proc.returncode == 1 and not counts["failed"] and not errors):
        print(out[-3000:])
        raise SystemExit(f"INFRASTRUCTURE FAILURE on {rel}: exit={proc.returncode} with no failure")

print()
print(f"files run    : {len(FILES)}")
print(f"split totals : {totals['passed']} passed, {totals['failed']} failed, "
      f"{totals['skipped']} skipped, {totals['error'] + totals['errors']} errors")
print(f"reds ({len(reds)}):")
for red in reds:
    print(f"    {red}")
raise SystemExit(0)
