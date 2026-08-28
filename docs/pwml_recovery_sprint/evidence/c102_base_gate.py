"""C-102: measure the gates and the G9 behavioural proof on the BASE SHA.

Why not ``c045b_base_tree.py``. Its pathspec deliberately excludes ``runs_verify``
and ``runs``, and every measurement this card makes reads a committed run
artifact, so an exported base tree would fail the gate for want of data rather
than for want of code. It would not be a base result.

What this does instead, and why it is exactly equivalent here. The only tracked
files this branch changes are the two ``src/t2pw/bench`` modules and one new test
file -- verified below, not asserted -- so restoring those two modules to their
base blobs and setting the new test aside puts every file the gate selection
reads into byte-identical agreement with the base SHA. The check is: nothing else
in ``src`` or ``tests`` differs from base while the base leg runs.

F-143. The interpreter is passed explicitly, never as a bare ``python``:
``bounded_run.py`` resolves a bare name from the CHILD's PATH, which here finds a
system interpreter with no ``streamlit`` and produces a large block of import
errors that reads exactly like a regression.

Usage::

    <python> c102_base_gate.py <worktree-root> <base-sha>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASE = sys.argv[2]
PY = sys.executable
REVERT = ("src/t2pw/bench/acceptance.py", "src/t2pw/bench/render.py")
NEW_TEST = "tests/test_c102_coverage_denominator.py"
GOLD_READERS = """
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
#: F-142, chartered as C-103. Pre-existing at base, untouched by this card, and
#: named here so a THIRD failure is unmistakably ours.
KNOWN_RED = "only_unrelated_reactions_survive"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def run(label: str, argv: list[str]) -> int:
    proc = subprocess.run(
        argv, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    print(f"\n--- {label} --- exit={proc.returncode}")
    for line in out.splitlines():
        if re.match(r"^(FAILED|ERROR)", line) or re.search(r"\d+ (passed|failed|error)", line):
            print(f"    {line}")
        elif label.startswith("G9") :
            print(f"    {line}")
    return proc.returncode


def pytest(paths: list[str], basetemp: str) -> list[str]:
    return [PY, "-m", "pytest", *paths, "-q", "--no-header", "-rf", f"--basetemp={basetemp}"]


changed = set(git("diff", "--name-only", BASE, "HEAD").splitlines())
code_changed = {p for p in changed if p.startswith(("src/", "tests/"))}
print(f"base                : {git('rev-parse', BASE)}")
print(f"tip                 : {git('rev-parse', 'HEAD')}")
print(f"src/tests changed   : {sorted(code_changed)}")
assert code_changed == {*REVERT, NEW_TEST}, code_changed
assert git("status", "--porcelain", "--", "src", "tests") == "", "src/tests must be clean"

git("checkout", BASE, "--", *REVERT)
Path(ROOT / NEW_TEST).rename(ROOT / (NEW_TEST + ".setaside"))
try:
    residual = git("diff", "--name-only", BASE, "--", "src", "tests")
    print(f"residual diff vs base while the base leg runs: {residual!r}  (must be empty)")
    assert residual == "", residual

    gold = run("GOLD-READERS at base", pytest(GOLD_READERS, "C:/t/bt/c102gb"))
    g9 = run(
        "G9 behavioural proof at base",
        [PY, "docs/pwml_recovery_sprint/evidence/c102_g9_denominator_proof.py", str(ROOT)],
    )
    ab = run(
        "A/B probe at base",
        [PY, "docs/pwml_recovery_sprint/evidence/c102_f132_coverage_ab.py", str(ROOT)],
    )
    # REV-102 F3/F5. The base report sizes, so the growth this card causes is a
    # measured difference rather than a quoted one -- including on a run with NO
    # leg carrying a coverage block, which is the case the first round got wrong.
    run(
        "G9-REPORT-SIZE at base",
        [PY, "docs/pwml_recovery_sprint/evidence/c102_report_size.py", str(ROOT)],
    )
finally:
    git("checkout", "HEAD", "--", *REVERT)
    Path(ROOT / (NEW_TEST + ".setaside")).rename(ROOT / NEW_TEST)

print(f"\ntree after restore  : {git('status', '--porcelain', '--', 'src', 'tests')!r}  (must be empty)")
print(f"gold-readers at base: exit={gold}   (1 is CORRECT -- the two F-142 reds)")
print(f"G9 proof at base    : exit={g9}     (MUST be nonzero: the behaviour is absent at base)")
print(f"A/B probe at base   : exit={ab}     (3 = post-change half unavailable, the pre-change leg)")
if g9 == 0:
    print("\nG9 FAILED: the proof passes on the base SHA, so it proves nothing")
    raise SystemExit(1)
raise SystemExit(0 if git("status", "--porcelain", "--", "src", "tests") == "" else 1)
