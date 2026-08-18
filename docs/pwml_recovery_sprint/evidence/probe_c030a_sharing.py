"""C-030a / A0-C7 probe: measure the sharing sites, and prove the card is test-only.

Three measurements in one committed JSON, so a reviewer checks an artifact rather
than a paragraph.

1. **The discriminator line, re-measured in THIS tree.** F-041 recorded
   ``streamlit_app.py:3748``; the orchestrator re-measured ``:3746`` before C-052
   merged ~116 lines into the file. Every card in this sprint that inherited a
   line number was wrong, so the measured value and both deltas are recorded.

2. **The test-only certification.** For a test-only card, "a guard that passes at
   the base SHA" (G9) reduces to one checkable fact: the production source is
   identical at base and tip. Checked for all of ``src/`` and, separately, as the
   AST hash of the two functions involved -- the normalization
   ``test_c011_freeze_seam_golden_equivalence.py`` uses for its own hard stop.
   Deliberately NOT a committed test: against the working tree it would fail the
   moment a later card legitimately edits the orchestrator, and a gate that fires
   for a bookkeeping reason is one somebody eventually weakens.

3. **The mutant-discrimination matrix**, re-derived through the committed tests'
   own helpers. F-008 found a share->copy mutant indistinguishable from the tip on
   all nine seam observables across 39 legs; this is the evidence these clauses
   are not.

Run bounded, per G11::

    <py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label c030a-probe \\
        --timeout 600 --json <g11 slot> -- <py> \\
        docs/pwml_recovery_sprint/evidence/probe_c030a_sharing.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# F-045: resolve every output path from ``__file__`` BEFORE anything can change
# the working directory, and assert where the artifact landed at the end.
EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
OUT = EVIDENCE / "c030a_sharing_probe.json"

APP_REL = "src/t2pw/app/streamlit_app.py"
APP_PATH = ROOT / APP_REL
BASE_SHA = "01bb7eff2d1fb07e7ab7a7ebfe4e3403e71757c5"
EP1 = "run_post_pipeline_sbml_artifacts"
SEAM = "freeze_canonical_payload"

#: What the record said before this card measured it. Both are stale.
F041_RECORDED_LINE = 3748
ORCHESTRATOR_PRE_C052_LINE = 3746
DISCRIMINATOR_SOURCE = '"final_mapped": canonical_export_payload or final_export_payload,'

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True).stdout


def _ast_hash(source: str, name: str) -> str:
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    dumped = ast.dump(node, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def measure_sites() -> Dict[str, Any]:
    """Every line where the orchestrator publishes or binds the canonical object."""

    lines = APP_PATH.read_text(encoding="utf-8").splitlines()
    found: Dict[str, int] = {}
    wanted = {
        "seam_binding": 'canonical_export_payload = _freeze["payload"]',
        "canonical_payload_key": "CANONICAL_PAYLOAD_KEY: canonical_export_payload,",
        "final_mapped_quarantined": '"final_mapped_quarantined": canonical_export_payload,',
        "final_mapped": DISCRIMINATOR_SOURCE,
        "final_export_input": '"final_export_input": canonical_export_payload or final_export_payload,',
        "enriched_republication": '"final_mapped_enriched": final_export_payload,',
    }
    for label, needle in wanted.items():
        hits = [n for n, line in enumerate(lines, 1) if line.strip() == needle]
        assert len(hits) == 1, f"{label}: expected one site, found {hits}"
        found[label] = hits[0]

    discriminator = found["final_mapped"]
    return {
        "measured_lines": found,
        "discriminator": {
            "line": discriminator,
            "source": DISCRIMINATOR_SOURCE,
            "f041_recorded_line": F041_RECORDED_LINE,
            "delta_vs_f041": discriminator - F041_RECORDED_LINE,
            "orchestrator_pre_c052_line": ORCHESTRATOR_PRE_C052_LINE,
            "delta_vs_orchestrator_pre_c052": discriminator - ORCHESTRATOR_PRE_C052_LINE,
            "note": ("C-052 merged 116 lines into this file after the orchestrator's "
                     "re-measurement, so both prior figures are stale. This is the "
                     "value measured in this worktree."),
        },
    }


def certify_test_only() -> Dict[str, Any]:
    """The whole G9 claim for a test-only card, as a measurement."""

    changed_src = [p for p in _git("diff", "--name-only", BASE_SHA, "HEAD", "--", "src")
                   .decode("utf-8").splitlines() if p.strip()]
    base_source = _git("show", f"{BASE_SHA}:{APP_REL}").decode("utf-8")
    tip_source = APP_PATH.read_text(encoding="utf-8")
    return {
        "base_sha": BASE_SHA,
        "src_files_changed_base_to_head": changed_src,
        "src_is_untouched": changed_src == [],
        "app_source_identical": base_source == tip_source,
        "function_ast_hashes": {
            name: {"base": _ast_hash(base_source, name),
                   "tip": _ast_hash(tip_source, name),
                   "equal": _ast_hash(base_source, name) == _ast_hash(tip_source, name)}
            for name in (EP1, SEAM)
        },
        "g9_classification": "guard_on_pre_existing_behaviour_passing_at_base",
        "g9_note": ("No base failure is claimed or offered. The production source is "
                    "identical at base and tip, so every green here is a green at base "
                    "by construction; manufacturing a base failure would be a "
                    "fabrication. What is new is the DISCHARGE of A0-C7, not a "
                    "capability. Same split as C-052's test_a0c8_guard_* names."),
    }


def run_mutant_matrix() -> List[Dict[str, Any]]:
    """The committed tests' own helpers, one row per mutant. Non-vacuity, measured."""

    import test_c030a_object_sharing_at_the_freeze_seam as c030a  # noqa: PLC0415

    class _Patch:
        """The two lines of ``monkeypatch`` this probe needs, without pytest."""

        def __init__(self) -> None:
            self.undo: List[Any] = []

        def setattr(self, target: Any, name: str, value: Any) -> None:
            self.undo.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def restore(self) -> None:
            for target, name, old in reversed(self.undo):
                setattr(target, name, old)

    cases = [
        ("copy_at_final_mapped", dict(mutant=lambda: c030a._CopyAtKeys(
            ["final_mapped"], c030a.CANONICAL_LOCAL))),
        ("copy_at_final_mapped_quarantined", dict(mutant=lambda: c030a._CopyAtKeys(
            ["final_mapped_quarantined"], c030a.CANONICAL_LOCAL))),
        ("copy_at_final_export_input", dict(mutant=lambda: c030a._CopyAtKeys(
            ["final_export_input"], c030a.CANONICAL_LOCAL))),
        ("copy_at_canonical_payload_key", dict(mutant=lambda: c030a._CopyAtKeys(
            [c030a.CANONICAL_KEY_NODE], c030a.CANONICAL_LOCAL))),
        ("copy_at_seam_binding", dict(mutant=c030a._CopyAtSeamBinding)),
        ("unmutated_tip", {}),
        ("refusal_arm_unmutated", dict(refuse=True)),
        ("refusal_arm_copy_at_final_mapped", dict(refuse=True, mutant=lambda: c030a._CopyAtKeys(
            ["final_mapped"], c030a.PRE_SEAM_LOCAL))),
    ]

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as raw:
        for index, (label, options) in enumerate(cases):
            patch = _Patch()
            c030a._offline_prefreeze(patch)
            factory = options.pop("mutant", None)
            work = Path(raw) / f"case{index}"
            work.mkdir(parents=True, exist_ok=True)
            try:
                result, seen = c030a._run(
                    work, **({"mutant": factory()} if factory else {}), **options)
                rows.append({
                    "case": label,
                    "refusal_arm": bool(options.get("refuse")),
                    "success_arm_clauses_broken": sorted(
                        c030a._sharing_violations(result, seen)),
                    "refusal_arm_clauses_broken": sorted(
                        c030a._refusal_violations(result, seen)),
                })
            finally:
                patch.restore()
    return rows


def main() -> int:
    document = {
        "task": "C-030a",
        "requirement": "A0-C7 -- object sharing across the freeze seam, proved not assumed",
        "boundary": (f"{APP_REL}::{EP1}. NOT {SEAM}: F-008 proved A0-C7 undischargeable "
                     "there, so a test written against it would be vacuous by construction."),
        "sites": measure_sites(),
        "test_only_certification": certify_test_only(),
        "mutant_matrix": run_mutant_matrix(),
    }
    OUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # F-045: say where it landed, absolutely, rather than trusting the cwd.
    assert OUT.is_file(), OUT
    print(f"probe written: {OUT}")
    print(json.dumps(document["sites"]["discriminator"], indent=2, sort_keys=True))
    for row in document["mutant_matrix"]:
        print(f"  {row['case']:38s} success_arm={row['success_arm_clauses_broken']} "
              f"refusal_arm={row['refusal_arm_clauses_broken']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
