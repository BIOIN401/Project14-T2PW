"""Non-vacuity harness for C-057's golden-move helper (shared block section 6).

``_with_c057_lineage_hashes`` states C-057's payload-digest delta against the
C-011 freeze-seam golden instead of regenerating the fixture. A stated delta that
cannot FAIL is a rewritten fixture with extra steps, so this perturbs the
produced document one field at a time and requires the helper to reject each one.

The two arms that carry the whole thing are ``connectivity`` and
``normalization_stats``: those are the BIOLOGY. A lineage write that had also
moved the graph would move them, and the helper has to fail there rather than
absorb it. That is merge rules 6 and 7 made mechanical rather than argued.

Committed runnable, not just as a transcript, because the next card that has to
decide whether ITS golden move is safe needs a harness it can point at its own
helper -- not a result it has to re-implement from a report.

    .venv/Scripts/python.exe \\
      docs/pwml_recovery_sprint/evidence/c057_golden_delta_perturbations.py \\
      --tree .

Exit 0 when every perturbation is rejected; exit 1 if any is silently accepted,
which is the finding this exists to catch. Reads the tree, writes nothing.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, help="checkout carrying src/ and tests/")
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    sys.path[:0] = [str(tree / "src"), str(tree / "tests")]
    import test_c011_freeze_seam_golden_equivalence as g  # noqa: E402

    before: Dict[str, Any] = json.loads(g._fixture_bytes())
    after = g._document(None)

    quiet = sorted(n for n, l in after["legs"].items() if l["final_stage3_gate_report"])[0]
    moved = sorted(
        n for n, l in before["legs"].items()
        if l["final_stage3_gate_report"]
        and l["final_stage3_gate_report"]["payload_sha256"]
        != after["legs"][n]["final_stage3_gate_report"]["payload_sha256"])
    refusal = sorted(n for n, l in before["legs"].items() if not l["final_stage3_gate_report"])[0]

    def report(leg: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        return doc["legs"][leg]["final_stage3_gate_report"]

    def digest(doc: Dict[str, Any]) -> None:
        report(quiet, doc)["payload_sha256"] = "sha256:deadbeef"
        doc["legs"][quiet]["canonical_payload_sha256"] = "sha256:deadbeef"

    arms: List[Tuple[str, Callable[[Dict[str, Any]], Any]]] = [
        ("biology moved: connectivity on a quiet leg",
         lambda d: report(quiet, d).__setitem__("connectivity", "PERTURBED")),
        ("biology moved: normalization_stats on a moved leg",
         lambda d: report(moved[0], d).__setitem__("normalization_stats", "PERTURBED")),
        ("gate verdict moved: ok on a quiet leg",
         lambda d: report(quiet, d).__setitem__("ok", "PERTURBED")),
        ("leg field outside the report moved: sbml_input_source",
         lambda d: d["legs"][quiet].__setitem__("sbml_input_source", "PERTURBED")),
        ("an 8th leg's digest moves, so the leg count is wrong", digest),
        ("the two recordings of one digest disagree",
         lambda d: d["legs"][moved[0]].__setitem__("canonical_payload_sha256", "sha256:mismatch")),
        ("a refusal leg grew a gate report",
         lambda d: d["legs"][refusal].__setitem__("final_stage3_gate_report", {"ok": True})),
    ]

    g._with_c057_lineage_hashes(before, after)
    print(f"baseline accepted; {len(moved)} leg(s) move, quiet leg {quiet}")

    vacuous: List[str] = []
    for label, mutate in arms:
        perturbed = copy.deepcopy(after)
        mutate(perturbed)
        try:
            g._with_c057_lineage_hashes(before, perturbed)
        except AssertionError as exc:
            print(f"  RED    {label}: AssertionError({exc})")
            continue
        print(f"  GREEN  {label}: THE GUARD IS VACUOUS HERE")
        vacuous.append(label)

    print(f"\n{len(arms) - len(vacuous)}/{len(arms)} perturbations rejected")
    return 1 if vacuous else 0


if __name__ == "__main__":
    raise SystemExit(main())
