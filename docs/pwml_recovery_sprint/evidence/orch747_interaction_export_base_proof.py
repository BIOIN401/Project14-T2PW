"""G9 proof for the ORCH-747 interaction export: it FAILS on the base commit.

Merge gate 9 requires a claimed correction of pre-existing behaviour to carry a proof
that fails behaviourally at the base SHA and passes at the tip. Symbol absence is not
proof, so this does not look for a function or a tag name in source: it runs the PWML
exporter AS COMMITTED at the chosen revision over a payload holding one Activation, one
Inhibition and one untyped interaction with nucleic-acid endpoints, and inspects the
PWML it writes. The base emits every interaction (the untyped one included) as
``id`` / ``name`` / ``interaction-type`` with no element blocks, no ``pwi-id`` and no
visualization; the tip emits the two typed ones in PathBank's shape with a
visualization each and withholds the third.

The committed package is exported with ``git archive`` into a temporary directory and
imported there by a child interpreter whose ``PYTHONPATH`` names that tree, so the
working tree is never modified, stashed or checked out. **Both** legs -- base and tip --
run through the same child code path and each verifies that the ``t2pw`` it imported
resolves inside the tree it was asked to measure (TEST_MATRIX section 0 rule 10: a
printed path is not evidence, the resolved one is).

Usage::

    python docs/pwml_recovery_sprint/evidence/orch747_interaction_export_base_proof.py [--rev HEAD]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
PACKAGE_REL = "src/t2pw"
REFERENCE = REPO / "reference" / "PW000001.pwml"
PWML_INTERACTION_TYPES = {"Activation", "Inhibition"}

PAYLOAD: Dict[str, Any] = {
    "entities": {
        "species": [{"name": "Escherichia coli", "pathwhiz_id": 1}],
        "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
        "compounds": [],
        "proteins": [
            {"name": "AraC", "species": "Escherichia coli", "pathbank_protein_id": 301,
             "mapped_ids": {"uniprot": "P0A9E0"}},
        ],
        "protein_complexes": [
            {"name": "AraC complex", "species": "Escherichia coli", "components": ["AraC"],
             "generated": True, "generation_reason": "single_protein_pathwhiz_wrapper"},
        ],
        "nucleic_acids": [
            {"name": "A site", "class": "nucleic_acid"},
            {"name": "Pc promoter", "class": "nucleic_acid"},
            {"name": "araBAD promoter", "class": "nucleic_acid"},
        ],
    },
    "biological_states": [
        {"name": "cytosol", "species": "Escherichia coli", "subcellular_location": "cytosol",
         "compartment_canonical": "cytosol"},
    ],
    "processes": {
        "reactions": [],
        "transports": [],
        "interactions": [
            {"name": "AraC activates araBAD", "entity_1": "AraC complex",
             "relationship": "activation", "entity_2": "araBAD promoter",
             "biological_state": "cytosol", "evidence": "q"},
            {"name": "AraC represses Pc", "entity_1": "AraC complex",
             "relationship": "inhibition", "entity_2": "Pc promoter",
             "biological_state": "cytosol", "evidence": "q"},
            {"name": "AraC binds A site", "entity_1": "AraC complex",
             "relationship": "binding", "entity_2": "A site",
             "biological_state": "cytosol", "evidence": "q"},
        ],
    },
}


# --------------------------------------------------------------------------- child
def _emit(expect_tree: Path) -> int:
    """Run in the CHILD: export PAYLOAD with whatever ``t2pw`` resolves, print JSON."""

    from lxml import etree

    import t2pw
    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.validate import discover_structure_signature, repair_tree
    from t2pw.pwml.writer import DeterministicPwmlBuilder

    resolved = Path(t2pw.__file__).resolve()
    expected_pkg = (expect_tree / PACKAGE_REL).resolve()
    if expected_pkg not in resolved.parents:
        print(json.dumps({"error": "T2PW_MEASUREMENT_TREE_REFUSED", "t2pw_file": str(resolved),
                          "expected": str(expected_pkg), "sys_path": sys.path[:4]}))
        return 98

    ir, report = build_pwml_ir(PAYLOAD, strict_db=True, name_index=None)
    signature = discover_structure_signature(REFERENCE)
    args = argparse.Namespace(name="Generated Pathway", description="", subject="Metabolic",
                              pw_id="PW000000", height=1400, width=3200,
                              background_color="#FFFFFF", ref=str(REFERENCE))
    build = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args).build()
    repaired = repair_tree(etree.ElementTree(build.root), signature)
    xml = etree.tostring(repaired.getroot(), encoding="utf-8", xml_declaration=True,
                         pretty_print=True)
    print(json.dumps({
        "t2pw_file": str(resolved),
        "ir_error_codes": sorted({str(e.get("code")) for e in report["errors"]}),
        "withheld": getattr(build, "withheld_interactions", None),
        "xml": xml.decode("utf-8"),
    }))
    return 0


# -------------------------------------------------------------------------- parent
def _export_base_tree(rev: str) -> Path:
    """``src/t2pw`` as committed at ``rev``, unpacked into a fresh temp dir. Read-only."""

    out = subprocess.run(["git", "archive", "--format=tar", rev, PACKAGE_REL], cwd=REPO,
                         capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"git archive {rev} failed: {out.stderr.decode(errors='replace').strip()}")
    tree = Path(tempfile.mkdtemp(prefix="orch747_base_tree_"))
    with tarfile.open(fileobj=io.BytesIO(out.stdout)) as tar:
        tar.extractall(tree)
    return tree


def _run_child(tree: Path) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "src")
    env["T2PW_OFFLINE_CURATOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--emit-xml",
                          "--expect-tree", str(tree)],
                         cwd=tree, env=env, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit(f"child failed in {tree} (exit {out.returncode}):\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _inspect(xml: str) -> Dict[str, Any]:
    """The facts the verdict rests on, read from the PWML itself."""

    from lxml import etree

    root = etree.fromstring(xml.encode("utf-8"))
    interactions: List[Dict[str, Any]] = []
    for node in root.iter("interaction"):
        left = node.find("interaction-left-elements")
        right = node.find("interaction-right-elements")
        interactions.append({
            "type": node.findtext("interaction-type"),
            "children": [child.tag for child in node],
            "has_name": node.find("name") is not None,
            "has_pwi_id": node.find("pwi-id") is not None,
            "left_elements": len(left) if left is not None else None,
            "right_elements": len(right) if right is not None else None,
        })
    visualizations = []
    for viz in root.iter("interaction-visualization"):
        visualizations.append({
            "interaction_id": viz.findtext("interaction-id"),
            "sides": sorted((s.text or "") for s in viz.iter("side")),
            "member_tags": sorted({m.tag for container in viz if container.tag.startswith("interaction_")
                                   for m in container}),
        })
    return {"interactions": interactions, "interaction_visualizations": visualizations}


def _pathbank_shaped(row: Dict[str, Any]) -> bool:
    return (row["children"] == ["id", "interaction-type", "pwi-id", "interaction-left-elements",
                                "interaction-right-elements"]
            and row["left_elements"] == 1 and row["right_elements"] == 1
            and row["type"] in PWML_INTERACTION_TYPES)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rev", default="HEAD", help="the base revision to prove against")
    ap.add_argument("--emit-xml", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--expect-tree", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.emit_xml:
        return _emit(Path(args.expect_tree))

    sha = subprocess.run(["git", "rev-parse", args.rev], cwd=REPO, capture_output=True,
                         text=True).stdout.strip()
    base_tree = _export_base_tree(args.rev)
    base = _run_child(base_tree)
    tip = _run_child(REPO)
    base_facts = _inspect(base["xml"])
    tip_facts = _inspect(tip["xml"])

    # The tip's QA, applied to BOTH outputs: the base shape had nothing checking it.
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from t2pw.pwml.qa import run_pwml_qa
    base_qa = run_pwml_qa(base["xml"].encode("utf-8"))
    tip_qa = run_pwml_qa(tip["xml"].encode("utf-8"))

    print(f"BASE  {args.rev} = {sha}  ({PACKAGE_REL} archived to {base_tree})")
    print(f"  t2pw resolved at : {base['t2pw_file']}")
    print(f"  ir error codes   : {base['ir_error_codes']}")
    print(json.dumps(base_facts, indent=1))
    print(f"  withheld         : {base['withheld']}")
    print(f"  tip QA on base   : ok={base_qa['ok']} errors={len(base_qa['errors'])}")
    for err in base_qa["errors"]:
        print(f"    - {err}")
    print()
    print(f"TIP   working tree {REPO}")
    print(f"  t2pw resolved at : {tip['t2pw_file']}")
    print(f"  ir error codes   : {tip['ir_error_codes']}")
    print(json.dumps(tip_facts, indent=1))
    print(f"  withheld         : {json.dumps(tip['withheld'])}")
    print(f"  tip QA on tip    : ok={tip_qa['ok']} errors={tip_qa['errors']}")
    print()

    base_rows = base_facts["interactions"]
    base_broken = (
        len(base_rows) == 3                                   # the untyped one went out too
        and all(row["has_name"] for row in base_rows)         # a field PathBank never writes
        and not any(row["has_pwi_id"] for row in base_rows)
        and all(row["left_elements"] is None and row["right_elements"] is None for row in base_rows)
        and not base_facts["interaction_visualizations"]
        and base_qa["ok"] is False
    )
    tip_rows = tip_facts["interactions"]
    tip_viz = tip_facts["interaction_visualizations"]
    withheld = tip["withheld"] or []
    tip_correct = (
        [row["type"] for row in tip_rows] == ["Activation", "Inhibition"]
        and all(_pathbank_shaped(row) and not row["has_name"] and row["has_pwi_id"] for row in tip_rows)
        and len(tip_viz) == 2
        and all(viz["sides"] == ["Left", "Right"] for viz in tip_viz)
        and any("interaction-nucleic-acid-visualization" in viz["member_tags"] for viz in tip_viz)
        and len(withheld) == 1
        and withheld[0].get("reason") == "untyped_interaction"
        and withheld[0].get("raw_interaction_type") == "binding"
        and tip_qa["ok"] is True
        and not tip["ir_error_codes"]
    )

    print(f"base fails behaviourally : {base_broken}")
    print(f"tip passes               : {tip_correct}")
    if base_broken and tip_correct:
        print("VERDICT: G9 satisfied -- the base writes every interaction as id/name/type with no "
              "elements, no pwi-id and no visualization (and lets the untyped one through); "
              "the tip writes PathBank's shape for the two typed ones and withholds the third.")
        return 0
    print("VERDICT: NOT a valid G9 proof for this change.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
