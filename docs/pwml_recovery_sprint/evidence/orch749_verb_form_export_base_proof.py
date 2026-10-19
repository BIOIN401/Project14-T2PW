"""G9 proof for ORCH-749: a verb-form interaction payload reaches the PWML only at the tip.

Merge gate 9 requires a claimed correction of pre-existing behaviour to carry a proof
that fails behaviourally at the base SHA and passes at the tip. Symbol absence is not
proof, so this does not look for a table entry in source: it runs the PWML exporter AS
COMMITTED at the chosen revision over a payload whose relationships are spelled the way
the pipeline really spells them -- ``inhibits``, ``represses``, ``activates``,
``INHIBITS``, ``"competes with".``, ``blocks``, ``up-regulates`` -- beside four words
that carry no sign (``binds``, ``SAME_AS``, ``promoter_of``, ``interacts_with``), and
inspects the PWML it writes.

The base fails in one of two ways, both recorded: a tree before ORCH-747 writes every row
as a bare ``id`` / ``name`` / ``interaction-type`` carrying the raw word, with no element
blocks (PathWhiz cannot import it); a tree carrying ORCH-747 but not this change writes
no interaction at all and withholds all eleven as ``untyped_interaction``. Either way the
base exports **zero** PathBank-typed interactions from a payload of seven signed rows. The
tip exports exactly those seven, each with the type its word names, in PathBank's shape,
and withholds exactly the four unsigned rows -- it never reads a sign into a word that
does not carry one.

The committed package is exported with ``git archive`` into a temporary directory and
imported there by a child interpreter whose ``PYTHONPATH`` names that tree, so the
working tree is never modified, stashed or checked out. **Both** legs run through the
same child code path and each verifies that the ``t2pw`` it imported resolves inside the
tree it was asked to measure (TEST_MATRIX section 0 rule 10).

Usage::

    python docs/pwml_recovery_sprint/evidence/orch749_verb_form_export_base_proof.py [--rev HEAD]
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

#: (relationship as the pipeline writes it, the PathBank type its word names, or None)
ROWS = [
    ("inhibits", "Inhibition"),
    ("represses", "Inhibition"),
    ("activates", "Activation"),
    ("INHIBITS", "Inhibition"),
    ('"competes with".', "Inhibition"),
    ("blocks", "Inhibition"),
    ("up-regulates", "Activation"),
    ("binds", None),
    ("SAME_AS", None),
    ("promoter_of", None),
    ("interacts_with", None),
]
SIGNED = [t for _, t in ROWS if t]
UNSIGNED = [w for w, t in ROWS if not t]


def _payload() -> Dict[str, Any]:
    return {
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
            "nucleic_acids": [{"name": "araBAD promoter", "class": "nucleic_acid"}],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Escherichia coli", "subcellular_location": "cytosol",
             "compartment_canonical": "cytosol"},
        ],
        "processes": {
            "reactions": [],
            "transports": [],
            "interactions": [
                {"name": f"row {i}", "entity_1": "AraC complex", "relationship": word,
                 "entity_2": "araBAD promoter", "biological_state": "cytosol", "evidence": "q"}
                for i, (word, _) in enumerate(ROWS)
            ],
        },
    }


# --------------------------------------------------------------------------- child
def _emit(expect_tree: Path) -> int:
    """Run in the CHILD: export the payload with whatever ``t2pw`` resolves, print JSON."""

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

    ir, report = build_pwml_ir(_payload(), strict_db=True, name_index=None)
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
        "ir_rows": [
            {"interaction_type": row.get("interaction_type"),
             "export_status": row.get("export_status"),
             "withheld": row.get("withheld")}
            for row in ir["processes"]["interactions"]
        ],
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
    tree = Path(tempfile.mkdtemp(prefix="orch749_base_tree_"))
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


def _inspect(xml: str) -> List[Dict[str, Any]]:
    """Every ``<interaction>`` in the PWML: its type and whether it is PathBank-shaped."""

    from lxml import etree

    root = etree.fromstring(xml.encode("utf-8"))
    rows = []
    for node in root.iter("interaction"):
        left = node.find("interaction-left-elements")
        right = node.find("interaction-right-elements")
        rows.append({
            "type": node.findtext("interaction-type"),
            "pathbank_shaped": (
                [c.tag for c in node] == ["id", "interaction-type", "pwi-id",
                                          "interaction-left-elements", "interaction-right-elements"]
                and left is not None and len(left) == 1
                and right is not None and len(right) == 1
            ),
        })
    return rows


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
    base_rows = _inspect(base["xml"])
    tip_rows = _inspect(tip["xml"])

    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from t2pw.pwml.qa import run_pwml_qa
    base_qa = run_pwml_qa(base["xml"].encode("utf-8"))
    tip_qa = run_pwml_qa(tip["xml"].encode("utf-8"))

    def typed(rows: List[Dict[str, Any]]) -> List[str]:
        return [r["type"] for r in rows if r["pathbank_shaped"] and r["type"] in PWML_INTERACTION_TYPES]

    print(f"PAYLOAD  {len(ROWS)} interactions: {len(SIGNED)} signed verb forms, {len(UNSIGNED)} unsigned words")
    print(f"BASE  {args.rev} = {sha}  ({PACKAGE_REL} archived to {base_tree})")
    print(f"  t2pw resolved at           : {base['t2pw_file']}")
    print(f"  ir error codes             : {base['ir_error_codes']}")
    print(f"  <interaction> rows written : {len(base_rows)} -> types {[r['type'] for r in base_rows]}")
    print(f"  PathBank-typed and shaped  : {typed(base_rows)}")
    print(f"  withheld                   : {json.dumps(base['withheld'])}")
    print(f"  tip QA on base             : ok={base_qa['ok']} errors={len(base_qa['errors'])}")
    print()
    print(f"TIP   working tree {REPO}")
    print(f"  t2pw resolved at           : {tip['t2pw_file']}")
    print(f"  ir error codes             : {tip['ir_error_codes']}")
    print(f"  <interaction> rows written : {len(tip_rows)} -> types {[r['type'] for r in tip_rows]}")
    print(f"  PathBank-typed and shaped  : {typed(tip_rows)}")
    print(f"  withheld                   : {json.dumps(tip['withheld'])}")
    print(f"  tip QA on tip              : ok={tip_qa['ok']} errors={tip_qa['errors']}")
    print()

    # The base: zero PathBank-typed interactions, whichever way it fails.
    base_mode = None
    if not base_rows and base["withheld"] and len(base["withheld"]) == len(ROWS):
        base_mode = "withheld_everything (ORCH-747 shape, nouns only)"
    elif len(base_rows) == len(ROWS) and not any(r["pathbank_shaped"] for r in base_rows):
        base_mode = "raw_word_written_without_element_blocks (pre-ORCH-747 shape)"
    base_broken = not typed(base_rows) and base_mode is not None

    tip_withheld = tip["withheld"] or []
    tip_correct = (
        typed(tip_rows) == SIGNED                                    # each word's own sign, in order
        and len(tip_rows) == len(SIGNED)                             # and nothing else went out
        and sorted(w.get("raw_interaction_type") for w in tip_withheld) == sorted(UNSIGNED)
        and all(w.get("reason") == "untyped_interaction" for w in tip_withheld)
        and all(                                                     # unsigned words kept as written
            row["interaction_type"] == word
            for row, (word, t) in zip(tip["ir_rows"], ROWS) if t is None
        )
        and tip_qa["ok"] is True
        and not tip["ir_error_codes"]
    )

    print(f"base failure mode        : {base_mode}")
    print(f"base fails behaviourally : {base_broken}")
    print(f"tip passes               : {tip_correct}")
    if base_broken and tip_correct:
        print("VERDICT: G9 satisfied -- the base exports zero PathBank-typed interactions from a "
              f"payload of {len(SIGNED)} signed verb forms; the tip exports exactly those "
              f"{len(SIGNED)}, each with the type its word names, and withholds exactly the "
              f"{len(UNSIGNED)} unsigned words as untyped_interaction.")
        return 0
    print("VERDICT: NOT a valid G9 proof for this change.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
