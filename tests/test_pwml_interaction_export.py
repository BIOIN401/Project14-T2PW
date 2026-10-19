"""ORCH-747 -- interaction export in the PWML writer.

**New acceptance tests** (merge rule G9, new-capability form) for the pieces that
make a PathWhiz-importable ``<interaction>``: the element block shape against a
PathBank reference export, the per-type member id column, the Activation arrow
versus the Inhibition T-bar, the untyped-interaction side file, and the QA check
that fires on a malformed interaction. The base-SHA behavioural proof is
``docs/pwml_recovery_sprint/evidence/orch747_interaction_export_base_proof.py``.

Nothing here resolves or invents biology: the fixture declares its endpoints and
its signs, and the exporter only plumbs them through.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helpers_prefreeze import prefrozen  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir  # noqa: E402
from t2pw.pwml.qa import run_pwml_qa  # noqa: E402
from t2pw.pwml.validate import (  # noqa: E402
    discover_structure_signature,
    parse_xml,
    repair_tree,
    validate_generated_tree,
)
from t2pw.pwml.writer import (  # noqa: E402
    DeterministicPwmlBuilder,
    build_pwml_pipeline_arg_parser,
    run_pwml_pipeline_export,
)

REFERENCE = ROOT / "reference" / "PW000001.pwml"
PATHBANK_EXPORT_WITH_INTERACTIONS = ROOT / "reference" / "PW741182.pwml"


def _payload() -> dict:
    """One Activation, one Inhibition, one untyped and one bare-protein interaction.

    Every exportable endpoint is a nucleic acid or a protein complex: the
    nucleic-acid member container is the one no reference file demonstrates.
    """
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [],
            "proteins": [
                {"name": "AraC", "species": "Escherichia coli", "pathbank_protein_id": 301, "mapped_ids": {"uniprot": "P0A9E0"}},
                {"name": "AraR", "species": "Escherichia coli", "pathbank_protein_id": 302, "mapped_ids": {"uniprot": "P0A9E1"}},
            ],
            "protein_complexes": [
                {
                    "name": "AraC complex",
                    "species": "Escherichia coli",
                    "components": ["AraC"],
                    "generated": True,
                    "generation_reason": "single_protein_pathwhiz_wrapper",
                }
            ],
            "nucleic_acids": [
                {"name": "A site", "class": "nucleic_acid"},
                {"name": "Pc promoter", "class": "nucleic_acid"},
                {"name": "araBAD promoter", "class": "nucleic_acid"},
            ],
            "element_collections": [{"name": "ara regulatory elements", "class": "element_collection"}],
        },
        "biological_states": [
            {
                "name": "cytosol",
                "species": "Escherichia coli",
                "subcellular_location": "cytosol",
                "compartment_canonical": "cytosol",
            },
        ],
        "processes": {
            "reactions": [],
            "transports": [],
            "interactions": [
                {
                    "name": "AraC activates araBAD",
                    "entity_1": "AraC complex",
                    "relationship": "activation",
                    "entity_2": "araBAD promoter",
                    "biological_state": "cytosol",
                    "evidence": "q",
                },
                {
                    "name": "AraC represses Pc",
                    "entity_1": "AraC complex",
                    "relationship": "inhibition",
                    "entity_2": "Pc promoter",
                    "biological_state": "cytosol",
                    "evidence": "q",
                },
                {
                    "name": "AraC binds A site",
                    "entity_1": "AraC complex",
                    "relationship": "binding",
                    "entity_2": "A site",
                    "biological_state": "cytosol",
                    "evidence": "q",
                },
                {
                    "name": "AraR represses the A site",
                    "entity_1": "AraR",
                    "relationship": "inhibition",
                    "entity_2": "A site",
                    "biological_state": "cytosol",
                    "evidence": "q",
                },
            ],
        },
    }


def _writer_args() -> SimpleNamespace:
    return SimpleNamespace(
        name="Generated Pathway",
        description="",
        subject="Metabolic",
        pw_id="PW000000",
        height=1400,
        width=3200,
        background_color="#FFFFFF",
        ref=str(REFERENCE),
    )


def _export(payload: dict, **ir_kwargs: object):
    """payload -> IR -> writer -> repaired tree, the production sequence."""
    ir, report = build_pwml_ir(payload, strict_db=True, **ir_kwargs)
    assert report["errors"] == [], report["errors"]
    validation = validate_pwml_ir(ir)
    assert validation["ok"], validation["errors"]
    signature = discover_structure_signature(REFERENCE)
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=_writer_args())
    build = builder.build()
    repaired = repair_tree(etree.ElementTree(build.root), signature)
    assert validate_generated_tree(repaired, signature)["ok"]
    return ir, build, repaired.getroot()


def _interaction_by_type(root: etree._Element, interaction_type: str) -> etree._Element:
    return next(node for node in root.iter("interaction") if node.findtext("interaction-type") == interaction_type)


def _edges_by_id(root: etree._Element) -> dict[str, etree._Element]:
    return {edge.findtext("id"): edge for edge in root.iter("edge")}


def _viz_for(root: etree._Element, interaction: etree._Element) -> etree._Element:
    iid = interaction.findtext("id")
    return next(v for v in root.iter("interaction-visualization") if v.findtext("interaction-id") == iid)


# ── 1. element block shape, against a PathBank export ──────────────────────


def test_interaction_element_block_matches_the_pathbank_reference_shape() -> None:
    _ir, _build, root = _export(_payload())
    ours = _interaction_by_type(root, "Activation")

    # PathBank exports carry undeclared ``option:`` prefixes; the project parser recovers.
    reference = parse_xml(PATHBANK_EXPORT_WITH_INTERACTIONS).getroot()
    theirs = next(iter(reference.iter("interaction")))

    # Same children, same order, as every PathBank interaction. No <name>.
    assert [c.tag for c in ours] == [c.tag for c in theirs]
    assert [c.tag for c in ours] == [
        "id",
        "interaction-type",
        "pwi-id",
        "interaction-left-elements",
        "interaction-right-elements",
    ]
    for side in ("interaction-left-elements", "interaction-right-elements"):
        ours_side = ours.find(side)
        theirs_side = theirs.find(side)
        assert len(ours_side) == 1 == len(theirs_side)
        assert ours_side[0].tag == theirs_side[0].tag == side[:-1]
        assert [c.tag for c in ours_side[0]] == [c.tag for c in theirs_side[0]]
        assert [c.tag for c in ours_side[0]] == ["id", "element-id", "element-type", "stoichiometry"]
    assert ours.find("id").get("type") == "integer"
    assert ours.findtext("pwi-id") == f"PW_I{int(ours.findtext('id')):06d}"

    # The endpoints point at the declared entities, with PathWhiz's type names.
    complex_id = next(pc.findtext("id") for pc in root.iter("protein-complex") if pc.findtext("name") == "AraC complex")
    promoter_id = next(na.findtext("id") for na in root.iter("nucleic-acid") if na.findtext("name") == "araBAD promoter")
    left = ours.find("interaction-left-elements")[0]
    right = ours.find("interaction-right-elements")[0]
    assert (left.findtext("element-type"), left.findtext("element-id")) == ("ProteinComplex", complex_id)
    assert (right.findtext("element-type"), right.findtext("element-id")) == ("NucleicAcid", promoter_id)


# ── 2. the member container id column differs by type ──────────────────────


def test_interaction_visualization_uses_the_per_type_id_column() -> None:
    payload = _payload()
    payload["entities"]["compounds"] = [
        {"name": "L-arabinose", "pathbank_compound_id": 101, "mapped_ids": {"chebi": "CHEBI:17535"}}
    ]
    payload["processes"]["interactions"] = [
        {
            "name": "arabinose activates AraC",
            "entity_1": "L-arabinose",
            "relationship": "activation",
            "entity_2": "AraC complex",
            "biological_state": "cytosol",
            "evidence": "q",
        },
        {
            "name": "AraC represses Pc",
            "entity_1": "AraC complex",
            "relationship": "inhibition",
            "entity_2": "Pc promoter",
            "biological_state": "cytosol",
            "evidence": "q",
        },
        {
            "name": "AraC activates the elements",
            "entity_1": "AraC complex",
            "relationship": "activation",
            "entity_2": "ara regulatory elements",
            "biological_state": "cytosol",
            "evidence": "q",
        },
    ]
    _ir, _build, root = _export(prefrozen(payload))

    def member(container: str, side: str) -> etree._Element:
        for viz in root.iter("interaction-visualization"):
            for item in viz.find(container):
                if item.findtext("side") == side:
                    return item
        raise AssertionError(f"no {side} member in {container}")

    locations = {
        "compound-location": {loc.findtext("id") for loc in root.iter("compound-location")},
        "nucleic-acid-location": {loc.findtext("id") for loc in root.iter("nucleic-acid-location")},
        "element-collection-location": {loc.findtext("id") for loc in root.iter("element-collection-location")},
        "protein-complex-visualization": {v.findtext("id") for v in root.iter("protein-complex-visualization")},
    }
    cases = [
        ("interaction_compound_visualizations", "interaction-compound-visualization", "compound-location-id", "Left"),
        ("interaction_protein_complex_visualizations", "interaction-protein-complex-visualization", "protein-complex-visualization-id", "Right"),
        ("interaction_nucleic_acid_visualizations", "interaction-nucleic-acid-visualization", "nucleic-acid-location-id", "Right"),
        ("interaction_element_collection_visualizations", "interaction-element-collection-visualization", "element-collection-location-id", "Right"),
    ]
    for container, item_tag, id_column, side in cases:
        item = member(container, side)
        assert item.tag == item_tag, container
        assert [c.tag for c in item] == ["id", id_column, "edge-id", "side"], container
        assert item.find(id_column).get("type") == "integer"
        assert item.findtext(id_column) in locations[id_column[: -len("-id")]], container


# ── 3. Activation arrow versus Inhibition T-bar ─────────────────────────────


def test_right_edge_carries_the_sign_as_pathbank_stores_it() -> None:
    _ir, _build, root = _export(_payload())
    edges = _edges_by_id(root)

    def sides(interaction_type: str) -> dict[str, etree._Element]:
        viz = _viz_for(root, _interaction_by_type(root, interaction_type))
        out = {}
        for edge_ref in viz.iter("edge-id"):
            out[edge_ref.getparent().findtext("side")] = edges[edge_ref.text]
        assert set(out) == {"Left", "Right"}
        return out

    activation = sides("Activation")
    inhibition = sides("Inhibition")

    # Sampled 40 of each in PathBank: the sign is ONLY in the right edge's options.
    activation_opts = json.loads(activation["Right"].findtext("options"))
    assert activation_opts["start_arrow"] is True
    assert activation_opts["start_flat_arrow"] is False
    assert activation_opts["start_arrow_path"].startswith("M ")
    assert activation_opts["start_flat_arrow_path"] is None

    inhibition_opts = json.loads(inhibition["Right"].findtext("options"))
    assert inhibition_opts["start_arrow"] is False
    assert inhibition_opts["start_flat_arrow"] is True
    assert inhibition_opts["start_arrow_path"] is None
    assert inhibition_opts["start_flat_arrow_path"].startswith("M ")

    # The left edge is the undecorated, hidden reverse of the same line.
    for pair in (activation, inhibition):
        assert json.loads(pair["Left"].findtext("options")) == {}
        assert pair["Left"].findtext("hidden") == "true"
        assert pair["Right"].findtext("hidden") == "false"
        left_path = pair["Left"].findtext("path").split()
        right_path = pair["Right"].findtext("path").split()
        assert left_path[0] == "M" + right_path[-2] and left_path[1] == right_path[-1]
    # The arrow tip sits on the right element: the T-bar runs through the path's start.
    rx, ry = inhibition["Right"].findtext("path").split()[0][1:], inhibition["Right"].findtext("path").split()[1]
    assert f"L {rx} {ry} " in inhibition_opts["start_flat_arrow_path"]
    assert activation["Right"].findtext("visualization-template-id") == "149"
    assert inhibition["Right"].findtext("visualization-template-id") == "148"


# ── 4. untyped and undrawable interactions are withheld, not dropped or guessed ──


def test_untyped_and_bare_protein_interactions_are_withheld_with_a_reason() -> None:
    ir, build, root = _export(_payload())

    exported = {node.findtext("interaction-type") for node in root.iter("interaction")}
    assert exported == {"Activation", "Inhibition"}
    assert len(list(root.iter("interaction"))) == 2
    assert len(list(root.iter("interaction-visualization"))) == 2

    # Still in the IR, with the verdict on the row, never silently removed.
    by_name = {row["name"]: row for row in ir["processes"]["interactions"]}
    assert by_name["AraC binds A site"]["export_status"] == "withheld"
    assert by_name["AraC binds A site"]["interaction_type"] == "binding"
    assert by_name["AraR represses the A site"]["export_status"] == "withheld"
    assert by_name["AraC activates araBAD"]["export_status"] == "exportable"

    withheld = {row["name"]: row for row in build.withheld_interactions}
    assert withheld["AraC binds A site"]["reason"] == "untyped_interaction"
    assert withheld["AraC binds A site"]["raw_interaction_type"] == "binding"
    assert withheld["AraC binds A site"]["left"] == "AraC complex"
    assert withheld["AraC binds A site"]["right"] == "A site"
    assert withheld["AraC binds A site"]["pointer"] == "/processes/interactions/2"
    assert withheld["AraR represses the A site"]["reason"] == "bare_protein_endpoint"
    assert len(withheld) == 2

    # No orphan geometry for a withheld row: only the two drawn pairs have edges.
    assert len(ir["edges"]) == 4


def test_cli_export_writes_the_withheld_side_file_next_to_the_pwml(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    out_dir = tmp_path / "out"
    args = build_pwml_pipeline_arg_parser().parse_args(
        ["--in", str(input_path), "--out-dir", str(out_dir), "--ref", str(REFERENCE), "--non-strict-db"]
    )
    summary = run_pwml_pipeline_export(args)

    assert summary.get("pwml_file"), summary
    assert summary["pwml_qa_ok"] is True
    assert summary["withheld_interaction_count"] == 2
    side_file = Path(summary["withheld_interactions"])
    assert side_file == out_dir / "pathway.withheld_interactions.json"
    assert side_file.exists()
    record = json.loads(side_file.read_text(encoding="utf-8"))
    assert record["pwml_file"] == "pathway.pwml"
    assert record["count"] == 2
    assert {row["reason"] for row in record["interactions"]} == {"untyped_interaction", "bare_protein_endpoint"}
    assert "binding" in {row["raw_interaction_type"] for row in record["interactions"]}

    pwml = etree.parse(str(out_dir / "pathway.pwml")).getroot()
    assert {node.findtext("interaction-type") for node in pwml.iter("interaction")} == {"Activation", "Inhibition"}
    ir_report = json.loads((out_dir / "pwml_ir_report.json").read_text(encoding="utf-8"))
    assert sum(1 for w in ir_report["warnings"] if w["code"] == "interaction_withheld_from_pwml") == 2


def test_cli_export_writes_no_side_file_when_nothing_is_withheld(tmp_path: Path) -> None:
    payload = _payload()
    payload["processes"]["interactions"] = payload["processes"]["interactions"][:2]
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "out"
    args = build_pwml_pipeline_arg_parser().parse_args(
        ["--in", str(input_path), "--out-dir", str(out_dir), "--ref", str(REFERENCE), "--non-strict-db"]
    )
    summary = run_pwml_pipeline_export(args)
    assert summary["withheld_interaction_count"] == 0
    assert summary["withheld_interactions"] is None
    assert not (out_dir / "pathway.withheld_interactions.json").exists()


# ── 5. the QA check fails loudly on a malformed interaction ─────────────────


def _qa_doc(interactions_xml: str, visualizations_xml: str = "") -> bytes:
    return (
        "<pathway-visualization-context><pathway-visualization>"
        f"<interactions>{interactions_xml}</interactions>"
        f"<interaction-visualizations>{visualizations_xml}</interaction-visualizations>"
        "</pathway-visualization></pathway-visualization-context>"
    ).encode("utf-8")


def test_qa_rejects_the_shape_the_writer_used_to_emit() -> None:
    # id + name + interaction-type only: what reached PathWhiz before ORCH-747.
    report = run_pwml_qa(_qa_doc('<interaction><id type="integer">7</id><name>x</name><interaction-type>binding</interaction-type></interaction>'))
    assert report["ok"] is False
    assert report["stats"]["interactions"] == 1
    messages = "\n".join(report["errors"])
    assert "interaction-type 'binding'" in messages
    assert "<name>" in messages
    assert "0 interaction-left element(s)" in messages
    assert "0 interaction-right element(s)" in messages
    assert "no interaction-visualization" in messages
    assert len(report["errors"]) == 5


def test_qa_requires_exactly_one_element_per_side_and_both_visualization_sides() -> None:
    element = '<interaction-left-element><id type="integer">1</id></interaction-left-element>'
    two_left = (
        '<interaction><id type="integer">7</id><interaction-type>Activation</interaction-type><pwi-id>PW_I000007</pwi-id>'
        f"<interaction-left-elements>{element}{element}</interaction-left-elements>"
        '<interaction-right-elements><interaction-right-element><id type="integer">2</id></interaction-right-element></interaction-right-elements>'
        "</interaction>"
    )
    one_sided_viz = (
        '<interaction-visualization><id type="integer">9</id><interaction-id type="integer">7</interaction-id>'
        "<interaction_nucleic_acid_visualizations><interaction-nucleic-acid-visualization>"
        '<id type="integer">10</id><nucleic-acid-location-id type="integer">3</nucleic-acid-location-id>'
        '<edge-id type="integer">4</edge-id><side>Left</side>'
        "</interaction-nucleic-acid-visualization></interaction_nucleic_acid_visualizations>"
        "</interaction-visualization>"
    )
    report = run_pwml_qa(_qa_doc(two_left, one_sided_viz))
    assert report["ok"] is False
    messages = "\n".join(report["errors"])
    assert "2 interaction-left element(s)" in messages
    assert "sides ['Left']" in messages
    assert len(report["errors"]) == 2


def test_qa_accepts_the_writer_output_and_counts_it() -> None:
    _ir, _build, root = _export(_payload())
    report = run_pwml_qa(etree.tostring(root, encoding="utf-8"))
    assert report["ok"] is True, report["errors"]
    assert report["stats"]["interactions"] == 2


# ── 6. the IR keeps the rows it withholds and validates the ones it draws ──


def test_ir_interaction_members_are_registered_for_visualization_validation() -> None:
    ir, _report = build_pwml_ir(_payload(), strict_db=True)
    validation = validate_pwml_ir(ir)
    assert validation["ok"], validation["errors"]
    drawn = [v for v in ir["process_visualizations"] if v["type"] == "interaction_visualization"]
    assert [v["process_key"] for v in drawn] == ["int_1", "int_2"]
    for viz in drawn:
        assert [m["role"] for m in viz["members"]] == ["left", "right"]
        assert viz["members"][0]["process_member_key"] == f"{viz['process_key']}_left"
        assert viz["members"][1]["process_member_key"] == f"{viz['process_key']}_right"
        assert all(m["edge_key"] and m["location_key"] for m in viz["members"])

    # A member key that points nowhere is still refused.
    broken = deepcopy(ir)
    broken["process_visualizations"][0]["members"][0]["process_member_key"] = "int_99_left"
    assert "visualization_unknown_member" in {e["code"] for e in validate_pwml_ir(broken)["errors"]}
