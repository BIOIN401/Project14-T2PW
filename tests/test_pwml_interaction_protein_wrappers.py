"""ORCH-754b / D-104 -- protein endpoints of interactions reach the PWML as single-protein complexes.

**NEW ACCEPTANCE TESTS** (merge rule G9, new-capability form). Before this card every
interaction with a protein endpoint was withheld as ``bare_protein_endpoint``, because
PathWhiz has no bare-protein interaction member. D-104 rules that such a protein is
wrapped exactly as a reaction enzyme is, by the existing Stage-6 wrapper path, before
the freeze, and never by the exporter. The behavioural base-vs-tip record (the base
withholds the fixture's protein-endpoint rows; the tip exports them; and an input
with no protein-endpoint interaction is byte-identical) is
``docs/pwml_recovery_sprint/evidence/orch754b_wrapper_base_proof.py``.

Nothing here invents biology: the fixture declares its proteins, their identities and
their signs. The wrapper adds no component the payload did not name, and a protein
without species or identity is left unwrapped and withheld with the reason.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.map_ids import MappingCache, _rewrite_reaction_protein_enzymes_to_complexes  # noqa: E402
from t2pw.pipeline.draft_graph import build_draft_graph  # noqa: E402
from t2pw.pipeline.reaction_summary import generate_reaction_summary  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir, validate_required_pwml_contract  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature, parse_xml, repair_tree, validate_generated_tree  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402
from t2pw.rag.research_report import build_citation_report  # noqa: E402

REFERENCE = ROOT / "reference" / "PW000001.pwml"
#: A real PathBank export whose interaction 15843 is Activation between two
#: single-protein complexes (Ephrin-A1 -> Ephrin type-A receptor 2).
PATHBANK_PROTEIN_INTERACTION = ROOT / "reference" / "PW687775.pwml"


def _protein(name: str, uniprot: str = "", species: str = "Homo sapiens", pathbank_id: int = 0) -> dict:
    row: dict = {"name": name}
    if species:
        row.update(species=species, organism=species, species_id=1)
    ids = {}
    if uniprot:
        ids["uniprot"] = uniprot
    if pathbank_id:
        row["pathbank_protein_id"] = pathbank_id
        ids["pathbank_protein_id"] = str(pathbank_id)
    row["mapped_ids"] = ids
    return row


def _interaction(name: str, left: str, word: str, right: str) -> dict:
    return {
        "name": name,
        "entity_1": left,
        "relationship": word,
        "entity_2": right,
        "biological_state": "mitochondrion",
        "evidence": f"quote for {name}",
    }


def _payload() -> dict:
    """The PMC12180156 shape: SFXN4 and ALAS2 are mapped proteins, both endpoints."""
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "mitochondrion", "pathwhiz_id": 2}],
            "compounds": [],
            "proteins": [
                _protein("SFXN4", "Q6P4A7", pathbank_id=3910),
                _protein("ALAS2", "P22557", pathbank_id=17),
                _protein("FtsH"),  # species, no UniProt/DrugBank identity
                _protein("Orphan", "P00001", species=""),  # identity, no species
            ],
            "protein_complexes": [],
            "nucleic_acids": [{"name": "ALAS2 mRNA", "class": "nucleic_acid"}],
        },
        "biological_states": [
            {
                "name": "mitochondrion",
                "species": "Homo sapiens",
                "subcellular_location": "mitochondrion",
                "compartment_canonical": "mitochondrion",
            }
        ],
        "processes": {
            "reactions": [],
            "transports": [],
            "interactions": [
                _interaction("SFXN4 inhibits ALAS2", "SFXN4", "inhibits", "ALAS2"),
                _interaction("ALAS2 activates its mRNA", "ALAS2", "activates", "ALAS2 mRNA"),
                _interaction("SFXN4 binds ALAS2", "SFXN4", "binds", "ALAS2"),
                _interaction("FtsH inhibits ALAS2", "FtsH", "inhibits", "ALAS2"),
                _interaction("Orphan inhibits ALAS2 mRNA", "Orphan", "inhibits", "ALAS2 mRNA"),
            ],
        },
    }


def _stage6(payload: dict, db: object = None) -> dict:
    """The Stage-6 wrapper step, offline: no DB, no cache file."""
    return _rewrite_reaction_protein_enzymes_to_complexes(
        payload,
        db=db,  # type: ignore[arg-type]
        cache=MappingCache(Path("unused.json"), enabled=False),
        global_organism="Homo sapiens",
    )


def _export(payload: dict):
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert report["errors"] == [], report["errors"]
    assert validate_pwml_ir(ir)["ok"]
    signature = discover_structure_signature(REFERENCE)
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF", ref=str(REFERENCE))
    build = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args).build()
    repaired = repair_tree(etree.ElementTree(build.root), signature)
    assert validate_generated_tree(repaired, signature)["ok"]
    return ir, build, repaired.getroot()


# ── Stage 6 builds the wrapper, with the enzyme machinery ────────────────────


def test_signed_protein_endpoints_are_rewritten_to_single_protein_wrappers() -> None:
    payload = _payload()
    report = _stage6(payload)
    rows = payload["processes"]["interactions"]
    assert (rows[0]["entity_1"], rows[0]["entity_2"]) == ("SFXN4 complex", "ALAS2 complex")
    assert rows[1]["entity_1"] == "ALAS2 complex" and rows[1]["entity_2"] == "ALAS2 mRNA"
    # The relationship word and the evidence are untouched: wrapping is representation only.
    assert [r["relationship"] for r in rows] == ["inhibits", "activates", "binds", "inhibits", "inhibits"]
    assert rows[0]["evidence"] == "quote for SFXN4 inhibits ALAS2"

    complexes = {row["name"]: row for row in payload["entities"]["protein_complexes"]}
    # One wrapper per protein, reused across rows; nothing for an unsigned or unwrappable row.
    assert set(complexes) == {"SFXN4 complex", "ALAS2 complex"}
    for name, protein in (("SFXN4 complex", "SFXN4"), ("ALAS2 complex", "ALAS2")):
        wrapper = complexes[name]
        assert wrapper["generated"] is True
        assert wrapper["generation_reason"] == "single_protein_pathwhiz_wrapper"
        assert [c["name"] for c in wrapper["components"]] == [protein]
    # The proteins stay declared: the wrapper refers to them, it does not replace them.
    assert {p["name"] for p in payload["entities"]["proteins"]} == {"SFXN4", "ALAS2", "FtsH", "Orphan"}

    assert report["summary"]["interaction_protein_endpoints_rewritten_to_complexes"] == 3
    assert report["summary"]["interaction_protein_endpoints_skipped_invalid_component"] == 2
    rewritten = [a for a in report["actions"] if a["type"] == "interaction_protein_endpoint_rewritten_to_complex"]
    assert {(a["json_pointer"], a["protein"], a["protein_complex"]) for a in rewritten} == {
        ("/processes/interactions/0/entity_1", "SFXN4", "SFXN4 complex"),
        ("/processes/interactions/0/entity_2", "ALAS2", "ALAS2 complex"),
        ("/processes/interactions/1/entity_1", "ALAS2", "ALAS2 complex"),
    }


def test_unsigned_and_unwrappable_rows_are_left_exactly_as_written() -> None:
    payload = _payload()
    original = deepcopy(payload["processes"]["interactions"])
    report = _stage6(payload)
    rows = payload["processes"]["interactions"]
    assert rows[2] == original[2]  # binds: no sign, no wrapper
    # FtsH has no identity, so NEITHER endpoint of its row is wrapped: a wrapper for
    # ALAS2 alone would not make the row drawable.
    assert rows[3] == original[3]
    assert rows[4] == original[4]
    skipped = {a["json_pointer"]: a["missing"] for a in report["actions"]
               if a["type"] == "interaction_protein_endpoint_wrapper_skipped_invalid_component"}
    assert skipped == {
        "/processes/interactions/3": {"FtsH": ["uniprot_or_drugbank"]},
        "/processes/interactions/4": {"Orphan": ["species"]},
    }


def test_an_interaction_free_payload_gets_no_new_summary_keys() -> None:
    payload = _payload()
    payload["processes"]["interactions"] = []
    report = _stage6(payload)
    assert not any(key.startswith("interaction_") for key in report["summary"])
    assert payload["entities"]["protein_complexes"] == []


def test_an_enzyme_and_an_interaction_endpoint_share_one_wrapper() -> None:
    payload = _payload()
    payload["processes"]["reactions"] = [
        {"name": "ALAS2 step", "inputs": [], "outputs": [], "enzymes": [{"protein": "ALAS2"}]}
    ]
    _stage6(payload)
    names = [row["name"] for row in payload["entities"]["protein_complexes"]]
    assert names.count("ALAS2 complex") == 1
    assert payload["processes"]["reactions"][0]["enzymes"][0]["entity"] == "ALAS2 complex"
    assert payload["processes"]["interactions"][0]["entity_2"] == "ALAS2 complex"


def test_an_existing_own_wrapper_is_reused_as_it_stands() -> None:
    """A wrapper an earlier pass built (here: its database provenance) is not re-resolved."""
    payload = _payload()
    existing = {"name": "ALAS2 complex", "species": "Homo sapiens", "species_id": 1, "components": [{"name": "ALAS2"}],
                "generated": True, "generation_reason": "single_protein_pathwhiz_wrapper",
                "mapping_meta": {"provider": "PathBankDB", "chosen_rule": "novel_complex_from_resolved_components"}}
    payload["entities"]["protein_complexes"] = [deepcopy(existing)]
    _stage6(payload)
    assert payload["processes"]["interactions"][0]["entity_2"] == "ALAS2 complex"
    assert payload["entities"]["protein_complexes"][0] == existing


def test_a_declared_complex_that_happens_to_carry_the_wrapper_name_is_never_borrowed() -> None:
    payload = _payload()
    declared = {"name": "ALAS2 complex", "species": "Homo sapiens", "components": [{"name": "ALAS2"}, {"name": "SFXN4"}]}
    payload["entities"]["protein_complexes"] = [deepcopy(declared)]
    payload["processes"]["interactions"] = payload["processes"]["interactions"][:1]
    report = _stage6(payload)
    assert payload["processes"]["interactions"][0]["entity_2"] == "ALAS2"
    assert payload["entities"]["protein_complexes"] == [declared]
    skipped = [a for a in report["actions"] if a["type"] == "interaction_protein_endpoint_wrapper_skipped_invalid_component"]
    assert skipped[0]["missing"] == {"ALAS2": ["free_wrapper_name"]}


class _SupersetDb:
    """PathBank answering with a two-protein complex that merely contains ALAS2."""

    last_error = ""

    def available(self) -> bool:
        return True

    def map_enzyme_protein_to_complex(self, protein_row: dict, species: str) -> dict:
        name = protein_row["name"]
        other = {"name": "STRANGER", "pathbank_protein_id": 99, "stoichiometry": 1,
                 "mapped_ids": {"uniprot": "P99999"}}
        mine = {"name": name, "pathbank_protein_id": protein_row.get("pathbank_protein_id"), "stoichiometry": 1,
                "mapped_ids": dict(protein_row.get("mapped_ids") or {})}
        return {"status": "mapped", "provider": "PathBankDB", "source": "db", "name": f"{name}-STRANGER assembly",
                "pathbank_complex_id": 7, "pathbank_protein_complex_id": 7, "species_id": 1,
                "components": [mine, other], "confidence": 0.9, "chosen_rule": "enzyme_component_species",
                "candidates": [], "issues": [], "resolution": {"status": "matched", "order_step": "enzyme_component_species"}}


def test_the_enzyme_superset_refusal_applies_to_interaction_endpoints() -> None:
    """The F-116 rule is inherited, not re-implemented: a database complex whose other
    components the interaction never named is refused, and the one-protein wrapper stands."""
    payload = _payload()
    payload["processes"]["interactions"] = payload["processes"]["interactions"][:1]
    report = _stage6(payload, db=_SupersetDb())
    assert payload["processes"]["interactions"][0]["entity_2"] == "ALAS2 complex"
    wrapper = next(r for r in payload["entities"]["protein_complexes"] if r["name"] == "ALAS2 complex")
    assert [c["name"] for c in wrapper["components"]] == ["ALAS2"]
    assert "STRANGER" not in str(payload["entities"]["protein_complexes"])
    assert report["summary"]["reaction_enzyme_complex_superset_promotions_refused"] == 2


# ── the wrappers pass the generated-complex gates, untouched ─────────────────


def test_every_wrapper_satisfies_the_generated_complex_species_and_identity_gates() -> None:
    payload = _payload()
    _stage6(payload)
    contract = validate_required_pwml_contract(payload, strict_db=False)
    codes = {e["code"] for e in contract["errors"]}
    assert not {c for c in codes if c.startswith("generated_complex_") or c.startswith("protein_complex_")}, codes

    # Non-vacuous: the same gate, on the wrapper Stage 6 refused to build, does fire.
    payload["entities"]["protein_complexes"].append(
        {"name": "FtsH complex", "species": "Homo sapiens", "components": ["FtsH"],
         "generated": True, "generation_reason": "single_protein_pathwhiz_wrapper"}
    )
    codes = {e["code"] for e in validate_required_pwml_contract(payload, strict_db=False)["errors"]}
    assert "generated_complex_component_missing_external_identity" in codes


# ── the exporter draws them, and withholds the rest with a precise reason ────


def test_wrapped_rows_export_and_the_rest_are_withheld_with_their_reason() -> None:
    payload = _payload()
    _stage6(payload)
    ir, build, root = _export(payload)

    statuses = [row["export_status"] for row in ir["processes"]["interactions"]]
    assert statuses == ["exportable", "exportable", "withheld", "withheld", "withheld"]
    withheld = {row["name"]: row for row in build.withheld_interactions}
    assert withheld["SFXN4 binds ALAS2"]["reason"] == "untyped_interaction"
    assert withheld["FtsH inhibits ALAS2"]["reason"] == "protein_endpoint_missing_external_identity"
    assert withheld["FtsH inhibits ALAS2"]["unwrappable_proteins"] == [{"name": "FtsH", "missing": ["external_identity"]}]
    assert withheld["Orphan inhibits ALAS2 mRNA"]["reason"] == "protein_endpoint_missing_species"
    assert {n.findtext("interaction-type") for n in root.iter("interaction")} == {"Inhibition", "Activation"}
    assert len(list(root.iter("interaction"))) == 2


def test_both_missing_names_both_and_a_wrappable_bare_protein_keeps_the_fallback_reason() -> None:
    payload = _payload()
    payload["entities"]["proteins"].append(_protein("Nothing", species=""))
    payload["processes"]["interactions"] = [
        _interaction("Nothing inhibits SFXN4", "Nothing", "inhibits", "SFXN4"),
        # Not run through Stage 6 below: a mapped payload from before the wrapper.
        _interaction("SFXN4 inhibits ALAS2", "SFXN4", "inhibits", "ALAS2"),
    ]
    ir, _report = build_pwml_ir(payload, strict_db=True)
    reasons = [row["withheld"]["reason"] for row in ir["processes"]["interactions"]]
    assert reasons == ["protein_endpoint_missing_species_and_external_identity", "bare_protein_endpoint"]


def test_the_exported_shape_matches_pathbanks_own_protein_complex_interaction() -> None:
    payload = _payload()
    _stage6(payload)
    _ir, _build, root = _export(payload)
    reference = parse_xml(PATHBANK_PROTEIN_INTERACTION).getroot()
    theirs = next(n for n in reference.iter("interaction") if n.findtext("id") == "15843")
    ours = next(n for n in root.iter("interaction") if n.findtext("interaction-type") == "Inhibition")

    assert [c.tag for c in ours] == [c.tag for c in theirs]
    complexes = {pc.findtext("id"): pc for pc in root.iter("protein-complex")}
    proteins = {p.findtext("id"): p.findtext("name") for p in root.iter("protein")}
    for side in ("interaction-left-elements", "interaction-right-elements"):
        mine, ref = ours.find(side)[0], theirs.find(side)[0]
        assert [c.tag for c in mine] == [c.tag for c in ref]
        assert mine.findtext("element-type") == ref.findtext("element-type") == "ProteinComplex"
        # The element is a complex in this document's protein-complexes section, holding
        # exactly one protein -- PathBank's single-protein shape (PW687775 complex 416678).
        wrapper = complexes[mine.findtext("element-id")]
        members = list(wrapper.iter("protein-complex-protein"))
        assert len(members) == 1
        assert proteins[members[0].findtext("protein-id")] in {"SFXN4", "ALAS2"}

    def viz(doc: etree._Element, interaction: etree._Element) -> etree._Element:
        return next(v for v in doc.iter("interaction-visualization")
                    if v.findtext("interaction-id") == interaction.findtext("id"))

    our_items = list(viz(root, ours).find("interaction_protein_complex_visualizations"))
    ref_items = list(viz(reference, theirs).find("interaction_protein_complex_visualizations"))
    assert [i.findtext("side") for i in our_items] == [i.findtext("side") for i in ref_items] == ["Left", "Right"]
    assert [[c.tag for c in i] for i in our_items] == [[c.tag for c in i] for i in ref_items]
    pcv_ids = {v.findtext("id") for v in root.iter("protein-complex-visualization")}
    assert all(i.findtext("protein-complex-visualization-id") in pcv_ids for i in our_items)


# ── the text outputs: word as extracted, evidence, and the PWML verdict ──────


def test_reaction_summary_rows_carry_the_word_the_evidence_and_the_exporters_verdict() -> None:
    """The verdict is the exporter's own rule on the payload the graph was built from:
    before Stage 6 a protein endpoint is still bare, after it the wrapper is drawn."""

    def blocks(payload: dict) -> dict:
        text = generate_reaction_summary(build_draft_graph(payload), {})
        return {b.splitlines()[0]: b for b in text.split("\n\n") if b.startswith("INTERACTION")}

    before = blocks(_payload())
    assert len(before) == 5
    binds = before["INTERACTION 3: SFXN4 binds ALAS2"]
    assert "  Relationship: binds" in binds
    assert "  Evidence: “quote for SFXN4 binds ALAS2”" in binds
    assert ("  PWML (if exported from this payload): withheld (untyped_interaction): "
            "'binds' is neither Activation nor Inhibition") in binds
    assert "withheld (bare_protein_endpoint)" in before["INTERACTION 1: SFXN4 inhibits ALAS2"]

    mapped = _payload()
    _stage6(mapped)
    after = blocks(mapped)
    assert "  PWML (if exported from this payload): Inhibition, drawn" in after["INTERACTION 1: SFXN4 inhibits ALAS2"]
    ftsh = after["INTERACTION 4: FtsH inhibits ALAS2"]
    assert "  Relationship: inhibits" in ftsh
    assert ("withheld (protein_endpoint_missing_external_identity): FtsH lacks a UniProt/DrugBank "
            "identity, so Stage 6 cannot wrap it") in ftsh
    assert "Orphan lacks a species" in after["INTERACTION 5: Orphan inhibits ALAS2 mRNA"]


def test_reaction_summary_says_when_an_interaction_has_no_evidence() -> None:
    payload = _payload()
    payload["processes"]["interactions"] = [
        {"name": "ALAS2 activates its mRNA", "entity_1": "ALAS2", "relationship": "activates", "entity_2": "ALAS2 mRNA"}
    ]
    assert "  Evidence: NONE RECORDED — treat as unsupported" in generate_reaction_summary(build_draft_graph(payload), {})


def test_an_interaction_free_draft_graph_serializes_as_before() -> None:
    payload = _payload()
    payload["processes"]["interactions"] = []
    assert "interaction_records" not in build_draft_graph(payload).metadata


def test_research_report_rows_carry_the_exporters_verdict_in_json_and_text() -> None:
    mapped = _payload()
    _stage6(mapped)
    report = build_citation_report(mapped)
    rows = {row["name"]: row for row in report.interactions}
    assert rows["SFXN4 binds ALAS2"]["relationship"] == "binds"
    assert rows["SFXN4 binds ALAS2"]["pwml_interaction_type"] == ""
    assert rows["SFXN4 binds ALAS2"]["pwml_withheld_reason"] == "untyped_interaction"
    assert rows["SFXN4 inhibits ALAS2"]["pwml_interaction_type"] == "Inhibition"
    assert rows["SFXN4 inhibits ALAS2"]["pwml_withheld_reason"] == ""
    assert rows["FtsH inhibits ALAS2"]["pwml_withheld_reason"] == "protein_endpoint_missing_external_identity"
    text = report.to_text()
    assert text.count("PWML: none in research mode; a strict export of this payload:") == 5
    assert "quote for FtsH inhibits ALAS2" in text
    assert "FtsH lacks a UniProt/DrugBank identity" in " ".join(text.split())  # wrapped at 80 columns


def test_the_withheld_record_carries_the_evidence() -> None:
    """The record the app persists (pwml_ir.json) and the CLI side file read from it."""
    payload = _payload()
    _stage6(payload)
    _ir, build, _root = _export(payload)
    by_name = {row["name"]: row for row in build.withheld_interactions}
    assert by_name["SFXN4 binds ALAS2"]["evidence"] == "quote for SFXN4 binds ALAS2"
    assert by_name["SFXN4 binds ALAS2"]["raw_interaction_type"] == "binds"


# ── CORRECTION (G9: fails on base f9892cdf, passes at the tip) ───────────────


def test_competes_with_is_withheld_as_untyped_not_drawn_as_a_directed_inhibition() -> None:
    """Competition is symmetric. The base drew ``A competes_with B`` as A -| B, the
    exporter picking a direction after the freeze (merge rule 8); D-104 § 4 forbids
    mapping an unsigned word onto a sign. Base-SHA failure:
    ``orch754b_wrapper_base_proof.py``, input ``correction:competes_with``."""
    payload = _payload()
    payload["entities"]["nucleic_acids"].append({"name": "SFXN4 mRNA", "class": "nucleic_acid"})
    payload["processes"]["interactions"] = [
        _interaction("the two mRNAs compete", "ALAS2 mRNA", "competes_with", "SFXN4 mRNA"),
        _interaction("the two mRNAs compete, spaced", "SFXN4 mRNA", "Competes  With", "ALAS2 mRNA"),
    ]
    ir, _report = build_pwml_ir(payload, strict_db=True)
    rows = ir["processes"]["interactions"]
    assert [r["export_status"] for r in rows] == ["withheld", "withheld"]
    assert [r["withheld"]["reason"] for r in rows] == ["untyped_interaction"] * 2
    assert [r["withheld"]["raw_interaction_type"] for r in rows] == ["competes_with", "Competes With"]
