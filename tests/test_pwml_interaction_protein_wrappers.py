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

import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Only names that exist at the base too, so the G9 runs at earlier SHAs fail on behaviour.
from t2pw.app.reaction_evidence import build_report, render_text  # noqa: E402
from t2pw.mapping.map_ids import MappingCache, _rewrite_reaction_protein_enzymes_to_complexes  # noqa: E402
from t2pw.pipeline.draft_graph import build_draft_graph  # noqa: E402
from t2pw.pipeline.qa_graph import generate_qa_report  # noqa: E402
from t2pw.pipeline.reaction_summary import generate_reaction_summary  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir, validate_required_pwml_contract  # noqa: E402
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution  # noqa: E402
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
    # A row left alone leaves no trace in this report: it reaches an LLM prompt (D-104,
    # 2026-10-20 ruling). The exporter's own record names why the row was withheld.
    assert [k for k in report["summary"] if k.startswith("interaction_")] == [
        "interaction_protein_endpoints_rewritten_to_complexes"
    ]
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
    touched = {a["json_pointer"].rsplit("/", 1)[0] for a in report["actions"] if a["type"].startswith("interaction_")}
    assert touched == {"/processes/interactions/0", "/processes/interactions/1"}


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
    assert not [a for a in report["actions"] if a["type"].startswith("interaction_")]


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


# ── the text record: human-facing only, and read from the IR ─────────────────
#
# Repair round 1 (REV of d73e9bc8). The text verdict had been recomputed beside the
# exporter, with a name lookup the exporter does not use, and it was written into
# ``reaction_summary.txt`` -- which two live LLMs read. Now the LLM-bound text is the
# base's, byte for byte, and the full record (word, evidence, verdict) is in the
# reaction-evidence report, which READS the verdict out of the built IR.


class _DownDb:
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


#: The committed leg whose ``entC`` / ``entF`` endpoints are synonyms of the proteins
#: ``EntC`` / ``EntF`` -- the 2 of 336 rows where the earlier text said "drawn" and the
#: IR said withheld.
SYNONYM_LEG = ROOT / "runs_verify" / "2026-08-21_2239" / "papers" / "PMC12452463" / "research" / "final_mapped.json"
LIPID_A_LEG = ROOT / "runs_verify" / "2026-09-01_1612" / "papers" / "PMC12444477" / "research" / "final_mapped.json"


def _ir_for(payload: dict) -> dict:
    """The export entry sequence, offline: pre-freeze with the DB down, then the IR."""
    staged = deepcopy(payload)
    run_prefreeze_resolution(staged, strict_db=True, db_resolver=_DownDb(), name_index=None)
    ir, _report = build_pwml_ir(staged, strict_db=True, name_index=None)
    return ir


def _evidence_blocks(payload: dict, ir: dict) -> dict:
    """``{"INTERACTION n": its block}`` from the reaction-evidence .txt."""
    text = render_text(build_report({"final_mapped.json": payload, "pwml_ir.json": ir}))
    chunks = re.split(r"\n(?=INTERACTION \d+  \[)", text)
    return {c.split("  [")[0]: c.split("\n\n")[0] for c in chunks if re.match(r"INTERACTION \d+  \[", c)}


def _assert_the_record_states_the_ir(blocks: dict, ir: dict) -> None:
    rows = ir["processes"]["interactions"]
    assert len(blocks) == len(rows)
    for number, row in enumerate(rows, 1):
        block = blocks[f"INTERACTION {number}"]
        if row["export_status"] == "exportable":
            assert f"PWML: drawn as {row['interaction_type']}" in block
        else:
            assert f"PWML: withheld -- {row['withheld']['reason']}" in block


def _synonym_payload() -> dict:
    """The reviewer's probe: the endpoint is a synonym of a protein with UniProt and species."""
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "pathwhiz_id": 2}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 1}],
            "compounds": [],
            "proteins": [{"name": "EntC", "species": "Escherichia coli", "organism": "Escherichia coli", "species_id": 2,
                          "synonyms": ["Isochorismate synthase EntC"], "mapped_ids": {"uniprot": "P0AEJ2"}}],
            "protein_complexes": [],
            "nucleic_acids": [{"name": "entC mRNA", "class": "nucleic_acid"}],
        },
        "biological_states": [{"name": "cytosol", "species": "Escherichia coli", "subcellular_location": "cytosol",
                               "compartment_canonical": "cytosol"}],
        "processes": {"reactions": [], "transports": [], "interactions": [
            {"name": "synonym endpoint", "entity_1": "Isochorismate synthase EntC", "relationship": "inhibits",
             "entity_2": "entC mRNA", "biological_state": "cytosol", "evidence": "q"}]},
    }


def test_committed_synonym_endpoints_no_text_output_contradicts_the_ir() -> None:
    """REGRESSION (G9): fails behaviourally at ``d73e9bc8``, where ``reaction_summary``
    said "Inhibition, drawn" for these two rows while the IR withheld them."""
    payload = json.loads(SYNONYM_LEG.read_text(encoding="utf-8"))
    ir = _ir_for(payload)
    rows = ir["processes"]["interactions"]
    assert [r["export_status"] for r in rows[1:3]] == ["withheld", "withheld"]
    _assert_the_record_states_the_ir(_evidence_blocks(payload, ir), ir)
    # The LLM-bound summary states no PWML verdict at all, so it has none to get wrong.
    assert "PWML" not in generate_reaction_summary(build_draft_graph(deepcopy(payload)), {})


def test_a_protein_named_by_its_synonym_is_wrapped_and_the_record_says_drawn() -> None:
    """REGRESSION (G9): fails behaviourally at ``d73e9bc8``, where Stage 6 looked names up
    only, left the endpoint bare, and the IR withheld a row the text called drawn."""
    payload = _synonym_payload()
    report = _stage6(payload)
    assert payload["processes"]["interactions"][0]["entity_1"] == "EntC complex"
    wrapper = next(r for r in payload["entities"]["protein_complexes"] if r["name"] == "EntC complex")
    assert [c["name"] for c in wrapper["components"]] == ["EntC"]  # the declared protein, not the alias
    assert report["summary"]["interaction_protein_endpoints_rewritten_to_complexes"] == 1
    ir = _ir_for(payload)
    assert (ir["processes"]["interactions"][0]["export_status"], ir["processes"]["interactions"][0]["interaction_type"]) == (
        "exportable", "Inhibition")
    _assert_the_record_states_the_ir(_evidence_blocks(payload, ir), ir)


def test_an_alias_two_proteins_share_is_not_guessed_and_the_record_still_states_the_ir() -> None:
    payload = _synonym_payload()
    twin = deepcopy(payload["entities"]["proteins"][0])
    twin.update(name="EntC2", mapped_ids={"uniprot": "P99998"})
    payload["entities"]["proteins"].append(twin)
    _stage6(payload)
    assert payload["processes"]["interactions"][0]["entity_1"] == "Isochorismate synthase EntC"
    ir = _ir_for(payload)
    assert ir["processes"]["interactions"][0]["withheld"]["reason"] == "bare_protein_endpoint"
    _assert_the_record_states_the_ir(_evidence_blocks(payload, ir), ir)


def test_the_human_facing_record_carries_word_evidence_and_the_ir_verdict_for_every_row() -> None:
    payload = _payload()
    _stage6(payload)
    ir = _ir_for(payload)
    blocks = _evidence_blocks(payload, ir)
    _assert_the_record_states_the_ir(blocks, ir)
    for number, row in enumerate(payload["processes"]["interactions"], 1):
        block = blocks[f"INTERACTION {number}"]
        assert f"Relationship (as extracted): {row['relationship']}" in block
        assert f'1. "{row["evidence"]}"' in block
    assert ("PWML: withheld -- protein_endpoint_missing_external_identity: FtsH has no UniProt/DrugBank "
            "identity, which a single-protein complex wrapper requires") in blocks["INTERACTION 4"]
    assert "Orphan has no species" in blocks["INTERACTION 5"]
    assert "PWML: withheld -- untyped_interaction" in blocks["INTERACTION 3"]


def test_every_reason_is_glossed_truthfully_including_a_refused_wrapper() -> None:
    """NON-BLOCKING finding: the gloss for ``bare_protein_endpoint`` must not claim
    Stage 6 builds a wrapper -- here it refused one (the name is a declared complex's)."""
    payload = _payload()
    payload["entities"]["protein_complexes"] = [
        {"name": "ALAS2 complex", "species": "Homo sapiens", "components": [{"name": "ALAS2"}, {"name": "SFXN4"}]}
    ]
    payload["processes"]["interactions"] = payload["processes"]["interactions"][:1]
    _stage6(payload)
    ir = _ir_for(payload)
    assert ir["processes"]["interactions"][0]["withheld"]["reason"] == "bare_protein_endpoint"
    block = _evidence_blocks(payload, ir)["INTERACTION 1"]
    assert "is not wrapped in a single-protein complex in this payload" in block
    assert "Stage 6 builds" not in block


def test_the_record_says_not_recorded_rather_than_guessing_without_an_ir() -> None:
    text = render_text(build_report({"final_mapped.json": _payload()}))
    assert text.count("PWML: Not recorded (no pwml_ir.json") == 5


def test_the_research_report_says_no_row_reached_a_pwml_and_keeps_its_json_shape() -> None:
    from t2pw.rag.research_report import build_citation_report

    report = build_citation_report(_payload())
    assert "Research mode writes no PWML, so none of these rows reached one." in report.to_text()
    assert set(report.interactions[0]) == {"pointer", "name", "source", "relationship", "target",
                                           "biological_state", "evidence", "provenance", "confidence"}


# ── D-104 ruling of 2026-10-20: every LLM input is the base's, byte for byte ──


def _sha(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _llm_bound_digests() -> dict:
    """What reaches a model, for fixed payloads: the reaction summary (the curator and the
    enrichment agent), the draft graph and the QA report it is built from, and the Stage-6
    report and payload for a payload whose signed rows cannot be wrapped (the report feeds
    the interactive curator's "Mapping Misses")."""
    out = {}
    payloads = {
        "fixture": _payload(),
        "synonym_fixture": _synonym_payload(),
        "committed_synonym_leg": json.loads(SYNONYM_LEG.read_text(encoding="utf-8")),
        "committed_lipid_a_research": json.loads(LIPID_A_LEG.read_text(encoding="utf-8")),
    }
    for name, payload in payloads.items():
        graph = build_draft_graph(deepcopy(payload))
        qa = generate_qa_report(graph, deepcopy(payload))
        out[name] = {
            "reaction_summary": _sha(generate_reaction_summary(graph, qa)),
            "draft_graph": _sha(graph.to_dict()),
            "qa_report": _sha(qa),
        }
    unwrappable = _payload()
    unwrappable["processes"]["interactions"] = unwrappable["processes"]["interactions"][2:]
    report = _stage6(unwrappable)
    out["stage6_all_unwrappable"] = _sha({"report": report, "payload": unwrappable})
    return out


#: ``_llm_bound_digests()`` at base ``f9892cdf``, computed by
#: ``docs/pwml_recovery_sprint/evidence/orch754b_llm_input_identity.py``.
LLM_BOUND_BASE_DIGESTS: dict = {
    "fixture": {
        "reaction_summary": "8457bc67e09062d1cfa9af9aed322aafd66010304ec8863db41024fb2807d7e6",
        "draft_graph": "ec6e5a6d34b72e143d35820574d7c1a0179cddd99d32907a203bd19beb0f7908",
        "qa_report": "8c37bfb999bf7eb6cd15971fef5006dfb69881a74c6aba527c796f731939f98c"
    },
    "synonym_fixture": {
        "reaction_summary": "c9dd31145a2426a6061388d659e9bb3bd48f5c5c918359844cf63afb34621020",
        "draft_graph": "572b83f03eb5d6e68ea00440dbf0ae3c08b7b105725b8b857900078009bc4b65",
        "qa_report": "54f1dbc313bb0aabc42f602c94512ca00246885ff9fe478a4f77bbb19e6f08d4"
    },
    "committed_synonym_leg": {
        "reaction_summary": "5281285486026c01daf33c6554d16cf5c0fe11de32db3375601a8d31040b1923",
        "draft_graph": "3f6c686b182db63886f2da911c531c3fc5d08f50488788f8ffa36a6056f546f2",
        "qa_report": "31c2790466776f8156b42136c07c8065a722084022d8536160bca90545e88204"
    },
    "committed_lipid_a_research": {
        "reaction_summary": "adeccd8fb34baa690e739d3b7b97a83d8ad701514b2792dcc4a5ce3a4656984a",
        "draft_graph": "978f3a09c9768f1ba4b166bfd6782a5f5a021c389ebb8758494c4cdd06504119",
        "qa_report": "1829e4ef6c7869aba66cabca5c818174dfe5eaac4ff35c4b2a1d87c23c18f689"
    },
    "stage6_all_unwrappable": "75d9b8b2f729368908d6d847b3f3d539815e6d959d0b01770f3f31db88c968e6"
}


def test_every_llm_bound_input_is_byte_identical_to_the_base() -> None:
    """D-104 ruling (2026-10-20). Fails behaviourally at ``d73e9bc8``, whose reaction
    summary grew an Evidence and a PWML line per interaction."""
    assert _llm_bound_digests() == LLM_BOUND_BASE_DIGESTS


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
