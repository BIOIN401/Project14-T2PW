"""ORCH-759a / ORCH-759 finding E1: an unreferenced duplicate protein row must not block a
correct ``review_required`` pathway.

On the enterobactin 3-draw leg (``runs_papers/ORCH-759/legs/enterobactin-strict/
2026-10-22_1037``) the audit tied protein row ``enterobactin synthase`` -- the one component
of the generated wrapper ``enterobactin synthase complex`` -- to EntF's UniProt P11454. The
Stage-6 mapping pass reconciles complex components on UniProt first, first row wins, so the
component became ``EntF`` and the row ``enterobactin synthase`` was referenced by nothing.
The final pre-export gate then raised "Protein has degree 0" and no PWML was written,
although quarantine had released the pathway as ``review_required``.

**G9 -- CORRECTION of observable behaviour.** The two ``test_g9_*`` cases replay the leg's
own artifact through ``map_payload``'s real Stage-6 entry (``merge_identity_duplicates=True``,
the keyword the protected ``streamlit_app.py`` passes at its Stage-6 call). Lookups are not
made: each protein row's recorded resolution is replayed (``_replayed_lookup``), every other
door is shut. On the base SHA both fail on the final pre-export gate's own verdict, "Protein
has degree 0 after normalization: enterobactin synthase" -- behaviour, not a missing symbol:
they use nothing this lane adds (the record key is spelled out).

**NEW ACCEPTANCE TESTS** (``test_new_*``, labelled as such; no base failure is claimed for
them): the conditions of the 2026-10-22 ruling -- what may fold (admitted accession, species,
unreferenced, Stage 6 only, at any draw count), what must stay untouched for the gate, and that
the fold record reaches no LLM input. Some call the lane's fold step
(``map_ids._fold_unreferenced_duplicate_proteins``) directly. The real-app proof is in
``tests/test_orch759a_fold_app_path.py``.

Fixture: ``tests/fixtures/orch759a/enterobactin_final_mapped.excerpt.json``, cut from the
leg's ``final_mapped.json`` by ``docs/pwml_recovery_sprint/evidence/orch759a/make_fixture.py``
(README beside the fixture).

No model, network or database is called anywhere here.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping import map_ids  # noqa: E402
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    run_strict_post_normalization_gates,
)

FIXTURE = ROOT / "tests" / "fixtures" / "orch759a" / "enterobactin_final_mapped.excerpt.json"
#: The mapping-report key of the fold record, spelled out rather than imported (G9: at base
#: the constant does not exist, and an ImportError would prove symbol absence, not behaviour).
FOLDS_KEY = "duplicate_protein_folds"
DEGREE_ZERO = "Protein has degree 0 after normalization: enterobactin synthase"
WRAPPER = "enterobactin synthase complex"


def _leg_payload() -> Dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _final_gate_errors(payload: Dict[str, Any]) -> List[str]:
    """The final pre-export gate exactly as ``streamlit_app.py`` calls it in strict mode."""

    try:
        details = run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
    except GateValidationError as exc:
        details = exc.details or {}
    return [str(error.get("reason")) for error in details.get("errors") or []]


def _replayed_lookup(*, name: str, protein_row: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:
    """Stand-in for ``map_ids._map_protein_with_strategy``: the row's OWN recorded resolution.

    Every protein row of the leg carries what Stage 6 resolved for it (``mapped_ids``,
    ``mapping_meta``); this hands that back as the resolver's answer, so the mapping pass
    re-admits exactly the identities the leg shipped and decides everything else itself
    (name gate, admission passes, component reconciliation). A row with no recorded identity
    stays unmapped.
    """

    row = protein_row if isinstance(protein_row, dict) else {"name": name}
    meta = row.get("mapping_meta") if isinstance(row.get("mapping_meta"), dict) else {}
    ids = row.get("mapped_ids") if isinstance(row.get("mapped_ids"), dict) else {}
    if not ids.get("uniprot") and not ids.get("drugbank"):
        return {"status": "unmapped", "reason": "no_recorded_identity", "provider": "none", "source": "none",
                "candidates": [], "resolution": {"status": "unresolved", "issue": "no_recorded_identity"}}
    return {
        "status": "mapped",
        "mapped_ids": dict(ids),
        "pathbank_protein_id": row.get("pathbank_protein_id"),
        "provider": meta.get("provider") or "PathBankDB",
        "source": meta.get("source") or "db",
        "confidence": meta.get("confidence", 1.0),
        "chosen_rule": meta.get("chosen_rule") or "pathbank_protein_id",
        "candidates": deepcopy(meta.get("candidates") or []),
        "resolution": deepcopy(meta.get("resolution") or {"status": "matched"}),
        "identity_verdict": deepcopy(meta.get("identity_verdict")),
        "resolved_name": meta.get("resolved_name"),
    }


def _stage6(payload: Dict[str, Any], tmp_path: Path, *, keyword: Any = True) -> Dict[str, Any]:
    """``map_payload`` as the app's Stage-6 call makes it, offline (lookups replayed)."""

    kwargs = {} if keyword is None else {"merge_identity_duplicates": keyword}
    with patch.dict(os.environ, {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}), patch.object(
        map_ids, "_map_protein_with_strategy", side_effect=_replayed_lookup
    ), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(map_ids, "_ai_protein_synonym_lookup", return_value=[]), patch.object(
        map_ids.HttpClient, "get", side_effect=RuntimeError("network call during an offline run")
    ):
        return map_ids.map_payload(deepcopy(payload), cache_path=tmp_path / "cache.json", id_source="db",
                                   use_cache=False, **kwargs)


def _stage6_input_reconstructed() -> Dict[str, Any]:
    """The leg's payload with the wrapper's component as the audit left it before Stage 6:
    named ``enterobactin synthase``, carrying P11454 (ORCH-759 diagnosis, variant A)."""

    payload = _leg_payload()
    [wrapper] = [c for c in payload["entities"]["protein_complexes"] if c["name"] == WRAPPER]
    [component] = wrapper["components"]
    component["name"] = "enterobactin synthase"
    return payload


def _protein_names(payload: Dict[str, Any]) -> List[str]:
    return [row["name"] for row in payload["entities"]["proteins"]]


def _component_names(payload: Dict[str, Any], complex_name: str) -> List[str]:
    [row] = [c for c in payload["entities"]["protein_complexes"] if c["name"] == complex_name]
    return [c["name"] if isinstance(c, dict) else c for c in row["components"]]


def test_the_excerpt_carries_the_legs_defect() -> None:
    """Precondition, true at base and tip: the fixture fails the final gate exactly as the leg
    did (``final_stage3_gate_report.json``), and the reconstructed Stage-6 input does not."""

    payload = _leg_payload()
    assert _final_gate_errors(payload) == [DEGREE_ZERO]
    rows = [(r["name"], r["mapped_ids"]["uniprot"]) for r in payload["entities"]["proteins"]
            if r["mapped_ids"].get("uniprot") == "P11454"]
    assert rows == [("EntF", "P11454"), ("enterobactin synthase", "P11454")]
    assert _final_gate_errors(_stage6_input_reconstructed()) == []


@pytest.mark.parametrize("build", [_leg_payload, _stage6_input_reconstructed],
                         ids=["leg_final_mapped_as_shipped", "stage6_input_reconstructed"])
def test_g9_the_stage6_pass_leaves_a_payload_the_final_gate_passes(build: Any, tmp_path: Path) -> None:
    result = _stage6(build(), tmp_path)
    mapped = result["payload"]

    # Reconciliation decides what it always decided: the wrapper's component is EntF.
    assert _component_names(mapped, WRAPPER) == ["EntF"]
    # THE G9 ASSERTION. Base: ["Protein has degree 0 after normalization: enterobactin synthase"].
    assert _final_gate_errors(mapped) == []

    # The duplicate is folded into EntF, not lost: its name is an alias there, and the fold
    # is recorded in the mapping report with the shared accession and the species.
    assert "enterobactin synthase" not in _protein_names(mapped)
    [entf] = [r for r in mapped["entities"]["proteins"] if r["name"] == "EntF"]
    assert "enterobactin synthase" in entf["synonyms"]
    [fold] = result["report"][FOLDS_KEY]
    assert (fold["survivor"], fold["folded"]) == ("EntF", "enterobactin synthase")
    assert fold["shared_accessions"]["uniprot"] == "P11454"
    assert fold["species"]["name"] == "Escherichia coli"


# ============================================================== NEW ACCEPTANCE TESTS (ORCH-759a)


def _leg_with(row_name: str, **changes: Any) -> Dict[str, Any]:
    """The leg's payload with one protein row's fields replaced (``None`` deletes a field)."""

    payload = _leg_payload()
    [row] = [r for r in payload["entities"]["proteins"] if r["name"] == row_name]
    for key, value in changes.items():
        if value is None:
            row.pop(key, None)
        else:
            row[key] = value
    return payload


def _assert_untouched_and_still_blocked(result: Dict[str, Any]) -> None:
    assert FOLDS_KEY not in result["report"]
    assert "enterobactin synthase" in _protein_names(result["payload"])
    assert _final_gate_errors(result["payload"]) == [DEGREE_ZERO]


def test_new_an_unreferenced_row_with_no_admitted_accession_is_untouched_and_the_gate_still_catches_it(
        tmp_path: Path) -> None:
    """Preservation (ruling 1, last sentence): no identity, no fold -- the degree-0 row stays and
    the final gate still stops the run on it."""

    _assert_untouched_and_still_blocked(_stage6(_leg_with("enterobactin synthase", mapped_ids={}), tmp_path))


def test_new_an_identity_admission_withholding_blocks_the_fold(tmp_path: Path) -> None:
    """An accession identity admission WITHHELD (pass A: no name of the row is in the paper) is
    not admitted, so it cannot fold -- read through the real admission pass, not a raw field."""

    from t2pw.mapping import identity_admission

    payload = _leg_with("enterobactin synthase", synonyms=[])
    payload[identity_admission.SOURCE_INDEX_KEY] = identity_admission.build_source_index(
        "EntF, the nonribosomal peptide synthetase, makes the siderophore.")
    result = _stage6(payload, tmp_path)
    [row] = [r for r in result["payload"]["entities"]["proteins"] if r["name"] == "enterobactin synthase"]
    assert row["mapping_meta"]["rejected_mapped_ids"]["uniprot"] == "P11454"  # withheld, not admitted
    assert "uniprot" not in row["mapped_ids"]
    _assert_untouched_and_still_blocked(result)


@pytest.mark.parametrize("keyword", [None, False], ids=["default", "explicit_false"])
def test_new_only_the_stage6_pass_folds(tmp_path: Path, keyword: Any) -> None:
    """Stage 2B and the RAG-round passes call ``map_payload`` without the keyword: no fold."""

    _assert_untouched_and_still_blocked(_stage6(_leg_payload(), tmp_path, keyword=keyword))


@pytest.mark.parametrize("draws", [None, "1", "3"], ids=["draws_unset", "draws_1", "draws_3_not_pooled"])
def test_new_the_fold_fires_at_any_draw_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draws: Any) -> None:
    """Ruling 2: never gated on the draw count or on rule 3's pooled-draws check. At "3" with no
    Stage-1 boundary in this context -- what the app's post-pipeline script run sees (ORCH-758
    D4) -- rule 3 stays off and the fold still fires."""

    from t2pw.pipeline.stage_one_ensemble import ENV_VAR, _pooled_draws

    if draws is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, draws)
    assert _pooled_draws() == 0
    result = _stage6(_leg_payload(), tmp_path)
    assert [(f["survivor"], f["folded"]) for f in result["report"][FOLDS_KEY]] == [("EntF", "enterobactin synthase")]
    assert _final_gate_errors(result["payload"]) == []


def test_new_the_report_pointers_follow_the_fold(tmp_path: Path) -> None:
    """Row 7 is gone: its log entry names its survivor, and no protein pointer names a row past
    the end of the list."""

    import re

    result = _stage6(_leg_payload(), tmp_path)
    proteins = result["payload"]["entities"]["proteins"]
    logs = {row["name"]: row["json_pointer"] for row in result["report"]["entities"]
            if row.get("entity_type") == "protein"}
    assert logs["enterobactin synthase"] == logs["EntF"] == f"/entities/proteins/{_protein_names(result['payload']).index('EntF')}"
    for name, pointer in logs.items():
        if name != "enterobactin synthase":
            assert proteins[int(pointer.rsplit("/", 1)[1])]["name"] == name
    assert all(int(i) < len(proteins) for i in re.findall(r'"/entities/proteins/(\d+)', json.dumps(result["report"])))


# ------------------------------------------------- the predicate, on the fold step itself


def _row(name: str, uniprot: Optional[str] = "P11454", **extra: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": name, "organism": "Escherichia coli", "species": "Escherichia coli", "taxonomy_id": "562",
        "species_ref": {"name": "Escherichia coli", "taxonomy_id": "562", "pathbank_species_id": 3},
        "mapped_ids": {"uniprot": uniprot} if uniprot else {},
        "mapping_meta": {"confidence": 1.0, "identity_status": "verified", "chosen_rule": "pathbank_protein_id"},
    }
    row.update(extra)
    return row


def _e1(duplicate: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """EntF catalyses through its wrapper; ``duplicate`` is the second row sharing (or not) its id."""

    payload = {
        "entities": {
            "proteins": [_row("EntF", synonyms=["EntF"], evidence="EntF is the synthetase."), duplicate],
            "protein_complexes": [{"name": "EntF complex", "generated": True,
                                   "components": [{"name": "EntF", "mapped_ids": {"uniprot": "P11454"}}]}],
            "compounds": [{"name": "L-serine"}, {"name": "enterobactin"}],
        },
        "element_locations": {"protein_locations": [], "compound_locations": []},
        "processes": {
            "reactions": [{"name": "enterobactin synthesis", "inputs": ["L-serine"], "outputs": ["enterobactin"],
                           "enzymes": [{"entity": "EntF complex", "entity_type": "protein_complex"}]}],
            "transports": [], "interactions": [],
        },
    }
    payload.update(extra)
    return payload


def _fold(payload: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"entities": [{"entity_type": "protein", "name": r["name"], "json_pointer": f"/entities/proteins/{i}"}
                                           for i, r in enumerate(payload["entities"]["proteins"])]}
    map_ids._fold_unreferenced_duplicate_proteins(payload, report)
    return report


def test_new_the_fold_keeps_the_name_and_every_quote_and_records_them() -> None:
    lineage = [{"stage": "audit_repair", "origin": "audit_modified", "sources": []}]
    rag = {"source_id": "PMC0000001", "section": "results", "chunk_id": "c7"}
    papers = [{"source_id": "PMC0000001", "title": "a supporting paper"}]
    duplicate = _row("enterobactin synthase", evidence="EntF (enterobactin synthase) acts last.",
                     source_refs=["the synthase assembles the trilactone", "EntF is the synthetase."],
                     synonyms=["enterobactin synthetase"], provenance_lineage=lineage, rag_provenance=rag,
                     source_papers=papers)
    duplicate["mapped_ids"]["gene_name"] = "entF"
    payload = _e1(duplicate)
    report = _fold(payload)
    assert _protein_names(payload) == ["EntF"]
    [entf] = payload["entities"]["proteins"]
    assert entf["synonyms"] == ["EntF", "enterobactin synthase"]
    assert entf["evidence"] == "EntF is the synthetase."
    assert entf["source_refs"] == ["EntF (enterobactin synthase) acts last.", "the synthase assembles the trilactone"]
    assert report[FOLDS_KEY] == [{
        "rule": "unreferenced_protein_row_shares_an_admitted_accession",
        "survivor": "EntF", "folded": "enterobactin synthase",
        "shared_accessions": {"uniprot": "P11454"},
        "species": {"name": "Escherichia coli", "taxonomy_id": "562", "pathbank_species_id": 3},
        "alias_added": "enterobactin synthase",
        "quotes_moved": ["EntF (enterobactin synthase) acts last.", "the synthase assembles the trilactone"],
        "folded_row_names": ["enterobactin synthase", "enterobactin synthetase"],
        # Repair round 1, N1 and N5: the folded row's whole identity and provenance are kept.
        "folded_mapped_ids": {"uniprot": "P11454", "gene_name": "entF"},
        "folded_provenance_lineage": lineage,
        "folded_rag_provenance": rag,
        "folded_source_papers": papers,
    }]
    assert [e["json_pointer"] for e in report["entities"]] == ["/entities/proteins/0", "/entities/proteins/0"]


def test_new_every_quote_field_moves_with_the_fold() -> None:
    """Repair round 1, B2: every evidence-bearing field ``entity_admission._spans`` and
    ``apply_audit_patch._ACTOR_ROW_EVIDENCE_KEYS`` read, dict ``span``/``text`` entries included."""

    duplicate = _row("enterobactin synthase", evidence="q evidence", evidence_quote="q evidence_quote",
                     source_evidence=["q source_evidence"], source_text="q source_text",
                     source_refs=[{"span": "q span"}, {"text": "q text"}, "EntF is the synthetase."])
    payload = _e1(duplicate)
    [fold] = _fold(payload)[FOLDS_KEY]
    # "EntF is the synthetase." is already the survivor's own evidence, so it does not move twice.
    expected = ["q evidence", "q span", "q text", "q evidence_quote", "q source_evidence", "q source_text"]
    assert fold["quotes_moved"] == expected
    [entf] = payload["entities"]["proteins"]
    assert entf["source_refs"] == expected


@pytest.mark.parametrize("duplicate", [
    _row("enterobactin synthase", uniprot=None,
         mapping_meta={"confidence": 1.0, "identity_status": "unresolved",
                       "candidates": [{"name": "Enterobactin synthase component F", "uniprot": "P11454"}]}),
    _row("enterobactin synthase", uniprot=None,
         mapping_meta={"confidence": 1.0, "identity_status": "verified", "rejected_mapped_ids": {"uniprot": "P11454"}}),
    _row("enterobactin synthase", mapping_meta={"confidence": 0.5, "identity_status": "verified"}),
    _row("enterobactin synthase", mapping_meta={"confidence": 1.0, "identity_status": "verified", "best_effort": True}),
    _row("enterobactin synthase", mapping_meta={"confidence": 1.0, "identity_status": "placeholder"}),
    _row("enterobactin synthase", mapping_meta={"confidence": 1.0, "identity_status": "unresolved"}),
], ids=["candidate_only", "withheld", "low_confidence", "best_effort", "placeholder", "unresolved"])
def test_new_a_candidate_withheld_or_provisional_accession_never_folds(duplicate: Dict[str, Any]) -> None:
    payload = _e1(duplicate)
    before = json.dumps(payload, sort_keys=True)
    assert FOLDS_KEY not in _fold(payload)
    assert json.dumps(payload, sort_keys=True) == before


@pytest.mark.parametrize("duplicate", [
    _row("enterobactin synthase", uniprot="P0ADI4"),
    _row("enterobactin synthase", mapped_ids={"uniprot": "P11454", "pathbank_protein_id": "9999"}),
], ids=["another_accession", "one_shared_one_conflicting"])
def test_new_rows_whose_admitted_accessions_differ_never_fold(duplicate: Dict[str, Any]) -> None:
    payload = _e1(duplicate)
    payload["entities"]["proteins"][0]["mapped_ids"]["pathbank_protein_id"] = "6312"
    assert FOLDS_KEY not in _fold(payload)
    assert len(payload["entities"]["proteins"]) == 2


@pytest.mark.parametrize("species", [
    {"organism": "Salmonella enterica", "species": "Salmonella enterica", "taxonomy_id": "28901",
     "species_ref": {"name": "Salmonella enterica", "taxonomy_id": "28901"}},
    {"organism": "", "species": "", "taxonomy_id": "", "species_ref": {}},
], ids=["different_species", "no_species"])
def test_new_a_different_or_unknown_species_never_folds(species: Dict[str, Any]) -> None:
    payload = _e1(_row("enterobactin synthase", **species))
    assert FOLDS_KEY not in _fold(payload)
    assert len(payload["entities"]["proteins"]) == 2


def _referenced_by(where: str) -> Dict[str, Any]:
    payload = _e1(_row("enterobactin synthase", synonyms=["holo-EntF synthase"]))
    reaction = payload["processes"]["reactions"][0]
    if where == "reaction_enzyme":
        reaction["enzymes"].append({"entity": "enterobactin synthase", "entity_type": "protein"})
    elif where == "reaction_modifier":
        reaction["modifiers"] = [{"entity": "Enterobactin-Synthase", "entity_type": "protein", "role": "activator"}]
    elif where == "transport":
        payload["processes"]["transports"] = [{"name": "export", "cargo": "enterobactin",
                                               "transporters": [{"entity": "enterobactin synthase"}]}]
    elif where == "interaction":
        payload["processes"]["interactions"] = [{"entity_1": "enterobactin synthase", "entity_2": "EntF",
                                                 "relationship": "activates"}]
    elif where == "location":
        payload["element_locations"]["protein_locations"] = [{"protein": "enterobactin synthase",
                                                              "biological_state": "cytosol"}]
    elif where == "complex_component":
        payload["entities"]["protein_complexes"].append({"name": "synthase complex", "components": ["enterobactin synthase"]})
    elif where == "only_alias":
        reaction["enzymes"].append({"entity": "holo-EntF synthase", "entity_type": "protein"})
    return payload


@pytest.mark.parametrize("where", ["reaction_enzyme", "reaction_modifier", "transport", "interaction", "location",
                                   "complex_component", "only_alias"])
def test_new_a_referenced_duplicate_is_never_folded(where: str) -> None:
    payload = _referenced_by(where)
    before = json.dumps(payload, sort_keys=True)
    assert FOLDS_KEY not in _fold(payload)
    assert json.dumps(payload, sort_keys=True) == before


def test_new_an_alias_another_row_holds_does_not_count_as_a_reference_to_the_duplicate() -> None:
    """The leg's own shape: ``enterobactin synthase`` lists ``enterobactin synthase complex`` and
    ``entF`` among its synonyms, and a reaction names both -- as the complex and as EntF."""

    payload = _e1(_row("enterobactin synthase", synonyms=["enterobactin synthase complex", "entF"]))
    payload["entities"]["protein_complexes"].append({"name": "enterobactin synthase complex", "generated": True,
                                                    "components": [{"name": "EntF"}]})
    payload["processes"]["reactions"][0]["enzymes"].append({"entity": "enterobactin synthase complex"})
    payload["processes"]["interactions"] = [{"entity_1": "entF", "entity_2": "enterobactin", "relationship": "binds"}]
    assert [f["folded"] for f in _fold(payload)[FOLDS_KEY]] == ["enterobactin synthase"]


def _shared_alias_payload(holder: str) -> Dict[str, Any]:
    """Repair round 1, B1 (the reviewer's probe 34): the duplicate lists ``shared alias``, a
    reaction enzyme names it, and ``holder`` says who else holds it."""

    payload = _e1(_row("enterobactin synthase", synonyms=["shared alias"]))
    if holder == "unrelated_protein":
        payload["entities"]["proteins"].append(_row("OtherP", uniprot="P99999", synonyms=["shared alias"]))
    elif holder == "complex_alias":
        payload["entities"]["protein_complexes"].append({"name": "other complex", "synonyms": ["shared alias"],
                                                        "components": [{"name": "EntF"}]})
    elif holder == "same_identity_protein":
        payload["entities"]["proteins"][0]["synonyms"].append("shared alias")
    payload["processes"]["reactions"][0]["enzymes"].append({"entity": "shared alias", "entity_type": "protein"})
    return payload


@pytest.mark.parametrize("holder", ["unrelated_protein", "complex_alias"])
def test_new_an_alias_held_by_an_unrelated_row_still_names_the_duplicate(holder: str) -> None:
    """B1: the registry resolves ``shared alias`` to the duplicate today; folding it would move
    the reaction's enzyme to another protein (P99999) or to a complex. So it is a reference."""

    from t2pw.pipeline.strict_quarantine import _build_registry

    payload = _shared_alias_payload(holder)
    assert _build_registry(payload).resolve("shared alias")[1]["name"] == "enterobactin synthase"
    before = json.dumps(payload, sort_keys=True)
    assert FOLDS_KEY not in _fold(payload)
    assert json.dumps(payload, sort_keys=True) == before


def test_new_an_alias_held_only_by_rows_of_the_same_identity_is_not_a_reference() -> None:
    """B1, the other side: when every other protein holding the alias shares the duplicate's
    admitted identity, the alias still resolves to that protein after the fold."""

    payload = _shared_alias_payload("same_identity_protein")
    assert [(f["survivor"], f["folded"]) for f in _fold(payload)[FOLDS_KEY]] == [("EntF", "enterobactin synthase")]


def test_new_two_unreferenced_rows_never_fold_into_each_other() -> None:
    """The survivor must be referenced: two orphans sharing an id stay for the gate to catch."""

    payload = _e1(_row("enterobactin synthase"))
    payload["entities"]["proteins"].append(_row("EntF synthetase"))
    payload["entities"]["proteins"][0]["mapped_ids"] = {"uniprot": "P0ADI4"}
    assert FOLDS_KEY not in _fold(payload)
    assert len(payload["entities"]["proteins"]) == 3


# ------------------------------------------------- the record reaches no LLM input (D-104 am. 2)


def test_new_the_fold_record_reaches_no_llm_input(tmp_path: Path) -> None:
    """The record lives only in the Stage-6 mapping report. The one LLM that reads that report is
    the interactive curator, through ``compact_mapping_misses``; its whole prompt is byte-identical
    with and without the record. The payload -- which every LLM may read -- carries no record."""

    from t2pw.curation.interactive_curator import build_interactive_curator_messages, compact_mapping_misses

    result = _stage6(_leg_payload(), tmp_path)
    report = result["report"]
    assert report[FOLDS_KEY]
    without = {key: value for key, value in report.items() if key != FOLDS_KEY}
    assert compact_mapping_misses(report) == compact_mapping_misses(without)

    def prompt(mapping: Dict[str, Any]) -> str:
        messages = build_interactive_curator_messages(result["payload"], b"", "fix it", mapping_misses=mapping)
        return json.dumps(messages, ensure_ascii=False)

    assert prompt(report) == prompt(without)
    assert prompt(compact_mapping_misses(report)) == prompt(compact_mapping_misses(without))
    payload_text = json.dumps(result["payload"], ensure_ascii=False)
    assert FOLDS_KEY not in payload_text
    assert "unreferenced_protein_row_shares_an_admitted_accession" not in payload_text


def test_new_the_record_key_is_skipped_whatever_it_holds() -> None:
    """Provable, not lucky: entries shaped exactly like mapping misses under the record key are
    still never walked, at the top level or nested."""

    from t2pw.curation.interactive_curator import compact_mapping_misses

    miss = {"name": "enterobactin synthase", "status": "unmapped", "reason": "ambiguous", "confidence": 0.1}
    base = {"summary": {}, "entities": [{"name": "EntA", "status": "unmapped", "reason": "no_match"}]}
    assert compact_mapping_misses({**base, FOLDS_KEY: [miss]}) == compact_mapping_misses(base)
    assert compact_mapping_misses({**base, "nested": {FOLDS_KEY: [miss]}}) == compact_mapping_misses({**base, "nested": {}})
    assert compact_mapping_misses(base)  # the walker itself still works
