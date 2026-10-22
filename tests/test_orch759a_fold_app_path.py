"""ORCH-759a, ruling condition 2: the duplicate-protein fold fires in the REAL app path.

NEW ACCEPTANCE TEST (G9 label: new capability; no base failure is claimed, though the base
outcome is recorded in ``ORCH-759a-RESULT.md``).

ORCH-758 D4 proved D-104 rule 3 dead in the app: it is gated on ``_pooled_draws()``, a
ContextVar the post-pipeline script run cannot see. The fold is gated only on the Stage-6
pass itself. This drives the protected ``streamlit_app.py`` through ``AppTest`` exactly as
``t2pw.batch.driver`` does -- one script run to load, the finished pipeline seeded, then the
post-pipeline button in a script run of its own -- using the harness of
``test_streamlit_quarantine_boundary.py``. Nothing in the app is replaced except what would
leave the machine or what the run never had: the REAL ``map_payload`` runs at Stage 2B and
Stage 6 (protein and compound lookups answered from a name-keyed table, no database, no network,
no LLM), enrichment is a copy, and the curator -- offline in every deterministic run -- is the
audit's stand-in: it gives the wrapper's component the UniProt the audit gave it on the
enterobactin leg, after Stage 3, which is what lets Stage-6 reconciliation rename it and
orphan the row (ORCH-759 E1, step 3).

Asserted on the app's own session state: the Stage-6 mapping report records the fold, the
pre-export gate passes and refinement review opens, the LLM-bound mapping misses do not carry
the record -- and rule 3's ``_pooled_draws()``, read at the moment the app's Stage-6 pass
called it, was 0 at both draw counts. Run this file in its own process (AppTest).
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

# The harness, imported rather than duplicated (``real_streamlit`` is autouse at its definition).
from test_streamlit_quarantine_boundary import (  # noqa: F401
    STAGE_ZERO_CONTEXT,
    _app_exceptions,
    _core_payload,
    _run_post_pipeline,
    real_streamlit,
)
from test_orch759a_duplicate_protein_fold import FOLDS_KEY

DUPLICATE = "glutathione synthase"
SURVIVOR = "glutathione synthetase"  # the core payload's row, P48637
WRAPPER = "glutathione synthase complex"
CHEBI = {"L-glutamate": "16015", "L-cysteine": "17561", "gamma-glutamylcysteine": "17515",
         "glycine": "15428", "glutathione": "16856"}


def _e1_payload() -> Dict[str, Any]:
    """The boundary suite's core payload plus the E1 shape: a second row for GSS, the one
    component of a generated wrapper that also catalyses glutathione synthesis. No accession
    is on the component yet, so Stage 2B and Stage 3 leave it named after its own row.

    The row carries its record's names as synonyms, as ``enterobactin synthase`` did
    (``entF``, ``Enterobactin synthase component F``). That is what kept the orphan alive on
    the leg: the quarantine pruner resolves a row by name UNION synonyms and finds
    ``Glutathione synthetase`` among the surviving complex's components, while the final gate
    tests the name alone. Without such a synonym the pruner deletes the orphan at base and
    the run passes -- an earlier version of this test did exactly that."""

    payload = _core_payload()
    payload["entities"]["proteins"].append(
        {"name": DUPLICATE, "species": "Homo sapiens", "mapped_ids": {"uniprot": "P48637"},
         "synonyms": ["Glutathione synthetase", "GSS"]})
    payload["entities"]["protein_complexes"].append({
        "name": WRAPPER, "class": "protein_complex", "generated": True,
        "generation_reason": "single_protein_pathwhiz_wrapper", "species": "Homo sapiens",
        "components": [{"name": DUPLICATE, "stoichiometry": 1}],
    })
    payload["processes"]["reactions"][1]["enzymes"].append(
        {"entity": WRAPPER, "entity_type": "protein_complex", "role": "catalyst"})
    return payload


#: What the resolver would answer for each protein NAME: (UniProt, recommended name, gene).
#: ``glutathione synthase`` resolves to GSS's own record, as ``enterobactin synthase`` did to EntF's.
UNIPROT = {"glutamate-cysteine ligase": ("P48506", "Glutamate--cysteine ligase catalytic subunit", "GCLC"),
           SURVIVOR: ("P48637", "Glutathione synthetase", "GSS"),
           DUPLICATE: ("P48637", "Glutathione synthetase", "GSS")}


def _protein_lookup(*, name: str, **_: Any) -> Dict[str, Any]:
    hit = UNIPROT.get(name)
    if hit is None:
        return {"status": "unmapped", "reason": "no_match", "candidates": [], "resolution": {"status": "unresolved"}}
    accession, resolved, gene = hit
    candidate = {"name": resolved, "uniprot": accession, "gene_name": gene, "organism": "Homo sapiens",
                 "taxonomy_id": "9606", "score": 1.0}
    return {"status": "mapped", "mapped_ids": {"uniprot": accession, "gene_name": gene}, "confidence": 1.0,
            "provider": "UniProt", "source": "api", "chosen_rule": "exact_name", "resolved_name": resolved,
            "candidates": [candidate], "resolution": {"status": "matched", "order_step": "api_uniprot"}}


def _compound_lookup(*, name: str, **_: Any) -> Dict[str, Any]:
    chebi = CHEBI.get(name)
    if chebi is None:
        return {"status": "unmapped", "reason": "no_match", "candidates": [], "resolution": {"status": "unresolved"}}
    return {"status": "mapped", "mapped_ids": {"chebi": chebi}, "confidence": 1.0, "provider": "fixture",
            "source": "api", "chosen_rule": "exact_name", "resolved_name": name,
            "candidates": [{"name": name, "chebi": chebi}], "resolution": {"status": "matched"}}


@pytest.fixture()
def offline_app(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    """Shut every external door at the modules the app binds at exec time; keep map_payload real.
    Returns the list rule 3's ``_pooled_draws()`` values are appended to, one per Stage-6 call."""

    import t2pw.curation.pathway_curator as pathway_curator
    import t2pw.mapping.enrich_entities as enrich_entities
    import t2pw.mapping.map_ids as map_ids
    import t2pw.pipeline.stage_one_ensemble as ensemble

    monkeypatch.setenv("T2PW_SPECIES_LLM", "0")
    monkeypatch.setenv("T2PW_SPECIES_NCBI", "0")
    monkeypatch.setenv("T2PW_OFFLINE_CURATOR", "1")
    monkeypatch.setattr(map_ids, "_map_protein_with_strategy", _protein_lookup)
    monkeypatch.setattr(map_ids, "_map_compound_with_strategy", _compound_lookup)
    monkeypatch.setattr(map_ids, "_ai_protein_synonym_lookup", lambda *a, **k: [])
    monkeypatch.setattr(map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None))
    monkeypatch.setattr(map_ids.MappingCache, "save", lambda self: None)  # the tracked cache file stays as it is

    def no_network(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("network call during an offline app run")

    monkeypatch.setattr(map_ids.HttpClient, "get", no_network)

    def no_enrichment(input_path: Any, output_path: Any, report_path: Any, **_k: Any) -> Dict[str, Any]:
        Path(output_path).write_text(Path(input_path).read_text(encoding="utf-8"), encoding="utf-8")
        Path(report_path).write_text("{}", encoding="utf-8")
        return {"summary": {"enrichment_skipped_for_test": True}}

    monkeypatch.setattr(enrich_entities, "run_enrichment", no_enrichment)

    def audit_stand_in(input_path: Path, output_path: Path, report_path: Path, **_k: Any) -> Dict[str, Any]:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        for row in payload["entities"]["protein_complexes"]:
            if row.get("name") == WRAPPER:
                for component in row["components"]:
                    component["mapped_ids"] = {"uniprot": "P48637"}
        Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
        report = {"summary": {"patches_accepted": 1}, "stand_in": "the enterobactin leg's accepted audit patch"}
        Path(report_path).write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(pathway_curator, "run_pathway_curator", audit_stand_in)

    pooled: List[int] = []
    real_rule3 = ensemble.merge_on_mapped_identity

    def rule3_spy(payload: Dict[str, Any], report: Any = None) -> Dict[str, Any]:
        pooled.append(ensemble._pooled_draws())
        return real_rule3(payload, report=report)

    monkeypatch.setattr(ensemble, "merge_on_mapped_identity", rule3_spy)
    return pooled


@pytest.mark.parametrize("draws", ["1", "3"], ids=["one_draw", "three_draws"])
def test_new_the_fold_fires_through_the_apps_own_stage6_call(offline_app: List[int], draws: str,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from t2pw.curation.interactive_curator import compact_mapping_misses
    from t2pw.pipeline.gate_reports import FINAL_GATE_REPORT_KEY
    from t2pw.pipeline.stage_one_ensemble import ENV_VAR

    monkeypatch.setenv(ENV_VAR, draws)
    at = _run_post_pipeline(_e1_payload())

    assert _app_exceptions(at) == []
    # The Stage-6 call ran once, in a script run where rule 3's gate reads 0 (ORCH-758 D4).
    assert offline_app == [0]
    artifacts = at.session_state["post_pipeline_artifacts"]
    # The final pre-export gate passes and refinement review opens. (Base: this run stops on
    # "Protein has degree 0 after normalization: glutathione synthase".)
    assert at.session_state["refinement_gate_errors"] == []
    assert artifacts[FINAL_GATE_REPORT_KEY]["ok"] is True
    folds = artifacts["mapping_report"][FOLDS_KEY]
    assert [(f["survivor"], f["folded"], f["shared_accessions"].get("uniprot")) for f in folds] == [
        (SURVIVOR, DUPLICATE, "P48637")]
    assert at.session_state["refinement_mapping_report"][FOLDS_KEY] == folds
    # Reconciliation decided as it always does; the orphan it left is gone, its name kept --
    # in the canonical payload the gate judged and the exporter received.
    mapped = artifacts["final_mapped"]
    [wrapper] = [r for r in mapped["entities"]["protein_complexes"] if r["name"] == WRAPPER]
    assert [c["name"] for c in wrapper["components"]] == [SURVIVOR]
    assert DUPLICATE not in [r["name"] for r in mapped["entities"]["proteins"]]
    [gss] = [r for r in mapped["entities"]["proteins"] if r["name"] == SURVIVOR]
    assert DUPLICATE in gss["synonyms"]
    # What the interactive curator's prompt is built from never carries the record.
    misses = at.session_state["refinement_mapping_misses"]
    assert FOLDS_KEY not in json.dumps(misses)
    without = {k: v for k, v in deepcopy(artifacts["mapping_report"]).items() if k != FOLDS_KEY}
    assert misses == compact_mapping_misses(without)
