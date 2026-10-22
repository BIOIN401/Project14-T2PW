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
nothing here imports anything this lane adds (the record key is spelled out).

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
