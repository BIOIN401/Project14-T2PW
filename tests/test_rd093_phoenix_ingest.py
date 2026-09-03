"""D-093 s.5.6 -- Phoenix ingestion mapping. NEW ACCEPTANCE TEST, explicitly labelled.

**G9.** ``rd093_phoenix_ingest.py`` is a new evaluation-only module. It repairs no
pre-existing observable behaviour, so this file is an explicitly labelled NEW
acceptance test with no base-SHA failure proof.

**THESE TESTS RUN IN THE PROJECT VENV, WHICH HAS NO PHOENIX.** That is deliberate and
is the reason the ingest module keeps its attribute mapping in pure functions that
import nothing: ``arize-phoenix`` pulls SQLAlchemy, strawberry-graphql and pandas, and
installing that into the venv every merge gate runs in would put a 508-test gate at the
mercy of a transitive pin. Phoenix lives in a separate eval venv. So the module is
loaded by path and only its pure mapping functions are exercised -- the OTel/Phoenix
imports happen inside ``main()`` and are never reached here.

WHAT IS ACTUALLY BEING DEFENDED. D-093 section 2 DENIED the F-176 runtime change and
required the evaluation layer to report the gate independently, keeping
``runtime_gate_applicable``, ``offline_evaluable`` and ``offline_verdict`` APART. An
ingestion that merged any two of them would re-create the confusion D-092 refused, and
would do it in a dashboard that looks authoritative.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "rd093_phoenix_ingest.py"


@pytest.fixture(scope="module")
def px() -> Any:
    if not INSTRUMENT.is_file():
        pytest.skip(f"instrument not present: {INSTRUMENT}")
    spec = importlib.util.spec_from_file_location("rd093_phoenix_ingest", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rd093_phoenix_ingest"] = module
    spec.loader.exec_module(module)   # must not require phoenix/otel at import time
    return module


def _rec(**over: Any) -> Dict[str, Any]:
    base = {
        "run": "runs/R", "leg_dir": "runs/R/papers/PMC1/strict", "population": "canonical",
        "payload_file": "final_mapped.json", "target_paper": "PMC1", "row_index": 0,
        "reaction_name": "r", "inputs": ["a"], "outputs": ["b"], "enzymes": ["E"],
        "support_class": "target_paper_supported", "support_reason": "because",
        "attribution_tier": "row_lineage", "admission_result": "unavailable",
        "chunk_join_scope": "unavailable", "chunk_join_reaction_specific": False,
        "survives_in_payload": True, "retrieved_span": "unavailable",
        "row_evidence_present": True, "row_evidence_chars": 35029,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The module must import without Phoenix present
# ---------------------------------------------------------------------------

def test_module_imports_without_phoenix_or_otel_installed(px: Any) -> None:
    """If this fails, the gate venv has been coupled to the dashboard's dependencies."""

    assert "phoenix" not in sys.modules
    assert hasattr(px, "reaction_attributes") and hasattr(px, "reintroduction_fields")


# ---------------------------------------------------------------------------
# D-093 s.2 -- the three fields stay three
# ---------------------------------------------------------------------------

def test_the_three_f176_fields_are_distinct_keys(px: Any) -> None:
    assert len({px.RUNTIME_GATE_APPLICABLE, px.OFFLINE_EVALUABLE, px.OFFLINE_VERDICT}) == 3


def test_runtime_gate_applicable_is_always_false_because_the_change_was_denied(px: Any) -> None:
    """D-093 s.2. Production behaviour is unchanged; no span may imply otherwise."""

    for rec in (_rec(), _rec(admission_result="rejected", chunk_join_reaction_specific=True),
                _rec(admission_result="accepted")):
        assert px.reintroduction_fields(rec)[px.RUNTIME_GATE_APPLICABLE] is False


def test_no_admission_record_is_unavailable_and_never_a_pass(px: Any) -> None:
    """"We could not evaluate it" must not be reported as "it passed"."""

    f = px.reintroduction_fields(_rec(admission_result="unavailable"))
    assert f[px.OFFLINE_EVALUABLE] is False
    assert f[px.OFFLINE_VERDICT] == "unavailable"
    assert f[px.OFFLINE_VERDICT] != "no_rejected_candidate_reintroduced"


def test_a_surviving_row_from_a_rejected_reaction_specific_record_is_flagged(px: Any) -> None:
    f = px.reintroduction_fields(_rec(admission_result="rejected",
                                      chunk_join_reaction_specific=True,
                                      survives_in_payload=True))
    assert f[px.OFFLINE_EVALUABLE] is True
    assert f[px.OFFLINE_VERDICT] == "rejected_candidate_reintroduced"


def test_a_rejection_that_is_not_reaction_specific_is_not_a_reintroduction(px: Any) -> None:
    """D-093 condition 1 again: a chunk match that names participants proves nothing."""

    f = px.reintroduction_fields(_rec(admission_result="rejected",
                                      chunk_join_reaction_specific=False))
    assert f[px.OFFLINE_VERDICT] == "no_rejected_candidate_reintroduced"


def test_the_join_scope_travels_with_the_verdict(px: Any) -> None:
    """A cross-run verdict must never be readable as this run's verdict."""

    f = px.reintroduction_fields(_rec(admission_result="rejected",
                                      chunk_join_reaction_specific=True,
                                      chunk_join_scope="cross_run"))
    assert f["eval.chunk_join_scope"] == "cross_run"


# ---------------------------------------------------------------------------
# Vocabulary drift is loud, not silent
# ---------------------------------------------------------------------------

def test_an_unknown_support_class_is_reported_rather_than_ingested(px: Any) -> None:
    assert px.check_vocabulary([_rec()]) == []
    assert px.check_vocabulary([_rec(support_class="probably_fine")]) == ["probably_fine"]


# ---------------------------------------------------------------------------
# Attribute mapping
# ---------------------------------------------------------------------------

def test_evidence_is_carried_as_presence_only_and_labelled_so(px: Any) -> None:
    """The string is 35k chars and silently carries EXTERNAL text (measured fact 4).

    It must never reach a span as target-paper attribution.
    """

    attrs = px.reaction_attributes(_rec())
    assert attrs["t2pw.evidence.present_only"] is True
    assert attrs["t2pw.evidence.chars"] == 35029
    # The blob itself must not be shipped as evidence content.
    assert not any("evidence.content" in k or "evidence.text" in k for k in attrs)


def test_support_class_is_carried_across_unchanged(px: Any) -> None:
    attrs = px.reaction_attributes(_rec(support_class="external_rag_supported"))
    assert attrs["t2pw.support_class"] == "external_rag_supported"


def test_no_retrieval_span_when_nothing_was_recovered(px: Any) -> None:
    """An empty RETRIEVER span reads as "retrieval found nothing" -- a different claim."""

    assert px.retrieval_attributes(_rec(retrieved_span="unavailable")) is None
    assert px.retrieval_attributes(_rec(retrieved_span="")) is None


def test_a_recovered_span_becomes_an_openinference_retriever_document(px: Any) -> None:
    attrs = px.retrieval_attributes(_rec(
        retrieved_span="the span text", retrieval_score=0.88,
        retrieved_chunk_ids=["chunk1"], retrieved_source_paper="PMC999",
        admission_result="rejected", chunk_join_scope="within_run"))
    assert attrs is not None
    assert attrs["openinference.span.kind"] == "RETRIEVER"
    assert attrs["retrieval.documents.0.document.content"] == "the span text"
    assert attrs["retrieval.documents.0.document.score"] == 0.88
    assert attrs["retrieval.documents.0.document.id"] == "chunk1"
    assert "rejected" in attrs["retrieval.documents.0.document.metadata"]


def test_attribute_values_are_otel_safe_scalars_or_string_lists(px: Any) -> None:
    """OTel rejects heterogeneous sequences and nested mappings; those are JSON-encoded."""

    attrs = px.reaction_attributes(_rec(enzymes=[None, "MenI"], origin_stages=["a", "b"]))
    for key, value in attrs.items():
        ok = isinstance(value, (str, bool, int, float)) or (
            isinstance(value, list) and all(isinstance(x, str) for x in value))
        assert ok, f"attribute {key} is not an OTel-safe value: {value!r}"
