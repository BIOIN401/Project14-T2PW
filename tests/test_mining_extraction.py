"""Structured extraction: schema validation, evidence attribution, and the
empty / truncated completion pattern.

The load-bearing property: **a field the evidence does not support never becomes
a claim**, and a failed completion never becomes a reaction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from helpers_mining import empty_model, truncated_model  # noqa: E402
from t2pw.mining.budget import BudgetLedger  # noqa: E402
from t2pw.mining.config import MiningBudgets, MiningModelConfig  # noqa: E402
from t2pw.mining.extraction import (  # noqa: E402
    CLAIM_SCHEMA,
    build_claims,
    extract_claims,
    render_passages,
)
from t2pw.mining.models import (  # noqa: E402
    DIRECTION_UNSPECIFIED,
    EXTRACTION_EMPTY,
    EXTRACTION_OK,
    EXTRACTION_TRUNCATED,
    NON_ADMISSIBLE_EXTRACTION_STATUSES,
    EvidenceSpan,
)
from t2pw.mining.modelclient import (  # noqa: E402
    MiningModelClient,
    looks_truncated,
    parse_json_object,
    validate_against_schema,
)
from t2pw.mining.species import SPECIES_NOT_STATED, SPECIES_TARGET_SUPPORTED  # noqa: E402

TARGET = "arabidopsis thaliana"


def spans():
    return [
        EvidenceSpan(
            evidence_id="ev_one",
            paper_id="PMC1",
            # ORCH-745 F-1: "in Arabidopsis thaliana leaves" (base: "in Arabidopsis
            # thaliana."). A species is supported only by a sentence showing the
            # organism perform the reaction; without the in-planta context the claim
            # of test_each_field_keeps_its_own_evidence is species_ambiguous.
            text="LOX2 converts alpha-linolenic acid to 13-HPOT in Arabidopsis thaliana leaves.",
            section="results",
            locator="results#0",
            chunk_id="c0",
        ),
        EvidenceSpan(
            evidence_id="ev_two",
            paper_id="PMC1",
            text="The reaction occurs in the chloroplast.",
            section="results",
            locator="results#1",
            chunk_id="c1",
        ),
    ]


def client(chat_fn, **model_kwargs):
    ledger = BudgetLedger(MiningBudgets(max_llm_calls=20, max_cost_usd=10.0))
    return MiningModelClient(
        MiningModelConfig(model="test/model", **model_kwargs),
        ledger,
        chat_fn=chat_fn,
        token_stats_fn=lambda: {},
    )


def ok_response(payload):
    def _fn(*_a, **_k):
        return SimpleNamespace(
            text=json.dumps(payload),
            diagnostics=SimpleNamespace(
                finish_reason="stop", attempts=1, model="test/model",
                request_hash="", response_hash="", attempt_log=[],
            ),
        )

    return _fn


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def test_passages_are_rendered_with_citable_ids_and_locators():
    rendered = render_passages(spans(), paper_title="A paper")
    assert "[evidence_id: ev_one]" in rendered
    assert "[evidence_id: ev_two]" in rendered
    assert "results#0" in rendered
    assert "A paper" in rendered


# ---------------------------------------------------------------------------
# Evidence-to-field attribution
# ---------------------------------------------------------------------------
def test_each_field_keeps_its_own_evidence():
    payload = {
        "claims": [
            {
                "substrates": [{"name": "alpha-linolenic acid", "evidence_ids": ["ev_one"]}],
                "products": [{"name": "13-HPOT", "evidence_ids": ["ev_one"]}],
                "enzyme": "LOX2",
                "enzyme_evidence_ids": ["ev_one"],
                "compartment": "chloroplast",
                "compartment_evidence_ids": ["ev_two"],
                "species": "Arabidopsis thaliana",
                "species_evidence_ids": ["ev_one"],
                "evidence_ids": ["ev_one"],
            }
        ]
    }
    claims, dropped = build_claims(
        payload, spans=spans(), paper_id="PMC1", target_species_normalized=TARGET,
        round_index=1, model="test/model",
    )
    assert not dropped
    claim = claims[0]
    assert claim.evidence_for("enzyme") == ["ev_one"]
    assert claim.evidence_for("compartment") == ["ev_two"]
    assert claim.evidence_for("species") == ["ev_one"]
    assert claim.species_state == SPECIES_TARGET_SUPPORTED
    assert claim.unsupported_fields() == []


def test_an_uncited_field_is_dropped_not_believed():
    """The passage supports substrate and product but not the enzyme."""
    payload = {
        "claims": [
            {
                "substrates": [{"name": "A", "evidence_ids": ["ev_one"]}],
                "products": [{"name": "B", "evidence_ids": ["ev_one"]}],
                "enzyme": "a plausible-sounding synthase",
                "compartment": "mitochondrion",
                "direction": "reverse",
                "reversible": True,
                "evidence_ids": ["ev_one"],
            }
        ]
    }
    claims, dropped = build_claims(
        payload, spans=spans(), paper_id="PMC1", target_species_normalized=TARGET,
        round_index=1, model="test/model",
    )
    claim = claims[0]
    assert claim.enzyme == ""
    assert claim.compartment == ""
    assert claim.direction == DIRECTION_UNSPECIFIED
    assert claim.reversible is None
    # The reaction itself survives -- an unsupported enzyme is not a reason to
    # throw away supported chemistry.
    assert claim.substrate_names == ["A"] and claim.product_names == ["B"]
    assert len(dropped) == 4
    assert any("enzyme" in d for d in dropped)


def test_a_citation_to_an_id_that_was_never_shown_is_rejected():
    payload = {
        "claims": [
            {
                "substrates": [{"name": "A", "evidence_ids": ["ev_fabricated"]}],
                "products": [{"name": "B", "evidence_ids": ["ev_one"]}],
                "enzyme": "X",
                "enzyme_evidence_ids": ["ev_also_fabricated"],
                "evidence_ids": ["ev_one", "ev_fabricated"],
            }
        ]
    }
    claims, dropped = build_claims(
        payload, spans=spans(), paper_id="PMC1", target_species_normalized=TARGET,
        round_index=1, model="test/model",
    )
    claim = claims[0]
    assert "ev_fabricated" not in claim.evidence_ids
    assert claim.enzyme == ""
    assert any("enzyme" in d for d in dropped)


def test_species_not_stated_when_nothing_names_one():
    payload = {
        "claims": [
            {
                "substrates": [{"name": "A", "evidence_ids": ["ev_one"]}],
                "products": [{"name": "B", "evidence_ids": ["ev_one"]}],
                "evidence_ids": ["ev_one"],
            }
        ]
    }
    claims, _ = build_claims(
        payload, spans=spans(), paper_id="PMC1", target_species_normalized=TARGET,
        round_index=1, model="test/model",
    )
    assert claims[0].species_state == SPECIES_NOT_STATED


def test_a_papers_observed_species_is_never_the_claims_species():
    """ORCH-745 B3, a deliberate move: base fed the screen's paper-level organisms in
    as the claim's species, so this claim was ``species_ambiguous``. A paper-level
    organism is a fact about the paper, not about this reaction: ``species_not_stated``."""
    payload = {
        "claims": [
            {
                "substrates": [{"name": "A", "evidence_ids": ["ev_one"]}],
                "products": [{"name": "B", "evidence_ids": ["ev_one"]}],
                "evidence_ids": ["ev_one"],
            }
        ]
    }
    claims, _ = build_claims(
        payload, spans=spans(), paper_id="PMC1", target_species_normalized=TARGET,
        round_index=1, model="test/model",
        paper_observed_organisms=["Arabidopsis thaliana"],
    )
    assert claims[0].species_state == SPECIES_NOT_STATED
    assert claims[0].species_text == ""


def test_a_hypothesis_is_carried_as_a_hypothesis():
    payload = {
        "claims": [
            {
                "substrates": [{"name": "A", "evidence_ids": ["ev_one"]}],
                "products": [{"name": "B", "evidence_ids": ["ev_one"]}],
                "hypothesis": True,
                "evidence_ids": ["ev_one"],
            }
        ]
    }
    claims, _ = build_claims(
        payload, spans=spans(), paper_id="PMC1", target_species_normalized=TARGET,
        round_index=1, model="test/model",
    )
    assert claims[0].hypothesis is True


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def test_schema_validation_accepts_a_conforming_payload():
    payload = {"claims": [{"substrates": [{"name": "A"}], "products": [{"name": "B"}],
                           "evidence_ids": ["ev_one"]}]}
    assert validate_against_schema(payload, CLAIM_SCHEMA) == ""


def test_schema_validation_rejects_a_non_conforming_payload():
    assert validate_against_schema({"claims": "not a list"}, CLAIM_SCHEMA) != ""
    assert validate_against_schema([], CLAIM_SCHEMA) != ""


def test_schema_error_yields_no_claims():
    result = extract_claims(
        client=client(ok_response({"claims": "not a list"})),
        spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=1,
    )
    assert result.claims == []
    assert result.status in NON_ADMISSIBLE_EXTRACTION_STATUSES


# ---------------------------------------------------------------------------
# Empty / truncated / malformed completions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"claims": [{"substrates": [{"name": "x"', True),
        ('{"claims": []}', False),
        ("[1, 2", True),
        ("not json at all", False),
        ("", False),
    ],
)
def test_truncation_is_detected_by_bracket_balance(raw, expected):
    assert looks_truncated(raw) is expected


def test_lenient_parsing_does_not_repair_truncated_json():
    payload, error = parse_json_object('{"claims": [{"substrates": [{"name": "x"')
    assert payload is None
    assert error


def test_a_code_fence_is_stripped_but_content_is_not_invented():
    payload, error = parse_json_object('```json\n{"claims": []}\n```')
    assert payload == {"claims": []} and error == ""


def test_an_empty_completion_produces_no_claims_and_is_recorded():
    result = extract_claims(
        client=client(empty_model),
        spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=1,
    )
    assert result.claims == []
    assert result.status == EXTRACTION_EMPTY
    assert result.outcome is not None
    assert result.outcome.record.status == EXTRACTION_EMPTY
    assert "no admissible claims" in result.notes[0]


def test_a_truncated_completion_produces_no_claims_and_is_recorded():
    result = extract_claims(
        client=client(truncated_model, max_attempts=2),
        spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=1,
    )
    assert result.claims == []
    assert result.status == EXTRACTION_TRUNCATED
    record = result.outcome.record
    assert record.finish_reason == "length"
    assert record.status == EXTRACTION_TRUNCATED


def test_a_retry_shrinks_the_passage_rather_than_resending_it():
    """An identical re-send buys latency and the same failure."""
    sent = []

    def recording(messages, **_kwargs):
        user = "".join(m["content"] for m in messages if m["role"] == "user")
        sent.append(len(user))
        return truncated_model()

    result = extract_claims(
        client=client(recording, max_attempts=3),
        spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=1,
    )
    assert result.claims == []
    assert len(sent) == 3
    assert sent[0] > sent[1] > sent[2]


def test_a_parseable_but_length_stopped_payload_is_diagnostic_only():
    """``finish_reason=length`` on a payload that happens to parse is not a result."""
    def lucky(*_a, **_k):
        return SimpleNamespace(
            text='{"claims": []}',
            diagnostics=SimpleNamespace(
                finish_reason="length", attempts=1, model="test/model",
                request_hash="", response_hash="", attempt_log=[],
            ),
        )

    result = extract_claims(
        client=client(lucky, max_attempts=1),
        spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=1,
    )
    assert result.claims == []
    assert result.status == "partial_repair_diagnostic_only"
    assert result.status in NON_ADMISSIBLE_EXTRACTION_STATUSES


def test_a_provider_exception_is_an_outcome_not_a_crash():
    def boom(*_a, **_k):
        raise RuntimeError("provider exploded")

    result = extract_claims(
        client=client(boom, max_attempts=1),
        spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=1,
    )
    assert result.claims == []
    assert "provider exploded" in result.outcome.record.parse_error


def test_a_configured_fallback_is_explicit_and_logged():
    calls = []

    def flaky(messages, **kwargs):
        calls.append(kwargs.get("model_override"))
        if kwargs.get("model_override") == "test/model":
            return empty_model()
        return ok_response({"claims": []})()

    result = extract_claims(
        client=client(flaky, max_attempts=1, fallback_model="vendor/backup"),
        spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=1,
    )
    assert "vendor/backup" in calls
    record = result.outcome.record
    assert record.fallback_used is True
    assert "vendor/backup" in record.fallback_reason
    assert record.requested_model == "test/model"


def test_no_fallback_means_the_failure_stays_loud():
    result = extract_claims(
        client=client(empty_model, max_attempts=1),
        spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=1,
    )
    assert result.outcome.record.fallback_used is False
    assert result.status == EXTRACTION_EMPTY


def test_every_call_is_priced_and_recorded():
    c = client(ok_response({"claims": []}))
    extract_claims(
        client=c, spans=spans(), paper_id="PMC1", paper_title="t",
        target_species="Arabidopsis thaliana", target_species_normalized=TARGET,
        round_index=2,
    )
    record = c.records[0]
    assert record.round_index == 2
    assert record.purpose == "extract_claims"
    assert record.requested_model == "test/model"
    assert record.input_tokens > 0
    assert record.estimated_cost_usd >= 0.0
    assert c.ledger.usage.llm_calls == 1
    assert record.status == EXTRACTION_OK
