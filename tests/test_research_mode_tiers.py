from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.rag.tiers import (  # noqa: E402
    REVIEW_SOURCE_LABEL,
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_D,
    assign_tiers,
)


# ---------------------------------------------------------------------------
# Payload builders (copied, not shared -- house style in this suite).
# ---------------------------------------------------------------------------
def _prov(source_id: str, *, chunk_id: str = "", title: str = "", section: str = "results") -> dict:
    return {
        "source_id": source_id,
        "source_title": title or source_id,
        "source_type": "paper",
        "source_uri": f"https://example.org/{source_id}",
        "section": section,
        "organism": "lactococcus lactis",
        "chunk_id": chunk_id,
    }


def _evidence(source_id: str, *, chunk_id: str, text: str) -> dict:
    return {
        "text": text,
        "source_id": source_id,
        "source_uri": f"https://example.org/{source_id}",
        "section": "results",
        "score": 0.83,
        "chunk_id": chunk_id,
    }


def _payload(*, proteins=None, compounds=None, reactions=None) -> dict:
    return {
        "entities": {
            "species": [{"name": "Lactococcus lactis"}],
            "compounds": list(compounds or []),
            "proteins": list(proteins or []),
        },
        "processes": {"reactions": list(reactions or [])},
    }


def _grounded_protein() -> dict:
    return {
        "name": "MenG",
        "mapped_ids": {"uniprot": "P0A6C0"},
        "rag_provenance": _prov("PMC1", chunk_id="a" * 32),
        "source_refs": ["PMC1"],
        "rag_confidence": 0.83,
    }


def _unknown_sentinel_protein() -> dict:
    return {
        "name": "Unknown",
        "pathbank_protein_id": 9659,
        "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
        "mapping_meta": {
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "cross_species_placeholder": True,
            "confidence": 0.0,
        },
        "rag_provenance": _prov("PMC1", chunk_id="a" * 32),
        "source_refs": ["PMC1"],
    }


def _best_effort_protein() -> dict:
    return {
        "name": "MenH",
        "mapped_ids": {"uniprot": "Q9XYZ1"},
        "mapping_meta": {
            "chosen_rule": "best_effort_fallback",
            "best_effort": True,
            "confidence": 0.41,
        },
        "rag_provenance": _prov("PMC1", chunk_id="a" * 32),
        "source_refs": ["PMC1"],
    }


def _corroborated_enzyme_payload(corroborator: str = "PMC2") -> dict:
    """Enzyme with no id, stated in the seed AND in one distinct retrieved paper."""

    actor = {
        "entity": "MenG",
        "entity_type": "protein",
        "role": "catalyst",
        "rag_provenance": _prov("seed_paper", title="uploaded seed paper", section=""),
        "evidence": [
            {"text": "MenG methylates demethylmenaquinone."},
            _evidence(corroborator, chunk_id="b" * 32, text="MenG is the C-methyltransferase."),
        ],
        "source_papers": [
            {"source_id": "seed_paper", "title": "uploaded seed paper", "uri": ""},
            {"source_id": corroborator, "title": "Menaquinone biosynthesis", "uri": "https://example.org/PMC2"},
        ],
        "source_refs": ["seed_paper", corroborator],
        "rag_confidence": 0.6,
    }
    reaction = {
        "name": "demethylmenaquinone -> menaquinone",
        "inputs": ["demethylmenaquinone"],
        "outputs": ["menaquinone"],
        "enzymes": [actor],
        "rag_provenance": _prov("seed_paper", title="uploaded seed paper", section=""),
        "evidence": list(actor["evidence"]),
        "source_refs": ["seed_paper", corroborator],
    }
    return _payload(reactions=[reaction])


def _rag_result(doc_types: dict) -> SimpleNamespace:
    scores = {
        paper_id: SimpleNamespace(paper_id=paper_id, score=0.5, document_type=doc_type)
        for paper_id, doc_type in doc_types.items()
    }
    return SimpleNamespace(selection=SimpleNamespace(scores=scores))


def _exploding_resolver(name, organism):  # pragma: no cover - must never run
    raise AssertionError(f"network resolver called for {name!r}/{organism!r}")


def _only(report, pointer):
    return report.by_pointer[pointer]


# ---------------------------------------------------------------------------
# Tier A and its two poison sites.
# ---------------------------------------------------------------------------
def test_db_grounded_enzyme_is_tier_a():
    report = assign_tiers(_payload(proteins=[_grounded_protein()]))
    assignment = _only(report, "/entities/proteins/0")
    assert assignment.tier == TIER_A
    assert assignment.identifier == "P0A6C0"
    assert assignment.identifier_source == "uniprot"
    assert report.counts[TIER_A] == 1


def test_pathbank_unknown_sentinel_is_not_tier_a():
    report = assign_tiers(_payload(proteins=[_unknown_sentinel_protein()]))
    assignment = _only(report, "/entities/proteins/0")
    assert assignment.tier != TIER_A
    assert assignment.identifier == ""
    assert any("identity rejected" in note for note in assignment.notes)
    assert "identity rejected" in assignment.reason


def test_best_effort_mapping_is_not_tier_a():
    report = assign_tiers(_payload(proteins=[_best_effort_protein()]))
    assignment = _only(report, "/entities/proteins/0")
    assert assignment.tier != TIER_A
    assert assignment.identifier == ""
    assert any("best_effort" in note for note in assignment.notes)


# ---------------------------------------------------------------------------
# Tier B / C -- evidence, never identity.
# ---------------------------------------------------------------------------
def test_seed_plus_distinct_retrieved_paper_is_tier_b():
    payload = _corroborated_enzyme_payload()
    report = assign_tiers(payload, rag_result=_rag_result({"PMC2": "primary_research"}))
    assignment = _only(report, "/processes/reactions/0/enzymes/0")
    assert assignment.tier == TIER_B
    assert assignment.identifier == ""  # corroboration never invents an accession
    assert assignment.distinct_paper_count == 2
    assert {s["source_id"] for s in assignment.sources} == {"seed_paper", "PMC2"}
    assert not assignment.is_review_only
    quoted = [s for s in assignment.sources if s["retrieved"]]
    assert len(quoted) == 1 and quoted[0]["quote"]


def test_review_only_corroborator_is_tier_c_and_labelled_review():
    payload = _corroborated_enzyme_payload()
    report = assign_tiers(payload, rag_result=_rag_result({"PMC2": "multi_example_review"}))
    assignment = _only(report, "/processes/reactions/0/enzymes/0")
    assert assignment.tier == TIER_C
    assert assignment.is_review_only
    assert "review" in assignment.reason
    labels = {s["label"] for s in assignment.sources}
    assert REVIEW_SOURCE_LABEL in labels


def test_single_source_entity_is_tier_c():
    entity = {
        "name": "menaquinone",
        "rag_provenance": _prov("PMC1", chunk_id="c" * 32),
        "source_papers": [{"source_id": "PMC1", "title": "One paper", "uri": ""}],
        "source_refs": ["PMC1"],
    }
    report = assign_tiers(_payload(compounds=[entity]), rag_result=_rag_result({"PMC1": "primary_research"}))
    assignment = _only(report, "/entities/compounds/0")
    assert assignment.tier == TIER_C
    assert assignment.distinct_paper_count == 1


def test_entity_with_no_provenance_is_tier_d():
    report = assign_tiers(_payload(compounds=[{"name": "orphan compound"}]))
    assignment = _only(report, "/entities/compounds/0")
    assert assignment.tier == TIER_D
    assert assignment.label == "UNSOURCED (review)"
    assert assignment.sources == []
    assert report.review_required is True


def test_quoted_source_ref_cannot_manufacture_corroboration():
    """A verbatim quote in source_refs becomes a fake source_id -- never a paper."""

    entity = {
        "name": "menaquinone",
        "rag_provenance": _prov("seed_paper", title="uploaded seed paper", section=""),
        "source_refs": [
            "seed_paper",
            "MenG converts demethylmenaquinone to menaquinone in L. lactis.",
        ],
    }
    report = assign_tiers(_payload(compounds=[entity]))
    assignment = _only(report, "/entities/compounds/0")
    assert assignment.tier == TIER_C
    assert assignment.distinct_paper_count == 1


# ---------------------------------------------------------------------------
# Side-car and hermeticity guarantees.
# ---------------------------------------------------------------------------
def test_payload_is_not_mutated():
    payload = _corroborated_enzyme_payload()
    payload["entities"]["proteins"] = [_grounded_protein(), _unknown_sentinel_protein()]
    before = copy.deepcopy(payload)
    report = assign_tiers(payload, rag_result=_rag_result({"PMC1": "primary_research"}))
    assert payload == before
    assert report.assignments  # the side-car carries everything instead


def test_online_off_never_touches_the_network():
    payload = _payload(proteins=[{"name": "MenG", "rag_provenance": _prov("PMC1", chunk_id="d" * 32)}])
    report = assign_tiers(payload, resolver=_exploding_resolver, online=False)
    assert _only(report, "/entities/proteins/0").tier != TIER_A


def test_online_retry_is_read_only_and_swallows_failures():
    calls = []

    def _boom(name, organism):
        calls.append((name, organism))
        raise RuntimeError("uniprot unreachable")

    payload = _payload(proteins=[{"name": "MenG", "rag_provenance": _prov("PMC1", chunk_id="d" * 32)}])
    before = copy.deepcopy(payload)
    report = assign_tiers(payload, resolver=_boom, online=True)
    assignment = _only(report, "/entities/proteins/0")
    assert calls == [("MenG", "lactococcus lactis")]
    assert assignment.tier == TIER_C
    assert any("UniProt retry failed" in note for note in assignment.notes)
    assert payload == before


def test_online_retry_rejects_best_effort_result():
    def _guess(name, organism):
        return {
            "status": "mapped",
            "reason": "best_effort_fallback",
            "chosen_rule": "best_effort_fallback",
            "best_effort": True,
            "mapped_ids": {"uniprot": "Q9XYZ1"},
        }

    payload = _payload(proteins=[{"name": "MenG", "rag_provenance": _prov("PMC1", chunk_id="d" * 32)}])
    assignment = _only(assign_tiers(payload, resolver=_guess, online=True), "/entities/proteins/0")
    assert assignment.tier != TIER_A
    assert assignment.identifier == ""
    assert any("best_effort" in note for note in assignment.notes)


def test_online_retry_accepts_a_confident_match():
    def _hit(name, organism):
        return {
            "status": "mapped",
            "chosen_rule": "top_unique_candidate",
            "confidence": 0.91,
            "mapped_ids": {"uniprot": "P0A6C0"},
        }

    payload = _payload(proteins=[{"name": "MenG", "rag_provenance": _prov("PMC1", chunk_id="d" * 32)}])
    assignment = _only(assign_tiers(payload, resolver=_hit, online=True), "/entities/proteins/0")
    assert assignment.tier == TIER_A
    assert assignment.identifier == "P0A6C0"
    assert assignment.identifier_source == "uniprot:online_retry"


def test_report_to_dict_is_json_shaped():
    report = assign_tiers(
        _corroborated_enzyme_payload(),
        rag_result=_rag_result({"PMC2": "primary_research"}),
        mode="research",
    )
    data = report.to_dict()
    assert data["mode"] == "research"
    assert data["online"] is False
    assert set(data["counts"]) == {TIER_A, TIER_B, TIER_C, TIER_D}
    assert all(isinstance(item["pointer"], str) for item in data["assignments"])
    assert data["tier_labels"][TIER_D] == "UNSOURCED (review)"


@pytest.mark.parametrize("payload", [None, {}, {"entities": [], "processes": None}])
def test_degenerate_payloads_do_not_raise(payload):
    report = assign_tiers(payload)
    assert report.assignments == []
    assert report.counts[TIER_A] == 0
