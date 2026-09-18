"""Admission, deduplication, conflict retention and reconsideration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining.admission import (  # noqa: E402
    ADMITTED_STATUSES,
    EVIDENCE_LEVEL_MULTI_PAPER,
    EVIDENCE_LEVEL_REVIEW_ONLY,
    EVIDENCE_LEVEL_SINGLE_PAPER,
    judge,
)
from t2pw.mining.claims import ClaimStore  # noqa: E402
from t2pw.mining.models import (  # noqa: E402
    DIRECTION_FORWARD,
    EXTRACTION_TRUNCATED,
    EvidenceSpan,
    ExtractedClaim,
    PaperRecord,
    ReactionParticipant,
    STATUS_ACCEPTED,
    STATUS_CANDIDATE,
    STATUS_CONFLICTING,
    STATUS_HYPOTHESIS_ONLY,
    STATUS_INCOMPLETE,
    STATUS_REJECTED,
    provenance_chain_complete,
    reaction_identity,
)
from t2pw.mining.normalize import EntityRegistry  # noqa: E402
from t2pw.mining.species import (  # noqa: E402
    SPECIES_NON_TARGET,
    SPECIES_TARGET_SUPPORTED,
)

TARGET = "arabidopsis thaliana"


def paper(pid="PMC1", evidence_type="primary"):
    return PaperRecord(paper_id=pid, pmcid=pid, title=f"Paper {pid}", evidence_type=evidence_type)


def span(eid="ev_1", pid="PMC1", text="A is converted to B in Arabidopsis thaliana."):
    return EvidenceSpan(evidence_id=eid, paper_id=pid, text=text, section="results")


def claim(
    *,
    subs=("A",),
    prods=("B",),
    pid="PMC1",
    eid="ev_1",
    enzyme="",
    species_state=SPECIES_TARGET_SUPPORTED,
    hypothesis=False,
    compartment="",
    direction=DIRECTION_FORWARD,
    extraction_status="ok",
    round_index=1,
):
    identity = reaction_identity(subs, prods, species=TARGET)
    field_evidence = {"substrates": [eid], "products": [eid], "species": [eid],
                      "direction": [eid]}
    if enzyme:
        field_evidence["enzyme"] = [eid]
    if compartment:
        field_evidence["compartment"] = [eid]
    return ExtractedClaim(
        claim_id=f"cl_{pid}_{identity[-8:]}",
        claim_identity=identity,
        paper_id=pid,
        subject=" + ".join(subs),
        object=" + ".join(prods),
        substrates=[ReactionParticipant(name=s, evidence_ids=[eid]) for s in subs],
        products=[ReactionParticipant(name=p, evidence_ids=[eid]) for p in prods],
        enzyme=enzyme,
        compartment=compartment,
        direction=direction,
        species_text="Arabidopsis thaliana",
        species_state=species_state,
        hypothesis=hypothesis,
        extraction_status=extraction_status,
        field_evidence=field_evidence,
        evidence_ids=[eid],
        first_seen_round=round_index,
        last_updated_round=round_index,
    )


_UNSET = object()


def judged(c, *, store=None, papers=_UNSET, evidence=_UNSET, registry=None, round_index=1):
    # `is _UNSET` rather than a falsiness check: an intentionally EMPTY papers
    # dict is the whole point of the broken-chain test.
    store = store or ClaimStore()
    store.register(c)
    return judge(
        c,
        store=store,
        registry=registry or EntityRegistry(),
        papers={"PMC1": paper()} if papers is _UNSET else papers,
        evidence={"ev_1": span()} if evidence is _UNSET else evidence,
        round_index=round_index,
    ), store


# ---------------------------------------------------------------------------
# The ordered gates
# ---------------------------------------------------------------------------
def test_a_truncated_extraction_is_rejected_before_any_biological_test():
    outcome, _ = judged(claim(extraction_status=EXTRACTION_TRUNCATED))
    assert outcome.decision.status == STATUS_REJECTED
    assert outcome.reaction is None
    assert "truncated" in outcome.decision.reason


def test_a_claim_with_no_resolvable_evidence_is_rejected():
    c = claim()
    c.evidence_ids = ["ev_nonexistent"]
    outcome, _ = judged(c)
    assert outcome.decision.status == STATUS_REJECTED
    assert "paper evidence is the admission basis" in outcome.decision.reason


def test_evidence_pointing_at_an_unknown_paper_breaks_the_chain():
    outcome, _ = judged(claim(), papers={}, evidence={"ev_1": span()})
    assert outcome.decision.status == STATUS_REJECTED
    assert "provenance chain does not close" in outcome.decision.reason


def test_an_unsupported_enzyme_is_stripped_and_the_reaction_still_admitted():
    c = claim()
    c.enzyme = "a synthase nobody evidenced"
    outcome, _ = judged(c)
    assert outcome.decision.status == STATUS_ACCEPTED
    assert "enzyme" in outcome.stripped_fields
    assert outcome.reaction.enzyme_names == []
    assert outcome.reaction.substrate_names == ["A"]


def test_a_claim_with_no_products_is_incomplete_and_reconsiderable():
    c = claim(prods=())
    c.products = []
    outcome, _ = judged(c)
    assert outcome.decision.status == STATUS_INCOMPLETE
    assert outcome.decision.reconsiderable is True
    assert outcome.reaction is None


def test_a_hypothesis_is_quarantined_not_established():
    outcome, _ = judged(claim(hypothesis=True))
    assert outcome.decision.status == STATUS_HYPOTHESIS_ONLY
    assert outcome.reaction is not None
    assert outcome.reaction.admitted is False
    assert outcome.reaction.quarantine_reason == "hypothesis_only"


def test_a_non_target_species_claim_never_enters_the_accepted_graph():
    outcome, _ = judged(claim(species_state=SPECIES_NON_TARGET))
    assert outcome.decision.status == STATUS_CANDIDATE
    assert outcome.admitted is False
    assert outcome.reaction.admitted is False
    assert outcome.reaction.quarantine_reason == f"species:{SPECIES_NON_TARGET}"
    assert "not target-species-supported" in outcome.decision.reason


def test_a_clean_target_species_claim_is_accepted_with_no_database_lookup():
    outcome, _ = judged(claim(enzyme="LOX2"))
    assert outcome.decision.status == STATUS_ACCEPTED
    assert outcome.admitted is True
    assert "no external database lookup" in outcome.decision.reason
    assert outcome.reaction.external_validation == {"status": "not_evaluated"}
    assert outcome.decision.evidence_level == EVIDENCE_LEVEL_SINGLE_PAPER


def test_admitted_statuses_are_exactly_accepted_and_conflicting():
    assert ADMITTED_STATUSES == {STATUS_ACCEPTED, STATUS_CONFLICTING}


# ---------------------------------------------------------------------------
# Deduplication and independent support
# ---------------------------------------------------------------------------
def test_two_papers_reporting_the_same_reaction_keep_both_citations():
    store = ClaimStore()
    registry = EntityRegistry()
    papers = {"PMC1": paper("PMC1"), "PMC2": paper("PMC2")}
    evidence = {"ev_1": span("ev_1", "PMC1"), "ev_2": span("ev_2", "PMC2")}

    first = claim()
    store.register(first)
    judge(first, store=store, registry=registry, papers=papers, evidence=evidence, round_index=1)

    second = claim(pid="PMC2", eid="ev_2")
    store.register(second)
    outcome = judge(
        second, store=store, registry=registry, papers=papers, evidence=evidence, round_index=2
    )

    assert first.claim_identity == second.claim_identity
    assert sorted(store.supporting_papers(first.claim_identity)) == ["PMC1", "PMC2"]
    assert len(store.siblings(first.claim_identity)) == 2
    assert outcome.decision.evidence_level == EVIDENCE_LEVEL_MULTI_PAPER
    assert outcome.reaction.independent_paper_count == 2
    assert outcome.reaction.multi_paper_supported is True
    # Both evidence spans survive; neither citation was collapsed away.
    assert set(store.evidence_ids(first.claim_identity)) == {"ev_1", "ev_2"}


def test_the_same_paper_twice_merges_rather_than_double_counting():
    store = ClaimStore()
    first = claim()
    store.register(first)
    again = claim(eid="ev_1")
    again.evidence_ids = ["ev_1", "ev_1b"]
    stored, is_new = store.register(again)
    assert is_new is False
    assert stored is first
    assert store.supporting_papers(first.claim_identity) == ["PMC1"]
    assert "ev_1b" in stored.evidence_ids


def test_a_review_only_reaction_is_labelled_as_such():
    papers = {"PMC1": paper("PMC1", evidence_type="review")}
    outcome, _ = judged(claim(), papers=papers)
    assert outcome.decision.evidence_level == EVIDENCE_LEVEL_REVIEW_ONLY


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------
def test_two_papers_disagreeing_about_an_enzyme_produce_a_retained_conflict():
    store = ClaimStore()
    registry = EntityRegistry()
    papers = {"PMC1": paper("PMC1"), "PMC2": paper("PMC2")}
    evidence = {"ev_1": span("ev_1", "PMC1"), "ev_2": span("ev_2", "PMC2")}

    first = claim(enzyme="allene oxide synthase")
    store.register(first)
    judge(first, store=store, registry=registry, papers=papers, evidence=evidence, round_index=1)

    second = claim(pid="PMC2", eid="ev_2", enzyme="allene oxide cyclase")
    store.register(second)
    outcome = judge(
        second, store=store, registry=registry, papers=papers, evidence=evidence, round_index=2
    )

    assert outcome.decision.status == STATUS_CONFLICTING
    assert outcome.admitted is True, "a disagreement about an enzyme does not unmake chemistry"
    conflicts = store.conflicts_for(first.claim_identity)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.attribute == "enzyme"
    assert sorted(conflict.paper_ids) == ["PMC1", "PMC2"]
    assert {p["value"] for p in conflict.positions} == {
        "allene oxide synthase", "allene oxide cyclase"
    }
    assert "cross_paper" in conflict.resolution_note
    # The disputed attribute is left unset rather than silently chosen.
    assert outcome.reaction.enzyme_names == []


def test_a_single_paper_disagreeing_with_itself_is_labelled_same_paper():
    store = ClaimStore()
    first = claim(enzyme="enzyme A")
    store.register(first)
    second = claim(enzyme="enzyme B")
    second.claim_id = "cl_other"
    second.paper_id = "PMC1"
    # A second claim record from the same paper only exists through a distinct
    # claim id; force the sibling list to hold both.
    store.claims[second.claim_id] = second
    store.by_identity[second.claim_identity].append(second.claim_id)
    conflicts = store.detect_conflicts(first.claim_identity, round_index=1)
    assert conflicts and "same_paper" in conflicts[0].resolution_note


# ---------------------------------------------------------------------------
# Reconsideration
# ---------------------------------------------------------------------------
def test_a_rejected_claim_stays_reconsiderable_and_its_history_survives():
    store = ClaimStore()
    registry = EntityRegistry()
    papers = {"PMC1": paper("PMC1"), "PMC2": paper("PMC2")}
    evidence = {"ev_1": span("ev_1", "PMC1"), "ev_2": span("ev_2", "PMC2")}

    # Round 2: species is not target-supported -> held as a candidate lead.
    weak = claim(species_state=SPECIES_NON_TARGET, round_index=2)
    store.register(weak)
    store.record_decision(
        judge(weak, store=store, registry=registry, papers=papers, evidence=evidence,
              round_index=2).decision
    )
    assert store.current_status(weak.claim_identity) == STATUS_CANDIDATE
    assert store.reconsiderable(weak.claim_identity) is True

    # Round 4: a second paper supplies the missing species evidence.
    strong = claim(pid="PMC2", eid="ev_2", round_index=4)
    strong.claim_identity = weak.claim_identity
    store.register(strong)
    store.record_decision(
        judge(strong, store=store, registry=registry, papers=papers, evidence=evidence,
              round_index=4).decision
    )

    history = store.history(weak.claim_identity)
    assert [d.status for d in history.decisions] == [STATUS_CANDIDATE, STATUS_ACCEPTED]
    assert history.decisions[1].supersedes == history.decisions[0].decision_id
    narrative = history.narrative()
    assert "round 2" in narrative[0] and "round 4" in narrative[1]
    # Nothing was erased.
    assert len(history.claim_ids) == 2


def test_an_accepted_claim_is_not_silently_re_decided_downward():
    store = ClaimStore()
    c = claim()
    store.register(c)
    outcome, _ = judged(c, store=store)
    store.record_decision(outcome.decision)
    assert store.current_status(c.claim_identity) == STATUS_ACCEPTED
    assert store.reconsiderable(c.claim_identity) is False


# ---------------------------------------------------------------------------
# Provenance chain
# ---------------------------------------------------------------------------
def test_an_accepted_reaction_has_a_complete_provenance_chain():
    outcome, store = judged(claim(enzyme="LOX2"))
    problems = provenance_chain_complete(
        reaction=outcome.reaction,
        claims=store.claims,
        evidence={"ev_1": span()},
        papers={"PMC1": paper()},
    )
    assert problems == []


def test_a_paper_without_a_source_identifier_breaks_the_chain():
    outcome, store = judged(claim())
    bare = PaperRecord(paper_id="PMC1", title="No identifier at all")
    problems = provenance_chain_complete(
        reaction=outcome.reaction,
        claims=store.claims,
        evidence={"ev_1": span()},
        papers={"PMC1": bare},
    )
    assert problems and "no resolvable source identifier" in problems[0]
