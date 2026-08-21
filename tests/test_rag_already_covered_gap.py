"""C-059 -- the RAG gate must not re-import a reaction the paper already locked.

What these tests pin
--------------------
F-057: on both strict legs of ``runs_verify/2026-08-18_1328`` the gap-filling
gate admitted ``Isochorismate to 2,3-Dihydro-2,3-Dihydroxybenzoate (DHB)`` from
PMC12452463 into a payload whose own ``/processes/reactions/1`` already stated
``isochorismate -> {2,3-dihydro-2,3-dihydroxybenzoate, pyruvate}``. The import
carried a second protein row for EntB, that row lost its only reference during
identifier mapping, and ``degree_zero_export:1`` refused the export. Nothing in
the gate refused it, because nothing in the gate ever asked whether the "gap" was
already filled: the seed payload was read only as a set of anchors that WIDEN
admission.

Two independent arms, labelled, because they are graded differently (G9):

* **CORRECTION** -- ``test_correction_*``. Pre-existing observable behaviour,
  proved by replaying the committed artifacts through the real
  ``detect_gaps`` / ``admit_candidates`` path. These tests import no name this
  card introduced, so at the base SHA they RUN and FAIL on a number, which is
  what "fails behaviourally at base" means. ``counts["accepted"]`` is 2 at base
  and 0 here.
* **NEW CAPABILITY** -- ``test_new_capability_*``. The two refusal reason codes
  did not exist in any form; these are labelled new acceptance tests and are
  expected to error at base on the missing name.

Everything else is the empirical safety case merge rule 7 demands
(``test_safety_*``), the F-043 arm, and the non-vacuity arms that show the guard
can fail.

Offline / deterministic: reads only committed artifacts, no network, no LLM. The
``openai`` stub mirrors tests/test_rag_gap_admission.py because importing
``t2pw.rag.retrieve`` pulls ``openai`` at import time.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

from t2pw.rag import admission as admission_mod  # noqa: E402
from t2pw.rag import synonyms  # noqa: E402
from t2pw.rag.admission import (  # noqa: E402
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    AdmissionPolicy,
    RagReactionCandidate,
    admit_candidates,
)
from t2pw.rag.retrieve import Gap, detect_gaps  # noqa: E402

LEG = (
    ROOT / "runs_verify" / "2026-08-18_1328" / "papers" / "PMC12096016" / "strict"
)
REQUESTED_PATHWAY = "enterobactin biosynthesis"
REQUESTED_ORGANISM = "Escherichia coli"
#: The refusal code, spelled out. Tests that must run and pass at the BASE SHA
#: may not import a name this card introduced -- an ImportError proves nothing.
_ALREADY_COVERED_CODE = "gap_already_covered_by_existing_reaction"


# ---------------------------------------------------------------------------
# The committed leg, replayed.
# ---------------------------------------------------------------------------
def _leg(name: str) -> Any:
    return json.loads((LEG / name).read_text(encoding="utf-8"))


def _candidate_from_report_row(row: Dict[str, Any]) -> RagReactionCandidate:
    """Rebuild the candidate the committed admission report describes."""
    return RagReactionCandidate(
        gap_id=row["gap_id"],
        name=row["name"],
        inputs=list(row["inputs"]),
        outputs=list(row["outputs"]),
        enzymes=list(row["enzymes"]),
        reversible=bool(row["reversible"]),
        gap_ids=list(row.get("gap_ids") or []),
        source_paper=dict(row["source_paper"]),
        evidence=dict(row["evidence"]),
        evidence_span=row["evidence"]["span"],
        organism=row.get("organism", ""),
        observed_organisms=list(row.get("observed_organisms") or []),
        observed_pathways=list(row.get("observed_pathways") or []),
        requested_pathway=row.get("requested_pathway", ""),
        requested_organism=row.get("requested_organism", ""),
        requested_pathway_match=row.get("requested_pathway_match", ""),
        organism_match=row.get("organism_match", ""),
        confidence=float(row.get("confidence") or 0.0),
    )


def _replay_the_leg():
    """The real production path over the committed artifacts.

    ``detect_gaps`` with ``reports=None``: the four ``dangling_reaction`` gaps
    are derived from the payload itself, which is what makes the replay possible
    at all -- the report inputs of the original run were never written down.
    """
    payload = _leg("stage1_payload.json")
    committed = _leg("rag_admission_report.json")
    gaps = detect_gaps(
        payload,
        None,
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    candidates = [_candidate_from_report_row(r) for r in committed["accepted"]]
    accepted, report = admit_candidates(
        candidates,
        gaps=gaps,
        seed_payload=payload,
        policy=AdmissionPolicy(**dict(committed["policy"])),
        name_resolver=synonyms.build_offline_synonym_resolver(),
    )
    return payload, committed, gaps, candidates, accepted, report


# ---------------------------------------------------------------------------
# CORRECTION arm (G9 clause 1) -- imports nothing this card introduced.
# ---------------------------------------------------------------------------
def test_correction_the_replay_reproduces_the_committed_run_before_it_judges_it():
    """The instrument first: a replay that does not reproduce the run measures nothing.

    Also the regression tripwire for the gap DETECTOR, which C-059 does not touch:
    ``make_gap_id`` / ``Gap.key`` / ``detect_gaps._add`` are correct, and the four
    ``dangling_reaction`` ids must keep coming out byte-identical. F-057's stated
    mechanism -- "two distinct gap ids for a single dangling edge" -- is false:
    the two ids belong to two ADJACENT reactions, each dangling on a different
    open metabolite, and the per-edge dedup demonstrably fires on a third
    (``EntF``, dangling on two open ends, one id).
    """
    _payload, committed, gaps, _cands, _accepted, _report = _replay_the_leg()

    replayed = sorted(g.gap_id for g in gaps if g.kind == "dangling_reaction")
    from_run = sorted(
        g for g in committed["by_gap"] if g.startswith("gap-dangling_reaction-")
    )
    assert replayed == from_run, (replayed, from_run)
    assert len(replayed) == 4

    # The per-edge dedup fired: EntF is dangling on BOTH enterobactin (terminal)
    # and L-serine (unfed) and still holds exactly one id.
    entf = [g for g in gaps if g.label == "EntF enterobactin synthase reaction"]
    assert len(entf) == 1

    # And the two ids that carried the duplicate are two different reactions.
    labels = {g.gap_id: g.label for g in gaps}
    assert labels["gap-dangling_reaction-555124de"] == "EntC reaction"
    assert labels["gap-dangling_reaction-7e0b4a06"] == "EntB isochorismatase reaction"


def test_correction_the_leg_no_longer_admits_a_reaction_the_paper_already_locked():
    """FAILS AT BASE on the number, not on a missing symbol.

    Base: ``counts["accepted"] == 2`` -- the committed report's own figure,
    reproduced by the replay. Tip: 0.
    """
    _payload, committed, _gaps, candidates, accepted, report = _replay_the_leg()

    assert committed["counts"]["accepted"] == 2, "the committed run admitted two"

    assert report.counts["accepted"] == 0, (
        "the seed payload's /processes/reactions/1 already states "
        "isochorismate -> 2,3-dihydro-2,3-dihydroxybenzoate; re-importing it is "
        "what put a second EntB protein row in the export"
    )
    assert accepted == []
    assert report.counts["considered"] == 2
    assert report.counts["rejected"] == 2
    assert all(c.status == STATUS_REJECTED for c in candidates)
    # Nothing was silently dropped: every refusal is in the report with a reason.
    assert len(report.rejected) == 2
    assert all(row["reasons"] for row in report.rejected)


def test_correction_the_refusal_names_the_existing_reaction_that_covers_it():
    """The reason is auditable: it says WHICH row already covers the claim.

    Behavioural at base too -- at base the candidates are ACCEPTED and carry
    ``fills_named_gap_directly``, so the substring assertion fails on content.
    """
    _payload, _committed, _gaps, candidates, _accepted, _report = _replay_the_leg()
    for candidate in candidates:
        reason = " ".join(candidate.reasons)
        assert "EntB isochorismatase reaction" in reason
        assert "fills_named_gap_directly" not in reason


# ---------------------------------------------------------------------------
# NEW CAPABILITY arm (G9 clause 2) -- explicitly labelled.
# These reason codes did not exist in any form at the base SHA. They are new
# acceptance tests and carry no fabricated base failure.
# ---------------------------------------------------------------------------
def test_new_capability_the_gate_has_a_refusal_code_for_an_already_covered_gap():
    from t2pw.rag.admission import ADMISSION_RULES, REASON_ALREADY_COVERED

    _p, _c, _g, candidates, _a, report = _replay_the_leg()
    assert REASON_ALREADY_COVERED in report.reason_counts
    assert report.reason_counts[REASON_ALREADY_COVERED] == 2
    assert all(
        c.reasons[0].startswith(REASON_ALREADY_COVERED) for c in candidates
    )
    # The rule is published in the report's own rule list, not just enforced.
    assert "gap_not_already_covered" in {name for name, _text in ADMISSION_RULES}


def test_new_capability_one_claim_two_gaps_is_admitted_once_carrying_both_gap_ids():
    """The cross-gap dedup, isolated from the coverage refusal.

    A claim NO existing reaction covers, retrieved for two adjacent gaps whose
    target sets overlap on their shared metabolite. Before: two acceptances.
    Now: one, and it carries both gap ids so neither gap loses its attribution.
    """
    from t2pw.rag.admission import REASON_DUPLICATE_ACROSS_GAPS

    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    dangling = [g for g in gaps if g.kind == "dangling_reaction"]
    assert len(dangling) >= 2, [g.label for g in gaps]
    first, second = dangling[0], dangling[1]

    pair = [
        _novel_candidate(first.gap_id),
        _novel_candidate(second.gap_id),
    ]
    accepted, report = admit_candidates(
        pair, gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert len(accepted) == 1, [c.reasons for c in pair]
    assert accepted[0].status == STATUS_ACCEPTED
    assert set(accepted[0].gap_ids) == {first.gap_id, second.gap_id}
    assert pair[1].status == STATUS_REJECTED
    assert pair[1].reasons[0].startswith(REASON_DUPLICATE_ACROSS_GAPS)
    assert report.counts == {"accepted": 1, "rejected": 1, "considered": 2}


# ---------------------------------------------------------------------------
# Fixtures for the safety case -- a small linear pathway, dangling at both ends.
# ---------------------------------------------------------------------------
def _linear_seed() -> Dict[str, Any]:
    """``alpha -> beta -> gamma + delta``. Dangling on alpha, gamma and delta."""
    return {
        "entities": {
            "compounds": [
                {"name": "alpha"},
                {"name": "beta"},
                {"name": "gamma"},
                {"name": "delta"},
            ],
            "proteins": [{"name": "EnzOne"}, {"name": "EnzTwo"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "first reaction",
                    "inputs": ["alpha"],
                    "outputs": ["beta"],
                    "enzymes": [{"protein": "EnzOne"}],
                },
                {
                    "name": "second reaction",
                    "inputs": ["beta"],
                    "outputs": ["gamma", "delta"],
                    "enzymes": [{"protein": "EnzTwo"}],
                },
            ]
        },
    }


def _policy() -> AdmissionPolicy:
    return AdmissionPolicy(
        organism_policy="allow_unknown", require_pathway_match=False
    ).normalized()


def _make_candidate(
    gap_id: str,
    *,
    name: str,
    inputs: List[str],
    outputs: List[str],
    enzymes: List[str],
    span: str,
    reversible: bool = False,
    chunk_id: str = "chunk-a",
) -> RagReactionCandidate:
    return RagReactionCandidate(
        gap_id=gap_id,
        name=name,
        inputs=list(inputs),
        outputs=list(outputs),
        enzymes=list(enzymes),
        reversible=reversible,
        source_paper={"source_id": "PMC-other", "source_type": "paper"},
        evidence={
            "chunk_id": chunk_id,
            "source_id": "PMC-other",
            "text": span,
            "score": 0.9,
        },
        evidence_span=span,
        observed_pathways=["test pathway"],
        requested_pathway="test pathway",
        requested_pathway_match="match",
        organism_match="match",
        confidence=0.9,
    )


def _novel_candidate(gap_id: str) -> RagReactionCandidate:
    """Genuinely new chemistry touching ``beta`` -- covered by no seed reaction."""
    return _make_candidate(
        gap_id,
        name="beta to epsilon",
        inputs=["beta"],
        outputs=["epsilon"],
        enzymes=["EnzThree"],
        span="EnzThree converts beta into epsilon",
    )


# ---------------------------------------------------------------------------
# EMPIRICAL SAFETY CASE -- merge rule 7.
#
# REV-061 binds: "a filter can only remove readings" is REJECTED as proof by
# itself, because removing a reading can delete a legitimately recovered
# relation. So each way a genuine candidate could be lost is exercised, and each
# one shows it is still admitted.
# ---------------------------------------------------------------------------
def test_safety_a_genuine_gap_filling_reaction_is_still_admitted():
    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    candidate = _novel_candidate(gap.gap_id)
    accepted, _report = admit_candidates(
        [candidate], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert candidate.status == STATUS_ACCEPTED, candidate.reasons
    assert accepted == [candidate]


def test_safety_a_candidate_correcting_an_incomplete_seed_reaction_is_admitted():
    """The case the charter names as the way this card destroys value.

    F-057's seed row was the MORE complete one. The opposite is real: the paper
    states ``alpha -> beta`` and the retrieved passage states
    ``alpha -> beta + omega``. That candidate CORRECTS the pathway and must be
    admitted -- refusing it would delete a legitimate recovery.
    """
    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    candidate = _make_candidate(
        gap.gap_id,
        name="corrected first reaction",
        inputs=["alpha"],
        outputs=["beta", "omega"],
        enzymes=["EnzOne"],
        span="EnzOne converts alpha into beta and omega",
    )
    accepted, _report = admit_candidates(
        [candidate], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert candidate.status == STATUS_ACCEPTED, candidate.reasons
    assert accepted == [candidate]


def test_safety_a_candidate_correcting_the_substrate_side_is_admitted():
    """The mirror of the above, on inputs rather than outputs."""
    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    candidate = _make_candidate(
        gap.gap_id,
        name="first reaction with its missing cosubstrate",
        inputs=["alpha", "omega"],
        outputs=["beta"],
        enzymes=["EnzOne"],
        span="EnzOne converts alpha and omega into beta",
    )
    admit_candidates(
        [candidate], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert candidate.status == STATUS_ACCEPTED, candidate.reasons


def test_safety_a_reversibility_correction_is_admitted():
    """A reversible claim over a one-way seed row is a DIRECTIONAL correction."""
    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    candidate = _make_candidate(
        gap.gap_id,
        name="first reaction is reversible",
        inputs=["alpha"],
        outputs=["beta"],
        enzymes=["EnzOne"],
        span="EnzOne reversibly converts alpha into beta",
        reversible=True,
    )
    admit_candidates(
        [candidate], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert candidate.status == STATUS_ACCEPTED, candidate.reasons


def test_safety_sharing_a_metabolite_with_an_adjacent_gap_is_not_coverage():
    """``beta`` is in two adjacent gaps' target sets. That must refuse nothing.

    This is the structural property of every linear pathway that the F-057
    correction block identifies as the real mechanism, and it is NOT a defect.
    """
    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    dangling = [g for g in gaps if g.kind == "dangling_reaction"]
    survivors = []
    for gap in dangling:
        candidate = _novel_candidate(gap.gap_id)
        admit_candidates(
            [candidate], gaps=gaps, seed_payload=payload, policy=_policy()
        )
        survivors.append(candidate.status)
    assert survivors == [STATUS_ACCEPTED] * len(dangling), survivors


def test_safety_the_reverse_direction_of_a_seed_reaction_is_not_covered():
    """``beta -> alpha`` is a different claim from ``alpha -> beta``.

    Coverage is directional: reading the seed row's outputs as covering a
    candidate's inputs would silently refuse every back-reaction a paper states.
    """
    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    candidate = _make_candidate(
        gap.gap_id,
        name="beta back to alpha",
        inputs=["beta"],
        outputs=["alpha"],
        enzymes=["EnzBack"],
        span="EnzBack converts beta into alpha",
    )
    admit_candidates(
        [candidate], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert candidate.status == STATUS_ACCEPTED, candidate.reasons


def test_safety_a_seed_transport_does_not_cover_a_reaction():
    """Moving A across a membrane is not converting A.

    ``processes.transports`` is excluded from the coverage index on purpose; if
    it were not, a real reaction would be refused on the strength of a real
    transport.
    """
    payload = _linear_seed()
    payload["processes"]["transports"] = [
        {
            "name": "beta export",
            "inputs": ["beta"],
            "outputs": ["epsilon"],
        }
    ]
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    candidate = _novel_candidate(gap.gap_id)
    admit_candidates(
        [candidate], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert candidate.status == STATUS_ACCEPTED, candidate.reasons


def test_safety_a_non_connectivity_gap_is_never_refused_by_this_rule():
    """An ``unmapped_enzyme`` gap asks about catalyst identity, not connectivity.

    The coverage rule is scoped to connectivity gaps, so it may not be what
    refuses a candidate aimed at any other kind. Whatever verdict such a
    candidate gets, it is not this one.

    The code is spelled out rather than imported: this property holds at the base
    SHA too (vacuously -- the rule did not exist there), and a test that ERRORS at
    base on a missing name cannot say so.
    """
    payload = _linear_seed()
    gap = Gap(
        kind="unmapped_enzyme",
        label="EnzOne",
        symbols=["EnzOne"],
        requested_pathway="test pathway",
    )
    candidate = _make_candidate(
        gap.gap_id,
        name="first reaction restated",
        inputs=["alpha"],
        outputs=["beta"],
        enzymes=["EnzOne"],
        span="EnzOne converts alpha into beta",
    )
    admit_candidates(
        [candidate], gaps=[gap], seed_payload=payload, policy=_policy()
    )
    assert not any(
        r.startswith(_ALREADY_COVERED_CODE) for r in candidate.reasons
    ), candidate.reasons


def test_safety_an_empty_seed_payload_refuses_nothing_by_coverage():
    """No index, no coverage verdict. The rule fails OPEN on missing data.

    Spelled out, not imported, for the same reason as the test above.
    """
    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    candidate = _make_candidate(
        gap.gap_id,
        name="first reaction restated",
        inputs=["alpha"],
        outputs=["beta"],
        enzymes=["EnzOne"],
        span="EnzOne converts alpha into beta",
    )
    admit_candidates([candidate], gaps=gaps, seed_payload={}, policy=_policy())
    assert not any(
        r.startswith(_ALREADY_COVERED_CODE) for r in candidate.reasons
    ), candidate.reasons


def test_safety_the_breadth_first_pass_is_unchanged_so_a_chained_child_survives():
    """A hop-1 candidate reachable only THROUGH a covered candidate is kept.

    This is why the refusal runs after the BFS rather than inside phase 1. The
    covered candidate still extended the frontier exactly as it did before; only
    the emitted set shrinks. Folding the refusal into the per-candidate screen
    would have deleted the child too -- a merge-rule-7 loss with no benefit.
    """
    payload = _linear_seed()
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [
        g for g in gaps if g.kind == "dangling_reaction" and "first" in g.label
    ][0]
    # Fills the "first reaction" gap on its target ``beta`` (hop 0), and is
    # covered by the seed's OTHER row, "second reaction": beta -> gamma + delta.
    parent = _make_candidate(
        gap.gap_id,
        name="second reaction restated",
        inputs=["beta"],
        outputs=["gamma"],
        enzymes=["EnzTwo"],
        span="EnzTwo converts beta into gamma",
        chunk_id="chunk-parent",
    )
    # ``gamma`` is NOT in this gap's target set (alpha, beta): the child is
    # reachable only through the metabolite the covered parent contributed.
    child = _make_candidate(
        gap.gap_id,
        name="gamma to zeta",
        inputs=["gamma"],
        outputs=["zeta"],
        enzymes=["EnzFour"],
        span="EnzFour converts gamma into zeta",
        chunk_id="chunk-child",
    )
    accepted, _report = admit_candidates(
        [parent, child], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert parent.status == STATUS_REJECTED
    assert parent.reasons[0].startswith(_ALREADY_COVERED_CODE)
    assert child.status == STATUS_ACCEPTED, child.reasons
    assert accepted == [child]
    assert int(child.chain.get("hop", 0)) == 1


# ---------------------------------------------------------------------------
# F-043 -- the discriminator may not be built from identifier equality, and the
# gloss rule may not collapse the PG / PG phosphate / (PGP) family.
# ---------------------------------------------------------------------------
def test_f043_the_pg_family_does_not_collapse_under_the_gloss_rule():
    """``PG``, ``PG phosphate`` and ``(PGP)`` all carry PathBank 193.

    That accession is UDP-glucose and wrong for all three, which is why a shared
    identifier is evidence and never proof. The three names must stay three
    distinct nodes for the coverage test, and a whole-name parenthetical must not
    be stripped down to nothing.
    """
    forms = admission_mod._gloss_forms
    assert forms("PG") == ("PG",)
    assert forms("PG phosphate") == ("PG phosphate",)
    assert forms("(PGP)") == ("(PGP)",)
    assert forms("2,3-dihydro-2,3-dihydroxybenzoate (DHB)") == (
        "2,3-dihydro-2,3-dihydroxybenzoate (DHB)",
        "2,3-dihydro-2,3-dihydroxybenzoate",
    )

    token = lambda v: admission_mod._fold(admission_mod._canonical(v))  # noqa: E731
    pg = admission_mod._covering_tokens(["PG"], token)
    pgp = admission_mod._covering_tokens(["(PGP)"], token)
    pgphos = admission_mod._covering_tokens(["PG phosphate"], token)
    assert not (pg & pgp)
    assert not (pg & pgphos)
    assert not (pgp & pgphos)


def test_f043_a_shared_accession_does_not_make_a_candidate_covered():
    """The duplicate protein rows in F-057 DO share UniProt P0ADI4.

    The gate is handed a seed row and a candidate that agree on every identifier
    and disagree on chemistry. Coverage must follow the chemistry.
    """
    payload = _linear_seed()
    payload["processes"]["reactions"][0]["mapped_ids"] = {"uniprot": "P0ADI4"}
    payload["processes"]["reactions"][0]["locked_reaction_id"] = "LOCK-1"
    payload["entities"]["proteins"][0]["mapped_ids"] = {"uniprot": "P0ADI4"}
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    # Same catalyst name, same accession on the seed row, and one product the
    # seed row does not have. The chemistry says "extends", so it is admitted --
    # the identifiers, which say "same", are never consulted.
    candidate = _make_candidate(
        gap.gap_id,
        name="first reaction, corrected",
        inputs=["alpha"],
        outputs=["beta", "omega"],
        enzymes=["EnzOne"],
        span="EnzOne converts alpha into beta and omega",
    )
    admit_candidates(
        [candidate], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert candidate.status == STATUS_ACCEPTED, candidate.reasons


def test_the_gloss_head_is_what_matches_never_the_abbreviation():
    """``DHB`` alone must not match ``2,3-dihydro-2,3-dihydroxybenzoate``.

    Matching an abbreviation against a full name is exactly the identifier-shaped
    shortcut F-043 forbids, so the rule fails open there instead.
    """
    payload = _linear_seed()
    payload["processes"]["reactions"][0]["outputs"] = ["beta (BTA)"]
    gaps = detect_gaps(payload, None, requested_pathway="test pathway")
    gap = [g for g in gaps if g.kind == "dangling_reaction"][0]
    # "beta" matches the gloss HEAD of "beta (BTA)" -> covered.
    covered = _make_candidate(
        gap.gap_id,
        name="alpha to beta",
        inputs=["alpha"],
        outputs=["beta"],
        enzymes=["EnzOne"],
        span="EnzOne converts alpha into beta",
    )
    # "BTA" is only the abbreviation -> not matched, admitted.
    abbreviated = _make_candidate(
        gap.gap_id,
        name="alpha to BTA",
        inputs=["alpha"],
        outputs=["BTA"],
        enzymes=["EnzOne"],
        span="EnzOne converts alpha into BTA",
        chunk_id="chunk-b",
    )
    admit_candidates([covered], gaps=gaps, seed_payload=payload, policy=_policy())
    admit_candidates(
        [abbreviated], gaps=gaps, seed_payload=payload, policy=_policy()
    )
    assert covered.status == STATUS_REJECTED, covered.reasons
    assert abbreviated.status == STATUS_ACCEPTED, abbreviated.reasons


# ---------------------------------------------------------------------------
# NON-VACUITY -- the guard must be able to fail.
#
# `_admit_for_gap` and `_dedupe_candidates` had ZERO direct references anywhere
# in tests/ before this file, so nothing pinned this seam. Each arm below removes
# one piece of the guard and shows the leg goes back to what F-057 recorded.
# ---------------------------------------------------------------------------
def test_nonvacuity_neutralizing_the_coverage_test_restores_the_reimport(monkeypatch):
    monkeypatch.setattr(admission_mod, "_already_covered", lambda *a, **k: None)
    _p, _c, _g, _cands, accepted, report = _replay_the_leg()
    assert report.counts["accepted"] == 1, "only the cross-gap dedup is left"
    assert accepted[0].name.startswith("Isochorismate to 2,3-Dihydro")


def test_nonvacuity_neutralizing_the_whole_phase_restores_the_committed_counts(
    monkeypatch,
):
    """Both refusals off == the base SHA's behaviour, on this same tree.

    F-051 governs the shape of this control: the tree, its ``.env`` and the
    artifacts are held constant and only the function under test is swapped.
    """
    monkeypatch.setattr(
        admission_mod,
        "_refuse_covered_and_duplicate",
        lambda accepted, **kwargs: (list(accepted), []),
    )
    _p, committed, _g, _cands, accepted, report = _replay_the_leg()
    assert report.counts["accepted"] == 2 == committed["counts"]["accepted"]
    assert {c.gap_id for c in accepted} == {
        "gap-dangling_reaction-555124de",
        "gap-dangling_reaction-7e0b4a06",
    }


def test_nonvacuity_the_seed_index_is_what_the_refusal_reads():
    """Empty the index and the same candidates are admitted again.

    Distinguishes "the guard refused it" from "something else refused it".
    """
    payload = _leg("stage1_payload.json")
    stripped = json.loads(json.dumps(payload))
    stripped["processes"]["reactions"] = []
    committed = _leg("rag_admission_report.json")
    gaps = detect_gaps(
        payload,
        None,
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    candidates = [_candidate_from_report_row(r) for r in committed["accepted"]]
    _accepted, report = admit_candidates(
        candidates,
        gaps=gaps,
        seed_payload=stripped,
        policy=AdmissionPolicy(**dict(committed["policy"])),
        name_resolver=synonyms.build_offline_synonym_resolver(),
    )
    # The coverage rule contributes nothing; the cross-gap dedup still fires.
    assert report.counts["accepted"] == 1


# ---------------------------------------------------------------------------
# The measured contradiction of the charter's section 2, pinned so it cannot
# drift back into being an assumption.
# ---------------------------------------------------------------------------
def test_the_cross_gap_dedup_is_not_byte_neutral_on_the_payload(monkeypatch):
    """MEASURED: the dedup alone moves ``rag_confidence``, and only that.

    ``_merge_into`` unions ``_Reaction.scores`` and ``_confidence`` takes their
    max, so the committed row carries the higher of the two per-retrieval scores
    (0.930233 at ``merged_payload.json /processes/reactions/5/rag_confidence``).
    Deduping at admission leaves only the surviving candidate's own score,
    0.914815. The ROW is unchanged in every other respect and is not removed --
    removing it is the coverage refusal's doing, which is the whole point of
    separating the two.
    """
    from t2pw.rag import synthesize as synthesize_mod

    merged = _leg("merged_payload.json")
    assert merged["processes"]["reactions"][5]["rag_confidence"] == 0.930233

    resolver = synonyms.build_offline_synonym_resolver()

    def _rows(accepted: List[RagReactionCandidate]) -> List[Dict[str, Any]]:
        reactions = [
            synthesize_mod._Reaction(
                name=c.name,
                inputs=[synthesize_mod._Participant(name=n) for n in c.inputs],
                outputs=[synthesize_mod._Participant(name=n) for n in c.outputs],
                enzymes=list(c.enzymes),
                provenance=[dict(c.evidence)],
                evidence=[dict(c.evidence)],
                source_papers=[dict(c.source_paper)],
                scores=[float(c.evidence.get("score") or 0.0)],
                gap_id=c.gap_id,
                gap_ids=list(c.gap_ids or [c.gap_id]),
                evidence_span=c.evidence_span,
            )
            for c in accepted
        ]
        resolved, _conflicts = synthesize_mod._resolve_reactions(reactions, resolver)
        return [synthesize_mod._reaction_row(r) for r in resolved]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            admission_mod,
            "_refuse_covered_and_duplicate",
            lambda accepted, **kwargs: (list(accepted), []),
        )
        *_head, base_accepted, _base_report = _replay_the_leg()
    base_rows = _rows(base_accepted)

    monkeypatch.setattr(admission_mod, "_already_covered", lambda *a, **k: None)
    *_h, dedup_accepted, _r = _replay_the_leg()
    dedup_rows = _rows(dedup_accepted)

    assert len(base_rows) == len(dedup_rows) == 1
    assert base_rows[0]["rag_confidence"] == 0.930233
    assert dedup_rows[0]["rag_confidence"] == 0.914815
    differing = {
        key
        for key in set(base_rows[0]) | set(dedup_rows[0])
        if base_rows[0].get(key) != dedup_rows[0].get(key)
    }
    # Only the confidence, and the enzyme actor that carries a copy of it.
    assert differing == {"rag_confidence", "enzymes"}
