"""C-071 / F-079: an actor its own cited span does not name stops being releasable.

``runs_verify/2026-08-18_1328/papers/PMC12096016/research`` shipped ``release_ready`` /
``semantic_evaluation: passed`` / ``strict_acceptance_eligible: true`` while carrying
``transporters: [{"entity": "EntE"}]`` on a span reading *"secreted ... by a
**TolC**-dependent process"*. Re-measured at the base SHA the classifier still answers
exactly that (``evidence/c071_g9_base.json``), so the first test is a **G9 CORRECTION** of
pre-existing observable behaviour, not a new capability, and fails behaviourally at base.

The rest are labelled **NEW ACCEPTANCE**: the detector's own semantics have no base
behaviour to regress, so no base failure is claimed for them. Both obligations apply and
neither substitutes for the other -- the correction proves the defect is closed, the
acceptance tests pin *how*, so a later edit cannot keep the leg demoted while quietly
changing the rule that demoted it.

**F-053 is respected, not tested.** Nothing here reads ``semantic_evaluation == "passed"``
affirmatively or builds a rate on it: the defect produces ``SEMANTIC_FAILED`` through a
gating check and the pre-existing subtractive cap does the demotion.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench import semantic as _s  # noqa: E402
from t2pw.bench import semantic_production as sp  # noqa: E402
from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    REASON_SEMANTIC_EVALUATION_FAILED,
    RELEASE_READY,
    REVIEW_REQUIRED,
    SEMANTIC_FAILED,
    SEMANTIC_GATING_CHECKS,
    SEMANTIC_NOT_EVALUATED,
    SEMANTIC_PASSED,
    classify_release_status,
    semantic_verdict,
)

#: F-079's leg, and the Stage-0 context that run recorded for it. The two values are
#: ``observed_pathways[0]`` / ``observed_organisms[0]`` of the committed
#: ``manifest.jsonl`` row, restated under the keys the PINNED derivation reads.
RESEARCH_LEG = (
    ROOT / "runs_verify" / "2026-08-18_1328" / "papers" / "PMC12096016" / "research"
    / "final_mapped.json"
)
OBSERVED_CONTEXT = {
    "pathway_name": "enterobactin biosynthesis",
    "likely_organism": "Escherichia coli",
}

#: A coverage verdict that on its own classifies ``release_ready``, so that when a run
#: below lands anywhere else the semantic cap is the only thing that could have moved it.
_COVERAGE: Dict[str, Any] = {
    "requested_core_declared": True, "surviving_processes": 3,
    "minimum_core_satisfied": True, "reasons": [],
}


def _evaluate(payload: Any, **kwargs: Any) -> Any:
    requested = pathway_context_from_stage_zero(dict(OBSERVED_CONTEXT))
    kwargs.setdefault("requested_pathway", requested.requested_pathway)
    kwargs.setdefault("requested_organism", requested.organism)
    kwargs.setdefault("mode", "research")
    return sp.evaluate_production_semantics(payload, **kwargs)


def _release(payload: Any, **kwargs: Any) -> Any:
    """The production chain, end to end: report -> verdict -> release status."""

    evaluation, reason, failed, evaluability = semantic_verdict(_evaluate(payload, **kwargs))
    return classify_release_status(
        _COVERAGE, strict_gates_passed=True,
        semantic_evaluation=evaluation, semantic_not_evaluated_reason=reason,
        semantic_failed_checks=failed, semantic_check_evaluability=evaluability,
    )


def _actor_payload(entity: str, span: str, bucket: str = "enzymes") -> Dict[str, Any]:
    """One retained reaction, one actor, the span that actor cites. Everything else is
    deliberately clean, so the only gating check that can move is the one under test."""

    return {
        "entities": {"compounds": [{"name": "enterobactin"}]},
        "processes": {"reactions": [{
            "name": "enterobactin biosynthesis step",
            "inputs": ["2,3-dihydroxybenzoate"], "outputs": ["enterobactin"],
            "organism": "Escherichia coli", "evidence": span,
            bucket: [{"entity": entity, "entity_type": "protein", "role": "catalyst",
                      "evidence": span}],
        }]},
    }


# ── G9 CORRECTION — fails behaviourally at the base SHA ──────────────────────


def test_correction_the_committed_f079_leg_is_no_longer_release_ready() -> None:
    """**FAILS BEHAVIOURALLY AT THE BASE SHA**, on the committed payload, through the
    production entry point.

    Measured at ``f2a959ff`` on the same bytes (``payload_sha256`` recorded on both probe
    arms): ``passed`` / ``release_ready`` / eligible / no failed checks. The first
    assertion below is therefore an assertion failure at base with every symbol it touches
    present there; symbol absence is not offered as proof and none is needed. The leg
    lands in ``review_required`` carrying every process it arrived with -- invariant 2 /
    merge rule 7, preserved for review rather than deleted.
    """

    payload = json.loads(RESEARCH_LEG.read_text(encoding="utf-8"))
    report = _evaluate(payload)
    evaluation, _reason, failed, _evaluability = semantic_verdict(report)

    # ── THE BASE FAILURE. ``passed`` at base, ``failed`` at the tip. ────────
    assert evaluation != SEMANTIC_PASSED, (
        "a payload whose transporter cites a span naming TolC while claiming EntE "
        "is still semantically passed"
    )
    assert evaluation == SEMANTIC_FAILED
    assert _s.CHECK_ACTOR_EVIDENCE in failed

    status = _release(payload)
    assert status.status == REVIEW_REQUIRED
    assert status.strict_acceptance_eligible is False
    assert any(
        reason.startswith(REASON_SEMANTIC_EVALUATION_FAILED) for reason in status.reasons
    ), status.reasons
    assert _s.CHECK_ACTOR_EVIDENCE in " ".join(status.reasons)

    # The finding names the row, so a reviewer is told WHICH actor and WHICH span.
    pointers = {
        finding["pointer"] for finding in report.checks[_s.CHECK_ACTOR_EVIDENCE].findings
    }
    assert "/processes/transports/0/transporters/0" in pointers, pointers

    # Preserved, not dropped: review_required is a classification, not a deletion.
    assert payload["processes"]["transports"], "the leg lost the transport it was demoted for"
    assert len(payload["processes"]["reactions"]) == 5


# ── NEW ACCEPTANCE — the detector's own semantics have no base behaviour ─────


# NEW ACCEPTANCE — this check does not exist at the base SHA
def test_new_acceptance_the_rule_fires_on_a_swapped_actor_and_not_on_a_renamed_one() -> None:
    """Self-discriminating: over-firing fails this as surely as never firing. Every span
    is a real committed ``final_mapped.json`` row. ``must_fail`` are actors whose span
    names some OTHER protein; ``must_pass`` are the same protein under a canonical,
    wrapper or DB name while the span uses the paper's symbol."""

    must_fail = (
        ("EntE", "Enterobactin is produced in the cytoplasm and secreted to the "
                 "extracellular environment by a TolC-dependent process"),
        ("EntE", "the ferric-enterobactin complex is then imported via TonB-dependent uptake"),
        ("EntA", "2,3-DHB is produced from chorismate by sequential reactions catalyzed "
                 "by three enzymes"),
        ("ALAS2 complex", "rate-limiting step for heme biosynthesis"),
    )
    must_pass = (
        ("EntC", "EntC-catalyzed isomerization of chorismate to isochorismate"),
        ("Serine hydroxymethyltransferase, mitochondrial",
         "serine is cleaved to glycine in the reaction catalyzed by serine "
         "hydroxymethyltransferase"),
        ("ALAS2 complex", "ALAS2 mediates the condensation of glycine and succinyl-CoA"),
        ("oxidoreductase (entA)", "EntA (2,3-dihydro-2,3-dihydroxybenzoate dehydrogenase; "
                                  "EC 1.3.1.28)"),
    )

    for entity, span in must_fail:
        check = _evaluate(_actor_payload(entity, span)).checks[_s.CHECK_ACTOR_EVIDENCE]
        assert check.applicable is True, (entity, span)
        assert check.ok is False, f"{entity!r} must not be corroborated by {span!r}"
        assert check.findings[0]["entity"] == entity
    for entity, span in must_pass:
        check = _evaluate(_actor_payload(entity, span)).checks[_s.CHECK_ACTOR_EVIDENCE]
        assert check.applicable is True, (entity, span)
        assert check.ok is True, f"{entity!r} IS named by {span!r}; demoting it over-fires"


# NEW ACCEPTANCE — this check does not exist at the base SHA
def test_new_acceptance_whole_token_boundaries_are_what_catch_the_fabricated_actor() -> None:
    """``EntE`` is a substring of ``enterobactin``, so a ``name in span`` test reports
    F-079's transporter as corroborated -- over the 21 committed legs it accepts **20 of
    221** rows this rule rejects, every one an ``Ent*`` symbol swallowed by the word
    ``enterobactin``. Fails if a later edit relaxes the comparison back to a substring."""

    entity, span = "EntE", ("Enterobactin is produced in the cytoplasm and secreted to "
                            "the extracellular environment by a TolC-dependent process")
    assert entity.casefold() in span.casefold(), "the premise: a substring test accepts it"
    assert sp._actor_named_in_span(entity, [span]) is False

    # ...and the boundary is not merely a prefix rule: the token must match in full.
    assert sp._actor_named_in_span("EntE", ["EntF-catalyzed condensation"]) is False
    assert sp._actor_named_in_span("EntE", ["adenylated by EntE in an ATP-dependent step"]) is True


# NEW ACCEPTANCE — this check does not exist at the base SHA
def test_new_acceptance_an_actor_with_no_cited_span_is_neither_a_pass_nor_a_failure() -> None:
    """The third disposition, and why it is not the other two. Failing a spanless actor
    would enforce ``CHECK_SOURCE_CARRIER``'s deliberately NON-gating hygiene question
    through a gate, against its own stated meaning; passing one would make deleting the
    evidence the cheapest way past the gate. So it is counted, not examined -- and when
    nothing at all was examinable the check reports NOT EVALUATED with a reason."""

    spanless = _actor_payload("EntE", "irrelevant")
    spanless["processes"]["reactions"][0]["enzymes"][0].pop("evidence")
    check = _evaluate(spanless).checks[_s.CHECK_ACTOR_EVIDENCE]
    assert check.applicable is False
    assert check.inapplicable_reason == sp.NO_ACTOR_SPANS
    assert _s.CHECK_ACTOR_EVIDENCE not in _evaluate(spanless).failed_checks
    assert _release(spanless).semantic_evaluation != SEMANTIC_FAILED

    # A name with no token that could identify a protein is the same case: undecidable,
    # so unexamined. ``normalize_name`` splits ``2,3`` into bare numerals.
    assert sp._actor_named_in_span("2,3", ["catalyzed by EntA"]) is None
    assert sp._actor_named_in_span("", ["catalyzed by EntA"]) is None

    # ...and one spanless actor beside one examinable one does NOT disarm the check: the
    # census reports both, and the examinable one is still judged.
    mixed = _actor_payload("EntE", "secreted by a TolC-dependent process")
    mixed["processes"]["reactions"][0]["enzymes"].append({"entity": "EntB"})
    mixed_check = _evaluate(mixed).checks[_s.CHECK_ACTOR_EVIDENCE]
    assert mixed_check.applicable is True
    assert mixed_check.ok is False
    assert "1 examined, 1 carried no comparable name or span" in mixed_check.summary


# NEW ACCEPTANCE — this check does not exist at the base SHA
def test_new_acceptance_the_gate_demotes_and_never_deletes_or_widens() -> None:
    """Invariants 1 and 2 as behaviour: the cap is one-directional, so a run this check
    clears cannot be RAISED by it, and a run it fails lands in ``review_required`` rather
    than being dropped."""

    clean = _actor_payload("EntE", "2,3-DHB is adenylated by EntE in an ATP-dependent reaction")
    dirty = _actor_payload("EntE", "secreted to the extracellular environment by TolC")

    assert _release(clean).status == RELEASE_READY
    assert _release(dirty).status == REVIEW_REQUIRED
    assert _release(dirty).strict_acceptance_eligible is False

    # Subtractive: a run already below release_ready is not lifted by a clearing check.
    already_low = classify_release_status(
        None, strict_gates_passed=True, semantic_evaluation=SEMANTIC_PASSED,
    )
    assert already_low.status != RELEASE_READY

    # The payload the gate judged is the payload it hands on, byte for byte.
    before = deepcopy(dirty)
    _release(dirty)
    assert dirty == before


# ── RULING 13 non-vacuity ────────────────────────────────────────────────────


def test_non_vacuity_a_detector_that_examines_nothing_cannot_look_like_a_pass(
    monkeypatch: Any,
) -> None:
    """The failure mode this detector is most likely to die of, exhibited then closed.
    Neutralized two ways -- empty the actor buckets, raise the token floor above every
    real token -- F-079's own leg becomes ``release_ready`` again both times, with ``ok``
    ``True``: a detector that stopped looking and one that looked and found nothing agree
    on ``ok``, and only ``applicable`` tells them apart."""

    payload = json.loads(RESEARCH_LEG.read_text(encoding="utf-8"))
    shipped = _evaluate(payload).checks[_s.CHECK_ACTOR_EVIDENCE]
    assert (shipped.applicable, shipped.ok) == (True, False)
    assert _release(payload).status == REVIEW_REQUIRED

    for attribute, disarmed in (("_ACTOR_KEYS", ()), ("_MIN_IDENTIFYING_TOKEN", 99)):
        monkeypatch.setattr(sp, attribute, disarmed)
        blind = _evaluate(payload).checks[_s.CHECK_ACTOR_EVIDENCE]
        assert blind.ok is True, attribute            # indistinguishable from success...
        assert blind.applicable is False, attribute   # ...except here.
        assert blind.inapplicable_reason == sp.NO_ACTOR_SPANS
        assert _release(payload).status == RELEASE_READY, attribute
        monkeypatch.undo()


def test_non_vacuity_the_check_gates_only_because_it_is_in_the_gating_set() -> None:
    """Membership IS the mechanism, so without this the demotion above could be coming
    from somewhere else entirely. The widening 4 -> 5 is declared, not absorbed."""

    assert _s.CHECK_ACTOR_EVIDENCE in SEMANTIC_GATING_CHECKS
    assert _s.CHECK_ACTOR_EVIDENCE in _s.CHECK_ORDER

    # The non-gating twin, unchanged: a payload failing only CHECK_SOURCE_CARRIER is not
    # demoted, so "any failed check demotes" is not what is being observed.
    hygiene_only = {"processes": {"reactions": [
        {"name": "r", "inputs": ["a"], "outputs": ["b"]},
    ]}}
    report = sp.evaluate_production_semantics(hygiene_only)
    assert _s.CHECK_SOURCE_CARRIER in report.failed_checks
    assert semantic_verdict(report)[0] != SEMANTIC_FAILED


def test_the_serialized_report_carries_the_new_check_by_name() -> None:
    """``to_dict`` iterates ``CHECK_ORDER``; a check missing from the record is one no
    downstream reader can see."""

    payload = json.loads(RESEARCH_LEG.read_text(encoding="utf-8"))
    serialized = _evaluate(payload).to_dict()
    assert _s.CHECK_ACTOR_EVIDENCE in serialized["checks"]
    assert serialized["checks"][_s.CHECK_ACTOR_EVIDENCE]["applicable"] is True
    assert serialized["checks"][_s.CHECK_ACTOR_EVIDENCE]["ok"] is False
    assert _s.CHECK_ACTOR_EVIDENCE in serialized["failed_checks"]

    # No error COUNT was added: ``scientific_errors`` is a fixed vocabulary and a new key
    # there moves every benchmark total. The findings carry the detail instead.
    assert set(serialized["scientific_errors"]) == set(_s.ERROR_ORDER)

    # A run with no payload still reports nothing rather than a zero-actor pass.
    empty: List[str] = sp.evaluate_production_semantics(None).failed_checks
    assert empty == []
    assert semantic_verdict(sp.evaluate_production_semantics(None))[0] == SEMANTIC_NOT_EVALUATED
