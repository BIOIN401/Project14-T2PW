"""C-088 / F-107 / **D-065 (LOCKED)** -- the ``extracted_not_serialized`` disposition.

**What was untrue.** ``PRODUCT_CONTRACT`` section 4 glosses ``diagnostic_only`` as
*"recovery and retrieval could not establish a defensible pathway core"*. Measured on
the committed run ``runs_verify/2026-08-24_1428``, six legs across three papers carry
that status because a Stage-0 SCOPE GUARD correctly stopped them -- the batch asked
for *Bacillus subtilis* and the papers are *E. coli*, *L. lactis* and
*L. monocytogenes*, all deliberate gold organism traps -- and on four of those six the
gloss says something false: ``PMC12421875``'s two legs each reached a connected core
of **9** against that case's gold floor of **7**.

D-065 adopts a distinct disposition for exactly that shape and prefers **reading 3**:
an additional explicit field BESIDE the existing runtime status. So this card
preserves the safe runtime refusal, fabricates no gate result, pretends no PWML
exists, extends ``RELEASE_STATES`` not at all, and creates no route toward strict
export. **C-077 was right to refuse ``review_required``** and its reviewer proved it
with a 196,800-combination sweep; nothing here reaches for it.

**G9 -- THE LABEL.** Every test in this file is an **EXPLICITLY LABELLED NEW
ACCEPTANCE TEST** (``test_new_acceptance_*``) and **claims no base failure**. A
disposition that was never emitted anywhere is new capability, which is G9's second
arm. Run against base ``5b408f6`` these fail on ``ImportError`` /
``AttributeError`` / an unexpected keyword -- that is SYMBOL ABSENCE, it is NOT
offered as behavioural proof, and F-122 registered exactly the mistake of dressing it
up as one. Correcting ``PRODUCT_CONTRACT`` section 4's untrue gloss is likewise not a
behavioural regression fix: no run's status, no gate, no rate and no artifact name
moves anywhere in this card.

**Non-vacuity.** Every refusal assertion below carries its own control arm in the same
test -- the identical inputs with the ONE fact under test repaired, asserted to reach
the disposition -- so a ``""`` can never pass because the fixture could not have been
placed anyway. ``test_new_acceptance_every_refusal_in_this_file_is_non_vacuous``
re-states that as one sweep over the whole rule.

**PMC12312563 is deliberately NOT placed, and that is a measured product decision.**
Its two legs reach a connected core of **1** against a gold floor of **1**, so the
case floor clears -- and that case's own gold ``export_rationale`` says in terms *"A
single reaction cannot form a connected pathway, and no second reaction anywhere in
the text shares a metabolite with it."* ``MIN_CONNECTED_CORE_REACTIONS`` (C-074 /
F-101) says the same thing in code: *"one is not a pathway"*. Granting those legs a
disposition that asserts a defensible pathway core would replace one untruth with
another, so the rule applies BOTH floors and the population is the four legs on the
two papers D-065 names.

This file is in **no chunk and not in SMOKE**, deliberately and for the reason
``test_c087_prefreeze_declination_demotes_release_status.py`` records: ``chunk_d_gate
.py`` hard-codes its file list and ENFORCES its component counts, so adding to a chunk
member would move a sprint-gate baseline for an unrelated reason. It runs as a NAMED
FOCUSED obligation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.acceptance import ModeResult, score_run  # noqa: E402
from t2pw.bench.goldset import load_gold_set  # noqa: E402
from t2pw.bench.semantic import SemanticReport  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    DISPOSITION_EXTRACTED_NOT_SERIALIZED,
    MIN_CONNECTED_CORE_REACTIONS,
    NO_DISPOSITION,
    RELEASE_DISPOSITIONS,
    RELEASE_STATES,
    SCOPE_GUARD_STOP_REASON,
    ReleaseStatus,
    classify_release_status,
    release_disposition,
)

#: Read as STRINGS, exactly as C-087's file does: these are ``PRODUCT_CONTRACT`` 4
#: vocabulary and predate this card, so spelling them out keeps the assertions free
#: of anything this card introduced.
RELEASE_READY = "release_ready"
REVIEW_REQUIRED = "review_required"
DIAGNOSTIC_ONLY = "diagnostic_only"

#: The run D-065 and F-107 are measured from. Committed (299 files under git).
RUN_DIR = ROOT / "runs_verify" / "2026-08-24_1428"

#: The FOUR legs the disposition places, measured. ``PMC12312563`` is absent for the
#: reason the module docstring gives, and its absence is asserted, not assumed.
EXPECTED_PLACED = [
    "PMC12421875:research",
    "PMC12421875:strict",
    "PMC12657337:research",
    "PMC12657337:strict",
]

#: ``PMC12421875``'s real numbers: a connected core of 9 against a gold floor of 7.
CORE = 9
FLOOR = 7

#: Every key ``ReleaseStatus.to_dict`` wrote BEFORE this card. Pinned by value so a
#: silently added key is a failure rather than a surprise in a downstream digest.
BASE_RECORD_KEYS = {
    "status",
    "pipeline_executed",
    "strict_gates_passed",
    "semantic_evaluation",
    "semantic_not_evaluated_reason",
    "semantic_failed_checks",
    "semantic_check_evaluability",
    "strict_acceptance_eligible",
    "completeness",
    "missing_anchors",
    "retrieval_attempts",
    "expansion_blocked_reason",
    "coverage_evaluated",
    "reasons",
}


def _scope_conflict_leg(
    *,
    core: Optional[int] = CORE,
    floor: Optional[int] = FLOOR,
    reasons: Any = (SCOPE_GUARD_STOP_REASON,),
    pipeline_executed: bool = True,
    strict_gates_passed: bool = False,
) -> ReleaseStatus:
    """The classifier product for a Stage-0 scope-guard stop.

    Exactly the call ``batch/driver.py::_finalize_scope_conflict`` makes -- pipeline
    executed, strict gates NOT passed, the scope-guard reason -- plus the two measured
    sizes this card adds. Nothing is hand-built: the record under test is a real
    ``classify_release_status`` product, so a test can never assert about a shape
    production does not produce.
    """

    return classify_release_status(
        pipeline_executed=pipeline_executed,
        strict_gates_passed=strict_gates_passed,
        extra_reasons=reasons,
        connected_core_reactions=core,
        required_connected_reactions=floor,
    )


def _satisfied_coverage() -> Dict[str, Any]:
    """A coverage verdict that reaches ``release_ready`` on its own."""

    return {
        "requested_core_declared": True,
        "requested_core_source": "explicit_argument",
        "requested_context": {"pathway_name": "menaquinone biosynthesis"},
        "coverage_ratio": 1.0,
        "unmatched_terms": [],
        "minimum_core_satisfied": True,
        "surviving_processes": 9,
        "reasons": [],
        "thresholds": {"min_core_coverage": 0.5},
    }


def _scored_leg(
    *,
    core: Optional[int],
    floor: Optional[int],
    record: Optional[Dict[str, Any]],
    pwml_artifact: str = "",
    deliverable: bool = False,
) -> ModeResult:
    """One acceptance leg, assembled the way ``score_run`` assembles it."""

    leg = ModeResult(
        paper_id="PMC12421875",
        mode="strict",
        attempted=True,
        required_connected_reactions=floor,
    )
    leg.release_status = record
    leg.pwml_artifact = pwml_artifact
    leg.deliverable = deliverable
    if core is not None:
        leg.semantic = SemanticReport(
            paper_id="PMC12421875",
            mode="strict",
            graph={"largest_core_size": core, "n_reactions": core + 2},
            evaluated=True,
        )
    return leg


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- the disposition itself. No base failure is claimed.
# ---------------------------------------------------------------------------
def test_new_acceptance_a_scope_conflict_leg_with_a_defensible_core_is_placed() -> None:
    """NEW ACCEPTANCE. D-065's whole population, and every honesty constraint at once.

    The disposition appears; the runtime refusal is PRESERVED. Would catch a card
    that reached the disposition by moving the status, by asserting a gate result
    nobody measured, or by claiming a PWML that does not exist.
    """

    release = _scope_conflict_leg()

    assert release.disposition == DISPOSITION_EXTRACTED_NOT_SERIALIZED
    # The safe runtime refusal, unchanged. Each of these is a D-065 constraint.
    assert release.status == DIAGNOSTIC_ONLY
    assert release.strict_gates_passed is False
    assert release.produced_pwml is False
    assert release.pipeline_executed is True
    # TRAP-1 / PRODUCT_CONTRACT 13: no route toward strict export.
    assert release.strict_acceptance_eligible is False

    record = release.to_dict()
    assert record["disposition"] == DISPOSITION_EXTRACTED_NOT_SERIALIZED
    # D-065 reading 3 in one assertion: BESIDE the status, not instead of it.
    assert list(record)[:2] == ["status", "disposition"]
    assert record["status"] == DIAGNOSTIC_ONLY


def test_new_acceptance_the_disposition_is_not_a_fourth_release_state() -> None:
    """NEW ACCEPTANCE. D-065 charters extending ``RELEASE_STATES`` separately.

    Would catch this card smuggling in the change D-065 says is reviewed on its own
    merits, which the charter instructs an implementer to STOP and report instead.
    """

    assert DISPOSITION_EXTRACTED_NOT_SERIALIZED not in RELEASE_STATES
    assert RELEASE_STATES == (RELEASE_READY, REVIEW_REQUIRED, DIAGNOSTIC_ONLY)
    assert RELEASE_DISPOSITIONS == (DISPOSITION_EXTRACTED_NOT_SERIALIZED,)
    assert NO_DISPOSITION == ""


def test_new_acceptance_the_scope_guard_reason_matches_the_driver_that_emits_it() -> None:
    """NEW ACCEPTANCE. The local restatement is kept in step BY TEST, not by comment.

    Would catch ``batch.driver`` renaming its reason and this rule silently placing
    nothing ever again -- a failure that is invisible in every other assertion here,
    because they all build the reason from this module's own constant.
    """

    from t2pw.batch import driver

    assert SCOPE_GUARD_STOP_REASON == driver.REASON_STAGE0_SCOPE_CONFLICT


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- ANTI-WIDENING. The property D-065 / C-088 section 4 demands.
# ---------------------------------------------------------------------------
def test_new_acceptance_anti_widening_another_diagnostic_only_stop_is_not_placed() -> None:
    """NEW ACCEPTANCE. A ``diagnostic_only`` run stopped for ANY other reason.

    All four other routes to ``diagnostic_only`` are exercised WITH the two measured
    sizes supplied, so the only thing separating them from the placed leg is the
    scope-guard fact itself. Would catch a rule that read the STATUS, or the strict
    gate flag, or the core size alone -- any of which would place a crash, a gate
    failure and an empty graph as "a defensible core a scope guard stopped".
    """

    others = {
        "pipeline_never_ran": classify_release_status(
            pipeline_executed=False,
            connected_core_reactions=CORE,
            required_connected_reactions=FLOOR,
        ),
        "strict_gates_blocked_without_a_scope_guard": classify_release_status(
            pipeline_executed=True,
            strict_gates_passed=False,
            connected_core_reactions=CORE,
            required_connected_reactions=FLOOR,
        ),
        "serialization_would_require_invention": classify_release_status(
            _satisfied_coverage(),
            pipeline_executed=True,
            strict_gates_passed=True,
            serializable_without_invention=False,
            connected_core_reactions=CORE,
            required_connected_reactions=FLOOR,
        ),
        "nothing_survived": classify_release_status(
            {"requested_core_declared": True, "surviving_processes": 0,
             "reasons": ["no_surviving_process"], "coverage_ratio": 0.0},
            pipeline_executed=True,
            strict_gates_passed=True,
            connected_core_reactions=CORE,
            required_connected_reactions=FLOOR,
        ),
    }

    for name, release in others.items():
        # NON-VACUITY, first half: each really did reach diagnostic_only, so the
        # refusal below is about the REASON and not about an unreachable status.
        assert release.status == DIAGNOSTIC_ONLY, name
        assert release.disposition == NO_DISPOSITION, name
        assert "disposition" not in release.to_dict(), name

    # NON-VACUITY, second half: the same classifier, the same sizes, PLUS the
    # scope-guard reason -> placed. So the refusals above are the reason's doing.
    assert _scope_conflict_leg().disposition == DISPOSITION_EXTRACTED_NOT_SERIALIZED


def test_new_acceptance_a_single_reaction_core_is_not_a_defensible_pathway_core() -> None:
    """NEW ACCEPTANCE. ``PMC12312563``'s real numbers: core 1 against a gold floor 1.

    The case floor clears and the leg is STILL not placed, because gold's own
    ``export_rationale`` for that case says a single reaction cannot form a connected
    pathway and ``MIN_CONNECTED_CORE_REACTIONS`` says the same. Would catch a rule
    that applied only the gold floor and so asserted a defensible pathway core on the
    one leg where the existing ``diagnostic_only`` gloss is TRUE.
    """

    assert MIN_CONNECTED_CORE_REACTIONS == 2
    assert _scope_conflict_leg(core=1, floor=1).disposition == NO_DISPOSITION
    # NON-VACUITY: one more reaction, same floor -> placed.
    assert (
        _scope_conflict_leg(core=2, floor=1).disposition
        == DISPOSITION_EXTRACTED_NOT_SERIALIZED
    )


def test_new_acceptance_a_core_under_its_own_gold_floor_is_not_placed() -> None:
    """NEW ACCEPTANCE. The case's own floor is load-bearing, not decoration.

    Would catch a rule that dropped the gold floor and placed every leg clearing the
    global two-reaction minimum -- which on ``PMC12421875`` would call a core of 2
    against a floor of 7 "a defensible pathway core".
    """

    assert _scope_conflict_leg(core=FLOOR - 1, floor=FLOOR).disposition == NO_DISPOSITION
    # NON-VACUITY: exactly at the floor -> placed.
    assert (
        _scope_conflict_leg(core=FLOOR, floor=FLOOR).disposition
        == DISPOSITION_EXTRACTED_NOT_SERIALIZED
    )


def test_new_acceptance_unmeasured_evidence_never_reaches_the_disposition() -> None:
    """NEW ACCEPTANCE. Not measured is never a fact (the D-038 rule).

    Would catch the defect this whole card exists to avoid: ASSUMING the defensible
    core. Each row omits or corrupts exactly one of the two sizes; none may be placed.
    A ``bool`` is included because ``isinstance(True, int)`` is ``True`` in Python and
    a flag arriving where a count belongs would otherwise read as the count 1.
    """

    unmeasured: List[Dict[str, Any]] = [
        {"core": None, "floor": FLOOR},
        {"core": CORE, "floor": None},
        {"core": None, "floor": None},
        {"core": True, "floor": FLOOR},
        {"core": CORE, "floor": True},
        {"core": CORE, "floor": 0},
        {"core": CORE, "floor": -1},
    ]
    for row in unmeasured:
        release = _scope_conflict_leg(**row)
        assert release.status == DIAGNOSTIC_ONLY, row
        assert release.disposition == NO_DISPOSITION, row

    # An unparseable size is the same answer for the same reason. Asserted on the
    # rule directly rather than through ``classify_release_status``: the C-074
    # connected-core floor there does ``int(connected_core_reactions)`` unguarded and
    # raises on a non-numeric string, which is PRE-EXISTING behaviour this card does
    # not own and must not silently change.
    assert release_disposition(
        {
            "status": DIAGNOSTIC_ONLY,
            "pipeline_executed": True,
            "strict_gates_passed": False,
            "reasons": [SCOPE_GUARD_STOP_REASON],
        },
        connected_core_reactions="nine",
        required_connected_reactions=FLOOR,
    ) == NO_DISPOSITION

    # NON-VACUITY: both measured -> placed.
    assert _scope_conflict_leg().disposition == DISPOSITION_EXTRACTED_NOT_SERIALIZED


def test_new_acceptance_a_fabricated_gate_result_withdraws_the_disposition() -> None:
    """NEW ACCEPTANCE. ``strict_gates_passed=True`` is not a scope-guard stop.

    A scope guard stops the run BEFORE the strict technical gates run, so a record
    claiming they passed did not come from one. Would catch a future caller reaching
    the disposition by asserting the very measurement C-077 refused to fabricate.
    """

    fabricated = release_disposition(
        {
            "status": DIAGNOSTIC_ONLY,
            "pipeline_executed": True,
            "strict_gates_passed": True,
            "reasons": [SCOPE_GUARD_STOP_REASON],
        },
        connected_core_reactions=CORE,
        required_connected_reactions=FLOOR,
    )
    assert fabricated == NO_DISPOSITION
    # NON-VACUITY: the same record with the honest gate result -> placed.
    assert release_disposition(
        {
            "status": DIAGNOSTIC_ONLY,
            "pipeline_executed": True,
            "strict_gates_passed": False,
            "reasons": [SCOPE_GUARD_STOP_REASON],
        },
        connected_core_reactions=CORE,
        required_connected_reactions=FLOOR,
    ) == DISPOSITION_EXTRACTED_NOT_SERIALIZED


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- the other two states, and every pre-existing record.
# ---------------------------------------------------------------------------
def test_new_acceptance_release_ready_and_review_required_are_untouched() -> None:
    """NEW ACCEPTANCE. Neither state can acquire a disposition, on any input.

    Both are built WITH the scope-guard reason and both measured sizes -- a shape
    production cannot produce -- so the only thing refusing them is the status gate.
    Would catch a rule that placed a run which actually shipped a PWML, which is
    D-065's *"never pretend a PWML exists"* read from the other end.
    """

    ready = classify_release_status(
        _satisfied_coverage(),
        pipeline_executed=True,
        strict_gates_passed=True,
        extra_reasons=(SCOPE_GUARD_STOP_REASON,),
        connected_core_reactions=CORE,
        required_connected_reactions=FLOOR,
    )
    review = classify_release_status(
        pipeline_executed=True,
        strict_gates_passed=True,
        extra_reasons=(SCOPE_GUARD_STOP_REASON,),
        connected_core_reactions=CORE,
        required_connected_reactions=FLOOR,
    )

    # NON-VACUITY: the two statuses really were reached, so the refusals are real.
    assert ready.status == RELEASE_READY
    assert review.status == REVIEW_REQUIRED
    for release in (ready, review):
        assert release.produced_pwml is True
        assert release.disposition == NO_DISPOSITION
        assert "disposition" not in release.to_dict()
    # And the pre-existing eligibility invariant is exactly what it was.
    assert ready.strict_acceptance_eligible is True
    assert review.strict_acceptance_eligible is False


def test_new_acceptance_a_record_with_no_disposition_is_byte_identical() -> None:
    """NEW ACCEPTANCE. No existing artifact, digest or golden capture moves.

    ``to_dict`` writes the key ONLY when a disposition was established. Would catch
    an always-present ``"disposition": ""`` -- a placeholder that reads like a
    measurement, and one that would have moved all seven slots of
    ``tests/test_batch_driver_seam_golden.py``'s digest, which hashes this dict.
    """

    for release in (
        classify_release_status(),
        classify_release_status(pipeline_executed=False),
        classify_release_status(_satisfied_coverage(), strict_gates_passed=True),
        # The exact call production's scope-conflict seam makes TODAY: it supplies
        # neither size, so it is unchanged by this card.
        classify_release_status(
            pipeline_executed=True,
            strict_gates_passed=False,
            extra_reasons=(SCOPE_GUARD_STOP_REASON,),
        ),
    ):
        assert release.disposition == NO_DISPOSITION
        assert set(release.to_dict()) == BASE_RECORD_KEYS

    # NON-VACUITY: the one shape that DOES record it grows exactly one key.
    placed = _scope_conflict_leg().to_dict()
    assert set(placed) - BASE_RECORD_KEYS == {"disposition"}


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- the acceptance scorer seam.
# ---------------------------------------------------------------------------
def test_new_acceptance_the_scorer_reports_the_disposition_and_its_evidence() -> None:
    """NEW ACCEPTANCE. ``ModeResult`` places a leg and carries the numbers behind it.

    Would catch a scorer that asserted the disposition without the core size and the
    gold floor beside it -- the assumption D-065 forbids -- or one that reported it
    on a leg whose eligibility flag it had also moved.
    """

    record = _scope_conflict_leg().to_dict()
    leg = _scored_leg(core=CORE, floor=FLOOR, record=record)

    assert leg.connected_core_reactions == CORE
    assert leg.release_disposition == DISPOSITION_EXTRACTED_NOT_SERIALIZED
    assert leg.strict_acceptance_eligible is False

    row = leg.to_dict()
    assert row["release_disposition"] == DISPOSITION_EXTRACTED_NOT_SERIALIZED
    assert row["connected_core_reactions"] == CORE
    assert row["required_connected_reactions"] == FLOOR

    # NON-VACUITY: a leg with no classification at all reports nothing and grows
    # no key, so the three assertions above are the disposition's doing.
    bare = _scored_leg(core=CORE, floor=FLOOR, record=None)
    assert bare.release_disposition == NO_DISPOSITION
    assert "release_disposition" not in bare.to_dict()
    assert "connected_core_reactions" not in bare.to_dict()


def test_new_acceptance_a_pwml_that_actually_landed_withdraws_the_disposition() -> None:
    """NEW ACCEPTANCE. The scorer's artifact observation is INDEPENDENT of the record.

    ``produced_pwml`` is read from the manifest row's own artifact name and from the
    deliverable scan, not from the record's self-description, and it may only REFUSE.
    Would catch a scorer that trusted a record calling itself ``diagnostic_only``
    beside a PWML that is on disk.
    """

    record = _scope_conflict_leg().to_dict()

    for kwargs in (
        {"pwml_artifact": "pathway.pwml"},
        {"pwml_artifact": "pathway.review_required.pwml"},
        {"deliverable": True},
    ):
        leg = _scored_leg(core=CORE, floor=FLOOR, record=record, **kwargs)
        assert leg.release_disposition == NO_DISPOSITION, kwargs

    # NON-VACUITY: the same record and sizes with no artifact -> placed.
    assert (
        _scored_leg(core=CORE, floor=FLOOR, record=record).release_disposition
        == DISPOSITION_EXTRACTED_NOT_SERIALIZED
    )


def test_new_acceptance_a_leg_with_no_measured_core_is_not_placed() -> None:
    """NEW ACCEPTANCE. A leg whose payload never evaluated has no core to defend.

    Would catch ``connected_core_reactions`` reading a missing measurement as zero,
    or as the case floor, either of which turns "we could not look" into a finding.
    """

    leg = _scored_leg(core=None, floor=FLOOR, record=_scope_conflict_leg().to_dict())
    assert leg.semantic is None
    assert leg.connected_core_reactions is None
    assert leg.release_disposition == NO_DISPOSITION

    # A report that RAN but did not evaluate is the same answer for the same reason.
    leg.semantic = SemanticReport(
        paper_id="PMC12421875", mode="strict",
        graph={"largest_core_size": CORE}, evaluated=False,
    )
    assert leg.connected_core_reactions is None
    assert leg.release_disposition == NO_DISPOSITION

    # NON-VACUITY: flip evaluability alone -> placed.
    leg.semantic.evaluated = True
    assert leg.release_disposition == DISPOSITION_EXTRACTED_NOT_SERIALIZED


def test_new_acceptance_the_runtime_field_and_the_scorer_read_one_rule() -> None:
    """NEW ACCEPTANCE. D-065: *"no reader has to decide which of two fields to believe."*

    The frozen record's own ``disposition`` and the scorer's answer are the same
    function over the same facts. Would catch a scorer that grew a second, drifting
    reading of the ruling -- the exact failure mode C-087 avoided by making
    ``prefreeze_review_reasons`` a single normalizer.
    """

    release = _scope_conflict_leg()
    record = release.to_dict()
    leg = _scored_leg(core=CORE, floor=FLOOR, record=record)

    assert record["disposition"] == leg.release_disposition
    # And the rule applied directly to the frozen object agrees with both.
    assert release_disposition(
        release, connected_core_reactions=CORE, required_connected_reactions=FLOOR
    ) == record["disposition"]
    # A record whose ``disposition`` key was tampered with does not win: the rule
    # reads the FACTS, never the assertion.
    tampered = dict(record)
    tampered["disposition"] = DISPOSITION_EXTRACTED_NOT_SERIALIZED
    tampered["reasons"] = ["strict_technical_gates_blocked_export"]
    assert _scored_leg(core=CORE, floor=FLOOR, record=tampered).release_disposition == (
        NO_DISPOSITION
    )


def test_new_acceptance_a_non_record_is_never_placed() -> None:
    """NEW ACCEPTANCE. Every non-mapping input answers ``""`` rather than raising.

    Report and scorer code renders rows written by older runs and must never raise.
    """

    for value in (None, "", "diagnostic_only", 0, [], object()):
        assert release_disposition(
            value, connected_core_reactions=CORE, required_connected_reactions=FLOOR
        ) == NO_DISPOSITION


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- the real committed run.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="the pinned verification run is absent")
def test_new_acceptance_the_committed_run_places_exactly_four_legs() -> None:
    """NEW ACCEPTANCE. The measurement D-065 and F-107 are written from.

    Scores ``runs_verify/2026-08-24_1428`` and asserts the placed population by name,
    every honesty constraint on each placed leg, and that ``PMC12312563`` -- whose
    legs ARE scope-conflict stops -- is excluded. Would catch a rule that placed all
    six, or none, or that moved a leg's eligibility while placing it.
    """

    report = score_run(RUN_DIR, load_gold_set())

    assert report.extracted_not_serialized_legs == EXPECTED_PLACED
    assert report.release_dispositions == {
        DISPOSITION_EXTRACTED_NOT_SERIALIZED: EXPECTED_PLACED
    }
    assert report.to_dict()["release_dispositions"] == report.release_dispositions

    placed = {
        f"{paper.paper_id}:{mode}": leg
        for paper in report.papers
        for mode, leg in paper.legs.items()
        if leg.release_disposition
    }
    assert sorted(placed) == EXPECTED_PLACED
    for name, leg in placed.items():
        record = leg.release_status or {}
        assert record.get("status") == DIAGNOSTIC_ONLY, name
        assert record.get("pipeline_executed") is True, name
        assert record.get("strict_gates_passed") is False, name
        # TRAP-1 / PRODUCT_CONTRACT 13, on every placed leg.
        assert leg.strict_acceptance_eligible is False, name
        assert record.get("strict_acceptance_eligible") is False, name
        assert leg.pwml_artifact == "", name
        assert leg.deliverable is False, name
        # The evidence, not the assumption.
        assert leg.connected_core_reactions >= leg.required_connected_reactions, name
        assert leg.connected_core_reactions >= MIN_CONNECTED_CORE_REACTIONS, name

    # NON-VACUITY, and the PMC12312563 finding stated as a measurement: its two legs
    # ARE scope-conflict stops with a core that clears their own gold floor, and they
    # are still not placed.
    excluded = {
        f"{paper.paper_id}:{mode}": leg
        for paper in report.papers
        if paper.paper_id == "PMC12312563"
        for mode, leg in paper.legs.items()
    }
    assert sorted(excluded) == ["PMC12312563:research", "PMC12312563:strict"]
    for name, leg in excluded.items():
        record = leg.release_status or {}
        assert SCOPE_GUARD_STOP_REASON in (record.get("reasons") or []), name
        assert leg.connected_core_reactions == 1, name
        assert leg.required_connected_reactions == 1, name
        assert leg.release_disposition == NO_DISPOSITION, name


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="the pinned verification run is absent")
def test_new_acceptance_the_disposition_moves_no_rate() -> None:
    """NEW ACCEPTANCE. Every denominator on the committed run, pinned by value.

    D-065 removes two cases from priority 5's strict denominator by RECONCILING GOLD,
    which the orchestrator already did; nothing in this card may move a number.
    Would catch a disposition wired into any numerator, denominator or priority --
    the one way a truthful record could still buy an untruthful score.
    """

    report = score_run(RUN_DIR, load_gold_set())
    rendered = {key: d.render() for key, d in report.denominators.items()}

    assert rendered == {
        "gold_relevance_prevalence": "8/10 = 80%",
        "extraction_success": "8/8 = 100%",
        "semantic_pathway_success": "0/8 = 0%",
        "research_deliverable_produced": "4/8 = 50%",
        "research_semantically_confirmed": "0/8 = 0%",
        # D-065's reconciled denominator: 4 -> 2, and still 0. The two survivors
        # are exactly the papers whose gold is NOT partial_only.
        "strict_pwml_success": "0/2 = 0%",
    }
    strict = report.denominators["strict_pwml_success"]
    assert strict.numerator_names == []
    assert sorted(strict.denominator_names) == ["PMC12096016", "PMC12782028"]

    # No placed leg reaches the STRICT numerator, which is the one D-065 is about.
    # A blanket "in no numerator at all" would be FALSE and would have to be weakened
    # to pass: both placed papers really are in ``extraction_success``'s numerator
    # (they produced payloads) and in ``gold_relevance_prevalence``'s (they are
    # mechanistically relevant), and both of those facts are exactly why their
    # scope-conflict stop deserves a disposition rather than the existing gloss.
    for name in EXPECTED_PLACED:
        assert name.split(":", 1)[0] not in strict.numerator_names, name

    # Priority 1-3 error totals are the run's, untouched.
    assert report.errors.totals["unsupported_reactions"] == 0
    assert report.errors.totals["orphaned_references"] == 0
    assert report.errors.totals["false_real_identifiers"] == 8


# ---------------------------------------------------------------------------
# NON-VACUITY, restated as one sweep.
# ---------------------------------------------------------------------------
def test_new_acceptance_every_refusal_in_this_file_is_non_vacuous() -> None:
    """NEW ACCEPTANCE. One fact removed at a time; each removal must flip the answer.

    The placed baseline is asserted first, so a rule that placed NOTHING could not
    make this file green by refusing everything -- which is the way a non-vacuity
    proof most often rots.
    """

    baseline = dict(
        release={
            "status": DIAGNOSTIC_ONLY,
            "pipeline_executed": True,
            "strict_gates_passed": False,
            "reasons": [SCOPE_GUARD_STOP_REASON, "strict_technical_gates_blocked_export"],
        },
        connected_core_reactions=CORE,
        required_connected_reactions=FLOOR,
    )
    assert release_disposition(**baseline) == DISPOSITION_EXTRACTED_NOT_SERIALIZED

    mutations: Dict[str, Dict[str, Any]] = {
        "status_release_ready": {"release": {**baseline["release"], "status": RELEASE_READY}},
        "status_review_required": {
            "release": {**baseline["release"], "status": REVIEW_REQUIRED}
        },
        "pipeline_did_not_execute": {
            "release": {**baseline["release"], "pipeline_executed": False}
        },
        "pipeline_executed_merely_truthy": {
            "release": {**baseline["release"], "pipeline_executed": "yes"}
        },
        "gates_claimed_to_pass": {
            "release": {**baseline["release"], "strict_gates_passed": True}
        },
        "no_scope_guard_reason": {
            "release": {
                **baseline["release"],
                "reasons": ["strict_technical_gates_blocked_export"],
            }
        },
        "core_unmeasured": {"connected_core_reactions": None},
        "floor_unmeasured": {"required_connected_reactions": None},
        "core_under_its_gold_floor": {"connected_core_reactions": FLOOR - 1},
        "core_is_one_reaction": {
            "connected_core_reactions": 1, "required_connected_reactions": 1
        },
        "a_pwml_landed": {"produced_pwml": True},
    }

    for name, mutation in mutations.items():
        assert release_disposition(**{**baseline, **mutation}) == NO_DISPOSITION, name


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="the pinned verification run is absent")
def test_new_acceptance_the_whole_report_differs_only_by_the_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW ACCEPTANCE. The differential proof, over the entire serialized report.

    Scores the committed run twice -- once for real, once with the ONE rule the whole
    mechanism reads through neutered -- and asserts the two reports are IDENTICAL once
    the disposition keys are stripped. Not a spot check on a rate anyone remembered to
    pin: every priority, denominator, blocker, boundary, error count and per-leg field
    is compared by value. Would catch the disposition reaching ANY number at all.

    Non-vacuous by construction: the neutered arm is asserted to place nothing and the
    real arm to place four, so the two cannot be the same run twice.
    """

    real = score_run(RUN_DIR, load_gold_set()).to_dict()

    monkeypatch.setattr(
        "t2pw.bench.acceptance._release_disposition",
        lambda *args, **kwargs: NO_DISPOSITION,
    )
    neutered = score_run(RUN_DIR, load_gold_set()).to_dict()

    assert real.pop("release_dispositions") == {
        DISPOSITION_EXTRACTED_NOT_SERIALIZED: EXPECTED_PLACED
    }
    assert "release_dispositions" not in neutered

    stripped = 0
    for paper in real["papers"]:
        for leg in paper["legs"].values():
            for key in (
                "release_disposition",
                "connected_core_reactions",
                "required_connected_reactions",
            ):
                if key in leg:
                    del leg[key]
                    stripped += 1
    # Four legs times three keys: the neutered arm removed exactly this much and
    # nothing else, so the equality below is a real comparison.
    assert stripped == len(EXPECTED_PLACED) * 3

    assert real == neutered


def test_new_acceptance_describe_carries_a_recorded_disposition() -> None:
    """NEW ACCEPTANCE. The record's own one-line renderer does not drop the field.

    A renderer that silently dropped it would put the reader back where D-065 found
    them: reading ``diagnostic_only`` and supplying the contract gloss that is untrue
    of this run. Would catch that regression.

    NOTHING CURRENTLY RENDERED MOVES, and the second half asserts it: a record with
    no disposition produces the byte-identical line it always did, which is what
    keeps ``batch/report.py`` and ``bench/render.py`` -- the two callers -- unchanged.
    """

    from t2pw.pipeline.release_status import describe

    placed = describe(_scope_conflict_leg())
    assert DISPOSITION_EXTRACTED_NOT_SERIALIZED in placed
    assert placed.startswith(DIAGNOSTIC_ONLY)

    # NON-VACUITY, and the no-movement guarantee in one assertion: the same record
    # without the two sizes renders exactly the pre-C-088 line.
    unplaced = classify_release_status(
        pipeline_executed=True,
        strict_gates_passed=False,
        extra_reasons=(SCOPE_GUARD_STOP_REASON,),
    )
    assert describe(unplaced) == (
        "diagnostic_only  [pipeline ran; strict gates failed; "
        "semantic evaluation NOT PERFORMED]"
    )
    assert DISPOSITION_EXTRACTED_NOT_SERIALIZED not in describe(unplaced)
