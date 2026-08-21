"""C-056c: evaluability travels beside the semantic verdict (F-053 / D-054).

**This is a CORRECTION, not a new capability.** ``semantic_evaluation`` is already
written by C-056a, already serialized into ``quarantine_report["release"]`` and the batch
manifest row, and already read. The observable defect is that ``"passed"`` records nothing
about *how much answered*: "one of four gating checks was evaluable and that one passed"
serialized **byte-identically** to "four of four were evaluable and all passed".

So per G9 this file's proof is **behavioural, not symbolic**.
:func:`test_correction_a_partly_evaluated_pass_serializes_differently_from_a_fully_evaluated_pass`
builds both payloads, serializes both, and asserts the two artifacts DIFFER. At the base
SHA they are equal and it fails on that comparison -- **not** on a missing attribute, and
it reaches that comparison through :func:`_classify` which adapts to either arity of
``semantic_verdict``. A test that merely errored because a new key does not exist would
prove nothing, and is not what is offered here.

WHAT THIS CARD DOES NOT DO, asserted rather than promised:

* it adds **no affirmative reader** of ``passed`` -- F-053's prohibition is untouched, and
  ``test_the_carrier_adds_no_affirmative_reader_of_passed`` counts the comparisons in
  ``src/`` by AST so a later card cannot add one quietly;
* it does **not** close the ``CHECK_RAG_REINTRODUCTION`` shortfall. That needs an
  ``admission`` report which does not exist at this seam and a signature change which is
  ungranted (``DECISIONS.md:2277``). The shortfall is made VISIBLE and is pinned as still
  open by ``test_the_rag_shortfall_is_recorded_and_deliberately_not_closed``;
* it hard-codes **no evaluable count**. Production reaches three evaluable; a replay of
  context-free committed artifacts reaches one. Both are derived per run, never written
  down (D-054 section 4).
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.semantic import CHECK_ACTOR_EVIDENCE, CHECK_RAG_REINTRODUCTION  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    SEMANTIC_FAILED,
    SEMANTIC_GATING_CHECKS,
    SEMANTIC_NOT_EVALUATED,
    SEMANTIC_PASSED,
    ReleaseStatus,
    classify_release_status,
    semantic_verdict,
)
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402

from test_strict_quarantine_contract_alignment import _base  # noqa: E402

#: A production-shaped Stage-0 request: the pathway the base payload really is about, AND
#: an organism. ``pathway_context_from_stage_zero`` (``entity_admission.py:396-399``) reads
#: exactly these two fields, and BOTH are needed to reach D-054 section 4's three-evaluable
#: production shape -- ``semantic_production.py:123-124`` makes ``CHECK_ORGANISM``
#: INAPPLICABLE when no organism was requested. A ``pathway_name``-only context therefore
#: reaches two, not three; ``_PATHWAY_ONLY_CONTEXT`` below pins that difference so the
#: figure is never mistaken for a constant of the seam.
_RIGHT_PATHWAY_CONTEXT: Dict[str, Any] = {
    "pathway_name": "glutathione biosynthesis",
    "likely_organism": "Homo sapiens",
}

#: The same request with the organism withheld.
_PATHWAY_ONLY_CONTEXT: Dict[str, Any] = {"pathway_name": "glutathione biosynthesis"}

#: A coverage verdict that clears the threshold, so the technical chain never caps the
#: status and the only thing under test is the semantic record.
_COVERAGE: Dict[str, Any] = {
    "requested_core_declared": True,
    "surviving_processes": 3,
    "minimum_core_satisfied": True,
    "reasons": [],
}


# ── duck-typed reports, exactly the shape ``semantic_verdict`` documents ─────


class _Check:
    def __init__(self, ok: bool = True, reason: str = "") -> None:
        self.ok, self.inapplicable_reason = ok, reason

    @property
    def applicable(self) -> bool:
        return not self.inapplicable_reason


class _Report:
    """A report in which the NAMED checks are inapplicable and the rest pass."""

    evaluated = True
    not_evaluated_reason = ""

    def __init__(self, inapplicable: Tuple[str, ...] = ()) -> None:
        self.checks = {
            name: _Check(reason=f"{name} could not be evaluated here" if name in inapplicable else "")
            for name in SEMANTIC_GATING_CHECKS
        }


def _classify(report: Any) -> ReleaseStatus:
    """Classify through ``semantic_verdict`` at EITHER arity.

    This is the G9 shim the shared execution block asks for. At the base SHA
    ``semantic_verdict`` answers three values and no evaluability keyword is passed; at the
    tip it answers four and the keyword is. **Identical call shape on both sides**, so the
    proof below compares serialized artifacts and never a symbol's existence.
    """

    verdict = semantic_verdict(report)
    extra = {"semantic_check_evaluability": verdict[3]} if len(verdict) > 3 else {}
    return classify_release_status(
        _COVERAGE,
        strict_gates_passed=True,
        semantic_evaluation=verdict[0],
        semantic_not_evaluated_reason=verdict[1],
        semantic_failed_checks=verdict[2],
        **extra,
    )


def _serialized(report: Any) -> str:
    return json.dumps(_classify(report).to_dict(), sort_keys=True)


def _evaluability(status: ReleaseStatus) -> List[Dict[str, Any]]:
    return list(status.to_dict()["semantic_check_evaluability"])


# ── G9 CORRECTION — the base-failing behavioural proof ───────────────────────


def test_correction_a_partly_evaluated_pass_serializes_differently_from_a_fully_evaluated_pass() -> None:
    """**FAILS BEHAVIOURALLY AT THE BASE SHA.** F-053, stated as an artifact comparison.

    Two runs, both ``passed``. One had a single gating check evaluable; the other had all
    of them. At the base these serialize to the SAME string -- that is the whole defect,
    and the assertion that fails there is ``one != every``, a comparison of two artifacts
    this file builds, reached through :func:`_classify`'s arity-tolerant shim. No
    attribute that exists only at the tip is touched before it.

    **Width-parametric since C-071 moved the gating set 4 -> 5.** The widths were literals
    when the set was closed at four; they now derive from the set, so the assertions stay
    exact without naming a number a later ratified widening would falsify again.
    """

    all_but_one_inapplicable = _Report(inapplicable=SEMANTIC_GATING_CHECKS[1:])
    all_applicable = _Report()

    # The premise both sides agree on: the VERDICT is identical and is a pass. If this
    # ever fails the comparison below would be trivially true and would prove nothing.
    one_status, every_status = _classify(all_but_one_inapplicable), _classify(all_applicable)
    assert one_status.semantic_evaluation == SEMANTIC_PASSED
    assert every_status.semantic_evaluation == SEMANTIC_PASSED
    assert one_status.status == every_status.status

    one, four = _serialized(all_but_one_inapplicable), _serialized(all_applicable)

    # ── THE BASE FAILURE. Equal at base, different at the tip. ──────────────
    assert one != four, (
        "a partly evaluated pass and a fully evaluated pass serialize identically: "
        "the manifest cannot say how much was evaluated"
    )

    # ...and the tip's difference is the ruled shape (D-054 section 6), not a bare count.
    width = len(SEMANTIC_GATING_CHECKS)
    one_map, four_map = _evaluability(one_status), _evaluability(every_status)
    assert [entry["check"] for entry in one_map] == list(SEMANTIC_GATING_CHECKS)
    assert [entry["applicable"] for entry in one_map] == [True] + [False] * (width - 1)
    assert [entry["applicable"] for entry in four_map] == [True] * width
    # A reader can reconstruct WHY each one did not count; a count could not say this.
    assert all(entry["inapplicable_reason"] for entry in one_map[1:])
    assert not any(entry["inapplicable_reason"] for entry in four_map)


def test_correction_the_count_is_derivable_from_the_map_at_every_evaluable_width() -> None:
    """The map answers at every evaluable width, 0 up to the whole set -- none is special.

    Also fails behaviourally at base: every one of these runs serializes to one of only
    two strings there (``passed`` or ``not_evaluated``), so the set below collapses.
    """

    widths = range(len(SEMANTIC_GATING_CHECKS) + 1)
    reports = {width: _Report(inapplicable=SEMANTIC_GATING_CHECKS[width:]) for width in widths}

    # ── THE BASE FAILURE, taken before any new key is read. At base these runs
    # collapse onto TWO strings (one ``not_evaluated``, the rest identical
    # ``passed``); at the tip every one of them is distinguishable. ──────────
    assert len({_serialized(report) for report in reports.values()}) == len(widths)

    for width, report in reports.items():
        entries = _evaluability(_classify(report))
        # DERIVED, never stored: the count is a property of the record, and neither "1"
        # nor "3" is written anywhere in src/ (D-054 section 4).
        assert len([entry for entry in entries if entry["applicable"]]) == width
        assert len(entries) == len(SEMANTIC_GATING_CHECKS)


# ── the production seam: three evaluable, and the reason for the rest ────────


def test_the_seam_records_three_evaluable_gating_checks_beside_a_passed_verdict() -> None:
    """TIP CONTENT, not a base-failure proof -- the base has no key to read.

    D-054 section 4: under a context-carrying request the seam reaches THREE evaluable,
    never all of them, and never the ``1`` the context-free replay arm reports. Measured here
    through the real ``quarantine_and_close``, so the figure is the seam's own.
    """

    result = quarantine_and_close(
        _base(), strict_db=True, pathway_context=_RIGHT_PATHWAY_CONTEXT
    )
    release = dict(result.quarantine_report["release"])
    evaluability = list(release["semantic_check_evaluability"])

    assert release["semantic_evaluation"] == SEMANTIC_PASSED
    assert [entry["check"] for entry in evaluability] == list(SEMANTIC_GATING_CHECKS)

    applicable = [entry["check"] for entry in evaluability if entry["applicable"]]
    inapplicable = [entry for entry in evaluability if not entry["applicable"]]
    assert len(applicable) == 3, applicable
    # THREE is unchanged by C-071's widening: ``_base()``'s reactions carry ``enzymes:
    # []``, so the fifth gating check has no actor row to examine and reports itself
    # inapplicable rather than joining the evaluable three.
    assert [entry["check"] for entry in inapplicable] == [
        CHECK_RAG_REINTRODUCTION, CHECK_ACTOR_EVIDENCE,
    ]
    assert all(entry["inapplicable_reason"] for entry in inapplicable)

    # ...and THREE is a property of this request, not of the seam. Withhold the organism
    # and the same payload reaches two, with the organism check naming its own reason.
    # This is why no count is hard-coded in src/ (D-054 section 4).
    thinner = quarantine_and_close(
        _base(), strict_db=True, pathway_context=_PATHWAY_ONLY_CONTEXT
    )
    thin_map = thinner.quarantine_report["release"]["semantic_check_evaluability"]
    assert len([entry for entry in thin_map if entry["applicable"]]) == 2, thin_map
    assert all(
        entry["inapplicable_reason"] for entry in thin_map if not entry["applicable"]
    )

    # The bump that carries this key, disclosed under merge rule 4 (D-054 section 8).
    assert result.quarantine_report["schema_version"] == 6


def test_the_rag_shortfall_is_recorded_and_deliberately_not_closed() -> None:
    """The shortfall is made VISIBLE by this card and left OPEN by it, on purpose.

    Closing it needs an ``admission`` report at this seam. There is none, and giving
    ``quarantine_and_close`` an ``admission`` parameter is ungranted (``DECISIONS.md:2277``)
    and collides with C-057. This test pins BOTH halves: the reason string reaches the
    record, and the signature that would close it did not grow.
    """

    result = quarantine_and_close(
        _base(), strict_db=True, pathway_context=_RIGHT_PATHWAY_CONTEXT
    )
    recorded = {
        entry["check"]: entry
        for entry in result.quarantine_report["release"]["semantic_check_evaluability"]
    }
    reason = recorded[CHECK_RAG_REINTRODUCTION]["inapplicable_reason"]
    # RELOCATED from bench.semantic, not invented here: the check's own words.
    assert "admission" in reason.lower(), reason

    # The signature is untouched -- if a later card closes the shortfall it must do so
    # through its own decision, and this line goes red when it does.
    assert "admission" not in inspect.signature(quarantine_and_close).parameters


# ── non-vacuity: attack the carrier ──────────────────────────────────────────


def test_non_vacuity_the_map_tracks_the_report_position_by_position() -> None:
    """Delete the applicability the map reports and the map must follow it.

    A carrier that answered a constant would pass every assertion above. This walks one
    inapplicable check across every position and requires the record to move with it.
    """

    for index, name in enumerate(SEMANTIC_GATING_CHECKS):
        status = _classify(_Report(inapplicable=(name,)))
        entries = _evaluability(status)
        assert [entry["applicable"] for entry in entries] == [
            position != index for position in range(len(SEMANTIC_GATING_CHECKS))
        ], name
        assert entries[index]["check"] == name
        assert entries[index]["inapplicable_reason"]
        # ...and no OTHER position picked up a reason it was not given.
        assert not any(
            entry["inapplicable_reason"] for position, entry in enumerate(entries)
            if position != index
        )


def test_non_vacuity_an_absent_reason_is_recorded_as_absent_not_invented() -> None:
    """A duck with no ``inapplicable_reason`` records ``""`` -- never a synthetic string.

    ``test_semantic_release_gating.py`` already relies on exactly this duck, and D-038's
    rule is that a fabricated value must never be indistinguishable from a measured one.
    """

    class _Bare:
        def __init__(self) -> None:
            self.ok, self.applicable = True, False

    class _BareReport:
        evaluated = True
        not_evaluated_reason = ""
        checks = {name: _Bare() for name in SEMANTIC_GATING_CHECKS}

    entries = _evaluability(_classify(_BareReport()))
    assert [entry["applicable"] for entry in entries] == [False] * len(SEMANTIC_GATING_CHECKS)
    assert [entry["inapplicable_reason"] for entry in entries] == [""] * len(SEMANTIC_GATING_CHECKS)

    # A check the report never carried at all is recorded the same honest way, and every
    # name is still present -- an omission never shortens the record.
    class _MissingReport:
        evaluated = True
        not_evaluated_reason = ""
        checks: Dict[str, Any] = {}

    missing = _evaluability(_classify(_MissingReport()))
    assert [entry["check"] for entry in missing] == list(SEMANTIC_GATING_CHECKS)
    assert not any(entry["applicable"] for entry in missing)


def test_empty_means_not_recorded_and_is_never_a_measured_all_inapplicable() -> None:
    """The one ambiguity a carrier could itself introduce, closed by length.

    A report that never evaluated has NO per-check applicability, so the record is empty.
    A report that evaluated and found everything inapplicable records one entry per gating
    check. Both
    answer ``not_evaluated``; only the record tells them apart.
    """

    class _NeverEvaluated:
        evaluated = False
        not_evaluated_reason = "no payload"
        checks: Dict[str, Any] = {}

    never = _classify(_NeverEvaluated())
    measured = _classify(_Report(inapplicable=SEMANTIC_GATING_CHECKS))

    assert never.semantic_evaluation == SEMANTIC_NOT_EVALUATED
    assert measured.semantic_evaluation == SEMANTIC_NOT_EVALUATED
    assert _evaluability(never) == []
    assert len(_evaluability(measured)) == len(SEMANTIC_GATING_CHECKS)
    assert all(entry["inapplicable_reason"] for entry in _evaluability(measured))


# ── the carrier is a carrier: nothing reads it ───────────────────────────────


def test_the_carrier_adds_no_affirmative_reader_of_passed() -> None:
    """F-053's prohibition, asserted mechanically instead of promised in a comment.

    Exactly ONE comparison against ``SEMANTIC_PASSED`` exists in ``src/`` -- the
    ``semantic_confirmed`` property, which D-054 section 5 measured as having zero ``src/``
    consumers. This card adds none, and no property that reads the new record. Counted by
    AST so the prose in this card's own comments cannot satisfy it.
    """

    comparisons = 0
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Compare):
                continue
            # Both spellings, so a later card cannot slip one past this by importing
            # the module instead of the name.
            if any(
                (isinstance(item, ast.Name) and item.id == "SEMANTIC_PASSED")
                or (isinstance(item, ast.Attribute) and item.attr == "SEMANTIC_PASSED")
                for item in [node.left, *node.comparators]
            ):
                comparisons += 1
    assert comparisons == 1, f"an affirmative reader of 'passed' was added ({comparisons})"

    # No new property on the record either -- the two that exist are C-056a's and C-041's.
    properties = {
        name for name, value in vars(ReleaseStatus).items() if isinstance(value, property)
    }
    assert properties == {"semantic_confirmed", "produced_pwml"}, properties

    # ...and ``semantic_confirmed`` still answers only to an actual pass, unchanged.
    for evaluation, expected in (
        (SEMANTIC_PASSED, True), (SEMANTIC_NOT_EVALUATED, False), (SEMANTIC_FAILED, False),
    ):
        status = classify_release_status(
            _COVERAGE, strict_gates_passed=True, semantic_evaluation=evaluation,
            semantic_check_evaluability=[(SEMANTIC_GATING_CHECKS[0], True, "")],
        )
        assert status.semantic_confirmed is expected, evaluation


def test_the_record_changes_no_status_no_cap_and_no_eligibility() -> None:
    """The carrier must be inert. Same run, every evaluability width -- one outcome.

    A record that could move a verdict would be a second gate wearing a record's name, and
    would make C-056b's subtractive-only design unsound.
    """

    baseline = classify_release_status(_COVERAGE, strict_gates_passed=True,
                                       semantic_evaluation=SEMANTIC_PASSED)
    for width in range(len(SEMANTIC_GATING_CHECKS) + 1):
        status = _classify(_Report(inapplicable=SEMANTIC_GATING_CHECKS[width:]))
        if status.semantic_evaluation != SEMANTIC_PASSED:
            continue
        assert status.status == baseline.status
        assert status.strict_acceptance_eligible == baseline.strict_acceptance_eligible
        assert status.reasons == baseline.reasons


def test_omitting_the_input_changes_exactly_one_serialized_key() -> None:
    """The GOLDEN delta's claim, tested rather than asserted in a comment.

    A caller that passes nothing -- ``batch/driver.py:1770`` is one, and it is NOT edited
    by this card -- gets the record it got before, plus one new key whose value is the
    empty list meaning "not recorded". Zero keys removed, zero shared values changed.
    """

    without = classify_release_status(_COVERAGE, strict_gates_passed=True).to_dict()
    with_empty = classify_release_status(
        _COVERAGE, strict_gates_passed=True, semantic_check_evaluability=(),
    ).to_dict()
    assert without == with_empty

    new_keys = set(without) - {
        "status", "pipeline_executed", "strict_gates_passed", "semantic_evaluation",
        "semantic_not_evaluated_reason", "semantic_failed_checks",
        "strict_acceptance_eligible", "completeness", "missing_anchors",
        "retrieval_attempts", "expansion_blocked_reason", "coverage_evaluated", "reasons",
    }
    assert new_keys == {"semantic_check_evaluability"}
    assert without["semantic_check_evaluability"] == []


def test_a_classification_rebuilt_from_its_own_serialization_keeps_its_evaluability() -> None:
    """``to_dict`` writes mappings; the parameter accepts them back.

    Without this, a record round-tripped through JSON would silently flatten to "not
    recorded" -- an absence indistinguishable from a measured one, which is the exact
    failure mode this card exists to remove.
    """

    original = _classify(_Report(inapplicable=SEMANTIC_GATING_CHECKS[2:]))
    serialized = json.loads(json.dumps(original.to_dict()))
    rebuilt = classify_release_status(
        _COVERAGE,
        strict_gates_passed=True,
        semantic_evaluation=serialized["semantic_evaluation"],
        semantic_check_evaluability=serialized["semantic_check_evaluability"],
    )
    assert rebuilt.to_dict()["semantic_check_evaluability"] == serialized[
        "semantic_check_evaluability"
    ]
    assert rebuilt.semantic_check_evaluability == original.semantic_check_evaluability
