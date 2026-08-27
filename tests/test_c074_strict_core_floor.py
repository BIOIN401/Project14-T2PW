"""C-074: ``release_ready`` needs a real connected pathway AND a stated request.

Two defects, one seam, both registered from T-105 as ``product_contract_violation``
and both reaching ``release_ready`` with a bare ``pathway.pwml`` -- which
``PRODUCT_CONTRACT.md`` section 13 defines as "ship it, no review needed".

**F-101, arm A.** A leg whose payload held ONE reaction and two interactions
shipped as a multi-step pathway. Every requested anchor matched, so C-072's
unmatched-anchor cap correctly abstained; coverage was 1.0; every structural gate
was green. What no gate asked was whether the surviving reactions form a
*pathway*. The coverage floor next to it counts accepted processes INCLUDING
interactions at a threshold of 1, so it read 3 where the chemically connected core
read 1. Gold's ``export_rationale`` on that leg: emitting the requested pathway
from that paper "requires importing seven steps the paper never mentions".

**F-100, arm B.** The batch's declared NEGATIVE CONTROL shipped ``release_ready``
against a context carrying **no pathway name at all**. The "requested core" was
Stage 0's reading of the paper's own key compounds and proteins, so coverage
scored 6/6 against terms derived from the paper and ``completeness`` came back
1.0. "Nothing was asked for" was read as "nothing is missing".

WHAT THIS MODULE PROVES, in the order the card asks for it:

* the two G9 base-failing behavioural proofs, one per arm, driven through
  ``quarantine_and_close`` -- the production path -- so neither depends on a
  symbol that does not exist at the base SHA;
* NON-VACUITY: a genuinely multi-step pathway with a stated request still reaches
  ``release_ready``. Without this the card would be a blanket demotion;
* interactions cannot clear the floor, for any number of them;
* PRESERVATION (merge rule 7): the demoted leg keeps its payload, its surviving
  processes and its coverage record, and lands on ``review_required`` -- never
  ``diagnostic_only``, never dropped;
* the UNDECLARED regime is untouched, which is arm B's regression risk;
* a real-artifact replay of the two named legs;
* the FULL-CORPUS replay (card section 7): both arms over every committed
  ``quarantine_report.json``, with the control that the replay reproduces each
  leg's committed status before either arm is applied. C-073 was rejected this
  sprint for chartering a rule on ten legs that stripped 41 legitimate rows over
  the corpus, so the measurement is a test rather than a probe;
* no benchmark paper, PMC id or gold pathway name entered ``src/`` in this diff.

NO SERIALIZED KEY WAS ADDED, so no ``schema_version`` moves: arm A's size travels
as a classifier ARGUMENT and both arms record themselves by appending to the
existing ``reasons`` list.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    REASON_COVERAGE_NOT_EVALUATED,
    RELEASE_READY,
    REVIEW_REQUIRED,
    CoverageVerdict,
    classify_release_status,
)
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    StrictQuarantineResult,
    quarantine_and_close,
)

# ── Why C-074's own names are NOT imported at module scope ──────────────────
#
# THIS MODULE MUST IMPORT AT THE BASE SHA. G9 requires the two proofs below to
# fail BEHAVIOURALLY at ``9cb491c``, and "symbol absence is not proof": a
# module-level ``from ... import MIN_CONNECTED_CORE_REACTIONS`` would turn the
# whole file into a collection error there, which proves only that a name was
# added. Every symbol imported above exists at base, so a base run of this file
# collects and reports the two demotions that did not happen -- as assertion
# failures on a status, which is the fact the card is about.
#
# The reason strings are pinned here as LITERALS for the same reason they are
# pinned anywhere: they are WIRE values that travel into ``quarantine_report.json``
# and a consumer string-matches them. ``test_the_reason_vocabulary_is_the_
# production_vocabulary`` below imports the production constants -- function-local,
# so collection is unaffected -- and asserts these literals ARE those constants,
# so the two can never drift.
MIN_CONNECTED_CORE_REACTIONS = 2
REASON_CONNECTED_CORE_BELOW_FLOOR = "connected_core_below_minimum"
REASON_REQUESTED_PATHWAY_NOT_STATED = "requested_pathway_not_stated"


def requested_scope_is_a_single_reaction(pathway_context: Any) -> bool:
    """Function-local import of C-074's request-side predicate; see above."""

    from t2pw.pipeline.strict_quarantine import (
        requested_scope_is_a_single_reaction as impl,
    )

    return impl(pathway_context)


def largest_connected_core_reactions(semantic_report: Any) -> Optional[int]:
    """Function-local import of C-074's connected-core reader; see above."""

    from t2pw.pipeline.strict_quarantine import (
        largest_connected_core_reactions as impl,
    )

    return impl(semantic_report)


def test_the_reason_vocabulary_is_the_production_vocabulary() -> None:
    """The literals above ARE the shipped constants, so pinning them here cannot
    quietly stop pinning anything."""

    from t2pw.pipeline import release_status as production

    assert MIN_CONNECTED_CORE_REACTIONS == production.MIN_CONNECTED_CORE_REACTIONS
    assert REASON_CONNECTED_CORE_BELOW_FLOOR == production.REASON_CONNECTED_CORE_BELOW_FLOOR
    assert REASON_REQUESTED_PATHWAY_NOT_STATED == production.REASON_REQUESTED_PATHWAY_NOT_STATED
    assert MIN_CONNECTED_CORE_REACTIONS > 1, "a floor of 1 is the floor this card replaces"

from test_strict_quarantine_contract_alignment import _base  # noqa: E402

#: The base SHA the card names. Used only by the diff-hygiene test, which skips
#: cleanly where it is not reachable.
BASE_SHA = "9cb491c9867a254c7dbf29bdbb01789803680ef9"

#: The two production files C-074 owns. The diff guard at the bottom of this
#: module is scoped to these rather than to all of ``src/``.
#:
#: NOT A WEAKENING. Every production line this card adds is in one of the two, so
#: the guard still reads the whole C-074 diff. What the scope removes is reach
#: this test never had any business having: once merged, an unscoped guard would
#: police every LATER card additions to ``src/`` from inside a C-074 test file,
#: failing on work this module knows nothing about. Boundary enforcement belongs
#: to the merge gate, not here.
OWNED_SOURCE_FILES = (
    "src/t2pw/pipeline/release_status.py",
    "src/t2pw/pipeline/strict_quarantine.py",
)

#: The committed run the two findings were measured on.
RUN_DIR = ROOT / "runs_verify" / "2026-08-22_2147"


# ── Fixtures: one request, two payload shapes ───────────────────────────────
#
# Shaped rather than replayed, for the reason
# ``test_strict_quarantine_release_seam`` gives: no archived PAYLOAD carries a
# Stage-0 context, and without one the run is in the undeclared regime where
# neither of these defects can arise at all. The real legs are replayed further
# down, from their committed reports.

#: A STATED request: it names a pathway, names an organism, names anchors the
#: payload covers, and names more than one step. Every field is request-side --
#: nothing here is read back off the payload.
STATED_REQUEST: Dict[str, Any] = {
    "pathway_name": "glutathione biosynthesis",
    "likely_organism": "Homo sapiens",
    "key_compounds": ["L-glutamate", "gamma-glutamylcysteine", "glutathione"],
    "main_subprocesses": [
        "gamma-glutamylcysteine formation",
        "glutathione formation",
        "glutathione turnover",
    ],
}

#: F-100's shape: a context that CLAIMS a requested core -- it carries key
#: compounds, they all match, coverage comes back 1.0 -- while naming no pathway.
UNSTATED_REQUEST: Dict[str, Any] = dict(STATED_REQUEST, pathway_name="")

#: A stated request for exactly ONE step. The only exemption arm A's floor has,
#: and it is read from the request. Measured over all 38 committed legs, none
#: names a single subprocess, so this shape exists here and nowhere in the corpus.
SINGLE_STEP_REQUEST: Dict[str, Any] = dict(
    STATED_REQUEST, main_subprocesses=["gamma-glutamylcysteine formation"]
)


def multi_step_payload() -> Dict[str, Any]:
    """Two reactions sharing gamma-glutamylcysteine: a connected core of 2.

    ``metadata.key_compounds`` is dropped for the reason the release-seam module
    gives -- payload-derived anchors are anchors taken from the survivors, which
    is not a request.
    """

    payload = _base()
    payload["metadata"].pop("key_compounds", None)
    return payload


def one_reaction_payload(interactions: int = 2) -> Dict[str, Any]:
    """F-101's shape: ONE reaction plus ``interactions`` interactions.

    The interactions are what make this the exact 3-vs-1 discrepancy rather than
    a payload that is merely small: they are ``core_accepted`` like any other
    admitted process, so the coverage floor counts 1 + ``interactions`` of them
    and clears its threshold of 1 comfortably, while the chemically connected
    core is 1 whatever ``interactions`` is.

    One endpoint is ``glutathione`` on purpose. Dropping the second reaction
    leaves that compound unreferenced, closure removes it, and the payload then
    stops mentioning the requested pathway at all -- which fails the ANCHOR check
    and would demote the leg for a reason that is not this card's. Keeping it
    referenced is what makes the base leg ``release_ready``, which is the whole
    premise of the proof below.
    """

    payload = multi_step_payload()
    payload["processes"]["reactions"] = [payload["processes"]["reactions"][0]]
    endpoints = [
        ("glutathione synthetase complex", "glutathione"),
        ("glutamate-cysteine ligase complex", "gamma-glutamylcysteine"),
    ]
    payload["processes"]["interactions"] = [
        {
            "name": f"regulatory contact {index + 1}",
            "entity_1": endpoints[index % len(endpoints)][0],
            "entity_1_type": "protein_complex",
            "entity_2": endpoints[index % len(endpoints)][1],
            "entity_2_type": "compound",
            "interaction_type": "binding",
            "biological_state": "cytosol",
        }
        for index in range(interactions)
    ]
    return payload


def run(payload: Dict[str, Any], context: Optional[Dict[str, Any]]) -> StrictQuarantineResult:
    return quarantine_and_close(
        deepcopy(payload), strict_db=True, pathway_context=deepcopy(context) if context else context
    )


def release(result: StrictQuarantineResult) -> Dict[str, Any]:
    return dict(result.quarantine_report["release"])


def reasons_of(result: StrictQuarantineResult) -> List[str]:
    return [str(reason) for reason in release(result)["reasons"]]


def has_reason(result: StrictQuarantineResult, prefix: str) -> bool:
    return any(reason.split(":", 1)[0] == prefix for reason in reasons_of(result))


# ── 1. G9 base-failing proof, ARM A ─────────────────────────────────────────


def test_g9_arm_a_one_reaction_and_two_interactions_is_not_release_ready() -> None:
    """**G9 BASE-FAILING PROOF, ARM A (F-101).**

    Every premise the card states is asserted before the verdict, because the
    proof is worthless if the leg was demoted by some other gate: all anchors
    matched, coverage 1.0, semantics PASSED, every structural gate green, the
    coverage floor SATISFIED at 3 accepted processes -- and a chemically
    connected core of 1.

    At ``9cb491c`` this call returns ``release_ready`` with
    ``strict_acceptance_eligible True`` and an empty ``reasons`` list. Nothing
    about the call shape differs between base and tip, so the failure is
    behavioural and not symbol absence.
    """

    result = run(one_reaction_payload(interactions=2), STATED_REQUEST)
    coverage = result.coverage
    record = release(result)

    # -- the premises, so a later failure cannot be misread ------------------
    assert coverage["requested_core_declared"] is True
    assert coverage["requested_core_source"] == "pathway_context"
    assert coverage["unmatched_terms"] == [], "C-072's cap must abstain here"
    assert float(coverage["coverage_ratio"]) == 1.0
    assert coverage["minimum_core_satisfied"] is True
    assert coverage["core_accepted_processes"] == 3, "1 reaction + 2 interactions"
    assert coverage["surviving_processes"] == 3
    assert record["strict_gates_passed"] is True
    assert record["semantic_evaluation"] == "passed"
    assert len(result.payload["processes"]["reactions"]) == 1

    # -- the verdict --------------------------------------------------------
    assert record["status"] == REVIEW_REQUIRED
    assert record["strict_acceptance_eligible"] is False
    assert f"{REASON_CONNECTED_CORE_BELOW_FLOOR}:1<{MIN_CONNECTED_CORE_REACTIONS}" in record["reasons"]


# ── 2. G9 base-failing proof, ARM B ─────────────────────────────────────────


def test_g9_arm_b_a_context_that_states_no_pathway_cannot_be_satisfied() -> None:
    """**G9 BASE-FAILING PROOF, ARM B (F-100).**

    A MULTI-STEP payload -- two connected reactions, so arm A has nothing to say
    about it -- judged against a context that declares a core and names no
    pathway. Coverage comes back 1.0 with no unmatched anchor, exactly as it did
    on the negative control, and at ``9cb491c`` that is ``release_ready``.

    The two arms are deliberately separated here: this leg clears arm A's floor,
    so the demotion can only be arm B's.
    """

    result = run(multi_step_payload(), UNSTATED_REQUEST)
    coverage = result.coverage
    record = release(result)

    assert coverage["requested_core_declared"] is True
    assert coverage["unmatched_terms"] == []
    assert float(coverage["coverage_ratio"]) == 1.0
    assert coverage["minimum_core_satisfied"] is True
    assert str((coverage["requested_context"] or {}).get("pathway_name") or "") == ""
    assert record["strict_gates_passed"] is True
    assert record["semantic_evaluation"] == "passed"

    assert record["status"] == REVIEW_REQUIRED
    assert record["strict_acceptance_eligible"] is False
    assert REASON_REQUESTED_PATHWAY_NOT_STATED in record["reasons"]
    # Recorded as UNEVALUABLE, following the vocabulary that already existed --
    # never as a fabricated coverage failure.
    assert REASON_COVERAGE_NOT_EVALUATED in record["reasons"]
    assert not has_reason(result, REASON_CONNECTED_CORE_BELOW_FLOOR)


def test_the_connected_core_reader_treats_an_absent_graph_as_unmeasured() -> None:
    """``largest_connected_core_reactions`` is the ONE place the size is read.
    A report with no graph, or a graph with no size, is NOT MEASURED -- never
    zero, which would demote every such run."""

    class _Report:
        def __init__(self, graph: Any) -> None:
            self.graph = graph

    assert largest_connected_core_reactions(None) is None
    assert largest_connected_core_reactions(_Report(None)) is None
    assert largest_connected_core_reactions(_Report({})) is None
    assert largest_connected_core_reactions(_Report({"largest_core_size": "x"})) is None
    assert largest_connected_core_reactions(_Report({"largest_core_size": 0})) == 0
    assert largest_connected_core_reactions(_Report({"largest_core_size": 4})) == 4


# ── 3. NON-VACUITY, both arms ───────────────────────────────────────────────


def test_non_vacuity_a_multi_step_pathway_with_a_stated_request_still_ships() -> None:
    """**NON-VACUITY.** Neither arm is a blanket demotion.

    Two connected reactions, a request that names its pathway, every anchor
    matched: ``release_ready``, with ``strict_acceptance_eligible True`` and
    neither new reason on the record. If this ever goes red the card has stopped
    being a floor and become a ban.
    """

    result = run(multi_step_payload(), STATED_REQUEST)
    record = release(result)

    assert record["status"] == RELEASE_READY
    assert record["strict_acceptance_eligible"] is True
    assert not has_reason(result, REASON_CONNECTED_CORE_BELOW_FLOOR)
    assert REASON_REQUESTED_PATHWAY_NOT_STATED not in record["reasons"]


def test_a_request_that_asks_for_one_step_is_exempt_from_the_floor() -> None:
    """The floor's ONE exemption, and it is read from the REQUEST.

    The same one-reaction payload that is demoted above reaches ``release_ready``
    when the request itself names a single subprocess. The payload is byte for
    byte identical between the two tests -- only the request differs -- which is
    what proves the exemption cannot be earned by what survived.
    """

    payload = one_reaction_payload(interactions=2)
    demoted = run(payload, STATED_REQUEST)
    exempt = run(payload, SINGLE_STEP_REQUEST)

    assert requested_scope_is_a_single_reaction(SINGLE_STEP_REQUEST) is True
    assert requested_scope_is_a_single_reaction(STATED_REQUEST) is False
    assert release(demoted)["status"] == REVIEW_REQUIRED
    assert release(exempt)["status"] == RELEASE_READY


def test_the_exemption_cannot_be_reached_without_a_stated_pathway() -> None:
    """A context that names no pathway asked for nothing, so it cannot have asked
    for one reaction either. Otherwise arm B's own legs would exempt themselves
    from arm A."""

    assert requested_scope_is_a_single_reaction(dict(SINGLE_STEP_REQUEST, pathway_name="")) is False
    assert requested_scope_is_a_single_reaction(None) is False
    assert requested_scope_is_a_single_reaction({}) is False
    # A one-anchor requested core is not a scope statement: "one anchor" is the
    # most ordinary way to ask for a long pathway, and nothing reads it here.
    assert (
        requested_scope_is_a_single_reaction(
            {"pathway_name": "a pathway", "requested_core": ["one anchor"], "key_compounds": ["one"]}
        )
        is False
    )


# ── 4. Interactions do not count toward the floor ───────────────────────────


@pytest.mark.parametrize("interactions", [2, 3, 5])
def test_interactions_never_clear_the_connected_core_floor(interactions: int) -> None:
    """**The exact 3-vs-1 discrepancy.** One reaction plus N interactions, N >= 2.

    The coverage floor counts 1 + N and is satisfied every time; the connected
    core is 1 every time. ``bench.semantic._connected_core`` walks
    ``("reactions", "transports", "reaction_coupled_transports")`` and
    ``interactions`` is not in that tuple, so no number of them can reach the
    floor.
    """

    result = run(one_reaction_payload(interactions=interactions), STATED_REQUEST)
    coverage = result.coverage

    assert coverage["surviving_processes"] == 1 + interactions
    assert coverage["core_accepted_processes"] == 1 + interactions, (
        "every admitted process counts toward the coverage floor, interactions included"
    )
    assert coverage["minimum_core_satisfied"] is True, "the coverage floor is cleared"
    assert release(result)["status"] == REVIEW_REQUIRED
    assert f"{REASON_CONNECTED_CORE_BELOW_FLOOR}:1<{MIN_CONNECTED_CORE_REACTIONS}" in release(result)["reasons"]


# ── 5. Preservation (merge rule 7) ──────────────────────────────────────────


def test_the_demoted_leg_keeps_everything_and_is_never_diagnostic_only() -> None:
    """**PRESERVATION.** A cap is one step down, not a deletion.

    The demoted leg is compared against the SAME payload run without the request
    that demotes it: same surviving processes, same resulting payload hash, same
    coverage record. The only differences are the status, the eligibility and the
    added reason. ``diagnostic_only`` is asserted against by name, and ``ok``
    stays True so the graph is still frozen and still exported.
    """

    payload = one_reaction_payload(interactions=2)
    demoted = run(payload, STATED_REQUEST)
    exempt = run(payload, SINGLE_STEP_REQUEST)

    assert release(demoted)["status"] == REVIEW_REQUIRED
    assert release(demoted)["status"] != DIAGNOSTIC_ONLY
    assert demoted.ok is True, "a cap does not refuse the freeze"

    assert demoted.payload == exempt.payload
    assert (
        demoted.quarantine_report["resulting_payload_hash"]
        == exempt.quarantine_report["resulting_payload_hash"]
    )
    # Everything the coverage verdict MEASURED is identical; only the request it
    # was measured against differs, which is the one input that differs.
    measured = lambda verdict: {k: v for k, v in verdict.items() if k != "requested_context"}
    assert measured(demoted.coverage) == measured(exempt.coverage), (
        "the coverage RECORD is untouched by the cap"
    )
    assert demoted.coverage["requested_context"] == STATED_REQUEST
    assert demoted.coverage["surviving_processes"] == 3
    assert demoted.quarantine_report["refusal_reasons"] == exempt.quarantine_report["refusal_reasons"]


def test_arm_b_leaves_the_measured_coverage_record_on_the_row() -> None:
    """Arm B records UNEVALUABLE without overwriting what was measured.

    The ratio that was computed stays on the coverage record as the evidence it
    is -- a reviewer has to be able to see that 1.0 was scored against terms
    nobody asked for. Replacing it with a number the cap invented would destroy
    exactly the fact that makes the leg reviewable.
    """

    result = run(multi_step_payload(), UNSTATED_REQUEST)

    assert float(result.coverage["coverage_ratio"]) == 1.0
    assert result.coverage["minimum_core_satisfied"] is True
    assert result.coverage["requested_core_declared"] is True
    assert release(result)["completeness"] == 1.0
    assert release(result)["status"] == REVIEW_REQUIRED


# ── 6. The undeclared regime is unchanged ───────────────────────────────────


def test_a_payload_with_no_context_at_all_is_untouched_by_arm_b() -> None:
    """**ARM B'S REGRESSION RISK, pinned.**

    No context reaches the check, so ``requested_context`` is ``None``, relevance
    is unjudgeable and ``completeness`` is ``None`` -- all of which was already
    true. The new refusal is specifically about a context that CLAIMS to declare
    a core while stating no pathway, and a run with no context makes no such
    claim.
    """

    result = run(multi_step_payload(), None)
    record = release(result)

    assert result.coverage["requested_core_declared"] is False
    assert result.coverage["requested_context"] is None
    assert record["completeness"] is None
    assert record["status"] == RELEASE_READY
    assert REASON_REQUESTED_PATHWAY_NOT_STATED not in record["reasons"]


def test_the_undeclared_regime_can_never_reach_arm_b_at_the_classifier() -> None:
    """The predicate itself, over the shapes the classifier actually receives."""

    undeclared_no_context = CoverageVerdict(
        {"requested_core_declared": False, "requested_context": None, "requested_core_source": "none"}
    )
    undeclared_with_context = CoverageVerdict(
        {
            "requested_core_declared": False,
            "requested_context": {"pathway_name": ""},
            "requested_core_source": "none",
        }
    )
    declared_no_context = CoverageVerdict(
        {"requested_core_declared": True, "requested_context": None, "requested_core_source": "payload"}
    )
    # A context that carries only a PathWhiz CATEGORY, with the terms scraped off
    # the payload because nobody handed a request over. It never claimed to
    # declare a core, and ``test_semantic_release_gating`` ships exactly this
    # shape -- so arm B must not touch it.
    declared_from_payload_beside_a_category = CoverageVerdict(
        {
            "requested_core_declared": True,
            "requested_context": {"pathway_subject": "Metabolic"},
            "requested_core_source": "payload",
        }
    )
    declared_explicit_argument = CoverageVerdict(
        {
            "requested_core_declared": True,
            "requested_context": {"pathway_name": ""},
            "requested_core_source": "explicit_argument",
        }
    )
    declared_unstated = CoverageVerdict(
        {
            "requested_core_declared": True,
            "requested_context": {"pathway_name": "   "},
            "requested_core_source": "pathway_context",
        }
    )

    assert undeclared_no_context.declares_core_without_stating_a_pathway is False
    assert undeclared_with_context.declares_core_without_stating_a_pathway is False
    assert declared_no_context.declares_core_without_stating_a_pathway is False
    assert declared_from_payload_beside_a_category.declares_core_without_stating_a_pathway is False, (
        "a context that did not supply the terms never claimed to declare a core"
    )
    assert declared_explicit_argument.declares_core_without_stating_a_pathway is False, (
        "an explicit requested_core argument IS a stated request"
    )
    assert declared_unstated.declares_core_without_stating_a_pathway is True


def test_an_unmeasured_connected_core_never_demotes() -> None:
    """``None`` is not zero. A caller that does not measure connectivity gets the
    record it always got, which is what keeps every pre-C-074 caller unmoved."""

    coverage = CoverageVerdict(
        {
            "requested_core_declared": True,
            "requested_context": {"pathway_name": "a pathway"},
            "requested_core_source": "pathway_context",
            "surviving_processes": 3,
            "minimum_core_satisfied": True,
            "coverage_ratio": 1.0,
            "unmatched_terms": [],
            "reasons": [],
        }
    )
    unmeasured = classify_release_status(coverage, strict_gates_passed=True)
    measured = classify_release_status(coverage, strict_gates_passed=True, connected_core_reactions=1)

    assert unmeasured.status == RELEASE_READY
    assert unmeasured.reasons == ()
    assert measured.status == REVIEW_REQUIRED


def test_both_caps_are_subtractive_and_never_deepen_a_status() -> None:
    """Neither arm may CREATE a ``release_ready`` and neither may go past
    ``review_required``. Asserted against the two statuses the chain above them
    can produce."""

    already_review = CoverageVerdict(
        {
            "requested_core_declared": True,
            "requested_context": {"pathway_name": ""},
            "requested_core_source": "pathway_context",
            "surviving_processes": 2,
            "minimum_core_satisfied": False,
            "reasons": ["requested_core_coverage_below_minimum:0.100<0.500"],
        }
    )
    empty = CoverageVerdict(
        {
            "requested_core_declared": True,
            "requested_context": {"pathway_name": ""},
            "requested_core_source": "pathway_context",
            "surviving_processes": 0,
            "minimum_core_satisfied": False,
            "reasons": ["no_surviving_process"],
        }
    )

    capped = classify_release_status(
        already_review, strict_gates_passed=True, connected_core_reactions=0
    )
    diagnostic = classify_release_status(empty, strict_gates_passed=True, connected_core_reactions=0)
    blocked = classify_release_status(
        already_review, strict_gates_passed=False, connected_core_reactions=0
    )

    assert capped.status == REVIEW_REQUIRED
    assert REASON_CONNECTED_CORE_BELOW_FLOOR not in [r.split(":", 1)[0] for r in capped.reasons]
    assert diagnostic.status == DIAGNOSTIC_ONLY, "a cap never deepens to diagnostic_only"
    assert blocked.status == DIAGNOSTIC_ONLY


# ── 7. Real-artifact replay of the two named legs ───────────────────────────
#
# Follows ``tests/test_strict_quarantine_real_artifact_replay.py``: the run
# directory is a working directory, not a committed fixture obligation, so every
# test here skips cleanly where it is absent.


def _load(leg: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    report = json.loads((leg / "quarantine_report.json").read_text(encoding="utf-8"))
    payload_path = leg / "final_mapped.json"
    payload = (
        json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.is_file() else None
    )
    return report, payload


def _connected_core_size(report: Dict[str, Any], payload: Dict[str, Any]) -> int:
    """The connected core of the graph the leg's release record was computed on.

    ``final_mapped.json`` is that graph already for a strict leg -- quarantine
    dropped the rest before it was written. For a research leg it is the
    UNREDUCED payload, because research applies nothing, so the rows quarantine
    would have removed are subtracted here by name. Measured across the corpus
    every ``release_ready`` leg has zero quarantined rows, so the two readings
    agree on every leg the table below reports; the subtraction is here so the
    replay stays faithful if that ever stops being true.
    """

    from t2pw.bench.semantic import _cofactor_names, _connected_core, _processes

    quarantined = {
        str(row.get("name") or "").strip().casefold()
        for row in report.get("quarantined") or []
    }
    processes = {
        bucket: [
            row
            for row in rows
            if str(row.get("name") or "").strip().casefold() not in quarantined
        ]
        for bucket, rows in _processes(payload).items()
    }
    return int(_connected_core(processes, _cofactor_names()).get("largest_core_size", 0))


class _ArmBDisabled(CoverageVerdict):
    """The coverage verdict with arm B's ONE predicate switched off.

    Arm B reads nothing but the verdict, so it cannot be disabled by withholding
    an argument the way arm A can. Overriding exactly the property it reads --
    and nothing else -- is what gives the corpus measurement below a control that
    differs from the treatment in C-074 and in nothing else. Test-side only: no
    production caller ever builds this.
    """

    @property
    def declares_core_without_stating_a_pathway(self) -> bool:
        return False


def _replay(report: Dict[str, Any], payload: Dict[str, Any], *, arms: bool) -> Any:
    """One committed leg back through the production classifier.

    ``arms=False`` is the CONTROL: the classifier exactly as it stands at this
    tip with C-074 removed and nothing else changed. ``arms=True`` is the
    treatment. Everything else fed in is read off the committed report; nothing
    is invented and nothing is read off the gold set.

    THE CONTROL IS NOT THE COMMITTED STATUS, and that is deliberate. Six of the
    nine committed ``release_ready`` legs were written before C-072's
    unmatched-anchor cap merged, so replaying them at this tip demotes them for a
    reason that is not this card's. Measuring C-074 against the committed status
    would credit this card with C-072's work; measuring it against the tip
    without C-074 attributes exactly what C-074 does.
    """

    record = report.get("release") or {}
    invariants = report.get("strict_invariants") or {}
    raw = report.get("coverage") or {}
    coverage = CoverageVerdict(raw) if arms else _ArmBDisabled(raw)
    return classify_release_status(
        coverage,
        pipeline_executed=bool(record.get("pipeline_executed", True)),
        strict_gates_passed=bool(record.get("strict_gates_passed")),
        serializable_without_invention=not (invariants.get("unexportable_entities") or []),
        semantic_evaluation=str(record.get("semantic_evaluation") or "not_evaluated"),
        semantic_not_evaluated_reason=str(record.get("semantic_not_evaluated_reason") or ""),
        semantic_failed_checks=list(record.get("semantic_failed_checks") or []),
        connected_core_reactions=_connected_core_size(report, payload) if arms else None,
        single_reaction_scope_requested=requested_scope_is_a_single_reaction(
            raw.get("requested_context")
        ),
    )


def _arms_that_fired(status: Any) -> List[str]:
    arms: List[str] = []
    if any(str(r).split(":", 1)[0] == REASON_CONNECTED_CORE_BELOW_FLOOR for r in status.reasons):
        arms.append("arm A")
    if REASON_REQUESTED_PATHWAY_NOT_STATED in status.reasons:
        arms.append("arm B")
    return arms


def _committed_legs() -> List[Tuple[str, Path]]:
    """Every COMMITTED ``quarantine_report.json``, by ``run/paper/mode`` label.

    Read from ``git ls-files``, not from a filesystem glob: the corpus section 7
    charters the rule against is the committed one, and a working directory that
    happens to hold an extra overnight run must not silently change what the
    measurement below is measuring.
    """

    try:
        listed = subprocess.run(
            ["git", "ls-files", "runs_verify/*/papers/*/*/quarantine_report.json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git here
        return []
    if listed.returncode != 0:
        return []
    out: List[Tuple[str, Path]] = []
    for line in sorted(listed.stdout.splitlines()):
        path = ROOT / line.strip()
        if not line.strip() or not path.is_file():
            continue
        leg = path.parent
        out.append((f"{leg.parents[2].name}/{leg.parent.name}/{leg.name}", leg))
    return out


@pytest.mark.parametrize("paper", ["PMC12856317", "PMC13231680"])
def test_the_two_named_legs_replay_to_review_required(paper: str) -> None:
    """**REAL-ARTIFACT REPLAY.** The two legs F-101 and F-100 were registered
    from, replayed from their committed reports and payloads.

    Both recorded ``release_ready`` and emitted a bare ``pathway.pwml``. Both are
    still ``release_ready`` under the control -- the tip classifier without
    C-074 -- and both must now reach ``review_required``, keeping their PWML.
    The paper ids appear HERE, in a test, over committed artifacts; never in
    ``src/``, which the diff test below asserts.
    """

    leg = RUN_DIR / "papers" / paper / "strict"
    if not (leg / "quarantine_report.json").is_file():
        pytest.skip(f"no committed strict leg for {paper} under {RUN_DIR.name}")
    report, payload = _load(leg)
    if payload is None:
        pytest.skip(f"{paper}/strict carries no final_mapped.json")

    assert str((report.get("release") or {}).get("status")) == RELEASE_READY, (
        "the committed leg is not release_ready; this replay is about a different run"
    )
    control = _replay(report, payload, arms=False)
    assert control.status == RELEASE_READY, (
        "the leg is already demoted without C-074; the replay would prove nothing"
    )

    applied = _replay(report, payload, arms=True)
    assert applied.status == REVIEW_REQUIRED
    assert applied.status != DIAGNOSTIC_ONLY
    assert applied.strict_acceptance_eligible is False
    assert _arms_that_fired(applied), "demoted, but by neither arm of this card"


# -- The full-corpus measurement (card section 7) ----------------------------

#: Every leg C-074 demotes, over the whole committed corpus, with the arm(s) that
#: fired. A demotion not named here fails the test below rather than being
#: absorbed -- C-073 was rejected this sprint for a rule chartered on ten legs
#: that stripped 41 legitimate rows over the corpus.
#:
#: WHY EACH IS CORRECT RATHER THAN COLLATERAL:
#:
#: ``PMC12856317/strict``   F-101 itself. ONE reaction and two interactions, with
#:   every requested anchor matched, shipped as a multi-step pathway. Gold's
#:   ``export_rationale``: a single reaction cannot constitute an exportable
#:   multi-step pathway, and emitting the requested one from that paper would
#:   require importing seven steps the paper never mentions.
#: ``PMC13231680/strict``   F-100 itself, and the batch's declared NEGATIVE
#:   CONTROL. One reaction AND a context naming no pathway, so both arms fire and
#:   both are recorded. Gold: nothing requested is exportable at any level of
#:   partiality; the correct outcome is an empty pathway plus a rejection reason.
#: ``PMC13231680/research`` the same negative-control paper in research mode. Two
#:   reactions sharing no non-cofactor metabolite, so there is no chain and the
#:   largest connected core is 1. Research mode records its classification as a
#:   FINDING (``release.applied`` is False), so nothing it emits changes.
CORPUS_DEMOTED: Dict[str, str] = {
    "2026-08-22_2147/PMC12856317/strict": "arm A",
    "2026-08-22_2147/PMC13231680/research": "arm A",
    "2026-08-22_2147/PMC13231680/strict": "arm A + arm B",
}

#: The other six committed ``release_ready`` legs, with the connected-core size
#: measured on each one's own graph. C-074 does not touch any of them: every one
#: is a multi-step pathway against a request that names itself. Demoting one of
#: these would be exactly the collateral damage this measurement exists to catch.
#:
#: They are NOT ``release_ready`` at this tip either -- C-072's unmatched-anchor
#: cap, merged after these runs were written, already demotes all six. That is
#: asserted below so the number cannot be read as C-074's doing.
CORPUS_UNTOUCHED_BY_C074: Dict[str, int] = {
    "2026-08-18_1328/PMC12096016/research": 5,
    "2026-08-21_1822/PMC12782028/research": 2,
    "2026-08-21_2057/PMC12452463/research": 4,
    "2026-08-21_2239/PMC12096016/research": 3,
    "2026-08-21_2239/PMC12452463/strict": 2,
    "2026-08-21_2239/PMC12782028/research": 2,
}


# -- C-092: the census, re-based from an equality to a floor ----------------
#
# C-074 wrote its corpus measurement as three literal equalities: 38 legs with a
# release record, 9 of them recorded ``release_ready``, and a demotion set equal
# to ``CORPUS_DEMOTED`` exactly. The committed corpus is APPEND-ONLY -- run
# directories land under ``runs_verify/`` as the sprint proceeds -- so every one
# of those equalities goes red on a legitimate addition. All three did: the
# census reached 60 and this module stayed red across five consecutive cards,
# each of which paid to re-derive that the red was pre-existing. Worse, while it
# stood, a genuine regression in the neighbouring semantic-denominator module was
# twice reported as "pre-existing" because a standing red had stopped being
# informative.
#
# What replaces them is not a bigger literal. It is the property C-074 was
# actually chartered to protect -- **no demotion is unaccounted for** -- restated
# so that the account can be settled per leg rather than by set equality:
#
#   * the census is a FLOOR, because a committed corpus cannot legitimately
#     SHRINK. A leg deleted, a leg silently dropped by ``_committed_legs()``
#     discovery, or reports corrupted so they stop carrying a release record all
#     drive the count DOWN and still go red;
#   * the historically measured legs must still be PRESENT and still fire exactly
#     the arms they were measured firing;
#   * every other demotion must be INDEPENDENTLY JUSTIFIED, and every other
#     preservation independently defensible.
#
# C-073 was rejected this sprint for chartering a rule on ten legs that stripped
# 41 legitimate rows over the corpus. That is the failure this measurement exists
# to catch, and "no demotion is unjustified" catches it strictly better than "the
# demotion set is this literal list", because the literal list stopped being
# checkable the moment the corpus grew.

#: C-074's measured census, kept as a FLOOR and never as an equality.
C074_CENSUS_FLOOR = 38
#: C-074's measured ``release_ready`` count. FLOOR, for the same reason.
C074_RECORDED_RELEASE_READY_FLOOR = 9


def _raw_declared_core_without_a_stated_pathway(report: Dict[str, Any]) -> bool:
    """Arm B's question, answered from the RAW committed JSON.

    Deliberately **not** ``CoverageVerdict.declares_core_without_stating_a_pathway``.
    An expectation derived from the very object it is validating cannot disagree
    with it; the whole point of justifying an unpinned demotion is that the
    justification is a SECOND derivation. This one reads the committed mapping's
    own keys and additionally requires the context to actually CARRY TERMS --
    which the production property never checks, inferring a declared core from the
    ``requested_core_declared`` flag alone. The two can therefore disagree, and
    ``test_nonvacuity_c092_an_unjustified_arm_b_demotion_turns_the_corpus_red``
    exhibits a leg on which they do.
    """

    coverage = report.get("coverage") or {}
    context = coverage.get("requested_context")
    if not isinstance(context, dict):
        return False
    carries_terms = any(
        context.get(key)
        for key in ("key_compounds", "key_proteins", "main_subprocesses", "subprocesses")
    )
    return bool(
        coverage.get("requested_core_declared")
        and str(coverage.get("requested_core_source") or "") == "pathway_context"
        and carries_terms
        and not str(context.get("pathway_name") or "").strip()
    )


def _independently_justified_arms(report: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    """Which arms this leg DESERVES, derived without asking the cap that fired.

    Arm A is measured with this module's ``_connected_core_size()``, which reads
    ``t2pw.bench.semantic`` -- a different module from the one production consults
    (``strict_quarantine.largest_connected_core_reactions``) -- and the request-side
    exemption is applied on top, so a leg whose request asks for exactly one step
    is NOT justified in being demoted for having one step. Arm B comes from the raw
    JSON above.
    """

    justified: List[str] = []
    context = (report.get("coverage") or {}).get("requested_context")
    if _connected_core_size(report, payload) < MIN_CONNECTED_CORE_REACTIONS and not (
        requested_scope_is_a_single_reaction(context)
    ):
        justified.append("arm A")
    if _raw_declared_core_without_a_stated_pathway(report):
        justified.append("arm B")
    return justified


def test_the_full_corpus_replay_demotes_nothing_it_cannot_justify() -> None:
    """**CARD SECTION 7, as a test**, re-based by C-092 so it stops rotting.

    Five things are asserted, and the first is the load-bearing one:

    1. **THE CAP PROPERTY, universally quantified.** For EVERY leg carrying a
       release record -- not only the ``release_ready`` ones -- the treatment is
       at or below the control, and never reaches ``diagnostic_only`` unless the
       control already did. This is the cap property and merge rule 7 together.
       It is derived, never pinned, and C-092 did not touch it.
    2. **The census can only GROW.** A committed corpus does not legitimately
       shrink; a floor hears loss, exclusion and corruption, and stays quiet for
       an addition.
    3. **The historically measured legs are still there and still behave.** Each
       named demotion still fires exactly the arms it was measured firing, and
       each named preservation still measures the connected core it was measured
       at.
    4. **No demotion is unjustified.** Every demotion -- named or not -- is
       independently justified by ``_independently_justified_arms()``, and its
       control status really was ``release_ready``, so the demotion is attributable
       to this card rather than to something else. That last per-leg statement is
       what C-074's ``control_release_ready == len(CORPUS_DEMOTED)`` count was
       reaching for; said per leg it survives a corpus that grows.
    5. **No preservation is indefensible.** A preserved ``release_ready`` leg that
       is NOT one of the six C-074 measured must clear the connected-core floor on
       its own graph AND name the pathway it was judged against. A silent
       preservation of a genuinely defective leg still goes red.
    """

    legs = _committed_legs()
    if not legs:
        pytest.skip("no committed runs_verify legs in this checkout")

    order = {RELEASE_READY: 2, REVIEW_REQUIRED: 1, DIAGNOSTIC_ONLY: 0}
    with_release = 0
    recorded_release_ready = 0
    demoted: Dict[str, str] = {}
    untouched: Dict[str, int] = {}
    justified: Dict[str, List[str]] = {}
    control_of: Dict[str, str] = {}
    pathway_named: Dict[str, str] = {}

    for label, leg in legs:
        report, payload = _load(leg)
        record = report.get("release") or {}
        recorded = str(record.get("status") or "")
        if recorded not in order or payload is None:
            continue
        with_release += 1
        control = _replay(report, payload, arms=False)
        applied = _replay(report, payload, arms=True)

        # 1. THE CAP PROPERTY. Every leg, every time.
        assert order[applied.status] <= order[control.status], (
            f"{label}: {control.status} -> {applied.status} is not a cap"
        )
        assert applied.status != DIAGNOSTIC_ONLY or control.status == DIAGNOSTIC_ONLY, (
            f"{label}: a cap reached diagnostic_only and the PWML would be dropped"
        )

        if recorded != RELEASE_READY:
            continue
        recorded_release_ready += 1
        control_of[label] = control.status
        context = (report.get("coverage") or {}).get("requested_context") or {}
        pathway_named[label] = str(context.get("pathway_name") or "").strip()
        arms = _arms_that_fired(applied)
        if arms:
            demoted[label] = " + ".join(arms)
            justified[label] = _independently_justified_arms(report, payload)
        else:
            untouched[label] = _connected_core_size(report, payload)

    # 2. The census is a FLOOR. Growth is legitimate; shrinkage never is.
    assert with_release >= C074_CENSUS_FLOOR, (
        f"the committed corpus has SHRUNK below C-074's measured census "
        f"({C074_CENSUS_FLOOR}): {with_release} legs carry a release record. A leg "
        f"was deleted, excluded from discovery, or corrupted so it no longer "
        f"carries one"
    )
    assert recorded_release_ready >= C074_RECORDED_RELEASE_READY_FLOOR, (
        f"the corpus holds fewer release_ready legs than C-074 measured "
        f"({C074_RECORDED_RELEASE_READY_FLOOR}): {recorded_release_ready}"
    )

    # 3. The historically measured legs are present and behave as measured.
    lost = sorted(set(CORPUS_DEMOTED) - set(demoted))
    assert not lost, (
        f"C-074 demoted these legs and no longer does: {lost}. Either the leg left "
        f"the committed corpus or the arm stopped firing on it"
    )
    for label, arms_measured in sorted(CORPUS_DEMOTED.items()):
        assert demoted[label] == arms_measured, (
            f"{label}: measured as '{arms_measured}', now '{demoted[label]}'"
        )
    dropped = sorted(set(CORPUS_UNTOUCHED_BY_C074) - set(untouched))
    assert not dropped, (
        f"C-074 preserved these release_ready legs and no longer does: {dropped}"
    )
    for label, size in sorted(CORPUS_UNTOUCHED_BY_C074.items()):
        assert untouched[label] == size, (
            f"{label}: connected core measured at {size}, now {untouched[label]}"
        )

    # 4. No demotion is unjustified -- the property that replaces set equality.
    for label, fired in sorted(demoted.items()):
        unjustified = sorted(set(fired.split(" + ")) - set(justified[label]))
        assert not unjustified, (
            f"{label}: demoted by {fired}, but an independent reading of the "
            f"committed artifact justifies only "
            f"{justified[label] or 'no arm at all'} -- {unjustified} is collateral"
        )
        assert control_of[label] == RELEASE_READY, (
            f"{label}: an arm of this card fired on a leg the tip already demoted "
            f"for another reason ({control_of[label]}); the demotion is not C-074's"
        )

    # 5. No preservation is indefensible.
    for label in sorted(set(untouched) - set(CORPUS_UNTOUCHED_BY_C074)):
        assert untouched[label] >= MIN_CONNECTED_CORE_REACTIONS, (
            f"{label} was preserved as release_ready with a connected core of "
            f"{untouched[label]}, below the floor: arm A failed to fire on a leg "
            f"it exists for"
        )
        assert pathway_named[label], (
            f"{label} was preserved as release_ready while naming no pathway. Over "
            f"the COMMITTED corpus every real run carries a Stage-0 context, so a "
            f"shipped leg with a blank pathway_name is F-100's shape reaching "
            f"release_ready by a route arm B did not see"
        )


def test_every_leg_c074_leaves_alone_is_a_genuinely_multi_step_pathway() -> None:
    """The collateral question, answered per leg (card section 7).

    A preservation is only defensible if the leg really is what the floor is
    protecting. Each of the six clears the floor on its own measured graph AND
    names the pathway it was judged against -- so neither arm has anything to say
    about it, which is why neither fires.

    Their demotion at this tip belongs to C-072's unmatched-anchor cap, asserted
    by name here so this card is never credited with it.

    **C-092 scope note.** This test stays scoped to the SIX legs C-074 measured,
    and the C-072 attribution above is a fact about those six rather than about
    preservation in general: a later run whose anchors all match is preserved and
    is NOT capped by C-072, so generalising the attribution here would be a false
    red. The obligation on preserved legs C-074 never saw lives in
    ``test_the_full_corpus_replay_demotes_nothing_it_cannot_justify`` section 5 --
    clear the floor, name a pathway -- which is the part that must not rot.
    """

    legs = dict(_committed_legs())
    if not legs:
        pytest.skip("no committed runs_verify legs in this checkout")
    for label, size in CORPUS_UNTOUCHED_BY_C074.items():
        leg = legs.get(label)
        if leg is None:
            pytest.skip(f"{label} is not in this checkout")
        report, payload = _load(leg)
        assert payload is not None
        assert _connected_core_size(report, payload) == size
        assert size >= MIN_CONNECTED_CORE_REACTIONS, f"{label} would fail arm A floor"
        context = (report.get("coverage") or {}).get("requested_context") or {}
        assert str(context.get("pathway_name") or "").strip(), f"{label} names no pathway"
        control = _replay(report, payload, arms=False)
        assert control.status == REVIEW_REQUIRED
        assert any(
            str(reason).split(":", 1)[0] == "requested_core_anchors_unmatched"
            for reason in control.reasons
        ), f"{label} is demoted at this tip by something other than C-072 cap"


#: The arm-B hit C-074 measured over the corpus. A PRESENCE pin, not an equality.
#:
#: C-074 wrote ``assert hits == ["2026-08-22_2147/PMC13231680/strict"]``. The same
#: declared negative control has since been re-run and committed, so a second and
#: equally correct hit landed and the equality went red on it. Correctness here was
#: never "there is exactly one"; it was "arm B fires on legs that declare a core
#: while naming no pathway, and on nothing else". That is what is asserted now.
CORPUS_ARM_B_HITS: Tuple[str, ...] = ("2026-08-22_2147/PMC13231680/strict",)


def test_every_committed_arm_b_hit_declares_a_core_and_names_no_pathway() -> None:
    """Arm B's blast radius over the corpus, measured rather than argued.

    Three claims:

    1. **The historical hit is still a hit.** If it is not in this checkout it is
       skipped, the way every other real-artifact test here skips.
    2. **Every hit is independently justified.** The production property is
       ``CoverageVerdict.declares_core_without_stating_a_pathway``; the expectation
       is ``_raw_declared_core_without_a_stated_pathway()``, which reads the
       committed mapping's own keys and additionally demands that terms actually
       stand behind the ``requested_core_declared`` flag. Asserting the two sets
       are EQUAL is an if-and-only-if over the corpus from two derivations, which
       an equality against a literal list was not.
    3. **The blast radius holds.** Arm B never fires on a leg that names a
       pathway. This is the collateral question and it is stated separately from
       (2) so it cannot be lost if (2) is ever loosened.
    """

    legs = _committed_legs()
    if not legs:
        pytest.skip("no committed runs_verify legs in this checkout")

    present = {label for label, _ in legs}
    hits: List[str] = []
    justified_hits: List[str] = []
    names_a_pathway: List[str] = []
    for label, leg in legs:
        report = json.loads((leg / "quarantine_report.json").read_text(encoding="utf-8"))
        if CoverageVerdict(report.get("coverage") or {}).declares_core_without_stating_a_pathway:
            hits.append(label)
        if _raw_declared_core_without_a_stated_pathway(report):
            justified_hits.append(label)
        context = (report.get("coverage") or {}).get("requested_context")
        if isinstance(context, dict) and str(context.get("pathway_name") or "").strip():
            names_a_pathway.append(label)

    for label in CORPUS_ARM_B_HITS:
        if label not in present:
            continue
        assert label in hits, (
            f"{label} was measured as an arm-B hit and no longer is: the leg that "
            f"registered F-100 stopped being detected"
        )

    assert sorted(hits) == sorted(justified_hits), (
        f"arm B and the raw committed JSON disagree about which legs declare a "
        f"core without naming a pathway. Fired on but unjustified: "
        f"{sorted(set(hits) - set(justified_hits))}; justified but not fired on: "
        f"{sorted(set(justified_hits) - set(hits))}"
    )
    collateral = sorted(set(hits) & set(names_a_pathway))
    assert not collateral, f"arm B fired on legs that DO name a pathway: {collateral}"


# ── C-092 NON-VACUITY: the re-based corpus tests still bite ─────────────────
#
# These are permanent, explicitly labelled non-vacuity tests, not throwaway
# probes. C-092 replaced three exact-set/exact-count pins with floors and per-leg
# justifications; a floor that cannot go red would be the defect C-092 was sent to
# fix, reproduced. Each test below perturbs the CENSUS -- never production -- and
# asserts the re-based test turns red on it.
#
# The perturbations are real committed legs copied to a temporary directory and
# mutated, with ``_committed_legs()`` monkeypatched to serve the perturbed corpus.
# Nothing is written into the repository and no production symbol is touched.

#: The two legs whose behaviour the perturbations below depend on.
_F101_LEG = "2026-08-22_2147/PMC12856317/strict"
_F100_LEG = "2026-08-22_2147/PMC13231680/strict"
_PRESERVED_LEG = "2026-08-21_2239/PMC12452463/strict"


def _synthetic_leg(tmp_path: Path, source: Path, name: str, mutate: Any) -> Path:
    """A real committed leg, copied out of the repository and mutated in place."""

    dest = tmp_path / name
    dest.mkdir(parents=True, exist_ok=True)
    report = json.loads((source / "quarantine_report.json").read_text(encoding="utf-8"))
    payload = json.loads((source / "final_mapped.json").read_text(encoding="utf-8"))
    mutate(report, payload)
    (dest / "quarantine_report.json").write_text(json.dumps(report), encoding="utf-8")
    (dest / "final_mapped.json").write_text(json.dumps(payload), encoding="utf-8")
    return dest


def _real_corpus() -> Dict[str, Path]:
    legs = dict(_committed_legs())
    if not legs:
        pytest.skip("no committed runs_verify legs in this checkout")
    return legs


def _serve_perturbed_corpus(
    monkeypatch: pytest.MonkeyPatch,
    *,
    drop: Tuple[str, ...] = (),
    replace: Optional[Dict[str, Path]] = None,
    add: Tuple[Tuple[str, Path], ...] = (),
    keep: Optional[int] = None,
) -> None:
    real = _committed_legs()
    swap = replace or {}
    perturbed = [
        (label, swap.get(label, path)) for label, path in real if label not in set(drop)
    ]
    perturbed.extend(add)
    if keep is not None:
        perturbed = perturbed[:keep]
    monkeypatch.setattr(
        sys.modules[__name__], "_committed_legs", lambda: sorted(perturbed)
    )


def test_nonvacuity_c092_a_shrinking_corpus_turns_the_census_floor_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**NON-VACUITY.** The floor replaced ``== 38``. It must still hear LOSS.

    Serve three legs where the corpus holds sixty and the census floor fires.
    This is the event the equality was there for; growth, which it also fired on,
    was never one.
    """

    _real_corpus()
    _serve_perturbed_corpus(monkeypatch, keep=3)
    with pytest.raises(AssertionError, match="SHRUNK below C-074's measured census"):
        test_the_full_corpus_replay_demotes_nothing_it_cannot_justify()


def test_nonvacuity_c092_losing_a_named_demotion_turns_the_corpus_test_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**NON-VACUITY.** A historically measured demotion vanishing from the
    committed corpus still goes red, which is what the old set equality bought."""

    legs = _real_corpus()
    if _F101_LEG not in legs:
        pytest.skip(f"{_F101_LEG} is not in this checkout")
    _serve_perturbed_corpus(monkeypatch, drop=(_F101_LEG,))
    with pytest.raises(AssertionError, match="demoted these legs and no longer does"):
        test_the_full_corpus_replay_demotes_nothing_it_cannot_justify()


def test_nonvacuity_c092_losing_a_named_preservation_turns_the_corpus_test_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**NON-VACUITY.** The preservation side of the same property: one of the six
    legs C-074 measured itself NOT touching disappearing is still a red."""

    legs = _real_corpus()
    if _PRESERVED_LEG not in legs:
        pytest.skip(f"{_PRESERVED_LEG} is not in this checkout")
    _serve_perturbed_corpus(monkeypatch, drop=(_PRESERVED_LEG,))
    with pytest.raises(AssertionError, match="preserved these release_ready legs"):
        test_the_full_corpus_replay_demotes_nothing_it_cannot_justify()


def test_nonvacuity_c092_a_named_leg_that_stops_firing_an_arm_turns_the_test_red(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """**NON-VACUITY, behaviour change rather than presence.** The F-100 leg is
    measured firing ``arm A + arm B``. Move its declared core to an
    ``explicit_argument`` source -- which STATES the request whatever the context
    says, so arm B correctly abstains -- and the leg is still demoted, still
    present, and still wrong against what was measured."""

    legs = _real_corpus()
    if _F100_LEG not in legs:
        pytest.skip(f"{_F100_LEG} is not in this checkout")

    def _explicit_source(report: Dict[str, Any], _payload: Dict[str, Any]) -> None:
        (report.setdefault("coverage", {}))["requested_core_source"] = "explicit_argument"

    swapped = _synthetic_leg(tmp_path, legs[_F100_LEG], "arm_b_silenced", _explicit_source)
    _serve_perturbed_corpus(monkeypatch, replace={_F100_LEG: swapped})
    with pytest.raises(AssertionError, match="measured as 'arm A \\+ arm B', now 'arm A'"):
        test_the_full_corpus_replay_demotes_nothing_it_cannot_justify()


def _flag_without_terms(report: Dict[str, Any], _payload: Dict[str, Any]) -> None:
    """Declare a requested core by FLAG with nothing standing behind it.

    ``CoverageVerdict`` reads the flag and calls this a declared core with no
    stated pathway, so arm B fires. ``_raw_declared_core_without_a_stated_pathway``
    reads the context's terms and does not, so the demotion is unjustified. This
    is the one place the two derivations genuinely disagree, which is what makes
    the justification a second opinion rather than a restatement.
    """

    coverage = report.setdefault("coverage", {})
    coverage["requested_core_declared"] = True
    coverage["requested_core_source"] = "pathway_context"
    context = coverage.setdefault("requested_context", {})
    context["pathway_name"] = ""
    for key in ("key_compounds", "key_proteins", "main_subprocesses", "subprocesses"):
        context[key] = []


def test_nonvacuity_c092_an_unjustified_arm_b_demotion_turns_the_corpus_red(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """**NON-VACUITY.** This is the perturbation the old set equality could not
    tell from a legitimate new run, and the whole reason the replacement is a
    JUSTIFICATION rather than a bigger list: a leg arm B demotes on a bare flag,
    with no requested terms behind it, is collateral and must go red."""

    legs = _real_corpus()
    if _F101_LEG not in legs:
        pytest.skip(f"{_F101_LEG} is not in this checkout")
    fabricated = _synthetic_leg(tmp_path, legs[_F101_LEG], "flag_only", _flag_without_terms)
    _serve_perturbed_corpus(
        monkeypatch, add=(("2099-01-01_0000/PMCSYNTH/strict", fabricated),)
    )
    with pytest.raises(AssertionError, match="is collateral"):
        test_the_full_corpus_replay_demotes_nothing_it_cannot_justify()


def test_nonvacuity_c092_an_unjustified_arm_b_hit_turns_the_blast_radius_red(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """**NON-VACUITY** for the other re-based test: the same flag-without-terms leg
    makes the two derivations disagree, and the if-and-only-if goes red."""

    legs = _real_corpus()
    if _F101_LEG not in legs:
        pytest.skip(f"{_F101_LEG} is not in this checkout")
    fabricated = _synthetic_leg(tmp_path, legs[_F101_LEG], "flag_only_b", _flag_without_terms)
    _serve_perturbed_corpus(
        monkeypatch, add=(("2099-01-01_0000/PMCSYNTH/strict", fabricated),)
    )
    with pytest.raises(AssertionError, match="Fired on but unjustified"):
        test_every_committed_arm_b_hit_declares_a_core_and_names_no_pathway()


def test_nonvacuity_c092_a_defective_silent_preservation_turns_the_corpus_red(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """**NON-VACUITY** for the preservation side. A leg that ships ``release_ready``
    naming NO pathway, by a route arm B does not see, must not be absorbed as
    "just another legitimate addition". The old equality would have caught it, for
    the wrong reason -- it caught every addition. This catches only this one."""

    legs = _real_corpus()
    if _PRESERVED_LEG not in legs:
        pytest.skip(f"{_PRESERVED_LEG} is not in this checkout")

    def _blank_the_pathway(report: Dict[str, Any], _payload: Dict[str, Any]) -> None:
        coverage = report.setdefault("coverage", {})
        # explicit_argument keeps arm B silent, so the leg really is PRESERVED.
        coverage["requested_core_source"] = "explicit_argument"
        coverage.setdefault("requested_context", {})["pathway_name"] = ""

    fabricated = _synthetic_leg(tmp_path, legs[_PRESERVED_LEG], "no_pathway", _blank_the_pathway)
    _serve_perturbed_corpus(
        monkeypatch, add=(("2099-01-01_0000/PMCQUIET/strict", fabricated),)
    )
    with pytest.raises(AssertionError, match="while naming no pathway"):
        test_the_full_corpus_replay_demotes_nothing_it_cannot_justify()


def test_no_committed_request_asks_for_a_single_step() -> None:
    """The floor's exemption fires on nothing in the corpus, so no leg above is
    preserved by it."""

    legs = _committed_legs()
    if not legs:
        pytest.skip("no committed runs_verify legs in this checkout")
    exempt = [
        label
        for label, leg in legs
        if requested_scope_is_a_single_reaction(
            (
                json.loads((leg / "quarantine_report.json").read_text(encoding="utf-8")).get(
                    "coverage"
                )
                or {}
            ).get("requested_context")
        )
    ]
    assert exempt == []


# ── 8. No benchmark identity entered production ─────────────────────────────


def _gold_identity() -> Tuple[List[str], List[str]]:
    """Paper ids and requested-pathway names, read from the pinned gold set so
    the guard cannot rot as the gold set grows."""

    gold_path = SRC / "t2pw" / "bench" / "gold" / "pinned_v1.json"
    if not gold_path.is_file():
        return [], []
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    cases = gold.get("cases") if isinstance(gold, dict) else gold
    papers: List[str] = []
    pathways: List[str] = []
    for case in cases if isinstance(cases, list) else []:
        if not isinstance(case, dict):
            continue
        for key in ("paper_id", "pmcid", "pmc_id", "id"):
            value = str(case.get(key) or "")
            if value.upper().startswith("PMC"):
                papers.append(value)
        name = str(case.get("requested_pathway") or "").strip()
        if name:
            pathways.append(name)
    return sorted(set(papers)), sorted(set(pathways))


def test_this_diff_puts_no_benchmark_paper_or_gold_pathway_name_into_src() -> None:
    """**THE PRODUCT OWNER'S EXPLICIT RULE.** No benchmark paper, PMC id, gold
    value or gold-specific pathway name may be hardcoded into ``src/``.

    Asserted over the DIFF rather than over the files, because both files this
    card touches already cite measured legs by id in comments that predate it.
    What must hold is that C-074 added none.

    Scoped to :data:`OWNED_SOURCE_FILES`, which is where every production line of
    this card lives; see that constant for why the scope is not a weakening.
    """

    try:
        diff = subprocess.run(
            ["git", "diff", BASE_SHA, "--", *OWNED_SOURCE_FILES],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git here
        pytest.skip("git is not available to read the diff")
    if diff.returncode != 0:
        pytest.skip(f"cannot diff against {BASE_SHA[:7]}: {diff.stderr.strip()[:200]}")

    added = [
        line[1:]
        for line in diff.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    # An ASSERT, not a skip: the base SHA is an ancestor of every branch this
    # runs on, so the diff is never empty, and a guard that silently skips is a
    # guard that has stopped guarding.
    assert added, f"no added lines in {OWNED_SOURCE_FILES} against {BASE_SHA[:7]}"

    papers, pathways = _gold_identity()
    offenders: List[str] = []
    for line in added:
        lowered = line.casefold()
        if re.search(r"PMC\d{4,}", line):
            offenders.append(f"PMC id: {line.strip()}")
        for paper in papers:
            if paper.casefold() in lowered:
                offenders.append(f"gold paper {paper}: {line.strip()}")
        for pathway in pathways:
            if pathway.casefold() in lowered:
                offenders.append(f"gold pathway {pathway!r}: {line.strip()}")
    assert not offenders, "benchmark identity entered src/:\n" + "\n".join(offenders)


def test_neither_arm_reads_the_payload_for_its_request_side_input() -> None:
    """The floor exemption is derived from the REQUEST, never from what survived.

    Asserted structurally, and against the PRODUCTION symbol: imported
    function-locally so the module still collects at the base SHA, which is what
    keeps the two G9 proofs above behavioural rather than symbol-absence.

    INSPECTING THE MODULE-LEVEL WRAPPER AT THE TOP OF THIS FILE INSTEAD WOULD BE
    VACUOUS. The wrapper takes ``pathway_context`` by construction, so the
    assertion would stay green if production later grew a ``payload=``
    parameter -- which is the exact regression this guard exists to catch. The
    module check below is what makes that impossible to reintroduce quietly.
    """

    import inspect

    from t2pw.pipeline.strict_quarantine import (
        requested_scope_is_a_single_reaction as production,
    )

    assert production.__module__ == "t2pw.pipeline.strict_quarantine", (
        "this guard must bind to production, never to the wrapper in this module"
    )
    signature = inspect.signature(production)
    assert list(signature.parameters) == ["pathway_context"], (
        "a payload-shaped parameter here would let the floor score itself"
    )
