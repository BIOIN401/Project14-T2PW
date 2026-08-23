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


def test_the_full_corpus_replay_moves_exactly_the_legs_that_are_named() -> None:
    """**CARD SECTION 7, as a test.** Both arms over every committed leg.

    Four things are asserted:

    1. **The corpus is the measured one.** 38 committed reports carry a release
       record; 9 of them recorded ``release_ready``. A checkout where those
       numbers moved is measuring something else.
    2. **Nothing is promoted and nothing is dropped.** For EVERY leg, not only
       the ``release_ready`` ones, the treatment is at or below the control and
       never reaches ``diagnostic_only`` unless the control already did. That is
       the cap property and merge rule 7 together.
    3. **The demotions are exactly the named set**, each with the arm that fired.
    4. **Every ``release_ready`` leg C-074 does not demote is a genuinely
       multi-step pathway** -- the collateral question, answered per leg rather
       than in aggregate, by the test after this one.
    """

    legs = _committed_legs()
    if not legs:
        pytest.skip("no committed runs_verify legs in this checkout")

    order = {RELEASE_READY: 2, REVIEW_REQUIRED: 1, DIAGNOSTIC_ONLY: 0}
    with_release = 0
    recorded_release_ready = 0
    control_release_ready = 0
    demoted: Dict[str, str] = {}
    untouched: Dict[str, int] = {}

    for label, leg in legs:
        report, payload = _load(leg)
        record = report.get("release") or {}
        recorded = str(record.get("status") or "")
        if recorded not in order or payload is None:
            continue
        with_release += 1
        control = _replay(report, payload, arms=False)
        applied = _replay(report, payload, arms=True)

        assert order[applied.status] <= order[control.status], (
            f"{label}: {control.status} -> {applied.status} is not a cap"
        )
        assert applied.status != DIAGNOSTIC_ONLY or control.status == DIAGNOSTIC_ONLY, (
            f"{label}: a cap reached diagnostic_only and the PWML would be dropped"
        )
        if control.status == RELEASE_READY:
            control_release_ready += 1
        if recorded != RELEASE_READY:
            continue
        recorded_release_ready += 1
        arms = _arms_that_fired(applied)
        if arms:
            demoted[label] = " + ".join(arms)
        else:
            untouched[label] = _connected_core_size(report, payload)

    assert with_release == 38, f"the corpus is not the measured 38 legs: {with_release}"
    assert recorded_release_ready == 9, (
        f"the corpus does not hold the measured 9 release_ready legs: {recorded_release_ready}"
    )
    assert demoted == CORPUS_DEMOTED, f"unaccounted demotion(s): {demoted}"
    assert untouched == CORPUS_UNTOUCHED_BY_C074, f"unaccounted preservation delta: {untouched}"
    # Every leg still release_ready at this tip is one C-074 demotes, and every
    # one of those is a single-connected-reaction payload. Stated as a measured
    # number rather than left implicit.
    assert control_release_ready == len(CORPUS_DEMOTED)


def test_every_leg_c074_leaves_alone_is_a_genuinely_multi_step_pathway() -> None:
    """The collateral question, answered per leg (card section 7).

    A preservation is only defensible if the leg really is what the floor is
    protecting. Each of the six clears the floor on its own measured graph AND
    names the pathway it was judged against -- so neither arm has anything to say
    about it, which is why neither fires.

    Their demotion at this tip belongs to C-072's unmatched-anchor cap, asserted
    by name here so this card is never credited with it.
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


def test_exactly_one_committed_leg_declares_a_core_without_stating_a_pathway() -> None:
    """Arm B's blast radius over the corpus, measured rather than argued."""

    legs = _committed_legs()
    if not legs:
        pytest.skip("no committed runs_verify legs in this checkout")
    hits = [
        label
        for label, leg in legs
        if CoverageVerdict(
            json.loads((leg / "quarantine_report.json").read_text(encoding="utf-8")).get("coverage")
            or {}
        ).declares_core_without_stating_a_pathway
    ]
    assert hits == ["2026-08-22_2147/PMC13231680/strict"]


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
    """

    try:
        diff = subprocess.run(
            ["git", "diff", BASE_SHA, "--", "src/"],
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
    if not added:
        pytest.skip("no added source lines to inspect")

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
    """The floor's exemption is derived from the REQUEST, never from what
    survived. Asserted structurally: the predicate's only parameter is the
    context, so there is no payload for it to read."""

    import inspect

    signature = inspect.signature(requested_scope_is_a_single_reaction)
    assert list(signature.parameters) == ["pathway_context"], (
        "a payload-shaped parameter here would let the floor score itself"
    )
