"""C-125. A NARROWER Stage-0 reading of the requested pathway is not a conflict.

WHAT THIS FILE IS ABOUT
-----------------------
Measured in the C-122 final smoke (``runs_smoke/2026-09-10_1135``), two of twelve
papers died at Stage 0 with ZERO reactions because the scope guard refused a
Stage-0 scope that is a strict SPECIALIZATION of what the batch asked for::

    PMC13474940  requested 'fumonisin biosynthesis'
                 Stage 0 read 'fumonisin B1 biosynthesis'
    PMC13123502  requested 'steroidal saponin biosynthesis'
                 Stage 0 read 'steroidal saponin (polyphyllin) biosynthesis in
                               Paris polyphylla, focusing on UGT-mediated
                               3-O-glucosylation'

Every requested/observed pair used here as a base-failure proof or as an
ORCH-734 guard is the string the run actually recorded, copied from that run's
``papers/<id>/strict/RESULT.txt`` (and ``runs_smoke/2026-09-08_1528`` for
PMC7910490), not a paraphrase.

HOW THE LABELS ARE USED (G9)
----------------------------
* **G9 BASE-FAILURE PROOF** -- fails ON VALUES at ``4d417e4f`` and passes at the
  tip. These drive the production seam ``driver._reconcile_stage0_scope``, which
  exists at the base SHA, and assert the RUN PROCEEDS. At the base the same call
  returns "stop" and stamps ``status='scope_conflict'``. No test in this file
  proves anything from the absence of a symbol.
* **NEW-CAPABILITY ACCEPTANCE** -- the rule module ``t2pw.batch.scope_compat`` is
  new in C-125, so its own unit table is labelled as new acceptance, not as a
  fabricated base failure.
* **REGRESSION GUARD** -- passes at the base SHA and must keep passing: the
  conflicts that MUST still stop a run, and the artifact/message/reason shape
  that must not move.

ONE HONEST LIMIT, STATED UP FRONT
---------------------------------
``apply_stage0_observation`` only raises a pathway conflict when its own lexicon
comparison separates the two names. Measured: it raises NO conflict at all for
``'heme biosynthesis'`` vs ``'heme degradation'``, vs ``'riboflavin
biosynthesis'``, or vs ``'heme biosynthesis inhibitor screening'`` -- those pairs
never reach the new rule in production. That is pre-existing behaviour this card
does not touch. So each of those required REJECT cases is proven at the RULE
level (where it decides what the rule would do if consulted), and the seam-level
REJECT guards use pairs the comparator does flag: the real ORCH-734 pair, a
process-kind flip on the fumonisin subject, and the reversed direction.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from t2pw.batch import driver

# THE G9 PROOFS BELOW IMPORT NONE OF THE NEW SYMBOLS AT MODULE LEVEL, ON PURPOSE.
# A base-tree run must fail on VALUES -- here, on ``stop is False`` -- and a
# module-level import of a module the base tree does not have would fail
# COLLECTION instead, which is symbol absence dressed up as a behavioural proof
# and is explicitly not evidence (G9). So the new rule module is imported
# optionally, and only the arms that exercise the rule API directly -- every one
# of them labelled NEW-CAPABILITY ACCEPTANCE -- depend on it. Same shape as
# ``tests/test_c121_autostate_lifecycle.py``.
try:
    from t2pw.batch.scope_compat import (  # noqa: E402
        REASON_EMPTY_OBSERVED,
        REASON_EMPTY_REQUEST,
        REASON_KIND_MISMATCH,
        REASON_NO_REQUEST_KIND,
        REASON_NO_REQUEST_SUBJECT,
        REASON_SPECIALIZES,
        REASON_SUBJECT_DROPPED,
        KIND_INHIBITION,
        compare_scope,
        kind_of,
        partition_specialization_conflicts,
    )

    _RULE_API = True
except ImportError:  # pragma: no cover -- taken on the base tree only
    REASON_EMPTY_OBSERVED = "scope_compat absent"
    REASON_EMPTY_REQUEST = "scope_compat absent"
    REASON_KIND_MISMATCH = "scope_compat absent"
    REASON_NO_REQUEST_KIND = "scope_compat absent"
    REASON_NO_REQUEST_SUBJECT = "scope_compat absent"
    REASON_SPECIALIZES = "scope_compat absent"
    REASON_SUBJECT_DROPPED = "scope_compat absent"
    KIND_INHIBITION = "scope_compat absent"
    compare_scope = None  # type: ignore[assignment]
    kind_of = None  # type: ignore[assignment]
    partition_specialization_conflicts = None  # type: ignore[assignment]
    _RULE_API = False

#: New capability, so no base failure is claimed for these arms (G9): on a tree
#: without the rule module they SKIP rather than erroring in collection.
new_capability = pytest.mark.skipif(
    not _RULE_API,
    reason="new acceptance arm for the C-125 specialization rule; no base failure claimed",
)

# ── The recorded strings, verbatim ─────────────────────────────────────────
#: runs_smoke/2026-09-10_1135/papers/PMC13474940/strict/RESULT.txt
FUMONISIN_REQUESTED = "fumonisin biosynthesis"
FUMONISIN_STAGE0 = "fumonisin B1 biosynthesis"
#: runs_smoke/2026-09-10_1135/papers/PMC13123502/strict/RESULT.txt
SAPONIN_REQUESTED = "steroidal saponin biosynthesis"
SAPONIN_STAGE0 = (
    "steroidal saponin (polyphyllin) biosynthesis in Paris polyphylla, "
    "focusing on UGT-mediated 3-O-glucosylation"
)
#: runs_smoke/2026-09-08_1528/papers/PMC7910490/strict/RESULT.txt (ORCH-734).
#: A genuinely different pathway: Stage 0 was RIGHT to refuse it.
GLYCOALKALOID_REQUESTED = "steroidal glycoalkaloid biosynthesis"
GLYCOALKALOID_STAGE0 = "potato solanidane glycoalkaloid biosynthesis"

#: The count key and the artifact name the tip stamps when a conflict is
#: withdrawn. Resolved through ``getattr`` so that a BASE-TREE run of this file
#: fails on VALUES rather than on a missing attribute (G9): at ``4d417e4f``
#: neither name exists on the driver and neither is ever written, so every
#: REGRESSION GUARD that asserts their ABSENCE must still be able to run there.
COUNT_SPECIALIZATIONS = getattr(
    driver, "COUNT_SCOPE_SPECIALIZATIONS", "stage0_scope_specializations"
)
SPECIALIZATION_ARTIFACT = getattr(
    driver, "SCOPE_SPECIALIZATION_NAME", "scope_specialization.json"
)


# ── Seam harness ───────────────────────────────────────────────────────────
def _reconcile(
    requested_pathway: str,
    stage0_pathway: str,
    *,
    requested_organism: str = "",
    stage0_organism: str = "",
) -> Tuple[bool, driver.RunOutcome]:
    """Drive the production seam and return ``(should_stop, outcome)``.

    ``_reconcile_stage0_scope`` reads the plan record through ``_paper_field``
    (dict-aware) and the app through ``_ss(at, "pathway_context")``, which is a
    plain ``at.session_state[key]`` lookup. Both halves are supplied exactly as
    the driver reads them, so this exercises the real function -- not a copy of
    its logic -- with no Streamlit app, no network and no LLM.
    """
    paper: Dict[str, Any] = {
        "paper_id": "PMC-TEST",
        "requested_pathway": requested_pathway,
        "requested_organism": requested_organism,
    }
    context: Dict[str, Any] = {"pathway_name": stage0_pathway}
    if stage0_organism:
        context["likely_organism"] = stage0_organism
    at = SimpleNamespace(session_state={"pathway_context": context})
    outcome = driver.RunOutcome(paper_id="PMC-TEST")
    stop = driver._reconcile_stage0_scope(at, paper, outcome)
    return stop, outcome


# ===========================================================================
# G9 BASE-FAILURE PROOFS -- the two papers that died for nothing.
#
# At 4d417e4f each of these asserts fails on a VALUE: ``stop`` is True, the
# status is 'scope_conflict', and scope_conflict.json is in the artifacts.
# ===========================================================================
def test_g9_base_failure_a_narrower_fumonisin_reading_lets_the_run_proceed() -> None:
    """G9 BASE-FAILURE PROOF -- PMC13474940, the real recorded pair."""
    stop, outcome = _reconcile(FUMONISIN_REQUESTED, FUMONISIN_STAGE0)

    assert stop is False, "the run must PROCEED: same pathway, said more precisely"
    assert outcome.status != driver._STATUS_SCOPE_CONFLICT
    assert outcome.scope_conflicts == []
    assert driver.SCOPE_CONFLICT_NAME not in outcome.artifacts
    assert "scope_conflicts" not in outcome.counts
    assert outcome.release_status is None, "nothing is classified: nothing stopped"

    # Stage 0's reading is still recorded as an OBSERVATION, unchanged.
    assert outcome.observed_context["observed_pathways"] == [FUMONISIN_STAGE0]
    # And the overrule is auditable rather than silent.
    assert outcome.counts[COUNT_SPECIALIZATIONS] == 1
    note = json.loads(outcome.artifacts[SPECIALIZATION_ARTIFACT])
    assert note["requested"]["requested_pathway"] == FUMONISIN_REQUESTED
    assert note["observed"]["observed_pathways"] == [FUMONISIN_STAGE0]
    assert len(note["withdrawn_conflicts"]) == 1
    assert FUMONISIN_STAGE0 in note["withdrawn_conflicts"][0]

    # THE CHANNEL MATTERS, not just the fact. A warning on a passing run sets
    # ``report.ModeRun.warned``, which renders the verdict "PASS (no research
    # deliverable)" and files the run under "!! PASSED BUT PRODUCED NO
    # DELIVERABLE !! ... This is NOT a clean night." -- reporting every run this
    # card RESCUES as having produced nothing. Round-1 review caught exactly that.
    assert outcome.warnings == [], "a rescued run must not be filed as unclean"


def test_g9_base_failure_a_narrower_saponin_reading_lets_the_run_proceed() -> None:
    """G9 BASE-FAILURE PROOF -- PMC13123502, organism and focus clause and all."""
    stop, outcome = _reconcile(SAPONIN_REQUESTED, SAPONIN_STAGE0)

    assert stop is False
    assert outcome.status != driver._STATUS_SCOPE_CONFLICT
    assert outcome.scope_conflicts == []
    assert driver.SCOPE_CONFLICT_NAME not in outcome.artifacts
    assert outcome.counts[COUNT_SPECIALIZATIONS] == 1
    assert SPECIALIZATION_ARTIFACT in outcome.artifacts
    assert outcome.warnings == []
    assert outcome.observed_context["observed_pathways"] == [SAPONIN_STAGE0]


# ===========================================================================
# REGRESSION GUARDS at the seam -- what must still stop a run.
# ===========================================================================
def test_regression_guard_orch734_glycoalkaloid_paper_still_stops() -> None:
    """REGRESSION GUARD -- ORCH-734's PMC7910490. A different pathway, refused.

    The request says *steroidal*; Stage 0 says *potato solanidane*. The request's
    own subject term is gone, so this is a different claim, not a narrower one.
    """
    stop, outcome = _reconcile(GLYCOALKALOID_REQUESTED, GLYCOALKALOID_STAGE0)

    assert stop is True
    assert outcome.status == driver._STATUS_SCOPE_CONFLICT
    assert outcome.counts["scope_conflicts"] == 1
    assert COUNT_SPECIALIZATIONS not in outcome.counts
    assert outcome.warnings == []


def test_regression_guard_a_stopped_run_keeps_its_artifact_message_and_reason() -> None:
    """REGRESSION GUARD -- the operator-facing shape of a genuine conflict.

    Same pair as above; this test is about the RECORD rather than the decision.
    """
    _, outcome = _reconcile(
        GLYCOALKALOID_REQUESTED, GLYCOALKALOID_STAGE0, requested_organism="Solanum tuberosum"
    )

    assert outcome.failure_kind == ""
    assert driver.OUTCOME_SCOPE_CONFLICT in outcome.issue_codes
    assert outcome.message.startswith(
        "Stage 0 read a scope that contradicts the batch request, so this paper "
        "is not the paper the topic line asked for: "
    )
    assert outcome.detail == "; ".join(outcome.scope_conflicts)

    artifact = json.loads(outcome.artifacts[driver.SCOPE_CONFLICT_NAME])
    assert artifact["requested"]["requested_pathway"] == GLYCOALKALOID_REQUESTED
    assert artifact["requested"]["requested_organism"] == "Solanum tuberosum"
    assert artifact["observed"]["observed_pathways"] == [GLYCOALKALOID_STAGE0]
    assert artifact["conflicts"] == outcome.scope_conflicts
    assert artifact["stage0_context"]["pathway_name"] == GLYCOALKALOID_STAGE0

    # The reason code still travels on the release classification.
    reasons = list(getattr(outcome.release_status, "reasons", ()) or ())
    assert driver.REASON_STAGE0_SCOPE_CONFLICT in reasons
    assert outcome.requested_scope["requested_pathway"] == GLYCOALKALOID_REQUESTED


def test_regression_guard_a_process_kind_flip_still_stops_the_run() -> None:
    """REGRESSION GUARD -- biosynthesis requested, degradation read: still a conflict.

    The subject tokens match perfectly (``fumonisin``), so this is exactly the
    case a subject-only rule would wrongly admit.
    """
    stop, outcome = _reconcile(FUMONISIN_REQUESTED, "fumonisin B1 degradation")

    assert stop is True
    assert outcome.status == driver._STATUS_SCOPE_CONFLICT
    assert COUNT_SPECIALIZATIONS not in outcome.counts


def test_regression_guard_the_reverse_process_kind_flip_also_stops_the_run() -> None:
    """REGRESSION GUARD -- degradation requested, biosynthesis read. Both directions."""
    stop, outcome = _reconcile("fumonisin degradation", "fumonisin B1 biosynthesis")

    assert stop is True
    assert outcome.status == driver._STATUS_SCOPE_CONFLICT


def test_regression_guard_a_more_specific_request_is_still_a_conflict() -> None:
    """REGRESSION GUARD -- direction is one-way.

    Asking for ``fumonisin B1`` and getting a paper about fumonisins generally is
    a DIFFERENT, unproven claim: the paper may never establish B1. Only the
    request may be the more general of the two.
    """
    stop, outcome = _reconcile("fumonisin B1 biosynthesis", "fumonisin biosynthesis")

    assert stop is True
    assert outcome.status == driver._STATUS_SCOPE_CONFLICT
    assert COUNT_SPECIALIZATIONS not in outcome.counts


def test_regression_guard_an_unrelated_pathway_and_organism_still_stop_the_run() -> None:
    """REGRESSION GUARD -- the pre-C-125 canonical case, unchanged.

    ``lipid A biosynthesis`` in E. coli vs ``cholesterol biosynthesis`` in
    H. sapiens: two conflicts, both surviving.
    """
    stop, outcome = _reconcile(
        "lipid A biosynthesis",
        "cholesterol biosynthesis",
        requested_organism="Escherichia coli",
        stage0_organism="Homo sapiens",
    )

    assert stop is True
    assert outcome.counts["scope_conflicts"] == 2
    assert any("cholesterol biosynthesis" in c for c in outcome.scope_conflicts)
    assert any("Homo sapiens" in c for c in outcome.scope_conflicts)


def test_g9_base_failure_an_organism_conflict_survives_a_withdrawn_pathway() -> None:
    """G9 BASE-FAILURE PROOF -- this card relaxes the PATHWAY comparison, nothing else.

    At 4d417e4f this run records TWO conflicts; at the tip it records one, and
    that one is the organism's.

    The pathway conflict is withdrawn (a real specialization), the organism
    conflict is not, and the organism conflict alone still stops the run.
    """
    stop, outcome = _reconcile(
        FUMONISIN_REQUESTED,
        FUMONISIN_STAGE0,
        requested_organism="Fusarium verticillioides",
        stage0_organism="Homo sapiens",
    )

    assert stop is True
    assert outcome.counts["scope_conflicts"] == 1
    assert "Homo sapiens" in outcome.scope_conflicts[0]
    assert not any("pathway" in c for c in outcome.scope_conflicts)
    # The withdrawal is still recorded, even though the run stopped for the organism.
    assert outcome.counts[COUNT_SPECIALIZATIONS] == 1


def test_regression_guard_a_pinned_paper_is_untouched_by_the_new_rule() -> None:
    """REGRESSION GUARD -- no request, so nothing to specialize and nothing to refuse."""
    at = SimpleNamespace(
        session_state={"pathway_context": {"pathway_name": "cholesterol biosynthesis"}}
    )
    outcome = driver.RunOutcome()
    stop = driver._reconcile_stage0_scope(at, {"paper_id": "PMC-PINNED"}, outcome)

    assert stop is False
    assert outcome.scope_conflicts == []
    assert outcome.observed_context == {}
    assert outcome.warnings == []


def test_g9_base_failure_a_rescued_run_is_not_filed_as_unclean() -> None:
    """G9 BASE-FAILURE PROOF -- the OPERATOR-facing half of the rescue.

    Round-1 review of C-125 found the first implementation announcing the rescue
    through ``outcome.warnings``, which sets ``report.ModeRun.warned``
    (``report.py:448``) and prints the run as ``PASS (no research deliverable)``
    under "!! PASSED BUT PRODUCED NO DELIVERABLE !!". A rescued paper would have
    been reported as producing nothing. This asserts the report's own verdict, not
    the driver's field, because that is where the defect was visible.

    ``status`` is forced to ``pass`` because the driver records the leg status
    further down; what is under test is what the REPORT makes of a row that
    carries this rescue. At 4d417e4f the same forced row fails on a VALUE: no
    ``scope_specialization.json`` is ever written, so the artifact assertion below
    fails on a list that does not contain it.
    """
    from t2pw.batch import report as batch_report

    _, outcome = _reconcile(FUMONISIN_REQUESTED, FUMONISIN_STAGE0)
    outcome.status = "pass"  # what the driver goes on to record for this leg
    run = batch_report._to_run(outcome.to_dict())

    assert run.passed is True
    assert run.warned is False
    assert run.verdict == "PASS"
    # ...and the rescue is still legible in the row: the artifact is listed and
    # the count is there for an aggregator.
    assert SPECIALIZATION_ARTIFACT in [f["name"] for f in outcome.to_dict()["files"]]
    assert outcome.to_dict()["counts"][COUNT_SPECIALIZATIONS] == 1


def test_the_skip_switch_cannot_hide_a_broken_tip() -> None:
    """META-GUARD -- deliberately NOT decorated with ``@new_capability``.

    ``_RULE_API`` exists so a BASE tree collects this file instead of erroring in
    collection (G9). Round-1 review named the failure mode that buys: a partially
    broken ``scope_compat`` at the TIP would silently downgrade every arm below to
    a skip and leave the suite green. So the switch is pinned to the seam -- if
    the driver carries the C-125 seam, the rule API MUST have imported -- and this
    arm skips only on a tree that has neither half, which is the base tree.
    """
    if not hasattr(driver, "COUNT_SCOPE_SPECIALIZATIONS"):
        pytest.skip("base tree: neither half of C-125 is present")
    assert _RULE_API is True, (
        "the driver carries the C-125 seam but t2pw.batch.scope_compat did not "
        "import: every rule arm in this file is silently skipping"
    )


# ===========================================================================
# NEW-CAPABILITY ACCEPTANCE -- the rule itself, as one table.
#
# ``t2pw.batch.scope_compat`` is new in C-125, so this table is labelled new
# acceptance rather than dressed up as a base failure. It is the whole rule in
# one place: read it top to bottom to see what is admitted and what is not.
# ===========================================================================
#: ``(requested, stage0_observed, expected_compatible, expected_reason, why)``
RULE_TABLE: List[Tuple[str, str, bool, str, str]] = [
    # -- the two production defects this card exists for ---------------------
    (
        FUMONISIN_REQUESTED,
        FUMONISIN_STAGE0,
        True,
        REASON_SPECIALIZES,
        "PMC13474940: 'B1' is added detail, the subject and process are the same",
    ),
    (
        SAPONIN_REQUESTED,
        SAPONIN_STAGE0,
        True,
        REASON_SPECIALIZES,
        "PMC13123502: congener, organism and focus clause are all ADDITIONS",
    ),
    (
        FUMONISIN_REQUESTED,
        "fumonisin B1 biosynthetic pathway",
        True,
        REASON_SPECIALIZES,
        "'biosynthetic pathway' is the same process kind said another way",
    ),
    (
        "fumonisins biosynthesis",
        FUMONISIN_STAGE0,
        True,
        REASON_SPECIALIZES,
        "a plural request is the same request",
    ),
    (
        "lipid A biosynthesis",
        "lipid A biosynthesis in Escherichia coli",
        True,
        REASON_SPECIALIZES,
        "'A' is NOT a stopword: lipid A survives as a subject token on both sides",
    ),
    # -- ORCH-734: the case that must keep failing ---------------------------
    (
        GLYCOALKALOID_REQUESTED,
        GLYCOALKALOID_STAGE0,
        False,
        REASON_SUBJECT_DROPPED,
        "ORCH-734: the request's 'steroidal' is gone, so it is a different claim",
    ),
    # -- process-kind flips, both directions ---------------------------------
    (
        "heme biosynthesis",
        "heme degradation",
        False,
        REASON_KIND_MISMATCH,
        "identical subject, opposite process: never satisfied",
    ),
    (
        "heme degradation",
        "heme biosynthesis",
        False,
        REASON_KIND_MISMATCH,
        "and the same in reverse",
    ),
    (
        "heme biosynthesis",
        "heme catabolism",
        False,
        REASON_KIND_MISMATCH,
        "'catabolism' is the degradation family",
    ),
    (
        "heme biosynthesis",
        "heme breakdown",
        False,
        REASON_KIND_MISMATCH,
        "so is 'breakdown'",
    ),
    (
        "heme biosynthesis",
        "heme transport",
        False,
        REASON_KIND_MISMATCH,
        "moving the molecule is not making it",
    ),
    (
        "heme biosynthesis",
        "heme biosynthesis inhibition",
        False,
        REASON_KIND_MISMATCH,
        "acting ON the process is not running it",
    ),
    # -- the forbidden substring rule ---------------------------------------
    (
        "heme biosynthesis",
        "heme biosynthesis inhibitor screening",
        False,
        REASON_KIND_MISMATCH,
        "THE substring trap: contained as text, refused on process-kind SET equality",
    ),
    # -- ROUND-1 REVIEW: the twelve seam flips rule 1 used to fail open on ------
    # Every one of these is a phrase Stage 0 plausibly writes, and every one was
    # ADMITTED by the first C-125 implementation because its extra words were
    # outside the closed process lexicon and were therefore filed as harmless
    # subject additions. They are the reason the inhibition/attenuation family,
    # the study-design family and the negation prefixes exist.
    (
        FUMONISIN_REQUESTED,
        "suppression of fumonisin B1 biosynthesis",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 1: a common mycotoxin paper title shape",
    ),
    (
        FUMONISIN_REQUESTED,
        "fumonisin B1 biosynthesis repression",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 2",
    ),
    (
        FUMONISIN_REQUESTED,
        "fumonisin B1 biosynthesis gene cluster silencing",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 3",
    ),
    (
        FUMONISIN_REQUESTED,
        "fumonisin B1 biosynthesis knockdown",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 4",
    ),
    (
        FUMONISIN_REQUESTED,
        "fumonisin B1 biosynthesis blockade",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 5",
    ),
    (
        FUMONISIN_REQUESTED,
        "fumonisin B1 biosynthesis deficiency",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 6",
    ),
    (
        FUMONISIN_REQUESTED,
        "fumonisin B1 biosynthesis drug screening",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 7: one word from the inhibitor-screening case, same answer now",
    ),
    (
        FUMONISIN_REQUESTED,
        "antifungal agents that abolish fumonisin B1 biosynthesis",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 8: 'antifungal' is negation-bearing, 'abolish' is inhibition",
    ),
    (
        FUMONISIN_REQUESTED,
        "non-fumonisin B1 biosynthesis",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 9: the negation prefix is a KIND, never a subject token",
    ),
    (
        FUMONISIN_REQUESTED,
        "loss of fumonisin B1 biosynthesis",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 10",
    ),
    (
        FUMONISIN_REQUESTED,
        "review of fumonisin B1 biosynthesis",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 11: a review OF the pathway is a different document scope",
    ),
    (
        "siderophore uptake",
        "siderophore efflux",
        False,
        REASON_KIND_MISMATCH,
        "round-1 flip 12: a direction flip INSIDE the transport family",
    ),
    (
        "siderophore transport",
        "siderophore efflux",
        False,
        REASON_KIND_MISMATCH,
        "deliberate over-refusal: undirected transport does not match a direction",
    ),
    (
        "siderophore uptake",
        "siderophore B uptake in Pseudomonas aeruginosa",
        True,
        REASON_SPECIALIZES,
        "the transport split still admits a genuine narrowing within one direction",
    ),
    # -- a different subject entirely ---------------------------------------
    (
        "heme biosynthesis",
        "riboflavin biosynthesis",
        False,
        REASON_SUBJECT_DROPPED,
        "same process, different molecule",
    ),
    (
        "lipid A biosynthesis",
        "cholesterol biosynthesis",
        False,
        REASON_SUBJECT_DROPPED,
        "the pre-C-125 canonical conflict, unchanged",
    ),
    # -- direction is one-way ------------------------------------------------
    (
        FUMONISIN_STAGE0,
        FUMONISIN_REQUESTED,
        False,
        REASON_SUBJECT_DROPPED,
        "a request MORE SPECIFIC than Stage 0 is a different, unproven claim",
    ),
    (
        "lipid A biosynthesis",
        "lipid biosynthesis",
        False,
        REASON_SUBJECT_DROPPED,
        "dropping the 'A' generalizes the wrong way",
    ),
    # -- degenerate requests: never compatible-with-everything ---------------
    ("", FUMONISIN_STAGE0, False, REASON_EMPTY_REQUEST, "a blank request matches nothing"),
    ("   ", FUMONISIN_STAGE0, False, REASON_EMPTY_REQUEST, "and neither does whitespace"),
    (
        "(),-",
        FUMONISIN_STAGE0,
        False,
        REASON_EMPTY_REQUEST,
        "punctuation alone yields no tokens",
    ),
    (FUMONISIN_REQUESTED, "", False, REASON_EMPTY_OBSERVED, "nor an empty Stage-0 scope"),
    (
        "fumonisin",
        FUMONISIN_STAGE0,
        False,
        REASON_NO_REQUEST_KIND,
        "a request that names no process cannot pin the process down",
    ),
    (
        "biosynthesis",
        FUMONISIN_STAGE0,
        False,
        REASON_NO_REQUEST_SUBJECT,
        "and a request that is only a process names no subject",
    ),
]


@pytest.mark.parametrize(
    "requested,observed,expected,reason,why",
    RULE_TABLE,
    ids=[f"{i:02d}" for i in range(len(RULE_TABLE))],
)
@new_capability
def test_new_capability_the_specialization_rule_table(
    requested: str, observed: str, expected: bool, reason: str, why: str
) -> None:
    """NEW-CAPABILITY ACCEPTANCE -- the whole rule, case by case."""
    verdict = compare_scope(requested, observed)
    assert verdict.compatible is expected, f"{why}: {verdict.to_dict()}"
    assert verdict.reason == reason, f"{why}: {verdict.to_dict()}"


@new_capability
def test_new_capability_the_rule_is_not_a_substring_test() -> None:
    """NEW-CAPABILITY ACCEPTANCE -- stated as the explicit prohibition it is.

    The substring relation HOLDS for this pair. The rule refuses it anyway; a
    naive ``in`` would have admitted a biosynthesis extraction from an
    inhibitor-screening paper.
    """
    requested, observed = "heme biosynthesis", "heme biosynthesis inhibitor screening"
    assert requested in observed, "the trap only means something if it is a substring"
    verdict = compare_scope(requested, observed)
    assert verdict.compatible is False
    assert verdict.reason == REASON_KIND_MISMATCH
    assert sorted(verdict.observed_kinds) == [
        "biosynthesis",
        "inhibition",
        "study_design",
    ]


@new_capability
def test_new_capability_the_rule_is_deterministic_and_case_insensitive() -> None:
    """NEW-CAPABILITY ACCEPTANCE -- same answer every time, no normalization drift."""
    first = compare_scope(FUMONISIN_REQUESTED, FUMONISIN_STAGE0)
    again = compare_scope(FUMONISIN_REQUESTED.upper(), FUMONISIN_STAGE0.lower())
    assert first.compatible is True and again.compatible is True
    assert first.reason == again.reason
    assert compare_scope(FUMONISIN_REQUESTED, FUMONISIN_STAGE0).to_dict() == first.to_dict()


# ===========================================================================
# The partition: what it may and may not withdraw.
# ===========================================================================
@new_capability
def test_new_capability_the_partition_withdraws_only_the_pathway_conflict() -> None:
    """NEW-CAPABILITY ACCEPTANCE -- organism conflicts are not this rule's business."""
    conflicts = [
        f"Stage 0 read pathway '{FUMONISIN_STAGE0}' which does not match the "
        f"requested '{FUMONISIN_REQUESTED}'",
        "Stage 0 read organism 'Homo sapiens' but the batch requested "
        "'Fusarium verticillioides'",
    ]
    remaining, withdrawn = partition_specialization_conflicts(
        conflicts,
        requested_pathway=FUMONISIN_REQUESTED,
        observed_pathways=[FUMONISIN_STAGE0],
    )

    assert remaining == [conflicts[1]]
    assert len(withdrawn) == 1 and FUMONISIN_STAGE0 in withdrawn[0]


@new_capability
def test_new_capability_an_unrecognised_conflict_wording_fails_closed() -> None:
    """NEW-CAPABILITY ACCEPTANCE -- fail-closed arm of the new partition.

    A reworded conflict message keeps the conflict.

    The partition matches on the exact prefix ``apply_stage0_observation`` emits.
    If that wording ever changes, this rule must go back to withdrawing nothing
    rather than start withdrawing by accident.
    """
    conflicts = [
        f"the Stage-0 pathway was '{FUMONISIN_STAGE0}', not '{FUMONISIN_REQUESTED}'"
    ]
    remaining, withdrawn = partition_specialization_conflicts(
        conflicts,
        requested_pathway=FUMONISIN_REQUESTED,
        observed_pathways=[FUMONISIN_STAGE0],
    )

    assert remaining == conflicts
    assert withdrawn == []


@new_capability
def test_new_capability_one_bad_pathway_among_several_still_conflicts() -> None:
    """NEW-CAPABILITY ACCEPTANCE -- withdrawal is per conflict, never wholesale."""
    good = (
        f"Stage 0 read pathway '{FUMONISIN_STAGE0}' which does not match the "
        f"requested '{FUMONISIN_REQUESTED}'"
    )
    bad = (
        "Stage 0 read pathway 'cholesterol biosynthesis' which does not match the "
        f"requested '{FUMONISIN_REQUESTED}'"
    )
    remaining, withdrawn = partition_specialization_conflicts(
        [good, bad],
        requested_pathway=FUMONISIN_REQUESTED,
        observed_pathways=[FUMONISIN_STAGE0, "cholesterol biosynthesis"],
    )

    assert remaining == [bad]
    assert len(withdrawn) == 1


# ===========================================================================
# THE LIMIT, stated as a test rather than as a comment.
# ===========================================================================
#: Phrases that this rule ADMITS and that a stricter reader might not want
#: admitted. They are here because the guard is a CLOSED token list against an
#: OPEN world: a framing word the lexicon does not know is filed as a subject
#: token, subject additions are exactly what rule 2 permits, and the pair passes.
#: For such a phrase the module degenerates to the substring acceptance the card
#: forbids -- which is why this is asserted rather than described.
RESIDUAL_OPEN_WORLD_CASES: List[Tuple[str, str]] = [
    (FUMONISIN_REQUESTED, "mathematical modelling of fumonisin B1 biosynthesis"),
    (FUMONISIN_REQUESTED, "evolutionary origin of fumonisin B1 biosynthesis"),
    (FUMONISIN_REQUESTED, "in vitro reconstitution of fumonisin B1 biosynthesis"),
]


@new_capability
@pytest.mark.parametrize("requested,observed", RESIDUAL_OPEN_WORLD_CASES)
def test_new_capability_the_residual_open_world_exposure_is_real(
    requested: str, observed: str
) -> None:
    """NEW-CAPABILITY ACCEPTANCE -- the honest limit of the rule. READ THIS.

    These pass. The module claims NO completeness: extending
    ``PROCESS_KINDS`` closes the phrases we have seen, and cannot close the ones
    we have not. Round-1 review closed twelve; this arm exists so the thirteenth
    is met as a known property with a known fix -- add the word to the lexicon and
    add its row to ``RULE_TABLE`` -- rather than as a surprise.

    Note also what the rule is NOT judging: whether the paper is a mechanism
    study. That is the eligibility gate's question, not this one. These arms are
    about scope only, and admitting them is not by itself a product defect; a
    lexicon that refused every unfamiliar framing verb would also refuse the two
    papers this card exists to rescue.
    """
    assert compare_scope(requested, observed).compatible is True, (
        "if this now FAILS the lexicon grew, which is fine -- move the row into "
        "RULE_TABLE as a reject and record what closed it"
    )


@new_capability
def test_new_capability_extending_the_lexicon_can_only_refuse_more() -> None:
    """NEW-CAPABILITY ACCEPTANCE -- why a lexicon patch is a safe fix.

    A new PROCESS_KINDS entry can only ADD a family to one side's kind set, and
    rule 1 is set EQUALITY, so a pair that was refused cannot become admitted by
    an entry the request does not also carry. Demonstrated on the word that
    closed round-1 flip 1: it moves 'suppression of X' from admitted to refused
    and leaves the two rescue cases untouched.
    """
    assert kind_of("suppression") == KIND_INHIBITION
    assert kind_of("fumonisin") == ""
    assert compare_scope(FUMONISIN_REQUESTED, FUMONISIN_STAGE0).compatible is True
    assert compare_scope(SAPONIN_REQUESTED, SAPONIN_STAGE0).compatible is True
