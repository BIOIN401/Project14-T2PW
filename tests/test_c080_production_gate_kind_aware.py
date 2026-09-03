"""C-080 / F-108 -- the PRODUCTION release gate follows the same identity rule as the scorer.

Why this file exists
--------------------
``bench/semantic_production.py::_audit_entities`` carried a **second, independent
copy** of the accession-collision predicate, and that copy is the one that gates
real runs: ``pipeline/strict_quarantine.py`` imports
``evaluate_production_semantics``, and ``no_real_id_or_name_conflict`` is a member
of ``pipeline/release_status.SEMANTIC_GATING_CHECKS``.

That copy had **no kind check of any sort** -- any accession answering to two
differently-*named* rows was a conflict. That is the exact predicate C-073's
review rejected in the pipeline, for contradicting **D-035 clause 3c** (a matching
stable external identifier is *proof* that two differently-named rows are the same
entity), and the exact rule the product owner's 2026-08-23 identity ruling forbids
flagging. C-076 corrected ``bench/semantic.py``, the acceptance scorer, and only
that; between C-076 and this card the two seams disagreed, so the scorer said
``EntB``/``holo-EntB`` on ``uniprot:P0ADI4`` was not a conflict while the
production gate went on demoting the leg for it.

What is pinned here
-------------------
1. **within-kind agreement no longer demotes a run** -- on the real ``EntB``/
   ``holo-EntB`` (``uniprot:P0ADI4``) and ``EntE``/``enterobactin synthase``
   (``uniprot:P10378``) shapes, and on the committed legs that carry them;
2. **cross-kind still fires** and still fails the check -- without this the card
   is a blanket disabling. Pinned on a constructed payload AND on the real
   ``drugbank:DB00114`` / ``ALAS2`` / ``Pyridoxal 5'-phosphate`` collision in
   ``runs_verify/2026-08-21_2239/PMC12856317/research``;
3. **the placeholder arm, the census and both returned counts are untouched** --
   ``_audit_entities`` also emits ``placeholder_claims_real_identity`` and
   ``placeholder_not_distinguished`` and returns
   ``(id_check, placeholder_check, census, forged_count, backed)``. A patch that
   made collisions kind-aware while quietly dropping a forgery finding is the
   failure mode this card had to avoid;
4. **one definition, not three** -- both seams read
   ``semantic.accession_claimed_across_kinds``, proved behaviourally by patching
   it once and observing both;
5. **the gate still gates** -- the check is still in ``SEMANTIC_GATING_CHECKS``
   and a genuine cross-kind conflict still demotes a run end to end;
6. **merge rule 6** -- replayed across every committed leg, no leg flips to
   ``release_ready`` and ``strict_acceptance_eligible`` stays ``False``.

Offline and deterministic: no network, no database, no LLM, no live leg. Every
input is a constructed payload or a committed artifact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench import semantic as bench_semantic  # noqa: E402
from t2pw.bench import semantic_production as sp  # noqa: E402
from t2pw.bench.semantic import (  # noqa: E402
    CHECK_ID_CONFLICT,
    CHECK_PLACEHOLDER_IDENTITY,
    ERR_FALSE_REAL_IDENTIFIERS,
    ERR_PLACEHOLDER_BACKED_PROTEINS,
)
from t2pw.pipeline.release_status import (  # noqa: E402
    RELEASE_READY,
    REVIEW_REQUIRED,
    SEMANTIC_GATING_CHECKS,
    classify_release_status,
    semantic_verdict,
)

COLLISION = "accession_claimed_by_multiple_entities"
FORGERY = "placeholder_claims_real_identity"
NOT_DISTINGUISHED = "placeholder_not_distinguished"

#: The name of the shared predicate. Looked up dynamically, never imported at
#: module level: importing it would turn every test in this file into a
#: symbol-absence failure on the base SHA, and G9 says symbol absence is not
#: proof. Every assertion below is on emitted findings or on a classified status.
PREDICATE = "accession_claimed_across_kinds"

#: The two real within-kind pairs measured at T-105, both claimants of both pairs
#: in ``entities.proteins``.
ENTB = "P0ADI4"
ENTE = "P10378"

#: The real cross-kind collision, measured in T-104's committed artifacts.
PLP = "DB00114"

RUNS = ROOT / "runs_verify"

#: The committed legs whose ``no_real_id_or_name_conflict`` verdict this card
#: moves, and the one it must NOT move. Measured, not assumed -- see the corpus
#: tests below, which re-derive the classification from the artifacts.
WITHIN_KIND_LEGS = (
    "2026-08-18_1328/papers/PMC12096016/strict",
    "2026-08-18_1328/papers/PMC12452463/strict",
    "2026-08-22_2147/papers/PMC12096016/research",
    "2026-08-22_2147/papers/PMC12452463/strict",
)
CROSS_KIND_LEGS = ("2026-08-21_2239/papers/PMC12856317/research",)


# ---------------------------------------------------------------------------
# Payloads. Semantically clean apart from whatever the test injects, so a failed
# check names the injected defect and nothing else.
# ---------------------------------------------------------------------------
def payload(proteins=None, compounds=None):
    return {
        "pathway": {"name": "enterobactin biosynthesis"},
        "entities": {
            "compounds": list(compounds if compounds is not None else [
                {"name": "chorismate"},
                {"name": "2,3-dihydroxybenzoate"},
                {"name": "enterobactin"},
            ]),
            "proteins": list(proteins if proteins is not None else [
                {"name": "EntB", "uniprot": ENTB, "identity_status": "verified"},
            ]),
        },
        "processes": {"reactions": [
            {"name": "dihydroxybenzoate formation", "inputs": ["chorismate"],
             "outputs": ["2,3-dihydroxybenzoate"], "enzymes": ["EntB"],
             "organism": "Escherichia coli", "evidence": "the paper states this step"},
            {"name": "enterobactin assembly", "inputs": ["2,3-dihydroxybenzoate"],
             "outputs": ["enterobactin"], "enzymes": ["EntE"],
             "organism": "Escherichia coli", "evidence": "the paper states this step"},
        ]},
    }


def within_kind_payload():
    """The two REAL T-105 pairs, all four claimants in ``entities.proteins``."""
    return payload(proteins=[
        {"name": "EntB", "uniprot": ENTB, "identity_status": "verified"},
        {"name": "holo-EntB", "uniprot": ENTB, "identity_status": "verified"},
        {"name": "EntE", "uniprot": ENTE, "identity_status": "verified"},
        {"name": "enterobactin synthase", "uniprot": ENTE, "identity_status": "verified"},
    ])


def cross_kind_payload():
    """The real ``drugbank:DB00114`` collision: a protein and a metabolite."""
    return payload(
        proteins=[
            {"name": "EntB", "uniprot": ENTB, "identity_status": "verified"},
            {"name": "ALAS2", "drugbank": PLP, "identity_status": "verified"},
        ],
        compounds=[
            {"name": "chorismate"},
            {"name": "2,3-dihydroxybenzoate"},
            {"name": "enterobactin"},
            {"name": "Pyridoxal 5'-phosphate", "drugbank": PLP},
        ],
    )


def evaluate(case, **kw):
    kw.setdefault("requested_pathway", "enterobactin biosynthesis")
    kw.setdefault("requested_organism", "Escherichia coli")
    return sp.evaluate_production_semantics(case, **kw)


def collisions(report):
    return [f for f in report.checks[CHECK_ID_CONFLICT].findings if f.get("kind") == COLLISION]


def legs(names):
    out = []
    for name in names:
        leg = RUNS / name
        if all((leg / artifact).is_file() for artifact in
               ("quarantine_report.json", "coverage_summary.json", "final_mapped.json")):
            out.append((name, leg))
    return out


#: The three artifacts a leg needs before the replay harness can reconstruct it.
_REPLAYABLE = ("quarantine_report.json", "coverage_summary.json", "final_mapped.json")


def all_committed_legs():
    """The legs GIT has. F-178 (ORCH-722): the name was aspirational, not true.

    This walked the filesystem and called the result "committed". It is not the
    same population: a benchmark run leaves an UNTRACKED run directory, so T-109
    took the corpus from 83 to 93 and ``test_no_committed_leg_flips_to_release_ready``
    went red in the primary checkout while staying green in every worktree. Second
    instance of the same defect in one wave -- the first was
    ``test_d088_coverage_diagnostics._committed_legs``.

    THIS IS A FIXED REGRESSION CORPUS and its value depends on being fixed. Silently
    absorbing whatever the last benchmark left means the assertion below changes
    meaning between machines, which is the opposite of a regression test.

    WHAT THE ABSORBED LEG ACTUALLY SHOWED, recorded rather than discarded, because
    dropping a signal while "fixing attribution" is the failure mode F-177 warns
    about. ``2026-09-02_2052/papers/PMC12444477/strict`` replays to ``release_ready``
    while its committed record says ``diagnostic_only``. That is NOT the flip this
    test hunts: it replays identically WITH and WITHOUT the id check, so dropping
    the check is not what moves it. The leg FAILED in production and took the
    gate-failure path to ``diagnostic_only``, and this harness feeds only the
    quarantine and coverage artifacts to ``classify_release_status``, which that
    path is not reachable from. So the corpus filter admits a leg the harness cannot
    model. Evaluation-instrument observation, NOT a production defect, and it is
    registered as such rather than asserted away.
    """

    out = []
    if not RUNS.is_dir():
        return out
    try:
        listed = set(subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "runs_verify"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.split())
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip("git is required to enumerate the committed corpus: %s" % exc)
    for run in sorted(RUNS.iterdir()):
        papers = run / "papers"
        if not papers.is_dir():
            continue
        for paper in sorted(papers.iterdir()):
            for leg in sorted(paper.iterdir()):
                if not all((leg / artifact).is_file() for artifact in _REPLAYABLE):
                    continue
                if not all(
                    (leg / artifact).relative_to(ROOT).as_posix() in listed
                    for artifact in _REPLAYABLE
                ):
                    continue  # present on disk, absent from git: not this corpus
                out.append(("%s/papers/%s/%s" % (run.name, paper.name, leg.name), leg))
    return out


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. The correction. FAILS ON THE BASE SHA -- on base both pairs are collisions.
# ---------------------------------------------------------------------------
def test_within_kind_accessions_emit_no_collision_from_the_production_gate():
    """G9. The real EntB/holo-EntB and EntE/enterobactin synthase shapes.

    Asserted on the EMITTED FINDINGS, not on the presence of a symbol: on the
    base SHA this list holds two ``accession_claimed_by_multiple_entities``
    findings and ``ok`` is ``False``.
    """

    report = evaluate(within_kind_payload(), min_connected_reactions=2)
    check = report.checks[CHECK_ID_CONFLICT]
    assert [f.get("kind") for f in check.findings] == []
    assert collisions(report) == []
    assert check.ok is True
    assert CHECK_ID_CONFLICT not in report.failed_checks


def test_within_kind_agreement_does_not_demote_the_run():
    """The consequence that matters: the whole semantic verdict passes."""

    report = evaluate(within_kind_payload(), min_connected_reactions=2)
    evaluation, _reason, failed, _evaluability = semantic_verdict(report)
    assert CHECK_ID_CONFLICT not in failed
    assert evaluation == "passed"
    assert report.ok is True


def test_the_committed_within_kind_legs_stop_failing_the_check():
    """The same correction on real payloads rather than constructed ones."""

    measured = legs(WITHIN_KIND_LEGS)
    assert len(measured) == len(WITHIN_KIND_LEGS), "committed artifacts moved"
    for name, leg in measured:
        report = sp.evaluate_production_semantics(read(leg / "final_mapped.json"))
        assert collisions(report) == [], name
        assert report.checks[CHECK_ID_CONFLICT].ok is True, name


# ---------------------------------------------------------------------------
# 2. Cross-kind still fires. Without this the card is a blanket disabling.
# ---------------------------------------------------------------------------
def test_cross_kind_collision_is_still_a_conflict_and_still_fails_the_check():
    report = evaluate(cross_kind_payload(), min_connected_reactions=2)
    check = report.checks[CHECK_ID_CONFLICT]
    found = collisions(report)
    assert len(found) == 1
    assert found[0]["namespace"] == "drugbank"
    assert found[0]["identifier"] == PLP.casefold()
    assert sorted(found[0]["entities"]) == ["ALAS2", "Pyridoxal 5'-phosphate"]
    assert check.ok is False
    assert CHECK_ID_CONFLICT in report.failed_checks


def test_the_committed_cross_kind_leg_keeps_failing_the_check():
    """``drugbank:DB00114`` on ``ALAS2`` (protein) and ``Pyridoxal 5'-phosphate``
    (compound), in T-104's own artifacts. This leg is named in the charter as one
    the correction clears; measured, it is cross-kind and must NOT clear."""

    measured = legs(CROSS_KIND_LEGS)
    assert len(measured) == len(CROSS_KIND_LEGS), "committed artifacts moved"
    for name, leg in measured:
        report = sp.evaluate_production_semantics(read(leg / "final_mapped.json"))
        found = collisions(report)
        assert len(found) == 1, name
        assert found[0]["identifier"] == PLP.casefold(), name
        assert sorted(found[0]["entities"]) == ["ALAS2", "Pyridoxal 5'-phosphate"], name
        assert report.checks[CHECK_ID_CONFLICT].ok is False, name


def test_a_shared_accession_within_one_kind_and_one_name_is_not_a_conflict_either():
    """The other half of the pipeline's rule: same kind, same normalized name.

    A row written into two buckets is one entity twice, not a type error, and was
    never a finding -- the ``len(distinct) > 1`` guard already excluded it. Pinned
    so a later edit cannot turn a routing artefact into a collision.
    """

    same_name = payload(
        proteins=[{"name": "EntB", "uniprot": ENTB}],
        compounds=[{"name": "chorismate"}, {"name": "2,3-dihydroxybenzoate"},
                   {"name": "enterobactin"}, {"name": "entB", "uniprot": ENTB}],
    )
    report = evaluate(same_name, min_connected_reactions=2)
    assert collisions(report) == []


# ---------------------------------------------------------------------------
# 3. The placeholder arm, the census and both counts are untouched.
#    THESE PASS ON THE BASE SHA TOO -- that is the point.
# ---------------------------------------------------------------------------
def forger_payload():
    """A placeholder posing as a real mapping, and NO collision of any kind.

    Deliberately collision-free: this fixture's obligation is that the correction
    left the forgery arm alone, so it must hold IDENTICALLY on the base SHA and at
    the tip. A within-kind collision beside it is a separate obligation, pinned in
    the test after it.
    """

    return payload(proteins=[
        {"name": "EntB", "uniprot": ENTB, "identity_status": "verified"},
        {"name": "EntE", "uniprot": ENTE, "identity_status": "verified"},
        {"name": "Unknown", "identity_status": "placeholder", "uniprot": "P99999"},
    ])


def test_placeholder_forgery_still_fires_and_still_fails_both_checks():
    """UNCHANGED BEHAVIOUR: this passes on the base SHA and at the tip alike."""

    entities = bench_semantic._entities(forger_payload())
    id_check, placeholder_check, census, forged, backed = sp._audit_entities(entities)

    assert [f.get("kind") for f in id_check.findings] == [FORGERY]
    assert id_check.findings[0]["pointer"] == "/entities/proteins/2"
    assert id_check.findings[0]["identifiers"] == {"uniprot": "P99999"}
    assert id_check.ok is False

    assert [f.get("kind") for f in placeholder_check.findings] == [NOT_DISTINGUISHED]
    assert placeholder_check.findings[0]["pointer"] == "/entities/proteins/2"
    assert placeholder_check.ok is False

    assert forged == 1
    assert backed == 1
    assert census == {
        "verified": 2, "placeholder": 1, "unresolved": 0,
        "proteins_total": 3, "compounds_total": 3,
    }


def test_a_within_kind_collision_beside_a_forgery_leaves_the_forgery_alone():
    """The failure mode this card had to avoid, stated directly: the collision
    stops being a finding and the forgery does not go with it. On the base SHA
    ``id_check.findings`` carries both."""

    both = payload(proteins=[
        {"name": "EntB", "uniprot": ENTB, "identity_status": "verified"},
        {"name": "holo-EntB", "uniprot": ENTB, "identity_status": "verified"},
        {"name": "Unknown", "identity_status": "placeholder", "uniprot": "P99999"},
    ])
    id_check, placeholder_check, census, forged, backed = sp._audit_entities(
        bench_semantic._entities(both))
    assert [f.get("kind") for f in id_check.findings] == [FORGERY]
    assert [f.get("kind") for f in placeholder_check.findings] == [NOT_DISTINGUISHED]
    assert id_check.ok is False
    assert (forged, backed) == (1, 1)
    assert census == {
        "verified": 2, "placeholder": 1, "unresolved": 0,
        "proteins_total": 3, "compounds_total": 3,
    }


def test_the_five_tuple_and_the_error_counts_are_unchanged():
    """``_audit_entities`` returns ``(id_check, placeholder_check, census,
    forged_count, backed)`` and feeds two ``scientific_errors`` keys."""

    result = sp._audit_entities(bench_semantic._entities(forger_payload()))
    assert len(result) == 5
    id_check, placeholder_check, census, forged, backed = result
    assert id_check.name == CHECK_ID_CONFLICT
    assert placeholder_check.name == CHECK_PLACEHOLDER_IDENTITY
    assert isinstance(census, dict) and isinstance(forged, int) and isinstance(backed, int)

    report = evaluate(forger_payload(), min_connected_reactions=2)
    assert report.scientific_errors[ERR_FALSE_REAL_IDENTIFIERS] == 1
    assert report.scientific_errors[ERR_PLACEHOLDER_BACKED_PROTEINS] == 1
    assert report.identity_census == census
    assert CHECK_ID_CONFLICT in report.failed_checks
    assert CHECK_PLACEHOLDER_IDENTITY in report.failed_checks


def test_a_correctly_marked_placeholder_is_still_not_a_forgery():
    """The census still counts it, and neither check fails."""

    honest = payload(proteins=[
        {"name": "EntB", "uniprot": ENTB, "identity_status": "verified"},
        # A correctly-formed placeholder records WHY it is one.
        {"name": "Unknown", "identity_status": "placeholder",
         "mapping_meta": {"fallback_used": True}},
    ])
    id_check, placeholder_check, census, forged, backed = sp._audit_entities(
        bench_semantic._entities(honest))
    assert id_check.findings == [] and id_check.ok is True
    assert placeholder_check.findings == [] and placeholder_check.ok is True
    assert (forged, backed) == (0, 1)
    assert census["placeholder"] == 1 and census["proteins_total"] == 2


# ---------------------------------------------------------------------------
# 4. One definition, not three. The card's actual purpose.
# ---------------------------------------------------------------------------
def test_both_seams_read_one_predicate_and_a_change_in_it_is_observable_in_both(monkeypatch):
    """Behavioural, not structural: patch the shared predicate ONCE and watch the
    production gate change with it.

    ``raising=False`` deliberately: on the base SHA the name does not exist and
    the patch is inert, so the production gate still emits its cross-kind
    collision and this test fails on the FINDINGS -- a behavioural failure, not a
    symbol-absence one.
    """

    calls = []

    def never_a_conflict(claimants):
        calls.append(sorted(set(claimants)))
        return False

    monkeypatch.setattr(bench_semantic, PREDICATE, never_a_conflict, raising=False)
    report = evaluate(cross_kind_payload(), min_connected_reactions=2)
    assert collisions(report) == []
    assert calls, "the production gate never consulted the shared predicate"
    # It was handed the two claimants of DB00114, one of each kind.
    kinds = {kind for call in calls for kind, _name in call}
    assert kinds == {"protein", "compound"}, calls
    assert any(len(call) == 2 for call in calls), calls


def test_the_scorer_reads_the_same_object_as_the_production_gate():
    """Structural companion to the test above: one function, not two copies."""

    predicate = getattr(bench_semantic, PREDICATE, None)
    assert predicate is not None
    assert getattr(sp._s, PREDICATE) is predicate
    # The production module holds no second copy of the comparison.
    assert "left[0] != right[0]" not in Path(sp.__file__).read_text(encoding="utf-8")
    assert Path(bench_semantic.__file__).read_text(encoding="utf-8").count(
        "left[0] != right[0]") == 1


def test_the_predicate_needs_both_a_kind_and_a_name_difference():
    """A unit pin on the shared definition. Structural, like the test above it --
    on the base SHA it fails because the name is absent, which is NOT the G9
    proof; the G9 proof is the findings-based tests at the top of this file."""

    predicate = getattr(bench_semantic, PREDICATE)
    assert predicate([("protein", "entb"), ("protein", "holo entb")]) is False
    assert predicate([("compound", "atp"), ("compound", "adenosine triphosphate")]) is False
    assert predicate([("protein", "entb"), ("compound", "entb")]) is False
    assert predicate([("protein", "alas2"), ("compound", "pyridoxal 5 phosphate")]) is True
    assert predicate([]) is False
    assert predicate([("protein", "entb")]) is False


# ---------------------------------------------------------------------------
# 5. The gate still gates.
# ---------------------------------------------------------------------------
def test_the_check_is_still_a_gating_semantic_check():
    assert CHECK_ID_CONFLICT in SEMANTIC_GATING_CHECKS
    assert CHECK_ID_CONFLICT == "no_real_id_or_name_conflict"


def _release_ready_coverage():
    """A coverage verdict whose technical chain reaches ``release_ready``.

    ``explicit_argument`` deliberately: it excludes C-074's F-100 rule, so the
    only thing left that can demote is the semantic cap under test.
    """

    return {
        "requested_core_declared": True,
        "requested_core_source": "explicit_argument",
        "coverage_ratio": 1.0,
        "minimum_core_satisfied": True,
        "surviving_processes": 2,
        "unmatched_terms": [],
        "reasons": [],
    }


def _classify(report):
    evaluation, reason, failed, evaluability = semantic_verdict(report)
    return classify_release_status(
        coverage=_release_ready_coverage(),
        pipeline_executed=True,
        strict_gates_passed=True,
        semantic_evaluation=evaluation,
        semantic_not_evaluated_reason=reason,
        semantic_failed_checks=failed,
        semantic_check_evaluability=evaluability,
    ).to_dict()


def test_a_cross_kind_conflict_still_demotes_a_run_end_to_end():
    status = _classify(evaluate(cross_kind_payload(), min_connected_reactions=2))
    assert status["status"] == REVIEW_REQUIRED
    assert status["strict_acceptance_eligible"] is False
    assert "semantic_evaluation_failed:%s" % CHECK_ID_CONFLICT in status["reasons"]


def test_a_within_kind_run_with_every_anchor_matched_is_release_ready():
    """THE T-106 PREDICTION, made executable (charter section 6).

    A leg that fails ONLY the within-kind check and has all declared anchors
    matched becomes ``release_ready`` -- legitimately, under the 2026-08-23
    ruling. It is the correct outcome, not an unexplained strict-rate
    improvement, and it is what this assertion is for. On the base SHA the same
    inputs land on ``review_required``.

    No COMMITTED leg has this shape -- see the corpus test below, where the
    upper-bound replay flips nothing.
    """

    status = _classify(evaluate(within_kind_payload(), min_connected_reactions=2))
    assert status["status"] == RELEASE_READY
    assert status["strict_acceptance_eligible"] is True
    assert status["reasons"] == []


# ---------------------------------------------------------------------------
# 6. Merge rule 6 -- the safety property. Replayed over the whole committed corpus.
# ---------------------------------------------------------------------------
def _replay(leg, drop_id_check):
    release = read(leg / "quarantine_report.json").get("release") or {}
    coverage = read(leg / "coverage_summary.json")
    recorded = [str(name) for name in (release.get("semantic_failed_checks") or ())]
    evaluation = str(release.get("semantic_evaluation") or "not_evaluated")
    failed = [name for name in recorded if name != CHECK_ID_CONFLICT] if drop_id_check else recorded
    if evaluation == "failed" and not failed:
        evaluation = "passed"
    return classify_release_status(
        coverage=coverage,
        pipeline_executed=bool(release.get("pipeline_executed")),
        strict_gates_passed=bool(release.get("strict_gates_passed")),
        semantic_evaluation=evaluation,
        semantic_not_evaluated_reason=release.get("semantic_not_evaluated_reason") or "",
        semantic_failed_checks=failed,
        semantic_check_evaluability=release.get("semantic_check_evaluability") or [],
        retrieval_attempts=release.get("retrieval_attempts"),
        expansion_blocked_reason=release.get("expansion_blocked_reason") or "",
    ).to_dict(), release


def test_no_committed_leg_flips_to_release_ready():
    """MERGE RULE 6. The upper bound: the check is assumed to PASS on every leg.

    The correction can only ever turn this check from failing to passing, never
    the reverse, so a corpus with no flip here has no flip under the rule as
    implemented -- or under any kind-aware rule at all.
    """

    corpus = all_committed_legs()
    assert len(corpus) >= 22, "sample size collapsed: %d legs" % len(corpus)
    flipped = []
    for name, leg in corpus:
        after, release = _replay(leg, drop_id_check=True)
        if after["status"] == RELEASE_READY and release.get("status") != RELEASE_READY:
            flipped.append((name, after["reasons"]))
        if release.get("status") != RELEASE_READY:
            assert after["strict_acceptance_eligible"] is False, name
    assert flipped == []


def test_f094s_leg_stays_review_required_under_the_second_independent_cap():
    """``PMC12452463/strict`` is F-094's leg and F-094 is a PRODUCT_CONTRACT 13
    violation. Removing the semantic failure inverts the cap order: the run stays
    ``release_ready`` through the semantic cap and the INCOMPLETE-CORE cap then
    fires, because the leg declares a core with three unmatched anchors."""

    leg = RUNS / "2026-08-22_2147" / "papers" / "PMC12452463" / "strict"
    if not (leg / "quarantine_report.json").is_file():
        pytest.skip("committed artifact absent")
    before, release = _replay(leg, drop_id_check=False)
    after, _ = _replay(leg, drop_id_check=True)
    assert before["status"] == release.get("status") == REVIEW_REQUIRED
    assert after["status"] == REVIEW_REQUIRED
    assert after["strict_acceptance_eligible"] is False
    assert after["reasons"] == [
        "requested_core_anchors_unmatched:2,3-dihydroxybenzoate (DHB),EntA,Fur"]


def test_the_measured_id_verdict_reproduces_every_committed_record():
    """What makes the counterfactual above trustworthy: on every leg that
    recorded a semantic failure, this tree's measurement of the check on the
    leg's own payload agrees with what the run recorded -- EXCEPT on the legs
    this card deliberately moves."""

    corpus = all_committed_legs()
    assert len(corpus) >= 22
    moved, agreed = [], 0
    for name, leg in corpus:
        release = read(leg / "quarantine_report.json").get("release") or {}
        if str(release.get("semantic_evaluation") or "") != "failed":
            continue
        recorded = CHECK_ID_CONFLICT in (release.get("semantic_failed_checks") or ())
        report = sp.evaluate_production_semantics(read(leg / "final_mapped.json"))
        measured = not report.checks[CHECK_ID_CONFLICT].ok
        if recorded == measured:
            agreed += 1
        else:
            moved.append(name)
    assert agreed >= 1
    # Exactly the four within-kind legs move, and nothing else does.
    assert sorted(moved) == sorted(WITHIN_KIND_LEGS)
