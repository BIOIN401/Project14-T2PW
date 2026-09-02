"""C-102 / D-072 (Ruling A) -- the contract-accepted coverage denominator.

**G9 LABEL: CORRECTION OF PRE-EXISTING OBSERVABLE BEHAVIOUR.**

The denominator is wrong today, so this file is not a new-capability suite and
does not get to skip a base failure. The behavioural proof lives in
``docs/pwml_recovery_sprint/evidence/c102_g9_denominator_proof.py``: it asks the
PUBLIC acceptance surface which requested-core denominator it reports for
``PMC12782028/strict`` and, on the base tree, finds only 27 -- four of whose
terms that same case forbids exporting -- against 23 beside the preserved 27 at
the tip. That is a statement about values in the report, not about a symbol.

F-144 discipline, which the previous card paid three correction rounds for.
Every test here whose point is a null or a negative result

  * runs the production path rather than a reconstruction of it,
  * asserts that a finding OF THE KIND THAT PATH EMITS was actually produced
    before asserting anything about its content, and
  * was attacked by mutation before it was reported --
    ``docs/pwml_recovery_sprint/evidence/c102_mutation_attack.log``.

D-083 follow-on 1 landed in test 4: the deep copy `to_dict` takes of
``coverage_reconciliation`` shipped with no proof, and REV-102's mutation R5
reverting it went green. Test 4 now asserts identity and the consequence, and
R5 is registered in the attack set beside M1-M7.

Test 11 also encodes the attack inside the suite: it drives the same real
coverage block through the same function with the exclusion list emptied, which
is exactly the pre-change instrument, and pins that raw and accepted then
COLLAPSE onto each other. If the exclusion ever stops happening, 2, 9 and 11 go
red together.
"""

from __future__ import annotations

import dataclasses
import io
import json
import subprocess
from pathlib import Path

import pytest

from t2pw.bench.acceptance import (
    COVERAGE_MEASURED,
    COVERAGE_UNDEFINED_ALL_FORBIDDEN,
    contract_accepted_coverage,
    forbidden_coverage_match,
    score_run,
)
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path
from t2pw.bench.render import render_text
from t2pw.bench.semantic import (
    CHECK_ID_CONFLICT,
    ERR_FALSE_REAL_IDENTIFIERS,
    validate_semantic_coverage,
)

REPO = Path(__file__).resolve().parents[1]

#: The sharpest F-132 instance: the four gold-forbidden terms this leg drew as
#: requested core are the exact four Priority-1 survivors on the same paper.
PINNED_RUN = REPO / "runs_verify/2026-08-24_1428"
PINNED_LEG = PINNED_RUN / "papers/PMC12782028/strict/quarantine_report.json"
#: Frozen by the artifact, not by this file: 6 matched of 27 drawn = 0.222222,
#: against an unchanged minimum of 0.5.
RAW_MATCHED, RAW_DENOMINATOR, RAW_RATIO = 6, 27, 0.222222
FORBIDDEN_ON_12782028 = ("LBR", "LIPA", "SREBF1", "SREBF2")

#: The six papers the recovered ORCH-702 probe named.
F132_PAPERS = (
    "PMC12096016", "PMC12312563", "PMC12444477",
    "PMC12452463", "PMC12782028", "PMC12856317",
)

_GOLD = {case.paper_id: case for case in load_gold_set(pinned_gold_set_path()).cases}


@pytest.fixture(scope="module")
def gold():
    return _GOLD


@pytest.fixture(scope="module")
def raw_coverage():
    return json.load(io.open(PINNED_LEG, encoding="utf-8"))["coverage"]


@pytest.fixture(scope="module")
def recon(gold, raw_coverage):
    return contract_accepted_coverage(gold["PMC12782028"], raw_coverage)


def _block(terms, matched=(), minimum=0.5, declared=True):
    """A coverage block shaped exactly like the artifact's, for the edge cases."""

    return {
        "requested_core_terms": list(terms),
        "requested_core_declared": declared,
        "matched_terms": list(matched),
        "unmatched_terms": [t for t in terms if t not in matched],
        "coverage_ratio": (len(matched) / len(terms)) if terms else 0.0,
        "thresholds": {"min_core_processes": 1, "min_core_coverage": minimum},
    }


# ---------------------------------------------------------------------------
# 1-2. The same term: present in raw, absent from accepted.
# ---------------------------------------------------------------------------
def test_1_forbidden_terms_are_present_in_the_raw_anchor_set(gold, raw_coverage, recon):
    """The raw draw contains terms this same case forbids exporting. F-132 itself."""

    terms = raw_coverage["requested_core_terms"]
    assert len(terms) == RAW_DENOMINATOR, "the pinned artifact moved; re-pin before reading on"

    # The finding first, THEN its content: assert the matcher actually returned a
    # ForbiddenIdentifier for each, so this cannot pass over an empty list.
    hits = {t: forbidden_coverage_match(gold["PMC12782028"], t) for t in FORBIDDEN_ON_12782028}
    assert all(hit is not None for hit in hits.values()), hits
    assert {hit.kind for hit in hits.values()} == {"heading_or_prose", "regulator_as_metabolite"}
    assert set(FORBIDDEN_ON_12782028) <= set(terms)

    # And they are preserved in the raw half, unchanged in value and meaning.
    assert recon["raw_denominator"] == RAW_DENOMINATOR
    assert recon["raw_matched"] == RAW_MATCHED
    assert recon["raw_ratio"] == pytest.approx(RAW_RATIO, abs=5e-7)
    assert recon["raw_below_minimum"] is True


def test_2_the_same_terms_are_excluded_from_the_accepted_denominator(recon):
    assert recon["excluded_count"] == len(FORBIDDEN_ON_12782028)
    assert {e["term"] for e in recon["excluded_terms"]} == set(FORBIDDEN_ON_12782028)
    assert recon["accepted_denominator"] == RAW_DENOMINATOR - len(FORBIDDEN_ON_12782028) == 23
    assert recon["accepted_state"] == COVERAGE_MEASURED
    assert recon["accepted_ratio"] == pytest.approx(6 / 23, abs=1e-6)
    # The threshold did NOT move. PRODUCT_CONTRACT 7 is respected exactly, and
    # this leg still fails it -- measured, not predicted, and reported as it fell.
    assert recon["min_core_coverage"] == 0.5
    assert recon["accepted_below_minimum"] is True
    assert recon["cleared_by_reconciliation"] is False


# ---------------------------------------------------------------------------
# 3. Priority 1 must remain able to detect a forbidden export.
# ---------------------------------------------------------------------------
def test_3_priority1_still_scores_a_forbidden_identifier_that_is_exported(gold):
    """Exempting a term from coverage does NOT make exporting it acceptable.

    Runs the PRODUCTION Priority-1 path -- ``validate_semantic_coverage`` -- over
    a payload that exports ``SREBF1`` with a real UniProt accession, which is
    exactly the row D-072 must leave punishable.
    """

    payload = {
        "entities": {
            "compounds": [],
            "proteins": [{"name": "SREBF1", "uniprot_id": "P36956"}],
            "protein_complexes": [],
        },
        "processes": {"reactions": [], "interactions": []},
    }
    report = validate_semantic_coverage(gold["PMC12782028"], payload, mode="strict")

    # The finding of the specific kind this path emits, before anything about it.
    findings = [
        f
        for f in report.checks[CHECK_ID_CONFLICT].findings
        if f.get("kind") == "false_real_identifier"
    ]
    assert findings, report.checks[CHECK_ID_CONFLICT].findings
    assert [f["name"] for f in findings] == ["SREBF1"]
    assert findings[0]["identifiers"] == {"uniprot": "P36956"}
    assert report.scientific_errors[ERR_FALSE_REAL_IDENTIFIERS] >= 1

    # And the coverage instrument, on the same case, still withholds it -- the
    # two verdicts coexist on one row, which is the whole point of the ruling.
    assert forbidden_coverage_match(gold["PMC12782028"], "SREBF1") is not None


# ---------------------------------------------------------------------------
# 4. Guard rail 3 -- extracted but withheld stays visible.
# ---------------------------------------------------------------------------
def test_4_withheld_terms_remain_in_the_diagnostics(gold, recon):
    """Out of the denominator is never out of the record."""

    by_term = {e["term"]: e for e in recon["excluded_terms"]}
    assert set(by_term) == set(FORBIDDEN_ON_12782028)
    for entry in by_term.values():
        assert entry["forbidden_name"]
        assert entry["forbidden_kind"]
        assert entry["matched_in_raw"] is False
    # SREBF2 is reached by ALIAS, not by its own gold row: dropping alias support
    # would silently shrink the exclusion set and this pins it.
    assert by_term["SREBF2"]["forbidden_name"] == "SREBF1"

    # It survives serialization to the report a human actually reads.
    report = score_run(PINNED_RUN)
    serialized = json.dumps(report.to_dict())
    for term in FORBIDDEN_ON_12782028:
        assert f'"term": "{term}"' in serialized

    # -- D-083 follow-on 1 / REV-102 F7 -------------------------------------
    # `to_dict` deep-copies this block. A SHALLOW `dict(...)` hands the caller
    # the live `excluded_terms` list by identity, so a reader mutating the
    # serialized report reaches back into the scored leg. That fix shipped with
    # nothing asserting it: REV-102's mutation R5, reverting
    # `deepcopy(dict(...))` to `dict(...)`, left the whole file GREEN.
    #
    # The load-bearing property is IDENTITY, not equality -- `==` holds for a
    # shallow copy too, which is exactly the vacuous guard F-144 is about. The
    # stakes are low and stated plainly: a shallow copy cannot produce a wrong
    # number, and nothing in the tree mutates `to_dict()` output today. This
    # guards a future caller. R5 is now in the attack set
    # (`evidence/c102_mutation_attack.py`), so the next reviewer inherits the
    # mutation instead of re-deriving it.
    live = next(p for p in report.papers if p.paper_id == "PMC12782028").legs["strict"]
    # The finding first, THEN its content: this leg really does carry the four
    # withheld terms, so nothing below can pass over an empty list.
    assert live.coverage_reconciliation["excluded_count"] == len(FORBIDDEN_ON_12782028)
    emitted = report.to_dict()
    leg = next(
        p["legs"]["strict"] for p in emitted["papers"] if p["paper_id"] == "PMC12782028"
    )["coverage_reconciliation"]

    # The nested list, and each entry inside it, are FRESH objects. Asserting
    # on the outer dict would prove nothing -- `dict(...)` copies that much.
    assert leg["excluded_terms"] is not live.coverage_reconciliation["excluded_terms"]
    assert leg["excluded_terms"][0] is not live.coverage_reconciliation["excluded_terms"][0]

    # ...and the CONSEQUENCE, which is what the deep copy exists to prevent and
    # is the better test because it does not depend on the mechanism: mutating
    # the serialized structure -- in place on one entry, and structurally on the
    # list -- leaves the scored leg untouched.
    leg["excluded_terms"][0]["term"] = "CLOBBERED"
    leg["excluded_terms"].clear()
    assert {e["term"] for e in live.coverage_reconciliation["excluded_terms"]} == set(
        FORBIDDEN_ON_12782028
    )
    # So the next reader of the same report still finds all four, and guard
    # rail 3 survives a careless caller as well as a careless scorer.
    again = json.dumps(report.to_dict())
    for term in FORBIDDEN_ON_12782028:
        assert f'"term": "{term}"' in again
    assert "CLOBBERED" not in again


# ---------------------------------------------------------------------------
# 5-7. Guard rail 1 -- the exemption is narrow, and it is case-local.
# ---------------------------------------------------------------------------
def test_5_a_supported_non_forbidden_anchor_is_still_required(gold, raw_coverage, recon):
    """A term that is merely hard, or merely unmatched, keeps its place."""

    excluded = {e["term"] for e in recon["excluded_terms"]}
    # HMGCR was DRAWN, went UNMATCHED, and is not forbidden. It stays in the
    # denominator and still costs this leg its coverage.
    assert "HMGCR" in raw_coverage["unmatched_terms"]
    assert forbidden_coverage_match(gold["PMC12782028"], "HMGCR") is None
    assert "HMGCR" not in excluded
    # CYP51A1 matched and is not forbidden: it stays in BOTH halves.
    assert "CYP51A1" in raw_coverage["matched_terms"]
    assert "CYP51A1" not in excluded
    assert recon["accepted_matched"] == RAW_MATCHED


def test_6_a_similar_but_unlisted_name_is_still_required(gold):
    """Containment is refused in both directions, so near-misses stay required."""

    lipid_a = gold["PMC12444477"]
    # The gold's own worked example: `coenzyme A` is forbidden as a protein,
    # `coenzyme A ligase` is a real enzyme and must never be condemned by it.
    assert forbidden_coverage_match(lipid_a, "coenzyme A") is not None
    assert forbidden_coverage_match(lipid_a, "coenzyme A ligase") is None
    assert forbidden_coverage_match(gold["PMC12782028"], "SREBF1-AS1") is None
    assert forbidden_coverage_match(gold["PMC12782028"], "LIPA2") is None

    block = _block(["coenzyme A", "coenzyme A ligase", "LpxC"], matched=["LpxC"])
    out = contract_accepted_coverage(lipid_a, block)
    assert [e["term"] for e in out["excluded_terms"]] == ["coenzyme A"]
    assert out["accepted_denominator"] == 2
    assert out["accepted_ratio"] == pytest.approx(0.5)


def test_7_the_exclusion_is_case_local(gold):
    """Forbidden on paper A is not forbidden on paper B, and never leaks."""

    # The finding first: MenD really is forbidden on PMC12096016.
    assert forbidden_coverage_match(gold["PMC12096016"], "MenD") is not None
    # And really is not on PMC12452463, which has its own, different list.
    assert forbidden_coverage_match(gold["PMC12452463"], "MenD") is None
    # Symmetrically, in the other direction.
    assert forbidden_coverage_match(gold["PMC12452463"], "RyhB") is not None
    assert forbidden_coverage_match(gold["PMC12096016"], "RyhB") is None

    block = _block(["MenD", "EntB"], matched=["EntB"])
    on_a = contract_accepted_coverage(gold["PMC12096016"], block)
    on_b = contract_accepted_coverage(gold["PMC12452463"], block)
    assert on_a["accepted_denominator"] == 1 and on_a["excluded_count"] == 1
    assert on_b["accepted_denominator"] == 2 and on_b["excluded_count"] == 0
    assert on_b["accepted_ratio"] == pytest.approx(0.5)
    assert on_b["accepted_ratio"] == pytest.approx(on_b["raw_ratio"])


# ---------------------------------------------------------------------------
# 8. The empty accepted denominator, said out loud.
# ---------------------------------------------------------------------------
def test_8_an_empty_accepted_denominator_is_undefined_and_not_a_success(gold):
    """Every drawn term forbidden => there is NO rate, and no coverage success."""

    case = gold["PMC12782028"]
    block = _block(["SREBF1", "LIPA", "LBR"], matched=[])
    out = contract_accepted_coverage(case, block)

    assert out["excluded_count"] == 3
    assert out["accepted_denominator"] == 0
    assert out["accepted_state"] == COVERAGE_UNDEFINED_ALL_FORBIDDEN
    # Not 1.0, not 0.0, and explicitly not a pass: `accepted_below_minimum` is
    # None because an undefined rate cannot be compared to a threshold, and
    # `cleared_by_reconciliation` is False because nothing cleared -- there is
    # simply nothing left the case permits the pipeline to cover.
    assert out["accepted_ratio"] is None
    assert out["accepted_below_minimum"] is None
    assert out["cleared_by_reconciliation"] is False
    # The raw half still says what actually happened.
    assert out["raw_denominator"] == 3 and out["raw_below_minimum"] is True
    # And a matched forbidden term does not rescue it either.
    matched_all = contract_accepted_coverage(case, _block(["SREBF1"], matched=["SREBF1"]))
    assert matched_all["accepted_ratio"] is None
    assert matched_all["accepted_state"] == COVERAGE_UNDEFINED_ALL_FORBIDDEN
    assert matched_all["excluded_terms"][0]["matched_in_raw"] is True


# ---------------------------------------------------------------------------
# 9. Reported separately, and shown to differ on a real case.
# ---------------------------------------------------------------------------
def test_9_raw_and_accepted_are_reported_separately_and_differ(recon):
    report = score_run(PINNED_RUN)
    data = report.to_dict()

    leg = next(
        p["legs"]["strict"] for p in data["papers"] if p["paper_id"] == "PMC12782028"
    )["coverage_reconciliation"]
    assert leg["raw_denominator"] != leg["accepted_denominator"] == 23
    assert leg["raw_ratio"] != leg["accepted_ratio"]
    assert leg["raw_ratio"] == pytest.approx(RAW_RATIO, abs=5e-7)

    # Priorities 4 and 5 read the reconciliation; priorities 1-3 do not.
    priorities = {entry["rank"]: entry for entry in data["acceptance_priorities"]}
    for rank in (4, 5):
        entry = priorities[rank]["requested_core_coverage"]
        assert entry["legs_with_forbidden_terms"] > 0
        # F5: the counts travel on the entry, the 12 KB row array does not.
        assert "legs" not in entry
        assert entry["legs_at"] == "coverage_reconciliation_corpus.legs"
    for rank in (1, 2, 3):
        assert "requested_core_coverage" not in priorities[rank]
    # Priority 1 is UNCHANGED by this card, raw and accepted alike.
    assert priorities[1]["raw"] == priorities[1]["observed"]

    corpus = data["coverage_reconciliation_corpus"]
    assert corpus["legs_still_below_minimum"] == ["PMC12782028:strict"]
    assert corpus["legs_cleared_by_reconciliation"] == []


# ---------------------------------------------------------------------------
# 10. The F-132 affected-paper regression, over the six named papers.
# ---------------------------------------------------------------------------
def test_10_f132_population_regression_over_the_six_papers(gold):
    """Offline, over every committed quarantine_report.json. No leg is re-run."""

    listed = subprocess.run(
        ["git", "ls-files", "*quarantine_report.json"],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", check=True,
    )
    paths = sorted(line.strip() for line in listed.stdout.splitlines() if line.strip())
    # FLOOR, not a census. This one catches the population SHRINKING, which is
    # why it is `>=` while the two derived pins below are `==`. Moved 62 -> 72 by
    # C-106: commit `e77ad3d` ("T-107 official result") committed
    # `runs_verify/2026-08-28_1816`, which carries 10 quarantine_report.json
    # files. Raising the floor keeps it catching a shrink against current truth.
    # 72 -> 83 at C-115. `runs_verify/2026-09-01_1612` -- T-108's run, 10 papers,
    # 20 legs, `complete: true` -- was committed at `479128b3` ("T-108 RESULT: NOT
    # ACCEPTED. Ran once, 20/20 legs, 6.37h, scored honestly and preserved") and
    # contributes EXACTLY 11 quarantine_report.json files; its other nine legs
    # never reached quarantine and emit none. The other fourteen runs still sum
    # to 72. That tree is preserved deliberately as the record of a failed
    # release candidate, so the CORPUS is correct and the floor was stale.
    # Measured, not hand-counted: `evidence/orch717_census_probe.py`, P1 = 83.
    assert len(paths) >= 83, "the committed artifact population shrank; re-pin before reading on"

    legs = 0
    affected_papers: dict[str, int] = {}
    withheld = 0
    cleared: list[str] = []
    for rel in paths:
        leg_dir = (REPO / rel).parent
        case = gold.get(leg_dir.parent.name)
        if case is None:
            continue
        coverage = json.load(io.open(REPO / rel, encoding="utf-8")).get("coverage") or {}
        out = contract_accepted_coverage(case, coverage)
        if out is None:
            continue
        legs += 1
        if out["excluded_count"]:
            affected_papers[case.paper_id] = affected_papers.get(case.paper_id, 0) + 1
            withheld += out["excluded_count"]
        if out["cleared_by_reconciliation"]:
            cleared.append(f"{case.paper_id}:{leg_dir.name}")

    # CENSUS PIN -- 62 -> 72, moved deliberately by C-106 under merge rule 4.
    # Commit `e77ad3d` ("T-107 official result") committed
    # `runs_verify/2026-08-28_1816`, which contributes EXACTLY 10 legs; every
    # other run's contribution is unchanged, and 62 is precisely the sum over the
    # other thirteen runs. Per-run attribution: `evidence/c106_census_probe.log`.
    #
    # This stays `==` and must NOT be relaxed to `>=`. The floor above uses `>=`
    # to catch the corpus SHRINKING. This one asserts the loop above VISITED
    # every leg it should have, and the census is how it knows how many that is.
    # A `>=` here would let a leg join the corpus and go unvisited -- a quiet
    # vacuity of exactly the F-144 shape. Moving it by hand whenever a run is
    # committed is the correct cost: that is the moment someone confirms the new
    # legs were meant to be there.
    # 72 -> 83 at C-115, and this is the confirmation the paragraph above asks
    # for: all 11 quarantine-carrying legs `runs_verify/2026-09-01_1612` added
    # are gold-set papers with readable coverage, so every one of them is a leg
    # this loop MUST visit -- none is skipped. Per-run attribution (the probe's
    # ATTRIBUTION table): that run contributes 11 legs and the other fourteen
    # runs still sum to exactly 72, unmoved. T-108 ran 20/20 legs once, was
    # scored NOT ACCEPTED, and its tree was kept on purpose. Stays `==`.
    assert legs == 83
    # Every one of the six ORCH-702 papers is still in the population.
    assert set(F132_PAPERS) <= set(affected_papers), sorted(affected_papers)
    # And TWO papers outside it.
    #
    # PMC13231680, which ORCH-702 could not see: it counted forbidden terms only
    # among the UNMATCHED, so a forbidden term the pipeline actually matched --
    # and was given coverage credit for -- was invisible to it. PMC13231680 has
    # three such legs and no unmatched forbidden term.
    #
    # PMC12180156 JOINS THE SET at C-113. This is a DELIBERATE pinned-baseline
    # move under merge rule 4's second clause, NOT a new capability: F-150 half 1
    # (gold blob `aee8cb4f` -> `36f4b7b6`) adds the two delta spellings of
    # 5-aminolevulinic acid to that case's EXISTING forbidden entry, so the
    # spelling three committed legs actually emitted stops escaping the gate.
    #
    # C-113'S RECORD, KEPT AS HISTORY AND TO BE READ IN C-113'S TENSE (C-117 /
    # F-171 relabelled it; nothing about the attribution changed). AT C-113 NO
    # LEG JOINED THE CORPUS -- `legs` was unmoved at 72 then. `legs` has since
    # moved to the value the assert above now pins, when T-108's run tree joined
    # the corpus, which C-115 then re-pinned to. F-171 measured that tree at 83
    # already at `479128b3`, T-108's own result commit, so C-115 moved the PINS
    # and not the corpus; either way it is a LATER and SEPARATE move and it did
    # not disturb the three-leg attribution below. Three legs already in the
    # corpus at C-113 changed classification, and it is exactly these three:
    #   runs_verify/2026-08-21_2239/papers/PMC12180156/research  excluded 0 -> 1
    #   runs_verify/2026-08-21_2239/papers/PMC12180156/strict    excluded 0 -> 1
    #   runs_verify/2026-08-28_1816/papers/PMC12180156/research  excluded 0 -> 1
    # each drawing the single term "delta-aminolevulinic acid" in its Greek
    # spelling (U+03B4) as requested core. The other five committed
    # PMC12180156 legs are unchanged, and no leg anywhere leaves the set.
    #
    # CONFIRMED these three legs were meant to be here, and HOW: a FOURTH
    # committed leg, runs_verify/2026-08-24_1402/papers/PMC12180156/research,
    # also draws a Greek delta spelling -- but it draws the ENZYME,
    # "<delta>-aminolevulinic acid synthase (ALAS2)", which the paper really
    # names and which this same gold case quotes as acceptable. That leg is NOT
    # in the set, because `forbidden_match` refuses containment. So the gate
    # separates the fabricated metabolite from the legitimately named enzyme on
    # the real corpus, leg by leg -- which is what makes the three additions
    # correct rather than merely expected.
    #
    # Measured INDEPENDENTLY of this assert, against both gold blobs in one
    # tree: `evidence/c113_census_attribution.py` / `c113_census_attribution.log`.
    assert set(affected_papers) - set(F132_PAPERS) == {"PMC12180156", "PMC13231680"}
    # 92 -> 97 at C-106: the same run under the same commit --
    # `runs_verify/2026-08-28_1816` under `e77ad3d` -- withheld 5 further terms,
    # and the other thirteen runs still summed to 92. DERIVED from the census
    # above, and it had never executed against the grown corpus before C-106
    # because the `legs` assert aborted the test first. That is why F-151,
    # REV-104 and the wave handoff all described this repair as "re-pin to 72"
    # and all three were wrong. Measured, not inferred:
    # `evidence/c106_census_probe.log`.
    #
    # 97 -> 100 at C-113, and this pin had STILL never executed under the
    # post-F-150 gold: the set-equality above aborts the test before reaching it,
    # so at the failed merge `b05a7281` nobody knew whether it moved. It does.
    # The delta is exactly the three legs named above, one withheld term apiece:
    #   runs_verify/2026-08-21_2239/.../PMC12180156/research  excluded_count 0 -> 1
    #   runs_verify/2026-08-21_2239/.../PMC12180156/strict    excluded_count 0 -> 1
    #   runs_verify/2026-08-28_1816/.../PMC12180156/research  excluded_count 0 -> 1
    # Every other leg's `excluded_count` is unchanged and the other 69 legs still
    # sum to 97 under the pre-edit gold. Deliberate pinned-baseline move under
    # merge rule 4's second clause; NOT a new capability. Measured, not read off
    # a failure message: `evidence/c113_census_attribution.log`.
    # 100 -> 109 at C-115: `runs_verify/2026-09-01_1612` withholds 9 forbidden
    # terms across those 11 legs, and the other fourteen runs still sum to
    # exactly 100. DERIVED from the census above, so it moves with it. The
    # affected-paper SET is unchanged -- the same eight papers, PMC12180156 and
    # PMC13231680 still the only two outside F132_PAPERS -- so the set equality
    # above did not move; only this count did. Measured, not read off a failure
    # message: `evidence/orch717_census_probe.py` P4 = 109 and its per-run table.
    assert withheld == 109
    # Measured, not predicted: no leg in the corpus clears the unchanged
    # threshold on the accepted ratio. Reported as it fell.
    assert cleared == []


# ---------------------------------------------------------------------------
# 11. Non-vacuity: restoring the contradictory denominator collapses 2 and 9.
# ---------------------------------------------------------------------------
def test_11_restoring_the_contradictory_denominator_collapses_raw_and_accepted(
    gold, raw_coverage, recon
):
    """The pre-change instrument, reconstructed by DATA rather than by code.

    Emptying the case's ``forbidden_identifiers`` is exactly the state the scorer
    was in before D-072: nothing to exclude, so the accepted half is forced onto
    the raw half. If that collapse ever happens on the real case, tests 2 and 9
    are asserting nothing, and this test says so first.
    """

    unreconciled = dataclasses.replace(gold["PMC12782028"], forbidden_identifiers=())
    collapsed = contract_accepted_coverage(unreconciled, raw_coverage)

    assert collapsed["excluded_count"] == 0
    assert collapsed["accepted_denominator"] == collapsed["raw_denominator"] == RAW_DENOMINATOR
    assert collapsed["accepted_ratio"] == pytest.approx(collapsed["raw_ratio"], abs=5e-7)
    assert collapsed["accepted_below_minimum"] is True

    # ...and on the real case they must NOT collapse. Stated here so the two
    # halves are read together: this is the difference the card exists to make.
    assert recon["accepted_denominator"] != recon["raw_denominator"]
    assert recon["accepted_ratio"] != recon["raw_ratio"]
    assert recon["excluded_count"] == 4


# ---------------------------------------------------------------------------
# 12-13. REV-102 F1 -- the NUMERATOR half, which shipped untested.
#
# D-072's text says "excluded from the accepted positive-coverage DENOMINATOR".
# This card removes forbidden terms from the numerator as well, and that
# deviation was escalated rather than taken silently. It survived the first
# round with no assertion behind it: reverting the numerator half (mutation M7)
# left all eleven tests green. These two are what make it bite.
# ---------------------------------------------------------------------------
def test_12_a_matched_forbidden_term_is_withheld_from_the_numerator_too(gold):
    """Denominator-only removal INVERTS the ruling on a MATCHED forbidden term.

    The term leaves the denominator but stays in the numerator, so a pipeline is
    paid a coverage bonus for exactly the export Priority 1 penalises, and
    obeying the gold scores BELOW breaking it. Measured on the real corpus the
    same reading reports ratios above 1.0 on nine committed legs, eight of them
    at 1.2000 -- not a coverage rate at all
    (``evidence/c102_numerator_verify.log``).
    """

    case = gold["PMC12782028"]
    terms = ["SREBF1", "HMGCR", "CYP51A1"]
    out = contract_accepted_coverage(case, _block(terms, matched=["SREBF1"]))

    # The finding first: the forbidden term really was MATCHED on this block.
    assert out["excluded_count"] == 1
    assert out["excluded_terms"][0]["matched_in_raw"] is True
    assert out["raw_matched"] == 1 and out["raw_denominator"] == 3

    # Denominator-only would report 1/2 = 0.5 here. Both-sides reports 0/2 = 0.0.
    assert out["accepted_matched"] == 0
    assert out["accepted_denominator"] == 2
    assert out["accepted_ratio"] == pytest.approx(0.0)

    # And the property that removes the inversion: matching a forbidden term is
    # exactly NEUTRAL. Obeying the gold and breaking it score the same.
    obeyed = contract_accepted_coverage(case, _block(terms, matched=["HMGCR"]))
    broke = contract_accepted_coverage(case, _block(terms, matched=["HMGCR", "SREBF1"]))
    assert obeyed["accepted_ratio"] == pytest.approx(0.5)
    assert broke["accepted_ratio"] == pytest.approx(0.5)
    assert obeyed["accepted_ratio"] == broke["accepted_ratio"]


def test_13_the_accepted_rate_is_a_rate_on_every_committed_leg(gold):
    """A numerator can never exceed its denominator, on any committed leg."""

    listed = subprocess.run(
        ["git", "ls-files", "*quarantine_report.json"],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", check=True,
    )
    paths = sorted(line.strip() for line in listed.stdout.splitlines() if line.strip())

    checked = with_matched_forbidden = 0
    for rel in paths:
        leg_dir = (REPO / rel).parent
        case = gold.get(leg_dir.parent.name)
        if case is None:
            continue
        coverage = json.load(io.open(REPO / rel, encoding="utf-8")).get("coverage") or {}
        out = contract_accepted_coverage(case, coverage)
        if out is None:
            continue
        checked += 1
        assert out["accepted_matched"] <= out["accepted_denominator"], rel
        if out["accepted_ratio"] is not None:
            assert 0.0 <= out["accepted_ratio"] <= 1.0, f"{rel}: {out['accepted_ratio']}"
        if any(e["matched_in_raw"] for e in out["excluded_terms"]):
            with_matched_forbidden += 1
            # Withheld from the numerator, so the accepted count MUST drop.
            assert out["accepted_matched"] < out["raw_matched"], rel

    # CENSUS PIN -- 62 -> 72. Same delta, same run, same commit as test 10's
    # `legs`: `e77ad3d` committed `runs_verify/2026-08-28_1816` and its 10 legs.
    # Stays `==`; see the note beside test 10 for why a `>=` here would make this
    # loop vacuous rather than tolerant.
    # 72 -> 83 at C-115. Same delta, same run, same commit as test 10's `legs`:
    # `479128b3` committed `runs_verify/2026-09-01_1612` (T-108) and its 11
    # quarantine-carrying legs. Stays `==`; a `>=` here would let a leg join the
    # corpus and go unvisited, which is what the note beside test 10 refuses.
    assert checked == 83
    # Non-vacuity, and the reason this test is not an empty loop: the corpus
    # really does contain matched forbidden terms. Twenty-three legs when
    # c102_numerator_verify.log measured it independently; TWENTY-NINE now.
    # 23 -> 26 at C-106 is the 3 matched-forbidden legs
    # `runs_verify/2026-08-28_1816` contributes under commit `e77ad3d`; the other
    # thirteen runs still sum to 23. Like `withheld` in test 10 this pin is
    # DERIVED and sat behind the census assert, so C-106 was the first time it
    # had ever run against the grown corpus.
    #
    # 26 -> 29 at C-113 is the 3 legs that F-150 half 1 newly puts a MATCHED
    # forbidden term on -- and only those three; no leg stops carrying one:
    #   runs_verify/2026-08-21_2239  PMC12180156:research   +1
    #   runs_verify/2026-08-21_2239  PMC12180156:strict     +1
    #   runs_verify/2026-08-28_1816  PMC12180156:research   +1
    # i.e. +2 in run tree `runs_verify/2026-08-21_2239` and +1 in
    # `runs_verify/2026-08-28_1816`; the other twelve run trees are untouched and
    # the corpus still sums to 26 under the pre-edit gold. All 72 tracked legs
    # live under `runs_verify/`; `runs/` contributes none. The same three legs
    # move `withheld` 97 -> 100 and put PMC12180156 into `affected_papers` in
    # test 10 -- see the attribution and the enzyme/metabolite confirmation
    # beside that assert. `checked` above is UNMOVED at 72: no leg joined.
    # Deliberate pinned-baseline move under merge rule 4's second clause, NOT a
    # new capability. Measured independently of this assert:
    # `evidence/c113_census_attribution.py` / `.log`. Stays `==`.
    # 29 -> 33 at C-115 is the 4 legs of `runs_verify/2026-09-01_1612` that
    # carry a MATCHED forbidden term; no leg anywhere stops carrying one, and
    # the other fourteen run trees still sum to exactly 29. Like `withheld`
    # this pin is DERIVED and sat behind the census assert, so C-115 is the
    # first time it has executed against the T-108-grown corpus. Deliberate
    # pinned-baseline move under merge rule 4's second clause, NOT a new
    # capability. Measured: `evidence/orch717_census_probe.py` P5 = 33. `==`.
    assert with_matched_forbidden == 33, with_matched_forbidden


# ---------------------------------------------------------------------------
# 14. REV-102 F4 -- the rendered surface a human reads to make the T-107 call.
# ---------------------------------------------------------------------------
def test_14_the_reconciliation_is_rendered_under_priorities_4_and_5_only():
    """The text report is the surface the call is actually made from."""

    text = render_text(score_run(PINNED_RUN))
    assert "ACCEPTANCE PRIORITIES" in text, "the report did not render at all"

    marker = "D-072 requested-core reconciliation"
    assert text.count(marker) == 2, f"expected one block under each of 4 and 5, got {text.count(marker)}"

    # The affected leg, both halves, and the terms named -- guard rail 3 on the
    # surface, not only in the JSON.
    assert "PMC12782028:strict" in text
    assert "raw 6/27=0.222" in text
    assert "accepted 6/23=0.261" in text
    for term in FORBIDDEN_ON_12782028:
        assert f"{term} [" in text, term
    assert "withheld (still forbidden to export, still recorded)" in text

    # A leg with no forbidden draw prints nothing, so the surface reads exactly
    # as it did before wherever the reconciliation has nothing to say.
    corpus = score_run(PINNED_RUN).coverage_reconciliation_corpus
    quiet = [row for row in corpus["legs"] if not row["excluded_count"]]
    assert quiet, "no unaffected leg in this run -- the assertion below is vacuous"
    for row in quiet:
        assert f"- {row['paper_id']}:{row['mode']}  raw " not in text
