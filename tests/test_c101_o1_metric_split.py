"""C-101 -- the O-1 metric split, the row-aware sentinel seam, and Ruling B.

**G9 LABEL: NEW ACCEPTANCE TESTS FOR NEW CAPABILITY.**

Three capabilities land here and none of them is a correction of prior *correct*
observable behaviour, so none carries a fabricated base failure:

* the **16/5 split** (D-070) -- the instrument could not previously distinguish a
  PathBank sentinel row from a generated functional wrapper, and the F-141
  population had no metric at all;
* the **row-aware sentinel seam** (D-074) -- the name-only matcher is not a prior
  correct behaviour being restored. It was the only behaviour available, because
  the scorer passed only a name;
* the **raw / accepted split and the variance statuses** (D-073).

The ONE regression half is
:func:`test_9_placeholder_backed_proteins_unchanged_on_pinned_payload`, labelled
below, which pins ``placeholder_backed_proteins`` on a REAL pinned-run payload
and must pass at the base SHA and at the tip alike. Its new-capability half is
:func:`test_9b_the_split_of_the_pinned_leg_is_one_sentinel_and_eight_wrappers`,
split out precisely so the regression half can RUN on the base SHA.

The authoritative A/B row for the sentinel work is recorded in
``docs/pwml_recovery_sprint/evidence/c101_a4_authoritative_row.md``:
``runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json``
``/entities/proteins/4``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from t2pw.bench.goldset import (
    ForbiddenIdentifier,
    GoldSetError,
    GoldTerm,
    load_gold_set,
    pinned_gold_set_path,
)

# The symbols this card ADDS are imported inside the tests that need them, not at
# module scope. That is deliberate and it is what makes G9 measurable: the
# regression half (test 9) pins a PRE-EXISTING number and has to RUN on the base
# SHA, which it cannot do if importing this module needs symbols the base tree
# does not have. At base the new-capability tests below fail on the missing
# import -- correctly, and visibly -- while test 9 passes on both trees.
from t2pw.bench.semantic import (
    CHECK_ID_CONFLICT,
    CHECK_PLACEHOLDER_IDENTITY,
    ERR_FALSE_REAL_IDENTIFIERS,
    ERR_PLACEHOLDER_BACKED_PROTEINS,
    validate_semantic_coverage,
)

REPO = Path(__file__).resolve().parents[1]
#: The authoritative A/B leg (A4). Not interchangeable with the other two
#: archived sentinel legs, which serve only as preservation controls.
PINNED_LEG = REPO / "runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json"
#: The two archived sentinel rows that are NOT the A/B target (A4 controls).
CONTROL_LEGS = (
    REPO / "runs_verify/2026-08-24_1428/papers/PMC12444477/strict/final_mapped.json",
    REPO / "runs_verify/2026-08-25_1216/papers/PMC12444477/strict/final_mapped.json",
)


# ---------------------------------------------------------------------------
# Fixtures / builders.
# ---------------------------------------------------------------------------
_GOLD_BY_ID = {case.paper_id: case for case in load_gold_set(pinned_gold_set_path()).cases}


@pytest.fixture(scope="module")
def gold():
    return _GOLD_BY_ID


@pytest.fixture(scope="module")
def lipid_a(gold):
    return gold["PMC12444477"]


def _load(path: Path):
    return json.load(io.open(path, encoding="utf-8"))


def sentinel_row(name="Unknown", **over):
    """The confirmed bare PathBank sentinel, shaped exactly like the pinned row."""

    row = {
        "name": name,
        "species": "Arabidopsis thaliana",
        "organism": "Arabidopsis thaliana",
        "species_id": 4,
        "pathbank_protein_id": 9659,
        "uniprot_id": "Unknown",
        "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
        "identity_status": "placeholder",
        "mapping_meta": {
            "provider": "PathBankDB",
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "identity_status": "placeholder",
            "pathbank_protein_id": 9659,
            "fallback_used": True,
            "cross_species_placeholder": True,
        },
    }
    row.update(over)
    return row


def wrapper_row(name):
    """A generated single-protein PathWhiz wrapper -- O-1b's population."""

    return {
        "name": name,
        "generated": True,
        "generation_reason": "single_protein_pathwhiz_wrapper",
        "identity_status": "placeholder",
        "mapping_meta": {
            "identity_status": "placeholder",
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "generation_reason": "single_protein_pathwhiz_wrapper",
            # Load-bearing: without it `placeholder_claims_real_identity` fires
            # with `placeholder_does_not_record_fallback_used` and the row is
            # reported as a forged identity instead of reaching the tolerance
            # branch. The real pinned wrapper rows all carry it.
            "fallback_used": True,
        },
    }


def payload(proteins=(), complexes=()):
    return {
        "entities": {
            "compounds": [],
            "proteins": list(proteins),
            "protein_complexes": list(complexes),
        },
        "processes": {},
    }


def census_of(case, pl):
    return validate_semantic_coverage(case, pl, mode="strict").identity_census


def tolerance_findings(case, pl):
    report = validate_semantic_coverage(case, pl, mode="strict")
    return [
        f
        for f in report.checks[CHECK_PLACEHOLDER_IDENTITY].findings
        if f["kind"] == "unknown_backed_protein_not_acceptable"
    ]


# ===========================================================================
# Section 6 -- the 16/5 split.  NEW CAPABILITY.
# ===========================================================================
def test_1_five_sentinel_rows_count_as_sentinels_and_not_as_wrappers(lipid_a):
    """Charter test 1. Catches a sentinel silently classified as a wrapper."""

    c = census_of(lipid_a, payload(proteins=[sentinel_row() for _ in range(5)]))
    assert c["placeholder_sentinel_rows"] == 5
    assert c["placeholder_generated_wrappers"] == 0
    assert c["placeholder_other_rows"] == 0


def test_2_sixteen_generated_wrappers_count_as_wrappers_and_not_as_sentinels(lipid_a):
    """Charter test 2. Catches a wrapper silently classified as a sentinel."""

    rows = [wrapper_row(f"Enz{i}") for i in range(16)]
    c = census_of(lipid_a, payload(complexes=rows))
    assert c["placeholder_generated_wrappers"] == 16
    assert c["placeholder_sentinel_rows"] == 0
    assert c["placeholder_other_rows"] == 0


def test_3_row_satisfying_both_shapes_lands_in_exactly_one_category(lipid_a):
    """Charter test 3. THE mutual-exclusion test.

    Catches the pre-existing defect in terms: ``pathbank_unknown_sentinel`` was
    computed on a separate pass from ``placeholder_backed`` with no relationship
    enforced, so a row could be counted in both and nothing noticed.
    """

    both = sentinel_row(generated=True, generation_reason="single_protein_pathwhiz_wrapper")
    from t2pw.pipeline.entity_identity import (
        is_generated_complex_wrapper,
        is_pathbank_unknown_protein,
    )

    assert is_pathbank_unknown_protein(both) and is_generated_complex_wrapper(both), (
        "the fixture must genuinely satisfy BOTH shapes or this test is vacuous"
    )
    c = census_of(lipid_a, payload(proteins=[both]))
    assert c["placeholder_sentinel_rows"] + c["placeholder_generated_wrappers"] + c[
        "placeholder_other_rows"
    ] == 1, "a row counted twice, or dropped"
    assert c["placeholder_sentinel_rows"] == 1, "sentinel is the narrower statement and wins"
    assert c["placeholder_generated_wrappers"] == 0


def test_4_invariant_holds_on_a_payload_where_other_is_non_zero(lipid_a):
    """Charter test 4. `other` is REPORTED, never assumed to be zero.

    Catches an instrument that asserts the partition by dropping the remainder,
    which would hide the first row that does not fit.
    """

    other = {
        "name": "MysteryProtein",
        "identity_status": "placeholder",
        "mapping_meta": {"identity_status": "placeholder"},
    }
    pl = payload(proteins=[sentinel_row(), other], complexes=[wrapper_row("LpxA")])
    report = validate_semantic_coverage(lipid_a, pl, mode="strict")
    c = report.identity_census
    backed = report.scientific_errors[ERR_PLACEHOLDER_BACKED_PROTEINS]
    assert c["placeholder_other_rows"] == 1, "the remainder must be non-zero for this test to bite"
    assert backed == (
        c["placeholder_sentinel_rows"]
        + c["placeholder_generated_wrappers"]
        + c["placeholder_other_rows"]
    )
    assert backed == 3


def test_5_category_three_counts_a_correctly_withheld_identity(lipid_a):
    """Charter test 5. F-141's confirmed-correct population."""

    row = {
        "name": "EntC",
        "mapped_ids": {},
        "mapping_meta": {
            "identity_verdict": {
                "identity": "uniprot:P0AEJ2",
                "organism": "Escherichia coli",
                "checks": {"species": "unknown"},
                "judged_candidate": {},
            },
            "candidates": [],
        },
    }
    c = census_of(lipid_a, payload(proteins=[row]))
    assert c["withheld_identity_correct"] == 1
    assert c["withheld_identity_recoverable"] == 0


def test_6_category_four_counter_moves_on_a_constructed_recoverable_row(lipid_a):
    """Charter test 6. The counter must be EXERCISABLE, not hard-wired to zero.

    A counter that cannot move is vacuous, and reporting a vacuous zero as a
    measurement is the failure this test exists to prevent.
    """

    row = {
        "name": "EntC",
        "mapped_ids": {},
        "mapping_meta": {
            "identity_verdict": {
                "identity": "uniprot:P0AEJ2",
                "organism": "Escherichia coli",
                "checks": {"species": "unknown"},
                # BOTH sides present, and the rung still unknown.
                "judged_candidate": {"organism": "Escherichia coli", "taxonomy_id": "562"},
            },
            "candidates": [{"organism": "Escherichia coli"}],
        },
    }
    c = census_of(lipid_a, payload(proteins=[row]))
    assert c["withheld_identity_recoverable"] == 1, "category 4 is hard-wired and cannot move"
    assert c["withheld_identity_correct"] == 0


def test_6b_an_unrecognised_species_rung_lands_in_the_third_bucket(lipid_a):
    """REV-101 correction 1. The F-141 remainder is REPORTED, not folded.

    Mirrors ``placeholder_other_rows``: a rung this reader does not recognise is
    UNCLASSIFIED, and counting it as confirmed-correct withholding would report a
    mechanism nobody measured as a clean result. Catches exactly that regression.
    """

    def rung_row(rung):
        return {
            "name": "EntC",
            "mapped_ids": {},
            "mapping_meta": {
                "identity_verdict": {
                    "identity": "uniprot:P0AEJ2",
                    "organism": "Escherichia coli",
                    "checks": {"species": rung},
                    "judged_candidate": {},
                },
                "candidates": [],
            },
        }

    # An unrecognised rung -- D-070 O-1c category 6, "other measured mechanism".
    c = census_of(lipid_a, payload(proteins=[rung_row("deferred_pending_review")]))
    assert c["withheld_identity_other"] == 1, "the counter is hard-wired and cannot move"
    assert c["withheld_identity_correct"] == 0, (
        "an unclassified mechanism was reported as correct withholding"
    )
    assert c["withheld_identity_recoverable"] == 0

    # Conflicting species evidence is its OWN row in F-141's table and is not
    # called correct withholding there either.
    for rung in ("mismatch", "conflict"):
        c = census_of(lipid_a, payload(proteins=[rung_row(rung)]))
        assert c["withheld_identity_other"] == 1, rung
        assert c["withheld_identity_correct"] == 0, rung

    # An ABSENT rung stays correct: F-141 calls the two Fur rows -- candidate
    # does not describe the shipped identifier -- withholding CORRECT.
    c = census_of(lipid_a, payload(proteins=[rung_row("")]))
    assert c["withheld_identity_correct"] == 1
    assert c["withheld_identity_other"] == 0


def test_6c_the_third_bucket_zero_is_distinguishable_from_not_evaluated(lipid_a):
    """REV-101 correction 1, second half. A measured 0 is never 'not asked'."""

    scored = validate_semantic_coverage(lipid_a, payload(proteins=[sentinel_row()]), mode="strict")
    assert scored.identity_census["withheld_identity_other"] == 0
    assert scored.identity_census["withheld_identity_evaluated"] is True

    unscored = validate_semantic_coverage(lipid_a, None, mode="strict")
    assert unscored.evaluated is False
    assert "withheld_identity_other" not in unscored.identity_census


def test_7_category_four_zero_is_distinguishable_from_not_evaluated(lipid_a):
    """Charter test 7. PRODUCT_CONTRACT 8 -- `not_evaluated` is never `false`."""

    scored = validate_semantic_coverage(lipid_a, payload(proteins=[sentinel_row()]), mode="strict")
    assert scored.identity_census["withheld_identity_recoverable"] == 0
    assert scored.identity_census["withheld_identity_evaluated"] is True

    unscored = validate_semantic_coverage(lipid_a, None, mode="strict")
    assert unscored.evaluated is False
    assert "withheld_identity_recoverable" not in unscored.identity_census, (
        "a run that produced no payload must not report a measured 0"
    )


def test_8_placeholder_claims_real_identity_takes_precedence_over_every_category(lipid_a):
    """Charter test 8, and D-074 condition 5's hardest edge.

    A row shaped like the sentinel but carrying a fabricated accession must be
    reported as a forged identity, NOT tolerated -- whatever any tolerance says.
    """

    forged = sentinel_row()
    forged["mapped_ids"] = {"uniprot": "P0A6T1", "pathbank_protein_id": 9659}
    forged["uniprot_id"] = "P0A6T1"
    report = validate_semantic_coverage(lipid_a, payload(proteins=[forged]), mode="strict")
    kinds = [f["kind"] for f in report.checks[CHECK_PLACEHOLDER_IDENTITY].findings]
    assert "placeholder_not_distinguished" in kinds
    assert "unknown_backed_protein_not_acceptable" not in kinds, (
        "a forged identity must never be routed through the tolerance branch"
    )


@pytest.mark.skipif(not PINNED_LEG.exists(), reason="pinned run not in this checkout")
def test_9_placeholder_backed_proteins_unchanged_on_pinned_payload(lipid_a):
    """Charter test 9 -- **THE REGRESSION HALF. Must pass at base AND at tip.**

    Pinned against the REAL pinned-run payload, not a synthetic one. D-070
    measured 9 placeholder-backed rows on this leg and 21 across the pinned run;
    this card must not move either number, only split them.
    """

    report = validate_semantic_coverage(lipid_a, _load(PINNED_LEG), mode="strict")
    assert report.scientific_errors[ERR_PLACEHOLDER_BACKED_PROTEINS] == 9

    total = 0
    for leg in sorted((REPO / "runs/2026-08-02_2130/papers").glob("*/*/final_mapped.json")):
        case = _GOLD_BY_ID.get(leg.parent.parent.name)
        if case is None:
            continue
        total += validate_semantic_coverage(
            case, _load(leg), mode=leg.parent.name
        ).scientific_errors[ERR_PLACEHOLDER_BACKED_PROTEINS]
    assert total == 21, "D-070's pinned 21 moved -- this card splits it, never changes it"


@pytest.mark.skipif(not PINNED_LEG.exists(), reason="pinned run not in this checkout")
def test_9b_the_split_of_the_pinned_leg_is_one_sentinel_and_eight_wrappers(lipid_a):
    """NEW CAPABILITY half of test 9 -- the values the split adds beside it.

    Deliberately NOT folded into test 9: the regression half must run on the base
    SHA, and these census keys do not exist there.
    """

    c = validate_semantic_coverage(lipid_a, _load(PINNED_LEG), mode="strict").identity_census
    assert (c["placeholder_sentinel_rows"], c["placeholder_generated_wrappers"]) == (1, 8)
    assert c["placeholder_other_rows"] == 0


@pytest.mark.skipif(not PINNED_LEG.exists(), reason="pinned run not in this checkout")
def test_10_non_vacuity_collapsing_the_split_turns_tests_1_to_3_red(lipid_a):
    """Charter test 10. Prove tests 1-3 are load-bearing, not decorative.

    The collapse simulated here is the pre-change instrument: one number over
    both populations. Under it, tests 1-3's assertions cannot all hold.
    """

    conflated = census_of(lipid_a, payload(proteins=[sentinel_row() for _ in range(5)]))
    collapsed = conflated["placeholder_sentinel_rows"] + conflated["placeholder_generated_wrappers"]
    # Test 1 asserts sentinels==5 AND wrappers==0. A collapsed instrument reports
    # one number, so "wrappers" is indistinguishable from the total.
    assert collapsed == 5
    with pytest.raises(AssertionError):
        assert conflated["placeholder_sentinel_rows"] == 0, "collapse would make test 1 unprovable"

    both = sentinel_row(generated=True, generation_reason="single_protein_pathwhiz_wrapper")
    c = census_of(lipid_a, payload(proteins=[both]))
    double_counted = c["pathbank_unknown_sentinel"] + c["placeholder_generated_wrappers"]
    assert c["placeholder_sentinel_rows"] == 1
    assert double_counted == 1, (
        "the separate-pass census must not double-count the both-shapes row"
    )


# ===========================================================================
# A7 -- the row-aware sentinel seam (D-074).  NEW CAPABILITY.
# ===========================================================================
def test_a7_1_exact_confirmed_sentinel_row_is_tolerated(lipid_a):
    """A7.1. The authoritative A/B row's shape -> tolerated."""

    assert lipid_a.tolerates_unknown_backed("Unknown", sentinel_row()) is True
    assert tolerance_findings(lipid_a, payload(proteins=[sentinel_row()])) == []


def test_a7_2_arbitrary_row_named_unknown_without_sentinel_identity_is_a_finding(lipid_a):
    """A7.2. THE leak test. A name-only allowlist would wrongly excuse this."""

    impostor = {
        "name": "Unknown",
        "identity_status": "placeholder",
        "mapping_meta": {"identity_status": "placeholder", "fallback_used": True},
    }
    assert lipid_a.tolerates_unknown_backed("Unknown", impostor) is False
    findings = tolerance_findings(lipid_a, payload(proteins=[impostor]))
    assert [f["name"] for f in findings] == ["Unknown"]


def test_a7_3_sentinel_like_row_with_the_wrong_pathbank_id_is_a_finding(lipid_a):
    row = sentinel_row(pathbank_protein_id=9660)
    row["mapped_ids"]["pathbank_protein_id"] = 9660
    row["mapping_meta"]["pathbank_protein_id"] = 9660
    assert lipid_a.tolerates_unknown_backed("Unknown", row) is False
    assert len(tolerance_findings(lipid_a, payload(proteins=[row]))) == 1


def test_a7_4_wrong_provenance_or_wrong_uniprot_sentinel_state_is_a_finding(lipid_a):
    wrong_rule = sentinel_row()
    wrong_rule["mapping_meta"]["chosen_rule"] = "some_other_rule"
    wrong_rule["mapping_meta"]["cross_species_placeholder"] = False
    assert lipid_a.tolerates_unknown_backed("Unknown", wrong_rule) is False

    wrong_uniprot = sentinel_row()
    wrong_uniprot["uniprot_id"] = "P0A6T1"
    wrong_uniprot["mapped_ids"]["uniprot"] = "P0A6T1"
    assert lipid_a.tolerates_unknown_backed("Unknown", wrong_uniprot) is False


def test_a7_5_confirmed_sentinel_on_a_different_unauthorized_paper_is_a_finding(gold):
    """A7.5. The licence is per-case and does not travel."""

    for paper_id, case in gold.items():
        if paper_id == "PMC12444477":
            continue
        assert case.unknown_backed_tolerated_sentinel is None
        assert case.tolerates_unknown_backed("Unknown", sentinel_row()) is False, paper_id


def test_a7_6_lpxh_remains_a_priority_finding_on_pmc12444477(lipid_a):
    """A7.6 / A6 / D-074. **Removing this is a merge-rule-6 reject.**

    LpxH is the resolvable *E. coli* enzyme for the organism-dependent ninth
    Raetz step. An Unknown-backed LpxH is a genuine loss, so the paper goes
    9 -> 8, never 9 -> 7.
    """

    assert lipid_a.tolerates_unknown_backed("LpxH", wrapper_row("LpxH")) is False
    findings = tolerance_findings(lipid_a, payload(complexes=[wrapper_row("LpxH")]))
    assert [f["name"] for f in findings] == ["LpxH"]


@pytest.mark.skipif(not PINNED_LEG.exists(), reason="pinned run not in this checkout")
def test_a7_6b_pinned_leg_goes_nine_to_eight_with_lpxh_kept(lipid_a):
    """A6, measured on the authoritative A/B leg rather than on a fixture."""

    findings = tolerance_findings(lipid_a, _load(PINNED_LEG))
    names = sorted(f["name"] for f in findings)
    assert len(findings) == 8, f"expected 9 -> 8, got {len(findings)}: {names}"
    assert "LpxH" in names, "LpxH was removed -- merge rule 6 reject"
    assert "Unknown" not in names, "the confirmed sentinel should be tolerated"
    assert names == ["LpxA", "LpxB", "LpxD", "LpxH", "LpxK", "LpxL", "LpxM", "WaaA"]


@pytest.mark.parametrize("leg", CONTROL_LEGS, ids=lambda p: p.parent.parent.parent.parent.name)
def test_a7_preservation_controls_other_archived_sentinel_legs(lipid_a, leg):
    """A4 preservation controls -- NOT A/B targets.

    Recorded because the A4 determination's safety rests on these three rows
    being identical under the predicate: if the choice of A/B target could move
    an outcome, the ambiguity would have been material and this card would have
    stopped.
    """

    if not leg.exists():
        pytest.skip(f"{leg} not in this checkout")
    from t2pw.pipeline.entity_identity import is_pathbank_unknown_protein

    rows = _load(leg)["entities"].get("proteins") or []
    sentinels = [r for r in rows if is_pathbank_unknown_protein(r)]
    assert len(sentinels) == 1
    assert lipid_a.tolerates_unknown_backed(sentinels[0].get("name"), sentinels[0]) is True


def test_a7_12_non_vacuity_restoring_the_name_only_matcher_turns_tests_red(lipid_a):
    """A7.12. Mutation: call the seam with the NAME ALONE, as before D-074.

    Under the name-only call the confirmed sentinel is refused, so
    ``test_a7_1`` goes red. That is the proof the widened input is load-bearing.
    """

    assert lipid_a.tolerates_unknown_backed("Unknown") is False, (
        "name-only must NOT grant the licence -- otherwise the row is decorative"
    )
    assert lipid_a.tolerates_unknown_backed("Unknown", sentinel_row()) is True


def test_a7_13_non_vacuity_removing_the_sentinel_predicate_turns_tests_red(lipid_a):
    """A7.13. Mutation: drop the gold's row-predicated entry.

    Without it the confirmed sentinel is refused and ``test_a7_1`` goes red.
    """

    import dataclasses

    stripped = dataclasses.replace(lipid_a, unknown_backed_tolerated_sentinel=None)
    assert stripped.tolerates_unknown_backed("Unknown", sentinel_row()) is False


def test_a7_14_non_sentinel_classifications_are_preserved(lipid_a):
    """A7.14. C-100's seven named entities still tolerated; the eight core
    Raetz enzymes still refused. This card narrows nothing and widens nothing
    outside the one row-predicated licence."""

    for name in ("LapA", "YciS", "LapB", "YciM", "Ght", "LabP", "LpxG", "YhcB", "lipoprotein"):
        assert lipid_a.tolerates_unknown_backed(name, wrapper_row(name)) is True, name
    for name in ("LpxA", "LpxC", "LpxD", "LpxB", "LpxK", "WaaA", "LpxL", "LpxM"):
        assert lipid_a.tolerates_unknown_backed(name, wrapper_row(name)) is False, name


# ---------------------------------------------------------------------------
# A3 -- backward compatibility of the widened signature.
# ---------------------------------------------------------------------------
def test_a3_three_documented_states_survive_the_widened_signature(gold):
    """A3. All three states, and the Boolean still only ever NARROWS."""

    from t2pw.bench.goldset import GoldCase, GoldSentinelTolerance

    # 1. no scope declared -> the inherited Boolean, unchanged.
    base = dict(title="t", requested_pathway="p", requested_organism="o")
    inherit_true = GoldCase(paper_id="X", unknown_backed_proteins_acceptable=True, **base)
    inherit_false = GoldCase(paper_id="Y", unknown_backed_proteins_acceptable=False, **base)
    assert inherit_true.tolerates_unknown_backed("anything") is True
    assert inherit_false.tolerates_unknown_backed("anything") is False

    # 2. scope declared, candidate in it -> tolerated.
    # 3. scope declared, candidate absent -> refused.
    scoped = GoldCase(
        paper_id="Z",
        **base,
        unknown_backed_proteins_acceptable=True,
        unknown_backed_tolerated_entities=(GoldTerm(name="LapA", aliases=("YciS",)),),
    )
    assert scoped.tolerates_unknown_backed("LapA") is True
    assert scoped.tolerates_unknown_backed("YciS") is True
    assert scoped.tolerates_unknown_backed("LpxA") is False

    # A False case can NEVER be widened, sentinel or not.
    never = GoldCase(
        paper_id="W",
        **base,
        unknown_backed_proteins_acceptable=False,
        unknown_backed_tolerated_sentinel=GoldSentinelTolerance(
            name="Unknown", pathbank_protein_id=9659, uniprot="Unknown"
        ),
    )
    assert never.tolerates_unknown_backed("Unknown", sentinel_row()) is False


def test_a3_every_name_only_caller_keeps_its_answer(gold):
    """A3. The row defaults to None, so pre-D-074 callers are unaffected."""

    for paper_id, case in gold.items():
        for name in ("Unknown", "LpxA", "LapA", "lipoprotein", "nonsense"):
            one_arg = case.tolerates_unknown_backed(name)
            two_arg = case.tolerates_unknown_backed(name, None)
            assert one_arg == two_arg, f"{paper_id}/{name}"


def test_gold_refuses_a_name_only_sentinel_entry():
    """The gold representation cannot degrade into a name-only allowlist."""

    from t2pw.bench.goldset import _sentinel_tolerance

    with pytest.raises(GoldSetError, match="pathbank_protein_id"):
        _sentinel_tolerance({"name": "Unknown"}, where="case")
    with pytest.raises(GoldSetError, match="pathbank_protein_id"):
        _sentinel_tolerance({"name": "Unknown", "uniprot": "Unknown"}, where="case")


def test_the_name_keyed_scope_cannot_reach_the_sentinel_licence(lipid_a):
    """**Why a name-only read of the gold entry cannot leak.**

    The licence lives in its own field, and the name-only matcher iterates only
    the name-keyed list. No name-keyed read can reach it, whatever the caller.
    """

    assert "Unknown" not in [t.name for t in lipid_a.unknown_backed_tolerated_entities]
    assert lipid_a.unknown_backed_tolerance_match("Unknown") is None
    assert lipid_a.sentinel_tolerance_match("Unknown", None) is None
    assert lipid_a.sentinel_tolerance_match("Unknown", sentinel_row()) is not None


# ===========================================================================
# A5 / A7.7-11 -- Ruling B: raw and accepted Priority 1.  NEW CAPABILITY.
# ===========================================================================
def test_a7_9_10_11_accepted_counts_map_to_the_three_statuses():
    """A7.9/10/11. The band, and PASS_WITHIN_VARIANCE as its own value."""

    from t2pw.bench.acceptance import (
        PRIORITY1_FAIL,
        PRIORITY1_PASS,
        PRIORITY1_PASS_WITHIN_VARIANCE,
        PRIORITY1_TARGET,
        priority1_status,
    )

    assert priority1_status(PRIORITY1_TARGET) == PRIORITY1_PASS
    assert priority1_status(7) == PRIORITY1_PASS_WITHIN_VARIANCE
    assert priority1_status(8) == PRIORITY1_FAIL
    assert PRIORITY1_PASS_WITHIN_VARIANCE != PRIORITY1_PASS
    assert PRIORITY1_PASS not in (PRIORITY1_FAIL,)
    # It must not be readable as a Boolean synonym of PASS.
    assert PRIORITY1_PASS_WITHIN_VARIANCE.startswith(PRIORITY1_PASS)
    assert PRIORITY1_PASS_WITHIN_VARIANCE != PRIORITY1_PASS


def _report_with(rows):
    from t2pw.bench.acceptance import AcceptanceReport

    report = AcceptanceReport(run_dir="x", gold_version="v", gold_path="p")
    report.priority1_rows = list(rows)
    report.errors.totals[ERR_FALSE_REAL_IDENTIFIERS] = len(rows)
    return report


def _p1(report):
    return next(e for e in report.priorities() if e["rank"] == 1)


def test_a7_7_raw_count_is_preserved():
    """A7.7. Raw is the error total, unchanged in meaning."""

    rows = [
        {"paper_id": "P", "mode": "strict", "pointer": f"/e/{i}", "name": f"n{i}",
         "kind": "false_real_identifier", "identifiers": {}, "contract_tolerance": "",
         "accepted": True}
        for i in range(6)
    ]
    entry = _p1(_report_with(rows))
    assert entry["raw"] == 6
    assert entry["observed"] == 6, "the pre-existing field must not change meaning"
    assert entry["ok"] is False, "the absolute Boolean must not be widened by the band"


def test_a7_8_accepted_count_is_computed_separately_and_can_differ():
    """A7.8 + A5. **Raw and accepted must not agree merely by construction.**

    They are computed by different code paths over different predicates: raw from
    the error total, accepted from the rows' contract adjustments. This case makes
    them legitimately differ and asserts the difference.

    NOTE (REV-101 round 2): this exercises the REPORTING layer, which is where
    the two counts are combined. The SCORER cannot currently emit a non-empty
    `contract_tolerance` at all -- see
    :func:`test_a5_no_row_shape_can_be_contract_adjusted_under_the_current_gold`.
    So the report layer keeps them separable; today's equality is structural.
    """

    rows = [
        {"paper_id": "P", "mode": "strict", "pointer": "/e/0", "name": "a",
         "kind": "false_real_identifier", "identifiers": {}, "contract_tolerance": "",
         "accepted": True},
        {"paper_id": "P", "mode": "strict", "pointer": "/e/1", "name": "Unknown",
         "kind": "false_real_identifier", "identifiers": {},
         "contract_tolerance": "pathbank_unknown_sentinel", "accepted": False},
    ]
    entry = _p1(_report_with(rows))
    assert entry["raw"] == 2
    assert entry["accepted"] == 1
    assert entry["raw"] != entry["accepted"], "raw and accepted agreed by construction"
    assert len(entry["contract_adjusted_rows"]) == 1
    assert entry["contract_adjusted_rows"][0]["contract_tolerance"] == "pathbank_unknown_sentinel"
    # A5: complete row composition for BOTH results.
    assert len(entry["raw_rows"]) == 2
    assert len(entry["accepted_rows"]) == 1


def test_a5_row_composition_is_reported_for_both_results():
    rows = [
        {"paper_id": "P", "mode": "strict", "pointer": "/e/0", "name": "a",
         "kind": "placeholder_claims_real_identity", "identifiers": {"uniprot": "P0A6T1"},
         "contract_tolerance": "", "accepted": True}
    ]
    entry = _p1(_report_with(rows))
    for key in ("raw_rows", "accepted_rows", "contract_adjusted_rows"):
        assert key in entry
    assert entry["raw_rows"][0]["identifiers"] == {"uniprot": "P0A6T1"}
    assert "variance_note" in entry and "target" in entry


def test_a5_a_forged_identity_can_never_be_contract_adjusted(lipid_a):
    """The safety guard, exercised rather than asserted about.

    A row that claims a real accession is never excused, whatever tolerance the
    gold declares. This is the merge-rule-6 boundary of Ruling B.
    """

    forged = sentinel_row()
    forged["mapped_ids"] = {"uniprot": "P0A6T1", "pathbank_protein_id": 9659}
    forged["uniprot_id"] = "P0A6T1"
    report = validate_semantic_coverage(lipid_a, payload(proteins=[forged]), mode="strict")
    p1 = [
        f
        for f in report.checks[CHECK_ID_CONFLICT].findings
        if f["kind"] in ("false_real_identifier", "placeholder_claims_real_identity")
    ]
    assert p1, "the forged row must produce a Priority-1 finding"
    assert all(not f.get("contract_tolerance") for f in p1), (
        "a forged identity was contract-adjusted out of the accepted count"
    )


def _seam_reachable_case(lipid_a):
    """The ONLY configuration in which `_contract_adjustment` is ever CALLED.

    Round 2's tests scored rows that structurally cannot reach the seam, so their
    assertions were vacuous and both survived a maximal mutation of the guard.
    The cause: `_contract_adjustment`'s single call site
    (``semantic.py`` `false_real_identifier` branch) is entered only for a row
    whose name the gold FORBIDS **and** which carries at least one external id.
    `Unknown` is not in PMC12444477's `forbidden_identifiers`, so no sentinel row
    ever got there and the scorer emitted nothing to assert on.

    This synthetic case adds `Unknown` to `forbidden_identifiers` and changes
    nothing else, so a sentinel-shaped row carrying an accession reaches the
    branch and the seam is genuinely exercised. Same `dataclasses.replace` idiom
    the file already uses in `test_a3_*` and `test_a7_5`.

    SYNTHETIC AND TEST-ONLY. The shipped gold is untouched; this exists so the
    bareness guard is attacked rather than described.
    """

    import dataclasses

    return dataclasses.replace(
        lipid_a,
        forbidden_identifiers=lipid_a.forbidden_identifiers
        + (
            ForbiddenIdentifier(
                name="Unknown",
                kind="heading_or_prose",
                reason="SYNTHETIC, test-only: routes a sentinel row into the "
                "false_real_identifier branch so the bareness guard is exercised.",
            ),
        ),
    )


def _p1_findings(case, row):
    report = validate_semantic_coverage(case, payload(proteins=[row]), mode="strict")
    return [
        f
        for f in report.checks[CHECK_ID_CONFLICT].findings
        if f["kind"] in ("false_real_identifier", "placeholder_claims_real_identity")
    ]


def test_a5_bare_means_bare_a_sentinel_with_any_accession_is_not_adjusted(lipid_a):
    """REV-101 correction 3. D-074 condition 5 licenses only the BARE sentinel.

    Round 0's guard matched UniProt-SHAPED strings only, so a sentinel carrying a
    kegg / chebi / drugbank / hmdb / pubchem id WAS contract-adjusted. This test
    is what stops that coming back.

    Round 3: it now scores through `_seam_reachable_case`, so the seam is
    actually entered. Its two previous forms asserted the predicate and inferred
    the rest (round 1), then called the scorer on rows that produced no finding
    at all so `all([])` passed vacuously (round 2). Both survived deleting the
    guard outright. This form goes red under that mutation.
    """

    from t2pw.bench.semantic import _external_ids

    case = _seam_reachable_case(lipid_a)
    bare = sentinel_row()
    assert _external_ids(bare) == {}, "the genuine sentinel must still read as bare"
    assert case.sentinel_tolerance_match("Unknown", bare) is not None

    for namespace, value in (
        ("kegg", "K00912"),
        ("chebi", "CHEBI:16856"),
        ("hmdb", "HMDB0000122"),
        ("drugbank", "DB00114"),
        ("pubchem", "5793"),
    ):
        row = sentinel_row()
        row["mapped_ids"][namespace] = value
        assert _external_ids(row), f"{namespace} must survive _external_ids"
        # The row still satisfies the sentinel predicate -- it is the BARENESS
        # guard, not the predicate, that must refuse it.
        assert case.sentinel_tolerance_match("Unknown", row) is not None, namespace

        findings = _p1_findings(case, row)
        forged = [f for f in findings if f["kind"] == "false_real_identifier"]
        # NON-VACUITY: the seam's only call site must actually have been reached.
        assert forged, (
            f"{namespace} produced no false_real_identifier finding, so "
            "_contract_adjustment was never called and this assertion proves nothing"
        )
        assert all(f.get("contract_tolerance", "") == "" for f in forged), (
            f"a sentinel carrying a {namespace} id was contract-adjusted; "
            "D-074 condition 5 licenses only the BARE sentinel"
        )


def test_a5_no_row_shape_can_be_contract_adjusted_under_the_current_gold(lipid_a):
    """REV-101 round 2/3. The seam is UNREACHABLE, and that is pinned deliberately.

    `_contract_adjustment`'s only call site sits inside `if ids:`, and its
    bareness guard refuses any row with `ids`. D-074 licenses only the bare
    sentinel, which therefore can never BE a Priority-1 row. So accepted == raw
    today by CONSTRUCTION, not by measurement.

    Scored through `_seam_reachable_case` so the seam is entered: under the
    SHIPPED gold no sentinel row reaches it at all, which made round 2's version
    of this test count a `placeholder_claims_real_identity` finding -- whose
    `contract_tolerance` is a hard-coded `""` literal that never consults the
    seam -- as evidence of exercise. It was not.
    """

    case = _seam_reachable_case(lipid_a)
    forged = sentinel_row()
    forged["mapped_ids"] = {"uniprot": "P0A6T1", "pathbank_protein_id": 9659}
    forged["uniprot_id"] = "P0A6T1"
    shapes = [
        sentinel_row(),                                   # bare: no finding at all
        forged,                                           # forged accession
        sentinel_row(name="LpxA product"),                # a differently forbidden name
        {"name": "Unknown", "identity_status": "placeholder",
         "mapping_meta": {"identity_status": "placeholder", "fallback_used": True}},
        wrapper_row("LpxH"),
    ]
    for namespace, value in (("kegg", "K00912"), ("chebi", "CHEBI:16856")):
        row = sentinel_row()
        row["mapped_ids"][namespace] = value
        shapes.append(row)

    exercised = 0
    for row in shapes:
        for f in _p1_findings(case, row):
            # ONLY this kind calls the seam. `placeholder_claims_real_identity`
            # carries a hard-coded "" and can never exercise it, so counting it
            # as evidence of exercise is the substitution that made the previous
            # guard pass while the seam went untouched.
            if f["kind"] == "false_real_identifier":
                exercised += 1
            assert f.get("contract_tolerance", "") == "", (
                f"{f['name']} was contract-adjusted; the seam is meant to be "
                "unreachable under D-074 as ruled"
            )
    # Exactly three of the shapes carry BOTH a forbidden name and an external id,
    # which is what the branch requires: the forged-uniprot row, sentinel+kegg and
    # sentinel+chebi. The bare sentinel, the impostor and the LpxH wrapper carry no
    # id, and `LpxA product` carries none either once `_external_ids` drops the
    # sentinel's own `uniprot: Unknown`. Pinned as an equality, not a floor, so
    # adding or removing a shape has to be noticed here.
    assert exercised == 3, (
        f"{exercised} false_real_identifier finding(s) reached the seam, expected 3; "
        "this test proves nothing about the guard unless it is actually called"
    )


def test_a7_12b_non_vacuity_status_collapse_turns_the_band_tests_red():
    """A7.12/13 for Ruling B: collapsing the band to a Boolean loses the fact."""

    from t2pw.bench.acceptance import PRIORITY1_FAIL, priority1_status

    collapsed = {n: (priority1_status(n) != PRIORITY1_FAIL) for n in (6, 7, 8)}
    assert collapsed == {6: True, 7: True, 8: False}
    # Under the collapse, 6 and 7 are indistinguishable -- which is exactly what
    # D-073 forbids, and what the three-valued status preserves.
    assert collapsed[6] == collapsed[7]
    assert priority1_status(6) != priority1_status(7)


# ---------------------------------------------------------------------------
# Whole-run invariant on the real pinned corpus.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not PINNED_LEG.exists(), reason="pinned run not in this checkout")
def test_pinned_run_reproduces_the_sixteen_five_partition(gold):
    """D-070's measured partition, reproduced by the instrument that reports it."""

    pinned = REPO / "runs/2026-08-02_2130/papers"
    backed = sentinels = wrappers = other = 0
    for leg in sorted(pinned.glob("*/*/final_mapped.json")):
        paper_id, mode = leg.parent.parent.name, leg.parent.name
        case = gold.get(paper_id)
        if case is None:
            continue
        report = validate_semantic_coverage(case, _load(leg), mode=mode)
        c = report.identity_census
        backed += report.scientific_errors[ERR_PLACEHOLDER_BACKED_PROTEINS]
        sentinels += c["placeholder_sentinel_rows"]
        wrappers += c["placeholder_generated_wrappers"]
        other += c["placeholder_other_rows"]
    assert (backed, sentinels, wrappers, other) == (21, 5, 16, 0)
