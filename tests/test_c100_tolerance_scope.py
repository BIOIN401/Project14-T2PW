"""PMC12444477's Unknown tolerance is per-entity, and the scorer enforces it.

Why this file exists
--------------------
``unknown_backed_proteins_acceptable`` is a case-WIDE Boolean.
``pinned_v1.json`` sets it ``true`` for PMC12444477 and for no other case, and the
``unknown_backed_rationale`` beside it names both the entities the tolerance is *for*
and, in the same sentence, the class it must **not** cover. The scorer read only the
Boolean, so the rationale was parsed, round-tripped and never enforced: the unscoped
``true`` excused the very core Raetz enzymes the rationale excludes.

D-071 rules that the Boolean is **scoped, not flipped** -- flipping it would penalise
faithful extraction of the seven entities the rationale legitimately tolerates, which is
the opposite error rather than a fix. These tests pin both halves of that ruling.

Measured, not assumed (``04-tip-probe`` / ``03-base-probe``): exactly **one** of the ten
pinned cases sets the Boolean ``true``; the archived leg
``runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json`` carries **nine**
placeholder rows -- ``Unknown``, ``LpxA``, ``LpxD``, ``LpxH``, ``LpxB``, ``LpxK``,
``WaaA``, ``LpxL``, ``LpxM`` -- every one of which the unscoped Boolean excused.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.goldset import (  # noqa: E402
    GoldCase,
    GoldSetError,
    load_gold_set,
    pinned_gold_set_path,
)
from t2pw.bench.semantic import (  # noqa: E402
    CHECK_PLACEHOLDER_IDENTITY,
    ERR_FALSE_REAL_IDENTIFIERS,
    ERR_PLACEHOLDER_BACKED_PROTEINS,
    validate_semantic_coverage,
)

PAPER = "PMC12444477"
FINDING = "unknown_backed_protein_not_acceptable"

#: The entities PMC12444477's rationale names as tolerated, by their canonical name.
TOLERATED = ("LapA", "LapB", "Ght", "LabP", "LpxG", "YhcB", "lipoprotein")

#: The seven core Raetz enzymes MEASURED as excused by the unscoped Boolean
#: (D-071, ``evidence/g11/ORCH-710/05``; re-measured here as C-100's ``03-base-probe``).
SEVEN_CORE = ("LpxA", "LpxD", "LpxB", "LpxK", "WaaA", "LpxL", "LpxM")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _pinned_case() -> GoldCase:
    case = load_gold_set().by_id(PAPER)
    assert case is not None, f"{PAPER} is missing from the pinned gold set"
    return case


def _placeholder(name: str, **overrides: Any) -> Dict[str, Any]:
    """A correctly-formed Unknown-backed row carrying ``name``.

    This is the shape ``map_ids`` writes for a single-protein PathWhiz wrapper it
    could not resolve: it ADMITS it is a placeholder (``fallback_used``), ships no
    real accession, and therefore returns ``""`` from
    ``placeholder_claims_real_identity``. Anything else would be caught by the
    Priority-1 forgery arm before the tolerance arm is ever reached, and these tests
    would be measuring the wrong branch.
    """

    row: Dict[str, Any] = {
        "name": name,
        "uniprot": "Unknown",
        "pathbank_protein_id": 9659,
        "mapping_meta": {
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "identity_status": "placeholder",
            "cross_species_placeholder": True,
            "fallback_used": True,
            "resolution": {"status": "fallback", "issue": "pathbank_unknown_sentinel_component"},
        },
    }
    row.update(overrides)
    return row


def _payload(*rows: Dict[str, Any]) -> Dict[str, Any]:
    """A minimal lipid-A payload carrying ``rows`` as protein complexes.

    The rows go in ``protein_complexes`` because that is where the measured
    wrappers live (D-070: 16 generated wrappers there, 5 sentinels in ``proteins``).
    """

    return {
        "entities": {
            "compounds": [{"name": "UDP-GlcNAc"}, {"name": "lipid IVA"}],
            "proteins": [],
            "protein_complexes": [dict(r) for r in rows],
        },
        "processes": {
            "reactions": [
                {
                    "name": "lipid A biosynthesis step",
                    "inputs": ["UDP-GlcNAc"],
                    "outputs": ["lipid IVA"],
                    "evidence": "quoted from the paper",
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _tolerance_findings(case: GoldCase, name: str) -> List[Dict[str, Any]]:
    """The tolerance findings the SCORER raises for one Unknown-backed row.

    Deliberately routed through ``validate_semantic_coverage`` rather than through
    the matcher: the defect was that the gold's statement never reached the scorer,
    so a schema-only assertion would not have caught it and does not prove it fixed.
    Only ``CHECK_PLACEHOLDER_IDENTITY`` is read -- the pinned case also demands
    anchors and three connected reactions, which this stub payload does not supply,
    and asserting on ``report.ok`` would measure those instead.
    """

    report = validate_semantic_coverage(case, _payload(_placeholder(name)))
    check = report.checks[CHECK_PLACEHOLDER_IDENTITY]
    return [f for f in check.findings if f.get("kind") == FINDING]


# ---------------------------------------------------------------------------
# 1-2. What the tolerance is FOR.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", TOLERATED)
def test_1_each_tolerated_entity_is_excused_by_name(name: str) -> None:
    """The half of D-071 that flipping the Boolean to false would have broken."""

    case = _pinned_case()
    assert case.unknown_backed_tolerance_match(name) is not None
    assert case.tolerates_unknown_backed(name) is True
    assert _tolerance_findings(case, name) == []
    # Still COUNTED. The count is a fact about the pathway, not a verdict (TRAP-3).
    report = validate_semantic_coverage(case, _payload(_placeholder(name)))
    assert report.scientific_errors[ERR_PLACEHOLDER_BACKED_PROTEINS] == 1


@pytest.mark.parametrize("alias, canonical", [("YciS", "LapA"), ("YciM", "LapB")])
def test_2_lapa_and_lapb_are_excused_through_their_former_names(
        alias: str, canonical: str) -> None:
    """LapA/YciS and LapB/YciM are one protein under two names, validated AS ALIASES.

    D-071 requires this specifically: not fuzzily, and not by containment -- the two
    names share no substring at all, so only a declared alias can connect them.
    """

    case = _pinned_case()
    matched = case.unknown_backed_tolerance_match(alias)
    assert matched is not None and matched.name == canonical
    assert _tolerance_findings(case, alias) == []


# ---------------------------------------------------------------------------
# 3-4. What the tolerance is NOT for. THIS IS THE G9 PROOF.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enzyme", SEVEN_CORE)
def test_3_the_seven_core_raetz_enzymes_are_not_excused(enzyme: str) -> None:
    """G9 PROOF -- this fails behaviourally on the base SHA.

    It names no symbol C-100 introduced: ``load_gold_set``,
    ``validate_semantic_coverage`` and ``CHECK_PLACEHOLDER_IDENTITY`` all exist at
    base. At base the unscoped ``true`` excuses these seven rows, the findings list
    is empty and this assertion fails on BEHAVIOUR, not on an AttributeError.
    Symbol absence would not be proof.
    """

    findings = _tolerance_findings(_pinned_case(), enzyme)
    assert len(findings) == 1, (
        f"{enzyme} is an expected core Raetz enzyme; PMC12444477's rationale "
        "explicitly excludes it from the Unknown-backed tolerance"
    )
    assert findings[0]["name"] == enzyme
    # The reason must say WHICH refusal this is: a scoped case does not hold the
    # "every protein must resolve" expectation that an untolerant case holds.
    assert "names one by one" in findings[0]["reason"]


def test_4_lpxc_is_not_excused_either_though_it_was_not_among_the_measured_seven() -> None:
    """LpxC is the eighth expected enzyme and is equally outside the tolerance.

    It is absent from the measured seven for a reason that has nothing to do with
    tolerance: in the archived leg
    ``runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json`` LpxC has **no
    protein or protein_complex row at all** -- it appears only inside reactions and
    interactions (``"protein": "LpxC"``, ``"entity_2": "LpxC"``). A row that does not
    exist cannot be excused, so it could never show up in a census of rows the
    Boolean excused. That is an absence in one run's payload, not a licence, and the
    moment a run does emit an Unknown-backed LpxC the scope must refuse it.

    This also guards the arithmetic D-071 settled: EIGHT expected enzymes, NINE
    pathway steps. LpxC is expected; the ninth step's enzyme is organism-dependent
    and therefore acceptable rather than expected.
    """

    case = _pinned_case()
    expected = {t.name for t in case.expected_enzymes}
    assert "LpxC" in expected
    assert expected == {"LpxA", "LpxC", "LpxD", "LpxB", "LpxK", "WaaA", "LpxL", "LpxM"}
    assert len(expected) == 8

    # Scorer first, deliberately: this assertion names no symbol C-100 added, so on
    # the base SHA it fails on BEHAVIOUR rather than on an AttributeError.
    assert len(_tolerance_findings(case, "LpxC")) == 1
    assert case.tolerates_unknown_backed("LpxC") is False


# ---------------------------------------------------------------------------
# 5. No broad matching.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "candidate",
    [
        "LpxA product",      # a forbidden placeholder name that CONTAINS an enzyme
        "LpxG product",      # ... and one that contains a TOLERATED name
        "pre-LpxG",
        "LapB-like protein",
        "Lap",               # a substring of a tolerated name
        "Ght-family regulator",
        "lipoprotein sorting complex",
        "YhcB homolog",
    ],
)
def test_5_containment_never_excuses_anything(candidate: str) -> None:
    """Exact and alias matches only -- the ``forbidden_match`` rule, and its reason.

    A tolerance is a licence to leave a row unresolved, so it is spelled out rather
    than inferred. Containment would run both ways and both are wrong: outward,
    ``LpxG`` would excuse every longer token containing it; inward, ``LpxA product``
    -- which the same case lists as a FORBIDDEN placeholder name -- would be excused
    by the enzyme it is named after.
    """

    case = _pinned_case()
    # Scorer first: base-SHA failure is behavioural, not symbol absence.
    assert len(_tolerance_findings(case, candidate)) == 1
    assert case.unknown_backed_tolerance_match(candidate) is None
    assert case.tolerates_unknown_backed(candidate) is False


def test_5b_the_bare_pathbank_sentinel_name_is_not_a_tolerated_entity() -> None:
    """``Unknown`` is not one of the seven, and the scope does not invent an eighth.

    Measured: the same archived leg carries a PathBank sentinel row literally named
    ``Unknown`` (``is_pathbank_unknown_protein`` requires that name). D-070 rules the
    sentinel is PathBank's own record and not a forged identity -- and that ruling is
    about forgery and species, not about coverage. Pinned here so the consequence is
    visible rather than discovered later: after C-100 the sentinel produces a
    coverage finding on this one case.
    """

    case = _pinned_case()
    assert len(_tolerance_findings(case, "Unknown")) == 1
    assert case.tolerates_unknown_backed("Unknown") is False


# ---------------------------------------------------------------------------
# 6-7. Backward compatibility: the blast radius really is one case.
# ---------------------------------------------------------------------------
def test_6_boolean_true_with_no_scope_keeps_case_wide_tolerance() -> None:
    """Every gold case written before this field existed grades exactly as it did."""

    unscoped = replace(
        _pinned_case(),
        paper_id="PMCCOMPAT",
        unknown_backed_tolerated_entities=(),
    )
    assert unscoped.unknown_backed_proteins_acceptable is True
    for name in SEVEN_CORE + TOLERATED + ("Unknown", "anything at all"):
        assert unscoped.tolerates_unknown_backed(name) is True, name
    assert _tolerance_findings(unscoped, "LpxA") == []


def test_7_a_case_with_the_boolean_false_is_unaffected() -> None:
    """The Boolean gates everything: no scope can widen a case that tolerates none.

    Measured on the shipped gold rather than asserted: nine of the ten pinned cases
    set the Boolean ``false`` and none of them declares a scope, so C-100's blast
    radius is exactly one case.
    """

    gold = load_gold_set()
    tolerant = [c.paper_id for c in gold if c.unknown_backed_proteins_acceptable]
    assert tolerant == [PAPER], tolerant
    for case in gold:
        if case.paper_id == PAPER:
            continue
        assert case.unknown_backed_tolerated_entities == ()
        assert case.tolerates_unknown_backed("LapA") is False
        assert case.tolerates_unknown_backed("anything") is False

    strict = replace(_pinned_case(), paper_id="PMCSTRICT",
                     unknown_backed_proteins_acceptable=False,
                     unknown_backed_tolerated_entities=())
    assert len(_tolerance_findings(strict, "LapA")) == 1
    assert "expects every protein to resolve" in _tolerance_findings(strict, "LapA")[0]["reason"]


@pytest.mark.parametrize("name", ["LpxA", "LapA"])
def test_7c_an_entity_absent_from_a_scope_inherits_the_case_boolean(name: str) -> None:
    """THE ABSENT BRANCH, on a case whose case-wide Boolean is ``false``.

    A scope tested only with entities that are IN it never exercises the absent
    branch at all. This case does: ``LpxA`` is deliberately left out of the scope,
    and it must stay strict. ``LapA`` is the harder half of the same question -- it
    IS in the scope, and it must ALSO stay strict, because a scope may only narrow
    a tolerant case and may never widen an untolerant one.

    What an unnamed row inherits is the case Boolean, read explicitly. Written as a
    bare ``name in scope`` membership test this would be untolerant only by luck,
    and any form that consulted the scope before the Boolean would override the nine
    pinned cases that set it explicitly ``false`` -- overriding an explicit value,
    not filling an absent one.
    """

    strict_with_scope = replace(
        _pinned_case(),
        paper_id="PMCSCOPEDFALSE",
        unknown_backed_proteins_acceptable=False,
    )
    assert strict_with_scope.unknown_backed_tolerated_entities != ()
    assert strict_with_scope.tolerates_unknown_backed(name) is False
    findings = _tolerance_findings(strict_with_scope, name)
    assert len(findings) == 1
    assert "expects every protein to resolve" in findings[0]["reason"]


def test_7d_an_entity_absent_from_a_scope_on_a_TOLERANT_case_is_still_refused() -> None:
    """The other half: a declared scope is an exhaustive enumeration.

    A name a scope does not list has been left out deliberately, which is a
    different fact from no scope existing -- so it is refused rather than granted
    the case-wide tolerance. Reading an absent name as the inherited Boolean here
    would make every scope decoration, since a scope can only exist on a case whose
    Boolean is already ``True``.
    """

    case = _pinned_case()
    assert case.unknown_backed_proteins_acceptable is True
    assert "MsbA" not in {t.name for t in case.unknown_backed_tolerated_entities}
    assert case.tolerates_unknown_backed("MsbA") is False
    assert len(_tolerance_findings(case, "MsbA")) == 1


def test_7b_an_unreachable_scope_is_refused_at_load_time(tmp_path: Path) -> None:
    """A scope under a ``false`` Boolean can never be read, so it is not accepted.

    The same failure mode, one field over, that let the rationale sit in the file for
    a month meaning nothing.
    """

    import json

    raw = json.loads(pinned_gold_set_path().read_text(encoding="utf-8"))
    for case in raw["cases"]:
        if case["paper_id"] == PAPER:
            case["unknown_backed_proteins_acceptable"] = False
    target = tmp_path / "unreachable.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(GoldSetError, match="can never be reached"):
        load_gold_set(target)


# ---------------------------------------------------------------------------
# 8. Round trip.
# ---------------------------------------------------------------------------
def test_8_the_tolerance_list_round_trips_unchanged(tmp_path: Path) -> None:
    """A field that round-trips asymmetrically is how the rationale became dead prose.

    ``to_dict`` -> JSON -> parser -> ``to_dict`` must be a fixed point, names,
    aliases, quotes, identifiers and roles included.
    """

    import json

    original = _pinned_case()
    assert [t.name for t in original.unknown_backed_tolerated_entities] == list(TOLERATED)

    emitted = original.to_dict()
    assert "unknown_backed_tolerated_entities" in emitted
    assert [t["name"] for t in emitted["unknown_backed_tolerated_entities"]] == list(TOLERATED)

    source = json.loads(pinned_gold_set_path().read_text(encoding="utf-8"))
    target = tmp_path / "round_trip.json"
    target.write_text(json.dumps(source), encoding="utf-8")
    reloaded = load_gold_set(target).by_id(PAPER)
    assert reloaded is not None
    assert reloaded.unknown_backed_tolerated_entities == original.unknown_backed_tolerated_entities
    assert reloaded.to_dict()["unknown_backed_tolerated_entities"] == \
        emitted["unknown_backed_tolerated_entities"]

    # Every entry carries its evidence, and the one that cannot says so in `role`
    # rather than carrying a quote the paper does not contain.
    for term in original.unknown_backed_tolerated_entities:
        if term.name == "lipoprotein":
            assert term.quote == ""
            assert "does not occur" in term.role
        else:
            assert term.quote.strip(), term.name


# ---------------------------------------------------------------------------
# 9. LpxG holds two jobs at once.
# ---------------------------------------------------------------------------
def test_9_lpxg_is_both_an_alias_of_acceptable_lpxh_and_a_tolerated_entity() -> None:
    """The case a Boolean cannot express and a per-entity list can (D-071).

    And the distinction that makes the scope non-trivial: ``LpxG`` -- the
    organism-dependent enzyme this E. coli-scoped request cannot resolve -- is
    tolerated, while ``LpxH`` -- the E. coli enzyme for the same ninth step -- is
    not. Measured: the archived leg emits an Unknown-backed ``LpxH`` wrapper, and
    after C-100 that row is a finding.
    """

    case = _pinned_case()

    lpxh = next(t for t in case.acceptable_enzymes if t.name == "LpxH")
    assert "LpxG" in lpxh.aliases and "LpxI" in lpxh.aliases
    assert lpxh.match("LpxG") == "alias"

    # Scorer first: base-SHA failure is behavioural, not symbol absence.
    assert len(_tolerance_findings(case, "LpxH")) == 1
    assert len(_tolerance_findings(case, "LpxI")) == 1
    assert _tolerance_findings(case, "LpxG") == []

    tolerated = case.unknown_backed_tolerance_match("LpxG")
    assert tolerated is not None and tolerated.name == "LpxG"
    # Neither fact leaks into the other: being an alias of an acceptable enzyme is
    # not itself a tolerance.
    assert case.tolerates_unknown_backed("LpxH") is False
    assert case.tolerates_unknown_backed("LpxI") is False


# ---------------------------------------------------------------------------
# 10. Priority 1 does not move.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["LapA", "LpxG", "lipoprotein"])
def test_10_a_forged_identity_is_still_priority_1_however_tolerated_the_name(
        name: str) -> None:
    """Tolerance is about coverage; forgery is about safety. They do not trade.

    ``placeholder_claims_real_identity`` stays AHEAD of the tolerance branch, so a
    placeholder shipping a real accession fails even under a name the case tolerates
    -- and it fails as a forgery, never as a coverage finding.
    """

    case = _pinned_case()
    forger = _placeholder(name, uniprot="P0A6Q6")
    report = validate_semantic_coverage(case, _payload(forger))
    check = report.checks[CHECK_PLACEHOLDER_IDENTITY]

    assert not check.ok
    assert report.scientific_errors[ERR_FALSE_REAL_IDENTIFIERS] >= 1
    kinds = {f.get("kind") for f in check.findings}
    assert "placeholder_not_distinguished" in kinds
    assert FINDING not in kinds


def test_10b_the_summary_discloses_acceptance_that_actually_happened() -> None:
    """The acceptance sentence is keyed on what happened, not on the case Boolean.

    ``False`` and "tolerated by scope instead" are not the same fact. A summary
    keyed on the case-wide Boolean cannot tell them apart, so it would stop
    disclosing acceptance for exactly the cases a per-entity scope creates -- a row
    would be tolerated and the report would not say so. That is under-reporting: it
    fails quietly instead of contradicting an adjacent number, which makes it the
    harder of the two to notice.
    """

    case = _pinned_case()
    report = validate_semantic_coverage(
        case, _payload(_placeholder("LapA"), _placeholder("YciM")))
    check = report.checks[CHECK_PLACEHOLDER_IDENTITY]

    assert check.ok
    assert "accepts 2 of them" in check.summary
    assert "named by the case" in check.summary

    # And when nothing was tolerated the sentence must not appear at all.
    refused = validate_semantic_coverage(case, _payload(_placeholder("LpxA")))
    assert "this case accepts" not in refused.checks[CHECK_PLACEHOLDER_IDENTITY].summary


# ---------------------------------------------------------------------------
# 11. Non-vacuity.
# ---------------------------------------------------------------------------
def test_11_removing_the_scope_makes_tests_3_and_4_go_red() -> None:
    """The control. Without it, tests 3 and 4 could be passing for any other reason.

    Same case, same payloads, same assertions -- only
    ``unknown_backed_tolerated_entities`` removed. Every finding tests 3 and 4 rely
    on disappears, which is exactly the base-SHA behaviour and exactly what makes
    those tests non-vacuous.
    """

    scoped = _pinned_case()
    unscoped = replace(scoped, unknown_backed_tolerated_entities=())

    for enzyme in SEVEN_CORE + ("LpxC",):
        assert len(_tolerance_findings(scoped, enzyme)) == 1, enzyme      # test 3 / 4 green
        assert _tolerance_findings(unscoped, enzyme) == [], enzyme        # ... and red without

    # The tolerated seven are unaffected by removing the scope -- they were excused
    # before and after -- which is why the scope is a NARROWING and not a new gate.
    for name in TOLERATED:
        assert _tolerance_findings(scoped, name) == []
        assert _tolerance_findings(unscoped, name) == []
