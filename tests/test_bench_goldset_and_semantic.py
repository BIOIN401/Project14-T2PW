"""The gold set loads honestly and the semantic validator catches what it claims to.

Why this file exists
--------------------
The benchmark's only job is to be harder to fool than the export gate. Every test
here pins a way the benchmark itself could be fooled:

* a malformed or anchor-less case must be *refused at load*, not silently dropped
  -- a gold set that loses a case reports a better score than it earned;
* a check that cannot be evaluated (no admission report on disk) must NOT count
  as a pass;
* an identity audit run against a *pre-mapping* payload must not read as clean,
  because that payload has no accessions in it to be wrong;
* the connectivity model must not link two reactions through ATP, and must not
  link them through a shared enzyme.

Offline and deterministic: no network, no LLM, no database. The bench package
imports ``t2pw.pipeline.entity_identity`` and ``t2pw.rag.synthesize`` lazily and
defensively, so nothing here needs the heavy stack.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.goldset import (  # noqa: E402
    EXPORT_PARTIAL,
    EXPORT_STRICT,
    ForbiddenIdentifier,
    GoldCase,
    GoldSetError,
    GoldTerm,
    MATCH_ALIAS,
    MATCH_CONTAINS,
    MATCH_EXACT,
    MATCH_NONE,
    RELEVANCE_CONTEXT_ONLY,
    RELEVANCE_CORE,
    contains_term,
    load_gold_set,
    normalize_name,
    pinned_gold_set_path,
)
from t2pw.bench.metrics import (  # noqa: E402
    BOUNDARY_EXTRACTION,
    BOUNDARY_PWML_REQUIRED_FIELD,
    BOUNDARY_STAGE3_GATE,
    BOUNDARY_TIMEOUT,
    RESEARCH_FAIL_AMBIGUITY,
    RESEARCH_FAIL_DEFECT,
    RESEARCH_FAIL_PROVIDER,
    RESEARCH_FAIL_RELAXED_GATE,
    RESEARCH_FAIL_STRUCTURAL,
    RESEARCH_FAIL_TIMEOUT,
    Denominator,
    classify_research_failure,
    classify_strict_boundary,
    rank_blockers,
)
from t2pw.bench.semantic import (  # noqa: E402
    CHECK_ANCHORS,
    CHECK_CONNECTED_CORE,
    CHECK_SOURCE_CARRIER,
    CHECK_SUPPORTED_REACTIONS,
    CHECK_ID_CONFLICT,
    CHECK_ORGANISM,
    CHECK_PLACEHOLDER_IDENTITY,
    CHECK_RAG_REINTRODUCTION,
    ERR_CROSS_ORGANISM_REACTIONS,
    ERR_FALSE_REAL_IDENTIFIERS,
    ERR_ORPHANED_REFERENCES,
    ERR_PLACEHOLDER_BACKED_PROTEINS,
    ERR_MISSING_SOURCE_CARRIER,
    ERR_UNSUPPORTED_REACTIONS,
    validate_semantic_coverage,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _case(**overrides) -> GoldCase:
    base = dict(
        paper_id="PMCTEST",
        title="a test paper",
        requested_pathway="menaquinone biosynthesis",
        requested_organism="Bacillus subtilis",
        mechanistic_relevance=RELEVANCE_CORE,
        expected_pathway_anchors=(GoldTerm(name="menaquinone biosynthesis"),),
        min_connected_reactions=1,
        expected_export=EXPORT_STRICT,
    )
    base.update(overrides)
    return GoldCase(**base)


def _payload(reactions, *, compounds=(), proteins=()) -> dict:
    return {
        "entities": {
            "compounds": [dict(c) for c in compounds],
            "proteins": [dict(p) for p in proteins],
            "protein_complexes": [],
        },
        "processes": {"reactions": [dict(r) for r in reactions], "transports": [], "interactions": []},
    }


def _declare(payload: dict) -> dict:
    """Declare every participant so referential integrity is clean by default."""

    names = set()
    for row in payload["processes"]["reactions"]:
        names.update(row.get("inputs") or [])
        names.update(row.get("outputs") or [])
    declared = {c["name"] for c in payload["entities"]["compounds"]}
    declared |= {p["name"] for p in payload["entities"]["proteins"]}
    for name in sorted(names - declared):
        payload["entities"]["compounds"].append({"name": name})
    return payload


# ---------------------------------------------------------------------------
# Normalization and matching.
# ---------------------------------------------------------------------------
def test_greek_letters_are_spelled_out_not_stripped():
    # Stripping would collapse alpha- and beta-ketoglutarate onto one molecule.
    assert normalize_name("β-hydroxyacyl-ACP") == normalize_name("beta-hydroxyacyl-ACP")
    assert normalize_name("α-ketoglutarate") != normalize_name("β-ketoglutarate")


def test_unicode_dashes_and_charge_notation_normalize():
    assert normalize_name("2,3–dihydroxybenzoate") == normalize_name("2,3-dihydroxybenzoate")
    assert normalize_name("NAD+") == normalize_name("NAD")


def test_digits_are_identity_bearing():
    # MK-7 and MK-4 are different molecules and must never compare equal.
    assert normalize_name("MK-7") != normalize_name("MK-4")


def test_containment_respects_token_boundaries():
    assert contains_term("heme biosynthesis pathway", "heme")
    # Without word boundaries "heme" matches inside "hemeprotein", and the anchor
    # check then passes on a payload that never mentions free heme.
    assert not contains_term("hemeprotein complex", "heme")
    assert not contains_term("porphobilinogenase", "porphobilinogen")


def test_short_terms_never_match_by_containment():
    # A 3-character term like LSS or ALA would otherwise match half the payload,
    # so it is reachable only by exact or alias match. This is why the gold set
    # spells such enzymes out and carries the abbreviation as an alias.
    assert not contains_term("lanosterol synthase", "LSS")
    assert GoldTerm(name="LSS").match("LSS") == MATCH_EXACT
    assert GoldTerm(name="lanosterol synthase", aliases=("LSS",)).match("LSS") == MATCH_ALIAS


def test_match_rules_are_reported():
    term = GoldTerm(name="lanosterol synthase", aliases=("LSS",))
    assert term.match("lanosterol synthase") == MATCH_EXACT
    assert term.match("LSS") == MATCH_ALIAS
    assert term.match("mitochondrial lanosterol synthase") == MATCH_CONTAINS
    assert term.match("squalene epoxidase") == MATCH_NONE


# ---------------------------------------------------------------------------
# Gold-set loading.
# ---------------------------------------------------------------------------
def _write_gold(tmp_path: Path, cases: list) -> Path:
    target = tmp_path / "gold.json"
    target.write_text(json.dumps({"version": "test", "cases": cases}), encoding="utf-8")
    return target


def test_case_without_anchors_is_refused(tmp_path):
    path = _write_gold(tmp_path, [{"paper_id": "P1", "requested_pathway": "x"}])
    with pytest.raises(GoldSetError, match="expected_pathway_anchor"):
        load_gold_set(path)


def test_negative_control_may_omit_anchors_but_must_be_context_only(tmp_path):
    ok = _write_gold(
        tmp_path,
        [
            {
                "paper_id": "P1",
                "requested_pathway": "x",
                "mechanistic_relevance": "context_only",
                "min_connected_reactions": 0,
                "max_retained_reactions": 0,
            }
        ],
    )
    gold = load_gold_set(ok)
    assert gold.cases[0].is_negative_control

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "paper_id": "P1",
                        "requested_pathway": "x",
                        "mechanistic_relevance": "core",
                        "min_connected_reactions": 0,
                        "max_retained_reactions": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GoldSetError, match="negative control"):
        load_gold_set(bad)


def test_unsatisfiable_bounds_are_refused(tmp_path):
    path = _write_gold(
        tmp_path,
        [
            {
                "paper_id": "P1",
                "requested_pathway": "x",
                "expected_pathway_anchors": ["a term"],
                "min_connected_reactions": 5,
                "max_retained_reactions": 2,
            }
        ],
    )
    with pytest.raises(GoldSetError, match="no result could satisfy both"):
        load_gold_set(path)


def test_duplicate_paper_ids_are_refused(tmp_path):
    entry = {"paper_id": "P1", "requested_pathway": "x", "expected_pathway_anchors": ["a term"]}
    path = _write_gold(tmp_path, [entry, dict(entry)])
    with pytest.raises(GoldSetError, match="duplicate paper_id"):
        load_gold_set(path)


def test_forbidden_match_is_exact_not_substring():
    case = _case(
        forbidden_identifiers=(ForbiddenIdentifier(name="coenzyme A", kind="cofactor_as_protein"),)
    )
    assert case.forbidden_match("coenzyme A") is not None
    # "coenzyme A ligase" is a real enzyme and must not be condemned by the
    # cofactor rule.
    assert case.forbidden_match("coenzyme A ligase") is None


# ---------------------------------------------------------------------------
# The shipped gold set.
# ---------------------------------------------------------------------------
def test_shipped_gold_set_loads_and_is_usable():
    gold = load_gold_set()
    assert len(gold) >= 6, "the pinned set is meant to hold roughly 6-10 papers"
    mechanistic = [c for c in gold if not c.is_negative_control]
    assert len(mechanistic) >= 6
    assert gold.strict_expected_ids, "at least one case must be strict-exportable or the rate has no denominator"
    for case in gold:
        assert case.requested_pathway
        assert case.requested_organism
        if not case.is_negative_control:
            assert case.expected_pathway_anchors
            # Every expectation must be traceable to the paper.
            assert any(term.quote for term in case.expected_pathway_anchors), case.paper_id


def test_shipped_gold_set_topics_lines_carry_the_requested_scope():
    """A bare pinned id fetches the paper with an EMPTY request.

    ``runs/2026-08-01_1724`` proved this is silent: ten papers fetched, nothing
    skipped, a clean funnel, and every plan row carrying ``requested_pathway: ""``.
    The generated topics file must therefore emit the scoped form, and
    ``parse_topics`` must round-trip it back to the gold case exactly.
    """
    from t2pw.batch.fetch import parse_topics

    gold = load_gold_set()
    body = [line for line in gold.topics_lines() if line and not line.startswith("#")]
    assert len(body) == len(gold)

    specs = parse_topics("\n".join(gold.topics_lines()))
    assert len(specs) == len(gold)
    for spec, case in zip(specs, gold.cases):
        assert spec.is_pinned, "a benchmark line must never become a literature search"
        assert spec.pinned_id == case.paper_id
        assert spec.topic == case.requested_pathway
        assert spec.organism == case.requested_organism
        scope = spec.requested_scope()
        assert scope.pinned is True
        assert scope.requested_pathway == case.requested_pathway
        assert scope.requested_organism == case.requested_organism


def test_pinned_gold_set_path_points_at_a_real_file():
    assert pinned_gold_set_path().is_file()


# ---------------------------------------------------------------------------
# Semantic checks.
# ---------------------------------------------------------------------------
def test_missing_anchor_fails_and_is_counted():
    case = _case(expected_pathway_anchors=(GoldTerm(name="menaquinone", quote="q"),))
    payload = _declare(_payload([{"name": "r1", "inputs": ["glucose"], "outputs": ["pyruvate"], "evidence": "e"}]))
    report = validate_semantic_coverage(case, payload)
    assert not report.checks[CHECK_ANCHORS].ok
    assert report.scientific_errors["missing_pathway_anchors"] == 1


def test_reaction_without_any_source_field_fails_the_carrier_check():
    case = _case()
    payload = _declare(
        _payload(
            [
                {"name": "menaquinone biosynthesis step", "inputs": ["a"], "outputs": ["b"], "evidence": "quoted"},
                {"name": "invented", "inputs": ["b"], "outputs": ["c"]},
            ]
        )
    )
    report = validate_semantic_coverage(case, payload)
    assert not report.checks[CHECK_SOURCE_CARRIER].ok
    assert report.scientific_errors[ERR_MISSING_SOURCE_CARRIER] == 1


def test_a_plain_evidence_string_counts_as_a_carrier_but_not_as_support():
    # A non-RAG run carries a bare `evidence` string and no rag_provenance.
    # Treating that as unsupported would report every such run as fabricated.
    case = _case()
    payload = _declare(
        _payload([{"name": "menaquinone biosynthesis", "inputs": ["a"], "outputs": ["b"], "evidence": "the paper says"}])
    )
    report = validate_semantic_coverage(case, payload)
    assert report.checks[CHECK_SOURCE_CARRIER].ok
    assert report.scientific_errors[ERR_MISSING_SOURCE_CARRIER] == 0


def test_forbidden_organism_on_a_reaction_is_a_cross_organism_error():
    case = _case(forbidden_organisms=("Listeria monocytogenes",))
    payload = _declare(
        _payload(
            [
                {
                    "name": "menaquinone biosynthesis",
                    "inputs": ["a"],
                    "outputs": ["b"],
                    "evidence": "e",
                    "organism": "Listeria monocytogenes",
                }
            ]
        )
    )
    report = validate_semantic_coverage(case, payload)
    assert not report.checks[CHECK_ORGANISM].ok
    assert report.scientific_errors[ERR_CROSS_ORGANISM_REACTIONS] == 1


def test_blank_organism_is_not_a_cross_organism_error():
    # The payload routinely leaves organism blank until propagation runs; counting
    # blanks would report a violation for every early-stage payload.
    case = _case(forbidden_organisms=("Listeria monocytogenes",))
    payload = _declare(_payload([{"name": "menaquinone biosynthesis", "inputs": ["a"], "outputs": ["b"], "evidence": "e"}]))
    report = validate_semantic_coverage(case, payload)
    assert report.checks[CHECK_ORGANISM].ok


def test_forbidden_identifier_carrying_a_real_accession_is_a_false_real_identifier():
    case = _case(
        forbidden_identifiers=(
            ForbiddenIdentifier(name="coenzyme A", kind="cofactor_as_protein", reason="a cofactor, not a protein"),
        )
    )
    payload = _payload(
        [{"name": "menaquinone biosynthesis", "inputs": ["a"], "outputs": ["b"], "evidence": "e"}],
        proteins=[{"name": "coenzyme A", "uniprot": "P12345"}],
    )
    _declare(payload)
    report = validate_semantic_coverage(case, payload)
    assert not report.checks[CHECK_ID_CONFLICT].ok
    assert report.scientific_errors[ERR_FALSE_REAL_IDENTIFIERS] == 1


def test_one_accession_claimed_by_two_entities_is_a_conflict():
    case = _case()
    payload = _payload(
        [{"name": "menaquinone biosynthesis", "inputs": ["a"], "outputs": ["b"], "evidence": "e"}],
        proteins=[{"name": "MenD", "uniprot": "P80867"}, {"name": "MenF", "uniprot": "P80867"}],
    )
    _declare(payload)
    report = validate_semantic_coverage(case, payload)
    check = report.checks[CHECK_ID_CONFLICT]
    assert not check.ok
    assert any(f.get("kind") == "accession_claimed_by_multiple_entities" for f in check.findings)


def test_undeclared_participant_is_an_orphaned_reference():
    case = _case()
    payload = _payload(
        [{"name": "menaquinone biosynthesis", "inputs": ["chorismate"], "outputs": ["isochorismate"], "evidence": "e"}],
        compounds=[{"name": "chorismate"}],
    )
    report = validate_semantic_coverage(case, payload)
    assert report.scientific_errors[ERR_ORPHANED_REFERENCES] == 1


def test_cofactors_do_not_connect_two_unrelated_reactions():
    # Without the cofactor exclusion, any bag of reactions is "one pathway"
    # because nearly all of them touch ATP.
    case = _case(min_connected_reactions=2)
    payload = _declare(
        _payload(
            [
                {"name": "menaquinone biosynthesis a", "inputs": ["chorismate", "ATP"], "outputs": ["isochorismate", "ADP"], "evidence": "e"},
                {"name": "unrelated", "inputs": ["glucose", "ATP"], "outputs": ["glucose-6-phosphate", "ADP"], "evidence": "e"},
            ]
        )
    )
    report = validate_semantic_coverage(case, payload)
    graph = report.graph
    if graph.get("cofactors_excluded"):
        assert graph["largest_core_size"] == 1
        assert not report.checks[CHECK_CONNECTED_CORE].ok


def test_a_shared_enzyme_does_not_connect_two_reactions():
    case = _case(min_connected_reactions=2)
    payload = _declare(
        _payload(
            [
                {
                    "name": "menaquinone biosynthesis a",
                    "inputs": ["x1"],
                    "outputs": ["x2"],
                    "evidence": "e",
                    "enzymes": [{"protein": "PromiscuousKinase"}],
                },
                {
                    "name": "menaquinone biosynthesis b",
                    "inputs": ["y1"],
                    "outputs": ["y2"],
                    "evidence": "e",
                    "enzymes": [{"protein": "PromiscuousKinase"}],
                },
            ]
        )
    )
    payload["entities"]["proteins"].append({"name": "PromiscuousKinase"})
    report = validate_semantic_coverage(case, payload)
    assert report.graph["largest_core_size"] == 1
    assert not report.checks[CHECK_CONNECTED_CORE].ok


def test_a_shared_metabolite_does_connect_two_reactions():
    case = _case(min_connected_reactions=2)
    payload = _declare(
        _payload(
            [
                {"name": "menaquinone biosynthesis a", "inputs": ["chorismate"], "outputs": ["isochorismate"], "evidence": "e"},
                {"name": "menaquinone biosynthesis b", "inputs": ["isochorismate"], "outputs": ["SEPHCHC"], "evidence": "e"},
            ]
        )
    )
    report = validate_semantic_coverage(case, payload)
    assert report.graph["largest_core_size"] == 2
    assert report.checks[CHECK_CONNECTED_CORE].ok


def test_negative_control_fails_when_any_reaction_is_retained():
    case = _case(
        mechanistic_relevance=RELEVANCE_CONTEXT_ONLY,
        expected_pathway_anchors=(),
        min_connected_reactions=0,
        max_retained_reactions=0,
        expected_export=EXPORT_PARTIAL,
    )
    empty = _payload([])
    assert validate_semantic_coverage(case, empty).checks[CHECK_CONNECTED_CORE].ok

    invented = _declare(_payload([{"name": "hallucinated", "inputs": ["a"], "outputs": ["b"], "evidence": "quoted"}]))
    report = validate_semantic_coverage(case, invented)
    assert not report.checks[CHECK_CONNECTED_CORE].ok
    # A well-quoted reaction is still unsupported when the paper has no pathway.
    assert report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 1


def test_a_nonzero_ceiling_only_charges_the_excess():
    # PMC12180156 states two reactions, neither of them heme biosynthesis. Two
    # retained is faithful; the third onward is invention, and only the excess
    # should be charged as unsupported.
    case = _case(
        mechanistic_relevance=RELEVANCE_CONTEXT_ONLY,
        expected_pathway_anchors=(GoldTerm(name="menaquinone biosynthesis"),),
        min_connected_reactions=0,
        max_retained_reactions=2,
        expected_export=EXPORT_PARTIAL,
    )
    rows = [
        {"name": f"menaquinone biosynthesis {i}", "inputs": [f"m{i}"], "outputs": [f"m{i + 1}"], "evidence": "quoted"}
        for i in range(5)
    ]
    report = validate_semantic_coverage(case, _declare(_payload(rows)))
    assert report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 3
    assert not report.checks[CHECK_CONNECTED_CORE].ok

    ok_rows = rows[:2]
    ok_report = validate_semantic_coverage(case, _declare(_payload(ok_rows)))
    assert ok_report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 0
    assert ok_report.checks[CHECK_CONNECTED_CORE].ok


# ---------------------------------------------------------------------------
# Supported-reaction signatures: the check that actually measures support.
# ---------------------------------------------------------------------------
def _sig_case(**overrides):
    from t2pw.bench.goldset import SupportedReaction

    signature = SupportedReaction(
        inputs=(GoldTerm(name="chorismate"),),
        outputs=(GoldTerm(name="isochorismate"),),
        enzyme=GoldTerm(name="MenF"),
        quote="MenF catalyzes the conversion of chorismate to isochorismate",
        label="MenF step",
    )
    base = dict(supported_reactions=(signature,))
    base.update(overrides)
    return _case(**base), signature


PAPER = "In this organism MenF catalyzes the conversion of chorismate to isochorismate, the first step."


def test_support_is_inapplicable_without_the_stored_paper_text():
    # An unverifiable quote is an assertion. Missing text must never pass.
    case, signature = _sig_case()
    payload = _declare(_payload([{"name": "menaquinone biosynthesis", "inputs": ["chorismate"],
                                  "outputs": ["isochorismate"], "evidence": "q"}]))
    report = validate_semantic_coverage(case, payload, paper_text=None)
    check = report.checks[CHECK_SUPPORTED_REACTIONS]
    assert not check.applicable
    assert not report.confirmed


def test_a_gold_quote_absent_from_the_paper_is_reported_as_a_gold_defect():
    case, _sig = _sig_case()
    payload = _declare(_payload([{"name": "menaquinone biosynthesis", "inputs": ["chorismate"],
                                  "outputs": ["isochorismate"], "evidence": "q"}]))
    report = validate_semantic_coverage(case, payload, paper_text="an unrelated paper about kinetics")
    check = report.checks[CHECK_SUPPORTED_REACTIONS]
    assert not check.ok
    assert any(f["kind"] == "gold_quote_not_found_in_paper" for f in check.findings)
    # The signature is excluded from scoring rather than blamed on the pipeline.
    assert report.support["signatures_quote_verified"] == 0


def test_quote_verification_survives_the_spacing_artifacts_in_cached_text():
    from t2pw.bench.goldset import fold_for_quote

    # The cached full text inserts spaces inside tokens where italic tags were
    # stripped: "lipid IV A", "R -3-hydroxymyristoyl-ACP", "sigma 32".
    assert fold_for_quote("phosphorylated by LpxK to produce lipid IVA") == fold_for_quote(
        "phosphorylated by LpxK to produce lipid IV A"
    )


def test_a_matching_reaction_is_a_true_positive_and_a_missing_one_is_a_false_negative():
    case, _sig = _sig_case()
    hit = _declare(_payload([{"name": "menaquinone biosynthesis", "inputs": ["chorismate"],
                              "outputs": ["isochorismate"], "evidence": "q",
                              "enzymes": [{"protein": "MenF"}]}]))
    hit["entities"]["proteins"].append({"name": "MenF"})
    report = validate_semantic_coverage(case, hit, paper_text=PAPER)
    assert report.support["true_positives"] == 1
    assert report.support["false_negatives"] == 0
    assert report.support["recall"] == 1.0
    assert report.checks[CHECK_SUPPORTED_REACTIONS].ok

    miss = _declare(_payload([{"name": "menaquinone biosynthesis", "inputs": ["glucose"],
                               "outputs": ["pyruvate"], "evidence": "q"}]))
    missed = validate_semantic_coverage(case, miss, paper_text=PAPER)
    assert missed.support["false_negatives"] == 1
    assert missed.support["recall"] == 0.0
    assert missed.scientific_errors["missing_supported_reactions"] == 1
    assert not missed.checks[CHECK_SUPPORTED_REACTIONS].ok


def test_a_subset_signature_list_does_not_call_unmatched_rows_fabrications():
    """The default. RAG legitimately adds reactions from OTHER papers, which can
    never match a seed-paper signature, so counting them as hallucinations
    reported 227 fabricated reactions in a run that produced far fewer."""
    case, _sig = _sig_case()
    payload = _declare(
        _payload(
            [
                {"name": "menaquinone biosynthesis", "inputs": ["chorismate"], "outputs": ["isochorismate"], "evidence": "q"},
                {"name": "from another paper", "inputs": ["isochorismate"], "outputs": ["SEPHCHC"], "evidence": "q"},
            ]
        )
    )
    report = validate_semantic_coverage(case, payload, paper_text=PAPER)
    assert report.support["signature_set_complete"] is False
    assert report.support["unattributed_reactions"] == 1
    assert report.support["precision"] is None, "precision is only defined for an exhaustive set"
    assert report.support["attribution_rate"] is not None
    assert report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 0
    assert report.checks[CHECK_SUPPORTED_REACTIONS].ok


def test_an_exhaustive_signature_list_does_charge_unmatched_rows():
    case, signature = _sig_case(supported_reactions_complete=True)
    payload = _declare(
        _payload(
            [
                {"name": "menaquinone biosynthesis", "inputs": ["chorismate"], "outputs": ["isochorismate"], "evidence": "q"},
                {"name": "invented", "inputs": ["x"], "outputs": ["y"], "evidence": "q"},
            ]
        )
    )
    report = validate_semantic_coverage(case, payload, paper_text=PAPER)
    assert report.support["signature_set_complete"] is True
    assert report.support["precision"] == 0.5
    assert report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 1
    assert not report.checks[CHECK_SUPPORTED_REACTIONS].ok


def test_every_shipped_gold_quote_is_present_in_a_cached_copy_of_its_paper():
    """The gold set must not assert a quote the papers do not contain."""
    import glob
    from t2pw.bench.goldset import fold_for_quote

    gold = load_gold_set()
    missing = []
    for case in gold:
        if not case.supported_reactions:
            continue
        best = ""
        for path in glob.glob(str(ROOT / "runs" / "*" / "papers" / f"{case.paper_id}*" / "01_source_text.txt")):
            try:
                text = Path(path).read_text(encoding="utf-8")
            except OSError:
                continue
            if len(text) > len(best):
                best = text
        if not best:
            continue  # no cached copy on this machine; nothing to verify against
        folded = fold_for_quote(best)
        for signature in case.supported_reactions:
            if fold_for_quote(signature.quote) not in folded:
                missing.append(f"{case.paper_id}: {signature.describe()}")
    assert not missing, "gold quotes not found in the cached paper text: " + "; ".join(missing)


def test_rag_reintroduction_is_inapplicable_without_an_admission_report():
    case = _case()
    payload = _declare(_payload([{"name": "menaquinone biosynthesis", "inputs": ["a"], "outputs": ["b"], "evidence": "e"}]))
    report = validate_semantic_coverage(case, payload)
    check = report.checks[CHECK_RAG_REINTRODUCTION]
    assert not check.applicable
    # Inapplicable is not a pass: `ok` may be True but `confirmed` must not be.
    assert not report.confirmed


def test_a_rejected_rag_claim_that_reappears_is_caught():
    case = _case()
    payload = _declare(
        _payload(
            [
                {
                    "name": "renamed by synthesis",
                    "inputs": ["chorismate"],
                    "outputs": ["isochorismate"],
                    "evidence": "e",
                    "enzymes": [{"protein": "MenF"}],
                }
            ]
        )
    )
    payload["entities"]["proteins"].append({"name": "MenF"})
    admission = {
        "rejected": [
            {
                "name": "some other label",
                "inputs": ["isochorismate"],
                "outputs": ["chorismate"],
                "enzymes": ["MenF"],
                "reasons": ["organism_mismatch"],
                "gap_id": "g1",
            }
        ]
    }
    # Same claim, opposite listed order, different display name: the key is
    # order-independent within a slot but direction-sensitive across slots, so
    # this must NOT match.
    report = validate_semantic_coverage(case, payload, admission=admission)
    assert report.checks[CHECK_RAG_REINTRODUCTION].ok

    admission["rejected"][0]["inputs"] = ["chorismate"]
    admission["rejected"][0]["outputs"] = ["isochorismate"]
    report = validate_semantic_coverage(case, payload, admission=admission)
    check = report.checks[CHECK_RAG_REINTRODUCTION]
    assert not check.ok
    assert check.findings[0]["rejected_for"] == ["organism_mismatch"]


def test_placeholder_backed_protein_is_counted_and_gated_by_the_case():
    placeholder = {
        "name": "Unknown",
        "uniprot": "Unknown",
        "pathbank_protein_id": 9659,
        "mapping_meta": {
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "identity_status": "placeholder",
            "cross_species_placeholder": True,
            # A well-formed placeholder must ADMIT it is one. Omitting this is
            # itself a forgery rule (`placeholder_does_not_record_fallback_used`).
            "fallback_used": True,
            "resolution": {"status": "fallback", "issue": "pathbank_unknown_sentinel_component"},
        },
    }
    payload = _payload(
        [{"name": "menaquinone biosynthesis", "inputs": ["a"], "outputs": ["b"], "evidence": "e"}],
        proteins=[placeholder],
    )
    _declare(payload)

    strict_case = _case(unknown_backed_proteins_acceptable=False)
    strict_report = validate_semantic_coverage(strict_case, payload)
    assert strict_report.scientific_errors[ERR_PLACEHOLDER_BACKED_PROTEINS] == 1
    assert not strict_report.checks[CHECK_PLACEHOLDER_IDENTITY].ok

    lenient_case = _case(unknown_backed_proteins_acceptable=True)
    lenient_report = validate_semantic_coverage(lenient_case, payload)
    # Still counted -- the count is a fact about the pathway, not a verdict --
    # but no longer a check failure.
    assert lenient_report.scientific_errors[ERR_PLACEHOLDER_BACKED_PROTEINS] == 1
    assert lenient_report.checks[CHECK_PLACEHOLDER_IDENTITY].ok


def test_a_placeholder_forging_an_accession_always_fails():
    forger = {
        "name": "Unknown",
        "uniprot": "P0A6Q6",
        "pathbank_protein_id": 9659,
        "mapping_meta": {
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "identity_status": "placeholder",
            "cross_species_placeholder": True,
            # A well-formed placeholder must ADMIT it is one. Omitting this is
            # itself a forgery rule (`placeholder_does_not_record_fallback_used`).
            "fallback_used": True,
            "resolution": {"status": "fallback", "issue": "pathbank_unknown_sentinel_component"},
        },
    }
    payload = _payload(
        [{"name": "menaquinone biosynthesis", "inputs": ["a"], "outputs": ["b"], "evidence": "e"}],
        proteins=[forger],
    )
    _declare(payload)
    # Even a case that tolerates Unknown-backed proteins must refuse one that is
    # posing as a real mapping.
    report = validate_semantic_coverage(_case(unknown_backed_proteins_acceptable=True), payload)
    assert not report.checks[CHECK_PLACEHOLDER_IDENTITY].ok
    assert report.scientific_errors[ERR_FALSE_REAL_IDENTIFIERS] >= 1


def test_a_placeholder_that_does_not_admit_being_one_is_a_forgery():
    # The sentinel shape without `fallback_used` looks, to every downstream
    # reader, like a row that simply happens to be named "Unknown".
    silent = {
        "name": "Unknown",
        "uniprot": "Unknown",
        "pathbank_protein_id": 9659,
        "mapping_meta": {
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "identity_status": "placeholder",
            "cross_species_placeholder": True,
        },
    }
    payload = _payload(
        [{"name": "menaquinone biosynthesis", "inputs": ["a"], "outputs": ["b"], "evidence": "e"}],
        proteins=[silent],
    )
    _declare(payload)
    report = validate_semantic_coverage(_case(unknown_backed_proteins_acceptable=True), payload)
    check = report.checks[CHECK_PLACEHOLDER_IDENTITY]
    assert not check.ok
    assert check.findings[0]["rule"] == "placeholder_does_not_record_fallback_used"


def test_no_payload_is_not_evaluated_rather_than_failed():
    report = validate_semantic_coverage(_case(), None)
    assert not report.evaluated
    assert not report.ok
    assert report.error_total() == 0


# ---------------------------------------------------------------------------
# Failure taxonomies.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "row, expected",
    [
        ({"status": "timeout", "failure_kind": "timeout"}, RESEARCH_FAIL_TIMEOUT),
        ({"status": "fail", "failure_kind": "llm"}, RESEARCH_FAIL_PROVIDER),
        ({"status": "fail", "failure_kind": "network"}, RESEARCH_FAIL_PROVIDER),
        ({"status": "fail", "failure_kind": "ambiguous_review_scope"}, RESEARCH_FAIL_AMBIGUITY),
        ({"status": "fail", "failure_kind": "contract", "issue_codes": ["processes_required"]}, RESEARCH_FAIL_STRUCTURAL),
        ({"status": "fail", "failure_kind": "no_reactions"}, RESEARCH_FAIL_STRUCTURAL),
        (
            {"status": "fail", "failure_kind": "contract", "stage": "post_pipeline", "issue_codes": ["gate.protein_x"]},
            RESEARCH_FAIL_RELAXED_GATE,
        ),
        ({"status": "fail", "failure_kind": "crash", "stage": "research_report"}, RESEARCH_FAIL_DEFECT),
    ],
)
def test_research_failures_are_attributed_to_the_right_cause(row, expected):
    category, _evidence = classify_research_failure(row)
    assert category == expected


def test_a_structural_guard_is_not_filed_as_a_research_defect():
    # The old doctrine said any research failure means the code is wrong. The
    # structural guards abort in BOTH modes by design.
    row = {"status": "fail", "failure_kind": "contract", "stage": "stage1", "issue_codes": ["entities_required"]}
    category, evidence = classify_research_failure(row)
    assert category == RESEARCH_FAIL_STRUCTURAL
    assert "entities_required" in evidence


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"status": "fail", "stage": "stage1", "failure_kind": "contract"}, BOUNDARY_EXTRACTION),
        (
            {"status": "fail", "stage": "post_pipeline", "failure_kind": "contract", "issue_codes": ["gate.protein_x"]},
            BOUNDARY_STAGE3_GATE,
        ),
        (
            {"status": "fail", "stage": "pwml_export", "failure_kind": "contract", "issue_codes": ["protein_missing_external_identity"]},
            BOUNDARY_PWML_REQUIRED_FIELD,
        ),
        ({"status": "timeout", "failure_kind": "timeout"}, BOUNDARY_TIMEOUT),
    ],
)
def test_strict_boundaries_separate_what_failure_kind_conflated(row, expected):
    boundary, _evidence = classify_strict_boundary(row)
    assert boundary == expected


def test_blockers_rank_by_papers_released_not_by_occurrence_count():
    # `noisy` touches more papers but never alone, so fixing it releases nobody.
    # `lonely` blocks one paper by itself and must rank first.
    failures = [
        {"paper_id": "A", "boundary": BOUNDARY_STAGE3_GATE, "issue_codes": ["gate.protein_a_is_missing_species"]},
        {"paper_id": "B", "boundary": BOUNDARY_STAGE3_GATE, "issue_codes": ["gate.protein_b_is_missing_species", "gate.x_degree_after_normalization"]},
        {"paper_id": "C", "boundary": BOUNDARY_STAGE3_GATE, "issue_codes": ["gate.protein_c_is_missing_species", "gate.y_degree_after_normalization"]},
    ]
    ranked = rank_blockers(failures)
    assert ranked[0].root_cause == "protein_organism_unset"
    assert ranked[0].sole_blocker_for == {"A"}
    assert ranked[0].expected_benefit == 1
    disconnected = next(b for b in ranked if b.root_cause == "protein_disconnected_after_normalization")
    assert disconnected.expected_benefit == 0
    assert len(disconnected.papers) == 2


def test_a_run_that_attempted_nothing_is_not_reported_as_acceptable(tmp_path):
    # The trap: no legs means no scientific errors, so acceptance priorities 1-3
    # all hold and the report reads as a clean bill of health.
    from t2pw.bench.acceptance import score_run

    run_dir = tmp_path / "2026-01-01_0000"
    (run_dir / "papers").mkdir(parents=True)
    (run_dir / "manifest.jsonl").write_text("", encoding="utf-8")

    report = score_run(run_dir, load_gold_set())
    assert report.legs_attempted == 0
    assert report.legs_scored == 0
    assert all(entry["ok"] for entry in report.priorities() if entry["rank"] <= 3)
    assert any("NO LEG WAS ATTEMPTED" in note for note in report.notes)


def _stage_run(tmp_path: Path, paper_id: str, slug: str, payload: dict, *, filename: str) -> Path:
    """A minimal run directory with one strict leg, as the batch driver writes it."""

    run_dir = tmp_path / "2026-01-01_0000"
    leg = run_dir / "papers" / slug / "strict"
    leg.mkdir(parents=True)
    (leg / filename).write_text(json.dumps(payload), encoding="utf-8")
    row = {
        "paper_id": paper_id,
        "slug": slug,
        "mode": "strict",
        "status": "fail",
        "stage": "post_pipeline",
        "failure_kind": "contract",
        "issue_codes": ["gate.protein_x_is_missing_a_unipro"],
        "files": [],
    }
    (run_dir / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return run_dir


def test_the_scorer_prefers_the_mapped_payload_and_records_which_it_used(tmp_path):
    """End to end: a forged placeholder in final_mapped.json must be reported.

    This is the whole reason the driver now writes ``final_mapped.json`` on the
    common path. Scored against ``merged_payload.json`` the same run reports zero
    false identifiers -- not because there are none, but because a pre-mapping
    payload carries no accession that could be wrong.
    """
    from t2pw.bench.acceptance import score_run

    gold = load_gold_set()
    case = gold.by_id("PMC12444477")
    assert case is not None and case.forbidden_match("coenzyme A") is not None

    forged = {
        "entities": {
            "compounds": [{"name": "UDP-GlcNAc"}, {"name": "lipid IVA"}],
            "proteins": [
                {
                    "name": "coenzyme A",
                    "uniprot": "P0A6Q6",
                    "mapping_meta": {"identity_status": "verified"},
                }
            ],
            "protein_complexes": [],
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

    # Short slug on purpose: the real one plus a pytest tmp_path exceeds the
    # Windows 260-character path limit, and _find_slug reads it from the manifest.
    slug = "PMC12444477__lipid-a"
    mapped_run = _stage_run(tmp_path / "m1", "PMC12444477", slug, forged, filename="final_mapped.json")
    report = score_run(mapped_run, gold)
    leg = report.papers[0].leg("strict")
    assert leg is not None and leg.payload_source == "final_mapped.json"
    assert report.errors.totals[ERR_FALSE_REAL_IDENTIFIERS] == 1
    assert not report.priorities()[0]["ok"]

    merged_run = _stage_run(tmp_path / "m2", "PMC12444477", slug, forged, filename="merged_payload.json")
    merged_report = score_run(merged_run, gold)
    merged_leg = merged_report.papers[0].leg("strict")
    assert merged_leg is not None and merged_leg.payload_source == "merged_payload.json"
    # Same forgery, and it is still caught -- but the report must say out loud
    # that it was scored from a pre-mapping file, or a clean run and an
    # unmeasurable one look identical.
    assert any("pre-mapping" in note for note in merged_report.notes)


def test_a_truncated_issue_code_still_finds_its_root_cause():
    # driver._slug_code truncates at 68 chars and the entity name comes FIRST, so
    # a long protein name eats the end of the diagnosis. Matching on the full
    # phrase routed every long-named protein into "other" and mis-ranked the
    # fix-list.
    from t2pw.bench.metrics import root_cause

    full = "gate.protein_coenzyme_a_coa_is_missing_a_uniprot_or_drugbank_identifi"
    truncated = "gate.protein_hydroxyacyl_acyl_carrier_protein_acp_is_missing_a_unipro"
    assert root_cause(full)[0] == "protein_identity_unresolved"
    assert root_cause(truncated)[0] == "protein_identity_unresolved"

    truncated_species = "gate.protein_hydroxyacyl_acyl_carrier_protein_acp_is_missing_species_"
    assert root_cause(truncated_species)[0] == "protein_organism_unset"


def test_an_empty_population_reports_n_a_not_zero_percent():
    denominator = Denominator(key="k", question="q", population="p")
    assert denominator.rate is None
    assert "n/a" in denominator.render()
