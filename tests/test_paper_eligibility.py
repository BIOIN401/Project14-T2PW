"""Tests for ``t2pw.rag.eligibility`` -- the pre-full-text paper gate.

Everything here is offline and deterministic: no network, no LLM, no full-text
fetch. Candidates are mocked and the real 2026-07-28_2122 plan titles are used as
the fixture set, because those are the papers that actually got through the old
unfiltered fetcher and cost two app runs each.

Thresholds are pinned explicitly (``EligibilityThresholds()``) rather than read
from the environment, so a developer's ``.env`` cannot change a verdict here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch.fetch import TopicSpec, fetch_papers  # noqa: E402
from t2pw.batch.report import (  # noqa: E402
    STATUS_INELIGIBLE,
    _is_failure,
    _norm_status,
    build_summary,
    group_papers,
)
from t2pw.config import rag_config  # noqa: E402
from t2pw.rag.acquire import CandidatePaper  # noqa: E402
from t2pw.rag.eligibility import (  # noqa: E402
    ORGANISM_GENUS_LEVEL,
    ORGANISM_MATCH,
    ORGANISM_MISMATCH,
    ORGANISM_UNKNOWN,
    OUTCOME_ELIGIBLE,
    OUTCOME_INELIGIBLE_ORGANISM,
    OUTCOME_INELIGIBLE_PATHWAY,
    OUTCOME_INSUFFICIENT_METADATA,
    OUTCOME_PINNED_OVERRIDE,
    CLASS_CONTEXT_ONLY,
    CLASS_MECHANISTIC,
    CLASS_OMICS_ONLY,
    EligibilityThresholds,
    ObservedContext,
    RequestedScope,
    apply_stage0_observation,
    compare_organism,
    observe_metadata,
    outcome_is_ineligible,
    screen_candidate,
    screen_paper,
    thresholds_from_config,
)

#: Fixed thresholds -- the shipped defaults, stated here so a verdict below can
#: never be moved by an environment variable.
THRESHOLDS = EligibilityThresholds()

# ---------------------------------------------------------------------------
# Real titles from runs/2026-07-28_2122/plan.json. These are verbatim, HTML
# entities included, because handling that noise is part of the job.
# ---------------------------------------------------------------------------
T_FOURNIER = "Ochrobactrum anthropi Causing Fournier's Gangrene: A Report of a Rare Case."
T_POULTRY = (
    "Genotypic Characterization of Virulence Factors in Extended-Spectrum "
    "Beta-Lactamase (ESBL)-Producing &lt;i&gt;Escherichia coli&lt;/i&gt; Strains "
    "from Chickens in Hungary."
)
T_TURKEY = (
    "Virulence Gene Profiles of Extended-Spectrum Beta-Lactamase (ESBL)-Producing "
    "&lt;i&gt;Escherichia coli&lt;/i&gt; Isolated from Turkeys in Hungary: A "
    "Whole-Genome Sequencing Study."
)
T_COVID = (
    "Association of gender and main comorbidities with expression of lncRNAs and "
    "mRNAs in COVID-19 patients"
)
T_ENTB = (
    "The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a "
    "competitive inhibitor of the Escherichia coli isochorismatase EntB."
)
T_MEND_LISTERIA = (
    "Structures of Listeria monocytogenes MenD in ThDP-bound and in-crystallo "
    "captured intermediate I-bound forms."
)
T_HEME_FEEDBACK = "A reversible feedback mechanism regulating mitochondrial heme synthesis."
T_LIPID_A_REVIEW = "The regulation of lipid A biosynthesis."

LIPID_A_ECOLI = RequestedScope(
    requested_pathway="lipid A biosynthesis", requested_organism="Escherichia coli"
)
ENTEROBACTIN_ECOLI = RequestedScope(
    requested_pathway="enterobactin biosynthesis", requested_organism="Escherichia coli"
)
MENAQUINONE_SUBTILIS = RequestedScope(
    requested_pathway="menaquinone biosynthesis", requested_organism="Bacillus subtilis"
)
HEME_HUMAN = RequestedScope(
    requested_pathway="heme biosynthesis", requested_organism="Homo sapiens"
)


def _screen(title: str, scope: RequestedScope, abstract: str = ""):
    return screen_paper(
        paper_id="PMC1",
        title=title,
        abstract=abstract,
        scope=scope,
        thresholds=THRESHOLDS,
    )


# ---------------------------------------------------------------------------
# The required rejections.
# ---------------------------------------------------------------------------
def test_fournier_gangrene_case_report_is_rejected() -> None:
    """A clinical case report is not a mechanistic pathway paper."""
    decision = _screen(T_FOURNIER, LIPID_A_ECOLI)
    assert decision.ineligible
    assert not decision.eligible
    # Two independent grounds, and both are recorded: the host organism is not
    # E. coli, and the paper is a case report.
    assert decision.outcome == OUTCOME_INELIGIBLE_ORGANISM
    assert "Ochrobactrum anthropi" in decision.observed_organisms
    assert decision.organism_match == ORGANISM_MISMATCH
    assert any("clinical_case_report" in term for term in decision.matched_negative)
    # A confident reject: nothing here warrants a human's time.
    assert decision.needs_manual_review is False


def test_case_report_is_rejected_even_when_it_names_the_pathway() -> None:
    """The case-report veto beats a strong positive score, by design."""
    title = (
        "Lipid A biosynthesis inhibitor therapy in Escherichia coli sepsis: a "
        "report of a rare case"
    )
    decision = _screen(title, LIPID_A_ECOLI)
    assert decision.outcome == OUTCOME_INELIGIBLE_PATHWAY
    assert any("clinical_case_report" in t for t in decision.matched_negative)
    # The positive evidence really was there -- it just does not override a veto.
    assert any("pathway_alias" in t for t in decision.matched_positive)


def test_unrelated_poultry_virulence_survey_is_rejected() -> None:
    """Chickens + virulence typing + no pathway anchor = the poultry-survey shape."""
    decision = _screen(T_POULTRY, ENTEROBACTIN_ECOLI)
    assert decision.ineligible
    assert decision.outcome == OUTCOME_INELIGIBLE_PATHWAY
    assert any("animal_virulence_survey" in t for t in decision.matched_negative)
    # The requested organism IS present, so this cannot be an organism rejection:
    # it is rejected for having nothing to do with enterobactin biosynthesis.
    assert decision.organism_match == ORGANISM_MATCH
    assert "Gallus gallus" in decision.observed_organisms


def test_turkey_whole_genome_survey_is_rejected() -> None:
    decision = _screen(T_TURKEY, ENTEROBACTIN_ECOLI)
    assert decision.ineligible
    assert any("animal_virulence_survey" in t for t in decision.matched_negative)


def test_covid_lncrna_paper_is_rejected_for_a_bacterial_pathway_request() -> None:
    """Human/viral clinical association study vs a bacterial pathway request."""
    decision = _screen(T_COVID, LIPID_A_ECOLI)
    assert decision.ineligible
    assert decision.outcome == OUTCOME_INELIGIBLE_ORGANISM
    assert decision.organism_match == ORGANISM_MISMATCH
    assert "Homo sapiens" in decision.observed_organisms
    assert "SARS-CoV-2" in decision.observed_organisms
    assert "Escherichia coli" not in decision.observed_organisms


def test_covid_lncrna_paper_is_still_rejected_for_the_human_request_it_was_found_for() -> None:
    """In the 2122 plan this was fetched for heme/Homo sapiens -- organism matches,
    so it has to be rejected on the epidemiology/association evidence instead."""
    decision = _screen(T_COVID, HEME_HUMAN)
    assert decision.organism_match == ORGANISM_MATCH
    assert decision.outcome == OUTCOME_INELIGIBLE_PATHWAY
    assert any("epidemiology_survey" in t for t in decision.matched_negative)


def test_software_only_paper_is_rejected() -> None:
    title = "PathViz: a user friendly web server for browsing metabolic pathway maps"
    decision = _screen(title, HEME_HUMAN)
    assert decision.ineligible
    assert any("software_only" in t for t in decision.matched_negative)


def test_software_paper_with_wet_lab_validation_is_not_vetoed_as_software() -> None:
    """"Software-only" means only software. Experimental work disqualifies the veto."""
    title = (
        "A web server for heme biosynthesis flux modelling, with in vitro "
        "validation using purified ferrochelatase"
    )
    decision = _screen(title, HEME_HUMAN)
    assert not any("software_only" in t for t in decision.matched_negative)
    assert decision.outcome == OUTCOME_ELIGIBLE


def test_pathway_named_only_in_the_background_is_penalized() -> None:
    """Anchor in the abstract, absent from the title, no mechanism language."""
    decision = _screen(
        "Iron acquisition strategies of uropathogenic isolates",
        ENTEROBACTIN_ECOLI,
        abstract=(
            "Bacteria acquire iron by many routes. Enterobactin biosynthesis is one "
            "such route. Here we survey clinical isolates by growth phenotype."
        ),
    )
    assert any("pathway_only_in_background" in t for t in decision.matched_negative)
    assert decision.ineligible


# ---------------------------------------------------------------------------
# The required acceptance.
# ---------------------------------------------------------------------------
def test_exact_pathway_and_organism_enzyme_paper_is_accepted() -> None:
    """EntB in E. coli, for an enterobactin/E. coli request: the target case."""
    decision = _screen(T_ENTB, ENTEROBACTIN_ECOLI)
    assert decision.eligible
    assert decision.outcome == OUTCOME_ELIGIBLE
    assert decision.organism_match == ORGANISM_MATCH
    assert decision.score >= THRESHOLDS.title_only_min_score
    positives = " ".join(decision.matched_positive)
    assert "pathway_alias:enterobactin biosynthetic" in positives
    assert "pathway_term:entb" in positives
    assert "organism_match:Escherichia coli" in positives
    # Mechanistic language in the title, so no human triage needed even title-only.
    assert decision.needs_manual_review is False


def test_on_topic_mechanism_paper_is_accepted() -> None:
    """"feedback" and "mechanism" are framing words, not reaction evidence.

    So this title alone is ``context_only`` -- the pathway is named and nothing
    local to it is a reaction or an enzyme. One sentence of its real abstract
    (ferrochelatase, the mitochondrion) is enough to anchor it.
    """
    title_only = _screen(T_HEME_FEEDBACK, HEME_HUMAN)
    assert title_only.ineligible
    assert title_only.classification == CLASS_CONTEXT_ONLY
    assert any("pathway_alias:heme synthesis" in t for t in title_only.matched_positive)

    with_abstract = _screen(
        T_HEME_FEEDBACK,
        HEME_HUMAN,
        abstract=(
            "Proper heme biosynthesis is essential for cellular function. We show "
            "that ferrochelatase activity is controlled by a reversible cofactor "
            "intermediate."
        ),
    )
    assert with_abstract.eligible
    assert with_abstract.classification == CLASS_MECHANISTIC


def test_a_pathway_named_in_a_title_with_no_local_mechanism_is_context_only() -> None:
    """"The regulation of lipid A biosynthesis" names the pathway and nothing else.

    On the title alone there is no reaction, enzyme or characterization word local
    to the pathway mention, so it cannot be called mechanistic yet -- it is
    ``context_only``, rejected, and flagged for a human. Its abstract names LpxC and
    the acyltransferase steps, and then it is accepted (see the dry-run tests).
    """
    decision = _screen(T_LIPID_A_REVIEW, LIPID_A_ECOLI)
    assert decision.ineligible
    assert decision.classification == CLASS_CONTEXT_ONLY
    assert decision.provisional is True
    assert decision.needs_manual_review is True
    assert any("pathway_context_only" in t for t in decision.matched_negative)


def test_the_same_paper_is_accepted_once_its_abstract_supplies_local_evidence() -> None:
    """The other half of the test above: local evidence flips it."""
    decision = _screen(
        T_LIPID_A_REVIEW,
        LIPID_A_ECOLI,
        abstract=(
            "LpxC catalyzes the committed step of lipid A biosynthesis in "
            "Escherichia coli, and its substrate flux is controlled by FabZ."
        ),
    )
    assert decision.eligible
    assert decision.classification == CLASS_MECHANISTIC


# ---------------------------------------------------------------------------
# Organism handling: wrong organism, and missing organism.
# ---------------------------------------------------------------------------
def test_right_pathway_wrong_organism_is_ineligible_organism_not_pathway() -> None:
    """MenD is genuine menaquinone biochemistry -- in the wrong organism.

    The distinction matters: ``ineligible_organism`` says "re-run this topic
    against Listeria and it is a good paper", ``ineligible_pathway`` would say
    "this paper is off-topic", which is false.
    """
    decision = _screen(T_MEND_LISTERIA, MENAQUINONE_SUBTILIS)
    assert decision.outcome == OUTCOME_INELIGIBLE_ORGANISM
    assert decision.organism_match == ORGANISM_MISMATCH
    assert decision.observed_organisms == ["Listeria monocytogenes"]
    assert decision.requested_organism == "Bacillus subtilis"
    # The pathway anchor was found: the rejection is purely about the organism.
    assert any("pathway_term:mend" in t for t in decision.matched_positive)
    assert any("incompatible_organism" in t for t in decision.matched_negative)
    assert any("organism mismatch" in w for w in decision.warnings)


def test_vitamin_k2_in_the_wrong_host_is_an_organism_rejection() -> None:
    title = (
        "High-level production of vitamin K2 in &lt;i&gt;Escherichia coli&lt;/i&gt; "
        "via modular molecular engineering."
    )
    decision = _screen(title, MENAQUINONE_SUBTILIS)
    assert decision.outcome == OUTCOME_INELIGIBLE_ORGANISM
    assert "Escherichia coli" in decision.observed_organisms


def test_a_missing_organism_is_unknown_and_is_never_copied_from_the_request() -> None:
    """The core requested-vs-observed guarantee."""
    title = "Enzymatic characterization of the enterobactin synthetase EntF"
    decision = _screen(title, ENTEROBACTIN_ECOLI)
    assert decision.requested_organism == "Escherichia coli"
    # The title names no organism -> nothing observed, and NOT "Escherichia coli".
    assert decision.observed_organisms == []
    assert decision.organism_match == ORGANISM_UNKNOWN
    # Unknown is neutral: it neither rejects the paper nor scores as a match.
    assert not any("organism_match" in t for t in decision.matched_positive)
    assert not any("incompatible_organism" in t for t in decision.matched_negative)
    assert decision.eligible


def test_unknown_organism_is_not_treated_as_a_mismatch() -> None:
    assert (
        compare_organism(ENTEROBACTIN_ECOLI, ObservedContext()) == ORGANISM_UNKNOWN
    )
    assert (
        compare_organism(
            RequestedScope(requested_pathway="x"),
            ObservedContext(observed_organisms=["Homo sapiens"]),
        )
        == ORGANISM_UNKNOWN
    )


def test_any_observed_organism_matching_is_a_match() -> None:
    """A paper that works in E. coli and validates in mice reports both."""
    decision = _screen(
        "Lipid A biosynthesis by LpxC in Escherichia coli, validated in mice",
        LIPID_A_ECOLI,
    )
    assert set(decision.observed_organisms) >= {"Escherichia coli", "Mus musculus"}
    assert decision.organism_match == ORGANISM_MATCH


def test_observe_metadata_never_invents_an_organism() -> None:
    observed = observe_metadata(
        "Structural basis of catalysis", "", LIPID_A_ECOLI
    )
    assert observed.observed_organisms == []
    assert observed.observed_pathways == []


def test_an_organism_absent_from_the_lexicon_is_still_observed_from_the_paper() -> None:
    scope = RequestedScope(
        requested_pathway="caffeine degradation",
        requested_organism="Pseudomonas fluorescens",
    )
    hit = observe_metadata("Caffeine demethylation in Pseudomonas fluorescens", "", scope)
    assert "Pseudomonas fluorescens" in hit.observed_organisms
    miss = observe_metadata("Caffeine demethylation in soil bacteria", "", scope)
    assert miss.observed_organisms == []


# ---------------------------------------------------------------------------
# Pinned papers.
# ---------------------------------------------------------------------------
def test_pinned_paper_bypasses_the_score_but_records_the_mismatch() -> None:
    scope = RequestedScope(
        requested_pathway="lipid A biosynthesis",
        requested_organism="Escherichia coli",
        pinned=True,
    )
    decision = _screen(T_FOURNIER, scope)
    assert decision.outcome == OUTCOME_PINNED_OVERRIDE
    assert decision.eligible is True
    assert decision.ineligible is False
    # Bypassed, not blind: the mismatch is still computed and reported.
    assert decision.organism_match == ORGANISM_MISMATCH
    assert any("organism mismatch" in w for w in decision.warnings)
    assert any("clinical_case_report" in t for t in decision.matched_negative)
    assert "pinned" in decision.reason


def test_pinned_paper_with_no_metadata_at_all_still_passes() -> None:
    """A pinned id has no title until its full text is fetched -- that is fine."""
    scope = RequestedScope(
        requested_pathway="", requested_organism="", pinned=True
    )
    decision = screen_paper(
        paper_id="PMC4412817", title="", scope=scope, thresholds=THRESHOLDS
    )
    assert decision.outcome == OUTCOME_PINNED_OVERRIDE
    assert decision.eligible


# ---------------------------------------------------------------------------
# Insufficient metadata.
# ---------------------------------------------------------------------------
def test_a_too_short_title_with_no_abstract_is_insufficient_metadata() -> None:
    decision = _screen("Erratum", LIPID_A_ECOLI)
    assert decision.outcome == OUTCOME_INSUFFICIENT_METADATA
    assert outcome_is_ineligible(decision.outcome)
    assert "not enough metadata" in decision.reason


def test_a_short_title_with_an_abstract_is_judged_normally() -> None:
    decision = _screen(
        "LpxC catalysis",
        LIPID_A_ECOLI,
        abstract=(
            "We report the kinetics of LpxC, the committed step of lipid A "
            "biosynthesis in Escherichia coli, with purified enzyme."
        ),
    )
    assert decision.outcome == OUTCOME_ELIGIBLE
    assert decision.provisional is False


# ---------------------------------------------------------------------------
# Stage 0 may not clobber the requested scope.
# ---------------------------------------------------------------------------
def test_requested_scope_is_immutable() -> None:
    scope = LIPID_A_ECOLI
    with pytest.raises(Exception):
        scope.requested_pathway = "cholesterol biosynthesis"  # type: ignore[misc]
    with pytest.raises(Exception):
        scope.requested_organism = "Homo sapiens"  # type: ignore[misc]
    assert scope.requested_pathway == "lipid A biosynthesis"
    assert scope.requested_organism == "Escherichia coli"


def test_stage0_populates_observed_context_only() -> None:
    scope, observed, conflicts = apply_stage0_observation(
        LIPID_A_ECOLI,
        {
            "pathway_name": "lipid A biosynthesis",
            "likely_organism": "Escherichia coli",
            "aliases": ["Raetz pathway"],
            "candidate_examples": [{"name": "E. coli K-12"}],
        },
    )
    # The request came back untouched...
    assert scope.requested_pathway == "lipid A biosynthesis"
    assert scope.requested_organism == "Escherichia coli"
    # ...and Stage 0's reading landed in the observed half.
    assert observed.observed_pathways == ["lipid A biosynthesis"]
    assert observed.observed_organisms == ["Escherichia coli"]
    assert observed.aliases == ["Raetz pathway"]
    assert observed.ambiguities == ["E. coli K-12"]
    assert conflicts == []


def test_stage0_cannot_clobber_the_requested_scope_on_a_conflict() -> None:
    """The failure mode this guards: Stage 0 reads the paper, finds cholesterol
    biosynthesis in human cells, and the batch silently starts asking for that."""
    scope, observed, conflicts = apply_stage0_observation(
        LIPID_A_ECOLI,
        {"pathway_name": "cholesterol biosynthesis", "likely_organism": "Homo sapiens"},
    )
    assert scope.requested_pathway == "lipid A biosynthesis"
    assert scope.requested_organism == "Escherichia coli"
    assert scope == LIPID_A_ECOLI
    # The paper's apparent scope is recorded as an observation, not a new request.
    assert observed.observed_pathways == ["cholesterol biosynthesis"]
    assert observed.observed_organisms == ["Homo sapiens"]
    # And the conflict is reported so the caller can mark the paper ineligible /
    # scope-conflicted rather than adopting it.
    assert len(conflicts) == 2
    assert any("cholesterol biosynthesis" in c for c in conflicts)
    assert any("Homo sapiens" in c for c in conflicts)


def test_stage0_observation_merges_into_an_existing_observed_context() -> None:
    _scope, observed, _conflicts = apply_stage0_observation(
        HEME_HUMAN,
        {"pathway_name": "heme biosynthesis", "likely_organism": "Homo sapiens"},
        observed=ObservedContext(observed_organisms=["Mus musculus"]),
    )
    assert observed.observed_organisms == ["Mus musculus", "Homo sapiens"]


def test_stage0_conflict_can_drive_the_decision_without_rewriting_the_request() -> None:
    """End to end: a conflicting Stage-0 read makes the paper ineligible."""
    scope, observed, conflicts = apply_stage0_observation(
        LIPID_A_ECOLI,
        {"pathway_name": "cholesterol biosynthesis", "likely_organism": "Homo sapiens"},
    )
    assert conflicts
    decision = screen_paper(
        paper_id="PMC9",
        title="Cholesterol biosynthesis in human hepatocytes",
        scope=scope,
        thresholds=THRESHOLDS,
        observed=observed,
    )
    assert decision.outcome == OUTCOME_INELIGIBLE_ORGANISM
    assert decision.requested_pathway == "lipid A biosynthesis"


# ---------------------------------------------------------------------------
# Candidate integration: requested vs observed on the object itself.
# ---------------------------------------------------------------------------
def test_screen_candidate_writes_requested_and_observed_separately() -> None:
    candidate = CandidatePaper(
        id="PMC12096016",
        source="europepmc",
        title=T_ENTB,
        requested_organism="Escherichia coli",
    )
    decision = screen_candidate(candidate, ENTEROBACTIN_ECOLI, thresholds=THRESHOLDS)
    assert decision.eligible
    assert candidate.requested_pathway == "enterobactin biosynthesis"
    assert candidate.requested_organism == "Escherichia coli"
    assert candidate.observed_organisms == ["Escherichia coli"]
    assert candidate.organism_match == ORGANISM_MATCH
    # ``organism`` is the paper's own reported organism and nothing filled it in.
    assert candidate.organism == ""


def test_a_candidate_that_reports_no_organism_keeps_observed_empty() -> None:
    candidate = CandidatePaper(
        id="PMC2",
        source="ncbi",
        title="Enzymatic characterization of the enterobactin synthetase EntF",
        requested_organism="Escherichia coli",
    )
    screen_candidate(candidate, ENTEROBACTIN_ECOLI, thresholds=THRESHOLDS)
    assert candidate.observed_organisms == []
    assert candidate.organism_match == ORGANISM_UNKNOWN
    assert candidate.organism == ""


# ---------------------------------------------------------------------------
# Thresholds: configurable and deterministic.
# ---------------------------------------------------------------------------
def test_thresholds_come_from_the_rag_config_and_can_be_overridden() -> None:
    limits = thresholds_from_config(
        {
            "eligibility_min_score": "4.5",
            "eligibility_title_only_min_score": 3.25,
            "eligibility_min_title_chars": "12",
            "eligibility_organism_veto": "false",
            "eligibility_review_margin": "0",
        }
    )
    assert limits.min_score == 4.5
    assert limits.title_only_min_score == 3.25
    assert limits.min_title_chars == 12
    assert limits.organism_veto is False
    assert limits.review_margin == 0.0


def test_config_defaults_are_the_documented_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    import t2pw.config as _config

    monkeypatch.setattr(_config, "_ENV_LOADED", True, raising=False)
    for var in (
        "RAG_ELIGIBILITY_ENABLED",
        "RAG_ELIGIBILITY_MIN_SCORE",
        "RAG_ELIGIBILITY_TITLE_ONLY_MIN_SCORE",
        "RAG_ELIGIBILITY_MIN_TITLE_CHARS",
        "RAG_ELIGIBILITY_REQUIRE_ANCHOR",
        "RAG_ELIGIBILITY_ORGANISM_VETO",
        "RAG_ELIGIBILITY_NEGATIVE_VETO",
        "RAG_ELIGIBILITY_REVIEW_MARGIN",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = rag_config()
    assert cfg["eligibility_enabled"] is True
    assert cfg["eligibility_min_score"] == 2.0
    assert cfg["eligibility_title_only_min_score"] == 1.5
    assert cfg["eligibility_min_title_chars"] == 20
    assert cfg["eligibility_require_pathway_anchor"] is True
    assert cfg["eligibility_organism_veto"] is True
    assert cfg["eligibility_negative_veto"] is True
    assert cfg["eligibility_review_margin"] == 0.5
    assert thresholds_from_config() == EligibilityThresholds()


def test_a_bad_threshold_value_falls_back_to_the_default() -> None:
    limits = thresholds_from_config({"eligibility_min_score": "not-a-number"})
    assert limits.min_score == 2.0


def test_organism_veto_can_be_switched_off() -> None:
    limits = EligibilityThresholds(organism_veto=False)
    decision = screen_paper(
        paper_id="PMC1", title=T_MEND_LISTERIA, scope=MENAQUINONE_SUBTILIS,
        thresholds=limits,
    )
    # Still recorded as a mismatch, and still penalised -- just not decisive.
    assert decision.organism_match == ORGANISM_MISMATCH
    assert decision.outcome == OUTCOME_INELIGIBLE_PATHWAY
    assert "score" in decision.reason


def test_disabling_the_gate_admits_everything_but_still_reports() -> None:
    limits = EligibilityThresholds(enabled=False)
    decision = screen_paper(
        paper_id="PMC1", title=T_FOURNIER, scope=LIPID_A_ECOLI, thresholds=limits
    )
    assert decision.outcome == OUTCOME_ELIGIBLE
    assert "screening disabled" in decision.reason
    # The evidence is still computed, so the report explains what was let through.
    assert decision.organism_match == ORGANISM_MISMATCH
    assert decision.matched_negative


def test_screening_is_deterministic() -> None:
    first = _screen(T_ENTB, ENTEROBACTIN_ECOLI)
    second = _screen(T_ENTB, ENTEROBACTIN_ECOLI)
    assert first.to_dict() == second.to_dict()


def test_an_unknown_pathway_still_screens_on_generated_aliases() -> None:
    """No lexicon entry for this topic: the gate degrades in power, not correctness."""
    scope = RequestedScope(
        requested_pathway="caffeine degradation", requested_organism="Pseudomonas putida"
    )
    on_topic = screen_paper(
        paper_id="PMC1",
        title="Caffeine degradation pathway in Pseudomonas putida: NdmA catalysis",
        scope=scope,
        thresholds=THRESHOLDS,
    )
    assert on_topic.eligible
    off_topic = screen_paper(
        paper_id="PMC2",
        title="Coffee consumption and sleep quality in university students",
        scope=scope,
        thresholds=THRESHOLDS,
    )
    assert off_topic.ineligible


# ---------------------------------------------------------------------------
# Integration through the batch fetcher.
# ---------------------------------------------------------------------------
def _searcher(mapping: Dict[str, List[CandidatePaper]]):
    def _search(context: Dict[str, Any], **kwargs: Any) -> List[CandidatePaper]:
        status = kwargs.get("status")
        if isinstance(status, dict):
            status["query"] = f'"{context.get("pathway_name")}"'
        return list(mapping.get(str(context.get("pathway_name") or ""), []))

    return _search


def _fetched_ids() -> List[str]:
    return []


def test_ineligible_papers_never_reach_the_full_text_fetch() -> None:
    """The whole point of screening before the download."""
    asked: List[str] = []

    def _texter(candidate: CandidatePaper, **_: Any) -> str:
        asked.append(candidate.id)
        return "body text"

    papers, skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=5)],
        search_fn=_searcher(
            {
                "enterobactin biosynthesis": [
                    CandidatePaper(id="PMC12096016", source="europepmc", title=T_ENTB),
                    CandidatePaper(id="PMC12649316", source="europepmc", title=T_POULTRY),
                    CandidatePaper(id="PMC12971581", source="europepmc", title=T_FOURNIER),
                ]
            }
        ),
        fetch_text_fn=_texter,
        thresholds=THRESHOLDS,
    )
    assert [p.paper_id for p in papers] == ["PMC12096016"]
    # Only the accepted paper's full text was ever requested.
    assert asked == ["PMC12096016"]
    reasons = {s["paper_id"]: s["reason"] for s in skipped}
    assert reasons["PMC12649316"] == OUTCOME_INELIGIBLE_PATHWAY
    assert reasons["PMC12971581"] == OUTCOME_INELIGIBLE_ORGANISM


def test_fetch_papers_records_the_eligibility_report_on_both_sides() -> None:
    papers, skipped = fetch_papers(
        [TopicSpec(topic="menaquinone biosynthesis", organism="Bacillus subtilis", count=5)],
        search_fn=_searcher(
            {
                "menaquinone biosynthesis": [
                    CandidatePaper(
                        id="PMC1",
                        source="europepmc",
                        title=(
                            "Kinetic characterization of MenB from Bacillus subtilis, "
                            "the committed step of menaquinone biosynthesis"
                        ),
                    ),
                    CandidatePaper(id="PMC2", source="europepmc", title=T_MEND_LISTERIA),
                ]
            }
        ),
        fetch_text_fn=lambda candidate, **_: "body",
        thresholds=THRESHOLDS,
    )
    kept = papers[0]
    assert kept.requested_pathway == "menaquinone biosynthesis"
    assert kept.requested_organism == "Bacillus subtilis"
    assert kept.observed_organisms == ["Bacillus subtilis"]
    assert kept.organism_match == ORGANISM_MATCH
    report = kept.eligibility
    # Requirement: score, matched terms both ways, requested vs observed, decision.
    for key in (
        "score",
        "matched_positive",
        "matched_negative",
        "requested_organism",
        "observed_organisms",
        "organism_match",
        "outcome",
        "reason",
    ):
        assert key in report, key
    assert report["outcome"] == OUTCOME_ELIGIBLE

    rejected = skipped[0]
    assert rejected["reason"] == OUTCOME_INELIGIBLE_ORGANISM
    assert rejected["requested_organism"] == "Bacillus subtilis"
    assert rejected["observed_organisms"] == ["Listeria monocytogenes"]
    assert rejected["eligibility"]["outcome"] == OUTCOME_INELIGIBLE_ORGANISM
    assert rejected["eligibility"]["matched_negative"]


def test_a_requested_organism_is_never_written_onto_a_batch_paper_as_observed() -> None:
    papers, _skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=1)],
        search_fn=_searcher(
            {
                "enterobactin biosynthesis": [
                    CandidatePaper(
                        id="PMC1",
                        source="europepmc",
                        title="Enzymatic characterization of the enterobactin synthetase EntF",
                    )
                ]
            }
        ),
        fetch_text_fn=lambda candidate, **_: "body",
        thresholds=THRESHOLDS,
    )
    paper = papers[0]
    assert paper.requested_organism == "Escherichia coli"
    assert paper.observed_organisms == []
    assert paper.organism_match == ORGANISM_UNKNOWN
    assert "Escherichia coli" not in paper.describe().split("observed organisms")[1]


def test_pinned_paper_passes_through_fetch_papers_with_its_warning() -> None:
    papers, skipped = fetch_papers(
        [TopicSpec(pinned_id="PMC4412817")],
        search_fn=_searcher({}),
        fetch_text_fn=lambda candidate, **_: "pinned body",
        thresholds=THRESHOLDS,
    )
    assert skipped == []
    assert [p.paper_id for p in papers] == ["PMC4412817"]
    assert papers[0].eligibility["outcome"] == OUTCOME_PINNED_OVERRIDE


def test_screen_false_restores_the_unfiltered_behavior() -> None:
    papers, skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=5)],
        search_fn=_searcher(
            {
                "enterobactin biosynthesis": [
                    CandidatePaper(id="PMC1", source="europepmc", title=T_FOURNIER)
                ]
            }
        ),
        fetch_text_fn=lambda candidate, **_: "body",
        screen=False,
    )
    assert [p.paper_id for p in papers] == ["PMC1"]
    assert skipped == []
    # Off, but not silent: the record still says what the gate would have said.
    assert papers[0].eligibility["organism_match"] == ORGANISM_MISMATCH


# ---------------------------------------------------------------------------
# Ineligible papers are not pipeline failures.
# ---------------------------------------------------------------------------
def test_an_ineligible_status_is_not_a_failure() -> None:
    for raw in (
        "ineligible_pathway",
        "ineligible_organism",
        "insufficient_metadata",
        "scope_conflict",
        "ineligible",
    ):
        assert _norm_status(raw) == STATUS_INELIGIBLE, raw
        assert _is_failure(_norm_status(raw)) is False, raw
    # The words that DO mean failure still do.
    assert _is_failure(_norm_status("fail")) is True
    assert _is_failure(_norm_status("timeout")) is True


def test_ineligible_papers_do_not_appear_as_pipeline_failures(tmp_path: Path) -> None:
    """End to end: screened-out papers produce no manifest rows, and a manifest
    that does carry an ineligible row is not scored as a failure either."""
    papers, skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=5)],
        search_fn=_searcher(
            {
                "enterobactin biosynthesis": [
                    CandidatePaper(id="PMC12096016", source="europepmc", title=T_ENTB),
                    CandidatePaper(id="PMC12649316", source="europepmc", title=T_POULTRY),
                    CandidatePaper(id="PMC12971581", source="europepmc", title=T_FOURNIER),
                    CandidatePaper(id="PMC12737783", source="europepmc", title=T_TURKEY),
                ]
            }
        ),
        fetch_text_fn=lambda candidate, **_: "body",
        thresholds=THRESHOLDS,
    )
    screened_out = {s["paper_id"] for s in skipped}
    assert screened_out == {"PMC12649316", "PMC12971581", "PMC12737783"}

    # The manifest only ever gets rows for papers that ran.
    rows: List[Dict[str, Any]] = []
    for paper in papers:
        for mode in ("strict", "research"):
            rows.append(
                {
                    "paper_id": paper.paper_id,
                    "slug": paper.slug,
                    "mode": mode,
                    "status": "pass",
                    "title": paper.title,
                }
            )
    # ...plus one hostile row that DOES carry an ineligible status, to prove the
    # report cannot score it as a defect even then.
    rows.append(
        {
            "paper_id": "PMC12971581",
            "slug": "PMC12971581__fournier",
            "mode": "research",
            "status": OUTCOME_INELIGIBLE_ORGANISM,
            "title": T_FOURNIER,
        }
    )
    rows.append(
        {
            "paper_id": "PMC12971581",
            "slug": "PMC12971581__fournier",
            "mode": "strict",
            "status": OUTCOME_INELIGIBLE_ORGANISM,
            "title": T_FOURNIER,
        }
    )

    text = build_summary(rows)
    assert "strict PWML      : 1 pass / 0 fail" in text
    assert "research relaxed : 1 pass / 0 fail" in text
    assert "ineligible       : 2" in text
    # The screened paper is classed "incomplete" (nothing was run), never
    # "research-defect" or "broken". Asserted on the triage matrix counts rather
    # than on the section heading, which is part of the always-printed legend.
    triage = group_papers(rows)
    by_class = {p.label: p.triage_class for p in triage}
    assert by_class["PMC12971581"] == "incomplete"
    assert by_class["PMC12096016"] == "ok"
    assert not any(p.research_defect for p in triage)
    assert not any(p.triage_class == "broken" for p in triage)


# ---------------------------------------------------------------------------
# The plan artifact: a run must carry the eligibility report it was screened with.
# ---------------------------------------------------------------------------
def test_the_plan_carries_the_eligibility_report_and_thresholds(tmp_path: Path) -> None:
    """Requirement 7, at the plan-artifact seam.

    Runs the real ``_plan_for_fresh_run`` with a fetch that returns one accepted
    and one screened-out paper, then reads ``plan.json`` off disk.
    """
    import json

    from t2pw.batch import runner

    papers, skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=5)],
        search_fn=_searcher(
            {
                "enterobactin biosynthesis": [
                    CandidatePaper(id="PMC12096016", source="europepmc", title=T_ENTB),
                    CandidatePaper(id="PMC12971581", source="europepmc", title=T_FOURNIER),
                ]
            }
        ),
        fetch_text_fn=lambda candidate, **_: "body text",
        thresholds=THRESHOLDS,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    topics = tmp_path / "topics.txt"
    topics.write_text("enterobactin biosynthesis | Escherichia coli | 5\n", encoding="utf-8")
    runner._plan_for_fresh_run(
        run_dir,
        topics_path=topics,
        modes=["strict", "research"],
        limit=None,
        fetch_fn=lambda text, limit=None: (papers, skipped),
        log=lambda _message: None,
    )

    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    record = plan["papers"][0]
    assert record["paper_id"] == "PMC12096016"
    assert record["requested_pathway"] == "enterobactin biosynthesis"
    assert record["requested_organism"] == "Escherichia coli"
    assert record["observed_organisms"] == ["Escherichia coli"]
    assert record["organism_match"] == ORGANISM_MATCH
    report = record["eligibility"]
    assert report["outcome"] == OUTCOME_ELIGIBLE
    assert report["score"] > 0
    assert report["matched_positive"]
    assert report["reason"]

    # The plan's own eligibility block: thresholds + tally + review list.
    block = plan["eligibility"]
    assert block["thresholds"]["min_score"] == THRESHOLDS.min_score
    assert block["accepted"] == 1
    assert block["outcomes"][OUTCOME_ELIGIBLE] == 1
    assert block["outcomes"][OUTCOME_INELIGIBLE_ORGANISM] == 1

    # The screened-out paper is in skipped.json with its report, and has NO
    # paper folder -- it never entered the pipeline.
    dropped = json.loads((run_dir / "skipped.json").read_text(encoding="utf-8"))
    assert [d["paper_id"] for d in dropped] == ["PMC12971581"]
    assert dropped[0]["reason"] == OUTCOME_INELIGIBLE_ORGANISM
    assert dropped[0]["eligibility"]["organism_match"] == ORGANISM_MISMATCH
    assert [p.name for p in (run_dir / "papers").iterdir()] == [papers[0].slug]


# ---------------------------------------------------------------------------
# The offline dry-run tool, against the real stored 2026-07-28_2122 plan.
# ---------------------------------------------------------------------------
DRY_RUN_PLAN = ROOT / "runs" / "2026-07-28_2122"

pytestmark_plan = pytest.mark.skipif(
    not (DRY_RUN_PLAN / "plan.json").exists(),
    reason="the stored 2026-07-28_2122 run directory is not present",
)


#: The six papers from runs/2026-07-28_2122 that a pathway extractor could never
#: have succeeded on, and that each cost a full-text download plus two app runs.
JUNK_2122 = (
    "PMC12971581",  # Fournier's gangrene case report
    "PMC12649316",  # chicken ESBL virulence survey
    "PMC12737783",  # turkey ESBL whole-genome survey
    "PMC12797059",  # COVID-19 lncRNA comorbidity study
    "PMC13139079",  # river resistome surveillance
    "PMC12898691",  # gene-set evolution tool
)

#: Genuine Lpx / Men / PPOX / Ent reaction papers -- the acceptances that must
#: survive every tightening of the contextual rules.
GENUINE_2122 = (
    "PMC12444477",  # regulation of lipid A biosynthesis
    "PMC12096016",  # enterobactin EntB / isochorismatase
    "PMC13264790",  # PPOX, heme pathway
    "PMC12856317",  # mitochondrial heme synthesis feedback
    "PMC11946230",  # menaquinone-7 in B. subtilis
)

#: The five real cholesterol false-positive shapes: pathway genes named in an
#: omics hit list, or the molecule named with no local reaction evidence.
CHOLESTEROL_FALSE_POSITIVES_2122 = (
    "PMC12113831",  # Quercus robur keratinocyte proteomics
    "PMC12428349",  # cholesterol regulates airway differentiation
    "PMC12782028",  # CaSR antagonism in osteosarcoma, transcriptome
    "PMC12993329",  # CD79A/CD40 CAR-T metabolic pathway
    "PMC12705669",  # PRDM9 persister cells in glioblastoma
)


def _dry_run_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import eligibility_dry_run  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    return eligibility_dry_run


def _dry_run(*, title_only: bool = True):
    return _dry_run_module().dry_run(
        DRY_RUN_PLAN, thresholds=THRESHOLDS, title_only=title_only
    )


@pytestmark_plan
def test_the_offline_dry_run_reads_the_stored_plan_without_touching_it() -> None:
    before = sorted(p.name for p in DRY_RUN_PLAN.iterdir())
    report = _dry_run()
    assert sorted(p.name for p in DRY_RUN_PLAN.iterdir()) == before
    assert report["papers_in_plan"] == 28
    # This legacy plan stores no abstracts, so a title-only run is provisional.
    assert report["title_only"] is True
    assert report["provisional"] is True
    assert len(report["decisions"]) == 28
    assert len(report["accepted"]) + len(report["rejected"]) == 28


@pytestmark_plan
def test_the_dry_run_verdicts_on_the_real_2122_papers() -> None:
    """Pins the decisions this change was written to produce, on real data."""
    report = _dry_run()
    outcome = {d["paper_id"]: d["outcome"] for d in report["decisions"]}

    # Only PMC12096016 clears a TITLE-ONLY screen with local mechanistic evidence
    # ("biosynthetic intermediate ... competitive inhibitor ... isochorismatase").
    assert outcome["PMC12096016"] == OUTCOME_ELIGIBLE

    # Rejected on the organism: right biochemistry, wrong host.
    assert outcome["PMC12312563"] == OUTCOME_INELIGIBLE_ORGANISM  # Listeria MenD
    assert outcome["PMC12657337"] == OUTCOME_INELIGIBLE_ORGANISM  # vitamin K2 in E. coli
    assert outcome["PMC12971581"] == OUTCOME_INELIGIBLE_ORGANISM  # Fournier's case report

    # Every one of the six known junk papers is rejected on titles alone.
    for paper_id in JUNK_2122:
        assert outcome[paper_id] not in (
            OUTCOME_ELIGIBLE,
            OUTCOME_PINNED_OVERRIDE,
        ), paper_id

    # Title-only rejects of genuine papers are PROVISIONAL, and every one of them
    # is flagged for a human rather than quietly dropped.
    for paper_id in ("PMC12444477", "PMC12856317", "PMC13264790"):
        assert outcome[paper_id] == OUTCOME_INELIGIBLE_PATHWAY, paper_id
        assert paper_id in report["needs_manual_review"], paper_id


@pytestmark_plan
def test_the_dry_run_is_reproducible() -> None:
    assert _dry_run()["decisions"] == _dry_run()["decisions"]
