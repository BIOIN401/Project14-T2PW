"""Regressions for the six corrections to the paper-eligibility gate.

Companion to ``tests/test_paper_eligibility.py``, which covers the gate's original
contract. This module covers what that first pass got wrong:

1. legacy acquisition caches must not become observed-organism evidence;
2. Stage-0 reconciliation must run at a real production boundary;
3. species matching must not treat a related species or a bare genus as a match;
4. mechanistic evidence must be local to the pathway mention, not anywhere;
5. a selective gate must not silently under-deliver a topic's requested count;
6. the exact screening input must be persisted for offline audit.

Everything here is offline and deterministic: no network, no LLM, no full-text
fetch. Thresholds are pinned explicitly so a developer's ``.env`` cannot move a
verdict.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch.fetch import (  # noqa: E402
    BatchPaper,
    TopicSpec,
    eligibility_summary,
    fetch_papers,
)
from t2pw.config import RAG_DEFAULTS, rag_config  # noqa: E402
from t2pw.rag.acquire import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    CandidatePaper,
    _hash_key,
    _ncbi_term,
    build_query,
    is_legacy_candidate_row,
    migrate_cached_payload,
    search_candidates,
)
from t2pw.rag.eligibility import (  # noqa: E402
    CLASS_CONTEXT_ONLY,
    CLASS_MECHANISTIC,
    CLASS_OMICS_ONLY,
    MAX_PERSISTED_ABSTRACT,
    ORGANISM_GENUS_LEVEL,
    ORGANISM_MATCH,
    ORGANISM_MISMATCH,
    ORGANISM_UNKNOWN,
    OUTCOME_ELIGIBLE,
    OUTCOME_INELIGIBLE_ORGANISM,
    OUTCOME_INELIGIBLE_PATHWAY,
    OUTCOME_PINNED_OVERRIDE,
    EligibilityThresholds,
    RequestedScope,
    screen_candidate,
    screen_paper,
)

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from helpers_eligibility import (  # noqa: E402
    CHOLESTEROL_FALSE_POSITIVES_2122,
    CHOLESTEROL_HUMAN,
    DRY_RUN_PLAN,
    ENTEROBACTIN_ECOLI,
    GENUINE_2122,
    HEME_HUMAN,
    JUNK_2122,
    LIPID_A_ECOLI,
    MENAQUINONE_SUBTILIS,
    T_ENTB,
    T_FOURNIER,
    T_HEME_FEEDBACK,
    T_POULTRY,
    THRESHOLDS,
    dry_run_module,
    screen,
    searcher,
)

requires_2122 = pytest.mark.skipif(
    not (DRY_RUN_PLAN / "plan.json").exists(),
    reason="the stored 2026-07-28_2122 run directory is not present",
)


# ===========================================================================
# Correction 1 -- legacy acquisition caches.
# ===========================================================================
#: A candidate row exactly as the pre-2026-07-29 fetchers wrote it: ``organism``
#: holds the organism the SEARCH asked for, and none of the requested/observed
#: fields exist yet. This is the real legacy shape the requirement names.
LEGACY_ROW: Dict[str, Any] = {
    "id": "PMC12971581",
    "source": "europepmc",
    "title": T_FOURNIER,
    "abstract": "",
    "organism": "Escherichia coli",
    "full_text": "",
    "source_uri": "https://europepmc.org/article/PMC/PMC12971581",
    "year": "2026",
}


def test_a_legacy_cache_row_is_recognised_as_such() -> None:
    assert is_legacy_candidate_row(LEGACY_ROW) is True
    # A row from the current serializer is not legacy, even carrying an organism.
    current = CandidatePaper(
        id="PMC1", source="europepmc", organism="Escherichia coli"
    ).to_dict()
    assert is_legacy_candidate_row(current) is False


def test_a_legacy_stamped_organism_is_demoted_to_the_request() -> None:
    paper = CandidatePaper.from_dict(LEGACY_ROW)
    # The stamped value moves to the REQUEST side...
    assert paper.requested_organism == "Escherichia coli"
    # ...and the observed side stays empty, because the paper never said it.
    assert paper.organism == ""
    assert paper.observed_organisms == []
    assert paper.organism_match == ORGANISM_UNKNOWN


def test_an_organism_on_a_modern_row_is_kept_as_the_papers_own() -> None:
    """Key presence, not value, is the legacy signal."""
    row = dict(LEGACY_ROW)
    row.update(
        {
            "organism": "Ochrobactrum anthropi",
            "requested_organism": "Escherichia coli",
            "requested_pathway": "lipid A biosynthesis",
            "observed_organisms": [],
            "observed_pathways": [],
            "organism_match": ORGANISM_MISMATCH,
        }
    )
    paper = CandidatePaper.from_dict(row)
    assert paper.organism == "Ochrobactrum anthropi"
    assert paper.requested_organism == "Escherichia coli"


def test_migrating_a_legacy_cache_payload_demotes_every_row() -> None:
    payload = {"query": {"europepmc_query": "x"}, "candidates": [dict(LEGACY_ROW)]}
    migrated, changed = migrate_cached_payload(payload)
    assert changed is True
    assert migrated["schema_version"] == CACHE_SCHEMA_VERSION
    assert migrated["migrated_from_schema_version"] == 1
    row = migrated["candidates"][0]
    assert row["organism"] == ""
    assert row["requested_organism"] == "Escherichia coli"
    assert row["observed_organisms"] == []
    assert row["organism_match"] == ORGANISM_UNKNOWN
    # Everything else survives, so an existing offline cache stays usable.
    assert row["title"] == T_FOURNIER
    assert row["source_uri"].endswith("PMC12971581")
    assert row["year"] == "2026"


def test_migrating_a_current_payload_is_a_no_op() -> None:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "candidates": [CandidatePaper(id="PMC1", source="x").to_dict()],
    }
    migrated, changed = migrate_cached_payload(payload)
    assert changed is False
    assert migrated == payload


def test_a_legacy_cache_hit_cannot_reintroduce_the_false_ecoli_observation(
    tmp_path: Path,
) -> None:
    """Integration through ``search_candidates`` on a real legacy cache file.

    This is the regression that matters. The cached Fournier's-gangrene row claims
    ``organism: "Escherichia coli"`` because the old fetcher stamped it there. If
    that survived into the observed fields the paper would look like a confirmed
    E. coli paper and sail through the organism check.
    """
    context = {
        "pathway_name": "lipid A biosynthesis",
        "likely_organism": "Escherichia coli",
    }
    key = _hash_key(
        {
            "europepmc_query": build_query(context),
            "ncbi_term": _ncbi_term(context),
            "sources": ["europepmc"],
            "max_papers": 5,
        }
    )
    # Written the way the OLD code did: no schema_version anywhere.
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"query": {}, "candidates": [LEGACY_ROW]}), encoding="utf-8"
    )

    class _NoNetwork:
        def get(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("a cache hit must not touch the network")

    status: Dict[str, Any] = {}
    papers = search_candidates(
        context,
        sources=("europepmc",),
        max_papers=5,
        client=_NoNetwork(),
        cache_dir=tmp_path,
        status=status,
    )
    assert status["from_cache"] is True
    assert status["cache_schema_migrated"] is True
    assert status["cache_schema_version"] == CACHE_SCHEMA_VERSION
    (paper,) = papers
    assert paper.organism == ""
    assert paper.requested_organism == "Escherichia coli"

    # ...and screening it now sees Ochrobactrum, not the stamped E. coli.
    decision = screen_candidate(paper, LIPID_A_ECOLI, thresholds=THRESHOLDS)
    assert decision.observed_organisms == ["Ochrobactrum anthropi"]
    assert decision.organism_match == ORGANISM_MISMATCH
    assert decision.outcome == OUTCOME_INELIGIBLE_ORGANISM


def test_a_legacy_row_flows_through_fetch_papers_without_a_false_observation() -> None:
    """The same guarantee at the batch seam, not only at the acquire seam."""
    legacy = CandidatePaper.from_dict(LEGACY_ROW)
    papers, skipped = fetch_papers(
        [TopicSpec(topic="lipid A biosynthesis", organism="Escherichia coli", count=3)],
        search_fn=searcher({"lipid A biosynthesis": [legacy]}),
        fetch_text_fn=lambda candidate, **_: "body",
        thresholds=THRESHOLDS,
    )
    assert papers == []
    assert skipped[0]["reason"] == OUTCOME_INELIGIBLE_ORGANISM
    assert skipped[0]["observed_organisms"] == ["Ochrobactrum anthropi"]
    assert "Escherichia coli" not in skipped[0]["observed_organisms"]


def test_the_candidate_organism_is_never_promoted_into_observed_organisms() -> None:
    """The unconditional promotion is gone: only the decision writes observed."""
    candidate = CandidatePaper(
        id="PMC1",
        source="europepmc",
        title="Enzymatic characterization of the enterobactin synthetase EntF",
        # Hostile value: whatever this says, it is not an observation the screener
        # made, so it must not reach observed_organisms.
        organism="Escherichia coli",
    )
    papers, _skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=1)],
        search_fn=searcher({"enterobactin biosynthesis": [candidate]}),
        fetch_text_fn=lambda c, **_: "body",
        thresholds=THRESHOLDS,
    )
    paper = papers[0]
    assert paper.observed_organisms == []
    assert paper.organism_match == ORGANISM_UNKNOWN


def test_observed_fields_can_never_disagree_with_the_eligibility_report() -> None:
    papers, _skipped = fetch_papers(
        [
            TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=3),
            TopicSpec(topic="heme biosynthesis", organism="Homo sapiens", count=3),
        ],
        search_fn=searcher(
            {
                "enterobactin biosynthesis": [
                    CandidatePaper(id="PMC12096016", source="europepmc", title=T_ENTB)
                ],
                "heme biosynthesis": [
                    CandidatePaper(
                        id="PMC12856317",
                        source="europepmc",
                        title=T_HEME_FEEDBACK,
                        abstract=(
                            "We show that ferrochelatase activity in human "
                            "mitochondria is controlled by a reversible cofactor "
                            "intermediate."
                        ),
                    )
                ],
            }
        ),
        fetch_text_fn=lambda c, **_: "body",
        thresholds=THRESHOLDS,
    )
    assert len(papers) == 2
    for paper in papers:
        assert paper.scope_disagreements() == []
        report = paper.eligibility
        assert paper.observed_organisms == report["observed_organisms"]
        assert paper.observed_pathways == report["observed_pathways"]
        assert paper.organism_match == report["organism_match"]
        assert paper.requested_organism == report["requested_organism"]
    assert eligibility_summary(papers, [])["scope_disagreements"] == []


def test_scope_disagreements_reports_a_hand_corrupted_record() -> None:
    """The invariant is checked, not assumed."""
    paper = BatchPaper(
        paper_id="PMC1",
        requested_organism="Escherichia coli",
        observed_organisms=["Escherichia coli"],
        organism_match=ORGANISM_MATCH,
        eligibility={
            "observed_organisms": [],
            "observed_pathways": [],
            "organism_match": ORGANISM_UNKNOWN,
            "requested_organism": "Escherichia coli",
        },
    )
    problems = paper.scope_disagreements()
    assert len(problems) == 2
    assert any("observed_organisms" in p for p in problems)
    assert any("organism_match" in p for p in problems)
    summary = eligibility_summary([paper], [])
    assert summary["scope_disagreements"][0]["paper_id"] == "PMC1"


# ===========================================================================
# Correction 3 -- species precision.
# ===========================================================================
@pytest.mark.parametrize(
    "observed",
    ["Escherichia fergusonii", "Escherichia colicinogenes", "Escherichia albertii"],
)
def test_a_different_escherichia_species_is_genus_level_not_a_match(
    observed: str,
) -> None:
    """Word boundaries plus species comparison: ``coli`` != ``colicinogenes``."""
    decision = screen(
        f"Enterobactin biosynthesis and EntB catalysis in {observed}",
        ENTEROBACTIN_ECOLI,
    )
    assert observed in decision.observed_organisms
    assert "Escherichia coli" not in decision.observed_organisms
    assert decision.organism_match == ORGANISM_GENUS_LEVEL
    assert decision.organism_match != ORGANISM_MATCH


def test_bacillus_cereus_is_genus_level_against_a_subtilis_request() -> None:
    decision = screen(
        "MenD catalysis and menaquinone biosynthesis in Bacillus cereus",
        MENAQUINONE_SUBTILIS,
    )
    assert decision.observed_organisms == ["Bacillus cereus"]
    assert decision.organism_match == ORGANISM_GENUS_LEVEL


def test_genus_level_evidence_gets_no_full_organism_bonus_and_a_warning() -> None:
    genus = screen(
        "Enterobactin biosynthesis and EntB catalysis in Escherichia fergusonii",
        ENTEROBACTIN_ECOLI,
    )
    species = screen(
        "Enterobactin biosynthesis and EntB catalysis in Escherichia coli",
        ENTEROBACTIN_ECOLI,
    )
    # Separately represented...
    assert genus.organism_match == ORGANISM_GENUS_LEVEL
    assert species.organism_match == ORGANISM_MATCH
    # ...scored strictly below a real species match...
    assert genus.score < species.score
    assert any("organism_genus_level" in t for t in genus.matched_positive)
    assert not any(t.startswith("organism_match:") for t in genus.matched_positive)
    assert any(t.startswith("organism_match:") for t in species.matched_positive)
    # ...warned about explicitly and sent to a human...
    assert any("GENUS level only" in w for w in genus.warnings)
    assert genus.needs_manual_review is True
    # ...but not rejected: a related taxon is not an incompatible organism.
    assert genus.eligible
    assert not any("incompatible_organism" in t for t in genus.matched_negative)


def test_a_bare_genus_never_infers_the_requested_species() -> None:
    """``Escherichia`` alone is not ``Escherichia coli``."""
    decision = screen(
        "Enterobactin biosynthesis and EntB catalysis in Escherichia",
        ENTEROBACTIN_ECOLI,
    )
    assert decision.observed_organisms == ["Escherichia"]
    assert decision.organism_match == ORGANISM_GENUS_LEVEL
    assert decision.organism_match != ORGANISM_MATCH


@pytest.mark.parametrize(
    "observed",
    ["Escherichia coli K-12", "Escherichia coli K-12 MG1655", "Escherichia coli BW25113"],
)
def test_a_strain_qualified_binomial_is_an_exact_match(observed: str) -> None:
    decision = screen(
        f"EntB catalysis and enterobactin biosynthesis in {observed}",
        ENTEROBACTIN_ECOLI,
    )
    assert decision.organism_match == ORGANISM_MATCH


def test_a_strain_qualified_request_still_matches_the_plain_species() -> None:
    scope = RequestedScope(
        requested_pathway="enterobactin biosynthesis",
        requested_organism="Escherichia coli K-12",
    )
    decision = screen(
        "EntB catalysis and enterobactin biosynthesis in Escherichia coli", scope
    )
    assert decision.organism_match == ORGANISM_MATCH


def test_a_different_genus_is_still_a_mismatch_not_genus_level() -> None:
    decision = screen(
        "MenD catalysis and menaquinone biosynthesis in Listeria monocytogenes",
        MENAQUINONE_SUBTILIS,
    )
    assert decision.organism_match == ORGANISM_MISMATCH
    assert decision.outcome == OUTCOME_INELIGIBLE_ORGANISM


def test_a_species_match_beats_a_genus_level_relation_in_the_same_paper() -> None:
    decision = screen(
        "Enterobactin biosynthesis: EntB catalysis compared between Escherichia "
        "fergusonii and Escherichia coli",
        ENTEROBACTIN_ECOLI,
    )
    assert set(decision.observed_organisms) >= {
        "Escherichia coli",
        "Escherichia fergusonii",
    }
    assert decision.organism_match == ORGANISM_MATCH


def test_a_genus_only_mention_alongside_the_species_does_not_dilute_the_match() -> None:
    decision = screen(
        "EntB catalysis in Escherichia coli, a member of Escherichia",
        ENTEROBACTIN_ECOLI,
    )
    # The bare-genus row is pruned when a species of that genus is also observed.
    assert decision.observed_organisms == ["Escherichia coli"]
    assert decision.organism_match == ORGANISM_MATCH


# ===========================================================================
# Correction 4 -- contextual mechanistic evidence.
# ===========================================================================
CHOLESTEROL_HUMAN = RequestedScope(
    requested_pathway="cholesterol biosynthesis", requested_organism="Homo sapiens"
)


def test_an_alias_plus_a_generic_word_far_away_is_not_evidence() -> None:
    """The exact shape the requirement forbids: alias here, "mechanism" there."""
    decision = screen(
        "Cholesterol biosynthesis in disease",
        CHOLESTEROL_HUMAN,
        abstract=(
            "Cholesterol biosynthesis is a hallmark of proliferating cells. "
            + "We surveyed cases using clinical scoring alone. " * 12
            + "The mechanism of tumour growth is complex and inhibition of growth "
            "was observed."
        ),
    )
    assert decision.ineligible
    assert decision.classification in (CLASS_CONTEXT_ONLY, CLASS_OMICS_ONLY)


def test_an_alias_with_reaction_evidence_in_the_same_sentence_is_accepted() -> None:
    decision = screen(
        "A study of sterols",
        CHOLESTEROL_HUMAN,
        abstract=(
            "We show that cholesterol biosynthesis proceeds through a squalene "
            "intermediate, and measured the catalytic rate of the purified enzyme."
        ),
    )
    assert decision.eligible
    assert decision.classification == CLASS_MECHANISTIC
    assert any("local_mechanism" in t for t in decision.matched_positive)


def test_a_pathway_gene_in_an_omics_hit_list_does_not_anchor() -> None:
    """The PMC12113831 / PMC12782028 shape, reduced to its essentials."""
    decision = screen(
        "In vitro evaluation of a plant leaf extract in human keratinocytes",
        CHOLESTEROL_HUMAN,
        abstract=(
            "Proteomic analysis of treated cells identified differentially "
            "expressed proteins. Among the upregulated hits were SQLE and CYP51A1. "
            "The mechanism of wound healing remains unclear."
        ),
    )
    assert decision.ineligible
    assert decision.classification == CLASS_OMICS_ONLY
    assert any("local_omics_context" in t for t in decision.matched_positive)


def test_a_pathway_gene_named_in_the_title_still_anchors_in_an_omics_paper() -> None:
    """The PMC13264790 / PPOX shape: the enzyme IS the paper's subject."""
    decision = screen(
        "Comprehensive bioinformatics and experimental analysis of PPOX in carcinoma",
        HEME_HUMAN,
        abstract=(
            # Shaped like the real paper: a clean Background sentence naming the
            # enzyme's role, then a Methods sentence that is pure screening.
            "PPOX is the penultimate enzyme of the heme biosynthesis pathway and "
            "catalyzes protoporphyrinogen oxidation. Methods: we profiled it "
            "across transcriptomic datasets."
        ),
    )
    assert decision.eligible
    assert decision.classification == CLASS_MECHANISTIC


@pytest.mark.parametrize("generic", ["inhibition", "mechanism", "flux"])
def test_generic_words_alone_do_not_rescue_a_context_only_mention(
    generic: str,
) -> None:
    """The requirement names inhibition / mechanism / flux / substrate together.

    Three of them are excluded from the strong vocabulary entirely. ``substrate``
    is genuine reaction language and is covered by its own test below, so which is
    which is pinned rather than left to inference.
    """
    decision = screen(
        f"Cholesterol biosynthesis and {generic} in cultured cells", CHOLESTEROL_HUMAN
    )
    assert decision.ineligible
    assert decision.classification == CLASS_CONTEXT_ONLY


def test_substrate_is_genuine_reaction_language_and_does_anchor() -> None:
    decision = screen(
        "Cholesterol biosynthesis and substrate specificity in cultured cells",
        CHOLESTEROL_HUMAN,
    )
    assert decision.eligible
    assert decision.classification == CLASS_MECHANISTIC


def test_evidence_outside_the_local_window_does_not_anchor() -> None:
    """Same document, same words -- only the distance changes."""
    # The same two facts -- the pathway alias and "catalytic rate" -- in one
    # sentence, then far apart. Nothing else differs. (A pathway-specific term such
    # as "squalene" is avoided here: it would anchor on its own via path (a) and
    # the test would stop being about distance.)
    near = screen(
        "Sterols in cells",
        CHOLESTEROL_HUMAN,
        abstract="Cholesterol biosynthesis proceeds at a measured catalytic rate.",
    )
    far = screen(
        "Sterols in cells",
        CHOLESTEROL_HUMAN,
        abstract=(
            "Cholesterol biosynthesis was considered. "
            + "Cells were grown and counted in triplicate over many days. " * 8
            + "A catalytic rate was reported elsewhere."
        ),
    )
    assert near.eligible
    assert far.ineligible
    assert far.classification == CLASS_CONTEXT_ONLY


@requires_2122
@pytest.mark.parametrize("paper_id", CHOLESTEROL_FALSE_POSITIVES_2122)
def test_the_real_cholesterol_false_positives_are_rejected(paper_id: str) -> None:
    """All five real shapes, on the real stored plan, with cached abstracts."""
    report = dry_run_module().dry_run(
        DRY_RUN_PLAN, thresholds=THRESHOLDS, title_only=False
    )
    decision = {d["paper_id"]: d for d in report["decisions"]}[paper_id]
    assert decision["outcome"] == OUTCOME_INELIGIBLE_PATHWAY, decision["reason"]
    assert decision["classification"] in (CLASS_OMICS_ONLY, CLASS_CONTEXT_ONLY)


@requires_2122
@pytest.mark.parametrize("paper_id", GENUINE_2122)
def test_the_genuine_reaction_papers_are_still_accepted(paper_id: str) -> None:
    report = dry_run_module().dry_run(
        DRY_RUN_PLAN, thresholds=THRESHOLDS, title_only=False
    )
    decision = {d["paper_id"]: d for d in report["decisions"]}[paper_id]
    assert decision["outcome"] == OUTCOME_ELIGIBLE, decision["reason"]
    assert decision["classification"] == CLASS_MECHANISTIC


@requires_2122
@pytest.mark.parametrize("paper_id", JUNK_2122)
def test_the_six_known_junk_papers_are_rejected_with_abstracts_too(
    paper_id: str,
) -> None:
    report = dry_run_module().dry_run(
        DRY_RUN_PLAN, thresholds=THRESHOLDS, title_only=False
    )
    decision = {d["paper_id"]: d for d in report["decisions"]}[paper_id]
    assert decision["outcome"] in (
        OUTCOME_INELIGIBLE_PATHWAY,
        OUTCOME_INELIGIBLE_ORGANISM,
    ), decision["reason"]


# ===========================================================================
# Correction 5 -- success rate under a selective gate.
# ===========================================================================
def _paged_searcher(pool: List[CandidatePaper]):
    """A search fake that honours ``max_papers`` the way the real one does."""
    calls: List[int] = []

    def _search(context: Dict[str, Any], **kwargs: Any) -> List[CandidatePaper]:
        size = int(kwargs.get("max_papers") or 0)
        calls.append(size)
        status = kwargs.get("status")
        if isinstance(status, dict):
            status["query"] = f'"{context.get("pathway_name")}"'
        return pool[:size]

    _search.calls = calls  # type: ignore[attr-defined]
    return _search


def _mixed_pool(*, eligible: int, junk: int) -> List[CandidatePaper]:
    """``junk`` rejects first, then ``eligible`` genuine papers behind them."""
    pool = [
        CandidatePaper(id=f"JUNK{i}", source="europepmc", title=T_POULTRY)
        for i in range(junk)
    ]
    pool += [
        CandidatePaper(
            id=f"GOOD{i}",
            source="europepmc",
            title=(
                "Kinetic characterization of EntB catalysis in enterobactin "
                f"biosynthesis in Escherichia coli, part {i}"
            ),
        )
        for i in range(eligible)
    ]
    return pool


def test_acquisition_keeps_searching_until_the_requested_count_is_filled() -> None:
    """A fixed 3x over-fetch would stop at 9 candidates and deliver nothing."""
    search = _paged_searcher(_mixed_pool(eligible=3, junk=20))
    stats: Dict[str, Any] = {}
    papers, _skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=3)],
        search_fn=search,
        fetch_text_fn=lambda c, **_: "body",
        thresholds=THRESHOLDS,
        stats=stats,
    )
    assert len(papers) == 3
    assert stats["accepted"] == 3
    assert stats["requested"] == 3
    # It escalated rather than settling for the first page.
    assert len(search.calls) > 1  # type: ignore[attr-defined]
    assert search.calls[-1] > search.calls[0]  # type: ignore[attr-defined]
    assert stats["topics"][0]["stop_reason"] == "filled"
    assert stats["topics_short"] == []


def test_the_funnel_records_every_stage() -> None:
    search = _paged_searcher(_mixed_pool(eligible=2, junk=4))
    stats: Dict[str, Any] = {}
    papers, _skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=2)],
        search_fn=search,
        # One eligible paper is paywalled -> no_full_text, which is not a failure.
        fetch_text_fn=lambda c, **_: "" if c.id == "GOOD0" else "body",
        thresholds=THRESHOLDS,
        stats=stats,
    )
    for key in (
        "requested",
        "examined",
        "eligible",
        "ineligible",
        "no_full_text",
        "accepted",
    ):
        assert key in stats, key
    assert stats["requested"] == 2
    assert stats["accepted"] == len(papers) == 1
    assert stats["no_full_text"] == 1
    assert stats["ineligible"] == 4
    assert stats["eligible"] == 2  # cleared the gate; one then had no full text
    assert stats["examined"] == 6
    topic = stats["topics"][0]
    assert topic["ineligible_by_outcome"][OUTCOME_INELIGIBLE_PATHWAY] == 4
    assert topic["filled"] is False
    assert stats["topics_short"][0]["accepted"] == 1


def test_acquisition_stops_at_the_configured_candidate_ceiling() -> None:
    """A topic whose literature is all junk must terminate, and say why."""
    search = _paged_searcher(_mixed_pool(eligible=0, junk=200))
    stats: Dict[str, Any] = {}
    papers, skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=5)],
        search_fn=search,
        fetch_text_fn=lambda c, **_: "body",
        thresholds=EligibilityThresholds(candidate_ceiling=25),
        stats=stats,
    )
    assert papers == []
    assert stats["examined"] == 25
    assert stats["topics"][0]["stop_reason"] == "candidate_ceiling"
    # Nothing silently dropped: every examined candidate has a skip record.
    assert len(skipped) == 25


def test_acquisition_stops_when_the_source_runs_dry() -> None:
    search = _paged_searcher(_mixed_pool(eligible=1, junk=2))
    stats: Dict[str, Any] = {}
    papers, _skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=5)],
        search_fn=search,
        fetch_text_fn=lambda c, **_: "body",
        thresholds=THRESHOLDS,
        stats=stats,
    )
    assert len(papers) == 1
    assert stats["topics"][0]["stop_reason"] == "source_exhausted"


def test_an_empty_first_search_is_still_no_candidates() -> None:
    stats: Dict[str, Any] = {}
    papers, skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=3)],
        search_fn=_paged_searcher([]),
        fetch_text_fn=lambda c, **_: "body",
        thresholds=THRESHOLDS,
        stats=stats,
    )
    assert papers == []
    assert [s["reason"] for s in skipped] == ["no_candidates"]
    assert stats["topics"][0]["stop_reason"] == "no_candidates"


def test_a_pinned_paper_appears_in_the_funnel_too() -> None:
    stats: Dict[str, Any] = {}
    papers, _skipped = fetch_papers(
        [TopicSpec(pinned_id="PMC4412817")],
        search_fn=searcher({}),
        fetch_text_fn=lambda c, **_: "pinned body",
        thresholds=THRESHOLDS,
        stats=stats,
    )
    assert [p.paper_id for p in papers] == ["PMC4412817"]
    assert stats["accepted"] == 1
    assert stats["examined"] == 1
    assert stats["topics"][0]["stop_reason"] == "filled"


# ===========================================================================
# Correction 6 -- persisted screening inputs.
# ===========================================================================
def test_the_screening_input_is_persisted_with_its_hash() -> None:
    abstract = (
        "LpxC catalyzes the committed step of lipid A biosynthesis in Escherichia "
        "coli, measured with purified enzyme."
    )
    decision = screen("LpxC kinetics", LIPID_A_ECOLI, abstract=abstract)
    stored = decision.to_dict()["screening_input"]
    assert stored["title"] == "LpxC kinetics"
    assert stored["abstract"] == abstract
    assert stored["abstract_chars"] == len(abstract)
    assert stored["abstract_sha256"] == hashlib.sha256(
        abstract.encode("utf-8")
    ).hexdigest()
    assert stored["abstract_truncated"] is False


def test_a_long_abstract_is_bounded_but_hashed_in_full() -> None:
    abstract = ("LpxC catalyzes lipid A biosynthesis. " * 400).strip()
    decision = screen("LpxC", LIPID_A_ECOLI, abstract=abstract)
    stored = decision.to_dict()["screening_input"]
    assert len(stored["abstract"]) == MAX_PERSISTED_ABSTRACT
    assert stored["abstract_chars"] == len(abstract)
    assert stored["abstract_truncated"] is True
    # The hash is of the FULL text, so the stored slice can be proved to belong.
    assert stored["abstract_sha256"] == hashlib.sha256(
        abstract.encode("utf-8")
    ).hexdigest()


def test_a_rejected_paper_keeps_enough_input_to_reproduce_it_offline() -> None:
    abstract = (
        "Ochrobactrum anthropi was isolated from a patient. We report a rare case "
        "of Fournier's gangrene."
    )
    papers, skipped = fetch_papers(
        [TopicSpec(topic="lipid A biosynthesis", organism="Escherichia coli", count=1)],
        search_fn=searcher(
            {
                "lipid A biosynthesis": [
                    CandidatePaper(
                        id="PMC12971581",
                        source="europepmc",
                        title=T_FOURNIER,
                        abstract=abstract,
                    )
                ]
            }
        ),
        fetch_text_fn=lambda c, **_: "body",
        thresholds=THRESHOLDS,
    )
    assert papers == []
    record = skipped[0]
    stored = record["eligibility"]["screening_input"]
    # Re-screen from ONLY what skipped.json kept: no network, no candidate object.
    replayed = screen_paper(
        paper_id=record["paper_id"],
        title=stored["title"],
        abstract=stored["abstract"],
        scope=RequestedScope(
            requested_pathway=record["requested_pathway"],
            requested_organism=record["requested_organism"],
        ),
        thresholds=THRESHOLDS,
    )
    assert replayed.outcome == record["eligibility"]["outcome"]
    assert replayed.to_dict()["score"] == record["eligibility"]["score"]
    assert replayed.observed_organisms == record["eligibility"]["observed_organisms"]


def test_the_plan_persists_the_screening_input_for_accepted_papers(
    tmp_path: Path,
) -> None:
    from t2pw.batch import runner

    abstract = (
        "The enterobactin biosynthetic protein EntB catalyzes hydrolysis of "
        "isochorismate in Escherichia coli."
    )
    stats: Dict[str, Any] = {}
    papers, skipped = fetch_papers(
        [TopicSpec(topic="enterobactin biosynthesis", organism="Escherichia coli", count=1)],
        search_fn=searcher(
            {
                "enterobactin biosynthesis": [
                    CandidatePaper(
                        id="PMC12096016",
                        source="europepmc",
                        title=T_ENTB,
                        abstract=abstract,
                    )
                ]
            }
        ),
        fetch_text_fn=lambda c, **_: "body",
        thresholds=THRESHOLDS,
        stats=stats,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    topics = tmp_path / "topics.txt"
    topics.write_text(
        "enterobactin biosynthesis | Escherichia coli | 1\n", encoding="utf-8"
    )
    runner._plan_for_fresh_run(
        run_dir,
        topics_path=topics,
        modes=["strict"],
        limit=None,
        fetch_fn=lambda text, limit=None, stats=None: (papers, skipped),
        log=lambda _m: None,
    )
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    stored = plan["papers"][0]["eligibility"]["screening_input"]
    assert stored["abstract"] == abstract
    assert stored["abstract_sha256"]


def test_the_plan_records_the_acquisition_funnel(tmp_path: Path) -> None:
    """The runner threads ``stats`` into ``plan["eligibility"]["acquisition"]``."""
    from t2pw.batch import runner

    captured: Dict[str, Any] = {}
    search = _paged_searcher(_mixed_pool(eligible=1, junk=3))

    def _fetch_fn(text: str, limit: Any = None, stats: Any = None):
        # Stands in for the real ``fetch_papers``, including its ``stats`` out-param.
        papers, skipped = fetch_papers(
            [
                TopicSpec(
                    topic="enterobactin biosynthesis",
                    organism="Escherichia coli",
                    count=1,
                )
            ],
            search_fn=search,
            fetch_text_fn=lambda c, **_: "body",
            thresholds=THRESHOLDS,
            stats=captured,
        )
        if isinstance(stats, dict):
            stats.update(captured)
        return papers, skipped

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    topics = tmp_path / "topics.txt"
    topics.write_text(
        "enterobactin biosynthesis | Escherichia coli | 1\n", encoding="utf-8"
    )
    runner._plan_for_fresh_run(
        run_dir,
        topics_path=topics,
        modes=["strict"],
        limit=None,
        fetch_fn=_fetch_fn,
        log=lambda _m: None,
    )
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    funnel = plan["eligibility"]["acquisition"]
    assert funnel["requested"] == 1
    assert funnel["accepted"] == 1
    assert funnel["ineligible"] == 3
    assert funnel["examined"] == 4
    assert funnel["topics"][0]["stop_reason"] == "filled"


def test_a_fetch_fn_without_a_stats_parameter_still_works(tmp_path: Path) -> None:
    """Back-compat: the out-param is optional, so old callers/fakes keep working."""
    from t2pw.batch import runner

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    topics = tmp_path / "topics.txt"
    topics.write_text("x | y | 1\n", encoding="utf-8")
    plan = runner._plan_for_fresh_run(
        run_dir,
        topics_path=topics,
        modes=["strict"],
        limit=None,
        fetch_fn=lambda text, limit=None: ([], []),
        log=lambda _m: None,
    )
    assert plan["papers"] == []
    assert "acquisition" not in plan["eligibility"]


def test_an_internal_fetch_type_error_is_not_retried(tmp_path: Path) -> None:
    """A fetcher's own TypeError must not trigger a second acquisition attempt."""
    from t2pw.batch import runner

    calls = 0

    def _broken_fetch(text: str, limit: Any = None, stats: Any = None):
        nonlocal calls
        calls += 1
        raise TypeError("bug inside modern fetcher")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    topics = tmp_path / "topics.txt"
    topics.write_text("x | y | 1\n", encoding="utf-8")
    plan = runner._plan_for_fresh_run(
        run_dir,
        topics_path=topics,
        modes=["strict"],
        limit=None,
        fetch_fn=_broken_fetch,
        log=lambda _m: None,
    )
    skipped = json.loads((run_dir / "skipped.json").read_text(encoding="utf-8"))
    assert calls == 1
    assert plan["papers"] == []
    assert skipped[0]["reason"] == "fetch_crashed"
    assert "bug inside modern fetcher" in skipped[0]["detail"]


def test_invalid_numeric_gate_configuration_fails_safe() -> None:
    """NaN/Infinity cannot make every ``score < threshold`` comparison false."""
    config = rag_config(
        {
            "eligibility_min_score": "nan",
            "eligibility_title_only_min_score": "inf",
            "eligibility_review_margin": "-4",
            "eligibility_min_title_chars": "-2",
            "eligibility_local_window_tokens": "0",
            "eligibility_candidate_ceiling": "-10",
        }
    )
    assert config["eligibility_min_score"] == RAG_DEFAULTS["eligibility_min_score"]
    assert (
        config["eligibility_title_only_min_score"]
        == RAG_DEFAULTS["eligibility_title_only_min_score"]
    )
    assert config["eligibility_review_margin"] == 0.0
    assert config["eligibility_min_title_chars"] == 0
    assert config["eligibility_local_window_tokens"] == 1
    assert config["eligibility_candidate_ceiling"] == 1


def test_a_future_dry_run_uses_the_stored_abstract_automatically(
    tmp_path: Path,
) -> None:
    """Requirement 6: no extra input needed to reproduce a plan's verdicts."""
    abstract = (
        "The enterobactin biosynthetic protein EntB catalyzes hydrolysis of "
        "isochorismate in Escherichia coli."
    )
    run_dir = tmp_path / "run"
    (run_dir / "papers").mkdir(parents=True)
    (run_dir / "plan.json").write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "paper_id": "PMC12096016",
                        "title": T_ENTB,
                        "topic": "enterobactin biosynthesis",
                        "requested_pathway": "enterobactin biosynthesis",
                        "requested_organism": "Escherichia coli",
                        "slug": "PMC12096016__x",
                        "eligibility": {
                            "screening_input": {
                                "title": T_ENTB,
                                "abstract": abstract,
                                "abstract_chars": len(abstract),
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = dry_run_module().dry_run(run_dir, thresholds=THRESHOLDS)
    assert report["title_only"] is False
    assert report["abstract_sources"] == {"plan_screening_input": 1}
    decision = report["decisions"][0]
    assert decision["screening_input"]["abstract"] == abstract
    assert decision["outcome"] == OUTCOME_ELIGIBLE
    # A plan-stored abstract is authoritative, so the verdict is NOT provisional.
    assert decision["provisional"] is False


def test_a_future_dry_run_replays_screened_rejections_from_skipped_json(
    tmp_path: Path,
) -> None:
    """Rejected inputs are audit records too, not dead data outside plan.json."""
    decision = screen_paper(
        paper_id="PMC-JUNK-OFFLINE",
        title=T_FOURNIER,
        abstract=(
            "A Fournier gangrene case caused by Ochrobactrum anthropi was treated "
            "with antibiotics after surgical debridement."
        ),
        scope=LIPID_A_ECOLI,
        thresholds=THRESHOLDS,
        abstract_source="candidate_metadata",
    )
    assert decision.ineligible

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "plan.json").write_text(
        json.dumps({"papers": []}), encoding="utf-8"
    )
    (run_dir / "skipped.json").write_text(
        json.dumps(
            [
                {
                    "paper_id": decision.paper_id,
                    "title": decision.title,
                    "requested_pathway": LIPID_A_ECOLI.requested_pathway,
                    "requested_organism": LIPID_A_ECOLI.requested_organism,
                    "reason": decision.outcome,
                    "eligibility": decision.to_dict(),
                }
            ]
        ),
        encoding="utf-8",
    )

    report = dry_run_module().dry_run(run_dir, thresholds=THRESHOLDS)
    assert report["papers_in_plan"] == 0
    assert report["screened_records"] == 1
    assert report["screened_records_from_skipped"] == 1
    assert report["rejected"] == ["PMC-JUNK-OFFLINE"]
    assert report["decisions"][0]["record_source"] == "skipped"
    assert report["decisions"][0]["outcome"] == decision.outcome


def test_screening_never_reads_beyond_the_persisted_abstract_boundary() -> None:
    """A long abstract must produce the same verdict from its stored audit slice."""
    prefix = ("Unrelated clinical background without pathway evidence. " * 100)
    tail = (
        " EntB catalyzes hydrolysis of isochorismate during enterobactin "
        "biosynthesis in Escherichia coli."
    )
    full_abstract = prefix + tail
    assert len(prefix) > MAX_PERSISTED_ABSTRACT

    original = screen_paper(
        paper_id="PMC-LONG",
        title="A clinical observational study",
        abstract=full_abstract,
        scope=ENTEROBACTIN_ECOLI,
        thresholds=THRESHOLDS,
        abstract_source="candidate_metadata",
    )
    persisted = original.screening_input
    replay = screen_paper(
        paper_id="PMC-LONG",
        title=persisted.title,
        abstract=persisted.abstract,
        scope=ENTEROBACTIN_ECOLI,
        thresholds=THRESHOLDS,
        abstract_source=persisted.abstract_source,
        abstract_is_authoritative=persisted.abstract_authoritative,
    )

    assert persisted.abstract_truncated is True
    assert persisted.abstract_chars == len(full_abstract)
    assert persisted.abstract_sha256 == hashlib.sha256(
        full_abstract.encode("utf-8")
    ).hexdigest()
    assert tail.strip() not in persisted.abstract
    assert (original.outcome, original.score, original.classification) == (
        replay.outcome,
        replay.score,
        replay.classification,
    )
    assert original.ineligible
    assert any("truncated before screening" in warning for warning in original.warnings)


def test_a_legacy_plan_falls_back_to_the_stored_full_text(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    paper_dir = run_dir / "papers" / "PMC1__x"
    paper_dir.mkdir(parents=True)
    (paper_dir / "01_source_text.txt").write_text(
        "pmc J Biol Chem 0021-9258 PMC1 12345 10.1000/x 2025 "
        "https://creativecommons.org/licenses/by/4.0/ This is an open access "
        "article under the CC BY license. LpxC catalyzes the committed step of "
        "lipid A biosynthesis in Escherichia coli, and we measured its kinetics "
        "with purified enzyme.",
        encoding="utf-8",
    )
    (run_dir / "plan.json").write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "paper_id": "PMC1",
                        "title": "LpxC regulation",
                        "topic": "lipid A biosynthesis",
                        "organism": "Escherichia coli",
                        "slug": "PMC1__x",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = dry_run_module().dry_run(run_dir, thresholds=THRESHOLDS)
    assert report["abstract_sources"] == {"derived_from_stored_full_text": 1}
    decision = report["decisions"][0]
    assert "lpxc catalyzes" in decision["screening_input"]["abstract"].lower()
    # A derived abstract is a proxy, so the verdict stays provisional.
    assert decision["provisional"] is True
    assert decision["outcome"] == OUTCOME_ELIGIBLE


def test_derive_abstract_skips_front_matter_and_licence_boilerplate() -> None:
    derived = dry_run_module().derive_abstract(
        "pmc Int J Mol Sci 1422-0067 MDPI PMC1 12345 10.3390/x 2026 "
        "Department of Oncology, University of Somewhere, City, Country "
        "https://creativecommons.org/licenses/by/4.0/ This is an open access "
        "article distributed under the terms of the Creative Commons licence. "
        "Heme biosynthesis is catalyzed by ferrochelatase in the mitochondrion, "
        "and we determined its kinetics."
    )
    assert derived.startswith("Heme biosynthesis is catalyzed")
    assert "creativecommons" not in derived.lower()
    assert "Department of Oncology" not in derived


def test_derive_abstract_is_deterministic_and_never_raises() -> None:
    module = dry_run_module()
    assert module.derive_abstract("") == ""
    assert module.derive_abstract("   ") == ""
    text = "Menaquinone biosynthesis proceeds via MenD catalysis. " * 200
    assert module.derive_abstract(text) == module.derive_abstract(text)
    assert len(module.derive_abstract(text)) <= module.DERIVED_ABSTRACT_CHARS


@requires_2122
def test_the_2122_evaluation_with_cached_abstracts_beats_title_only() -> None:
    """The point of persisting inputs: more evidence, better decisions."""
    module = dry_run_module()
    titles = module.dry_run(DRY_RUN_PLAN, thresholds=THRESHOLDS, title_only=True)
    cached = module.dry_run(DRY_RUN_PLAN, thresholds=THRESHOLDS, title_only=False)
    assert cached["abstract_sources"] == {
        "acquisition_cache": 27,
        "derived_from_stored_full_text": 1,
    }
    # Title-only misses genuine papers that the cached abstracts recover...
    assert len(cached["accepted"]) > len(titles["accepted"])
    for paper_id in GENUINE_2122:
        assert paper_id in cached["accepted"], paper_id
    # ...without letting a single known-junk or false-positive shape through.
    for paper_id in tuple(JUNK_2122) + tuple(CHOLESTEROL_FALSE_POSITIVES_2122):
        assert paper_id not in cached["accepted"], paper_id


@requires_2122
def test_the_2122_evaluation_is_reproducible() -> None:
    module = dry_run_module()
    first = module.dry_run(DRY_RUN_PLAN, thresholds=THRESHOLDS, title_only=False)
    second = module.dry_run(DRY_RUN_PLAN, thresholds=THRESHOLDS, title_only=False)
    assert first["decisions"] == second["decisions"]
