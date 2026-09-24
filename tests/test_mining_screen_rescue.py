"""ORCH-743 item 4: the mining screen misses relevant enzymology (pilot-001 finding F5).

The production eligibility screen excluded PMC9947334 (primary recombinant AtLOX2
enzymology, score 4.0) as ``context_only``: its pathway-anchor rule fired before
the score was consulted. Mining now reconsiders such an exclusion, for mining
only. ``src/t2pw/rag/eligibility.py`` is unchanged; see ``Acquirer.rescue``.

G9 correction: ``test_g9_*`` fails behaviourally at 23e1ae83 (the paper is
excluded through the mining acquisition path). The fixture test pins every one of
the 21 pilot-001 screening outcomes against the bio-audit's adjudication.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining.acquisition import Acquirer, OfflinePaperSource  # noqa: E402
from t2pw.mining.config import MiningConfig, StartingInput  # noqa: E402
from t2pw.mining.models import FrontierItem, LiteratureQuery  # noqa: E402
from t2pw.rag.acquire import CandidatePaper  # noqa: E402

FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "mining" / "pilot001_screened_papers.json").read_text(encoding="utf-8")
)
#: Repair round 1: the papers the bio-audit found wrongly rescued at 16fd8a4b.
NEGATIVES = {
    p["candidate_id"]: p for p in json.loads(
        (ROOT / "tests" / "fixtures" / "mining" / "orch743_rescue_negatives.json").read_text(encoding="utf-8")
    )["papers"]
}


def _config(tmp_path):
    seeds = [StartingInput(kind=s["kind"], raw=s["value"], value=s["value"]) for s in FIXTURE["seeds"]]
    return MiningConfig.build(
        target_species=FIXTURE["target_species"], topic=FIXTURE["topic"], starting_inputs=seeds,
        overrides={"runs_dir": str(tmp_path / "runs"), "offline": True, "screen_papers": True,
                   "enabled_via": "test"},
    )


def _candidate(paper):
    return CandidatePaper(id=paper["candidate_id"], source="fixture", title=paper["title"],
                          abstract=paper["abstract"], full_text=paper["abstract"])


def _paper(paper_id):
    return next(p for p in FIXTURE["papers"] if p["paper_id"] == paper_id)


def _acquire(config, papers, round_index, **kwargs):
    acquirer = Acquirer(config, source=OfflinePaperSource([_candidate(p) for p in papers]))
    query = LiteratureQuery(query_id="qr_fixture", query_text="fixture")
    return acquirer.acquire(query, {}, round_index=round_index, max_papers=len(papers), **kwargs)


# ---------------------------------------------------------------------------
# G9
# ---------------------------------------------------------------------------
def test_g9_the_mining_screen_includes_primary_atlox2_enzymology(tmp_path):
    result = _acquire(_config(tmp_path), [_paper("PMC:PMC9947334")], 1)
    assert [p.record.paper_id for p in result.papers] == ["PMC:PMC9947334"]
    record = result.papers[0].record
    assert record.inclusion_reasons[0].startswith("mining_rescue: ")
    assert record.screening_outcome == "mining_rescue"
    assert any("the production screen excluded it: context_only" in n for n in record.notes)


# ---------------------------------------------------------------------------
# The 21 pilot-001 papers, each screened with the vocabulary its round had
# ---------------------------------------------------------------------------
def test_the_pilot001_screening_fixture(tmp_path):
    from t2pw.mining.queries import rescue_terms

    config = _config(tmp_path)
    included, reasons = {}, {}
    for rnd in (1, 2):
        frontier = [FrontierItem(frontier_item_id=f"fr_{i}", kind=f["kind"], target=f["target"])
                    for i, f in enumerate(FIXTURE["frontier_by_round"][str(rnd)])]
        papers = [p for p in FIXTURE["papers"] if p["discovered_round"] == rnd]
        result = _acquire(config, papers, rnd, run_terms=rescue_terms(config, frontier))
        for loaded in result.papers:
            included[loaded.record.paper_id] = True
            reasons[loaded.record.paper_id] = loaded.record.inclusion_reasons[0]
        for record in result.excluded:
            included[record.paper_id] = False
    assert len(included) == 21
    expected = {p["paper_id"]: p["expectation"] for p in FIXTURE["papers"]}

    assert included["PMC:PMC9947334"] is True  # REQUIRED
    assert included["PMID:35404431"] is True  # desired, reached
    assert included["PMC:PMC13364333"] is True  # production already included it
    wrongly_in = [pid for pid, e in expected.items() if e == "must_exclude" and included[pid]]
    assert wrongly_in == []
    # Every rescue says what matched; a production inclusion keeps its own reason.
    assert reasons["PMC:PMC9947334"] == (
        "mining_rescue: Arabidopsis thaliana named; score 4.0 >= 3.0 with the pathway-anchor rule off")
    assert not reasons["PMC:PMC13364333"].startswith("mining_rescue")

    # A run's vocabulary only grows. Screened with the richest one (round 2's),
    # every must-exclude paper still stays out -- including the wheat transgene
    # paper, whose abstract names "ALLENE OXIDE SYNTHASE from Arabidopsis thaliana".
    frontier = [FrontierItem(frontier_item_id=f"fr_{i}", kind=f["kind"], target=f["target"])
                for i, f in enumerate(FIXTURE["frontier_by_round"]["2"])]
    richest = _acquire(config, FIXTURE["papers"], 2, run_terms=rescue_terms(config, frontier))
    assert "allene oxide" in rescue_terms(config, frontier)
    assert [p.record.paper_id for p in richest.papers
            if expected[p.record.paper_id] == "must_exclude"] == []


# ---------------------------------------------------------------------------
# Repair round 1: route 1 (score only) is bounded
# ---------------------------------------------------------------------------
def test_g9_b2_route1_never_rescues_an_off_topic_paper(tmp_path):
    """Bio B-2: two live rescues at 16fd8a4b with no topic term at all (protease, chalcone synthase)."""
    config = _config(tmp_path)
    acquirer = Acquirer(config)
    for pmcid in ("PMC13181208", "PMC13153360"):
        paper = NEGATIVES[pmcid]
        decision = acquirer.screen(_candidate(paper))
        assert (decision.classification, decision.organism_match, decision.score) == ("off_topic", "match", 3.0)
        result = _acquire(config, [paper], 2)
        assert [p.record.paper_id for p in result.papers] == []
        assert [r.paper_id for r in result.excluded] == [paper["paper_id"]]
        assert acquirer.rescue(_candidate(paper), decision) == ""


def test_g9_r1_a_title_only_paper_must_clear_the_raised_title_only_bar(tmp_path):
    """Review B1: with no abstract the production screen applies ``title_only_min_score`` (1.5).
    At 16fd8a4b route 1 left that bar unraised and recorded "score 1.5 >= 3.0"."""
    config = _config(tmp_path)
    title = "Cold stress transcriptome of Arabidopsis thaliana rosettes and jasmonate biosynthesis"
    candidate = CandidatePaper(id="PMC8888888", source="fixture", title=title, abstract="",
                               full_text="Results. A body of text the source can serve.")
    decision = Acquirer(config).screen(candidate)
    assert decision.provisional and decision.organism_match == "match"
    assert decision.classification in {"context_only", "omics_only"}
    assert 1.5 <= decision.score < 2.5

    acquirer = Acquirer(config, source=OfflinePaperSource([candidate]))
    result = acquirer.acquire(LiteratureQuery(query_id="qr_fixture", query_text="fixture"), {},
                              round_index=1, max_papers=1)
    assert [p.record.paper_id for p in result.papers] == []
    assert [r.paper_id for r in result.excluded] == ["PMC:PMC8888888"]
    assert acquirer.rescue(candidate, decision) == ""


# ---------------------------------------------------------------------------
# The rule's boundaries
# ---------------------------------------------------------------------------
def _low_score_candidate(abstract, title="Cold responses of Arabidopsis thaliana rosette leaves"):
    return SimpleNamespace(id="PMC7777777", title=title, abstract=abstract)


def test_species_and_a_run_name_in_one_sentence_rescue_a_low_scoring_paper(tmp_path):
    config = _config(tmp_path)
    acquirer = Acquirer(config)
    candidate = _low_score_candidate("Growth slowed. In Arabidopsis thaliana, 12-OPDA levels rose.")
    decision = acquirer.screen(candidate)
    assert decision.ineligible and decision.classification in {"context_only", "omics_only", "off_topic"}
    assert acquirer.rescue(candidate, decision, ["12-OPDA"]) == (
        "names 'arabidopsis thaliana' and '12-OPDA' in one sentence")
    # No run name, no rescue: nothing is hard-coded.
    assert acquirer.rescue(candidate, decision, []) == ""


def test_species_and_a_run_name_in_different_sentences_do_not_rescue(tmp_path):
    config = _config(tmp_path)
    acquirer = Acquirer(config)
    candidate = _low_score_candidate("Plants of Arabidopsis thaliana were grown. Wheat 12-OPDA levels rose.")
    decision = acquirer.screen(candidate)
    assert decision.ineligible
    assert acquirer.rescue(candidate, decision, ["12-OPDA"]) == ""


def test_the_species_only_in_the_abstract_does_not_rescue(tmp_path):
    """The wheat shape: the target species is named, but only as a gene source."""
    acquirer = Acquirer(_config(tmp_path))
    candidate = _low_score_candidate("12-OPDA from Arabidopsis thaliana was fed to wheat.",
                                     title="Cold responses of wheat rosette leaves")
    decision = acquirer.screen(candidate)
    assert decision.ineligible
    assert acquirer.rescue(candidate, decision, ["12-OPDA"]) == ""


def test_a_veto_is_never_rescued(tmp_path):
    acquirer = Acquirer(_config(tmp_path))
    candidate = _low_score_candidate("In Arabidopsis thaliana, 12-OPDA levels rose.")
    for decision in (
        SimpleNamespace(classification="context_only", organism_match="match",
                        reason="observed organism (Mus musculus) does not match the requested organism"),
        SimpleNamespace(classification="mechanistic", organism_match="match",
                        reason="decisive negative evidence: review -- not a mechanistic pathway paper"),
    ):
        assert acquirer.rescue(candidate, decision, ["12-OPDA"]) == ""
