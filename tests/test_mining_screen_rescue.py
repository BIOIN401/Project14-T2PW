"""ORCH-743 item 4: the mining screen misses relevant enzymology (pilot-001 finding F5).

The production eligibility screen excluded PMC9947334 (primary recombinant AtLOX2
enzymology, score 4.0) as ``context_only``: its pathway-anchor rule fired before
the score was consulted. Mining now reconsiders such an exclusion, for mining
only. ``src/t2pw/rag/eligibility.py`` is unchanged; see ``Acquirer.rescue``.

G9 correction: ``test_g9_the_mining_screen_includes_*`` fails behaviourally at
23e1ae83 (the paper is excluded through the mining acquisition path). The fixture
test pins every one of the 21 pilot-001 screening outcomes against the bio-audit's
adjudication.

Repair round 1 (bio B-2, B-3, review B1): ``test_g9_b2_*``, ``test_g9_r1_*`` and
``test_g9_b3_*`` fail behaviourally at 16fd8a4b, where each of those papers was
rescued, or the vocabulary held a quarantined lead's names.

Repair round 2 (bio B-4): ``test_g9_b4_*`` fail behaviourally at 59eb761d, where
route 1 rescued a paper whose title names another plant and whose abstract names
the target only as a transgene source.
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

import helpers_mining as H  # noqa: E402
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
# The 21 pilot-001 papers
# ---------------------------------------------------------------------------
def _frontier_names(config, rnd):
    """Every name pilot-001's round-``rnd`` frontier held, as 16fd8a4b's ``rescue_terms``
    took them: the seeds and each item's search terms, minus the topic and the species.
    Most come from quarantined Marchantia claims."""
    from t2pw.mining.queries import target_terms

    skip = {config.topic.casefold(), config.target_species.casefold()}
    frontier = [FrontierItem(frontier_item_id=f"fr_{i}", kind=f["kind"], target=f["target"])
                for i, f in enumerate(FIXTURE["frontier_by_round"][str(rnd)])]
    names = [s["value"] for s in FIXTURE["seeds"]] + [t for f in frontier for t in target_terms(f, config)]
    return [t for t in dict.fromkeys(names) if t.casefold() not in skip]


def test_the_pilot001_screening_fixture(tmp_path):
    from t2pw.mining.queries import rescue_terms

    config = _config(tmp_path)
    # Pilot-001 admitted no target_species_supported reaction, so the rescue
    # vocabulary of both rounds is the seeds.
    vocabulary = rescue_terms(config, [])
    assert vocabulary == ["12-OPDA", "AOS"]
    included, reasons = {}, {}
    for rnd in (1, 2):
        papers = [p for p in FIXTURE["papers"] if p["discovered_round"] == rnd]
        result = _acquire(config, papers, rnd, run_terms=vocabulary)
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

    # Stress: screened with every name pilot-001's round-2 frontier held (16fd8a4b's
    # vocabulary, quarantined leads included), every must-exclude paper still stays
    # out -- including the wheat transgene paper, whose abstract names "ALLENE OXIDE
    # SYNTHASE from Arabidopsis thaliana" -- and both rescues still hold.
    stress = _frontier_names(config, 2)
    assert "allene oxide" in stress and "ALA" in stress
    richest = _acquire(config, FIXTURE["papers"], 2, run_terms=stress)
    ids = [p.record.paper_id for p in richest.papers]
    assert [pid for pid in ids if expected[pid] == "must_exclude"] == []
    assert "PMC:PMC9947334" in ids and "PMID:35404431" in ids


# ---------------------------------------------------------------------------
# Repair round 2, bio B-4: route 1 and a target that is only a transgene source
# ---------------------------------------------------------------------------
def test_g9_b4_route1_does_not_rescue_a_transgene_source_paper(tmp_path):
    """PMC10742046: emmer wheat 'overexpressing two genes ... from Arabidopsis thaliana'.
    The production screen still reads an organism match."""
    fixture = json.loads((ROOT / "tests" / "fixtures" / "mining" / "orch743_transgene_source.json")
                         .read_text(encoding="utf-8"))
    paper = fixture["papers"][0]
    config = _config(tmp_path)
    acquirer = Acquirer(config)
    decision = acquirer.screen(_candidate(paper))
    assert (decision.classification, decision.organism_match, decision.score) == ("context_only", "match", 3.5)
    assert acquirer.rescue(_candidate(paper), decision, ["12-OPDA", "AOS"]) == ""
    result = _acquire(config, [paper], 1, run_terms=["12-OPDA", "AOS"])
    assert [p.record.paper_id for p in result.papers] == []


def test_g9_b4_a_triticum_title_is_not_rescued_even_above_the_bar(tmp_path):
    """Pilot-001's PMC11478715 (Triticum, AtAOS / AtOPR3 transgenes) scores 2.5. One synthetic
    sentence lifts it to 3.25 while it stays context_only / match; it must still stay out."""
    paper = _paper("PMC:PMC11478715")
    paper = dict(paper, abstract=paper["abstract"]
                 + " We measured the activity of the AOS enzyme and its substrate conversion in leaves.")
    config = _config(tmp_path)
    acquirer = Acquirer(config)
    decision = acquirer.screen(_candidate(paper))
    assert (decision.classification, decision.organism_match) == ("context_only", "match")
    assert decision.score >= 3.0
    assert acquirer.rescue(_candidate(paper), decision, ["12-OPDA", "AOS"]) == ""


def test_the_title_organism_check_keeps_the_two_required_rescues(tmp_path):
    acquirer = Acquirer(_config(tmp_path))
    for pid in ("PMC:PMC9947334", "PMID:35404431"):  # names Arabidopsis / names no organism
        assert acquirer._other_organisms_in_title(_paper(pid)["title"]) == []
    assert acquirer._other_organisms_in_title("Maize ZmLOX2 in Arabidopsis") == ["Maize"]
    assert acquirer._other_organisms_in_title("A lipoxygenase of Escherichia coli") == ["Escherichia coli"]


# ---------------------------------------------------------------------------
# Repair round 1, bio B-3: route 2's vocabulary and matching
# ---------------------------------------------------------------------------
def test_g9_b3_a_bare_acronym_does_not_rescue_a_paper(tmp_path):
    """Live at 16fd8a4b: 'ALA' (here 5-aminolevulinic acid) rescued a salt-stress paper."""
    config = _config(tmp_path)
    paper = NEGATIVES["PMC12977409"]
    assert paper["title"] == ("Harnessing Rhodopseudomonas palustris strains for salt stress "
                              "mitigation in Arabidopsis thaliana.")
    decision = Acquirer(config).screen(_candidate(paper))
    assert (decision.classification, decision.organism_match, decision.score) == ("off_topic", "match", 0.25)
    run_terms = _frontier_names(config, 2)
    assert "ALA" in run_terms
    result = _acquire(config, [paper], 2, run_terms=run_terms)
    assert [p.record.paper_id for p in result.papers] == []
    assert Acquirer(config).rescue(_candidate(paper), decision, run_terms) == ""


def test_g9_b3_a_foreign_gene_in_a_transgenic_target_host_is_not_rescued(tmp_path):
    """Must-exclude PMC13468208: maize ZmLOX2 'in Transgenic Arabidopsis'. At 16fd8a4b the
    canonical name 'lipoxygenase' in the vocabulary rescued it."""
    config = _config(tmp_path)
    paper = _paper("PMC:PMC13468208")
    assert paper["expectation"] == "must_exclude"
    result = _acquire(config, [paper], 2, run_terms=["lipoxygenase"])
    assert [p.record.paper_id for p in result.papers] == []
    assert [r.paper_id for r in result.excluded] == ["PMC:PMC13468208"]


def test_g9_b3_the_rescue_vocabulary_never_comes_from_a_quarantined_lead(tmp_path):
    """At 16fd8a4b round 2's vocabulary was the whole active frontier, including the items
    the fixture's Zea mays (non-target) reaction produced. Now: seeds, and the names of
    target_species_supported reactions."""
    wired = H.build_controller(tmp_path, budgets={"max_rounds": 2})
    vocabulary = {}
    acquire = wired.controller.acquirer.acquire

    def spy(query, context, **kwargs):
        vocabulary.setdefault(kwargs["round_index"], list(kwargs.get("run_terms") or []))
        return acquire(query, context, **kwargs)

    wired.controller.acquirer.acquire = spy
    wired.controller.run(probe_model=False, probe_embeddings=False)

    graph = wired.controller.state.graph
    maize = next(r for r in graph.reactions.values() if r.species_state == "non_target_species")
    assert maize.substrate_names == ["linoleic acid"]
    assert "linoleic acid" not in vocabulary[2] and "9-HPOD" not in vocabulary[2]
    supported = [r for r in graph.reactions.values()
                 if r.species_state == "target_species_supported" and r.discovery_round == 1]
    assert supported and set(supported[0].product_names) <= set(vocabulary[2])
    assert vocabulary[1] == ["alpha-linolenic acid"]  # round 1: the seed only


def test_rescue_terms_are_seeds_and_target_supported_reaction_names(tmp_path):
    from t2pw.mining.queries import rescue_terms

    config = _config(tmp_path)
    reactions = [
        SimpleNamespace(species_state="target_species_supported", substrate_names=["ALA (18:3n3)", "O2"],
                        product_names=["13-HPOT"], enzyme_names=["lipoxygenase 2"]),
        SimpleNamespace(species_state="non_target_species", substrate_names=["DGLA (20:3n6)"],
                        product_names=["ARA (20:4n6)"], enzyme_names=["MpDES5"]),
        SimpleNamespace(species_state="species_ambiguous", substrate_names=["EPA (20:5n3)"],
                        product_names=["15-HPEPE"], enzyme_names=[]),
    ]
    # The disambiguator is kept; a cofactor, the topic and the species never are terms.
    assert rescue_terms(config, reactions) == ["12-OPDA", "AOS", "ALA (18:3n3)", "13-HPOT", "lipoxygenase 2"]


def test_route2_ignores_short_names_and_matches_acronyms_in_case(tmp_path):
    acquirer = Acquirer(_config(tmp_path))
    title = "Cold responses of Arabidopsis thaliana rosette leaves"
    for abstract, terms in (
        ("In Arabidopsis thaliana, ALA levels rose.", ["ALA"]),  # too short
        ("In Arabidopsis thaliana, Ala levels rose.", ["Ala"]),  # too short
        ("In Arabidopsis thaliana, 12-opda levels rose.", ["12-OPDA"]),  # an acronym, other case
    ):
        candidate = _low_score_candidate(abstract, title=title)
        decision = acquirer.screen(candidate)
        assert decision.ineligible
        assert acquirer.rescue(candidate, decision, terms) == "", abstract
    candidate = _low_score_candidate("In Arabidopsis thaliana, ALA (18:3n3) levels rose.", title=title)
    assert acquirer.rescue(candidate, acquirer.screen(candidate), ["ALA (18:3n3)"]) == (
        "names 'arabidopsis thaliana' and 'ALA (18:3n3)' in one sentence")


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
