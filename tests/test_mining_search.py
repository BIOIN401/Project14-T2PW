"""ORCH-743 item 3: literature search no longer collapses (pilot-001 finding F2).

In pilot-001's round 2, 60 queries x 10 results held only 20 unique papers. There
were three causes, and each has a G9 test here that fails behaviourally at 23e1ae83:

(a) the frontier item's own term was OR'd with the topic and generic words, so it
    was optional -> ``test_g9_a_*``: a paper that never names the item's target
    is returned for that item's query;
(b) a hard-coded top-10 window with EuropePMC filling it before NCBI ->
    ``test_g9_b_*``: the window is 10 and an NCBI-only paper is never reached;
(c) one attempt with no new paper exhausted the item -> ``test_g9_c_*``: no
    second formulation is ever tried.

The remaining tests are new acceptance tests for the knobs and invariants.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.acquisition import PaperSource  # noqa: E402
from t2pw.mining.config import StartingInput  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402
from t2pw.rag import acquire as rag_acquire  # noqa: E402

SPECIES = "Arabidopsis thaliana"


def _matches(query: str, text: str) -> bool:
    """Every AND-clause of a rendered EuropePMC query has one of its phrases in ``text``."""
    low = text.casefold()
    return all(
        any(p.casefold() in low for p in re.findall(r'"([^"]*)"', clause))
        for clause in query.split(" AND ")
    )


class BooleanSource:
    """A literature service that honours the boolean query the context renders to."""

    name = "boolean"

    def __init__(self, papers):
        self.papers = list(papers)  # [(pmcid, text)], in the service's rank order
        self.calls = {}  # rendered query -> returned ids

    def search(self, context, *, sources, max_papers):  # noqa: ARG002
        query = rag_acquire.build_query(dict(context))
        hits = [(i, t) for i, t in self.papers if _matches(query, t)][: int(max_papers)]
        self.calls[query] = [i for i, _ in hits]
        return [rag_acquire.CandidatePaper(id=i, source="boolean", title=t[:80], abstract=t,
                                           full_text=t, source_uri=f"https://example.org/{i}")
                for i, t in hits]

    def full_text(self, candidate):
        return dict(self.papers).get(candidate.id, "")


def _run(tmp_path, papers, *, seeds=("12-OPDA",), explicit=(), budgets=None):
    inputs = [StartingInput(kind="seed_metabolite", raw=s, value=s) for s in seeds]
    inputs += [StartingInput(kind="paper", raw=p, value=p, id_type="pmcid") for p in explicit]
    config = H.make_config(tmp_path, budgets={"max_rounds": 1, "max_papers_discovered": 150,
                                              "max_llm_calls": 200, **(budgets or {})},
                           starting_inputs=inputs)
    source = BooleanSource(papers)
    controller = MiningController(config, source=source, chat_fn=H.ScriptedModel(),
                                  index=H.offline_index(config, tmp_path), enabled_override=True)
    outcome = controller.run(probe_model=False, probe_embeddings=False)
    targets = {i["frontier_item_id"]: i["target"] for i in outcome.state.round_records[0]["input_frontier"]}
    queries = [q for q in outcome.state.queries if targets.get(q.frontier_item_id) == "12-OPDA"]
    return outcome, source, queries


def _paper(n, text):
    return (f"PMC{2000000 + n}", text)


# ---------------------------------------------------------------------------
# (a) the item's own term is required
# ---------------------------------------------------------------------------
def test_g9_a_a_paper_that_never_names_the_target_is_not_returned_for_it(tmp_path):
    decoy = _paper(1, "Jasmonate biosynthesis, metabolism and pathway regulation in Arabidopsis thaliana.")
    hit = _paper(2, "12-OPDA accumulates in wounded Arabidopsis thaliana leaves.")
    outcome, source, queries = _run(tmp_path, [decoy, hit])
    assert queries
    for query in queries:
        returned = source.calls[query.query_text]  # the recorded query IS the executed one
        assert returned and all("12-opda" in dict([decoy, hit])[i].casefold() for i in returned), (
            query.query_text, returned)
    assert queries[0].query_text == f'"12-OPDA" AND "{SPECIES}"'


# ---------------------------------------------------------------------------
# (b) a wider result window, and NCBI actually reached
# ---------------------------------------------------------------------------
def test_g9_b_the_default_result_window_is_wider_than_ten(tmp_path):
    papers = [_paper(n, f"Study {n}: 12-OPDA in Arabidopsis thaliana.") for n in range(30)]
    outcome, _, queries = _run(tmp_path, papers)
    assert queries[0].result_count == 25
    assert len(queries[0].new_paper_ids) == 25


def test_the_result_window_is_a_budget_knob(tmp_path):
    papers = [_paper(n, f"Study {n}: 12-OPDA in Arabidopsis thaliana.") for n in range(30)]
    outcome, _, queries = _run(tmp_path, papers, budgets={"max_results_per_query": 7})
    assert queries[0].result_count == 7
    assert outcome.manifest.config["budgets"]["max_results_per_query"] == 7


def test_g9_b_an_ncbi_only_paper_is_reached_through_search_candidates(tmp_path, monkeypatch):
    """The real ``search_candidates``, with only its network fetchers replaced."""

    def epmc(client, context, *, page_size, organism):  # noqa: ARG001
        return [rag_acquire.CandidatePaper(id=f"PMC{3000000 + n}", source="europepmc",
                                           title=f"EuropePMC paper {n}") for n in range(page_size)]

    def ncbi(client, context, *, retmax, organism):  # noqa: ARG001
        return [rag_acquire.CandidatePaper(id=f"{4000000 + n}", source="ncbi",
                                           title=f"PubMed-only paper {n}") for n in range(3)]

    monkeypatch.setattr(rag_acquire, "_default_cache_dir", lambda: tmp_path / "acquire_cache")
    monkeypatch.setattr(rag_acquire, "_fetch_europepmc", epmc)
    monkeypatch.setattr(rag_acquire, "_fetch_ncbi", ncbi)
    monkeypatch.setitem(rag_acquire._SOURCE_FETCHERS, "europepmc", epmc)
    monkeypatch.setitem(rag_acquire._SOURCE_FETCHERS, "ncbi", ncbi)

    context = {"pathway_name": "", "likely_organism": SPECIES, "key_compounds": ["12-OPDA"],
               "key_proteins": [], "gap_terms": []}
    found = PaperSource().search(context, sources=["europepmc", "ncbi"], max_papers=10)
    assert len(found) == 10
    assert {c.source for c in found} == {"europepmc", "ncbi"}
    assert [c.id for c in found if c.source == "ncbi"] == ["4000000", "4000001", "4000002"]


# ---------------------------------------------------------------------------
# (c) no one-attempt exhaustion
# ---------------------------------------------------------------------------
def test_g9_c_a_second_formulation_reaches_a_paper_the_first_did_not(tmp_path):
    known = [_paper(n, f"Study {n}: 12-OPDA levels in Arabidopsis thaliana.") for n in range(30)]
    deeper = _paper(99, "12-OPDA and jasmonate biosynthesis in Arabidopsis thaliana chloroplasts.")
    outcome, _, queries = _run(tmp_path, [*known, deeper], explicit=[i for i, _ in known])

    assert len(queries) == 2 and len({q.query_text for q in queries}) == 2
    record = outcome.state.papers.get("PMC:PMC2000099")
    assert record is not None and record.discovered_via_query_id == queries[1].query_id
    assert record.processed_round == 1
    assert queries[1].query_text == f'"12-OPDA" AND "jasmonate biosynthesis" AND "{SPECIES}"'
    assert [q.attempt for q in queries] == [1, 2]


def test_a_zero_result_query_exhausts_the_item_at_once(tmp_path):
    outcome, _, queries = _run(tmp_path, [_paper(1, "Nothing relevant in Arabidopsis thaliana.")])
    assert [q.attempt for q in queries] == [1] and queries[0].result_count == 0
    assert queries[0].frontier_item_id in outcome.state.exhausted_frontier


def test_every_formulation_tried_without_a_new_paper_exhausts_the_item(tmp_path):
    known = [_paper(n, f"Study {n}: 12-OPDA and jasmonate biosynthesis in Arabidopsis thaliana.")
             for n in range(3)]
    outcome, _, queries = _run(tmp_path, known, explicit=[i for i, _ in known])
    assert [q.attempt for q in queries] == [1, 2]
    assert queries[0].frontier_item_id in outcome.state.exhausted_frontier


def test_a_new_paper_on_the_first_attempt_needs_no_second(tmp_path):
    outcome, _, queries = _run(tmp_path, [_paper(1, "12-OPDA in Arabidopsis thaliana.")])
    assert [q.attempt for q in queries] == [1]
    assert queries[0].frontier_item_id not in outcome.state.exhausted_frontier


# ---------------------------------------------------------------------------
# The rendered strings: pinned, because the AND rides on rag's quoting
# ---------------------------------------------------------------------------
def test_rendered_queries_require_every_own_term_and_the_species(tmp_path):
    from t2pw.mining import queries as Q
    from t2pw.mining.models import FrontierItem

    config = H.make_config(tmp_path)
    cases = {
        ("bridge_search", "EPA (20:5n3) | jasmonate biosynthesis"):
            '"EPA" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana"',
        ("missing_species_evidence", "13-HPOT -> allene oxide (12,13-EOT)"):
            '"13-HPOT" AND "allene oxide" AND "Arabidopsis thaliana"',
        ("missing_direction", "jasmonic acid + L-isoleucine <=> JA-Ile"):
            '"jasmonic acid" AND "JA-Ile" AND "Arabidopsis thaliana"',
        ("conflict_resolution", "enzyme: 13-HPOT -> 12-OPDA"):
            '"13-HPOT" AND "12-OPDA" AND "Arabidopsis thaliana"',
        ("downstream_expansion", "12-OPDA"): '"12-OPDA" AND "Arabidopsis thaliana"',
    }
    for (kind, target), expected in cases.items():
        item = FrontierItem(frontier_item_id="fr_x", kind=kind, target=target)
        context = Q.build_context(item, config)
        assert rag_acquire.build_query(context) == expected
        assert Q.build_query(item, config, round_index=1).query_text == expected
    ncbi = rag_acquire._ncbi_term(Q.build_context(
        FrontierItem(frontier_item_id="fr_x", kind="bridge_search", target="EPA | 12-OPDA"), config))
    assert ncbi == '("EPA" AND "12-OPDA") AND Arabidopsis thaliana[Organism]'

    topic_item = FrontierItem(frontier_item_id="fr_t", kind="seed_expansion", target=config.topic)
    assert Q.build_query(topic_item, config, round_index=1, attempt=2) is None
