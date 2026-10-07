"""ORCH-745 change set C: run control.

The pilot-003 audit's run-control defects: the per-round cap stopped the run on
quarantined junk (B1 / F-7), a round started after the discovery budget was spent
(A5), the frontier cap cut every target-species search (F-3), thin reasoning-free
answers passed as ``ok`` (B2), and the CLI never loaded ``.env`` (F9).

Every ``test_g9_745c_*`` node fails behaviourally at base ``f9556288`` (a wrong
value or a failed assertion, never an ImportError) and imports only what the base
has. A ``test_new_acceptance_745c_*`` node pins a new helper and may import it
inside the test. No network and no model: every source and model is scripted.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402

_EVIDENCE = re.compile(r"\[evidence_id:\s*([A-Za-z0-9_]+)\]")


# ---------------------------------------------------------------------------
# B1 / F-7: max_new_reactions_per_round counts newly admitted reactions only
# ---------------------------------------------------------------------------
def _off_species_review(count):
    """A model reading a dense review of another organism: ``count`` distinct,
    fully cited Zea mays conversions per batch. Every one is a quarantined lead."""

    def chat(messages, **kwargs):  # noqa: ARG001
        user = "".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        ev = _EVIDENCE.findall(user)[0]
        claims = [
            {
                "substrates": [{"name": f"maize lipid {k}", "evidence_ids": [ev]}],
                "products": [{"name": f"maize oxylipin {k}", "evidence_ids": [ev]}],
                "species": "Zea mays",
                "species_evidence_ids": [ev],
                "evidence_ids": [ev],
            }
            for k in range(count)
        ]
        return SimpleNamespace(text=json.dumps({"claims": claims}), diagnostics=None)

    return chat


def test_g9_745c_b1_quarantined_candidates_do_not_stop_the_run(tmp_path):
    """Pilot-003: round 1 added 0 admitted and 35 quarantined reactions, and
    35 >= 25 ended the run. At f9556288 this run stops after round 1 with
    max_new_reactions_per_round_reached."""
    config = H.make_config(tmp_path, budgets={"max_rounds": 3})
    assert config.budgets.max_new_reactions_per_round == 25
    controller = MiningController(
        config,
        source=H.ScriptedPaperSource({1: ["P5"]}),
        chat_fn=_off_species_review(26),
        index=H.offline_index(config, tmp_path),
        enabled_override=True,
    )
    outcome = controller.run(probe_model=False, probe_embeddings=False)

    first = outcome.state.deltas[0]
    assert first.added_reaction_ids == []
    assert len(first.added_candidate_reaction_ids) >= 25
    assert outcome.stop_reason != "max_new_reactions_per_round_reached"
    assert outcome.manifest.rounds_completed >= 2


def test_g9_745c_b1_the_fixture_round_counts_its_admitted_reactions_only(tmp_path):
    """Fixture round 1 admits 2 reactions and quarantines 1 (Zea mays). With the cap
    at 3, f9556288 counted 3 and stopped after round 1; only 2 were admitted."""
    mined = H.run_fixture(tmp_path, budgets={"max_new_reactions_per_round": 3})
    first = mined.outcome.state.deltas[0]
    assert (len(first.added_reaction_ids), len(first.added_candidate_reaction_ids)) == (2, 1)
    assert mined.outcome.stop_reason != "max_new_reactions_per_round_reached"
    assert mined.outcome.manifest.rounds_completed > 1


def test_745c_b1_a_round_admitting_the_cap_still_stops_the_run(tmp_path):
    """The cap stays a stop condition: 2 admitted reactions meet a cap of 2."""
    mined = H.run_fixture(tmp_path, budgets={"max_new_reactions_per_round": 2})
    outcome = mined.outcome
    assert outcome.stop_reason == "max_new_reactions_per_round_reached"
    assert outcome.manifest.rounds_completed == 1
    assert "newly admitted reactions in round 1: 2 >= limit 2" in outcome.manifest.stop_detail


# ---------------------------------------------------------------------------
# A5: no phantom round after the discovery budget is spent
# ---------------------------------------------------------------------------
def test_g9_745c_a5_a_spent_discovery_budget_starts_no_further_round(tmp_path):
    """The seed's query takes the whole 4-paper discovery budget in round 1, and the
    topic item records max_papers_discovered_reached mid-round. At f9556288 a
    round 2 still started; it could not search, and a round 3 followed."""
    mined = H.run_fixture(tmp_path, budgets={"max_papers_discovered": 4})
    outcome = mined.outcome
    assert outcome.ledger.usage.papers_discovered == 4
    assert outcome.stop_reason == "max_papers_discovered_reached"
    assert outcome.ledger.usage.rounds == 1
    assert outcome.manifest.rounds_completed == 1
    assert outcome.run_dir.completed_rounds() == [1]
    assert not outcome.run_dir.path("rounds", "round_002").exists()


def test_g9_745c_a5_check_before_round_stops_on_the_discovery_budget():
    """The budget can be spent exactly by a round's last query, so no mid-round check
    records it. At f9556288 check_before_round did not look at it (returned None)."""
    from t2pw.mining.budget import BudgetLedger
    from t2pw.mining.config import MiningBudgets

    ledger = BudgetLedger(MiningBudgets(max_papers_discovered=3))
    ledger.record_papers_discovered(3)
    assert ledger.check_before_round() == "max_papers_discovered_reached"
    assert ledger.stop_detail == "papers discovered: 3 >= limit 3"


def test_g9_745c_a5_a_stop_recorded_mid_round_forbids_the_next_round():
    """A stop recorded mid-round is final. At f9556288 check_before_round ignored it
    (returned None) whenever the counters it re-checks were under their caps."""
    from t2pw.mining.budget import BudgetLedger
    from t2pw.mining.config import MiningBudgets

    ledger = BudgetLedger(MiningBudgets())
    ledger.stop("max_papers_discovered_reached", "recorded mid-round")
    assert ledger.check_before_round() == "max_papers_discovered_reached"


# ---------------------------------------------------------------------------
# F-3: the frontier cap keeps a quarter of its slots for species evidence
# ---------------------------------------------------------------------------
def _crowded_state(tmp_path, *, admitted=35, leads=20, cap=60):
    """A graph whose next frontier holds far more than ``cap`` items: ``admitted``
    disconnected target-species reactions (a bridge, a downstream and an upstream
    item each, all priority >= 6.5) and ``leads`` quarantined Zea mays leads (one
    missing_species_evidence item each, priority 6.0, so they sort last). Built
    directly, with no admission policy involved."""
    from t2pw.mining.claims import ClaimStore
    from t2pw.mining.graph import ResearchGraph, ResearchReaction
    from t2pw.mining.models import ClaimDecision, ExtractedClaim, GraphDelta, ReactionParticipant

    config = H.make_config(tmp_path, budgets={"max_frontier_size": cap})
    graph = ResearchGraph(topic=config.topic)
    store = ClaimStore()
    delta = GraphDelta(round_index=1)

    def place(identity, substrate, product, *, admitted_, species):
        sub = graph.registry.upsert(substrate, declared_type="metabolite")
        prod = graph.registry.upsert(product, declared_type="metabolite")
        reaction = graph.add_reaction(
            ResearchReaction(
                reaction_id="", claim_identity=identity,
                substrate_ids=[sub.entity_id], product_ids=[prod.entity_id],
                substrate_names=[substrate], product_names=[product],
                direction="forward", compartment="chloroplast",
                species_text=species, admitted=admitted_,
                species_state="target_species_supported" if admitted_ else "non_target_species",
                quarantine_reason="" if admitted_ else "non_target_species",
            ),
            round_index=1,
        )
        (delta.added_reaction_ids if admitted_ else delta.added_candidate_reaction_ids).append(
            reaction.reaction_id)

    for k in range(admitted):
        place(f"arabidopsis thaliana::a{k}|b{k}", f"precursor a{k}", f"product b{k}",
              admitted_=True, species=H.TARGET_SPECIES)
    for k in range(leads):
        identity = f"zea mays::lead x{k}|lead y{k}"
        claim = ExtractedClaim(
            claim_id=f"cl_lead_{k:02d}", claim_identity=identity, paper_id="PMC9",
            subject=f"lead x{k}", object=f"lead y{k}", species_text="Zea mays",
            species_state="non_target_species",
            substrates=[ReactionParticipant(name=f"lead x{k}")],
            products=[ReactionParticipant(name=f"lead y{k}")],
        )
        store.register(claim)
        store.record_decision(ClaimDecision.build(
            claim=claim, status="candidate", reason="non-target species", round_index=1))
    return SimpleNamespace(graph=graph, store=store, delta=delta, config=config)


def _generate(state, **kwargs):
    from t2pw.mining import frontier

    return frontier.generate(graph=state.graph, store=state.store, delta=state.delta,
                             config=state.config, round_index=1, **kwargs)


def test_g9_745c_f3_the_cap_keeps_a_quarter_of_its_slots_for_species_items(tmp_path):
    """Pilot-003: the cap kept everything at priority >= 6.5 and cut all 33
    missing_species_evidence items. At f9556288 this keeps 0 species items."""
    state = _crowded_state(tmp_path)
    items = _generate(state)
    species = [i for i in items if i.kind == "missing_species_evidence"]
    assert len(items) == 60
    assert len(species) >= 15


def test_g9_745c_f3_the_cut_is_recorded_in_the_round_artifacts(tmp_path):
    """The fixture round 1 frontier holds 17 items; with the cap at 8 the cut is
    written to frontier_cut.jsonl and counted in metrics. At f9556288 the cap cut
    silently and neither exists."""
    mined = H.run_fixture(tmp_path, budgets={"max_frontier_size": 8, "max_rounds": 1})
    round_dir = mined.outcome.run_dir.path("rounds", "round_001")
    assert (round_dir / "frontier_cut.jsonl").is_file()
    rows = [json.loads(line) for line in
            (round_dir / "frontier_cut.jsonl").read_text(encoding="utf-8").splitlines() if line]
    metrics = json.loads((round_dir / "metrics.json").read_text(encoding="utf-8"))
    assert rows and metrics["frontier_cut"]["count"] == len(rows)
    assert sum(metrics["frontier_cut"]["by_kind"].values()) == len(rows)
    kept = {i.frontier_item_id for i in mined.outcome.state.frontier}
    assert len(kept) == 8
    for row in rows:
        assert {"frontier_item_id", "kind", "target"} <= set(row)
        assert row["frontier_item_id"] not in kept


def test_new_acceptance_745c_f3_cap_frontier_orders_the_reserve_by_relevance():
    """The reserved slots go to the most on-route species items, each in every
    fourth place; everything else keeps the priority order; the cut is returned."""
    from t2pw.mining.frontier import cap_frontier
    from t2pw.mining.models import FrontierItem

    def item(n, kind, priority, relevance=0):
        return FrontierItem(frontier_item_id=f"fr_{n:02d}", kind=kind, target=f"t{n:02d}",
                            priority=priority, relevance=relevance)

    others = [item(n, "downstream_expansion", 7.0) for n in range(10)]
    species = [item(10 + n, "missing_species_evidence", 6.0, relevance=n % 3) for n in range(6)]
    kept, cut = cap_frontier(others + species, 8)
    assert [i.frontier_item_id for i in kept] == [
        "fr_00", "fr_01", "fr_02", "fr_12", "fr_03", "fr_04", "fr_05", "fr_15"]
    assert {i.frontier_item_id for i in cut} == {
        "fr_06", "fr_07", "fr_08", "fr_09", "fr_10", "fr_11", "fr_13", "fr_14"}
    # Fewer species items than the reserve: all of them are kept.
    kept, _ = cap_frontier(others + species[:1], 8)
    assert [i.kind for i in kept].count("missing_species_evidence") == 1
    # No species item: the plain priority order.
    kept, cut = cap_frontier(list(reversed(others)), 4)
    assert [i.frontier_item_id for i in kept] == ["fr_00", "fr_01", "fr_02", "fr_03"]
    assert len(cut) == 6


def test_new_acceptance_745c_f3_on_route_relevance_is_scored_per_lead(tmp_path):
    """One point each for naming the target species, sharing a name with a seed or a
    target-supported reaction, and a component touching the primary graph."""
    from t2pw.mining.graph import ResearchReaction
    from t2pw.mining.models import ClaimDecision, ExtractedClaim, ReactionParticipant

    state = _crowded_state(tmp_path, admitted=2, leads=0)
    state.graph.select_primary_component(round_index=1, topic=state.config.topic)
    primary = state.graph.primary_reactions[0]

    def lead(k, substrate, product, species):
        claim = ExtractedClaim(
            claim_id=f"cl_rel_{k}", claim_identity=f"x::{substrate}|{product}", paper_id="PMC9",
            subject=substrate, object=product, species_text=species,
            substrates=[ReactionParticipant(name=substrate)],
            products=[ReactionParticipant(name=product)],
        )
        state.store.register(claim)
        state.store.record_decision(ClaimDecision.build(
            claim=claim, status="candidate", reason="lead", round_index=1))
        return claim.claim_identity

    lead(0, "unrelated m", "unrelated n", "Zea mays")
    lead(1, "unrelated p", "unrelated q", "Arabidopsis thaliana")
    # alpha-linolenic acid is the fixture config seed.
    lead(2, "alpha-linolenic acid", "unrelated r", "Arabidopsis thaliana")
    items = {i.target: i.relevance for i in _generate(state)
             if i.kind == "missing_species_evidence"}
    assert items == {"unrelated m -> unrelated n": 0, "unrelated p -> unrelated q": 1,
                     "alpha-linolenic acid -> unrelated r": 2}

    # A lead in a quarantine component that touches the primary graph scores the third point.
    sub = state.graph.registry.get(primary.product_ids[0])
    other = state.graph.registry.upsert("unrelated s", declared_type="metabolite")
    identity = lead(3, sub.name, "unrelated s", "Arabidopsis thaliana")
    state.graph.add_reaction(ResearchReaction(
        reaction_id="", claim_identity=identity, substrate_ids=[sub.entity_id],
        product_ids=[other.entity_id], substrate_names=[sub.name], product_names=["unrelated s"],
        admitted=False, quarantine_reason="inferred_only", species_state="inferred_only"),
        round_index=1)
    items = {i.target: i.relevance for i in _generate(state)
             if i.kind == "missing_species_evidence"}
    assert items[f"{sub.name} -> unrelated s"] == 3


# ---------------------------------------------------------------------------
# B2: a thin, reasoning-free answer is re-asked once and marked; provider recorded
# ---------------------------------------------------------------------------
THIN_FIXTURE = ROOT / "tests" / "fixtures" / "mining" / "pilot003_thin_answers.json"


def _pilot003_answers():
    data = json.loads(THIN_FIXTURE.read_text(encoding="utf-8"))
    return {row["call_id"][-4:]: row for row in data["thin"] + data["cited"]}


def _cited_answer(evidence_ids):
    ev = evidence_ids[0]
    return json.dumps({"claims": [{
        "substrates": [{"name": "alpha-linolenic acid", "evidence_ids": [ev]}],
        "products": [{"name": "13-HPOT", "evidence_ids": [ev]}],
        "enzyme": "LOX2", "enzyme_evidence_ids": [ev],
        "species": "Arabidopsis thaliana", "species_evidence_ids": [ev],
        "evidence_ids": [ev],
    }]})


def _scripted_chat(texts, calls, **diagnostics):
    """A chat_detailed stand-in that answers ``texts`` in turn, like a provider route."""

    def chat(messages, **kwargs):
        calls.append({"messages": messages, "kwargs": kwargs})
        text = texts[min(len(calls), len(texts)) - 1]
        return SimpleNamespace(text=text, diagnostics=SimpleNamespace(
            finish_reason="stop", attempts=1, model="deepseek/deepseek-v4-flash",
            request_hash="", response_hash=f"sha256:{len(calls)}", attempt_log=[
                {"attempt": 1, "status": "ok", "content_chars": len(text), "finish_reason": "stop"}],
            **diagnostics))

    return chat


def _client(chat, max_llm_calls=20):
    from t2pw.mining.budget import BudgetLedger
    from t2pw.mining.config import MiningBudgets, MiningModelConfig
    from t2pw.mining.modelclient import MiningModelClient

    ledger = BudgetLedger(MiningBudgets(max_llm_calls=max_llm_calls, max_cost_usd=10.0))
    return MiningModelClient(MiningModelConfig(model="deepseek/deepseek-v4-flash"), ledger,
                             chat_fn=chat, token_stats_fn=lambda: {}), ledger


def _call(client, evidence_ids):
    from t2pw.mining.extraction import CLAIM_SCHEMA, TIGHT_CLAIM_SCHEMA

    user = "\n".join(f"[evidence_id: {e}]\nPassage." for e in evidence_ids)
    return client.call_json(purpose="extract_claims", system="extract", user=user,
                            schema=CLAIM_SCHEMA, tight_schema=TIGHT_CLAIM_SCHEMA, round_index=1,
                            paper_id="PMC:PMC8508710", evidence_ids=evidence_ids)


def test_g9_745c_b2_a_thin_answer_is_re_asked_and_the_cited_answer_is_used():
    """Pilot-003 call _0003 (209 output tokens, 4.8 s; stoichiometry
    "has_stoichiometry", no citation but the claim-level list) was accepted as ok.
    At f9556288 the call returns that payload after 1 request."""
    thin = _pilot003_answers()["0003"]
    cited = _cited_answer(thin["evidence_ids"])
    calls = []
    client, ledger = _client(_scripted_chat([thin["raw_text"], cited], calls))
    outcome = _call(client, thin["evidence_ids"])

    assert len(calls) == 2
    assert outcome.ok
    assert outcome.payload == json.loads(cited)
    assert ledger.usage.llm_calls == 2, "the re-ask is a counted call"
    record = outcome.record.to_dict()
    assert record["answer_quality"] == "retried_after_thin"
    assert [r["parse_outcome"] for r in outcome.raw_attempts] == ["thin_answer", "ok"]
    assert "has_stoichiometry" in outcome.raw_attempts[0]["thin_reason"]
    assert outcome.raw_attempts[1]["retry_of_thin_answer"] is True
    assert {r["answer_quality"] for r in outcome.lineage_rows()} == {"retried_after_thin"}
    assert outcome.raw_attempts[0]["output_tokens"] > 0


def test_g9_745c_b2_a_thin_answer_twice_is_kept_and_marked():
    """Pilot-003 call _0014 twice: the second thin answer is kept, never upgraded.
    At f9556288 the first was returned after 1 request and nothing marked it."""
    thin = _pilot003_answers()["0014"]
    calls = []
    client, _ = _client(_scripted_chat([thin["raw_text"], thin["raw_text"]], calls))
    outcome = _call(client, thin["evidence_ids"])

    assert len(calls) == 2
    assert outcome.ok and outcome.payload == json.loads(thin["raw_text"])
    assert outcome.record.to_dict().get("answer_quality") == "thin_after_retry"
    assert [r["parse_outcome"] for r in outcome.raw_attempts] == ["thin_answer", "thin_answer"]


def test_g9_745c_b2_the_provider_is_recorded_or_said_to_be_unreported():
    """Pilot-003: provider was "" on all 16 records. The shared client reports no
    provider, and at f9556288 the record stayed "" either way."""
    answer = _pilot003_answers()["0010"]
    calls = []
    client, _ = _client(_scripted_chat([answer["raw_text"]], calls))
    outcome = _call(client, answer["evidence_ids"])
    assert outcome.record.provider == "not_reported_by_llm_client"
    assert outcome.record.to_dict().get("answer_quality") == "ok"
    assert len(calls) == 1, "a cited answer is not re-asked"

    client, _ = _client(_scripted_chat([answer["raw_text"]], [], provider="DeepInfra"))
    outcome = _call(client, answer["evidence_ids"])
    assert outcome.record.provider == "DeepInfra"
    assert outcome.raw_attempts[0]["provider"] == "DeepInfra"


def test_new_acceptance_745c_b2_thin_answer_names_the_pilot_shapes():
    from t2pw.mining.modelclient import thin_answer

    answers = _pilot003_answers()
    for key in ("0001", "0003", "0014"):
        assert thin_answer(json.loads(answers[key]["raw_text"])), key
    assert "has_stoichiometry" in thin_answer(json.loads(answers["0003"]["raw_text"]))
    assert "products_evidence_ids_placeholder" in thin_answer(json.loads(answers["0014"]["raw_text"]))
    assert thin_answer(json.loads(answers["0010"]["raw_text"])) == ""
    assert thin_answer({"claims": []}) == "", "an empty list is a legitimate answer"
    assert thin_answer({"claims": [{"substrates": [{"name": "x"}], "products": [{"name": "y"}],
                                    "species_evidence_ids": ["ev_1"], "evidence_ids": ["ev_1"]}]}) == ""
    assert thin_answer({"claims": [{"substrates": [{"name": "x", "evidence_ids": ["ev_..."]}],
                                    "products": [], "evidence_ids": ["ev_1"]}]})
    assert thin_answer({"claims": [{"substrates": [{"name": "x", "evidence_ids": ["ev_1"]}],
                                    "products": [], "gene": "HAS_1", "gene_evidence_ids": ["ev_1"],
                                    "evidence_ids": ["ev_1"]}]}) == "", "a gene symbol is not an echo"


def test_new_acceptance_745c_b2_no_budget_for_the_re_ask_keeps_the_thin_answer_marked():
    thin = _pilot003_answers()["0001"]
    calls = []
    client, _ = _client(_scripted_chat([thin["raw_text"], "{}"], calls), max_llm_calls=1)
    outcome = _call(client, thin["evidence_ids"])
    assert len(calls) == 1
    assert outcome.ok and outcome.record.answer_quality == "thin_not_retried"


def test_new_acceptance_745c_b2_a_failed_re_ask_keeps_the_first_thin_answer():
    thin = _pilot003_answers()["0003"]
    calls = []
    client, _ = _client(_scripted_chat([thin["raw_text"], "not json at all"], calls))
    outcome = _call(client, thin["evidence_ids"])
    assert len(calls) == 2
    assert outcome.ok and outcome.payload == json.loads(thin["raw_text"])
    assert outcome.record.answer_quality == "thin_after_retry"
    assert outcome.record.response_hash == "sha256:1"
    assert outcome.record.response_chars == len(thin["raw_text"])


# ---------------------------------------------------------------------------
# F9: the mining CLI loads the project .env before anything reads the environment
# ---------------------------------------------------------------------------
def _spy_dotenv(monkeypatch, events):
    monkeypatch.setattr("t2pw.config.ensure_dotenv_loaded", lambda: events.append("dotenv"))


def test_g9_745c_f9_preflight_loads_dotenv_before_it_reads_the_environment(tmp_path, monkeypatch):
    """With --enable-pathway-mining, mining_enabled() never loads .env, so a live
    preflight said LLM_PROVIDER was unset. At f9556288 ensure_dotenv_loaded is never
    called on this path (events == ["preflight"])."""
    from t2pw.mining.cli import main

    events = []
    _spy_dotenv(monkeypatch, events)

    def fake_preflight(config, **kwargs):  # noqa: ARG001
        events.append("preflight")
        return SimpleNamespace(ok=True, render=lambda: "", to_dict=lambda: {})

    monkeypatch.setattr("t2pw.mining.preflight.run_preflight", fake_preflight)
    code = main(["--enable-pathway-mining", "preflight", "--species", H.TARGET_SPECIES,
                 "--runs-dir", str(tmp_path / "runs"), "--no-probe"])
    assert code == 0
    assert events == ["dotenv", "preflight"]


def test_g9_745c_f9_run_and_resume_load_dotenv_before_their_configuration(tmp_path, monkeypatch):
    """The same defect on the run and resume paths. At f9556288 neither loads .env
    when the flag is passed (events == ["configured", "configured"])."""
    import pytest

    from t2pw.mining import cli
    from t2pw.mining.controller import MiningController

    events = []
    _spy_dotenv(monkeypatch, events)

    def stop(*_a, **_k):
        events.append("configured")
        raise SystemExit("stop here")

    monkeypatch.setattr(cli, "_make_config", stop)
    monkeypatch.setattr(MiningController, "resume", classmethod(lambda cls, *a, **k: stop()))
    with pytest.raises(SystemExit):
        cli.main(["--enable-pathway-mining", "run", "--species", H.TARGET_SPECIES,
                  "--runs-dir", str(tmp_path / "runs")])
    with pytest.raises(SystemExit):
        cli.main(["--enable-pathway-mining", "resume", str(tmp_path / "runs" / "r")])
    assert events == ["dotenv", "configured", "dotenv", "configured"]
    # status, report and list read only files: no .env.
    events.clear()
    cli.main(["list", "--runs-dir", str(tmp_path / "none")])
    assert events == []
