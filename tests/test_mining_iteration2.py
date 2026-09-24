"""ORCH-743 iteration 2: the pilot-002 audit's five defects inside the ORCH-743 areas.

Each ``test_g9_*`` node fails behaviourally at ecf6b9a0, the tip pilot-002 ran on. The
other tests pin the new behaviour's edges. No network and no model: every provider, HTTP
client and model is scripted.
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
from t2pw.mining.controller import MiningController  # noqa: E402
from t2pw.mining.models import EvidenceSpan  # noqa: E402

PER_FIELD_LISTS = ("enzyme_evidence_ids", "gene_evidence_ids", "species_evidence_ids",
                   "compartment_evidence_ids", "direction_evidence_ids", "reversible_evidence_ids")


# ---------------------------------------------------------------------------
# IT2-1: extraction in passage batches, with an output budget that fits one
# ---------------------------------------------------------------------------
def _twelve_passages(paper_id, query_text, **kwargs):  # noqa: ARG001
    tag = paper_id.split(":")[-1].lower()
    return [EvidenceSpan(evidence_id=f"ev_{tag}_{n:02d}", paper_id=paper_id, section="results",
                         text=f"Passage {n}: alpha-linolenic acid is oxygenated in Arabidopsis thaliana.")
            for n in range(12)]


def test_g9_it2_1_a_paper_is_extracted_in_batches_with_the_extraction_budget(tmp_path):
    """Pilot-002 sent 20 passages in one call with max_tokens 4000 and a 120 s timeout;
    the hidden reasoning spent the budget and the call came back empty (evidence 032)."""
    calls = []

    def chat(messages, **kwargs):
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        calls.append({"passages": user.count("[evidence_id: "), "max_tokens": kwargs.get("max_tokens"),
                      "timeout": kwargs.get("timeout"), "system": system})
        return SimpleNamespace(text='{"claims": []}', diagnostics=None)

    config = H.make_config(tmp_path, budgets={"max_rounds": 1})
    index = H.offline_index(config, tmp_path)
    index.spans_for_paper = _twelve_passages
    controller = MiningController(config, source=H.ScriptedPaperSource({1: ["P1"]}), chat_fn=chat,
                                  index=index, enabled_override=True)
    outcome = controller.run(probe_model=False, probe_embeddings=False)

    assert [c["passages"] for c in calls] == [5, 5, 2]
    assert {c["max_tokens"] for c in calls} == {12000}
    assert {c["timeout"] for c in calls} == {360.0}
    assert len([r for r in outcome.state.papers.values() if r.processed_round == 1]) == 1
    assert outcome.ledger.usage.papers_processed == 1
    # The prompt is unchanged: every per-field citation list is still demanded.
    for call in calls:
        assert all(key in call["system"] for key in PER_FIELD_LISTS)


def test_the_extraction_settings_and_prompt_shape_are_pinned():
    from t2pw.mining.config import MiningBudgets, MiningModelConfig
    from t2pw.mining.extraction import SYSTEM_PROMPT

    model = MiningModelConfig()
    assert (model.max_tokens, model.request_timeout_seconds, model.prompt_version) == (
        12000, 360.0, "mining-extract/1.1.0")
    assert MiningBudgets().max_passages_per_call == 5
    assert all(key in SYSTEM_PROMPT for key in PER_FIELD_LISTS)
    assert "a filled\n   field whose own list is missing or empty will be discarded" in SYSTEM_PROMPT


def test_the_batch_knob_is_honoured(tmp_path):
    seen = []

    def chat(messages, **kwargs):  # noqa: ARG001
        seen.append(next(m["content"] for m in messages if m["role"] == "user").count("[evidence_id: "))
        return SimpleNamespace(text='{"claims": []}', diagnostics=None)

    config = H.make_config(tmp_path, budgets={"max_rounds": 1, "max_passages_per_call": 20})
    index = H.offline_index(config, tmp_path)
    index.spans_for_paper = _twelve_passages
    MiningController(config, source=H.ScriptedPaperSource({1: ["P1"]}), chat_fn=chat, index=index,
                     enabled_override=True).run(probe_model=False, probe_embeddings=False)
    assert seen == [12]


# ---------------------------------------------------------------------------
# IT2-5: every ladder rung, with its fraction and passage count, in model_raw.jsonl
# ---------------------------------------------------------------------------
def _diag(finish_reason, requests):
    return SimpleNamespace(
        finish_reason=finish_reason, attempts=requests, model="scripted", request_hash="req",
        response_hash="res", attempt_log=[{"attempt": n + 1, "finish_reason": finish_reason,
                                           "status": "empty" if finish_reason == "length" else "ok"}
                                          for n in range(requests)])


def test_g9_it2_5_every_ladder_rung_is_recorded_with_its_fraction_and_passages(tmp_path):
    """Pilot-002 overwrote attempt_log per rung (3 of _0002's 9 requests survived) and
    model_raw.jsonl carried no shrink fraction."""
    from t2pw.mining.artifacts import read_jsonl

    empties = []

    def chat(messages, **kwargs):  # noqa: ARG001
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "Lipoxygenase LOX2 oxygenates" in user and len(empties) < 2:
            empties.append(1)
            return SimpleNamespace(text="", diagnostics=_diag("length", 3))
        return SimpleNamespace(text='{"claims": []}', diagnostics=_diag("stop", 1))

    config = H.make_config(tmp_path, budgets={"max_rounds": 1})
    outcome = MiningController(config, source=H.ScriptedPaperSource({1: ["P1"]}), chat_fn=chat,
                               index=H.offline_index(config, tmp_path),
                               enabled_override=True).run(probe_model=False, probe_embeddings=False)
    root = outcome.run_dir.root
    rows = [r for r in read_jsonl(root / "rounds" / "round_001" / "model_raw.jsonl")
            if r.get("paper_id") == "PMC:PMC1000001"]
    call_ids = {r["call_id"] for r in rows}
    assert len(call_ids) == 1 and len(rows) == 3
    assert [r.get("shrink_fraction") for r in rows] == [1.0, 0.6, 0.35]
    sent = [r.get("passages_sent") for r in rows]
    assert sent[0] >= 1 and sent[0] >= sent[1] >= sent[2]
    assert [r.get("schema") for r in rows] == ["full", "full", "tight"]
    assert [len(r.get("provider_requests") or []) for r in rows] == [3, 3, 1]

    record = next(r for r in read_jsonl(root / "model_calls.jsonl") if r["call_id"] in call_ids)
    assert [e.get("rung") for e in record["attempt_log"]] == [1, 1, 1, 2, 2, 2, 3]


# ---------------------------------------------------------------------------
# IT2-4: the grape host
# ---------------------------------------------------------------------------
def _screen_config(tmp_path):
    from t2pw.mining.config import MiningConfig, StartingInput

    seeds = [StartingInput(kind="seed_metabolite", raw="12-OPDA", value="12-OPDA"),
             StartingInput(kind="seed_enzyme", raw="AOS", value="AOS")]
    return MiningConfig.build(
        target_species="Arabidopsis thaliana", topic="jasmonate biosynthesis", starting_inputs=seeds,
        overrides={"runs_dir": str(tmp_path / "runs"), "offline": True, "screen_papers": True,
                   "enabled_via": "test"},
    )


def test_g9_it2_4_route1_does_not_rescue_a_grape_host_paper(tmp_path):
    """Pilot-002's PMC5104190: grape VaPAT1, 'ectopic expression ... in Arabidopsis'. The
    title says 'grape', which neither title detector knew."""
    from t2pw.mining.acquisition import Acquirer, OfflinePaperSource
    from t2pw.mining.models import LiteratureQuery
    from t2pw.rag.acquire import CandidatePaper

    paper = json.loads((ROOT / "tests" / "fixtures" / "mining" / "pilot002_grape_host.json")
                       .read_text(encoding="utf-8"))["papers"][0]
    candidate = CandidatePaper(id=paper["candidate_id"], source="fixture", title=paper["title"],
                               abstract=paper["abstract"], full_text=paper["abstract"])
    config = _screen_config(tmp_path)
    acquirer = Acquirer(config, source=OfflinePaperSource([candidate]))
    decision = acquirer.screen(candidate)
    assert (decision.classification, decision.organism_match, decision.score) == ("omics_only", "match", 3.25)
    assert acquirer.rescue(candidate, decision, ["12-OPDA", "AOS"]) == ""
    result = acquirer.acquire(LiteratureQuery(query_id="qr_fixture", query_text="fixture"), {},
                              round_index=2, max_papers=1, run_terms=["12-OPDA", "AOS"])
    assert [p.record.paper_id for p in result.papers] == []
