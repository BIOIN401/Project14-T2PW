"""ORCH-743 item 2: enzyme / gene / species values the model filled but did not cite.

Pilot-001 dropped 59 of 59 such values although every one occurs in a passage the
model was shown. The replay (docs/pathway_mining/evidence/013-...) showed why:
the model filled ``enzyme`` / ``gene`` / ``species`` and cited only the claim-level
``evidence_ids`` and each participant's ``evidence_ids``; it never emitted a
per-field ``*_evidence_ids`` key.

Two fixes, both tested here:

* the request now names every per-field citation key and shows the shape;
* a value that occurs verbatim in one of THAT claim's own cited participant spans
  is attributed to that span and recorded as ``verbatim_cited_span``.

G9 correction: the tests marked ``G9`` fail behaviourally at 23e1ae83 (the value
is dropped, the reaction never reaches the target graph, the request never
names the keys). The rest pin the invariants the fallback must not break.
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
from t2pw.mining.artifacts import read_jsonl  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402
from t2pw.mining.extraction import build_claims  # noqa: E402
from t2pw.mining.models import EvidenceSpan  # noqa: E402

TARGET = "arabidopsis thaliana"
VERBATIM = "verbatim_cited_span"


def _span(eid, text):
    return EvidenceSpan(evidence_id=eid, paper_id="PMC1", text=text, section="results",
                        locator=f"results#{eid}", chunk_id=eid)


SHOWN = [
    _span("ev_one", "In Arabidopsis   thaliana leaves, LOX2 (At3g45140) converts "
                    "alpha-linolenic acid to 13-HPOT."),
    _span("ev_two", "Zea mays ZmLOX10 is unrelated to this step."),
]


def _pilot_shaped(**extra):
    """What the replay's model returned: values filled, no per-field *_evidence_ids."""
    row = {
        "substrates": [{"name": "alpha-linolenic acid", "evidence_ids": ["ev_one"]}],
        "products": [{"name": "13-HPOT", "evidence_ids": ["ev_one"]}],
        "enzyme": "LOX2",
        "gene": "at3g45140",
        "species": "Arabidopsis thaliana",
        "species_inferred": False,
        "hypothesis": False,
        "evidence_ids": ["ev_one"],
    }
    row.update(extra)
    return {"claims": [row]}


def _build(payload, **kwargs):
    return build_claims(payload, spans=SHOWN, paper_id="PMC1", target_species_normalized=TARGET,
                        round_index=1, model="test/model", **kwargs)


# ---------------------------------------------------------------------------
# G9: the pilot-001 shape keeps its verifiable fields
# ---------------------------------------------------------------------------
def test_g9_uncited_values_found_verbatim_in_a_cited_participant_span_are_kept():
    claims, dropped = _build(_pilot_shaped())
    claim = claims[0]
    assert (claim.enzyme, claim.gene, claim.species_text) == ("LOX2", "at3g45140", "Arabidopsis thaliana")
    assert dropped == []
    for name in ("enzyme", "gene", "species"):
        assert claim.evidence_for(name) == ["ev_one"]
        assert claim.field_attribution[name] == VERBATIM
    # The recovered species evidence flows into the unchanged species guardrail.
    assert claim.species_state == "target_species_supported"
    assert claim.unsupported_fields() == []


def test_g9_a_pilot_shaped_model_now_yields_a_target_species_reaction(tmp_path):
    """End to end through the scripted controller: the model cites participants only."""
    scripted = H.ScriptedModel()
    systems = []

    def chat(messages, **kwargs):
        systems.append("".join(str(m.get("content") or "") for m in messages if m.get("role") == "system"))
        result = scripted(messages, **kwargs)
        payload = json.loads(result.text)
        for row in payload.get("claims") or []:
            for key in [k for k in row if k.endswith("_evidence_ids")]:
                del row[key]
        return SimpleNamespace(text=json.dumps(payload), diagnostics=result.diagnostics)

    config = H.make_config(tmp_path, budgets={"max_rounds": 1})
    source = H.ScriptedPaperSource()
    controller = MiningController(config, source=source, chat_fn=chat,
                                  index=H.offline_index(config, tmp_path), enabled_override=True)
    outcome = controller.run(probe_model=False, probe_embeddings=False)

    graph = outcome.state.graph
    lox = next(r for r in graph.reactions.values() if "alpha-linolenic acid" in r.substrate_names)
    assert lox.admitted and lox in graph.primary_reactions
    maize = next(r for r in graph.reactions.values() if "linoleic acid" in r.substrate_names)
    assert maize.species_state == "non_target_species" and not maize.admitted

    rows = read_jsonl(outcome.run_dir.root / "claims.jsonl")
    lox_claim = next(r for r in rows if r["enzyme"] == "LOX2")
    assert lox_claim["field_attribution"]["enzyme"] == VERBATIM
    assert lox_claim["field_evidence"]["enzyme"] == lox_claim["substrates"][0]["evidence_ids"][:1]
    # The request itself now names every per-field citation key.
    for key in ("enzyme_evidence_ids", "gene_evidence_ids", "species_evidence_ids"):
        assert all(key in s for s in systems)


# ---------------------------------------------------------------------------
# Invariants the fallback must not break
# ---------------------------------------------------------------------------
def test_an_unattributable_value_is_still_dropped():
    claims, dropped = _build(_pilot_shaped(enzyme="allene oxide synthase", gene="AOS1"))
    assert claims[0].enzyme == "" and claims[0].gene == ""
    assert "enzyme" not in claims[0].field_evidence and "enzyme" not in claims[0].field_attribution
    assert len(dropped) == 2


def test_never_from_a_span_the_model_was_not_shown():
    payload = _pilot_shaped(enzyme="ZmLOX10")
    payload["claims"][0]["substrates"][0]["evidence_ids"] = ["ev_hidden"]
    payload["claims"][0]["products"][0]["evidence_ids"] = ["ev_hidden"]
    claims, dropped = _build(payload)
    assert claims[0].enzyme == "" and claims[0].species_text == ""
    assert any("enzyme='ZmLOX10'" in d for d in dropped)


def test_never_from_a_span_the_claim_does_not_cite_nor_from_paper_metadata():
    # ev_two is shown and names the value, but this claim does not cite it; the
    # claim-level evidence_ids is not a participant citation either.
    payload = _pilot_shaped(enzyme="ZmLOX10", species="Zea mays", evidence_ids=["ev_one", "ev_two"])
    claims, dropped = _build(payload, paper_observed_organisms=["Zea mays"])
    claim = claims[0]
    assert claim.enzyme == "" and claim.species_text == ""
    assert "species" not in claim.field_evidence and "species" not in claim.field_attribution
    assert len(dropped) == 2


def test_an_explicit_empty_or_unshown_citation_is_respected_not_recovered():
    """Only a MISSING list is recovered. An empty list is the model declining to cite
    (the fixture's ambiguous-species P6 case); an unshown id is a failed citation."""
    claims, dropped = _build(_pilot_shaped(species_evidence_ids=[], enzyme_evidence_ids=["ev_hidden"]))
    claim = claims[0]
    assert claim.species_text == "" and claim.enzyme == ""
    assert claim.field_attribution == {"gene": VERBATIM}
    assert len(dropped) == 2


def test_a_short_value_never_matches_inside_a_longer_word():
    claims, dropped = _build(_pilot_shaped(enzyme="OX", gene="LOX"))
    assert claims[0].enzyme == "" and claims[0].gene == ""
    assert len(dropped) == 2


def test_a_model_citation_is_kept_and_named_as_one():
    claims, dropped = _build(_pilot_shaped(enzyme_evidence_ids=["ev_one"],
                                           direction="forward", direction_evidence_ids=["ev_one"]))
    claim = claims[0]
    assert claim.field_attribution["enzyme"] == "model_citation"
    assert claim.field_attribution["direction"] == "model_citation"
    assert claim.field_attribution["species"] == VERBATIM
    assert dropped == []
