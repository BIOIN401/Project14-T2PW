"""ORCH-745 change set B: reading the claim's own evidence (F-5, F-1, B3, F-6, N-5).

Everything here runs on what the pilots actually recorded: pilot-003's 16 extraction
calls (the evidence ids each was shown and the raw text the model returned) and
pilot-002's 13-LOX claim, in ``tests/fixtures/mining/orch745b_pilot_payloads.json``,
with span texts from change set A's ``orch745a_pilot_texts.json``. The payloads are
REPLAYED through ``extraction.build_claims``; a few T-rows edit one claim the way the
pwml-bio-auditor's adjudication of F-1 describes (its section 3), to show the rule
holds when the model fills what pilot-003's did not.

* ``test_g9_745b_*`` fail behaviourally at f9556288 (a wrong value through a code path
  that exists there; each docstring says what base returns) and at 11032e8d, the
  provisional A+C merge this change set starts from. They import only what exists at
  base, at module level.
* ``test_new_acceptance_745b_*`` pin the adjudication's guard rows (T1b, T4-T11, #33),
  the full replay, and new fields. They need not fail at base; they must fail against a
  literal implementation of the first F-1 proposal.

No network and no model: the model is a recorded payload.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.acquisition import OfflinePaperSource  # noqa: E402
from t2pw.mining.admission import ADMITTED_STATUSES, judge  # noqa: E402
from t2pw.mining.claims import ClaimStore  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402
from t2pw.mining.extraction import build_claims  # noqa: E402
from t2pw.mining.models import EvidenceSpan, ExtractedClaim, PaperRecord  # noqa: E402
from t2pw.mining.normalize import EntityRegistry  # noqa: E402
from t2pw.rag.acquire import CandidatePaper  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "mining"
TEXTS = json.loads((FIXTURES / "orch745a_pilot_texts.json").read_text(encoding="utf-8"))
RECORDED = json.loads((FIXTURES / "orch745b_pilot_payloads.json").read_text(encoding="utf-8"))
PAPERS = {p["paper_id"]: p for p in TEXTS["pilot003"]["papers"]}
ROLE = {s["evidence_id"]: s["role"] for s in TEXTS["pilot003"]["spans"]}
CALLS = {c["call_id"]: c for c in RECORDED["pilot003"]["calls"]}
TYPES = RECORDED["pilot003"]["evidence_types"]
ADJUDICATION = RECORDED["pilot003"]["adjudication"]
TARGET = "arabidopsis thaliana"
BRYOPHYTE, GH3_10, LOX13, PEPPER = "PMC:PMC13364333", "PMC:PMC12096059", "PMC:PMC8508710", "PMC:PMC12787732"
TSS, IVO, AMBIGUOUS = "target_species_supported", "in_vitro_or_cell_free", "species_ambiguous"


def _span_text(row):
    paper = PAPERS[row["paper_id"]]
    return paper["abstract"] if row["char_start"] is None else paper["full_text"][row["char_start"]:row["char_end"]]


#: Every span pilot-003 minted, rebuilt from change set A's fixture (text exact).
SPANS = {
    row["evidence_id"]: EvidenceSpan(
        evidence_id=row["evidence_id"], paper_id=row["paper_id"], text=_span_text(row),
        section=row["locator"].split("#")[0], locator=row["locator"],
        char_start=row["char_start"], char_end=row["char_end"])
    for row in TEXTS["pilot003"]["spans"]
}


def _call(n):
    return CALLS[f"mc_001_extract_claims_{n:04d}"]


def _shown(call, *, every_span=False, extra=()):
    """The spans the call was shown. By default only those change set A still mints: a
    reference list, a journal's front matter or a publisher note is never a span now."""
    ids = [e for e in call["shown"] if every_span or ROLE[e] == "prose"]
    return [SPANS[e] for e in ids] + [SPANS[e] if isinstance(e, str) else e for e in extra]


def _payload(call, rows=None):
    payload = json.loads(call["raw_text"])
    if rows is not None:
        payload["claims"] = [payload["claims"][i] for i in rows]
    return payload


def _build(payload, spans, paper_id, **kwargs):
    observed = PAPERS[paper_id]["observed_organisms"] if paper_id in PAPERS else []
    return build_claims(payload, spans=spans, paper_id=paper_id, target_species_normalized=TARGET,
                        round_index=1, model="recorded", paper_observed_organisms=observed, **kwargs)


def _typed(paper_id):
    """``paper_evidence_type`` for a new-acceptance test (the parameter is new)."""
    return {"paper_evidence_type": TYPES.get(paper_id, "primary")}


def _judge(claims, spans, *, evidence_type="primary"):
    """Register and judge each claim as the controller does. Returns (store, decisions)."""
    store, registry = ClaimStore(), EntityRegistry()
    evidence = {s.evidence_id: s for s in spans}
    papers = {c.paper_id: PaperRecord(paper_id=c.paper_id, pmcid=c.paper_id.split(":")[-1],
                                      evidence_type=evidence_type) for c in claims}
    decisions = []
    for claim in claims:
        stored, _ = store.register(claim)
        outcome = judge(stored, store=store, registry=registry, papers=papers, evidence=evidence,
                        round_index=1)
        decisions.append(store.record_decision(outcome.decision))
    return store, decisions


def _graded(claim, eid, text):
    """The grades this claim's reading gave the supporting sentence(s) of ``eid`` containing ``text``."""
    return [s["grade"] for s in claim.species_support["sentences"]
            if s["evidence_id"] == eid and text in s["text"]]


# ===========================================================================
# F-5 (IT2-N1): participants, placeholders and tissue need evidence
# ===========================================================================
def test_g9_745b_f5_uncited_participants_absent_from_the_cited_spans_are_not_kept():
    """Pilot-003 #24 (call 0003, a thin answer): no participant cites anything, and none
    of "α-linolenic acid (LeA)", "13(S)-hydroperoxy-...acid (13-HPOT)", "12-oxo-
    phytodienoic acid (12-OPDA)" occurs verbatim in the cited ev_6915 / ev_bb4c. Base keeps
    all three (substrates ['α-linolenic acid (LeA)']) and the claim is a ``candidate``; now
    none is kept and the claim is ``incomplete``, never a reaction."""
    call = _call(3)
    claims, dropped = _build(_payload(call), _shown(call), LOX13)
    claim = claims[0]
    assert claim.substrates == [] and claim.products == []
    store, decisions = _judge(claims, _shown(call))
    assert decisions[0].status == "incomplete"
    assert sum("no valid citation and not verbatim" in d for d in dropped) == 3


def test_g9_745b_f5_a_placeholder_echo_is_never_stored_as_a_value():
    """Pilot-003 stored ``"stoichiometry": "products_evidence_ids_placeholder"`` (call 0014)
    and ``"has_stoichiometry"`` (call 0003). Base keeps both strings; now a placeholder is
    no value, and a stoichiometry must be a coefficient."""
    call = _call(14)
    claims, dropped = _build(_payload(call), _shown(call), PEPPER)
    assert [p.stoichiometry for c in claims for p in c.products] == ["", ""]
    assert sum("placeholder" in d for d in dropped) == 2

    cited = _payload(_call(3))  # the same answer, its participants cited to the shown spans
    for side in ("substrates", "products"):
        for participant in cited["claims"][0][side]:
            participant["evidence_ids"] = ["ev_bb4c9802d1637278"]
    claims, _ = _build(cited, _shown(_call(3)), LOX13)
    assert {p.stoichiometry for p in (*claims[0].substrates, *claims[0].products)} == {""}


def test_g9_745b_f5_an_uncited_tissue_is_dropped():
    """Pilot-003 #11 (call 0002) stored ``tissue: "leaf, root, inflorescence, wounded leaf"``,
    which no cited passage says. Base keeps it; now it is dropped like any uncited field."""
    call = _call(2)
    claims, dropped = _build(_payload(call), _shown(call), LOX13)
    assert claims[0].tissue == ""
    assert "tissue" not in claims[0].field_evidence
    assert any(".tissue='leaf, root, inflorescence, wounded leaf'" in d for d in dropped)


def test_new_acceptance_745b_f5_a_participant_records_how_it_was_kept():
    call = _call(2)  # thin: LeA and 13-HPOT uncited, both verbatim in ev_baf1
    claims, _ = _build(_payload(call), _shown(call), LOX13)
    for participant in (*claims[0].substrates, *claims[0].products):
        assert participant.attribution == "verbatim_cited_span"
        assert len(participant.evidence_ids) == 1
    claims, _ = _build(_payload(_call(9), [0]), _shown(_call(9)), GH3_10)
    assert {p.attribution for p in (*claims[0].substrates, *claims[0].products)} == {"model_citation"}
    # The label is persisted with the participant (claims.jsonl).
    assert claims[0].to_dict()["substrates"][0]["attribution"] == "model_citation"


def test_new_acceptance_745b_f5_a_tissue_a_cited_span_states_is_kept_and_attributed():
    call = _call(1)  # "plastids": ev_5863 [3] "...13-LOX localization to plastids."
    claims, _ = _build(_payload(call), _shown(call), LOX13)
    claim = claims[0]
    assert claim.tissue == "plastids"
    assert claim.field_attribution["tissue"] == "verbatim_cited_span"
    assert claim.field_evidence["tissue"] == ["ev_5863efff93736770"]


def test_new_acceptance_745b_f5_admission_strips_an_unsupported_tissue():
    from t2pw.mining.admission import STRIPPABLE_FIELDS

    assert "tissue" in STRIPPABLE_FIELDS
    call = _call(9)
    claims, _ = _build(_payload(call, [0]), _shown(call), GH3_10)
    claims[0].tissue = "rosette"  # set after extraction, with no span behind it
    _, decisions = _judge(claims, _shown(call))
    assert claims[0].tissue == ""


def test_new_acceptance_745b_f5_an_all_unsupported_claim_is_a_separate_incomplete_lead():
    from t2pw.mining import frontier
    from t2pw.mining.extraction import UNSUPPORTED_PARTICIPANTS_NAMESPACE
    from t2pw.mining.graph import ResearchGraph
    from t2pw.mining.models import FRONTIER_UPSTREAM, GraphDelta

    call = _call(3)
    claims, _ = _build(_payload(call), _shown(call), LOX13)
    claim = claims[0]
    assert claim.claim_identity.startswith(f"{UNSUPPORTED_PARTICIPANTS_NAMESPACE}::")
    # The asserted names stay visible: on the claim and in its unsupported list.
    assert claim.subject == "α-linolenic acid (LeA)"
    assert "12-oxo-phytodienoic acid (12-OPDA)" in claim.object
    assert [u["side"] for u in claim.unsupported_participants] == ["substrates", "products", "products"]
    store, decisions = _judge(claims, _shown(call))
    assert decisions[0].status == "incomplete"
    config = H.make_config(Path("."))
    items = frontier.generate(graph=ResearchGraph(registry=EntityRegistry()), store=store,
                              delta=GraphDelta(round_index=1), config=config, round_index=1)
    upstream = [i for i in items if i.kind == FRONTIER_UPSTREAM and i.claim_identity == claim.claim_identity]
    assert upstream and "α-linolenic acid (LeA)" in upstream[0].target
