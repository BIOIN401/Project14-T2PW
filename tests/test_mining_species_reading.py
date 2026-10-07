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


# ===========================================================================
# F-1: the species markers are read on the claim's own supporting sentences
# ===========================================================================
def test_g9_745b_f1_t1_gh3_10_jasmonic_acid_to_ja_ile_is_admitted_on_its_knockout_evidence():
    """T1, pilot-003 #14 (call 0009 row 0): "In flowers and wounded leaves of gh3.10 jar1
    knockout lines JA-Ile was below the detection limit." (ev_7d27 [7]). Base reads the whole
    abstract, finds "in vitro" in [5], and stores ``in_vitro_or_cell_free``; now the claim is
    ``target_species_supported`` on the knockout sentence, graded ``in_planta``, and admitted
    with a primary paper's evidence level."""
    call = _call(9)
    claims, _ = _build(_payload(call, [0]), _shown(call), GH3_10)
    claim = claims[0]
    assert claim.species_state == TSS
    assert claim.claim_id == "cl_b89be46f92d4c4e1"
    assert claim.species_support["grade"] == "in_planta"
    assert _graded(claim, "ev_7d2787cbb14b4905", "knockout lines JA-Ile was below") == ["in_planta"]
    _, decisions = _judge(claims, _shown(call))
    assert decisions[0].status in ADMITTED_STATUSES
    assert decisions[0].evidence_level == "single_paper"


def _bryophyte_run(tmp_path):
    """Pilot-003's bryophyte review through the controller: the paper is acquired and graded
    as change set A grades it (its introduction says "this review makes four specific
    contributions"), retrieval returns the passages call 0006 was shown, and the model
    returns call 0006's recorded answer."""
    paper = PAPERS[BRYOPHYTE]
    candidate = CandidatePaper(id=paper["pmcid"], source="europepmc", title=paper["title"],
                               abstract=paper["abstract"], full_text=paper["full_text"],
                               source_uri=paper["url"], year=paper["year"])
    call = _call(6)
    config = H.make_config(tmp_path, budgets={"max_rounds": 1})
    index = H.offline_index(config, tmp_path)
    index.spans_for_paper = lambda pid, text, **kw: [  # noqa: ARG005
        dataclasses.replace(s, paper_id=pid) for s in _shown(call)]
    controller = MiningController(
        config, source=OfflinePaperSource([candidate]), index=index, enabled_override=True,
        chat_fn=lambda *a, **k: SimpleNamespace(text=call["raw_text"], diagnostics=None))
    return controller.run(probe_model=False, probe_embeddings=False)


def test_g9_745b_f1_t2_t3_opr3_steps_of_the_bryophyte_review_are_review_grade(tmp_path):
    """T2 (#34 OPDA -> OPC-8:0) and T3 (#3 dn-cis-OPDA -> OPC-6:0), OPR3, cited to ev_c98d:
    "In Arabidopsis, OPDA and dn- cis -OPDA are reduced by OPR3 in the peroxisomes to produce
    OPC-8:0 and OPC-6:0, respectively." ([12]). Base reads "orthologs" / "orthologous"
    elsewhere in the 5,700-character span and stores ``inferred_only`` for both; now both are
    ``target_species_supported`` with grade ``review`` and ``review_article_only``, never
    ``in_planta``, and the Marchantia knockout sentences [27], [30] give no credit."""
    outcome = _bryophyte_run(tmp_path)
    claims = {c.claim_id: c for c in outcome.state.store.claims.values()}
    for claim_id in ("cl_0bc93d248b7f31db", "cl_5338b4fc07a47cea"):
        claim = claims[claim_id]
        assert claim.species_state == TSS
        assert claim.species_support["grade"] == "review"
        assert "in_planta" not in {s["grade"] for s in claim.species_support["sentences"]}
        assert _graded(claim, "ev_c98d84c7d84855c1", "are reduced by OPR3") == ["review"]
        decision = outcome.state.store.history(claim.claim_identity).current
        assert decision.status in ADMITTED_STATUSES
        assert decision.evidence_level == "review_article_only"
    marchantia = claims["cl_0bc93d248b7f31db"]
    assert _graded(marchantia, "ev_c98d84c7d84855c1", "Genetic knockout of Mp FAD5") == ["none"]
    assert _graded(marchantia, "ev_c98d84c7d84855c1", "Mp fad5 mutant plants") == ["none"]


def test_new_acceptance_745b_f1_the_grade_reaches_claims_decisions_and_the_reaction_node(tmp_path):
    outcome = _bryophyte_run(tmp_path)
    assert outcome.state.papers[BRYOPHYTE].evidence_type == "review"
    root = outcome.run_dir.root
    claims = {json.loads(l)["claim_id"]: json.loads(l)
              for l in (root / "claims.jsonl").read_text(encoding="utf-8").splitlines()}
    assert claims["cl_0bc93d248b7f31db"]["species_support"]["grade"] == "review"
    assert claims["cl_0bc93d248b7f31db"]["species_support"]["paper_evidence_type"] == "review"
    decisions = [json.loads(l) for l in (root / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {d["species_grade"] for d in decisions if d["claim_id"] == "cl_0bc93d248b7f31db"} == {"review"}
    primary = json.loads((root / "primary_graph.json").read_text(encoding="utf-8"))
    candidates = json.loads((root / "candidate_subgraphs.json").read_text(encoding="utf-8"))
    nodes = [*primary["reactions"], *(r for c in candidates["components"] for r in c["reactions"])]
    grades = {" + ".join(r["product_names"]): r["species_grade"] for r in nodes}
    assert grades["OPC-8:0"] == grades["OPC-6:0"] == "review"
    assert all("species_grade" in r for r in nodes)


def test_new_acceptance_745b_f1_t1b_without_the_knockout_sentence_there_is_no_credit():
    """T1b: [7] removed from ev_7d27. No credit from [9] "...in the absence of JA-Ile..." or
    [2] "gh3.10 jar1 knockout lines are still fertile"."""
    call = _call(9)
    stripped = dataclasses.replace(SPANS["ev_7d2787cbb14b4905"], text=SPANS["ev_7d2787cbb14b4905"].text.replace(
        "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit. ", ""))
    assert "below the detection limit" not in stripped.text
    spans = [stripped if s.evidence_id == stripped.evidence_id else s for s in _shown(call)]
    claims, _ = _build(_payload(call, [0]), spans, GH3_10, **_typed(GH3_10))
    claim = claims[0]
    assert claim.species_state != TSS
    assert claim.species_state == IVO
    assert set(_graded(claim, stripped.evidence_id, "in the absence of JA-Ile")) == {"none"}
    assert _graded(claim, stripped.evidence_id, "knockout lines are still fertile") == ["none"]


def test_new_acceptance_745b_f1_t4_pilot002_13lox_is_in_vitro():
    """T4: pilot-002's only primary reaction, LeA -> 13-HPOT (13-LOX), cited to the 13-LOX
    abstract ev_b58e: "LOX-catalyzed LeA oxidation ... Km value of 5.8 µM"."""
    record = RECORDED["pilot002"]
    span = next(EvidenceSpan(**s) for s in record["spans"] if s["evidence_id"] == "ev_b58e2cb64e2f1d87")
    claims, _ = build_claims(json.loads(record["call"]["raw_text"]), spans=[span],
                             paper_id=record["paper"]["paper_id"], target_species_normalized=TARGET,
                             round_index=1, model="recorded",
                             paper_observed_organisms=record["paper"]["observed_organisms"],
                             paper_evidence_type="primary")
    claim = claims[0]
    assert claim.species_state == IVO
    assert _graded(claim, span.evidence_id, "Km value of 5.8") == ["in_vitro"]


def test_new_acceptance_745b_f1_t5_recombinant_13lox_with_a_cited_species_stays_out():
    """T5: #11 (call 0002) with species "Arabidopsis thaliana" cited to the abstract ev_5863:
    in vitro from ev_baf1 [8] "...when incubated with LA or LeA"; no credit from the pea
    tissue extract (ev_4dcc [3]), others' work (ev_baf1 [5]) or hedges (ev_baf1 [9], [10])."""
    call = _call(2)
    payload = _payload(call)
    payload["claims"][0].update(species="Arabidopsis thaliana", species_evidence_ids=["ev_5863efff93736770"])
    claims, _ = _build(payload, _shown(call, extra=["ev_5863efff93736770"]), LOX13, **_typed(LOX13))
    claim = claims[0]
    assert claim.species_state == IVO
    assert "in_planta" not in {s["grade"] for s in claim.species_support["sentences"]}
    assert _graded(claim, "ev_baf1514923696f5c", "when incubated with LA or LeA") == ["in_vitro"]
    assert _graded(claim, "ev_4dcc372e77b84976", "injection of plant tissue extract") == ["in_vitro"]
    for text in ("who observed that Cd 2+ increased the content of 13-HPOT", "This might imply",
                 "The results support the view"):
        grades = _graded(claim, "ev_baf1514923696f5c", text)
        assert grades and "in_planta" not in grades, text


def test_new_acceptance_745b_f1_t6_the_collapsed_13lox_cascade_with_a_cited_species_stays_out():
    """T6: #24 (call 0003) with species cited to ev_6915 (and, so that its participants are
    kept and could earn credit, each participant cited to ev_6915 / ev_bb4c): in vitro from
    ev_bb4c [15] "...with 13-LOX-, AOS- and AOC-lysates"; no credit from the pea roots ex
    vivo method (ev_bb4c [26], [27])."""
    call = _call(3)
    payload = _payload(call)
    row = payload["claims"][0]
    row.update(species="Arabidopsis thaliana", species_evidence_ids=["ev_6915f534ca2dfd74"])
    for side in ("substrates", "products"):
        for participant in row[side]:
            participant["evidence_ids"] = ["ev_6915f534ca2dfd74", "ev_bb4c9802d1637278"]
    claims, _ = _build(payload, _shown(call), LOX13, **_typed(LOX13))
    claim = claims[0]
    assert claim.substrates and claim.products
    assert claim.species_state == IVO
    assert "in_planta" not in {s["grade"] for s in claim.species_support["sentences"]}
    assert _graded(claim, "ev_bb4c9802d1637278", "AOS- and AOC-lysates") == ["in_vitro"]
    assert set(_graded(claim, "ev_bb4c9802d1637278", "we employed pea roots")) <= {"in_vitro"}
    assert set(_graded(claim, "ev_bb4c9802d1637278", "injection of plant tissue extract")) <= {"in_vitro"}


def test_new_acceptance_745b_f1_t7_gh3_10_conversion_and_mutant_in_different_clauses_stay_out():
    """T7: #30 (call 0010) with ev_1f2c reduced to its text from "Abstract" onward, U+2010
    normalised. [17] "12-hydroxy jasmonate is directly converted into 12-hydroxy
    jasmonoyl-isoleucine and fertile jasmonoyl-isoleucine deficient Arabidopsis mutants point
    towards..." -- the mutants are another clause's subject."""
    call = _call(10)
    full = SPANS["ev_1f2c624a181ba5ef"]
    text = full.text[full.text.index("Abstract"):].replace("‐", "-")
    reduced = dataclasses.replace(full, text=text)
    claims, _ = _build(_payload(call), _shown(call, extra=[reduced]), GH3_10, **_typed(GH3_10))
    claim = claims[0]
    assert claim.species_state != TSS
    assert claim.species_state == IVO
    assert _graded(claim, reduced.evidence_id, "is directly converted into") == ["none"]


@pytest.mark.parametrize("every_span", [False, True], ids=["as_change_set_A_mints", "as_pilot003_held"])
def test_new_acceptance_745b_f1_t8_gh3_10_purified_enzyme_stays_out(every_span):
    """T8: #31 (call 0011), JA + Ile + ATP -> JA-Ile: in vitro from ev_c7a1 [7] "...purified in
    parallel and tested for enzyme activity by in vitro incubations...". The species was cited
    only to ev_ae0d, which is the paper's reference list: change set A never mints it, and
    even shown as pilot-003 showed it, none of its text is positive evidence."""
    call = _call(11)
    claims, _ = _build(_payload(call, [0]), _shown(call, every_span=every_span), GH3_10, **_typed(GH3_10))
    claim = claims[0]
    assert claim.species_state == IVO
    assert _graded(claim, "ev_c7a1b8c0ccdfe997", "were purified in parallel") == ["in_vitro"]
    assert "in_planta" not in {s["grade"] for s in claim.species_support["sentences"]}
    if not every_span:
        assert "ev_ae0d92db16547d2c" not in claim.evidence_ids
        assert not [s for s in claim.species_support["sentences"] if s["evidence_id"] == "ev_ae0d92db16547d2c"]


@pytest.mark.parametrize("row,product", [(2, "JA-Gln"), (3, "JA-Asn"), (4, "JA-Glu")])
def test_new_acceptance_745b_f1_t9_ja_conjugates_found_in_wounded_leaves_are_ambiguous(row, product):
    """T9 (#12, #13, #15; product-owner hinge): ev_7d27 [9] "Additional JA conjugates found in
    wounded leaves (JA-Gln) or formed in flowers upon MeJA treatment in the absence of JA-Ile
    (JA-Gln, JA-Asn, JA-Glu) were identified." is occurrence, not conversion."""
    call = _call(9)
    claims, _ = _build(_payload(call, [row]), _shown(call), GH3_10, **_typed(GH3_10))
    claim = claims[0]
    assert claim.object == product
    assert claim.species_state == AMBIGUOUS
    assert _graded(claim, "ev_7d2787cbb14b4905", "Additional JA conjugates") == ["none"]
    # H4: with the substrate written "JA" [9] is still no conversion (and "JA" now also
    # names 12-hydroxy-JA's recombinant sentence [8], loosely: in vitro).
    payload = _payload(call, [row])
    payload["claims"][0]["substrates"][0]["name"] = "JA"
    claims, _ = _build(payload, _shown(call), GH3_10, **_typed(GH3_10))
    assert claims[0].species_state != TSS
    assert _graded(claims[0], "ev_7d2787cbb14b4905", "Additional JA conjugates") == ["none"]


def test_new_acceptance_745b_f1_t10_an_ortholog_sentence_never_transfers_the_species():
    """T10: #29 (call 0005 row 0, 16:0 -> 16:1n9, MpFAD5) with species "Arabidopsis thaliana"
    cited to ev_c98d: "MpFAD5 is orthologous to AtFAD5, an Arabidopsis enzyme known to
    synthesize 16:1n9..." ([28])."""
    call = _call(5)
    payload = _payload(call, [0])
    payload["claims"][0].update(species="Arabidopsis thaliana", species_evidence_ids=["ev_c98d84c7d84855c1"])
    claims, _ = _build(payload, _shown(call, extra=["ev_c98d84c7d84855c1"]), BRYOPHYTE, **_typed(BRYOPHYTE))
    claim = claims[0]
    assert claim.species_state == "inferred_only"
    assert _graded(claim, "ev_c98d84c7d84855c1", "MpFAD5 is orthologous to AtFAD5") == ["inference"]


def test_new_acceptance_745b_f1_t11_an_introduction_only_mutant_tool_earns_nothing():
    """T11: #14 citing only the GH3.10 introduction ev_d216 (species cited to it): [39] "We
    introduce the aos gh3.10 jar1 mutant as a new tool to study jasmonate signalling in the
    absence of JA-Ile..." -- an introduction, a purpose clause, a tool."""
    call = _call(9)
    payload = _payload(call, [0])
    text = json.dumps(payload).replace("ev_7d2787cbb14b4905", "ev_d216d8e2a43cb218")
    claims, _ = _build(json.loads(text), [SPANS["ev_d216d8e2a43cb218"]], GH3_10, **_typed(GH3_10))
    claim = claims[0]
    assert claim.species_state == AMBIGUOUS
    assert _graded(claim, "ev_d216d8e2a43cb218", "We introduce the aos gh3.10 jar1 mutant") == ["none"]


def test_new_acceptance_745b_f1_33_a_marchantia_table_row_is_non_target_not_an_inference():
    """Recommended #33 (call 0005 row 2, HDA (16:2n6) -> HTA (16:3n3), MpFAD7): its row of the
    summary table, ev_20a7 [10], has no ortholog marker; the claim is ``non_target_species``.
    The neighbouring row's "Inferred by pathway logic" ([9]) is about 16:2(n6) (HDA) as a
    product of MpFAD6 and does not name this claim's "HDA (16:2n6)" (a name keeps its note)."""
    call = _call(5)
    claims, _ = _build(_payload(call, [2]), _shown(call), BRYOPHYTE, **_typed(BRYOPHYTE))
    assert claims[0].species_state == "non_target_species"


def test_g9_745b_b3_the_papers_screen_organisms_are_never_the_claims_species():
    """B3, pilot-003 #25 (call 0001): the model gave no species. Base fed the paper's screen
    organisms (Arabidopsis thaliana, and Homo sapiens from "human LOX homologs") in as the
    claim's species and stored ``multiple_species``; now the claim states none, and its own
    sentences say in vitro: ev_5863 [4] "...LeA oxidation ... with a Km value of 5.8 µM"."""
    call = _call(1)
    claims, _ = _build(_payload(call), _shown(call), LOX13)
    claim = claims[0]
    assert claim.species_state == IVO
    assert claim.species_text == ""


def _replay():
    """All 16 recorded calls, in order, through extraction, the claim store and admission,
    as the controller runs them (change set A's spans; each paper graded as A grades it)."""
    store, registry = ClaimStore(), EntityRegistry()
    papers = {pid: PaperRecord(paper_id=pid, evidence_type=TYPES[pid]) for pid in TYPES}
    evidence = dict(SPANS)
    identity_of = {}
    admitted = set()
    for call in RECORDED["pilot003"]["calls"]:
        claims, _ = _build(json.loads(call["raw_text"]), _shown(call), call["paper_id"],
                           **_typed(call["paper_id"]))
        for row, claim in enumerate(claims):
            identity_of[(call["call_id"], row)] = claim.claim_identity
            stored, _ = store.register(claim)
            outcome = judge(stored, store=store, registry=registry, papers=papers, evidence=evidence,
                            round_index=1)
            store.record_decision(outcome.decision)
            if outcome.admitted:
                admitted.add(stored.claim_id)
    return store, identity_of, admitted


#: Where this implementation's state differs from the adjudication table (section 2). Every
#: one is a non-admissible state either way; none is a T-row.
TABLE_DELTAS = {
    # ev_3b1f [6] is one "sentence" of the summary table: the cis-to-iso isomerase row
    # ("Enzymatic activity in crude extracts", a W7 in-vitro marker) glued to the MpGH3A row.
    # The table read "crude extracts" as no marker (it notes #4 "Mp; cell-free" itself).
    "2": IVO, "4": IVO, "6": IVO,
    # Adjudication 1b: "header glued to row 1 (#29 inferred_only instead of non_target)".
    "29": "inferred_only",
    # Its product and species cite only ev_1f2c, the GH3.10 front matter change set A never
    # mints: the product is unsupported (claim incomplete) and the species uncited. The
    # table: "IVO (species_ambiguous if ev_1f2c fully excluded)".
    "30": "species_not_stated",
}


def test_new_acceptance_745b_f1_the_full_pilot003_replay_admits_exactly_3_14_and_34():
    store, identity_of, admitted = _replay()
    by_number = {row["n"]: row for row in ADJUDICATION}
    assert admitted == {by_number[n]["claim_id"] for n in ("3", "14", "34")}
    assert {row["n"] for row in ADJUDICATION if row["admit"]} == {"3", "14", "34"}
    states = {}
    for row in ADJUDICATION:
        identity = identity_of[(row["call"], row["row"])]
        claim = next(c for c in store.siblings(identity))
        states[row["n"]] = claim.species_state
    expected = {row["n"]: TABLE_DELTAS.get(row["n"], row["required_state"]) for row in ADJUDICATION}
    assert states == expected
    assert all(state != TSS for state in TABLE_DELTAS.values())
    grades = {n: next(iter(store.siblings(identity_of[(by_number[n]["call"], by_number[n]["row"])]))).species_support["grade"]
              for n in ("3", "14", "34")}
    assert grades == {"3": "review", "14": "in_planta", "34": "review"}
