"""R-D092-1 -- row-level RAG lineage. NEW ACCEPTANCE TEST for a NEW capability.

**G9 LABELLING, STATED EXPLICITLY.** ``rd092_1_reaction_lineage.py`` is a new
evaluation-only module chartered by D-093 section 3. It corrects no pre-existing
observable behaviour and repairs no regression, so under merge gate G9 this file is an
**explicitly labelled new acceptance test** and carries no base-SHA failure proof. No
production module is imported, edited or exercised by anything here.

WHY CONSTRUCTED FIXTURES *AND* A REAL ARCHIVED LEG. Following F-177's reasoning: a
corpus is a draw, so the classifier rules are pinned against hand-built payloads whose
shapes are chosen to break a careless implementation. But this instrument's whole
purpose is to read ARCHIVED RUNS, and the sprint's review standard requires proving
that against a real archived leg directory -- so the last test does exactly that,
against the leg D-092 turns on, and skips rather than passing if that leg is absent.

EVERY RULE BELOW WAS A REAL DEFECT OR A REAL TRAP, not a hypothetical:

  * ``test_paper_stated_lineage_without_sources_...`` -- the instrument's own shipped
    bug. 650 of 692 reaction lineage entries in the committed corpus are
    ``(paper_stated, explicit)`` with ZERO sources, because ``lineage.py`` requires a
    named source only for ``SOURCED_ORIGINS``. Tiering on sources demoted all 650 to
    inheritance or ``no_signal``. A missing key read as zero -- this project's
    standing defect.
  * ``test_lineage_support_field_is_never_read_...`` -- D-091's failure one level
    down. Lineage ``support`` grades whether a NAMED SOURCE backs a row, so a
    paper-stated row is ``support="unsupported"`` while being paper-explicit.
  * ``test_rag_literature_origin_pointing_at_the_target_...`` -- 11 of the 14
    ``rag_literature`` source references in the corpus point AT the target paper.
  * ``test_canonical_enzyme_schema_participates`` -- the two populations key the
    enzyme name differently (``entity`` vs ``protein``), so reading one drops the
    enzyme from every row of the other population.
  * ``test_cross_run_rejection_is_indeterminate_...`` -- the sprint's standing trap
    that identical legs give materially different draws at temperature 0.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INSTRUMENT = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "rd092_1_reaction_lineage.py"

#: The leg D-092 turns on: PMC12312563 states one reaction, and the MenI row's four
#: participants carry PMC8091085 -- a paper PMC12312563 cites in its own reference
#: list. Setting ``supported_reactions_complete`` charged this row as invented.
ARCHIVED_LEG = (ROOT / "runs" / "2026-07-27_1623" / "papers"
                / "PMC12312563__structures-of-listeria-monocytogenes-mend-in-th" / "strict")
MENI_CHUNK = "fb1cf2b2fe9282c18bd86aaa6187ed8d"
MENI_SOURCE = "PMC8091085"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("rd092_1_reaction_lineage", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rd092_1_reaction_lineage"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rd() -> Any:
    if not INSTRUMENT.is_file():
        pytest.skip(f"instrument not present: {INSTRUMENT}")
    return _load()


def _leg(population: str = "canonical", target: str = "PMC111") -> Dict[str, str]:
    return {
        "leg_dir": "runs/R/papers/%s/strict" % target,
        "payload_file": "final_mapped.json" if population == "canonical" else "merged_payload.json",
        "population": population,
        "run": "runs/R",
        "target_paper": target,
    }


def _classify(rd: Any, reaction: Dict[str, Any], payload: Dict[str, Any] | None = None,
              adm: Dict[str, List[Dict[str, Any]]] | None = None,
              target: str = "PMC111") -> Dict[str, Any]:
    payload = payload or {"entities": {}}
    index = rd.entity_provenance_index(payload)
    return rd.build_record(reaction, _leg(target=target), 0, index, adm or {})


# ---------------------------------------------------------------------------
# The shipped bug: lineage presence, not lineage sources, decides the tier
# ---------------------------------------------------------------------------

def test_paper_stated_lineage_without_sources_is_row_lineage_and_target_supported(rd: Any) -> None:
    """650 corpus rows look exactly like this. Tiering on sources loses every one."""

    reaction = {
        "name": "r", "inputs": ["a"], "outputs": ["b"], "evidence": "",
        "provenance_lineage": [{
            "stage": "paper_extraction", "origin": "paper_stated",
            "support": "unsupported",        # the trap: source-grading, not a verdict
            "paper_explicit": "explicit",
            "reason": "stated by the paper", "review_required": False,
            "uncertainty": "", "sources": [],   # <-- zero sources, and correctly so
        }],
    }
    rec = _classify(rd, reaction)
    assert rec["attribution_tier"] == rd.TIER_ROW_LINEAGE
    assert rec["support_class"] == rd.TARGET_PAPER_SUPPORTED
    # The stage must survive too: reading stages off SOURCES alone reports
    # ``unavailable`` for every one of those 650 attributed rows.
    assert rec["origin_stages"] == ["paper_extraction"]


def test_lineage_support_field_is_never_read_as_a_biological_verdict(rd: Any) -> None:
    """D-091's collapse, one level down. ``support`` grades sources, not biology."""

    reaction = {
        "name": "r", "inputs": ["a"], "outputs": ["b"], "evidence": "",
        "provenance_lineage": [{
            "stage": "paper_extraction", "origin": "paper_stated",
            "support": "unsupported", "paper_explicit": "explicit",
            "reason": "", "review_required": False, "uncertainty": "", "sources": [],
        }],
    }
    rec = _classify(rd, reaction)
    assert rec["support_class"] != rd.UNSUPPORTED, (
        "a paper-explicit row was relabelled unsupported by reading lineage.support -- "
        "this is exactly the collapse D-091 was withdrawn for"
    )


# ---------------------------------------------------------------------------
# Source resolution: externality is decided by id, never by origin
# ---------------------------------------------------------------------------

def test_seed_paper_sentinel_resolves_to_the_target_paper(rd: Any) -> None:
    assert rd.resolve_source(rd.SEED_PAPER_SENTINEL, "PMC111") == rd.SRC_TARGET


def test_empty_source_id_is_unresolved_and_never_the_target(rd: Any) -> None:
    """Absence is not attribution: defaulting it to the target launders content."""

    assert rd.resolve_source("", "PMC111") == rd.SRC_UNRESOLVED
    assert rd.resolve_source(None, "PMC111") == rd.SRC_UNRESOLVED


def test_rag_literature_origin_pointing_at_the_target_is_not_external(rd: Any) -> None:
    """11 of 14 ``rag_literature`` refs in the corpus point at the target paper."""

    reaction = {
        "name": "r", "inputs": ["a"], "outputs": ["b"], "evidence": "",
        "provenance_lineage": [{
            "stage": "rag_admission", "origin": "rag_literature",
            "support": "direct", "paper_explicit": "not_evaluated",
            "reason": "", "review_required": False, "uncertainty": "",
            "sources": [{"source_id": "PMC111", "source_type": "paper",
                         "uri": "", "locator": "c1"}],
        }],
    }
    rec = _classify(rd, reaction, target="PMC111")
    assert rec["source_resolutions"] == [rd.SRC_TARGET]
    assert rec["support_class"] == rd.TARGET_PAPER_SUPPORTED


# ---------------------------------------------------------------------------
# Participant inheritance
# ---------------------------------------------------------------------------

def test_canonical_enzyme_schema_participates_in_inheritance(rd: Any) -> None:
    """Canonical rows key the enzyme name ``entity``; fallback rows key ``protein``.

    Reading only ``protein`` drops the enzyme from all 433 canonical rows.
    """

    reaction = {
        "name": "r", "inputs": [], "outputs": [], "evidence": "",
        "enzymes": [{"entity": "MenI", "entity_type": "protein",
                     "role": "catalyst", "provenance": "extracted"}],
    }
    assert "MenI" in rd.reaction_participants(reaction)

    payload = {"entities": {"proteins": [
        {"name": "MenI", "rag_provenance": {"source_id": "PMC999", "chunk_id": "cx"}},
    ]}}
    rec = _classify(rd, reaction, payload=payload)
    assert rec["attribution_tier"] == rd.TIER_PARTICIPANT_INHERITANCE
    assert rec["source_resolutions"] == [rd.SRC_EXTERNAL]


def test_enzyme_provenance_string_is_not_read_as_source_attribution(rd: Any) -> None:
    """``provenance: "extracted"`` is HOW, not WHERE. Same word, other vocabulary."""

    reaction = {
        "name": "r", "inputs": [], "outputs": [], "evidence": "",
        "enzymes": [{"entity": "MenD", "provenance": "extracted"}],
    }
    rec = _classify(rd, reaction, payload={"entities": {}})
    assert rec["enzyme_extraction_modes"] == ["extracted"]
    # It contributes no source, so the row must not acquire attribution from it.
    assert rec["attribution_tier"] == rd.TIER_NO_SIGNAL
    assert rec["source_ids"] == rd.UNAVAILABLE


def test_colliding_entity_names_keep_every_record(rd: Any) -> None:
    """Names collide inside one payload; a lookup returning one record picks a winner."""

    payload = {"entities": {"compounds": [
        {"name": "isochorismate"},                                    # no provenance
        {"name": "isochorismate",
         "rag_provenance": {"source_id": "seed_paper", "chunk_id": ""}},
    ]}}
    index = rd.entity_provenance_index(payload)
    assert len(index["isochorismate"]) == 2


def test_inheritance_without_reaction_specific_evidence_is_indeterminate(rd: Any) -> None:
    """D-093 condition 1. Entity provenance names a span that mentions a PARTICIPANT.

    Inheriting it proves the participants were retrieved, never that the chunk states
    the reaction -- so inheritance alone can never reach ``external_rag_supported``.
    """

    reaction = {"name": "r", "inputs": ["DHNA-CoA"], "outputs": ["DHNA"], "evidence": "x"}
    payload = {"entities": {"compounds": [
        {"name": "DHNA-CoA", "rag_provenance": {"source_id": "PMC999", "chunk_id": "cx"}},
        {"name": "DHNA", "rag_provenance": {"source_id": "PMC999", "chunk_id": "cx"}},
    ]}}
    rec = _classify(rd, reaction, payload=payload)   # empty admission index
    assert rec["attribution_tier"] == rd.TIER_PARTICIPANT_INHERITANCE
    assert rec["support_class"] == rd.INDETERMINATE
    assert rec["support_class"] != rd.EXTERNAL_RAG_SUPPORTED


# ---------------------------------------------------------------------------
# The chunk join
# ---------------------------------------------------------------------------

def _adm(group: str, run: str, inputs: List[str], outputs: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    return {"cx": [{
        "leg_dir": f"{run}/papers/PMC111/strict", "run": run, "group": group,
        "gap_id": "gap-1", "candidate_name": "c", "inputs": inputs, "outputs": outputs,
        "enzymes": [], "source_paper": "PMC999", "section": "abstract", "score": 0.9,
        "span": "a span", "reasons": ["r"], "requested_pathway_match": "match",
        "organism_match": "match", "scope_membership": "in_scope",
    }]}


def test_within_run_accepted_reaction_specific_join_earns_external_rag_supported(rd: Any) -> None:
    """The class D-093 exists to create: correctly-attributed cross-paper biology."""

    reaction = {"name": "r", "inputs": ["DHNA-CoA"], "outputs": ["DHNA"], "evidence": "x"}
    payload = {"entities": {"compounds": [
        {"name": "DHNA-CoA", "rag_provenance": {"source_id": "PMC999", "chunk_id": "cx"}},
    ]}}
    rec = _classify(rd, reaction, payload=payload,
                    adm=_adm("accepted", "runs/R", ["DHNA-CoA"], ["DHNA"]))
    assert rec["chunk_join_scope"] == rd.JOIN_WITHIN_RUN
    assert rec["chunk_join_reaction_specific"] is True
    assert rec["support_class"] == rd.EXTERNAL_RAG_SUPPORTED


def test_a_chunk_match_that_is_not_reaction_specific_does_not_support_the_row(rd: Any) -> None:
    """"not a span that merely names the participants" -- D-093 condition 1, literally."""

    reaction = {"name": "r", "inputs": ["DHNA-CoA"], "outputs": ["DHNA"], "evidence": "x"}
    payload = {"entities": {"compounds": [
        {"name": "DHNA-CoA", "rag_provenance": {"source_id": "PMC999", "chunk_id": "cx"}},
    ]}}
    # Same chunk, DIFFERENT reaction.
    rec = _classify(rd, reaction, payload=payload,
                    adm=_adm("accepted", "runs/R", ["chorismate"], ["isochorismate"]))
    assert rec["chunk_join_reaction_specific"] is False
    assert rec["support_class"] == rd.INDETERMINATE


def test_cross_run_rejection_is_indeterminate_not_an_accusation(rd: Any) -> None:
    """Identical legs give materially different draws at temperature 0 (standing trap).

    A rejection recorded in a different run is evidence about THAT run. It is carried
    on the record for a reader, but it does not charge this row as unsupported.
    """

    reaction = {"name": "r", "inputs": ["DHNA-CoA"], "outputs": ["DHNA"], "evidence": "x"}
    payload = {"entities": {"compounds": [
        {"name": "DHNA-CoA", "rag_provenance": {"source_id": "PMC999", "chunk_id": "cx"}},
    ]}}
    rec = _classify(rd, reaction, payload=payload,
                    adm=_adm("rejected", "runs/OTHER", ["DHNA-CoA"], ["DHNA"]))
    assert rec["chunk_join_scope"] == rd.JOIN_CROSS_RUN
    assert rec["support_class"] == rd.INDETERMINATE
    assert rec["rejection_reasons"] == ["r"], "the rejection must still be recorded"


def test_within_run_rejection_does_charge_the_row(rd: Any) -> None:
    """The counterpart: inside one run the gate's own verdict is decisive evidence."""

    reaction = {"name": "r", "inputs": ["DHNA-CoA"], "outputs": ["DHNA"], "evidence": "x"}
    payload = {"entities": {"compounds": [
        {"name": "DHNA-CoA", "rag_provenance": {"source_id": "PMC999", "chunk_id": "cx"}},
    ]}}
    rec = _classify(rd, reaction, payload=payload,
                    adm=_adm("rejected", "runs/R", ["DHNA-CoA"], ["DHNA"]))
    assert rec["chunk_join_scope"] == rd.JOIN_WITHIN_RUN
    assert rec["support_class"] == rd.UNSUPPORTED


# ---------------------------------------------------------------------------
# Populations, and the denominator discipline F-177 established
# ---------------------------------------------------------------------------

def test_populations_are_reported_separately_and_never_summed(rd: Any) -> None:
    records = [
        {"population": "canonical", "leg_dir": "l1", "support_class": rd.TARGET_PAPER_SUPPORTED,
         "attribution_tier": rd.TIER_ROW_LINEAGE, "chunk_join_scope": rd.JOIN_UNAVAILABLE,
         "chunk_join_reaction_specific": False},
        {"population": "fallback", "leg_dir": "l2", "support_class": rd.UNSUPPORTED,
         "attribution_tier": rd.TIER_NO_SIGNAL, "chunk_join_scope": rd.JOIN_UNAVAILABLE,
         "chunk_join_reaction_specific": False},
    ]
    s = rd.summarize(records)
    assert s["canonical"]["reactions"] == 1 and s["fallback"]["reactions"] == 1
    assert s["canonical"]["support_class"][rd.TARGET_PAPER_SUPPORTED] == 1
    assert s["fallback"]["support_class"][rd.TARGET_PAPER_SUPPORTED] == 0
    text = rd.render(s, records)
    # Every rate must name its denominator -- F-177's refusal, extended not regressed.
    for line in text.splitlines():
        if "%" in line:
            assert " of " in line, f"rate printed without a denominator: {line!r}"


def test_an_empty_population_is_reported_rather_than_skipped(rd: Any) -> None:
    s = rd.summarize([])
    assert s["canonical"]["reactions"] == 0 and s["fallback"]["reactions"] == 0
    assert "0 evaluated" in rd.render(s, [])


# ---------------------------------------------------------------------------
# The real archived leg -- the review standard for archived-run evaluation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARCHIVED_LEG.is_dir(), reason="archived leg not present")
def test_real_archived_leg_recovers_the_external_lineage_d092_turns_on() -> None:
    """Proven against a REAL archived leg directory, as the review standard requires.

    This is the row whose four participants carry ``PMC8091085`` and which
    ``supported_reactions_complete`` charged as invented. The instrument must recover
    the external source and must NOT call it target-paper-supported: measured, that
    leg's own ``01_source_text.txt`` contains zero occurrences of ``MenI``,
    ``DHNA-CoA thioesterase`` or ``LMRG_02730``.
    """

    rd = _load()
    payload_file = "final_mapped.json" if (ARCHIVED_LEG / "final_mapped.json").is_file() \
        else "merged_payload.json"
    payload = json.loads((ARCHIVED_LEG / payload_file).read_text(encoding="utf-8"))
    leg = {
        "leg_dir": ARCHIVED_LEG.relative_to(ROOT).as_posix(),
        "payload_file": payload_file,
        "population": "canonical" if payload_file == "final_mapped.json" else "fallback",
        "run": "runs/2026-07-27_1623",
        "target_paper": "PMC12312563",
    }
    index = rd.entity_provenance_index(payload)
    reactions = payload["processes"]["reactions"]
    rows = [rd.build_record(r, leg, i, index, {}) for i, r in enumerate(reactions)]

    meni = [r for r in rows if "DHNA" in (r["reaction_name"] or "")]
    assert meni, "the MenI row is missing from the archived leg"
    rec = meni[0]

    assert MENI_SOURCE in rec["source_ids"], (
        "the external source PMC8091085 was not recovered from participant provenance")
    assert MENI_CHUNK in rec["retrieved_chunk_ids"]
    assert rd.SRC_EXTERNAL in rec["source_resolutions"]
    assert rec["support_class"] != rd.TARGET_PAPER_SUPPORTED, (
        "a row the target paper never mentions was classified target_paper_supported")
    # And the sibling rows the paper DOES state must not be dragged down with it.
    others = [r for r in rows if "DHNA" not in (r["reaction_name"] or "")]
    assert others and all(r["support_class"] == rd.TARGET_PAPER_SUPPORTED for r in others), (
        "paper-stated rows in the same leg lost their target-paper attribution")


@pytest.mark.skipif(not ARCHIVED_LEG.is_dir(), reason="archived leg not present")
def test_real_archived_leg_row_evidence_string_is_not_treated_as_attribution() -> None:
    """The evidence string on that row is 35k characters of PMC8091085's ABSTRACT.

    A "row has evidence, therefore the paper supports it" test passes every row in the
    corpus and launders external text as target-paper support.
    """

    rd = _load()
    payload_file = "final_mapped.json" if (ARCHIVED_LEG / "final_mapped.json").is_file() \
        else "merged_payload.json"
    payload = json.loads((ARCHIVED_LEG / payload_file).read_text(encoding="utf-8"))
    reactions = payload["processes"]["reactions"]
    meni = [r for r in reactions if "DHNA" in (r.get("name") or "")][0]
    assert isinstance(meni.get("evidence"), str) and len(meni["evidence"]) > 10_000, (
        "fixture drift: the row no longer carries the long evidence blob this pins")
    leg = {"leg_dir": ARCHIVED_LEG.relative_to(ROOT).as_posix(), "payload_file": payload_file,
           "population": "canonical", "run": "runs/2026-07-27_1623",
           "target_paper": "PMC12312563"}
    rec = rd.build_record(meni, leg, 0, rd.entity_provenance_index(payload), {})
    assert rec["row_evidence_present"] is True
    assert rec["support_class"] != rd.TARGET_PAPER_SUPPORTED, (
        "presence of an evidence string was read as target-paper attribution")


# ---------------------------------------------------------------------------
# Fact (6): a database grounding is not literature evidence
#
# Found while diagnosing the 2026-09-02 run: a reaction grounded only in ChEBI /
# KEGG / DrugBank / CAS / taxonomy identifiers was reported as carrying "external
# participant provenance", borrowing the vocabulary D-093 reserves for retrieved
# literature. These pin the separation.
# ---------------------------------------------------------------------------

def test_database_grounded_origin_resolves_to_database_not_external(rd: Any) -> None:
    """The ORIGIN decides, and it is checked before any id comparison.

    Id-shape heuristics cannot do this job: a PathBank protein id is a bare integer
    and an NCBI taxonomy id is ``9606``, so guessing would be guessing about which
    evidence is literature.
    """

    assert rd.resolve_source("CHEBI:17627", "PMC111", "database_grounded") == rd.SRC_DATABASE
    assert rd.resolve_source("9606", "PMC111", "database_grounded") == rd.SRC_DATABASE
    assert rd.resolve_source("C00032", "PMC111", "database_grounded") == rd.SRC_DATABASE
    # Without the origin it is only an unrecognised id, and stays external.
    assert rd.resolve_source("CHEBI:17627", "PMC111") == rd.SRC_EXTERNAL


def test_a_row_grounded_only_in_databases_is_not_called_external_provenance(rd: Any) -> None:
    """The exact defect: ChEBI/KEGG groundings described as external literature."""

    reaction = {"name": "heme biosynthesis reaction", "inputs": ["glycine"],
                "outputs": ["heme"], "evidence": "x"}
    payload = {"entities": {"compounds": [
        {"name": "glycine", "provenance_lineage": [{
            "stage": "identifier_mapping", "origin": "database_grounded",
            "support": "direct", "paper_explicit": "not_evaluated",
            "reason": "", "review_required": False, "uncertainty": "",
            "sources": [{"source_id": "CHEBI:15428", "source_type": "database",
                         "uri": "", "locator": "CHEBI:15428"}]}]},
        {"name": "heme", "provenance_lineage": [{
            "stage": "identifier_mapping", "origin": "database_grounded",
            "support": "direct", "paper_explicit": "not_evaluated",
            "reason": "", "review_required": False, "uncertainty": "",
            "sources": [{"source_id": "CHEBI:17627", "source_type": "database",
                         "uri": "", "locator": "CHEBI:17627"}]}]},
    ]}}
    rec = _classify(rd, reaction, payload=payload)
    assert rec["source_resolutions"] == [rd.SRC_DATABASE, rd.SRC_DATABASE]
    assert rec["support_class"] == rd.INDETERMINATE
    assert "database groundings" in rec["support_reason"]
    assert "external" not in rec["support_reason"], (
        "a ChEBI grounding was described with D-093's literature vocabulary")


def test_a_database_grounding_can_never_reach_external_rag_supported(rd: Any) -> None:
    """Even with a chunk join that would otherwise promote the row."""

    reaction = {"name": "r", "inputs": ["glycine"], "outputs": ["heme"], "evidence": "x"}
    payload = {"entities": {"compounds": [
        {"name": "glycine", "provenance_lineage": [{
            "stage": "identifier_mapping", "origin": "database_grounded",
            "support": "direct", "paper_explicit": "not_evaluated",
            "reason": "", "review_required": False, "uncertainty": "",
            "sources": [{"source_id": "CHEBI:15428", "source_type": "database",
                         "uri": "", "locator": "cx"}]}]},
    ]}}
    rec = _classify(rd, reaction, payload=payload,
                    adm=_adm("accepted", "runs/R", ["glycine"], ["heme"]))
    assert rec["support_class"] != rd.EXTERNAL_RAG_SUPPORTED


def test_database_groundings_do_not_weaken_a_target_paper_attribution(rd: Any) -> None:
    """A paper-stated row whose participants are also ChEBI-grounded stays supported."""

    reaction = {"name": "r", "inputs": ["a"], "outputs": ["b"], "evidence": "x"}
    payload = {"entities": {"compounds": [
        {"name": "a", "provenance_lineage": [
            {"stage": "rag_admission", "origin": "rag_literature", "support": "direct",
             "paper_explicit": "not_evaluated", "reason": "", "review_required": False,
             "uncertainty": "", "sources": [{"source_id": "seed_paper",
                                             "source_type": "paper", "uri": "", "locator": ""}]},
            {"stage": "identifier_mapping", "origin": "database_grounded",
             "support": "direct", "paper_explicit": "not_evaluated", "reason": "",
             "review_required": False, "uncertainty": "",
             "sources": [{"source_id": "CHEBI:1", "source_type": "database",
                          "uri": "", "locator": "CHEBI:1"}]}]},
    ]}}
    rec = _classify(rd, reaction, payload=payload)
    assert rd.SRC_TARGET in rec["source_resolutions"]
    assert rd.SRC_DATABASE in rec["source_resolutions"]
    assert rec["support_class"] == rd.TARGET_PAPER_SUPPORTED
