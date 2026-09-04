"""F-179 — no PWML may be serialized when no reaction has defensible support.

**G9.** This is a REPAIR of pre-existing observable behaviour, not a new capability, so
it carries a base-SHA proof rather than a "new acceptance test" label: at the base SHA
``t2pw.pipeline.reaction_support`` does not exist and ``validate_pre_export`` returns
``ok`` for the F-179 payload, so :func:`test_base_sha_behaviour_is_the_defect` documents
the failing base behaviour and the archived artifact proves it independently — the
2026-09-02 leg shipped ``pathway.review_required.pwml`` with
``semantic_evaluation: passed``. **Symbol absence is not the proof**; the archived
exported PWML is.

WHAT MUST NOT BE IN THE PRODUCTION RULE, and these tests check for it: no pathway name,
no paper id, no gold reaction, no reaction-count threshold, no chemistry. The discovery
census used "terminal product" and "precursor → terminal shortcut"; those are detection
heuristics and are deliberately not the rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.reaction_support import (  # noqa: E402
    CODE_NO_DEFENSIBLE_REACTION_SUPPORT,
    SUPPORT_EXTERNAL_RAG,
    SUPPORT_TARGET_PAPER,
    evaluate_reaction_support,
    lineage_carrier_active,
    reaction_support_class,
    reaction_support_issue,
)
from t2pw.pipeline.stage_contracts import StageContractError, validate_pre_export  # noqa: E402

#: The archived leg F-179 was found on. Untracked run: these tests SKIP without it
#: rather than failing, and the constructed fixtures carry the load-bearing assertions.
ARCHIVED_LEG = ROOT / "runs_verify" / "2026-09-02_2052" / "papers" / "PMC12180156" / "strict"


def _entry(stage: str, origin: str, *, paper_explicit: str = "not_evaluated",
           sources: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    return {"stage": stage, "origin": origin, "support": "direct",
            "paper_explicit": paper_explicit, "reason": "", "review_required": False,
            "uncertainty": "", "sources": sources or []}


def _payload(reactions: List[Dict[str, Any]], *,
             entity_lineage: bool = True) -> Dict[str, Any]:
    """A payload whose ENTITIES are paper-stated and database-grounded.

    That is the F-179 shape: both participants resolve to real identities and are named
    by the paper, which must never make the REACTION supported.
    """

    ent: Dict[str, Any] = {"name": "glycine"}
    if entity_lineage:
        ent["provenance_lineage"] = [
            _entry("paper_extraction", "paper_stated", paper_explicit="explicit"),
            _entry("identifier_mapping", "database_grounded",
                   sources=[{"source_id": "CHEBI:15428", "source_type": "database",
                             "uri": "", "locator": "CHEBI:15428"}]),
        ]
    return {"entities": {"compounds": [ent]}, "processes": {"reactions": reactions}}


def _f179_codes(payload: Dict[str, Any]) -> List[str]:
    """Issue codes from the pre-export contract, however it terminates.

    The minimal fixtures here deliberately carry no pathway metadata or biological
    states, so the PRE-EXISTING contract fails them for unrelated reasons. What these
    tests assert is only whether the F-179 code is among the errors -- never that an
    unrelated minimal fixture passes the whole contract.
    """

    try:
        report = validate_pre_export(payload, strict_db=False)
    except StageContractError as exc:
        report = exc.report
    return [i.get("code") for i in (report.get("errors") or [])]


# ---------------------------------------------------------------------------
# MUST BLOCK
# ---------------------------------------------------------------------------

def test_must_block_a_reaction_with_no_paper_no_rag_and_no_supported_parent() -> None:
    """§9 case 1 — the F-179 shape, reduced to provenance."""

    payload = _payload([{"name": "heme biosynthesis reaction", "inputs": ["glycine"],
                         "outputs": ["heme"],
                         "evidence": "Glycine is also required for the rate-limiting "
                                     "step in heme biosynthesis"}])
    report = evaluate_reaction_support(payload)
    assert report["verdict"] == "no_defensible_core"
    assert reaction_support_issue(payload)["code"] == CODE_NO_DEFENSIBLE_REACTION_SUPPORT

    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT in _f179_codes(payload)


def test_must_block_a_reaction_whose_only_provenance_is_identifier_mapping() -> None:
    """§9 case 2. Identity is not occurrence."""

    row = {"name": "r", "inputs": ["glycine"], "outputs": ["heme"],
           "provenance_lineage": [
               _entry("identifier_mapping", "database_grounded",
                      sources=[{"source_id": "CHEBI:17627", "source_type": "database",
                                "uri": "", "locator": "CHEBI:17627"}]),
               _entry("normalization", "inferred"),
           ]}
    assert reaction_support_class(row) is None
    assert evaluate_reaction_support(_payload([row]))["verdict"] == "no_defensible_core"


def test_entity_level_paper_support_never_reaches_the_reaction() -> None:
    """The ruling's forbidden inference, stated as a test.

    Both participants are paper-stated entities. The reaction still has no support.
    """

    payload = _payload([{"name": "r", "inputs": ["glycine"], "outputs": ["heme"]}])
    assert payload["entities"]["compounds"][0]["provenance_lineage"]
    assert evaluate_reaction_support(payload)["verdict"] == "no_defensible_core"


# ---------------------------------------------------------------------------
# MUST ALLOW
# ---------------------------------------------------------------------------

def test_must_allow_paper_supported_reaction_later_modified_by_identifier_mapping() -> None:
    """§9 case 3 — the ruling's central distinction.

    ``identifier_mapping modified a supported reaction`` is valid; it is only invalid
    when mapping is the ONLY discoverable reason the reaction exists.
    """

    row = {"name": "r", "inputs": ["a"], "outputs": ["b"],
           "provenance_lineage": [
               _entry("paper_extraction", "paper_stated", paper_explicit="explicit"),
               _entry("identifier_mapping", "database_grounded",
                      sources=[{"source_id": "CHEBI:1", "source_type": "database",
                                "uri": "", "locator": "CHEBI:1"}]),
           ]}
    assert reaction_support_class(row) == SUPPORT_TARGET_PAPER
    assert evaluate_reaction_support(_payload([row]))["verdict"] == "supported"
    assert reaction_support_issue(_payload([row])) is None
    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT not in _f179_codes(_payload([row]))


def test_must_allow_external_rag_reaction_later_normalized_and_mapped() -> None:
    """§9 case 4."""

    row = {"name": "r", "inputs": ["a"], "outputs": ["b"],
           "provenance_lineage": [
               _entry("rag_admission", "rag_literature",
                      sources=[{"source_id": "PMC999", "source_type": "paper",
                                "uri": "", "locator": "chunk1"}]),
               _entry("normalization", "inferred"),
               _entry("identifier_mapping", "database_grounded"),
           ]}
    assert reaction_support_class(row) == SUPPORT_EXTERNAL_RAG
    assert evaluate_reaction_support(_payload([row]))["verdict"] == "supported"


def test_must_allow_the_row_level_rag_provenance_carrier() -> None:
    """§9 case 4, the other carrier: ``_RAG_ROW_CARRIER_KEYS`` on the row."""

    row = {"name": "r", "inputs": ["a"], "outputs": ["b"],
           "rag_provenance": {"source_id": "PMC8091085", "chunk_id": "c1"}}
    assert reaction_support_class(row) == SUPPORT_EXTERNAL_RAG


def test_must_allow_supported_reaction_with_synonym_replacement() -> None:
    """§9 case 5 — a synonym swap changes the name, not the support."""

    row = {"name": "r", "inputs": ["alpha-ketoglutarate"], "outputs": ["SEPHCHC"],
           "provenance_lineage": [
               _entry("paper_extraction", "paper_stated", paper_explicit="explicit"),
               _entry("normalization", "inferred"),
           ]}
    assert reaction_support_class(row) == SUPPORT_TARGET_PAPER


def test_must_allow_deterministic_canonicalization_preserving_identity() -> None:
    """§9 case 6 — audit/repair on top of a supported row keeps it supported."""

    row = {"name": "r", "inputs": ["a"], "outputs": ["b"],
           "provenance_lineage": [
               _entry("paper_extraction", "paper_stated", paper_explicit="explicit"),
               _entry("audit_repair", "audit_modified"),
           ]}
    assert reaction_support_class(row) == SUPPORT_TARGET_PAPER


def test_a_defensible_core_allows_the_export_even_with_unattributed_rows() -> None:
    """The check is leg-level, never a per-row deletion (merge rule 8)."""

    supported = {"name": "ok", "inputs": ["a"], "outputs": ["b"],
                 "provenance_lineage": [_entry("paper_extraction", "paper_stated",
                                               paper_explicit="explicit")]}
    bare = {"name": "bare", "inputs": ["x"], "outputs": ["y"]}
    report = evaluate_reaction_support(_payload([supported, bare]))
    assert report["verdict"] == "supported"
    assert report["unattributed"] == 1, "the unattributed row must still be reported"


# ---------------------------------------------------------------------------
# MUST DISTINGUISH — §9 case 7
# ---------------------------------------------------------------------------

def test_indeterminate_archival_lineage_is_not_turned_into_unsupported() -> None:
    """A payload from a run predating the carrier cannot be judged by reading it.

    "We cannot tell" and "it is unsupported" are different facts, and collapsing them
    is the failure D-091 was withdrawn for.
    """

    payload = _payload([{"name": "r", "inputs": ["a"], "outputs": ["b"]}],
                       entity_lineage=False)
    assert lineage_carrier_active(payload) is False
    report = evaluate_reaction_support(payload)
    assert report["verdict"] == "indeterminate"
    assert reaction_support_issue(payload) is None
    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT not in _f179_codes(payload)


def test_carrier_active_via_entities_alone_still_judges_the_reactions() -> None:
    """The positive case: the carrier IS running, so absence on reactions is a finding."""

    payload = _payload([{"name": "r", "inputs": ["a"], "outputs": ["b"]}],
                       entity_lineage=True)
    assert lineage_carrier_active(payload) is True
    assert evaluate_reaction_support(payload)["verdict"] == "no_defensible_core"


def test_a_payload_with_no_reactions_is_indeterminate_not_a_violation() -> None:
    assert evaluate_reaction_support(_payload([]))["verdict"] == "indeterminate"
    assert reaction_support_issue(_payload([])) is None


# ---------------------------------------------------------------------------
# The rule must be general
# ---------------------------------------------------------------------------

def test_the_production_rule_contains_no_paper_pathway_or_gold_specifics() -> None:
    """No leakage of the discovery heuristic or of any case identity."""

    import ast

    path = SRC / "t2pw" / "pipeline" / "reaction_support.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Docstrings and comments cite the case history ON PURPOSE -- a rule whose
    # provenance is undocumented is worse, not better. What must be free of case
    # identity is the EXECUTABLE code, so strip every docstring and re-render.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    code = ast.unparse(ast.fix_missing_locations(tree))

    for banned in ("PMC", "heme", "cholesterol", "enterobactin", "glycine",
                   "terminal_product", "precursor", "gold", "supported_reactions",
                   "12180156"):
        assert banned not in code, f"production rule leaks {banned!r} in executable code"


def test_chemistry_is_never_read() -> None:
    """Identical provenance, opposite chemistry -> identical verdict."""

    lin = [_entry("paper_extraction", "paper_stated", paper_explicit="explicit")]
    a = {"name": "a", "inputs": ["glycine"], "outputs": ["heme"], "provenance_lineage": lin}
    b = {"name": "b", "inputs": ["water"], "outputs": ["water"], "provenance_lineage": lin}
    assert reaction_support_class(a) == reaction_support_class(b) == SUPPORT_TARGET_PAPER


# ---------------------------------------------------------------------------
# Real archived payload — the F-179 case itself
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (ARCHIVED_LEG / "final_mapped.json").is_file(),
                    reason="archived F-179 leg not present (untracked run)")
def test_base_sha_behaviour_is_the_defect() -> None:
    """The archived leg IS the base-SHA proof: it exported with the gate passing."""

    leg = ARCHIVED_LEG
    assert (leg / "pathway.review_required.pwml").is_file(), (
        "the archived leg no longer carries the exported PWML this proof rests on")
    release = json.loads((leg / "quarantine_report.json").read_text(encoding="utf-8"))
    rel = release.get("release") or {}
    assert rel.get("semantic_evaluation") == "passed"
    assert rel.get("strict_gates_passed") is True


@pytest.mark.skipif(not (ARCHIVED_LEG / "final_mapped.json").is_file(),
                    reason="archived F-179 leg not present (untracked run)")
def test_repaired_gate_refuses_the_archived_f179_payload() -> None:
    """§10 — prove the repair on the real archived canonical payload."""

    payload = json.loads((ARCHIVED_LEG / "final_mapped.json").read_text(encoding="utf-8"))
    rows = payload["processes"]["reactions"]
    assert len(rows) == 1 and rows[0]["inputs"] == ["glycine"] and rows[0]["outputs"] == ["heme"]

    report = evaluate_reaction_support(payload)
    assert report["verdict"] == "no_defensible_core"
    assert report["lineage_carrier_active"] is True, (
        "the guard must confirm the carrier was running before this counts as a finding")
    assert report["supported"] == 0 and report["unattributed"] == 1

    try:
        pre = validate_pre_export(payload, strict_db=False)
    except StageContractError as exc:
        pre = exc.report
    issues = [i for i in (pre.get("errors") or [])
              if i.get("code") == CODE_NO_DEFENSIBLE_REACTION_SUPPORT]
    assert issues, "the repaired pre-export gate did not raise the F-179 issue"
    assert "do not establish that a reaction occurs" in issues[0]["support_report"]["reason"]


# ---------------------------------------------------------------------------
# Export-mode behaviour: strict aborts, research records a review flag
# ---------------------------------------------------------------------------

def test_research_mode_downgrades_rather_than_aborting() -> None:
    """``run_stage_contract`` in research mode re-severities the finding.

    SCOPE, STATED EXACTLY. This pins ``run_stage_contract``'s behaviour, which is what
    a research-mode caller of this boundary would get. It does NOT describe the
    production PWML seam: that caller invokes ``validate_pre_export`` directly and
    never routes it through ``run_stage_contract``, so no relaxation happens there in
    either mode. That costs nothing, because PWML deliverables are STRICT-mode only and
    a research leg emits no PWML to refuse — but the two must not be conflated, and an
    earlier revision of this module's docstring did conflate them.
    """

    from t2pw.pipeline.export_mode import RESEARCH
    from t2pw.pipeline.stage_contracts import run_stage_contract

    payload = _payload([{"name": "r", "inputs": ["glycine"], "outputs": ["heme"]}])
    report = run_stage_contract(validate_pre_export, payload, strict_db=False,
                                mode=RESEARCH)
    codes = [i.get("code") for i in (report.get("errors") or [])]
    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT not in codes, (
        "research mode must not abort on a biology/provenance finding")


def test_strict_mode_records_the_support_report_even_when_it_passes() -> None:
    """The report is always attached, so "supported" is auditable, not just silence."""

    row = {"name": "r", "inputs": ["a"], "outputs": ["b"],
           "provenance_lineage": [_entry("paper_extraction", "paper_stated",
                                         paper_explicit="explicit")]}
    try:
        report = validate_pre_export(_payload([row]), strict_db=False)
    except StageContractError as exc:
        report = exc.report
    support = report.get("reaction_support_report")
    assert support and support["verdict"] == "supported"
    assert support["target_paper_supported"] == 1


# ---------------------------------------------------------------------------
# REAL-PATH PROOF — the caller's own decision variable
#
# Independent review found the first revision of this repair raised on the OUTER
# stage report while the production caller decides on the INNER one, so the export
# proceeded and the defect was not fixed -- with eighteen green tests over it. These
# assert the variable ``run_pwml_export`` actually branches on.
# ---------------------------------------------------------------------------

#: The caller's decision, transcribed from ``streamlit_app.py`` (``run_pwml_export``):
#:     required_gate_report = pre_export_contract.get("pwml_contract_report")
#:     if not required_gate_report.get("ok", False):
#:         return {"ok": False, "output_path": "", ...}
#: That file is PROTECTED and is not imported here (importing it executes a Streamlit
#: script); the branch is replicated exactly instead.
def _caller_would_export(payload: Dict[str, Any]) -> bool:
    try:
        pre = validate_pre_export(payload, strict_db=False)
    except StageContractError as exc:
        pre = exc.report
    required_gate_report = pre.get("pwml_contract_report") or {}
    return bool(required_gate_report.get("ok", False))


def test_the_finding_lands_on_the_report_the_caller_decides_on() -> None:
    payload = _payload([{"name": "r", "inputs": ["glycine"], "outputs": ["heme"]}])
    try:
        pre = validate_pre_export(payload, strict_db=False)
    except StageContractError as exc:
        pre = exc.report
    inner = pre.get("pwml_contract_report") or {}
    assert inner.get("ok") is False, (
        "the F-179 refusal did not reach pwml_contract_report -- the caller reads THAT "
        "report, so the export would still proceed")
    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT in [i.get("code") for i in (inner.get("errors") or [])]
    assert inner.get("summary", {}).get("error_count", 0) >= 1


def test_the_caller_would_refuse_to_export_the_unsupported_payload() -> None:
    payload = _payload([{"name": "r", "inputs": ["glycine"], "outputs": ["heme"]}])
    assert _caller_would_export(payload) is False


def test_the_caller_still_exports_a_supported_payload() -> None:
    """False-positive protection at the same seam: support must still get through."""

    row = {"name": "r", "inputs": ["a"], "outputs": ["b"],
           "provenance_lineage": [_entry("paper_extraction", "paper_stated",
                                         paper_explicit="explicit")]}
    payload = _payload([row])
    try:
        pre = validate_pre_export(payload, strict_db=False)
    except StageContractError as exc:
        pre = exc.report
    inner = pre.get("pwml_contract_report") or {}
    codes = [i.get("code") for i in (inner.get("errors") or [])]
    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT not in codes, (
        "a supported payload was refused by the F-179 rule")


@pytest.mark.skipif(not (ARCHIVED_LEG / "final_mapped.json").is_file(),
                    reason="archived F-179 leg not present (untracked run)")
def test_real_archived_f179_payload_is_refused_at_the_callers_decision() -> None:
    """§15 real-path proof, on the archived canonical payload that actually shipped.

    The archived ``pwml_required_field_gate_report.json`` for this leg records
    ``ok: true`` -- that is the base behaviour, and it is why the PWML exists. The
    repaired gate must make that same report false.
    """

    payload = json.loads((ARCHIVED_LEG / "final_mapped.json").read_text(encoding="utf-8"))
    payload.setdefault("metadata", {}).update(
        {"pathway_name": "p", "name": "p", "description": "d",
         "width": 800, "height": 600, "pathway_subject": "s", "subject": "s"})

    archived = json.loads(
        (ARCHIVED_LEG / "pwml_required_field_gate_report.json").read_text(encoding="utf-8"))
    assert archived.get("ok") is True, (
        "fixture drift: the archived gate report no longer shows the passing base state")

    assert _caller_would_export(payload) is False, (
        "the repaired gate still lets the F-179 payload reach serialization")


def test_the_persisted_summary_is_self_consistent() -> None:
    """``error_codes`` must mirror ``errors``, as ``ir.py:3087`` guarantees.

    This mapping is written to disk as ``pwml_required_field_gate_report.json``. An
    artifact saying ``error_count: 1`` beside ``error_codes: []`` gives a later reader
    two answers and no way to choose. Raised by independent review.
    """

    payload = _payload([{"name": "r", "inputs": ["glycine"], "outputs": ["heme"]}])
    try:
        pre = validate_pre_export(payload, strict_db=False)
    except StageContractError as exc:
        pre = exc.report
    inner = pre.get("pwml_contract_report") or {}
    summary = inner.get("summary") or {}
    codes = [i.get("code") for i in (inner.get("errors") or [])]
    assert summary.get("error_count") == len(inner.get("errors") or [])
    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT in summary.get("error_codes", [])
    assert sorted(set(c for c in codes if c)) == sorted(summary.get("error_codes", []))
