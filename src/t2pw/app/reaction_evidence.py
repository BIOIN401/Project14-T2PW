"""Reaction-evidence review model and deterministic plain-text export (ORCH-742b).

PRESENTATION ONLY. Everything here is read from a run's EXISTING artifacts --
``coverage_summary.json``, ``quarantine_report.json`` and
``removed_entity_report.json`` (the strict-quarantine set written by
``strict_quarantine.write_quarantine_artifacts``), ``final_mapped.json`` (the
canonical payload, ``batch.driver._add_identity_artifacts``, including its
``quarantined_locked_reactions`` record, which ``pipeline.filter_unresolvable_reactions``,
``process_normalizer.dedupe_processes`` and strict quarantine all append to; a row
from it is labelled by that record name, never by a guessed stage), ``pwml_ir.json``
(the IR the PWML was written from, ``batch.driver._add_pwml_artifacts``: each
interaction's export verdict is READ from it, never recomputed, and only when a PWML
was written) and the first
record of ``LEG_TRACE.jsonl`` for the paper id. Nothing
is recomputed, repaired, merged or deleted, and no model, embedding or network
call is made: this module imports only the standard library, the quarantine
filename constants and the interaction sign table (``t2pw.pwml.interaction_types``,
itself standard-library only).

A value an artifact does not carry is rendered ``Not recorded``; it is never
guessed. A missing artifact (``Not recorded``) and a recorded empty list
(``None recorded``) stay distinct, and ``False`` is a value, not an absence.

The duplicate check is a REVIEWER WARNING only: two surviving reactions with the
same normalized substrates and products are flagged "Possible duplicate
reaction" and both stay in the report exactly as the pathway has them.

D-114 (ORCH-772): the PWML coverage at the top (processes drawn in the PWML / processes
in the text record) and the section "OMITTED FROM THE PWML", each left-out process with
its recorded reason, the known-issue id that reason maps to (:data:`KNOWN_ISSUES`, a copy
of the ``PWML_KNOWN_ISSUES.md`` appendix) and the catalogue's workaround. "Drawn" is read
from the IR only when a PWML was written. This text is for people only: no LLM input is
built from this module (D-104 AMENDMENT 2).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Standard library only (the sign table, no exporter): used to check that an IR row
# belongs to the payload row beside it, never to decide a verdict.
from t2pw.pwml.interaction_types import interaction_relationship_word, pwml_interaction_type

from t2pw.pipeline.strict_quarantine import (
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_REPORT_FILENAME,
    REMOVED_ENTITY_REPORT_FILENAME,
)

FINAL_MAPPED_FILENAME = "final_mapped.json"
#: ``t2pw.curation.prompt_payload_keys.BORROWED_PROTEIN_IDENTITIES_KEY``, spelled out: this
#: module imports only the standard library and two leaf tables (see the module docstring).
BORROWED_PROTEIN_IDENTITIES_KEY = "borrowed_protein_identities"
PWML_IR_FILENAME = "pwml_ir.json"
#: The PWML a strict leg writes (``batch.driver``: release-ready / review-required). A verdict
#: is shown only when one of them is present: an IR exists even when validation stopped the
#: export (``streamlit_app`` returns ``ok: False`` with ``pwml_ir``; the driver still writes
#: ``pwml_ir.json``), and a verdict then would claim a row was drawn in a file that was never
#: written.
PWML_FILENAMES = ("pathway.pwml", "pathway.review_required.pwml")
LEG_TRACE_FILENAME = "LEG_TRACE.jsonl"
JSON_ARTIFACTS = (
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_REPORT_FILENAME,
    REMOVED_ENTITY_REPORT_FILENAME,
    FINAL_MAPPED_FILENAME,
    PWML_IR_FILENAME,
)
NOT_RECORDED = "Not recorded"
NONE_RECORDED = "None recorded"
DUPLICATE_WARNING = "Possible duplicate reaction"
LOCKED_RECORD = "final payload quarantined_locked_reactions"

# ── D-114: PWML coverage and the omission record (ORCH-772) ─────────────────────────────
#: Coverage = processes drawn in the PWML / processes in the text record. The headline counts
#: these kinds; an omitted row of another kind (a reaction-coupled transport, a sub-pathway) is
#: listed but not counted. Orchestrator ruling, ORCH-772 (the product owner may overrule).
COVERAGE_KINDS = ("reaction", "transport", "interaction")
COVERAGE_TARGET_PERCENT = 75
#: A pre-freeze removal labelled with this id is a duplicate folded into a kept row: listed,
#: not counted (orchestrator ruling, ORCH-772).
MERGED_DUPLICATE_KI = "KI-014"
#: An interaction the IR withholds with this id cannot be typed in PathWhiz at all. It counts
#: in the headline; one secondary line leaves it out (orchestrator ruling, ORCH-772).
BY_DESIGN_KI = "KI-008"
KI_UNCLASSIFIED = "unclassified"
KNOWN_ISSUES_DOC = "docs/pwml_recovery_sprint/PWML_KNOWN_ISSUES.md"

#: ``PWML_KNOWN_ISSUES.md``, appendix "Reason code -> KI id", as data. ``codes`` rows are
#: (code as the appendix writes it, match, patterns, KI id, then label by the inner reason);
#: ``messages`` rows are (the quoted message patterns, KI id); ``workarounds`` is each such KI
#: entry's "User workaround" cell, verbatim. ``tests/test_orch772_pwml_coverage.py`` parses the
#: catalogue and fails when this copy and the catalogue disagree.
KNOWN_ISSUES: Dict[str, Any] = {
    "codes": (
        ("<bucket>_missing_name", "suffix", ("_missing_name",), "KI-015", False),
        ("accession_claimed_across_incompatible_entity_kinds", "exact",
         ("accession_claimed_across_incompatible_entity_kinds",), "KI-017", False),
        ("actor_schema_not_canonical", "exact", ("actor_schema_not_canonical",), "KI-023", False),
        ("ambiguous_entity_type:<bucket>+<bucket>", "prefix", ("ambiguous_entity_type:",), "KI-011", False),
        ("ambiguous_protein_endpoint", "exact", ("ambiguous_protein_endpoint",), "KI-010", False),
        ("bare_protein_endpoint", "exact", ("bare_protein_endpoint",), "KI-009", False),
        ("budget_exhausted", "exact", ("budget_exhausted",), "KI-028", False),
        ("budget_indeterminate", "exact", ("budget_indeterminate",), "KI-056", False),
        ("class_term_never_narrowed_to_specific_molecule", "exact",
         ("class_term_never_narrowed_to_specific_molecule",), "KI-029", False),
        ("closure_not_converged:<n>", "prefix", ("closure_not_converged:",), "KI-016", False),
        ("complex_named_source_entity_wrapper", "exact", ("complex_named_source_entity_wrapper",), "KI-012", False),
        ("component_protein_unresolved", "exact", ("component_protein_unresolved",), "KI-015", False),
        ("confidence_below_floor:<x><<y>", "prefix", ("confidence_below_floor:",), "KI-015", False),
        ("degree_zero_export:<n>", "prefix", ("degree_zero_export:",), "KI-016", False),
        ("diagnostic_only", "exact", ("diagnostic_only",), "KI-016", False),
        ("duplicate_locked_reaction", "exact", ("duplicate_locked_reaction",), "KI-014", False),
        ("entity_type_overlap:<n>", "prefix", ("entity_type_overlap:",), "KI-011", False),
        ("explicit_out_of_scope_verdict", "exact", ("explicit_out_of_scope_verdict",), "KI-015", False),
        ("final_gate_report_missing", "exact", ("final_gate_report_missing",), "KI-016", False),
        ("final_pre_export_gate_lifecycle", "exact", ("final_pre_export_gate_lifecycle",), "KI-016", False),
        ("final_pre_export_stage3_gates", "exact", ("final_pre_export_stage3_gates",), "KI-004", False),
        ("gate.generated_protein_complex_<...>", "prefix", ("gate.generated_protein_complex_",), "KI-012", False),
        ("gate.protein_<name>_is_missing_a_uniprot_or_drugbank_identifier", "prefix+suffix",
         ("gate.protein_", "_is_missing_a_uniprot_or_drugbank_identifier"), "KI-001", False),
        ("gate.protein_has_degree_after_normalization", "exact",
         ("gate.protein_has_degree_after_normalization",), "KI-013", False),
        ("gate.unknown_protein_modifier_reference", "exact", ("gate.unknown_protein_modifier_reference",), "KI-004", False),
        ("gate.unknown_transporter_reference", "exact", ("gate.unknown_transporter_reference",), "KI-004", False),
        ("generated_complex_component_missing_external_identity", "exact",
         ("generated_complex_component_missing_external_identity",), "KI-001", False),
        ("generated_complex_component_missing_species", "exact",
         ("generated_complex_component_missing_species",), "KI-003", False),
        ("generated_wrapper_reason_invalid", "exact", ("generated_wrapper_reason_invalid",), "KI-012", False),
        ("identity_refused_review_required", "exact", ("identity_refused_review_required",), "KI-029", False),
        ("implausible_name_match", "exact", ("implausible_name_match",), "KI-001", False),
        ("in_process_deadline", "exact", ("in_process_deadline",), "KI-028", False),
        ("interaction_withheld_from_pwml", "exact", ("interaction_withheld_from_pwml",), "KI-008", True),
        ("malformed_participant_in_<slot>", "prefix", ("malformed_participant_in_",), "KI-015", False),
        ("malformed_placeholder:<claim>", "prefix", ("malformed_placeholder:",), "KI-015", False),
        ("minimum_core:requested_core_coverage_below_minimum:<x><<y>", "prefix", ("minimum_core:",), "KI-038", False),
        ("missing_<slot>", "prefix", ("missing_",), "KI-015", False),
        ("nested_location_merged", "exact", ("nested_location_merged",), "KI-043", False),
        ("no_defensible_reaction_support", "exact", ("no_defensible_reaction_support",), "KI-022", False),
        ("no_drawable_process", "exact", ("no_drawable_process",), "KI-021", False),
        ("no_reactions", "exact", ("no_reactions",), "KI-021", False),
        ("no_shared_meaningful_token", "exact", ("no_shared_meaningful_token",), "KI-006", False),
        ("non_protein_actor_in_<slot>", "prefix", ("non_protein_actor_in_",), "KI-004", False),
        ("operation_timeout", "exact", ("operation_timeout",), "KI-028", False),
        ("outer_parent_kill", "exact", ("outer_parent_kill",), "KI-028", False),
        ("parenthetical_definition_bridge", "exact", ("parenthetical_definition_bridge",), "KI-041", False),
        ("process_row_not_an_object", "exact", ("process_row_not_an_object",), "KI-015", False),
        ("process_row_vanished_during_closure", "exact", ("process_row_vanished_during_closure",), "KI-015", False),
        ("protein_complex_component_cycle", "exact", ("protein_complex_component_cycle",), "KI-015", False),
        ("protein_complex_component_omitted:<component>:<reason>", "prefix",
         ("protein_complex_component_omitted:",), "KI-015", True),
        ("protein_complex_component_unnamed", "exact", ("protein_complex_component_unnamed",), "KI-015", False),
        ("protein_complex_component_unrepresentable:<name>:<reason>", "prefix",
         ("protein_complex_component_unrepresentable:",), "KI-015", True),
        ("protein_complex_component_unresolved:<name>", "prefix", ("protein_complex_component_unresolved:",), "KI-015", False),
        ("protein_complex_missing_components", "exact", ("protein_complex_missing_components",), "KI-015", False),
        ("protein_complex_missing_species", "exact", ("protein_complex_missing_species",), "KI-003", False),
        ("protein_endpoint_missing_external_identity", "exact",
         ("protein_endpoint_missing_external_identity",), "KI-009", False),
        ("protein_endpoint_missing_species", "exact", ("protein_endpoint_missing_species",), "KI-009", False),
        ("protein_endpoint_missing_species_and_external_identity", "exact",
         ("protein_endpoint_missing_species_and_external_identity",), "KI-009", False),
        ("protein_missing_external_identity", "exact", ("protein_missing_external_identity",), "KI-001", False),
        ("protein_missing_species", "exact", ("protein_missing_species",), "KI-003", False),
        ("reaction_coupled_transport_has_no_exportable_ir_representation", "exact",
         ("reaction_coupled_transport_has_no_exportable_ir_representation",), "KI-015", False),
        ("reaction_enzyme_complex_wrapper_skipped_invalid_component", "exact",
         ("reaction_enzyme_complex_wrapper_skipped_invalid_component",), "KI-006", False),
        ("reaction_enzyme_must_be_protein_complex", "exact", ("reaction_enzyme_must_be_protein_complex",), "KI-006", False),
        ("reaction_enzyme_proteins_dropped_no_external_id", "exact",
         ("reaction_enzyme_proteins_dropped_no_external_id",), "KI-007", False),
        ("refinement_review_crash", "exact", ("refinement_review_crash",), "KI-020", False),
        ("requested_core_coverage_below_minimum:<x><<y>", "prefix", ("requested_core_coverage_below_minimum",), "KI-038", False),
        ("scope_conflict", "exact", ("scope_conflict",), "KI-025", False),
        ("serialization_would_require_invention", "exact", ("serialization_would_require_invention",), "KI-016", False),
        ("species_mismatch", "exact", ("species_mismatch",), "KI-002", False),
        ("species_missing_classification", "exact", ("species_missing_classification",), "KI-019", False),
        ("species_missing_taxonomy", "exact", ("species_missing_taxonomy",), "KI-019", False),
        ("species_not_found:<species>", "prefix", ("species_not_found:",), "KI-002", False),
        ("stage0_scope_conflict_stopped_the_run_before_serialization", "exact",
         ("stage0_scope_conflict_stopped_the_run_before_serialization",), "KI-025", False),
        ("stage1_empty_chunk", "exact", ("stage1_empty_chunk",), "KI-027", False),
        ("typo_snapped", "exact", ("typo_snapped",), "KI-039", False),
        ("unaccounted_locked_reactions:<n>", "prefix", ("unaccounted_locked_reactions:",), "KI-016", False),
        ("undeclared_entity_in_<slot>", "prefix", ("undeclared_entity_in_",), "KI-015", False),
        ("undefined_state_treated_as_absent", "exact", ("undefined_state_treated_as_absent",), "KI-039", False),
        ("unexportable_entity:<n>", "prefix", ("unexportable_entity:",), "KI-016", False),
        ("unnamed_actor_in_<slot>", "prefix", ("unnamed_actor_in_",), "KI-015", False),
        ("unresolved_actor_in_<slot>", "prefix", ("unresolved_actor_in_",), "KI-015", False),
        ("untyped_interaction", "exact", ("untyped_interaction",), "KI-008", False),
        ("valid_json_zero_processes", "exact", ("valid_json_zero_processes",), "KI-027", False),
    ),
    "messages": (
        (("Chunk <n> failed to produce valid JSON.",), "KI-027"),
        (("Composite entity '<name>' in /entities/compounds has no protein-like left component; unsupported.",), "KI-024"),
        (("Composite token '<text>' at <pointer> has no protein-like left component; ...",), "KI-024"),
        (("DraftGraph has no reaction nodes — nothing to render.",), "KI-020"),
        (("extraction did not finish inside the time budget", "audit and DB mapping ...", "PWML export ..."), "KI-028"),
        (("AppTest script run timed out after 3600.0(s)",), "KI-028"),
        (("LLM returned an empty completion for <stage> (model <m>, finish_reason=length, ...)",), "KI-054"),
        (("Generated protein complex '<c>' component '<p>' is missing positive stoichiometry.",), "KI-012"),
        (("Generated protein complex '<c>' component protein '<p>' is missing a UniProt or DrugBank identifier.",), "KI-001"),
        (("No refinement graph image is available.",), "KI-020"),
        (("no UniProt or DrugBank identifier; other enzymes remain",), "KI-007"),
        (("Protein '<name>' is missing a UniProt or DrugBank identifier.",), "KI-001"),
        (("Protein has degree 0 after normalization: <name>",), "KI-013"),
        (("Located protein is isolated in connectivity graph: <name>",), "KI-013"),
        (("PWML export stopped by the quarantine boundary: <reasons>",), "KI-016"),
        (("PWML export stopped by validation-only pre-export Stage 3 revalidation; Stage 8 did not repair the payload.",),
         "KI-023"),
        (("PWML required-field gate failed.",), "KI-006"),
        (("Unknown protein/modifier reference: <name>",), "KI-004"),
        (("Unknown transporter reference: <name>",), "KI-004"),
        (("UniProt <acc>, from strain <strain>",), "KI-033"),
    ),
    "workarounds": {
        "KI-001": "Search UniProt for the protein in the paper's organism. If an entry exists, create the protein in "
                  "PathWhiz with that accession and attach it as the enzyme (or interaction partner) of the listed "
                  "process. If none exists, PathWhiz cannot hold the protein; leave the process out.",
        "KI-002": "MenI and MenA: add them in PathWhiz by hand only if you accept an IL1403 record (UniProt Q9CHK5 for "
                  "MenI is the identical sequence; ORCH-761 § 0 item 3); otherwise leave them out.",
        "KI-003": "Add the protein in PathWhiz under the paper's organism, then attach it to the listed process.",
        "KI-004": "For each listed reaction, add the enzyme in PathWhiz as a protein (with its UniProt id) and attach "
                  "it; the text record names the actor that was refused.",
        "KI-006": "In PathWhiz, create a one-protein complex for the enzyme (same UniProt id) and set it as the "
                  "reaction's enzyme.",
        "KI-007": "Compare the enzymes named in the text record's quote with the PWML's; add any missing enzyme in "
                  "PathWhiz.",
        "KI-008": "None inside PathWhiz's model: it has no type for these relations. Read them in the text record, and "
                  "add an Activation or Inhibition by hand only where the paper states the direction.",
        "KI-009": "Give the protein its UniProt id and species in PathWhiz, make a one-protein complex of it, and draw "
                  "the interaction from or to that complex.",
        "KI-010": "Read the quote: if the paper means the protein, draw the interaction from the protein's complex; if "
                  "it means the gene (e.g. \"Fur represses entD\"), draw it to the gene.",
        "KI-011": "Add the gene (nucleic acid) and the protein as separate elements in PathWhiz, and redraw each listed "
                  "regulation to the one the quote means.",
        "KI-012": "Delete the \"enterobactin synthase complex\" catalyst; the paper's own enzyme (EntF) is the catalyst.",
        "KI-013": "Delete the stray protein element from the PathWhiz pathway; it has no reactions.",
        "KI-014": "None needed; check that the kept twin is in the PWML.",
        "KI-015": "Read the process in the text record and draw it in PathWhiz if the quote supports it.",
        "KI-016": "None for a user (no file is written). Operator: read `quarantine_report.json` `strict_invariants` "
                  "for the entity named, and see its cause entry.",
        "KI-017": "If a PWML shows EntB (or another protein) without its UniProt id, set it in PathWhiz (EntB: P0ADI4).",
        "KI-019": "None needed after the fix.",
        "KI-020": "Batch runs are unaffected. In the app, refine such a pathway by hand rather than with \"Submit "
                  "Changes to AI\".",
        "KI-021": "Expect a thin PWML: add the withheld interactions (KI-008) by hand where the paper gives a direction.",
        "KI-022": "None for a user (no file). Operator: check whether the leg's reactions lost their "
                  "`provenance_lineage` (an indeterminate carrier is not refused).",
        "KI-023": "None for a user (no file).",
        "KI-024": "Check ions in the PWML (e.g. `Ca2+`) keep their charge; correct the compound in PathWhiz if not.",
        "KI-025": "Re-run with the requested pathway written in the paper's own phrase.",
        "KI-027": "None needed after the fix.",
        "KI-028": "None for a user. Operator: give long papers 10,800 s and re-run a timed-out leg.",
        "KI-029": "In PathWhiz, keep the compound as \"prenyl diphosphate\" (or attach the polyprenyl diphosphate class "
                  "if you curate one); do not accept DMAPP.",
        "KI-033": "Review each flagged protein; keep it if the ortholog is acceptable for your curation, otherwise "
                  "remove its accession and treat it as KI-001.",
        "KI-038": "None needed in PathWhiz; read the coverage figure as a lower bound.",
        "KI-039": "In the text record, rows marked 1/3 support are the likely duplicates; delete a PWML reaction that "
                  "repeats another with a different spelling.",
        "KI-041": "Merge the two MenA rows and the two MenG rows by hand in PathWhiz (delete the `DMK` duplicates).",
        "KI-043": "None needed; review merges flagged `nested_location_merged`.",
        "KI-054": "Operator: keep `LLM_MAX_TOKENS` and the stage variables at 128,000 (extraction 64,000).",
        "KI-056": "None.",
    },
}


def _matches(text: str, match: str, patterns: tuple) -> bool:
    if match == "exact":
        return text == patterns[0]
    if match == "prefix":
        return text.startswith(patterns[0])
    if match == "suffix":
        return text.endswith(patterns[0])
    return text.startswith(patterns[0]) and text.endswith(patterns[1])  # prefix+suffix


def _message_regex(pattern: str) -> "re.Pattern[str]":
    # A ``<placeholder>`` or a trailing ``...`` stands for the varying text; the rest is fixed words.
    parts = re.split(r"<[^<>]*>|\.\.\.", pattern)
    return re.compile(".*?".join(map(re.escape, parts)), re.S)


#: Exact codes first, then the longest pattern; messages with the most fixed text first, so a broad
#: pattern ("PWML export ...") never takes a message a narrower one names.
_CODE_ORDER = sorted(KNOWN_ISSUES["codes"], key=lambda r: (r[1] != "exact", -len("".join(r[2])), r[0]))
_MESSAGE_ORDER = sorted(
    ((_message_regex(p), len(re.sub(r"<[^<>]*>|\.\.\.", "", p)), ki) for ps, ki in KNOWN_ISSUES["messages"] for p in ps),
    key=lambda r: -r[1],
)


def known_issue_id(reason: Any) -> str:
    """The catalogue id of one recorded reason, matched as the appendix says; never guessed."""

    text = _text(reason)
    for _code, match, patterns, ki, inner in _CODE_ORDER:
        if text and _matches(text, match, patterns):
            # "then label by <reason>": the code wraps another reason, after its last colon.
            refined = known_issue_id(text.rsplit(":", 1)[1]) if inner and ":" in text else KI_UNCLASSIFIED
            return refined if refined != KI_UNCLASSIFIED else ki
    for regex, _fixed, ki in _MESSAGE_ORDER:
        if text and regex.fullmatch(text):
            return ki
    return KI_UNCLASSIFIED

#: Identifier fields shown verbatim, by their payload key, so nothing is relabelled.
_ID_FIELDS = (
    "pathbank_compound_id",
    "chebi_id",
    "kegg_id",
    "hmdb_id",
    "uniprot",
    "pathbank_protein_id",
    "pathbank_protein_complex_id",
)


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _show(value: Any) -> str:
    """One scalar for display. ``False`` and ``0`` are values; only absence is Not recorded."""

    if value is None or value == "":
        return NOT_RECORDED
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    return _text(value)


def _texts(value: Any) -> Optional[List[str]]:
    """A recorded list of strings, or ``None`` when the artifact does not carry one."""

    return [_text(v) for v in value if _text(v)] if isinstance(value, list) else None


def _norm(name: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", _text(name)).casefold().split())


def _participant(entry: Any) -> str:
    """A bare name, or ``{name, stoichiometry}`` rendered as ``"2 ATP"``."""

    if not isinstance(entry, dict):
        return _text(entry)
    name = _text(entry.get("name"))
    stoich = _text(entry.get("stoichiometry"))
    return f"{stoich} {name}".strip() if stoich and stoich not in {"1", "1.0"} else name


def _side(row: Mapping[str, Any], key: str) -> List[str]:
    return [p for p in (_participant(e) for e in _list(row.get(key))) if p]


def _participant_names(row: Mapping[str, Any]) -> List[str]:
    """Participant NAMES, without stoichiometry, for the entity lookup."""

    names = [
        _text(_dict(e).get("name")) if isinstance(e, dict) else _text(e)
        for key in ("inputs", "outputs")
        for e in _list(row.get(key))
    ]
    return list(dict.fromkeys(n for n in names if n))


def _equation(inputs: List[str], outputs: List[str]) -> str:
    # Built from the lists: names like "NAD+" contain "+", so this string is
    # display-only and is never split back apart.
    return (
        f"{' + '.join(inputs) or '(no substrates recorded)'} -> "
        f"{' + '.join(outputs) or '(no products recorded)'}"
    )


def _quotes(row: Mapping[str, Any]) -> List[str]:
    """Verbatim evidence strings on a process row and its enzymes, first-seen order."""

    found: List[str] = []

    def add(value: Any) -> None:
        if isinstance(value, dict):
            value = value.get("quote") or value.get("span") or value.get("text")
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, (dict, list)):
                add(item)
            elif _text(item) and _text(item) not in found:
                found.append(_text(item))

    add(row.get("evidence"))
    for enzyme in _list(row.get("enzymes")):
        add(_dict(enzyme).get("evidence"))
    return found


def _lineage(row: Mapping[str, Any]) -> List[str]:
    out = []
    for step in map(_dict, _list(row.get("provenance_lineage"))):
        text = f"{_show(step.get('stage'))}/{_show(step.get('origin'))} (paper_explicit={_show(step.get('paper_explicit'))}"
        out.append(text + (", review_required)" if step.get("review_required") is True else ")"))
    return out


def _sources(row: Mapping[str, Any], paper_id: str) -> List[str]:
    out: List[str] = []
    rag = _dict(row.get("rag_provenance"))
    if rag:
        out.append(
            f"RAG {_show(rag.get('source_id'))}; section={_show(rag.get('section'))}; "
            f"chunk={_show(rag.get('chunk_id'))}"
        )
    for paper in map(_dict, _list(row.get("source_papers"))):
        out.append(" ".join(p for p in (_text(paper.get(k)) for k in ("source_id", "title", "uri")) if p))
    lineage = [_dict(step) for step in _list(row.get("provenance_lineage"))]
    for source in (_dict(s) for step in lineage for s in _list(step.get("sources"))):
        if source:
            out.append(
                f"{_show(source.get('source_type'))} {_show(source.get('source_id'))} "
                f"({_show(source.get('locator'))})"
            )
    if any(step.get("stage") == "paper_extraction" for step in lineage):
        out.append(f"supplied paper {paper_id} via paper_extraction")
    return list(dict.fromkeys(line for line in out if line))


def _entity_index(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for bucket, rows in _dict(payload.get("entities")).items():
        if bucket not in ("species", "subcellular_locations"):
            for row in map(_dict, _list(rows)):
                index.setdefault(_norm(row.get("name")), row)
    return index


def _borrowed_flags(payload: Mapping[str, Any]) -> Dict[str, str]:
    """D-108 § 3 (ORCH-762): name -> "UniProt Q9CHK2, from strain IL1403", as the mapping recorded it."""

    return {
        _norm(row.get("name")): _text(row.get("flag"))
        for row in map(_dict, _list(payload.get(BORROWED_PROTEIN_IDENTITIES_KEY)))
        if _text(row.get("name")) and _text(row.get("flag"))
    }


def _borrowed_for(name: str, index: Mapping[str, Dict[str, Any]], borrowed: Mapping[str, str]) -> str:
    """The flag of ``name`` itself, or of each component when ``name`` is a complex (its wrapper)."""

    flags = [borrowed[_norm(name)]] if _norm(name) in borrowed else []
    for component in _list(_dict(index.get(_norm(name))).get("components")):
        component_name = _text(_dict(component).get("name") if isinstance(component, dict) else component)
        if _norm(component_name) in borrowed and _norm(component_name) != _norm(name):
            flags.append(f"{component_name}: {borrowed[_norm(component_name)]}")
    return "; ".join(flags)


def _mapping(name: str, entity: Mapping[str, Any], borrowed: str = "") -> str:
    if not entity:
        return f"{name}: {NOT_RECORDED}"
    parts = [_show(_dict(_dict(entity.get("mapping_meta")).get("resolution")).get("status"))]
    if _text(entity.get("db_status")):
        parts.append(f"db_status={_text(entity['db_status'])}")
    ids = [f"{key}={_text(entity[key])}" for key in _ID_FIELDS if _text(entity.get(key))]
    parts.append(", ".join(ids) or "no identifier recorded")
    if borrowed:
        parts.append(f"borrowed identity: {borrowed}")
    return f"{name}: " + "; ".join(parts)


def _enzyme(enzyme: Mapping[str, Any], index: Mapping[str, Dict[str, Any]]) -> str:
    name = _text(enzyme.get("entity") or enzyme.get("name"))
    components = [_text(_dict(c).get("name")) for c in _list(_dict(index.get(_norm(name))).get("components"))]
    components = [c for c in components if c]
    detail = [_text(enzyme.get("entity_type"))] + ([f"components: {', '.join(components)}"] if components else [])
    detail = [d for d in detail if d]
    return f"{name} ({'; '.join(detail)})" if detail else name


def _same_process(lock_id: str, name: Any, row: Mapping[str, Any]) -> bool:
    if lock_id and row["lock_id"]:
        return lock_id == row["lock_id"]
    return bool(_norm(name)) and _norm(name) == _norm(row["name"])


#: ORCH-754b / D-104. What each exporter reason means, worded to be true whatever led
#: to it -- in particular ``bare_protein_endpoint`` says only that no wrapper exists,
#: never why Stage 6 did not build one.
_WITHHELD_GLOSS = {
    "untyped_interaction": (
        "the relationship word is neither Activation nor Inhibition, the only interaction types PathWhiz stores"
    ),
    "bare_protein_endpoint": (
        "a protein endpoint is not wrapped in a single-protein complex in this payload, "
        "and PathWhiz has no bare-protein interaction member"
    ),
    "ambiguous_protein_endpoint": (
        "a protein endpoint's name also names another kind of entity in this payload, "
        "so it is not known to be the protein and is not drawn as one"
    ),
}
_MISSING = {"species": "species", "external_identity": "UniProt/DrugBank identity"}


def _pwml_verdict(row: Mapping[str, Any]) -> str:
    """One IR interaction row's export verdict, as the IR states it."""

    status = _text(row.get("export_status"))
    if status == "exportable":
        return f"drawn as {_text(row.get('interaction_type')) or NOT_RECORDED}"
    if status != "withheld":
        return f"{NOT_RECORDED} (this IR predates per-row export verdicts)"
    withheld = _dict(row.get("withheld"))
    reason = _text(withheld.get("reason")) or NOT_RECORDED
    gloss = _WITHHELD_GLOSS.get(reason, "")
    lacking = [
        f"{_text(u.get('name')) or NOT_RECORDED} has no "
        + " and ".join(_MISSING.get(_text(m), _text(m)) for m in _list(u.get("missing")))
        for u in map(_dict, _list(withheld.get("unwrappable_proteins")))
    ]
    if lacking:
        gloss = "; ".join(lacking) + ", which a single-protein complex wrapper requires"
    shared = [
        f"{_text(a.get('name')) or NOT_RECORDED} also names a "
        + " and a ".join(_text(k).replace("_", " ") for k in _list(a.get("also_names")))
        for a in map(_dict, _list(withheld.get("ambiguous_endpoints")))
    ]
    if shared:
        gloss = "; ".join(shared) + " in this payload, so it is not known to be the protein and is not drawn as one"
    return f"withheld -- {reason}" + (f": {gloss}" if gloss else "")


def _one_line(value: Any) -> str:
    return " ".join(_text(value).split())


def _ir_norm(value: Any) -> str:
    # The exporter's name key (``ir._norm``), used only to see whether the entity an IR row
    # was bound to carries the name the payload row writes.
    return re.sub(r"[^a-z0-9:+ ]+", " ", _one_line(value).casefold()).strip()


def _same_row(row: Mapping[str, Any], ir_row: Mapping[str, Any], ir_entities: Mapping[str, Any]) -> bool:
    """Was ``ir_row`` built from ``row``? Word and both endpoints, as the exporter read them."""

    word = interaction_relationship_word(row)
    left = _one_line(row.get("left") or row.get("entity_1") or row.get("source"))
    right = _one_line(row.get("right") or row.get("entity_2") or row.get("target"))
    if _text(ir_row.get("export_status")) == "withheld":
        withheld = _dict(ir_row.get("withheld"))
        return (_text(withheld.get("raw_interaction_type")), _text(withheld.get("left")), _text(withheld.get("right"))) == (
            word, left, right)
    # A drawn row keeps the sign, not the word, and the entity each side was bound to.
    if pwml_interaction_type(word) != _text(ir_row.get("interaction_type")):
        return False
    for side, written in (("left", left), ("right", right)):
        entity = _dict(ir_entities.get(_text(_dict(ir_row.get(side)).get("entity_key"))))
        names = [entity.get(f) for f in ("name", "raw_name", "short_name", "common_name")]
        if _ir_norm(written) not in {_ir_norm(n) for n in [*names, *_list(entity.get("synonyms"))] if _ir_norm(n)}:
            return False
    return True


def _interactions(payload: Mapping[str, Any], ir: Any, ir_source: str, no_pwml: str) -> List[Dict[str, Any]]:
    """Every interaction in the final payload, with the IR's own verdict on it.

    ``no_pwml`` is why no PWML was written, or ``""`` when one was.
    """

    rows = [(i, r) for i, r in enumerate(_list(_dict(payload.get("processes")).get("interactions"))) if isinstance(r, dict)]
    ir_rows = _list(_dict(_dict(ir).get("processes")).get("interactions")) if isinstance(ir, dict) else None
    # The IR keeps one row per dict row, in order, named as below (``ir.build_pwml_ir``).
    # Anything else means the IR was built from a different payload: no verdict is shown.
    lined_up = ir_rows is not None and len(ir_rows) == len(rows) and all(
        _text(_dict(v).get("name")) == (" ".join(_text(r.get("name")).split()) or f"Interaction {k}")
        for k, ((_i, r), v) in enumerate(zip(rows, ir_rows), 1)
    )
    ir_entities = {
        _text(e.get("key")): e
        for bucket in _dict(_dict(ir).get("entities")).values()
        for e in map(_dict, _list(bucket))
        if _text(e.get("key"))
    }
    out: List[Dict[str, Any]] = []
    for k, (index, row) in enumerate(rows, 1):
        ir_row = _dict(ir_rows[k - 1]) if lined_up else {}
        # D-114: True only where the IR's own verdict says drawn in a written PWML; no PWML -> False;
        # None = not recorded (the verdict line says why).
        drawn: Optional[bool] = False if no_pwml else None
        if ir_rows is None:
            verdict = f"{NOT_RECORDED} (no {PWML_IR_FILENAME}: no PWML IR was built for this run)"
        elif not lined_up:
            verdict = f"{NOT_RECORDED} ({ir_source} does not line up with the interactions of the final payload)"
        elif _text(ir_row.get("export_status")) not in ("exportable", "withheld"):
            verdict = _pwml_verdict(ir_row)  # an IR older than per-row verdicts says so
        elif no_pwml:
            verdict = f"{NOT_RECORDED} (no PWML written: {no_pwml})"
        elif _text(ir_row.get("export_status")) == "exportable" and not all(
            isinstance(ir_row.get(side), dict) for side in ("left", "right")
        ):
            verdict = f"{NOT_RECORDED} (the IR resolved no entity for an endpoint of this row)"
        elif not _same_row(row, ir_row, ir_entities):
            verdict = f"{NOT_RECORDED} ({ir_source} row {k} was built from a different relationship or endpoints)"
        else:
            verdict = _pwml_verdict(ir_row)
            drawn = _text(ir_row.get("export_status")) == "exportable"
        out.append(
            {
                "number": k,
                "pointer": f"/processes/interactions/{index}",
                "name": _text(row.get("name")),
                "relationship": _text(row.get("relationship")),
                "source": _text(row.get("entity_1") or row.get("left") or row.get("source")),
                "target": _text(row.get("entity_2") or row.get("right") or row.get("target")),
                "state": _text(row.get("biological_state")),
                "pwml": verdict,
                "drawn": drawn,
                "withheld_reason": "" if drawn is not False or no_pwml else _text(_dict(ir_row.get("withheld")).get("reason")),
                "evidence": _quotes(row),
            }
        )
    return out


_KIND_OF_BUCKET = {"reactions": "reaction", "transports": "transport", "interactions": "interaction"}
_TEXT_LABEL = {"interaction": "Endpoints", "transport": "Cargo"}


def _process_text(kind: str, row: Mapping[str, Any]) -> str:
    """Equation (reaction), endpoints (interaction) or cargo and states (transport), display only."""

    if kind == "interaction":
        ends = [_text(row.get(a) or row.get(b) or row.get(c)) or NOT_RECORDED
                for a, b, c in (("entity_1", "left", "source"), ("entity_2", "right", "target"))]
        word = _text(row.get("relationship") or row.get("interaction_type")) or NOT_RECORDED
        return f"{ends[0]} -[{word}]-> {ends[1]}"
    if kind == "transport":
        cargo = [_text(_dict(c).get("element") or _dict(c).get("cargo") or _dict(c).get("name")) if isinstance(c, dict)
                 else _text(c) for c in _list(row.get("transport_elements"))] or [_text(row.get("cargo") or row.get("cargo_complex"))]
        states = [_text(row.get(k)) or NOT_RECORDED for k in ("from_biological_state", "to_biological_state")]
        return f"{', '.join(c for c in cargo if c) or '(no cargo recorded)'}: {states[0]} -> {states[1]}"
    inputs, outputs = _side(row, "inputs"), _side(row, "outputs")
    return _equation(inputs, outputs) if (inputs or outputs) else ""


def _omission_detail(row: Mapping[str, Any]) -> str:
    """``detail.omission`` as strict quarantine records it (ORCH-773), verbatim, or ``""``."""

    cause = _dict(_dict(row.get("detail")).get("omission"))
    parts = [f"{key}={_show(cause.get(key))}" for key in ("gate", "cause", "slot", "name") if key in cause]
    failing = cause.get("failing_entity")
    if isinstance(failing, dict):
        parts.append(f"failing entity={_show(failing.get('bucket'))} {_show(failing.get('name'))}")
    elif failing:
        parts.append(f"failing entity={_show(failing)}")
    return "; ".join(parts)


def _entry(kind: str, name: Any, where: str, text: str, source: str, reason: str, message: str = "",
           omission: str = "", group: str = "pre_freeze", ki_of: Optional[str] = None) -> Dict[str, Any]:
    ki = known_issue_id(reason if ki_of is None else ki_of)
    return {
        "kind": kind, "name": _text(name), "where": where, "text": text, "source": source,
        "reason": reason, "message": message, "omission": omission, "ki": ki,
        # pre_freeze | ir (a written PWML does not draw it) | no_pwml | not_recorded
        "group": group,
        "counted": kind in COVERAGE_KINDS and ki != MERGED_DUPLICATE_KI,
        "merged_duplicate": ki == MERGED_DUPLICATE_KI,
    }


def _same_ir_rows(ir: Any, bucket: str, rows: List[Dict[str, Any]], label: str) -> bool:
    """The IR keeps one row per dict row, in order (``ir.build_pwml_ir``); else it is another payload's."""

    ir_rows = _list(_dict(_dict(ir).get("processes")).get(bucket)) if isinstance(ir, dict) else None
    return ir_rows is not None and len(ir_rows) == len(rows) and all(
        _text(_dict(v).get("name")) == (" ".join(_text(r.get("name")).split()) or f"{label} {k}")
        for k, (r, v) in enumerate(zip(rows, ir_rows), 1)
    )


def _omissions(
    payload: Mapping[str, Any],
    quarantine: Any,
    removed: Any,
    ir: Any,
    ir_source: str,
    interactions: List[Dict[str, Any]],
    no_pwml: str,
) -> Dict[str, Any]:
    """D-114 §§ 1, 3: every process of the text record the PWML does not draw, and the coverage.

    Pre-freeze removals are the strict quarantine's ``quarantined[]`` rows (only when it was
    applied: research mode and a refusal leave them in the payload read) and the payload's
    ``quarantined_locked_reactions`` entries it does not already list. A payload process is
    drawn only when a PWML was written and the IR says so; nothing here recomputes a verdict.
    """

    research = _dict(quarantine).get("research_mode")
    applied = (isinstance(quarantine, dict) and quarantine.get("ok") is not False
               and not (isinstance(research, dict) and research.get("applied") is False))
    entries: List[Dict[str, Any]] = []
    strict_locks = set()
    for row in map(_dict, _list(_dict(quarantine).get("quarantined"))):
        original = _dict(row.get("original_process"))
        lock = _text(original.get("locked_reaction_id"))
        strict_locks.add(lock)
        if applied:
            kind = _text(row.get("process_kind")) or NOT_RECORDED
            where = "; ".join(p for p in (f"pre-quarantine {_text(row.get('pointer'))}" if _text(row.get("pointer")) else "",
                                          f"lock {lock}" if lock else "") if p)
            entries.append(_entry(kind, row.get("name") or original.get("name"), where, _process_text(kind, original),
                                  f"{QUARANTINE_REPORT_FILENAME} quarantined", _text(row.get("reason")),
                                  omission=_omission_detail(row)))
    for record in map(_dict, _list(payload.get("quarantined_locked_reactions"))):
        original = _dict(record.get("original_reaction"))
        lock = _text(record.get("locked_reaction_id") or original.get("locked_reaction_id"))
        if lock and lock in strict_locks:
            continue
        where = "; ".join(p for p in (f"json_pointer {_text(record.get('json_pointer'))}" if _text(record.get("json_pointer")) else "",
                                      f"lock {lock}" if lock else "") if p)
        entries.append(_entry("reaction", record.get("reaction_name") or original.get("name"), where,
                              _process_text("reaction", original), LOCKED_RECORD, _text(record.get("reason"))))

    processes = _dict(payload.get("processes"))
    for bucket, label in (("reactions", "Reaction"), ("transports", "Transport")):
        rows = [r for r in _list(processes.get(bucket)) if isinstance(r, dict)]
        if no_pwml:
            group, message = "no_pwml", f"no PWML written: {no_pwml}"
        elif _same_ir_rows(ir, bucket, rows, label):
            continue  # the writer draws every IR reaction and transport row
        elif not isinstance(ir, dict):
            group, message = "not_recorded", f"{NOT_RECORDED} (no {PWML_IR_FILENAME}: no PWML IR was built for this run)"
        else:
            group, message = "not_recorded", f"{NOT_RECORDED} ({ir_source} does not line up with the {bucket} of the final payload)"
        kind = _KIND_OF_BUCKET[bucket]
        for index, row in enumerate(_list(processes.get(bucket))):
            if isinstance(row, dict):
                entries.append(_entry(kind, row.get("name"), f"final payload /processes/{bucket}/{index}",
                                      _process_text(kind, row), "", "", message, group=group,
                                      ki_of=no_pwml if no_pwml else ""))
    for item, row in zip(interactions, (r for r in _list(processes.get("interactions")) if isinstance(r, dict))):
        if item["drawn"] is True:
            continue
        group = "no_pwml" if no_pwml else ("ir" if item["drawn"] is False else "not_recorded")
        reason = item["withheld_reason"] or (NOT_RECORDED if group == "ir" else "")
        message = f"no PWML written: {no_pwml}" if no_pwml else item["pwml"]
        entries.append(_entry("interaction", item["name"], f"final payload {item['pointer']}",
                              _process_text("interaction", row), ir_source if group == "ir" else "",
                              reason, message, group=group, ki_of=no_pwml if no_pwml else None))

    drawn: Optional[int] = 0
    total = 0
    by_kind: Dict[str, Dict[str, Any]] = {}
    for bucket, kind in _KIND_OF_BUCKET.items():
        in_payload = sum(isinstance(r, dict) for r in _list(processes.get(bucket)))
        mine = [e for e in entries if e["kind"] == kind and e["counted"]]
        removed_before = sum(e["group"] == "pre_freeze" for e in mine)
        unknown = any(e["group"] == "not_recorded" for e in mine)
        drawn_k = None if unknown else in_payload - sum(e["group"] != "pre_freeze" for e in mine)
        by_kind[kind] = {"drawn": drawn_k, "total": in_payload + removed_before, "omitted_before_freeze": removed_before}
        total += in_payload + removed_before
        drawn = None if drawn is None or drawn_k is None else drawn + drawn_k
    why = ""
    if not payload:
        drawn, why = None, f"no final payload ({FINAL_MAPPED_FILENAME})"
    elif drawn is None:
        why = "the PWML verdict of some process is not recorded; they are listed below"
    by_design = sum(e["counted"] and e["ki"] == BY_DESIGN_KI and e["group"] == "ir" for e in entries)
    entities = [
        {
            "bucket": _show(e.get("bucket")), "name": _show(e.get("name")), "reason": _show(e.get("reason")),
            "ki": known_issue_id(e.get("reason")), "gate": _text(_dict(e.get("detail")).get("gate")),
            "actor_removed_from": [
                f"{_show(a.get('process_name'))} ({_show(a.get('process'))}): {', '.join(_texts(a.get('slots')) or [])}"
                + (f"; catalysts kept: {', '.join(_texts(a.get('catalysts_kept')) or []) or 'none'}"
                   if "catalysts_kept" in a else "")
                for a in map(_dict, _list(_dict(e.get("detail")).get("actor_removed_from")))
            ],
        }
        for e in map(_dict, _list(_dict(removed).get("removed_entities")))
    ] if isinstance(removed, dict) else None
    # ORCH-773: an actor left out of a process that is still drawn (D-113 s1 inferred wrapper,
    # D-114 AMENDMENT 1 s2 transporter). Listed under OMITTED; the process counts as drawn.
    actors = [
        {
            "process": _show(a.get("process_name")), "where": _text(a.get("process")),
            "kind": "transport" if "transporters" in (_texts(a.get("slots")) or []) else "reaction",
            "actor": _show(e.get("name")), "slots": ", ".join(_texts(a.get("slots")) or []) or NOT_RECORDED,
            "reason": _show(e.get("reason")), "ki": known_issue_id(e.get("reason")),
            "gate": _text(_dict(e.get("detail")).get("gate")), "decision": _text(a.get("decision")),
            "kept": _texts(a.get("catalysts_kept")),
        }
        for e in map(_dict, _list(_dict(removed).get("removed_entities")))
        for a in map(_dict, _list(_dict(e.get("detail")).get("actor_removed_from")))
    ] if isinstance(removed, dict) else []
    return {
        "coverage": {
            "pwml_written": not no_pwml, "no_pwml_cause": no_pwml, "drawn": drawn, "total": total, "not_recorded": why,
            "by_kind": by_kind, "by_design_withheld": by_design,
            "merged_duplicates": sum(e["merged_duplicate"] for e in entries),
            "not_counted": sum(not e["counted"] for e in entries),
        },
        "omitted": entries,
        "omitted_entities": entities,
        "omitted_actors": actors,
    }


def load_run_artifacts(directory: Any) -> Dict[str, Any]:
    """Read whichever of the known artifacts exist in ``directory``. Missing = absent key."""

    root = Path(directory)
    found: Dict[str, Any] = {}
    for name in JSON_ARTIFACTS:
        try:
            found[name] = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    try:
        with open(root / LEG_TRACE_FILENAME, encoding="utf-8") as handle:
            found[LEG_TRACE_FILENAME] = json.loads(handle.readline())
    except (OSError, ValueError):
        pass
    for name in PWML_FILENAMES:  # presence only: the report never reads a PWML
        if (root / name).is_file():
            found[name] = True
    return found


def build_report(
    artifacts: Mapping[str, Any],
    *,
    payload_source: str = FINAL_MAPPED_FILENAME,
    ir_source: str = PWML_IR_FILENAME,
    pwml_written: Optional[bool] = None,
    no_pwml_cause: str = "",
) -> Dict[str, Any]:
    """The review model, a plain dict built only from ``artifacts`` (``{filename: parsed}``)."""

    coverage = _dict(artifacts.get(COVERAGE_REPORT_FILENAME))
    quarantine = artifacts.get(QUARANTINE_REPORT_FILENAME)
    removed = artifacts.get(REMOVED_ENTITY_REPORT_FILENAME)
    payload = _dict(artifacts.get(FINAL_MAPPED_FILENAME))
    context = _dict(coverage.get("requested_context"))
    paper_id = _show(_dict(artifacts.get(LEG_TRACE_FILENAME)).get("paper_id"))
    index = _entity_index(payload)
    borrowed = _borrowed_flags(payload)
    processes = _dict(payload.get("processes"))

    admitted: Dict[str, set] = {}
    for row in map(_dict, _list(_dict(quarantine).get("accepted"))):
        if row.get("bucket") == "reactions":
            admitted.setdefault(_norm(row.get("name")), set()).add(_text(row.get("state")))

    strict_rows: Optional[List[Dict[str, Any]]] = None
    if isinstance(quarantine, dict):
        strict_rows = []
        for row in map(_dict, _list(quarantine.get("quarantined"))):
            # ``detail.row`` carries ELIDED evidence ("123 chars elided"); only the
            # verbatim ``original_process`` is ever quoted.
            original = _dict(row.get("original_process"))
            inputs, outputs = _side(original, "inputs"), _side(original, "outputs")
            state, reason = _text(row.get("state")), _text(row.get("reason"))
            essential = _text(row.get("essential_participant"))
            strict_rows.append(
                {
                    "source": "",
                    "name": _text(row.get("name") or original.get("name")),
                    "pointer": f"pre-quarantine {_text(row['pointer'])}" if _text(row.get("pointer")) else "",
                    "lock_id": _text(original.get("locked_reaction_id")),
                    "kind": _text(row.get("process_kind")),
                    "equation": _equation(inputs, outputs) if (inputs or outputs) else "",
                    "state": state,
                    "reason": reason,
                    "essential_participant": essential,
                    "fields": [("State", state), ("Reason", reason), ("Essential participant", essential)],
                    "evidence": _quotes(original),
                }
            )

    # The payload's quarantined_locked_reactions record has three writers (the
    # locked-reaction filter, dedupe_processes, strict quarantine); the entry's own
    # action / quarantine_stage, when recorded, is the only stage shown. A lock
    # strict quarantine already lists is shown once.
    locked_records = payload.get("quarantined_locked_reactions")
    strict_locks = {row["lock_id"] for row in strict_rows or [] if row["lock_id"]}
    locked_rows: List[Dict[str, Any]] = []
    already_listed: List[str] = []
    for entry in map(_dict, _list(locked_records)):
        original = _dict(entry.get("original_reaction"))
        lock_id = _text(entry.get("locked_reaction_id") or original.get("locked_reaction_id"))
        if lock_id in strict_locks:
            already_listed.append(lock_id)
            continue
        inputs, outputs = _side(original, "inputs"), _side(original, "outputs")
        fields = [("Reason", _text(entry.get("reason")))]
        if isinstance(entry.get("missing_entities"), list):
            fields.append(("Missing entities", ", ".join(_texts(entry["missing_entities"])) or NONE_RECORDED))
        for key, label in (("action", "Action"), ("quarantine_stage", "Quarantine stage"), ("admission_state", "Admission state")):
            if _text(entry.get(key)):
                fields.append((label, _text(entry[key])))
        locked_rows.append(
            {
                "source": LOCKED_RECORD,
                "name": _text(entry.get("reaction_name") or original.get("name")),
                "pointer": f"json_pointer {_text(entry['json_pointer'])}" if _text(entry.get("json_pointer")) else "",
                "lock_id": lock_id,
                "kind": "",
                "equation": _equation(inputs, outputs) if (inputs or outputs) else "",
                "state": _text(entry.get("admission_state")),
                "reason": _text(entry.get("reason")),
                "essential_participant": "",
                "fields": fields,
                "evidence": _quotes(original),
            }
        )
    quarantined: Optional[List[Dict[str, Any]]] = None
    if strict_rows is not None or isinstance(locked_records, list):
        quarantined = [*(strict_rows or []), *locked_rows]
    locked_line = f"From {LOCKED_RECORD}: "
    if not isinstance(locked_records, list):
        locked_line += NOT_RECORDED
    else:
        locked_line += f"{len(locked_records)}"
        if already_listed:
            locked_line += (
                f" ({len(already_listed)} already listed from {QUARANTINE_REPORT_FILENAME} "
                f"by lock id: {', '.join(already_listed)})"
            )
    quarantine_sources = [f"From {QUARANTINE_REPORT_FILENAME} quarantined: {_count(strict_rows)}", locked_line]

    # Whether the quarantine decision was applied to the payload read above, in the
    # report's own terms. Research mode flags without removing; ok=false refused it.
    research = quarantine.get("research_mode") if isinstance(quarantine, dict) else None
    release_status = _show(_dict(_dict(quarantine).get("release")).get("status"))
    surviving_note: List[str] = []
    quarantined_note: List[str] = []
    if isinstance(research, dict) and research.get("applied") is False:
        surviving_note.append("research_mode.applied=false: quarantine removed nothing; reactions it flagged are still listed here")
        quarantined_note.append(f"research_mode.applied=false: {QUARANTINE_REPORT_FILENAME} rows are review flags, not removals")
    if _dict(quarantine).get("ok") is False:
        refused = f"{QUARANTINE_REPORT_FILENAME} ok=false: payload not released (release status {release_status})"
        surviving_note.append(refused)
        quarantined_note.append(refused)

    reactions: List[Dict[str, Any]] = []
    for number, row in enumerate(map(_dict, _list(processes.get("reactions"))), 1):
        inputs, outputs = _side(row, "inputs"), _side(row, "outputs")
        enzyme_names = [_text(_dict(e).get("entity") or _dict(e).get("name")) for e in _list(row.get("enzymes"))]
        states = sorted(admitted.get(_norm(row.get("name")), ()))
        lock_id = _text(row.get("locked_reaction_id"))
        reactions.append(
            {
                "number": number,
                "pointer": f"/processes/reactions/{number - 1}",
                "lock_id": lock_id,
                "name": _text(row.get("name")),
                "substrates": inputs,
                "products": outputs,
                "equation": _equation(inputs, outputs),
                "enzymes": [_enzyme(_dict(e), index) for e in _list(row.get("enzymes"))],
                "species": sorted({_text(index.get(_norm(n), {}).get("species")) for n in enzyme_names} - {""}),
                # Matched by name; ambiguous (two states for one name) is not guessed.
                "admission": states[0] if len(states) == 1 else "",
                # The same reaction listed under QUARANTINED / REJECTED: by lock id when
                # both carry one, else by name.
                "quarantine": [
                    f"listed as QUARANTINED {n}: state={q['state'] or NOT_RECORDED}, reason={q['reason'] or NOT_RECORDED}"
                    for n, q in enumerate(quarantined or [], 1)
                    if _same_process(lock_id, row.get("name"), q)
                ],
                "scope": _text(row.get("scope_membership")),
                "preservation": _text(row.get("preservation_status")),
                "lineage": _lineage(row),
                "mapping": [
                    _mapping(n, index.get(_norm(n), {}), _borrowed_for(n, index, borrowed))
                    for n in [*_participant_names(row), *enzyme_names]
                    if n
                ],
                "evidence": _quotes(row),
                "sources": _sources(row, paper_id),
                "duplicate_of": [],
                "_key": (tuple(sorted(map(_norm, inputs))), tuple(sorted(map(_norm, outputs)))),
            }
        )
    for reaction in reactions:
        if any(reaction["_key"]):
            reaction["duplicate_of"] = [
                other["number"] for other in reactions if other is not reaction and other["_key"] == reaction["_key"]
            ]
    for reaction in reactions:
        del reaction["_key"]

    removed_entities: Optional[List[str]] = None
    if isinstance(removed, dict):
        removed_entities = [
            f"{_show(e.get('bucket'))} {_show(e.get('name'))}: {_show(e.get('reason'))}"
            for e in map(_dict, _list(removed.get("removed_entities")))
        ]

    # A run directory says by its files; the app says by its export result.
    no_pwml = (
        ""
        if (any(name in artifacts for name in PWML_FILENAMES) if pwml_written is None else pwml_written)
        else (no_pwml_cause or f"neither {' nor '.join(PWML_FILENAMES)} is in this run")
    )
    interactions = _interactions(payload, artifacts.get(PWML_IR_FILENAME), ir_source, no_pwml)
    omissions = _omissions(payload, quarantine, removed, artifacts.get(PWML_IR_FILENAME), ir_source, interactions, no_pwml)

    return {
        "paper_id": paper_id,
        "pathway": _show(context.get("pathway_name")),
        "requested_organism": _show(context.get("likely_organism")),
        "species": [
            f"{_text(s.get('name'))} (taxonomy {_show(s.get('taxonomy_id'))})"
            for s in map(_dict, _list(_dict(payload.get("entities")).get("species")))
            if _text(s.get("name"))
        ],
        "coverage_ratio": _show(coverage.get("coverage_ratio")),
        "requested_core_terms": _texts(coverage.get("requested_core_terms")),
        "matched_terms": _texts(coverage.get("matched_terms")),
        "unmatched_terms": _texts(coverage.get("unmatched_terms")),
        "coverage_reasons": _texts(coverage.get("reasons")) or [],
        "surviving_processes": _show(coverage.get("surviving_processes")),
        "quarantined_processes": _show(coverage.get("quarantined_processes")),
        "minimum_core_satisfied": _show(coverage.get("minimum_core_satisfied")),
        "release_status": release_status,
        "export_mode": _show(_dict(quarantine).get("export_mode")),
        "quarantine_ok": _show(_dict(quarantine).get("ok")),
        "research_mode": (
            [f"Research mode applied: {_show(research.get('applied'))}", f"Research mode note: {_show(research.get('note'))}"]
            if isinstance(research, dict)
            else []
        ),
        "surviving_note": "; ".join(surviving_note),
        "quarantined_note": "; ".join(quarantined_note),
        "quarantine_sources": quarantine_sources,
        "payload_source": payload_source if payload else NOT_RECORDED,
        "transport_count": len(_list(processes.get("transports"))),
        "interaction_count": len(_list(processes.get("interactions"))),
        "artifacts": sorted(
            (payload_source if name == FINAL_MAPPED_FILENAME else name for name in artifacts), key=str.lower
        ),
        "reactions": reactions,
        # Repair round 1, B2: every borrowed identity, whatever the protein takes part in
        # (a reaction, an interaction, a transport, or nothing), not only reaction lines.
        "borrowed_identities": [
            f"{_text(row.get('name'))} ({_text(row.get('json_pointer'))}): {_text(row.get('flag'))}"
            f" -- record of {_show(row.get('source_organism'))} (taxon {_show(row.get('source_taxonomy_id'))}),"
            f" requested {_show(row.get('requested_organism'))} (taxon {_show(row.get('requested_taxonomy_id'))}),"
            f" route {_show(row.get('route'))}"
            for row in map(_dict, _list(payload.get(BORROWED_PROTEIN_IDENTITIES_KEY)))
        ],
        "interactions": interactions,
        "ir_source": ir_source if PWML_IR_FILENAME in artifacts else NOT_RECORDED,
        "quarantined": quarantined,
        "removed_entities": removed_entities,
        "pwml_coverage": omissions["coverage"],
        "omitted": omissions["omitted"],
        "omitted_entities": omissions["omitted_entities"],
        "omitted_actors": omissions["omitted_actors"],
    }


def _terms(label: str, terms: Optional[List[str]]) -> str:
    if terms is None:
        return f"{label}: {NOT_RECORDED}"
    return f"{label} ({len(terms)}): {', '.join(terms) or 'none'}"


def coverage_lines(report: Mapping[str, Any]) -> List[str]:
    lines = [
        _terms("Requested", report["requested_core_terms"]),
        _terms("Matched", report["matched_terms"]),
        _terms("Unmatched", report["unmatched_terms"]),
    ]
    if report["coverage_reasons"]:
        lines.append(f"Reasons: {'; '.join(report['coverage_reasons'])}")
    return lines


def header_lines(report: Mapping[str, Any]) -> List[str]:
    return [
        f"Paper: {report['paper_id']}",
        f"Pathway: {report['pathway']}",
        f"Requested organism (Stage 0): {report['requested_organism']}",
        f"Species (final payload): {', '.join(report['species']) or NOT_RECORDED}",
        f"Core-term coverage: {report['coverage_ratio']}",
        f"Minimum core satisfied: {report['minimum_core_satisfied']}",
        f"Surviving processes: {report['surviving_processes']}",
        f"Quarantined processes: {report['quarantined_processes']}",
        f"Release status: {report['release_status']}",
        f"Export mode: {report['export_mode']}",
        f"Quarantine ok: {report['quarantine_ok']}",
        *report["research_mode"],
        f"Reactions itemised: {len(report['reactions'])}; interactions itemised: {len(report['interactions'])} "
        f"(transports: {report['transport_count']}; counted above, not itemised)",
        f"Surviving reactions read from: {report['payload_source']}",
        f"Artifacts read: {', '.join(report['artifacts']) or 'none'}",
    ]


def _percent(drawn: Optional[int], total: int) -> str:
    return f"{100 * drawn / total:.1f}%" if drawn is not None and total else "n/a"


def _of(drawn: Optional[int], total: int) -> str:
    return f"{NOT_RECORDED if drawn is None else drawn} of {total}"


def pwml_coverage_lines(report: Mapping[str, Any]) -> List[str]:
    """D-114 § 3: the coverage headline, printed right under the title."""

    cov = report["pwml_coverage"]
    drawn, total = cov["drawn"], cov["total"]
    head = f"PWML coverage: {_of(drawn, total)} processes ({_percent(drawn, total)})"
    if cov["not_recorded"]:
        head = f"PWML coverage: {NOT_RECORDED} ({cov['not_recorded']}); {total} processes in the text record"
    kinds = "; ".join(f"{kind}s {_of(v['drawn'], v['total'])}" for kind, v in cov["by_kind"].items())
    lines = [
        f"{head} -- target {COVERAGE_TARGET_PERCENT}%",
        f"  By kind, drawn of total (total = final payload + omitted before the freeze): {kinds}",
    ]
    rest = total - cov["by_design_withheld"]
    lines.append(
        f"  Excluding the {cov['by_design_withheld']} interactions PathWhiz cannot type ({BY_DESIGN_KI}, withheld by design): "
        f"{_of(drawn, rest)} ({_percent(drawn, rest)})"
    )
    if not cov["pwml_written"]:
        lines.append(f"  No PWML written: {cov['no_pwml_cause']}")
    if cov["not_counted"]:
        lines.append(f"  Listed but not counted: {cov['not_counted']} (merged duplicates: {cov['merged_duplicates']})")
    lines.append("  Every process left out, with its reason: section OMITTED FROM THE PWML, below.")
    return lines


def _workaround(ki: str) -> str:
    return KNOWN_ISSUES["workarounds"].get(ki) or f"{NOT_RECORDED} (this reason has no catalogue entry)"


def _actor_hand_fix(a: Mapping[str, Any]) -> str:
    """ORCH-773: the process is in the PWML, so the fix is about the one actor, not the KI's whole-file workaround."""

    if a["decision"] == "D-114 AMENDMENT 1 s2":
        return (f"Add \"{a['actor']}\" as a transporter of \"{a['process']}\" in PathWhiz if the paper supports it; "
                f"the dropped row is kept verbatim in {REMOVED_ENTITY_REPORT_FILENAME} (detail.actor).")
    if a["decision"] == "D-113 s1":
        return ("None needed: the wrapper is inferred, not named by the paper (D-113 s1); "
                "check that the kept catalysts are drawn.")
    return _workaround(a["ki"])


def omitted_lines(report: Mapping[str, Any]) -> List[str]:
    """D-114 § 3: "Omitted from the PWML", every process the text record holds and the PWML does not draw."""

    counted = [e for e in report["omitted"] if e["counted"] and e["group"] in ("pre_freeze", "ir")]
    no_pwml = [e for e in report["omitted"] if e["counted"] and e["group"] == "no_pwml"]
    unknown = [e for e in report["omitted"] if e["counted"] and e["group"] == "not_recorded"]
    others = [e for e in report["omitted"] if not e["counted"]]
    actors = report.get("omitted_actors") or []
    kept = "still drawn" if report["pwml_coverage"]["pwml_written"] else "kept (no PWML written: nothing is drawn)"
    title = f"{len(counted) + len(no_pwml)}" + (f"; {len(actors)} actors left out of processes {kept}" if actors else "")
    lines = _section(f"OMITTED FROM THE PWML ({title})")
    lines.append(f"Known-issue ids: {KNOWN_ISSUES_DOC}, appendix \"Reason code -> KI id\".")
    if not report["pwml_coverage"]["pwml_written"]:
        cause = report["pwml_coverage"]["no_pwml_cause"]
        lines.append(f"No PWML written: {cause} (known issue: {known_issue_id(cause)}). "
                     f"Nothing is drawn; the {len(no_pwml)} processes of the final payload are listed after the removals.")
    lines.append("")
    for number, e in enumerate(counted, 1):
        lines += [
            f"OMITTED {number}  [{'; '.join(p for p in (e['source'], e['where']) if p)}]",
            f"Kind: {e['kind']}",
            f"Name: {e['name'] or NOT_RECORDED}",
            *([f"{_TEXT_LABEL.get(e['kind'], 'Equation')}: {e['text']}"] if e["text"] else []),
            f"Reason: {e['reason'] or NOT_RECORDED}",
            *([f"Verdict: {e['message']}"] if e["message"] else []),
            *([f"Omission: {e['omission']}"] if e["omission"] else []),
            f"Known issue: {e['ki']}",
            f"Add by hand: {_workaround(e['ki'])}",
            "",
        ]
    for number, a in enumerate(actors, 1):  # ORCH-773: the process is drawn and counted as drawn
        lines += [
            f"OMITTED ACTOR {number}  [{REMOVED_ENTITY_REPORT_FILENAME} removed_entities; process {a['where'] or NOT_RECORDED}]",
            f"Kind: actor left out of a {a['kind']} that is {kept}",
            f"Process: {a['process']}",
            f"Actor: {a['actor']}",
            f"Slot: {a['slots']}",
            f"Reason: {a['reason']}",
            *([f"Gate: {a['gate']}"] if a["gate"] else []),
            *([f"Decision: {a['decision']}"] if a["decision"] else []),
            *([f"Catalysts kept: {', '.join(a['kept']) or 'none'}"] if a["kept"] is not None else []),
            f"Known issue: {a['ki']}",
            f"Add by hand: {_actor_hand_fix(a)}",
            "",
        ]
    if no_pwml:
        lines.append(f"Not drawn because no PWML was written ({len(no_pwml)}):")
        lines += [f"- {e['kind']} {e['name'] or NOT_RECORDED}: {e['text'] or NOT_RECORDED}  [{e['where']}]" for e in no_pwml]
        lines.append("")
    if unknown:  # not called omitted: whether the PWML draws them is not recorded
        lines.append(f"Whether the PWML draws these is not recorded ({len(unknown)}):")
        lines += [f"- {e['kind']} {e['name'] or NOT_RECORDED}: {e['text'] or NOT_RECORDED}  [{e['where']}]: {e['message']}"
                  for e in unknown]
        lines.append("")
    if others:
        lines.append(f"Listed, not counted ({len(others)}):")
        for e in others:
            label = "merged duplicate, not counted" if e["merged_duplicate"] else f"{e['kind']}, not a counted kind"
            where = "; ".join(p for p in (e["source"], e["where"]) if p)
            lines.append(f"- {label}: {e['name'] or NOT_RECORDED}  [{where}]: {e['reason'] or NOT_RECORDED} ({e['ki']})")
        lines.append("")
    entities = report["omitted_entities"]
    lines.append(f"Entities left out ({_count(entities)}), from {REMOVED_ENTITY_REPORT_FILENAME}:")
    for e in entities or []:
        gate = f"; gate {e['gate']}" if e["gate"] else ""
        lines.append(f"- {e['bucket']} {e['name']}: {e['reason']} ({e['ki']}){gate}")
        lines += [f"    removed from {a}" for a in e["actor_removed_from"]]
    if not entities:
        lines.append(NOT_RECORDED if entities is None else NONE_RECORDED)
    return lines + [""]


def _evidence(quotes: List[str]) -> List[str]:
    return ["Evidence:", *([f'  {i}. "{q}"' for i, q in enumerate(quotes, 1)] or [f"  {NOT_RECORDED}"])]


def reaction_lines(reaction: Mapping[str, Any]) -> List[str]:
    # Pointers index the payload each artifact describes: survivors the final payload,
    # quarantined rows the pre-quarantine payload. The lock id is the cross-payload key.
    key = f"final payload {reaction['pointer']}" + (f"; lock {reaction['lock_id']}" if reaction["lock_id"] else "")
    status = [f"admission={reaction['admission']}"] if reaction["admission"] else []
    status += reaction["quarantine"] or ([] if status else [f"admission={NOT_RECORDED}"])
    status.append(f"scope={reaction['scope'] or NOT_RECORDED}")
    if reaction["preservation"]:
        status.append(f"preservation={reaction['preservation']}")
    lines = [
        f"REACTION {reaction['number']}  [{key}]",
        f"Name: {reaction['name'] or NOT_RECORDED}",
        f"Equation: {reaction['equation']}",
        f"Substrates: {', '.join(reaction['substrates']) or NOT_RECORDED}",
        f"Products: {', '.join(reaction['products']) or NOT_RECORDED}",
        f"Enzyme(s): {'; '.join(reaction['enzymes']) or NOT_RECORDED}",
        f"Species: {', '.join(reaction['species']) or NOT_RECORDED}",
        f"Status: {'; '.join(status)}",
        f"Lineage: {' | '.join(reaction['lineage']) or NOT_RECORDED}",
        "Mapping:",
        *[f"  - {m}" for m in reaction["mapping"] or [NOT_RECORDED]],
        *_evidence(reaction["evidence"]),
        "Source:",
        *[f"  - {s}" for s in reaction["sources"] or [NOT_RECORDED]],
    ]
    if reaction["duplicate_of"]:
        others = ", ".join(str(n) for n in reaction["duplicate_of"])
        lines.append(f"WARNING: {DUPLICATE_WARNING} (same substrates and products as reaction {others})")
    return lines


def interaction_lines(item: Mapping[str, Any]) -> List[str]:
    return [
        f"INTERACTION {item['number']}  [final payload {item['pointer']}]",
        f"Name: {item['name'] or NOT_RECORDED}",
        f"Relationship (as extracted): {item['relationship'] or NOT_RECORDED}",
        f"Source: {item['source'] or NOT_RECORDED}",
        f"Target: {item['target'] or NOT_RECORDED}",
        f"State: {item['state'] or NOT_RECORDED}",
        f"PWML: {item['pwml']}",
        *_evidence(item["evidence"]),
    ]


def borrowed_identities_title(report: Mapping[str, Any]) -> str:
    return f"BORROWED IDENTITIES ({len(report['borrowed_identities'])}) -- D-108: a UniProt record of another strain"


def interactions_title(report: Mapping[str, Any]) -> str:
    return f"INTERACTIONS ({len(report['interactions'])}) -- PWML verdicts read from: {report['ir_source']}"


def quarantined_lines(number: int, row: Mapping[str, Any]) -> List[str]:
    parts = (row["source"], row["pointer"], row["kind"], f"lock {row['lock_id']}" if row["lock_id"] else "")
    key = "; ".join(p for p in parts if p) or NOT_RECORDED
    lines = [f"QUARANTINED {number}  [{key}]", f"Name: {row['name'] or NOT_RECORDED}"]
    if row["equation"]:
        lines.append(f"Equation: {row['equation']}")
    return lines + [f"{label}: {value or NOT_RECORDED}" for label, value in row["fields"]] + _evidence(row["evidence"])


def with_note(title: str, note: str) -> str:
    return f"{title} -- {note}" if note else title


def surviving_title(report: Mapping[str, Any]) -> str:
    return with_note(f"SURVIVING REACTIONS ({len(report['reactions'])})", report["surviving_note"])


def quarantined_title(report: Mapping[str, Any]) -> str:
    return with_note(f"QUARANTINED / REJECTED ({_count(report['quarantined'])})", report["quarantined_note"])


def _section(title: str) -> List[str]:
    return [title, "-" * len(title)]


def _count(rows: Optional[List[Any]]) -> str:
    return NOT_RECORDED if rows is None else str(len(rows))


def _absent(rows: Optional[List[Any]]) -> List[str]:
    return [NOT_RECORDED if rows is None else NONE_RECORDED, ""]


def render_text(report: Mapping[str, Any]) -> str:
    """The .txt export: a pure function of ``report``, so same artifacts give the same bytes."""

    lines = ["PATHWAY EVIDENCE REPORT", "=======================", *pwml_coverage_lines(report), *header_lines(report), ""]
    lines += _section(surviving_title(report))
    for reaction in report["reactions"]:
        lines += [*reaction_lines(reaction), ""]
    if not report["reactions"]:
        lines += _absent(None if report["payload_source"] == NOT_RECORDED else [])
    lines += _section(interactions_title(report))
    for item in report["interactions"]:
        lines += [*interaction_lines(item), ""]
    if not report["interactions"]:
        lines += _absent(None if report["payload_source"] == NOT_RECORDED else [])
    if report["borrowed_identities"]:  # absent section when nothing is borrowed: other reports keep their bytes
        lines += [*_section(borrowed_identities_title(report)), *(f"- {line}" for line in report["borrowed_identities"]), ""]
    quarantined = report["quarantined"]
    lines += [*_section(quarantined_title(report)), *report["quarantine_sources"], ""]
    for number, row in enumerate(quarantined or [], 1):
        lines += [*quarantined_lines(number, row), ""]
    if not quarantined:
        lines += _absent(quarantined)
    removed = report["removed_entities"]
    lines += _section(f"REMOVED ENTITIES ({_count(removed)})")
    lines += [*(f"- {e}" for e in removed), ""] if removed else _absent(removed)
    lines += omitted_lines(report)
    lines += [*_section("CORE COVERAGE"), *coverage_lines(report)]
    return "\n".join(lines) + "\n"
