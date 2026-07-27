"""Export-mode policy: PathWhiz-strict (default) vs research (relaxed).

The pipeline has two audiences. The default one is the PathWhiz Rails importer,
which refuses a file over rules that say nothing about biology: a missing UniProt
accession, an enzyme that is not wrapped in a synthetic ``protein_complex``, a
``+`` inside a compound name. The second audience is a human reviewing a *novel*
multi-paper pathway, for whom those rules are noise and an abort is a dead end --
a genuinely new enzyme from a 2026 paper legitimately has no accession yet.

``research`` mode serves the second audience:

* **FORMAT** findings (:data:`FORMAT_ISSUE_CODES`) are skipped -- recorded, but
  they do not block.
* **BIOLOGY / PROVENANCE** findings still run and are collected as *review
  flags*: loud, per-item, and non-blocking.
* **Structural guards** (:data:`STRUCTURAL_GUARD_CODES`) still abort in both
  modes. They are not PathWhiz rules; they are crash guards. Relaxing
  ``invalid_payload`` does not produce a candidate pathway, it produces an
  ``AttributeError`` several frames down.

Two design rules make this auditable:

1. **Nothing is deleted, only re-severitied.** A skipped FORMAT rule still runs
   and still emits its exact issue code and JSON pointer; it lands in
   ``warnings`` instead of ``errors``. The strict/research diff is therefore
   readable off the report itself.
2. **Unknown codes flag rather than skip.** :func:`classify_issue` returns
   ``"review"`` for any code it does not recognise, so a check added later --
   or one of the duplicate codes ``payload_models.py`` emits under the same
   names as ``stage_contracts.py`` -- fails safe toward human review rather
   than being silently dropped.

Fail-open must never mean fail-silent: :func:`relax_report` always records
counts and stamps every downgraded issue, so a research run can say exactly
what it ignored and what it wants looked at.

This module is core-side and deliberately imports nothing from the RAG package,
per ``docs/rag/03_separation_invariant.md``. RAG may import this; the reverse is
forbidden, and the invariant's verification grep must keep returning nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping

__all__ = [
    "ExportMode",
    "PATHWHIZ",
    "RESEARCH",
    "DEFAULT_EXPORT_MODE",
    "FORMAT_ISSUE_CODES",
    "STRUCTURAL_GUARD_CODES",
    "coerce_mode",
    "is_research",
    "classify_issue",
    "relax_report",
    "review_flags",
    "format_gaps",
]


ExportMode = Literal["pathwhiz", "research"]

PATHWHIZ: ExportMode = "pathwhiz"
RESEARCH: ExportMode = "research"

#: Today's behaviour. Every public entry point defaults to this, so an
#: un-updated caller is byte-for-byte unchanged.
DEFAULT_EXPORT_MODE: ExportMode = PATHWHIZ


#: Findings that exist only so the PathWhiz importer accepts the file.
#: Skipped in research mode. Each entry is justified in docs/research_mode.md.
FORMAT_ISSUE_CODES: frozenset[str] = frozenset(
    {
        # The generated protein-complex wrapper. It exists solely because the
        # importer refuses a bare protein as a reaction enzyme; research mode
        # drops the wrapping entirely, so every rule about it is moot.
        "generated_wrapper_missing_components",
        "generated_wrapper_component_protein_unresolved",
        "generated_wrapper_component_missing_species",
        "generated_wrapper_component_missing_external_identity",
        "generated_wrapper_flag_required",
        "generated_wrapper_reason_invalid",
        "generated_wrapper_component_not_structured",
        "generated_wrapper_component_name_required",
        # A type-only assertion on optional mapping bookkeeping. No extracted
        # content is lost when the field is present but not a string.
        "mapping_resolution_field_invalid",
    }
)


#: Container/row type guards. These abort in BOTH modes: they are not importer
#: rules, and skipping one converts a clean StageContractError into an
#: AttributeError inside an unrelated stage.
STRUCTURAL_GUARD_CODES: frozenset[str] = frozenset(
    {
        "invalid_payload",
        "entities_required",
        "processes_required",
        "entity_not_object",
        "process_not_object",
    }
)


def coerce_mode(value: Any) -> ExportMode:
    """Normalize anything user- or session-supplied to a valid mode.

    Unrecognised input resolves to :data:`PATHWHIZ`, so a typo degrades to the
    safe, strict default rather than silently relaxing the gates.
    """

    text = str(value or "").strip().lower()
    if text in {RESEARCH, "relaxed", "research (relaxed)"}:
        return RESEARCH
    return PATHWHIZ


def is_research(mode: Any) -> bool:
    """True when relaxed research behaviour is requested."""

    return coerce_mode(mode) == RESEARCH


def classify_issue(issue: Mapping[str, Any]) -> str:
    """Return ``"guard"``, ``"format"`` or ``"review"`` for one report issue.

    Unrecognised codes deliberately classify as ``"review"`` -- the categorization
    errs toward keeping the check and flagging it for a human.
    """

    code = str(_get(issue, "code") or "").strip()
    if code in STRUCTURAL_GUARD_CODES:
        return "guard"
    if code in FORMAT_ISSUE_CODES:
        return "format"
    return "review"


def relax_report(report: Mapping[str, Any], *, stage: str = "") -> Dict[str, Any]:
    """Return a research-mode copy of a stage report, or raise for hard guards.

    FORMAT findings move to ``warnings`` stamped ``research_mode="skipped_format"``.
    Everything else moves to ``warnings`` stamped ``research_mode="review_flag"``
    so it stays visible without blocking. ``ok`` becomes ``True`` because the run
    must continue -- ``research_review_required`` and ``research_summary`` carry
    the "this needs eyes" signal instead, and the caller is expected to render
    them prominently.

    Structural guard findings are left in ``errors`` and reported back via
    ``research_blocked``; the caller re-raises, because there is no candidate
    pathway to show when the payload is not a payload.
    """

    relaxed: Dict[str, Any] = {
        key: value
        for key, value in dict(report).items()
        if key not in {"errors", "warnings", "ok", "summary"}
    }
    relaxed["stage"] = str(report.get("stage") or stage or "unknown_stage")

    errors = [item for item in _as_list(report.get("errors")) if isinstance(item, dict)]
    warnings = [item for item in _as_list(report.get("warnings")) if isinstance(item, dict)]

    blocking: List[Dict[str, Any]] = []
    downgraded: List[Dict[str, Any]] = []
    for issue in errors:
        kind = classify_issue(issue)
        if kind == "guard":
            blocking.append(dict(issue))
            continue
        stamped = dict(issue)
        stamped["research_mode"] = "skipped_format" if kind == "format" else "review_flag"
        stamped["research_category"] = "FORMAT" if kind == "format" else "BIOLOGY/PROVENANCE"
        stamped["was_blocking"] = True
        stamped.setdefault("stage", relaxed["stage"])
        downgraded.append(stamped)

    relaxed["errors"] = blocking
    # Pre-existing warnings are preserved untouched; they were never blocking.
    relaxed["warnings"] = [*warnings, *downgraded]
    relaxed["ok"] = not blocking
    relaxed["export_mode"] = RESEARCH
    relaxed["effect_on_failure"] = "annotate_only" if not blocking else "abort"

    skipped = [i for i in downgraded if i.get("research_mode") == "skipped_format"]
    flagged = [i for i in downgraded if i.get("research_mode") == "review_flag"]
    relaxed["research_blocked"] = blocking
    relaxed["research_review_required"] = bool(flagged)
    relaxed["research_summary"] = {
        "format_skipped": len(skipped),
        "review_flagged": len(flagged),
        "still_blocking": len(blocking),
    }
    relaxed["summary"] = {
        "error_count": len(blocking),
        "warning_count": len(relaxed["warnings"]),
    }
    return relaxed


def review_flags(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every BIOLOGY/PROVENANCE finding a research run downgraded, for the panel."""

    return [
        dict(issue)
        for issue in _as_list(report.get("warnings"))
        if isinstance(issue, dict) and issue.get("research_mode") == "review_flag"
    ]


def format_gaps(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every FORMAT rule a research run skipped, so the skip is never invisible."""

    return [
        dict(issue)
        for issue in _as_list(report.get("warnings"))
        if isinstance(issue, dict) and issue.get("research_mode") == "skipped_format"
    ]


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _get(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping.get(key)
    except AttributeError:
        return None
