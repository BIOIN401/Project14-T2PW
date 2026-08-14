"""The Stage-1 → Stage-2 boundary: settle it, or say precisely why you cannot.

Before this module the boundary was three lines in the app::

    stage_one, chunk_details = run_stage_one_with_chunking(...)
    validate_post_extraction(stage_one)
    # except StageContractError: st.error(...); st.stop()

That has two problems, and they are the two this module fixes.

FIRST: a recoverable failure was treated as terminal. A payload can fail the
post-extraction contract for reasons that are *local and mechanical* -- one
reaction row whose participants arrived in the wrong shape, one entity row that
lost its name -- and the response was to stop the whole run. Nothing tried to fix
the one row. :func:`settle_stage_one` does, through
:func:`localized_repair.repair_invalid_rows`, and it re-validates rather than
assuming the repair worked.

SECOND: a *silent* loss was never addressed at all. A model that names a
participant in a reaction but never declares it in the entity registry produces a
payload that passes this contract and then loses that reaction later, quietly, in
``filter_unresolvable_reactions``. The reaction was correct and evidenced; the
registry was short. :func:`localized_repair.reconstruct_registry_shells` runs
here, unconditionally, because the point is to prevent the loss rather than to
react to a symptom that never surfaces as an error.

WHAT THIS MODULE WILL NOT DO
----------------------------
It will not manufacture a pass. When repair and reconstruction are both spent and
the contract still refuses the payload, the result is
:data:`~t2pw.pipeline.localized_repair.REPAIR_INCOMPLETE`: the payload comes back
as far as it got, ``ok`` is ``False``, ``incomplete_reason`` says what could not
be recovered, the diagnostics are already on disk, and the caller stops. Nothing
here invents a reaction to satisfy ``processes_required`` or an entity to satisfy
``entities_required`` -- an empty pathway reported as empty is a usable result,
and a fabricated one is not.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from t2pw.pipeline import lineage
from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE, ExportMode
from t2pw.pipeline.extraction_diagnostics import (
    BOUNDARY_STAGE1_EXTRACTION,
    OUTCOME_CONTRACT_FAILED,
    OUTCOME_DISCARDED_BY_CLEANING,
    OUTCOME_OK,
    OUTCOME_ZERO_PROCESSES,
    count_entities,
    count_processes,
    current as current_diagnostics,
    payload_hash,
)
from t2pw.pipeline.localized_repair import (
    REPAIR_INCOMPLETE,
    REPAIR_NOT_ATTEMPTED,
    REPAIR_OK,
    SHELL_PROVENANCE,
    RowRepairResult,
    reconstruct_registry_shells,
    repair_invalid_rows,
    resolve_pointer,
)
from t2pw.pipeline.stage_contracts import (
    StageContractError,
    run_stage_contract,
    validate_post_extraction,
)

logger = logging.getLogger(__name__)

__all__ = ["BoundaryOutcome", "settle_stage_one"]


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


@dataclass
class BoundaryOutcome:
    """What the boundary made of the payload, and what it could not.

    ``payload`` is always usable by the caller: on the incomplete path it is the
    best state reached, not ``None``, so the run can persist it alongside the
    failure artifacts instead of the operator being left with a message and an
    empty directory.
    """

    payload: Dict[str, Any]
    ok: bool
    outcome: str
    contract_report: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Dict[str, Any] = field(default_factory=dict)
    repair: Optional[RowRepairResult] = None
    incomplete_reason: str = ""
    failure: Optional[StageContractError] = None

    def to_summary(self) -> Dict[str, Any]:
        """A bounded, JSON-safe description for artifacts and session state."""

        summary: Dict[str, Any] = {
            "ok": self.ok,
            "outcome": self.outcome,
            "payload_hash": payload_hash(self.payload),
            "entity_counts": count_entities(self.payload),
            "process_counts": count_processes(self.payload),
            "contract_summary": _safe_dict(self.contract_report.get("summary")),
        }
        if self.incomplete_reason:
            summary["incomplete_reason"] = self.incomplete_reason
        if self.reconstruction:
            summary["registry_reconstruction"] = self.reconstruction
        if self.repair is not None:
            summary["row_repair"] = {
                "outcome": self.repair.outcome,
                "attempts": self.repair.attempts,
                "repaired_pointers": self.repair.repaired_pointers,
                "unrepaired_pointers": self.repair.unrepaired_pointers,
                "rejected": self.repair.rejected,
                "reason": self.repair.reason,
            }
        return summary


def _validate(payload: Dict[str, Any], mode: ExportMode) -> Dict[str, Any]:
    """Run the post-extraction contract, returning its report either way.

    The report is what repair needs -- it carries the per-row pointers -- so a
    raised ``StageContractError`` is unwrapped rather than propagated. The caller
    decides what a non-ok report means; this function only reports.
    """

    try:
        return run_stage_contract(validate_post_extraction, payload, mode=mode)
    except StageContractError as exc:
        return dict(exc.report)


def _contract_errors(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in _safe_list(report.get("errors")) if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Lineage. This boundary is the only place that sees all three Stage-1 origins
# at once -- what the model drew, what deterministic reconstruction added, and
# what localized repair rewrote -- so it is where ``PRODUCT_CONTRACT`` § 3's
# "attribute false content empirically to Stage 1" becomes recordable.
#
# Every write happens AFTER the exit's ``record_boundary`` call and touches no
# key but ``lineage.LINEAGE_KEY``: the boundary record fingerprints the payload
# the boundary JUDGED, and lineage played no part in that judgement. Writing it
# first would move a diagnostic to describe a decision it did not influence.
# ---------------------------------------------------------------------------

#: The two payload sections holding attributable rows.
_ROW_SECTIONS: Tuple[str, ...] = ("entities", "processes")

#: Rows the caller handed us. No ``sources``: this seam receives a payload, not
#: the paper it was drawn from, and has no PMCID/DOI/filename to name -- so
#: ``support`` is ``unsupported``. ``direct`` would assert a named source that
#: does not exist here, and minting one is exactly what ``SOURCED_ORIGINS``
#: exists to forbid. The fact we DO have, that the paper stated it, rides
#: ``paper_explicit``.
_PAPER_STATED = lineage.LineageEntry(
    stage="paper_extraction",
    origin="paper_stated",
    support="unsupported",
    paper_explicit="explicit",
    reason="present in the Stage-1 extraction payload when it reached the "
           "Stage-1 to Stage-2 boundary",
    review_required=False,
)

#: The same row and the same origin when the row's OWN markers say the model did
#: not read it off the page. ``origin`` STAYS ``paper_stated``: which Stage-1
#: bucket a row came out of is not a claim about how the model arrived at it, and
#: re-bucketing it is a product decision this card does not hold. What changes is
#: the part that was false. ``paper_explicit`` drops to ``not_evaluated`` --
#: never ``not_explicit``, because this stage did not read the paper either -- and
#: the row is flagged for review, since a complex carrying ``confidence: 0.6``
#: and ``provenance: "inferred"`` is precisely what § 3 exists to make findable.
_PAPER_STATED_UNVERIFIED = lineage.LineageEntry(
    stage="paper_extraction",
    origin="paper_stated",
    support="unsupported",
    paper_explicit="not_evaluated",
    reason="present in the Stage-1 extraction payload when it reached the "
           "Stage-1 to Stage-2 boundary, marked by the extraction as not read "
           "verbatim from the paper",
    review_required=True,
    uncertainty="the row's own provenance/inference marker says this content was "
                "reasoned or looked up rather than stated; nothing here "
                "established what the paper does say about it",
)

#: ``pwml_system.txt:116`` -- ``provenance`` values are ``"extracted"`` (verbatim
#: from text), ``"enriched"`` (from an API lookup) and ``"inferred"``
#: (LLM-reasoned). The last two are the model telling us it did not read the row
#: off the page. Matched on those SPECIFIC values and never as
#: ``provenance != "extracted"``: :data:`SHELL_PROVENANCE` is a ``provenance``
#: value too, and a blanket negation would additionally sweep in every row that
#: simply carries no marker at all.
_PROVENANCE_NOT_READ: Tuple[str, ...] = ("inferred", "enriched")

#: Row keys whose mere presence says the same thing. ``inference`` is the prompt's
#: free-text note naming what was reasoned (``pwml_system.txt:109``, ``:401``);
#: ``rag_provenance`` is C-038's carrier naming retrieved literature.
_MARKS_NOT_READ: Tuple[str, ...] = ("inference", "rag_provenance")

#: Rows ``reconstruct_registry_shells`` added. ``derived``, not ``direct``: a
#: process row the payload already carried named this participant, so the
#: registry row is bookkeeping over stated content -- but nothing here states
#: what the entity IS, which is why the shell is reviewable rather than clean.
#: This stage read a name off a process row; whether the paper declared that
#: participant as an entity is a question it never asked, hence
#: ``not_evaluated`` -- which is never ``not_explicit``.
_RECONSTRUCTED = lineage.LineageEntry(
    stage="normalization",
    origin="inferred",
    support="derived",
    paper_explicit="not_evaluated",
    reason="registry shell reconstructed for a participant a process row named "
           "but the entity registry did not carry",
    review_required=True,
    uncertainty="the shell carries no identifier; the entity stays unresolved "
                "until something grounds it",
)

#: Rows ``repair_invalid_rows`` rewrote. ``unsupported``, not ``derived``: the
#: replacement came from a model. ``preserves_original_values`` and
#: ``evidence_supports`` CONSTRAIN that rewrite; they do not derive it.
_REPAIRED = lineage.LineageEntry(
    stage="audit_repair",
    origin="audit_modified",
    support="unsupported",
    paper_explicit="not_evaluated",
    reason="row rewritten by localized repair after the post-extraction "
           "contract rejected it",
    review_required=True,
    uncertainty="the replacement row was model-produced; the contract accepts "
                "it but nothing named a source for the change",
)


def _row_census(payload: Dict[str, Any]) -> Dict[Tuple[str, str], int]:
    """``(section, bucket) -> row count``. A pure read; it mutates nothing.

    Identity by INDEX is exact here where a name match would not be.
    ``reconstruct_registry_shells`` only ever appends to an entity bucket and
    ``repair_invalid_rows`` only ever overwrites a pointer that already
    resolves (``localized_repair._assign_pointer`` refuses to create one), so
    between two censuses no row is removed and none is reordered: an index
    below the entry census is the row the caller supplied, and an index at or
    above it is a row this stage added. The shell records themselves cannot
    serve -- ``ReconstructionResult.shells`` clips names at 120 characters and
    ``to_report`` truncates the list.
    """

    census: Dict[Tuple[str, str], int] = {}
    for section in _ROW_SECTIONS:
        for bucket, rows in _safe_dict(payload.get(section)).items():
            if isinstance(rows, list):
                census[(section, str(bucket))] = len(rows)
    return census


def _row_at(
    payload: Dict[str, Any], section: str, bucket: str, index: int
) -> Optional[Dict[str, Any]]:
    """The row at that position, or ``None`` when it is not one we can annotate.

    A bucket may hold bare strings; attaching lineage to one would mean turning
    it into an object, which is a change to the payload's content and not this
    card's to make. Such a row is skipped, not converted.
    """

    rows = _safe_dict(payload.get(section)).get(bucket)
    if not isinstance(rows, list) or not 0 <= index < len(rows):
        return None
    row = rows[index]
    return row if isinstance(row, dict) else None


def _paper_entry(row: Dict[str, Any]) -> lineage.LineageEntry:
    """Which paper-extraction entry this row has earned.

    The extraction prompt does not only return what it read: it instructs the
    model to CREATE content it could not read and to mark it -- a complex whose
    subunit membership is unknown gets ``confidence < 1.0`` and
    ``provenance: "inferred"`` (``pwml_system.txt:109``, ``:116``, ``:401``) --
    and those markers survive into this seam intact, because
    ``pipeline._clean_entities`` copies entity rows key-for-key and the process
    cleaners carry ``provenance``/``inference`` through. Reporting such a row as
    ``paper_explicit="explicit"`` with no review flag writes the opposite of
    what the row says about itself, in the one clause this card exists to serve.

    ``rag_provenance`` is treated the same way and cannot arrive here today --
    this boundary's only call site runs before any RAG -- so it costs nothing
    now and is not a claim that has to be un-made when one does.
    """

    if str(row.get("provenance") or "").strip().casefold() in _PROVENANCE_NOT_READ:
        return _PAPER_STATED_UNVERIFIED
    if any(row.get(key) for key in _MARKS_NOT_READ):
        return _PAPER_STATED_UNVERIFIED
    return _PAPER_STATED


def _record_once(
    row: Dict[str, Any],
    entry: lineage.LineageEntry,
    *,
    unless: Optional[lineage.LineageEntry] = None,
) -> None:
    """Append ``entry`` to ``row``'s lineage unless it is already there.

    Three properties, all of which have to live in the writer:

    * **Idempotent.** ``Lineage`` deliberately KEEPS duplicates -- dedup
      removes, and removal is what its append-only rule forbids. So a payload
      settled twice (a resumed run, or a caller feeding an outcome back in)
      would otherwise stack two identical "the paper stated this" facts. Every
      entry this module writes is a module-level constant carrying no
      timestamp, counter or pointer, so the same fact rebuilds to an EQUAL
      entry and this test recognizes it. That is C-015's content-derived
      identity, not a private key.
    * **Additive.** ``lineage.record`` re-emits everything already stored, so a
      row arriving with another stage's attribution keeps it.
    * **Non-fatal.** A row carrying malformed stored lineage makes
      ``lineage.read`` raise. Letting that escape would turn a boundary that
      returns a usable payload into one that aborts the run -- a decision
      change this card is not entitled to make -- so it is reported against the
      row and attribution for that row is skipped.

    ``unless`` suppresses the write when a different entry is present. It has
    exactly one use: a shell THIS stage reconstructed on an earlier settle is,
    on a later settle, a row "present in the input" -- and calling it
    paper-stated would be false. The shell's own entry is the proof it was not.
    """

    try:
        present = lineage.read(row).entries
        if unless is not None and unless in present:
            return
        if entry not in present:
            lineage.record(row, entry)
    except lineage.LineageError as exc:
        logger.warning(
            "stage_one_boundary: leaving a row unattributed, its stored %s is "
            "malformed: %s", lineage.LINEAGE_KEY, exc,
        )


def _attribute(
    working: Dict[str, Any],
    inbound: Dict[Tuple[str, str], int],
    reconstructed: Dict[Tuple[str, str], int],
    repair: Optional[RowRepairResult],
) -> None:
    """Write this stage's three attributions onto ``working``'s rows.

    ``working`` is safe to write into and is the only object written: it is the
    ``deepcopy`` :func:`settle_stage_one` takes of its argument, or a further
    deepcopy of that made inside ``reconstruct_registry_shells`` /
    ``repair_invalid_rows``. No row reachable from it is reachable from the
    caller's ``payload``, so no lineage write can land on an object the caller
    still holds.

    A repaired row keeps whichever origin entry it already had: repair records
    what repair did, it does not restate where the row came from.
    """

    for (section, bucket), count in inbound.items():
        for index in range(count):
            row = _row_at(working, section, bucket, index)
            if row is None or row.get("provenance") == SHELL_PROVENANCE:
                # The index partition is exact against localized_repair AS IT IS,
                # but that module is not this card's to pin: were reconstruction
                # ever to dedupe, sort or drop an entity row, a shell would land
                # below the inbound count and be called paper-stated with no test
                # failing anywhere. The shell's own marker is a content-derived
                # second condition that does not depend on that module's shape,
                # and unlike the lineage check below it holds on the FIRST settle
                # and on a rebuild that dropped the lineage key.
                continue
            _record_once(row, _paper_entry(row), unless=_RECONSTRUCTED)

    for (section, bucket), count in reconstructed.items():
        if section != "entities":
            continue
        for index in range(inbound.get((section, bucket), 0), count):
            row = _row_at(working, section, bucket, index)
            if row is not None:
                _record_once(row, _RECONSTRUCTED)

    for pointer in (repair.repaired_pointers if repair is not None else []):
        row = resolve_pointer(working, pointer)
        if isinstance(row, dict):
            _record_once(row, _REPAIRED)


def settle_stage_one(
    payload: Dict[str, Any],
    *,
    mode: ExportMode = DEFAULT_EXPORT_MODE,
    cleaning_report: Optional[Dict[str, Any]] = None,
    reconstruct: bool = True,
    repair_rows: bool = True,
    chat_fn: Optional[Callable[..., Any]] = None,
) -> BoundaryOutcome:
    """Get the Stage-1 payload through its contract, or report what blocked it.

    The sequence, in order, and it does not vary:

    1. **Deterministic registry reconstruction** (no model). Runs first and
       unconditionally: it is free, it cannot fail, and a shell it adds can turn a
       row the contract was about to reject into a valid one -- so doing it before
       validation saves a model call rather than following one.
    2. **Validate.** If the contract is satisfied, done.
    3. **Localized row repair** (bounded, model). Only the rows named by contract
       errors are sent, with their exact errors and their own evidence; valid rows
       are neither sent nor touched. Re-validated afterwards, so "the repair
       worked" is measured, not assumed.
    4. **Report incomplete.** Anything still failing is named, the payload is
       returned as-is, and ``ok`` is ``False``.

    ``reconstruct``/``repair_rows``/``chat_fn`` are seams for tests and for a
    caller that wants the boundary judged without either recovery running.
    """

    working = deepcopy(payload) if isinstance(payload, dict) else {}
    inbound_rows = _row_census(working)  # lineage bookkeeping only; reads, never writes
    diagnostics = current_diagnostics()
    reconstruction_report: Dict[str, Any] = {}

    if reconstruct:
        rebuilt = reconstruct_registry_shells(working)
        working = rebuilt.payload
        if rebuilt.changed:
            reconstruction_report = rebuilt.to_report()
    reconstructed_rows = _row_census(working)

    report = _validate(working, mode)
    errors = _contract_errors(report)

    if not errors:
        outcome = _passing_outcome(working, cleaning_report)
        diagnostics.record_boundary(
            stage="extraction",
            boundary=BOUNDARY_STAGE1_EXTRACTION,
            outcome=outcome,
            contract_ok=True,
            raw_entity_counts=count_entities(working),
            raw_process_counts=count_processes(working),
            response_hash=payload_hash(working),
            note="stage_one_boundary_settled",
            registry_shells_added=len(_safe_list(reconstruction_report.get("shells"))) or None,
        )
        _attribute(working, inbound_rows, reconstructed_rows, None)
        return BoundaryOutcome(
            payload=working,
            ok=True,
            outcome=outcome,
            contract_report=report,
            reconstruction=reconstruction_report,
        )

    repair: Optional[RowRepairResult] = None
    if repair_rows:
        repair = repair_invalid_rows(
            working,
            errors,
            revalidate=lambda candidate: _contract_errors(_validate(candidate, mode)),
            stage="extraction",
            chat_fn=chat_fn,
        )
        if repair.changed:
            working = repair.payload
            report = _validate(working, mode)
            errors = _contract_errors(report)

    if not errors:
        outcome = _passing_outcome(working, cleaning_report)
        diagnostics.record_boundary(
            stage="extraction",
            boundary=BOUNDARY_STAGE1_EXTRACTION,
            outcome=outcome,
            contract_ok=True,
            raw_entity_counts=count_entities(working),
            raw_process_counts=count_processes(working),
            response_hash=payload_hash(working),
            note="stage_one_boundary_settled_after_localized_repair",
            repaired_pointers=repair.repaired_pointers if repair else None,
        )
        _attribute(working, inbound_rows, reconstructed_rows, repair)
        return BoundaryOutcome(
            payload=working,
            ok=True,
            outcome=outcome,
            contract_report=report,
            reconstruction=reconstruction_report,
            repair=repair,
        )

    reason = _incomplete_reason(errors, repair, cleaning_report)
    diagnostics.record_boundary(
        stage="extraction",
        boundary=BOUNDARY_STAGE1_EXTRACTION,
        outcome=OUTCOME_CONTRACT_FAILED,
        contract_ok=False,
        raw_entity_counts=count_entities(working),
        raw_process_counts=count_processes(working),
        response_hash=payload_hash(working),
        error=reason,
        note="stage_one_boundary_incomplete",
        failing_codes=sorted({str(item.get("code") or "") for item in errors if item.get("code")}),
        repair_outcome=(repair.outcome if repair else REPAIR_NOT_ATTEMPTED),
    )
    logger.warning(
        "Stage-1 boundary could not be settled: %s (%d contract error(s) remain).",
        reason,
        len(errors),
    )
    _attribute(working, inbound_rows, reconstructed_rows, repair)
    return BoundaryOutcome(
        payload=working,
        ok=False,
        outcome=OUTCOME_CONTRACT_FAILED,
        contract_report=report,
        reconstruction=reconstruction_report,
        repair=repair,
        incomplete_reason=reason,
        failure=StageContractError(
            str(report.get("stage") or "post_extraction"),
            reason,
            report,
        ),
    )


def _passing_outcome(
    payload: Dict[str, Any],
    cleaning_report: Optional[Dict[str, Any]],
) -> str:
    """The outcome for a payload the contract accepted.

    "Accepted" is not "complete": a payload with no processes satisfies the
    structural guard (``processes`` is an object; it is merely empty) and is still
    a failure of extraction. Which failure it is depends on what cleaning saw --
    rows in and none out is a cleaning-rule problem, nothing in at all is a
    prompt/scope problem -- so the cleaning report decides between the two rather
    than both being reported as "ok".
    """

    if count_processes(payload).get("total", 0) > 0:
        return OUTCOME_OK
    if _safe_dict(cleaning_report).get("all_processes_discarded"):
        return OUTCOME_DISCARDED_BY_CLEANING
    return OUTCOME_ZERO_PROCESSES


def _incomplete_reason(
    errors: List[Dict[str, Any]],
    repair: Optional[RowRepairResult],
    cleaning_report: Optional[Dict[str, Any]],
) -> str:
    """One sentence naming what blocked the boundary and what was already tried.

    Written for the person reading ``RESULT.txt`` at 09:00, so it states the
    count, the first code, whether repair ran, and -- when cleaning is the real
    cause -- that the rows existed before cleaning removed them. That last clause
    is the difference between "the model gave us nothing" and "we deleted what it
    gave us", which no message in this pipeline used to draw.
    """

    codes = [str(item.get("code") or "") for item in errors if item.get("code")]
    head = codes[0] if codes else "unknown_code"
    parts = [
        f"{len(errors)} post-extraction contract error(s) remain (first: {head})",
    ]
    if repair is None:
        parts.append("localized row repair was not attempted")
    elif repair.outcome == REPAIR_NOT_ATTEMPTED:
        parts.append(f"localized row repair was not attempted: {repair.reason}")
    elif repair.outcome == REPAIR_INCOMPLETE:
        parts.append(
            f"localized row repair ran {repair.attempts} attempt(s) and "
            f"recovered {len(repair.repaired_pointers)} row(s): {repair.reason}"
        )
    elif repair.outcome == REPAIR_OK:
        parts.append(
            f"localized row repair recovered {len(repair.repaired_pointers)} row(s) "
            "but the contract still refuses the payload"
        )

    cleaning = _safe_dict(cleaning_report)
    if cleaning.get("all_processes_discarded"):
        raw_total = _safe_dict(cleaning.get("raw_process_counts")).get("total", 0)
        parts.append(
            f"cleaning discarded every one of the {raw_total} process row(s) the model "
            f"returned ({cleaning.get('discarded_by_reason')})"
        )
    return "; ".join(parts)
