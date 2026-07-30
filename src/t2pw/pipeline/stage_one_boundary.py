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
from typing import Any, Callable, Dict, List, Optional

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
    RowRepairResult,
    reconstruct_registry_shells,
    repair_invalid_rows,
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
    diagnostics = current_diagnostics()
    reconstruction_report: Dict[str, Any] = {}

    if reconstruct:
        rebuilt = reconstruct_registry_shells(working)
        working = rebuilt.payload
        if rebuilt.changed:
            reconstruction_report = rebuilt.to_report()

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
