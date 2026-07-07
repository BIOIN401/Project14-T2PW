from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


Issue = Dict[str, Any]
Report = Dict[str, Any]


class StageContractError(ValueError):
    """Raised when a stage boundary contract must abort the pipeline."""

    def __init__(self, stage: str, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.stage = stage
        self.report = dict(report)
        self.errors = list(self.report.get("errors", []))


def validate_post_extraction(payload: Any) -> Report:
    """Validate the structural contract emitted by extraction."""

    report = _new_report("post_extraction", "structural", effect_on_failure="abort")
    _validate_payload_container(payload, report)
    if report["errors"]:
        _abort(report)

    assert isinstance(payload, dict)
    _validate_named_entities(payload, report)
    _validate_extracted_processes(payload, report)
    return _raise_or_return(report)


def validate_post_mapping(payload: Any) -> Report:
    """Validate the structural contract emitted by ID mapping."""

    report = _new_report("post_mapping", "structural", effect_on_failure="abort")
    _validate_payload_container(payload, report)
    if report["errors"]:
        _abort(report)

    assert isinstance(payload, dict)
    entities = _safe_dict(payload.get("entities"))
    species = entities.get("species")
    if not isinstance(species, list) or not species:
        _add_error(
            report,
            "species_required",
            "Mapped payload must include at least one species row.",
            "/entities/species",
        )

    for bucket, row, idx in _iter_entity_rows(entities):
        if not _text(row.get("name")):
            continue
        if "mapping_meta" not in row:
            _add_error(
                report,
                "entity_missing_mapping_meta",
                "Mapped entity with a name is missing mapping_meta.",
                f"/entities/{bucket}/{idx}/mapping_meta",
                bucket=bucket,
                name=row.get("name"),
            )

    return _raise_or_return(report)


def validate_post_normalization(payload: Any, gate_report: Mapping[str, Any] | None = None) -> Report:
    """
    Validate the post-normalization boundary.

    Structural garbage still aborts. Semantic gate failures are returned as a
    report so the orchestrator can pass them to the audit loop.
    """

    report = _new_report("post_normalization", "semantic", effect_on_failure="feed_audit")
    _validate_payload_container(payload, report)
    if report["errors"]:
        report["contract_type"] = "structural"
        report["effect_on_failure"] = "abort"
        _abort(report)

    gate_report = dict(gate_report or {})
    gate_errors = _issue_list(gate_report.get("errors"))
    if gate_errors or gate_report.get("ok") is False:
        report["ok"] = False
        report["errors"] = gate_errors
        report["gate_report"] = gate_report
        report["summary"] = _summary(report)
    return report


def validate_post_audit(payload: Any) -> Report:
    """Validate that audit returned a structurally usable payload."""

    report = _new_report("post_audit", "structural", effect_on_failure="abort")
    _validate_payload_container(payload, report)
    if report["errors"]:
        _abort(report)

    assert isinstance(payload, dict)
    _validate_named_entities(payload, report)
    return _raise_or_return(report)


def validate_pre_export(payload: Any, *, strict_db: bool = True) -> Report:
    """Validate the final semantic PWML contract before export."""

    report = _new_report("pre_export", "semantic", effect_on_failure="abort")
    if not isinstance(payload, dict):
        _add_error(report, "invalid_payload", "Pre-export payload must be a dict.", "/")
        _abort(report)

    from t2pw.pwml.ir import validate_required_pwml_contract

    pwml_report = validate_required_pwml_contract(payload, strict_db=strict_db)
    report["pwml_contract_report"] = pwml_report
    report["warnings"] = _issue_list(pwml_report.get("warnings"))
    if pwml_report.get("ok") is False:
        report["ok"] = False
        report["errors"] = _issue_list(pwml_report.get("errors"))
        report["summary"] = _summary(report)
        _abort(report, "Pre-export PWML contract failed.")

    report["summary"] = _summary(report)
    return report


def _new_report(stage: str, contract_type: str, *, effect_on_failure: str) -> Report:
    return {
        "ok": True,
        "stage": stage,
        "contract_type": contract_type,
        "effect_on_failure": effect_on_failure,
        "errors": [],
        "warnings": [],
        "summary": {"error_count": 0, "warning_count": 0},
    }


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _issue_list(value: Any) -> List[Issue]:
    return [item for item in _safe_list(value) if isinstance(item, dict)]


def _add_error(report: Report, code: str, message: str, pointer: str = "", **extra: Any) -> None:
    issue: Issue = {"code": code, "message": message}
    if pointer:
        issue["pointer"] = pointer
    issue.update(extra)
    report.setdefault("errors", []).append(issue)
    report["ok"] = False
    report["summary"] = _summary(report)


def _summary(report: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "error_count": len(_safe_list(report.get("errors"))),
        "warning_count": len(_safe_list(report.get("warnings"))),
    }


def _abort(report: Report, message: str | None = None) -> None:
    errors = _safe_list(report.get("errors"))
    first = _safe_dict(errors[0]) if errors else {}
    detail = str(first.get("message") or "Stage contract failed.")
    raise StageContractError(
        str(report.get("stage") or "unknown_stage"),
        message or detail,
        report,
    )


def _raise_or_return(report: Report) -> Report:
    report["summary"] = _summary(report)
    if report["errors"]:
        _abort(report)
    return report


def _validate_payload_container(payload: Any, report: Report) -> None:
    if not isinstance(payload, dict):
        _add_error(report, "invalid_payload", "Payload must be a dict.", "/")
        return
    if not isinstance(payload.get("entities"), dict):
        _add_error(report, "entities_required", "Payload must include an entities object.", "/entities")
    if not isinstance(payload.get("processes"), dict):
        _add_error(report, "processes_required", "Payload must include a processes object.", "/processes")


def _iter_entity_rows(entities: Mapping[str, Any]) -> Iterable[tuple[str, Dict[str, Any], int]]:
    for bucket, rows in entities.items():
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                yield str(bucket), row, idx


def _validate_named_entities(payload: Mapping[str, Any], report: Report) -> None:
    entities = _safe_dict(payload.get("entities"))
    for bucket, rows in entities.items():
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                _add_error(
                    report,
                    "entity_not_object",
                    "Entity rows must be objects.",
                    f"/entities/{bucket}/{idx}",
                    bucket=bucket,
                )
                continue
            if not _text(row.get("name")):
                _add_error(
                    report,
                    "entity_missing_name",
                    "Every entity row must have a non-empty name.",
                    f"/entities/{bucket}/{idx}/name",
                    bucket=bucket,
                )


def _validate_extracted_processes(payload: Mapping[str, Any], report: Report) -> None:
    processes = _safe_dict(payload.get("processes"))
    for bucket, rows in processes.items():
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            pointer = f"/processes/{bucket}/{idx}"
            if not isinstance(row, dict):
                _add_error(
                    report,
                    "process_not_object",
                    "Process rows must be objects.",
                    pointer,
                    bucket=bucket,
                )
                continue
            if not _has_any_field(row, ["inputs", "outputs", "cargo"]):
                _add_error(
                    report,
                    "process_missing_participants",
                    "Every extracted process must include inputs, outputs, or cargo.",
                    pointer,
                    bucket=bucket,
                    required_any=["inputs", "outputs", "cargo"],
                )


def _has_any_field(row: Mapping[str, Any], fields: Sequence[str]) -> bool:
    for field in fields:
        value = row.get(field)
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and value:
            return True
        if _text(value):
            return True
    return False
