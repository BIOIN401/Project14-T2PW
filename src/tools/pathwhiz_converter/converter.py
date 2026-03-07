from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

from .llm_plan import request_patch_plan
from .sbml_clean import run_deterministic_cleanup
from .sbml_metrics import compute_model_metrics
from .sbml_parse import parse_sbml_bytes, serialize_document, validate_document
from .sbml_patch import apply_patch_plan
from .sbml_summarize import summarize_model
from .schemas import ConverterReport, PatchPlan

MODE_DETERMINISTIC = "Deterministic cleanup only"
MODE_LLM = "LLM-assisted (learn from examples)"


def _hash_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _count_block(metrics: Dict[str, Any]) -> Dict[str, int]:
    return {
        "isolated_species_count": int(metrics.get("isolated_species_count", 0)),
        "duplicate_reactions_count": int(metrics.get("duplicate_reactions_count", 0)),
        "compartments_count": int(metrics.get("compartments_count", 0)),
    }


def _parse_example_summaries(example_sbml_files: Sequence[Tuple[str, bytes]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    summaries: List[Dict[str, Any]] = []
    warnings: List[Dict[str, str]] = []
    for file_name, payload in example_sbml_files:
        try:
            doc, model = parse_sbml_bytes(payload)
            summary = summarize_model(model)
            summary["file_name"] = file_name
            summary["validation"] = {
                "has_errors": bool(validate_document(doc).get("has_errors", False)),
            }
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            warnings.append({"type": "EXAMPLE_PARSE_FAILED", "message": f"{file_name}: {exc}"})
    return summaries, warnings


def convert_sbml_for_pathwhiz(
    *,
    input_sbml_bytes: bytes,
    mode: str,
    llm_client: Any = None,
    example_sbml_files: Sequence[Tuple[str, bytes]] | None = None,
    allow_add_reaction: bool = False,
    max_llm_rounds: int = 2,
) -> Dict[str, Any]:
    doc, model = parse_sbml_bytes(input_sbml_bytes)
    validation_initial = validate_document(doc)
    if validation_initial.get("has_errors"):
        raise ValueError(
            f"Input SBML failed libSBML validation ({validation_initial.get('severe_count', 0)} severe messages)."
        )

    metrics_before = compute_model_metrics(model)
    cleanup = run_deterministic_cleanup(model)
    deterministic_ops = list(cleanup.get("ops", []))
    warnings: List[Dict[str, Any]] = list(cleanup.get("warnings", []))
    metrics_post_clean = compute_model_metrics(model)

    validation_post_clean = validate_document(doc)
    if validation_post_clean.get("has_errors"):
        raise ValueError(
            f"Deterministic cleanup produced invalid SBML ({validation_post_clean.get('severe_count', 0)} severe messages)."
        )

    applied_ops: List[Dict[str, Any]] = list(deterministic_ops)
    llm_rounds: List[Dict[str, Any]] = []

    input_hashes: Dict[str, Any] = {"input_sbml_sha256": _hash_bytes(input_sbml_bytes), "example_files": []}
    example_summaries: List[Dict[str, Any]] = []
    if mode == MODE_LLM:
        if not llm_client:
            raise ValueError("LLM mode selected but llm_client was not provided.")
        files = list(example_sbml_files or [])
        if not files:
            raise ValueError("LLM mode requires at least one PathWhiz-working example SBML file.")
        for name, payload in files:
            input_hashes["example_files"].append({"file_name": name, "sha256": _hash_bytes(payload)})
        example_summaries, parse_warnings = _parse_example_summaries(files)
        warnings.extend(parse_warnings)
        if not example_summaries:
            raise ValueError("None of the uploaded example SBML files could be parsed.")

        rounds = max(1, min(int(max_llm_rounds), 2))
        for round_idx in range(rounds):
            current_metrics = compute_model_metrics(model)
            model_summary = summarize_model(
                model,
                metrics={
                    "pre_clean_metrics": _count_block(metrics_before),
                    "post_clean_baseline_metrics": _count_block(metrics_post_clean),
                    "current_metrics": _count_block(current_metrics),
                },
            )
            plan, llm_debug = request_patch_plan(
                llm_client=llm_client,
                input_model_summary=model_summary,
                example_summaries=example_summaries,
                allow_add_reaction=bool(allow_add_reaction),
            )
            apply_result = apply_patch_plan(model, plan, allow_add_reaction=bool(allow_add_reaction))
            applied_now = list(apply_result.get("applied_ops", []))
            apply_warnings = list(apply_result.get("warnings", []))
            for w in plan.warnings:
                warnings.append(w.model_dump())
            warnings.extend(apply_warnings)
            applied_ops.extend(applied_now)
            after_metrics = compute_model_metrics(model)
            llm_rounds.append(
                {
                    "round": round_idx + 1,
                    "proposed_ops": len(plan.ops),
                    "applied_ops": len(applied_now),
                    "metrics_before": _count_block(current_metrics),
                    "metrics_after": _count_block(after_metrics),
                    "llm_raw_preview": str(llm_debug.get("raw", ""))[:1200],
                }
            )
            if not applied_now:
                break
            if (
                int(after_metrics.get("isolated_species_count", 0)) == 0
                and int(after_metrics.get("duplicate_reactions_count", 0)) == 0
            ):
                break

    metrics_after = compute_model_metrics(model)
    validation_final = validate_document(doc)

    patch_plan = PatchPlan.model_validate(
        {
            "version": "1.0",
            "objective": "Deterministic PathWhiz cleanup"
            if mode == MODE_DETERMINISTIC
            else "Deterministic PathWhiz cleanup + LLM-assisted style alignment",
            "ops": applied_ops,
            "warnings": [
                {
                    "type": str(item.get("type", "WARNING")),
                    "message": str(item.get("message", "")),
                }
                for item in warnings
                if isinstance(item, dict)
            ],
        }
    )

    report = ConverterReport.model_validate(
        {
            "mode_used": mode,
            "allow_add_reaction": bool(allow_add_reaction),
            "counts_before": _count_block(metrics_before),
            "counts_after": _count_block(metrics_after),
            "applied_ops": applied_ops,
            "warnings": warnings,
            "validation": {
                "initial": validation_initial,
                "post_clean": validation_post_clean,
                "final": validation_final,
            },
            "llm_rounds": llm_rounds,
            "input_hashes": input_hashes,
        }
    )

    return {
        "converted_sbml_bytes": serialize_document(doc),
        "patch_plan": patch_plan.model_dump(),
        "report": report.model_dump(),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }

