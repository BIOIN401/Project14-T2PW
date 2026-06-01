"""Helpers for the future interactive pathway curation loop."""

from __future__ import annotations

import base64
import copy
import json
import tempfile
from pathlib import Path
from typing import Any


INTERACTIVE_CURATOR_SYSTEM_PROMPT = """You are a surgical JSON patch generator for biological pathway curation.

You receive the current pathway JSON, a rendered graph image, QA issues, mapping misses, recent edit history, and a human user's requested change.

Return JSON only in this exact shape:
{
  "patch": [
    {
      "action": "add|replace|remove",
      "json_pointer": "/valid/json/pointer",
      "new_value": null,
      "confidence": 0.95,
      "reason": "brief reason"
    }
  ],
  "rationale": "brief explanation of how you interpreted the user request",
  "change_summary": "short user-facing summary",
  "needs_user_review": true
}

Rules:
- Do not rewrite the full pathway JSON.
- Do not return markdown.
- Make the smallest safe patch.
- Preserve unrelated content.
- The JSON is authoritative. The graph image is only visual context.
- Do not modify mapped_ids or database identifiers.
- Do not invent GO IDs, HMDB IDs, KEGG IDs, ChEBI IDs, UniProt IDs, taxonomy IDs, PathBank IDs, or other database IDs.
- New entities should be added unmapped; deterministic mapping will run later.
- Every reaction must keep at least one input and one output.
- If adding or renaming an entity, update all references consistently.
- If removing an entity, remove or update all references in reactions, transports, interactions, biological states, element locations, and collections.
- Do not create disconnected orphan entities unless explicitly requested.
- Enzymes should be proteins or protein complexes, not compounds.
- Cofactors should be compounds with the appropriate role, not enzymes.
- If the request is ambiguous, make the most conservative safe edit.
- If the request cannot be safely applied, return an empty patch and explain why."""


_CORE_PATHWAY_KEYS = {
    "entities",
    "processes",
    "reactions",
    "transports",
    "interactions",
    "compartments",
    "biological_states",
    "element_locations",
}

_MAPPED_ID_KEYS = {
    "mapped_id",
    "mapped_ids",
    "selected_id",
    "selected_ids",
    "chosen_id",
    "chosen_ids",
    "database_id",
    "database_ids",
    "db_id",
    "db_ids",
    "go_id",
    "hmdb_id",
    "kegg_id",
    "chebi_id",
    "uniprot_id",
    "taxonomy_id",
    "pathbank_id",
    "pathbank_compound_id",
    "pathbank_protein_id",
    "pathbank_complex_id",
    "pathbank_reaction_id",
    "pathwhiz_id",
    "pathwhiz_compound_id",
    "pathwhiz_protein_id",
    "pathwhiz_complex_id",
    "pathwhiz_reaction_id",
    "pwc_id",
    "pwp_id",
    "pwpe_id",
}

_INTERACTIVE_RESPONSE_KEYS = {
    "patch",
    "rationale",
    "change_summary",
    "needs_user_review",
}

_BULKY_KEY_TOKENS = (
    "candidate",
    "candidates",
    "search_result",
    "search_results",
    "match_result",
    "match_results",
    "enrichment",
    "enrichment_cache",
    "enrichment_caches",
    "raw_enrichment",
    "raw_enrichments",
    "enrichment_raw",
    "cache",
    "cached",
)

_RAW_TEXT_KEYS = {
    "raw_text",
    "raw_source_text",
    "source_text",
    "full_text",
    "document_text",
    "article_text",
    "paper_text",
    "extracted_text",
    "ocr_text",
    "abstract_text",
}

_MISS_TOKENS = (
    "fail",
    "failed",
    "failure",
    "miss",
    "missing",
    "novel",
    "unmapped",
    "uncertain",
    "low_conf",
    "low-confidence",
    "low confidence",
    "no_match",
    "no match",
    "ambiguous",
)

_HISTORY_KEYS = (
    "round",
    "user_request",
    "change_summary",
    "patch_op_count",
    "entities_remapped",
    "gate_errors",
    "norm_warnings",
    "patch_apply_errors",
)


def strip_payload_for_interactive_context(payload: dict) -> dict:
    """Return a pathway payload copy with bulky non-authoritative context removed."""
    if not isinstance(payload, dict):
        return {}
    return _strip_payload_value(copy.deepcopy(payload), parent_key=None)


def compact_qa_report(qa_report: dict | None) -> dict:
    """Remove nulls and empty containers from a QA report."""
    if not isinstance(qa_report, dict):
        return {}
    compacted = _drop_empty_values(copy.deepcopy(qa_report))
    return compacted if isinstance(compacted, dict) else {}


def compact_mapping_misses(mapping_report: dict | list | None) -> list[dict]:
    """Return a flat list of likely failed, novel, unmapped, or uncertain mappings."""
    if mapping_report is None:
        return []

    misses: list[dict] = []
    seen: set[str] = set()

    def add_record(record: dict[str, Any]) -> None:
        compacted = compact_qa_report(strip_payload_for_interactive_context(record))
        if not compacted:
            return
        key = json.dumps(compacted, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            return
        seen.add(key)
        misses.append(compacted)

    def walk(value: Any, *, parent_key: str | None = None, suspect_context: bool = False) -> None:
        if isinstance(value, dict):
            current_context = suspect_context or _key_suggests_mapping_miss(parent_key)
            looks_like_entity = _looks_like_entity_mapping_record(value)
            if (_dict_suggests_mapping_miss(value) and (looks_like_entity or current_context)) or (
                current_context and looks_like_entity
            ):
                add_record(value)
                return
            for key, child in value.items():
                walk(
                    child,
                    parent_key=str(key),
                    suspect_context=current_context or _key_suggests_mapping_miss(str(key)),
                )
            return

        if isinstance(value, list):
            current_context = suspect_context or _key_suggests_mapping_miss(parent_key)
            for item in value:
                walk(item, parent_key=parent_key, suspect_context=current_context)

    walk(mapping_report, parent_key=None, suspect_context=isinstance(mapping_report, list))
    return misses


def compact_history(history: list[dict] | None, max_rounds: int = 2) -> list[dict]:
    """Keep recent edit history metadata without full patch payloads."""
    if not isinstance(history, list) or max_rounds <= 0:
        return []

    compacted: list[dict] = []
    for round_record in history[-max_rounds:]:
        if not isinstance(round_record, dict):
            continue
        entry = {key: copy.deepcopy(round_record[key]) for key in _HISTORY_KEYS if key in round_record}
        if "patch_op_count" not in entry and isinstance(round_record.get("patch"), list):
            entry["patch_op_count"] = len(round_record["patch"])
        cleaned = compact_qa_report(entry)
        if cleaned:
            compacted.append(cleaned)
    return compacted


def build_interactive_curator_messages(
    working_json: dict,
    graph_png_bytes: bytes,
    user_request: str,
    qa_report: dict | None = None,
    mapping_misses: list[dict] | None = None,
    history: list[dict] | None = None,
) -> list[dict]:
    """Build an OpenAI-compatible multimodal message list for curator patching."""
    image_b64 = base64.b64encode(graph_png_bytes or b"").decode("ascii")
    text = "\n\n".join(
        [
            "Current Working Pathway JSON",
            json.dumps(
                strip_payload_for_interactive_context(working_json),
                indent=2,
                ensure_ascii=False,
            ),
            "Current QA Issues",
            json.dumps(compact_qa_report(qa_report), indent=2, ensure_ascii=False),
            "Mapping Misses or Uncertain Entities",
            json.dumps(compact_mapping_misses(mapping_misses), indent=2, ensure_ascii=False),
            "Recent Edit History",
            json.dumps(compact_history(history), indent=2, ensure_ascii=False),
            "User Request",
            str(user_request or ""),
        ]
    )

    return [
        {"role": "system", "content": INTERACTIVE_CURATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]


def run_interactive_curator_round(
    working_json: dict,
    graph_png_bytes: bytes,
    user_request: str,
    qa_report: dict | None = None,
    mapping_misses: list[dict] | None = None,
    history: list[dict] | None = None,
    max_tokens: int = 4096,
) -> dict:
    from t2pw.llm.client import chat

    messages = build_interactive_curator_messages(
        working_json=working_json,
        graph_png_bytes=graph_png_bytes,
        user_request=user_request,
        qa_report=qa_report,
        mapping_misses=mapping_misses,
        history=history,
    )

    try:
        raw_response = chat(
            messages,
            temperature=0.2,
            max_tokens=max_tokens,
            response_json=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "patch": [],
            "rationale": "",
            "change_summary": "",
            "needs_user_review": True,
            "error": f"LLM interactive curator call failed: {exc}",
        }

    try:
        parsed = _extract_json_object(raw_response)
        if parsed is None:
            raise ValueError("LLM output was not a JSON object.")
        return validate_interactive_patch_response(parsed)
    except Exception as first_exc:  # noqa: BLE001
        validation_error = str(first_exc)

    retry_messages = _build_interactive_curator_retry_messages(
        user_request=user_request,
        validation_error=validation_error,
    )
    try:
        raw_retry_response = chat(
            retry_messages,
            temperature=0.2,
            max_tokens=max_tokens,
            response_json=True,
        )
        retry_parsed = _extract_json_object(raw_retry_response)
        if retry_parsed is None:
            raise ValueError("LLM retry output was not a JSON object.")
        return validate_interactive_patch_response(retry_parsed)
    except Exception as retry_exc:  # noqa: BLE001
        return {
            "patch": [],
            "rationale": "",
            "change_summary": "",
            "needs_user_review": True,
            "error": f"Initial validation failed: {validation_error}; retry failed: {retry_exc}",
        }


def validate_interactive_patch_response(obj: dict) -> dict:
    """Validate and normalize an interactive curator patch response."""
    if not isinstance(obj, dict):
        raise ValueError("Top-level response must be a JSON object.")

    unexpected_keys = set(obj) - _INTERACTIVE_RESPONSE_KEYS
    if unexpected_keys:
        raise ValueError(
            "Top-level response must contain only patch metadata, not pathway JSON "
            f"or extra keys: {sorted(unexpected_keys)}."
        )

    patch = obj.get("patch")
    if not isinstance(patch, list):
        raise ValueError("Response field 'patch' must be a list.")

    rationale = obj.get("rationale", "")
    if rationale is None:
        rationale = ""
    if not isinstance(rationale, str):
        raise ValueError("Response field 'rationale' must be a string when present.")

    change_summary = obj.get("change_summary", "")
    if change_summary is None:
        change_summary = ""
    if not isinstance(change_summary, str):
        raise ValueError("Response field 'change_summary' must be a string when present.")

    needs_user_review = obj.get("needs_user_review", True)
    if not isinstance(needs_user_review, bool):
        raise ValueError("Response field 'needs_user_review' must be a boolean when present.")

    normalized_patch: list[dict[str, Any]] = []
    for index, raw_op in enumerate(patch):
        if not isinstance(raw_op, dict):
            raise ValueError(f"Patch operation {index} must be an object.")

        action = raw_op.get("action")
        if action not in {"add", "replace", "remove"}:
            raise ValueError(f"Patch operation {index} has invalid action {action!r}.")

        json_pointer = raw_op.get("json_pointer")
        if not isinstance(json_pointer, str) or not json_pointer.startswith("/"):
            raise ValueError(f"Patch operation {index} json_pointer must be a string starting with '/'.")
        if json_pointer == "/":
            raise ValueError(f"Patch operation {index} must not target the whole pathway JSON.")
        if _json_pointer_touches_db_identifier(json_pointer):
            raise ValueError(f"Patch operation {index} must not modify mapped IDs or database identifiers.")

        if action in {"add", "replace"} and "new_value" not in raw_op:
            raise ValueError(f"Patch operation {index} action {action!r} requires new_value.")
        if action in {"add", "replace"}:
            new_value = raw_op.get("new_value")
            if _value_looks_like_full_pathway_payload(new_value):
                raise ValueError(f"Patch operation {index} must not rewrite the full pathway JSON.")
            if _value_contains_db_identifier(new_value):
                raise ValueError(f"Patch operation {index} must not invent mapped IDs or database identifiers.")

        normalized_op = {
            "action": action,
            "json_pointer": json_pointer,
            "new_value": (
                raw_op.get("new_value")
                if action in {"add", "replace"}
                else raw_op.get("new_value", None)
            ),
        }

        if action == "remove" and "new_value" not in raw_op:
            normalized_op["new_value"] = None

        if "confidence" in raw_op:
            confidence = raw_op["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError(f"Patch operation {index} confidence must be numeric when present.")
            normalized_op["confidence"] = confidence

        if "reason" in raw_op:
            reason = raw_op["reason"]
            if reason is None:
                reason = ""
            if not isinstance(reason, str):
                raise ValueError(f"Patch operation {index} reason must be a string when present.")
            normalized_op["reason"] = reason

        normalized_patch.append(normalized_op)

    return {
        "patch": normalized_patch,
        "rationale": rationale,
        "change_summary": change_summary,
        "needs_user_review": needs_user_review,
    }


def apply_patch_and_rerender(
    working_json: dict,
    patch: list[dict],
    mapping_cache_path,
    work_dir,
    id_source: str = "hybrid",
    db_config: dict | None = None,
) -> dict:
    from t2pw.curation.apply_audit_patch import run_apply
    from t2pw.mapping.map_ids import run_mapping
    from t2pw.pipeline.draft_graph import build_draft_graph
    from t2pw.pipeline.draft_graph_render import render_draft_graph_to_png_bytes
    from t2pw.pipeline.process_normalizer import (
        GateValidationError,
        apply_biochemical_aliases,
        canonicalize_same_as_aliases,
        normalize_process_actor_schema,
        run_strict_post_normalization_gates,
    )
    from t2pw.pipeline.qa_graph import generate_qa_report

    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    source_json = copy.deepcopy(working_json) if isinstance(working_json, dict) else {}
    entities_before = extract_entity_names(source_json)
    patch_ops = patch if isinstance(patch, list) else []
    patch_apply_errors: list[str] = []

    patched_json = source_json
    patch_apply_report: dict[str, Any] = {}
    patch_files: list[Path] = []
    try:
        patch_input = _temp_json_path(work_path, "interactive_patch_input")
        patch_file = _temp_json_path(work_path, "interactive_patch_ops")
        patch_output = _temp_json_path(work_path, "interactive_patch_output")
        patch_report_file = _temp_json_path(work_path, "interactive_patch_report")
        patch_files.extend([patch_input, patch_file, patch_output, patch_report_file])

        patch_input.write_text(
            json.dumps(source_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        patch_file.write_text(
            json.dumps(_patch_ops_for_apply(patch_ops), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        patch_apply_report = run_apply(
            patch_input,
            patch_file,
            patch_output,
            apply_report_path=patch_report_file,
        )
        if patch_output.exists():
            loaded = json.loads(patch_output.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                patched_json = loaded

        for rejected in _safe_list(patch_apply_report.get("rejected")):
            if isinstance(rejected, dict):
                index = rejected.get("index", "?")
                reason = rejected.get("reason", "rejected")
                patch_apply_errors.append(f"patch[{index}]: {reason}")
    except Exception as exc:  # noqa: BLE001
        patch_apply_errors.append(f"Patch application failed: {exc}")
        patched_json = source_json
    finally:
        _cleanup_temp_files(patch_files)

    normalization_report: dict[str, Any] = {"summary": {}, "actions": []}
    norm_warnings: list[str] = []
    gate_errors: list[str] = []
    normalized_json = copy.deepcopy(patched_json)

    try:
        apply_biochemical_aliases(normalized_json, report=normalization_report)
    except Exception as exc:  # noqa: BLE001
        norm_warnings.append(f"apply_biochemical_aliases failed: {exc}")

    try:
        canonicalize_same_as_aliases(normalized_json, report=normalization_report)
    except Exception as exc:  # noqa: BLE001
        norm_warnings.append(f"canonicalize_same_as_aliases failed: {exc}")

    try:
        normalize_process_actor_schema(normalized_json, report=normalization_report)
    except Exception as exc:  # noqa: BLE001
        norm_warnings.append(f"normalize_process_actor_schema failed: {exc}")

    try:
        gate_details = run_strict_post_normalization_gates(
            normalized_json,
            report=normalization_report,
            enforce_all_proteins_connected=False,
        )
        normalization_report["gate_details"] = gate_details
    except GateValidationError as exc:
        normalization_report["gate_details"] = exc.details
        gate_errors.extend(_gate_error_strings(exc.details))
        if not gate_errors:
            gate_errors.append(str(exc))
    except Exception as exc:  # noqa: BLE001
        gate_errors.append(str(exc))

    entities_after = extract_entity_names(normalized_json)
    entities_remapped = sorted(
        (entities_after - entities_before) | _patched_entity_names(patch_ops, entities_after)
    )

    updated_json = normalized_json
    mapping_report: dict[str, Any] = {}
    mapping_files: list[Path] = []
    try:
        mapping_input = _temp_json_path(work_path, "interactive_mapping_input")
        mapping_output = _temp_json_path(work_path, "interactive_mapping_output")
        mapping_report_path = _temp_json_path(work_path, "interactive_mapping_report")
        mapping_files.extend([mapping_input, mapping_output, mapping_report_path])

        mapping_input.write_text(
            json.dumps(normalized_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        mapping_report = run_mapping(
            mapping_input,
            mapping_output,
            mapping_report_path,
            cache_path=Path(mapping_cache_path),
            id_source=id_source,
            db_config=db_config,
        )
        if mapping_report_path.exists():
            loaded_report = json.loads(mapping_report_path.read_text(encoding="utf-8"))
            if isinstance(loaded_report, dict):
                mapping_report = loaded_report
        if mapping_output.exists():
            loaded_output = json.loads(mapping_output.read_text(encoding="utf-8"))
            if isinstance(loaded_output, dict):
                updated_json = loaded_output
    except Exception as exc:  # noqa: BLE001
        mapping_report = {"error": f"Mapping failed: {exc}"}
    finally:
        _cleanup_temp_files(mapping_files)

    mapping_misses = compact_mapping_misses(mapping_report)

    graph_dict: dict[str, Any] = {}
    graph_png_bytes = b""
    qa_report: dict[str, Any] = {}
    try:
        draft_graph = build_draft_graph(updated_json)
        graph_dict = draft_graph.to_dict()
        graph_png_bytes = render_draft_graph_to_png_bytes(graph_dict, dpi=100)
        qa_report = generate_qa_report(draft_graph, updated_json)
    except Exception as exc:  # noqa: BLE001
        qa_report = {"error": f"Graph render or QA failed: {exc}"}

    return {
        "updated_json": updated_json,
        "graph": graph_dict,
        "graph_png_bytes": graph_png_bytes,
        "qa_report": qa_report,
        "normalization_report": normalization_report,
        "mapping_report": mapping_report,
        "mapping_misses": mapping_misses,
        "entities_remapped": entities_remapped,
        "norm_warnings": norm_warnings,
        "gate_errors": gate_errors,
        "patch_apply_errors": patch_apply_errors,
    }


def extract_entity_names(payload: dict) -> set[str]:
    """Extract known entity names from a pathway payload defensively."""
    if not isinstance(payload, dict):
        return set()
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        return set()

    names: set[str] = set()
    for key in (
        "compounds",
        "proteins",
        "protein_complexes",
        "nucleic_acids",
        "element_collections",
    ):
        rows = entities.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def _temp_json_path(work_dir: Path, prefix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{prefix}_",
        dir=work_dir,
        delete=False,
        encoding="utf-8",
    )
    path = Path(handle.name)
    handle.close()
    return path


def _cleanup_temp_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _patch_ops_for_apply(patch: list[dict]) -> list[Any]:
    """Convert interactive patch shape to the audit patch applier shape."""
    converted: list[Any] = []
    for raw_op in patch:
        if not isinstance(raw_op, dict):
            converted.append(raw_op)
            continue

        op = dict(raw_op)
        if "action" in op and "op" not in op:
            op["op"] = op["action"]
        if "json_pointer" in op and "path" not in op:
            op["path"] = op["json_pointer"]
        if "new_value" in op and "value" not in op:
            op["value"] = op["new_value"]
        if "reason" in op and "evidence" not in op:
            op["evidence"] = op["reason"]
        converted.append(op)
    return converted


def _patched_entity_names(patch: list[dict], known_after: set[str]) -> set[str]:
    names: set[str] = set()
    for op in patch:
        if not isinstance(op, dict):
            continue

        path = op.get("json_pointer", op.get("path", ""))
        if isinstance(path, str) and path.startswith("/entities/"):
            _collect_names_from_value(op.get("new_value", op.get("value")), names)

        action = op.get("action", op.get("op"))
        if action == "replace" and isinstance(path, str) and path.endswith("/name"):
            value = op.get("new_value", op.get("value"))
            if isinstance(value, str) and value.strip():
                names.add(value.strip())

    if known_after:
        names = {name for name in names if name in known_after}
    return names


def _collect_names_from_value(value: Any, names: set[str]) -> None:
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
        for child in value.values():
            _collect_names_from_value(child, names)
        return
    if isinstance(value, list):
        for item in value:
            _collect_names_from_value(item, names)


def _gate_error_strings(details: Any) -> list[str]:
    if not isinstance(details, dict):
        return []
    errors = details.get("errors")
    if not isinstance(errors, list):
        return []

    out: list[str] = []
    for error in errors:
        if isinstance(error, dict):
            path = error.get("path")
            reason = error.get("reason")
            if path and reason:
                out.append(f"{path}: {reason}")
            elif reason:
                out.append(str(reason))
            else:
                out.append(json.dumps(error, ensure_ascii=False, default=str))
        elif error is not None:
            out.append(str(error))
    return out


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _strip_payload_value(value: Any, *, parent_key: str | None) -> Any:
    if isinstance(value, dict):
        stripped: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if _should_drop_payload_key(key_text, child):
                continue
            stripped[key] = _strip_payload_value(child, parent_key=key_text)
        return stripped

    if isinstance(value, list):
        if _is_huge_evidence_list(parent_key, value):
            return []
        return [_strip_payload_value(item, parent_key=parent_key) for item in value]

    return value


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.replace("```json", "```").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _json_pointer_touches_db_identifier(path: str) -> bool:
    try:
        tokens = _decode_json_pointer(path)
    except ValueError:
        return False
    return any(_is_mapped_id_key(str(token).lower()) for token in tokens)


def _decode_json_pointer(path: str) -> list[str]:
    if not path.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {path}")
    return [token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/")]


def _value_looks_like_full_pathway_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {str(key) for key in value}
    return "entities" in keys and ("processes" in keys or "reactions" in keys)


def _value_contains_db_identifier(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_mapped_id_key(str(key).lower()):
                return True
            if _value_contains_db_identifier(child):
                return True
        return False
    if isinstance(value, list):
        return any(_value_contains_db_identifier(item) for item in value)
    return False


def _build_interactive_curator_retry_messages(*, user_request: str, validation_error: str) -> list[dict]:
    required_shape = {
        "patch": [
            {
                "action": "add|replace|remove",
                "json_pointer": "/valid/json/pointer",
                "new_value": None,
                "confidence": 0.95,
                "reason": "brief reason",
            }
        ],
        "rationale": "brief explanation of how you interpreted the user request",
        "change_summary": "short user-facing summary",
        "needs_user_review": True,
    }
    retry_text = "\n\n".join(
        [
            "Your previous response failed validation.",
            "Validation Error",
            str(validation_error or "Unknown validation error."),
            "Original User Request",
            str(user_request or ""),
            "Required JSON Response Shape",
            json.dumps(required_shape, indent=2, ensure_ascii=False),
            (
                "Return JSON only. Return a patch only, not the full pathway JSON. "
                "Use action add, replace, or remove. json_pointer must start with '/'. "
                "new_value is required for add and replace. For remove, use null new_value."
            ),
        ]
    )
    return [
        {"role": "system", "content": INTERACTIVE_CURATOR_SYSTEM_PROMPT},
        {"role": "user", "content": retry_text},
    ]


def _should_drop_payload_key(key: str, value: Any) -> bool:
    key_lower = key.lower()
    if key_lower in _CORE_PATHWAY_KEYS or _is_mapped_id_key(key_lower):
        return False
    if key_lower in _RAW_TEXT_KEYS:
        return True
    if any(token in key_lower for token in _BULKY_KEY_TOKENS):
        return True
    if "raw" in key_lower and ("text" in key_lower or "source" in key_lower or "enrichment" in key_lower):
        return True
    if key_lower in {"provenance", "evidence", "evidences"} and _is_huge_evidence_list(key_lower, value):
        return True
    return False


def _is_mapped_id_key(key_lower: str) -> bool:
    if key_lower in _MAPPED_ID_KEYS:
        return True
    return ("selected" in key_lower or "chosen" in key_lower or "mapped" in key_lower) and "id" in key_lower


def _is_huge_evidence_list(parent_key: str | None, value: Any) -> bool:
    if not isinstance(value, list):
        return False
    key_lower = (parent_key or "").lower()
    if "evidence" not in key_lower and "provenance" not in key_lower:
        return False
    if len(value) > 8:
        return True
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str)) > 3000
    except TypeError:
        return len(str(value)) > 3000


def _drop_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        compacted = {}
        for key, child in value.items():
            cleaned = _drop_empty_values(child)
            if cleaned is None or cleaned == {} or cleaned == []:
                continue
            compacted[key] = cleaned
        return compacted

    if isinstance(value, list):
        compacted_list = []
        for item in value:
            cleaned = _drop_empty_values(item)
            if cleaned is None or cleaned == {} or cleaned == []:
                continue
            compacted_list.append(cleaned)
        return compacted_list

    return value


def _key_suggests_mapping_miss(key: str | None) -> bool:
    if not key:
        return False
    key_lower = key.lower().replace("_", " ")
    return any(token in key_lower for token in _MISS_TOKENS)


def _dict_suggests_mapping_miss(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        key_lower = str(key).lower()
        if _key_suggests_mapping_miss(key_lower) and value not in (False, None, [], {}):
            return True

        if key_lower in {"mapped", "is_mapped", "success", "ok", "passed"} and value is False:
            return True

        if any(token in key_lower for token in ("status", "result", "reason", "error", "warning")):
            if isinstance(value, str) and _key_suggests_mapping_miss(value):
                return True

        if _confidence_key_is_low(key_lower, value):
            return True

        if key_lower in {"mapped_id", "mapped_ids", "selected_id", "selected_ids"} and value in (
            None,
            "",
            [],
            {},
        ):
            return True

    return False


def _confidence_key_is_low(key_lower: str, value: Any) -> bool:
    if "confidence" not in key_lower and "score" not in key_lower:
        return False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) < 0.75


def _looks_like_entity_mapping_record(record: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in record}
    if keys.intersection(
        {
            "name",
            "label",
            "entity",
            "entity_name",
            "canonical_name",
            "display_name",
            "entity_id",
            "element_id",
            "object_id",
            "type",
            "category",
            "class",
        }
    ):
        return True
    return any("entity" in key for key in keys) or any(key.endswith("_id") for key in keys)
