"""Build and write observational Stage-1 reaction lock artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MANIFEST_FILENAME = "locked_reaction_manifest.json"
REPORT_FILENAME = "reaction_lock_report.json"
RAW_STAGE1_FILENAME = "stage1_raw_extraction.json"

_SOURCE_ID_KEYS = (
    "reaction_id",
    "id",
    "key",
    "source_reaction_id",
    "pathwhiz_reaction_id",
    "pathbank_reaction_id",
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_string_items(value: Any, *, dict_keys: Iterable[str] = ("name", "entity")) -> List[str]:
    items = value if isinstance(value, list) else ([value] if value not in (None, "") else [])
    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            if item.strip():
                out.append(item)
            continue
        if isinstance(item, dict):
            for key in dict_keys:
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    out.append(raw)
                    break
    return out


def _source_reaction_id(reaction: Dict[str, Any]) -> str:
    for key in _SOURCE_ID_KEYS:
        value = reaction.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def _scope_membership(reaction: Dict[str, Any]) -> str:
    raw = reaction.get("scope_membership")
    if isinstance(raw, str) and raw.strip():
        scope = raw.strip()
        if scope.casefold() == "out_of_scope":
            return "out_of_scope"
        return scope
    return "core"


def _evidence_quote(reaction: Dict[str, Any]) -> str:
    evidence = reaction.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        return evidence

    for ref in _safe_list(reaction.get("source_refs")):
        if isinstance(ref, str) and ref.strip():
            return ref

    for modifier in _safe_list(reaction.get("modifiers")) + _safe_list(reaction.get("enzymes")):
        if isinstance(modifier, dict):
            mod_evidence = modifier.get("evidence")
            if isinstance(mod_evidence, str) and mod_evidence.strip():
                return mod_evidence

    return ""


def _source_chunk(reaction: Dict[str, Any], fallback: Optional[str]) -> str:
    for key in ("source_chunk", "source_chunk_id", "chunk_id"):
        value = reaction.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return fallback or ""


def _sort_key(entry: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    source_id = str(entry.get("source_reaction_id") or "")
    name = str(entry.get("name") or "")
    return (
        source_id or name,
        source_id,
        name,
        "\u001f".join(entry.get("inputs") or []),
        "\u001f".join(entry.get("outputs") or []),
        str(entry.get("source_chunk") or ""),
    )


def build_locked_reaction_manifest(
    payloads: Dict[str, Any] | Iterable[Dict[str, Any]],
    *,
    source_chunk: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return the Stage-1 reaction manifest and summary report.

    The input is expected to be raw Stage-1 output. Reactions marked
    ``out_of_scope`` are excluded, but malformed in-scope reactions are kept and
    marked ``incomplete`` so later stages can observe the original Stage-1 set.
    """
    if isinstance(payloads, dict):
        payload_entries: Iterable[Dict[str, Any]] = [{"payload": payloads, "source_chunk": source_chunk}]
    else:
        payload_entries = payloads

    entries: List[Dict[str, Any]] = []
    out_of_scope_count = 0

    for payload_entry in payload_entries:
        has_wrapper_payload = isinstance(payload_entry, dict) and "payload" in payload_entry
        payload = _safe_dict(payload_entry.get("payload") if has_wrapper_payload else payload_entry)
        fallback_chunk = ""
        if has_wrapper_payload:
            raw_chunk = payload_entry.get("source_chunk", source_chunk)
            if raw_chunk not in (None, ""):
                fallback_chunk = str(raw_chunk)

        reactions = _safe_list(_safe_dict(payload.get("processes")).get("reactions"))
        for reaction in reactions:
            if not isinstance(reaction, dict):
                continue

            scope = _scope_membership(reaction)
            if scope.casefold() == "out_of_scope":
                out_of_scope_count += 1
                continue

            inputs = _as_string_items(reaction.get("inputs"))
            outputs = _as_string_items(reaction.get("outputs"))
            modifiers = _as_string_items(
                _safe_list(reaction.get("enzymes")) + _safe_list(reaction.get("modifiers")),
                dict_keys=("entity", "protein", "protein_complex", "name"),
            )
            evidence = _evidence_quote(reaction)

            issues: List[str] = []
            if not inputs:
                issues.append("missing_inputs")
            if not outputs:
                issues.append("missing_outputs")
            if not evidence:
                issues.append("missing_evidence")

            entry: Dict[str, Any] = {
                "locked_reaction_id": "",
                "source_reaction_id": _source_reaction_id(reaction),
                "name": reaction.get("name") if isinstance(reaction.get("name"), str) else "",
                "inputs": inputs,
                "outputs": outputs,
                "modifiers_or_enzymes": modifiers,
                "evidence_quote": evidence,
                "scope_membership": scope,
                "source_chunk": _source_chunk(reaction, fallback_chunk),
                "confidence": reaction.get("confidence") if isinstance(reaction.get("confidence"), (int, float)) else 0.0,
                "locked": True,
                "status": "incomplete" if issues else "valid",
            }
            if issues:
                entry["issues"] = issues
            entries.append(entry)

    entries.sort(key=_sort_key)
    for idx, entry in enumerate(entries, start=1):
        entry["locked_reaction_id"] = f"rxn_lock_{idx:03d}"

    incomplete = [entry for entry in entries if entry["status"] == "incomplete"]
    report = {
        "locked_reaction_count": len(entries),
        "valid_count": len(entries) - len(incomplete),
        "incomplete_count": len(incomplete),
        "out_of_scope_excluded_count": out_of_scope_count,
        "duplicate_candidates": [],
        "incomplete_reactions": [
            {
                "locked_reaction_id": entry["locked_reaction_id"],
                "source_reaction_id": entry.get("source_reaction_id", ""),
                "name": entry.get("name", ""),
                "issues": entry.get("issues", []),
            }
            for entry in incomplete
        ],
    }
    return entries, report


def write_locked_reaction_manifest(
    payloads: Dict[str, Any] | Iterable[Dict[str, Any]],
    output_dir: str | Path,
    *,
    source_chunk: Optional[str] = None,
) -> Dict[str, Any]:
    """Write locked reaction manifest artifacts into *output_dir*."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest, report = build_locked_reaction_manifest(payloads, source_chunk=source_chunk)
    manifest_path = out_path / MANIFEST_FILENAME
    report_path = out_path / REPORT_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "manifest": manifest,
        "report": report,
    }


def write_stage1_lock_artifacts(
    payloads: Dict[str, Any] | Iterable[Dict[str, Any]],
    output_dir: str | Path,
    *,
    source_chunk: Optional[str] = None,
) -> Dict[str, Any]:
    """Write raw Stage-1 output plus observational lock artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_path = out_path / RAW_STAGE1_FILENAME
    raw_path.write_text(json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8")
    result = write_locked_reaction_manifest(payloads, out_path, source_chunk=source_chunk)
    result["raw_stage1_path"] = str(raw_path)
    return result
