"""
pathway_curator.py

One-shot LLM curation step run after the audit loop, before ID mapping.
The curator receives the post-audit JSON and a fresh reaction summary and:
  1. Fixes entity name mismatches between the entity list and reaction
     inputs/outputs (e.g. "NAD" vs "NAD+").
  2. Assigns compartments to entities currently showing unknown/empty state.
  3. Proposes missing transporter proteins for transport reactions whose
     Transporter field is MISSING.
  4. Adds a reaction_order list if one does not already exist.

All changes are emitted as JSON-Pointer patches in the format that
apply_audit_patch.apply_patch_with_policy already understands.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from t2pw.curation.apply_audit_patch import (
    APPLIED_PATCH_LOG_FILENAME,
    REJECTED_PATCH_LOG_FILENAME,
    apply_patch_with_policy,
    committed_change_count,
)
from t2pw.llm.client import chat_with_tools
from t2pw.paths import PROMPTS_DIR
from t2pw.pipeline.reaction_preservation_validator import load_locked_reaction_manifest
from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def _load_nearby_locked_manifest(*paths: Path) -> Any | None:
    seen_dirs: set[Path] = set()
    for path in paths:
        parent = path if path.is_dir() else path.parent
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        manifest = load_locked_reaction_manifest(parent / MANIFEST_FILENAME)
        if manifest is not None:
            return manifest
    return None


# ---------------------------------------------------------------------------
# System prompt (loaded once)
# ---------------------------------------------------------------------------

_CURATOR_SYSTEM_PROMPT: Optional[str] = None


def _get_curator_system_prompt() -> str:
    global _CURATOR_SYSTEM_PROMPT  # noqa: PLW0603
    if _CURATOR_SYSTEM_PROMPT is None:
        prompt_path = PROMPTS_DIR / "pathway_curator_system.txt"
        _CURATOR_SYSTEM_PROMPT = (
            prompt_path.read_text(encoding="utf-8")
            if prompt_path.exists()
            else (
                "You are a biochemistry curator. Fix entity name mismatches, "
                "fill unknown compartments, add missing transporters, and propose "
                "a reaction_order list using propose_patch."
            )
        )
    return _CURATOR_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tool definitions (propose_patch only — no external lookups needed)
# ---------------------------------------------------------------------------

_CURATOR_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "propose_patch",
            "description": (
                "Commit a JSON patch operation to the pathway payload. "
                "Use this to write your curation decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["add", "replace", "remove"],
                        "description": "JSON patch operation type",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "JSON Pointer path, e.g. /entities/compounds/3/name "
                            "or /reaction_order"
                        ),
                    },
                    "value": {
                        "description": "New value (required for add/replace).",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "One sentence explaining why this patch is proposed.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0.0–1.0.",
                    },
                },
                "required": ["op", "path", "evidence", "confidence"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Build entity index (same approach as gap_resolver._build_entity_index)
# ---------------------------------------------------------------------------

def _build_entity_index(payload: Dict[str, Any]) -> Dict[str, Tuple[str, int]]:
    """Return {normalized_name: (json_pointer_prefix, index)}."""
    out: Dict[str, Tuple[str, int]] = {}
    entities = _safe_dict(payload.get("entities"))
    for list_key, prefix in [
        ("compounds", "/entities/compounds"),
        ("proteins", "/entities/proteins"),
        ("protein_complexes", "/entities/protein_complexes"),
        ("nucleic_acids", "/entities/nucleic_acids"),
    ]:
        for idx, item in enumerate(_safe_list(entities.get(list_key))):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                out[name.lower()] = (prefix, idx)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pathway_curator(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    reaction_summary: Optional[str] = None,
    pathway_name: str = "",
    organism: str = "",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 2000,
) -> Dict[str, Any]:
    """
    Run the curator agent on *input_path*, write the patched payload to
    *output_path*, and write a report dict to *report_path*.

    Returns the report dict.
    """
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Curator input JSON must be an object.")

    accumulated_patches: List[Dict[str, Any]] = []

    def tool_executor(tool_name: str, tool_args: Dict[str, Any]) -> Any:
        if tool_name == "propose_patch":
            patch = {
                "op": tool_args.get("op"),
                "path": tool_args.get("path"),
                "value": tool_args.get("value"),
                "evidence": tool_args.get("evidence", ""),
                "confidence": float(tool_args.get("confidence", 0.7)),
            }
            accumulated_patches.append(patch)
            return {"accepted": True, "patch_index": len(accumulated_patches) - 1}
        return {"error": f"unknown tool: {tool_name}"}

    entity_index = _build_entity_index(payload)

    # Build the processes summary sent to the LLM
    processes = _safe_dict(payload.get("processes"))
    transport_list = [
        {"index": i, "name": t.get("name", ""), "from": t.get("from_state", ""),
         "to": t.get("to_state", ""), "modifiers": _safe_list(t.get("modifiers"))}
        for i, t in enumerate(_safe_list(processes.get("transports")))
        if isinstance(t, dict)
    ]
    reaction_list = [
        {"index": i, "name": r.get("name", ""),
         "inputs": _safe_list(r.get("inputs")), "outputs": _safe_list(r.get("outputs"))}
        for i, r in enumerate(_safe_list(processes.get("reactions")))
        if isinstance(r, dict)
    ]

    user_payload: Dict[str, Any] = {
        "task": (
            "Curate this pathway JSON. Fix entity name mismatches, fill unknown "
            "compartments, propose missing transporters, and always propose a "
            "reaction_order patch (add or replace) with the correct biological sequence."
        ),
        "pathway_name": pathway_name,
        "organism": organism,
        "entity_index": {k: list(v) for k, v in entity_index.items()},
        "reactions": reaction_list,
        "transports": transport_list,
        "reaction_order_exists": "reaction_order" in payload,
    }
    if reaction_summary and reaction_summary.strip():
        user_payload["reaction_summary"] = reaction_summary.strip()

    user_content = json.dumps(user_payload, ensure_ascii=False)

    report: Dict[str, Any] = {
        "summary": {
            "patches_proposed": 0,
            "patches_accepted": 0,
            "patches_rejected": 0,
        },
        "patches": [],
        "apply_summary": {},
    }

    final_text = ""
    try:
        final_text = chat_with_tools(
            messages=[
                {"role": "system", "content": _get_curator_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            tools=_CURATOR_TOOLS,
            tool_executor=tool_executor,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            max_tool_rounds=20,
            model_env_var="OPENROUTER_CURATOR_MODEL",
            stage_name="curator",
        )
        report["raw"] = final_text[:600]
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        report["raw"] = final_text[:400]

    # Deduplicate patches by (op, path)
    seen: Dict[Tuple[str, str], bool] = {}
    deduped: List[Dict[str, Any]] = []
    for p in accumulated_patches:
        key = (str(p.get("op", "")), str(p.get("path", "")))
        if key not in seen:
            seen[key] = True
            deduped.append(p)

    report["summary"]["patches_proposed"] = len(deduped)
    report["patches"] = deduped

    if deduped:
        locked_manifest = _load_nearby_locked_manifest(input_path, output_path, report_path)
        patched_payload, apply_report = apply_patch_with_policy(
            payload,
            deduped,
            locked_manifest=locked_manifest,
            stage="curator",
            applied_log_path=(report_path.parent / APPLIED_PATCH_LOG_FILENAME) if locked_manifest is not None else None,
            rejected_log_path=(report_path.parent / REJECTED_PATCH_LOG_FILENAME) if locked_manifest is not None else None,
        )
        apply_summary = _safe_dict(apply_report.get("summary", {}))
        # Committed, not accepted: a rolled-back batch left the payload untouched
        # and must not be reported here as curation progress.
        report["summary"]["patches_accepted"] = committed_change_count(apply_report)
        report["summary"]["patches_rejected"] = int(apply_summary.get("rejected_count", 0))
        report["apply_summary"] = apply_summary
        if "lock_policy" in apply_report:
            report["lock_policy"] = apply_report["lock_policy"]
    else:
        patched_payload = deepcopy(payload)

    output_path.write_text(json.dumps(patched_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report
