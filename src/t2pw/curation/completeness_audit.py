"""Final completeness audit for the T2PW pipeline.

This module provides an optional, non-mutating audit that compares the final
pathway JSON against the locked reaction manifest and the original source text.
It uses a (potentially stronger) LLM to identify any missed or changed
reactions, producing a report without modifying the pathway.

Usage::

    from t2pw.curation.completeness_audit import run_final_completeness_audit

    report = run_final_completeness_audit(
        source_text=original_text,
        locked_manifest_path=Path("tmp/locked_reaction_manifest.json"),
        final_pathway_json=final_payload,
        output_path=Path("tmp"),
    )
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

AUDIT_REPORT_FILENAME = "final_completeness_audit.json"

# ---------------------------------------------------------------------------
# System prompt for the completeness audit LLM call
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a biological pathway completeness auditor. Your job is to verify that \
the final pathway JSON faithfully represents the locked reaction manifest and \
that no obvious in-scope reactions from the source text were missed.

Rules:
- Do NOT rewrite the pathway.
- Do NOT return patches.
- Do NOT add new reactions directly.
- Only produce a completeness report as a JSON object.

Identify:
- locked reactions that are preserved in the final JSON
- locked reactions that are missing from the final JSON
- locked reactions that were changed (name, inputs, outputs modified)
- locked reactions that were quarantined
- obvious in-scope reactions from the source text that are NOT in the locked manifest
- questionable out-of-scope reactions that might actually be in-scope
- confidence and evidence for each finding

Return ONLY a single JSON object with exactly these top-level keys:
  locked_reaction_count, preserved_count, missing_locked_reactions,
  changed_locked_reactions, quarantined_locked_reactions,
  possible_missed_in_scope_reactions, possible_out_of_scope_reactions,
  overall_completeness, confidence

overall_completeness must be one of: "complete", "mostly_complete",
"incomplete", "critically_incomplete"

Each entry in missing_locked_reactions, changed_locked_reactions,
quarantined_locked_reactions, possible_missed_in_scope_reactions, and
possible_out_of_scope_reactions should have: name, evidence, confidence, reason.
"""


# ---------------------------------------------------------------------------
# Model resolution helper
# ---------------------------------------------------------------------------
_MODEL_ENV_VARS = (
    "OPENROUTER_FINAL_COMPLETENESS_MODEL",
    "OPENROUTER_EXTRACTION_MODEL",
    "OPENROUTER_MODEL",
)


def _resolve_completeness_model() -> Optional[str]:
    """Try env vars in priority order and return the first non-empty value.

    Returns ``None`` when no explicit override is found, which lets the
    ``chat()`` fallback chain handle resolution.
    """
    for env_var in _MODEL_ENV_VARS:
        value = (os.getenv(env_var) or "").strip()
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# User prompt construction helpers
# ---------------------------------------------------------------------------
_MAX_SOURCE_WORDS = 8000


def _truncate_source_text(text: str, max_words: int = _MAX_SOURCE_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + f"\n\n[... truncated at {max_words} words ...]"


def _compact_reactions(pathway_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract a compact view of reactions: name, inputs, outputs."""
    processes = pathway_json.get("processes", {})
    if not isinstance(processes, dict):
        processes = {}
    reactions = processes.get("reactions", [])
    if not isinstance(reactions, list):
        reactions = []

    compact: List[Dict[str, str]] = []
    for rxn in reactions:
        if not isinstance(rxn, dict):
            continue
        entry: Dict[str, Any] = {"name": rxn.get("name", "")}
        inputs = rxn.get("inputs", [])
        outputs = rxn.get("outputs", [])
        entry["inputs"] = _string_items(inputs)
        entry["outputs"] = _string_items(outputs)
        locked_id = rxn.get("locked_reaction_id", "")
        if locked_id:
            entry["locked_reaction_id"] = locked_id
        compact.append(entry)
    return compact


def _string_items(value: Any) -> List[str]:
    """Normalize a list of items to a list of strings."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for key in ("name", "entity"):
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    out.append(raw)
                    break
    return out


def _build_user_prompt(
    *,
    source_text: str,
    preprocessor_context: Optional[Dict[str, Any]],
    locked_manifest: List[Dict[str, Any]],
    final_pathway_json: Dict[str, Any],
    final_preservation_report: Optional[Dict[str, Any]],
    quarantined_reactions: Optional[List[Dict[str, Any]]],
) -> str:
    """Assemble the user prompt with all relevant context sections."""
    sections: List[str] = []

    # 1. Source text
    truncated = _truncate_source_text(source_text)
    sections.append(
        "## Source Text\n\n" + truncated
    )

    # 2. Preprocessor context summary
    if preprocessor_context:
        ctx_summary = json.dumps(preprocessor_context, indent=2, ensure_ascii=False)
        # Keep it manageable
        if len(ctx_summary) > 3000:
            ctx_summary = ctx_summary[:3000] + "\n... [truncated]"
        sections.append(
            "## Preprocessor Context Summary\n\n" + ctx_summary
        )

    # 3. Locked reaction manifest (full)
    manifest_json = json.dumps(locked_manifest, indent=2, ensure_ascii=False)
    sections.append(
        "## Locked Reaction Manifest\n\n" + manifest_json
    )

    # 4. Final pathway reactions (compact)
    compact = _compact_reactions(final_pathway_json)
    compact_json = json.dumps(compact, indent=2, ensure_ascii=False)
    sections.append(
        "## Final Pathway Reactions (compact)\n\n" + compact_json
    )

    # 5. Preservation report
    if final_preservation_report:
        pres_json = json.dumps(final_preservation_report, indent=2, ensure_ascii=False)
        if len(pres_json) > 4000:
            pres_json = pres_json[:4000] + "\n... [truncated]"
        sections.append(
            "## Preservation Report\n\n" + pres_json
        )

    # 6. Quarantined reactions
    if quarantined_reactions:
        quar_json = json.dumps(quarantined_reactions, indent=2, ensure_ascii=False)
        sections.append(
            "## Quarantined Reactions\n\n" + quar_json
        )

    sections.append(
        "Produce the completeness report as a single JSON object. "
        "Do not include any text outside the JSON object."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Manifest loading helper
# ---------------------------------------------------------------------------
def _load_manifest(
    locked_manifest_path: Optional[Path],
    locked_manifest: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Return the manifest list, loading from path if needed."""
    if locked_manifest is not None:
        return locked_manifest

    if locked_manifest_path is not None:
        path = Path(locked_manifest_path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            # Some manifests wrap the list in a dict
            if isinstance(data, dict):
                for key in ("locked_reactions", "reactions", "manifest"):
                    if isinstance(data.get(key), list):
                        return data[key]
            return None
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------
_VALID_COMPLETENESS = {"complete", "mostly_complete", "incomplete", "critically_incomplete"}


def _parse_llm_response(raw: str) -> Dict[str, Any]:
    """Parse the LLM JSON response, applying light validation."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

    report = json.loads(text)
    if not isinstance(report, dict):
        raise ValueError("LLM response is not a JSON object")

    # Validate and normalize overall_completeness
    completeness = report.get("overall_completeness", "")
    if completeness not in _VALID_COMPLETENESS:
        report["overall_completeness"] = "incomplete"

    # Ensure numeric fields
    for key in ("locked_reaction_count", "preserved_count"):
        if key in report:
            try:
                report[key] = int(report[key])
            except (TypeError, ValueError):
                report[key] = 0

    if "confidence" in report:
        try:
            report["confidence"] = float(report["confidence"])
        except (TypeError, ValueError):
            report["confidence"] = 0.0

    # Ensure list fields
    for key in (
        "missing_locked_reactions",
        "changed_locked_reactions",
        "quarantined_locked_reactions",
        "possible_missed_in_scope_reactions",
        "possible_out_of_scope_reactions",
    ):
        if not isinstance(report.get(key), list):
            report[key] = []

    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_final_completeness_audit(
    *,
    source_text: str,
    preprocessor_context: Optional[Dict[str, Any]] = None,
    locked_manifest_path: Optional[Path] = None,
    locked_manifest: Optional[List[Dict[str, Any]]] = None,
    final_pathway_json: Dict[str, Any],
    final_preservation_report: Optional[Dict[str, Any]] = None,
    quarantined_reactions: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Run an optional final completeness audit on the pipeline output.

    This function is read-only: it never mutates ``final_pathway_json``.

    Parameters
    ----------
    source_text:
        The original source text fed into the pipeline.
    preprocessor_context:
        Optional preprocessor context dict from the pipeline.
    locked_manifest_path:
        Path to the locked reaction manifest JSON file.
    locked_manifest:
        Pre-loaded locked manifest list (takes precedence over path).
    final_pathway_json:
        The final pathway JSON after all pipeline stages.
    final_preservation_report:
        Optional preservation report from the validator.
    quarantined_reactions:
        Optional list of quarantined reaction dicts.
    output_path:
        If provided, the report is written as ``final_completeness_audit.json``
        inside this directory.
    temperature:
        LLM temperature for the audit call.
    max_tokens:
        Maximum tokens for the LLM response.
    enabled:
        When ``False``, return a minimal skip report immediately.

    Returns
    -------
    dict
        The completeness audit report.
    """
    # --- Disabled ---
    if not enabled:
        report: Dict[str, Any] = {"enabled": False, "skipped": True}
        _maybe_write_report(report, output_path)
        return report

    # --- No manifest ---
    manifest = _load_manifest(locked_manifest_path, locked_manifest)
    if manifest is None:
        report = {"enabled": True, "skipped": True, "reason": "no_locked_manifest"}
        _maybe_write_report(report, output_path)
        return report

    # --- Build prompts ---
    user_prompt = _build_user_prompt(
        source_text=source_text,
        preprocessor_context=preprocessor_context,
        locked_manifest=manifest,
        final_pathway_json=final_pathway_json,
        final_preservation_report=final_preservation_report,
        quarantined_reactions=quarantined_reactions,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # --- LLM call ---
    try:
        from t2pw.llm.client import chat

        model_override = _resolve_completeness_model()
        raw_response = chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_json=True,
            model_override=model_override,
            model_env_var="OPENROUTER_FINAL_COMPLETENESS_MODEL",
            stage_name="final_completeness_audit",
        )
        report = _parse_llm_response(raw_response)
        report["enabled"] = True
        report["llm_ok"] = True

    except Exception as exc:
        logger.warning("Final completeness audit LLM call failed: %s", exc)
        report = {
            "enabled": True,
            "error": str(exc),
            "llm_ok": False,
        }

    _maybe_write_report(report, output_path)
    return report


# ---------------------------------------------------------------------------
# File output helper
# ---------------------------------------------------------------------------
def _maybe_write_report(report: Dict[str, Any], output_path: Optional[Path]) -> None:
    """Write the report to ``output_path / AUDIT_REPORT_FILENAME`` if a path is given."""
    if output_path is None:
        return
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / AUDIT_REPORT_FILENAME
    file_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Final completeness audit report written to %s", file_path)
