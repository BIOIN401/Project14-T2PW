"""Observational validation for locked reaction preservation."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME


REPORT_FILENAMES: Dict[str, Tuple[str, ...]] = {
    "after_stage2": (
        "reaction_preservation_report_after_stage2.json",
        "reaction_preservation_after_stage2.json",
    ),
    "after_normalization": (
        "reaction_preservation_report_after_normalization.json",
        "reaction_preservation_after_normalization.json",
    ),
    "after_audit": (
        "reaction_preservation_report_after_audit.json",
        "reaction_preservation_after_audit.json",
    ),
    "before_final_export": (
        "final_reaction_preservation_report.json",
        "reaction_preservation_report_before_final_export.json",
    ),
    "after_final_export": (
        "reaction_preservation_report_after_final_export.json",
    ),
}

_SOURCE_ID_KEYS = (
    "reaction_id",
    "id",
    "key",
    "source_reaction_id",
    "pathwhiz_reaction_id",
    "pathbank_reaction_id",
)
_UNRESOLVED_MARKERS = ("unresolved", "unknown", "tbd", "not specified", "unspecified", "?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(_TOKEN_RE.findall(str(value).casefold()))


def _token_set(value: Any) -> set[str]:
    return set(_TOKEN_RE.findall(str(value or "").casefold()))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _string_items(value: Any, *, dict_keys: Sequence[str] = ("name", "entity")) -> List[str]:
    items = value if isinstance(value, list) else ([value] if value not in (None, "") else [])
    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue
        if isinstance(item, dict):
            for key in dict_keys:
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    out.append(raw.strip())
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


def _locked_reaction_id(reaction: Dict[str, Any]) -> str:
    value = reaction.get("locked_reaction_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _evidence_text(reaction: Dict[str, Any]) -> str:
    evidence = reaction.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        return evidence.strip()
    evidence_quote = reaction.get("evidence_quote")
    if isinstance(evidence_quote, str) and evidence_quote.strip():
        return evidence_quote.strip()
    refs = [str(ref).strip() for ref in _safe_list(reaction.get("source_refs")) if str(ref).strip()]
    if refs:
        return " ".join(refs)
    return ""


def _reaction_name(reaction: Dict[str, Any]) -> str:
    value = reaction.get("name")
    return value.strip() if isinstance(value, str) else ""


def _reaction_inputs(reaction: Dict[str, Any]) -> List[str]:
    return _string_items(reaction.get("inputs"))


def _reaction_outputs(reaction: Dict[str, Any]) -> List[str]:
    return _string_items(reaction.get("outputs"))


def _reaction_modifiers(reaction: Dict[str, Any]) -> List[str]:
    return _string_items(
        _safe_list(reaction.get("modifiers"))
        + _safe_list(reaction.get("enzymes"))
        + _safe_list(reaction.get("modifiers_or_enzymes")),
        dict_keys=("entity", "protein", "protein_complex", "name"),
    )


def _canonical_set(values: Iterable[str]) -> set[str]:
    return {_norm_text(value) for value in values if _norm_text(value)}


def _manifest_entries(locked_manifest: Any) -> List[Dict[str, Any]]:
    if isinstance(locked_manifest, list):
        return [entry for entry in locked_manifest if isinstance(entry, dict)]
    if isinstance(locked_manifest, dict):
        for key in ("locked_reactions", "reactions", "manifest", "details"):
            value = locked_manifest.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
    return []


def _iter_reactions(payload: Any) -> List[Dict[str, Any]]:
    payload_dict = _safe_dict(payload)
    reactions = _safe_list(_safe_dict(payload_dict.get("processes")).get("reactions"))
    if not reactions:
        reactions = _safe_list(payload_dict.get("reactions"))
    return [reaction for reaction in reactions if isinstance(reaction, dict)]


def _iter_quarantine_records(payload: Any) -> List[Dict[str, Any]]:
    payload_dict = _safe_dict(payload)
    containers: List[Any] = []
    for key in ("quarantined_locked_reactions", "quarantined_reactions", "quarantine", "quarantined", "reaction_quarantine"):
        containers.append(payload_dict.get(key))
    processes = _safe_dict(payload_dict.get("processes"))
    for key in ("quarantined_locked_reactions", "quarantined_reactions", "quarantine", "quarantined", "reaction_quarantine"):
        containers.append(processes.get(key))

    records: List[Dict[str, Any]] = []
    def _record_with_original(record: Dict[str, Any]) -> Dict[str, Any]:
        original = _safe_dict(record.get("original_reaction"))
        if original:
            merged = dict(original)
            for key, value in record.items():
                if key != "original_reaction":
                    merged[key] = value
        else:
            merged = dict(record)
        if not merged.get("name") and isinstance(record.get("reaction_name"), str):
            merged["name"] = record["reaction_name"]
        return merged

    for container in containers:
        if isinstance(container, dict):
            records.append(_record_with_original(container))
            records.extend(_record_with_original(item) for item in container.values() if isinstance(item, dict))
        elif isinstance(container, list):
            records.extend(_record_with_original(item) for item in container if isinstance(item, dict))
            for item in container:
                if isinstance(item, str):
                    records.append({"name": item})
    return records


def _has_unresolved_entities(reaction: Dict[str, Any]) -> bool:
    values = _reaction_inputs(reaction) + _reaction_outputs(reaction)
    for value in values:
        norm = _norm_text(value)
        if not norm:
            return True
        if any(marker in norm for marker in _UNRESOLVED_MARKERS):
            return True
    return bool(reaction.get("unresolved_entities"))


def _is_quarantined_record(reaction: Dict[str, Any]) -> bool:
    raw_status = reaction.get("status") or reaction.get("preservation_status") or reaction.get("scope_membership")
    return isinstance(raw_status, str) and "quarantine" in raw_status.casefold()


def _score_candidate(locked: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    locked_id_score = 1.0 if _locked_reaction_id(locked) and _locked_reaction_id(locked) == _locked_reaction_id(candidate) else 0.0
    source_score = 1.0 if _source_reaction_id(locked) and _source_reaction_id(locked) == _source_reaction_id(candidate) else 0.0
    name_score = SequenceMatcher(None, _norm_text(_reaction_name(locked)), _norm_text(_reaction_name(candidate))).ratio()
    input_score = _jaccard(_canonical_set(_reaction_inputs(locked)), _canonical_set(_reaction_inputs(candidate)))
    output_score = _jaccard(_canonical_set(_reaction_outputs(locked)), _canonical_set(_reaction_outputs(candidate)))
    io_score = (input_score + output_score) / 2.0
    evidence_score = _jaccard(_token_set(_evidence_text(locked)), _token_set(_evidence_text(candidate)))
    modifier_score = _jaccard(_canonical_set(_reaction_modifiers(locked)), _canonical_set(_reaction_modifiers(candidate)))
    signals = {
        "locked_reaction_id": locked_id_score,
        "source_reaction_id": source_score,
        "name": name_score,
        "inputs_outputs": io_score,
        "evidence": evidence_score,
        "enzyme_modifier": modifier_score,
    }
    score = (
        0.25 * locked_id_score
        + 0.20 * source_score
        + 0.22 * name_score
        + 0.28 * io_score
        + 0.04 * evidence_score
        + 0.01 * modifier_score
    )
    return round(min(1.0, score), 4), signals


def _candidate_id(reaction: Dict[str, Any], index: int) -> str:
    return _source_reaction_id(reaction) or _locked_reaction_id(reaction) or _reaction_name(reaction) or f"reaction_{index + 1}"


def _base_detail(locked: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "locked_reaction_id": _locked_reaction_id(locked),
        "status": "unknown",
        "score": 0.0,
        "matched_reaction_id": "",
        "warnings": [],
    }


def _quarantine_match_score(locked: Dict[str, Any], quarantine_records: List[Dict[str, Any]]) -> Tuple[float, str]:
    best_score = 0.0
    best_id = ""
    for index, record in enumerate(quarantine_records):
        score, _signals = _score_candidate(locked, record)
        if score > best_score:
            best_score = score
            best_id = _candidate_id(record, index)
    return best_score, best_id


def _status_for_match(
    locked: Dict[str, Any],
    candidate: Dict[str, Any],
    score: float,
    signals: Dict[str, float],
    alternate_scores: List[float],
) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    if score < 0.25:
        return "missing", warnings

    if _is_quarantined_record(candidate):
        return "quarantined", warnings

    if _has_unresolved_entities(candidate):
        warnings.append("matched_reaction_has_unresolved_entities")
        if score < 0.7:
            return "unresolved_entities", warnings

    locked_name = _norm_text(_reaction_name(locked))
    candidate_name = _norm_text(_reaction_name(candidate))
    locked_inputs = _canonical_set(_reaction_inputs(locked))
    candidate_inputs = _canonical_set(_reaction_inputs(candidate))
    locked_outputs = _canonical_set(_reaction_outputs(locked))
    candidate_outputs = _canonical_set(_reaction_outputs(candidate))
    name_changed = bool(locked_name and candidate_name and locked_name != candidate_name)
    io_changed = locked_inputs != candidate_inputs or locked_outputs != candidate_outputs

    if len([alt for alt in alternate_scores if alt >= 0.28]) >= 2 and signals.get("inputs_outputs", 0.0) < 0.75:
        warnings.append("multiple_partial_matches_detected")
        return "split_candidate", warnings

    if signals.get("locked_reaction_id", 0.0) == 0 and signals.get("source_reaction_id", 0.0) == 0 and score < 0.45:
        warnings.append("weak_match")
        return "unknown", warnings

    if name_changed:
        warnings.append("name_changed")
    if io_changed:
        warnings.append("inputs_outputs_changed")

    if io_changed:
        return "preserved_with_input_output_changes", warnings
    if name_changed:
        return "preserved_with_name_changes", warnings
    return "preserved_exactly", warnings


def validate_reaction_preservation(current_json: Any, locked_manifest: Any, *, stage: str = "") -> Dict[str, Any]:
    """Compare current pathway JSON against a locked reaction manifest."""
    locked_entries = _manifest_entries(locked_manifest)
    current_reactions = _iter_reactions(current_json)
    quarantine_records = _iter_quarantine_records(current_json)
    candidates = current_reactions + quarantine_records

    details: List[Dict[str, Any]] = []
    matched_to_details: Dict[str, List[Dict[str, Any]]] = {}
    for locked in locked_entries:
        detail = _base_detail(locked)
        quarantine_score, quarantine_id = _quarantine_match_score(locked, quarantine_records)
        if quarantine_score >= 0.35:
            detail.update(
                {
                    "status": "quarantined",
                    "score": round(quarantine_score, 4),
                    "matched_reaction_id": quarantine_id,
                    "warnings": ["matched_quarantine_record"],
                }
            )
            details.append(detail)
            matched_to_details.setdefault(quarantine_id, []).append(detail)
            continue

        scored: List[Tuple[float, Dict[str, float], Dict[str, Any], int]] = []
        for index, candidate in enumerate(candidates):
            score, signals = _score_candidate(locked, candidate)
            scored.append((score, signals, candidate, index))
        scored.sort(key=lambda item: item[0], reverse=True)

        if not scored:
            detail["status"] = "missing"
            details.append(detail)
            continue

        best_score, best_signals, best_candidate, best_index = scored[0]
        alternate_scores = [score for score, _signals, _candidate, _index in scored[1:4]]
        status, warnings = _status_for_match(
            locked,
            best_candidate,
            best_score,
            best_signals,
            alternate_scores,
        )
        matched_id = _candidate_id(best_candidate, best_index) if best_score >= 0.25 else ""
        detail.update(
            {
                "status": status,
                "score": round(best_score, 4),
                "matched_reaction_id": matched_id,
                "warnings": warnings,
            }
        )
        details.append(detail)
        if matched_id:
            matched_to_details.setdefault(matched_id, []).append(detail)

    for matched_id, matched_details in matched_to_details.items():
        if len(matched_details) <= 1:
            continue
        for detail in matched_details:
            if detail["status"].startswith("preserved") or detail["status"] in {"unknown", "split_candidate"}:
                detail["status"] = "merged_candidate"
                detail.setdefault("warnings", []).append("multiple_locked_reactions_match_same_current_reaction")

    preserved_statuses = {
        "preserved_exactly",
        "preserved_with_name_changes",
        "preserved_with_input_output_changes",
        "merged_candidate",
        "split_candidate",
    }
    changed_statuses = {"preserved_with_name_changes", "preserved_with_input_output_changes", "merged_candidate", "split_candidate"}
    report = {
        "stage": stage,
        "locked_reaction_count": len(locked_entries),
        "preserved_count": sum(1 for detail in details if detail["status"] in preserved_statuses),
        "missing_count": sum(1 for detail in details if detail["status"] == "missing"),
        "changed_count": sum(1 for detail in details if detail["status"] in changed_statuses),
        "quarantined_count": sum(1 for detail in details if detail["status"] == "quarantined"),
        "details": details,
    }
    return report


def load_locked_reaction_manifest(manifest_path: str | Path) -> Optional[Any]:
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_reaction_preservation_report(
    current_json: Any,
    locked_manifest: Any,
    output_dir: str | Path,
    *,
    stage: str,
) -> Dict[str, Any]:
    report = validate_reaction_preservation(current_json, locked_manifest, stage=stage)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filenames = REPORT_FILENAMES.get(stage, (f"reaction_preservation_report_{stage}.json",))
    for filename in filenames:
        (out_dir / filename).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_reaction_preservation_report_if_manifest(
    current_json: Any,
    manifest_dir: str | Path,
    *,
    stage: str,
) -> Optional[Dict[str, Any]]:
    manifest_dir_path = Path(manifest_dir)
    manifest = load_locked_reaction_manifest(manifest_dir_path / MANIFEST_FILENAME)
    if manifest is None:
        return None
    return write_reaction_preservation_report(
        current_json,
        manifest,
        manifest_dir_path,
        stage=stage,
    )
