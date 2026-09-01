from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from t2pw.pipeline import lineage
from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME

# The referential-integrity guard below MUST agree, name for name, with the Stage 3
# gate that later rejects the payload, so it borrows that gate's own private
# helpers instead of re-deriving entity identity locally. See the long comment
# above _registry_coverage() for why re-deriving it would be a bug in itself.
# process_normalizer pulls in nothing but stdlib plus three pure pipeline modules
# (entity_identity, enzyme_cues, export_mode), so this import adds no optional
# dependency and cannot cycle back into t2pw.curation. That matters here: the
# 2026-07-28 07:xx batch burned all 56 legs in 67 seconds on a single
# ModuleNotFoundError raised at import time, and this module must never become the
# next such tripwire.
from t2pw.pipeline.process_normalizer import (  # noqa: PLC2701 - deliberate reuse, see below
    _actor_name_from_row as _registry_actor_name,
    _canonical as _registry_canonical,
    _entity_name_norms as _registry_name_norms,
    _normalize as _registry_normalize,
)


logger = logging.getLogger(__name__)

DEFAULT_CONNECTIVITY_CONFIDENCE_THRESHOLD = 0.98
DEFAULT_MAJOR_TOPOLOGY_CONFIDENCE_THRESHOLD = 0.98
APPLIED_PATCH_LOG_FILENAME = "applied_patch_log.json"
REJECTED_PATCH_LOG_FILENAME = "rejected_patch_log.json"

_SOURCE_ID_KEYS = (
    "reaction_id",
    "id",
    "key",
    "source_reaction_id",
    "pathwhiz_reaction_id",
    "pathbank_reaction_id",
)


def _decode_pointer(path: str) -> List[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {path}")
    tokens = path[1:].split("/")
    return [token.replace("~1", "/").replace("~0", "~") for token in tokens]


def _is_array_index(token: str) -> bool:
    return token.isdigit()


def _resolve_parent(doc: Any, tokens: Sequence[str]) -> Tuple[Any, str]:
    if not tokens:
        raise ValueError("Cannot resolve parent of root pointer.")
    current = doc
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                current[token] = {}
            current = current[token]
        elif isinstance(current, list):
            if not _is_array_index(token):
                raise ValueError(f"Expected array index token, got: {token}")
            idx = int(token)
            if idx < 0 or idx >= len(current):
                raise IndexError(f"Array index out of range: {idx}")
            current = current[idx]
        else:
            raise TypeError(f"Cannot traverse through non-container: {type(current).__name__}")
    return current, tokens[-1]


def _set_value(doc: Any, path: str, value: Any, op: str) -> None:
    tokens = _decode_pointer(path)
    if not tokens:
        raise ValueError("Root replacement is not allowed.")
    parent, leaf = _resolve_parent(doc, tokens)
    if isinstance(parent, dict):
        if op == "replace" and leaf not in parent:
            raise KeyError(f"replace target does not exist: {path}")
        parent[leaf] = value
        return
    if isinstance(parent, list):
        if leaf == "-" and op == "add":
            parent.append(value)
            return
        if not _is_array_index(leaf):
            raise ValueError(f"Expected array index at leaf token: {leaf}")
        idx = int(leaf)
        if op == "add":
            if idx < 0 or idx > len(parent):
                raise IndexError(f"add index out of range: {idx}")
            parent.insert(idx, value)
            return
        if idx < 0 or idx >= len(parent):
            raise IndexError(f"{op} index out of range: {idx}")
        parent[idx] = value
        return
    raise TypeError(f"Cannot assign into non-container: {type(parent).__name__}")


def _remove_value(doc: Any, path: str) -> None:
    tokens = _decode_pointer(path)
    if not tokens:
        raise ValueError("Root removal is not allowed.")
    parent, leaf = _resolve_parent(doc, tokens)
    if isinstance(parent, dict):
        if leaf not in parent:
            raise KeyError(f"remove target does not exist: {path}")
        del parent[leaf]
        return
    if isinstance(parent, list):
        if not _is_array_index(leaf):
            raise ValueError(f"Expected array index at leaf token: {leaf}")
        idx = int(leaf)
        if idx < 0 or idx >= len(parent):
            raise IndexError(f"remove index out of range: {idx}")
        parent.pop(idx)
        return
    raise TypeError(f"Cannot remove from non-container: {type(parent).__name__}")


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_connectivity_path(path: str) -> bool:
    return "/processes/reactions/" in path and (
        path.endswith("/inputs")
        or path.endswith("/outputs")
        or "/inputs/" in path
        or "/outputs/" in path
    )


def _is_major_topology_path(path: str) -> bool:
    return bool(
        re.match(
            r"^/processes/(?:reactions|transports|reaction_coupled_transports|interactions)(?:/\d+|/-)?$",
            path,
        )
    )


def _is_core_semantics_path(path: str) -> bool:
    return any(
        token in path
        for token in [
            "/processes/reactions/",
            "/processes/transports/",
            "/processes/reaction_coupled_transports/",
        ]
    )


def _threshold_for_op(op: Dict[str, Any]) -> float:
    action = str(op.get("op", "")).lower()
    path = str(op.get("path", ""))
    if action == "add" and ("subcellular_location" in path or "compartment" in path):
        return 0.70
    if action == "add":
        return 0.75
    if action == "replace":
        return 0.88
    if action == "remove":
        return 0.95
    return 1.0


def _get_value_at_pointer(doc: Any, path: str) -> Any:
    tokens = _decode_pointer(path)
    current = doc
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"Pointer does not exist: {path}")
            current = current[token]
            continue
        if isinstance(current, list):
            if not _is_array_index(token):
                raise ValueError(f"Expected array index token, got: {token}")
            idx = int(token)
            if idx < 0 or idx >= len(current):
                raise IndexError(f"Array index out of range: {idx}")
            current = current[idx]
            continue
        raise TypeError(f"Cannot traverse through non-container: {type(current).__name__}")
    return current


def _split_tokens(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    parts = re.split(r"\s*\+\s*|\s+and\s+", value.strip(), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]


def _canonical_token(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", cleaned)


def _flatten_process_tokens(raw: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, str):
            continue
        for token in _split_tokens(item):
            norm = _canonical_token(token)
            if norm:
                out.append(norm)
    return out


def _is_noop_reaction_obj(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    inputs = sorted(_flatten_process_tokens(obj.get("inputs")))
    outputs = sorted(_flatten_process_tokens(obj.get("outputs")))
    return bool(inputs) and inputs == outputs


def _is_noop_transport_obj(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    src = (obj.get("from_biological_state") or "").strip() if isinstance(obj.get("from_biological_state"), str) else ""
    dst = (obj.get("to_biological_state") or "").strip() if isinstance(obj.get("to_biological_state"), str) else ""
    cargo = (obj.get("cargo") or "").strip() if isinstance(obj.get("cargo"), str) else ""
    return bool(src and dst and cargo) and src == dst


def _is_safe_core_remove(op: Dict[str, Any], source_payload: Dict[str, Any]) -> bool:
    path = str(op.get("path", ""))
    confidence = _float_or_default(op.get("confidence"), 0.0)
    evidence = str(op.get("evidence", "")).strip()
    if confidence < 0.97 or not evidence:
        return False
    if not re.match(r"^/processes/(reactions|transports|reaction_coupled_transports)/\d+$", path):
        return False
    try:
        target = _get_value_at_pointer(source_payload, path)
    except Exception:  # noqa: BLE001
        return False
    if "/processes/reactions/" in path:
        return _is_noop_reaction_obj(target)
    if "/processes/transports/" in path:
        return _is_noop_transport_obj(target)
    return False


def _normalize_patch_op(op: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise enrichment-format patches to internal format.

    Accepts:
    - ``action`` as an alias for ``op``
    - ``reason`` as an alias for ``evidence``
    Returns a shallow copy with both aliases resolved.
    """
    if not isinstance(op, dict):
        return op
    normalized = dict(op)
    if "action" in normalized and "op" not in normalized:
        normalized["op"] = normalized.pop("action")
    if "reason" in normalized and "evidence" not in normalized:
        normalized["evidence"] = normalized.pop("reason")
    return normalized


def _entity_path_from_mapped_ids_patch(path: str) -> Optional[str]:
    """Given a path like /entities/compounds/0/mapped_ids/chebi,
    return the parent entity path /entities/compounds/0.

    Returns None if the path does not match the expected pattern.
    """
    match = re.match(
        r"^(/entities/(?:compounds|proteins|protein_complexes|nucleic_acids)/\d+)/mapped_ids/",
        path,
    )
    return match.group(1) if match else None


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _manifest_entries(locked_manifest: Any) -> List[Dict[str, Any]]:
    if isinstance(locked_manifest, list):
        return [entry for entry in locked_manifest if isinstance(entry, dict)]
    if isinstance(locked_manifest, dict):
        for key in ("locked_reactions", "reactions", "manifest", "details"):
            value = locked_manifest.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
    return []


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


def _norm_lock_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", str(value or "").strip().casefold())).strip()


def _canonical_item_set(values: Iterable[Any]) -> set[str]:
    return {_norm_lock_text(value) for value in values if _norm_lock_text(value)}


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


def _reaction_name(reaction: Dict[str, Any]) -> str:
    value = reaction.get("name")
    return value.strip() if isinstance(value, str) else ""


def _iter_reactions(payload: Any) -> List[Dict[str, Any]]:
    payload_dict = _safe_dict(payload)
    reactions = _safe_list(_safe_dict(payload_dict.get("processes")).get("reactions"))
    if not reactions:
        reactions = _safe_list(payload_dict.get("reactions"))
    return [reaction for reaction in reactions if isinstance(reaction, dict)]


def _build_lock_context(locked_manifest: Any) -> Optional[Dict[str, Any]]:
    entries = _manifest_entries(locked_manifest)
    if not entries:
        return None

    locked_ids = {
        str(entry.get("locked_reaction_id", "")).strip()
        for entry in entries
        if str(entry.get("locked_reaction_id", "")).strip()
    }
    used_entity_to_lock_id: Dict[str, str] = {}
    for entry in entries:
        lock_id = str(entry.get("locked_reaction_id", "")).strip()
        used_items = (
            _string_items(entry.get("inputs"))
            + _string_items(entry.get("outputs"))
            + _string_items(entry.get("modifiers_or_enzymes"))
        )
        for item in used_items:
            key = _norm_lock_text(item)
            if key:
                used_entity_to_lock_id.setdefault(key, lock_id)

    return {
        "entries": entries,
        "locked_ids": locked_ids,
        "used_entity_to_lock_id": used_entity_to_lock_id,
    }


def _match_lock_entry_for_reaction(reaction: Any, lock_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(reaction, dict):
        return None

    explicit_lock_id = _locked_reaction_id(reaction)
    if explicit_lock_id:
        for entry in lock_context["entries"]:
            if str(entry.get("locked_reaction_id", "")).strip() == explicit_lock_id:
                return entry

    source_id = _source_reaction_id(reaction)
    if source_id:
        for entry in lock_context["entries"]:
            if str(entry.get("source_reaction_id", "")).strip() == source_id:
                return entry

    name = _norm_lock_text(_reaction_name(reaction))
    inputs = _canonical_item_set(_reaction_inputs(reaction))
    outputs = _canonical_item_set(_reaction_outputs(reaction))
    if not name or not inputs or not outputs:
        return None

    for entry in lock_context["entries"]:
        if _norm_lock_text(entry.get("name")) != name:
            continue
        entry_inputs = _canonical_item_set(_string_items(entry.get("inputs")))
        entry_outputs = _canonical_item_set(_string_items(entry.get("outputs")))
        if entry_inputs == inputs and entry_outputs == outputs:
            return entry
    return None


def _locked_id_for_reaction(reaction: Any, lock_context: Dict[str, Any]) -> str:
    entry = _match_lock_entry_for_reaction(reaction, lock_context)
    return str(entry.get("locked_reaction_id", "")).strip() if entry else ""


def _reaction_path_info(tokens: Sequence[str]) -> Optional[Tuple[Tuple[str, ...], int, List[str]]]:
    if len(tokens) >= 2 and tokens[0] == "reactions" and _is_array_index(tokens[1]):
        return ("reactions",), int(tokens[1]), list(tokens[2:])
    if len(tokens) >= 3 and tokens[0] == "processes" and tokens[1] == "reactions" and _is_array_index(tokens[2]):
        return ("processes", "reactions"), int(tokens[2]), list(tokens[3:])
    return None


def _is_reactions_array_path(tokens: Sequence[str]) -> bool:
    return list(tokens) in (["reactions"], ["processes", "reactions"])


def _is_quarantine_path(tokens: Sequence[str]) -> bool:
    return any("quarantine" in str(token).casefold() or "quarantined" in str(token).casefold() for token in tokens)


def _reaction_at_path(payload: Dict[str, Any], list_path: Tuple[str, ...], index: int) -> Optional[Dict[str, Any]]:
    current: Any = payload
    for token in list_path:
        current = _safe_dict(current).get(token)
    if not isinstance(current, list) or index < 0 or index >= len(current):
        return None
    item = current[index]
    return item if isinstance(item, dict) else None


def _entity_path_info(tokens: Sequence[str]) -> Optional[Tuple[str, Optional[int], List[str]]]:
    if len(tokens) < 2 or tokens[0] != "entities":
        return None
    bucket = tokens[1]
    if bucket not in {"compounds", "proteins"}:
        return None
    if len(tokens) == 2:
        return bucket, None, []
    if _is_array_index(tokens[2]):
        return bucket, int(tokens[2]), list(tokens[3:])
    return None


def _entity_name(entity: Any) -> str:
    if not isinstance(entity, dict):
        return ""
    value = entity.get("name")
    return value.strip() if isinstance(value, str) else ""


def _entity_list(payload: Dict[str, Any], bucket: str) -> List[Any]:
    return _safe_list(_safe_dict(payload.get("entities")).get(bucket))


def _entity_delete_rejection(
    source_payload: Dict[str, Any],
    op: Dict[str, Any],
    tokens: Sequence[str],
    lock_context: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    entity_info = _entity_path_info(tokens)
    if entity_info is None:
        return None

    bucket, index, tail = entity_info
    action = str(op.get("op", "")).lower()
    used = lock_context["used_entity_to_lock_id"]
    if action == "remove" and index is not None and not tail:
        entities = _entity_list(source_payload, bucket)
        if index < len(entities):
            key = _norm_lock_text(_entity_name(entities[index]))
            if key in used:
                return "attempted_to_delete_locked_reaction_entity", used[key]
    if action == "remove" and index is None:
        for entity in _entity_list(source_payload, bucket):
            key = _norm_lock_text(_entity_name(entity))
            if key in used:
                return "attempted_to_delete_locked_reaction_entity", used[key]
    if action == "replace" and index is None:
        replacement_names = {
            _norm_lock_text(_entity_name(item))
            for item in _safe_list(op.get("value"))
            if _norm_lock_text(_entity_name(item))
        }
        for entity in _entity_list(source_payload, bucket):
            key = _norm_lock_text(_entity_name(entity))
            if key in used and key not in replacement_names:
                return "attempted_to_delete_locked_reaction_entity", used[key]
    return None


def _all_locked_ids_in_value(value: Any, locked_ids: set[str]) -> List[str]:
    found: List[str] = []
    if isinstance(value, dict):
        for child in value.values():
            found.extend(_all_locked_ids_in_value(child, locked_ids))
    elif isinstance(value, list):
        for item in value:
            found.extend(_all_locked_ids_in_value(item, locked_ids))
    elif isinstance(value, str) and value in locked_ids:
        found.append(value)
    return found


def _explicit_lock_id_counts(payload: Dict[str, Any], lock_context: Dict[str, Any]) -> Dict[str, int]:
    counts = {lock_id: 0 for lock_id in lock_context["locked_ids"]}
    for reaction in _iter_reactions(payload):
        lock_id = _locked_reaction_id(reaction)
        if lock_id in counts:
            counts[lock_id] += 1
    return counts


def _contains_identity_trace(reaction: Dict[str, Any], locked_entry: Dict[str, Any], old_name: str) -> bool:
    if _locked_reaction_id(reaction) == str(locked_entry.get("locked_reaction_id", "")).strip():
        return True
    if _source_reaction_id(reaction) and _source_reaction_id(reaction) == str(locked_entry.get("source_reaction_id", "")).strip():
        return True

    old_norm = _norm_lock_text(old_name)
    if not old_norm:
        return False

    trace_values: List[Any] = [
        reaction.get("original_name"),
        reaction.get("previous_name"),
        reaction.get("evidence"),
        reaction.get("evidence_quote"),
    ]
    trace_values.extend(_safe_list(reaction.get("aliases")))
    trace_values.extend(_safe_list(reaction.get("same_as")))
    trace_values.extend(_safe_list(reaction.get("source_refs")))
    for value in trace_values:
        if old_norm and old_norm in _norm_lock_text(value):
            return True
    return False


def _validate_locked_reaction_shape(
    reaction: Any,
    locked_entry: Dict[str, Any],
    old_reaction: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    lock_id = str(locked_entry.get("locked_reaction_id", "")).strip()
    if not isinstance(reaction, dict):
        return "attempted_to_delete_locked_reaction", lock_id
    if _locked_reaction_id(old_reaction or {}) and _locked_reaction_id(reaction) != _locked_reaction_id(old_reaction or {}):
        return "attempted_to_change_locked_reaction_id", lock_id
    if "locked_reaction_id" in reaction and _locked_reaction_id(reaction) and _locked_reaction_id(reaction) != lock_id:
        return "attempted_to_change_locked_reaction_id", lock_id
    if not _reaction_inputs(reaction):
        return "attempted_to_remove_all_locked_reaction_inputs", lock_id
    if not _reaction_outputs(reaction):
        return "attempted_to_remove_all_locked_reaction_outputs", lock_id

    old_name = _reaction_name(old_reaction or locked_entry)
    new_name = _reaction_name(reaction)
    if old_name and new_name and _norm_lock_text(old_name) != _norm_lock_text(new_name):
        if not _contains_identity_trace(reaction, locked_entry, old_name):
            return "attempted_to_rename_locked_reaction_without_traceability", lock_id
    return None


def _validate_lock_pre_op(
    source_payload: Dict[str, Any],
    op: Dict[str, Any],
    lock_context: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    action = str(op.get("op", "")).lower()
    path = str(op.get("path", ""))
    try:
        tokens = _decode_pointer(path)
    except Exception:  # noqa: BLE001
        return None

    if _is_reactions_array_path(tokens) and action == "replace":
        return "attempted_to_replace_reactions_array", ""
    if _is_reactions_array_path(tokens) and action == "remove":
        first_lock_id = ""
        if lock_context["entries"]:
            first_lock_id = str(lock_context["entries"][0].get("locked_reaction_id", "")).strip()
        return "attempted_to_delete_locked_reaction", first_lock_id

    if not _is_quarantine_path(tokens):
        entity_rejection = _entity_delete_rejection(source_payload, op, tokens, lock_context)
        if entity_rejection is not None:
            return entity_rejection

    reaction_info = _reaction_path_info(tokens)
    if reaction_info is not None:
        list_path, index, tail = reaction_info
        old_reaction = _reaction_at_path(source_payload, list_path, index)
        locked_entry = _match_lock_entry_for_reaction(old_reaction, lock_context)
        lock_id = str(locked_entry.get("locked_reaction_id", "")).strip() if locked_entry else ""

        if locked_entry and action == "remove" and not tail:
            return "attempted_to_delete_locked_reaction", lock_id
        if locked_entry and tail and tail[0] == "locked_reaction_id" and action in {"replace", "remove"}:
            return "attempted_to_change_locked_reaction_id", lock_id
        if locked_entry and tail and tail[0] in {"inputs", "outputs"}:
            side = tail[0]
            old_items = _string_items((old_reaction or {}).get(side))
            if action == "remove" and (len(tail) == 1 or len(old_items) <= 1):
                reason = (
                    "attempted_to_remove_all_locked_reaction_inputs"
                    if side == "inputs"
                    else "attempted_to_remove_all_locked_reaction_outputs"
                )
                return reason, lock_id
            if action == "replace" and len(tail) == 1 and not _string_items(op.get("value")):
                reason = (
                    "attempted_to_remove_all_locked_reaction_inputs"
                    if side == "inputs"
                    else "attempted_to_remove_all_locked_reaction_outputs"
                )
                return reason, lock_id
        if locked_entry and action == "replace" and not tail:
            return _validate_locked_reaction_shape(op.get("value"), locked_entry, old_reaction)

    if not _is_quarantine_path(tokens) and action in {"add", "replace"}:
        ids_in_value = set(_all_locked_ids_in_value(op.get("value"), lock_context["locked_ids"]))
        if len(ids_in_value) > 1:
            return "attempted_to_merge_locked_reactions", sorted(ids_in_value)[0]
    return None


def _validate_lock_post_op(
    before_payload: Dict[str, Any],
    after_payload: Dict[str, Any],
    op: Dict[str, Any],
    lock_context: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    try:
        tokens = _decode_pointer(str(op.get("path", "")))
    except Exception:  # noqa: BLE001
        tokens = []
    if _is_quarantine_path(tokens):
        return None

    before_counts = _explicit_lock_id_counts(before_payload, lock_context)
    after_counts = _explicit_lock_id_counts(after_payload, lock_context)
    for lock_id, count in before_counts.items():
        if count > 0 and after_counts.get(lock_id, 0) == 0:
            return "attempted_to_delete_locked_reaction", lock_id
    for lock_id, count in after_counts.items():
        if count > 1:
            return "attempted_to_split_locked_reaction", lock_id

    for reaction in _iter_reactions(after_payload):
        ids_in_reaction = set(_all_locked_ids_in_value(reaction, lock_context["locked_ids"]))
        if len(ids_in_reaction) > 1:
            return "attempted_to_merge_locked_reactions", sorted(ids_in_reaction)[0]

    reaction_info = _reaction_path_info(tokens)
    if reaction_info is not None:
        list_path, index, tail = reaction_info
        old_reaction = _reaction_at_path(before_payload, list_path, index)
        locked_entry = _match_lock_entry_for_reaction(old_reaction, lock_context)
        if locked_entry and tail and tail[0] == "name" and str(op.get("op", "")).lower() == "replace":
            new_reaction = _reaction_at_path(after_payload, list_path, index)
            if new_reaction is not None:
                return _validate_locked_reaction_shape(new_reaction, locked_entry, old_reaction)
    return None


def _patch_log_record(stage: str, op: Any, reason: str, locked_reaction_id: str = "") -> Dict[str, Any]:
    return {
        "stage": stage,
        "patch": op,
        "reason": reason,
        "locked_reaction_id": locked_reaction_id,
    }


def _append_json_log(path: Path, records: List[Dict[str, Any]]) -> None:
    existing: List[Any] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except Exception:  # noqa: BLE001
            existing = []
    if not records and path.exists():
        return
    path.write_text(json.dumps(existing + records, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_locked_manifest(path: str | Path | None) -> Optional[Any]:
    if path is None:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _discover_locked_manifest_path(paths: Sequence[Optional[Path]]) -> Optional[Path]:
    seen_dirs: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        parent = path if path.is_dir() else path.parent
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        candidate = parent / MANIFEST_FILENAME
        if candidate.exists():
            return candidate
    return None


def _apply_single_op(working: Dict[str, Any], op: Dict[str, Any]) -> None:
    action = str(op["op"]).lower()
    path = str(op["path"])
    if action == "remove":
        _remove_value(working, path)
    else:
        _set_value(working, path, op.get("value"), action)

    enrichment_provenance = op.get("provenance")
    if enrichment_provenance is not None:
        entity_path = _entity_path_from_mapped_ids_patch(path)
        if entity_path:
            try:
                entity_obj = _get_value_at_pointer(working, entity_path)
                if isinstance(entity_obj, dict):
                    meta = entity_obj.setdefault("mapping_meta", {})
                    meta["provenance"] = str(enrichment_provenance)
                    if op.get("confidence") is not None:
                        meta["confidence"] = _float_or_default(op.get("confidence"))
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Referential integrity on entity removal.
#
# WHY THIS EXISTS -- two failures, both reproducible from artifacts on disk:
#
#   1. runs/2026-07-27_1623/papers/PMC13231680__mechanistic-insights-into-
#      phthalylsulfacetamide/strict/final_stage3_gate_report.json records
#          "/processes/reactions/2/inputs/0 unknown entity: phthalylsulfacetamide (PSA)"
#      while that same leg's merged_payload.json still carries five near-duplicate
#      compound rows: "phthalylsulfacetamide" (twice), "phthalylsulfacetamide (PSA)",
#      "PSA" and "sulfacetamide". Normalization output passed the registry gate
#      with all five present -- a curation step then deleted the redundant
#      "phthalylsulfacetamide (PSA)" row as duplicate cleanup and left
#      reactions[2].inputs[0], which still spells the name WITH the parenthetical,
#      pointing at nothing.
#   2. runs/2026-07-28_0919 PMC12444477/research failed the same shape at scale:
#      24 of the 25 errors in final_stage3_gate_report.json are
#      unknown_protein_modifier_reference, e.g.
#          /processes/reactions/8/enzymes/7  ->  "acetyl-CoA carboxylase enzyme
#          complex (comprising AccA, B, C, and D components)"
#      That leg burned 2778s before failing.
#
# WHY THE HOLE WAS THERE: _is_core_semantics_path() scopes only to
# "/processes/reactions/", "/processes/transports/" and
# "/processes/reaction_coupled_transports/", so the _is_safe_core_remove() guard in
# _should_accept() never looks at "/entities/...". A remove of /entities/compounds/N
# therefore needed nothing beyond confidence >= 0.95 (_threshold_for_op), and
# audit_json_llm.py's prompt actively solicits exactly that: its patch policy lists
# "duplicate cleanup" under "High confidence (>=0.95)". The net effect was an
# asymmetry -- _is_safe_core_remove refuses to delete a REACTION unless it is a
# provable no-op at >= 0.97, but the entities that reaction points at could be
# deleted freely, and _apply_single_op does a bare remove with no cascade.
#
# WHY THIS RUNS AFTER APPLICATION AND NOT INSIDE _should_accept(): _should_accept
# is handed `source_payload`, not the live `working` copy (see the call site in
# apply_patch_with_policy). For a multi-op patch that payload is stale -- after the
# first "remove /entities/compounds/3" every later index has shifted by one, so a
# pre-application check would inspect the wrong row and refuse (or clear) the wrong
# entity. Diffing the payload immediately before and after the op is the only way
# to know what the op actually cost, and it makes the guard shape-agnostic: it
# catches a whole-row remove, a whole-bucket remove, a "remove /entities/compounds/3/name"
# and a shortening "replace /entities/compounds" with one predicate instead of four.
#
# WHY REFUSE RATHER THAN CASCADE: rewriting the surviving references is a second
# guess stacked on the model's first, and the PSA case shows how bad that guess
# gets -- "phthalylsulfacetamide (PSA)", "phthalylsulfacetamide" and "PSA" are three
# spellings whose merge target is a judgement call about whether the parenthetical
# is an alias or part of the name. A refusal leaves the payload exactly as the gate
# last accepted it and lands verbatim in rejected_patch_log.json, so the next audit
# round can propose the synonym-add repair that audit_json_llm.py's own prompt
# already calls "the lowest-risk fix". The one cascade that IS provably safe needs
# no code: when the deleted row is a true duplicate, some surviving row still
# supplies the same normalized name, the coverage diff below comes back empty and
# the removal is accepted untouched. Duplicate cleanup keeps working; only the
# deletions that would strand a reference are refused.
# ---------------------------------------------------------------------------

# Exactly the buckets process_normalizer.validate_registry_references unions into
# its registry. Widening this set would make the guard disagree with the gate in
# the permissive direction; narrowing it would make it over-block.
#
# ``nucleic_acids`` was missing here after the gate grew it (process_normalizer.py
# :4143). The two sides disagreeing is not a cosmetic drift: coverage that does not
# count a bucket cannot register a loss in it, so `lost` came back empty for every
# nucleic-acid removal and the guard waved through deletions of rows that reactions
# name as inputs -- the exact orphaned reference the gate then aborts the export
# over. 'pmrHFIJKLM operon' is a reaction input of PMC13278307 and lives in this
# bucket.
_REGISTRY_ENTITY_BUCKETS = ("compounds", "proteins", "protein_complexes", "nucleic_acids")

# Stable, greppable prefix for the rejection reason, so batch tooling can count
# these the way it already counts the "attempted_to_*" lock reasons.
REFERENTIAL_INTEGRITY_REASON_PREFIX = "referential_integrity"

# Prefix stamped on applied-log entries that a batch rollback undid, so a reader
# grepping the rejected log can tell "this op was refused" from "this op was
# applied and then unwound with the rest of its set".
ROLLED_BACK_REASON_PREFIX = "rolled_back"

# How many orphaned references to name in the rejection reason before summarising.
# PMC12444477/research produced 24 of them from two entities; a reason string
# carrying all 24 is unreadable in a report and useless in a log line.
_MAX_ORPHANED_REFERENCES_IN_REASON = 5


def _is_entities_subtree_path(tokens: Sequence[str]) -> bool:
    return bool(tokens) and tokens[0] == "entities"


def _registry_coverage(payload: Dict[str, Any]) -> set[str]:
    """Every normalized name a process reference can currently resolve against.

    Uses process_normalizer._entity_name_norms so names AND declared synonyms both
    count, matching validate_registry_references exactly: a reaction that says
    "NAD" resolves against a compound named "NAD+" that lists "NAD" in synonyms
    (tests/test_process_normalizer.py::test_validate_registry_references_recognizes
    _declared_synonyms), and dropping that compound must therefore be refused even
    though no row is *named* "NAD".

    Deliberately does NOT call process_normalizer._entity_lists(): that helper does
    payload.setdefault("entities", {}) and back-fills the three bucket lists, i.e.
    it mutates. This function runs against the live `working` payload on every
    entity op, so it has to be strictly read-only or it would silently graft empty
    entity lists onto payloads that legitimately have none.
    """
    entities = _safe_dict(payload.get("entities"))
    norms: set[str] = set()
    for bucket in _REGISTRY_ENTITY_BUCKETS:
        norms |= _registry_name_norms(_safe_list(entities.get(bucket)))
    return norms


def _referenced_entity_norms(payload: Dict[str, Any]) -> Dict[str, str]:
    """Map normalized referenced name -> a human-readable "'name' at /pointer".

    This mirrors the traversal in process_normalizer.validate_registry_references
    site for site -- reaction inputs/outputs/enzymes/modifiers, transport
    cargo(_complex)/transporters, interaction entity_1/entity_2 -- including its
    quirks: it reads only processes.reactions (never a top-level payload["reactions"],
    unlike this module's _iter_reactions), it skips non-dict actor rows, and it
    prefers cargo_complex over cargo when the former is a non-blank string. Any
    site added here that the gate does not check would over-block a removal the
    gate would have been happy with; any site dropped would let the original bug
    back in through that door.
    """
    processes = _safe_dict(payload.get("processes"))
    found: Dict[str, str] = {}

    def record(name: Any, pointer: str) -> None:
        if not isinstance(name, str) or not _registry_canonical(name):
            return
        norm = _registry_normalize(name)
        if norm:
            found.setdefault(norm, f"{name!r} at {pointer}")

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        for side in ("inputs", "outputs"):
            for tidx, token in enumerate(_safe_list(reaction.get(side))):
                record(token, f"/processes/reactions/{ridx}/{side}/{tidx}")
        for actor_key in ("enzymes", "modifiers"):
            for eidx, actor in enumerate(_safe_list(reaction.get(actor_key))):
                if isinstance(actor, dict):
                    record(_registry_actor_name(actor), f"/processes/reactions/{ridx}/{actor_key}/{eidx}")

    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        cargo_complex = transport.get("cargo_complex")
        cargo = (
            cargo_complex
            if isinstance(cargo_complex, str) and _registry_canonical(cargo_complex)
            else transport.get("cargo")
        )
        record(cargo, f"/processes/transports/{tidx}/cargo")
        for tridx, transporter in enumerate(_safe_list(transport.get("transporters"))):
            if isinstance(transporter, dict):
                record(_registry_actor_name(transporter), f"/processes/transports/{tidx}/transporters/{tridx}")

    for iidx, interaction in enumerate(_safe_list(processes.get("interactions"))):
        if not isinstance(interaction, dict):
            continue
        left = interaction.get("left") or interaction.get("entity_1") or interaction.get("source")
        right = interaction.get("right") or interaction.get("entity_2") or interaction.get("target")
        record(left, f"/processes/interactions/{iidx}/entity_1")
        record(right, f"/processes/interactions/{iidx}/entity_2")

    return found


def _entity_names_introduced_by_op(op: Dict[str, Any]) -> set[str]:
    """Registry names an add/replace op under /entities would put back.

    Needed so the guard does not break the entity-type repair audit_json_llm.py's
    prompt explicitly asks for: "A small-molecule cofactor (PLP, ... NAD+, ...)
    listed under entities.proteins[] is a type error. Propose removing it from
    proteins[] and adding it to entities.compounds[]." That arrives as a two-op
    patch, and if the remove is evaluated in isolation it strands every reaction
    that names the cofactor as a modifier -- the guard would refuse a repair the
    pipeline wants. Ops that cannot clear their own confidence bar are excluded,
    because a compensating add that _should_accept will reject is not compensation.
    """
    action = str(op.get("op", "")).lower()
    if action not in {"add", "replace"}:
        return set()
    try:
        tokens = _decode_pointer(str(op.get("path", "")))
    except Exception:  # noqa: BLE001
        return set()
    if not _is_entities_subtree_path(tokens):
        return set()
    if _float_or_default(op.get("confidence"), 0.0) < _threshold_for_op(op):
        return set()

    value = op.get("value")
    norms: set[str] = set()
    if isinstance(value, dict):
        # A whole entity row: {"name": ..., "synonyms": [...]}.
        norms |= _registry_name_norms([value])
    elif isinstance(value, list):
        # Either a whole bucket of rows, or a replaced synonyms array of strings.
        norms |= _registry_name_norms(value)
        for item in value:
            if isinstance(item, str):
                norm = _registry_normalize(item)
                if norm:
                    norms.add(norm)
    elif isinstance(value, str):
        # A scalar leaf. Only /name and /synonyms/<i> carry registry weight; a
        # /class or /evidence string must not be mistaken for restored coverage.
        leaf_fields = {"name", "synonyms"}
        if tokens[-1] in leaf_fields or (len(tokens) >= 2 and tokens[-2] in leaf_fields):
            norm = _registry_normalize(value)
            if norm:
                norms.add(norm)
    return norms


def _pending_entity_name_norms(patch_ops: Sequence[Any]) -> List[set[str]]:
    """For each op index, the registry names introduced by ops STRICTLY AFTER it.

    Ops earlier in the batch need no look-ahead: their names are already present in
    the post-application payload the coverage diff measures.
    """
    per_op = [
        _entity_names_introduced_by_op(_normalize_patch_op(op)) if isinstance(op, dict) else set()
        for op in patch_ops
    ]
    later: List[set[str]] = [set() for _ in range(len(per_op) + 1)]
    for idx in range(len(per_op) - 1, -1, -1):
        later[idx] = per_op[idx] | later[idx + 1]
    return [later[idx + 1] for idx in range(len(per_op))]


def _referential_integrity_rejection(
    before_payload: Dict[str, Any],
    after_payload: Dict[str, Any],
    op: Dict[str, Any],
    pending_entity_names: set[str],
) -> Optional[str]:
    """Refuse an /entities op that would leave a process reference unresolvable.

    Returns a rejection reason, or None when the op is safe. The fast path is the
    common one: any op that does not shrink registry coverage (every add, every
    mapped_ids or class edit, and every true duplicate cleanup) returns after one
    set difference without ever walking the processes block.
    """
    try:
        tokens = _decode_pointer(str(op.get("path", "")))
    except Exception:  # noqa: BLE001
        return None
    if not _is_entities_subtree_path(tokens):
        return None

    lost = _registry_coverage(before_payload) - _registry_coverage(after_payload)
    lost -= pending_entity_names
    if not lost:
        return None

    referenced = _referenced_entity_norms(after_payload)
    orphaned = sorted(referenced[norm] for norm in lost if norm in referenced)
    if not orphaned:
        return None

    shown = "; ".join(orphaned[:_MAX_ORPHANED_REFERENCES_IN_REASON])
    if len(orphaned) > _MAX_ORPHANED_REFERENCES_IN_REASON:
        shown += f"; and {len(orphaned) - _MAX_ORPHANED_REFERENCES_IN_REASON} more"
    return (
        f"{REFERENTIAL_INTEGRITY_REASON_PREFIX}: {str(op.get('op', '')).lower()} on "
        f"{str(op.get('path', ''))} would orphan {len(orphaned)} process reference(s): {shown}."
    )


# ---------------------------------------------------------------------------
# Lineage emission.
#
# R-004 asked whether three reactions in a committed payload had been re-added
# by the audit and could not tell. Nothing in final.audited.json, the applied
# patch log or the apply report says which ROWS a patch produced: the logs record
# ops, and "/processes/reactions/2" is not a row -- every later insert or removal
# shifts that index, so a pointer read after the fact addresses a different
# reaction than the one it was written about.
#
# lineage.LineageEntry carries no entity or reaction id, so attribution is
# POSITIONAL: an entry is true of the row it sits on and of nothing else. That
# fixes where a record may be written -- onto the row itself, never onto a
# sibling, an index or a side table -- and it fixes what may be claimed: only
# what this stage actually knows.
#
# What this stage knows is one fact, and the rule below is exactly that fact:
# after an accepted op, a row whose content matches NO row that was in its
# container immediately before the op is content that operation produced. Hence
# a content diff over the container rather than a walk of the op's pointer. It
# needs no index arithmetic, so index drift cannot misattribute it; it is
# shape-agnostic, so a whole-row add, a wholesale array replace, an edit to a
# field nested inside a row and the mapping_meta stamp _apply_single_op writes
# all reduce to the same question; and it cannot claim a row the batch merely
# passed through, because such a row is identical to its own pre-image.
#
# It under-claims in one direction, deliberately: an op adding a row that
# duplicates one already present produces no new content, so nothing is
# recorded. A row asserting provenance it does not have is worse than a row with
# none, and silence is the honest reading of an ambiguous case.
#
# WHY NO SOURCE, AND WHY "unsupported": an op's ``evidence`` is free text from
# the auditing model and ``provenance`` is a free label. Neither is a PMCID, DOI,
# accession or file, and putting one in LineageSource.source_id would be
# accepting an identifier for its shape. With no source, "direct" and "indirect"
# are unavailable by construction, and "derived" would claim a deterministic
# derivation from supported content that a model-proposed repair is not.
#
# WHY paper_explicit IS "not_evaluated": this stage never asks whether the paper
# stated the content, only whether an op clears a confidence threshold and
# survives the lock and referential-integrity guards. "not_explicit" would be a
# finding it never made.
#
# WHY review_required IS False: the stage's verdict on this op is already
# recorded -- it accepted it. Demanding review for every audit-touched row would
# be a policy introduced as a side effect of instrumentation. The provisional
# status is stated where it is a fact: in ``support`` and ``uncertainty``.
#
# NOTHING HERE MAY CHANGE A VERDICT. Attribution is computed read-only, the
# records land after the op's own logging, and neither helper can raise into the
# patch loop. On rollback ``working`` is replaced by a fresh copy of the source
# payload, so a batch that changed nothing carries no records either.
# ---------------------------------------------------------------------------

_LINEAGE_UNCERTAINTY = (
    "audit repair is a model-proposed change accepted on a confidence threshold; "
    "this stage holds no resolvable source record for it"
)

_AUDIT_LINEAGE_ENTRIES: Dict[str, lineage.LineageEntry] = {
    action: lineage.LineageEntry(
        stage="audit_repair",
        origin="audit_modified",
        support="unsupported",
        paper_explicit="not_evaluated",
        reason=(
            f"an accepted audit patch '{action}' operation produced this row's content; "
            "it matched no row in this container immediately before the operation"
        ),
        review_required=False,
        uncertainty=_LINEAGE_UNCERTAINTY,
    )
    for action in ("add", "replace", "remove")
}


def _row_fingerprint(row: Any) -> Optional[str]:
    """A row's content as a comparable string, or None when it is not a row.

    LINEAGE_KEY is excluded so a row recorded earlier in the same batch still
    compares equal to its own pre-image. Including it would make every recorded
    row look new to every later op in the batch and stack duplicate records.
    """
    if not isinstance(row, dict):
        return None
    content = {key: value for key, value in row.items() if key != lineage.LINEAGE_KEY}
    try:
        return json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def _row_container_paths(payload: Dict[str, Any], tokens: Sequence[str]) -> List[Tuple[str, ...]]:
    """The row lists an op at ``tokens`` can rewrite.

    Rows live in lists: entities.<bucket>, processes.<kind>, and the top-level
    reactions[] that _iter_reactions() falls back to. Walking the pointer from
    the root and stopping at the FIRST list is what makes the owner the ROW and
    not one of its nested lists -- /processes/reactions/2/inputs/0 stops at
    processes.reactions, so the reaction owns an edit to its own inputs, which is
    the row a reader can attribute. A pointer that runs out on a dict (a
    whole-subtree op such as "replace /entities") owns every list under it.
    """
    current: Any = payload
    prefix: List[str] = []
    for token in tokens:
        if isinstance(current, list):
            return [tuple(prefix)]
        if not isinstance(current, dict) or token not in current:
            return []
        prefix.append(token)
        current = current[token]
    if isinstance(current, list):
        return [tuple(prefix)]
    if isinstance(current, dict):
        return [tuple(prefix + [key]) for key, value in current.items() if isinstance(value, list)]
    return []


def _rows_at(payload: Any, path: Tuple[str, ...]) -> List[Any]:
    current: Any = payload
    for token in path:
        current = _safe_dict(current).get(token)
    return _safe_list(current)


def _rows_written_by_op(
    before_payload: Dict[str, Any],
    after_payload: Dict[str, Any],
    op: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """The rows IN ``after_payload`` whose content this op produced.

    Read-only and total by construction: an audit trail entry is never a reason
    to fail an op the policy already accepted, so anything unexpected here yields
    "no attribution" rather than an exception into the patch loop.
    """
    try:
        tokens = _decode_pointer(str(op.get("path", "")))
        written: List[Dict[str, Any]] = []
        for container in _row_container_paths(after_payload, tokens):
            before_fingerprints = {
                fingerprint
                for fingerprint in (
                    _row_fingerprint(row) for row in _rows_at(before_payload, container)
                )
                if fingerprint is not None
            }
            for row in _rows_at(after_payload, container):
                fingerprint = _row_fingerprint(row)
                if fingerprint is not None and fingerprint not in before_fingerprints:
                    written.append(row)
        return written
    except Exception as exc:  # noqa: BLE001
        logger.warning("apply_patch_with_policy: lineage attribution skipped: %s", exc)
        return []


def _record_audit_lineage(rows: Sequence[Dict[str, Any]], action: str) -> None:
    """Append this stage's attribution to each row it produced."""
    entry = _AUDIT_LINEAGE_ENTRIES.get(action)
    if entry is None:
        return
    for row in rows:
        try:
            lineage.record(row, entry)
        except lineage.LineageError as exc:
            logger.warning("apply_patch_with_policy: lineage not recorded: %s", exc)


def committed_change_count(report: Any) -> int:
    """How many patch ops actually changed the payload the caller received.

    ``summary["accepted_count"]`` counts per-op verdicts and is the wrong number
    to report as "changes made": after a batch rollback it is non-zero while the
    payload is byte-identical to the input. Every consumer that reports progress,
    scores a candidate, or decides whether to re-run should ask this instead.

    Falls back to ``accepted_count`` only for reports written before
    ``transaction`` existed, so an archived report still reads sensibly.
    """

    if not isinstance(report, dict):
        return 0
    legacy = int(_safe_dict(report.get("summary")).get("accepted_count") or 0)
    transaction = report.get("transaction")
    if not isinstance(transaction, dict):
        return legacy
    value = transaction.get("applied_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return 0 if transaction.get("committed") is False else legacy


def _batch_rollback_reason(
    source_payload: Dict[str, Any],
    working: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    """Whether the accepted set, taken as a whole, left the payload inconsistent.

    The per-op guard cannot answer this. ``_referential_integrity_rejection``
    clears a removal on the strength of a *later* add that has not run yet, and
    until this check existed nothing ever confirmed the promise was kept: if that
    add is malformed, addresses an index that does not exist, or is rejected by
    its own confidence bar, the removal is already committed and the reference it
    stranded is a dangling name the Stage 3 gate aborts the export over. A patch
    is an operation *set*, so the set is what has to be valid -- an op that is
    individually safe and collectively destructive must not commit.

    Scoped, deliberately, to references orphaned by LOST COVERAGE -- a declaration
    the batch removed and did not restore. That is the same question the per-op
    guard asks, now asked once about the finished payload instead of once per op,
    and it is the whole of what a look-ahead can get wrong. A reference that
    dangles because a process op *introduced* it is a different defect with a
    different owner (the Stage 3 registry gate), and rolling back over it here
    would refuse the reactions-array replacements this module has always applied.

    Pre-existing dangling references are excluded for free: a name that was never
    in coverage cannot appear in the coverage that was lost.
    """

    lost_coverage = _registry_coverage(source_payload) - _registry_coverage(working)
    if not lost_coverage:
        return None, []
    new_orphans = sorted(
        where
        for norm, where in _referenced_entity_norms(working).items()
        if norm in lost_coverage
    )
    if not new_orphans:
        return None, []
    shown = "; ".join(new_orphans[:_MAX_ORPHANED_REFERENCES_IN_REASON])
    if len(new_orphans) > _MAX_ORPHANED_REFERENCES_IN_REASON:
        shown += f"; and {len(new_orphans) - _MAX_ORPHANED_REFERENCES_IN_REASON} more"
    return (
        f"{REFERENTIAL_INTEGRITY_REASON_PREFIX}: operation set rolled back; applying it "
        f"would orphan {len(new_orphans)} process reference(s): {shown}."
    ), new_orphans


# ---------------------------------------------------------------------------
# An audit patch may not invent an actor role it has no evidence for.
#
# WHY THIS EXISTS -- one failure, reproducible from artifacts on disk:
#
#   runs_verify/2026-08-28_1816/papers/PMC13231680/research/final_mapped.json
#   /processes/reactions/0 was frozen with the SAME protein listed both as the
#   reaction's enzyme and, on that same row, as its inhibitor. The reaction's own
#   evidence string names no actor at all, and the audit round that wrote it says
#   why in audit_iteration_summary.json /rounds/0/llm_repair_rationale: the enzyme
#   was added "to resolve the structural inconsistency where an inhibitor is listed
#   without a target enzyme". The row's own lineage carrier agrees with that
#   diagnosis -- support "unsupported", sources [] -- and was still written with
#   review_required false. PRODUCT_CONTRACT section 1 forbids exactly this: the
#   system must never invent enzymes, and satisfying a schema-shaped complaint is
#   the clearest available case of inventing one "merely to guarantee a PWML file".
#   The repair that WAS available -- drop the incoherent modifier, or flag the row
#   for review -- costs no biology. Promoting the inhibitor to catalyst asserts the
#   opposite of the paper's thesis.
#
# WHY THE HOLE WAS THERE: the same shape the referential-integrity comment above
# describes, on the actor-role axis. An add to /processes/reactions/N/enzymes/- is
# not connectivity (_is_connectivity_path matches only /inputs and /outputs), is
# not major topology (_is_major_topology_path stops at the container or an index),
# and is not a remove -- so it cleared on confidence >= 0.75 alone
# (_threshold_for_op) with nothing anywhere asking whether any evidence named the
# proposed actor as an actor.
#
# THE MATCHING RULE IS bench.semantic_production._actor_named_in_span's, AND IT IS
# REPRODUCED HERE RATHER THAN IMPORTED. An actor is named when its name and the
# span share at least one IDENTIFYING TOKEN (>= 3 characters, not all digits),
# compared on whole-token boundaries after folding. Not the whole name: the
# payload's canonical names are not the paper's words, and that function's own
# docstring records what the whole-name rule costs -- "ALAS2 complex" cites "ALAS2
# mediates ...", "oxidoreductase (entA)" cites "EntA (2,3-dihydro-...)", and a
# whole-name rule "demotes five of the 21 legs over that". An earlier draft of this
# guard implemented the whole-name rule while claiming agreement with that
# function; measured against the corpus it refused 12 of 29 legitimately evidenced
# cases and 150 corpus rows that the calibrated rule licenses. The duplication is
# deliberate: t2pw.bench is the evaluation layer and importing it from
# t2pw.curation would invert the layering this module's import comment (top of
# file) exists to protect. tests/test_c105_actor_role_evidence.py pins the two
# implementations against each other so they cannot drift apart silently.
#
# WHERE THIS GUARD IS DELIBERATELY STRICTER, AND WHY. _actor_named_in_span is a
# NAMING test joining a closed gating set, so it under-reports on purpose and
# returns "not examined" when a name yields no identifying token. This is an
# ADMISSION test on an unapplied patch, so it must return a verdict every time. Two
# differences follow, and neither contradicts that function:
#   * a name with no identifying token (a one- or two-character symbol, which that
#     function declines to judge) is matched here on the whole name, whole-token,
#     and refused when absent -- the closed direction;
#   * naming is necessary but not sufficient: a role cue must also sit within
#     _ACTOR_CUE_WINDOW characters of the matched token, because the defect above
#     is a protein the span DOES name, in the wrong role.
#
# WHY THIS PREDICATE READS THE OP AND NOTHING ELSE -- AND WHAT THAT IS WORTH.
# _should_accept is handed `source_payload`, so this restriction is SELF-IMPOSED,
# not structural: the payload was available and was not taken. The trade is
# coverage for a property a reviewer can check from the signature instead of
# having to trust the body. It is worth making because the measured defect is a
# protein LEGITIMATELY PRESENT on the reaction as an inhibitor being promoted to
# that reaction's catalyst, so a guard able to consult the payload can be argued
# into reading that presence as corroboration -- and would then accept the exact
# patch this guard exists to refuse. It also sidesteps the index staleness
# _should_accept already documents. The cost is stated in the next paragraph.
#
# WHAT THIS GUARD DOES NOT COVER. A cue near a name is a NECESSARY condition, never
# a sufficient one: it cannot prove the biology, only refuse a patch that offers no
# evidence of the role at all. Scoring the frozen graph is a different seam with an
# owner (bench.semantic_production._check_actor_evidence, F-079). Three residual
# routes to the same defect are open, and are named here because a reader is
# entitled to know them rather than discover them:
#   1. `replace /processes/reactions/N/modifiers/M/role` promotes an existing
#      inhibitor row in place. The actor name is not in the op value, so covering
#      it needs the payload read this predicate declines. A model refused on the
#      `add` path can simply take this one -- and the rejection reason names the
#      role, which tells it so.
#   2. An actor arriving inside a whole-reaction `add /processes/reactions/-`.
#      This is NOT protected territory: _is_major_topology_path only matters when
#      `enforce_major_topology_threshold` is True, which is passed at exactly one
#      call site (interactive_curator.py:507) and defaults False at every
#      automated caller, so such an add is accepted today.
#   3. A rename, `replace /processes/reactions/N/modifiers/M/entity`, which swaps
#      one actor identity for another inside an existing row. Deliberately out of
#      scope: pathway_curator's first documented job is repairing entity name
#      mismatches, and guarding a rename here would block it.
#
# WHY enzyme_cues.ENZYME_EVIDENCE_CUE_RE IS NOT REUSED AS THE CUE SET, though it is
# this repository's existing catalysis-cue predicate and sits in a leaf module this
# one could import. Its stems are (catalyz|catalys|catalytic|ENZYME|ENZYMATIC|
# mediated|dependent|ACTIVITY|activat|promot|facilitat), calibrated for a name-based
# attacher reading PAPER PROSE, where "enzyme" and "activity" are weak but honest
# signals. Some of the text judged here is not paper prose: _normalize_patch_op
# promotes a model's `reason` into `evidence` when no separate evidence field was
# supplied, so a patch JUSTIFICATION -- written in the payload's own schema
# vocabulary -- arrives as a span. Both strings behind the defect above carry
# "enzyme" within that module's 80-character window of the protein name, so reusing
# that regex would admit the very patch this guard refuses. The window discipline is
# borrowed deliberately; the bare schema nouns are not.
# ---------------------------------------------------------------------------

# Stable, greppable prefix, matching the convention
# REFERENTIAL_INTEGRITY_REASON_PREFIX established above, so batch tooling can count
# these the way it already counts the "attempted_to_*" lock reasons.
UNEVIDENCED_ACTOR_ROLE_REASON_PREFIX = "unevidenced_actor_role"

# The actor-role containers, on the three process buckets _is_core_semantics_path
# already treats as core. Matches the container itself and one element of it (an
# index or the "-" append token) -- the shapes that INTRODUCE an actor. It
# deliberately matches no deeper field: see residual route 3 above for the rename,
# and route 1 for the role flip.
_ACTOR_ROLE_PATH_RE = re.compile(
    r"^/processes/(?:reactions|transports|reaction_coupled_transports)/\d+"
    r"/(?P<container>enzymes|modifiers|modifiers_or_enzymes|catalysts|transporters|cargo|cargo_complex)"
    r"(?:/(?:\d+|-))?$"
)

# Containers whose rows ARE catalysts by construction, so no `role` field can make
# them anything else: process_normalizer migrates enzymes[] into modifiers[] with
# role "catalyst" and then rebuilds enzymes[] from the role=catalyst rows only
# (process_normalizer.py:3011-3062).
_CATALYST_ACTOR_CONTAINERS = ("enzymes", "catalysts", "modifiers_or_enzymes")
_TRANSPORT_ACTOR_CONTAINERS = ("transporters", "cargo", "cargo_complex")

# Keyed exactly as process_normalizer's exported role vocabulary spells it
# (`enzyme_export_roles`, process_normalizer.py:2751 -- "", catalyst, enzyme,
# activator, inhibitor). An unroled modifier is judged as a catalyst because that
# is what the normalizer makes it: it setdefaults role to "catalyst".
_ROLE_FAMILY_BY_ROLE = {
    "catalyst": "catalysis",
    "enzyme": "catalysis",
    "activator": "activation",
    "inhibitor": "inhibition",
    "repressor": "inhibition",
    "transporter": "transport",
    # C-107 1e. "cofactor" is the role a modifier row carries most often outside
    # the exported vocabulary above, and with no entry here _actor_role_family
    # returned "other", which sent _span_licenses_actor to the _ANY_ROLE_CUE_RE
    # fallback its own docstring calls "strictly more permissive than the four
    # families". Measured, the fallback still REFUSED every cofactor span,
    # because it is built from the four families' patterns and not one of them
    # contains a single cofactor-predicating word: no "cofactor", no "requires",
    # no "dependent on", no "in the presence of". A permissive fallback with no
    # vocabulary for the role has nothing to be permissive with.
    "cofactor": "cofactor",
}

# Enzyme-family noun suffixes, as an ALLOWLIST of real EC-class stems rather than a
# bare "-ase" pattern. "increase", "disease", "release" and "database" all end in
# "ase"; none of these stems does.
# Ordinary English words ending in "ase". The general rule below is bounded by a
# CLOSED stoplist rather than by trying to enumerate enzymology: an unlisted
# English word over-accepts a CUE, which costs little on its own because the actor
# name must independently match, whereas an unlisted ENZYME under-accepts and
# refuses a legitimate repair -- the failure REV-105 measured at 12 of 29 cases.
#
# C-107 1c. THE STOPLIST IS STILL CLOSED AND IT STAYS CLOSED. The fix below is to
# the RULE, not to the list: the entries are ordinary English, never enzymes, so
# the asymmetry the paragraph above states is preserved exactly. Replacing this
# with an allowlist of enzyme stems would invert it, and that inversion is how
# C-105's first draft refused 12 of 29 legitimate evidenced cases.
#
# Two defects, both measured against the shipped code:
#
#   (i)  a SINGULAR-ONLY entry was bypassed by its own plural. 13 of the 33
#        entries listed no plural, and five of those plurals matched as an enzyme
#        noun -- briefcases, pleases, purchases, showcases, staircases. The
#        exclusion is now written "<entry>s?" in _ENZYME_NOUN_RE_SRC below, so a
#        singular entry covers its plural by construction and no future entry can
#        reintroduce the bypass. Listing the five measured instances would have
#        fixed the instances and not the defect.
#   (ii) the list was short. The entries from "appease" down are the ordinary
#        English words measured to match the generic rule -- every one at least
#        three characters before "ase" and therefore reachable by it.
_NON_ENZYME_ASE_WORDS = (
    "disease", "diseases", "increase", "increases", "increased", "decrease",
    "decreases", "decreased", "release", "releases", "released", "database",
    "databases", "purchase", "phrase", "phrases", "chase", "erase", "cease",
    "ceases", "ceased", "lease", "please", "case", "cases", "base", "bases",
    "phase", "phases", "vase", "showcase", "staircase", "briefcase",
    "appease", "bookcase", "crease", "debase", "decease", "displease",
    "encase", "grease", "lowercase", "nutcase", "paraphrase", "pillowcase",
    "rebase", "rephrase", "suitcase", "surcease", "uppercase",
)
# Enzyme nouns with fewer than three characters before "ase". The generic rule
# below cannot reach them and dropping them was a REGRESSION against this card's
# own first commit, where "lyase" was an explicit stem: "P is the lyase for this
# step" was licensed at 28d8443 and refused after the generic rule replaced the
# stem list. DNase and RNase are lost the same way and for the same reason.
_SHORT_ENZYME_NOUNS = ("lyase", "lyases", "dnase", "dnases", "rnase", "rnases")

_ENZYME_NOUN_RE_SRC = (
    # The left boundary is load-bearing: without it the stoplist is bypassed by
    # starting the match one character in, and "database" matches as "atabase".
    # The trailing "s?" inside the exclusion is C-107 1c's general fix: a
    # singular-only stoplist entry covers its own plural, so "purchases" can no
    # longer walk through by failing the exclusion one character early.
    r"(?:(?<![a-z])(?!(?:" + "|".join(_NON_ENZYME_ASE_WORDS) + r")s?(?![a-z]))"
    r"[a-z]{3,}ases?(?![a-z]))"
    r"|(?:(?<![a-z])(?:" + "|".join(_SHORT_ENZYME_NOUNS) + r")(?![a-z]))"
)

# C-108 (b). THE GAP CONSTRUCTS BELOW ARE LENGTH BOUNDS AND NOTHING ELSE, AND
# THIS IS THE SPELLING THAT SAYS SO.
#
# They used to be written "[^.]{0,N}", which reads as "within one sentence" and
# is not: every pattern in this module is applied to the output of _match_fold,
# and _match_fold substitutes a single space for every run of characters outside
# [a-z0-9] BEFORE any pattern runs. NO "." EVER REACHES THE HAYSTACK, so "[^.]"
# excluded nothing. The alphabet of a folded string is exactly {a-z, 0-9, " "},
# which is what this class spells, so substituting it for "[^.]" cannot change a
# single verdict -- and the reader is no longer told a sentence bound exists.
#
# Two properties this rests on, both PROVEN rather than asserted, in
# evidence/c108_fold_equivalence.py and pinned in tests/test_c108_f155_class.py:
#
#   1. casefold() + NFKC + re.sub(r"[^a-z0-9]+", " ") admits no other character,
#      so "[^.]" and "[a-z0-9 ]" match the same set ON THIS INPUT.
#   2. "[^.]" also matches a newline, where an unflagged "." does not -- a real
#      difference between the two spellings on arbitrary text. Fold collapses
#      every newline to a space, so the difference is unreachable here.
#
# THIS MEMBER MOVES NO BEHAVIOUR. _match_fold is shared and calibrated and is
# NOT touched; no real sentence bound is introduced, which would be a behaviour
# change in the over-refusal direction and is not chartered.
_FOLDED_CHAR_SRC = r"[a-z0-9 ]"

# ---------------------------------------------------------------------------
# C-108 (d). THE INHIBITION VOCABULARY, SPELLED ONCE AND USED FOR TWO OPPOSITE
# JOBS.
#
# _CATALYSIS_CONTRA_RE was a bare alias of _ROLE_CUE_RES["inhibition"], so one
# object licensed an INHIBITOR row and refused a CATALYST row at the same time,
# while the comment above the alias already claimed the two were separate. C-105
# wrote these stems bare, and bare they match inside their own AGENT NOUNS:
# "inhibit" inside "inhibitor", "suppress" inside "suppressor", "repress" inside
# "repressor". In a CUE that costs nothing (it over-accepts, and the actor name
# must still match independently). In a CONTRA it is the opposite failure -- it
# over-REFUSES -- and F-155 (d) is its three measured instances:
#
#     REFUSED  "the repressor complex P4X catalyses the conversion of A to B"
#     REFUSED  "the suppressor protein P4X catalyses ..."
#     REFUSED  "the inhibitor protein P4X catalyses ..."
#
# In all three P4X IS the attenuator. That is not evidence P4X is being shut
# down, and the span says in so many words that it catalyses.
#
# MERGE RULE 6 BINDS HERE AND IT IS WHY THE FIX IS A SPLIT, NOT A DELETION. What
# is removed to stop "inhibit" matching inside "inhibitor" would be removed from
# the contra as well, and the contra is a biological gate. So the CUE below is
# unchanged -- byte for byte, which is what leaves _ANY_ROLE_CUE_RE and the
# "other" fallback for every unmapped role exactly where C-107 left them -- and
# only the CONTRA is anchored, with the agent nouns returned to it in the
# TARGET-DIRECTED frames built per needle in _span_licenses_actor.
#
# The grammatical distinction is WHO THE ATTENUATION IS AIMED AT, and English
# marks it by word order and by the preposition:
#
#     "the inhibitor P4X catalyses ..."   appositive  -- P4X IS the inhibitor
#     "the inhibitor of P4X was added"    PP complement -- P4X is the TARGET
#     "the P4X inhibitor was added"       head-final compound -- P4X is the TARGET
#
# which is the same question, and the same actor-anchored machinery, C-107
# correction round 1 built for the attenuation stems in F1/F2 below.
_C105_INHIBITION_STEMS_SRC = (
    r"inhibit|suppress|repress|downregulat|down regulat|blocks|blocked|blocking"
    r"|antagonis|inactivat|abolish|attenuat"
)

# The same stems, stopped from COMPLETING as their agent noun. The lookahead is
# deliberately "(?!ors?(?![a-z]))" and not a word anchor: a word anchor would
# also stop "inhibitory", "attenuating" and "repression", every one of which is
# inhibitory whatever its object and must keep firing the contra. Only the exact
# agent noun and its plural are withheld.
_C105_INHIBITION_STEMS_CONTRA_SRC = (
    r"inhibit(?!ors?(?![a-z]))|suppress(?!ors?(?![a-z]))|repress(?!ors?(?![a-z]))"
    r"|downregulat(?!ors?(?![a-z]))|down regulat(?!ors?(?![a-z]))"
    r"|blocks|blocked|blocking"
    r"|antagonis(?!ts?(?![a-z]))|inactivat(?!ors?(?![a-z]))"
    r"|abolish(?!ers?(?![a-z]))|attenuat(?!ors?(?![a-z]))"
)

# C-107 correction round 2's six additions, already anchored and completed on
# both sides. They are shared VERBATIM by the cue and the contra: nothing in this
# card touches them, and member (e) below pins the anchors that make them safe.
# "silencer" and "blocker" are deliberately NOT reachable through them, which is
# exactly what the right anchor buys, so neither appears in the agent-noun set.
_C107_INHIBITION_WORDS_SRC = (
    r"(?<![a-z])(?:blockades?|impair(?:s|ed|ing|ment|ments)?"
    r"|silenc(?:e|es|ed|ing)"
    r"|sequestr(?:ation|ations|ate|ates|ated|ating)"
    r"|ablat(?:e|es|ed|ing|ion|ions)"
    r"|interfer(?:e|es|ed|ing|ence))(?![a-z])"
)

# The agent nouns the contra above now withholds, returned to it ONLY in a frame
# that says the actor is what is being attenuated. Exactly the eight C-105 stems
# have one; "silencer" and "blocker" are absent for the reason stated above.
_ATTENUATION_AGENT_NOUN_SRC = (
    r"(?:inhibitors?|suppressors?|repressors?|downregulators?|antagonists?"
    r"|inactivators?|abolishers?|attenuators?)"
)

# What may stand between the actor and a head-final agent noun and still leave the
# actor the noun's TARGET. A CLOSED list of ADJECTIVES ONLY, and the omission of
# determiners is the whole point: "P4X specific inhibitor" is an inhibitor OF
# P4X, whereas "P4X, the inhibitor, catalyses ..." is an apposition saying P4X IS
# the inhibitor, and a determiner is what marks the difference.
_ATTENUATION_AGENT_ADJ_SRC = (
    r"(?:specific|selective|targeted|directed|potent|competitive|noncompetitive"
    r"|uncompetitive|reversible|irreversible|allosteric|small|molecule|peptide"
    r"|protein|antibody|chemical|synthetic|natural|classical|dependent|mediated"
    r"|activity|enzyme)"
)

#: How many such adjectives may stand between the actor and a head-final agent noun.
_ATTENUATION_AGENT_MAX_ADJ = 2

# What may head the phrase that names an agent noun's TARGET. The bare
# preposition is the core case ("the inhibitor OF P4X"), and the participial
# heads beside it are the paraphrases that survive dropping it -- "the inhibitor
# TARGETING P4X", "the inhibitor DIRECTED AGAINST P4X", "an inhibitor SELECTIVE
# FOR P4X". Every one was found by attacking this card's own fix rather than by
# design, and each was firing the contra at f67e00a through the unanchored stem,
# so omitting it would WEAKEN the gate: see evidence/c108_paraphrase_attack.log.
# A bare modifier is deliberately NOT a target head. "the inhibitor protein P4X"
# has no head at all, and that absence is what marks it as the appositive
# reading in which P4X IS the inhibitor.
_ATTENUATION_TARGET_HEAD_SRC = (
    r"(?:of|for|against|on|upon|to|toward|towards"
    r"|targeting|targeted at|inhibiting|blocking|suppressing|repressing"
    r"|acting on|acting against|directed against|directed at|aimed at"
    r"|specific for|selective for|specific to)"
)

# ---------------------------------------------------------------------------
# C-108 (a). THE TRANSPORT FAMILY'S AGENT NOUNS, AND THE ONLY CONSTRUCTIONS THAT
# MAKE ONE A CUE.
#
# The grammatical line, and the one the card asks for: the VERB and its EVENT
# nominalisation predicate something of the actor ("P transports A", "the
# transport of A is carried out by P"); the AGENT NOUN names the slot the patch
# is asking to fill. `_ACTOR_ROLE_PATH_RE` protects containers literally called
# `transporters`, `cargo` and `cargo_complex` and `_ROLE_FAMILY_BY_ROLE` spells
# the role `transporter`, so a rationale arguing from payload shape is written in
# exactly these words -- and `transport` matched inside `transporter`, so it
# licensed the row it was asking for. Measured at f67e00a, every one of
# transporter, importer, exporter, translocator, symporter, antiporter,
# uniporter, extruder, channel, carrier and pump did
# (evidence/c108_base_frames.log, section A).
#
# They come back only in a PREDICATION. "P is the transporter for A" licenses;
# "add P as a transporter" does not -- the same single line C-105 drew for
# "enzyme" and C-107 1e drew for "cofactor", generated here from the noun list
# rather than hand-written per noun, so a noun added later cannot arrive without
# its predicating forms.
_TRANSPORT_AGENT_NOUNS = (
    "transporter", "importer", "exporter", "translocator", "symporter",
    "antiporter", "uniporter", "extruder", "channel", "carrier", "pump",
)


def _agent_noun_predication_src(nouns: Sequence[str]) -> str:
    """The constructions in which a role-naming AGENT NOUN is role evidence.

    Two, and both are predications of the actor rather than namings of it:

        "<actor> is a|an|the <up to three modifiers> <noun>"
        "<noun>[s] for|of|through which|responsible for ..."

    "as a <noun>" is deliberately absent, and that absence is the whole fix:
    a promoted rationale says "add P as a transporter to resolve the structural
    inconsistency", never "P is the transporter for A".
    """

    alternatives: List[str] = []
    for noun in nouns:
        alternatives.append(
            r"(?:is|are|was|were|remains|remain) (?:a|an|the) "
            r"(?:[a-z0-9]+ ){0,3}" + noun + r"s?(?![a-z])"
        )
        # The copular-equivalent verbs. "P acts as a transporter" is the same
        # predication as "P is a transporter" and a paper writes both; "add P as
        # a transporter" is neither, and no payload-editing verb appears here.
        # Found by attacking this fix with paraphrases rather than by design --
        # see evidence/c108_paraphrase_attack.log.
        alternatives.append(
            r"(?:acts?|acted|acting|functions?|functioned|functioning"
            r"|serves?|served|serving|operates?|operated|operating"
            r"|works?|worked|working) as (?:a|an|the) "
            r"(?:[a-z0-9]+ ){0,3}" + noun + r"s?(?![a-z])"
        )
        alternatives.append(noun + r"s? (?:for|of|through which|responsible for)")
    return "|".join(alternatives)

# Role-predicating vocabulary, matched against the folded span (lower case, every
# separator run collapsed to a single space), so every pattern here is written in
# that spelling: "NDM-1-catalyzed" reaches this regex as "ndm 1 catalyzed" and
# "up-regulated" as "up regulated".
#
# The catalysis set is the vocabulary of enzyme-catalysed transformation: the verb
# "catalyse", the EC classes' reaction verbs, the enzyme-family nouns above, the
# passive-with-agent forms, and the periphrastic constructions a paper uses instead
# of a verb ("is the enzyme responsible for", "acts on", "breaks down"). Process
# words that name an event without predicating an agent ("decomposes", "forms",
# "occurs") are NOT cues: they describe something happening, not a protein doing
# it, and a reaction's own name is full of them. The bare schema nouns "enzyme",
# "enzymatic" and "activity" are not cues either, for the reason in the comment
# above -- a promoted rationale is written in exactly those words.
#
# C-108 (a). THE SCHEMA NOUN IS NOT A CUE, AND THAT RULE IS NOW ENFORCED RATHER
# THAN INTENDED. The paragraph above states it for "enzyme", "enzymatic" and
# "activity" -- and the pattern below then let "catalyst" through anyway, because
# the VERB stem "catalys" is a prefix of the schema NOUN "catalyst". C-105 wrote
# "is the catalyst", "catalyst responsible", "catalyst for" and "catalyst of this"
# as the periphrastic re-admissions of that noun, which is only sensible if the
# bare noun was meant to be excluded; those four alternatives have been DEAD CODE
# since they were written, because "catalys" matched "catalyst" first.
#
# The distinction is grammatical, not lexical: the verb and its event
# nominalisation predicate ("P catalyses", "the catalysis of A by P"); the AGENT
# NOUN names the slot the patch is asking to fill, and a rationale promoted out
# of `reason` by _normalize_patch_op is written in exactly the schema's words.
# So each stem is stopped from COMPLETING as its family's agent noun -- the same
# device C-107 correction round 2 used on the attenuation words, and narrower
# than a word anchor, which would also kill "inhibitory"-shaped inflections.
#
# Measured at f67e00a through the real seam, "add P as a <word> to resolve the
# structural inconsistency" was ACCEPTED for catalyst, catalysts and biocatalyst
# (evidence/c108_base_frames.log section A2) and for transporter (section A, and
# the battery's own C5 case). It is the words the CONTAINERS and the ROLE MAP are
# spelled in that a shape-argument rationale can be written in, which is why the
# fix covers exactly those and not the whole vocabulary -- see the report for the
# transport words measured and deliberately left.
_ROLE_CUE_RES = {
    "catalysis": re.compile(
        # C-108 (a): "catalys" may not complete as "catalyst"/"catalysts". It
        # still matches catalyse, catalysed, catalysing and catalysis, and the
        # noun returns below in the four predicating phrases C-105 already wrote.
        r"catalys(?!ts?(?![a-z]))|catalyz|catalytic|biocatalys(?!ts?(?![a-z]))"
        r"|hydroly|cleav|degrad|metabolis|metaboliz"
        # C-107 1f. "mediat" is a REQUIRED catalysis cue -- "<enzyme> mediates
        # the condensation ..." is _actor_named_in_span's own docstring example
        # and a legitimate repair -- but unanchored it also matches inside
        # "intermediate", a word pathway prose is full of, so "<protein> is an
        # intermediate carrier in this pathway" licensed that protein as the
        # reaction's CATALYST. The left boundary is written against the FOLDED
        # haystack, which holds only [a-z0-9 ], so a preceding letter is the
        # only thing that can precede a mid-word occurrence. "X-mediated" folds
        # to "x mediated" and still matches.
        r"|(?<![a-z])mediat"
        r"|oxidis|oxidiz|dehydrogenat|hydroxylat|oxygenat"
        r"|reduces|reducing|reduction of"
        r"|phosphorylat|methylat|acetylat|acylat|glycosylat|sulfonat|adenylat|prenylat"
        r"|transaminat|deaminat|decarboxylat|carboxylat|dehydrat"
        r"|isomeris|isomeriz|epimeris|epimeriz|racemis|racemiz"
        r"|ligat|synthesis|synthesiz"
        r"|converts|converting|conversion of"
        # periphrastic and agentive-noun constructions
        r"|breaks down|break down|broken down|breakdown of"
        r"|acts on|act on|acting on|acts upon"
        r"|is the enzyme|is the catalyst|enzyme responsible|catalyst responsible"
        r"|enzyme for|catalyst for|enzyme of this|catalyst of this"
        # C-108 (a). The predicating re-admissions of the agent nouns the anchor
        # above now withholds. "is a catalyst" licenses; "as a catalyst" does
        # not, which is the single line C-105 drew for "enzyme" and C-107 1e drew
        # for "cofactor", carried across a third time.
        r"|is a catalyst|is a biocatalyst|is the biocatalyst|biocatalyst for"
        r"|(?:removes|adds|attaches|transfers|incorporates)\b" + _FOLDED_CHAR_SRC + r"{0,40}"
        r"\b(?:group|residue|moiety|molecule|atom|phosphate|acyl|methyl|sugar)"
        r"|(?:removal|addition|transfer|incorporation) of\b" + _FOLDED_CHAR_SRC + r"{0,40}"
        r"\b(?:group|residue|moiety|molecule|atom|phosphate|acyl|methyl|sugar)"
        # C-107 1b. The passive-with-agent construction USED to live here, as
        # "<passive verb> [^.]{0,80} by", and it is gone from this pattern on
        # purpose. Written that way it fired whenever the verb was followed by
        # "by" within 80 characters REGARDLESS OF WHO FOLLOWED "by", so "A is
        # converted to B by Q, and P was also detected in the assay" licensed P
        # as the reaction's catalyst. The construction is evidence only when the
        # agent named after "by" IS the actor under judgement, and that is a
        # question about the actor, which a family-wide vocabulary cannot ask.
        # It is asked in _span_licenses_actor instead, against the needle, using
        # _PASSIVE_AGENT_VERBS_SRC and _PASSIVE_AGENT_MODIFIERS_SRC below.
        r"|" + _ENZYME_NOUN_RE_SRC
    ),
    "activation": re.compile(
        r"activat|stimulat|upregulat|up regulat|enhanc|induc|potentiat|agonis|promot"
    ),
    "inhibition": re.compile(
        # C-108 (d). The stems are spelled once, in _C105_INHIBITION_STEMS_SRC
        # above, and reach TWO patterns that do opposite jobs: this CUE, which
        # licenses an inhibitor row and is UNCHANGED at this card's tip, and the
        # catalysis CONTRA below, which refuses a catalyst row and is where the
        # agent-noun anchoring lives. Keeping the cue byte-identical is what
        # keeps _ANY_ROLE_CUE_RE -- and therefore the "other" fallback for every
        # unmapped role -- exactly where C-107 left it.
        _C105_INHIBITION_STEMS_SRC +
        # C-107 1a, the half of the near-synonyms that is inhibitory whatever its
        # object. These are safe as BARE stems: none of them describes a
        # transformation an enzyme performs, so adding them costs no catalysis
        # window. The other six near-synonyms are NOT here, and deliberately --
        # "reduction", "loss", "depletion", "disruption", "quenching" and
        # "blocking" all name things enzymes legitimately do to substrates, so
        # they are matched only in the activity-directed form built below.
        # CORRECTION ROUND 2. Anchored and completed on both sides, for the reason
        # given at _ATTENUATION_WORD_SRC below: written as bare stems, "silenc"
        # matched inside "silencer" and refused "the silencer complex <protein>
        # catalyses ...". Only these six are touched -- they are this card's
        # additions. The stems above them are C-105's and are left exactly as they
        # were merged.
        r"|" + _C107_INHIBITION_WORDS_SRC
    ),
    "transport": re.compile(
        # C-108 (a), F-155's own reproduction. "transport" may not complete as
        # "transporter"/"transporters" -- the ROLE the map spells and the
        # CONTAINER the guard protects are both called exactly that, so
        # "add P as a transporter to resolve the structural inconsistency" was a
        # rationale licensing the row it was asking for.
        # Each -er/-or stem is stopped from COMPLETING as its agent noun and
        # keeps every other inflection: transports, transported, transporting and
        # "the transport of A" all still match, which is why no stem is deleted.
        r"transport(?!ers?(?![a-z]))|translocat(?!ors?(?![a-z]))"
        r"|import(?!ers?(?![a-z]))|export(?!ers?(?![a-z]))"
        r"|symport(?!ers?(?![a-z]))|antiport(?!ers?(?![a-z]))"
        r"|uniport(?!ers?(?![a-z]))|extrud(?!ers?(?![a-z]))"
        r"|efflux|influx|uptake|secret|shuttl|permeas"
        # "channel", "carrier" and "pump" are agent nouns with no separate stem to
        # anchor -- the noun IS the string. The VERB readings that matched at
        # f67e00a are kept explicitly and the bare noun is dropped; "carries" is
        # NOT added, because it did not match at base and adding it would be a
        # widening this card is not chartered for.
        r"|channels|channelled|channeling|channelling|pumps|pumped|pumping"
        # C-108 (a). The predicating re-admissions, generated from
        # _TRANSPORT_AGENT_NOUNS so no noun can be listed without them.
        r"|" + _agent_noun_predication_src(_TRANSPORT_AGENT_NOUNS) +
        # C-107 1d. The enzyme-family noun rule was appended to the catalysis
        # pattern only, so a transporter named the way papers actually name one
        # -- "<protein> is the translocase of the inner membrane" -- refused,
        # while "permeas" sat in the list above as a one-off: the tell that the
        # general rule was missing rather than
        # deliberately withheld. It is the SAME rule, with the same closed
        # stoplist, so the asymmetry argued above holds here unchanged: an
        # over-accepted cue costs little because the actor name must still match
        # independently, an under-accepted transporter refuses a legitimate
        # repair. It admits enzyme-family nouns that are not transport nouns
        # ("P is a hydrolase" would cue a transporter row); that is the accepted
        # cost of a general rule over an allowlist of transporter families,
        # which is the inversion C-105 round 1 was rejected for.
        r"|" + _ENZYME_NOUN_RE_SRC
    ),
    # C-107 1e. Cofactor-predicating vocabulary, and it is written the way the
    # catalysis family writes "enzyme" -- the BARE SCHEMA NOUN IS NOT A CUE. A
    # rationale the audit stage promotes into `evidence` says "add P as a
    # cofactor to resolve the structural inconsistency", so "as a cofactor" is
    # deliberately absent while "is the cofactor", "cofactor for" and "cofactor
    # of" are present, exactly as "is the enzyme" is a cue above and "as an
    # enzyme" is not. That is the single line between licensing a real cofactor
    # and re-admitting F-146 in a fifth family.
    #
    # THIS IS NOT THE GATE THAT TYPES AN ENTITY. Licensing the ROLE lets an
    # evidenced cofactor modifier row survive this guard; whether the thing named
    # is a protein at all is decided elsewhere and is untouched here -- see the
    # note under _ROLE_FAMILY_BY_ROLE's "cofactor" entry and F-100.
    # CORRECTION ROUND 1. Only phrases that NAME A COFACTOR are family-wide cues.
    # The first version also listed bare "requires", "required for", "depends on",
    # "dependent on" and "in the presence of", and REV-107 measured what that
    # bought: "the reaction requires a cofactor, so <protein> is added to resolve
    # the structural inconsistency" and "<protein> depends on the payload
    # structure being consistent" went from REFUSED at base to ACCEPTED. Drawing
    # the line at the bare schema noun and then admitting the vocabulary a
    # structural rationale is actually WRITTEN IN is the F-146 class reopened in a
    # fifth family, and C-107 section 5's explicit stop condition.
    #
    # It was also wider than this role. _ANY_ROLE_CUE_RE is rebuilt from every
    # value of this dict, so those five terms widened the "other" fallback for
    # EVERY unmapped role: role "chaperone" with "<protein> is required for the
    # reaction to proceed" was refused at base and accepted at that tip. That is
    # the corollary of this card's own M6 mutation finding -- the vocabulary is
    # what leaks -- and it is why the dependence terms are now actor-anchored in
    # _span_licenses_actor and reach the cofactor family ONLY.
    "cofactor": re.compile(
        r"is a cofactor|is the cofactor|cofactor for|cofactor of|cofactor in this"
        r"|is a coenzyme|is the coenzyme|coenzyme for|coenzyme of"
        r"|prosthetic group"
    ),
}

# C-107 1b. The passive-with-agent cue, as an ACTOR-ANCHORED construction. These
# are the passive participles a paper uses to name the enzyme after "by"; most of
# them carry their own stem cue in the catalysis set above, so what this route
# genuinely adds is "converted", "produced" and "formed", which do not.
_PASSIVE_AGENT_VERBS_SRC = (
    r"(?:converted|catalyzed|catalysed|hydrolyzed|hydrolysed|cleaved|oxidized"
    r"|oxidised|phosphorylated|methylated|acetylated|degraded|metabolized"
    r"|metabolised|synthesized|synthesised|produced|formed)"
)

# What may stand between "by" and the actor's own token and still leave the actor
# the HEAD of the agent phrase: determiners, and the adjectives and appositive
# nouns papers put in front of a protein name. A CLOSED list, for the reason the
# stoplist above is closed -- an unlisted modifier costs one supplementary route
# on a span that usually still carries an ordinary stem cue, whereas an open gap
# is the defect itself. "by Q, and P ..." and "by Q; P ..." both fold to a token
# that is not on this list standing between "by" and P, so neither licenses P.
_PASSIVE_AGENT_MODIFIERS_SRC = (
    r"(?:the|a|an|its|their|his|her|this|that|these|those|of"
    r"|action|activity|activities|enzyme|enzymes|protein|proteins|complex"
    r"|purified|recombinant|native|endogenous|exogenous|bacterial|microbial"
    r"|human|host|intestinal|gut|mitochondrial|cytosolic|periplasmic|membrane"
    r"|bound|associated)"
)

#: How many such modifiers may stand between "by" and the actor's token.
_PASSIVE_AGENT_MAX_MODIFIERS = 4

# CORRECTION ROUND 1. The dependence vocabulary that USED to sit in the cofactor
# family as bare terms. It is inhibitory of nothing and predicating of nothing on
# its own -- "requires", "dependent on" and "in the presence of" are exactly the
# words a structural rationale is written in -- so it is only evidence when the
# thing required IS THE ACTOR. Anchored that way in _span_licenses_actor:
#
#     "the reaction requires <NEEDLE> as a cofactor"        licenses
#     "the enzyme is dependent on <NEEDLE> for activity"    licenses
#     "proceeds only in the presence of <NEEDLE>"           licenses
#     "the reaction requires a cofactor, so <NEEDLE> is     refuses -- "so" is not
#      added to resolve the structural inconsistency"        a modifier, and the
#                                                            actor is not what is
#                                                            required
#     "<NEEDLE> depends on the payload structure"           refuses -- the actor
#                                                            is the SUBJECT here,
#                                                            not the object
_COFACTOR_DEPENDENCE_SRC = (
    r"(?:requires|requiring|required|requirement"
    r"|depends|depend|dependent|dependence"
    r"|in the presence)"
)

# The modifiers that may stand between the dependence term and the actor's token.
# The passive-agent list plus the nouns a paper puts in front of a cofactor name.
_COFACTOR_MODIFIERS_SRC = (
    _PASSIVE_AGENT_MODIFIERS_SRC[:-1]
    + r"|cofactor|cofactors|coenzyme|coenzymes|metal|divalent|ion|ions"
    + r"|essential|catalytic|added|exogenous|free)"
)

#: How many such modifiers may stand between the dependence term and the actor.
_COFACTOR_MAX_MODIFIERS = 4

# Used only for a declared role outside the exported vocabulary above (a "cofactor"
# modifier, say). The actor must still be NAMED in a span that says something
# role-predicating about it; this guard simply cannot narrow WHICH role, so it does
# not pretend to. Strictly more permissive than the four families and never more
# permissive than the base behaviour, which asked for nothing at all.
_ANY_ROLE_CUE_RE = re.compile(
    "|".join(pattern.pattern for pattern in _ROLE_CUE_RES.values())
)

# C-107 1a. THE DISTINCTION THIS CARD TURNS ON: chemical reduction OF A SUBSTRATE
# is catalysis; reduction OF AN ACTIVITY, A LEVEL OR AN EXPRESSION is inhibition.
#
# "reduces|reducing|reduction of" is a catalysis cue in the set above and it stays
# one, because redox chemistry is half of enzymology -- "NADH-dependent reduction
# of the substrate", "ferrochelatase reduces ...". Deleting it to close the
# paraphrase would break redox to fix a rephrase, which is the wrong trade and the
# trap this finding carries. But left alone it made "the reduction of <protein>
# activity by <inhibitor>" read as a catalysis cue WITH NO CONTRA, which is F-146
# one rephrase away, and the audit stage regenerates its rationale every round.
#
# So the attenuation words are matched as a PHRASE and never as bare words. WHAT
# BINDS THE PHRASE IS THE ACTOR, NOT AN ADJACENT NOUN -- and that is correction
# round 1's repair. The first version bound the stem to an
# activity/level/expression noun within 40 characters, which closed exactly the
# frame the card quoted and nothing else: REV-107 measured 15 of 44 still open,
# because five of the eleven stems went in conditionally and every one of them
# survived
#
#     "the reduction of <protein> is mediated by ..."        object ABSENT
#     "<protein> activity showed reduction in the ... assay"  object BEFORE the stem
#     "the reduction of <protein> <45 chars> activity ..."    object BEYOND 40 chars
#
# A fix that is itself one rephrase from defeat has not closed a route whose whole
# thesis is that the audit stage regenerates its rationale every round.
#
# The discriminator that survives rephrasing is GRAMMATICAL, not lexical: chemical
# reduction acts on A SUBSTRATE, inhibition acts on THE ACTOR. So the contra now
# asks whether the attenuation stem's object IS the actor under judgement, which
# is a question only _span_licenses_actor can ask, because only it knows the
# needle. Two frames, built per needle:
#
#   F1  stem [of|in] <modifiers> NEEDLE     "the reduction of P", "reduction of
#                                            P activity", "quenching of the P signal"
#   F2  NEEDLE ... <object> ... stem        "P activity showed reduction"
#
# F1 makes the object IRRELEVANT, which is what closes the absent and beyond-40
# frames at once; F2 covers the object-first ordering. Redox survives both because
# its object is a substrate, not the actor: "reduction of the substrate by P" puts
# "substrate" -- not a modifier -- between "of" and the needle, and
# "ferrochelatase reduces the cellular level of protoporphyrin" names the level
# AFTER the stem, which F2 does not match.
#
# Making it actor-bound also retired the B2 residual REV-107 raised: "level" and
# "function" no longer fire on the substrate's level or the enzyme's function,
# because those are not the actor's.
# CORRECTION ROUND 2. THESE ARE WHOLE WORDS, NOT STEMS, AND BOTH SIDES ARE
# ANCHORED. The first version wrote them as stems and let "[a-z]*" run to the end
# of whatever word they started, which is finding 1f's own defect class -- the one
# this card spent a whole finding anchoring "mediat" for -- reintroduced inside the
# contra written to fix 1a, and in the OVER-REFUSAL direction that got C-105 round
# 1 rejected. "reduc" matched inside "reductase", "block" inside "blocker",
# "silenc" inside "silencer" and "interfer" inside "interferon", so a legitimate
# evidenced catalysis span naming an EC class 1 enzyme -- whose members are
# literally called reductase -- was REFUSED: 8 of 10 measured, against 0 at the
# C-106 base.
#
# A left anchor alone does NOT fix it, and that was tested rather than assumed:
# it repairs "nitroreductase" and "oxidoreductase", where the stem starts mid-word,
# and leaves bare "reductase", "blocker", "silencer" and "interferon" broken,
# because there the stem STARTS a genuine word and "[a-z]*" eats the rest of it.
# The stem has to COMPLETE as an attenuation word, which is what the explicit
# inflections below do. Ordering inside the group is safe: a shorter alternative
# that matches leaves the trailing (?![a-z]) to fail, and the engine backtracks to
# the longer one -- "blocking" is not matched as "block".
_ATTENUATION_WORD_SRC = (
    r"(?<![a-z])(?:"
    r"reduce|reduces|reduced|reducing|reduction|reductions"
    r"|loss|losses"
    r"|deplete|depletes|depleted|depleting|depletion|depletions"
    r"|disrupt|disrupts|disrupted|disrupting|disruption|disruptions"
    r"|quench|quenches|quenched|quenching"
    r"|blockade|blockades|block|blocks|blocked|blocking"
    r"|impair|impairs|impaired|impairing|impairment|impairments"
    r"|silence|silences|silenced|silencing"
    r"|sequester|sequesters|sequestered|sequestering|sequestration|sequestrations"
    r"|ablate|ablates|ablated|ablating|ablation|ablations"
    r"|interfere|interferes|interfered|interfering|interference"
    r")(?![a-z])"
)
_ATTENUATION_OBJECT_SRC = r"(?:activit|express|level|abundance|function)"

#: How far F2 may look from the actor to its object, and from the object to the stem.
_ATTENUATION_GAP = 40

# What refuses a catalysis window that also says the protein is being shut down.
# Separate from _ROLE_CUE_RES["inhibition"] because the two do different jobs:
# that one licenses an INHIBITOR row, this one refuses a CATALYST row, and 1a's
# near-synonyms belong in the second without all of them belonging in the first.
# This half carries the six stems that are inhibitory whatever their object; the
# five that are not are built per-needle inside _span_licenses_actor, the same
# way the passive-agent route is, because both are questions about the actor.
# C-108 (d). NO LONGER AN ALIAS. The comment above has claimed since C-105 that
# the cue and the contra do different jobs; until this card they were the same
# object, and the sentence was aspiration rather than description. They are now
# two patterns, composed from the same sources, differing in exactly one respect:
# the contra will not let a C-105 stem complete as its own AGENT NOUN. The agent
# nouns come back to the contra, target-directed and actor-anchored, as frames F3
# and F4 in _span_licenses_actor -- so nothing the contra refused with the actor
# as the TARGET is admitted here, and only the appositive reading moves.
_CATALYSIS_CONTRA_RE = re.compile(
    _C105_INHIBITION_STEMS_CONTRA_SRC + r"|" + _C107_INHIBITION_WORDS_SRC
)

# The +/-80 character window enzyme_cues.cue_near_name scans around a name,
# measured here from the MATCHED TOKEN rather than from the whole name, since the
# rule above matches on one shared token.
_ACTOR_CUE_WINDOW = 80

# bench.semantic_production._MIN_IDENTIFYING_TOKEN. A token shorter than this
# cannot identify a protein on its own ("of", "n", "a" fall out of a canonical name
# like "UDP-N-acetylglucosamine acyltransferase").
_MIN_IDENTIFYING_TOKEN = 3

# The evidence-bearing fields of an actor row, spelled as audit_json_llm._evidence_strings
# spells them, so a row that carries its own quote is read the same way on both sides.
_ACTOR_ROW_EVIDENCE_KEYS = ("evidence", "evidence_quote", "source_evidence", "source_text")

_MATCH_FOLD_RE = re.compile(r"[^a-z0-9]+")


def _match_fold(value: Any) -> str:
    """bench.goldset.normalize_name's folding, for the token comparison above.

    The one property that matters and that _registry_normalize does NOT have:
    every run of non-alphanumerics becomes a SINGLE SPACE rather than being
    deleted. Deleting them welds words together -- "NDM-1-catalyzed hydrolysis"
    folds to "ndm1catalyzedhydrolysis" under the registry rule, where no token
    boundary can be found and both the name and the cue become invisible. Here it
    folds to "ndm 1 catalyzed hydrolysis".
    """

    text = unicodedata.normalize("NFKC", str(value if value is not None else "")).casefold()
    return _MATCH_FOLD_RE.sub(" ", text).strip()


def _identifying_match_tokens(name: str) -> List[str]:
    """The tokens of ``name`` that can identify an actor, folded.

    bench.semantic_production._identifying_tokens, reproduced: at least
    _MIN_IDENTIFYING_TOKEN characters and not a bare number, so "UDP-N-
    acetylglucosamine acyltransferase" identifies on {udp, acetylglucosamine,
    acyltransferase} and the stray "n" is dropped.
    """

    return [
        token for token in _match_fold(name).split(" ")
        if len(token) >= _MIN_IDENTIFYING_TOKEN and not token.isdigit()
    ]


_FOLDED_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: How many tokens before a name occurrence may hold the agentive "by" and still
#: mark that occurrence as an AGENT rather than a naming. Three, which reaches
#: "by <symbol> <name>" and "by the <symbol> <name>" and stops well short of the
#: next clause.
_AGENTIVE_BY_LOOKBACK = 3


def _mask_actor_name(window: str, actor: str) -> str:
    """Blank out the actor's OWN NAME where it stands in ``window``, for C-108 (c).

    A NAME IS NOT A CLAIM. The cue is sought in a window around the matched name
    token, so an actor whose name IS an enzyme noun supplied its own cue and
    cleared the guard on nothing else:

        actor "LpxC hydrolase" · "LpxC hydrolase was quantified in the lysate"

    which says the protein was MEASURED. Any actor named "... synthase",
    "... transferase" or "... hydrolase" walked through the same door, and the
    name is written by the same patch the guard is judging -- so the patch was
    supplying both halves of its own evidence.

    WHAT IS MASKED, AND WHY IT IS NOT THE NAME'S TOKENS. Only a run of ADJACENT
    tokens that are ALL tokens of this actor's name is blanked, and only when the
    run is at least two tokens long or is the whole of a single-token name. That
    is the difference between APPOSITION and PREDICATION, and it is what saves the
    over-refusal trap the card names explicitly:

        actor "inner membrane translocase"
        span  "P is the translocase of the inner membrane"

    Masking the name's TOKENS would delete "translocase" and kill the only cue in
    the span. Masking contiguous RUNS deletes "inner membrane" -- which is
    naming -- and leaves "translocase", which is the predication. Measured, not
    reasoned: see evidence/c108_tip_frames.log section C2.

    Replacement is space-for-character, so every offset and every word boundary
    in the window survives and no pattern can match across a masked region.

    THE AGENTIVE "BY" IS AN EXEMPTION, AND IT WAS FOUND BY MEASUREMENT. An
    occurrence standing inside an agent phrase -- "... is converted to 2,3-diDHB
    and pyruvate BY EntB isochorismatase activity" -- is not a naming: "by" marks
    the phrase it heads as the AGENT, which is a predication of exactly the role
    under judgement. Three corpus rows said so before this exemption existed
    (evidence/c108_corpus_diff_r1.log, the `isochorismatase` rows), and they are
    legitimate catalysis spans, so refusing them would have broken pinned
    property P3 to close member (c). The exemption can never admit anything the
    base refused: it only declines to mask, and an unmasked window is exactly the
    base's window.

    THE CONTRA NEVER SEES THIS. Only the positive cue search is masked. Refusing
    is a biological gate and a masked window would be a weaker one, so the contra,
    the actor-anchored attenuation frames and the passive-agent route are all
    judged on the untouched text.
    """

    folded = _match_fold(actor)
    if not folded:
        return window
    name_tokens = set(folded.split(" "))
    single = folded if " " not in folded else None
    spans = [(m.start(), m.end(), m.group(0))
             for m in _FOLDED_TOKEN_RE.finditer(window)]
    out = list(window)
    run: List[Tuple[int, int, str]] = []

    tokens_only = [tok for _s, _e, tok in spans]
    index_of = {start: i for i, (start, _e, _t) in enumerate(spans)}

    def flush(run: List[Tuple[int, int, str]]) -> None:
        if not run:
            return
        if len(run) < 2 and not (single is not None and run[0][2] == single):
            return
        first = index_of[run[0][0]]
        if "by" in tokens_only[max(0, first - _AGENTIVE_BY_LOOKBACK):first]:
            return
        for start, end, _tok in run:
            for i in range(start, end):
                out[i] = " "

    previous_end = -1
    for start, end, token in spans:
        if token in name_tokens and (not run or start == previous_end + 1):
            run.append((start, end, token))
        else:
            flush(run)
            run = [(start, end, token)] if token in name_tokens else []
        previous_end = end
    flush(run)
    return "".join(out)


def _actor_role_target(path: str) -> str:
    """The actor-role container this path addresses, or "" if it addresses none."""
    match = _ACTOR_ROLE_PATH_RE.match(path)
    return match.group("container") if match else ""


def _proposed_actor_names(value: Any) -> List[str]:
    """Every actor name the patch value would introduce, in payload spelling.

    Reuses process_normalizer._actor_name_from_row, so a bare string, a
    ``{"entity": ...}`` row and a ``{"protein_complex": ...}`` row all resolve to a
    name the same way the registry gate resolves them. A value that names no actor
    -- ``[]``, ``None``, a number -- returns empty, and the guard then has nothing
    to license: emptying or clearing a role introduces no actor.
    """

    names: List[str] = []
    if isinstance(value, list):
        for item in value:
            names.extend(_proposed_actor_names(item))
        return names
    if isinstance(value, (str, dict)):
        name = _registry_actor_name(value)
        if name:
            names.append(name)
    return names


def _actor_role_family(container: str, value: Any) -> str:
    if container in _TRANSPORT_ACTOR_CONTAINERS:
        return "transport"
    if container in _CATALYST_ACTOR_CONTAINERS:
        return "catalysis"
    role = _registry_normalize(str(value.get("role", ""))) if isinstance(value, dict) else ""
    if not role:
        return "catalysis"
    return _ROLE_FAMILY_BY_ROLE.get(role, "other")


def _patch_evidence_spans(op: Dict[str, Any], value: Any) -> List[str]:
    """The spans this patch offers, and nothing else.

    The op's own ``evidence`` field plus, when the proposed actor row carries its
    own quote, that row's evidence fields. Nothing is read from the payload: see the
    comment above for why that restriction is taken and what it costs.

    ``_normalize_patch_op`` promotes ``reason`` into ``evidence`` when the model
    supplied no ``evidence`` of its own, so a bare rationale does arrive here as a
    span. It needs no special case: it is judged on its own text, and a rationale
    that argues from the payload's shape names no actor performing a role.
    """

    spans: List[str] = []
    candidate = op.get("evidence")
    if isinstance(candidate, str) and candidate.strip():
        spans.append(candidate)
    if isinstance(value, dict):
        for key in _ACTOR_ROW_EVIDENCE_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip() and candidate not in spans:
                spans.append(candidate)
    return spans


def _span_licenses_actor(span: str, actor: str, family: str) -> bool:
    """Does this span name ``actor`` performing a ``family`` role?

    Naming follows _actor_named_in_span: one shared identifying token, whole-token
    boundaries, after :func:`_match_fold`. Where the name yields no identifying
    token -- the case that function declines to judge -- the whole folded name is
    required instead, so a one- or two-character symbol is still located rather than
    waved through. A name that folds away entirely cannot be located at all, and the
    span does not license it: the patch is refused and the payload is left exactly
    as the gate last accepted it.

    Then the role test: a cue for ``family`` within :data:`_ACTOR_CUE_WINDOW`
    characters of the matched token. That is what separates this from a naming
    check -- the defect this guard exists for is a protein its span DOES name, in
    the wrong role.

    THE CATALYSIS FAMILY ADDITIONALLY REFUSES A WINDOW THAT ALSO CARRIES AN
    INHIBITION CUE, and this is not belt-and-braces -- it closes a paraphrase route
    straight back to the defect. ``mediat`` has to be a catalysis cue: "ALAS2
    mediates the condensation ..." is _actor_named_in_span's own docstring example
    and a legitimate repair. But it also makes "PSA-mediated inhibition of NDM-1
    activity" and "the inhibition of NDM-1 is mediated by PSA" read as catalysis,
    so a span stating the protein is INHIBITED would license it as the reaction's
    CATALYST -- the exact promotion this card exists to prevent, one rephrase away,
    and the audit stage regenerates its rationale every round. Refusing the window
    rather than the whole span keeps a long span that discusses inhibition in one
    clause and catalysis in another: scanning continues at the next occurrence.
    """

    haystack = _match_fold(span)
    if not haystack:
        return False
    needles = _identifying_match_tokens(actor)
    if not needles:
        whole = _match_fold(actor)
        needles = [whole] if whole else []
    if not needles:
        return False
    cue = _ROLE_CUE_RES.get(family, _ANY_ROLE_CUE_RE)
    contra = _CATALYSIS_CONTRA_RE if family == "catalysis" else None
    for needle in needles:
        escaped = re.escape(needle)
        # CORRECTION ROUND 1. The other half of the catalysis contra: an
        # attenuation stem whose OBJECT IS THIS ACTOR. F1 makes the
        # activity/level/expression noun irrelevant, which is what closes the
        # object-absent and object-beyond-40-characters rephrasings together; F2
        # covers the object-first ordering. Redox is untouched by both, because
        # its object is a substrate and never the actor.
        actor_contra = None
        if family == "catalysis":
            actor_contra = re.compile(
                # F1: "<stem> [of|in] <modifiers> <actor>"
                _ATTENUATION_WORD_SRC + r"(?:\s+(?:of|in))?"
                r"(?:\s+" + _PASSIVE_AGENT_MODIFIERS_SRC + r"){0,4}\s+"
                + escaped + r"(?![a-z0-9])"
                # F2: "<actor> ... <activity noun> ... <stem>"
                r"|(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
                + _FOLDED_CHAR_SRC + r"{0," + str(_ATTENUATION_GAP) + r"}?\b"
                + _ATTENUATION_OBJECT_SRC + r"[a-z]*\b"
                + _FOLDED_CHAR_SRC + r"{0," + str(_ATTENUATION_GAP) + r"}?\b"
                + _ATTENUATION_WORD_SRC +
                # C-108 (d), F3: "<agent noun> of|for|against|... <modifiers>
                # <actor>". The attenuation is aimed AT the actor, so the contra
                # fires -- "the inhibitor of P4X was added", "an inhibitor of the
                # enzyme P4X". This is the half of the agent noun the anchored
                # contra above withholds, given back target-directed.
                r"|" + _ATTENUATION_AGENT_NOUN_SRC
                + r"\s+" + _ATTENUATION_TARGET_HEAD_SRC
                + r"(?:\s+" + _PASSIVE_AGENT_MODIFIERS_SRC + r"){0,4}\s+"
                + escaped + r"(?![a-z0-9])"
                # C-108 (d), F4: "<actor> <adjectives> <agent noun>". The
                # head-final compound -- "the P4X inhibitor", "the P4X specific
                # inhibitor" -- where the actor is again the TARGET. No
                # determiner may intervene, which is what keeps the apposition
                # "P4X, the inhibitor, catalyses ..." out of this frame.
                r"|(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
                r"(?:\s+" + _ATTENUATION_AGENT_ADJ_SRC + r"){0,"
                + str(_ATTENUATION_AGENT_MAX_ADJ) + r"}\s+"
                + _ATTENUATION_AGENT_NOUN_SRC + r"(?![a-z])"
            )
        # CORRECTION ROUND 1. The cofactor dependence route, anchored the same way
        # and reaching the cofactor family ONLY, so it can never widen the "other"
        # fallback the way the bare terms did.
        dependence = None
        if family == "cofactor":
            dependence = re.compile(
                _COFACTOR_DEPENDENCE_SRC + r"\b(?:\s+(?:on|of|upon|for))?"
                r"(?:\s+" + _COFACTOR_MODIFIERS_SRC + r"){0,"
                + str(_COFACTOR_MAX_MODIFIERS) + r"}\s+"
                + escaped + r"(?![a-z0-9])"
            )
        for match in re.finditer(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", haystack):
            start = max(0, match.start() - _ACTOR_CUE_WINDOW)
            end = min(len(haystack), match.end() + _ACTOR_CUE_WINDOW)
            window = haystack[start:end]
            # C-108 (c). The POSITIVE search runs on a window with the actor's
            # own name blanked out, because a name is not a claim: an actor
            # called "LpxC hydrolase" was clearing this guard on its own
            # spelling. The contra searches below deliberately keep the
            # UNMASKED window -- masking a refusal would weaken a biological
            # gate, and merge rule 6 forbids that.
            cue_window = _mask_actor_name(window, actor)
            # The cofactor dependence route is ALREADY actor-anchored -- its
            # pattern ends on the needle -- so it can never be satisfied by the
            # actor's own name and it reads the UNMASKED window. Masking it would
            # delete the needle the frame is built around and refuse every
            # evidenced cofactor row: measured, 3 of 7 in the battery's C5
            # section (evidence/c108_tip_battery_r1.log), which is how this was
            # caught rather than reasoned about.
            licensed = cue.search(cue_window) or (
                dependence is not None and dependence.search(window)
            )
            if not licensed:
                continue
            if contra is not None and contra.search(window):
                continue
            if actor_contra is not None and actor_contra.search(window):
                continue
            return True
        if family != "catalysis":
            continue
        # C-107 1b, the passive-with-agent route, asked of THIS actor rather than
        # of the window. The match is anchored on the needle as the agent, so an
        # agent who is somebody else cannot license it, and the contra-cue is
        # still judged on the ordinary window around the agent so the paraphrase
        # route 1a closes is not reopened through this door.
        passive = (
            _PASSIVE_AGENT_VERBS_SRC
            + r"\b" + _FOLDED_CHAR_SRC + r"{0,80}\bby(?:\s+" + _PASSIVE_AGENT_MODIFIERS_SRC
            + r"){0," + str(_PASSIVE_AGENT_MAX_MODIFIERS) + r"}\s+"
            + escaped + r"(?![a-z0-9])"
        )
        for match in re.finditer(passive, haystack):
            start = max(0, match.end() - len(needle) - _ACTOR_CUE_WINDOW)
            end = min(len(haystack), match.end() + _ACTOR_CUE_WINDOW)
            window = haystack[start:end]
            if contra is not None and contra.search(window):
                continue
            if actor_contra is not None and actor_contra.search(window):
                continue
            return True
    return False


def _unevidenced_actor_role_rejection(op: Dict[str, Any]) -> Optional[str]:
    """Why this patch may not introduce the actor role it proposes, or ``None``.

    Pure in ``op`` -- self-imposed, see the block comment above -- so "the entity is
    already in this reaction" is not merely rejected as evidence, it is unreachable
    as evidence.
    """

    action = str(op.get("op", "")).lower()
    if action not in {"add", "replace"}:
        return None
    container = _actor_role_target(str(op.get("path", "")))
    if not container:
        return None
    value = op.get("value")
    names = _proposed_actor_names(value)
    if not names:
        return None
    spans = _patch_evidence_spans(op, value)
    listed = ", ".join(f"'{name}'" for name in names)
    if not spans:
        return (
            f"{UNEVIDENCED_ACTOR_ROLE_REASON_PREFIX}: {action} of {listed} to "
            f"{container} carries no evidence span."
        )
    family = _actor_role_family(container, value)
    unlicensed = [
        name for name in names
        if not any(_span_licenses_actor(span, name, family) for span in spans)
    ]
    if not unlicensed:
        return None
    listed = ", ".join(f"'{name}'" for name in unlicensed)
    return (
        f"{UNEVIDENCED_ACTOR_ROLE_REASON_PREFIX}: no evidence span names {listed} "
        f"performing the {family} role of {container}; an actor role may not be "
        f"added to satisfy payload structure."
    )


def _should_accept(
    op: Dict[str, Any],
    source_payload: Dict[str, Any],
    *,
    connectivity_confidence_threshold: float = DEFAULT_CONNECTIVITY_CONFIDENCE_THRESHOLD,
    major_topology_confidence_threshold: float = DEFAULT_MAJOR_TOPOLOGY_CONFIDENCE_THRESHOLD,
    enforce_major_topology_threshold: bool = False,
) -> Tuple[bool, str]:
    action = str(op.get("op", "")).lower()
    path = str(op.get("path", ""))
    confidence = _float_or_default(op.get("confidence"), 0.0)
    evidence = str(op.get("evidence", ""))

    if action not in {"add", "replace", "remove"}:
        return False, f"Unsupported op '{action}'."
    if not path.startswith("/"):
        return False, "Patch path must be an RFC6901 pointer."
    if confidence < _threshold_for_op(op):
        return False, f"Confidence {confidence:.3f} is below threshold for {action}."
    if _is_connectivity_path(path):
        if confidence < connectivity_confidence_threshold:
            return False, f"Connectivity changes require confidence >= {connectivity_confidence_threshold:.2f}."
        if not evidence.strip():
            return False, "Connectivity changes require explicit evidence."
    if enforce_major_topology_threshold and _is_major_topology_path(path):
        if confidence < major_topology_confidence_threshold:
            return False, f"Major topology changes require confidence >= {major_topology_confidence_threshold:.2f}."
        if not evidence.strip():
            return False, "Major topology changes require explicit evidence."
    if action == "remove" and _is_core_semantics_path(path):
        if not _is_safe_core_remove(op, source_payload):
            return False, "Remove on core process semantics is blocked unless target is provable no-op."
    # The actor-role guard. Runs last so every reason string the existing guards
    # emit keeps its precedence in reports that already grep for them, and because
    # a patch below the confidence bar should still be reported as below the bar.
    actor_role_rejection = _unevidenced_actor_role_rejection(op)
    if actor_role_rejection is not None:
        return False, actor_role_rejection
    # Removals under /entities are intentionally NOT judged here. _is_core_semantics_path
    # covers only /processes/*, and widening it would still leave this function blind:
    # `source_payload` is the pre-batch payload, so after any earlier entity removal the
    # indices in `path` no longer address the rows this function would inspect. The
    # referential-integrity guard runs in apply_patch_with_policy against the live
    # before/after pair instead -- see _referential_integrity_rejection.
    return True, "accepted"


def apply_patch_with_policy(
    source_payload: Dict[str, Any],
    patch_ops: List[Dict[str, Any]],
    *,
    connectivity_confidence_threshold: float = DEFAULT_CONNECTIVITY_CONFIDENCE_THRESHOLD,
    major_topology_confidence_threshold: float = DEFAULT_MAJOR_TOPOLOGY_CONFIDENCE_THRESHOLD,
    enforce_major_topology_threshold: bool = False,
    locked_manifest: Any | None = None,
    stage: str = "patch_application",
    applied_log_path: str | Path | None = None,
    rejected_log_path: str | Path | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    working = deepcopy(source_payload)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    applied_log_records: List[Dict[str, Any]] = []
    rejected_log_records: List[Dict[str, Any]] = []
    lock_context = _build_lock_context(locked_manifest)
    # Look-ahead for the referential-integrity guard. Computed once for the whole
    # batch because the cofactor-relocation repair audit_json_llm.py prompts for
    # arrives as "remove /entities/proteins/N" followed by "add /entities/compounds/-",
    # and the remove is only safe in light of the add that has not run yet.
    pending_entity_names = _pending_entity_name_norms(patch_ops)

    for idx, raw_op in enumerate(patch_ops):
        if not isinstance(raw_op, dict):
            reason = "Patch op is not an object."
            rejected.append({"index": idx, "reason": reason, "op": raw_op})
            rejected_log_records.append(_patch_log_record(stage, raw_op, reason))
            continue
        # Accept both the internal format (op/evidence) and the enrichment format (action/reason)
        op = _normalize_patch_op(raw_op)
        if lock_context is not None:
            lock_rejection = _validate_lock_pre_op(working, op, lock_context)
            if lock_rejection is not None:
                reason, locked_reaction_id = lock_rejection
                record = {"index": idx, "reason": reason, "op": op}
                if locked_reaction_id:
                    record["locked_reaction_id"] = locked_reaction_id
                rejected.append(record)
                rejected_log_records.append(_patch_log_record(stage, op, reason, locked_reaction_id))
                continue
        allow, reason = _should_accept(
            op,
            source_payload,
            connectivity_confidence_threshold=connectivity_confidence_threshold,
            major_topology_confidence_threshold=major_topology_confidence_threshold,
            enforce_major_topology_threshold=enforce_major_topology_threshold,
        )
        record = {"index": idx, "reason": reason, "op": op}
        if not allow:
            rejected.append(record)
            rejected_log_records.append(_patch_log_record(stage, op, reason))
            continue
        try:
            next_working = deepcopy(working)
            _apply_single_op(next_working, op)
            if lock_context is not None:
                lock_rejection = _validate_lock_post_op(working, next_working, op, lock_context)
                if lock_rejection is not None:
                    reason, locked_reaction_id = lock_rejection
                    record["reason"] = reason
                    if locked_reaction_id:
                        record["locked_reaction_id"] = locked_reaction_id
                    rejected.append(record)
                    rejected_log_records.append(_patch_log_record(stage, op, reason, locked_reaction_id))
                    continue
            # Runs after the lock validators so the specific "attempted_to_*" lock
            # reasons keep their precedence in reports that already grep for them;
            # this guard is the backstop for the far larger set of entity removals
            # no lock manifest covers (PMC12444477/research had no manifest entry
            # for either of the two entities behind its 24 dangling references).
            integrity_rejection = _referential_integrity_rejection(
                working, next_working, op, pending_entity_names[idx]
            )
            if integrity_rejection is not None:
                record["reason"] = integrity_rejection
                rejected.append(record)
                rejected_log_records.append(_patch_log_record(stage, op, integrity_rejection))
                continue
            # Read-only, and computed against the pre-op payload while it is
            # still addressable. The records themselves land last, after this
            # op's own logging, so nothing that decides or reports on this op --
            # including _locked_id_for_reaction below, which reads a value the
            # payload may alias -- can observe them.
            lineage_rows = _rows_written_by_op(working, next_working, op)
            working = next_working
            accepted.append(record)
            touched_lock_id = _locked_id_for_reaction(op.get("value"), lock_context) if lock_context is not None else ""
            applied_log_records.append(_patch_log_record(stage, op, "accepted", touched_lock_id))
            _record_audit_lineage(lineage_rows, str(op.get("op", "")).lower())
        except Exception as exc:  # noqa: BLE001
            record["reason"] = f"Application failed: {exc}"
            rejected.append(record)
            rejected_log_records.append(_patch_log_record(stage, op, record["reason"]))

    # Commit point. Everything above ran against `working`, a copy; nothing has
    # been handed back yet, so a failure here can still undo the entire batch.
    rollback_reason, orphaned_references = _batch_rollback_reason(source_payload, working)
    committed = rollback_reason is None
    if not committed:
        logger.warning("apply_patch_with_policy: %s", rollback_reason)
        working = deepcopy(source_payload)
        # The applied log is the record of what CHANGED the payload. After a
        # rollback nothing did, so leaving these entries in it -- each stamped
        # "accepted" -- would have the run's own audit trail assert edits that
        # were undone before anyone saw them. They move to the rejected log,
        # relabelled, and the applied log for this batch is emptied.
        for record in accepted:
            record["rolled_back"] = True
            record["rollback_reason"] = rollback_reason
        for entry in applied_log_records:
            rolled = dict(entry)
            rolled["reason"] = f"{ROLLED_BACK_REASON_PREFIX}: {rollback_reason}"
            rolled["rolled_back"] = True
            rejected_log_records.append(rolled)
        applied_log_records = []
        rejected_log_records.append(
            _patch_log_record(stage, {"op": "batch", "path": "/"}, rollback_reason or "")
        )

    report = {
        # summary keeps its exact three-key shape: several callers and tests
        # compare it as a whole, and the commit verdict belongs to the operation
        # SET rather than to the per-op tallies. It lives in "transaction".
        "summary": {
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "total": len(patch_ops),
        },
        "accepted": accepted,
        "rejected": rejected,
        # accepted_count records the per-op verdicts; committed records whether
        # the caller actually received them. They differ exactly when the set was
        # individually safe and collectively destructive.
        "transaction": {
            "committed": committed,
            "rolled_back": not committed,
            "reason": rollback_reason or "",
            "orphaned_references": orphaned_references,
            "accepted_before_rollback": len(accepted),
            "applied_count": len(accepted) if committed else 0,
        },
    }
    if lock_context is not None or applied_log_path is not None or rejected_log_path is not None:
        report["lock_policy"] = {
            "enabled": lock_context is not None,
            "locked_reaction_count": len(lock_context["entries"]) if lock_context is not None else 0,
        }
        report["applied_patch_log"] = applied_log_records
        report["rejected_patch_log"] = rejected_log_records
    if applied_log_path is not None:
        _append_json_log(Path(applied_log_path), applied_log_records)
    if rejected_log_path is not None:
        _append_json_log(Path(rejected_log_path), rejected_log_records)
    return working, report


def apply_audit_patch_payload(
    payload: Dict[str, Any],
    patch_ops: List[Dict[str, Any]],
    *,
    connectivity_confidence_threshold: float = DEFAULT_CONNECTIVITY_CONFIDENCE_THRESHOLD,
    major_topology_confidence_threshold: float = DEFAULT_MAJOR_TOPOLOGY_CONFIDENCE_THRESHOLD,
    enforce_major_topology_threshold: bool = False,
    locked_manifest: Any | None = None,
    stage: str = "audit",
    applied_patch_log_path: str | Path | None = None,
    rejected_patch_log_path: str | Path | None = None,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    if not isinstance(patch_ops, list):
        raise ValueError("Patch ops must be a JSON list.")
    audited, apply_report = apply_patch_with_policy(
        payload,
        patch_ops,
        connectivity_confidence_threshold=connectivity_confidence_threshold,
        major_topology_confidence_threshold=major_topology_confidence_threshold,
        enforce_major_topology_threshold=enforce_major_topology_threshold,
        locked_manifest=locked_manifest,
        stage=stage,
        applied_log_path=applied_patch_log_path,
        rejected_log_path=rejected_patch_log_path,
    )
    return {"payload": audited, "report": apply_report}


def run_apply(
    input_path: Path,
    patch_path: Path,
    output_path: Path,
    *,
    audit_report_path: Path | None = None,
    apply_report_path: Path | None = None,
    connectivity_confidence_threshold: float = DEFAULT_CONNECTIVITY_CONFIDENCE_THRESHOLD,
    major_topology_confidence_threshold: float = DEFAULT_MAJOR_TOPOLOGY_CONFIDENCE_THRESHOLD,
    enforce_major_topology_threshold: bool = False,
    locked_manifest_path: Path | None = None,
    stage: str = "audit",
    applied_patch_log_path: Path | None = None,
    rejected_patch_log_path: Path | None = None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    patch_ops = json.loads(patch_path.read_text(encoding="utf-8"))
    if not isinstance(patch_ops, list):
        raise ValueError("Patch file must be a JSON list.")

    effective_manifest_path = locked_manifest_path or _discover_locked_manifest_path(
        [input_path, patch_path, output_path, audit_report_path, apply_report_path]
    )
    locked_manifest = _load_locked_manifest(effective_manifest_path)
    if locked_manifest is not None:
        log_dir = output_path.parent
        applied_patch_log_path = applied_patch_log_path or (log_dir / APPLIED_PATCH_LOG_FILENAME)
        rejected_patch_log_path = rejected_patch_log_path or (log_dir / REJECTED_PATCH_LOG_FILENAME)

    result = apply_audit_patch_payload(
        payload,
        patch_ops,
        connectivity_confidence_threshold=connectivity_confidence_threshold,
        major_topology_confidence_threshold=major_topology_confidence_threshold,
        enforce_major_topology_threshold=enforce_major_topology_threshold,
        locked_manifest=locked_manifest,
        stage=stage,
        applied_patch_log_path=applied_patch_log_path,
        rejected_patch_log_path=rejected_patch_log_path,
    )
    audited = _safe_dict(result.get("payload"))
    apply_report = _safe_dict(result.get("report"))
    output_path.write_text(json.dumps(audited, indent=2, ensure_ascii=False), encoding="utf-8")

    if apply_report_path is not None:
        apply_report_path.write_text(json.dumps(apply_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if audit_report_path is not None and audit_report_path.exists():
        audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
        if isinstance(audit_report, dict):
            audit_report["patch_application"] = apply_report
            audit_report_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")

    return apply_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply audit patch operations with deterministic acceptance policy.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input final JSON path")
    parser.add_argument("--patch", dest="patch_path", required=True, help="Patch JSON path")
    parser.add_argument("--out", dest="output_path", default="final.audited.json", help="Audited output JSON path")
    parser.add_argument(
        "--audit-report",
        dest="audit_report_path",
        default="audit_report.json",
        help="Audit report path to enrich with apply logs",
    )
    parser.add_argument(
        "--apply-report",
        dest="apply_report_path",
        default="audit_apply_report.json",
        help="Standalone patch application report path",
    )
    parser.add_argument(
        "--locked-manifest",
        dest="locked_manifest_path",
        default=None,
        help="Optional locked_reaction_manifest.json path. Auto-discovered beside inputs when omitted.",
    )
    parser.add_argument("--stage", dest="stage", default="audit", help="Stage label for patch logs")
    parser.add_argument(
        "--applied-patch-log",
        dest="applied_patch_log_path",
        default=None,
        help="Optional applied patch log path",
    )
    parser.add_argument(
        "--rejected-patch-log",
        dest="rejected_patch_log_path",
        default=None,
        help="Optional rejected patch log path",
    )
    args = parser.parse_args()

    report = run_apply(
        Path(args.input_path),
        Path(args.patch_path),
        Path(args.output_path),
        audit_report_path=Path(args.audit_report_path),
        apply_report_path=Path(args.apply_report_path),
        locked_manifest_path=Path(args.locked_manifest_path) if args.locked_manifest_path else None,
        stage=args.stage,
        applied_patch_log_path=Path(args.applied_patch_log_path) if args.applied_patch_log_path else None,
        rejected_patch_log_path=Path(args.rejected_patch_log_path) if args.rejected_patch_log_path else None,
    )
    print(f"Wrote audited JSON: {args.output_path}")
    # Applied, not merely accepted: after a rollback the two differ and the
    # number a human acts on is the one that changed the file.
    print(
        f"Patch applied: {committed_change_count(report)}, "
        f"accepted: {report['summary']['accepted_count']}, "
        f"rejected: {report['summary']['rejected_count']}"
    )


if __name__ == "__main__":
    main()
