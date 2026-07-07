from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME


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
            working = next_working
            accepted.append(record)
            touched_lock_id = _locked_id_for_reaction(op.get("value"), lock_context) if lock_context is not None else ""
            applied_log_records.append(_patch_log_record(stage, op, "accepted", touched_lock_id))
        except Exception as exc:  # noqa: BLE001
            record["reason"] = f"Application failed: {exc}"
            rejected.append(record)
            rejected_log_records.append(_patch_log_record(stage, op, record["reason"]))

    report = {
        "summary": {
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "total": len(patch_ops),
        },
        "accepted": accepted,
        "rejected": rejected,
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
    print(f"Patch accepted: {report['summary']['accepted_count']}, rejected: {report['summary']['rejected_count']}")


if __name__ == "__main__":
    main()
