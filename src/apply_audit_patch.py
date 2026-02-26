from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


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


def _should_accept(op: Dict[str, Any]) -> Tuple[bool, str]:
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
        if confidence < 0.98:
            return False, "Connectivity changes require confidence >= 0.98."
        if not evidence.strip():
            return False, "Connectivity changes require explicit evidence."
    if action == "remove" and _is_core_semantics_path(path):
        return False, "Remove on core process semantics is blocked."
    return True, "accepted"


def apply_patch_with_policy(
    source_payload: Dict[str, Any],
    patch_ops: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    working = deepcopy(source_payload)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for idx, op in enumerate(patch_ops):
        if not isinstance(op, dict):
            rejected.append({"index": idx, "reason": "Patch op is not an object.", "op": op})
            continue
        allow, reason = _should_accept(op)
        record = {"index": idx, "reason": reason, "op": op}
        if not allow:
            rejected.append(record)
            continue
        try:
            action = str(op["op"]).lower()
            path = str(op["path"])
            if action == "remove":
                _remove_value(working, path)
            else:
                value = op.get("value")
                _set_value(working, path, value, action)
            accepted.append(record)
        except Exception as exc:  # noqa: BLE001
            record["reason"] = f"Application failed: {exc}"
            rejected.append(record)

    report = {
        "summary": {
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "total": len(patch_ops),
        },
        "accepted": accepted,
        "rejected": rejected,
    }
    return working, report


def run_apply(
    input_path: Path,
    patch_path: Path,
    output_path: Path,
    *,
    audit_report_path: Path | None = None,
    apply_report_path: Path | None = None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    patch_ops = json.loads(patch_path.read_text(encoding="utf-8"))
    if not isinstance(patch_ops, list):
        raise ValueError("Patch file must be a JSON list.")

    audited, apply_report = apply_patch_with_policy(payload, patch_ops)
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
    args = parser.parse_args()

    report = run_apply(
        Path(args.input_path),
        Path(args.patch_path),
        Path(args.output_path),
        audit_report_path=Path(args.audit_report_path),
        apply_report_path=Path(args.apply_report_path),
    )
    print(f"Wrote audited JSON: {args.output_path}")
    print(f"Patch accepted: {report['summary']['accepted_count']}, rejected: {report['summary']['rejected_count']}")


if __name__ == "__main__":
    main()

