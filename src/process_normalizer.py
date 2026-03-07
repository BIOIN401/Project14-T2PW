
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


PROTEIN_LIKE_RE = re.compile(
    r"(protein|globulin|peroxidase|symporter|deiodinase|atpase|enzyme|receptor|transporter|kinase|phosphatase)",
    flags=re.IGNORECASE,
)
DEFAULT_SCAFFOLD_NAMES = {"thyroglobulin"}
BYPRODUCT_SUFFIX_DENYLIST = ("acid",)
BYPRODUCT_TOKEN_DENYLIST = {
    "water",
    "proton",
    "oxygen",
    "hydrogen peroxide",
    "carbon dioxide",
    "phosphate",
    "pyrophosphate",
}


class GateValidationError(ValueError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize(value: str) -> str:
    lowered = re.sub(r"\s+", " ", (value or "").strip().casefold())
    return re.sub(r"[^a-z0-9: ]+", "", lowered)


def _canonical(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _has_plus_token(value: str) -> bool:
    return "+" in _canonical(value)


def _split_composite(value: str) -> List[str]:
    text = _canonical(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"\s*\+\s*", text) if part.strip()]


def _composite_key(value: str) -> str:
    parts = [_normalize(part) for part in _split_composite(value)]
    parts = [part for part in parts if part]
    return "+".join(parts)


def _dedupe_preserve(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        c = _canonical(value)
        n = _normalize(c)
        if not c or not n or n in seen:
            continue
        seen.add(n)
        out.append(c)
    return out


def _entity_lists(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    entities = _safe_dict(payload.setdefault("entities", {}))
    if not isinstance(entities.get("compounds"), list):
        entities["compounds"] = []
    if not isinstance(entities.get("proteins"), list):
        entities["proteins"] = []
    if not isinstance(entities.get("protein_complexes"), list):
        entities["protein_complexes"] = []
    return _safe_list(entities["compounds"]), _safe_list(entities["proteins"]), _safe_list(entities["protein_complexes"])


def _process_lists(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    processes = _safe_dict(payload.setdefault("processes", {}))
    if not isinstance(processes.get("reactions"), list):
        processes["reactions"] = []
    if not isinstance(processes.get("transports"), list):
        processes["transports"] = []
    return _safe_list(processes["reactions"]), _safe_list(processes["transports"])


def _new_report() -> Dict[str, Any]:
    return {
        "summary": {
            "complexes_created": 0,
            "composites_rewritten": 0,
            "reactions_rewritten": 0,
            "scaffold_split_reactions": 0,
            "entities_moved_out_of_compounds": 0,
            "entities_added_as_compounds": 0,
            "entities_added_as_proteins": 0,
            "catalysts_promoted_to_enzymes": 0,
            "scaffold_inputs_added": 0,
            "scaffold_in_modifiers_count": 0,
            "n_plus_tokens_remaining": 0,
            "complexes_list": [],
            "n_autostate_created": 0,
            "n_entities_assigned_to_autostate": 0,
            "transporters_attached": 0,
            "modifier_refs_canonicalized": 0,
            "modifier_refs_dropped": 0,
            "forbidden_complexes_removed": 0,
            "dedupe_removed_reactions": 0,
            "dedupe_removed_transports": 0,
            "dedupe_removed": 0,
            "dedupe_removed_total": 0,
            "no_op_removed_count": 0,
        },
        "rewrite_map": {},
        "actions": [],
    }


def _entity_name_norms(rows: Sequence[Any]) -> Set[str]:
    out: Set[str] = set()
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("name"), str) and row.get("name").strip():
            out.add(_normalize(row["name"]))
    return out


def _find_entity_row(rows: Sequence[Any], name: str) -> Optional[Dict[str, Any]]:
    target = _normalize(name)
    if not target:
        return None
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("name"), str) and _normalize(row["name"]) == target:
            return row
    return None


def _remove_entity(rows: List[Dict[str, Any]], name: str) -> bool:
    target = _normalize(name)
    before = len(rows)
    rows[:] = [
        row
        for row in rows
        if not (
            isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and _normalize(row["name"]) == target
        )
    ]
    return len(rows) != before


def _merge_dicts_keep_existing(primary: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(primary)
    for key, value in extra.items():
        if key == "name":
            continue
        if key not in out:
            out[key] = deepcopy(value)
            continue
        if key == "mapped_ids" and isinstance(out.get(key), dict) and isinstance(value, dict):
            merged = dict(value)
            merged.update(out[key])
            out[key] = merged
    return out


def _dedupe_named_rows(rows: List[Dict[str, Any]]) -> None:
    by_norm: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _canonical(str(row.get("name", "")))
        if not name:
            continue
        row["name"] = name
        norm = _normalize(name)
        current = by_norm.get(norm)
        if current is None:
            by_norm[norm] = row
        else:
            by_norm[norm] = _merge_dicts_keep_existing(current, row)
    rows[:] = list(by_norm.values())


def _protein_like_norms(payload: Dict[str, Any]) -> Set[str]:
    _, proteins, complexes = _entity_lists(payload)
    norms = _entity_name_norms(proteins)
    for row in complexes:
        if not isinstance(row, dict):
            continue
        name = _canonical(str(row.get("name", "")))
        if name:
            norms.add(_normalize(name))
        for component in _safe_list(row.get("components")):
            if isinstance(component, str) and component.strip() and PROTEIN_LIKE_RE.search(component):
                norms.add(_normalize(component))
    return norms


def _is_protein_like(name: str, payload: Dict[str, Any]) -> bool:
    norm = _normalize(name)
    if not norm:
        return False
    protein_like_set = _protein_like_norms(payload)
    if norm in protein_like_set:
        return True
    try:
        from map_ids import route_entity_for_mapping  # type: ignore

        routed = route_entity_for_mapping(name, "compound", protein_like_names=protein_like_set)
        if str(routed.get("route", "")).strip().lower() in {"protein", "complex"}:
            return True
    except Exception:  # noqa: BLE001
        pass
    return bool(PROTEIN_LIKE_RE.search(name))


def _scaffold_norms(payload: Dict[str, Any]) -> Set[str]:
    _, _, complexes = _entity_lists(payload)
    scaffolds = {_normalize(name) for name in DEFAULT_SCAFFOLD_NAMES if _normalize(name)}
    for row in complexes:
        if not isinstance(row, dict):
            continue
        name = _canonical(str(row.get("name", "")))
        if ":" not in name:
            continue
        first = _canonical(name.split(":", 1)[0])
        if first:
            scaffolds.add(_normalize(first))
    return scaffolds

def _ensure_protein(name: str, payload: Dict[str, Any], report: Dict[str, Any]) -> str:
    c_name = _canonical(name)
    if not c_name:
        return ""
    compounds, proteins, _ = _entity_lists(payload)
    if _find_entity_row(proteins, c_name) is None:
        proteins.append({"name": c_name})
        report["summary"]["entities_added_as_proteins"] += 1
        report["actions"].append({"type": "entity_added_protein", "name": c_name})
    if _remove_entity(compounds, c_name):
        report["summary"]["entities_moved_out_of_compounds"] += 1
        report["actions"].append({"type": "entity_moved_compound_to_protein", "name": c_name})
    _dedupe_named_rows(proteins)
    return c_name


def _ensure_compound(name: str, payload: Dict[str, Any], report: Dict[str, Any]) -> str:
    c_name = _canonical(name)
    if not c_name:
        return ""
    compounds, proteins, complexes = _entity_lists(payload)
    if _find_entity_row(proteins, c_name) or _find_entity_row(complexes, c_name):
        return c_name
    if _find_entity_row(compounds, c_name) is None:
        compounds.append({"name": c_name})
        report["summary"]["entities_added_as_compounds"] += 1
        report["actions"].append({"type": "entity_added_compound", "name": c_name})
    _dedupe_named_rows(compounds)
    return c_name


def _complex_components(name: str) -> List[str]:
    text = _canonical(name)
    if ":" not in text:
        return []
    return [part.strip() for part in text.split(":") if part.strip()]


def _is_likely_byproduct(token: str) -> bool:
    t = _canonical(token).casefold()
    if not t:
        return False
    if t in BYPRODUCT_TOKEN_DENYLIST:
        return True
    return any(t.endswith(suffix) for suffix in BYPRODUCT_SUFFIX_DENYLIST)


def canonicalize_modifier_name(name: str) -> str:
    text = _canonical(name).replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    lowered = text.casefold()
    if lowered.endswith(" complex"):
        text = text[: -len(" complex")].strip()
    return re.sub(r"\s+", " ", text).strip()


def _canonical_complex_name(name: str) -> str:
    text = _canonical(name)
    if not text:
        return ""
    parts = _complex_components(text)
    if len(parts) >= 2:
        return ":".join(parts)
    return text


def _has_explicit_composite_token(left: str, right: str, evidence_text: str) -> bool:
    if not _canonical(evidence_text):
        return False
    left_e = re.escape(_canonical(left))
    right_e = re.escape(_canonical(right))
    return bool(re.search(rf"{left_e}\s*\+\s*{right_e}", evidence_text, flags=re.IGNORECASE))


def _reaction_output_complex_evidence(
    *,
    left: str,
    right: str,
    evidence_text: str,
) -> Dict[str, bool]:
    evidence = _canonical(evidence_text).casefold()
    if not evidence:
        return {"supported": False, "exact_composite": False, "complex_phrase": False}
    left_e = re.escape(_canonical(left))
    right_e = re.escape(_canonical(right))

    exact_plus_patterns = [
        rf"{left_e}\s*\+\s*{right_e}",
    ]
    exact_composite = any(re.search(pattern, evidence, flags=re.IGNORECASE) for pattern in exact_plus_patterns)

    binding_patterns = [
        rf"{left_e}\s+bound\s+to\s+{right_e}",
        rf"{left_e}\s*-\s*{right_e}\s+complex",
        rf"{left_e}\s+conjugated\s+to\s+{right_e}",
    ]
    complex_phrase = any(re.search(pattern, evidence, flags=re.IGNORECASE) for pattern in binding_patterns)
    return {
        "supported": bool(exact_composite or complex_phrase),
        "exact_composite": bool(exact_composite),
        "complex_phrase": bool(complex_phrase),
    }


def materialize_complex(
    nameA: str,
    nameB: str,
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
    extra_components: Optional[Sequence[str]] = None,
    source_token: str = "",
    evidence_text: str = "",
    caller: str = "",
) -> str:
    rep = report if isinstance(report, dict) else _new_report()
    parts = [nameA, nameB]
    if extra_components:
        parts.extend(list(extra_components))
    clean_parts = _dedupe_preserve(parts)
    if len(clean_parts) < 2:
        return clean_parts[0] if clean_parts else ""

    for idx, part in enumerate(clean_parts):
        if _is_protein_like(part, payload) or idx == 0:
            _ensure_protein(part, payload, rep)
        else:
            _ensure_compound(part, payload, rep)

    complex_name = ":".join(clean_parts)
    _, _, complexes = _entity_lists(payload)
    existing = _find_entity_row(complexes, complex_name)
    if existing is not None:
        existing["name"] = complex_name
        existing["components"] = clean_parts
        rep["actions"].append(
            {
                "type": "complex_creation_debug",
                "status": "existing",
                "name": complex_name,
                "components": clean_parts,
                "composite_token": _canonical(source_token),
                "evidence_snippet": _canonical(evidence_text)[:240],
                "calling_function": caller or "materialize_complex",
            }
        )
        return complex_name

    complexes.append({"name": complex_name, "components": clean_parts})
    _dedupe_named_rows(complexes)
    rep["summary"]["complexes_created"] += 1
    rep["actions"].append(
        {
            "type": "complex_created",
            "name": complex_name,
            "components": clean_parts,
            "composite_token": _canonical(source_token),
            "evidence_snippet": _canonical(evidence_text)[:240],
            "calling_function": caller or "materialize_complex",
        }
    )
    rep["actions"].append(
        {
            "type": "complex_creation_debug",
            "status": "created",
            "name": complex_name,
            "components": clean_parts,
            "composite_token": _canonical(source_token),
            "evidence_snippet": _canonical(evidence_text)[:240],
            "calling_function": caller or "materialize_complex",
        }
    )
    return complex_name


def _rewrite_token(
    token: str,
    payload: Dict[str, Any],
    report: Dict[str, Any],
    rewrite_map: Dict[str, str],
    pointer: str,
    *,
    evidence_text: str = "",
) -> List[str]:
    text = _canonical(token)
    if not text:
        return []
    direct_norm = _normalize(text)
    if direct_norm in rewrite_map:
        return [rewrite_map[direct_norm]]
    ckey = _composite_key(text)
    if ckey and ckey in rewrite_map:
        return [rewrite_map[ckey]]

    if not _has_plus_token(text):
        if ":" in text and len(_complex_components(text)) >= 2:
            parts = _complex_components(text)
            complex_name = materialize_complex(
                parts[0],
                parts[1],
                payload,
                report=report,
                extra_components=parts[2:],
                source_token=text,
                evidence_text=evidence_text,
                caller="_rewrite_token",
            )
            rewrite_map[direct_norm] = complex_name
            return [complex_name]
        if _is_protein_like(text, payload):
            _ensure_protein(text, payload, report)
        else:
            _ensure_compound(text, payload, report)
        return [text]

    parts = _split_composite(text)
    if len(parts) < 2:
        return [text]

    if _is_protein_like(parts[0], payload):
        is_reaction_output_pointer = "/processes/reactions/" in pointer and pointer.endswith("/outputs")
        if is_reaction_output_pointer:
            right = parts[1] if len(parts) > 1 else ""
            signal = _reaction_output_complex_evidence(left=parts[0], right=right, evidence_text=evidence_text)
            supported = bool(signal.get("supported", False))
            exact_composite = bool(signal.get("exact_composite", False))
            block_for_byproduct = _is_likely_byproduct(right) and not exact_composite
            if (not supported) or block_for_byproduct:
                out: List[str] = []
                for part in parts:
                    c_part = _canonical(part)
                    if not c_part:
                        continue
                    if _is_protein_like(c_part, payload):
                        _ensure_protein(c_part, payload, report)
                    else:
                        _ensure_compound(c_part, payload, report)
                    out.append(c_part)
                report["actions"].append(
                    {
                        "type": "composite_not_materialized_without_evidence",
                        "json_pointer": pointer,
                        "from": text,
                        "to": out,
                        "supported_by_evidence": supported,
                        "exact_composite_match": exact_composite,
                        "blocked_by_byproduct_rule": block_for_byproduct,
                    }
                )
                return out
        complex_name = materialize_complex(
            parts[0],
            parts[1],
            payload,
            report=report,
            extra_components=parts[2:],
            source_token=text,
            evidence_text=evidence_text,
            caller="_rewrite_token",
        )
        rewrite_map[ckey] = complex_name
        rewrite_map[direct_norm] = complex_name
        report["summary"]["composites_rewritten"] += 1
        report["actions"].append(
            {
                "type": "composite_rewritten_to_complex",
                "json_pointer": pointer,
                "from": text,
                "to": complex_name,
            }
        )
        return [complex_name]

    raise ValueError(
        f"Composite token '{text}' at {pointer} has no protein-like left component; "
        "compound+compound composite materialization is not supported."
    )


def _collapse_reaction_outputs(outputs: List[str]) -> List[str]:
    complex_parts: Set[str] = set()
    for token in outputs:
        for part in _complex_components(token):
            complex_parts.add(_normalize(part))
    if not complex_parts:
        return outputs
    collapsed: List[str] = []
    for token in outputs:
        if _normalize(token) in complex_parts and not _complex_components(token):
            continue
        collapsed.append(token)
    return collapsed


def rewrite_process_references(
    payload: Dict[str, Any],
    rewrite_map: Dict[str, str],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    reactions, transports = _process_lists(payload)

    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        changed = False
        reaction_evidence = _canonical(str(reaction.get("evidence", "")))
        for side in ["inputs", "outputs"]:
            pointer = f"/processes/reactions/{ridx}/{side}"
            new_tokens: List[str] = []
            for token in _safe_list(reaction.get(side)):
                if not isinstance(token, str):
                    continue
                rewritten = _rewrite_token(
                    token,
                    payload,
                    rep,
                    rewrite_map,
                    pointer,
                    evidence_text=reaction_evidence,
                )
                if [_canonical(token)] != rewritten:
                    changed = True
                new_tokens.extend(rewritten)
            new_tokens = _dedupe_preserve(new_tokens)
            if side == "outputs":
                collapsed = _collapse_reaction_outputs(new_tokens)
                if collapsed != new_tokens:
                    changed = True
                    rep["summary"]["scaffold_split_reactions"] += 1
                new_tokens = collapsed
            reaction[side] = new_tokens
        if changed:
            rep["summary"]["reactions_rewritten"] += 1

    rewritten_transports: List[Dict[str, Any]] = []
    for tidx, transport in enumerate(transports):
        if not isinstance(transport, dict):
            continue
        pointer = f"/processes/transports/{tidx}/cargo"
        row = deepcopy(transport)
        transport_evidence = _canonical(str(row.get("evidence", "")))
        cargo_raw = _canonical(
            str(
                row.get("cargo_complex") if isinstance(row.get("cargo_complex"), str) else row.get("cargo") or ""
            )
        )
        if not cargo_raw:
            rewritten_transports.append(row)
            continue

        rewritten = _rewrite_token(
            cargo_raw,
            payload,
            rep,
            rewrite_map,
            pointer,
            evidence_text=transport_evidence,
        )
        if len(rewritten) == 1:
            cargo_value = rewritten[0]
            if ":" in cargo_value:
                row["cargo"] = None
                row["cargo_complex"] = cargo_value
            else:
                row["cargo"] = cargo_value
                row.pop("cargo_complex", None)
            rewritten_transports.append(row)
            continue

        for part in rewritten:
            clone = deepcopy(row)
            clone["cargo"] = part
            clone.pop("cargo_complex", None)
            rewritten_transports.append(clone)
            rep["actions"].append({"type": "transport_split_row", "json_pointer": pointer, "from": cargo_raw, "to": part})

    _safe_dict(payload.setdefault("processes", {}))["transports"] = rewritten_transports
    rep["rewrite_map"] = dict(rewrite_map)
    return payload

def _rewrite_element_locations(payload: Dict[str, Any], rewrite_map: Dict[str, str], report: Dict[str, Any]) -> None:
    element_locations = _safe_dict(payload.setdefault("element_locations", {}))
    if not isinstance(element_locations.get("compound_locations"), list):
        element_locations["compound_locations"] = []
    if not isinstance(element_locations.get("protein_locations"), list):
        element_locations["protein_locations"] = []

    compound_locations = _safe_list(element_locations["compound_locations"])
    protein_locations = _safe_list(element_locations["protein_locations"])

    def _append_unique(rows: List[Dict[str, Any]], row: Dict[str, Any], key: str) -> None:
        name = _canonical(str(row.get(key, "")))
        state = _canonical(str(row.get("biological_state", "")))
        if not name:
            return
        for existing in rows:
            if not isinstance(existing, dict):
                continue
            ex_name = _canonical(str(existing.get(key, "")))
            ex_state = _canonical(str(existing.get("biological_state", "")))
            if _normalize(ex_name) == _normalize(name) and _normalize(ex_state) == _normalize(state):
                return
        rows.append(row)

    kept_compounds: List[Dict[str, Any]] = []
    for idx, row in enumerate(compound_locations):
        if not isinstance(row, dict):
            continue
        raw_name = _canonical(str(row.get("compound", "")))
        if not raw_name:
            continue
        rewritten = _rewrite_token(raw_name, payload, report, rewrite_map, f"/element_locations/compound_locations/{idx}/compound")
        for token in rewritten:
            if ":" in token or _is_protein_like(token, payload):
                moved = dict(row)
                moved.pop("compound", None)
                moved["protein"] = token
                _append_unique(protein_locations, moved, "protein")
            else:
                kept = dict(row)
                kept["compound"] = token
                _append_unique(kept_compounds, kept, "compound")

    cleaned_proteins: List[Dict[str, Any]] = []
    for idx, row in enumerate(protein_locations):
        if not isinstance(row, dict):
            continue
        raw_name = _canonical(str(row.get("protein", "")))
        if not raw_name:
            continue
        rewritten = _rewrite_token(raw_name, payload, report, rewrite_map, f"/element_locations/protein_locations/{idx}/protein")
        for token in rewritten:
            moved = dict(row)
            moved["protein"] = token
            _append_unique(cleaned_proteins, moved, "protein")

    element_locations["compound_locations"] = kept_compounds
    element_locations["protein_locations"] = cleaned_proteins


def normalize_composites(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    compounds, proteins, complexes = _entity_lists(payload)
    _process_lists(payload)

    for rows in [compounds, proteins, complexes]:
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                row["name"] = _canonical(row["name"])

    for row in list(complexes):
        if not isinstance(row, dict):
            continue
        name = _canonical(str(row.get("name", "")))
        if not name:
            continue
        if _has_plus_token(name):
            parts = _split_composite(name)
            if len(parts) >= 2:
                if not _is_protein_like(parts[0], payload):
                    raise ValueError(
                        f"Composite complex '{name}' has non protein-like left token '{parts[0]}'; unsupported."
                    )
                canonical = materialize_complex(
                    parts[0],
                    parts[1],
                    payload,
                    report=rep,
                    extra_components=parts[2:],
                    source_token=name,
                    caller="normalize_composites",
                )
                rep["rewrite_map"][_composite_key(name)] = canonical
                rep["rewrite_map"][_normalize(name)] = canonical
                _remove_entity(complexes, name)
                continue
        parts = _complex_components(name)
        if len(parts) >= 2:
            row["name"] = ":".join(parts)
            row["components"] = _dedupe_preserve(parts)

    kept_compounds: List[Dict[str, Any]] = []
    for row in compounds:
        if not isinstance(row, dict):
            continue
        name = _canonical(str(row.get("name", "")))
        if not name:
            continue
        if _has_plus_token(name):
            parts = _split_composite(name)
            if len(parts) >= 2 and _is_protein_like(parts[0], payload):
                canonical = materialize_complex(
                    parts[0],
                    parts[1],
                    payload,
                    report=rep,
                    extra_components=parts[2:],
                    source_token=name,
                    caller="normalize_composites",
                )
                rep["rewrite_map"][_composite_key(name)] = canonical
                rep["rewrite_map"][_normalize(name)] = canonical
                rep["summary"]["entities_moved_out_of_compounds"] += 1
                rep["summary"]["composites_rewritten"] += 1
                continue
            raise ValueError(
                f"Composite entity '{name}' in /entities/compounds has no protein-like left component; unsupported."
            )
        kept_compounds.append(row)
    compounds[:] = kept_compounds

    _dedupe_named_rows(compounds)
    _dedupe_named_rows(proteins)
    _dedupe_named_rows(complexes)

    rewrite_process_references(payload, _safe_dict(rep.get("rewrite_map")), report=rep)
    _rewrite_element_locations(payload, _safe_dict(rep.get("rewrite_map")), rep)
    _dedupe_named_rows(compounds)
    _dedupe_named_rows(proteins)
    _dedupe_named_rows(complexes)
    return payload


def rewrite_reactions_to_complex_states(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    reactions, _ = _process_lists(payload)
    scaffold_norms = _scaffold_norms(payload)

    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        outputs = _dedupe_preserve([str(v) for v in _safe_list(reaction.get("outputs")) if isinstance(v, str)])
        if not outputs:
            continue

        scaffold_tokens: List[str] = []
        non_protein_tokens: List[str] = []
        for token in outputs:
            norm = _normalize(token)
            if ":" in token:
                continue
            if norm in scaffold_norms:
                scaffold_tokens.append(token)
            elif not _is_protein_like(token, payload):
                non_protein_tokens.append(token)

        if not scaffold_tokens or not non_protein_tokens:
            continue

        reaction_evidence = _canonical(str(reaction.get("evidence", "")))
        rewritten = False
        consumed_non_protein: Set[str] = set()
        new_outputs: List[str] = []
        base_outputs = list(outputs)

        for scaffold in scaffold_tokens:
            chosen = ""
            chosen_exact = False
            best_score = -1
            for candidate in non_protein_tokens:
                if _normalize(candidate) in consumed_non_protein:
                    continue
                signal = _reaction_output_complex_evidence(
                    left=scaffold,
                    right=candidate,
                    evidence_text=reaction_evidence,
                )
                if not bool(signal.get("supported", False)):
                    continue
                exact_composite = bool(signal.get("exact_composite", False))
                if _is_likely_byproduct(candidate) and not exact_composite:
                    continue
                score = 2 if exact_composite else 1
                if score > best_score:
                    chosen = candidate
                    chosen_exact = exact_composite
                    best_score = score
            if not chosen:
                continue
            complex_name = materialize_complex(
                scaffold,
                chosen,
                payload,
                report=rep,
                source_token=f"{scaffold}+{chosen}",
                evidence_text=reaction_evidence,
                caller="rewrite_reactions_to_complex_states",
            )
            consumed_non_protein.add(_normalize(chosen))
            base_outputs = [
                tok
                for tok in base_outputs
                if _normalize(tok) not in {_normalize(scaffold), _normalize(chosen)}
            ]
            new_outputs.append(complex_name)
            rewritten = True
            rep["actions"].append(
                {
                    "type": "reaction_output_scaffold_state_bound",
                    "json_pointer": f"/processes/reactions/{ridx}/outputs",
                    "scaffold": scaffold,
                    "product": chosen,
                    "complex": complex_name,
                    "exact_composite_match": chosen_exact,
                }
            )

        if rewritten:
            base_outputs.extend(new_outputs)
            reaction["outputs"] = _dedupe_preserve(base_outputs)
            rep["summary"]["reactions_rewritten"] += 1
            rep["summary"]["scaffold_split_reactions"] += 1
            rep["actions"].append(
                {
                    "type": "reaction_output_scaffold_split_rewrite",
                    "json_pointer": f"/processes/reactions/{ridx}/outputs",
                    "outputs": reaction["outputs"],
                }
            )
    return payload


def cleanup_disallowed_complexes(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
    forbidden_complexes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("forbidden_complexes_removed", 0)
    compounds, _, complexes = _entity_lists(payload)
    reactions, transports = _process_lists(payload)
    element_locations = _safe_dict(payload.setdefault("element_locations", {}))
    compound_locations = _safe_list(element_locations.get("compound_locations"))
    protein_locations = _safe_list(element_locations.get("protein_locations"))
    if not isinstance(element_locations.get("compound_locations"), list):
        element_locations["compound_locations"] = compound_locations
    if not isinstance(element_locations.get("protein_locations"), list):
        element_locations["protein_locations"] = protein_locations

    forbidden_norms = {
        _normalize(_canonical_complex_name(name))
        for name in (forbidden_complexes or ["thyroglobulin:2-aminoacrylic acid"])
        if _normalize(_canonical_complex_name(name))
    }
    evidence_by_complex_norm: Dict[str, List[str]] = {}

    def _collect_evidence(complex_name: str, evidence: str) -> None:
        norm = _normalize(_canonical_complex_name(complex_name))
        if not norm:
            return
        evidence_by_complex_norm.setdefault(norm, []).append(_canonical(evidence))

    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        evidence = _canonical(str(reaction.get("evidence", "")))
        for side in ["inputs", "outputs"]:
            for token in _safe_list(reaction.get(side)):
                if isinstance(token, str) and ":" in token and len(_complex_components(token)) >= 2:
                    _collect_evidence(token, evidence)

    for transport in transports:
        if not isinstance(transport, dict):
            continue
        evidence = _canonical(str(transport.get("evidence", "")))
        token = transport.get("cargo_complex")
        if isinstance(token, str) and token.strip():
            _collect_evidence(token, evidence)

    disallowed_by_norm: Dict[str, Dict[str, Any]] = {}
    kept_complexes: List[Dict[str, Any]] = []
    for idx, row in enumerate(complexes):
        if not isinstance(row, dict):
            continue
        name = _canonical_complex_name(str(row.get("name", "")))
        if not name:
            continue
        row["name"] = name
        parts = _complex_components(name)
        if len(parts) < 2:
            kept_complexes.append(row)
            continue
        left = _canonical(parts[0])
        right = _canonical(parts[1])
        norm = _normalize(name)
        explicit_supported = any(
            _has_explicit_composite_token(left, right, evidence_text)
            for evidence_text in evidence_by_complex_norm.get(norm, [])
            if _canonical(evidence_text)
        )
        forbidden = norm in forbidden_norms
        byproduct_block = _is_likely_byproduct(right) and not explicit_supported
        if forbidden or byproduct_block:
            disallowed_by_norm[norm] = {
                "name": name,
                "left": left,
                "right": right,
                "forbidden": forbidden,
                "byproduct_block": byproduct_block,
                "explicit_supported": explicit_supported,
                "entity_pointer": f"/entities/protein_complexes/{idx}/name",
            }
            summary["forbidden_complexes_removed"] += 1
            rep["actions"].append(
                {
                    "type": "forbidden_or_byproduct_complex_removed",
                    "json_pointer": f"/entities/protein_complexes/{idx}/name",
                    "complex": name,
                    "left": left,
                    "right": right,
                    "forbidden": forbidden,
                    "byproduct_block": byproduct_block,
                    "explicit_composite_support": explicit_supported,
                }
            )
            _ensure_compound(right, payload, rep)
            _ensure_protein(left, payload, rep)
            continue
        kept_complexes.append(row)
    complexes[:] = kept_complexes

    if not disallowed_by_norm:
        return payload

    def _rewrite_token_if_disallowed(token: str, *, side: str, pointer: str) -> List[str]:
        token_name = _canonical_complex_name(token)
        norm = _normalize(token_name)
        info = disallowed_by_norm.get(norm)
        if info is None:
            return [_canonical(token)]
        left = _canonical(str(info.get("left", "")))
        right = _canonical(str(info.get("right", "")))
        if side == "outputs":
            replacement = [right] if right else []
        elif side == "inputs":
            replacement = [left, right] if left and right else [right or left]
        else:
            replacement = [right] if right else []
        rep["actions"].append(
            {
                "type": "forbidden_complex_reference_rewritten",
                "json_pointer": pointer,
                "from": token_name,
                "to": replacement,
            }
        )
        return [part for part in replacement if part]

    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            pointer = f"/processes/reactions/{ridx}/{side}"
            rewritten: List[str] = []
            for token in _safe_list(reaction.get(side)):
                if not isinstance(token, str):
                    continue
                rewritten.extend(_rewrite_token_if_disallowed(token, side=side, pointer=pointer))
            reaction[side] = _dedupe_preserve([tok for tok in rewritten if tok])

    for tidx, transport in enumerate(transports):
        if not isinstance(transport, dict):
            continue
        cargo = _canonical(str(transport.get("cargo", "")))
        cargo_norm = _normalize(_canonical_complex_name(cargo))
        cargo_info = disallowed_by_norm.get(cargo_norm)
        if cargo_info is not None:
            right = _canonical(str(cargo_info.get("right", "")))
            if right:
                transport["cargo"] = right
                transport.pop("cargo_complex", None)
                rep["actions"].append(
                    {
                        "type": "forbidden_complex_transport_rewritten",
                        "json_pointer": f"/processes/transports/{tidx}/cargo",
                        "from": cargo,
                        "to": right,
                    }
                )
        cargo_complex = _canonical(str(transport.get("cargo_complex", "")))
        if cargo_complex:
            norm = _normalize(_canonical_complex_name(cargo_complex))
            info = disallowed_by_norm.get(norm)
            if info is not None:
                right = _canonical(str(info.get("right", "")))
                if right:
                    transport["cargo"] = right
                    transport.pop("cargo_complex", None)
                    rep["actions"].append(
                        {
                            "type": "forbidden_complex_transport_rewritten",
                            "json_pointer": f"/processes/transports/{tidx}",
                            "from": cargo_complex,
                            "to": right,
                        }
                    )

    for cidx, row in enumerate(compound_locations):
        if not isinstance(row, dict):
            continue
        cname = _canonical_complex_name(str(row.get("compound", "")))
        info = disallowed_by_norm.get(_normalize(cname))
        if info is None:
            continue
        right = _canonical(str(info.get("right", "")))
        if right:
            row["compound"] = right
            rep["actions"].append(
                {
                    "type": "forbidden_complex_location_rewritten",
                    "json_pointer": f"/element_locations/compound_locations/{cidx}/compound",
                    "from": cname,
                    "to_compound": right,
                }
            )

    new_protein_locations: List[Dict[str, Any]] = []
    for pidx, row in enumerate(protein_locations):
        if not isinstance(row, dict):
            continue
        pname = _canonical_complex_name(str(row.get("protein", "")))
        norm = _normalize(pname)
        info = disallowed_by_norm.get(norm)
        if info is None:
            new_protein_locations.append(row)
            continue
        right = _canonical(str(info.get("right", "")))
        if right:
            moved = dict(row)
            moved.pop("protein", None)
            moved["compound"] = right
            compound_locations.append(moved)
        rep["actions"].append(
            {
                "type": "forbidden_complex_location_rewritten",
                "json_pointer": f"/element_locations/protein_locations/{pidx}/protein",
                "from": pname,
                "to_compound": right,
            }
        )
    element_locations["protein_locations"] = new_protein_locations
    element_locations["compound_locations"] = compound_locations
    _dedupe_named_rows(compounds)
    _dedupe_named_rows(complexes)
    return payload


def normalize_process_actor_schema(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("modifier_refs_canonicalized", 0)
    summary.setdefault("modifier_refs_dropped", 0)
    entities = _safe_dict(payload.get("entities"))
    protein_rows = _safe_list(entities.get("proteins"))
    complex_rows = _safe_list(entities.get("protein_complexes"))
    protein_by_norm = {
        _normalize(_canonical(str(row.get("name", "")))): _canonical(str(row.get("name", "")))
        for row in protein_rows
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    }
    complex_by_norm = {
        _normalize(_canonical_complex_name(str(row.get("name", "")))): _canonical_complex_name(str(row.get("name", "")))
        for row in complex_rows
        if isinstance(row, dict) and _canonical_complex_name(str(row.get("name", "")))
    }
    reactions, transports = _process_lists(payload)

    def _resolve_actor_name(raw_value: str) -> Optional[Tuple[str, str]]:
        raw = _canonical(raw_value)
        if not raw:
            return None
        candidates = [raw, canonicalize_modifier_name(raw), _canonical_complex_name(raw)]
        deduped_candidates = _dedupe_preserve([c for c in candidates if _canonical(c)])
        for candidate in deduped_candidates:
            norm = _normalize(_canonical_complex_name(candidate))
            if norm in protein_by_norm:
                return "protein", protein_by_norm[norm]
            if norm in complex_by_norm:
                return "protein_complex", complex_by_norm[norm]
        return None

    def _rewrite_actor_rows(rows: List[Any], pointer_prefix: str, *, drop_unknown: bool = True) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            pointer = f"{pointer_prefix}/{idx}"
            source_field = ""
            raw_name = ""
            for field in ["protein", "protein_complex", "name"]:
                candidate = _canonical(str(row.get(field, "")))
                if candidate:
                    source_field = field
                    raw_name = candidate
                    break
            if not raw_name:
                if not drop_unknown:
                    kept.append(row)
                continue

            resolved = _resolve_actor_name(raw_name)
            if resolved is None:
                if drop_unknown:
                    summary["modifier_refs_dropped"] += 1
                    rep["actions"].append(
                        {
                            "type": "schema_drift_drop_unknown_actor",
                            "json_pointer": pointer,
                            "name": raw_name,
                            "source_field": source_field,
                        }
                    )
                    continue
                kept.append(row)
                continue

            target_field, canonical_name = resolved
            updated = dict(row)
            for field in ["protein", "protein_complex", "name"]:
                updated.pop(field, None)
            updated[target_field] = canonical_name
            if source_field != target_field or _normalize(raw_name) != _normalize(canonical_name):
                summary["modifier_refs_canonicalized"] += 1
                rep["actions"].append(
                    {
                        "type": "schema_drift_actor_canonicalized",
                        "json_pointer": pointer,
                        "from_field": source_field,
                        "to_field": target_field,
                        "from_name": raw_name,
                        "to_name": canonical_name,
                    }
                )
            kept.append(updated)
        return kept

    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        for key in ["enzymes", "modifiers"]:
            rows = _safe_list(reaction.get(key))
            if not isinstance(reaction.get(key), list):
                reaction[key] = rows
            reaction[key] = _rewrite_actor_rows(rows, f"/processes/reactions/{ridx}/{key}")

    for tidx, transport in enumerate(transports):
        if not isinstance(transport, dict):
            continue
        rows = _safe_list(transport.get("transporters"))
        if not isinstance(transport.get("transporters"), list):
            transport["transporters"] = rows
        transport["transporters"] = _rewrite_actor_rows(rows, f"/processes/transports/{tidx}/transporters")
    return payload


def ensure_autostates(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    auto_state_name = "__auto_state__"
    auto_location_name = "cell"

    if not isinstance(payload.get("biological_states"), list):
        payload["biological_states"] = []
    biological_states = _safe_list(payload.get("biological_states"))
    existing_states = {
        _normalize(str(row.get("name", "")))
        for row in biological_states
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    if _normalize(auto_state_name) not in existing_states:
        biological_states.append({"name": auto_state_name, "subcellular_location": auto_location_name})
        rep["summary"]["n_autostate_created"] += 1

    element_locations = _safe_dict(payload.setdefault("element_locations", {}))
    for list_key in ["compound_locations", "protein_locations"]:
        rows = _safe_list(element_locations.get(list_key))
        for row in rows:
            if not isinstance(row, dict):
                continue
            state = _canonical(str(row.get("biological_state", "")))
            if state:
                continue
            row["biological_state"] = auto_state_name
            rep["summary"]["n_entities_assigned_to_autostate"] += 1

    _, transports = _process_lists(payload)
    for row in transports:
        if not isinstance(row, dict):
            continue
        from_state = _canonical(str(row.get("from_biological_state", "")))
        to_state = _canonical(str(row.get("to_biological_state", "")))
        if not from_state:
            row["from_biological_state"] = auto_state_name
            rep["summary"]["n_entities_assigned_to_autostate"] += 1
        if not to_state:
            row["to_biological_state"] = auto_state_name
            rep["summary"]["n_entities_assigned_to_autostate"] += 1
    return payload


def attach_transporters_from_evidence(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    entities = _safe_dict(payload.get("entities"))
    element_locations = _safe_dict(payload.get("element_locations"))
    _, transports = _process_lists(payload)

    known_transporters: List[str] = [
        _canonical(str(row.get("name", "")))
        for row in _safe_list(entities.get("proteins"))
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    ]
    if not known_transporters:
        return payload

    protein_location_rows = [
        row
        for row in _safe_list(element_locations.get("protein_locations"))
        if isinstance(row, dict) and _canonical(str(row.get("evidence", "")))
    ]

    cue_prefix = r"(?:using|via|through|transported by)"

    known_norm_to_name = {_normalize(name): name for name in known_transporters if _normalize(name)}

    def _match_transporter(
        text: str,
        *,
        cargo_tokens: Sequence[str],
        require_cargo_hit: bool,
    ) -> Optional[Tuple[str, str]]:
        evidence_text = _canonical(text)
        if not evidence_text:
            return None
        evidence_norm = evidence_text.casefold()
        for transporter_name in sorted(known_transporters, key=len, reverse=True):
            pname = _canonical(transporter_name)
            if not pname:
                continue
            pattern = re.compile(rf"\b{cue_prefix}\s+{re.escape(pname)}\b", flags=re.IGNORECASE)
            match = pattern.search(evidence_text)
            if not match:
                continue
            if require_cargo_hit and cargo_tokens:
                cargo_hit = any(_canonical(token).casefold() in evidence_norm for token in cargo_tokens if _canonical(token))
                if not cargo_hit:
                    continue
            return pname, evidence_text[match.start() : match.end()]
        return None

    for tidx, transport in enumerate(transports):
        if not isinstance(transport, dict):
            continue
        transporters = _safe_list(transport.get("transporters"))
        if not isinstance(transport.get("transporters"), list):
            transport["transporters"] = transporters
        existing_norms: Set[str] = set()
        for existing in transporters:
            if not isinstance(existing, dict):
                continue
            for key in ["protein", "protein_complex", "name"]:
                existing_name = _canonical(str(existing.get(key, "")))
                if existing_name and _normalize(existing_name) in known_norm_to_name:
                    existing_norms.add(_normalize(existing_name))
                    break

        cargo_value = (
            transport.get("cargo_complex")
            if isinstance(transport.get("cargo_complex"), str) and _canonical(str(transport.get("cargo_complex", "")))
            else transport.get("cargo")
        )
        cargo = _canonical(str(cargo_value or ""))
        cargo_tokens = [cargo]
        if ":" in cargo:
            cargo_tokens.extend(_complex_components(cargo))

        evidence = _canonical(str(transport.get("evidence", "")))
        matched = (
            _match_transporter(evidence, cargo_tokens=cargo_tokens, require_cargo_hit=False)
            if evidence
            else None
        )

        if matched is None:
            from_state = _canonical(str(transport.get("from_biological_state", "")))
            to_state = _canonical(str(transport.get("to_biological_state", "")))
            for prow in protein_location_rows:
                prow_evidence = _canonical(str(prow.get("evidence", "")))
                prow_state = _canonical(str(prow.get("biological_state", "")))
                if prow_state and from_state and to_state and prow_state not in {from_state, to_state}:
                    continue
                matched = _match_transporter(
                    prow_evidence,
                    cargo_tokens=cargo_tokens,
                    require_cargo_hit=True,
                )
                if matched is not None:
                    break

        if matched is None:
            continue
        transporter_name, snippet = matched
        transporter_norm = _normalize(transporter_name)
        if transporter_norm not in existing_norms:
            transporters.append({"protein": known_norm_to_name.get(transporter_norm, transporter_name), "evidence": snippet})
            existing_norms.add(transporter_norm)
            rep["summary"]["transporters_attached"] += 1
            rep["actions"].append(
                {
                    "type": "transporter_attached_from_evidence",
                    "json_pointer": f"/processes/transports/{tidx}/transporters",
                    "protein": known_norm_to_name.get(transporter_norm, transporter_name),
                    "snippet": snippet,
                }
            )
        transport["transporters"] = transporters
    return payload


def promote_catalysts(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    reactions, _ = _process_lists(payload)
    _, proteins, complexes = _entity_lists(payload)
    protein_norms = _entity_name_norms(proteins) | _entity_name_norms(complexes)
    scaffold_norms = _scaffold_norms(payload)

    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        inputs = _dedupe_preserve([str(v) for v in _safe_list(reaction.get("inputs")) if isinstance(v, str)])
        outputs = _dedupe_preserve([str(v) for v in _safe_list(reaction.get("outputs")) if isinstance(v, str)])
        enzymes = _safe_list(reaction.get("enzymes"))
        if not isinstance(reaction.get("enzymes"), list):
            reaction["enzymes"] = enzymes

        output_complex_parts: Set[str] = set()
        for token in outputs:
            for part in _complex_components(token):
                output_complex_parts.add(_normalize(part))
        output_norms = {_normalize(token) for token in outputs}
        enzyme_norms: Set[str] = set()
        for enzyme in enzymes:
            if not isinstance(enzyme, dict):
                continue
            for key in ["protein", "protein_complex", "name"]:
                value = _canonical(str(enzyme.get(key, "")))
                if value:
                    enzyme_norms.add(_normalize(value))
                    break

        kept_inputs: List[str] = []
        for token in inputs:
            norm = _normalize(token)
            is_protein_token = norm in protein_norms or _is_protein_like(token, payload)
            if not is_protein_token:
                kept_inputs.append(token)
                continue
            if norm in scaffold_norms:
                kept_inputs.append(token)
                continue
            if norm in output_complex_parts:
                kept_inputs.append(token)
                continue
            if norm not in enzyme_norms:
                enzyme_key = "protein_complex" if ":" in token else "protein"
                enzymes.append({enzyme_key: token})
                enzyme_norms.add(norm)
                rep["summary"]["catalysts_promoted_to_enzymes"] += 1
                rep["actions"].append({"type": "catalyst_promoted_to_modifier", "json_pointer": f"/processes/reactions/{ridx}/inputs", "name": token})

        present_inputs = {_normalize(token) for token in kept_inputs}
        for out_token in outputs:
            parts = _complex_components(out_token)
            if len(parts) < 2:
                continue
            scaffold = parts[0]
            scaffold_norm = _normalize(scaffold)
            if scaffold_norm and scaffold_norm not in present_inputs and _is_protein_like(scaffold, payload):
                kept_inputs.append(scaffold)
                present_inputs.add(scaffold_norm)
                _ensure_protein(scaffold, payload, rep)
                rep["summary"]["scaffold_inputs_added"] += 1
                rep["actions"].append({"type": "scaffold_input_added", "json_pointer": f"/processes/reactions/{ridx}/inputs", "name": scaffold, "for_output_complex": out_token})

        reaction["inputs"] = _dedupe_preserve(kept_inputs)
        reaction["outputs"] = _dedupe_preserve(outputs)
        reaction["enzymes"] = enzymes
    return payload

def _reaction_modifier_names(reaction: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ["enzymes", "modifiers"]:
        for row in _safe_list(reaction.get(key)):
            if not isinstance(row, dict):
                continue
            for name_key in ["protein", "protein_complex", "name"]:
                value = _canonical(str(row.get(name_key, "")))
                if value:
                    names.append(value)
                    break
    return _dedupe_preserve(names)


def _evidence_length(row: Dict[str, Any]) -> int:
    score = len(_canonical(str(row.get("evidence", ""))))
    for key in ["enzymes", "transporters"]:
        for item in _safe_list(row.get(key)):
            if isinstance(item, dict):
                score += len(_canonical(str(item.get("evidence", ""))))
    return score


def _is_inferred(row: Dict[str, Any]) -> bool:
    if isinstance(row.get("inference"), dict):
        return True
    if bool(row.get("inferred", False)):
        return True
    for key in ["enzymes", "transporters"]:
        for item in _safe_list(row.get(key)):
            if isinstance(item, dict) and isinstance(item.get("inference"), dict):
                return True
    return False


def _best_record(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    def _score(item: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        inferred = 1 if _is_inferred(item) else 0
        observed = 1 if bool(item.get("observed", False)) else 0
        enzyme_evidence = 0
        for key in ["enzymes", "transporters"]:
            for row in _safe_list(item.get(key)):
                if isinstance(row, dict) and _canonical(str(row.get("evidence", ""))):
                    enzyme_evidence += 1
        evidence_len = _evidence_length(item)
        payload_len = len(json.dumps(item, ensure_ascii=False))
        return (1 - inferred, observed, enzyme_evidence, evidence_len, payload_len)

    return a if _score(a) >= _score(b) else b


def dedupe_processes(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    rep = report if isinstance(report, dict) else _new_report()
    reactions, transports = _process_lists(payload)

    reaction_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        essential = bool(reaction.get("essential", False))
        in_norm = [_normalize(v) for v in _safe_list(reaction.get("inputs")) if isinstance(v, str) and _canonical(v)]
        out_norm = [_normalize(v) for v in _safe_list(reaction.get("outputs")) if isinstance(v, str) and _canonical(v)]
        if not essential and in_norm and out_norm:
            if sorted(in_norm) == sorted(out_norm):
                rep["summary"]["no_op_removed_count"] += 1
                continue
            # Proteolysis/no-op style: all outputs already present in inputs, no novel produced token.
            if set(out_norm).issubset(set(in_norm)):
                rep["summary"]["no_op_removed_count"] += 1
                continue
        key = (
            "reaction",
            tuple(sorted(in_norm)),
            tuple(sorted(out_norm)),
            _normalize(str(reaction.get("biological_state", ""))),
            tuple(sorted(_normalize(v) for v in _reaction_modifier_names(reaction))),
        )
        existing = reaction_by_key.get(key)
        reaction_by_key[key] = reaction if existing is None else _best_record(existing, reaction)

    transport_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for transport in transports:
        if not isinstance(transport, dict):
            continue
        cargo_value = (
            transport.get("cargo_complex")
            if isinstance(transport.get("cargo_complex"), str) and _canonical(transport.get("cargo_complex"))
            else transport.get("cargo")
        )
        transporter_names = []
        for row in _safe_list(transport.get("transporters")):
            if not isinstance(row, dict):
                continue
            for key in ["protein", "protein_complex", "name"]:
                value = _canonical(str(row.get(key, "")))
                if value:
                    transporter_names.append(value)
                    break
        key = (
            "transport",
            _normalize(str(cargo_value or "")),
            _normalize(str(transport.get("from_biological_state", ""))),
            _normalize(str(transport.get("to_biological_state", ""))),
            tuple(sorted(_normalize(v) for v in transporter_names)),
        )
        existing = transport_by_key.get(key)
        transport_by_key[key] = transport if existing is None else _best_record(existing, transport)

    deduped_reactions = list(reaction_by_key.values())
    deduped_transports = list(transport_by_key.values())
    removed_reactions = max(0, len(reactions) - len(deduped_reactions))
    removed_transports = max(0, len(transports) - len(deduped_transports))

    processes = _safe_dict(payload.setdefault("processes", {}))
    processes["reactions"] = deduped_reactions
    processes["transports"] = deduped_transports

    rep["summary"]["dedupe_removed_reactions"] += removed_reactions
    rep["summary"]["dedupe_removed_transports"] += removed_transports
    rep["summary"]["dedupe_removed"] += removed_reactions + removed_transports
    rep["summary"]["dedupe_removed_total"] = rep["summary"]["dedupe_removed"]
    return {
        "reactions_removed": removed_reactions,
        "transports_removed": removed_transports,
        "dedupe_removed": removed_reactions + removed_transports,
        "no_op_removed_count": int(rep["summary"].get("no_op_removed_count", 0)),
    }


def validate_no_composites(payload: Dict[str, Any]) -> None:
    entities = _safe_dict(payload.get("entities"))
    processes = _safe_dict(payload.get("processes"))
    errors: List[str] = []

    for idx, row in enumerate(_safe_list(entities.get("compounds"))):
        if isinstance(row, dict) and _has_plus_token(_canonical(str(row.get("name", "")))):
            errors.append(f"/entities/compounds/{idx}/name has '+' token: {row.get('name', '')}")

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            for tidx, token in enumerate(_safe_list(reaction.get(side))):
                if isinstance(token, str) and _has_plus_token(token):
                    errors.append(f"/processes/reactions/{ridx}/{side}/{tidx} has '+' token: {token}")

    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        for key in ["cargo", "cargo_complex"]:
            token = transport.get(key)
            if isinstance(token, str) and _has_plus_token(token):
                errors.append(f"/processes/transports/{tidx}/{key} has '+' token: {token}")

    if errors:
        raise ValueError("Composite validation failed:\n" + "\n".join(errors[:40]))


def validate_registry_references(payload: Dict[str, Any]) -> None:
    compounds, proteins, complexes = _entity_lists(payload)
    registry = _entity_name_norms(compounds) | _entity_name_norms(proteins) | _entity_name_norms(complexes)
    processes = _safe_dict(payload.get("processes"))
    errors: List[str] = []

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            for tidx, token in enumerate(_safe_list(reaction.get(side))):
                if isinstance(token, str) and _canonical(token) and _normalize(token) not in registry:
                    errors.append(f"/processes/reactions/{ridx}/{side}/{tidx} unknown entity: {token}")
        for actor_key in ["enzymes", "modifiers"]:
            for eidx, enzyme in enumerate(_safe_list(reaction.get(actor_key))):
                if not isinstance(enzyme, dict):
                    continue
                enzyme_name = ""
                for key in ["protein", "protein_complex", "name"]:
                    candidate = _canonical(str(enzyme.get(key, "")))
                    if candidate:
                        enzyme_name = candidate
                        break
                if enzyme_name and _normalize(enzyme_name) not in registry:
                    errors.append(
                        f"/processes/reactions/{ridx}/{actor_key}/{eidx} unknown modifier: {enzyme_name}"
                    )

    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        cargo = transport.get("cargo_complex") if isinstance(transport.get("cargo_complex"), str) and _canonical(transport.get("cargo_complex")) else transport.get("cargo")
        if isinstance(cargo, str) and _canonical(cargo) and _normalize(cargo) not in registry:
            errors.append(f"/processes/transports/{tidx}/cargo unknown entity: {cargo}")
        for tridx, transporter in enumerate(_safe_list(transport.get("transporters"))):
            if not isinstance(transporter, dict):
                continue
            transporter_name = ""
            for key in ["protein", "protein_complex", "name"]:
                candidate = _canonical(str(transporter.get(key, "")))
                if candidate:
                    transporter_name = candidate
                    break
            if transporter_name and _normalize(transporter_name) not in registry:
                errors.append(
                    f"/processes/transports/{tidx}/transporters/{tridx} unknown transporter: {transporter_name}"
                )

    if errors:
        raise ValueError("Registry validation failed:\n" + "\n".join(errors[:40]))


def validate_no_scaffold_modifiers(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> None:
    rep = report if isinstance(report, dict) else _new_report()
    scaffold_norms = _scaffold_norms(payload)
    processes = _safe_dict(payload.get("processes"))
    errors: List[str] = []
    found = 0

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        for key in ["enzymes", "modifiers"]:
            for midx, row in enumerate(_safe_list(reaction.get(key))):
                if not isinstance(row, dict):
                    continue
                name = ""
                for field in ["protein", "protein_complex", "name"]:
                    candidate = _canonical(str(row.get(field, "")))
                    if candidate:
                        name = candidate
                        break
                if not name:
                    continue
                if _normalize(name) in scaffold_norms:
                    found += 1
                    errors.append(f"/processes/reactions/{ridx}/{key}/{midx} scaffold in modifier: {name}")
    rep["summary"]["scaffold_in_modifiers_count"] = found
    if errors:
        raise ValueError("Scaffold modifier validation failed:\n" + "\n".join(errors[:40]))


def compute_normalization_stats(payload: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    entities = _safe_dict(payload.get("entities"))
    processes = _safe_dict(payload.get("processes"))
    complexes = [
        _canonical(str(row.get("name", "")))
        for row in _safe_list(entities.get("protein_complexes"))
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    ]

    plus_remaining = 0
    for row in _safe_list(entities.get("compounds")):
        if isinstance(row, dict) and _has_plus_token(str(row.get("name", ""))):
            plus_remaining += 1
    for reaction in _safe_list(processes.get("reactions")):
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            plus_remaining += sum(
                1
                for token in _safe_list(reaction.get(side))
                if isinstance(token, str) and _has_plus_token(token)
            )
    for transport in _safe_list(processes.get("transports")):
        if not isinstance(transport, dict):
            continue
        for key in ["cargo", "cargo_complex"]:
            token = transport.get(key)
            if isinstance(token, str) and _has_plus_token(token):
                plus_remaining += 1

    rep["summary"]["n_plus_tokens_remaining"] = plus_remaining
    rep["summary"]["complexes_list"] = sorted(set(complexes))
    return _safe_dict(rep.get("summary"))


def run_strict_post_normalization_gates(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
    forbidden_complexes: Optional[Sequence[str]] = None,
    enforce_all_proteins_connected: bool = False,
) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    stats = compute_normalization_stats(payload, rep)
    errors: List[Dict[str, str]] = []
    forbidden_norms = {
        _normalize(_canonical(name))
        for name in (forbidden_complexes or ["thyroglobulin:2-aminoacrylic acid"])
        if _normalize(_canonical(name))
    }

    entities = _safe_dict(payload.get("entities"))
    processes = _safe_dict(payload.get("processes"))
    element_locations = _safe_dict(payload.get("element_locations"))
    proteins = _safe_list(entities.get("proteins"))
    complexes = _safe_list(entities.get("protein_complexes"))
    protein_pointer_by_norm = {
        _normalize(_canonical(str(row.get("name", "")))): f"/entities/proteins/{idx}/name"
        for idx, row in enumerate(proteins)
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    }
    protein_registry_norms = _entity_name_norms(proteins) | _entity_name_norms(complexes)

    def _add_error(path: str, reason: str) -> None:
        errors.append({"path": path, "reason": reason})

    def _check_forbidden(path: str, token: str) -> None:
        if _normalize(_canonical(token)) in forbidden_norms:
            _add_error(path, f"Forbidden complex reference detected: {token}")

    if int(stats.get("n_plus_tokens_remaining", 0)) != 0:
        _add_error(
            "/normalization_stats/n_plus_tokens_remaining",
            f"Expected 0 plus tokens, found {int(stats.get('n_plus_tokens_remaining', 0))}.",
        )

    for idx, row in enumerate(_safe_list(entities.get("compounds"))):
        if isinstance(row, dict):
            _check_forbidden(f"/entities/compounds/{idx}/name", str(row.get("name", "")))
    for idx, row in enumerate(_safe_list(entities.get("proteins"))):
        if isinstance(row, dict):
            _check_forbidden(f"/entities/proteins/{idx}/name", str(row.get("name", "")))
    for idx, row in enumerate(_safe_list(entities.get("protein_complexes"))):
        if isinstance(row, dict):
            _check_forbidden(f"/entities/protein_complexes/{idx}/name", str(row.get("name", "")))

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            for tidx, token in enumerate(_safe_list(reaction.get(side))):
                if isinstance(token, str):
                    _check_forbidden(f"/processes/reactions/{ridx}/{side}/{tidx}", token)
        for key in ["enzymes", "modifiers"]:
            for midx, actor in enumerate(_safe_list(reaction.get(key))):
                if not isinstance(actor, dict):
                    continue
                actor_name = ""
                for field in ["protein", "protein_complex", "name"]:
                    value = _canonical(str(actor.get(field, "")))
                    if value:
                        actor_name = value
                        break
                if not actor_name:
                    continue
                _check_forbidden(f"/processes/reactions/{ridx}/{key}/{midx}", actor_name)
                if _normalize(actor_name) not in protein_registry_norms:
                    _add_error(
                        f"/processes/reactions/{ridx}/{key}/{midx}",
                        f"Unknown protein/modifier reference: {actor_name}",
                    )

    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        for field in ["cargo_complex", "cargo"]:
            token = transport.get(field)
            if isinstance(token, str):
                _check_forbidden(f"/processes/transports/{tidx}/{field}", token)
        for tridx, actor in enumerate(_safe_list(transport.get("transporters"))):
            if not isinstance(actor, dict):
                continue
            actor_name = ""
            for field in ["protein", "protein_complex", "name"]:
                value = _canonical(str(actor.get(field, "")))
                if value:
                    actor_name = value
                    break
            if not actor_name:
                continue
            _check_forbidden(f"/processes/transports/{tidx}/transporters/{tridx}", actor_name)
            if _normalize(actor_name) not in protein_registry_norms:
                _add_error(
                    f"/processes/transports/{tidx}/transporters/{tridx}",
                    f"Unknown transporter reference: {actor_name}",
                )

    try:
        validate_no_composites(payload)
    except Exception as exc:  # noqa: BLE001
        _add_error("/processes", f"Composite validation failed: {exc}")
    try:
        validate_registry_references(payload)
    except Exception as exc:  # noqa: BLE001
        _add_error("/processes", f"Registry validation failed: {exc}")
    try:
        validate_no_scaffold_modifiers(payload, report=rep)
    except Exception as exc:  # noqa: BLE001
        _add_error("/processes/reactions/*/enzymes", f"Scaffold modifier validation failed: {exc}")

    from qa_graph import build_graph, connected_components, degrees, get_entities, node  # type: ignore

    adj, meta = build_graph(payload)
    comps = connected_components(adj)
    deg = degrees(adj)
    n_edges = sum(len(v) for v in adj.values()) // 2
    main_size = max((len(c) for c in comps), default=0)
    n_nodes = len(adj)
    ents = get_entities(payload)
    protein_nodes = [node("protein", name) for name in sorted(ents.get("proteins", set()))]
    proteins_degree0 = sum(1 for pname in protein_nodes if deg.get(pname, 0) == 0)
    proteins_total = len(protein_nodes)
    proteins_attached = max(0, proteins_total - proteins_degree0)
    connectivity = {
        **meta,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": len(comps),
        "main_component_size": main_size,
        "largest_component_pct": round((100.0 * main_size / n_nodes), 2) if n_nodes else 0.0,
        "n_isolated_nodes": sum(1 for _, d in deg.items() if d == 0),
        "proteins_degree0": proteins_degree0,
        "proteins_attached_pct": round((100.0 * proteins_attached / proteins_total), 2) if proteins_total else 100.0,
    }

    located_norm_to_pointers: Dict[str, List[str]] = {}
    for idx, prow in enumerate(_safe_list(element_locations.get("protein_locations"))):
        if not isinstance(prow, dict):
            continue
        pname = _canonical(str(prow.get("protein", "")))
        if not pname:
            continue
        norm = _normalize(pname)
        located_norm_to_pointers.setdefault(norm, []).append(f"/element_locations/protein_locations/{idx}/protein")

    for protein_name in sorted(ents.get("proteins", set())):
        pnode = node("protein", protein_name)
        if deg.get(pnode, 0) > 0:
            continue
        norm = _normalize(protein_name)
        if norm in located_norm_to_pointers:
            _add_error(
                located_norm_to_pointers[norm][0],
                f"Located protein is isolated in connectivity graph: {protein_name}",
            )

    if enforce_all_proteins_connected and proteins_degree0 > 0:
        for protein_name in sorted(ents.get("proteins", set())):
            pnode = node("protein", protein_name)
            if deg.get(pnode, 0) == 0:
                pnorm = _normalize(protein_name)
                _add_error(
                    protein_pointer_by_norm.get(pnorm, f"/entities/proteins/{protein_name}"),
                    f"Protein has degree 0 after normalization: {protein_name}",
                )

    details = {
        "normalization_stats": stats,
        "connectivity": connectivity,
        "errors": errors,
    }
    if errors:
        raise GateValidationError("Hard-gate validation failed after normalization.", details=details)
    return details


def normalize_process_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = deepcopy(payload)
    report = _new_report()
    normalize_composites(data, report=report)
    rewrite_reactions_to_complex_states(data, report=report)
    cleanup_disallowed_complexes(data, report=report)
    ensure_autostates(data, report=report)
    attach_transporters_from_evidence(data, report=report)
    promote_catalysts(data, report=report)
    normalize_process_actor_schema(data, report=report)
    dedupe_processes(data, report=report)
    run_strict_post_normalization_gates(data, report=report, enforce_all_proteins_connected=True)
    return data, report
