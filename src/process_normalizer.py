
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


PROTEIN_LIKE_RE = re.compile(
    r"(protein|globulin|peroxidase|symporter|deiodinase|atpase|enzyme|receptor|transporter|kinase|phosphatase)",
    flags=re.IGNORECASE,
)


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
            "entities_moved_out_of_compounds": 0,
            "entities_added_as_compounds": 0,
            "entities_added_as_proteins": 0,
            "catalysts_promoted_to_enzymes": 0,
            "scaffold_inputs_added": 0,
            "dedupe_removed_reactions": 0,
            "dedupe_removed_transports": 0,
            "dedupe_removed": 0,
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
    if norm in _protein_like_norms(payload):
        return True
    return bool(PROTEIN_LIKE_RE.search(name))

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


def materialize_complex(
    nameA: str,
    nameB: str,
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
    extra_components: Optional[Sequence[str]] = None,
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
        return complex_name

    complexes.append({"name": complex_name, "components": clean_parts})
    _dedupe_named_rows(complexes)
    rep["summary"]["complexes_created"] += 1
    rep["actions"].append({"type": "complex_created", "name": complex_name, "components": clean_parts})
    return complex_name


def _rewrite_token(
    token: str,
    payload: Dict[str, Any],
    report: Dict[str, Any],
    rewrite_map: Dict[str, str],
    pointer: str,
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
            complex_name = materialize_complex(parts[0], parts[1], payload, report=report, extra_components=parts[2:])
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

    if any(_is_protein_like(part, payload) for part in parts):
        complex_name = materialize_complex(parts[0], parts[1], payload, report=report, extra_components=parts[2:])
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

    out: List[str] = []
    for part in parts:
        c_part = _canonical(part)
        if not c_part:
            continue
        _ensure_compound(c_part, payload, report)
        out.append(c_part)
    report["summary"]["composites_rewritten"] += 1
    report["actions"].append({"type": "composite_split", "json_pointer": pointer, "from": text, "to": out})
    return out


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
        for side in ["inputs", "outputs"]:
            pointer = f"/processes/reactions/{ridx}/{side}"
            new_tokens: List[str] = []
            for token in _safe_list(reaction.get(side)):
                if not isinstance(token, str):
                    continue
                rewritten = _rewrite_token(token, payload, rep, rewrite_map, pointer)
                if [_canonical(token)] != rewritten:
                    changed = True
                new_tokens.extend(rewritten)
            new_tokens = _dedupe_preserve(new_tokens)
            if side == "outputs":
                collapsed = _collapse_reaction_outputs(new_tokens)
                if collapsed != new_tokens:
                    changed = True
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
        cargo_raw = _canonical(
            str(
                row.get("cargo_complex") if isinstance(row.get("cargo_complex"), str) else row.get("cargo") or ""
            )
        )
        if not cargo_raw:
            rewritten_transports.append(row)
            continue

        rewritten = _rewrite_token(cargo_raw, payload, rep, rewrite_map, pointer)
        if len(rewritten) == 1:
            cargo_value = rewritten[0]
            row["cargo"] = cargo_value
            if ":" in cargo_value:
                row["cargo_complex"] = cargo_value
            else:
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
                canonical = materialize_complex(parts[0], parts[1], payload, report=rep, extra_components=parts[2:])
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
            if len(parts) >= 2 and any(_is_protein_like(part, payload) for part in parts):
                canonical = materialize_complex(parts[0], parts[1], payload, report=rep, extra_components=parts[2:])
                rep["rewrite_map"][_composite_key(name)] = canonical
                rep["rewrite_map"][_normalize(name)] = canonical
                rep["summary"]["entities_moved_out_of_compounds"] += 1
                rep["summary"]["composites_rewritten"] += 1
                continue
            for part in _dedupe_preserve(parts):
                _ensure_compound(part, payload, rep)
            rep["summary"]["composites_rewritten"] += 1
            continue
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


def promote_catalysts(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    reactions, _ = _process_lists(payload)
    _, proteins, complexes = _entity_lists(payload)
    protein_norms = _entity_name_norms(proteins) | _entity_name_norms(complexes)

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
    def _score(item: Dict[str, Any]) -> Tuple[int, int, int, int]:
        inferred = 1 if _is_inferred(item) else 0
        observed = 1 if bool(item.get("observed", False)) else 0
        evidence_len = _evidence_length(item)
        payload_len = len(json.dumps(item, ensure_ascii=False))
        return (1 - inferred, observed, evidence_len, payload_len)

    return a if _score(a) >= _score(b) else b


def dedupe_processes(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    rep = report if isinstance(report, dict) else _new_report()
    reactions, transports = _process_lists(payload)

    reaction_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        key = (
            _normalize(str(reaction.get("name", ""))),
            tuple(sorted(_normalize(v) for v in _safe_list(reaction.get("inputs")) if isinstance(v, str) and _canonical(v))),
            tuple(sorted(_normalize(v) for v in _safe_list(reaction.get("outputs")) if isinstance(v, str) and _canonical(v))),
            _normalize(str(reaction.get("biological_state", ""))),
            tuple(sorted(_normalize(v) for v in _reaction_modifier_names(reaction))),
            _normalize(str(reaction.get("type", "reaction"))),
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
            _normalize(str(cargo_value or "")),
            _normalize(str(transport.get("from_biological_state", ""))),
            _normalize(str(transport.get("to_biological_state", ""))),
            tuple(sorted(_normalize(v) for v in transporter_names)),
            _normalize(str(transport.get("type", "transport"))),
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
    return {"reactions_removed": removed_reactions, "transports_removed": removed_transports, "dedupe_removed": removed_reactions + removed_transports}


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
        for eidx, enzyme in enumerate(_safe_list(reaction.get("enzymes"))):
            if not isinstance(enzyme, dict):
                continue
            enzyme_name = ""
            for key in ["protein", "protein_complex", "name"]:
                candidate = _canonical(str(enzyme.get(key, "")))
                if candidate:
                    enzyme_name = candidate
                    break
            if enzyme_name and _normalize(enzyme_name) not in registry:
                errors.append(f"/processes/reactions/{ridx}/enzymes/{eidx} unknown modifier: {enzyme_name}")

    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        cargo = transport.get("cargo_complex") if isinstance(transport.get("cargo_complex"), str) and _canonical(transport.get("cargo_complex")) else transport.get("cargo")
        if isinstance(cargo, str) and _canonical(cargo) and _normalize(cargo) not in registry:
            errors.append(f"/processes/transports/{tidx}/cargo unknown entity: {cargo}")

    if errors:
        raise ValueError("Registry validation failed:\n" + "\n".join(errors[:40]))


def normalize_process_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = deepcopy(payload)
    report = _new_report()
    normalize_composites(data, report=report)
    promote_catalysts(data, report=report)
    dedupe_processes(data, report=report)
    validate_no_composites(data)
    validate_registry_references(data)
    return data, report
