from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (value or "").strip().casefold()))


def _canonical(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _split_composite(value: str) -> List[str]:
    text = _canonical(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"\s*\+\s*|\s+and\s+", text, flags=re.IGNORECASE) if part.strip()]


def _is_composite(value: str) -> bool:
    text = _canonical(value)
    if not text:
        return False
    return bool(re.search(r"\s\+\s|\sand\s", text, flags=re.IGNORECASE))


def _protein_like_token(token: str, known_proteins_norm: set, known_complexes_norm: set) -> bool:
    norm = _normalize(token)
    if norm in known_proteins_norm or norm in known_complexes_norm:
        return True
    return bool(
        re.search(
            r"(protein|globulin|peroxidase|symporter|deiodinase|atpase|enzyme|receptor|transporter)",
            norm,
            flags=re.IGNORECASE,
        )
    )


def _dedupe_preserve(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for value in values:
        c = _canonical(value)
        n = _normalize(c)
        if not c or not n or n in seen:
            continue
        seen.add(n)
        out.append(c)
    return out


def normalize_process_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = deepcopy(payload)
    entities = _safe_dict(data.setdefault("entities", {}))
    processes = _safe_dict(data.setdefault("processes", {}))
    if not isinstance(entities.get("compounds"), list):
        entities["compounds"] = []
    if not isinstance(entities.get("proteins"), list):
        entities["proteins"] = []
    if not isinstance(entities.get("protein_complexes"), list):
        entities["protein_complexes"] = []
    if not isinstance(processes.get("reactions"), list):
        processes["reactions"] = []
    if not isinstance(processes.get("transports"), list):
        processes["transports"] = []

    compounds = _safe_list(entities.get("compounds"))
    proteins = _safe_list(entities.get("proteins"))
    complexes = _safe_list(entities.get("protein_complexes"))

    compound_norm_to_name: Dict[str, str] = {}
    protein_norm_to_name: Dict[str, str] = {}
    complex_norm_to_name: Dict[str, str] = {}
    complex_norm_to_components: Dict[str, List[str]] = {}

    for row in compounds:
        if isinstance(row, dict) and isinstance(row.get("name"), str) and row.get("name", "").strip():
            name = _canonical(row["name"])
            compound_norm_to_name.setdefault(_normalize(name), name)
    for row in proteins:
        if isinstance(row, dict) and isinstance(row.get("name"), str) and row.get("name", "").strip():
            name = _canonical(row["name"])
            protein_norm_to_name.setdefault(_normalize(name), name)
    for row in complexes:
        if not isinstance(row, dict):
            continue
        name = _canonical(str(row.get("name", "")))
        if not name:
            continue
        norm = _normalize(name)
        complex_norm_to_name.setdefault(norm, name)
        parts = _safe_list(row.get("components"))
        comp_parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
        if comp_parts:
            complex_norm_to_components[norm] = comp_parts

    report: Dict[str, Any] = {
        "summary": {
            "complexes_created": 0,
            "composite_tokens_rewritten": 0,
            "composite_tokens_split": 0,
            "transport_rows_split": 0,
            "catalysts_promoted_to_enzymes": 0,
            "scaffold_inputs_added": 0,
            "entities_added_as_compounds": 0,
        },
        "actions": [],
    }

    def ensure_compound(name: str) -> str:
        c = _canonical(name)
        n = _normalize(c)
        if not c:
            return ""
        existing = compound_norm_to_name.get(n) or protein_norm_to_name.get(n) or complex_norm_to_name.get(n)
        if existing:
            return existing
        compounds.append({"name": c})
        compound_norm_to_name[n] = c
        report["summary"]["entities_added_as_compounds"] += 1
        report["actions"].append({"type": "entity_added_compound", "name": c})
        return c

    def ensure_complex(parts: Sequence[str], *, pointer: str) -> str:
        clean_parts = _dedupe_preserve(parts)
        if len(clean_parts) < 2:
            return clean_parts[0] if clean_parts else ""
        name = ":".join(clean_parts)
        norm = _normalize(name)
        existing = complex_norm_to_name.get(norm)
        if existing:
            return existing
        for part in clean_parts:
            ensure_compound(part)
        row = {"name": name, "components": clean_parts}
        complexes.append(row)
        complex_norm_to_name[norm] = name
        complex_norm_to_components[norm] = clean_parts
        report["summary"]["complexes_created"] += 1
        report["actions"].append(
            {
                "type": "complex_created",
                "json_pointer": pointer,
                "name": name,
                "components": clean_parts,
            }
        )
        return name

    def should_materialize(parts: Sequence[str]) -> bool:
        if len(parts) < 2:
            return False
        kp = set(protein_norm_to_name.keys())
        kc = set(complex_norm_to_name.keys())
        return any(_protein_like_token(part, kp, kc) for part in parts)

    def rewrite_tokens(tokens: Sequence[Any], *, pointer: str) -> List[str]:
        out: List[str] = []
        for token in tokens:
            if not isinstance(token, str):
                continue
            t = _canonical(token)
            if not t:
                continue
            if _is_composite(t):
                parts = _split_composite(t)
                if should_materialize(parts):
                    cname = ensure_complex(parts, pointer=pointer)
                    if cname:
                        out.append(cname)
                        report["summary"]["composite_tokens_rewritten"] += 1
                        report["actions"].append(
                            {
                                "type": "composite_rewritten_to_complex",
                                "json_pointer": pointer,
                                "from": t,
                                "to": cname,
                            }
                        )
                else:
                    out.extend(parts)
                    report["summary"]["composite_tokens_split"] += 1
                    report["actions"].append(
                        {
                            "type": "composite_split",
                            "json_pointer": pointer,
                            "from": t,
                            "to": parts,
                        }
                    )
            else:
                out.append(t)
        return out

    reactions = _safe_list(processes.get("reactions"))
    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        iptr = f"/processes/reactions/{ridx}/inputs"
        optr = f"/processes/reactions/{ridx}/outputs"
        inputs = rewrite_tokens(_safe_list(reaction.get("inputs")), pointer=iptr)
        outputs = rewrite_tokens(_safe_list(reaction.get("outputs")), pointer=optr)
        reaction["inputs"] = inputs
        reaction["outputs"] = outputs

        enz_list = _safe_list(reaction.get("enzymes"))
        enzyme_names_norm: set = set()
        for enz in enz_list:
            if not isinstance(enz, dict):
                continue
            for key in ["protein", "protein_complex", "name"]:
                val = _canonical(str(enz.get(key, "")))
                if val:
                    enzyme_names_norm.add(_normalize(val))
                    break

        new_inputs: List[str] = []
        for token in _safe_list(reaction.get("inputs")):
            tok = _canonical(str(token))
            ntok = _normalize(tok)
            if ntok in protein_norm_to_name or ntok in complex_norm_to_name:
                if ntok not in enzyme_names_norm:
                    if ntok in complex_norm_to_name:
                        enz_list.append({"protein_complex": tok})
                    else:
                        enz_list.append({"protein": tok})
                    enzyme_names_norm.add(ntok)
                    report["summary"]["catalysts_promoted_to_enzymes"] += 1
                    report["actions"].append(
                        {
                            "type": "catalyst_promoted",
                            "json_pointer": iptr,
                            "name": tok,
                        }
                    )
                continue
            new_inputs.append(tok)
        reaction["inputs"] = new_inputs
        reaction["enzymes"] = enz_list

        input_norms = {_normalize(x) for x in _safe_list(reaction.get("inputs")) if isinstance(x, str)}
        input_complex_parts: set = set()
        for token in _safe_list(reaction.get("inputs")):
            ntok = _normalize(str(token))
            for part in complex_norm_to_components.get(ntok, []):
                input_complex_parts.add(_normalize(part))
        for token in _safe_list(reaction.get("outputs")):
            ntok = _normalize(str(token))
            parts = complex_norm_to_components.get(ntok, [])
            for part in parts:
                npart = _normalize(part)
                if npart in input_norms or npart in input_complex_parts:
                    continue
                if _protein_like_token(part, set(protein_norm_to_name.keys()), set(complex_norm_to_name.keys())):
                    reaction["inputs"].append(part)
                    input_norms.add(npart)
                    report["summary"]["scaffold_inputs_added"] += 1
                    report["actions"].append(
                        {
                            "type": "scaffold_input_added",
                            "json_pointer": iptr,
                            "name": part,
                            "for_output_complex": token,
                        }
                    )

    new_transports: List[Dict[str, Any]] = []
    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        row = deepcopy(transport)
        cargo_raw = _canonical(str(row.get("cargo_complex") or row.get("cargo") or ""))
        ptr = f"/processes/transports/{tidx}/cargo"
        if cargo_raw and _is_composite(cargo_raw):
            parts = _split_composite(cargo_raw)
            if should_materialize(parts):
                cname = ensure_complex(parts, pointer=ptr)
                row["cargo"] = cname
                row["cargo_complex"] = cname
                report["summary"]["composite_tokens_rewritten"] += 1
                report["actions"].append(
                    {
                        "type": "transport_cargo_rewritten_to_complex",
                        "json_pointer": ptr,
                        "from": cargo_raw,
                        "to": cname,
                    }
                )
                new_transports.append(row)
            else:
                first = True
                for part in parts:
                    cloned = deepcopy(row)
                    cloned["cargo"] = part
                    cloned.pop("cargo_complex", None)
                    new_transports.append(cloned)
                    if not first:
                        report["summary"]["transport_rows_split"] += 1
                    first = False
                report["summary"]["composite_tokens_split"] += 1
                report["actions"].append(
                    {
                        "type": "transport_cargo_split",
                        "json_pointer": ptr,
                        "from": cargo_raw,
                        "to": parts,
                    }
                )
        else:
            cargo_norm = _normalize(cargo_raw)
            if cargo_norm in complex_norm_to_name:
                row["cargo"] = complex_norm_to_name[cargo_norm]
                row["cargo_complex"] = complex_norm_to_name[cargo_norm]
            elif cargo_raw:
                row["cargo"] = cargo_raw
            new_transports.append(row)

    processes["transports"] = new_transports
    return data, report
