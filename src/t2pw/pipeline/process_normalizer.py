
from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from t2pw.pipeline.entity_identity import (
    has_protein_external_identity,
    is_generated_complex_wrapper,
    protein_external_identity,
    protein_species_context,
    route_entity_for_mapping,
)


# ---------------------------------------------------------------------------
# Biochemical alias map — collapses common synonyms to one canonical name
# before graph building so variants don't create phantom duplicate nodes.
# Keys are the result of .strip().casefold(); values are canonical display names.
# ---------------------------------------------------------------------------
BIOCHEMICAL_ALIAS_MAP: Dict[str, str] = {
    # Acetyl-CoA variants
    "acetyl-coa": "Acetyl-CoA",
    "acetyl coa": "Acetyl-CoA",
    "acetyl coenzyme a": "Acetyl-CoA",
    "acetyl-coenzyme a": "Acetyl-CoA",
    # CoA variants
    "coa-sh": "CoA-SH",
    "hscoa": "CoA-SH",
    "coenzyme a": "CoA-SH",
    # NAD+/NADH
    "nad": "NAD+",
    "nad+": "NAD+",
    "nadh": "NADH",
    # NADP+/NADPH
    "nadp": "NADP+",
    "nadp+": "NADP+",
    "nadph": "NADPH",
    # FAD/FADH2
    "fad": "FAD",
    "fadh2": "FADH2",
    "fadh₂": "FADH2",
    # TCA cycle metabolites
    "oxaloacetic acid": "oxaloacetate",
    "alpha-ketoglutarate": "α-ketoglutarate",
    "alpha ketoglutarate": "α-ketoglutarate",
    "2-oxoglutarate": "α-ketoglutarate",
    "2 oxoglutarate": "α-ketoglutarate",
    "succinic acid": "succinate",
    "fumaric acid": "fumarate",
    "malic acid": "malate",
    "citric acid": "citrate",
    "isocitric acid": "isocitrate",
    "pyruvic acid": "pyruvate",
    # Phosphate / nucleotides
    "phosphoenolpyruvate": "phosphoenolpyruvate",
    "pep": "phosphoenolpyruvate",
    "atp": "ATP",
    "adp": "ADP",
    "amp": "AMP",
    "gtp": "GTP",
    "gdp": "GDP",
    # Other
    "pi": "Pi",
    "ppi": "PPi",
    "inorganic phosphate": "Pi",
    "inorganic pyrophosphate": "PPi",
}


def force_reactions_non_spontaneous(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Spontaneity is not modeled currently; every reaction is forced to spontaneous=false."""
    processes = _safe_dict(payload.get("processes"))
    for row in _safe_list(processes.get("reactions")):
        if isinstance(row, dict):
            row["spontaneous"] = False
    return payload


def apply_biochemical_aliases(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Rewrite compound (and protein) names that are known biochemical synonyms to
    their canonical form, using BIOCHEMICAL_ALIAS_MAP.

    This runs *before* other normalization steps so that downstream passes see
    consistent names and don't create phantom duplicate nodes for e.g.
    "Acetyl-CoA" vs "acetyl coa" vs "Acetyl CoA".

    Only the entity registry rows and reaction input/output token lists are
    rewritten here; full pointer-based rewriting is left to the existing
    rewrite_process_references / canonicalize_same_as_aliases passes.
    """
    rep = report if isinstance(report, dict) else _new_report()
    rewrites: List[Dict[str, Any]] = []

    def _alias(name: str) -> str:
        key = _canonical(name).casefold()
        return BIOCHEMICAL_ALIAS_MAP.get(key, name)

    def _alias_list(rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            if not isinstance(row, dict):
                continue
            orig = _canonical(str(row.get("name", "")))
            canon = _alias(orig)
            if canon != orig:
                row["name"] = canon
                rewrites.append({"from": orig, "to": canon})

    entities = _safe_dict(payload.get("entities", {}))
    _alias_list(_safe_list(entities.get("compounds")))
    _alias_list(_safe_list(entities.get("proteins")))

    # Rewrite tokens in reaction inputs / outputs
    processes = _safe_dict(payload.get("processes", {}))
    for rxn in _safe_list(processes.get("reactions")):
        if not isinstance(rxn, dict):
            continue
        for side in ("inputs", "outputs"):
            tokens = rxn.get(side)
            if not isinstance(tokens, list):
                continue
            rxn[side] = [_alias(t) if isinstance(t, str) else t for t in tokens]

    if rewrites:
        rep.setdefault("actions", []).append({
            "type": "biochemical_alias_rewrite",
            "count": len(rewrites),
            "rewrites": rewrites[:30],  # cap log size
        })

    return payload


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
BIOCHEMICAL_COLON_RE = re.compile(r"(?<![A-Za-z0-9])\d+\s*:\s*\d+(?![A-Za-z0-9])")


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


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_plus_token(value: str) -> bool:
    # Strip trailing charge notation (e.g. NAD+, H+, Ca2+) before checking
    # for a composite "+" separator so chemical names aren't mis-parsed.
    stripped = re.sub(r"\+$", "", _canonical(value).rstrip())
    stripped = re.sub(r"\d+\+$", "", stripped.rstrip())
    return "+" in stripped


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
            "interaction_enzymes_promoted": 0,
            "scaffold_inputs_added": 0,
            "scaffold_in_modifiers_count": 0,
            "n_plus_tokens_remaining": 0,
            "complexes_list": [],
            "n_autostate_created": 0,
            "n_entities_assigned_to_autostate": 0,
            "transporters_attached": 0,
            "enzymes_attached_from_reaction_evidence": 0,
            "modifier_refs_canonicalized": 0,
            "modifier_refs_dropped": 0,
            "forbidden_complexes_removed": 0,
            "dedupe_removed_reactions": 0,
            "dedupe_removed_transports": 0,
            "dedupe_removed": 0,
            "dedupe_removed_total": 0,
            "no_op_removed_count": 0,
            "no_op_quarantined_count": 0,
            "locked_no_op_quarantined_count": 0,
            "evidenced_no_op_quarantined_count": 0,
            "unsupported_no_op_dropped_count": 0,
            "duplicate_locked_reactions_quarantined_count": 0,
            "n_same_as_groups": 0,
            "n_aliases_rewritten": 0,
            "n_entities_deduped": 0,
            "n_single_protein_complexes_removed": 0,
            "unresolved_complex_components_dropped": 0,
            "component_only_proteins_removed": 0,
            "pruned_disconnected_proteins": [],
            "pruned_disconnected_proteins_count": 0,
            "alias_example_mappings": [],
        },
        "rewrite_map": {},
        "actions": [],
    }


def _species_confidence(row: Dict[str, Any]) -> float:
    meta = _safe_dict(row.get("mapping_meta"))
    species_resolution = _safe_dict(meta.get("species_resolution"))
    mapping_resolution = _safe_dict(meta.get("resolution"))
    return max(
        _to_float(row.get("confidence")),
        _to_float(species_resolution.get("confidence")),
        _to_float(mapping_resolution.get("confidence")),
    )


def _select_default_species_name(entities: Dict[str, Any]) -> str:
    species_rows: List[Tuple[str, Dict[str, Any], int]] = []
    seen_species: Set[str] = set()
    for index, row in enumerate(_safe_list(entities.get("species"))):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            continue
        name = _canonical(row["name"])
        norm = _normalize(name)
        if not name or not norm or norm in seen_species:
            continue
        seen_species.add(norm)
        species_rows.append((name, row, index))
    if not species_rows:
        return ""
    if len(species_rows) == 1:
        return species_rows[0][0]

    usage_counts: Dict[str, int] = {}
    for bucket in ("proteins", "protein_complexes"):
        for item in _safe_list(entities.get(bucket)):
            if not isinstance(item, dict):
                continue
            species = (
                item.get("species")
                or item.get("organism")
                or item.get("species_name")
                or _safe_dict(item.get("species_ref")).get("name")
                or ""
            )
            if not isinstance(species, str):
                continue
            norm = _normalize(species)
            if norm:
                usage_counts[norm] = usage_counts.get(norm, 0) + 1

    def score(entry: Tuple[str, Dict[str, Any], int]) -> Tuple[float, int, int, int]:
        name, row, index = entry
        norm = _normalize(name)
        has_db_id = int(bool(row.get("pathbank_species_id") or row.get("species_id") or row.get("taxonomy_id")))
        return (_species_confidence(row), usage_counts.get(norm, 0), has_db_id, -index)

    return max(species_rows, key=score)[0]


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
    def _is_blank(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, dict):
            return len(value) == 0
        return False

    out = deepcopy(primary)
    for key, value in extra.items():
        if key == "name":
            continue
        if key not in out:
            out[key] = deepcopy(value)
            continue
        if key == "evidence":
            current = out.get(key)
            if _is_blank(current) and not _is_blank(value):
                out[key] = deepcopy(value)
            elif isinstance(current, str) and isinstance(value, str) and len(value.strip()) > len(current.strip()):
                out[key] = value
            continue
        if key == "mapped_ids" and isinstance(out.get(key), dict) and isinstance(value, dict):
            merged = dict(value)
            merged.update(out[key])
            out[key] = merged
            continue
        if isinstance(out.get(key), list) and isinstance(value, list):
            if all(isinstance(v, str) for v in out.get(key, [])) and all(isinstance(v, str) for v in value):
                out[key] = _dedupe_preserve([str(v) for v in out.get(key, [])] + [str(v) for v in value])
            elif _is_blank(out.get(key)) and not _is_blank(value):
                out[key] = deepcopy(value)
            continue
        if _is_blank(out.get(key)) and not _is_blank(value):
            out[key] = deepcopy(value)
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
    if _is_biochemical_colon_name(name):
        return False
    norm = _normalize(name)
    if not norm:
        return False
    protein_like_set = _protein_like_norms(payload)
    if norm in protein_like_set:
        return True
    try:
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
        parts = _complex_components(name, payload=payload, assume_complex=True)
        if len(parts) < 2:
            continue
        first = _canonical(parts[0])
        if first:
            scaffolds.add(_normalize(first))
    return scaffolds

def _ensure_protein(name: str, payload: Dict[str, Any], report: Dict[str, Any]) -> str:
    c_name = _canonical(name)
    if not c_name:
        return ""
    compounds, proteins, complexes = _entity_lists(payload)
    existing_complex = _find_entity_row(complexes, c_name)
    if existing_complex is not None:
        return _canonical(str(existing_complex.get("name", c_name)))
    if _find_entity_row(proteins, c_name) is None:
        # Carry class/confidence/provenance/source_refs from compound row if present
        existing = _find_entity_row(compounds, c_name) or {}
        new_row: Dict[str, Any] = {"name": c_name}
        for key in ("class", "confidence", "provenance", "source_refs"):
            if key in existing:
                new_row[key] = deepcopy(existing[key])
        proteins.append(new_row)
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
        compounds.append({"name": c_name, "class": "compound", "confidence": 0.8, "provenance": "inferred"})
        report["summary"]["entities_added_as_compounds"] += 1
        report["actions"].append({"type": "entity_added_compound", "name": c_name})
    _dedupe_named_rows(compounds)
    return c_name


def _colon_parts(name: str) -> List[str]:
    text = _canonical(name)
    if ":" not in text:
        return []
    return [part.strip() for part in text.split(":") if part.strip()]


def _is_biochemical_colon_name(name: str) -> bool:
    text = _canonical(name)
    if ":" not in text:
        return False
    lowered = text.casefold()
    if re.search(r"\(\s*\d+\s*:\s*\d+\s*\)", lowered):
        return True
    if re.search(r"\b[A-Za-z][A-Za-z0-9]*-\d+\s*:\s*\d+-CoA\b", text, flags=re.IGNORECASE):
        return True
    return bool(BIOCHEMICAL_COLON_RE.search(text))


def _known_complex_norms(payload: Dict[str, Any]) -> Set[str]:
    entities = _safe_dict(payload.get("entities"))
    return {
        _normalize(_canonical(str(row.get("name", ""))))
        for row in _safe_list(entities.get("protein_complexes"))
        if isinstance(row, dict) and _canonical(str(row.get("name", ""))) and not _is_biochemical_colon_name(str(row.get("name", "")))
    }


def _is_known_complex_name(name: str, payload: Dict[str, Any]) -> bool:
    norm = _normalize(_canonical(name))
    return bool(norm and norm in _known_complex_norms(payload))


def _is_explicit_complex_colon_syntax(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    assume_complex: bool = False,
) -> bool:
    text = _canonical(name)
    if ":" not in text or _is_biochemical_colon_name(text):
        return False
    parts = _colon_parts(text)
    if len(parts) < 2:
        return False
    if assume_complex:
        return True
    if re.search(r"\bcomplex\b", text, flags=re.IGNORECASE):
        return True
    if any(PROTEIN_LIKE_RE.search(part) for part in parts):
        return True
    if payload is not None:
        if _is_known_complex_name(text, payload):
            return True
        if _is_protein_like(parts[0], payload):
            return True
    return False


def _complex_components(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    assume_complex: bool = False,
) -> List[str]:
    if not _is_explicit_complex_colon_syntax(name, payload, assume_complex=assume_complex):
        return []
    return _colon_parts(name)


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
    source = _canonical(source_token)
    if source and ":" in source and not _is_explicit_complex_colon_syntax(source, payload):
        _ensure_compound(source, payload, rep)
        return source

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

    complexes.append({"name": complex_name, "components": clean_parts, "class": "protein_complex", "confidence": 0.8, "provenance": "inferred"})
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
        existing_complex = _find_entity_row(_entity_lists(payload)[2], text)
        if existing_complex is not None:
            return [_canonical(str(existing_complex.get("name", text)))]
        if len(_complex_components(text, payload=payload)) >= 2:
            parts = _complex_components(text, payload=payload)
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
            if _complex_components(token, payload=payload) or _is_protein_like(token, payload):
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
        if _is_biochemical_colon_name(name):
            _remove_entity(complexes, name)
            _ensure_compound(name, payload, rep)
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
        parts = _complex_components(name, payload=payload, assume_complex=True)
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
                if isinstance(token, str) and len(_complex_components(token, payload=payload)) >= 2:
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
        if _is_biochemical_colon_name(name):
            summary["forbidden_complexes_removed"] += 1
            rep["actions"].append(
                {
                    "type": "biochemical_colon_complex_removed",
                    "json_pointer": f"/entities/protein_complexes/{idx}/name",
                    "name": name,
                }
            )
            _remove_entity(complexes, name)
            _ensure_compound(name, payload, rep)
            continue
        parts = _complex_components(name, payload=payload, assume_complex=True)
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


class _AliasUnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, item: str) -> None:
        key = _normalize(item)
        if key and key not in self.parent:
            self.parent[key] = key

    def find(self, item: str) -> str:
        key = _normalize(item)
        if not key:
            return ""
        self.add(key)
        root = self.parent[key]
        while root != self.parent[root]:
            root = self.parent[root]
        current = key
        while current != root:
            nxt = self.parent[current]
            self.parent[current] = root
            current = nxt
        return root

    def union(self, left: str, right: str) -> None:
        lnorm = _normalize(left)
        rnorm = _normalize(right)
        if not lnorm or not rnorm:
            return
        lroot = self.find(lnorm)
        rroot = self.find(rnorm)
        if not lroot or not rroot or lroot == rroot:
            return
        if lroot < rroot:
            self.parent[rroot] = lroot
        else:
            self.parent[lroot] = rroot

    def groups(self) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for key in list(self.parent.keys()):
            root = self.find(key)
            if not root:
                continue
            out.setdefault(root, set()).add(key)
        return out


def _is_same_as_relationship(value: str) -> bool:
    norm = _normalize(value).replace(" ", "")
    return norm in {"sameas", "same_as"}


def _actor_name_from_row(row: Any) -> str:
    if isinstance(row, str):
        return _canonical(row)
    if not isinstance(row, dict):
        return ""
    for field in ["entity", "protein", "protein_name", "protein_complex", "enzyme", "modifier", "name"]:
        value = _canonical(str(row.get(field, "")))
        if value:
            return value
    return ""


def _component_name_from_row(row: Any) -> str:
    if isinstance(row, str):
        return _canonical(row)
    if not isinstance(row, dict):
        return ""
    for field in ["name", "protein", "entity", "protein_name"]:
        value = _canonical(str(row.get(field, "")))
        if value:
            return value
    return ""


def _rewrite_component_rows(
    rows: List[Any],
    rewrite_name: Callable[[str], str],
) -> List[Any]:
    rewritten_rows: List[Any] = []
    seen_norms: Set[str] = set()
    for row in rows:
        name = _component_name_from_row(row)
        if not name:
            continue
        rewritten_name = rewrite_name(name)
        norm = _normalize(rewritten_name)
        if not rewritten_name or not norm or norm in seen_norms:
            continue
        seen_norms.add(norm)
        if isinstance(row, dict):
            updated = deepcopy(row)
            for field in ["name", "protein", "entity", "protein_name"]:
                if _canonical(str(updated.get(field, ""))):
                    updated[field] = rewritten_name
                    break
            else:
                updated["name"] = rewritten_name
            rewritten_rows.append(updated)
        else:
            rewritten_rows.append(rewritten_name)
    return rewritten_rows


def drop_unresolved_complex_component_proteins(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Drop synthetic component-only proteins that cannot satisfy export identity gates."""
    rep = report if isinstance(report, dict) else _new_report()
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("unresolved_complex_components_dropped", 0)
    summary.setdefault("component_only_proteins_removed", 0)
    rep.setdefault("actions", [])

    entities = _safe_dict(payload.setdefault("entities", {}))
    proteins = _safe_list(entities.get("proteins"))
    complexes = _safe_list(entities.get("protein_complexes"))
    processes = _safe_dict(payload.get("processes"))

    process_ref_norms: Set[str] = set()

    def remember_ref(value: Any) -> None:
        name = _canonical(str(value or ""))
        norm = _normalize(name)
        if norm:
            process_ref_norms.add(norm)

    for reaction in _safe_list(processes.get("reactions")):
        if not isinstance(reaction, dict):
            continue
        for side in ("inputs", "outputs"):
            for token in _safe_list(reaction.get(side)):
                remember_ref(token)
        for key in ("enzymes", "modifiers", "catalysts"):
            for row in _safe_list(reaction.get(key)):
                remember_ref(_actor_name_from_row(row))
                if isinstance(row, dict):
                    remember_ref(row.get("entity"))
        for field in ("enzyme", "modifier", "protein", "protein_name", "protein_complex"):
            remember_ref(reaction.get(field))
    for transport in _safe_list(processes.get("transports")):
        if not isinstance(transport, dict):
            continue
        remember_ref(transport.get("cargo"))
        remember_ref(transport.get("cargo_complex"))
        for row in _safe_list(transport.get("transporters")):
            remember_ref(_actor_name_from_row(row))
            if isinstance(row, dict):
                remember_ref(row.get("entity"))
    for interaction in _safe_list(processes.get("interactions")):
        if not isinstance(interaction, dict):
            continue
        remember_ref(interaction.get("entity_1") or interaction.get("left") or interaction.get("source"))
        remember_ref(interaction.get("entity_2") or interaction.get("right") or interaction.get("target"))

    component_refs: Dict[str, List[Tuple[Dict[str, Any], Any]]] = defaultdict(list)
    identified_component_norms: Set[str] = set()
    process_referenced_component_norms: Set[str] = set()
    for complex_row in complexes:
        if not isinstance(complex_row, dict):
            continue
        complex_is_process_referenced = (
            _normalize(str(complex_row.get("name", ""))) in process_ref_norms
        )
        for component in _safe_list(complex_row.get("components")):
            component_name = _component_name_from_row(component)
            norm = _normalize(component_name)
            if not norm:
                continue
            component_refs[norm].append((complex_row, component))
            if isinstance(component, dict) and has_protein_external_identity(component):
                identified_component_norms.add(norm)
            if complex_is_process_referenced:
                process_referenced_component_norms.add(norm)

    rogue_norms: Set[str] = set()
    for protein in proteins:
        if not isinstance(protein, dict):
            continue
        name = _canonical(str(protein.get("name", "")))
        norm = _normalize(name)
        if (
            norm
            and norm in component_refs
            and norm not in process_ref_norms
            and norm not in process_referenced_component_norms
            and norm not in identified_component_norms
            and not has_protein_external_identity(protein)
        ):
            rogue_norms.add(norm)

    if not rogue_norms:
        return payload

    kept_proteins: List[Any] = []
    for protein in proteins:
        if not isinstance(protein, dict):
            kept_proteins.append(protein)
            continue
        norm = _normalize(str(protein.get("name", "")))
        if norm in rogue_norms:
            summary["component_only_proteins_removed"] += 1
            rep["actions"].append(
                {
                    "type": "component_only_protein_removed",
                    "name": _canonical(str(protein.get("name", ""))),
                }
            )
            continue
        kept_proteins.append(protein)
    entities["proteins"] = kept_proteins

    for complex_idx, complex_row in enumerate(complexes):
        if not isinstance(complex_row, dict):
            continue
        components = _safe_list(complex_row.get("components"))
        kept_components: List[Any] = []
        for component_idx, component in enumerate(components):
            component_name = _component_name_from_row(component)
            norm = _normalize(component_name)
            if norm in rogue_norms:
                summary["unresolved_complex_components_dropped"] += 1
                rep["actions"].append(
                    {
                        "type": "unresolved_complex_component_dropped",
                        "json_pointer": f"/entities/protein_complexes/{complex_idx}/components/{component_idx}",
                        "complex": _canonical(str(complex_row.get("name", ""))),
                        "component": component_name,
                    }
                )
                continue
            kept_components.append(component)
        complex_row["components"] = kept_components

    return payload


def drop_process_orphan_proteins(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Drop proteins from entities that are never referenced in any process and have no external identity.

    This catches the case where extraction produces individual subunit entries (e.g., NdmC, NdmD,
    NdmE) while the reactions only reference the complex form (e.g., NdmCDE complex), leaving the
    subunits as degree-0 orphans that would fail the hard-gate connectivity check.
    """
    rep = report if isinstance(report, dict) else _new_report()
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("orphan_proteins_dropped", 0)
    rep.setdefault("actions", [])

    entities = _safe_dict(payload.setdefault("entities", {}))
    proteins = _safe_list(entities.get("proteins"))
    processes = _safe_dict(payload.get("processes"))

    process_ref_norms: Set[str] = set()

    def _remember(value: Any) -> None:
        name = _canonical(str(value or ""))
        norm = _normalize(name)
        if norm:
            process_ref_norms.add(norm)

    for reaction in _safe_list(processes.get("reactions")):
        if not isinstance(reaction, dict):
            continue
        for side in ("inputs", "outputs"):
            for token in _safe_list(reaction.get(side)):
                _remember(token)
        for key in ("enzymes", "modifiers", "catalysts"):
            for row in _safe_list(reaction.get(key)):
                _remember(_actor_name_from_row(row))
                if isinstance(row, dict):
                    _remember(row.get("entity"))
        for field in ("enzyme", "modifier", "protein", "protein_name", "protein_complex"):
            _remember(reaction.get(field))

    for transport in _safe_list(processes.get("transports")):
        if not isinstance(transport, dict):
            continue
        _remember(transport.get("cargo"))
        _remember(transport.get("cargo_complex"))
        for row in _safe_list(transport.get("transporters")):
            _remember(_actor_name_from_row(row))
            if isinstance(row, dict):
                _remember(row.get("entity"))

    for interaction in _safe_list(processes.get("interactions")):
        if not isinstance(interaction, dict):
            continue
        _remember(interaction.get("entity_1") or interaction.get("left") or interaction.get("source"))
        _remember(interaction.get("entity_2") or interaction.get("right") or interaction.get("target"))

    # A component of a surviving complex is still semantically referenced even
    # when processes point at the complex rather than the component protein.
    for complex_row in _safe_list(entities.get("protein_complexes")):
        if not isinstance(complex_row, dict):
            continue
        for component in _safe_list(complex_row.get("components")):
            _remember(_component_name_from_row(component))

    kept: List[Any] = []
    for protein in proteins:
        if not isinstance(protein, dict):
            kept.append(protein)
            continue
        name = _canonical(str(protein.get("name", "")))
        norm = _normalize(name)
        if norm and norm not in process_ref_norms and not has_protein_external_identity(protein):
            summary["orphan_proteins_dropped"] += 1
            rep["actions"].append({"type": "orphan_protein_dropped", "name": name})
        else:
            kept.append(protein)

    entities["proteins"] = kept
    return payload


def _token_parts_for_aliasing(token: str) -> List[str]:
    text = _canonical(token)
    if not text:
        return []
    parts: List[str] = [text]
    if _complex_components(text):
        parts.extend(_complex_components(text))
    elif "+" in text:
        parts.extend(_split_composite(text))
    return _dedupe_preserve(parts)


def _dedupe_location_rows(rows: List[Any], *, key_name: str) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _canonical(str(row.get(key_name, "")))
        state = _canonical(str(row.get("biological_state", "")))
        if not name:
            continue
        key = (_normalize(name), _normalize(state))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def canonicalize_same_as_aliases(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("n_same_as_groups", 0)
    summary.setdefault("n_aliases_rewritten", 0)
    summary.setdefault("n_entities_deduped", 0)
    summary.setdefault("n_single_protein_complexes_removed", 0)
    summary.setdefault("alias_example_mappings", [])

    compounds, proteins, complexes = _entity_lists(payload)
    processes = _safe_dict(payload.setdefault("processes", {}))
    reactions = _safe_list(processes.get("reactions"))
    transports = _safe_list(processes.get("transports"))
    if not isinstance(processes.get("interactions"), list):
        processes["interactions"] = []
    interactions = _safe_list(processes.get("interactions"))
    element_locations = _safe_dict(payload.setdefault("element_locations", {}))
    if not isinstance(element_locations.get("compound_locations"), list):
        element_locations["compound_locations"] = []
    if not isinstance(element_locations.get("protein_locations"), list):
        element_locations["protein_locations"] = []

    alias_display: Dict[str, str] = {}
    enzyme_ref_freq: Dict[str, int] = {}
    io_ref_freq: Dict[str, int] = {}
    entity_registry_norms: Set[str] = set()
    single_complex_map: Dict[str, str] = {}

    def _remember_display(name: str) -> None:
        cname = _canonical(name)
        norm = _normalize(cname)
        if not norm:
            return
        if norm not in alias_display:
            alias_display[norm] = cname

    def _bump(counter: Dict[str, int], name: str) -> None:
        cname = _canonical(name)
        norm = _normalize(cname)
        if not norm:
            return
        counter[norm] = counter.get(norm, 0) + 1
        _remember_display(cname)

    # Preserve single-protein complexes: reaction enzymes are represented as
    # complexes even when the complex has one protein component.
    kept_complexes: List[Dict[str, Any]] = []
    for idx, row in enumerate(complexes):
        if not isinstance(row, dict):
            continue
        complex_name = _canonical(str(row.get("name", "")))
        components = _rewrite_component_rows(_safe_list(row.get("components")), _canonical)
        if not components and complex_name and ":" not in complex_name:
            components = [complex_name]
        component_names = [_component_name_from_row(component) for component in components]
        if len(component_names) == 1 and not _is_known_complex_name(component_names[0], payload):
            _ensure_protein(component_names[0], payload, rep)
        row["name"] = complex_name
        if components:
            row["components"] = components
        kept_complexes.append(row)
    complexes[:] = kept_complexes

    # Gather registry names.
    for rows in [proteins, compounds, complexes]:
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = _canonical(str(row.get("name", "")))
            norm = _normalize(name)
            if not norm:
                continue
            entity_registry_norms.add(norm)
            _remember_display(name)
            if rows is complexes:
                for comp in _safe_list(row.get("components")):
                    comp_name = _component_name_from_row(comp)
                    comp_norm = _normalize(comp_name)
                    if comp_norm:
                        entity_registry_norms.add(comp_norm)
                        _remember_display(comp_name)

    # Gather reaction references for canonical-selection priority.
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            for token in _safe_list(reaction.get(side)):
                if not isinstance(token, str):
                    continue
                for part in _token_parts_for_aliasing(token):
                    _bump(io_ref_freq, part)
        for key in ["enzymes", "modifiers"]:
            for row in _safe_list(reaction.get(key)):
                actor_name = _actor_name_from_row(row)
                if actor_name:
                    _bump(enzyme_ref_freq, actor_name)
        for field in ["enzyme", "modifier", "protein", "protein_name", "protein_complex"]:
            value = reaction.get(field)
            if isinstance(value, str):
                _bump(enzyme_ref_freq, value)

    # Build SAME_AS equivalence classes from interactions.
    uf = _AliasUnionFind()
    for idx, row in enumerate(interactions):
        if not isinstance(row, dict):
            continue
        if not _is_same_as_relationship(str(row.get("relationship", ""))):
            continue
        left = _canonical(str(row.get("entity_1", "")))
        right = _canonical(str(row.get("entity_2", "")))
        if not left or not right:
            continue
        uf.add(left)
        uf.add(right)
        uf.union(left, right)
        _remember_display(left)
        _remember_display(right)
        rep["actions"].append(
            {
                "type": "same_as_edge_registered",
                "json_pointer": f"/processes/interactions/{idx}",
                "entity_1": left,
                "entity_2": right,
            }
        )

    groups = [group for group in uf.groups().values() if len(group) >= 2]
    groups.sort(key=lambda group: tuple(sorted(group)))
    summary["n_same_as_groups"] = int(summary.get("n_same_as_groups", 0)) + len(groups)

    alias_map: Dict[str, str] = {}
    example_mappings: List[Dict[str, str]] = []

    def _candidate_rank(norm: str) -> Tuple[int, int, str, str]:
        display = alias_display.get(norm, norm)
        if norm in enzyme_ref_freq:
            return (0, -int(enzyme_ref_freq.get(norm, 0)), _normalize(display), display)
        if norm in io_ref_freq:
            return (1, -int(io_ref_freq.get(norm, 0)), _normalize(display), display)
        if norm in entity_registry_norms:
            return (2, 0, _normalize(display), display)
        return (3, 0, _normalize(display), display)

    for group in groups:
        candidates = sorted(group)
        canonical_norm = sorted(candidates, key=_candidate_rank)[0]
        canonical_name = alias_display.get(canonical_norm, canonical_norm)
        for norm in candidates:
            alias_map[norm] = canonical_name
            alias_name = alias_display.get(norm, norm)
            if _normalize(alias_name) != _normalize(canonical_name) and len(example_mappings) < 10:
                example_mappings.append({"alias": alias_name, "canonical": canonical_name})

    for norm, target in single_complex_map.items():
        alias_map[norm] = target
        if len(example_mappings) < 10:
            alias_name = alias_display.get(norm, norm)
            if _normalize(alias_name) != _normalize(target):
                example_mappings.append({"alias": alias_name, "canonical": target})

    def _resolve_alias_target(norm: str) -> str:
        current_norm = _normalize(norm)
        if not current_norm:
            return ""
        seen: Set[str] = set()
        current_name = alias_map.get(current_norm, alias_display.get(current_norm, current_norm))
        while True:
            next_norm = _normalize(current_name)
            if not next_norm or next_norm in seen:
                break
            seen.add(next_norm)
            next_name = alias_map.get(next_norm)
            if not next_name:
                break
            current_name = next_name
        return _canonical(current_name)

    resolved_alias_map: Dict[str, str] = {}
    for norm in sorted(alias_map.keys()):
        target = _resolve_alias_target(norm)
        if target:
            resolved_alias_map[norm] = target

    rewrite_count = 0

    def _rewrite_name(name: str) -> str:
        nonlocal rewrite_count
        cname = _canonical(name)
        norm = _normalize(cname)
        if not norm:
            return ""
        target = resolved_alias_map.get(norm)
        if not target:
            return cname
        if _canonical(target) != cname:
            rewrite_count += 1
            rep["actions"].append(
                {
                    "type": "alias_rewrite",
                    "from": cname,
                    "to": target,
                }
            )
        return _canonical(target)

    def _rewrite_token(token: str) -> str:
        text = _canonical(token)
        if not text:
            return ""
        parts = _complex_components(text, payload=payload)
        if parts:
            rewritten = _dedupe_preserve(
                [
                    rewritten_part
                    for rewritten_part in [_rewrite_name(part) for part in parts]
                    if rewritten_part
                ]
            )
            return ":".join(rewritten)
        if "+" in text:
            parts = _split_composite(text)
            rewritten = _dedupe_preserve(
                [
                    rewritten_part
                    for rewritten_part in [_rewrite_name(part) for part in parts]
                    if rewritten_part
                ]
            )
            return " + ".join(rewritten)
        return _rewrite_name(text)

    def _rewrite_actor_rows(rows: List[Any]) -> List[Any]:
        out: List[Any] = []
        for row in rows:
            if isinstance(row, str):
                out.append(_rewrite_name(row))
                continue
            if not isinstance(row, dict):
                continue
            updated = dict(row)
            for field in ["entity", "protein", "protein_name", "protein_complex", "enzyme", "modifier", "name"]:
                if isinstance(updated.get(field), str):
                    updated[field] = _rewrite_token(str(updated.get(field)))
            out.append(updated)
        return out

    # Rewrite entity lists.
    for rows in [proteins, compounds]:
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["name"] = _rewrite_name(str(row.get("name", "")))
    rewritten_complexes: List[Dict[str, Any]] = []
    for row in complexes:
        if not isinstance(row, dict):
            continue
        updated = dict(row)
        updated["name"] = _rewrite_token(str(updated.get("name", "")))
        comps = _safe_list(updated.get("components"))
        updated["components"] = _rewrite_component_rows(comps, _rewrite_name)
        component_names = [_component_name_from_row(component) for component in updated["components"]]
        if len(component_names) == 1 and not _is_known_complex_name(component_names[0], payload):
            _ensure_protein(component_names[0], payload, rep)
        rewritten_complexes.append(updated)
    complexes[:] = rewritten_complexes

    # Rewrite process references.
    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        for side in ["inputs", "outputs"]:
            rewritten = []
            for token in _safe_list(reaction.get(side)):
                if not isinstance(token, str):
                    continue
                value = _rewrite_token(token)
                if value:
                    rewritten.append(value)
            reaction[side] = _dedupe_preserve(rewritten)
        for key in ["enzymes", "modifiers"]:
            rows = _safe_list(reaction.get(key))
            if not isinstance(reaction.get(key), list):
                reaction[key] = rows
            reaction[key] = _rewrite_actor_rows(rows)
        for field in ["enzyme", "modifier", "protein", "protein_name", "protein_complex"]:
            if isinstance(reaction.get(field), str):
                reaction[field] = _rewrite_token(str(reaction.get(field)))
        for eidx, row in enumerate(_safe_list(reaction.get("elements_with_states"))):
            if not isinstance(row, dict):
                continue
            if isinstance(row.get("element"), str):
                row["element"] = _rewrite_token(str(row.get("element")))
                rep["actions"].append(
                    {
                        "type": "elements_with_states_element_rewritten",
                        "json_pointer": f"/processes/reactions/{ridx}/elements_with_states/{eidx}/element",
                        "value": row["element"],
                    }
                )

    for tidx, transport in enumerate(transports):
        if not isinstance(transport, dict):
            continue
        if isinstance(transport.get("cargo"), str):
            transport["cargo"] = _rewrite_token(str(transport.get("cargo")))
        if isinstance(transport.get("cargo_complex"), str):
            transport["cargo_complex"] = _rewrite_token(str(transport.get("cargo_complex")))
        rows = _safe_list(transport.get("transporters"))
        if not isinstance(transport.get("transporters"), list):
            transport["transporters"] = rows
        transport["transporters"] = _rewrite_actor_rows(rows)
        for eidx, row in enumerate(_safe_list(transport.get("elements_with_states"))):
            if not isinstance(row, dict):
                continue
            if isinstance(row.get("element"), str):
                row["element"] = _rewrite_token(str(row.get("element")))
                rep["actions"].append(
                    {
                        "type": "elements_with_states_element_rewritten",
                        "json_pointer": f"/processes/transports/{tidx}/elements_with_states/{eidx}/element",
                        "value": row["element"],
                    }
                )

    # Rewrite interactions (including SAME_AS pointers).
    rewritten_interactions: List[Dict[str, Any]] = []
    seen_interaction_keys: Set[Tuple[str, str, str]] = set()
    for idx, row in enumerate(interactions):
        if not isinstance(row, dict):
            continue
        updated = dict(row)
        if isinstance(updated.get("entity_1"), str):
            updated["entity_1"] = _rewrite_token(str(updated.get("entity_1")))
        if isinstance(updated.get("entity_2"), str):
            updated["entity_2"] = _rewrite_token(str(updated.get("entity_2")))
        key = (
            _normalize(str(updated.get("relationship", ""))),
            _normalize(str(updated.get("entity_1", ""))),
            _normalize(str(updated.get("entity_2", ""))),
        )
        if key in seen_interaction_keys:
            rep["actions"].append(
                {
                    "type": "interaction_deduped_after_alias_rewrite",
                    "json_pointer": f"/processes/interactions/{idx}",
                }
            )
            continue
        seen_interaction_keys.add(key)
        rewritten_interactions.append(updated)
    processes["interactions"] = rewritten_interactions

    # Rewrite element-location references.
    compound_locations = _safe_list(element_locations.get("compound_locations"))
    protein_locations = _safe_list(element_locations.get("protein_locations"))
    for row in compound_locations:
        if isinstance(row, dict) and isinstance(row.get("compound"), str):
            row["compound"] = _rewrite_token(str(row.get("compound")))
    for row in protein_locations:
        if isinstance(row, dict) and isinstance(row.get("protein"), str):
            row["protein"] = _rewrite_token(str(row.get("protein")))
    element_locations["compound_locations"] = _dedupe_location_rows(compound_locations, key_name="compound")
    element_locations["protein_locations"] = _dedupe_location_rows(protein_locations, key_name="protein")

    # Ensure every enzyme actor has a concrete registry entry.
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        for key in ["enzymes", "modifiers"]:
            for row in _safe_list(reaction.get(key)):
                actor_name = _actor_name_from_row(row)
                if not actor_name:
                    continue
                if ":" in actor_name:
                    continue
                if _find_entity_row(complexes, actor_name) is not None:
                    continue
                _ensure_protein(actor_name, payload, rep)

    # Final entity dedupe pass.
    pre_dedupe_count = len(compounds) + len(proteins) + len(complexes)
    _dedupe_named_rows(compounds)
    _dedupe_named_rows(proteins)
    _dedupe_named_rows(complexes)
    post_dedupe_count = len(compounds) + len(proteins) + len(complexes)
    summary["n_entities_deduped"] = int(summary.get("n_entities_deduped", 0)) + max(0, pre_dedupe_count - post_dedupe_count)
    summary["n_aliases_rewritten"] = int(summary.get("n_aliases_rewritten", 0)) + rewrite_count
    summary["alias_example_mappings"] = example_mappings[:10]
    return payload


def normalize_process_actor_schema(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("modifier_refs_canonicalized", 0)
    summary.setdefault("modifier_refs_dropped", 0)
    summary.setdefault("non_protein_catalysts_dropped", 0)
    rep.setdefault("actions", [])
    allowed_enzyme_entity_types = {"protein", "protein_complex"}
    dropped_enzyme_entity_types = {"compound", "cofactor", "ion", "small_molecule", "metabolite"}
    enzyme_export_roles = {"", "catalyst", "enzyme", "activator", "inhibitor"}
    entities = _safe_dict(payload.get("entities"))
    protein_rows = _safe_list(entities.get("proteins"))
    complex_rows = _safe_list(entities.get("protein_complexes"))
    compound_rows = _safe_list(entities.get("compounds"))
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
    compound_by_norm = {
        _normalize(_canonical(str(row.get("name", "")))): _canonical(str(row.get("name", "")))
        for row in compound_rows
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
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
            if norm in complex_by_norm:
                return "protein_complex", complex_by_norm[norm]
            if norm in protein_by_norm:
                return "protein", protein_by_norm[norm]
            if norm in compound_by_norm:
                return "compound", compound_by_norm[norm]
        return None

    def _record_non_protein_catalyst_drop(pointer: str, name: str, entity_type: str, role: str, source_field: str) -> None:
        summary["non_protein_catalysts_dropped"] += 1
        summary["modifier_refs_dropped"] += 1
        rep["actions"].append(
            {
                "type": "non_protein_catalyst_dropped",
                "json_pointer": pointer,
                "name": name,
                "entity_type": entity_type,
                "role": role or "catalyst",
                "source_field": source_field,
            }
        )

    def _rewrite_actor_rows(rows: List[Any], pointer_prefix: str, *, drop_unknown: bool = True) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            pointer = f"{pointer_prefix}/{idx}"
            source_field = ""
            raw_name = ""
            for field in ["entity", "protein", "protein_complex", "name"]:
                candidate = _canonical(str(row.get(field, "")))
                if candidate:
                    source_field = field
                    raw_name = candidate
                    break
            if not raw_name:
                if not drop_unknown:
                    kept.append(row)
                continue

            explicit_entity_type = _canonical(str(row.get("entity_type") or row.get("type") or "")).casefold()
            role = _canonical(str(row.get("role", ""))).casefold()
            if (
                explicit_entity_type in dropped_enzyme_entity_types
                and (pointer_prefix.endswith("/enzymes") or role in enzyme_export_roles)
            ):
                _record_non_protein_catalyst_drop(pointer, raw_name, explicit_entity_type, role, source_field)
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

            entity_type, canonical_name = resolved
            updated = dict(row)
            for field in ["protein", "protein_complex", "name", "protein_name"]:
                updated.pop(field, None)
            updated["entity"] = canonical_name
            updated["entity_type"] = entity_type
            # Keep the legacy typed key during migration for downstream SBML
            # consumers that have not yet switched to the canonical fields.
            if entity_type in {"protein", "protein_complex"}:
                updated[entity_type] = canonical_name
            if source_field != "entity" or _normalize(raw_name) != _normalize(canonical_name):
                summary["modifier_refs_canonicalized"] += 1
                rep["actions"].append(
                    {
                        "type": "schema_drift_actor_canonicalized",
                        "json_pointer": pointer,
                        "from_field": source_field,
                        "to_field": "entity",
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

    # Post-process: normalise modifiers[] to entity/entity_type schema and migrate legacy enzymes[].
    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        # 1. Ensure modifiers[] rows use entity/entity_type schema.
        new_modifiers: List[Dict[str, Any]] = []
        for midx, mod in enumerate(_safe_list(reaction.get("modifiers"))):
            if not isinstance(mod, dict):
                continue
            updated_mod = dict(mod)
            if not updated_mod.get("entity"):
                # Migrate from old protein/protein_complex/name key to entity/entity_type.
                for old_key, old_type in [("protein_complex", "protein_complex"), ("protein", "protein"), ("name", "protein")]:
                    val = _canonical(str(updated_mod.get(old_key, "")))
                    if val:
                        updated_mod["entity"] = val
                        updated_mod.setdefault("entity_type", old_type)
                        updated_mod.pop(old_key, None)
                        break
            if not updated_mod.get("entity_type") and updated_mod.get("entity"):
                updated_mod["entity_type"] = "protein"
            for old_key in ["protein", "protein_complex", "name", "protein_name"]:
                updated_mod.pop(old_key, None)
            updated_mod.setdefault("role", "catalyst")
            entity_type = _canonical(str(updated_mod.get("entity_type", ""))).casefold()
            role = _canonical(str(updated_mod.get("role", "catalyst"))).casefold() or "catalyst"
            if entity_type in dropped_enzyme_entity_types and role in enzyme_export_roles:
                _record_non_protein_catalyst_drop(
                    f"/processes/reactions/{ridx}/modifiers/{midx}",
                    _canonical(str(updated_mod.get("entity", ""))),
                    entity_type,
                    role,
                    "entity",
                )
                continue
            if updated_mod.get("entity"):
                new_modifiers.append(updated_mod)
        reaction["modifiers"] = new_modifiers

        # 1b. Correct entity_type when entity name matches a known protein_complex
        #     (covers cases where the LLM used entity_type="protein" for a complex).
        for mod in reaction["modifiers"]:
            if not isinstance(mod, dict):
                continue
            entity = _canonical(str(mod.get("entity", "")))
            if not entity or mod.get("entity_type") != "protein":
                continue
            norm = _normalize(_canonical_complex_name(entity))
            if norm in complex_by_norm:
                mod["entity_type"] = "protein_complex"
                mod["entity"] = complex_by_norm[norm]
                mod.pop("protein_complex", None)

        # 1c. Migrate enzymes[] rows to entity/entity_type schema (mirrors modifier migration above).
        new_enzymes: List[Dict[str, Any]] = []
        for eidx, enz in enumerate(_safe_list(reaction.get("enzymes"))):
            if not isinstance(enz, dict):
                continue
            updated_enz = dict(enz)
            if not updated_enz.get("entity"):
                # Migrate from old protein_complex/protein/name key to entity/entity_type.
                for old_key, old_type in [("protein_complex", "protein_complex"), ("protein", "protein"), ("name", "protein")]:
                    val = _canonical(str(updated_enz.get(old_key, "")))
                    if val:
                        updated_enz["entity"] = val
                        updated_enz.setdefault("entity_type", old_type)
                        updated_enz.pop(old_key, None)
                        break
            if not updated_enz.get("entity_type") and updated_enz.get("entity"):
                updated_enz["entity_type"] = "protein"
            for old_key in ["protein", "protein_complex", "name", "protein_name"]:
                updated_enz.pop(old_key, None)
            updated_enz.setdefault("role", "catalyst")
            entity_type = _canonical(str(updated_enz.get("entity_type", ""))).casefold()
            role = _canonical(str(updated_enz.get("role", "catalyst"))).casefold() or "catalyst"
            if entity_type in dropped_enzyme_entity_types and role in enzyme_export_roles:
                _record_non_protein_catalyst_drop(
                    f"/processes/reactions/{ridx}/enzymes/{eidx}",
                    _canonical(str(updated_enz.get("entity", ""))),
                    entity_type,
                    role,
                    "entity",
                )
                continue
            if updated_enz.get("entity"):
                new_enzymes.append(updated_enz)
        reaction["enzymes"] = new_enzymes

        # 2. Migrate legacy enzymes[] → modifiers[] with role: "catalyst".
        enzyme_rows = _safe_list(reaction.get("enzymes"))
        if enzyme_rows:
            existing_modifier_norms: Set[str] = {
                _normalize(_canonical(str(m.get("entity", ""))))
                for m in reaction["modifiers"]
                if isinstance(m, dict) and _canonical(str(m.get("entity", "")))
            }
            for enz in enzyme_rows:
                if not isinstance(enz, dict):
                    continue
                ename = ""
                etype = "protein"
                for k, t in [("entity", None), ("protein_complex", "protein_complex"), ("protein", "protein"), ("name", "protein")]:
                    v = _canonical(str(enz.get(k, "")))
                    if v:
                        ename = v
                        if k == "entity":
                            etype = _canonical(str(enz.get("entity_type") or "protein")).casefold()
                        elif t:
                            etype = t
                        break
                if not ename or _normalize(ename) in existing_modifier_norms:
                    continue
                if etype not in allowed_enzyme_entity_types:
                    _record_non_protein_catalyst_drop(
                        f"/processes/reactions/{ridx}/enzymes",
                        ename,
                        etype,
                        str(enz.get("role") or "catalyst"),
                        "entity",
                    )
                    continue
                reaction["modifiers"].append({
                    "entity": ename,
                    "entity_type": etype,
                    "role": str(enz.get("role") or "catalyst"),
                    "evidence": _canonical(str(enz.get("evidence", ""))),
                    "confidence": float(enz.get("confidence", 1.0)),
                    "provenance": str(enz.get("provenance") or "extracted"),
                })
                existing_modifier_norms.add(_normalize(ename))
                summary["modifier_refs_canonicalized"] += 1
        # Rebuild enzymes[] from modifiers[] (role=catalyst only) using entity/entity_type schema.
        # The canonical representation remains modifiers[]; enzymes[] is kept in sync as a view.
        canonical_enzymes: List[Dict[str, Any]] = []
        seen_enzyme_norms: Set[Tuple[str, str]] = set()
        for mod in reaction["modifiers"]:
            if not isinstance(mod, dict):
                continue
            role = _canonical(str(mod.get("role", "catalyst"))).casefold() or "catalyst"
            if role != "catalyst":
                continue
            entity = _canonical(str(mod.get("entity", "")))
            if not entity:
                continue
            entity_type = _canonical(str(mod.get("entity_type", "protein"))).casefold() or "protein"
            if entity_type not in allowed_enzyme_entity_types:
                if entity_type in dropped_enzyme_entity_types:
                    _record_non_protein_catalyst_drop(
                        f"/processes/reactions/{ridx}/modifiers",
                        entity,
                        entity_type,
                        role,
                        "entity",
                    )
                continue
            key = (entity_type, _normalize(entity))
            if key in seen_enzyme_norms:
                continue
            seen_enzyme_norms.add(key)
            canonical_row: Dict[str, Any] = {
                "entity": entity,
                "entity_type": entity_type,
                "role": "catalyst",
                "confidence": mod.get("confidence", 1.0),
                "provenance": mod.get("provenance", "extracted"),
            }
            evidence = _canonical(str(mod.get("evidence", "")))
            if evidence:
                canonical_row["evidence"] = evidence
            canonical_enzymes.append(canonical_row)
        if canonical_enzymes or "enzymes" in reaction:
            reaction["enzymes"] = canonical_enzymes

    for tidx, transport in enumerate(transports):
        if not isinstance(transport, dict):
            continue
        rows = _safe_list(transport.get("transporters"))
        if not isinstance(transport.get("transporters"), list):
            transport["transporters"] = rows
        transport["transporters"] = _rewrite_actor_rows(rows, f"/processes/transports/{tidx}/transporters")

    processes = _safe_dict(payload.get("processes"))
    for iidx, interaction in enumerate(_safe_list(processes.get("interactions"))):
        if not isinstance(interaction, dict):
            continue
        rows = _safe_list(interaction.get("participants"))
        if "participants" in interaction:
            interaction["participants"] = _rewrite_actor_rows(
                rows,
                f"/processes/interactions/{iidx}/participants",
                drop_unknown=False,
            )
    return payload


def ensure_autostates(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    auto_state_name = "__auto_state__"
    auto_location_name = "cell"

    if not isinstance(payload.get("entities"), dict):
        payload["entities"] = {}
    entities = _safe_dict(payload.get("entities"))
    if not isinstance(entities.get("subcellular_locations"), list):
        entities["subcellular_locations"] = []
    subcellular_locations = _safe_list(entities.get("subcellular_locations"))
    if _find_entity_row(subcellular_locations, auto_location_name) is None:
        subcellular_locations.append({"name": auto_location_name})
    auto_species_name = _select_default_species_name(entities)

    if not isinstance(payload.get("biological_states"), list):
        payload["biological_states"] = []
    biological_states = _safe_list(payload.get("biological_states"))
    auto_state = None
    for row in biological_states:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            continue
        if _normalize(row["name"]) == _normalize(auto_state_name):
            auto_state = row
            break
    if auto_state is None:
        auto_state = {"name": auto_state_name, "subcellular_location": auto_location_name}
        if auto_species_name:
            auto_state["species"] = auto_species_name
        biological_states.append(auto_state)
        rep["summary"]["n_autostate_created"] += 1
    else:
        if not _canonical(str(auto_state.get("subcellular_location", ""))):
            auto_state["subcellular_location"] = auto_location_name
        if auto_species_name and not _canonical(str(auto_state.get("species") or auto_state.get("organism") or "")):
            auto_state["species"] = auto_species_name

    if auto_species_name:
        for row in biological_states:
            if not isinstance(row, dict):
                continue
            if not _canonical(str(row.get("species") or row.get("organism") or "")):
                row["species"] = auto_species_name

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


def backfill_reaction_compartments(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    For every reaction whose ``biological_state`` is empty, attempt to inherit
    a compartment from its reactants/products/modifiers.

    Strategy
    --------
    1. Build a map of {entity_name_norm: biological_state_name} from all
       compound_locations and protein_locations.
    2. For each reaction with no biological_state, collect the biological_states
       of all its inputs, outputs, and modifier entities.
    3. If all resolved states agree on the same canonical compartment, assign
       that state to the reaction's biological_state.
    """
    rep = report if isinstance(report, dict) else _new_report()
    rep.setdefault("summary", {}).setdefault("n_reactions_compartment_backfilled", 0)

    # Build entity → biological_state lookup
    element_locations = _safe_dict(payload.get("element_locations", {}))
    entity_state: Dict[str, str] = {}
    for list_key in ("compound_locations", "protein_locations", "nucleic_acid_locations",
                     "element_collection_locations"):
        for row in _safe_list(element_locations.get(list_key)):
            if not isinstance(row, dict):
                continue
            for key in ("compound", "protein", "nucleic_acid", "element_collection"):
                name = _canonical(str(row.get(key, "")))
                if not name:
                    continue
                state = _canonical(str(row.get("biological_state", "")))
                if state:
                    entity_state[_normalize(name)] = state

    # Build biological_state → compartment_canonical lookup
    state_compartment: Dict[str, str] = {}
    for bs in _safe_list(payload.get("biological_states")):
        if not isinstance(bs, dict):
            continue
        name = _canonical(str(bs.get("name", "")))
        canon = _canonical(str(bs.get("compartment_canonical", "")))
        if name and canon:
            state_compartment[name] = canon

    processes = _safe_dict(payload.get("processes", {}))
    reactions = _safe_list(processes.get("reactions"))

    for rxn in reactions:
        if not isinstance(rxn, dict):
            continue
        if _canonical(str(rxn.get("biological_state", ""))):
            continue  # already assigned

        # Collect entity names from inputs + outputs + modifiers
        participant_names: List[str] = []
        for side in ("inputs", "outputs"):
            for token in _safe_list(rxn.get(side)):
                if isinstance(token, str):
                    participant_names.append(token)
        for mod in _safe_list(rxn.get("modifiers")):
            if isinstance(mod, dict):
                name = _canonical(str(mod.get("entity", "")))
                if name:
                    participant_names.append(name)

        # Resolve each participant to its compartment canonical
        compartments_seen: Set[str] = set()
        for name in participant_names:
            state_name = entity_state.get(_normalize(name))
            if state_name:
                comp = state_compartment.get(state_name)
                if comp:
                    compartments_seen.add(comp)

        if len(compartments_seen) == 1:
            inferred_comp = next(iter(compartments_seen))
            # Find the first biological_state that has this canonical compartment
            inferred_state = next(
                (s for s, c in state_compartment.items() if c == inferred_comp),
                None,
            )
            if inferred_state:
                rxn["biological_state"] = inferred_state
                rep["summary"]["n_reactions_compartment_backfilled"] += 1
                rep.setdefault("actions", []).append({
                    "type": "reaction_compartment_backfill",
                    "reaction": rxn.get("name", ""),
                    "assigned_state": inferred_state,
                    "inferred_from": participant_names[:5],
                })

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
            existing_name = _actor_name_from_row(existing)
            if existing_name and _normalize(existing_name) in known_norm_to_name:
                existing_norms.add(_normalize(existing_name))

        cargo_value = (
            transport.get("cargo_complex")
            if isinstance(transport.get("cargo_complex"), str) and _canonical(str(transport.get("cargo_complex", "")))
            else transport.get("cargo")
        )
        cargo = _canonical(str(cargo_value or ""))
        cargo_tokens = [cargo]
        cargo_tokens.extend(_complex_components(cargo, payload=payload))

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


_ENZYME_EVIDENCE_CUE_RE = re.compile(
    r"(catalyz|catalys|catalytic|enzyme|enzymatic|mediated|dependent|activity|activat|promot|facilitat)",
    flags=re.IGNORECASE,
)


def _cue_near_name(text: str, name: str, *, window: int = 80) -> Optional[str]:
    evidence = _canonical(text)
    actor_name = _canonical(name)
    if not evidence or not actor_name:
        return None
    for match in re.finditer(re.escape(actor_name), evidence, flags=re.IGNORECASE):
        start = max(0, match.start() - window)
        end = min(len(evidence), match.end() + window)
        snippet = evidence[start:end].strip()
        if _ENZYME_EVIDENCE_CUE_RE.search(snippet):
            return snippet
    return None


def attach_enzymes_from_reaction_evidence(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach enzyme actors when reaction evidence names one protein near a catalysis cue."""
    rep = report if isinstance(report, dict) else _new_report()
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("enzymes_attached_from_reaction_evidence", 0)
    rep.setdefault("actions", [])

    entities = _safe_dict(payload.get("entities"))
    protein_names = [
        _canonical(str(row.get("name", "")))
        for row in _safe_list(entities.get("proteins"))
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    ]
    complex_names = [
        _canonical(str(row.get("name", "")))
        for row in _safe_list(entities.get("protein_complexes"))
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    ]
    protein_norms = {_normalize(name) for name in protein_names}
    generated_wrapper_member_by_norm: Dict[str, str] = {}
    for row in _safe_list(entities.get("protein_complexes")):
        if not isinstance(row, dict) or not is_generated_complex_wrapper(row):
            continue
        complex_name = _canonical(str(row.get("name", "")))
        components = _safe_list(row.get("components"))
        if not complex_name or len(components) != 1:
            continue
        component = components[0]
        component_name = _component_name_from_row(component)
        component_norm = _normalize(component_name)
        if not component_norm or component_norm not in protein_norms:
            continue
        if isinstance(component, dict):
            stoichiometry = component.get("stoichiometry")
            try:
                if stoichiometry in (None, "") or int(stoichiometry) < 1:
                    continue
            except (TypeError, ValueError):
                continue
        generated_wrapper_member_by_norm[_normalize(complex_name)] = component_norm
    actor_candidates: List[Tuple[str, str]] = [("protein", name) for name in protein_names]
    actor_candidates.extend(("protein_complex", name) for name in complex_names)
    if not actor_candidates:
        return payload

    reactions, _ = _process_lists(payload)
    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        if not isinstance(reaction.get("enzymes"), list):
            reaction["enzymes"] = _safe_list(reaction.get("enzymes"))
        if not isinstance(reaction.get("modifiers"), list):
            reaction["modifiers"] = _safe_list(reaction.get("modifiers"))

        existing_norms = {
            _normalize(_actor_name_from_row(row))
            for row in _safe_list(reaction.get("enzymes")) + _safe_list(reaction.get("modifiers"))
            if _actor_name_from_row(row)
        }
        # A Stage 6 generated single-protein wrapper already attached to the
        # reaction represents its declared component.  Derive that relationship
        # from the wrapper metadata and component structure, never from a name
        # suffix; genuine multi-protein complexes must not suppress members.
        existing_norms.update(
            generated_wrapper_member_by_norm[actor_norm]
            for actor_norm in tuple(existing_norms)
            if actor_norm in generated_wrapper_member_by_norm
        )
        evidence_text = _canonical(
            " ".join(
                str(value or "")
                for value in (reaction.get("name", ""), reaction.get("evidence", ""))
            )
        )
        matches: List[Tuple[str, str, str]] = []
        for entity_type, actor_name in sorted(actor_candidates, key=lambda item: len(item[1]), reverse=True):
            actor_norm = _normalize(actor_name)
            if not actor_norm or actor_norm in existing_norms:
                continue
            snippet = _cue_near_name(evidence_text, actor_name)
            if snippet:
                matches.append((entity_type, actor_name, snippet))

        if len(matches) != 1:
            continue
        entity_type, actor_name, snippet = matches[0]
        reaction["enzymes"].append(
            {
                "entity": actor_name,
                "entity_type": entity_type,
                "role": "catalyst",
                "evidence": snippet,
                "confidence": 0.75,
                "provenance": "inferred",
            }
        )
        existing_norms.add(_normalize(actor_name))
        summary["enzymes_attached_from_reaction_evidence"] += 1
        rep["actions"].append(
            {
                "type": "enzyme_attached_from_reaction_evidence",
                "json_pointer": f"/processes/reactions/{ridx}/enzymes",
                "entity": actor_name,
                "entity_type": entity_type,
                "snippet": snippet,
            }
        )

    return payload


_CATALYTIC_REL_RE = re.compile(
    r"(catalyz|catalys|enzyme|activat|catalytic|promotes|facilitate)",
    flags=re.IGNORECASE,
)
_INHIBITION_REL_RE = re.compile(
    r"(inhibit|block|suppress|downregulat|repress)",
    flags=re.IGNORECASE,
)
_TRANSPORT_REL_RE = re.compile(
    r"(transport|carry|carri|shuttle|translocat)",
    flags=re.IGNORECASE,
)


def _modifier_role_from_relationship(relationship: str) -> str:
    """Determine modifier role from relationship text."""
    if _INHIBITION_REL_RE.search(relationship):
        return "inhibitor"
    if _TRANSPORT_REL_RE.search(relationship):
        return "transporter"
    if _CATALYTIC_REL_RE.search(relationship):
        if re.search(r"(activat|promot|stimulat)", relationship, re.IGNORECASE) and not re.search(r"(catalyz|catalys|enzyme)", relationship, re.IGNORECASE):
            return "activator"
    return "catalyst"


def promote_interaction_enzymes(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Move protein-catalysis interactions into reaction enzymes lists.

    The LLM sometimes generates enzyme-reaction relationships as
    processes/interactions entries (entity_1=protein, entity_2=reaction name,
    relationship="catalyzes") rather than putting the protein in
    reactions[].enzymes.  This step finds those interactions, adds the protein
    to the matching reaction's enzymes list, and removes the interaction so it
    doesn't interfere with later validation.
    """
    rep = report if isinstance(report, dict) else _new_report()
    _, proteins, complexes = _entity_lists(payload)
    protein_norms = _entity_name_norms(proteins) | _entity_name_norms(complexes)

    processes = _safe_dict(payload.setdefault("processes", {}))
    reactions = _safe_list(processes.get("reactions", []))
    interactions = _safe_list(processes.get("interactions", []))

    # Build normalised-name → reaction index map
    rxn_norm_to_idx: Dict[str, int] = {}
    for ridx, rxn in enumerate(reactions):
        if not isinstance(rxn, dict):
            continue
        rname = _canonical(str(rxn.get("name", "")))
        if rname:
            rxn_norm_to_idx[_normalize(rname)] = ridx

    kept_interactions: List[Dict[str, Any]] = []
    promoted = 0
    for iidx, inter in enumerate(interactions):
        if not isinstance(inter, dict):
            kept_interactions.append(inter)
            continue

        relationship = str(inter.get("relationship", ""))
        e1 = _canonical(str(inter.get("entity_1", "")))
        e2 = _canonical(str(inter.get("entity_2", "")))

        _is_modifier_rel = (
            _CATALYTIC_REL_RE.search(relationship)
            or _INHIBITION_REL_RE.search(relationship)
            or _TRANSPORT_REL_RE.search(relationship)
        )
        if not _is_modifier_rel:
            kept_interactions.append(inter)
            continue

        # Determine which entity is the protein and which is the reaction
        protein_name: Optional[str] = None
        rxn_idx: Optional[int] = None

        e1_norm = _normalize(e1)
        e2_norm = _normalize(e2)

        if e1_norm in protein_norms and e2_norm in rxn_norm_to_idx:
            protein_name = e1
            rxn_idx = rxn_norm_to_idx[e2_norm]
        elif e2_norm in protein_norms and e1_norm in rxn_norm_to_idx:
            protein_name = e2
            rxn_idx = rxn_norm_to_idx[e1_norm]
        elif e1_norm in protein_norms:
            # Protein is clear but reaction name doesn't match exactly —
            # try fuzzy: does any reaction name appear as substring of e2?
            for rnorm, ridx in rxn_norm_to_idx.items():
                if rnorm and (rnorm in e2_norm or e2_norm in rnorm):
                    protein_name = e1
                    rxn_idx = ridx
                    break
        elif e2_norm in protein_norms:
            for rnorm, ridx in rxn_norm_to_idx.items():
                if rnorm and (rnorm in e1_norm or e1_norm in rnorm):
                    protein_name = e2
                    rxn_idx = ridx
                    break

        if protein_name is None or rxn_idx is None:
            kept_interactions.append(inter)
            continue

        rxn = reactions[rxn_idx]
        if not isinstance(rxn.get("modifiers"), list):
            rxn["modifiers"] = []
        modifiers_list = rxn["modifiers"]
        modifier_entity_norms: set = set()
        for mod in modifiers_list:
            if isinstance(mod, dict):
                for k in ["entity", "protein", "protein_complex", "name"]:
                    v = _canonical(str(mod.get(k, "")))
                    if v:
                        modifier_entity_norms.add(_normalize(v))
                        break

        pnorm = _normalize(protein_name)
        if pnorm not in modifier_entity_norms:
            role = _modifier_role_from_relationship(relationship)
            entity_type = "protein_complex" if _is_known_complex_name(protein_name, payload) else "protein"
            new_mod: Dict[str, Any] = {
                "entity": protein_name,
                "entity_type": entity_type,
                "role": role,
                "evidence": _canonical(str(inter.get("evidence", ""))),
                "confidence": float(inter.get("confidence", 1.0)),
                "provenance": "inferred",
            }
            modifiers_list.append(new_mod)
            promoted += 1
            if isinstance(rep.get("actions"), list):
                rep["actions"].append({
                    "type": "interaction_modifier_promoted",
                    "json_pointer": f"/processes/interactions/{iidx}",
                    "protein": protein_name,
                    "role": role,
                    "reaction_idx": rxn_idx,
                })
        # Drop the interaction — it is now encoded in the reaction modifiers list

    if promoted:
        processes["interactions"] = kept_interactions
        if isinstance(rep.get("summary"), dict):
            rep["summary"]["interaction_enzymes_promoted"] += promoted

    return payload


def promote_catalysts(payload: Dict[str, Any], *, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = report if isinstance(report, dict) else _new_report()
    reactions, _ = _process_lists(payload)
    _, proteins, complexes = _entity_lists(payload)
    protein_norms = _entity_name_norms(proteins) | _entity_name_norms(complexes)
    complex_norms = _entity_name_norms(complexes)
    scaffold_norms = _scaffold_norms(payload)

    for ridx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        inputs = _dedupe_preserve([str(v) for v in _safe_list(reaction.get("inputs")) if isinstance(v, str)])
        outputs = _dedupe_preserve([str(v) for v in _safe_list(reaction.get("outputs")) if isinstance(v, str)])
        modifiers_list = _safe_list(reaction.get("modifiers"))
        if not isinstance(reaction.get("modifiers"), list):
            reaction["modifiers"] = modifiers_list

        output_complex_parts: Set[str] = set()
        for token in outputs:
            for part in _complex_components(token, payload=payload):
                output_complex_parts.add(_normalize(part))
        modifier_entity_norms: Set[str] = set()
        for mod in modifiers_list:
            if not isinstance(mod, dict):
                continue
            for key in ["entity", "protein", "protein_complex", "name"]:
                value = _canonical(str(mod.get(key, "")))
                if value:
                    modifier_entity_norms.add(_normalize(value))
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
            if norm not in modifier_entity_norms:
                entity_type = "protein_complex" if norm in complex_norms else "protein"
                modifiers_list.append({
                    "entity": token,
                    "entity_type": entity_type,
                    "role": "catalyst",
                    "evidence": "",
                    "confidence": 1.0,
                    "provenance": "extracted",
                })
                modifier_entity_norms.add(norm)
                rep["summary"]["catalysts_promoted_to_enzymes"] += 1
                rep["actions"].append({"type": "catalyst_promoted_to_modifier", "json_pointer": f"/processes/reactions/{ridx}/inputs", "name": token})

        present_inputs = {_normalize(token) for token in kept_inputs}
        for out_token in outputs:
            parts = _complex_components(out_token, payload=payload)
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
        reaction["modifiers"] = modifiers_list
    return payload

def _reaction_modifier_names(reaction: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ["modifiers", "enzymes"]:
        for row in _safe_list(reaction.get(key)):
            if not isinstance(row, dict):
                continue
            for name_key in ["entity", "protein", "protein_complex", "name"]:
                value = _canonical(str(row.get(name_key, "")))
                if value:
                    names.append(value)
                    break
    return _dedupe_preserve(names)


def _evidence_length(row: Dict[str, Any]) -> int:
    score = len(_canonical(str(row.get("evidence", ""))))
    for key in ["modifiers", "enzymes", "transporters"]:
        for item in _safe_list(row.get(key)):
            if isinstance(item, dict):
                score += len(_canonical(str(item.get("evidence", ""))))
    return score


def _is_inferred(row: Dict[str, Any]) -> bool:
    if isinstance(row.get("inference"), dict):
        return True
    if bool(row.get("inferred", False)):
        return True
    for key in ["modifiers", "enzymes", "transporters"]:
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
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("no_op_removed_count", 0)
    summary.setdefault("no_op_quarantined_count", 0)
    summary.setdefault("locked_no_op_quarantined_count", 0)
    summary.setdefault("evidenced_no_op_quarantined_count", 0)
    summary.setdefault("unsupported_no_op_dropped_count", 0)
    summary.setdefault("duplicate_locked_reactions_quarantined_count", 0)
    summary.setdefault("dedupe_removed_reactions", 0)
    summary.setdefault("dedupe_removed_transports", 0)
    summary.setdefault("dedupe_removed", 0)
    summary.setdefault("dedupe_removed_total", 0)
    rep.setdefault("actions", [])
    reactions, transports = _process_lists(payload)

    existing_quarantine = payload.get("quarantined_locked_reactions")
    quarantined: List[Dict[str, Any]] = (
        [deepcopy(row) for row in existing_quarantine if isinstance(row, dict)]
        if isinstance(existing_quarantine, list)
        else []
    )

    def _lock_id(reaction: Dict[str, Any]) -> str:
        value = reaction.get("locked_reaction_id")
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        return ""

    def _has_direct_evidence(reaction: Dict[str, Any]) -> bool:
        return _evidence_length(reaction) > 0 or bool(reaction.get("observed", False))

    def _quarantine_reaction(
        reaction: Dict[str, Any],
        *,
        reaction_index: int,
        reason: str,
        count_as_no_op: bool,
    ) -> None:
        lock_id = _lock_id(reaction)
        record: Dict[str, Any] = {
            "reaction_name": _canonical(str(reaction.get("name", ""))) or "<unnamed>",
            "reason": reason,
            "action": "quarantined_from_active_reactions",
            "json_pointer": f"/processes/reactions/{reaction_index}",
            "original_reaction": deepcopy(reaction),
        }
        if lock_id:
            record["locked_reaction_id"] = lock_id
        quarantined.append(record)
        if count_as_no_op:
            summary["no_op_quarantined_count"] += 1
        if lock_id and count_as_no_op:
            summary["locked_no_op_quarantined_count"] += 1
        elif lock_id:
            summary["duplicate_locked_reactions_quarantined_count"] += 1
        else:
            summary["evidenced_no_op_quarantined_count"] += 1
        actions = rep.get("actions")
        if isinstance(actions, list):
            actions.append(
                {
                    "type": "reaction_quarantined",
                    "reason": reason,
                    "json_pointer": record["json_pointer"],
                    "reaction_name": record["reaction_name"],
                    **({"locked_reaction_id": lock_id} if lock_id else {}),
                }
            )

    reaction_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    reaction_index_by_key: Dict[Tuple[Any, ...], int] = {}
    for reaction_index, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        in_norm = [_normalize(v) for v in _safe_list(reaction.get("inputs")) if isinstance(v, str) and _canonical(v)]
        out_norm = [_normalize(v) for v in _safe_list(reaction.get("outputs")) if isinstance(v, str) and _canonical(v)]
        if in_norm and out_norm:
            no_op_reason = ""
            if sorted(in_norm) == sorted(out_norm):
                no_op_reason = "coarse_grained_same_entity_transformation"
            # Proteolysis/no-op style: all outputs already present in inputs, no novel produced token.
            elif set(out_norm).issubset(set(in_norm)):
                no_op_reason = "output_subset_of_input_without_distinct_product"
            if no_op_reason:
                summary["no_op_removed_count"] += 1
                if _lock_id(reaction) or _has_direct_evidence(reaction):
                    _quarantine_reaction(
                        reaction,
                        reaction_index=reaction_index,
                        reason=no_op_reason,
                        count_as_no_op=True,
                    )
                else:
                    summary["unsupported_no_op_dropped_count"] += 1
                    actions = rep.get("actions")
                    if isinstance(actions, list):
                        actions.append(
                            {
                                "type": "reaction_dropped",
                                "reason": "unsupported_no_op",
                                "classification_reason": no_op_reason,
                                "json_pointer": f"/processes/reactions/{reaction_index}",
                                "reaction_name": _canonical(str(reaction.get("name", ""))) or "<unnamed>",
                            }
                        )
                continue
        key = (
            "reaction",
            tuple(sorted(in_norm)),
            tuple(sorted(out_norm)),
            _normalize(str(reaction.get("biological_state", ""))),
            tuple(sorted(_normalize(v) for v in _reaction_modifier_names(reaction))),
        )
        existing = reaction_by_key.get(key)
        if existing is None:
            reaction_by_key[key] = reaction
            reaction_index_by_key[key] = reaction_index
        elif _lock_id(existing) and not _lock_id(reaction):
            reaction_by_key[key] = existing
        elif _lock_id(reaction) and not _lock_id(existing):
            reaction_by_key[key] = reaction
            reaction_index_by_key[key] = reaction_index
        elif (
            _lock_id(existing)
            and _lock_id(reaction)
            and _lock_id(existing) != _lock_id(reaction)
        ):
            selected = _best_record(existing, reaction)
            if selected is existing:
                loser = reaction
                loser_index = reaction_index
            else:
                loser = existing
                loser_index = reaction_index_by_key[key]
                reaction_index_by_key[key] = reaction_index
            reaction_by_key[key] = selected
            _quarantine_reaction(
                loser,
                reaction_index=loser_index,
                reason="duplicate_locked_reaction",
                count_as_no_op=False,
            )
        else:
            reaction_by_key[key] = _best_record(existing, reaction)

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
            value = _actor_name_from_row(row)
            if value:
                transporter_names.append(value)
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

    if quarantined or isinstance(existing_quarantine, list):
        payload["quarantined_locked_reactions"] = quarantined

    active_locked_ids = {
        _lock_id(reaction)
        for reaction in deduped_reactions
        if isinstance(reaction, dict) and _lock_id(reaction)
    }
    quarantined_locked_ids = {
        _lock_id(record)
        for record in quarantined
        if _lock_id(record)
    }
    prior_filter_report = _safe_dict(payload.get("locked_reaction_filter_report"))
    locked_found = max(
        int(prior_filter_report.get("locked_reactions_found") or 0),
        len(active_locked_ids | quarantined_locked_ids),
    )
    if locked_found:
        payload["locked_reaction_filter_report"] = {
            "locked_reactions_found": locked_found,
            "exported_locked_reactions": len(active_locked_ids),
            "quarantined_locked_reactions": len(quarantined_locked_ids),
            "unaccounted_locked_reactions": max(
                0,
                locked_found - len(active_locked_ids | quarantined_locked_ids),
            ),
        }

    rep["summary"]["dedupe_removed_reactions"] += removed_reactions
    rep["summary"]["dedupe_removed_transports"] += removed_transports
    rep["summary"]["dedupe_removed"] += removed_reactions + removed_transports
    rep["summary"]["dedupe_removed_total"] = rep["summary"]["dedupe_removed"]
    return {
        "reactions_removed": removed_reactions,
        "transports_removed": removed_transports,
        "dedupe_removed": removed_reactions + removed_transports,
        "no_op_removed_count": int(rep["summary"].get("no_op_removed_count", 0)),
        "no_op_quarantined_count": int(rep["summary"].get("no_op_quarantined_count", 0)),
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
                enzyme_name = _actor_name_from_row(enzyme)
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
            transporter_name = _actor_name_from_row(transporter)
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
                name = _actor_name_from_row(row)
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
    summary = _safe_dict(rep.setdefault("summary", {}))
    summary.setdefault("n_same_as_groups", 0)
    summary.setdefault("n_aliases_rewritten", 0)
    summary.setdefault("n_entities_deduped", 0)
    summary.setdefault("n_single_protein_complexes_removed", 0)
    summary.setdefault("alias_example_mappings", [])
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
    rep["summary"]["alias_canonicalization"] = {
        "n_same_as_groups": int(rep["summary"].get("n_same_as_groups", 0)),
        "n_aliases_rewritten": int(rep["summary"].get("n_aliases_rewritten", 0)),
        "n_entities_deduped": int(rep["summary"].get("n_entities_deduped", 0)),
        "n_single_protein_complexes_removed": int(rep["summary"].get("n_single_protein_complexes_removed", 0)),
        "example_mappings": _safe_list(rep["summary"].get("alias_example_mappings"))[:10],
    }
    return _safe_dict(rep.get("summary"))


def prune_disconnected_proteins(
    payload: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Remove degree-0 proteins only when they have no external identity."""
    from t2pw.pipeline.qa_graph import build_graph, degrees, get_entities, node

    rep = report if isinstance(report, dict) else None
    summary = _safe_dict(rep.setdefault("summary", {})) if rep is not None else {}
    if rep is not None:
        summary.setdefault("pruned_disconnected_proteins", [])
        summary.setdefault("pruned_disconnected_proteins_count", 0)
        rep.setdefault("actions", [])

    adj, _ = build_graph(payload)
    deg = degrees(adj)
    ents = get_entities(payload)

    component_norms = {
        _normalize(_component_name_from_row(component))
        for complex_row in _safe_list(_safe_dict(payload.get("entities")).get("protein_complexes"))
        if isinstance(complex_row, dict)
        for component in _safe_list(complex_row.get("components"))
        if _normalize(_component_name_from_row(component))
    }
    disconnected = {
        name
        for name in ents.get("proteins", set())
        if deg.get(node("protein", name), 0) == 0 and _normalize(name) not in component_norms
    }
    if not disconnected:
        return []

    proteins_list = _safe_list(_safe_dict(payload.get("entities")).get("proteins"))
    pruned: List[str] = []
    kept: List[Any] = []
    for row in proteins_list:
        if not isinstance(row, dict):
            kept.append(row)
            continue
        name = _canonical(str(row.get("name", "")))
        if name in disconnected and not has_protein_external_identity(row):
            pruned.append(name)
            continue
        kept.append(row)
    payload.setdefault("entities", {})["proteins"] = kept  # type: ignore[index]

    pruned = sorted(pruned)
    if rep is not None:
        summary["pruned_disconnected_proteins"] = pruned
        summary["pruned_disconnected_proteins_count"] = len(pruned)
        actions = rep.setdefault("actions", [])
        if isinstance(actions, list):
            for name in pruned:
                actions.append({"type": "disconnected_protein_pruned", "name": name})
    return pruned


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
    complex_pointer_by_norm = {
        _normalize(_canonical(str(row.get("name", "")))): f"/entities/protein_complexes/{idx}/name"
        for idx, row in enumerate(complexes)
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    }
    protein_registry_norms = _entity_name_norms(proteins) | _entity_name_norms(complexes)
    generated_complex_norms = {
        _normalize(_canonical(str(row.get("name", ""))))
        for row in complexes
        if isinstance(row, dict) and is_generated_complex_wrapper(row) and _canonical(str(row.get("name", "")))
    }
    proteins_by_norm = {
        _normalize(_canonical(str(row.get("name", "")))): row
        for row in proteins
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    }
    proteins_by_uniprot = {
        protein_external_identity(row).casefold(): row
        for row in proteins
        if isinstance(row, dict) and protein_external_identity(row)
    }

    def _add_error(path: str, reason: str) -> None:
        errors.append({"path": path, "reason": reason})

    def _check_forbidden(path: str, token: str) -> None:
        if _normalize(_canonical(token)) in forbidden_norms:
            _add_error(path, f"Forbidden complex reference detected: {token}")

    def _has_positive_component_stoichiometry(component: Any) -> bool:
        if isinstance(component, str):
            return True
        if not isinstance(component, dict):
            return False
        value = component.get("stoichiometry")
        if value in (None, ""):
            return False
        try:
            return int(value) >= 1
        except (TypeError, ValueError):
            return False

    lock_report = payload.get("locked_reaction_filter_report")
    if lock_report is not None:
        lock_pointer = "/locked_reaction_filter_report/unaccounted_locked_reactions"
        if not isinstance(lock_report, dict):
            _add_error(
                "/locked_reaction_filter_report",
                "Locked reaction accounting report must be an object.",
            )
        elif "unaccounted_locked_reactions" in lock_report:
            unaccounted = lock_report.get("unaccounted_locked_reactions")
            if isinstance(unaccounted, bool) or not isinstance(unaccounted, int) or unaccounted < 0:
                _add_error(
                    lock_pointer,
                    "Locked reaction accounting is malformed: "
                    "unaccounted_locked_reactions must be a non-negative integer.",
                )
            elif unaccounted > 0:
                _add_error(
                    lock_pointer,
                    f"Locked reaction accounting failed: {unaccounted} locked reaction(s) are neither active nor quarantined.",
                )

    for duplicate_norm in sorted(set(protein_pointer_by_norm) & set(complex_pointer_by_norm)):
        protein_row = next(
            row
            for row in proteins
            if isinstance(row, dict)
            and _normalize(_canonical(str(row.get("name", "")))) == duplicate_norm
        )
        duplicate_name = _canonical(str(protein_row.get("name", "")))
        _add_error(
            protein_pointer_by_norm[duplicate_norm],
            f"Entity '{duplicate_name}' is declared as both a protein and a protein_complex; "
            f"entity types must be disjoint (complex declaration: {complex_pointer_by_norm[duplicate_norm]}).",
        )

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
    for idx, row in enumerate(_safe_list(entities.get("proteins"))):
        if not isinstance(row, dict):
            continue
        pname = str(row.get("name", "")).strip()
        if not pname:
            continue
        pnorm = _normalize(_canonical(pname))
        if pnorm in generated_complex_norms or pname.casefold().endswith(" complex"):
            _add_error(
                f"/entities/proteins/{idx}",
                f"Generated protein complex wrapper '{pname}' must be listed under protein_complexes, not proteins.",
            )
        species = protein_species_context(row)
        if not species:
            _add_error(
                f"/entities/proteins/{idx}",
                f"Protein '{pname}' is missing species/organism.",
            )
        ext_id = protein_external_identity(row)
        if not ext_id:
            _add_error(
                f"/entities/proteins/{idx}",
                f"Protein '{pname}' is missing a UniProt or DrugBank identifier.",
            )
    for idx, row in enumerate(_safe_list(entities.get("protein_complexes"))):
        if not isinstance(row, dict):
            continue
        _check_forbidden(f"/entities/protein_complexes/{idx}/name", str(row.get("name", "")))
        if not is_generated_complex_wrapper(row):
            continue
        pcname = str(row.get("name") or idx).strip()
        if not protein_species_context(row):
            _add_error(
                f"/entities/protein_complexes/{idx}",
                f"Generated protein complex '{pcname}' is missing species/organism.",
            )
        components = _safe_list(row.get("components"))
        if not components:
            _add_error(
                f"/entities/protein_complexes/{idx}/components",
                f"Generated protein complex '{pcname}' must include at least one protein component.",
            )
            continue
        for cidx, component in enumerate(components):
            cname = _component_name_from_row(component)
            comp_identity = protein_external_identity(component)
            if not _has_positive_component_stoichiometry(component):
                _add_error(
                    f"/entities/protein_complexes/{idx}/components/{cidx}",
                    f"Generated protein complex '{pcname}' component '{cname or cidx}' is missing positive stoichiometry.",
                )
            match = proteins_by_norm.get(_normalize(_canonical(cname))) if cname else None
            if match is None and comp_identity:
                match = proteins_by_uniprot.get(comp_identity.casefold())
            if match is None:
                _add_error(
                    f"/entities/protein_complexes/{idx}/components/{cidx}",
                    f"Generated protein complex '{pcname}' component '{cname or cidx}' does not resolve to a declared protein.",
                )
                continue
            if not protein_species_context(match):
                _add_error(
                    f"/entities/protein_complexes/{idx}/components/{cidx}",
                    f"Generated protein complex '{pcname}' component protein '{match.get('name')}' is missing species/organism.",
                )
            if not protein_external_identity(match):
                _add_error(
                    f"/entities/protein_complexes/{idx}/components/{cidx}",
                    f"Generated protein complex '{pcname}' component protein '{match.get('name')}' is missing a UniProt or DrugBank identifier.",
                )

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
                actor_name = _actor_name_from_row(actor)
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
            actor_name = _actor_name_from_row(actor)
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

    from t2pw.pipeline.qa_graph import build_graph, connected_components, degrees, get_entities, node

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

    # Proteins declared as components of any protein_complex are degree-0 by design:
    # their network connection flows through the complex, not directly from reactions.
    _complex_component_norms: Set[str] = {
        _normalize(_component_name_from_row(comp))
        for pc_row in complexes
        if isinstance(pc_row, dict)
        for comp in _safe_list(pc_row.get("components"))
        if _normalize(_component_name_from_row(comp))
    }
    if enforce_all_proteins_connected and proteins_degree0 > 0:
        for protein_name in sorted(ents.get("proteins", set())):
            pnode = node("protein", protein_name)
            if deg.get(pnode, 0) == 0:
                pnorm = _normalize(protein_name)
                if pnorm in _complex_component_norms:
                    continue
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


def normalize_process_payload(
    payload: Dict[str, Any],
    *,
    on_checkpoint: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = deepcopy(payload)
    report = _new_report()

    def _checkpoint(name: str) -> None:
        if on_checkpoint is not None:
            on_checkpoint(name, data, report)

    force_reactions_non_spontaneous(data, report=report)
    _checkpoint("force_reactions_non_spontaneous")
    # Step 0 — collapse biochemical synonyms before any other pass so that all
    # downstream logic sees consistent canonical names (Issue 3).
    apply_biochemical_aliases(data, report=report)
    _checkpoint("apply_biochemical_aliases")
    normalize_composites(data, report=report)
    _checkpoint("normalize_composites")
    rewrite_reactions_to_complex_states(data, report=report)
    _checkpoint("rewrite_reactions_to_complex_states")
    cleanup_disallowed_complexes(data, report=report)
    _checkpoint("cleanup_disallowed_complexes")
    ensure_autostates(data, report=report)
    _checkpoint("ensure_autostates")
    # Backfill missing reaction compartments from participant entity locations (Issue 4).
    backfill_reaction_compartments(data, report=report)
    _checkpoint("backfill_reaction_compartments")
    attach_transporters_from_evidence(data, report=report)
    _checkpoint("attach_transporters_from_evidence")
    attach_enzymes_from_reaction_evidence(data, report=report)
    _checkpoint("attach_enzymes_from_reaction_evidence")
    promote_interaction_enzymes(data, report=report)
    _checkpoint("promote_interaction_enzymes")
    promote_catalysts(data, report=report)
    _checkpoint("promote_catalysts")
    canonicalize_same_as_aliases(data, report=report)
    _checkpoint("canonicalize_same_as_aliases")
    normalize_process_actor_schema(data, report=report)
    _checkpoint("normalize_process_actor_schema")
    drop_unresolved_complex_component_proteins(data, report=report)
    _checkpoint("drop_unresolved_complex_component_proteins")
    dedupe_processes(data, report=report)
    _checkpoint("dedupe_processes")
    # Reaction classification must run before orphan cleanup. Otherwise a protein
    # attached only to a rejected/quarantined reaction remains long enough to fail
    # the final identity/connectivity gate even though it is no longer exportable.
    drop_process_orphan_proteins(data, report=report)
    _checkpoint("drop_process_orphan_proteins")
    prune_disconnected_proteins(data, report=report)
    _checkpoint("prune_disconnected_proteins")
    try:
        gate_details = run_strict_post_normalization_gates(
            data,
            report=report,
            enforce_all_proteins_connected=True,
        )
        report["gate"] = {"ok": True, **gate_details}
    except GateValidationError as exc:
        gate_details = _safe_dict(exc.details)
        report["gate"] = {"ok": False, **gate_details}
        report.setdefault("actions", []).append(
            {
                "type": "normalization_gate_failed",
                "error_count": len(_safe_list(gate_details.get("errors"))),
            }
        )
    report["gate_details"] = report["gate"]
    _safe_dict(report.setdefault("summary", {}))["gate_error_count"] = len(
        _safe_list(_safe_dict(report.get("gate")).get("errors"))
    )
    _checkpoint("run_strict_post_normalization_gates")
    from t2pw.pipeline.stage_contracts import validate_post_normalization

    actor_contract = validate_post_normalization(data, {"ok": True, "errors": []})
    assert actor_contract.get("ok") is True, actor_contract.get("errors")
    return data, report
