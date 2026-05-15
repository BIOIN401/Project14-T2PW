from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from t2pw.curation.apply_audit_patch import apply_patch_with_policy
try:
    from t2pw.llm.client import chat, chat_with_tools
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when optional LLM deps are absent
    _LLM_IMPORT_ERROR = exc

    def chat(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError(f"LLM client dependencies are not available: {_LLM_IMPORT_ERROR}") from _LLM_IMPORT_ERROR

    def chat_with_tools(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError(f"LLM client dependencies are not available: {_LLM_IMPORT_ERROR}") from _LLM_IMPORT_ERROR
from t2pw.paths import PROMPTS_DIR
from t2pw.mapping.map_ids import (
    HttpClient,
    PathBankDbResolver,
    lookup_compound_api_background,
    lookup_hmdb_background,
    lookup_protein_api_background,
    map_compound_all,
    map_protein_uniprot,
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (value or "").strip().casefold()))


def _canonical(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize(value)).strip("_") or "state"


def _extract_global_organism(payload: Dict[str, Any]) -> str:
    entities = _safe_dict(payload.get("entities"))
    species_names = [
        (item.get("name") or "").strip()
        for item in _safe_list(entities.get("species"))
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
    ]
    if len(species_names) == 1:
        return species_names[0]
    biological_states = _safe_list(payload.get("biological_states"))
    state_species = {
        (state.get("species") or "").strip()
        for state in biological_states
        if isinstance(state, dict) and isinstance(state.get("species"), str) and state.get("species").strip()
    }
    if len(state_species) == 1:
        return sorted(state_species)[0]
    return ""


def infer_entity_species(
    payload: Dict[str, Any],
    *,
    entity_type: str,
    entity_name: str,
    use_llm: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 450,
) -> Dict[str, Any]:
    """Infer a species name for a protein/protein-complex gap.

    This is intentionally narrow so ID mapping can call it as a species-only
    gap resolver without invoking the broader stage-3 enrichment workflow.
    """
    name = _canonical(entity_name)
    kind = _canonical(entity_type).lower()
    if not name or kind not in {"protein", "protein_complex"}:
        return {"status": "unmapped", "reason": "invalid_entity", "name": "", "confidence": 0.0}
    if not use_llm:
        return {"status": "unmapped", "reason": "llm_disabled", "name": "", "confidence": 0.0}

    entities = _safe_dict(payload.get("entities"))
    species_rows = [
        {
            "name": _canonical(str(row.get("name", ""))),
            "taxonomy_id": _canonical(str(row.get("taxonomy_id") or row.get("taxonomy-id") or "")),
            "pathbank_species_id": row.get("pathbank_species_id") or row.get("species_id") or row.get("pathwhiz_id"),
        }
        for row in _safe_list(entities.get("species"))
        if isinstance(row, dict) and _canonical(str(row.get("name", "")))
    ][:8]
    biological_states = [
        {
            "name": _canonical(str(state.get("name", ""))),
            "species": _canonical(str(state.get("species") or state.get("organism") or state.get("species_name") or "")),
            "subcellular_location": _canonical(str(state.get("subcellular_location", ""))),
        }
        for state in _safe_list(payload.get("biological_states"))
        if isinstance(state, dict)
    ][:12]
    metadata = _safe_dict(payload.get("metadata"))
    processes = _safe_dict(payload.get("processes"))
    reaction_evidence: List[Dict[str, Any]] = []
    for reaction in _safe_list(processes.get("reactions"))[:12]:
        if not isinstance(reaction, dict):
            continue
        enzymes = [
            enz for enz in _safe_list(reaction.get("enzymes"))
            if isinstance(enz, dict) and name.casefold() in json.dumps(enz, ensure_ascii=False).casefold()
        ]
        if enzymes:
            reaction_evidence.append(
                {
                    "reaction": _canonical(str(reaction.get("name", ""))),
                    "evidence": _canonical(str(reaction.get("evidence", "")))[:280],
                    "enzymes": enzymes[:3],
                }
            )
    prompt = {
        "task": "Infer the organism/species for this protein or protein complex.",
        "entity": {"type": kind, "name": name},
        "metadata": metadata,
        "known_pathway_species": species_rows,
        "biological_states": biological_states,
        "reaction_evidence": reaction_evidence[:6],
        "rules": [
            "Return JSON only with keys: name, taxonomy_id, confidence, reason.",
            "Use a known_pathway_species name when the context clearly indicates it.",
            "If the species cannot be inferred, return an empty name and confidence 0.",
            "Do not infer a species from generic protein names alone.",
        ],
    }
    system = "You are a strict biological species resolver. Return only compact JSON."
    raw = ""
    try:
        raw = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_json=True,
        )
        parsed = _extract_json_object(raw) or {}
        species_name = _canonical(str(parsed.get("name") or parsed.get("species") or ""))
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
        if not species_name or confidence < 0.55:
            return {
                "status": "unmapped",
                "reason": str(parsed.get("reason") or "low_confidence"),
                "name": "",
                "confidence": confidence,
                "raw": raw[:400],
            }
        return {
            "status": "mapped",
            "name": species_name,
            "taxonomy_id": _canonical(str(parsed.get("taxonomy_id") or "")),
            "confidence": confidence,
            "reason": _canonical(str(parsed.get("reason") or "llm_inferred_species")),
            "raw": raw[:400],
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "unmapped", "reason": f"llm_error:{exc}", "name": "", "confidence": 0.0, "raw": raw[:400]}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.replace("```json", "```").replace("```", "").strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _index_locations(payload: Dict[str, Any], *, key: str, field: str) -> Dict[str, List[Dict[str, Any]]]:
    rows = _safe_list(_safe_dict(payload.get("element_locations")).get(key))
    out: Dict[str, List[Dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = (row.get(field) or "").strip() if isinstance(row.get(field), str) else ""
        if not name:
            continue
        out.setdefault(name, []).append({"index": idx, "row": row})
    return out


def _state_context_key(
    *,
    species: str = "",
    subcellular_location: str = "",
    cell_type: str = "",
    tissue: str = "",
) -> Tuple[str, str, str, str]:
    return (
        _normalize(species),
        _normalize(subcellular_location),
        _normalize(cell_type),
        _normalize(tissue),
    )


def _state_maps(payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str, str, str], str]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    by_context: Dict[Tuple[str, str, str, str], str] = {}
    for state in _safe_list(payload.get("biological_states")):
        if not isinstance(state, dict):
            continue
        name = (state.get("name") or "").strip() if isinstance(state.get("name"), str) else ""
        if not name:
            continue
        by_name[name] = state
        location = (state.get("subcellular_location") or "").strip() if isinstance(state.get("subcellular_location"), str) else ""
        species = (state.get("species") or state.get("organism") or "").strip() if isinstance(state.get("species") or state.get("organism"), str) else ""
        cell_type = (state.get("cell_type") or "").strip() if isinstance(state.get("cell_type"), str) else ""
        tissue = (state.get("tissue") or "").strip() if isinstance(state.get("tissue"), str) else ""
        if species and location:
            by_context.setdefault(
                _state_context_key(
                    species=species,
                    subcellular_location=location,
                    cell_type=cell_type,
                    tissue=tissue,
                ),
                name,
            )
    return by_name, by_context


_CANONICAL_COMPARTMENT_VOCAB = {
    "cytosol", "nucleus", "mitochondrion", "mitochondrial_matrix",
    "endoplasmic_reticulum", "golgi", "lysosome", "peroxisome",
    "plasma_membrane", "extracellular", "endosome",
}

# Maps common synonyms/aliases to canonical compartment names.
_COMPARTMENT_ALIAS_MAP: Dict[str, str] = {
    "cytoplasm": "cytosol",
    "cytosolic": "cytosol",
    "cytoplasmic": "cytosol",
    "mitochondria": "mitochondrion",
    "mitochondrial": "mitochondrion",
    "mitochondrial matrix": "mitochondrial_matrix",
    "er": "endoplasmic_reticulum",
    "endoplasmic reticulum": "endoplasmic_reticulum",
    "golgi apparatus": "golgi",
    "golgi body": "golgi",
    "cell membrane": "plasma_membrane",
    "plasma membrane": "plasma_membrane",
    "extracellular space": "extracellular",
    "extracellular matrix": "extracellular",
    "nuclear": "nucleus",
    "peroxisomal": "peroxisome",
    "lysosomal": "lysosome",
    "endosomal": "endosome",
}


def _resolve_canonical_compartment(location: str) -> str:
    """Return canonical compartment name for location, or empty string if no match."""
    if not location:
        return ""
    norm = location.strip().lower()
    if norm in _CANONICAL_COMPARTMENT_VOCAB:
        return norm
    alias = _COMPARTMENT_ALIAS_MAP.get(norm, "")
    if alias:
        return alias
    # Try matching against vocab by checking if location contains a vocab term.
    for term in _CANONICAL_COMPARTMENT_VOCAB:
        if term in norm:
            return term
    return ""


def _ensure_biological_state(
    payload: Dict[str, Any],
    location: str,
    species: str,
    *,
    cell_type: str = "",
    tissue: str = "",
) -> str:
    states = payload.setdefault("biological_states", [])
    if not isinstance(states, list):
        states = []
        payload["biological_states"] = states
    by_name, by_context = _state_maps(payload)
    context_key = _state_context_key(
        species=species,
        subcellular_location=location,
        cell_type=cell_type,
        tissue=tissue,
    )
    existing_name = by_context.get(context_key)
    if existing_name:
        return existing_name
    if not _canonical(species) or not _canonical(location):
        return ""
    name_parts = [species, location, cell_type, tissue]
    candidate_name = f"AutoState_{_slug('_'.join(part for part in name_parts if _canonical(part)))}"
    used = set(by_name.keys())
    if candidate_name in used:
        i = 2
        while f"{candidate_name}_{i}" in used:
            i += 1
        candidate_name = f"{candidate_name}_{i}"
    canonical = _resolve_canonical_compartment(location)
    state_obj: Dict[str, Any] = {"name": candidate_name, "subcellular_location": _canonical(location)}
    if canonical:
        state_obj["compartment_canonical"] = canonical
    if species:
        state_obj["species"] = species
    if cell_type:
        state_obj["cell_type"] = cell_type
    if tissue:
        state_obj["tissue"] = tissue
    states.append(state_obj)
    return candidate_name


def _db_location_candidates(db: Optional[PathBankDbResolver], *, kind: str, name: str, max_items: int = 6) -> List[Dict[str, Any]]:
    if not db or not db.available():
        return []
    term = _canonical(name)
    if not term:
        return []
    if kind == "compound":
        sql = (
            "SELECT sl.name AS location, COUNT(*) AS freq "
            "FROM compounds c "
            "JOIN compound_locations cl ON cl.compound_id = c.id "
            "JOIN biological_states bs ON bs.id = cl.biological_state_id "
            "JOIN subcellular_locations sl ON sl.id = bs.subcellular_location_id "
            "WHERE LOWER(c.name)=LOWER(%s) "
            "   OR LOWER(c.short_name)=LOWER(%s) "
            "   OR LOWER(c.synonyms) LIKE LOWER(%s) "
            "GROUP BY sl.name "
            "ORDER BY freq DESC "
            f"LIMIT {int(max_items)}"
        )
    else:
        sql = (
            "SELECT sl.name AS location, COUNT(*) AS freq "
            "FROM proteins p "
            "JOIN protein_locations pl ON pl.protein_id = p.id "
            "JOIN biological_states bs ON bs.id = pl.biological_state_id "
            "JOIN subcellular_locations sl ON sl.id = bs.subcellular_location_id "
            "WHERE LOWER(p.name)=LOWER(%s) "
            "   OR LOWER(p.gene_name)=LOWER(%s) "
            "   OR LOWER(p.synonyms) LIKE LOWER(%s) "
            "GROUP BY sl.name "
            "ORDER BY freq DESC "
            f"LIMIT {int(max_items)}"
        )
    rows = db._query(sql, (term, term, f"%{term}%"))  # pylint: disable=protected-access
    out: List[Dict[str, Any]] = []
    for row in rows:
        loc = (row.get("location") or "").strip() if isinstance(row.get("location"), str) else ""
        if not loc:
            continue
        out.append(
            {
                "location": loc,
                "score": float(row.get("freq") or 0.0),
                "source": "pathbank_db",
                "evidence": f"location_frequency={int(row.get('freq') or 0)}",
            }
        )
    return out


def _llm_choose_location(
    *,
    kind: str,
    name: str,
    candidates: List[Dict[str, Any]],
    use_llm: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if not candidates:
        return {"choice": "", "confidence": 0.0, "reason": "no_candidates", "source": "none"}
    if len(candidates) == 1:
        only = candidates[0]
        return {
            "choice": only.get("location", ""),
            "confidence": 0.95,
            "reason": "single_candidate",
            "source": "deterministic",
        }
    if not use_llm:
        top = candidates[0]
        return {
            "choice": top.get("location", ""),
            "confidence": min(0.95, 0.55 + 0.08 * len(candidates)),
            "reason": "deterministic_top_candidate",
            "source": "deterministic",
        }

    prompt = {
        "task": "Choose best subcellular location for missing entity location.",
        "entity_type": kind,
        "entity_name": name,
        "candidate_locations": candidates,
        "rules": [
            "Pick exactly one candidate location from candidate_locations.",
            "Do not invent locations outside candidate list.",
            "Prefer higher evidence score and biological plausibility.",
            "Return JSON only with keys: choice, confidence, reason.",
        ],
    }
    system = (
        "You are a strict location resolver. Return only JSON. "
        "Never invent a location not present in candidate list."
    )
    try:
        raw = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_json=True,
        )
        parsed = _extract_json_object(raw) or {}
        choice = _canonical(str(parsed.get("choice", "")))
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        reason = str(parsed.get("reason", "") or "").strip()
        if choice and any(_normalize(choice) == _normalize(str(c.get("location", ""))) for c in candidates):
            return {
                "choice": choice,
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": reason or "llm_selected_candidate",
                "source": "llm",
                "raw": raw[:400],
            }
        return {
            "choice": candidates[0].get("location", ""),
            "confidence": 0.55,
            "reason": "llm_invalid_choice_fallback_top",
            "source": "llm_fallback",
            "raw": raw[:400],
        }
    except Exception as exc:  # noqa: BLE001
        top = candidates[0]
        return {
            "choice": top.get("location", ""),
            "confidence": 0.55,
            "reason": f"llm_error_fallback:{exc}",
            "source": "deterministic_fallback",
        }


def _issue_key(kind: str, name: str) -> str:
    return f"{kind}:{_normalize(name)}"


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return value > 0
    return bool(value)


def _first_present(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if _present(value):
            return value
    return None


def _row_species_name(row: Dict[str, Any]) -> str:
    ref = _safe_dict(row.get("species_ref"))
    return _canonical(
        str(
            row.get("species")
            or row.get("organism")
            or row.get("species_name")
            or ref.get("name")
            or ""
        )
    )


def _has_species_ref(row: Dict[str, Any]) -> bool:
    ref = _safe_dict(row.get("species_ref"))
    return bool(
        _row_species_name(row)
        or _first_present(row, "pathbank_species_id", "species_id", "pw_species_id")
        or _first_present(ref, "pathbank_species_id", "species_id", "pw_species_id")
    )


def _has_entity_db_id(row: Dict[str, Any], kind: str) -> bool:
    if kind == "species":
        return bool(_first_present(row, "pathbank_species_id", "species_id", "pw_species_id", "pathwhiz_id"))
    if kind == "compound":
        return bool(
            _first_present(
                row,
                "pathbank_compound_id",
                "pw_compound_id",
                "pwc_id",
                "pathwhiz_id",
            )
            or _safe_dict(row.get("mapped_ids"))
        )
    if kind == "protein":
        return bool(
            _first_present(row, "pathbank_protein_id", "pw_protein_id", "pathwhiz_id")
            or _safe_dict(row.get("mapped_ids"))
        )
    if kind == "protein_complex":
        return bool(
            _first_present(
                row,
                "pathbank_protein_complex_id",
                "pathbank_complex_id",
                "pw_complex_id",
                "pathwhiz_id",
            )
            or _safe_dict(row.get("mapped_ids"))
        )
    return bool(_safe_dict(row.get("mapped_ids")))


def _component_name(value: Any) -> str:
    if isinstance(value, str):
        return _canonical(value)
    if isinstance(value, dict):
        return _canonical(
            str(
                value.get("name")
                or value.get("protein")
                or value.get("component")
                or value.get("entity")
                or ""
            )
        )
    return ""


def _component_has_stoichiometry(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return _present(value.get("stoichiometry")) or _present(value.get("coefficient"))


def _component_is_resolved(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if _first_present(value, "pathbank_protein_id", "pw_protein_id", "pathwhiz_id"):
        return True
    mapped_ids = _safe_dict(value.get("mapped_ids"))
    if mapped_ids.get("uniprot") or mapped_ids.get("pathbank_protein_id") or mapped_ids.get("pw_protein_id"):
        return True
    return str(value.get("mapping_status") or "").strip().lower() == "mapped"


def _indexed_locations(payload: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    locations = _safe_dict(payload.get("element_locations"))
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "compound": {},
        "protein": {},
        "protein_complex": {},
    }
    specs = [
        ("compound_locations", "compound", "compound"),
        ("protein_locations", "protein", "protein"),
        ("protein_locations", "protein_complex", "protein_complex"),
        # Protein complexes are still often stored in the legacy protein key.
        ("protein_locations", "protein_complex", "protein"),
    ]
    for list_key, kind, name_field in specs:
        for idx, row in enumerate(_safe_list(locations.get(list_key))):
            if not isinstance(row, dict):
                continue
            name = _canonical(str(row.get(name_field) or ""))
            if not name:
                continue
            out.setdefault(kind, {}).setdefault(name, []).append({"index": idx, "row": row, "list_key": list_key})
    return out


def _location_gap_fields(location_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    has_location_row = bool(location_rows)
    has_location_state = any(
        isinstance(_safe_dict(wrap.get("row")).get("biological_state"), str)
        and _safe_dict(wrap.get("row")).get("biological_state", "").strip()
        for wrap in location_rows
    )
    return {
        "needs_location_link": not has_location_row,
        "needs_location_state_fill": has_location_row and not has_location_state,
        "visible_entity_missing_location": not has_location_row,
    }


def _compound_unmapped_after_db_pass(row: Dict[str, Any]) -> bool:
    if _first_present(row, "pathbank_compound_id", "pw_compound_id", "pwc_id", "pathwhiz_id"):
        return False
    meta = _safe_dict(row.get("mapping_meta"))
    resolution = _safe_dict(meta.get("resolution"))
    providers = meta.get("providers")
    provider_text = " ".join(str(p) for p in providers) if isinstance(providers, list) else str(meta.get("provider") or "")
    source_text = str(meta.get("source") or "")
    db_attempted = bool(resolution or "pathbank" in provider_text.casefold() or source_text.casefold() == "db")
    if not db_attempted:
        return False
    status = str(resolution.get("status") or "").strip().lower()
    issue = str(resolution.get("issue") or "").strip().lower()
    return status in {"unresolved", "ambiguous"} or issue in {"no_db_candidates", "db_unavailable"}


def _direct_protein_complex_enzyme_issues(
    payload: Dict[str, Any],
    *,
    protein_names: set[str],
    complex_names: set[str],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    processes = _safe_dict(payload.get("processes"))
    complexish = re.compile(r"\b(complex|holoenzyme|heterodimer|homodimer|multimer|subunit)\b|:", re.IGNORECASE)
    for reaction_idx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        reaction_name = _canonical(str(reaction.get("name") or reaction.get("id") or f"reaction_{reaction_idx}"))
        rows: List[Tuple[str, int, Dict[str, Any]]] = []
        rows.extend(("enzymes", idx, row) for idx, row in enumerate(_safe_list(reaction.get("enzymes"))) if isinstance(row, dict))
        rows.extend(("modifiers", idx, row) for idx, row in enumerate(_safe_list(reaction.get("modifiers"))) if isinstance(row, dict))
        for list_key, row_idx, row in rows:
            entity_name = _canonical(str(row.get("protein") or row.get("protein_name") or ""))
            if not entity_name and str(row.get("entity_type") or "").strip().lower() == "protein":
                entity_name = _canonical(str(row.get("entity") or row.get("name") or ""))
            if not entity_name:
                continue
            norm = _normalize(entity_name)
            requires_complex = norm in complex_names or bool(complexish.search(entity_name))
            represented_as_protein = "protein" in row or str(row.get("entity_type") or "").strip().lower() == "protein"
            if represented_as_protein and requires_complex:
                issues.append(
                    {
                        "issue_key": f"enzyme_direct_protein_requires_complex:{reaction_idx}:{row_idx}:{_normalize(entity_name)}",
                        "entity_type": "reaction_modifier",
                        "name": entity_name,
                        "reaction": reaction_name,
                        "path": f"/processes/reactions/{reaction_idx}/{list_key}/{row_idx}",
                        "enzyme_still_direct_protein_when_complex_required": True,
                        "protein_entity_exists": norm in protein_names,
                        "protein_complex_entity_exists": norm in complex_names,
                        "missing_fields": ["protein_complex_reference"],
                        "reasons": ["enzyme_direct_protein_requires_complex"],
                    }
                )
    return issues


def _collect_stage3_issues(payload: Dict[str, Any], *, max_items: int) -> List[Dict[str, Any]]:
    entities = _safe_dict(payload.get("entities"))
    locations_by_kind = _indexed_locations(payload)
    issues: List[Dict[str, Any]] = []

    for idx, item in enumerate(_safe_list(entities.get("species"))):
        if not isinstance(item, dict):
            continue
        name = _canonical(str(item.get("name") or item.get("common_name") or item.get("taxonomy_id") or f"species_{idx}"))
        if not _has_entity_db_id(item, "species"):
            issues.append(
                {
                    "issue_key": _issue_key("species", name),
                    "entity_type": "species",
                    "name": name,
                    "path": f"/entities/species/{idx}",
                    "needs_id_mapping": True,
                    "species_missing_db_id": True,
                    "taxonomy_id": _canonical(str(item.get("taxonomy_id") or "")),
                    "missing_fields": ["pathbank_species_id"],
                    "reasons": ["species_missing_db_id"],
                }
            )

    for idx, state in enumerate(_safe_list(payload.get("biological_states"))):
        if not isinstance(state, dict):
            continue
        name = _canonical(str(state.get("name") or f"biological_state_{idx}"))
        missing_fields: List[str] = []
        reasons: List[str] = []
        needs_species = not _has_species_ref(state)
        needs_subcellular_location = not _canonical(str(state.get("subcellular_location") or ""))
        if needs_species:
            missing_fields.append("species")
            reasons.append("biological_state_missing_species")
        if needs_subcellular_location:
            missing_fields.append("subcellular_location")
            reasons.append("biological_state_missing_subcellular_location")
        if missing_fields:
            issues.append(
                {
                    "issue_key": _issue_key("biological_state", name),
                    "entity_type": "biological_state",
                    "name": name,
                    "path": f"/biological_states/{idx}",
                    "needs_species": needs_species,
                    "needs_subcellular_location": needs_subcellular_location,
                    "missing_fields": missing_fields,
                    "reasons": reasons,
                }
            )

    for kind, rows in [("protein", _safe_list(entities.get("proteins"))), ("compound", _safe_list(entities.get("compounds")))]:
        for idx, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            name = _canonical(str(item.get("name", "")))
            if not name:
                continue
            location_fields = _location_gap_fields(locations_by_kind.get(kind, {}).get(name, []))
            needs_id_mapping = not _has_entity_db_id(item, kind)
            needs_species = kind == "protein" and not _has_species_ref(item)
            compound_missing_class_type = bool(
                kind == "compound"
                and not _canonical(str(item.get("class") or item.get("type") or item.get("compound_class") or item.get("compound_type") or ""))
            )
            compound_unmapped_after_db_pass = bool(kind == "compound" and _compound_unmapped_after_db_pass(item))
            missing_fields: List[str] = []
            reasons: List[str] = []
            if needs_id_mapping:
                missing_fields.append("mapped_ids")
                reasons.append("entity_missing_id_mapping")
            if needs_species:
                missing_fields.append("species")
                reasons.append("protein_missing_species")
            if compound_missing_class_type:
                missing_fields.append("class_or_type")
                reasons.append("compound_missing_class_type")
            if compound_unmapped_after_db_pass:
                reasons.append("compound_unmapped_after_db_pass")
            if bool(location_fields["needs_location_link"]):
                missing_fields.append("location")
                reasons.append("visible_entity_missing_location")
            if bool(location_fields["needs_location_state_fill"]):
                missing_fields.append("biological_state")
                reasons.append("location_row_missing_biological_state")
            issue = {
                "issue_key": _issue_key(kind, name),
                "entity_type": kind,
                "name": name,
                "path": f"/entities/{'proteins' if kind == 'protein' else 'compounds'}/{idx}",
                "needs_id_mapping": needs_id_mapping,
                "needs_location_link": bool(location_fields["needs_location_link"]),
                "needs_location_state_fill": bool(location_fields["needs_location_state_fill"]),
                "visible_entity_missing_location": bool(location_fields["visible_entity_missing_location"]),
                "needs_organism": needs_species,
                "needs_species": needs_species,
                "compound_missing_class_type": compound_missing_class_type,
                "compound_unmapped_after_db_pass": compound_unmapped_after_db_pass,
                "missing_fields": missing_fields,
                "reasons": reasons,
            }
            if reasons:
                issues.append(issue)

    for idx, item in enumerate(_safe_list(entities.get("protein_complexes"))):
        if not isinstance(item, dict):
            continue
        name = _canonical(str(item.get("name", "")))
        if not name:
            continue
        location_fields = _location_gap_fields(locations_by_kind.get("protein_complex", {}).get(name, []))
        components = _safe_list(item.get("components"))
        missing_stoich = []
        unresolved_components = []
        for component_idx, component in enumerate(components):
            component_name = _component_name(component) or f"component_{component_idx}"
            if not _component_has_stoichiometry(component):
                missing_stoich.append(
                    {"component_index": component_idx, "component": component_name, "path": f"/entities/protein_complexes/{idx}/components/{component_idx}"}
                )
            if not _component_is_resolved(component):
                unresolved_components.append(
                    {"component_index": component_idx, "component": component_name, "path": f"/entities/protein_complexes/{idx}/components/{component_idx}"}
                )
        needs_species = not _has_species_ref(item)
        missing_components = not bool(components)
        needs_id_mapping = not _has_entity_db_id(item, "protein_complex")
        missing_fields = []
        reasons = []
        if needs_id_mapping:
            missing_fields.append("pathbank_protein_complex_id")
            reasons.append("protein_complex_missing_id_mapping")
        if needs_species:
            missing_fields.append("species")
            reasons.append("protein_complex_missing_species")
        if missing_components:
            missing_fields.append("components")
            reasons.append("protein_complex_missing_components")
        if missing_stoich:
            missing_fields.append("components.stoichiometry")
            reasons.append("component_missing_stoichiometry")
        if unresolved_components:
            missing_fields.append("components.pathbank_protein_id")
            reasons.append("component_protein_unresolved")
        if bool(location_fields["needs_location_link"]):
            missing_fields.append("location")
            reasons.append("visible_entity_missing_location")
        if bool(location_fields["needs_location_state_fill"]):
            missing_fields.append("biological_state")
            reasons.append("location_row_missing_biological_state")
        if reasons:
            issues.append(
                {
                    "issue_key": _issue_key("protein_complex", name),
                    "entity_type": "protein_complex",
                    "name": name,
                    "path": f"/entities/protein_complexes/{idx}",
                    "needs_id_mapping": needs_id_mapping,
                    "needs_species": needs_species,
                    "needs_location_link": bool(location_fields["needs_location_link"]),
                    "needs_location_state_fill": bool(location_fields["needs_location_state_fill"]),
                    "visible_entity_missing_location": bool(location_fields["visible_entity_missing_location"]),
                    "protein_complex_missing_components": missing_components,
                    "component_missing_stoichiometry": bool(missing_stoich),
                    "component_protein_unresolved": bool(unresolved_components),
                    "missing_component_stoichiometry": missing_stoich,
                    "unresolved_components": unresolved_components,
                    "missing_fields": missing_fields,
                    "reasons": reasons,
                }
            )

    protein_names = {
        _normalize(str(item.get("name") or ""))
        for item in _safe_list(entities.get("proteins"))
        if isinstance(item, dict) and _canonical(str(item.get("name") or ""))
    }
    complex_names = {
        _normalize(str(item.get("name") or ""))
        for item in _safe_list(entities.get("protein_complexes"))
        if isinstance(item, dict) and _canonical(str(item.get("name") or ""))
    }
    issues.extend(_direct_protein_complex_enzyme_issues(payload, protein_names=protein_names, complex_names=complex_names))
    return issues[: max(1, int(max_items))]


def _default_stage3_plan(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for issue in issues:
        kind = _canonical(str(issue.get("entity_type", ""))).lower()
        ops.append(
            {
                "issue_key": issue.get("issue_key", ""),
                "resolve_ids": {"strategy": "db_then_api"},
                "resolve_location": {"strategy": "db_then_default"},
                "background": {"api_lookup": "auto", "hmdb_lookup": bool(kind == "compound"), "max_results": 6},
                "rationale": "default_plan",
            }
        )
    return ops


def _llm_plan_stage3(
    issues: List[Dict[str, Any]],
    *,
    use_llm: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if not issues or not use_llm:
        return {"source": "deterministic_default", "operations": _default_stage3_plan(issues), "raw": ""}

    prompt_obj = {
        "task": "Plan DB/API resolution operations for entity gaps.",
        "issues": issues,
        "rules": [
            "Return JSON only.",
            "Do not invent issue_key values outside provided list.",
            "For resolve_ids.strategy use one of: db_then_api, api_then_db, db_only, api_only, skip.",
            "For resolve_location.strategy use one of: db_then_default, default_only, skip.",
            "For background.api_lookup use one of: auto, none, full.",
            "For background.hmdb_lookup use true only for compounds when additional ID context is useful.",
            "For background.max_results use an integer from 1 to 12.",
            "Plan should prefer deterministic evidence sources and minimize API calls.",
        ],
        "output_schema": {
            "operations": [
                {
                    "issue_key": "string",
                    "resolve_ids": {"strategy": "db_then_api|api_then_db|db_only|api_only|skip"},
                    "resolve_location": {"strategy": "db_then_default|default_only|skip"},
                    "background": {"api_lookup": "auto|none|full", "hmdb_lookup": "boolean", "max_results": "integer"},
                    "rationale": "string",
                }
            ]
        },
    }
    system = "You are a strict planner for deterministic DB/API resolution. Output JSON only."
    raw = ""
    try:
        raw = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt_obj, ensure_ascii=False)},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_json=True,
        )
        parsed = _extract_json_object(raw) or {}
        operations_raw = _safe_list(parsed.get("operations"))
        allowed_issue_keys = {str(i.get("issue_key", "")) for i in issues}
        out_ops: List[Dict[str, Any]] = []
        for op in operations_raw:
            if not isinstance(op, dict):
                continue
            issue_key = _canonical(str(op.get("issue_key", "")))
            if issue_key not in allowed_issue_keys:
                continue
            issue_obj = next((it for it in issues if str(it.get("issue_key", "")) == issue_key), {})
            issue_kind = _canonical(str(_safe_dict(issue_obj).get("entity_type", ""))).lower()
            ids_strategy = _canonical(str(_safe_dict(op.get("resolve_ids")).get("strategy", "db_then_api"))).lower()
            if ids_strategy not in {"db_then_api", "api_then_db", "db_only", "api_only", "skip"}:
                ids_strategy = "db_then_api"
            loc_strategy = _canonical(str(_safe_dict(op.get("resolve_location")).get("strategy", "db_then_default"))).lower()
            if loc_strategy not in {"db_then_default", "default_only", "skip"}:
                loc_strategy = "db_then_default"
            bg_raw = _safe_dict(op.get("background"))
            bg_api_lookup = _canonical(str(bg_raw.get("api_lookup", "auto"))).lower()
            if bg_api_lookup not in {"auto", "none", "full"}:
                bg_api_lookup = "auto"
            bg_hmdb_lookup = bool(bg_raw.get("hmdb_lookup", bool(issue_kind == "compound")))
            bg_max_results = int(bg_raw.get("max_results", 6) or 6)
            bg_max_results = max(1, min(12, bg_max_results))
            out_ops.append(
                {
                    "issue_key": issue_key,
                    "resolve_ids": {"strategy": ids_strategy},
                    "resolve_location": {"strategy": loc_strategy},
                    "background": {"api_lookup": bg_api_lookup, "hmdb_lookup": bg_hmdb_lookup, "max_results": bg_max_results},
                    "rationale": _canonical(str(op.get("rationale", ""))),
                }
            )
        if not out_ops:
            out_ops = _default_stage3_plan(issues)
            return {"source": "llm_empty_fallback_default", "operations": out_ops, "raw": raw[:500]}
        # Ensure one op per issue by filling missing with defaults.
        existing = {str(op.get("issue_key", "")) for op in out_ops}
        for fallback in _default_stage3_plan(issues):
            if str(fallback.get("issue_key", "")) not in existing:
                out_ops.append(fallback)
        return {"source": "llm", "operations": out_ops, "raw": raw[:500]}
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "llm_error_fallback_default",
            "error": str(exc),
            "operations": _default_stage3_plan(issues),
            "raw": raw[:500],
        }


def _collect_id_candidates(kind: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if str(result.get("status", "")).strip().lower() != "mapped":
            continue
        mapped_ids = _safe_dict(result.get("mapped_ids"))
        if not mapped_ids:
            continue
        conf = float(result.get("confidence", 0.0) or 0.0)
        source = str(result.get("source", "")).strip() or "unknown"
        provider = str(result.get("provider", "")).strip() or "unknown"
        candidates.append(
            {
                "source": source,
                "provider": provider,
                "confidence": conf,
                "mapped_ids": mapped_ids,
                "chosen_rule": str(result.get("chosen_rule", "")).strip(),
            }
        )
    # deterministic ordering: confidence desc, prefer db ties
    candidates.sort(key=lambda c: (float(c.get("confidence", 0.0)), 1 if str(c.get("source", "")) == "db" else 0), reverse=True)
    return candidates


def _llm_choose_id_candidate(
    *,
    issue: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    background_context: Optional[Dict[str, Any]],
    use_llm: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if not candidates:
        return {"selected_index": -1, "confidence": 0.0, "reason": "no_candidates", "source": "none"}
    if len(candidates) == 1 or not use_llm:
        return {"selected_index": 0, "confidence": float(candidates[0].get("confidence", 0.0)), "reason": "deterministic_top", "source": "deterministic"}

    prompt_obj = {
        "task": "Select the best ID-mapping candidate.",
        "issue": issue,
        "candidates": candidates,
        "background_context": _safe_dict(background_context),
        "rules": [
            "Choose exactly one index from candidates by mapped ID quality and confidence.",
            "Prefer higher confidence and richer mapped_ids coverage.",
            "Use background_context only as supporting evidence; do not invent new IDs.",
            "If top candidates are close, prefer database-backed evidence over weaker API-only candidates.",
            "Do not reject all candidates when at least one candidate has confidence >= 0.55.",
            "Return JSON only with keys: selected_index, confidence, reason.",
        ],
    }
    try:
        raw = chat(
            [
                {"role": "system", "content": "You are a strict candidate selector. Output JSON only."},
                {"role": "user", "content": json.dumps(prompt_obj, ensure_ascii=False)},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_json=True,
        )
        parsed = _extract_json_object(raw) or {}
        selected_index = int(parsed.get("selected_index", -1))
        if 0 <= selected_index < len(candidates):
            return {
                "selected_index": selected_index,
                "confidence": float(parsed.get("confidence", candidates[selected_index].get("confidence", 0.0)) or 0.0),
                "reason": _canonical(str(parsed.get("reason", ""))) or "llm_selected",
                "source": "llm",
                "raw": raw[:400],
            }
        return {"selected_index": 0, "confidence": float(candidates[0].get("confidence", 0.0)), "reason": "llm_invalid_index_fallback", "source": "llm_fallback"}
    except Exception as exc:  # noqa: BLE001
        return {"selected_index": 0, "confidence": float(candidates[0].get("confidence", 0.0)), "reason": f"llm_error_fallback:{exc}", "source": "deterministic_fallback"}


def _id_candidates_from_hmdb_background(background: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    rows = _safe_list(background.get("candidates"))
    for row in rows:
        if not isinstance(row, dict):
            continue
        hid = _canonical(str(row.get("hmdb_id", ""))).upper()
        if not hid or not hid.startswith("HMDB"):
            continue
        if hid in seen:
            continue
        seen.add(hid)
        score = float(row.get("score", 0.0) or 0.0)
        confidence = max(0.45, min(0.88, score))
        out.append(
            {
                "source": "api",
                "provider": "HMDB",
                "confidence": confidence,
                "mapped_ids": {"hmdb": hid},
                "chosen_rule": "stage3_hmdb_background_candidate",
            }
        )
    return out


def _id_candidates_from_api_background(kind: str, background: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    rows = _safe_list(background.get("candidates"))
    if kind == "protein":
        for row in rows:
            if not isinstance(row, dict):
                continue
            accession = _canonical(str(row.get("accession", "")))
            if not accession:
                continue
            key = f"uniprot:{accession}"
            if key in seen:
                continue
            seen.add(key)
            raw_score = float(row.get("score", 0.0) or 0.0)
            confidence = max(0.42, min(0.9, raw_score))
            out.append(
                {
                    "source": "api",
                    "provider": "UniProt",
                    "confidence": confidence,
                    "mapped_ids": {"uniprot": accession},
                    "chosen_rule": "stage3_api_background_candidate",
                }
            )
    else:
        for row in rows:
            if not isinstance(row, dict):
                continue
            db = _canonical(str(row.get("database", ""))).lower()
            cid = _canonical(str(row.get("id", "")))
            if not db or not cid:
                continue
            key = f"{db}:{cid}"
            if key in seen:
                continue
            seen.add(key)
            raw_score = float(row.get("score", 0.0) or 0.0)
            confidence = max(0.42, min(0.9, raw_score))
            out.append(
                {
                    "source": "api",
                    "provider": "CompoundAPI",
                    "confidence": confidence,
                    "mapped_ids": {db: cid},
                    "chosen_rule": "stage3_api_background_candidate",
                }
            )
    return out


def _id_candidates_from_attempt_candidates(kind: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        source = _canonical(str(result.get("source", ""))) or "api"
        provider = _canonical(str(result.get("provider", ""))) or ("UniProt" if kind == "protein" else "CompoundAPI")
        rows = _safe_list(result.get("candidates"))
        for row in rows:
            if not isinstance(row, dict):
                continue
            mapped_ids: Dict[str, str] = {}
            if kind == "protein":
                accession = _canonical(str(row.get("accession", "")))
                if accession:
                    mapped_ids = {"uniprot": accession}
            else:
                db = _canonical(str(row.get("database", ""))).lower()
                cid = _canonical(str(row.get("id", "")))
                if db and cid:
                    mapped_ids = {db: cid}
            if not mapped_ids:
                continue
            norm_key = tuple(sorted((k, v) for k, v in mapped_ids.items()))
            if norm_key in seen:
                continue
            seen.add(norm_key)
            raw_score = float(row.get("score", result.get("confidence", 0.0)) or 0.0)
            if raw_score < 0.5:
                continue
            confidence = max(0.4, min(0.88, raw_score * 0.95))
            out.append(
                {
                    "source": source,
                    "provider": provider,
                    "confidence": confidence,
                    "mapped_ids": mapped_ids,
                    "chosen_rule": "stage3_attempt_candidate_promotion",
                }
            )
    return out


def _run_id_strategy(
    *,
    kind: str,
    name: str,
    organism: str,
    strategy: str,
    db: Optional[PathBankDbResolver],
    client: HttpClient,
) -> Dict[str, Any]:
    strategy_norm = _canonical(strategy).lower()
    if strategy_norm not in {"db_then_api", "api_then_db", "db_only", "api_only", "skip"}:
        strategy_norm = "db_then_api"

    ordered_sources: List[str] = []
    if strategy_norm == "skip":
        ordered_sources = []
    elif strategy_norm == "db_only":
        ordered_sources = ["db"]
    elif strategy_norm == "api_only":
        ordered_sources = ["api"]
    elif strategy_norm == "api_then_db":
        ordered_sources = ["api", "db"]
    else:
        ordered_sources = ["db", "api"]

    attempts: List[Dict[str, Any]] = []
    for source in ordered_sources:
        if source == "db":
            if kind == "protein":
                result = db.map_protein(name, organism) if db and db.available() else {"status": "unmapped", "reason": "db_unavailable", "source": "db", "provider": "PathBankDB"}
            else:
                result = db.map_compound(name) if db and db.available() else {"status": "unmapped", "reason": "db_unavailable", "source": "db", "provider": "PathBankDB"}
        else:
            if kind == "protein":
                result = map_protein_uniprot(client, name, organism)
                result.setdefault("provider", "UniProt")
                result.setdefault("source", "api")
            else:
                result = map_compound_all(client, name)
                result.setdefault("provider", "ChEBI/KEGG/HMDB")
                result.setdefault("source", "api")
        attempts.append(result)
        if str(result.get("status", "")).strip().lower() == "mapped":
            # keep gathering in case planner wants comparison; do not break
            continue
    return {"strategy": strategy_norm, "attempts": attempts}


def _map_ids_for_entity(
    *,
    kind: str,
    name: str,
    organism: str,
    id_source: str,
    db: Optional[PathBankDbResolver],
    client: HttpClient,
) -> Dict[str, Any]:
    mode = (id_source or "hybrid").strip().lower()
    if mode not in {"api", "db", "hybrid"}:
        mode = "hybrid"

    if kind == "protein":
        if mode in {"db", "hybrid"} and db and db.available():
            db_result = db.map_protein(name, organism)
            if db_result.get("status") == "mapped" or mode == "db":
                return db_result
        if mode in {"api", "hybrid"}:
            api_result = map_protein_uniprot(client, name, organism)
            api_result.setdefault("provider", "UniProt")
            api_result.setdefault("source", "api")
            return api_result
    else:
        if mode in {"db", "hybrid"} and db and db.available():
            db_result = db.map_compound(name)
            if db_result.get("status") == "mapped" or mode == "db":
                return db_result
        if mode in {"api", "hybrid"}:
            api_result = map_compound_all(client, name)
            api_result.setdefault("provider", "ChEBI/KEGG/HMDB")
            api_result.setdefault("source", "api")
            return api_result
    return {"status": "unmapped", "reason": "no_strategy"}


# ---------------------------------------------------------------------------
# Step 10 — Agentic enrichment helpers
# ---------------------------------------------------------------------------

_ENRICHMENT_SYSTEM_PROMPT: Optional[str] = None


def _get_enrichment_system_prompt() -> str:
    global _ENRICHMENT_SYSTEM_PROMPT  # noqa: PLW0603
    if _ENRICHMENT_SYSTEM_PROMPT is None:
        prompt_path = PROMPTS_DIR / "enrichment_system.txt"
        _ENRICHMENT_SYSTEM_PROMPT = (
            prompt_path.read_text(encoding="utf-8")
            if prompt_path.exists()
            else "You are an enrichment agent. Return JSON patches only."
        )
    return _ENRICHMENT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Agentic enrichment — tool schemas and prompt loader
# ---------------------------------------------------------------------------

def _function_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: List[str],
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


_ID_OBJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Known identifiers supplied by the payload. Do not invent IDs.",
    "additionalProperties": {"type": "string"},
}


_ENRICHMENT_TOOLS: List[Dict[str, Any]] = [
    _function_tool(
        "lookup_species",
        "Return PathBank species candidates by name and/or taxonomy ID. This lookup returns candidates only.",
        {
            "name": {"type": "string", "description": "Species or organism name", "default": ""},
            "taxonomy_id": {"type": "string", "description": "NCBI taxonomy ID if present", "default": ""},
        },
        [],
    ),
    _function_tool(
        "lookup_subcellular_location",
        "Return PathBank subcellular location candidates by name. This lookup returns candidates only.",
        {"name": {"type": "string", "description": "Subcellular location name or synonym"}},
        ["name"],
    ),
    _function_tool(
        "lookup_biological_state",
        "Return PathBank biological state candidates for species + subcellular location, optionally narrowed by cell type/tissue.",
        {
            "species": {"type": "string", "description": "Species name"},
            "subcellular_location": {"type": "string", "description": "Subcellular location name"},
            "cell_type": {"type": "string", "description": "Optional cell type", "default": ""},
            "tissue": {"type": "string", "description": "Optional tissue", "default": ""},
        },
        ["species", "subcellular_location"],
    ),
    _function_tool(
        "lookup_compound_db",
        "Return PathBank compound candidates from DB IDs first, then name/synonym/fuzzy search. This lookup returns candidates only.",
        {
            "name": {"type": "string", "description": "Compound name or synonym", "default": ""},
            "ids": _ID_OBJECT_SCHEMA,
        },
        [],
    ),
    _function_tool(
        "lookup_protein_db",
        "Return PathBank protein candidates from direct IDs or species-aware name/gene search. Name/gene search requires species.",
        {
            "name": {"type": "string", "description": "Protein name or gene", "default": ""},
            "species": {"type": "string", "description": "Species name required for name/gene search", "default": ""},
            "ids": _ID_OBJECT_SCHEMA,
        },
        [],
    ),
    _function_tool(
        "lookup_protein_complex_db",
        "Return PathBank protein complex candidates by direct complex ID or complex name + species. This lookup returns candidates only.",
        {
            "name": {"type": "string", "description": "Protein complex name", "default": ""},
            "species": {"type": "string", "description": "Species name required for name search", "default": ""},
            "ids": _ID_OBJECT_SCHEMA,
        },
        [],
    ),
    _function_tool(
        "lookup_complex_by_component",
        "Return PathBank protein complex candidates containing a named protein component, filtered by species.",
        {
            "component_name": {"type": "string", "description": "Protein component name or gene"},
            "species": {"type": "string", "description": "Species name"},
        },
        ["component_name", "species"],
    ),
    _function_tool(
        "create_novel_compound",
        "Create a structured novel compound record for patching when DB lookup candidates are absent or unsafe. Does not create DB IDs.",
        {
            "name": {"type": "string", "description": "Compound name"},
            "compound_class": {"type": "string", "description": "Compound class/type if known", "default": ""},
            "reason": {"type": "string", "description": "Why this should be treated as novel", "default": ""},
        },
        ["name"],
    ),
    _function_tool(
        "create_novel_protein",
        "Create a structured novel protein record for patching when DB lookup candidates are absent or unsafe. Does not create DB IDs.",
        {
            "name": {"type": "string", "description": "Protein name"},
            "species": {"type": "string", "description": "Species name", "default": ""},
            "gene_name": {"type": "string", "description": "Gene symbol if known", "default": ""},
            "reason": {"type": "string", "description": "Why this should be treated as novel", "default": ""},
        },
        ["name"],
    ),
    _function_tool(
        "create_novel_complex",
        "Create a structured novel protein complex record around supplied components. Does not create DB IDs.",
        {
            "name": {"type": "string", "description": "Protein complex name"},
            "species": {"type": "string", "description": "Species name", "default": ""},
            "components": {
                "type": "array",
                "description": "Component records. Use DB IDs only if they came from lookup tools.",
                "items": {"type": "object"},
            },
            "reason": {"type": "string", "description": "Why this should be treated as novel", "default": ""},
        },
        ["name"],
    ),
    _function_tool(
        "propose_patch",
        "Commit a JSON patch operation to the pathway payload. Use this to write decisions based on tool results.",
        {
            "op": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "JSON patch operation type",
            },
            "path": {
                "type": "string",
                "description": "JSON Pointer path e.g. /entities/protein_complexes/0/species_id",
            },
            "value": {
                "description": "New value to write. Required for add/replace.",
            },
            "evidence": {
                "type": "string",
                "description": "One sentence explaining which tool result supports this patch",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0",
            },
        },
        ["op", "path", "evidence", "confidence"],
    ),
    _function_tool(
        "lookup_compound",
        "Compatibility alias for lookup_compound_db. Returns candidates only.",
        {
            "name": {"type": "string", "description": "Compound name to look up"},
            "ids": _ID_OBJECT_SCHEMA,
        },
        ["name"],
    ),
    _function_tool(
        "lookup_protein",
        "Compatibility alias for lookup_protein_db. Returns candidates only.",
        {
            "name": {"type": "string", "description": "Protein name to look up"},
            "organism": {"type": "string", "description": "Organism/species name", "default": ""},
            "species": {"type": "string", "description": "Organism/species name", "default": ""},
            "ids": _ID_OBJECT_SCHEMA,
        },
        ["name"],
    ),
    _function_tool(
        "lookup_compartment_candidates",
        "Compatibility alias for lookup_subcellular_location/location evidence. Returns candidates only.",
        {
            "entity_name": {"type": "string", "description": "Name of the entity"},
            "entity_type": {
                "type": "string",
                "enum": ["compound", "protein", "protein_complex"],
                "description": "Entity type",
            },
        },
        ["entity_name", "entity_type"],
    ),
]

_ENRICHMENT_AGENTIC_SYSTEM_PROMPT: Optional[str] = None


def _get_enrichment_agentic_system_prompt() -> str:
    global _ENRICHMENT_AGENTIC_SYSTEM_PROMPT  # noqa: PLW0603
    if _ENRICHMENT_AGENTIC_SYSTEM_PROMPT is None:
        prompt_path = PROMPTS_DIR / "enrichment_agentic_system.txt"
        _ENRICHMENT_AGENTIC_SYSTEM_PROMPT = (
            prompt_path.read_text(encoding="utf-8")
            if prompt_path.exists()
            else "You are an enrichment agent. Use tools to look up IDs and propose patches."
        )
    return _ENRICHMENT_AGENTIC_SYSTEM_PROMPT


def _candidate_lookup_response(result: Dict[str, Any], *, max_candidates: int = 10) -> Dict[str, Any]:
    candidates = [dict(c) for c in _safe_list(result.get("candidates")) if isinstance(c, dict)][:max_candidates]
    if candidates and _safe_list(result.get("components")):
        candidates[0].setdefault("components", _safe_list(result.get("components")))
    if candidates and _safe_list(result.get("issues")):
        candidates[0].setdefault("issues", _safe_list(result.get("issues")))
    return {
        "status": str(result.get("status") or "unmapped"),
        "reason": str(result.get("reason") or ""),
        "source": str(result.get("source") or "pathbank_db"),
        "provider": str(result.get("provider") or "PathBankDB"),
        "chosen_rule": str(result.get("chosen_rule") or ""),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "candidates": candidates,
    }


def _db_tool_unavailable(db: Optional[PathBankDbResolver]) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": "db_unavailable",
        "source": "pathbank_db",
        "provider": "PathBankDB",
        "chosen_rule": "",
        "confidence": 0.0,
        "candidates": [],
        "last_error": getattr(db, "last_error", "") if db is not None else "db_not_configured",
    }


def _tool_ids(tool_args: Dict[str, Any]) -> Dict[str, str]:
    alias_map = {
        "pathwhiz_id": "pathwhiz_id",
        "pathbank_compound_id": "pathbank_compound_id",
        "pathbank_protein_id": "pathbank_protein_id",
        "pathbank_complex_id": "pathbank_protein_complex_id",
        "pathbank_protein_complex_id": "pathbank_protein_complex_id",
        "pw_compound_id": "pathbank_compound_id",
        "pw_protein_id": "pathbank_protein_id",
        "pw_complex_id": "pathbank_protein_complex_id",
        "pwc_id": "pwc_id",
        "hmdb_id": "hmdb",
        "hmdb": "hmdb",
        "kegg_id": "kegg",
        "kegg": "kegg",
        "chebi_id": "chebi",
        "chebi": "chebi",
        "pubchem_cid": "pubchem",
        "pubchem_id": "pubchem",
        "pubchem": "pubchem",
        "cas_id": "cas",
        "cas_number": "cas",
        "cas": "cas",
        "drugbank_id": "drugbank",
        "drugbank": "drugbank",
        "biocyc_id": "biocyc",
        "biocyc": "biocyc",
        "chemspider_id": "chemspider",
        "chemspider": "chemspider",
        "uniprot_id": "uniprot",
        "uniprot": "uniprot",
        "gene": "gene_name",
        "gene_name": "gene_name",
    }
    ids = dict(_safe_dict(tool_args.get("ids")))
    for key in alias_map:
        if key in tool_args and key not in ids:
            ids[key] = tool_args[key]
    out: Dict[str, str] = {}
    for key, value in ids.items():
        sval = _canonical(str(value or ""))
        if sval:
            out[alias_map.get(str(key), str(key))] = sval
    return out


def _first_lookup_with_candidates(*results: Dict[str, Any]) -> Dict[str, Any]:
    fallback: Dict[str, Any] = {
        "status": "unmapped",
        "reason": "no_lookup_attempted",
        "provider": "PathBankDB",
        "source": "db",
        "chosen_rule": "",
        "confidence": 0.0,
        "candidates": [],
    }
    for result in results:
        if not isinstance(result, dict):
            continue
        fallback = result
        if _safe_list(result.get("candidates")):
            return result
        if result.get("status") == "mapped":
            return result
    return fallback


def _lookup_compound_db_candidates(db: PathBankDbResolver, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    ids = _tool_ids(tool_args)
    name = _canonical(str(tool_args.get("name") or ""))
    attempts: List[Dict[str, Any]] = []

    pathbank_id = ids.get("pathbank_compound_id") or ids.get("pathwhiz_id")
    if pathbank_id and hasattr(db, "_map_compound_by_pathbank_id"):
        attempts.append(db._map_compound_by_pathbank_id(pathbank_id))  # noqa: SLF001
    if ids.get("pwc_id") and hasattr(db, "_map_compound_by_pwc_id"):
        attempts.append(db._map_compound_by_pwc_id(ids["pwc_id"]))  # noqa: SLF001

    external_ids = {
        key: value
        for key, value in ids.items()
        if key in {"hmdb", "kegg", "chebi", "pubchem", "cas", "drugbank", "biocyc", "chemspider"}
    }
    if external_ids:
        attempts.append(db.map_compound_by_ids(external_ids))
    if name:
        attempts.append(db.map_compound_by_name(name))

    return _first_lookup_with_candidates(*attempts)


def _lookup_protein_db_candidates(db: PathBankDbResolver, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    ids = _tool_ids(tool_args)
    name = _canonical(str(tool_args.get("name") or ""))
    species = _canonical(str(tool_args.get("species") or tool_args.get("organism") or ""))
    attempts: List[Dict[str, Any]] = []

    pathbank_id = ids.get("pathbank_protein_id") or ids.get("pathwhiz_id")
    if pathbank_id and hasattr(db, "_map_protein_by_pathbank_id"):
        attempts.append(db._map_protein_by_pathbank_id(pathbank_id))  # noqa: SLF001

    protein_ids = {key: value for key, value in ids.items() if key in {"uniprot", "gene_name"}}
    if protein_ids:
        attempts.append(db.map_protein_by_ids(protein_ids, species=species or None))
    if name:
        if species:
            attempts.append(db.map_protein_by_name_species(name, species))
        else:
            attempts.append(
                {
                    "status": "unmapped",
                    "reason": "needs_species",
                    "provider": "PathBankDB",
                    "source": "db",
                    "chosen_rule": "",
                    "confidence": 0.0,
                    "candidates": [],
                }
            )

    return _first_lookup_with_candidates(*attempts)


def _lookup_protein_complex_db_candidates(db: PathBankDbResolver, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    ids = _tool_ids(tool_args)
    name = _canonical(str(tool_args.get("name") or ""))
    species = _canonical(str(tool_args.get("species") or tool_args.get("organism") or ""))
    attempts: List[Dict[str, Any]] = []

    pathbank_id = ids.get("pathbank_protein_complex_id") or ids.get("pathwhiz_id")
    if pathbank_id and hasattr(db, "_map_complex_by_pathbank_id"):
        attempts.append(db._map_complex_by_pathbank_id(pathbank_id))  # noqa: SLF001
    if name:
        if species:
            attempts.append(db.map_protein_complex(name, species))
        else:
            attempts.append(
                {
                    "status": "unmapped",
                    "reason": "needs_species",
                    "provider": "PathBankDB",
                    "source": "db",
                    "chosen_rule": "",
                    "confidence": 0.0,
                    "candidates": [],
                }
            )

    return _first_lookup_with_candidates(*attempts)


def _novel_compound_record(tool_args: Dict[str, Any]) -> Dict[str, Any]:
    name = _canonical(str(tool_args.get("name") or ""))
    record: Dict[str, Any] = {
        "name": name,
        "mapping_meta": {
            "provider": "novel",
            "source": "llm_tool",
            "resolution": {"status": "novel", "issue": "no_safe_db_candidate"},
        },
    }
    compound_class = _canonical(str(tool_args.get("compound_class") or tool_args.get("class") or tool_args.get("type") or ""))
    if compound_class:
        record["class"] = compound_class
    reason = _canonical(str(tool_args.get("reason") or ""))
    if reason:
        record["mapping_meta"]["novel_reason"] = reason
    return {"status": "novel", "record": record}


def _novel_protein_record(tool_args: Dict[str, Any]) -> Dict[str, Any]:
    name = _canonical(str(tool_args.get("name") or ""))
    species = _canonical(str(tool_args.get("species") or tool_args.get("organism") or ""))
    gene_name = _canonical(str(tool_args.get("gene_name") or tool_args.get("gene") or ""))
    record: Dict[str, Any] = {
        "name": name,
        "mapping_meta": {
            "provider": "novel",
            "source": "llm_tool",
            "resolution": {"status": "novel", "issue": "no_safe_db_candidate"},
        },
    }
    if species:
        record["species"] = species
    if gene_name:
        record["gene_name"] = gene_name
    reason = _canonical(str(tool_args.get("reason") or ""))
    if reason:
        record["mapping_meta"]["novel_reason"] = reason
    return {"status": "novel", "record": record}


def _novel_complex_record(tool_args: Dict[str, Any]) -> Dict[str, Any]:
    name = _canonical(str(tool_args.get("name") or ""))
    species = _canonical(str(tool_args.get("species") or tool_args.get("organism") or ""))
    components: List[Dict[str, Any]] = []
    for raw in _safe_list(tool_args.get("components")):
        if not isinstance(raw, dict):
            continue
        component = dict(raw)
        component.pop("pathbank_complex_id", None)
        component.pop("pathbank_protein_complex_id", None)
        if not _component_has_stoichiometry(component):
            component["stoichiometry"] = 1
        components.append(component)
    record: Dict[str, Any] = {
        "name": name,
        "components": components,
        "mapping_meta": {
            "provider": "novel",
            "source": "llm_tool",
            "resolution": {"status": "novel", "issue": "no_safe_db_candidate"},
        },
    }
    if species:
        record["species"] = species
    reason = _canonical(str(tool_args.get("reason") or ""))
    if reason:
        record["mapping_meta"]["novel_reason"] = reason
    return {"status": "novel", "record": record}


def _build_entity_index(payload: Dict[str, Any]) -> Dict[str, Tuple[str, int]]:
    """Return {normalized_name: (array_path_prefix, index)} for path construction.

    E.g. "glucose 6 phosphate" -> ("/entities/compounds", 3)
    so the entity's JSON pointer is /entities/compounds/3
    """
    out: Dict[str, Tuple[str, int]] = {}
    entities = _safe_dict(payload.get("entities"))
    for list_key, path_prefix in [
        ("species", "/entities/species"),
        ("compounds", "/entities/compounds"),
        ("proteins", "/entities/proteins"),
        ("protein_complexes", "/entities/protein_complexes"),
        ("nucleic_acids", "/entities/nucleic_acids"),
    ]:
        for idx, item in enumerate(_safe_list(entities.get(list_key))):
            if not isinstance(item, dict):
                continue
            name = _canonical(str(item.get("name", "")))
            if name:
                out[_normalize(name)] = (path_prefix, idx)
    for idx, item in enumerate(_safe_list(payload.get("biological_states"))):
        if not isinstance(item, dict):
            continue
        name = _canonical(str(item.get("name", "")))
        if name:
            out[_normalize(name)] = ("/biological_states", idx)
    return out


def _pre_fetch_for_flag(
    flag_type: str,
    flag_entry: Dict[str, Any],
    *,
    db: Optional[PathBankDbResolver],
    client: HttpClient,
    global_organism: str,
    max_candidates: int = 3,
) -> Dict[str, Any]:
    """Pre-fetch API/DB candidates for a single QA flag entry.

    Returns a dict with pre-fetched data to include in the enrichment LLM context.
    No LLM call is made here — this is pure deterministic API/DB retrieval.
    """
    entity_name = _canonical(str(flag_entry.get("entity", flag_entry.get("reaction", ""))))
    entity_type = _canonical(str(flag_entry.get("type", ""))).lower()

    result: Dict[str, Any] = {
        "flag_type": flag_type,
        "entity": entity_name,
        "entity_type": entity_type,
        "candidates": [],
    }

    if flag_type == "missing_ids":
        if entity_type in {"protein", "protein_complex"}:
            api_result = map_protein_uniprot(client, entity_name, global_organism)
            if api_result.get("status") == "mapped":
                result["candidates"].append(
                    {
                        "source": "UniProt",
                        "mapped_ids": _safe_dict(api_result.get("mapped_ids")),
                        "confidence": float(api_result.get("confidence", 0.8) or 0.8),
                    }
                )
            bg = lookup_protein_api_background(client, entity_name, global_organism, max_results=max_candidates)
            for cand in _safe_list(_safe_dict(bg).get("candidates", []))[:max_candidates]:
                if not isinstance(cand, dict):
                    continue
                acc = _canonical(str(cand.get("accession", "")))
                if acc:
                    result["candidates"].append(
                        {
                            "source": "UniProt",
                            "mapped_ids": {"uniprot": acc},
                            "name": _canonical(str(cand.get("name", ""))),
                            "confidence": float(cand.get("score", 0.7) or 0.7),
                        }
                    )
        else:
            # compound (default)
            api_result = map_compound_all(client, entity_name)
            if api_result.get("status") == "mapped":
                result["candidates"].append(
                    {
                        "source": "ChEBI/KEGG/HMDB",
                        "mapped_ids": _safe_dict(api_result.get("mapped_ids")),
                        "confidence": float(api_result.get("confidence", 0.8) or 0.8),
                    }
                )
            bg = lookup_compound_api_background(client, entity_name, max_results=max_candidates)
            for cand in _safe_list(_safe_dict(bg).get("candidates", []))[:max_candidates]:
                if not isinstance(cand, dict):
                    continue
                db_key = _canonical(str(cand.get("database", ""))).lower()
                cid = _canonical(str(cand.get("id", "")))
                if db_key and cid:
                    result["candidates"].append(
                        {
                            "source": db_key,
                            "mapped_ids": {db_key: cid},
                            "name": _canonical(str(cand.get("name", ""))),
                            "confidence": float(cand.get("score", 0.7) or 0.7),
                        }
                    )
            hmdb_bg = lookup_hmdb_background(client, entity_name, max_results=max_candidates)
            for cand in _safe_list(_safe_dict(hmdb_bg).get("candidates", []))[:max_candidates]:
                if not isinstance(cand, dict):
                    continue
                hid = _canonical(str(cand.get("hmdb_id", ""))).upper()
                if hid and hid.startswith("HMDB"):
                    result["candidates"].append(
                        {
                            "source": "HMDB",
                            "mapped_ids": {"hmdb": hid},
                            "name": _canonical(str(cand.get("name", ""))),
                            "confidence": float(cand.get("score", 0.7) or 0.7),
                        }
                    )

    elif flag_type == "missing_compartments":
        kind = entity_type if entity_type in {"compound", "protein"} else "compound"
        result["location_candidates"] = _db_location_candidates(db, kind=kind, name=entity_name, max_items=max_candidates)

    elif flag_type == "possible_complexes":
        bg = lookup_protein_api_background(client, entity_name, global_organism, max_results=max_candidates)
        result["uniprot_candidates"] = _safe_list(_safe_dict(bg).get("candidates", []))[:max_candidates]

    elif flag_type == "transport_like_reactions":
        result["note"] = "Reaction name or structure implies transport; consider whether a transport process is more appropriate."

    return result


def _format_enrichment_context(
    pre_fetched_items: List[Dict[str, Any]],
    entity_index: Dict[str, Tuple[str, int]],
) -> str:
    """Render pre-fetched candidates as a structured text block for the enrichment LLM."""
    lines: List[str] = []

    for entry in pre_fetched_items:
        flag_type = entry.get("flag_type", "")
        entity_name = entry.get("entity", "")
        entity_type = entry.get("entity_type", "")

        norm_name = _normalize(entity_name)
        path_info = entity_index.get(norm_name)
        json_path = f"{path_info[0]}/{path_info[1]}" if path_info else None

        lines.append("\n---")
        lines.append(f"ENTITY: {entity_name}")
        lines.append(f"TYPE: {entity_type or 'unknown'}")
        lines.append(f"FLAG: {flag_type}")
        if json_path:
            lines.append(f"JSON_PATH: {json_path}")

        if flag_type == "missing_ids":
            candidates = entry.get("candidates", [])
            if candidates:
                lines.append("API CANDIDATES:")
                for cand in candidates[:3]:
                    mids = cand.get("mapped_ids", {})
                    ids_str = " | ".join(f"{k}={v}" for k, v in mids.items() if v)
                    cand_name = cand.get("name", "")
                    conf = float(cand.get("confidence", 0.0))
                    name_part = f" | name: {cand_name}" if cand_name else ""
                    lines.append(f"  - {ids_str}{name_part} | confidence={conf:.2f}")
            else:
                lines.append("API CANDIDATES: none found")
            lines.append(
                "INSTRUCTION: Assign the best matching ID(s). "
                "If confidence < 0.60, emit a warning instead of a patch."
            )

        elif flag_type == "missing_compartments":
            loc_candidates = entry.get("location_candidates", [])
            if loc_candidates:
                lines.append("LOCATION CANDIDATES (from PathBank DB):")
                for lc in loc_candidates[:3]:
                    lines.append(
                        f"  - {lc.get('location', '')} | source={lc.get('source', '')} | score={lc.get('score', 0)}"
                    )
            else:
                lines.append("LOCATION CANDIDATES: none from DB — infer from entity class.")
            lines.append(
                "INSTRUCTION: Assign the most supported compartment. "
                "If uncertain, set provenance=inferred and confidence=0.70."
            )

        elif flag_type == "possible_complexes":
            uniprot_cands = entry.get("uniprot_candidates", [])
            if uniprot_cands:
                lines.append("UNIPROT CANDIDATES:")
                for uc in uniprot_cands[:3]:
                    acc = uc.get("accession", "")
                    cand_name = uc.get("name", "")
                    lines.append(f"  - UniProt:{acc} | {cand_name}")
            else:
                lines.append("UNIPROT CANDIDATES: none found")
            lines.append(
                "INSTRUCTION: If this is a known complex subunit, add a warning. "
                "Do not rename or reclassify the entity."
            )

        elif flag_type == "transport_like_reactions":
            lines.append(f"NOTE: {entry.get('note', '')}")
            lines.append(
                "INSTRUCTION: Do not modify the reaction. "
                "Add a warning entry if transport reclassification should be reviewed."
            )

    return "\n".join(lines)


def _run_enrichment_agent(
    payload: Dict[str, Any],
    qa_report: Dict[str, Any],
    *,
    db: Optional[PathBankDbResolver],
    client: HttpClient,
    global_organism: str,
    llm_temperature: float,
    llm_max_tokens: int,
    max_flags_per_type: int = 20,
    reaction_summary: Optional[str] = None,
    stage3_issues: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Agentic enrichment: LLM uses tools to look up IDs/compartments and propose patches.

    Returns (normalized_patches, enrichment_report).
    The patches use the same format as apply_audit_patch (op/evidence) after normalization.
    """
    flags = _safe_dict(qa_report.get("flags"))
    entity_index = _build_entity_index(payload)

    actionable = ["missing_ids", "missing_compartments", "possible_complexes", "transport_like_reactions"]
    flags_for_agent: Dict[str, Any] = {}
    total_flag_entries = 0
    for flag_type in actionable:
        entries = [e for e in _safe_list(flags.get(flag_type, [])) if isinstance(e, dict)][:max_flags_per_type]
        if entries:
            flags_for_agent[flag_type] = entries
            total_flag_entries += len(entries)
    stage3_for_agent = [
        issue for issue in _safe_list(stage3_issues)
        if isinstance(issue, dict) and _safe_list(issue.get("reasons"))
    ][: max(1, max_flags_per_type * 2)]

    enrichment_report: Dict[str, Any] = {
        "flags_processed": total_flag_entries,
        "stage3_issues_processed": len(stage3_for_agent),
    }

    if total_flag_entries == 0 and not stage3_for_agent:
        enrichment_report["patches_proposed"] = 0
        return [], enrichment_report

    accumulated_patches: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []

    def tool_executor(tool_name: str, tool_args: Dict[str, Any]) -> Any:
        result: Dict[str, Any]
        if tool_name in {"lookup_species"}:
            if db is None or not db.available():
                result = _db_tool_unavailable(db)
            else:
                result = _candidate_lookup_response(
                    db.find_species(
                        _canonical(str(tool_args.get("name") or "")),
                        taxonomy_id=_canonical(str(tool_args.get("taxonomy_id") or "")) or None,
                    )
                )

        elif tool_name in {"lookup_subcellular_location"}:
            if db is None or not db.available():
                result = _db_tool_unavailable(db)
            else:
                result = _candidate_lookup_response(db.find_subcellular_location(_canonical(str(tool_args.get("name") or ""))))

        elif tool_name in {"lookup_biological_state"}:
            if db is None or not db.available():
                result = _db_tool_unavailable(db)
            else:
                result = _candidate_lookup_response(
                    db.find_biological_state(
                        _canonical(str(tool_args.get("species") or "")),
                        _canonical(str(tool_args.get("subcellular_location") or "")),
                        cell_type=_canonical(str(tool_args.get("cell_type") or "")) or None,
                        tissue=_canonical(str(tool_args.get("tissue") or "")) or None,
                    )
                )

        elif tool_name in {"lookup_compound_db", "lookup_compound"}:
            if db is None or not db.available():
                result = _db_tool_unavailable(db)
            else:
                result = _candidate_lookup_response(_lookup_compound_db_candidates(db, tool_args))

        elif tool_name in {"lookup_protein_db", "lookup_protein"}:
            if db is None or not db.available():
                result = _db_tool_unavailable(db)
            else:
                result = _candidate_lookup_response(_lookup_protein_db_candidates(db, tool_args))

        elif tool_name in {"lookup_protein_complex_db"}:
            if db is None or not db.available():
                result = _db_tool_unavailable(db)
            else:
                result = _candidate_lookup_response(_lookup_protein_complex_db_candidates(db, tool_args))

        elif tool_name in {"lookup_complex_by_component"}:
            if db is None or not db.available():
                result = _db_tool_unavailable(db)
            else:
                result = _candidate_lookup_response(
                    db.find_complex_by_component(
                        _canonical(str(tool_args.get("component_name") or "")),
                        _canonical(str(tool_args.get("species") or tool_args.get("organism") or "")),
                    )
                )

        elif tool_name == "lookup_compartment_candidates":
            if db is None or not db.available():
                result = _db_tool_unavailable(db)
            else:
                entity_name = _canonical(str(tool_args.get("entity_name") or ""))
                entity_type = _canonical(str(tool_args.get("entity_type") or "compound")).lower()
                kind = entity_type if entity_type in {"compound", "protein"} else "protein"
                result = {
                    "status": "mapped",
                    "reason": "",
                    "source": "pathbank_db",
                    "provider": "PathBankDB",
                    "chosen_rule": "location_frequency",
                    "confidence": 0.0,
                    "candidates": _db_location_candidates(db, kind=kind, name=entity_name),
                }

        elif tool_name == "create_novel_compound":
            result = _novel_compound_record(tool_args)

        elif tool_name == "create_novel_protein":
            result = _novel_protein_record(tool_args)

        elif tool_name == "create_novel_complex":
            result = _novel_complex_record(tool_args)

        elif tool_name == "propose_patch":
            patch = {
                "op": tool_args.get("op"),
                "path": tool_args.get("path"),
                "value": tool_args.get("value"),
                "evidence": tool_args.get("evidence", ""),
                "confidence": tool_args.get("confidence", 0.7),
            }
            accumulated_patches.append(patch)
            result = {"accepted": True, "patch_index": len(accumulated_patches) - 1}

        else:
            result = {"error": f"unknown tool: {tool_name}"}

        tool_calls.append(
            {
                "tool": tool_name,
                "args": {k: v for k, v in tool_args.items() if k not in {"value", "components"}},
                "status": result.get("status", "ok") if isinstance(result, dict) else "ok",
                "candidate_count": len(_safe_list(result.get("candidates"))) if isinstance(result, dict) else 0,
            }
        )
        return result

    user_content_dict: Dict[str, Any] = {
        "task": "Process these QA flags and Stage 3 gap issues. Use DB lookup tools before patching DB-backed IDs.",
        "global_organism": global_organism,
        "entity_index": {k: list(v) for k, v in entity_index.items()},
        "qa_flags": flags_for_agent,
        "stage3_issues": stage3_for_agent,
        "rules": [
            "Lookup tools return candidates only; choose IDs only from returned candidates.",
            "If candidates are absent or unsafe, use create_novel_* and patch the structured novel record, not a DB ID.",
            "For protein and complex name searches, resolve or provide species first.",
            "For complexes, use lookup_species, lookup_protein_db or lookup_complex_by_component, then propose_patch for species/components.",
        ],
    }
    if reaction_summary and isinstance(reaction_summary, str) and reaction_summary.strip():
        user_content_dict["pathway_reaction_summary"] = reaction_summary.strip()
    user_content = json.dumps(user_content_dict, ensure_ascii=False)

    final_text = ""
    try:
        final_text = chat_with_tools(
            messages=[
                {"role": "system", "content": _get_enrichment_agentic_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            tools=_ENRICHMENT_TOOLS,
            tool_executor=tool_executor,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            max_tool_rounds=15,
        )

        # Also try to parse patches from the final text response
        parsed = _extract_json_object(final_text) or {}
        final_response_patches = _safe_list(parsed.get("patches"))
        enrichment_report["warnings"] = _safe_list(parsed.get("warnings"))
        enrichment_report["raw"] = final_text[:800]

        # Merge: prefer mid-loop tool-call patches, deduplicate by (op, path)
        seen: Dict[Tuple[str, str], bool] = {}
        merged: List[Dict[str, Any]] = []
        for patch in accumulated_patches:
            key = (str(patch.get("op", "")), str(patch.get("path", "")))
            if key not in seen:
                seen[key] = True
                merged.append(patch)
        for patch in final_response_patches:
            if not isinstance(patch, dict):
                continue
            key = (str(patch.get("op", patch.get("action", ""))), str(patch.get("path", "")))
            if key not in seen:
                seen[key] = True
                merged.append(patch)

        enrichment_report["patches_from_tool_calls"] = len(accumulated_patches)
        enrichment_report["patches_from_final_response"] = len(final_response_patches)
        enrichment_report["patches_proposed"] = len(merged)
        enrichment_report["tool_calls"] = tool_calls

        # Normalize patch format → apply_audit_patch format (action → op, reason → evidence)
        normalized: List[Dict[str, Any]] = []
        for patch in merged:
            if not isinstance(patch, dict):
                continue
            norm = dict(patch)
            if "action" in norm and "op" not in norm:
                norm["op"] = norm.pop("action")
            if "reason" in norm and "evidence" not in norm:
                norm["evidence"] = norm.pop("reason")
            normalized.append(norm)

        return normalized, enrichment_report

    except Exception as exc:  # noqa: BLE001
        enrichment_report["error"] = str(exc)
        enrichment_report["raw"] = final_text[:400]
        enrichment_report["patches_proposed"] = 0
        enrichment_report["tool_calls"] = tool_calls
        return [], enrichment_report


def resolve_gaps(
    payload: Dict[str, Any],
    *,
    id_source: str = "hybrid",
    db_config: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
    llm_temperature: float = 0.15,
    llm_max_tokens: int = 900,
    max_items: int = 80,
    enable_id_resolution: bool = True,
    qa_report: Optional[Dict[str, Any]] = None,
    reaction_summary: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    working = deepcopy(payload)
    report: Dict[str, Any] = {
        "summary": {
            "mapped_ids_added": 0,
            "organisms_added": 0,
            "locations_added": 0,
            "location_states_filled": 0,
            "items_considered": 0,
        },
        "actions": [],
        "stage3": {
            "issues": [],
            "planner": {},
            "operations": [],
            "executions": [],
        },
    }
    entities = _safe_dict(working.get("entities"))
    elem_locs = _safe_dict(working.setdefault("element_locations", {}))
    if not isinstance(elem_locs.get("compound_locations"), list):
        elem_locs["compound_locations"] = []
    if not isinstance(elem_locs.get("protein_locations"), list):
        elem_locs["protein_locations"] = []

    global_organism = _extract_global_organism(working)
    client = HttpClient()
    db = PathBankDbResolver.from_env(db_config) if id_source in {"db", "hybrid"} else None

    issues = _collect_stage3_issues(working, max_items=max_items)
    report["summary"]["items_considered"] = len(issues)
    report["stage3"]["issues"] = issues
    plan = _llm_plan_stage3(
        issues,
        use_llm=use_llm,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
    report["stage3"]["planner"] = {
        "source": plan.get("source", ""),
        "error": plan.get("error", ""),
        "raw": plan.get("raw", ""),
    }
    operations = _safe_list(plan.get("operations"))[: max(1, int(max_items))]
    report["stage3"]["operations"] = operations

    issue_by_key = {str(issue.get("issue_key", "")): issue for issue in issues if isinstance(issue, dict)}
    entity_by_key: Dict[str, Dict[str, Any]] = {}
    for kind, rows in [("protein", _safe_list(entities.get("proteins"))), ("compound", _safe_list(entities.get("compounds")))]:
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = _canonical(str(row.get("name", "")))
            if not name:
                continue
            entity_by_key[_issue_key(kind, name)] = row

    fallback_location = "cell"
    for op in operations:
        if not isinstance(op, dict):
            continue
        issue_key = _canonical(str(op.get("issue_key", "")))
        issue = _safe_dict(issue_by_key.get(issue_key))
        item = entity_by_key.get(issue_key)
        if not issue or not isinstance(item, dict):
            report["stage3"]["executions"].append(
                {"issue_key": issue_key, "status": "skipped", "reason": "issue_not_found"}
            )
            continue

        kind = _canonical(str(issue.get("entity_type", "")))
        name = _canonical(str(issue.get("name", "")))
        op_exec: Dict[str, Any] = {"issue_key": issue_key, "entity_type": kind, "name": name, "status": "ok"}

        if kind == "protein" and bool(issue.get("needs_organism")) and global_organism:
            if not _canonical(str(item.get("organism", ""))):
                item["organism"] = global_organism
                report["summary"]["organisms_added"] += 1
                report["actions"].append(
                    {
                        "type": "organism_added",
                        "entity_type": kind,
                        "name": name,
                        "organism": global_organism,
                        "source": "global_species",
                    }
                )
                op_exec["organism_added"] = global_organism

        # Stage 3A: ID query plan -> deterministic execution -> LLM selection
        if enable_id_resolution:
            ids_strategy = _canonical(str(_safe_dict(op.get("resolve_ids")).get("strategy", "db_then_api"))).lower()
            background_cfg = _safe_dict(op.get("background"))
            background_api_lookup = _canonical(str(background_cfg.get("api_lookup", "auto"))).lower()
            if background_api_lookup not in {"auto", "none", "full"}:
                background_api_lookup = "auto"
            background_hmdb_lookup = bool(background_cfg.get("hmdb_lookup", bool(kind == "compound")))
            background_max_results = max(1, min(12, int(background_cfg.get("max_results", 6) or 6)))
            background_context: Dict[str, Any] = {}

            if background_api_lookup in {"auto", "full"}:
                if kind == "compound":
                    background_context["compound_api"] = lookup_compound_api_background(
                        client, name, max_results=background_max_results
                    )
                elif kind == "protein":
                    org_for_bg = _canonical(str(item.get("organism", ""))) or global_organism
                    background_context["protein_api"] = lookup_protein_api_background(
                        client, name, org_for_bg, max_results=background_max_results
                    )
            if kind == "compound" and background_hmdb_lookup:
                background_context["hmdb"] = lookup_hmdb_background(client, name, max_results=background_max_results)
            if background_context:
                op_exec["background"] = background_context

            if bool(issue.get("needs_id_mapping")) and ids_strategy != "skip":
                organism = _canonical(str(item.get("organism", ""))) if kind == "protein" else ""
                id_exec = _run_id_strategy(
                    kind=kind,
                    name=name,
                    organism=organism,
                    strategy=ids_strategy,
                    db=db,
                    client=client,
                )
                id_candidates = _collect_id_candidates(kind, _safe_list(id_exec.get("attempts")))
                id_candidates.extend(_id_candidates_from_attempt_candidates(kind, _safe_list(id_exec.get("attempts"))))
                if kind == "compound":
                    if isinstance(background_context.get("compound_api"), dict):
                        id_candidates.extend(_id_candidates_from_api_background(kind, _safe_dict(background_context.get("compound_api"))))
                    if isinstance(background_context.get("hmdb"), dict):
                        id_candidates.extend(_id_candidates_from_hmdb_background(_safe_dict(background_context.get("hmdb"))))
                elif kind == "protein":
                    if isinstance(background_context.get("protein_api"), dict):
                        id_candidates.extend(_id_candidates_from_api_background(kind, _safe_dict(background_context.get("protein_api"))))

                # de-duplicate by mapped_ids and keep strongest confidence.
                deduped: Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]] = {}
                for cand in id_candidates:
                    if not isinstance(cand, dict):
                        continue
                    mapped_ids = _safe_dict(cand.get("mapped_ids"))
                    if not mapped_ids:
                        continue
                    key = tuple(sorted((str(k), str(v)) for k, v in mapped_ids.items()))
                    existing = deduped.get(key)
                    if not existing or float(cand.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                        deduped[key] = cand
                id_candidates = list(deduped.values())
                id_candidates.sort(
                    key=lambda c: (float(c.get("confidence", 0.0)), 1 if str(c.get("source", "")) == "db" else 0),
                    reverse=True,
                )
                id_choice = _llm_choose_id_candidate(
                    issue=issue,
                    candidates=id_candidates,
                    background_context=background_context,
                    use_llm=use_llm,
                    temperature=llm_temperature,
                    max_tokens=max(500, int(llm_max_tokens)),
                )
                op_exec["id_execution"] = id_exec
                op_exec["id_candidates"] = id_candidates[:8]
                op_exec["id_choice"] = id_choice
                idx = int(id_choice.get("selected_index", -1))
                if 0 <= idx < len(id_candidates):
                    selected = id_candidates[idx]
                    new_ids = _safe_dict(selected.get("mapped_ids"))
                    if new_ids:
                        old_ids = _safe_dict(item.get("mapped_ids"))
                        merged_ids = {**old_ids, **new_ids}
                        if merged_ids != old_ids:
                            item["mapped_ids"] = merged_ids
                            item.setdefault("mapping_meta", {})
                            item["mapping_meta"]["provider"] = selected.get("provider", "")
                            item["mapping_meta"]["source"] = selected.get("source", "")
                            item["mapping_meta"]["confidence"] = float(selected.get("confidence", 0.0))
                            item["mapping_meta"]["chosen_rule"] = selected.get("chosen_rule", "")
                            report["summary"]["mapped_ids_added"] += 1
                            report["actions"].append(
                                {
                                    "type": "mapped_ids_added",
                                    "entity_type": kind,
                                    "name": name,
                                    "mapped_ids": new_ids,
                                    "provider": selected.get("provider", ""),
                                    "source": selected.get("source", ""),
                                    "confidence": float(selected.get("confidence", 0.0)),
                                    "stage": "stage3",
                                }
                            )
        else:
            op_exec["id_resolution"] = {
                "status": "skipped",
                "reason": "disabled_by_configuration",
            }

        # Stage 3B: location query plan -> deterministic execution -> LLM selection
        loc_strategy = _canonical(str(_safe_dict(op.get("resolve_location")).get("strategy", "db_then_default"))).lower()
        if (bool(issue.get("needs_location_link")) or bool(issue.get("needs_location_state_fill"))) and loc_strategy != "skip":
            compound_locs = _index_locations(working, key="compound_locations", field="compound")
            protein_locs = _index_locations(working, key="protein_locations", field="protein")
            loc_key = "compound_locations" if kind == "compound" else "protein_locations"
            name_key = "compound" if kind == "compound" else "protein"
            by_name = compound_locs if kind == "compound" else protein_locs
            rows = by_name.get(name, [])
            has_valid_state = any(
                isinstance(_safe_dict(wrap.get("row")).get("biological_state"), str)
                and _safe_dict(wrap.get("row")).get("biological_state", "").strip()
                for wrap in rows
            )
            need_fill_state = bool(rows) and not has_valid_state
            need_add_row = not bool(rows)

            loc_candidates: List[Dict[str, Any]] = []
            if loc_strategy in {"db_then_default"}:
                loc_candidates = _db_location_candidates(db, kind=kind, name=name, max_items=6)
            if not loc_candidates:
                loc_candidates = [{"location": fallback_location, "score": 1.0, "source": "default", "evidence": "fallback_cell"}]
            loc_decision = _llm_choose_location(
                kind=kind,
                name=name,
                candidates=loc_candidates,
                use_llm=use_llm,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )
            chosen_loc = _canonical(str(loc_decision.get("choice", ""))) or fallback_location
            state_name = _ensure_biological_state(working, chosen_loc, global_organism)
            op_exec["location_candidates"] = loc_candidates[:6]
            op_exec["location_decision"] = loc_decision
            op_exec["chosen_location"] = chosen_loc
            op_exec["chosen_state"] = state_name
            if not state_name:
                report["actions"].append(
                    {
                        "type": "biological_state_not_created",
                        "entity_type": kind,
                        "name": name,
                        "chosen_location": chosen_loc,
                        "reason": "missing_species_or_location",
                        "stage": "stage3",
                    }
                )

            if need_fill_state and state_name:
                for row_wrap in rows:
                    row = _safe_dict(row_wrap.get("row"))
                    if _canonical(str(row.get("biological_state", ""))):
                        continue
                    row["biological_state"] = state_name
                    report["summary"]["location_states_filled"] += 1
                    report["actions"].append(
                        {
                            "type": "location_state_filled",
                            "entity_type": kind,
                            "name": name,
                            "chosen_location": chosen_loc,
                            "biological_state": state_name,
                            "decision": loc_decision,
                            "candidates": loc_candidates[:6],
                            "stage": "stage3",
                        }
                    )

            if need_add_row and state_name:
                new_row = {name_key: name, "biological_state": state_name}
                _safe_list(elem_locs.get(loc_key)).append(new_row)
                report["summary"]["locations_added"] += 1
                report["actions"].append(
                    {
                        "type": "location_added",
                        "entity_type": kind,
                        "name": name,
                        "location_key": loc_key,
                        "row": new_row,
                        "decision": loc_decision,
                        "candidates": loc_candidates[:6],
                        "stage": "stage3",
                    }
                )

        report["stage3"]["executions"].append(op_exec)

    # -----------------------------------------------------------------------
    # Step 10: Enrichment agent - uses QA flags and Stage 3 issues for tool-backed patches
    # -----------------------------------------------------------------------
    agent_issue_types = {"species", "biological_state", "protein_complex", "reaction_modifier"}
    stage3_agent_issues = [
        issue for issue in issues
        if isinstance(issue, dict) and str(issue.get("entity_type") or "") in agent_issue_types
    ]
    if use_llm and (qa_report is not None or stage3_agent_issues):
        enrichment_patches, enrichment_report = _run_enrichment_agent(
            working,
            qa_report or {"flags": {}},
            db=db,
            client=client,
            global_organism=global_organism,
            llm_temperature=llm_temperature,
            llm_max_tokens=max(llm_max_tokens, 1200),
            max_flags_per_type=max(1, max_items // 4),
            reaction_summary=reaction_summary,
            stage3_issues=stage3_agent_issues,
        )
        report["enrichment"] = enrichment_report

        if enrichment_patches:
            working, patch_apply_report = apply_patch_with_policy(working, enrichment_patches)
            report["enrichment"]["patch_application"] = patch_apply_report
            report["summary"]["enrichment_patches_accepted"] = patch_apply_report.get("summary", {}).get("accepted_count", 0)
            report["summary"]["enrichment_patches_rejected"] = patch_apply_report.get("summary", {}).get("rejected_count", 0)
        else:
            report["summary"]["enrichment_patches_accepted"] = 0
            report["summary"]["enrichment_patches_rejected"] = 0
    else:
        report["summary"]["enrichment_patches_accepted"] = 0
        report["summary"]["enrichment_patches_rejected"] = 0

    if db is not None:
        report["db"] = {"available": db.available(), "last_error": db.last_error}
        db.close()
    else:
        report["db"] = {"available": False, "last_error": "db_not_used"}

    return working, report


def run_gap_resolution(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    id_source: str = "hybrid",
    db_config: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
    llm_temperature: float = 0.15,
    llm_max_tokens: int = 900,
    max_items: int = 80,
    enable_id_resolution: bool = True,
    qa_report: Optional[Dict[str, Any]] = None,
    qa_report_path: Optional[Path] = None,
    reaction_summary: Optional[str] = None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    # Load QA report from path if not passed directly
    effective_qa_report = qa_report
    if effective_qa_report is None and qa_report_path is not None and qa_report_path.exists():
        raw_qa = json.loads(qa_report_path.read_text(encoding="utf-8"))
        effective_qa_report = raw_qa if isinstance(raw_qa, dict) else None

    resolved, report = resolve_gaps(
        payload,
        id_source=id_source,
        db_config=db_config,
        use_llm=use_llm,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        max_items=max_items,
        enable_id_resolution=enable_id_resolution,
        qa_report=effective_qa_report,
        reaction_summary=reaction_summary,
    )
    output_path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve missing pathway JSON fields via DB/API retrieval and constrained LLM decisions.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSON path")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSON path")
    parser.add_argument("--report", dest="report_path", default="gap_resolution_report.json", help="Gap resolution report path")
    parser.add_argument("--id-source", dest="id_source", choices=["api", "db", "hybrid"], default="hybrid")
    parser.add_argument("--db-host", dest="db_host", default="")
    parser.add_argument("--db-port", dest="db_port", type=int, default=3306)
    parser.add_argument("--db-user", dest="db_user", default="")
    parser.add_argument("--db-password", dest="db_password", default="")
    parser.add_argument("--db-schema", dest="db_schema", default="pathbank")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM candidate selection and use deterministic top-choice.")
    parser.add_argument(
        "--skip-id-resolution",
        action="store_true",
        help="Skip Stage-3 ID resolution and only fill organism/location gaps.",
    )
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-items", type=int, default=80)
    args = parser.parse_args()

    run_gap_resolution(
        Path(args.input_path),
        Path(args.output_path),
        Path(args.report_path),
        id_source=args.id_source,
        db_config={
            "host": args.db_host,
            "port": args.db_port,
            "user": args.db_user,
            "password": args.db_password,
            "schema": args.db_schema,
        },
        use_llm=not args.no_llm,
        llm_temperature=float(args.temperature),
        llm_max_tokens=int(args.max_tokens),
        max_items=int(args.max_items),
        enable_id_resolution=not args.skip_id_resolution,
    )


if __name__ == "__main__":
    main()
