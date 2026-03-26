from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_client import chat
from map_ids import (
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


def _state_maps(payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    by_loc_norm: Dict[str, str] = {}
    for state in _safe_list(payload.get("biological_states")):
        if not isinstance(state, dict):
            continue
        name = (state.get("name") or "").strip() if isinstance(state.get("name"), str) else ""
        if not name:
            continue
        by_name[name] = state
        location = (state.get("subcellular_location") or "").strip() if isinstance(state.get("subcellular_location"), str) else ""
        if location:
            by_loc_norm.setdefault(_normalize(location), name)
    return by_name, by_loc_norm


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


def _ensure_biological_state(payload: Dict[str, Any], location: str, species: str) -> str:
    states = payload.setdefault("biological_states", [])
    if not isinstance(states, list):
        states = []
        payload["biological_states"] = states
    by_name, by_loc_norm = _state_maps(payload)
    loc_norm = _normalize(location)
    existing_name = by_loc_norm.get(loc_norm)
    if existing_name:
        return existing_name
    candidate_name = f"AutoState_{_slug(location)}"
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


def _collect_stage3_issues(payload: Dict[str, Any], *, max_items: int) -> List[Dict[str, Any]]:
    entities = _safe_dict(payload.get("entities"))
    compound_locs = _index_locations(payload, key="compound_locations", field="compound")
    protein_locs = _index_locations(payload, key="protein_locations", field="protein")
    issues: List[Dict[str, Any]] = []

    for kind, rows in [("protein", _safe_list(entities.get("proteins"))), ("compound", _safe_list(entities.get("compounds")))]:
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = _canonical(str(item.get("name", "")))
            if not name:
                continue
            mapped_ids = _safe_dict(item.get("mapped_ids"))
            location_rows = protein_locs.get(name, []) if kind == "protein" else compound_locs.get(name, [])
            has_location_row = bool(location_rows)
            has_location_state = any(
                isinstance(_safe_dict(wrap.get("row")).get("biological_state"), str)
                and _safe_dict(wrap.get("row")).get("biological_state", "").strip()
                for wrap in location_rows
            )
            issue = {
                "issue_key": _issue_key(kind, name),
                "entity_type": kind,
                "name": name,
                "needs_id_mapping": not bool(mapped_ids),
                "needs_location_link": not has_location_row,
                "needs_location_state_fill": has_location_row and not has_location_state,
                "needs_organism": bool(kind == "protein" and not _canonical(str(item.get("organism", "")))),
            }
            if any(
                bool(issue.get(field))
                for field in [
                    "needs_id_mapping",
                    "needs_location_link",
                    "needs_location_state_fill",
                    "needs_organism",
                ]
            ):
                issues.append(issue)
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

            if need_fill_state:
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

            if need_add_row:
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
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    resolved, report = resolve_gaps(
        payload,
        id_source=id_source,
        db_config=db_config,
        use_llm=use_llm,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        max_items=max_items,
        enable_id_resolution=enable_id_resolution,
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
