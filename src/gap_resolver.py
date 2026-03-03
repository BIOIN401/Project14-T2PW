from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_client import chat
from map_ids import HttpClient, PathBankDbResolver, map_compound_all, map_protein_uniprot


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
    state_obj = {"name": candidate_name, "subcellular_location": _canonical(location)}
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
            use_cache=False,
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

    # Fill missing organisms + IDs
    items: List[Tuple[str, Dict[str, Any]]] = []
    for protein in _safe_list(entities.get("proteins")):
        if isinstance(protein, dict):
            items.append(("protein", protein))
    for compound in _safe_list(entities.get("compounds")):
        if isinstance(compound, dict):
            items.append(("compound", compound))

    for kind, item in items[: max(1, int(max_items))]:
        name = _canonical(str(item.get("name", "")))
        if not name:
            continue
        report["summary"]["items_considered"] += 1
        if kind == "protein":
            organism = _canonical(str(item.get("organism", "")))
            if not organism and global_organism:
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

        mapped_ids = _safe_dict(item.get("mapped_ids"))
        if not mapped_ids:
            organism = _canonical(str(item.get("organism", ""))) if kind == "protein" else ""
            mapping = _map_ids_for_entity(
                kind=kind,
                name=name,
                organism=organism,
                id_source=id_source,
                db=db,
                client=client,
            )
            if mapping.get("status") == "mapped":
                new_ids = _safe_dict(mapping.get("mapped_ids"))
                if new_ids:
                    item["mapped_ids"] = {**mapped_ids, **new_ids}
                    item.setdefault("mapping_meta", {})
                    item["mapping_meta"]["provider"] = mapping.get("provider", "")
                    item["mapping_meta"]["source"] = mapping.get("source", "")
                    item["mapping_meta"]["confidence"] = float(mapping.get("confidence", 0.0))
                    item["mapping_meta"]["chosen_rule"] = mapping.get("chosen_rule", "")
                    report["summary"]["mapped_ids_added"] += 1
                    report["actions"].append(
                        {
                            "type": "mapped_ids_added",
                            "entity_type": kind,
                            "name": name,
                            "mapped_ids": new_ids,
                            "provider": mapping.get("provider", ""),
                            "source": mapping.get("source", ""),
                            "confidence": float(mapping.get("confidence", 0.0)),
                        }
                    )

    # Fill missing location links
    compound_locs = _index_locations(working, key="compound_locations", field="compound")
    protein_locs = _index_locations(working, key="protein_locations", field="protein")
    state_by_name, _ = _state_maps(working)
    fallback_location = "cell"

    for kind, entries in [("compound", _safe_list(entities.get("compounds"))), ("protein", _safe_list(entities.get("proteins")))]:
        loc_key = "compound_locations" if kind == "compound" else "protein_locations"
        name_key = "compound" if kind == "compound" else "protein"
        by_name = compound_locs if kind == "compound" else protein_locs

        for item in entries[: max(1, int(max_items))]:
            if not isinstance(item, dict):
                continue
            name = _canonical(str(item.get("name", "")))
            if not name:
                continue
            rows = by_name.get(name, [])
            valid_rows = [
                row
                for row in rows
                if isinstance(row, dict)
                and isinstance(_safe_dict(row.get("row")).get("biological_state"), str)
                and _safe_dict(row.get("row")).get("biological_state", "").strip()
            ]

            if valid_rows:
                # fill missing biological_state on existing location rows if any
                for row_wrap in rows:
                    row = _safe_dict(row_wrap.get("row"))
                    state_name = _canonical(str(row.get("biological_state", "")))
                    if state_name:
                        continue
                    candidates = _db_location_candidates(db, kind=kind, name=name, max_items=6)
                    if not candidates:
                        candidates = [{"location": fallback_location, "score": 1.0, "source": "default", "evidence": "fallback_cell"}]
                    decision = _llm_choose_location(
                        kind=kind,
                        name=name,
                        candidates=candidates,
                        use_llm=use_llm,
                        temperature=llm_temperature,
                        max_tokens=llm_max_tokens,
                    )
                    chosen_loc = _canonical(str(decision.get("choice", ""))) or fallback_location
                    state = _ensure_biological_state(working, chosen_loc, global_organism)
                    row["biological_state"] = state
                    report["summary"]["location_states_filled"] += 1
                    report["actions"].append(
                        {
                            "type": "location_state_filled",
                            "entity_type": kind,
                            "name": name,
                            "chosen_location": chosen_loc,
                            "biological_state": state,
                            "decision": decision,
                            "candidates": candidates[:6],
                        }
                    )
                continue

            candidates = _db_location_candidates(db, kind=kind, name=name, max_items=6)
            if not candidates:
                candidates = [{"location": fallback_location, "score": 1.0, "source": "default", "evidence": "fallback_cell"}]
            decision = _llm_choose_location(
                kind=kind,
                name=name,
                candidates=candidates,
                use_llm=use_llm,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )
            chosen_loc = _canonical(str(decision.get("choice", ""))) or fallback_location
            state_name = _ensure_biological_state(working, chosen_loc, global_organism)
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
                    "decision": decision,
                    "candidates": candidates[:6],
                }
            )

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
    )


if __name__ == "__main__":
    main()
