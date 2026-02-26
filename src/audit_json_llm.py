from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_client import PROVIDER, chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict JSON auditor for PWML-like pathway payloads.
You must detect schema, connectivity, location, transport, and ID-readiness issues.

Rules:
- Output ONLY valid JSON.
- Do not change biological meaning.
- Propose minimal patch operations only when evidence exists in the provided JSON.
- No outside facts. Evidence must quote existing JSON snippets.

Return JSON object:
{
  "issues": {
    "errors": [ { "path": "/json/pointer", "reason": "", "evidence": "" } ],
    "warnings": [ { "path": "/json/pointer", "reason": "", "evidence": "" } ],
    "suggestions": [ { "path": "/json/pointer", "reason": "", "evidence": "" } ]
  },
  "patch": [
    {
      "action": "add|replace|remove",
      "json_pointer": "/path",
      "new_value": null,
      "reason": "",
      "confidence": 0.0,
      "evidence": ""
    }
  ]
}
"""


def _escape_pointer(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _join_pointer(parts: Sequence[Any]) -> str:
    if not parts:
        return ""
    tokens = [_escape_pointer(str(part)) for part in parts]
    return "/" + "/".join(tokens)


def _normalize_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", cleaned)


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.replace("```json", "```").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _all_entity_name_sets(payload: Dict[str, Any]) -> Dict[str, set]:
    entities = _safe_dict(payload.get("entities"))
    out: Dict[str, set] = {
        "compounds": set(),
        "proteins": set(),
        "protein_complexes": set(),
        "element_collections": set(),
        "nucleic_acids": set(),
        "species": set(),
        "subcellular_locations": set(),
    }
    for key in list(out.keys()):
        for item in _safe_list(entities.get(key)):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                out[key].add(name.strip())
    return out


def _location_alias_suggestions(locations: Sequence[str]) -> List[Dict[str, str]]:
    aliases = {
        "cytosol": "cytoplasm",
        "cytoplasmic": "cytoplasm",
        "mitochondrial matrix": "mitochondrion matrix",
        "golgi": "golgi apparatus",
        "er": "endoplasmic reticulum",
    }
    present = {_normalize_name(loc): loc for loc in locations if isinstance(loc, str) and loc.strip()}
    out: List[Dict[str, str]] = []
    for src, dst in aliases.items():
        nsrc = _normalize_name(src)
        ndst = _normalize_name(dst)
        if nsrc in present and ndst in present and present[nsrc] != present[ndst]:
            out.append({"from": present[nsrc], "to": present[ndst]})
    return out


def _deterministic_audit(payload: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    issues: Dict[str, List[Dict[str, Any]]] = {"errors": [], "warnings": [], "suggestions": []}
    patch_ops: List[Dict[str, Any]] = []

    entities = _safe_dict(payload.get("entities"))
    processes = _safe_dict(payload.get("processes"))
    locations = _safe_dict(payload.get("element_locations"))
    biological_states = _safe_list(payload.get("biological_states"))

    for top in ["entities", "processes"]:
        if not isinstance(payload.get(top), dict):
            issues["errors"].append(
                {
                    "path": _join_pointer([top]),
                    "reason": f"Missing required object '{top}'.",
                    "evidence": f"{top} is absent or not an object.",
                    "source": "deterministic",
                }
            )

    for name in ["compounds", "proteins"]:
        if name not in entities:
            issues["warnings"].append(
                {
                    "path": _join_pointer(["entities", name]),
                    "reason": f"Missing entities.{name} list.",
                    "evidence": "Field not found.",
                    "source": "deterministic",
                }
            )
        elif not _safe_list(entities.get(name)):
            issues["warnings"].append(
                {
                    "path": _join_pointer(["entities", name]),
                    "reason": f"entities.{name} is empty.",
                    "evidence": "List exists but has no entries.",
                    "source": "deterministic",
                }
            )

    if "reactions" not in processes:
        issues["warnings"].append(
            {
                "path": _join_pointer(["processes", "reactions"]),
                "reason": "Missing processes.reactions list.",
                "evidence": "Field not found.",
                "source": "deterministic",
            }
        )

    # Entity name checks + duplicate conflicting attributes
    for section_name, entries in entities.items():
        if not isinstance(entries, list):
            continue
        seen: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(entries):
            ptr = _join_pointer(["entities", section_name, idx])
            if not isinstance(entry, dict):
                issues["warnings"].append(
                    {
                        "path": ptr,
                        "reason": "Entity entry is not an object.",
                        "evidence": str(entry)[:80],
                        "source": "deterministic",
                    }
                )
                continue
            name = (entry.get("name") or "").strip() if isinstance(entry.get("name"), str) else ""
            if not name:
                issues["errors"].append(
                    {
                        "path": _join_pointer(["entities", section_name, idx, "name"]),
                        "reason": "Entity is missing a non-empty name.",
                        "evidence": json.dumps(entry, ensure_ascii=False)[:200],
                        "source": "deterministic",
                    }
                )
                continue

            normalized = _normalize_name(name)
            comparable = {k: v for k, v in entry.items() if k != "name"}
            if normalized in seen and seen[normalized] != comparable:
                issues["warnings"].append(
                    {
                        "path": ptr,
                        "reason": "Duplicate entity name has conflicting attributes.",
                        "evidence": name,
                        "source": "deterministic",
                    }
                )
            else:
                seen[normalized] = comparable

    names = _all_entity_name_sets(payload)
    compound_like = names["compounds"] | names["element_collections"]
    protein_like = names["proteins"] | names["protein_complexes"]

    reaction_inputs = set()
    reaction_outputs = set()

    for idx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        ptr = _join_pointer(["processes", "reactions", idx])
        if not isinstance(reaction, dict):
            issues["errors"].append(
                {
                    "path": ptr,
                    "reason": "Reaction entry is not an object.",
                    "evidence": str(reaction)[:120],
                    "source": "deterministic",
                }
            )
            continue
        inputs = [x.strip() for x in _safe_list(reaction.get("inputs")) if isinstance(x, str) and x.strip()]
        outputs = [x.strip() for x in _safe_list(reaction.get("outputs")) if isinstance(x, str) and x.strip()]
        if not inputs or not outputs:
            issues["errors"].append(
                {
                    "path": ptr,
                    "reason": "Reaction must include at least one input and one output.",
                    "evidence": json.dumps({"inputs": reaction.get("inputs"), "outputs": reaction.get("outputs")}),
                    "source": "deterministic",
                }
            )
        for c in inputs:
            reaction_inputs.add(c)
            if c not in compound_like:
                issues["errors"].append(
                    {
                        "path": _join_pointer(["processes", "reactions", idx, "inputs"]),
                        "reason": f"Reaction input '{c}' does not exist in entities.",
                        "evidence": c,
                        "source": "deterministic",
                    }
                )
        for c in outputs:
            reaction_outputs.add(c)
            if c not in compound_like:
                issues["errors"].append(
                    {
                        "path": _join_pointer(["processes", "reactions", idx, "outputs"]),
                        "reason": f"Reaction output '{c}' does not exist in entities.",
                        "evidence": c,
                        "source": "deterministic",
                    }
                )

        for enz_idx, enzyme in enumerate(_safe_list(reaction.get("enzymes"))):
            if not isinstance(enzyme, dict):
                continue
            candidate = ""
            if isinstance(enzyme.get("protein_complex"), str):
                candidate = enzyme["protein_complex"].strip()
            elif isinstance(enzyme.get("protein"), str):
                candidate = enzyme["protein"].strip()
            if candidate and candidate not in protein_like:
                issues["warnings"].append(
                    {
                        "path": _join_pointer(["processes", "reactions", idx, "enzymes", enz_idx]),
                        "reason": f"Enzyme reference '{candidate}' not found in proteins/protein_complexes.",
                        "evidence": json.dumps(enzyme, ensure_ascii=False)[:160],
                        "source": "deterministic",
                    }
                )

    for idx, transport in enumerate(_safe_list(processes.get("transports"))):
        ptr = _join_pointer(["processes", "transports", idx])
        if not isinstance(transport, dict):
            issues["warnings"].append(
                {
                    "path": ptr,
                    "reason": "Transport entry is not an object.",
                    "evidence": str(transport)[:120],
                    "source": "deterministic",
                }
            )
            continue
        cargo = (transport.get("cargo") or "").strip() if isinstance(transport.get("cargo"), str) else ""
        source_state = (
            transport.get("from_biological_state") or ""
            if isinstance(transport.get("from_biological_state"), str)
            else ""
        )
        dest_state = (
            transport.get("to_biological_state") or ""
            if isinstance(transport.get("to_biological_state"), str)
            else ""
        )
        if not cargo:
            issues["errors"].append(
                {
                    "path": _join_pointer(["processes", "transports", idx, "cargo"]),
                    "reason": "Transport is missing cargo compound.",
                    "evidence": json.dumps(transport, ensure_ascii=False)[:200],
                    "source": "deterministic",
                }
            )
        elif cargo not in compound_like:
            issues["errors"].append(
                {
                    "path": _join_pointer(["processes", "transports", idx, "cargo"]),
                    "reason": f"Transport cargo '{cargo}' is not in entity registries.",
                    "evidence": cargo,
                    "source": "deterministic",
                }
            )
        if not source_state or not dest_state:
            issues["warnings"].append(
                {
                    "path": ptr,
                    "reason": "Transport should provide source and destination biological states.",
                    "evidence": json.dumps(
                        {
                            "from_biological_state": transport.get("from_biological_state"),
                            "to_biological_state": transport.get("to_biological_state"),
                        }
                    ),
                    "source": "deterministic",
                }
            )

    # Orphans
    used_compounds = reaction_inputs | reaction_outputs
    for idx, item in enumerate(_safe_list(entities.get("compounds"))):
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip() if isinstance(item.get("name"), str) else ""
        if name and name not in used_compounds:
            issues["warnings"].append(
                {
                    "path": _join_pointer(["entities", "compounds", idx, "name"]),
                    "reason": f"Compound '{name}' is not used in any reaction/transport.",
                    "evidence": name,
                    "source": "deterministic",
                }
            )

    # Locations and compartments readiness
    states_by_name: Dict[str, Dict[str, Any]] = {}
    for sidx, state in enumerate(biological_states):
        if not isinstance(state, dict):
            continue
        name = (state.get("name") or "").strip() if isinstance(state.get("name"), str) else ""
        if not name:
            continue
        states_by_name[name] = state
        subcell = (state.get("subcellular_location") or "").strip() if isinstance(state.get("subcellular_location"), str) else ""
        if not subcell:
            patch_ops.append(
                {
                    "op": "add",
                    "path": _join_pointer(["biological_states", sidx, "subcellular_location"]),
                    "value": "cell",
                    "reason": "Missing subcellular_location; default compartment can be applied.",
                    "confidence": 0.7,
                    "evidence": json.dumps(state, ensure_ascii=False)[:200],
                    "source": "deterministic",
                }
            )
            issues["warnings"].append(
                {
                    "path": _join_pointer(["biological_states", sidx]),
                    "reason": "Biological state missing subcellular_location.",
                    "evidence": json.dumps(state, ensure_ascii=False)[:160],
                    "source": "deterministic",
                }
            )

    entity_to_states: Dict[str, set] = defaultdict(set)
    for lkey, entity_key in [
        ("compound_locations", "compound"),
        ("protein_locations", "protein"),
        ("nucleic_acid_locations", "nucleic_acid"),
        ("element_collection_locations", "element_collection"),
    ]:
        for lidx, row in enumerate(_safe_list(locations.get(lkey))):
            if not isinstance(row, dict):
                continue
            entity = (row.get(entity_key) or "").strip() if isinstance(row.get(entity_key), str) else ""
            if not entity:
                continue
            state_name = (row.get("biological_state") or "").strip() if isinstance(row.get("biological_state"), str) else ""
            if not state_name:
                issues["warnings"].append(
                    {
                        "path": _join_pointer(["element_locations", lkey, lidx]),
                        "reason": "Location row missing biological_state; compartment resolution may degrade.",
                        "evidence": json.dumps(row, ensure_ascii=False)[:180],
                        "source": "deterministic",
                    }
                )
            else:
                entity_to_states[entity].add(state_name)
                if state_name not in states_by_name:
                    issues["errors"].append(
                        {
                            "path": _join_pointer(["element_locations", lkey, lidx, "biological_state"]),
                            "reason": f"biological_state '{state_name}' does not exist in biological_states.",
                            "evidence": state_name,
                            "source": "deterministic",
                        }
                    )

    for pidx, protein in enumerate(_safe_list(entities.get("proteins"))):
        if not isinstance(protein, dict):
            continue
        name = (protein.get("name") or "").strip() if isinstance(protein.get("name"), str) else ""
        if not name:
            continue
        if not entity_to_states.get(name):
            issues["warnings"].append(
                {
                    "path": _join_pointer(["entities", "proteins", pidx]),
                    "reason": "Protein has no location link; default compartment may be used.",
                    "evidence": name,
                    "source": "deterministic",
                }
            )

    # ID readiness: propagate organism when globally available
    species_names = sorted(names["species"])
    global_organism = species_names[0] if len(species_names) == 1 else ""
    if not global_organism:
        organisms = {
            (state.get("species") or "").strip()
            for state in biological_states
            if isinstance(state, dict) and isinstance(state.get("species"), str) and state.get("species").strip()
        }
        global_organism = sorted(organisms)[0] if len(organisms) == 1 else ""

    if global_organism:
        for pidx, protein in enumerate(_safe_list(entities.get("proteins"))):
            if not isinstance(protein, dict):
                continue
            org = (protein.get("organism") or "").strip() if isinstance(protein.get("organism"), str) else ""
            if not org:
                patch_ops.append(
                    {
                        "op": "add",
                        "path": _join_pointer(["entities", "proteins", pidx, "organism"]),
                        "value": global_organism,
                        "reason": "Propagate global organism onto protein for mapping readiness.",
                        "confidence": 0.72,
                        "evidence": f"Global organism in payload: {global_organism}",
                        "source": "deterministic",
                    }
                )

    location_names = [x for x in names["subcellular_locations"]]
    alias_pairs = _location_alias_suggestions(location_names)
    if alias_pairs:
        issues["suggestions"].append(
            {
                "path": _join_pointer(["entities", "subcellular_locations"]),
                "reason": "Potential inconsistent location spelling; consider normalization table.",
                "evidence": json.dumps(alias_pairs, ensure_ascii=False),
                "normalization_map": alias_pairs,
                "source": "deterministic",
            }
        )

    return issues, patch_ops


def _build_llm_prompt(payload: Dict[str, Any]) -> str:
    payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
    return "\n".join(
        [
            "Audit this pathway JSON for SBML conversion readiness.",
            "Return issues and a minimal patch list.",
            "Use RFC6901 json pointers for paths.",
            "Use action add/replace/remove and include confidence 0..1.",
            "Input JSON:",
            "<<<",
            payload_str,
            ">>>",
        ]
    )


def _normalize_patch_op(op: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(op, dict):
        return None
    action = op.get("action") or op.get("op")
    if not isinstance(action, str):
        return None
    action = action.strip().lower()
    if action not in {"add", "replace", "remove"}:
        return None
    pointer = op.get("json_pointer") if "json_pointer" in op else op.get("path")
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    normalized: Dict[str, Any] = {
        "op": action,
        "path": pointer,
        "reason": str(op.get("reason", "")).strip(),
        "confidence": float(op.get("confidence", 0.0)),
        "evidence": str(op.get("evidence", "")).strip(),
        "source": op.get("source", "llm"),
    }
    if action != "remove":
        normalized["value"] = op.get("new_value") if "new_value" in op else op.get("value")
    return normalized


def _merge_issues(
    deterministic: Dict[str, List[Dict[str, Any]]],
    llm_issues: Optional[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    merged = {
        "errors": list(deterministic.get("errors", [])),
        "warnings": list(deterministic.get("warnings", [])),
        "suggestions": list(deterministic.get("suggestions", [])),
    }
    if not isinstance(llm_issues, dict):
        return merged
    for key in ["errors", "warnings", "suggestions"]:
        for issue in _safe_list(llm_issues.get(key)):
            if not isinstance(issue, dict):
                continue
            item = dict(issue)
            item.setdefault("source", "llm")
            merged[key].append(item)
    return merged


def run_audit(
    input_path: Path,
    audit_report_path: Path,
    audit_patch_path: Path,
    *,
    use_llm: bool = True,
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 1800,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    deterministic_issues, deterministic_patch = _deterministic_audit(payload)
    llm_raw = ""
    llm_error = ""
    llm_payload: Optional[Dict[str, Any]] = None

    if use_llm:
        try:
            llm_raw = chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_llm_prompt(payload)},
                ],
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )
            llm_payload = _extract_json_object(llm_raw)
            if llm_payload is None:
                llm_error = "LLM output was not valid JSON."
        except Exception as exc:  # noqa: BLE001
            llm_error = f"LLM audit call failed: {exc}"
            logger.warning("LLM audit failed: %s", exc)

    llm_issues = _safe_dict(llm_payload.get("issues")) if isinstance(llm_payload, dict) else {}
    merged_issues = _merge_issues(deterministic_issues, llm_issues)

    patch_ops: List[Dict[str, Any]] = list(deterministic_patch)
    if isinstance(llm_payload, dict):
        for raw_op in _safe_list(llm_payload.get("patch")):
            op = _normalize_patch_op(raw_op)
            if op:
                patch_ops.append(op)

    audit_report = {
        "summary": {
            "error_count": len(merged_issues["errors"]),
            "warning_count": len(merged_issues["warnings"]),
            "suggestion_count": len(merged_issues["suggestions"]),
            "patch_count": len(patch_ops),
        },
        "errors": merged_issues["errors"],
        "warnings": merged_issues["warnings"],
        "suggestions": merged_issues["suggestions"],
        "llm": {
            "enabled": use_llm,
            "provider": PROVIDER,
            "ok": bool(use_llm and llm_payload is not None),
            "error": llm_error,
            "raw_preview": llm_raw[:800],
        },
    }

    audit_report_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_patch_path.write_text(json.dumps(patch_ops, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_report


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-backed JSON auditor for post-pipeline SBML readiness.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input final JSON path")
    parser.add_argument(
        "--audit-report",
        dest="audit_report_path",
        default="audit_report.json",
        help="Output audit report JSON path",
    )
    parser.add_argument(
        "--audit-patch",
        dest="audit_patch_path",
        default="audit_patch.json",
        help="Output audit patch JSON path",
    )
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM auditing and run deterministic checks only.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature for audit call")
    parser.add_argument("--max-tokens", type=int, default=1800, help="LLM max tokens for audit call")
    args = parser.parse_args()

    run_audit(
        Path(args.input_path),
        Path(args.audit_report_path),
        Path(args.audit_patch_path),
        use_llm=not args.no_llm,
        llm_temperature=float(args.temperature),
        llm_max_tokens=int(args.max_tokens),
    )
    print(f"Wrote audit report: {args.audit_report_path}")
    print(f"Wrote audit patch: {args.audit_patch_path}")


if __name__ == "__main__":
    main()

