from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from t2pw.llm.client import PROVIDER, chat
from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict JSON auditor and repair planner for PWML-like pathway payloads.
Your job is to produce a safe, minimal patch plan that improves SBML conversion readiness.

Priorities (highest to lowest):
1) Structural validity (required fields/object/list shapes)
2) Process coherence (reaction/transport references, no impossible links)
3) Location/compartment readiness
4) ID-mapping readiness
5) Cosmetic consistency

Hard constraints:
- Output ONLY valid JSON, no markdown, no prose outside JSON.
- Do not invent biology not supported by the input payload.
- Do not remove core process steps unless they are exact duplicates or clearly invalid.
- Prefer add/replace over remove.
- Every issue and patch must include concrete evidence from input JSON.
- Be conservative: if uncertain, emit warning/suggestion instead of risky patch.
- If upstream context includes a "selected_example" field (non-empty), any entity or process whose evidence quotes belong to a different named example than selected_example may be removed at high confidence (>= 0.90). This is a valid out-of-scope removal, not a data loss.
- If upstream context indicates document_type = "multi_example_review" and selected_example is empty, do NOT repair the payload into a single pathway. Emit a scope_ambiguity error/warning in the issues block instead of adding bridging reactions or compartment assignments.

TYPE-CONFLICT REPAIRS (high confidence, apply before other patches):
These are deterministic type errors that should be patched at confidence >= 0.90:
- A small-molecule cofactor (PLP, pyridoxal phosphate, FAD, FMN, NAD+, NADH, NADP+, NADPH, SAM, CoA, ATP, ADP, AMP, heme, Mg2+, Fe2+, Fe3+, Zn2+, TPP, biotin, lipoate) listed under entities.proteins[] is a type error. Propose removing it from proteins[] and adding it to entities.compounds[] with class "cofactor".
- Before adding a moved cofactor to entities.compounds[], check whether the same name or an expanded synonym already exists there. If it does, only propose removing the incorrect proteins[] entry — do not create a duplicate compound entry.
- A named gene or protein listed under entities.compounds[] is a type error. Do NOT classify an abbreviation as a protein by pattern alone — move a compound to proteins[] only if the payload evidence explicitly identifies it as a gene, protein, enzyme, subunit, transporter, or catalyst. Pattern matching alone (e.g., 2-5 uppercase letters) is insufficient — require explicit evidence in the entity's evidence field or reaction modifiers.
- An ambiguous abbreviation (e.g., PLP, FAD, CoA) must NOT be reclassified to a different entity type unless the source evidence in the payload explicitly identifies it as that type. If evidence is ambiguous, emit a warning, not a patch.

Patch policy guidance:
- High confidence (>=0.95): deterministic structure fixes, obvious token splits, duplicate cleanup.
- Medium confidence (0.75-0.94): location/organism propagation, minor normalization.
- Low confidence (<0.75): avoid patching; report as warning/suggestion.
- Type-conflict repairs: confidence >= 0.90 (high — these are structural errors).
- Out-of-scope entity removal when selected_example is set: confidence >= 0.90.
- Scope ambiguity: emit warning/suggestion, not a patch.

Return JSON object with this shape:
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
  ],
  "repair_rationale": "Short 1-3 sentence summary of why these changes are the safest next step."
}

Cross-cutting constraints:
- Partial pathway evidence should remain partial. Do not add bridging reactions to make a partial mechanism look complete.
- Type compatibility beats confidence. Never patch an entity to a different type solely on confidence score.
- Named genes/proteins outrank generic enzyme class names.
- Cofactors are compounds. Proteins/genes are not compounds.
- Do not wrap a single enzyme into a protein_complex unless complex evidence exists in the payload.
"""

LOCK_AWARE_SYSTEM_PROMPT_FRAGMENT = """

LOCKED REACTION PRESERVATION POLICY:
The reaction set is locked. You may not delete, merge, split, reorder, or replace locked reactions.
Return patches only. Prefer additive repairs. If a locked reaction appears invalid, flag it for
quarantine instead of deleting it.

Specifically:
- Do NOT propose "remove" patches targeting any locked reaction index.
- Do NOT propose "replace" patches on /processes/reactions that would eliminate a locked reaction.
- Do NOT merge two or more locked reactions into a single reaction.
- Do NOT split a locked reaction into multiple new reactions.
- If a locked reaction has a structural issue, emit a warning/suggestion with quarantine recommendation
  rather than a remove/replace patch.
- Additive patches (adding missing entities, fixing enzyme references, etc.) are encouraged.
- Replacing individual fields within a locked reaction (name normalization, enzyme ref fixes) is
  acceptable as long as the reaction identity (inputs, outputs, core semantics) is preserved.
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


def _split_composite_name(value: str) -> List[str]:
    text = (value or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*\+\s*|\s+and\s+", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part and part.strip()]


def _is_composite_name(value: str) -> bool:
    return len(_split_composite_name(value)) > 1


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for value in values:
        norm = _normalize_name(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(value)
    return out


def _clean_preserve_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned:
            out.append(cleaned)
    return out


def _same_multiset(values_a: Sequence[str], values_b: Sequence[str]) -> bool:
    norm_a = [_normalize_name(v) for v in values_a if isinstance(v, str) and _normalize_name(v)]
    norm_b = [_normalize_name(v) for v in values_b if isinstance(v, str) and _normalize_name(v)]
    return Counter(norm_a) == Counter(norm_b)


def _expand_stoich_token(token: str) -> List[str]:
    text = (token or "").strip()
    if not text:
        return []
    match = re.match(r"^\s*(\d+)\s+(.+)$", text)
    if not match:
        return [text]
    count = max(1, min(8, int(match.group(1))))
    base = match.group(2).strip()
    if not base:
        return [text]
    return [base] * count


def _extract_equation_tokens(name: str) -> Tuple[List[str], List[str]]:
    text = (name or "").strip()
    if not text:
        return [], []
    arrow_match = re.search(r"(->|→)", text)
    if not arrow_match:
        return [], []
    lhs = text[: arrow_match.start()].strip()
    rhs = text[arrow_match.end() :].strip()
    if ":" in lhs:
        lhs = lhs.split(":", 1)[1].strip()
    lhs_parts = _split_composite_name(lhs) if lhs else []
    rhs_parts = _split_composite_name(rhs) if rhs else []
    lhs_out: List[str] = []
    rhs_out: List[str] = []
    for part in lhs_parts:
        lhs_out.extend(_expand_stoich_token(part))
    for part in rhs_parts:
        rhs_out.extend(_expand_stoich_token(part))
    lhs_out = _clean_preserve_order(lhs_out)
    rhs_out = _clean_preserve_order(rhs_out)
    return lhs_out, rhs_out


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


def _load_locked_manifest(path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    """Load a locked reaction manifest file, returning None on failure."""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict)]
        return None
    except Exception:  # noqa: BLE001
        return None


def _discover_locked_manifest_path(input_path: Path) -> Optional[Path]:
    """Auto-discover the locked manifest in the same directory as *input_path*."""
    parent = input_path.parent if not input_path.is_dir() else input_path
    candidate = parent / MANIFEST_FILENAME
    if candidate.exists():
        return candidate
    return None


def _locked_reaction_indices(
    payload: Dict[str, Any],
    locked_manifest: List[Dict[str, Any]],
) -> set:
    """Return the set of reaction indices that correspond to locked reactions.

    Matching uses locked_reaction_id, then source_reaction_id, then
    (name + inputs + outputs) fingerprint.
    """
    reactions = payload.get("processes", {}).get("reactions", []) if isinstance(payload.get("processes"), dict) else []
    if not isinstance(reactions, list):
        return set()

    locked_ids = set()
    source_ids = set()
    fingerprints: set = set()
    for entry in locked_manifest:
        lid = str(entry.get("locked_reaction_id", "")).strip()
        if lid:
            locked_ids.add(lid)
        sid = str(entry.get("source_reaction_id", "")).strip()
        if sid:
            source_ids.add(sid)
        name = _normalize_name(str(entry.get("name", "")))
        inputs = tuple(sorted(_normalize_name(i) for i in (entry.get("inputs") or []) if isinstance(i, str)))
        outputs = tuple(sorted(_normalize_name(o) for o in (entry.get("outputs") or []) if isinstance(o, str)))
        if name or inputs or outputs:
            fingerprints.add((name, inputs, outputs))

    indices: set = set()
    for idx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        # Match by locked_reaction_id
        rid = str(reaction.get("locked_reaction_id", "")).strip()
        if rid and rid in locked_ids:
            indices.add(idx)
            continue
        # Match by source_reaction_id
        for key in ("reaction_id", "id", "key", "source_reaction_id", "pathwhiz_reaction_id", "pathbank_reaction_id"):
            sid = reaction.get(key)
            if isinstance(sid, str) and sid.strip() in source_ids:
                indices.add(idx)
                break
            if isinstance(sid, (int, float)) and str(sid) in source_ids:
                indices.add(idx)
                break
        else:
            # Match by fingerprint
            name = _normalize_name(str(reaction.get("name", "") or ""))
            raw_inputs = reaction.get("inputs", [])
            raw_outputs = reaction.get("outputs", [])
            inputs = tuple(sorted(_normalize_name(i) for i in (raw_inputs if isinstance(raw_inputs, list) else []) if isinstance(i, str)))
            outputs = tuple(sorted(_normalize_name(o) for o in (raw_outputs if isinstance(raw_outputs, list) else []) if isinstance(o, str)))
            if (name, inputs, outputs) in fingerprints:
                indices.add(idx)
    return indices


def _deterministic_audit(payload: Dict[str, Any], *, locked_reaction_indices: Optional[set] = None) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
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

    composite_entity_removals: List[Tuple[str, int, str]] = []

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

            if section_name in {"compounds", "proteins"} and _is_composite_name(name):
                issues["warnings"].append(
                    {
                        "path": ptr,
                        "reason": "Composite entity name detected; process-style names should not be stored as standalone entities.",
                        "evidence": name,
                        "source": "deterministic",
                    }
                )
                composite_entity_removals.append((section_name, idx, name))

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

    for section_name, idx, name in sorted(composite_entity_removals, key=lambda item: item[1], reverse=True):
        patch_ops.append(
            {
                "op": "remove",
                "path": _join_pointer(["entities", section_name, idx]),
                "reason": "Remove composite entity name; split process mentions should be represented in reaction/transports.",
                "confidence": 0.99,
                "evidence": name,
                "source": "deterministic",
            }
        )

    names = _all_entity_name_sets(payload)
    compound_like = names["compounds"] | names["element_collections"]
    protein_like = names["proteins"] | names["protein_complexes"]
    compound_norms = {_normalize_name(name) for name in compound_like if name}
    protein_norms = {_normalize_name(name) for name in protein_like if name}

    reaction_inputs = set()
    reaction_outputs = set()
    process_compound_refs: List[str] = []
    process_protein_refs: List[str] = []

    reaction_remove_indices: List[int] = []
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
        raw_inputs = [x.strip() for x in _safe_list(reaction.get("inputs")) if isinstance(x, str) and x.strip()]
        raw_outputs = [x.strip() for x in _safe_list(reaction.get("outputs")) if isinstance(x, str) and x.strip()]
        inputs: List[str] = []
        outputs: List[str] = []
        for item in raw_inputs:
            inputs.extend(_split_composite_name(item) if _is_composite_name(item) else [item])
        for item in raw_outputs:
            outputs.extend(_split_composite_name(item) if _is_composite_name(item) else [item])
        inputs = _clean_preserve_order(inputs)
        outputs = _clean_preserve_order(outputs)

        reaction_name = (reaction.get("name") or "").strip() if isinstance(reaction.get("name"), str) else ""
        eq_inputs, eq_outputs = _extract_equation_tokens(reaction_name)
        if eq_inputs and not _same_multiset(eq_inputs, inputs):
            patch_ops.append(
                {
                    "op": "replace",
                    "path": _join_pointer(["processes", "reactions", idx, "inputs"]),
                    "value": eq_inputs,
                    "reason": "Align reaction inputs with explicit equation in reaction name.",
                    "confidence": 0.985,
                    "evidence": reaction_name[:220],
                    "source": "deterministic",
                }
            )
            issues["warnings"].append(
                {
                    "path": _join_pointer(["processes", "reactions", idx, "inputs"]),
                    "reason": "Reaction name equation disagrees with listed inputs.",
                    "evidence": reaction_name[:220],
                    "source": "deterministic",
                }
            )
            inputs = list(eq_inputs)
        if eq_outputs and not _same_multiset(eq_outputs, outputs):
            patch_ops.append(
                {
                    "op": "replace",
                    "path": _join_pointer(["processes", "reactions", idx, "outputs"]),
                    "value": eq_outputs,
                    "reason": "Align reaction outputs with explicit equation in reaction name.",
                    "confidence": 0.985,
                    "evidence": reaction_name[:220],
                    "source": "deterministic",
                }
            )
            issues["warnings"].append(
                {
                    "path": _join_pointer(["processes", "reactions", idx, "outputs"]),
                    "reason": "Reaction name equation disagrees with listed outputs.",
                    "evidence": reaction_name[:220],
                    "source": "deterministic",
                }
            )
            outputs = list(eq_outputs)

        if inputs and outputs and _same_multiset(inputs, outputs):
            is_locked = locked_reaction_indices is not None and idx in locked_reaction_indices
            if is_locked:
                issues["warnings"].append(
                    {
                        "path": ptr,
                        "reason": "Locked reaction has identical inputs and outputs; flagged for quarantine instead of removal.",
                        "evidence": json.dumps({"inputs": inputs, "outputs": outputs}, ensure_ascii=False)[:220],
                        "source": "deterministic",
                        "locked_preservation": True,
                    }
                )
            else:
                issues["errors"].append(
                    {
                        "path": ptr,
                        "reason": "Reaction has identical inputs and outputs with no transformation.",
                        "evidence": json.dumps({"inputs": inputs, "outputs": outputs}, ensure_ascii=False)[:220],
                        "source": "deterministic",
                    }
                )
                reaction_remove_indices.append(idx)
            continue

        if inputs != raw_inputs and inputs:
            patch_ops.append(
                {
                    "op": "replace",
                    "path": _join_pointer(["processes", "reactions", idx, "inputs"]),
                    "value": inputs,
                    "reason": "Split composite reaction input token(s) into individual entities.",
                    "confidence": 0.99,
                    "evidence": ", ".join(raw_inputs)[:200],
                    "source": "deterministic",
                }
            )
            issues["warnings"].append(
                {
                    "path": _join_pointer(["processes", "reactions", idx, "inputs"]),
                    "reason": "Composite reaction token detected; splitting into individual names.",
                    "evidence": ", ".join(raw_inputs)[:200],
                    "source": "deterministic",
                }
            )
        if outputs != raw_outputs and outputs:
            patch_ops.append(
                {
                    "op": "replace",
                    "path": _join_pointer(["processes", "reactions", idx, "outputs"]),
                    "value": outputs,
                    "reason": "Split composite reaction output token(s) into individual entities.",
                    "confidence": 0.99,
                    "evidence": ", ".join(raw_outputs)[:200],
                    "source": "deterministic",
                }
            )
            issues["warnings"].append(
                {
                    "path": _join_pointer(["processes", "reactions", idx, "outputs"]),
                    "reason": "Composite reaction token detected; splitting into individual names.",
                    "evidence": ", ".join(raw_outputs)[:200],
                    "source": "deterministic",
                }
            )

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
            process_compound_refs.append(c)
            reaction_inputs.add(c)
            norm_c = _normalize_name(c)
            if norm_c not in compound_norms and norm_c not in protein_norms:
                issues["errors"].append(
                    {
                        "path": _join_pointer(["processes", "reactions", idx, "inputs"]),
                        "reason": f"Reaction input '{c}' does not exist in entities.",
                        "evidence": c,
                        "source": "deterministic",
                    }
                )
        for c in outputs:
            process_compound_refs.append(c)
            reaction_outputs.add(c)
            norm_c = _normalize_name(c)
            if norm_c not in compound_norms and norm_c not in protein_norms:
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
            if candidate:
                process_protein_refs.append(candidate)
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
        cargo_items = _split_composite_name(cargo) if cargo else []
        if len(cargo_items) > 1:
            issues["warnings"].append(
                {
                    "path": _join_pointer(["processes", "transports", idx, "cargo"]),
                    "reason": "Composite transport cargo detected; split into one cargo per transport row.",
                    "evidence": cargo,
                    "source": "deterministic",
                }
            )
            patch_ops.append(
                {
                    "op": "replace",
                    "path": _join_pointer(["processes", "transports", idx, "cargo"]),
                    "value": cargo_items[0],
                    "reason": "Transport cargo must be a single entity token.",
                    "confidence": 0.99,
                    "evidence": cargo,
                    "source": "deterministic",
                }
            )
            for extra in cargo_items[1:]:
                cloned = deepcopy(transport)
                cloned["cargo"] = extra
                patch_ops.append(
                    {
                        "op": "add",
                        "path": _join_pointer(["processes", "transports", "-"]),
                        "value": cloned,
                        "reason": "Split composite transport cargo into separate transport entries.",
                        "confidence": 0.99,
                        "evidence": cargo,
                        "source": "deterministic",
                    }
                )
        elif len(cargo_items) == 1:
            cargo = cargo_items[0]

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
        else:
            process_compound_refs.append(cargo)
            norm_cargo = _normalize_name(cargo)
            if norm_cargo not in compound_norms and norm_cargo not in protein_norms:
                issues["errors"].append(
                    {
                        "path": _join_pointer(["processes", "transports", idx, "cargo"]),
                        "reason": f"Transport cargo '{cargo}' is not in entity registries.",
                        "evidence": cargo,
                        "source": "deterministic",
                    }
                )

        for t_idx, transporter in enumerate(_safe_list(transport.get("transporters"))):
            if not isinstance(transporter, dict):
                continue
            candidate = ""
            if isinstance(transporter.get("protein_complex"), str):
                candidate = transporter["protein_complex"].strip()
            elif isinstance(transporter.get("protein"), str):
                candidate = transporter["protein"].strip()
            if candidate:
                process_protein_refs.append(candidate)
            if candidate and candidate not in protein_like:
                issues["warnings"].append(
                    {
                        "path": _join_pointer(["processes", "transports", idx, "transporters", t_idx]),
                        "reason": f"Transporter reference '{candidate}' not found in proteins/protein_complexes.",
                        "evidence": json.dumps(transporter, ensure_ascii=False)[:160],
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
        transport_name = (transport.get("name") or "").strip() if isinstance(transport.get("name"), str) else ""
        if transport_name and source_state and dest_state:
            match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", transport_name, flags=re.IGNORECASE)
            if match:
                name_src = match.group(1).strip()
                name_dst = match.group(2).strip()
                if _normalize_name(name_src) and _normalize_name(name_dst):
                    if _normalize_name(name_src) != _normalize_name(source_state) or _normalize_name(name_dst) != _normalize_name(dest_state):
                        issues["warnings"].append(
                            {
                                "path": _join_pointer(["processes", "transports", idx, "name"]),
                                "reason": "Transport name source/destination disagrees with transport states.",
                                "evidence": json.dumps(
                                    {
                                        "name": transport_name,
                                        "from_biological_state": source_state,
                                        "to_biological_state": dest_state,
                                    },
                                    ensure_ascii=False,
                                )[:260],
                                "source": "deterministic",
                            }
                        )
                        patched_name = re.sub(
                            r"\bfrom\s+(.+?)\s+to\s+(.+)$",
                            f"from {source_state} to {dest_state}",
                            transport_name,
                            flags=re.IGNORECASE,
                        )
                        if patched_name != transport_name:
                            patch_ops.append(
                                {
                                    "op": "replace",
                                    "path": _join_pointer(["processes", "transports", idx, "name"]),
                                    "value": patched_name,
                                    "reason": "Normalize transport name to match declared source/destination states.",
                                    "confidence": 0.93,
                                    "evidence": transport_name[:220],
                                    "source": "deterministic",
                                }
                            )

    if not isinstance(entities.get("compounds"), list):
        patch_ops.append(
            {
                "op": "add",
                "path": _join_pointer(["entities", "compounds"]),
                "value": [],
                "reason": "Create missing entities.compounds list for process-linked compounds.",
                "confidence": 0.99,
                "evidence": "entities.compounds missing or not a list",
                "source": "deterministic",
            }
        )
    if not isinstance(entities.get("proteins"), list):
        patch_ops.append(
            {
                "op": "add",
                "path": _join_pointer(["entities", "proteins"]),
                "value": [],
                "reason": "Create missing entities.proteins list for process-linked proteins.",
                "confidence": 0.99,
                "evidence": "entities.proteins missing or not a list",
                "source": "deterministic",
            }
        )

    for name in _dedupe_preserve_order(process_compound_refs):
        norm_name = _normalize_name(name)
        if not norm_name:
            continue
        if norm_name in compound_norms or norm_name in protein_norms:
            continue
        patch_ops.append(
            {
                "op": "add",
                "path": _join_pointer(["entities", "compounds", "-"]),
                "value": {"name": name},
                "reason": "Add missing compound referenced by reaction/transport process.",
                "confidence": 0.99,
                "evidence": name,
                "source": "deterministic",
            }
        )
        issues["warnings"].append(
            {
                "path": _join_pointer(["entities", "compounds"]),
                "reason": f"Added missing compound '{name}' referenced in process definitions.",
                "evidence": name,
                "source": "deterministic",
            }
        )
        compound_norms.add(norm_name)

    for name in _dedupe_preserve_order(process_protein_refs):
        norm_name = _normalize_name(name)
        if not norm_name:
            continue
        if norm_name in protein_norms:
            continue
        patch_ops.append(
            {
                "op": "add",
                "path": _join_pointer(["entities", "proteins", "-"]),
                "value": {"name": name},
                "reason": "Add missing protein referenced by enzyme/transporter relation.",
                "confidence": 0.99,
                "evidence": name,
                "source": "deterministic",
            }
        )
        issues["warnings"].append(
            {
                "path": _join_pointer(["entities", "proteins"]),
                "reason": f"Added missing protein '{name}' referenced in process definitions.",
                "evidence": name,
                "source": "deterministic",
            }
        )
        protein_norms.add(norm_name)

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

    for idx in sorted(set(reaction_remove_indices), reverse=True):
        if locked_reaction_indices is not None and idx in locked_reaction_indices:
            # Locked reactions are protected; the warning was already emitted above.
            continue
        patch_ops.append(
            {
                "op": "remove",
                "path": _join_pointer(["processes", "reactions", idx]),
                "reason": "Remove degenerate reaction with identical inputs and outputs.",
                "confidence": 0.995,
                "evidence": f"processes.reactions[{idx}] has identical input/output multiset",
                "source": "deterministic",
            }
        )

    return issues, patch_ops


def _build_llm_prompt(
    payload: Dict[str, Any],
    *,
    context_note: str = "",
    retrieval_context: str = "",
    locked_reaction_indices: Optional[set] = None,
    locked_manifest: Optional[List[Dict[str, Any]]] = None,
) -> str:
    payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
    note = (context_note or "").strip()
    references = (retrieval_context or "").strip()

    lock_section = ""
    if locked_reaction_indices and locked_manifest:
        locked_ids = []
        for entry in locked_manifest:
            lid = str(entry.get("locked_reaction_id", "")).strip()
            if lid:
                locked_ids.append(lid)
        sorted_indices = sorted(locked_reaction_indices)
        lock_section = "\n".join([
            "LOCKED REACTIONS:",
            f"Total locked reactions: {len(locked_manifest)}",
            f"Locked reaction IDs: {', '.join(locked_ids[:50])}{'...' if len(locked_ids) > 50 else ''}",
            f"Locked reaction indices in processes.reactions[]: {sorted_indices}",
            "Do NOT propose remove patches for any of these indices.",
            "Do NOT propose patches that merge, split, or replace these reactions.",
            "If a locked reaction has issues, flag for quarantine instead of deletion.",
        ])

    return "\n".join(
        [
            "Audit this pathway JSON for SBML conversion readiness.",
            "Return issues and a minimal patch list.",
            "Use RFC6901 json pointers for paths.",
            "Use action add/replace/remove and include confidence 0..1.",
            "repair_rationale must be short and concrete.",
            f"Retry context: {note if note else 'none'}",
            lock_section if lock_section else "",
            "Reference motifs:",
            "<<<",
            references if references else "none",
            ">>>",
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
    llm_max_tokens: int = 3200,
    context_note: str = "",
    retrieval_context: str = "",
    locked_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    # --- Locked manifest discovery and loading ---
    effective_manifest_path = locked_manifest_path
    if effective_manifest_path is None:
        effective_manifest_path = _discover_locked_manifest_path(input_path)
    locked_manifest = _load_locked_manifest(effective_manifest_path)
    locked_indices: Optional[set] = None
    if locked_manifest is not None:
        locked_indices = _locked_reaction_indices(payload, locked_manifest)
        logger.info(
            "Locked manifest loaded: %d entries, %d matched reaction indices",
            len(locked_manifest),
            len(locked_indices),
        )

    deterministic_issues, deterministic_patch = _deterministic_audit(
        payload, locked_reaction_indices=locked_indices
    )
    llm_raw = ""
    llm_error = ""
    llm_payload: Optional[Dict[str, Any]] = None

    if use_llm:
        # Build lock-aware system prompt when manifest is available
        effective_system_prompt = SYSTEM_PROMPT
        if locked_manifest is not None:
            effective_system_prompt = SYSTEM_PROMPT + LOCK_AWARE_SYSTEM_PROMPT_FRAGMENT

        try:
            llm_raw = chat(
                [
                    {"role": "system", "content": effective_system_prompt},
                    {
                        "role": "user",
                        "content": _build_llm_prompt(
                            payload,
                            context_note=context_note,
                            retrieval_context=retrieval_context,
                            locked_reaction_indices=locked_indices,
                            locked_manifest=locked_manifest,
                        ),
                    },
                ],
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                response_json=True,
                model_env_var="OPENROUTER_AUDIT_MODEL",
                stage_name="audit loop",
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
            "repair_rationale": str(_safe_dict(llm_payload).get("repair_rationale", "")) if isinstance(llm_payload, dict) else "",
            "context_note": context_note,
            "retrieval_context_present": bool((retrieval_context or "").strip()),
            "retrieval_context_chars": len((retrieval_context or "").strip()),
            "temperature": float(llm_temperature),
            "max_tokens": int(llm_max_tokens),
        },
        "lock_policy": {
            "enabled": locked_manifest is not None,
            "locked_reaction_count": len(locked_manifest) if locked_manifest is not None else 0,
            "locked_reaction_indices_matched": len(locked_indices) if locked_indices is not None else 0,
            "manifest_path": str(effective_manifest_path) if effective_manifest_path is not None else "",
        },
    }

    # Compute and attach preservation scoring if manifest available
    if locked_manifest is not None:
        candidate_score = score_audit_candidate(payload, patch_ops, locked_manifest)
        audit_report["preservation_score"] = candidate_score

    audit_report_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_patch_path.write_text(json.dumps(patch_ops, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_report


def score_audit_candidate(
    payload: Dict[str, Any],
    patch_ops: List[Dict[str, Any]],
    locked_manifest: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Score an audit candidate for locked-reaction preservation.

    Scoring rules:
    - -10000 if candidate deletes any locked reaction
    - -5000  if candidate replaces full reactions array
    - -1000  per missing locked reaction
    - -250   per destructive input/output change on a locked reaction
    - +100   if all locked reactions preserved
    - Then subtract schema_error_count from the total

    Returns a dict with detailed scoring breakdown. When no locked manifest
    is available, returns a neutral score of 0 with all counters zeroed.
    """
    if locked_manifest is None or not locked_manifest:
        return {
            "schema_error_count": 0,
            "locked_reactions_preserved": 0,
            "locked_reactions_missing": 0,
            "locked_reactions_changed": 0,
            "quarantined_locked_reactions": 0,
            "reaction_preservation_score": 0,
            "final_candidate_score": 0,
        }

    locked_indices = _locked_reaction_indices(payload, locked_manifest)

    # Simulate patch application to detect deletions and changes
    deletes_locked = False
    replaces_reactions_array = False
    destructive_io_changes = 0
    schema_error_count = 0
    quarantined_count = 0

    for op in patch_ops:
        if not isinstance(op, dict):
            continue
        action = str(op.get("op", op.get("action", ""))).lower()
        path = str(op.get("path", op.get("json_pointer", "")))

        # Check for full reactions array replacement
        if path in ("/processes/reactions", "/reactions") and action == "replace":
            replaces_reactions_array = True

        # Check for locked reaction deletion
        if action == "remove" and re.match(r"^/processes/reactions/(\d+)$", path):
            match = re.match(r"^/processes/reactions/(\d+)$", path)
            if match:
                idx = int(match.group(1))
                if idx in locked_indices:
                    deletes_locked = True

        # Check for destructive input/output changes on locked reactions
        if action in ("remove", "replace"):
            io_match = re.match(r"^/processes/reactions/(\d+)/(inputs|outputs)$", path)
            if io_match:
                idx = int(io_match.group(1))
                if idx in locked_indices:
                    if action == "remove":
                        destructive_io_changes += 1
                    elif action == "replace":
                        # A replace of the entire inputs/outputs array is destructive
                        destructive_io_changes += 1

    # Count schema errors from the deterministic audit issues
    # We re-check the payload structure for common schema issues
    processes = payload.get("processes") if isinstance(payload.get("processes"), dict) else {}
    reactions = processes.get("reactions") if isinstance(processes.get("reactions"), list) else []
    for reaction in reactions:
        if not isinstance(reaction, dict):
            schema_error_count += 1
            continue
        if not reaction.get("inputs") or not isinstance(reaction.get("inputs"), list):
            schema_error_count += 1
        if not reaction.get("outputs") or not isinstance(reaction.get("outputs"), list):
            schema_error_count += 1

    # Count preserved, missing, and changed locked reactions
    # Build the set of reaction indices that would survive after patch application
    removed_indices: set = set()
    for op in patch_ops:
        if not isinstance(op, dict):
            continue
        action = str(op.get("op", op.get("action", ""))).lower()
        path = str(op.get("path", op.get("json_pointer", "")))
        if action == "remove":
            match = re.match(r"^/processes/reactions/(\d+)$", path)
            if match:
                removed_indices.add(int(match.group(1)))

    preserved_count = 0
    missing_count = 0
    changed_count = 0
    for idx in locked_indices:
        if idx in removed_indices:
            missing_count += 1
        else:
            preserved_count += 1

    # Check for locked reactions that had IO changes proposed
    io_changed_indices: set = set()
    for op in patch_ops:
        if not isinstance(op, dict):
            continue
        action = str(op.get("op", op.get("action", ""))).lower()
        path = str(op.get("path", op.get("json_pointer", "")))
        if action in ("remove", "replace"):
            io_match = re.match(r"^/processes/reactions/(\d+)/(inputs|outputs)", path)
            if io_match:
                idx = int(io_match.group(1))
                if idx in locked_indices and idx not in removed_indices:
                    io_changed_indices.add(idx)
    changed_count = len(io_changed_indices)

    # Compute score
    score = 0
    if deletes_locked:
        score -= 10000
    if replaces_reactions_array:
        score -= 5000
    score -= 1000 * missing_count
    score -= 250 * destructive_io_changes
    if missing_count == 0 and len(locked_indices) > 0 and not replaces_reactions_array:
        score += 100
    final_score = score - schema_error_count

    return {
        "schema_error_count": schema_error_count,
        "locked_reactions_preserved": preserved_count,
        "locked_reactions_missing": missing_count,
        "locked_reactions_changed": changed_count,
        "quarantined_locked_reactions": quarantined_count,
        "reaction_preservation_score": score,
        "final_candidate_score": final_score,
    }


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
    parser.add_argument("--max-tokens", type=int, default=3200, help="LLM max tokens for audit call")
    parser.add_argument(
        "--locked-manifest",
        dest="locked_manifest_path",
        default=None,
        help="Optional locked_reaction_manifest.json path. Auto-discovered beside input when omitted.",
    )
    args = parser.parse_args()

    run_audit(
        Path(args.input_path),
        Path(args.audit_report_path),
        Path(args.audit_patch_path),
        use_llm=not args.no_llm,
        llm_temperature=float(args.temperature),
        llm_max_tokens=int(args.max_tokens),
        locked_manifest_path=Path(args.locked_manifest_path) if args.locked_manifest_path else None,
    )
    print(f"Wrote audit report: {args.audit_report_path}")
    print(f"Wrote audit patch: {args.audit_patch_path}")


if __name__ == "__main__":
    main()
