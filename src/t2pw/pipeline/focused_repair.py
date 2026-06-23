"""
Focused repair passes for the T2PW pipeline.

These run AFTER Stage 2 inference and BEFORE filter_unresolvable_reactions().
They use the locked reaction manifest to identify what needs repair and
return small additive patches instead of asking the model to rewrite everything.

Pipeline order:
  Stage 1 extraction -> locked manifest -> Stage 2 inference ->
  focused repair passes (this module) -> filter_unresolvable_reactions ->
  audit -> patch application -> final export
"""

from __future__ import annotations

import json
import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from t2pw.curation.apply_audit_patch import apply_patch_with_policy

logger = logging.getLogger(__name__)

# ── All passes share these lock-safety instructions ──────────────────────────
_LOCK_SAFETY_INSTRUCTIONS = """\
The reaction set is locked.
Do not add/delete/merge/split/reorder reactions.
Do not rewrite the full pathway JSON.
Return patches only.
Every patch must include a reason.
Evidence is required when adding biological claims.
If uncertain, return a warning instead of changing the pathway."""

# ── Standard cofactors (compounds, NOT enzymes) ─────────────────────────────
_STANDARD_COFACTORS: frozenset[str] = frozenset(
    name.strip().casefold()
    for name in (
        "ATP", "ADP", "AMP", "GTP", "GDP", "GMP",
        "NAD+", "NAD", "NADH",
        "NADP+", "NADP", "NADPH",
        "FAD", "FADH2",
        "FMN", "FMNH2",
        "CoA", "acetyl-CoA", "coenzyme A",
        "PLP", "pyridoxal phosphate",
        "TPP", "thiamine pyrophosphate",
        "biotin",
        "tetrahydrofolate", "THF",
        "SAM", "S-adenosylmethionine",
        "UDP", "UTP", "CTP", "CDP",
        "Pi", "PPi",
        "H2O", "water",
        "CO2", "carbon dioxide",
        "O2", "oxygen",
        "H+", "proton",
        "NH3", "ammonia",
    )
)

# ── Ordered list of all pass names ───────────────────────────────────────────
ALL_PASS_NAMES: List[str] = [
    "missing_entity_repair",
    "modifier_enzyme_repair",
    "cofactor_repair",
    "compartment_state_repair",
    "alias_repair",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Lower-case, collapse whitespace, strip non-alphanumeric."""
    lowered = re.sub(r"\s+", " ", name.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _collect_declared_entity_names(pathway: Dict[str, Any]) -> Dict[str, set[str]]:
    """Return {bucket: set_of_normalised_names} from entities."""
    entities = _safe_dict(pathway.get("entities"))
    result: Dict[str, set[str]] = {}
    for bucket in ("compounds", "proteins", "protein_complexes",
                    "nucleic_acids", "element_collections"):
        names: set[str] = set()
        for item in _safe_list(entities.get(bucket)):
            if isinstance(item, dict):
                raw = item.get("name")
                if isinstance(raw, str) and raw.strip():
                    names.add(_normalize_name(raw))
        result[bucket] = names
    return result


def _all_entity_names_flat(declared: Dict[str, set[str]]) -> set[str]:
    return set().union(*declared.values())


def _reaction_participant_names(reaction: Dict[str, Any], key: str) -> List[str]:
    """Extract participant names from a reaction field (inputs, outputs)."""
    raw = reaction.get(key)
    items = raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) and raw.strip() else [])
    out: List[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            for k in ("name", "entity"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
                    break
    return out


def _reaction_modifier_names(reaction: Dict[str, Any]) -> List[str]:
    """Extract enzyme/modifier names from a reaction."""
    names: List[str] = []
    for key in ("enzymes", "modifiers"):
        for item in _safe_list(reaction.get(key)):
            if isinstance(item, dict):
                for field in ("entity", "protein", "protein_complex", "name"):
                    v = item.get(field)
                    if isinstance(v, str) and v.strip():
                        names.append(v.strip())
                        break
    return names


def _parse_llm_patches(raw_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse LLM response expecting {patches: [...], warnings: [...]}."""
    patches: List[Dict[str, Any]] = []
    warnings: List[str] = []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON object from markdown fences or surrounding text
        text = raw_text.strip().replace("```json", "```").replace("```", "")
        obj_start = text.find("{")
        if obj_start == -1:
            return [], [f"Failed to parse LLM response as JSON: {raw_text[:200]}"]
        try:
            parsed = json.loads(text[obj_start:])
        except json.JSONDecodeError:
            return [], [f"Failed to parse LLM response as JSON: {raw_text[:200]}"]

    if isinstance(parsed, dict):
        raw_patches = parsed.get("patches")
        if isinstance(raw_patches, list):
            patches = [p for p in raw_patches if isinstance(p, dict)]
        raw_warnings = parsed.get("warnings")
        if isinstance(raw_warnings, list):
            warnings = [str(w) for w in raw_warnings if w]
    return patches, warnings


def _call_repair_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4000,
) -> str:
    """Call the LLM using the inference model for repair passes."""
    from t2pw.llm.client import chat

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_json=True,
        model_env_var="OPENROUTER_INFERENCE_MODEL",
        stage_name="focused_repair",
    )


def _apply_pass_patches(
    pathway: Dict[str, Any],
    patches: List[Dict[str, Any]],
    locked_manifest: List[Dict[str, Any]],
    *,
    stage: str,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], int, int]:
    """Apply patches through the lock-safe policy engine.

    Returns (updated_pathway, applied_count, rejected_count).
    """
    if not patches:
        return pathway, 0, 0

    applied_log_path = None
    rejected_log_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        applied_log_path = output_dir / "applied_patch_log.json"
        rejected_log_path = output_dir / "rejected_patch_log.json"

    updated, report = apply_patch_with_policy(
        pathway,
        patches,
        locked_manifest=locked_manifest,
        stage=stage,
        applied_log_path=applied_log_path,
        rejected_log_path=rejected_log_path,
    )
    summary = report.get("summary", {})
    return updated, summary.get("accepted_count", 0), summary.get("rejected_count", 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1: Missing Entity Repair
# ═══════════════════════════════════════════════════════════════════════════════

def _run_missing_entity_repair(
    pathway: Dict[str, Any],
    locked_manifest: List[Dict[str, Any]],
    *,
    source_text: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pass 1: Find reaction inputs/outputs/modifiers that lack matching declared entities.

    Deterministic for clear cases. Only calls LLM for ambiguous entity type classification.
    """
    t0 = time.monotonic()
    declared = _collect_declared_entity_names(pathway)
    all_names = _all_entity_names_flat(declared)

    reactions = _safe_list(
        _safe_dict(pathway.get("processes")).get("reactions")
    )

    # Collect missing entity names and the reactions that reference them
    missing_entities: Dict[str, List[str]] = {}  # normalized -> [reaction context strings]
    missing_original_names: Dict[str, str] = {}  # normalized -> first original-case name

    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        rxn_name = reaction.get("name", "<unnamed>")
        lock_id = reaction.get("locked_reaction_id", "")
        context_label = f"{rxn_name} ({lock_id})" if lock_id else rxn_name

        for key in ("inputs", "outputs"):
            for name in _reaction_participant_names(reaction, key):
                norm = _normalize_name(name)
                if norm and norm not in all_names:
                    missing_entities.setdefault(norm, []).append(
                        f"{key} of {context_label}"
                    )
                    if norm not in missing_original_names:
                        missing_original_names[norm] = name

        for name in _reaction_modifier_names(reaction):
            norm = _normalize_name(name)
            if norm and norm not in all_names:
                missing_entities.setdefault(norm, []).append(
                    f"modifier/enzyme in {context_label}"
                )
                if norm not in missing_original_names:
                    missing_original_names[norm] = name

    if not missing_entities:
        duration = time.monotonic() - t0
        return {
            "pass": "missing_entity_repair",
            "patches": [],
            "warnings": [],
            "stats": {"missing_found": 0, "repaired": 0, "ambiguous": 0},
            "duration_seconds": round(duration, 2),
        }

    # Deterministic: clearly compound-like names (cofactors, small molecules) go
    # directly as compounds. Proteins/complexes that look ambiguous go to LLM.
    deterministic_patches: List[Dict[str, Any]] = []
    ambiguous_names: List[Tuple[str, str, List[str]]] = []  # (norm, original, contexts)

    for norm, contexts in missing_entities.items():
        original = missing_original_names[norm]
        # Check if it's a known cofactor or standard small molecule
        if norm in _STANDARD_COFACTORS:
            deterministic_patches.append({
                "op": "add",
                "path": "/entities/compounds/-",
                "value": {"name": original},
                "reason": f"{original} is listed as {contexts[0]} but not declared in entities",
                "confidence": 0.95,
            })
        else:
            # Heuristic: if the name is short, all-caps, or contains common
            # compound patterns, treat as compound deterministically
            is_likely_compound = (
                len(original) <= 6
                or original.isupper()
                or re.match(r"^[A-Z0-9+\-()]+$", original)
            )
            if is_likely_compound:
                deterministic_patches.append({
                    "op": "add",
                    "path": "/entities/compounds/-",
                    "value": {"name": original},
                    "reason": f"{original} is listed as {contexts[0]} but not declared in entities",
                    "confidence": 0.90,
                })
            else:
                ambiguous_names.append((norm, original, contexts))

    # For ambiguous names, ask the LLM to classify
    llm_patches: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if ambiguous_names:
        entity_list_text = "\n".join(
            f"- {orig} (appears as: {'; '.join(ctxs[:3])})"
            for _, orig, ctxs in ambiguous_names
        )
        system_prompt = (
            "You are a bioinformatics assistant that classifies biological entities.\n"
            "Return a JSON object with keys 'patches' (list) and 'warnings' (list).\n\n"
            f"{_LOCK_SAFETY_INSTRUCTIONS}"
        )
        user_prompt = (
            "Given these entity names that appear in reactions but are not declared "
            "in the entity registry:\n\n"
            f"{entity_list_text}\n\n"
            "For each, determine:\n"
            "1. Is this a compound, protein, protein_complex, or nucleic_acid?\n"
            "2. What should its name be (prefer the exact name used in the reaction)?\n\n"
            "Return JSON with patches to add these entities. Each patch must have:\n"
            '- "op": "add"\n'
            '- "path": "/entities/<bucket>/-" where bucket is compounds, proteins, '
            "protein_complexes, or nucleic_acids\n"
            '- "value": {"name": "<entity_name>"}\n'
            '- "reason": why this entity type was chosen\n'
            '- "confidence": float 0-1\n\n'
            "If uncertain about any entity, add it to warnings instead of patches."
        )
        try:
            raw_response = _call_repair_llm(
                system_prompt, user_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
            llm_patches, llm_warnings = _parse_llm_patches(raw_response)
            warnings.extend(llm_warnings)
        except Exception as exc:
            logger.error("Missing entity repair LLM call failed: %s", exc)
            warnings.append(f"LLM classification failed: {exc}")

    all_patches = deterministic_patches + llm_patches

    # Apply through lock-safe policy
    updated_pathway, applied, rejected = _apply_pass_patches(
        pathway, all_patches, locked_manifest,
        stage="focused_repair_missing_entity",
        output_dir=output_dir,
    )

    duration = time.monotonic() - t0
    return {
        "pass": "missing_entity_repair",
        "patches": all_patches,
        "applied_pathway": updated_pathway,
        "warnings": warnings,
        "stats": {
            "missing_found": len(missing_entities),
            "repaired": applied,
            "rejected": rejected,
            "ambiguous": len(ambiguous_names),
            "deterministic": len(deterministic_patches),
        },
        "duration_seconds": round(duration, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2: Modifier/Enzyme Repair
# ═══════════════════════════════════════════════════════════════════════════════

def _run_modifier_enzyme_repair(
    pathway: Dict[str, Any],
    locked_manifest: List[Dict[str, Any]],
    *,
    source_text: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pass 2: Add catalyst/enzyme links for reactions missing them."""
    t0 = time.monotonic()
    declared = _collect_declared_entity_names(pathway)
    protein_names = declared.get("proteins", set())
    complex_names = declared.get("protein_complexes", set())

    if not protein_names and not complex_names:
        duration = time.monotonic() - t0
        return {
            "pass": "modifier_enzyme_repair",
            "patches": [],
            "warnings": ["No proteins or protein_complexes declared; skipping enzyme repair"],
            "stats": {"reactions_checked": 0, "patches_proposed": 0, "applied": 0, "rejected": 0},
            "duration_seconds": round(duration, 2),
        }

    reactions = _safe_list(
        _safe_dict(pathway.get("processes")).get("reactions")
    )

    # Find reactions that have no enzyme/modifier links
    reactions_needing_enzymes: List[Dict[str, Any]] = []
    for idx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        existing_modifiers = _reaction_modifier_names(reaction)
        if existing_modifiers:
            continue  # Already has enzyme links
        rxn_summary = {
            "index": idx,
            "name": reaction.get("name", "<unnamed>"),
            "locked_reaction_id": reaction.get("locked_reaction_id", ""),
            "inputs": _reaction_participant_names(reaction, "inputs"),
            "outputs": _reaction_participant_names(reaction, "outputs"),
            "evidence": (reaction.get("evidence") or "")[:300],
        }
        reactions_needing_enzymes.append(rxn_summary)

    if not reactions_needing_enzymes:
        duration = time.monotonic() - t0
        return {
            "pass": "modifier_enzyme_repair",
            "patches": [],
            "warnings": [],
            "stats": {"reactions_checked": len(reactions), "patches_proposed": 0, "applied": 0, "rejected": 0},
            "duration_seconds": round(duration, 2),
        }

    # Build entity lists for the prompt
    entities = _safe_dict(pathway.get("entities"))
    protein_list = [
        item.get("name") for item in _safe_list(entities.get("proteins"))
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    complex_list = [
        item.get("name") for item in _safe_list(entities.get("protein_complexes"))
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]

    # Batch reactions (max 15 per LLM call to keep context small)
    batch_size = 15
    all_patches: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for batch_start in range(0, len(reactions_needing_enzymes), batch_size):
        batch = reactions_needing_enzymes[batch_start:batch_start + batch_size]
        batch_text = json.dumps(batch, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are a bioinformatics assistant that identifies enzyme-reaction links.\n"
            "Return a JSON object with keys 'patches' (list) and 'warnings' (list).\n\n"
            f"{_LOCK_SAFETY_INSTRUCTIONS}"
        )
        user_prompt = (
            "For each reaction below, determine if it should have an enzyme/catalyst link.\n"
            "Only use proteins or protein_complexes that are already declared in the entity registry.\n"
            "Do NOT use compounds/cofactors as enzymes.\n"
            f"Cofactors like {', '.join(sorted(list(_STANDARD_COFACTORS))[:15])} are compounds, NOT enzymes.\n\n"
            f"Reactions without enzyme links:\n{batch_text}\n\n"
            f"Declared proteins: {json.dumps(protein_list)}\n"
            f"Declared protein_complexes: {json.dumps(complex_list)}\n\n"
            "Return patches adding enzyme links. Each patch should:\n"
            '- "op": "add"\n'
            '- "path": "/processes/reactions/<index>/enzymes/-" where <index> is the reaction index\n'
            '- "value": {"protein": "<name>"} or {"protein_complex": "<name>"}\n'
            '- "reason": evidence-based justification\n'
            '- "confidence": float 0-1\n\n'
            "Do not modify reaction inputs/outputs. If uncertain, add to warnings."
        )
        try:
            raw = _call_repair_llm(
                system_prompt, user_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
            batch_patches, batch_warnings = _parse_llm_patches(raw)
            all_patches.extend(batch_patches)
            warnings.extend(batch_warnings)
        except Exception as exc:
            logger.error("Modifier enzyme repair LLM call failed: %s", exc)
            warnings.append(f"LLM call failed for batch starting at {batch_start}: {exc}")

    updated_pathway, applied, rejected = _apply_pass_patches(
        pathway, all_patches, locked_manifest,
        stage="focused_repair_modifier_enzyme",
        output_dir=output_dir,
    )

    duration = time.monotonic() - t0
    return {
        "pass": "modifier_enzyme_repair",
        "patches": all_patches,
        "applied_pathway": updated_pathway,
        "warnings": warnings,
        "stats": {
            "reactions_checked": len(reactions),
            "patches_proposed": len(all_patches),
            "applied": applied,
            "rejected": rejected,
        },
        "duration_seconds": round(duration, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 3: Cofactor Repair
# ═══════════════════════════════════════════════════════════════════════════════

def _run_cofactor_repair(
    pathway: Dict[str, Any],
    locked_manifest: List[Dict[str, Any]],
    *,
    source_text: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pass 3: Add missing standard cofactors when explicitly stated or biologically standard."""
    t0 = time.monotonic()
    declared = _collect_declared_entity_names(pathway)
    compound_names = declared.get("compounds", set())

    reactions = _safe_list(
        _safe_dict(pathway.get("processes")).get("reactions")
    )

    # Build reaction summaries for the LLM
    reaction_summaries: List[Dict[str, Any]] = []
    for idx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        reaction_summaries.append({
            "index": idx,
            "name": reaction.get("name", "<unnamed>"),
            "locked_reaction_id": reaction.get("locked_reaction_id", ""),
            "inputs": _reaction_participant_names(reaction, "inputs"),
            "outputs": _reaction_participant_names(reaction, "outputs"),
            "evidence": (reaction.get("evidence") or "")[:300],
        })

    if not reaction_summaries:
        duration = time.monotonic() - t0
        return {
            "pass": "cofactor_repair",
            "patches": [],
            "warnings": [],
            "stats": {"reactions_checked": 0, "patches_proposed": 0, "applied": 0, "rejected": 0},
            "duration_seconds": round(duration, 2),
        }

    declared_compounds = [
        item.get("name") for item in _safe_list(
            _safe_dict(pathway.get("entities")).get("compounds")
        ) if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]

    batch_size = 15
    all_patches: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for batch_start in range(0, len(reaction_summaries), batch_size):
        batch = reaction_summaries[batch_start:batch_start + batch_size]
        batch_text = json.dumps(batch, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are a bioinformatics assistant that identifies missing cofactors.\n"
            "Return a JSON object with keys 'patches' (list) and 'warnings' (list).\n\n"
            f"{_LOCK_SAFETY_INSTRUCTIONS}"
        )
        user_prompt = (
            "For each reaction below, identify missing standard cofactors.\n"
            "Only add cofactors that are:\n"
            "- Explicitly mentioned in the evidence text, OR\n"
            "- Biologically standard for this reaction type with high confidence\n\n"
            "Do NOT remove existing compounds. Only ADD missing ones.\n\n"
            f"Reactions:\n{batch_text}\n\n"
            f"Already declared compounds: {json.dumps(declared_compounds)}\n\n"
            "For missing cofactors, return two types of patches:\n"
            "1. Entity declarations (if the cofactor is not yet in entities/compounds):\n"
            '   {"op": "add", "path": "/entities/compounds/-", "value": {"name": "<cofactor>"}, '
            '"reason": "...", "confidence": 0.9}\n'
            "2. Reaction input/output additions (if the cofactor belongs in a reaction):\n"
            '   {"op": "add", "path": "/processes/reactions/<idx>/inputs/-", "value": "<cofactor>", '
            '"reason": "...", "evidence": "...", "confidence": 0.9}\n\n'
            "Include confidence and evidence for every patch. If uncertain, add to warnings."
        )
        try:
            raw = _call_repair_llm(
                system_prompt, user_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
            batch_patches, batch_warnings = _parse_llm_patches(raw)
            all_patches.extend(batch_patches)
            warnings.extend(batch_warnings)
        except Exception as exc:
            logger.error("Cofactor repair LLM call failed: %s", exc)
            warnings.append(f"LLM call failed for batch starting at {batch_start}: {exc}")

    updated_pathway, applied, rejected = _apply_pass_patches(
        pathway, all_patches, locked_manifest,
        stage="focused_repair_cofactor",
        output_dir=output_dir,
    )

    duration = time.monotonic() - t0
    return {
        "pass": "cofactor_repair",
        "patches": all_patches,
        "applied_pathway": updated_pathway,
        "warnings": warnings,
        "stats": {
            "reactions_checked": len(reaction_summaries),
            "patches_proposed": len(all_patches),
            "applied": applied,
            "rejected": rejected,
        },
        "duration_seconds": round(duration, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 4: Compartment/State Repair
# ═══════════════════════════════════════════════════════════════════════════════

def _run_compartment_state_repair(
    pathway: Dict[str, Any],
    locked_manifest: List[Dict[str, Any]],
    *,
    source_text: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pass 4: Fill missing biological state/compartment fields."""
    t0 = time.monotonic()

    # Collect existing biological states
    existing_states = _safe_list(pathway.get("biological_states"))
    state_names = {
        (s.get("name") or "").strip().casefold()
        for s in existing_states
        if isinstance(s, dict) and isinstance(s.get("name"), str)
    }

    # Check for global organism context
    entities = _safe_dict(pathway.get("entities"))
    species_list = _safe_list(entities.get("species"))
    global_organism = ""
    if species_list:
        first = species_list[0] if species_list else None
        if isinstance(first, dict):
            global_organism = (first.get("name") or "").strip()

    # Collect reactions and entities missing compartment/state info
    reactions = _safe_list(
        _safe_dict(pathway.get("processes")).get("reactions")
    )
    reactions_missing_state: List[Dict[str, Any]] = []
    for idx, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        if reaction.get("biological_state"):
            continue  # Already has state
        reactions_missing_state.append({
            "index": idx,
            "name": reaction.get("name", "<unnamed>"),
            "locked_reaction_id": reaction.get("locked_reaction_id", ""),
            "evidence": (reaction.get("evidence") or "")[:300],
        })

    if not reactions_missing_state:
        duration = time.monotonic() - t0
        return {
            "pass": "compartment_state_repair",
            "patches": [],
            "warnings": [],
            "stats": {
                "reactions_checked": len(reactions),
                "missing_states_found": 0,
                "patches_proposed": 0,
                "applied": 0,
                "rejected": 0,
            },
            "duration_seconds": round(duration, 2),
        }

    batch_size = 20
    all_patches: List[Dict[str, Any]] = []
    warnings: List[str] = []

    source_excerpt = source_text[:3000] if source_text else ""

    for batch_start in range(0, len(reactions_missing_state), batch_size):
        batch = reactions_missing_state[batch_start:batch_start + batch_size]
        batch_text = json.dumps(batch, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are a bioinformatics assistant that assigns biological compartment/state information.\n"
            "Return a JSON object with keys 'patches' (list) and 'warnings' (list).\n\n"
            f"{_LOCK_SAFETY_INSTRUCTIONS}"
        )
        context_parts = [
            "For each reaction below, determine the correct biological_state (compartment).\n",
            "Rules:\n"
            "- Use explicit context from the source text\n"
            "- Do not invent species/cell/tissue\n",
        ]
        if global_organism:
            context_parts.append(
                f"- Global organism context: {global_organism}\n"
            )
        if state_names:
            context_parts.append(
                f"- Already declared biological states: {json.dumps(sorted(state_names))}\n"
            )
        context_parts.extend([
            f"\nReactions missing biological_state:\n{batch_text}\n\n",
        ])
        if source_excerpt:
            context_parts.append(
                f"Source text excerpt (for context):\n<<<\n{source_excerpt}\n>>>\n\n"
            )
        context_parts.append(
            "Return patches:\n"
            '- To set reaction state: {"op": "add", "path": "/processes/reactions/<idx>/biological_state", '
            '"value": "<state_name>", "reason": "...", "confidence": 0.8}\n'
            '- To add new biological states: {"op": "add", "path": "/biological_states/-", '
            '"value": {"name": "<state>", ...}, "reason": "...", "confidence": 0.8}\n\n'
            "If no clear evidence for compartment, add to warnings instead."
        )
        user_prompt = "".join(context_parts)

        try:
            raw = _call_repair_llm(
                system_prompt, user_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
            batch_patches, batch_warnings = _parse_llm_patches(raw)
            all_patches.extend(batch_patches)
            warnings.extend(batch_warnings)
        except Exception as exc:
            logger.error("Compartment state repair LLM call failed: %s", exc)
            warnings.append(f"LLM call failed for batch starting at {batch_start}: {exc}")

    updated_pathway, applied, rejected = _apply_pass_patches(
        pathway, all_patches, locked_manifest,
        stage="focused_repair_compartment_state",
        output_dir=output_dir,
    )

    duration = time.monotonic() - t0
    return {
        "pass": "compartment_state_repair",
        "patches": all_patches,
        "applied_pathway": updated_pathway,
        "warnings": warnings,
        "stats": {
            "reactions_checked": len(reactions),
            "missing_states_found": len(reactions_missing_state),
            "patches_proposed": len(all_patches),
            "applied": applied,
            "rejected": rejected,
        },
        "duration_seconds": round(duration, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 5: Alias Repair
# ═══════════════════════════════════════════════════════════════════════════════

def _run_alias_repair(
    pathway: Dict[str, Any],
    locked_manifest: List[Dict[str, Any]],
    *,
    source_text: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Pass 5: Add SAME_AS alias links between entities that refer to the same thing."""
    t0 = time.monotonic()
    declared = _collect_declared_entity_names(pathway)
    entities = _safe_dict(pathway.get("entities"))

    # Build a summary of all declared entity names by bucket
    entity_lists: Dict[str, List[str]] = {}
    for bucket in ("compounds", "proteins", "protein_complexes", "nucleic_acids"):
        names = [
            item.get("name") for item in _safe_list(entities.get(bucket))
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        if names:
            entity_lists[bucket] = names

    if not entity_lists:
        duration = time.monotonic() - t0
        return {
            "pass": "alias_repair",
            "patches": [],
            "warnings": [],
            "stats": {"entities_checked": 0, "patches_proposed": 0, "applied": 0, "rejected": 0},
            "duration_seconds": round(duration, 2),
        }

    # Check if there's already a same_as / aliases section
    existing_interactions = _safe_list(
        _safe_dict(pathway.get("processes")).get("interactions")
    )

    system_prompt = (
        "You are a bioinformatics assistant that identifies entity aliases.\n"
        "Return a JSON object with keys 'patches' (list) and 'warnings' (list).\n\n"
        f"{_LOCK_SAFETY_INSTRUCTIONS}"
    )
    user_prompt = (
        "Examine the entity names below and identify pairs that refer to the same "
        "biological entity (aliases/synonyms).\n\n"
        "Common examples:\n"
        "- GSH / glutathione\n"
        "- NAD+ / NAD / nicotinamide adenine dinucleotide\n"
        "- CoA / coenzyme A\n"
        "- Abbreviation / full name pairs\n\n"
        f"Declared entities:\n{json.dumps(entity_lists, indent=2)}\n\n"
        "Return patches to add SAME_AS interaction links:\n"
        '{"op": "add", "path": "/processes/interactions/-", '
        '"value": {"entity_1": "<name1>", "entity_2": "<name2>", '
        '"relationship": "SAME_AS", "evidence": "..."}, '
        '"reason": "...", "confidence": 0.9}\n\n'
        "Rules:\n"
        "- Do NOT merge entities destructively\n"
        "- Only link entities that are genuinely the same molecule/protein\n"
        "- Do not link entities that are merely related (e.g. NAD+ and NADH are different)\n"
        "- If uncertain, add to warnings\n"
    )
    all_patches: List[Dict[str, Any]] = []
    warnings: List[str] = []

    try:
        raw = _call_repair_llm(
            system_prompt, user_prompt,
            temperature=temperature, max_tokens=max_tokens,
        )
        all_patches, warnings = _parse_llm_patches(raw)
    except Exception as exc:
        logger.error("Alias repair LLM call failed: %s", exc)
        warnings.append(f"LLM call failed: {exc}")

    total_entities = sum(len(v) for v in entity_lists.values())

    updated_pathway, applied, rejected = _apply_pass_patches(
        pathway, all_patches, locked_manifest,
        stage="focused_repair_alias",
        output_dir=output_dir,
    )

    duration = time.monotonic() - t0
    return {
        "pass": "alias_repair",
        "patches": all_patches,
        "applied_pathway": updated_pathway,
        "warnings": warnings,
        "stats": {
            "entities_checked": total_entities,
            "patches_proposed": len(all_patches),
            "applied": applied,
            "rejected": rejected,
        },
        "duration_seconds": round(duration, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

_PASS_REGISTRY: Dict[str, Any] = {
    "missing_entity_repair": _run_missing_entity_repair,
    "modifier_enzyme_repair": _run_modifier_enzyme_repair,
    "cofactor_repair": _run_cofactor_repair,
    "compartment_state_repair": _run_compartment_state_repair,
    "alias_repair": _run_alias_repair,
}


def run_focused_repair_passes(
    pathway_json: Dict[str, Any],
    locked_manifest: List[Dict[str, Any]],
    *,
    source_text: str = "",
    output_dir: Optional[Path] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    enabled_passes: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run focused repair passes on the pathway JSON.

    Parameters
    ----------
    pathway_json : dict
        The merged Stage-1 + Stage-2 pathway JSON.
    locked_manifest : list[dict]
        The locked reaction manifest entries.
    source_text : str
        Original source text (paper) for context.
    output_dir : Path, optional
        Directory for repair artifacts.
    temperature : float
        LLM temperature.
    max_tokens : int
        LLM max tokens per call.
    enabled_passes : list[str], optional
        Which passes to run. Defaults to all. Each name must be one of
        ALL_PASS_NAMES.

    Returns
    -------
    (repaired_pathway, repair_report)
    """
    # If no locked manifest, skip all passes
    if not locked_manifest:
        logger.info("No locked manifest provided; skipping focused repair passes.")
        report: Dict[str, Any] = {
            "passes_run": [],
            "total_patches_proposed": 0,
            "total_patches_applied": 0,
            "total_patches_rejected": 0,
            "pass_details": [],
            "skipped_reason": "no_locked_manifest",
        }
        return deepcopy(pathway_json), report

    passes_to_run = enabled_passes if enabled_passes is not None else ALL_PASS_NAMES
    # Validate pass names
    valid_passes = [p for p in passes_to_run if p in _PASS_REGISTRY]
    invalid_passes = [p for p in passes_to_run if p not in _PASS_REGISTRY]
    if invalid_passes:
        logger.warning("Unknown repair passes requested (skipped): %s", invalid_passes)

    working_pathway = deepcopy(pathway_json)
    total_proposed = 0
    total_applied = 0
    total_rejected = 0
    pass_details: List[Dict[str, Any]] = []
    passes_run: List[str] = []

    for pass_name in valid_passes:
        pass_fn = _PASS_REGISTRY[pass_name]
        logger.info("Running focused repair pass: %s", pass_name)

        try:
            result = pass_fn(
                working_pathway,
                locked_manifest,
                source_text=source_text,
                temperature=temperature,
                max_tokens=max_tokens,
                output_dir=output_dir,
            )
        except Exception as exc:
            logger.error("Focused repair pass %s failed: %s", pass_name, exc)
            pass_details.append({
                "pass": pass_name,
                "patches_proposed": 0,
                "patches_applied": 0,
                "patches_rejected": 0,
                "warnings": [f"Pass failed: {exc}"],
                "duration_seconds": 0.0,
                "error": str(exc),
            })
            continue

        # Extract updated pathway if pass produced one
        if "applied_pathway" in result and result["applied_pathway"] is not None:
            working_pathway = result["applied_pathway"]

        patches = result.get("patches", [])
        stats = result.get("stats", {})
        applied = stats.get("repaired", stats.get("applied", 0))
        rejected = stats.get("rejected", 0)

        total_proposed += len(patches)
        total_applied += applied
        total_rejected += rejected
        passes_run.append(pass_name)

        pass_details.append({
            "pass": pass_name,
            "patches_proposed": len(patches),
            "patches_applied": applied,
            "patches_rejected": rejected,
            "warnings": result.get("warnings", []),
            "duration_seconds": result.get("duration_seconds", 0.0),
        })

        logger.info(
            "Pass %s: proposed=%d, applied=%d, rejected=%d",
            pass_name, len(patches), applied, rejected,
        )

    report = {
        "passes_run": passes_run,
        "total_patches_proposed": total_proposed,
        "total_patches_applied": total_applied,
        "total_patches_rejected": total_rejected,
        "pass_details": pass_details,
    }

    # Write report if output_dir is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "focused_repair_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Focused repair report saved to %s", report_path)

    return working_pathway, report
