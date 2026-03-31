"""
pre_export_validator.py

Deterministic validation of the merged pathway JSON before SBML export.
Raises ValidationError (blocking) or returns warnings (advisory) via ValidationResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

_VALID_ENTITY_CLASSES: Dict[str, Set[str]] = {
    "compounds": {"compound", "cofactor", "ion", "element_collection"},
    "proteins": {"protein", "transporter"},
    "protein_complexes": {"protein_complex"},
    "nucleic_acids": {"nucleic_acid"},
    "element_collections": {"element_collection", "compound"},
}

_VALID_REACTION_CLASSES: Set[str] = {
    "biochemical_reaction",
    "transport_reaction",
    "interaction",
    "subpathway",
    "activation",
    "inhibition",
    "binding",
}

_VALID_INTERACTION_CLASSES: Set[str] = {
    "activation",
    "inhibition",
    "binding",
    "interaction",
}


# ---------------------------------------------------------------------------
# Result + exception types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)    # block export
    warnings: List[str] = field(default_factory=list)  # allow export with notes


class ValidationError(Exception):
    """Raised by build_sbml() when pre-export validation fails."""

    def __init__(self, errors: List[str], warnings: List[str]) -> None:
        self.errors = errors
        self.warnings = warnings
        super().__init__(
            f"Pre-export validation failed with {len(errors)} error(s): "
            + "; ".join(errors[:3])
            + ("..." if len(errors) > 3 else "")
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_entity_name_sets(payload: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Return a mapping from entity-type key to the set of declared names."""
    entities = _safe_dict(payload.get("entities"))
    result: Dict[str, Set[str]] = {
        "compounds": set(),
        "proteins": set(),
        "protein_complexes": set(),
        "nucleic_acids": set(),
        "element_collections": set(),
    }
    for key in result:
        for item in _safe_list(entities.get(key)):
            name = _safe_dict(item).get("name", "")
            if isinstance(name, str) and name.strip():
                result[key].add(name.strip())
    return result


def _build_states_to_loc(payload: Dict[str, Any]) -> Dict[str, str]:
    """Map biological-state name -> compartment/location name."""
    states: Dict[str, str] = {}
    for state in _safe_list(payload.get("biological_states")):
        if not isinstance(state, dict):
            continue
        name = (state.get("name") or "").strip() if isinstance(state.get("name"), str) else ""
        loc = (state.get("compartment_canonical") or "").strip() if isinstance(state.get("compartment_canonical"), str) else ""
        if not loc:
            loc = (state.get("subcellular_location") or "").strip() if isinstance(state.get("subcellular_location"), str) else ""
        if name and loc:
            states[name] = loc
    return states


def _build_declared_compartments(payload: Dict[str, Any]) -> Set[str]:
    """Return all compartment names/canonical names from the compartments[] list."""
    declared: Set[str] = set()
    for item in _safe_list(payload.get("compartments")):
        if not isinstance(item, dict):
            continue
        canonical = (item.get("canonical_name") or "").strip() if isinstance(item.get("canonical_name"), str) else ""
        raw = (item.get("name") or "").strip() if isinstance(item.get("name"), str) else ""
        if canonical:
            declared.add(canonical)
        if raw:
            declared.add(raw)
    return declared


def _extract_modifier_names(modifiers_raw: List[Any]) -> List[str]:
    """Extract entity name strings from a mixed modifiers/enzymes list."""
    names: List[str] = []
    for mod in modifiers_raw:
        if isinstance(mod, dict):
            ent = (
                mod.get("entity")
                or mod.get("name")
                or mod.get("protein")
                or mod.get("protein_complex")
                or ""
            )
            ent = (ent or "").strip() if isinstance(ent, str) else ""
            if ent:
                names.append(ent)
        elif isinstance(mod, str) and mod.strip():
            names.append(mod.strip())
    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_graph(draft_graph: Any, payload: Dict[str, Any]) -> ValidationResult:
    """
    Validate the pathway payload before SBML export.

    Parameters
    ----------
    draft_graph:
        Optional DraftGraph instance (may be None); reserved for future graph-
        structural checks that require the intermediate representation.
    payload:
        The merged pathway JSON dict passed to build_sbml().

    Returns
    -------
    ValidationResult
        ``passed=True`` when no blocking errors were found.
        ``errors`` block export; ``warnings`` are advisory.
    """
    errors: List[str] = []
    warnings: List[str] = []

    name_sets = _build_entity_name_sets(payload)
    all_names: Set[str] = set()
    for s in name_sets.values():
        all_names.update(s)

    protein_names: Set[str] = name_sets["proteins"] | name_sets["protein_complexes"]
    entities_dict = _safe_dict(payload.get("entities"))
    processes = _safe_dict(payload.get("processes"))
    states_to_loc = _build_states_to_loc(payload)
    declared_compartments = _build_declared_compartments(payload)

    # ------------------------------------------------------------------
    # 1. Reaction checks
    # ------------------------------------------------------------------
    seen_reaction_keys: Set[Tuple[FrozenSet[str], FrozenSet[str]]] = set()
    seen_reaction_ids: Set[str] = set()

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        label = f"processes.reactions[{ridx}]"

        inputs = [
            x.strip()
            for x in _safe_list(reaction.get("inputs"))
            if isinstance(x, str) and x.strip()
        ]
        outputs = [
            x.strip()
            for x in _safe_list(reaction.get("outputs"))
            if isinstance(x, str) and x.strip()
        ]

        # Every referenced input/output entity must exist in the registry
        for name in inputs:
            if name not in all_names:
                errors.append(
                    f"{label}: input entity '{name}' is not in the entity registry."
                )
        for name in outputs:
            if name not in all_names:
                errors.append(
                    f"{label}: output entity '{name}' is not in the entity registry."
                )

        # No reaction with BOTH empty inputs AND empty outputs
        if not inputs and not outputs:
            errors.append(f"{label}: has both empty inputs and empty outputs.")

        # Modifier entities must be proteins or protein_complexes
        modifier_names = _extract_modifier_names(
            _safe_list(reaction.get("modifiers")) + _safe_list(reaction.get("enzymes"))
        )
        for name in modifier_names:
            if name not in protein_names:
                warnings.append(
                    f"{label}: modifier/enzyme '{name}' not found among proteins "
                    f"or protein_complexes."
                )

        # Explicit reaction_id uniqueness (if the field is present)
        rid = (reaction.get("reaction_id") or reaction.get("id") or "").strip()
        if isinstance(rid, str) and rid:
            if rid in seen_reaction_ids:
                errors.append(f"{label}: duplicate reaction id '{rid}'.")
            seen_reaction_ids.add(rid)

        # Semantic duplicate detection (same inputs+outputs multiset)
        key: Tuple[FrozenSet[str], FrozenSet[str]] = (
            frozenset(inputs),
            frozenset(outputs),
        )
        if (inputs or outputs) and key in seen_reaction_keys:
            warnings.append(
                f"{label}: apparent duplicate reaction — same inputs and outputs "
                f"as a previously seen reaction."
            )
        seen_reaction_keys.add(key)

    # ------------------------------------------------------------------
    # 2. Transport checks
    # ------------------------------------------------------------------
    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        label = f"processes.transports[{tidx}]"

        from_state = (
            (transport.get("from_biological_state") or "").strip()
            if isinstance(transport.get("from_biological_state"), str)
            else ""
        )
        to_state = (
            (transport.get("to_biological_state") or "").strip()
            if isinstance(transport.get("to_biological_state"), str)
            else ""
        )
        from_loc = states_to_loc.get(from_state, "") if from_state else ""
        to_loc = states_to_loc.get(to_state, "") if to_state else ""

        # Transport must cross compartment boundaries
        if from_loc and to_loc and from_loc == to_loc:
            warnings.append(
                f"{label}: from-compartment and to-compartment are both '{from_loc}'. "
                f"Transport reactions should cross compartment boundaries."
            )

        # Referenced compartments must be declared (if compartments[] is non-empty)
        if declared_compartments:
            if from_loc and from_loc not in declared_compartments:
                warnings.append(
                    f"{label}: from-compartment '{from_loc}' is not declared in "
                    f"compartments[]."
                )
            if to_loc and to_loc not in declared_compartments:
                warnings.append(
                    f"{label}: to-compartment '{to_loc}' is not declared in "
                    f"compartments[]."
                )

        # Cargo must reference a known entity
        cargo: str = ""
        raw_cargo = transport.get("cargo_complex") if isinstance(transport.get("cargo_complex"), str) else transport.get("cargo")
        if isinstance(raw_cargo, str):
            cargo = raw_cargo.strip()
        if cargo and cargo not in all_names:
            errors.append(
                f"{label}: cargo entity '{cargo}' is not in the entity registry."
            )

    # ------------------------------------------------------------------
    # 3. Class-vocabulary checks
    # ------------------------------------------------------------------
    for entity_type, valid_classes in _VALID_ENTITY_CLASSES.items():
        for eidx, item in enumerate(_safe_list(entities_dict.get(entity_type))):
            if not isinstance(item, dict):
                continue
            cls = (item.get("class") or "").strip()
            if cls and cls not in valid_classes:
                warnings.append(
                    f"entities.{entity_type}[{eidx}] '{item.get('name', '')}': "
                    f"class '{cls}' is not in the valid vocabulary "
                    f"{sorted(valid_classes)}."
                )

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        cls = (reaction.get("class") or "").strip()
        if cls and cls not in _VALID_REACTION_CLASSES:
            warnings.append(
                f"processes.reactions[{ridx}]: class '{cls}' not in valid "
                f"vocabulary {sorted(_VALID_REACTION_CLASSES)}."
            )

    for iidx, interaction in enumerate(_safe_list(processes.get("interactions"))):
        if not isinstance(interaction, dict):
            continue
        cls = (interaction.get("class") or "").strip()
        if cls and cls not in _VALID_INTERACTION_CLASSES:
            warnings.append(
                f"processes.interactions[{iidx}]: class '{cls}' not in valid "
                f"vocabulary {sorted(_VALID_INTERACTION_CLASSES)}."
            )

    # ------------------------------------------------------------------
    # 4. provenance="extracted" but confidence < 0.9 (contradiction)
    # ------------------------------------------------------------------
    for entity_type in (
        "compounds",
        "proteins",
        "protein_complexes",
        "nucleic_acids",
        "element_collections",
    ):
        for eidx, item in enumerate(_safe_list(entities_dict.get(entity_type))):
            if not isinstance(item, dict):
                continue
            prov = item.get("provenance")
            if not isinstance(prov, str) or prov.strip() != "extracted":
                continue
            conf = item.get("confidence")
            if conf is None:
                continue
            try:
                if float(conf) < 0.9:
                    warnings.append(
                        f"entities.{entity_type}[{eidx}] '{item.get('name', '')}': "
                        f"provenance='extracted' but confidence={conf} < 0.9 "
                        f"(contradictory — extracted entities should have high confidence)."
                    )
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------
    # 5. mapped_ids values must be non-empty strings when present
    # ------------------------------------------------------------------
    for entity_type in (
        "compounds",
        "proteins",
        "protein_complexes",
        "nucleic_acids",
        "element_collections",
    ):
        for eidx, item in enumerate(_safe_list(entities_dict.get(entity_type))):
            if not isinstance(item, dict):
                continue
            mapped_ids = item.get("mapped_ids")
            if not isinstance(mapped_ids, dict) or not mapped_ids:
                continue
            for k, v in mapped_ids.items():
                if v is None or (isinstance(v, str) and not v.strip()):
                    warnings.append(
                        f"entities.{entity_type}[{eidx}] '{item.get('name', '')}': "
                        f"mapped_ids['{k}'] is empty or null."
                    )

    return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)
