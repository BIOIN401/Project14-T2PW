from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


def _safe_reaction_id(reaction: Any, index: int) -> str:
    rid = str(reaction.getId() or "").strip()
    return rid or f"reaction_{index + 1}"


def _stoich_key(value: float) -> str:
    rounded = round(float(value), 8)
    if abs(rounded - int(round(rounded))) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.8g}"


def _reaction_signature(reaction: Any) -> Tuple[str, Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...], Tuple[str, ...], bool]:
    compartment = str(reaction.getCompartment() or "").strip()
    reactants: List[Tuple[str, str]] = []
    products: List[Tuple[str, str]] = []
    modifiers: List[str] = []

    for idx in range(int(reaction.getNumReactants())):
        ref = reaction.getReactant(idx)
        if ref is None:
            continue
        sid = str(ref.getSpecies() or "").strip()
        if not sid:
            continue
        reactants.append((sid, _stoich_key(float(ref.getStoichiometry()))))

    for idx in range(int(reaction.getNumProducts())):
        ref = reaction.getProduct(idx)
        if ref is None:
            continue
        sid = str(ref.getSpecies() or "").strip()
        if not sid:
            continue
        products.append((sid, _stoich_key(float(ref.getStoichiometry()))))

    for idx in range(int(reaction.getNumModifiers())):
        ref = reaction.getModifier(idx)
        if ref is None:
            continue
        sid = str(ref.getSpecies() or "").strip()
        if sid:
            modifiers.append(sid)

    reactants.sort()
    products.sort()
    modifiers.sort()
    return (compartment, tuple(reactants), tuple(products), tuple(modifiers), bool(reaction.getReversible()))


def species_referenced_in_reactions(model: Any) -> Set[str]:
    referenced: Set[str] = set()
    for rxn_idx in range(int(model.getNumReactions())):
        reaction = model.getReaction(rxn_idx)
        if reaction is None:
            continue
        for idx in range(int(reaction.getNumReactants())):
            ref = reaction.getReactant(idx)
            if ref is not None:
                sid = str(ref.getSpecies() or "").strip()
                if sid:
                    referenced.add(sid)
        for idx in range(int(reaction.getNumProducts())):
            ref = reaction.getProduct(idx)
            if ref is not None:
                sid = str(ref.getSpecies() or "").strip()
                if sid:
                    referenced.add(sid)
        for idx in range(int(reaction.getNumModifiers())):
            ref = reaction.getModifier(idx)
            if ref is not None:
                sid = str(ref.getSpecies() or "").strip()
                if sid:
                    referenced.add(sid)
    return referenced


def duplicate_reaction_groups(model: Any) -> List[List[str]]:
    groups: Dict[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...], Tuple[str, ...], bool], List[str]] = {}
    for rxn_idx in range(int(model.getNumReactions())):
        reaction = model.getReaction(rxn_idx)
        if reaction is None:
            continue
        sig = _reaction_signature(reaction)
        rid = _safe_reaction_id(reaction, rxn_idx)
        groups.setdefault(sig, []).append(rid)
    return [ids for ids in groups.values() if len(ids) > 1]


def compute_model_metrics(model: Any) -> Dict[str, Any]:
    species_ids = [
        str(model.getSpecies(i).getId() or "").strip()
        for i in range(int(model.getNumSpecies()))
        if model.getSpecies(i) is not None
    ]
    species_ids = [sid for sid in species_ids if sid]
    referenced = species_referenced_in_reactions(model)
    isolated = sorted([sid for sid in species_ids if sid not in referenced])
    dup_groups = duplicate_reaction_groups(model)
    return {
        "isolated_species_count": len(isolated),
        "isolated_species_ids": isolated,
        "duplicate_reactions_count": sum(len(group) - 1 for group in dup_groups),
        "duplicate_reaction_groups": dup_groups,
        "compartments_count": int(model.getNumCompartments()),
        "species_count": int(model.getNumSpecies()),
        "reactions_count": int(model.getNumReactions()),
    }

