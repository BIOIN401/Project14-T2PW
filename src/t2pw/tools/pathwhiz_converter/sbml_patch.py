from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .schemas import (
    AddReactionOp,
    DeleteCompartmentOp,
    DeleteSpeciesOp,
    MergeReactionsOp,
    MoveSpeciesCompartmentOp,
    PatchPlan,
    RenameCompartmentOp,
    RenameSpeciesOp,
)


def _species_ids_in_reaction(reaction: Any) -> Set[str]:
    ids: Set[str] = set()
    for idx in range(int(reaction.getNumReactants())):
        ref = reaction.getReactant(idx)
        if ref is not None:
            sid = str(ref.getSpecies() or "").strip()
            if sid:
                ids.add(sid)
    for idx in range(int(reaction.getNumProducts())):
        ref = reaction.getProduct(idx)
        if ref is not None:
            sid = str(ref.getSpecies() or "").strip()
            if sid:
                ids.add(sid)
    for idx in range(int(reaction.getNumModifiers())):
        ref = reaction.getModifier(idx)
        if ref is not None:
            sid = str(ref.getSpecies() or "").strip()
            if sid:
                ids.add(sid)
    return ids


def _species_is_referenced(model: Any, species_id: str) -> bool:
    for idx in range(int(model.getNumReactions())):
        reaction = model.getReaction(idx)
        if reaction is None:
            continue
        if species_id in _species_ids_in_reaction(reaction):
            return True
    return False


def _compartment_is_used(model: Any, compartment_id: str) -> bool:
    for idx in range(int(model.getNumSpecies())):
        sp = model.getSpecies(idx)
        if sp is not None and str(sp.getCompartment() or "").strip() == compartment_id:
            return True
    for idx in range(int(model.getNumReactions())):
        rxn = model.getReaction(idx)
        if rxn is not None and str(rxn.getCompartment() or "").strip() == compartment_id:
            return True
    return False


def _move_compartment_references(model: Any, old_id: str, new_id: str) -> None:
    for idx in range(int(model.getNumSpecies())):
        sp = model.getSpecies(idx)
        if sp is None:
            continue
        if str(sp.getCompartment() or "").strip() == old_id:
            sp.setCompartment(new_id)
    for idx in range(int(model.getNumReactions())):
        rxn = model.getReaction(idx)
        if rxn is None:
            continue
        if str(rxn.getCompartment() or "").strip() == old_id:
            rxn.setCompartment(new_id)


def _sanitize_sid(value: str, fallback: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", (value or "").strip() or fallback)
    if not candidate:
        candidate = fallback
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    return candidate


def _unique_reaction_id(model: Any, base: str) -> str:
    existing = {
        str(model.getReaction(idx).getId() or "").strip()
        for idx in range(int(model.getNumReactions()))
        if model.getReaction(idx) is not None
    }
    existing.discard("")
    if base not in existing:
        return base
    n = 2
    while True:
        candidate = f"{base}_{n}"
        if candidate not in existing:
            return candidate
        n += 1


def _add_modifier_if_missing(keep_rxn: Any, species_id: str) -> None:
    existing = {
        str(keep_rxn.getModifier(i).getSpecies() or "").strip()
        for i in range(int(keep_rxn.getNumModifiers()))
        if keep_rxn.getModifier(i) is not None
    }
    if species_id in existing:
        return
    ref = keep_rxn.createModifier()
    ref.setSpecies(species_id)


def _create_participant(reaction: Any, kind: str, species_id: str, stoich: float) -> None:
    if kind == "reactant":
        ref = reaction.createReactant()
    elif kind == "product":
        ref = reaction.createProduct()
    else:
        raise ValueError(f"Unsupported participant kind: {kind}")
    ref.setSpecies(species_id)
    ref.setStoichiometry(float(stoich))
    ref.setConstant(True)


def apply_patch_plan(model: Any, patch_plan: PatchPlan, *, allow_add_reaction: bool) -> Dict[str, List[Dict[str, Any]]]:
    applied_ops: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for op in patch_plan.ops:
        if isinstance(op, DeleteSpeciesOp):
            sid = op.species_id.strip()
            if not sid or model.getSpecies(sid) is None:
                warnings.append({"type": "PATCH_SKIP", "message": f"delete_species skipped (missing): {sid}"})
                continue
            if _species_is_referenced(model, sid):
                warnings.append({"type": "PATCH_SKIP", "message": f"delete_species skipped (still referenced): {sid}"})
                continue
            for idx in reversed(range(int(model.getNumSpecies()))):
                sp = model.getSpecies(idx)
                if sp is not None and str(sp.getId() or "").strip() == sid:
                    model.removeSpecies(idx)
            applied_ops.append(op.model_dump())
            continue

        if isinstance(op, MergeReactionsOp):
            keep = model.getReaction(op.keep_reaction_id.strip())
            if keep is None:
                warnings.append(
                    {"type": "PATCH_SKIP", "message": f"merge_reactions skipped (missing keep): {op.keep_reaction_id}"}
                )
                continue
            drop_set: Set[str] = {rid.strip() for rid in op.drop_reaction_ids if rid.strip()}
            if not drop_set:
                warnings.append({"type": "PATCH_SKIP", "message": "merge_reactions skipped (empty drop set)."})
                continue
            for rid in sorted(drop_set):
                drop = model.getReaction(rid)
                if drop is None:
                    continue
                for idx in range(int(drop.getNumModifiers())):
                    ref = drop.getModifier(idx)
                    if ref is None:
                        continue
                    sid = str(ref.getSpecies() or "").strip()
                    if sid:
                        _add_modifier_if_missing(keep, sid)
            for idx in reversed(range(int(model.getNumReactions()))):
                rxn = model.getReaction(idx)
                if rxn is None:
                    continue
                rid = str(rxn.getId() or "").strip()
                if rid in drop_set:
                    model.removeReaction(idx)
            applied_ops.append(op.model_dump())
            continue

        if isinstance(op, RenameCompartmentOp):
            old_id = op.old_id.strip()
            new_id = op.new_id.strip()
            if not old_id or not new_id:
                warnings.append({"type": "PATCH_SKIP", "message": "rename_compartment skipped (empty id)."})
                continue
            old_comp = model.getCompartment(old_id)
            if old_comp is None:
                warnings.append({"type": "PATCH_SKIP", "message": f"rename_compartment skipped (missing): {old_id}"})
                continue
            if old_id == new_id:
                applied_ops.append(op.model_dump())
                continue
            target = model.getCompartment(new_id)
            if target is None:
                target = model.createCompartment()
                target.setId(new_id)
                target.setName(str(old_comp.getName() or "").strip() or new_id)
                target.setConstant(bool(old_comp.getConstant()))
                target.setSize(float(old_comp.getSize()))
                target.setSpatialDimensions(float(old_comp.getSpatialDimensions()))
            _move_compartment_references(model, old_id, new_id)
            if not _compartment_is_used(model, old_id):
                for idx in reversed(range(int(model.getNumCompartments()))):
                    comp = model.getCompartment(idx)
                    if comp is not None and str(comp.getId() or "").strip() == old_id:
                        model.removeCompartment(idx)
            applied_ops.append(op.model_dump())
            continue

        if isinstance(op, DeleteCompartmentOp):
            cid = op.compartment_id.strip()
            if not cid or model.getCompartment(cid) is None:
                warnings.append({"type": "PATCH_SKIP", "message": f"delete_compartment skipped (missing): {cid}"})
                continue
            if _compartment_is_used(model, cid):
                warnings.append({"type": "PATCH_SKIP", "message": f"delete_compartment skipped (still used): {cid}"})
                continue
            for idx in reversed(range(int(model.getNumCompartments()))):
                comp = model.getCompartment(idx)
                if comp is not None and str(comp.getId() or "").strip() == cid:
                    model.removeCompartment(idx)
            applied_ops.append(op.model_dump())
            continue

        if isinstance(op, MoveSpeciesCompartmentOp):
            sid = op.species_id.strip()
            cid = op.new_compartment_id.strip()
            sp = model.getSpecies(sid) if sid else None
            comp = model.getCompartment(cid) if cid else None
            if sp is None or comp is None:
                warnings.append(
                    {"type": "PATCH_SKIP", "message": f"move_species_compartment skipped (missing species or compartment): {sid}->{cid}"}
                )
                continue
            sp.setCompartment(cid)
            applied_ops.append(op.model_dump())
            continue

        if isinstance(op, RenameSpeciesOp):
            sid = op.species_id.strip()
            sp = model.getSpecies(sid) if sid else None
            if sp is None:
                warnings.append({"type": "PATCH_SKIP", "message": f"rename_species skipped (missing): {sid}"})
                continue
            sp.setName(op.new_name)
            applied_ops.append(op.model_dump())
            continue

        if isinstance(op, AddReactionOp):
            if not allow_add_reaction:
                warnings.append({"type": "PATCH_SKIP", "message": "add_reaction rejected (feature disabled)."})
                continue
            if model.getCompartment(op.compartment_id) is None:
                warnings.append(
                    {
                        "type": "PATCH_SKIP",
                        "message": f"add_reaction skipped (missing compartment): {op.compartment_id}",
                    }
                )
                continue
            rid = _unique_reaction_id(model, _sanitize_sid(op.reaction_id, "added_reaction"))
            rxn = model.createReaction()
            rxn.setId(rid)
            rxn.setName(op.name or rid)
            rxn.setCompartment(op.compartment_id)
            rxn.setReversible(bool(op.reversible))
            rxn.setFast(False)

            added_reactants = 0
            added_products = 0
            for item in op.reactants:
                if model.getSpecies(item.species_id) is None:
                    warnings.append({"type": "PATCH_PARTIAL", "message": f"add_reaction missing reactant species: {item.species_id}"})
                    continue
                _create_participant(rxn, "reactant", item.species_id, item.stoichiometry)
                added_reactants += 1
            for item in op.products:
                if model.getSpecies(item.species_id) is None:
                    warnings.append({"type": "PATCH_PARTIAL", "message": f"add_reaction missing product species: {item.species_id}"})
                    continue
                _create_participant(rxn, "product", item.species_id, item.stoichiometry)
                added_products += 1
            for sid in op.modifiers:
                if model.getSpecies(sid) is None:
                    warnings.append({"type": "PATCH_PARTIAL", "message": f"add_reaction missing modifier species: {sid}"})
                    continue
                _add_modifier_if_missing(rxn, sid)

            if added_reactants == 0 and added_products == 0:
                for idx in reversed(range(int(model.getNumReactions()))):
                    candidate = model.getReaction(idx)
                    if candidate is not None and str(candidate.getId() or "").strip() == rid:
                        model.removeReaction(idx)
                        break
                warnings.append({"type": "PATCH_SKIP", "message": f"add_reaction skipped (no valid participants): {rid}"})
                continue
            applied_ops.append(op.model_dump())
            continue

    for warning in patch_plan.warnings:
        warnings.append(warning.model_dump())
    return {"applied_ops": applied_ops, "warnings": warnings}

