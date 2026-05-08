from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from .sbml_metrics import species_referenced_in_reactions


_SID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sanitize_sid(value: str, fallback: str) -> str:
    raw = (value or "").strip() or fallback
    out = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not out:
        out = fallback
    if out[0].isdigit():
        out = f"_{out}"
    if not _SID_PATTERN.match(out):
        out = fallback
    return out


def _unique_sid(base: str, seen: Set[str]) -> str:
    if base not in seen:
        return base
    idx = 2
    while True:
        candidate = f"{base}_{idx}"
        if candidate not in seen:
            return candidate
        idx += 1


def _ensure_unique_reaction_ids(model: Any, warnings: List[Dict[str, str]]) -> None:
    seen: Set[str] = set()
    for idx in range(int(model.getNumReactions())):
        reaction = model.getReaction(idx)
        if reaction is None:
            continue
        current = str(reaction.getId() or "").strip()
        fallback = f"reaction_{idx + 1}"
        base = _sanitize_sid(current, fallback)
        unique = _unique_sid(base, seen)
        seen.add(unique)
        if current != unique:
            reaction.setId(unique)
            warnings.append(
                {
                    "type": "REACTION_ID_NORMALIZED",
                    "message": f"Normalized reaction id '{current or '<empty>'}' -> '{unique}'.",
                }
            )


def _stoich_key(value: float) -> str:
    rounded = round(float(value), 8)
    if abs(rounded - int(round(rounded))) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.8g}"


def _reaction_signature(reaction: Any) -> Tuple[str, Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...], Tuple[str, ...], bool]:
    comp = str(reaction.getCompartment() or "").strip()
    reactants: List[Tuple[str, str]] = []
    products: List[Tuple[str, str]] = []
    modifiers: List[str] = []
    for i in range(int(reaction.getNumReactants())):
        ref = reaction.getReactant(i)
        if ref is None:
            continue
        sid = str(ref.getSpecies() or "").strip()
        if sid:
            reactants.append((sid, _stoich_key(float(ref.getStoichiometry()))))
    for i in range(int(reaction.getNumProducts())):
        ref = reaction.getProduct(i)
        if ref is None:
            continue
        sid = str(ref.getSpecies() or "").strip()
        if sid:
            products.append((sid, _stoich_key(float(ref.getStoichiometry()))))
    for i in range(int(reaction.getNumModifiers())):
        ref = reaction.getModifier(i)
        if ref is None:
            continue
        sid = str(ref.getSpecies() or "").strip()
        if sid:
            modifiers.append(sid)
    reactants.sort()
    products.sort()
    modifiers.sort()
    return (comp, tuple(reactants), tuple(products), tuple(modifiers), bool(reaction.getReversible()))


def _merge_duplicate_reactions(model: Any) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...], Tuple[str, ...], bool], List[str]] = {}
    order: List[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...], Tuple[str, ...], bool]] = []
    for idx in range(int(model.getNumReactions())):
        rxn = model.getReaction(idx)
        if rxn is None:
            continue
        rid = str(rxn.getId() or "").strip()
        if not rid:
            continue
        sig = _reaction_signature(rxn)
        if sig not in groups:
            groups[sig] = []
            order.append(sig)
        groups[sig].append(rid)

    ops: List[Dict[str, Any]] = []
    drop_ids_all: Set[str] = set()
    for sig in order:
        ids = groups[sig]
        if len(ids) <= 1:
            continue
        keep_id = ids[0]
        drop_ids = ids[1:]
        keep_rxn = model.getReaction(keep_id)
        if keep_rxn is None:
            continue
        keep_modifiers = {
            str(keep_rxn.getModifier(i).getSpecies() or "").strip()
            for i in range(int(keep_rxn.getNumModifiers()))
            if keep_rxn.getModifier(i) is not None
        }
        for drop_id in drop_ids:
            drop_rxn = model.getReaction(drop_id)
            if drop_rxn is None:
                continue
            for i in range(int(drop_rxn.getNumModifiers())):
                ref = drop_rxn.getModifier(i)
                if ref is None:
                    continue
                sid = str(ref.getSpecies() or "").strip()
                if sid and sid not in keep_modifiers:
                    keep_ref = keep_rxn.createModifier()
                    keep_ref.setSpecies(sid)
                    keep_modifiers.add(sid)
            drop_ids_all.add(drop_id)

        ops.append(
            {
                "op": "merge_reactions",
                "keep_reaction_id": keep_id,
                "drop_reaction_ids": drop_ids,
                "reason": "Deterministic dedupe: identical reactants/products/modifiers/compartment/reversible.",
            }
        )

    if drop_ids_all:
        for idx in reversed(range(int(model.getNumReactions()))):
            rxn = model.getReaction(idx)
            if rxn is None:
                continue
            rid = str(rxn.getId() or "").strip()
            if rid in drop_ids_all:
                model.removeReaction(idx)
    return ops


def _normalize_compartment_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").strip().lower())


def _find_blood_compartment_ids(model: Any) -> Tuple[List[str], List[str]]:
    blood_ids: List[str] = []
    bloodstream_ids: List[str] = []
    for i in range(int(model.getNumCompartments())):
        comp = model.getCompartment(i)
        if comp is None:
            continue
        cid = str(comp.getId() or "").strip()
        cname = str(comp.getName() or "").strip()
        key_id = _normalize_compartment_token(cid)
        key_name = _normalize_compartment_token(cname)
        tokens = {key_id, key_name}
        if "bloodstream" in tokens:
            bloodstream_ids.append(cid)
        elif "blood" in tokens:
            blood_ids.append(cid)
    return blood_ids, bloodstream_ids


def _compartment_usage(model: Any) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    for i in range(int(model.getNumSpecies())):
        sp = model.getSpecies(i)
        if sp is None:
            continue
        cid = str(sp.getCompartment() or "").strip()
        if cid:
            usage[cid] = usage.get(cid, 0) + 1
    for i in range(int(model.getNumReactions())):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        cid = str(rxn.getCompartment() or "").strip()
        if cid:
            usage[cid] = usage.get(cid, 0) + 1
    return usage


def _move_compartment_refs(model: Any, old_id: str, new_id: str) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for i in range(int(model.getNumSpecies())):
        sp = model.getSpecies(i)
        if sp is None:
            continue
        if str(sp.getCompartment() or "").strip() == old_id:
            sp.setCompartment(new_id)
            sid = str(sp.getId() or "").strip()
            if sid:
                ops.append(
                    {
                        "op": "move_species_compartment",
                        "species_id": sid,
                        "new_compartment_id": new_id,
                        "reason": f"Merged compartment '{old_id}' into '{new_id}'.",
                    }
                )
    for i in range(int(model.getNumReactions())):
        rxn = model.getReaction(i)
        if rxn is None:
            continue
        if str(rxn.getCompartment() or "").strip() == old_id:
            rxn.setCompartment(new_id)
    return ops


def _cleanup_unused_compartments(model: Any, candidate_ids: Set[str]) -> List[Dict[str, Any]]:
    usage = _compartment_usage(model)
    ops: List[Dict[str, Any]] = []
    for idx in reversed(range(int(model.getNumCompartments()))):
        comp = model.getCompartment(idx)
        if comp is None:
            continue
        cid = str(comp.getId() or "").strip()
        if cid in candidate_ids and usage.get(cid, 0) == 0:
            model.removeCompartment(idx)
            ops.append(
                {
                    "op": "delete_compartment",
                    "compartment_id": cid,
                    "reason": "Removed unused compartment after deterministic merge.",
                }
            )
    return ops


def run_deterministic_cleanup(model: Any) -> Dict[str, Any]:
    warnings: List[Dict[str, str]] = []
    ops: List[Dict[str, Any]] = []

    _ensure_unique_reaction_ids(model, warnings)

    referenced_species = species_referenced_in_reactions(model)
    prune_ids: Set[str] = set()
    for idx in range(int(model.getNumSpecies())):
        sp = model.getSpecies(idx)
        if sp is None:
            continue
        sid = str(sp.getId() or "").strip()
        if sid and sid not in referenced_species:
            prune_ids.add(sid)
    if prune_ids:
        for idx in reversed(range(int(model.getNumSpecies()))):
            sp = model.getSpecies(idx)
            if sp is None:
                continue
            sid = str(sp.getId() or "").strip()
            if sid in prune_ids:
                model.removeSpecies(idx)
                ops.append(
                    {
                        "op": "delete_species",
                        "species_id": sid,
                        "reason": "Deterministic cleanup: species is not referenced by any reaction.",
                    }
                )

    ops.extend(_merge_duplicate_reactions(model))

    blood_ids, bloodstream_ids = _find_blood_compartment_ids(model)
    if blood_ids and bloodstream_ids:
        usage = _compartment_usage(model)
        blood_usage = sum(usage.get(cid, 0) for cid in blood_ids)
        bloodstream_usage = sum(usage.get(cid, 0) for cid in bloodstream_ids)
        canonical_id = bloodstream_ids[0] if (blood_usage == 0 or bloodstream_usage > 0) else blood_ids[0]
        merge_from = {cid for cid in (blood_ids + bloodstream_ids) if cid != canonical_id}
        for old_id in sorted(merge_from):
            ops.extend(_move_compartment_refs(model, old_id, canonical_id))
        ops.extend(_cleanup_unused_compartments(model, merge_from))

    return {"ops": ops, "warnings": warnings}

