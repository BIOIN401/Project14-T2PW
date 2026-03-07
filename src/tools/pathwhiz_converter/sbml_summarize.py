from __future__ import annotations

from typing import Any, Dict, List


def _truncate(rows: List[Dict[str, Any]], max_items: int) -> Dict[str, Any]:
    if len(rows) <= max_items:
        return {"rows": rows, "omitted_count": 0}
    return {"rows": rows[:max_items], "omitted_count": len(rows) - max_items}


def summarize_model(
    model: Any,
    *,
    metrics: Dict[str, Any] | None = None,
    max_compartments: int = 64,
    max_species: int = 600,
    max_reactions: int = 600,
) -> Dict[str, Any]:
    compartments: List[Dict[str, Any]] = []
    for idx in range(int(model.getNumCompartments())):
        comp = model.getCompartment(idx)
        if comp is None:
            continue
        cid = str(comp.getId() or "").strip()
        if not cid:
            continue
        compartments.append({"id": cid, "name": str(comp.getName() or "").strip() or cid})

    species_rows: List[Dict[str, Any]] = []
    for idx in range(int(model.getNumSpecies())):
        sp = model.getSpecies(idx)
        if sp is None:
            continue
        sid = str(sp.getId() or "").strip()
        if not sid:
            continue
        species_rows.append(
            {
                "id": sid,
                "name": str(sp.getName() or "").strip() or sid,
                "compartment": str(sp.getCompartment() or "").strip(),
            }
        )

    reaction_rows: List[Dict[str, Any]] = []
    for idx in range(int(model.getNumReactions())):
        rxn = model.getReaction(idx)
        if rxn is None:
            continue
        rid = str(rxn.getId() or "").strip() or f"reaction_{idx + 1}"
        reactants: List[str] = []
        products: List[str] = []
        modifiers: List[str] = []
        for j in range(int(rxn.getNumReactants())):
            ref = rxn.getReactant(j)
            if ref is not None:
                sid = str(ref.getSpecies() or "").strip()
                if sid:
                    reactants.append(sid)
        for j in range(int(rxn.getNumProducts())):
            ref = rxn.getProduct(j)
            if ref is not None:
                sid = str(ref.getSpecies() or "").strip()
                if sid:
                    products.append(sid)
        for j in range(int(rxn.getNumModifiers())):
            ref = rxn.getModifier(j)
            if ref is not None:
                sid = str(ref.getSpecies() or "").strip()
                if sid:
                    modifiers.append(sid)
        reaction_rows.append(
            {
                "id": rid,
                "name": str(rxn.getName() or "").strip() or rid,
                "compartment": str(rxn.getCompartment() or "").strip(),
                "reactants": sorted(reactants),
                "products": sorted(products),
                "modifiers": sorted(modifiers),
            }
        )

    comp_block = _truncate(compartments, max_compartments)
    species_block = _truncate(species_rows, max_species)
    reaction_block = _truncate(reaction_rows, max_reactions)

    summary: Dict[str, Any] = {
        "compartments": comp_block["rows"],
        "species": species_block["rows"],
        "reactions": reaction_block["rows"],
        "totals": {
            "compartments": len(compartments),
            "species": len(species_rows),
            "reactions": len(reaction_rows),
        },
        "omitted": {
            "compartments": comp_block["omitted_count"],
            "species": species_block["omitted_count"],
            "reactions": reaction_block["omitted_count"],
        },
    }
    if metrics is not None:
        summary["metrics"] = metrics
    return summary

