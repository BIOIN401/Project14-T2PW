from __future__ import annotations

import re
from typing import Any, Dict, List

from lxml import etree


def run_pwml_qa(pwml_bytes: bytes) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        root = etree.fromstring(pwml_bytes)
    except etree.XMLSyntaxError as exc:
        return {"ok": False, "errors": [f"XML parse error: {exc}"], "warnings": [], "stats": {}}

    # Collect all compound ids for orphan check
    compound_ids: set = set()
    for compound in root.iter("compound"):
        id_el = compound.find("id")
        if id_el is not None and id_el.text:
            compound_ids.add(id_el.text.strip())

    stats: Dict[str, int] = {
        "compounds": 0,
        "proteins": 0,
        "reactions": 0,
        "transports": 0,
        "interactions": 0,
        "enzymes_attached": 0,
        "reactions_no_enzyme": 0,
        "spontaneous_reactions": 0,
    }

    # Count compounds and proteins
    for _ in root.iter("compound"):
        stats["compounds"] += 1
    for _ in root.iter("protein"):
        stats["proteins"] += 1

    # Check reactions
    for reaction in root.iter("reaction"):
        stats["reactions"] += 1
        name_el = reaction.find("name")
        rxn_name = name_el.text.strip() if name_el is not None and name_el.text else "<unnamed>"

        spontaneous_el = reaction.find("spontaneous")
        is_spontaneous = spontaneous_el is not None and (spontaneous_el.text or "").strip().lower() == "true"
        if is_spontaneous:
            stats["spontaneous_reactions"] += 1

        enzymes_el = reaction.find("reaction-enzymes")
        has_enzyme = enzymes_el is not None and len(enzymes_el) > 0
        if has_enzyme:
            stats["enzymes_attached"] += 1
            if is_spontaneous:
                errors.append(f"Reaction '{rxn_name}' is marked spontaneous but also has an enzyme.")
            seen_complex_ids: set[str] = set()
            for enzyme in enzymes_el:
                complex_id_el = enzyme.find("protein-complex-id")
                complex_id = (complex_id_el.text or "").strip() if complex_id_el is not None else ""
                if not complex_id:
                    continue
                if complex_id in seen_complex_ids:
                    errors.append(
                        f"Reaction '{rxn_name}' assigns protein complex '{complex_id}' more than once as an enzyme."
                    )
                seen_complex_ids.add(complex_id)
        else:
            # Spontaneity is not modeled currently, so an enzyme-less reaction is
            # expected and not itself an error.
            stats["reactions_no_enzyme"] += 1

    # Check transports
    for transport in root.iter("transport-element"):
        stats["transports"] += 1
        left_el = transport.find("left-biological-state-id")
        right_el = transport.find("right-biological-state-id")
        left = (left_el.text or "").strip() if left_el is not None else None
        right = (right_el.text or "").strip() if right_el is not None else None
        if left is not None and right is not None and left == right:
            warnings.append(
                f"Transport element has identical left/right biological-state-id '{left}'."
            )

    # Check interactions (ORCH-747). PathWhiz imports an interaction only with
    # exactly one left and one right element, a type it stores, and a
    # visualization carrying a member on each side. Anything else is an error:
    # an interaction that cannot import is not a warning.
    _PWML_INTERACTION_TYPES = {"Activation", "Inhibition"}
    viz_sides_by_interaction: Dict[str, set] = {}
    for viz in root.iter("interaction-visualization"):
        ref_el = viz.find("interaction-id")
        ref = (ref_el.text or "").strip() if ref_el is not None else ""
        viz_sides_by_interaction[ref] = {(side.text or "").strip() for side in viz.iter("side")}
    for interaction in root.iter("interaction"):
        stats["interactions"] += 1
        id_el = interaction.find("id")
        iid = (id_el.text or "").strip() if id_el is not None else ""
        label = f"Interaction '{iid or '<no id>'}'"
        type_el = interaction.find("interaction-type")
        itype = (type_el.text or "").strip() if type_el is not None else ""
        if itype not in _PWML_INTERACTION_TYPES:
            errors.append(f"{label} has interaction-type '{itype}'; PathWhiz stores only Activation or Inhibition.")
        if interaction.find("name") is not None:
            errors.append(f"{label} carries a <name> child, which PathBank never writes on an interaction.")
        for side_tag in ("interaction-left-elements", "interaction-right-elements"):
            side_el = interaction.find(side_tag)
            count = len(side_el) if side_el is not None else 0
            if count != 1:
                errors.append(f"{label} has {count} {side_tag.replace('-elements', '')} element(s); exactly one is required.")
        sides = viz_sides_by_interaction.get(iid)
        if sides is None:
            errors.append(f"{label} has no interaction-visualization.")
        elif sides != {"Left", "Right"}:
            errors.append(f"{label} visualization has sides {sorted(sides)}; one Left and one Right member are required.")

    # Check stoichiometry values
    for stoich_el in root.iter("stoichiometry"):
        # An explicitly nil stoichiometry is valid PWML: PathWhiz stores
        # protein_complex_proteins.stoichiometry as a nullable column and its
        # parser skips blank nodes, so an unstated count exports blank.
        if stoich_el.get("nil"):
            continue
        val_text = (stoich_el.text or "").strip()
        try:
            val = int(val_text)
            if val <= 0:
                raise ValueError
        except (ValueError, TypeError):
            errors.append(f"Invalid stoichiometry value '{val_text}' (must be a positive integer).")

    # Check HMDB IDs
    _hmdb_re = re.compile(r"^HMDB\d{7}$")
    for hmdb_el in root.iter("hmdb-id"):
        if hmdb_el.get("nil"):
            continue
        text = (hmdb_el.text or "").strip()
        if text and not _hmdb_re.match(text):
            warnings.append(f"HMDB ID '{text}' is not zero-padded to 7 digits (expected HMDB\\d{{7}}).")

    # Check ChEBI IDs
    for chebi_el in root.iter("chebi-id"):
        if chebi_el.get("nil"):
            continue
        text = (chebi_el.text or "").strip()
        if text and not text.startswith("CHEBI:"):
            warnings.append(f"ChEBI ID '{text}' is missing the 'CHEBI:' prefix.")

    # Check orphaned compound-locations
    for cl in root.iter("compound-location"):
        cid_el = cl.find("compound-id")
        if cid_el is not None and cid_el.text:
            ref = cid_el.text.strip()
            if ref not in compound_ids:
                errors.append(f"Orphaned compound-location references compound-id '{ref}' which does not exist.")

    ok = len(errors) == 0
    return {"ok": ok, "errors": errors, "warnings": warnings, "stats": stats}
