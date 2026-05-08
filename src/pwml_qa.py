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
        elif not is_spontaneous:
            stats["reactions_no_enzyme"] += 1
            errors.append(f"Reaction '{rxn_name}' has no enzyme and is not spontaneous.")

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

    # Check stoichiometry values
    for stoich_el in root.iter("stoichiometry"):
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
