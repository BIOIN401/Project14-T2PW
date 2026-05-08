#!/usr/bin/env python3
"""sbml_strip_unmapped.py

Produce a "clean" copy of a PathWhiz render-ready SBML by removing any
elements that lack the mandatory PathWhiz IDs PathWhiz requires for import:

  - species  without pathwhiz:species_id
  - reactions without pathwhiz:reaction_id  (or that reference a removed species)
  - compartments without pathwhiz:compartment_id (cascade: their species go too)

The original file is not modified.  Returns the cleaned XML bytes plus a
summary dict describing what was removed and why.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Set, Tuple

PW_NS = "http://www.spmdb.ca/pathwhiz"
SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"

# Register namespaces so ET preserves prefixes on re-serialisation
ET.register_namespace("", SBML_NS)
ET.register_namespace("pathwhiz", PW_NS)
ET.register_namespace("bqbiol", "http://biomodels.net/biology-qualifiers/")
ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")


def _has_pw_attr(elem: ET.Element, attr: str) -> bool:
    """Return True if elem or any descendant has pathwhiz:<attr>."""
    key = f"{{{PW_NS}}}{attr}"
    if elem.get(key):
        return True
    for child in elem.iter():
        if child.get(key):
            return True
    return False


def strip_unmapped(sbml_bytes: bytes) -> Tuple[bytes, Dict]:
    """Strip entities lacking PathWhiz IDs from *sbml_bytes*.

    Returns
    -------
    (clean_bytes, summary)
        clean_bytes : XML bytes of the stripped SBML
        summary     : dict with removal counts and lists
    """
    root = ET.fromstring(sbml_bytes)

    # ── helpers ────────────────────────────────────────────────────────────
    def _tag(elem: ET.Element) -> str:
        t = elem.tag
        return t.split("}")[-1] if "}" in t else t

    def _find_list_of(parent: ET.Element, local: str):
        """Return the listOf<X> element and its children matching local tag.

        SBML uses listOfSpecies (not listOfSpeciess), listOfReactions, listOfCompartments.
        We match by checking that the lowercased tag starts with 'listof' and ends with
        the local name (handles the species/speciess edge case).
        """
        local_lower = local.lower()
        for child in parent:
            t = _tag(child).lower()
            if t.startswith("listof") and t.rstrip("s").endswith(local_lower.rstrip("s")):
                return child, [c for c in child if _tag(c) == local]
        return None, []

    # Locate <model>
    model = None
    for child in root:
        if _tag(child) == "model":
            model = child
            break
    if model is None:
        return sbml_bytes, {"error": "No <model> element found"}

    summary: Dict = {
        "removed_compartments": [],
        "removed_species": [],
        "removed_reactions": [],
        "cascade_removed_reactions": [],
    }

    # ── 1. Compartments ────────────────────────────────────────────────────
    cpt_list, compartments = _find_list_of(model, "compartment")
    bad_compartments: Set[str] = set()
    if cpt_list is not None:
        for cpt in list(compartments):
            if not _has_pw_attr(cpt, "compartment_id"):
                cid = cpt.get("id", "")
                bad_compartments.add(cid)
                cpt_list.remove(cpt)
                summary["removed_compartments"].append(
                    {"id": cid, "name": cpt.get("name", ""), "reason": "no_pathwhiz_compartment_id"}
                )

    # ── 2. Species ─────────────────────────────────────────────────────────
    sp_list, species_elems = _find_list_of(model, "species")
    bad_species: Set[str] = set()
    if sp_list is not None:
        for sp in list(species_elems):
            sid = sp.get("id", "")
            in_bad_cpt = sp.get("compartment", "") in bad_compartments
            no_pw_id = not _has_pw_attr(sp, "species_id")
            if in_bad_cpt or no_pw_id:
                bad_species.add(sid)
                sp_list.remove(sp)
                reason = []
                if no_pw_id:
                    reason.append("no_pathwhiz_species_id")
                if in_bad_cpt:
                    reason.append("compartment_removed")
                summary["removed_species"].append(
                    {"id": sid, "name": sp.get("name", ""), "reason": "+".join(reason)}
                )

    # ── 3. Reactions ───────────────────────────────────────────────────────
    rxn_list, reactions = _find_list_of(model, "reaction")
    if rxn_list is not None:
        for rxn in list(reactions):
            rid = rxn.get("id", "")
            no_pw_id = not _has_pw_attr(rxn, "reaction_id")

            # Check if any species reference points to a removed species
            refs_bad = any(
                sr.get("species", "") in bad_species
                for sr in rxn.iter()
                if _tag(sr) in ("speciesReference", "modifierSpeciesReference")
            )

            if no_pw_id or refs_bad:
                rxn_list.remove(rxn)
                reason = []
                if no_pw_id:
                    reason.append("no_pathwhiz_reaction_id")
                if refs_bad:
                    reason.append("references_removed_species")
                entry = {"id": rid, "name": rxn.get("name", ""), "reason": "+".join(reason)}
                if refs_bad and not no_pw_id:
                    summary["cascade_removed_reactions"].append(entry)
                else:
                    summary["removed_reactions"].append(entry)

    # ── 4. Summary counts ──────────────────────────────────────────────────
    summary["total_removed_compartments"] = len(summary["removed_compartments"])
    summary["total_removed_species"] = len(summary["removed_species"])
    summary["total_removed_reactions"] = len(summary["removed_reactions"])
    summary["total_cascade_removed_reactions"] = len(summary["cascade_removed_reactions"])

    # ── 5. Serialise ───────────────────────────────────────────────────────
    ET.indent(root, space="  ")
    clean_bytes = ET.tostring(root, encoding="unicode", xml_declaration=False).encode("utf-8")
    # Prepend XML declaration
    clean_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n' + clean_bytes

    return clean_bytes, summary
