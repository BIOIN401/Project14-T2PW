#!/usr/bin/env python3
"""build_pathwhiz_id_db.py

Parse all .pwml files and build a lookup database of PathWhiz internal IDs.

Outputs pathwhiz_id_db.json with structure:
{
  "compounds": {
    "hmdb": {"HMDB0002111": 1420, ...},
    "kegg": {"C00001": 1420, ...},
    "chebi": {"15377": 1420, ...},
    "pubchem": {"962": 1420, ...},
    "by_id": {"1420": {"name": "Water", "hmdb": "HMDB0002111", ...}}
  },
  "proteins": {
    "uniprot": {"Q9LZA6": 11601, ...},
    "by_id": {"11601": {"name": "...", "uniprot": "Q9LZA6"}}
  },
  "protein_complexes": {
    "by_id": {"5479": {"name": "...", "protein_ids": [11601, ...]}}
  },
  "element_collections": {
    "kegg": {"C00422": 309, ...},
    "by_id": {"309": {"name": "Triglyceride", ...}}
  },
  "biological_states": {
    "by_id": {"227": {"subcellular_location_id": 5, "subcellular_location_name": "Oil Body"}},
    "by_location_name": {"oil body": 227, ...},
    "by_location_id": {"5": [8, 227, ...]}
  },
  "reactions": {
    "by_id": {"47595": {"left": [{"element_id": 309, "element_type": "ElementCollection"}], "right": [...], "enzymes": [...]}},
    "by_elements": {"309,310": 47595, ...}
  },
  "subcellular_locations": {
    "by_id": {"5": "Cytoplasm", ...},
    "by_name": {"cytoplasm": 5, ...}
  }
}

Usage:
  python build_pathwhiz_id_db.py [--pwml-dir .] [--out pathwhiz_id_db.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET


def _text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


def _int(el: Optional[ET.Element]) -> Optional[int]:
    t = _text(el)
    if not t or t == "nil" or el is not None and el.get("nil") == "true":
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _find_pathway_vis(root: ET.Element) -> Optional[ET.Element]:
    """Navigate to the <pathway-visualization> element."""
    for ctx in root.findall(".//pathway-visualization-context"):
        pv = ctx.find("pathway-visualization")
        if pv is not None:
            return pv
    return None


def _load_pwml_xml(path: Path) -> Optional[ET.ElementTree]:
    """Load a .pwml file, working around the `option:` unbound-prefix issue."""
    import re
    raw = path.read_bytes().decode("utf-8", errors="replace")
    # Replace <option:foo>...</option:foo> and <option:foo/> with plain
    # <pw_option_foo>...</pw_option_foo> so the XML is well-formed.
    raw = re.sub(r"<option:(\w+)([^>]*)/>", r"<pw_option_\1\2/>", raw)
    raw = re.sub(r"<option:(\w+)([^>]*)>", r"<pw_option_\1\2>", raw)
    raw = re.sub(r"</option:(\w+)>", r"</pw_option_\1>", raw)
    try:
        return ET.ElementTree(ET.fromstring(raw.encode("utf-8")))
    except ET.ParseError as e:
        print(f"  WARNING: parse error in {path.name}: {e}", file=sys.stderr)
        return None


def parse_pwml(path: Path, db: Dict[str, Any]) -> None:
    """Parse a single .pwml file and merge its entities into db."""
    tree = _load_pwml_xml(path)
    if tree is None:
        return

    root = tree.getroot()
    pv = _find_pathway_vis(root)
    if pv is None:
        print(f"  WARNING: no pathway-visualization in {path.name}", file=sys.stderr)
        return

    # ------------------------------------------------------------------
    # Subcellular locations
    # ------------------------------------------------------------------
    for sl in pv.findall(".//subcellular-locations/subcellular-location"):
        sl_id = _int(sl.find("id"))
        sl_name = _text(sl.find("name"))
        if sl_id is None or not sl_name:
            continue
        db["subcellular_locations"]["by_id"][str(sl_id)] = sl_name
        db["subcellular_locations"]["by_name"][sl_name.lower()] = sl_id

    # ------------------------------------------------------------------
    # Biological states (compartments)
    # ------------------------------------------------------------------
    for bs in pv.findall(".//biological-states/biological-state"):
        bs_id = _int(bs.find("id"))
        if bs_id is None:
            continue
        sl_id = _int(bs.find("subcellular-location-id"))
        sp_id = _int(bs.find("species-id"))
        sl_name = db["subcellular_locations"]["by_id"].get(str(sl_id), "") if sl_id else ""
        db["biological_states"]["by_id"][str(bs_id)] = {
            "subcellular_location_id": sl_id,
            "subcellular_location_name": sl_name,
            "species_id": sp_id,
        }
        if sl_name:
            key = sl_name.lower()
            db["biological_states"]["by_location_name"][key] = bs_id
            db["biological_states"]["by_location_id"].setdefault(str(sl_id), [])
            if bs_id not in db["biological_states"]["by_location_id"][str(sl_id)]:
                db["biological_states"]["by_location_id"][str(sl_id)].append(bs_id)

    # ------------------------------------------------------------------
    # Compounds
    # ------------------------------------------------------------------
    for c in pv.findall(".//compounds/compound"):
        cid = _int(c.find("id"))
        if cid is None:
            continue
        name = _text(c.find("name"))
        hmdb = _text(c.find("hmdb-id"))
        kegg = _text(c.find("kegg-id"))
        chebi = _text(c.find("chebi-id"))
        pubchem = _text(c.find("pubchem-cid"))
        drugbank = _text(c.find("drugbank-id"))
        db["compounds"]["by_id"][str(cid)] = {
            "name": name, "hmdb": hmdb, "kegg": kegg,
            "chebi": chebi, "pubchem": pubchem, "drugbank": drugbank,
        }
        if hmdb:
            db["compounds"]["hmdb"][hmdb] = cid
            # Normalize HMDB: some stored without leading zeros
            hmdb_norm = hmdb.replace("HMDB", "HMDB").lstrip("HMDB0").lstrip("0")
            db["compounds"]["hmdb"][f"HMDB{hmdb_norm.zfill(7)}"] = cid
        if kegg:
            db["compounds"]["kegg"][kegg] = cid
        if chebi:
            db["compounds"]["chebi"][chebi] = cid
            db["compounds"]["chebi"][f"CHEBI:{chebi}"] = cid
        if pubchem:
            db["compounds"]["pubchem"][pubchem] = cid
        if drugbank:
            db["compounds"]["drugbank"][drugbank] = cid

    # ------------------------------------------------------------------
    # Element collections (compound families, gene families)
    # ------------------------------------------------------------------
    for ec in pv.findall(".//element-collections/element-collection"):
        ec_id = _int(ec.find("id"))
        if ec_id is None:
            continue
        name = _text(ec.find("name"))
        ext_id = _text(ec.find("external-id"))
        ext_type = _text(ec.find("external-id-type"))
        pwec = _text(ec.find("pwec-id"))
        etype = _text(ec.find("element-type"))
        inner_eid = _int(ec.find("element-id"))
        db["element_collections"]["by_id"][str(ec_id)] = {
            "name": name, "pwec_id": pwec,
            "element_type": etype, "element_id": inner_eid,
            "external_id": ext_id, "external_id_type": ext_type,
        }
        if ext_id and ext_type:
            key = ext_type.lower().replace(" ", "_")
            db["element_collections"].setdefault(key, {})
            db["element_collections"][key][ext_id] = ec_id

    # ------------------------------------------------------------------
    # Proteins
    # ------------------------------------------------------------------
    for p in pv.findall(".//proteins/protein"):
        pid = _int(p.find("id"))
        if pid is None:
            continue
        name = _text(p.find("name"))
        uniprot = _text(p.find("uniprot-id"))
        gene = _text(p.find("gene-name"))
        db["proteins"]["by_id"][str(pid)] = {"name": name, "uniprot": uniprot, "gene": gene}
        if uniprot:
            db["proteins"]["uniprot"][uniprot] = pid

    # ------------------------------------------------------------------
    # Protein complexes
    # ------------------------------------------------------------------
    for pc in pv.findall(".//protein-complexes/protein-complex"):
        pc_id = _int(pc.find("id"))
        if pc_id is None:
            continue
        name = _text(pc.find("name"))
        protein_ids: List[int] = []
        for pcp in pc.findall(".//protein_complex-proteins/protein-complex-protein"):
            p_id = _int(pcp.find("protein-id"))
            if p_id is not None:
                protein_ids.append(p_id)
        db["protein_complexes"]["by_id"][str(pc_id)] = {
            "name": name, "protein_ids": protein_ids,
        }
        # Index by uniprot IDs of member proteins
        for p_id in protein_ids:
            p_info = db["proteins"]["by_id"].get(str(p_id), {})
            uniprot = p_info.get("uniprot", "")
            if uniprot:
                db["protein_complexes"].setdefault("by_uniprot_member", {})
                db["protein_complexes"]["by_uniprot_member"].setdefault(uniprot, [])
                if pc_id not in db["protein_complexes"]["by_uniprot_member"][uniprot]:
                    db["protein_complexes"]["by_uniprot_member"][uniprot].append(pc_id)

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------
    for rxn in pv.findall(".//reactions/reaction"):  # type: ignore[union-attr]
        # reactions might be under connections or visualization-elements
        pass

    # Reactions appear in pathway/reactions or in connections
    for rxn in (
        list(pv.findall(".//reactions/reaction"))
        + list(root.findall(".//reaction"))
    ):
        rxn_id = _int(rxn.find("id"))
        if rxn_id is None:
            continue
        if str(rxn_id) in db["reactions"]["by_id"]:
            continue
        direction = _text(rxn.find("direction"))
        left: List[Dict] = []
        right: List[Dict] = []
        enzymes: List[int] = []
        for le in rxn.findall(".//reaction-left-elements/reaction-left-element"):
            el_id = _int(le.find("element-id"))
            el_type = _text(le.find("element-type"))
            stoich = _int(le.find("stoichiometry")) or 1
            if el_id is not None:
                left.append({"element_id": el_id, "element_type": el_type, "stoichiometry": stoich})
        for re_ in rxn.findall(".//reaction-right-elements/reaction-right-element"):
            el_id = _int(re_.find("element-id"))
            el_type = _text(re_.find("element-type"))
            stoich = _int(re_.find("stoichiometry")) or 1
            if el_id is not None:
                right.append({"element_id": el_id, "element_type": el_type, "stoichiometry": stoich})
        for re_ in rxn.findall(".//reaction-enzymes/reaction-enzyme"):
            pc_id = _int(re_.find("protein-complex-id"))
            if pc_id is not None:
                enzymes.append(pc_id)
        db["reactions"]["by_id"][str(rxn_id)] = {
            "direction": direction, "left": left, "right": right, "enzymes": enzymes,
        }
        # Index by sorted element IDs for matching
        left_ids = sorted(str(e["element_id"]) for e in left)
        right_ids = sorted(str(e["element_id"]) for e in right)
        key = ",".join(left_ids) + "→" + ",".join(right_ids)
        db["reactions"]["by_elements"][key] = rxn_id


def build_db(pwml_dir: Path) -> Dict[str, Any]:
    db: Dict[str, Any] = {
        "compounds": {"hmdb": {}, "kegg": {}, "chebi": {}, "pubchem": {}, "drugbank": {}, "by_id": {}},
        "proteins": {"uniprot": {}, "by_id": {}},
        "protein_complexes": {"by_id": {}},
        "element_collections": {"by_id": {}},
        "biological_states": {"by_id": {}, "by_location_name": {}, "by_location_id": {}},
        "reactions": {"by_id": {}, "by_elements": {}},
        "subcellular_locations": {"by_id": {}, "by_name": {}},
    }
    pwml_files = sorted(pwml_dir.glob("*.pwml"))
    if not pwml_files:
        print(f"WARNING: no .pwml files found in {pwml_dir}", file=sys.stderr)
    for f in pwml_files:
        print(f"  Parsing {f.name}...")
        parse_pwml(f, db)

    print(f"  Compounds: {len(db['compounds']['by_id'])}")
    print(f"  Proteins:  {len(db['proteins']['by_id'])}")
    print(f"  ProteinComplexes: {len(db['protein_complexes']['by_id'])}")
    print(f"  ElementCollections: {len(db['element_collections']['by_id'])}")
    print(f"  BiologicalStates: {len(db['biological_states']['by_id'])}")
    print(f"  Reactions: {len(db['reactions']['by_id'])}")
    return db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pwml-dir", default=".", help="Directory containing .pwml files")
    ap.add_argument("--out", default="pathwhiz_id_db.json", help="Output JSON path")
    args = ap.parse_args()
    db = build_db(Path(args.pwml_dir))
    out = Path(args.out)
    out.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
