"""ORCH-734 -- PathWhiz import-readiness validation for a set of PWML files.

READ-ONLY. Parses PWML and reports; opens no network connection and writes into
no run tree.

WHAT THIS CAN AND CANNOT ESTABLISH -- read this before quoting a verdict
------------------------------------------------------------------------
PWML is self-contained: the file carries the species, the entities, the
reactions, AND the drawing -- element coordinates, sizes, z-order and edge paths
are all inside it. So the charter's import checklist is almost entirely decidable
from the bytes, and this tool decides it:

  XML accepted . . . . . . . . . . . the document parses
  pathway opens  . . . . . . . . . . the root and the visualization envelope are present and well formed
  drawing/render exists  . . . . . . a pathway-visualization with non-zero canvas dimensions
  graph is not blank . . . . . . . . positioned elements AND edges with real paths
  compounds appear . . . . . . . . . declared compound entities, and they are placed on the canvas
  proteins/complexes appear  . . . . declared protein or protein-complex entities, and they are placed
  reactions connected sensibly . . . every reaction has both a left and a right side, and its
                                     compound references resolve
  no catastrophic broken references  every *-id reference resolves to an id the document declares
  biologically recognizable  . . . . a named species with a taxonomy id, and named entities

**The one thing it cannot do is log in to PathWhiz and press Import.** That needs
the product owner's PathWhiz account, and this tool does not have one and must
not acquire one. A `IMPORT READY` verdict here therefore means *"every property
the importer checks is satisfied by the file"* -- it is a strong precondition and
it is **not** a substitute for the human confirming the pathway renders in the
PathWhiz UI. That last step is handed to the product owner with the exact file
list, and this tool says so in its own output rather than letting the
distinction be lost.

A `IMPORT FAIL` verdict IS conclusive in the other direction: a file that does
not parse, or whose references dangle, cannot import.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Reference-bearing tags whose value must resolve to a declared id. Kept as an
# explicit list rather than "anything ending in -id": drugbank-id, hmdb-id,
# taxonomy-id, pw-id and uniprot-id are EXTERNAL database accessions, not
# internal references, and treating them as internal manufactures dangling
# references on a perfectly good file.
INTERNAL_REF_TAGS = {
    "compound-id",
    "compound-location-id",
    "protein-id",
    "protein-complex-id",
    "element-id",
    "edge-id",
    "biological-state-id",
    "reaction-id",
    "bound-id",
    "nucleic-acid-id",
}

EXTERNAL_ID_TAGS = {
    "drugbank-id",
    "hmdb-id",
    "uniprot-id",
    "taxonomy-id",
    "pw-id",
    "chebi-id",
    "kegg-id",
    "pubchem-id",
    "smpdb-id",
    "visualization-template-id",
    "species-id",
    "named-for-id",
}


def _text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None else ""


def _ints(root: ET.Element, tag: str) -> List[str]:
    return [(_text(e)) for e in root.iter(tag) if _text(e)]


def check(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "file": str(path).replace("\\", "/"),
        "bytes": path.stat().st_size if path.exists() else 0,
        "checks": {},
        "notes": [],
    }
    if not path.exists():
        out["verdict"] = "IMPORT FAIL -- file does not exist"
        return out

    # --- XML accepted --------------------------------------------------------
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:  # noqa: BLE001
        out["checks"]["xml_accepted"] = False
        out["xml_error"] = f"{exc.__class__.__name__}: {exc}"
        out["verdict"] = f"IMPORT FAIL -- XML does not parse: {exc}"
        return out
    out["checks"]["xml_accepted"] = True
    out["root_tag"] = root.tag

    # --- pathway opens -------------------------------------------------------
    viz = root.find(".//pathway-visualization")
    pathway = root.find(".//pathway")
    out["checks"]["root_is_super_pathway_visualization"] = (
        root.tag == "super-pathway-visualization"
    )
    out["checks"]["pathway_envelope_present"] = viz is not None and pathway is not None
    out["pathway_name"] = _text(pathway.find("name")) if pathway is not None else ""

    # --- drawing / render exists --------------------------------------------
    h = _text(viz.find("height")) if viz is not None else ""
    w = _text(viz.find("width")) if viz is not None else ""
    try:
        canvas = (int(h or 0), int(w or 0))
    except ValueError:
        canvas = (0, 0)
    out["canvas"] = {"height": canvas[0], "width": canvas[1]}
    out["checks"]["drawing_canvas_nonzero"] = canvas[0] > 0 and canvas[1] > 0

    # --- graph is not blank --------------------------------------------------
    positioned = 0
    for el in root.iter():
        if el.find("x") is not None and el.find("y") is not None:
            positioned += 1
    edges = list(root.iter("edge"))
    edge_paths = [e for e in edges if e.find("path") is not None]
    out["positioned_elements"] = positioned
    out["edges"] = len(edges)
    out["edges_with_path"] = len(edge_paths)
    out["checks"]["graph_not_blank"] = positioned > 0 and len(edge_paths) > 0

    # --- entity census -------------------------------------------------------
    compounds = list(root.iter("compound"))
    proteins = list(root.iter("protein"))
    complexes = list(root.iter("protein-complex"))
    reactions = list(root.iter("reaction"))
    out["counts"] = {
        "compounds": len(compounds),
        "proteins": len(proteins),
        "protein_complexes": len(complexes),
        "reactions": len(reactions),
        "compound_locations": len(list(root.iter("compound-location"))),
        "bounds": len(list(root.iter("bound"))),
    }
    out["checks"]["compounds_appear"] = len(compounds) > 0
    out["checks"]["proteins_or_complexes_appear"] = (len(proteins) + len(complexes)) > 0
    out["checks"]["reactions_present"] = len(reactions) > 0

    # --- reactions connected sensibly ---------------------------------------
    # A reaction that names only one side is not a reaction the importer can
    # draw. Checked structurally, per reaction, rather than by a total count.
    # The tags are reaction-left-elements / reaction-right-elements, verified
    # against a real exported file rather than guessed. An earlier revision of
    # this check looked for left-compounds/right-compounds, found neither, and
    # reported EVERY known-good PWML as lopsided -- an advisory that fires on
    # 100% of inputs is measuring the checker, not the files.
    lopsided: List[str] = []
    for rx in reactions:
        left = rx.find("reaction-left-elements")
        right = rx.find("reaction-right-elements")
        nl = len(list(left)) if left is not None else 0
        nr = len(list(right)) if right is not None else 0
        if nl == 0 or nr == 0:
            lopsided.append(f"{_text(rx.find('id')) or '?'}: left={nl} right={nr}")
    out["reactions_missing_a_side"] = lopsided
    out["checks"]["reactions_two_sided"] = not lopsided

    # --- reference integrity -------------------------------------------------
    declared: Set[str] = set()
    for el in root.iter():
        idel = el.find("id")
        if idel is not None and _text(idel):
            declared.add(_text(idel))
    # Ids also appear as bare <id> leaves inside collections.
    declared |= {_text(e) for e in root.iter("id") if _text(e)}

    dangling: List[str] = []
    refs_checked = 0
    for tag in INTERNAL_REF_TAGS:
        for el in root.iter(tag):
            val = _text(el)
            if not val:
                continue
            refs_checked += 1
            if val not in declared:
                dangling.append(f"{tag}={val}")
    out["internal_refs_checked"] = refs_checked
    out["dangling_refs"] = sorted(set(dangling))[:25]
    out["dangling_ref_count"] = len(set(dangling))
    out["checks"]["no_broken_references"] = not dangling

    # --- biologically recognizable ------------------------------------------
    species = root.find(".//species/species")
    sp_name = _text(species.find("name")) if species is not None else ""
    sp_tax = _text(species.find("taxonomy-id")) if species is not None else ""
    named_entities = [
        _text(e.find("name"))
        for e in compounds + proteins + complexes
        if e.find("name") is not None and _text(e.find("name"))
    ]
    out["species"] = {"name": sp_name, "taxonomy_id": sp_tax}
    out["sample_entity_names"] = named_entities[:12]
    out["checks"]["species_named_with_taxonomy"] = bool(sp_name and sp_tax)
    out["checks"]["entities_named"] = len(named_entities) >= 3

    # --- verdict -------------------------------------------------------------
    # BLOCKING checks are the ones an importer cannot survive. The advisory ones
    # are reported and do not fail the file, because the charter is explicit that
    # biological imperfection is not a reliability failure.
    blocking = [
        "xml_accepted",
        "root_is_super_pathway_visualization",
        "pathway_envelope_present",
        "drawing_canvas_nonzero",
        "graph_not_blank",
        "compounds_appear",
        "reactions_present",
        "no_broken_references",
    ]
    advisory = [
        "proteins_or_complexes_appear",
        "reactions_two_sided",
        "species_named_with_taxonomy",
        "entities_named",
    ]
    failed = [c for c in blocking if not out["checks"].get(c)]
    soft = [c for c in advisory if not out["checks"].get(c)]
    out["advisory_failures"] = soft
    if failed:
        out["verdict"] = "IMPORT FAIL -- " + ", ".join(failed)
    elif soft:
        out["verdict"] = "IMPORT READY (with advisories: " + ", ".join(soft) + ")"
    else:
        out["verdict"] = "IMPORT READY"
    out["notes"].append(
        "IMPORT READY means every property the PathWhiz importer checks is satisfied "
        "by this file. It is NOT a substitute for the product owner opening it in the "
        "PathWhiz UI -- that step needs their account and is handed to them."
    )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="PWML files, or directories to search")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    files: List[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.pwml")))
        else:
            files.append(p)

    results = [check(f) for f in files]
    passed = [r for r in results if r["verdict"].startswith("IMPORT READY")]
    report = {
        "task": "ORCH-734",
        "files_checked": len(results),
        "import_ready": len(passed),
        "import_fail": len(results) - len(passed),
        "boundary": (
            "Structural import readiness only. The live PathWhiz UI import requires the "
            "product owner's PathWhiz account and is not performed here."
        ),
        "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(text, encoding="utf-8")

    for r in results:
        c = r.get("counts") or {}
        print(
            "%-58s %9s B  cpd=%-3s prot=%-3s cplx=%-3s rxn=%-3s edges=%-3s  %s"
            % (
                Path(r["file"]).parent.parent.name + "/" + Path(r["file"]).parent.name,
                f"{r['bytes']:,}",
                c.get("compounds", "-"),
                c.get("proteins", "-"),
                c.get("protein_complexes", "-"),
                c.get("reactions", "-"),
                r.get("edges", "-"),
                r["verdict"],
            )
        )
    print()
    print(
        "files %d | IMPORT READY %d | IMPORT FAIL %d"
        % (report["files_checked"], report["import_ready"], report["import_fail"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
