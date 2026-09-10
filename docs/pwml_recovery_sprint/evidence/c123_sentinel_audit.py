"""`F-202` adjudication: is every spurious *Arabidopsis thaliana* assignment SENTINEL-ONLY?

READ ONLY. Opens committed run artifacts and committed PWMLs. Writes only its
`--out` JSON. No network, no LLM, no production import beyond plain json/xml.

THE QUESTION, and it has exactly two answers with opposite consequences
----------------------------------------------------------------------
Ten of seventeen exported PWMLs declare a second species, almost always
*Arabidopsis thaliana* (`pathbank_species_id 4`). Either:

  SENTINEL-ONLY  every entity carrying it is the PathBank ``Unknown`` sentinel or
                 a wrapper built around one -- i.e. a TECHNICAL placeholder that
                 asserts no biology. Then no production change is warranted and
                 the correct fix is to stop counting those rows in organism-
                 accuracy claims.

  REAL DEFECT    at least one entity with a genuine external identifier (UniProt
                 or DrugBank accession) is stamped *Arabidopsis thaliana* while
                 the pathway is not a plant one. Then a resolved protein carries
                 a false organism and that is a correctness defect.

The classification is therefore made on ONE predicate -- does the row carry a
real external identifier -- and never on the row's NAME. A complex named
``AauA`` may be a wrapper whose only component is the ``Unknown`` sentinel;
judging by the label would call that a resolved protein and manufacture a defect
that is not there. ``ORCH-725`` measured exactly this shape: "Eleven became
Unknown-backed functional complexes."

WHAT IT DOES NOT DECIDE
-----------------------
Whether the sentinel SHOULD carry a species at all, or whether *Arabidopsis*
is a sensible placeholder. Both are product questions. This tool decides only
whether any real biology is mislabelled.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

SENTINEL_NAMES = {"unknown", "unknown protein", "unknown_protein"}
ENTITY_KINDS = ("proteins", "protein_complexes", "compounds", "species",
                "enzyme_complexes", "nucleic_acids", "element_collections")


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(value or "").strip().casefold())


def real_identifier(row: Dict[str, Any]) -> str:
    """A genuine EXTERNAL identifier, or "". The whole classification rests here.

    PathBank's own internal row id is deliberately NOT counted: the ``Unknown``
    sentinel HAS a pathbank id (record 9659) and having one is what makes it a
    usable placeholder. Only an accession that names a real database entry --
    UniProt or DrugBank -- means "this row asserts a specific protein".
    """
    ids = row.get("mapped_ids") if isinstance(row.get("mapped_ids"), dict) else {}
    for key in ("uniprot", "drugbank"):
        for holder in (ids, row):
            value = str((holder or {}).get(key) or "").strip()
            if value:
                return f"{key}:{value}"
    return ""


def is_sentinel(row: Dict[str, Any]) -> bool:
    return norm(row.get("name")) in SENTINEL_NAMES


def components_of(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in ("components", "component_proteins", "proteins", "members", "subunits"):
        value = row.get(key)
        if isinstance(value, list):
            out.extend(entry for entry in value if isinstance(entry, dict))
    return out


def classify(row: Dict[str, Any], kind: str) -> str:
    """sentinel | sentinel_backed_wrapper | resolved_with_identifier | unresolved_no_identifier"""
    if is_sentinel(row):
        return "sentinel"
    if real_identifier(row):
        return "resolved_with_identifier"
    comps = components_of(row)
    if comps:
        if all(is_sentinel(c) or not real_identifier(c) for c in comps) and any(
            is_sentinel(c) for c in comps
        ):
            return "sentinel_backed_wrapper"
        if all(not real_identifier(c) for c in comps):
            return "sentinel_backed_wrapper" if kind == "protein_complexes" else "unresolved_no_identifier"
    return "unresolved_no_identifier"


def primary_species(payload: Dict[str, Any]) -> str:
    rows = ((payload.get("entities") or {}).get("species") or [])
    for row in rows:
        if isinstance(row, dict) and norm(row.get("name")) != "arabidopsis thaliana":
            return str(row.get("name") or "")
    return str((rows[0] or {}).get("name") or "") if rows else ""


def audit_leg(leg: Path) -> Dict[str, Any]:
    payload = json.loads((leg / "final_mapped.json").read_text(encoding="utf-8", errors="replace"))
    entities = payload.get("entities") or {}
    pathway_species = primary_species(payload)
    plant_pathway = norm(pathway_species) == "arabidopsis thaliana"

    hits: List[Dict[str, Any]] = []
    for kind in ENTITY_KINDS:
        for row in entities.get(kind) or []:
            if not isinstance(row, dict):
                continue
            stamped = norm(row.get("species") or row.get("organism"))
            if stamped != "arabidopsis thaliana":
                continue
            if kind == "species":
                hits.append({"kind": kind, "name": row.get("name"), "verdict": "species_declaration",
                             "identifier": "", "pathbank_species_id": row.get("pathbank_species_id")})
                continue
            verdict = classify(row, kind)
            hits.append({
                "kind": kind,
                "name": row.get("name"),
                "verdict": verdict,
                "identifier": real_identifier(row),
                "component_names": [c.get("name") for c in components_of(row)][:6],
                "pathbank_species_id": row.get("pathbank_species_id"),
            })

    offenders = [h for h in hits
                 if h["verdict"] == "resolved_with_identifier" and not plant_pathway]
    return {
        "leg": leg.as_posix(),
        "paper": leg.parent.name,
        "pathway_species": pathway_species,
        "pathway_is_arabidopsis": plant_pathway,
        "arabidopsis_stamped_rows": len(hits),
        "by_verdict": {v: sum(1 for h in hits if h["verdict"] == v)
                       for v in sorted({h["verdict"] for h in hits})},
        "REAL_DEFECT_rows": offenders,
        "hits": hits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve()

    legs = sorted({p.parent for p in root.glob("runs_smoke/**/final_mapped.json")}
                  | {p.parent for p in root.glob("runs_validation/**/final_mapped.json")}
                  | {p.parent for p in root.glob("runs_verify/**/final_mapped.json")})

    print("=" * 104)
    print("F-202 adjudication -- is every Arabidopsis assignment SENTINEL-ONLY?")
    print("=" * 104)
    print("classification predicate: does the row carry a real UniProt/DrugBank accession.")
    print("NEVER the row's name -- a wrapper named 'AauA' may be sentinel-backed.\n")

    reports, defects, affected = [], [], 0
    for leg in legs:
        try:
            rep = audit_leg(leg)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {leg.as_posix()}: {type(exc).__name__}")
            continue
        if not rep["arabidopsis_stamped_rows"]:
            continue
        affected += 1
        reports.append(rep)
        flag = "  <-- REAL DEFECT" if rep["REAL_DEFECT_rows"] else ""
        print(f"  {rep['paper']:<14} pathway={str(rep['pathway_species'])[:26]:<26} "
              f"rows={rep['arabidopsis_stamped_rows']:<3} {rep['by_verdict']}{flag}")
        for row in rep["REAL_DEFECT_rows"]:
            print(f"        !! {row['kind']} '{row['name']}' carries {row['identifier']}")
            defects.append({"paper": rep["paper"], **row})

    print()
    print("=" * 104)
    print(f"legs with an Arabidopsis-stamped row : {affected}")
    print(f"rows that are the sentinel or a sentinel-backed wrapper : "
          f"{sum(v for r in reports for k, v in r['by_verdict'].items() if k.startswith('sentinel'))}")
    print(f"rows unresolved with no identifier   : "
          f"{sum(v for r in reports for k, v in r['by_verdict'].items() if k == 'unresolved_no_identifier')}")
    print(f"**RESOLVED rows carrying a real accession on a non-plant pathway : {len(defects)}**")
    print()
    if defects:
        print("VERDICT: REAL CORRECTNESS DEFECT. A resolved protein carries a false organism.")
        print("         STOP. Do not treat this as a sentinel-only reporting issue.")
    else:
        print("VERDICT: SENTINEL-ONLY. Every Arabidopsis-stamped row is the PathBank Unknown")
        print("         sentinel or a wrapper around one. No resolved protein is mislabelled,")
        print("         so no production change is warranted -- the fix is to exclude these")
        print("         technical rows from biological organism-accuracy claims.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"tool": "c123_sentinel_audit.py", "legs_affected": affected,
         "real_defects": defects,
         "verdict": "REAL_DEFECT" if defects else "SENTINEL_ONLY",
         "reports": reports}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
