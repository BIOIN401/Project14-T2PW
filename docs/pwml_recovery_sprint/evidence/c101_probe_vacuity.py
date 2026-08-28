"""C-101 round 3: is the seam actually exercised by the two tests? ASCII only."""
import dataclasses
import os
import sys

TREE = sys.argv[1]
os.chdir(TREE)

from t2pw.bench.goldset import ForbiddenIdentifier, load_gold_set, pinned_gold_set_path
from t2pw.bench.semantic import CHECK_ID_CONFLICT, validate_semantic_coverage

gold = {c.paper_id: c for c in load_gold_set(pinned_gold_set_path()).cases}
lipid_a = gold["PMC12444477"]

P1 = ("false_real_identifier", "placeholder_claims_real_identity")


def sentinel_row(name="Unknown", **over):
    row = {
        "name": name, "species": "Arabidopsis thaliana", "species_id": 4,
        "pathbank_protein_id": 9659, "uniprot_id": "Unknown",
        "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
        "identity_status": "placeholder",
        "mapping_meta": {
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "identity_status": "placeholder", "pathbank_protein_id": 9659,
            "fallback_used": True, "cross_species_placeholder": True,
        },
    }
    row.update(over)
    return row


def payload(rows):
    return {"entities": {"compounds": [], "proteins": rows, "protein_complexes": []},
            "processes": {}}


def findings(case, row):
    r = validate_semantic_coverage(case, payload([row]), mode="strict")
    return [f for f in r.checks[CHECK_ID_CONFLICT].findings if f["kind"] in P1]


print("=" * 72)
print("A. THE CURRENT TESTS, under the REAL gold case")
print("=" * 72)
print("'Unknown' in PMC12444477 forbidden_identifiers:",
      lipid_a.forbidden_match("Unknown") is not None)
for ns, val in (("kegg", "K00912"), ("chebi", "CHEBI:16856"), ("hmdb", "HMDB0000122")):
    row = sentinel_row()
    row["mapped_ids"][ns] = val
    fs = findings(lipid_a, row)
    print("  sentinel+%-6s -> %d P1 finding(s) %s   all(t=='') is %s"
          % (ns, len(fs), [f["kind"] for f in fs], "VACUOUS" if not fs else "real"))

forged = sentinel_row()
forged["mapped_ids"] = {"uniprot": "P0A6T1", "pathbank_protein_id": 9659}
forged["uniprot_id"] = "P0A6T1"
shapes = [sentinel_row(), forged, sentinel_row(name="LpxA product"),
          {"name": "Unknown", "identity_status": "placeholder",
           "mapping_meta": {"identity_status": "placeholder", "fallback_used": True}}]
for ns, val in (("kegg", "K00912"), ("chebi", "CHEBI:16856")):
    r = sentinel_row()
    r["mapped_ids"][ns] = val
    shapes.append(r)
seen, by_kind = 0, {}
for row in shapes:
    for f in findings(lipid_a, row):
        seen += 1
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
print("  `seen` across the shapes:", seen, by_kind)
print("  false_real_identifier count (the ONLY kind that calls the seam):",
      by_kind.get("false_real_identifier", 0))

print()
print("=" * 72)
print("B. THE CONSTRUCTION THAT ACTUALLY REACHES THE SEAM")
print("=" * 72)
synth = dataclasses.replace(
    lipid_a,
    forbidden_identifiers=lipid_a.forbidden_identifiers + (
        ForbiddenIdentifier(
            name="Unknown", kind="heading_or_prose",
            reason="SYNTHETIC, test-only: makes the row reach the false_real branch.",
        ),),
)
print("synthetic case: 'Unknown' forbidden ->", synth.forbidden_match("Unknown") is not None)
print("sentinel licence still declared      ->",
      synth.unknown_backed_tolerated_sentinel is not None)
for label, row in (("bare sentinel", sentinel_row()),):
    fs = findings(synth, row)
    print("  %-22s -> %d P1 finding(s) %s" % (label, len(fs), [f["kind"] for f in fs]))
for ns, val in (("kegg", "K00912"), ("chebi", "CHEBI:16856"), ("hmdb", "HMDB0000122"),
                ("drugbank", "DB00114"), ("pubchem", "5793")):
    row = sentinel_row()
    row["mapped_ids"][ns] = val
    fs = findings(synth, row)
    tol = [f.get("contract_tolerance", "<absent>") for f in fs]
    print("  sentinel+%-9s -> %d finding(s) %s  contract_tolerance=%s"
          % (ns, len(fs), [f["kind"] for f in fs], tol))
print()
print("READ: at this tip every contract_tolerance must be ''. Under round 0's")
print("_REAL_ACCESSION guard these same rows returned 'pathbank_unknown_sentinel'.")
print("PROBE OK")
