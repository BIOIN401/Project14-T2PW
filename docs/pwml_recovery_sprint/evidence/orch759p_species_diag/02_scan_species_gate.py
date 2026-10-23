"""Stdlib-only: scan run trees for species_missing_taxonomy failures.

For each required-field gate report (or RESULT.txt) naming species_missing_*,
print the leg path, the failing species names, and the leg's species rows with
their taxonomy ids (from final_mapped.json when present).
"""
import json
import os
import re
import sys

ROOT = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
TREES = sys.argv[1:] or ["runs_papers", "runs_verify", "runs_validation", "runs_smoke"]
NAME_RE = re.compile(r"No taxonomy ID found for species '([^']*)'")

hits = {}
for tree in TREES:
    base = os.path.join(ROOT, tree)
    for dirpath, dirnames, filenames in os.walk(base):
        for fn in filenames:
            if fn not in ("pwml_required_field_gate_report.json", "RESULT.txt"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if "species_missing_taxonomy" not in text:
                continue
            names = sorted(set(NAME_RE.findall(text)))
            hits.setdefault(dirpath, set()).update(names)

for leg in sorted(hits):
    rel = os.path.relpath(leg, ROOT).replace("\\", "/")
    print("LEG", rel)
    print("   failing:", sorted(hits[leg]))
    fm = os.path.join(leg, "final_mapped.json")
    if os.path.exists(fm):
        try:
            d = json.load(open(fm, encoding="utf-8"))
            rows = d.get("entities", {}).get("species", [])
            for r in rows:
                if isinstance(r, dict):
                    print("   row:", repr(r.get("name")), "tax=", r.get("taxonomy_id"),
                          "cls=", r.get("classification"), "pb=", r.get("pathbank_species_id"))
        except Exception as exc:  # noqa: BLE001
            print("   final_mapped unreadable:", exc)
print("TOTAL legs:", len(hits))
