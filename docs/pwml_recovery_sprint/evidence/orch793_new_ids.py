"""ORCH-793: every protein accession the tip ships that base did not, per leg (orch792_replay.py work dirs).

usage: python orch793_new_ids.py <base workdir> <tip workdir>
For a reader to audit each newly admitted identity by name, accession, organism and the verdict's reason.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
base, tip = Path(sys.argv[1]), Path(sys.argv[2])


def ids(path):
    out = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["entities"].get("proteins") or []:
        acc = (row.get("mapped_ids") or {}).get("uniprot")
        if acc:
            meta = row.get("mapping_meta") or {}
            v = meta.get("identity_verdict") or {}
            j = v.get("judged_candidate") or {}
            out[row["name"]] = (acc, j.get("protein_name"), j.get("organism"), v.get("reason"),
                                v.get("outranked_by_primary_gene_name"), v.get("collapsed_duplicate_accessions"))
    return out


for leg in sorted(p.name for p in tip.iterdir() if (p / "stage6.json").exists()):
    b, t = ids(base / leg / "stage6.json"), ids(tip / leg / "stage6.json")
    print(f"######## {leg}: base shipped {len(b)}, tip shipped {len(t)}")
    for name in sorted(set(t) - set(b)):
        print("  NEW ", name, t[name])
    for name in sorted(set(b) - set(t)):
        print("  LOST", name, b[name])
    for name in sorted(set(b) & set(t)):
        if b[name][0] != t[name][0]:
            print("  CHANGED", name, b[name][0], "->", t[name][0])
