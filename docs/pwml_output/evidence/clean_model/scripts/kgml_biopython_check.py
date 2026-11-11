"""Parse every KGML file written by kgml_all.py with Biopython's own KGML parser and report successes.

Run with the SCRATCH venv (biopython is installed only there, never in the project):
  C:/t/ev-cm/venv/Scripts/python.exe -I -X utf8 kgml_biopython_check.py [--dir C:/t/ev-cm/kgml_out]
Writes kgml_biopython.json next to this script's parent folder (evidence/clean_model/).
"""

import argparse
import glob
import json
import os
from collections import Counter

from Bio.KEGG.KGML.KGML_parser import read

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="C:/t/ev-cm/kgml_out")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.kgml")))
    ok, fails, tot = 0, [], Counter()
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                p = read(fh)
            ok += 1
            tot["entries"] += len(p.entries)
            tot["reactions"] += len(p.reactions)
            tot["relations"] += len(p.relations)
            tot["reaction_entries_type_reaction"] += sum(1 for e in p.entries.values() if e.type == "reaction")
        except Exception as e:  # report, never hide
            fails.append({"file": os.path.basename(f), "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
    res = {"files": len(files), "parsed_ok": ok, "failed": len(fails), "totals": dict(tot), "failures": fails[:20]}
    with open(os.path.join(HERE, "..", "kgml_biopython.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}), fails[:3])


if __name__ == "__main__":
    main()
