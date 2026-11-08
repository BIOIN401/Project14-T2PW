"""Apply one mutation to the extraction module of the scratch copy, or restore it.

    mutate.py restore
    mutate.py additions-empty | raw-merge | export-mode-none
"""
import re
import sys
from pathlib import Path

TARGET = Path("C:/t/wsR26tmp/mut/src/t2pw/service/extraction.py")
ORIG = Path("C:/t/wsR26tmp/mut_extraction.orig")

MUTATIONS = {
    "additions-empty": (r"(final_payload = merge_additions\(\r?\n\s+stage_one_in_scope,\r?\n\s+)stage_two if isinstance\(stage_two, dict\) else \{\},",
                        r"\1{},"),
    "raw-merge": (r"(final_payload = merge_additions\(\r?\n\s+)stage_one_in_scope,(\r?\n\s+stage_two if isinstance)",
                  r"\1stage_one,\2"),
    "export-mode-none": (r'state\["export_mode"\] = request\.export_mode', 'state["export_mode"] = None'),
}

name = sys.argv[1]
if name == "restore":
    TARGET.write_bytes(ORIG.read_bytes())
    print("restored")
    sys.exit(0)
if not ORIG.exists():
    ORIG.write_bytes(TARGET.read_bytes())
text = ORIG.read_bytes().decode("utf-8")
pat, rep = MUTATIONS[name]
new, n = re.subn(pat, rep, text)
assert n == 1, (name, n)
TARGET.write_bytes(new.encode("utf-8"))
print("mutated", name)
