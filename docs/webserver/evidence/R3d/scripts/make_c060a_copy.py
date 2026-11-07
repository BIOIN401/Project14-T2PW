"""Build the proposed Q-R0-1 repoint of tests/test_c060a_requested_pathway_wiring.py as an
out-of-suite proof copy (no test_ prefix). The ONLY changes the commit would carry are the
path/line-locator lines; the copy additionally pins ROOT to the tree, since it lives outside tests/."""
from pathlib import Path
import sys

tree = Path(sys.argv[1])
out = Path(sys.argv[2])
src = (tree / "tests/test_c060a_requested_pathway_wiring.py").read_text(encoding="utf-8")
edits = [
    ("OWNED_FLOOR = 1440\n",
     "#: R3d: both merge sites moved with Stage 2 into service/extraction.py (``run_stage_two``,\n"
     "#: 874/931, gate imports 872/929); no other merge_additions assignment or gate import sits\n"
     "#: in that file. Same locator, same selection: the floor is ``def run_stage_two``.\n"
     "OWNED_FLOOR = 790\n"
     "EXTRACTION_PATH = SRC / \"t2pw\" / \"service\" / \"extraction.py\"\n"),
    ("    sites = [n for n in _merge_calls(_tree(APP_PATH)) if n.lineno >= OWNED_FLOOR]\n",
     "    sites = [n for n in _merge_calls(_tree(EXTRACTION_PATH)) if n.lineno >= OWNED_FLOOR]\n"),
    ("        node for node in ast.walk(_tree(APP_PATH))\n        if isinstance(node, ast.ImportFrom) and node.module == GATE_MODULE\n",
     "        node for node in ast.walk(_tree(EXTRACTION_PATH))\n        if isinstance(node, ast.ImportFrom) and node.module == GATE_MODULE\n"),
]
for old, new in edits:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
pin_old = "ROOT = Path(__file__).resolve().parents[1]\n"
assert src.count(pin_old) == 1
proof = src.replace(pin_old, "ROOT = Path(__file__).resolve().parents[5]  # proof copy only: it lives in "
                              "docs/webserver/evidence/R3d/proposed/\n")
out.write_text(proof, encoding="utf-8", newline="\n")
(out.with_name(out.stem + ".committed-form.txt")).write_text(src, encoding="utf-8", newline="\n")
print("wrote", out)
