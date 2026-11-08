"""R3d: the repointed c060a selects exactly the statements it selected at base. Applies c060a's own selection
rule (``X = merge_additions(...)`` Name-call assignments at or below OWNED_FLOOR, exactly two; then, per site,
the entity_admission ImportFrom nodes between the previous site (or the floor) and the site) to base APP with
floor 1440 and to tip EXT with floor 790, and compares the selected statements' source (ast.unparse, so the
indent of the new home does not matter).

usage: c060a_selection.py BASE_TREE TIP_TREE
"""
import ast
import sys
from pathlib import Path

GATE = "t2pw.pipeline.entity_admission"


def select(path, floor):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    sites = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name) and n.value.func.id == "merge_additions"
                    and n.lineno >= floor), key=lambda n: n.lineno)
    assert len(sites) == 2, [s.lineno for s in sites]
    out = []
    for i, site in enumerate(sites):
        lower = sites[i - 1].lineno if i else floor - 1
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == GATE
                   and lower < n.lineno < site.lineno]
        out.append(([ast.unparse(n) for n in imports], ast.unparse(site), [n.lineno for n in imports] + [site.lineno]))
    return out


base = select(Path(sys.argv[1], "src/t2pw/app/streamlit_app.py"), 1440)
tip = select(Path(sys.argv[2], "src/t2pw/service/extraction.py"), 790)
same = True
for i, ((bi, bs, bl), (ti, ts, tl)) in enumerate(zip(base, tip)):
    eq = bi == ti and bs == ts
    same &= eq
    print(f"site {i}: base lines {bl} | tip lines {tl} | {'IDENTICAL' if eq else 'DIFFERENT'}")
    print("  imports:", ti)
    print("  statement:", ts.replace("\n", " "))
print("selection identical" if same else "SELECTION DIFFERS")
sys.exit(0 if same else 1)
