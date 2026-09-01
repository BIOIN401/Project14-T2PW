"""C-109: prove the `src/t2pw/rag/controller.py` edit is PROVABLY INERT.

`controller.py` is a production file and C-109's boundary allows the module docstring
ONLY -- no statement, signature, import or constant may change. "The diff looks like
prose" is not a proof, so this measures it.

Method. Take the base blob and the tip file, parse both, then strip EVERY docstring in
each tree -- module, class and function -- and compare the resulting ASTs. If the two
docstring-stripped ASTs are identical, then every statement, signature, import, default
and constant is identical, and the only difference between the two files is docstring
text. It also reports, separately, WHICH docstrings differ, so a change smuggled into a
function docstring cannot hide behind "docstring-only" either.

Usage::

    <python> c109_controller_inertness.py <worktree-root> <base-sha>
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

TARGET = "src/t2pw/rag/controller.py"

ROOT = Path(sys.argv[1]).resolve()
BASE = sys.argv[2]


def base_source() -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BASE}:{TARGET}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        sys.exit(f"cannot read {TARGET} at {BASE}: {proc.stderr.strip()}")
    return proc.stdout


def docstring_sites(tree: ast.AST) -> dict[str, str]:
    """Every docstring in the tree, keyed by the node that owns it."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                continue
            name = "<module>" if isinstance(node, ast.Module) else node.name
            key = f"{type(node).__name__}:{name}:{getattr(node, 'lineno', 0)}"
            out[key] = doc
    return out


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.fix_missing_locations(tree)


def main() -> int:
    tip_src = (ROOT / TARGET).read_text(encoding="utf-8")
    bas_src = base_source()

    print(f"target      : {TARGET}")
    print(f"base        : {BASE}")
    print(f"bytes       : base {len(bas_src.encode('utf-8'))}  tip {len(tip_src.encode('utf-8'))}")

    tip_tree = ast.parse(tip_src)
    bas_tree = ast.parse(bas_src)

    tip_docs = docstring_sites(tip_tree)
    bas_docs = docstring_sites(bas_tree)

    tip_code = ast.dump(strip_docstrings(ast.parse(tip_src)), annotate_fields=True)
    bas_code = ast.dump(strip_docstrings(ast.parse(bas_src)), annotate_fields=True)

    identical = tip_code == bas_code
    print(f"ast nodes   : base {len(list(ast.walk(bas_tree)))}  tip {len(list(ast.walk(tip_tree)))}")
    print(f"docstring-stripped AST identical : {identical}")

    # Which docstrings moved? Compare by owner name, since line numbers shift.
    def by_name(sites: dict[str, str]) -> dict[str, str]:
        return {k.rsplit(":", 1)[0]: v for k, v in sites.items()}

    b, t = by_name(bas_docs), by_name(tip_docs)
    changed = sorted(k for k in set(b) | set(t) if b.get(k) != t.get(k))
    print(f"docstring owners : base {len(b)}  tip {len(t)}")
    print(f"docstrings changed : {len(changed)} -> {changed}")

    ok = identical and changed == ["Module:<module>"]
    print(f"VERDICT : {'INERT -- module docstring only' if ok else 'NOT INERT'}")
    if not ok:
        if not identical:
            print("  the docstring-stripped ASTs DIFFER: this is a code change", file=sys.stderr)
        if changed != ["Module:<module>"]:
            print(f"  docstrings changed outside the module docstring: {changed}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
