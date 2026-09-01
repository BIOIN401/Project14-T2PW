"""C-107 section 2: every production caller of the functions this card changes.

C-105 round 1's whole failure was that its guard reached FOUR production callers
and its card named one. This derives the list mechanically -- by AST, over
``src/`` -- rather than by grep, and it walks the call graph outward from the
changed symbols until it reaches module-level entry points.

Usage::  <python> c107_caller_enumeration.py <repo-root>
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
SRC = REPO / "src"

#: The symbols C-107 changes, and the two functions that read them.
CHANGED = {
    "_ROLE_CUE_RES", "_ROLE_FAMILY_BY_ROLE", "_NON_ENZYME_ASE_WORDS",
    "_ENZYME_NOUN_RE_SRC", "_ANY_ROLE_CUE_RE", "_CATALYSIS_CONTRA_RE",
    "_ACTIVITY_ATTENUATION_SRC", "_ATTENUATION_STEM_SRC",
    "_ATTENUATION_OBJECT_SRC", "_PASSIVE_AGENT_VERBS_SRC",
    "_PASSIVE_AGENT_MODIFIERS_SRC", "_PASSIVE_AGENT_MAX_MODIFIERS",
    "_span_licenses_actor", "_actor_role_family",
}

#: The chain outward from them, each name being looked up as a callee.
CHAIN = [
    "_span_licenses_actor",
    "_actor_role_family",
    "_unevidenced_actor_role_rejection",
    "_should_accept",
    "apply_patch_with_policy",
    "apply_audit_patch_payload",
    "run_apply",
]

files = sorted(p for p in SRC.rglob("*.py"))
trees = {}
for p in files:
    try:
        trees[p] = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except SyntaxError:
        continue


def callsites(name):
    """(file, line, enclosing def) for every Call whose func spells `name`."""
    hits = []
    for p, tree in trees.items():
        stack = []

        class V(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                f = node.func
                spelled = getattr(f, "id", None) or getattr(f, "attr", None)
                if spelled == name:
                    hits.append((p.relative_to(REPO).as_posix(), node.lineno,
                                 stack[-1] if stack else "<module>"))
                self.generic_visit(node)

        V().visit(tree)
    return sorted(hits)


print("=" * 78)
print("C-107 -- CALL GRAPH OUTWARD FROM THE CHANGED SYMBOLS  (src/ only, by AST)")
print("=" * 78)
print(f"changed symbols: {len(CHANGED)}")
for name in sorted(CHANGED):
    print(f"    {name}")

for name in CHAIN:
    hits = callsites(name)
    print(f"\n{name}()  --  {len(hits)} call site(s) in src/")
    for rel, line, fn in hits:
        mark = "  <-- SAME MODULE" if rel.endswith("curation/apply_audit_patch.py") else "  <-- PRODUCTION CALLER"
        print(f"    {rel}:{line}  in {fn}(){mark}")

print()
print("=" * 78)
print("PRODUCTION ENTRY POINTS THAT REACH THE GUARD")
print("=" * 78)
entries = set()
for name in ("apply_patch_with_policy", "run_apply", "apply_audit_patch_payload"):
    for rel, line, fn in callsites(name):
        if rel.endswith("curation/apply_audit_patch.py"):
            continue
        entries.add((rel, line, fn, name))
for rel, line, fn, name in sorted(entries):
    print(f"    {rel}:{line}  {fn}() -> {name}()")
print(f"\n    distinct call sites outside the module: {len(entries)}")
print(f"    distinct modules                      : "
      f"{len({e[0] for e in entries})}")
