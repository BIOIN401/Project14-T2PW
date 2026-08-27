"""C-096 -- prove the REV-096 corrections changed no executable code.

The orchestrator's instruction not to re-run SMOKE, the affected sweep and
Chunk D rests entirely on the corrections being docstring/comment only. That is
a claim about the diff, so it is proved here rather than asserted: both
revisions of ``src/t2pw/pwml/compound_resolution.py`` are parsed, every
docstring is stripped from every module/class/function body, and the two
abstract syntax trees are compared.

Comments never reach the AST at all, so a comment edit is invisible by
construction; stripping docstrings removes the other half of what a "comment
only" change is allowed to touch. Anything left that differs is executable.

    <py> c096_comment_only_probe.py <sha-before> <sha-after>
"""

from __future__ import annotations

import ast
import subprocess
import sys

PATH = "src/t2pw/pwml/compound_resolution.py"
BEFORE, AFTER = sys.argv[1], sys.argv[2]


def _blob(sha: str) -> str:
    return subprocess.run(
        ["git", "show", f"{sha}:{PATH}"],
        capture_output=True, text=True, check=True,
    ).stdout


def _strip_docstrings(tree: ast.AST) -> ast.AST:
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
    return tree


before_src, after_src = _blob(BEFORE), _blob(AFTER)
before_ast = _strip_docstrings(ast.parse(before_src))
after_ast = _strip_docstrings(ast.parse(after_src))

before_dump = ast.dump(before_ast, annotate_fields=True)
after_dump = ast.dump(after_ast, annotate_fields=True)

print(f"path            : {PATH}")
print(f"before          : {BEFORE}  ({len(before_src.splitlines())} lines)")
print(f"after           : {AFTER}  ({len(after_src.splitlines())} lines)")
print(f"raw text equal  : {before_src == after_src}")
print(f"EXECUTABLE AST EQUAL AFTER STRIPPING DOCSTRINGS : {before_dump == after_dump}")

if before_dump != after_dump:
    print("!! the change is NOT comment-only")
    raise SystemExit(1)

# Non-vacuity: the instrument must be able to SEE a real executable change.
# Flip the `elif` to an `if` in the after-text and confirm the comparison goes red.
mutated = after_src.replace("    elif db_resolver is None:", "    if db_resolver is None:", 1)
assert mutated != after_src, "control mutation did not apply; this probe is vacuous"
mutated_dump = ast.dump(_strip_docstrings(ast.parse(mutated)), annotate_fields=True)
print(f"control (elif -> if) detected as different       : {mutated_dump != after_dump}")
if mutated_dump == after_dump:
    raise SystemExit(1)
