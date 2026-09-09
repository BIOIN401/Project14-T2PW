"""Is the correction round PROVABLY comment-and-docstring only?

Compares the two revisions' ASTs with every docstring blanked. If the dumps are
equal, no executable construct changed -- stronger than reading a diff.
"""
import ast, subprocess, sys

REPO = r"C:/t/c121"
PATH = "src/t2pw/pipeline/process_normalizer.py"
A, B = sys.argv[1], sys.argv[2]

def blank_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body[0].value.value = "<docstring>"
    return tree

dumps = []
for rev in (A, B):
    src = subprocess.run(["git", "-C", REPO, "show", f"{rev}:{PATH}"],
                         capture_output=True, check=True).stdout.decode("utf-8")
    dumps.append(ast.dump(blank_docstrings(ast.parse(src)), annotate_fields=True))

print(f"{PATH}")
print(f"  {A} vs {B}")
print(f"  AST with docstrings blanked: {'IDENTICAL' if dumps[0]==dumps[1] else 'DIFFERS'}")
if dumps[0] != dumps[1]:
    sys.exit(1)
