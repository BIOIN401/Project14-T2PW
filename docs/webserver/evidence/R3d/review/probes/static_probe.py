"""Reviewer static probes: dead globals (function bodies included), and a normalized verbatim diff."""
import ast, difflib, re, sys
BASE = "C:/t/ws-R3d-base/src/t2pw/app/streamlit_app.py"
TIPA = "C:/t/ws-R3d/src/t2pw/app/streamlit_app.py"
TIPE = "C:/t/ws-R3d/src/t2pw/service/extraction.py"

base_text = open(BASE, encoding="utf-8").read()
tip_text = open(TIPA, encoding="utf-8").read()
base_lines = base_text.splitlines()
base_tree = ast.parse(base_text)
tip_tree = ast.parse(tip_text)

start = next(i for i, l in enumerate(base_lines) if l.strip() == "stage_one = _stage_one.stage_one")
end = next(i for i, l in enumerate(base_lines) if "session_state.pop" in l and "post_pipeline_artifacts" in l and i > start)
print("base moved range (1-based):", start + 1, end + 1)

def binds_in(tree, lo, hi):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and lo <= n.lineno <= hi:
            out.add(n.id)
        if isinstance(n, ast.FunctionDef) and lo <= n.lineno <= hi:
            out.add(n.name)
    return out

bound = binds_in(base_tree, start + 1, end + 1)
print("names bound in the moved range:", sorted(bound))

submit = next(n for n in tip_tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "submit")
submit_end = submit.end_lineno
print("tip if-submit:", submit.lineno, submit_end)
tip_bound_before_end = binds_in(tip_tree, 1, submit_end)
print("no longer bound by end of submit at tip:", sorted(bound - tip_bound_before_end))
print("still bound somewhere before end of submit at tip:", sorted(bound & tip_bound_before_end))

fn_hits = []
for fn in ast.walk(tip_tree):
    if isinstance(fn, (ast.FunctionDef, ast.Lambda)):
        local = set()
        args = fn.args
        for a in args.args + args.kwonlyargs + args.posonlyargs:
            local.add(a.arg)
        if args.vararg: local.add(args.vararg.arg)
        if args.kwarg: local.add(args.kwarg.arg)
        body = fn.body if isinstance(fn.body, list) else [fn.body]
        for b in body:
            for n in ast.walk(b):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    local.add(n.id)
        for b in body:
            for n in ast.walk(b):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in bound and n.id not in local:
                    fn_hits.append((getattr(fn, "name", "<lambda>"), n.id, n.lineno))
print("global reads of those names from function bodies:", fn_hits)

def walk_module_level(node):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield from walk_module_level(child)
        if isinstance(child, ast.Name) and child.id in bound:
            yield child

post = []
for stmt in tip_tree.body:
    if stmt.lineno > submit_end:
        for n in walk_module_level(stmt):
            post.append((n.lineno, n.col_offset, n.id, type(n.ctx).__name__))
post.sort()
first = {}
for line, col, name, ctx in post:
    first.setdefault(name, (line, ctx))
print("first post-submit module-level event per name:")
for name in sorted(first):
    print("   ", name, first[name])
print("names whose FIRST post-submit event is a Load:", sorted(n for n, (l, c) in first.items() if c == "Load"))

ready = next(n for n in tip_tree.body if isinstance(n, ast.If) and n.lineno > submit_end)
print("next top-level if after submit:", ready.lineno, ready.end_lineno, ast.unparse(ready.test)[:80])
outside = [(l, name, c) for l, col, name, c in post if l > ready.end_lineno and c == "Load"]
print("Loads after that block:", outside)

dyn = [(n.lineno, n.func.id) for n in ast.walk(tip_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"globals", "locals", "vars", "eval", "exec"}]
print("dynamic namespace callers in the app:", dyn)
cbs = sorted({(k.value.lineno, ast.unparse(k.value)[:60]) for n in ast.walk(tip_tree) if isinstance(n, ast.Call) for k in n.keywords if k.arg in ("on_click", "on_change")})
print("widget callbacks:", cbs)

ext_lines = open(TIPE, encoding="utf-8").read().splitlines()
ext_start = next(i for i, l in enumerate(ext_lines) if l.strip() == "stage_one = stage_one_result.stage_one")
ext_end = next(i for i, l in enumerate(ext_lines) if l.strip().startswith("state.pop") and "post_pipeline_artifacts" in l)
print("tip EXT body (1-based):", ext_start + 1, ext_end + 1)

def norm(line):
    s = line.strip()
    s = s.replace("_stage_one.", "stage_one_result.")
    s = s.replace("st.session_state", "state").replace("st.spinner", "reporter.spinner").replace("st.error", "reporter.error")
    s = s.replace("_rag_orchestration.", "").replace("_postpipeline.", "").replace("_pipeline.", "")
    s = s.replace("current_extraction_diagnostics()", "_diag.current()")
    s = s.replace("StreamlitReporter(st)", "reporter")
    return s
a = [norm(l) for l in base_lines[start:end + 1]]
b = [norm(l) for l in ext_lines[ext_start:ext_end + 1]]
diff = list(difflib.unified_diff(a, b, "base-norm", "tip-norm", lineterm="", n=0))
print("normalized diff lines:", len(diff))
open("C:/t/rev-R3d/out/verbatim_norm.diff", "w", encoding="utf-8").write("\n".join(diff) + "\n")
print("\n".join(diff))
