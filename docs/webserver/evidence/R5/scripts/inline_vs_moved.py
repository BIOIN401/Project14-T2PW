"""R5: the two moves the brief asks to prove (the ambiguous-review gate and the review-state function).

Compares base-app source to tip-service source after the mechanical rewrites only: ``st.session_state`` to
``state`` (attribute and item forms), ``st.`` to ``reporter.``, module-qualified callees, the bound message.
Prints every line that still differs; anything not in EXPECTED is a finding.
"""
import ast
import difflib
import re
import sys

BASE = "C:/t/ws-R5-base/src/t2pw/app/streamlit_app.py"
TIP_EX = "C:/t/ws-R5/src/t2pw/service/extraction.py"
TIP_PR = "C:/t/ws-R5/src/t2pw/service/post_run.py"


def norm(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        s = re.sub(r"st\.session_state\.(\w+) = ", r'state["\1"] = ', s)
        s = s.replace("st.session_state", "state").replace("st.", "reporter.")
        s = s.replace("_draft_graph.build_draft_graph", "build_draft_graph")
        s = s.replace("_render.render_draft_graph_to_png_bytes", "render_draft_graph_to_png_bytes")
        s = s.replace("_qa_graph.generate_qa_report", "generate_qa_report")
        s = s.replace("_curator.compact_mapping_misses", "compact_mapping_misses")
        if s:
            out.append(s)
    return out


def node_src(path, finder):
    src = open(path, encoding="utf-8").read()
    node = finder(ast.parse(src))
    return ast.get_source_segment(src, node).splitlines()


def is_fn(name):
    return lambda t: next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == name)


base_gate = node_src(BASE, lambda t: next(n for n in ast.walk(t) if isinstance(n, ast.If) and
                                          ast.unparse(n.test) == "is_ambiguous_multi_example_review_context(pathway_context)"))
tip_gate = node_src(TIP_EX, is_fn("ambiguous_review_gate"))
base_rev = node_src(BASE, is_fn("initialize_refinement_review_state"))
tip_rev = node_src(TIP_PR, is_fn("open_refinement_review"))

report = []
for title, a, b in (("gate", base_gate, tip_gate), ("review state", base_rev, tip_rev)):
    report.append(f"== {title} ==")
    report.extend(l for l in difflib.unified_diff(norm(a), norm(b), "base-app", "tip-service", lineterm="", n=0))
text = "\n".join(report) + "\n"
open(sys.argv[1], "w", encoding="utf-8").write(text)
print(text)
