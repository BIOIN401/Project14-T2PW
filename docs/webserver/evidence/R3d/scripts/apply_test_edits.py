"""R3d: apply exactly the three authorised test edits (Q-R0-1 c060a, Q-R3d-1 order test, Q-R3d-3 cassette check)
plus the harness service arm's ``unused`` field the cassette check reads."""
from pathlib import Path

T = Path("C:/t/ws-R3d")


def rw(path, fn):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    src = fn(raw.replace("\r\n", "\n"))
    path.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))


# 1. c060a: the proposed committed form, verbatim.
c060a = T / "tests/test_c060a_requested_pathway_wiring.py"
form = (T / "docs/webserver/evidence/R3d/proposed/c060a_repoint_check.committed-form.txt").read_bytes().decode("utf-8")
rw(c060a, lambda _src: form.replace("\r\n", "\n"))

# 2. the order test: replace that one function with the proposed one.
order = T / "tests/test_streamlit_stage2_orchestration.py"
proposed = (T / "docs/webserver/evidence/R3d/proposed/s2orch_order_proposed.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
new_fn = proposed[proposed.index("def test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference"):].rstrip("\n") + "\n"


def swap(src):
    start = src.index("def test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference")
    end = src.index("def test_stage2_artifacts_are_visible_and_both_ui_paths_share_the_orchestrator")
    assert src.count("def test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference") == 1
    return src[:start] + new_fn + "\n\n" + src[end:]


rw(order, swap)

# 3. the cassette check in test_c (2 lines).
eq = T / "tests/test_equivalence_app_vs_service.py"
anchor = '    assert service["misses"] == [] and service["refused"] == []\n'


def add_check(src):
    assert src.count(anchor) == 1
    return src.replace(anchor, anchor +
                       "    # Q-R3d-3: from R3d the service arm runs from the text, so it must consume every recorded LLM reply.\n"
                       '    assert [row for row in service.get("unused", []) if row[0] == "llm"] == [], service.get("unused")\n')


rw(eq, add_check)

# harness service arm: report what the cassette left unused (both return shapes).
h = T / "tests/equivalence/harness.py"


def unused(src):
    old1 = '                "export_verdict": None, "misses": list(cassette.misses), "refused": list(cassette.refused)}\n'
    old2 = '        "refused": list(cassette.refused),\n        "frozen_bindings": frozen_bindings,\n'
    assert src.count(old1) == 1 and src.count(old2) == 1
    src = src.replace(old1, '                "export_verdict": None, "misses": list(cassette.misses), "refused": list(cassette.refused),\n'
                            '                "unused": cassette.unused()}\n')
    return src.replace(old2, '        "refused": list(cassette.refused),\n        "unused": cassette.unused(),\n'
                             '        "frozen_bindings": frozen_bindings,\n')


rw(h, unused)
print("ok")
