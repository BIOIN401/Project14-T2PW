"""The test edits proposed to the orchestrator, applied to the SCRATCH COPY only."""
import sys

tree = sys.argv[1]


def edit(rel, pairs):
    p = f"{tree}/{rel}"
    s = open(p, encoding="utf-8", newline="").read()
    for a, b in pairs:
        a, b = a.replace("\n", "\r\n"), b.replace("\n", "\r\n")
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8", newline="").write(s)


edit("tests/test_service_driver_bridge.py", [
    ('assert worker.DEFAULT_RUN_FN == "t2pw.service.driver_bridge:run_pathway"\n    for spec, fn in ((worker.DEFAULT_RUN_FN, br.run_pathway),',
     'assert worker.DEFAULT_RUN_FN == "t2pw.service.pathway:run_pathway"\n    for spec, fn in (("t2pw.service.driver_bridge:run_pathway", br.run_pathway),'),
    ('assert worker._child_main(worker.DEFAULT_RUN_FN, str(rq), str(rs)) == 0',
     'assert worker._child_main("t2pw.service.driver_bridge:run_pathway", str(rq), str(rs)) == 0'),
])
edit("tests/test_service_extraction_stage2.py", [
    ('''    assert res.failure == ex.InputFailure(
        ex.KIND_STAGE_ONE_FAILED, "Extraction failed: multi_example_review detected with no selected_example. "
        "Extraction skipped to prevent mixed-pathway output.")
    assert "pipeline_ready" not in state and "stage_one_boundary" not in state
''', '''    assert res.failure == ex.InputFailure(
        ex.KIND_AMBIGUOUS_REVIEW_SCOPE, "multi_example_review detected with no selected_example. "
        "Extraction skipped to prevent mixed-pathway output.")
    assert state["pipeline_ready"] is False and "stage_one_boundary" not in state
'''),
])
p = f"{tree}/tests/test_service_pathway_interface.py"
s = open(p, encoding="utf-8", newline="").read()
start = s.index("def test_stub_returns_fixed_error_and_writes_nothing(tmp_path):")
end = s.index("def ", start + 10)
s = s[:start] + s[end:]
open(p, "w", encoding="utf-8", newline="").write(s)
print("applied")
