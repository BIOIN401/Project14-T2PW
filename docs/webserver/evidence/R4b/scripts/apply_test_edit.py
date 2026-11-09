"""R4b (Q-R4-1): repoint the one authorised test at the moved handler. CRLF-preserving."""
from pathlib import Path

P = Path(r"C:/t/ws-R4b/tests/test_streamlit_stage2_orchestration.py")
raw = P.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")

reps = [
    (
        '''    tree = _source_tree()
    live_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_post_pipeline_sbml_artifacts"
    ]
    assert len(live_calls) == 2
''',
        '''    tree = _source_tree()
    # R4b (Q-R4-1): the audit handler's call moved to t2pw.service.post_run; the legacy
    # button's stays here. Both still reach the one orchestrator.
    post_run_path = ROOT / "src" / "t2pw" / "service" / "post_run.py"
    post_run_tree = ast.parse(post_run_path.read_text(encoding="utf-8"))

    def orchestrator_calls(module: ast.AST) -> list:
        return [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "run_post_pipeline_sbml_artifacts")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "run_post_pipeline_sbml_artifacts")
            )
        ]

    assert len(orchestrator_calls(tree)) == 1
    assert len(orchestrator_calls(post_run_tree)) == 1
    live_calls = [*orchestrator_calls(tree), *orchestrator_calls(post_run_tree)]
    assert len(live_calls) == 2
''',
    ),
    (
        '''    orchestrator_tries = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "run_post_pipeline_sbml_artifacts"
            for statement in node.body
            for candidate in ast.walk(statement)
        )
    ]
    assert len(orchestrator_tries) == 2
    for try_node in orchestrator_tries:
        assert isinstance(try_node.handlers[0].type, ast.Name)
        assert try_node.handlers[0].type.id == "StageContractError"
        assert any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "_render_stage_contract_failure"
            for statement in try_node.handlers[0].body
            for candidate in ast.walk(statement)
        )
''',
        '''    orchestrator_tries = [
        node
        for module in (tree, post_run_tree)
        for node in ast.walk(module)
        if isinstance(node, ast.Try)
        and any(
            any(candidate is call for call in orchestrator_calls(statement))
            for statement in node.body
            for candidate in ast.walk(statement)
        )
    ]
    assert len(orchestrator_tries) == 2
    for try_node in orchestrator_tries:
        assert isinstance(try_node.handlers[0].type, ast.Name)
        assert try_node.handlers[0].type.id == "StageContractError"
        # R4b: in the service the handler calls the render callback the app passes (below).
        assert any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id in {"_render_stage_contract_failure", "render_contract_failure"}
            for statement in try_node.handlers[0].body
            for candidate in ast.walk(statement)
        )
''',
    ),
    (
        '''            and candidate.args[0].value == "post_pipeline_artifacts"
            for candidate in ast.walk(button_branch)
        )

    source = APP_PATH.read_text(encoding="utf-8")
    assert "Mapped pathway still fails the pre-export Stage 3 revalidation." in source
''',
        '''            and candidate.args[0].value == "post_pipeline_artifacts"
            for candidate in ast.walk(button_branch)
        )
        if button_label == "Run audit and DB mapping":
            # R4b: the button's branch reaches the handler, and the handler's stage-contract
            # failure still renders through the app's shared renderer.
            handler_call = next(
                candidate
                for candidate in ast.walk(button_branch)
                if isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Name)
                and candidate.func.id == "run_post_pipeline"
            )
            render_arg = next(keyword.value for keyword in handler_call.keywords
                              if keyword.arg == "render_contract_failure")
            assert any(
                isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Name)
                and candidate.func.id == "_render_stage_contract_failure"
                for candidate in ast.walk(render_arg)
            )

    assert "Mapped pathway still fails the pre-export Stage 3 revalidation." in post_run_path.read_text(encoding="utf-8")
    source = APP_PATH.read_text(encoding="utf-8")
''',
    ),
]
for old, new in reps:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
out = s.replace("\n", "\r\n") if crlf else s
P.write_bytes(out.encode("utf-8"))
print("ok")
