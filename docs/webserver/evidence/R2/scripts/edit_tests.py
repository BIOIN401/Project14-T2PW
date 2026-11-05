"""R2 test repoints (scratch). Path / patch-target edits only."""
from pathlib import Path

T = Path("C:/t/ws-R2/tests")


def edit(name, pairs):
    p = T / name
    s = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in s
    s = s.replace("\r\n", "\n")
    for old, new in pairs:
        assert s.count(old) == 1, (name, old, s.count(old))
        s = s.replace(old, new)
    if crlf:
        s = s.replace("\n", "\r\n")
    p.write_bytes(s.encode("utf-8"))
    print("edited", name)


RAG = "t2pw.service.rag_orchestration"

edit("test_c055_rag_loop_wiring.py", [
    # AST reader: the three functions now live in the RAG service module.
    ('APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"\n',
     'APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"\n'
     '#: R2: maybe_run_rag / run_rag_rounds / build_rag_round_stages now live here.\n'
     'RAG_PATH = SRC / "t2pw" / "service" / "rag_orchestration.py"\n'),
    ('    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))\n'
     '    functions = {node.name: node for node in ast.walk(tree)\n',
     '    tree = ast.parse(RAG_PATH.read_text(encoding="utf-8"))\n'
     '    functions = {node.name: node for node in ast.walk(tree)\n'),
    # app_module fixture: re-import the RAG module with the app, as R1 F4 does for postpipeline.
    ('    sys.modules.pop("t2pw.service.postpipeline", None)\n'
     '    before = set(sys.modules)\n',
     '    sys.modules.pop("t2pw.service.postpipeline", None)\n'
     '    sys.modules.pop("' + RAG + '", None)  # R2: same, for the moved RAG code\n'
     '    before = set(sys.modules)\n'),
    ('        sys.modules.pop("t2pw.service.postpipeline", None)\n'
     '        for name in [n for n in list(sys.modules)\n',
     '        sys.modules.pop("t2pw.service.postpipeline", None)\n'
     '        sys.modules.pop("' + RAG + '", None)  # R2\n'
     '        for name in [n for n in list(sys.modules)\n'),
    # offline_stages: the round adapters bind these names in the RAG module.
    ('        monkeypatch.setattr(cached, "run_strict_post_normalization_gates", _gates)\n'
     '    yield calls\n'
     '    if cached is None:\n'
     '        sys.modules.pop("t2pw.service.postpipeline", None)\n',
     '        monkeypatch.setattr(cached, "run_strict_post_normalization_gates", _gates)\n'
     '    # R2: the same for the RAG round adapters, which moved to ' + RAG + '.\n'
     '    cached_rag = sys.modules.get("' + RAG + '")\n'
     '    if cached_rag is not None:\n'
     '        monkeypatch.setattr(cached_rag, "normalize_process_payload", _normalize)\n'
     '        monkeypatch.setattr(cached_rag, "run_strict_post_normalization_gates", _gates)\n'
     '        monkeypatch.setattr(cached_rag, "build_and_save_draft_graph", _persist)\n'
     '        monkeypatch.setattr(cached_rag, "map_payload", _map)\n'
     '    yield calls\n'
     '    if cached is None:\n'
     '        sys.modules.pop("t2pw.service.postpipeline", None)\n'
     '    if cached_rag is None:\n'
     '        sys.modules.pop("' + RAG + '", None)\n'),
    # _rebind: the four stage names the round adapters use are now bound in the RAG module.
    ('    app.normalize_process_payload = normalizer.normalize_process_payload\n'
     '    app.run_strict_post_normalization_gates = normalizer.run_strict_post_normalization_gates\n'
     '    app.build_and_save_draft_graph = pipeline_module.build_and_save_draft_graph\n'
     '    app.quarantine_and_close = quarantine.quarantine_and_close\n'
     '    app.map_payload = map_ids.map_payload\n',
     '    # R2: the round adapters moved; their names are bound in the RAG service module.\n'
     '    rag = sys.modules["' + RAG + '"]\n'
     '    rag.normalize_process_payload = normalizer.normalize_process_payload\n'
     '    rag.run_strict_post_normalization_gates = normalizer.run_strict_post_normalization_gates\n'
     '    rag.build_and_save_draft_graph = pipeline_module.build_and_save_draft_graph\n'
     '    app.quarantine_and_close = quarantine.quarantine_and_close\n'
     '    rag.map_payload = map_ids.map_payload\n'),
    ('    monkeypatch.setattr(app_module, "maybe_run_rag", _chain)\n',
     '    monkeypatch.setattr(sys.modules["' + RAG + '"], "maybe_run_rag", _chain)  # R2\n'),
    ('    app_module.build_and_save_draft_graph = _explode\n',
     '    sys.modules["' + RAG + '"].build_and_save_draft_graph = _explode  # R2\n'),
])

edit("test_rag_triage_orchestration.py", [
    ('    monkeypatch.setattr(app, "rag_config", lambda *a, **k: {"enabled": False})\n',
     '    # R2: maybe_run_rag moved; patch rag_config where the function reads it.\n'
     '    monkeypatch.setitem(app.maybe_run_rag.__globals__, "rag_config", lambda *a, **k: {"enabled": False})\n'),
])
p = T / "test_rag_triage_orchestration.py"
s = p.read_bytes().decode("utf-8")
old = 'monkeypatch.setattr(app, "rag_config", lambda *a, **k: {"enabled": True})'
assert s.count(old) == 2
s = s.replace(old, 'monkeypatch.setitem(app.maybe_run_rag.__globals__, "rag_config", lambda *a, **k: {"enabled": True})  # R2')
p.write_bytes(s.encode("utf-8"))
print("edited test_rag_triage_orchestration.py (x2 more)")

edit("test_c060a_requested_pathway_wiring.py", [
    ('SERVICE_PATH = SRC / "t2pw" / "service" / "postpipeline.py"\n',
     'SERVICE_PATH = SRC / "t2pw" / "service" / "postpipeline.py"\n'
     '#: R2: ``maybe_run_rag`` now lives here.\n'
     'RAG_PATH = SRC / "t2pw" / "service" / "rag_orchestration.py"\n'),
    ('        node.name: node for node in (*_tree(APP_PATH).body, *_tree(SERVICE_PATH).body)\n',
     '        node.name: node for node in (*_tree(APP_PATH).body, *_tree(SERVICE_PATH).body,\n'
     '                                     *_tree(RAG_PATH).body)\n'),
])

edit("test_stage_one_paper_dictionary.py", [
    ('        paths = [rel] + (["src/t2pw/service/postpipeline.py"] if rel == "src/t2pw/app/streamlit_app.py" else [])\n',
     '        # R2: the third (the RAG round mapping adapter) moved to rag_orchestration.py.\n'
     '        paths = [rel] + (["src/t2pw/service/postpipeline.py", "src/t2pw/service/rag_orchestration.py"]\n'
     '                         if rel == "src/t2pw/app/streamlit_app.py" else [])\n'),
])
