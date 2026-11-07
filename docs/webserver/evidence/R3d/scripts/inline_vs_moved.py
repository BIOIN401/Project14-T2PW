"""R3d display-equivalence proof: the BASE app's inline Stage-2 .. publish code (plus the
base app's own helpers) vs the TIP app's call site + run_stage_two + StreamlitReporter,
against the same recording fake ``st`` and the same fake callees, over every branch.

Both snippets are cut out of the real app files and exec'd at module level, as Streamlit
runs the script:
  base: from ``    stage_one = _stage_one.stage_one`` (R3c's rebinding) to the token-stats line;
  tip:  from the R3d call-site comment to the token-stats line.
So the unchanged tail (``reset_quarantine_state`` + token stats) is in both, and its order
relative to the publish is compared too. The base helpers (``render_attempts``,
``emit_rag_loop_messages``, ``rag_loop_session_record``) are lifted out of the BASE app by
AST; the tip uses the committed service module.

The fakes reach the base through its namespace (the names the app script bound with
``from ... import``) and the tip through the SOURCE modules (``pipeline``,
``postpipeline``, ``rag_orchestration``, ``extraction_diagnostics``), which is what the
service reads at call time. ``pathway_context_from_stage_zero`` is the real factory.

Compared per scenario: the ordered st.* calls (spinner / expander enter and exit
included), every session_state write AND pop in order with its value, where st.stop()
fired (or what was raised), every callee call with its arguments (the reporter by type
and target, ``state`` by identity), and the script globals afterwards.

Globals: the base binds ~19 names inside the block; the tip binds ``_stage_two`` only.
A difference there is allowed (SAME-EXCEPT-DEAD-GLOBALS) only if everything else is
identical and every differing name is DEAD after the block in the tip app: a static
check below parses the tip app and requires that each such name is never loaded after
the ``if submit:`` block without first being re-assigned there.

usage: inline_vs_moved.py BASE_TREE TIP_TREE     (PYTHONPATH must be TIP_TREE/src)
"""
import ast
import dataclasses
import json
import sys
import tempfile
import textwrap
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

base_tree, tip_tree = sys.argv[1], sys.argv[2]

import t2pw.paths as paths  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as xd  # noqa: E402
import t2pw.pipeline.pipeline as pipeline  # noqa: E402
import t2pw.service.extraction as ex  # noqa: E402
import t2pw.service.postpipeline as postpipeline  # noqa: E402
import t2pw.service.rag_orchestration as rag  # noqa: E402
from t2pw.service.postpipeline import _safe_dict, _safe_list  # noqa: E402
from t2pw.service.reporter import StreamlitReporter  # noqa: E402

assert ex.__file__.replace("\\", "/").startswith(tip_tree.replace("\\", "/")), ex.__file__
ROOT = Path(tempfile.mkdtemp(prefix="r3d-ivm-"))
REAL = {(m, n): getattr(m, n) for m, n in [
    (pipeline, "run_stage_two_with_feedback_loop"), (pipeline, "merge_additions"),
    (pipeline, "build_and_save_draft_graph"), (postpipeline, "_write_reaction_preservation_report"),
    (rag, "run_rag_rounds"), (xd, "current")]}
DEAD = ["stage_one", "chunk_details", "stage_one_in_scope", "out_of_scope_removed_reactions", "rag_result",
        "user_task_context", "final_payload", "stage_two", "stage_two_chunks", "stage_two_rounds", "qa_hints",
        "label", "failure", "rag_loop_record", "stage2_preservation_report", "draft_graph", "qa_report",
        "reaction_summary", "_rag_merge_site", "_stage_two"]


def _default(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {"<" + type(o).__name__ + ">": dataclasses.asdict(o)}
    return f"<{type(o).__name__}>"


def norm(value):
    return json.dumps(value, sort_keys=True, default=_default)


def app_src(tree):
    return Path(tree, "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8")


END = '    st.session_state["token_stats"] = llm_client_module.get_token_stats()'


def cut(tree, first):
    lines = app_src(tree).splitlines()
    start, end = lines.index(first), lines.index(END)
    return textwrap.dedent("\n".join(lines[start:end + 1]))


def lift(tree, names):
    keep = [n for n in ast.parse(app_src(tree)).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in keep} == set(names), names
    return compile(ast.Module(body=keep, type_ignores=[]), f"{tree}/streamlit_app.py", "exec")


BASE_CUT = cut(base_tree, "    stage_one = _stage_one.stage_one")
TIP_CUT = cut(tip_tree, "    # R3d: Stage 2, both merge_additions sites, the RAG rounds, the draft graph and the")
assert "run_stage_two_with_feedback_loop(" in BASE_CUT and "run_stage_two(" not in BASE_CUT
assert "= run_stage_two(" in TIP_CUT and "merge_additions(" not in TIP_CUT and "run_rag_rounds" not in TIP_CUT
BASE_HELPERS = lift(base_tree, {"render_attempts", "emit_rag_loop_messages", "rag_loop_session_record"})


# ── static: the names the tip no longer binds are dead after the submit block ──
def dead_after_block(tree):
    mod = ast.parse(app_src(tree))
    submit = next(n for n in mod.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                  and n.test.id == "submit")
    after = [n for n in mod.body if n.lineno > submit.end_lineno]
    problems = []
    for name in DEAD:
        loads, stores = [], []
        for top in after:
            for node in ast.walk(top):
                if isinstance(node, ast.Name) and node.id == name:
                    (loads if isinstance(node.ctx, ast.Load) else stores).append(node.lineno)
                elif isinstance(node, (ast.ExceptHandler,)) and node.name == name:
                    stores.append(node.lineno)
        first_store = min(stores) if stores else None
        bad = [ln for ln in loads if first_store is None or ln <= first_store]
        if bad:
            problems.append((name, bad, first_store))
    return problems


STATIC = dead_after_block(tip_tree)
print("static dead-globals check (tip app, after `if submit:`):", "OK" if not STATIC else STATIC)


class Stop(Exception):
    pass


class Ctx:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __enter__(self):
        self.log.append([self.name + ":enter"])

    def __exit__(self, *a):
        self.log.append([self.name + ":exit"])


class State(dict):
    def __init__(self, log):
        super().__init__()
        self._log = log

    def __setitem__(self, key, value):
        self._log.append(["session_state", key, norm(value)])
        super().__setitem__(key, value)

    def pop(self, key, *default):
        self._log.append(["session_state.pop", key])
        return super().pop(key, *default)


class FakeSt:
    def __init__(self):
        self.log = []
        self.session_state = State(self.log)

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append([name, [norm(x) for x in a], {kk: norm(v) for kk, v in k.items()}])
            if name == "stop":
                raise Stop()
            if name in ("spinner", "expander"):
                return Ctx(self.log, name)
        return call


class Graph:
    def __init__(self, n):
        self.n = n

    def to_dict(self):
        return {"draft_graph_of": self.n}


def payload(*names):
    return {"entities": {"compounds": []}, "processes": {"reactions": [{"name": n} for n in names]}}


class Frozen(SimpleNamespace):
    def __setattr__(self, name, value):
        if name == "loop":
            raise AttributeError("read-only")
        super().__setattr__(name, value)


ATTEMPTS = [{"attempt": 1, "raw": "{bad", "error": "Expecting value", "phase": "infer", "note": "cut"},
            {"attempt": 2, "raw": "{}", "error": None}]
ADD = {"additions": {"processes": {"reactions": [{"name": "b"}]}}, "qa_hints": {"disconnected": ["x"]}}
MSGS = [["info", "round 0 ok"], ["warning", "w"], ["error", "e"], ["json", {"j": 1}], ["other", "dropped"]]


def failure(stage):
    f = pipeline.PipelineFailure("stage2", "no JSON after 2", ATTEMPTS)
    f.stage = stage
    return f


# stage2 = additions | an exception; rag = None | a result object (factory, so each arm gets its own)
S = {
    "infer-success": dict(stage2=ADD),
    "infer-non-dict": dict(stage2=["not", "a", "dict"]),
    "infer-dict-no-hints": dict(stage2={"additions": {}}),
    "infer-off": dict(stage2=ADD, run_inference=False),
    "infer-off-int-zero": dict(stage2=ADD, run_inference=0),
    "infer-float-widgets": dict(stage2=ADD, sizes=(900.0, 150.0, 3.0, 4.0, 12000.0), enable_chunking=True),
    "infer-failure-with-stage": dict(stage2=failure("stage2-chunk-1")),
    "infer-failure-no-stage": dict(stage2=failure(None)),
    "infer-failure-no-attempts": dict(stage2=pipeline.PipelineFailure("stage2", "empty", [])),
    "infer-other-exception": dict(stage2=RuntimeError("LLM down")),
    "pops-and-stale-keys": dict(stage2=ADD, seed={"draft_graph_png_bytes": b"png", "post_pipeline_artifacts": {"o": 1},
                                                  "draft_graph_render_error": "e"}),
    "rag-rounds": dict(stage2=ADD, rag=lambda: SimpleNamespace(synthesized=True, payload={"p": 1}), flag=1,
                       loop=dict(payload=payload("a", "b", "r"), messages=MSGS, rounds=[{"i": 0}], max_rounds=1)),
    "rag-rounds-strict-db-off": dict(stage2=ADD, rag=lambda: SimpleNamespace(synthesized=True, payload={"p": 1}),
                                     seed={"pwml_strict_db": 0}, loop=dict(payload=payload("r"), messages=[])),
    "rag-rounds-research-mode": dict(stage2=ADD, export_mode="research",
                                     rag=lambda: SimpleNamespace(synthesized=True, payload={"p": 1}),
                                     loop=dict(payload=payload("r"), messages=None)),
    "rag-rounds-infer-off": dict(stage2=ADD, run_inference=False,
                                 rag=lambda: SimpleNamespace(synthesized=True, payload={"p": 1}),
                                 loop=dict(payload=payload("r"), messages=MSGS[:1])),
    "rag-rounds-frozen-result": dict(stage2=ADD, rag=lambda: Frozen(synthesized=True, payload={"p": 1}),
                                     loop=dict(payload=payload("r"), messages=[])),
    "rag-not-synthesized": dict(stage2=ADD, rag=lambda: SimpleNamespace(synthesized=False, payload={"p": 1})),
    "rag-no-payload": dict(stage2=ADD, rag=lambda: SimpleNamespace(synthesized=True, payload=None)),
    "rag-empty-payload": dict(stage2=ADD, rag=lambda: SimpleNamespace(synthesized=True, payload={})),
    "rag-rounds-raise": dict(stage2=ADD, rag=lambda: SimpleNamespace(synthesized=True, payload={"p": 1}),
                             loop="raise"),
    "draft-graph-raises": dict(stage2=ADD, draft_raises=True),
    "preservation-none": dict(stage2=ADD, preservation=None),
}


def run(arm, name, sc):
    st, calls = FakeSt(), []

    def rec(label, fn):
        def wrapper(*a, **k):
            calls.append([label, [norm(x) for x in a],
                          {kk: (f"<{type(v).__name__} st={getattr(v, '_st', None) is st}>" if kk == "reporter"
                                else f"<state is session_state={v is st.session_state}>" if kk == "state"
                                else f"<callable>" if callable(v) else norm(v)) for kk, v in k.items()}])
            return fn(*a, **k)
        return wrapper

    def stage2(*_a, **_k):
        if isinstance(sc["stage2"], Exception):
            raise sc["stage2"]
        return deepcopy(sc["stage2"]), [{"chunk": 1, "qa_round": 1}], [{"round": 1}]

    def merge(base, additions, **_k):
        return {"merged": [base, additions]}

    def build(p):
        if sc.get("draft_raises"):
            raise RuntimeError("disk full")
        return Graph(len(norm(p))), {"qa": len(norm(p))}, f"summary of {len(norm(p))}"

    def rounds(graph, **k):
        if sc.get("loop") == "raise":
            raise RuntimeError("round 1 blew up")
        k["merge_site"](graph, {"additions": {"processes": {"reactions": [{"name": "r"}]}}}, k["pathway_context"])
        return deepcopy(sc["loop"])

    recorder = SimpleNamespace(tag="recorder")
    fakes = {"run_stage_two_with_feedback_loop": rec("stage2", stage2),
             "merge_additions": rec("merge", merge),
             "build_and_save_draft_graph": rec("draft_graph", build),
             "_write_reaction_preservation_report": rec("preservation",
                                                        lambda stage, p: sc.get("preservation", {"stage": stage})),
             "run_rag_rounds": rec("rag_rounds", rounds)}
    sizes = sc.get("sizes", (1200.0, 200.0, 2.0, 3.0, 8000.0))
    rag_result = sc["rag"]() if sc.get("rag") else None
    stage_one_result = ex.StageOneResult(
        stage_one=payload("a", "oos"), chunk_details=[{"chunk_id": 1}], stage_one_in_scope=payload("a"),
        out_of_scope_removed_reactions=["oos"], rag_result=rag_result, user_task_context="focus\n\nEVIDENCE")
    ns = dict(st=st, text="Menaquinone text.", pathway_context={"pathway_name": "Menaquinone",
                                                                 "likely_organism": "Escherichia coli"},
              user_task_context="focus", _stage_one=stage_one_result, run_inference=sc.get("run_inference", True),
              enable_chunking=sc.get("enable_chunking", False), chunk_size=sizes[0], chunk_overlap=sizes[1],
              infer_rounds=sizes[2], infer_attempts=sizes[3], temperature=0.2, infer_tokens=sizes[4],
              rag_incomplete_flag=sc.get("flag", False), export_mode=sc.get("export_mode", "pathwhiz"),
              PROJECT_ROOT=ROOT, Any=Any, Dict=Dict, List=List,
              reset_quarantine_state=rec("reset_quarantine_state", lambda *a, **k: None),
              llm_client_module=SimpleNamespace(get_token_stats=rec("get_token_stats", lambda: {"tokens": 7})),
              StreamlitReporter=StreamlitReporter)
    for key, value in (sc.get("seed") or {}).items():
        dict.__setitem__(st.session_state, key, value)
    xd.current = rec("diag.current", lambda: recorder)
    if arm == "base":
        ns.update(fakes, PipelineFailure=pipeline.PipelineFailure, current_extraction_diagnostics=xd.current,
                  _safe_dict=_safe_dict, _safe_list=_safe_list, _service_render_attempts=ex.render_attempts)
        exec(BASE_HELPERS, ns)
    else:
        paths.PROJECT_ROOT = ROOT
        for (module, attr) in REAL:
            if attr in fakes:
                setattr(module, attr, fakes[attr])
        ns.update(run_stage_two=ex.run_stage_two)
    outcome = "completed"
    try:
        exec(compile(BASE_CUT if arm == "base" else TIP_CUT, f"<{arm}:{name}>", "exec"), ns)
    except Stop:
        outcome = "stopped"
    except Exception as exc:  # noqa: BLE001 - the raise itself is compared
        outcome = f"raised {type(exc).__name__}: {exc}"
    finally:
        for (module, attr), real in REAL.items():
            setattr(module, attr, real)
    after = {k: norm(ns.get(k, "<unbound>")) for k in DEAD}
    loop = getattr(rag_result, "loop", "<none>") if rag_result is not None else "<no result>"
    return {"st": st.log, "outcome": outcome, "calls": calls, "after": after, "rag_result.loop": norm(loop),
            "state_final": norm(dict(st.session_state))}


bad = 0
for name, sc in S.items():
    a, b = run("base", name, sc), run("tip", name, sc)
    fields = [k for k in a if norm(a[k]) != norm(b[k])]
    differing = sorted(k for k in a["after"] if a["after"][k] != b["after"][k])
    static_bad = {p[0] for p in STATIC}
    dead_only = (fields == ["after"] and all(k in DEAD and k not in static_bad for k in differing))
    verdict = "IDENTICAL" if not fields else ("SAME-EXCEPT-DEAD-GLOBALS" if dead_only else "DIFFERENT")
    bad += verdict == "DIFFERENT"
    n_state = sum(1 for e in a["st"] if e[0].startswith("session_state"))
    print(f"{name:28s} {verdict:25s} outcome={a['outcome'][:40]:<40s} st-events={len(a['st'])} "
          f"state-ops={n_state} callee-calls={len(a['calls'])}")
    if dead_only:
        print(f"{'':28s}   differing globals (dead after the block): {differing}")
    if verdict == "DIFFERENT":
        print("  differing fields:", fields)
        for f in fields:
            print("   base", f, norm(a[f])[:3000])
            print("   tip ", f, norm(b[f])[:3000])

bad += bool(STATIC)
print(f"{len(S) - (bad - bool(STATIC))}/{len(S)} identical or known-difference only; "
      f"static check {'OK' if not STATIC else 'FAILED'}; {bad} unexplained")
sys.exit(1 if bad else 0)
