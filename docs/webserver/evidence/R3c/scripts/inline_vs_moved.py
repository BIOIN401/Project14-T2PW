"""R3c display-equivalence proof: the BASE app's inline Stage-1 / RAG-seed code (plus the
base app's own helpers) vs the TIP app's call site + run_stage_one + StreamlitReporter,
against the same recording fake ``st`` and the same fake callees, over every branch.

Both snippets are cut out of the real app files, from the Stage-1 comment (base) / the
R3c call-site comment (tip) up to ``final_payload = stage_one_in_scope``, and exec'd at
module level, as Streamlit runs the script. The base helpers (``render_attempts``,
``_last_cleaning_report``, ``_sync_extraction_diagnostics``) are lifted out of the BASE
app by AST; the tip uses the committed service module.

The fakes reach the base through its namespace (the names the app script bound with
``from ... import``) and the tip through the SOURCE modules (``pipeline``,
``stage_one_boundary``, ``draft_graph``, ``qa_graph``, ``rag_orchestration``,
``config``, ``extraction_diagnostics``), which is what the service reads at call time.

Compared per scenario: the ordered st.* calls (spinner / expander enter and exit
included), where st.stop() fired (or what was raised), every session_state write in
order with its value, every callee call with its arguments, and the values the rest of
the submit block reads afterwards: stage_one, chunk_details, stage_one_in_scope,
out_of_scope_removed_reactions, rag_result (identity too) and user_task_context.

Also compared: the base app's ``render_attempts`` vs the tip app's shim, directly.

usage: inline_vs_moved.py BASE_TREE TIP_TREE     (PYTHONPATH must be TIP_TREE/src)
"""
import ast
import json
import sys
import tempfile
import textwrap
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

base_tree, tip_tree = sys.argv[1], sys.argv[2]

import t2pw.config as config  # noqa: E402
import t2pw.paths as paths  # noqa: E402
import t2pw.pipeline.draft_graph as draft_graph  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as xd  # noqa: E402
import t2pw.pipeline.pipeline as pipeline  # noqa: E402
import t2pw.pipeline.qa_graph as qa_graph  # noqa: E402
import t2pw.pipeline.stage_one_boundary as sob  # noqa: E402
import t2pw.service.extraction as ex  # noqa: E402
import t2pw.service.rag_orchestration as rag  # noqa: E402
from t2pw.pipeline.export_mode import coerce_mode  # noqa: E402
from t2pw.pipeline.stage_contracts import StageContractError  # noqa: E402
from t2pw.service.postpipeline import _safe_dict, _safe_list  # noqa: E402
from t2pw.service.reporter import StreamlitReporter  # noqa: E402

assert ex.__file__.replace("\\", "/").startswith(tip_tree.replace("\\", "/")), ex.__file__
ROOT = Path(tempfile.mkdtemp(prefix="r3c-ivm-"))
DIAG_DIR = ROOT / "tmp" / "extraction_diagnostics" / "run-X"
REAL = {(m, n): getattr(m, n) for m, n in [
    (pipeline, "run_stage_one_with_chunking"), (pipeline, "filter_out_of_scope_reactions"),
    (sob, "settle_stage_one"), (draft_graph, "build_draft_graph"), (qa_graph, "generate_qa_report"),
    (rag, "maybe_run_rag"), (config, "rag_config"), (xd, "current")]}
REAL_FILTER = pipeline.filter_out_of_scope_reactions


def norm(value):
    return json.dumps(value, sort_keys=True, default=lambda o: f"<{type(o).__name__}>")


def app_src(tree):
    return Path(tree, "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8")


def cut(tree, first):
    lines = app_src(tree).splitlines()
    start = lines.index(first)
    end = lines.index("    final_payload = stage_one_in_scope")
    return textwrap.dedent("\n".join(lines[start:end]))


def lift(tree, names):
    keep = [n for n in ast.parse(app_src(tree)).body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in keep} == set(names), names
    return compile(ast.Module(body=keep, type_ignores=[]), f"{tree}/streamlit_app.py", "exec")


BASE_CUT = cut(base_tree, "    # Stage 1: strict extraction with auto-repair")
TIP_CUT = cut(tip_tree, "    # R3c: Stage 1 lives in t2pw.service.extraction.run_stage_one: the extraction and")
assert "run_stage_one_with_chunking(" in BASE_CUT and "run_stage_one(" not in BASE_CUT.replace(
    "run_stage_one_with_chunking(", "")
assert "= run_stage_one(" in TIP_CUT and "run_stage_one_with_chunking" not in TIP_CUT
BASE_HELPERS = lift(base_tree, {"render_attempts", "_last_cleaning_report", "_sync_extraction_diagnostics"})
TIP_SHIM = lift(tip_tree, {"render_attempts"})


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


class Boundary:
    def __init__(self, payload, ok=True, failure=None, reconstruction=None, reason=""):
        self.payload, self.ok, self.failure = payload, ok, failure
        self.reconstruction, self.incomplete_reason = reconstruction or {}, reason

    def to_summary(self):
        return {"ok": self.ok, "reason": self.incomplete_reason, "rec": self.reconstruction}


def payload(*names, out=()):
    rx = [{"name": n} for n in names] + [{"name": n, "scope_membership": " Out_Of_Scope "} for n in out]
    return {"entities": {"compounds": []}, "processes": {"reactions": rx}}


ATTEMPTS = [{"attempt": 1, "raw": "{bad", "error": "Expecting value", "phase": "extract", "note": "cut"},
            {"attempt": 2, "raw": "{}", "error": None}, {"attempt": 3, "raw": "[]", "phase": "repair"}]
CLEAN = {"cleaning_report.json": {"passes": [{"pass": 1}, {"pass": 2, "dropped_all": True}]}}

# Each scenario: stage1 = payload or an exception; boundary = kwargs for Boundary (payload
# defaults to the settled stage-1 payload); rag = None (off) or ("result"|None|"raise", evidence)
S = {
    "success": dict(stage1=payload("a", "b")),
    "success-chunked": dict(stage1=payload("a"), chunk_details=[{"chunk": 1}, {"chunk": 2}],
                            enable_chunking=True, sizes=(900.0, 150.0, 4.0, 12000.0)),
    "success-research-mode": dict(stage1=payload("a"), export_mode="research"),
    "success-bad-mode": dict(stage1=payload("a"), export_mode="nonsense"),
    "cleaning-report-last-pass": dict(stage1=payload("a"), artifacts=CLEAN),
    "diagnostics-current-raises": dict(stage1=payload("a"), artifacts="raise"),
    "reconstruction-info": dict(stage1=payload("a"), boundary=dict(reconstruction={"shell_count": 3})),
    "reconstruction-no-count": dict(stage1=payload("a"), boundary=dict(reconstruction={"shells": []})),
    "pipeline-failure": dict(stage1=pipeline.PipelineFailure("stage1", "no JSON after 3", ATTEMPTS)),
    "pipeline-failure-no-attempts": dict(stage1=pipeline.PipelineFailure("stage1", "empty", [])),
    "contract-error": dict(stage1=StageContractError("post_extraction", "Payload must be a dict.",
                                                      {"errors": ["x"], "stage": "post_extraction"})),
    "contract-error-in-settle": dict(stage1=payload("a"), settle_raises=StageContractError(
        "post_extraction", "Entity rows must be objects.", {"errors": ["e"]})),
    "pipeline-failure-in-settle": dict(stage1=payload("a"), settle_raises=pipeline.PipelineFailure(
        "repair", "repair gave up", ATTEMPTS[:1])),
    "incomplete-no-failure": dict(stage1=payload("a"), boundary=dict(ok=False, reason="no processes")),
    "incomplete-with-failure": dict(stage1=payload("a"), boundary=dict(
        ok=False, reason="rows rejected", failure=SimpleNamespace(report={"errors": ["r"]}))),
    "other-exception": dict(stage1=RuntimeError("LLM down")),
    "out-of-scope-removed": dict(stage1=payload("a", out=("x", "y"))),
    "out-of-scope-and-reconstruction": dict(stage1=payload("a", out=("x",)),
                                            boundary=dict(reconstruction={"shell_count": 1})),
    "rag-off": dict(stage1=payload("a"), rag=None, utc="menaquinone"),
    "rag-on-skip": dict(stage1=payload("a"), rag=(None, None), utc="menaquinone", flag=0),
    "rag-on-no-evidence": dict(stage1=payload("a"), rag=("result", ""), utc="menaquinone", flag=1),
    "rag-on-fold": dict(stage1=payload("a", out=("x",)), rag=("result", "PMC1: MenG."), utc="focus", flag=True),
    "rag-on-fold-no-utc": dict(stage1=payload("a"), rag=("result", "PMC1: MenG."), utc=None, flag=False),
    "rag-on-fold-empty-utc": dict(stage1=payload("a"), rag=("result", "PMC1: MenG."), utc="", flag=True),
    "rag-on-error-degraded": dict(stage1=payload("a"), rag=("degraded", None), utc="focus", flag=True),
    "rag-on-raises": dict(stage1=payload("a"), rag=("raise", None), utc="focus", flag=True),
    "rag-on-after-incomplete": dict(stage1=payload("a"), boundary=dict(ok=False, reason="r"),
                                    rag=("result", "E")),
}


def run(arm, name, sc):
    st, calls = FakeSt(), []
    rag_returned = {}

    def rec(label, fn):
        def wrapper(*a, **k):
            calls.append([label, [norm(x) for x in a],
                          {kk: (f"<{type(v).__name__} st={getattr(v, '_st', None) is st}>"
                                if kk == "reporter" else norm(v)) for kk, v in k.items()}])
            return fn(*a, **k)
        return wrapper

    def stage1(*_a, **_k):
        if isinstance(sc["stage1"], Exception):
            raise sc["stage1"]
        return deepcopy(sc["stage1"]), deepcopy(sc.get("chunk_details", []))

    def settle(stage_one, **_k):
        if sc.get("settle_raises"):
            raise sc["settle_raises"]
        bkw = dict(sc.get("boundary") or {})
        return Boundary(dict(stage_one, settled=True), **bkw)

    def current():
        art = sc.get("artifacts", {"x.json": {"n": 1}})
        if art == "raise":
            raise RuntimeError("no recorder")
        return SimpleNamespace(artifacts=lambda: deepcopy(art))

    def rag_config():
        return {"enabled": sc.get("rag") is not None}

    def maybe_run_rag(**_k):
        kind, evidence = sc["rag"]
        if kind == "raise":
            raise RuntimeError("chain blew up")
        if kind is None:
            return None
        result = SimpleNamespace(evidence_context=evidence, payload=None if kind == "degraded" else {"p": 1},
                                 synthesized=kind == "result")
        rag_returned["obj"] = result
        return result

    fakes = {"run_stage_one_with_chunking": rec("stage1", stage1),
             "filter_out_of_scope_reactions": rec("filter", REAL_FILTER),
             "settle_stage_one": rec("settle", settle),
             "build_draft_graph": rec("draft_graph", lambda p: {"graph_of": len(p["processes"]["reactions"])}),
             "generate_qa_report": rec("qa_report", lambda g, p: {"qa": g}),
             "maybe_run_rag": rec("maybe_run_rag", maybe_run_rag),
             "rag_config": rec("rag_config", rag_config)}
    sizes = sc.get("sizes", (1200.0, 200.0, 3.0, 8000.0))
    ns = dict(st=st, text="Menaquinone text.", pathway_context={"pathway_name": "Menaquinone"},
              user_task_context=sc.get("utc"), _diagnostics_dir=DIAG_DIR,
              enable_chunking=sc.get("enable_chunking", False), chunk_size=sizes[0],
              chunk_overlap=sizes[1], extract_attempts=sizes[2], temperature=0.2,
              extract_tokens=sizes[3], rag_incomplete_flag=sc.get("flag", False),
              Any=Any, Dict=Dict, List=List)
    if "export_mode" in sc:
        dict.__setitem__(st.session_state, "export_mode", sc["export_mode"])
    xd.current = rec("diag.current", current)
    if arm == "base":
        ns.update(fakes, PROJECT_ROOT=ROOT, coerce_mode=coerce_mode,
                  PipelineFailure=pipeline.PipelineFailure, StageContractError=StageContractError,
                  StreamlitReporter=StreamlitReporter, CLEANING_REPORT_NAME=xd.CLEANING_REPORT_NAME,
                  current_extraction_diagnostics=xd.current, _safe_dict=_safe_dict, _safe_list=_safe_list,
                  _service_sync_extraction_diagnostics=ex._sync_extraction_diagnostics)
        exec(BASE_HELPERS, ns)
    else:
        paths.PROJECT_ROOT = ROOT
        for (module, attr) in REAL:
            if attr in fakes:
                setattr(module, attr, fakes[attr])
        ns.update(run_stage_one=ex.run_stage_one, StreamlitReporter=StreamlitReporter)
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
    after = {k: norm(ns.get(k, "<unbound>")) for k in (
        "stage_one", "chunk_details", "stage_one_in_scope", "out_of_scope_removed_reactions",
        "rag_result", "user_task_context")}
    after["rag_result_is_returned_object"] = ("obj" in rag_returned and ns.get("rag_result") is rag_returned["obj"])
    return {"st": st.log, "outcome": outcome, "calls": calls, "after": after,
            "globals_only_here": sorted(k for k in ns if k in ("_boundary", "_rag_seed_graph", "_stage_one"))}


bad = 0
for name, sc in S.items():
    a, b = run("base", name, sc), run("tip", name, sc)
    fields = [k for k in a if k != "globals_only_here" and norm(a[k]) != norm(b[k])]
    same = not fields
    # Known, reported difference (as R3b's): when the run ENDS EARLY (st.stop() or a
    # raise), the base had already bound some script globals (stage_one, chunk_details,
    # ...) before the exit; the tip binds them only from a returned result, so there they
    # are unbound. The script run is over either way and Streamlit re-execs the script in
    # a fresh namespace, so nothing can read them. Allowed ONLY if the outcome is an early
    # exit, identical on both arms, everything else is identical, and every differing
    # value is unbound on the tip.
    unbound_only = (fields == ["after"] and a["outcome"] == b["outcome"] and a["outcome"] != "completed"
                    and all(b["after"][k] == norm("<unbound>")
                            for k in a["after"] if a["after"][k] != b["after"][k]))
    bad += not (same or unbound_only)
    n_state = sum(1 for e in a["st"] if e[0] == "session_state")
    verdict = "IDENTICAL" if same else ("SAME-EXCEPT-UNBOUND-GLOBALS" if unbound_only else "DIFFERENT")
    if unbound_only:
        fields = []
        print(f"{name:34s} {verdict}  (tip unbound: "
              f"{sorted(k for k in a['after'] if a['after'][k] != b['after'][k])})")
    print(f"{name:34s} {verdict}  outcome={a['outcome'][:44]:<44s} "
          f"st-events={len(a['st'])} state-writes={n_state} callee-calls={len(a['calls'])} "
          f"dead-globals base={a['globals_only_here']} tip={b['globals_only_here']}")
    if fields:
        print("  differing fields:", fields)
        for f in fields:
            print("   base", f, norm(a[f])[:2500])
            print("   tip ", f, norm(b[f])[:2500])

# render_attempts: base app function vs the tip app's shim over the service function.
for label, attempts in [("Stage 2 attempts", ATTEMPTS), ("Chunk 1 attempts", []),
                        ("x attempts", [{"attempt": 7, "raw": "", "error": "", "note": "n"}])]:
    logs = []
    for arm in ("base", "tip"):
        st = FakeSt()
        ns = dict(st=st, Any=Any, Dict=Dict, List=List, StreamlitReporter=StreamlitReporter,
                  _service_render_attempts=ex.render_attempts)
        exec(BASE_HELPERS if arm == "base" else TIP_SHIM, ns)
        ns["render_attempts"](label, deepcopy(attempts))
        logs.append(st.log)
    same = norm(logs[0]) == norm(logs[1])
    bad += not same
    print(f"render_attempts[{label}]".ljust(35), "IDENTICAL" if same else "DIFFERENT", f"st-events={len(logs[0])}")

total = len(S) + 3
print(f"{total - bad}/{total} identical or known-difference only; {bad} unexplained")
sys.exit(1 if bad else 0)
