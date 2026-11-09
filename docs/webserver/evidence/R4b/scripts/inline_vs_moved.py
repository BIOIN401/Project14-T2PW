"""R4b display-equivalence proof: the BASE app's inline handler vs the TIP app's call site +
run_post_pipeline + StreamlitReporter, against the same recording fake ``st`` and the same
fake callees, over every path in R4-scout section 3 and more.

Both snippets are cut out of the real app files and exec'd at module level, as Streamlit
runs the script, from ``if st.button("Run audit and DB mapping"...`` to the line before the
"Rendered from session_state" comment. The helpers (reset_quarantine_state,
initialize_refinement_review_state, _render_stage_contract_failure, _render_issue_detail,
_sync_extraction_diagnostics) are lifted out of each arm's own app by AST. The fakes reach
the base through its namespace and the tip through the SOURCE modules, which the service
reads at call time.

Compared per scenario: the ordered st.* calls, every session_state write AND pop in one
combined order, each callee's calls with arguments, the outcome (or the raised exception),
the final state, and the script globals afterwards. A globals difference is allowed
(SAME-EXCEPT-DEAD-GLOBALS) only if everything else is identical and every differing name is
dead after the handler in the tip app (static check).

usage: inline_vs_moved.py BASE_TREE TIP_TREE     (PYTHONPATH must be TIP_TREE/src)
"""
import ast
import json
import logging
import sys
import tempfile
import textwrap
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

base_tree, tip_tree = sys.argv[1], sys.argv[2]

import t2pw.config as config  # noqa: E402
import t2pw.paths as paths  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as xd  # noqa: E402
import t2pw.pipeline.pipeline as pipeline  # noqa: E402
import t2pw.service.extraction as ex  # noqa: E402
import t2pw.service.post_run as pr  # noqa: E402
import t2pw.service.postpipeline as postpipeline  # noqa: E402
from t2pw.pipeline.export_mode import coerce_mode, is_research  # noqa: E402
from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY, FINAL_GATE_REPORT_KEY  # noqa: E402
from t2pw.pipeline.stage_contracts import StageContractError  # noqa: E402
from t2pw.service.postpipeline import _safe_dict, _safe_list  # noqa: E402
from t2pw.service.reporter import StreamlitReporter  # noqa: E402

assert pr.__file__.replace("\\", "/").startswith(tip_tree.replace("\\", "/")), pr.__file__
ROOT = Path(tempfile.mkdtemp(prefix="r4b-ivm-"))
REAL = {(m, n): getattr(m, n) for m, n in [
    (postpipeline, "run_post_pipeline_sbml_artifacts"), (postpipeline, "_record_mapped_failure_snapshot"),
    (pipeline, "propagate_context_organism"), (xd, "reenter"), (xd, "current"), (config, "rag_config")]}
REAL_ROOT = paths.PROJECT_ROOT
DEAD = ["artifacts", "_pa", "final_mapped_payload", "mapping_report", "_quarantine_flags", "_refusal_reasons",
        "final_gate", "_stage3_failed", "_gate_err", "_gate_errors_for_display", "failure", "exc", "reason", "_post_run"]
HELPERS = {"reset_quarantine_state", "initialize_refinement_review_state", "_render_stage_contract_failure",
           "_render_issue_detail", "_sync_extraction_diagnostics"}


def _default(o):
    if isinstance(o, Path):
        return f"<path {o.as_posix().replace(ROOT.as_posix(), '<ROOT>')}>"
    return f"<{type(o).__name__}>"


def norm(value):
    return json.dumps(value, sort_keys=True, default=_default)


def app_src(tree):
    return Path(tree, "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8")


FIRST = '    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):'
LAST = "    # Rendered from session_state on EVERY pass, not inside the Run button's"


def cut(tree):
    lines = app_src(tree).splitlines()
    start, end = lines.index(FIRST), lines.index(LAST)
    return textwrap.dedent("\n".join(lines[start:end]))


def lift(tree):
    keep = [n for n in ast.parse(app_src(tree)).body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert {n.name for n in keep} == HELPERS, [n.name for n in keep]
    return compile(ast.Module(body=keep, type_ignores=[]), f"{tree}/streamlit_app.py", "exec")


BASE_CUT = cut(base_tree)
TIP_CUT = cut(tip_tree)
assert "run_post_pipeline_sbml_artifacts(" in BASE_CUT and "run_post_pipeline(" not in BASE_CUT
assert "run_post_pipeline(" in TIP_CUT and "run_post_pipeline_sbml_artifacts" not in TIP_CUT
HELPER_CODE = {arm: lift(tree) for arm, tree in (("base", base_tree), ("tip", tip_tree))}


def dead_after_handler(tree):
    mod = ast.parse(app_src(tree))
    handler = next(n for n in ast.walk(mod) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                   and isinstance(n.test.func, ast.Attribute) and n.test.func.attr == "button"
                   and n.test.args and isinstance(n.test.args[0], ast.Constant)
                   and n.test.args[0].value == "Run audit and DB mapping")
    funcs = set()
    for f in ast.walk(mod):
        if isinstance(f, (ast.FunctionDef, ast.Lambda)):
            funcs.update(id(n) for n in ast.walk(f))
    problems = []
    for name in DEAD:
        loads, stores = [], []
        for node in ast.walk(mod):
            line = getattr(node, "lineno", 0)
            if line <= handler.end_lineno or id(node) in funcs:
                continue
            if isinstance(node, ast.Name) and node.id == name:
                (loads if isinstance(node.ctx, ast.Load) else stores).append(node.lineno)
            elif isinstance(node, ast.ExceptHandler) and node.name == name:
                stores.append(node.lineno)
        first = min(stores) if stores else None
        bad = [ln for ln in loads if first is None or ln <= first]
        if bad:
            problems.append((name, bad, first))
    return problems


STATIC = dead_after_handler(tip_tree)
print("static dead-globals check (tip app, after the handler):", "OK" if not STATIC else STATIC)


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
        object.__setattr__(self, "_log", log)

    def __setitem__(self, key, value):
        self._log.append(["session_state", key, norm(value)])
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        self[key] = value

    def pop(self, key, *default):
        self._log.append(["session_state.pop", key])
        return super().pop(key, *default)


class FakeSt:
    def __init__(self, sc):
        self.log = []
        self.session_state = State(self.log)
        self._sc = sc

    def button(self, label, key=None, **_k):
        self.log.append(["button", label, key])
        return bool(self._sc.get("click", True)) and key == "pwml_generate_btn"

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append([name, [norm(x) for x in a], {kk: norm(v) for kk, v in k.items()}])
            if name in ("spinner", "expander"):
                return Ctx(self.log, name)
            if name == "download_button" and self._sc.get("download_raises"):
                raise TypeError("not serialisable")
        return call


class Graph:
    def to_dict(self):
        return {"g": 1}


WIDGETS = {"use_llm_audit": True, "use_sbml_overwatch": False, "default_compartment": "  nucleus ",
           "mapping_cache_text": " cache.json ", "id_source_mode": " DB ", "db_host": " h ", "db_port": 3306.0,
           "db_user": " u ", "db_password": "pw", "db_schema": " sch ", "audit_max_rounds": 4.0,
           "audit_timeout_seconds": 240.0, "audit_candidate_count": 3.0, "use_example_retrieval": 0,
           "example_index_path": " i.json ", "example_top_k": 3.0, "use_gap_resolver": 1, "use_llm_gap_resolver": 0,
           "gap_resolver_max_items": 80.0, "use_stoich_agent": 0}
PAYLOAD = {"entities": {"compounds": []}, "processes": {"reactions": [{"name": "a"}]}}
GATE_ERRS = [{"path": "/a", "reason": "bad", "detail": {"summary": "s", "x": 1}}, {"path": "/b", "reason": "b"}, "plain"]
MAPPED = {"mapped": True, "processes": {"reactions": []}}


def contract(stage="post_mapping"):
    return StageContractError(stage, f"boundary {stage} failed",
                              {"stage": stage, "errors": [{"code": "c", "message": "m", "detail": {"k": 1}}],
                               "warnings": [{"code": "w", "reason": "r"}], "summary": {"n": 1}, "effect_on_failure": "abort"})


def ok_arts(**kw):
    return {CANONICAL_PAYLOAD_KEY: dict(MAPPED), "quarantine_ok": True, FINAL_GATE_REPORT_KEY: {"ok": True},
            "mapping_report": {"misses": []}, **kw}


S = {
    "not-a-dict-none": dict(final_payload=None),
    "not-a-dict-list": dict(final_payload=["x"]),
    "not-a-dict-empty": dict(final_payload={}),
    "gate-failed": dict(artifacts=lambda: {"gate_failed": True}),
    "gate-failed-post-audit-reports": dict(artifacts=lambda: {"gate_failed": True, "post_audit_qa_report": {"q": 2},
                                                              "post_audit_reaction_summary": "new"},
                                           seed={"qa_report": {"q": 1}, "reaction_summary": "old"}),
    "gate-failed-seed-snapshot": dict(artifacts=lambda: {"gate_failed": True}, seed={"final_payload_snapshot": {"s": 1}}),
    "artifacts-not-a-dict": dict(artifacts=lambda: None),
    "no-mapped-payload": dict(artifacts=lambda: {}),
    "mapped-payload-empty": dict(artifacts=lambda: {CANONICAL_PAYLOAD_KEY: {}, "quarantine_ok": True}),
    "final-mapped-fallback": dict(artifacts=lambda: {"final_mapped": dict(MAPPED), "quarantine_ok": True,
                                                     FINAL_GATE_REPORT_KEY: {"ok": True}}),
    "pass": dict(artifacts=ok_arts),
    "pass-stale-errors-replaced": dict(artifacts=ok_arts, seed={"refinement_gate_errors": ["stale"]}),
    "pass-with-quarantine-flags": dict(artifacts=lambda: ok_arts(quarantine_review_flags=["f1", "f2"])),
    "pass-no-final-gate": dict(artifacts=lambda: {CANONICAL_PAYLOAD_KEY: dict(MAPPED), "quarantine_ok": True}),
    "quarantine-refused": dict(artifacts=lambda: {CANONICAL_PAYLOAD_KEY: dict(MAPPED), "quarantine_ok": False,
                                                  "quarantine_refusal_reasons": ["no core", 7],
                                                  "quarantine_coverage": {"c": 1},
                                                  "quarantine_report": {"strict_invariants": {"i": 1}}}),
    "quarantine-refused-with-flags": dict(artifacts=lambda: {CANONICAL_PAYLOAD_KEY: dict(MAPPED), "quarantine_ok": False,
                                                             "quarantine_review_flags": ["f"], "quarantine_refusal_reasons": []}),
    "research-stage3-fail": dict(artifacts=lambda: ok_arts(**{FINAL_GATE_REPORT_KEY: {"ok": False, "errors": GATE_ERRS}}),
                                 seed={"export_mode": "research"}),
    "research-stage3-no-errors": dict(artifacts=lambda: ok_arts(**{FINAL_GATE_REPORT_KEY: {"ok": False, "errors": []}}),
                                      seed={"export_mode": "research"}),
    "strict-stage3-fail": dict(artifacts=lambda: ok_arts(**{FINAL_GATE_REPORT_KEY: {"ok": False, "errors": GATE_ERRS}})),
    "strict-stage3-no-errors": dict(artifacts=lambda: ok_arts(**{FINAL_GATE_REPORT_KEY: {"ok": False, "errors": []}})),
    "contract-post-mapping": dict(raises=contract("post_mapping")),
    "contract-post-extraction": dict(raises=contract("post_extraction")),
    "contract-other-stage": dict(raises=contract("weird")),
    "contract-render-raises": dict(raises=contract(), download_raises=True),
    "exception": dict(raises=RuntimeError("boom")),
    "exception-from-propagate": dict(propagate_raises=RuntimeError("no organism")),
    "reenter-raises": dict(reenter_raises=RuntimeError("recorder gone")),
    "review-raises": dict(artifacts=ok_arts, draft_raises=True),
    "diag-sync-raises": dict(artifacts=ok_arts, diag_raises=True),
    "rag-on-with-evidence": dict(artifacts=ok_arts, rag=True, seed={"rag_evidence_context": "EV"}),
    "rag-on-no-evidence": dict(artifacts=ok_arts, rag=True),
    "rag-off-with-evidence": dict(artifacts=ok_arts, rag=False, seed={"rag_evidence_context": "EV"}),
    "strict-db-false-qa-seeded": dict(artifacts=ok_arts, seed={"pwml_strict_db": False, "qa_report": {"q": 1},
                                                               "reaction_summary": "r", "export_mode": "research"}),
    "widgets-blank": dict(artifacts=ok_arts, widgets=dict(default_compartment="  ", mapping_cache_text="  ",
                                                          id_source_mode=None, db_host=None, db_user=None,
                                                          db_password=None, db_schema="  ", example_index_path=None)),
    "stale-keys-popped": dict(artifacts=ok_arts, seed={"post_pipeline_artifacts": {"old": 1}, "quarantine_report": {"q": 1},
                                                       "quarantine_ok": True}),
    "not-clicked": dict(click=False),
}


def run(arm, name, sc):
    st, calls = FakeSt(sc), []

    def rec(label, fn):
        def wrapper(*a, **k):
            calls.append([label, [norm(x) for x in a],
                          {kk: f"<state is session_state={v is st.session_state}>" if kk == "state"
                           else norm(v) for kk, v in k.items()}])
            return fn(*a, **k)
        return wrapper

    def artifacts(payload, **k):
        if sc.get("raises") is not None:
            raise sc["raises"]
        return sc["artifacts"]() if sc.get("artifacts") else {}

    def propagate(payload, ctx):
        if sc.get("propagate_raises") is not None:
            raise sc["propagate_raises"]
        return {**deepcopy(payload), "organism": "rebound"}

    def reenter(recorder):
        if sc.get("reenter_raises") is not None:
            raise sc["reenter_raises"]

    def render_png(graph_dict, dpi=100):
        if sc.get("draft_raises"):
            raise RuntimeError("png failed")
        return b"png"

    recorder = SimpleNamespace(artifacts=(lambda: (_ for _ in ()).throw(RuntimeError("diag")))
                               if sc.get("diag_raises") else (lambda: {"diag": 1}))

    fakes = {"run_post_pipeline_sbml_artifacts": rec("artifacts", artifacts),
             "_record_mapped_failure_snapshot": rec("snapshot", lambda *a, **k: None),
             "propagate_context_organism": rec("propagate", propagate)}
    widgets = {**WIDGETS, **(sc.get("widgets") or {})}
    ns = dict(st=st, final_payload=sc.get("final_payload", deepcopy(PAYLOAD)), post_artifacts="<before>",
              PROJECT_ROOT=ROOT, Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, Path=Path,
              StreamlitReporter=StreamlitReporter, json=json, deepcopy=deepcopy, logger=logging.getLogger("ivm"),
              QUARANTINE_STATE_KEYS=("quarantine_report", "quarantine_artifacts", "quarantine_coverage", "quarantine_ok",
                                     "quarantine_decision_controls"),
              clear_quarantine_artifacts=rec("clear_quarantine_artifacts", lambda d: None),
              build_draft_graph=rec("build_draft_graph", lambda p: Graph()),
              render_draft_graph_to_png_bytes=rec("render_png", render_png),
              generate_qa_report=rec("generate_qa_report", lambda g, p: {"qa": 1}),
              compact_mapping_misses=lambda r: [],
              _safe_dict=_safe_dict, _safe_list=_safe_list, detail_headline=lambda d: "headline",
              StageContractError=StageContractError,
              _service_sync_extraction_diagnostics=ex._sync_extraction_diagnostics, **widgets)
    for key, value in (sc.get("seed") or {}).items():
        dict.__setitem__(st.session_state, key, value)
    dict.__setitem__(st.session_state, "extraction_diagnostics_recorder", "RECORDER")
    dict.__setitem__(st.session_state, "pathway_context", {"pathway_name": "x"})
    xd.current = lambda: recorder
    xd.reenter = rec("reenter", reenter)
    config.rag_config = lambda *a, **k: {"enabled": bool(sc.get("rag"))}
    if arm == "base":
        ns.update(fakes, reenter_extraction_diagnostics=xd.reenter, rag_config=config.rag_config, coerce_mode=coerce_mode,
                  is_research=is_research, CANONICAL_PAYLOAD_KEY=CANONICAL_PAYLOAD_KEY, FINAL_GATE_REPORT_KEY=FINAL_GATE_REPORT_KEY)
    else:
        paths.PROJECT_ROOT = ROOT
        for name_, fake in fakes.items():
            module = postpipeline if name_ != "propagate_context_organism" else pipeline
            setattr(module, name_, fake)
        ns.update(run_post_pipeline=pr.run_post_pipeline, PostRunRequest=pr.PostRunRequest)
    exec(HELPER_CODE[arm], ns)
    outcome = "completed"
    try:
        exec(compile(BASE_CUT if arm == "base" else TIP_CUT, f"<{arm}:{name}>", "exec"), ns)
    except Exception as exc:  # noqa: BLE001 - the raise itself is compared
        outcome = f"raised {type(exc).__name__}: {exc}"
    finally:
        for (module, attr), real in REAL.items():
            setattr(module, attr, real)
        paths.PROJECT_ROOT = REAL_ROOT
    after = {k: norm(ns.get(k, "<unbound>")) for k in DEAD}
    return {"st": st.log, "outcome": outcome, "calls": calls, "after": after,
            "final_payload": norm(ns.get("final_payload")), "post_artifacts": norm(ns.get("post_artifacts")),
            "state_final": norm(dict(st.session_state))}


bad = 0
for name, sc in S.items():
    a, b = run("base", name, sc), run("tip", name, sc)
    fields = [k for k in a if norm(a[k]) != norm(b[k])]
    # A script that raised has ended: nothing can read the app's final_payload global (Streamlit
    # reports the exception and the next run reads the session). The rebind is compared only
    # when the handler completed.
    raised_same = a["outcome"].startswith("raised") and a["outcome"] == b["outcome"]
    if raised_same and "final_payload" in fields:
        fields.remove("final_payload")
    differing = sorted(k for k in a["after"] if a["after"][k] != b["after"][k])
    static_bad = {p[0] for p in STATIC}
    dead_only = fields == ["after"] and all(k in DEAD and k not in static_bad for k in differing)
    verdict = "IDENTICAL" if not fields else ("SAME-EXCEPT-DEAD-GLOBALS" if dead_only else "DIFFERENT")
    bad += verdict == "DIFFERENT"
    n_state = sum(1 for e in a["st"] if e[0].startswith("session_state"))
    print(f"{name:30s} {verdict:25s} outcome={a['outcome'][:40]:<40s} st-events={len(a['st'])} "
          f"state-ops={n_state} callee-calls={len(a['calls'])}")
    if dead_only:
        print(f"{'':30s}   differing globals (dead after the handler): {differing}")
    if verdict == "DIFFERENT":
        print("  differing fields:", fields)
        for f in fields:
            print("   base", f, norm(a[f])[:2500])
            print("   tip ", f, norm(b[f])[:2500])
bad += bool(STATIC)
print(f"{len(S) - (bad - bool(STATIC))}/{len(S)} identical or known-difference only; "
      f"static check {'OK' if not STATIC else 'FAILED'}; {bad} unexplained")
sys.exit(1 if bad else 0)
