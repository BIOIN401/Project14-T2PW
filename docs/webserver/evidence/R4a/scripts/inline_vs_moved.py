"""R4a display-equivalence proof: the BASE app's inline "Generate PWML" step vs the TIP app's
call site + run_export + StreamlitReporter, against the same recording fake ``st`` and the
same fake ``run_pwml_export``, over every branch.

Both sides are cut out of the real app files by AST: the four moved helpers (base only) and
the ``if pwml_col.button("Generate PWML", ...)`` statement (both). The tip side uses the
committed service module; the fake reaches it through ``postpipeline.run_pwml_export``, which
is what the service reads at call time. ``_safe_dict``, ``_safe_list`` and ``coerce_mode`` are
the real ones on both sides.

Compared per scenario: one ordered event stream (st calls with spinner enter/exit, session
writes with values, the export call with every argument, the payload's identity, the ready
flag and the snapshot bytes at that moment), then the final session, the artifacts' identity
and the files on disk. Paths are normalised to <ROOT>.

Non-vacuity: every sabotage below edits the tip's service source in memory only (the export
stays faked) and must make at least one scenario differ.

usage: inline_vs_moved.py BASE_TREE TIP_TREE OUT.json     (PYTHONPATH must be TIP_TREE/src)
"""
import ast
import contextlib
import json
import shutil
import sys
import tempfile
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import t2pw.service.export as export  # noqa: E402
import t2pw.service.postpipeline as postpipeline  # noqa: E402
from t2pw.pipeline.export_mode import coerce_mode  # noqa: E402
from t2pw.service.postpipeline import _safe_dict, _safe_list  # noqa: E402
from t2pw.service.reporter import StreamlitReporter  # noqa: E402

base_tree, tip_tree, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
assert export.__file__.replace("\\", "/").startswith(tip_tree.replace("\\", "/")), export.__file__
MOVED = ["quarantine_session_state", "_pwml_reference_path", "_write_reviewed_payload_snapshot",
         "_generate_pwml_from_refinement_working_json"]
UNSET = object()


def app_tree(tree):
    return ast.parse(Path(tree, "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8"))


def button_block(module):
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_render_review_refine_section")
    return next(n for n in ast.walk(function)
                if isinstance(n, ast.If) and isinstance(n.test, ast.Call) and getattr(n.test.func, "attr", "") == "button"
                and getattr(n.test.args[0], "value", "") == "Generate PWML")


BASE_MODULE, TIP_MODULE = app_tree(base_tree), app_tree(tip_tree)
BASE_HELPERS = [next(n for n in BASE_MODULE.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in MOVED]
BASE_BLOCK, TIP_BLOCK = button_block(BASE_MODULE), button_block(TIP_MODULE)


def compiled(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), "<lifted>", "exec")


def norm(value, root):
    text = json.dumps(value, sort_keys=True, default=lambda o: f"<{type(o).__name__}:{o}>", ensure_ascii=False)
    # The temp folder's own name, so a path repr'd inside an exception message (more backslashes) normalises too.
    return text.replace(str(root).replace("\\", "\\\\"), "<ROOT>").replace(str(root), "<ROOT>").replace(root.name, "<ROOT>")


class Session:
    """Mapping with attribute access, recording every write, as st.session_state is read and written."""

    def __init__(self, initial, log, root):
        object.__setattr__(self, "data", dict(initial))
        object.__setattr__(self, "log", log)
        object.__setattr__(self, "root", root)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.log.append(["session-set", key, norm(value, self.root)])
        self.data[key] = value

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        return self.data[key]


class FakeSt:
    def __init__(self, session, log, root):
        self.session_state, self.log, self.root = session, log, root

    @contextlib.contextmanager
    def spinner(self, text, **kwargs):
        self.log.append(["spinner-enter", text, kwargs])
        try:
            yield
        finally:
            self.log.append(["spinner-exit", text])

    def _say(self, kind, body):
        self.log.append([kind, norm(body, self.root)])

    def success(self, body):
        self._say("success", body)

    def error(self, body):
        self._say("error", body)


class Column:
    def __init__(self, clicked, log):
        self.clicked, self.log = clicked, log

    def button(self, label, key):
        self.log.append(["button", label, key])
        return self.clicked


def run_arm(arm, scenario, run_export):
    root = Path(tempfile.mkdtemp(prefix="r4a-ivm-"))
    log = []
    try:
        for name in scenario.get("references", []):
            (root / "reference").mkdir(exist_ok=True)
            (root / "reference" / name).write_text("ref", encoding="utf-8")
        work = root / "tmp" / "nested"
        if scenario.get("work_is_file"):
            work.parent.mkdir(parents=True)
            work.write_text("a file", encoding="utf-8")
        work_dir = str(work) if scenario.get("work_as_str") else work
        initial = deepcopy(scenario["state"])
        session = Session(initial, log, root)
        st = FakeSt(session, log, root)
        artifacts_before = session.get("post_pipeline_artifacts")

        def fake_export(payload, *args, **kwargs):
            log.append(["export", norm(payload, root), norm(list(args), root), norm(kwargs, root),
                        payload is session.get("refinement_working_json"),
                        session.get("refinement_pwml_ready", "<unset>"),
                        (work / "final.mapped.json").read_bytes().decode("utf-8")])
            if scenario.get("export") == "raise":
                raise RuntimeError("export blew up")
            return deepcopy(scenario.get("export", {"ok": True, "output_path": "out.pwml"}))

        ns: Dict[str, Any] = {"st": st, "pwml_col": Column(scenario.get("clicked", True), log), "work_dir": work_dir,
                              "PROJECT_ROOT": root}
        real = postpipeline.run_pwml_export
        raised = None
        try:
            if arm == "base":
                ns.update(json=json, Path=Path, deepcopy=deepcopy, Dict=Dict, Any=Any, _safe_dict=_safe_dict,
                          _safe_list=_safe_list, coerce_mode=coerce_mode, run_pwml_export=fake_export)
                exec(compiled(BASE_HELPERS), ns)
                exec(compiled([BASE_BLOCK]), ns)
            else:
                postpipeline.run_pwml_export = fake_export
                ns.update(run_export=run_export, StreamlitReporter=StreamlitReporter)
                exec(compiled([TIP_BLOCK]), ns)
        except BaseException as exc:  # noqa: BLE001 -- compared, not swallowed
            raised = [type(exc).__name__, str(exc).replace(str(root), "<ROOT>")]
        finally:
            postpipeline.run_pwml_export = real
        files = {p.relative_to(root).as_posix(): p.read_bytes().decode("utf-8")
                 for p in sorted(root.rglob("*")) if p.is_file()}
        return {"log": log, "raised": raised, "final_session": norm(session.data, root), "files": files,
                "artifacts_same_object": session.get("post_pipeline_artifacts") is artifacts_before}
    finally:
        shutil.rmtree(root, ignore_errors=True)


FULL = {"pwml_name": "N", "pwml_description": "D", "pwml_subject": "S", "pwml_width": "10", "pwml_height": 20,
        "pwml_bg": "#000000", "pwml_strict_db": False, "pathway_context": {"o": 1}, "export_mode": "research",
        "quarantine_report": {"r": 1}, "quarantine_artifacts": {"q": 2}}
W = {"refinement_working_json": {"name": "β-carotene", "n": [1, 2]}}
SCENARIOS = {
    "not-clicked": {"state": dict(W), "clicked": False},
    "no-payload-absent": {"state": {}},
    "no-payload-empty": {"state": {"refinement_working_json": {}}},
    "no-payload-list": {"state": {"refinement_working_json": ["x"]}},
    "no-payload-none": {"state": {"refinement_working_json": None}},
    "gate-errors": {"state": {**W, "refinement_gate_errors": ["e1", "e2"], "quarantine_report": {"r": 1},
                              "quarantine_artifacts": {"q": 2}}},
    "gate-errors-bad-quarantine": {"state": {**W, "refinement_gate_errors": ["e1"], "quarantine_report": "x"}},
    "gate-errors-not-a-list": {"state": {**W, "refinement_gate_errors": "not a list"}},
    "defaults": {"state": dict(W)},
    "work-dir-as-str": {"state": dict(W), "work_as_str": True},
    "work-dir-is-a-file": {"state": dict(W), "work_is_file": True},
    "all-settings": {"state": {**W, **FULL}},
    "falsy-settings": {"state": {**W, "pwml_name": "", "pwml_description": None, "pwml_subject": "", "pwml_width": 0,
                                 "pwml_height": 0, "pwml_bg": "", "pathway_context": "not a dict"}},
    "bad-width": {"state": {**W, "pwml_width": "wide"}},
    "strict-explicit-true": {"state": {**W, "pwml_strict_db": True}},
    "mode-pathwhiz": {"state": {**W, "export_mode": "pathwhiz"}},
    "mode-garbage": {"state": {**W, "export_mode": "garbage"}},
    "grounding-on": {"state": {**W, "pwml_grounding": True, "pwml_grounding_dict": {"g": 1}}},
    "grounding-on-no-dict": {"state": {**W, "pwml_grounding": True}},
    "grounding-off-with-dict": {"state": {**W, "pwml_grounding": False, "pwml_grounding_dict": {"g": 1}}},
    "artifacts-dict": {"state": {**W, "post_pipeline_artifacts": {"keep": 1, "final_mapped_db": "old"}}},
    "artifacts-list": {"state": {**W, "post_pipeline_artifacts": ["a"]}},
    "artifacts-none": {"state": {**W, "post_pipeline_artifacts": None}},
    "reference-first": {"state": dict(W), "references": ["PW000001.pwml", "PW012926.pwml"]},
    "reference-second-only": {"state": dict(W), "references": ["PW012926.pwml"]},
    "export-not-ok": {"state": dict(W), "export": {"ok": False, "error": "the gate rejected the payload"}},
    "export-not-ok-no-error": {"state": dict(W), "export": {"ok": False}},
    "export-ok-extra-keys": {"state": {**W, **FULL}, "export": {"ok": True, "output_path": "o.pwml", "qa": {"a": 1}}},
    "export-raises": {"state": {**W, "post_pipeline_artifacts": {"keep": 1}}, "export": "raise"},
}
SABOTAGES = {
    "spinner-text": ('reporter.spinner("Building PWML from reviewed pathway JSON...")', 'reporter.spinner("Building PWML...")'),
    "no-payload-message": ('"No reviewed mapped pathway is available for PWML export."', '"No payload."'),
    "payload-guard-dropped": ('if not isinstance(working_json, dict) or not working_json:', 'if working_json is None:'),
    "gate-message": ('"PWML export stopped by pre-export Stage 3 revalidation. "', '"Stopped. "'),
    "gate-quarantine-dropped": ('"stage3_gate_errors": refinement_gate_errors,\n', '"stage3_gate_errors": refinement_gate_errors,\n            } or {\n'),
    "ready-after-export": ('state["refinement_pwml_ready"] = True\n', 'pass\n'),
    "ready-false": ('state["refinement_pwml_ready"] = True', 'state["refinement_pwml_ready"] = False'),
    "snapshot-before-ready-swapped": ('_write_reviewed_payload_snapshot(working_json, work_dir, state=state)\n        state["refinement_pwml_ready"] = True\n',
                                      'state["refinement_pwml_ready"] = True\n        _write_reviewed_payload_snapshot(working_json, work_dir, state=state)\n'),
    "snapshot-ascii": ('ensure_ascii=False', 'ensure_ascii=True'),
    "snapshot-indent": ('indent=2', 'indent=4'),
    "snapshot-name": ('"final.mapped.json"', '"final.json"'),
    "artifacts-guard": ('if isinstance(post_artifacts, dict):', 'if post_artifacts:'),
    "artifacts-not-copied": ('post_artifacts["final_mapped_db"] = reviewed_payload', 'post_artifacts["final_mapped_db"] = {}'),
    "artifacts-path": ('str(reviewed_json_path)', 'reviewed_json_path.name'),
    "artifacts-not-stored": ('state["post_pipeline_artifacts"] = post_artifacts', 'pass'),
    "grounding-default": ('state.get("pwml_grounding", False)', 'state.get("pwml_grounding", True)'),
    "name-default": ('or "Generated Pathway"', 'or "Pathway"'),
    "subject-default": ('or "Metabolic"', 'or "Signaling"'),
    "width-default": ('or 3200', 'or 1600'),
    "height-default": ('or 1400', 'or 700'),
    "bg-default": ('or "#FFFFFF"', 'or "#000000"'),
    "strict-default": ('state.get("pwml_strict_db", True)', 'state.get("pwml_strict_db", False)'),
    "context-not-safe": ('pathway_context=_postpipeline._safe_dict(state.get("pathway_context"))', 'pathway_context=state.get("pathway_context")'),
    "mode-not-coerced": ('_export_mode.coerce_mode(state.get("export_mode"))', 'state.get("export_mode")'),
    "reference-order": ('project_root / "reference" / "PW000001.pwml",\n        project_root / "reference" / "PW012926.pwml",',
                        'project_root / "reference" / "PW012926.pwml",\n        project_root / "reference" / "PW000001.pwml",'),
    "project-root-ignored": ('            project_root=project_root,\n', '            project_root=Path("elsewhere"),\n'),
    "quarantine-keys-swapped": ('"quarantine_report": _postpipeline._safe_dict(state.get("quarantine_report"))',
                                '"quarantine_report": _postpipeline._safe_dict(state.get("quarantine_artifacts"))'),
}


def compare(run_export):
    diffs = []
    for name, scenario in SCENARIOS.items():
        base, tip = run_arm("base", scenario, None), run_arm("tip", scenario, run_export)
        if base != tip:
            diffs.append(name)
    return diffs


def sabotaged(old, new):
    source = Path(export.__file__).read_text(encoding="utf-8")
    assert source.count(old) >= 1, old
    module = types.ModuleType("export_sabotaged")
    exec(compile(source.replace(old, new), "export_sabotaged", "exec"), module.__dict__)
    return module.run_export


def main():
    results = {"scenarios": {}, "sabotage": {}}
    for name, scenario in SCENARIOS.items():
        base, tip = run_arm("base", scenario, None), run_arm("tip", scenario, export.run_export)
        results["scenarios"][name] = "SAME" if base == tip else {"base": base, "tip": tip}
    for name, (old, new) in SABOTAGES.items():
        results["sabotage"][name] = compare(sabotaged(old, new))
    same = sum(1 for v in results["scenarios"].values() if v == "SAME")
    detected = sum(1 for v in results["sabotage"].values() if v)
    undetected = sorted(k for k, v in results["sabotage"].items() if not v)
    Path(out_path).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"scenarios SAME {same}/{len(SCENARIOS)}; sabotage detected {detected}/{len(SABOTAGES)}; undetected={undetected}")
    for name, value in results["scenarios"].items():
        if value != "SAME":
            print("DIFF", name)
    return 0 if same == len(SCENARIOS) and detected == len(SABOTAGES) else 1


if __name__ == "__main__":
    sys.exit(main())
