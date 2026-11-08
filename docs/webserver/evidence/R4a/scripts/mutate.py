"""R4a mutation check, on a SCRATCH COPY only: each single mutation of service/export.py or of
the app's call site must make tests/test_service_export.py fail. A disarmed fake can make
real code run, so this never touches a worktree: the copy is a ``git archive`` of the commit
under test, and the mutated file is restored after every run.

The whole batch runs inside one bounded_run with the heavy lock (runpy.sh); each pytest run is
pinned_pytest.py with its own --basetemp. Pytest exit 1 is a kill; exit 0 is a survivor;
anything else is an invalid mutation and fails the whole check.

usage: mutate.py SCRATCH_TREE OUTDIR [LO:HI]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

WRAPPER_TREE = "C:/t/ws-R4a"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
TREE = Path(sys.argv[1])
EV = Path(sys.argv[2])
EV.mkdir(parents=True, exist_ok=True)
Path("C:/t/wsR4atmp/mut-bt").mkdir(parents=True, exist_ok=True)
SVC = TREE / "src/t2pw/service/export.py"
APP = TREE / "src/t2pw/app/streamlit_app.py"
TOP = "from t2pw.service.reporter import Reporter\n"


def bound(import_line, old, new):
    return [TOP, old], [TOP + import_line + "\n", new]


CALL = '            _pwml_result = run_export(\n                state=st.session_state,\n'
MUTATIONS = {  # name: (file, old (str or list), new (str or list))
    # call-time lookups reverted to import-time bindings
    "bound-run_pwml_export": (SVC, *bound("from t2pw.service.postpipeline import run_pwml_export as _bound",
                                          "return _postpipeline.run_pwml_export(", "return _bound(")),
    "bound-coerce_mode": (SVC, *bound("from t2pw.pipeline.export_mode import coerce_mode as _bound",
                                      "_export_mode.coerce_mode(", "_bound(")),
    "bound-safe_dict": (SVC, *bound("from t2pw.service.postpipeline import _safe_dict as _bound",
                                    "_postpipeline._safe_dict(", "_bound(")),
    "bound-safe_list": (SVC, *bound("from t2pw.service.postpipeline import _safe_list as _bound",
                                    "_postpipeline._safe_list(", "_bound(")),
    # service behaviour
    "spinner-text": (SVC, 'reporter.spinner("Building PWML from reviewed pathway JSON...")', 'reporter.spinner("x")'),
    "no-payload-message": (SVC, '"No reviewed mapped pathway is available for PWML export."', '"No payload."'),
    "payload-guard-dropped": (SVC, 'if not isinstance(working_json, dict) or not working_json:', 'if working_json is None:'),
    "gate-message": (SVC, '"PWML export stopped by pre-export Stage 3 revalidation. "', '"Stopped. "'),
    "gate-issues": (SVC, '"issues": len(refinement_gate_errors),', '"issues": 0,'),
    "gate-quarantine-dropped": (SVC, '                **quarantine_session_state(state=state),\n            }\n', '            }\n'),
    "export-quarantine-dropped": (SVC, '            **quarantine_session_state(state=state),\n        )\n', '        )\n'),
    "ready-flag-removed": (SVC, 'state["refinement_pwml_ready"] = True\n', 'pass\n'),
    "ready-false": (SVC, 'state["refinement_pwml_ready"] = True', 'state["refinement_pwml_ready"] = False'),
    "ready-before-snapshot": (SVC, '_write_reviewed_payload_snapshot(working_json, work_dir, state=state)\n        state["refinement_pwml_ready"] = True\n',
                              'state["refinement_pwml_ready"] = True\n        _write_reviewed_payload_snapshot(working_json, work_dir, state=state)\n'),
    "snapshot-removed": (SVC, '        _write_reviewed_payload_snapshot(working_json, work_dir, state=state)\n', ''),
    "snapshot-ascii": (SVC, 'ensure_ascii=False', 'ensure_ascii=True'),
    "snapshot-indent": (SVC, 'indent=2', 'indent=4'),
    "snapshot-name": (SVC, '"final.mapped.json"', '"final.json"'),
    "artifacts-guard": (SVC, 'if isinstance(post_artifacts, dict):', 'if post_artifacts is not None:'),
    "artifacts-db-not-a-copy": (SVC, 'reviewed_payload = deepcopy(working_json)', 'reviewed_payload = working_json'),
    "artifacts-input-not-a-copy": (SVC, 'post_artifacts["final_export_input"] = deepcopy(reviewed_payload)', 'post_artifacts["final_export_input"] = reviewed_payload'),
    "artifacts-path": (SVC, 'str(reviewed_json_path)', 'reviewed_json_path.name'),
    "grounding-default": (SVC, 'state.get("pwml_grounding", False)', 'state.get("pwml_grounding", True)'),
    "grounding-ignored": (SVC, 'grounding_dict = state.get("pwml_grounding_dict")', 'pass'),
    "name-default": (SVC, 'or "Generated Pathway"', 'or "Pathway"'),
    "description-key": (SVC, 'state.get("pwml_description")', 'state.get("pwml_desc")'),
    "subject-default": (SVC, 'or "Metabolic"', 'or "Signaling"'),
    "width-default": (SVC, 'or 3200', 'or 1600'),
    "width-not-int": (SVC, 'int(state.get("pwml_width") or 3200)', '(state.get("pwml_width") or 3200)'),
    "height-default": (SVC, 'or 1400', 'or 700'),
    "bg-default": (SVC, 'or "#FFFFFF"', 'or "#000000"'),
    "strict-default": (SVC, 'state.get("pwml_strict_db", True)', 'state.get("pwml_strict_db", False)'),
    "context-not-safe": (SVC, 'pathway_context=_postpipeline._safe_dict(state.get("pathway_context"))', 'pathway_context=state.get("pathway_context")'),
    "mode-not-coerced": (SVC, '_export_mode.coerce_mode(state.get("export_mode"))', 'state.get("export_mode")'),
    "reference-order": (SVC, 'project_root / "reference" / "PW000001.pwml",\n        project_root / "reference" / "PW012926.pwml",',
                        'project_root / "reference" / "PW012926.pwml",\n        project_root / "reference" / "PW000001.pwml",'),
    "reference-root-ignored": (SVC, 'ref_path=_pwml_reference_path(project_root)', 'ref_path=_pwml_reference_path(Path("elsewhere"))'),
    "project-root-ignored": (SVC, '            project_root=project_root,\n', '            project_root=Path("elsewhere"),\n'),
    "quarantine-keys-swapped": (SVC, '"quarantine_report": _postpipeline._safe_dict(state.get("quarantine_report"))',
                                '"quarantine_report": _postpipeline._safe_dict(state.get("quarantine_artifacts"))'),
    "signature-default-reporter": (SVC, '    reporter: Reporter,\n    work_dir', '    reporter: Reporter = None,\n    work_dir'),
    "signature-positional": (SVC, 'def run_export(\n    *,\n', 'def run_export(\n'),
    # the app's call site
    "app-state-copied": (APP, CALL, CALL.replace("state=st.session_state", "state=dict(st.session_state)")),
    "app-reporter-none": (APP, 'reporter=StreamlitReporter(st),\n                work_dir=work_dir,', 'reporter=None,\n                work_dir=work_dir,'),
    "app-work-dir": (APP, '                work_dir=work_dir,\n                project_root=PROJECT_ROOT,', '                work_dir=PROJECT_ROOT,\n                project_root=PROJECT_ROOT,'),
    "app-project-root": (APP, '                work_dir=work_dir,\n                project_root=PROJECT_ROOT,', '                work_dir=work_dir,\n                project_root=Path("x"),'),
    "app-result-key": (APP, 'st.session_state["pwml_export_result"] = _pwml_result', 'st.session_state["pwml_result"] = _pwml_result'),
    "app-success-text": (APP, 'st.success(f"PWML generated from reviewed JSON: ', 'st.success(f"PWML generated: '),
    "app-error-text": (APP, 'st.error(f"PWML export failed: ', 'st.error(f"PWML failed: '),
    "app-error-default": (APP, "_pwml_result.get('error', 'unknown')", "_pwml_result.get('error', 'none')"),
    "app-except-narrowed": (APP, '        except Exception as exc:  # noqa: BLE001\n            _pwml_result = {', '        except ValueError as exc:  # noqa: BLE001\n            _pwml_result = {'),
    "app-reference-reexport-dropped": (APP, "from t2pw.service.export import _pwml_reference_path, run_export", "from t2pw.service.export import run_export"),
    "app-helper-restored": (APP, "def reset_quarantine_state(", "def quarantine_session_state():\n    return {}\n\n\ndef reset_quarantine_state("),
}


def apply(text, old, new):
    olds, news = ([old], [new]) if isinstance(old, str) else (old, new)
    for o, n in zip(olds, news):
        assert o in text, f"mutation target not found: {o[:70]!r}"
        text = text.replace(o, n)
    return text


def pytest_once(label):
    """Exit code of one pinned pytest run on the scratch copy. This driver runs inside ONE bounded_run with the
    heavy lock (runpy.sh), held for the whole batch: with several lanes sharing the lock, one acquire per mutation
    could wait out another lane's run each time."""
    env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
    cmd = [PY, "-u", f"{WRAPPER_TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", str(TREE),
           "-q", f"--basetemp=C:/t/wsR4atmp/mut-bt/{label}", "tests/test_service_export.py"]
    with open(EV / f"{label}.log", "w", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=str(TREE), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=90).returncode


def main():
    span = sys.argv[3] if len(sys.argv) > 3 else f"0:{len(MUTATIONS)}"
    lo, hi = map(int, span.split(":"))
    names = list(MUTATIONS)[lo:hi]
    results = {}
    if lo == 0:
        results["control"] = pytest_once("control")
        print("control", results["control"], flush=True)
        if results["control"] != 0:
            return 2
    for name in names:
        path, old, new = MUTATIONS[name]
        original = path.read_bytes()
        try:
            text = original.decode("utf-8").replace("\r\n", "\n")
            path.write_bytes(apply(text, old, new).encode("utf-8"))
            rc = pytest_once(name)
        finally:
            path.write_bytes(original)
        verdict = "killed" if rc == 1 else "SURVIVED" if rc == 0 else f"INVALID({rc})"
        results[name] = verdict
        print(f"{name}: {verdict}", flush=True)
    out = EV / f"results-{lo}-{hi}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    bad = [k for k, v in results.items() if k != "control" and v != "killed"]
    print(f"{sum(1 for k in results if k != 'control') - len(bad)}/{len(names)} killed; not killed: {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
