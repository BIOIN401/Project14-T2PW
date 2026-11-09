"""R4b vacuity table for the one authorised test edit (Q-R4-1), on a SCRATCH COPY of the tip: the
repointed test must pass on the untouched copy and fail under each deliberate break.

usage: vacuity.py SCRATCH_TREE OUTDIR
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TREE = sys.argv[1].replace("\\", "/")
EV = Path(sys.argv[2])
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
SVC = Path(TREE, "src/t2pw/service/post_run.py")
APP = Path(TREE, "src/t2pw/app/streamlit_app.py")
NODE = "tests/test_streamlit_stage2_orchestration.py::test_stage2_artifacts_are_visible_and_both_ui_paths_share_the_orchestrator"
EV.mkdir(parents=True, exist_ok=True)
Path("C:/t/wsR4btmp/vac-tmp").mkdir(parents=True, exist_ok=True)
CASES = {
    "untouched-tip": (None, None, None),
    "handler-call-missing": (SVC, "artifacts = _postpipeline.run_post_pipeline_sbml_artifacts(", "artifacts = _postpipeline.run_something_else("),
    "wrong-build-legacy-sbml": (SVC, "build_legacy_sbml=False", "build_legacy_sbml=True"),
    "StageContractError-handling-removed": (SVC, "    except StageContractError as exc:\n        failure = exc\n        render_contract_failure(exc)\n", ""),
    "generic-handler-removed": (SVC, "    except Exception as exc:\n        failure = exc\n        _postpipeline._record_mapped_failure_snapshot(\n            final_payload,\n            stage=\"post_pipeline\",\n            reason=f\"{type(exc).__name__}: {exc}\",\n        )\n        reporter.error(f\"Post-pipeline conversion failed: {exc}\")\n        reporter.exception(exc)  # full traceback (file:line) so the cause isn't a mystery\n", ""),
    "pop-removed": (APP, '            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second audit/mapping run', "            # A second audit/mapping run"),
    "failure-message-removed": (SVC, '"Mapped pathway still fails the pre-export Stage 3 revalidation. "', '"Mapped pathway fails revalidation. "'),
    "app-render-callback-not-wired": (APP, "                render_contract_failure=lambda failure: _render_stage_contract_failure(\n                    failure,\n                    operation_label=\"Audit and DB mapping\",\n                    download_key=\"dl_audit_mapping_stage_contract_error\",\n                ),\n",
                                      "                render_contract_failure=lambda failure: None,\n"),
    "app-button-does-not-call-the-handler": (APP, "            _post_run = run_post_pipeline(\n", "            _post_run = run_post_pipeline_other(\n"),
    "service-render-call-replaced": (SVC, "        failure = exc\n        render_contract_failure(exc)\n", "        failure = exc\n"),
    "legacy-handler-pop-removed": (APP, '            st.session_state.pop("post_pipeline_artifacts", None)\n            post_artifacts = {}\n            try:\n                with st.spinner("Running legacy SBML export..."):',
                                   '            post_artifacts = {}\n            try:\n                with st.spinner("Running legacy SBML export..."):'),
}
originals = {p: p.read_bytes() for p in (SVC, APP)}
good = 0
rows = []
env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
try:
    for name, (target, old, new) in CASES.items():
        if target is not None:
            raw = originals[target].decode("utf-8")
            crlf = "\r\n" in raw
            src = raw.replace("\r\n", "\n")
            assert src.count(old) == 1, (name, src.count(old))
            src = src.replace(old, new)
            target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"vac-{name}",
               "--timeout", "120", "--heavy-lock", "R4b", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR4btmp/vac-tmp/{name}", NODE]
        try:
            while True:
                proc = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, timeout=300)
                if proc.returncode != 95:
                    break
                print(f"  lock busy, waiting 8 s ({name})", flush=True)
                time.sleep(8)
        finally:
            for p, b in originals.items():
                p.write_bytes(b)
        out = proc.stdout
        (EV / f"{name}.log").write_text(out, encoding="utf-8")
        summary = re.findall(r"^(\d+ (?:failed|passed).*?)(?: in [0-9.]+s)?\s*$", out, re.M)
        result = summary[-1] if summary else "?"
        first_assert = re.findall(r"^E\s+(.*)$", out, re.M)
        expected_pass = target is None
        ok = ("passed" in result and "failed" not in result) if expected_pass else "failed" in result
        good += ok
        line = f"{name:40s} expected={'pass' if expected_pass else 'fail'} got={result}  {'OK' if ok else 'WRONG'}  {first_assert[0] if first_assert else ''}"
        print(line[:260], flush=True)
        rows.append(line)
finally:
    for p, b in originals.items():
        p.write_bytes(b)
(EV / "summary.txt").write_text("\n".join(rows) + f"\n{good}/{len(CASES)} as expected\n", encoding="utf-8")
print(f"{good}/{len(CASES)} as expected")
sys.exit(0 if good == len(CASES) else 1)
