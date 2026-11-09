"""Non-vacuity check for inline_vs_moved.py (R4b): with one mutation applied to a SCRATCH COPY of
the tip, the proof script must report a difference (exit 1). Each run goes through bounded_run
with the heavy lock; the file is restored after every mutation and in a finally.

usage: ivm_sanity.py SCRATCH_TREE OUTDIR
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
EV.mkdir(parents=True, exist_ok=True)
SVCSC = '        _pa = _safe_dict(artifacts)\n'
MUTATIONS = {
    "spinner-text": (SVC, 'reporter.spinner("Running audit and DB mapping...")', 'reporter.spinner("Running audit")'),
    "reenter-into-try": (SVC, '    _diag.reenter(state.get("extraction_diagnostics_recorder"))\n    try:\n',
                         '    try:\n        _diag.reenter(state.get("extraction_diagnostics_recorder"))\n'),
    "strict-db-default-false": (SVC, 'bool(state.get("pwml_strict_db", True))', 'bool(state.get("pwml_strict_db", False))'),
    "snapshot-pre-rebind": (SVC, "        _postpipeline._record_mapped_failure_snapshot(\n            final_payload,",
                            "        _postpipeline._record_mapped_failure_snapshot(\n            request.final_payload,"),
    "pa-a-copy": (SVC, SVCSC, "        _pa = dict(_safe_dict(artifacts))\n"),
    "em-dash-in-markdown": (SVC, "did NOT pass — this candidate", "did NOT pass - this candidate"),
    "pass-info-before-review": (SVC, '                        open_review(final_mapped_payload, mapping_report)\n                        reporter.info("Mapped pathway is ready for review. PWML generation is paused until approval.")\n',
                                '                        reporter.info("Mapped pathway is ready for review. PWML generation is paused until approval.")\n                        open_review(final_mapped_payload, mapping_report)\n'),
    "research-errors-set-after-review": (SVC, '                        state["refinement_gate_errors"] = []\n                        reporter.warning(\n                            "Research mode:',
                                         '                        reporter.warning(\n                            "Research mode:'),
    "contract-render-after-sync": (SVC, "        failure = exc\n        render_contract_failure(exc)\n    except Exception as exc:",
                                   "        failure = exc\n    except Exception as exc:"),
    "sync-before-messages": (SVC, "        reporter.error(f\"Post-pipeline conversion failed: {exc}\")\n",
                             "        _extraction._sync_extraction_diagnostics(state=state)\n        reporter.error(f\"Post-pipeline conversion failed: {exc}\")\n"),
    "app-no-pop": (APP, '            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second audit/mapping run', "            # A second audit/mapping run"),
    "app-no-reset": (APP, '            reset_quarantine_state(PROJECT_ROOT / "outputs")\n            post_artifacts = {}\n            # R4b:', "            post_artifacts = {}\n            # R4b:"),
    "app-pop-after-reset": (APP, '            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second audit/mapping run',
                            "            # A second audit/mapping run"),
    "app-no-rebind": (APP, "            final_payload = _post_run.final_payload\n", ""),
    "app-state-not-session": (APP, "                state=st.session_state,\n                reporter=StreamlitReporter(st),\n                open_review",
                              "                state={},\n                reporter=StreamlitReporter(st),\n                open_review"),
    "app-swapped-widget": (APP, "                    use_stoich_agent=use_stoich_agent,\n", "                    use_stoich_agent=use_gap_resolver,\n"),
    "app-guard-removed": (APP, '        if not isinstance(final_payload, dict) or not final_payload:\n            st.error("No pipeline output in session state. Run the pipeline first.")\n        else:\n            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second',
                          '        if True:\n            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second'),
}
# app-pop-after-reset: the pop moves to after the reset.
MUTATIONS["app-pop-after-reset"] = (APP, '            st.session_state.pop("post_pipeline_artifacts", None)\n            # A second audit/mapping run produces a new payload version',
                                    "            # A second audit/mapping run produces a new payload version")
EXTRA = {"app-pop-after-reset": ('            reset_quarantine_state(PROJECT_ROOT / "outputs")\n            post_artifacts = {}\n            # R4b:',
                                 '            reset_quarantine_state(PROJECT_ROOT / "outputs")\n            st.session_state.pop("post_pipeline_artifacts", None)\n            post_artifacts = {}\n            # R4b:')}
originals = {p: p.read_bytes() for p in (SVC, APP)}
ok = 0
try:
    for name, (target, old, new) in MUTATIONS.items():
        raw = originals[target].decode("utf-8")
        crlf = "\r\n" in raw
        src = raw.replace("\r\n", "\n")
        assert src.count(old) == 1, (name, src.count(old))
        src = src.replace(old, new)
        if name in EXTRA:
            o2, n2 = EXTRA[name]
            assert src.count(o2) == 1, name
            src = src.replace(o2, n2)
        target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
        env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"ivm-{name}",
               "--timeout", "300", "--heavy-lock", "R4b", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", "C:/t/wsR4btmp/scripts/inline_vs_moved.py", "C:/t/ws-R4-base", TREE]
        try:
            while True:
                proc = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, timeout=400)
                if proc.returncode != 95:
                    break
                print(f"  lock busy, waiting 8 s ({name})", flush=True)
                time.sleep(8)
        finally:
            target.write_bytes(originals[target])
        (EV / f"{name}.log").write_text(proc.stdout, encoding="utf-8")
        tail = re.findall(r"^\d+/\d+ identical.*$", proc.stdout, re.M)
        detected = bool(tail) and int(re.search(r"; (\d+) unexplained", tail[-1]).group(1)) > 0
        ok += detected
        print(f"{name:32s} {'DETECTED' if detected else 'MISSED'}  {tail[-1] if tail else '?'}", flush=True)
finally:
    for p, b in originals.items():
        p.write_bytes(b)
print(f"detected {ok}/{len(MUTATIONS)}")
sys.exit(0 if ok == len(MUTATIONS) else 1)
