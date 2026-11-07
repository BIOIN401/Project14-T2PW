"""Non-vacuity check for inline_vs_moved.py (R3d): with one mutation applied to the tip, the
proof script must report a difference (exit 1). Each run goes through bounded_run with the
heavy lock; the file is restored after every mutation and in a finally.

usage: ivm_sanity.py OUTDIR   (run against C:/t/ws-R3d, committed clean tree)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = "C:/t/ws-R3d"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
SVC = Path(TREE, "src/t2pw/service/extraction.py")
APP = Path(TREE, "src/t2pw/app/streamlit_app.py")
EV = Path(sys.argv[1])
EV.mkdir(parents=True, exist_ok=True)
MUTATIONS = {
    "spinner-text": (SVC, 'reporter.spinner("Running Stage 2 inference...")', 'reporter.spinner("Running Stage 2")'),
    "publish-order-swapped": (SVC, '    state["qa_report"] = qa_report\n    state["reaction_summary"] = reaction_summary\n',
                              '    state["reaction_summary"] = reaction_summary\n    state["qa_report"] = qa_report\n'),
    "strict-db-default-false": (SVC, 'strict_db=bool(state.get("pwml_strict_db", True))',
                                'strict_db=bool(state.get("pwml_strict_db", False))'),
    "merge-source-text-dropped": (SVC, "            source_text=text,\n        )\n", "        )\n"),
    "attempts-not-rendered": (SVC, "            render_attempts(label, failure.attempts, reporter=reporter)\n", ""),
    "loop-messages-dropped": (SVC, '        emit_rag_loop_messages(rag_loop_record["messages"], reporter=reporter)\n', ""),
    "rag-merge-no-context": (SVC, "            rag_envelope,\n            pathway_context=pathway_context_from_stage_zero(pathway_context),\n",
                             "            rag_envelope,\n"),
    "app-no-stop-on-failure": (APP, "    if _stage_two.failure is not None:\n        st.stop()\n",
                               "    if _stage_two.failure is not None:\n        pass\n"),
    "app-reset-dropped": (APP, '    reset_quarantine_state(PROJECT_ROOT / "outputs")\n    st.session_state["token_stats"]',
                          '    st.session_state["token_stats"]'),
}
originals = {p: p.read_bytes() for p in (SVC, APP)}
ok = 0
try:
    for name, (target, old, new) in MUTATIONS.items():
        raw = originals[target].decode("utf-8")
        crlf = "\r\n" in raw
        src = raw.replace("\r\n", "\n")
        assert src.count(old) == 1, name
        src = src.replace(old, new)
        target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
        env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"ivm-{name}",
               "--timeout", "300", "--heavy-lock", "R3d", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", "C:/t/wsR3dtmp/scripts/inline_vs_moved.py", "C:/t/ws-R3d-base", TREE]
        try:
            out = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=400).stdout
        finally:
            target.write_bytes(originals[target])
        (EV / f"{name}.log").write_text(out, encoding="utf-8")
        tail = re.findall(r"^\d+/\d+ identical.*$", out, re.M)
        detected = bool(tail) and not tail[-1].endswith("0 unexplained")
        ok += detected
        print(f"{name:28s} {'DETECTED' if detected else 'MISSED'}  {tail[-1] if tail else '?'}", flush=True)
finally:
    for p, b in originals.items():
        p.write_bytes(b)
print(f"detected {ok}/{len(MUTATIONS)}")
sys.exit(0 if ok == len(MUTATIONS) else 1)
