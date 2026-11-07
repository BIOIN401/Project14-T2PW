"""Non-vacuity check for inline_vs_moved.py: with one mutation applied to the tip, the
proof script must report a difference (exit 1). Each run goes through bounded_run with
the heavy lock; the file is restored after every mutation.

usage: ivm_sanity.py   (run from C:/t/ws-R3c, committed clean tree)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = "C:/t/ws-R3c"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
SVC = Path(TREE, "src/t2pw/service/extraction.py")
APP = Path(TREE, "src/t2pw/app/streamlit_app.py")
EV = Path(TREE, "docs/webserver/evidence/R3c/ivm-sanity")
EV.mkdir(parents=True, exist_ok=True)
MUTATIONS = {
    "fold-order-swapped": (SVC, "for part in (user_task_context, rag_result.evidence_context)",
                           "for part in (rag_result.evidence_context, user_task_context)"),
    "spinner-text": (SVC, 'reporter.spinner("Running Stage 1 extraction...")', 'reporter.spinner("Running Stage 1")'),
    "contract-json-dropped": (SVC, "        reporter.json(failure.report)\n", ""),
    "seed-from-unfiltered": (SVC, "            seed_payload=stage_one_in_scope,\n", "            seed_payload=stage_one,\n"),
    "app-focus-not-rebound": (APP, "    user_task_context = _stage_one.user_task_context\n", ""),
    "app-no-stop-on-failure": (APP, "    if _stage_one.failure is not None:\n        st.stop()\n",
                               "    if _stage_one.failure is not None:\n        pass\n"),
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
               "--timeout", "300", "--heavy-lock", "ws-R3c", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", f"{TREE}/docs/webserver/evidence/R3c/scripts/inline_vs_moved.py",
               "C:/t/ws-R3c-base", TREE]
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
