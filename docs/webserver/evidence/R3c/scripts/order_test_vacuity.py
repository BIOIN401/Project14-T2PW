"""Q-R3c-1 non-vacuity: each temp edit below must make the rewritten order test FAIL, and
the test must PASS on the untouched tip. Edits are applied to the committed tree, run
through bounded_run with the heavy lock, and reverted (also in a finally); they are never
committed. The caller checks ``git status`` afterwards.

usage: order_test_vacuity.py   (run from C:/t/ws-R3c, committed clean tree)
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
EV = Path(TREE, "docs/webserver/evidence/R3c/order-test-vacuity")
EV.mkdir(parents=True, exist_ok=True)
Path("C:/t/wsR3ctmp/vac-tmp").mkdir(parents=True, exist_ok=True)  # the --basetemp parent
TEST = ("tests/test_streamlit_stage2_orchestration.py::"
        "test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference")
FILTER = "    stage_one_in_scope, out_of_scope_removed_reactions = _pipeline.filter_out_of_scope_reactions(stage_one)\n"
STAGE1 = "    # Stage 1: strict extraction with auto-repair\n    try:\n"

EDITS = {  # name: (file, [(old, new), ...], expect_fail)
    "control-untouched-tip": (SVC, [], False),
    "filter-removed": (SVC, [(FILTER, "    stage_one_in_scope, out_of_scope_removed_reactions = stage_one, []\n")], True),
    "filter-before-extraction": (SVC, [(FILTER, ""), (STAGE1, FILTER.replace("(stage_one)", "({})") + STAGE1)], True),
    "returns-raw-extraction": (SVC, [("        stage_one_in_scope=stage_one_in_scope,\n",
                                      "        stage_one_in_scope=stage_one,\n")], True),
    "app-stage2-gets-raw-stage_one": (APP, [("                    text,\n                    stage_one_in_scope,\n"
                                             "                    chunk_details=chunk_details,\n",
                                             "                    text,\n                    stage_one,\n"
                                             "                    chunk_details=chunk_details,\n")], True),
}
originals = {p: p.read_bytes() for p in (SVC, APP)}
good = 0
try:
    for name, (target, reps, expect_fail) in EDITS.items():
        raw = originals[target].decode("utf-8")
        crlf = "\r\n" in raw
        src = raw.replace("\r\n", "\n")
        for old, new in reps:
            assert src.count(old) == 1, (name, old[:60], src.count(old))
            src = src.replace(old, new)
        target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
        env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"vac-{name}",
               "--timeout", "300", "--heavy-lock", "ws-R3c", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR3ctmp/vac-tmp/{name}", TEST]
        try:
            out = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=400).stdout
        finally:
            target.write_bytes(originals[target])
        (EV / f"{name}.log").write_text(out, encoding="utf-8")
        summary = (re.findall(r"^\d+ (?:failed|passed).*$", out, re.M) or ["?"])[-1]
        surv = (re.findall(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", out) or ["?"])[-1]
        failed = "failed" in summary
        ok = failed == expect_fail and surv == "0"
        good += ok
        line = (f"{name:32s} expect={'FAIL' if expect_fail else 'PASS'}  got={summary[:40]:<40s} "
                f"survivors={surv}  {'OK' if ok else 'WRONG'}")
        print(line, flush=True)
        with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
finally:
    for p, b in originals.items():
        p.write_bytes(b)
tail = f"{good}/{len(EDITS)} as expected"
print(tail)
with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
    fh.write(tail + "\n")
sys.exit(0 if good == len(EDITS) else 1)
