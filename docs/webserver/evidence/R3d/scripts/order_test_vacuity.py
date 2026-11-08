"""R3d / Q-R3d-1: non-vacuity of the committed order-test rewrite. Each temporary single edit of the
committed tip must make tests/test_streamlit_stage2_orchestration.py::
test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference fail (the control must pass).
Every pytest run goes through bounded_run + pinned_pytest with the heavy lock; every file is restored after
its run and in a finally; the caller checks ``git status``.

usage: order_test_vacuity.py OUTDIR
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
Path("C:/t/wsR3dtmp/vac-tmp").mkdir(parents=True, exist_ok=True)  # the --basetemp parent
TEST = ("tests/test_streamlit_stage2_orchestration.py::"
        "test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference")
FILTER = "    stage_one_in_scope, out_of_scope_removed_reactions = _pipeline.filter_out_of_scope_reactions(stage_one)\n"
STAGE1 = "    # Stage 1: strict extraction with auto-repair\n    try:\n"
EDITS = {  # name: (file, [(old, new), ...], expect_fail)
    "control-untouched-tip": (SVC, [], False),
    "filter-removed": (SVC, [(FILTER, "    stage_one_in_scope, out_of_scope_removed_reactions = stage_one, []\n")], True),
    "filter-before-extraction": (SVC, [(FILTER, ""), (STAGE1, FILTER.replace("(stage_one)", "({})") + STAGE1)], True),
    "returns-raw-extraction": (SVC, [("        stage_one_in_scope=stage_one_in_scope,\n        out_of_scope_removed_reactions=",
                                      "        stage_one_in_scope=stage_one,\n        out_of_scope_removed_reactions=")], True),
    "stage2-fed-raw-stage_one": (SVC, [("                    text,\n                    stage_one_in_scope,\n",
                                        "                    text,\n                    stage_one,\n")], True),
    "first-merge-fed-raw-stage_one": (SVC, [("        final_payload = merge_additions(\n            stage_one_in_scope,\n",
                                             "        final_payload = merge_additions(\n            stage_one,\n")], True),
    "in-scope-bound-from-raw (F1)": (SVC, [("    stage_one_in_scope = stage_one_result.stage_one_in_scope\n",
                                            "    stage_one_in_scope = stage_one_result.stage_one\n")], True),
    "app-hands-run_stage_two-another-result": (APP, [("    _stage_two = run_stage_two(\n        text,\n        _stage_one,\n",
                                                      "    _stage_two = run_stage_two(\n        text,\n        _stage_zero,\n")], True),
}
originals = {p: p.read_bytes() for p in (SVC, APP)}
ok = 0
try:
    for name, (target, edits, expect_fail) in EDITS.items():
        raw = originals[target].decode("utf-8")
        crlf = "\r\n" in raw
        src = raw.replace("\r\n", "\n")
        for old, new in edits:
            assert src.count(old) == 1, (name, old[:60], src.count(old))
            src = src.replace(old, new)
        target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
        env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"vac-{slug}",
               "--timeout", "300", "--heavy-lock", "R3d", "--json", str(EV / f"{slug}.json"), "--",
               PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR3dtmp/vac-tmp/{slug}", TEST]
        try:
            out = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=400).stdout
        finally:
            target.write_bytes(originals[target])
        (EV / f"{slug}.log").write_text(out, encoding="utf-8")
        summary = re.findall(r"^(?:\d+ (?:failed|passed)).*$", out, re.M)
        result = summary[-1] if summary else "?"
        failed = "failed" in result
        good = failed == expect_fail and summary
        ok += bool(good)
        surv = re.findall(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", out)
        assertion = re.findall(r"^E\s+(assert .*)$", out, re.M)
        line = (f"{name:40s} expect={'fail' if expect_fail else 'pass'} {result[:30]:30s} "
                f"survivors={surv[-1] if surv else '?'}  {assertion[0][:100] if assertion else ''}")
        print(line, flush=True)
        with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
finally:
    for p, b in originals.items():
        p.write_bytes(b)
tail = f"as expected {ok}/{len(EDITS)}"
print(tail)
with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
    fh.write(tail + "\n")
sys.exit(0 if ok == len(EDITS) else 1)
