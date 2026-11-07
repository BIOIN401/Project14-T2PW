"""Reviewer probes for R3c. Applies one mutation at a time to the SCRATCH tree
C:/t/rev-R3c/tree (a plain copy of ws-R3c @ cdf47c54 src+tests), runs the named tests
through bounded_run with the heavy lock, restores the file. Never touches a git tree."""
import os, re, shutil, subprocess, sys, time
from pathlib import Path

TREE = Path("C:/t/rev-R3c/tree")
PRISTINE = Path("C:/t/ws-R3c")
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
BR = str(PRISTINE / "docs/pwml_recovery_sprint/evidence/bounded_run.py")
OUT = Path("C:/t/rev-R3c/out"); OUT.mkdir(parents=True, exist_ok=True)
BT = Path("C:/t/rev-R3c/bt"); BT.mkdir(parents=True, exist_ok=True)
SVC = "src/t2pw/service/extraction.py"
APP = "src/t2pw/app/streamlit_app.py"
ORDER = ("tests/test_streamlit_stage2_orchestration.py::"
         "test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference")
NEW = "tests/test_service_extraction_stage1.py"
FILTER = "    stage_one_in_scope, out_of_scope_removed_reactions = _pipeline.filter_out_of_scope_reactions(stage_one)\n"
STAGE1 = "    # Stage 1: strict extraction with auto-repair\n    try:\n"
S2 = "                    text,\n                    stage_one_in_scope,\n                    chunk_details=chunk_details,\n"
M1 = "        final_payload = merge_additions(\n            stage_one_in_scope,\n"

MUTS = {
    "control": (SVC, [], [ORDER, NEW]),
    "O1-filter-removed": (SVC, [(FILTER, "    stage_one_in_scope, out_of_scope_removed_reactions = stage_one, []\n")], [ORDER]),
    "O2-filter-before-extraction": (SVC, [(FILTER, ""), (STAGE1, FILTER.replace("(stage_one)", "({})") + STAGE1)], [ORDER]),
    "O3-returns-raw": (SVC, [("        stage_one_in_scope=stage_one_in_scope,\n", "        stage_one_in_scope=stage_one,\n")], [ORDER]),
    "O4-app-stage2-raw": (APP, [(S2, S2.replace("stage_one_in_scope,", "stage_one,"))], [ORDER]),
    "O5-app-merge1-raw": (APP, [(M1, M1.replace("stage_one_in_scope,", "stage_one,"))], [ORDER]),
    "E1-app-rebind-in-scope-to-raw": (APP, [("    stage_one_in_scope = _stage_one.stage_one_in_scope\n", "    stage_one_in_scope = _stage_one.stage_one\n")], [ORDER, NEW]),
    "E2-svc-reassign-after-filter": (SVC, [(FILTER, FILTER + "    stage_one_in_scope = stage_one\n")], [ORDER, NEW]),
    "E3-svc-rag-seed-raw": (SVC, [("            seed_payload=stage_one_in_scope,\n", "            seed_payload=stage_one,\n")], [ORDER, NEW]),
    "E4-svc-filter-arg-presettle": (SVC, [("            stage_one = _boundary.payload\n", "            _raw = stage_one\n            stage_one = _boundary.payload\n"),
                                          (FILTER, FILTER.replace("(stage_one)", "(_raw)"))], [ORDER, NEW]),
    "B1-bound-settle": (SVC, [("            _boundary = _stage_one_boundary.settle_stage_one(\n", "            _boundary = _BOUND_SETTLE(\n"),
                              ("from t2pw.service.reporter import Reporter\n", "from t2pw.service.reporter import Reporter\nfrom t2pw.pipeline.stage_one_boundary import settle_stage_one as _BOUND_SETTLE\n")], [NEW]),
}

only = sys.argv[1:] or list(MUTS)
for name in only:
    rel, reps, tests = MUTS[name]
    target = TREE / rel
    orig = (PRISTINE / rel).read_bytes()
    raw = orig.decode("utf-8"); crlf = "\r\n" in raw; src = raw.replace("\r\n", "\n")
    for old, new in reps:
        assert src.count(old) == 1, (name, old[:70], src.count(old))
        src = src.replace(old, new)
    target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
    env = dict(os.environ, PYTHONPATH=str(TREE / "src"), T2PW_OFFLINE_CURATOR="1")
    cmd = [PY, BR, "--label", f"rev-R3c-{name}", "--timeout", "420", "--heavy-lock", "rev-R3c",
           "--json", str(OUT / f"{name}.json"), "--cwd", str(TREE), "--",
           PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={BT / name}", *tests]
    try:
        for attempt in range(40):
            p = subprocess.run(cmd, cwd=str(TREE), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=540)
            if p.returncode != 95:
                break
            time.sleep(20)
    finally:
        target.write_bytes(orig)
    out = p.stdout
    (OUT / f"{name}.log").write_text(out, encoding="utf-8")
    summ = (re.findall(r"^=*\s*\d+ (?:failed|passed|error).*$", out, re.M) or ["?"])[-1]
    surv = (re.findall(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", out) or ["?"])[-1]
    fails = sorted(set(re.findall(r"^FAILED (\S+)", out, re.M)))
    line = f"{name:34s} rc={p.returncode} {summ.strip('= ')[:60]:<60s} surv={surv} failed={fails}"
    print(line, flush=True)
    with open(OUT / "summary.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
# verify scratch restored
for rel in (SVC, APP):
    assert (TREE / rel).read_bytes() == (PRISTINE / rel).read_bytes(), rel
print("scratch restored")
