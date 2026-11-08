"""Reviewer instrumentation on the scratch copy: what does Stage 2 contribute on the fixture? Writes out/instr.txt."""
import os, re, shutil, subprocess, time
from pathlib import Path
TREE = "C:/t/rev-R3d/tree"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
REL = "src/t2pw/service/extraction.py"
OUT = Path("C:/t/rev-R3d/out")
original = Path("C:/t/ws-R3d", REL).read_text(encoding="utf-8")
NL = chr(10)
A = "        qa_hints = stage_two.get(" + chr(34) + "qa_hints" + chr(34) + ", {}) if isinstance(stage_two, dict) else {}" + NL
B = "    stage2_preservation_report = _postpipeline._write_reaction_preservation_report(" + chr(34) + "after_stage2" + chr(34) + ", final_payload)" + NL
PATH = chr(34) + "C:/t/rev-R3d/out/instr.txt" + chr(34)
SNIP_A = A + "        print(" + chr(34) + "stage_two sizes" + chr(34) + ", {k: (len(v) if hasattr(v, " + chr(34) + "__len__" + chr(34) + ") else v) for k, v in stage_two.items()} if isinstance(stage_two, dict) else type(stage_two), " + chr(34) + "out_of_scope" + chr(34) + ", out_of_scope_removed_reactions, " + chr(34) + "mode" + chr(34) + ", repr(state.get(" + chr(34) + "export_mode" + chr(34) + ")), file=open(" + PATH + ", " + chr(34) + "a" + chr(34) + "))" + NL
SNIP_B = "    print(" + chr(34) + "final == in_scope:" + chr(34) + ", final_payload == stage_one_in_scope, " + chr(34) + "reactions in_scope/final" + chr(34) + ", len(stage_one_in_scope.get(" + chr(34) + "reactions" + chr(34) + ", [])), len(final_payload.get(" + chr(34) + "reactions" + chr(34) + ", [])), file=open(" + PATH + ", " + chr(34) + "a" + chr(34) + "))" + NL + B
assert original.count(A) == 1 and original.count(B) == 1
text = original.replace(A, SNIP_A).replace(B, SNIP_B)
Path(TREE, REL).write_text(text, encoding="utf-8", newline="")
try:
    Path("C:/t/rev-R3d/out/instr.txt").unlink(missing_ok=True)
    env = dict(os.environ, PYTHONPATH=TREE + "/src", T2PW_OFFLINE_CURATOR="1")
    bt = "C:/t/rev-R3d/bt/instr"; shutil.rmtree(bt, ignore_errors=True)
    cmd = [PY, TREE + "/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", "rev-instr", "--timeout", "420", "--heavy-lock", "rev-R3d",
           "--json", str(OUT / "mut-instr.json"), "--", PY, "-u", TREE + "/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
           "--pin-verdict", str(OUT / "pin-instr.json"), "-q", "--basetemp=" + bt, "tests/test_equivalence_app_vs_service.py"]
    for attempt in range(30):
        with open(OUT / "mut-instr.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=TREE, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 95:
            break
        time.sleep(15)
    print("rc", rc)
finally:
    Path(TREE, REL).write_text(original, encoding="utf-8", newline="")
