"""Run the two proposed test edits (out-of-suite proof copies) on the committed tip, untouched and
under temporary single mutations of the tip. Each pytest run goes through bounded_run + pinned_pytest
with the heavy lock; every mutated file is restored after its run and in a finally.

usage: proof_runs.py OUTDIR
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
Path("C:/t/wsR3dtmp/proof-tmp").mkdir(parents=True, exist_ok=True)
C060A = f"{TREE}/docs/webserver/evidence/R3d/proposed/c060a_repoint_check.py"
ORDER = f"{TREE}/docs/webserver/evidence/R3d/proposed/s2orch_order_proposed.py"
RUNS = {  # name: (file, old, new, test, expect)
    "c060a-control": (None, None, None, C060A, "pass"),
    "c060a-stage2-site-dict-context": (SVC, "    pathway_context=pathway_context_from_stage_zero(pathway_context),\n            source_text=text,",
                                       "    pathway_context=pathway_context,\n            source_text=text,", C060A, "fail"),
    "c060a-rag-site-dict-context": (SVC, "            rag_envelope,\n            pathway_context=pathway_context_from_stage_zero(pathway_context),\n",
                                    "            rag_envelope,\n            pathway_context=pathway_context,\n", C060A, "fail"),
    "c060a-rag-site-seed-text": (SVC, "            rag_envelope,\n            pathway_context=pathway_context_from_stage_zero(pathway_context),\n",
                                 "            rag_envelope,\n            pathway_context=pathway_context_from_stage_zero(pathway_context),\n            seed_text=text,\n",
                                 C060A, "fail"),
    "order-control": (None, None, None, ORDER, "pass"),
    "order-filter-removed": (SVC, "= _pipeline.filter_out_of_scope_reactions(stage_one)", "= stage_one, []", ORDER, "fail"),
    "order-returns-raw-extraction": (SVC, "        stage_one_in_scope=stage_one_in_scope,\n        out_of_scope_removed_reactions=",
                                     "        stage_one_in_scope=stage_one,\n        out_of_scope_removed_reactions=", ORDER, "fail"),
    "order-stage2-gets-raw-stage_one": (SVC, "                    text,\n                    stage_one_in_scope,\n",
                                        "                    text,\n                    stage_one,\n", ORDER, "fail"),
    "order-in-scope-bound-from-raw": (SVC, "    stage_one_in_scope = stage_one_result.stage_one_in_scope\n",
                                      "    stage_one_in_scope = stage_one_result.stage_one\n", ORDER, "fail"),
    "order-merge-base-raw": (SVC, "        final_payload = merge_additions(\n            stage_one_in_scope,\n",
                             "        final_payload = merge_additions(\n            stage_one,\n", ORDER, "fail"),
    "order-app-hands-other-result": (APP, "    _stage_two = run_stage_two(\n        text,\n        _stage_one,\n",
                                     "    _stage_two = run_stage_two(\n        text,\n        _stage_zero,\n", ORDER, "fail"),
}
originals = {p: p.read_bytes() for p in (SVC, APP)}
ok = 0
try:
    for name, (target, old, new, test, expect) in RUNS.items():
        if target is not None:
            raw = originals[target].decode("utf-8")
            crlf = "\r\n" in raw
            src = raw.replace("\r\n", "\n")
            assert src.count(old) == 1, (name, src.count(old))
            src = src.replace(old, new)
            target.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
        env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1", T2PW_PROOF_TREE=TREE)
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"proof-{name}",
               "--timeout", "300", "--heavy-lock", "R3d", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR3dtmp/proof-tmp/{name}", test]
        try:
            out = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=400).stdout
        finally:
            if target is not None:
                target.write_bytes(originals[target])
        (EV / f"{name}.log").write_text(out, encoding="utf-8")
        summary = re.findall(r"^(?:\d+ (?:failed|passed)|ERROR|=+ .*error).*$", out, re.M)
        result = summary[-1] if summary else "?"
        got = "fail" if ("failed" in result or "error" in result.lower()) else ("pass" if "passed" in result else "?")
        ok += got == expect
        surv = re.findall(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", out)
        assertion = re.findall(r"^E\s+(assert .*)$", out, re.M)
        line = (f"{name:36s} expect={expect} got={got:4s} {result[:50]:50s} survivors={surv[-1] if surv else '?'}"
                f"  {assertion[0][:90] if assertion else ''}")
        print(line, flush=True)
        with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
finally:
    for p, b in originals.items():
        p.write_bytes(b)
tail = f"as expected {ok}/{len(RUNS)}"
print(tail)
with open(EV / "summary.txt", "a", encoding="utf-8") as fh:
    fh.write(tail + "\n")
sys.exit(0 if ok == len(RUNS) else 1)
