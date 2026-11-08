"""Reviewer mutations. Scratch copy only. usage: mutate.py NAME [NAME...] ; prints one summary line each."""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path
TREE = "C:/t/rev-R3d/tree"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
EXT_REL = "src/t2pw/service/extraction.py"
APP_REL = "src/t2pw/app/streamlit_app.py"
PRISTINE = "C:/t/ws-R3d"
OUT = Path("C:/t/rev-R3d/out")
ORDER = ["tests/test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference"]
C060A = ["tests/test_c060a_requested_pathway_wiring.py"]
STAGE2 = ["tests/test_service_extraction_stage2.py"]
STAGE1 = ["tests/test_service_extraction_stage1.py"]
EQ = ["tests/test_equivalence_app_vs_service.py"]
FILTER_LINE = """    stage_one_in_scope, out_of_scope_removed_reactions = _pipeline.filter_out_of_scope_reactions(stage_one)
"""
STAGE2_ARG = """                    text,
                    stage_one_in_scope,
"""
MERGE1 = """        final_payload = merge_additions(
            stage_one_in_scope,
"""
SITE1_CTX = """            pathway_context=pathway_context_from_stage_zero(pathway_context),
            source_text=text,
"""
SITE2_CTX = """            rag_envelope,
            pathway_context=pathway_context_from_stage_zero(pathway_context),
"""
MUTS = {
 "baseline-order": (EXT_REL, None, None, ORDER),
 "baseline-c060a": (EXT_REL, None, None, C060A),
 "baseline-stage2": (EXT_REL, None, None, STAGE2),
 "baseline-eq": (EXT_REL, None, None, EQ),
 # the five Q-R3d-1 order-test mutations
 "order-filter-removed": (EXT_REL, FILTER_LINE, """    stage_one_in_scope, out_of_scope_removed_reactions = stage_one, []
""", ORDER),
 "order-filter-reordered": (EXT_REL, "    # Stage 1: strict extraction with auto-repair\n", """    _early_in, _early_rm = _pipeline.filter_out_of_scope_reactions(text)
    # Stage 1: strict extraction with auto-repair
""", ORDER, (FILTER_LINE, """    stage_one_in_scope, out_of_scope_removed_reactions = _early_in, _early_rm
""")),
 "order-raw-returned": (EXT_REL, "        stage_one_in_scope=stage_one_in_scope,\n", "        stage_one_in_scope=stage_one,\n", ORDER),
 "order-stage2-raw": (EXT_REL, STAGE2_ARG, """                    text,
                    stage_one,
""", ORDER),
 "order-merge-raw": (EXT_REL, MERGE1, """        final_payload = merge_additions(
            stage_one,
""", ORDER),
 # hunting survivors of the order test: raw payload when inference is off
 "order-e1-final-payload-raw": (EXT_REL, "    final_payload = stage_one_in_scope\n", "    final_payload = stage_one\n", ORDER),
 "order-e2-rebind-after-filter": (EXT_REL, FILTER_LINE, FILTER_LINE + "    stage_one_in_scope = stage_one\n", ORDER),
 "order-e3-merge-alias": (EXT_REL, MERGE1, """        _seed = stage_one
        final_payload = merge_additions(
            _seed,
""", ORDER),
 "order-e4-extraction-passes-raw": (EXT_REL, """        text,
        stage_one,
        pathway_context=stage_zero.pathway_context,
""", """        text,
        __import__("dataclasses").replace(stage_one, stage_one_in_scope=stage_one.stage_one),
        pathway_context=stage_zero.pathway_context,
""", ORDER),
 "s2-e1-final-payload-raw": (EXT_REL, "    final_payload = stage_one_in_scope\n", "    final_payload = stage_one\n", STAGE2),
 "s1-e2-rebind-after-filter": (EXT_REL, FILTER_LINE, FILTER_LINE + "    stage_one_in_scope = stage_one\n", STAGE1),
 "s2-e3-merge-alias": (EXT_REL, MERGE1, """        _seed = stage_one
        final_payload = merge_additions(
            _seed,
""", STAGE2),
 "s2-e4-extraction-passes-raw": (EXT_REL, """        text,
        stage_one,
        pathway_context=stage_zero.pathway_context,
""", """        text,
        __import__("dataclasses").replace(stage_one, stage_one_in_scope=stage_one.stage_one),
        pathway_context=stage_zero.pathway_context,
""", STAGE2),
 # c060a bite on the moved statements
 "c060a-stage2-dict": (EXT_REL, SITE1_CTX, """            pathway_context=pathway_context,
            source_text=text,
""", C060A),
 "c060a-rag-dict": (EXT_REL, SITE2_CTX, """            rag_envelope,
            pathway_context=pathway_context,
""", C060A),
 "c060a-stage2-no-source-text": (EXT_REL, SITE1_CTX, """            pathway_context=pathway_context_from_stage_zero(pathway_context),
""", C060A),
 "c060a-rag-seed-text": (EXT_REL, SITE2_CTX, """            rag_envelope,
            pathway_context=pathway_context_from_stage_zero(pathway_context),
            seed_text=text,
""", C060A),
 "c060a-stage2-no-context": (EXT_REL, SITE1_CTX, """            source_text=text,
""", C060A),
 # equivalence
 "eq-one-qa-round": (EXT_REL, "        infer_rounds=request.infer_rounds,\n", "        infer_rounds=1,\n", EQ),
 "eq-stage2-raw-payload": (EXT_REL, MERGE1, """        final_payload = merge_additions(
            stage_one,
""", EQ),
 "eq-no-merge": (EXT_REL, """            stage_two if isinstance(stage_two, dict) else {},
            pathway_context=pathway_context_from_stage_zero(pathway_context),
""", """            {},
            pathway_context=pathway_context_from_stage_zero(pathway_context),
""", EQ),
 "eq-extraction-wrong-mode": (EXT_REL, "    state[\"export_mode\"] = request.export_mode\n", "    state[\"export_mode\"] = None\n", EQ),
}

MUTS['s2-no-merge']=(EXT_REL,"""            stage_two if isinstance(stage_two, dict) else {},
            pathway_context=pathway_context_from_stage_zero(pathway_context),
""","""            {},
            pathway_context=pathway_context_from_stage_zero(pathway_context),
""",STAGE2)
MUTS['baseline-c055']=(EXT_REL,None,None,['tests/test_c055_rag_loop_wiring.py'])
MUTS['baseline-s2orch']=(EXT_REL,None,None,['tests/test_streamlit_stage2_orchestration.py'])

def run(name):
    spec = MUTS[name]
    rel, old, new, tests = spec[0], spec[1], spec[2], spec[3]
    extra = spec[4] if len(spec) > 4 else None
    target = Path(TREE, rel)
    original = Path(PRISTINE, rel).read_text(encoding="utf-8")
    text = original
    if old is not None:
        assert text.count(old) == 1, (name, text.count(old))
        text = text.replace(old, new)
        if extra:
            assert text.count(extra[0]) == 1, (name, "extra", text.count(extra[0]))
            text = text.replace(extra[0], extra[1])
    target.write_text(text, encoding="utf-8", newline="")
    try:
        env = dict(os.environ, PYTHONPATH=TREE + "/src", T2PW_OFFLINE_CURATOR="1")
        bt = "C:/t/rev-R3d/bt/" + name
        shutil.rmtree(bt, ignore_errors=True)
        log = OUT / ("mut-" + name + ".log")
        cmd = [PY, TREE + "/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", "rev-" + name, "--timeout", "420",
               "--heavy-lock", "rev-R3d", "--json", str(OUT / ("mut-" + name + ".json")), "--", PY, "-u",
               TREE + "/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "--pin-verdict", str(OUT / ("pin-" + name + ".json")), "-q", "--basetemp=" + bt] + tests
        for attempt in range(30):
            with open(log, "w", encoding="utf-8") as fh:
                rc = subprocess.run(cmd, cwd=TREE, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 95:
                break
            time.sleep(15)
        out = log.read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"^.*\b(?:passed|failed|error|errors)\b.*$", out, re.M)
        surv = re.search(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", out)
        print(name.ljust(34), "rc", rc, (m[-1] if m else "NO-SUMMARY")[:90], "| survivors", surv.group(1) if surv else "?", flush=True)
    finally:
        target.write_text(original, encoding="utf-8", newline="")

for n in sys.argv[1:]:
    run(n)
