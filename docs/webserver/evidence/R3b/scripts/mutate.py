"""R3b mutation check: each single mutation of the moved code (service/extraction.py) or of
the app's call site / shim must make the named tests fail. Every pytest run goes through
bounded_run + pinned_pytest with the heavy lock. The file is restored after every mutation
(and in a finally); the caller checks ``git status`` afterwards.

usage: mutate.py [LO:HI]   (run from C:/t/ws-R3b, committed clean tree)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = "C:/t/ws-R3b"
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
SVC = Path(TREE, "src/t2pw/service/extraction.py")
APP = Path(TREE, "src/t2pw/app/streamlit_app.py")
EV = Path(TREE, "docs/webserver/evidence/R3b/mutation")
EV.mkdir(parents=True, exist_ok=True)
Path("C:/t/wsR3btmp/mut-tmp").mkdir(parents=True, exist_ok=True)  # the --basetemp parent must exist
NEW = ["tests/test_service_extraction_stage0.py"]
CLOBBER = ["tests/test_stage0_case_c_retry_clobber.py"]

MUTATIONS = {  # name: (file, old, new, tests)
    "retry-reads-full-text": (SVC, "        text[:_PREPROCESS_RETRY_CHARS],\n", "        text,\n", NEW),
    "retry-at-head-length": (SVC, "if len(text) <= _PREPROCESS_RETRY_CHARS:", "if len(text) < _PREPROCESS_RETRY_CHARS:", NEW),
    "case-c-retried": (SVC, "    if is_ambiguous_multi_example_review_context(first_context):\n",
                       "    if False:\n", NEW),
    "retry-adopted-unconditionally": (SVC, "if _stage0_context_rank(retry_context) > _stage0_context_rank(first_context):",
                                      "if True:", NEW),
    "case-c-rank-lowered": (SVC, "        return 2\n", "        return 0\n", NEW),
    "usable-ignores-lists": (SVC, "if isinstance(value, list) and any(", "if False and any(", NEW),
    "seed-organism-4-words": (SVC, "1 <= len(tail.split()) <= 3", "1 <= len(tail.split()) <= 4", NEW),
    "seed-overwrites-organism": (SVC, 'if organism_name and not str(seeded.get("likely_organism") or "").strip():',
                                 "if organism_name:", NEW),
    "first-call-temperature": (SVC, "        text, temperature=temperature, user_task_context=user_task_context\n",
                               "        text, temperature=0.0, user_task_context=user_task_context\n", NEW),
    "retry-drops-focus": (SVC, "        user_task_context=user_task_context,\n    )\n    # Defect 2",
                          "        user_task_context=None,\n    )\n    # Defect 2", NEW),
    "seed-info-as-warning": (SVC, "                reporter.info(\n", "                reporter.warning(\n", NEW),
    "failed-warning-as-info": (SVC, '                reporter.warning(\n                    f"Stage 0 failed:',
                               '                reporter.info(\n                    f"Stage 0 failed:', NEW),
    "recovered-branch-dropped": (SVC, "        elif preprocess_was_recovered(pathway_context):", "        elif False:", NEW),
    "spinner-text": (SVC, 'reporter.spinner("Running preprocessor...")', 'reporter.spinner("Running preprocessor")', NEW),
    "diagnostics-not-unique": (SVC, "        unique=True,\n", "        unique=False,\n", NEW),
    "diagnostics-root-not-paths": (SVC, 'artifact_dir=paths.PROJECT_ROOT / "tmp" / DIAGNOSTICS_DIRNAME,',
                                   'artifact_dir=Path.cwd() / "tmp" / DIAGNOSTICS_DIRNAME,', NEW),
    "diagnostics-dir-key-dropped": (SVC, '    state[EXTRACTION_DIAGNOSTICS_DIR_KEY] = str(recorder.artifact_dir or "")\n',
                                    "    pass\n", NEW),
    "post-stage0-sync-dropped": (SVC, "    _sync_extraction_diagnostics(state=state)\n\n    return StageZeroResult",
                                 "    return StageZeroResult", NEW),
    "sync-may-raise": (SVC, "    except Exception:  # noqa: BLE001 - diagnostics must never break the run",
                       "    except KeyError:  # noqa: BLE001 - diagnostics must never break the run", NEW),
    "state-and-reporter-defaulted": (SVC, "    state: MutableMapping[str, Any],\n    reporter: Reporter,\n",
                                     "    state: MutableMapping[str, Any] = None,\n    reporter: Reporter = None,\n", NEW),
    "app-state-not-session": (APP, "        state=st.session_state,\n        reporter=StreamlitReporter(st),\n        preprocess_fn=preprocess,\n",
                              "        state={},\n        reporter=StreamlitReporter(st),\n        preprocess_fn=preprocess,\n", NEW),
    "app-shim-ignores-app-preprocess": (APP, "        preprocess_fn=preprocess,\n    )\n\n\ndef emit_rag_loop_messages",
                                        "        preprocess_fn=lambda *a, **k: {},\n    )\n\n\ndef emit_rag_loop_messages", CLOBBER),
}

if len(sys.argv) > 1:  # optional subset by index range, e.g. 0:11
    lo, hi = (int(x) for x in sys.argv[1].split(":"))
    MUTATIONS = dict(list(MUTATIONS.items())[lo:hi])
originals = {p: p.read_bytes() for p in (SVC, APP)}
results = {}
try:
    for name, (target, old, new, tests) in MUTATIONS.items():
        src = originals[target].decode("utf-8")
        assert src.count(old) == 1, (name, src.count(old))
        target.write_bytes(src.replace(old, new).encode("utf-8"))
        env = dict(os.environ, PYTHONPATH=f"{TREE}/src", T2PW_OFFLINE_CURATOR="1")
        cmd = [PY, f"{TREE}/docs/pwml_recovery_sprint/evidence/bounded_run.py", "--label", f"mut-{name}",
               "--timeout", "300", "--heavy-lock", "ws-R3b", "--json", str(EV / f"{name}.json"), "--",
               PY, "-u", f"{TREE}/docs/pwml_recovery_sprint/evidence/pinned_pytest.py", "--expect-tree", TREE,
               "-q", "-p", "no:cacheprovider", f"--basetemp=C:/t/wsR3btmp/mut-tmp/{name}", *tests]
        out = subprocess.run(cmd, cwd=TREE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=400).stdout
        target.write_bytes(originals[target])
        (EV / f"{name}.log").write_text(out, encoding="utf-8")
        summary = re.findall(r"^(?:\d+ (?:failed|passed)|ERROR|=+ .*error).*$", out, re.M)
        surv = re.findall(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", out)
        results[name] = (summary[-1] if summary else "?", surv[-1] if surv else "?")
        print(f"{name:34s} {results[name][0][:70]}  survivors={results[name][1]}", flush=True)
finally:
    for p, b in originals.items():
        p.write_bytes(b)
killed = [n for n, (s, _) in results.items() if "failed" in s or "error" in s.lower()]
print(f"killed {len(killed)}/{len(MUTATIONS)}; survived: {sorted(set(MUTATIONS) - set(killed))}")
sys.exit(0 if len(killed) == len(MUTATIONS) else 1)
