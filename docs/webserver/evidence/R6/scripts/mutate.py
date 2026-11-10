"""Revert one piece of logic at a time on the scratch tree and record whether a test fails.

usage: mutate.py [name ...]   (no names = all)   output: C:/t/ev-R6/rev/tally.txt
"""
import os
import subprocess
import sys
import time
from pathlib import Path

TREE = Path("C:/t/ev-R6/rev/tree")
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
NEW = "tests/test_batch_driver_run_pathway.py"
PRE = "tests/test_batch_preflight.py"
SVC = "tests/test_service_run_pathway.py"
DRV, RUN, PW = "src/t2pw/batch/driver.py", "src/t2pw/batch/runner.py", "src/t2pw/service/pathway.py"

M = {
    "default-engine-not-run-pathway": (RUN, [("engine=engine or driver.ENGINE_PATHWAY,", "engine=engine,")], [NEW]),
    "engine-fn-always-apptest": (DRV, [("return run_one_pathway if _text(engine) == ENGINE_PATHWAY else run_one", "return run_one")], [NEW]),
    "run-single-ignores-the-plan": (RUN, [('driver.engine_fn(engine or load_plan(run_dir).get("engine"))', "driver.engine_fn(engine)")], [NEW]),
    "run-single-ignores-the-argument": (RUN, [('driver.engine_fn(engine or load_plan(run_dir).get("engine"))', 'driver.engine_fn(load_plan(run_dir).get("engine"))')], [NEW]),
    "no-empty-text-guard": (DRV, [("        if _paper_text(paper):\n            from t2pw.service import pathway", "        if True:\n            from t2pw.service import pathway")], [NEW]),
    "no-deadline": (DRV, [("time.time() + float(timeout))[0]", "None)[0]")], [NEW]),
    "research-leg-runs-strict": (DRV, [("pathway.PathwayRequest(export_mode=mode_token)", "pathway.PathwayRequest()")], [NEW]),
    "crash-not-caught": (DRV, [("    except Exception:  # noqa: BLE001 -- the same promise run_one makes", "    except KeyError:  # noqa: BLE001 -- the same promise run_one makes")], [NEW]),
    "env-var-ignored": (RUN, [("requested = engine or os.environ.get(driver.ENGINE_ENV)", "requested = engine")], [NEW]),
    "resume-cannot-override": (RUN, [('if engine and engine != plan.get("engine"):', "if False:")], [NEW]),
    "unknown-engine-not-a-usage-error": (RUN, [("        return 2\n    state: Dict[str, Any] = {\"run_dir\": None}", "        return 1\n    state: Dict[str, Any] = {\"run_dir\": None}")], [NEW]),
    "unknown-engine-accepted": (DRV, [("    if name not in ENGINES:\n        raise ValueError", "    if False:\n        raise ValueError")], [NEW]),
    "preflight-entry-missing": (RUN, [('"t2pw.service.pathway",\n        "the default leg engine', '"t2pw.service.pathwayX",\n        "the default leg engine')], [PRE]),
    "driver-imports-the-service-at-module-level": (DRV, [("from t2pw.paths import PACKAGE_ROOT\n", "from t2pw.paths import PACKAGE_ROOT\nfrom t2pw.service import pathway as _pw_cycle\n")], [NEW]),
    # part 2
    "lookahead-without-punctuation": (PW, [(r'''(?=[\\/\"'\s)\],.:;}]|$)''', r'''(?=[\\/\"'\s]|$)''')], [SVC]),
    "rescrub-as-text": (PW, [("    _rescrub_json_file(record, out_dir)\n", "    try:\n        record.write_text(_scrub_server_paths({'m': record.read_text(encoding=\"utf-8\")}, out_dir)['m'], encoding=\"utf-8\")\n    except OSError:\n        pass\n")], [SVC]),
    "rescrub-not-atomic": (PW, [("        tmp.write_text(json.dumps(data", "        record.write_text(json.dumps(data"), ("        os.replace(tmp, record)\n", "")], [SVC]),
    "rescrub-catches-only-oserror": (PW, [("    except (OSError, ValueError):  # ValueError", "    except OSError:  # ValueError")], [SVC]),
    "rescrub-leaves-the-temp-file": (PW, [("        with contextlib.suppress(OSError):\n            tmp.unlink()\n", "        pass\n")], [SVC]),
    "rescrub-skips-nested-values": (PW, [("    if isinstance(value, list):\n        return [_scrub_json_strings(v, pattern) for v in value]\n", "    if isinstance(value, list):\n        return value\n")], [SVC]),
}


def run(name):
    rel, subs, tests = M[name]
    target = TREE / rel
    pristine = target.read_bytes()
    text = pristine.decode("utf-8").replace("\r\n", "\n")
    for old, new in subs:
        assert text.count(old) == 1, (name, old[:60], text.count(old))
        text = text.replace(old, new)
    target.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    try:
        out = Path(f"C:/t/ev-R6/rev/{name}.out")
        env = dict(os.environ, PYTHONPATH=str(TREE / "src"), T2PW_OFFLINE_CURATOR="1")
        cmd = [PY, str(TREE / "docs/pwml_recovery_sprint/evidence/bounded_run.py"), "--label", f"R6-rev-{name}",
               "--timeout", "540", "--heavy-lock", "R6", "--json", f"C:/t/ev-R6/rev/{name}.json", "--",
               PY, "-u", str(TREE / "docs/pwml_recovery_sprint/evidence/pinned_pytest.py"), "--expect-tree", str(TREE),
               "-q", f"--basetemp=C:/t/bt/R6/rev-{name}-{int(time.time())}", *tests]
        with open(out, "wb") as fh:
            rc = subprocess.run(cmd, cwd=str(TREE), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    finally:
        target.write_bytes(pristine)
    lines = out.read_text(encoding="utf-8", errors="replace").splitlines()
    summary = next((l for l in reversed(lines) if " passed" in l or " failed" in l), "(no summary)")
    failed = sorted({l.split(" - ")[0].replace("FAILED ", "").split("::")[-1] for l in lines if l.startswith("FAILED ")})
    verdict = "KILLED" if rc == 1 and failed else ("SURVIVED" if rc == 0 else f"ERROR rc={rc}")
    row = f"{verdict:9} {name:48} {summary.strip()} | {', '.join(failed)[:260]}"
    with open("C:/t/ev-R6/rev/tally.txt", "a", encoding="utf-8") as fh:
        fh.write(row + "\n")
    print(row, flush=True)


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(M)):
        run(name)
