"""Does the batch preflight actually CATCH a broken deferred module? (F-073, C-069)

A list-consistency test going green proves ``CHILD_IMPORTS`` agrees with
``driver.py``. It does not prove the preflight gained coverage. This does, through
the production entry point (F-074): the real ``check_preflight``, the real
``probe_imports``, a real child process, and a module that is genuinely
unimportable in that child.

``_probe_source`` does ``sys.path.insert(0, PROJECT_ROOT/'src')`` inside the child,
so no ``PYTHONPATH`` ordering trick can hide a module from it. A ``sitecustomize``
on the child's ``PYTHONPATH`` is imported by ``site`` at startup, before ``-c``
runs, and installs a ``sys.meta_path`` finder that raises ``ImportError`` for
exactly one dotted name. Meta-path finders outrank every path-based finder, so the
break is total, targeted, and confined to children -- the parent is already running.
``child_env()`` copies ``os.environ``, which is what carries it into the probe child.

Silence has two possible causes, so each case carries three controls:
``poison_bites`` (a bare ``import`` under the same env must exit non-zero),
``control_module_reported`` (an untouched declared module must NOT be reported) and
``probe_fatal`` (the probe must have answered at all). ``--expect blind`` (base) and
``--expect covered`` (tip) make the artifact self-validating.

Output path is resolved before anything else (F-045).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.batch import runner  # noqa: E402

DEFAULT_MODULES = ("t2pw.pipeline.release_status", "t2pw.pipeline.strict_quarantine")
CONTROL_MODULE = "t2pw.batch.driver"

_SITECUSTOMIZE = '''\
import sys

TARGET = {target!r}


class _DenyOne:
    """Raise for exactly one dotted name; defer everything else."""

    def find_spec(self, name, path=None, target=None):
        if name == TARGET:
            raise ImportError("C-069 probe poison: %s is unimportable here" % name)
        return None


sys.meta_path.insert(0, _DenyOne())
'''


def _one_case(module: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="c069poison") as tmp:
        poison = Path(tmp)
        (poison / "sitecustomize.py").write_text(
            _SITECUSTOMIZE.format(target=module), encoding="utf-8")
        saved = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(poison) + (os.pathsep + saved if saved else "")
        try:
            bite = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                capture_output=True, text=True, timeout=120)
            report = runner.check_preflight(executable=sys.executable)
        finally:
            if saved is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = saved

    subjects = [problem.subject for problem in report.problems]
    named = [problem for problem in report.problems if problem.subject == module]
    return {
        "poisoned_module": module,
        "declared_in_child_imports": module in {n for n, _w in runner.CHILD_IMPORTS},
        "poison_bites": bite.returncode != 0,
        "poison_child_stderr_tail": bite.stderr.strip().splitlines()[-1:] or [""],
        "preflight_problem_subjects": subjects,
        "reported": bool(named),
        "reported_why": named[0].why if named else "",
        "reported_error": named[0].error if named else "",
        "control_module_reported": CONTROL_MODULE in subjects,
        "probe_fatal": [s for s in subjects if s == "the child process itself"],
    }


def _reaches_llm_client(names: List[str]) -> bool:
    source = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        f"[__import__(n) for n in {names!r}]\n"
        "print('t2pw.llm.client' in sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", source],
                          capture_output=True, text=True, timeout=180)
    return done.returncode == 0 and done.stdout.strip().endswith("True")


def _llm_client_reach() -> Dict[str, Any]:
    """Is ``t2pw.llm.client`` still the one entry nothing else on the list reaches?

    ``runner.py:1341`` says so of the pre-C-069 entries -- "importing all four of
    the entries above in a fresh interpreter leaves ``t2pw.llm.client`` absent from
    ``sys.modules``", measured 2026-07-28 -- and that measurement is the stated
    reason entry 5 exists. Re-measured here per module, in a fresh interpreter,
    because it is a claim about a tree that has moved on. RECORDED, NOT JUDGED: it
    is an observation about pre-existing prose, not about this card's change, so it
    is deliberately kept out of ``verdict``.
    """

    others = [name for name, _why in runner.CHILD_IMPORTS if name != "t2pw.llm.client"]
    return {
        "imported": others,
        "per_module": {name: _reaches_llm_client([name]) for name in others},
        "llm_client_in_sys_modules": _reaches_llm_client(others),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect", required=True, choices=("blind", "covered"))
    parser.add_argument("--module", action="append", default=None)
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve()  # BEFORE anything runs (F-045)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The measured tree is verified, not asserted (TEST_MATRIX 0 rule 10).
    resolved = Path(runner.__file__).resolve()
    cases = [_one_case(module) for module in (args.module or list(DEFAULT_MODULES))]
    reach = _llm_client_reach()

    want = args.expect == "covered"
    verdict = {
        "tree_ok": resolved.is_relative_to(ROOT / "src" / "t2pw"),
        "all_poisons_bit": all(case["poison_bites"] for case in cases),
        "no_probe_fatal": all(not case["probe_fatal"] for case in cases),
        "controls_clean": all(not case["control_module_reported"] for case in cases),
        "observation_matches_expectation": all(case["reported"] is want for case in cases),
    }
    record = {
        "question": "does check_preflight report a broken deferred module from a real probe child?",
        "expect": args.expect,
        "runner_file": str(resolved),
        "project_root": str(runner.PROJECT_ROOT),
        "child_imports_declared": [name for name, _why in runner.CHILD_IMPORTS],
        "cases": cases,
        "llm_client_reach": reach,
        "verdict": verdict,
        "passed": all(verdict.values()),
    }
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    print(f"runner: {resolved}")
    for case in cases:
        print(f"  {case['poisoned_module']}: declared={case['declared_in_child_imports']} "
              f"poison_bites={case['poison_bites']} reported={case['reported']} "
              f"control_reported={case['control_module_reported']}")
        if case["reported"]:
            print(f"      PreflightProblem.error: {case['reported_error']}")
            print(f"      PreflightProblem.why:   {case['reported_why']}")
    print(f"llm_client_reach: {json.dumps(reach, sort_keys=True)}")
    print(f"verdict: {json.dumps(verdict, sort_keys=True)}")
    print(f"wrote {out_path}  (exists={out_path.is_file()})")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
