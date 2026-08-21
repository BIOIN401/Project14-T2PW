"""What ACTUALLY breaks in the child when one declared module is unimportable.

REV-069 rejected two of C-069's three ``CHILD_IMPORTS`` reason strings because they
asserted a runtime failure mode from a static read of the deferral sites alone --
[S5]: "Never assert a runtime behaviour from a static code path alone." A reason
string is ``PreflightProblem.why``, printed to an operator at 2am, so it has to be
measured like anything else this sprint asserts.

The deferred site is only the FIRST loss if nothing on the child's module-scope path
already needs the module. Three probes per poisoned module, each a real child:

* ``import t2pw.batch.driver``      -- the child's entry point (``driver.run_one``).
* ``import t2pw.pipeline.pipeline`` -- the extraction pipeline every leg runs.
* the app's UNGUARDED module-scope imports, executed in order. ``driver.py:111``
  says the app is "Never imported -- AppTest execs it as a script", so what a child
  must satisfy is that statement list, not an ``import`` of the module.

Poisoning is C-069's mechanism unchanged: a ``sitecustomize`` on the child's
``PYTHONPATH`` installing a ``sys.meta_path`` finder that raises for one dotted name.

Section 2 settles the two claims C-069's ``_submodule_names`` docstring made about
the form it declined (``find_spec(f"{pkg}.{name}")``): its marginal import cost, and
what it really does when ``sys.modules`` carries a ``MagicMock``.

Output path is resolved before anything else (F-045).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
APP = SRC / "t2pw" / "app" / "streamlit_app.py"
TARGETS = ("t2pw.pipeline.release_status", "t2pw.pipeline.strict_quarantine",
           "t2pw.pipeline.deadline")

_POISON = '''\
import sys

TARGET = {target!r}


class _DenyOne:
    def find_spec(self, name, path=None, target=None):
        if name == TARGET:
            raise ImportError("C-069 poison: %s is unimportable here" % name)
        return None


sys.meta_path.insert(0, _DenyOne())
'''


def _app_module_scope_imports() -> List[str]:
    """The app's UNGUARDED module-scope import statements, verbatim, in order.

    Direct children of ``Module`` only: an import inside a ``try`` is guarded, and a
    guarded import is not something a child is obliged to satisfy.
    """

    source = APP.read_text(encoding="utf-8")
    lines = source.splitlines()
    out: List[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
    return out


def _child(code: str, env: Dict[str, str]) -> Dict[str, Any]:
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300, env=env)
    tail = [line for line in done.stderr.strip().splitlines() if line.strip()][-1:]
    return {"exit": done.returncode, "stderr_tail": (tail or [""])[0][:200]}


def _matrix() -> List[Dict[str, Any]]:
    seed = "import sys; sys.path.insert(0, %r)\n" % str(SRC)
    app_block = seed + "\n".join(_app_module_scope_imports()) + "\n"
    rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        with tempfile.TemporaryDirectory(prefix="c069reality") as tmp:
            poison = Path(tmp)
            (poison / "sitecustomize.py").write_text(
                _POISON.format(target=target), encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(poison)
            rows.append({
                "poisoned": target,
                "import_driver": _child(seed + "import t2pw.batch.driver", env),
                "import_pipeline": _child(seed + "import t2pw.pipeline.pipeline", env),
                "app_module_scope": _child(app_block, env),
            })
    return rows


_COST = """\
import sys, json, time
sys.path.insert(0, %r)
import importlib.util, pkgutil
n0 = len(sys.modules); t0 = time.perf_counter()
spec = importlib.util.find_spec('streamlit.testing.v1')
list(pkgutil.iter_modules(list(getattr(spec, 'submodule_search_locations', None) or ())))
n1 = len(sys.modules); t1 = time.perf_counter()
importlib.util.find_spec('streamlit.testing.v1.AppTest')
print(json.dumps({'chosen_modules': n1 - n0, 'chosen_s': round(t1 - t0, 3),
                  'declined_extra_modules': len(sys.modules) - n1,
                  'declined_extra_s': round(time.perf_counter() - t1, 3)}))
"""

_MOCKED = """\
import sys, json
sys.path.insert(0, %r)
import importlib.util
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()
out = {}
try:
    result = importlib.util.find_spec('streamlit.testing.v1.AppTest')
    out['declined'] = 'returned ' + type(result).__name__
except BaseException as exc:
    out['declined'] = type(exc).__name__ + ': ' + str(exc)[:90]
print(json.dumps(out))
"""

_FOOTPRINT = """\
import sys, json
sys.path.insert(0, %r)
sys.path.insert(0, %r)
from test_batch_preflight import DRIVER_PATH, _deferred_imports
before = set(sys.modules)
_deferred_imports(DRIVER_PATH)
added = sorted(set(sys.modules) - before)
roots = ('streamlit', 'anyio', 'click', 'blinker', 'asyncio', 'tornado')
print(json.dumps({'added_count': len(added),
                  'notable_roots': sorted({m.split('.')[0] for m in added
                                           if m.split('.')[0] in roots})}))
"""


def _run_json(code: str) -> Any:
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300)
    if done.returncode != 0:
        return {"error": done.stderr[-300:]}
    return json.loads(done.stdout.strip().splitlines()[-1])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out_path = Path(args.out).resolve()  # BEFORE anything runs (F-045)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _matrix()
    record = {
        "question": "what actually breaks in the child when one declared module is unimportable?",
        "app_module_scope_statements": len(_app_module_scope_imports()),
        "matrix": rows,
        "declined_form_cost": _run_json(_COST % str(SRC)),
        "declined_form_under_magicmock": _run_json(_MOCKED % str(SRC)),
        "guard_import_footprint": _run_json(_FOOTPRINT % (str(SRC), str(ROOT / "tests"))),
    }
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    print("app module-scope import statements: %d" % record["app_module_scope_statements"])
    print("%-34s %8s %9s %10s" % ("poisoned", "driver", "pipeline", "app-scope"))
    for row in rows:
        print("%-34s %8d %9d %10d" % (
            row["poisoned"], row["import_driver"]["exit"],
            row["import_pipeline"]["exit"], row["app_module_scope"]["exit"]))
    for row in rows:
        for probe in ("import_driver", "import_pipeline", "app_module_scope"):
            if row[probe]["exit"]:
                print("  %s / %s: %s" % (row["poisoned"], probe, row[probe]["stderr_tail"]))
    print("declined_form_cost: %s" % json.dumps(record["declined_form_cost"], sort_keys=True))
    print("declined_under_magicmock: %s" % json.dumps(record["declined_form_under_magicmock"], sort_keys=True))
    print("guard_import_footprint: %s" % json.dumps(record["guard_import_footprint"], sort_keys=True))
    print("wrote %s  (exists=%s)" % (out_path, out_path.is_file()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
