"""Every ``tests/test_*.py`` must be collectable on its own (C-070, closes F-066).

The base defect
---------------
The repository carries no ``conftest.py``, ``t2pw`` is not installed into the venv as
an editable package, and ``pytest.ini`` declared no ``pythonpath``.  Each test file
therefore had to put ``<repo>/src`` on ``sys.path`` itself -- directly, or by importing
a ``tests/`` helper that does it -- and 21 of the 156 files do neither.  In a
*multi-file* run the first file that primes ``sys.path`` primes it for every file
collected after it, so those 21 are invisible: SMOKE and all four chunks stay green.
Collected *alone* they fail before a single test runs::

    ERROR tests/test_attempt_cap_termination_reason.py
    ModuleNotFoundError: No module named 't2pw'
    !!! Interrupted: 1 error during collection !!!

The remedy is one line in ``pytest.ini``: ``pythonpath = src``.  pytest prepends it at
config time, before any collection, so collection order stops mattering and a newly
written test file inherits the fix without having to remember the incantation.

There is no cheap "at-risk subset", because measured at ``e616846`` the obvious one is
wrong in both directions: of the 23 files with no ``sys.path`` mention, 5 collect alone
perfectly well (four reach ``src`` via ``from helpers_prefreeze import ...``, one needs
no ``t2pw``), while 3 files that *do* mention it still fail -- in a docstring, or inside
a test body, never at import time.  No static predicate separates the sets, so this file
does not pretend one does.

What these tests cover, and what they do not
--------------------------------------------
``test_pytest_ini_places_src_on_sys_path`` is the mechanism check and costs nothing.
It reads ``pythonpath`` through pytest's own config object -- not by string-matching the
file -- and confirms the entry it names really is on ``sys.path``.  It goes red if the
line is removed, renamed, misspelled or repointed, but cannot tell you that any
*particular* file collects.

``test_a_naive_new_test_file_collects_alone`` writes a throwaway test file into
``tests/`` that imports ``t2pw`` the way a newly written test would, collects it alone
in a fresh interpreter, and asserts it succeeds.  One interpreter start, well under a
second.  It is *generated* rather than named, so it is the standing answer to "what
happens when test file 157 arrives" and needs no maintenance when it does.

``test_every_test_file_collects_alone`` is the complete sweep, and the only one of the
three that catches a file failing in isolation for a reason other than ``sys.path``.
It is opt-in via ``T2PW_ISOLATED_COLLECT_ALL=1`` because it costs one interpreter start
per test file; the measured cost is in the C-070 report.

The ``PYTHONPATH`` scrub in :func:`_child_env` is load-bearing.  The sprint's measured
launcher (``docs/pwml_recovery_sprint/evidence/pinned_pytest.py``) *requires*
``PYTHONPATH=<tree>/src``, so a child process that inherited its parent's environment
would import ``t2pw`` from the environment and pass no matter what ``pytest.ini`` said.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

#: Non-vacuity floor.  The suite had 156 test files when this was written; a discovery
#: bug that quietly returns nothing, or a glob that stops matching, must fail loudly
#: rather than pass by finding nothing to check (RULING 13).
MIN_TEST_FILES = 100

#: Per-child wall clock.  Collection imports the module under test, nothing more.
COLLECT_TIMEOUT_SECONDS = 120

#: Environment variable that opts in to the complete sweep.
FULL_SWEEP_ENV = "T2PW_ISOLATED_COLLECT_ALL"

#: Optional path to write a machine-readable per-file result.  pytest truncates long
#: assertion messages, so the rendered failure is not a reliable census; this is.
REPORT_ENV = "T2PW_ISOLATED_COLLECT_REPORT"

#: The naive file, written the way a test author who has never heard of this problem
#: would write it.  It must live under ``tests/`` so that pytest's rootdir, ini file and
#: import mode are the real ones rather than a reconstruction.
CANARY_SOURCE = '''\
"""Throwaway file written by tests/test_isolated_collection.py; safe to delete.

Deliberately naive: it imports t2pw the way a newly written test file would, with no
sys.path handling of its own. Collected on its own it succeeds only because pytest.ini
declares `pythonpath = src`.
"""

import t2pw


def test_t2pw_is_importable() -> None:
    assert t2pw is not None
'''


def all_test_files() -> List[Path]:
    """Every test file pytest would collect from ``tests/``, sorted."""

    return sorted(TESTS_DIR.glob("test_*.py"))


def _child_env() -> Dict[str, str]:
    """A copy of the environment with every inherited import path removed."""

    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        env.pop(name, None)
    return env


def collect_alone(path: Path, basetemp: Path) -> Tuple[int, str, float]:
    """Collect *path* by itself in a fresh interpreter.  ``(returncode, output, seconds)``."""

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={basetemp}",
        path.relative_to(REPO_ROOT).as_posix(),
    ]
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=COLLECT_TIMEOUT_SECONDS,
    )
    elapsed = time.monotonic() - started
    return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), elapsed


def _write_report(scope: str, results: List[Dict[str, object]]) -> None:
    """Record one sweep under *scope* in the file named by :data:`REPORT_ENV`, if set."""

    destination = os.environ.get(REPORT_ENV, "")
    if not destination:
        return
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    document: Dict[str, object] = {}
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            document = loaded
    document[scope] = {
        "files": len(results),
        "failed": sum(1 for entry in results if entry["returncode"] != 0),
        "seconds_total": round(sum(float(entry["seconds"]) for entry in results), 3),
        "results": results,
    }
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sweep(scope: str, paths: List[Path], basetemp: Path) -> List[str]:
    """Collect each of *paths* alone; return one rendered failure per broken file."""

    results: List[Dict[str, object]] = []
    failures: List[str] = []
    for index, path in enumerate(paths):
        returncode, output, seconds = collect_alone(path, basetemp / f"c{index:03d}")
        results.append(
            {"file": path.name, "returncode": returncode, "seconds": round(seconds, 3)}
        )
        if returncode != 0:
            tail = "\n".join(output.strip().splitlines()[-12:])
            failures.append(f"{path.name} (exit {returncode})\n{tail}")
    _write_report(scope, results)
    return failures


def _render(failures: List[str]) -> str:
    """One line per broken file, then the full tail of the first only.

    pytest truncates a long assertion message, and a truncated census reads as a smaller
    defect than the one really present -- which is the mistake this card exists to
    avoid.  The names are short enough to survive; the detail is shown once.
    """

    names = [entry.splitlines()[0] for entry in failures]
    listing = "\n".join(f"  - {name}" for name in names)
    return (
        f"{len(failures)} test file(s) cannot be collected on their own:\n"
        f"{listing}\n\nfirst failure in full:\n{failures[0]}"
    )


def test_pytest_ini_places_src_on_sys_path(request: pytest.FixtureRequest) -> None:
    """``pytest.ini`` must put ``<repo>/src`` on ``sys.path`` for every run."""

    configured = list(request.config.getini("pythonpath"))
    assert configured, (
        "pytest.ini declares no `pythonpath`. Without it a test file can only import "
        "t2pw when some earlier file in the same run has already put src on sys.path, "
        "which is why the F-066 files fail when collected alone."
    )

    src = (REPO_ROOT / "src").resolve()
    resolved = {Path(entry).resolve() for entry in configured}
    assert src in resolved, f"pytest.ini `pythonpath` does not name {src}: {sorted(resolved)}"

    on_path = {Path(entry).resolve() for entry in sys.path if entry}
    assert src in on_path, f"{src} is configured but absent from sys.path: {sys.path[:5]}"


def test_a_naive_new_test_file_collects_alone(tmp_path: Path) -> None:
    """A brand-new test file that does nothing about ``sys.path`` still collects alone.

    The file is generated, not named, so this keeps working for test file 157 and every
    one after it.  It is removed on every exit path; a stray copy would be harmless
    (it holds one trivially passing test) but is still a bug in this test.
    """

    canary = TESTS_DIR / f"test_c070_isolation_canary_{os.getpid()}.py"
    assert not canary.exists(), f"a previous run left {canary.name} behind"
    canary.write_text(CANARY_SOURCE, encoding="utf-8")
    try:
        returncode, output, _ = collect_alone(canary, tmp_path / "canary")
    finally:
        canary.unlink(missing_ok=True)

    assert returncode == 0, (
        "a newly written test file cannot be collected on its own. pytest.ini is no "
        "longer putting src on sys.path, so every test file must prime it by hand:\n\n"
        + "\n".join(output.strip().splitlines()[-15:])
    )


@pytest.mark.skipif(
    os.environ.get(FULL_SWEEP_ENV, "") != "1",
    reason=f"complete sweep is opt-in: set {FULL_SWEEP_ENV}=1 (one interpreter start per file)",
)
def test_every_test_file_collects_alone(tmp_path: Path) -> None:
    """The complete sweep: every test file, collected by itself."""

    every = all_test_files()
    assert len(every) >= MIN_TEST_FILES, (
        f"discovered only {len(every)} test files under {TESTS_DIR}; expected at least "
        f"{MIN_TEST_FILES}. Discovery is broken, so this test would otherwise pass by "
        f"checking nothing."
    )

    failures = sweep("all", every, tmp_path)
    assert not failures, _render(failures)
