"""Tests for the parent-side preflight environment check.

WHY this file exists, in one run directory
------------------------------------------
``runs/2026-07-27_2135`` is on disk and can be read line by line. The parent
started at 21:35:06, fetched 28 papers by 21:35:49, and then recorded all 56
paper+mode legs as failures between 21:35:49 and 21:36:13 -- 24 seconds for work
that takes ten hours when it is real. All 56 manifest rows carry the same detail
(verbatim, so the line number is the one driver.py had that night; the same
import is at line 1199 today)::

    File "...\\src\\t2pw\\batch\\driver.py", line 1074, in _drive
        from streamlit.testing.v1 import AppTest
    ModuleNotFoundError: No module named 'streamlit'

That import is deferred to ``_drive``, so nothing in the parent -- not the plan,
not the fetch, not ``import t2pw.batch.driver`` -- ever touched it. One
environment fault was therefore rediscovered once per child and written to the
manifest 56 times as ``failure_kind=crash``; SUMMARY.txt then opened with
``!! RESEARCH-MODE DEFECT !! papers affected: 28``, i.e. 28 pipeline defects that
do not exist. The cause was never in this codebase: on Windows ``.py`` is
associated with ``C:\\WINDOWS\\py.exe``, which ignores an active virtualenv, so
``scripts\\batch_run.py`` and ``.venv\\Scripts\\python.exe scripts\\batch_run.py``
are two different interpreters and only one of them has streamlit.

What is asserted here is therefore narrow and absolute: the check knows every
module a child defers (proved by parsing driver.py, not by trusting a list), it
fails on a missing OR broken one, it names the cure, it exits with a code nobody
else uses, and -- the part that matters most in the morning -- it leaves the disk
untouched, because a run directory *is* the record that a night happened.

The last test spawns a real interpreter with a poisoned ``streamlit`` on its
PYTHONPATH: exit codes and "wrote nothing" are properties of a process, and an
in-process ``main()`` call cannot prove either.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch import driver, runner  # noqa: E402

CLI_PATH = ROOT / "scripts" / "batch_run.py"
DRIVER_PATH = SRC / "t2pw" / "batch" / "driver.py"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cli() -> Any:
    spec = importlib.util.spec_from_file_location("t2pw_batch_run_cli_preflight", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def _submodule_names(package: str) -> frozenset:
    """The real submodule names of ``package``, WITHOUT importing ``package``.

    ``find_spec(package)`` imports only the ANCESTORS of ``package``, never
    ``package`` itself, and hands back ``submodule_search_locations`` when -- and
    only when -- it is a package. ``pkgutil.iter_modules`` then reads those
    directories off disk.

    The ancestors are not free, and an earlier draft of this docstring claimed they
    were. MEASURED: one ``_deferred_imports(DRIVER_PATH)`` adds **483** modules to
    ``sys.modules`` -- the whole ``streamlit`` runtime plus ``anyio``, ``asyncio``,
    ``blinker`` and ``click`` -- because ``streamlit.testing.v1``'s ancestors get
    imported to locate it. The true and narrow claim is only this: the module whose
    module-ness is being decided is never executed; its ancestors are.

    Why not ``find_spec(f"{package}.{name}")`` -- with the reasons that survived
    measurement. NOT cost: that form adds 18 modules and 0.058s on top of what this
    one already pays, which is noise against 483 and 0.373s. NOT ``MagicMock``
    safety: under a poisoned ``sys.modules['streamlit']`` it raises
    ``ModuleNotFoundError: ... 'streamlit' is not a package``, which the
    ``BaseException`` handler below would have caught anyway. What is left is
    small and true -- this form resolves ONCE PER PACKAGE and answers every name on
    the statement from one cached directory listing, where the other asks the import
    system once per imported name.

    A module that is not a package answers the empty set, which is the truthful
    answer: ``t2pw.pipeline.release_status`` has no submodules, so
    ``RELEASE_STATES`` cannot be one.

    ``BaseException`` on the way out, and the empty set as the fallback: an
    unresolvable package must degrade to the OLD, narrower behaviour rather than
    invent names. Being unable to answer is never grounds for a new assertion.
    """

    try:
        spec = importlib.util.find_spec(package)
        locations = list(getattr(spec, "submodule_search_locations", None) or ())
    except BaseException:  # noqa: BLE001 - an unimportable ancestor is not our verdict
        return frozenset()
    return frozenset(name for _finder, name, _ispkg in pkgutil.iter_modules(locations))


def _deferred_imports(path: Path) -> List[str]:
    """Every module imported INSIDE a function in ``path``, by dotted name.

    Parsed rather than grepped so an import that is indented for any reason --
    inside a ``try``, inside a nested helper -- is still found. These are exactly
    the imports the parent does not exercise by importing the module itself, and
    therefore exactly the ones a preflight has to know about.

    ``from <pkg> import <name>`` RESOLVES THE SUBMODULE (F-086)
    -----------------------------------------------------------
    ``inner.module`` alone is the package, and for
    ``from t2pw.pipeline import deadline`` (``driver.py:1718``) that recorded
    ``t2pw.pipeline`` and threw ``deadline`` away. ``_covered`` was then asked
    about the PARENT and answered True -- correctly, because
    ``t2pw.pipeline.export_mode`` is listed and startswith ``t2pw.pipeline.`` --
    while ``t2pw.pipeline.deadline`` itself was undeclared and unvalidatable in the
    child. ``_covered``'s docstring already refuses to cover descendants; the
    descendant name simply never reached it. A guard asking the wrong question is
    strictly worse than one that finds nothing, because it looks like success.

    So the package is still recorded -- ``from t2pw.pipeline import deadline`` does
    import ``t2pw.pipeline``, and dropping that would lose coverage this helper
    already had -- and each imported name that is a REAL SUBMODULE is recorded as
    well. A name that is an ordinary attribute is not: ``AppTest`` is a class
    re-exported by ``streamlit.testing.v1``, and promoting it to
    ``streamlit.testing.v1.AppTest`` would put a non-module in ``missed`` and make
    the guard unsatisfiable. Both directions are asserted by
    ``test_a_from_package_import_resolves_submodules_but_not_attributes``.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom):
                if inner.level == 0 and inner.module:
                    found.append(inner.module)
                    children = _submodule_names(inner.module)
                    found.extend(
                        f"{inner.module}.{alias.name}"
                        for alias in inner.names
                        if alias.name in children
                    )
            elif isinstance(inner, ast.Import):
                found.extend(alias.name for alias in inner.names)
    return found


def _covered(module_name: str) -> bool:
    """Does :data:`runner.CHILD_IMPORTS` prove ``module_name`` is importable?

    A listed name covers itself and every ancestor of itself: importing
    ``streamlit.testing.v1`` necessarily imports ``streamlit``. It does NOT cover
    its own descendants -- importing ``t2pw.rag`` says nothing about
    ``t2pw.rag.research_report``, which is one of the two modules this whole
    check exists for.
    """

    return any(
        listed == module_name or listed.startswith(module_name + ".")
        for listed, _why in runner.CHILD_IMPORTS
    )


def _probe_reporting(*broken: str):
    """A ``probe_fn`` whose imaginary child cannot import ``broken``."""

    def fake(names: Any, *, executable: str = "", **_kwargs: Any):
        return ([(name, f"ModuleNotFoundError: No module named '{name.split('.')[0]}'")
                 for name in names if name in broken], "")

    return fake


def _probe_that_cannot_run(reason: str):
    """A ``probe_fn`` whose child process never answered at all."""

    def fake(_names: Any, *, executable: str = "", **_kwargs: Any):
        return ([], reason)

    return fake


# ---------------------------------------------------------------------------
# The list is derived from the code, not from memory.
# ---------------------------------------------------------------------------
def test_every_import_driver_defers_is_covered_by_the_preflight() -> None:
    """The regression guard for the whole feature.

    A preflight that checks a hand-written list is one refactor away from being
    useless again: move one more import into a function to shave startup time and
    the parent silently stops covering it. So the list is checked against
    driver.py itself. Stdlib modules are exempt (an interpreter that cannot
    import ``json`` has already failed in louder ways).
    """

    third_party = [
        name
        for name in _deferred_imports(DRIVER_PATH)
        if name.split(".")[0] not in sys.stdlib_module_names
    ]
    # NON-VACUITY FLOOR (RULING 13). This guard derives its list from driver.py, so
    # a parse that finds NOTHING passes it -- and looks exactly like a list that is
    # genuinely complete. Measured for C-069: neutralize ``_deferred_imports`` to
    # return ``[]`` and the assertion below goes green against a CHILD_IMPORTS that
    # is missing six real deferral sites. This is the line that tells the two apart.
    assert third_party, (
        "_deferred_imports found no non-stdlib deferred import in driver.py at all. "
        "That is a broken parse, not a clean bill of health -- the check below would "
        "pass vacuously."
    )
    missed = [name for name in third_party if not _covered(name)]
    assert missed == [], (
        "driver.py defers these imports and the preflight would not catch them "
        f"missing: {missed}. Add them to runner.CHILD_IMPORTS with the reason a "
        "child needs them."
    )


def test_a_from_package_import_resolves_submodules_but_not_attributes(tmp_path) -> None:
    """F-086, both directions, because only one of them is safe to get wrong.

    ``from <pkg> import <name>`` is two different statements wearing one syntax.
    When ``<name>`` is a submodule the child imports a module the preflight has to
    know about; when it is a class or a constant there is no such module, and
    minting ``<pkg>.<name>`` would put a name in ``missed`` that can never be
    declared, leaving the guard permanently and unfixably red. So the permissive
    direction is asserted just as hard as the strict one.

    The synthetic file carries both cases; the real ``driver.py`` assertion below
    it is the live instance F-086 was found on.
    """

    source = tmp_path / "sample.py"
    source.write_text(
        """
def f():
    from t2pw.pipeline import deadline                              # a submodule
    from t2pw.pipeline.release_status import RELEASE_STATES         # a constant
    from streamlit.testing.v1 import AppTest                        # a class
""",
        encoding="utf-8",
    )
    found = _deferred_imports(source)

    assert "t2pw.pipeline.deadline" in found, (
        "a submodule imported as 'from <pkg> import <submodule>' must be resolved to "
        f"its own dotted name, or the preflight cannot probe it. got: {found}"
    )
    assert "t2pw.pipeline" in found, (
        "the package is still imported by that statement and must still be recorded"
    )
    assert "t2pw.pipeline.release_status.RELEASE_STATES" not in found, (
        "RELEASE_STATES is a constant, not a submodule; minting a module name for it "
        f"would make the guard unsatisfiable. got: {found}"
    )
    assert "streamlit.testing.v1.AppTest" not in found, (
        f"AppTest is a class re-exported by that package, not a submodule. got: {found}"
    )

    # The classifier itself, stated directly: a package answers with its real
    # children, and a MODULE answers with the empty set rather than guessing.
    assert "deadline" in _submodule_names("t2pw.pipeline")
    assert "export_mode" in _submodule_names("t2pw.pipeline")
    assert _submodule_names("t2pw.pipeline.release_status") == frozenset()

    assert "t2pw.pipeline.deadline" in _deferred_imports(DRIVER_PATH), (
        "driver.py:1718 defers 'from t2pw.pipeline import deadline'; if that site is "
        "gone this assertion should be retired deliberately, not deleted in passing"
    )


def test_the_llm_backend_is_probed_even_though_nothing_else_reaches_it() -> None:
    """``t2pw.llm.client`` is the one entry no other entry can prove.

    The list above it is derived from ``driver.py``'s deferred imports, and the
    client is not one of them: importing all four of the others in a fresh
    interpreter (measured 2026-07-28) leaves ``t2pw.llm.client`` absent from
    ``sys.modules``, so until it was listed the preflight was blind to the LLM
    backend entirely. It has to be probed because it raises at IMPORT time on three
    faults that are exactly the "one environment fault, 56 identical crashes after
    the fetch" shape this whole file exists to stop -- a missing
    ``OPENROUTER_API_KEY`` (``client.py:38``), one that does not start with
    ``sk-or-`` (``:41``), and an ``LLM_PROVIDER`` that is neither ``local`` nor
    ``openrouter`` (``:63``) -- and every stage of every leg calls ``chat()``.

    Asserted by NAME rather than by ``_covered``: a future refactor that happens to
    make some other listed module import the client would satisfy coverage while
    silently deleting the operator's reason for trusting the check.
    """

    listed = {name for name, _why in runner.CHILD_IMPORTS}
    assert "t2pw.llm.client" in listed


def test_the_two_imports_that_burned_the_night_are_named_explicitly() -> None:
    """Named, not merely covered: these are the ones with a run directory."""

    listed = {name for name, _why in runner.CHILD_IMPORTS}
    assert "streamlit.testing.v1" in listed
    assert "t2pw.rag.research_report" in listed
    for _name, why in runner.CHILD_IMPORTS:
        assert why.strip(), "every entry must say WHY, it is printed to the operator"


def test_the_real_environment_passes_its_own_preflight() -> None:
    """The check must be true of the interpreter that is meant to run the batch.

    Without this the suite could go green on a preflight that rejects every
    environment, including the correct one. No ``importorskip`` guard: the probe
    is a fresh process, so this holds even though four other test modules have
    already put a ``MagicMock`` in *this* process's ``sys.modules["streamlit"]``
    -- which is the property that made the probe a subprocess in the first place.
    """

    report = runner.check_preflight()
    assert [(p.subject, p.error) for p in report.problems] == []
    assert report.ok is True


def test_the_probe_is_immune_to_a_doctored_sys_modules(monkeypatch: Any) -> None:
    """A ``MagicMock`` in this process must not condemn a healthy environment.

    ``tests/test_stage0_scope_fallback.py``, ``test_stage0_case_c_retry_clobber.py``
    and the two ``test_rag_*`` modules install a stub as
    ``sys.modules["streamlit"]`` at *import* time -- they have to, because
    importing ``t2pw.app.streamlit_app`` executes a Streamlit script -- and pytest
    imports every test module during collection. An in-process preflight therefore
    reported "No module named 'streamlit'" on a machine with streamlit 1.58.0
    installed, and three pre-existing CLI tests in ``test_batch_run.py`` started
    failing. The child is a subprocess, so the question is asked in a subprocess.
    """

    monkeypatch.setitem(sys.modules, "streamlit", object())
    for name in [n for n in list(sys.modules) if n.startswith("streamlit.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    failures, fatal = runner.probe_imports(
        ["streamlit.testing.v1"], executable=sys.executable
    )
    assert fatal == ""
    assert failures == []


# ---------------------------------------------------------------------------
# What it catches.
# ---------------------------------------------------------------------------
def test_a_missing_streamlit_is_a_preflight_problem_with_its_reason() -> None:
    report = runner.check_preflight(probe_fn=_probe_reporting("streamlit.testing.v1"))
    assert report.ok is False
    assert [p.subject for p in report.problems] == ["streamlit.testing.v1"]
    problem = report.problems[0]
    assert "No module named 'streamlit'" in problem.error
    assert "AppTest" in problem.why  # the operator is told what it is for
    assert problem.cure


def test_a_broken_install_is_caught_as_well_as_a_missing_one(tmp_path: Path) -> None:
    """``find_spec`` would have called this environment healthy.

    A half-finished ``pip install`` or a compiled dependency built for the wrong
    Python leaves a package that is *present* and raises on execution. The probe
    does a real import, so this is caught; a spec-only check would let the batch
    sail into 56 identical ImportErrors instead. Proved against a real process
    with a real poisoned package rather than against a stub of one.
    """

    stub = tmp_path / "half_installed"
    (stub / "streamlit").mkdir(parents=True)
    (stub / "streamlit" / "__init__.py").write_text(
        "raise ImportError('DLL load failed while importing _pywrap: "
        "%1 is not a valid Win32 application')\n",
        encoding="utf-8",
    )
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(stub) + os.pathsep + old
    try:
        failures, fatal = runner.probe_imports(
            ["streamlit.testing.v1"], executable=sys.executable
        )
    finally:
        os.environ["PYTHONPATH"] = old
        if not old:
            os.environ.pop("PYTHONPATH", None)

    assert fatal == ""
    assert failures and failures[0][0] == "streamlit.testing.v1"
    assert "DLL load failed" in failures[0][1]


def test_a_module_that_exits_at_import_time_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    """``SystemExit`` is not an ``Exception``.

    A module that calls ``sys.exit()`` at import time would escape an
    ``except Exception`` in the probe and make the probe process exit silently --
    turning a diagnosable environment fault back into the unexplained thing this
    file exists to prevent. The probe catches ``BaseException`` for that reason.
    """

    stub = tmp_path / "suicidal"
    (stub / "streamlit").mkdir(parents=True)
    (stub / "streamlit" / "__init__.py").write_text("import sys; sys.exit(1)\n", encoding="utf-8")
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(stub) + os.pathsep + old
    try:
        failures, fatal = runner.probe_imports(
            ["streamlit.testing.v1"], executable=sys.executable
        )
    finally:
        os.environ["PYTHONPATH"] = old
        if not old:
            os.environ.pop("PYTHONPATH", None)

    assert fatal == "", "the probe must still have answered"
    assert failures and failures[0][0] == "streamlit.testing.v1"
    assert "SystemExit" in failures[0][1]


def test_a_missing_app_script_is_caught_in_the_parent_too(tmp_path: Path) -> None:
    """Same shape, different cause: driver reports it once per child otherwise."""

    report = runner.check_preflight(
        probe_fn=_probe_reporting(), app_path=tmp_path / "not_the_app.py"
    )
    assert report.ok is False
    assert any("not_the_app.py" in problem.subject for problem in report.problems)


def test_an_unrunnable_interpreter_is_caught(tmp_path: Path) -> None:
    """Every child is ``[sys.executable, batch_run.py, --single, ...]``.

    An interpreter that is not on disk makes every spawn fail with the same
    FileNotFoundError: the identical 56-rows-of-one-fault shape, from a different
    direction. This runs the real probe, so it also proves the probe reports a
    failed *spawn* rather than raising it.
    """

    failures, fatal = runner.probe_imports(
        ["streamlit.testing.v1"], executable=str(tmp_path / "no_such_python.exe")
    )
    assert failures == []
    assert "could not be started" in fatal

    report = runner.check_preflight(executable=str(tmp_path / "no_such_python.exe"))
    assert report.ok is False
    assert any("could not be started" in problem.error for problem in report.problems)


def test_an_empty_sys_executable_is_caught_without_spawning_anything() -> None:
    failures, fatal = runner.probe_imports(["streamlit.testing.v1"], executable="")
    assert failures == []
    assert "no child process can be started" in fatal


def test_a_probe_that_cannot_answer_is_itself_the_failure() -> None:
    """Never "the question could not be asked, so carry on".

    A probe that dies before printing (a broken interpreter, a sitecustomize that
    raises, a DLL that will not load) has already demonstrated that a child cannot
    run. Treating silence as consent would start the night on the strength of a
    question nobody managed to ask.
    """

    report = runner.check_preflight(
        probe_fn=_probe_that_cannot_run("the probe process exited with code 1 without answering")
    )
    assert report.ok is False
    assert "without answering" in report.problems[0].error
    assert "child process" in report.problems[0].subject


def test_noise_printed_around_the_probes_answer_cannot_refuse_a_healthy_night(
    tmp_path: Path,
) -> None:
    """A talkative environment must not be mistaken for a broken one.

    Raised in review of this change on 2026-07-28. The first version of
    :func:`runner.probe_imports` did ``stdout.split(sentinel, 1)[1]`` and fed the
    whole remainder to ``json.loads``, and the probe wrote its verdict with no
    trailing newline. So ANY byte that reached the probe's stdout after the
    verdict -- an ``atexit`` handler, a late-flushed ``sitecustomize`` banner, a
    corporate wrapper interpreter announcing itself -- made the JSON unparseable,
    which ``probe_imports`` returns as ``fatal``, which ``check_preflight`` turns
    into a problem, which exits 3 and refuses a ten-hour batch on a machine where
    all four imports actually succeed. That is the same class of harm as the
    2026-07-27 night this file exists to prevent, only inverted: a night that did
    not happen because of the check rather than in spite of it.

    Proved with a REAL ``sitecustomize`` on a real child's PYTHONPATH, printing
    both before the probe body runs and from an ``atexit`` handler after it, which
    is exactly how the field version of this arrives. A stubbed stdout string
    would not prove the newline framing in ``_probe_source``.
    """

    stub = tmp_path / "chatty"
    stub.mkdir(parents=True)
    (stub / "sitecustomize.py").write_text(
        "import atexit, sys\n"
        "sys.stdout.write('loading corporate python shim v4.2')\n"
        "atexit.register(lambda: sys.stdout.write('shim: 1 package was updated\\n'))\n",
        encoding="utf-8",
    )
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(stub) + os.pathsep + old
    try:
        failures, fatal = runner.probe_imports(["json", "sys"], executable=sys.executable)
    finally:
        os.environ["PYTHONPATH"] = old
        if not old:
            os.environ.pop("PYTHONPATH", None)

    assert fatal == "", f"chatter around the verdict must not become a refusal: {fatal}"
    assert failures == []


def test_a_verdict_line_survives_output_on_both_sides_of_it() -> None:
    """The parse is line-scanning, like the child protocol two hundred lines up.

    Same defect as the test above, asserted at the parsing level so the property
    is pinned even for noise shapes no test process happens to produce -- a
    warning that lands mid-stream, a progress bar, a wrapper's own JSON. The
    sentinel line is authoritative and everything else on stdout is ignored,
    which is the convention :func:`runner.parse_child_output` has always used.
    """

    class _Proc:
        returncode = 0
        stderr = ""
        stdout = (
            "loading corporate python shim v4.2\n"
            '{"status": "this is not the verdict"}\n'
            + runner._PROBE_SENTINEL
            + '[["streamlit.testing.v1", "ModuleNotFoundError: No module named \'streamlit\'"]]\n'
            "shim: 1 package was updated\n"
        )

    failures, fatal = _probe_against(_Proc())
    assert fatal == ""
    assert failures == [
        ("streamlit.testing.v1", "ModuleNotFoundError: No module named 'streamlit'")
    ]


def test_a_truncated_verdict_is_reported_as_unreadable_not_as_success() -> None:
    """Tolerating noise must not tolerate a missing answer.

    The line-scanning parse makes it possible to see the sentinel and still have
    no usable verdict (a probe killed mid-flush). That is NOT consent: the
    question was never answered, and every paper+mode leg is the same kind of
    process. It is reported as its own message so the operator can tell "the
    probe was cut off" from "the probe never started" -- two different
    environments with two different cures.
    """

    class _Proc:
        returncode = -9
        stderr = ""
        stdout = runner._PROBE_SENTINEL + '[["streamlit.testin'

    failures, fatal = _probe_against(_Proc())
    assert failures == []
    assert "unreadable" in fatal and "truncated" in fatal


def _probe_against(proc: Any):
    """Run :func:`runner.probe_imports` against a canned ``CompletedProcess``.

    Patching ``subprocess.run`` rather than building the stdout inside a real
    child, because the shapes under test here (a torn line, a wrapper's own JSON
    on stdout) are not reproducible on demand from a real interpreter -- and the
    real-process case is covered by the ``sitecustomize`` test above.
    """

    import unittest.mock as mock

    with mock.patch.object(runner.subprocess, "run", return_value=proc):
        return runner.probe_imports(["streamlit.testing.v1"], executable=sys.executable)


def test_the_real_app_path_exists_so_this_check_is_not_vacuous() -> None:
    assert Path(driver.DEFAULT_APP_PATH).is_file()


# ---------------------------------------------------------------------------
# The message: it has to name the cure, not just the problem.
# ---------------------------------------------------------------------------
def test_the_message_names_the_problem_the_interpreter_and_the_cure() -> None:
    """The wrong-interpreter case: the one that actually happened.

    The cure has to be a command, not advice. "Use the venv" is not actionable at
    2am; ``C:\\...\\.venv\\Scripts\\python.exe scripts\\batch_run.py`` is, and the
    root cause -- ``py.exe`` ignoring the active virtualenv -- has to be named or
    the operator fixes it once and hits it again next week.
    """

    venv = runner.venv_python()
    assert venv is not None, "this project ships a .venv; the test assumes it"
    outsider = str(ROOT / "scripts" / "batch_run.py")  # a real file outside .venv/
    report = runner.check_preflight(
        probe_fn=_probe_reporting("streamlit.testing.v1"),
        executable=outsider,
    )
    text = runner.preflight_message(report)

    assert "PREFLIGHT FAILED" in text
    assert "nothing was written" in text
    assert "streamlit.testing.v1" in text
    assert "No module named 'streamlit'" in text
    assert outsider in text
    assert str(venv) in text
    assert "batch_run.py" in text
    assert "pip install -r" in text
    assert "py.exe" in text  # the actual root cause, named


def test_the_message_does_not_tell_the_venv_to_use_itself() -> None:
    """When the project interpreter IS running, the venv is what is incomplete.

    Printing "rerun with the venv" to somebody who is already in the venv is the
    kind of wrong paragraph that gets a whole message skimmed, and the py.exe
    explanation is then a false lead pointing at a cause that is not present.
    """

    venv = runner.venv_python()
    if venv is None:
        pytest.skip("no project venv on this machine")
    report = runner.check_preflight(
        probe_fn=_probe_reporting("streamlit.testing.v1"),
        executable=str(venv),
    )
    assert report.in_project_venv is True
    text = runner.preflight_message(report)
    assert "pip install -r" in text
    assert "py.exe" not in text
    assert "double-click run_overnight.bat" not in text


def test_the_message_survives_a_console_that_cannot_encode_it() -> None:
    """It is ASCII by construction; a cp1252 console must never refuse it.

    A UnicodeEncodeError while explaining an environment fault would leave the
    operator with nothing at all -- the exact failure this message is meant to
    replace. The report is built by hand with ASCII paths so the assertion is
    about this module's own wording rather than about whose machine it runs on.
    """

    report = runner.PreflightReport(
        problems=[
            runner.PreflightProblem(
                subject="streamlit.testing.v1",
                why="the child drives the real app through AppTest",
                error="ModuleNotFoundError: No module named 'streamlit'",
                cure="rerun with the venv interpreter",
            )
        ],
        interpreter=r"C:\WINDOWS\py.exe",
        venv_python=r"C:\project\.venv\Scripts\python.exe",
    )
    runner.preflight_message(report).encode("cp1252")  # raises if unencodable


def test_run_preflight_returns_a_code_nobody_else_uses() -> None:
    lines: List[str] = []
    code = runner.run_preflight(
        echo=lines.append,
        probe_fn=_probe_reporting("streamlit.testing.v1"),
    )
    assert code == runner.EXIT_PREFLIGHT
    assert code not in (0, 1, 2), "0/1 are batch verdicts, 2 is argparse and run_overnight.bat"
    assert any("PREFLIGHT FAILED" in line for line in lines)


def test_run_preflight_is_silent_and_zero_when_the_environment_is_fine() -> None:
    lines: List[str] = []
    assert runner.run_preflight(echo=lines.append) == 0
    assert [line for line in lines if "PREFLIGHT FAILED" in line] == []


def test_a_dead_stdout_cannot_turn_a_preflight_failure_into_a_crash() -> None:
    """The message is best-effort; refusing to start is not."""

    def dead(_line: str) -> None:
        raise ValueError("stdout is closed")

    assert runner.run_preflight(
        echo=dead,
        probe_fn=_probe_reporting("streamlit.testing.v1"),
    ) == runner.EXIT_PREFLIGHT


# ---------------------------------------------------------------------------
# The interpreter warning: judgement, not a block.
# ---------------------------------------------------------------------------
def test_an_interpreter_outside_the_venv_warns_but_never_blocks() -> None:
    """A custom environment (conda, a container, a global install) is legitimate.

    Blocking it would be a worse bug than the one being fixed, so this is a
    warning only -- and only when the imports it is warning about all succeeded.
    """

    if runner.venv_python() is None:
        pytest.skip("no project venv on this machine, so there is nothing to be outside of")

    # A path outside .venv/ -- the shape of a conda python or C:\WINDOWS\py.exe,
    # without needing either installed. The probe is stubbed green so what is
    # under test is the judgement, not the imports.
    outsider = ROOT / "scripts" / "batch_run.py"
    report = runner.check_preflight(probe_fn=_probe_reporting(), executable=str(outsider))

    assert report.ok is True, "a warning must never stop the night"
    assert report.warnings, "a python outside the venv is worth exactly one line"
    assert "outside this project's venv" in report.warnings[0]
    assert str(outsider) in report.warnings[0]


def test_a_failing_preflight_does_not_also_claim_the_imports_succeeded() -> None:
    """One wrong paragraph is enough to get a whole message skimmed.

    The interpreter warning is written for the clean path ("everything imported,
    but you are not in the venv"). Printed above a PREFLIGHT FAILED block it would
    contradict the block it introduces, and the HOW TO FIX section already names
    the venv in far more useful terms.
    """

    if runner.venv_python() is None:
        pytest.skip("no project venv on this machine")
    report = runner.check_preflight(
        probe_fn=_probe_reporting("streamlit.testing.v1"),
        executable=str(ROOT / "scripts" / "batch_run.py"),
    )
    assert report.ok is False
    assert report.warnings == []


def test_the_projects_own_interpreter_is_not_warned_about() -> None:
    """No warning on the happy path, or nobody reads the warnings by week two."""

    venv = runner.venv_python()
    if venv is None:
        pytest.skip("no project venv on this machine")
    report = runner.check_preflight(probe_fn=_probe_reporting(), executable=str(venv))
    assert [w for w in report.warnings if "outside this project's venv" in w] == []


def test_venv_python_points_at_a_real_interpreter_or_says_none() -> None:
    venv = runner.venv_python()
    assert venv is None or venv.is_file()


# ---------------------------------------------------------------------------
# The CLI: where it runs, and what it refuses to leave behind.
# ---------------------------------------------------------------------------
def test_the_cli_runs_the_preflight_before_the_batch(cli: Any, monkeypatch: Any) -> None:
    order: List[str] = []
    monkeypatch.setattr(runner, "run_preflight", lambda **_k: order.append("preflight") or 0)
    monkeypatch.setattr(runner, "run_batch", lambda **_k: order.append("batch") or 0)
    assert cli.main([]) == 0
    assert order == ["preflight", "batch"]


def test_a_failed_preflight_stops_the_cli_with_its_own_code(cli: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(runner, "run_preflight", lambda **_k: runner.EXIT_PREFLIGHT)
    monkeypatch.setattr(
        runner, "run_batch", lambda **_k: pytest.fail("a failed preflight must not start the batch")
    )
    assert cli.main([]) == runner.EXIT_PREFLIGHT


def test_a_failed_preflight_writes_absolutely_nothing(cli: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """No run directory, no plan.json, no manifest -- not even an empty folder.

    ``runs/2026-07-27_2135`` is still on disk with 28 paper folders, a cache
    snapshot and a 56-row manifest, and at a glance it is indistinguishable from
    a night that really ran. A failed preflight must leave the tree exactly as it
    was before the command was typed.
    """

    out_dir = tmp_path / "runs"
    # Patched at the probe, not at run_preflight: everything above the probe --
    # main(), run_preflight, check_preflight, the message, the exit code -- is
    # then the real code path, which is the part that has to leave no trace.
    monkeypatch.setattr(runner, "probe_imports", _probe_reporting("streamlit.testing.v1"))
    monkeypatch.setattr(
        runner, "run_batch", lambda **_k: pytest.fail("a failed preflight must not start the batch")
    )
    assert cli.main(["--out", str(out_dir), "--fresh"]) == runner.EXIT_PREFLIGHT
    assert not out_dir.exists(), f"the preflight created {out_dir}"
    assert list(tmp_path.iterdir()) == []


def test_status_stays_pollable_and_never_runs_the_preflight(cli: Any, monkeypatch: Any) -> None:
    """--status is documented as always exiting 0 so a second terminal can poll it.

    An environment check has no business blocking a read of yesterday's manifest,
    and 0.6s of imports per poll would be paid for nothing.
    """

    monkeypatch.setattr(runner, "run_preflight", lambda **_k: pytest.fail("--status must not preflight"))
    monkeypatch.setattr(runner, "print_status", lambda out: 0)
    assert cli.main(["--status"]) == 0


def test_the_child_does_not_repeat_the_preflight(cli: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """The parent already vouched, once, for the interpreter it spawns children with.

    Re-checking in the child would spend 0.6s per leg re-proving a settled fact,
    and a disagreement would break the one-row-per-pair contract the manifest
    depends on.
    """

    monkeypatch.setattr(runner, "run_preflight", lambda **_k: pytest.fail("--single must not preflight"))
    monkeypatch.setattr(
        runner,
        "run_single",
        lambda rd, slug, mode, timeout=None: {"status": "pass", "slug": slug, "mode": mode},
    )
    code = cli.main(
        ["--single", "--run-dir", str(tmp_path), "--slug", "PMC1__alpha", "--mode", "strict"]
    )
    assert code == 0


def test_the_exit_code_is_documented_where_the_others_are(cli: Any) -> None:
    """A distinct code is useless if nobody reading ERRORLEVEL knows what it means."""

    doc = cli.__doc__ or ""
    assert "Exit codes" in doc
    assert str(runner.EXIT_PREFLIGHT) in doc
    assert "PREFLIGHT" in doc
    assert "preflight" in cli.build_parser().epilog.lower()


def test_the_prose_docs_the_cli_points_at_really_document_the_code() -> None:
    """The CLI docstring cites ``docs/batch_runner.md`` section 2d by name.

    Raised in review on 2026-07-28: that citation was written before the section
    existed, so a reader following it landed on a list of 0/1/2 and concluded the
    3 they were looking at was undefined. A docstring that points at
    documentation which does not say what it claims is worse than no pointer at
    all -- it costs the reader the lookup and then tells them nothing. Asserted
    rather than merely fixed so the two cannot drift apart again.
    """

    text = (ROOT / "docs" / "batch_runner.md").read_text(encoding="utf-8")
    section = text.split("### 2d. Exit codes", 1)
    assert len(section) == 2, "section 2d is what the CLI docstring points the reader at"
    body = section[1].split("---", 1)[0]
    assert f"`{runner.EXIT_PREFLIGHT}`" in body, "the preflight code must be listed with the others"
    assert "preflight" in body.lower()
    # The single fact an operator most needs from this row: there is no new run
    # directory, so nothing in runs/ describes what just happened.
    assert "no run directory" in body.lower() or "no run folder" in body.lower()


def test_the_launcher_does_not_send_a_refused_night_to_last_nights_summary() -> None:
    """``run_overnight.bat`` is the last thing on screen after a double-click.

    Raised in review on 2026-07-28. The launcher printed one message for every
    non-zero code: "exit code N: something did not pass. Read SUMMARY.txt and
    failures_by_code.txt in the newest runs\\ folder." After a preflight refusal
    there IS no new run folder, so "the newest" is the PREVIOUS night's -- and the
    operator is invited to debug a pipeline that never ran, from a summary that
    describes a different night. That is precisely the "a failed night looks like
    a night that happened" confusion the preflight exists to prevent, arriving via
    the one surface a double-click user actually reads.

    Asserted against the .bat source (structurally, on the branch for
    ``EXIT_PREFLIGHT``) rather than by executing it, because executing it starts a
    real ten-hour batch.
    """

    text = (ROOT / "run_overnight.bat").read_text(encoding="utf-8")
    marker = f'if "%EXITCODE%"=="{runner.EXIT_PREFLIGHT}" ('
    assert marker in text, "the preflight code needs its own branch in the launcher"

    branch = text.split(marker, 1)[1].split(") else", 1)[0].lower()
    assert "no run folder" in branch or "no run directory" in branch
    assert "summary.txt to read" in branch, "it must say there is no summary, not point at one"
    assert "read summary.txt and failures_by_code.txt" not in branch, (
        "the generic 'read the newest runs\\ folder' advice is exactly what is wrong here"
    )


# ---------------------------------------------------------------------------
# End to end, in a real process. Exit codes and "wrote nothing" are properties
# of a process; an in-process main() call cannot prove either.
# ---------------------------------------------------------------------------
def test_a_real_run_with_no_streamlit_exits_three_and_leaves_no_run_directory(tmp_path: Path) -> None:
    """The 2026-07-27_2135 scenario, reproduced and then refused.

    ``streamlit`` is shadowed by a stub package on PYTHONPATH that raises the
    exact error the manifest recorded 56 times. PYTHONPATH precedes site-packages
    on ``sys.path``, so the real installation is invisible to this child without
    anything being uninstalled. What must happen: exit 3, a message naming the
    module and the venv interpreter, and ``--out`` never created -- no fetch, no
    plan.json, no manifest, in well under the 43 seconds the real fetch took.
    """

    stub_root = tmp_path / "poison"
    (stub_root / "streamlit").mkdir(parents=True)
    (stub_root / "streamlit" / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'streamlit'\")\n", encoding="utf-8"
    )
    # An empty work list as a seatbelt, NOT as part of what is being tested: if
    # this assertion's premise ever breaks and the preflight lets the run through,
    # the test must fail on the returncode rather than quietly start fetching
    # papers from PubMed and driving the real pipeline inside the test suite.
    topics = tmp_path / "empty_topics.txt"
    topics.write_text("# deliberately empty\n", encoding="utf-8")

    out_dir = tmp_path / "runs"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--out", str(out_dir),
            "--fresh",
            "--topics", str(topics),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")

    assert proc.returncode == runner.EXIT_PREFLIGHT, combined
    assert "PREFLIGHT FAILED" in combined
    assert "streamlit.testing.v1" in combined
    assert not out_dir.exists(), "a refused run must not leave a run directory behind"
    # And nothing was fetched: the log line the real run printed at 21:35:06 is
    # the first thing run_batch does, and it must never have been reached.
    assert "fetching papers" not in combined
    assert "STARTING" not in combined


def test_a_real_run_with_a_healthy_environment_reaches_the_batch(tmp_path: Path) -> None:
    """The other half: the preflight must not become a new way to fail.

    The batch itself is stubbed out through ``--topics`` pointing at an empty
    file, so this fetches nothing and exits 1 ("NOTHING RAN") -- what is proved is
    that the preflight let it through and a run directory was created.
    """

    topics = tmp_path / "empty_topics.txt"
    topics.write_text("# no topics at all\n", encoding="utf-8")
    out_dir = tmp_path / "runs"

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--out", str(out_dir),
            "--fresh",
            "--topics", str(topics),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")

    assert "PREFLIGHT FAILED" not in combined, combined
    assert proc.returncode != runner.EXIT_PREFLIGHT
    assert out_dir.is_dir(), combined
    run_dirs = runner.list_run_dirs(out_dir)
    assert run_dirs, "the batch should have created a run directory"
    plan: Dict[str, Any] = json.loads((run_dirs[0] / runner.PLAN_NAME).read_text(encoding="utf-8"))
    assert plan.get("papers") == []
