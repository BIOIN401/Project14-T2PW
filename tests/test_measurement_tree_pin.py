"""NEW ACCEPTANCE TESTS for the measurement-tree source pin (H-010).

**G9 label: NEW CAPABILITY, explicitly.** No repository mechanism has ever refused a
mis-pinned run -- ``c045_pinned_pytest.py`` printed and proceeded, ``chunk_d_gate.py``
prints and proceeds, ``bounded_run.py`` records no environment at all. There is no prior
behaviour to restore, so **no base failure is offered or expected** for these five cases;
a manufactured one would rest on ``tree_pin`` not existing at the base SHA, which is
symbol absence and is not proof. (The *separate* CWD/``scripts`` defect in
``c045_pinned_pytest.py`` **is** a pre-existing regression with a real behavioural base
failure -- proved outside this file, by running the base's own launcher at the base SHA.)

**The real venv is never touched.** Each case builds a synthetic checkout under the run's
own basetemp -- ``pyproject.toml``, ``src/t2pw/__init__.py`` with a unique marker,
``scripts/``, ``tests/``, and a copy of ``tree_pin.py`` + ``pinned_pytest.py`` so that
``expected_tree()`` resolves to it. The editable ``.pth`` is simulated by a
``sitecustomize.py`` on the child's ``PYTHONPATH``: it drops every ``src`` entry and
*appends* a decoy, and a tail append is exactly ``.pth`` semantics.

**Guard-removal proof.** Every case also runs ``legacy.py``: the pre-H-010
``c045_pinned_pytest.py`` body verbatim -- import ``t2pw``, print it, hand off to
``pytest.main``. That is the guard genuinely removed rather than mocked, and each case
asserts the legacy launcher sails straight through the same input.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "pwml_recovery_sprint" / "evidence"
PY = sys.executable
MARKER = "T2PW_MEASUREMENT_TREE_REFUSED"

_SITECUSTOMIZE = """import os, sys
if os.environ.get("PIN_TEST_DROP_ALL_SRC"):
    sys.path[:] = [p for p in sys.path if os.path.basename(os.path.normpath(p)) != "src"]
_ins = os.environ.get("PIN_TEST_INSERT_SRC")
if _ins:
    sys.path.insert(0, _ins)
_app = os.environ.get("PIN_TEST_APPEND_SRC")
if _app:
    sys.path.append(_app)
"""

# The pre-H-010 c045_pinned_pytest.py, behaviour for behaviour. The guard removed.
_LEGACY = """import sys

import pytest

import t2pw

print(f"T2PW: {t2pw.__file__}", flush=True)
raise SystemExit(pytest.main(sys.argv[1:]))
"""

# A consumer that calls the guard WITHOUT the launcher's cwd/sys.path repair.
_BARE_ENFORCE = """import os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tree_pin

tree_pin.enforce(
    expected=Path(os.environ["PIN_TEST_EXPECT"]),
    require_scripts=True,
    verdict_path=os.environ.get("PIN_TEST_VERDICT"),
)
print("BARE_ENFORCE_OK", flush=True)
"""


def _make_tree(root: Path, marker: str, *, launcher: bool = True) -> Path:
    """A synthetic checkout that ``expected_tree()`` will recognise."""

    (root / "src" / "t2pw").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'synthetic'\n", encoding="utf-8")
    (root / "src" / "t2pw" / "__init__.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
    (root / "scripts" / "marker.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
    (root / "tests" / "test_plain.py").write_text(
        "def test_plain():\n    assert True\n", encoding="utf-8")
    (root / "tests" / "test_uses_scripts.py").write_text(
        "from scripts import marker\n\n\ndef test_marker():\n    assert marker.MARKER\n",
        encoding="utf-8")
    if launcher:
        (root / "ev").mkdir(exist_ok=True)
        for name in ("tree_pin.py", "pinned_pytest.py"):
            shutil.copy2(EVIDENCE / name, root / "ev" / name)
        (root / "ev" / "legacy.py").write_text(_LEGACY, encoding="utf-8")
        (root / "ev" / "bare_enforce.py").write_text(_BARE_ENFORCE, encoding="utf-8")
    return root


def _sitedir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    return root


def _run(script: Path, args: Sequence[str], *, cwd: Path,
         env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    child = os.environ.copy()
    child.pop("PYTHONPATH", None)
    child["T2PW_OFFLINE_CURATOR"] = "1"
    child.update(env or {})
    return subprocess.run([PY, "-u", str(script), *args], cwd=str(cwd), env=child,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)


def _out(proc: subprocess.CompletedProcess) -> str:
    return (proc.stdout or "") + (proc.stderr or "")


def _verdict(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _under(child: object, parent: Path) -> bool:
    return Path(str(child)).resolve().is_relative_to(parent.resolve())


def _assert_refused(proc: subprocess.CompletedProcess, verdict: Path,
                    codes: Sequence[str]) -> Dict:
    text = _out(proc)
    assert proc.returncode == 98, f"expected exit 98, got {proc.returncode}\n{text}"
    assert MARKER in text, text
    assert "passed" not in text and "collected" not in text, text  # nothing was counted
    assert verdict.is_file(), f"no verdict artifact at {verdict}\n{text}"
    data = _verdict(verdict)
    assert data["refused"] is True
    for code in codes:
        assert code in data["violations"], f"{code} not in {data['violations']}"
    return data


def test_missing_or_wrong_pythonpath_is_refused_before_collection(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path / "measured", "MEASURED")
    decoy = _make_tree(tmp_path / "decoy", "DECOY", launcher=False)
    launcher = tree / "ev" / "pinned_pytest.py"
    bt = tmp_path / "bt"
    bt.mkdir()

    # (a) PYTHONPATH absent: the real editable .pth supplies the PRIMARY checkout's t2pw.
    #     F-003 itself, live -- and refused.
    v1 = tmp_path / "v1.json"
    proc = _run(launcher, [f"--pin-verdict={v1}", "-q", f"--basetemp={bt / 'a'}",
                           "tests/test_plain.py"], cwd=tree)
    data = _assert_refused(proc, v1, ["T2PW_FROM_WRONG_TREE"])
    assert not _under(data["t2pw_file"], tree)
    assert data["pythonpath"] in (None, "")

    # (b) PYTHONPATH present but pointing at the wrong tree.
    v2 = tmp_path / "v2.json"
    proc = _run(launcher, [f"--pin-verdict={v2}", "-q", f"--basetemp={bt / 'b'}",
                           "tests/test_plain.py"],
                cwd=tree, env={"PYTHONPATH": str(decoy / "src")})
    data = _assert_refused(proc, v2, ["T2PW_FROM_WRONG_TREE"])
    assert _under(data["t2pw_file"], decoy)

    # GUARD REMOVED: the same input counts a test against the wrong tree, silently.
    legacy = _run(tree / "ev" / "legacy.py", ["-q", f"--basetemp={bt / 'c'}", "tests/test_plain.py"],
                  cwd=tree, env={"PYTHONPATH": str(decoy / "src")})
    assert legacy.returncode == 0, _out(legacy)
    assert "1 passed" in _out(legacy)
    assert str(decoy) in _out(legacy)


def test_wrong_cwd_is_refused_and_scripts_is_named(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path / "measured", "MEASURED")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    bt = tmp_path / "bt"
    bt.mkdir()
    pinned = {"PYTHONPATH": str(tree / "src")}

    # (a) The literal C-050g shape: a third-directory cwd and a selection needing
    #     `scripts`. The launcher REPAIRS cwd/sys.path[0] to the measured tree -- what
    #     `python -m pytest` from the root does -- then VERIFIES `scripts` came from there.
    v1 = tmp_path / "v1.json"
    proc = _run(tree / "ev" / "pinned_pytest.py",
                [f"--pin-verdict={v1}", "--require-scripts", "-q", f"--basetemp={bt / 'a'}",
                 "tests/test_uses_scripts.py"], cwd=elsewhere, env=pinned)
    assert proc.returncode == 0, _out(proc)
    assert "1 passed" in _out(proc)
    data = _verdict(v1)
    assert data["violations"] == []
    assert _under(data["cwd"], tree)
    assert Path(str(data["cwd_at_entry"])).resolve() == elsewhere.resolve()
    assert all(_under(loc, tree) for loc in data["scripts_locations"])

    # GUARD REMOVED: same input, the reproduced ModuleNotFoundError.
    legacy = _run(tree / "ev" / "legacy.py",
                  ["-q", f"--basetemp={bt / 'b'}", str(tree / "tests" / "test_uses_scripts.py")],
                  cwd=elsewhere, env=pinned)
    assert legacy.returncode != 0
    assert "No module named 'scripts'" in _out(legacy)

    # (b) A consumer that does NOT repair -- the guard alone must refuse. b1 is the
    #     C-050g environment: the root is on sys.path nowhere, so the cwd is wrong AND
    #     `scripts` cannot be found at all.
    bare = tree / "ev" / "bare_enforce.py"
    rooted = {"PYTHONPATH": os.pathsep.join([str(tree / "src"), str(tree)])}
    v2 = tmp_path / "v2.json"
    proc = _run(bare, [], cwd=elsewhere,
                env=dict(pinned, PIN_TEST_EXPECT=str(tree), PIN_TEST_VERDICT=str(v2)))
    _assert_refused(proc, v2, ["CWD_NOT_EXPECTED_ROOT", "SCRIPTS_UNIMPORTABLE"])
    assert "BARE_ENFORCE_OK" not in _out(proc)

    # b2 vs b3: identical environment, ONLY the cwd differs.
    v3 = tmp_path / "v3.json"
    proc = _run(bare, [], cwd=elsewhere,
                env=dict(rooted, PIN_TEST_EXPECT=str(tree), PIN_TEST_VERDICT=str(v3)))
    assert _assert_refused(proc, v3, ["CWD_NOT_EXPECTED_ROOT"])["violations"] == [
        "CWD_NOT_EXPECTED_ROOT"]

    v4 = tmp_path / "v4.json"
    ok = _run(bare, [], cwd=tree,
              env=dict(rooted, PIN_TEST_EXPECT=str(tree), PIN_TEST_VERDICT=str(v4)))
    assert ok.returncode == 0, _out(ok)
    assert "BARE_ENFORCE_OK" in _out(ok)
    assert _verdict(v4)["violations"] == []


def test_editable_pth_pointing_at_another_checkout_is_refused(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path / "measured", "MEASURED")
    decoy = _make_tree(tmp_path / "primary_decoy", "PRIMARY_DECOY", launcher=False)
    site = _sitedir(tmp_path / "site")
    bt = tmp_path / "bt"
    bt.mkdir()

    # No PYTHONPATH pin at all; a tail append is exactly what a path-line .pth does.
    env = {
        "PYTHONPATH": str(site),
        "PIN_TEST_DROP_ALL_SRC": "1",
        "PIN_TEST_APPEND_SRC": str(decoy / "src"),
    }
    v1 = tmp_path / "v1.json"
    proc = _run(tree / "ev" / "pinned_pytest.py",
                [f"--pin-verdict={v1}", "-q", f"--basetemp={bt / 'a'}", "tests/test_plain.py"],
                cwd=tree, env=env)
    data = _assert_refused(proc, v1, ["T2PW_FROM_WRONG_TREE"])
    # The guard reports the RESOLUTION, never the intention.
    assert _under(data["t2pw_file"], decoy)
    assert any(_under(entry, decoy) for entry in data["foreign_src_entries"])

    legacy = _run(tree / "ev" / "legacy.py",
                  ["-q", f"--basetemp={bt / 'b'}", "tests/test_plain.py"], cwd=tree, env=env)
    assert legacy.returncode == 0, _out(legacy)
    assert "1 passed" in _out(legacy)


def test_base_tree_measured_with_tip_on_pythonpath_is_refused_both_ways(tmp_path: Path) -> None:
    base = _make_tree(tmp_path / "base", "BASE")
    tip = _make_tree(tmp_path / "tip", "TIP")
    site = _sitedir(tmp_path / "site")
    bt = tmp_path / "bt"
    bt.mkdir()
    sel = ["-q", "tests/test_plain.py"]

    # (a) base under measurement, tip on PYTHONPATH.
    va = tmp_path / "a.json"
    proc = _run(base / "ev" / "pinned_pytest.py",
                [f"--pin-verdict={va}", f"--basetemp={bt / 'a'}", *sel],
                cwd=base, env={"PYTHONPATH": str(tip / "src")})
    da = _assert_refused(proc, va, ["T2PW_FROM_WRONG_TREE"])
    assert _under(da["expected_tree"], base) and _under(da["t2pw_file"], tip)

    # (b) the mirror: tip under measurement, base on PYTHONPATH.
    vb = tmp_path / "b.json"
    proc = _run(tip / "ev" / "pinned_pytest.py",
                [f"--pin-verdict={vb}", f"--basetemp={bt / 'b'}", *sel],
                cwd=tip, env={"PYTHONPATH": str(base / "src")})
    db = _assert_refused(proc, vb, ["T2PW_FROM_WRONG_TREE"])
    assert _under(db["expected_tree"], tip) and _under(db["t2pw_file"], base)

    # Different trees per arm: a guard echoing its own input could not produce this.
    assert da["t2pw_file"] != db["t2pw_file"]
    assert da["expected_tree"] != db["expected_tree"]

    # (c) The probe-G defeat: PYTHONPATH is pinned CORRECTLY at base and an in-process
    #     sys.path.insert(0, tip/src) -- add_src_to_path() in a worktree -- beats it. A
    #     guard comparing PYTHONPATH to `expected` would pass this. This one must not.
    vc = tmp_path / "c.json"
    proc = _run(tip / "ev" / "pinned_pytest.py",
                [f"--expect-tree={base}", f"--pin-verdict={vc}", f"--basetemp={bt / 'c'}", *sel],
                cwd=base,
                env={"PYTHONPATH": os.pathsep.join([str(site), str(base / "src")]),
                     "PIN_TEST_INSERT_SRC": str(tip / "src")})
    dc = _assert_refused(proc, vc, ["T2PW_FROM_WRONG_TREE"])
    assert _under(dc["expected_tree"], base)
    assert str(base / "src") in str(dc["pythonpath"])  # the pin was correct...
    assert _under(dc["t2pw_file"], tip)                # ...and was defeated anyway.

    legacy = _run(base / "ev" / "legacy.py", [f"--basetemp={bt / 'd'}", *sel],
                  cwd=base, env={"PYTHONPATH": str(tip / "src")})
    assert legacy.returncode == 0, _out(legacy)
    assert "1 passed" in _out(legacy)


def test_correct_tree_passes_and_wrong_tree_fails_visibly_before_any_count(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path / "measured", "MEASURED")
    decoy = _make_tree(tmp_path / "decoy", "DECOY", launcher=False)
    launcher = tree / "ev" / "pinned_pytest.py"
    bt = tmp_path / "bt"
    bt.mkdir()

    # Correct tree: t2pw AND scripts both from the tree under measurement.
    ok_verdict = tmp_path / "ok.json"
    ok = _run(launcher, [f"--pin-verdict={ok_verdict}", "--require-scripts", "-q",
                         f"--basetemp={bt / 'a'}", "tests/test_uses_scripts.py"],
              cwd=tree, env={"PYTHONPATH": str(tree / "src")})
    assert ok.returncode == 0, _out(ok)
    announced = re.search(r"^T2PW: (.+)$", _out(ok), re.M)
    assert announced is not None, _out(ok)
    assert _under(announced.group(1).strip(), tree / "src" / "t2pw")
    assert "1 passed" in _out(ok)
    assert _verdict(ok_verdict)["violations"] == []

    # Wrong tree, identical selection.
    bad_verdict = tmp_path / "bad.json"
    bad = _run(launcher, [f"--pin-verdict={bad_verdict}", "--require-scripts", "-q",
                          f"--basetemp={bt / 'b'}", "tests/test_uses_scripts.py"],
               cwd=tree, env={"PYTHONPATH": str(decoy / "src")})
    text = _out(bad)
    _assert_refused(bad, bad_verdict, ["T2PW_FROM_WRONG_TREE"])

    # "Before any test is counted", machine-checkable: no count token after the marker.
    offset = text.index(MARKER)
    for token in ("collected", "passed", "failed", "error"):
        assert token not in text[offset:], f"{token!r} appears after the refusal:\n{text}"

    # And the banner can never be misread as a pytest summary by chunk_d_gate.
    assert re.search(r"\bin\s+\d+(?:\.\d+)?\s*s\b", text) is None, text

    legacy = _run(tree / "ev" / "legacy.py", ["-q", f"--basetemp={bt / 'c'}", "tests/test_plain.py"],
                  cwd=tree, env={"PYTHONPATH": str(decoy / "src")})
    assert legacy.returncode == 0 and "1 passed" in _out(legacy)
