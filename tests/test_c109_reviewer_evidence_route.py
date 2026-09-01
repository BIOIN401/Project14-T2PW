"""C-109 -- acceptance test for a NEW capability: the reviewer-evidence route check.

**G9 label: NEW ACCEPTANCE TEST for a NEW capability.** `reviewer_evidence_route.py` did
not exist before C-109, so there is no pre-existing observable behaviour to preserve and
**no base failure is claimed or fabricated**. This file is the acceptance test the G9
standard asks for in that case.

The point of the suite is the negative half. **A check that cannot fail is not a check**,
so the failing cases are asserted on their exact exit code and on the specific item named
in the failure output -- not merely on "non-zero".

The three failure modes that matter, all proven here:

* evidence that is simply absent from integration (``test_missing_evidence_fails``);
* evidence whose FILENAME is in integration but whose BYTES differ -- the one a filename
  check calls green (``test_same_name_different_bytes_is_not_reachable``);
* nothing enumerated at all, e.g. a mistyped task id, which must not pass silently
  (``test_nothing_enumerated_does_not_pass_silently``).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "reviewer_evidence_route.py"
EVIDENCE_REL = "docs/pwml_recovery_sprint/evidence"

TASK = "REV-109"
STEM = "rev109"


def _load_module():
    spec = importlib.util.spec_from_file_location("c109_reviewer_evidence_route", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def route():
    assert SCRIPT.is_file(), f"the check under test is missing: {SCRIPT}"
    return _load_module()


# ---------------------------------------------------------------- fixtures


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_worktree(root: Path) -> dict[str, str]:
    """A reviewer worktree holding two G11 reports and two probe files."""
    contents = {
        f"{EVIDENCE_REL}/g11/{TASK}/01-{STEM}-focused.json": json.dumps(
            {"label": f"{STEM}-focused", "final_surviving_count": 0, "cleanup": "success"}
        ),
        f"{EVIDENCE_REL}/g11/{TASK}/02-{STEM}-smoke.json": json.dumps(
            {"label": f"{STEM}-smoke", "final_surviving_count": 0, "cleanup": "success"}
        ),
        f"{EVIDENCE_REL}/{STEM}_boundary_probe.py": "print('the probe source')\n",
        f"{EVIDENCE_REL}/{STEM}_boundary_probe.log": "the probe output, 3 findings\n",
    }
    for rel, text in contents.items():
        _write(root, rel, text)
    # Noise the enumeration must ignore: another task's evidence, and a probe whose
    # stem merely CONTAINS ours.
    _write(root, f"{EVIDENCE_REL}/g11/REV-108/01-rev108-focused.json", "{}")
    _write(root, f"{EVIDENCE_REL}/c107_unrelated.log", "not this task\n")
    return contents


def _run(module, **kwargs) -> tuple[int, dict]:
    argv = []
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    code = module.run(argv)
    report_path = kwargs.get("json")
    report = json.loads(Path(report_path).read_text(encoding="utf-8")) if report_path else {}
    return code, report


# ---------------------------------------------------------------- enumeration


def test_enumerates_g11_reports_and_probes_and_nothing_else(route, tmp_path):
    worktree = tmp_path / "wt"
    contents = _make_worktree(worktree)

    items = route.enumerate_evidence(worktree, TASK)
    found = {p.relative_to(worktree).as_posix() for p in (i["path"] for i in items)}

    assert found == set(contents), found
    kinds = {i["kind"] for i in items}
    assert kinds == {"g11_report", "probe"}


def test_task_stem(route):
    assert route.task_stem("REV-105") == "rev105"
    assert route.task_stem("C-107") == "c107"
    assert route.task_stem("INIT-001") == "init001"


# ---------------------------------------------- reachable case: it PASSES


def test_reachable_evidence_passes(route, tmp_path):
    worktree = tmp_path / "wt"
    integration = tmp_path / "integration"
    contents = _make_worktree(worktree)
    for rel, text in contents.items():
        _write(integration, rel, text)

    code, report = _run(
        route,
        task=TASK,
        worktree=worktree,
        integration_dir=integration,
        json=tmp_path / "ok.json",
    )

    assert code == route.EXIT_OK
    assert report["enumerated"] == 4
    assert report["unreachable"] == 0
    assert report["exit_reason"] == "all_reachable"
    assert all(i["verdict"] == route.REACHABLE for i in report["items"])


def test_reachable_under_a_different_path_still_counts(route, tmp_path):
    """Content is what must survive. A renamed but byte-identical file is reachable."""
    worktree = tmp_path / "wt"
    integration = tmp_path / "integration"
    contents = _make_worktree(worktree)
    for rel, text in contents.items():
        _write(integration, "archive/" + rel, text)

    code, report = _run(
        route,
        task=TASK,
        worktree=worktree,
        integration_dir=integration,
        json=tmp_path / "renamed.json",
    )

    assert code == route.EXIT_OK
    for item in report["items"]:
        assert item["verdict"] == route.REACHABLE
        assert item["found_at"] and item["found_at"][0].startswith("archive/")


# ------------------------------------------- THE CHECK MUST BE ABLE TO FAIL


def test_missing_evidence_fails(route, tmp_path):
    """One probe log is never committed. The gate must go red and name it."""
    worktree = tmp_path / "wt"
    integration = tmp_path / "integration"
    contents = _make_worktree(worktree)
    withheld = f"{EVIDENCE_REL}/{STEM}_boundary_probe.log"
    for rel, text in contents.items():
        if rel == withheld:
            continue
        _write(integration, rel, text)

    code, report = _run(
        route,
        task=TASK,
        worktree=worktree,
        integration_dir=integration,
        json=tmp_path / "missing.json",
    )

    assert code == route.EXIT_UNREACHABLE, "a check that cannot fail is not a check"
    assert report["exit_reason"] == "unreachable_evidence"
    assert report["unreachable"] == 1
    bad = [i for i in report["items"] if i["verdict"] != route.REACHABLE]
    assert [i["worktree_path"] for i in bad] == [withheld]
    assert bad[0]["verdict"] == route.UNREACHABLE_ABSENT
    assert bad[0]["found_at"] == []


def test_all_evidence_missing_fails_and_lists_every_item(route, tmp_path):
    worktree = tmp_path / "wt"
    integration = tmp_path / "integration"
    _make_worktree(worktree)
    _write(integration, "README.md", "an integration tree with no evidence in it\n")

    code, report = _run(
        route,
        task=TASK,
        worktree=worktree,
        integration_dir=integration,
        json=tmp_path / "all_missing.json",
    )

    assert code == route.EXIT_UNREACHABLE
    assert report["unreachable"] == 4
    assert report["reachable"] == 0


def test_same_name_different_bytes_is_not_reachable(route, tmp_path):
    """THE point of the check: identical filename, different content, still lost.

    A filename-based check reports this green and the evidence is gone anyway.
    """
    worktree = tmp_path / "wt"
    integration = tmp_path / "integration"
    contents = _make_worktree(worktree)
    tampered = f"{EVIDENCE_REL}/g11/{TASK}/02-{STEM}-smoke.json"
    for rel, text in contents.items():
        _write(integration, rel, text if rel != tampered else text.replace("0", "7"))

    code, report = _run(
        route,
        task=TASK,
        worktree=worktree,
        integration_dir=integration,
        json=tmp_path / "tampered.json",
    )

    assert code == route.EXIT_UNREACHABLE
    bad = [i for i in report["items"] if i["verdict"] != route.REACHABLE]
    assert [i["worktree_path"] for i in bad] == [tampered]
    assert bad[0]["verdict"] == route.UNREACHABLE_DIFFERS, (
        "a same-named file with different bytes must be reported as its own class, "
        "not as absent -- it is the failure mode that looks green"
    )


def test_nothing_enumerated_does_not_pass_silently(route, tmp_path):
    """A mistyped task id must not read as a clean gate -- that is the F-154 class."""
    worktree = tmp_path / "wt"
    integration = tmp_path / "integration"
    _make_worktree(worktree)
    _write(integration, "README.md", "x\n")

    code, report = _run(
        route,
        task="REV-999",
        worktree=worktree,
        integration_dir=integration,
        json=tmp_path / "empty.json",
    )

    assert code == route.EXIT_NOTHING_ENUMERATED
    assert report["enumerated"] == 0
    assert report["exit_reason"] == "nothing_enumerated"

    code_allowed, _ = _run(
        route,
        task="REV-999",
        worktree=worktree,
        integration_dir=integration,
        allow_empty=True,
        json=tmp_path / "empty_allowed.json",
    )
    assert code_allowed == route.EXIT_OK


def test_bad_worktree_is_a_usage_error_not_a_verdict(route, tmp_path):
    code = route.run(
        [
            "--task",
            TASK,
            "--worktree",
            str(tmp_path / "does-not-exist"),
            "--integration-dir",
            str(tmp_path),
        ]
    )
    assert code == route.EXIT_USAGE


# ------------------------------------------------- the production git path


def _git_available() -> bool:
    return shutil.which("git") is not None


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
def test_git_ref_mode_passes_and_fails(route, tmp_path):
    """The mode the merge gate actually runs in: content looked up in a git ref."""
    git = shutil.which("git")
    base = ["-c", "user.email=c109@t2pw.local", "-c", "user.name=C-109"]

    integration = tmp_path / "int_repo"
    integration.mkdir()
    subprocess.run([git, "init", "-q", "-b", "integration", str(integration)], check=True)

    worktree = tmp_path / "wt_repo"
    worktree.mkdir()
    subprocess.run([git, "init", "-q", "-b", "work", str(worktree)], check=True)
    contents = _make_worktree(worktree)
    subprocess.run([git, "-C", str(worktree), "add", "-A"], check=True)
    subprocess.run(
        [git, "-C", str(worktree)] + base + ["commit", "-q", "-m", "reviewer evidence"],
        check=True,
    )

    # Integration carries only three of the four items.
    withheld = f"{EVIDENCE_REL}/{STEM}_boundary_probe.log"
    for rel, text in contents.items():
        if rel != withheld:
            _write(integration, rel, text)
    subprocess.run([git, "-C", str(integration), "add", "-A"], check=True)
    subprocess.run(
        [git, "-C", str(integration)] + base + ["commit", "-q", "-m", "partial"],
        check=True,
    )

    code, report = _run(
        route,
        task=TASK,
        worktree=worktree,
        integration_repo=integration,
        integration_ref="integration",
        json=tmp_path / "git_partial.json",
    )
    assert code == route.EXIT_UNREACHABLE
    assert [i["worktree_path"] for i in report["items"] if i["verdict"] != route.REACHABLE] == [
        withheld
    ]

    # Commit the missing probe log; the same gate must now pass.
    _write(integration, withheld, contents[withheld])
    subprocess.run([git, "-C", str(integration), "add", "-A"], check=True)
    subprocess.run(
        [git, "-C", str(integration)] + base + ["commit", "-q", "-m", "complete"],
        check=True,
    )
    code_after, report_after = _run(
        route,
        task=TASK,
        worktree=worktree,
        integration_repo=integration,
        integration_ref="integration",
        json=tmp_path / "git_complete.json",
    )
    assert code_after == route.EXIT_OK
    assert report_after["enumerated"] == 4
    assert report_after["unreachable"] == 0


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
def test_bad_ref_is_a_usage_error(route, tmp_path):
    git = shutil.which("git")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([git, "init", "-q", str(repo)], check=True)
    worktree = tmp_path / "wt"
    _make_worktree(worktree)
    code = route.run(
        [
            "--task",
            TASK,
            "--worktree",
            str(worktree),
            "--integration-repo",
            str(repo),
            "--integration-ref",
            "no/such/ref",
        ]
    )
    assert code == route.EXIT_USAGE


# ----------------------------------------------------- invoked as a script


def test_runs_as_a_subprocess_with_a_nonzero_exit(tmp_path):
    """It must gate a merge from a shell, so the exit code has to survive argv."""
    worktree = tmp_path / "wt"
    integration = tmp_path / "integration"
    _make_worktree(worktree)
    _write(integration, "README.md", "x\n")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            TASK,
            "--worktree",
            str(worktree),
            "--integration-dir",
            str(integration),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "UNREACHABLE" in proc.stderr
    assert f"{STEM}_boundary_probe.log" in proc.stderr


def test_scope_discipline_it_does_not_judge_evidence_quality(route, tmp_path):
    """Reachability only. A report saying FAIL is still reachable evidence."""
    worktree = tmp_path / "wt"
    integration = tmp_path / "integration"
    _make_worktree(worktree)
    failing = f"{EVIDENCE_REL}/g11/{TASK}/03-{STEM}-red.json"
    text = json.dumps({"label": "red", "final_surviving_count": 9, "cleanup": "failure"})
    _write(worktree, failing, text)
    for path in (worktree / EVIDENCE_REL).rglob("*"):
        if path.is_file():
            _write(integration, path.relative_to(worktree).as_posix(), path.read_text(encoding="utf-8"))

    code, report = _run(
        route,
        task=TASK,
        worktree=worktree,
        integration_dir=integration,
        json=tmp_path / "quality.json",
    )
    assert code == route.EXIT_OK
    assert report["enumerated"] == 5
