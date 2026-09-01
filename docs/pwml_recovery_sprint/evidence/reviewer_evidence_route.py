"""C-109 -- NEW CAPABILITY. Is a reviewer's evidence reachable from the integration branch?

**G9 label: this is a NEW capability, not a correction of existing behaviour.** Nothing
in the sprint performed this check before, so there is no base failure to show and none
is fabricated. Its acceptance test is
``tests/test_c109_reviewer_evidence_route.py``.

Why it exists
-------------
**72 reviewer G11 reports and 94 probes were nearly lost this sprint**, because a
reviewer's evidence lived only in a worktree and the merge that accepted the review did
not carry it. Prose does not hold that. This is the deterministic check.

What it does, and the one thing that makes it worth having
----------------------------------------------------------
Given a task id and a worktree path it enumerates the reviewer's load-bearing evidence:

* ``docs/pwml_recovery_sprint/evidence/g11/<TASK>/*.json`` -- the G11 reports;
* ``docs/pwml_recovery_sprint/evidence/<task-stem>_*`` -- the probe sources and their
  outputs, where the stem is the task id lowercased with separators removed
  (``REV-105`` -> ``rev105``, ``C-107`` -> ``c107``).

and decides reachability **BY CONTENT, NEVER BY FILENAME.** Each worktree file is
hashed to its git blob object id, and that object id is looked for in the integration
branch's tree. **A same-named file with different bytes is NOT reachable evidence** --
that distinction is the whole point of the check, and it is reported as its own class,
``unreachable_content_differs``, because it is the failure mode that looks green to a
human eye. A filename check would call it reachable and lose the evidence anyway.

Conversely a file whose content is present in the integration tree **under a different
path** IS reachable -- the bytes survived, which is what the sprint needs. That is
reported with the path where the content was found.

Exit codes -- it is a merge gate, so silence is not an option
-------------------------------------------------------------
* ``0`` -- every enumerated item is reachable.
* ``1`` -- at least one item is unreachable. Every unreachable item is listed.
* ``2`` -- usage or infrastructure error (bad path, git unavailable, bad ref).
* ``3`` -- **nothing was enumerated at all.** A gate that passes when the task id was
  mistyped is the F-154 silent-failure class over again. Pass ``--allow-empty`` when a
  task genuinely produced no evidence and you have said so out loud.

Scope discipline
----------------
**This is about evidence REACHABILITY and nothing else.** It is not a linter, not a
commit hook, and it does not and must not judge whether the evidence is *good*, whether
a report says PASS, or whether a probe proves anything. Those are the reviewer's job and
the orchestrator's; this answers exactly one question: *if this worktree vanished right
now, would the evidence still exist?*

Usage
-----
::

    <venv-python> docs/pwml_recovery_sprint/evidence/reviewer_evidence_route.py \\
        --task REV-109 --worktree C:/t/rev109 \\
        --integration-repo C:/Users/.../Project14-T2PW \\
        --integration-ref sprint/pwml-recovery

A directory may stand in for the integration branch when there is no git ref to name
(``--integration-dir``); the same content hash is used on both sides so the two modes
are interchangeable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

EVIDENCE_REL = "docs/pwml_recovery_sprint/evidence"

REACHABLE = "reachable"
UNREACHABLE_DIFFERS = "unreachable_content_differs"
UNREACHABLE_ABSENT = "unreachable_absent"

EXIT_OK = 0
EXIT_UNREACHABLE = 1
EXIT_USAGE = 2
EXIT_NOTHING_ENUMERATED = 3


class RouteError(Exception):
    """Usage or infrastructure failure -- never a reachability verdict."""


# --------------------------------------------------------------------------
# content identity
# --------------------------------------------------------------------------

def blob_oid(data: bytes) -> str:
    """The git blob object id of ``data``. Content identity, not filename identity."""
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def blob_oid_of_file(path: Path) -> str:
    return blob_oid(path.read_bytes())


def _git() -> str:
    exe = shutil.which("git")
    if exe is None:
        raise RouteError("git is not on PATH; cannot read the integration tree")
    return exe


def git_blob_oids(cwd: Path, paths: list[Path]) -> dict[Path, str]:
    """Hash worktree files THROUGH git, so the ids are comparable with ``ls-tree``.

    ``cwd`` must be the WORKTREE, not the integration repo: ``git hash-object`` refuses
    a path outside the repository it is run in, and in production a sprint worktree and
    the integration repo share one object store and one filter configuration anyway.

    Hashing the raw bytes in Python would disagree with the tree on any repo whose
    checkout filters (``core.autocrlf``, ``.gitattributes``) rewrite the file, and the
    disagreement would surface as a *false* unreachable -- a check that cries wolf gets
    turned off. ``git hash-object`` applies the same clean filter the tree was built
    with.
    """
    if not paths:
        return {}
    stdin = "\n".join(str(p) for p in paths) + "\n"
    proc = subprocess.run(
        [_git(), "-C", str(cwd), "hash-object", "--stdin-paths"],
        input=stdin,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RouteError(
            f"git hash-object failed in {cwd}: {proc.stderr.strip()} "
            f"(is the worktree a git checkout? use --integration-dir if not)"
        )
    oids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(oids) != len(paths):
        raise RouteError(
            f"git hash-object returned {len(oids)} ids for {len(paths)} paths"
        )
    return dict(zip(paths, oids))


# --------------------------------------------------------------------------
# the integration side: an index of CONTENT, keyed by object id
# --------------------------------------------------------------------------

def index_from_git_ref(repo: Path, ref: str) -> dict[str, list[str]]:
    proc = subprocess.run(
        [_git(), "-C", str(repo), "ls-tree", "-r", "-z", ref],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RouteError(
            f"cannot read integration ref {ref!r} in {repo}: {proc.stderr.strip()}"
        )
    index: dict[str, list[str]] = {}
    for record in proc.stdout.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if len(parts) < 3 or parts[1] != "blob":
            continue
        index.setdefault(parts[2], []).append(path)
    return index


def index_from_directory(root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        index.setdefault(blob_oid_of_file(path), []).append(rel)
    return index


def paths_in_git_ref(repo: Path, ref: str) -> set[str]:
    proc = subprocess.run(
        [_git(), "-C", str(repo), "ls-tree", "-r", "-z", "--name-only", ref],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RouteError(f"cannot list paths of {ref!r}: {proc.stderr.strip()}")
    return {p for p in proc.stdout.split("\0") if p}


def paths_in_directory(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------

def task_stem(task: str) -> str:
    """``REV-105`` -> ``rev105``; ``C-107`` -> ``c107``."""
    return "".join(ch for ch in task if ch.isalnum()).lower()


def enumerate_evidence(worktree: Path, task: str) -> list[dict]:
    """G11 reports for ``task``, plus probe sources and outputs under its stem."""
    evidence_root = worktree / EVIDENCE_REL
    if not evidence_root.is_dir():
        raise RouteError(f"no evidence directory at {evidence_root}")

    items: list[dict] = []

    g11_dir = evidence_root / "g11" / task
    if g11_dir.is_dir():
        for path in sorted(g11_dir.glob("*.json")):
            if path.is_file():
                items.append({"kind": "g11_report", "path": path})

    stem = task_stem(task)
    for path in sorted(evidence_root.glob(f"{stem}_*")):
        if path.is_file():
            items.append({"kind": "probe", "path": path})

    return items


# --------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------

def check(
    task: str,
    worktree: Path,
    integration_index: dict[str, list[str]],
    integration_paths: set[str],
    oid_of: dict[Path, str],
) -> dict:
    items = []
    for item in enumerate_evidence(worktree, task):
        path: Path = item["path"]
        oid = oid_of[path]
        rel = path.relative_to(worktree).as_posix()
        found_at = integration_index.get(oid, [])
        if found_at:
            verdict = REACHABLE
        elif rel in integration_paths:
            # The failure mode that looks green to a human: the name is there, the
            # bytes are not. This is the distinction the check exists for.
            verdict = UNREACHABLE_DIFFERS
        else:
            verdict = UNREACHABLE_ABSENT
        items.append(
            {
                "kind": item["kind"],
                "worktree_path": rel,
                "blob": oid,
                "verdict": verdict,
                "found_at": found_at,
            }
        )
    unreachable = [i for i in items if i["verdict"] != REACHABLE]
    return {
        "task": task,
        "task_stem": task_stem(task),
        "worktree": str(worktree),
        "enumerated": len(items),
        "reachable": len(items) - len(unreachable),
        "unreachable": len(unreachable),
        "items": items,
    }


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Is this task's reviewer evidence reachable from integration?"
    )
    ap.add_argument("--task", required=True, help="task id, e.g. REV-109 or C-107")
    ap.add_argument("--worktree", required=True, help="the reviewer's worktree root")
    ap.add_argument("--integration-repo", help="repo holding the integration ref")
    ap.add_argument(
        "--integration-ref",
        default="sprint/pwml-recovery",
        help="the integration branch (default: sprint/pwml-recovery)",
    )
    ap.add_argument(
        "--integration-dir",
        help="a directory standing in for the integration tree, instead of a ref",
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="do not fail when the task enumerated no evidence at all",
    )
    ap.add_argument("--json", dest="json_out", help="write the full report here")
    args = ap.parse_args(argv)

    try:
        worktree = Path(args.worktree).resolve()
        if not worktree.is_dir():
            raise RouteError(f"worktree does not exist: {worktree}")

        if args.integration_dir:
            integration_root = Path(args.integration_dir).resolve()
            if not integration_root.is_dir():
                raise RouteError(f"integration dir does not exist: {integration_root}")
            index = index_from_directory(integration_root)
            known_paths = paths_in_directory(integration_root)
            source = f"dir:{integration_root}"
            hashed = {
                item["path"]: blob_oid_of_file(item["path"])
                for item in enumerate_evidence(worktree, args.task)
            }
        else:
            repo = Path(args.integration_repo or worktree).resolve()
            index = index_from_git_ref(repo, args.integration_ref)
            known_paths = paths_in_git_ref(repo, args.integration_ref)
            source = f"{repo}@{args.integration_ref}"
            hashed = git_blob_oids(
                worktree,
                [item["path"] for item in enumerate_evidence(worktree, args.task)],
            )

        report = check(args.task, worktree, index, known_paths, hashed)
        report["integration"] = source
    except RouteError as exc:
        print(f"ROUTE ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if report["enumerated"] == 0:
        report["exit_reason"] = "nothing_enumerated"
        code = EXIT_OK if args.allow_empty else EXIT_NOTHING_ENUMERATED
    elif report["unreachable"]:
        report["exit_reason"] = "unreachable_evidence"
        code = EXIT_UNREACHABLE
    else:
        report["exit_reason"] = "all_reachable"
        code = EXIT_OK
    report["exit_code"] = code

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    print(f"task        : {report['task']} (stem {report['task_stem']})")
    print(f"worktree    : {report['worktree']}")
    print(f"integration : {report['integration']}")
    print(
        f"enumerated  : {report['enumerated']}  "
        f"reachable {report['reachable']}  unreachable {report['unreachable']}"
    )
    if report["enumerated"] == 0:
        print(
            "NOTHING ENUMERATED. Either the task id is wrong or this reviewer produced "
            "no evidence. Not a pass -- pass --allow-empty to assert the latter.",
            file=sys.stderr,
        )
    for item in report["items"]:
        if item["verdict"] == REACHABLE:
            continue
        print(
            f"UNREACHABLE [{item['verdict']}] {item['kind']}: "
            f"{item['worktree_path']}  blob={item['blob']}",
            file=sys.stderr,
        )
    if report["unreachable"]:
        print(
            f"\n{report['unreachable']} item(s) of reviewer evidence exist ONLY in the "
            f"worktree. Commit them to the integration branch before merging this card "
            f"and before releasing or cleaning the worktree.",
            file=sys.stderr,
        )
    print(f"exit_reason : {report['exit_reason']}")
    print(f"exit_code   : {code}")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
