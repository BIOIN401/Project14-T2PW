"""Verify -- before pytest exists in the process -- WHICH TREE is being measured.

``PYTHONPATH`` is not evidence and a printed path is not evidence. The venv's editable
``.pth`` names the primary checkout's ``src``, ``pytest.ini`` sets no ``pythonpath`` and
there is no ``conftest.py`` (F-003); separately any in-process ``sys.path.insert(0, ...)``
-- ``_repo_root.add_src_to_path()``, and the self-pin in 24 of the 27 smoke/Chunk-D test
modules -- **overrides** ``PYTHONPATH`` in a worktree while being a no-op in the primary
checkout. Only the *resolved* path, compared against the tree under measurement, settles
which tree was measured.

``bounded_run.py`` records no environment and deletes the child's merged log (F-004), so a
verdict printed to stdout is as uncheckable as ``c045_pinned_pytest.py``'s ``print`` was.
Every refusal and every pass therefore writes its own JSON verdict, produced by the process
that actually resolved the imports. This module imports nothing from ``src/``, nothing from
``bounded_run``, and **never ``pytest``**: it runs before pytest exists in the process.
``expected_tree`` duplicates ``_repo_root._find_root`` rather than importing it -- importing
``_repo_root`` would execute its ``REPO_ROOT`` resolution and entangle this guard with the
very asymmetry it audits.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Greppable first token of the refusal banner.
REFUSAL_MARKER = "T2PW_MEASUREMENT_TREE_REFUSED"

#: Reserved sprint-wide. Distinct from pytest's 0-5, from 97
#: (``bounded_run.EXIT_INFRASTRUCTURE_FAILURE``), 124 (timeout) and 130 (cancelled).
EXIT_MEASUREMENT_TREE_REFUSED = 98

#: Emitted in this order, so a verdict diff is stable. A bad *expectation* reads
#: differently from a wrong *tree*: they are different operator errors.
VIOLATION_CODES = (
    "EXPECT_TREE_NOT_A_CHECKOUT",
    "T2PW_UNIMPORTABLE",
    "T2PW_FROM_WRONG_TREE",
    "CWD_NOT_EXPECTED_ROOT",
    "SCRIPTS_UNIMPORTABLE",
    "SCRIPTS_FROM_WRONG_TREE",
    "SELECTION_OUTSIDE_EXPECTED_TREE",
    "VERDICT_UNWRITABLE",
)

_IMPORTS_SCRIPTS = re.compile(r"^[ \t]*(?:from|import)[ \t]+scripts\b", re.M)


def is_checkout(path: Path) -> bool:
    """The tree-identity rule -- one definition, shared by discovery and by
    ``--expect-tree`` validation, so the two can never drift."""

    return (path / "pyproject.toml").is_file() and (path / "src" / "t2pw").is_dir()


def expected_tree(start: Optional[Path] = None) -> Path:
    """Nearest ancestor holding ``pyproject.toml`` **and** ``src/t2pw``, walking up from
    this module's own file by default."""

    base = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (base, *base.parents):
        if is_checkout(candidate):
            return candidate
    raise RuntimeError(
        "could not locate the tree under measurement above " + str(base)
        + "; expected a directory containing both pyproject.toml and src/t2pw")


def _under(child: Optional[str], parent: Path) -> bool:
    """``resolve()``-based containment -- never string equality, because ``PYTHONPATH``,
    the ``.pth`` and the cwd disagree on separator spelling and on case."""

    if not child:
        return False
    try:
        return Path(child).resolve().is_relative_to(parent.resolve())
    except (OSError, ValueError):
        return False


def selection_paths(selection: Sequence[str]) -> List[str]:
    """Existing filesystem paths named by a pytest selection. Flags are skipped and a
    token that is not an existing path is ignored -- a ``-k`` expression is pytest's
    business, not the guard's."""

    found: List[str] = []
    for token in selection:
        head = "" if token.startswith("-") else token.split("::", 1)[0]
        try:
            if head and Path(head).exists():
                found.append(head)
        except OSError:
            continue
    return found


def selection_needs_scripts(paths: Sequence[str]) -> bool:
    """Selection-derived default for ``require_scripts``, defaulting to **off**: a
    selection whose sources cannot be read is not made to require ``scripts``, because no
    currently-green command may be newly failed (and exported base trees carry no
    ``scripts/`` at all -- Finding H-3)."""

    for raw in paths:
        path = Path(raw)
        for candidate in sorted(path.rglob("test_*.py"))[:400] if path.is_dir() else [path]:
            try:
                if candidate.suffix == ".py" and _IMPORTS_SCRIPTS.search(
                        candidate.read_text(encoding="utf-8", errors="replace")):
                    return True
            except OSError:
                continue
    return False


def resolve_facts(expected: Path, *, selection: Sequence[str] = (),
                  cwd_at_entry: Optional[str] = None) -> Dict[str, Any]:
    """What this process actually resolved. Never raises: a failed import is a recorded
    fact, not a traceback."""

    cwd = os.getcwd()
    facts: Dict[str, Any] = {
        "expected_tree": str(expected),
        "cwd": cwd,
        "cwd_at_entry": cwd_at_entry if cwd_at_entry is not None else cwd,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "sys_path_head": list(sys.path[:8]),
        "executable": sys.executable,
        "python_version": sys.version.split()[0],
        "selection_paths": selection_paths(selection),
    }
    try:
        import t2pw  # noqa: PLC0415 - the resolution under test

        facts["t2pw_file"] = str(getattr(t2pw, "__file__", "") or "")
        facts["t2pw_error"] = None
    except BaseException as exc:  # noqa: BLE001 - a failed import is a fact
        facts["t2pw_file"] = None
        facts["t2pw_error"] = f"{type(exc).__name__}: {exc}"

    locations: List[str] = []
    try:
        spec = importlib.util.find_spec("scripts")
        facts["scripts_error"] = None
    except BaseException as exc:  # noqa: BLE001
        spec = None
        facts["scripts_error"] = f"{type(exc).__name__}: {exc}"
    if spec is not None:
        if spec.origin and spec.origin != "namespace":
            locations.append(spec.origin)
        locations.extend(str(entry) for entry in (spec.submodule_search_locations or ()))
    facts["scripts_locations"] = locations

    # The trap, named explicitly: a `src` on sys.path belonging to another tree.
    facts["foreign_src_entries"] = [
        entry for entry in sys.path
        if entry and Path(entry).name == "src" and not _under(entry, expected)]
    return facts


def check(expected: Path, facts: Dict[str, Any], *, require_scripts: bool) -> List[str]:
    """Pure. Ordered violation codes, or ``[]``.

    **Containment is against the exact package directory, never the tree root.** Every
    agent worktree lives at ``<primary>/.claude/worktrees/``, *inside* the primary checkout,
    so a root-level ``is_relative_to`` would accept ~70 wrong trees and vouch for each.
    """

    found = set()
    if not is_checkout(expected):
        found.add("EXPECT_TREE_NOT_A_CHECKOUT")

    if not facts.get("t2pw_file"):
        found.add("T2PW_UNIMPORTABLE")
    elif not _under(facts["t2pw_file"], expected / "src" / "t2pw"):
        found.add("T2PW_FROM_WRONG_TREE")

    cwd = facts.get("cwd")
    if not (cwd and _under(cwd, expected) and _under(str(expected), Path(cwd))):
        found.add("CWD_NOT_EXPECTED_ROOT")

    if require_scripts:
        locations = facts.get("scripts_locations") or []
        if not locations:
            found.add("SCRIPTS_UNIMPORTABLE")
        elif not all(_under(entry, expected / "scripts") for entry in locations):
            found.add("SCRIPTS_FROM_WRONG_TREE")

    if any(not _under(entry, expected) for entry in facts.get("selection_paths") or []):
        found.add("SELECTION_OUTSIDE_EXPECTED_TREE")
    return [code for code in VIOLATION_CODES if code in found]


def write_verdict(path: Optional[str], facts: Dict[str, Any],
                  violations: List[str]) -> Optional[str]:
    """Write the JSON verdict. Never raises; an unwritable destination is announced on
    stderr with its own marker, mirroring ``bounded_run.emit_json_report``."""

    if not path:
        return None
    payload = dict(facts, violations=violations, refused=bool(violations))
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException as exc:  # noqa: BLE001
        print(f"T2PW_PIN_VERDICT_UNWRITABLE {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None
    return str(target)


def _banner(expected: Path, facts: Dict[str, Any], violations: List[str],
            verdict: Optional[str]) -> str:
    """The refusal banner. It must **not** contain the substring pytest closes a summary
    with (``in <n>s``): ``chunk_d_gate._SUMMARY_TAIL`` anchors a verdict on exactly that
    and would otherwise misparse this banner as a pytest summary."""

    rows = [
        ("expected tree", str(expected)),
        ("t2pw resolved", facts.get("t2pw_file") or f"<unimportable> {facts.get('t2pw_error')}"),
        ("scripts", ", ".join(facts.get("scripts_locations") or []) or "<not importable>"),
        ("cwd", facts.get("cwd")),
        ("cwd at entry", facts.get("cwd_at_entry")),
        ("PYTHONPATH", facts.get("pythonpath") or "<unset>"),
        ("sys.path[0:4]", repr(facts.get("sys_path_head", [])[:4])),
        ("violations", ", ".join(violations)),
        ("verdict written", verdict or "<none>"),
    ]
    body = "\n".join(f"  {name:<15} : {value}" for name, value in rows)
    return f"{REFUSAL_MARKER}\n{body}\nREFUSED before collection. No test was run. Exit 98."


def enforce(*, expected: Optional[Path] = None, require_scripts: bool,
            verdict_path: Optional[str] = None, selection: Sequence[str] = (),
            cwd_at_entry: Optional[str] = None) -> Dict[str, Any]:
    """The one entry point: refuse with exit 98 on any violation, else return the facts."""

    tree = expected if expected is not None else expected_tree()
    facts = resolve_facts(tree, selection=selection, cwd_at_entry=cwd_at_entry)
    violations = check(tree, facts, require_scripts=require_scripts)
    verdict = write_verdict(verdict_path, facts, violations)
    # Asked for and unwritable is fatal: TEST_MATRIX § 0 requires the artifact on every
    # gate record, so a green run without one would over-claim.
    if verdict_path and verdict is None:
        violations = violations + ["VERDICT_UNWRITABLE"]
    facts["violations"] = violations
    facts["verdict_path"] = verdict
    if violations:
        print(_banner(tree, facts, violations, verdict), file=sys.stderr, flush=True)
        raise SystemExit(EXIT_MEASUREMENT_TREE_REFUSED)
    return facts
