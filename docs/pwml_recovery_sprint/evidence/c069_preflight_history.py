"""How long has ``runner.CHILD_IMPORTS`` been blind, and to exactly what? (F-073, C-069)

F-073 says four of the six unregistered deferral sites in ``batch/driver.py`` were
already unregistered before C-056d, and that recording *which four, and since
when*, is the only evidence of how long the batch preflight has been partially
blind. This measures that rather than arguing it.

The guard's predicate is purely static -- ``ast`` over ``driver.py`` plus a name
comparison against ``runner.CHILD_IMPORTS`` -- so it can be evaluated against any
revision in one interpreter, as ``c053_preflight_base_attribution.py`` already does
for a single base SHA. This walks a sequence of revisions and attributes each newly
blind site to the commit that introduced it. Blobs come from ``git show`` into
memory; nothing is checked out and nothing in the worktree is written.

``_deferred_imports`` is imported from the test module, never re-implemented.
``CHILD_IMPORTS`` must be read out of each historical ``runner.py`` by ``ast``,
because the imported ``runner`` is the tip's; the ancestor rule is copied from
``_covered`` and then cross-checked -- the worktree row is recomputed with the real
``_covered`` and any disagreement is recorded as ``covered_rule_drift`` and exits 1.

Output path is resolved before anything else (F-045).
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DRIVER = "src/t2pw/batch/driver.py"
RUNNER = "src/t2pw/batch/runner.py"

#: Oldest first: the sprint origin, every commit ``git log -S`` attributes a
#: deferral site to, then this card's base.
REVS: Tuple[Tuple[str, str], ...] = (
    ("9e1b9abe7ba8a1a228558fd03ca6c394cc22c31e", "ORIGIN_SHA (sprint baseline)"),
    ("d179c4935b6b47bf958d9dbccc7a9b72b96e32b3", "C-041 release status as a runtime classification"),
    ("f3ab5a9b96adcb09e26a6ca514039078607111e4", "C-031 persist quarantine artifacts at the boundary"),
    ("57be026a9f4ea367a66b9597629710e86365c7d3", "C-053 the artifact set names its PWML by release state"),
    ("7e04a1fb65de4ce5cb02f6e4579bcfb4be223a8c", "C-056d gate-failure path carries the semantic verdict"),
    ("e616846de75e2098e3fb76592665955b3cfe3bbc", "C-069 base SHA"),
)


def _blob(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=str(ROOT), capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout


def _child_imports(runner_source: str) -> List[str]:
    """The declared module names in one revision's ``CHILD_IMPORTS`` literal."""

    for node in ast.walk(ast.parse(runner_source)):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name):
            target = node.targets[0].id
        if target == "CHILD_IMPORTS" and node.value is not None:
            return [pair[0] for pair in ast.literal_eval(node.value)]
    return []


def _covered_by(listed: Sequence[str], module_name: str) -> bool:
    """``_covered``'s rule, verbatim, against an arbitrary declared list."""

    return any(name == module_name or name.startswith(module_name + ".") for name in listed)


def _sites(driver_source: str) -> List[Dict[str, Any]]:
    """Every deferred import with its enclosing function and line, in file order."""

    found: List[Dict[str, Any]] = []
    for node in ast.walk(ast.parse(driver_source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            names: List[str] = []
            if isinstance(inner, ast.ImportFrom):
                if inner.level == 0 and inner.module:
                    names = [inner.module]
            elif isinstance(inner, ast.Import):
                names = [alias.name for alias in inner.names]
            for name in names:
                found.append({"module": name, "function": node.name, "line": inner.lineno})
    return sorted(found, key=lambda row: (row["line"], row["module"]))


def _measure(driver_source: str, runner_source: str) -> Dict[str, Any]:
    listed = _child_imports(runner_source)
    blind = [
        site for site in _sites(driver_source)
        if site["module"].split(".")[0] not in sys.stdlib_module_names
        and not _covered_by(listed, site["module"])
    ]
    distinct = sorted({site["module"] for site in blind})
    return {
        "child_imports_declared": listed,
        "child_imports_count": len(listed),
        "missed_occurrences": [site["module"] for site in blind],
        "missed_occurrence_count": len(blind),
        "missed_distinct_modules": distinct,
        "missed_distinct_count": len(distinct),
        "blind_sites": blind,
    }


def _drift_check(worktree_row: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the worktree row through the guard's OWN helpers and compare."""

    from test_batch_preflight import DRIVER_PATH, _covered, _deferred_imports

    guard = [
        name for name in _deferred_imports(DRIVER_PATH)
        if name.split(".")[0] not in sys.stdlib_module_names and not _covered(name)
    ]
    return {
        "guard_missed": guard,
        "instrument_missed": worktree_row["missed_occurrences"],
        "covered_rule_drift": sorted(guard) != sorted(worktree_row["missed_occurrences"]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve()  # BEFORE anything runs (F-045)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeline: List[Dict[str, Any]] = []
    seen: set = set()
    for rev, note in REVS:
        row = _measure(_blob(rev, DRIVER), _blob(rev, RUNNER))
        row["rev"], row["note"] = rev[:7], note
        row["sites_first_blind_here"] = [
            site for site in row["blind_sites"]
            if (site["module"], site["function"]) not in seen
        ]
        seen |= {(site["module"], site["function"]) for site in row["blind_sites"]}
        timeline.append(row)

    worktree = _measure(
        (ROOT / DRIVER).read_text(encoding="utf-8"),
        (ROOT / RUNNER).read_text(encoding="utf-8"),
    )
    worktree["rev"], worktree["note"] = "WORKTREE", "the tree as it stands now"
    drift = _drift_check(worktree)

    out_path.write_text(json.dumps({
        "question": "which deferral sites has the preflight been blind to, and since when?",
        "timeline": timeline, "worktree": worktree, "drift_check": drift,
    }, indent=2, sort_keys=True), encoding="utf-8")

    for row in timeline + [worktree]:
        print(f"{row['rev']:<9} CHILD_IMPORTS={row['child_imports_count']} "
              f"missed_occurrences={row['missed_occurrence_count']} "
              f"distinct={row['missed_distinct_count']} {row['missed_distinct_modules']}")
        for site in row.get("sites_first_blind_here", []):
            print(f"          + newly blind: {site['module']}  "
                  f"in {site['function']}()  driver.py:{site['line']}")
    print(f"\ndrift_check: {json.dumps(drift, sort_keys=True)}")
    print(f"wrote {out_path}  (exists={out_path.is_file()})")
    return 1 if drift["covered_rule_drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
