"""The measured launcher: pin the tree, verify it, refuse it, then run pytest.

Order is the whole point. (1) Parse only this launcher's own flags. (2) Resolve the tree
under measurement -- ``--expect-tree``, else this file's own checkout. (3) **Close the CWD
hole**: put the tree root at ``sys.path[0]`` and ``chdir`` into it, reproducing exactly
what ``python -m pytest`` from the repo root already does. Run as a *script*, CPython sets
``sys.path[0]`` to the script's directory (``evidence/``), so the repository root joined
``sys.path`` nowhere and ``from scripts import batch_run``
(``tests/test_bench_controls.py:344``) raised ``ModuleNotFoundError``. (4) ``enforce`` --
refuses with exit 98 here, before plugins, ``rootdir`` and collection. (5) print the
``T2PW: ...`` line byte-identically to what ``c045_pinned_pytest.py`` always printed, so
every committed report form stays valid. (6) only now ``import pytest``.

Note what step 3 deliberately does **not** do: it does not put ``<expected>/src`` on
``sys.path``. Repairing that would make the wrong-tree refusal unreachable and turn
``PYTHONPATH`` into decoration. Half 1 is *verified*; only the cwd hole is repaired.

Usage::

    <python> pinned_pytest.py [--expect-tree DIR] [--pin-verdict FILE]
                              [--require-scripts | --no-require-scripts]
                              -q --basetemp=C:/t/x/y tests/test_x.py

``--require-scripts`` defaults to selection-derived and **off**, so no currently-green
command is newly failed; exported base trees carry no ``scripts/`` at all (Finding H-3).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import tree_pin  # noqa: E402 - must follow the sys.path pin above

_VALUE_FLAGS = ("--expect-tree", "--pin-verdict")


def split_argv(
        argv: Sequence[str]) -> Tuple[Optional[str], Optional[str], Optional[bool], List[str]]:
    """Hand-rolled on purpose: ``argparse`` prefix-matching and its handling of unknown
    short options would silently claim arguments meant for pytest."""

    expect: Optional[str] = None
    verdict: Optional[str] = None
    require: Optional[bool] = None
    rest: List[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("--require-scripts", "--no-require-scripts"):
            require = token == "--require-scripts"
        elif token in _VALUE_FLAGS:
            index += 1
            if index >= len(argv):
                raise SystemExit(f"pinned_pytest: {token} needs a value")
            if token == "--expect-tree":
                expect = argv[index]
            else:
                verdict = argv[index]
        elif token.startswith("--expect-tree="):
            expect = token.split("=", 1)[1]
        elif token.startswith("--pin-verdict="):
            verdict = token.split("=", 1)[1]
        else:
            rest.append(token)
        index += 1
    return expect, verdict, require, rest


def main(argv: Optional[Sequence[str]] = None) -> int:
    expect, verdict, require, rest = split_argv(list(sys.argv[1:] if argv is None else argv))
    expected = Path(expect).resolve() if expect else tree_pin.expected_tree()
    cwd_at_entry = os.getcwd()

    if str(expected) not in sys.path:
        sys.path.insert(0, str(expected))
    try:
        os.chdir(expected)
    except OSError as exc:  # recorded, then judged by the guard as CWD_NOT_EXPECTED_ROOT
        print(f"pinned_pytest: cannot chdir to {expected}: {exc}", file=sys.stderr, flush=True)

    if require is None:
        require = tree_pin.selection_needs_scripts(tree_pin.selection_paths(rest))

    facts = tree_pin.enforce(expected=expected, require_scripts=require, verdict_path=verdict,
                             selection=rest, cwd_at_entry=cwd_at_entry)
    print(f"T2PW: {facts['t2pw_file']}", flush=True)

    import pytest  # noqa: PLC0415 - deliberately after the guard

    return int(pytest.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
