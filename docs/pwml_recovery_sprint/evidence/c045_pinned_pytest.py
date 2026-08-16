"""Run pytest and prove, from **inside** that process, which ``t2pw`` it imported.

The venv's editable ``.pth`` hard-codes the primary checkout's ``src`` and there
is no repo ``conftest.py``, so a pytest launched from a worktree without
``PYTHONPATH`` imports the *primary* tree and a card that modifies an existing
file passes while testing code it did not write (shared block S2). ``PYTHONPATH``
alone is not evidence of anything -- what it was actually resolved to is. So this
imports ``t2pw`` first, prints the file, and only then hands off to pytest.

Card-scoped by name on purpose: a shared helper here would be a file two branches
could both add.

Usage::

    <python> c045_pinned_pytest.py -q --basetemp=C:/t/c045/foc tests/test_x.py
"""

from __future__ import annotations

import sys

import pytest

import t2pw

print(f"T2PW: {t2pw.__file__}", flush=True)
raise SystemExit(pytest.main(sys.argv[1:]))
