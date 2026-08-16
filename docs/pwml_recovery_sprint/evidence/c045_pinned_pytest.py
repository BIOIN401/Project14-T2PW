"""Run pytest and prove, from **inside** that process, which ``t2pw`` it imported.

The venv's editable ``.pth`` hard-codes the primary checkout's ``src`` and there
is no repo ``conftest.py``, so a pytest launched from a worktree without
``PYTHONPATH`` imports the *primary* tree and a card that modifies an existing
file passes while testing code it did not write (shared block S2). ``PYTHONPATH``
alone is not evidence of anything -- what it was actually resolved to is.

**Two defects this file used to carry, both fixed by delegating to
``pinned_pytest.main`` (H-010).** It *printed* the resolved ``t2pw.__file__`` and
proceeded regardless, so a run against the wrong tree printed the wrong path and
still reached a green summary; and, being invoked as a *script*, its
``sys.path[0]`` was ``evidence/`` rather than the repository root, so the root
joined ``sys.path`` nowhere and any selection reaching
``tests/test_bench_controls.py:344`` died on
``ModuleNotFoundError: No module named 'scripts'``.

Argv semantics and the ``T2PW: <path>`` line are unchanged, so the committed
invocations at ``g11/C-045/17-focused.json:6`` and ``g11/C-045a/20-focused.json:6``
keep working -- and now actually run.

Usage::

    <python> c045_pinned_pytest.py -q --basetemp=C:/t/c045/foc tests/test_x.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinned_pytest import main  # noqa: E402 - must follow the sys.path pin above

raise SystemExit(main(sys.argv[1:]))
