"""C-096 / F-129 -- what does the AMBIENT PathBank answer in THIS tree?

Read-only. Prints the tree it measured, whether ``PathBankDbResolver.from_env()``
returns a resolver at all, and whether that resolver reports ``available()``.
Prints **no** connection setting and no credential: only the resolved module
path and three booleans, so the output is safe to paste into a report.

Why it exists: F-129's four failing tests are green or red depending on whether
an ambient PathBank happens to be reachable, and an agent worktree carries no
``.env`` while the primary checkout does. Before any delta against the committed
baseline can mean anything, the ambient answer in the measured tree has to be
recorded rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw  # noqa: E402
from t2pw.mapping.map_ids import PathBankDbResolver  # noqa: E402

print("t2pw.__file__       :", t2pw.__file__)
resolver = PathBankDbResolver.from_env()
print("from_env() is None  :", resolver is None)
if resolver is not None:
    try:
        print("available()         :", bool(resolver.available()))
    except Exception as exc:  # noqa: BLE001
        print("available() raised  :", type(exc).__name__)
    print("last_error is empty :", not str(getattr(resolver, "last_error", "") or ""))
