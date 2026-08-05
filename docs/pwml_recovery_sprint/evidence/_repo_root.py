"""Repo-root discovery shared by every evidence script.

These scripts are committed inside the repository they measure, so the root is
found by walking up from ``__file__`` rather than being hard-coded. The original
scratch versions carried an absolute Windows path and would not run anywhere
else; that is the one change made to them when they were committed.

Usage::

    from _repo_root import REPO_ROOT, add_src_to_path
    add_src_to_path()
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "t2pw").is_dir():
            return candidate
    raise RuntimeError(
        "could not locate the repository root above " + str(start) + "; expected a "
        "directory containing both pyproject.toml and src/t2pw"
    )


REPO_ROOT: Path = _find_root(Path(__file__).resolve())


def add_src_to_path() -> Path:
    """Put ``<repo>/src`` on ``sys.path`` and return it."""

    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def require(relative: str) -> Path | None:
    """Return ``<repo>/<relative>`` if it exists, else ``None`` with a notice.

    Several scripts read run artifacts that are committed at different times --
    ``runs_verify/2026-08-04_1754/`` in particular is committed by INIT-001, not
    by the control-plane setup. A missing artifact is reported and skipped rather
    than raising, so a script still produces its other measurements.
    """

    path = REPO_ROOT / relative
    if path.exists():
        return path
    print(f"  [skip] {relative} is not present in this checkout")
    return None
