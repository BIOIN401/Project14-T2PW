"""Every third-party module ``src/t2pw`` imports at module scope must be declared.

Acceptance test for C-066 / F-067. ``src/t2pw/llm/client.py:10`` imports ``httpx``
unconditionally and uses it (``httpx.Timeout`` at :55, :59), yet no dependency file
declared it -- and the installed ``openai`` 3.3.1 requires ``httpx2``, a *different*
distribution, so a fresh ``pip install -r requirements.txt`` yields a venv where the
LLM client every stage of every leg calls raises ``ModuleNotFoundError`` on import.

A hard-coded ``assert "httpx" in requirements`` would pass for the wrong reason, so this
walks the tree with ``ast``. It reads only files, judging the *declarations* and never
the venv running it -- which is the thing that is not reproducible.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "t2pw"
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"
FIRST_PARTY = {"t2pw"}

#: Import name -> distribution, for the cases where they differ. A name missing here
#: is assumed to equal its distribution; when that is wrong the gate fails LOUDLY.
IMPORT_TO_DISTRIBUTION = {
    "dotenv": "python-dotenv",
    "libsbml": "python-libsbml",
    "PIL": "Pillow",
}

#: Undeclared distributions whose declaration is a PRODUCT-OWNER decision, not this
#: card's; C-066 declares ``httpx`` only. Guarded by the third test below, so a stale
#: entry turns the suite red instead of quietly widening the gate.
PENDING_DECLARATION_DECISIONS = {
    "pandas": "imported only inside module-scope `if` blocks of streamlit_app.py, and "
    "guaranteed transitively by the declared `streamlit>=1.30` -- unlike httpx, whose "
    "provider dropped it",
}

#: Non-vacuity anchors (RULING 13). An ast-walking gate's failure mode is to stop
#: collecting and go green, so these pin repo facts C-066's own fix does not create.
MIN_FILES_WALKED = 100
WALK_ANCHORS = {"lxml", "openai", "pydantic", "requests", "streamlit", "python-dotenv"}
WALK_SITE_ANCHOR = ("openai", "src/t2pw/llm/client.py")
DECLARED_ANCHORS = {"pytest", "ruff", "lxml", "openai", "python-libsbml"}

_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    """PEP 503 normalisation, so ``Pillow`` and ``pillow`` are one name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_in_requirements(path: Path) -> Set[str]:
    declared: Set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQUIREMENT_NAME.match(line)
        if match:
            declared.add(_normalize(match.group(1)))
    return declared


def _declared_in_pyproject(path: Path) -> Set[str]:
    project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
    specs: List[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    declared: Set[str] = set()
    for spec in specs:
        match = _REQUIREMENT_NAME.match(spec.strip())
        if match:
            declared.add(_normalize(match.group(1)))
    return declared


def declared_distributions() -> Set[str]:
    """Union of both files: this gate does not prejudge which one is authoritative."""
    return _declared_in_requirements(REQUIREMENTS) | _declared_in_pyproject(PYPROJECT)


def _module_scope_imports(path: Path) -> List[Tuple[str, int]]:
    """Imports reachable without entering a function or class body.

    ``if`` / ``try`` / ``with`` bodies are module scope and are descended into; ``def``
    and ``class`` bodies are not, so a deliberately lazy import is correctly ignored.
    """
    found: List[Tuple[str, int]] = []

    def visit(body) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                found.extend((a.name.split(".")[0], node.lineno) for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found.append((node.module.split(".")[0], node.lineno))
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                visit(node.body)

    visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body)
    return found


def third_party_imports() -> Tuple[Dict[str, List[str]], int]:
    """``{distribution: ["<path>:<line>", ...]}`` and the number of files walked.

    Stdlib names come from :data:`sys.stdlib_module_names`, never a hand-maintained
    list, which would rot silently against a new Python.
    """
    sites: Dict[str, List[str]] = {}
    files = sorted(PACKAGE.rglob("*.py"))
    for file in files:
        for name, lineno in _module_scope_imports(file):
            if name in sys.stdlib_module_names or name in sys.builtin_module_names or name in FIRST_PARTY:
                continue
            dist = _normalize(IMPORT_TO_DISTRIBUTION.get(name, name))
            sites.setdefault(dist, []).append(f"{file.relative_to(ROOT).as_posix()}:{lineno}")
    return sites, len(files)


def _assert_walk_is_live(sites: Dict[str, List[str]], files_walked: int) -> None:
    """Called by the main gate BEFORE its subset check, so a walk that collects
    nothing fails that gate too -- not merely a separate advisory test."""
    assert files_walked >= MIN_FILES_WALKED, f"walked only {files_walked} files under {PACKAGE}"
    missing = {_normalize(a) for a in WALK_ANCHORS} - set(sites)
    assert not missing, f"walk lost known module-scope imports: {sorted(missing)}"
    dist, path = WALK_SITE_ANCHOR
    hits = sites[_normalize(dist)]
    assert any(s.startswith(path + ":") for s in hits), f"walk lost {dist} at {path}"


def test_every_module_scope_third_party_import_is_declared() -> None:
    sites, files_walked = third_party_imports()
    _assert_walk_is_live(sites, files_walked)
    pending = {_normalize(name) for name in PENDING_DECLARATION_DECISIONS}
    undeclared = sorted(set(sites) - declared_distributions() - pending)
    detail = "\n".join(f"  {dist}  <- {', '.join(sorted(set(sites[dist])))}" for dist in undeclared)
    assert not undeclared, (
        "third-party distributions imported at module scope by src/t2pw but declared in "
        f"neither requirements.txt nor pyproject.toml:\n{detail}\n"
        "A fresh `pip install -r requirements.txt` gives a venv where importing these "
        "raises ModuleNotFoundError (F-067)."
    )


def test_walk_is_not_vacuous() -> None:
    """RULING 13: neutralising the walk, or a parser that returns everything, nothing
    or comment text, must not be able to make this gate green."""
    sites, files_walked = third_party_imports()
    _assert_walk_is_live(sites, files_walked)
    declared = declared_distributions()
    assert DECLARED_ANCHORS <= declared, f"parse lost {sorted(DECLARED_ANCHORS - declared)}"
    assert 15 <= len(declared) <= 60, f"implausible declared set of {len(declared)}"
    assert "chromadb" not in declared, "commented-out optional deps must not count"


def test_pending_declaration_decisions_are_still_pending() -> None:
    """A quarantine entry no longer both imported and undeclared is dead weight, and
    dead weight in an exclusion list is how a gate stops gating."""
    sites, _ = third_party_imports()
    declared = declared_distributions()
    for name in PENDING_DECLARATION_DECISIONS:
        dist = _normalize(name)
        assert dist in sites, f"{dist} is no longer imported at module scope: drop the entry"
        assert dist not in declared, f"{dist} is now declared: drop the entry, the gate covers it"
