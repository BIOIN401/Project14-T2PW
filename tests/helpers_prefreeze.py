"""Take a test fixture through the pre-freeze stage before the exporter sees it.

Not a test module (pytest only collects ``test_*.py``), same shape as the
existing ``helpers_eligibility.py``.

Why this exists
---------------
**D-015 (LOCKED)** moved compound identity resolution ahead of the canonical
freeze, and **D-032 (LOCKED)** wired the pre-freeze sequence at the production
export entry points. **C-051** then deleted the exporter's own
``_resolve_compound_rows`` call and replaced it with a fail-closed assertion: a
compound row that reaches ``build_pwml_ir`` carrying no resolution verdict is
**refused**, because resolving it there is an exporter mutating biology after
the canonical graph is frozen -- permanent **merge rule 8** and
``PRODUCT_CONTRACT`` §5.

A test that hands ``build_pwml_ir`` a raw extraction payload is therefore
feeding it a configuration **production no longer produces**: both documented
entry points, and the CLI one measurably, run the pre-freeze sequence first.
:func:`prefrozen` takes such a payload through the stage that now does the work.
This is the same re-pointing **C-045** established for species with
``test_pwml_writer._species_canonicalized_payload`` under D-016.

**What is NOT changed:** every re-pointed test still asserts exactly what it
asserted before, on the same fixture, by the same comparison. Nothing is
relaxed to a subset, an allowlist or a normalized comparand. Only the *stage
that produced the input* moves -- which is the whole point of D-015.

Non-vacuity is enforced here, not assumed
-----------------------------------------
``REV-045`` established the standard: a re-pointed test must **fail** if the
pre-freeze stage is absent or does nothing, or it is worth less than the test it
replaced. :func:`prefrozen` therefore asserts, on every call:

1. the payload has compound rows at all -- otherwise the helper asserts nothing
   and the caller should not be using it;
2. **no** row arrived already carrying a verdict -- otherwise the stage is a
   no-op here and the re-pointing is vacuous;
3. **every** row leaves carrying one, and the row count is unchanged.

Assertion 3 is the load-bearing one. Delete the pre-freeze stage, stub it out,
or let it silently skip the rows, and every test routed through this helper
fails at that line rather than passing quietly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.ir import COMPOUND_RESOLUTION_VERDICT_FIELD  # noqa: E402
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution  # noqa: E402

__all__ = [
    "prefrozen",
    "prefrozen_when_compounded",
    "prefreeze_report",
    "compound_verdicts",
]


def compound_verdicts(payload: Dict[str, Any]) -> List[str]:
    """The resolution verdict on each compound row, ``""`` where there is none."""

    rows = ((payload.get("entities") or {}).get("compounds")) or []
    return [
        str(row.get(COMPOUND_RESOLUTION_VERDICT_FIELD) or "").strip()
        for row in rows
        if isinstance(row, dict)
    ]


def prefreeze_report(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """:func:`prefrozen`, returning the stage's report instead of the payload."""

    return _run(payload, kwargs)[1]


def prefrozen(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Run the production pre-freeze sequence over ``payload``, in place.

    Returns the same object, so a call site becomes
    ``build_pwml_ir(prefrozen(payload), ...)``. ``kwargs`` are forwarded to
    ``run_prefreeze_resolution`` -- pass the same ``db_resolver`` / ``name_index``
    the test passes to ``build_pwml_ir`` so the stage resolves the way the test
    intends rather than the way the ambient environment happens to.
    """

    return _run(payload, kwargs)[0]


def prefrozen_when_compounded(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """:func:`prefrozen`, but a no-op on a payload that has no compound rows.

    **This is not a relaxation of the non-vacuity rule.** ``build_pwml_ir``'s
    own fail-closed assertion iterates the compound rows: on a payload with none
    it has nothing to refuse, so a pre-freeze stage there would be exactly as
    vacuous as the assertion it feeds. Every payload that *does* carry compound
    rows -- the entire population C-051 governs -- goes through :func:`prefrozen`
    with all three of its assertions live, including the one that fails if the
    stage rules on nothing.

    It exists for shared call sites (a module-level wrapper, a parametrized
    helper) that legitimately see both shapes. A test written against one known
    fixture should call :func:`prefrozen` directly and get the stricter contract.
    """

    if not compound_verdicts(payload):
        return payload
    return prefrozen(payload, **kwargs)


def _run(payload: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
    before = compound_verdicts(payload)
    assert before, (
        "prefrozen() was given a payload with no compound rows, so it would "
        "assert nothing. Call build_pwml_ir directly for that fixture."
    )
    assert not any(before), (
        "prefrozen() was given a payload whose compound rows ALREADY carry "
        f"resolution verdicts ({before}); the stage is a no-op here and routing "
        "this test through it would be vacuous."
    )

    kwargs.setdefault("strict_db", False)
    report = run_prefreeze_resolution(payload, **kwargs)

    after = compound_verdicts(payload)
    assert len(after) == len(before), (
        f"the pre-freeze stage changed the compound row count {len(before)} -> "
        f"{len(after)}; the fixture is no longer the one the test pins."
    )
    assert all(after), (
        "the pre-freeze compound stage did not rule on every compound row "
        f"({after}). Every test routed through this helper would be vacuous, so "
        "it fails here instead of passing quietly."
    )
    return payload, report
