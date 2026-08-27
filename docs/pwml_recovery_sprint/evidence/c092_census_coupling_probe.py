"""C-092 correction round 2: does non-vacuity #1 survive a GROWING corpus?

Finding 6 of the round-2 review predicted that
``test_nonvacuity_c092_a_shrinking_corpus_turns_the_census_floor_red`` passes
TODAY and dies with ``Failed: DID NOT RAISE`` on the next committed run, because
``keep=len(corpus) - 1`` is satisfiable only while the live census equals
``C074_CENSUS_FLOOR``. A green run now therefore does not disconfirm it.

This probe simulates the future instead of waiting for it. It grows the committed
corpus by N synthetic legs -- clones of a real committed leg under fresh labels --
and runs the non-vacuity test against that grown corpus, for several N.

It measures BOTH shapes:

* ``old`` reproduces the round-1 perturbation (serve ``len(corpus) - 1`` legs
  against the module-level floor). It must FIRE at N=0 and DIE at N>=1, which is
  the probe proving it can actually detect the coupling it is looking for;
* ``new`` runs the shipped test, which pins the floor to the census it measures
  and then removes exactly one leg. It must FIRE at every N.

Read-only with respect to the repository: synthetic legs are written to a
temporary directory and every monkeypatch is undone.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

from _pytest.monkeypatch import MonkeyPatch  # noqa: E402

import test_c074_strict_core_floor as M  # noqa: E402


def _grow(real: List[Tuple[str, Path]], extra: int, tmp: Path) -> List[Tuple[str, Path]]:
    """``real`` plus ``extra`` clones of a real leg, under fresh sortable labels."""

    if extra == 0:
        return list(real)
    # Clone a leg that is NOT one of the nine C-074 names and is not recorded
    # release_ready, so the clones cannot trip assertions 3, 4 or 5.
    named = set(M.CORPUS_DEMOTED) | set(M.CORPUS_UNTOUCHED_BY_C074)
    source = None
    for label, leg in real:
        if label in named:
            continue
        report, payload = M._load(leg)
        if payload is None:
            continue
        recorded = str((report.get("release") or {}).get("status") or "")
        if recorded in (M.REVIEW_REQUIRED, M.DIAGNOSTIC_ONLY):
            source = leg
            break
    if source is None:
        raise SystemExit("no clonable leg found")

    grown = list(real)
    for index in range(extra):
        dest = tmp / f"clone{index}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        # Labels sort AFTER every real one, exactly as a newly committed run would.
        grown.append((f"2099-{index:02d}-01_0000/PMCGROWTH/research", dest))
    return grown


def _run(shape: str, grown: List[Tuple[str, Path]]) -> str:
    """Run one perturbation shape against ``grown``; report what happened."""

    outer = MonkeyPatch()
    inner = MonkeyPatch()
    try:
        outer.setattr(M, "_committed_legs", lambda: sorted(grown))
        if shape == "new":
            M.test_nonvacuity_c092_a_shrinking_corpus_turns_the_census_floor_red(inner)
        else:
            # The round-1 shape, reproduced verbatim: serve one fewer leg against
            # the MODULE-LEVEL floor, with no floor monkeypatch.
            import pytest

            served = sorted(grown)[: len(grown) - 1]
            inner.setattr(M, "_committed_legs", lambda: served)
            with pytest.raises(
                AssertionError, match="SHRUNK below C-074's measured census"
            ):
                M.test_the_full_corpus_replay_demotes_nothing_it_cannot_justify()
        return "FIRED"
    except BaseException as exc:  # noqa: BLE001 - the outcome IS the measurement
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
    finally:
        inner.undo()
        outer.undo()


def main() -> int:
    real = M._committed_legs()
    if not real:
        print(json.dumps({"skipped": "no committed legs"}))
        return 0

    results: Dict[str, Any] = {"real_corpus_legs": len(real), "floor": M.C074_CENSUS_FLOOR}
    table: Dict[str, Dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="c092probe") as raw:
        tmp = Path(raw)
        for extra in (0, 1, 5, 37):
            grown = _grow(real, extra, tmp / f"n{extra}")
            table[f"corpus+{extra}"] = {
                "old_shape": _run("old", grown),
                "new_shape": _run("new", grown),
            }
    results["by_growth"] = table
    results["verdict"] = (
        "new shape fires at every simulated corpus size"
        if all(row["new_shape"] == "FIRED" for row in table.values())
        else "NEW SHAPE IS STILL COUPLED"
    )
    json.dump(results, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
