"""Deterministic split-process Chunk D gate (authorization CONTROL-PLANE-RECONCILE-001).

Chunk D's monolithic definition runs 177 tests across seven files in ONE process.
Two of those files drive Streamlit ``AppTest``, and several ``AppTest`` instances
in one process eventually leave a worker thread without a ``ScriptRunContext``:
the run dies with ``RuntimeError: FragmentThreadState not initialized``
(``script_run_context.py:144``) via ``streamlit_app.py:6187`` -> ``ui.py:26``.
The repository calls this "a Streamlit harness fault, not an app fault"
(``tests/test_streamlit_quarantine_boundary.py:425-430``).

The monolithic gate therefore FLAPS -- passing and failing on the same trees, on
a DIFFERENT test each time. A flapping gate is not a gate: it cannot distinguish
a regression from harness noise, in either direction.

Separating the AppTest surface from the other five files is process isolation, not
a test change: no test body, fixture, assertion, source file or datum is touched,
and no retry is added. MEASURED LIMIT: ``core``/``s8`` become deterministic, while
``qb`` stays NONDETERMINISTIC -- 3 green / 3 red / 2 bound-killed in 8 runs, cause
unresolved. See ``TEST_MATRIX.md`` § "Chunk D"; never read a ``qb`` red as expected.

``collect`` proves the split is faithful: it collects node IDs from the ORIGINAL
monolithic selection and from each component, compares the SETS -- not the counts
-- and fails on any omission, addition or overlap. Each parse is cross-checked
against pytest's own "N tests collected" line, so a parse miss is not agreement.

``run`` also asserts each component EXECUTED its full expected count.

Every pytest invocation, collection included, goes through ``bounded_run.py`` with
its own ``--json`` cleanup report (G11 / ``[S8]``), its own SHORT ``--basetemp``
(``MAX_PATH`` is 260) which it CREATES, and a ``PYTHONPATH`` pinned to THIS
checkout's ``src`` -- the venv's ``.pth`` aims at the main checkout. One at a time.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from _repo_root import REPO_ROOT

BOUNDED_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bounded_run.py")

CORE = [
    "tests/test_process_normalizer.py", "tests/test_pwml_ir.py",
    "tests/test_pwml_writer.py", "tests/test_stage_contracts.py",
    "tests/test_payload_models.py",
]
#: One AppTest file per component -- necessary, and for ``qb`` not sufficient.
S8 = ["tests/test_streamlit_stage8_export_contract.py"]
QB = ["tests/test_streamlit_quarantine_boundary.py"]
#: ``(name, files, expected_count)`` -- ENFORCED by ``run``; ``collect`` only reports.
COMPONENTS: List[Tuple[str, List[str], int]] = [
    ("core", CORE, 150), ("s8", S8, 4), ("qb", QB, 23)]
#: The original monolithic selection, verbatim -- what the split must equal.
MONOLITHIC = CORE + S8 + QB

#: Permissive after ``::`` on purpose: a parameterised ID may contain spaces.
NODE_ID = re.compile(r"^tests/.+\.py::.+$")
COLLECTED = re.compile(r"^(\d+) tests? collected")
#: VERDICTS only -- skipped/error/deselected excluded, so "145 passed, 5 errors" is 145.
RAN = re.compile(r"(\d+) (?:passed|failed|xfailed|xpassed)")


def _parse(text: str) -> Tuple[Set[str], Optional[int]]:
    """Node IDs, and pytest's own collected count for cross-checking the parse."""

    ids, reported = set(), None
    for raw in text.splitlines():
        line = raw.strip()
        if NODE_ID.match(line):
            ids.add(line)
        else:
            found = COLLECTED.match(line)
            if found:
                reported = int(found.group(1))
    return ids, reported


def _pytest(label: str, report: str, tmp: str, files: Sequence[str],
            timeout: float, extra: Iterable[str] = ()) -> Tuple[int, str]:
    """One wrapped, isolated pytest process. Returns ``(exit_code, stdout)``.

    ``stdout`` is captured because ``collect`` must read node IDs out of it, and
    echoed back so a ``run`` still shows its failures. The wrapper's own cleanup
    report goes to stderr and passes straight through.
    """

    os.makedirs(tmp, exist_ok=True)  # pytest mkdirs basetemp with parents=False
    proc = subprocess.run(
        [sys.executable, BOUNDED_RUN, "--label", f"chunkd-{label}",
         "--timeout", str(timeout), "--json", report, "--",
         sys.executable, "-m", "pytest", "-q", f"--basetemp={tmp}/{label}",
         *extra, *files],
        cwd=str(REPO_ROOT), env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        stdout=subprocess.PIPE, text=True, errors="replace", check=False,
    )
    return proc.returncode, proc.stdout


def collect(reports: Dict[str, str], tmp: str, timeout: float) -> int:
    """Prove the split collects exactly the monolithic node-ID set."""

    ok = True
    code, out = _pytest("mono", reports["mono"], tmp, MONOLITHIC, timeout,
                        ("--collect-only",))
    base, said = _parse(out)
    ok &= code == 0 and said == len(base) and bool(base)
    print(f"monolithic : exit={code} parsed={len(base)} pytest_said={said}")

    union: Set[str] = set()
    for name, files, _expect in COMPONENTS:
        code, out = _pytest(name, reports[name], tmp, files, timeout, ("--collect-only",))
        ids, said = _parse(out)
        overlap = union & ids
        ok &= code == 0 and said == len(ids) and not overlap
        print(f"  {name:<5}: exit={code} parsed={len(ids)} pytest_said={said} "
              f"overlap_with_earlier={len(overlap)}")
        union |= ids

    missing, extra = sorted(base - union), sorted(union - base)
    print(f"union={len(union)} monolithic={len(base)} missing={len(missing)} "
          f"extra={len(extra)} SETS_EQUAL={union == base}")
    for node in missing + extra:
        print(f"  DIFFERENCE {node}")
    return 0 if ok and union == base else 1


def run(reports: Dict[str, str], tmp: str, timeout: float,
        only: Optional[str] = None) -> int:
    """Run each selected component in its own fresh process, serialised.

    ``--only`` drives one component per call WITHOUT splitting the gate's
    definition; it is met when all three passed on one tree, over however many
    calls. A zero exit is not trusted alone: the count must match, read from the
    LAST verdict line -- ``bounded_run`` merges stderr in after pytest's summary.
    """

    selected = [(n, f, e) for n, f, e in COMPONENTS if only in (None, n)]
    failed = []
    for name, files, expect in selected:
        code, out = _pytest(name, reports[name], tmp, files, timeout)
        print(out, end="")
        summary = [ln for ln in out.splitlines() if RAN.search(ln)] or [""]
        ran = sum(int(n) for n in RAN.findall(summary[-1]))
        print(f"[chunk-d] component {name}: exit={code} ran={ran}/{expect}", flush=True)
        if code != 0 or ran != expect:
            failed.append(f"{name}(exit={code},ran={ran}/{expect})")
    print(f"[chunk-d] components={len(selected)}/{len(COMPONENTS)} "
          f"failed={failed or 'none'}")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("collect", "run"))
    parser.add_argument("--report", action="append", default=[], metavar="LABEL=PATH",
                        help="G11 report path per component (g11_evidence.py next)")
    parser.add_argument("--tmp", required=True, help="SHORT basetemp root (MAX_PATH)")
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--only", choices=[c[0] for c in COMPONENTS],
                        help="run ONE component (collect always proves all three)")
    args = parser.parse_args(argv)

    paths = dict(pair.split("=", 1) for pair in args.report)
    if args.mode == "collect":
        needed = ["mono"] + [c[0] for c in COMPONENTS]
    else:
        needed = [args.only] if args.only else [c[0] for c in COMPONENTS]
    absent = [name for name in needed if not paths.get(name)]
    if absent:
        parser.error(f"missing --report for: {absent}")
    if args.mode == "collect":
        return collect(paths, args.tmp, args.timeout)
    return run(paths, args.tmp, args.timeout, args.only)


if __name__ == "__main__":
    raise SystemExit(main())
