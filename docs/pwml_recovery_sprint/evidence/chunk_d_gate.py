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
and no retry is added.

PER-FILE ISOLATION WAS MEASURED INSUFFICIENT (H-007 supersedes the earlier reading
that no process partition could work). Running the whole 23-test ``qb`` FILE in one
fresh process leaves the documented cause untouched, because that one process still
builds 23 ``AppTest`` objects. ``run`` therefore isolates per NODE: the 150 core
tests keep their single deterministic process, and each of the 27 AppTest node IDs
gets a fresh process that has built exactly one ``AppTest``. See ``TEST_MATRIX.md``
§ "Chunk D" for the standing result; never read a ``qb`` red as expected.

``partition`` proves the split is faithful and runs on EVERY invocation, ``collect``
and ``run`` alike: it collects node IDs from the ORIGINAL monolithic selection and
from each component, compares the SETS -- not the counts -- and fails on any
omission, addition or overlap. Each parse is cross-checked against pytest's own
"N tests collected" line, so a parse miss is not agreement and a deselection cannot
hide. ``run`` then compares the set it actually EXECUTED against that same
collected set, so a substitution fails the gate even when every job it ran is green.

Every pytest invocation, collection included, goes through ``bounded_run.py`` with
its own ``--json`` cleanup report (G11 / ``[S8]``), its own SHORT ``--basetemp``
(``MAX_PATH`` is 260) which it CREATES, and a ``PYTHONPATH`` pinned to THIS
checkout's ``src`` -- the venv's ``.pth`` aims at the main checkout. One at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from _repo_root import REPO_ROOT

BOUNDED_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bounded_run.py")

_G11_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g11")
if _G11_DIR not in sys.path:
    sys.path.insert(0, _G11_DIR)
from g11_evidence import REPORTS_ROOT as G11_ROOT, allocate as g11_allocate  # noqa: E402

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
#: 150 + 4 + 23. Chunk D is 177 tests and stays 177 tests.
TOTAL = sum(expect for _n, _f, expect in COMPONENTS)
#: The two AppTest components. ``run`` executes each of their 27 node IDs ALONE.
APPTEST = ("s8", "qb")

#: Permissive after ``::`` on purpose: a parameterised ID may contain spaces.
NODE_ID = re.compile(r"^tests/.+\.py::.+$")
COLLECTED = re.compile(r"^(\d+) tests? collected")
#: Every outcome a component must report as ZERO; ``passed`` is compared to the
#: expected count instead. "145 passed, 5 errors" is a FAILURE, not 145.
NON_PASSING = ("failed", "errors", "skipped", "xfailed", "xpassed", "deselected")
_OUTCOME = re.compile(
    r"(?<![\w.])(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b")
#: pytest closes every summary with ``in <n>s``. The wrapper's CLEANUP REPORT and
#: whatever the merged stream appends at shutdown do NOT, so anchoring on this is
#: what stops trailing noise being read as a verdict.
_SUMMARY_TAIL = re.compile(r"\bin\s+\d+(?:\.\d+)?\s*s\b")
_NO_TESTS = re.compile(r"\bno tests ran\b")


def parse_summary(text: str) -> Optional[Dict[str, int]]:
    """Counts from pytest's LAST summary line, or ``None`` if there is none.

    ``None`` is NOT an implicit zero. A run whose summary is missing or
    malformed -- a timeout kills pytest before it prints one -- is a failure,
    and returning a dict of zeros here would have made it look like a clean
    "nothing went wrong".
    """

    found: Optional[Dict[str, int]] = None
    for raw in text.splitlines():
        line = raw.strip().strip("=").strip()
        if not _SUMMARY_TAIL.search(line):
            continue
        hits = _OUTCOME.findall(line)
        if not hits and not _NO_TESTS.search(line):
            continue
        counts = {name: 0 for name in ("passed",) + NON_PASSING}
        for number, word in hits:
            counts["errors" if word.startswith("error") else word] += int(number)
        found = counts
    return found


def job_verdict(expect: int, code: int, out: str,
                report: Dict[str, Any]) -> Tuple[bool, str]:
    """A component passes only on ALL of: the exact passed count, zero of every
    other outcome, a zero pytest exit, and a bounded, fully cleaned-up child.

    Three independent legs, because each alone has been observed to lie: an exit
    code can be nonzero beside an apparently valid summary, a summary can be
    absent because the job was killed at its bound, and only the wrapper's own
    committed report can say whether the child's process tree was accounted for.
    """

    bad: List[str] = []
    counts = parse_summary(out)
    if counts is None:
        bad.append("no_summary")
    else:
        if counts["passed"] != expect:
            bad.append(f"passed={counts['passed']}/{expect}")
        bad += [f"{k}={counts[k]}" for k in NON_PASSING if counts[k]]
    if code != 0:
        bad.append(f"exit={code}")
    reason = report.get("exit_reason", "report_unreadable")
    if reason != "completed":
        bad.append(f"exit_reason={reason}")
    if report.get("cleanup_success") is not True:
        bad.append("cleanup_not_successful")
    if report.get("final_surviving_count") != 0:
        bad.append(f"survivors={report.get('final_surviving_count')}")
    return not bad, ", ".join(bad) or "ok"


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


def _job(alloc: Callable[[str], str], label: str, tmp: str, selection: Sequence[str],
         timeout: float, extra: Iterable[str] = ()) -> Tuple[int, str, Dict[str, Any]]:
    """One wrapped pytest child, plus the committed cleanup report it wrote."""

    path = alloc(label)
    code, out = _pytest(label, path, tmp, selection, timeout, extra)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return code, out, json.load(fh)
    except (OSError, ValueError):
        return code, out, {}


def _collect_one(alloc: Callable[[str], str], label: str, tmp: str,
                 files: Sequence[str], timeout: float) -> Tuple[Set[str], bool, str]:
    code, out, report = _job(alloc, label, tmp, files, timeout, ("--collect-only",))
    ids, said = _parse(out)
    ok = (code == 0 and said == len(ids) and bool(ids)
          and report.get("cleanup_success") is True)
    return ids, ok, f"exit={code} parsed={len(ids)} pytest_said={said}"


def partition(alloc: Callable[[str], str], tmp: str,
              timeout: float) -> Tuple[Set[str], Dict[str, Set[str]], bool]:
    """Prove the split still collects exactly the monolithic node-ID set.

    Cross-checking each parse against pytest's own "N tests collected" is what
    catches a DESELECTION: a deselected node is counted there and never printed,
    so the two numbers stop agreeing. ``run`` calls this too -- coverage is a
    property of the invocation being gated, not of a separate earlier command.
    """

    base, ok, note = _collect_one(alloc, "mono", tmp, MONOLITHIC, timeout)
    print(f"monolithic : {note} expected={TOTAL}")
    ok &= len(base) == TOTAL
    parts: Dict[str, Set[str]] = {}
    union: Set[str] = set()
    for name, files, expect in COMPONENTS:
        ids, good, note = _collect_one(alloc, name, tmp, files, timeout)
        overlap = union & ids
        ok &= good and not overlap and len(ids) == expect
        print(f"  {name:<5}: {note} expected={expect} "
              f"overlap_with_earlier={len(overlap)}")
        parts[name], union = ids, union | ids
    missing, extra = sorted(base - union), sorted(union - base)
    print(f"union={len(union)} monolithic={len(base)} missing={len(missing)} "
          f"extra={len(extra)} SETS_EQUAL={union == base}")
    for node in missing + extra:
        print(f"  DIFFERENCE {node}")
    return base, parts, bool(ok and union == base)


def run(alloc: Callable[[str], str], tmp: str, timeout: float, node_timeout: float,
        only: Optional[str] = None) -> int:
    """Core as ONE process; each AppTest node ALONE in a fresh process, serially.

    Isolating the two AppTest FILES was already measured insufficient: the
    23-test ``qb`` file builds 23 ``AppTest`` objects in one process, which is
    the documented cause, so a per-FILE partition leaves the fault intact. This
    gives each of the 27 AppTest nodes a process that has built exactly one.

    ``--only`` still drives one component per call; the partition above is
    proven in full on every call regardless, and the executed node-ID set is
    compared to the collected one at the end -- an omission, an addition or a
    substitution fails the gate even if every job it did run was green.
    """

    base, parts, ok = partition(alloc, tmp, timeout)
    if not ok:
        print("[chunk-d] PARTITION PROOF FAILED -- gate NOT run")
        return 1

    jobs: List[Tuple[str, str, List[str], int, float]] = []
    if only in (None, "core"):
        jobs.append(("core", "run-core", CORE, len(parts["core"]), timeout))
    index = 0
    for name in APPTEST:
        for node in sorted(parts[name]):
            if only in (None, name):
                jobs.append((name, f"node{index:02d}", [node], 1, node_timeout))
            index += 1

    executed: Set[str] = set()
    failed: List[str] = []
    for name, label, selection, expect, bound in jobs:
        code, out, report = _job(alloc, label, tmp, selection, bound)
        print(out, end="")
        good, detail = job_verdict(expect, code, out, report)
        target = selection[0] if expect == 1 else f"{len(selection)} files"
        print(f"[chunk-d] {name}/{label} [{target}]: {detail}", flush=True)
        if good:
            executed |= set(selection) if expect == 1 else parts[name]
        else:
            failed.append(f"{label}={detail}")

    wanted = base if only is None else parts[only]
    print(f"[chunk-d] jobs={len(jobs)} executed={len(executed)}/{len(wanted)} "
          f"omissions={len(wanted - executed)} additions={len(executed - wanted)} "
          f"failed={failed or 'none'}")
    return 0 if not failed and executed == wanted else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("collect", "run"))
    parser.add_argument("--report", action="append", default=[], metavar="LABEL=PATH",
                        help="G11 report path per label (g11_evidence.py next)")
    parser.add_argument("--task", default=None, metavar="TASK-ID",
                        help="allocate every report --report does not supply, under "
                             "this task. Required by `run`: 27 isolated nodes plus "
                             "the collection proof is more paths than a CLI can carry")
    parser.add_argument("--report-root", default=None,
                        help="evidence/g11 root for --task (default: this checkout's). "
                             "Point it at the branch when measuring an exported tree")
    parser.add_argument("--label-prefix", default="", metavar="P",
                        help="prefix every allocated label, e.g. base1-. A scheduled "
                             "matrix needs each run's ~32 artifacts attributable to it")
    parser.add_argument("--tmp", required=True, help="SHORT basetemp root (MAX_PATH)")
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--node-timeout", type=float, default=900.0,
                        help="outer bound for ONE isolated AppTest node")
    parser.add_argument("--only", choices=[c[0] for c in COMPONENTS],
                        help="run ONE component (the partition is always proven in full)")
    args = parser.parse_args(argv)

    fixed = dict(pair.split("=", 1) for pair in args.report)

    def alloc(label: str) -> str:
        if fixed.get(label):
            return fixed[label]
        if not args.task:
            parser.error(f"no --report for {label!r} and no --task to allocate one")
        return g11_allocate(args.task, f"{args.label_prefix}{label}",
                            root=args.report_root or G11_ROOT)

    # The measured tree, printed because the venv's .pth aims at the main
    # checkout: this is the one line that says which tree the run belongs to.
    print(f"[chunk-d] tree={REPO_ROOT}", flush=True)
    if args.mode == "collect":
        return 0 if partition(alloc, args.tmp, args.timeout)[2] else 1
    return run(alloc, args.tmp, args.timeout, args.node_timeout, args.only)


if __name__ == "__main__":
    raise SystemExit(main())
