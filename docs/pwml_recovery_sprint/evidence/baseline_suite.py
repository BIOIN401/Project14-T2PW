"""INIT-001 Step 4a -- the full test suite, chunked, every chunk through the wrapper.

Ten chunks, alphabetical partition (reproducible and unbiased), unique
``--basetemp`` per chunk, run **strictly sequentially** -- ``[S8]`` item 7, one
heavy job at a time. Never ``-n auto``.

Emits the per-chunk passed/failed/skipped table for ``BASELINE.md`` § 1 and a
cleanup report per chunk for gate G11.

::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/baseline_suite.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bounded_run  # noqa: E402
from _repo_root import REPO_ROOT  # noqa: E402

ROOT = str(REPO_ROOT)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):  # POSIX checkout
    PY = sys.executable

BASETEMP_ROOT = os.path.join(ROOT, ".pytest_tmp_baseline")
CHUNK_TIMEOUT = 2400.0

#: Files that must be *named* in the report regardless of which chunk they land
#: in -- chunk E skips silently when ``runs/`` inputs are absent, and INIT-001
#: requires an explicit ran-or-skipped statement about it.
CHUNK_E_FILE = "test_strict_quarantine_real_artifact_replay.py"

_SUMMARY = re.compile(
    r"(?:(\d+) failed)?[,\s]*(?:(\d+) passed)?[,\s]*(?:(\d+) skipped)?"
    r"[,\s]*(?:(\d+) error(?:s)?)?[,\s]*(?:(\d+) xfailed)?[,\s]*(?:(\d+) warning)?"
)


def test_files() -> List[str]:
    tests = sorted(
        name for name in os.listdir(os.path.join(ROOT, "tests"))
        if name.startswith("test_") and name.endswith(".py")
    )
    return tests


def partition(items: List[str], groups: int) -> List[List[str]]:
    """Contiguous alphabetical partition into *groups* near-equal chunks."""

    out: List[List[str]] = []
    n = len(items)
    start = 0
    for i in range(groups):
        size = n // groups + (1 if i < n % groups else 0)
        out.append(items[start : start + size])
        start += size
    return out


def parse_counts(text: str) -> Dict[str, int]:
    """Read the pytest terminal summary line."""

    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "xfailed": 0}
    line = ""
    for candidate in reversed(text.strip().splitlines()):
        if "=" in candidate and any(
            word in candidate
            for word in ("passed", "failed", "error", "skipped", "no tests ran")
        ):
            line = candidate
            break
    for key, pattern in (
        ("passed", r"(\d+) passed"),
        ("failed", r"(\d+) failed"),
        ("skipped", r"(\d+) skipped"),
        ("errors", r"(\d+) error"),
        ("xfailed", r"(\d+) xfailed"),
    ):
        match = re.search(pattern, line)
        if match:
            counts[key] = int(match.group(1))
    counts["_summary_line"] = line  # type: ignore[assignment]
    return counts


def run_chunk(label: str, files: List[str], *, capture: Optional[str] = None) -> Dict[str, Any]:
    basetemp = os.path.join(BASETEMP_ROOT, label)
    os.makedirs(basetemp, exist_ok=True)
    cmd = [PY, "-m", "pytest", "-q", f"--basetemp={basetemp}"] + [
        f"tests/{name}" for name in files
    ]

    log = os.path.join(BASETEMP_ROOT, f"{label}.out")
    os.makedirs(BASETEMP_ROOT, exist_ok=True)

    # Tee the child's output to a file so counts can be parsed after the fact
    # without the wrapper having to buffer it in memory.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    class _Tee:
        def __init__(self, path: str) -> None:
            self.fh = open(path, "w", encoding="utf-8", errors="replace")

        def write(self, chunk: str) -> int:
            self.fh.write(chunk)
            self.fh.flush()
            return sys.__stdout__.write(chunk)

        def flush(self) -> None:
            self.fh.flush()
            sys.__stdout__.flush()

    tee = _Tee(log)
    old = sys.stdout
    sys.stdout = tee  # type: ignore[assignment]
    try:
        report = bounded_run.run(
            cmd, timeout=CHUNK_TIMEOUT, label=label, cwd=ROOT, echo=True, env=env
        )
    finally:
        sys.stdout = old
        tee.fh.close()

    with open(log, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    counts = parse_counts(text)

    print(report.render(), file=sys.stderr)
    return {
        "label": label,
        "files": files,
        "counts": counts,
        "exit_reason": report.exit_reason,
        "exit_code": report.exit_code,
        "returned_code": report.returned_code,
        "duration_seconds": round(report.duration_seconds, 1),
        "root_pid": report.root_pid,
        "isolation": report.isolation,
        "descendants_observed": len(report.descendants_observed),
        "descendants_terminated": len(report.descendants_terminated),
        "final_surviving_count": report.final_surviving_count,
        "cleanup_success": report.cleanup_success,
        "cleanup_report": report.to_dict(),
    }


def main() -> int:
    files = test_files()
    chunks = partition(files, 10)

    print(f"full suite: {len(files)} test files in {len(chunks)} chunks, "
          f"sequential, unique --basetemp per chunk\n")

    results: List[Dict[str, Any]] = []
    for index, group in enumerate(chunks, start=1):
        label = f"chunk{index:02d}"
        print(f"\n{'#' * 78}\n# {label}: {len(group)} files\n{'#' * 78}")
        results.append(run_chunk(label, group))

    print("\n" + "=" * 96)
    print(f"{'chunk':<9}{'files':>6}{'passed':>8}{'failed':>8}{'skipped':>9}"
          f"{'errors':>8}{'runtime':>10}{'survivors':>11}{'cleanup':>10}")
    print("-" * 96)
    totals = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    survivors_total = 0
    for row in results:
        c = row["counts"]
        for key in totals:
            totals[key] += c[key]
        survivors_total += row["final_surviving_count"]
        print(f"{row['label']:<9}{len(row['files']):>6}{c['passed']:>8}{c['failed']:>8}"
              f"{c['skipped']:>9}{c['errors']:>8}{row['duration_seconds']:>9.1f}s"
              f"{row['final_surviving_count']:>11}"
              f"{'ok' if row['cleanup_success'] else 'FAILURE':>10}")
    print("-" * 96)
    print(f"{'TOTAL':<9}{len(files):>6}{totals['passed']:>8}{totals['failed']:>8}"
          f"{totals['skipped']:>9}{totals['errors']:>8}"
          f"{sum(r['duration_seconds'] for r in results):>9.1f}s"
          f"{survivors_total:>11}")
    print("=" * 96)

    # Chunk E: ran or skipped? INIT-001 4a requires this stated explicitly.
    for row in results:
        if CHUNK_E_FILE in row["files"]:
            print(f"\nchunk E ({CHUNK_E_FILE}) landed in {row['label']}; "
                  f"that chunk reported {row['counts']['skipped']} skipped. "
                  f"Summary: {row['counts'].get('_summary_line', '')}")

    out = os.path.join(ROOT, "docs", "pwml_recovery_sprint", "evidence",
                       "baseline_suite_result.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"totals": totals, "survivors_total": survivors_total,
                   "chunks": results}, fh, indent=2)
    print(f"\nwrote {out}")

    if survivors_total:
        print("INFRASTRUCTURE FAILURE: surviving owned processes. STOP.")
        return bounded_run.EXIT_INFRASTRUCTURE_FAILURE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
