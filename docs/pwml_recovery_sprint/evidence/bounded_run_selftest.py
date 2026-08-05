"""Validation of ``bounded_run.py`` against the six cases INIT-001 § Step 0c requires.

Every case must pass **and** end with zero surviving owned processes. Survival is
proved by an explicit post-run liveness check of the recorded PIDs against a fresh
process-table snapshot -- never by assuming that a kill worked.

Children are trivial ``python`` scripts. No pytest, no pipeline, no LLM.

::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run_selftest.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bounded_run  # noqa: E402

PY = sys.executable
RESULTS: List[Dict[str, Any]] = []


def _write(name: str, body: str) -> str:
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def _alive(pids: List[int]) -> List[int]:
    """Independent liveness check -- the PROOF, not the wrapper's own claim."""

    table = bounded_run.snapshot_processes()
    return [pid for pid in pids if pid in table]


def _record(case: str, report: bounded_run.CleanupReport, tracked: List[int],
            checks: List[tuple]) -> bool:
    leaked = _alive(tracked)
    ok = all(result for _label, result in checks) and not leaked
    RESULTS.append(
        {
            "case": case,
            "command": " ".join(report.command[:2]) + " ...",
            "exit_reason": report.exit_reason,
            "exit_code": report.exit_code,
            "returned": report.returned_code,
            "observed": len(report.descendants_observed),
            "terminated": len(report.descendants_terminated),
            "graceful": report.graceful_attempted,
            "forced": report.forced,
            "survivors_reported": report.final_surviving_count,
            "survivors_proved": len(leaked),
            "pids_checked": tracked,
            "checks": checks,
            "pass": ok,
        }
    )
    print(f"[{'PASS' if ok else 'FAIL'}] {case}")
    for label, result in checks:
        print(f"        {'ok ' if result else 'BAD'}  {label}")
    print(f"        {'ok ' if not leaked else 'BAD'}  post-run liveness: "
          f"{len(leaked)} of {len(tracked)} recorded PIDs still alive")
    return ok


# --------------------------------------------------------------------------- #

def case_1_normal() -> bool:
    r = bounded_run.run([PY, "-c", "print('hello')"], timeout=30,
                        label="c1-normal", echo=False)
    return _record("1. normal completion", r, [r.root_pid or 0], [
        ("exit_reason == completed", r.exit_reason == "completed"),
        ("real exit code 0 returned", r.exit_code == 0 and r.returned_code == 0),
        ("cleanup_success", r.cleanup_success),
        ("reported survivors == 0", r.final_surviving_count == 0),
    ])


def case_2_nonzero() -> bool:
    r = bounded_run.run([PY, "-c", "import sys; sys.exit(42)"], timeout=30,
                        label="c2-nonzero", echo=False)
    return _record("2. nonzero exit", r, [r.root_pid or 0], [
        ("exit_reason == nonzero", r.exit_reason == "nonzero"),
        ("REAL code 42 preserved, not masked", r.exit_code == 42 and r.returned_code == 42),
        ("cleanup_success", r.cleanup_success),
        ("reported survivors == 0", r.final_surviving_count == 0),
    ])


def case_3_hang() -> bool:
    r = bounded_run.run([PY, "-c", "import time; time.sleep(600)"], timeout=4,
                        label="c3-hang", echo=False)
    return _record("3. hanging child, outer timeout", r, [r.root_pid or 0], [
        ("exit_reason == timeout", r.exit_reason == "timeout"),
        ("timeout actually bounded (< 20 s)", r.duration_seconds < 20),
        ("returned 124 (timeout), not a fake pass", r.returned_code == 124),
        ("cleanup_success", r.cleanup_success),
        ("reported survivors == 0", r.final_surviving_count == 0),
    ])


def case_4_grandchild() -> bool:
    """The case a naive ``proc.kill()`` fails: the grandchild outlives it."""

    marker = os.path.join(tempfile.gettempdir(), "boundedrun_grandchild.pid")
    if os.path.exists(marker):
        os.unlink(marker)
    child = _write("boundedrun_case4_child.py", f"""
import subprocess, sys, time
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
with open(r"{marker}", "w") as fh:
    fh.write(str(p.pid))
time.sleep(600)
""")
    r = bounded_run.run([PY, child], timeout=6, label="c4-grandchild", echo=False)

    grandchild = None
    if os.path.exists(marker):
        try:
            grandchild = int(open(marker).read().strip())
        except ValueError:
            grandchild = None
    tracked = [p for p in (r.root_pid, grandchild) if p]
    return _record("4. child that spawns a child", r, tracked, [
        ("grandchild PID was captured", grandchild is not None),
        ("grandchild was OBSERVED as a descendant",
         any(d["pid"] == grandchild for d in r.descendants_observed)),
        ("grandchild recorded as terminated", grandchild in r.descendants_terminated),
        ("exit_reason == timeout", r.exit_reason == "timeout"),
        ("cleanup_success", r.cleanup_success),
        ("reported survivors == 0", r.final_surviving_count == 0),
    ])


def case_5_forced() -> bool:
    """A child that ignores graceful termination -- the escalation must show."""

    child = _write("boundedrun_case5_child.py", """
import signal, sys, time
for name in ("SIGTERM", "SIGBREAK", "SIGINT"):
    sig = getattr(signal, name, None)
    if sig is not None:
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
sys.stdout.write("ignoring graceful signals\\n"); sys.stdout.flush()
time.sleep(600)
""")
    r = bounded_run.run([PY, child], timeout=4, label="c5-forced", echo=False, grace=2.0)
    return _record("5. forced after graceful refused", r, [r.root_pid or 0], [
        ("graceful was ATTEMPTED first", r.graceful_attempted is True),
        ("escalated to FORCED", r.forced is True),
        ("exit_reason == timeout", r.exit_reason == "timeout"),
        ("cleanup_success", r.cleanup_success),
        ("reported survivors == 0", r.final_surviving_count == 0),
    ])


def case_6_cancelled() -> bool:
    """KeyboardInterrupt in the parent: cleanup must still run via ``finally``."""

    def interrupt_later() -> None:
        time.sleep(1.5)
        try:
            import _thread
            _thread.interrupt_main()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=interrupt_later, daemon=True).start()
    try:
        r = bounded_run.run([PY, "-c", "import time; time.sleep(600)"], timeout=120,
                            label="c6-cancelled", echo=False)
    except KeyboardInterrupt:
        print("[FAIL] 6. interruption -- KeyboardInterrupt escaped run()")
        RESULTS.append({"case": "6. interruption / cancellation", "pass": False,
                        "checks": [("KeyboardInterrupt escaped run()", False)]})
        return False
    return _record("6. interruption / cancellation", r, [r.root_pid or 0], [
        ("exit_reason == cancelled", r.exit_reason == "cancelled"),
        ("cleanup ran despite the interrupt", r.cleanup_success),
        ("did not wait out the 120 s timeout", r.duration_seconds < 30),
        ("reported survivors == 0", r.final_surviving_count == 0),
    ])


def main() -> int:
    print("=" * 74)
    print("bounded_run.py -- INIT-001 Step 0c validation")
    print(f"platform={sys.platform}  python={sys.version.split()[0]}  pid={os.getpid()}")
    print("=" * 74)

    outcomes = [
        case_1_normal(), case_2_nonzero(), case_3_hang(),
        case_4_grandchild(), case_5_forced(), case_6_cancelled(),
    ]

    print()
    print("=" * 108)
    print(f"{'case':<34}{'exit reason':<14}{'code':>6}{'obs':>5}{'term':>6}"
          f"{'grace':>7}{'force':>7}{'surv(rep)':>11}{'surv(proved)':>14}{'':>4}")
    print("-" * 108)
    for row in RESULTS:
        print(f"{row['case']:<34}{row.get('exit_reason', '-'):<14}"
              f"{str(row.get('returned', '-')):>6}{str(row.get('observed', '-')):>5}"
              f"{str(row.get('terminated', '-')):>6}{str(row.get('graceful', '-')):>7}"
              f"{str(row.get('forced', '-')):>7}{str(row.get('survivors_reported', '-')):>11}"
              f"{str(row.get('survivors_proved', '-')):>14}"
              f"{'  PASS' if row['pass'] else '  FAIL':>4}")
    print("=" * 108)

    total_survivors = sum(r.get("survivors_proved", 0) or 0 for r in RESULTS)
    passed = sum(1 for r in RESULTS if r["pass"])
    print(f"\n{passed}/6 cases passed | total proved survivors across all cases: "
          f"{total_survivors} (must be 0)")
    ok = all(outcomes) and total_survivors == 0
    print("RESULT:", "WRAPPER VALIDATED" if ok else "WRAPPER NOT VALIDATED -- STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
