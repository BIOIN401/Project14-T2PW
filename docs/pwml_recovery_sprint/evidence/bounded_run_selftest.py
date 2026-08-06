"""Validation of ``bounded_run.py``: the six cases INIT-001 § Step 0c requires,
plus the two H-003 / D-017 cases for the restricted-console drain defect.

Every case must pass **and** end with zero surviving owned processes. Survival is
proved by an explicit post-run liveness check of the recorded PIDs against a fresh
process-table snapshot -- never by assuming that a kill worked.

Cases 1-6 call ``bounded_run.run`` in-process. Cases 7-8 run ``bounded_run.py``
as a *subprocess*, because they are about ``main()``: the ``--json`` report and
the encoding of the parent's own stdout. Case 7 forces
``PYTHONIOENCODING=cp1252:strict`` on that subprocess, so the restricted-console
failure reproduces deterministically whether the developer's console is UTF-8 or
cp1252. That environment variable is the *fault injection*, never a fix.

Children are trivial ``python`` scripts. No pytest, no pipeline, no LLM.

::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run_selftest.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bounded_run  # noqa: E402

PY = sys.executable
RESULTS: List[Dict[str, Any]] = []

#: Where each case's cleanup report is persisted. H-003 VERIFY: *every* case must
#: produce one -- a case that passes without leaving a cleanup report is a
#: failure, because a job with no report is uncertifiable under G11.
REPORT_DIR = os.environ.get("BOUNDED_RUN_SELFTEST_REPORTS") or os.path.join(
    tempfile.gettempdir(), "boundedrun_selftest_reports"
)


def _write(name: str, body: str) -> str:
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def _alive(pids: List[int]) -> List[int]:
    """Independent liveness check -- the PROOF, not the wrapper's own claim."""

    table = bounded_run.snapshot_processes()
    return [pid for pid in pids if pid in table]


def _write_case_report(case: str, report: bounded_run.CleanupReport) -> Optional[str]:
    """Persist this case's cleanup report as JSON and prove it re-reads."""

    slug = re.sub(r"[^a-z0-9]+", "_", case.lower()).strip("_")
    path = os.path.join(REPORT_DIR, f"{slug}.json")
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
    except (OSError, ValueError, TypeError):
        return None
    return path


def _record(case: str, report: bounded_run.CleanupReport, tracked: List[int],
            checks: List[tuple]) -> bool:
    leaked = _alive(tracked)
    report_path = _write_case_report(case, report)
    checks = list(checks) + [
        ("cleanup report produced and re-readable", report_path is not None)
    ]
    ok = all(result for _label, result in checks) and not leaked
    RESULTS.append(
        {
            "case": case,
            "report_path": report_path,
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


# --------------------------------------------------------------------------- #
# H-003 / D-017 -- the restricted-console drain defect.
# --------------------------------------------------------------------------- #

WRAPPER = os.path.abspath(bounded_run.__file__)

#: Writes UTF-8 bytes STRAIGHT to the byte buffer, bypassing its own text encoder
#: -- otherwise the forced cp1252 stdio would stop the child before the parent's
#: drain saw the bytes. The invalid UTF-8 sequence makes the parent's
#: ``errors="replace"`` read *introduce* U+FFFD too: both halves of the mechanism.
_UNICODE_CHILD = (
    "import sys\n"
    "out = sys.stdout.buffer\n"
    "out.write('drain probe: pathway \\u2192 caf\\u00e9 \\u2713 \\ufffd\\n'.encode('utf-8'))\n"
    "out.write(b'raw invalid utf-8: \\xff\\xfe end\\n')\n"
    "out.flush()\n"
    "sys.exit(7)\n"
)

_CP1252_ENV = {"PYTHONIOENCODING": "cp1252:strict", "PYTHONUTF8": "0"}

#: Spelled out here rather than read off the module, so a build that lacks the
#: constant FAILS case 8 on observed behaviour instead of aborting with
#: AttributeError before ``_record`` can run. Case 8 also checks that the wrapper
#: still exports it under this name, which keeps the two in step.
_EXPECTED_UNWRITABLE_MARKER = "BOUNDED_RUN_JSON_REPORT_UNWRITABLE"


def _wrapper_subprocess(label: str, child_code: str, json_path: Optional[str],
                        env_extra: Optional[Dict[str, str]] = None,
                        timeout: float = 60.0) -> Tuple[int, bytes, bytes]:
    """Run ``bounded_run.py`` itself as a child. Returns (rc, stdout, stderr)."""

    argv = [PY, WRAPPER, "--label", label, "--timeout", "45"]
    if json_path is not None:
        argv += ["--json", json_path]
    argv += ["--", PY, "-c", child_code]
    env = os.environ.copy()
    env.update(env_extra or {})
    proc = subprocess.run(argv, capture_output=True, env=env, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def _forced_stdout_encoding(env_extra: Dict[str, str]) -> str:
    """What ``sys.stdout.encoding`` really becomes under *env_extra*. Asserted by
    case 7, so it can never pass vacuously where the injection did not take."""

    env = os.environ.copy()
    env.update(env_extra)
    probe = subprocess.run(
        [PY, "-c", "import sys; sys.stderr.write(sys.stdout.encoding)"],
        capture_output=True, env=env, timeout=60,
    )
    return probe.stderr.decode("ascii", "replace").strip().lower()


_REPORT_FIELDS = {f.name for f in dataclasses.fields(bounded_run.CleanupReport)}


def _report_from_json(path: str) -> Optional[bounded_run.CleanupReport]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return bounded_run.CleanupReport(
        **{k: v for k, v in payload.items() if k in _REPORT_FIELDS}
    )


def _report_from_render(stderr_text: str, label: str) -> bounded_run.CleanupReport:
    """Reconstruct the summary-table fields from the stderr render. Case 8's whole
    point is that no JSON file exists, so the rendered report is the only
    surviving record -- parsing it proves the cleanup result was not lost."""

    def _grab(pattern: str) -> Optional[str]:
        match = re.search(pattern, stderr_text)
        return match.group(1).strip() if match else None

    def _int(value: Optional[str]) -> Optional[int]:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    rep = bounded_run.CleanupReport(label=label, command=[PY, "-c"], cwd="", started_at="")
    rep.root_pid = _int(_grab(r"root PID / process group: (\S+) /"))
    rep.exit_reason = _grab(r"exit reason\s+: (.+)") or "unknown"
    rep.exit_code = _int(_grab(r"exit code \(real\)\s+: (\S+)"))
    rep.returned_code = _int(_grab(r"returned code\s+: (\S+)"))
    rep.final_surviving_count = _int(_grab(r"FINAL SURVIVING COUNT\s+: (\S+)")) or 0
    rep.graceful_attempted = _grab(r"graceful attempted\s+: (\S+)") == "True"
    rep.forced = _grab(r"forced\s+: (\S+)") == "True"
    rep.json_report_written = _grab(r"json report written\s+: (\S+)") == "True"
    rep.json_report_error = _grab(r"json report ERROR\s+: (.+)") or ""
    # These two must be reconstructed as well, or the persisted artifact keeps the
    # dataclass defaults and reads `cleanup_success: false` next to
    # `final_surviving_count: 0` -- the ambiguous pair a G11 checker keying on the
    # survivor count alone would misread. Anchored to line starts so a note that
    # merely mentions cleanup cannot be picked up instead.
    rep.cleanup_success = _grab(r"(?m)^cleanup\s+: (\S+)") == "success"
    rep.json_report_path = _grab(r"(?m)^json report\s+: (.+)") or ""
    return rep


def case_7_unencodable_child_output() -> bool:
    """H-003 (a)+(b): unencodable child output must not cost us the report.

    Pre-fix, ``_drain`` raises ``UnicodeEncodeError`` (a ``ValueError``, which
    ``except OSError`` misses), the exception escapes ``main()`` before the
    ``--json`` write, and the run becomes uncertifiable: no cleanup report, and
    the child's real code replaced by the interpreter's 1.
    """

    encoding = _forced_stdout_encoding(_CP1252_ENV)
    json_path = os.path.join(tempfile.gettempdir(), "boundedrun_case7_report.json")
    if os.path.exists(json_path):
        os.unlink(json_path)

    rc, out, err = _wrapper_subprocess("c7-unicode", _UNICODE_CHILD, json_path,
                                       env_extra=_CP1252_ENV)
    err_text = err.decode("cp1252", "replace")
    out_text = out.decode("cp1252", "replace")
    rep = _report_from_json(json_path)
    blank = bounded_run.CleanupReport(label="c7-unicode", command=[PY, "-c"],
                                      cwd="", started_at="")

    return _record("7. unencodable child output", rep or blank,
                   [rep.root_pid or 0] if rep else [], [
        ("fault injection took effect (stdout encoding == cp1252)", encoding == "cp1252"),
        ("no UnicodeEncodeError escaped the drain",
         "UnicodeEncodeError" not in err_text and "Traceback" not in err_text),
        ("--json cleanup report WAS written", rep is not None),
        ("report records the child's REAL exit code 7",
         rep is not None and rep.exit_code == 7 and rep.returned_code == 7),
        ("exit_reason == nonzero (not completed, not timeout)",
         rep is not None and rep.exit_reason == "nonzero"),
        ("report records ZERO surviving owned processes",
         rep is not None and rep.final_surviving_count == 0 and rep.cleanup_success),
        ("wrapper process exited 7, not 1 from a traceback", rc == 7),
        ("child output still forwarded, unencodable chars escaped",
         "drain probe" in out_text and "\\u2192" in out_text),
    ])


def case_8_unwritable_json() -> bool:
    """H-003 (c): an unwritable --json destination is reported, not fatal. The
    destination is an existing DIRECTORY, so ``open(..., "w")`` fails on both
    Windows and POSIX. Cleanup must still run, the condition must be named, and
    the child's exit code must survive."""

    json_dir = os.path.join(tempfile.gettempdir(), "boundedrun_case8_unwritable")
    if os.path.exists(json_dir) and not os.path.isdir(json_dir):
        os.unlink(json_dir)
    os.makedirs(json_dir, exist_ok=True)
    before_entries = sorted(os.listdir(json_dir))

    rc, _out, err = _wrapper_subprocess(
        "c8-unwritable-json", "import sys; sys.exit(5)\n", json_dir)
    err_text = err.decode("utf-8", "replace")
    rep = _report_from_render(err_text, "c8-unwritable-json")

    ok = _record("8. unwritable --json destination", rep, [rep.root_pid or 0], [
        ("wrapper did not crash: child's REAL code 5 returned", rc == 5),
        ("condition NAMED on stderr, not silently skipped",
         _EXPECTED_UNWRITABLE_MARKER in err_text),
        ("wrapper exports that marker constant",
         getattr(bounded_run, "JSON_REPORT_UNWRITABLE_MARKER", None)
         == _EXPECTED_UNWRITABLE_MARKER),
        ("report itself records json_report_written == False",
         rep.json_report_written is False and rep.json_report_error != ""),
        ("cleanup result NOT lost: full report still rendered",
         "FINAL SURVIVING COUNT" in err_text and rep.final_surviving_count == 0),
        ("cleanup_success recorded True", rep.cleanup_success is True),
        ("report location recorded (json_report_path)", rep.json_report_path != ""),
        ("child's exit code preserved in the record", rep.exit_code == 5),
        ("no traceback", "Traceback" not in err_text),
        ("no stray file created at the destination",
         os.path.isdir(json_dir) and sorted(os.listdir(json_dir)) == before_entries),
    ])
    shutil.rmtree(json_dir, ignore_errors=True)
    return ok


def main() -> int:
    print("=" * 74)
    print("bounded_run.py -- INIT-001 Step 0c validation + H-003 drain cases")
    print(f"platform={sys.platform}  python={sys.version.split()[0]}  pid={os.getpid()}")
    print(f"wrapper under test: {WRAPPER}")
    print("=" * 74)

    outcomes = [
        case_1_normal(), case_2_nonzero(), case_3_hang(),
        case_4_grandchild(), case_5_forced(), case_6_cancelled(),
        case_7_unencodable_child_output(), case_8_unwritable_json(),
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
    reported = sum(1 for r in RESULTS if r.get("report_path"))
    print(f"\n{passed}/{len(outcomes)} cases passed | total proved survivors across "
          f"all cases: {total_survivors} (must be 0)")
    print(f"cleanup reports produced: {reported}/{len(outcomes)} "
          f"(must equal the case count) -> {REPORT_DIR}")
    ok = all(outcomes) and total_survivors == 0 and reported == len(outcomes)
    print("RESULT:", "WRAPPER VALIDATED" if ok else "WRAPPER NOT VALIDATED -- STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
