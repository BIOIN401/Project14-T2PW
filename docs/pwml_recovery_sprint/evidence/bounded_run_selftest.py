"""Validation of ``bounded_run.py``: the six cases INIT-001 § Step 0c requires,
the two H-003 / D-017 cases for the restricted-console drain defect, the four
H-006 cases for the report schema version and the wrapper build identity, and the
six C-063 cases for the heavy mutex (F-072) and the G11 report lifecycle (F-071).

Every case must pass **and** end with zero surviving owned processes. Survival is
proved by an explicit post-run liveness check of the recorded PIDs against a fresh
process-table snapshot -- never by assuming that a kill worked.

Cases 1-6 call ``bounded_run.run`` in-process. Cases 7-18 run ``bounded_run.py``
as a *subprocess*, because they are about ``main()``: the ``--json`` report, the
encoding of the parent's own stdout, for 9-12 the identity fields of the artifact
a real invocation leaves behind, and for 13-18 the mutex and report lifecycle,
which exist only in ``main()`` by design. Case 7 forces
``PYTHONIOENCODING=cp1252:strict`` on that subprocess, so the restricted-console
failure reproduces deterministically whether the developer's console is UTF-8 or
cp1252. That environment variable is the *fault injection*, never a fix.

Children are trivial ``python`` scripts. No pytest, no pipeline, no LLM.

::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run_selftest.py
"""

from __future__ import annotations

import dataclasses
import hashlib
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


# --------------------------------------------------------------------------- #
# H-006 -- report schema version and wrapper build identity.
#
# All four run ``bounded_run.py`` as a subprocess: the claim is about the ARTIFACT
# a real invocation leaves behind, not about an in-process value. Every expected
# digest and every git fact is computed HERE, independently, and never read back
# out of the report being judged.
# --------------------------------------------------------------------------- #

EVIDENCE_DIR = os.path.dirname(WRAPPER)
G11_DIR = os.path.join(EVIDENCE_DIR, "g11")
REPO_ROOT = os.path.abspath(os.path.join(EVIDENCE_DIR, "..", "..", ".."))

sys.path.insert(0, G11_DIR)
import g11_evidence  # noqa: E402


def _digest_of(path: str) -> str:
    """Independent SHA-256 of a file's raw bytes, in the report's own spelling."""

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _git_in_repo(*args: str) -> str:
    """Ground truth from git, asked independently of the wrapper."""

    proc = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", REPO_ROOT, *args],
        capture_output=True, text=True, timeout=120, check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def _wrapper_report(wrapper_path: str, label: str,
                    json_path: str) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Run *wrapper_path* itself; return (rc, its decoded ``--json`` report)."""

    argv = [PY, wrapper_path, "--label", label, "--timeout", "60", "--quiet",
            "--json", json_path, "--", PY, "-c", "print('h006 child')"]
    proc = subprocess.run(argv, capture_output=True, timeout=180)
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            return proc.returncode, json.load(fh)
    except (OSError, ValueError):
        return proc.returncode, None


def _report_payload(path: str) -> Optional[Dict[str, Any]]:
    """The report a wrapper run left at *path*, decoded, or ``None``."""

    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_report(payload: Optional[Dict[str, Any]], label: str
               ) -> bounded_run.CleanupReport:
    if not payload:
        return bounded_run.CleanupReport(label=label, command=[PY, "-c"],
                                         cwd="", started_at="")
    return bounded_run.CleanupReport(
        **{k: v for k, v in payload.items() if k in _REPORT_FIELDS}
    )


def case_9_schema_and_build_fields() -> bool:
    """H-006 (a): the new fields are present and correctly populated in a report
    the wrapper actually produced, the schema VALIDATES, and the repository SHA
    and dirty-state recorded beside them are TRUTHFUL."""

    tmp = tempfile.mkdtemp(prefix="boundedrun_case9_")
    json_path = os.path.join(tmp, "01-identity.json")
    rc, payload = _wrapper_report(WRAPPER, "c9-identity", json_path)
    data = payload or {}
    build = data.get("wrapper_build") or {}
    head = _git_in_repo("rev-parse", "HEAD").strip()
    repo_dirty = bool(
        _git_in_repo("status", "--porcelain", "--untracked-files=no").strip())
    wrapper_dirty = bool(_git_in_repo("status", "--porcelain", "--", WRAPPER).strip())
    rep = _as_report(data, "c9-identity")

    ok = _record("9. schema + build identity", rep, [rep.root_pid or 0], [
        ("the wrapper wrote its --json report", payload is not None),
        ("schema_version == the wrapper's declared version",
         data.get("schema_version") == bounded_run.REPORT_SCHEMA_VERSION),
        ("schema VALIDATES: no violations",
         bounded_run.validate_report_schema(data) == []),
        ("digest == independently hashed bytes of the wrapper that ran",
         build.get("digest") == _digest_of(WRAPPER)),
        ("wrapper_build.path names the executed wrapper",
         os.path.normcase(build.get("path") or "") == os.path.normcase(WRAPPER)),
        ("repo_head TRUTHFUL: equals `git rev-parse HEAD`",
         bool(head) and build.get("repo_head") == head),
        ("repo dirty-state TRUTHFUL: equals `git status --porcelain -uno`",
         build.get("repo_tracked_files_dirty") is repo_dirty),
        ("wrapper-vs-HEAD state TRUTHFUL: 'clean' iff git reports it clean",
         (build.get("wrapper_vs_head") == "clean") is (not wrapper_dirty)),
        ("child's REAL exit code still 0", rc == 0 and data.get("exit_code") == 0),
        ("cleanup_success", data.get("cleanup_success") is True),
        ("reported survivors == 0", data.get("final_surviving_count") == 0),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def case_10_identity_stable_and_sensitive() -> bool:
    """H-006 (b): the identity is STABLE across runs of one build and CHANGES
    when the wrapper's own content changes. A constant would satisfy case 9 and
    be worthless, so both halves are asserted here."""

    tmp = tempfile.mkdtemp(prefix="boundedrun_case10_")
    copy = os.path.join(tmp, "bounded_run.py")
    shutil.copyfile(WRAPPER, copy)
    build_a = _digest_of(copy)
    rc1, r1 = _wrapper_report(copy, "c10-build-a1", os.path.join(tmp, "01-a1.json"))
    rc2, r2 = _wrapper_report(copy, "c10-build-a2", os.path.join(tmp, "02-a2.json"))
    with open(copy, "ab") as fh:
        fh.write(b"\r\n# H-006 selftest: the wrapper's own content changed here.\r\n")
    build_b = _digest_of(copy)
    rc3, r3 = _wrapper_report(copy, "c10-build-b", os.path.join(tmp, "03-b.json"))

    runs = [r1, r2, r3]
    d1, d2, d3 = [((r or {}).get("wrapper_build") or {}).get("digest") for r in runs]
    tracked = [pid for pid in ((r or {}).get("root_pid") for r in runs) if pid]
    rep = _as_report(r3, "c10-build-b")

    ok = _record("10. identity stable and sensitive", rep, tracked, [
        ("all three runs produced a report", all(r is not None for r in runs)),
        ("STABLE: two runs of the SAME build agree", d1 is not None and d1 == d2),
        ("...and both equal the independently hashed build", d1 == build_a),
        ("SENSITIVE: a content change moved the digest",
         d3 is not None and d3 == build_b and d3 != d1),
        ("not a constant: the two builds really do differ", build_a != build_b),
        ("every run kept its child's real exit code",
         rc1 == 0 and rc2 == 0 and rc3 == 0),
        ("every run reported cleanup_success",
         all((r or {}).get("cleanup_success") is True for r in runs)),
        ("every run reported 0 survivors",
         all((r or {}).get("final_surviving_count") == 0 for r in runs)),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def case_11_identity_from_the_executing_copy() -> bool:
    """H-006 (c): the CROSS-CHECKOUT failure mode made into a test. A copy of the
    wrapper runs from OUTSIDE the repository; the recorded identity must be the
    copy that RAN, never the build sitting in the tree."""

    tmp = tempfile.mkdtemp(prefix="boundedrun_case11_")
    outside = os.path.join(tmp, "bounded_run.py")
    shutil.copyfile(WRAPPER, outside)
    # Deliberately NOT byte-identical: an identical copy would make every digest
    # comparison below vacuously true.
    with open(outside, "ab") as fh:
        fh.write(b"\r\n# H-006 selftest: out-of-tree build, distinct from the tree's.\r\n")
    json_path = os.path.join(tmp, "01-crosscheckout.json")
    rc, payload = _wrapper_report(outside, "c11-crosscheckout", json_path)
    data = payload or {}
    build = data.get("wrapper_build") or {}
    inside_repo = os.path.normcase(os.path.abspath(tmp)).startswith(
        os.path.normcase(os.path.abspath(REPO_ROOT)) + os.sep)
    rep = _as_report(data, "c11-crosscheckout")

    ok = _record("11. identity from executing copy", rep, [rep.root_pid or 0], [
        ("the copy really is outside the repository", not inside_repo),
        ("the out-of-tree wrapper produced a report", payload is not None),
        ("digest == the COPY that ran", build.get("digest") == _digest_of(outside)),
        ("digest != the wrapper build in the tree",
         build.get("digest") != _digest_of(WRAPPER)),
        ("path == the copy that ran, not the tree's",
         os.path.normcase(build.get("path") or "") == os.path.normcase(outside)),
        ("identity NOT taken from the tree's repository",
         os.path.normcase(build.get("repo_root") or "")
         != os.path.normcase(REPO_ROOT)),
        ("schema still VALIDATES for an out-of-tree build",
         bounded_run.validate_report_schema(data) == []),
        ("child's REAL exit code preserved", rc == 0 and data.get("exit_code") == 0),
        ("cleanup_success", data.get("cleanup_success") is True),
        ("reported survivors == 0", data.get("final_surviving_count") == 0),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def case_12_preexisting_reports_still_validate() -> bool:
    """H-006 (d): reports written before these fields existed must STILL validate,
    and no committed one may be edited or regenerated to acquire them -- a
    reconstruction is not evidence of the original run."""

    tmp = tempfile.mkdtemp(prefix="boundedrun_case12_")
    fresh = os.path.join(tmp, "01-fresh.json")
    rc, payload = _wrapper_report(WRAPPER, "c12-compat", fresh)
    data = payload or {}
    legacy = {k: v for k, v in data.items()
              if k not in ("schema_version", "wrapper_build")}
    legacy_path = os.path.join(tmp, "02-legacy.json")
    with open(legacy_path, "w", encoding="utf-8") as fh:
        json.dump(legacy, fh, indent=2)

    task_dirs = [
        os.path.join(G11_DIR, name) for name in sorted(os.listdir(G11_DIR))
        if os.path.isdir(os.path.join(G11_DIR, name))
        and g11_evidence.TASK_RE.match(name)
    ]
    # COMMITTED means tracked by git, not merely present: the path this very run
    # allocated is sitting in one of these directories as an unwritten placeholder
    # while the case executes, and it is not a pre-existing report.
    committed = [
        os.path.normpath(os.path.join(REPO_ROOT, line.strip()))
        for line in _git_in_repo("ls-files", "--", *task_dirs).splitlines()
        if line.strip()
    ]
    without_fields = []
    for path in committed:
        with open(path, "r", encoding="utf-8") as fh:
            if "schema_version" not in json.load(fh):
                without_fields.append(path)
    # `-uno` so this task's own, still-untracked new reports are not counted as
    # tampering with a committed one.
    edited = _git_in_repo("status", "--porcelain", "--untracked-files=no", "--",
                          *task_dirs).strip()
    rep = _as_report(data, "c12-compat")

    ok = _record("12. pre-existing reports validate", rep, [rep.root_pid or 0], [
        ("g11 check accepts the NEW, versioned report",
         g11_evidence.check_report(fresh) == []),
        ("the legacy shape carries NEITHER new field",
         "schema_version" not in legacy and "wrapper_build" not in legacy),
        ("g11 check accepts a report WITHOUT the new fields",
         g11_evidence.check_report(legacy_path) == []),
        ("validate_report_schema treats absence as valid, not schema 0 == invalid",
         bounded_run.validate_report_schema(legacy) == []),
        ("committed pre-H-006 reports were found", len(without_fields) >= 16),
        ("every committed pre-H-006 report still validates",
         all(g11_evidence.check_report(p) == [] for p in without_fields)),
        ("no committed report was edited or regenerated", edited == ""),
        ("child's REAL exit code preserved", rc == 0 and data.get("exit_code") == 0),
        ("cleanup_success", data.get("cleanup_success") is True),
        ("reported survivors == 0", data.get("final_surviving_count") == 0),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


# --------------------------------------------------------------------------- #
# C-063 -- the heavy mutex (F-072) and the G11 report lifecycle (F-071).
#
# Every lock case points --heavy-lock-path at a SCRATCH directory under the case's
# own tempdir. The real C:/t/heavylock is never created, read or removed here: a
# selftest that manipulated the live mutex would be the very defect F-072 records.
# --------------------------------------------------------------------------- #

#: A foreign holder file, in the shape the wrapper itself writes. The token is
#: what release compares against, so this one can never match a live job's.
_FOREIGN_HOLDER = {
    "token": "C-999-SOMEONE-ELSE:4242:ffffffffffffffff",
    "holder": "C-999-SOMEONE-ELSE",
    "label": "a-job-that-is-still-running",
    "pid": 4242,
    "command": ["<the other card's heavy job>"],
    "cwd": "<elsewhere>",
    "acquired_at": "2026-08-21T00:00:00+00:00",
}


def _blank(label: str) -> bounded_run.CleanupReport:
    return bounded_run.CleanupReport(label=label, command=[PY, "-c"], cwd="",
                                     started_at="")


def _plant_foreign_lock(lock_dir: str) -> bytes:
    """Create *lock_dir* held by someone else. Returns the holder file's bytes."""

    shutil.rmtree(lock_dir, ignore_errors=True)
    os.makedirs(lock_dir)
    holder = os.path.join(lock_dir, bounded_run.HEAVY_LOCK_HOLDER_FILENAME)
    with open(holder, "w", encoding="utf-8") as fh:
        json.dump(_FOREIGN_HOLDER, fh, indent=2)
    with open(holder, "rb") as fh:
        return fh.read()


def _lock_run(lock_dir: str, holder: str, label: str, child_code: str,
              json_path: Optional[str] = None, timeout: float = 45.0,
              wall: float = 180.0) -> Tuple[int, str]:
    """Run the wrapper WITH --heavy-lock against a scratch lock. (rc, stderr)."""

    argv = [PY, WRAPPER, "--label", label, "--timeout", str(timeout), "--quiet",
            "--heavy-lock", holder, "--heavy-lock-path", lock_dir]
    if json_path is not None:
        argv += ["--json", json_path]
    argv += ["--", PY, "-c", child_code]
    proc = subprocess.run(argv, capture_output=True, timeout=wall)
    return proc.returncode, proc.stderr.decode("utf-8", "replace")


#: Writes a marker file, so "did the child run?" is answered by the filesystem
#: rather than by parsing output the wrapper may or may not have forwarded.
_MARKER_CHILD = "open(r'{path}', 'w').write('the child ran')\n"


def case_13_lock_held_blocks_the_job() -> bool:
    """F-072 (a): a held lock must STOP the job, and must survive untouched.

    The exact REV-058 shape: someone else holds the mutex, and a second job goes
    for it. What made that incident a finding was not the failed acquire -- it
    was that the job ran anyway and a trailing unconditional release then cleared
    a lock the agent had never held.
    """

    tmp = tempfile.mkdtemp(prefix="boundedrun_case13_")
    lock_dir = os.path.join(tmp, "scratch_heavylock")
    holder_file = os.path.join(lock_dir, bounded_run.HEAVY_LOCK_HOLDER_FILENAME)
    before = _plant_foreign_lock(lock_dir)
    marker = os.path.join(tmp, "case13_child_ran.txt")
    json_path = os.path.join(tmp, "01-blocked.json")

    rc, err = _lock_run(lock_dir, "C-063-BLOCKED", "c13-lock-held",
                        _MARKER_CHILD.format(path=marker), json_path)
    after = open(holder_file, "rb").read() if os.path.isfile(holder_file) else b""

    ok = _record("13. held lock blocks the job", _blank("c13-lock-held"), [], [
        ("the scratch lock is NOT the real one",
         os.path.normcase(lock_dir)
         != os.path.normcase(os.path.abspath(bounded_run.DEFAULT_HEAVY_LOCK_PATH))),
        ("the CHILD NEVER RAN", not os.path.exists(marker)),
        (f"exit == EXIT_HEAVY_LOCK_UNAVAILABLE "
         f"({bounded_run.EXIT_HEAVY_LOCK_UNAVAILABLE})",
         rc == bounded_run.EXIT_HEAVY_LOCK_UNAVAILABLE),
        ("the lock SURVIVED", os.path.isdir(lock_dir)),
        ("the foreign holder file is byte-identical", after == before and bool(before)),
        ("the condition is NAMED on stderr",
         bounded_run.HEAVY_LOCK_HELD_MARKER in err),
        ("the current holder is printed, so no second command is needed",
         "C-999-SOMEONE-ELSE" in err),
        ("no cleanup report certifies a job that never started",
         not os.path.exists(json_path)),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def case_14_lock_acquired_attributed_released() -> bool:
    """F-072 (b): the wrapper writes the holder file and releases it itself.

    The child reads the holder file WHILE the job runs, so attribution is proved
    from inside the critical section rather than inferred from an artifact
    written after the fact. The agent is never asked to remember either half.
    """

    tmp = tempfile.mkdtemp(prefix="boundedrun_case14_")
    lock_dir = os.path.join(tmp, "scratch_heavylock")
    seen = os.path.join(tmp, "holder_as_the_child_saw_it.json")
    json_path = os.path.join(tmp, "01-acquired.json")
    child = (
        "import shutil\n"
        f"shutil.copyfile(r'{os.path.join(lock_dir, bounded_run.HEAVY_LOCK_HOLDER_FILENAME)}',"
        f" r'{seen}')\n"
    )

    rc, _err = _lock_run(lock_dir, "C-063-OWNER", "c14-lock-ok", child, json_path)
    payload: Dict[str, Any] = {}
    if os.path.isfile(seen):
        with open(seen, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    rep = _as_report(_report_payload(json_path), "c14-lock-ok")
    lock_record = (_report_payload(json_path) or {}).get("heavy_lock") or {}

    ok = _record("14. lock acquired and released", rep, [rep.root_pid or 0], [
        ("the wrapper wrote the holder file, not the agent", bool(payload)),
        ("it names this job's holder", payload.get("holder") == "C-063-OWNER"),
        ("it carries the wrapper's own PID and command",
         isinstance(payload.get("pid"), int) and bool(payload.get("command"))),
        ("the child's REAL exit code is still returned", rc == 0),
        ("the lock was RELEASED on the normal exit path",
         not os.path.exists(lock_dir)),
        ("the report records acquire and release",
         lock_record.get("acquired") is True and lock_record.get("released") is True),
        ("the report records no refusal", lock_record.get("release_refusal") == ""),
        ("cleanup_success", rep.cleanup_success is True),
        ("reported survivors == 0", rep.final_surviving_count == 0),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def case_15_release_refused_for_a_foreign_holder() -> bool:
    """F-072 (c): the wrapper must REFUSE to remove a lock naming someone else.

    The child overwrites the holder file mid-job, so at release time the lock
    names another card. This is the rule that converts the F-072 failure from
    silent to visible: an unconditional release would remove it and nobody would
    ever learn that it had.
    """

    tmp = tempfile.mkdtemp(prefix="boundedrun_case15_")
    lock_dir = os.path.join(tmp, "scratch_heavylock")
    holder_file = os.path.join(lock_dir, bounded_run.HEAVY_LOCK_HOLDER_FILENAME)
    json_path = os.path.join(tmp, "01-refused.json")
    child = (
        "import json\n"
        f"json.dump({_FOREIGN_HOLDER!r}, open(r'{holder_file}', 'w'), indent=2)\n"
    )

    rc, err = _lock_run(lock_dir, "C-063-OWNER", "c15-lock-stolen", child, json_path)
    payload = _report_payload(json_path) or {}
    lock_record = payload.get("heavy_lock") or {}
    still_foreign = False
    if os.path.isfile(holder_file):
        with open(holder_file, "r", encoding="utf-8") as fh:
            still_foreign = json.load(fh).get("holder") == _FOREIGN_HOLDER["holder"]
    rep = _as_report(payload, "c15-lock-stolen")

    ok = _record("15. release refused (foreign)", rep, [rep.root_pid or 0], [
        ("the lock was NOT removed", os.path.isdir(lock_dir)),
        ("it still names the other holder", still_foreign),
        (f"exit == EXIT_HEAVY_LOCK_RELEASE_REFUSED "
         f"({bounded_run.EXIT_HEAVY_LOCK_RELEASE_REFUSED})",
         rc == bounded_run.EXIT_HEAVY_LOCK_RELEASE_REFUSED),
        ("the refusal is NAMED on stderr",
         bounded_run.HEAVY_LOCK_RELEASE_REFUSED_MARKER in err),
        ("the report records the refusal and its reason",
         lock_record.get("released") is False
         and str(lock_record.get("release_refusal", "")).startswith(
             "holder_is_not_this_job")),
        ("the report still records the job itself",
         rep.exit_code == 0 and rep.cleanup_success is True),
        ("reported survivors == 0", rep.final_surviving_count == 0),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def case_16_lock_released_on_the_timeout_path() -> bool:
    """F-072 (d): release runs on EVERY exit path the wrapper controls.

    A hanging child taken down at the outer bound is the path an agent is least
    likely to hand-roll correctly, and the one where a leaked mutex would stall
    the next heavy job with no explanation.
    """

    tmp = tempfile.mkdtemp(prefix="boundedrun_case16_")
    lock_dir = os.path.join(tmp, "scratch_heavylock")
    json_path = os.path.join(tmp, "01-timeout.json")

    rc, _err = _lock_run(lock_dir, "C-063-OWNER", "c16-lock-timeout",
                         "import time; time.sleep(600)\n", json_path, timeout=4.0)
    payload = _report_payload(json_path) or {}
    lock_record = payload.get("heavy_lock") or {}
    rep = _as_report(payload, "c16-lock-timeout")

    ok = _record("16. lock released on timeout", rep, [rep.root_pid or 0], [
        ("the job really did time out", rep.exit_reason == "timeout" and rc == 124),
        ("the lock was released anyway", not os.path.exists(lock_dir)),
        ("the report says so",
         lock_record.get("acquired") is True and lock_record.get("released") is True),
        ("cleanup_success", rep.cleanup_success is True),
        ("reported survivors == 0", rep.final_surviving_count == 0),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def case_17_report_promoted_on_completion() -> bool:
    """F-071 (a): a finished job's report reaches the reports tree, whole.

    Promotion is the half that must not break while fixing the other one: if a
    completed job stopped publishing, every card would look uncertified instead
    of every killed one looking non-compliant.
    """

    tmp = tempfile.mkdtemp(prefix="boundedrun_case17_")
    target = g11_evidence.allocate("C-063", "promoted", root=tmp)
    staged = g11_evidence.staging_path_for(target)
    reserved_before = os.path.isfile(staged) and not os.path.exists(target)

    argv = [PY, WRAPPER, "--label", "c17-promote", "--timeout", "60", "--quiet",
            "--json", target, "--", PY, "-c", "print('c17')"]
    rc = subprocess.run(argv, capture_output=True, timeout=180).returncode
    payload = _report_payload(target) or {}
    rep = _as_report(payload, "c17-promote")

    ok = _record("17. staged report promoted", rep, [rep.root_pid or 0], [
        ("the reservation was staged, not published", reserved_before),
        ("the finished report IS in the reports tree", os.path.isfile(target)),
        ("the staging slot is gone -- a rename, not a copy",
         not os.path.exists(staged)),
        ("it is a full report, not the placeholder",
         g11_evidence.RESERVED_KEY not in payload and len(payload) > 20),
        ("g11 check accepts it", g11_evidence.check_report(target) == []),
        ("the report names its real destination",
         os.path.normcase(payload.get("json_report_path") or "")
         == os.path.normcase(target) and payload.get("json_report_written") is True),
        ("the child's REAL exit code is preserved", rc == 0 and rep.exit_code == 0),
        ("cleanup_success", rep.cleanup_success is True),
        ("reported survivors == 0", rep.final_surviving_count == 0),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def case_18_killed_wrapper_leaves_the_gate_green() -> bool:
    """F-071 (b): kill the wrapper mid-job; the merge gate must stay green.

    This is the F-071 mechanism itself -- four occurrences, every one an agent's
    own wall clock killing the parent before the wrapper could write. The kill is
    of the WRAPPER only, never a tree: the Job Object is what takes the child
    down with it, and this case proves that too.
    """

    tmp = tempfile.mkdtemp(prefix="boundedrun_case18_")
    target = g11_evidence.allocate("C-063", "killed", root=tmp)
    staged = g11_evidence.staging_path_for(target)
    marker = os.path.join(tmp, "c18_child_started.txt")
    child = (f"open(r'{marker}', 'w').write('x')\n"
             "import time; time.sleep(600)\n")

    proc = subprocess.Popen(
        [PY, WRAPPER, "--label", "c18-killed", "--timeout", "300", "--quiet",
         "--json", target, "--", PY, "-c", child],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30.0
    while not os.path.exists(marker) and time.monotonic() < deadline:
        time.sleep(0.2)
    started = os.path.exists(marker)
    descendants = bounded_run.descendants_of(proc.pid,
                                             bounded_run.snapshot_processes())
    proc.kill()          # the parent dies; the wrapper never reaches its report
    proc.wait(timeout=60)
    settle = time.monotonic() + 15.0
    while time.monotonic() < settle:
        table = bounded_run.snapshot_processes()
        if not [pid for pid in descendants + [proc.pid] if pid in table]:
            break
        time.sleep(0.2)

    tree = sorted(os.listdir(os.path.join(tmp, "C-063")))
    ok = _record("18. killed wrapper, gate green", _blank("c18-killed"),
                 descendants + [proc.pid], [
        ("the job really was in flight when it was killed", started),
        ("NOTHING was left in the reports tree",
         tree == [g11_evidence.STAGING_DIRNAME]),
        ("no report at the allocated path", not os.path.exists(target)),
        ("the orphan reservation is in staging, where check cannot see it",
         os.path.isfile(staged)),
        ("whole-tree check is GREEN -- merge gate 10 does not fail",
         g11_evidence.check_many(
             *g11_evidence.resolve_selection([], None, root=tmp)) == 0),
        ("the killed wrapper still took its child down (Job Object)",
         not _alive(descendants)),
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main() -> int:
    print("=" * 74)
    print("bounded_run.py -- INIT-001 Step 0c + H-003 drain + H-006 identity cases")
    print(f"platform={sys.platform}  python={sys.version.split()[0]}  pid={os.getpid()}")
    print(f"wrapper under test: {WRAPPER}")
    # [S8] self-reference: name the wrapper build that produced everything below.
    print(f"wrapper build under test: "
          f"{bounded_run.wrapper_build_identity().get('digest')}")
    print(f"report schema version   : {bounded_run.REPORT_SCHEMA_VERSION}")
    print("=" * 74)

    outcomes = [
        case_1_normal(), case_2_nonzero(), case_3_hang(),
        case_4_grandchild(), case_5_forced(), case_6_cancelled(),
        case_7_unencodable_child_output(), case_8_unwritable_json(),
        case_9_schema_and_build_fields(),
        case_10_identity_stable_and_sensitive(),
        case_11_identity_from_the_executing_copy(),
        case_12_preexisting_reports_still_validate(),
        case_13_lock_held_blocks_the_job(),
        case_14_lock_acquired_attributed_released(),
        case_15_release_refused_for_a_foreign_holder(),
        case_16_lock_released_on_the_timeout_path(),
        case_17_report_promoted_on_completion(),
        case_18_killed_wrapper_leaves_the_gate_green(),
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
