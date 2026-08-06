"""Durable G11 cleanup-evidence: path allocator, compliance checker, selftest.

Bookkeeping tooling for merge gate **G11** (`TEST_MATRIX.md` § 0, shared block
`[S8]`). It does not run jobs and it does not kill anything -- `bounded_run.py`
does that and is **not** modified by this file.

The gap it closes: G11 already required a cleanup report on every test record,
but every agent wrote its ``--json`` into a session scratchpad that is deleted,
then pasted a table nobody could check. This module gives every job a **unique,
version-controlled** report path and a mechanical checker, so "this branch is
G11-compliant" becomes an assertion about committed artifacts.

Usage
-----
::

    # allocate one path per job, BEFORE running it
    <py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py next \\
         --task H-004 --label smoke

    # validate one report, a task's reports, or every committed report
    <py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py check
    <py> ... check --task H-004
    <py> ... check <path.json> [...]

    # prove this specification still matches what bounded_run.py emits
    <py> ... selftest

``next`` and ``check`` are sub-second bookkeeping utilities that spawn no child
process. They are not tests, benchmarks, pipeline legs or LLM-backed commands,
so they are outside the four job classes `[S8]` item 1 names. That is a
statement of scope, not an exemption: nothing that runs pytest, a benchmark, a
pipeline leg or an LLM call may skip the wrapper, and no job may be declared
compliant without a committed report. ``selftest`` **is** a test and runs under
the wrapper like any other.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS_ROOT = HERE
BOUNDED_RUN = os.path.join(os.path.dirname(HERE), "bounded_run.py")

#: Version of THIS specification. The artifact itself carries no schema version
#: (see README "Finding 5"); the required field set below is the structural
#: contract, and ``selftest`` proves it still holds against the live wrapper.
G11_SPEC_VERSION = 1

#: A structured record, never a log dump. 64 KiB is ~40x the size of a real
#: report (1.5 KiB measured) and still far below anything that could be captured
#: child stdout.
MAX_REPORT_BYTES = 65_536

#: ``<task>/<seq>-<label>.json``. Two or more digits so ordering is lexical.
NAME_RE = re.compile(r"^(\d{2,})-([a-z0-9][a-z0-9._-]*)\.json$")
TASK_RE = re.compile(r"^[A-Z0-9]+-\d{3}[a-z]?$")
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

#: Written by ``allocate`` under O_EXCL to reserve the name. A file still
#: holding this key means the job never wrote its report: uncertifiable, never
#: committed, and a hard failure in ``check``.
RESERVED_KEY = "g11_reserved"

VALID_EXIT_REASONS = {
    "completed", "nonzero", "timeout", "cancelled", "infrastructure_failure",
}

#: Required minimum content, keyed by the name ``bounded_run.CleanupReport``
#: actually emits. Verified against live wrapper output by ``selftest``.
REQUIRED_FIELDS: Dict[str, Tuple[type, ...]] = {
    # job identity
    "label": (str,), "command": (list,), "cwd": (str,), "root_pid": (int, type(None)),
    "isolation": (str,),
    # start + completion classification
    "started_at": (str,), "finished_at": (str,), "exit_reason": (str,),
    # the real exit code, and what the wrapper returned
    "exit_code": (int, type(None)), "returned_code": (int, type(None)),
    # owned-process accounting
    "descendants_observed": (list,), "descendants_terminated": (list,),
    "final_surviving_count": (int,), "survivors": (list,), "cleanup_success": (bool,),
    # runtime
    "duration_seconds": (int, float), "timeout_seconds": (int, float),
    # report delivery + free-text record
    "json_report_path": (str,), "json_report_written": (bool,),
    "json_report_error": (str,), "notes": (list,),
}

#: Credential shapes a command line can carry into a committed artifact.
CRED_PATTERNS = [
    ("openai_style_key", re.compile(r"sk-[A-Za-z0-9_\-]{16,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{20,}")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{10,}")),
    ("inline_secret_assignment", re.compile(
        r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|secret|"
        r"password|passwd)\b\s*[=:]\s*[^\s\"',]{6,}")),
    ("credentialed_url", re.compile(r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s/@:]+:[^\s/@]+@")),
]


# --------------------------------------------------------------------------- #
# Allocation -- unique per job, by construction
# --------------------------------------------------------------------------- #


def allocate(task: str, label: str, root: str = REPORTS_ROOT) -> str:
    """Reserve and return the report path for ONE job.

    Uniqueness is not "unlikely to collide", it is structural:

    * the directory is the task ID, and a task owns exactly one branch;
    * the sequence is ``max(existing) + 1`` **and** the file is created with
      ``O_CREAT|O_EXCL``, so two allocations -- even concurrent ones, even for
      the same label -- can never resolve to the same path;
    * sequences are never reused, so a re-run cannot overwrite the report of the
      attempt it is replacing (an infrastructure failure must stay on record).

    No clock is involved: the caller needs no timestamp it cannot obtain.
    """

    if not TASK_RE.match(task):
        raise ValueError(f"task id {task!r} must look like H-004 / C-056a / INIT-001")
    if not LABEL_RE.match(label):
        raise ValueError(f"label {label!r} must match {LABEL_RE.pattern}")
    directory = os.path.join(root, task)
    os.makedirs(directory, exist_ok=True)
    used = [
        int(m.group(1)) for m in
        (NAME_RE.match(n) for n in os.listdir(directory)) if m
    ]
    seq = max(used, default=0) + 1
    while True:
        path = os.path.join(directory, f"{seq:02d}-{label}.json")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            seq += 1
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({RESERVED_KEY: True, "task": task, "label": label}, fh, indent=2)
        return path


# --------------------------------------------------------------------------- #
# Checking -- artifact-first. The job's exit code is never consulted.
# --------------------------------------------------------------------------- #


def check_report(path: str) -> List[str]:
    """Return the violation codes for one report. Empty list means compliant.

    RULE 1: existence is tested here, on the artifact, with no reference to any
    exit code. When ``--json`` is unwritable ``bounded_run.main`` returns the
    *child's* real code, so an exit-code-based check would certify a job that
    wrote no report at all.
    """

    bad: List[str] = []
    name = os.path.basename(path)
    if not NAME_RE.match(name):
        bad.append(f"bad_filename:{name}")
    if not os.path.isfile(path):
        return bad + ["report_missing"]
    size = os.path.getsize(path)
    if size > MAX_REPORT_BYTES:  # RULE 5 (size): a record, not a log dump
        bad.append(f"report_too_large:{size}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        return bad + [f"report_unreadable:{type(exc).__name__}"]
    if not isinstance(data, dict):
        return bad + ["report_not_an_object"]
    if data.get(RESERVED_KEY):
        return bad + ["report_never_written"]  # placeholder survived: no evidence

    # RULE 3: schema
    for field, types in REQUIRED_FIELDS.items():
        if field not in data:
            bad.append(f"missing_field:{field}")
        elif not isinstance(data[field], types) or (
            isinstance(data[field], bool) and bool not in types
        ):
            bad.append(f"bad_type:{field}")
    reason = data.get("exit_reason")
    if isinstance(reason, str) and reason not in VALID_EXIT_REASONS:
        bad.append(f"bad_exit_reason:{reason}")

    # RULE 2: cleanup_success is REQUIRED. final_surviving_count == 0 alone is
    # not sufficient -- that field keeps its 0 default when verification never
    # ran, so a report can read 0 survivors AND cleanup_success false while a
    # child is still alive (observed in H-003 review).
    if data.get("cleanup_success") is not True:
        bad.append("cleanup_not_successful")
    if data.get("final_surviving_count") != 0:
        bad.append(f"survivors:{data.get('final_surviving_count')}")
    if data.get("survivors"):
        bad.append("survivor_list_non_empty")

    # RULE 4: json_report_written is meaningful ONLY when --json was requested.
    # emit_json_report runs from main() alone, so an in-process run() caller
    # (e.g. baseline_suite.py:127) legitimately shows "" / false.
    if data.get("json_report_path"):
        if data.get("json_report_written") is not True:
            bad.append("json_report_declared_but_unwritten")
        if data.get("json_report_error"):
            bad.append("json_report_error_present")

    # RULE 5 (credentials): scan the whole artifact, not just the command.
    for cred_name, pattern in CRED_PATTERNS:
        if pattern.search(raw):
            bad.append(f"possible_credential:{cred_name}")
    return bad


def iter_reports(root: str = REPORTS_ROOT, task: Optional[str] = None) -> List[str]:
    """Every candidate artifact under the evidence root, sorted."""

    out: List[str] = []
    for entry in sorted(os.listdir(root)):
        directory = os.path.join(root, entry)
        if not os.path.isdir(directory) or not TASK_RE.match(entry):
            continue
        if task and entry != task:
            continue
        for name in sorted(os.listdir(directory)):
            if not name.startswith("."):
                out.append(os.path.join(directory, name))
    return out


def check_many(paths: List[str]) -> int:
    failures = 0
    for path in paths:
        if not path.endswith(".json"):
            # RULE 5 (size): captured stdout / logs never belong here.
            print(f"FAIL {path}\n     unexpected_artifact (only .json reports)")
            failures += 1
            continue
        bad = check_report(path)
        if bad:
            failures += 1
            print(f"FAIL {path}\n     " + "\n     ".join(bad))
        else:
            print(f"ok   {path}")
    print(f"\nG11 evidence: {len(paths)} artifact(s), {failures} non-compliant "
          f"(spec v{G11_SPEC_VERSION})")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Selftest -- keeps this specification honest against the real wrapper
# --------------------------------------------------------------------------- #


def test_required_fields_present_in_real_wrapper_output() -> None:
    """(a) Generate a report from a trivial wrapped command; assert the fields.

    Catches specification drift: if ``CleanupReport`` ever loses or renames a
    field this README documents as required, this fails instead of the docs
    quietly becoming fiction.
    """

    with tempfile.TemporaryDirectory(prefix="g11sel_") as tmp:
        out = os.path.join(tmp, "01-fieldcheck.json")
        rc = subprocess.run(
            [sys.executable, BOUNDED_RUN, "--label", "g11-fieldcheck",
             "--timeout", "120", "--quiet", "--json", out,
             "--", sys.executable, "-c", "print('g11')"],
            capture_output=True, text=True, timeout=180,
        ).returncode
        assert rc == 0, f"wrapper returned {rc}"
        assert os.path.isfile(out), "wrapper wrote no --json report"
        with open(out, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        missing = sorted(set(REQUIRED_FIELDS) - set(data))
        assert not missing, f"documented fields absent from real output: {missing}"
        assert not check_report(out), check_report(out)

    sys.path.insert(0, os.path.dirname(BOUNDED_RUN))
    import bounded_run  # noqa: PLC0415 - deliberate late import, read-only
    declared = {f.name for f in dataclasses.fields(bounded_run.CleanupReport)}
    assert not set(REQUIRED_FIELDS) - declared, "spec names a non-existent field"


def test_naming_scheme_cannot_collide() -> None:
    """(b) Two jobs in one task never share a path -- same label included."""

    with tempfile.TemporaryDirectory(prefix="g11nam_") as tmp:
        a = allocate("H-004", "smoke", root=tmp)
        b = allocate("H-004", "smoke", root=tmp)
        c = allocate("H-004", "chunk-d", root=tmp)
        d = allocate("C-010", "smoke", root=tmp)
        assert len({a, b, c, d}) == 4, (a, b, c, d)
        assert all(os.path.isfile(p) for p in (a, b, c, d)), "reservation not durable"
        assert NAME_RE.match(os.path.basename(a)) and NAME_RE.match(os.path.basename(b))
        assert int(NAME_RE.match(os.path.basename(b)).group(1)) > \
            int(NAME_RE.match(os.path.basename(a)).group(1))
        # A reserved-but-unwritten path is evidence of nothing and must fail.
        assert "report_never_written" in check_report(a)


def _write(tmp: str, name: str, **over: Any) -> str:
    base: Dict[str, Any] = {
        "label": "x", "command": ["py", "-c", "pass"], "cwd": tmp, "root_pid": 1,
        "isolation": "posix_process_group", "started_at": "t", "finished_at": "t",
        "exit_reason": "completed", "exit_code": 0, "returned_code": 0,
        "descendants_observed": [], "descendants_terminated": [],
        "final_surviving_count": 0, "survivors": [], "cleanup_success": True,
        "duration_seconds": 0.1, "timeout_seconds": 60.0, "json_report_path": "",
        "json_report_written": False, "json_report_error": "", "notes": [],
    }
    base.update(over)
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(base, fh)
    return path


def test_compliance_rules() -> None:
    """Rules 1, 2, 4 and 5 each rejected/accepted on a purpose-built artifact."""

    with tempfile.TemporaryDirectory(prefix="g11rul_") as tmp:
        # RULE 1: absence is failure, whatever any exit code said.
        assert "report_missing" in check_report(os.path.join(tmp, "01-gone.json"))
        # RULE 2: the exact H-003 shape -- 0 survivors, cleanup_success false.
        assert "cleanup_not_successful" in check_report(
            _write(tmp, "01-a.json", cleanup_success=False, final_surviving_count=0))
        # RULE 4: an in-process run() caller is compliant with "" / false.
        assert not check_report(_write(tmp, "02-b.json"))
        # ... but a --json report claiming a path it never reached is not.
        assert "json_report_declared_but_unwritten" in check_report(
            _write(tmp, "03-c.json", json_report_path="/x", json_report_written=False))
        # RULE 5: a credential in the command line blocks the commit.
        assert any(v.startswith("possible_credential") for v in check_report(
            _write(tmp, "04-d.json",
                   command=["py", "run.py", "--api-key=sk-" + "A" * 24])))
        # RULE 3: an unknown exit classification is not a valid record.
        assert "bad_exit_reason:unknown" in check_report(
            _write(tmp, "05-e.json", exit_reason="unknown"))


TESTS = [
    test_required_fields_present_in_real_wrapper_output,
    test_naming_scheme_cannot_collide,
    test_compliance_rules,
]


def selftest() -> int:
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(TESTS) - failed} passed, {failed} failed")
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    nxt = sub.add_parser("next", help="reserve a unique report path for one job")
    nxt.add_argument("--task", required=True)
    nxt.add_argument("--label", required=True)
    chk = sub.add_parser("check", help="validate committed reports")
    chk.add_argument("--task", default=None)
    chk.add_argument("paths", nargs="*")
    sub.add_parser("selftest", help="prove this spec matches bounded_run.py")
    args = parser.parse_args(argv)

    if args.cmd == "next":
        print(allocate(args.task, args.label))
        return 0
    if args.cmd == "selftest":
        return selftest()
    return check_many(args.paths or iter_reports(task=args.task))


if __name__ == "__main__":
    raise SystemExit(main())
