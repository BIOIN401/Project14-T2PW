"""Bounded foreground process wrapper for the PWML recovery sprint.

Orchestration tooling, **not** pipeline code. Nothing under ``src/`` imports this
and this imports nothing from ``src/``. It exists to satisfy merge gate **G11**
(``TEST_MATRIX.md`` § 0, shared block ``[S8]``): every test, benchmark, pipeline
leg and LLM-backed command in this sprint runs through it, and a run that leaves
a surviving owned process is an *infrastructure failure*, not a test result.

Why it is not ``batch/runner.py``
---------------------------------
``runner.launch_child`` / ``_kill_tree`` (``runner.py:1107-1180``) already have the
**correct ownership model** -- they target only the PID they created and are not
global killers. Audited and confirmed. What they lack against ``[S8]``:

* cleanup runs only on the ``TimeoutExpired`` and ``KeyboardInterrupt`` paths.
  There is no ``finally``, so any other exception leaves the child alive.
* termination goes straight to ``taskkill /F`` -- no graceful stage.
* no post-kill survivor verification; no structured cleanup report.
* ``CREATE_NEW_PROCESS_GROUP`` is *not* a Job Object. It does not guarantee that
  descendants die with the root.
* it is used for batch legs only -- never for pytest runs, which are the largest
  memory consumers in this sprint.

``runner.py`` is owned by branch C-032 and **must not be modified here**. This is a
separate, orchestration-only wrapper that reuses the same ownership discipline.

Guarantees
----------
1. The child is placed in an isolated container: a Windows **Job Object** with
   ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, or a POSIX process group.
2. An outer wall-clock timeout is enforced by the parent.
3. Cleanup runs in ``finally`` on **every** exit path -- success, nonzero,
   timeout, ``KeyboardInterrupt``, ``SystemExit``, or any other exception.
4. Termination is graceful first, forced after a grace period, and the report
   records whether the escalation was needed.
5. Survivors are **verified** by re-snapshotting the process table, not assumed.
6. The child's real exit code is returned unless cleanup verification itself
   failed.
7. Arbitrary child output cannot crash the parent-side drain: forwarding is
   encoded against the destination stream's own encoding, never assuming a UTF-8
   console and never depending on ``PYTHONIOENCODING`` (H-003 / D-017).
8. A writable ``--json`` destination receives the report on **every** exit path,
   including one where an exception escapes the wait loop; an unwritable one is a
   distinct reported condition, never a silent skip.
9. Every report states its own ``schema_version`` and the ``wrapper_build`` that
   produced it -- a digest of the *executing* module, so the artifact answers
   ``[S8]``'s self-reference question without git archaeology (H-006).
10. ``--heavy-lock`` makes "one heavy job at a time" an enforced invariant rather
    than a shell protocol every agent re-implements. A failed acquire **does not
    run the child**; a release **refuses** to remove a lock whose holder file
    does not name this exact job (F-072).
11. A ``--json`` destination that a G11 reservation is waiting behind is written
    through **staging and one atomic promotion**, so a job killed before it
    finishes leaves nothing in the reports tree at all (F-071).

Two process defects, one root cause
-----------------------------------
F-071 and F-072 are both cases of an agent hand-rolling infrastructure in shell
and getting it wrong a new way each time: an unconditional ``rm -rf`` after a
failed ``mkdir`` cleared a live holder's mutex, and a wall-clock kill left a
reservation that turned the merge gate red on a job that produced no result
either way. Both primitives now live here, next to the process lifecycle this
module already owns, because the shell cannot be relied on to test a result it
was never made to look at.

**Neither is on by default.** ``--heavy-lock`` is opt-in and every call site that
does not pass it behaves exactly as before; promotion engages only when a real
``g11_reserved`` reservation is sitting at the staging path, so a ``--json``
scratch path is still written straight through.

Forbidden here, permanently: ``taskkill /IM``, ``pkill``, or any kill by image
name. Cleanup targets only PIDs this job created. Processes that already existed
when the job started are **reported**, never killed.

Usage
-----
::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \\
        --label smoke --timeout 900 -- \\
        .venv/Scripts/python.exe -m pytest -q --basetemp=<tmp>/smoke tests/...

A heavy job takes the mutex in the same command, and cannot get past a failed
acquire::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \\
        --label chunk-d --timeout 900 --heavy-lock C-063 \\
        --json <path allocated by g11_evidence.py next> -- <command>

The process exit code is the child's own -- except on the two mutex conditions,
which are infrastructure and not test results: :data:`EXIT_HEAVY_LOCK_UNAVAILABLE`
(the lock was held; the child never started) and
:data:`EXIT_HEAVY_LOCK_RELEASE_REFUSED` (the job ran but its lock now names
someone else, so it was left alone). ``--json <path>`` also writes the structured
cleanup report required on every test record.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

IS_WINDOWS = os.name == "nt"

#: Exit code returned when the child finished but cleanup verification failed.
#: Distinct from any plausible pytest code so it can never be misread as a test
#: result. ``[S8]`` item 6: survivors are an infrastructure failure.
EXIT_INFRASTRUCTURE_FAILURE = 97

#: Seconds between "ask nicely" and "force". Deliberately short: this is cleanup
#: after the job is already over, not a shutdown hook worth waiting on.
DEFAULT_GRACE_SECONDS = 3.0

#: How long to keep re-checking the process table for survivors after forcing.
_SURVIVOR_SETTLE_SECONDS = 2.0
_POLL_INTERVAL = 0.1

#: Emitted on stderr, and recorded in ``notes``, when a ``--json`` destination
#: could not be written. A missing cleanup report makes a run uncertifiable under
#: G11, so the condition is announced loudly rather than skipped silently.
JSON_REPORT_UNWRITABLE_MARKER = "BOUNDED_RUN_JSON_REPORT_UNWRITABLE"

# --------------------------------------------------------------------------- #
# G11 report promotion  (F-071)
#
# ``g11_evidence.allocate`` reserves a report path by writing a placeholder --
# that placeholder is what stops an agent hand-writing a ``--json`` path and
# abandoning the audit trail, and it is not going away. It used to sit in the
# reports tree, where a wrapper killed before it could write left it behind with
# three keys instead of ~28: whole-tree ``check`` went red and merge gate 10
# failed on a job that produced no result either way. Four occurrences.
#
# The reservation now waits one directory deeper, in the task's staging
# directory, and this module promotes it. The two constants below ARE the
# contract; ``g11_evidence.test_staging_contract_matches_the_wrapper`` fails if
# either side drifts. Nothing here imports ``g11_evidence``: this wrapper stays
# usable with no evidence tree at all, and a ``--json`` path with no reservation
# behind it is written straight through exactly as before.
# --------------------------------------------------------------------------- #

#: Mirror of ``g11_evidence.STAGING_DIRNAME``.
G11_STAGING_DIRNAME = ".staging"

#: Mirror of ``g11_evidence.RESERVED_KEY``. Only a file that really carries it is
#: treated as a reservation, so an unrelated sibling can never divert a report.
G11_RESERVED_KEY = "g11_reserved"

#: A report written to staging but not promoted is invisible to ``check`` -- so
#: the failure is announced here instead of silently producing no evidence.
G11_PROMOTION_FAILED_MARKER = "BOUNDED_RUN_G11_PROMOTION_FAILED"

# --------------------------------------------------------------------------- #
# Heavy mutex  (F-072)
#
# ``mkdir`` on an existing directory is the CORRECT primitive: atomic, and it
# fails when held. The defect was never the primitive, it was how a shell
# consumed its failure -- ``mkdir X && echo ACQUIRED`` suppressed only the echo,
# the following statements ran anyway, and a trailing unconditional ``rm -rf``
# cleared a lock its holder had never acquired. A rule against deliberately
# clearing someone's lock does not help: the agent believed it held the lock it
# was releasing.
#
# So the two-phase protocol moves in here, where a failed acquire can stop the
# job by construction rather than by the caller remembering to test a result.
# --------------------------------------------------------------------------- #

#: The sprint's "one heavy job at a time" mutex. A DIRECTORY, because ``mkdir``
#: is atomic. ``--heavy-lock-path`` overrides it so a test can exercise the
#: protocol against a scratch lock and never touch the real one.
DEFAULT_HEAVY_LOCK_PATH = "C:/t/heavylock" if IS_WINDOWS else "/tmp/heavylock"

#: Written by the wrapper on acquire, read by the wrapper on release. The agent
#: never has to remember to create it, which is the point.
HEAVY_LOCK_HOLDER_FILENAME = "holder.json"

#: Cap on how much of a foreign holder file is copied into a report or printed.
HEAVY_LOCK_HOLDER_MAX_BYTES = 4096

#: The lock was held by someone else: the child was NEVER STARTED.
#:
#: The two codes below are chosen against the sprint's reserved set, so no caller
#: can misread either as a test result or as another tool's verdict: pytest's
#: 0-5, ``runner.EXIT_PREFLIGHT`` 3, :data:`EXIT_INFRASTRUCTURE_FAILURE` 97,
#: ``tree_pin.EXIT_MEASUREMENT_TREE_REFUSED`` **98**, 124 (timeout) and 130
#: (cancelled). 98 in particular is already spoken for and must not be reused.
#: No ``--json`` report is written on this path -- see :func:`main`.
EXIT_HEAVY_LOCK_UNAVAILABLE = 95

#: The job ran, but the lock could not be released because its holder file no
#: longer names this job. The lock is left ALONE and the condition is reported;
#: clearing another holder's lock is the orchestrator's decision alone.
EXIT_HEAVY_LOCK_RELEASE_REFUSED = 96

HEAVY_LOCK_HELD_MARKER = "BOUNDED_RUN_HEAVY_LOCK_HELD"
HEAVY_LOCK_RELEASE_REFUSED_MARKER = "BOUNDED_RUN_HEAVY_LOCK_RELEASE_REFUSED"


def _forward_text(stream: Any, text: str) -> None:
    """Write *text* to *stream* without letting the console encoding raise.

    The child's output is arbitrary bytes, read back with ``errors="replace"``
    -- which can itself *introduce* U+FFFD -- and a cp1252 console cannot encode
    that. ``TextIOWrapper.write`` then raises ``UnicodeEncodeError``, a
    ``ValueError``, from inside the drain. The encoding is therefore handled
    *here*, at the point of forwarding: no global interpreter state is mutated
    and no console configuration is assumed.

    Encodability is tested *before* the write, not caught after it, so a
    tee-style stream that already committed the chunk to a log file is never
    handed it twice. Text the console can take is written verbatim; only text it
    cannot is ``backslashreplace``d, so nothing is silently dropped. Never
    raises: an unwritable console is no reason to abandon a job or its report.
    """

    if not text:
        return
    # A wrapper stream (e.g. baseline_suite's _Tee) may declare no encoding of
    # its own while ultimately writing to the real console; ask that console.
    encoding = (
        getattr(stream, "encoding", None)
        or getattr(sys.__stdout__, "encoding", None)
        or "utf-8"
    )
    payload = text
    try:
        text.encode(encoding)
    except (UnicodeError, LookupError):
        try:
            payload = text.encode(encoding, errors="backslashreplace").decode(encoding)
        except Exception:  # noqa: BLE001 - unknown/stateful codec: ASCII is universal
            payload = text.encode("ascii", errors="backslashreplace").decode("ascii")
    except Exception:  # noqa: BLE001 - a stream lying about its encoding
        payload = text

    try:
        stream.write(payload)
        stream.flush()
        return
    except UnicodeEncodeError:
        pass  # the declared encoding was not the real one: escape everything
    except Exception:  # noqa: BLE001 - closed/detached stream: nothing to forward to
        return
    try:
        stream.write(text.encode("ascii", errors="backslashreplace").decode("ascii"))
        stream.flush()
    except Exception:  # noqa: BLE001
        return


# --------------------------------------------------------------------------- #
# Process-table access (no psutil in this venv -- ctypes on Windows, /proc on
# POSIX). Used for BOTH descendant discovery and survivor verification, so a
# survivor claim is measured rather than inferred.
# --------------------------------------------------------------------------- #

if IS_WINDOWS:  # pragma: no cover - platform branch
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def snapshot_processes() -> Dict[int, Tuple[int, str]]:
        """``{pid: (ppid, image_name)}`` for every visible process."""

        snap = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == INVALID_HANDLE_VALUE or not snap:
            return {}
        out: Dict[int, Tuple[int, str]] = {}
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            if not _kernel32.Process32FirstW(snap, ctypes.byref(entry)):
                return {}
            while True:
                out[int(entry.th32ProcessID)] = (
                    int(entry.th32ParentProcessID),
                    str(entry.szExeFile),
                )
                if not _kernel32.Process32NextW(snap, ctypes.byref(entry)):
                    break
        finally:
            _kernel32.CloseHandle(snap)
        return out

else:  # POSIX

    def snapshot_processes() -> Dict[int, Tuple[int, str]]:
        """``{pid: (ppid, comm)}`` read from ``/proc``."""

        out: Dict[int, Tuple[int, str]] = {}
        proc_root = "/proc"
        if not os.path.isdir(proc_root):
            return out
        for name in os.listdir(proc_root):
            if not name.isdigit():
                continue
            try:
                with open(f"{proc_root}/{name}/stat", "r", encoding="utf-8") as fh:
                    raw = fh.read()
                # comm may contain spaces and parentheses; split on the LAST ')'.
                close = raw.rindex(")")
                comm = raw[raw.index("(") + 1 : close]
                rest = raw[close + 2 :].split()
                out[int(name)] = (int(rest[1]), comm)
            except (OSError, ValueError, IndexError):
                continue
        return out


def descendants_of(root_pid: int, table: Dict[int, Tuple[int, str]]) -> List[int]:
    """Every transitive child of *root_pid* in *table*, excluding the root."""

    children: Dict[int, List[int]] = {}
    for pid, (ppid, _name) in table.items():
        children.setdefault(ppid, []).append(pid)
    found: List[int] = []
    stack = list(children.get(root_pid, ()))
    seen = {root_pid}
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        found.append(pid)
        stack.extend(children.get(pid, ()))
    return sorted(found)


# --------------------------------------------------------------------------- #
# Report schema version  (H-006)
# --------------------------------------------------------------------------- #

#: Version of the REPORT CONTRACT -- the set, types and meanings of the fields a
#: consumer reads out of a cleanup report. Distinct from the wrapper build
#: identity below on purpose: this number describes the SHAPE of the record, the
#: digest describes the CODE that produced it. Neither substitutes for the other.
#:
#: BUMP DISCIPLINE. Bump when a consumer that validated against version N could
#: MISREAD an N+1 report: a field is removed, renamed, or retyped; a field's
#: meaning changes under an unchanged name (including the value domain of
#: ``exit_reason`` or ``isolation``); a field's units change.
#: DO NOT bump for: ADDING a field (additive change is compatible -- a consumer
#: ignores keys it does not know); any change to cleanup, termination or
#: survivor-verification BEHAVIOUR, or to comments, docstrings and internals
#: (that is ``wrapper_build.digest``'s job, and it moves on EVERY content
#: change); or new *values* a field's documented meaning already permits.
#:
#: Version 1 is the first to carry the field at all. A report without it came from
#: a pre-H-006 wrapper: it is schema 0, still valid (:func:`validate_report_schema`
#: treats absence as valid), and it must NEVER be backfilled -- a reconstruction is
#: not evidence of the original run.
REPORT_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Wrapper build identity  (H-006).  Rationale in full: evidence/g11/README.md.
#
# ``command`` names the CHILD; nothing named the WRAPPER, so ``[S8]``'s
# self-reference obligation could not be met FROM the artifact. The substitute was
# git archaeology (``git log -1 -- <report>`` -> the wrapper in that tree), which
# fails three ways. The identity below is instead a SHA-256 over the RAW BYTES of
# the module ACTUALLY EXECUTING, read via ``__file__`` -- never from the tree,
# from git, or from an environment variable a caller can set:
#   (a) CROSS-CHECKOUT EXECUTION (one tree's wrapper run while committing on
#       another's branch): defeated -- digest and path are the file's that ran,
#       whichever checkout it came from. Selftest case 11 proves it.
#   (b) REBASE / SQUASH moving the commit ``git log -1`` resolves to: defeated --
#       a content digest is not a commit reference and does not move.
#   (c) A STALE WRAPPER run before the commit: defeated -- it hashes to its own
#       stale content, which cannot match the wrapper the commit carries.
# Raw bytes, so two checkouts whose line endings differ are correctly different
# builds. Repository SHA and dirty state are recorded IN ADDITION, resolved from
# the WRAPPER's own directory (never the caller's cwd, which may be another
# checkout). They are context, never identity: a wrapper can be modified relative
# to HEAD or live outside any repository, and ``repo_head`` would then name bytes
# other than the ones that ran.
# --------------------------------------------------------------------------- #

#: Bound on each read-only ``git`` metadata call. Exceeding it degrades the
#: repository context to "unavailable"; it never blocks or fails a job.
GIT_METADATA_TIMEOUT_SECONDS = 20.0

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256_file(path: str) -> Tuple[str, int, str]:
    """``(hexdigest, size_bytes, error)`` for *path*. Never raises."""

    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as fh:
            while True:
                block = fh.read(65536)
                if not block:
                    break
                size += len(block)
                digest.update(block)
    except OSError as exc:
        return "", 0, f"{type(exc).__name__}: {exc}"
    return digest.hexdigest(), size, ""


def _git(cwd: str, *args: str) -> Tuple[Optional[str], str]:
    """Run one read-only ``git`` command in *cwd*. ``(stdout, error)``.

    ``core.fsmonitor=false`` stops git from spawning a background file-system
    monitor daemon that would outlive this process: a wrapper that leaks a
    process while recording its own provenance would be self-defeating.
    """

    try:
        proc = subprocess.run(  # noqa: S603 - fixed read-only argv
            ["git", "-c", "core.fsmonitor=false", "-C", cwd, *args],
            capture_output=True, text=True, check=False,
            timeout=GIT_METADATA_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - git absent, or slower than the bound
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()
        return None, lines[0] if lines else f"git exited {proc.returncode}"
    return proc.stdout, ""


def _repository_facts(wrapper_path: str) -> Dict[str, Any]:
    """Repository context for the executing wrapper, resolved from ITS directory.

    ``repo_head`` is never wrapper identity on its own. ``wrapper_vs_head`` says
    whether the executing bytes match what that commit carries, and
    ``repo_tracked_files_dirty`` says whether the checkout as a whole was clean,
    so a reader can never mistake the SHA for a promise about the bytes that ran.
    """

    facts: Dict[str, Any] = {
        "repo_root": "",
        "repo_head": "",
        "repo_source": "unavailable",
        "wrapper_vs_head": "unknown",
        "repo_tracked_files_dirty": None,
        "repo_error": "",
    }
    directory = os.path.dirname(wrapper_path) or "."
    out, err = _git(directory, "rev-parse", "HEAD", "--show-toplevel")
    if out is None:
        facts["repo_source"] = "not_a_repository"
        facts["repo_error"] = err
        return facts
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if len(lines) < 2:
        facts["repo_error"] = "unexpected rev-parse output"
        return facts
    facts["repo_head"], facts["repo_root"] = lines[0], os.path.normpath(lines[1])
    facts["repo_source"] = "git"

    status, err = _git(directory, "status", "--porcelain", "--", wrapper_path)
    if status is None:
        facts["repo_error"] = err
    else:
        # The porcelain status code is COLUMN-significant (" M" unstaged, "M "
        # staged), so the leading space must survive: never ``strip()`` it away.
        rows = [line for line in status.splitlines() if line.strip()]
        code = rows[0][:2] if rows else ""
        if not code:
            facts["wrapper_vs_head"] = "clean"
        elif code == "??":
            facts["wrapper_vs_head"] = "untracked"
        else:
            facts["wrapper_vs_head"] = f"modified:{code!r}"

    dirty, err = _git(directory, "status", "--porcelain", "--untracked-files=no")
    if dirty is None:
        facts["repo_error"] = facts["repo_error"] or err
    else:
        facts["repo_tracked_files_dirty"] = bool(dirty.strip())
    return facts


def compute_wrapper_build() -> Dict[str, Any]:
    """Identify the wrapper build that is running, from the running module."""

    path = os.path.abspath(__file__)
    hexdigest, size, error = _sha256_file(path)
    build: Dict[str, Any] = {
        "digest": f"sha256:{hexdigest}" if hexdigest else "",
        "digest_algorithm": "sha256",
        "digest_scope": "raw_bytes_of_the_executing_module_file",
        "path": path,
        "size_bytes": size,
        "digest_error": error,
    }
    build.update(_repository_facts(path))
    return build


_WRAPPER_BUILD: Optional[Dict[str, Any]] = None


def wrapper_build_identity() -> Dict[str, Any]:
    """Cached per process. A fresh dict each call, so no report can mutate it.

    Resolved when the first :class:`CleanupReport` is constructed -- which is
    before ``run()`` snapshots the pre-existing process table -- so the
    short-lived ``git`` children it needs are finished and gone before ownership
    is established, and can neither be mistaken for nor interfere with the job's
    owned descendants.
    """

    global _WRAPPER_BUILD
    if _WRAPPER_BUILD is None:
        _WRAPPER_BUILD = compute_wrapper_build()
    return dict(_WRAPPER_BUILD)


def validate_report_schema(payload: Dict[str, Any]) -> List[str]:
    """Validate the H-006 identity fields of a decoded report. ``[]`` == valid.

    ABSENCE IS VALID. A report with no ``schema_version`` came from a wrapper
    build predating the field; it is schema 0 and it is not defective. Backfilling
    one would be a reconstruction, not evidence, and is forbidden.
    """

    bad: List[str] = []
    if "schema_version" not in payload:
        if "wrapper_build" in payload:
            bad.append("wrapper_build_without_schema_version")
        return bad
    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        return bad + [f"schema_version_not_an_integer:{type(version).__name__}"]
    if version < 1:
        bad.append(f"schema_version_out_of_range:{version}")
    if version > REPORT_SCHEMA_VERSION:
        bad.append(f"schema_version_from_a_newer_wrapper:{version}")
    build = payload.get("wrapper_build")
    if not isinstance(build, dict):
        return bad + ["wrapper_build_missing_or_not_an_object"]
    digest = build.get("digest")
    if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
        bad.append(f"wrapper_build_digest_malformed:{digest!r}")
    if not isinstance(build.get("path"), str) or not build.get("path"):
        bad.append("wrapper_build_path_missing")
    return bad


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class CleanupReport:
    """The record ``[S8]`` item 9 requires on every test entry."""

    label: str
    command: List[str]
    cwd: str
    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0
    timeout_seconds: float = 0.0
    root_pid: Optional[int] = None
    isolation: str = "none"
    process_group: Optional[int] = None
    exit_reason: str = "unknown"
    exit_code: Optional[int] = None
    returned_code: Optional[int] = None
    descendants_observed: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    descendants_terminated: List[int] = dataclasses.field(default_factory=list)
    graceful_attempted: bool = False
    forced: bool = False
    final_surviving_count: int = 0
    survivors: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    preexisting_reported: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    cleanup_success: bool = False
    notes: List[str] = dataclasses.field(default_factory=list)
    #: Where the structured report was asked to go, and whether it got there.
    #: A run whose report was never written is uncertifiable under G11, so the
    #: outcome is part of the record rather than an unobservable side effect.
    json_report_path: str = ""
    json_report_written: bool = False
    json_report_error: str = ""
    #: Version of the report contract this record was written against. See
    #: :data:`REPORT_SCHEMA_VERSION` for the bump discipline. Its absence in an
    #: artifact means schema 0 -- a pre-H-006 wrapper -- not a defect.
    schema_version: int = REPORT_SCHEMA_VERSION
    #: WHICH WRAPPER BUILD produced this record, digested from the running module.
    #: ``command`` names the child; this names the wrapper.
    wrapper_build: Dict[str, Any] = dataclasses.field(
        default_factory=wrapper_build_identity
    )
    #: Heavy-mutex outcome, or ``{}`` when ``--heavy-lock`` was not used (F-072).
    #: ADDITIVE: a consumer that never heard of it reads the record exactly as
    #: before, so :data:`REPORT_SCHEMA_VERSION` does not move for it.
    heavy_lock: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def render(self) -> str:
        build = self.wrapper_build if isinstance(self.wrapper_build, dict) else {}
        lines = [
            "",
            "================ CLEANUP REPORT ================",
            f"report schema version   : {self.schema_version}",
            f"wrapper build           : {build.get('digest', '')}",
            f"wrapper path (executed) : {build.get('path', '')}",
            f"wrapper vs HEAD         : {build.get('wrapper_vs_head', 'unknown')}"
            f"  @ {build.get('repo_head', '') or '(no repository)'}"
            f"  [{build.get('repo_source', 'unavailable')}"
            f", worktree dirty={build.get('repo_tracked_files_dirty')}]",
            f"label                   : {self.label}",
            f"command                 : {' '.join(self.command)}",
            f"cwd                     : {self.cwd}",
            f"root PID / process group: {self.root_pid} / {self.process_group}",
            f"isolation               : {self.isolation}",
            f"timeout                 : {self.timeout_seconds:g} s",
            f"duration                : {self.duration_seconds:.2f} s",
            f"exit reason             : {self.exit_reason}",
            f"exit code (real)        : {self.exit_code}",
            f"returned code           : {self.returned_code}",
            f"descendants observed    : {len(self.descendants_observed)}"
            + (
                "  " + ", ".join(f"{d['pid']}:{d['name']}" for d in self.descendants_observed)
                if self.descendants_observed
                else ""
            ),
            f"descendants terminated  : {len(self.descendants_terminated)}"
            + ("  " + ", ".join(str(p) for p in self.descendants_terminated)
               if self.descendants_terminated else ""),
            f"graceful attempted      : {self.graceful_attempted}",
            f"forced                  : {self.forced}",
            f"FINAL SURVIVING COUNT   : {self.final_surviving_count}",
            f"cleanup                 : {'success' if self.cleanup_success else 'FAILURE'}",
            f"json report             : {self.json_report_path or '(not requested)'}",
            f"json report written     : {self.json_report_written}",
        ]
        if self.json_report_error:
            lines.append(f"json report ERROR       : {self.json_report_error}")
        # Only when the mutex was actually used, so no existing caller's output
        # -- or the regexes that parse it -- changes shape (F-072).
        lock = self.heavy_lock if isinstance(self.heavy_lock, dict) else {}
        if lock.get("requested"):
            lines.append(
                f"heavy lock              : {lock.get('path', '')}"
                f"  holder={lock.get('holder', '')}"
                f"  acquired={lock.get('acquired')}  released={lock.get('released')}"
            )
            if lock.get("release_refusal"):
                lines.append(
                    f"heavy lock NOT RELEASED : {HEAVY_LOCK_RELEASE_REFUSED_MARKER} "
                    f"{lock['release_refusal']}"
                )
        for surv in self.survivors:
            lines.append(f"  SURVIVOR pid={surv.get('pid')} name={surv.get('name')} "
                         f"ppid={surv.get('ppid')} rss={surv.get('rss_mb')}")
        if self.preexisting_reported:
            lines.append(
                f"pre-existing (reported, NEVER killed): {len(self.preexisting_reported)}"
            )
        for note in self.notes:
            lines.append(f"note                    : {note}")
        lines.append("================================================")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Heavy mutex -- acquire, attribute, refuse  (F-072)
# --------------------------------------------------------------------------- #


class HeavyLock:
    """The sprint heavy mutex, held for the lifetime of one wrapped job.

    Three properties, each answering one half of F-072:

    1. **Acquire is atomic and its failure is fatal.** ``os.mkdir`` either
       creates the directory or raises ``FileExistsError``; :func:`main` does not
       spawn the child unless :meth:`acquire` returned ``True``. There is no
       compound statement for a failure to slip past, because there is no
       statement -- the caller cannot express "run anyway".
    2. **The holder file is the wrapper's job, not the agent's.** It is written
       immediately after the directory exists, so a lock this class created is
       always attributable. If it cannot be written the lock is given straight
       back rather than left held by nobody.
    3. **Release is conditional on identity.** :attr:`token` carries a random
       nonce as well as holder and PID, so "does this lock name me?" is decided
       by an exact match and not by a name two jobs of the same card would share.
       Anything else -- a foreign token, a vanished holder file, an unreadable
       one, a stray file inside the lock -- is a REFUSAL to remove. Clearing
       another holder's lock is the orchestrator's decision alone, so the wrapper
       reports the condition and leaves the lock exactly as it found it.

    Not covered, and deliberately so: a wrapper whose own process is killed
    outright leaves the lock held. USUALLY it is attributable -- the holder file
    names the job that died -- and that case is the one worth having. It is NOT
    always: the ``holder_file_vanished`` branch in :meth:`release` refuses a lock
    whose holder file is gone, and such a lock is **exactly as anonymous as the
    one the shell protocol left**. The refusal is still right -- a lock this job
    cannot identify is not a lock it may clear -- but no general claim to being
    better off than the shell protocol survives that branch. Either way this is
    not an automatic release and must not be mistaken for one.
    """

    def __init__(self, holder: str, path: str, label: str = "",
                 command: Sequence[str] = ()) -> None:
        self.holder = holder
        self.path = os.path.abspath(path)
        self.holder_file = os.path.join(self.path, HEAVY_LOCK_HOLDER_FILENAME)
        self.label = label
        self.command = [str(part) for part in command]
        self.token = f"{holder}:{os.getpid()}:{os.urandom(8).hex()}"
        self.acquired = False
        self.released = False
        self.acquire_error = ""
        self.release_refusal = ""
        self.holder_seen = ""

    def read_holder_text(self) -> str:
        """Whatever the holder file says right now. Never raises."""

        try:
            with open(self.holder_file, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(HEAVY_LOCK_HOLDER_MAX_BYTES)
        except OSError as exc:
            return f"<holder file unreadable: {type(exc).__name__}: {exc}>"

    def acquire(self) -> bool:
        """Take the lock, or return ``False`` having recorded who holds it."""

        try:
            os.mkdir(self.path)
        except FileExistsError:
            self.acquire_error = "held"
            self.holder_seen = self.read_holder_text()
            return False
        except OSError as exc:
            self.acquire_error = f"{type(exc).__name__}: {exc}"
            return False
        payload = {
            "token": self.token,
            "holder": self.holder,
            "label": self.label,
            "pid": os.getpid(),
            "command": self.command,
            "cwd": os.getcwd(),
            "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            with open(self.holder_file, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, indent=2))
        except OSError as exc:
            # We know we created this directory, so removing it is not clearing
            # anyone's lock. A lock nobody can attribute is worse than no lock:
            # the next holder could never be told from a stale one.
            self.acquire_error = f"holder file unwritable: {type(exc).__name__}: {exc}"
            try:
                os.rmdir(self.path)
            except OSError:
                self.acquire_error += " (and the lock directory could not be removed)"
            return False
        self.acquired = True
        return True

    def release(self) -> bool:
        """Remove the lock ONLY if its holder file still names this job."""

        if not self.acquired:
            return True
        try:
            with open(self.holder_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            self.release_refusal = "holder_file_vanished"
            return False
        except (OSError, ValueError) as exc:
            self.release_refusal = f"holder_file_unreadable:{type(exc).__name__}"
            self.holder_seen = self.read_holder_text()
            return False
        if not isinstance(data, dict) or data.get("token") != self.token:
            self.holder_seen = self.read_holder_text()
            named = data.get("holder") if isinstance(data, dict) else None
            self.release_refusal = f"holder_is_not_this_job:{named!r}"
            return False
        try:
            stray = sorted(set(os.listdir(self.path)) - {HEAVY_LOCK_HOLDER_FILENAME})
        except OSError as exc:
            self.release_refusal = f"lock_unlistable:{type(exc).__name__}: {exc}"
            return False
        if stray:
            self.release_refusal = f"lock_directory_has_foreign_content:{stray}"
            return False
        try:
            os.unlink(self.holder_file)
            os.rmdir(self.path)
        except OSError as exc:
            self.release_refusal = f"remove_failed:{type(exc).__name__}: {exc}"
            return False
        self.released = True
        return True

    def snapshot(self) -> Dict[str, Any]:
        """The record of this lock's whole life, for the cleanup report."""

        return {
            "requested": True,
            "path": self.path,
            "holder": self.holder,
            "token": self.token,
            "holder_file": self.holder_file,
            "acquired": self.acquired,
            "released": self.released,
            "acquire_error": self.acquire_error,
            "release_refusal": self.release_refusal,
            "holder_seen": self.holder_seen[:HEAVY_LOCK_HOLDER_MAX_BYTES],
        }


# --------------------------------------------------------------------------- #
# Windows Job Object
# --------------------------------------------------------------------------- #


class _JobObject:
    """A Windows Job Object set to kill its members when the handle closes.

    Assignment happens immediately after ``Popen`` returns. The interpreter has
    not finished initialising by then, so it cannot have spawned anything that
    escapes. If assignment fails the caller falls back to the process-group +
    owned-PID model and the report records that.
    """

    def __init__(self, name_hint: str) -> None:  # pragma: no cover - platform branch
        self.handle = None
        self.ok = False
        if not IS_WINDOWS:
            return
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _kernel32.SetInformationJobObject(
            ctypes.c_void_p(handle),
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            _kernel32.CloseHandle(ctypes.c_void_p(handle))
            return
        self.handle = handle
        self.ok = True

    def assign(self, pid: int) -> bool:  # pragma: no cover - platform branch
        if not self.ok:
            return False
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        hproc = _kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not hproc:
            return False
        try:
            return bool(
                _kernel32.AssignProcessToJobObject(
                    ctypes.c_void_p(self.handle), ctypes.c_void_p(hproc)
                )
            )
        finally:
            _kernel32.CloseHandle(ctypes.c_void_p(hproc))

    def terminate(self) -> None:  # pragma: no cover - platform branch
        if self.ok:
            _kernel32.TerminateJobObject(ctypes.c_void_p(self.handle), 1)

    def close(self) -> None:  # pragma: no cover - platform branch
        if self.ok and self.handle:
            _kernel32.CloseHandle(ctypes.c_void_p(self.handle))
            self.handle, self.ok = None, False


# --------------------------------------------------------------------------- #
# Termination -- OWNED PIDS ONLY
# --------------------------------------------------------------------------- #


def _graceful(root_pid: int, owned: Sequence[int], report: CleanupReport) -> None:
    """Ask nicely. Windows: ``taskkill /T`` WITHOUT ``/F``. POSIX: ``SIGTERM``."""

    report.graceful_attempted = True
    if IS_WINDOWS:
        # /T walks the tree; no /F means WM_CLOSE / CTRL, not an immediate kill.
        # The PID is one we created -- never an image name.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(root_pid), "/T"],
                capture_output=True, timeout=30, check=False,
            )
        except Exception as exc:  # noqa: BLE001
            report.notes.append(f"graceful taskkill raised {exc!r}")
        return
    try:
        os.killpg(os.getpgid(root_pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001 - already gone
        for pid in owned:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass


def _force(job: Optional[_JobObject], root_pid: int, owned: Sequence[int],
           report: CleanupReport) -> None:
    """Forced termination of the job / owned PIDs. Still never by image name."""

    report.forced = True
    if job is not None and job.ok:
        job.terminate()
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(root_pid)],
                capture_output=True, timeout=30, check=False,
            )
        except Exception as exc:  # noqa: BLE001
            report.notes.append(f"forced taskkill raised {exc!r}")
        for pid in owned:
            if pid == root_pid:
                continue
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=15, check=False,
                )
            except Exception:  # noqa: BLE001
                pass
        return
    try:
        os.killpg(os.getpgid(root_pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    for pid in owned:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def _still_alive(owned: Dict[int, str]) -> List[Dict[str, Any]]:
    """Re-snapshot and report which owned PIDs are genuinely still present.

    Matching on ``(pid, image name)`` rather than PID alone, so a recycled PID
    belonging to some unrelated process is not misreported as a survivor.
    """

    table = snapshot_processes()
    out: List[Dict[str, Any]] = []
    for pid, name in owned.items():
        entry = table.get(pid)
        if entry is None:
            continue
        ppid, live_name = entry
        if name and live_name and live_name.lower() != name.lower():
            continue  # PID reused by an unrelated process
        out.append({"pid": pid, "name": live_name, "ppid": ppid, "rss_mb": None})
    return out


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #


def run(
    command: Sequence[str],
    *,
    timeout: float,
    label: str = "job",
    cwd: Optional[str] = None,
    grace: float = DEFAULT_GRACE_SECONDS,
    echo: bool = True,
    env: Optional[Dict[str, str]] = None,
    report_out: Optional[List[CleanupReport]] = None,
) -> CleanupReport:
    """Run *command* in the foreground, bounded, with guaranteed cleanup.

    *report_out*, if given, receives the :class:`CleanupReport` as soon as it is
    constructed, so a caller still holds the cleanup record on the paths where
    this function re-raises instead of returning.
    """

    command = [str(part) for part in command]
    cwd = cwd or os.getcwd()

    # Windows ``CreateProcess`` will not resolve a RELATIVE executable written
    # with forward slashes -- and every prompt in this sprint spells the
    # interpreter ``.venv/Scripts/python.exe``. Normalise argv[0] against cwd so
    # the documented invocation works from Git Bash as well as from PowerShell.
    if command and (os.sep in command[0] or "/" in command[0]):
        candidate = os.path.normpath(os.path.join(cwd, command[0]))
        if os.path.exists(candidate):
            command[0] = candidate
        else:
            command[0] = os.path.normpath(command[0])
    report = CleanupReport(
        label=label,
        command=command,
        cwd=cwd,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        timeout_seconds=float(timeout),
    )
    if report_out is not None:
        report_out.append(report)

    # Pre-existing processes are REPORTED, never killed ([S8] item 4). Snapshot
    # before the child starts so ownership is unambiguous afterwards.
    before = snapshot_processes()
    started = time.monotonic()

    job: Optional[_JobObject] = None
    proc: Optional[subprocess.Popen] = None
    owned: Dict[int, str] = {}
    log_handle = None
    log_path = None
    read_cursor = 0

    # A file, not a pipe: a pipe can deadlock on a full buffer, and the whole
    # point of this wrapper is that the parent never blocks unboundedly.
    fd, log_path = tempfile.mkstemp(prefix=f"boundedrun_{label}_", suffix=".log")
    os.close(fd)

    def _note_once(note: str) -> None:
        # _drain runs every poll interval; a repeated fault must not grow the
        # report without bound.
        if note not in report.notes:
            report.notes.append(note)

    def _drain() -> None:
        """Forward whatever the child has written since the last call.

        Nothing here may raise: this runs inside the wait loop *and* as the first
        statement of the cleanup ``finally``, so an escaping exception would skip
        termination, survivor verification and the cleanup report entirely.
        """

        nonlocal read_cursor
        if not echo:
            return
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(read_cursor)
                chunk = fh.read()
                read_cursor = fh.tell()
        except OSError:
            return
        except Exception as exc:  # noqa: BLE001 - drain faults never abort a job
            _note_once(f"drain read failed: {type(exc).__name__}: {exc}")
            return
        if not chunk:
            return
        try:
            # sys.stdout is resolved per call on purpose: callers such as
            # baseline_suite.py swap in a tee for the duration of the run.
            _forward_text(sys.stdout, chunk)
        except Exception as exc:  # noqa: BLE001 - belt and braces
            _note_once(f"drain forward failed: {type(exc).__name__}: {exc}")

    try:
        kwargs: Dict[str, Any] = {"cwd": cwd, "env": env or os.environ.copy()}
        if IS_WINDOWS:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True

        log_handle = open(log_path, "wb")
        proc = subprocess.Popen(  # noqa: S603 - argv is ours
            command, stdout=log_handle, stderr=subprocess.STDOUT, **kwargs
        )
        report.root_pid = proc.pid
        owned[proc.pid] = ""

        if IS_WINDOWS:
            job = _JobObject(label)
            if job.assign(proc.pid):
                report.isolation = "windows_job_object(KILL_ON_JOB_CLOSE)"
            else:
                report.isolation = "windows_process_group(fallback)"
                report.notes.append(
                    "Job Object assignment failed; fell back to owned-PID tree kill"
                )
        else:
            try:
                report.process_group = os.getpgid(proc.pid)
            except Exception:  # noqa: BLE001
                report.process_group = None
            report.isolation = "posix_process_group"

        # Record the root's real image name now that it exists, so survivor
        # verification can distinguish it from a recycled PID.
        table = snapshot_processes()
        if proc.pid in table:
            owned[proc.pid] = table[proc.pid][1]

        # ---- bounded wait; interruptible so `finally` always runs ----
        deadline = started + float(timeout)
        timed_out = False
        while True:
            rc = proc.poll()
            _drain()
            if rc is not None:
                break
            live = snapshot_processes()
            for pid in descendants_of(proc.pid, live):
                if pid not in owned and pid not in before:
                    owned[pid] = live[pid][1]
                    report.descendants_observed.append(
                        {"pid": pid, "name": live[pid][1], "ppid": live[pid][0]}
                    )
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(_POLL_INTERVAL)

        if timed_out:
            report.exit_reason = "timeout"
            report.exit_code = None
        else:
            report.exit_code = proc.returncode
            report.exit_reason = "completed" if proc.returncode == 0 else "nonzero"

    except KeyboardInterrupt:
        report.exit_reason = "cancelled"
        report.notes.append("KeyboardInterrupt in the parent; cleanup ran via finally")
    except BaseException as exc:  # noqa: BLE001 - cleanup must run for ANY exit
        report.exit_reason = "cancelled"
        report.notes.append(f"parent raised {type(exc).__name__}: {exc}")
        raise
    finally:
        # ------------------------------------------------------------------ #
        # CLEANUP -- every exit path reaches here, including cancellation.
        # ------------------------------------------------------------------ #
        _drain()
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError:
                pass

        if proc is not None:
            # Capture descendants one last time before killing anything. Guarded:
            # a fault while *observing* must not skip the termination and
            # survivor verification below -- the PIDs owned so far still get killed.
            try:
                live = snapshot_processes()
                for pid in descendants_of(proc.pid, live):
                    if pid not in owned and pid not in before:
                        owned[pid] = live[pid][1]
                        report.descendants_observed.append(
                            {"pid": pid, "name": live[pid][1], "ppid": live[pid][0]}
                        )
            except Exception as exc:  # noqa: BLE001
                _note_once(
                    f"final descendant capture failed: {type(exc).__name__}: {exc}"
                )

            outstanding = _still_alive(owned)
            if outstanding:
                _graceful(proc.pid, list(owned), report)
                grace_deadline = time.monotonic() + max(0.0, float(grace))
                while time.monotonic() < grace_deadline:
                    if not _still_alive(owned):
                        break
                    time.sleep(_POLL_INTERVAL)
                if _still_alive(owned):
                    _force(job, proc.pid, list(owned), report)

            if job is not None:
                job.close()  # KILL_ON_JOB_CLOSE: last-resort sweep of the job

            try:
                proc.poll()
            except Exception:  # noqa: BLE001
                pass

            # VERIFY, do not assume. Re-check until the table settles.
            settle_deadline = time.monotonic() + _SURVIVOR_SETTLE_SECONDS
            survivors = _still_alive(owned)
            while survivors and time.monotonic() < settle_deadline:
                time.sleep(_POLL_INTERVAL)
                survivors = _still_alive(owned)

            terminated = [
                pid for pid in owned if pid not in {s["pid"] for s in survivors}
            ]
            report.descendants_terminated = sorted(
                pid for pid in terminated if pid != proc.pid
            )
            report.survivors = survivors
            report.final_surviving_count = len(survivors)
            report.cleanup_success = not survivors

        report.duration_seconds = time.monotonic() - started
        report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Pre-existing processes: reported only. NEVER killed.
        report.preexisting_reported = [
            {"pid": pid, "name": before[pid][1]}
            for pid in sorted(before)
            if before[pid][1].lower() in {"python.exe", "streamlit.exe", "pytest.exe"}
        ]

        if not report.cleanup_success:
            report.exit_reason = "infrastructure_failure"
            report.returned_code = EXIT_INFRASTRUCTURE_FAILURE
        elif report.exit_reason == "timeout":
            report.returned_code = 124
        elif report.exit_reason == "cancelled":
            report.returned_code = 130
        else:
            report.returned_code = report.exit_code

        if log_path and os.path.exists(log_path):
            try:
                os.unlink(log_path)
            except OSError:
                pass

    return report


def _staged_reservation_for(json_path: str) -> str:
    """The staging path holding a G11 reservation for *json_path*, or ``""``.

    Recognised only when the sibling really is an unwritten reservation: it must
    parse as an object and carry :data:`G11_RESERVED_KEY`. Anything else -- a
    scratch ``--json`` path with nothing behind it, an unrelated file that
    happens to sit in a ``.staging`` directory, a half-written orphan -- returns
    ``""`` and the report is written straight to the destination, which is what
    every caller predating this saw and must keep seeing.
    """

    directory, name = os.path.split(os.path.abspath(json_path))
    staged = os.path.join(directory, G11_STAGING_DIRNAME, name)
    try:
        with open(staged, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict) or payload.get(G11_RESERVED_KEY) is not True:
        return ""
    return staged


def emit_json_report(report: CleanupReport, json_path: Optional[str]) -> None:
    """Persist *report* to *json_path*. Never raises.

    An unwritable destination is a **distinct, reported condition**: named on
    stderr with :data:`JSON_REPORT_UNWRITABLE_MARKER`, recorded in ``notes`` and
    ``json_report_error``, and the rendered report still reaches stderr, so the
    cleanup result is not lost with it. It must not become an exception -- that
    would destroy the very record the caller needs.

    PROMOTION (F-071). When a G11 reservation is waiting behind *json_path*, the
    report is written to that staging path and then moved onto the destination
    with :func:`os.replace`. The reports tree therefore sees the artifact whole
    or not at all -- ``os.replace`` publishes the target in the same step that
    removes the source, and both live in one directory tree, so the rename is a
    rename and never a copy. A partially written report in the reports tree would
    be worse than none, which is the same reason the payload is serialised before
    anything is opened.
    """

    report.json_report_path = json_path or ""
    if not json_path:
        return
    staged = _staged_reservation_for(json_path)
    marker = G11_PROMOTION_FAILED_MARKER if staged else JSON_REPORT_UNWRITABLE_MARKER
    try:
        report.json_report_written = True
        # Serialise first: a serialisation fault must not leave a truncated file
        # that a later reader would mistake for a cleanup report.
        payload = json.dumps(report.to_dict(), indent=2)
        if staged:
            with open(staged, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(staged, json_path)
        else:
            with open(json_path, "w", encoding="utf-8") as fh:
                fh.write(payload)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        report.json_report_written = False
        report.json_report_error = f"{type(exc).__name__}: {exc}"
        report.notes.append(
            f"{marker}: {json_path}: {report.json_report_error}"
        )
        if staged and os.path.exists(staged):
            # A report stranded in staging still claims it was written, and it is
            # invisible to `check`. Correct it in place so whoever finds the
            # orphan is not misled about what happened to it.
            try:
                with open(staged, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(report.to_dict(), indent=2))
            except Exception as inner:  # noqa: BLE001
                report.notes.append(f"{marker}: staged copy not corrected: {inner!r}")
        _forward_text(
            sys.stderr,
            f"\n{marker} path={json_path} "
            f"error={report.json_report_error}\n",
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded foreground process wrapper (sprint gate G11 / [S8])."
    )
    parser.add_argument("--timeout", type=float, required=True, help="outer wall clock, seconds")
    parser.add_argument("--label", default="job")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--grace", type=float, default=DEFAULT_GRACE_SECONDS)
    parser.add_argument("--json", dest="json_path", default=None,
                        help="write the structured cleanup report here")
    parser.add_argument("--quiet", action="store_true", help="do not echo child output")
    parser.add_argument(
        "--heavy-lock", default=None, metavar="HOLDER",
        help="acquire the sprint heavy mutex as HOLDER (e.g. a card id) before "
             f"starting the child. A failed acquire exits "
             f"{EXIT_HEAVY_LOCK_UNAVAILABLE} WITHOUT running it. Opt-in: omit it "
             "and nothing about this wrapper changes")
    parser.add_argument(
        "--heavy-lock-path", default=DEFAULT_HEAVY_LOCK_PATH, metavar="DIR",
        help=f"override the lock directory (default {DEFAULT_HEAVY_LOCK_PATH}). "
             "Exists so a test can exercise the protocol against a scratch lock "
             "instead of the real one")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv) if argv is not None else None)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given (use: ... --timeout N -- <command>)")

    # ---- heavy mutex: acquire, or STOP. There is no third branch (F-072) ----
    lock: Optional[HeavyLock] = None
    if args.heavy_lock:
        lock = HeavyLock(args.heavy_lock, args.heavy_lock_path,
                         label=args.label, command=command)
        if not lock.acquire():
            # No --json report is written here ON PURPOSE. The child never ran,
            # so there is no cleanup to certify, and a report claiming
            # cleanup_success on a job that never started is exactly the shape a
            # G11 checker would accept as a pass. Any reservation the caller
            # allocated simply stays in staging, where `check` cannot see it.
            _forward_text(
                sys.stderr,
                f"\n{HEAVY_LOCK_HELD_MARKER} path={lock.path} "
                f"holder_attempted={lock.holder} reason={lock.acquire_error}\n"
                f"---- current holder file ({lock.holder_file}) ----\n"
                f"{lock.holder_seen or '(no holder file present)'}\n"
                f"---- the child was NOT started; nothing was removed ----\n",
            )
            return EXIT_HEAVY_LOCK_UNAVAILABLE

    # Emitted from a `finally`, with `run` handing the report over as soon as it
    # exists, so a writable --json destination receives it on EVERY exit path --
    # including one where an exception escapes `run`. A run with no cleanup
    # report is uncertifiable under G11.
    holder: List[CleanupReport] = []
    released = True
    try:
        report = run(
            command,
            timeout=args.timeout,
            label=args.label,
            cwd=args.cwd,
            grace=args.grace,
            echo=not args.quiet,
            report_out=holder,
        )
    finally:
        # Release BEFORE the report is written, so the artifact records what
        # actually happened to the lock. This `finally` is every exit path the
        # wrapper controls: success, nonzero, timeout kill, child crash,
        # KeyboardInterrupt, or an exception escaping `run`.
        if lock is not None:
            released = lock.release()
        if holder:
            if lock is not None:
                holder[0].heavy_lock = lock.snapshot()
                if not released:
                    holder[0].notes.append(
                        f"{HEAVY_LOCK_RELEASE_REFUSED_MARKER}: {lock.path}: "
                        f"{lock.release_refusal}"
                    )
                    # BEFORE the artifact is written, so ``returned_code`` --
                    # which ``g11_evidence.REQUIRED_FIELDS`` documents as "what
                    # the wrapper returned" -- is true on this path as well.
                    # Without it the record reads the child's 0 while the process
                    # exits 96, and nothing in the artifact says which happened.
                    # ``exit_reason`` deliberately does NOT move to
                    # ``infrastructure_failure``: process cleanup really did
                    # succeed, and relabelling it would misreport what ``run()``
                    # measured. The lock condition is carried by
                    # ``heavy_lock.release_refusal`` and by the note above.
                    holder[0].returned_code = EXIT_HEAVY_LOCK_RELEASE_REFUSED
            emit_json_report(holder[0], args.json_path)
            _forward_text(sys.stderr, holder[0].render() + "\n")
        if lock is not None and not released:
            _forward_text(
                sys.stderr,
                f"\n{HEAVY_LOCK_RELEASE_REFUSED_MARKER} path={lock.path} "
                f"holder_expected={lock.holder} reason={lock.release_refusal}\n"
                f"---- holder file as found ({lock.holder_file}) ----\n"
                f"{lock.holder_seen or '(absent)'}\n"
                f"---- the lock was LEFT IN PLACE. Clearing another holder's "
                f"lock is the orchestrator's decision alone ----\n",
            )
    if lock is not None and not released:
        # The lock is now in a state this job cannot vouch for, which is an
        # infrastructure condition and not a test result -- so it takes
        # precedence over the child's own code, exactly as a failed cleanup
        # verification does. That precedent overrides ``report.returned_code``
        # too, and the ``finally`` above now does the same, so artifact and
        # process AGREE on this path instead of the parity being asserted only
        # in a comment (REV-063 correction 1).
        return EXIT_HEAVY_LOCK_RELEASE_REFUSED
    return int(report.returned_code if report.returned_code is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
