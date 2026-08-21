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

Report lifecycle  (F-071)
-------------------------
``next`` prints the path the FINISHED report will occupy, exactly as it always
has. What changed is where the **reservation** lives: :func:`allocate` writes its
``g11_reserved`` placeholder into the task's :data:`STAGING_DIRNAME`
subdirectory, and ``bounded_run.py`` promotes it -- one atomic rename, inside the
task directory -- when the job actually produces a report.

So a job that never finishes (an agent's own wall clock killing the parent shell
was the failure four times over) leaves **nothing the reports tree can see**,
instead of a three-key placeholder that turns whole-tree ``check`` red and fails
merge gate 10 for a job that produced no result either way. The
``report_never_written`` rule below is untouched: a placeholder that reaches the
reports tree by any route is still a hard failure.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

#: Where reservations wait until a job has something to promote (F-071). One
#: subdirectory per task, inside the task directory, so promotion is a rename
#: within one directory tree and therefore atomic on every platform this sprint
#: runs on.
#:
#: The leading dot is load-bearing, not cosmetic: ``iter_reports`` skips names
#: beginning with ``.``, and that single rule is what keeps a live reservation
#: out of every ``check`` selection. ``bounded_run.py`` hard-codes this same name
#: (``bounded_run.G11_STAGING_DIRNAME``); the two must stay equal, and
#: ``test_staging_contract_matches_the_wrapper`` fails if they drift.
STAGING_DIRNAME = ".staging"

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
#:
#: F-082. The leading ``\b`` on the first two is load-bearing, not tidiness.
#: Without it, any ordinary word ending in ``sk`` -- or in ``gh`` plus one of
#: ``pousr`` -- starts a match inside a perfectly innocent label or path, and a
#: clean artifact is rejected as carrying a credential. That fails whole-tree
#: ``check``, which is merge gate 10. Measured on labels REV-068 actually used:
#: ``ondisk-nonvacuity-control-green`` matched ``sk-nonvacuity-control-green``,
#: and ``task-``, ``risk-``, ``disk-``, ``desk-``, ``mask-`` are all ordinary
#: job vocabulary. The boundary costs no true positive: a real key stands at a
#: string start or after ``=``, ``"``, whitespace or ``/``, each of which IS a
#: boundary, so ``--api-key=sk-...`` still matches.
CRED_PATTERNS = [
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    # Deliberately NOT given a boundary. The shape is the same, but the only
    # false positive that exists needs a capital ``A`` immediately followed by
    # ``Iza`` INSIDE a longer word: a label cannot contain one (LABEL_RE is
    # lowercase) and no path or command this sprint produces does either.
    # Changing it would be a fix with no demonstrated defect behind it. If a
    # real false positive ever turns up, this is the line to change.
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


def staging_path_for(report_path: str) -> str:
    """The reservation path whose promotion target is *report_path*.

    Purely structural -- same basename, one directory deeper -- so
    ``bounded_run.py`` derives it from the ``--json`` path it was handed without
    importing this module, and a reader can verify any promotion by looking at
    two paths side by side.
    """

    directory, name = os.path.split(report_path)
    return os.path.join(directory, STAGING_DIRNAME, name)


def allocate(task: str, label: str, root: str = REPORTS_ROOT) -> str:
    """Reserve and return the report path for ONE job.

    The RETURNED path is where the finished report will live, unchanged from
    before. The RESERVATION is made at :func:`staging_path_for` of that path, so
    an unfinished job leaves the reports tree untouched (F-071).

    Uniqueness is not "unlikely to collide", it is structural, and staging does
    not weaken any of the three properties -- it only moves which directory the
    ``O_EXCL`` happens in:

    * the directory is the task ID, and a task owns exactly one branch;
    * the sequence is ``max(existing) + 1`` over the task directory **and its
      staging directory together**, and the reservation is created with
      ``O_CREAT|O_EXCL``, so two allocations -- even concurrent ones, even for
      the same label -- can never resolve to the same path. Two racing callers
      that compute the same sequence race for one ``O_EXCL`` create of one
      staging path; exactly one wins and the loser increments;
    * sequences are never reused. Every sequence ever allocated is held by
      exactly one file at every instant: by its reservation in
      ``.staging/`` until promotion, and by the report itself in the task
      directory afterwards. Promotion is ``os.replace``, which publishes the
      target in the same step that removes the source, so there is no window in
      which neither holds it. A re-run therefore cannot overwrite the report of
      the attempt it is replacing (an infrastructure failure must stay on
      record), and cannot overwrite the reservation of a job still running.

    No clock is involved: the caller needs no timestamp it cannot obtain.
    """

    if not TASK_RE.match(task):
        raise ValueError(f"task id {task!r} must look like H-004 / C-056a / INIT-001")
    if not LABEL_RE.match(label):
        raise ValueError(f"label {label!r} must match {LABEL_RE.pattern}")
    directory = os.path.join(root, task)
    staging = os.path.join(directory, STAGING_DIRNAME)
    os.makedirs(staging, exist_ok=True)
    used = [
        int(m.group(1))
        for listing in (directory, staging)
        for m in (NAME_RE.match(n) for n in os.listdir(listing)) if m
    ]
    seq = max(used, default=0) + 1
    while True:
        path = os.path.join(directory, f"{seq:02d}-{label}.json")
        # A promoted report holds its sequence here even though its reservation
        # is gone, so the task directory is consulted before the staging one.
        if os.path.exists(path):
            seq += 1
            continue
        staged = staging_path_for(path)
        try:
            fd = os.open(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            seq += 1
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({RESERVED_KEY: True, "task": task, "label": label,
                       "promote_to": path}, fh, indent=2)
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
    """Every candidate artifact under the evidence root, sorted.

    The dot-prefix skip below is what keeps :data:`STAGING_DIRNAME` -- and every
    live reservation inside it -- out of every selection (F-071). It is not a
    convenience: without it a running job's reservation is a directory in a task
    folder, ``check_many`` would call it ``unexpected_artifact``, and the merge
    gate would go red on a job that has not finished yet.
    """

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


def selector_violation(selector: str) -> str:
    """Why an explicitly supplied task selector resolved to nothing."""

    kind = "unmatched_task" if TASK_RE.match(selector) else "malformed_task"
    return f"{kind}:{selector} -- selector matched 0 committed artifacts"


def resolve_selection(
    paths: List[str], task: Optional[str], root: str = REPORTS_ROOT,
) -> Tuple[List[str], List[str]]:
    """Split a CLI selection into artifacts to check and unmatched selectors.

    RULE 6: a supplied ``--task`` that matches nothing comes back as an
    **unmatched selector**, never as an empty artifact list. ``check_many([])``
    prints "0 artifact(s), 0 non-compliant" and exits 0, so an empty resolution
    used to certify a task with no evidence at all -- and an explicit path
    alongside it concealed the miss entirely, because the task was then ignored.

    Whole-tree behaviour is unchanged: no selector of either kind means every
    committed artifact under ``root``.
    """

    selected = list(paths)
    unmatched: List[str] = []
    if task is not None:
        matched = iter_reports(root=root, task=task)
        if matched:
            selected.extend(matched)
        else:
            unmatched.append(task)
    elif not selected:
        selected = iter_reports(root=root)
    return selected, unmatched


def check_many(paths: List[str], unmatched: Sequence[str] = ()) -> int:
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
    for selector in unmatched:
        print(f"FAIL --task {selector}\n     {selector_violation(selector)}")
    print(f"\nG11 evidence: {len(paths)} artifact(s), {failures} non-compliant "
          f"(spec v{G11_SPEC_VERSION})"
          + (f"; {len(unmatched)} unmatched selector(s)" if unmatched else ""))
    return 1 if failures or unmatched else 0


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
    """(b) Two jobs in one task never share a path -- same label included.

    DELIBERATE DELTA (C-063 / F-071). Durability is still asserted, but of the
    RESERVATION, which now lives in ``.staging``; the four returned paths are the
    promotion targets and none of them exists until a job promotes one. The
    property that mattered is unchanged and still asserted below: a
    reserved-but-unwritten slot is evidence of nothing.
    """

    with tempfile.TemporaryDirectory(prefix="g11nam_") as tmp:
        a = allocate("H-004", "smoke", root=tmp)
        b = allocate("H-004", "smoke", root=tmp)
        c = allocate("H-004", "chunk-d", root=tmp)
        d = allocate("C-010", "smoke", root=tmp)
        assert len({a, b, c, d}) == 4, (a, b, c, d)
        staged = [staging_path_for(p) for p in (a, b, c, d)]
        assert len(set(staged)) == 4, staged
        assert all(os.path.isfile(p) for p in staged), "reservation not durable"
        assert not any(os.path.exists(p) for p in (a, b, c, d)), \
            "an unpromoted reservation must leave the reports tree untouched"
        assert NAME_RE.match(os.path.basename(a)) and NAME_RE.match(os.path.basename(b))
        assert int(NAME_RE.match(os.path.basename(b)).group(1)) > \
            int(NAME_RE.match(os.path.basename(a)).group(1))
        # A reserved-but-unwritten path is evidence of nothing and must fail.
        assert "report_never_written" in check_report(staged[0])
        # ... and its promotion target does not exist at all until it is earned.
        assert "report_missing" in check_report(a)
        # A sequence stays consumed once promoted, even though the reservation
        # that held it is gone: a re-run cannot land on the attempt it replaces.
        os.replace(staged[0], a)
        assert allocate("H-004", "smoke", root=tmp) not in {a, b, c}


def test_reservation_is_invisible_to_the_merge_gate() -> None:
    """F-071: an unfinished job must leave the merge gate nothing to see.

    Whole-tree ``check`` is the consumer that gates merges -- four times over,
    an agent's own wall clock killed a wrapper and this went red on a job that
    produced no result either way. It must now be GREEN with a live reservation
    outstanding, and it must still be RED for a placeholder that reaches the
    reports tree by any other route.
    """

    with tempfile.TemporaryDirectory(prefix="g11res_") as tmp:
        target = allocate("H-004", "killed", root=tmp)
        assert iter_reports(root=tmp) == [], iter_reports(root=tmp)
        assert check_many(*resolve_selection([], None, root=tmp)) == 0, \
            "a live reservation turned whole-tree check red -- F-071 is back"
        # The `report_never_written` rule is untouched: promote the placeholder
        # itself (the one route that still puts one in the reports tree) and the
        # gate must reject it.
        os.replace(staging_path_for(target), target)
        assert check_many(*resolve_selection([], None, root=tmp)) == 1
        assert "report_never_written" in check_report(target)


def test_staging_contract_matches_the_wrapper() -> None:
    """The promoter and the allocator must agree on where reservations live.

    ``bounded_run.py`` derives the staging path itself rather than importing this
    module, so if the two sides ever disagree, every promotion silently stops
    happening and every finished job starts looking like an unfinished one.

    MEASURED COVERAGE (REV-063). Both failure shapes are caught, because the
    equality below is asserted against what ``_staged_reservation_for`` ANSWERS
    for a real reservation, not against the constant alone:

    * a RENAMED wrapper constant -- caught by the first assertion;
    * a RELOCATED wrapper staging path with the constant left untouched -- also
      caught, because a wrapper looking somewhere else finds no reservation and
      returns ``""``, which is not the path this module reserved.

    The implementer's own report claimed only the first of these. It was wrong
    in its own disfavour, and the record is corrected here rather than left for
    someone to file a card against a gap that does not exist.
    """

    sys.path.insert(0, os.path.dirname(BOUNDED_RUN))
    import bounded_run  # noqa: PLC0415 - deliberate late import, read-only

    assert bounded_run.G11_STAGING_DIRNAME == STAGING_DIRNAME, (
        bounded_run.G11_STAGING_DIRNAME, STAGING_DIRNAME)
    assert bounded_run.G11_RESERVED_KEY == RESERVED_KEY, (
        bounded_run.G11_RESERVED_KEY, RESERVED_KEY)
    with tempfile.TemporaryDirectory(prefix="g11stg_") as tmp:
        target = allocate("H-004", "contract", root=tmp)
        assert bounded_run._staged_reservation_for(target) == \
            os.path.abspath(staging_path_for(target)), (
                "the wrapper does not look where this module reserves: "
                f"wrapper={bounded_run._staged_reservation_for(target)!r} "
                f"allocator={os.path.abspath(staging_path_for(target))!r}")
        # A path with no reservation behind it is NOT promotable: every legacy
        # caller that hands the wrapper a scratch --json path keeps writing
        # straight to it.
        assert bounded_run._staged_reservation_for(
            os.path.join(tmp, "H-004", "99-nothing.json")) == "", (
                "the wrapper treated a path with no reservation behind it as "
                "promotable; every legacy caller's scratch --json path would "
                "stop being written straight through")


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


def test_selector_resolution() -> None:
    """RULE 6: a task selector that matches nothing must never report success.

    Five cases, matching the five ways ``check`` can be invoked: known task,
    unmatched well-formed task, malformed task, whole-tree, and an explicit
    path supplied *alongside* an unmatched task -- the concealment case, which
    is the only reason ``--task`` and ``paths`` are resolved together rather
    than one falling back to the other.
    """

    with tempfile.TemporaryDirectory(prefix="g11sel_") as tmp:
        os.makedirs(os.path.join(tmp, "H-004"))
        known = _write(tmp, os.path.join("H-004", "01-a.json"))

        # (1) a known task validates normally, over a NONZERO artifact count.
        paths, unmatched = resolve_selection([], "H-004", root=tmp)
        assert paths == [known] and not unmatched, (paths, unmatched)
        assert check_many(paths, unmatched) == 0

        # (2) a well-formed task matching zero artifacts exits NONZERO, and the
        # message names the selector that matched nothing.
        paths, unmatched = resolve_selection([], "Z-999", root=tmp)
        assert paths == [] and unmatched == ["Z-999"], (paths, unmatched)
        assert check_many(paths, unmatched) == 1
        assert "unmatched_task:Z-999" in selector_violation("Z-999")

        # (3) a malformed task is still rejected, and is classified as malformed.
        # `next` keeps its own pre-existing rejection: allocate() raises.
        assert check_many(*resolve_selection([], "nope", root=tmp)) == 1
        assert "malformed_task:nope" in selector_violation("nope")
        try:
            allocate("nope", "x", root=tmp)
        except ValueError:
            pass
        else:  # pragma: no cover - the guard exists; this records that it does
            raise AssertionError("allocate() stopped rejecting a malformed task")

        # (4) whole-tree checking (no selector at all) is unchanged.
        paths, unmatched = resolve_selection([], None, root=tmp)
        assert paths == [known] and not unmatched, (paths, unmatched)

        # (5) one valid selector may not conceal another unmatched one.
        paths, unmatched = resolve_selection([known], "Z-999", root=tmp)
        assert paths == [known] and unmatched == ["Z-999"], (paths, unmatched)
        assert check_many(paths, unmatched) == 1


def test_credential_scan_is_word_bounded_and_still_bites() -> None:
    r"""F-082: an ordinary label must not read as a credential -- nor the reverse.

    RAW docstring, deliberately. It names a regex escape, and in a non-raw one
    Python would parse that escape away -- this very text shipped for a few
    minutes with a real backspace byte in it, which is the defect it describes.

    REV-068 had two evidence artifacts rejected because their labels contained
    ``...sk-``: the openai pattern had no left word boundary, so
    ``ondisk-nonvacuity-control-green`` matched ``sk-nonvacuity-control-green``.
    Any label built from ``disk-``, ``task-``, ``risk-``, ``mask-`` or ``desk-``
    plus 16 more characters failed whole-tree G11 -- merge gate 10 -- on an
    artifact carrying no credential at all.

    BOTH DIRECTIONS ARE ASSERTED, and that is the whole point. A pattern
    loosened until it matched nothing would sail through a false-positive-only
    test while letting a real key into a committed artifact; the true-positive
    half below is what stops that, and it covers every prefix pattern in
    :data:`CRED_PATTERNS`, not just the one that was changed.

    The true-positive half exists because a pattern can be silently neutralised
    by TRANSPORT, not only by a careless edit. While this very fix was being
    written the \b arrived in the file as a literal backspace byte
    (0x08): it compiles, it reads almost identically at a glance, and it matches
    nothing. A credential scanner that matches nothing reports every artifact
    clean, which is the worst outcome this module has. A false-positive-only
    test would have gone green on it.

    Both halves go through :func:`check_report` on real artifacts rather than
    against the regexes directly, because ``check_report`` is the consumer whose
    verdict actually blocks a commit.
    """

    with tempfile.TemporaryDirectory(prefix="g11cred_") as tmp:
        def flagged(path: str) -> bool:
            return any(v.startswith("possible_credential")
                       for v in check_report(path))

        # (a) ordinary vocabulary, in the fields that really carry it.
        clean = [
            ("01-clean.json", {"label": "ondisk-nonvacuity-control-green"}),
            ("02-clean.json", {"label": "task-reconciliation-evidence"}),
            ("03-clean.json", {"label": "risk-assessment-baseline-run"}),
            ("04-clean.json", {"label": "disk-pressure-probe-run-01"}),
            # the same defect one pattern over: a path segment, not a label,
            # because LABEL_RE forbids the underscore github_token needs.
            ("05-clean.json", {"cwd": "C:/runs/troughs_0123456789abcdefghij"}),
        ]
        for name, over in clean:
            path = _write(tmp, name, **over)
            assert not flagged(path), (
                f"ordinary artifact {over!r} read as a credential: "
                f"{check_report(path)}")

        # (b) the shapes that MUST still be rejected -- one per prefix pattern,
        # plus a control, so this loop cannot pass by asserting nothing.
        biting = [
            ("10-cred.json", {"command": ["py", "r.py", "--api-key=sk-" + "A" * 24]}, True),
            ("11-cred.json", {"command": ["py", "r.py", "sk-" + "b3" * 12]}, True),
            ("12-cred.json", {"command": ["py", "r.py", "ghp_" + "c4" * 12]}, True),
            ("13-cred.json", {"command": ["py", "r.py", "AIzaSy" + "d5" * 12]}, True),
            ("14-cred.json", {"command": ["py", "r.py", "AKIA" + "E" * 16]}, True),
            ("15-cred.json", {"command": ["py", "r.py", "--quiet"]}, False),
        ]
        for name, over, want in biting:
            path = _write(tmp, name, **over)
            assert flagged(path) is want, (
                f"{over!r}: expected possible_credential={want}, "
                f"got {check_report(path)}")


TESTS = [
    test_required_fields_present_in_real_wrapper_output,
    test_naming_scheme_cannot_collide,
    test_reservation_is_invisible_to_the_merge_gate,
    test_staging_contract_matches_the_wrapper,
    test_compliance_rules,
    test_credential_scan_is_word_bounded_and_still_bites,
    test_selector_resolution,
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
    chk.add_argument("--task", default=None,
                     help="ONE task selector; matching zero artifacts is an error")
    chk.add_argument("paths", nargs="*")
    sub.add_parser("selftest", help="prove this spec matches bounded_run.py")
    args = parser.parse_args(argv)

    if args.cmd == "next":
        path = allocate(args.task, args.label)
        print(path)
        # stdout stays exactly one path, because every caller substitutes it into
        # a --json argument. Where the reservation is actually waiting goes to
        # stderr, so whoever has to dispose of an orphan can find it (F-071).
        print(f"reserved at {staging_path_for(path)} "
              f"(promoted here by bounded_run.py when the job writes a report)",
              file=sys.stderr)
        return 0
    if args.cmd == "selftest":
        return selftest()
    return check_many(*resolve_selection(args.paths, args.task))


if __name__ == "__main__":
    raise SystemExit(main())
