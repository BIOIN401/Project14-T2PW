"""C-063 G9 proof: the F-072 mutex and the F-071 report lifecycle, base vs tip.

Both findings are **corrections of pre-existing observable behaviour**, not new
capabilities, so merge gate G9 requires a proof that fails behaviourally on the
base SHA and passes at the tip. Symbol absence is not proof, and this harness
never uses it: it never asks whether ``--heavy-lock`` parses, only what happens
to the lock and to the reports tree.

Why the base arms drive a SHELL and the tip arms drive a FLAG
------------------------------------------------------------
At ``ac77668`` the heavy mutex has no implementation anywhere -- F-072's whole
point is that the protocol lived only in prose, so every agent hand-rolled the
same three-step dance and got it wrong a new way each time. The base arm
therefore executes **the only protocol that existed at the base SHA**, in the
exact shape REV-058 ran it, driving the real base wrapper as the job. Running
``--heavy-lock`` against the base wrapper instead would produce an argparse
error, the child would never start, the lock would survive -- and the base arm
would *pass the fix's own assertions* while proving nothing. That is precisely
the trap G9's "symbol absence is not proof" names, and it is why the comparison
is protocol-vs-protocol rather than flag-vs-missing-flag.

**Scope limit, stated rather than hidden:** the tip arm proves the primitive
enforces the protocol *when it is used*. An agent at the tip who ignores
``--heavy-lock`` and hand-rolls the shell dance anyway still reproduces F-072.
This card removes the primitive from agent hands; it cannot remove the shell.

F-074, the replica trap
-----------------------
Nothing here is a stand-in. The base arms run the real historical modules,
extracted with ``git show ac77668:<path>`` and executed; the tip arms run the
real working-tree modules. Every extracted or copied file's SHA-256 is asserted
against its source before it is used, and the digests are recorded in the
artifact, so a reader can confirm no arm ran a replica.

RULING 13 -- non-vacuity
------------------------
Both findings audit infrastructure, so both harnesses can silently stop testing
anything. Each guard therefore has a NEUTRALIZED arm: the tip module is copied,
one exact substitution disables the guard being audited, the substitution is
asserted to have applied exactly once, and the tip's own assertions are re-run
and required to go RED. A harness that stays green against a neutered guard is
reported as a harness failure, not as a pass.

::

    <py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label c063-g9 \\
        --timeout 300 --json <allocated> -- \\
        <py> -u docs/pwml_recovery_sprint/evidence/c063_lifecycle_g9.py \\
             --work <scratch dir> --out <artifact.json>

The heavy mutex is never taken by this harness and ``C:/t/heavylock`` is never
created, read or removed: every lock arm points at a scratch directory under
``--work``. A harness that manipulated the live mutex to test the mutex would be
the very defect F-072 records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

#: The integration tip this card was cut from. The base arms are this commit's
#: real modules, not a reconstruction of them.
BASE_SHA = "ac776682d36012b0b583952d78ac8f0cf02115a3"

WRAPPER_REL = "docs/pwml_recovery_sprint/evidence/bounded_run.py"
G11_REL = "docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py"

TIP_WRAPPER = os.path.join(REPO_ROOT, *WRAPPER_REL.split("/"))
TIP_G11 = os.path.join(REPO_ROOT, *G11_REL.split("/"))

PY = sys.executable

sys.path.insert(0, HERE)
import bounded_run  # noqa: E402  - the TIP module, read-only, for its exit codes

#: The task id the F-071 arms allocate under, inside their own scratch reports
#: tree. It never touches the committed evidence tree.
SCRATCH_TASK = "C-063"

#: A foreign holder, in the shape the tip wrapper writes. Its token can never
#: match a live job's, which is what release compares against.
FOREIGN_HOLDER = {
    "token": "C-999-SOMEONE-ELSE:4242:ffffffffffffffff",
    "holder": "C-999-SOMEONE-ELSE",
    "label": "a-heavy-job-that-is-still-running",
    "pid": 4242,
    "command": ["<the other card's heavy job>"],
    "cwd": "<elsewhere>",
    "acquired_at": "2026-08-21T00:00:00+00:00",
}
HOLDER_FILENAME = "holder.json"


# --------------------------------------------------------------------------- #
# Materialising the real modules -- never a replica  (F-074)
# --------------------------------------------------------------------------- #


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return _sha256(fh.read())


def _git_show(rel_path: str) -> bytes:
    """The exact bytes of *rel_path* at :data:`BASE_SHA`. Binary, never text."""

    proc = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", REPO_ROOT,
         "show", f"{BASE_SHA}:{rel_path}"],
        capture_output=True, timeout=120, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git show {BASE_SHA}:{rel_path} failed: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def materialise_base(path: str, rel_path: str) -> str:
    """Write the base-SHA copy of *rel_path* to *path*; return its digest."""

    blob = _git_show(rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    written = _sha256_file(path)
    if written != _sha256(blob):
        raise RuntimeError(f"base extraction of {rel_path} did not round-trip")
    return written


def materialise_tip(path: str, source: str) -> str:
    """Copy the working-tree module to *path* byte-for-byte; return its digest."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copyfile(source, path)
    written = _sha256_file(path)
    if written != _sha256_file(source):
        raise RuntimeError(f"tip copy of {source} is not byte-identical")
    return written


def neutralize(source: str, dest: str, old: str, new: str) -> Dict[str, Any]:
    """Copy *source* to *dest* with exactly one occurrence of *old* replaced.

    RULING 13. The count is asserted, so a neutralization that silently stopped
    applying -- because the line it targets moved or was rewritten -- is a loud
    harness failure and never a quiet green.
    """

    with open(source, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"neutralization target occurs {count} times in {source} (need exactly 1):"
            f"\n  {old!r}"
        )
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace(old, new))
    return {
        "module": os.path.basename(source),
        "removed": old.strip(),
        "replaced_with": new.strip(),
        "occurrences_substituted": count,
        "neutralized_digest": _sha256_file(dest),
    }


# --------------------------------------------------------------------------- #
# Small process helpers
# --------------------------------------------------------------------------- #


def _fwd(path: str) -> str:
    """Forward slashes, so a path can be embedded in a bash script verbatim."""

    return os.path.abspath(path).replace("\\", "/")


def _write_script(path: str, body: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def _run(argv: Sequence[str], timeout: float = 180.0) -> Tuple[int, str, str]:
    proc = subprocess.run(list(argv), capture_output=True, timeout=timeout)
    return (proc.returncode,
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"))


def _plant_foreign_lock(lock_dir: str) -> bytes:
    shutil.rmtree(lock_dir, ignore_errors=True)
    os.makedirs(lock_dir)
    with open(os.path.join(lock_dir, HOLDER_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(FOREIGN_HOLDER, fh, indent=2)
    return _holder_bytes(lock_dir)


def _holder_bytes(lock_dir: str) -> bytes:
    try:
        with open(os.path.join(lock_dir, HOLDER_FILENAME), "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def _bash() -> str:
    found = shutil.which("bash")
    if not found:
        raise RuntimeError(
            "bash is required: the base arms must execute the real shell protocol, "
            "and reimplementing its control flow in Python would be the replica "
            "trap F-074 names. Refusing to run a narrower arm silently."
        )
    return found


# --------------------------------------------------------------------------- #
# F-072 -- the heavy mutex
# --------------------------------------------------------------------------- #

#: The documented protocol as REV-058 executed it, verbatim in shape: an acquire
#: chained to an echo with ``&&``, the job as its own statement, and an
#: unconditional release. `mkdir` failing suppresses ONLY the echo.
_SHELL_DANCE = """mkdir '{lock}' && echo ACQUIRED
"{py}" "{wrapper}" --label {label} --timeout 60 --quiet --json "{json}" -- "{py}" "{child}"
rm -rf '{lock}'
"""


def _probe_child(path: str, marker: str) -> str:
    return _write_script(path, f"open(r'{marker}', 'w').write('the child ran')\n")


def _steal_child(path: str, holder_file: str) -> str:
    return _write_script(path, (
        "import json\n"
        f"json.dump({FOREIGN_HOLDER!r}, open(r'{holder_file}', 'w'), indent=2)\n"
    ))


def _observe_lock(lock_dir: str, marker: str, rc: int,
                  holder_before: bytes) -> Dict[str, Any]:
    """What the filesystem says after an arm ran. No parsing of any output."""

    return {
        "child_ran": os.path.exists(marker),
        "lock_survives": os.path.isdir(lock_dir),
        "holder_unchanged": _holder_bytes(lock_dir) == holder_before,
        "holder_now": _holder_bytes(lock_dir).decode("utf-8", "replace")[:512],
        "exit_code": rc,
    }


def f072_acquire_base(work: str) -> Dict[str, Any]:
    """Someone else holds the mutex; the documented shell protocol runs anyway."""

    root = os.path.join(work, "f072a_base")
    lock = os.path.join(root, "scratch_heavylock")
    marker = os.path.join(root, "child_ran.txt")
    before = _plant_foreign_lock(lock)
    child = _probe_child(os.path.join(root, "probe.py"), _fwd(marker))
    script = _write_script(os.path.join(root, "dance.sh"), _SHELL_DANCE.format(
        lock=_fwd(lock), py=_fwd(PY), wrapper=_fwd(BASE["wrapper_path"]),
        label="c063-f072a-base", json=_fwd(os.path.join(root, "report.json")),
        child=_fwd(child)))
    rc, out, err = _run([_bash(), _fwd(script)])
    obs = _observe_lock(lock, marker, rc, before)
    obs["shell_stdout_tail"] = out.strip().splitlines()[-3:]
    obs["acquire_echoed"] = "ACQUIRED" in out
    obs["stderr_tail"] = err.strip().splitlines()[-2:]
    return obs


def f072_acquire_tip(wrapper: str, work: str, tag: str) -> Dict[str, Any]:
    """Someone else holds the mutex; *wrapper* is asked to take it."""

    root = os.path.join(work, f"f072a_{tag}")
    lock = os.path.join(root, "scratch_heavylock")
    marker = os.path.join(root, "child_ran.txt")
    os.makedirs(root, exist_ok=True)
    before = _plant_foreign_lock(lock)
    child = _probe_child(os.path.join(root, "probe.py"), _fwd(marker))
    rc, _out, err = _run([
        PY, wrapper, "--label", f"c063-f072a-{tag}", "--timeout", "60", "--quiet",
        "--heavy-lock", "C-063-PROBE", "--heavy-lock-path", lock,
        "--json", os.path.join(root, "report.json"), "--", PY, child])
    obs = _observe_lock(lock, marker, rc, before)
    obs["held_marker_printed"] = "BOUNDED_RUN_HEAVY_LOCK_HELD" in err
    obs["other_holder_named"] = FOREIGN_HOLDER["holder"] in err
    obs["report_written_for_a_job_that_never_ran"] = os.path.exists(
        os.path.join(root, "report.json"))
    return obs


def f072_release_base(work: str) -> Dict[str, Any]:
    """The lock is free; the job takes it, the holder file changes under it, and
    the documented protocol's unconditional ``rm -rf`` removes it regardless."""

    root = os.path.join(work, "f072r_base")
    lock = os.path.join(root, "scratch_heavylock")
    os.makedirs(root, exist_ok=True)
    shutil.rmtree(lock, ignore_errors=True)
    child = _steal_child(os.path.join(root, "steal.py"),
                         _fwd(os.path.join(lock, HOLDER_FILENAME)))
    script = _write_script(os.path.join(root, "dance.sh"), _SHELL_DANCE.format(
        lock=_fwd(lock), py=_fwd(PY), wrapper=_fwd(BASE["wrapper_path"]),
        label="c063-f072r-base", json=_fwd(os.path.join(root, "report.json")),
        child=_fwd(child)))
    rc, out, _err = _run([_bash(), _fwd(script)])
    return {
        "acquire_echoed": "ACQUIRED" in out,
        "lock_survives": os.path.isdir(lock),
        "foreign_holder_still_named":
            FOREIGN_HOLDER["holder"].encode() in _holder_bytes(lock),
        "exit_code": rc,
    }


def f072_release_tip(wrapper: str, work: str, tag: str) -> Dict[str, Any]:
    """The lock is free; *wrapper* takes it, the holder file changes under it."""

    root = os.path.join(work, f"f072r_{tag}")
    lock = os.path.join(root, "scratch_heavylock")
    os.makedirs(root, exist_ok=True)
    shutil.rmtree(lock, ignore_errors=True)
    child = _steal_child(os.path.join(root, "steal.py"),
                         os.path.join(lock, HOLDER_FILENAME))
    rc, _out, err = _run([
        PY, wrapper, "--label", f"c063-f072r-{tag}", "--timeout", "60", "--quiet",
        "--heavy-lock", "C-063-OWNER", "--heavy-lock-path", lock,
        "--json", os.path.join(root, "report.json"), "--", PY, child])
    return {
        "acquire_echoed": True,
        "lock_survives": os.path.isdir(lock),
        "foreign_holder_still_named":
            FOREIGN_HOLDER["holder"].encode() in _holder_bytes(lock),
        "refusal_marker_printed":
            "BOUNDED_RUN_HEAVY_LOCK_RELEASE_REFUSED" in err,
        "exit_code": rc,
    }


# --------------------------------------------------------------------------- #
# F-071 -- the G11 report lifecycle
# --------------------------------------------------------------------------- #


def _g11(module: str, *args: str) -> Tuple[int, str, str]:
    """Run a REAL ``g11_evidence.py`` -- base or tip -- over its own tree.

    ``REPORTS_ROOT`` is the module's own directory, so placing each arm's copy in
    its own scratch directory points the real allocator and the real checker at a
    scratch reports tree without a line of the module being changed. The
    committed evidence tree is never read or written by this harness.
    """

    return _run([PY, module, *args])


def f071_arm(g11_module: str, wrapper: str, work: str, tag: str) -> Dict[str, Any]:
    """Seed one finished job, then kill a wrapper mid-job. Ask ``check``.

    The killed arm is F-071's mechanism exactly: the wrapper's own ``--timeout``
    is 300 s and never fires; what ends the job is the PARENT being killed, which
    is what an agent's tool wall clock does. The seed job is not decoration --
    without it the tip arm could pass by never producing evidence at all, and a
    whole-tree check over an empty tree proves nothing.
    """

    reports = os.path.dirname(g11_module)
    obs: Dict[str, Any] = {"reports_root": reports}

    # ---- 1. a job that FINISHES. Its report must reach the reports tree. ----
    rc, out, _err = _g11(g11_module, "next", "--task", SCRATCH_TASK, "--label", "seed")
    seed = out.strip().splitlines()[-1].strip()
    obs["seed_allocated"] = seed
    seed_rc, _o, _e = _run([PY, wrapper, "--label", f"c063-f071-{tag}-seed",
                            "--timeout", "60", "--quiet", "--json", seed,
                            "--", PY, "-c", "print('seed')"])
    obs["seed_exit"] = seed_rc
    obs["seed_report_published"] = os.path.isfile(seed)

    # ---- 2. a job KILLED mid-flight, by its parent, not by its timeout ----
    rc, out, _err = _g11(g11_module, "next", "--task", SCRATCH_TASK,
                         "--label", "killed")
    killed = out.strip().splitlines()[-1].strip()
    obs["killed_allocated"] = killed
    marker = os.path.join(work, f"f071_{tag}_started.txt")
    child = _write_script(os.path.join(work, f"f071_{tag}_child.py"),
                          f"open(r'{marker}', 'w').write('x')\n"
                          "import time; time.sleep(600)\n")
    proc = subprocess.Popen(
        [PY, wrapper, "--label", f"c063-f071-{tag}-killed", "--timeout", "300",
         "--quiet", "--json", killed, "--", PY, child],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30.0
    while not os.path.exists(marker) and time.monotonic() < deadline:
        time.sleep(0.2)
    obs["killed_job_was_in_flight"] = os.path.exists(marker)
    descendants = _descendants(proc.pid)
    proc.kill()
    proc.wait(timeout=60)
    obs["owned_pids_after_kill"] = _settle([proc.pid, *descendants])

    # ---- 3. the consumer that actually gates merges ----
    task_dir = os.path.join(reports, SCRATCH_TASK)
    obs["reports_tree_listing"] = sorted(os.listdir(task_dir))
    obs["placeholder_in_reports_tree"] = sorted(
        name for name in os.listdir(task_dir)
        if name.endswith(".json") and _is_placeholder(os.path.join(task_dir, name)))
    whole_rc, whole_out, _e = _g11(g11_module, "check")
    task_rc, task_out, _e = _g11(g11_module, "check", "--task", SCRATCH_TASK)
    obs["whole_tree_check_exit"] = whole_rc
    obs["whole_tree_check_tail"] = whole_out.strip().splitlines()[-4:]
    obs["whole_tree_says_never_written"] = "report_never_written" in whole_out
    obs["task_check_exit"] = task_rc
    obs["task_check_tail"] = task_out.strip().splitlines()[-3:]
    return obs


def _is_placeholder(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("g11_reserved") is True


def _descendants(pid: int) -> List[int]:
    return bounded_run.descendants_of(pid, bounded_run.snapshot_processes())


def _settle(pids: Sequence[int], seconds: float = 15.0) -> List[int]:
    """Which of *pids* are still alive after the kill has had time to land."""

    deadline = time.monotonic() + seconds
    alive = list(pids)
    while alive and time.monotonic() < deadline:
        table = bounded_run.snapshot_processes()
        alive = [pid for pid in pids if pid in table]
        if not alive:
            break
        time.sleep(0.2)
    return alive


# --------------------------------------------------------------------------- #
# Signatures -- what "the defect" and "the fix" look like, as predicates
# --------------------------------------------------------------------------- #


def f072_acquire_defect(obs: Dict[str, Any]) -> List[Tuple[str, bool]]:
    return [
        ("the job RAN while another holder held the lock", obs["child_ran"] is True),
        ("the foreign holder's lock was REMOVED", obs["lock_survives"] is False),
    ]


def f072_acquire_fixed(obs: Dict[str, Any]) -> List[Tuple[str, bool]]:
    return [
        ("the job did NOT run", obs["child_ran"] is False),
        ("the lock SURVIVED", obs["lock_survives"] is True),
        ("it still names the foreign holder, byte for byte",
         obs["holder_unchanged"] is True),
        (f"exit {bounded_run.EXIT_HEAVY_LOCK_UNAVAILABLE} (EXIT_HEAVY_LOCK_UNAVAILABLE)",
         obs["exit_code"] == bounded_run.EXIT_HEAVY_LOCK_UNAVAILABLE),
        ("the holder was printed, not merely refused",
         obs.get("other_holder_named") is True),
        ("no cleanup report certifies a job that never started",
         obs.get("report_written_for_a_job_that_never_ran") is False),
    ]


def f072_release_defect(obs: Dict[str, Any]) -> List[Tuple[str, bool]]:
    return [
        ("the lock was removed although it named someone else",
         obs["lock_survives"] is False),
    ]


def f072_release_fixed(obs: Dict[str, Any]) -> List[Tuple[str, bool]]:
    return [
        ("the lock was LEFT IN PLACE", obs["lock_survives"] is True),
        ("it still names the other holder",
         obs["foreign_holder_still_named"] is True),
        (f"exit {bounded_run.EXIT_HEAVY_LOCK_RELEASE_REFUSED} (EXIT_HEAVY_LOCK_RELEASE_REFUSED)",
         obs["exit_code"] == bounded_run.EXIT_HEAVY_LOCK_RELEASE_REFUSED),
        ("the refusal was announced", obs.get("refusal_marker_printed") is True),
    ]


def f071_defect(obs: Dict[str, Any]) -> List[Tuple[str, bool]]:
    return [
        ("the killed job really was in flight", obs["killed_job_was_in_flight"]),
        ("a g11_reserved placeholder SURVIVED in the reports tree",
         bool(obs["placeholder_in_reports_tree"])),
        ("whole-tree check is RED -- merge gate 10 fails",
         obs["whole_tree_check_exit"] == 1),
        ("...and it says report_never_written",
         obs["whole_tree_says_never_written"] is True),
        ("no owned process leaked (the asymmetry F-071 names)",
         obs["owned_pids_after_kill"] == []),
    ]


def f071_fixed(obs: Dict[str, Any]) -> List[Tuple[str, bool]]:
    return [
        ("the killed job really was in flight", obs["killed_job_was_in_flight"]),
        ("NOTHING was left in the reports tree",
         obs["placeholder_in_reports_tree"] == []),
        ("whole-tree check is GREEN", obs["whole_tree_check_exit"] == 0),
        ("check is silent about a never-written report",
         obs["whole_tree_says_never_written"] is False),
        ("a job that FINISHED still published its report",
         obs["seed_report_published"] is True),
        ("...so `check --task` is green over real evidence, not over nothing",
         obs["task_check_exit"] == 0),
        ("no owned process leaked", obs["owned_pids_after_kill"] == []),
    ]


# --------------------------------------------------------------------------- #

BASE: Dict[str, Any] = {}
RESULTS: List[Dict[str, Any]] = []


def _emit(name: str, kind: str, checks: List[Tuple[str, bool]],
          obs: Dict[str, Any], expect_red: bool = False) -> bool:
    """Record one arm. *expect_red* inverts the verdict for a RULING 13 arm."""

    failed = [label for label, held in checks if not held]
    ok = bool(failed) if expect_red else not failed
    RESULTS.append({
        "arm": name, "kind": kind, "expect_red": expect_red,
        "checks": [{"assertion": label, "held": held} for label, held in checks],
        "assertions_that_went_red": failed,
        "verdict": "PASS" if ok else "FAIL",
        "observations": obs,
    })
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({kind})")
    for label, held in checks:
        print(f"        {'ok ' if held else 'RED'}  {label}")
    if expect_red:
        print(f"        -> {len(failed)} assertion(s) went red, as a neutralized "
              f"guard must")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, help="scratch directory (wiped)")
    parser.add_argument("--out", required=True, help="where to write the artifact")
    args = parser.parse_args(list(argv) if argv is not None else None)

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    # ---- the real modules, base and tip, digest-checked (F-074) ----
    BASE["wrapper_path"] = os.path.join(work, "base_modules", "bounded_run.py")
    base_wrapper_digest = materialise_base(BASE["wrapper_path"], WRAPPER_REL)
    base_g11 = os.path.join(work, "base_reports", "g11_evidence.py")
    base_g11_digest = materialise_base(base_g11, G11_REL)
    tip_g11 = os.path.join(work, "tip_reports", "g11_evidence.py")
    tip_g11_digest = materialise_tip(tip_g11, TIP_G11)

    provenance = {
        "base_sha": BASE_SHA,
        "base_bounded_run": {"path": BASE["wrapper_path"],
                             "digest": base_wrapper_digest},
        "base_g11_evidence": {"path": base_g11, "digest": base_g11_digest},
        "tip_bounded_run": {"path": TIP_WRAPPER,
                            "digest": _sha256_file(TIP_WRAPPER)},
        "tip_g11_evidence": {"path": tip_g11, "digest": tip_g11_digest,
                             "source": TIP_G11},
        "base_and_tip_wrappers_differ":
            base_wrapper_digest != _sha256_file(TIP_WRAPPER),
        "base_and_tip_g11_differ": base_g11_digest != tip_g11_digest,
        "bash": _bash(),
    }
    print("=" * 78)
    print(f"C-063 G9 base-vs-tip harness   base={BASE_SHA[:7]}  python={sys.version.split()[0]}")
    for key in ("base_bounded_run", "base_g11_evidence", "tip_bounded_run",
                "tip_g11_evidence"):
        print(f"  {key:<20}: {provenance[key]['digest']}")
    print("=" * 78)

    outcomes: List[bool] = []

    # ---------------- F-072 (a): a failed acquire must stop the job ----------
    base_a = f072_acquire_base(work)
    outcomes.append(_emit(
        "F-072a base @ ac77668 -- shell protocol, real base wrapper",
        "base reproduces the defect", f072_acquire_defect(base_a), base_a))

    tip_a = f072_acquire_tip(TIP_WRAPPER, work, "tip")
    outcomes.append(_emit("F-072a tip -- --heavy-lock on the real tip wrapper",
                          "tip removes the defect", f072_acquire_fixed(tip_a), tip_a))

    n3 = neutralize(TIP_WRAPPER, os.path.join(work, "neutralized", "n3.py"),
                    "        if not lock.acquire():",
                    "        if lock.acquire() and False:  # NEUTRALIZED (RULING 13)")
    neut_a = f072_acquire_tip(os.path.join(work, "neutralized", "n3.py"), work, "n3")
    neut_a["neutralization"] = n3
    outcomes.append(_emit(
        "F-072a NEUTRALIZED -- 'a failed acquire stops the job' removed",
        "RULING 13: the harness must go red",
        f072_acquire_fixed(neut_a), neut_a, expect_red=True))

    # ---------------- F-072 (b): release must refuse a foreign holder -------
    base_r = f072_release_base(work)
    outcomes.append(_emit("F-072b base @ ac77668 -- unconditional rm -rf",
                          "base reproduces the defect",
                          f072_release_defect(base_r), base_r))

    tip_r = f072_release_tip(TIP_WRAPPER, work, "tip")
    outcomes.append(_emit("F-072b tip -- release checks the holder file first",
                          "tip removes the defect", f072_release_fixed(tip_r), tip_r))

    n4 = neutralize(
        TIP_WRAPPER, os.path.join(work, "neutralized", "n4.py"),
        '        if not isinstance(data, dict) or data.get("token") != self.token:',
        "        if False:  # NEUTRALIZED (RULING 13)")
    neut_r = f072_release_tip(os.path.join(work, "neutralized", "n4.py"), work, "n4")
    neut_r["neutralization"] = n4
    outcomes.append(_emit(
        "F-072b NEUTRALIZED -- the holder-identity check removed",
        "RULING 13: the harness must go red",
        f072_release_fixed(neut_r), neut_r, expect_red=True))

    # ---------------- F-071: a reservation must not outlive its job ---------
    base_71 = f071_arm(base_g11, BASE["wrapper_path"], work, "base")
    outcomes.append(_emit("F-071 base @ ac77668 -- real base allocator + wrapper",
                          "base reproduces the defect",
                          f071_defect(base_71), base_71))

    tip_71 = f071_arm(tip_g11, TIP_WRAPPER, work, "tip")
    outcomes.append(_emit("F-071 tip -- staging + promotion",
                          "tip removes the defect", f071_fixed(tip_71), tip_71))

    n1_dir = os.path.join(work, "n1_reports")
    n1 = neutralize(tip_g11, os.path.join(n1_dir, "g11_evidence.py"),
                    "    return os.path.join(directory, STAGING_DIRNAME, name)",
                    "    return os.path.join(directory, name)  # NEUTRALIZED (RULING 13)")
    neut_71a = f071_arm(os.path.join(n1_dir, "g11_evidence.py"), TIP_WRAPPER, work, "n1")
    neut_71a["neutralization"] = n1
    outcomes.append(_emit(
        "F-071 NEUTRALIZED (staging) -- the reservation goes back in the reports tree",
        "RULING 13: the harness must go red",
        f071_fixed(neut_71a), neut_71a, expect_red=True))

    n2_wrapper = os.path.join(work, "neutralized", "n2.py")
    n2 = neutralize(TIP_WRAPPER, n2_wrapper,
                    "            os.replace(staged, json_path)",
                    "            pass  # NEUTRALIZED (RULING 13)")
    n2_dir = os.path.join(work, "n2_reports")
    materialise_tip(os.path.join(n2_dir, "g11_evidence.py"), TIP_G11)
    neut_71b = f071_arm(os.path.join(n2_dir, "g11_evidence.py"), n2_wrapper, work, "n2")
    neut_71b["neutralization"] = n2
    outcomes.append(_emit(
        "F-071 NEUTRALIZED (promotion) -- the atomic publish removed",
        "RULING 13: the harness must go red",
        f071_fixed(neut_71b), neut_71b, expect_red=True))

    payload = {
        "task": "C-063",
        "findings": ["F-071", "F-072"],
        "provenance": provenance,
        "arms": RESULTS,
        "arms_passed": sum(1 for r in RESULTS if r["verdict"] == "PASS"),
        "arms_total": len(RESULTS),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)

    print()
    print("=" * 78)
    for row in RESULTS:
        print(f"{row['verdict']:<6}{row['kind']:<34}{row['arm']}")
    print("=" * 78)
    ok = all(outcomes)
    print(f"{payload['arms_passed']}/{payload['arms_total']} arms as expected "
          f"-> {os.path.abspath(args.out)}")
    print("RESULT:", "G9 DELTA PROVED, BOTH GUARDS NON-VACUOUS"
          if ok else "NOT PROVED -- STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
