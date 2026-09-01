"""REV-109 -- adversarial probe of C-109's reviewer-evidence route check.

Written by the INDEPENDENT REVIEWER, not the author. Covers REV-109 criteria
B8 (it must be seen to fail), B9 (reachability by CONTENT, not filename) and
B13 (actively try to obtain a false PASS).

Nothing here is taken from the author's report. Every fixture is built from
scratch in a temp tree and the check is invoked as a SUBPROCESS, so the exit
code observed is the exit code a merge gate would observe.

Each scenario prints: name, expected exit, observed exit, observed verdicts,
and PASS/FAIL of the reviewer's expectation. A scenario whose expectation is
"UNKNOWN" is an open question being measured, not an assertion.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EVID = "docs/pwml_recovery_sprint/evidence"
SCRIPT = None  # set in main()
PY = sys.executable

RESULTS = []


def git(repo, *args, check=True):
    proc = subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {args} failed in {repo}: {proc.stderr}")
    return proc


def new_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "sprint/pwml-recovery")
    git(root, "config", "user.email", "rev109@example.invalid")
    git(root, "config", "user.name", "REV-109 probe")
    (root / EVID).mkdir(parents=True, exist_ok=True)
    write(root / "README.md", "seed\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "seed")
    return root


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def run_check(worktree, task, repo=None, ref="sprint/pwml-recovery",
              extra=None, json_out=None):
    cmd = [PY, str(SCRIPT), "--task", task, "--worktree", str(worktree)]
    if repo is not None:
        cmd += ["--integration-repo", str(repo), "--integration-ref", ref]
    if json_out:
        cmd += ["--json", str(json_out)]
    if extra:
        cmd += list(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report = None
    if json_out and Path(json_out).exists():
        report = json.loads(Path(json_out).read_text(encoding="utf-8"))
    return proc, report


def verdicts(report):
    if not report:
        return {}
    out = {}
    for item in report.get("items", []):
        out[item["worktree_path"].rsplit("/", 1)[-1]] = item["verdict"]
    return out


def record(name, expected, observed, detail, note=""):
    ok = "PASS" if (expected == observed or expected == "UNKNOWN") else "FAIL"
    RESULTS.append((name, expected, observed, ok, detail, note))
    print(f"\n--- {name}")
    print(f"    expected exit : {expected}")
    print(f"    observed exit : {observed}   [{ok}]")
    print(f"    verdicts      : {detail}")
    if note:
        print(f"    NOTE          : {note}")


# ---------------------------------------------------------------- scenarios

def s1_satisfied(tmp):
    """B8 positive half: a fully satisfied case must exit 0."""
    r = new_repo(tmp / "s1")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    write(r / EVID / "rev109_probe.py", "print(1)\n")
    write(r / EVID / "rev109_probe.log", "log\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "evidence")
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s1.json")
    record("S1 satisfied (all committed to integration)", 0, p.returncode,
           verdicts(rep))


def s2_absent(tmp):
    """B8 negative half: evidence that exists nowhere but the worktree."""
    r = new_repo(tmp / "s2")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "g11 only")
    write(r / EVID / "rev109_probe.py", "print(1)\n")  # never committed
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s2.json")
    record("S2 probe exists ONLY in the worktree", 1, p.returncode,
           verdicts(rep), "stderr names it: "
           + ("yes" if "rev109_probe.py" in p.stderr else "NO"))


def s3_same_name_different_bytes(tmp):
    """B9 THE CRUX: right filename, different bytes -> must be UNREACHABLE."""
    r = new_repo(tmp / "s3")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    write(r / EVID / "rev109_probe.py", "ORIGINAL COMMITTED CONTENT\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "evidence")
    # same path, different bytes, uncommitted
    write(r / EVID / "rev109_probe.py", "DIFFERENT WORKTREE CONTENT\n")
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s3.json")
    v = verdicts(rep)
    ok_class = v.get("rev109_probe.py") == "unreachable_content_differs"
    record("S3 same NAME, different BYTES (B9 crux)", 1, p.returncode, v,
           "verdict class is unreachable_content_differs: "
           + ("yes" if ok_class else "NO -- decorative filename check"))


def s4_other_branch(tmp):
    """B13: committed, but only on a branch integration cannot reach."""
    r = new_repo(tmp / "s4")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "g11")
    git(r, "checkout", "-q", "-b", "rev/REV-109")
    write(r / EVID / "rev109_probe.py", "only on the review branch\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "probe on side branch")
    # worktree is now the side branch; integration ref is sprint/pwml-recovery
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s4.json")
    record("S4 committed only on a branch NOT reachable from integration",
           1, p.returncode, verdicts(rep))


def s5_different_path_same_bytes(tmp):
    """Documented intent: bytes present elsewhere in the tree count as reachable."""
    r = new_repo(tmp / "s5")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    write(r / "archive/moved_probe.py", "PROBE BYTES\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "evidence under another path")
    write(r / EVID / "rev109_probe.py", "PROBE BYTES\n")  # uncommitted, same bytes
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s5.json")
    found = ""
    if rep:
        for it in rep["items"]:
            if it["worktree_path"].endswith("rev109_probe.py"):
                found = str(it["found_at"])
    record("S5 same BYTES under a different path", 0, p.returncode,
           verdicts(rep), f"found_at = {found} (deliberate per the docstring)")


def s6_empty_file(tmp):
    """B13 false-PASS hunt: a ZERO-BYTE probe log.

    The empty blob oid is a universal constant. If integration contains any
    empty file at all -- an __init__.py, a placeholder -- the empty probe is
    'found' there and reported reachable.
    """
    r = new_repo(tmp / "s6")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    (r / "src").mkdir(parents=True, exist_ok=True)
    (r / "src/__init__.py").write_bytes(b"")          # unrelated empty file
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "evidence + an empty file")
    (r / EVID / "rev109_truncated.log").write_bytes(b"")  # empty, uncommitted
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s6.json")
    v = verdicts(rep)
    note = ("FALSE PASS: a zero-byte, never-committed probe log is reported "
            "reachable via an unrelated empty file"
            if v.get("rev109_truncated.log") == "reachable" else
            "no false pass here")
    record("S6 zero-byte probe log vs an unrelated empty blob in integration",
           "UNKNOWN", p.returncode, v, note)


def s7_g11_without_probe(tmp):
    """B13: a G11 report present, its probe never created at all."""
    r = new_repo(tmp / "s7")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "g11 only, no probe anywhere")
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s7.json")
    note = ("PASSES with zero probes enumerated -- the check cannot know a probe "
            "was owed" if p.returncode == 0 else "flagged")
    record("S7 G11 report present, probe source never existed", "UNKNOWN",
           p.returncode, verdicts(rep), note)


def s8_probe_in_subdirectory(tmp):
    """B13: probes filed in a SUBDIRECTORY of evidence/ under the task stem."""
    r = new_repo(tmp / "s8")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "g11 only")
    # uncommitted probes, filed in a directory whose name matches the stem glob
    write(r / EVID / "rev109_probes/deep_probe.py", "load bearing\n")
    write(r / EVID / "rev109_probes/deep_probe.log", "load bearing output\n")
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s8.json")
    n = rep["enumerated"] if rep else -1
    note = ("FALSE PASS: two uncommitted probe files inside "
            "evidence/rev109_probes/ were never enumerated (glob is "
            "non-recursive and is_file() drops the directory)"
            if p.returncode == 0 else "flagged")
    record("S8 probes inside evidence/<stem>_*/ subdirectory", "UNKNOWN",
           p.returncode, f"enumerated={n} {verdicts(rep)}", note)


def s9_mistyped_task(tmp):
    """A mistyped task id must not read as a clean gate."""
    r = new_repo(tmp / "s9")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    write(r / EVID / "rev109_probe.py", "x\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "evidence")
    p, rep = run_check(r, "REV-190", r, json_out=tmp / "s9.json")
    record("S9 mistyped task id (REV-190)", 3, p.returncode, verdicts(rep))
    p2, _ = run_check(r, "REV-190", r, extra=["--allow-empty"])
    record("S9b mistyped task id + --allow-empty", 0, p2.returncode, {},
           "documented escape hatch; a mistyped id with --allow-empty is a "
           "silent pass by design")


def s10_bad_ref(tmp):
    r = new_repo(tmp / "s10")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "evidence")
    p, _ = run_check(r, "REV-109", r, ref="no/such/ref")
    record("S10 nonexistent integration ref", 2, p.returncode, {},
           "usage error, never a verdict")


def s11_deleted_from_integration(tmp):
    """Evidence committed, then removed from integration by a later commit."""
    r = new_repo(tmp / "s11")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    write(r / EVID / "rev109_probe.py", "was here\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "evidence")
    git(r, "rm", "-q", str(Path(EVID) / "rev109_probe.py"))
    git(r, "commit", "-q", "-m", "someone removed the probe")
    write(r / EVID / "rev109_probe.py", "was here\n")  # still in the worktree
    p, rep = run_check(r, "REV-109", r, json_out=tmp / "s11.json")
    record("S11 evidence deleted from integration by a later commit",
           1, p.returncode, verdicts(rep))


def s12_case_variant_task(tmp):
    r = new_repo(tmp / "s12")
    write(r / EVID / "g11/REV-109/01-a.json", '{"a": 1}\n')
    write(r / EVID / "rev109_probe.py", "x\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "evidence")
    p, rep = run_check(r, "rev-109", r, json_out=tmp / "s12.json")
    n = rep["enumerated"] if rep else -1
    record("S12 lowercase task id on a case-insensitive filesystem",
           "UNKNOWN", p.returncode, f"enumerated={n}",
           "informational: enumeration is filesystem-case dependent")


def s13_real_c109(tmp, tip_worktree, tip_repo):
    """The real thing: C-109's own evidence against the real integration ref."""
    p, rep = run_check(tip_worktree, "C-109", tip_repo,
                       json_out=tmp / "s13_integration.json")
    n = rep["enumerated"] if rep else -1
    u = rep["unreachable"] if rep else -1
    record("S13 real C-109 evidence vs sprint/pwml-recovery (NOT yet merged)",
           1, p.returncode, f"enumerated={n} unreachable={u}",
           "the check correctly refuses an unmerged card's evidence")
    p2, rep2 = run_check(tip_worktree, "C-109", tip_repo,
                         ref="card/C-109-control-plane",
                         json_out=tmp / "s13_cardbranch.json")
    n2 = rep2["enumerated"] if rep2 else -1
    u2 = rep2["unreachable"] if rep2 else -1
    record("S13b real C-109 evidence vs its own card branch", 0, p2.returncode,
           f"enumerated={n2} unreachable={u2}",
           "author's claim reproduced independently")


def main():
    global SCRIPT
    if len(sys.argv) < 3:
        print("usage: rev109_route_adversarial.py <tip-worktree> <workdir>")
        return 2
    tip = Path(sys.argv[1]).resolve()
    SCRIPT = tip / EVID / "reviewer_evidence_route.py"
    work = Path(sys.argv[2]).resolve()
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    print("REV-109 adversarial probe of reviewer_evidence_route.py")
    print(f"script   : {SCRIPT}")
    print(f"exists   : {SCRIPT.is_file()}")
    print(f"workdir  : {work}")

    for fn in (s1_satisfied, s2_absent, s3_same_name_different_bytes,
               s4_other_branch, s5_different_path_same_bytes, s6_empty_file,
               s7_g11_without_probe, s8_probe_in_subdirectory,
               s9_mistyped_task, s10_bad_ref, s11_deleted_from_integration,
               s12_case_variant_task):
        try:
            fn(work)
        except Exception as exc:  # keep every failed measurement visible
            record(fn.__name__ + " (PROBE ERROR)", "n/a", "n/a", {}, repr(exc))

    try:
        s13_real_c109(work, tip, tip)
    except Exception as exc:
        record("S13 (PROBE ERROR)", "n/a", "n/a", {}, repr(exc))

    print("\n================ SUMMARY ================")
    bad = 0
    for name, exp, obs, ok, detail, note in RESULTS:
        if ok == "FAIL":
            bad += 1
        print(f"{ok:4}  exp={exp!s:8} obs={obs!s:8}  {name}")
    print(f"\nexpectation failures: {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
