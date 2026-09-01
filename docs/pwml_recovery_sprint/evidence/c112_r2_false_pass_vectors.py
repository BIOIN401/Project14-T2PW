"""C-112 -- G9 base-vs-tip proof for the three REV-109 R2 false-PASS vectors.

**Measured against the COMMITTED BLOB of ``reviewer_evidence_route.py`` at the C-112 base
SHA, never a working tree.** That is F-157's own lesson: a claim checked against a working
copy is a claim about bytes that exist in no commit. The base module is materialised with
``git show <base>:<path>`` into a temp file and imported from there.

**Symbol absence is not proof**, so nothing here greps for a name. Each vector is executed
end to end on both modules with the SAME fixture, and both directions are recorded:

* at BASE the check must return **0 -- a green, silent, false PASS**;
* at TIP the same fixture must go **red or refused**;
* and at TIP the *good* fixture must still go **green**, so the fix is not just "fail
  more".

Before any vector is trusted the harness asserts a **known-positive and a known-negative**
on both modules: an all-committed fixture is 0, and an item withheld from integration is
non-zero. If those two disagree the probe never reached the code and every result below
would be an artefact -- the Lead's first C-108 verification probe failed exactly that way,
returned all-permissive at base, and looked precisely like a finding.

Usage::  <venv-python> c112_r2_false_pass_vectors.py <worktree-root> <base-sha>
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EVID = "docs/pwml_recovery_sprint/evidence"
SCRIPT_REL = f"{EVID}/reviewer_evidence_route.py"
TASK = "REV-109"
STEM = "rev109"

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialise_base(root: Path, sha: str, dest: Path) -> Path:
    proc = subprocess.run(
        [shutil.which("git"), "-C", str(root), "show", f"{sha}:{SCRIPT_REL}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit(f"cannot read {SCRIPT_REL} at {sha}: {proc.stderr}")
    dest.write_text(proc.stdout, encoding="utf-8")
    return dest


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def flat_fixture(root: Path) -> dict[str, str]:
    contents = {
        f"{EVID}/g11/{TASK}/01-{STEM}-focused.json": json.dumps({"label": "focused"}),
        f"{EVID}/{STEM}_boundary_probe.py": "print('the probe source')\n",
        f"{EVID}/{STEM}_boundary_probe.log": "the probe output, 3 findings\n",
    }
    for rel, text in contents.items():
        write(root, rel, text)
    return contents


def run(module, tmp: Path, tag: str, **kw) -> tuple[int, dict]:
    out = tmp / f"{tag}.json"
    argv = ["--task", kw.pop("task", TASK),
            "--worktree", str(kw.pop("worktree")),
            "--integration-dir", str(kw.pop("integration")),
            "--json", str(out)]
    if kw.pop("allow_empty", False):
        argv.append("--allow-empty")
    assert not kw, kw
    code = module.run(argv)
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    return code, report


def sanity(module, label: str, tmp: Path) -> None:
    """Known-positive AND known-negative, before any vector result is trusted."""
    wt, integ = tmp / f"{label}_s_wt", tmp / f"{label}_s_int"
    contents = flat_fixture(wt)
    for rel, text in contents.items():
        write(integ, rel, text)
    code_pos, _ = run(module, tmp, f"{label}_pos", worktree=wt, integration=integ)
    check(f"{label}: KNOWN-POSITIVE, everything committed -> 0", code_pos == 0, f"exit={code_pos}")

    wt2, integ2 = tmp / f"{label}_n_wt", tmp / f"{label}_n_int"
    contents2 = flat_fixture(wt2)
    withheld = f"{EVID}/{STEM}_boundary_probe.log"
    for rel, text in contents2.items():
        if rel != withheld:
            write(integ2, rel, text)
    code_neg, _ = run(module, tmp, f"{label}_neg", worktree=wt2, integration=integ2)
    check(f"{label}: KNOWN-NEGATIVE, one item withheld -> non-zero", code_neg != 0,
          f"exit={code_neg}")


# ---------------------------------------------------------------- vector 1

def vector1(base, tip, tmp: Path) -> None:
    print("\n=== VECTOR 1 -- a probe SUBDIRECTORY is dropped silently "
          "(the one that matters) ===")
    deep = f"{EVID}/{STEM}_probes/deep_probe.py"
    results = {}
    for label, module in (("BASE", base), ("TIP", tip)):
        wt, integ = tmp / f"v1_{label}_wt", tmp / f"v1_{label}_int"
        contents = flat_fixture(wt)
        write(wt, deep, "print('REV-109 demonstrated the vector with exactly this')\n")
        for rel, text in contents.items():
            write(integ, rel, text)          # every FLAT item committed; the dir is NOT
        code, report = run(module, tmp, f"v1_{label}", worktree=wt, integration=integ)
        results[label] = (code, report)
        print(f"  {label}: exit={code} enumerated={report.get('enumerated')} "
              f"unreachable={report.get('unreachable')}")

    base_code, base_report = results["BASE"]
    tip_code, tip_report = results["TIP"]
    check("BASE: green, silent FALSE PASS -- exit 0 with the subdirectory never enumerated",
          base_code == 0, f"exit={base_code} enumerated={base_report.get('enumerated')}")
    check("BASE: the subdirectory's file is absent from the enumeration entirely",
          deep not in [i["worktree_path"] for i in base_report.get("items", [])])
    check("TIP: goes RED on the same fixture", tip_code == 1, f"exit={tip_code}")
    check("TIP: names the subdirectory's file as unreachable",
          deep in [i["worktree_path"] for i in tip_report.get("items", [])
                   if i["verdict"] != "reachable"])
    check("TIP: announces the probe directory LOUDLY rather than dropping it",
          tip_report.get("probe_directories") == [f"{EVID}/{STEM}_probes"],
          str(tip_report.get("probe_directories")))

    # ...and TIP must go GREEN once the subdirectory is committed: not merely "fail more".
    wt, integ = tmp / "v1_TIP_wt", tmp / "v1_TIP_int"
    write(integ, deep, (wt / deep).read_text(encoding="utf-8"))
    code_ok, report_ok = run(tip, tmp, "v1_TIP_fixed", worktree=wt, integration=integ)
    check("TIP: GREEN once the subdirectory's bytes are committed", code_ok == 0,
          f"exit={code_ok} enumerated={report_ok.get('enumerated')}")


# ---------------------------------------------------------------- vector 2

def vector2(base, tip, tmp: Path) -> None:
    print("\n=== VECTOR 2 -- a ZERO-BYTE file always reports reachable ===")
    truncated = f"{EVID}/{STEM}_truncated_probe.log"
    results = {}
    for label, module in (("BASE", base), ("TIP", tip)):
        wt, integ = tmp / f"v2_{label}_wt", tmp / f"v2_{label}_int"
        contents = flat_fixture(wt)
        write(wt, truncated, "")                     # the failed redirect
        for rel, text in contents.items():
            write(integ, rel, text)
        write(integ, f"{EVID}/.gitkeep", "")         # the real unrelated empty file
        code, report = run(module, tmp, f"v2_{label}", worktree=wt, integration=integ)
        results[label] = (code, report)
        verdicts = {i["worktree_path"]: i["verdict"] for i in report.get("items", [])}
        print(f"  {label}: exit={code} verdict(truncated)={verdicts.get(truncated)!r} "
              f"found_at={[i['found_at'] for i in report.get('items', []) if i['worktree_path'] == truncated]}")

    base_code, base_report = results["BASE"]
    tip_code, tip_report = results["TIP"]
    base_v = {i["worktree_path"]: i for i in base_report.get("items", [])}
    tip_v = {i["worktree_path"]: i for i in tip_report.get("items", [])}
    check("BASE: FALSE PASS -- exit 0", base_code == 0, f"exit={base_code}")
    check("BASE: the empty probe log is called 'reachable'",
          base_v[truncated]["verdict"] == "reachable")
    check("BASE: and it 'resolved' against an UNRELATED empty file",
          base_v[truncated]["found_at"] == [f"{EVID}/.gitkeep"],
          str(base_v[truncated]["found_at"]))
    check("TIP: goes RED", tip_code == 1, f"exit={tip_code}")
    check("TIP: its own verdict class, not 'reachable' and not 'absent'",
          tip_v[truncated]["verdict"] == "empty_blob_indeterminate",
          tip_v[truncated]["verdict"])
    check("TIP: exit_reason names it", tip_report.get("exit_reason") == "indeterminate_empty_evidence",
          str(tip_report.get("exit_reason")))

    wt, integ = tmp / "v2_TIP_wt", tmp / "v2_TIP_int"
    said = "probe ran; 0 findings. Said so, rather than shipping zero bytes.\n"
    write(wt, truncated, said)
    write(integ, truncated, said)
    code_ok, _ = run(tip, tmp, "v2_TIP_fixed", worktree=wt, integration=integ)
    check("TIP: GREEN when the probe writes its result instead of nothing", code_ok == 0,
          f"exit={code_ok}")


# ---------------------------------------------------------------- vector 3

def vector3(base, tip, tmp: Path) -> None:
    print("\n=== VECTOR 3 -- --allow-empty disarms the mistyped-id protection ===")
    results = {}
    for label, module in (("BASE", base), ("TIP", tip)):
        wt, integ = tmp / f"v3_{label}_wt", tmp / f"v3_{label}_int"
        flat_fixture(wt)
        write(integ, "README.md", "x\n")
        code, report = run(module, tmp, f"v3_{label}", task="REV-999", worktree=wt,
                           integration=integ, allow_empty=True)
        results[label] = (code, report)
        print(f"  {label}: --task REV-999 --allow-empty -> exit={code} "
              f"reason={report.get('exit_reason')!r}")

    base_code, _ = results["BASE"]
    tip_code, tip_report = results["TIP"]
    check("BASE: FALSE PASS -- a task id this tree never heard of exits 0",
          base_code == 0, f"exit={base_code}")
    check("TIP: refused as a usage error", tip_code == 2, f"exit={tip_code}")
    check("TIP: reason names the coupling that was broken",
          tip_report.get("exit_reason") == "allow_empty_on_an_unknown_task",
          str(tip_report.get("exit_reason")))

    # Shape guard, independent of the flag AND of enumeration.
    #
    # ATTEMPT 1 OF THIS CHECK ASSERTED `b == 0` AND FAILED ON TWO OF THE FOUR TYPOS. The
    # failing run is preserved beside this one at
    # ``c112_r2_false_pass_vectors.attempt1-typo-assertion.log``, because the anomaly is
    # a finding, not noise: at BASE, `rev109` and `REV_109` did not return 0 -- they
    # returned 1, having ENUMERATED ANOTHER TASK'S EVIDENCE. `rev109` resolves
    # `evidence/g11/rev109` onto `evidence/g11/REV-109` because the Windows filesystem is
    # case-insensitive, and `REV_109` reaches the same probes because ``task_stem``
    # strips non-alphanumerics, so both spellings collapse to the stem `rev109`. A
    # mistyped id was therefore not merely passed -- it was silently ATTRIBUTED TO THE
    # WRONG TASK, which is worse than the exit-0 the card described.
    #
    # The correct invariant is therefore not "base returns 0" but "**base never
    # REFUSES** a malformed id -- it processes it, one way or another -- while the tip
    # always does". That is what is asserted now.
    wt, integ = tmp / "v3_TIP_wt", tmp / "v3_TIP_int"
    for typo in ("REV-1O9", "rev109", "C112", "REV_109"):
        b = base.run(["--task", typo, "--worktree", str(wt), "--integration-dir", str(integ),
                      "--allow-empty"])
        t = tip.run(["--task", typo, "--worktree", str(wt), "--integration-dir", str(integ),
                     "--allow-empty"])
        check(f"typo {typo!r}: BASE never refuses it (exit != 2), TIP refuses with 2",
              b != 2 and t == 2, f"base={b} tip={t}")
    check("BASE mis-attributed 'rev109' to REV-109's evidence (case-insensitive FS) "
          "instead of refusing -- worse than the exit-0 the card described",
          base.run(["--task", "rev109", "--worktree", str(wt),
                    "--integration-dir", str(integ), "--allow-empty"]) == 1)

    # ...and the flag still does its ONE job at TIP.
    (wt / EVID / "g11" / "REV-107").mkdir(parents=True, exist_ok=True)
    code_ok, report_ok = run(tip, tmp, "v3_TIP_known", task="REV-107", worktree=wt,
                             integration=integ, allow_empty=True)
    check("TIP: --allow-empty still passes a REAL task that produced no evidence",
          code_ok == 0 and report_ok.get("task_dir_exists") is True, f"exit={code_ok}")
    code_no_flag = tip.run(["--task", "REV-107", "--worktree", str(wt),
                            "--integration-dir", str(integ)])
    check("TIP: and without the flag a real-but-empty task is still exit 3",
          code_no_flag == 3, f"exit={code_no_flag}")


def main(root: Path, sha: str) -> int:
    print("C-112 -- REV-109 R2: three false-PASS vectors, BASE vs TIP")
    print(f"worktree : {root}")
    print(f"base SHA : {sha}  (committed blob of {SCRIPT_REL})")
    with tempfile.TemporaryDirectory(prefix="c112_r2_") as td:
        tmp = Path(td)
        base_path = materialise_base(root, sha, tmp / "route_base.py")
        base = load(base_path, "route_base")
        tip = load(root / SCRIPT_REL, "route_tip")
        print(f"base blob: {len(base_path.read_bytes())} bytes")
        print(f"tip  file: {len((root / SCRIPT_REL).read_bytes())} bytes")

        print("\n=== SANITY -- known-positive and known-negative on BOTH modules ===")
        print("    (if these disagree the probe never reached the code and every "
              "result below is an artefact)")
        sanity(base, "BASE", tmp)
        sanity(tip, "TIP", tmp)

        vector1(base, tip, tmp)
        vector2(base, tip, tmp)
        vector3(base, tip, tmp)

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print(f"  FAILED: {f}")
        return 1
    print("RESULT: ALL CHECKS PASSED -- all three vectors were green at base and are "
          "closed at tip, and the good cases still pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve(), sys.argv[2]))
