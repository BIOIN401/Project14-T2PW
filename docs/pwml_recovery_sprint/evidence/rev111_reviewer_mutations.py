"""REV-111 B18 — the reviewer's OWN mutations of C-111's preservations.

*"Remove one of the nine preservations and confirm a test goes red."*

Five mutations, each removing or corrupting exactly one thing the card claims to
preserve. A guard nobody has seen go red is not a guard. Each mutation must turn
the C-111 acceptance file RED, and the saved bytes must then replay EXACTLY --
**D-084: restores replay SAVED BYTES.** ``git checkout --`` reverts more and a
text-mode write reverts less; neither is used, and the restore is proved by
sha256 AND crlf count, never by ``git status --porcelain``.

**F-160.** A same-length edit inside the same second leaves a stale ``.pyc``
valid and the OLD BYTECODE RUNS -- which would show a mutation as green and be
read as a missing guard. Every arm purges ``__pycache__`` under ``src/t2pw`` and
``tests`` ONLY, before and after. An unscoped purge would delete the 56 TRACKED
``.pyc`` files elsewhere in this tree; the count is asserted at both ends.

The reviewer cannot fix this diff and does not: every byte is put back.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:/t/rev111")
PY = r"c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
TESTS = "tests/test_c111_timeout_observability.py"
BASETEMP = "C:/t/btrev111/mutations"
FAILURES: list = []


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crlf_count(data: bytes) -> int:
    return data.count(b"\r\n")


def newline_of(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def apply_mutation(path: Path, old: str, new: str) -> bytes:
    saved = path.read_bytes()
    text = saved.decode("utf-8")
    newline = newline_of(text)
    old_nl = old.replace("\n", newline)
    new_nl = new.replace("\n", newline)
    count = text.count(old_nl)
    if count != 1:
        raise ValueError(f"the substitution matched {count} times, not 1")
    path.write_bytes(text.replace(old_nl, new_nl, 1).encode("utf-8"))
    return saved


def restore_saved_bytes(path: Path, saved: bytes) -> None:
    path.write_bytes(saved)
    after = path.read_bytes()
    if sha256_of(after) != sha256_of(saved):
        raise AssertionError(f"restore not byte-exact: {sha256_of(saved)} -> {sha256_of(after)}")
    if crlf_count(after) != crlf_count(saved):
        raise AssertionError(
            f"restore changed line endings: {crlf_count(saved)} -> {crlf_count(after)}")


def purge_bytecode() -> int:
    """SCOPED to src/t2pw and tests. Never the whole tree -- 56 .pyc are tracked."""
    removed = 0
    for base in (ROOT / "src" / "t2pw", ROOT / "tests"):
        for cache in base.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
            removed += 1
    return removed


def tracked_pyc_count() -> int:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.pyc"],
                         capture_output=True, text=True)
    return len([line for line in out.stdout.splitlines() if line.strip()])


def run_suite() -> tuple[int, list[str], str]:
    purge_bytecode()
    proc = subprocess.run(
        [PY, "-m", "pytest", TESTS, "-q", "--no-header", "-rf", "--basetemp=" + BASETEMP],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r"FAILED [^:]+::(\w+)", out)))
    tail = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
    return proc.returncode, failed, (tail[-1] if tail else "(no summary line)")


#: (name, file, old, new, what preservation this removes)
MUTATIONS = [
    (
        "M1_drop_item_6_payload_before_cleanup",
        "src/t2pw/batch/leg_trace.py",
        '        "payload_before_cleanup": terminal.get("payload_before_cleanup", {}),',
        '        "payload_before_cleanup": {},',
        "item 6 -- whether a payload existed before cleanup",
    ),
    (
        "M2_drop_item_2_retry_reasons",
        "src/t2pw/batch/leg_trace.py",
        '        "retry_reasons": retry_reasons,',
        '        "retry_reasons": [],',
        "item 2 -- the retry reason per retry. THE reason the card exists",
    ),
    (
        "M3_stop_fsyncing_each_event",
        "src/t2pw/batch/leg_trace.py",
        "                    handle.flush()\n                    os.fsync(handle.fileno())",
        "                    pass",
        "durability: the write is no longer forced before the next event",
    ),
    (
        "M4_collapse_outer_kill_into_the_in_process_label",
        "src/t2pw/batch/leg_trace.py",
        "    if parent_killed and not child_reported:\n        return SOURCE_OUTER_PARENT_KILL",
        "    if parent_killed and not child_reported:\n        return SOURCE_IN_PROCESS_DEADLINE",
        "item 5 -- the two mechanisms F-148 says the run reports as one",
    ),
    (
        "M5_stop_publishing_attempts_to_disk",
        "src/t2pw/llm/client.py",
        "        _publish_attempt(self, row)",
        "        pass",
        "items 1/8 at the LLM seam -- attempts stay in memory and die with it",
    ),
]

#: A mutation that removes NOTHING. The suite must stay GREEN, or a red above
#: proves only that the file was touched.
NEUTRAL = (
    "N0_neutral_comment_only",
    "src/t2pw/batch/leg_trace.py",
    "def deactivate() -> None:",
    "def deactivate() -> None:  # reviewer neutral control\n    ",
)


def main() -> int:
    print("REV-111 -- the reviewer's own mutations of the nine preservations")
    print(f"tree   : {ROOT}")
    print(f"target : {TESTS}\n")

    tracked_before = tracked_pyc_count()
    print(f"tracked .pyc before : {tracked_before}")

    code, failed, summary = run_suite()
    print(f"\nBASELINE (unmutated) : exit={code}  {summary}")
    if code != 0:
        print("BASELINE IS NOT GREEN. Nothing below would mean anything.")
        return 1

    for name, rel, old, new, removes in MUTATIONS:
        path = ROOT / rel
        print(f"\n--- {name}")
        print(f"    removes: {removes}")
        try:
            saved = apply_mutation(path, old, new)
        except ValueError as exc:
            FAILURES.append(f"{name}: could not apply -- {exc}")
            print(f"    APPLY FAILED: {exc}")
            continue
        try:
            code, failed, summary = run_suite()
            print(f"    exit={code}  {summary}")
            print(f"    red tests: {failed or '(none)'}")
            if code == 0:
                FAILURES.append(f"{name}: the suite stayed GREEN -- this is NOT guarded")
                print("    *** SURVIVED -- the preservation is not guarded ***")
            else:
                print("    caught.")
        finally:
            restore_saved_bytes(path, saved)
            print(f"    restored byte-exact: sha256={sha256_of(path.read_bytes())[:16]}")

    name, rel, old, new = NEUTRAL
    path = ROOT / rel
    print(f"\n--- {name} (NEGATIVE CONTROL -- must stay GREEN)")
    saved = apply_mutation(path, old, new)
    try:
        code, failed, summary = run_suite()
        print(f"    exit={code}  {summary}")
        if code != 0:
            FAILURES.append(f"{name}: the neutral control went RED -- the suite is "
                            f"reacting to the edit, not the semantics ({failed})")
            print("    *** NEUTRAL CONTROL WENT RED ***")
        else:
            print("    stayed green, as required.")
    finally:
        restore_saved_bytes(path, saved)

    code, failed, summary = run_suite()
    print(f"\nFINAL (restored) : exit={code}  {summary}")
    if code != 0:
        FAILURES.append(f"the tree did not come back green after restore: {failed}")

    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                            "src/t2pw", "tests"], capture_output=True, text=True).stdout
    print(f"git status --porcelain src/t2pw tests : {dirty.strip() or '(clean)'}")
    if dirty.strip():
        FAILURES.append(f"the tree is dirty after restore: {dirty.strip()}")

    tracked_after = tracked_pyc_count()
    print(f"tracked .pyc after  : {tracked_after}")
    if tracked_after != tracked_before:
        FAILURES.append(f"tracked .pyc changed: {tracked_before} -> {tracked_after}")

    print("\n================ VERDICT ================")
    if FAILURES:
        for item in FAILURES:
            print(f"  FAILED: {item}")
        return 1
    print("  Every mutation was caught; the neutral control stayed green;")
    print("  every restore replayed the saved bytes exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
