"""REV-111 addendum — what does the fsync actually buy, on the threat model?

My own mutation **M3** removed ``handle.flush()`` + ``os.fsync(handle.fileno())``
from :meth:`leg_trace.LegTrace.event` and **the C-111 suite stayed GREEN**. F-160's
lesson is that a survived mutation reads as *"this guard has no test, delete it"* —
so before reporting it as a gap I measure what the guard is actually for, rather
than reasoning about it.

**The hypothesis, written before the run.** The threat C-111 names is a FORCE KILL
OF THE CHILD PROCESS (``taskkill /F /T``). A process kill does not discard the OS
page cache, and ``LegTrace.event`` re-opens the file per event inside a ``with``
block, so the close alone already hands every completed event to the OS. If that is
right, the events survive a real force kill **with the fsync removed**, and the
fsync is buying durability against a MACHINE crash — a strictly wider threat than
the one the module documents.

**Predictions, fixed in advance:**

* If the fsync is what delivers force-kill durability -> the mutated arm loses
  events that the unmutated arm keeps. The guard is load-bearing and untested.
* If the per-event open/close is what delivers it -> both arms preserve the same
  events. The fsync is insurance against a wider threat, the suite cannot see it
  by construction, and M3's survival is not a missing guard.

Either way the finding is REPORTED, not repaired: the reviewer does not fix this
diff. Every byte is restored (D-084: replay the SAVED BYTES) and proved by sha256
and CRLF count.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:/t/rev111")
TARGET = ROOT / "src" / "t2pw" / "batch" / "leg_trace.py"
FAILURES: list = []

OLD = "                    handle.flush()\n                    os.fsync(handle.fileno())"
NEW = "                    pass"

CHILD = r'''
import sys, time
from pathlib import Path
LEG = Path(sys.argv[2])
LEG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, sys.argv[1])
from t2pw.batch import leg_trace
t = leg_trace.activate(LEG / leg_trace.LEG_TRACE_NAME)
for i in range(25):
    t.model_attempt(stage="extraction", attempt=i + 1, status="error",
                    model="m", reason="attempt %d" % (i + 1))
while True:
    time.sleep(0.05)
'''


def sha256_of(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def crlf_count(b: bytes) -> int:
    return b.count(b"\r\n")


def newline_of(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def apply_mutation(path: Path, old: str, new: str) -> bytes:
    saved = path.read_bytes()
    text = saved.decode("utf-8")
    nl = newline_of(text)
    old_nl, new_nl = old.replace("\n", nl), new.replace("\n", nl)
    if text.count(old_nl) != 1:
        raise ValueError(f"matched {text.count(old_nl)} times, not 1")
    path.write_bytes(text.replace(old_nl, new_nl, 1).encode("utf-8"))
    return saved


def restore_saved_bytes(path: Path, saved: bytes) -> None:
    path.write_bytes(saved)
    after = path.read_bytes()
    if sha256_of(after) != sha256_of(saved):
        raise AssertionError("restore was not byte-exact")
    if crlf_count(after) != crlf_count(saved):
        raise AssertionError("restore changed line endings")


def purge_bytecode() -> None:
    """SCOPED. 56 .pyc are tracked elsewhere in this tree and must not be touched."""
    for base in (ROOT / "src" / "t2pw", ROOT / "tests"):
        for cache in base.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)


def kill_arm(tmp: Path, name: str) -> dict:
    """Force-kill a real child through the real parent seam; read the trace back."""
    purge_bytecode()
    script = tmp / f"child_{name}.py"
    script.write_text(CHILD, encoding="utf-8")
    leg = tmp / name
    reader = (
        "import sys, json\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from t2pw.batch import runner, leg_trace\n"
        "r = runner.launch_child([sys.executable, sys.argv[2], sys.argv[1], sys.argv[3]], 3.0)\n"
        "s = leg_trace.summarize(sys.argv[3])\n"
        "print(json.dumps({'timed_out': r.timed_out, 'calls': s['total_model_calls'],\n"
        "                  'events': s['_trace_events'], 'present': s['_trace_present']}))\n"
    )
    rp = tmp / f"reader_{name}.py"
    rp.write_text(reader, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(rp), str(ROOT / "src"), str(script), str(leg)],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        FAILURES.append(f"{name}: reader failed rc={proc.returncode} {proc.stderr[-300:]}")
        return {}
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    print("REV-111 addendum -- what the fsync buys, measured")
    print(f"target : {TARGET}\n")
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="rev111fsync_"))
    print(f"tmp    : {tmp}\n")
    try:
        print("ARM 1 -- UNMUTATED (flush + fsync present)")
        unmutated = kill_arm(tmp, "with_fsync")
        print(f"  {unmutated}")

        saved = apply_mutation(TARGET, OLD, NEW)
        try:
            print("\nARM 2 -- MUTATED (flush + fsync REMOVED; the per-event open/close remains)")
            mutated = kill_arm(tmp, "no_fsync")
            print(f"  {mutated}")
        finally:
            restore_saved_bytes(TARGET, saved)
            purge_bytecode()
            print(f"\n  restored byte-exact: sha256={sha256_of(TARGET.read_bytes())[:16]}")

        print("\n---- controls ----")
        for label, arm in (("with_fsync", unmutated), ("no_fsync", mutated)):
            ok = bool(arm) and arm.get("timed_out") is True
            print(f"  {label}_was_really_force_killed : {'OK' if ok else 'FAILED'}")
            if not ok:
                FAILURES.append(f"{label} was not actually killed")

        print("\n---- result ----")
        u, m = unmutated.get("calls"), mutated.get("calls")
        print(f"  attempts preserved WITH fsync    : {u} / 25")
        print(f"  attempts preserved WITHOUT fsync : {m} / 25")
        if u == m == 25:
            print("\n  VERDICT: the per-event open/close ALONE survives a force kill.")
            print("  The fsync is insurance against a MACHINE crash, which is a wider")
            print("  threat than the one C-111 documents -- so M3's survival is NOT a")
            print("  missing guard for the threatened failure mode. It is an unguarded")
            print("  belt-and-braces, and it is also the entire source of the")
            print("  +0.55 ms/attempt timing side effect measured in arm E.")
        elif m is not None and u is not None and m < u:
            print("\n  VERDICT: the fsync IS load-bearing for the force kill and NO test")
            print("  holds it. That is a real gap in the card's own central claim.")
        else:
            FAILURES.append(f"uninterpretable: with={u} without={m}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src/t2pw"],
                               capture_output=True, text=True).stdout.strip()
        print(f"\ngit status --porcelain src/t2pw : {dirty or '(clean)'}")
        if dirty:
            FAILURES.append(f"tree dirty after restore: {dirty}")

    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nOK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
