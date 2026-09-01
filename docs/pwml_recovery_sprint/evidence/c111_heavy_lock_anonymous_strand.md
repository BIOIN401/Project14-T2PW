# C-111 — an ANONYMOUS stranded heavy lock, reported and NOT cleared

**Reported to the Lead rather than worked around.** This blocked C-111's SMOKE and gold-reader
gates. It is recorded here because the lock protocol's own recovery procedure does not cover this
state, and because the next agent to hit it will otherwise re-derive the whole thing.

## The state

```
C:/t/heavylock/          exists, EMPTY, mtime 2026-09-01 14:42
C:/t/heavylock/holder.json   ABSENT
```

`bounded_run.HeavyLock.acquire` is `os.mkdir(path)`, so an existing directory is `FileExistsError`
→ `acquire_error = "held"` → **exit 95 for every caller, forever**, while
`read_holder_text()` returns
`<holder file unreadable: FileNotFoundError: ... C:\heavylock\holder.json>`.

## How this state is produced

`HeavyLock.release` does `os.unlink(self.holder_file)` and then `os.rmdir(self.path)`. **A process
killed between those two statements leaves exactly this**: a lock directory with no holder file.

`bounded_run.py`'s own class docstring anticipates the shape and declines to handle it:

> the ``holder_file_vanished`` branch in :meth:`release` refuses a lock whose holder file is gone,
> and such a lock is **exactly as anonymous as the one the shell protocol left**. … Clearing another
> holder's lock is the orchestrator's decision alone.

## Why C-111 did not clear it

The standing rule is *"never clear, break or steal a lock you do not hold; never delete a lock you
cannot attribute"*, and the clearing checklist — **multiple byte-identical holder samples, dead
holder PID, zero matching processes, no peer ownership** — cannot be satisfied when there is **no
holder file to sample and no PID to check**. Its first condition is unsatisfiable by construction
here, so the answer is not "the checklist passes", it is "the checklist does not apply".

## Evidence that it is not C-111's lock

| Check | Result |
|---|---|
| Lock directory sampled 3× | empty every time, no `holder.json` |
| Running `python.exe` processes | **2**, both `ms-python.isort` IDE language servers — matched **on command line, never PID**, and **not cleanup targets** |
| Any `bounded_run` / `pytest` / batch process | **none** |
| `C:/t/btc111/combined` (the basetemp C-111's blocked job would have used) | **empty**, mtime 14:30 — pytest never started there |
| `evidence/g11/pin/C-111/05-*.pin.json` | **absent** — the pinned launcher never ran |
| `evidence/g11/C-111/05-*.json` | **absent**; only its `.staging` reservation exists |
| The last completed wrapper attempt before C-111's shell died | already reported `holder file unreadable: FileNotFoundError` — **the lock was already anonymous before that shell was killed** |

Every C-111 attempt exited **95 `BOUNDED_RUN_HEAVY_LOCK_HELD`, the child never started**. C-111
therefore never held this lock and cannot release it.

## What C-111 did instead

Waited, and reported the block with this evidence. **Nothing was deleted, moved, renamed or
overridden, and `--heavy-lock-path` was NOT used to route around the mutex** — a scratch lock would
have restored throughput by abandoning the mutual exclusion the gate exists for.

## Registered

**R-C111-4 — `HeavyLock.release` is not atomic, and its non-atomic window produces a lock nobody may
clear.** Two syscalls, and a kill between them converts an attributable lock into an anonymous one
that the protocol forbids every agent — including its own owner — from removing. Out of scope for
C-111 (it is not this card's seam) and registered for the Lead.
