# ORCH-718 — the anonymous heavy-lock strand, and the Lead's decision to clear it

**Incident, 2026-09-01.** `C:/t/heavylock` existed as an **empty directory with no `holder.json`**.
`HeavyLock.acquire` is `os.mkdir`, so every job — three cards and the Lead — got
`BOUNDED_RUN_HEAVY_LOCK_HELD` (exit 95) **forever**, with no holder to attribute and therefore no
way for any of them to clear it.

**Exit 95 is an infrastructure event, not a result.** No test number in this wave was affected by
it; the blocked jobs simply never started.

---

## Why no agent could clear it, and why that was correct

The standing clearing checklist requires **multiple byte-identical holder samples, a dead holder
PID, zero matching processes, and no peer ownership.** With **no `holder.json` at all**, the first
condition is **unsatisfiable by construction**.

**The C-111 agent hit this, declined to clear the lock, and escalated with evidence**
(`c111_heavy_lock_anonymous_strand.md`). **That was the right call and it is why this is a clean
recovery rather than a guess.** An agent that had reasoned *"there is no holder, so nothing owns it,
so I may delete it"* would have been right by luck and wrong by method — the same reasoning deletes
a lock during another job's acquisition window.

---

## The evidence the Lead assembled, which an agent inside a blocked job could not

1. **Three independent samples** of the directory: empty each time. Not a transient.
2. **Zero sprint-owned Python processes** anywhere on the machine — only the two
   `ms-python.isort` IDE processes, **matched on command line, never on PID**, and never cleanup
   targets. Nothing could have been mid-acquisition, because nothing was running.
3. **A directly observed holder sample.** At 14:31 the Lead's own gold-readers job took exit 95 and
   printed the holder file verbatim:
   `{"holder": "C-111", "label": "deadline-probe", "pid": 473280, …}`.
   **That is the byte-exact holder sample the checklist asks for** — recovered from the failed
   acquire's own diagnostic rather than from the lock directory.
4. **PID 473280 confirmed DEAD.**
5. **No peer ownership.** `project14-t2pw-93` confirmed read-only earlier in the wave: no branch, no
   worktree, no lock, no job.

**So the lock was attributable after all** — not from the lock directory, which had been stripped,
but from a *failed acquire's* diagnostic output. The checklist's conditions were all met; only the
usual *source* of the holder sample was gone.

**Cleared with `rmdir`, never `rm -rf`** — `rmdir` refuses on a non-empty directory, so a race that
had written a holder between the last sample and the call would have **failed the clear rather than
silently overwriting it.** Use the primitive that can refuse.

---

## The defect, registered as R-C111-4

**`HeavyLock.release` is not atomic.** It **unlinks `holder.json`, then `rmdir`s the directory.** A
kill landing between those two statements leaves the directory present and anonymous — a lock that
**no one is permitted to clear under the project's own rules.**

**The diagnosis is provable from the artifact rather than inferred:** if the wrapper's `finally` had
simply never run, **`holder.json` would still be there.** It was gone *and* the directory remained,
which is only reachable by a kill inside `release`, between the two statements.

`bounded_run.py`'s own docstring anticipates this shape and declines to handle it.

**Not repaired in this wave, deliberately.** `bounded_run.py` is the instrument every job in flight
is being measured through, and its build hash is recorded in every G11 report
(`sha256:83d1395…`). **Changing the instrument mid-wave would break comparability across the
reports already written.** Chartered for the next wave; the fix is to write the holder file first
and remove the directory in a way that cannot leave the anonymous state — or to treat an empty lock
directory as a documented, clearable verdict rather than an undefined one.

**Related standing note:** the previously recorded failure mode is a shell clock shorter than the
wrapper's `--timeout`, which kills the wrapper so its `finally` never runs and leaves a holder
naming a dead PID. **This incident is the sibling case one statement further along** — the `finally`
*did* run, and was killed partway through it.
