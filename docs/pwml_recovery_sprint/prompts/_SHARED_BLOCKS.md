# Shared prompt blocks

Referenced by ID from every task prompt. Written once so ~90 prompts can be generated
without drift. When a prompt says `[S1]`, paste this block's text.

---

## S1 — Environment contract

```
ENVIRONMENT
  Repo         : BIOIN401/Project14-T2PW
  Integration  : sprint/pwml-recovery
  Python       : .venv/Scripts/python.exe  (3.13.6, Windows)
  Authority    : docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md
  Your row     : docs/pwml_recovery_sprint/MASTER_PLAN.md § 9 branch register

TESTS
  ALWAYS pass --basetemp=<unique dir>. Without it 83 tests error with
  PermissionError and you will report a false regression.
  NEVER run the full suite unchunked (~16 GB).
  Commands: docs/pwml_recovery_sprint/TEST_MATRIX.md

DO NOT
  - modify any file outside your OWNED list. A diff touching an unowned file is
    an automatic reject, even if the change is correct.
  - run a full pinned benchmark (7 h). Milestone runs are the orchestrator's.
  - commit a cache modification. data/enrichment_cache.json is 39 MB and
    tracked; .git is already 158 MB.
  - read other branches' prompts.
```

## S2 — Test commands

See `TEST_MATRIX.md` § Commands. Smoke = **460** tests, ~40 s. Chunk D = **177** tests: the
deterministic core is 150 tests in ~1 s, and the **complete 177-test gate costs 9–13 min**,
dominated by the 27 per-node AppTest processes. The older ~222 s figure is stale and must
not be used for the complete gate.

## S3 — Non-negotiable product invariants

```
You MUST NOT, under any circumstance:
  1. Weaken a biological gate to increase PWML production.
  2. Delete an incomplete-but-correct pathway instead of exporting it as
     review_required.
  3. Let an exporter add, remove, resolve or reinterpret biological content
     after the canonical graph is frozen.
  4. Accept an identifier because its FORMAT is valid.
  5. Invent biology to satisfy a completeness target.
  6. Make a pinned baseline test pass by reverting behaviour. If a pinned
     baseline must move, move it deliberately, document the exact delta, and
     say so in your report.
  7. Collapse `not_evaluated` into `false`.
  8. Treat a lookup or network failure as biological rejection.
If your change appears to require any of these, STOP and report instead.
```

## S4 — Stop conditions

```
STOP and report WITHOUT committing if:
  - The change requires editing a file you do not own.
  - A test outside your focused set fails and you did not expect it.
  - Your dependency's merged behaviour differs from what this prompt assumes.
  - You would have to violate S3.
  - Either DECLARED BUDGET below is predicted or discovered to be exceeded.
    Propose a split instead. An over-budget commit may only be created after
    RENEWED EXPLICIT AUTHORITY granting a REVISED budget; no implementer,
    reviewer or test-runner may self-authorize one, and technical success or
    later approval does not cure one.

DECLARED BUDGETS (D-019). Every card declares both BEFORE dispatch; there is no
universal line threshold, the card's own numbers bind:
  - HAND-AUTHORED: exact allowed manifest, and max additions plus deletions.
  - MACHINE-GENERATED EVIDENCE, budgeted separately: max artifact count AND a
    size limit (bytes or changed lines); an explicit 0 when unauthorized.

SPLITTING. Acceptance-criterion atomicity is weighed when the card is scoped and
budgeted. Split ONLY where each half is independently implementable AND
independently validatable. A budget never justifies merging or leaving behind an
unvalidated semantic half -- if no such boundary exists, stop and say so.
```

## S5 — Evidence standard

```
Claims in your report must be evidenced:
  "test passes"     -> paste the pytest summary line.
  "behaviour X"     -> paste the command and its output.
  "artifact says Y" -> cite file path + line or key.
Never assert a runtime behaviour from a static code path alone.
Never claim a benchmark number whose source run is not committed.
```

## S6 — Implementer report format

```
## BRANCH
## BASE SHA
## FILES CHANGED   (path :: function, one per line)
## WHAT CHANGED    (<= 10 bullets)
## TESTS ADDED     (name :: exact failure it would catch)
## TEST OUTPUT     (pasted summaries)
## INVARIANTS      (one line per S3 item touched, with evidence)
## OUT OF SCOPE    (what you noticed and did NOT do)
## RISK            (what a reviewer should look hardest at)
```

## S8 — Test-process lifecycle policy (HARD MERGE RULE)

```
This is a merge gate (G11), not a suggestion. A run that violates it is an
INFRASTRUCTURE FAILURE, not a test result, and must never be reported as passed.

WHY: orphaned pytest/Streamlit/LLM descendants survive their parent and consume
the developer's machine memory for hours. A full suite alone approaches 16 GB.

1. BOUNDED FOREGROUND ONLY
   Every test, benchmark, pipeline leg and LLM-backed command runs through the
   bounded foreground-process wrapper (path recorded by INIT-001).
   NEVER: detached processes, nohup, untracked background jobs, or Start-Process
   without bounded waiting and guaranteed cleanup.

2. THE WRAPPER MUST
   - record root PID, command, start time, working directory, timeout, ownership
   - place the command in an isolated process group (POSIX) or Job Object (Windows)
   - enforce an outer wall-clock timeout
   - run cleanup in finally/trap on EVERY exit path: success, failure, timeout,
     cancellation, shell interruption, agent failure
   - terminate all remaining descendants OWNED BY THAT JOB
   - graceful termination first, forced termination after a short grace period
   - VERIFY no tracked process from that job is still alive
   - preserve and return the REAL test exit code, unless cleanup verification
     itself failed

3. PLATFORM
   Windows : prefer a Job Object set to terminate members when closed. If no safe
             implementation is available, track the root PID and use
             `taskkill /PID <owned-pid> /T /F` inside guaranteed cleanup.
   POSIX   : new process group; TERM the group, then KILL after the grace period.

4. NEVER GLOBAL CLEANUP
   FORBIDDEN: `taskkill /IM python.exe`, `pkill python`, or killing every Java,
   Node, pytest or Python process.
   Cleanup targets ONLY PIDs / process groups created and recorded by the current
   test job. Pre-existing processes are REPORTED, never silently killed.

5. COMPLETION IS NOT "PYTEST PRINTED A SUMMARY"
   A job is complete only when: root process exited AND all owned descendants
   exited AND cleanup verification passed AND exit status + cleanup result were
   recorded.

6. SURVIVORS ARE AN INFRASTRUCTURE FAILURE
   If any owned process survives cleanup: classify the run as an infrastructure
   failure, STOP further dispatch, and report the surviving PID, command line,
   start time and memory usage. Do NOT report the test as passed.

7. ONE HEAVY JOB AT A TIME
   At most one full suite, benchmark, or memory-heavy pipeline leg concurrently.
   NEVER `pytest -n auto`. Never concurrent full benchmarks. Focused tests may run
   concurrently only when their resource limits and ownership stay explicit.

8. BASETEMP
   Keep the unique `--basetemp` path. Remove temp dirs after completion when safe.
   Do not confuse temporary FILES with active MEMORY consumption -- deleting a
   basetemp directory does not reclaim a leaked process's RAM.

9. CLEANUP REPORT -- required on EVERY test record
   root PID / process group | timeout | exit reason | exit code |
   descendants observed | descendants terminated | final surviving count |
   cleanup success/failure

10. THE REPORT IS COMMITTED, NOT PASTED
   A pasted table is not evidence. Every job's --json report goes to a UNIQUE,
   version-controlled path and is committed with your branch:

     <py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py next \
          --task <YOUR-ID> --label <label>          # allocate; never reuse one
     <py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label <label> \
          --timeout <s> --json <the allocated path> -- <command...>
     <py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py check \
          --task <YOUR-ID>                          # before you commit

   A job with no committed report is UNCERTIFIABLE under G11 -- it is not a
   passed test. The artifact must exist and validate ON ITS OWN: an acceptable
   exit code proves nothing, because an unwritable --json still returns the
   child's real exit code. `cleanup_success: true` is required; a final
   surviving count of 0 alone is NOT sufficient.

   Reports are credential-free and small. `command` is copied verbatim, so a
   token, key or connection string on the command line lands in the artifact:
   run `check` before committing, and if it finds one, do NOT commit and do NOT
   edit the report -- rotate the credential and re-run with the secret in the
   environment. Never commit captured stdout, logs, caches or basetemp trees.

   PROSPECTIVE ONLY. Never reconstruct, synthesize or backfill a report for a
   job that already ran. Details: docs/pwml_recovery_sprint/evidence/g11/README.md
```

## S7 — Standing traps

```
TRAP-1  PMC12452463's gold export_rationale records the route as chemically
        BROKEN (EntA absent; nothing converts 2,3-dihydro-2,3-dihydroxybenzoate
        onward). After C-010 it passes quarantine, the final Stage-3 gate and
        the required-field gate. Its CORRECT outcome is review_required with
        strict_acceptance_eligible=false. It must NEVER count as strict success.
        Any agent optimizing toward "PMC12452463 passes strict" is chasing the
        wrong target.
TRAP-2  tests/test_strict_quarantine_real_artifact_replay.py pins
        FULL_STACK_BASELINE at :384-393 and asserts it by exact equality at
        :432. C-010 changes it BY DESIGN. An agent that makes this test pass by
        reverting behaviour is rejected.
        NOTE: H-001 freezes this gate's cohort to an explicit manifest and
        re-records its expectations BEFORE C-010 is dispatched, because the pin
        was already stale on ORIGIN_SHA for unrelated reasons (BASELINE.md
        section 5). You inherit a PASSING gate. Exactly TWO of C-010's six
        allowlist legs fall inside its cohort; the other four are under
        runs_verify/, which this gate does not read.
TRAP-3  placeholder_backed_proteins (21 in the pinned run) is a standing POLICY
        DISAGREEMENT between gold set and pipeline, not a defect. No agent may
        "fix" it. Escalate to the product owner.
TRAP-4  Do not rebuild what already exists. See MASTER_PLAN.md § 2:
        the UniProt accession fetch (enrich_entities.py:507 + EnrichmentCache),
        accession-keyed PathBank lookup (map_ids.py:1822), the provenance
        carrier (pipeline._carry_rag_provenance:1880), alias traversal
        (reference_repair.REFERENCE_FIELDS:149), semantic not_evaluated
        (SemanticReport.evaluated), the SBML canonical binding, the artifact
        replay harness, the RAG gap detector (retrieve.detect_gaps:656 +
        class Gap:220), the RAG claim schema (admission.RagReactionCandidate),
        or the RAG admission gate (admission.py, 3100 lines).
TRAP-5  data/enrichment_cache.json is 39 MB and TRACKED. No branch may commit a
        cache modification.
```
