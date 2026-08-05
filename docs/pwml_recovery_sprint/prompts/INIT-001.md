# INIT-001 — Sprint initialization, process-lifecycle helper, baseline capture

**Blocks every other prompt.** Nothing is dispatched until this reports ACCEPTED.

---

```
[S1] [S5] [S8]

ROLE
  Sprint initialization. You establish the process-lifecycle helper, capture the
  baseline, and commit the missing run evidence. You write no pipeline source
  code and you modify no file under src/.

STEP 0 -- PROCESS-LIFECYCLE HELPER. THIS IS THE FIRST EXECUTABLE STEP.
Do NOT run the expensive baseline until this step passes. The baseline is the
single largest memory consumer in the sprint; running it without a verified
wrapper is exactly the failure this step exists to prevent.

  0a. INSPECT FOR AN EXISTING SAFE HELPER.
      Search the repository for process-lifecycle machinery before writing any.
      Known and already audited:

        src/t2pw/batch/runner.py
          launch_child   :1140-1180   CREATE_NEW_PROCESS_GROUP (nt) /
                                      start_new_session (posix); bounded
                                      communicate(timeout=); _kill_tree on both
                                      TimeoutExpired and KeyboardInterrupt;
                                      _DRAIN_TIMEOUT for pipe close
          _kill_tree     :1107-1137   taskkill /F /T /PID <OWNED pid>, then
                                      os.killpg, then proc.kill()
          child_env      : 265-276

      This is the CORRECT OWNERSHIP MODEL -- it targets only the PID it created
      and is not a global killer. Verify that for yourself; do not take it on
      faith.

      What it LACKS against [S8]: no graceful-then-forced escalation (it goes
      straight to /F), no post-kill survivor verification, no structured cleanup
      report, and it is used only for batch legs -- NOT for pytest runs.

      YOU MUST NOT MODIFY runner.py. That file is owned by C-032. Build an
      ORCHESTRATION-ONLY wrapper that reuses this discipline.

  0b. REUSE OR CREATE.
      If an existing helper satisfies [S8] in full, record its path and skip to
      0c. Otherwise create an orchestration-only helper. Suggested location:
        docs/pwml_recovery_sprint/evidence/bounded_run.py
      (evidence/, not src/ -- it is sprint tooling, not pipeline code).

      It must, per [S8]:
        - record root PID, command, start time, cwd, timeout, ownership
        - isolate: Job Object (Windows, preferred, set to terminate members when
          closed) or new process group (POSIX)
        - enforce an outer wall-clock timeout
        - clean up in finally/trap on EVERY exit path: success, nonzero, timeout,
          cancellation, shell interruption, agent failure
        - graceful termination first, forced after a short grace period
        - VERIFY no tracked process from the job survives
        - return the REAL exit code unless cleanup verification itself failed
        - emit the cleanup report: root PID/process group, timeout, exit reason,
          exit code, descendants observed, descendants terminated, final
          surviving count, cleanup success/failure

      FORBIDDEN, absolutely: taskkill /IM python.exe, pkill python, or any kill
      by image name. Cleanup targets only PIDs/groups this job created.
      Pre-existing processes are REPORTED, never killed.

  0c. VALIDATE THE HELPER WITH SYNTHETIC CASES.
      Six cases. Every one must pass, and every one must end with ZERO surviving
      owned processes. Use trivial python -c children; no pytest, no pipeline.

        1. normal completion            -> exit 0, real code returned, 0 survivors
        2. nonzero exit                 -> the REAL nonzero code returned
        3. hanging child                -> outer timeout fires, child terminated
        4. child that spawns a child    -> BOTH terminated; this is the case a
                                           naive proc.kill() fails
        5. forced timeout               -> graceful attempted, then forced;
                                           report shows the escalation
        6. interruption / cancellation  -> cleanup still runs via finally/trap

      For EACH case record: the command, the exit reason, the exit code, the
      descendants observed and terminated, and PROOF that no synthetic child
      survived -- an explicit post-run liveness check of the recorded PIDs, not
      an assumption.

      IF ANY CASE FAILS: stop. Do not proceed to Step 4. Report.

STEP 1 -- RECORD THE INTEGRATION SHA
  git rev-parse HEAD
  Record it. Every downstream prompt cites the CURRENT integration SHA read at
  dispatch time, never a SHA hard-coded in a document.

STEP 2 -- VERIFY AND RESOLVE THE WORKTREE
  git status --porcelain | grep -v '^??'
  EXPECT exactly these 7 tracked modifications, all pre-existing scratch:
    data/enrichment_cache.json    (39 MB -- see TRAP-5)
    data/id_mapping_cache.json    (4.4 MB)
    out/enrichment_dump.json      outputs/pathway.pwml
    tmp/draft_graph.json          tmp/qa_report.json
    tmp/reaction_summary.txt
  Decide per file: commit, stash, or restore. Report what you did with each.
  Do NOT branch or dispatch over an unexplained dirty tree.
  If ANY OTHER tracked file is modified, STOP -- something changed the repo since
  the control plane was committed.
  Open question O-3 in DECISIONS.md covers the 39 MB cache; if you cannot resolve
  it, leave it untouched and say so.

STEP 3 -- COMMIT THE MISSING RUN EVIDENCE
  Required by C-010's allowlist, C-014 and C-042.
  Target: runs_verify/2026-08-04_1754/

  SIZE CHECK (measured 2026-08-05; verify before committing):
    total             44 MB
    cache_snapshot/   38 MB   <- EXCLUDE
    papers/          5.7 MB
    other           ~140 KB
    commit size     ~5.9 MB   against a 158 MB .git

  Precedent is mixed: 16 cache_snapshot files are tracked under older runs/, none
  under recent runs_verify/. Follow the RECENT convention -- exclude.
  Consider adding runs_verify/*/cache_snapshot/ to .gitignore (open question O-2;
  if you do not resolve it, say so rather than deciding silently).

  Verify after committing:
    git ls-files runs_verify/2026-08-04_1754 | wc -l                     > 0
    git ls-files 'runs_verify/2026-08-04_1754/cache_snapshot/*' | wc -l  = 0

  WHY: without it, agents in isolated worktrees cannot see the
  identical_empty_response evidence (two boundaries, identical response_hash
  sha256:44136fa355b3678a) or the degree_zero_export:2 shape, and
  test_strict_quarantine_real_artifact_replay SKIPS SILENTLY when runs/ inputs
  are absent. Two of C-010's six allowlist legs live in this run.

STEP 4 -- PRESERVE THE EXACT BASELINE
  EVERY command in this step runs through the Step-0 wrapper. ONE heavy job at a
  time. Never pytest -n auto.
  Fill docs/pwml_recovery_sprint/BASELINE.md and commit.

  4a. Full test suite, ~10 chunks, unique --basetemp per chunk. Record per-chunk
      passed/failed/skipped and the total. Note explicitly whether chunk E
      (test_strict_quarantine_real_artifact_replay) RAN or SKIPPED.
  4b. Smoke suite. EXPECT 457 passed.
  4c. Chunk D. EXPECT 177 passed.
  4d. Benchmark scoring:
        .venv/Scripts/python.exe scripts/bench_acceptance.py \
          --run-dir runs/2026-08-02_2130 \
          --json docs/pwml_recovery_sprint/evidence/baseline_acceptance.json

      A copy of this file is ALREADY COMMITTED, generated at
      b0b95184f1c5c2693058e4d22ddd128cc8988a27 with
      sha256 d3538f4b1cefc1f8e7aca933318df13c9967ce1071e48f8e6a3e4bd6830f4ec3
      (242,199 bytes). See evidence/PROVENANCE.md.

      Re-run it and COMPARE. If your output's sha256 differs, investigate before
      proceeding: either an input drifted or the scorer changed, and every
      downstream acceptance criterion is invalid until that is explained.

      EXPECT: false real identifiers 10 | unsupported reactions 7 | orphaned
      references 2 | missing supported reactions 3 | placeholder-backed proteins
      21 | requested-pathway coverage 0/8 | semantic confirmed 0/8 |
      strict PWML 0/4.
      If ANY differs, STOP.
  4e. Record FULL_STACK_BASELINE and RESIDUAL_CODES_BY_{LEG,ROW} as currently
      pinned in tests/test_strict_quarantine_real_artifact_replay.py. C-010
      changes these BY DESIGN (TRAP-2); the pre-change values must be on record
      so the delta is provable.

STEP 5 -- CONFIRM THE INTEGRATION BRANCH
  You are already on sprint/pwml-recovery. Confirm, and report its SHA after your
  commits. This is what Wave A0 cuts from.

DO NOT
  - dispatch any agent
  - modify anything under src/
  - mark any implementation branch READY
  - resolve open questions O-1, O-2 or O-3 on your own authority

REPORT
  ## PROCESS-LIFECYCLE HELPER
       path | reused-or-created | how it isolates (Job Object / process group)
       six synthetic cases: command | exit reason | exit code | descendants
       observed | descendants terminated | SURVIVORS (must be 0) | proof method
  ## INTEGRATION SHA (before and after)
  ## WORKTREE RESOLUTION   per file
  ## EVIDENCE COMMIT       file count | size | cache_snapshot excluded? (verify)
  ## BASELINE              4a-4e, with the baseline_acceptance.json sha256 compared
  ## CLEANUP REPORTS       one row per heavy job in step 4; surviving count = 0
  ## OPEN QUESTIONS        O-2 / O-3 status
  ## ANYTHING UNEXPECTED

STOP AFTER THIS. Do not dispatch Wave A0.
```
