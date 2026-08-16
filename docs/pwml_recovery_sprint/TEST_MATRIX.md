# PWML Recovery Sprint — Test Matrix

**`--basetemp` is mandatory on every invocation.** Without it, 83 tests error with
`PermissionError` and you will report a false regression. Never run the full suite
unchunked — it approaches 16 GB.

Runtimes measured on `ORIGIN_SHA`, Windows, `.venv/Scripts/python.exe` (3.13.6).

---

## 0. Test-process lifecycle policy — HARD MERGE RULE (G11)

**This is a merge gate, not a suggestion.** A run that violates it is an
**infrastructure failure**, not a test result, and must never be reported as passed.

**Why it exists.** Orphaned pytest, Streamlit and LLM descendants outlive their parent
and consume the developer's machine memory for hours. A full suite alone approaches
16 GB, and a killed batch child can leave a Streamlit process holding the socket that
hung it.

### The rules

1. **Bounded, wrapped, tracked.** Every test, benchmark, pipeline leg and LLM-backed
   command runs through the bounded-process wrapper whose path INIT-001 records.
   **Never** detached processes, `nohup`, untracked background jobs, or
   `Start-Process` without bounded waiting and guaranteed cleanup.

   **Foreground is the default. Tracked background is authorized when a bounded job is
   expected to exceed the interactive limit** — see **D-026**, which settled this after
   three cards (C-034, C-041a, C-050a) relitigated it independently. A `qb` cohort is
   ~10.5 min across 23 AppTest processes and exceeds the 10-minute interactive cap by
   construction.

   A tracked background job is compliant **only** when all of these hold: same approved
   `bounded_run.py`, unmodified · task/process id and output path recorded **immediately** ·
   **one heavy job at a time** · the orchestrator **polls rather than launching duplicates** ·
   wrapper cleanup executes · descendant counts and **zero survivors verified** · the final
   canonical JSON report **inspected** · no detached or unowned job remains.

   Prefer **one tracked bounded cohort** where splitting would change the gate's semantics or
   materially increase overhead. Use `--only` partitions only where the gate is explicitly
   partition-safe — `chunk_d_gate.py` proves its `179 = 152 + 4 + 23` partition on every
   invocation, so its partitions qualify.

2. **The wrapper must** record root PID, command, start time, working directory, timeout
   and ownership · place the command in an isolated process group (POSIX) or Job Object
   (Windows) · enforce an outer wall-clock timeout · run cleanup in `finally`/trap on
   **every** exit path including cancellation and agent failure · terminate all remaining
   descendants **owned by that job** · attempt graceful termination first, then forced
   termination after a short grace period · **verify** no tracked process from that job is
   still alive · preserve and return the **real** test exit code unless cleanup
   verification itself failed.

3. **Platform.** Windows: prefer a Job Object configured to terminate members when
   closed; failing that, track the root PID and use `taskkill /PID <owned-pid> /T /F`
   inside guaranteed cleanup. POSIX: new process group, `TERM` the group, then `KILL`
   after the grace period.

4. **Never global cleanup.** `taskkill /IM python.exe`, `pkill python`, or killing every
   Java/Node/pytest/Python process are **forbidden**. Cleanup may target only PIDs and
   process groups created and recorded by the current job. Pre-existing processes are
   **reported**, never silently killed.

5. **Completion is not "pytest printed a summary."** A job is complete only when the root
   process exited **and** all owned descendants exited **and** cleanup verification
   passed **and** exit status plus cleanup result were recorded.

6. **Survivors are an infrastructure failure.** If any owned process survives cleanup:
   classify the run as an infrastructure failure, **stop further dispatch**, and report
   the surviving PID, command line, start time and memory usage. Do not report the test
   as passed.

7. **One heavy job at a time.** At most one full suite, benchmark or memory-heavy
   pipeline leg concurrently. **Never `pytest -n auto`.** Never concurrent full
   benchmarks. Focused tests may run concurrently only when their resource limits and
   ownership remain explicit.

8. **Basetemp.** Keep the unique `--basetemp` path. Remove temp directories after
   completion when safe — but do not confuse temporary *files* with active *memory*.
   Deleting a basetemp directory does not reclaim a leaked process's RAM.
   The parent of `--basetemp` must already exist; pytest does not create intermediate
   directories and every test errors in setup if it is missing.

9. **The cleanup report is committed, not pasted.** Every job's `--json` report goes to a
   path allocated under `evidence/g11/<TASK-ID>/<SEQ>-<label>.json` and is committed with
   the branch. A pasted table is not evidence: a G11 claim must be checkable against a
   committed artifact. See `evidence/g11/README.md`.

10. **The measured tree is verified, not asserted.** Every pytest invocation that is a
    merge gate, a focused obligation, a G9 proof or a baseline capture runs through a
    launcher that **verifies, before collection begins**, that the imported `t2pw` and —
    where the selection needs it — the repository `scripts` package both resolve inside
    the tree under measurement. On any mismatch the launcher prints
    `T2PW_MEASUREMENT_TREE_REFUSED` with the expected tree, the resolved `t2pw.__file__`,
    the cwd, `PYTHONPATH` and `sys.path[0:4]`, and **exits 98 without counting a test**.

    **Exit 98 is reserved sprint-wide** for this and nothing else. It is a *measurement
    failure*, in the same class as an infrastructure failure (rule 6): never a test
    result, never a pass. It is distinct from 97 (`bounded_run.EXIT_INFRASTRUCTURE_FAILURE`),
    124 (timeout), 130 (cancelled) and pytest's 0–5. Do not allocate it to anything else.

    **`PYTHONPATH` is not evidence and a printed path is not evidence.** The venv's
    editable `.pth` names the primary checkout's `src`, `pytest.ini` sets **no**
    `pythonpath` and there is no `conftest.py`; separately, any in-process
    `sys.path.insert(0, …)` — including `_repo_root.add_src_to_path()` and the self-pin in
    24 of the 27 smoke and Chunk D test modules — **overrides `PYTHONPATH`** in a worktree
    while being a no-op in the primary checkout. Only the **resolved** path, compared
    against the expected tree and written to a committed verdict, settles which tree was
    measured. `pytest.ini` **must not** gain `pythonpath = src`: it was considered as a
    remedy for F-003 and is **refused**, because pytest *prepends* those entries, so it
    would sit ahead of the `PYTHONPATH` pin and make every base-tree G9 proof silently
    measure the tip — the same defect class as F-003, aimed at the proofs themselves.

    **"Inside the tree" means inside the package directory, not the tree root.** `t2pw`
    must resolve under `<expected>/src/t2pw` and `scripts` under `<expected>/scripts`: every
    agent worktree lives at `<primary>/.claude/worktrees/`, inside the primary checkout, so
    a root-level containment test would pass ~70 wrong trees and vouch for each. An
    `--expect-tree` override is validated by the same rule and refused with
    `EXPECT_TREE_NOT_A_CHECKOUT`, so naming a common ancestor cannot launder a run.

    **Every gate, G9 proof and baseline capture must pass `--pin-verdict`.** A run with no
    verdict is **uncertifiable**, exactly as a job with no cleanup report is; a verdict
    requested and unwritable is itself a refusal (`VERDICT_UNWRITABLE`, exit 98). The
    verdict goes to `evidence/g11/pin/<TASK-ID>/<SEQ>-<label>.pin.json`, committed with the
    branch — required because `bounded_run.py` records no environment and discards child
    stdout (rule 9 and F-004). It goes in `g11/pin/`, **not** `g11/<TASK-ID>/`: measured, a
    `.pin.json` beside the cleanup reports is picked up by `iter_reports`, checked against
    the `bounded_run` schema and fails `g11_evidence.py check --task` with 22 spurious
    violations. `pin` does not match `TASK_RE`, so that directory is skipped.

#### The measured launcher

```bash
# The pinned form of any focused / G9 / baseline pytest run. Sets cwd and sys.path[0]
# to the tree under measurement exactly as `python -m pytest` from the repo root does,
# then REFUSES with exit 98 if t2pw or scripts resolve elsewhere.
<py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label <l> --timeout <s> \
     --json docs/pwml_recovery_sprint/evidence/g11/<ID>/<SEQ>-<l>.json -- \
     <py> -u docs/pwml_recovery_sprint/evidence/pinned_pytest.py \
       --pin-verdict docs/pwml_recovery_sprint/evidence/g11/pin/<ID>/<SEQ>-<l>.pin.json \
       -q --basetemp=<short-tmp> <selection...>
```

`docs/pwml_recovery_sprint/evidence/c045_pinned_pytest.py` is a thin delegator to the same
`main()`, so its two committed invocation forms keep working unchanged.

- `--expect-tree <dir>` overrides the default (the launcher's own checkout). Required only
  when measuring an **exported base tree** from another checkout's launcher.
- `--require-scripts` / `--no-require-scripts` override the default, which is
  **selection-derived and off**: a selection whose sources do not visibly import `scripts`
  does not demand it, so no currently-green command is newly failed.
- **Base trees exported by `c045b_base_tree.py` / `c051a_base_tree_batch.py` do not contain
  `scripts/`** (`PATHSPEC`, `c045b_base_tree.py:35-38`), so Chunk B and SMOKE cannot be
  measured on one until that `PATHSPEC` is widened — **Finding H-3**. This is the second
  reason `require_scripts` defaults off.

### Cleanup report — required on every test record

| Field | |
|---|---|
| root PID / process group | |
| timeout | |
| exit reason | `completed` \| `nonzero` \| `timeout` \| `cancelled` \| `infrastructure_failure` |
| exit code | the real one |
| descendants observed | |
| descendants terminated | |
| final surviving count | **must be 0** |
| cleanup success/failure | **must be `true`** — a count of 0 alone is not sufficient |
| measured-tree verdict | path to the `*.pin.json`; **`violations` must be `[]`** |

### Durable evidence — where that report lives

```bash
# 1. allocate a unique path BEFORE the job (never hand-write one, never reuse one)
<py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py next --task <ID> --label <l>
# 2. run the job with --json pointing at it
<py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label <l> --timeout <s> \
     --json docs/pwml_recovery_sprint/evidence/g11/<ID>/<SEQ>-<l>.json -- <cmd...>
# 3. validate, then commit the report with the branch
<py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py check --task <ID>
```

A job with no committed report is **uncertifiable under G11** — it is not a passed test.
The artifact must exist and validate on its own; an acceptable exit code proves nothing,
because an unwritable `--json` still returns the child's real code. Reports are
credential-free and small: no captured stdout, no logs, no caches.
`evidence/g11/README.md` states the required fields, the naming scheme and the credential
and size rules. **Prospective only — no historical report is ever reconstructed.**

### Existing machinery to build on

`batch/runner.py` already implements owned-PID tree termination and is the correct model
— it is **not** a global killer:

| Function | Lines | Does |
|---|---|---|
| `launch_child` | 1140–1180 | `CREATE_NEW_PROCESS_GROUP` (nt) / `start_new_session` (posix); bounded `communicate(timeout=)`; `_kill_tree` on both `TimeoutExpired` and `KeyboardInterrupt`; `_DRAIN_TIMEOUT` for pipe close |
| `_kill_tree` | 1107–1137 | `taskkill /F /T /PID <owned pid>`, then `os.killpg`, then `proc.kill()` — **owned PID only** |
| `child_env` | 265–276 | child environment |

**What it lacks** against this policy: no graceful-then-forced escalation (it goes
straight to `/F`), no post-kill survivor verification, no structured cleanup report, and
it is used only for batch legs — **not for pytest runs**. INIT-001 extends this
discipline into an orchestration-only wrapper rather than replacing it, and must not
modify `runner.py` (that file is owned by C-032).

---

## Chunks

| Chunk | Files | Tests | Runtime |
|---|---|---|---|
| **A** | `test_reference_repair`, `test_strict_quarantine`, `test_strict_quarantine_contract_alignment`, `test_strict_quarantine_locks_and_scope`, `test_strict_quarantine_versioning`, `test_empty_extraction_payload` | 123 | **12 s** |
| **B** | `test_bench_goldset_and_semantic`, `test_bench_acquisition_and_artifacts`, `test_bench_controls`, `test_completeness_audit`, `test_batch_driver`, `test_stage3_gate_report_lifecycle` | 225 | **25 s** |
| **C** | `test_rag_admission_production_path`, `test_rag_gap_admission`, `test_rag_triage_orchestration`, `test_rag_provenance_gates`, `test_pipeline_reaction_rag_provenance`, `test_research_mode_orchestration`, `test_map_ids_name_gate`, `test_db_candidate_species_evidence` | 109 | **2 s** |
| **D-core** | `test_process_normalizer`, `test_pwml_ir`, `test_pwml_writer`, `test_stage_contracts`, `test_payload_models` | 152 | **0.9 s** |
| **D-apptest** | `test_streamlit_stage8_export_contract` · `test_streamlit_quarantine_boundary` — one process **per NODE** (H-007) | 4 + 23 | ~10.5 min, all 27 |
| **E** | `test_strict_quarantine_real_artifact_replay` | parameterized over `runs/` | tens of s per leg |

**SMOKE = A + B + C = 460 tests, ~40 s.** Runs after **every** merge, on the integration
branch. Gate G10. **457 was the INIT-001 figure and is obsolete**: C-010 moved the pinned
baseline deliberately, 457 → 460, with an exact documented delta, and every A0 merge from
`72ee20f` onward measured 460. Any live instruction still saying 457 is stale.

**Chunk D is excluded from the smoke gate.** Its deterministic core is 152 tests in
**~1 s**, but the complete 179-test gate cost **9–13 min** over six runs — the 27 AppTest
processes, not the core — which is too slow per merge. It is **mandatory as a focused
test** for every branch marked ✔ below, because that is exactly where their regressions
land and none of it appears in the smoke suite.

**Chunk E skips silently when `runs/` is absent.** `runs/` is committed, so a clean clone
has it; `runs_verify/2026-08-04_1754/` is not yet committed and INIT-001 must fix that or
C-010's allowlist is unverifiable in an isolated worktree.

---

## Commands

```bash
# SMOKE (every merge) — expect 460 passed
.venv/Scripts/python.exe -m pytest -q --basetemp=<tmp>/smoke \
  tests/test_reference_repair.py tests/test_strict_quarantine.py \
  tests/test_strict_quarantine_contract_alignment.py \
  tests/test_strict_quarantine_locks_and_scope.py \
  tests/test_strict_quarantine_versioning.py tests/test_empty_extraction_payload.py \
  tests/test_bench_goldset_and_semantic.py tests/test_bench_acquisition_and_artifacts.py \
  tests/test_bench_controls.py tests/test_completeness_audit.py \
  tests/test_batch_driver.py tests/test_stage3_gate_report_lifecycle.py \
  tests/test_rag_admission_production_path.py tests/test_rag_gap_admission.py \
  tests/test_rag_triage_orchestration.py tests/test_rag_provenance_gates.py \
  tests/test_pipeline_reaction_rag_provenance.py tests/test_research_mode_orchestration.py \
  tests/test_map_ids_name_gate.py tests/test_db_candidate_species_evidence.py

# CHUNK D — AUTHORITATIVE GATE is the split-process runner, never the one-process form.
# ONE call runs the whole gate: it proves the partition, runs the 152-test core in one
# process, then each of the 27 AppTest node IDs ALONE in a fresh process, serially.
# --task lets it allocate its own ~32 G11 reports; 32 paths do not fit on a CLI.
#
# *** T2PW_OFFLINE_CURATOR=1 IS MANDATORY ON EVERY DETERMINISTIC qb RUN. ***
# Required from C-050b (merged 1383624). Without it, run_pathway_curator issues ONE
# ungated LLM call per post-pipeline app run at temperature 0.2, whose accepted patches
# are written back into audited_json and flow through mapping into final_mapped_db.
# That is the MEASURED root cause of BL-003. And because .env is untracked, a worktree
# silently gets LLM_PROVIDER=local (call 400s, exception swallowed, curator a no-op BY
# ACCIDENT) while the primary checkout issues real BILLED remote calls -- so a green qb
# cohort obtained in a worktree does NOT certify the same cohort in the primary.
# Set it in the BOUNDED CHILD environment, not just your shell.
T2PW_OFFLINE_CURATOR=1 \
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label chunkd --timeout 3000 --json <outer-report> -- \
  .venv/Scripts/python.exe -u docs/pwml_recovery_sprint/evidence/chunk_d_gate.py run \
  --tmp <short-tmp> --task <ID> --timeout 900 --node-timeout 600

# The flag is opt-in and default-off: omitting it preserves production behaviour exactly.
# An ACCEPTANCE run that deliberately exercises the live curator is separate, bounded work
# requiring explicit cost authorization -- it is NOT a deterministic gate.
#   --only core|s8|qb narrows EXECUTION; the partition proof always covers all three.
#   --label-prefix attributes each run's artifacts when a matrix runs the gate repeatedly.
#   --report-root writes them to the branch when the measured tree is an export.
# The runner's own focused tests (parser + runner, 10 cases):
#   ... bounded_run.py --label chunkd-selftest --timeout 900 --json <report> -- \
#       .venv/Scripts/python.exe -u docs/pwml_recovery_sprint/evidence/chunk_d_gate_selftest.py

# CHUNK E (quarantine artifact replay)
.venv/Scripts/python.exe -m pytest -q --basetemp=<tmp>/e \
  tests/test_strict_quarantine_real_artifact_replay.py
```

### Chunk D — split-process gate (RECONCILE-B, execution partition superseded by H-007)

The one-process form ran all 179 tests together and **flapped**, on a *different* test each
time; the exact committed evidence is tabulated under § "The historical `qb` sample" below,
which supersedes every looser count of it. Cause, documented at
`tests/test_streamlit_quarantine_boundary.py:425-430`: several `AppTest` instances in one
process eventually lose their `ScriptRunContext`, so `streamlit_app.py:6187` → `ui.py:26`
raises `FragmentThreadState not initialized` and the test fails on a widget never created.

`chunk_d_gate.py` runs the same 179 tests as isolated processes. **Set-identity is proven
on every invocation** — `partition` compares node-ID *sets*: 179 = 152 + 4 + 23, missing 0,
extra 0, overlap 0 — and `run` then compares the set it EXECUTED to the set it collected,
so a substitution fails the gate even when every job it ran was green.

### The execution partition is per NODE, not per file (H-007)

**RECONCILE-B's per-FILE partition was insufficient, and this supersedes the reading that
no process partition could work.** The documented cause is *intra-file* — one file builds
23 `AppTest` objects in one process — so running that whole file in one fresh process
leaves the cause in place, which is what the RECONCILE-B measurements show. Per-node
isolation had never been tested. It now is:

| Component | Execution | Result | Runtime |
|---|---|---|---|
| `core` (5 files, 152 tests) | one process | **152 passed** — deterministic | ~1 s |
| `s8` (4 AppTest tests) | **4 processes, one per node** | **4 × 1 passed** | ~6 s |
| `qb` (23 AppTest tests) | **23 processes, one per node** | see § below | ~10.5 min |

### The historical `qb` sample — a fixed observation, cited exactly

The earlier tally *"eight known runs: 3 green · 3 red · 2 killed at too short a bound"*
is **retained as a historical observation and is not a probability or a property**. It is
corrected here in one respect: **it mixed two different selection shapes and included runs
with no committed artifact.** What the repository can actually support:

| Selection | Committed artifacts | Result |
|---|---|---|
| monolithic, all **7** files | `evidence/g11/C-011/` `04` `05` `07` `18` `25` `26` `44` `45` | **2 green** (`05` 449 s, `18` 494 s) · **6 red** (414–528 s) |
| `qb` file **alone** | `evidence/g11/RECONCILEB-001/` `18` `19` `21` `34` | **0 green · 2 red** (453 s, 455 s) · **2 killed at bound** (`18` 488 s, `19` 308 s) |

The green-vs-red duration intervals overlap, so duration predicts nothing. **No committed
artifact records a green `qb`-alone run**; the "3 green" of the old tally rests on
observations, including a reviewer run, for which **no `--json` cleanup report was
committed**, and that is stated plainly rather than given a false precision. G11 keeps no
stdout, so the failing test *names* from those runs are lost.

**Never read a `qb` red as expected.** A new deterministic failure, a candidate-only
failure, or a traceback implicating the diff under test still blocks a merge.

### What blocks a merge today (D-022, measured by H-007's six-run matrix)

**Chunk D is 179 tests, and all 179 are still mandatory to run and to report.**

| Component | Status |
|---|---|
| `core` 152 + `s8` 4 = **156** | **BLOCKING.** Green in all six runs, both trees |
| `qb` 23 | **Mandatory to run and report; temporarily NON-BLOCKING** |

The six runs gave **base 1 green / 2 red, candidate 1 green / 2 red** on trees whose `tests/`
and `src/` are byte-identical: a test-infrastructure race, filed as **BL-003**.

**This is not permission to read a `qb` red as expected.** A new *deterministic* failure, a
failure a diff *reproduces*, or a traceback *implicating* the diff still blocks the merge,
and no `qb` test may be weakened, deselected, retried until green or called environmental.

### Chunk D cadence — forward policy (D-023)

- Deterministic core plus `s8`, **154 tests: BLOCKING** on every D-marked card.
- The **23-node `qb` cohort: mandatory to run and to report.**
- **C-030, C-050 and C-052** directly touch the Streamlit/freeze/UI seam and **must run
  `qb` before merge**.
- **Non-UI D-marked cards such as C-040 and C-051** may run `qb` once at their **pack-level
  integration checkpoint**.
- A **deterministic, diff-reproducible or traceback-implicated `qb` failure blocks** the
  affected lane.
- **No deselection, no retry-until-green, no false environmental relabelling.**

---

## Per-branch obligations

| Branch | Focused | Chunk D | Golden diff | Notes |
|---|---|---|---|---|
| C-010 | A, E | — | — | **plus** the exact per-leg allowlist test (6 legs) and a re-pinned `FULL_STACK_BASELINE` |
| C-011 | D | ✔ | ✔ artifacts dict | pure move — golden diff must be EMPTY |
| C-012 | B | — | ✔ full driver-observable behaviour | see below |
| C-013 | smoke | — | — | |
| C-014 | A, C | — | — | |
| C-015 | new only | — | — | |
| C-016 | new, C | — | — | |
| C-017 | B | — | — | |
| C-018 | new, C | — | — | |
| C-020 | new, D | ✔ | — | equivalence proven by parsing JSON + PWML + SBML |
| C-021 | new, C | — | — | |
| C-030 | D | ✔ | — | |
| C-031 | B | — | ✔ driver | |
| C-032 | B | — | ✔ driver | |
| C-033 | C | — | — | |
| C-034…C-037 | A or C | — | — | |
| C-038 | A + `test_pipeline_reaction_rag_provenance` | — | — | |
| C-040 | D | ✔ | — | |
| C-041 | A, B | — | ✔ driver | |
| C-042 | A | — | — | |
| C-043 | C | — | — | |
| C-044 | C | — | — | |
| C-050 | D | ✔ | — | |
| C-051 | D | ✔ | — | |
| C-052 | D | ✔ | — | |
| C-053 | B | — | ✔ driver | |
| C-054 | B | — | — | |
| C-055 | C + AppTest (`test_rag_triage_orchestration`, `test_batch_driver`) | — | — | least-verifiable branch; senior reviewer |
| C-056a | B | — | — | |
| C-056b | B | — | — | |
| C-057 | A, E | — | — | |

**"Chunk D" here = the split-process gate** (`chunk_d_gate.py`), all 179 tests, now
executed as the 152-test core plus 27 per-node AppTest processes — see § Chunk D and D-022.

---

## Golden behavioural-equivalence diffs

Two seams and every later `driver.py` branch must prove they changed nothing observable.

### C-012 and every `driver.py` branch — full driver-observable behaviour

Payload hash alone is insufficient: a driver refactor can preserve the biological payload
while dropping a quarantine report or misclassifying a timeout. Compare, byte-for-byte,
across every committed leg fixture:

| Field | Source |
|---|---|
| exit classification | `status`, `stage`, `failure_kind` |
| release status | `release_status` (absent pre-C-041 — assert absent) |
| artifact filenames and paths | `sorted(outcome.artifacts.keys())`, `files[].name` after `_relocate_files` |
| persisted diagnostic artifacts | `files[].bytes` per name |
| manifest fields | full `RunOutcome.to_dict()` (`driver.py:714`) |
| failure reasons | `issue_codes`, `detail` |
| messages | `message`, `warnings` |
| canonical payload hash | `canonical_payload_sha256` |

### C-011 — canonical-freeze seam, same protection

A freeze-seam refactor can preserve the payload while dropping a gate report, changing
`sbml_input_source`, or reordering a side effect. Compare the `post_pipeline_artifacts`
dict before and after on: sorted key set · `canonical_payload_sha256` ·
`final_stage3_gate_report` (full dict incl. `phase`, `payload_sha256`) · `quarantine_ok`,
`quarantine_refusal_reasons`, `quarantine_coverage` · `sbml_input_source` ·
`final_mapped` ↔ `final_mapped_quarantined` identity relationship · presence of
`tmp/final.canonical.json`.

"Before" fixtures are captured on `ORIGIN_SHA` and committed under `evidence/`.

---

## Regression-test standard (gate G9)

Authoritative statement (D-023). Every other live site is a compact restatement of it.

1. **G9 is required for a claimed correction or preservation of pre-existing observable
   behaviour.** That claim is what needs proof.
2. The proof must **fail behaviourally at the base SHA and pass at the tip**. The reviewer
   checks out the base, applies only the test, runs it, and pastes the failure.
3. **Symbol absence is not behavioural proof.** A test that errors at the base only because
   a new name does not exist proves nothing — supply a shim, or assert on artifact content,
   so the base fails on *behaviour*.
4. A genuinely **new capability or module** receives an **explicitly labelled new
   acceptance test** and does **not** require a fabricated base failure.
5. **Reviewers must reject any attempt to mislabel a regression as new functionality.** A
   test that passes without the fix is not a regression test.

---

## Milestone benchmarks

Scheduled by the Lead Orchestrator only. Agents never launch a benchmark.

| ID | Milestone | After | Legs | Wall clock | Acceptance |
|---|---|---|---|---|---|
| T-100 | M1 | Wave B | PMC12452463 ×2, PMC12096016 ×2 | ~1.5 h | both pass the quarantine boundary; **PMC12452463 → `review_required`, not strict success** (TRAP-1) |
| T-101 | M2 | Wave C | + PMC12444477 ×2, PMC12782028, PMC12312563 | ~2 h | no leg reports "produced nothing"; `identical_empty_response` recorded where two draws share a hash; `budget_exhausted` distinct from failure |
| T-102 | M3 | C-052 | PMC12856317 | ~25 min | reload + re-export with **all resolvers disabled** → identical `canonical_graph_sha256`; equivalence proven by parsing and normalizing JSON, PWML and SBML |
| T-103 | M4 | C-055 | 4 RAG legs | ~1.5 h | every RAG round re-entered normalization, mapping, gates, persistence, classification |
| T-104 | M5 first RC | Wave E | full pinned, 20 legs | ~7 h | full acceptance matrix vs `BASELINE.md` |
| T-105 | M5 second RC | Day 6 | full pinned, 20 legs | ~7 h | remaining failures explained and classified |

Measured leg times: 1308 s and 1511 s. A full run is ~7 h; the runner's own
`DEFAULT_DEADLINE_HOURS = 10.0` confirms overnight sizing.

**Never use a full benchmark as a per-patch merge gate.** And before calling any
single-leg change a regression, re-run that leg — identical legs give materially
different Stage-1 draws at temperature 0 in this repository.

---

## Baseline to preserve (filled by INIT-001)

Full suite per-chunk counts · smoke **460** (457 at INIT-001, moved 457→460 by C-010 with
an exact documented delta) · chunk D 179 · `bench_acceptance.py` on
`runs/2026-08-02_2130` · `FULL_STACK_BASELINE` and `RESIDUAL_CODES_BY_{LEG,ROW}` as
currently pinned. See `BASELINE.md`.

> **C-011 clarification (C011-ARTIFACT-MANIFEST-RATIFICATION-001).** For C-011 the
> "before" state is the behaviour at its **dispatch-time `BASE`** — the integration SHA
> its branch is cut from — not `ORIGIN_SHA`. `ORIGIN_SHA` is retained as an **equality
> witness**: C-011 must prove `run_post_pipeline_sbml_artifacts` is unchanged between
> the two and record both SHAs plus the result in
> `evidence/c011_freeze_seam_before.json`, its one authorized artifact. Inequality is a
> hard stop. No other card's obligation changes.
