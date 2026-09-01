# PWML Recovery Sprint — Test Matrix

**`--basetemp` is mandatory on every invocation.** Without it, 83 tests error with
`PermissionError` and you will report a false regression. Never run the full suite
unchunked — it approaches 16 GB.

**Two infrastructure modes look exactly like a large regression (F-114, folded in by
D-066).** The first is the omitted `--basetemp` above. The second: **a `--basetemp` whose
PARENT directory does not exist errors the run outright** — pytest does not create
intermediate directories, so every test errors in setup. One measured instance errored
**55 tests**; creating the parent gave **339 passed** on the same tree and selection.
**Pre-create the basetemp parent before pytest.** Neither mode is a test result.

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
   partition-safe — `chunk_d_gate.py` proves its `187 = 160 + 4 + 23` partition on every
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
    measured. **`pytest.ini` carries `pythonpath = src`** (C-070, `5bc600e`), which this
    rule previously **refused**. Under **D-066 (LOCKED, 2026-08-25) the refusal is
    SUPERSEDED, not forgotten.**

    The hazard it named is real and unchanged: pytest *prepends* those entries, so the
    setting sits ahead of the `PYTHONPATH` pin and could make a base-tree G9 proof
    silently measure the tip — the same defect class as F-003, aimed at the proofs
    themselves. It is neutralised by the **pin**, not by the setting's absence. Removing
    the setting would re-break individual-file collection for 21 of 156 test files
    (the real, unrelated defect C-070 fixed) to solve a problem the pin already solves.
    Neither side of the contradiction was wrong on its own merits and neither author
    knew of the other; **this is not chargeable to C-070 or C-079.**

    **Therefore, mandatory rather than customary: every base-tree measurement MUST run
    through `pinned_pytest.py` with `--expect-tree` and a committed `--pin-verdict`. An
    unpinned base-tree run is NOT evidence.**

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
# then REFUSES with exit 98 if t2pw resolves elsewhere. PYTHONPATH IS REQUIRED (F-076):
PYTHONPATH=<tree>/src <py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label <l> --timeout <s> \
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
| **A** | `test_reference_repair`, `test_strict_quarantine`, `test_strict_quarantine_contract_alignment`, `test_strict_quarantine_locks_and_scope`, `test_strict_quarantine_versioning`, `test_empty_extraction_payload` | 134 | **12 s** |
| **B** | `test_bench_goldset_and_semantic`, `test_bench_acquisition_and_artifacts`, `test_bench_controls`, `test_completeness_audit`, `test_batch_driver`, `test_stage3_gate_report_lifecycle` | 230 | **25 s** |
| **C** | `test_rag_admission_production_path`, `test_rag_gap_admission`, `test_rag_triage_orchestration`, `test_rag_provenance_gates`, `test_pipeline_reaction_rag_provenance`, `test_research_mode_orchestration`, `test_map_ids_name_gate`, `test_db_candidate_species_evidence` | 109 | **2 s** |
| **D-core** | `test_process_normalizer`, `test_pwml_ir`, `test_pwml_writer`, `test_stage_contracts`, `test_payload_models` | 160 | **0.9 s** |
| **D-apptest** | `test_streamlit_stage8_export_contract` · `test_streamlit_quarantine_boundary` — one process **per NODE** (H-007) | 4 + 23 | ~10.5 min, all 27 |
| **E** | `test_strict_quarantine_real_artifact_replay` | parameterized over `runs/` | tens of s per leg |

**SMOKE = A + B + C + the two C-106 gate files = 503 tests, ~38 s.** Runs after **every** merge, on the integration
branch. Gate G10. **457, 460, 465 and 473 are all stale**: C-010 moved 457 -> 460, C-054 moved 460 -> 465, C-067 moved
465 -> 473 (eight ADDED tests, `test_strict_quarantine.py` 32 -> 40; Chunk **A** 126 -> 134), and **C-106 moved 473 -> 503**
(+14 `test_c102_coverage_denominator.py`, +16 `test_c106_mutation_harness_executable.py` — see the C-106 entry at end-of-file). Each under merge rule 4; nothing has ever been removed.

**Chunk D is excluded from the smoke gate.** Its deterministic core is 160 tests in
**~1 s**, but the complete 187-test gate cost **9–13 min** over six runs — the 27 AppTest
processes, not the core — which is too slow per merge. It is **mandatory as a focused
test** for every branch marked ✔ below, because that is exactly where their regressions
land and none of it appears in the smoke suite.

**Chunk E skips silently when `runs/` is absent.** `runs/` is committed, so a clean clone
has it; `runs_verify/2026-08-04_1754/` is not yet committed and INIT-001 must fix that or
C-010's allowlist is unverifiable in an isolated worktree.

---

## Commands

```bash
# SMOKE (every merge) — expect 503 passed  (22 files; C-106 added the last two, +14 +16)
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
  tests/test_map_ids_name_gate.py tests/test_db_candidate_species_evidence.py tests/test_c102_coverage_denominator.py tests/test_c106_mutation_harness_executable.py

# CHUNK D — AUTHORITATIVE GATE is the split-process runner, never the one-process form.
# ONE call runs the whole gate: it proves the partition, runs the 160-test core in one
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

The one-process form ran all 187 tests together and **flapped**, on a *different* test each
time; the exact committed evidence is tabulated under § "The historical `qb` sample" below,
which supersedes every looser count of it. Cause, documented at
`tests/test_streamlit_quarantine_boundary.py:425-430`: several `AppTest` instances in one
process eventually lose their `ScriptRunContext`, so `streamlit_app.py:6187` → `ui.py:26`
raises `FragmentThreadState not initialized` and the test fails on a widget never created.

`chunk_d_gate.py` runs the same 187 tests as isolated processes. **Set-identity is proven
on every invocation** — `partition` compares node-ID *sets*: 187 = 160 + 4 + 23, missing 0,
extra 0, overlap 0 — and `run` then compares the set it EXECUTED to the set it collected,
so a substitution fails the gate even when every job it ran was green.

> **`187` holds only on a tree carrying C-050k (D-044 § 2).** The count moved twice, both
> times as an authorized merge-rule-4 baseline move with `SETS_EQUAL=True, missing=0,
> extra=0, overlap=0` across the move: `core` 150 → 152 (C-045b, `TOTAL` 177 → 179) and
> `core` 152 → 160 (C-050k, `TOTAL` 179 → 187). The C-050k `+8` is exactly the eight
> alias-ambiguity acceptance arms its charter required be added to `tests/test_pwml_ir.py`,
> a `core` file that previously carried **zero** `resolve_entity` / `ambiguous` / `synonyms`
> coverage. **Until C-050k merges, `179 = 152 + 4 + 23` remains correct**, and a card in
> flight that reports 187 on a tree without C-050k is measuring the wrong thing.

### The execution partition is per NODE, not per file (H-007)

**RECONCILE-B's per-FILE partition was insufficient, and this supersedes the reading that
no process partition could work.** The documented cause is *intra-file* — one file builds
23 `AppTest` objects in one process — so running that whole file in one fresh process
leaves the cause in place, which is what the RECONCILE-B measurements show. Per-node
isolation had never been tested. It now is:

| Component | Execution | Result | Runtime |
|---|---|---|---|
| `core` (5 files, 160 tests) | one process | **160 passed** — deterministic | ~1 s |
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

**Chunk D is 187 tests, and all 187 are still mandatory to run and to report.**

| Component | Status |
|---|---|
| `core` 160 + `s8` 4 = **164** | **BLOCKING.** Green in all six runs, both trees |
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

**"Chunk D" here = the split-process gate** (`chunk_d_gate.py`), all 187 tests, now
executed as the 160-test core plus 27 per-node AppTest processes — see § Chunk D and D-022.

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
| T-101 | M2 | Wave C | + PMC12444477 ×2, PMC12782028, PMC12312563 | ~2 h | no leg reports "produced nothing"; `identical_empty_response` recorded where two draws share a hash; `budget_exhausted` distinct from failure — **RAN 2026-08-21 (`runs_verify/2026-08-21_1822` + `_2014`, deepseek-v4-flash). STATUS: `MEASURED`, NOT PASS. Clause 1 VIOLATED (the phrase is hard-coded in the runner's timeout message, so it cannot pass for any outer-killed leg); clause 2 UNEXERCISED (no two draws shared a hash — not a pass); clause 3 INCONSISTENT — F-092. Never record as PASS.** |
| T-102 | M3 | C-052 | PMC12856317 | ~25 min | reload + re-export with **all resolvers disabled** → identical `canonical_graph_sha256`; equivalence proven by parsing and normalizing JSON, PWML and SBML — **STATUS: `MEASURED — organism/SBML axis structurally unreachable (F-009)`. This is its only legitimate status. Never record as PASS.** |
| T-103 | M4 | C-055 | 4 RAG legs | ~1.5 h | every RAG round re-entered normalization, mapping, gates, persistence, classification — **RAN 2026-08-21 (`runs_verify/2026-08-21_2057`). STATUS: `MEASURED — acceptance satisfied at round_count=1; MULTI-ROUND re-entry UNTESTED`. All 4 legs `round_count=1` (`RAG_LOOP_MAX_ROUNDS` unset → 1×, as D-058 predicted). Re-entry proven by `post_remap_contract_report` (a SECOND mapping contract after audit) plus post_normalization / pre_export / persisted artifacts / release block. "Every round" is here verified over a set of size ONE — do not quote it for N>1.** |
| T-104 | M5 first RC | Wave E | full pinned, 20 legs | ~7 h | full acceptance matrix vs `BASELINE.md` — **RAN 2026-08-21/22 (`runs_verify/2026-08-21_2239`, deepseek-v4-flash, 5.44 h, G11 0 survivors). STATUS: `MEASURED — NOT ACCEPTED`. Never record as PASS.** Topics set built this session as `topics_t104.txt` (none existed): the 10 pinned gold cases, scope lines verbatim from `bench/gold/pinned_v1.json`, ratified by `bench_acceptance.py --verify-plan` = `OK`, all 10 `[pinned_override]`. **BENCHMARK COMPLETION: `COMPLETE (10/10 papers, 20/20 legs)` — the first complete run of the sprint; T-100's `INCOMPLETE (2/10)` bar is cleared and rates are quotable.** Priorities: **1 FAIL** (7 false real identifiers, papers PMC12180156/PMC12782028/PMC12856317), 2 PASS (0 unsupported retained reactions), 3 PASS (0 referential-integrity violations), **4 FAIL** (requested-pathway coverage 0/7 = 0%), **5 FAIL** (strict PWML 0/4 = 0%). Separated denominators: extraction success 7/8 = 88%; semantic pathway success 0/7 = 0%; research deliverable produced 4/8 = 50%; research semantically confirmed 0/8 = 0%. Legs: 10 PASS, 6 `scope_conflict`, 2 `no_reactions`, 2 TIMEOUT. **Four triaged findings, F-094..F-097; F-092 re-confirmed. Only F-094 and F-096 are `product_contract_violation` and therefore the only two that justify code; F-095 and F-097 are `policy_disagreement` and need a product-owner decision, not a patch.** ⚠ **T-105 NOT RUN: its precondition is a correction pass that this session was not authorized to implement. Re-running the same 20 legs with no code change would collapse the two RCs, which the sprint forbids.** |
| T-105 | M5 second RC | **F-094 + F-096 corrections merged** (was: "Day 6") | full pinned, 20 legs | ~7 h | remaining failures explained and classified — **RAN 2026-08-22/23 (`runs_verify/2026-08-22_2147`, deepseek-v4-flash, 4.85 h, G11 0 survivors). STATUS: `MEASURED — NOT ACCEPTED`. Never record as PASS.** Preconditions met: C-072 (`d7f4f96`) and C-073 (`6373ad1`) both merged, independently reviewed, SMOKE 473 green after each; `--verify-plan` = `OK`, all 10 `[pinned_override]`, 0 search calls. **BENCHMARK COMPLETION `COMPLETE (10/10 papers, 20/20 legs)`, 19 scored.** Legs: 12 PASS, 6 `scope_conflict`, 2 TIMEOUT, **0 `no_reactions`**. Priorities: **1 FAIL** (7 false real identifiers — count unchanged from T-104 but composition almost entirely different, see below), **2 FAIL** (3 unsupported reactions — was PASS at T-104), **3 FAIL** (2 orphaned references — was PASS at T-104), **4 PASS** (coverage 1/8 = 12%, was 0/7 FAIL), **5 FAIL** (strict PWML 0/4). Separated denominators: extraction success **8/8 = 100%** (was 7/8); semantic pathway success **1/8 = 12%** — PMC12421875, at the time called **the first semantic confirmation of the sprint** -- **RETRACTED 2026-08-25 by C-085 / F-121.** That confirmation was an artifact of priority 2's blind spot: the leg retained 11 reactions, attributed 3 (27%) and left 8 matching nothing, and its unsupported-reaction verdict was never reached. Re-scored at C-085's tip the same run gives **priority 4 = 0/8 = 0%** and this run has **no** semantic confirmation; research deliverable 4/8 = 50%; research confirmed 0/8. ✅ **F-094 CLOSED: PMC12452463/strict → `review_required`, `strict_acceptance_eligible=false`, `pathway.review_required.pwml`, no bare `pathway.pwml`.** ⚠ **But the bare-PWML count went UP, 1 → 2**: PMC12856317/strict and PMC13231680/strict both reached `release_ready`, both on papers whose gold `export_rationale` forbids strict export → **F-101** and **F-100**. **C-072 correctly abstained on both** (4/4 anchors matched; 0 anchors declared) — the card fixed its mechanism, the class is wider. ⚠ **Priority 1 = 7 was predicted before the run and the number matching is coincidence** — `succinyl-CoA`, `SREBF1/2`, `LIPA`, `LBR` all vanished by draw variance and were replaced by `protoporphyrin IX`, `NADH`, `NAD+`, `holo-EntB`. **C-073 could not have fixed any of them:** its source-support pass is dormant (card §4a premise was false — no caller passes seed text to `merge_additions`), and its live cross-kind pass found 0 conflicts this run. **Four new findings F-100/F-101/F-102 + F-092 re-confirmed on both PMC12444477 legs.** F-100/F-101 are `product_contract_violation`; F-102 is `policy_disagreement` (the scorer flags the within-kind accession rule C-073's review rejected as contradicting D-035). **T-106 recommended** — do NOT rerun under the T-105 identity. |

Measured leg times: 1308 s and 1511 s. A full run is ~7 h; the runner's own
`DEFAULT_DEADLINE_HOURS = 10.0` confirms overnight sizing.

**Never use a full benchmark as a per-patch merge gate.** And before calling any
single-leg change a regression, re-run that leg — identical legs give materially
different Stage-1 draws at temperature 0 in this repository.

---

## Baseline to preserve (filled by INIT-001)

Full suite per-chunk counts - smoke **503** (457 at INIT-001; 457->460 C-010, 460->465 C-054, 465->473 C-067,
473->503 C-106, each an exact documented delta) - chunk A **134** - chunk D 187 -
`runs/2026-08-02_2130` · `FULL_STACK_BASELINE` and `RESIDUAL_CODES_BY_{LEG,ROW}` as
currently pinned. See `BASELINE.md`.

> **C-011 clarification (C011-ARTIFACT-MANIFEST-RATIFICATION-001).** For C-011 the
> "before" state is the behaviour at its **dispatch-time `BASE`** — the integration SHA
> its branch is cut from — not `ORIGIN_SHA`. `ORIGIN_SHA` is retained as an **equality
> witness**: C-011 must prove `run_post_pipeline_sbml_artifacts` is unchanged between
> the two and record both SHAs plus the result in
> `evidence/c011_freeze_seam_before.json`, its one authorized artifact. Inequality is a
> hard stop. No other card's obligation changes.


---

## Stale SMOKE / Chunk D counts still live elsewhere — enumerated by C-054, deliberately NOT edited

C-054 moved SMOKE 460 → 465 under merge rule 4 and updated every record inside its
bounds: this file, `MASTER_PLAN.md` § 5 *Merge gates — all must hold*, and the `## Test discipline` SMOKE bullet of `.claude/agents/pwml-test-runner.md`.
The entries below are **outside an implementer's bounds** — `prompts/` and `CLAUDE.md`
are the orchestrator's, `DECISIONS.md` is append-only and the product owner's. They are
listed rather than changed so they can be routed. **Verified at the C-054 + integration
merged tree, not assumed.** *(Appended at end-of-file on purpose: this file's line
addresses are pinned by citation up to `:477`, so nothing may be inserted above that.)*

**Would actively mis-instruct a live gate — route first:**

| Location | Reads | Why it bites |
|---|---|---|
| `prompts/_TEMPLATE_INTEGRATE.md:72` | `expect 460 passed` | the checklist G10 is executed from at **every** merge — it tells the next integrator to expect 460 and they will measure 465 |
| `prompts/_TEMPLATE_INTEGRATE.md:33` | `SMOKE SUITE (460 tests, ~40 s)` | same checklist |
| `prompts/_SHARED_BLOCKS.md:35` | `Smoke = 460`, `Chunk D = 177` | pasted verbatim into every card charter; **doubly stale — Chunk D has been 187 since C-050k** |
| `prompts/_SHARED_BLOCKS.md:36` | `core is 150`, `177-test gate` | same block, same paste path |
| `prompts/_TEMPLATE_INTEGRATE.md:36` | `gate (177 tests, 9-13 min)` | Chunk D, same integrate checklist |
| `CLAUDE.md:52` | `smoke suite (460 tests, ~40 s)` | merge rule 10 as stated to every agent on entry |
| `prompts/PROMPT-000-orchestrator.md:75` | `G10 smoke suite (460 tests, ~40 s)` | the orchestrator's own gate list |

**A standing rule that cites the baseline by value** — still correct as a rule, wrong as a
number: `FINDINGS.md:302` (F-018), *"Do not fix by adding files to a chunk — that moves the
pinned 460 baseline."*

**Historical records, correctly left alone.** These state what a past run *measured* and
rewriting them would falsify the record: `prompts/C-060a.md:76`, `prompts/H-007.md:97`,
`:134`, `:137`, `prompts/C-011.md:117`, `:122`, `prompts/INIT-001.md:144`,
`FINDINGS.md:1129`, and `LEDGER.md`'s 21 hits. `DECISIONS.md:799-802` states the 460
decision itself and only the product owner may amend it.

---

## Isolated collection (F-066 / C-070) — appended 2026-08-21 under D-061

**⚠ Why this sits at the end of the file and not in § Chunks, where a reader would look for
it.** The standing constraint pins citations through **line 477**, and § Chunks begins at the
**`## Chunks`** heading — ~~line 209~~, struck by C-112; measured `:228`. Inserting there would break those
citations. **D-061 authorized the file to grow; it did not authorize moving pinned lines.**
So the entry is appended and § Chunks is left byte-identical. Cross-referenced, not relocated.

`tests/test_isolated_collection.py` runs **two sub-second arms** in any selection that
includes it:

* `test_pytest_ini_places_src_on_sys_path` — the mechanism.
* `test_a_naive_new_test_file_collects_alone` — a **generated** canary, not a named list, so
  test file 157 is covered the day it is written without touching this file.

Its **complete sweep** — every `tests/test_*.py` collected alone in a fresh interpreter,
**~95 s for 156 files**, 156 interpreter starts — is opt-in via **`T2PW_ISOLATED_COLLECT_ALL=1`**
and **skips by default**. It belongs at **release-gate / pre-merge-window cadence, not per
merge**: 94 seconds per merge is a test someone later deletes.

Set **`T2PW_ISOLATED_COLLECT_REPORT=<path>`** for a machine-readable per-file census. **Use it
rather than the rendered assertion**, which truncates and under-reports.

**The file is in no chunk and not in SMOKE**, deliberately — it is a repo-wide property, not a
subsystem's. It cannot enter Chunk D: `chunk_d_gate.py:62-70` hard-codes its file list, so
187 = 160 + 4 + 23 is structurally safe.

**Known export-only failure (F-089).** On a tree exported by `c045b_base_tree.py`, the sweep
reports **one** failure that is not a real defect: `tests/test_c030_canonical_identity_fallback.py:88`
shells out to `git ls-files` at **import** time, and the export's `PATHSPEC` excludes `.git`.
A base sweep reporting 22 rather than 21 is this, and only this.

**Pinned line count after this entry: 578** (was 541). D-061 requires the new value be
recorded here so the tripwire keeps working at its new value rather than being abandoned.


---

## C-102 / D-072 — the authorized Priority-4/5 baseline move, with its A/B

**Nothing in this move is a threshold change.** `min_core_coverage` stays `0.5` everywhere. What
moves is the **denominator** the acceptance instrument reports beside the raw one.

**The A/B ran offline, against committed `quarantine_report.json` artifacts.** Re-running the corpus
was forbidden this wave and no leg, cohort or paper leg was run.
`evidence/c102_f132_coverage_ab.py` measures the population;
`evidence/c102_g9_denominator_proof.py` is the base-vs-tip behavioural proof;
`evidence/c102_base_gate.py` measures the gates on `bcf9a23` by restoring the two changed modules to
their base blobs and setting the new test aside, then verifying `git diff` against base is empty for
`src` and `tests` while the base leg runs. That is used **instead of `c045b_base_tree.py`**, whose
pathspec excludes `runs_verify` — an exported base tree would have failed for want of data rather
than for want of code, and that is not a base result.

| Gate | base `bcf9a23` | tip | verdict |
|---|---|---|---|
| gold-readers (22 files) | **2 failed, 453 passed, 8 skipped** (exit 1) | **2 failed, 453 passed, 8 skipped** (exit 1) | identical; both reds are F-142 `[only_unrelated_reactions_survive]`, chartered as C-103. **No third failure.** |
| SMOKE (20 files) | — | **473 passed** | merge rule 10 |
| SMOKE (22 files) — **superseded by C-106**, the row above is C-102's measurement at its own tip and stays as it fell | — | **503 passed** | merge rule 10. `473 + 14 + 16`; see the C-106 entry below |
| focused `test_c102_coverage_denominator.py` | **does not collect — `ImportError`** | **14 passed** | **not the G9 proof.** A missing import is *symbol absence*, which G9 explicitly refuses. The behavioural proof is `evidence/c102_g9_denominator_proof.py`, in the row below |
| `c102_g9_denominator_proof.py` — which denominators does the report state for `PMC12782028/strict`? | **`[]`**, and no withheld term named | **`[23, 27]`**, all four named | **this is G9.** A statement about values in the report, not about a symbol |

**Corpus delta**, 62 legs with a coverage block, 860 requested-core terms drawn: **92 gold-forbidden
terms withheld across 47 legs**; **32 legs rise, 7 fall, 8 unchanged**; **zero legs clear** the
unchanged 0.500 minimum, and the six that were below it stay below it. `PMC12782028/strict` moves
`0.222 -> 0.261` and does **not** clear. **Priority 4 stays `0/8` and Priority 5 stays `0/2`** on
`runs_verify/2026-08-24_1428` — measured, not predicted, and reported as it fell.

**A leg that falls is the instrument working.** The exclusion is symmetric: a forbidden term the
pipeline matched is withheld from the numerator as well as the denominator, so it no longer earns
coverage credit. 26 of the 92 withheld terms are of that kind, and they are invisible to the
ORCH-702 probe, which counted forbidden terms only among the unmatched.

**Serialization note for anyone diffing reports — corrected, the first version was wrong.**
Only the **per-leg** key is conditional: `ModeResult.to_dict` omits `coverage_reconciliation` when
the leg stored no coverage block. The **report-level** key is not. `AcceptanceReport.to_dict` always
writes `coverage_reconciliation_corpus`, and priorities 4 and 5 always carry
`requested_core_coverage`, so **no report serializes byte-identically to before** — including one
with no coverage block anywhere. Measured (`evidence/c102_report_size.py` / `.log`):
`runs_verify/2026-08-04_1207` has **zero** such legs and still grows **48,773 → 49,681 bytes**.
Priorities 1-3 carry no `requested_core_coverage` key; only 4 and 5 do.

**Two names, because they are two different records (REV-102 F6).**
`coverage_reconciliation_corpus` at the top level is the aggregate and owns the per-leg `legs`
array; `coverage_reconciliation` inside a leg is that leg's own row. Their key sets are disjoint and
they briefly shared a name, which invited a reader who found one to assume the shape of the other.

**Size, and the choice behind it (REV-102 F5).** The corpus record is ~12 KB on a full run and
priorities 4 and 5 both referenced it, so it was serialized three times. The priority entries now
carry the **counts only** — how many legs, how many terms, which legs cleared, which are still below
the minimum, which have no defined rate — plus `legs_at`, naming the one key that holds the rows.
A priority read alone still states the size and outcome of the reconciliation; only the row-by-row
detail moved, one key away in the same document.

Every figure below is measured by one probe across all three shapes
(`evidence/c102_report_size.py`, whose `--as-if-uncompacted` swaps the summary property back to the
pre-F5 shape without editing a file), so the deltas are self-consistent:

| run | base `bcf9a23` | tip, pre-F5 | tip, shipped |
|---|---|---|---|
| `2026-08-24_1428` (10 legs with a coverage block) | 199,706 | 248,425 · **+24.4%** | **224,607 · +12.5%** |
| `2026-08-04_1207` (**zero** such legs) | 48,773 | 49,605 | **49,681 · +908 B** |

F5 removes **23,818 bytes, 49% of the growth**, on the full run: the two priority copies fall from
12,288 bytes each to 379. **It makes the small report 76 bytes larger**, because the summary adds
`legs_at` where the whole record had an empty `legs` array — stated rather than rounded away.

---

## C-106 — the authorized SMOKE baseline move, 473 -> 503, and why the gate grew

**Merge rule 4's escape hatch, used deliberately.** Nothing was removed from SMOKE. Two files
were ADDED, and the delta is exact and attributable.

| | files | tests | note |
|---|---|---|---|
| SMOKE before C-106 | 20 | **473** | A + B + C |
| `+ tests/test_c102_coverage_denominator.py` | +1 | **+14** | C-102's suite. Measured 3.87 s by the Lead, 3.49 s here |
| `+ tests/test_c106_mutation_harness_executable.py` | +1 | **+16** | new, structural only, **0.17 s** |
| **SMOKE after C-106** | **22** | **503** | `473 + 14 + 16 = 503`, measured, not inferred |

### Why the c102 file was added — this is the root cause F-151 actually had

`tests/test_c102_coverage_denominator.py` was **in no chunk, in no SMOKE and in no
gold-readers selection.** The integration tip had been RED there since `e77ad3d` and every
gate the sprint ran stayed green. Nobody saw it for a whole card, and the consequence was
not cosmetic: the harness at `evidence/c102_mutation_attack.py` asserts its unmutated
baseline is green before applying any mutation, so **the sprint's mutation-attack driver
was unrunnable** while D-078 and F-144 made mutation testing mandatory on every card.

Re-pinning the census without closing that hole would have left the next census drift
equally invisible. The file is now in the gate.

### The census pins the c102 file carries, and their cost

Four pins move together whenever a benchmark run is committed — not one, which is what
F-151, REV-104 and the wave handoff all said:

| Pin | was | now | attribution |
|---|---|---|---|
| `len(paths)` floor | `>= 62` | `>= 72` | floor; catches a shrink |
| test 10 `legs` | `== 62` | `== 72` | `runs_verify/2026-08-28_1816` (`e77ad3d`) contributes exactly 10 legs |
| test 10 `withheld` | `== 92` | `== 97` | the same run withholds 5 further terms |
| test 13 `checked` | `== 62` | `== 72` | same 10 legs |
| test 13 `with_matched_forbidden` | `== 23` | `== 26` | the same run contributes 3 matched-forbidden legs |

The other thirteen runs sum to exactly `62 / 92 / 23`, so the whole delta is one named run.
Measured in `evidence/c106_census_probe.log`; per-run table reproduced there.

**The derived pins stay `==` and must not be relaxed to `>=`.** The floor is `>=` because it
catches the corpus SHRINKING; the derived pins assert their loop VISITED every leg it should
have, and the census is how they know how many. `tests/test_c106_mutation_harness_executable.py`
tests 08 and 09 now enforce both halves of that, so the next drift is reported with an
actionable message instead of as an unexplained red.

**The cost is that the pins must be moved by hand every time a run is committed. That is the
correct cost** — moving them is the moment someone confirms the new legs were meant to be there.

### Line-address discipline

C-054's end-of-file note records that this file's line addresses are pinned by citation up to
`:477`. C-106's edits at `:239-242`, `:259`, `:271` and `:514-515` are therefore **line-neutral
by construction** — each rewrites or extends a line in place, none inserts or deletes one — and
this record is appended at end-of-file for the same reason C-054's was. `:213-218`, `:242-252`,
`:265` and `:268-273` all still address what they addressed.

---

## Reviewer-evidence reachability (C-109) — appended 2026-08-31, end-of-file by rule

**⚠ Why this sits at the end of the file and not in § 0, where a reader would look for it.**
The standing constraint pins this file's citations through **line 477**. § 0 is far above
that, so inserting there would shift every pinned address and break them — the same class of
defect F-154 registered. C-109 therefore made **zero edits at or above `:477`**: base and tip
have the identical 476 lines above the pin, `:477` is byte-identical, and the first differing
line in the whole file is below it. Proof: `evidence/c109_citation_probe.py` § 3 and
`evidence/c109_citation_probe.log`.

### G11-R — reviewer evidence must be reachable from integration before the merge

**New capability, added by C-109.** `evidence/reviewer_evidence_route.py`, acceptance test
`tests/test_c109_reviewer_evidence_route.py`.

**72 reviewer G11 reports and 94 probes were nearly lost this sprint**, because a reviewer's
evidence lived only in a worktree and the merge that accepted the review did not carry it.
Nothing detected it, because nothing was looking. This is the check.

| The obligation | Who it binds |
|---|---|
| **The orchestrator may not merge a reviewed card until this check passes for the reviewing task.** | Lead Orchestrator |
| **The orchestrator may not release or clean a reviewer worktree until this check passes for that worktree.** | Lead Orchestrator |
| **No worktree is ever pruned regardless.** That rule is unchanged and absolute — this gate is a *precondition* on release and cleaning, never a licence to prune. | everyone |

```bash
<venv-python> docs/pwml_recovery_sprint/evidence/reviewer_evidence_route.py \
  --task REV-1xx --worktree C:/t/rev1xx \
  --integration-repo <primary checkout> --integration-ref sprint/pwml-recovery
```

Exit codes: `0` all reachable · `1` **one or more items exist only in the worktree**, each
listed · `2` usage or infrastructure error · `3` **nothing was enumerated at all** — a
mistyped task id must not read as a clean gate, which is the F-154 silent-failure class
again. `--allow-empty` asserts, out loud, that a task genuinely produced no evidence.

**Reachability is decided BY CONTENT, never by filename.** Each worktree file is hashed to
its git blob id and that id is looked for in the integration ref's tree. Consequences, both
deliberate:

* **a same-named file with different bytes is NOT reachable** — reported as its own class,
  `unreachable_content_differs`, because it is the failure mode that looks green to a human
  eye and to any filename-based check;
* **byte-identical content under a different path IS reachable** — the bytes survived, which
  is what the sprint needs, and the report names where they were found.

**Scope discipline.** It answers exactly one question: *if this worktree vanished right now,
would the evidence still exist?* It is **not** a linter and **must not** judge whether the
evidence is good, whether a report says PASS, or whether a probe proves anything.

### Citing addresses in sprint documents — the F-154 rule

**Prefer an anchor that cannot drift: a heading, a unique string, a named table, a symbol, a
test name.** A line address is frozen by a pin the moment it is declared, *including a
mistake* — `.claude/agents/pwml-test-runner.md:59` cited `TEST_MATRIX.md:213-218` for the
chunk table and `:242-252` for the SMOKE block, and both had drifted +17 to +19 into content
containing **no test-file stems at all**, so the stem-exact match it ordered silently matched
nothing. **Correcting the numbers is not the fix; it drifts again.** The anchors now in force:

| Purpose | Anchor |
|---|---|
| chunk membership | the first markdown table under the **`## Chunks`** heading; rows `**A**` `**B**` `**C**` `**D-core**` `**D-apptest**` `**E**` |
| the SMOKE selection | the `bash` block under **`## Commands`** beginning `# SMOKE (every merge) — expect 503 passed` |
| Chunk D's file list | `evidence/chunk_d_gate.py`'s **`CORE`** / **`S8`** / **`QB`** / **`MONOLITHIC`** symbols |

**An anchor replaces a *locator*, not a *provenance pin*.** Keep the commit SHA or evidence
artifact that says *when a thing was measured*; replace only the *where to look*. **Where a
line address is genuinely unavoidable, state in the same breath how it is to be re-verified**
— by which symbol or string — so the next reader can tell a drifted number from a broken
claim. `controller.py`'s docstring is the worked example: it names three `streamlit_app.py`
line numbers and, immediately after, says to re-verify them by grepping `run_rag_loop` and
`run_rag_rounds`.

**Still carrying drifted addresses, out of C-109's boundary and routed rather than changed:**
`FINDINGS.md:1125-1126` (`chunk_d_gate.py:70` for Chunk D-qb, `TEST_MATRIX.md:218` for
Chunk E — Chunk E's row is at `:237`). C-109's boundary was `FINDINGS.md:1120-1124` only;
the two rows below it were left rather than silently exceeding a merge-gate boundary. The
same file's `:1129` is a **historical record** and correctly stays as it fell.

### Never cite a line number in a file that carries an uncommitted diff — the F-157 rule, C-112

**A protected uncommitted file is a citation hazard, not just a merge hazard.** F-153, the merged
`MASTER_PLAN.md` § 2 correction and the Lead's own C-109 charter all cited `streamlit_app.py:5669`
as the production `run_rag_rounds` call site. **The committed value is `:5636`**; the `+33` is
exactly the protected product-owner diff (35 insertions / 2 deletions, `sha256:47e4fafa789d359d…`),
so `:5669` was read off a **working copy** and **resolves for no reader** — it addresses bytes that
exist in **no commit**, and the number propagated through three documents before anyone noticed.

**The rule, in force from C-112:**

1. **Never cite a line number in `src/t2pw/app/streamlit_app.py`**, or in any other file the sprint
   deliberately holds uncommitted. Cite the **symbol** — `run_rag_rounds`, `run_rag_loop`,
   `freeze_canonical_payload` — which is identical in both trees and cannot drift by 33 lines.
2. **Verify every citation against the committed blob, never the working tree:**
   `git show <ref>:<path> | sed -n '<n>p'`. A claim checked against a working copy is a claim about
   bytes that exist in no commit. This is the whole lesson of F-157.
3. Where a line address is genuinely unavoidable, name **in the same breath** the symbol or unique
   string that re-verifies it, per the F-154 rule above.

### C-112 closure notes — appended, because the records they concern are left as they fell

* **The "Still carrying drifted addresses" paragraph above is C-109's routing record and is left
  byte-identical.** Both items it routed are now closed: `FINDINGS.md` row **E** was converted to
  the `## Chunks` anchor, and row **D-qb**'s `chunk_d_gate.py:70` was **verified correct and
  deliberately left** — REV-109 checked both rows and C-109's routing was half-wrong in its own
  disfavour. "Fixing" a correct citation would have repeated the error in reverse.
* **`TEST_MATRIX.md:568`'s *"§ Chunks begins at line 209"*** was **struck in place** and replaced
  with the `## Chunks` heading anchor; the measured value at C-112's tip is `:228`. It was not
  silently restated. **`DECISIONS.md:3619` repeats the same `209` and was LEFT UNTOUCHED** —
  `DECISIONS.md` is append-only and read-only to an implementer, and only the product owner may
  amend it.
* **`TEST_MATRIX.md:726-727`**, C-106's signed record that those addresses *"all still address what
  they addressed"*, **is false and was LEFT UNTOUCHED.** It is another card's signed record;
  editing one is exactly what this sprint refuses. It is annotated here instead.
* **The 26 stale citations of REV-109 R1 were bucketed before any one was touched.** **Seven are
  live** and were converted to anchors — never renumbered, because a corrected number drifts again
  on the next insertion and that is the entire finding. **Nineteen are frozen historical records
  and every one was left**: six in append-only `DECISIONS.md`; four in signed artifacts and another
  card's charter (`evidence/c054_gate_counts.json`, `evidence/c106_predictions.md`,
  `evidence/c109_citation_probe.log`, `prompts/C-109.md`); three in `SPIKE-002-REPORT.md`'s signed
  `CONTROL-PLANE-RECONCILE-001` annotations; two in the dispatched `C-011` charter; one in
  `FINDINGS.md`'s probe-antecedent narrative; and three named frozen by the C-112 charter itself.
* **C-112 created no new line drift.** Every citation edit is **line-neutral, rewritten in place**;
  the only growth in any file is this end-of-file block, which nothing cites by number.

### Same class, found while doing the above — RAISED, not fixed

* **Three of the 26 addresses were already wrong before C-109 inserted anything.** The drift log's
  `+16` arithmetic is mechanical: it assumes the base content was the intended target.
  `MASTER_PLAN.md:336` (F-031) addressed § 7's *heading*; `:363` (F-032) addressed **TRAP-5**; and
  `:372` — cited **five times**, by `DECISIONS.md:1919`, `MASTER_PLAN.md:477` and
  `SPIKE-002-REPORT.md` `:138 :143 :253` — addressed the § 8 **Schedule** table, never the `C-045`
  § 9 row it is offered for. The two live ones were anchored to the rows they were always meant to
  name; the four frozen ones were left. **A drift measurement that starts from a wrong address
  reports a shift, not a resolution.**
* **`FINDINGS.md` F-032's remaining action is already discharged.** § 9's `C-035` row now carries
  the `parse_span_relation` / `validate_evidence_span` carve-out and the `C-061` row records the
  correction, so *"the §9 row still needs fixing before C-061 is dispatched"* is stale in
  substance. **Reported, not edited** — the address column was C-112's to anchor; the finding's
  substance is not C-112's to rewrite.
* **`FINDINGS.md` F-031's own row cites `streamlit_app.py:454` and `:645`** — two more line
  addresses inside the protected uncommitted file, the exact class rule 1 above forbids. The same
  is true of F-153's `:1270` and `:1426`, which F-157 measured as correct in **both** trees.
  **Reported, not edited**: only `:5669` was chartered, and only `:5669` was changed.
* **The citing *site* of one of the 26 has itself drifted.** REV-109 measured the F-154 row at
  `RESUME-NEXT-SESSION.md:94`; at C-112's base it is `:189`. A citation *of* a citation is no more
  stable than the citation.
* **`evidence/c107_mutation_attack.py` is run by no gate.** SMOKE's
  `test_c106_mutation_harness_executable.py` targets `c102_mutation_attack.py` **only**, which is
  why M16's `ABORT` (exit 3) survived a green 503 from C-108's merge until C-112 repaired it.
  **Whether the c107 harness should also be gated is RAISED here, not answered** — it is a
  `TEST_MATRIX` change with its own cost and it is not C-112's to make.

### C-112 G9 labelling — one of the seven live citations is HARDENING, not correction

**Measured, and recorded because mislabelling it would be a reject.** Six of the seven live
citations C-112 converted were **false at the C-112 base SHA** and are corrections of currently-false
documentation, each proved false-at-base / true-at-tip against **committed blobs** in
`evidence/c112_citation_proof.log`. The seventh is not:

* **`.claude/agents/pwml-test-runner.md:52`**, cited from `TEST_MATRIX.md:533`, **still resolved to
  the SMOKE bullet at the base SHA.** REV-109 counted it among the 26 under its *other* class —
  *"base content rewritten, no verbatim match at tip"* (the bullet's own figure moved `465 → 503`) —
  **not** under *"shift +N"*. The address was therefore **drift-prone, not false**, and converting it
  to the `## Test discipline` SMOKE-bullet anchor is **hardening**. **No base failure is claimed for
  it.** The first version of C-112's own probe asserted that it *was* false at base and **failed**;
  that run is preserved at `evidence/c112_citation_proof.attempt2-ptr52-not-false-at-base.log`
  beside its correction, rather than being dropped.

**The general point, for the next card that inherits a drift list:** a mechanical drift log mixes two
classes — *the number now points somewhere else* and *the number still points here but the content
under it changed*. Both are worth converting to anchors. **Only the first is a false-at-base
correction.** Counting them together is what produced this card's one mislabel.
