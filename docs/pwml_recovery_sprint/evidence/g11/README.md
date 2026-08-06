# G11 durable cleanup evidence

Every test, benchmark, pipeline leg and LLM-backed job in this sprint runs through
`../bounded_run.py`. **Its `--json` cleanup report is committed here.** A G11 claim is
then checkable against an artifact instead of a pasted table.

Why this directory exists: G11 has always required a cleanup report on every test record
(`TEST_MATRIX.md` § 0, `[S8]` item 9), but every agent wrote its `--json` into a session
scratchpad that is deleted afterwards. The closeout review could show the ledger's
universal G11 claim was contradicted elsewhere, yet **could not establish from the
repository what actually happened on any individual job.** That is what a committed
report fixes.

**Prospective only.** Nothing here is reconstructed. Reports exist for jobs run after this
directory was introduced, and for no others. A report synthesised after the fact is not
evidence of the original run, and the standing statement that the pre-A0 jobs cannot be
reconstructed remains true. **Never** hand-write, edit or backfill a file in this tree.

---

## Location and naming

```
docs/pwml_recovery_sprint/evidence/g11/<TASK-ID>/<SEQ>-<label>.json
                                       H-004/     03-smoke.json
```

| Part | Rule |
|---|---|
| `<TASK-ID>` | the branch-register task ID exactly: `H-004`, `C-056a`, `INIT-001`, `T-100` |
| `<SEQ>` | zero-padded, `>= 2` digits, allocated per task, **never reused** |
| `<label>` | the `--label` passed to the wrapper; `[a-z0-9][a-z0-9._-]*` |

Allocate the path — do not hand-write it:

```bash
<py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py next --task H-004 --label smoke
```

### Why two jobs cannot collide

1. The directory is the task ID, and one task owns exactly one branch.
2. Inside it the sequence is `max(existing) + 1`, **and** the file is created with
   `O_CREAT|O_EXCL`. Two allocations — concurrent, same task, same label — cannot resolve
   to the same path; the loser sees `FileExistsError` and takes the next sequence.
3. Sequences are never reused, so a **re-run cannot overwrite the attempt it replaces**.
   An infrastructure failure stays on the record. (`H-004/02-smoke.json` is a real
   example: a mis-invoked smoke run, kept; `03-smoke.json` is the retry.)
4. No clock is involved, so a caller that cannot obtain a timestamp is not blocked.

`next` reserves the name by writing a `{"g11_reserved": true}` placeholder, which the
wrapper's `--json` then overwrites. **A placeholder that survives is a job that produced
no report** — `check` fails it as `report_never_written` and it must never be committed.

---

## Required minimum content

Specified against what `bounded_run.CleanupReport` actually emits (`bounded_run.py:285`,
written at `:777`), not against a wish. `g11_evidence.py selftest` regenerates a report
from a live wrapped command and asserts this exact set, so the specification cannot drift
away from the tool.

| Required | Field(s) in the artifact |
|---|---|
| command / job identity | `label`, `command`, `cwd`, `root_pid`, `isolation` |
| start + completion classification | `started_at`, `finished_at`, `exit_reason` |
| the actual exit code | `exit_code` (the child's real code), `returned_code` |
| observed / terminated owned processes | `descendants_observed`, `descendants_terminated` |
| final surviving owned-process count | `final_surviving_count`, `survivors`, `cleanup_success` |
| runtime | `duration_seconds`, `timeout_seconds` |
| report delivery + free-text record | `json_report_path`, `json_report_written`, `json_report_error`, `notes` |

`exit_reason` must be one of `completed`, `nonzero`, `timeout`, `cancelled`,
`infrastructure_failure`. The dataclass default `unknown` means `run()` never classified
the exit, so it is not a valid record.

**Two fields the objective asks for and the artifact does not carry.** Adding either
requires editing `bounded_run.py`, which H-004 does not own, so both are stated here as
requirements for a follow-up task rather than implemented:

* **report-schema version** — `CleanupReport` has none. This specification is versioned
  instead (`G11_SPEC_VERSION`), and the required field set is the structural contract.
* **wrapper build identity** — `command` names the *child*, never the wrapper, and there
  is no build/SHA field. `[S8]`'s self-reference clause wants each result to state which
  wrapper build produced it. Partial mitigation only: the report is committed on the
  branch that ran it, so `git log -1 -- <report>` gives the commit and
  `git show <commit>:docs/pwml_recovery_sprint/evidence/bounded_run.py` gives the wrapper
  build present in that tree. That is an inference from the tree, **not proof from the
  artifact** — a stale wrapper run before the commit would not be visible. A
  `wrapper_build` field in `CleanupReport` is the real fix.

A compliant report may describe a **failed job**. G11 is about process lifecycle; the
test outcome is a separate question, read from `exit_reason` / `exit_code`.

---

## The five compliance rules `check` enforces

```bash
<py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py check            # everything
<py> ... check --task H-004                                                 # one task
```

1. **The artifact must exist and parse**, independently of any exit code. `check` never
   consults an exit status. This matters concretely: when `--json` is unwritable the
   wrapper deliberately returns the *child's* real exit code, so an exit-code-based check
   would certify a job that wrote no report at all.
2. **`cleanup_success` must be `true`.** `final_surviving_count == 0` alone is *not*
   sufficient — that field keeps its `0` default when verification never ran, so a report
   can read 0 survivors **and** `cleanup_success: false` while a child is still alive.
   All three of `cleanup_success`, `final_surviving_count == 0` and an empty `survivors`
   list are required.
3. **The schema is validated**: every required field present, correctly typed, with a
   valid `exit_reason`.
4. **`json_report_written: false` is not a violation for a direct `run()` caller.**
   `emit_json_report` runs only from `main()`, so an in-process caller such as
   `baseline_suite.py:127` legitimately shows `json_report_path: ""` and
   `json_report_written: false`; those fields are *not applicable* when `--json` was never
   requested. A report that names a `json_report_path` but reports it unwritten **is** a
   violation.
5. **Only credential-free, bounded evidence is committed** — see below.

`check` is a compliance checker, not a permission to skip anything. It cannot declare a
job compliant without a report, and it weakens no G11 rule.

---

## Credential rule — check before you commit

A command line can carry an API key, a token or a connection string, and `command` is
copied verbatim into the artifact. Before committing, run `check`: it scans the whole
report text for OpenAI-style keys, GitHub tokens, Google API keys, AWS key IDs, bearer
tokens, inline `key=`/`token=`/`password=` assignments, and `scheme://user:pass@` URLs.

**If a report contains a credential:** do not commit it, do not edit the report to remove
the secret (an edited report is not evidence), **treat the credential as leaked** — rotate
it — and re-run the job with the secret supplied through the environment instead of the
command line. Report the incident to the orchestrator. Automated scanning is a backstop:
you are still responsible for reading the `command` field.

## Size rule — a record, not a log dump

Reports are structured JSON of ~1.5–3 KB. `check` rejects anything over 64 KiB and rejects
any non-`.json` file in a task directory. **Captured child stdout never enters version
control**; `.gitignore` excludes `*.log` / `*.out` / `*.out.txt` under this tree as a
backstop. `bounded_run.py` already deletes its own temporary child log (`:768`). Never
commit a cache, a `--basetemp` tree or a benchmark output directory here.

---

## Reviewer path: "this branch claims G11" → "here are its reports"

```bash
git diff --name-only <BASE_SHA>..<branch> -- docs/pwml_recovery_sprint/evidence/g11/
<py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py check <each path>
```

One report per job the branch ran. A branch that claims a test, benchmark, pipeline leg or
LLM-backed job and adds no report for it has not evidenced G11, and no agent needs to be
asked anything.

## Traps

* **`bounded_run_selftest.py` overwrites its per-case reports.** `REPORT_DIR` defaults to a
  single fixed temp path, so consecutive runs clobber each other. Set
  `BOUNDED_RUN_SELFTEST_REPORTS` to a **fresh directory per run**.
* Pytest's `--basetemp` parent must already exist; pytest does not create intermediate
  directories, and every test errors in setup if it is missing.
* `next` and `check` spawn no child process and are not tests, benchmarks, pipeline legs or
  LLM-backed commands, so they are outside the four job classes `[S8]` item 1 names. That
  is scope, not an exemption. `selftest` **is** a test and runs under the wrapper.

## Selftest

```bash
<py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py selftest   # or: pytest <that file>
```

Run it under the wrapper, with `--json` pointing at an allocated path, like any other test.
It proves the required field set is present in real wrapper output, that the naming scheme
cannot collide, and that each compliance rule rejects or accepts the artifact it is meant
to.
