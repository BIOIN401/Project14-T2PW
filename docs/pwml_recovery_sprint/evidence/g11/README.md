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

Specified against what `bounded_run.CleanupReport` actually emits (the `CleanupReport`
dataclass, `bounded_run.py:507-508`, written out by `emit_json_report` at `:1016`), not
against a wish. `g11_evidence.py selftest` regenerates a report
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

A compliant report may describe a **failed job**. G11 is about process lifecycle; the
test outcome is a separate question, read from `exit_reason` / `exit_code`.

---

## Schema version and wrapper build identity

H-004 recorded both as **missing** and could not add them: that needed `bounded_run.py`,
which it did not own. **H-006 added them.** Every report a post-H-006 wrapper writes now
carries two more fields — and `check` does **not** require either (see "Which artifacts
carry it").

| Field | |
|---|---|
| `schema_version` | version of the *report contract* — the set, types and meanings of the fields. Bump discipline is stated at `bounded_run.REPORT_SCHEMA_VERSION`: **bump** when a consumer validating against version N could misread an N+1 report (a field removed, renamed, retyped, re-meant, or its units changed); **do not bump** for an *added* field, for a change in cleanup/termination behaviour, or for a new value a field's documented meaning already permits. |
| `wrapper_build` | which wrapper build wrote the record: `digest` (SHA-256 over the raw bytes of the **executing** module), `digest_algorithm`, `digest_scope`, `path`, `size_bytes`, `digest_error`, plus repository context `repo_root`, `repo_head`, `repo_source`, `wrapper_vs_head`, `repo_tracked_files_dirty`, `repo_error`. |

**`repo_head` is context, never the identity.** It is recorded *in addition*. The
executing wrapper may differ from HEAD (`wrapper_vs_head: "modified:' M'"`) or live
outside any repository (`repo_source: not_a_repository`); in both cases the SHA names
bytes other than the ones that ran, and only the digest names the bytes that ran. Not
hypothetical: H-006's own evidence was necessarily produced while its wrapper was still
uncommitted, and `H-006/*.json` say exactly that. Repository facts are resolved from the
**wrapper's own directory**, never the caller's cwd, which may be a different checkout.

### The archaeology substitute, and its three failure modes

Before H-006 a report could only be attributed to a wrapper by inference from the tree:
`git log -1 -- <report>` for the commit, then
`git show <commit>:docs/pwml_recovery_sprint/evidence/bounded_run.py` for the wrapper in
it. That is an inference from the tree, **not proof from the artifact**, and it fails
three ways — all three defeated by a digest of the executing module:

1. **Cross-checkout execution.** An agent holding a main checkout *and* a worktree can run
   one tree's wrapper while committing on the other's branch; the archaeology then names a
   file that never executed. The digest is taken from the module that ran, wherever it
   lived. `bounded_run_selftest.py` case 11 proves this by running a modified copy from
   **outside** the repository and checking the recorded identity is the copy's.
2. **Rebase / squash.** History rewriting moves the commit `git log -1` resolves to, and
   the "wrapper in that tree" moves with it. A content digest is not a commit reference.
3. **A stale wrapper**, run before the commit that carries it, hashes to its own stale
   content, which cannot match the wrapper in the commit.

### Which artifacts carry it

* **`H-006/*.json` and every task after it** carry `schema_version` and `wrapper_build`.
* **`H-004/*.json` and `H-005/*.json` never will.** They were produced by a wrapper build
  that had no such field. They are *schema 0*: still valid, still compliant, and **not**
  to be edited or regenerated to acquire the fields. The rule that forbids backfilling a
  report forbids backfilling a field into one — a reconstruction is not evidence of the
  original run. Their wrapper build is, and remains, **unproven from the artifact**; the
  weak archaeology above is the only attribution they will ever have.

### Byte identity: what the digest hashes, and what it does not

`wrapper_build.digest` is SHA-256 over the **raw bytes of the executing module**, read
from `os.path.abspath(__file__)`. It is deliberately sensitive to **every** byte-level
difference, including CRLF/LF line-ending transformations. That sensitivity is the point:
two checkouts whose line endings differ are genuinely running different bytes, and the
field is meant to say so.

**Consequence an auditor must know before concluding anything.** This repository has
`core.autocrlf=true` and **no `.gitattributes`**. The checked-out file that actually
executes therefore contains **CRLF**, while the committed Git blob contains **LF**. The
two hash differently, and both hashes are correct answers to different questions:

| What you hash | For `bounded_run.py` at `4afcc6d` | Answers |
|---|---|---|
| the **executing worktree bytes** (CRLF) | `sha256:69f9f1b5…aad5`, 46 712 B | *which bytes ran* — this is what `wrapper_build.digest` records |
| the **committed Git blob** (LF, normalized) | `sha256:ffd5b424…fd98`, 45 620 B | *what the repository stores* |

So hashing `git show <commit>:<path>` **may correctly produce a different value, and that
by itself does not disprove the artifact.** Comparing the normalized blob hash against a
recorded raw-byte digest is a category error. Compare like with like, or expect a
mismatch on any Windows checkout.

**`repo_head` remains contextual metadata, never a substitute for the digest.** A
repository SHA cannot identify bytes that were never committed.

### Verifying a recorded digest

Cross-platform, read-only. Nothing below writes to the repository or touches protected
state.

**1 — hash the candidate executing file as raw bytes** (no text mode, no newline
translation), and print both the digest and the byte length:

```
python -c "import hashlib,pathlib,sys; b=pathlib.Path(sys.argv[1]).read_bytes(); print(len(b),hashlib.sha256(b).hexdigest())" "<path-to-bounded_run.py>"
```

Compare **both** numbers against the artifact's `wrapper_build.digest` (strip the
`sha256:` prefix) and `wrapper_build.size_bytes`. Read them with:

```
python -c "import json,sys; d=json.load(open(sys.argv[1]))['wrapper_build']; print(d['size_bytes'], d['digest'], d['path'])" <artifact.json>
```

A match on digest **and** size means the file you hashed is byte-identical to the module
that produced the artifact. `path` tells you where that module lived at run time — useful
when the same build exists in several checkouts.

**2 — separately, hash the committed blob** when you want to compare worktree bytes
against repository content. This is a *different* question from step 1 and will differ
whenever line endings are normalized:

```
python -c "import hashlib,subprocess,sys; p=subprocess.run(['git','cat-file','blob',sys.argv[1]],capture_output=True,check=True); b=p.stdout; print(len(b),hashlib.sha256(b).hexdigest())" "<rev>:<path>"
```

**Python** does the reading in both steps: step 1 reads the worktree file as raw bytes, and
step 2 has Python invoke Git and capture Git's stdout as raw bytes. **No Git byte stream
passes through a shell pipeline, `>`, `Out-File`, or any other text-decoding layer** — which
is what avoids Windows PowerShell 5.1 re-encoding native output; `check=True` makes step 2
fail closed if `git cat-file` fails. A difference here is expected under `core.autocrlf=true`
and is **not** evidence against the artifact: the repository stores a normalized form.

**What this can and cannot establish.** The digest is a **fingerprint of the executed
bytes**, nothing more. It lets you confirm that a wrapper file you already hold is or is
not the one that ran. It **cannot reconstruct** the executing module: if the wrapper was
dirty or lived outside any repository and those bytes no longer exist anywhere, the digest
proves only that they differed from whatever you can still obtain — it will not give them
back, and no repository SHA will either. An artifact whose wrapper bytes are gone is
attributable only to the extent that some surviving copy hashes to the recorded value.

`check` deliberately does not require the two fields: requiring them would make every
pre-H-006 artifact non-compliant overnight, which is precisely the pressure to backfill.
`bounded_run.validate_report_schema()` validates them when present and treats their
absence as valid, and selftest case 12 asserts every committed pre-H-006 report still
passes `check`, unmodified.

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
backstop. `bounded_run.py` already deletes its own temporary child log (the guarded
`os.unlink(log_path)` at `bounded_run.py:1007-1010`). Never
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
