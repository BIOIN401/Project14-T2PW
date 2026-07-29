# The overnight batch runner

## 1. What it is

The overnight runner takes N papers fetched from the literature and pushes every
one of them through the pipeline **twice** — once in `Strict PWML` and once in
`Research (relaxed)` — unattended, one paper+mode at a time, writing every
artifact plus a ranked fix-list into `runs/TIMESTAMP/`.

It does this by driving the *real* Streamlit app (`src/t2pw/app/streamlit_app.py`)
through Streamlit's own headless harness, `streamlit.testing.v1.AppTest`: it sets
the same widgets a human sets and reads the same `st.session_state` the app
writes, so the app is never imported, refactored or duplicated and any change to
it is picked up automatically on the next run.

Code: `src/t2pw/batch/{fetch,driver,runner,report}.py`, CLI
`scripts/batch_run.py`, launcher `run_overnight.bat`, work list `topics.txt`.

---

## 2. How to run it

### 2a. Double-click `run_overnight.bat`

This is the intended way. It `cd`s to the repo root, checks that
`.venv\Scripts\python.exe` exists (exit code **2** with a message if not), runs
`scripts\batch_run.py`, prints whether everything passed, and **pauses** so the
window does not vanish with the result in it. Any arguments given to the `.bat`
are forwarded verbatim:

```bat
run_overnight.bat
run_overnight.bat --fresh
run_overnight.bat --limit 5 --deadline 8
```

### 2b. The explicit command

```bash
.venv/Scripts/python.exe scripts/batch_run.py
```

Every flag, verified against `build_parser()` in `scripts/batch_run.py`:

| Flag | Metavar | Default | Meaning |
| --- | --- | --- | --- |
| `--topics` | `PATH` | `None` → `topics.txt` in the project root | the work list |
| `--limit` | `N` | `None` (no cap) | cap the **total** papers fetched across all topics |
| `--modes` | `LIST` | `None` → both | comma-separated: `strict`, `research` |
| `--out` | `DIR` | `None` → `runs/`, resolved against the **project root**, not the shell's cwd | where run directories live |
| `--timeout` | `SECONDS` | `3600` (`runner.DEFAULT_PAPER_TIMEOUT`) | wall-clock ceiling for ONE paper+mode; past it the child process **tree** is killed and the pair is recorded as a timeout |
| `--deadline` | `HOURS` | `10` (`runner.DEFAULT_DEADLINE_HOURS`) | whole-night ceiling; once it passes no further pair is started, the summary is written, and the run stays resumable |
| `--status` | — | off | print one progress line for the current/most recent run and exit |
| `--fresh` | — | off | start a new run directory instead of continuing an incomplete one |

There is a second argument group, `internal (used by the parent, not by humans)`:
`--single`, `--run-dir DIR`, `--slug SLUG`, `--mode MODE`. That is how the parent
re-invokes this same script for exactly one pair — do not call it by hand unless
you are debugging one paper.

A fuller invocation, all defaults made explicit:

```bash
.venv/Scripts/python.exe scripts/batch_run.py \
  --topics topics.txt --limit 10 --modes strict,research \
  --out runs --timeout 2700 --deadline 8
```

### 2c. Checking on it from a second terminal

```bash
.venv/Scripts/python.exe scripts/batch_run.py --status
```

This reads the **newest** run directory (it does not need the running process),
prints one `status_line` — e.g.
`2026-07-27_2210  4/10 done | strict 3 pass 1 fail | research 4 pass | 1 RESEARCH DEFECT | running: 12 pair(s) pending`
— then the path to `SUMMARY.txt`. It always exits 0, so it is safe to poll.

`Ctrl+C` in the running terminal is a clean stop: the manifest was flushed after
every pair, the child is killed with it, and rerunning the same command continues
where it stopped.

### 2d. Exit codes

`0` — every attempted pair passed (`ALL GREEN`), **or** every pair passed but at
least one produced no deliverable (`PASSED WITH WARNINGS`).
`1` — any failure, a reached `--deadline` with pairs still pending, a `Ctrl+C`,
"nothing ran at all", or an unexpected error in the batch loop.
`2` — a usage error (argparse exits 2 on an unknown flag or a bad value), or —
from `run_overnight.bat` only — no virtualenv at `.venv\Scripts\python.exe`.
`3` — **preflight failed.** The interpreter you launched cannot import what the
child processes need (`streamlit.testing.v1` and `t2pw.rag.research_report` are
the two that matter — `driver.py` defers both, so nothing the parent does
touches them). Nothing was fetched, nothing ran, and **no run directory was
created**: there is no new `SUMMARY.txt`, and the newest folder in `runs/` is
still the previous night's. The console message names the module, the
interpreter in use, the project venv, and the exact command to rerun with.

The `3` exists because `2` is already taken twice over, and because anything
reading `ERRORLEVEL` has to be able to tell *a paper failed* from *the
environment is wrong*. On 2026-07-27 it could not: run `runs/2026-07-27_2135`
was launched under an interpreter without `streamlit`, fetched 28 papers in 43s,
then recorded all 56 paper+mode legs as `failure_kind=crash` in 24 seconds,
every row carrying the same `ModuleNotFoundError: No module named 'streamlit'`.
`SUMMARY.txt` opened with `!! RESEARCH-MODE DEFECT !! papers affected: 28` — 28
pipeline defects that did not exist. The usual cause on Windows is that `.py` is
associated with `C:\WINDOWS\py.exe`, which ignores an active virtualenv, so
`scripts\batch_run.py` and `.venv\Scripts\python.exe scripts\batch_run.py` are
two different interpreters and only one of them has the dependencies.

`run_overnight.bat` prints its own message for `3` ("STOPPED BEFORE STARTING ...
no run folder was created") instead of the generic "read `SUMMARY.txt` in the
newest `runs\` folder", which would otherwise point at the previous night.

---

## 3. `topics.txt`

One line per topic. Blank lines and everything after a `#` are ignored, and
**malformed lines are skipped rather than raised on** — this is hand-edited input
to an unattended job. Accepted shapes (`parse_topics` in `fetch.py`):

```
topic | organism | count      # take `count` papers for this topic
topic | organism              # count defaults to 3 (DEFAULT_PER_TOPIC)
topic                         # no organism scoping, count 3
PMC4412817                    # pinned paper: fetched by id, no search
10.1038/nchembio.1687         # a pinned DOI works too
```

A one-field line is read as a **pinned paper id** when it looks like
`PMC<digits>`, a bare PubMed id of 6+ digits, or anything starting with `10.`
(every real DOI carries that registrant prefix). Anything else on a one-field
line is a topic.

Worked example — the shipped `topics.txt`:

```
lipid A biosynthesis | Escherichia coli | 3
menaquinone biosynthesis | Bacillus subtilis | 3
mycolic acid biosynthesis | Mycobacterium tuberculosis | 2

# Uncomment to pin a specific paper you want in every run:
# PMC4412817
```

That asks for 3 + 3 + 2 = 8 papers. It will usually yield fewer. Only
open-access papers with retrievable full text can be run; the rest land in the
run's `skipped.json` with `reason: "no_full_text"`, which is the **normal** case,
not an error. `fetch_papers` therefore over-fetches (`max(want * 3, want + 2)`
candidates per topic) and never raises — every per-paper problem becomes a skip
record with a reason: `no_full_text`, `duplicate`, `no_candidates`,
`search_failed`, or `fetch_failed`. Ask for a few more papers than you need and
cap the night with `--limit`.

Fetching is not new code: it drives Stage R1 (`t2pw.rag.acquire`), which already
owns the literature endpoints, the retrying HTTP client and the on-disk cache.

---

## 4. The run directory, and what to read first

```
runs/2026-07-27_2210/            # minute resolution; sorts lexicographically in time order
├── SUMMARY.txt                  # ← READ THIS FIRST
├── failures_by_code.txt         # the ranked fix-list, worst blast radius first
├── manifest.jsonl               # one JSON row per (paper, mode); this IS the resume state
├── plan.json                    # the work list: modes, limit, one record per paper
├── skipped.json                 # candidates that were fetched but not runnable, with reasons
├── batch.log                    # one timestamped line per event, opened/closed per line
├── cache_snapshot/              # id_mapping_cache.json, enrichment_cache.json as they were
└── papers/
    └── PMC4412817__lipid-a-biosynthesis-in-e-coli/
        ├── 00_PAPER.txt         # id, title, year, organism, uri, topic, query, text size
        ├── 01_source_text.txt   # the EXACT string handed to the app — paste it into the browser to reproduce
        ├── strict/
        │   ├── RESULT.txt       # PASS / FAIL in plain words, with the traceback
        │   ├── pathway.pwml     # the strict deliverable
        │   ├── pwml_ir.json, pwml_ir_report.json, pwml_ir_validation.json,
        │   ├── pwml_validation_report.json, pwml_qa.json,
        │   ├── pwml_required_field_gate_report.json, final_mapped.json
        │   └── stage1_payload.json, merged_payload.json, contract_reports.json,
        │       gate_fail_report.json, final_stage3_gate_report.json
        └── research/
            ├── RESULT.txt
            ├── research_pathway_report.txt   # the research deliverable
            ├── research_pathway_citations.json
            ├── research_pathway_elements.csv
            ├── review_flags.json             # biology flags, skipped FORMAT rules, content preserved
            └── stage1_payload.json, merged_payload.json, contract_reports.json, ...
```

**Read `SUMMARY.txt` first.** It is plain text, ≤ 100 columns, Notepad-safe over a
remote session, and it is regenerated from the manifest after *every* pair, so a
half-finished run is fully readable at 3am. Its four sections, in order:

1. **Header counts** — papers attempted, strict pass/fail, research pass/fail,
   timeouts, skipped, `pass, no deliv.`, failed artifact writes, manifest rows
   (and how many were unreadable).
2. **Triage matrix** — research-mode defects first and loudest
   (`!! RESEARCH-MODE DEFECT !!`), then passes that produced nothing
   (`!! PASSED BUT PRODUCED NO DELIVERABLE !!`), then the class tallies.
3. **Per-paper breakdown** — worst class first, every artifact named with its
   size, and `!! WRITE FAILED` against any artifact that never reached the disk.
4. **What to fix first** — the three ordered buckets of §5.

Then `failures_by_code.txt` if you want the fix order by issue code, ranked by
how many *distinct papers* hit each code (a code blocking six papers outranks one
blocking a single paper six times). Then the individual `RESULT.txt` for the pair
you decided to attack. `manifest.jsonl` is machine-shaped; you should not need it.

Folder names are `PAPERID__short-title`, ≤ 60 characters, with Windows-illegal
characters and control characters replaced, whitespace collapsed to `-`, and a
6-hex-digit content hash appended on a collision.

---

## 5. The triage matrix — why the whole thing exists

Running each paper twice is not redundancy. The **pair** of outcomes is the
diagnostic, and it is what makes morning triage cheap:

| strict | research | class | what it means |
| --- | --- | --- | --- |
| PASS | PASS | `ok` | nothing to do |
| **FAIL** | PASS | `format-blocked` | a PathWhiz **FORMAT** rule stopped strict export while the relaxed run still produced a candidate. **Expected wear, not a bug** — catalogue it. Grouped by issue code so you fix one rule at a time. |
| **FAIL** | **FAIL** | `broken` | a **real bug**: the failure is upstream of export — a crash, an LLM refusal, a network fault, or biology the extractor cannot read. Also counts as a research defect, so it appears in that section too. |
| PASS | **FAIL** | `research-defect` | the pure case, and the loudest |
| — | — | `incomplete` | a mode is missing, skipped or unrecognised |

The rule that carries the most weight: **any research failure is a code defect.**
Research mode is *fail-open by construction* — it relaxes FORMAT rules and
records review flags instead of stopping — so it has no legitimate way to fail on
real data. If it failed, the code is wrong, not the paper. That is why
`research-defect` and `broken` are printed first, why `status_line` shouts
`N RESEARCH DEFECT`, and why "What to fix first" is ordered:

1. **RESEARCH-MODE DEFECTS** — research mode must never fail. Fix first.
2. **BROKEN** — both modes failed: crash / LLM / network / biology.
3. **FORMAT-BLOCKED** — strict only, grouped by issue code (research is fine).

One more state the pass/fail columns cannot express, so it gets its own shout:
a research run that passed but produced **no** report, no citations and no tiers
because RAG never triggered. That is not a failure (the pipeline ran; RAG
declining to trigger is not a code defect), but it is not a clean night either.
It is recorded as a warning that travels into the manifest, printed as
`PASS (no research deliverable)`, counted separately as `pass, no deliv.`, and it
downgrades the final line from `ALL GREEN` to `PASSED WITH WARNINGS`.

Failures are classified into one `failure_kind` — `contract`, `llm`, `network`,
`ambiguous_review_scope`, `no_reactions`, `crash`, `timeout`, `unknown` — with
structured evidence beating wording and wording beating the mere presence of a
traceback (the app calls `st.exception(exc)` in its generic handler, so
`at.exception` being non-empty says nothing about *what* broke).

Reading the app's verdict is genuinely subtle and worth knowing when you doubt a
row: the app reports almost every problem with `st.error(...)` + `st.stop()`,
never by raising, and `st.stop()` is a *clean* stop to AppTest. A failed run
therefore looks identical to a passing one unless you inspect the emitted
elements — so the driver checks `at.error` / `at.exception` /
`session_state["pipeline_error"]` **before** it looks for artifacts, and refuses
to call anything a pass it cannot positively confirm. Strict mode also needs a
**third** click: `pwml_generate_btn` only runs audit + DB mapping, and PWML stays
gated behind the review step until `refinement_generate_pwml` is pressed.

---

## 6. Operational notes

**It is strictly sequential, one pair at a time, and must stay that way.**
`data/id_mapping_cache.json`, `data/enrichment_cache.json` and the RAG index
under `data/rag_index` are shared **mutable** state with read-modify-write access
patterns and no locking. Two concurrent runs interleave writes and corrupt them.
The wall-clock cost of being sequential is the price of a run whose caches are
still usable in the morning. Do not "optimise" the loop into a pool.

**Caches are snapshotted so a bad night is revertible.** Before the first paper
of a *fresh* run, `id_mapping_cache.json` and `enrichment_cache.json` are copied
to `cache_snapshot/`; copy them back over `data/` to undo the night's cache
writes. Note honestly what this does *not* cover: the RAG index under
`data/rag_index` is **not** snapshotted (it is a directory, not a file, and
`SNAPSHOT_FILES` lists only those two JSON caches), and a *resumed* run does not
re-snapshot — the snapshot belongs to the night the directory was created.

**Every paper+mode runs in a child process, and that is not optional.** A Python
thread cannot be killed (there is no `Thread.kill`, and a thread blocked in a
socket `recv` will never look at a stop flag), and `signal.SIGALRM` — the usual
in-process watchdog — does not exist on Windows. A hung LLM request, a wedged
PathBank MySQL socket or a Streamlit rerun that never returns is therefore
unkillable in-process. On Windows the child is started in its own process group
and killed with `taskkill /F /T` so any worker it spawned dies with it; on POSIX
this degrades to `killpg`.

**The LLM endpoint must stay up and the machine must not sleep.** The runner has
no way to pause and wait for either. A dead endpoint produces a night of
`failure_kind=llm` / `network` rows; a sleeping machine produces a night of
timeouts. Disable sleep and hibernation for the duration, and keep the model
server running. Every environment variable the parent has — API keys, the
PathBank DSN — is *copied* into the child's environment (plus forced
`PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`), so a missing `.env` fails the child
for a completely unrelated-looking reason.

**Two clocks bound the night.**
- Per pair: `--timeout` (default 3600 s), enforced by the **parent**, which kills
  the child tree past it and records a `timeout` row. The child is given a
  slightly smaller internal budget (`--timeout` minus a 120 s grace) so it has
  room to write its artifacts and print its row instead of being killed one
  second before finishing. Inside the child, one *interaction* may take up to
  `DEFAULT_APP_TIMEOUT` = 3600 s and the whole pair shares the budget the parent
  handed it.
- Whole night: `--deadline` (default 10 h). Ten papers × two modes × one hour is
  twenty hours, which is not "overnight" — it is still running when you need the
  machine. The check happens *before* each pair starts, so a pair already in
  flight may overrun the deadline by up to `--timeout`. On reaching it the loop
  stops cleanly, the summary is written, the exit code is 1, and the run stays
  resumable so the remainder finishes the next night.

**Everything is written by the child, at the moment it happens.** Artifacts and
`RESULT.txt` are written in the child (so a crash after a successful pipeline run
still leaves the artifacts on disk), and `manifest.jsonl` is appended, flushed and
`fsync`ed by the parent after every pair. A "pass" whose required artifact
(`pathway.pwml` for strict, `research_pathway_report.txt` for research) was
produced but could not be *written* is downgraded to a failure — the driver hands
over bytes and only the write knows whether they landed.

**Everything on disk is UTF-8.** Entity names really do contain non-cp1252 bytes
(`β-hydroxymyristoyl` is the canonical lipid A intermediate) and journal titles
carry Greek letters and em dashes. Console output degrades to
`backslashreplace`-escaped ASCII rather than failing.

---

## 7. Known limitations

**`AppTest.run(timeout=...)` does not abort a blocking call.** This is the most
important thing to know about the runner and the reason the child process exists.
Streamlit's harness requests a stop and then **joins the script thread
unbounded**: it raises the timeout error only *after* the blocked call finally
returns on its own. It is a report, not an interruption.

Measured on this repo: `at.run(timeout=2.0)` against a script blocking for 12
seconds **raised the timeout but returned after 12.1 s** — 10 seconds past the
deadline it had just reported.

Consequences, stated plainly:

- A wedged LLM or DB socket inside an `at.run()` is bounded **only** by the
  parent's process-tree kill at `--timeout`. The driver's own `_Budget` /
  `app_timeout` numbers bound *cooperative* runs and the gaps between
  interactions; they cannot bound a blocked syscall.
- On the process-tree-kill path the child is killed **before it can write its
  artifacts** or print its result row. That pair therefore yields a synthesized
  `timeout` row and an empty mode folder — the partial work is genuinely gone,
  not merely unreported. (If the child *did* manage to print its row a moment
  before the stopwatch expired, the parent uses the child's own row rather than
  the fabricated timeout — that case is logged explicitly.)
- So `--timeout` is a real ceiling and the driver's timeouts are best-effort.
  Size `--timeout` for the slowest paper you are willing to wait for, not for the
  average one.

Other limitations worth knowing:

- **Only the newest run directory is ever resumed** (see §8). An older
  incomplete run is never picked up again; its directory is left on disk as an
  archive and its pending pairs are simply never run.
- **`--modes` on a resumed run rewrites `plan.json`.** Narrowing modes mid-run
  changes the work list rather than the manifest, so previously completed pairs
  in the dropped mode stay in the manifest and in `SUMMARY.txt`.
- **Only the two JSON caches are snapshotted**, not the RAG index (§6).
- **`AppTest` cannot drive `st.file_uploader`**, so input mode is always
  `Paste text`. A PDF-only ingestion path is not exercised by the batch.
- **Group-by-paper is last-row-wins.** A retried pair supersedes the attempt it
  retried in `SUMMARY.txt`; the history stays in `manifest.jsonl`.
- **A truncated final manifest line is terminated, not recovered.** A hard reboot
  mid-write costs you that one row (the pair is re-run on resume); the damage is
  confined to the line the reboot broke rather than fusing two rows.

---

## 8. Resume semantics

**Resume is automatic and needs no flag.** Rerun the same command — or
double-click the `.bat` again — and the run continues. `find_resumable` decides,
and it prints one line explaining its choice into `batch.log`.

The rules, in the order they are applied:

1. **Only the NEWEST run directory is considered.** Reaching back past it was a
   trap: an aborted run from weeks ago still owes work, so it beat last night's
   *complete* run, and every night from then on re-ran one stale paper, fetched no
   new papers, and exited — forever. If the newest run is finished, the right
   answer is a fresh run, not an older incomplete one.
2. **It must have a readable `plan.json` listing papers.** Without the work list
   there is nothing to continue, so a fresh run starts instead.
3. **It must still owe pairs.** Pending = `plan.json`'s (paper × mode) pairs minus
   the pairs already present in `manifest.jsonl`. The manifest — not a marker
   file — is the resume state, because it is flushed as each pair finishes and
   therefore cannot disagree with reality.
4. **It must be younger than 24 hours** (`RESUME_MAX_AGE_HOURS`). "Resume" is
   meant for last night's interrupted batch, not for archaeology. Age is measured
   from the *newest* of three signals — the timestamp in the directory name, the
   directory's mtime, and `manifest.jsonl`'s mtime — because any one of them can
   lie (a directory restored from backup has a fresh mtime and an ancient name; a
   run made with a wrong clock has the reverse). A directory is only called stale
   when nothing about it looks recent.

If any check fails, a **fresh** run directory is created and the papers are
fetched again. `--fresh` forces that unconditionally.

A resumed run **never re-fetches and never re-runs finished work**, because the
paper text lives in the run directory (`01_source_text.txt`) — that file is also
exactly what you paste into the browser to reproduce a failure by hand.

**"Finished" includes "failed".** `completed_pairs` builds its done-set from *any*
manifest row and never looks at `status`, so a leg that failed is as done as a leg
that passed and a resume will not revisit it. This is deliberate — a paper that
fails deterministically (a clinical case report with no pathway in it; a review the
extractor cannot get a `processes` object out of) would otherwise be retried every
single relaunch and eat the night twice — but it has a sharp edge worth knowing
before you rely on a resume to validate a fix. `runs/2026-07-28_0919` ended with
five manifest rows, three of them `fail`; relaunching it starts at leg 6 with 51
pending and never touches those three again. If the point of tonight is to see
whether a fix works on the legs that broke, a resume is the wrong instrument: use
`--fresh`, or delete the failed rows from `manifest.jsonl` first (it is one JSON
object per line and nothing else reads it).

Two safety nets around the seams: if a crash lands after the directory is created
but before `plan.json` is written, a minimal aborted `plan.json` is written so the
directory is not an orphan that every future launch refuses and replaces with
another empty one; and `plan.json` is saved *before* any paper title is logged,
because logging arbitrary journal text is the riskiest thing in setup.
