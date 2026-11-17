# CLAUDE.md

Project instructions for T2PW (Text-to-PathWhiz) live in **`AGENT_INSTRUCTIONS.md`**
(pipeline architecture, key source files, stage contracts, conventions). Read that first
for anything about how the pipeline itself works.

---

## Active track: T2PW Web Server (NAR Web Server Issue)

Goal: a stand-alone, public T2PW web server, live and working for the NAR Web Server
Issue letter of intent (**Dec 15, 2026**), and tested ready for the paper (**~Feb 15, 2027**).

Integration branch: **`webserver/refactor`**, cut from `main`. Task branches are
`ws/<TASK-ID>-<slug>` and merge into `webserver/refactor`. Only the product owner merges
`webserver/refactor` into `main`.

**Read before doing any web-server work:**

| Document | Why |
|---|---|
| [`docs/webserver/PLAN.md`](docs/webserver/PLAN.md) | Scope, architecture, the refactor steps (R1–R6), the web workstreams (W1–W9), timeline |
| [`docs/webserver/DECISIONS.md`](docs/webserver/DECISIONS.md) | Product decisions already made. Product owner only; never append yourself |
| [`docs/webserver/LEDGER.md`](docs/webserver/LEDGER.md) | Live task state: dispatched, in review, merged, blocked. Single source of truth |
| [`docs/webserver/ORCHESTRATOR.md`](docs/webserver/ORCHESTRATOR.md) | How the orchestrator runs the track, including model and effort routing |
| [`docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md`](docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md) | Still the definition of biological correctness. Outranks any test or benchmark |

### Roles

The main session is the **Orchestrator**. It plans, dispatches, routes reviews and merges.
**It writes no code and never approves its own work.** The only files it edits are
`docs/webserver/LEDGER.md` and task RESULT docs.

Subagents (`.claude/agents/`):

| Agent | Does | Default model / effort |
|---|---|---|
| `ws-scout` | Read-only code mapping: find symbols, callers, coupling. Returns short summaries | haiku / low |
| `ws-implementer` | One task, one branch, isolated worktree, inside a declared file boundary | sonnet / medium |
| `ws-ui-builder` | Front-end templates, CSS and JS for the public site | sonnet / high |
| `ws-reviewer` | Independent review of the actual diff. No edit tools | opus / high |
| `ws-test-runner` | Runs tests through the bounded wrapper, reports numbers. Fixes nothing | haiku / medium |

The orchestrator **overrides model and effort per task** using the routing table in
`docs/webserver/ORCHESTRATOR.md` § 3.

### Merge gates (web-server track)

No task branch merges into `webserver/refactor` unless **all** of these hold:

1. Its dependencies are merged.
2. Its diff stays inside the task's declared file boundary.
3. Its focused tests pass, run through the bounded wrapper with **zero surviving processes**.
4. The SMOKE suite (**511** tests, ~45 s, command in `docs/pwml_recovery_sprint/TEST_MATRIX.md`
   "Commands") passes on the merge result.
5. **Refactor tasks (R-series) change no behaviour.** The golden-seam tests pass unchanged:
   `test_batch_driver_seam_golden.py`, `test_c011_freeze_seam_golden_equivalence.py`,
   `test_streamlit_stage8_export_contract.py`, `test_streamlit_quarantine_boundary.py`.
   From R3 on, the app-vs-`run_pathway` equivalence test also passes. A test that had to be
   edited to pass is a reject unless the reviewer confirms the edit is only an import path.
6. **No biology changes.** Logic under `src/t2pw/pipeline/`, `pwml/`, `mapping/`, `rag/`,
   `curation/` and `llm/prompts/` is not changed by this track. Moving code out of
   `streamlit_app.py` is allowed; changing what it does is not. If a web feature needs
   a biology change, stop and record it as an open question for the product owner.
7. An independent `ws-reviewer` approved the **actual diff**, not the implementer's report.
8. A RESULT doc exists (`docs/webserver/results/<TASK-ID>.md`) and a LEDGER row was appended.

### Standing rules (carried over from the PWML recovery sprint, still binding)

- **Every test, benchmark and pipeline command runs through the bounded wrapper**
  `docs/pwml_recovery_sprint/evidence/bounded_run.py`. No detached processes, no `nohup`.
  Cleanup targets only PIDs the job created; `taskkill /IM python.exe` and `pkill python`
  are forbidden. One heavy job at a time; never `pytest -n auto`.
  Full policy: `docs/pwml_recovery_sprint/TEST_MATRIX.md` § 0.
- **`--basetemp=<dir>` on every pytest run, and create its parent directory first.**
  Without it, 83 tests error and you will report a false regression. Never run the full
  suite unchunked (~16 GB).
- **`T2PW_OFFLINE_CURATOR=1`** on any deterministic AppTest run (see TEST_MATRIX "Commands").
- **Never commit a cache change.** `data/enrichment_cache.json` is 39 MB and tracked.
- **Never commit secrets.** `.env` is gitignored and holds the OpenRouter key and the
  PathBank MySQL credentials. The server gets its own read-only DB user.
- **Do not rebuild what exists.** `docs/pwml_recovery_sprint/MASTER_PLAN.md` § 2 lists
  what is already implemented: fetch, RAG, PDF parsing, PWML export, rendering, batch runner.
- Identical runs give different Stage-1 results at temperature 0. Re-run before calling
  a single run a regression.

### Logging

Work that only exists in a chat transcript is lost. Before a task is reported done:

| what | where |
|---|---|
| the task record, including what failed and dead ends | `docs/webserver/results/<TASK-ID>.md` |
| live state | one appended row in `docs/webserver/LEDGER.md` |
| test reports, proofs, verdicts | `docs/webserver/evidence/<TASK-ID>/`, committed, never pasted |
| an open product question | the RESULT doc, under **Open questions**, with options and costs |

Docs-only commits are separate from code commits. Check `git status --porcelain` afterwards.

---

## Archived: PWML Recovery Sprint (closed 2026-11-05)

The sprint ran on `sprint/pwml-recovery` and is fully merged into `main` (`06a0b8c0`).
Its documents stay in `docs/pwml_recovery_sprint/` as reference: `PRODUCT_CONTRACT.md`
(still binding for biology), `TEST_MATRIX.md` (still binding for test discipline),
`FINDINGS.md`, `DECISIONS.md` and `LEDGER.md` (history). Its `pwml-*` subagents remain
available for biology work, which is outside the web-server track.

Left open at close: `ORCH-804` (`orch804/ir-empty-state`) is **held, not merged** after a
reviewer REJECT. Its record is on that branch (`ORCH-804-RESULT.md` § 9) and its last
`LEDGER.md` row is on `main`.
