# BR - Driver bridge

- **Branch:** `ws/BR-driver-bridge`, from `webserver/refactor` @ `63647b24`.
- **Code commits:** `100070d8` (bridge, worker default, test), `1f241684` (test tightened).
  This docs commit follows them.
- **Status:** done. New capability (PO ruling 2026-11-05, SCHEDULE F1), so the proof is a
  labelled new acceptance test, not a base-failing test.

## What was built

`src/t2pw/service/driver_bridge.py` (274 lines). `run_pathway(PathwayRequest) -> PathwayResult`
turns the request into the one "paper" `batch.driver.run_one` takes, calls the driver once,
writes the driver's in-memory artifacts into `request.out_dir`, and maps the `RunOutcome`
onto `PathwayResult`. Nothing in the pipeline, the driver, `service/pathway.py` or the app
was edited.

`src/t2pw/web/worker.py`: `DEFAULT_RUN_FN = "t2pw.service.driver_bridge:run_pathway"` (was
the W0 stub), plus the matching docstring line. The stub stays selectable with
`--run-fn t2pw.service.pathway:run_pathway` or `Worker(run_fn=...)`. No web test asserted
the old default string, so no web test was changed. The tests that build a `Worker` with the
default only call `start()`/`close()`, never `run_one()`, so none of them now starts the
real driver.

Reused, not rebuilt: `batch.runner.write_artifacts` / `failed_writes` / `required_artifacts`
(the batch's own artifact writer and its "required artifact lost" rule),
`batch.runner.force_utf8_stdio`, `pipeline.deadline.child_deadline_seconds` with the batch's
120 s grace, `driver._extraction_focus`, W1 `resolve_name` / `resolve_pmids` / `read_pdfs`.

## Several articles per pathway (requirement 2)

What the app does (`streamlit_app.py`, the "Run pipeline" handler):

```
text_parts = [entry.strip() for entry in text_entries if entry.strip()]
...  for each uploaded PDF:  if _pdf["text"].strip(): text_parts.append(_pdf["text"].strip())
text = "\n\n".join(text_parts)
```

After that line only `text` is used. `text_entries`, `text_entry_count`, `input_mode`,
`uses_pdf_input` and `uploaded_pdfs` are not read again (grep, all hits are above the join).
The driver pastes its paper text into `pathway_text_0` with one text box. The app then
computes `text = joined.strip()`, and because every part is already stripped and non-empty,
`joined.strip() == joined`. So **the driver can feed exactly what the app would see, with no
edit**, and the bridge mirrors the app: every stripped, non-empty part, pasted texts first,
joined with `"\n\n"`. Order across kinds: texts, then PMIDs, then PDFs (the app has no PMID
input; PMID texts are treated as pasted text, so they go before PDFs). A test guards the
three app lines by reading the app source, so a change to the app's rule fails the bridge test.

PDFs are read through W1 `read_pdfs` with `parse_fn=partial(parse_pdf, enable_ocr_fallback=False)`.
That is the app's default (OCR checkbox off, references skipped, all pages). `parse_pdf`'s own
default is OCR on, so plain `read_pdfs` would NOT have matched the app.

Name mode (a pathway name and no sources) has no app counterpart. The bridge runs it as one
batch topic leg, `name | organism | 1`: W1 `resolve_name(..., count=1)`, and the one
`BatchPaper` is passed to `run_one` unchanged, exactly as the batch passes a planned paper.
The app's multi-paper RAG then runs inside that leg as usual (D-W03). See Q-BR-2.

## Field mapping

### `PathwayRequest` -> driver

| request field | how | if it cannot be honoured |
|---|---|---|
| `texts` | stripped, empty ones skipped (as the app), joined into `paper["full_text"]` | all empty and nothing else: `fail` / `no_reactions` |
| `pmids` | W1 `resolve_pmids`, Europe PMC text with `use_cache=False` (W1 Q3: a failed download is otherwise cached as empty), joined | `lookup_failed` -> `error` / `network` (W4 retries); `no_pmcid` / `no_full_text` -> `fail` / `no_reactions`; `invalid` -> `error` / `contract`. W1's message is the result message. The job fails; the other PMIDs are not run without it |
| `pdf_paths` | W1 `read_pdfs`, app defaults (OCR off), joined | `empty` -> `fail` / `no_reactions`; `error` (unreadable, not a PDF, parser error) -> `error` / `contract`. Same as the app, which stops on a PDF error; stricter than the app for an empty PDF, which the app skips silently |
| `scope.requested_pathway` | `paper["requested_pathway"]` (source mode) or the fetched `BatchPaper` (name mode). The driver derives the extraction focus and Stage-0 reconciliation from it | - |
| `scope.requested_organism` | `paper["requested_organism"]`; reaches the app through the focus box (`"<pathway> in <organism>"`) | organism with no pathway name: `error` / `contract` "add the pathway name". The driver pins a paper with no pathway name, and a pinned paper's organism reaches the pipeline nowhere (no focus, no reconciliation). See Q-BR-1 |
| `scope.pinned` | the driver derives `pinned = not requested_pathway` | `pinned=True` with a pathway name: `error` / `contract` |
| `extraction_focus` | accepted only when equal to the focus the driver derives (`driver._extraction_focus`) | anything else: `error` / `contract`. The driver types its own focus |
| `export_mode` | passed to `run_one` | not exactly `strict` / `research`: `error` / `contract`. `driver._resolve_mode` would silently turn any other string into strict |
| `out_dir` | every artifact and `run_outcome.json` are written here | `None`: a fresh `tempfile.mkdtemp(prefix="t2pw-job-")` |
| `deadline` | `run_one(timeout=child_deadline_seconds(deadline - now, grace=120))`, computed after any fetch. `None`: the driver's own default (7200 s) | already passed: `timeout` / `timeout` without a run (W4 retries) |

All bridge refusals have `stage="init"` and one issue code: `bridge_unsupported_request`,
`bridge_input_unavailable` or `bridge_deadline_passed`. `driver.run_one` is never called for a
refused request (every refusal test asserts this).

When several inputs fail, one kind is reported, the most final first: `contract`, then
`no_reactions`, then `network`. Retrying cannot fix a permanent problem, so a retryable kind
never wins. The messages of all inputs of that kind are joined.

### `RunOutcome` -> `PathwayResult`

| RunOutcome | PathwayResult | note |
|---|---|---|
| `status`, `stage`, `failure_kind`, `message` | same fields, verbatim | the vocabularies are equal (W0 test) |
| `warnings`, `issue_codes`, `counts` | same fields, copied | the bridge may append warnings (below) |
| `release_status` (a `ReleaseStatus` from the gate-failure / scope-conflict paths) | `release_status`, the same object | - |
| `release_status` (the boundary's frozen dict on the strict PASS path) | `release_status` rebuilt as `ReleaseStatus`; `to_dict()` equals the dict for every field the type has | the boundary's three extra keys (`applied`, `review_reasons`, `retrieval_attempts_source`) are kept in `attempt_log[0]["release_status"]` and `run_outcome.json`. An unreadable record becomes `None` plus a warning, never a guessed record |
| `pwml_artifact` + `artifacts[pwml_artifact]` | `pwml_path = out_dir/<name>` | only when this run wrote it; both PWML names are never `report_paths` keys |
| `artifacts` (all other names) | `report_paths`, `{name: out_dir/name}` | only names whose write succeeded. A failed write of a required artifact (the PWML, the research report) makes the result `fail` / `crash`, as `batch.runner.run_single` does; any other failed write is a warning |
| `paper_id`, `mode`, `seconds`, `detail`, `scope_conflicts`, `observed_context`, `requested_scope`, `termination_reason`, `termination_is_operational` | no slot in the W0 interface | the driver's whole manifest row (`RunOutcome.to_dict()`) goes to `out_dir/run_outcome.json` (with `detail`), and the same row without `detail` goes to `attempt_log[0]` (with `"attempt": 1, "runner": "t2pw.batch.driver.run_one"`). `run_outcome.json` is NOT a `report_paths` key, because those are public downloads and `detail` can hold a traceback |
| - | `image_path` | always `None`. The driver produces no image (W0 F5) |

Checked against the consumers, through `jobs.result_to_dict`:

- **W4 `retry.decide`** reads `status`, `failure_kind`, `release_status is None`, `pwml_path`.
  Tested: strict pass with PWML -> final; gate failure carrying a `ReleaseStatus` -> final;
  timeout, crash, and `error` with kind `""` -> retry; `fail` / `unknown` -> final; every
  bridge refusal -> final except `network` and the passed deadline (retry).
- **W7 `curator_queue.qualify`** reads `status == pass`, `release_status["status"] == release_ready`,
  `pwml_path` is a file. Tested: a release-ready strict pass qualifies; a review-required
  PWML (`pathway.review_required.pwml`) does not.
- **The worker child** (`worker._child_main` with `DEFAULT_RUN_FN`, in-process, driver
  stubbed) writes a `result.json` with the PWML path under the job's `out_dir`.

## Per-job isolation (requirement 3)

- The result is built only from this call's in-memory `RunOutcome`: artifacts are `bytes`/`str`
  and the PWML is the exporter's `xml_bytes` from session state. The bridge never reads
  `outputs/` or `tmp/`. `pwml_path` and `report_paths` name only files this call wrote, so a
  stale file already in `out_dir` is never reported (tested with a stale `pathway.pwml` and
  `final_mapped.json`).
- Filenames pass through `write_artifacts`, which keeps only `Path(name).name`, so an
  artifact key like `../escape.json` lands inside `out_dir` (tested).
- The worker always sets `out_dir = <work_root>/item-<id>/attempt-<n>/out`, and W6 serves
  only files under that directory.
- **What this relies on.** The app itself still writes the fixed `PROJECT_ROOT/outputs/` and
  `tmp/`, and clears `outputs/` at the start of a run (`reset_quarantine_state`). The driver
  reads the quarantine report files back from paths this run's session state names. So two
  pipeline runs at the same time **in the same checkout** could clobber each other's working
  files. The bridge relies on one pipeline run per checkout at a time: one worker (D-W05,
  the worker's per-database lock) and no batch run or Streamlit session running pipeline
  jobs on the server's checkout. The batch has always run on the same rule. See Q-BR-3.

## Imports (requirement 4)

The module imports only `t2pw.service.pathway` and `t2pw.pipeline.release_status` at module
level. `t2pw.batch.driver`, `t2pw.batch.runner`, `t2pw.service.inputs`, `t2pw.rag.acquire`
and `t2pw.pipeline.deadline` are imported inside the functions. A fresh-process test asserts
that importing the bridge loads no `streamlit*`, no `t2pw.app*` and no `t2pw.batch*`. No
cycle: nothing imports the bridge except the worker, by name, in its child.

## Tests

`tests/test_service_driver_bridge.py`, 34 tests, labelled NEW ACCEPTANCE TEST (BR). `run_one`
and the W1 adapters are monkeypatched: no AppTest, no LLM, no network, no DB.

All runs: repo venv interpreter for the wrapper and the child, `PYTHONPATH=C:/t/ws-BR/src`,
`T2PW_OFFLINE_CURATOR=1`, `pinned_pytest --expect-tree C:/t/ws-BR`, fresh `--basetemp`,
foreground, no heavy lock (no AppTest or golden file in either chunk).

| run | selection | result | tree | survivors | evidence |
|---|---|---|---|---|---|
| 01 focused | the new file + the 12 web-layer files (baseline 261) | **295 passed** in 48.1 s (261 + 34) | clean @ `1f241684` | 0 | `docs/webserver/evidence/BR/01-focused.json`, `.log`, `pin/01-focused.pin.json` |
| 02 batch runner | `tests/test_batch_run.py` (uses `run_fn` fakes; no AppTest) | **65 passed** in 11.0 s | clean @ `1f241684` | 0 | `docs/webserver/evidence/BR/02-batch-run.*`, `pin/02-batch-run.pin.json` |

Not run: `tests/test_batch_driver.py` and the other `run_one` tests drive real AppTest
scripts, so they are not light; my diff does not touch the driver. SMOKE: not run, per the
brief (the independent test-runner owns it). No live LLM job (the orchestrator schedules one).

**Mutation probes** (each on a deliberately broken working tree, then restored with
`git checkout`; reports in `docs/webserver/evidence/BR/mutation/`):

| mutation | failing test(s) |
|---|---|
| `APP_JOIN = "\n"` | `test_several_articles_are_joined_exactly_as_the_app_joins_them`, `test_app_still_joins_text_parts_that_way` |
| OCR fallback on | `test_several_articles_are_joined...` |
| `NAME_MODE_COUNT = 3` | `test_name_mode_runs_one_fetched_paper_unchanged` |
| network ranked before contract | `test_a_permanent_input_problem_outranks_a_retryable_one` |
| `pwml_path=None` | 4 tests (child end-to-end, strict pass, review required, isolation) |

## What failed, and dead ends

- **First test run: 1 failure, a real bridge bug.** PMID and PDF texts were joined
  unstripped. W1 already strips them, so it would not have shown in production, but the
  app strips every part, so the bridge now does too (`parts.append(r.text.strip())`).
- **Two mutation runs were no-ops.** The `APP_JOIN` mutation did not apply through `sed`, then
  through `python -c`, because of `\n` escaping. Both runs show 34 passed with no diff. The
  third attempt (made with the Edit tool) applied and failed 2 tests. Its report overwrote
  the no-op under the same label.
- **A stray `python -` (stdin) was launched by mistake** in a shell one-liner (`python -
  2>/dev/null; sed ...`) while editing `worker.py`. It hung on stdin and was moved to the
  background by the harness. I stopped it with TaskStop, then checked the process list: no
  `python -` process survived (only VS Code LSP servers and another worktree's
  `waitlock.py`, which I did not touch). The `sed` edits in that line never ran; `worker.py`
  was then edited with the Edit tool.
- **Considered and rejected: an environment variable `T2PW_RUN_FN`.** `--run-fn` already
  selects the run function, from the CLI and from a systemd `ExecStart`. A second switch adds
  nothing for A7.
- **Considered and rejected: a bridge-level file lock on `PROJECT_ROOT`.** It would stop only
  a second bridge process. The batch and the Streamlit app, which share the same fixed
  directories, would not take it. See Q-BR-3.
- **Considered and rejected: a new failure kind for "invalid request".** The W0 vocabulary
  is frozen and tested equal to the driver's. `contract` is the existing kind W4 treats as
  final. See Q-BR-4.

## Open questions

**Q-BR-1. PMID / PDF jobs with an organism and no pathway name are refused.** The web form
requires an organism in every mode but a pathway name only in name mode (`web/app.py`). The
driver uses the organism only together with a pathway name, so the bridge refuses such a job
("add the pathway name and submit again") rather than drop the organism silently.

| option | cost |
|---|---|
| A (recommended) | W6: require the pathway name in all three modes. Small form and validation edit. The organism then reaches the run exactly as in a batch topic leg (focus box + Stage-0 reconciliation) |
| B | Bridge: run anyway, with a warning that the organism was not used. One-line change. The result is never shown as clean, but the user's organism has no effect |
| C | Bridge: type `"in <organism>"` alone into the focus box. Not a batch-measured shape, so it changes what reaches the pipeline relative to every benchmark. Needs a PO ruling (no-regression rule) |

**Q-BR-2. Name mode runs one fetched paper (`count=1`).** This is exactly one batch leg.

| option | cost |
|---|---|
| A (current) | `count=1`, then the app's own multi-paper RAG. No new rule. Uses only the first eligible open-access paper as the seed |
| B | `count=N`, one driver leg per paper, then pick one PWML. N times the LLM cost and wall clock (a 60 min attempt cap fits about 1 leg), and needs a selection rule nobody has defined |
| C | `count=N`, papers concatenated with the app's rule. The app's rule, but never benchmarked on fetched reviews. Longer input, more chunks, different results |

**Q-BR-3. One pipeline run per checkout.** The isolation above holds only while nothing else
runs the pipeline in the server's checkout. Options: (A) state it in the W9 runbook (no
batch or Streamlit session on the server checkout), at zero cost; (B) a lock taken by the
bridge, the batch runner and the app, which edits `batch/` and `streamlit_app.py`; (C) give
each job its own `PROJECT_ROOT`, which is R5's `out_dir` threading (Q-W0-1 A). Recommended: A
now, C through R5 and A7.

**Q-BR-4. Kind for an unsupported request.** The bridge uses `error` / `contract`. W4 treats it
as final, which is the point. But the W6 failure text for `contract` reads "did not pass the
quality checks ... A different or richer source may help", which is wrong for "add a pathway
name". The specific message is still shown above it. Options: (A) keep it, at zero cost;
(B) W6 keys its text on the issue code `bridge_unsupported_request`, a small views edit;
(C) a new kind, which is a W0 vocabulary change plus a driver edit.

**Q-BR-5. UTF-8 mode parity with the batch.** Batch legs ran with `PYTHONUTF8=1` and
`PYTHONIOENCODING=utf-8` (`runner.child_env`). The worker's child runs with the worker's own
environment. The bridge forces UTF-8 on stdout/stderr (`force_utf8_stdio`), but UTF-8 *mode*
(the default encoding of `open()` without `encoding=`) is fixed at interpreter start
(Python 3.13 here), so it cannot be set from inside the bridge. Options: (A) W9 sets
`PYTHONUTF8=1` in the worker's systemd unit, which the child inherits, at zero code cost
(recommended); (B) the worker passes `child_env()` to the child, a `worker.py` edit outside
this boundary; (C) accept the difference. Whether it matters is not measured. The
orchestrator's live end-to-end run is the first check.

**Q-BR-6. PMID full text with `use_cache=False`.** This follows the W1 Q3 advice: a failed
download is no longer cached as empty. But `fetch_full_text` still swallows errors into `""`,
so a Europe PMC outage still reads as "no open-access full text" (`no_reactions`, final, not
retried). Fixing that needs an `acquire.py` change, which is outside the web-server boundary.
