R5 prep, measured at `2e3d6a28` in C:/t/ws-R3d-post. Read-only, nothing run. Paths are under `src/t2pw/` unless noted.

**Headline**
- `run_pathway` is still the W0 stub (`service/pathway.py:145-153`). The live path is the BR bridge `service/driver_bridge.py:277`, which calls `batch/driver.run_one` (AppTest). R5 replaces that body.
- `run_extraction` (`service/extraction.py:1022`) covers input assembly through Stage 2 and takes no deadline and no `out_dir`.
- `run_post_pipeline` and `run_export` do not exist. Only the old `run_post_pipeline_sbml_artifacts` (`service/postpipeline.py:489`) and `run_pwml_export` (:1749) exist, until R4 lands.
- Seven service failure kinds have no `PathwayResult` mapping at all. They are `InputFailure(kind, message)` and carry no status, stage or `KIND_*`.
- No product question blocks R5. Q-W0-1 (`out_dir`) is the closest; I could not confirm it was ruled.

**1. Contract**
`PathwayRequest` (frozen, `pathway.py:56-78`). Who reads each:
| field | read by | state |
|---|---|---|
| `texts` | bridge `_paper` (`driver_bridge.py:172`), `jobs.request_to_dict` (`web/jobs.py:101`) | filled |
| `pdf_paths` | bridge :191, `jobs.py:102` | filled |
| `pmids` | bridge :175 | filled |
| `scope` (`RequestedScope`) | bridge `_unsupported` (:104) and `_paper`; `views._title` (`web/views.py:167`) | filled |
| `extraction_focus` | bridge :116, refused unless equal to the driver-derived value (`driver._extraction_focus`, `batch/driver.py:542`) | custom focus unsupported |
| `export_mode` | bridge :105, :302 | filled |
| `out_dir` | bridge :280; worker overwrites it with `<attempt>/out` (`web/worker.py:265`) | filled |
| `deadline` | bridge :290-297 | filled by worker |

`PathwayResult` (`pathway.py:81-142`). Who fills it today and who reads it:
| field | filled by bridge | read by |
|---|---|---|
| `status`, `stage`, `failure_kind`, `message` | yes (:250) | `retry.decide`, `views.py:134-163` |
| `warnings`, `issue_codes`, `counts` | yes (:251-252) | `views.py:151` (`issue_codes`) |
| `release_status` | yes (:254, from the `RunOutcome` dict) | `retry.py:89`, `curator_queue.py:91`, `views.py:134` |
| `pwml_path` | yes (:253) | `retry.py:93`, `views.py:305` |
| `image_path` | **never set**: no assignment in the bridge, and `service/postpipeline.py` never mentions `image_path` | `views.py:272,308` (a download kind with nothing behind it) |
| `report_paths` | yes (:253) | `views.py:275,310`, `curator_queue.py:175` |
| `attempt_log` | one in-run row (:273) | `views.py` attempt history |

Unfilled: `image_path`. The bridge also does not hold the whole `ReleaseStatus` (extra keys are dropped, :217-219).

**2. Today's live path**
- `Worker.run_one` (`web/worker.py:253`) calls `store.claim_next(timeout_s, wall_cap_s)`, which sets the absolute deadline at dequeue (`web/worker.py:6`, :258). It writes `request.json` with `deadline` and `out_dir` (:265), then starts `python -m t2pw.web.worker --child <run_fn>` in a contained child (:268).
- The child imports `DEFAULT_RUN_FN = "t2pw.service.driver_bridge:run_pathway"` (:70, :332-333) and writes `result_to_dict` (:337). The worker kills the tree at `deadline` (:276) and records timeout/`KIND_TIMEOUT` if no result exists (:291-293). Per-attempt cap is 90 min (`web/retry.py:42`).
- Bridge: `_unsupported`, then `_paper` (name mode via `inputs.resolve_name`, count=1, :158-170; PMIDs via `resolve_pmids`; PDFs via `read_pdfs`). It joins parts with `"\n\n"` (:208), converts the deadline with `child_deadline_seconds(left, grace=120)` (:297, `pipeline/deadline.py:283`), and calls `driver.run_one(paper, mode, timeout=...)` (:302). `_to_result` (:239) writes the artifacts.
- **Where the driver sets the deadline:** the worker sets the absolute epoch at claim. The bridge turns it into a relative `timeout=` (`driver_bridge.py:297`). `run_one` (`batch/driver.py:2339`, default `DEFAULT_TIMEOUT` 7200 s at :120) builds its `_Budget(timeout, ...)` at :2362. The driver is the only place a deadline is enforced.
- Gap for R5: `run_extraction` and `run_stage_two` take no deadline. `rag_orchestration.run_rag_rounds` has a `deadline_seconds` parameter (`rag_orchestration.py:962`), but `run_stage_two` does not pass it (`extraction.py:941-952`), so the RAG loop falls back to its own 3480 s default (`rag_orchestration.py:632-650`). `run_pathway` must pass a remaining-time budget itself.
- Also unlike the driver, `run_extraction` has no per-interaction timeout of its own and no audit-time pin (the 240 s audit timeout is wall-clock, per the LEDGER R2.5 entry).

**3. Stitching** (bodies from R3d; the R4 names are my assumption, since R4 is not merged)
1. Request to `ExtractionRequest` (`extraction.py:995`, no defaults, all 20 settings stated). `run_pathway` must state the app's widget defaults itself, and the bridge never needed them. Source of those values: unverified.
2. `run_extraction(req, reporter=, state=)` (:1022). It does `state["export_mode"]` (:1041), then `assemble_inputs`, `run_stage_zero`, `run_stage_one` and `run_stage_two` in the app's order.
3. Returns `ExtractionResult(pathway_context, final_payload, failure)` (:780). `failure` is an `InputFailure`.
4. Then R4's `run_post_pipeline` and `run_export` (not present), wrapping freeze, stages 3-7 and stage 8. They are not in the app's `run_extraction` composition.
- **State that must pass:** the single `state` mapping, which R3d's `run_stage_two` fills with `final_payload`, `final_payload_snapshot`, `pathway_context`, `user_task_context`, `stage_one`, `stage_two*`, `draft_graph`, `qa_report`, `pipeline_source_text`, `extraction_diagnostics_recorder` and `rag_result`/`rag_loop` (`extraction.py:974-990`, :959). `pwml_strict_db` is read from `state` (:949, default True). The `_diag` recorder is module-global, so one run per process.
- Not covered by `run_extraction` (`extraction.py:1031-1038`): the app's `reset_refinement_state()`, the quarantine reset, the token stats, and the ambiguous-review gate.
- `out_dir` is NOT threaded through any moved function. The app writes the fixed `PROJECT_ROOT/outputs` and `tmp` (`driver_bridge.py:30-36`; `extraction.py:615` uses `paths.PROJECT_ROOT / "tmp"`).

**4. Failure mapping**
Service kinds (strings only, no `KIND_*`): `no_pdf_uploaded` (`extraction.py:40`), `pdf_extraction_failed` (:41), `no_text` (:42), `stage_one_failed` (:517), `stage_one_boundary_error` (:518), `stage_one_incomplete` (:519), `stage_two_failed` (:744). I found no others in `extraction.py`. R4 will add its own.

What the web layer does with an unknown kind:
- `retry.decide` retries only `{crash, llm, network}` (`retry.py:60`), or `status==error` with kind `""` or `unknown` and no PWML (:98).
- Any other kind is final with "not retryable" (:106).
- `views._failure_text` shows "an unexpected problem occurred" (`views.py:123,159`).
- None of the 7 kinds is in `RETRY_KINDS`, `CONTENT_KINDS` or `_KIND_TEXT`.

Proposed mapping (a design choice, to be ruled):
| service kind | suggested result | retry class | note |
|---|---|---|---|
| `no_text`, `no_pdf_uploaded`, `pdf_extraction_failed` | `status=error` or `fail`, `KIND_CONTRACT`, issue code `bridge_unsupported_request` | final | renders "request cannot be used" (`views.py:151`) |
| `stage_one_boundary_error` | `fail`, `KIND_CONTRACT` | final | boundary verdict |
| `stage_one_incomplete` | `fail`, `KIND_CONTRACT` | final | explicit incomplete, not an infra error |
| `stage_one_failed` | **ambiguous** | see below | |
| `stage_two_failed` | **ambiguous** | see below | |

- **No mapping, needs a rule:** `stage_one_failed` and `stage_two_failed` both wrap `PipelineFailure` (`extraction.py:637,857`). That covers infrastructure (empty replies, which should be `KIND_LLM`/retry) AND the ambiguous-review refusal (Q-R3d-2, which should be `KIND_AMBIGUOUS`/final). The driver classifies these lexically with `_classify` (`batch/driver.py:1637`), with LLM markers tested first (`retry.py:57-59`). R5 would have to reuse that or string-match `multi_example_review detected` (R3d.md:295, option c). If left as `status=error, failure_kind=""`, they retry and the ambiguous case burns attempts.
- `InputFailure` does not carry which stage, so `PathwayResult.stage` (`STAGE_INPUT`/`STAGE_STAGE1`) must be derived per kind.
- The deadline kill (`KIND_TIMEOUT`, `STATUS_TIMEOUT`) and `STATUS_SCOPE_CONFLICT` already exist as web kinds. The headless path has no code that produces a scope-conflict outcome (the driver does); that is a gap vs the bridge, unverified how R4 handles it.

**5. Reporter**
- The Protocol (`service/reporter.py:36-64`) has 12 methods: info, warning, error, json, progress, success, caption, text, spinner, expander, markdown, code.
- `LoggingReporter` implements all 12 (:129-173). There is no gap if the worker uses it.
- **The worker/bridge does not construct any reporter or state today.** The bridge goes through the driver, so `run_extraction` is never called on the live path (`driver_bridge.py` has no `reporter`/`state`).
- The process-global default `install_app_defaults` still resolves `reporter=None` to the app's Streamlit pair if the app was imported (:176-193). R2 review F1 says to remove it once the worker always passes both (LEDGER current state). Not yet done.
- PLAN R2 listed an `artifact` method (PLAN line 94), and the Protocol has none. This is a plan-vs-code mismatch; I do not know whether R5 needs it.
- `LoggingReporter` goes to a logger, so nothing reaches the job page. The worker's `child.log` captures stdout and stderr, but whether logging is configured to it is unverified.

**6. Carry-forwards** (LEDGER current-state block, lines 149-163)
| item | touches | inside R5? |
|---|---|---|
| Does research mode also write the six strict-mode export reports? | `driver` `_add_*` artifacts, `pathway.py:114-116` docstring | Yes (read-only check plus docstring). Biology gates untouched |
| W1 texts into `PathwayRequest.texts`? | `service/inputs.py`, bridge `_paper` | Yes. The bridge already does this (:172-190) |
| `use_cache=False` or `fetch_text_fn` for the W1 fetch | bridge :179 does it already | Yes, keep it in the R5 path |
| Enforce input caps | `web/limits.py` (W8) | Partly; it is a W8/web-layer concern. Where the cap lives is unverified |
| Q-R3d-2: map the ambiguous refusal; move the gate into the service; one clobber-test edit authorised then | `extraction.py` gate, app `streamlit_app.py` gate, clobber test | Yes as specced. The test edit is already authorised by the PO ruling (R3d.md:312) |
| `InputFailure` rename (R3c Q3, R3d.md:300) | `extraction.py:46` and tests | Yes, a rename |
| R3a N4: the worker reporter | see item 5 | Yes: use `LoggingReporter` |
| R3a Q1: `assemble_inputs` vs W1 `read_pdfs` (recommended: worker calls `assemble_inputs`) | `request` needs PDF options (page range, skip refs, OCR) | **Not decided.** `PathwayRequest` has no PDF options; adding them is an interface change that W0 says needs "a new W0-style task" (`pathway.py:6-7`). Needs a PO ruling |
| Remove `CODE_UNSUPPORTED` from the bridge before retiring it (BR carry) | `driver_bridge.py:82`, `views.py:151` | Yes. It is the stand-in input-verdict code, so keep or move it |
| Remove the global `install_app_defaults` hook (R2 F1) | `service/reporter.py`, app, c055:746 | Probably inside R5, but it touches a test patch site; unverified |
| Render attempts shim, F1 order assertion | R3d carry | Done in R3d; not re-checked |
| Q-W0-1: `out_dir` | see below | Needs a ruling |

**7. Open product questions**
- **Likely blocks R5, needs a PO ruling:** Q-W0-1, `out_dir` (W0.md:140-148). Option A (thread `out_dir` through moved functions with the old default) edits R1/R4 signatures under "R-tasks preserve behaviour". Option B (run in the fixed `outputs/` and copy) needs no signature change but is safe only with one worker. I did not find a ruling in `PO-RULINGS-2026-11-04.md` (grep hits only list Q-W0-2/3/5). The bridge currently uses B in effect (`driver_bridge.py:30-36`).
- **Needs a ruling because it changes the frozen interface:** PDF options in `PathwayRequest` (above), and `image_path`. Nothing produces an image today, so decide whether R5 renders one (the app does so from `draft_graph`, `state["draft_graph_png_bytes"]` at `extraction.py:969`).
- **Can default:** Q-W0-2 deadline form (implemented as an absolute epoch and working); Q-W0-3 / Q-W0-5 (the PO ruled release-ready only, line 12 of the rulings); the failure-kind mapping (propose the table in item 4, with a lexical `_classify` reuse for the two ambiguous kinds).

**Could not determine**
- Whether R4 will name its functions `run_post_pipeline` and `run_export`, or what their signatures are.
- The app's default values for the 20 `ExtractionRequest` fields (not read from `streamlit_app.py`).
- Whether `LoggingReporter` output reaches `child.log`.
- Whether Q-W0-1 has been ruled (the rulings file line 58 was truncated in my search).
- The remaining unread parts: `inputs.py` internals beyond its constants, `jobs.py` serialisation beyond the request, and `driver.run_one` stage handling.
