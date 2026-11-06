# R3a - the submit block's input assembly moves to `service.extraction.assemble_inputs`

- **Branch:** `ws/R3a-input-assembly` from `webserver/refactor` @ `480b2384`. **Code commit:** `d043e263`. Docs commit: this file plus `evidence/R3a/`.
- **Status:** done, pending independent review.
- **Behaviour change:** none intended. One invisible difference is listed in section 3, item 1.
- **NEW TESTS (R3a, new module):** `tests/test_service_extraction.py`, 29 tests.
- **Scope:** only R3a from the scout's split (`results/R3-scout.md` section 7). Stage 0 onwards is untouched.

## 1. What moved where

| what | base `480b2384` `streamlit_app.py` | tip `d043e263` |
|---|---|---|
| text parts, PDF loop (page range, skip_refs, OCR, temp file), their messages, the three `st.stop` points, `text = "\n\n".join(text_parts)` | 2084, 2086-2162 | `src/t2pw/service/extraction.py` 59-138 (`assemble_inputs`, 42) |
| `user_task_context = (...).strip() or None` (unrelated to text assembly) | 2085 | stays in the app, `streamlit_app.py` 2085 |
| app call site: `assemble_inputs(...)`, `st.stop()` on failure, `text = _assembled.text` | - | `streamlit_app.py` 2087-2101 |
| `from t2pw.service.extraction import assemble_inputs` | - | `streamlit_app.py` 191 |

- **New module** `src/t2pw/service/extraction.py` (138 lines). It holds `assemble_inputs`, `AssembledInputs(text, failure)`, `InputFailure(kind, message)` and three kind constants: `no_pdf_uploaded`, `pdf_extraction_failed` and `no_text`.
  - Its imports are `os`, `tempfile`, `parse_pdf`/`SKIP_SECTIONS` (the same module the app imports them from) and the `Reporter` protocol.
  - It imports neither streamlit nor `t2pw.app`. Two tests check this: `test_importing_the_module_loads_neither_streamlit_nor_the_app` and R2's `test_service_reporter` scan of every service module.
- **`reporter` is a required keyword with no default.** The function never touches `default_reporter()` or `install_app_defaults` (R2 F1). The app passes `StreamlitReporter(st)` explicitly.
- **The app keeps its `parse_pdf, SKIP_SECTIONS` import (line 130)** although it no longer calls them, so `app.parse_pdf` still resolves. No test patches it there (checked with grep).
- **Verbatim proof:** `evidence/R3a/verbatim-diff.txt` is a unified diff of base 2084-2162 against extraction.py 59-136. The indent is the same at both ends. The only differences are:
  - `uploaded_pdfs` / `uploaded_pdf.name` / `uploaded_pdf.read()` → `pdfs` / `name` / `data`;
  - `st.X(...)` → `reporter.X(...)`;
  - each `st.stop()` → `message = ...; reporter.X(message); return AssembledInputs(failure=...)`. The message text is unchanged;
  - the local `import tempfile` / `import os` became module imports;
  - the `user_task_context` line stays behind.

## 2. Reporter-call mapping

Five methods were added to `src/t2pw/service/reporter.py`, on the `Reporter` protocol and on both reporters. Nothing existing changed.

| base call | new call | `StreamlitReporter` forwards to | `LoggingReporter` |
|---|---|---|---|
| `st.warning(...)` x4 (no PDF, bad page range, per-PDF warning, no text) | `reporter.warning` (existing) | `st.warning` | warning |
| `st.error(f"PDF extraction failed ...")` | `reporter.error` (existing) | `st.error` | error |
| `st.spinner(f"Extracting text from {name}...")` | `reporter.spinner(text, **kw)` **new** | `st.spinner(text, **kw)` | info + null context |
| `st.success(...)` | `reporter.success` **new** | `st.success` | info |
| `st.caption(" \| ".join(_mp))` | `reporter.caption` **new** | `st.caption` | info |
| `st.expander(label, expanded=False)` | `reporter.expander(label, **kw)` **new** | `st.expander(label, **kw)` | info + null context |
| `st.text(_preview)` inside the expander | `reporter.text` **new** | `st.text`, inside the same `with` block, so it still renders in the expander | info |
| `st.stop()` x3 | `return AssembledInputs(failure=InputFailure(kind, message))` | the app calls `st.stop()` when `failure is not None` | - |

**Display-equivalence proof:** `evidence/R3a/inline-vs-moved.log`, script `scripts/inline_vs_moved.py`, run through the wrapper, survivors 0.

- The script cuts the base app's inline code and the tip app's call-site code out of the real files, then execs each against the same recording fake `st` and fake `parse_pdf`.
- It compares every `st.*` call in order (spinner and expander enter/exit included), where the run stopped, `text`, `user_task_context`, and every `parse_pdf` call (its bytes and kwargs).
- Result: **11/11 scenarios identical**. The scenarios are text-only, empty, upload-with-no-PDF, Text+PDF with no PDF and a bad range, two PDFs with warnings, metadata and a >1000-char preview, a PDF error mid-list, an empty PDF, half-bad and bad-start ranges, and author-only metadata.

## 3. Things the reviewer must look at

1. **PDF bytes are read up front.** The app call site builds `[(f.name, f.read()) for f in uploaded_pdfs or []]` before the loop. The base read each file inside the loop, just before its own spinner.
   - Nothing is shown between those reads, and the bytes are the same. The only difference: when PDF 2 of 3 fails, PDF 3 has already been read into memory. Its bytes are never parsed, as before.
   - `uploaded_pdfs` is `[]` outside PDF modes (app line 1904), so paste mode reads nothing.
2. **`uploaded_pdfs = []` no longer rebinds the app global** in Text+PDF mode with no upload. No later app code reads `uploaded_pdfs`, `text_parts`, `_pdf`, `_meta`, `_tmp_path` and so on (grep). These were inline leftovers that are now function locals.
3. **Quirks kept on purpose:**
   - `"2-x"` keeps `_ps=2` with `_pe=None` after warning.
   - A bad page range is warned about even in Text+PDF mode with no PDF.
   - An empty-text PDF still gets its success line, caption and preview before the no-text stop.

   Each quirk has a test.
4. **BR tripwire** (`test_service_driver_bridge.py::test_app_still_joins_text_parts_that_way`): the three join lines now live in `service/extraction.py` byte-for-byte. The test was not edited and passes, in `web` and alone in `at-bridge`.
5. **Equivalence harness `TODO(R3)`:** untouched. The scout puts the `run_extraction` service arm in R3d.

## 4. Test edits (Q-R0-1)

**None.** No existing test file was changed (`git diff 480b2384..d043e263 --stat -- tests` lists only the new file).

## 5. Tests: tip vs base

All runs used `bounded_run.py` in the foreground with the venv python for both wrapper and child.

- Environment: `PYTHONPATH=<tree>/src`, `pinned_pytest --expect-tree <tree>` with no `--`, and `T2PW_OFFLINE_CURATOR=1`.
- Each run had a fresh `--basetemp=C:/t/wsR3atmp/<t|b>-<chunk>` whose parent was created first.
- `--heavy-lock ws-R3a` was passed on every golden, AppTest, 2a-2g, `at-*`, equivalence and new-test run. `web` and `rep` ran without the lock, as in R2.
- Helpers: `evidence/R3a/scripts/run.sh` and `chunks.sh`, which contains the chunk lists.

**Tip** = `C:/t/ws-R3a`. Every report shows `clean @ d043e263` (dirty=False).
**Base** = detached `C:/t/ws-R3a-base` at `480b2384`. Every report shows `clean @ 480b2384` (dirty=False).

Reports are in `evidence/R3a/{t,b}-<chunk>.{json,log}`, pin verdicts in `pin/`, the comparison in `compare.txt`. **Survivors were 0 on every run**: no JSON has a non-zero `final_surviving_count`.

| chunk | tip @ d043e263 | base @ 480b2384 | failing-node set |
|---|---|---|---|
| g1 batch_driver_seam_golden | 2 passed | 2 passed | none |
| g2 c011 | 8 passed | 8 passed | none |
| g3 stage8_export_contract | 4 passed | 4 passed | none |
| g4 quarantine_boundary | 24 passed | 24 passed | none |
| **eq** `test_equivalence_app_vs_service.py` alone | **14 passed** | 14 passed | none |
| 2a | 78 passed, 1 failed | 78 passed, 1 failed | identical (c052) |
| 2b | 134 passed | 134 passed | none |
| 2b-c055 | 14 passed | 14 passed | none |
| 2c | 101 passed, 3 skipped, 1 failed | same | identical (prefreeze_third) |
| 2c-c126 | 2 passed | 2 passed | none |
| 2d | 413 passed, 1 failed | 413 passed, 1 failed | identical (rag_seed) |
| 2d-stage3 | 28 passed | 28 passed | none |
| 2d-c119g9 | 3 passed | 3 passed | none |
| 2e | 44 passed | 44 passed | none |
| 2e-759b | 7 passed | 7 passed | none |
| 2e-hook | 1 passed | 1 passed | none |
| 2e-759afold | 2 passed | 2 passed | none |
| 2f | 32 passed, 4 skipped, 1 failed | same | identical (batch_preflight) |
| 2f-driver | 61 passed | 61 passed | none |
| 2f-gate | 14 passed | 14 passed | none |
| 2g | 162 passed, 8 skipped | same | none |
| 2g-c082pp | 1 passed | 1 passed | none |
| 2g-c056d | 14 passed | 14 passed | none |
| 2g-c119sup | 37 passed | 37 passed | none |
| 2g-759d | 4 passed | 4 passed | none |
| 2g-revview | 11 passed | 11 passed | none |
| at-c052 (alone) | 12 passed, 1 failed | 12 passed, 1 failed | identical (c052) |
| at-c082sr (alone) | 8 passed | 8 passed | none |
| at-c083 (alone) | 12 passed | 12 passed | none |
| at-pf3 prefreeze_third (alone) | 11 passed, 1 failed | 11 passed, 1 failed | identical |
| at-bridge driver_bridge (alone) | 37 passed | 37 passed | none |
| web (13 files, R2.5 postmerge list) | 301 passed | 301 passed | none |
| rep test_service_reporter | 8 passed | 8 passed | none |
| **new** test_service_extraction | **29 passed** | n/a (file absent at base) | - |

- **Golden 38 passed on both.** The golden files are unchanged.
- **AppTest files that drive submit (scout section 4)** each ran in their own process, on both tip and base:
  - c055 (2b-c055), c052 (at-c052), c082 (at-c082sr and 2g-c082pp), c083 (at-c083);
  - 759a_fold, 759b, 759d, stage3, prefreeze_third (at-pf3), reaction_evidence_app_hook;
  - batch_driver (2f-driver), batch_preflight (2f), driver_bridge (at-bridge), quarantine_boundary (g4).
- **Pre-existing failures** have identical node ids on base and tip, and are the four the brief names:
  - `test_c052_...::test_new_acceptance_the_post_pipeline_seam_returns_its_prefreeze_report`
  - `test_prefreeze_third_export_seam.py::test_run_pwml_export_runs_the_prefreeze_sequence_once_before_build_pwml_ir`
  - `test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform`
  - `test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_and_the_cure`
- **SMOKE was not run.** That is the independent runner's job.

**Heavy-run gate** (`scripts/waitfree.py`, log in `waits.log`): 69 observations in total.

- The gate compares exact worktree paths. Every `c:/t/<dir>` in a pytest/bounded_run python command line, plus the primary checkout (its `.venv` interpreter path does not count), must be `ws-R3a`, `ws-R3a-base` or `wsR3atmp`. Pylance `get_pytest_options.py` is ignored.
- The first probe at 17:35 found the e2e wrapper (PID 715676, `c:/t/ws-e2e`, holding the heavy lock as `E2E-live`). No heavy run started until it exited at 18:26.
- All 65 observations after that were free at the first look. No run exited 95.

**Mutation checks** (`evidence/R3a/mutation/`, survivors 0 on each):

- `scripts/mutate.py` made 18 single-line mutations of `extraction.py`, covering:
  - the join, the order, both strips;
  - each of the three stops;
  - page end, single page, skip_refs, OCR;
  - caption, PDF warnings, preview length, `expanded=`, temp-file removal, the page-range warning;
  - the reporter default.

  **The unit tests killed all 18.**
- Two mutations of the app call site ran against the two real-app AppTest tests. Each mutation made **both tests fail**:

  | mutation | failure |
  |---|---|
  | drop the `st.stop()` | "the run went past the input-assembly stop" |
  | pass a `LoggingReporter` | the warning is missing from `at.warning` |

  The file was restored with `git checkout` afterwards and `git status` was clean.

## 6. What failed, dead ends

1. `d01` (pre-commit, dirty tree): collection error. My scripted edit of `reporter.py`, a `python -c` inside bash double quotes, dropped the quotes around `"%s"` in the five new `LoggingReporter` lines. I fixed it before the commit. `d02`: 27 unit tests passed. Both are in `dev-superseded/`.
2. **I accidentally ran `python -` (stdin interpreter) once** in a scratch Bash call before any test. It hung, the harness moved it to the background, and I stopped it with TaskStop within a minute or two. A `Get-CimInstance` check showed none of its processes survived.
   - That check also showed two `python.exe -` processes (709124/709136, parent bash 709232). They were started at 17:22 by **another session's** `cd C:/t/ws-e2e && ... python - <<EOF` call. They are not mine and I left them alone. The orchestrator may want to tell that session.
3. My first gate script polled for up to about 8 minutes. The Bash harness moves anything over 180 s to the background, so I stopped that probe and cut the gate to 3 polls 55 s apart. I waited out the e2e job with foreground 160 s `tasklist` polls on its PID. A Monitor watch expired after 30 minutes without firing, because the job had not ended yet.
4. The first draft of `test_caption_...` indexed the reporter calls by a fixed offset. It was wrong and I replaced it with a filter on `success` calls before the commit.

## 7. Open questions (for the product owner or orchestrator)

1. **Duplication with W1's `service/inputs.py::read_pdfs`.** Both turn uploaded PDFs into text with `parse_pdf`. As the brief required, I did not merge them or touch `inputs.py`.
   - What differs: W1's version has no page range or skip_refs, has its own failure shape, and the bridge pins `enable_ocr_fallback=False` to mirror the app defaults. `assemble_inputs` carries the app's page-range, skip_refs and OCR options and its exact messages.

   | option | what | cost |
   |---|---|---|
   | (a) | keep both until R5 | two code paths that can drift. The BR tripwire guards only the join rule |
   | (b) | in R5, let the web worker call `assemble_inputs` with a `LoggingReporter` | one source of truth. `PathwayRequest` gains the PDF options (scout section 3) |
   | (c) | make `assemble_inputs` call `read_pdfs` | changes the app's messages unless `read_pdfs` grows them. Not a pure move |

   I recommend (b).
2. **Failure kinds.** `no_pdf_uploaded`, `pdf_extraction_failed` and `no_text` are new strings, used today only by the app (which just stops). Should R5 map them to the web error kinds (for example `no_text` to a user-input error), or should R3d fold them into `ExtractionResult`? It does not block R3b.
3. **Up-front PDF read** (section 3, item 1). If the reviewer wants the per-PDF lazy read kept exactly, the alternative is to pass the upload objects and call `.read()` inside the loop. The cost is that the service then depends on a `.name`/`.read()` duck type instead of bytes.
