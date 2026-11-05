# T2PW Web Server — Plan

**Target.** The NAR Web Server Issue. The letter of intent is due **Dec 15, 2026**, and the
server must be live and working by then. The full paper is due around **Feb 15, 2027**.
Check this year's dates on the official NAR page (academic.oup.com/nar, "Submission — Web
Server"). Past calls judge: scientific quality, broad interest, computation on the user's
own data, and a fully working, well-designed site. Assume users get **no mandatory login**.

**The spec from Dr. Wishart:**
- A stand-alone system.
- Input is either (1) a pathway name, or (2) a set of articles (PDFs or PubMed IDs).
- Up to 5 pathways per submission.
- The organism comes from a drop-down.
- A checkbox "OK to add this pathway to PathBank", **on by default**.

Product owner decisions are in `DECISIONS.md`.

---

## 1. Starting point (measured on `main` @ `06a0b8c0`)

| Need | What exists | Gap |
|---|---|---|
| Pathway-name input | `batch/fetch.py` takes `topic \| organism \| count` and searches Europe PMC and NCBI. Multi-paper RAG is in `rag/acquire.py` | Only the batch CLI can reach it. About half the papers found have no open-access full text |
| PDF input | `extraction/pdf_parser.py`, multiple files, OCR fallback | none |
| PubMed ID input | Batch accepts pinned PMC IDs and DOIs | Plain PMIDs need a PMID→PMCID conversion, and a clear message when there is no open-access full text |
| Organism | Inferred by Stage 0, or typed into the focus box. Batch passes a `RequestedScope` (`rag/eligibility.py:1014`) | A drop-down of all PathBank species, passed as an explicit scope |
| Several pathways | The batch runner works sequentially | A job queue, max 5 per submission, with auto-retry |
| PathBank consent | nothing | A flag stored with the job, and approved PWML copied to a curator queue |
| Public front end | A Streamlit developer UI, with DB credentials in the sidebar | A new, polished public site |

**The blocker is how the code is structured.** Pipeline orchestration lives in
`src/t2pw/app/streamlit_app.py` (7,281 lines). Importing it runs the UI script. The only
headless path today is `batch/driver.py`, which clicks the app's three buttons through
`streamlit.testing.v1.AppTest`.

Streamlit coupling, counted as `st.` references in each section:

| Section | Lines | `st.` refs |
|---|---|---|
| `freeze_canonical_payload` | 3245–3352 | 0 |
| `run_post_pipeline_sbml_artifacts` (stages 3–7) | 3353–4595 | 0 |
| `run_pwml_export` (stage 8) | 4612–5058 | 0 |
| `run_rag_rounds` | 1240–1506 | 1 |
| `maybe_run_rag` | 412–912 | 10 |
| `run_quarantine_boundary` | 1791–1861 | 5 (all `session_state`) |
| "Run pipeline" handler, stages 0–2, **inline script** | 5284–5727 | 73 |
| "Run audit and DB mapping" handler | 6677–~7050 | many |

Line numbers are as of `06a0b8c0`. Re-measure before relying on them.

**Run time.** Even the fastest recent run took about 18 minutes, and the batch timeout is
60 minutes. So jobs are asynchronous: the user submits, gets a job page and an optional
email, and comes back later.

---

## 2. Architecture

```
Public site (FastAPI + Jinja2 templates + HTMX, hand-written CSS, no JS build step)
  ├─ Submit: mode {pathway name | PubMed IDs | PDFs}, organism, up to 5 pathways,
  │          PathBank consent (default on), optional email
  ├─ /job/<id>   status, progress, retries, PWML download, rendered image,
  │              citation report, review flags
  └─ Examples / Help / About / Cite
        ↓
Job store (SQLite) ── one worker process, one pipeline run at a time
        ↓
t2pw.service.run_pathway(PathwayRequest) -> PathwayResult     ← made by the refactor
        ↓ (when consent is on and a strict PWML exists)
PathBank curator queue (directory + DB table). A curator imports into PathWhiz. No auto-write.
```

- The Streamlit app stays as the developer and debugging tool. It calls the same service
  functions as the web server.
- The web layer lives in new paths only: `src/t2pw/service/`, `src/t2pw/web/`, `webserver/`.

---

## 3. Refactor: R-series (behaviour-preserving)

Every R-task changes structure only. Proof that behaviour is preserved:
- SMOKE 511 passes.
- The golden-seam tests pass unchanged.
- From R3 on, an **equivalence test** runs the same input with recorded or mocked LLM
  replies through both the app (`AppTest`) and `run_pathway()`. It asserts an equal
  canonical payload hash and equal PWML bytes.

| ID | Task | Depends on | Route (see ORCHESTRATOR § 3) |
|---|---|---|---|
| **R0** | Scout: list every symbol, helper, constant and test patch target the move in R1 touches, with line numbers. Read-only | — | scout |
| **R1** | Pure move of `freeze_canonical_payload`, `run_post_pipeline_sbml_artifacts`, `_validate_stage8_export_payload` and `run_pwml_export` (plus their private helpers) into `src/t2pw/service/`. The app re-imports them under the same names. No logic edits | R0 | implementer, mechanical |
| **R2** | `Reporter` protocol (`info`/`warning`/`error`/`progress`/`artifact`) with Streamlit and logging implementations. Move `maybe_run_rag`, `run_rag_rounds` and `run_quarantine_boundary` into the service, with an explicit state dict replacing `st.session_state` | R1 | implementer, hard |
| **R2.5** | Equivalence harness: recorded LLM fixture, app-vs-service comparison test. Labelled as a new acceptance test | R1 | implementer, hard |
| **R3** | Extract the "Run pipeline" handler (stages 0–2, lines ~5284–5727) into `run_extraction(request, reporter, state) -> ExtractionResult`. The button calls it | R2, R2.5 | implementer, hard |
| **R4** | Extract the "Run audit and DB mapping" handler and the PWML step into `run_post_pipeline(...)` and `run_export(...)` | R3 | implementer, hard |
| **R5** | `run_pathway(PathwayRequest) -> PathwayResult`. The request holds texts, PDF paths, PMIDs, pathway name, organism, export mode, out dir and deadline. The result holds status, failure class, PWML path, image path, report paths and attempt log | R4 | implementer, design |
| **R6** | The batch driver calls `run_pathway` directly. The `AppTest` path stays one release as a cross-check | R5 | implementer, standard |

## 4. Web server: W-series

These can start against a stub `run_pathway` as soon as the R5 interface is fixed (W0).

| ID | Task | Depends on | Route |
|---|---|---|---|
| **W0** | Freeze the `PathwayRequest` / `PathwayResult` dataclasses as an interface file with a stub implementation | R0 | implementer, design |
| **W1** | Input adapters: name + organism → `TopicSpec` and fetch; PMID → PMCID (NCBI idconv) with a "no open-access full text, please upload the PDF" outcome; PDF upload → `pdf_parser` | W0 | implementer, standard |
| **W2** | Organism list: every PathBank species from MySQL, frozen to `data/webserver/organisms.json` by a script, mapped to the scope string the pipeline expects | W0 | implementer, standard |
| **W3** | Job store and worker: SQLite jobs table, one worker, max 5 pathways per submission, total-queue cap, resume after restart, cleanup of the job's own process tree only | W0 | implementer, hard |
| **W4** | Auto-retry policy: up to 2 retries on infrastructure failure (timeout, empty LLM replies, provider error) or no PWML. Hard wall-clock cap per pathway. Attempts shown on the job page | W3 | implementer, standard |
| **W5** | Public UI, design system: layout, typography, colour tokens, light and dark themes, components | W0 | ui-builder, design |
| **W6** | Public UI, pages: submit form, job status (polling with HTMX), results (PWML download, rendered image, reports), examples, help, about/cite | W5, W3 | ui-builder |
| **W7** | PathBank curator queue: store the consent flag; for consented jobs with a strict PWML, copy the PWML, provenance and citation report to the queue. Never write to PathBank | W3 | implementer, standard |
| **W8** | Email when done (optional field), rate limit per IP, PDF size and page caps | W3 | implementer, standard |
| **W9** | Deploy: lab host, reverse proxy, HTTPS, server `.env` (OpenRouter key, read-only PathBank DB user), logs, disk quotas, LLM spending cap, `systemd` units | W6, R5 | implementer + PO |

## 5. Timeline

| Dates | Milestone |
|---|---|
| Nov 5–8 | R0, R1, W0. Settle the hosting question |
| Nov 9–22 | R2, R2.5, R3; W1, W2, W3, W5 in parallel |
| Nov 23–Dec 6 | R4, R5, R6; W4, W6, W7, W8. First end-to-end public job |
| Dec 7–14 | W9 deploy, bug fixing, 3–4 curated examples, LOI drafted with the live URL |
| Dec 15–Feb 15 | Benchmarking, user testing, reliability (ORCH-739 empty replies), the paper |

## 6. Risks

- **Reliability.** Empty LLM replies and run-to-run variance mean some public runs fail.
  Auto-retry (W4) plus clear failure messages, and curated examples for reviewers.
- **Name mode depends on open-access full text** (about 50% yield). We run with what RAG
  finds and say so plainly when it finds nothing.
- **R3 and R4 are the risky extractions.** They are long inline handlers full of
  `session_state` side effects. That is why the equivalence harness (R2.5) comes first.
