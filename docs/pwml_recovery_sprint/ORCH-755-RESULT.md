# ORCH-755 — overnight paper runs for hand comparison

**Status:** NIGHT FINISHED 2026-10-21 09:37 — **no strict leg wrote a PWML**; cause found, fix needs a product-owner decision (§ 5). Base: `main` = `sprint/pwml-recovery` @ `8b074dda`.
Requested by the product owner: run the papers in the background and have the text results and the
PWML files ready to check the next day. The id ORCH-755 comes from the 755-759 block the ORCH-760
session left free for this one.

## 1. What runs

Seven conditions, from the ORCH-751 cohort: enterobactin, menaquinone, chlorophyll, Ara operon, sour
taste, WNT PCP, Wnt/Ca2+. Each runs in strict and in research mode, for 14 legs. All strict legs go
first. Each leg is one `scripts/batch_run.py` pair through the real app, inside its own `bounded_run.py`
with the heavy lock, a 3,600 s pair timeout and a 4,200 s wrapper timeout. Configuration is D-104: 3
draws, 64,000 extraction tokens, prompt v03, live curator. Inputs and details are in
`docs/evaluation_v2/paper_runs_orch755/README.md`.

This is the **first end-to-end exercise of the 3-draw path** and of rule 3 (the Stage-6 database-ID
merge), which had been tested only offline and on synthetic payloads (ORCH-754-RESULT § 5).

## 2. Where the results land

`runs_papers/ORCH-755/REVIEW/INDEX.txt` is rewritten after every leg. Each `<condition>/<mode>/`
folder holds:
- `reaction_evidence.txt`: the app's deterministic Reaction Evidence export
- the PWML, if one was written
- the Stage-1 ensemble side files (support marks and merge flags)
- `RESULT.txt`

The batch driver does not copy the ensemble side files into a leg, because the app writes them under
`tmp/extraction_diagnostics/<unique>/`. The review script picks them by the leg's time window, and
every pick is listed in the index.

## 3. Results

14 legs, 22:01 → 09:37, zero surviving processes in every report (`g11/ORCH-755/08`–`22`; report 11 is a heavy-lock race,
retried as 12). Outcomes are in each leg's `manifest.jsonl` under `runs_papers/ORCH-755/legs/`.

| outcome | legs |
|---|---|
| **timeout at 3,600 s** (`budget_exhausted`) | menaquinone strict + research, chlorophyll strict + research, Ara strict, WNT PCP strict, Wnt/Ca2+ research (7) |
| Stage 1 fail, "Chunk 3 failed to produce valid JSON" (the 200,898-char WNT paper, chunked) | Wnt/Ca2+ strict, WNT PCP research (2) |
| strict contract fail | enterobactin (`final_gate_report_missing` at `final_pre_export_gate_lifecycle`), sour taste (stopped by Stage 3 revalidation before PWML export) (2) |
| **pass** | enterobactin, Ara operon and sour taste, all research mode (3). Research mode writes a citation report, not a PWML, by design |

**The 3-draw Stage 1 itself worked on every leg:** 3/3 draws succeeded on 14 of 14 reports (one later chunked draw lost 2 of 3),
in 4-27 min (`tmp/extraction_diagnostics/run-*`, mtime windows). Pooled reactions: enterobactin 4-5, menaquinone 9-11,
chlorophyll 41 (strict; 31 single-draw) and 33 (research), Wnt/Ca2+ 2, and 0 on the regulatory papers. **No merge was flagged in
any leg**, and chlorophyll's 41 matches the pre-fix inflation level on fresh draws. Not yet analysed (§ 5 item 3).

## 4. Root cause, measured

Every downstream stage still on a small budget returned **empty completions with `finish_reason=length`**: the reasoning model
spent its output budget reasoning before it wrote anything. The retries ate the hour and the gates that needed the answers failed.
Counts of model attempts across the LEG_TRACEs (`evidence/orch755_trace_stats.py`, output `evidence/orch755_trace_stats.txt`; payload sizes `evidence/orch755_payload_sizes.py`):

| stage | 2026-09-16 run (1 draw) | ORCH-755 (3 draws) |
|---|---|---|
| Stage 2 inference | 0/24 empty | **65/95** |
| audit loop | 0/28 | **35/38** |
| curator chat | 34/152 | **42/43** |
| gap resolver | 15/59 | 13/17 |
| Stage 1 extraction (64,000 budget) | 0/18 | 8/111 |

Same model both nights (`deepseek/deepseek-v4-flash`, every attempt). Two changes explain it:
1. **Pooled 3-draw payloads are about twice as large**: Stage-1 payload median 40.5 KB, against 16.5 KB (2026-09-16) and
   26.2 KB (2026-09-06).
2. **The Stage-2 budget fell from 20,000 to 16,000.** The product owner's `streamlit_app.py` edit (`_bounded_env_int`, committed in
   `70d05e03`) makes the UI read `OPENROUTER_INFERENCE_MAX_TOKENS=16000` from `.env` instead of its old hardcoded 20,000. `754r0`
   noted the change and called it authorised; its effect on a reasoning model was not foreseen.

Most of the other budgets are **hardcoded**, several in the protected `streamlit_app.py`:
- audit: `min(10000, …)`, plus 1,200 (`:3849`, `:3881`, `:3934`, `:3951`)
- gap resolver: `min(1400, …)` (`:4005`)
- curator: 2,000 (`:4240`)
- overwatch: 3,000 (`:4400`)
- RAG prose extraction: 1,500 (`rag/extract.py:48`)

ORCH-747 § 6.2 found this exact failure for extraction and fixed only that budget; D-104 § 1 raised only that one.

**Classification:** `product_contract_violation`, a pipeline configuration defect. D-104 § 2's 3x draws enlarge the payload that every
later small-budget stage must reason over. It is not a gold or policy issue.

## 5. Open for the product owner

1. **Getting PWMLs for hand comparison now.** Re-run the 7 strict legs at 1 draw, with Stage 2 at 64,000: the September configuration
   that produced PWMLs, plus today's merged work. About 4 h.
2. **The real fix for 3 draws.** Raise or scale the downstream budgets, Stage 2 in `.env` and the hardcoded ones in code. This touches
   the protected file, so it needs a D-xxx, a lane, a review, and a 3-draw re-run (about 10 h).
3. **Chlorophyll pooled 41, with 0 merges flagged.** Analyse last night's draws against the merge rules: missed bridges, or real
   variance.
4. **The WNT paper's chunked Stage-1 JSON failure** (200,898 chars) predates this work (ORCH-751 § 2.3 saw an empty draw on it).
