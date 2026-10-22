# ORCH-759 — the 7 strict cohort legs at 3 draws on the merged code

**Status:** IN PROGRESS (draft written during the run so a crash cannot lose it; completed at the end).
Decision: **D-105 § 3(b)** (product owner, 2026-10-21). Sub-orchestrator record.

## 1. What runs

- The 7 strict conditions of ORCH-755, from the **primary checkout** (its `.env`: `T2PW_EXTRACTION_DRAWS=3`,
  `LLM_MAX_TOKENS`/`OPENROUTER_PREPROCESSOR_MAX_TOKENS`/`OPENROUTER_INFERENCE_MAX_TOKENS` = 128000,
  `OPENROUTER_EXTRACTION_MAX_TOKENS` = 64000), HEAD `82b4fc33` = code `f392d447` (ORCH-757 budgets, 758a `Ca++`,
  758b empty chunk), live curator.
- Runner `evidence/orch756_paper_runs.sh` with task `ORCH-759`, strict only, **10,800 s per leg** (wrapper 11,400 s),
  `--heavy-lock ORCH-759`; review pack `evidence/orch756_review_pack.py`, which copies every `*.pwml`,
  `research_pathway_*`, and the Stage-1 side files `stage1_ensemble_report.*` and `ensemble_identity_merge_report.*`
  found in the leg's time window under `tmp/extraction_diagnostics/` (legs run one at a time).
- **Order (product owner, 2026-10-22):** enterobactin, menaquinone first (certified gold, scored first; one line to
  `main` after each), then chlorophyll, Ara operon, sour taste, WNT PCP, Wnt/Ca2+.
- **Timeout:** 10,800 s per leg, orchestrator operating ruling approved by `main` 2026-10-21 under D-105's intent
  (the product owner said to remove the limits; a timeout costs only wall time). Stage 2 is not lowered.
  Basis: `ORCH-756-RESULT.md` § 4 (Stage-2 calls of 21-26 min at 128k).

**Product owner's entry point: `runs_papers/ORCH-759/REVIEW/INDEX.txt`.** Run log `runs_papers/ORCH-759/night.log`,
legs `runs_papers/ORCH-759/legs/`, G11 `evidence/g11/ORCH-759/`.

## 2. The Ara operon request was reworded on the product owner's instruction

ORCH-756's 1-draw Ara leg (report `05`) stopped at Stage 0 with `scope_conflict`: the model's reading **"AraC
autoregulation in araBAD operon"** was rejected against the requested **"Ara operon inactivation (autoregulation of
AraC)"**, the PathBank name in the original topic line. Product-owner ruling via `main`, 2026-10-22: **reword the
request; no code or gate change; the scope check stays exactly as strict.**

- Original, kept for the record: `docs/evaluation_v2/paper_runs_orch755/topics/ara_operon.txt`
  `PMC11983282 | Ara operon inactivation (autoregulation of AraC) | Escherichia coli`
- New: `docs/evaluation_v2/paper_runs_orch755/topics/ara_operon_v2.txt`
  `PMC11983282 | AraC autoregulation of the ara operon | Escherichia coli`
- Dry run `--stage-only` (zero LLM legs): `g11/ORCH-759/02-stage-ara-operon-v2`, exit 0, 0 survivors; `plan.json` reads
  `requested_pathway "AraC autoregulation of the ara operon"`, accepted (`runs_papers/ORCH-759/stage/ara_operon_v2/`).
- The leg runs as condition `ara_operon_v2` (review folder `REVIEW/ara_operon_v2/strict/`). If it still gets
  `scope_conflict`, it is **not reworded again**: both names are recorded and reported to `main` as a finding.

To use the new file, the runner (which expands its condition list once, at start) was paused with its `PAUSE` file at
the first leg gap, the idle runner stopped, and relaunched with the remaining conditions (§ 5).

## 3. Results

(filled as legs finish)

## 4. Findings

1. **The Stage-0 scope check rejects a paraphrase of the same pathway** (ORCH-756 report `05`: observed "AraC
   autoregulation in araBAD operon" vs requested "Ara operon inactivation (autoregulation of AraC)"). Registered as an
   **open policy question for the product owner, not a defect to fix now** (product-owner ruling, 2026-10-22). The
   ORCH-755 Ara leg passed Stage 0 on another draw, so the verdict depends on the Stage-0 wording.

## 5. Incidents

(filled as they happen)

## 6. Not settled

(filled at the end)
