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

Run 2026-10-22 10:37 → 16:56 from the primary checkout (code `f392d447`; HEAD moved only by docs commits during the
run). Review `runs_papers/ORCH-759/REVIEW/INDEX.txt`; G11 `evidence/g11/ORCH-759/01, 03-08` (report `02` is the
stage-only dry run); every report `FINAL SURVIVING COUNT : 0`.

| paper | G11 | min | outcome | PWML at 3 draws | PWML at 1 draw (ORCH-756) | cause, and where it is handled |
|---|---|---|---|---|---|---|
| enterobactin | `01` | 25.7 | `final_pre_export_stage3_gates`, `gate.protein_has_degree_after_normalization`; release `review_required`, 6 reactions, 4 interactions | **no** | yes | E1 (§ 4.2): degree-0 duplicate row → fixed by **ORCH-759a**, merged `57713a11` |
| menaquinone | `03` | 42.9 | PWML required-field gate: `species_missing_taxonomy`, `species_missing_classification` for 'L. lactis subsp. lactis KF147'; release `review_required`, 16 reactions | **no** | yes | species identity (ORCH-761) → **ORCH-762** FX2/FX3, merged `9757234d`; 16 rows for 9 gold → **ORCH-763** (11 rows), **ORCH-764** (class name) |
| chlorophyll | `04` | 87.9 | `final_gate_report_missing`: strict quarantine refused `entity_type_overlap:4` | **no** | no (timeout) | quarantine closure kept gene rows by a protein component's name → **ORCH-759c** (in review); a further blocker (`CPO reaction` enzyme a bare protein) is open |
| Ara operon (reworded, `ara_operon_v2`) | `05` | 22.3 | Stage-0 `scope_conflict` again (§ 4.3) | **no** | no (scope_conflict) | not reworded again (product-owner ruling); reported to `main` |
| sour taste | `06` | 10.3 | Stage-0 `scope_conflict`: organism 'Mus musculus' read, 'Homo sapiens' requested (§ 4.4) | **no** | deferred | reported to `main` |
| WNT PCP | `07` | 55.9 | PASS (with warnings), release `review_required`, 3 reactions, 11 interactions | **yes**, `pathway.review_required.pwml` | deferred | — |
| Wnt/Ca2+ | `08` | 88.3 | "extraction did not finish inside the time budget", `in_process_deadline` at 5,296 s of 10,800 s | **no** | deferred | the batch driver's fixed 3,600 s per-interaction cap → **ORCH-759d**, merged `9f7d64ee` |

**The budget fix works at 3 draws.** Empty `finish_reason=length` completions per stage (`evidence/orch759_trace_stats.txt`,
`orch755_trace_stats.py`): **1 of 309 model attempts** across the 7 legs (Stage 1 1/61; Stage 2 0/32; audit 0/11; chat
0/20; gap resolver 0/46; RAG prose 0/120; preprocessor 0/19), against ORCH-755's Stage 2 65/95, audit 35/38, curator
42/43. **No call reached 20 minutes and no provider `finish_reason=error`** (`evidence/orch759_leg_stats.txt`), unlike
ORCH-756's chlorophyll Stage 2. The one empty Stage-1 attempt is Wnt/Ca2+ at the 64,000 extraction cap (D9).

**Not yet measured:** whether the merged fixes (759a, 762, 763, 764, 759b, 759c, 759d) turn these into PWMLs. That is
the proof re-run (§ 7).

## 4. Findings

1. **The Stage-0 scope check rejects a paraphrase of the same pathway** (ORCH-756 report `05`: observed "AraC
   autoregulation in araBAD operon" vs requested "Ara operon inactivation (autoregulation of AraC)"). Registered as an
   **open policy question for the product owner, not a defect to fix now** (product-owner ruling, 2026-10-22). The
   ORCH-755 Ara leg passed Stage 0 on another draw, so the verdict depends on the Stage-0 wording.
2. **E1 — enterobactin at 3 draws: a degree-0 duplicate protein row blocks a correct `review_required` pathway**
   (read-only diagnosis by an Opus agent, 2026-10-22; helpers and output `evidence/orch759_enterobactin_diag/`, its
   bounded run `01-diag.json` 0 survivors). Leg `runs_papers/ORCH-759/legs/enterobactin-strict/2026-10-22_1037/`.
   Chain, at `f392d447`:
   1. Stage 2 added the enzyme `{"protein_complex": "enterobactin synthase complex"}` (inferred, 0.95) to both EntF
      reactions; it is in `merged_payload.json`, not in `stage1_payload.json`, and its evidence sentence is not in the
      source text.
   2. Normalization added the undeclared name as a protein; `relocate_complex_named_proteins`
      (`process_normalizer.py:5140-5241`) made a generated wrapper (`generation_reason:
      complex_named_source_entity_wrapper`, `:5205`) with one bare component, protein `enterobactin synthase`.
   3. The audit, which now returns usable output (ORCH-757), accepted 2 patches tying `enterobactin synthase` to
      EntF's UniProt P11454 (`audit_iteration_summary.json`).
   4. The Stage-6 remap's `_reconcile_components_against_local_proteins` (`map_ids.py:6985-7078`) matches on UniProt
      first, first row wins: the component became `EntF`, leaving protein row 7 `enterobactin synthase` unreferenced.
   5. The strict orphan cleanup (`process_normalizer.py:5699-5702`) runs only at Stage 3, before this rename, so the
      first connectivity check after it is the final pre-export gate (`streamlit_app.py:3299-3307`), whose component
      exemption matches by name only (`process_normalizer.py:5092-5105`): "Protein has degree 0" (`:5106-5108`).
   6. The invalid `generation_reason` findings are a separate, informational schema mismatch
      (`payload_models.py:242,727`; `schema.py:270`), not the cause.

   Proof on the artifacts (no LLM): the gate fails on the leg's `final_mapped.json` (sha `e72d3a65…`) with exactly this
   error; renaming the component back, or removing the duplicate row, passes with 0 errors; ORCH-756's passes.
   **Why 1 draw passed:** the ORCH-756 payload has no "enterobactin synthase complex" (input), and its audit returned
   5/6 `length`-truncated attempts at 3,600 tokens, so no patch applied (code). ORCH-757 exposed a pre-existing
   defect; 758a/758b are not involved; interaction wrappers are not the cause (ORCH-756 passed with 7 bare
   protein-endpoint interactions). **Classification:** `product_contract_violation`: `quarantine_report.json`
   `release.status = review_required`, `strict_gates_passed: true`; PRODUCT_CONTRACT § 1 forbids an irrelevant
   degree-zero entity as a terminal blocker (merge rule 7). **Reach:** deterministic once the shape occurs, but the
   shape is LLM-dependent; across 5,195 historical `final_mapped.json`, `complex_named_source_entity_wrapper` occurs
   only for the two enterobactin papers. Not established: whether the audit wrote the name `EntF` or only the
   accession; pre-757 behaviour on this exact input; whether a PWML with two rows sharing P11454 imports.
   **Routed to `main` as a product question** (entity-row identity policy), § 6. **Ruled by `main` 2026-10-22: option (a)**
   (fold an unreferenced duplicate with the same ADMITTED accession and species, alias and quotes kept, firing in the real
   app path). Lane **ORCH-759a**, merged `57713a11`.
3. **The reworded Ara request still gets `scope_conflict`** (report `05`): Stage 0 read **"E. coli L-arabinose (P_BAD) and
   L-rhamnose (P_Rha) inducible promoter regulation"** against the requested **"AraC autoregulation of the ara operon"**.
   Per the ruling it was not reworded again. Both strings recorded; reported to `main`. With § 4.1 this is two different
   Stage-0 readings of one paper, both rejected by the same strict check.
4. **Sour taste: Stage-0 organism `scope_conflict`** (report `06`): Stage 0 read organism **'Mus musculus'**, the topic
   requested **'Homo sapiens'** (`topics/sour_taste.txt`, from the ORCH-751 cohort). At 3 draws ORCH-755's sour-taste leg
   passed Stage 0 (it failed later on D2), so this too depends on the Stage-0 draw. Whether the topic's organism is
   right for PMC7059786 is a question for the product owner; not changed.
5. **Wnt/Ca2+: the 10,800 s leg ruling was defeated by a fixed 3,600 s per-interaction cap** (`batch/driver.py:120`
   `DEFAULT_APP_TIMEOUT`, `_Budget.slice()`), never exposed to callers. The pipeline submit (Stage 1 ~3,071 s + Stage 2
   ~1,713 s) hit it. Lane **ORCH-759d** (the cap follows the leg budget), APPROVE WITH NOTES, merged `9f7d64ee`.
   Reviewer note: AppTest only stops the app when the pipeline yields; in this leg the stop came ~1,640-1,695 s after the
   cap, so a leg reaching its child deadline in Stage 2 will overrun the 120 s grace and be killed by the parent.
6. **Chlorophyll: the strict quarantine's closure kept 4 gene rows alive through a same-named protein component**
   (read-only diagnosis, `evidence/orch759_chlorophyll_diag/`): `keep_norms = referenced | _complex_component_norms(...)`
   ignores buckets (`strict_quarantine.py:~2454`, `_prune_entities` `~:1829`), so `_entity_type_overlaps` refused
   (`entity_type_overlap:4`) and no final gate ran. `product_contract_violation` (merge rule 7). Counterfactual replay:
   `ok`, `review_required`, strict final gate passes (30 reactions, 25 interactions). Lane **ORCH-759c**. A second
   blocker remains on this leg: the required-field contract fails `reaction_enzyme_must_be_protein_complex` on `CPO
   reaction` (bare protein `CPO` next to a placeholder wrapper `CPO1`). Diagnosed read-only
   (`evidence/orch759_cpo_diag/`): Stage 6 wrapped reaction enzymes BEFORE the Phase-2 UniProt fallback, so an enzyme
   resolved only in Phase 2 shipped bare (4 of 121 archived strict legs). Lane **ORCH-759e**, merged `360a6ac5`.
7. **Menaquinone at 3 draws: an unreferenced, abbreviated species row blocks the export** (read-only diagnosis,
   `evidence/orch759p_species_diag/`). The 3 draws pooled three species rows; the paper's own spelling "L. lactis subsp.
   lactis KF147" is referenced by nothing, NCBI cannot match the abbreviated genus, and the C-120 alias pass deliberately
   refuses strain-suffixed abbreviations (`map_ids.py:3574-3576`, D-098), while its twin row and all 7 Men proteins carry
   684738. The required-field gate checks every species row (`ir.py:2911-2937`). 2 of 2 three-draw legs. Product
   question to `main` (options: extend C-120 to suffixed abbreviations; drop unreferenced species rows before the freeze).

## 5. Incidents

1. **Runner swap for the reworded Ara topic.** The runner expands its condition list at start, so after enterobactin it
   was paused (`PAUSE` file), stopped while idle by exact PID (`taskkill /PID 34988`; no leg running, lock free) and
   relaunched with `ara_operon_v2` (`night.log` RELAUNCH line).
2. **A parallel session moved the primary HEAD** during legs (docs-only commits by `main`: gold-accuracy scoring,
   D-106-D-110). Checked each time: no `src/` or `scripts/` change, so the measured code is the start-of-run SHA.
3. **The machine crashed earlier the same day** during ORCH-756 (recorded in `ORCH-756-RESULT.md` § 5); no ORCH-759 leg
   was affected.

## 6. Not settled

1. **The scope check rejects paraphrases** (§ 4.1, 4.3): open policy question for the product owner (D-110: Ara dropped,
   the check unchanged).
2. **The menaquinone species row** (§ 4.7): awaiting the product owner's choice.
3. **ORCH-765** (parenthetical abbreviation bridges, D-110 § 3 / AMENDMENT 1): round 1 reviewed; one-draw corroborated
   lookalikes (`Optical (OD)`, `Luria (LB)`, `Free radicals (ROS)`, an "and"-list member) still bridge; product owner asked
   whether to require two corroborating draws. Not merged.
4. **Long reasoning-model calls persist at 128k:** an ORCH-759p audit attempt ran ~1,246 s and returned an empty
   `finish_reason=length` (`evidence/orch759p_leg_stats.txt`).

## 7. The proof run (ORCH-759p): 3 draws on the merged fixes

Run `ORCH-759p` from the primary checkout at 3 draws, 10,800 s per leg, review `runs_papers/ORCH-759p/REVIEW/INDEX.txt`,
G11 `evidence/g11/ORCH-759p/`, all reports `FINAL SURVIVING COUNT : 0`. Enterobactin, menaquinone and sour taste ran on
`21155be0` (759a, 762, 759b, 763, 759d, 759c, 764 merged); chlorophyll on `360a6ac5` (+ 759e).

| paper | ORCH-759 (before) | ORCH-759p | leg |
|---|---|---|---|
| enterobactin | no PWML (E1) | **PWML**, `review_required`, 5 reactions, 10 interactions, 53 min | `legs/enterobactin-strict/2026-10-22_2101/` |
| menaquinone | no PWML (species, 16 rows) | no PWML: now **`release_ready`**, 19 reactions, export blocked by the species row (§ 4.7) | `legs/menaquinone-strict/2026-10-22_2154/` |
| sour taste (*Mus musculus*, `sour_taste_v2`) | `scope_conflict` (organism) | **PWML**, `review_required`, 21 min | `legs/sour_taste_v2-strict/2026-10-22_2233/` |
| chlorophyll | no PWML (quarantine refusal) | running at the time of writing | `legs/chlorophyll-strict/` |

**D-107 proven in a real 3-draw app leg (enterobactin):** rule 3 ran and wrote `ensemble_identity_merge_report.json` (5 → 5
reactions, nothing to merge) in the Stage-1 run directory `tmp/extraction_diagnostics/run-338be6146e19/`, beside
`audit_iteration_summary.json` and `gap_iteration_summary.json`; the review pack copied the identity report.
LLM health over the first three legs: 1 empty-length completion in 96 attempts, 0 provider errors
(`evidence/orch759p_trace_stats.txt`, `orch759p_leg_stats.txt`).
