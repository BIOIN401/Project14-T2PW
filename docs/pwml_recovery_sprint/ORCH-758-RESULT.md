# ORCH-758 — why ORCH-755 wrote no PWML: per-leg diagnosis

**Status:** DONE (diagnosis only; no product code changed, no test, no LLM call, no pipeline run).
Base: `orch758/diagnosis` = `sprint/pwml-recovery` @ `a43a2b12`, code identical to `main` @ `8b074dda`, which ran ORCH-755.
Decisions read: D-104 + AMENDMENTS 1-5, D-105. Contract: `PRODUCT_CONTRACT.md`.

**Answer in one paragraph.** The small token budgets (the D-105 fix) explain **three** of the seven
strict failures outright (menaquinone, Ara operon and, partly, WNT PCP timed out on empty
`finish_reason=length` retries). Menaquinone then meets a second blocker: none of its *L. lactis*
Men enzymes gets a UniProt. That recurs at 1 draw with D-105 budgets in ORCH-756 (D8). **Four strict failures are separate defects that the budget fix does
not touch:** enterobactin (a correctly grounded enzyme lost its UniProt to an inferred alias, so the
strict quarantine refused the pathway); sour taste (normalization crashed on the name `Ca++`, and the
exporter then rejected the un-normalized transporter rows); Wnt/Ca2+ (a chunk that is 90 % reference list
correctly returned nothing, and that empty chunk is fatal); chlorophyll (its 3-draw Stage 1 alone took
2,103 s with **zero** empty completions, which leaves no room for the rest in a 3,600 s leg). Also, the
D-104 rule-3 database-ID merge **never ran in any leg**. It cannot run in the app, because the
post-pipeline is a separate Streamlit script run that does not see Stage 1's diagnostics context
(proven offline, § 4 D4).

---

## 1. Method and sources

Read-only, offline. Every number below names its artifact.

| what | where |
|---|---|
| per-leg trace, terminal, result, artifacts | primary `runs_papers/ORCH-755/legs/<cond>-<mode>/<ts>/papers/<PMC>/<mode>/` |
| Stage-1 boundary records + ensemble side files | primary `tmp/extraction_diagnostics/run-<id>/`. The night log names the directory per leg |
| wall-clock timeline per leg, stage and attempt | `evidence/orch758_leg_timeline.{py,txt,json}` (stdlib, reads `LEG_TRACE.jsonl`) |
| Stage-1 boundaries and ensembles per leg | `evidence/orch758_stage1_boundaries.{py,txt}` (stdlib, from `night.log`) |
| Stage 1 vs downstream timing, 2026-09 reference runs vs ORCH-755 | `evidence/orch758_reference_timing.{py,txt}` |
| rule-3 reachability, AppTest probe | `evidence/orch758_rule3_probe{,_app}.py`, output `orch758_rule3_probe.txt`, G11 `g11/ORCH-758/01-rule3-probe.json` |
| re-chunking, chlorophyll lexicon, counterfactual | `evidence/orch758_chlorophyll_lexicon.{py,txt}`, G11 `02-chl-lexicon.json` |
| ORCH-751 draws re-pooled with chunk-1 vs full-paper dictionary | `evidence/orch758_replay_dictionary_swap.{py,txt}`, G11 `03-dict-swap.json` |
| WNT chunk 3 content | `evidence/orch758_wnt_chunk3_terms.{py,txt}` |
| chlorophyll strict side file, copied | `evidence/orch758_inputs/chlorophyll_strict_stage1_ensemble_report.{json,txt}` (sha256 `63f047b5…` / `272532e0…`, identical to `tmp/extraction_diagnostics/run-44bc6ac5b7d7/`) |
| ORCH-756 extra evidence, read only | `.claude/worktrees/orch756/runs_papers/ORCH-756/legs/enterobactin-strict/2026-10-21_2020/` (1 draw, D-105 budgets) |

Every script that imports `t2pw` ran through `bounded_run.py`, in the foreground, without the heavy
lock. Each run left **0 surviving processes** (G11 reports 01-03; `g11_evidence.py check --task ORCH-758`
reports 3 artifacts, 0 non-compliant).

**Facts about the run that the rest depends on:**
- **Every leg chunks Stage 1.** The app uses `chunk_text` at the UI defaults of 10,000 words and 1,600
  overlap. Every chunk then runs its own 3-draw ensemble: chlorophyll and menaquinone have 2 chunks,
  Ara has 2, WNT has 4, enterobactin and sour taste have 1 (`orch758_stage1_boundaries.txt`). The section
  splitter labels every chunk `preamble` with relevance 0.5, so the ordering that puts references last
  never applies (`orch758_chlorophyll_lexicon.txt`).
- **Chunks are not ensembled with each other.** `merge_stage_one_outputs` unions the chunk pools.
  Chlorophyll's **41** pooled reactions are **chunk 1's** pool. Chunk 2 (46,704 chars) pooled 0 reactions.
- **The batch drives the app as two AppTest script runs:** "Run pipeline", which does Stages 0-2, then
  the post-pipeline button, which does audit, Stage 6, freeze and export (`batch/driver.py:2805-2817`,
  `:2898-2918`).
- **Every model attempt in every leg is `deepseek/deepseek-v4-flash`.** No fallback model was called
  anywhere (`orch758_leg_timeline.txt`). Retries come from `LLM_MAX_RETRIES=3` in `.env`, nested inside each
  stage's own attempts. Backoff sleeps are 1 s and 2 s per call, which is negligible. **The time goes to
  the model reasoning up to the token cap on every empty attempt.**

## 2. Per-leg table

"Budget" means the leg would have succeeded or progressed with non-empty completions, so D-105 /
ORCH-757 cure it. "Separate" means a defect that budgets do not cure (§ 4).

| leg | terminal outcome (`LEG_TERMINAL.json`) | chain to failure, one line | verdict | classification |
|---|---|---|---|---|
| enterobactin strict | fail 2,212 s, `final_gate_report_missing` @ `final_pre_export_gate_lifecycle` | Stage-2 location row `holo-EntB` → normalizer makes it a **compound** → mapper routes it to EntB's UniProt P0ADI4 → identity admission withholds P0ADI4 from **both** claimants → EntB, a component of a surviving complex, now has no accession → strict quarantine refuses (`unexportable_entity:1`) → freeze skips the final gate → the lifecycle check fails closed on the missing report | **separate (D1)**. The ORCH-756 1-draw rerun did not trigger it and wrote `pathway.review_required.pwml` | `product_contract_violation` (§ 1: "an entity groundable through an available database", "a missing … gate report") |
| menaquinone strict | timeout 3,600 s, parent kill, 0 payload files | Stage 1 2,196 s, of which 1,046 s were 2 empty-length Stage-1 attempts at 64,000 → Stage 2 600 s (266 s empty) → audit 259 s (236 s empty) → killed in the curator | **budget**. Would finish in about 2,400 s, or about 3,450 s if the Stage-1 empties recur. **But see D8:** the initial gate already showed 8 *L. lactis* Men enzymes without a UniProt (`implausible_name_match`), and this recurs in ORCH-756 | `product_contract_violation` (budget configuration); identity: open |
| chlorophyll strict | timeout, parent kill, 0 payload files | Stage 1 **2,103 s with no empty attempt** (557+486+573 s chunk 1, 138+149+168 s chunk 2) → RAG 519 s (407 s empty) → Stage 2 **8 of 8 attempts empty**, 937 s | **budget + separate (D7)**. Even with no empty attempt the projection is about 3,400-3,900 s, against a 3,480 s child deadline | `product_contract_violation` (configuration: the leg budget is not scaled to 3 draws) |
| Ara operon strict | timeout, parent kill | Stage 1 863 s → Stage 2 2,554 s (19 of 24 attempts empty, 2,084 s) → curator 6 of 6 empty → killed | **budget**. Would finish in about 2,000 s (Ara research finished in 2,382 s) | `product_contract_violation` (budget configuration) |
| sour taste strict | fail 1,655 s @ `pwml_export`, `actor_schema_not_canonical` | Stage-1 compound `Ca++` → `_has_plus_token` reads it as a composite → strict `normalize_composites` raises → the app catches it and continues with the **un-normalized** Stage-2 payload → quarantine ok, final gate ok, release `review_required` → the exporter's validation-only revalidation rejects 3 un-canonical `OTOP1` transporter rows → no PWML | **separate (D2)**. Not a biological gate: 0 reactions is not what stopped it | `product_contract_violation` (§ 4 / § 7: a `review_required` PWML was owed; merge rule 7) |
| WNT PCP strict | timeout, parent kill | 4 chunks × 3 draws: Stage 1 2,512 s, of which 1,021 s were 2 empty-length attempts at 64,000; chunk 3 got 2 of 3 draws → RAG 652 s (447 s empty) → Stage 2 2 of 2 empty → killed | **budget + D7 + D9**. About 2,950 s without the Stage-1 empties (fits), about 3,970 s with them (does not fit) | `product_contract_violation` (configuration) |
| Wnt/Ca2+ strict | fail 1,466 s @ stage1, "Chunk 3 failed to produce valid JSON" | chunk 3 (90 % reference list) returned **valid, structurally empty JSON** (`finish_reason=stop`, 1,153-1,249 chars) on 6 of 6 calls → rung 3 refused `budget_indeterminate` → `run_ensemble` re-raises → the chunk failure aborts Stage 1, although chunks 1-2 had pooled 2+3 reactions | **separate (D3)**. Not empty-length; the budget is irrelevant | `product_contract_violation` (§ 1: "an initial empty `{}` extraction" may not end a run) |
| enterobactin research | **pass**: `research_pathway_report.txt` written | reached its terminal output in 3,448 s, 32 s under the child deadline, after spending 1,960 s on empty attempts | reached its output | — |
| menaquinone research | timeout 3,583 s, `in_process_deadline` / `operation_timeout`, 0 artifacts | Stage 2 2,236 s, 19 of 23 attempts empty (1,797 s), across 3 calls | **budget** | `product_contract_violation` (budget configuration) |
| chlorophyll research | timeout, parent kill | Stage 1 2,212 s (one draw ended `length` with 39,443 chars and was salvaged) → RAG 638 s (395 s empty) → Stage 2 3 of 3 empty | **budget + D7** (as strict) | as strict |
| Ara operon research | **pass**: report written | 2,382 s | reached its output | — |
| sour taste research | **pass**: report written | 1,970 s | reached its output | — |
| WNT PCP research | fail 3,323 s @ stage1, chunk 3 | as Wnt/Ca2+ strict; also 2 empty-length Stage-1 attempts (1,281 s) | **separate (D3)** | as Wnt/Ca2+ strict |
| Wnt/Ca2+ research | timeout, parent kill | Stage 1 2,443 s (4 chunks; one 791 s empty attempt; chunk 3 got 2 of 3 draws, chunk 4 got 1 of 3) → RAG 705 s (28 of 42 empty) → Stage 2 still running (5 non-empty calls) | **budget + D7** (marginal) | `product_contract_violation` (configuration) |

Sources: the outcome and wall time of each leg are from `LEG_TERMINAL.json` and `RESULT.txt`. Stage seconds and empty counts are from `orch758_leg_timeline.txt`. Chunk and draw outcomes are from `orch758_stage1_boundaries.txt`. **None of the 7 strict legs failed for a reason a correct biological gate should cause.**

## 3. The seven timeouts: where the time went

From `orch758_leg_timeline.txt`. A trace stamps each attempt when it **ends**, so an attempt's duration is
the gap from the previous event. Non-LLM work (DB mapping, gates) hides inside those gaps. The trace has
no per-stage begin/end events below `driver`.

| leg | Stage 1 s (empty s) | downstream done at kill, s (empty s) | killed in | time on empty-length attempts, whole leg | est. total with non-empty completions | fits 3,480 s? |
|---|---|---|---|---|---|---|
| menaquinone strict | 2,196 (1,046) | 1,404 (708) | curator, after 3 audit rounds | 1,754 s | ~2,400 s; ~3,450 if the Stage-1 empties recur | yes / marginal |
| menaquinone research | 639 (0) | 2,944 (1,983) | Stage-2 JSON repair, 3rd call | 1,983 s | ~2,100 s | yes |
| chlorophyll strict | **2,103 (0)** | 1,497 (1,344) | Stage 2 (no call ever succeeded) | 1,344 s | ~3,400-3,900 s | **no** |
| chlorophyll research | **2,212 (0)** | 1,389 (1,024) | Stage 2 | 1,024 s | ~3,500-4,000 s | **no** |
| Ara strict | 863 (0) | 2,737 (2,194) | curator | 2,194 s | ~2,000 s | yes |
| WNT PCP strict | 2,512 (1,021) | 1,088 (795) | Stage 2 | 1,816 s | ~2,950 s; ~3,970 with Stage-1 empties | marginal / no |
| Wnt/Ca2+ research | 2,443 (791) | 1,158 (554) | Stage 2 (research rounds) | 1,345 s | ~3,050 s | marginal |

**How the estimates were made, and how uncertain they are.**
- Total = Stage 1 (measured, minus its empty seconds) + downstream (measured, minus its empty seconds) +
  what had not run yet.
- "What had not run yet" is priced from completed legs. The ORCH-756 1-draw enterobactin leg took **562 s
  from the end of Stage 2 to the end of the leg** (audit, curator, gap, Stage 6, freeze, export). The
  2026-09 1-draw strict passes spent a median of about 975 s downstream, net of empties, over a range of
  778-1,852 s (`orch758_reference_timing.txt`).
- Chlorophyll's payload is about twice the size (41 rows), so its downstream is priced at the top of that
  range.
- These are **projections ±25 %, not measurements.**

**Why the retries cost so much.** An empty `finish_reason=length` reply is retried as a transient with the
**identical** request (`llm/client.py:688-737`), and each retry reasons to the cap again. That runs inside
the stage's own attempts, so one Stage-2 call can cost up to 9 identical calls. Chlorophyll strict's
single Stage-2 request hash `…221e3f` was sent 8 times, for 937 s. Retries did sometimes recover: for
menaquinone strict request `…860907` the sequence was empty, empty, then ok.

**Two budget-adjacent findings the D-105 fix does not cure:**
- **Stage-1 empty-length at 64,000: 8 of 111 attempts, each costing 327-946 s** (`orch755_trace_stats.txt`;
  durations in `orch758_leg_timeline.txt`: menaquinone strict 476 s + 570 s, on a **16,202-char** chunk;
  WNT PCP strict 694 + 327; WNT PCP research 613 + 669; Wnt/Ca2+ research 791; sour research 946). D-105
  keeps extraction at 64,000. See D9.
- **The hardcoded budgets are still empty after D-105's `.env` change.** In ORCH-756, at 1 draw and 128k,
  enterobactin's Stage 2 succeeded on its first attempt (142 s, 15,029 chars), but its **audit loop
  returned 5 of 6 attempts empty** (`orch756/.../enterobactin-strict/.../LEG_TRACE.jsonl`, 1,232-1,509 s). That
  is ORCH-757's scope (the audit `min(10000, …)`/1,200 budgets), and it confirms ORCH-757 is needed.

## 4. Separate defects (numbered findings)

"Blocks PWML" means it blocks PWML production **after** the budget fix.

### D1 — enterobactin: an inferred alias strips a verified enzyme's UniProt, and the strict quarantine then refuses the whole pathway
**Blocks PWML: yes, when triggered.** The trigger is a Stage-2 output shape, so it is intermittent. It did
not recur in ORCH-756's enterobactin leg, which wrote `pathway.review_required.pwml` (83,687 bytes, 7 reactions).

Chain, with evidence from `runs_papers/ORCH-755/legs/enterobactin-strict/2026-10-20_2201/papers/PMC12096016/strict/`:
1. Stage 2 adds a protein-location row for `holo-EntB`, with evidence "…EntD to form holo-EntB". No such
   entity exists. Source: `merged_payload.json`; `stage1_payload.json` has 0 occurrences.
2. The normalizer resolves the dangling location token. `holo-EntB` is not protein-like to it, so it
   becomes a **compound** with `confidence 0.8, provenance inferred`. Code: `process_normalizer.py:1151`
   `_rewrite_token` → `:1188-1191` → `_ensure_compound` `:922-934`. The initial gate's normalization stats
   show `entities_added_as_compounds: 3` (`gate_fail_report.json`).
3. The mapper calls the same name `known_protein_like` (`pipeline/entity_identity.py:417`). It routes the
   compound to UniProt and takes **P0ADI4, EntB's own accession**, through the literature alias "EntB"
   (`final_mapped.json` `/entities/compounds/16/mapping_meta`).
4. Identity admission, pass B, finds P0ADI4 claimed by a compound and by a protein under different names,
   and withholds it from **every** claimant: "no claimant may ship it". Code: `mapping/map_ids.py:9940-9975`,
   `identity_admission.py:806-855`. EntB's lineage records "identity admission withheld uniprot=P0ADI4
   (accession_claimed_across_incompatible_entity_kinds)". EntB keeps only `pathbank_protein_id 6224`.
5. The strict quarantine removes `holo-EntB` as unreferenced (`removed_entity_report.json`). It does not
   restore EntB's accession, because admission ran at Stage 6, before this point. EntB stays alive as the
   component of PathBank complex 1189 `isochorismatase`, so it is reported unexportable
   (`protein_missing_external_identity`). The quarantine refuses with `refusal_reasons ["unexportable_entity:1"]`,
   `ok false` (`quarantine_report.json`). Code: `strict_quarantine.py:2097-2124`, `:2623`.
6. Freeze: `freeze_canonical_payload` runs the final gate only `if quarantine_ok`
   (`streamlit_app.py:3280-3288`). By design (comment `:3339-3343`), a refusal writes no final report.
   `gate_verdict` then fails closed on the missing report (`gate_reports.py:382-384`), and the driver files
   it as `final_gate_report_missing` (`batch/driver.py:3014`).

**What is wrong.** EntB is grounded and verified (`identity_verdict.verified true`, PathBank 6224 /
UniProt P0ADI4). The conflict exists only because the pipeline gave one inferred name two kinds:
compound to the normalizer, protein to the mapper. The "nobody ships it" resolution then penalises the
legitimate claimant. Its trigger is gone by step 5, but the penalty stays. Then a 4-locked-reaction core
(`locked_reactions_found 4`) became `diagnostic_only` instead of `review_required`.

**Budget link: indirect.** The row came from a Stage-2 reply truncated at 16,000 tokens and repaired
(`stage2_inference invalid_json finish=length 6724 chars → stage1_json_repair ok`,
`orch758_stage1_boundaries.txt`). But steps 2-6 are deterministic. Any Stage-2 reply that names
`holo-EntB` in a location row reproduces the failure at any budget.

**Not ORCH-754 code.** Complex 1189 is a PathBank complex, not a 754b wrapper. Rule 3 never ran (D4). The
Stage-1 ensemble was clean: 4 pooled reactions, 6 exact merges.

### D2 — sour taste: `Ca++` crashes normalization, the run continues un-normalized, and the exporter's schema check stops a review_required pathway
**Blocks PWML: yes.** It is deterministic for any strict leg whose Stage 1 writes an ion as `X++`. More
generally, it fires on any exception inside `normalize_process_payload`.

1. Stage 1 wrote the compound `"Ca++"` (the paper writes `Ca ++`; `stage1_payload.json`).
2. `_has_plus_token` (`process_normalizer.py:355-360`) strips one trailing `+`, or `<digits>+`, so `Ca++`
   becomes `Ca+` and still reads as a composite. In strict mode `normalize_composites` raises at `:1544-1546`
   with "Composite entity 'Ca++' … no protein-like left component". The comment at `:1532` says "Strict mode
   aborts the whole run here".
3. **It does not abort.** The app catches the exception (`streamlit_app.py:3655`) and keeps
   `normalized_input = stage2_payload` (`:3524`), the un-normalized Stage-2 output. `gate_fail_report.json`
   holds only that `error` string, the exception-branch shape.
4. `normalize_process_actor_schema` (`process_normalizer.py:2742`, transporters `:3100-3106`) never ran. The 3
   OTOP1 transporter rows stay in the Stage-1 shape `{"protein": "OTOP1"}` from `stage1_payload.json` through
   `final_mapped.json`. In every other leg these rows became `{entity, entity_type, protein}`.
5. Quarantine was `ok` with 7 core processes. Release was **`review_required`**, the correct state for
   coverage 0.333 < 0.5 (`quarantine_report.json`). The final Stage-3 gate passed with **0 reactions, 6
   transports, 1 interaction, 1 component** (`final_stage3_gate_report.json`). It validates and does not
   normalize (`process_normalizer.py:4595`).
6. `run_pwml_export` → `_validate_stage8_export_payload` (`streamlit_app.py:4595-4608`) →
   `validate_post_normalization` → `_validate_canonical_actor_rows` (`stage_contracts.py:792-821`). It flags
   `actor_schema_not_canonical` on `/processes/transports/{0,2,4}/transporters/0` and returns at
   `streamlit_app.py:4739-4776`: "PWML export stopped by validation-only pre-export Stage 3 revalidation"
   (primary `outputs/pwml_stage3_gate_report.json`, 02:07).
7. The same findings were "informational, not blocking" in the post-audit report: 10
   `superseded_contract_errors` (`RESULT.txt` warnings). Two consumers of one contract disagree.

**Verdict.** This is not a biological gate. The paper being regulatory with 0 reactions did not stop it.
The pipeline reached the correct `review_required` decision, and then a technical defect suppressed the
PWML. That is the merge-rule-7 failure, and contract § 1 lists it as an unacceptable terminal blocker. Not
budget. Not ORCH-754 code.

### D3 — WNT: a chunk that is mostly the reference list correctly returns nothing, and an empty chunk is fatal to the paper
**Blocks PWML: yes,** for long papers whose reference list fills a chunk. Two of the four WNT legs died
here; the other two got through chunk 3 by chance, on 2 of 3 draws.

- **Which chunk.** Chunk 3 of 4: 75,671 chars, chars 109,017-184,688 of the 199,748-char paper. The
  "References" heading is at char 116,372, so **90 % of chunk 3 is the reference list**. Its first 7,355 chars
  are body: the end of the disease section, Conclusion, Conflict of interest and Acknowledgements. Its "mechanism" sentences
  are reference titles (`orch758_wnt_chunk3_terms.txt`, `orch758_chlorophyll_lexicon.txt`).
- **Which draws.** Wnt/Ca2+ strict: draws 1-3, each with 2 calls (`normal`, then
  `structurally_empty_repair`). All 6 calls returned `outcome=valid_json_zero_processes`,
  `finish_reason=stop`, 1,153-1,249 chars, 0 entities, 0 processes. Request hashes `…9723e8` and `…5857e1`
  (`orch758_stage1_boundaries.txt`, from `run-4a18f108e2a3`). WNT PCP research had the same on its chunk 3
  (`run-d2a781b69084`).
- **This is not the empty-length failure.** No attempt ended in `length`, and the 64,000 budget is
  irrelevant. ORCH-751 § 2.3's empty WNT draw was a different mechanism: an unchunked full-paper draw that
  returned 0 and 0 and was not fatal.
- **This is not a JSON-repair gap.** The JSON was valid. The message "Chunk 3 failed to produce valid JSON"
  (`pipeline.py:2710-2715`) misreports it.
- **Fallback model: not tried.** Rung 3, `materially_different_strategy`, which would use
  `OPENROUTER_EXTRACTION_FALLBACK_MODEL` (set in `.env`), was **refused `budget_indeterminate`**: "no
  LegDeadline is in scope". `ExtractionLadder` takes `current_leg_deadline()` (`extraction_ladder.py:360`),
  and `activate_leg_deadline` (`:244`) has **no caller** in `src/` or `scripts/`. This is the known FINDINGS
  **P2-04** (HIGH, open). ORCH-739 counted rung 3 at "zero times issued". An alternate model would not help
  a references chunk anyway: an empty answer is the right one.
- **The fatal path.** Each chunk draw is `retry_on_empty_payload=True` (`pipeline.py:293`). The draw raises
  "structurally empty payload … on all 2 attempts" (`:4217-4222`). `run_ensemble` re-raises when every draw
  fails (`stage_one_ensemble.py:1033-1034`). The ensemble raises **before** it writes a side file, so the
  failed chunk leaves only boundary records. `run_stage_one_with_chunking` then turns that one chunk into a
  Stage-1 failure for the whole paper (`pipeline.py:2707-2715`).

**Not budget. Not ORCH-754 code.** The ensemble makes this failure less likely: it needs 3 of 3 failures, not 1 of 1.

### D4 — D-104 rule 3, the Stage-6 database-ID merge, never runs in the app
**Blocks PWML: no.** But D-104 § 3(c) and AMENDMENT 5 are **not in effect**. Synonym duplicates that rules 1
and 2 deliberately leave to it reach the PWML un-merged. Chlorophyll examples: `protoheme`/`heme b`,
`Mg-Proto IX`/`Mg-protoporphyrin IX`, `Chl a`/`chlorophyll a`; see § 5. This is ORCH-754a code.

- **Mechanism.** `merge_on_mapped_identity` applies only when `_pooled_draws()` > 1
  (`stage_one_ensemble.py:1116-1128`). That reads `current_diagnostics().boundaries`, a **ContextVar**
  (`extraction_diagnostics.py:872-889`). The Stage-6 `map_payload(..., merge_identity_duplicates=True)`
  (`streamlit_app.py:4248-4262`) runs in the **post-pipeline script run**. AppTest builds a new
  `LocalScriptRunner` for every run (`streamlit/testing/v1/app_test.py:395`). Each one starts a new
  `threading.Thread` (`streamlit/runtime/scriptrunner/script_runner.py:341`). On Python 3.13.6 a new thread
  starts with an empty context, so Stage 6 sees a fresh detached recorder, `_pooled_draws()` is 0, and rule 3
  returns `{}` silently.
- **Proof** (behavioural, offline). `orch758_rule3_probe.py` drives a two-button app through AppTest exactly
  as `batch/driver.py` does. In the Stage-1 run: `pooled_draws 3, rule3_applied true`. In the next run:
  `artifact_dir "", ensemble_boundaries 0, pooled_draws 0, rule3_applied false` (`orch758_rule3_probe.txt`,
  G11 01).
- **Corroboration.** There is no `ensemble_identity_merge_report.json` and no "DATABASE-IDENTITY MERGE"
  text in any of the **312** `tmp/extraction_diagnostics/run-*` directories, although 5 ORCH-755 legs
  reached Stage 6. The post-pipeline also records audit and gap rounds into the same recorder
  (`streamlit_app.py:2111-2125`, `:4131`), and **0 of 312** directories hold `audit_iteration_summary.json`,
  `gap_iteration_summary.json` or `mapped_failure_snapshot.json`. That diagnostics loss predates this sprint
  wave.
- **Live UI.** Not tested. Streamlit also serves each rerun from a ScriptRunner thread, so the same loss is
  expected there.

### D5 — the paper's own shared-locant phrase is not read as the pair the lexicon derived from it
**Blocks PWML: no.** It inflates review rows and, with D4, passes duplicate reactions to the PWML. 5 of
chlorophyll's 41 rows. ORCH-754a code.

- The paper writes `uroporphyrinogen (Urogen) III`. `abbreviation_pairs` turns that exact phrase into the
  pair `Urogen III` = `uroporphyrinogen III` (`shared_locant`; `paper_lexicon.py:155-157`).
- Draw 1 copied the phrase verbatim, as a participant name, for 4 molecules:
  `uroporphyrinogen (Urogen) III`, `coproporphyrinogen (Coprogen) III`, `protoporphyrinogen (Protogen) IX`,
  `protoporphyrin (Proto) IX`.
- `long_form` handles `X (Y)` only when the parenthesis is **last**: `_TRAILING_PAREN`
  (`paper_lexicon.py:61`), used at `:253-261`. So none of the 4 is rewritten.
  - Proof at this SHA: production `canonical()` returns each name unchanged with rules `[]`, while the
    lexicon holds the pair (`orch758_chlorophyll_lexicon.txt`).
  - D-104 § 3(a): a draw-written `X (Y)` bridges if "the paper defines that pair". The paper does, with this
    very phrase.
- Counterfactual, not production code: read `X (Y) L` as `Y L` = `X L`. That renames those 4 keys and
  takes the 41 rows to 32 distinct keys, an upper bound.
  - It pairs R5+R23, R6+R24, R7+R25(+R26), R8+R27(+R28) and R9+R30.
  - The enzymes resolve compatibly in 5 of those merges. Draw 1 names UROS, UROD, CPO, PPO and Mg-chelatase,
    as draws 2-3 do (the proteins bucket of the side file).
  - So the realistic effect is **41 → about 36**. R26 and R28 stay apart on CPO1/PPO1 (§ 5).

### D6 — a paper-defined abbreviation the Schwartz-Hearst match cannot see
**Blocks PWML: no. Low. Implementation ceiling.**
- The paper writes `3-vinyl Chlide a (MV-Chlide a )`: MV = monovinyl, and "M" appears nowhere in the long
  form.
- Schwartz-Hearst rejects the pair. `abbreviation_pairs` yields `DV-Pchlide` and `DV-Chlide` but no
  `MV-Chlide a` pair (`orch758_chlorophyll_lexicon.txt`).
- Draws wrote all three forms (R15, R31, R38; R17, R33, R40). D-104 § 3(a)'s authority is "the paper
  defines"; the chosen algorithm is narrower than the rule.

### D7 — the 3-draw Stage 1 does not fit the 3,600 s leg on long papers, and a timed-out leg preserves nothing
**Blocks PWML: yes for the 3-draw re-run on long papers, if leg timeouts stay at 3,600 s. No for ORCH-756**
(1 draw, `timeout 5400s` per its `batch.log`).

- **Stage-1 cost.** At 3 draws Stage 1 took **565-3,323 s, median about 1,400 s** across ORCH-755's 14 legs.
  The 2026-09 1-draw runs took **42-1,044 s, median about 230 s** (`orch758_reference_timing.txt`). The draws
  run **sequentially** (`run_ensemble`, `stage_one_ensemble.py:1023-1032`), once per chunk.
  - Chlorophyll strict: 2,103 s, with no empty attempt, before Stage 2 started.
  - D-104 § 2 accepted "the 3x Stage-1 cost", but the § 9 leg budget was not raised for it. § 9 allows an
    explicit, recorded override.
- **Calls past the child deadline.** Children issued LLM calls after their own 3,480 s deadline: menaquinone
  strict's last call ended at 3,569 s, chlorophyll strict's at 3,559 s. § 9 requires "one monotonic per-leg
  deadline" checked before each LLM call, and in production no `LegDeadline` is active (D3, P2-04).
- **Nothing preserved.** 6 of the 7 timeouts were `outer_parent_kill` with `payload_file_count 0`. The
  7th, menaquinone research's `in_process_deadline`, left `artifact_count 0` (`LEG_TERMINAL.json`).
  - Contract § 9 "On timeout … preserve: … current structured payload" is unmet. So is § 1, "a timeout
    without usable checkpoints".
  - Chlorophyll's 41-row Stage-1 pool, 34 minutes of work, survives only in the side file under `tmp/`.
- `product_contract_violation` (configuration and § 9).

### D8 — menaquinone: 8 *L. lactis* Men enzymes have no admitted UniProt (risk, not yet a failure)
**Blocks PWML: likely, by the D1 route (unexportable enzymes, so the strict quarantine refuses), unless
the audit or gap stages repair it.** The ORCH-755 leg was killed before the quarantine.

- The initial Stage-3 gate failed with 8 errors, "Protein 'Men{F,D,H,C,E,B,I,G}' is missing a UniProt or
  DrugBank identifier". Mapping ended `novel (implausible_name_match)` (primary
  `tmp/post_pipeline_3ded14a704d24183a8ab5b8d85ac864e/gate_fail_report.json`, 23:34). This is the leg's
  surviving post-pipeline scratch.
- The audit's truncated LLM reply was proposing exactly the repair: "MenF has a high-confidence UniProt hit
  (A2RM72) but identity_verdict was rejected only because the score margin is below 0.1" (`audit_report.json`,
  same directory). Its 1,200/10,000 budgets are ORCH-757's scope.
- **It recurs at 1 draw with the D-105 budgets.** ORCH-756's menaquinone strict leg, still running at
  21:04, failed its initial gate in the same way. This time 9 Men proteins (MenF, D, H, C, E, B, I, A, G),
  organism *L. lactis* subsp. *lactis* KF147, all `provider UniProt, chosen_rule novel_protein, issue
  implausible_name_match, order_step name_plausibility_gate`, with 0 candidates
  (`orch756/tmp/post_pipeline_f7541ff2e0494f3fb0391eb4dc6de9fc/gate_fail_report.json`). Its audit was again
  empty-length (4 of 4 attempts by 884 s), so nothing repaired it. **Not budget-caused.**
- Classification is open: mapping policy (the margin rule) against biology. It needs `pwml-bio-auditor`
  before any code.

### D9 — the extraction budget is capped for a fallback that is never called
**Blocks PWML: no. Cost: 300-950 s per occurrence (§ 3).**
- D-105 § 1 keeps `OPENROUTER_EXTRACTION_MAX_TOKENS` at 64,000 because the fallback model caps output at
  65,536.
- Stage 1 never fails over: rung 3 is unreachable (D3), and the LLM client has no model fallback on empty
  replies (`llm/client.py:681-737` retries the same `resolved_model`).
- All 1,000+ ORCH-755 attempts went to deepseek.
- The 64,000 cap therefore protects a path that does not execute, while costing 8 empty Stage-1 attempts a
  night. This is a product question (§ 9 item 4).

### D10 — minor diagnostics gaps found on the way
- A wholly failed chunk ensemble writes no side file (it raises at `stage_one_ensemble.py:1034`, before the
  write at `:1042`). That is why ORCH-755 counted "3/3 draws on 14 of 14 reports" (§ 6 C2).
- LEG_TRACE has no stage begin/end events below `driver`, so non-LLM stage time (DB mapping, gates) cannot be
  separated from LLM time.

## 5. Chlorophyll strict: 41 pooled, 0 merges flagged

Side file: `evidence/orch758_inputs/chlorophyll_strict_stage1_ensemble_report.txt`, report #1 of 2. Report #2
is chunk 2: 0 reactions, 8 pairs.

**"0 merges flagged" does not mean the merge never ran.**
- The N-draw path ran with 3 of 3 draws succeeding, 56 abbreviation pairs and 18 names rewritten.
- It made **14 merges**: 12 bridged by abbreviation, 2 exact.
- It also kept apart 4 pairs, each with a recorded reason.
- "Flagged" counts only `undefined_state_treated_as_absent` and `typo_snapped` merges
  (`stage_one_ensemble.py:944-946`). No ORCH-755 draw needed either: every state named was defined in its
  own payload, and no typo snapped.
- ORCH-754a's replay had 11 flagged merges, because ORCH-751 draw 2 used an undefined `plastidic state`.

**Single draws had 18, 19 and 18 reactions, not ~31.** `rows_per_draw [18, 19, 18]`. The "31" in
ORCH-755-RESULT § 3 and in this task's brief is the number of rows found by **one draw only** (§ 6 C1).
55 draw rows − 14 merges = 41.

**Why 41 here and 29 in the ORCH-754a replay.**
- **The code is the same, and so is the input regime; the draws differ.** Re-pooling ORCH-751's draws with
  chunk 1 as the dictionary gives **28**, against 29 with the full paper (`orch758_replay_dictionary_swap.txt`).
  So chunking does not explain it. It slightly helps: in chunk 1, `POR` is unambiguous.
- ORCH-755's draws named molecules differently:
  1. Draw 1 copied the paper's `X (Y) L` phrases. The lexicon cannot read them (D5): 5 rows.
  2. Draw 3 expanded or used synonyms the paper never defines as abbreviations: `Mg-protoporphyrin IX`,
     `heme b`, `3-vinyl Chlide a`, `chlorophyll a/b`. Those are rule-3 territory, and rule 3 is dead (D4).
  3. Draws 2-3 named isoforms (`PORB`, `CPO1`, `PPO1`) where draw 1 named the enzyme (`POR` → "Pchlide
     oxidoreductase", `CPO`, `PPO`). AMENDMENT 3 keeps non-nested enzyme sets apart.

**Per-group verdicts.** Rows are numbered as in the side file. 41 rows describe about 21 distinct reactions.

| group (enzyme) | rows (draws) | verdict | why |
|---|---|---|---|
| GSAT, ALAD, PBGD, CAO | R2, R3, R4, R19 (3/3 each) | merged ✓ | abbreviation / exact bridges worked |
| GluTR | R1 (d1 `Glu-tRNA`), R22 (d2,d3 `Glu-tRNAGlu`) | **same reaction, no D-104 rule bridges it** | the paper writes `Glu-tRNA Glu` (a flattened superscript). `Glu-tRNA` occurs in the paper, so rule 2 never snaps it. Rule 3 would bridge it but is dead (D4) |
| UROS | R5 (d1), R23 (d2,d3) | **missed bridge = defect D5** | d1's `uroporphyrinogen (Urogen) III` is the paper's defining phrase. Enzyme UROS on both |
| UROD | R6 (d1), R24 (d2,d3) | **missed bridge, D5** | as above |
| CPO | R7 (d1), R25 (d2), R26 (d3) | R7/R25 **missed bridge, D5**. R25/R26 **kept apart, correct under AMENDMENT 3 § 2** | `{coprogen iii oxidase}` vs `{cpo1}` are not nested sets. Biologically CPO1 *is* the Arabidopsis CPO gene: a **policy question** (isoform vs enzyme) |
| PPO | R8 (d1), R27 (d2), R28 (d3) | R8/R27 **missed bridge, D5**. R27/R28 **kept apart per rule** | `{ppo}` vs `{ppo1}`; same policy question |
| Mg-chelatase | R9 (d1), R30 (d2), R36 (d3) | R9/R30 **missed bridge, D5**. R36 **rule-3 territory** | d3's product `Mg-protoporphyrin IX`: the paper uses `Mg-Proto IX` undefined (the long form appears only in reference titles), so rules 1-2 cannot bridge it, by design |
| Mg-Proto IX methyltransferase | R10 (d1,d2), R37 (d3) | **rule-3 territory** | same `Mg-protoporphyrin IX` spelling |
| Mg-Proto ME cyclase | R11 (d1), R12 (d2,d3) | **kept apart, correct under AMENDMENT 3 § 1** | `biological_state location 'thylakoid membrane' vs 'plastid'`: two *defined* states differ. Biologically nested (the thylakoid is inside the plastid): **policy question** |
| POR | R13 (d1 `POR`→Pchlide oxidoreductase), R14 (d2,d3 `PORB`) | **kept apart, correct per rule** | `{pchlide oxidoreductase}` vs `{porb}`, family vs isoform: **policy question** |
| DVR | R15 (d1), R31 (d2), R38 (d3) | **missed bridge, D6** (paper-defined, SH ceiling) | products `3-vinyl Chlide a (MV-Chlide a)` / `MV-Chlide a` / `3-vinyl Chlide a` |
| CHLP | R16 (d1 chain→chain), R32 (d2 geranylgeranyl→phytyl), R39 (d3 geranylgeraniol→phytol) | **genuine variance; correct to keep apart** | different participants, not spellings. A reviewer picks one. The free-alcohol form is chemically the weakest |
| Chl synthase (GG) | R17 (d1), R33 (d2), R40 (d3) | **same reaction; needs D6, and `Chl a` ↔ `chlorophyll a` is rule-3 territory** | the paper defines `Chl` = chlorophyll but never `Chl a`. Rule 1 matches whole names only |
| Chl synthase (phytol) | R18 (d1) | **genuine distinct reaction** | different substrate (phytol vs geranylgeraniol) |
| CAO on Chl a | R20 (d1) | **genuine distinct reaction** | the paper: "Some of Chlide a or Chl a is converted into Chl b" |
| GluRS | R21 (d2) | genuine singleton | Glu + tRNA → Glu-tRNA |
| ferrochelatase | R29 (d2 `protoheme`), R35 (d3 `heme b`) | **rule-3 territory** | the paper writes `protoheme (heme b )`. `heme b` is a synonym, not an abbreviation, and SH cannot pair it |
| heme oxygenase | R34 (d2), R41 (d3) | **rule-3 territory** | same `protoheme`/`heme b` |

**Tally of the 20 surplus rows** (41 rows − 21 reactions):
- about 6 are **missed bridges by D-104 rule-1 defects**: D5 × 5, plus part of D6;
- 4 are **kept apart correctly under AMENDMENTS 3 § 1-2**: the policy questions in § 9 item 6;
- about 6 are **synonyms designed for rule 3**, which never runs (D4);
- about 3 are **genuine variance**: CHLP's three representations, plus residue of the Chl-synthase group.

**No row is a conflict merged in error. Nothing was dropped:** 31 rows carry `review_required: found by one draw only`.

## 6. Corrections to ORCH-755-RESULT (it stays as written; corrected here)

- **C1.** § 3 "chlorophyll 41 (strict; 31 single-draw)": single draws had **18/19/18** reactions. 31 is the
  number of 1/3-support rows.
- **C2.** § 3 "3/3 draws succeeded on 14 of 14 reports (one later chunked draw lost 2 of 3)": the side files
  record only ensembles that produced output.
  - WNT chunk 3 got **0 of 3** draws in Wnt/Ca2+ strict and WNT PCP research, and those ensembles raised
    before writing (D10).
  - It got **2 of 3** in WNT PCP strict and Wnt/Ca2+ research.
  - Chunk 4 got **1 of 3** in Wnt/Ca2+ research (`orch758_stage1_boundaries.txt`).
- **C3.** § 3 "No merge was flagged in any leg" is true, but it does not mean no merge: every leg with
  reactions merged, e.g. chlorophyll 14 and menaquinone 18 exact (`orch758_stage1_boundaries.txt`).
- **C4.** § 4's headline is necessary but not sufficient. Four strict legs fail for reasons budgets do not
  touch (§ 2).

## 7. Theories tested and disproved

| theory (what was believed) | disproved by |
|---|---|
| "0 merges flagged" means the N-draw merge never ran, or the wrong flag was set | side file: 3/3 draws, 14 merges by rule, 4 kept apart with reasons. The flag counts only two review conditions (§ 5) |
| the WNT chunk failure is the empty-length failure | all 6 chunk-3 calls `finish_reason=stop`, 1,153-1,249 chars, valid JSON (`orch758_stage1_boundaries.txt`) |
| the WNT chunk failure is a chunk-JSON repair gap | the JSON parsed. It was structurally empty, and the message "failed to produce valid JSON" is wrong |
| sour taste was stopped by a biological gate on a 0-reaction regulatory paper | the final gate passed with 0 reactions, quarantine `ok`, release `review_required`. The stop was a schema check (D2) |
| enterobactin's final gate failed | it never ran. The quarantine refused, and the missing report is the designed fail-closed symptom (D1 step 6) |
| enterobactin was broken by the 754b wrappers or the 754a identity merge | EntB's complex is PathBank 1189, not a generated wrapper; rule 3 never ran (D4); the Stage-1 merge was clean |
| the hour went on backoff sleeps | sleeps are ≤ 3 s per call (`LLM_MAX_RETRIES=3`, base 1 s). The time is model reasoning to the cap |
| with budgets fixed every timeout finishes | chlorophyll's Stage 1 alone took 2,103 s with no empty attempt (D7) |
| ORCH-755's 41 vs ORCH-754a's 29 comes from chunking (a different dictionary text) | ORCH-751's draws pool to 28 with the chunk-1 dictionary and 29 with the full paper (`orch758_replay_dictionary_swap.txt`). The difference is in the draws |
| the Stage-6 identity merge ran and found nothing | no identity side file anywhere (0 of 312), and the AppTest probe shows it cannot apply in the post-pipeline run (D4) |
| the extraction fallback model was tried | every attempt is deepseek. Rung 3 was refused `budget_indeterminate`, and the client has no fallback on empty replies (D3, D9) |

## 8. What is NOT settled

- **D8 (menaquinone identities)** is observed at the initial gate in both nights (ORCH-755 8 proteins,
  ORCH-756 9). The terminal outcome is not yet observed: the ORCH-756 leg was still running when this was
  written. Whether the audit at ORCH-757's budgets repairs it is unknown.
- **D1's recurrence rate.** It needs a Stage-2 reply that names `holo-EntB` in a location row. It happened
  1 time in 2 ORCH-755 enterobactin legs (strict only) and 0 times in 1 ORCH-756 leg.
- **D4 in the live Streamlit UI** was not tested, only the AppTest/batch path.
- The **§ 3 time projections** are ±25 % estimates. Non-LLM stage time is not separable in LEG_TRACE (D10).
- **The D5 counterfactual** is key-level. The compatibility rule could not be re-run, because the per-draw
  Stage-1 payloads are not kept anywhere: only response hashes are. The raw pooled chunk payload goes to
  `tmp/stage1_raw…` and is overwritten by the next leg.

## 9. Open for the product owner

1. **D2 (`Ca++` crash, then un-normalized export).** It blocks sour taste after the fix.
   - (a) Make `_has_plus_token` strip `++` and similar charge suffixes. That is one narrow regex: a lane,
     a G9 proof (`Ca++` fails at base) and a review. About 2 h.
   - (b) Also make a normalization exception in strict mode stop or degrade the run explicitly, instead of
     continuing with an un-normalized payload. This touches the protected `streamlit_app.py` (`:3655`), so it
     needs a D-xxx. About half a day.
   - (c) Do nothing: every strict paper that writes `X++` gets no PWML.
2. **D1 (alias strips EntB).**
   - (a) Make the normalizer and the mapper agree on the kind of a dangling name: create it as a protein
     where the mapper would route it as one. About half a day, with a review.
   - (b) Let a kind conflict whose compound claimant is `inferred` and the protein `verified` resolve in the
     protein's favour, instead of "nobody ships". This is a mapping-policy change: it needs a ruling and a
     bio-auditor. About half a day.
   - (c) Restore withheld identities when the conflicting claimant is later removed. About half a day.
   - Cost of nothing: intermittent loss of whole strict pathways.
3. **D3 (an empty references chunk is fatal).**
   - (a) Accept a structurally empty chunk when another chunk produced content; record it, do not fail.
     Small, in `pipeline.py:2707-2715`. About 2 h plus review.
   - (b) Fix section detection, or drop the reference list before chunking. Larger.
   - (c) Raise the chunk size so that ≤ 200 KB papers are one chunk (ORCH-751 ran this paper unchunked
     without failure). This is a UI default, but it changes every leg's Stage-1 input and needs a
     measurement.
   - (d) Answer P2-04: does "no deadline" mean the § 9 default budget? That makes rung 3 reachable. It
     would not fix references chunks.
4. **D9 (extraction budget).** The fallback that caps extraction at 64,000 is never called.
   - (a) Raise extraction to 128k for deepseek (its limit is 943,717) and accept a fallback that can never
     fire anyway.
   - (b) Keep 64,000 and accept 300-950 s per Stage-1 empty, about 8 a night.
   - (c) Make the fallback real first (P2-04). Then the cap is meaningful.
5. **D7 (3-draw re-run, D-105 § 3(b)).** Before the 3-draw re-run, set an explicit, recorded leg timeout
   (§ 9 permits it).
   - The measured need is about 4,000 s for chlorophyll even with every completion non-empty. ORCH-756
     already uses 5,400 s at 1 draw.
   - Alternatively, parallelise the draws. That is code; draws are sequential today.
   - Also decide whether a timed-out leg must persist its Stage-1 payload (§ 9 "preserve"). Today it saves
     nothing.
6. **D4 (rule 3 dead).**
   - (a) Carry the "this run pooled" fact across script runs (session state, or the side-file directory the
     run already records) instead of a ContextVar. This is ORCH-754a's code: lane, G9 proof (the probe
     fails at base), review. About half a day.
   - (b) Declare rule 3 out of scope and keep the duplicates for hand review.
   - The same carrier would also land the post-pipeline audit/gap diagnostics that have never reached their
     directory.
7. **AMENDMENT 3 policy questions (chlorophyll, 4 rows):**
   - should an enzyme family and its isoform (`POR`/`PORB`, `CPO`/`CPO1`, `PPO`/`PPO1`) be compatible?
   - should nested locations (`thylakoid membrane` ⊂ `plastid`) be compatible?
   - Either change is a merge-rule amendment. Cost of no change: 4 duplicate review rows per such paper.
8. **D8 (menaquinone: no Men enzyme gets a UniProt; recurs at 1 draw).**
   - (a) Route it to `pwml-bio-auditor`: are the *L. lactis* KF147 Men proteins genuinely in UniProt under
     these names, so that the `name_plausibility_gate` refusal is a defect? Read-only, about 1 h.
   - (b) Wait for ORCH-757's audit budgets. The ORCH-755 audit reply was already proposing the A2RM72
     patch for MenF.
   - (c) Accept it: menaquinone strict cannot produce a PWML until its identities resolve. Research mode is
     unaffected.
9. **D5/D6 (lexicon).** Reading the verbatim `X (Y) L` phrase is a small, ORCH-754a-scoped fix (about 2 h
   plus review, worth about 5 rows on chlorophyll). The SH ceiling (D6) is lower value. Accept it, or allow
   a paper-defined `long (SF)` pair that fails the character match. A ruling is needed: it loosens rule 1.
