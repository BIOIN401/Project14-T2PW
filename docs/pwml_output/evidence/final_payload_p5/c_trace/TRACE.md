# C (D-W21c): equal-side transports, stage-by-stage trace and extraction-fix candidates

Read-only investigation. No source edited, no live LLM call, no Docker. Method: the stored runs under
`C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/` (called `PRIMARY/` below), the sources in `C:/t/fp/src/t2pw/`.
Helper scripts (not committed, scratch): `C:/t/p5scratch/ctrace/{dump2,scan,scan2,scan3,scan4,timing,timing2,trace}.py`.

## 0. Short answer

| case | where states first go missing or equal | class |
|---|---|---|
| 1 TubI export (runs_smoke/2026-09-08_1528 PMC6112128 strict) | Stage 1: no `biological_states` at all, transport has neither from nor to, only `evidence` "out of the cell". Equal at `ensure_autostates` | **(c)** both sides derivable at generic level ("inside the cell" -> "outside the cell"); neither is a literal compartment word |
| 2 ferric-enterobactin import (runs_verify/2026-08-21_2057 PMC12096016 research) | Stage 1: transport has only name, cargo, `evidence`; no state. Equal at `ensure_autostates` | **(c) for the from side** (extracellular, named in the preceding sentence), **(d) for a specific to side** (paper only says "imported") |
| 3 enterobactin secretion (runs_verify/2026-08-27_1341 PMC12096016 strict) | Stage 1: transport has no state; the whole Stage-1 payload has no `biological_states`. Equal at `ensure_autostates` | **(c)**, the clearest: its own evidence names both "cytoplasm" and "extracellular environment" |
| 4 enterobactin secretion (runs_verify/2026-09-02_2052 PMC12096016 strict) | Stage 1: model wrote `E. coli cytoplasmic state` on both sides | **(a)** explicit equal states, and the same row's evidence names both sides |

Findings that change how the fix should be judged:

1. All 4 are **single-draw, pre-D-104 runs** (Aug 21 to Sep 8). `T2PW_EXTRACTION_DRAWS=3` became the paper-run configuration on 2026-10-19 (D-104; `docs/evaluation_v2/paper_runs_orch755/README.md`). Over the 11 `runs_papers` runs (3 draws) that carry transports: 42 transports, 34 with two distinct states, 8 one-sided, **0 with neither side and 0 equal**. Over the other 68 runs: 115 transports, 107 both distinct, 4 one-sided, **3 neither, 1 equal**. The union merge of draws (keyed on cargo, from, to; `stage_one_ensemble.py` ~567) lets a row that states more win. The one-sided family survives at 3 draws (OTOP1, wnt_pcp), so the same extraction weakness still exists, only its worst forms are masked.
2. In **all four** cases the model's own `evidence` string for the transport already contains the destination or the origin that the states are missing for. No new text is needed for a re-ask for cases 3 and 4; case 2 needs the preceding sentence; case 1 needs an entailment ("out of the cell" means to the outside).
3. Stage 2 cannot repair a transport row: `merge_additions` only appends (`pipeline.py:1244`, `_extend_unique` at 3273, a JSON-signature dedupe). In cases 1 and 3 Stage 2 did create the states and cargo locations (rule G of `pwml_infer_system.txt:183`) but never wrote them onto the transport; in case 4 it appended a correct twin row.
4. A prompt change is not narrow in effect. `pwml_system.txt` is pinned byte-for-byte to the measured control `r2_01_control_v03` (`tests/test_orch754c_production_prompt_is_measured_control.py`), and any change of the Stage-1 system prompt re-samples the whole extraction.

Recommendation: **candidate B**, a flag-gated, evidence-grounded re-ask for only the transports whose sides are missing or equal, applied once after Stage 1 and the boundary (section 4). Test paired (the off arm is the same Stage-1 payload before the step), on 2 affected papers (PMC12096016, PMC6112128) and 4 controls (PMC12421875, PMC12444477, PMC11487621, PMC12452463). One Stage-0-to-merge run is about 6 min at 1 draw, about 15 min at 3 draws (section 6).

## 1. Stage flow that produced the equal sides (common to cases 1 to 3)

| stage | artefact | what it shows |
|---|---|---|
| Stage 0 | preprocessor call (LEG_TRACE) | context only |
| Stage 1 | `stage1_payload.json` = `clean_stage_one` of the single draw (`pipeline.py` `run_stage_one_with_chunking`, single-chunk clean at `pipeline.py:2761`) | `_clean_processes` (`pipeline.py:2291-2340`) keeps `from_biological_state`/`to_biological_state` only if non-empty, and `_clean_elements_with_states` (`:1962`) drops an empty `biological_state`. The raw model reply is **not stored** (only hashes in LEG_TRACE), so "model wrote an empty string" and "model omitted the key" cannot be told apart |
| Stage 2 + merge | `merged_payload.json` | `merge_additions` only appends rows; an existing transport is never patched |
| Stage 3 | `process_normalizer.ensure_autostates` (`process_normalizer.py:3166`; fill at `:3232` and `:3236`) | a missing side gets `__auto_state__` (location `cell`). Both sides missing gives the same state twice. The P4 `_declared_side_state` (`:3143`) only reads a side declared in `elements_with_states`, so it does nothing here |
| IR / export | `final_mapped.json`, `pwml_ir.json` | `ir.py` copies the keys; `resolve_state` defaults an empty name to the first state (a second way to collapse, see `b4_left_equals_right.md`) |

Equal-side census over all 243 stored `merged_payload.json` (157 transports): both distinct 141, one side only 12, neither 3, equal 1.
Stage-1-level (151 transports matched by name): both distinct 136, one 10, neither 3, equal 2.
Of the 12 one-sided rows, 8 are in the 3-draw runs (OTOP1 H+ permeation and 5-HT release in PMC7059786; c-Jun and Dvl translocation in PMC10425814), 4 in single-draw runs. The one-sided rows end up `__auto_state__` (cell) on the missing side after the normalizer, which is not equal but is the same kind of loss (for example "enterobactin secretion: cytoplasmic -> cell" in the case-2 run).

## 2. Per-case traces

### Case 1: TubI-mediated TBN export, tubercidin

Run: `PRIMARY/runs_smoke/2026-09-08_1528/papers/PMC6112128/strict/` (pinned topic `tubercidin biosynthesis`, Streptomyces tubercidicus, 6017 words).

| stage | file | transport 0 and states |
|---|---|---|
| Stage 1 | `stage1_payload.json` | keys only `entities`, `processes` (no `biological_states`, no `element_locations`). `{"name":"TubI-mediated TBN export","cargo":"tubercidin","transporters":[{"protein":"TubI",...}],"elements_with_states":[{"element":"tubercidin","side":"left","evidence":"TubI is responsible for transporting TBN out of the cell"},{"element":"tubercidin","side":"right","evidence":"TubI is responsible for transporting TBN out of the cell"}],"evidence":"TubI is responsible for transporting TBN out of the cell once synthesized"}`. Both sides declared as rows, no `biological_state`, no from/to |
| Stage 2 + merge | `merged_payload.json` | transport row unchanged. Stage 2 added `biological_states: [{"name":"Streptomyces tubercidicus extracellular","subcellular_location":"extracellular",...}]` and one `compound_locations` row (tubercidin in that state). So the outside state exists but is not on the transport, and no inside state exists |
| Stage 3 | `final_mapped.json` | `"from_biological_state":"__auto_state__","to_biological_state":"__auto_state__"`; `__auto_state__` has `subcellular_location: "cell"` |
| model calls | `LEG_TRACE.jsonl` | one `Stage 1 extraction` call (finished at 100.8 s, 9886 chars, `deepseek/deepseek-v4-flash`), so 1 draw |

Paper text (`01_source_text.txt`): "There are two other enzymes, TubH (kinase), and TubI (MFS transporter), also present in the TBN pathway. It is more reasonable to propose that TubI is responsible for transporting TBN out of the cell once synthesized."

What can be derived without inventing: cargo moves from inside the cell to outside the cell. Neither "extracellular" nor a specific interior compartment is a literal word; "cell" interior is exactly what `__auto_state__` already stands for. Note the paper presents the role as a proposal ("more reasonable to propose"); that is already how the row was extracted, so the fix does not change what is claimed. Only the outside side needs a name; with the from side left empty the normalizer's `cell` placeholder is already distinct from `extracellular`.
Class **(c)**. Literal grounding: none (an entailment table would be needed, see the PO question in section 5).

### Case 2: ferric-enterobactin import, ferric enterobactin

Run: `PRIMARY/runs_verify/2026-08-21_2057/papers/PMC12096016/research/` (6509 words, topic `enterobactin biosynthesis`, Escherichia coli). No LEG_TRACE (older run); single draw by date.

| stage | file | transports |
|---|---|---|
| Stage 1 | `stage1_payload.json` | has `biological_states: [{"name":"cytoplasmic state","subcellular_location":"cytoplasm"}]` and cargo locations. T0 `enterobactin secretion`: `"from_biological_state":"cytoplasmic state"`, one `elements_with_states` left row, **no to side**. T1 `{"name":"ferric-enterobactin import","cargo":"ferric enterobactin","evidence":"the ferric-enterobactin complex is then imported via TonB-dependent uptake"}`: no states, no sides at all |
| Stage 2 + merge | `merged_payload.json` | T0 and T1 unchanged. Stage 2 added `compound_locations`: ferric enterobactin -> `cytoplasmic state` (evidence "ferric-enterobactin complex is then imported via TonB-dependent uptake"). It added no extracellular state |
| Stage 3 | `final_mapped.json` | T0: `cytoplasmic state` -> `__auto_state__` (one-sided; secretion ends in "cell"). T1: `__auto_state__` -> `__auto_state__` (the equal one) |

Paper text: "Enterobactin is produced in the cytoplasm and secreted to the extracellular environment by a TolC-dependent process (Corinna et al., 2005). Following secretion, enterobactin chelates extracellular ferric iron, and the ferric-enterobactin complex is then imported via TonB-dependent uptake (Raymond et al., 2003)." The source uses U+2010 hyphens ("ferric-enterobactin", "TonB-dependent"); the extracted evidence uses ASCII hyphens, so any "find the evidence in the text" step needs dash normalisation (NFKC alone does not fold U+2010; `name_keys.strict_key` is the existing helper to check).

Derivable: from = extracellular ("extracellular ferric iron", "Following secretion" in the sentence before the evidence); to = "into the cell" (generic). A specific to compartment (periplasm, cytoplasm) is **not** in the paper; the run's own Stage-2 location (`cytoplasmic state`) is a model inference, and the other 5 runs of this paper that got it right wrote extracellular -> cytoplasmic (knowledge, not text). The same run's secretion transport (T0) has the same defect in its to side and the evidence names it ("secreted to the extracellular environment").
Class **(c) for from, (d) for a specific to**. The evidence string alone does not support "extracellular"; the previous sentence does.

### Case 3: enterobactin secretion (2026-08-27), cargo enterobactin

Run: `PRIMARY/runs_verify/2026-08-27_1341/papers/PMC12096016/strict/`.

| stage | file | transport |
|---|---|---|
| Stage 1 | `stage1_payload.json` | keys `entities`, `processes` only. `{"name":"enterobactin secretion","cargo":"enterobactin","evidence":"Enterobactin is produced in the cytoplasm and secreted to the extracellular environment by a TolC-dependent process"}`. No sides, no states, no transporter |
| Stage 2 + merge | `merged_payload.json` | transport unchanged. Stage 2 added `biological_states` `cytoplasm` and `extracellular` (both `species: Escherichia coli`) and cargo locations `enterobactin` -> `cytoplasm` (evidence "enterobactin is synthesized in the cytoplasm") and `enterobactin` -> `extracellular` (evidence "secreted to the extracellular environment"). The run therefore holds exactly the two states, but not on the transport |
| Stage 3 | `final_mapped.json` | `__auto_state__` / `__auto_state__`; the two real states remain declared but unused by this transport |

Class **(c)**, strongest: the row's own evidence literally contains "cytoplasm" and "extracellular environment", and both states already exist in the payload.

### Case 4: enterobactin secretion (2026-09-02), cargo enterobactin

Run: `PRIMARY/runs_verify/2026-09-02_2052/papers/PMC12096016/strict/` (LEG_TRACE: one Stage 1 call at 310.2 s, 9884 chars; 2 Stage 2 calls).

| stage | file | transports |
|---|---|---|
| Stage 1 | `stage1_payload.json` | T0 `{"name":"enterobactin secretion","cargo":"enterobactin","from_biological_state":"E. coli cytoplasmic state","to_biological_state":"E. coli cytoplasmic state","evidence":"Enterobactin is produced in the cytoplasm and secreted to the extracellular environment"}`; `biological_states` holds only `E. coli cytoplasmic state`. The model wrote the same state on both sides |
| Stage 2 + merge | `merged_payload.json` | T0 unchanged; Stage 2 **appended T1** `enterobactin secretion`: `E. coli cytoplasmic state` -> `extracellular space` (evidence "...secreted to the extracellular environment by a TolC-dependent process.") and the state `extracellular space` |
| Stage 3 | `final_mapped.json` | T0 equal and T1 correct both present; the normalizer does not touch declared states |

Class **(a)**. Same evidence names both sides; a correct twin already exists, so fixing T0 turns it into a duplicate of T1 (same cargo, from, to). Whether the writer and importer tolerate or merge two identical transports is not established here; the stage_one_ensemble key (cargo, from, to) would merge them at 3 draws. This is the one case where "keep everything we have" and "fix it" can pull apart (the fixed row adds nothing new); the repair must therefore never delete the twin.

## 3. Information available in each run (for a derivation without inventing biology)

| case | evidence literally names | preceding or following sentence | states already in payload |
|---|---|---|---|
| 1 | "out of the cell" | nothing about compartments | `... extracellular` (Stage 2, merged) |
| 2 | "imported via TonB-dependent uptake" | "enterobactin chelates extracellular ferric iron"; "Following secretion" | `cytoplasmic state` only |
| 3 | "cytoplasm", "extracellular environment" | same sentence | `cytoplasm`, `extracellular` (Stage 2, merged) |
| 4 | "cytoplasm", "extracellular environment" | same sentence | `E. coli cytoplasmic state`; `extracellular space` (Stage 2 twin) |

Why the model leaves them empty: `pwml_system.txt:415` ("biological_state fields ... Leave "" if none is explicitly stated") and `:77` ("If something is not explicitly present, omit it or leave lists empty") push the Stage-1 model toward blank (the Stage-2 prompt has the same rule at `pwml_infer_system.txt:188`, "NOT ALLOWED: No cell type/tissue/subcellular location unless explicitly stated") for prose such as "out of the cell". The prompt's own Examples 3 and 3b (`:505-542`) show both states, but only for text that names both compartments with bracketed placeholders.

## 4. Extraction-step fix candidates (all behind `T2PW_TRANSPORT_SIDES`, default off)

Flag convention: same as `service/final_payload.py:50`, `os.environ.get("T2PW_TRANSPORT_SIDES", "").strip().casefold() in ("1","true","yes")`, read at call time, not import time. The batch child inherits the environment (`batch/runner.py:279`), so one `export` covers a leg.

### A. Prompt addendum for Stage 1

- Change: a new text file (suggested `src/t2pw/llm/prompts/pwml_transport_sides.txt`, **not** an edit of `pwml_system.txt`, which a test pins) appended to the system prompt at `pipeline.py` `run_extraction_pipeline` `_draw()` (the `system_prompt=` at ~281) when the flag is on. Wording sketch: every transport names two different states; never repeat one state on both sides; read sides from the sentence and its neighbours (secreted, exported, efflux, out of the cell -> to = extracellular; imported, uptake, into the cell -> from = extracellular; the other side = the compartment the text names for the cargo, else leave it empty); declare each state in `biological_states`.
- Seen by: every stage after Stage 1 (Stage 2 prompt carries the Stage-1 JSON).
- Other output: **yes, possible and not boundable**. The Stage-1 request hash changes, and at temperature 0 the identical prompt already gives different extractions run to run (CLAUDE.md standing rule), so a changed prompt may change reactions, proteins, locations and the new `extracellular`-type states for every paper. "Keeps everything" can only be shown statistically, against a flag-off vs flag-off noise baseline, and the repo's own prompt-variant protocol (`docs/evaluation_v2/prompt_variants_r2/`, gold-set scoring) is the bar.
- Cases: (c) and (a) plausibly improve; no guarantee on any single run.
- Verdict: weakest for the PO's rule.

### B. Post-extraction re-ask, transports only (recommended)

- Change: a new module (suggested `src/t2pw/pipeline/transport_sides.py`), one function applied to the Stage-1 payload right after `settle_stage_one` in `service/extraction.py:run_stage_one` (after line 697, before the RAG seed and Stage 2), guarded `if transport_sides.enabled():`. Chosen there because that is the one place both the app and `run_pathway` pass through and the paper `text` is in scope; `settle_stage_one` itself takes no text.
- Detection (pure Python, no model): effective sides = row-level from/to, else the `elements_with_states` left/right `biological_state`. Deficient = a side empty, or the two sides equal after `process_normalizer._normalize`. Rows with two distinct sides are never sent and never touched.
- Re-ask: one bounded call (batched over deficient rows, capped, `chat_fn` seam like `localized_repair.repair_invalid_rows`). Input per row: cargo, row evidence, the source-text window around the evidence (the evidence sentence and the sentence before and after, located with dash/NFKC normalisation), the existing `biological_states` names. Output per side: a state name, plus the verbatim quote that justifies it, or null.
- Acceptance (reuses the existing guard ideas, `localized_repair.preserves_original_values` and `evidence_supports`, but with the window as evidence and not the row's own string): accept a side only if its quote is located in the paper text; never write a side equal to the other side; fill only the sides the text names and leave the others empty (the normalizer then places `__auto_state__`/cell, which is distinct from a named extracellular side, so cases 1 and 2 need only the named side). Any rejection leaves the row exactly as it was.
- Writes: only `from_biological_state`, `to_biological_state` and the matching `elements_with_states[].biological_state` of the deficient rows, plus a `biological_states` row when the chosen state is not yet declared (species taken from existing states), tagged inferred with a lineage entry (`lineage.py`). Nothing else.
- Seen by: Stage 2 onward, which then receives better rows; the normalizer is unchanged.
- Other output: none by construction when no transport is deficient (zero calls, same object returned); a new state row is the only way it changes non-row content, the same kind of row Stage 2 already adds (cases 1 and 3). Open check for the implementer: Stage 2 later adds its own `cytoplasm`/`extracellular` states, so state-name duplicates must be counted before and after (the writer collision rule, `writer.py` keep rule, may already absorb them).
- Cases: 3 and 4 fully (literal quotes); 2 from side only (window needed, 'imported' names no compartment); 1 only if "out of the cell" is allowed to entail `extracellular` (PO question below). Case 4 yields a duplicate of its twin (section 2); decide whether the repair may skip a row whose corrected pair already exists as another row (skipping is safe: it changes nothing).
- Cost: 1 extra LLM call, only for papers that have a deficient transport (about 1 in 6 of the single-draw enterobactin runs).

### C. Validation re-ask inside the Stage-1 retry loop

- Change: treat an equal-side transport as a contract error and let `_run_json_stage` re-draw (`last_error`). Rejected as the way to go: it re-samples the whole Stage-1 reply (the same non-determinism as A, and the draw can return a different pathway), "neither side" cannot be an error because blank is allowed by the contract, and the ladder (`ExtractionLadder`, ceiling of three calls) is shared with truncation recovery. Not recommended.

### D. Deterministic repair from Stage 2 locations (outside the PO's wording, noted only)

- Case 3 could be fixed with no model: the payload holds the cargo in two states (`enterobactin` in `cytoplasm` and in `extracellular`, direction from the evidence order). It is a merge or normalizer change (a forbidden-by-this-ruling area), it covers 1 of the 4 cases, and direction would be a heuristic. Listed so the PO knows it exists.

## 5. Open questions for the product owner (to record in the RESULT doc)

1. Does "out of the cell" / "secreted" / "efflux" entail an `extracellular` state (needed for TubI), or only a literal word in the evidence/window (fixes cases 3, 4 and the from side of 2, not 1)? Cost of "literal only": 1 of 4 cases stays equal. Cost of an entailment table (about 6 phrases): a small biology rule for a human to bless.
2. Case 2 to side and case 1 from side: leave empty (normalizer places the generic `cell`), or name `cell`? Recommendation: leave empty; it adds nothing the paper does not say.
3. Case 4: if the repaired row duplicates its twin, keep both, or skip the repair for that row? Recommendation: skip (changes nothing, removes nothing).

## 6. Recommended test set, harness and time

### Test papers (stored text exists as `01_source_text.txt` in each run dir)

Affected:

| paper | PMID | run dir to start from | topic line (as in a topics file) | words | cases |
|---|---|---|---|---|---|
| PMC12096016 | 40400396 | `PRIMARY/runs_verify/2026-08-21_2057/papers/PMC12096016/research`, `.../2026-08-27_1341/.../strict`, `.../2026-09-02_2052/.../strict` | `PMC12096016 | enterobactin biosynthesis | Escherichia coli` (also `docs/evaluation_v2/paper_runs_orch755/topics/enterobactin.txt`) | 6509 | 2, 3, 4. 24 stored runs, 4 with a non-clean transport (about 17 percent per single-draw run), 20 clean: the best paired baseline |
| PMC6112128 | 30153835 | `PRIMARY/runs_smoke/2026-09-08_1528/papers/PMC6112128/strict` | `PMC6112128 | tubercidin biosynthesis | Streptomyces tubercidicus` | 6017 | 1. Only one stored run, so no baseline rate: measure it |
| optional PMC7059786 | 32185015 | `PRIMARY/runs_papers/ORCH-759p/legs/sour_taste_v2-strict/2026-10-22_2233/papers/PMC7059786/strict` | `PMC7059786 | Taste transduction: sour | Mus musculus` (`.../topics/sour_taste_v2.txt`) | 5401 | the one-sided family at 3 draws (OTOP1, 5-HT release); shows whether B generalises and is the cheapest paper |

Controls (transport-rich, all transports currently correct, mixed phrasing, 4 papers):

| paper | PMID | stored runs / non-clean | transports it carries | why |
|---|---|---|---|---|
| PMC12421875 | 40788026 | 11 / 0 | `DHNA secretion` and `DHNA diffusion out of cell`, cytoplasmic -> extracellular | "out of cell" phrasing like TubI, currently right |
| PMC12444477 | 40752575 | 6 / 0 | `MsbA flippase transport`, inner membrane cytoplasmic leaflet -> periplasmic leaflet | sides inside one membrane; any repair that renames states would show here |
| PMC11487621 | 39194188 | 1 / 0 | three imports, `extracellular` -> `cytoplasm` | import prose like case 2 |
| PMC12452463 | 40966180 | 12 / 1 | `enterobactin secretion`, `ferri-enterobactin import via FepA` | same topic as the affected paper, different text; 11 of 12 stored runs clean |

Run dirs for the controls: `PRIMARY/runs_verify/2026-09-02_2052/papers/{PMC12421875,PMC12444477,PMC12452463}/...`, `PRIMARY/runs_smoke/2026-09-07_2323/papers/PMC11487621/strict`. Topic lines come from each `00_PAPER.txt` ("requested pathway"/"requested organism"): `menaquinone biosynthesis | Bacillus subtilis`, `lipid A biosynthesis | Escherichia coli` (a multi-example review, so the extraction focus box matters; `driver._extraction_focus`), `coenzyme A biosynthesis | Bacillus subtilis`, `enterobactin biosynthesis | Escherichia coli`.

### Test design (why paired matters)

A single live run cannot show that a 17 percent per-run defect is fixed (an unlucky-good sample proves nothing). Candidate B is a function of the Stage-1 payload, so each live draw can be evaluated both ways from the same payload: off = payload as extracted, on = payload after the step. That removes sampling noise from the comparison. Plan:

1. Offline, no network: take the four stored `stage1_payload.json` and the stored `01_source_text.txt`, run the step with a fake `chat_fn` (unit tests) and then with the live model for those 4 rows only (about 4 calls). Expected: cases 3, 4 fixed; 2 from side fixed; 1 per the PO ruling.
2. Live paired, Stage 0 to the merge only (below): PMC12096016 and PMC6112128 at 1 draw, 10 draws each (the failing form is rare, about 1 in 6, so also keep every draw that has any deficient transport); then the same two at 3 draws, 3 draws each. Count: deficient rows before, after; rows changed; any change outside the transport fields (must be zero).
3. Controls: 4 papers x 2 live draws at 3 draws. Expected: zero repair calls and a byte-identical Stage-1 payload (the step returned its input). Any non-zero count is a finding.
4. Candidate A, if the PO wants it measured too: unpaired, flag-off vs flag-on vs a second flag-off arm (noise floor), at least 10 per arm on each affected paper, and compare every Stage-1 field, not only transports. Cost is about 3 times step 2.
5. One end-to-end confirmation leg per affected paper through `scripts/batch_run.py` with the flag exported, then the PWML transports compared.

### Harness without the whole pipeline

`scripts/batch_run.py` has no stop-after option (its arguments: topics, modes, out, fresh, timeout, deadline) and `service/pathway.run_pathway` calls `run_extraction` and then the post-pipeline (audit, curator, mapping, export), so it cannot stop early. The function seams that do exist, all Streamlit-free:

- Stage 0: `t2pw.pipeline.preprocessor.preprocess(text, user_task_context=...)` (or `service.extraction.run_stage_zero` with a `service.reporter` reporter).
- Stage 1: `t2pw.pipeline.pipeline.run_stage_one_with_chunking(text, pathway_context=..., artifact_dir=<scratch>, enable_chunking=..., chunk_word_limit, chunk_overlap, max_attempts, temperature, max_tokens)`; draws via `T2PW_EXTRACTION_DRAWS`; then `t2pw.pipeline.stage_one_boundary.settle_stage_one(...)`. This is exactly what `service/extraction.run_stage_one` runs (lines ~675-698) minus the RAG seed.
- Stage 2 and merge: `pipeline.run_stage_two_with_feedback_loop(...)` then `pipeline.merge_additions(base, additions)`.
- Normalizer: `process_normalizer.normalize_process_payload(payload, mode=...)`, pure and offline (P4's `b_before_after.py` already drives it from `merged_payload.json`).
- RAG seed is skipped by simply not calling it (or `RAG_ENABLED=0`, `config.RAG_ENV`).

A small driver of these calls (new file in the evidence dir, not in src) is the way; use the stored `01_source_text.txt` as input (no fetch, identical text every run) and the topic and organism from `00_PAPER.txt` as `user_task_context`. Run it through `docs/pwml_recovery_sprint/evidence/bounded_run.py` with the heavy lock, one job at a time. Note: Stage 2B ID mapping (DB/cache) is not part of this boundary; P4 showed the normalizer's state handling does not depend on it.

### Wall-time estimate (from the stored LEG_TRACE files: 140 files, 139 with a Stage-1 call)

| piece | measured | source |
|---|---|---|
| Stage 0 (preprocessor) | 14 to 50 s | LEG_TRACE first model call |
| Stage 1, one call (the gap since the previous model call) | median 96 s, p90 254 s, max 718 s; PMC12096016: first Stage-1 call ends at 102 to 310 s (preprocessor included), PMC6112128: about 68 s | `timing.py`, `timing2.py` |
| Stage 1 at 3 draws | 3 sequential calls, e.g. ends at 203, 366, 598 s (PMC12096016 research, 2026-10-21) | LEG_TRACE |
| Stage 2 (2 calls) | about 50 to 110 s per call, so 2 to 4 min | TubI run (calls end 406 and 512 s), 2026-09-02 run (743 and 795 s) |
| RAG prose seed (skipped by the harness) | about 4 to 6 min of `rag_prose_extraction` calls | LEG_TRACE |
| normalizer | under 1 s offline | b_before_after.py ran 243 payloads |
| full production leg, for comparison | median 1235 s (n=128), p90 2641 s; PMC12096016 at 3 draws 1537 to 3447 s | `manifest`/LEG_TRACE |

Planning figures for one live run, Stage 0 to normalizer, no RAG, no audit or curator: **about 6 min at 1 draw (range 3 to 9), about 15 min at 3 draws (range 10 to 20)**; Stage 0 plus Stage 1 only: about 2 to 5 min at 1 draw, 8 to 14 min at 3 draws. A full `batch_run.py` leg per paper: 20 to 35 min at 1 draw, 30 to 60 min at 3 draws. So step 2 above (about 20 draws at 1 draw plus 6 at 3 draws, Stage 0 to Stage 1 only) is roughly 1.5 to 2.5 hours serial; step 3 (8 runs at 3 draws) roughly 1.5 hours; step 5 about 1.5 hours. Calls are paid OpenRouter calls (Stage 1 replies are 10k to 19k characters).

## 7. Not established

- The raw Stage-1 model reply is not stored in any of the four runs (LEG_TRACE has hashes only; `stage1_raw_extraction.json` was written to `tmp/` and is not in the run dirs), so whether the states were empty strings or omitted keys is unknown. Raw Stage-2 replies are not stored either; what Stage 2 proposed for the transport rows is read off `merged_payload.json`.
- The Aug runs have no LEG_TRACE; "single draw" for cases 2 and 3 is by date (ORCH-750 landed 2026-10-16), and the model name is read only from the Sep and Oct traces (`deepseek/deepseek-v4-flash`).
- The census matches Stage-1 to merged rows by transport name, so the Stage-1-level numbers are approximate; the merged-level numbers (157 transports) are exact. `runs_papers` is treated as "3 draws" because D-104 and the ORCH-755 README say so for those legs and the traces show 3 Stage-1 calls; the Sep and Aug runs are treated as 1 draw.
- No candidate was run. Whether a prompt addendum or the re-ask returns the intended states, whether duplicate state names appear after Stage 2, and whether the writer tolerates a duplicate of case 4's twin are untested.
- Time figures are from other runs of the same configuration and model; provider latency varies by a factor of three between runs of the same paper (PMC12096016 Stage 1: 102 s versus 310 s).
