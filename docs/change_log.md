# Change Log

Every entry answers: what was the error, why did it appear, and how does the
fix stay consistent with the intended pipeline design.

---

## Fixed

---

### 2026-07-21 — RAG deep-dive: four live-run defects that emptied or degraded multi-paper output

**Files changed:** `src/t2pw/app/streamlit_app.py`, `src/t2pw/rag/retrieve.py`,
`src/t2pw/rag/store.py`, `src/t2pw/rag/select.py`, `.env`,
`tests/test_rag_retrieve.py`, `tests/test_rag_foundation.py`,
`tests/test_rag_select.py`, `docs/change_log.md`.

**What was the error:** with `RAG_ENABLED=true` and the "unknown / incomplete"
box checked, a real seed pathway (caffeine degradation, *Pseudomonas putida*)
came back with a **completely empty** `processes.reactions` — the pathway summary
and final merged JSON showed only a species and a biological state. The RAG unit
suite was fully green (offline fixtures), so none of these surfaced until the
subsystem actually ran end-to-end against LM Studio + live literature APIs. Four
independent defects, all inside `t2pw.rag` + the app orchestrator.

*Defect 1 (the empty payload — seed reactions dropped).* `maybe_run_rag` passed
`synthesize_with_report(seed_payload, bundles, seed_context_text)` a **string**
(`format_context_header(...)`) as the seed context. But `synthesize`'s
`_seed_source_descriptor` only accepts a **dict** carrying a `source_id`, so it
returned `None`; `_seed_reactions` then treated every seed reaction as having "no
supporting evidence" and **omitted all of them** (the no-invention guardrail,
misfiring on the seed paper). Synthesis emitted zero reactions.

*Defect 2 (the wipe).* Seam S3 replaced `final_payload = rag_result.payload`
wholesale whenever `rag_result.synthesized` was truthy — so the empty synthesized
payload from Defect 1 *overwrote* the real Stage 1/2 extraction. (The seed's
`species` / `biological_states` survived only because WP5's
`_carry_forward_scaffolding` re-copies them; the reactions did not.)

*Defect 3 (silent retrieval death — embedding dim mismatch).* `MemoryVectorStore.query`
scored every candidate with `_cosine`, which returns `0.0` on a length mismatch.
A lexical-fallback vector (256-dim) cached while the embeddings endpoint was down
(the user's exact earlier state) then sits in `embeddings_cache.json` beside real
API vectors (768-dim); once LM Studio is up, query 768 vs cached-chunk 256 → every
score `0.0` → semantic retrieval silently returns arbitrary chunks.

*Defect 4 (lost gap symbols).* `retrieve._reaction_symbols` read
`reaction["inputs"]/["outputs"]` only as `str`, but the real payload carries
participants as dicts (`{"name": "caffeine"}`). So a dangling-reaction gap query
lost its exact substrate/product symbols — the lexical half of the hybrid scorer
had only the reaction name and enzyme to match on.

Plus one robustness hole (Defect 5): `select._candidate_context` called the reused
`preprocess` per candidate **unguarded**, inside `maybe_run_rag`'s single
try/except — so one flaky LLM call (rate limit / timeout) among up to
`RAG_ACQUIRE_MAX_PAPERS` candidates aborted the *entire* RAG run.

**Why it appeared:** every WP was built and verified with offline, in-memory
fixtures where the seed context was already a dict, embeddings were a single
consistent width, and `preprocess` was mocked. The string-vs-dict seam (D1), the
wholesale replace (D2), the cache-poisoning dim mismatch (D3), the str-only
participant read (D4), and the unguarded fan-out (D5) are all boundary conditions
that only exist in a live run, so the green unit suite never exercised them.

**How the fix stays consistent with the design:** every change lives in
`t2pw.rag` + the app orchestrator (`src/t2pw/app`); no pipeline stage module was
edited and the separation invariant (docs/rag/03_separation_invariant.md) holds.
*D1:* `maybe_run_rag` now builds a seed **source descriptor** dict
(`{"text": seed_context_text, "source": {"source_id": "seed_paper",
"source_title": <pathway name>, "source_type": "paper"}}`) and passes it to
synthesis — the uploaded paper is legitimately evidence for its own reactions
(exactly what `_seed_source_descriptor`'s docstring intends), so seed reactions
carry `rag_provenance` and survive. *D2:* the S3 adoption is now guarded — the
synthesized payload replaces `final_payload` only when it preserves at least the
seed's reaction count (`_n_reactions(synth) >= max(1, _n_reactions(final))`),
otherwise the single-paper extraction is kept and a `st.warning` explains why; an
evidence-starved synthesis can never again blank the pathway. *D3:*
`MemoryVectorStore.query` falls back to lexical overlap whenever the query/chunk
vector widths differ (or either is missing) instead of scoring a silent `0.0`.
*D4:* `_reaction_symbols` now also reads dict-shaped participants
(`name`/`entity`/`compound`/…). *D5:* `_candidate_context` wraps `preprocess` in a
per-candidate try/except, degrading one failure to an organism-only context
rather than sinking the run. `.env` keeps `RAG_ENABLED=true` (LM Studio embeddings
verified reachable) now that the wipe is guarded. All fixes are additive/guarding;
with `RAG_ENABLED=false` the single-paper path is byte-for-byte unchanged.

**Verified:** offline repros confirmed each defect and its fix (string seed
context → 0 reactions vs dict → all preserved; 256-vs-768 cosine `0.0` → lexical
`0.4`; dict participants → symbols now include `caffeine`/`theobromine`). Three
regression tests added — `test_dangling_reaction_gap_captures_dict_participant_symbols`
(D4), `test_memory_query_dim_mismatch_falls_back_to_lexical` (D3),
`test_select_survives_a_failing_preprocess` (D5). RAG suite: 88 → **91 passed**;
full suite **515 passed, 0 failures**.

**Known limitation (not a regression — flagged for follow-up):** evidence-derived
reactions are still only transcribed from **arrow-style equations**
(`synthesize._parse_reaction_line`), which paper *prose* rarely contains, so
cross-paper stitching materializes almost entirely from structured DB records, not
free text. Until an LLM prose→reaction extraction step is added, multi-paper
synthesis will stay sparse even with all four defects fixed.

---

### 2026-07-21 — RAG synthesis: stop emitting `" ; "`-joined pathway-metadata blobs as entities (+ core defense-in-depth)

**Files changed:** `src/t2pw/rag/synthesize.py`,
`src/t2pw/pipeline/process_normalizer.py`, `tests/test_rag_synthesize.py`,
`tests/test_process_normalizer.py`, `docs/rag/03_separation_invariant.md`,
`docs/change_log.md`.

**Separation-invariant note (read this first):** this entry contains a **deliberate,
user-authorized exception** to the separation invariant
(docs/rag/03_separation_invariant.md): CHANGE 2 edits a **core stage module**
(`process_normalizer.py`). Every prior RAG entry kept all logic inside `t2pw.rag`;
this one does not, by explicit decision, to add defense-in-depth against a malformed
name class regardless of who produces it. The primary fix (CHANGE 1) is still fully
inside `t2pw.rag`; CHANGE 2 is an additive, narrowly-gated guard that changes no
existing behavior (zero test regressions). See the doc's new "Sanctioned exceptions"
section.

**What was the error:** with RAG enabled, the mapped pathway failed the pre-export
Stage 3 revalidation with "Protein '<blob>' is missing a UniProt or DrugBank
identifier.", where `<blob>` was an entire pathway serialized with `" ; "` separators
— e.g. `"Pathway12926 ; Arabidopsis thaliana, Cell, Plant-Type Vacuole ; Arabidopsis
thaliana, Cell, Cytosol ; ... ; Water ; Hydrogen Ion ; Triglyceride ; ... ; Glycerol
3-phosphate transporter ; Water"` (the Arabidopsis glycerolipid pathway), plus a
sibling `"Pathway4 ; Homo sapiens ... ; Adenosine triphosphate complex"` (glutathione)
routed into `protein_complexes` by the NAME-BASED COMPLEX RULE. Neither pathway was the
uploaded seed (a caffeine-degradation paper) — they were **retrieved corpus entries**.

**Root cause:** RAG synthesis re-parsed retrieved *evidence* chunk text as reaction
chemistry, but for corpus (`source_type="pwml_example"`) and DB (`pathbank`/`kegg`)
hits that text is not a clean equation — it is a `" ; "`-joined **bag** of
pathway-id + species + compartments + compounds + reaction-patterns that `ingest.py`
builds for *lexical scoring* (`_corpus_text_for_file` / `_extract_pwml_text` /
`_reaction_record_text`, all `" ; ".join(...)`). Every such bag contains a reaction
arrow somewhere (from the SBML `reaction_patterns`), so `synthesize._reactions_from_bundle`
→ `_parse_reaction_line` treated the whole ~800-token window as one equation, and
`_parse_side` — which split participants **only** on `" + "` (`_PLUS_SPLIT_RE`), never
on `" ; "` — collapsed the entire pre-arrow text into a **single giant participant
name**. That name became a compound entity (and, ending in "complex", a protein
complex), reaching the gate as an unresolvable protein/complex. Confirmed by live
repro: `_corpus_text_for_file('reference/PW012926 (1).sbml')` → first window →
`_parse_reaction_line(...)` returned a reaction whose `inputs[0].name` was the full
blob; the entity was present in `tmp/draft_graph.json` (built pre-mapping/pre-audit),
proving synthesis — not the audit or mapping — was the origin.

**Why the earlier robustness fixes did not catch it:** two structural reasons, i.e.
this is a genuine case of RAG **not** plugging into the prior stages despite the intent.
(1) Synthesis builds entities with its **own** ad-hoc parser (`_parse_reaction_line` /
`_parse_side`), far weaker than Stage 1 LLM extraction + `process_normalizer`; the prior
composite-splitting / name-sanitization fixes live in the extraction+normalizer path and
were never in synthesis's code path. (2) `" ; "` is a **RAG-invented** join delimiter —
the core normalizer's composite splitter keys on `" + "` / `_has_plus_token` and the core
pipeline never produced `" ; "`-joined names, so even downstream the splitter did not
recognize the blob as a composite and it survived to the gate untouched. Underneath both:
`build_motif_entry`'s corpus text is *lexical-retrieval scaffolding* (a token bag), never
meant to be round-tripped back into structured reactions.

**Fix — CHANGE 1 (primary, inside `t2pw.rag`, `synthesize.py`):** (a)
`_reactions_from_bundle` now reads `chunk.source_type` defensively and only transcribes
chunks whose type is in `{"paper", ""}` (+None) — `pwml_example` corpus scaffolding and
`pathbank`/`kegg` metadata bags are never parsed into reactions. (b) `_parse_side` /
`_parse_reaction_line` are hardened even for the chunks still parsed: participant sides
split on `";"` as well as `" + "` (`_SIDE_SPLIT_RE`), and any token that is clearly not a
single chemical species is rejected (`_is_invalid_species_token`: `^Pathway\d`, a
`", Cell,"`/biological-state descriptor, or > 12 words / > 120 chars); enzymes pass the
same filter; a reaction left with no valid participants is discarded. (c) Genuine
equations are unaffected (`"theobromine + O2 -> 7-methylxanthine + formaldehyde"` still
parses to 2+2; charge notation `NAD+`/`H+` survives — the split requires surrounding
spaces).

**Fix — CHANGE 2 (defense-in-depth, core exception, `process_normalizer.py`):** a
conservative, narrow guard `_quarantine_pathway_metadata_blobs`, called at the top of
`normalize_composites`, drops a `compounds`/`proteins`/`protein_complexes` row **only**
when its name contains `" ; "` **and** matches the garbage signature (`^Pathway\d` OR
`", Cell,"` OR > 12 words), recording a `pathway_metadata_blob_quarantined` action. The
**narrow guard was chosen over a broad `";"`-split** deliberately: the existing `" + "`
path *materializes a protein complex* (wrong semantics for a metadata bag), and the
narrow guard is the zero-regression path — the full suite confirmed no broadening was
needed. Real single-entity names essentially never contain `" ; "` (composites use
`" + "`), so the false-positive surface is negligible.

**Verified:** original bug reproduced then confirmed clean (the corpus blob no longer
yields a `" ; "` participant — the chunk is skipped, and the parser guards shatter the
bag even if it is mislabeled). New tests: `test_corpus_pwml_chunk_never_emits_pathway_blob_entities`
and `test_genuine_paper_equation_chunk_still_parses_cleanly`
(`tests/test_rag_synthesize.py`), `test_pathway_metadata_blob_is_quarantined_by_normalizer`
(`tests/test_process_normalizer.py`). Full suite: **512 passed, 0 failures** (509
baseline + 3 new); **no existing test regressed**. `ruff check src/t2pw/rag
src/t2pw/pipeline/process_normalizer.py` clean, no new violations.

---

### 2026-07-21 — RAG synthesis: carry the seed's species scaffolding + stop the `provenance` key collision

**Files changed:** `src/t2pw/rag/synthesize.py`, `src/t2pw/rag/provenance.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_rag_synthesize.py`,
`tests/test_rag_provenance_gates.py`, `tests/test_rag_foundation.py`,
`docs/rag/03_separation_invariant.md`, `docs/rag/00_overview.md`,
`docs/rag/agents/wp0_foundation.md`, `docs/rag/agents/wp5_synthesis.md`,
`docs/change_log.md`.

**What was the error:** with RAG enabled, a caffeine-degradation seed paper aborted
at the Stage 2B mapping output → Stage 3 normalization input boundary
(`validate_post_mapping`) with one error — `species_required`, "Mapped payload must
include at least one species row." — accompanied by 63 `runtime_schema_type_error`
warnings, one per compound, all "Expected a string." at
`/entities/compounds/N/provenance`. Two independent defects, both introduced by the
RAG work packages, both fixable entirely inside `t2pw.rag` + the app orchestrator.

*Defect 1 (the abort).* WP7 wires the synthesized payload to **replace**
`final_payload` wholesale at seam S3 (`streamlit_app.py`: `final_payload =
rag_result.payload`). But WP5 synthesis rebuilds the payload **from reactions only**:
`_build_entities`/`to_payload` emit just `entities.compounds` and `entities.proteins`
and never read `seed_payload["entities"]`. So the seed's contextual scaffolding —
`species`, `subcellular_locations`, `cell_types`, `tissues`, and top-level
`biological_states` — was silently dropped. Stage 2B mapping then produced a payload
with zero species rows, and the post-mapping gate (which legitimately requires
`entities.species` to be a non-empty list) aborted.

*Defect 2 (the 63 warnings).* WP0 chose `provenance` as one of the four additive RAG
keys, and WP5's `_attach_provenance` wrote a **dict** there
(`row["provenance"] = dict(primary)`). But `provenance` is **not** a free additive
name: the core schema already owns it as a *string*
(`PayloadProvenance = Literal["extracted","inferred","curated","enriched"]`, present
on every entity/process via `PayloadCommonRecord`). RAG was therefore repurposing and
retyping a core-owned field — exactly what the separation invariant's
additive-metadata rule forbids — which the runtime shape validator flagged (in report
mode, so non-fatal) on every compound.

**Why it appeared:** WP5 was written and reviewed as an *additive-evidence* layer, and
its own brief (docs/rag/agents/wp5_synthesis.md) called `species`/`subcellular_locations`
"contextual scaffolding … never emitted by RAG synthesis." That was harmless while
synthesis output only *augmented* the seed, but WP7 later made it *replace* the seed
payload, at which point "never emitted" became "silently deleted." The `provenance`
collision was latent from WP0: the additive keys were never checked against core-owned
field names, and because the shape validator runs in report mode the collision only
ever surfaced as a warning, so it was never treated as a defect.

**How the fix stays consistent with the design:** both fixes stay entirely inside
`t2pw.rag` + the app orchestrator; no stage module was edited and the separation
invariant (docs/rag/03_separation_invariant.md) holds. *Defect 1:* a new
`_carry_forward_scaffolding(payload, seed_payload)` in `synthesize.py` deep-copies the
seed's non-reaction scaffolding buckets (`species`, `subcellular_locations`,
`cell_types`, `tissues` into `entities`; `biological_states` at payload top level) into
the synthesized payload — only when present and not already rebuilt, so the
evidence-built `compounds`/`proteins` are never clobbered — guarded against a non-dict
seed / missing entities, and run before the existing `validate_post_extraction`
self-check. These buckets are evidence-*exempt* (they are not reaction chemistry; see
`_EVIDENCE_ENTITY_BUCKETS` in `provenance.py`), so carrying them verbatim does not
violate the "no element without evidence" guarantee. *Defect 2:* the additive source
pointer is renamed `provenance` → `rag_provenance` everywhere it is emitted or read as
the RAG key (`RAG_ADDITIVE_KEYS`, `RagAdditiveMetadata`, `_has_resolvable_source`,
`_attach_provenance`, `strip_provenance`/`validate_provenance` via the tuple, and the
app's provenance viewer). The core `provenance` string field is left untouched — in
particular `_seed_row_provenance` still reads the seed's core `provenance` string — so
a RAG-off or RAG-unaware stage sees an unchanged core row, and the namespaced `rag_*`
key can never again shadow a core one. The additive keys stay optional/additive and
`strip_provenance` still removes exactly `RAG_ADDITIVE_KEYS`.

**Verified:** new/extended tests in `tests/test_rag_synthesize.py` prove the seed's
`species` (and the other scaffolding) is carried forward and satisfies the
`validate_post_mapping` species predicate, that synthesized rows carry `rag_provenance`
(a dict) and never a dict under the core `provenance` key, and that `strip_provenance`
removes it; a new end-to-end test
(`test_synthesized_payload_survives_real_stage2b_mapping`) drives a synthesized payload
through the **real** `map_ids.map_payload` and asserts the **real**
`validate_post_mapping` passes with no `species_required` error and the seed species row
survives mapping with its `mapping_meta` (external resolver calls mocked offline,
mirroring `tests/test_stage2_mapping_boundary.py`; the gate is never stubbed).
`tests/test_rag_provenance_gates.py` and `tests/test_rag_foundation.py` updated for the
renamed key. Full suite: **509 passed, 0 failures** (508 baseline + 1 new integration
test); ruff clean on `src/t2pw/rag` with zero new violations in `streamlit_app.py`
(its pre-existing 34 E402 sys.path-shim baseline is unchanged). With `RAG_ENABLED=false`
the payload carries neither `rag_provenance` nor the carried-forward path, so today's
single-paper pipeline is byte-for-byte unchanged.

---

### 2026-07-20 — RAG defaults: dedicated embeddings endpoint + toggle-on default

**Files changed:** `src/t2pw/config.py`, `src/t2pw/rag/embed.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_rag_foundation.py`, `.env`,
`docs/change_log.md`.

**What was the error:** the embedder (`embed.py`) reused the shared chat client
(`t2pw.llm.client._client`) for embeddings and ignored `RAG_EMBEDDING_PROVIDER`.
With `LLM_PROVIDER=openrouter` that meant embedding calls went to OpenRouter,
which has **no embeddings endpoint**, so every call failed and silently dropped
to the lexical fallback — "full embeddings" could never actually run. The RAG
toggle also defaulted off and the master flag defaulted off, so RAG was never the
active default.

**Why it appeared:** WP0 wired embeddings against the single shared client on the
assumption chat and embeddings share a host. They don't when chat is OpenRouter.

**How the fix stays consistent with the design:** it stays entirely inside
`t2pw.rag` + config + the S5 orchestrator (no stage-module edit; separation
invariant intact). `config.py` gains two optional, default-safe keys
(`RAG_EMBEDDING_BASE_URL`, `RAG_EMBEDDING_API_KEY`); `embed.py` builds a
dedicated OpenAI-compatible client pointed at that base_url when set (e.g. LM
Studio at `http://127.0.0.1:1234/v1`) and otherwise reuses the shared client
exactly as before — the lexical offline fallback is unchanged, so a missing/
unreachable endpoint still degrades gracefully. The app toggle now defaults ON
(RAG is the default) and, when turned OFF, still takes the byte-for-byte pre-RAG
single-paper path (the orchestration call is guarded on both `RAG_ENABLED` and
the toggle). `.env` enables RAG with the `memory` vector backend (full semantic
search over real embeddings, no chromadb dependency) and LM Studio embeddings.
Blank config reproduces WP0 behavior, so `RAG_ENABLED=false` remains today's
pipeline.

---

### 2026-07-20 — RAG orchestration, UI & triage (WP7): wire R0–R5 behind the flag, no logic in the app

**Files changed:** `src/t2pw/rag/triage.py`, `src/t2pw/app/streamlit_app.py`,
`tests/test_rag_triage_orchestration.py`, `docs/change_log.md`.

**What this introduces:** the final RAG work package — stage **R0** (triage) plus the
orchestrator wiring (seam **S5**) that ties R0–R5 together and exposes them, without
moving any logic into the app. `triage.py` gains `should_run_rag(context, user_flag,
reports=None) -> TriageDecision` (the *one* piece of RAG decision logic the invariant
permits outside the app): an explicit user flag always runs RAG; otherwise it
auto-triggers on a low Stage-0 `scope_clarity_score` (< 0.5) or, when the core's
read-only reports are supplied, on the WP4 gap signals (dangling reactions, orphan
metabolites, unmapped enzymes) — delegating gap classification to `retrieve.detect_gaps`
(lazy import, so `import t2pw.rag.triage` needs no chromadb); a clean, in-scope pathway
returns `run=False`. `streamlit_app.py` gains **wiring only**: a thin, importable
`maybe_run_rag(...)` helper that CALLS `acquire.search_candidates` / `fetch_full_text`
-> `select.select` -> `ingest.ingest` -> `retrieve.detect_gaps` / `retrieve_evidence` /
`format_retrieval_context` -> `synthesize.synthesize_with_report` -> `validate_provenance`
and passes their results between the seams; the UI (a "This pathway is unknown /
incomplete (enable multi-paper RAG)" checkbox, a fetched+selected papers panel from the
WP1/WP2 reports, and a provenance viewer showing source papers per reaction/entity from
the WP5/WP6 provenance). Evidence rides the **existing** seams: S1 (folded into
`user_task_context`), S2 (appended to the audit's `retrieval_context` via a new
defaulted `rag_evidence_context=""` param), and S3 (the synthesized standard `Payload`
handed to the post-pipeline path).

**Why it appeared:** WP0–WP6 built the RAG subsystem but nothing exposed it or decided
*when* it should run. WP7 is that trigger + orchestration + UI layer: the last step that
makes multi-paper RAG reachable from the app while keeping it invisible when off.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. **S5 = wiring only, no logic:**
the app contains no normalization/mapping/retrieval/synthesis logic — `maybe_run_rag`
and `render_rag_panels` only call `t2pw.rag` (and existing stage) functions and read
their returned values; the sole RAG *decision* logic lives in `t2pw.rag.triage`, not the
app. **RAG-off byte-identity (definition of done):** every RAG addition is guarded by
`if rag_config()["enabled"]` (and `maybe_run_rag` returns `None` before importing or
calling any chain function when disabled or when triage declines), so with the default
`RAG_ENABLED=false` the checkbox is not rendered, the orchestration block and UI panels
do not run, `user_task_context`/`retrieval_context`/`final_payload` are untouched, and
the app path is identical to pre-initiative `main` — the new `rag_evidence_context`
param defaults to `""` (no-op), proven by the extracted-function orchestration test
still passing unchanged. **No core stage edited:** the only non-rag file changed is
`src/t2pw/app/streamlit_app.py` (the orchestrator, in `src/t2pw/app`, not a stage dir);
`git status --porcelain` on `pipeline`/`pwml`/`curation`/`mapping`/`schema.py`/`sbml`
shows no modified files, and no stage module imports `t2pw.rag` (`grep -rn "t2pw.rag"`
over the stage dirs is empty). Tests are offline, deterministic, self-contained (the
guarded `openai` + a MagicMock `streamlit` stub let the app-helper import run alone):
they cover the triage cases and a guard/wiring test proving the RAG path is not entered
with `RAG_ENABLED` off (`acquire` is never called) yet *is* entered when enabled+flagged.
Baseline 494 + 11 new = 505 passed, 0 failures; zero new ruff (streamlit_app.py stays at
34, the added import carries a scoped `# noqa: E402` matching the file's pre-existing
sys.path-shim pattern).

---

### 2026-07-20 — RAG provenance & gates (WP6): evidence-bound validation + the gate tripwire

**Files changed:** `src/t2pw/rag/provenance.py`, `tests/test_rag_provenance_gates.py`,
`docs/change_log.md`.

**What this introduces:** stage R6 of the RAG subsystem — the layer that *proves*
the initiative's core promise, **no element without evidence**, and that a
synthesized payload survives the existing Stage 3/8 gates unmodified. `provenance.py`
(the WP0 additive-key stub) is extended with: `validate_provenance(payload) ->
ProvenanceReport` — a read-only check that every reaction (`processes.reactions`) and
every non-cofactor entity (`entities.{compounds,proteins,protein_complexes}`) carries
at least one resolvable `source_id`/`source_uri` (via any of the four additive
carriers or the core-typed `source_refs`), flagging any that do not; the
`ProvenanceReport` / `ProvenanceIssue` dataclasses that carry the result; and
`strip_provenance(payload) -> Payload` — a deep copy with every `RAG_ADDITIVE_KEYS`
key removed at any depth (input never mutated), i.e. the plain payload a RAG-unaware
or RAG-off stage sees. The cofactor exemption reuses WP5's `COFACTOR_NAMES` (lazy
import to avoid a circular dependency and to keep the module import-cheap — verified
`import t2pw.rag.provenance` still needs no chromadb).

**Why it appeared:** WP5 emits a synthesized payload with additive provenance, but
nothing yet *enforced* that every element is evidence-bound, nor *demonstrated* that
the additive keys pass the core gates untouched. WP6 is that enforcement-and-proof
step: `validate_provenance` is the guardrail that catches an unsourced (invented)
element, and the tests are the tripwire that catches anyone loosening a gate to push
RAG output through.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. **Gates called unmodified,
directly:** the tests import and call the **real** `run_strict_post_normalization_gates`
(Stage 3) and `validate_required_pwml_contract` (Stage 8) — no RAG-specific variant,
no fork, no special-case — and assert a good synthesized payload passes both, and that
the *same* gates pass on `strip_provenance(payload)` (provenance is purely additive and
ignored). **Never weaken a gate:** no stage-module file was edited (verified: `git
status --porcelain` on the stage dirs / `schema.py` / `sbml` shows no modified files;
`process_normalizer.py` and `pwml/ir.py` are byte-unchanged); the gate-ready fixture is
built *honestly* — where Stage 3/8 legitimately require a mapped payload (external DB
identities, `biological_states`, generated-complex wrappers that the core mapping stage
adds *after* seam S3, needing the offline-unavailable reference DB), the fixture supplies
that mapping result rather than stubbing the gate, and a separate test ties it back to
reality by asserting genuine (un-mapped) `synthesize` output already satisfies
`validate_provenance`. **Additive-only provenance:** `strip_provenance` removes exactly
the four `RAG_ADDITIVE_KEYS`; a test proves none survive and that both gates still pass;
the `RAG_ENABLED=false` path asserts the plain payload carries no provenance keys and the
gates behave identically. **RAG → core only:** all code lives in `t2pw.rag`; nothing in
any stage module imports `t2pw.rag` (verified: `grep -rn "t2pw.rag"
src/t2pw/{pipeline,pwml,mapping,curation,sbml}` is empty). Tests are offline,
deterministic, and self-contained (the guarded `openai` stub keeps them passing run
alone) — baseline 486 + 8 new = 494 passed, 0 failures; zero new ruff.

---

### 2026-07-20 — RAG synthesis (WP5): stitch + reconcile + resolve + provenance → one standard Payload

**Files changed:** `src/t2pw/rag/synthesize.py`, `tests/test_rag_synthesize.py`,
`docs/change_log.md`.

**What this introduces:** stage R5 of the RAG subsystem — the layer that merges the
seed extraction plus WP4's per-gap `EvidenceBundle`s into **one connected pathway**
and emits it as a **standard** `Payload` (the `TypedDict` shapes in `t2pw.schema`) at
seam **S3**. `synthesize.py` adds: `synthesize(seed_payload, evidence_bundles,
seed_context) -> Payload` (the seam-S3 entry point named in wp5_synthesis.md) and its
sibling `synthesize_with_report(...) -> SynthesisResult`, which returns the same
payload **plus** the reports that ride alongside it (`unresolved_gaps`, `conflicts`,
`stitched`, `contract_report`); and `to_payload(entities, reactions) -> Payload`,
which assembles only the core `entities`/`processes` buckets. The four synthesis
steps: (1) **stitch** — reactions stated in evidence chunks are transcribed and
connected so a product feeds the next reaction's input across papers; a dangling end
is closed *only* where a retrieved reaction supplies the missing metabolite
(cross-paper links are detected and recorded in `stitched`); (2) **reconcile
synonyms** — every name is canonicalized through the core `BIOCHEMICAL_ALIAS_MAP`
(imported **read-only** from `process_normalizer`, the same casefold-keyed lookup it
performs, reproduced without importing RAG into it); (3) **resolve conflicts** —
reactions grouped by their unordered participant set; when variants disagree on
direction / stoichiometry / compartment the highest evidence-weight variant wins and
the losers are recorded in `conflicts` (nothing dropped silently); (4) **attach
provenance** — every reaction and every non-cofactor entity carries the additive
provenance keys WP0 defined (`provenance` / `evidence` / `source_papers` /
`rag_confidence`, `RAG_ADDITIVE_KEYS`) plus a core-typed `source_refs: List[str]`
pointer. Enzymes are emitted as canonical Actor rows (`entity` / `entity_type` /
`role`). Merging is out of scope for gate *enforcement* (WP6) and orchestration/UI
(WP7).

**Why it appeared:** WP0–WP4 built the store, embedder, hybrid scorer, and gap
retrieval, but nothing yet *assembled* the retrieved evidence into a single
exportable pathway. WP5 is that assembly step — the place the "novel pathway" (a
novel *connection* of individually evidence-backed steps; docs/rag/00_overview.md) is
actually built. It reuses landed pieces rather than duplicating them: WP4's
`EvidenceBundle` / `Gap` and the store `Chunk` / `Retrieved` are consumed by **duck
typing** (only their attributes are read) so importing `synthesize` pulls neither the
retrieval/ingest stack nor chromadb; the core alias map is imported read-only; and
the output is checked with the core `validate_post_extraction` before return.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. **Standard Payload at S3:**
`to_payload` emits only the core `entities`/`processes` buckets — the shape Stage 2B
already consumes — and the output **passes `validate_post_extraction`** (the module
imports and calls it as a self-check, raising `StageContractError` on any structural
failure; it never edits or weakens that contract). **Additive-only provenance:** the
only extra keys are the four `RAG_ADDITIVE_KEYS` plus the core-owned `source_refs`;
all are optional and ignored by any stage that does not know them (a test strips
every one and the payload still passes `validate_post_extraction`) — no RAG-only
*required* key is added. Runtime shape validation runs in report mode
(`RUNTIME_SCHEMA_MODE="report"`), so the additive keys surface as non-fatal warnings,
never errors. **No invented chemistry:** every reaction and every non-cofactor entity
must carry ≥1 provenance pointer; an element with none is **omitted** and reported in
`unresolved_gaps` (a gap whose evidence bundle has no hits stays unfilled and is
surfaced — never fabricated). **No pre-running the core:** synthesis does not
normalize, map, or audit — it emits the payload and lets Stage 2B→8 run. **RAG →
core only:** all code lives in `t2pw.rag`; it imports `BIOCHEMICAL_ALIAS_MAP`,
`validate_post_extraction`, and `RAG_ADDITIVE_KEYS` from core, and nothing in any
stage module imports `t2pw.rag` (verified: `grep -rn "t2pw.rag"
src/t2pw/{pipeline,mapping,curation,pwml,sbml}` is empty). No stage-module file was
edited (verified: `git status --porcelain` on the stage dirs / `schema.py` / `sbml`
shows no modified stage files). Tests are offline, deterministic, and self-contained
(the guarded `openai` stub is only needed to build the WP4 fixtures) and pass run
alone.

---

### 2026-07-20 — RAG gap retrieval (WP4): detect gaps → query → retrieve evidence → format context

**Files changed:** `src/t2pw/rag/retrieve.py`, `tests/test_rag_retrieve.py`,
`docs/change_log.md`.

**What this introduces:** stage R4 of the RAG subsystem — the layer that turns the
core's read-only gap signals into gap-targeted evidence and renders it to a string
the existing prompts already accept. `retrieve.py` adds: `detect_gaps(payload,
reports) -> list[Gap]`, which reads `qa_graph` connectivity/degree output (both the
`generate_qa_report` `flags` shape and the CLI `dangling_nodes` /
`missing_links_suspected` / `orphan_components` shape), the Stage-3 strict gate
report's `errors` list (from `run_strict_post_normalization_gates`), and mapping
reports (entities with `status="unmapped"`), classifying each into one of
`dangling_reaction` / `orphan_metabolite` / `unmapped_enzyme` / `missing_precursor`
/ `missing_compartment` (reaction gaps enriched, read-only, with participant/enzyme
symbols from the payload); `query_for_gap(gap, seed_context) -> str`, a
natural-language ask plus the exact gene/compound symbols (so the hybrid scorer's
lexical half never loses an exact symbol); `retrieve_evidence(gap, store, *,
top_k=rag_config()["retrieve_top_k"]) -> EvidenceBundle`, which retrieves via the
WP3 `build_hybrid_scorer(store)` and keeps each hit's `source_id` / `source_uri`
provenance; and `format_retrieval_context(bundles) -> str`, which mirrors/wraps the
existing `t2pw.sbml.examples.build_retrieval_context` renderer and appends the
mandatory additive provenance line per hit. `Gap` and `EvidenceBundle` are defined
within `t2pw.rag`. Merging evidence into a final payload (WP5) and gate enforcement
(WP6) are out of scope.

**Why it appeared:** WP0–WP3 built the store, embedder, and hybrid scorer, but
nothing yet detected *which* pieces of a pathway are missing, formed queries for
them, or rendered the retrieved evidence for injection. WP4 is that missing step. It
reuses the landed pieces rather than duplicating them: the WP3
`build_hybrid_scorer` (never a second scorer), the WP0 `VectorStore` / `Retrieved` /
`Chunk`, `rag_config()` for `retrieve_top_k`, and — critically — it **wraps** the
existing renderer `t2pw.sbml.examples.build_retrieval_context` (feeding it a
synthetic single-entry index with a self-matching query so it always renders, then
swapping its `[Example i]` header for a gap-tagged `[Evidence i]` header) instead of
writing a second formatter. Offline-first holds end to end: import requires no
chromadb / network / LLM, and with the `memory` backend + a stubbed embedder the
lexical half still retrieves an exact symbol (e.g. `NdmA`).

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly, and this is the first WP to
touch the core seams. **Evidence rides only the EXISTING seam params:**
`format_retrieval_context` returns a plain **string** meant to be folded into the
already-present `pathway_context` / `user_task_context` params of
`run_extraction_pipeline` (S1) and passed to the already-present
`run_audit(..., retrieval_context=...)` param (S2) — no new parameter is added to,
and no body is edited in, `pipeline.py` or `run_audit`; the actual wiring is left to
WP7 (S5). **Reports are read-only (S4):** `detect_gaps` inspects the `qa_graph` /
gate / mapping artifacts and never writes back (a test deep-compares `payload` and
`reports` before/after and asserts they are unchanged). All new code lives in
`t2pw.rag`; the dependency arrow points RAG → core only — `retrieve.py` imports the
WP3 scorer, the WP0 store, `rag_config`, and `t2pw.sbml.examples`, and nothing in any
stage module imports `t2pw.rag` (verified: `grep -rn "t2pw.rag"
src/t2pw/{pipeline,mapping,curation,pwml,sbml}` is empty). No stage-module file was
edited (verified: `git status --porcelain` on the stage dirs / `schema.py` /
`sbml` shows no modified stage files). Tests use the `memory` backend with a stubbed
offline embedder — no chromadb, no network, no live LLM — and pass run alone.

---

### 2026-07-20 — RAG ingest & index (WP3): chunk → embed → vector store + hybrid scorer

**Files changed:** `src/t2pw/rag/ingest.py`, `tests/test_rag_ingest.py`,
`docs/change_log.md`.

**What this introduces:** stage R3 of the RAG subsystem — the layer that turns the
WP2-selected papers (plus structured DB reaction records and the existing on-disk
example corpus) into a populated, persisted `VectorStore`, and exposes the hybrid
retriever WP4 will call. `ingest.py` adds: `chunk_paper(candidate) -> list[Chunk]`
(section-aware splitting into abstract / introduction / methods / results /
discussion / figure-caption chunks of ~500–1000 tokens with overlap, each carrying
`source_id` / `source_uri` / `organism` provenance and a chunk `id` that is a
stable hash of `(source_id, section, offset)`); `chunk_db_reactions(records) ->
list[Chunk]` (one chunk per reaction, `source_type` `"pathbank"` / `"kegg"`);
`chunk_corpus(dir)` (one-or-more chunks per `reference/*.pwml` / `*.sbml` file,
tagged `source_type="pwml_example"`); `ingest(selection) -> IngestReport` (chunk →
embed via the WP0 `Embedder` → `upsert` → `persist`); and
`build_hybrid_scorer(store)`, the WP4-facing callable that blends the store's
semantic score with the lexical motif score at `0.7*semantic + 0.3*lexical`
(weights tunable). No gap detection, query formulation, or synthesis happens here —
that is WP4/WP5.

**Why it appeared:** WP0–WP2 built the store/embedder and produced a small, on-topic
set of papers, but nothing yet chunked, embedded, indexed, or retrieved them. WP3
is that missing middle. It deliberately **reuses** the landed pieces rather than
duplicating them: the WP0 `Chunk` / `VectorStore` / `get_vector_store` / `Embedder`
(the embedder's cache means an unchanged chunk is never re-embedded), the WP1
`CandidatePaper`, and — critically — it **wraps** the existing lexical layer
`t2pw.sbml.examples` (`parse_sbml` + `build_motif_entry` for corpus text extraction,
and `_score_entry` for the lexical half of the hybrid scorer) instead of writing a
second token-overlap scorer. Offline-first is preserved end to end: with no
embedding endpoint the embedder falls back to its deterministic lexical vectors, and
the hybrid scorer's lexical half guarantees an exact gene/compound symbol (e.g.
`NdmA`) is still retrieved when embeddings are unavailable. Re-ingesting an unchanged
paper is a no-op — stable chunk ids overwrite the same records and the embedding
cache reports zero new embeddings.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. All new code lives in
`t2pw.rag`; the dependency arrow points RAG → core only — `ingest.py` imports the
WP0 store/embedder, the WP1 `CandidatePaper`, and `t2pw.sbml.examples`, and nothing
in `t2pw.sbml` (or any stage module) imports `t2pw.rag` (verified: `grep -rn
"t2pw.rag" src/t2pw/{pipeline,mapping,curation,pwml,sbml}` is empty). The lexical
layer is **wrapped, not edited** — `src/t2pw/sbml/examples.py` is untouched (verified:
`git status --porcelain` on it is empty); no RAG logic was added to any stage module.
WP3 uses **no** core seam: it changes no pipeline behavior and only builds the store
and scorer that WP4 will consume. All configuration is read through `rag_config()`;
tests use the `memory` backend with a stubbed offline embedder — no chromadb, no
network, no live LLM — and pass run alone.

---

### 2026-07-20 — RAG selection (WP2): rank / dedupe / cap candidates for embedding

**Files changed:** `src/t2pw/rag/select.py`, `tests/test_rag_select.py`,
`docs/change_log.md`.

**What this introduces:** stage R2 of the RAG subsystem — the selection layer
that turns the WP1 candidate papers into the small, on-topic subset worth
embedding (WP3). `select.py` adds `score_candidate(candidate, seed_context) ->
SelectionScore`, which combines organism match, overlap of the candidate's
entities with the seed's `key_compounds` / `key_proteins` / `gap_terms` /
`pathway_name`, the preprocessor's `pathway_relevance_score`, and a penalty for a
`multi_example_review` whose examples do not match the seed. `select(candidates,
seed_context, *, max_papers=RAG_SELECT_MAX_PAPERS) -> Selection` scores every
candidate, ranks them deterministically (score desc, then paper id), dedupes by
PMCID/PMID/DOI/normalized-title (reusing `CandidatePaper.identity_keys`), caps at
`RAG_SELECT_MAX_PAPERS`, and returns the kept subset plus a `selection_report`
that gives one entry per candidate and an explicit reason for **every** drop
(duplicate, non-matching `multi_example_review` below the score floor, or below
the cap). No chunking, embedding, or retrieval happens here — that is WP3/WP4.

**Why it appeared:** WP1 over-fetches candidates from several literature APIs; if
all of them reached the (expensive) embedding step, unrelated review examples
would bleed into the corpus and the pathway synthesis would be polluted. WP2 is
the gate that stops that. It deliberately **reuses the existing preprocessor**
(`t2pw.pipeline.preprocessor.preprocess`, run per candidate on its abstract or a
truncated full text, plus `is_ambiguous_multi_example_review_context`) rather
than building a second classifier, and reuses the name-normalization / safe-access
helpers from `t2pw.mapping.map_ids`. The `multi_example_review` handling follows
`preprocess_system.txt` STEP 3 locality discipline: a review whose examples do
not match the seed is penalized so it is dropped, and one whose example *does*
match is example-scoped and ranked below on-topic primary research — never
ingested wholesale.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. All new code lives in
`t2pw.rag`; the dependency arrow points RAG → core only — `select.py` imports
from `t2pw.pipeline.preprocessor` and `t2pw.mapping.map_ids`, and nothing in
`t2pw.pipeline` (or any stage module) imports `t2pw.rag` (verified: `grep -rn
"t2pw.rag" src/t2pw/{pipeline,mapping,curation,pwml}` is empty). WP2 uses **no**
core seam: it changes no pipeline behavior and edits no stage module — it only
filters the WP1 `list[CandidatePaper]` down for WP3 to consume. Determinism is
structural: given a fixed `preprocess` output the module is pure arithmetic and
stable sorting, so a re-run yields the same ranking and report; tests mock
`preprocess` and never touch the network / LLM. All configuration is read through
`rag_config()` (`RAG_SELECT_MAX_PAPERS`); nothing is hardcoded.

---

### 2026-07-20 — RAG acquisition (WP1): candidate paper fetch + offline cache

**Files changed:** `src/t2pw/rag/acquire.py`, `tests/test_rag_acquire.py`,
`docs/change_log.md`.

**What this introduces:** stage R1 of the RAG subsystem — the acquisition layer
that turns a seed pathway context into candidate papers. `acquire.py` adds a
`CandidatePaper` dataclass (`id, source, title, abstract, organism, full_text,
source_uri, year`), `search_candidates(context, *, sources, max_papers)` which
builds organism-scoped queries from the seed context (`pathway_name`,
`likely_organism`, `key_compounds`, `key_proteins`, `gap_terms`) and fetches from
EuropePMC and NCBI eutils (with optional Crossref / Semantic Scholar / bioRxiv
sources behind the `sources` flag), and `fetch_full_text(candidate)` which
downloads `fullTextXML` and converts it to plain text. Candidates are deduped
against the seed and each other by PMCID/PMID/DOI/normalized-title, capped at
`RAG_ACQUIRE_MAX_PAPERS`, and cached on disk under
`data/rag_index/acquire_cache/` keyed by a query hash. No ranking, chunking, or
embedding happens here — that is WP2/WP3.

**Why it appeared:** the RAG initiative needs an evidence source before it can
select (WP2) or embed (WP3). WP1 is that source. It deliberately reuses the
existing HTTP plumbing in `t2pw.mapping.map_ids` (`HttpClient`,
`_europepmc_full_text`, the `_NCBI_EUTILS_BASE` / `_ncbi_eutils_params` /
`_ncbi_throttle` eutils helpers) rather than re-deriving URL, retry, or
rate-limit logic, so acquisition inherits the same session, backoff, and NCBI
throttle the core already tuned.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. All new code lives in
`t2pw.rag`; the dependency arrow points RAG → core only — `acquire.py` imports
from `t2pw.mapping.map_ids`, and nothing in `t2pw.mapping` (or any stage module)
imports `t2pw.rag` (verified: `grep -rn "t2pw.rag"
src/t2pw/{pipeline,mapping,curation,pwml}` is empty). WP1 uses **no** core seam:
it changes no pipeline behavior and edits no stage module — it only produces a
`list[CandidatePaper]` for WP2 to consume. Offline-first is honored structurally,
matching the `id_mapping_cache.json` precedent: every network fetch is fail-safe
(a missing network or API error contributes an empty list, never a raised
exception), and the per-query-hash disk cache means a re-run is served from cache
without re-hitting the network. All configuration is read through `rag_config()`;
nothing is hardcoded.

---

### 2026-07-20 — RAG subsystem foundation (WP0): package, vector store, config, provenance

**Files changed:** `src/t2pw/rag/__init__.py`, `src/t2pw/rag/store.py`,
`src/t2pw/rag/embed.py`, `src/t2pw/rag/provenance.py`,
`src/t2pw/rag/{acquire,select,ingest,retrieve,synthesize,triage}.py` (stubs),
`src/t2pw/config.py`, `requirements.txt`, `.gitignore`,
`tests/test_rag_foundation.py`, `docs/change_log.md`.

**What this introduces:** the shared scaffolding for the RAG initiative — a new,
optional `t2pw.rag` package with a `VectorStore` `Protocol` plus a `memory` and a
default `chroma` backend, an offline-capable embedding client with a JSON cache,
a `rag_config()` reader for every `RAG_*` variable, and `TypedDict` definitions
for the additive provenance keys (`provenance`, `evidence`, `source_papers`,
`rag_confidence`). No pipeline behavior changes: with `RAG_ENABLED` unset,
nothing here runs and no core module imports it.

**Why it appeared:** the RAG initiative (docs/rag/) needs a single foundation
every later work package (WP1–WP7) builds on. Landing it first, fully green and
inert, lets those packages depend on stable interfaces instead of re-deriving
them, and keeps the risky pieces (an optional heavy dependency, a network
embedder) isolated behind guards from day one.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. All RAG code lives in
`t2pw.rag`; the dependency arrow points RAG → core only (verified: `grep -rn
"t2pw.rag" src/t2pw/{pipeline,mapping,curation,pwml}` is empty). WP0 touches no
seam except adding config: no stage module was edited, and the additive
provenance keys are optional `TypedDict`s that existing stages ignore —
`t2pw.schema` is referenced, never modified. The optional-dependency and
offline-first rules are honored structurally: `chromadb` is imported lazily and
guarded (importing the package never requires it), the embedder imports the LLM
client lazily and degrades to a deterministic lexical vector when no endpoint is
reachable, and the index lives in git-ignored `data/rag_index/` like the other
rebuildable caches. This mirrors the pipeline's own rule that logic spanning
stages belongs in the orchestrator, not inside a stage — here, RAG is that
independent subsystem.

---

### 2026-07-15 — An unstated complex-component stoichiometry is blank, not an error

**Files changed:** `src/t2pw/pipeline/entity_identity.py`,
`src/t2pw/pwml/ir.py`, `src/t2pw/pwml/writer.py`, `src/t2pw/pwml/qa.py`,
`src/t2pw/mapping/map_ids.py`, `src/t2pw/curation/audit_json_llm.py`,
`src/t2pw/curation/gap_resolver.py`, `tests/test_pwml_ir.py`,
`tests/test_audit_json_llm_payload.py`, `docs/change_log.md`.

**Error / symptom:** Export blocked at the Stage 8 strict gate with four
pointer-level errors on a caffeine-degradation run:
`/entities/protein_complexes/0/components/0 - Component[0] in complex 'NdmCDE
protein complex' is missing stoichiometry.` (likewise components 1–2 and
`Cdh`). This symptom class had already been "fixed" repeatedly —
`docs/pathwhiz_requirements.md` §4.2 records it as entry #6 of an 8-entry
circular-fix chain.

**Root cause:** two independent defects.

First, the requirement was never real. PathWhiz's own
`ProteinComplexProtein` declares
`validates :stoichiometry, allow_nil: true, numericality: {only_integer: true}`
over a nullable `protein_complex_proteins.stoichiometry` column, and
`lib/pwml_parser.rb` skips any node whose content is blank. An unstated
coefficient is valid PathWhiz. The pipeline was enforcing a constraint the
target system does not have — deliberately so, since `bound_elements` *does*
declare `presence: true`. Worse, `build_pwml_ir` *dropped* the offending
component, which manufactured an empty complex and violated the rule PathWhiz
actually has (`protein_complex_proteins, length: {minimum: 1}`).

Second, five enforcement points each owned a private copy of the rule and
disagreed: `map_ids._component_stoichiometry` defaulted to 1;
`ir._component_stoichiometry` and `writer._component_stoichiometry` returned
None; `build_pwml_ir`, `validate_required_pwml_contract` and
`validate_pwml_ir` each re-implemented the check; and `pwml/qa.py` rejected an
empty value outright. A fix landing in one copy left the others intact — the
re-divergence mechanism named in `docs/pathwhiz_requirements.md` §5 items 2–3.

The deadlock was then held in place by policy: Stage 4a's
`_defer_complex_stoichiometry_patches` refuses (correctly) to fabricate a
count and defers to audit, while the audit prompt instructed the model to
"leave unresolved stoichiometry as an error for review". A paper that never
states subunit counts — the normal case — could therefore never satisfy the
gate, and each run burned an enrichment round-trip on a patch that would be
discarded anyway.

**Fix:** one shared `component_stoichiometry` in `entity_identity.py` (the
module `map_ids` and `ir` already both import) returns an explicit count or
None; mapping, IR and the writer now delegate to it instead of re-deriving it.
An unstated count is left blank end-to-end rather than assumed: `map_ids`
omits the field, `build_pwml_ir` keeps the component and omits the key, the
writer emits `<stoichiometry nil="true" type="integer"/>` via the existing
`_append_scalar` nil path, and `qa.py` skips nil nodes exactly as its
neighbouring `hmdb-id` check already did. All three IR validators now warn
(`component_stoichiometry_unstated`) instead of erroring, the writer no longer
raises, deterministic audit reports an error only when evidence gives an exact
count it can act on, and `gap_resolver` no longer holds a complex open on an
unstated count. A stated value is still preserved verbatim, and nothing infers
one.

Separately, the biological-state gate required species *and* subcellular
location as hard errors; PathWhiz's `BiologicalState#has_at_least_one_component`
requires one of species/tissue/cell_type/subcellular_location. Only a fully
empty state is now fatal; individual gaps are warnings.

**Verified:** the previously blocked payload exports (`ok = True`, zero QA
errors) with `NdmCDE` retaining all three members at `nil="true"`, `Cdh`
retaining one, and `NdmA complex` still carrying its explicit `1`. Suite: 359
passed, 0 failed (baseline 358/1; the 55 errors are a pre-existing `tmp_path`
environment fault, unchanged).

**Pipeline consistency:** field ownership is unchanged — Stage 1 records
stated evidence, Stage 4 may still patch a count from explicit evidence, and
Stage 8 remains the export authority. What changed is that the rule now has a
single definition, and that definition matches PathWhiz ground truth rather
than a stricter invention. This is the §5 item 2/3 collapse applied to one
concrete field: unknown stays blank, as the Unknown-protein sentinel already
does for identity.

---

### 2026-07-15 — Wrap name-only protein complexes with the PathBank Unknown sentinel

**Files changed:** `src/t2pw/mapping/map_ids.py`, `src/t2pw/pwml/ir.py`,
`tests/test_pathbank_unknown_fallback.py`, `tests/test_pwml_ir.py`,
`docs/pipeline.md`, `docs/change_log.md`.

**Error / symptom:** Raw, uncaught `ValueError` in `writer.py`:
`PWML export failed: Protein complex 'oxoglutarate dehydrogenase complex' has
no protein_complex-proteins to export.` — a crash deep in
`_protein_complex_members` instead of a clean export-time error, for a
`protein_complexes[]` row that reached final PWML serialization with an empty
`components` list.

**Root cause:** the 2026-07-14 "NAME-BASED COMPLEX RULE" fix (above)
correctly routes any entity whose name contains "complex" into
`entities.protein_complexes[]` even when the source paper never enumerates
subunits, using `components: []` intentionally in that case. Nothing
downstream reliably backfills that for every row: Stage 4a's gap-resolver can
only fill components from evidence text that names real subunits, and Stage
6's existing `_apply_pathbank_unknown_enzyme_fallback` — the established
pattern for "genuinely unresolvable, must not block export" proteins — is
actor-driven; it only walks `processes.reactions[].enzymes[]` and
`processes.transports[].transporters[]` looking for an exact-name match, so a
`protein_complexes[]` row never tied to exactly one reaction/transport actor
sails through untouched. Separately, `validate_pwml_ir` — the gate the
writer and `run_pwml_export` both call before serialization — only logged
`protein_complex_missing_components` as a `warning`, so `ok` stayed `True`
and nothing stopped the payload before the writer's hard crash.

**Fix:** two changes. (1) Added
`_apply_pathbank_unknown_complex_fallback` in `map_ids.py`, structurally
parallel to `_apply_pathbank_unknown_enzyme_fallback` but entity-driven: it
scans `entities.protein_complexes[]` directly, and for any row whose
`components` is still empty after normal mapping/gap-resolution/the
actor-driven fallback have all run, and which has no real complex-level
PathBank ID (`pathbank_complex_id`/`pathbank_protein_complex_id` on the row
or its `mapping_meta`), attaches a single component built from PathBank's
`Unknown` sentinel protein (id `9659`, species *Arabidopsis thaliana*),
registers that protein in `entities.proteins`, and stamps `mapping_meta`
with `chosen_rule=pathbank_unknown_protein_fallback`,
`fallback_reason="complex_has_no_resolvable_components"`, and
`cross_species_placeholder=true`. It runs only when
`allow_complex_wrapper_creation=True`, wired in immediately after the
existing actor-driven fallback call in `map_payload`, and tracks a new
`complex_missing_components_unknown_fallbacks` summary counter. (2) In
`validate_pwml_ir` (`ir.py`) and the writer's own `_protein_complex_members`
(`writer.py`), a complex still at zero components now only fails cleanly if
it also has no real, confirmed complex-level PathBank identity
(`pathbank_complex_id`/`pathwhiz_id` on the IR record) — checked directly
against `reference/PW1.pwml`, a real prior PathBank export, which contains
two `<protein-complex>` records (e.g. "alanine aminotransferase (ALT)",
`pwp-id PW_P000036`) with a genuine identity and a self-closing, empty
`<protein_complex-proteins/>`. SPMDB does not itself require every complex to
have a member; only a complex with no real identity of its own (which, after
Fix (1), should only be a generated wrapper that somehow missed even the
Unknown-sentinel fallback) is required to. `validate_pwml_ir` now raises
`error` only in that case and `warning` otherwise (previously an unconditional
`error`, which would have wrongly blocked a real, confirmed complex whose
PathBank record legitimately lists no subunits); `_protein_complex_members`
takes a matching `allow_empty` flag driven by the same check, so it no longer
raises unconditionally either. `build_pwml_ir`'s own internal bookkeeping
warning and `validate_required_pwml_contract`'s already-correct
generated-complex-only strictness (a Stage 3 pre-remap check, which runs
before any complex has a real ID to check) were both left untouched.

**Pipeline consistency:** this stays entirely inside Stage 6's existing
wrapper-creation ownership (`map_ids.py`, gated by
`allow_complex_wrapper_creation`, the sole module allowed to create
wrappers) and Stage 8's validate-only role (`ir.py`'s `validate_pwml_ir`
gates export, it does not mutate the payload). It does not reach into Stage
3's pre-remap gate or Stage 4/4a, matching "stages are independent." This is
the closing case of the same PathBank-Unknown-sentinel fallback family as
the 2026-07-13 Stage 6 entry and the 2026-07-14 "Widen the Stage 6 PathBank
Unknown fallback to cover transporter-only proteins" entry above — those
made the sentinel reachable for every *actor role*; this makes it reachable
for every *entity*, regardless of whether an actor ever references it.

**Verification:** `tests/test_pathbank_unknown_fallback.py` gained
`test_orphan_named_complex_referenced_as_enzyme_gets_wrapped` (actor-driven
and entity-driven passes agree for a referenced complex),
`test_orphan_named_complex_not_referenced_anywhere_still_gets_wrapped` (the
case the old actor-driven-only fallback missed),
`test_stage2_wrapper_disabled_does_not_wrap_orphan_complex` (disabled when
`allow_complex_wrapper_creation=False`), and
`test_complex_with_real_pathbank_id_is_not_wrapped_despite_empty_components`
(a real DB identity is never overwritten), and
`test_real_pathbank_complex_with_no_listed_components_exports_with_empty_members`
(a hand-built payload shaped exactly like `reference/PW1.pwml`'s ALT complex —
real `pathbank_complex_id`, empty `components`, never touched by `map_payload`
at all — passes Stage 3, `validate_pwml_ir` (as a warning, `ok=True`), and the
writer, producing an actual empty `<protein_complex-proteins/>` element,
proving Fix (2)'s leniency holds independently of Fix (1)). `tests/test_pwml_ir.py`
gained `test_validate_pwml_ir_errors_on_protein_complex_missing_components`
(no real identity → still an error) and kept
`test_protein_complex_unresolved_component_is_exportable_with_warnings`
passing under the corrected, identity-based rule (a complex with a real
`pathbank_complex_id` whose one listed component fails to resolve still ends
up exportable with a warning, not an error, matching the ALT-complex
precedent — component-level identity resolution failures are a separate,
already-covered concern from "does this complex exist"). Full suite re-run:
414 passing.

---

### 2026-07-14 — Tighten default reaction scope and wire the out-of-scope reaction filter

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `src/t2pw/app/streamlit_app.py`,
`tests/test_pipeline_cleanup.py`, `tests/test_streamlit_stage2_orchestration.py`,
`docs/change_log.md`.

**Error / symptom:** For pathway-dense topics (e.g. the TCA cycle), extraction returned far
more reactions than the paper's actual core pathway — anaplerotic, cataplerotic, and
auxiliary reactions mentioned only as background context were included in the final PWML
output alongside the pathway's defining steps.

**Root cause:** Two compounding gaps. (1) `pathway_scope` is a real parameter on
`run_extraction_pipeline` that the prompt's `scope_membership` rule depends on, but no live
caller ever populates it. `pwml_system.txt`'s own scope rule only defined strict
`core`-only behavior for "no `pathway_scope` AND no `upstream_context`," leaving the actual
common case — no `pathway_scope`, `upstream_context` present, true on every live run —
undefined, so the model had no explicit instruction to stay tight and would reach for the
full core/anaplerotic/cataplerotic/auxiliary taxonomy. (2) `filter_out_of_scope_reactions()`,
whose entire job is to drop reactions tagged `scope_membership: out_of_scope`, existed fully
implemented in `pipeline.py` but was never called anywhere in the live orchestrator, so even
a reaction the model correctly tagged `out_of_scope` remained in the payload through export.

**Fix:** `pwml_system.txt`'s scope rule now applies strict `core`-only classification
whenever no explicit `<pathway_scope>` is supplied, regardless of whether `upstream_context`
is present — the anaplerotic/cataplerotic/auxiliary labels are reserved for when a
`<pathway_scope>` block explicitly requests that broader taxonomy. `streamlit_app.py` now
calls `filter_out_of_scope_reactions()` on the merged Stage 1 payload immediately after
Stage 1's structural contract validates (and after `write_stage1_lock_artifacts` has already
captured the raw output and lock manifest), and before that payload reaches Stage 2
inference.

**Pipeline consistency:** Verified against `reaction_lock_manifest.py`:
`build_locked_reaction_manifest()` already excludes `scope_membership == "out_of_scope"`
reactions from locking, so the lock/preservation contract
(`reaction_preservation_validator.py`) was already designed assuming these reactions get
removed from the payload — this fix completes that assumption rather than introducing a new
one. No locked reaction is affected. Filtering between Stage 1 and Stage 2 is orchestrator-owned
cross-stage logic per `docs/pipeline.md` ("logic spanning two stages belongs in the
orchestrator"); `filter_out_of_scope_reactions` itself, already Stage-1-owned, was untouched.

**Verification:** New tests in `tests/test_pipeline_cleanup.py` (filter behavior in
isolation) and `tests/test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference`
(AST-verified call ordering). Full suite: 408 passing.

---

### 2026-07-14 — Replace unreachable multi-example-review branches with defensive text

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`,
`src/t2pw/llm/prompts/pwml_infer_system.txt`, `docs/change_log.md`.

**Error / symptom:** Both the Stage 1 and Stage 2 prompts contained a documented branch
("Case A" / "Rule 1") instructing the model to proceed with extraction/inference when
`document_type == "multi_example_review"` and `selected_example` is empty, claiming "the
upstream pipeline gate has already approved this text." This is not true of production
behavior.

**Root cause:** `is_ambiguous_multi_example_review_context()` in `preprocessor.py`, called
from `run_extraction_pipeline` and `run_inference_pipeline` in `pipeline.py`, hard-aborts the
entire pipeline with `PipelineFailure` for exactly this condition, before either stage's LLM
is ever called. The prompt branches described a state the orchestrator guarantees can never
reach the model in normal operation — dead, misleading documentation that could confuse a
future prompt editor into thinking the fallback path is live and tested.

**Fix:** Both branches now state plainly that the production orchestrator does not invoke the
stage in this state, and specify defensive behavior (return an empty extraction/no additions
with a warning flag) only for the hypothetical case of a direct or manual call that bypasses
the orchestrator gate.

**Pipeline consistency:** Prompt-text-only change; no orchestrator or normalization logic
touched. This is documentation truth-alignment to the orchestrator's actual, already-correct
hard-stop behavior, not a behavior change.

**Verification:** No automated test covers LLM prompt text in this repo (confirmed via
`rg "pwml_infer_system" tests` — zero matches); verified by inspection against `pipeline.py`'s
actual gate logic.

---

### 2026-07-14 — Standardize the transporter actor schema in the Stage 1 extraction prompt

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/change_log.md`.

**Error / symptom:** `pwml_system.txt` showed three different shapes for
`transports[].transporters[]` actor rows within the same file: `{"protein_complex": "..."}`
in the formal OUTPUT JSON SCHEMA, and `{"protein": "..."}` in Example 3 — neither matching
the canonical `entity`/`entity_type`/`role` actor shape `docs/pipeline.md` documents for
every other actor list (`reactions[].enzymes`, `reactions[].modifiers`,
`interactions[].participants`).

**Root cause:** The same bug class as the 2026-07-07 fix to `normalize_process_actor_schema`
for `reactions[].enzymes` (that entry: "all enzyme actor dicts ... retained
`protein_complex` ... as the name field" because the prompt/schema was inconsistent) —
transporters had simply never been brought in line with the canonical actor shape.

**Fix:** Both occurrences in `pwml_system.txt` now use the canonical
`{"entity": "", "entity_type": "protein | protein_complex", "role": "transporter", ...}`
shape.

**Pipeline consistency:** Confirmed by direct code read that `normalize_process_actor_schema`'s
`_rewrite_actor_rows` helper (`process_normalizer.py:2509-2515`) already canonicalizes
`transports[].transporters[]` to `entity`/`entity_type` regardless of which legacy key Stage 1
emits (it reads `entity`/`protein`/`protein_complex`/`name` as fallback fields) — this fix
changes zero bytes of what reaches export; it only removes an internally-inconsistent prompt
that could lead a weaker model to pick the wrong key or omit `entity_type`/`role`.

**Verification:** Inspected `process_normalizer.py` directly to confirm the downstream
migration path already exists and is unaffected. No automated test covers LLM prompt text.

---

### 2026-07-14 — Make Stage 1 prompt examples schema-valid and consolidate complex-routing guidance

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/change_log.md`.

**Error / symptom:** Two related prompt-quality gaps in `pwml_system.txt`: (1) all four
MODIFIER EXAMPLES omitted required top-level reaction/interaction/transport fields
(`biological_state`, `class`, `scope_membership`, `confidence`, `provenance`,
`source_refs`) that the formal OUTPUT JSON SCHEMA requires — a weaker model imitating the
examples rather than the full schema block could emit incomplete objects. (2) protein-vs-
protein_complex routing guidance (NAME-BASED COMPLEX RULE, extraction-layer cofactor-rule
bullets) was scattered across roughly 450 lines of the file with no single, locally-visible
decision procedure — the same root cause named in the earlier 2026-07-14 "Stop LLM extraction
from routing 'X complex' entities into proteins[]" entry ("that guidance was buried among many
other extraction rules").

**Fix:** All four MODIFIER EXAMPLES (enzyme catalyst, regulator/catalyst with interactions,
transporter, protein complex catalyst) now include every field the formal schema requires, so
each is a complete, valid instance of its schema rather than an abbreviated illustration.
Added a new "PROTEIN-COMPLEX DECISION CHECKLIST" section immediately before the examples,
consolidating the existing scattered rules (name-contains-"complex" check, explicit
multi-subunit language check, components explicit-vs-unresolved handling, never infer a
complex for export-wrapper reasons) into one explicit, numbered procedure. The existing
scattered rules were left in place as reinforcement rather than removed, to avoid risking any
rule a downstream check implicitly depends on the model having seen phrased a specific way.

**Pipeline consistency:** Prompt-text-only; no schema, normalizer, or export code touched. The
checklist restates existing policy already enforced downstream by Stage 3's NAME-BASED
COMPLEX RULE gate and Stage 6's generated-wrapper contract — it does not introduce new policy.

**Verification:** No automated test covers LLM prompt text; verified by inspection that every
added field matches `t2pw/schema.py`'s existing TypedDict definitions exactly (no new fields
invented).

---

### 2026-07-14 — Remove the live Homo sapiens organism default from the Stage 2 inference prompt

**Files changed:** `src/t2pw/llm/prompts/pwml_infer_system.txt`, `docs/change_log.md`.

**Error / symptom:** `pwml_infer_system.txt` section E ("Biological state and location
linking") still instructed the model to default an unresolved organism to "Homo sapiens" —
directly contradicting Stage 1's own species rule ("if no organism can be confidently
selected, leave species empty — do not guess") and Stage 0's preprocessor, both of which
already forbid organism guessing. This is the same contradiction the 2026-07-08 change log
entry fixed for Stage 1's own BIOLOGICAL STATE RULE — that fix never reached this second copy
in the Stage 2 prompt.

**Root cause:** The 2026-07-08 fix ("Strengthen extraction scoping...") only touched
`pwml_system.txt` and added a species cross-reference note to `pwml_infer_system.txt`'s
modifier-linking section; the separate default-organism line in that same file's
biological-state/location section was never located or updated.

**Fix:** Replaced the Homo sapiens default with an explicit priority order (upstream-selected
organism → locally-evidenced organism → empty) and an explicit prohibition on defaulting,
noting that an unresolved species is a valid Stage 3 gate finding the Stage 4 audit loop is
designed to repair from real evidence.

**Pipeline consistency:** Prompt-text-only; matches the already-existing Stage 3 species gate
(added 2026-07-08, "Add protein species and external identity checks to Stage 3 gate") plus
the Stage 4 audit repair path, both unaffected. No code touched.

**Verification:** No automated test covers LLM prompt text; confirmed the pre-existing
uncommitted section C (alias/synonym bridging) addition to this same file — added by
concurrent, unrelated work on cofactor charge-notation canonicalization — was left untouched.

---

### 2026-07-14 — Retire dead prompt files and their dead loader code

**Files changed:** `src/t2pw/llm/prompts/extract_json.md` (deleted),
`src/t2pw/llm/prompts/repair_json.md` (deleted),
`src/t2pw/llm/prompts/enrichment_system.txt` (deleted),
`src/t2pw/curation/gap_resolver.py`, `docs/change_log.md`.

**Error / symptom:** Three prompt files sat in `src/t2pw/llm/prompts/` with no live reference
anywhere in the codebase: `extract_json.md` and `repair_json.md` were both 0 bytes and
unreferenced by filename anywhere in `src/`. `enrichment_system.txt` was a real, substantive
prompt (patch-based, non-agentic enrichment) loaded only by `_get_enrichment_system_prompt()`
in `gap_resolver.py`, a function that was itself never called anywhere — the live Stage 4a
enrichment path uses the separate, actually-wired `enrichment_agentic_system.txt` /
`_get_enrichment_agentic_system_prompt()`.

**Root cause:** Vestigial from an earlier prompt-per-stage design iteration; nothing removed
them when the live enrichment path moved to the agentic/tool-calling variant.

**Fix:** Deleted all three files after independent re-verification (fresh repo-wide grep, not
reliance on prior analysis) confirmed zero references. Removed the now-fully-dead
`_get_enrichment_system_prompt()` function and its module-level `_ENRICHMENT_SYSTEM_PROMPT`
global from `gap_resolver.py`. Left `src/t2pw/config.py` untouched: although it is 0 bytes,
`src/config.py` (a separate legacy shim) does `from t2pw.config import *`, so deleting it
would break that shim's import — this dependency was found during re-verification and the
file was correctly left in place.

**Pipeline consistency:** Removes dead code only; no live prompt, stage function, or call path
touched. `enrichment_agentic_system.txt` (the live Stage 4a prompt) was explicitly left
untouched and re-verified as the sole caller-reachable enrichment prompt.

**Verification:** Repo-wide grep for each deleted filename and the removed function/global
name returned zero remaining source references. `python -m py_compile
src/t2pw/curation/gap_resolver.py` succeeded.

---

### 2026-07-14 — Charge-notation-aware alias canonicalization; interaction registry coverage

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/llm/prompts/pwml_infer_system.txt`, `src/t2pw/llm/prompts/pathway_curator_system.txt`,
`src/t2pw/curation/audit_json_llm.py`, `tests/test_process_normalizer.py`,
`docs/pipeline.md`, `docs/change_log.md`.

**Error / symptom:** Stage 8 IR construction failed with 14 `Process member 'X' was
not found in entity registries.` errors (for `NAD`, `NADP`, `Ca2`) and 3
`Interaction must have exactly one left and one right member.` errors, on a TCA-cycle
paper run. `entities.compounds` correctly declared the redox-specific species
(`nad+`, `nadh`, `nadp+`, `nadph`), but 9 reactions referenced the bare, ambiguous
tokens `"NAD"`/`"NADP"`, and two interactions were self-referential `SAME_AS`
declarations (`entity_1 == entity_2 == "NAD"`/`"NADP"`, evidently a failed attempt to
declare "NAD" and "NAD+" as synonyms) plus one interaction (`Ca2+ activates IDH and
OGDH`) referenced calcium, which was never extracted as a compound at all.

**Root cause:** four compounding issues, none of which is a "missing entity" problem
for NAD/NADP:

1. `process_normalizer._normalize()` stripped the `+` character while
   `t2pw.pwml.ir._norm()` (Stage 8) preserved it. Stage 3/4's registry check
   (`validate_registry_references`, exposed to audit via `_stage3_validation_issues`)
   therefore treated `"NAD"` and `"nad+"` as the same name and never flagged the
   mismatch — the payload reached Stage 8 looking clean, where the stricter,
   charge-aware `_norm()` correctly rejected it with no repair path left.
2. `apply_biochemical_aliases` (step 1) already had the exact fix in
   `BIOCHEMICAL_ALIAS_MAP` (`"nad": "NAD+"`, `"nadp": "NADP+"`) and already rewrote
   reaction inputs/outputs, but never touched `processes.interactions[]` participants
   or `processes.transports[].cargo` — the same class of bare compound reference was
   simply never in scope for this pass.
3. Independently of (1)/(2), `_rewrite_token` and `_token_parts_for_aliasing` (used by
   step 11, `canonicalize_same_as_aliases`, which runs on every payload regardless of
   whether it contains a `SAME_AS` interaction) tested for a composite `"A + B"`
   token with a raw `"+" in text` check instead of the charge-aware `_has_plus_token`
   guard already used elsewhere (`normalize_composites`). This silently mangled any
   correctly-charged compound name — `"NAD+"` was split on `"+"` into `["NAD", ""]`,
   the empty part dropped, and the single remaining part rejoined with no `+` at all.
   This is the actual, deterministic reason `"NAD+"` never survived to Stage 8 even
   after `apply_biochemical_aliases` had just correctly produced it.
4. `validate_registry_references` never checked `processes.interactions[]` at all
   (only reactions and transports), so the genuinely-missing `Ca2+` compound, and the
   self-referential `SAME_AS` rows, had no path to becoming a Stage 3 gate failure —
   they were invisible to the audit loop and only surfaced as a hard Stage 8 abort.

**Fix:**
1. `_normalize()` now preserves `+`, matching `ir.py`'s `_norm()`.
2. `apply_biochemical_aliases` now also rewrites `processes.interactions[]`
   (`entity_1`/`entity_2`/`left`/`right`/`source`/`target`) and
   `processes.transports[].cargo`/`cargo_complex`.
3. `_rewrite_token` and `_token_parts_for_aliasing` now use `_has_plus_token` instead
   of a raw `"+" in text` check, so charge notation on compound names is never
   mistaken for a composite separator during alias canonicalization.
4. `canonicalize_same_as_aliases` now drops a `SAME_AS` interaction whose two sides
   normalize to the same name after rewriting (including a degenerate declaration
   that was self-referential to begin with) instead of carrying forward an inert
   self-interaction.
5. `validate_registry_references` now checks `processes.interactions[]` participants
   against the registry, mirroring the existing reactions/transports coverage.
6. `_entity_name_norms` now also recognizes an entity's declared `synonyms`, matching
   `ir.py`'s existing alias resolution, so a curator/audit-proposed synonym patch is
   honored by the gate and not only by export.
7. Prompt updates (defense in depth, not required for the deterministic fix above):
   the Stage 2A inference prompt now requires `SAME_AS` pairs to use two distinct
   literal strings and prefers the deterministic charge-form directly for known
   cofactors; the Stage 5 curator prompt now covers interaction participants (not
   just reaction inputs/outputs) and prefers an entity `synonyms` patch over editing
   a reaction/interaction reference when the entity's declared name is already
   correct; the Stage 4 audit system prompt gained equivalent "registry reference
   mismatch" guidance (synonym-patch first, new-entity only when genuinely absent,
   remove degenerate self-referential `SAME_AS` rows).

**Pipeline consistency:** all deterministic changes stay inside
`process_normalizer.py`, which owns Stage 3's alias canonicalization and gate. None
of them reach into Stage 6/8 or invent biology — they only make an existing,
already-tested deterministic mechanism (biochemical alias rewriting, same-as
canonicalization, registry validation) correctly cover interactions and correctly
preserve chemically-significant `+` notation, so a repairable naming issue is caught
and fed to Stage 4 audit (per "the gate is not a blocker before audit") instead of
surfacing as an unrepairable Stage 8 abort. The genuinely-missing `Ca2+` entity is
still left as an audit-owned gap — no stage invents the missing compound.

**Verification:** `tests/test_process_normalizer.py` gained
`test_apply_biochemical_aliases_rewrites_interaction_and_transport_participants`,
`test_validate_registry_references_flags_unknown_interaction_participant`,
`test_validate_registry_references_allows_known_interaction_participant`,
`test_validate_registry_references_recognizes_declared_synonyms`,
`test_canonicalize_same_as_aliases_preserves_charge_notation`,
`test_canonicalize_same_as_aliases_drops_noop_same_as_interaction`, and
`test_full_normalization_resolves_bare_cofactor_and_flags_missing_ion` (an
end-to-end reproduction of the exact TCA-cycle payload shape: bare "NAD" input
resolves to "NAD+", the self-referential alias interactions are dropped, and the
genuinely-missing Ca2+ reference correctly surfaces as a gate error). Verified
directly against the failing run's saved `tmp/final.mapped.json`: before the fix,
`validate_registry_references` raised on all 14 tokens exactly matching the reported
Stage 8 errors; after the fix, a reduced reproduction of the same reactions/
interactions normalizes to zero gate errors except the genuine Ca2+ gap. Full suite
re-run: 405 passing (398 previously existing + 7 new).

---

### 2026-07-14 — Prefer a confident Stage 6 DB complex match's components over stale extraction data

**Files changed:** `src/t2pw/mapping/map_ids.py`, `tests/test_map_ids.py`,
`docs/pipeline.md`, `docs/change_log.md`.

**Error / symptom:** Stage 8 export error `Component[0] in complex 'pyruvate
dehydrogenase complex' is missing stoichiometry.` (and `[1]`, `[2]`) — for a
complex that Stage 6 had, in the same run, successfully matched to a real
PathBank record. This surfaced immediately after the 2026-07-14 "NAME-BASED
COMPLEX RULE" fix (below) started correctly routing this entity into
`entities.protein_complexes` for the first time; it had never reached this
code path before because it used to be misfiled under `proteins[]` with no
`components` at all.

**Root cause:** two upstream stages compound into a gap Stage 6 didn't cover.
Stage 1 extracts `components` as plain subunit-name strings (its schema has no
concept of stoichiometry). Stage 4a's gap-resolver
(`gap_resolver.py:_resolve_declared_complex_components`) tries to attach a
protein identity to each subunit by name-matching against
`entities.proteins`; when a subunit was never separately extracted as its own
protein row (true here — the paper only names E1/E2/E3 as complex members),
that lookup fails, but the function *still* unconditionally upgrades the
plain string into a dict (`{"name": ...}`) and writes it back, flagging
`missing_stoichiometry` with `resolution_owner: "audit"`. Stage 4's
deterministic audit rule can only backfill stoichiometry from an *explicit*
per-subunit count stated in the evidence text (its precedent case: "three
NdmC, three NdmD..."); this paper's evidence never states a count, so nothing
fills it in. Stage 6 then DB-matches the complex to a real PathBank record
(`pathbank_complex_id`) whose components carry real, correct stoichiometry —
but the mapping loop in `map_payload` was hard-coded to keep
`complex_row`'s existing (by-then broken) components whenever they were
non-empty, discarding the DB match's authoritative data outright.

**Fix:** in `map_payload`'s per-complex loop, when the Stage 6 mapping result
has `status == "mapped"` — a confident match via direct
`pathbank_protein_complex_id`, name+species, or resolved-component-species —
the DB-hydrated `result["components"]` (already reconciled against local
`entities.proteins` earlier in the same loop iteration) now overwrites
`complex_row["components"]` outright. Every other outcome (`unmapped`,
`ambiguous`, `novel`, and the PathBank `Unknown`-sentinel fallback, all of
which carry a non-`"mapped"` status) is unaffected — extraction/gap-resolver
components are still preferred there, since there is no more-authoritative
DB version to prefer.

**Pipeline consistency:** this stays entirely inside `map_ids.py`, the sole
Stage 6 module. The two upstream gaps that let a stoichiometry-less dict
component reach Stage 6 in the first place (Stage 4a always promoting
strings to dicts without a stoichiometry fallback, and Stage 4's audit only
backfilling from explicit textual counts) were deliberately left untouched —
this fix does not reach into Stage 3, 4, or 4a, per "stages are independent."
It only changes which of two already-available component lists Stage 6
prefers when it has legitimate grounds (a confident DB identity) to prefer
one over the other.

**Verification:** `tests/test_map_ids.py` gained
`test_confident_db_complex_match_overrides_stale_extraction_components`
(a complex with plain-string, unresolvable components and a mocked confident
DB match ends up with the DB's stoichiometry-bearing components) and
`test_unconfident_db_complex_match_keeps_extraction_components` (an
ambiguous/unmapped result leaves the original extraction components
untouched — current behavior preserved). No existing test asserted the old
precedence (checked every test that mocks Stage 6 complex-matching). Full
suite re-run: 398 passing.

---

### 2026-07-14 — Stop LLM extraction from routing "X complex" entities into proteins[]

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/pipeline.md`,
`docs/change_log.md`.

**Error / symptom:** Stage 3 gate error `Generated protein complex wrapper
'pyruvate dehydrogenase complex' must be listed under protein_complexes, not
proteins.` for an entity that was never a pipeline-generated wrapper at all —
it is a real, well-known multi-subunit enzyme complex.

**Root cause:** the Stage 1 extraction prompt already told the LLM to use
`protein_complexes[]` when the source text "explicitly supports a complex,"
but that guidance was buried among many other extraction rules and had no
single unmissable rule tied to the literal entity name. The model extracted
an entity named "...complex" directly into `proteins[]`. The Stage 3 gate
that reports this (`process_normalizer.py:3766`, unchanged by this fix)
correctly detects any `proteins[]` row named "...complex" as suspicious, but
it is detection-only — there is no auto-repair step for this class of
misclassification (only pipeline-generated wrapper duplicates are guarded).

**Fix:** added one explicit, mandatory rule to the Stage 1 prompt: any entity
whose own name contains the word "complex" must be extracted under
`protein_complexes[]`, never `proteins[]`, even when the source text does not
enumerate subunits (in which case `components: []` and confidence `< 1.0` are
used, mirroring the existing "unknown subunit membership" convention already
in the same prompt).

**Pipeline consistency:** this is a Stage 1 (Extract) prompt change only —
the entity-type decision belongs at extraction, where `PayloadProtein` vs.
`PayloadProteinComplex` is first assigned. It does not touch Stage 3's gate
logic, Stage 6's remap logic, or any other stage's module, per the "stages
are independent" design principle.

**Verification:** no automated test covers LLM prompt text in this repo
(confirmed via `grep -rn "pwml_system.txt" tests/` — no matches); the fix was
verified by inspection against the file's existing rule conventions and by
re-running the full test suite to confirm nothing else references or
snapshots this file's content.

---

### 2026-07-14 — Widen the Stage 6 PathBank Unknown fallback to cover transporter-only proteins

**Files changed:** `src/t2pw/mapping/map_ids.py`,
`tests/test_pathbank_unknown_fallback.py`, `docs/pipeline.md`,
`docs/change_log.md`.

**Error / symptom:** Stage 3 gate error `Protein 'ABCG-116' is missing a
UniProt or DrugBank identifier.` for a protein that legitimately could not be
matched to a real identifier — the same situation the Stage 6 PathBank
`Unknown` sentinel fallback exists to handle for enzymes, but this protein's
only role in the payload was as a transporter, not a reaction catalyst.

**Root cause:** `_apply_pathbank_unknown_enzyme_fallback` in `map_ids.py` was
deliberately scoped to reaction enzymes only (see the 2026-07-13 Stage 6
entry above and `docs/pipeline.md` Stage 6 section). `_has_non_enzyme_reference`
disqualified any protein referenced outside a catalyst role — including as a
transporter — from ever receiving the fallback, so an unresolved
transporter-only protein was left with zero identifiers and no path to a
valid export state.

**Fix:** generalized the disqualification check to accept a caller-specified
"allowed role" (`enzyme` or `transporter`) instead of hard-coding "enzyme."
Added a second pass over `processes.transports[].transporters[]` that applies
the identical guards already used for enzymes (only after real mapping
strategies fail, never overrides a real mapping, reused/deduplicated Unknown
sentinel, excluded once any other disqualifying reference exists) and rewrites
a qualifying transporter entry to reference the generated Unknown-backed
complex the same way a qualifying reaction enzyme entry is rewritten. A
protein referenced as a transporter *and* anywhere else disqualifying (reaction
input/output, non-catalyst/non-transporter modifier, interaction, complex
component) remains excluded, matching the existing enzyme-side behavior. A
new `transporter_unknown_fallbacks` counter tracks this path separately from
the existing `reaction_enzyme_unknown_fallbacks` counter so no existing
caller's assertion on that counter changes meaning.

**Pipeline consistency:** this stays entirely inside `map_ids.py`, the sole
Stage 6 module allowed to call `map_payload(..., allow_complex_wrapper_creation=True)`.
It does not touch Stage 3's gate (which correctly just reports the symptom)
or any other stage. The fallback's core invariants are unchanged and merely
extended to a second, symmetric role — it still never applies to a protein
with any other kind of reference, and it is still the sole wrapper-creating
pass in the pipeline.

**Verification:** `tests/test_pathbank_unknown_fallback.py` gained
`test_unknown_fallback_wraps_transporter_only_protein` (transporter-only
unresolved protein gets wrapped and its transporter entry is rewritten to
`entity_type: "protein_complex"`) and
`test_unknown_fallback_excludes_transporter_referenced_elsewhere` (a
transporter also referenced via an interaction stays excluded, matching the
existing enzyme-side exclusion test). Full test suite re-run: all passing.

---

### 2026-07-14 — Retire the spontaneous-reaction flag; every reaction exports non-spontaneous

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`,
`src/t2pw/curation/audit_json_llm.py`, `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/pwml/ir.py`, `src/t2pw/pwml/qa.py`, `tests/test_pwml_ir.py`,
`tests/test_audit_json_llm_payload.py`, `docs/pipeline.md`,
`docs/change_log.md`.

**Error / symptom:** Stage 8 export error
`Reaction 'OPCL1-catalyzed CoA ligation of OPC-4:0' is marked spontaneous but
also has enzymes.` — Stage 1 extraction (or the Stage 4 audit's deterministic
enzyme-less rule) could mark a reaction `spontaneous: true` independently of
whether it also carried real enzyme references, and the Stage 8 semantic gate
(`ir.py`) rejected the combination outright with no repair path.

**Root cause:** the `spontaneous` field could be set from three independent
places (Stage 1 LLM extraction judgment, Stage 4's deterministic
enzyme-less-reaction rule, and manual/legacy payload data) with nothing
reconciling it against the reaction's actual enzyme list until the Stage 8
export gate, which only detects the conflict and aborts.

**Fix:** spontaneity is not modeled for now. Every source that could set
`spontaneous: true` was changed to never do so (Stage 1 prompt instruction,
Stage 4's deterministic audit rule removed), Stage 3's normalizer now forces
`spontaneous: false` on every reaction as its first step so the persisted
normalized payload is consistent, and Stage 8's IR builder hardcodes
`spontaneous: False` on export regardless of upstream payload content. The
now-unreachable Stage 8 mutual-exclusion check
(`spontaneous_reaction_has_enzymes`) was removed. The companion legacy XML QA
check in `qa.py` that required every non-spontaneous reaction to have an
enzyme was relaxed to match — an enzyme-less reaction is now expected, not an
error, since spontaneity can no longer be asserted to explain it.

**Pipeline consistency:** the enforcement point that matters for correctness
is Stage 8 (export), which now owns the invariant unconditionally rather than
validating an upstream assertion. Stage 1's prompt and Stage 4's audit rule
were also updated so the persisted payload stays consistent with what
actually exports, but Stage 8 does not depend on either of them having done
so correctly — it forces the value itself, per "broken stages must not
produce output" / each stage should not trust an earlier stage's optional
field to be correct.

**Verification:** `tests/test_pwml_ir.py::test_spontaneous_field_is_always_forced_false_on_export`
and `test_pre_export_and_qa_reject_duplicate_enzyme_complex_even_when_spontaneous_set`
(renamed/updated from the prior spontaneous-preserving tests) and
`tests/test_audit_json_llm_payload.py::test_deterministic_audit_does_not_mark_enzyme_less_reaction_spontaneous`
(renamed/updated) all pass, along with the full related test suite (96 tests
across `test_pwml_ir.py`, `test_audit_json_llm_payload.py`, and
`test_process_normalizer.py`).

---

### 2026-07-13 — Quarantine coarse reactions before orphan-protein validation

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`,
`tests/test_locked_noop_quarantine_policy.py`, `docs/pipeline.md`, and
`docs/change_log.md`.

**Error / symptom:** A new paper began with 21 locked source reactions. Stage 2
preserved all 21, but Stage 3 retained only nine and reported the other 12 as
missing. The 12 all used the same coarse compound label on both sides. The
`KlpD` claim, for example, was represented as `klebsazolicin -> klebsazolicin`.
After that row was silently removed, its unmapped `KlpD` protein remained and
the pre-export Stage 3 revalidation stopped on a missing UniProt/DrugBank ID.

**Root cause:** `dedupe_processes` equated identical/subset normalized labels
with a biochemical no-op and deleted the row without considering lock
accounting. It also ran after orphan cleanup, so an enzyme could become orphaned
only after the cleanup passes had finished. `essential` exempted self-loops from
classification, and distinct locked duplicates could be silently collapsed.
The Stage 6 PathBank `Unknown` fallback was correctly narrow: because the KlpD
reaction no longer survived, it had no valid reaction enzyme to process.

**Fix:** Stage 3 now classifies same-label and output-subset reactions before
the final orphan passes. Unsupported unlocked no-ops are dropped. Locked or
directly evidenced coarse reactions are removed from active processes and
written to the existing `quarantined_locked_reactions` ledger with stable reason
codes, the original reaction/provenance, source JSON pointer, and action. Neither
`locked` nor `essential` forces a biologically invalid equation into export.
Distinct locked duplicates retain one active representative and account for the
other lock in quarantine. The final orphan cleanup then removes unmapped
proteins used only by rejected reactions while retaining proteins referenced by
surviving reactions, interactions, transports, or complexes. Preservation
validation consequently treats the intended result as active plus quarantined,
with no silently missing locks. The strict Stage 3 gate now rejects a positive,
negative, or malformed `unaccounted_locked_reactions` value at
`/locked_reaction_filter_report/unaccounted_locked_reactions`; zero accounted
locks pass. After normalization and before audit, the live orchestrator also
rewrites canonical `tmp/quarantined_locked_reactions.json` from the normalized
ledger and returns it through post-pipeline JSON artifacts, so stale
pre-normalization content cannot mask quarantine results.

**Pipeline consistency:** Stage 3 owns deterministic reaction classification,
quarantine, and post-classification cleanup. Stage 4 may restore a quarantined
claim only from evidence that establishes distinct biochemical participants.
Stage 6 remains responsible for the narrow PathBank `Unknown` fallback only
after normal identity mapping fails for a confirmed catalyst on a valid
surviving reaction. Mapping failure does not remove that valid reaction, and the
fallback is not broadened to unrelated orphan proteins.

**Verification:** Focused regressions cover the KlpD-shaped locked/essential
self-loop, unlocked no-op removal, active-plus-quarantined preservation
accounting, duplicate locked IDs, post-quarantine orphan cleanup with all
surviving reference types, and Stage 6 fallback eligibility for an unresolved
enzyme on a valid distinct-participant reaction. Additional boundary tests cover
strict-gate accounting enforcement and post-normalization replacement of a
stale canonical quarantine artifact before audit.

---

### 2026-07-13 — Make Stage 8 validation-only and preserve Stage 6 enzyme wrappers

**Files changed:** `src/t2pw/app/streamlit_app.py`,
`src/t2pw/pipeline/process_normalizer.py`, `src/t2pw/pwml/ir.py`,
`tests/test_streamlit_stage8_export_contract.py`,
`tests/test_process_normalizer.py`, `tests/test_pwml_ir.py`,
`docs/pipeline.md`, and `docs/change_log.md`.

**Error / symptom:** PWML export of the saved final mapping stopped with
repeated bare-protein enzyme errors for actors such as `NdmA` and `NdmB`.
Inspection of `tmp/final.mapped.json` proved that Stage 6 had already done the
right work: it contained structurally valid generated single-protein PathWhiz
wrappers (`NdmA complex`, `NdmB complex`, and peers), and the reactions
referenced those wrappers.

**Root cause:** `run_pwml_export` reran the full Stage 3 normalizer after Stage
6. Reaction-evidence attachment compared exact actor names, did not recognize a
valid generated wrapper as equivalent to its sole member, and re-added bare
`NdmA`/`NdmB` actors beside their complex wrappers. Actor mirroring into both
`enzymes` and `modifiers`, together with a required-contract exemption, allowed
those bad actors through that gate; IR validation then emitted the repeated
bare-protein failures.

**Fix:** Stage 8 now performs validation only after optional grounding. It does
not rerun normalization, create autostates, attach/promote actors, map or infer
entities, or create wrappers. Evidence attachment is wrapper-aware and treats
only generated, structurally valid one-protein wrappers as equivalent to their
member, making supported reruns idempotent without conflating ordinary
biological complexes with proteins. The pre-export and direct-IR contracts now
reject bare catalytic proteins at their own actor pointers, accept one canonical
cross-field `enzymes`/`modifiers` mirror as a single logical enzyme, and still
reject duplicates within either field. IR construction likewise emits that
cross-field mirror only once.

**Pipeline consistency:** Stage 3 remains the sole full normalization stage,
and Stage 6 remains the sole owner of wrapper creation and reaction remapping.
Stage 8 validates and serializes the exact post-remap payload without repairing
it or inventing export structure.

**Verification:** With valid pathway metadata supplied, the exact saved
`tmp/final.mapped.json` now exports with `ok=true`, writes the output, and has
zero required-contract and IR-validation errors. The focused merged suite
passes 127 tests; the full suite passes 380 tests, and Ruff, compile, and diff
checks are green. A live UI click is the remaining manual check.

---

### 2026-07-13 — Repair named-complex resolution, audit convergence, and organism-aware locations

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/curation/gap_resolver.py`, `src/t2pw/curation/audit_json_llm.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_process_normalizer.py`,
`tests/test_gap_resolver.py`, `tests/test_gap_resolver_stage3_issues.py`,
`tests/test_gap_resolver_agent_tools.py`, `tests/test_audit_json_llm_payload.py`,
`tests/test_streamlit_stage2_orchestration.py`, `docs/pipeline.md`, and
`docs/change_log.md`.

**Error / symptom:** The `NdmCDE` paper run reached final remapping but stopped
at pre-export Stage 3 revalidation. Normalization duplicated the declared
complex as a bare protein, Gap Resolve skipped the complex as `issue_not_found`,
the audit loop stopped after one round despite gap progress, and bacterial
compounds received eukaryotic organelle locations.

**Root cause:** Protein synthesis did not consult the complex registry; the gap
executor indexed only proteins and compounds; convergence considered audit
patch counts but not gap-only changes/unresolved issues; audit had no
evidence-safe component-ratio repair; and location ranking used global frequency
without organism compatibility.

**Fix:** Stage 3 now preserves complex identity and gates protein/complex name
collisions. Gap Resolve executes complex issues, hydrates member identity from
declared mapped proteins, reports unsupported ratios to audit, treats a valid
novel complex ID as optional, and filters incompatible organelles. Stage 4 audit
adds exact component ratios only from unambiguous evidence (including the
`NdmCDE` `3/3/3` sentence) and canonicalizes positive legacy coefficients. The
orchestrator gates every changed settled payload, includes unresolved gap issues
in convergence, preserves loop bounds, and labels final failures as pre-export
Stage 3 revalidation.

**Pipeline consistency:** Stage 3 owns deterministic entity-type invariants;
Stage 4a owns targeted member/ID/location resolution; Stage 4 audit owns
evidence-dependent ratios; the orchestrator owns convergence; Stage 6 remaps;
and Stage 8 remains the export guard. No stage invents missing biology.

**Remaining validation:** A fresh live run completed the configured DB/LLM
stages and produced the final Stage 6 artifact. The separate Stage 8 regression
found on export is fixed above, and the exact saved artifact now passes
programmatic PWML export. Only a final manual export click in the live UI
remains.

---

### 2026-07-13 — Process-aware extraction contracts and visible boundary failures

**Files changed:** `src/t2pw/pipeline/stage_contracts.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_stage_contracts.py`,
`tests/test_streamlit_stage2_orchestration.py`, `docs/pipeline.md`, and
`docs/change_log.md`.

**Error / symptom:** Clicking **Run audit and DB mapping** could stop with only
"Every extracted process must include inputs, outputs, or cargo." Database
mapping and audit never ran, the structured issue pointer was hidden, and a
valid interaction could trigger the same reaction-oriented error.

**Root cause:** The post-extraction contract applied one reaction/transport
participant rule to every process bucket. The Streamlit handlers also caught
`StageContractError` as a generic exception and displayed only its summary
message, despite the exception carrying a structured report.

**Fix:** Post-extraction validation now dispatches bucket-specific structural
rules for reactions, transports, interactions, reaction-coupled transports,
and sub-pathways while keeping unknown additive buckets object-safe. Both live
post-pipeline handlers render contract failures separately with the exact
boundary, skipped stages, issue codes and JSON pointers, full report, and a
downloadable JSON report. A new run also clears stale successful artifacts
before executing.

**Pipeline consistency:** Genuine structural failures still abort before
mapping and audit, but valid process shapes are no longer rejected by another
process type's rule. The orchestrator exposes the contract state without
repairing or silently changing the payload.

---

### 2026-07-13 — Runtime payload reports and live Stage 2 mapping boundary

**Files changed:** `src/t2pw/pipeline/payload_models.py`,
`src/t2pw/pipeline/stage_contracts.py`, `src/t2pw/mapping/map_ids.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_payload_models.py`,
`tests/test_stage_contracts.py`, `tests/test_stage2_mapping_boundary.py`,
`tests/test_streamlit_stage2_orchestration.py`, `docs/pipeline.md`, and
`AGENT_INSTRUCTIONS.md`.

**Error / symptom:** The live Streamlit post-pipeline path normalized the
merged extraction/inference payload before any Stage 2 database mapping. Its
only mapping call was the post-curation Stage 6 remap, so the UI could not show
the exact mapped payload entering Stage 3 or distinguish early mapping misses
from later wrapper creation. TypedDict documentation also did not catch nested
runtime type errors in full payloads.

**Root cause:** The documented Stage 2 boundary had not been wired into the
orchestrator, and mapper compatibility behavior combined annotation, wrapper
creation, and structural cleanup. Boundary contracts covered selected
structural and semantic invariants but did not recursively validate known
container/value shapes at runtime.

**Fix:** Added non-mutating Pydantic runtime models with stable JSON-pointer
reports and report/enforce modes, integrated through the stage-contract
adapter. The live Stage 2B call now uses cache with wrapper creation and
structural cleanup explicitly disabled, requires object payload/report results,
and validates the nested `mapping_meta.resolution` shape before Stage 3. Stage
6 explicitly bypasses cache and enables wrappers/cleanup. The passes share one
database configuration but emit separate `stage2.mapped.json`,
`stage2_mapping_report.json`, `stage2_runtime_schema_report.json`,
`final.mapped.json`, and `mapping_report.json` UI artifacts. Runtime reports
are also exposed after enrichment and before export. Runtime validation remains
report-first by default and allows unknown additive metadata; it does not
replace the semantic PWML gate.

**Pipeline consistency:** Stage 2 owns annotation and mapping uncertainty,
Stage 3 receives that exact output and still owns normalization, Stage 4 keeps
its strict-gate-only repair cadence, Stage 6 remains the sole wrapper-creating
remap (including the PathBank `Unknown` fallback), and Stage 8 retains semantic
export authority. Malformed or failed Stage 2 results stop before Stage 3 and
cannot produce a successful mapped artifact.

---

### 2026-07-13 - Explicit PathBank Unknown fallback for unresolved enzymes

**Files changed:** `src/t2pw/mapping/map_ids.py`,
`src/t2pw/mapping/enrich_entities.py`, `src/t2pw/pipeline/entity_identity.py`,
`src/t2pw/pwml/ir.py`, `src/t2pw/pwml/writer.py`, `src/t2pw/schema.py`,
`tests/test_pathbank_unknown_fallback.py`, and `docs/pipeline.md`.

**Error / symptom:** A source-supported functional enzyme name could remain a
bare, unmapped protein after every protein identity strategy failed. PathWhiz
requires a protein-complex enzyme with a resolvable member, so export was
blocked even though PathBank provides a known `Unknown` protein sentinel.

**Root cause:** Stage 6 had no explicit, provenance-bearing route from a fully
unresolved enzyme to the known PathBank sentinel. Treating the sentinel's
`Unknown` UniProt text as a normal accession would also trigger an invalid
UniProt enrichment request.

**Fix:** After ordinary protein and complex mapping plus the API retry fail,
the wrapper-enabled Stage 6 pass may create or reuse one functional-name
complex backed by PathBank protein `9659` (`Unknown`, *Arabidopsis thaliana*,
species 4, taxon 3702). It records the target organism and
`cross_species_placeholder`, synchronizes catalyst-modifier mirrors, preserves
non-catalytic references, deduplicates reruns, and skips UniProt enrichment.
Stage 2 cannot activate it. PWML emits the reference-compatible
`protein-complex-protein` child and the sentinel's exact scalar identity.

**Pipeline consistency:** Real mappings always win. Mapping and wrapper
creation remain Stage 6 responsibilities; Stage 3 owns catalyst promotion and
contract checking; Stage 8 only serializes the explicit mapping.

---

### 2026-07-10 — Shared entity identity and enforceable Stage 2/3/6 contracts

**Files changed:** `src/t2pw/pipeline/entity_identity.py`,
`src/t2pw/mapping/map_ids.py`, `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/pipeline/stage_contracts.py`, `src/t2pw/schema.py`,
`tests/test_entity_identity_contracts.py`

**Error / symptom:** Mapping, normalization, and PWML code independently
decided whether an entity was protein-like, whether a protein had exportable
identity, and whether a complex was a generated wrapper. Stage 2 could also
create export wrappers even though Stage 6 owns that transformation, while
actor rows could leave Stage 3 without canonical `entity`/`entity_type` fields.

**Root cause:** Identity rules were copied across stage-specific modules, the
mapping API had no wrapper-creation control, and the Stage 3/6 exit guarantees
were documented but not asserted.

**Fix:** Added the dependency-light `entity_identity` module and switched
mapping and normalization to its shared routing, external-ID, species, and
generated-wrapper helpers. Added `allow_complex_wrapper_creation` to mapping,
canonicalized all supported process actor collections, typed `spontaneous` and
generated-wrapper fields in `schema.py`, extended the Stage 3 actor contract,
and added a Stage 6 generated-component identity contract.

**Pipeline consistency:** Entity identity is now neutral shared policy rather
than Stage 3 reaching backward into Stage 2. Stage 2 can map without structural
wrapper creation; Stage 6 alone may create wrappers and must validate them
before export.

---

### 2026-07-10 — Stage 8 fails invalid enzymes and serializes PathWhiz truthfully

**Files changed:** `src/t2pw/pwml/ir.py`, `src/t2pw/pwml/writer.py`,
`src/t2pw/pwml/qa.py`, `src/t2pw/pwml/to_pwml.py`,
`src/t2pw/pwml/legacy_validate.py`, `src/to_pwml.py`, `src/validate.py`,
`tests/test_pwml_ir.py`, `tests/test_pwml_writer.py`

**Error / symptom:** Stage 8 silently wrapped a bare protein enzyme, discarded
reaction spontaneity before serialization, assigned every protein the
pathway's first species, and could hide duplicate enzyme-complex assignments
that PathWhiz rejects. Dead standalone converter/validation paths also offered
an alternate exporter with incompatible behavior.

**Root cause:** Export attempted last-resort structural repair instead of
enforcing Stage 6 output, the IR reaction omitted `spontaneous`, protein
serialization ignored per-record species context, and duplicate targets were
silently collapsed before QA could report them.

**Fix:** Bare protein enzymes now fail the PWML contract without auto-wrapping;
the IR carries `spontaneous`; writer species IDs resolve from each protein with
the pathway species only as a true fallback; QA rejects spontaneous reactions
with enzymes and repeated enzyme-complex targets; and the confirmed-dead
converter/legacy validation modules and shims were removed. The CLI export
also writes and enforces its normalization gate before building IR.

**Pipeline consistency:** Stage 8 validates and serializes the Stage 6 payload
without inventing biology or concealing import errors. Obsolete alternate
export paths are removed so the IR-backed writer remains the authoritative
implementation.

---

### FIXED - Stage 8 PWML IR: direct protein enzyme wrapper lost source protein metadata

**Files changed:** `src/t2pw/pwml/ir.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
The PWML IR builder has a last-resort safety net that wraps direct protein enzyme actors
as generated single-protein protein_complex records. That wrapper check was running after
entity rows had been converted into IR records, but those records did not preserve `species`
context and `_protein_external_id` did not inspect nested `mapped_ids`. A valid protein
catalyst could therefore be reported as missing species/UniProt and remain as a bare protein
enzyme in the IR, triggering `reaction_enzyme_must_be_protein_complex`.

**Fix:**
IR entity records now preserve `species`, `taxonomy_id`, and `species_ref`, and the protein
external-ID helper reads UniProt/DrugBank IDs from `mapped_ids`, `ids`, and `mapping_meta`.
The existing direct-protein catalyst tests now verify that valid proteins are wrapped into
single-component protein_complex enzymes.

---

### FIXED - Stage 3 alias normalization: protein_complex component metadata was flattened

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
`canonicalize_same_as_aliases` rewrote `protein_complexes[].components` to plain strings.
That preserved the component name but discarded structured fields such as `stoichiometry`,
`mapped_ids`, `uniprot`, and `pathbank_protein_id`. Generated PathWhiz wrapper complexes
could then lose the exact data needed by Stage 3/Stage 8 contract checks.

**Fix:**
Component alias rewriting now preserves dict component rows and rewrites only the component
name-bearing field. The same pass also reads component names with `_component_name_from_row`
instead of stringifying dict components.

---

### FIXED - Stage 3 gate: generated protein_complex components missing stoichiometry

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
Generated single-protein PathWhiz wrapper complexes could pass Stage 3 with a component record
that resolved to a declared protein and had species/external identity, but omitted explicit
`stoichiometry`. Stage 8 and the SPMDB schema require structured protein_complex components
to carry positive stoichiometry, so the payload failed later during required PWML contract
validation.

**How the fix is consistent with the pipeline design:**
Stage 3 already hard-gates generated protein_complex component resolution, species, and external
identity. The fix adds the matching positive `stoichiometry` requirement to that same generated
component loop, so repair/audit sees the issue before export. A focused regression test covers
`NdmA complex` with a resolved `NdmA` component missing `stoichiometry`.

---

### FIXED — Stage 3 gate: `canonicalize_same_as_aliases` leaked protein_complex names into `entities.proteins`

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `canonicalize_same_as_aliases`

**Error / symptom:**
After Fix 1 (degree-0 exemption), the gate still reported 21 errors:
- `Generated protein complex wrapper 'NdmA complex' must be listed under protein_complexes, not proteins.`
- `Protein 'NdmA complex' is missing species/organism.`
- `Protein 'NdmA complex' is missing a UniProt or DrugBank identifier.`
Same pattern for `NdmB complex`, `NdmC complex`, `NdmCDE complex`, `NdmD complex`, `TmuM complex`, `caffeine dehydrogenase complex`.

**Why it appeared:**
`canonicalize_same_as_aliases` iterates over every reaction's `enzymes` and `modifiers` and calls
`_ensure_protein(actor_name, payload, rep)` for each actor. Stage 2/6 mapping had already rewritten
these reaction modifier references to point to generated complex wrappers (e.g. `NdmA complex`).
`_ensure_protein` checks whether `actor_name` is in `entities.proteins` but not whether it is in
`entities.protein_complexes`. So it unconditionally added `NdmA complex`, `NdmB complex`, etc. to
`entities.proteins`, even though they were already correctly placed in `entities.protein_complexes`.

**How the fix is consistent with the pipeline design:**
`_ensure_protein` is a safety net to guarantee that every reaction actor has a declared entity.
It should only fire for names that are not yet declared as *any* entity type. Since protein_complex
entries are real entity declarations, an actor that is already in `complexes` needs no fallback
protein row. The one-line guard `if _find_entity_row(complexes, actor_name) is not None: continue`
skips the `_ensure_protein` call for actors already declared as a complex. Stage 3 owns normalization;
this fix stays within `canonicalize_same_as_aliases` and touches no other stage.

---

### FIXED — Stage 3 gate: degree-0 check incorrectly flagged proteins that are complex components

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `run_strict_post_normalization_gates`

**Error / symptom:**
PWML export failed with "PWML export stopped by Stage 3 gate." The gate reported errors such as:
- `Protein has degree 0 after normalization: NdmA`
- `Protein has degree 0 after normalization: NdmB`
- `Protein has degree 0 after normalization: NdmC`
- `Protein has degree 0 after normalization: NdmD`

**Why it appeared:**
Stage 2/6 mapping wraps single-protein reaction enzymes in generated protein_complex records
(e.g. `NdmA complex`) and replaces the direct protein reference in the reaction modifier
with the complex name. This is correct — PathWhiz requires protein_complex as the enzyme
actor. The side effect is that `NdmA` is no longer referenced directly in any reaction;
its network connection flows through `NdmA complex`. `build_graph` does not add edges for
protein→complex component membership, so `NdmA` has degree 0 in the connectivity graph.
`prune_disconnected_proteins` (step 15) correctly keeps `NdmA` because it has a UniProt ID
(`_has_protein_identity` returns True). The `enforce_all_proteins_connected` check in step 17
then flagged it as a connectivity failure even though degree-0 is the expected and correct
state for a complex-component protein.

**How the fix is consistent with the pipeline design:**
`pipeline.md` states: "A protein survives all three passes if it has any of:
complex-component membership with external identity, a process reference, a non-zero
graph degree, or an external database ID." The gate check lacked this exemption.
The fix builds a set of protein name norms that appear as components in any declared
`protein_complexes[]` entry and skips those from the `enforce_all_proteins_connected`
error. No other gate check is changed. Stage 3 owns the gate; the fix lives entirely
within `run_strict_post_normalization_gates`.

---

## Fixed

---

### FIXED — Stage 3 gate: degree-0 check incorrectly flagged proteins that are complex components

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `run_strict_post_normalization_gates`

**Error / symptom:**
PWML export failed with "PWML export stopped by Stage 3 gate." Gate reported errors like `Protein has degree 0 after normalization: NdmA/B/C/D`.

**Why it appeared:**
Stage 2/6 mapping wraps single-protein reaction enzymes in generated `protein_complexes` entries (e.g. `NdmA complex`) and rewrites the reaction modifier reference to use the complex name — correct, because PathWhiz requires a protein_complex as the enzyme actor. The side effect is that `NdmA` no longer appears directly in any reaction; its network connection flows through `NdmA complex`. `build_graph` does not add edges for protein→complex component membership, so `NdmA` has degree 0. `prune_disconnected_proteins` (step 15) correctly kept it because `_has_protein_identity` returned True. The `enforce_all_proteins_connected` check in step 17 then flagged it as a connectivity failure even though degree-0 is the expected state for a complex-component protein.

**How the fix is consistent with the pipeline design:**
`pipeline.md` states: "A protein survives all three passes if it has any of: complex-component membership with external identity, a process reference, a non-zero graph degree, or an external database ID." The gate was missing this exemption. The fix builds `_complex_component_norms` from all `protein_complexes[].components` entries and skips those protein names from the `enforce_all_proteins_connected` error. No other gate check is changed. Stage 3 owns the gate; the fix lives entirely within `run_strict_post_normalization_gates`.

---

## Open Issues

Issues confirmed by running the pipeline. Ordered by pipeline stage. Each entry
records its current status, diagnosis, and planned fix; some older entries are
partially resolved and retain their remaining work here.

---

### IMPLEMENTED — LIVE RERUN VERIFIED: Stage 3/4a/pre-export `NdmCDE` repair

**Files involved:** `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/curation/gap_resolver.py`, `src/t2pw/app/streamlit_app.py`, tests for
normalization, Stage 3 gap issues, orchestration convergence, and organism-aware
location selection.

**Implementation status (2026-07-13):** The stage-owned repair described below
is implemented with deterministic regression coverage. The fresh live run
completed mapping, normalization, audit/gap resolution, curation, and final
Stage 6 remapping while keeping `NdmCDE` out of the protein registry. Its saved
artifact now also passes the repaired programmatic PWML export; only the final
manual Streamlit export click remains.

**Observed progress:** The earlier failed run stopped safely at pre-export
Stage 3 revalidation and exposed two pointer-addressed errors for the synthetic
`/entities/proteins/5` row. The subsequent live run cleared that boundary and
produced a valid final remap. Its first PWML attempt exposed the independent
Stage 8 re-normalization bug documented above; replaying the same artifact after
the repair returns `ok=true` with no required-contract or IR-validation errors.

**Error / symptom:** `NdmCDE` is correctly declared under
`entities.protein_complexes`, but the final normalization pass adds another
bare `NdmCDE` row under `entities.proteins`. The gate rejects that new protein
because it lacks species/organism and UniProt/DrugBank identity. In the Stage 3
resolution report, the real `protein_complex:ndmcde` issue is detected with
missing component stoichiometry and unresolved component references, then its
execution is skipped with `reason="issue_not_found"`. Only one gap-resolution
round is recorded.

**Root cause:**

1. `normalize_composites` rewrites `element_locations.protein_locations` through
   `_rewrite_token`. `_ensure_protein` checks only the protein registry and does
   not first preserve a matching declared protein complex, so the location row
   causes a cross-bucket duplicate.
2. `run_gap_resolution` builds `entity_by_key` for proteins and compounds only,
   although `_collect_stage3_issues` also emits protein-complex issues. The
   planner can therefore request a complex repair that the executor cannot find.
3. The outer audit loop decides convergence from audit patch counts. It can stop
   when no audit patch was accepted even if Gap Resolve changed the payload or
   still has actionable issues. Fresh gate evaluation is also conditional on an
   accepted audit patch instead of any settled-payload change.
4. Location candidate ranking uses broad PathBank frequency without a strong
   organism-compatibility filter. It selected endoplasmic-reticulum membrane for
   two compounds in *Pseudomonas putida*.

**Planned fix:**

1. Stage 3 normalization will perform type-aware registry lookup, preserve
   declared complex references in location/process rows, and gate cross-bucket
   duplicate names. It will not perform ID lookup or invent component ratios.
2. Stage 4a will index protein-complex entities, join component names to declared
   mapped proteins, and write structured component references. Stoichiometry
   must come from source evidence or an accepted audit patch; an absent value
   remains an explicit issue.
3. Stage 4 orchestration will run the fresh strict gate after audit or gap-only
   changes and use payload progress plus remaining issues for convergence. Loop
   safety remains bounded by unchanged/repeated payload detection, timeout, and
   maximum rounds.
4. Stage 4a location resolution will use resolved organism/taxonomy compatibility
   before LLM selection and reject clearly impossible compartments.
5. Stage 6 remains a remapper of the settled payload. Stage 8 remains the hard
   export guard, with UI wording clarified to identify the pre-export Stage 3
   revalidation.
6. Regression tests will cover complex location references, complex issue
   execution, gap-only convergence, organism-compatible locations, and an
   end-to-end `NdmCDE` export boundary.

**Pipeline consistency:** Entity classification is a Stage 3 normalization
invariant; targeted DB/component/location repair belongs to Stage 4a; iteration
and convergence belong to the Stage 4 orchestrator; Stage 6 refreshes mappings;
Stage 8 validates exportability. No proposed stage silently takes over another
stage's semantic responsibility.

---

### PARTIALLY RESOLVED - Stage 2/6/8: Generated PathWhiz protein-complex wrappers leak into proteins and bypass Stage 3 blocking

**Files to change:** `src/t2pw/mapping/map_ids.py`, `src/t2pw/pipeline/process_normalizer.py`, `src/t2pw/app/streamlit_app.py`, `src/t2pw/pwml/ir.py`, tests covering mapping, normalization gates, and PWML export blocking.

**Current status (2026-07-13):** The live Stage 2 pass is now annotation-only,
Stage 6 is the sole wrapper-creating remap, generated wrappers carry explicit
provenance and component-integrity requirements, and unresolved pre-export
Stage 3 failures stop PWML generation. The remaining named-complex duplication
case is not a Stage 2 wrapper leak: it is caused by Stage 3 location-reference
normalization and is tracked in the open `NdmCDE` issue above.

**Error / symptom:**
PWML required-field validation reports errors such as:

- `Protein 'NdmA complex' is missing species/organism.`
- `Protein 'NdmA complex' is missing a UniProt or DrugBank identifier.`
- Same pattern for `NdmB complex`, `NdmC complex`, `xanthine oxidase complex`, `urate oxidase complex`, `allantoinase complex`, `urease complex`, `TmuM complex`, and `TM-HIU hydrolase complex`.

These errors are misleading because the generated `* complex` names should not
be protein rows at all. In PathWhiz, the member protein needs the UniProt or
DrugBank ID; a protein-complex record can be created from valid member proteins
and does not necessarily need a complex-level PathBank ID.

**PathWhiz behavior confirmed from UI:**

1. The `New Protein` form requires `Name`, `Species`, and either `UniProt ID`
   or `DrugBank ID`.
2. The `New Protein Complex` form requires `Name`, `Species`, and at least one
   member `Protein` with stoichiometry.
3. Therefore a generated single-protein wrapper such as `NdmA complex` is valid
   only as a `protein_complexes[]` row with component `NdmA`; `NdmA` must be a
   valid `proteins[]` row with species and UniProt/DrugBank identity.
4. The pipeline should not try to find or assign a UniProt ID for
   `NdmA complex`; UniProt belongs to `NdmA`.

**Root cause:**
There are two interacting issues:

1. `map_ids._rewrite_reaction_protein_enzymes_to_complexes` creates novel
   single-component wrappers named `f"{protein_name} complex"` when PathBank DB
   lookup cannot resolve a real complex. This is acceptable only if the wrapper
   stays under `entities.protein_complexes` and its member protein is already
   mapped.
2. The Streamlit PWML export path calls `normalize_process_payload`, receives a
   Stage 3 gate report, but then proceeds to `validate_required_pwml_contract`
   instead of stopping on unresolved Stage 3 gate failures. As a result, issues
   that Stage 3 can detect still reach the Stage 8 hard gate.

The mapping cache also contains stale/generated `enzyme_complexes` records for
the affected names, including entries with `status: "unmapped"` and
`chosen_rule: "novel_enzyme_single_component_complex"`. These cache rows show
where the `* complex` names are being synthesized.

**Planned fix:**

1. Stage 2/6 mapping (`map_ids.py`):
   - Keep generated single-protein wrappers under `entities.protein_complexes`
     only.
   - Mark generated wrappers with explicit metadata such as
     `generated: true` and
     `generation_reason: "single_protein_pathwhiz_wrapper"`.
   - Before creating a usable wrapper, require the base protein row to have
     species plus UniProt/DrugBank identity.
   - If the base protein is unmapped, do not create an apparently exportable
     complex. Record a mapping issue instead.
   - Do not add or preserve rows like `NdmA complex` under
     `entities.proteins`.
2. Stage 3 normalization/gate (`process_normalizer.py`):
   - Add a hard gate check that rejects `entities.proteins[]` rows whose names
     are generated-complex shaped (`* complex`) when they correspond to
     generated wrappers.
   - Add a generated-complex integrity check: species present, at least one
     component, and every component resolves to a declared protein with
     UniProt/DrugBank identity.
   - Preserve the current design: Stage 3 reports these as gate failures for
     audit/review; it should not silently reclassify biological entities unless
     the operation is a deterministic generated-wrapper cleanup.
3. Orchestrator (`streamlit_app.py`):
   - Before initializing refinement review and before PWML generation, inspect
     `normalize_process_payload(...)[1]["gate"]`.
   - If the gate is not OK, stop and surface the Stage 3 gate errors. Do not
     continue to the PWML required-field gate.
4. Stage 8 PWML IR (`pwml/ir.py`):
   - Treat generated protein complexes without a complex-level PathBank ID as
     valid only when their component proteins satisfy the protein identity
     contract.
   - Keep strict validation for ordinary protein rows: species plus
     UniProt/DrugBank remains required.
5. Tests:
   - Add a fixture where `NdmA` has species and UniProt and `NdmA complex` is a
     generated protein complex. This should pass the generated-complex contract.
   - Add a fixture where `NdmA complex` appears under `entities.proteins`. This
     should fail Stage 3 before export.
   - Add a fixture where `NdmA complex` is generated but component `NdmA` lacks
     UniProt/DrugBank. This should fail before export.

**Pipeline consistency:**
The protein-vs-complex distinction belongs at the mapping and normalization
boundary. Stage 2/6 owns creation of generated PathWhiz wrapper complexes.
Stage 3 owns deterministic gate checks that prevent invalid rows from reaching
review/export. Stage 8 owns final PWML contract enforcement. The orchestrator
must wire these stages so unresolved Stage 3 failures block PWML generation
rather than being rediscovered later as required-field errors.

---

### OPEN â€” Stage 2 (Map): Best-effort UniProt fallback assigns lowest-scored candidate when no threshold passes

**Files to change:** `src/t2pw/mapping/map_ids.py` (replace best_effort_fallback block), `src/t2pw/curation/audit_json_llm.py` (add audit hint for best_effort IDs)

**Error / symptom:**
Generic enzyme names such as "N-methyltransferase complex" and "N-methylnucleosidase complex" have no species-specific UniProt entry that clears the 0.78 confidence threshold. As a temporary workaround (added 2026-07-08), the mapper now accepts the top-ranked UniProt candidate regardless of score and marks it `best_effort: True`. This prevents Stage 3 gate failures for missing external identity, but the assigned accession may be incorrect â€” it is simply the highest-scoring candidate from a name search, not a verified match.

**Root cause:**
Generic descriptive names ("N-methyltransferase", "N-methylnucleosidase") return many UniProt hits with similar, low Jaccard scores. None is definitively the right protein, so no candidate clears the strict acceptance threshold. The right fix is sequence-based disambiguation (BLAST or UniProt sequence search) â€” find the actual protein sequence from the paper or a reference, BLAST it against UniProt, and accept the top hit by sequence identity. This requires the pipeline to carry or fetch protein sequences, which it currently does not do.

**Planned fix:**
1. For proteins that reach the best_effort_fallback path, attempt a NCBI eSearch + efetch to retrieve the candidate sequence by gene name + organism.
2. Submit the retrieved sequence to the UniProt BLAST API.
3. Accept the BLAST top hit (â‰¥40% identity, â‰¥60% coverage) as the confirmed accession and replace the best_effort ID.
4. Add an audit hint in `audit_json_llm.py` that flags any entity with `best_effort: True` in its mapping metadata so the audit LLM knows to verify or propose a correction.

**Pipeline consistency:**
Sequence fetching and BLAST belong in Stage 2 mapping or Stage 4a gap resolution â€” both own external ID lookup. No normalization or export logic would change. The `best_effort` flag in mapping metadata is the audit signal; the audit loop owns the decision to accept or replace the provisional ID.

---

### OPEN â€” Stage 4 (Audit): LLM connection failure prevents semantic repair

**Files to change:** Configuration / environment (not source code)

**Error / symptom:**
`curator_report.json` shows `"error": "chat_with_tools call failed after
retries. Last error: Connection error."` The round-1 audit report shows
`"enabled": false`. The 3 reactions with missing inputs (see next issue) survive
to the final payload unremediated because no LLM repair ran.

**Root cause:**
The OpenRouter API is not reachable from this environment. Either the API key
is missing/invalid, the endpoint is blocked, or there is a network configuration
issue. The audit stage correctly identifies there are errors to fix (3 reactions
with empty inputs) but cannot reach the LLM to generate repair patches.

**Planned fix:**
1. Verify `OPENROUTER_API_KEY` (or equivalent) is set in the environment.
2. Confirm the OpenRouter endpoint is reachable (`curl https://openrouter.ai/api/v1/models`).
3. If using a local model or alternative provider, update the provider config in
   `src/t2pw/curation/audit_json_llm.py`.
4. Add a clear error message in the Streamlit UI when LLM is disabled so the
   user knows semantic repair was skipped, not that the pipeline succeeded.

**Pipeline consistency:**
This is a configuration / infrastructure issue. No stage logic needs to change.
The pipeline correctly passes gate failures to audit; the repair just cannot run
without an LLM connection.

---

### OPEN â€” Stage 1 (Extract): Empty reaction inputs in beta-oxidation chain

**Files to change:** None yet â€” this is a Stage 4 (Audit) repair task once
LLM is connected (see connection issue above).

**Error / symptom:**
Reactions `beta_oxidation_OPC8`, `beta_oxidation_OPC6`, `beta_oxidation_OPC4`
(indices 10, 11, 13 in the final payload) each have `inputs: []` and
`outputs: ["jasmonic acid"]`. The audit deterministic check reports:
`"Reaction must include at least one input and one output."` for all three.

In the real pathway, each beta-oxidation cycle takes an OPC-CoA ester (OPC-8:0-CoA,
OPC-6:0-CoA, OPC-4:0-CoA) as input and produces the next shorter chain as
output. The LLM collapsed the entire beta-oxidation chain and wrote only the
final product (jasmonic acid) as each reaction's output.

**Root cause:**
Stage 1 (Extract) LLM did not capture the intermediate acyl-CoA compounds as
reaction-level inputs/outputs. Each reaction was extracted as "produces JA" with
the intermediate steps omitted. This is an expected limitation of the extraction
stage â€” the LLM summarised rather than enumerated each cycle.

**Planned fix:**
This is the correct input for Stage 4 (Audit). Once the LLM connection is
restored (see above), the audit should:
1. Receive the gate failure list which includes these 3 reactions.
2. Propose patches that add the correct OPC-CoA intermediate as `inputs` for
   each reaction and the shortened OPC-CoA as `outputs` (except the last cycle
   which produces jasmonic acid).
No normalization or schema change is required â€” this is a data completeness
issue that the audit loop is designed to repair.

**Pipeline consistency:**
The gate correctly fires for these reactions. The pipeline design says gate
failures feed audit, not abort. No stage logic change needed. This issue will
resolve once the LLM connection issue is fixed.

---

### OPEN â€” Stage 1 (Extract): OPC-8, OPC-6, OPC-4 misclassified as proteins

**Files to change:** None yet â€” audit repair task.

**Error / symptom:**
Entities `OPC-8`, `OPC-6`, and `OPC-4` appear in `entities.proteins`. The audit
warns: `"Protein has no location link; default compartment may be used."` for
all three. These are 3-oxo-2-(2'-pentenyl)-cyclopentane-acyl-CoA intermediates â€”
chemical compounds, not proteins.

**Root cause:**
Stage 1 (Extract) LLM placed these in `proteins` because they appear alongside
enzyme names in the paper text, and their abbreviated names (OPC-N) look like
protein/gene identifiers. The normalization stage does not reclassify entity
types.

**Planned fix:**
Once the LLM connection is restored, the audit should propose moving these
entities from `entities.proteins` to `entities.compounds`. An alternative
heuristic: Stage 3 normalization could flag entities whose names match known
patterns (CoA suffix, lipid chain nomenclature) as candidate compound
misclassifications and include them in the gate report for audit review. This
would not require a hard reclassification in normalization (which would be
cross-stage logic), just an additional audit hint.

**Pipeline consistency:**
Reclassification is a semantic operation and belongs in Stage 4 (Audit). If a
normalization hint is added, it goes into the gate report as an audit input â€”
not as a normalization mutation. This preserves the rule that Stage 3 does not
make semantic corrections.

---

### OPEN â€” Stage 2 (Map): DB unavailable â€” degraded mapping rates

**Files to change:** Configuration only.

**Error / symptom:**
`mapping_report.dbonly.json` shows `"db_available": false`. Protein mapping:
38.46% (5/13). Compound mapping: 9.52% (2/21). All 12 protein complexes skipped
(10/12 have gap issues â€” component proteins unmapped, so complex cannot map).

**Root cause:**
The PathBank database is not reachable from this environment
(`db_available: false`). Stage 2 falls back to API-only mapping, which has
lower coverage than the local DB. Compound mapping is especially degraded
because most common metabolites depend on the local compound DB.

**Planned fix:**
1. Configure the local PathBank DB (host, schema, credentials) in the Streamlit
   sidebar or environment variables.
2. If DB is intentionally absent, document that compound ID coverage will be
   low and PWML export will operate in non-strict mode.
3. The protein complex gap issues (10/12) are downstream of the protein mapping
   problem: once component proteins map, the complexes should map too. Fix the
   DB connection first and re-run before filing separate complex issues.

**Pipeline consistency:**
This is a configuration / infrastructure issue. Stage 2 mapping logic is correct.
No source changes needed. The pipeline.md note on Stage 2 already says mapping
returns no-hit when an entity is unmapped, which is recorded in `mapping_meta`.

---

### OPEN â€” Stage 6 (Enrich): Enrichment stage produces data no stage consumes

**Files to change:** Decision required before code change.

**Error / symptom:**
`run_enrichment` fetches synonyms, cross-references, and properties and writes
`entity["enrichment"]` onto each entity. No downstream stage (normalization,
audit, PWML IR, SBML) reads this field. The enrichment API call and cache write
happen on every pipeline run with no effect on output.

**Root cause:**
The enrichment stage was built but not wired into the PWML IR builder or any
other consumer. This was documented as a product decision pending in the
refactoring plan (Step 8).

**Planned fix â€” choose one:**
- **Option A (Use it):** Wire `entity["enrichment"]` into the PWML IR builder
  (`src/t2pw/pwml/ir.py`) so synonyms and cross-references appear in the
  exported pathway file. This adds value to the PWML output.
- **Option B (Remove it):** Delete `run_enrichment` and its call site in
  `streamlit_app.py`. Mapping already attaches database IDs; enrichment adds
  nothing until Option A is implemented. Removal simplifies the orchestrator and
  eliminates dead code.
Until the decision is made, the enrichment stage runs silently on every pipeline
execution consuming API quota and cache space.

**Pipeline consistency:**
Option A change lives in `t2pw/pwml/ir.py`. Option B removes
`t2pw/mapping/enrich_entities.py` call site from the orchestrator. Neither
option adds cross-stage logic.

---

## Template

```
### YYYY-MM-DD â€” <short description>

**Files changed:** `path/to/file.py` (lines Xâ€“Y)

**Error / symptom:**
What the user or test saw. Quote the error message if there is one.

**Root cause:**
Why the error appeared. Name the specific stage boundary violation, field
mismatch, or misplaced logic that caused it.

**Fix:**
What was changed and where.

**Pipeline consistency:**
Which stage owns this change. Confirm it does not add cross-stage logic and
does not expand any module's scope beyond its intended area (see File ownership
table in pipeline.md).
```

---

## Entries

### 2026-07-08 â€” Best-effort UniProt fallback for generic enzyme names that clear no confidence threshold

**Files changed:** `src/t2pw/mapping/map_ids.py` (lines ~3580â€“3590, `map_protein_uniprot`), `docs/change_log.md`

**Error / symptom:**
Proteins with generic names such as "N-methyltransferase complex" and "N-methylnucleosidase complex" failed the Stage 3 gate checks for missing UniProt/DrugBank identifiers. These names have no species-specific UniProt entry that scores above the 0.78 acceptance threshold â€” the correct protein cannot be confidently distinguished from many similarly-named candidates by name alone.

**Root cause:**
`map_protein_uniprot` returned `status: "unmapped", reason: "ambiguous"` whenever candidates existed but none cleared the strict threshold. The caller only writes a UniProt accession when `status == "mapped"`, so ambiguous results produced no ID on the protein entity. For genuinely generic enzyme names, no name-based scoring strategy can reliably pick the right candidate â€” the correct fix is sequence-based lookup (BLAST), but the pipeline does not currently carry protein sequences.

**Fix:**
In `map_protein_uniprot`, when `_accepted_uniprot_candidate_result` returns None but at least one candidate has a non-empty accession, return that top candidate with `status: "mapped"`, `chosen_rule: "best_effort_fallback"`, and `best_effort: True` instead of `status: "unmapped"`. The caller writes the accession, which clears the Stage 3 gate check. The `best_effort: True` flag is preserved in the mapping metadata as a signal for the audit loop that this ID was not confidently matched and should be reviewed.

**Pipeline consistency:**
Change is entirely within `map_protein_uniprot` in `t2pw.mapping.map_ids`, which owns Stage 2 and Stage 6 ID mapping. No normalization, audit, or export logic was changed. The fallback is a last resort â€” it only activates when all three normal acceptance paths (strong_unique, reviewed_unique, reviewed_exact_gene_match) fail and candidates exist. The proper long-term fix (BLAST-based sequence lookup) is documented as an open issue. See OPEN issue: "Stage 2 (Map): Best-effort UniProt fallback assigns lowest-scored candidate when no threshold passes."

---

### 2026-07-08 â€” Strengthen extraction scoping to single-pathway + single-organism; fix doc inconsistencies

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
Three issues found after adding the initial species scoping rule: (1) The BIOLOGICAL STATE RULE still instructed the LLM to default to *Homo sapiens* when no organism was available, directly contradicting the new scoping rule's instruction to leave species empty. (2) The scoping rule told the LLM to pick one organism but did not tell it to first pick one pathway â€” papers covering multiple pathways (e.g. caffeine biosynthesis and caffeine degradation in the same review) would still produce a merged multi-pathway extraction scoped to one organism. (3) `docs/pipeline.md` had a duplicate copy of the Step 17 gate description mislabelled as Step 15, and the file ownership table listed no prompt files.

**Root cause:**
(1) The BIOLOGICAL STATE RULE predated the scoping rule and was never updated to match. An LLM reading both rules encounters conflicting instructions for the no-organism case. (2) The prior scoping rule said "choose one primary biological scope" but did not make pathway selection an explicit first decision â€” organism selection was the only named decision. (3) The pipeline.md duplicate was a copy-paste artifact from a prior edit; the prompt files were always owned but never listed in the table.

**Fix:**
1. Changed the BIOLOGICAL STATE RULE fallback from `"use 'Homo sapiens' as the default"` to `"leave species empty â€” do not guess or default to any organism"`, removing the contradiction.
2. Expanded the scoping rule into two explicit sequential decisions: Decision 1 (select one pathway â€” the most central to the paper) followed by Decision 2 (select one organism for that pathway). The pathway decision now comes first and is the primary filter; organism selection applies within it.
3. Removed the duplicate Step 15 paragraph from `docs/pipeline.md` (lines 135â€“138, copy-paste of Step 17 description).
4. Added `pwml_system.txt` and `pwml_infer_system.txt` to the file ownership table in `docs/pipeline.md`.

**Pipeline consistency:**
All changes are in prompt text files and documentation. No Python source was modified. The scoping decision remains Stage 1's responsibility â€” it is an extraction-time filter that prevents mixed-pathway, mixed-species entity sets from entering Stage 2 and beyond.

---

### 2026-07-08 â€” Add single-organism scoping rules to Stage 1 extraction prompt

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `src/t2pw/llm/prompts/pwml_infer_system.txt` (cross-reference note only), `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
Proteins from multiple organisms present in a single paper (e.g. *Coffea arabica* biosynthesis enzymes and *Pseudomonas putida* degradation enzymes) were extracted together into the same pathway payload, resulting in mixed species assignments across entities. This caused Stage 3 gate failures for missing species/organism on proteins that inherited no clear organism context, and UniProt mapping failures at Stage 2 and Stage 6 because the wrong species was searched for each protein.

**Root cause:**
The Stage 1 extraction prompt (`pwml_system.txt`) had no rule requiring the LLM to select a single primary organism before extracting reactions. Papers that cover multiple organisms â€” comparative studies, combined biosynthesis-plus-degradation reviews â€” caused the LLM to emit proteins from all mentioned organisms, mixing species context across entities. The BIOLOGICAL STATE RULE required species on every biological_state but gave no guidance for choosing among competing organisms.

**Fix:**
Added two rule blocks to `pwml_system.txt` immediately after the BIOLOGICAL STATE RULE:

1. **Species and organism scoping rule** â€” instructs the LLM to select one primary organism/species/strain before extracting reactions, assign it to all proteins, enzymes, complexes, reactions, and biological states, exclude entities from other organisms unless explicitly requested, and emit an audit warning rather than mix species when no organism can be confidently selected.
2. **Protein/enzyme species rule** â€” requires every protein, enzyme, and protein complex to inherit the selected pathway species before identifier mapping, and prohibits emitting a protein entity without a species/organism assignment and sufficient identifier context.

Added a species constraint cross-reference note to the locality constraint block in `pwml_infer_system.txt`: the Stage 2 mandatory modifier repair pass is now explicitly instructed to apply the Stage 1 species scoping rule and skip modifier links for proteins from other organisms.

**Pipeline consistency:**
Change is entirely within prompt text files. No Python source was modified. Species scoping is an extraction-time decision that Stage 1 owns â€” the correct stage boundary. Selecting a single organism at Stage 1 prevents mixed-species entity sets from propagating to Stage 2 mapping (where wrong-species queries fail silently) and Stage 3 gate checks (where missing species generates unrepaired gate failures). Stage 2â€“8 behavior is otherwise unchanged.

---

### 2026-07-08 â€” Strip "complex" from UniProt name query variants

**Files changed:** `src/t2pw/mapping/map_ids.py` (line ~50, `_name_variants`), `docs/change_log.md`

**Error / symptom:**
Proteins with "complex" in their names â€” e.g. "xanthine oxidase complex", "NdmA complex", "IMP dehydrogenase complex", "TmuM complex" â€” consistently failed the Stage 3 gate checks added on 2026-07-08 for missing UniProt/DrugBank identifiers. These proteins are findable in UniProt under their base names ("Xanthine oxidase", "NdmA", etc.) but the pipeline assigned no accession to any of them.

**Root cause:**
`_name_variants` (Stage 2 and 6 mapping) already strips "protein" and "enzyme" from name query strings to normalize them for UniProt lookup, but did not strip "complex". UniProt never includes "complex" in individual protein entry names â€” that word is a complex-level descriptor. Querying for "xanthine oxidase complex" produced a Jaccard similarity of 2/3 â‰ˆ 0.667 against the correct UniProt entry "Xanthine oxidase". After scoring (`base_score = 0.35 Ã— 0.667 = 0.234`, plus organism and reviewed bonuses), the total landed at â‰ˆ 0.53 â€” 0.25 points below the 0.78 acceptance threshold â€” so no accession was accepted despite the correct entry being returned by UniProt's API.

**Fix:**
Added `"complex"` to the word-strip regex in `_name_variants`:
```
re.sub(r"\b(protein|enzyme|complex)\b", " ", base, flags=re.IGNORECASE)
```
Names like "xanthine oxidase complex" now generate "xanthine oxidase" as a search variant. That variant scores 1.0 (exact name match, base_score = 0.55) plus organism and reviewed bonuses, clearing the 0.78 threshold and producing a mapped accession.

**Pipeline consistency:**
Change is entirely within `t2pw.mapping.map_ids`, which owns Stage 2 and Stage 6 ID mapping. No normalization, audit, export, or orchestrator logic was touched. The change is a query normalization improvement consistent with the pre-existing "protein" and "enzyme" stripping.

---

### 2026-07-08 â€” Add protein species and external identity checks to Stage 3 gate

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (inside `run_strict_post_normalization_gates`), `docs/change_log.md`

**Error / symptom:**
Stage 8 (Export) hard-aborted with `validate_required_pwml_contract` failures for two checks â€” `protein_missing_species` and `protein_missing_external_identity` â€” with no opportunity for the audit loop to repair the affected proteins. The specific failure: proteins (and compounds misclassified into `entities.proteins`) that had no species/organism field and no UniProt or DrugBank ID would pass Stage 3 and Stage 4 unchanged, then cause an unrecoverable abort at pre-export contract validation.

**Root cause:**
Both checks existed only in `t2pw/pwml/ir.py` (lines 1880â€“1915) as part of the hard Stage 8 pre-export semantic contract. They were absent from `run_strict_post_normalization_gates` in `process_normalizer.py`, so Stage 4 (Audit) never received them as gate failures to repair. The audit loop correctly repairs what the gate reports; the gate simply never reported these two conditions.

**Fix:**
Added two new loops inside `run_strict_post_normalization_gates`, immediately after the existing forbidden-name check loop on `entities.proteins`. Each loop iterates `entities.proteins`, skips unnamed rows (already caught by a separate check), and calls `_add_error` when the condition is unmet:

1. **Species/organism check** â€” mirrors the `species` resolution chain from `ir.py`: tries `species`, `organism`, `taxonomy_id`, `species_id`, `pathbank_species_id`, `species_ref.pathbank_species_id`, `species_ref.name`, `mapping_meta.species`, `mapping_meta.species_id`.
2. **External identity check** â€” emits an error if none of `uniprot`, `uniprot_id`, `drugbank`, `drugbank_id` are present and non-empty.

Both checks use only `_safe_dict` and `_safe_list`, which are already defined in `process_normalizer.py`. No imports from `t2pw.pwml.ir` or any other stage module were added.

**Pipeline consistency:**
The fix lives entirely within `run_strict_post_normalization_gates` in `process_normalizer.py`, which owns Stage 3's gate. The gate's return type and `errors` list shape (`{"path": str, "reason": str}`) are unchanged. The `GateValidationError` raise path is untouched. By surfacing these two conditions as Stage 3 gate failures, Stage 4 now receives them in its repair context and can propose patches (species assignment, ID lookup via gap resolution) before Stage 8 runs. No cross-stage logic was introduced â€” `process_normalizer.py` mirrors the field-level logic without importing from `ir.py`.

---

### 2026-07-07 â€” Fix `normalize_process_actor_schema` to write `entity`/`entity_type` for enzyme actors

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (blocks 1c and legacy-enzyme view), `tests/test_process_normalizer.py` (updated assertions)

**Error / symptom:**
All enzyme actor dicts in `reactions[].enzymes` retained `protein_complex` (or
`protein`) as the name field after normalization completed. `e.get("entity")`
returned `""` for every enzyme. After a full pipeline run on the Arabidopsis
jasmonic acid pathway, all 30 enzyme actors used `protein_complex`, while all 30
modifier actors correctly used `entity/entity_type`.

**Root cause:**
`normalize_process_actor_schema` has two passes. Pass 1 (`_rewrite_actor_rows`)
resolves each actor name against the protein and complex registries and writes
the canonical name back to `protein_complex` or `protein` â€” NOT to `entity`.
The post-process loop migrated `modifiers[]` rows to `entity/entity_type` schema,
but had no equivalent migration for `enzymes[]`. Additionally, the "legacy view"
reconstruction block that rebuilds `reaction["enzymes"]` from `modifiers[]`
wrote `protein`/`protein_complex` keys rather than `entity`/`entity_type`,
leaving enzymes in legacy field format after the schema-normalization step.

**Fix:**
1. Added block **1c** in the post-process loop (after the modifier migration and
   the 1b entity_type correction): iterates `reaction["enzymes"]`, migrates each
   dict from `protein_complex`/`protein`/`name` to `entity`/`entity_type`,
   drops actors whose `entity_type` is in `dropped_enzyme_entity_types`, and
   writes the result back to `reaction["enzymes"]`.
2. Updated the legacy-enzyme view reconstruction (formerly writing
   `protein_complex`/`protein` keys) to use `entity`/`entity_type` instead,
   keeping `reaction["enzymes"]` in sync with `modifiers[]` in the canonical
   schema.
3. Updated six test assertions in `tests/test_process_normalizer.py` that were
   checking for the old `protein`/`protein_complex` keys; all now verify
   `entity` and `entity_type` and confirm legacy keys are absent.

**Pipeline consistency:**
Change is entirely within `normalize_process_actor_schema` in
`process_normalizer.py`, which owns actor schema enforcement for Stage 3. No
orchestrator, mapping, audit, or export logic was touched. After the fix, any
code that calls `actor.get("entity")` works correctly for both enzyme and
modifier actors without special-casing field names.

---

### 2026-07-07 â€” Wire drop_process_orphan_proteins into normalize_process_payload

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (line ~3586), `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
`drop_process_orphan_proteins` was defined and documented but never called inside
`normalize_process_payload`. Standalone subunit proteins (e.g. NdmC, NdmD) that
appear only as `protein_complex.components` entries and are never referenced in any
reaction, transport, or interaction would pass through all normalization steps and
reach the gate as orphans, generating audit issues that should have been pre-empted
by pruning.

**Root cause:**
A prior implementation pass added the function to the module but omitted the call
site from the pipeline sequence in `normalize_process_payload`. The change log
stated it was wired in, but the code did not reflect this. The gap was discovered
by reviewing the actual step sequence (lines 3584â€“3588) against the documented
17-step list.

**Fix:**
Added the call `drop_process_orphan_proteins(data, report=report)` and its
corresponding `_checkpoint("drop_process_orphan_proteins")` between
`drop_unresolved_complex_component_proteins` and `prune_disconnected_proteins`.
Updated `docs/pipeline.md` to reflect the 17-step sequence and document why steps
13â€“15 run in sequence (each catches a different class of orphan; a protein must
fail all three to be treated as an orphan by the gate).

**Pipeline consistency:**
Change is entirely within `normalize_process_payload` in `process_normalizer.py`,
which owns all normalization steps. No orchestrator, mapping, audit, or export
logic was touched. The three pruning steps remain independent functions â€” each with
a single responsibility â€” rather than being merged into one function that would be
harder to reason about when a specific class of orphan slips through.

---

### 2026-07-07 - Deterministic PWML compound IDs with optional DB resolver

**Files changed:** `src/t2pw/pwml/ir.py`, `docs/change_log.md`

**Error / symptom:**
PWML IR tests could pass or fail depending on which modules pytest collected in
the same process. A payload with explicit `pathbank_compound_id` values produced
`compound_db_resolution_failed` errors when a DB resolver was importable.

**Root cause:**
`_resolve_compound_rows` only accepted direct PathWhiz compound IDs as a fallback
when no DB resolver was available. If resolver construction succeeded, the same
rows were sent through live DB matching and could fail despite already carrying
the required export ID.

**Fix:**
Accepted explicit `pathbank_compound_id` / `pw_compound_id` / `pathwhiz_id`
values before attempting resolver lookup, while still recording the
`legacy_id_unverified` DB-resolution status in the IR report.

**Pipeline consistency:**
This stays inside `t2pw.pwml.ir`, which owns pre-export IR construction. It does
not move mapping logic into export; it only makes already-mapped payload IDs
deterministic regardless of optional DB resolver availability.

---

### 2026-07-07 - Streamlit uses canonical normalization and post-audit cache bypass

**Files changed:** `src/t2pw/app/streamlit_app.py`, `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
The Streamlit post-pipeline path still owned a hand-built normalization sequence
and returned immediately on post-normalization gate failures, preventing the
audit loop from repairing the semantic issues documented by the gate.

**Root cause:**
Normalization logic lived partly in the orchestrator, including evidence-based
enzyme attachment and explicit gate handling. Post-audit mapping also reused the
normal mapping cache even though audit patches can rename entities.

**Fix:**
Replaced the manual normalization block with `normalize_process_payload` and an
`on_checkpoint` callback that writes the existing probe files. Gate failures are
now written to `gate_fail_report.json`, passed into the audit context, and
reported as audit input rather than a stopped pipeline. The post-audit mapping
pass now calls `map_payload` in memory with `use_cache=False` and writes the
same mapped payload/report artifacts.

**Pipeline consistency:**
The orchestrator now wires stage functions and artifacts only. Normalization
behavior remains in `process_normalizer.py`; mapping/cache behavior remains in
`map_ids.py`; the UI no longer owns enzyme-attachment logic or a parallel
normalization pipeline.

---

### 2026-07-07 - Normalizer actor lookup, evidence enzymes, pruning, and gate reporting

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `src/t2pw/pipeline/qa_graph.py`, `tests/test_process_normalizer.py`, `docs/change_log.md`

**Error / symptom:**
Normalizer actor rows could still resolve a stale legacy `protein` or
`protein_complex` field before the canonical `entity` field. Enzyme mentions in
reaction evidence were wired in Streamlit as plain strings, disconnected
proteins were pruned without respecting mapped identity, and
`normalize_process_payload` could abort on gate failure before the audit loop
received the gate details.

**Root cause:**
Normalizer-owned actor interpretation and evidence wiring were split across the
orchestrator and normalization stage. Some compatibility reads still used
legacy field order. Protein pruning and gate handling also mixed pre-audit
cleanup with semantic rejection, which conflicts with the audit-loop contract in
`docs/pipeline.md`.

**Fix:**
Moved `_norm_text` and `attach_enzymes_from_reaction_evidence` into
`process_normalizer.py`, with cue-near-name matching and canonical actor dict
output. Updated normalizer and QA graph actor lookup to read `entity` before
legacy fields. Changed `prune_disconnected_proteins` to remove only degree-0
proteins with no external identity and record report details. Wired
`normalize_process_payload` to run the enzyme-evidence step, support checkpoint
callbacks, and return gate details in `report["gate"]` instead of aborting.

**Pipeline consistency:**
All deterministic cleanup remains in `process_normalizer.py`, and graph
connectivity interpretation remains in `qa_graph.py`. Streamlit does not gain
new normalization logic. Gate failures remain semantic audit input after
normalization, preserving the documented normalize-to-audit loop rather than
turning the normalizer into a pre-audit hard abort.

### 2026-07-07 - Add stage boundary contract validators

**Files changed:** `src/t2pw/pipeline/stage_contracts.py` (lines 1-273), `tests/test_stage_contracts.py` (lines 1-104), `docs/change_log.md`

**Error / symptom:**
Step 6 of the pipeline refactor needed a dedicated `stage_contracts` module so
stage boundary checks are explicit and testable. Without it, callers had no
single place to distinguish structural aborts from semantic gate failures that
must be sent to audit.

**Root cause:**
Boundary contract ownership was documented in `docs/pipeline.md`, but no module
implemented those boundaries. That made it easy to collapse pre-audit semantic
gate failures into hard abort behavior, which would bypass the audit loop that
is supposed to repair them.

**Fix:**
Added `StageContractError` plus validators for post-extraction, post-mapping,
post-normalization, post-audit, and pre-export boundaries. Structural
validators raise `StageContractError`; post-normalization returns semantic gate
failures as a report for audit; pre-export wraps
`validate_required_pwml_contract` failures in `StageContractError`. Added
focused unit tests for missing required boundary fields and PWML contract
wrapping.

**Pipeline consistency:**
The change lives entirely in `t2pw.pipeline.stage_contracts`, the module named
as the owner of stage boundary validation. It does not add normalization,
mapping, audit, UI, or PWML IR logic, and it keeps pre-audit semantic failures
as audit input instead of making them aborts.

---

### 2026-07-07 - Document pipeline payload schema types

**Files changed:** `src/t2pw/schema.py`, `docs/change_log.md`

**Error / symptom:**
Step 1 of the refactor needed `t2pw/schema.py` to document the JSON payload
contracts, but the module was empty.

**Root cause:**
The pipeline stages pass dictionaries whose expected shapes are documented in
the extraction prompts and `docs/pipeline.md`, while the schema ownership module
did not yet expose those contracts for type checkers or importers.

**Fix:**
Added `TypedDict` definitions for the payload, entity buckets, biological
states, locations, process rows, visualizations, mapping metadata, and inference
additions. `PayloadReactionActor` documents `entity` as the canonical actor name
while retaining backwards-compatible `protein` and `protein_complex` fields.

**Pipeline consistency:**
This change is type/documentation only and lives entirely in the schema module
that owns payload shapes. It does not add validation, normalization, UI logic,
or cross-stage behavior, so runtime pipeline behavior is unchanged.

---

### 2026-07-07 â€” Gate validation errors not shown in UI

**Files changed:** `src/t2pw/app/streamlit_app.py` (lines 2637â€“2651)

**Error / symptom:**
PWML export failed with "Hard-gate validation failed after normalization" but
the UI showed no detail about which specific checks failed, making the error
unactionable.

**Root cause:**
The `st.error()` call at the gate failure block only displayed the top-level
error string from `gate_fail_report`. The `errors` list inside the report
(which contains per-check path and reason) was never rendered.

**Fix:**
Expanded the gate failure display block to iterate `gate_fail_report["errors"]`
and show each entry as a formatted line (path + reason) inside an expander.

**Pipeline consistency:**
Change is entirely within the orchestrator's display logic. No stage function
was modified. The gate report structure is owned by `process_normalizer.py` and
was not changed â€” only the UI reading of it was corrected. This is a pure
orchestrator responsibility: surface what a stage reported.

---

### 2026-07-07 â€” Orphan proteins not pruned when not complex components

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (after line 1639)

**Error / symptom:**
Proteins appeared in `entities.proteins` with no reference in any reaction,
transport, or interaction, and no external database identity. These caused
the gate's `enforce_all_proteins_connected` check to fail.

**Root cause:**
`drop_unresolved_complex_component_proteins` (the existing pruning step) only
dropped proteins that appeared as components of a declared `protein_complex`
entity. Proteins that the LLM extracted standalone, with no complex membership
and no process reference, were never caught.

**Fix:**
Added `drop_process_orphan_proteins` to `process_normalizer.py`. It collects
all entity names referenced across reactions, transports, and interactions,
then drops any protein not in that set that also has no external identity
(`_has_protein_identity` returns False). Wired into `normalize_process_payload`
between the existing complex-component pruning step and `dedupe_processes`.

**Pipeline consistency:**
Change lives entirely within `process_normalizer.py`, which owns all
normalization steps. The new function follows the existing pattern: takes
payload + optional report dict, mutates payload in-place on the deep copy,
records dropped items in the report. No orchestrator or UI code was changed.
No new stage or module was created.

---

### 2026-07-07 - In-memory wrappers for mapping and audit stages

**Files changed:** `src/t2pw/mapping/map_ids.py`, `src/t2pw/curation/audit_json_llm.py`, `src/t2pw/curation/apply_audit_patch.py`, `tests/test_map_ids.py`, `tests/test_audit_json_llm_payload.py`, `tests/test_apply_audit_patch_lock_policy.py`

**Error / symptom:**
Later orchestration work needed to pass Step 7 payload objects between mapping, audit, and patch application without forcing every stage through temporary JSON files. The existing file wrappers also made post-audit remapping vulnerable to stale cache reads unless callers could bypass or invalidate cache entries.

**Root cause:**
The core mapping implementation lived inside the file-based `run_mapping` adapter, so object-level orchestration could not reuse it directly. Audit and patch application had similar file-wrapper boundaries even though their core logic was already mostly payload-based. Mapping cache control was implicit in the cache file rather than exposed at the stage boundary.

**Fix:**
Added `map_payload` as the object-in/object-out mapping entry point and changed `run_mapping` to call it before writing the same mapped JSON/report files. Added `use_cache` and `invalidate_cache_keys` support to the mapping cache path for post-audit remapping correctness. Added `audit_payload` and `apply_audit_patch_payload` wrappers that reuse existing audit and patch core logic without duplicating manifest discovery. Added focused tests for the new wrapper contracts.

**Pipeline consistency:**
Mapping cache and ID assignment changes stay in `t2pw.mapping.map_ids`, which owns Stage 2 and post-audit remapping. Audit planning stays in `t2pw.curation.audit_json_llm`, and patch policy stays in `t2pw.curation.apply_audit_patch`. Existing `run_*` functions remain file adapters, and no Streamlit or normalization logic was moved into these stages.

### 2026-07-10 — Live stage contracts and fresh audit gate cadence

**Files changed:** `src/t2pw/app/streamlit_app.py`

**Error / symptom:** Boundary validators existed only in tests, and audit rounds
continued from a stale pre-audit gate report after accepting a repair patch.

**Root cause:** The Streamlit orchestrator used ad hoc report inspection instead
of the stage-contract API and never reran the cheap strict gate inside the audit
loop. Its only live `map_payload` call is the post-curation remap, so there was
also no honest Stage 2 mapping boundary to validate.

**Fix:** Wired the live extraction, normalization, audit, remap, and pre-export
boundaries to their contract functions; marked the existing remap call as the
wrapper-creating Stage 6 pass; and reran only
`run_strict_post_normalization_gates` after every selected patch with accepted
operations. Fresh pointer-level failures are saved in the iteration and passed
to the next audit prompt. No synthetic Stage 2 call was introduced.

**Pipeline consistency:** Boundary coordination stays in the Streamlit
orchestrator. Normalization still runs once before audit and once at export;
the loop invokes only the Stage 3-owned strict gate.

---

### 2026-07-10 — Audit reuses Stage 3 validators and resolves enzyme-less reactions

**Files changed:** `src/t2pw/curation/audit_json_llm.py`,
`tests/test_audit_json_llm_payload.py`

**Error / symptom:** Audit maintained separate composite/registry failure
definitions and could not explicitly resolve an enzyme-less reaction as
spontaneous.

**Root cause:** Stage 4 had grown its own deterministic checks rather than
consuming Stage 3's validators, and its patch policy had no allowed
`spontaneous` operation.

**Fix:** Audit now calls `validate_no_composites` and
`validate_registry_references` for the shared failure definitions while keeping
patch construction in Stage 4. When both enzyme rows and catalyst modifiers
lack a real actor reference, deterministic audit emits a documented
`spontaneous=true` patch.

**Pipeline consistency:** Stage 3 remains the owner of normalized composite and
registry validity. Stage 4 owns repair planning and is one of the two stages
permitted to write `spontaneous`.

---

### 2026-07-10 — Extraction and audit spontaneity instructions

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`,
`src/t2pw/curation/audit_json_llm.py`

**Error / symptom:** The model had no explicit distinction between
source-supported spontaneous extraction and audit-time resolution of a missing
enzyme.

**Root cause:** Neither prompt stated which stages may set `spontaneous` or the
enzyme-present contradiction.

**Fix:** The extraction prompt permits `spontaneous=true` only from explicit
source text. The audit prompt permits it only after checking both enzymes and
catalyst modifiers and finding no real catalyst; both forbid the flag when an
enzyme exists.

**Pipeline consistency:** The prompts mirror field ownership: Stage 1 records
explicit evidence and Stage 4 may resolve the absence of a real enzyme.

---

### 2026-07-10 — Concrete per-stage payload contracts

**Files changed:** `docs/pipeline.md`

**Error / symptom:** The eight stages described behavior but not exact input and
output shapes, allowing field ownership and wrapper creation to drift.

**Root cause:** Documentation named broad stage responsibilities without tying
them to `schema.py` types, boundary validators, or failure effects.

**Fix:** Added an eight-stage contract table naming the concrete TypedDict
inputs/outputs, exit guarantees, validator ownership, audit cadence, and the
Stage 2/Stage 6 wrapper-creation distinction. It also records that the current
Streamlit path has no separate live Stage 2 `map_payload` call.

**Pipeline consistency:** This documents the existing stage architecture and
its enforceable boundaries without assigning implementation logic to docs or
inventing an orchestration pass.

---

### 2026-07-10 — Remove the legacy non-IR PWML writer fallback

**Files changed:** `src/t2pw/pwml/writer.py`, `src/pwml_writer.py`,
`tests/test_pwml_writer.py`

**Error / symptom:** The writer still contained a second raw-payload export
implementation alongside the IR-backed pipeline. That branch used legacy
defaults and process builders, so invoking a different entrypoint could produce
PWML with behavior inconsistent with the validated Stage 8 path.

**Root cause:** `_populate_sections` and nine associated entity, process, and
layout builders predated the mapped-JSON → IR → PWML architecture but remained
reachable through `load_extraction`, `run_writer`, a raw argparse surface, and
the top-level `pwml_writer.py` shim.

**Fix:** Removed the 1,701-line raw/non-IR builder branch, its loading and CLI
entrypoints, and the obsolete top-level shim. The module entrypoint now exposes
only `run_pwml_pipeline_export`; writer tests were converted to exercise the IR
path or removed where they covered only the deleted fallback.

**Pipeline consistency:** There is now one authoritative Stage 8 serialization
route. Every command-line export passes through normalization, IR construction,
contract validation, the deterministic IR writer, and PWML QA.

---
