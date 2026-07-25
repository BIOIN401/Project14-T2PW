# Contract-Compliance Audit — RAG vs. the PWML Pipeline Contracts

**Date:** 2026-07-25
**Branch audited:** `rag-payload-gate-guardrails` (most recent commits; HEAD `0690f71`, 2026-07-24)
**Scope:** Read-only audit. No code was changed. This document catalogs the
"contract" failures that keep recurring when the new RAG subsystem is enabled,
records which recent fixes are actually present in the code, and lays out the
issues that are **still open or have regressed** and are blocking a clean PWML
export.

> **How to read this file.** Every finding is tagged:
> **✅ FIXED & VERIFIED** (changelog claim confirmed present in code),
> **🔴 OPEN / ACTIVE** (currently broken, blocking, or regressed),
> **🟡 PARTIAL** (fix landed but leaves a gap).
> File:line references were checked against the working tree on this branch.

---

## 0. What "the contracts" are

The pipeline enforces a chain of **boundary contracts**. A payload must satisfy
each one to advance; the last one is what actually gates PWML generation. RAG
plugs in just before the core pipeline, so a malformed RAG payload surfaces as a
contract failure downstream. The contracts live in two files:

| Contract (boundary) | Function | File | Effect on failure |
|---|---|---|---|
| Post-extraction | `validate_post_extraction` | `src/t2pw/pipeline/stage_contracts.py:90` | abort |
| Post-mapping (Stage 2B→3) | `validate_post_mapping` | `stage_contracts.py:104` | abort |
| Post-normalization (Stage 3) | `validate_post_normalization` + `run_strict_post_normalization_gates` | `stage_contracts.py:177`, `process_normalizer.py:3805` | feed audit / hard-gate |
| **Post-remap (Stage 6)** | `validate_post_remap` | `stage_contracts.py:204` | **abort** |
| Post-audit | `validate_post_audit` | `stage_contracts.py:278` | abort |
| **Pre-export (PWML required-field)** | `validate_pre_export` → `validate_required_pwml_contract` | `stage_contracts.py:291`, `pwml/ir.py:2010` | **abort — blocks PWML** |

The two contracts RAG keeps tripping are **post-remap** (generated
protein-complex wrappers) and **pre-export** (the PWML required-field gate:
`reaction_missing_*_participants`, `protein_missing_external_identity`,
`reaction_enzyme_must_be_protein_complex`, `duplicate_reaction_enzyme_complex`,
`generated_complex_component_missing_*`).

All six contracts are correctly wired into the orchestrator
(`src/t2pw/app/streamlit_app.py:51-57`, called at 1524 / 1544 / 1662 / post-remap
/ pre-export), and the strict Stage-3 gate feeds `refinement_gate_errors` which
blocks export until resolved (`streamlit_app.py:881-894`). **The enforcement
scaffolding is sound.** The failures below are about payloads that violate the
contracts, not about missing enforcement.

---

## 1. Executive summary

The recent RAG hardening work (2026-07-21 → 07-23) is real and largely landed:
the payload-gate guardrails, the reversible-reaction fix, the synonym-merge fix,
and the `" ; "`-blob fix are all present in the code as the changelog claims
(§3). **However, the three newest commits express unresolved frustration**
("papers are being rejected", "still not running rag from guardrails", "the
protein complex wrapper bug is abck"), and the audit confirms **three active,
still-open problems** — one of which (the wrapper regression) has **no changelog
entry and no code fix at all**:

| # | Issue | Status | Blocks |
|---|---|---|---|
| A | Protein-complex wrapper regression from **stale mapping cache** | 🔴 OPEN — undocumented, no fix | Post-remap + pre-export contracts |
| B | RAG "guardrail" auto-trigger is **unreachable dead code**; `RAG_ENABLED` defaults off | 🔴 OPEN | RAG never runs from scope/gap signals |
| C | Related **papers structurally un-selectable** (candidate re-preprocess without scope; sparse-seed scoring) | 🔴 OPEN | RAG fetches/keeps ~0 papers |
| D | Prompt-injection hardening is **partial** | 🟡 PARTIAL | Security seam |
| E | `_EMPTY_CONTEXT` omits scope fields → triage can't see clarity | 🔴 OPEN (compounds A/B) | Auto-triage |

Everything in §3 is verified done. Everything in §2 is still open.

---

## 2. OPEN / ACTIVE issues (blocking a clean PWML)

### 🔴 A. Protein-complex wrapper regression — stale `enzyme_complexes` cache rows

**This is the "protein complex wrapper bug is abck" from HEAD commit `0690f71`.**
That commit changed **only data/output artifacts** (`data/id_mapping_cache.json`,
`data/enrichment_cache.json`, `out/…`, `tmp/…`) — **no source code and no
changelog entry.** The regression is therefore currently **undocumented** and
**unfixed**.

**Symptom (the contract it breaks):** `validate_post_remap`
(`stage_contracts.py:226-273`) and the pre-export PWML gate
(`pwml/ir.py:2246-2336`) raise `generated_wrapper_component_protein_unresolved`
/ `generated_wrapper_component_missing_species` /
`generated_wrapper_component_missing_external_identity` (historically surfaced as
`Protein 'NdmA complex' is missing species/organism` etc.).

**Root cause — the cache, not the code.** Stage 6 synthesizes one-member
"wrapper" complexes named `"{protein} complex"` when no real PathBank complex
resolves (`map_ids.py:2432-2451` and the rewriter `_rewrite_reaction_protein_enzymes_to_complexes`
at `map_ids.py:4639-4747`). The code path was hardened over many commits (a guard
at `map_ids.py:4798-4831` skips wrapping a protein that lacks species/identity; an
authoritative `generated: True` flag was added; stoichiometry was consolidated).
**But the mechanism regresses through the persisted cache:**

1. `_resolve_complex_name` reads the cache **before** any resolution
   (`map_ids.py:4689`) and only re-resolves when the cache misses. A stale
   malformed row is used verbatim, bypassing the DB path and its guards.
2. `data/id_mapping_cache.json` currently holds **19 `enzyme_complexes` rows,
   every one with `chosen_rule: "novel_enzyme_single_component_complex"`,
   `status: "unmapped"`, and NO `generated` flag** — legacy rows that predate the
   flag. Verified directly; sample keys: `nmethyltransferase::::camellia sinensis`,
   `xanthine oxidase::::pseudomonas putida`, `tmum::::pseudomonas sp cbb1`. Many
   have `species_id: None` and components with no UniProt id.
3. `_merge_complex_resolution_into_row` copies `generated` **only if present**
   (`map_ids.py:4482-4483`) — so these rows never gain the flag — **but always**
   copies `mapping_meta.chosen_rule` and `mapping_meta.resolution`
   (`map_ids.py:4501, 4504`).
4. `is_generated_complex_wrapper` (`entity_identity.py:141-160`) has a
   **deliberate legacy fallback** (lines 156-160) that still flags a row as a
   generated wrapper when `chosen_rule`/`order_step ==
   "novel_enzyme_single_component_complex"`, even without the `generated` flag.
5. So every run re-flags these cache rows as generated wrappers, their unmapped
   members fail the wrapper-integrity checks, and the original errors reappear —
   **with no code change**, exactly matching "back for some reason."

**Live evidence in the tree:** `tmp/draft_graph.json` (regenerated by the HEAD
commit) contains a node `QTRT1/QTRT2 complex` classified as `kind: protein` — a
complex-shaped name sitting in the protein registry, the classic leak pattern.

**Not caused by RAG synthesis directly.** `src/t2pw/rag/synthesize.py` never
builds protein complexes/wrappers — it emits enzymes as plain protein actors
(`_enzyme_actor`, `synthesize.py:1124-1131`) and deliberately leaves
`compounds`/`proteins`/`protein_complexes` to be rebuilt (`to_payload`,
`synthesize.py:1167-1179`). RAG only *supplies the protein actors* that Stage 6
later wraps. The regression is the cache + Stage-6 reuse, independent of RAG.

**What needs doing (not done):** invalidate/rebuild the `enzyme_complexes`
namespace in `data/id_mapping_cache.json`, or backfill `generated` + component
identity on cache read, or make the legacy `chosen_rule` fallback require a
resolvable member before it flags a wrapper. History note: `docs/change_log.md`
calls this "entry #6 of an 8-entry circular-fix chain"
(`change_log.md:1078`); the cache-reintroduction ("re-divergence") mechanism is
the recurring theme (`change_log.md:2368-2371`, `1065-1133`). **No fix has been
attempted for the current recurrence.**

---

### 🔴 B. RAG "guardrails" never run — the auto-trigger is unreachable dead code

**This is "it still not running rag from guardrails we have set up" (commit `65b8112`).**

**Finding.** The scope/gap "guardrails" in `t2pw.rag.triage.should_run_rag`
(`triage.py:140-161`) — auto-trigger when `scope_clarity_score < 0.5` or when a
graph gap (dangling reaction / orphan metabolite / unmapped enzyme / missing
precursor / missing compartment) is present — **can never be the deciding
factor** in the real wiring:

1. The orchestrator only invokes RAG when the UI incomplete-flag is already set:
   `if rag_config()["enabled"] and rag_incomplete_flag:` (`streamlit_app.py:3024`).
2. It then passes that same flag through as `user_flag=bool(rag_incomplete_flag)`
   (`streamlit_app.py:3028`).
3. `should_run_rag` **short-circuits on `user_flag` at the very top**
   (`triage.py:133-138`) and returns `run=True` before ever reading
   `scope_clarity_score` or the gap reports.

So the guardrail branch (`triage.py:140-161`) is only reachable when
`should_run_rag` is called with `user_flag=False` — which the orchestrator never
does. It is exercised only by unit tests. **RAG runs iff the checkbox is on; it
never auto-starts *because* the scope/gap guardrails detected a novel/incomplete
pathway.** That is precisely the "not running rag from guardrails" complaint.

**Compounding cause.** `RAG_ENABLED` defaults to **`False`** (`config.py:117`).
If the env var isn't set, the toggle isn't even rendered
(`streamlit_app.py:2743`) and `rag_incomplete_flag` is forced `False`
(`streamlit_app.py:2756`) — no RAG path runs at all. This is the most literal
explanation for "still not running rag."

**What commit `65b8112` actually changed:** only `preprocessor.py` robustness
(`max_tokens` 500→2000, truncated-JSON salvage) and one UI warning branch. It did
**not** touch the gate at `streamlit_app.py:3024` or the `user_flag`
short-circuit. **The stated goal was not addressed by the diff.**

**What needs doing (not done):** to make guardrails auto-start RAG, `maybe_run_rag`
must be called regardless of the toggle (or `user_flag` must be decoupled from the
auto-triage decision), and `RAG_ENABLED` must be true in the environment.

---

### 🔴 C. Papers are rejected / "scoping seems rejected"

**This is "the papers are being rejected and the rag is basically not working,
the scoping seems to be rejected" (commit `daf37c5`).**

Two compounding mechanisms make related papers structurally un-selectable, both
**untouched** by the three recent commits (which were diagnostic-only):

**C1 — Candidates are re-preprocessed with no scoping context.**
`select._candidate_context` (`select.py:260-280`) calls `preprocess(text)` on each
candidate abstract **without** `user_task_context`. Per the Stage-0 prompt's
Case B/C logic (`preprocess_system.txt:49-91`), a review abstract with no named
example deterministically becomes **Case C** (`multi_example_review`, blank
scope). `select._review_assessment` (`select.py:283-336`) then finds no matching
example and applies `_REVIEW_PENALTY_NO_MATCH = 1.0` (`select.py:69`), driving the
score negative → dropped below the `_MIN_SCORE = 0.0` floor (`select.py:75,
497-500`). **Related review papers are therefore almost impossible to select.**

**C2 — A sparse seed context collapses scoring.** When the seed itself is a
novel/ambiguous pathway (Case C blanks `pathway_name` / `key_compounds` /
`key_proteins`), `total_terms == 0` → `entity_overlap == 0.0` for every candidate
(`select.py:365, 374`), and `organism_match` falls to the neutral 0.5. Scores
compress toward the 0.0 floor, so any small penalty tips candidates into
rejection. This is the "scoping seems rejected" symptom — **the seed's own empty
scope starves the scorer.**

**C3 — Empty-query short-circuit.** If Stage 0 produced no pathway name / organism
/ compounds / proteins / gaps, `acquire.search_candidates` returns `[]`
immediately (`acquire.py:727-732`, sets `status["empty_query"]=True`) and never
hits the network. A Case C ambiguous review, or any failed preprocess, produces
exactly this — 0 papers fetched.

**Live evidence:** `tmp/draft_graph.json` metadata shows `pathway_name: ""` and
`organism: ""` for the current queuosine-salvage (PNAS multi-example review) run —
the empty-scope condition that drives C1–C3.

**What the recent commits did do (real, but not this):** the 2026-07-23 Stage-0
fixes (`preprocessor.py`) made **Case B reachable** by threading
`user_task_context` into Stage 0, raised the token budget, and added the truncated
-JSON salvage + `preprocess_status` diagnostics. Those genuinely fix the *seed
paper's* scope when the user names a target example. **They do not change any
rejection logic in `select.py` / `acquire.py`** — the candidate-selection path
still re-preprocesses without scope (C1) and still starves on a sparse seed (C2).

---

### 🟡 D. Prompt-injection hardening is partial

Commit `59dce04` ("fix for prompt injection") added `_format_user_task_context`
(`preprocessor.py:56-71`), which wraps user scoping text in a
`<user_task_context>` block and neutralizes a closing `</user_task_context>` tag.
**Residual gaps:** (1) the Stage-0 prompt also honors a `<pathway_scope>` block
(`preprocess_system.txt:70`) whose tag is **not** neutralized; (2) the seed
**document text** is interpolated raw between `<<<`/`>>>` delimiters
(`preprocessor.py:112-118`) with no sanitization — an uploaded paper containing
those delimiters or injected instructions is undefended. The fix closes the
user-focus box but not the document body.

---

### 🔴 E. `_EMPTY_CONTEXT` omits the scope fields

`_EMPTY_CONTEXT` (`preprocessor.py:11-20`) does not contain `scope_status`,
`scope_clarity_score`, `document_type`, `selected_example`, or
`candidate_examples`. On any Stage-0 failure (`llm_error` / `empty_reply` /
`unparseable`), `context.get("scope_clarity_score")` is therefore `None`, and
triage's low-clarity auto-trigger cannot fire (`triage.py:143-145`). This silently
compounds issues A/B: a failed preprocess looks identical to a clear pathway to the
triage guardrail.

---

## 3. FIXED & VERIFIED — recent RAG contract hardening that IS present in the code

These changelog entries were checked against the working tree and are correctly
implemented. They are the guardrails that keep RAG payloads passing the contracts
today; **do not assume they are the problem** — they hold.

- **✅ RAG payload bypassed post-merge hardening (2026-07-22).** The adopted RAG
  payload is passed through `apply_post_merge_cleanup` **before** it can replace
  `final_payload` (`streamlit_app.py:3093-3101`, cleanup at 3093, assignment guarded
  at 3100-3101). The adoption is guarded so the synthesized payload only wins when
  it preserves at least the seed reaction count
  (`_count_reactions(rag_payload) >= max(1, _count_reactions(final_payload))`).
  `apply_post_merge_cleanup` / `filter_unresolvable_reactions` /
  `_looks_like_metabolite_fragment` all exist (`pipeline.py:1014, 716, 1002`), and
  the cleanup **deliberately excludes** `_inject_name_based_modifiers` (documented
  docstring `pipeline.py:1032-1034`; injection kept in the merge path at
  `pipeline.py:1094`). *(Naming nit: the changelog calls the count helper
  `_n_reactions`; the code names it `_count_reactions` — cosmetic only.)*
- **✅ Reversible reaction dropped a locked direction (2026-07-23).**
  `_Reaction.conflict_key` is direction-aware, keyed on `(sorted input names,
  sorted output names)` (`synthesize.py:223-247`), so a forward/reverse pair keys to
  two distinct groups and both survive — matching the core `dedupe_processes`.
- **✅ RAG under-merged cross-paper synonym duplicates (2026-07-23).**
  `t2pw.rag.synonyms.build_offline_synonym_resolver` exists and is threaded as an
  optional grouping-only resolver (default `None` = prior behavior).
- **✅ `" ; "`-joined pathway-metadata blobs emitted as entities (2026-07-21).**
  `_reactions_from_bundle` only transcribes `paper`-type chunks;
  `_is_invalid_species_token` + the `_quarantine_pathway_metadata_blobs` normalizer
  guard reject the blobs.
- **✅ Four live-run defects that emptied multi-paper output (2026-07-21)** — seed
  descriptor dict, guarded S3 adoption, dim-mismatch lexical fallback, dict-shaped
  participant symbols — all present.
- **✅ Stage-0 scope context + failure diagnostics (2026-07-23) and truncated-JSON
  salvage (2026-07-23).** `user_task_context` threading, `preprocess_status`
  (`ok`/`llm_error`/`unparseable`/`empty_reply`), and `_repair_truncated_json` are
  all in `preprocessor.py`. *(These fix the seed's scope, not candidate
  selection — see §2.C.)*

---

## 4. Pre-existing OPEN issues (from the changelog "Open Issues" section)

Still open independent of RAG; several block or degrade export. Carried forward
here so the full picture is in one place (source: `docs/change_log.md:2235-2611`).

- **🔴 `NdmCDE` cross-bucket duplicate (Stage 3 location normalization).** A declared
  complex re-appears as a bare protein row via `element_locations` rewriting;
  gate rejects it for missing identity. Marked "IMPLEMENTED — LIVE RERUN VERIFIED"
  but retains remaining work (`change_log.md:2243-2316`). Related in mechanism to
  issue A.
- **🟡 Best-effort UniProt fallback assigns lowest-scored candidate**
  (`change_log.md:2426`). Clears the identity gate but may assign a wrong accession;
  proper fix is BLAST sequence disambiguation. `best_effort: True` flag exists.
- **🔴 Stage 4 audit LLM connection failure prevents semantic repair**
  (`change_log.md:2447`). Config/infra: when the LLM endpoint is unreachable, gate
  failures fed to audit cannot be repaired. Blocks the OPC / beta-oxidation repairs
  below.
- **🔴 Empty reaction inputs in beta-oxidation chain** (`change_log.md:2478`) and
  **OPC-8/6/4 misclassified as proteins** (`change_log.md:2517`) — audit-repair
  tasks that need the LLM connection restored.
- **🟡 DB unavailable → degraded mapping** (`change_log.md:2550`). Non-strict export
  only; compound/protein coverage drops sharply without the PathBank MySQL DB.
- **🟡 Enrichment stage produces data no stage consumes** (`change_log.md:2581`). Dead
  Stage-7 output; decide wire-in (Option A) vs remove (Option B).

---

## 5. Recommended next actions (no code written here)

Ordered by impact on getting a clean PWML out with RAG on:

1. **Issue A (wrapper regression) — highest priority, currently undocumented.**
   Purge/rebuild the `enzyme_complexes` namespace of `data/id_mapping_cache.json`
   (19 stale rows) and re-run; if the errors clear, the regression is confirmed as
   pure cache poisoning. Durable fix: on cache read, either drop rows lacking the
   `generated` flag + a resolvable member, or require a resolvable member before the
   legacy `chosen_rule` fallback in `is_generated_complex_wrapper` fires. Add a
   changelog entry — there is none.
2. **Issue B — decide the guardrail semantics.** Either call `maybe_run_rag`
   unconditionally and let `should_run_rag` (with `user_flag=False`) decide from
   scope/gap signals, or accept that RAG is toggle-only and delete/relabel the
   auto-triage branch so it isn't mistaken for a live guardrail. Ensure
   `RAG_ENABLED=true` in the run environment.
3. **Issue C — fix candidate selection.** Thread the seed's `user_task_context`
   (or seed scope) into `select._candidate_context` so review candidates don't
   deterministically fall to Case C and eat the 1.0 no-match penalty; and/or relax
   the `_MIN_SCORE`/review-penalty interaction when the seed context is sparse.
4. **Issue E — add the scope keys to `_EMPTY_CONTEXT`** (or have triage treat a
   missing `scope_clarity_score` as low-clarity) so a failed preprocess can still
   trip the guardrail once Issue B is addressed.
5. **Issue D — extend injection neutralization** to `<pathway_scope>` and the raw
   document-body delimiters.

---

## 6. Proposed fix designs (PENDING AUTHORIZATION — no code changed)

> **Status: PROPOSED. Nothing here is implemented.** These outlines were produced
> by read-only design agents against the `rag-payload-gate-guardrails` code, then
> put through an adversarial **clash-check** (do the fixes conflict?) and a
> per-fix **eye-test** (does the fix introduce a *new* problem or new contract
> failure?). Only proposals that survived both are written as "APPROVED FOR
> IMPLEMENTATION"; where a proposal failed the eye-test, the failure and the
> required amendment are recorded rather than hidden. Author to review and
> authorize before any code is written.

### 6.0 Cross-fix clash matrix

| Pair | Shared surface | Clash? | Resolution |
|---|---|---|---|
| A ↔ B/C/E | none (A is in `map_ids.py`/cache; B/C/E in RAG + preprocessor) | No | Land independently |
| A ↔ D | none | No | Independent. Note: D must **not** sanitize `enzyme_complexes` cache keys (`map_ids.py:4687`) or it would mass-invalidate the namespace |
| B/C/E ↔ D | `preprocessor.py`, `pipeline.py` — **different functions** | Soft | Co-design: land together; D's escapers must wrap the seed-derived `user_task_context` that C newly threads into candidate `preprocess`. `_EMPTY_CONTEXT` (E) and `_format_user_task_context` (D) are different hunks → merge-clean |
| B ↔ E (internal) | `should_run_rag` decision path | Coupled | E's "do **not** default `scope_clarity_score` low" is load-bearing so a transient Stage-0 failure doesn't auto-fire RAG; must land with B |

No two approved fixes touch the same function. The only real coupling is the
B+E ordering and the B/C/E+D co-design, both noted per-fix below.

---

### 6.A Wrapper regression — stale `enzyme_complexes` cache · ✅ APPROVED FOR IMPLEMENTATION

**Approach (two parts):**
1. **One-time data purge** — drop the 19 stale rows in the `enzyme_complexes`
   namespace of `data/id_mapping_cache.json` (this is what HEAD commit `0690f71`
   should have done; the namespace is fully regenerable derived data).
2. **Durable read-side guard** — in `map_ids._resolve_complex_name` (cache read
   at `map_ids.py:4689`), add a predicate
   `_is_reusable_complex_cache_row(row) -> bool` that treats a cached row as a
   **miss** when it is wrapper-shaped (`is_generated_complex_wrapper`) *and* fails
   the same integrity bar the gates enforce (no `generated: True`, or no species
   context, or no component external identity). A rejected read drops into the
   existing re-resolution branch and `cache.set`/`save` self-heals the stored row.

**Rejected alternative:** loosening `is_generated_complex_wrapper`'s legacy
`chosen_rule`/`order_step` fallback (option "c"). It would make a genuinely broken
wrapper **pass the gate silently** into PWML (turning a loud abort into a bad
export) and would break `tests/test_entity_identity_contracts.py:22-27`. The
classifier is correct; the *data* is bad.

**Eye-test (new problems?):** None found. The predicate rejects only rows the
gates would already fail on, so it converts a downstream abort into an upstream
self-heal; legitimately-resolved wrappers (e.g. `NdmA complex` with a mapped,
UniProt-bearing `NdmA`) keep `generated: True` + real species + identity → judged
reusable → unchanged. No enzyme is ever silently dropped (re-resolution routes
through the existing `map_ids.py:4798-4831` guard, which logs its skips). RAG-off
path untouched (this is DB/cache mapping). Add one regression test: a
`_MemoryCache` seeded with a legacy malformed row asserts re-resolution rather than
a failing wrapper. Existing wrapper/cache tests (`test_map_ids.py:669/1038/1115`,
`test_entity_identity_contracts.py`) stay green.

**Files:** `map_ids.py` (new predicate + 2-line guard), `data/id_mapping_cache.json`
(purge), `tests/test_map_ids.py` (new test). No gate/classifier changes.

---

### 6.B RAG auto-triage unreachable · ⚠️ APPROVED **ONLY WITH AMENDMENT** (naive fix fails the eye-test)

**Design intent (good):** keep the UI toggle as an explicit "force RAG on"
override, and *also* let the scope/gap auto-triage decide when the toggle is off,
by removing only the `and rag_incomplete_flag` clause from the orchestrator gate
(`streamlit_app.py:3024` → `if rag_config()["enabled"]:`), keeping the
`rag_config()["enabled"]` guard so the RAG-off path stays byte-identical. Handle
the `RAG_ENABLED=False` default in the **environment** (`.env`), not by flipping
the code default (default-off is the regression firewall).

**🔴 Eye-test FAILURE (found during clash-check — the design agent's own trace
missed this):** With the gate opened, **every** enabled run with the toggle off
calls `should_run_rag(user_flag=False)`, which runs `_gap_signals` →
`retrieve.detect_gaps`. The gate already passes `reports={"qa_graph":
generate_qa_report(...)}` (`streamlit_app.py:3030`), and `detect_gaps`
(`retrieve.py:342-357`) turns **every** `flags.missing_compartments` entry into a
`missing_compartment` gap — and `missing_compartment` **and** `orphan_metabolite`
are in the auto-trigger set `_INCOMPLETE_GAP_KINDS` (`triage.py:52-58`). Those two
signals are present in most *healthy* pathways: the current queuosine
`tmp/qa_report.json` lists ~20 entities with "no subcellular location recorded."
So the naive gate-open would auto-fire RAG on nearly every pathway that has any
uncompartmentalized entity — effectively turning RAG on for everyone. **Rejected
as written.**

**Required amendment (makes it pass):** couple the gate-open with a **narrowing of
the auto-trigger signal set** so only low-noise signals can auto-fire RAG:
- Keep `scope_clarity_score < 0.5` (only genuinely ambiguous reviews score ≤0.2;
  clear papers score ≥0.7 — low false-positive rate).
- Restrict `_INCOMPLETE_GAP_KINDS` for **auto-trigger** to strong structural
  incompleteness only — `dangling_reaction`, `unmapped_enzyme`, `missing_precursor`
  — and **exclude `missing_compartment` and `orphan_metabolite`** (or require a
  minimum count / multiple distinct kinds). The excluded signals remain useful for
  *gap retrieval once RAG is already running*; they just must not be the thing that
  *starts* RAG unattended.

With the amendment, trace: (a) toggle on → `user_flag` short-circuit → runs, as
today; (b) toggle off + genuinely ambiguous review (score 0.2) → auto-fires
correctly; (c) toggle off + normal complete pathway with only missing compartments
→ **does not** fire (the amendment's whole point); (d) transient Stage-0 failure →
no `scope_clarity_score` present (E keeps it absent) → no trigger. No double-fire
(toggle + auto fold into one `should_run_rag` call whose first line is the
`user_flag` short-circuit).

**Files:** `streamlit_app.py` (gate line 3024 + toggle help text — its "OFF =
nothing changes" wording becomes false and must be rewritten to "OFF = auto-decide
from scope/gap"), `triage.py` (`_INCOMPLETE_GAP_KINDS` used for auto-trigger).
**Product decision needed:** if the team wants a hard per-run OFF, do **not** open
the gate — instead delete/relabel the dead auto-branch (option iii). The amendment
above assumes the team wants auto-triage to work.

---

### 6.C Related papers structurally un-selectable · ✅ APPROVED FOR IMPLEMENTATION

**Approach:**
1. **Thread seed scope into candidate preprocessing (C1).** Give
   `select._candidate_context(candidate, *, user_task_context=None)` the kwarg and
   forward it to `preprocess`. In `select()`, derive it once from the seed —
   `seed_focus = seed_context.get("selected_example") or
   seed_context.get("pathway_name")` — and pass it per candidate. A review abstract
   about the seed pathway then reaches Case B (matched example) and gets
   `_REVIEW_PENALTY_MATCH = 0.25` (rank-below) instead of
   `_REVIEW_PENALTY_NO_MATCH = 1.0` (force-drop below the 0.0 floor).
2. **Sparse-seed penalty gate (C2).** In `score_candidate`, compute
   `seed_has_signal = total_terms > 0 or bool(seed_org)`; when the review
   assessment would apply the 1.0 no-match penalty **and** the seed has no signal,
   downgrade it to 0.25 (rank-below, not drop). Gate the *penalty*, not
   `_MIN_SCORE` — lowering the floor would also admit zero-signal *primary* papers.

**Interaction note (C3, do not change):** `acquire.search_candidates` still
short-circuits to `[]` when the seed has no queryable terms
(`acquire.py:727-732`). The select fixes only bite once acquisition returns
candidates, i.e. when the seed carries at least an organism or named example. A
truly term-less seed correctly no-ops rather than fetching noise — document this,
don't "fix" it.

**Eye-test (new problems?):** No irrelevant-paper leakage in the common case: when
the seed *has* signal (the norm), the full 1.0 penalty is preserved, so off-topic
reviews still drop. The downgrade only applies when the seed is too sparse to
*prove* a review off-topic — relaxing exactly the case the penalty can't justify.
No extra LLM calls (`_candidate_context` already calls `preprocess` once per
candidate; only an argument is added). Tests: `_fake_preprocess(text, **_)`
absorbs the new kwarg; `_SEED` always has signal so penalty tests stay green. Add a
sparse-seed test asserting the downgrade.

**Files:** `select.py` (`_candidate_context`, `score_candidate`, `select`,
`_review_assessment`).

---

### 6.D Complete the prompt-injection hardening · ✅ APPROVED FOR IMPLEMENTATION

**Broader surface than the audit stated (found while designing):** the raw
`<<<`/`>>>` document fence is fed untrusted text at **four live sites**, not one —
`preprocessor.preprocess()`, `pipeline._build_extraction_prompt` /
`_build_inference_prompt`, and `audit_json_llm._build_llm_prompt` (whose
`references` is the external RAG retrieval payload). `_format_user_task_context` is
**duplicated** in `preprocessor.py` and `pipeline.py` (hand-synced to avoid a
circular import). `completeness_audit._build_user_prompt` uses a *different*,
weaker markdown-header scheme.

**Approach:**
1. Add a `_format_pathway_scope` helper mirroring `_format_user_task_context`, and
   neutralize the `</pathway_scope>` close-tag at its only emit site
   (`pipeline.py`, inside `_build_extraction_prompt`).
2. Add `_escape_fence(text)` that neutralizes **only the exact 3-char
   sequences** `>>>`/`<<<` (mirroring the existing `<\/tag>` style) and apply it to
   every untrusted body at the four fence sites (including the RAG `references`).
3. **Kill the duplication:** introduce a leaf module `t2pw/…/prompt_safety.py`
   (no project imports → no circular dependency) exposing
   `neutralize_close_tag` / `format_scoped_block` / `escape_fence`, and route all
   three call sites through it. Fallback if minimal churn is preferred: duplicate
   the helpers and document the sync burden as the code already does.
4. **`completeness_audit`:** either fence `source_text` with the same scheme or
   explicitly log it as a known residual gap — do not leave it silently
   inconsistent.

**Eye-test (new problems?):** Byte-identical-when-no-injection is preserved —
`_escape_fence` is a no-op on any string lacking the triple delimiter, so a normal
paper (the `test_preprocessor.py` fixtures use delimiter-free bodies) yields an
unchanged prompt. Restricting to the exact 3-char sequences leaves single `<`/`>`
(inequalities, `->` arrows, FASTA `>` headers, charge states) untouched → no
corruption of scientific notation. Before landing, grep RAG/audit fixtures for
literal `<<<`/`>>>` in payloads (unlikely in JSON) so `references`-escaping doesn't
shift an assertion. New unit tests needed for the `<pathway_scope>` escape and the
fence escaper (no existing coverage there).

**Sequencing with B/C/E:** land **together**; D's escapers must wrap the
seed-derived `user_task_context` that C newly threads into candidate `preprocess`.
D stays additive (new helpers, not a rewrite of `_format_user_task_context`), so
B/C/E's new *call sites* and D's *helpers* merge cleanly.

**Files:** `preprocessor.py`, `pipeline.py`, `audit_json_llm.py`,
`completeness_audit.py`, new `prompt_safety.py`, `tests/test_preprocessor.py`.

---

### 6.E `_EMPTY_CONTEXT` scope fields · ✅ APPROVED FOR IMPLEMENTATION (minimal scope)

**Approach (deliberately minimal):** add **only** `document_type: ""` and
`scope_status: ""` to `_EMPTY_CONTEXT` (`preprocessor.py:11-20`). Both are
**truthiness-guarded** in `format_context_header` (`if document_type:` /
`if scope_status:`, verified at `preprocessor.py:289,293`), so empty defaults never
emit a header line → single-paper extraction prompt stays byte-identical.

**Explicitly do NOT add** `scope_clarity_score` or `selected_example`:
`format_context_header` guards `scope_clarity_score` by *membership + not-None*
(lines 271,281,295) and `selected_example` by *membership* (line 297), so adding
either to `_EMPTY_CONTEXT` would emit a spurious header line for every
previously-headerless context — a single-paper regression (verified against the
code).

**Design call (important):** this fix intentionally does **not** make a *failed*
Stage-0 run auto-trigger RAG. Defaulting `scope_clarity_score` low would fire RAG
on every transient LLM hiccup (and buy nothing — an empty context makes
`build_query` return `""`, so acquisition short-circuits to zero anyway). Current
triage behavior (`_as_float(None) → None → no signal`) is correct and must be
preserved. E is therefore a robustness/observability fix, not a triage change; it
composes safely with the B amendment.

**Eye-test (new problems?):** None — two truthiness-guarded empty strings only.
Audit `test_preprocessor.py` for any exact `_EMPTY_CONTEXT` shape/length assertion
and update if present.

**Files:** `preprocessor.py` (shared with D — different hunk), possibly
`tests/test_preprocessor.py`.

---

### 6.F Recommended implementation sequence

1. **A** — independent, unblocks the current failing export; ship first, alone.
2. **B (amended) + E together** — the gap-set narrowing and the "no low-score
   default" discipline are jointly load-bearing; do not land the gate-open without
   the gap-set narrowing.
3. **C** — after/with B so auto-triage actually reaches selection; independent
   files.
4. **D** — with or immediately after B/C/E so its escapers cover every untrusted
   value the RAG cluster newly routes into a prompt.

Each step should land with `RAG_ENABLED=false` verified byte-identical to `main`
and the existing suite green before the next.

---

## Appendix — verification method

- Branch selected by most-recent commit date across all remotes
  (`rag-payload-gate-guardrails`, 2026-07-24; next was `RAG-setup`, 07-22).
- Contracts, orchestrator wiring, `triage`/`select`/`acquire`/`preprocessor`
  gating, and the stale-cache vector were read directly and cross-checked.
- The 19 stale `enzyme_complexes` rows, the `QTRT1/QTRT2 complex`-as-protein node,
  and the blank `pathway_name`/`organism` in `tmp/draft_graph.json` were confirmed
  from the tree.
- Changelog claims in §3 were confirmed against `src/t2pw/app/streamlit_app.py`,
  `src/t2pw/pipeline/pipeline.py`, and `src/t2pw/rag/synthesize.py`.
- §6 fix outlines were produced by three read-only design agents (Issue A; the
  B/C/E cluster; Issue D) and then clash-checked. Load-bearing claims were
  re-verified against `rag-payload-gate-guardrails` source: the Issue B eye-test
  failure was confirmed by reading `triage._INCOMPLETE_GAP_KINDS` and
  `retrieve.detect_gaps` (which emits `missing_compartment` from
  `qa_report.flags.missing_compartments`, ~20 populated in the current run), and
  the Issue E header-guard behavior was confirmed in
  `preprocessor.format_context_header` (truthiness vs. membership guards).
- These design agents ran while the working tree was on this audit branch (based
  on pre-RAG `main`), so they read the RAG-branch code via `git show
  rag-payload-gate-guardrails` — line references in §6 are on that branch.
