# ORCH-754b — protein endpoints of interactions reach the PWML as single-protein complexes

**Status:** implemented on `orch754/b-interaction-wrappers` (base `f9892cdf`), committed, not
merged and not pushed. Code commit `3bf40cb6`; this document and the evidence are the docs
commit on top of it. Product-owner rulings applied: 2026-10-19, being recorded as **D-104**
(not yet in `DECISIONS.md` when this was written). Orchestrator scope additions of the same
day (re-pin, `competes_with`, app-path text record) are covered in § 6–8.

No LLM call, no pipeline run, no network. Every test and measurement went through
`bounded_run.py` with `T2PW_OFFLINE_CURATOR=1`; reports are under
`evidence/g11/ORCH-754b/`, pin verdicts under `evidence/g11/pin/ORCH-754b/`.

---

## 1. What changed

| file | change |
|---|---|
| `src/t2pw/mapping/map_ids.py` | `_rewrite_reaction_protein_enzymes_to_complexes` gains one loop, after the reaction loop and before the dropped-protein repair. For every interaction whose word carries a sign, each protein endpoint is rewritten to its single-protein wrapper through the **same** `_resolve_complex_name` closure the enzymes use (same resolver, same cache, same F-116 superset refusal, same `generation_reason: single_protein_pathwhiz_wrapper`). No second wrapper mechanism. |
| `src/t2pw/pwml/ir.py` | `interaction_pwml_verdict` — the one rule for what the PWML does with an interaction, used by the IR loop and by both text reports. A bare protein endpoint that lacks a species or a UniProt/DrugBank identity is withheld with a precise reason; `bare_protein_endpoint` stays the fallback when no wrapper exists. The `withheld` record now carries the row's `evidence`. `competes with` / `competes_with` removed from the sign table. |
| `src/t2pw/pipeline/draft_graph.py` | each interaction's evidence and the exporter's verdict on that payload are kept in `graph.metadata["interaction_records"]` (absent for an interaction-free graph). |
| `src/t2pw/pipeline/reaction_summary.py` | interaction rows: word as extracted, `Evidence:` (or `NONE RECORDED`), `PWML (if exported from this payload): …`. |
| `src/t2pw/rag/research_report.py` | interaction rows (JSON and text): `pwml_interaction_type`, `pwml_withheld_reason`, `pwml_export`; text line `PWML: none in research mode; a strict export of this payload: …`. |
| `tests/test_pwml_interaction_protein_wrappers.py` | **new**, 17 tests. |
| `tests/test_pwml_interaction_type_spellings.py` | `competes with` / `competes_with` moved from INHIBITION to UNSIGNED; one variant row removed; the end-to-end word list swaps `"competes with".` for `prevents`. Net −1 test. |
| `tests/test_compound_resolution_extraction.py`, `tests/test_pwml_ir_duplicate_row_refusal.py` | the deliberate cumulative re-pin, § 6. |

`writer.py` is untouched: ORCH-747 already emits a ProteinComplex interaction element and its
`interaction_protein_complex_visualizations` (§ 2). `streamlit_app.py` is untouched. The tail
of `map_payload` is untouched (ORCH-754a's seam).

### The Stage-6 rule, exactly

A protein endpoint is wrapped only when **all** hold, otherwise **no endpoint of that row** is
wrapped and an `interaction_protein_endpoint_wrapper_skipped_invalid_component` action names
what is missing:

- the interaction's word maps to Activation/Inhibition (`ir.pwml_interaction_type`) — an
  unsigned row gets no wrapper, so no dangling complex enters the PWML for a row it cannot carry;
- the protein has a species (`protein_species_context`) and a UniProt/DrugBank identity
  (`has_protein_external_identity`) — **the same predicates** the IR's
  `generated_complex_component_missing_species` / `_missing_external_identity` errors use, so a
  wrapper can never turn a withheld row into a broken complex (the gate is not weakened; it is
  pre-empted);
- the protein is not the PathBank `Unknown` sentinel;
- no *other* complex already carries the wrapper's name. A complex named like the wrapper that
  **is** this protein's own one-protein wrapper (built for it as an enzyme, or on an earlier
  pass) is reused as it stands; a declared assembly is never borrowed, because pointing the
  endpoint at it would add its other members to the interaction.

Every pre-existing reaction, transport, protein and complex row is unchanged (measured, § 4).

---

## 2. PathWhiz import evidence

I cannot run PathWhiz. Everything below is read from `C:/Users/Angad/Desktop/SummerBIOIN/smpdb`
(read-only) and from a real PathBank export in this repo.

| claim | where |
|---|---|
| only two types | `app/models/data/interactions/interaction.rb:13` `INTERACTION_TYPES = %w(Activation Inhibition)`; validated `:32` |
| exactly one element per side | `interaction.rb:29-30` |
| the element is polymorphic; a protein-bearing element is a ProteinComplex | `interaction_element.rb:8` (`belongs_to :element, :polymorphic => true`), `:23-35` (`pwie_id` knows Compound, ElementCollection, NucleicAcid, ProteinComplex, Bound — nothing else); PathWhiz's own search index takes protein ids from `element_type: "ProteinComplex"` (`interaction.rb:50`) and species likewise (`interaction.rb:164-181`) |
| element validations | `interaction_element.rb:11` element present, `:12` stoichiometry > 0 |
| no visualization member for a bare protein | `app/models/visualization/interactions/` holds bound, compound, element_collection, nucleic_acid, protein_complex — no protein |
| complex member needs a complex visualization, an edge, a side | `interaction_protein_complex_visualization.rb:11-14` |
| the container exists on the visualization | `interaction_visualization.rb:13`, `:18` |
| importer order: complexes stored before interactions, complex visualizations before interaction visualizations | `lib/pwml_parser.rb:205-207` then `:219-221`; `:247-249` then `:270-275` |
| `element-id` remapped by its sibling `element-type` → `@protein_complex_list` | `pwml_parser.rb:572-581` (list declared `:117`) |
| `protein-complex-visualization-id` remapped | `pwml_parser.rb:544-547` (list `:118`) |
| saved with validations on | `pwml_parser.rb:768` (`object.save`), visualizations `:992` (`object.valid?`) |

**PathBank does exactly this.** `reference/PW687775.pwml` — a real PathBank export — interaction
15843 (lines 906–926) is an Activation whose left element is ProteinComplex 416678 "Ephrin-A1"
and right element ProteinComplex 416679 "Ephrin type-A receptor 2"; each complex (lines
751–776) has **one** `protein-complex-protein`. Its interaction-visualization (lines 1537–1553)
carries `interaction_protein_complex_visualizations` with a Left and a Right item, each
`id / protein-complex-visualization-id / edge-id / side`.

`test_the_exported_shape_matches_pathbanks_own_protein_complex_interaction` builds our PWML
(Stage 6 offline → IR → writer → repair) and asserts against that PathBank record: the same
interaction children, the same element children, `element-type` ProteinComplex on both sides,
the element id names a complex **in our protein-complexes section** holding exactly one
protein, and the visualization container with Left/Right items of PathBank's exact child
shape whose `protein-complex-visualization-id` exists in our document. The QA check
(`run_pwml_qa`) and `validate_generated_tree` pass on the same output.

**Import shape: confirmed by reading, not by running.** A live PathWhiz import is the only
proof that would settle it.

---

## 3. Exportable interactions, before and after

Script `evidence/orch754b_wrapper_base_proof.py` → `orch754b_wrapper_base_proof.json` / `.txt`
(G11 `ORCH-754b/14`). Corpus: the ORCH-749 one — 95 committed `final_mapped.json` legs with
interactions, 336 rows. Base `f9892cdf` vs tip, each in a child that verifies its `t2pw`.

**Direct** (`build_pwml_ir` on the committed payloads, DB down — the ORCH-749 census
configuration; reproduces its numbers exactly on the base). The committed payloads were mapped
before this card, so they carry no wrapper and **exportable cannot move here**; only reasons do:

| | exportable | untyped | bare_protein_endpoint | missing identity |
|---|---|---|---|---|
| base (63 legs build) | 6 | 139 | 57 | — |
| tip | 6 | 140 | 50 | 6 |

**Offline Stage-6 replay** (the wrapper step re-run on each payload with `db=None` and the
cache disabled, then pre-freeze with the DB down, then IR, then writer). **This is an estimate,
not a re-run of Stage 6**: with the database up a protein can resolve to a PathBank complex
instead, and later mapping passes (UniProt fallback, identity admission) can change which
proteins carry an identity.

| | exportable | untyped | bare_protein_endpoint | missing identity | `<interaction>` in PWML |
|---|---|---|---|---|---|
| base (94 legs build) | 13 | 231 | 86 | — | 4 |
| tip | **73** | 232 | 0 | 25 | **54** |

Strict legs 4 → 37 exportable rows, research legs 9 → 37. The PWML count is lower than the IR
count because the writer raises `ValueError` on 26 inputs **on both trees** (pre-existing,
identical stop; 19 of the 73 exportable rows sit in those legs).

**Upper bound, from the census:** the sign table (tip) maps 98 rows; of those, 85 have a
protein endpoint. So at most 98 rows can ever export from this corpus; the replay reaches 73;
the other 25 are withheld because a protein endpoint has no UniProt/DrugBank identity
(no row lacks a species). The untyped count rises by one because the corpus's single
`competes_with` row is no longer signed (§ 7).

Withheld rows by reason at the tip (replay): `untyped_interaction` 232,
`protein_endpoint_missing_external_identity` 25.

---

## 4. Preservation and confinement (G9)

Same script, same run.

- **Preservation:** on the 7 inputs with no protein endpoint (5 committed legs + 2 fixtures:
  an enzyme-only payload and an enzyme-catalysed reaction with a compound-only interaction),
  the Stage-6 payload, the Stage-6 report, the IR (bar the `evidence` key added to each
  withheld record), the IR report and the **PWML bytes** are identical base vs tip. PWML bytes
  were compared on 5 of the 7 (the other two do not reach a PWML on either tree).
  Verdict `PRESERVED`.
- **Confinement, all 99 inputs:** reactions, transports, proteins, every pre-existing complex
  row, and the IR's entities / reactions / transports are identical base vs tip; every added
  complex is a one-protein generated wrapper of a declared protein. Verdict `CONFINED`.
- No IR error and no `validate_required_pwml_contract` error added on any input.

New capability, so its tests are labelled **NEW ACCEPTANCE** and no base failure was
fabricated. The fixture's behavioural before/after is in the same record: base withholds all
four protein-endpoint rows as `bare_protein_endpoint`; tip exports 2 (0 → 2 `<interaction>`)
and withholds the other two with `protein_endpoint_missing_external_identity` /
`protein_endpoint_missing_species`.

---

## 5. Text output (D-104 § 3 / § 4)

**Verified against the standard, and it was not met.** ORCH-747 F-747-3 says both text outputs
carry per-row evidence. `reaction_summary.py` did not: a `DraftNode` has no evidence field, so
the interaction rows showed the word and the endpoints only. Neither report said what the PWML
does with a row.

Now each interaction row carries the word as extracted, its evidence (or
`NONE RECORDED — treat as unsupported`), and the exporter's own verdict computed by the same
`interaction_pwml_verdict` the IR uses, e.g.

```
  PWML (if exported from this payload): withheld (protein_endpoint_missing_external_identity): FtsH lacks a UniProt/DrugBank identity, so Stage 6 cannot wrap it
  PWML (if exported from this payload): withheld (untyped_interaction): 'binds' is neither Activation nor Inhibition, the only interaction types PathWhiz stores
  PWML (if exported from this payload): Inhibition, drawn
```

"If exported from this payload" is literal, and the limit is real — see § 8 and open question 1.

---

## 6. The cumulative re-pin (orchestrator Finding 3, merge rule 4)

Script `evidence/orch754b_cumulative_repin.py` → `orch754b_cumulative_repin.json`
(G11 `ORCH-754b/18`). Base **`c9265fb8`** (where both pins last passed) vs tip, **all 89 pinned
legs × all five configurations**.

| pin | old → new | legs moved |
|---|---|---|
| `GOLDEN` (`test_build_pwml_ir_matches_the_pre_extraction_golden`) | per leg, listed in `golden_old_to_new` | **73 of 89** |
| `R3_BASE_IR_DIGEST` | `32bf7893e62f…` → `14af62c13c45…` | the R3 leg |
| `R3_BASE_REPORT_DIGEST` | `026c8a8e6b69…` → `082729a8b749…` | the R3 leg |

Every moved pointer, by number of legs:
`/ir/processes/interactions/[]/{export_status 73, withheld 73, left/key 72, right/key 72,
interaction_type 58}`, `/ir/process_visualizations 7`, `/ir/edges 4`, `/ir/locations 2` (+ `x`/`y`
on the same 2 legs — an endpoint the orphan grid used to place is now placed beside its drawn
interaction: `runs_verify/2026-08-04_1754/…/PMC12856317/strict`,
`runs_verify/2026-08-21_2239/…/PMC12856317/research`), `/ir/protein_complex_visualizations 1`,
`/report/warnings 73`, `/report/ir_validation/counts/process_members 71` and the matching
`counts` for edges / locations / visualizations.

**Outside the interaction prefixes: none. Biological IR moved (entities, species, states,
reactions, transports, non-interaction visualizations): none. Errors, `ok`, stops moved: none.**

The `type="integer"` side effect ORCH-747 recorded is in the writer's XML, not the IR, so it
cannot reach these digests.

The R3 report delta is one added `interaction_withheld_from_pwml` warning
(`DHNA inhibits MenD`, `bare_protein_endpoint`) plus the `process_members` count; the test now
enumerates it before checking the digest. This card's own share of the move, from `f9892cdf`:
`orch754b_ir_digest_delta.json` (G11 `ORCH-754b/21`) — 73 legs, `withheld/evidence` on 73,
`withheld/reason` + warning text on 15, `unwrappable_proteins` on 14, `interaction_type` on 1
(the `competes_with` row), nothing else.

**New values are pytest's own.** `evidence/orch754b_pins_before_repin.pytest.log` is the failing
run (G11 `ORCH-754b/16`); the script parses it and checks: pytest reported 73 moved legs, the
instrument reports the same 73, every pytest `actual` equals the instrument's tip digest, and
every pytest `expected` equals the committed pin. After the edit both tests pass
(G11 `ORCH-754b/20`).

---

## 7. `competes with` (orchestrator, non-blocking)

Removed from `PWML_INTERACTION_TYPES`. Competition is symmetric, so drawing `A competes_with B`
as A ⊣ B was the exporter choosing a direction after the freeze (merge rule 8), and D-104 § 4
forbids mapping an unsigned word onto a sign. The rows are withheld as `untyped_interaction`
with their word kept, and appear in the text output with it.

**CORRECTION, G9:** `test_competes_with_is_withheld_as_untyped_not_drawn_as_a_directed_inhibition`.
Base failure measured: input `correction:competes_with` in `orch754b_wrapper_base_proof.json`
— base `{'exportable': 1}` (drawn), tip `{'untyped_interaction': 1}`, PWML `<interaction>` 1 → 0.

**The rest of the table, reviewed:** `activation`, `inhibition`, `repression` are nouns whose
direction comes from the row's entity_1 → entity_2 order, as every row's does. `activates`,
`stimulates`, `induces`, `upregulates`/`up-regulates`, `enhances`, `promotes`, `inhibits`,
`represses`, `blocks`, `suppresses`, `downregulates`/`down-regulates`, `prevents` are directed
verbs. **`antagonizes` is the one ambiguous entry**: as a verb it is directed (X antagonizes Y:
X opposes Y's action), but the word is also used for mutual antagonism ("A and B antagonize
each other"). Kept, because the verb form names a subject and an object; 0 committed rows use
it. Open question 4.

---

## 8. App path: withheld interactions in the text output (orchestrator)

**Partly possible without `streamlit_app.py`; the rest is not.**

What now holds on the app path:

- the **post-audit reaction summary** — the last one the app stores in
  `session_state["reaction_summary"]` and offers as `reaction_summary.txt` — is built
  (`streamlit_app.py:4219-4221`) from the payload *after* the first `map_payload` pass
  (`:3487`), so its interaction rows carry the word, the evidence and the exporter's verdict
  on a payload that already has Stage 6's wrappers;
- the IR the app persists (`final.pwml_ir.json`, `streamlit_app.py:3039`; the batch driver's
  `pwml_ir.json`, `driver.py:2234-2236`) carries every withheld row's `reason`,
  `raw_interaction_type`, endpoints, pointer and — new — `evidence`. That is the exact record,
  but it is JSON.

What does **not** hold: the verdict after the **second** mapping pass (`streamlit_app.py:4248`)
and the IR is never rendered into a text report on the app path. Both text reports are
generated before `_export…` builds the IR (`streamlit_app.py:4960-5005`), and that function
discards `build_result.withheld_interactions` and writes no side file. Closing that needs a
seam in `streamlit_app.py`; per the brief and the D-xxx rule for that file, I stopped there.
Options in open question 1.

---

## 9. Gates

| gate | result | report |
|---|---|---|
| focused (new file + interaction export + spellings) | **99 passed** | `g11/ORCH-754b/22-focused.json` |
| affected: mapping / wrapper / identity / text-report tests (30 files) | 647 passed, 3 skipped, **3 failed — identical E-lines at base** (`c075` corpus pass, `c100` 5b tolerance, `rag_seed_entity_reimport` conform) | tip `23`, base `24` |
| PWML export chunk (ORCH-749's 33-file command) | tip **848 passed, 8 skipped, 2 failed**; base (clean worktree at `f9892cdf`) 847 / 8 / **4 failed**. The two pins went red → green; the 2 left are the coverage tripwire (`test_the_golden_covers_every_committed_leg_fixture`, `test_the_repaired_coverage_and_digest_pins_themselves_go_red`), red at base too | tip `25`, base `26` |
| SMOKE (22 files) | **508 passed** | `27-smoke.json` |
| AppTest `test_c055_rag_loop_wiring` alone | 14 passed | `28` |
| AppTest `test_batch_preflight` alone | 32 passed, 4 skipped, 1 failed — identical at base (`this project ships a .venv`: a worktree has none) | tip `29`, base `31` |
| AppTest `test_streamlit_quarantine_boundary`, per node (Chunk D `--only qb`) | see § 9a | `32` outer + node reports |
| pinned digests before / after re-pin | 2 failed / **2 passed** | `16`, `20` |
| measurements | proof `14`, cumulative re-pin `18`, this-card delta `21`, R3 warnings `19`, complex-row diagnostic `05`, unwrappable endpoints `60` | |

Every report: `final_surviving_count: 0`; `g11_evidence.py check --task ORCH-754b`: 58
artifacts, 0 non-compliant. `30`, `61` and `62` were refused before starting (the shared heavy
lock was held by `ORCH-754` / `ORCH-754a`; nothing ran, nothing to clean, no report
promoted). `15-x` is an accidental no-op run of `true`; kept, not hidden. Earlier `dev-*` and
`wrapper-proof` runs are superseded iterations of the same commands. Runs `05`, `19` and `60`
executed scratch copies under `C:/t/bt-orch754b/`; the committed
`evidence/orch754b_diag_complex_change.py`, `orch754b_r3_warnings.py` and
`orch754b_unwrappable_endpoints.py` are byte-identical copies (sha256 checked for the last).
Chunk D's node runs call pytest directly, by that gate's design, so they carry no pin verdict.

**ORCH-747's "7 known failures" is not reproducible in a worktree**: measured here the export
chunk's base is 4 failures, not 7 — the 3 DB-reachability tests pass when no `.env` is present.
The baseline that matters is the measured one above.

### 9a. Chunk D quarantine boundary

`chunk_d_gate.py run --only qb`: **23/23 nodes executed, each alone in a fresh process, 23
passed, omissions 0, additions 0, failed none**, 103 s, zero survivors. Outer report `32`,
gate reports `33`–`59` (`36-qb.json` the component, `37`…`59` the nodes).

---

## 10. TEST_MATRIX delta (for the orchestrator to apply)

- New file `tests/test_pwml_interaction_protein_wrappers.py`, 17 tests, not in SMOKE. Belongs
  with the interaction export tests in the PWML/export chunk.
- `tests/test_pwml_interaction_type_spellings.py`: net −1 (INHIBITION −2, UNSIGNED +2, one
  variant row removed).
- PWML/export chunk, measured in a worktree: base 4 known failures → tip 2 (the coverage
  tripwire).
- The two digest pins moved deliberately (§ 6). SMOKE unchanged at 508.

---

## 11. What I believed and what disproved it

1. **"The IR and writer need new code to draw a complex endpoint."** No: ORCH-747 already
   resolves, places and emits protein-complex endpoints in PathBank's shape. The only missing
   piece was the wrapper itself, at Stage 6.
2. **"An offline replay of the wrapper step leaves pre-existing complexes alone."** Disproved by
   my own confinement check (G11 `04`): on 16 legs the replay re-merged an offline resolution
   into an existing enzyme wrapper's `mapping_meta` (diagnostic `05`; every one was the
   protein's own one-protein wrapper, `MenD`, `ALAS2`, `SFXN4`, `NDM-1`, `ferrochelatase`).
   Only provenance changed, never components, but it was a write to a row the card had no
   business touching, and the same path would have merged into a *declared* complex that
   happened to carry the wrapper's name. Fixed: own wrappers are reused as they stand, other
   complexes are never borrowed; tests for both.
3. **"The text reports can state the final PWML verdict."** Only partly: on the app path both
   are generated before the export (§ 8).
4. **ORCH-747 F-747-3 ("both text outputs carry per-row evidence").** Not true of
   `reaction_summary.py`; fixed here.
5. **F-047 — the out-of-pytest digest divergence recorded as "cause UNKNOWN".** With the test
   module's autouse `_no_live_db` monkeypatch applied by hand, the out-of-pytest instrument
   reproduced **all 89** committed `GOLDEN` digests at `c9265fb8` and **all 73** pytest digests
   at the tip. That is consistent with configuration E (`db_resolver=None` → `from_env`) being
   the divergence an un-patched harness sees. It is a correlation over one corpus, not a proven
   mechanism.
6. **Preview text first.** My first text line *described the rule* ("Inhibition if Stage 6
   wrapped …"). The orchestrator's app-path requirement showed that was weaker than computing
   the exporter's verdict on the payload in hand; replaced by the shared verdict function.

---

## 12. Not settled — for the product owner

1. **The final verdict in the app path's text output.** Options:
   (a) a `streamlit_app.py` seam after `builder.build()` that writes
   `outputs/pathway.withheld_interactions.json` and regenerates `reaction_summary.txt` from the
   frozen payload — needs a D-xxx for the protected file and a Chunk D re-run (~10 min);
   (b) the batch driver renders the persisted `pwml_ir.json` into a text file — covers batch
   legs only, not the interactive app, and `driver.py` is outside this card;
   (c) accept today's state: the post-audit summary is exact for the first-mapped payload, and
   `pwml_ir.json` is the exact final record in JSON.
2. **25 signed rows withheld for a missing UniProt/DrugBank identity** (replay estimate;
   `evidence/orch754b_unwrappable_endpoints.py` → `.txt`, 28 endpoint occurrences): `Fur` 8,
   `RyhB` 6, `EntB` 5, `YejM` 4, and `ALAS2`, `Fnr`, `EntF`, `EntD`, `MenD` once each.
   **`RyhB` is a small regulatory RNA typed as a protein** — wrapping it would have been
   wrong, and it is correctly withheld, but the mistyping is upstream of this card (Stage 1/2
   entity typing) and should go to whoever owns it. For the rest, options: improve identity
   grounding (cost: a mapping card, network-backed), or accept them withheld (they are in the
   text output with the reason).
3. **DB-backed wrappers for interaction endpoints.** With the database up, a protein endpoint
   can resolve to a PathBank complex all of whose components are that protein (e.g. a
   homodimer), exactly as an enzyme does. Options: accept (consistent with enzymes, PathBank-
   grounded), or force one-protein wrappers for interaction endpoints (cost: a second
   behaviour for one mechanism, which the ruling forbids).
4. **`antagonizes`** stays signed (§ 7). Remove it too if mutual antagonism is the reading the
   PO wants to guard against; 0 committed rows today.
5. **The reaction summary is also LLM context** (`gap_resolver.py:2727-2728` passes it to the
   enrichment agent). The new `Evidence:` / `PWML:` lines lengthen that prompt. Options: keep,
   or strip them for the LLM copy (cost: a second rendering).
6. **Coverage tripwire red at base and tip**: 23 committed `runs_verify` legs
   (`2026-08-27_1341`, `2026-08-28_1816`, `2026-09-01_1612`) are in neither `GOLDEN` nor
   `EXCLUDED`. Needs a C-093-style admission sweep; not this card's.
7. **Historical evidence now disagrees with the tip**: `evidence/orch749_verb_form_export_base_proof.py`
   expects `"competes with".` → Inhibition. Left as ORCH-749's record, not edited.
8. **Wrapper naming.** Ours are `"<protein> complex"`; PathBank names its own single-protein
   complexes after the protein (`Ephrin-A1`). Existing enzyme convention, unchanged.

---

## 13. Left behind

A detached worktree at `C:/t/bt-orch754b/base-wt` (`f9892cdf`), created for the base-SHA
baseline runs (`24`, `26`, `31`), is still registered: `git worktree remove` was refused by
the permission system in this session. Remove it with
`git worktree remove --force C:/t/bt-orch754b/base-wt`.
