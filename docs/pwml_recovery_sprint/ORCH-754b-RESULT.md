# ORCH-754b — protein endpoints of interactions reach the PWML as single-protein complexes

**Status:** implemented on `orch754/b-interaction-wrappers` (base `f9892cdf`), committed, not
merged and not pushed. Round 0: code `3bf40cb6`, docs `f8b8edf7` / `d73e9bc8` — **REJECTED** by
the independent review of `d73e9bc8`. Repair round 1 (§ 14): code `ff5f2337`, docs `9da8358e` —
**REJECTED** on two small blocking items. **Repair round 2** (§ 15): code `1f617891`, then the
docs commit carrying this document. Where a round-0 section below is superseded it says so at its
head; read § 14 for the current state. Product-owner rulings applied: 2026-10-19, being recorded as **D-104**
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
| `src/t2pw/pwml/ir.py` | `interaction_pwml_verdict` — the rule for what the PWML does with an interaction, used by the IR loop. (Round 0 also had the text reports call it, on their own name lookup — that was the defect § 14.1 removes.) A bare protein endpoint that lacks a species or a UniProt/DrugBank identity is withheld with a precise reason; `bare_protein_endpoint` stays the fallback when no wrapper exists. The `withheld` record now carries the row's `evidence`. `competes with` / `competes_with` removed from the sign table. |
| `src/t2pw/pipeline/draft_graph.py`, `src/t2pw/pipeline/reaction_summary.py` | **round 0 only — reverted in repair round 1**; both are byte-identical to `f9892cdf` at the tip (§ 14.2). |
| `src/t2pw/rag/research_report.py` | **round 0's per-row fields reverted** (§ 14.2); the tip adds one human-only sentence and nothing to the JSON. |
| `src/t2pw/app/reaction_evidence.py`, `reaction_evidence_view.py` | **repair round 1**: the human-facing interaction record, verdict read from the IR (§ 14.1). |
| `tests/test_pwml_interaction_protein_wrappers.py` | **new**; 17 tests in round 0, 21 at the tip. |
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

> **Superseded by § 14.1–14.2.** Round 0 put the record in `reaction_summary.txt`, which two
> live LLMs read, and computed its verdict with a name lookup the exporter does not use. The
> claim below that the verdict was "computed by the same `interaction_pwml_verdict` the IR
> uses" was true of the function and false of its inputs: 2 of 336 committed rows came out
> "drawn" in the text and withheld in the IR. Kept as the record of what was shipped and
> rejected.

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
on the same 2 legs). **Corrected in repair round 1** (round 0 implied one location per leg;
measured by `evidence/orch754b_location_count.py`, G11 `ORCH-754b/72`, configuration B):
on `runs_verify/2026-08-04_1754/…/PMC12856317/strict` **3** of 9 locations moved — `heme`
(240, 1050) → (1315, 489), now placed beside its drawn interaction, and the two orphan-grid
compounds after it each shifted one slot, `pyridoxal 5'-phosphate` (440, 1050) → (240, 1050)
and `coenzyme A` (640, 1050) → (440, 1050); on `runs_verify/2026-08-21_2239/…/PMC12856317/research`
**2** of 7 moved (`heme`, `Pyridoxal 5'-phosphate`). No location was added or removed, and only
compound geometry moved. `/ir/protein_complex_visualizations 1`,
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

> **Superseded by § 14.1.** The app path now has a human-facing text record carrying the final
> verdict: the Reaction Evidence section and its `.txt` download read the session's own export
> IR. The analysis below of what the round-0 summary could and could not say stands as history.

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

1. **Resolved in repair round 1 (§ 14.1):** the Reaction Evidence report reads the session's export IR, no `streamlit_app.py` seam needed. Round-0 text: **The final verdict in the app path's text output.** Options:
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
5. **Resolved by the 2026-10-20 ruling (§ 14.2):** the summary is the base's again. Round-0 text: **The reaction summary is also LLM context** (`gap_resolver.py:2727-2728` passes it to the
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

Five detached worktrees I created are left registered, because `git worktree remove` was refused by
the permission system in round 0: `C:/t/bt-orch754b/base-wt` (`f9892cdf`, the base-SHA baseline runs),
`prev-wt` (`d73e9bc8`, with the round-1 test file copied in for G9, so it is dirty) and `tip-wt`
(`ff5f2337`, for the c081 / c089 corpus tests), plus, from round 2, `r1-wt` (`9da8358e`, with the
round-2 test files and fixtures copied in for G9, dirty) and `tip2-wt` (`1f617891`). Remove each with
`git worktree remove --force C:/t/bt-orch754b/<name>`.

---

## 14. Repair round 1 (review of `d73e9bc8`: REJECT)

What the reviewer confirmed and this round does not touch: the biology gates, wrappers built
only at Stage 6, the re-pin (R3 reproduced exactly), the match to PathBank 15843, G9 at base,
SMOKE 508, Chunk D 23/23. Code commit **`ff5f2337`**.

### 14.1 FIX 1 (blocking) — the text record could contradict the PWML

**Finding, confirmed.** Round 0's text verdict was computed beside the exporter. Its endpoint
kinds came from `draft_graph`'s exact-name lookup (complex wins a tie) and `research_report`'s
lowercased name, while the IR resolves through normalized names **and synonyms**, protein
first. On the committed corpus, 2 of 336 rows disagreed: `runs_verify/2026-08-21_2239/papers/PMC12452463/research`,
endpoints `entC` / `entF` (synonyms of `EntC` / `EntF`): the text said "Inhibition, drawn", the
IR withheld them. The same class appeared after Stage 6 (reviewer probe `08-synonym`): an
endpoint named by a synonym of a protein with UniProt and species was not wrapped, because
Stage 6 looked names up only.

**My round-0 claim was wrong.** I wrote that the reports and the IR used "the same"
`interaction_pwml_verdict`, and (in a docstring) that "the two cannot disagree". The function
was shared; the resolution feeding it was not. That claim is withdrawn.

**What changed.**
- **No text output computes a verdict any more.** The human-facing record is the Reaction
  Evidence report (`src/t2pw/app/reaction_evidence.py`, the deterministic `.txt` export of
  ORCH-742b/744). It gains an `INTERACTIONS` section — for every interaction in
  `final_mapped.json`: the relationship word as extracted, endpoints, state, every evidence
  quote, and the PWML verdict **read from the IR the PWML was written from**
  (`pwml_ir.json`, which `batch.driver._add_pwml_artifacts` writes into each strict leg;
  in the app, the session's own `pwml_export_result["pwml_ir"]`). It is the IR's own
  resolution because it is the IR's own output. The rows are lined up one-to-one in order and
  checked by name; any mismatch, a missing IR, or an IR older than per-row verdicts reads
  `Not recorded (…)` with the cause — never a guess. The module still imports only the
  standard library (its own test 11 holds).
- **Stage 6 now resolves synonyms too**, so the post-Stage-6 case exports instead of being
  withheld. An endpoint not found by name is matched against each protein's `name`,
  `raw_name`, `short_name`, `common_name` and `synonyms` — the names the IR registers — but
  only **exactly** (case and spacing aside, narrower than the IR's normalization) and only
  when **one** protein carries it, so Stage 6 never picks a protein the IR would not. The
  wrapper is named after, and holds, the declared protein (`EntC complex` with component
  `EntC`, not the alias). No gate moved: the same species / identity / sentinel / free-name
  conditions apply.

**Can they still disagree?** The verdict shown is the verdict the IR recorded, so the report
cannot disagree with the IR it was given. What remains possible, and is stated rather than
hidden: a run directory whose `pwml_ir.json` and `final_mapped.json` come from different
payloads — the name check then prints `Not recorded`, not a verdict.

**Evidence.**
- Corpus, `evidence/orch754b_text_vs_ir_corpus.py` → `.json` (G11 `71`): 95 legs × {direct,
  replay}; the IR builds on 94 (one leg the exporter refuses: `DuplicateNamedRowError`, so no
  verdict exists to show). **330 of 330 rows in each configuration: the record states the
  IR's verdict. 0 disagreements. The LLM-bound summary states a PWML verdict on 0 legs.**
- Tests, each **failing behaviourally at `d73e9bc8`** (G11 `69`, the new test file run in a
  worktree at `d73e9bc8`: 11 failed, 10 passed, every failure an `AssertionError` or a missing
  block, none an import error) and passing at the tip:
  `test_committed_synonym_endpoints_no_text_output_contradicts_the_ir` (the committed leg),
  `test_a_protein_named_by_its_synonym_is_wrapped_and_the_record_says_drawn` (the reviewer's
  probe), plus `test_an_alias_two_proteins_share_is_not_guessed_and_the_record_still_states_the_ir`.

### 14.2 FIX 2 (product-owner ruling 2026-10-20) — the record goes to human-facing outputs only

**Finding, confirmed.** `reaction_summary.txt` is read by the curator (`streamlit_app.py:4232-4236`
→ `pathway_curator.py:299-300`) and the enrichment agent (`gap_resolver.py:2727-2728`); round 0
grew it ~32% with lines that read like instructions.

**What changed.**
- `src/t2pw/pipeline/reaction_summary.py` and `src/t2pw/pipeline/draft_graph.py`: **restored to
  `f9892cdf` byte for byte** (`git diff f9892cdf -- …` is empty; `qa_graph.py` was never touched).
- `src/t2pw/rag/research_report.py`: round 0's per-row JSON fields and lines removed; the JSON
  row keys are the base's again. One human-only sentence is added to the REGULATION header
  (research mode writes no PWML, so no row reached one).
- `map_ids`: Stage 6 **no longer writes anything for a row it leaves alone** (round 0 wrote an
  `interaction_protein_endpoint_wrapper_skipped_invalid_component` action with a `missing` key
  and a counter). The Stage-6 report reaches the **interactive curator's** prompt through
  `interactive_curator.compact_mapping_misses` (`interactive_curator.py:187-228`, used at
  `:270` and `:599`), whose `missing` token would have picked that action up. The exporter's
  own withheld record still names the reason.

**Every reader of every file this card writes (grep over `src/`).**

| output | readers | reaches an LLM? |
|---|---|---|
| `reaction_summary.txt` / the summary string | curator, enrichment agent, UI | **yes** — now byte-identical to base |
| `draft_graph.json`, `qa_report.json` | UI, curator inputs | **yes** — byte-identical to base |
| Stage-6 mapping report | interactive curator (`compact_mapping_misses`), logs | **yes** — byte-identical to base for every payload Stage 6 does not wrap; see open question 9 |
| `reaction_evidence.txt` (`app/reaction_evidence.py`) | only `reaction_evidence_view.py` (download, page) | no |
| `research_pathway_report.txt` / `.json` / `.csv` | app download; batch writes it; `batch/runner.py:124`, `bench/acceptance.py:126`, `batch/driver.py:3110,3125` test only that it **exists** | no |
| `pwml_ir.json`, `pwml_ir_report.json`, `pathway.withheld_interactions.json` | built only at the final export (`streamlit_app.py:4960`, `writer.py:2900`); read by downloads, the batch artifact writer, the deterministic contract audit (`streamlit_app.py:6559-6569`) | no |

`batch.driver._collect_app_text` (`driver.py:1725-1732`) collects the page's errors, warnings
and exceptions for failure classification — deterministic, not a model — and the Reaction
Evidence section emits none of them (its own AppTest asserts it).

**Evidence.**
- `evidence/orch754b_llm_input_identity.py` → `.json` (G11 `65`): `_llm_bound_digests()` —
  reaction summary, draft graph and QA report for 4 payloads (the fixture, the synonym probe,
  the committed synonym leg, the committed lipid-A research leg with 16 interactions) and the
  Stage-6 report + payload for an all-unwrappable payload — run under three trees, each child
  verifying every `t2pw` module it loaded: **tip == base on every digest; `d73e9bc8` differed
  on 9** (summary and draft graph for all 4 payloads, and the Stage-6 record).
- `test_every_llm_bound_input_is_byte_identical_to_the_base` pins those base digests; it fails
  at `d73e9bc8` (G11 `69`) and passes at the tip.
- The human-facing output carries the full record:
  `test_the_human_facing_record_carries_word_evidence_and_the_ir_verdict_for_every_row`,
  `test_session_run_directory_reads_the_verdicts_from_the_sessions_own_export_ir` (AppTest),
  and the regenerated `docs/reaction_evidence_sample.txt` (`evidence/orch754b_regen_reaction_evidence_sample.py`,
  G11 `67`, diff in `orch754b_reaction_evidence_sample.diff`).

### 14.3 FIX 3 — the `draft_graph.py` edit

Granted, and removed anyway: FIX 2 made it unnecessary. `draft_graph.py` is byte-identical to
base, so `draft_graph.json` carries no `interaction_records`; nothing to confirm about its readers.

### 14.4 Non-blocking — reason text true for every reason

Round 0's line for `bare_protein_endpoint` said "Stage 6 builds one" even where Stage 6 had
refused (name held by another complex, the Unknown sentinel). That function is gone; the
gloss now lives in `reaction_evidence._WITHHELD_GLOSS` and says only that no wrapper exists
in the payload. The IR docstring was corrected the same way.
`test_every_reason_is_glossed_truthfully_including_a_refused_wrapper` (fails at `d73e9bc8`).

### 14.5 Deliberate pinned moves this round (merge rule 4)

- `docs/reaction_evidence_sample.txt`: header line reworded ("interactions itemised: 3"),
  `pwml_ir.json` added to "Artifacts read", and the new INTERACTIONS section (3 rows; the
  committed leg's IR predates per-row verdicts, so each reads `Not recorded (this IR predates
  per-row export verdicts)`). Nothing else in the sample moved.
- The four `tests/fixtures/reaction_evidence/*.bundle.json`: each trimmed interaction row gains
  back the fields the report now renders, and the two bundles whose legs carry a
  `pwml_ir.json` gain that IR's interaction rows — all copied verbatim from the committed legs,
  so `test_trimmed_bundle_renders_like_its_committed_leg` still proves the trim.
- `tests/test_reaction_evidence_view.py`: two expected expander lists gain the Interactions
  expander.
- The digest pins of § 6 did **not** move again: the IR output is unchanged this round (re-run, G11 `118`).

### 14.6 Gates (round 1)

| gate | tip (`ff5f2337` tree) | base `f9892cdf` (clean worktree) | report |
|---|---|---|---|
| focused (new file + interaction export + spellings) | **103 passed** | — | `74` |
| affected (32 files: mapping, identity, text reports, reaction evidence) | 667 passed, 3 skipped, 3 failed | 670 passed, 3 failed — **identical E-lines** | tip `75`, base `76` |
| the 3 skips above: c081 / c089 corpus tests, run from paths without `worktrees` | **56 passed** (`C:/t/bt-orch754b/tip-wt` @ `ff5f2337`) | 56 passed (`base-wt`) | `119`, `120` |
| PWML export chunk (ORCH-749's 33 files) | 848 passed, 8 skipped, 2 failed (coverage tripwire) | 847 / 8 / 4 failed (round-0 measurement; base unchanged) | `78`, base `26` |
| SMOKE (22 files) | **508 passed** | — | `83` |
| AppTest `test_reaction_evidence_view` alone | **10 passed** (9 + the new session-IR test) | 9 passed | `87`, base `88` |
| AppTest `test_c055_rag_loop_wiring` alone | 14 passed | — | `85` |
| AppTest `test_batch_preflight` alone | 32 passed, 4 skipped, 1 failed — the same `.venv` assumption as base | (round 0: identical) | `89`, base `31` |
| Chunk D `--only qb`, one process per node | **23/23 executed, failed none** | — | outer `90`, gate `91`–`117` |
| digest pins (§ 6) | 2 passed | — | `118` |
| G9: the new test file at `d73e9bc8` | 11 failed (behavioural), 10 passed | — | `69` |
| measurements | LLM-input identity `65`, mapping-miss probe `66`, sample regeneration `67`, wrapper proof `70`, text-vs-IR corpus `71`, location count `72`, unwrappable endpoints `73` | | |

Every report: `final_surviving_count: 0`. Runs `63`, `77`, `79`, `80`, `81`, `82` were refused
before starting (the shared heavy lock was held by ORCH-748's mining pilot or ORCH-754a);
nothing ran and no report was promoted. `64` is superseded by `65` (the first run's summary
did not report a non-dict digest that differed). `66` executed a scratch copy of
`evidence/orch754b_probe_mapping_misses.py`. Base and previous-tip measurements ran in clean
detached worktrees `C:/t/bt-orch754b/base-wt` (`f9892cdf`) and `prev-wt` (`d73e9bc8`, with the
new test file copied in for `69`), each pin verdict naming that tree's `t2pw`.

**TEST_MATRIX delta (round 1):** `test_pwml_interaction_protein_wrappers.py` 17 → 21 tests;
`test_reaction_evidence_view.py` 9 → 10; `test_reaction_evidence.py` count unchanged (its
fixtures and sample moved, § 14.5); the export chunk still at 2 known failures; SMOKE 508.
The c081 / c089 corpus tests skip under any path containing `worktrees`, so a worktree's
affected-test run reports 3 skips that are not failures; run them from a path without it.

### 14.7 What I believed and what disproved it (round 1)

1. **"Sharing the verdict function makes the outputs agree."** Disproved by the review: the
   function's inputs — which entity an endpoint is — came from two different resolvers.
   Agreement needs the same *resolution*, so the record now reads the IR's output.
2. **"Stage 6 names are enough."** Disproved by the reviewer's probe: the IR also resolves
   synonyms. Stage 6 now does, narrowly.
3. **"A per-row skip record in the Stage-6 report is harmless provenance."** It feeds an LLM
   prompt through a heuristic that keys on `missing`. Removed.

### 14.8 Not settled — added this round

9. **D-104's wrappers themselves change later LLM inputs.** Byte identity holds for every
   payload Stage 6 does not wrap, and every text generator is the base's. But when Stage 6
   does wrap an endpoint, the payload changes by design (`ALAS2` → `ALAS2 complex` in the
   interaction, a new generated complex row), and so do the inputs of every model that reads
   the mapped payload afterwards: the audit-loop curator (post first `map_payload` pass), the
   interactive curator (the payload, and the new complex's `novel` mapping log under
   "Mapping Misses"), and the post-audit reaction summary. That is the representation change
   D-104 rule 1 ordered, not the text record, but the 2026-10-20 ruling's wording ("every
   input that reaches any LLM") could be read to cover it. Options: (a) confirm it is in scope
   of D-104 rule 1 (no cost); (b) move the interaction wrapper to the second `map_payload` pass
   only, after the audit loop (cost: a change in the call path ORCH-754a shares, and the
   interactive curator would still see it); (c) strip interaction wrappers from the curator's
   view of the payload (cost: the curator edits a payload it cannot fully see).
10. **No reaction-evidence record for a strict leg until it is re-run.** Committed strict legs'
    `pwml_ir.json` predate per-row verdicts (ORCH-747), so their reports read `Not recorded`.
    Only legs exported after this card carry verdicts.

---

## 15. Repair round 2 (review of `9da8358e`: REJECT, two blocking items)

What the reviewer confirmed and this round leaves alone: synonym matching requires exactly one
protein, no declared complex is borrowed, no gate moved, the bundles are verbatim, the gates
match base. Code commit **`1f617891`**.

### 15.1 BLOCKING 1 — a lane-added Stage-6 report key still reached an LLM

**Finding, reproduced.** Round 1 kept one piece of bookkeeping: the summary counter
`interaction_protein_endpoints_rewritten_to_complexes` and one `…_rewritten_to_complex` action
per endpoint. `interactive_curator.compact_mapping_misses` copies the Stage-6 step's summary dict
whole into the curator's "Mapping Misses": `_looks_like_entity_mapping_record` accepts any dict
with a key ending `_id` (`interactive_curator.py:1008-1027`) — the step summary has
`reaction_enzyme_proteins_dropped_no_external_id` — and a non-zero `…_novel` count trips the
miss test (`:973-1006`). My round-1 pin missed it because `stage6_all_unwrappable` covered only
a payload where nothing is wrapped, and my own probe used a summary without that `_id` key.

**What changed.** The counter and the actions are gone. A superset refusal met while wrapping
an interaction endpoint is also no longer written (`_resolve_complex_name(…, record=False)`;
the enzyme path still records its own). No new place holds them. The rewrite stays visible
where it happens: the interaction endpoint names a wrapper row with `generated: true` and
`generation_reason: single_protein_pathwhiz_wrapper`, and the IR's verdict covers the rest.

**Pinned WRAPPED case.** `_llm_bound_digests()` gains three entries. `stage6_wrapped_report`
and `stage6_wrapped_mapping_misses` are the Stage-6 report and the curator's Mapping Misses for
a payload Stage 6 does wrap (SFXN4 and ALAS2), each tree's own Stage 6 on the same input.
`wrapped_post_stage6` is the reaction summary, draft graph and QA report of the post-Stage-6
payload. The evidence script runs the tip first and hands that payload to the base, so both
trees read the same payload and the wrapper is the only difference.
`evidence/orch754b_llm_input_identity.py` → `.json` (G11 `122`): **tip == base on every digest,
including all three**; `d73e9bc8` differs on the three Stage-6 entries.

**95-leg replay.** `evidence/orch754b_mapping_misses_replay.py` → `.json` (G11 `124`) re-runs
the step offline on every committed leg with interactions and rebuilds the two places a
`map_payload` report carries it (`report["summary"]` and `report["enzyme_complex_conversion"]`).
Mapping Misses that differ from base: **`9da8358e`: 24 legs (the reviewer's number); tip: 0.**
The tip rewrote 71 endpoints on 56 legs in that replay, so the 0 means "nothing but the
wrapper", not "nothing was wrapped".

**One base counter, stated, not changed:** `map_payload`'s own summary key
`generated_wrappers_created` (`map_ids.py`, `map_payload`; base code) counts every generated
complex a pass creates, the interaction wrappers included, so its value rises when the
authorized wrapper is built. That is base bookkeeping counting the authorized change, and
outside the step this lane owns. Open question 11.

### 15.2 BLOCKING 2 — the `.txt` said "drawn" when no PWML was written

**Finding, reproduced.** On an IR-validation failure the app returns `ok: False`,
`output_path: ""` and still carries `pwml_ir` (`streamlit_app.py:4971-4985`). The batch driver
writes `pwml_ir.json` even without XML (`driver.py:2232-2236`). Round 1 read `export_status`
alone. `runs_verify/2026-08-25_1216/papers/PMC12856317/strict` has an IR and no PWML.

**What changed.** A verdict is shown only when a PWML was written. A run directory must hold
`pathway.pwml` or `pathway.review_required.pwml`; `load_run_artifacts` records their presence and
never reads them. In the app, `pwml_export_result` must be `ok: true` with PWML bytes or a path.
Otherwise every row reads `Not recorded (no PWML written: <cause>)`, where the cause is the
export's own `error` (e.g. `PWML IR validation failed.`) or "neither pathway.pwml nor
pathway.review_required.pwml is in this run".

**Tests**, each failing behaviourally at `9da8358e` (G11 `135`, `136`; every failure an
assertion, none an import):
- batch path — `test_no_verdict_is_shown_when_no_pwml_was_written_batch_path`;
- app path, AppTest — `test_an_export_that_wrote_no_pwml_shows_no_verdict_app_path`.

### 15.3 Non-blocking 3 — a stale IR row paired with a different word

**What changed.** Each IR row must have been built from the payload row beside it:
- for a withheld row, the IR's recorded word and both endpoint names must equal the payload's,
  read with the exporter's own field precedence;
- for a drawn row, the payload word's sign (the same table) must equal the IR's
  `interaction_type`, and the entity each side was bound to must carry the written name. That
  check uses the IR's own name key, applied to the entity's name, raw name, short name, common
  name and synonyms.

Any mismatch prints `Not recorded (pwml_ir.json row n was built from a different relationship
or endpoints)`. A drawn row whose endpoint the IR never resolved now says so, instead of "drawn".

The sign table moved, verbatim, to `src/t2pw/pwml/interaction_types.py` (standard library only;
`ir.py` re-exports every name), so the report can use it without importing the exporter. The
report's no-LLM / no-HTTP import test (`test_reaction_evidence.py` test 11) still passes.
Test: `test_a_stale_ir_row_is_never_paired_with_a_different_word_or_endpoint` (fails at `9da8358e`).

### 15.4 Non-blocking 4 — fail closed on a protein name shared with another kind

**What changed.** Stage 6 wraps no endpoint of a row if a protein endpoint's name, or the alias
it matched, also names a compound, nucleic acid or element collection in the payload. I added
element collection to the reviewer's two kinds, because the same ambiguity applies. The check
uses Stage 6's own normalization, which is looser than the IR's, so it refuses at least
everything the IR would call ambiguous.

The IR withholds such a row as `ambiguous_protein_endpoint`, with
`ambiguous_endpoints: [{name, also_names}]`, before any species / identity reason. The record
reads e.g. `withheld -- ambiguous_protein_endpoint: entD also names a nucleic acid in this
payload, so it is not known to be the protein and is not drawn as one`.

**Cost, measured:** 0 of 71 replay rewrites affected (G11 `124`, rewrite counts `9da8358e` 71 =
tip 71). On committed data the IR now calls **2 rows** ambiguous, both on
`runs_verify/2026-08-04_1358/papers/PMC12096016/research`: `2,3-DHB inhibits EntB isochorismatase`
and `Fur represses entD`. In both, `EntB` / `entD` is a protein and a gene in that payload, and
`Fur represses entD` is gene repression (`evidence/orch754b_ambiguous_endpoint_rows.py`, G11 `133`).
Test: `test_a_protein_endpoint_also_named_like_another_kind_is_never_wrapped_and_both_outputs_say_why`,
name and alias cases.

### 15.5 Deliberate pinned moves (merge rule 4)

- **One GOLDEN digest:** `runs_verify/2026-08-04_1358/papers/PMC12096016/research` `e0ef63b1…` →
  `d8bd9a2c…`, the two ambiguous rows above. The delta was measured from `9da8358e` by
  `evidence/orch754b_ir_digest_delta.py --rev 9da8358e` → `orch754b_ir_digest_delta_round2.json`
  (G11 `132`): **1 leg moved**, every pointer under `withheld/{reason, ambiguous_endpoints,
  unwrappable_proteins}` plus the matching warning text; errors, `ok` and stops unchanged. The new
  value is pytest's own (`evidence/orch754b_pins_round2_before.pytest.log`, G11 `129`). R3 did not
  move. Pins pass (G11 `134`).
- `docs/reaction_evidence_sample.txt`: the "Artifacts read" line now lists
  `pathway.review_required.pwml`.
- `tests/fixtures/reaction_evidence/{enterobactin_strict,lipid_a_strict}.bundle.json`: a presence
  marker for the PWML their committed legs hold (content is a placeholder; the report reads only
  that the file exists).
- `tests/test_reaction_evidence_view.py`: the session-IR test now gives a written export (`ok`,
  path, bytes) and a withheld row with its recorded endpoints.

G11 `131` re-ran `orch754b_cumulative_repin.py` from `c9265fb8` with its default output path. That
overwrote the round-0 record `orch754b_cumulative_repin.json`, which I restored from a copy taken
first. The round-2 move is the `9da8358e` delta above. `131` stands as a run whose output was
discarded.

### 15.6 Gates (round 2)

| gate | tip (`1f617891` tree) | base `f9892cdf` (clean worktree) | report |
|---|---|---|---|
| focused (new file + interaction export + spellings) | **106 passed** | — | `139` |
| affected (32 files) | 667 passed, 3 skipped, 3 failed | 670 passed, 3 failed — **identical E-lines** | `140`, base `76` |
| the 3 skips: c081 / c089 corpus tests from `C:/t/bt-orch754b/tip2-wt` @ `1f617891` | **56 passed** | 56 passed | `173`, base `120` |
| PWML export chunk (33 files) | 848 passed, 8 skipped, 2 failed (coverage tripwire) | 847 / 8 / 4 failed | `141`, base `26` |
| SMOKE (22 files) | **508 passed** | — | `142` |
| AppTest `test_reaction_evidence_view` alone | **11 passed** (9 + 2 new) | 9 passed | `130`, base `88` |
| AppTest `test_c055_rag_loop_wiring` alone | 14 passed | — | `143` |
| AppTest `test_batch_preflight` alone | 32 passed, 4 skipped, 1 failed — the same `.venv` assumption as base | identical | `174`, base `31` |
| Chunk D `--only qb`, one process per node | **23/23 executed, failed none** | — | outer `145`, gate `146`–`172` |
| digest pins | 1 failed before the re-pin, **2 passed** after | — | `129`, `134` |
| G9: the round-2 tests at `9da8358e` (`C:/t/bt-orch754b/r1-wt`) | 9 failed + the app-path AppTest failed, all on assertions | — | `135`, `136` |
| measurements | LLM-input identity `122`, Mapping Misses replay `124`, sample regeneration `127`, IR delta from `9da8358e` `132`, ambiguous rows `133`, text-vs-IR corpus `137`, wrapper proof `138`, unwrappable endpoints `144` | | |

Every report: `final_surviving_count: 0`. `121` (summary-shape probe; script kept as
`evidence/orch754b_probe_summary_shapes.py`) and `126` (a debug print of one IR row) were
diagnostics; `125` / `128` dev iterations; `131` is the discarded run noted in § 15.5; `123` was
refused before starting (heavy lock held elsewhere), so no report was promoted.

**Corpus, re-measured** (`137`): 330 of 330 rows in each configuration, the record states the IR,
0 disagreements. A PWML is placed beside the IR only where the export would have written one
(no blocking IR error, valid IR) — 41 of 94 legs direct, 67 of 94 after the replay — and every
other row reads `Not recorded (no PWML written …)`. Wrapper proof (`138`): unchanged from round 1
except 2 replay rows that are now `ambiguous_protein_endpoint` instead of a missing identity
(exportable still 13 → 73); preservation and confinement hold. Unwrappable endpoints (`144`):
25 occurrences (`EntB`'s and `entD`'s ambiguous rows left the list).

**TEST_MATRIX delta (round 2):** `test_pwml_interaction_protein_wrappers.py` 21 → 24 tests;
`test_reaction_evidence_view.py` 10 → 11; one GOLDEN digest moved (§ 15.5); export chunk still
at 2 known failures; SMOKE 508.

### 15.7 What I believed and what disproved it (round 2)

1. **"`compact_mapping_misses` ignores a summary dict."** My round-1 probe said so, but the
   probe's summary lacked the `…_id` key that makes the heuristic treat the dict as an entity.
   The reviewer measured real reports; the 95-leg replay now confirms 24 → 0.
2. **"An IR row with `export_status: exportable` is drawn."** False twice: when no PWML is
   written, and when the IR resolved no entity for an endpoint (that row is "exportable" with
   `left: null`). The record now says neither is drawn.
3. **"Shared protein / gene names are hypothetical."** They are on committed data: `EntB` and
   `entD` on one PMC12096016 leg.

### 15.8 Not settled — added this round

11. **`generated_wrappers_created`** (base's own `map_payload` counter) rises with every
    authorized interaction wrapper and sits in the summary the curator's Mapping Misses can copy.
    Options: (a) accept it as counting the authorized change (no cost); (b) exclude interaction
    wrappers from it (cost: the counter then undercounts generated complexes, and it is base
    behaviour outside this lane).
12. **Committed legs still read `Not recorded`** for every interaction. Their IRs predate per-row
    verdicts, and the PMC12856317 2026-08-25 strict leg wrote no PWML. Only legs exported after
    this card carry verdicts in the `.txt`.
