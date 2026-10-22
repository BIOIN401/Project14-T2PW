# ORCH-762 — protein identity by taxonomy, never across a subspecies (FX1-FX6)

**Implementer lane, 2026-10-22.** Branch `orch762/protein-identity`, base `493c1a10` (ORCH-759a tip).
Governing rulings: **D-106** (the identity requirement is not relaxed; fix the lookup) and **D-108** (an
identity may come from another strain of the same subspecies when gene and function match; never across a
subspecies or species boundary; every borrowed identity flagged with its source strain, never in LLM-bound
text). Scope: `ORCH-761-RESULT.md` § 7 FX1-FX6. No gate, threshold, species rung or Unknown-placeholder rule is
touched; the identity verdict (`verify_real_protein_identity`) is unchanged byte for byte.

| commit | what |
|---|---|
| `a001129b` | test first: `tests/test_orch762_protein_identity.py`, `tests/orch762_harness.py`, `tests/fixtures/orch762/` |
| `a0e9e5ae` | code: `src/t2pw/mapping/map_ids.py`, `src/t2pw/curation/prompt_payload_keys.py`, `src/t2pw/app/reaction_evidence.py` |
| `9d140ce7` | test: the walker test must see the flag before it proves a skip (NEW test, no G9 claim) |
| `1de46133` | comment only: a `ponytail:` marker naming the free-text fallback (§ 9 item 6) |
| docs commit | this file, G11 reports `evidence/g11/ORCH-762/01`–`30`, logs and scripts `evidence/orch762/` |

Line numbers below are at `1de46133`.

## 1. What each fix does

### FX1 — UniProt by taxonomy, the strain first, then its bound (`map_ids.py`)

* `_uniprot_taxon_record` (`:4639`) reads `https://rest.uniprot.org/taxonomy/<id>` (rank, scientific name,
  lineage), cached in the mapping cache section `uniprot_taxonomy`; a failed answer is not cached.
* `_uniprot_taxon_scope` (`:4681`) turns it into a query ladder and the D-108 bound (§ 2).
* `map_protein_uniprot` (`:4986`) takes `taxon_scope`; `_map_protein_uniprot_in_scope` (`:4858`) runs the
  tiers: the strain (`organism_id:<id>`, exact), then — only if it yields no candidate — FX5 (`:4790`), then the
  bound (`taxonomy_id:<bound>`, lineage). Each tier is the existing body with `organism_clause` in place of
  `organism_name:"…"`; under a clause no organism-free query is issued (`:5054`) and the model alias tier is
  skipped (`:5206`). Scoped candidates carry `matched_scope` = the clause they were found under.
* `_map_protein_with_strategy` (`:7134`, new kwarg `identity_context`; `None` = old behaviour): a scoped answer
  is cached under `api-v9::<name>::tax:<requested>><bound>::…` and **no legacy key is read** (`:7270`), so the
  ORCH-756 leg's `api-v8::men*::…kf147…` E. coli/human best efforts never replay. Unscoped lookups keep their
  `api-v8` key and legacy chain unchanged. A row's earlier candidate list is promoted only if found under this
  scope (`_with_taxon_clause_candidates`, `:4912`).
* The tail of `map_protein_uniprot` moved verbatim into `_uniprot_result_from_candidates` (`:4552`) so FX4 and
  the subspecies filter can re-accept a narrowed list; unscoped results are identical.
* The Phase-2 fallback (`:11374`) and the compound reroute (`:11667`) are scoped the same way. Unscoped, the
  Phase-2 fallback would bring A2RHR6 back after the main loop refused it (it did in a dead end, § 8 F2).
* Tests: `test_g9_kf147_men_enzymes_resolve_to_their_same_subspecies_records[stage2|stage6]`,
  `test_g9_a_cached_free_text_answer_does_not_replay`, `test_g9_mena_no_longer_takes_the_cremoris_record[…]`,
  `test_new_a_scoped_lookup_opens_no_organism_free_door`, `test_new_a_stale_candidate_list_is_not_promoted_under_a_scope`,
  `test_preservation_no_taxonomy_id_keeps_the_free_text_lookup`.

### FX2 — the resolved species, not the row's free text

`_requested_taxonomy_id` (`:4755`): the row's (or its `species_ref`'s) taxonomy id, else the payload species
row whose `name`, `raw_name` or `aliases` is the row's organism (NCBI backfill writes the id on species rows
only, ORCH-761 § 4.1). Two rows disagreeing give no id. The protein loop judges the row against the scope's
`verdict_organism` — the bound's UniProt scientific name — for every later re-check (`:11064`), and records
`mapping_meta.query.taxonomy_id`. Covered by the KF147 replay (taxonomy id from the species row at Stage 2 and,
after the rename, at Stage 6).

### FX3 — PathBank species by taxonomy id and "(strain X)"

`map_protein_row` (`:2135`): when the species name finds no PathBank species, it tries the row's taxonomy id
(`_find_species_ids` already matches `taxonomy_id=`), then `"<name> (strain <code>)"`
(`_pathbank_species_alternatives`, `:834`); every later rung runs unchanged on the species that answered.
The strategy passes the taxonomy id on the row it hands PathBank; a cached `species_not_found` is re-asked
when an id exists (`:7223`), other cached PathBank answers replay as before. A PathBank candidate whose stamped
taxon equals the requested one gets `requested_taxonomy_id`, so the verdict's existing exact-taxon branch can
read it. Tests: `test_g9_pathbank_species_tier_finds_the_strain[taxonomy_id|strain_spelling]`. PathBank has no
*Lactococcus* Men rows, so FX3 recovers no Men identity (as ORCH-761 predicted); it maps KF147 to species 10733.

### FX4 — the MenH synonym clash, by function

`_break_gene_synonym_collision` (`:4933`), applied to the value read, never to the cache (`:7316`), and in
Phase 2. When two or more records carry the queried symbol, a record that matches it only through a synonym
and whose **primary** gene is the name of another enzyme of this payload — one whose reactions share none of
this protein's (`_enzyme_reaction_signatures` `:5654`, `_foreign_enzyme_symbols` `:5689`) — is set aside and
the rest re-accepted; the survivor faces the unchanged verdict. MenH (SEPHCHC → SHCHC) drops P49016 (primary
`menG`; the payload's MenG catalyses DMK → MK) and keeps Q9CHK1 (primary `menX`). Recorded as
`mapping_meta.gene_synonym_collision`. Tests: `test_g9_menh_resolves_to_its_own_record_not_the_menG_methyltransferase[kf147|lactis_species]`,
`test_new_gene_synonym_tie_break_needs_a_different_reaction`. **The EC route was not built** (§ 9 item 5).

### FX5 — exact sequence through UniParc (NEW capability)

`_exact_sequence_candidates` (`:4790`), between the strain and the bound tier: UniParc
`taxonomy_id:<strain> AND gene:<symbol>` must name exactly one sequence (several = paralogs; none is chosen);
its active UniProtKB cross-references are fetched with `(accession:…) AND taxonomy_id:<bound>`, so a record
outside the bound cannot come back. The rows carry `identity_route: exact_sequence_uniparc` and `uniparc_id`,
and face the unchanged verdict. NEW acceptance tests:
`test_new_exact_sequence_route_returns_the_identical_records_accession` (MenB: UPI00000C68C6 → Q9CHK2),
`test_new_exact_sequence_route_refuses_paralogs` (MenA: three UPIs; MenF: two), `test_new_exact_sequence_route_stays_inside_the_subspecies`.

### FX6 — one kind per name (supersedes ORCH-761 Q2)

`_collect_protein_like_names` (`:5612`): a `protein_locations` row no longer makes a name protein-like when
the payload files that name as a compound and as no protein or complex. The normaliser files a dangling
location token it does not read as a protein (`holo-EntB`) as a compound (`process_normalizer._ensure_compound`);
the mapper now agrees and maps it on the compound route, so it never claims P0ADI4 and identity admission
has no cross-kind conflict to withhold. A compound that is also an enzyme or transporter keeps the protein route.
The normaliser was **not** changed: filing `holo-EntB` as a protein adds an initial-gate error ("Located protein
is isolated in connectivity graph", `process_normalizer.py:5070`), which reaches the audit prompt (§ 8 F5).
Tests: `test_g9_entb_keeps_its_grounded_accession_on_the_orch755_shape`,
`test_preservation_a_compound_row_that_is_also_an_enzyme_stays_on_the_protein_route`.

## 2. The D-108 bound, and how each path enforces it

| requested organism (NCBI rank of its taxonomy id) | tiers queried | bound | borrowed? |
|---|---|---|---|
| strain with a subspecies in its lineage (KF147 684738 → subsp. *lactis* 1360) | `organism_id:684738` → FX5 → `taxonomy_id:1360` | the subspecies | yes, when the record's taxon ≠ 684738 |
| strain with no subspecies in its lineage (e.g. *E. coli* K-12 83333 → 562) | `organism_id:` → FX5 → `taxonomy_id:<species>`, **dropping every subspecies-qualified record** (`_outside_unknown_subspecies`, `:4739`) | the species | yes, as above |
| species or subspecies (e.g. "Lactococcus lactis" 1358, "Escherichia coli" 562) | `taxonomy_id:<it>` only | itself | no: the request names no narrower organism |
| genus or above, or a lineage with no single species | — (no scope) | — | old free-text lookup |

Decision for "no subspecies known": the species is the bound, and a record UniProt files under one of that
species' subspecies is refused, because the requested strain may sit on the other side of a boundary nobody
can see. Decision for a species-level request: its records from any strain or subspecies inside it are the
requested organism's own and are not flagged (this is also what T2PW does today for *E. coli*). Both are
stated as open in § 9.

Enforcement, path by path: (1) the query clause — UniProt itself restricts every scoped query to the bound's
subtree, and no organism-free query exists under a scope; (2) the verdict's organism is the bound's name, so
a record from outside a subspecies bound is a `species_mismatch` on the rung's own unchanged string rule
(`test_new_the_climb_never_passes_the_subspecies` feeds the bound tier a cremoris record and it is refused);
(3) no legacy free-text cache entry and no unscoped earlier candidate is read under a scope; (4) FX5's
record fetch carries the bound clause; (5) Phase 2 and the compound reroute use the same scope. A
species-level bound's string rule still reads "Lactococcus lactis subsp. cremoris (strain MG1363)" as a
qualification of "Lactococcus lactis"; under a scope no lookup can return that record for 1358 (UniProt now
ranks 1359 as a species outside 1358), but an accession arriving on the row from elsewhere is judged only by
that rule (§ 9 item 1).

## 3. The flag (D-108 § 3) and why it never reaches a model

`_borrowed_protein_identities` (`:5713`), computed last in `map_payload` (`:11974`) from what each protein row
actually ships after admission, the ORCH-759a fold and rule 3: shipped UniProt verified, the verdict's judged
record for that accession, its taxon ≠ the requested taxon, and a strain-level scope. Each record:
`{json_pointer, name, uniprot, source_strain, source_organism, source_taxonomy_id, requested_organism,
requested_taxonomy_id, bound_name, bound_taxonomy_id, bound_rank, route, flag}`, `flag` =
**"UniProt Q9CHK2, from strain IL1403"**. It goes to the mapping report and to the payload's **top level**
under `borrowed_protein_identities`; the key is absent when nothing is borrowed, so every other payload and
report keeps its shape.

* Shown: the Reaction Evidence report (`reaction_evidence.py` `_borrowed_flags` `:208`, `_borrowed_for` `:218`,
  `_mapping` `:229`), text export and app view alike (the view renders the same `reaction_lines`). The enzyme a
  Stage-6 reaction names is its wrapper, so the line reads `MenB complex: …; borrowed identity: MenB: UniProt
  Q9CHK2, from strain IL1403`. The protected `streamlit_app.py` was not touched; no other identity panel shows it.
* Kept out of LLM input: the key is in `PROMPT_OMITTED_PAYLOAD_KEYS` (`prompt_payload_keys.py:80-84`). The
  audit serializer drops top-level keys of that set, the interactive curator drops them at any depth, and its
  mapping-report walker skips them (ORCH-759a's mechanism). Proof: `test_new_the_flag_never_reaches_an_llm_input`
  drives eight payload-reading LLM call sites (audit, completeness, gap species, map_ids species, interactive
  curator with `compact_mapping_misses(report)`, localized row repair, overwatch, pathway curator) through the
  faked transport with the KF147 Stage-6 payload and report, with and without the key: every request is
  byte-identical, and neither "from strain IL1403" nor the key name appears in any. `test_new_the_mapping_report_walker_skips_the_flag`
  covers the walker. The ORCH-757 golden (`tests/test_orch757_llm_token_budgets.py`) passes unchanged (`13`, `29`).
* The lookup data itself (queries with `taxonomy_id:`, candidates naming IL1403) is in `mapping_meta`, which
  the audit prompt carries; that is the intended identity change in the payload, not the flag.

## 4. Replay results per protein

`evidence/orch762/27-replay-table-base.log` (base `src/`) and `28-replay-table-tip.log` (tip), each a bounded
run of `evidence/orch762/replay_table.py` over the four scenarios, 0 unrecorded queries.

KF147 leg (ORCH-756 menaquinone, organism "Lactococcus lactis subsp. lactis KF147", 684738), Stage 2 and Stage 6:

| protein | base | tip | route | flag |
|---|---|---|---|---|
| MenF | none (P38051 *E. coli*, species_mismatch) | **Q9CHJ9** | bound tier (1360) | from strain IL1403 |
| MenD | none (P17109) | **Q9CHK0** | bound tier | from strain IL1403 |
| MenH | none (P37355) | **Q9CHK1** | bound tier + FX4 (P49016 set aside) | from strain IL1403 |
| MenC | none (A0A0P0ZBS7) | **Q9CHK4** | bound tier (`yhdB`, syn `menC`) | from strain IL1403 |
| MenE | none (P23971) | **Q9CHK3** | bound tier | from strain IL1403 |
| MenB | none (P0ABU0) | **Q9CHK2** | **FX5**: UPI00000C68C6 | from strain IL1403 |
| MenG | none (Q9VM90) | **P49016** | bound tier | from strain IL1403 |
| MenI | none (P77781) | none | UniParc `gene:MenI` in 684738: 0; `gene:MenI` in 1360: 0 (IL1403's record is `yhdA`) | — |
| MenA | none (Q8N8S7 human) | none | `gene:MenA` in 1360: 0; UniParc: three paralogs (menA_1/2/3), refused | — |

ORCH-755-night shape (organism "Lactococcus lactis", 1358): base admits **MenA → A2RHR6 (MG1363, *cremoris*) as
verified** at Stage 2 and Stage 6 and leaves the other eight in margin ties (as the night did); tip ships the
same seven accessions as above, **no A2RHR6 and no cremoris record anywhere**, none flagged (species request).

Enterobactin ORCH-755 night: base Stage 6 EntB `mapped_ids` lose P0ADI4 (`rejected_mapped_ids`); tip **EntB
keeps P0ADI4**, `holo-EntB` is on the compound route with no protein claim. EntC P0AEJ2, EntA P15047, EntE
P10378, EntF P11454, TolC P02930, TonB P02929 unchanged at both stages.

E. coli proteins through the UniProt tier alone (no PathBank): every Ent protein ties across E. coli strains
at base and at tip on the same judged K-12 record; LDH's refused record moved from *Thermus* P06150 to *E.
coli* P52643 (still refused). No identity changed.

Gold (eval-v2 `score_leg`, no paper text), Stage 2 and Stage 6, base and tip: menaquinone 9/9 (both shapes),
enterobactin 5/5. No gold reaction lost.

## 5. Empty species in a PWML — report only, no code

**Yes, but only when the IR holds no species at all.** The writer takes the pathway's `species-id` and every
fallback from the first IR species (`pwml/writer.py:1002-1003`, `:1186`, `:2702-2708`): with none it is
`None`, and `pwml/validate.py:308-311` normalises an empty pathway `species-id` to `nil="true"`. A protein's
`species-id` is its matched IR species, else that default (`writer.py:1188-1214`, emitted `:1264`); a complex's
is always the default, never its own row's (`:1310`, ORCH-761 § 9). No gate refuses an IR without species:
`validate_required_pwml_contract` checks species *rows* (taxonomy, classification; `pwml/ir.py:2900-2938`, an
empty list yields nothing), and the protein/complex gates (`ir.py:2778-2807`) check only that the row carries
some species text (`pipeline/entity_identity.py:297-314`), not that the writer resolves it; the IR's
"default state without context" is a warning (`ir.py:1720-1736`). Can a payload with proteins reach that? Species
hydration merges a species row for every protein and complex before mapping (`mapping/map_ids.py:4202`,
`:4273` → `_merge_species_record` `:3172`), so with proteins the IR has a species unless a later stage removes
every species row while keeping the proteins; not observed. A **compound-only** payload with no species row
and no state species does write `<species-id nil="true">`, which PathWhiz files under its default (Homo
sapiens, ORCH-761 § 1.1). A protein whose species text matches no IR species is filed under the first IR
species — wrong, not empty. Not fixed here (out of scope).

## 6. G9 RED / GREEN

| run | tree | report | result |
|---|---|---|---|
| RED | `a001129b` (tests on base `src/`) | `g11/ORCH-762/06-g9-red.json`, log `orch762/06-g9-red.log` | **20 failed, 9 passed.** All 10 `test_g9_*` nodes fail on behaviour (`AssertionError`: empty identities, `'A2RHR6' != 'A2RHR6'`, `None == 'P0ADI4'`, `species_not_found`); 10 `test_new_*` fail (no capability); the 7 preservation tests and 2 new tests pass |
| GREEN | `9d140ce7` | `08-g9-green` | **29 passed** |
| GREEN, final | `1de46133` | `29-final-focused` (with 759a, 757 golden, map_ids, lineage golden, reaction evidence, curator files) | 239 passed |

## 7. Gates

| gate | report(s) | result |
|---|---|---|
| 1 G9 / new / replay / preservation / LLM golden | `06`, `08`, `29`; golden in `13` and `29` | as § 6 |
| 2 affected suites (`grep -l -E "map_ids|uniprot|identity|pathbank|species|prefreeze|process_normalizer"` + `test_orch759a_duplicate_protein_fold.py`, 158 files batched, gap-resolver files last in their own job, AppTest files alone) | `09`, `10`, `12`, `13`, `14`, `15`, `16` (`11` timed out, § 8 F6) | 3806 passed, 16 skipped, **6 failed** — all six red on base `src/` (`17`) |
| 3 SMOKE | `18` (`9d140ce7`), `30` (`1de46133`) | **508 passed** both |
| 4 `tests/test_eval_v2_*.py` | `19` | 89 passed, 1 xfailed |
| 5 each alone | `20` quarantine_boundary, `21` c052, `22` c055, `23` reaction_evidence_view, `24` app_hook, `25` batch_preflight, `26` stage8_export, `16` orch759a_fold_app_path | 23 / 13 / 14 / 11 / 1 / 32+1 failed+4 skipped / 4 / 2 |

The six failing nodes in gate 2, each red on base `src/` in `17` with the same assertion: the five of
TEST_MATRIX § ORCH-754 "Known failures" (`test_c075…corpus…`, `test_c100…5b…`, two
`test_compound_resolution_extraction` coverage tripwires, `test_rag_seed_entity_reimport…merge_base…`) and
`test_paper_eligibility_corrections.py::test_the_2122_evaluation_with_cached_abstracts_beats_title_only`
(`{'derived_fro...ull_text': 28} == {'acquisition...full_text': 1}`), which is not in any list yet (§ 10). The
gate-5 failure is `test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_and_the_cure`,
worktree-only (§ ORCH-757). The 16 skips include the c081/c089 worktree-path skips. Chunk D was not run (the
orchestrator runs it). Every report: `FINAL SURVIVING COUNT : 0`; no `--heavy-lock`; one job at a time.

## 8. Findings, failures and dead ends

* **F1 — the recorded answers reproduce both nights.** Before writing tests I replayed base and tip offline
  from one live recording (`01-record-base`, `02-record-tip`; 58 + 53 answers fetched, 111 in the fixture).
  Base reproduces ORCH-761 § A (all nine species_mismatch) and § B (eight margin ties, MenA → A2RHR6 verified)
  and § C (EntB withheld) exactly, so the fixture is the nights' own behaviour, not a constructed one.
* **F2 — Phase 2 would have undone FX1.** My first draft scoped only `_map_protein_with_strategy`. The Phase-2
  fallback (`map_payload`'s last UniProt pass, free text, runs for every protein still without an id) would
  have re-issued `organism_name:"Lactococcus lactis"` for MenA and admitted A2RHR6 again. Found by reading,
  before the first run; scoped (`:11374`).
* **F3 — the flag first went nowhere visible.** The Reaction Evidence report shows ids only from a row's
  top-level fields, and a Stage-6 reaction names the wrapper complex, not the protein: my first rendering
  (flag on the protein's own line) never printed (`03-dev-new`, 1 failed). The flag now rides the wrapper's
  line via its components. Pre-existing: the report prints "no identifier recorded" for every wrapper.
* **F4 — FX6 on the normaliser side was rejected.** Filing `holo-EntB` as a protein is biologically closer
  (D-064: a state of EntB) but makes it a located protein with no edge, an initial-gate *error*, and initial-gate
  errors are the audit's input — an LLM-input change outside this lane. The mapper-side agreement changes no
  gate and no prompt.
* **F5 — FX4 by EC was not possible here.** The payload carries no EC numbers and SEPHCHC/SHCHC are novel
  compounds with no ChEBI id (ORCH-756 `final_mapped.json`), so neither an EC match nor a reaction-participant
  match against UniProt catalytic activity can be computed. The payload's own enzyme→reaction assignment is the
  function evidence used.
* **F6 — an AppTest file stalls a batch.** `test_orch759a_fold_app_path.py` (ORCH-759a) in a 41-file batch hung
  at 23% until the 540 s timeout (`11`); alone it passes in 5.6 s (`16`). It belongs on the never-batch list.
* **F7 — FX6 removed one model call on the enterobactin replay.** At base the reroute of `holo-EntF` reached
  the model alias tier once (`27`: `ai=1`); at tip `holo-EntF` is on the compound route (`28`: `ai=0`). The alias
  tier's prompt is unchanged; it is simply not reached. Under a scope the alias tier is skipped by design; in the
  KF147 and *L. lactis* replays it fired 0 times at base and 0 at tip.
* **F8 — more EuropePMC calls.** With no organism-free query, a scoped protein with no hit (MenI, MenA) now
  reaches the literature tier: 16 EuropePMC lookups per KF147 replay at tip, 0 at base (`27`/`28` `lit=`).
  Network, not model.
* **F9 — process slips, recorded:** twice I launched `python -` with an empty heredoc (forbidden pattern); both
  hung and were stopped with TaskStop before running anything (no file written, no test affected). A `tar -C C:/…`
  extraction also hung (GNU tar reads `C:` as a host); replaced by Python `tarfile`.

## 9. Not settled — for the product owner

1. **A row-carried accession across a species-level boundary.** For a species request ("Lactococcus lactis")
   the unchanged species rung accepts "Lactococcus lactis subsp. cremoris (strain MG1363)" by name. No lookup of
   this lane can produce that record, but an accession put on the row by Stage 1, the curator, or the gap
   resolver (`gap_resolver.py:1825/1858/2398` and `rag/tiers.py:351` still call `map_protein_uniprot` with free
   text; outside this boundary) is judged only by that rule. Options: (a) leave it (cost: the gap persists for
   non-lookup sources); (b) a lineage-aware species rung (one taxonomy call per candidate taxon; changes the
   verdict, needs a ruling under D-106); (c) scope the gap-resolver and RAG lookups too (a separate lane).
2. **Species-level requests are not flagged.** "Lactococcus lactis" answered by an IL1403 record ships without a
   flag (the request names no strain). If D-108 § 3 means "any record not of the paper's literal organism",
   the flag would also fire for every *E. coli* protein answered by K-12. Options: keep (no noise), or flag when
   the record's taxon is below the request (cost: flags on most bacterial papers).
3. **Strain with no subspecies:** the species is the bound and subspecies-qualified records are dropped by
   UniProt's "subsp." naming, not by lineage (`ponytail:` at `:4745`). Exact lineage per candidate costs a
   taxonomy call per taxon.
4. **MenA (three KF147 `menA` genes) and MenI (`yhdA`)** stay unresolved. MenA: FX5 refuses paralogs; ORCH-761
   Q1(a) on naming one paralog is still open. MenI: no symbol route exists; an alias `yhdA` would need a source.
5. **FX4 has no EC route** (F5). If payloads start carrying EC numbers, an EC match against UniProt's `ec`
   field is the next step.
6. **A failed taxonomy answer falls back to the old free-text query** (`ponytail:` at `:4723-4726`), which can
   cross a subspecies on a species-level name. Cached per taxon after the first success. A degraded
   `taxonomy_id:<requested>` scope would hold D-108 through the failure (cost: offline tests with taxonomy ids
   would change query shape).
7. **Not established until a leg runs:** whether the KF147 leg now writes a PWML with seven IL1403 identities,
   and the PathWhiz consequence (`check_uniprot` re-files each under IL1403, ORCH-761 § 1.2). The replay stops at
   Stage 6; quarantine, pre-freeze and export were not run.
8. UniProt answers are as of 2026-10-22 (recorded); entries move.

## 10. TEST_MATRIX delta (exact text to append, end-of-file by rule)

```
## ORCH-762 (D-106, D-108): test obligations — appended end-of-file by rule · 2026-10-22

Lane `orch762/protein-identity` (`a001129b` test-first, `a0e9e5ae` code, `9d140ce7` test, `1de46133`
comment). Reports `evidence/g11/ORCH-762/01`–`30`, logs `evidence/orch762/`.

**SMOKE is unchanged at 508** (`g11/ORCH-762/18`, `30`). No pin moved.

### New focused selection

`tests/test_orch762_protein_identity.py` (**29**, ~2 s): offline replays of the ORCH-756 KF147 and ORCH-755
menaquinone/enterobactin legs through Stage-2 `map_payload` -> `normalize_process_payload` -> Stage-6
`map_payload`, with UniProt answers recorded once (`tests/fixtures/orch762/uniprot_responses.json`, 111
answers) and every other host mocked (`tests/orch762_harness.py`). G9: 10 `test_g9_*` red at base
(`g11/ORCH-762/06`), 29 green at tip (`08`, `29`). Run it for any change to `map_protein_uniprot`,
`_map_protein_with_strategy`, `PathBankDbResolver.map_protein_row`, `_collect_protein_like_names`,
the identity verdict, `PROMPT_OMITTED_PAYLOAD_KEYS` or `reaction_evidence._mapping`. A change that issues a
UniProt query the fixture does not hold fails with an "unrecorded" list: re-record with
`docs/pwml_recovery_sprint/evidence/orch762/record_fixtures.py` (live UniProt, fixture-building only).

### Never batch: one more AppTest file

Add `tests/test_orch759a_fold_app_path.py` to the AppTest files that must each run alone: in a 41-file
batch it hung to the 540 s timeout (`g11/ORCH-762/11`); alone, 2 passed in 5.6 s (`16`).

### Known failures

Unchanged from § ORCH-754, § ORCH-757 and § ORCH-757i, plus one found by this lane's wider sweep, red on base
`src/` too (`g11/ORCH-762/17`):
`test_paper_eligibility_corrections.py::test_the_2122_evaluation_with_cached_abstracts_beats_title_only`
(`abstract_sources` count mismatch).
```

## 11. Files

| file | what |
|---|---|
| `src/t2pw/mapping/map_ids.py` | FX1-FX6, the flag record (§ 1, § 3) |
| `src/t2pw/curation/prompt_payload_keys.py` | `BORROWED_PROTEIN_IDENTITIES_KEY` in `PROMPT_OMITTED_PAYLOAD_KEYS` |
| `src/t2pw/app/reaction_evidence.py` | the flag on the Reaction Evidence mapping lines |
| `tests/test_orch762_protein_identity.py`, `tests/orch762_harness.py` | G9, new acceptance and preservation tests; the replay harness |
| `tests/fixtures/orch762/` (+ README) | leg payloads, leg cache rows, recorded UniProt answers |
| `docs/pwml_recovery_sprint/evidence/orch762/make_fixtures.py` | cuts the payload fixtures (prints every sha256) |
| `docs/pwml_recovery_sprint/evidence/orch762/record_fixtures.py` | the one-time live UniProt recorder (`01`, `02`) |
| `docs/pwml_recovery_sprint/evidence/orch762/replay_table.py` | the per-protein table of § 4 (`27`, `28`) |
| `docs/pwml_recovery_sprint/evidence/g11/ORCH-762/`, `…/g11/pin/ORCH-762/`, `…/orch762/*.log` | G11 reports, pins, logs |

Not touched: `streamlit_app.py`, `_fold_unreferenced_duplicate_proteins`, `extraction_diagnostics.py`,
`process_normalizer.py`, any gate, `TEST_MATRIX.md`, `LEDGER.md`, `DECISIONS.md`.
