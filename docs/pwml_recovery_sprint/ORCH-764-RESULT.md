# ORCH-764 — mapping never narrows a class name (D-109 § 2): RESULT

**Lane:** `orch764/no-class-narrowing`, base `79080db7` (ORCH-762 tip, on ORCH-759a). Implementer: general-purpose
agent, unattended. Not merged, not pushed, not self-approved: an independent reviewer judges the diff.

| commit | what |
|---|---|
| `9bcc99ad` | test first: `tests/test_orch764_class_term_not_narrowed.py` + `tests/fixtures/orch764/menA_prenyl_diphosphate.json`, on base code |
| `9c0a0f95` | the fix: `src/t2pw/pwml/compound_resolution.py`, `src/t2pw/mapping/map_ids.py` (+ one test docstring and one test parameter, § 8 F7) |
| docs commit | this file, `evidence/orch764/` (scripts, logs), `evidence/g11/ORCH-764/01-25`, `evidence/g11/pin/ORCH-764/` |

**Outcome.** On the ORCH-759 menaquinone leg the narrowing is reproduced offline, byte for byte, and removed:
MenA's "prenyl diphosphate" keeps the paper's term, carries no identifier, and is flagged
`db_status = identity_refused_review_required` with refusal reason
`class_term_never_narrowed_to_specific_molecule`. Offline replay, final payload: menaquinone **7/9 → 8/9**
(P 0.875 → 1.000, R8 regained), enterobactin **5/5 → 5/5**; Stage 1 untouched (9/9, 5/5). No gate relaxed; no
prompt changed. Every pipeline LLM stage (gap resolver, audit, curator) runs before pre-freeze resolution, and the
leg's Stage-6 output, which they read, is byte-identical; the only LLM-visible change is the intended one, in the
user-triggered interactive curator's view of the finished payload (§ 7 Q3).

---

## 1. The origin, traced

**Answer: pre-freeze compound resolution, on a PathBank synonym.** Not an LLM stage, not a cache entry, not a
fuzzy match, not the offline name index.

### 1.1 Artifact chain (leg `runs_papers/ORCH-759/legs/menaquinone-strict/2026-10-22_1148/papers/PMC12421875/strict/`)

| artifact | "prenyl diphosphate" | "Dimethylallylpyrophosphate" | what it shows |
|---|---|---|---|
| `stage1_payload.json` | 11 | 0 | Stage 1 extracted the paper's term |
| `merged_payload.json` | 16 | 0 | still the paper's term after the Stage-2 merge |
| `final_mapped.json` (sha256 `3eb9eae2…6bbf`) | 22 | 34 | renamed; row `entities/compounds/4` |

Row `entities/compounds/4` of `final_mapped.json`:

* `name` `Dimethylallylpyrophosphate`, `raw_name` `prenyl diphosphate`, `db_status` `matched`,
  `db_match.chosen` PathBank compound **874** (HMDB0001120, KEGG C00235, ChEBI 16057), `db_match.chosen_rule`
  **`exact_short_name_or_synonym`**, confidence **0.70**. `db_match`, `db_status` and `raw_name` are written only by
  `db_resolver.apply_compound_db_resolution`, i.e. by `PathWhizCompoundResolver` in pre-freeze resolution.
* `mapping_meta` (written by Stage 6, `map_ids`) says the opposite: `resolution {status: novel, issue:
  implausible_name_match, order_step: name_plausibility_gate}`, `candidates []`. **Stage 6 had refused DMAPP.**

### 1.2 Code path (tip line numbers; unchanged at base apart from the inserted lines)

1. `streamlit_app.py:4314-4323` `run_prefreeze_resolution` → `prefreeze_resolution.resolve_compounds_prefreeze` →
   `compound_resolution._resolve_compound_rows`. It runs after the curator (`:4232`) and the final `map_payload`
   (`:4248`): after every pipeline LLM call.
2. `db_resolver.py:209` `_exact_short_or_synonym_rows("prenyl diphosphate")` (rule `:364`, confidence 0.70): PathBank
   row 874's `synonyms` column holds an entry that normalizes to "prenyl diphosphate" (`short_name` is "Ipe", so it
   is the synonym). One row → `status matched`.
3. `compound_resolution._db_match_rule_decision`: D-028 rule 2 admits any exact name/synonym of more than
   `SHORT_ABBREVIATION_MAX_CHARS = 4` characters (`:450` onward). "prenyl diphosphate" is 18.
4. Confidence 0.70 < 0.85, so `_resolve_compound_rows` logs `compound_db_resolution_failed` and then applies anyway
   with `admit_identity=True` (`:834`): `apply_compound_db_resolution` renames (`db_resolver.py:463`) and stamps
   DMAPP's ten identifiers.
5. `resolve_compounds_prefreeze` propagates the rename (`rename_map {prenyl diphosphate: Dimethylallylpyrophosphate}`)
   into all three R8 reactions and `element_locations`, and files the paper's term under `synonyms`.

**Reproduced.** `evidence/orch764/replay.py` undoes the recorded pre-freeze stamps on the leg's
`final_mapped.json`, re-runs `resolve_compounds_prefreeze` with the recorded PathBank row as the only database
answer, and at base **reproduces the recorded `final_mapped.json` exactly** (sorted-JSON equality, whole payload;
`g11/ORCH-764/08`, log `orch764/08-replay-base-2.log`). The same replay is exact on the enterobactin leg.

### 1.3 Ruled out, with the artifact

| suspect | evidence it is not the origin |
|---|---|
| gap resolver (LLM) | `gap_iteration_summary.json`: 1 round, `payload_changed false`; audit summary `gap_mapped_ids_added 0` |
| audit loop (LLM) | `audit_iteration_summary.json`: 11 patches, rationale "Added UniProt identifiers … renamed the two KF147 biological states"; `merged_payload.json` and every pre-mapping artifact carry 0 DMAPP strings |
| `data/id_mapping_cache.json` | it holds `compounds["db::prenyl diphosphate::::{}"]` = DMAPP via PathBank `synonym`, confidence 0.95 — but Stage 6 gates every read through the name gate and refused it (no shared meaningful token: "prenyl diphosphate" vs "Dimethylallylpyrophosphate" / "Ipe" / "CPD-4211"; fixture `stage6_gate_at_base`). Pre-freeze resolution does not read this cache: it queries PathBank directly |
| `data/enrichment_cache.json` | no compound entry for the term; the only hits are UniProt MenA protein text |
| offline name index (`data/pathwhiz_id_db.json`) | no entry for the term; it is keyed on external ids, which the row did not carry |
| fuzzy match | the rule is exact; D-028 already never admits `fuzzy_name` |

**Data origin:** PathBank compound 874 lists "prenyl diphosphate" as a synonym of DMAPP. In strict nomenclature
that is correct (prenyl = 3-methylbut-2-enyl; the row's own description reads "also known as … delta-prenyl
diphosphate"); the paper uses it for the class of MenA's polyprenyl donors (gold R8 note: "the paper leaves the
chain length unspecified"). The two compound gates disagreed on the same PathBank row: Stage 6's name gate
refused it, pre-freeze D-028 admitted it.

## 2. The class-vs-specific rule, and its limits

**Rule.** A compound participant is a *class term* when its name, keyed as casefold alphanumeric runs joined by one
space, is in the committed closed list `compound_resolution.CLASS_PARTICIPANT_TERMS` =
{`prenyl diphosphate`, `prenyl pyrophosphate`}. For a class term:

1. **Pre-freeze** (`_db_match_rule_decision`): every database match is refused — exact name, exact synonym **and**
   identifier rules — with reason `class_term_never_narrowed_to_specific_molecule`. The existing D-028 record-only
   path then applies: no rename, no identifier, `db_status = identity_refused_review_required`, the refused answer
   kept in `db_match` and an entry in the resolution report's `unresolved.db_identities` with the admission record.
2. **Stage 6** (`map_ids`, both compound name-gate sites: a resolver result in `_apply_name_plausibility_gate`, and
   identifiers already on the row in `_enforce_shipped_identity_names`): a name-gate verdict that would let an
   identifier through becomes a refusal with the same reason; the row ends `novel` (lineage `review_required:
   true`), identifiers filed under `rejected_mapped_ids`. A verdict the gate already rejects keeps its own reason
   byte for byte, so no existing Stage-6 output moves.
3. **Class identity:** none. No committed source holds a class-level identity for either term (§ 7 Q1), so a class
   term is never mapped; the "map to a ChEBI class if one exists" branch of D-109 is not built.

**Why a list and not a general rule (fail closed):** the narrowing passed the obvious general test — "the target's
recorded synonym equals the paper's term" — so no lexical rule over the database answer can see it. A rule
"a synonym-only match must share a token with the target's primary name" (Stage 6's gate) would refuse legitimate
renames such as `DMAPP` → Dimethylallylpyrophosphate and `ferric iron` → `Fe3+` (§ 8 F4). Only knowing that the
paper's word names a class distinguishes the case, and that is what the list records.

**Limits.**

* An unlisted class name that a database files under one member still narrows exactly as before. The cure is one
  list entry plus a test row.
* Exact key only: articles and plurals ("a prenyl diphosphate", "prenyl diphosphates") and compounds of the term
  ("prenyl diphosphate (DMAPP)") are not listed and are not caught.
* `prenyl pyrophosphate` is listed as the same name in the other spelling; it is not measured on any leg.
* Stage 6 gates only `status: mapped` results (pre-existing): an `ambiguous` result is not judged here.
* ~~An identifier already on a class row when pre-freeze runs is not removed there (D-028 refusals are
  record-only); Stage 6's shipped-identity site removes it on the final `map_payload` pass, which runs first.~~
  **Wrong, corrected in repair round 1 (§ 13).** Stage 6 strips only `mapped_ids` and the `pathbank_compound_id`
  scalar and returns early when `mapped_ids` is empty; the legacy-id branch ran before the class check; and
  `run_pwml_export` runs pre-freeze resolution with no Stage 6 at all. So a class row arriving with a member id
  shipped it. Since repair round 1, pre-freeze resolution withholds every member identifier from a class row
  before the legacy-id branch, so the limit no longer holds.
* An incoming resolution account is kept (pre-existing D-015 rule, `_resolve_to_fixed_point`): a class row that
  arrives already carrying `db_status` (e.g. a saved, already-resolved payload re-exported) has its identifiers
  withheld, but its recorded `db_status` / `chosen_rule` are restored as they arrived.

## 3. The diff (`git diff 79080db7 -- src`: 2 files, +44 −4)

* `src/t2pw/pwml/compound_resolution.py` (+25): `CLASS_PARTICIPANT_TERMS`, `CLASS_TERM_NEVER_NARROWED`,
  `is_class_participant_term` beside the D-028 constants (`:382-398`); one early refusal in
  `_db_match_rule_decision` after the no-match check (`:440-444`). Imports unchanged (the
  `test_extraction_creates_no_import_cycle` import set still holds).
* `src/t2pw/mapping/map_ids.py` (+19 −4): import of the predicate (`:44`), `_refuse_class_term_identity`
  (`:6876`), wrapped around `_name_gate_verdict` at the two compound sites (`:6860`, `:7103`). Not touched:
  the protein branch, `_fold_unreferenced_duplicate_proteins`, ORCH-762's identity code, `streamlit_app.py`,
  `stage_one_ensemble.py`, `strict_quarantine.py`, `extraction_diagnostics.py`, any prompt.

## 4. G9 proof

`tests/test_orch764_class_term_not_narrowed.py`, 11 tests. The PathBank database is a small SQL emulator over the
recorded row (`FixtureCompoundDb`); no test imports a symbol the base lacks.

| run | tree | G11 | result |
|---|---|---|---|
| RED | test commit `9bcc99ad` (base `src/`) | `02-g9-red` | **4 failed, 6 passed**: both `test_g9_prefreeze…` (`'Dimethylallylpyrophosphate' == 'prenyl diphosphate'`), `test_g9_stage6_refuses…` (`'mapped' == 'unmapped'`), `test_g9_stage6_strips…` (`rejected False is True`) |
| RED, final test file | tip tests, base `src/` checked out | `05-g9-red-base-src` | **5 failed, 6 passed** (the added `prenyl pyrophosphate` parameter fails too) |
| GREEN | tip | `03-g9-green` (10 tests), `25-g9-green-final-focused` | **10 passed**; final file in a 210-test focused selection, all passed |

What each kind proves:

* `test_g9_prefreeze_keeps_the_paper_class_term_and_flags_it` — **recorded**: the leg's MenA slice before pre-freeze
  and its recorded PathBank answer. Base: renamed to DMAPP with DMAPP ids. Tip: paper term kept, no ids,
  `identity_refused_review_required`, `rename_map {}`, refusal reason recorded.
* `test_g9_stage6_*` — **constructed** provider answers (labelled in the test): a record named as the class term
  carrying DMAPP's ids. Base keeps it on shared tokens; tip refuses it, and strips such ids already on the row.
* `test_fidelity_…` — the emulator returns exactly the leg's recorded `db_match`.
* `test_preservation_*` (labelled): the leg's own Stage-6 answer is byte-identical to the base output pinned in the
  fixture; a paper naming DMAPP by its PathBank name still maps to 874; "dimethylallyl diphosphate" carrying KEGG
  C00235 still resolves and is renamed; Stage 6 keeps a same-shape answer under a specific name; an exact
  PathBank synonym of a specific molecule (`dextrose` → D-Glucose) is still admitted.

## 5. Offline replay (both ORCH-759 3-draw legs, eval-v2 `score_leg`, gold as in `orch759_gold_accuracy.txt`)

`evidence/orch764/replay.py`; base `08-replay-base-2`, tip `09-replay-tip`.

| leg | level | base | tip |
|---|---|---|---|
| menaquinone PMC12421875 | stage1 (recorded) | P 1.000 R 1.000 F1 1.000, 9/9 | same (not on this code path) |
| | final (recorded `final_mapped.json`) | P 0.875 R 0.778 F1 0.824, 7/9 | — |
| | final (replayed) | **P 0.875 R 0.778 F1 0.824, 7/9** — identical to the recorded payload | **P 1.000 R 0.889 F1 0.941, 8/9** (R8 regained) |
| enterobactin PMC12096016 | stage1 | 5/5 | 5/5 |
| | final (replayed) | 5/5, identical to the recorded payload | **5/5**, identical to the recorded payload |

Menaquinone R2 (MenD) stays lost to `protein_missing_external_identity` (ORCH-762's lane), so 9/9 at the final
level is not reachable from this lane. Replay limit: it starts from the post-quarantine payload, while production
runs pre-freeze resolution before the strict quarantine; strict quarantine has no compound identity rule
(`strict_quarantine.py` checks only `protein_missing_external_identity`), and the same leg's identity-less
compounds SHCHC, OSB-CoA and DHNA-CoA pass it with their reactions (R4-R7 matched).

## 6. Gates

| gate | G11 | result |
|---|---|---|
| 1 G9 RED/GREEN, preservation, replay | `02`, `05` / `03`, `25` / `08`, `09` | § 4, § 5 |
| 2 affected (`grep -l -E "map_ids|chebi|compound" tests/*.py` = 176 files, incl. `test_orch762_protein_identity.py`, `test_orch759a_duplicate_protein_fold.py`); 5 batches + 6 light app files + gap-resolver files last + 4 heavy files alone | `10`-`16`, `17` (759a_fold_app_path), `18` (c055), `19`, `20` | **4416 passed, 6 skipped, 5 failed** |
| 2a the 5 failures on base `src/` | `24-base-src-failures` | all 5 red at base with identical assertion lines (md5 of the sorted `E` lines equal) — the TEST_MATRIX § ORCH-754 known failures: `test_c075…corpus…`, `test_c100…5b…`, two `test_compound_resolution_extraction` coverage tripwires, `test_rag_seed_entity_reimport…merge_base…` |
| 3 SMOKE | `22-smoke` | **508 passed** |
| 4 `tests/test_eval_v2_*.py` | `23-eval-v2` | 89 passed, 1 xfailed |
| 5 alone | `19` quarantine_boundary, `20` c052, `21` reaction_evidence_view | 23 / 13 / 11 passed |

Every report: `FINAL SURVIVING COUNT : 0`; `g11_evidence.py check --task ORCH-764`: 25 artifacts, 0 non-compliant,
19 pytest reports all with a committed pin. No `--heavy-lock`, one job at a time, Chunk D not run.

## 7. NOT settled — for the product owner

**Q1. A class identity for "prenyl diphosphate".** None exists in a committed source, so the row stays unmapped
and flagged. The nearest committed evidence is UniProt P32166 (MenA) in `data/enrichment_cache.json`: catalytic
activity "an all-trans-polyprenyl diphosphate + 1,4-dihydroxy-2-naphthoate …", cross-references Rhea
`RHEA-COMP:9563` and ChEBI `CHEBI:58914` among others. Options: (a) keep as is (cost: the compound ships with no
identity; whether PathWhiz imports an identity-less compound is ORCH-761's importer contract); (b) add a reviewed
class-identity table entry mapping the term to the polyprenyl-diphosphate class (cost: a ruling that the paper's
"prenyl diphosphate" IS that class, a verified ChEBI class id — not verifiable offline here — and the D-109
"map to a class" branch built and tested).

**Q2. Who grows `CLASS_PARTICIPANT_TERMS`, and on what evidence.** Candidates seen but not added (unmeasured):
`polyprenyl diphosphate`, `isoprenyl diphosphate`, `ubiquinone`, `menaquinone`, `acyl-CoA`. Cost per term: one
line and one test row; risk: refusing an honest class-level PathBank row (e.g. a generic "Acyl-CoA" row) — the list
refuses every match.

**Q3. The refusal flag in the post-run interactive curator.** No payload key was added. The refusal reuses the
existing `db_status = identity_refused_review_required`, which `strip_payload_for_interactive_context` does not
drop, so the user-triggered interactive curator sees that value on this row after a run (it already sees it for
every D-028 / C-078 refusal, and saw `matched` plus DMAPP here before). The D-109 reason itself is only in the
pre-freeze report. Option: drop `db_status` from that prompt (cost: an `interactive_curator.py` change and its own
byte-identity proof).

**Q4. The two compound gates still disagree in general.** Stage 6's name gate refuses a synonym-only PathBank
match with no shared token; pre-freeze D-028 admits it. This lane closes the disagreement for listed class terms
only. Persisting Stage 6's refusal onto the row (so C-078's veto sees it) would close it generally, at the cost of
changing `mapping_meta` for every name-gate refusal — which the audit LLM reads (`payload_for_prompt` keeps
`mapping_meta`) — so it needs a D-104-AMENDMENT-2 ruling first.

**Q5. `menaquinone (MK)` → Menadione** (wrong identity in `orch759_gold_accuracy.txt`) is a different defect (a
wrong molecule, not a class narrowing) and is not touched here.

## 8. Findings, failures and dead ends

* **F1 — the origin is pre-freeze, not Stage 6.** My first theory was a `map_ids` synonym hit (the committed cache
  holds exactly that answer). Disproved by the row's own `mapping_meta` (novel, `implausible_name_match`) and by the
  fixture's `stage6_gate_at_base`: Stage 6 refused DMAPP; the rename is stamped by `apply_compound_db_resolution`.
* **F2 — why C-078's veto did not fire.** `_admit_db_identity` vetoes a match that would restore
  `mapping_meta.rejected_mapped_ids`, but `map_payload` writes that key only when it strips ids from a row
  (`map_ids.py` shipped-identity path). A fresh row's Stage-6 refusal stays in the resolver result, so pre-freeze
  never sees it. Not changed (Q4).
* **F3 — no general lexical rule works** (§ 2): the target's recorded synonym IS the paper's term.
* **F4 — rejected: "synonym-only match must share a token".** It would refuse `DMAPP` → Dimethylallylpyrophosphate
  (the preservation case D-109 names) and `ferric iron` → `Fe3+`.
* **F5 — dead end in the replay.** The first base replay raised `PREFREEZE_CONNECTIVITY_BROKEN` (`06-replay-base`):
  my reconstruction reverted only `entities` and `processes`, so `element_locations` still named DMAPP and the
  rename looked like a new edge. Fixed by reverting the whole payload; `07` re-cut the fixture at base and its
  sha256 is unchanged (`dbb6fd48…56dc8`), `08` reproduces the leg exactly. `04-replay-tip-dev` used the first
  reconstruction and is superseded by `09`.
* **F6 — "prenyl-diphosphate" (hyphen) is not reachable through PathBank**: the resolver's `LIKE '%prenyl-diphosphate%'`
  finds no row, so there is nothing to narrow; dropped from the G9 parameters before the RED run.
* **F7 — test file edited after the RED commit:** two docstrings (one to stop claiming ChEBI/KEGG/HMDB naming that
  cannot be verified offline) and one added parameter (`prenyl pyrophosphate`). `05` re-proves RED for the final file
  on base `src/`.
* **F8 — ChEBI's own name for CHEBI:16057 is not verified** (offline lane). The Stage-6 G9 inputs are therefore
  labelled constructed; the rule does not depend on it.

## 9. The stale cache entry — what an operator must do

`data/id_mapping_cache.json` → `compounds["db::prenyl diphosphate::::{}"]` (committed; also in the primary checkout's
working copy): PathBank 874 DMAPP via `synonym`, 0.95. **No action is required, and it should not be deleted.** It
is an accurate record of what PathBank answers; Stage 6 gates it on every read (name gate, and now the class-term
refusal), and deleting it would only make the next run re-query PathBank and cache the same answer. Never commit a
cache edit. The upstream datum is PathBank's synonym on row 874; reporting it to PathBank curation is optional and
outside the sprint.

## 10. TEST_MATRIX delta (exact text for the orchestrator; this lane does not edit TEST_MATRIX)

```
### ORCH-764 — D-109 § 2, class names never narrowed

New focused selection: `tests/test_orch764_class_term_not_narrowed.py` (**16**, <1 s): the ORCH-759
menaquinone leg's MenA slice before pre-freeze resolution and its recorded PathBank answer
(`tests/fixtures/orch764/menA_prenyl_diphosphate.json`, cut by
`docs/pwml_recovery_sprint/evidence/orch764/make_fixture.py`), a SQL emulator for the PathBank compound
table, no network. G9: 9 red at base `src/` (`g11/ORCH-764/41`), 16 green at tip (`42`). Run it for any change
to `compound_resolution._db_match_rule_decision`, `_withhold_class_identity`, `CLASS_PARTICIPANT_TERMS`, `PathWhizCompoundResolver`,
`prefreeze_resolution.resolve_compounds_prefreeze`, or the compound branches of
`map_ids._apply_name_plausibility_gate` / `_enforce_shipped_identity_names`.

SMOKE unchanged at **508** (`g11/ORCH-764/22`). No pin moved. Known failures unchanged (§ ORCH-754 five, red on
base `src/` in `g11/ORCH-764/24`).
```

## 11. Files

| file | what |
|---|---|
| `src/t2pw/pwml/compound_resolution.py` | the class-term list, predicate, pre-freeze refusal |
| `src/t2pw/mapping/map_ids.py` | the Stage-6 refusal at both compound gate sites |
| `tests/test_orch764_class_term_not_narrowed.py` | G9, fidelity and preservation tests; `FixtureCompoundDb` |
| `tests/fixtures/orch764/menA_prenyl_diphosphate.json` | recorded PathBank answer and row 874 (+ one reconstructed `synonyms` entry, documented in `_provenance`), the leg's Stage-6 cache answer and its base gate output, the pre-freeze MenA slice |
| `docs/pwml_recovery_sprint/evidence/orch764/make_fixture.py` | cuts the fixture (prints source and output sha256) |
| `docs/pwml_recovery_sprint/evidence/orch764/replay.py` | the § 5 replay and scoring |
| `docs/pwml_recovery_sprint/evidence/orch764/*.log`, `evidence/g11/ORCH-764/01-25`, `evidence/g11/pin/ORCH-764/` | bounded-run logs, G11 reports, pin verdicts |

## 12. Merge of 762 round 1

At the orchestrator's request the ORCH-762 repair-round-1 tip `0a9f9829` (which carries the ORCH-759a repair
`cea94c91`) was merged into this branch: merge commit `c1b89161`, `git merge --no-ff -F <file>`, **no conflict**.
`git diff 0a9f9829 c1b89161 -- src` is exactly this lane's delta (+44 −4 in the same two files; the predicate import
is `map_ids.py:44`, the wrapped gate calls `:6886` and `:7129`, the helper `:6902`). The merged range includes
ORCH-762's own merge commit `dc4a3c3e` whose message reads `tmp` (recorded in ORCH-762-RESULT.md § 12); it is
not this lane's and is unchanged.

| rerun after the merge | G11 | result |
|---|---|---|
| G9 file | `26-m1-g9` | 11 passed |
| offline replay (both legs) | `27-m1-replay-tip` | identical to `09`: menaquinone final 8/9 (P 1.000 R 0.889 F1 0.941), enterobactin 5/5, Stage 1 9/9 and 5/5 |
| `test_orch762_protein_identity.py` + `test_orch759a_duplicate_protein_fold.py` | `28` | 73 passed |
| `test_orch759a_fold_app_path.py` alone | `29` | 2 passed |
| SMOKE | `30-m1-smoke` | **508 passed** |
| affected (same 176 files; 5 batches, gap files last, c055 / quarantine_boundary / c052 alone) | `31`-`39` (+ `29`) | **4427 passed, 6 skipped, 5 failed**: the same 5 known failures, assertion lines identical to base `src/` (`24`; md5 of the sorted `E` lines equal) |

**MenD (R2) is not recovered by this replay, and it cannot be.** The replay starts from the leg's recorded
`final_mapped.json`, after Stage 6 had already quarantined `rxn_lock_007` for `protein_missing_external_identity`;
it re-runs only pre-freeze compound resolution, never protein mapping, so ORCH-762's change to the MenD lookup is
not on its path. Final 9/9 on this leg would need Stage 6 re-run over the ORCH-759 leg with ORCH-762's recorded
UniProt answers; ORCH-762's own replays are where MenD's identity is measured (ORCH-762-RESULT.md § 4, MenD →
Q9CHK0 on the KF147 replay). Not built here: outside this lane.

`g11_evidence.py check --task ORCH-764` after these reruns: 39 artifacts, 0 non-compliant, every pytest report pinned.

## 13. Repair round 1

**Review round 0: REJECT, finding 1 (standard item 1).** A class is still narrowed when a member's identifiers are
already on the row. The reviewer's probes: 42 at the tip, 04 at base, 43 for Stage 6 then pre-freeze. A
"prenyl diphosphate" row carrying DMAPP's `pathbank_compound_id` 874 took the legacy-id branch
(`compound_resolution.py`, `_db_id` → `_first_nonempty`, which reads the row scalar, `mapped_ids` and
`mapping_meta.candidates[0]`) before the class check. That covers three shapes: the id as a scalar, in `mapped_ids`,
or only on the top candidate. With KEGG/HMDB ids the D-109 refusal was record-only, so C00235 / HMDB0001120 stayed
on the row, and the writer emits them. Stage 6 does not cover it: it strips scalars only partly, and
`run_pwml_export` has no Stage 6 at all. The § 2 limit claiming otherwise was wrong and is corrected there.

**Fix (`043626fa`, `compound_resolution.py` +49 −1).** `_withhold_class_identity` runs inside
`_resolve_compound_rows` BEFORE `_normalize_compound_external_ids` and the legacy-id branch, on a row whose `name`
is a listed class term. It moves every member identifier into `mapping_meta.rejected_mapped_ids`, which is the
review record Stage 6 writes and C-078's veto reads:

- top-level scalars (every `_REFUSABLE_NAMESPACE` key);
- `mapped_ids`;
- a resolved `db_row`;
- the same scalars on `mapping_meta`;
- the top candidate's ids.

The candidates themselves move to `rejected_candidates`. The row then takes the resolver path, where the D-109
refusal makes it record-only with `identity_refused_review_required`. Each withholding is logged in the resolution
report, `unresolved.class_term_identities_withheld`. Every export seam runs pre-freeze resolution, so no Stage-6
change was needed. Unchanged: the closed list, Stage 6, every non-class row.

| run | G11 | result |
|---|---|---|
| tests first (`972d0d32`) on round-0 tip code | `40-r1-g9-red-round0-tip` | **4 failed**, 12 passed: all four `test_g9_r1_*` cases (scalar, mapped_ids, candidate_only, kegg_hmdb) fail with identifiers on the row |
| same tests on base `src/` (`0a9f9829`) | `41-r1-g9-red-base-src` | **9 failed**, 7 passed: the four new cases (three ship 874; kegg_hmdb is renamed to DMAPP) and the five round-0 G9 cases |
| GREEN | `42-r1-g9-green` | 16 passed (new preservation test: a DMAPP-named row carrying 874 is unchanged, legacy-id as before) |
| offline replay | `43-r1-replay-tip` | identical to `09`: menaquinone final 8/9 (P 1.000), enterobactin 5/5. The leg's row carried no id, so nothing was withheld there |
| `test_orch762_protein_identity.py` + `test_orch759a_duplicate_protein_fold.py` | `44` | 73 passed |
| `test_prefreeze_third_export_seam.py` alone | `45` | 12 passed |
| SMOKE | `46-r1-smoke` | **508 passed** |
| compound-affected (176 files) | `47`-`56` | **4432 passed**, 6 skipped, 5 failed (+5 passed = the new tests). The 5 known failures have assertion lines identical to base (`24`, md5 equal) |

Every report shows `FINAL SURVIVING COUNT : 0`.

New limit, stated in § 2: a class row that arrives already carrying a `db_status` keeps that recorded account (the
D-015 fixed point restores it), although its identifiers are withheld. It is not settled whether the account should
then read `identity_refused_review_required`; that would need a change to `prefreeze_resolution` (Q6).

**Q6 (new, for the product owner).** Should a withheld class row's restored `db_status` be overwritten? The cost is
an exception to D-015's "the payload's own account stands", in `prefreeze_resolution._authoritative_provenance`.
