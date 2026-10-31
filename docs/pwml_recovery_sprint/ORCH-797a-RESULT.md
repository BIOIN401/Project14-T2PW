# ORCH-797a RESULT: one reference key for every stage; tRNA is a nucleic acid (D-120 §§ 1, 4)

Branch `orch797a/fix` off `aefa5b30`. The code commits are listed in § 10. Not merged and not pushed. G11 task id: `ORCH-797a`.
**Merge rule 5 is waived by the PO (D-120 § 5): no reviewer round.**

**Status: done, except for one pinned fixture.** Three tests in `test_c011_freeze_seam_golden_equivalence` stay red. They are
a deliberate behaviour move that this card cannot register in c011's delta registry (§ 5.2, Not settled 1). Every other affected
test passes or fails the same way on base.

History: the first session (`effba332`, `fa7831ba`) was blocked by low RAM and could not run anything. After a reboot, this session
ran the full proof, and that proof exposed five defects in the first cut (§ 6). All five are fixed here.

## 1. Cause

Before this card, the stages that resolve references keyed names differently:

| key (pre-D-120) | `tRNA^Glu` | `5-HT 3 receptor` | `NAD+` | `α-ketoglutarate` |
|---|---|---|---|---|
| `process_normalizer._normalize` / `reference_repair._normalize` (quarantine, admission registry) | `trnaglu` | `5ht 3 receptor` | `nad+` | `ketoglutarate` |
| `ir._norm` / `prefreeze_resolution._norm` / `compound_resolution._norm` / `reaction_evidence._ir_norm` | `trna glu` | `5ht 3 receptor` | `nad+` | `ketoglutarate` |
| `entity_admission._norm` | `trnaglu` | `5ht3receptor` | `nad` | `ketoglutarate` |

- **Admission versus the IR.** The quarantine admitted `tRNAGlu` against the row `tRNA^Glu`, but the IR did not resolve it, so the
  whole pathway was refused. This is the ORCH-796 pass-1 chlorophyll failure (ORCH-794 § 11). G9 test 1 reproduces it on base:
  `ir:unresolved_entity_reference`.
- **Greek letters were stripped, not spelled out.** Every old key made `α-ketoglutarate` the same name as `β-ketoglutarate`. This
  latent wrong merge was found while moving the pins (§ 5.2). D-120 § 1 spells the letter out, which fixes it.

## 2. Change

| file | function | change |
|---|---|---|
| `pipeline/name_keys.py` | `reference_key`, `_spell_greek` (NEW) | NFKC, unicode dashes, any Greek letter spelled out (`α`->alpha, Unicode `LAMDA`->lambda, accents ignored), casefold, then everything outside `[a-z0-9:+]` removed. Digits, `+` and `:` stay identity-bearing. The key is idempotent and a strict coarsening of the old quarantine key. `normalize_name`/`strict_key` are untouched (twinned with the evaluator) |
| `process_normalizer._normalize`, `reference_repair._normalize`, `entity_admission._norm`, `ir._norm`, `prefreeze_resolution._norm`, `compound_resolution._norm`, `reaction_evidence._ir_norm` | | each returns `reference_key(value)`, so every caller uses one key |
| `pipeline/strict_quarantine.py` | `_spell_references_canonically` (new), `_spell_references_as` (generalised), `_identities_conflict`, `_identity_values` (new) | Runs pre-freeze, before the registry. (a) Every process, location or complex-component reference that names an entity row only under the shared key is rewritten to that row's exact name. A key held by rows of two kinds is not rewritten; admission omits each such row (KI-011). (b) Two rows of ONE kind spelled differently under one key: if their identities agree they fold into the first row, and the rest are removed as `same_entity_name_form:<bucket>` (KI-039). If their identities conflict, every member is omitted per row (`duplicate_canonical_row_identity_not_proven`, KI-015). Without (b), the new key would make the IR refuse the whole pathway post-freeze (`DuplicateNamedRowError`). Free-text fields are never rewritten |
| same | `_paper_kind`, `_keep_one_kind_per_name`, `_split_gene_protein_names`, `_TRNA_WORDING` | D-120 § 4: a tRNA declared in two kinds keeps its nucleic-acid row, before any identity rule (`decided_by: trna_is_nucleic_acid`). A tRNA nucleic acid that clashes with a protein is never renamed `<name> gene`; the protein row is removed instead (KI-051) |
| same | `_term_matches`, `_scored_term_matches` (new) | The core-term match re-keys the process terms (idempotent). The coverage SCORE accepts a match under either the D-120 key or the pre-D-120 fold, so an archived report, whose core terms had their Greek letters stripped, still replays to its own verdict. The score is never narrower than base and never wider than base plus D-120 |
| `pwml/prefreeze_resolution.py` | `_reject_duplicate_canonical_rows`, `_moved_fold` (new) | Whether a row "arrived" (canonicalisation moved its name) is judged on the pre-D-120 fold. Two spellings the new key already joins (`lipid IV_A` / `lipid IV A` -> `Lipid IVA`) are therefore still consolidated or omitted here, and never reach the IR as two rows with one name |
| `pwml/compound_resolution.py` | `_abbreviation_length` (new) | The D-028 short-abbreviation gate measures length on the pre-D-120 fold, so no name crosses the 4-character gate (no gate moves) |
| `pwml/ir.py` | `SPECIES_CREATE_DEFAULTS` | The table key is re-spelled in the new key (`narcissusspaffpseudonarcissus`). The literal old-format key had silently stopped matching, so the Narcissus create-default was lost (§ 6) |
| `pipeline/entity_identity.py` | `_named_in` (new) | Pre-folded name sets are matched under either fold: `map_ids` still builds them with its own `_normalize_name`, while `process_normalizer` now builds them with the new key. `entity_identity._normalize` and `map_ids` themselves are unchanged (integration owns them) |
| `pipeline/entity_admission.py` | `_core` | The currency check keeps `VOCABULARY`'s own fold, so `NAD+` stays currency |
| `app/reaction_evidence.py`, `PWML_KNOWN_ISSUES.md` appendix | `KNOWN_ISSUES` | new code row `same_entity_name_form:<bucket>` -> KI-039 |
| `tests/test_orch797a_reference_key.py` | new | 4 G9 tests, 2 preservation tests, 5 labelled NEW ACCEPTANCE tests |

`db_resolver._name_key` is left alone. It matches against PathBank DB names (compound identity), not a reference to an entity
row, and it drops `+`.

**Decision on `+`: it stays identity-bearing.** On the six legs the audit shows that dropping `+` would merge `H`/`H+`, `K`/`K+`,
`Ca2`/`Ca2+`, `Fe2`/`Fe2+` and `Mg2`/`Mg2+` (`orch797a_audit.txt`). `H` against `H+` is not safely one entity.

**Dropped from the first cut:** a `,` kept between two digits (`1,12` != `11,2`). Without it the new key is a strict coarsening of
the old quarantine key, which already merged those spellings. With it, archived old keys cannot be re-keyed. The corpus audit
found no pair that it would separate.

## 3. Proof

| what | tree | result | G11 (`evidence/g11/ORCH-797a/`) |
|---|---|---|---|
| G9 `test_orch797a_reference_key.py` | base `C:/t/orch797a/base` (`aefa5b30`), test copied in and removed | **9 failed, 2 passed**: the 4 `test_g9_*` fail on values (`ir:unresolved_entity_reference`; `undeclared_entity_in_entity_2`; the compound kept; `tRNAGlu gene`); the 5 new-acceptance tests fail (3 ImportError of the new function, 2 on the new fold); the 2 preservation tests pass | `34-g9-base-final2.json` (also `01-`, `25-`) |
| same + `test_orch772_pwml_coverage`, `test_orch794b_spelling`, `test_orch794_one_name_two_kinds` | tip | **54 passed** | `33-g9-tip-final2.json` |
| affected: every non-AppTest test file importing a changed module (110 files, 4 chunks, final code) | tip | **2590 passed, 11 skipped, 9 failed**: 6 fail the same on base (`12-failing-on-base-r2.json`: c075 corpus, compound-resolution golden-coverage ×2, paper eligibility, prefreeze third export seam, rag orchestrator merge base); 3 are c011 (§ 5.2) | `21-`..`24-affected-final-c0*.json` |
| AppTest, each alone: `test_reaction_evidence_view`, `test_reaction_evidence_app_hook`, `test_streamlit_quarantine_boundary` | tip | 11, 1, 24 passed | `27-`, `28-`, `29-` |
| collision audit, six ORCH-796 legs (brief item 5) | tip | **0** name groups merged by the new key and not by the old keys; **0** quarantined rows recoverable | `30-audit-final.json`; `evidence/orch797a_audit.txt` |
| corpus collision audit, all 122 committed `final_mapped.json` | tip | **1** group: `sn -glycerol 3-phosphate` / `sn-glycerol 3-phosphate` (the same molecule) | `31-pinprobe-tip-final.json` |
| six-leg tail replay (each leg's own `final_mapped.json`, unmodified) | base / tip | identical: all 6 write a PWML; **all 12 PWMLs byte-identical to the leg's own** (`pwml_matches_leg: true`) | `05-replay-base-r1.json`, `32-replay-tip-final.json`; `evidence/orch797a_replay.txt` |
| orch766 pin components | base / tip | payload, removed, closure, and the quarantine report minus key-text fields are byte-identical on all 27 kinds | `14-`, `15-pinprobe-*.json` |
| golden IR leg PMC12444477 | base / tip | the only difference is one synonym deduped (`Kdo-lipid IVA` vs `KDO-lipid IV(A)`) | `17-`, `18-goldenprobe-*.json` |
| `g11_evidence.py check --task ORCH-797a` | | exit 0 | `35-g11-check.json` |

**Replay: rows recovered per paper = 0 on all six.** These draws contain no reference that differs from its row only by format (the
audit found none). Their spellings already agreed, and they write the same PWML at base and tip. The defect shape is proven by the
G9 tests instead: the ORCH-796 pass-1 chlorophyll shape, and `5-HT 3` against `5-HT3`.

## 4. Leg replay table

| leg | base PWML | tip PWML | drawn/total base -> tip | PWML bytes |
|---|---|---|---|---|
| chlorophyll `_1632` | yes | yes | 52/62 -> 52/62 | = leg |
| wnt_ca `_1941` | yes | yes | 22/61 -> 22/61 | = leg |
| wnt_pcp `_1729` | yes | yes | 9/58 -> 9/58 | = leg |
| sour_taste_v2 `_1314` | yes | yes | 8/13 -> 8/13 | = leg |
| enterobactin `_1153` | yes | yes | 7/9 -> 7/9 | = leg |
| menaquinone `_1451` | yes | yes | 14/14 -> 14/14 | = leg |

## 5. Pins moved (merge rule 4), exact deltas

### 5.1 Moved and green

- **`test_orch766_unreferenced_species`, `PINS` (all 28).** Only the quarantine report's key-text fields moved: core terms and
  context core terms are now in the new key. All four components minus those fields are byte-identical (§ 3). Old -> new (first 16
  hex):
  `state_species` 5473ae762224adda->42e7c8ca79c87aa7, `state_organism` d1a653fd7ad29b61->6c322b23d01268af,
  `state_name` b9b75775ed0d8711->9b8c582965ceacac, `protein_species` 217e6725db50425c->97d31e50d9bfd4cb,
  `protein_organism` faaa2f603422f372->48eb53ef98941c6c, `protein_species_ref_name` 1127646fccbebfb1->60070b71652ad13b,
  `protein_mapping_meta_species` a2c5a41ad85c5307->c248f8aa5e035cae, `complex_species` 0ea864dbf298033f->68af94d31ba9d31d,
  `complex_organism` 117a13ff363177c7->a07648c205cb8ecb, `other_entity_species` 40a5d26c6c59b693->59848cca3ccaecaa,
  `location_row_species` 87121b45c0e13031->ab6b5819e73ea295, `pathway_metadata_organism` 47646d63cdae5201->180ebca37067f3e6,
  `surviving_process_field` 645cd7dbfbda5210->158d8a41f6d0086d, `alias` fd97ee0fe58bca3e->afe2e047be567544,
  `synonym` 1c363a105810f035->c61100abc184f807, `raw_name` 5dd910d5a1781103->50362983d670a2d6,
  `case_and_punctuation_variant` 1722f67370ff6d61->62c8ec9ca9cf2984, `protein_taxonomy_id` a9e624919b6d89c6->b41f47767ab7dfe9,
  `protein_species_id` 8d67b85e9db391fa->88cd97ddc10ce5d6, `protein_species_ref_ids` d7e44be2c93ca75e->6a8b7b1aeccb1eb1,
  `state_taxonomy_id` 25e894078c4e296c->8245a6733bfb71ed, `species_id_as_float_text` 06663f82309cfbad->cbaff37cecfa22be,
  `species_id_non_integral_float` a6089f7032ea91b6->8e036922090ab998, `species_id_non_integral_text` 2cbf0ae95e9b7b63->66653bde626002dd,
  `create_default_name` 1b9cf4685413e4c7->0fccf968036b093b, `first_row` 9163c511d40de910->0b721eb01f56141a,
  `requested_organism` 96c85ea8ad1ca7ad->4f25a124db191559, `base` df397a5102c870ee->7ee1e24baf7e35b7.
- **`decision_input_hash`** (context core terms in the new key). `test_orch759c` `6db09994c3616bb6`->`6efc8c92d0f65c07`;
  `test_orch769b` `a5cf19b0eebf9c75`->`5ee0f01968ff79f3`. All later assertions of both tests pass unchanged.
- **`test_pwml_ir_duplicate_row_refusal`, expected key text.** `"lipid iv a"`->`"lipidiva"` (×6 asserts, ×6 parametrize rows) and
  `"escherichia coli"`->`"escherichiacoli"`. The refusing pairs and the refusals are unchanged.
- **`test_compound_resolution_extraction`.**
  - `GOLDEN[...PMC12444477]` `268d1912...e2523`->`9f68387d...e8dc`: one synonym deduped (§ 3).
  - Import set + `t2pw.pipeline.name_keys`.
  - The `_norm` copies are source-identical again.
- **`test_compound_db_match_admission`.** The length assertion now reads `cr._abbreviation_length`. The lengths are unchanged
  (7, 20, …).
- **`test_prefreeze_compound_resolution`.**
  - c050f `"α"`->`"·"`: α now keys as `alpha`, so it is no longer an empty-norm name.
  - Key text `"glycerol 3 phosphate"`->`"glycerol3phosphate"` and `"sn glycerol 3 phosphate"`->`"snglycerol3phosphate"` (×5).
  - c050g consolidate: synonyms `[…, "sn-glycerol 3-phosphate"]` -> `["sn -glycerol 3-phosphate"]`. The two spellings are now one
    alias, deduped.
- **`test_prefreeze_species_resolution`, two strains.** The second strain was `subsp.`->`subsp`, which D-120 makes the SAME name.
  It is now `KF147`->`IL1403`, so the guard under test still sees two different strain names. The assertions are unchanged.
- **`test_rag_graph_delta`.** The element ids carry the key: `[second step]`->`[secondstep]`, `[core step]`->`[corestep]`.
- **`test_rag_seed_entity_reimport`.**
  - `"lipid iva precursor"`->`"lipidivaprecursor"`.
  - The spacing test: `lipid IV A` against the base's `lipid IV_A` is now dropped as already present (D-120 § 1), so the
    expected additions lost `"lipid IV A"`.
- **`test_orch772_pwml_coverage`.** Appendix 90->91 codes. New label case `same_entity_name_form:compounds`->KI-039 (NEW
  acceptance).

### 5.2 Moved and NOT green: `test_c011_freeze_seam_golden_equivalence` (3 tests)

The research leg `runs/2026-07-28_2122/…PMC12856317…/research` changes connectivity: `main_component_size` and `n_nodes` go from
28 to 27. The cause is attributed, not traced node by node. The leg names `δALA` (`succinyl-CoA -> δALA`) beside `ALA` /
`5-aminolevulinate (ALA)`:
- Every old key stripped `δ`, so `δALA` == `ALA`, a lucky merge (it is the same molecule).
- D-120 spells the letter out, so `δALA` (`deltaala`) != `ALA`.

This is the same rule that stops `α-ketoglutarate` == `β-ketoglutarate`.

The c011 fixture is a historical BEFORE document. It may move only through a per-card delta registered in `_DELTAS`, and the
registry accepts only helper names of the form `_with_c<digits>_*`. I did not add a delta under an invented `C-` number. See Not
settled 1.

## 6. Failures, dead ends, wrong theories

1. **The first session was blocked** on RAM below 3 GB (Docker/WSL, MATLAB). There was a reboot between sessions.
2. **The first cut failed 83 affected tests.** Five of them were real defects, now fixed:
   - (a) `SPECIES_CREATE_DEFAULTS` was keyed by a literal old-format key, so the Narcissus create-default silently stopped matching
     (also the cause of 6 c050j failures).
   - (b) Same-kind spellings that are equal under the new key would be refused post-freeze (`DuplicateNamedRowError`, the whole
     pathway). Fixed by the fold in § 2.
   - (c) Prefreeze's "arrived" test, read on the new key, skipped consolidating `lipid IV_A` / `lipid IV A` and left two rows named
     `Lipid IVA`.
   - (d) Archived coverage reports stopped replaying (their Greek letters were stripped).
   - (e) The `compound_resolution` copy of `_norm` drifted from its IR original.

   The other failures were key-text pins (§ 5.1).
3. **Wrong theory: "the d088 failures are only key text".** Re-keying archived terms fixed the space-only cases. The Greek-letter
   terms (`…3β-ol`, `14α-demethylase`) still failed, because the old key had deleted the letter. Hence the two-fold score.
4. **The kept separator between digits (`,`) was a mistake.** It made archived keys impossible to re-key, and it guarded against a
   collision the corpus does not contain. Dropped.
5. **Owned breaches.** One `python -c "print(1)"` (session 1) and one `python -I -c "print()"` (session 2) ran outside the wrappers.
   Both were no-ops, and no sprint job ran through either.

## 7. KI delta

- **New:** `same_entity_name_form:<bucket>` -> **KI-039** (name-form variance). It is the removed row of two spellings of one entity
  folded pre-freeze. Appendix row added.
- **Reused:** `duplicate_canonical_row_identity_not_proven` -> KI-015, for spellings whose identities conflict.
- **Reused:** `same_name_declared_as_another_kind:nucleic_acids` -> KI-051, with `decided_by: trna_is_nucleic_acid`.

## 8. LLM-side

None new. All six legs spell each reference exactly as its row.

## 9. NOT settled (for the PO / lead)

1. **c011 fixture (3 red tests).** Options:
   - (a) The lead registers a delta for this card. That needs a `C-`-style card name, or a widening of `_HELPER_RE`. Cost: about 40
     lines of test infrastructure.
   - (b) Retire the BEFORE baseline, which the module calls a product decision.
   - (c) Keep stripping Greek letters in the key. That goes against D-120 § 1 and re-opens `α-ketoglutarate` == `β-ketoglutarate`.

   Recommended: (a).
2. **Spellings with conflicting identities are now omitted per row** (new acceptance test). Base wrote them as two compounds of one
   name. Options:
   - (a) keep (correctness over coverage);
   - (b) keep the first row's identity. Cost: possibly a wrong id.
3. **`map_ids._normalize_name` / `entity_identity._normalize`** are still the old fold, bridged by `_named_in`. Unify them at
   integration with ORCH-797b, then drop `_named_in`.
4. **A tRNA declared only as a compound** is not converted to a nucleic acid. Every reference still resolves to its one row.
   Converting it would mean moving its locations across buckets after the kind was decided.
5. **The coverage score's legacy fold (`_scored_term_matches`)** keeps base's Greek-stripped matches. On a live run that includes
   base's α/β cross-match, for relevance scoring only (never identity). It can be dropped once no archived report needs replaying.

## 10. Commits and proposed LEDGER row

Code: `effba332` (first cut), `4ab21242` (proof fixes, pins). Docs: `fa7831ba`, then this record (the next commit). The final SHAs are in
`git log orch797a/fix`.

`| ORCH-797a | D-120 §§ 1, 4 one reference key; tRNA nucleic acid | orch797a/fix | built, tested, replayed; rule 5 waived (PO) | G9 34 red (4 G9 on values) / 33 green 54; affected 2590 passed, 6 pre-existing, 3 c011 pinned-fixture red (§ 5.2); replay 6/6 PWML byte-identical base and tip; collisions 0 (legs), 1 same-molecule (corpus) | ORCH-797a-RESULT.md |`
