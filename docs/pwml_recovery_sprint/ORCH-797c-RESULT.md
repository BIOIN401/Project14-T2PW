# ORCH-797c RESULT: name-key follow-through (c011 delta, one key for identity routing, tRNA item)

Branch `orch797c/fix` off the integration tip `5a66b2f5` (main + 797b + 797a). Code commit `2f037515`. Not merged, not pushed. G11 task `ORCH-797c`.
**Merge rule 5 is waived (D-120 § 5): no reviewer round.** Decisions are from the lead's brief, ruling on `ORCH-797a-RESULT.md`
§ 9: 9.1 → (a), 9.2 → (a) keep, 9.3 → unify, 9.4 → only if small.

## 1. What was done

| item | result |
|---|---|
| 9.1 c011 delta | **Done.** `_with_c797_reference_key` is registered as one-way card `C-797`, and c011 is green (8 passed, G11 `11`) |
| 9.2 conflicting spellings | Kept as built (per-row omission). No change |
| 9.3 one key for identity routing | **Done for the routing sets. `map_ids._normalize_name` itself keeps its old fold** (stop condition, § 3). `_named_in` is dropped |
| 9.4 compound-only tRNA | **Not done** (§ 5) |

## 2. c011 (9.1): the measured move

The fixture was compared leg by leg against two trees: the merge without ORCH-797a (`8b0fd98d`, its `src` extracted to
`C:/t/orch797c/pre797a`) and with it (`5a66b2f5`). Probes: `evidence/orch797c_c011legs.py`, `orch797c_c011sets.py` and `orch797c_c011payloads.py`;
G11 `02`-`05` and `10`; flat output in `evidence/orch797c_evidence.txt`.

- **C-057's 7 legs are unchanged.** ORCH-797a moves the frozen payload digest on **8 legs** and one connectivity block. Nothing
  else moves on any of the 39 legs; C-057's helper still asserts that.
  - Two of the 8 are also C-057 legs: `PMC12444477`/strict and `2026-08-02_2130 PMC12096016`/strict. C-057's helper states
    their digest, and it still counts exactly 7.
  - The other 6 are stated by C-797: `PMC12180156` research and strict; `PMC12444477`, `PMC12657337`, `PMC12856317` and
    `PMC13233763` research.
- **What moved, per leg** (frozen payload, pre-797a against 797a):
  - `lipid IV A` now folds into `lipid IV_A`, and references are respelled to the row.
  - `CPgenIII` and `PPIX` fold into their spaced spellings.
  - `dihydroxyacetone phosphate (DHAP)` folds into `dihydroxyacetone-phosphate (DHAP)`.
  - `Iron` is respelled `iron` to match its row. `2,3-diDHB` is respelled to its row.
  - `δ-aminolevulinic acid` is no longer folded into `aminolevulinic acid`.
  - On `PMC12657337`/research, four junk rows written with spaced letters (`I P P`, `N A D H`, `N A D P H`,
    `G l u c o s e`) fold with `IPP`, `NADH`, `NADPH` and `glucose`. The first spelling wins (Not settled 2). G11 `09`.
- **Connectivity, `PMC12856317`/research: 27 → 28 nodes** (`n_nodes` and `main_component_size` only). This leg's ALA is
  written both `δALA` and `ALA`. **δ-ALA is 5-aminolevulinate, the same molecule.** The old keys stripped `δ`, so the old graph
  merged the two correctly, by accident. The new key spells the letter out and splits them. The same stripping made
  `α-ketoglutarate` == `β-ketoglutarate`, so D-120 § 1 stands and the split is the stated move.
- **Pins moved in `test_c011_freeze_seam_golden_equivalence`.**
  - The three equivalence tests apply `_with_c797_reference_key` innermost.
  - The standing one-way set changes from `["C-057"]` to `["C-057", "C-797"]`.
  - The override is now `C-057,C-797`; `C-057` alone is now a wrong value.
  - The registry set adds `C-797`.

## 3. One key for identity routing (9.3)

- `entity_identity._normalize` now returns `reference_key`.
- The two `map_ids` sets that `entity_identity` reads are now built with `reference_key`:
  - `_collect_protein_like_names`, read only by `route_entity_for_mapping`;
  - the `compound_names` set in `_apply_pathbank_unknown_enzyme_fallback`, read only by `compound_name_block_rule`.
- `_named_in` (ORCH-797a's bridge) is removed. Every caller now folds its sets with the one key.
- **Stop condition held for `map_ids._normalize_name` itself, which keeps its old fold.** It builds cache keys
  (`_AI_ALIAS_CACHE_PREFIX::…`, `ncbi-taxonomy-v1::…`), query tokens (`.split(" ")`, `.replace(" ", "")`), Jaccard scores
  and substring tests against DB names. Moving it would change UniProt/PathBank cache keys and query strings. Its other 200+
  uses do not decide entity-row membership.
- **Identity outcomes, measured.** `evidence/orch797c_identity_routes.py` records, per leg, every compound row's
  `route_entity_for_mapping` decision as `map_payload` asks it, plus the full output of
  `_apply_pathbank_unknown_enzyme_fallback`. It ran over the 6 ORCH-796 legs (`merged_payload` and `final_mapped`) and all
  122 committed `final_mapped.json`. Base `5a66b2f5` against tip: **0 differing lines** (G11 `13`, `14`).
- **G9** `tests/test_orch797c_identity_key.py`:
  - `IkappaB` is recognised as the declared protein `IκB`. Base: `default_compound_route`.
  - A compound `IB` is no longer routed as protein `IκBα`. Base: both folded to `ib`, a wrong route.
  - One preservation test.
  - **Base: 2 failed, 1 passed** (G11 `15`). **Tip: green** (`16`).
- **Pin moved.** ORCH-797a's `test_new_acceptance_identity_routing_reads_either_fold` is renamed `…_reads_the_reference_key`.
  It now asserts that the old `map_ids` fold (`"lipid iv a"`) is **no longer** accepted.

## 4. Proof

| what | result | G11 (`evidence/g11/ORCH-797c/`) |
|---|---|---|
| G9 797c, base `C:/t/orch797c/base` (file copied in, removed, base clean) | 2 failed (both `test_g9_*`, on values), 1 passed | `15-g9-base-*.json` |
| tip: 797c + 797a + 797b G9 files + c011 | **39 passed** | `16-g9-tip-*.json` |
| affected: 126 files (every file importing `map_ids`, `entity_identity`, `process_normalizer` or `strict_quarantine`, plus 797b's `evidence/orch797x/affected_files.txt` and 797a's list; AppTests excluded) | **2942 passed, 11 skipped, 8 failed**, all failing identically at base `5a66b2f5`. 6 are the known ones (c075 corpus, compound-resolution golden-coverage ×2, paper eligibility, prefreeze third seam, rag merge base). 2 more not in 797a's list: `test_c100_tolerance_scope::test_5b…` and `test_stage2_mapping_boundary::test_compatibility_default_keeps_stage6_structural_cleanup`, both red at base (`24-two-on-base.json`; 797b reported 6 pre-existing in its own batch) | `17-`..`20-affected-*.json`, `24-` |
| AppTest alone: reaction-evidence view, app hook, streamlit quarantine boundary | 11, 1, 24 passed | `21-`, `22-`, `23-` |
| c011 | **8 passed** | `11-c011-r1.json` |
| six-leg tail replay, base `5a66b2f5` vs tip | identical: 6/6 write a PWML, **all 12 byte-identical to the leg's own**; drawn/total unchanged (chl 52/62, wnt_ca 22/61, wnt_pcp 9/58, sour 8/13, entero 7/9, mena 14/14). No row and no PWML lost | `25-replay-base.json`, `26-replay-tip.json` |
| `g11_evidence.py check --task ORCH-797c` | exit 0, 26 artifacts, 0 non-compliant | `27-g11-check.json` |

## 5. Not done: compound-only tRNA (9.4)

Not a small change. Re-kinding a compound row to a nucleic acid pre-freeze needs four things:
- a rule for its compound identity fields (KEGG/ChEBI ids have no place on a PathWhiz nucleic acid);
- moving its `compound_locations` rows to `nucleic_acid_locations`;
- a reason code for the text record;
- new pins.

Every reference to such a row still resolves to its one row, so nothing is unresolved today. Options:
- (a) leave it;
- (b) a lane: about 30 lines, a G9 test and a KI row, plus a decision on dropping the compound ids.

## 6. Failures and dead ends

1. **Wrong first reading of the c011 failure.** ORCH-797a's RESULT said "28 → 27". The measured direction is **27 → 28**: the
   fixture and pre-797a have 27 nodes, 797a has 28. Only the connectivity block was attributed then; the 6 extra moved digests
   were not. Both are now measured and stated.
2. **The first identity-route probe hashed the wrong object.** It hashed the fallback's return value (a summary) rather than the
   payload it mutates, so two legs showed the same hash. Fixed before the comparison (G11 `12` superseded by `13`).
3. **Rejected: re-key `map_ids._normalize_name` wholesale.** It is a cache-key and query-string function, which the stop
   condition excludes (§ 3).

## 7. KI delta, LLM-side

No new reason code.

LLM-side: Stage 1 emitted spaced-letter junk rows (`I P P`, `N A D H`, `G l u c o s e`) on the archived
`2026-07-28_2122 PMC12657337` research leg.

## 8. Not settled

1. **9.4 compound-only tRNA** (§ 5).
2. **The first spelling wins when two spellings fold** (`I P P` over `IPP`), in `process_normalizer._dedupe_named_rows` and in
   ORCH-797a's quarantine fold. Option: prefer the spelling the processes reference, or the one with the fewest spaces. Cost:
   one tie-break rule plus pins. Today the kept name can be the uglier one, but every reference resolves.
3. **`map_ids._normalize_name` keeps the old fold** for cache keys and scoring (§ 3). It could be unified only with a cache-key
   migration.

## 9. Proposed LEDGER row

`| ORCH-797c | name-key follow-through: c011 C-797 delta; entity_identity + map_ids routing sets on reference_key, _named_in dropped | orch797c/fix | built, tested, replayed; rule 5 waived | G9 15 red / 16 green; c011 8 passed; affected 2942 passed, 8 pre-existing; routes/fallback 0 diffs (6 legs + 122 corpus); replay 6/6 PWML byte-identical | ORCH-797c-RESULT.md |`
