# ORCH-783 — family / complex proteins become protein complexes (KI-001 subset)

Branch `orch783/family-complex` off `5e1edb3f`. Code commit `ee4586ce`. **Status: built, NOT TESTED.**
Nothing is merged or pushed.

## 1. What changed

| file | function | change |
|---|---|---|
| `src/t2pw/mapping/map_ids.py` | `is_family_or_complex_term`, `_paper_named_members`, `_uniprot_reviewed_hits`, `_apply_family_complexes` (new); `map_payload` (one call, before `_apply_pathbank_unknown_enzyme_fallback`) | the PO's member ladder |
| `src/t2pw/mapping/identity_admission.py` | `PROVENANCE_ROUTE_KEYS` | + `family_complex_provenance` |
| `src/t2pw/app/reaction_evidence.py` | `_mapping` | prints `family complex: <note>` when a row has one |
| `tests/test_orch783_family_complex.py` | new | 20 cases (14 parametrised detector cases + 6 tests) |

The rules as built:

- **Detector** (narrow, lexical): one of these holds:
  - the name is a slash member list (`STIM1/2`, `P2X2/P2X3`; not `PI3K/AKT`, not `Ca2+/calmodulin`);
  - the name carries family wording (family, subfamily, isoform(s), isozyme(s), subunits, paralogs);
  - the name is one of 17 known family terms (PKC, PLC, G protein, calcineurin, NFAT, VGCC, P2X receptor, 5-HT3 receptor, and their spelled-out and plural forms).

  NLK, Cdc42, TCF4 and PIF1 are not caught. UniProt is never asked to *confirm* a family: `family:"Cdc42"` would match the CDC42 subfamily.
- **Ladder**, stopping at the first source that yields members:
  1. members the paper names, taken from an existing same-species row or from UniProt `gene_exact:` with exactly one hit;
  2. a PathBank complex with exactly that name (`map_protein_complex`), keeping members whose taxonomy matches;
  3. UniProt `family:"<core> family"`, then `"<core> subfamily"`, reviewed, in the pathway's taxon. Only the most common 3-letter gene prefix is kept, which drops PKN1-3 from the PKC subfamily. Hits are sorted by gene name.

  Every source keeps at most **3** members, and every member must match the pathway's species. The source is recorded in `mapping_meta.family_complex.note`, for example "members from UniProt family search family:"PKC subfamily", first 3 of 9 …, not the paper".
- **Fallback**: no source yields members. The complex is created with no components. The existing `_apply_pathbank_unknown_enzyme_fallback` and `_apply_pathbank_unknown_complex_fallback` (D-070) then fill it with the Unknown member, for **any** role. The role-evidence gate on reaction enzymes still applies.
- The complex is `generated: true`, so the strict gates check every member's UniProt id and species (D-106 is unchanged). The family's protein row is removed, and reaction-enzyme and transporter actors are retyped to `protein_complex`. Network answers are cached in the mapping-cache section `family_complex_uniprot`. UniProt is skipped when `id_source="db"`.

Calibration probes (curl, 2026-10-27, human):
- `PKC subfamily` returns 9 PRKC entries and 3 PKN entries.
- `P2X receptor family` returns P2RX1-7.
- `Orai family` returns ORAI1-3.
- `NFAT family`, `NFAT subfamily`, `PKC family` and `P2X receptor subfamily` return 0.
- Bare `family:"G protein"` returns GNG and ADGR entries, and bare `family:"calcineurin"` returns PPP3R, CHP and TESC entries. Both are too loose, so the bare form is not used.

## 2. G9 / acceptance — run by the orchestrator, see § 5

| test | label | base `5e1edb3f` | tip |
|---|---|---|---|
| `test_g9_family_interaction_endpoint_gets_unknown_backed_complex` ("DAG activates PKC" through `map_payload`, offline) | **G9 behavioural** | expected FAIL: PKC stays a bare protein | expected PASS |
| detector ×14, paper-named, PathBank (cap and mouse dropped), UniProt (cap, PKN dropped, mouse dropped), admission route, NLK untouched | **new acceptance** | n/a | expected PASS |

**No proof report exists.** The free-RAM floor of 6 GB (`FreePhysicalMemory`) was never met. Readings over about 75 minutes of polling, 20 s apart, were 4.0-5.1 GB. They came from the user's applications, not from any sprint job: no sprint Python process was running and the heavy lock was free. The base tree is exported at `C:/t/orch783/base` with the test file copied in, ready for the run.

To do (one bounded heavy-lock job each, from the worktree root):
1. the focused file at the tip;
2. the same file in `C:/t/orch783/base` with the base tree's `pinned_pytest.py`; the G9 test must fail there;
3. the 47 non-AppTest test files that import `map_ids`, `identity_admission` or `reaction_evidence` (list: `C:/t/orch783/affected.txt`, 53 files minus 6 AppTest files: `test_c055_rag_loop_wiring`, `test_orch759a_fold_app_path`, `test_orch759b_recorder_reattach`, `test_reaction_evidence_app_hook`, `test_reaction_evidence_view`, `test_streamlit_quarantine_boundary`). Run the AppTest files singly.

Reports go to `evidence/g11/ORCH-783/` (`01-focused` is already reserved).

## 3. Dead ends

- **Approach: let UniProt family search detect families.** Rejected: it turns single proteins into families (Cdc42 matches the CDC42 subfamily).
- **Approach: bare `family:"X"` queries.** Rejected: too loose (see the probes above).
- **Approach: a new `generation_reason`.** Not possible: `payload_models.py:242` and `schema.py:270` pin the value to the literal `single_protein_pathwhiz_wrapper`, so the family complex reuses that name although it can hold 3 proteins.
- **Approach: put members only in `components`, with no protein rows.** Rejected: PathWhiz needs member proteins with UniProt ids, and `ir.py` checks components against protein rows only for generated complexes.

## 4. NOT settled (for the product owner / reviewer)

1. **Identity-admission route.** UniProt and PathBank members are not named in the paper, so pass A would withhold their ids and the complex would fail again. The new route key makes pass A abstain on them. A reviewer must decide whether this is a permitted provenance route under the PO's ruling or a gate weakening (merge rule 6).
   - Option A: keep the route.
   - Option B: abstain only when the family name itself is found in the source text. This needs a value-aware check, which `provenance_route` deliberately does not do.
2. **Fallback species.** The Unknown fallback for a family follows `_wrapper_species_fields` (D-070 O-1b). Unless the row's species was source-supported, the complex ships with *Arabidopsis*, the placeholder record's species.
3. **Gene-prefix filter.** The 3-letter gene-prefix majority is a heuristic. It is right for PKC, P2X and Orai, and unproven elsewhere.
4. **Which members are kept.** Members are the first 3 by gene symbol (PRKCA, PRKCB, PRKCD), not "the most relevant" ones.
5. **Untested effect on ORCH-781 rows.** Expected effect, not measured:
   - PKC, Orai and P2X get UniProt members;
   - STIM1/2 gets its paper-named members;
   - G protein, PLC, NFAT, VGCC and 5-HT3 receptor get the Unknown member;
   - calcineurin was already a PathBank complex.
6. **Base export not removed.** The base worktree `C:/t/orch783/base` is left in place for the G9 run.

## 5. Tests run (orchestrator, after the build; all bounded with `--heavy-lock`, 0 survivors)

The Lead lowered the RAM floor to 3 GB for normal jobs and 2 GB for short focused gates, so these ran at 3.8-4.2 GB free.

| report | tree | selection | result |
|---|---|---|---|
| `evidence/g11/ORCH-783/01-g9-base.json` | `C:/t/orch783/base` (5e1edb3f + the test file) | `tests/test_orch783_family_complex.py` | 20 failed. The G9 test fails **on its value** ("the family must not stay a bare, identity-less protein endpoint"), not on a missing symbol. The new-acceptance tests fail on a missing symbol, which is expected for new capability. |
| `evidence/g11/ORCH-783/02-focused-affected.json` | integration `9f7e5207` (782 + 783) | 12 files: both focused files, ORCH-759c, map_ids, mapping lineage, c073, c075, orch762, reaction_evidence, orch772, orch773, interaction wrappers | 315 passed, **2 failed**: (a) mapping-lineage base golden; (b) c075 corpus |
| `evidence/g11/ORCH-783/03-smoke.json` | integration `9f7e5207` | SMOKE, 22 files | **511 passed** |
| `evidence/g11/ORCH-783/04-tip-affected.json` | lane `f94291e2` | focused + mapping lineage + c075 | 87 passed, 1 failed (c075, see below) |
| `evidence/g11/ORCH-783/05-base-c075.json` | `C:/t/orch783/base` | `tests/test_c075_source_support_armed.py` | 1 failed, **identical message on base** |
| `evidence/g11/ORCH-783/06-smoke-final.json` | integration `cac97171` (final merged tree) | SMOKE, 22 files | **511 passed** |

Pass counts come from the piped wrapper logs in `C:/t/orch784/` and `C:/t/orch783/`, because the JSON reports carry no stdout.

**Failure (a), mapping-lineage golden. Fixed in `f94291e2`.** `map_payload` always wrote `family_complexes_unknown_fallback: 0` and `family_complexes_with_members: 0` into the summary. It now writes them only when they are nonzero, so a payload with no family maps byte-identically to base. The golden did not move.

**Failure (b), c075 corpus. Pre-existing; expectation NOT changed.** The test fails with "unexpected: [('PMC12096016', 'strict', "Pantetheine 4'-phosphate")]".
- It fails identically on base `5e1edb3f` (`05`).
- It replays `_admit_identities` over committed artifacts. ORCH-783 adds only a route key there, which can only remove refusals, and this row has no route key.
- Verdict: catching the row is **correct**. The row's name does not appear in its own source text (`runs_verify/2026-08-27_1341/.../PMC12096016/strict/final_mapped.json`, committed in `0beda95f`), so the armed pass is right to flag it, and `EXPECTED_CATCHES` is stale.
- Updating the list needs a triage of that row, which is outside this lane. Not in SMOKE.

## 6. Open questions for the product owner (orchestrator additions)

1. **Slash lists of distinct proteins.** `_paper_named_members` accepts any shared-prefix pair, so "IL-4/IL-13", "CD4/CD8" and "H2A/H2B" would become a complex of two different proteins. "P2X2/P2X3" is the same lexical shape, so no lexical guard separates them cleanly.
   - (a) Accept.
   - (b) Accept only a digit-only second part ("STIM1/2"). Cost: loses "P2X2/P2X3".
2. **Family terms beyond the ruling.** `_FAMILY_TERMS` includes NFAT, which the ruling does not list, and the family-word list matches "isoform" ("PKM2 isoform" would become a family). Confirm or strike.
3. **Arabidopsis species** on an Unknown-backed family complex (§ 4 item 2). The D-117 rule "species must match" is stated for members, and the Unknown placeholder is not a species-specific protein. Rule whether the complex's own species should be the pathway species.
4. **Identity-admission route** (§ 4 item 1). Reviewer verdict: the route is stamped only on member rows newly built from sources 1-3, so it is no wider than the ruling requires. Recorded in D-117 § 2.
5. **Effect on ORCH-781 rows**: to be measured by the ORCH-784 paper run.
