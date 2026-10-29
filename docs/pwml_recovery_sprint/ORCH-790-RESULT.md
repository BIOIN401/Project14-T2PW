# ORCH-790 RESULT: the family path (D-119 § 1; ORCH-789 C2, C3/PKC, C5, C8-i, D1, D3)

Branch `orch790/fix` off `f3a80e09`. Code commit `64a06040` (shared with ORCH-791, same file). Not merged, not pushed.
**Merge rule 5 is waived by the PO for ORCH-790..795 (D-119 § 4): no reviewer round was run.**

## 1. Cause (checked against artifacts)

| # | defect | evidence |
|---|---|---|
| D1 | "G protein": `_FAMILY_QUERY_STRIP` removes "protein", so the query is `family:"G family"`. UniProt matches that phrase anywhere in the family text, and returns EIF3G ("EIF-3 subunit G family"). | `evidence/orch790/probe_family_fields.json`; base replay `orch790_replay_base.txt`: `G protein -> ['EIF3G']` |
| C2 | The D-119 terms were missing, and matching was exact on spaces and plurals. Bare "GDI family" also returns the Rab GDIs (GDI1/2, CHM, CHML). | probe; base G9 |
| D3 | `_uniprot_result_from_candidates` gives any name the top candidate as `best_effort_fallback`. "PKC" got Par-3, which the gap resolver then wrote onto the row. | ORCH-785 wnt_ca `gap_iteration_summary.json` |
| C3/PKC | The real deletion is an **ordering** defect, so it is fixed in ORCH-791. The first Stage-6 rewrite drops the id-less "PKC" modifier of `PLC reaction` (`reaction_enzyme_protein_dropped_no_external_id`, KI-007), and it does this *before* the family conversion runs. The name gate had already removed Par-3 in Stage 6 (`implausible_name_match`). | replay trace, `orch790_replay_base.txt` (stage-6 report lines) |
| C8-i | `_apply_family_complexes` did not copy `synonyms`. | base G9 |
| determinism | `Counter.most_common` breaks ties in hit order, so the members depended on UniProt's ordering. | base G9 |

## 2. Change

| file | function | change |
|---|---|---|
| `mapping/map_ids.py` | `_family_key` (new), `_FAMILY_TERMS`, `is_family_or_complex_term` | Matching ignores case, spaces, hyphens and one plural "s". Adds JNK, ROCK, GDI, RhoGDI, DELLA, DELLA protein, PIF(s) and Orai. |
| | `_FAMILY_QUERY_CORE` (new), `_uniprot_reviewed_hits` | The request also fetches `protein_families`. The cache key now names the fields. |
| | `_apply_family_complexes` › `_uniprot_family_members` | A hit counts only when one of its comma-separated family levels **equals** `<core> family` or `<core> subfamily`. GDI is queried as "Rho GDI". "PIFs" becomes "PIF". The prefix tie and the sort use a total key (count, prefix; gene, accession). |
| | `_apply_family_complexes` | A family row keeps a single accession only when this pass matched it (`_matched_outright`). Copies `synonyms`. Retypes `modifiers` as well. |
| | `_uniprot_result_from_candidates` + its 4 call sites | New `name=`. A family name gets no best-effort guess (`family_name_has_no_single_protein`). |
| `tests/test_orch783_family_complex.py` | `_FakeUniProt` | The fake now returns the family column. This is a fixture change only: no expectation moved. |

## 3. Proof (all bounded, heavy lock, 0 survivors; `g11_evidence.py check --task ORCH-790`: 0 non-compliant)

| what | tree | result | report |
|---|---|---|---|
| G9 base arm (`tests/test_orch790_family_and_endpoint_order.py` + orch783) | `C:/t/orch790/base` | **20 failed**: every `test_g9_*` fails, each on a value | `evidence/g11/ORCH-790/15-g9-base-r1.json` |
| same, tip | worktree | 54 passed | `14-tip-focused-r1.json` |
| affected tests, 25 files (the brief's list, every file importing a changed function, and the best-effort/gap users) | tip | **517 passed** | `19-tip-affected-r1.json` |
| Stage-6 + quarantine replay, base | base | `orch790_replay_base.txt` | `17-replay-base-r1.json` |
| same, tip | tip | `orch790_replay_tip.txt` | `18-replay-tip-r1.json` |

UniProt answers were recorded once, live (`06`, `07`, `08`, `16`), into `evidence/orch790/uniprot_responses.json`. Both replay arms ran offline with 0 unrecorded queries.

**Replay rows (signed interactions drawn after quarantine), base → tip.** The ORCH-791 ordering is in the same tip.

| leg | base | tip | rows recovered |
|---|---|---|---|
| ORCH-785 wnt_ca | 4 | 11 | PKC ×5, Fzd2→G protein, STIM1/2→Orai. **EIF3G: present at base, gone at tip.** |
| ORCH-785 chlorophyll | 21 | 33 | 11 bare endpoints (ORCH-791), DELLA→SCL27 |
| ORCH-784 wnt_pcp | 3 | 12 | JNK ×5, GDI ×2 (RhoGDI = ARHGDIA/B/G), ROCK ×2 |

Members found at tip: PKC = PRKCA/B/D, Orai = ORAI1-3, DELLA = RGL1-3, GDI = ARHGDIA/B/G. JNK, ROCK, NFAT, PLC and G protein take the Unknown member (UniProt has no JNK or ROCK family).

Replay limits:
- The audit, curator and gap rounds are not run. PKC's gap-resolver accession is put back by hand.
- There is no PathBank DB. Rows the leg resolved are answered with its recorded resolution.
- The species taxonomy id comes from `final_mapped.json`.
- Rows can differ from the live leg. CaMKII rows are `ambiguous_entity_type` here because the audit's de-duplication does not run.

## 4. Pins moved

None. The `test_orch783` fake gained a column, and its assertions are unchanged.

## 5. Failures, dead ends, wrong theories

- **Wrong theory:** "PKC vanished because Par-3 made `_real(row)` true" (ORCH-789 C3). The trace (`10-replay-tip-trace2.json`) shows Stage 6's name gate had already removed Q8TEW0. The row was dropped as a redundant id-less modifier (KI-007) before the conversion. The guard against a guessed accession is kept, because a gap-resolver candidate (`stage3_*`) can still pass the name gate. The fix that recovers the rows is ORCH-791's order.
- The first replay found no family members at all. NCBI backfill is off offline, so the species row had no taxonomy id. The replay now copies the id from the leg (`05` and `06` came before this change).
- A core-length guard (refuse "G") was considered and replaced by the exact family-level match. The exact match also catches the Rab-GDI mix-up, which a length rule cannot.
- An earlier tip build re-ran the endpoint pass after conversion. It was superseded by moving the conversion, and removed.
- I ran one accidental `python -` (stdin). It was stopped at once, before it read anything (task `bqknwxbwf`). No survivor.

## 6. KI delta

- New mapping reason `family_name_has_no_single_protein` (mapping log only; never an omission reason) → **KI-001** (family path).
- D1 is a KI-030-class wrong identity. It is closed for the family-search route.
- No new omission reason codes.

## 7. LLM-side (not fixed)

- chlorophyll `TOC1 inhibits PIFs`: "PIFs" is not declared as an entity at Stage 1 (`undeclared_entity_in_entity_2`).
- wnt_pcp `ROCK activates MRLC` ×2: MRLC is undeclared after Stage 6. This is the C3 vanish, owned by ORCH-792.
- `JNK → c-Jun` ×2 and the `PIF1 …` rows are alias identity (C4, ORCH-793).

## 8. Not settled

1. **C5, the CaMKII holoenzyme with no components, is not reproduced offline.** The Stage-6 input is not persisted. Offline, with no DB, an empty declared complex does get the Unknown member. The leg probably mapped it by name to a PathBank complex id with no members, which makes `_apply_pathbank_unknown_complex_fallback` skip it. Today it is omitted per row with a record (`removed_entity_report.json`, `protein_complex_missing_components`, `entity_omitted_per_row`). Options:
   - (a) a PathBank-identified complex with zero members drops that identity and takes the Unknown member. Cost: a small change, plus one DB SELECT to confirm.
   - (b) keep today's per-row omission.
   Needs a PO call.
2. **Two complexes, one member set.** wnt_ca now carries both `Orai` and `Orai family proteins`, each holding ORAI1-3. Quarantine admits both. The pre-freeze duplicate-canonical check (D-118 § 2) was not run in this replay.
3. **DELLA members.** The 3-letter prefix majority keeps RGL1-3 and drops GAI and RGA. All five are DELLAs. This is the ORCH-783 heuristic, unchanged.
4. **Unknown-backed family species.** The ORCH-783 Q3 question is still open.
5. **SMOKE (merge rule 10) was not run here.** It is owed at merge.

## 9. Proposed LEDGER row

`ORCH-790 | builder | orch790/fix @ 64a06040 (+docs) | family path D-119 § 1: terms+normalisation, exact family-level UniProt match (EIF3G gone), no best-effort for family names, guessed accession no longer blocks conversion, synonyms kept, deterministic members | G9 base 20 red (15-g9-base-r1) / tip green (14); affected 517 passed (19); replay wnt_ca 4→11, chl 21→33, wnt_pcp 3→12 (17/18) | rule 5 waived (D-119 § 4) | open: C5 PathBank-identified empty complex`
