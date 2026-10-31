# ORCH-797b RESULT: alias identity widened to UniProt protein names (D-120 § 2) and more family terms (D-120 § 3)

**Branch:** `orch797b/fix`, base `aefa5b30`. The code is in one file, `src/t2pw/mapping/map_ids.py`. `identity_admission.py` was not needed.
**Merge rule 5 is waived** (D-120 § 5): no reviewer round was run.
**Lead rulings applied** (message to the builder after the reboot; the tie-break itself is recorded as D-120 addendum 7, main `16d08e74`):
- Bump the UniProt cache key only if the cached payload lacks the newly parsed fields. It does lack them (`protein_names`), so the key is bumped (§ 2).
- A Greek-suffixed family symbol ("PLCβ") is a family only when UniProt's exact family match returns members.
- Verbs never count as paper-wording support.

Every job ran through `C:/t/orch79x/{gate,run}.sh` (heavy lock, at least 3 GB RAM, `FINAL SURVIVING COUNT 0`). `g11_evidence.py check --task ORCH-797b`: 0 non-compliant (`evidence/g11/ORCH-797b/33-g11-check-final-r1.json`).

## 1. Cause (verified against the recorded UniProt answers)

| row(s) | cause at base |
|---|---|
| c-Jun (wnt_pcp ×4) | "c-Jun" appears only in JUN's alternative name "Proto-oncogene c-Jun". The parser scored the alternative names and then discarded them. JUN scored 0.47 < 0.5 (`score_below_minimum`). |
| Kir2.1 (sour ×1) | "Kir2.1" appears only in the alternative name "Inward rectifier K(+) channel Kir2.1". It was not kept, so the name gate returned `implausible_name_match`. |
| NARF (wnt_ca ×2), TAK1 (wnt_ca ×3) | Each name is carried by two reviewed human genes (NARF: Q9UHQ1 primary gene, Q8WVD3 synonym; TAK1: synonym of both MAP3K7 and NR2C2). The margin rung refuses both, and the gene fields cannot choose (ORCH-793 § 4). |
| InsP3R, SERCA, calmodulin, Dvl, Par6, Smurf, MRLC, Profilin, Wnt (wnt_ca, wnt_pcp) | Not family terms, so no family complex was built (KI-001). |
| wnt_ca "inositol-1,4,5-trisphosphate receptor (InsP3R)" | Two defects. (a) The paper's "long (SHORT)" spelling was matched against the family terms as a whole and missed. (b) A UniProt name search narrowed the family name to **ITPR1 alone** (`top_unique_alias_candidate`). That single "match" stopped the family conversion; admission then withheld it, and the row was lost. Debug: `g11/ORCH-797b/11-debug-insp3r2-r1.json`. |
| "PKCδ" | `_family_key` deleted the Greek letter, so "PKCδ" read as "PKC" and became the PKC family. Not present on any leg (only β-catenin carries a Greek letter). |
| Dvl at base | Matched one entry, P54792 (DVL1P1, a pseudogene record). |

## 2. Change (`src/t2pw/mapping/map_ids.py`)

| function | change |
|---|---|
| `_extract_uniprot_candidates` | Keeps `protein_names`: the recommended, alternative and **short** names. A name-match counts like an exact name match (0.55 base). |
| `_name_in_protein_names` (new) | The paper's name equals a protein name, or is its **head**: the trailing words. It must not follow a preposition such as "from" or "of". For a non-symbol word ("actin") everything before the head must be generic ("Protein inturned"). Words split on space, comma, slash and brackets, never on a hyphen. Greek letters are spelled out (`_spell_greek`). |
| `_name_gate_verdict` | New rescue `uniprot_protein_name_identity`: the name is one of the entry's protein names, the species is `ok`, and the name is not a family term. |
| `verify_real_protein_identity`, margin rung | A candidate that matches through its protein names also contests the margin as a rival. |
| `_break_alias_tie_by_paper_wording` (new), called after FX4 at both UniProt sites | Applies when two or more **reviewed** genes in the species carry the name. A gene wins only when its distinctive protein-name words occur in the paper's own **definition**, "LONG FORM (NAME)", and no rival's do. The rivals are then set aside, as FX4 does, and the unchanged ladder still judges the winner. Otherwise the result is untouched (KI-001, tagged `alias_tie_unresolved`). Verb forms (6 or more letters ending in -ed or -ing) never count. Never applied to a family name. |
| `_payload_evidence_quotes` (new), `_identity_context` | The payload's `evidence` sentences that name the protein. |
| scoped cache keys | `api-v9` → `api-v10` and `api-fallback-v2` → `api-fallback-v3`: a v9 entry has no `protein_names`. The unscoped `api-v8` chain is unchanged (§ 7). |
| `_FAMILY_TERMS`, `_FAMILY_QUERY_CORE`, `_FAMILY_GENE_MEMBERS` (new), `_family_term_part` (new) | New terms: InsP3R, IP3R, IP3/InsP3 receptor and inositol 1,4,5-trisphosphate receptor (UniProt "InsP3 receptor family"); SERCA; calmodulin; Dvl and Dishevelled ("DSH family"); Smurf; Par6; MRLC; profilin; Wnt. SERCA, Smurf and MRLC have no UniProt family name, so each member gene is queried with `gene_exact` (ATP2A1-3, SMURF1/2, MYL9/MYL12A/MYL12B). Either half of a "long (SHORT)" name counts. Not added: DIA1 (mDia1 = DIAPH1) and GBP (GluTR-binding protein), which are single proteins. |
| `_apply_family_complexes` | Generic detection: a plural of one symbol-like word, or a family symbol plus a Greek letter, converts **only** when UniProt's exact family match returns members; otherwise the row is untouched. A family row a UniProt name search narrowed to one member is no longer "matched outright" (a PathBank row still is: the ORCH-790 PLC preservation is green). A pseudogene record named after a member (DVL1P1) is not a member. Members sort naturally (WNT1, WNT2, WNT2B). The family row itself is never reused as its own member. |
| `_family_key` | Greek letters are spelled out ("PKCδ" → "pkcdelta"). |

## 3. Proof

| proof | result | report |
|---|---|---|
| G9 base arm (`tests/test_orch797b_alias_and_family.py` and `tests/fixtures/orch797b/uniprot.json` copied in, then removed; `git checkout -- .`, base clean) | **12 failed, 5 passed**. The 11 `test_g9_*` fail on values (c-Jun `unmapped`, Kir2.1 `implausible_name_match`, NARF/TAK1 `None`, four families `None`, InsP3R long form `None`, narrowed-calmodulin `None`, PKCδ family `True`). The labelled new-acceptance helper test fails on the absent symbol. Preservations pass. | `evidence/g11/ORCH-797b/31-g9-base-final-r1.json` |
| tip, focused (that file + orch793, orch790, orch783) | 77 passed | `32-focused-tip-final-r1.json` |
| affected batch at tip: 48 files, every test importing `map_ids` or the ORCH-762 harness except the AppTest files (`evidence/orch797x/affected_files.txt`) | **1219 passed, 3 skipped, 6 failed** | `27-affected-tip-r1.json` |
| the same 6 on base | **the same 6 fail**, so they pre-date this change (c075 corpus, c100 5b, compound_resolution ×2, rag_seed merge-base, stage2 boundary) | `28-affected-fails-base-r1.json` |
| UniProt answers, recorded once (live, read-only: tip, then base to fill gaps, then re-records after each code change that issued new queries) | 229 answers; both arms then replayed with **0 unrecorded** | `04`, `05`, `08`, `12`, `18`, `24`; `evidence/orch797x/orch797b_uniprot_responses.json` |
| leg replay, base vs tip (Stage 2 → Stage 6 → strict quarantine → ORCH-768 tail), § 4 | see § 4 | base `16-replay-base-all2-r1.json` (`evidence/orch797b_replay_base.txt`), tip `25-replay-tip-s5-r1.json` (`evidence/orch797b_replay_tip.txt`) |
| audit of every accession the tip ships that base did not | § 5 | `26-new-ids-audit-s5-r1.json` (`evidence/orch797b_new_ids.txt`) |

The G9 base arm copies in the fixture file as well as the test file (the test reads it). Both were removed afterwards.

## 4. Leg replay (`evidence/orch797b_replay.py`)

| leg | strict-accepted processes, base → tip | drawn in PWML, base → tip | PWML written, base / tip | rows recovered |
|---|---|---|---|---|
| chlorophyll | 55 → 55 | 53 → 53 | yes / yes | none; PIF1 stays omitted (§ 6) |
| wnt_ca | **26 → 42** | **16 → 23** | yes / yes | NLK–NARF; TAB1/2/3–TAK1; InsP3R ×5 (IP3 binds/activates, transports, IP3-induced release ×2); SERCA ×4; Ca2+ binds calmodulin |
| wnt_pcp | **11 → 27** | **6 → 11** | yes / yes | c-Jun ×4 (JNK activates/phosphorylates, nuclear import, translocation); Dvl–Par6, Dvl–Smurf, Smurf–Par6 ×2; ROCK–MRLC ×2; Wnt ×3; Inturned |
| sour_taste_v2 | **8 → 9** | n/a | n/a / n/a | Kir2.1 conducts K+ |
| enterobactin (no-loss check) | 9 → 9 | 8 → 8 | yes / yes | none lost |
| menaquinone (no-loss check) | 23 → 23 | 0 → 0 | no / no | none lost |

**No row was omitted at tip that was accepted at base, and no PWML was lost.** Two calmodulin rows (wnt_ca) changed reason only: they are still omitted, now for `ambiguous_entity_type` on their partner CAMKII/calcineurin. That is KI-015, pre-existing.

How the replay works:
- **Every** taxon-scoped UniProt answer is re-asked from the recording by each arm, so both judge identical answers. PathBank, taxonomy and family-search cache answers stay the leg's own.
- Limits:
  - There is no PathBank DB and the audit, curator and gap rounds are not run, so counts differ from the live leg.
  - The Unknown placeholder species is given PathBank species 4, which an offline replay cannot look up.
- Tail artifacts, identical in both arms:
  - sour_taste_v2 stops at pre-freeze with `AMBIGUOUS_RENAME_TARGET 'potassium'`: compound renaming runs offline.
  - menaquinone's export fails on `species_missing_taxonomy/classification`: a species row the offline Stage 6 adds that the leg's DB resolved.
  - In both, quarantine-level counts are the measure.

## 5. New accessions at tip, audited

The final audit lists 56 (row, accession) pairs. The single-protein rows:

| row | accession | verdict |
|---|---|---|
| NARF | Q8WVD3 (RNF138) | correct. The paper defines "NLK-associated RING finger protein (NARF)", which is RNF138's alternative name. **Never Q9UHQ1.** |
| TAK1 | O43318 (MAP3K7) | correct. The paper defines "TGF-β-activated protein kinase 1 (TAK1)". |
| c-Jun | P05412 (JUN) | correct |
| Kir2.1 | P35561 (Kcnj2, mouse) | correct |
| Inturned | Q9ULD6 (INTU, "Protein inturned") | correct |
| PPO1 (chlorophyll) | P55826 (PPOX1, Arabidopsis) | correct. No row changes, because the processes naming it were already accepted. |

The family members (ITPR1-3, ATP2A1-3, CALM1-3, DVL1-3, PARD6A/B/G, PFN1-3, SMURF1/2, MYL9/MYL12A/MYL12B, WNT1/2/2B) all come from UniProt's exact family or gene answers in *H. sapiens*, at most 3 per family.

**Wrong ids that the first tip shipped and the final tip does not.** These are the dead ends in § 6, items 1-3:
- CAMKII → CAMK2N1 ("CaMKII inhibitory protein alpha");
- RhoA → LAMTOR1 ("p27Kip1-releasing factor from RhoA");
- Frizzled → SFRP4;
- Prickle → PRICKLE3;
- DIA1 → DIPK2A ("Deleted in autism 1");
- actin → ACTC1 (cardiac actin);
- the Dvl family including the DVL1P1 pseudogene.

## 6. Failures, dead ends, wrong theories

1. **"A whole-word match anywhere in a protein name"** (the draft) shipped CAMKII → CAMK2N1, RhoA → LAMTOR1, Frizzled → SFRP4 and Prickle → PRICKLE3 (audit `17-new-ids-audit-r1.json`). Fixed: the name must be the protein name's head, not follow a preposition, and a plain word may only follow generic words.
2. **"Any word of any sentence naming the protein supports a gene"** (the draft tie-break) picked DIA1 → DIPK2A and helped RhoA → LAMTOR1 through incidental words. Fixed: only the paper's own "LONG FORM (NAME)" definition counts. The NARF and TAK1 definitions still separate.
3. **Plain-word head match:** "actin" → ACTC1 through "Alpha-cardiac actin" (`23-new-ids-audit-final-r1.json`). Fixed: a non-symbol word only matches after generic words.
4. **Family row narrowed by UniProt:** the InsP3R long form matched ITPR1 alone, so the family conversion was skipped. Found by a debug replay (`10`, `11`). Fixed in `_matched_outright` and `_family_term_part`. The first fix reused the family row as its own member; the guard `r is not row` fixed that.
5. **The DSH family includes DVL1P1** (a pseudogene record). Natural sort gave DVL1, DVL1P1, DVL2. Fixed by the pseudogene filter.
6. **Replay artifacts, fixed in the replay:**
   - The first tail failed on the Unknown species having no PathBank id offline. Fixed by giving it species 4, the DB's answer.
   - A sour_taste_v2 pre-freeze exception aborted the remaining legs. Fixed by recording it per leg.
7. **Rules breach in the first session:** one stray `python -` (stdin). It was stopped at once, ran nothing, and no process survived.
8. **The first session was blocked for about 2 h 20 min** on free RAM below 3 GB. It resumed after the reboot.
9. **Theory ruled out:** "PIF1 separates on the paper's wording". The chlorophyll paper never defines PIF1 in a "long (PIF1)" form. Its only sentences are "Loss of function of PIF1 and PIF3 …" and "PIF1 interacts with G-box …", so PIF1 stays omitted (KI-001).

## 7. Pins moved / KI delta / LLM-side

- **Pins moved:** none. `test_preservation_a_primary_gene_name_does_not_outrank_another_entrys_synonym` stays green for both NARF and PIF1 (the tie-break needs the paper's definition, and that test passes none).
- **KI delta:** no new omission reason code. Rows that cannot be separated stay KI-001. New diagnostics only:
  - result keys `alias_tie_break` and `alias_tie_unresolved`;
  - name-gate reason `uniprot_protein_name_identity`;
  - resolution `order_step` `alias_tie_break`;
  - family note "UniProt gene search … (D-120 family table)".
- **LLM-side:** none found in this lane. Still omitted, but deterministic and outside this lane: CAMKII and calcineurin `ambiguous_entity_type` (KI-015), and TCF4/YAP/TAZ lookups. TCF4 is the primary gene of TCF4 and a synonym of TCF7L2, and the paper gives no definition.

## 8. Not settled (for the PO)

1. **PIF1 (chlorophyll ×3), DIA1 (wnt_pcp ×3) and TCF4** stay omitted: two genes carry the name and the paper defines none of them. Options:
   - (a) keep omitting (now);
   - (b) a curated per-pathway alias table maintained by the PO. Costs PO time per paper.
2. **The unscoped cache chain (`api-v8`)** was not bumped. It is only used when a run has no taxonomy id, and its old entries refuse the new identities conservatively. Bumping it moves the `test_c122` api-v7→v8 pin.
3. **Hand family table (SERCA, Smurf, MRLC):** human gene symbols. In other species, `gene_exact` answers per taxon (case-insensitive), and an absent gene gives fewer members or the Unknown fallback. Expand the table only when a leg needs it.
4. **Members are the first 3 in natural order**, not the paper's isoform: Wnt → WNT1/2/2B, while the PCP paper's Wnt is Wnt5a/Wnt11. This is D-117's rule, unchanged.

## 9. Proposed LEDGER row

`ORCH-797b | builder | orch797b/fix @ <tip> | D-120 §§ 2-3: UniProt protein-name identity (head match), paper-definition alias tie-break (D-120 add. 7), 9 family terms + long-form detection + narrowed-family fix, api-v10 key | G9 base 12 red (g11/ORCH-797b/31) / tip 77 green (32); affected 1219 pass, 6 pre-existing fails identical on base (27/28); replay wnt_ca 26→42 (drawn 16→23), wnt_pcp 11→27 (6→11), sour 8→9, chl/entero/mena unchanged, no PWML lost (16/25); new ids audited (26) | rule 5 waived (D-120 § 5) | open: PIF1/DIA1/TCF4 undefined in paper`
