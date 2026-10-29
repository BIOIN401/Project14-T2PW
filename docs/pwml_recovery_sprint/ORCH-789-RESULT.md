# ORCH-789 RESULT: why PWML coverage is still low, by actual root cause

**Type:** read-only diagnosis. No code changed, no test or pipeline run.
**Tree:** `main` / `sprint/pwml-recovery` @ `edb29557`.
**Legs read:** ORCH-785 (code `9a4dc617`): chlorophyll, wnt_ca, sour_taste_v2. ORCH-784 (code `9312999c`): enterobactin, menaquinone, wnt_pcp. The run artifacts are in the pinned worktrees `.claude/worktrees/orch785run` and `orch784run`, under `runs_papers/<TASK>/legs/<paper>-strict/<stamp>/papers/<PMC>/strict/`. The text records are under `runs_papers/<TASK>/REVIEW/<paper>/strict/reaction_evidence.txt`.
**Method:** two small read-only scripts, each run once through `bounded_run.py --heavy-lock ORCH-789`, with 0 survivors:
- `evidence/orch789/omit_tab.py` lists every omitted row with its quarantine reason, the IR withheld reason, the relationship word, and where each named entity went (merged payload, final payload, `quarantined_proteins`, `removed_entity_report`). Output: `evidence/orch789/omit_tab.out.txt`. G11 report: `evidence/g11/ORCH-789/01-omit-tab.json`.
- `evidence/orch789/vanish.py` lists the protein rows that are present after the Stage-1 merge but reach neither the final payload nor any removal record. Output: `evidence/orch789/vanish.out.txt`. G11 report: `02-vanish.json`.

`g11_evidence.py check --task ORCH-789`: 2 artifacts, 0 non-compliant. Every count below comes from `omit_tab.out.txt` unless another file is named.

**Signed** means the relationship word maps to Activation or Inhibition (`pwml/interaction_types.PWML_INTERACTION_TYPES`). Only a signed interaction can ever be drawn. A reaction or a transport counts as drawable.

---

## 1. Headline

| paper | now (headline) | of which can never be drawn (unsigned or a same-entity duplicate) | coverage on drawable rows | ceiling if causes C1-C8 are fixed (headline) |
|---|---|---|---|---|
| chlorophyll | 43/82 (52%) | 14 | 43/68 (63%) | 68/82 (83%) |
| wnt_ca | 10/39 (26%) | 8 | 10/31 (32%) | 31/39 (79%) |
| wnt_pcp | 5/64 (8%) | **41** | 5/23 (22%) | 23/64 (36%), or 23/23 once the denominator is fixed |
| enterobactin | 8/12 (67%) | 2 | 8/10 (80%) | 10/12 (83%) |
| menaquinone | 22/25 (88%) | 1 unsigned, plus 2 paraphrase twins of a drawn step | 22/22 distinct (100%) | no gain available |
| sour_taste_v2 | 6/10 (60%) | 2 | 6/8 (75%) | 8/10 (80%), or 8/8 |

The ceilings are upper bounds. They assume each fix works as described and the same draw comes back. Draws vary at temperature 0 (CLAUDE.md), so a single leg proves nothing.

## 2. Cause table (each omitted row is counted once, under its primary cause)

Rows are **signed or drawable rows recovered**. Unsigned rows are not counted here, because fixing their entity would only move them to KI-008. They are counted in § 3.

| # | cause (actual, read from artifacts) | chl | wnt_ca | wnt_pcp | entero | mena | sour | example rows | code responsible | proposed fix | est. gain | PO decision? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C2** | **Family name the ORCH-783 detector misses, or a family term the conversion skips.** The 17-term list misses JNK, ROCK, GDI (RhoGDI), DELLA, PIFs, bare `Orai`, and `5-HT 3 receptors` (with a space). PKC *is* on the list but is counted under C3. | 2 | 2 | **13** | - | - | 1 | `Rac1 activates JNK`, `RhoA activates ROCK`, `GDI inhibits RhoA`, `DELLA inhibits SCL27`, `TOC1 inhibits PIFs`, `STIM1/2 activates Orai`, `5-HT activation of 5-HT 3 receptors` | `mapping/map_ids.py` `is_family_or_complex_term` (`_FAMILY_TERMS`, l.10104) | Widen the detector: plural `<X>s`, a space-insensitive match, and the corpus families above. Members come from the existing ladder (max 3, same species). | wnt_pcp +20 pp (5→18/64); chl +2; wnt_ca +2; sour +1 | **Yes.** It extends the D-117 § 2 term list. UniProt-sourced members not named in the paper are ORCH-783 § 4 Q1 (the identity-admission route). |
| **C1** | **Stage 6 wraps interaction endpoints before the ORCH-782 gene split and before the ORCH-783 family conversion.** When the other endpoint is, at wrap time, a gene/protein clash symbol (`CHLH`) or an unconverted family (`G protein`), the guard skips the whole row. The identified partner (HY5, PIF3, FHY3, RVE1, TOC1, ABI4, SCL27, Fzd2) is left bare and is withheld as `bare_protein_endpoint`. Every one of the 11 chlorophyll rows has a UniProt id and species (`omit_tab.out.txt` lines 99-136), and `HY5 complex` exists in the payload, but these rows still name bare `HY5`. | **11** | 1 | - | - | - | - | `HY5 activates CHLH` (→ `CHLH gene`), `PIF3 inhibits HEMA1`, `SCL27 inhibits PORC`, `Fzd2 activates G protein` | `map_ids._rewrite_reaction_protein_enzymes_to_complexes`, interaction loop (l.~8714-8769: `shared_name` and the `all(...)` guard), called at l.11768. `_apply_family_complexes` runs later (l.12331). `strict_quarantine._split_gene_protein_names` (l.1782) runs after all of Stage 6 (l.3293). | Re-run the endpoint-wrap pass (the `only_proteins` re-entry already exists) after family conversion. Also apply the gene/protein split before Stage 6, or re-wrap after it. Sequencing only: the wrapper rule (species + UniProt) is unchanged. | **chl +13 pp (43→54/82)**; wnt_ca +1, but only after D1 below | No. Pre-freeze, uses D-104 § 4 / ORCH-754b wrapping as ruled. |
| **C3** | **A protein row vanishes silently** between gap resolution and the quarantine input. It leaves no `quarantined_proteins` or `removed_entity_report` record, and its location row is left behind (later removed as `location_entity_removed`). Quarantine then reports `undeclared_entity_in_<slot>` or `unresolved_actor_in_<slot>`, labelled KI-015. Vanished rows (`vanish.out.txt`): chl GBP, CHL27, LPOR, ferrochelatase, Chl synthase, EXECUTER; wnt_ca PKC; wnt_pcp Frizzled, Fz receptor, Wnt, Smurf, DIA1, Profilin, MRLC, G-actin; mena MenA. | 4 | **5** | 3 | - | (0; see D2) | - | `DAG activates PKC` ×2, `Ca2+ activates PKC` ×2, `PKC activates Cdc42`, `POR reaction`, `ferrochelatase reaction`, `CHL27 promotes PhANG expression`, `DIA1-mediated actin polymerization` | **Deletion site not pinned** (§ 5 Q1). It is not `drop_process_orphan_proteins`, `prune_disconnected_proteins` or `structural_cleanup`, because those all write `quarantined_proteins`. Candidates: `map_ids.py:8460` (the KI-007 drop when another enzyme resolved; fits LPOR beside `POR complex`); the audit (menaquinone's audit "added UniProt identifier for MenA based on function", `audit_iteration_summary.json`). PKC: the gap resolver gave it Q8TEW0, *Par-3* (`best_effort_fallback`, score 0.39, `gap_iteration_summary.json`). That made `_real(row)` true, so `_apply_family_complexes` (l.10309) skipped a listed family. | (a) Make every removal write a record, so the true reason surfaces. (b) PKC: refuse a `best_effort_fallback` accession before the family check, so the ladder converts it. (c) A lone vanished enzyme takes the D-070 Unknown-backed complex (ferrochelatase). | wnt_ca +13 pp (PKC 5); chl +4; wnt_pcp +3 (MRLC/DIA1 rows also need C2) | (a), (b): no. (c) for an enzyme: no (D-070). POR, which keeps `POR complex` but loses `LPOR`, is the KI-004 open question (drop the actor, not the process): **yes**. |
| **C4** | **A single named protein whose identity check refuses the right UniProt candidate**, because the paper uses an alias and not the gene symbol. Seen as `ambiguous_unverified / no_candidate_passed_identity_verification` (c-Jun 2 candidates, Fur 10, Par6 1) or `implausible_name_match` (`gate_fail_report.json`). | 3 | 1 | 2 | **2** | - | 1 | `PIF1 inhibits HEMA1` (PIF1 = PIL5), `TAK1 activates NLK` (MAP3K7), `c-Jun nuclear translocation` (JUN), `Fur inhibits ent gene clusters` (E. coli Fur), `H+ inhibition of Kir2.1` (KCNJ2) | `map_ids._resolve_ambiguous_protein_candidates` (l.11663), `_name_gate_verdict`, `verify_real_protein_identity` | Admit a candidate whose UniProt gene name or gene-synonym list contains the written name exactly, in the pathway taxon. Still a real UniProt id, so D-106 stands. | entero +16 pp (→10/12); sour +10; chl +4; wnt_ca +3 (+2 more jointly with C5); wnt_pcp +3 | **Yes.** The protein-ID requirement is not relaxed, but this changes the identity verifier, which is the gate that admits ids. |
| **C7** | **One name declared in two non-protein kinds.** ORCH-782 splits only `nucleic_acids+proteins`. Chlorophyll declares `Glu-tRNAGlu` / `tRNAGlu` as both compound and nucleic acid, so `ambiguous_entity_type:compounds+nucleic_acids` costs 4 trunk reactions. wnt_ca declares `InsP3R` as compound and protein, so `compounds+proteins` costs 3 transports; the audit's removal of the compound row did not land (`removed_entity_report.json` still removes compound `InsP3R` at iteration 1). | 4 | 3 | - | - | - | - | `Glu-tRNAGlu formation`, `GluTR1 reaction`, `Glu-tRNAGlu reduction to GSA`, `IP3-induced Ca2+ release from ER` | `strict_quarantine._essential_participant_verdict`; `_split_gene_protein_names` (scope) | Consolidate same-name rows across kinds before the freeze. A protein wins over a compound for a receptor. For a tRNA, keep the kind that carries a DB identity. | chl +5 pp; wnt_ca +8 pp | **Yes.** It extends D-117 § 1, and the kind of a charged tRNA is a biology call. |
| **C5** | **A complex declared at Stage 1 with no components.** `Ca2+/calmodulin-dependent kinase II holoenzyme` (the audit deleted its duplicate protein row) is `protein_complex_missing_components`. The ORCH-783 ladder scans protein rows only, and the D-070 complex fallback did not fill it. | - | 3 | - | - | - | - | `Ca 2+ activates CAMKII`, `calmodulin activates CAMKII`, `Ca2+ activates CaMKII` (+2 that also need TAK1, C4) | `map_ids._apply_family_complexes` (proteins only); `_apply_pathbank_unknown_complex_fallback` (scope); `strict_quarantine._entity_representability` | Run the member ladder (CAMK2A/B/G/D), else the Unknown member, on an empty declared complex | wnt_ca +8 pp (+13 with C4) | Probably covered by D-117 § 2 step 4. Reviewer to confirm. |
| **C6** | **A family-complex member is called "isolated"** by the final pre-export connectivity check (`Located protein is isolated in connectivity graph: ORAI2`). ORCH-773 then omits ORAI2, which omits the `Orai family proteins` complex and both STIM→Orai rows. Complex membership is not counted as connectivity. | - | 2 | - | - | - | - | `STIM1/2 activates Orai1/2/3`, `STIM1/2 activation of Orai1/2/3`, plus the SOCE transport's transporter | `process_normalizer.run_strict_post_normalization_gates` (isolation message); `strict_quarantine._apply_per_row_failures` | Count complex membership as an edge in the isolation check, or do not give a member its own location row | wnt_ca +5 pp | **Yes** (a structural gate). KI-013 notes that a degree-0 test has no PathWhiz basis (ORCH-761 § 2). |
| **C8** | **Singles.** (i) A synonym is lost when ORCH-783 converts a protein to a family complex: `NF-ATp` was the audit's synonym of `NFAT`, and `_apply_family_complexes` copies no `synonyms`. (ii) The PLC reaction is sunk by an inferred extra catalyst, `G protein` (KI-031; it is also the EIF3G of D1). (iii) `FHY3 activates HEMA1` has no gene cue for the split. | 1 | 2 | - | - | - | - | `calcineurin activates NF-ATp`, `PLC-mediated PIP2 cleavage`, `FHY3 activates HEMA1` | `map_ids._apply_family_complexes` (keys copied, l.10325); D-113 § 1 path; `strict_quarantine._GENE_ROLE_CUE` | (i) Copy `synonyms` to the complex. (ii) The D-113 § 1 rule for an inferred catalyst when the paper's remains. (iii) PO. | wnt_ca +5 pp; chl +1 | (i) no; (ii) D-113 already; (iii) **yes** |

**Per-paper sum of § 2:** chl +25, wnt_ca +21, wnt_pcp +18, entero +2, mena 0, sour +2. These are the ceilings in § 1.

## 3. Question 5: yes, the denominator counts things that can never be PathWhiz processes

**F-1 (the large one).** The headline drops KI-008 rows only when the IR withheld them: `app/reaction_evidence.py:812` counts `e["ki"] == "KI-008" and e["group"] == "ir"`. An **unsigned** interaction that was quarantined *before* the IR, for its entity, keeps its quarantine label (KI-001 or KI-015) and stays in the headline denominator. PathWhiz could not carry it even if the entity were fixed. Counts:

| chl | wnt_ca | wnt_pcp | entero | mena | sour | total |
|---|---|---|---|---|---|---|
| 14 | 8 | **41** | 2 | 1 | 0 | 66 |

Examples:
- wnt_pcp `Fz receptors interact with PTK7` / `interacts with PTK7` / `Fz receptor binds PTK7` (each of 8 co-receptors, three times over), plus `Smurf binds Par6` and `Wnt binds Vangl2`;
- chl `PIF1 binds CAO promoter`, `FAR1-HY5 interaction`;
- wnt_ca `calmodulin binds CAMKII`, `NLK phosphorylates TCF`;
- entero `Fur binds the promoters of three ent gene clusters`;
- mena `NoxAB interacts with DHNA`.

This is why wnt_pcp reads 8%: 41 of its 59 quarantined rows are unsigned. Proposed fix: exclude from the headline every omitted interaction whose relationship word has no PWML type (`pwml_interaction_type(...) == ""`), whatever its omission reason, and keep the full count beside it as now. **PO decision: yes.** It is the metric definition of D-114 AMENDMENT 2, not a biological gate.

**F-2.** Same-entity "reactions" duplicate drawn transports. sour taste `OTOP1-mediated H+ permeation` (`H -> H`) and `voltage-gated Ca2+ channel-mediated Ca2+ influx` (`Ca2 -> Ca2`) are quarantined `coarse_grained_same_entity_transformation` (shown as "unclassified"). The same two processes are drawn as transports (`final_mapped.json` transports). They are not lost biology. Fix: count a quarantined same-entity reaction as a merged duplicate when a transport carries the same cargo and actor, and add a catalogue row for the code.

**F-3.** Paraphrase twins inflate both numerator and denominator (KI-039); not quantified here. Examples:
- menaquinone `MenA-catalyzed DHNA prenylation to DMK` and `MenA reaction` are twins of the drawn `DHNA prenylation`;
- wnt_ca has 5 PKC rows for 3 facts;
- the wnt_pcp co-receptor list appears ×3;
- the wnt_ca drawn rows include `Ca 2+ activates calcineurin` and `Ca2+ activates calcineurin`.

## 4. The five questions

1. **wnt_ca's 17 KI-015 rows** (16 processes + 1 actor; `omit_tab.out.txt` l.161-258):
   - `undeclared_entity` **PKC ×5**, all signed: the row vanished (C3/PKC);
   - **NF-ATp ×2**: synonym lost on family conversion (C8 i);
   - **TCF ×2**, both unsigned: TCF was pruned by `_prune_invalid_complex_components` as an invalid complex component although two interactions name it. `map_ids.py` l.11837: `_still_needed` checks only complex components, not process references. Recorded as `unresolved_complex_component_no_longer_referenced`.
   - `protein_complex_missing_components` **CaMKII holoenzyme ×5** (3 signed; C5);
   - `protein_complex_component_omitted:ORAI2` **×2**, plus the SOCE transporter actor (C6).

   None is a "new structural reason". Every one is a sequencing or bookkeeping defect upstream.
2. **Chlorophyll's 39 headline omissions** (65 with KI-008):
   - 11 `bare_protein_endpoint` (C1);
   - 22 quarantined interactions: 8 signed (PIF1 ×3 C4, DELLA C2, PIFs C2, FHY3→HEMA1 C8, CHL27 ×2 C3) and 14 unsigned (F-1);
   - 6 reactions: 4 tRNA dual-kind (C7), POR + ferrochelatase vanished enzymes (C3).
3. **wnt_pcp 5 of 64.**
   - 41 of the 59 quarantined rows are unsigned (F-1).
   - Of the remaining 18, 13 are family names outside the detector (JNK 6, ROCK 4, GDI 3; C2).
   - c-Jun (2 transports) is an alias-identity refusal (C4).
   - 3 rows lost a vanished row (C3).
   - Almost every Wnt/PCP protein the paper names by family (Frizzled, Wnt, Smurf, Par6, MRLC, Profilin) has no identity.
4. **Menaquinone 3:** two paraphrase twins of the drawn `DHNA prenylation` (F-3) and one unsigned interaction (F-1). Nothing biological is missing from the count. ORCH-781's 14/14 was a smaller draw (9 reactions vs 18), not a better result. **Enterobactin 4 (headline):** Fur ×2 signed (C4) and Fur ×2 unsigned (F-1); the other 9 are KI-008.
5. **Denominator:** yes. 66 unsigned pre-IR rows plus 2 same-entity duplicates (§ 3).

## 5. Correctness defects found on the way (not coverage)

- **D1 (wrong identity in a written PWML).** "G protein" becomes the UniProt query `family:"G family"`, because `_FAMILY_QUERY_STRIP` (`map_ids.py` l.10071) strips "protein". It returns **EIF3G** (O75821, a translation-initiation subunit). EIF3G is written into `runs_papers/ORCH-785/REVIEW/wnt_ca/strict/pathway.review_required.pwml` (l.511-512). KI-030 class. Fix before C1's `Fzd2 activates G protein` is drawn: do not strip "protein" when it leaves a core of 1-2 characters, or refuse cores under 3 characters.
- **D2.** Menaquinone's drawn `DHNA prenylation` ships with **no enzyme**, because MenA vanished (C3). That is the reaction the summary says PathWhiz drops on import ("would import 21/25"). The audit had given MenA an LLM-invented UniProt "based on function" (`audit_iteration_summary.json`).
- **D3.** The gap resolver's `best_effort_fallback` gave PKC Q8TEW0 (*Partitioning defective 3 homolog*, score 0.39).
- **D4.** TCF: process references are ignored by `_prune_invalid_complex_components` (above).

## 6. Stand-in KI labels (ORCH-787 / ORCH-788)

None of the six latest legs omits a process under `duplicate_canonical_row_identity_not_proven` (KI-015 stand-in), `duplicate_canonical_row_consolidated_same_db_entity` (KI-045 stand-in) or `biological_state_missing_subcellular_location` (`grep` over all six text records). These stand-ins cost 0 rows here. The KI-015 rows above are all `undeclared_entity_*`, `unresolved_actor_*` and complex reasons, which the appendix already maps to KI-015. The catalogue gap that matters is `coarse_grained_same_entity_transformation`, which has no appendix row (F-2).

## 7. Theories tested and dropped

- **"The KI-015 rows are new structural failure modes."** No. Every one traces to an upstream sequencing or deletion step (§ 4 Q1).
- **"Bare endpoints mean the protein lacks an id or species."** No. All 11 chlorophyll endpoints have both, and `HY5 complex` exists. The guard skipped the row because the *other* endpoint was a clash symbol at wrap time (C1).
- **"A vanished row means the audit's stale-index bug" (KI-018).** No. ORCH-769d fixed it (`00eb12d3`), and PKC is still present, at `/entities/proteins/4`, in the gap-resolution summary that runs after the audit.
- **"Vanished rows are what `drop_process_orphan_proteins` / `prune_disconnected_proteins` / `structural_cleanup` remove."** No. All three write `quarantined_proteins`, and none of the vanished names is there (`final_mapped.json` `quarantined_proteins`, listed per leg).
- **"Vanishing correlates with the gap resolver."** Only PKC went through gap resolution. The wnt_pcp and chlorophyll vanished rows have no gap entry (`vanish.out.txt`). The common trait is only that the resolver found no admissible id.
- **"Menaquinone regressed from 14/14."** No. A larger draw, 18 reactions vs 9 (§ 4 Q4).

## 8. NOT settled: for the product owner

1. **Where C3 deletes the row.** Not pinned from artifacts, because Stage 6's action log is not persisted. Options:
   - (a) one offline Stage-6 replay of the post-gap payload with a removal tracer (one bounded job; the ORCH-786 replay harness pattern);
   - (b) first make every protein-row removal write `quarantined_proteins`, then read the next live leg.

   Cost: (a) one replay script; (b) a small code lane, plus one live run.
2. **Metric (F-1).** Should the headline exclude unsigned interactions however they were omitted?
   - Yes: wnt_pcp 8% → 22% today and 100% at the C-ceiling; chlorophyll 52% → 63%.
   - No: wnt_pcp's headline can never pass 36%.
3. **C2 term list and member source.** Which of JNK, ROCK, GDI, DELLA, PIFs, Frizzled/Fz, Wnt, Smurf, Par6, MRLC and Profilin count as families? Are UniProt members the paper does not name acceptable (ORCH-783 § 4 Q1)?
   - Cost of no: wnt_pcp stays under 10%.
4. **C4 alias-aware verification.** Admit an exact match on a UniProt gene synonym (JUN for c-Jun, MAP3K7 for TAK1, KCNJ2 for Kir2.1)? The requirement for a real UniProt id stays.
   - Cost of no: 9 signed rows on 5 papers stay KI-001.
5. **C6 isolation gate and C7 cross-kind consolidation.** Both touch gates. Each needs a ruling before a lane.
6. **POR** keeps a valid `POR complex` but loses the whole reaction to a vanished second enzyme. This is the KI-004 open question: drop the actor, not the process.

The ranking of these causes by rows recovered is in § 2. One leg per paper is a single draw: re-run any leg before reading a change in it as a regression.
