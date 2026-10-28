# PWML known issues (ORCH-747 through ORCH-769)

**What this is.** One entry per failure mode found between ORCH-747 and ORCH-769, as D-114 § 5 asks. Each entry
says what a user or operator sees, the exact machine strings the artifacts carry, the diagnosed cause, where it
was seen, its status, what a person importing the PWML into PathWhiz does about it by hand, and the fix planned.
Written by ORCH-771 (docs only, no code run) at sprint tip `4befcb33`. Record of the work: `ORCH-771-RESULT.md`.

**How the next lane uses it.** The appendix, *Reason code → KI id*, maps every machine string below to its
entry. ORCH-772 labels each process listed under "Omitted from the PWML" with the KI id its reason code maps
to. Codes written `<n>`, `<slot>`, `<name>` or `<...>` are patterns; the appendix says how to match each one.

**Status words.** `fixed` (merge SHA, lane), `open`, `by design` (D-id), `product decision` (D-id). "Not
measurable offline" means the fix sits in Stage 3 or Stage 6, upstream of every replay checkpoint, so the
ORCH-768 corpus cannot show it (ORCH-768-RESULT § 8 F-6); only a live run can.

**Paper names used below.** enterobactin = PMC12096016 (*E. coli*); menaquinone = PMC12421875 (*L. lactis*
KF147); chlorophyll = PMC5130987; Ara operon = PMC11983282; sour taste = PMC7059786; WNT PCP and Wnt/Ca2+ =
PMC10425814 (one paper, two requests); heme = PMC12856317 (`runs_verify`). A "leg" is one paper in one mode
(strict here unless stated) in one run (ORCH-755, -756, -759, -759p, -767; the replay corpus is ORCH-768).

**Contents.**
A. A process the PWML cannot carry: KI-001 to KI-015 ·
B. No PWML at all: KI-016 to KI-028 ·
C. Wrong or doubtful content inside a written PWML: KI-029 to KI-038 ·
D. Duplicate rows and merge quality: KI-039 to KI-045 ·
E. Extraction and the early stages: KI-046 to KI-053 ·
F. LLM, runtime and harness: KI-054 to KI-060 ·
G. Operations: KI-061 to KI-064 ·
H. Added after ORCH-771: KI-065 ·
Appendix: Reason code → KI id.

---

## A. A process the PWML cannot carry

These are the per-row export requirements. Under D-114 § 2 each one should cost only its own process, which is
removed before the freeze and listed in the text output. Until ORCH-773 lands, several of them still refuse the
whole pathway when the offending entity survives (see KI-016).

### KI-001 — A protein with no UniProt or DrugBank identity

| | |
|---|---|
| **Symptom** | A reaction or interaction the text pathway shows is missing from the PWML. When the protein survives as a component of a kept complex, the whole pathway still fails today (KI-016) until ORCH-773 lands. |
| **Reason codes** | quarantine row reason `protein_missing_external_identity` (state `quarantined_unmapped_entity`; `strict_quarantine._entity_representability`); refusal `unexportable_entity:<n>` (`strict_quarantine.quarantine_and_close`); IR `protein_missing_external_identity`, `generated_complex_component_missing_external_identity` (`pwml/ir.validate_required_pwml_contract`); interaction `protein_endpoint_missing_external_identity` (`ir.interaction_pwml_verdict`, see KI-009); Stage-3 gate messages "Protein '`<name>`' is missing a UniProt or DrugBank identifier." and "Generated protein complex '`<c>`' component protein '`<p>`' is missing a UniProt or DrugBank identifier." (`process_normalizer.run_strict_post_normalization_gates`); mapping issue `implausible_name_match` (`map_ids._NAME_GATE_ISSUE`) |
| **Cause** | PathWhiz creates a protein only with a UniProt or DrugBank id (`smpdb/.../protein.rb:57,59`; ORCH-761 § 1.2). When T2PW's lookup finds none, or the identity verdict refuses the candidate, the protein cannot ship. The requirement is correct and stays (D-106). |
| **Papers and legs hit** | ORCH-756 menaquinone (1 draw): MenD lost, its paper-quoted reaction quarantined `protein_missing_external_identity`; ORCH-755 and ORCH-756 menaquinone initial gate: 8 and 9 Men proteins (ORCH-758 D8; cause KI-002); ORCH-767 Wnt/Ca2+: `phosphorylated CAMKII` (cause KI-018); 25 signed interaction rows in the ORCH-754b replay (Fur 8, RyhB 6, EntB 5, YejM 4, ALAS2, Fnr, EntF, EntD, MenD) |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D8; `evidence/orch756_gold_accuracy.txt` (reader notes, menaquinone); `ORCH-754b-RESULT.md` § 3, § 12 item 2, `evidence/orch754b_unwrappable_endpoints.txt`; `ORCH-761-RESULT.md` § 1.2, § 2 |
| **Status** | **by design** (D-106). Per-row omission instead of pathway refusal is **product decision** D-114 § 2, not yet built |
| **User workaround** | Search UniProt for the protein in the paper's organism. If an entry exists, create the protein in PathWhiz with that accession and attach it as the enzyme (or interaction partner) of the listed process. If none exists, PathWhiz cannot hold the protein; leave the process out. |
| **Planned fix** | ORCH-773 (omit the row, keep the pathway); ORCH-772 (list it with this id). **Family / multi-subunit subset (PKC, PLC, G protein, Orai family, STIM1/2, P2X receptor, VGCC...): fixed by ORCH-783 (D-117 § 2, merged `cac97171`)**: a protein complex named after the family, members from the paper, else a same-species PathBank complex, else UniProt family search (max 3, species must match), else the Unknown-backed complex for any role; the text record names the member source. A single named protein whose lookup failed (PIF1, NLK, Cdc42, TCF4) stays here. Live effect: ORCH-784 |

### KI-002 — Strain-named proteins looked up by free text (menaquinone, *L. lactis* KF147)

| | |
|---|---|
| **Symptom** | Every Men enzyme of the menaquinone paper came back without an identity, or with an accession from the wrong subspecies (MenA → A2RHR6, *cremoris*, counted as verified). |
| **Reason codes** | identity verdict `species_mismatch` (`map_ids.verify_real_protein_identity`); PathBank tier `species_not_found:<species>` (`map_ids.PathBankDbResolver`, e.g. `_map_protein_exact_name_species`); `implausible_name_match`; after the fix, the flag line "UniProt `<acc>`, from strain `<strain>`" (`borrowed_protein_identities`, `map_ids._borrowed_protein_identities`) |
| **Cause** | UniProt was queried with `organism_name:"<paper's strain text>"` (0 hits), then organism-free (returns *E. coli* and human records, correctly refused); PathBank spells the strain "(strain KF147)"; taxonomy ids never entered a query. KF147's Men sequences exist only in UniParc, which PathWhiz's format check rejects (ORCH-761 § 0 items 3-4). |
| **Papers and legs hit** | ORCH-755, ORCH-756, ORCH-759 menaquinone |
| **Evidence** | `ORCH-761-RESULT.md` § 4.2, § 6 761-A/761-A2; `evidence/orch761_uniprot_probe.txt`, `orch761_pathbank_probe2.txt`; replay `evidence/orch762/27-replay-table-base.log` vs `28-replay-table-tip.log` |
| **Status** | **fixed** `9757234d` (ORCH-762, FX1-FX5, under D-108): 7 of 9 recovered from same-subspecies strain IL1403, MenA no longer A2RHR6. **Open** for MenI (`yhdA`, no symbol route) and MenA (three KF147 paralogs). Stage-6 fix: **not measurable offline** |
| **User workaround** | MenI and MenA: add them in PathWhiz by hand only if you accept an IL1403 record (UniProt Q9CHK5 for MenI is the identical sequence; ORCH-761 § 0 item 3); otherwise leave them out. |
| **Planned fix** | none planned. Sketch: a reviewed alias table (`MenI` ↔ `yhdA`) and a ruling on naming one MenA paralog (ORCH-761 Q1(a)) |

### KI-003 — A protein or complex with no species

| | |
|---|---|
| **Symptom** | The process naming it is missing from the PWML. |
| **Reason codes** | quarantine `protein_missing_species`, `protein_complex_missing_species` (`strict_quarantine._entity_representability`); IR `protein_missing_species`, `protein_complex_missing_species`, `generated_complex_component_missing_species` (`ir.validate_required_pwml_contract`); interaction `protein_endpoint_missing_species`, `protein_endpoint_missing_species_and_external_identity` (`ir.interaction_pwml_verdict`) |
| **Cause** | PathWhiz requires a species on proteins and complexes (`protein.rb:47`, `protein_complex.rb:46`); an empty one silently defaults to *Homo sapiens* in PathWhiz (ORCH-761 § 1.1), so T2PW refuses rather than ship it. |
| **Papers and legs hit** | Not observed on a leg in this window (the ORCH-754b replay found no row lacking a species, `ORCH-754b-RESULT.md` § 3). Catalogued because the codes exist and ORCH-772 must be able to label them. |
| **Evidence** | `ORCH-761-RESULT.md` § 1.1, § 2 table rows 2 and 5 |
| **Status** | **by design** (D-106) |
| **User workaround** | Add the protein in PathWhiz under the paper's organism, then attach it to the listed process. |
| **Planned fix** | ORCH-773 (omit per row); ORCH-772 (list it) |

### KI-004 — A gene, compound or element collection named as an enzyme, modifier or transporter

| | |
|---|---|
| **Symptom** | Before the fix: no PWML, final gate "Unknown protein/modifier reference" (chlorophyll 9 times). After it: the reaction is missing from the PWML; some steps export with no catalyst. |
| **Reason codes** | quarantine `non_protein_actor_in_enzymes`, `non_protein_actor_in_modifiers`, `non_protein_actor_in_transporters` (pattern `non_protein_actor_in_<slot>`, state `quarantined_broken_reference`; `strict_quarantine._actor_verdict`); final-gate messages "Unknown protein/modifier reference: `<name>`" and "Unknown transporter reference: `<name>`" (`process_normalizer.run_strict_post_normalization_gates`), driver codes `gate.unknown_protein_modifier_reference`, `gate.unknown_transporter_reference` (`batch/driver._slug_code`) |
| **Cause** | Stage 1 declared PIF1, GBP, FAR1, CPO1, PPO1 both as proteins and as genes. The audit deleted the five protein rows; its guard pools every bucket (`apply_audit_patch.py:861,882-903`), so the surviving gene rows hid the loss. Quarantine looked actors up in all buckets, the final gate in proteins and complexes only. |
| **Papers and legs hit** | ORCH-767 chlorophyll (no PWML); `runs_verify/2026-08-22_2017` heme (see KI-005) |
| **Evidence** | `ORCH-769b-RESULT.md` §§ 1-2, § 7 F3; `evidence/orch769_diag/769b/03-guard.out.txt`; `ORCH-768-RESULT.md` § 6 item (b) |
| **Status** | **fixed** `509fe185` (ORCH-769b): quarantine uses the gate's own registry, so the process is removed instead of blocking the pathway. Cost: ALAD, CPO and PPO export uncatalysed, eight regulators and the GLK1 anchor are lost (completeness 0.933 → 0.867). **Open:** the audit guard that pools buckets; whole-process removal versus dropping only the bad actor |
| **User workaround** | For each listed reaction, add the enzyme in PathWhiz as a protein (with its UniProt id) and attach it; the text record names the actor that was refused. |
| **Planned fix** | none planned. Sketch: make the audit guard per-bucket (an LLM-stage change, measurable only live; ORCH-769b § 9 item 2) |

### KI-005 — A compound cofactor written into a reaction's modifiers

| | |
|---|---|
| **Symptom** | A reaction catalysed by a cofactor-dependent enzyme is missing from the PWML because its cofactor sat in the modifier slot. On the heme leg this was the only reaction, leaving an interactions-only pathway. |
| **Reason codes** | `non_protein_actor_in_modifiers` (`strict_quarantine._actor_verdict`); before ORCH-769b, `gate.unknown_protein_modifier_reference` |
| **Cause** | The extractor put pyridoxal 5'-phosphate (`entity_type: compound`, `role: cofactor`) in the modifiers of the ALAS2 reaction. The final gate accepts only proteins and complexes there, and quarantine now removes the whole process (KI-004). |
| **Papers and legs hit** | `runs_verify/2026-08-22_2017` and a second heme leg (PMC12856317) |
| **Evidence** | `ORCH-769b-RESULT.md` § 7 F2, `evidence/orch769b/moved_leg_probe.py` (G11 `ORCH-769b/17`) |
| **Status** | **open** (the cost of the ORCH-769b rule; ORCH-769b § 9 item 1) |
| **User workaround** | Re-create the reaction in PathWhiz with its protein enzyme, and add the cofactor as a reaction participant or a note. |
| **Planned fix** | none planned. Sketch: the normalizer re-files a `role: cofactor` compound modifier as a participant (a Stage-3 lane; option (c) of ORCH-769b § 9 item 1) |

### KI-006 — A reaction enzyme left as a bare protein (not wrapped in a complex)

| | |
|---|---|
| **Symptom** | "PWML required-field gate failed." on a pathway released `review_required`. |
| **Reason codes** | `reaction_enzyme_must_be_protein_complex` (`ir.build_pwml_ir` / `ir.validate_required_pwml_contract`; also `stage_contracts.validate_pre_export`); app error "PWML required-field gate failed." (`streamlit_app.run_pwml_export`); mapping action `reaction_enzyme_complex_wrapper_skipped_invalid_component` (`map_ids._rewrite_reaction_protein_enzymes_to_complexes`) |
| **Cause** | PathWhiz attaches enzymes to complexes only (`reaction_enzyme.rb:13`). Stage 6 wrapped enzymes before the Phase-2 UniProt fallback ran, so an enzyme that only Phase 2 resolved (CPO, Q9LR75) shipped bare. The same ordering left signed-interaction endpoints bare (KI-009). |
| **Papers and legs hit** | ORCH-759 chlorophyll (`CPO reaction`); 4 of 121 archived strict legs (CPO, LDH, AtCYP73A5, ceramide synthase); 3 tier-B corpus legs |
| **Evidence** | `ORCH-759c-RESULT.md` § 8 F3; `evidence/orch759_cpo_diag/`; `ORCH-759e-RESULT.md` § 1, § 4, § 12; `ORCH-768-RESULT.md` § 6 |
| **Status** | **fixed** `360a6ac5` (ORCH-759e). **Not measurable offline** (Stage 6): the ORCH-759 chlorophyll leg still fails on the corpus because its Stage-6 output is frozen (ORCH-768 § 8 F-6). Transporters are never wrapped by either path; the contract accepts a bare transporter |
| **User workaround** | In PathWhiz, create a one-protein complex for the enzyme (same UniProt id) and set it as the reaction's enzyme. |
| **Planned fix** | live confirmation run (ORCH-774) |

### KI-007 — An enzyme with no identity is dropped before the Phase-2 fallback could resolve it

| | |
|---|---|
| **Symptom** | A correct enzyme vanishes from a reaction and from the proteins list, with no quarantine record. |
| **Reason codes** | mapping action reason "no UniProt or DrugBank identifier; other enzymes remain"; summary counter `reaction_enzyme_proteins_dropped_no_external_id` (both `map_ids._rewrite_reaction_protein_enzymes_to_complexes`) |
| **Cause** | When a reaction has one enzyme with an identity and one without, the rewrite drops the one without before Phase 2 runs, so Phase 2 never gets to resolve it, and no `quarantined_proteins` record is written. |
| **Papers and legs hit** | Reproduced on a fixture (`[HEMA1 (main loop), CPO (Phase 2 only)]`); not measured on a leg |
| **Evidence** | `ORCH-759e-RESULT.md` § 8 F2, § 10 item 2; test `test_preservation_the_drop_rule_still_runs_before_phase2` |
| **Status** | **open** (merge rule 7 may be implicated) |
| **User workaround** | Compare the enzymes named in the text record's quote with the PWML's; add any missing enzyme in PathWhiz. |
| **Planned fix** | none planned. Sketch: defer the drop until after Phase 2, or at least record the removed row in `quarantined_proteins` (ORCH-759e § 10 item 2) |

### KI-008 — An interaction whose relationship is not Activation or Inhibition

| | |
|---|---|
| **Symptom** | Interactions shown in the text record (binds, competes with, part of, recruits, dissociates, modifies...) are absent from the PWML. WNT PCP draws 1 of its 18 interactions; enterobactin (ORCH-756) none of 7. |
| **Reason codes** | `untyped_interaction` (`ir.interaction_pwml_verdict`); IR report warning `interaction_withheld_from_pwml` (`ir.build_pwml_ir`) |
| **Cause** | PathWhiz validates an interaction's type against Activation and Inhibition only (`interaction.rb:13, :32`). Unsigned or symmetric words, `competes with` first among them, are never given a direction. |
| **Papers and legs hit** | ORCH-767 WNT PCP (17 of 18); ORCH-756 enterobactin (7 of 7); 232 rows in the ORCH-754b replay |
| **Evidence** | `ORCH-754-RESULT.md` § 2 item 6; `ORCH-754b-RESULT.md` § 3, § 7; `ORCH-769c-RESULT.md` § 7 item 1; `evidence/orch756_gold_accuracy.txt` |
| **Status** | **by design** (D-104 § 4; D-104 AMENDMENT 1 §§ 2-3) |
| **User workaround** | None inside PathWhiz's model: it has no type for these relations. Read them in the text record, and add an Activation or Inhibition by hand only where the paper states the direction. |
| **Planned fix** | none planned (a mapping of `binds` or `part_of` to a PathWhiz type would be a new decision; ORCH-769c § 7 item 1) |

### KI-009 — An interaction endpoint that is a protein Stage 6 could not wrap

| | |
|---|---|
| **Symptom** | A signed interaction (Activation or Inhibition) is missing from the PWML. |
| **Reason codes** | `bare_protein_endpoint`, `protein_endpoint_missing_species`, `protein_endpoint_missing_external_identity`, `protein_endpoint_missing_species_and_external_identity` (pattern `protein_endpoint_missing_<...>`; all `ir.interaction_pwml_verdict`) |
| **Cause** | PathWhiz has no bare-protein interaction member; a protein endpoint must be a one-protein complex (`interaction_element.rb`). Stage 6 builds that wrapper only when the protein has a species and a UniProt/DrugBank id and is not the Unknown sentinel or a name another complex holds. Before ORCH-759e an endpoint resolved only by Phase 2 also stayed bare. |
| **Papers and legs hit** | the R3 golden leg (`DHNA inhibits MenD`); 25 replay rows (KI-001's list); PMC11405693 `piperonylic acid inhibits C4H` (Phase-2 case) |
| **Evidence** | `ORCH-754b-RESULT.md` §§ 1-4, § 6; `ORCH-759e-RESULT.md` § 1 |
| **Status** | **fixed** for wrappable proteins: `c711d859` (ORCH-754b, exportable rows 13 → 73 in replay) and `360a6ac5` (ORCH-759e, Phase-2 endpoints; **not measurable offline**). The residual withholding is **by design** (D-104 § 4, D-106) |
| **User workaround** | Give the protein its UniProt id and species in PathWhiz, make a one-protein complex of it, and draw the interaction from or to that complex. |
| **Planned fix** | ORCH-772 (list it) |

### KI-010 — An interaction endpoint whose name also names a gene, compound or collection

| | |
|---|---|
| **Symptom** | A signed interaction is missing from the PWML although its protein has an identity. |
| **Reason codes** | `ambiguous_protein_endpoint` (`ir.interaction_pwml_verdict`, carries `ambiguous_endpoints: [{name, also_names}]`) |
| **Cause** | The payload holds a protein and a gene (or compound, collection) under one name, so the endpoint is not known to be the protein. Fail closed. |
| **Papers and legs hit** | `runs_verify/2026-08-04_1358` PMC12096016 research: `2,3-DHB inhibits EntB isochorismatase`, `Fur represses entD` |
| **Evidence** | `ORCH-754b-RESULT.md` § 15.4, `evidence/orch754b_ambiguous_endpoint_rows.py` (G11 `ORCH-754b/133`) |
| **Status** | **by design** (ORCH-754b fail-closed rule; D-106) |
| **User workaround** | Read the quote: if the paper means the protein, draw the interaction from the protein's complex; if it means the gene (e.g. "Fur represses entD"), draw it to the gene. |
| **Planned fix** | none planned |

### KI-011 — A gene and a protein share one bare symbol

| | |
|---|---|
| **Symptom** | Gene-regulation steps (e.g. `HY5 activates HEMA1`, `PIF1 inhibits GUN4`) and reactions naming the bare symbol are missing from the PWML. Before ORCH-759c the whole chlorophyll pathway was refused. |
| **Reason codes** | quarantine `ambiguous_entity_type:nucleic_acids+proteins` (pattern `ambiguous_entity_type:<bucket>+<bucket>`, state `quarantined_broken_reference`; `strict_quarantine._essential_participant_verdict`, `_actor_verdict`); refusal `entity_type_overlap:<n>` (`strict_quarantine.quarantine_and_close`, check `_entity_type_overlaps`) |
| **Cause** | Stage 1 declared gene rows `HEMA1`, `CAO`, `GUN4`, `PORB` beside proteins of the same names. Admission correctly quarantines every process naming the bare symbol. The closure then kept the gene rows alive through a same-named protein component, and `_entity_type_overlaps` refused the pathway. |
| **Papers and legs hit** | ORCH-759 chlorophyll (28 processes quarantined, 4 overlaps) |
| **Evidence** | `ORCH-759-RESULT.md` § 4 item 6; `evidence/orch759_chlorophyll_diag/`; `ORCH-759c-RESULT.md` § 1, § 10 item 3 |
| **Status** | refusal **fixed** `45aa9c28` (ORCH-759c, closure is bucket-aware). Lost gene-regulation biology **fixed** by ORCH-782 (D-117 § 1, merged `cac97171`): a clashing gene row becomes `<SYMBOL> gene` and each reference resolves by role; on the ORCH-759c chlorophyll fixture broken-reference quarantines went 28 -> 10. A row with no gene-level wording (e.g. "interacts with") is still omitted per row under this id. Live effect: ORCH-784 |
| **User workaround** | Add the gene (nucleic acid) and the protein as separate elements in PathWhiz, and redraw each listed regulation to the one the quote means. |
| **Planned fix** | done: ORCH-782 (`strict_quarantine._split_gene_protein_names`). Residual open questions: ORCH-782-RESULT § 6 and § 7 |

### KI-012 — An invented single-protein wrapper (Stage 2 infers a complex-named enzyme)

| | |
|---|---|
| **Symptom** | Enterobactin has no PWML although Stage 1 and the final payload score 5/5 against gold. In other runs the same invented row shows as an extra or wrong catalyst (KI-031). |
| **Reason codes** | final-gate message "Generated protein complex '`<complex>`' component '`<component>`' is missing positive stoichiometry." (`process_normalizer.run_strict_post_normalization_gates`); driver code is the 64-character slug of that whole message, so it embeds the names, e.g. `gate.generated_protein_complex_enterobactin_synthase_complex_componen` (`driver._slug_code`); stage `final_pre_export_stage3_gates`; wrapper `generation_reason` `complex_named_source_entity_wrapper` (`process_normalizer.relocate_complex_named_proteins`); informational `generated_wrapper_reason_invalid` (`payload_models._generated_wrapper_issues`) |
| **Cause** | Stage 2 inferred `enterobactin synthase complex` (its evidence sentence is not in the paper). Normalization made a generated wrapper whose only component is a new protein row `enterobactin synthase`; the audit nulled the default stoichiometry (`audit_json_llm.py:46` prompt rule); Stage 6 mapped the row to EntE's accession P10378; the gate requires a positive count on generated wrappers. |
| **Papers and legs hit** | ORCH-767 enterobactin (no PWML); `runs_verify/2026-08-21_2057` PMC12452463 (same shape); ORCH-759 enterobactin (the degree-0 variant, KI-013) |
| **Evidence** | `ORCH-769-RESULT.md` § (a); `evidence/orch769_diag/769a/01-diag.out.txt` (counterfactuals A-C pass the gate), `02-diag.out.txt` (2 actors cite a span naming a different protein); `evidence/orch767_gold_accuracy.txt` (`wrong_identity` P10378) |
| **Status** | **product decision** D-113 § 1 (quarantine the inferred wrapper when the paper's catalyst remains) and § 2 (the positive-count gate stays). **Open**: ORCH-769a not merged |
| **User workaround** | Delete the "enterobactin synthase complex" catalyst; the paper's own enzyme (EntF) is the catalyst. |
| **Planned fix** | ORCH-769a |

### KI-013 — An unreferenced duplicate protein row after the Stage-6 rename (degree 0)

| | |
|---|---|
| **Symptom** | No PWML; the final pre-export gate says a protein has degree 0, on a pathway released `review_required` with strict gates passed. |
| **Reason codes** | final-gate message "Protein has degree 0 after normalization: `<name>`" (`process_normalizer.run_strict_post_normalization_gates`), driver code `gate.protein_has_degree_after_normalization`, stage `final_pre_export_stage3_gates` (`streamlit_app.freeze_canonical_payload`) |
| **Cause** | The KI-012 wrapper's component `enterobactin synthase` got EntF's P11454 from the audit; Stage 6's `_reconcile_components_against_local_proteins` matched the component to `EntF` (UniProt first, first row wins), leaving the row unreferenced. Quarantine kept it because its synonyms were referenced; the final gate tests the name only. The degree-0 test has no PathWhiz basis (ORCH-761 § 2). |
| **Papers and legs hit** | ORCH-759 enterobactin |
| **Evidence** | `ORCH-759-RESULT.md` § 4 item 2; `evidence/orch759_enterobactin_diag/`; `ORCH-759a-RESULT.md` § 1, § 7 item 3 |
| **Status** | **fixed** `57713a11` (ORCH-759a: Stage 6 folds an unreferenced duplicate with the same admitted accession and species; Stage 6, so **not measurable offline**). **Open:** the quarantine pruner (name or synonyms) and the final gate (name only) still disagree for other orphans |
| **User workaround** | Delete the stray protein element from the PathWhiz pathway; it has no reactions. |
| **Planned fix** | ORCH-773 (omit a degree-0 row instead of refusing); none planned for the pruner/gate disagreement (ORCH-759a § 7 item 3) |

### KI-014 — A duplicate of a reaction that is kept

| | |
|---|---|
| **Symptom** | A row in the text record is listed as removed although the same reaction appears in the PWML. |
| **Reason codes** | `duplicate_locked_reaction` (`process_normalizer.dedupe_processes`; Stage-6 rule 3 `stage_one_ensemble.merge_on_mapped_identity`) |
| **Cause** | Two locked Stage-1 rows describe one reaction (e.g. the paper's MenD step and the same step re-read from the kinetic-model term `kcatD[MenD][B]`); one is kept, the other quarantined. Not lost biology, unless the survivor is itself removed (as MenD was in ORCH-756, KI-001). |
| **Papers and legs hit** | ORCH-756 menaquinone (rxn_lock_005) |
| **Evidence** | `evidence/orch756_gold_accuracy.txt` (rows 9-10 readings); `ORCH-754a-RESULT.md` § 10.2 N1/N3 |
| **Status** | **by design** |
| **User workaround** | None needed; check that the kept twin is in the PWML. |
| **Planned fix** | ORCH-772 (list as a duplicate, not as a loss) |

### KI-015 — Other structural quarantine reasons (catalogued for completeness)

| | |
|---|---|
| **Symptom** | A process is missing from the PWML for a structural reason not diagnosed in ORCH-747 to ORCH-769. |
| **Reason codes** | all `strict_quarantine`: `explicit_out_of_scope_verdict`, `confidence_below_floor:<x><<y>`, `malformed_participant_in_<slot>`, `missing_<slot>`, `undeclared_entity_in_<slot>`, `unnamed_actor_in_<slot>`, `unresolved_actor_in_<slot>`, `process_row_not_an_object`, `process_row_vanished_during_closure` (state `quarantined_disconnected`), `reaction_coupled_transport_has_no_exportable_ir_representation` (`RCT_NOT_EXPORTABLE_REASON`), `<bucket>_missing_name`, `malformed_placeholder:<claim>`, `protein_complex_missing_components`, `protein_complex_component_cycle`, `protein_complex_component_unnamed`, `protein_complex_component_unresolved:<name>`, `protein_complex_component_unrepresentable:<name>:<reason>`; IR `protein_complex_missing_components` and `component_protein_unresolved` (`ir.validate_required_pwml_contract`) |
| **Cause** | Each reason names its own shape; see the code. Only `protein_complex_missing_components` was seen in this window (2 tier-B corpus legs, IR level, not diagnosed). A reaction-coupled transport can never export: the IR builds it empty (`strict_quarantine.py` comment at `RCT_NOT_EXPORTABLE_REASON`). |
| **Papers and legs hit** | `protein_complex_missing_components`: 2 tier-B legs of the ORCH-768 corpus; the rest not observed |
| **Evidence** | `src/t2pw/pipeline/strict_quarantine.py` (the functions above); `ORCH-768-RESULT.md` § 6 |
| **Status** | **by design** (each is an existing structural gate; none was changed in this window) |
| **User workaround** | Read the process in the text record and draw it in PathWhiz if the quote supports it. |
| **Planned fix** | ORCH-772 (label with this id until a specific entry exists) |

---

## B. No PWML at all

### KI-016 — A strict-quarantine refusal sinks the whole pathway

| | |
|---|---|
| **Symptom** | No PWML. The leg's RESULT reads `final_gate_report_missing` at `final_pre_export_gate_lifecycle`, or the app says "PWML export stopped by the quarantine boundary: ..."; release `diagnostic_only`. |
| **Reason codes** | refusals `unexportable_entity:<n>`, `entity_type_overlap:<n>`, `degree_zero_export:<n>`, `unaccounted_locked_reactions:<n>`, `closure_not_converged:<n>` (`strict_quarantine.quarantine_and_close`); `final_gate_report_missing` (`gate_reports.REASON_FINAL_REPORT_MISSING`, returned by `gate_reports.gate_verdict`); stage `final_pre_export_gate_lifecycle` (`gate_reports._fail_closed`); "PWML export stopped by the quarantine boundary: `<reasons>`" (`streamlit_app.run_pwml_export`); release `diagnostic_only` with `serialization_would_require_invention` (`release_status.classify_release_status`) |
| **Cause** | The quarantine refuses when an entity it cannot export survives (e.g. a protein with no identity kept alive as a complex component). `freeze_canonical_payload` then runs no final gate, by design, and `gate_verdict` fails closed on the missing report. The symptom is the same whatever the underlying entity problem is; the cause entries are KI-001, KI-011, KI-017, KI-018. |
| **Papers and legs hit** | ORCH-755 enterobactin (KI-017); ORCH-759 chlorophyll (KI-011); ORCH-767 Wnt/Ca2+ (KI-018); 3 tier-B corpus legs |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D1 steps 5-6; `ORCH-768-RESULT.md` § 6 item (d) |
| **Status** | **product decision** D-114 § 2 (omit the offending rows, write the rest); not yet built |
| **User workaround** | None for a user (no file is written). Operator: read `quarantine_report.json` `strict_invariants` for the entity named, and see its cause entry. |
| **Planned fix** | ORCH-773 |

### KI-017 — An inferred alias strips a verified protein's accession (holo-EntB)

| | |
|---|---|
| **Symptom** | Enterobactin has no PWML; EntB, verified in PathBank, is reported as lacking a UniProt id. Intermittent: it needs one particular Stage-2 reply. |
| **Reason codes** | identity admission `accession_claimed_across_incompatible_entity_kinds` (`identity_admission.RULE_ACCESSION_KIND_CONFLICT`, applied in `map_ids._admit_identities`); then `protein_missing_external_identity`, `unexportable_entity:1`, `final_gate_report_missing` |
| **Cause** | Stage 2 wrote a location row for `holo-EntB`. The normalizer filed the dangling name as a compound; the mapper called it protein-like and gave it EntB's P0ADI4 through the alias "EntB"; admission saw one accession on a compound and a protein and withheld it from both, so EntB (a complex component) had no identity. |
| **Papers and legs hit** | ORCH-755 enterobactin strict (1 of 2 ORCH-755 enterobactin legs; 0 of 1 in ORCH-756) |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D1; `ORCH-761-RESULT.md` § 6 761-B |
| **Status** | **fixed** `9757234d` (ORCH-762 FX6, one kind per name: `holo-EntB` stays on the compound route and EntB keeps P0ADI4). Stage 6: **not measurable offline** |
| **User workaround** | If a PWML shows EntB (or another protein) without its UniProt id, set it in PathWhiz (EntB: P0ADI4). |
| **Planned fix** | live confirmation (ORCH-774) |

### KI-018 — The audit patch applier uses stale indices after a remove (Wnt/Ca2+)

| | |
|---|---|
| **Symptom** | No PWML; release `diagnostic_only`; the quarantine names `phosphorylated CAMKII` as a protein with no identity. |
| **Reason codes** | `unexportable_entity:1` with `protein_missing_external_identity` on `phosphorylated CAMKII`; `final_gate_report_missing`; release `diagnostic_only`, reason `serialization_would_require_invention` |
| **Cause** | `apply_audit_patch.apply_patch_with_policy` applies a remove and then later ops by their pre-patch indices without rebasing, so each synonym after the removed row lands one row later. A synonym moved onto the wrong row (`calcineurin B` onto `phosphorylated CAMKII`) kept an unidentified protein referenced. A second applier defect: `/-` on a missing key creates a dict. |
| **Papers and legs hit** | ORCH-767 Wnt/Ca2+ |
| **Evidence** | `ORCH-769-RESULT.md` § (d); `evidence/orch769_diag/769d/2-diag.out.txt` (dropping the cross-row synonyms gives `review_required`, gate pass, 9 transports, 13 interactions) |
| **Status** | **open**. Orchestrator ruling in D-113 (operational): rebase later ops after an accepted remove or insert (option a) |
| **User workaround** | None for a user (no file). After the fix, check the synonyms of each protein in the text record against its quote. |
| **Planned fix** | ORCH-769d |

### KI-019 — An unreferenced, untaxonomised species row

| | |
|---|---|
| **Symptom** | Menaquinone reaches `release_ready` with 19 reactions and still gets no PWML: the required-field gate fails on a species row. |
| **Reason codes** | `species_missing_taxonomy`, `species_missing_classification` on `/entities/species/<i>` (`ir.validate_required_pwml_contract`); "PWML required-field gate failed." |
| **Cause** | Three draws pooled three species rows. The paper's own spelling "L. lactis subsp. lactis KF147" is referenced by nothing, NCBI cannot match the abbreviated genus, and the C-120 alias pass refuses strain-suffixed abbreviations (D-098). The gate checks every species row. |
| **Papers and legs hit** | ORCH-759 and ORCH-759p menaquinone (2 of 2 three-draw legs then; not on the ORCH-767 draws) |
| **Evidence** | `ORCH-759-RESULT.md` § 4 item 7, § 7; `evidence/orch759p_species_diag/`; `ORCH-766-RESULT.md` § 1, `evidence/orch766/13-replay-tip.log` vs `14-replay-base.log` |
| **Status** | **fixed** `19a6a6ce` (ORCH-766, D-111 § 1: a species row nothing references is removed before the freeze, recorded like degree-zero removals). Corpus 51 → 53/91. Residual **by design**: the first species row is never removed, because the writer uses it as the pathway species |
| **User workaround** | None needed after the fix. |
| **Planned fix** | none |

### KI-020 — An interactions-only pathway crashed the review diagram (WNT PCP)

| | |
|---|---|
| **Symptom** | No PWML; "the app never rendered the `refinement_generate_pwml` button"; the driver filed the leg as a `crash` and reported 11 reactions for a graph with none. |
| **Reason codes** | corpus harness stage `refinement_review_crash` (`evidence/orch768_replay_corpus.py`); base error "DraftGraph has no reaction nodes — nothing to render." (`draft_graph_render.render_draft_graph_to_png_bytes` at `e3d5b65f`, removed by the fix); button key `refinement_generate_pwml` (`driver.KEY_GENERATE_PWML`) |
| **Cause** | The review diagram draws only reaction and transport nodes and raised on a graph of interactions only, before `refinement_working_json` was set, so the Generate PWML button never rendered. |
| **Papers and legs hit** | ORCH-767 WNT PCP (0 reactions, 18 interactions); `runs_verify/2026-08-22_2017` PMC12856317 (became interactions-only under ORCH-769b) |
| **Evidence** | `ORCH-769c-RESULT.md` §§ 1-2, § 6 |
| **Status** | **fixed** `ec800370` + `40122c63` (ORCH-769c). Corpus 54 → 56/91. **Open:** in the interactive app such a graph shows no diagram and "Submit Changes to AI" refuses with "No refinement graph image is available." (`streamlit_app.py:2824`); a fix needs a D-xxx for the protected file |
| **User workaround** | Batch runs are unaffected. In the app, refine such a pathway by hand rather than with "Submit Changes to AI". |
| **Planned fix** | none planned (needs a D-entry for `streamlit_app.py`) |

### KI-021 — Interactions-only pathways: export or decline

| | |
|---|---|
| **Symptom** | A pathway whose only surviving processes are interactions now writes a `review_required` PWML, which may draw very little (WNT PCP: 1 of 18). A pathway that is interactions-only from Stage 2 is declined instead. |
| **Reason codes** | driver `no_reactions` (`driver.KIND_NO_REACTIONS`, returned by `driver._classify`, used in `driver._drive`); `no_drawable_process` is the orchestrator's reading in D-113 § 3 and is **not in the code** at `4befcb33` |
| **Cause** | The batch driver declines a payload with no reaction and no transport at Stage 2, while the quarantine admits one that becomes interactions-only later. |
| **Papers and legs hit** | ORCH-767 WNT PCP; `runs_verify` PMC12856317 (draws none) |
| **Evidence** | `ORCH-769c-RESULT.md` § 7 item 1; `ORCH-769-RESULT.md` § (c) |
| **Status** | **product decision D-114 AMENDMENT 2: export** (§ 3: no decline for a PWML that draws zero processes; this overrules the D-113 § 3 reading, so `no_drawable_process` is not built). Such a leg is KI-065. The Stage-2 `no_reactions` decline is unchanged (inconsistent) |
| **User workaround** | Expect a thin PWML: add the withheld interactions (KI-008) by hand where the paper gives a direction. |
| **Planned fix** | none for a decline (D-114 AMENDMENT 2 § 3). Updated by ORCH-772b; was "ORCH-773 (with D-113 § 3)" |

### KI-022 — No reaction carries defensible reaction-level support

| | |
|---|---|
| **Symptom** | No PWML; the required-field gate refuses the pathway. Five corpus legs that wrote a PWML in August now fail this way. |
| **Reason codes** | `no_defensible_reaction_support` (`reaction_support.CODE_NO_DEFENSIBLE_REACTION_SUPPORT`, issued by `reaction_support.reaction_support_issue`, enforced in `stage_contracts.validate_pre_export`) |
| **Cause** | A pathway in which no reaction's lineage carries target-paper or admitted-RAG support is refused (F-179). Identifier mapping proves identity, not that a reaction occurs. |
| **Papers and legs hit** | 8 tier-B corpus legs, 5 of them PWML-writing in August (ORCH-768 § 8 F-8) |
| **Evidence** | `ORCH-768-RESULT.md` § 6, § 8 F-8; `src/t2pw/pipeline/reaction_support.py` module docstring |
| **Status** | **by design** (D-094, F-179) |
| **User workaround** | None for a user (no file). Operator: check whether the leg's reactions lost their `provenance_lineage` (an indeterminate carrier is not refused). |
| **Planned fix** | none |

### KI-023 — A normalization exception lets an un-normalized payload reach export

| | |
|---|---|
| **Symptom** | A pathway correctly released `review_required` writes no PWML: "PWML export stopped by validation-only pre-export Stage 3 revalidation; Stage 8 did not repair the payload." The same findings were "informational" in the post-audit report. |
| **Reason codes** | `actor_schema_not_canonical` (`stage_contracts._validate_canonical_actor_rows`; `payload_models._normalized_actor_issues`); the stop message above (`streamlit_app.run_pwml_export`); stage `pwml_export` |
| **Cause** | Any exception inside strict normalization is caught by the app (`streamlit_app.py:3655` at `8b074dda`), which continues with the Stage-2 payload; the actor-schema step never runs, and the export's revalidation rejects rows still in Stage-1 shape (`{"protein": "OTOP1"}`). |
| **Papers and legs hit** | ORCH-755 sour taste (trigger: KI-024); 1 tier-B corpus leg |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D2 steps 3-7; `ORCH-758a-RESULT.md` § 8 Q1 |
| **Status** | trigger **fixed** (KI-024). The class is **open**: any other normalization exception repeats it |
| **User workaround** | None for a user (no file). |
| **Planned fix** | none planned. Sketch: carry the exception into the release state as a `review_required` reason and say why no PWML was written (needs a D-entry for `streamlit_app.py`; ORCH-758a § 8 Q1 option (b)) |

### KI-024 — Charge notation (`Ca++`) read as a composite name

| | |
|---|---|
| **Symptom** | Sour taste has no PWML (via KI-023). |
| **Reason codes** | "Composite entity '`<name>`' in /entities/compounds has no protein-like left component; unsupported." (`process_normalizer.normalize_composites`); "Composite token '`<text>`' at `<pointer>` has no protein-like left component; ..." (`process_normalizer._rewrite_token`) |
| **Cause** | `_has_plus_token` stripped one trailing `+` and read the rest as an `A + B` separator, so `Ca++`, `Ca ++`, `Fe+++`, `Na+/K+-ATPase` raised in strict mode. |
| **Papers and legs hit** | ORCH-755 sour taste |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D2; `ORCH-758a-RESULT.md` §§ 1, 7, 10 (corpus scan: 12 names changed, all charge notation) |
| **Status** | **fixed** `e9c852a5` (ORCH-758a). **Open:** the Stage-1 splitter `pipeline._split_composite_token` and the curation copy `audit_json_llm._split_composite_name` still split `Ca++` into `Ca`; real composites still lose member charges (`NADH + H+` → `H`) |
| **User workaround** | Check ions in the PWML (e.g. `Ca2+`) keep their charge; correct the compound in PathWhiz if not. |
| **Planned fix** | none planned (ORCH-758a § 8 Q2, Q4) |

### KI-025 — Stage 0 rejects a paraphrase of the requested pathway (Ara operon)

| | |
|---|---|
| **Symptom** | The run stops on the Stage-0 scope check (`scope_conflict`); Stage 1 may already have run, but nothing reaches the audit, mapping or export, and no PWML is written. Whether it happens depends on the Stage-0 wording of that draw. |
| **Reason codes** | status `scope_conflict` (`driver._STATUS_SCOPE_CONFLICT` = `rag.eligibility.OUTCOME_SCOPE_CONFLICT`); release reason `stage0_scope_conflict_stopped_the_run_before_serialization` (`driver.REASON_STAGE0_SCOPE_CONFLICT`, `release_status.SCOPE_GUARD_STOP_REASON`; `driver._finalize_scope_conflict`); artifact `scope_conflict.json` |
| **Cause** | The scope check compares Stage 0's reading with the request and rejects a reading that names the same pathway in other words: "AraC autoregulation in araBAD operon" against "Ara operon inactivation (autoregulation of AraC)", then, after rewording, "E. coli L-arabinose (P_BAD) and L-rhamnose (P_Rha) inducible promoter regulation" against "AraC autoregulation of the ara operon". The ORCH-755 Ara leg passed Stage 0 on another draw. Sibling of F-203 (a generalisation refused). |
| **Papers and legs hit** | ORCH-756 Ara (report `05`); ORCH-759 Ara (reworded, report `05`) |
| **Evidence** | `ORCH-756-RESULT.md` § 3, § 6 item 1; `ORCH-759-RESULT.md` § 2, § 4 items 1 and 3; `FINDINGS.md` F-203 |
| **Status** | **open** (policy question registered by D-110 § 1; any change is a gate change with its own D-entry). Ara is dropped from the cohort |
| **User workaround** | Re-run with the requested pathway written in the paper's own phrase. |
| **Planned fix** | none planned (needs a product-owner ruling on paraphrase matching) |

### KI-026 — Stage 0 organism conflict (sour taste: human requested, mouse paper)

| | |
|---|---|
| **Symptom** | The run stops at Stage 0 with `scope_conflict`; release `diagnostic_only`; no PWML. |
| **Reason codes** | as KI-025: `scope_conflict`, `stage0_scope_conflict_stopped_the_run_before_serialization`, `scope_conflict.json` |
| **Cause** | Stage 0 read *Mus musculus* from a review of mostly mouse work; the topic requested *Homo sapiens* (from PathBank's human pathway PW739181). Note: D-062 says such a conflict should preserve the pathway as `review_required` under the observed organism, but the driver records `diagnostic_only` and says `review_required` is not constructible at that seam (`driver._finalize_scope_conflict` docstring). |
| **Papers and legs hit** | ORCH-759 sour taste (report `06`) |
| **Evidence** | `ORCH-759-RESULT.md` § 4 item 4; D-110 § 2; D-062 |
| **Status** | **by design** for the check. Input corrected under D-110 § 2 (request *Mus musculus*): ORCH-759p and ORCH-767 sour taste wrote PWMLs. The D-062 / driver disagreement is **open** |
| **User workaround** | Request the organism the paper studies; compare against the paper, not the other organism's PathBank pathway. |
| **Planned fix** | none planned |

### KI-027 — A chunk that is all reference list fails the whole paper

| | |
|---|---|
| **Symptom** | Stage 1 fails with "Chunk 3 failed to produce valid JSON." although the reply was valid; no PWML. |
| **Reason codes** | "Chunk `<n>` failed to produce valid JSON." (`pipeline.run_stage_one_with_chunking`); attempt outcome `valid_json_zero_processes` (`extraction_diagnostics.OUTCOME_ZERO_PROCESSES`); rung-3 refusal `budget_indeterminate` (`extraction_ladder.SKIP_BUDGET_INDETERMINATE`); after the fix, boundary record `stage1_empty_chunk` (`pipeline.BOUNDARY_STAGE1_EMPTY_CHUNK`, written by `pipeline._record_empty_chunk`) |
| **Cause** | The 199,748-character WNT paper is chunked; chunk 3 is 90 % reference list. Every draw correctly returned empty JSON (`finish_reason=stop`), the ensemble re-raised, and one empty chunk failed the paper. The message misreports valid JSON as invalid. |
| **Papers and legs hit** | ORCH-755 Wnt/Ca2+ strict, WNT PCP research |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D3; `evidence/orch758_wnt_chunk3_terms.txt`; `ORCH-758b-RESULT.md` §§ 1, 10-11 |
| **Status** | **fixed** `6b7766ec` (ORCH-758b: a chunk that is valid-empty with `stop` on every attempt no longer fails the paper; every other failure stays fatal). **Open:** whether such a release should be `review_required` (ORCH-758b § 7 Q1); the remaining fatal paths still say "failed to produce valid JSON" for a `length` stop (Q5) |
| **User workaround** | None needed after the fix. |
| **Planned fix** | none |

### KI-028 — Time limits: the 3,600 s leg, the fixed 3,600 s per-interaction cap, and nothing kept on timeout

| | |
|---|---|
| **Symptom** | The leg ends in a timeout with no payload files; 30-40 minutes of Stage-1 work survive only under `tmp/`. Under the 10,800 s ruling, Wnt/Ca2+ still stopped at 5,296 s with "AppTest script run timed out after 3600.0(s)". |
| **Reason codes** | `budget_exhausted` (`deadline.BUDGET_EXHAUSTED`); `outer_parent_kill`, `in_process_deadline` (`leg_trace.SOURCE_OUTER_PARENT_KILL`, `SOURCE_IN_PROCESS_DEADLINE`); `operation_timeout`; driver messages "extraction did not finish inside the time budget", "audit and DB mapping did not finish inside the time budget", "PWML export did not finish inside the time budget" (`driver._drive`); AppTest detail "AppTest script run timed out after 3600.0(s)" (Streamlit's own message, recorded in `RESULT.txt`) |
| **Cause** | (1) Three sequential draws per chunk took 565-3,323 s (median about 1,400 s) against about 230 s at one draw; chlorophyll's Stage 1 alone took 2,103 s with no empty attempt. (2) `driver.run_one` defaulted every app interaction to `DEFAULT_APP_TIMEOUT = 3600.0` whatever the leg budget, and no caller could change it. (3) AppTest stops a script only when the pipeline next yields, so a cap overruns by the work in flight (1,690 s here). (4) No `LegDeadline` is active in production (`activate_leg_deadline` has no caller; P2-04), and a parent kill preserves nothing. |
| **Papers and legs hit** | ORCH-755: 7 timeouts (menaquinone strict and research, chlorophyll strict and research, Ara, WNT PCP strict, Wnt/Ca2+ research); ORCH-756 chlorophyll (5,400 s); ORCH-759 Wnt/Ca2+ (the per-interaction cap) |
| **Evidence** | `ORCH-758-RESULT.md` § 3, § 4 D7; `evidence/orch758_leg_timeline.txt`, `orch758_reference_timing.txt`; `ORCH-756-RESULT.md` § 2; `ORCH-759-RESULT.md` § 4 item 5; `ORCH-759d-RESULT.md` §§ 1, 6, 7 |
| **Status** | per-interaction cap **fixed** `9f7d64ee` (ORCH-759d: it follows the leg budget). Leg length mitigated by an orchestrator operating ruling (10,800 s per leg, `ORCH-756-RESULT.md` § 2; not a D-entry). **Open:** draws run sequentially; a timed-out leg persists no payload (PRODUCT_CONTRACT § 9 "preserve" unmet); no leg deadline; a leg that reaches its child deadline mid-Stage-2 overruns the 120 s grace and is killed by the parent, losing its artifacts (ORCH-759d § 7 Q1) |
| **User workaround** | None for a user. Operator: give long papers 10,800 s and re-run a timed-out leg. |
| **Planned fix** | none planned. Sketch: persist the Stage-1 pool on timeout, run draws in parallel (ORCH-758 § 9 item 5), and let the pipeline refuse a call that cannot finish before the child deadline (ORCH-759d § 7 Q1 (c)) |

---

## C. Wrong or doubtful content inside a written PWML

These do not stop a PWML. They put something wrong, or something the paper does not state, into one.

### KI-029 — A class name narrowed to one specific molecule (prenyl diphosphate → DMAPP)

| | |
|---|---|
| **Symptom** | The MenA reaction draws dimethylallyl pyrophosphate where the paper says "prenyl diphosphate" (a class). Gold scores it a partial match (R8). |
| **Reason codes** | after the fix: refusal `class_term_never_narrowed_to_specific_molecule` (`compound_resolution.CLASS_TERM_NEVER_NARROWED`, set in `compound_resolution._db_match_rule_decision`, Stage-6 `map_ids._refuse_class_term_identity`); `db_status` `identity_refused_review_required` (`db_resolver.IDENTITY_REFUSED_STATUS`); before it: `db_match.chosen_rule` `exact_short_name_or_synonym` |
| **Cause** | Pre-freeze compound resolution matched the term to PathBank compound 874 (DMAPP) through that row's synonym "prenyl diphosphate", and D-028's rule admitted it, although Stage 6's name gate had refused the same answer. |
| **Papers and legs hit** | ORCH-756 and ORCH-759 menaquinone |
| **Evidence** | `ORCH-764-RESULT.md` §§ 1-2, § 13; `evidence/orch759_gold_accuracy.txt` (R8 rows); `evidence/orch756_gold_accuracy.txt` |
| **Status** | **fixed** `38188982` (ORCH-764, D-109 § 2): a listed class term keeps the paper's word and carries no identity. **Open:** the compound now ships with no identity (no committed ChEBI class id; ORCH-764 § 7 Q1); the class list holds two terms only; the two compound gates still disagree in general (Q4) |
| **User workaround** | In PathWhiz, keep the compound as "prenyl diphosphate" (or attach the polyprenyl diphosphate class if you curate one); do not accept DMAPP. |
| **Planned fix** | none planned (ORCH-764 § 7 Q1-Q2) |

### KI-030 — Wrong compound identity in a written PWML

| | |
|---|---|
| **Symptom** | A compound in the PWML carries another molecule's database id: `menaquinone (MK)` → Menadione (PathBank 1260), `OSB` → phosphoribosyl formamidocarboxamide (PathBank 1111), `demethylmenaquinone (DMK)` → 2-demethylmenaquinone 8 (PathBank 41547). |
| **Reason codes** | none in the pipeline (no gate catches it); the gold scorer marks `wrong_identity` |
| **Cause** | Not diagnosed. All three rest on the mapper cache only (`independent=False`); MK → Menadione is a wrong molecule, not a class narrowing (ORCH-764 § 7 Q5). |
| **Papers and legs hit** | ORCH-759 menaquinone (MK, DMK); ORCH-759p menaquinone (OSB); ORCH-756 menaquinone (DMK) |
| **Evidence** | `evidence/orch759_gold_accuracy.txt` lines 207-208; `evidence/orch759p_gold_accuracy.txt` line 223; `evidence/orch756_gold_accuracy.txt` line 205 |
| **Status** | **open** |
| **User workaround** | Before importing, check every compound's HMDB/PathBank id against its name; fix menaquinone, demethylmenaquinone and OSB by hand. |
| **Planned fix** | none planned. Sketch: a read-only diagnosis of the mapper-cache entries behind the three ids |

### KI-031 — Extra or wrong catalysts added after Stage 1 (EntE complex on the EntF step; EntCB complex)

| | |
|---|---|
| **Symptom** | PWML reactions list enzymes Stage 1 did not: the EntF row also lists EntE complex and "enterobactin synthase complex", the EntA row lists isochorismatase and `EntCB complex`. `EntCB complex` is a name the gold forbids (a pairwise assembly name) and it bound to PathBank complex 1143 "Isochorismate synthase". |
| **Reason codes** | none in the pipeline; gold scorer `wrong_identity` (`EntCB complex` → `pathbank_complex_id:1143`; `enterobactin synthase` → UniProt P10378, EntE) |
| **Cause** | Later stages (Stage 2, audit) add catalysts; the evaluator scores an enzyme correct when any listed one matches, so they are not penalised. PathWhiz binds a complex named like an existing one of the same species to that row and ignores the PWML's components (`protein_complex.rb:313-334`; ORCH-761 § 1.3). |
| **Papers and legs hit** | ORCH-756 enterobactin (PWML P 0.714); ORCH-767 enterobactin (final payload) |
| **Evidence** | `evidence/orch756_gold_accuracy.txt` (reader notes); `evidence/orch767_gold_accuracy.txt` line 66; `evidence/orch769_diag/769a/02-diag.out.txt`; `ORCH-761-RESULT.md` § 9 |
| **Status** | **open**; the invented-wrapper part is KI-012 (D-113 § 1, ORCH-769a) |
| **User workaround** | Check each reaction's enzymes against the quote in the text record and delete catalysts the paper does not name, especially names joining several proteins (`EntCB`). |
| **Planned fix** | ORCH-769a (invented wrapper only); none planned for other added catalysts |

### KI-032 — Rows the paper does not state for this pathway reach the PWML

| | |
|---|---|
| **Symptom** | The PWML draws a reaction the paper uses only as an assay reporter (LDH: pyruvate → lactate), a reaction transcribed from another paper by retrieval (a second EntA row with a second substrate node `2,3-dihydro-DHB`), or a reaction read from a kinetic-model term (`isochorismate → DHNA (MenD)` from `kcatD[MenD][B]... = kDHNA[DHNA]`). |
| **Reason codes** | none in the pipeline; gold row classes `paper_stated_off_pathway`, `unsupported`, `indeterminate_external`; evidence verdicts `not_in_paper`, `dump_only` |
| **Cause** | Stage 2 and the RAG round add rows; the per-row support rule is a leg-level floor (KI-022), not per-row enforcement. |
| **Papers and legs hit** | ORCH-756 enterobactin (P 0.714); ORCH-767 menaquinone (P 0.909, 11 rows for 9 gold) |
| **Evidence** | `evidence/orch756_gold_accuracy.txt` rows 5-6; `evidence/orch767_gold_accuracy.txt` line 149; `src/t2pw/pipeline/reaction_support.py` docstring |
| **Status** | **open** |
| **User workaround** | Delete PWML reactions whose quote in the text record is not from the target paper, is an assay description, or is an equation term. |
| **Planned fix** | none planned. Sketch: show each row's lineage (paper, Stage 2, RAG) in the "Omitted/kept" text so a reviewer sees it (ORCH-772 scope) |

### KI-033 — A protein identity borrowed from another strain of the same subspecies

| | |
|---|---|
| **Symptom** | Menaquinone proteins carry UniProt records of strain IL1403 for a KF147 paper; PathWhiz files each protein under IL1403. |
| **Reason codes** | flag "UniProt `<acc>`, from strain `<strain>`" in the `BORROWED IDENTITIES (n)` section of the human `.txt` (`app/reaction_evidence`, key `borrowed_protein_identities`) |
| **Cause** | KF147 has no UniProtKB entries for these genes; D-108 allows a same-subspecies record when gene and function match, never across a subspecies. PathWhiz's `check_uniprot` re-files a protein under its accession's organism on create (`protein.rb:171-203`). |
| **Papers and legs hit** | menaquinone after ORCH-762 (7 proteins) |
| **Evidence** | `ORCH-762-RESULT.md` §§ 3-4, § 12 B2; `ORCH-761-RESULT.md` § 1.2 |
| **Status** | **by design** (D-108) |
| **User workaround** | Review each flagged protein; keep it if the ortholog is acceptable for your curation, otherwise remove its accession and treat it as KI-001. |
| **Planned fix** | none |

### KI-034 — PathWhiz drops a failing element silently and reports success

| | |
|---|---|
| **Symptom** | An upload succeeds but the pathway in PathWhiz has fewer proteins, complexes or reactions than the PWML; a protein appears renamed or under another organism. |
| **Reason codes** | none in T2PW (PathWhiz-side behaviour) |
| **Cause** | PathWhiz skips an element whose save fails, records the error, and the upload response returns only the pathway id (`pathways_controller.rb:225-241`). The loss cascades: a complex loses its component, a reaction with `spontaneous=false` loses its enzyme and fails (`reaction.rb:90-96`). On create, `check_uniprot` overwrites a protein's name and species from UniProt; an empty species defaults to *Homo sapiens*. |
| **Papers and legs hit** | static reading of PathWhiz source; no import was run |
| **Evidence** | `ORCH-761-RESULT.md` § 0 item 1, § 1.1-1.2, § 10 |
| **Status** | **by design** (PathWhiz behaviour; outside T2PW). T2PW's strict gate is the only place the loss is visible |
| **User workaround** | After import, compare PathWhiz's counts of proteins, complexes and reactions with the PWML and with the text record's coverage line; re-add anything missing. |
| **Planned fix** | none |

### KI-035 — The writer emits fabricated `pwp-id` values that collide with PathBank complexes

| | |
|---|---|
| **Symptom** | Two different complexes can share a `pwp_id` in PathWhiz; e.g. `PW_P040000` is on PathBank complex 40000 and on complexes 431773 "Hexokinase complex", 431796 and 431802. |
| **Reason codes** | none |
| **Cause** | `writer.py:1311` writes `pwp-id` as `PW_P{document id:06d}`; PathWhiz keeps a non-blank `pwp_id` and has no unique index on it (`protein_complex.rb:383-389`). |
| **Papers and legs hit** | every written PWML with a complex; collisions confirmed for six ids only (the census query timed out) |
| **Evidence** | `ORCH-761-RESULT.md` § 9, § 10; `evidence/orch761_pathbank_probe5.txt` |
| **Status** | **open** (observed, not adjudicated) |
| **User workaround** | None needed for the import itself; do not rely on `pwp_id` to identify an imported complex. |
| **Planned fix** | none planned. Sketch: omit `pwp-id` so PathWhiz assigns one (needs a writer lane and an import check) |

### KI-036 — Complex species are written as the pathway default; a compound-only payload writes no species

| | |
|---|---|
| **Symptom** | A complex's species in the PWML is the pathway's first species, not its own; a protein whose species text matches no IR species is filed under the first one; a compound-only payload writes `<species-id nil="true">`, which PathWhiz files under *Homo sapiens*. |
| **Reason codes** | none |
| **Cause** | `writer.py:1310` gives every complex the default species; a protein falls back to it (`writer.py:1188-1214`); with no IR species the default is `None` (`writer.py:1002-1003`, `validate.py:308-311`). |
| **Papers and legs hit** | static reading; irrelevant for the single-species papers run so far |
| **Evidence** | `ORCH-761-RESULT.md` § 9; `ORCH-762-RESULT.md` § 5 |
| **Status** | **open** (report only, no code) |
| **User workaround** | For multi-species pathways, check each complex's and protein's species in PathWhiz after import. |
| **Planned fix** | none planned |

### KI-037 — The gate accepts an identity the writer does not serialise

| | |
|---|---|
| **Symptom** | (latent) A protein passes the identity gate and imports with no `uniprot-id`. |
| **Reason codes** | none |
| **Cause** | The gate's predicate (`entity_identity.py:62-84`) also accepts an id found only in `ids` or `mapping_meta`; the writer reads only `mapped_ids.uniprot`, `uniprot`, `uniprot_id` (`writer.py:1243-1256`). PathWhiz would then skip the protein (KI-034). |
| **Papers and legs hit** | not observed |
| **Evidence** | `ORCH-761-RESULT.md` § 2 (row `protein_missing_external_identity`) |
| **Status** | **open** (latent) |
| **User workaround** | If PathWhiz reports a missing UniProt on import, add it from the text record. |
| **Planned fix** | none planned. Sketch: make the predicate read only the fields the writer serialises |

### KI-038 — The coverage matcher does not see through wrapper names

| | |
|---|---|
| **Symptom** | A requested core term stops matching once its enzyme is wrapped, so a process moves from core to auxiliary and coverage falls (PMC11405693: 7/11 → 6/11). Release status can turn on it. |
| **Reason codes** | review reason `requested_core_coverage_below_minimum:<x><<y>` (`release_status.COVERAGE_REASON_BELOW_MINIMUM`; wrapped as `minimum_core:...` in quarantine `review_reasons`) |
| **Cause** | `strict_quarantine._term_matches` matches by substring against the actor's name; `CYP73A5 (AtCYP73A5)` matches `AtCYP73A5`, not `AtCYP73A5 complex`. Pre-existing for every main-loop wrapper. |
| **Papers and legs hit** | PMC11405693 (archive replay) |
| **Evidence** | `ORCH-759e-RESULT.md` § 8 F4, § 10 item 3 |
| **Status** | **open** |
| **User workaround** | None needed in PathWhiz; read the coverage figure as a lower bound. |
| **Planned fix** | none planned. Sketch: `_process_core_terms` adds a generated wrapper's component name (moves coverage baselines) |

---

## D. Duplicate rows and merge quality

Three draws produce near-duplicate rows. These entries are about which duplicates are merged, which are wrongly
kept apart, and which were once wrongly merged. A kept-apart duplicate reaches the PWML as a second reaction.

### KI-039 — Name-form variance across draws inflates the pooled rows (chlorophyll 41)

| | |
|---|---|
| **Symptom** | Chlorophyll's three draws pooled to 41 rows: draws of 16, 19 and 18 in ORCH-751, of 18, 19 and 18 in ORCH-755, where ORCH-758 counted about 21 distinct reactions among the 41. |
| **Reason codes** | merge flags `typo_snapped`, `undefined_state_treated_as_absent` (`stage_one_ensemble.TYPO_FLAG`, `UNDEFINED_STATE_FLAG`) in the side file and the human `.txt` only |
| **Cause** | Not cofactors (ORCH-751's theory, disproved by ORCH-752). Draws write `full name (ABBR)` versus `ABBR`, synonyms (`protoheme` / `heme b`), and one typo per draw, all in defining participants. |
| **Papers and legs hit** | ORCH-751 chlorophyll (41), ORCH-755 chlorophyll (41) |
| **Evidence** | `ORCH-751-RESULT.md` § 2.2; `ORCH-752-RESULT.md` § 2, `evidence/orch752_chlorophyll_key_inflation.txt`; `ORCH-754a-RESULT.md` § 10.4; `ORCH-758-RESULT.md` § 5 |
| **Status** | **fixed in part** `adc49c40` (ORCH-754a: the paper as dictionary, typo snaps, D-104 § 3 and AMENDMENTS 3-4): 41 → 29 on the ORCH-751 draws. The rest is KI-041, KI-042, KI-044, KI-045 and genuine variance. ORCH-755's 41 had 14 merges and no flag; "0 flagged" does not mean no merge |
| **User workaround** | In the text record, rows marked 1/3 support are the likely duplicates; delete a PWML reaction that repeats another with a different spelling. |
| **Planned fix** | see the residual entries |

### KI-040 — Chimeric merges in the first ensemble code (NAD-IDH + NADP-IDH)

| | |
|---|---|
| **Symptom** | (in pushed `70d05e03` only) Two different reactions merged into one row carrying both cofactor pairs; one enzyme vanished with no flag. |
| **Reason codes** | none |
| **Cause** | ORCH-752's cofactor-blind key ignored cofactors, state and enzymes. |
| **Papers and legs hit** | reviewer probe only; no paper run used that code at more than 1 draw before the fix |
| **Evidence** | `ORCH-754-RESULT.md` § 3.2 finding 1; `ORCH-754a-RESULT.md` § 1.5, § 10.1 B3 |
| **Status** | **fixed** `adc49c40` (ORCH-754a: only compatible rows merge; cofactors compared as nested sets) |
| **User workaround** | None needed. |
| **Planned fix** | none |

### KI-041 — A paper-defined abbreviation the matcher cannot read (DMK, MK, MV-Chlide a)

| | |
|---|---|
| **Symptom** | Menaquinone keeps 11 rows for 9 reactions: MenA and MenG appear twice (`demethylmenaquinone (DMK)` vs `DMK`). |
| **Reason codes** | none on the merged code (ORCH-765's flag `parenthetical_definition_bridge` exists only on the unmerged branch `orch765/parenthetical-bridge`) |
| **Cause** | Schwartz-Hearst requires the short form's letters in the long form; "K" is in neither "demethylmenaquinone" nor "menaquinone". The paper does define both pairs. |
| **Papers and legs hit** | ORCH-759 and ORCH-767 menaquinone; chlorophyll (`MV-Chlide a`) |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D6; `ORCH-763-RESULT.md` § 4, § 9 Q1; ORCH-765-RESULT § 13.2, § 14 (on branch, `9fa8f3be`) |
| **Status** | **open**. **Product decision** D-110 § 3, AMENDMENTS 1-2 (bridge only when two draws write the pair and the paper contains it). **ORCH-765 NOT MERGED**: its final round still bridges `Optical (OD)` and `Luria (LB)` when two draws write them |
| **User workaround** | Merge the two MenA rows and the two MenG rows by hand in PathWhiz (delete the `DMK` duplicates). |
| **Planned fix** | none planned. A new attempt needs a non-lexical refusal for two-letter short forms, or a ruling accepting OD/LB as flagged bridges (ORCH-765 § 14) |

### KI-042 — The paper's own shared-locant phrase is not read (`uroporphyrinogen (Urogen) III`)

| | |
|---|---|
| **Symptom** | Chlorophyll keeps about 5 duplicate rows where one draw copied the paper's `X (Y) L` phrase verbatim. |
| **Reason codes** | none |
| **Cause** | `paper_lexicon.abbreviation_pairs` derives the pair `Urogen III` = `uroporphyrinogen III`, but `long_form` handles `X (Y)` only when the parenthesis is last (`_TRAILING_PAREN`). |
| **Papers and legs hit** | ORCH-755 chlorophyll |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D5, § 5; `evidence/orch758_chlorophyll_lexicon.txt` |
| **Status** | **open** |
| **User workaround** | Delete the duplicate porphyrinogen rows in PathWhiz. |
| **Planned fix** | none planned (ORCH-758 § 9 item 9: about 2 h plus review) |

### KI-043 — Location splits (cytosol vs intracellular)

| | |
|---|---|
| **Symptom** | Menaquinone carried 16 rows for 9 reactions: 5 MenH-MenI duplicates kept apart only by "cytosol" vs "intracellular". |
| **Reason codes** | after the fix, merge flag `nested_location_merged` (`stage_one_ensemble.NESTED_LOCATION_FLAG`) in the side file and `.txt` |
| **Cause** | The merge treated any two different defined locations as a conflict. |
| **Papers and legs hit** | ORCH-759 menaquinone; chlorophyll (`thylakoid membrane` vs `plastid`, 5 splits) |
| **Evidence** | `ORCH-763-RESULT.md` § 1, § 6, § 12; `evidence/orch759_gold_accuracy.txt` |
| **Status** | **fixed** `2161a9ab` (ORCH-763, D-109 § 1): nested locations merge, the more specific is kept, flagged; menaquinone 16 → 11. **Open:** H1 (claims chain through an empty species) and H2 (one state label defined two ways), not observed in kept draws; the thylakoid ⊂ chloroplast edge is approximate (wrong for cyanobacteria) |
| **User workaround** | None needed; review merges flagged `nested_location_merged`. |
| **Planned fix** | none planned (ORCH-763 § 9 Q2) |

### KI-044 — Isoform, gene and subunit names keep the same reaction apart (POR/PORB, CPO/CPO1)

| | |
|---|---|
| **Symptom** | Chlorophyll keeps 7 of its 9 surplus rows because draws name an enzyme family in one and an isoform, gene or complex name in another. |
| **Reason codes** | none (merge record "kept apart", reason enzyme sets not nested) |
| **Cause** | D-104 AMENDMENT 3 keeps non-nested enzyme sets apart; `{POR}` vs `{PORB}` are not nested. |
| **Papers and legs hit** | ORCH-755, ORCH-759 chlorophyll |
| **Evidence** | `ORCH-758-RESULT.md` § 5 table, § 9 item 7; `ORCH-763-RESULT.md` § 6, § 9 Q5 |
| **Status** | **open** (policy question; a change is a merge-rule amendment) |
| **User workaround** | Merge the duplicate reactions in PathWhiz, keeping the enzyme name the paper uses for that step. |
| **Planned fix** | none planned |

### KI-045 — The database-identity merge (rule 3) never ran in the app; diagnostics were lost

| | |
|---|---|
| **Symptom** | Synonym duplicates rule 3 should merge (`protoheme` / `heme b`) reached the PWML; no `ensemble_identity_merge_report.json` in any of 312 run directories; audit and gap summaries missing; the leg directory's Stage-0/1 diagnostics replaced by the post-pipeline's. |
| **Reason codes** | none |
| **Cause** | Rule 3 and the side-file writer read a ContextVar recorder set at Stage 1. The batch drives the app as two AppTest script runs; each starts a new thread with an empty context, so Stage 6 saw zero pooled draws. The post-pipeline then published its detached recorder over Stage 0/1's (186 leg directories). |
| **Papers and legs hit** | every 3-draw app leg before the fix (ORCH-755) |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D4, `evidence/orch758_rule3_probe.txt`; `ORCH-759b-RESULT.md` § 1, § 8 F1; proof `ORCH-759-RESULT.md` § 7 |
| **Status** | **fixed** `45623ceb` (ORCH-759b, D-107; protected file +3 lines). **Open:** the legacy SBML export button does not re-enter (ORCH-759b § 7 item 2) |
| **User workaround** | None needed. |
| **Planned fix** | none |

---

## E. Extraction and the early stages

### KI-046 — Draws differ, and each run meets different downstream failures

| | |
|---|---|
| **Symptom** | One prompt on one paper matched 9, then 7, then 4 reactions; chlorophyll wrote a PWML in ORCH-759p and not in ORCH-767 on identical code. |
| **Reason codes** | none |
| **Cause** | Run-to-run variance at temperature 0 (noise floor about ±0.25 F1 per draw). Pooling recovers random misses, not systematic ones (enterobactin missed the same reaction in 3 of 3 draws in ORCH-751). |
| **Papers and legs hit** | all; quantified on enterobactin and menaquinone |
| **Evidence** | `ORCH-747-RESULT.md` §§ 11-12; `ORCH-751-RESULT.md` § 2.1; D-112 preamble |
| **Status** | **by design** (D-104 § 2: 3 draws; D-112: fixes measured on a frozen corpus) |
| **User workaround** | Re-run a paper whose PWML is missing a step the paper states; compare the 1/3-support rows. |
| **Planned fix** | none |

### KI-047 — Regulatory papers yield no reactions; is transcription a reaction?

| | |
|---|---|
| **Symptom** | Ara operon, sour taste and araC produce 0 reactions in every draw; their content is interactions (see KI-008, KI-021). |
| **Reason codes** | none |
| **Cause** | The prompt does not treat expression or transcription as a reaction; whether it should (`v06` vs `v07`) is undecided. |
| **Papers and legs hit** | ORCH-747 araC (all 16 variants recall 0), ORCH-751 Ara and sour taste |
| **Evidence** | `ORCH-747-RESULT.md` § 10.3, § 7 item 2; `ORCH-751-RESULT.md` § 2.3 |
| **Status** | **open** (D-open-2, carried in D-104 "still open") |
| **User workaround** | Draw the gene-expression steps in PathWhiz by hand from the text record's interactions. |
| **Planned fix** | none planned (needs a product ruling) |

### KI-048 — Two-column PDF text is interleaved

| | |
|---|---|
| **Symptom** | Quotes break mid-word and continue with the other column's text, and still "verify". |
| **Reason codes** | none |
| **Cause** | The production PDF reader groups words into lines by vertical position; two columns share a baseline. Quote verification folds whitespace, so shredded text still matches shredded quotes. |
| **Papers and legs hit** | PMC319880 (scanned 1981 PNAS; three text revisions needed) |
| **Evidence** | `ORCH-747-RESULT.md` § 5 F-747-1; `evidence/orch747_pdf_columns.py` |
| **Status** | **open** (the reader is still defective for any two-column PDF) |
| **User workaround** | Prefer a paper's HTML or XML full text over its PDF. |
| **Planned fix** | none planned |

### KI-049 — Five process-row fields are deleted before any reader sees them

| | |
|---|---|
| **Symptom** | (silent) Process rows lose `class`, `confidence`, `provenance`, `source_refs` and `spontaneous`; a `review_required` mark that depends on `provenance` never fires on a process row. |
| **Reason codes** | none |
| **Cause** | `_clean_processes` rebuilds every row from a key set without those fields (`pipeline.py:2229-2265` reactions, `:2383-2404` interactions at ORCH-747's base); `stage_one_boundary._paper_entry` tests `row.get("provenance")`. |
| **Papers and legs hit** | all legs |
| **Evidence** | `ORCH-747-RESULT.md` § 5 F-747-7 |
| **Status** | **open** (not fixed; out of ORCH-747's scope) |
| **User workaround** | None. |
| **Planned fix** | none planned |

### KI-050 — Alias rows and split names appear after Stage 1 (`SAME_AS`, `CAP` beside its long name)

| | |
|---|---|
| **Symptom** | Interactions like "cAMP is identical to cyclic AMP" (`relationship: SAME_AS`) and one protein under two names. |
| **Reason codes** | none |
| **Cause** | They originate in the retrieval-and-merge stage, not in extraction; the curator's `SAME_AS` cleanup only catches identical strings. |
| **Papers and legs hit** | PMC319880 run (ORCH-747) |
| **Evidence** | `ORCH-747-RESULT.md` § 5 F-747-5 |
| **Status** | **open** |
| **User workaround** | Delete `SAME_AS` rows and merge the two names into one element in PathWhiz. |
| **Planned fix** | none planned |

### KI-051 — An entity of the wrong kind from Stage 1 or 2 (RyhB, a small RNA, typed as a protein)

| | |
|---|---|
| **Symptom** | Interactions involving RyhB are withheld as a protein with no UniProt id (6 rows). |
| **Reason codes** | `protein_endpoint_missing_external_identity` (see KI-009) |
| **Cause** | Entity typing upstream of Stage 6. Wrapping it would have been wrong; withholding is correct. |
| **Papers and legs hit** | committed enterobactin-family legs in the ORCH-754b replay |
| **Evidence** | `ORCH-754b-RESULT.md` § 12 item 2 |
| **Status** | **open** (not routed to an owner) |
| **User workaround** | Add RyhB in PathWhiz as a nucleic acid (RNA) and draw its interactions from it. |
| **Planned fix** | none planned |

### KI-052 — The live curator prompt allowed ungated interaction adds

| | |
|---|---|
| **Symptom** | (in `70d05e03` only) The curator LLM could add interactions and nucleic-acid entities at confidence ≥ 0.75; the code never sent it the evidence text and no gate checked the add. |
| **Reason codes** | none |
| **Cause** | An ORCH-747 edit to `pathway_curator_system.txt` (§ 5 and an exception to "Do not add new reactions or transports"). |
| **Papers and legs hit** | none established; whether any live-curator app run used the text is unchecked |
| **Evidence** | `ORCH-754-RESULT.md` § 3.2 finding 2; `ORCH-754c-RESULT.md` § 7 Q6-Q7, § 8 |
| **Status** | **fixed** `ca852bb6` (ORCH-754c revert, D-104 AMENDMENT 1 § 1). **Open:** ORCH-754c Q6 (scan runs from that window); the code still does not stop the curator proposing an interaction (Q7) |
| **User workaround** | For PWMLs made with the live curator between 2026-10-15 (the ORCH-747 working tree) and the revert merge `ca852bb6` (2026-10-19 14:38 -0600), check that every interaction has a quote from the paper. |
| **Planned fix** | none planned |

### KI-053 — The paper's full text cannot be fetched

| | |
|---|---|
| **Symptom** | Most papers a curator cited cannot be run; three cohort pathways have no usable paper. |
| **Reason codes** | none (Europe PMC full-text endpoint HTTP 500) |
| **Cause** | Europe PMC answers 200 for `isOpenAccess=Y` and 500 for `=N`, with no exception in 40+ fetches; 13 of 36 cited papers were fetchable. A review that restates another paper cannot corroborate it. |
| **Papers and legs hit** | ORCH-751 cohort selection (*M. tuberculosis*, olfactory OR1A1, calcium signalling) |
| **Evidence** | `ORCH-751-RESULT.md` § 1, § 4 item 6 |
| **Status** | **open** (data access; D-104 lists the three pathways as still open) |
| **User workaround** | Supply the full text yourself (licensed copy) or choose an open-access primary paper. |
| **Planned fix** | none planned |

---

## F. LLM, runtime and harness

### KI-054 — Small token budgets return empty `finish_reason=length` completions

| | |
|---|---|
| **Symptom** | Legs time out or fail gates after hours of retries: Stage 2 empty 65 of 95 attempts, audit 35/38, curator 42/43 (ORCH-755); no strict PWML that night. |
| **Reason codes** | client log "LLM returned an empty completion for `<stage>` (model `<m>`, finish_reason=length, ...)" (`llm/client.chat_detailed`); trace `model_attempt` rows with `finish_reason=length` and empty content |
| **Cause** | The reasoning model spends its whole completion budget reasoning and writes nothing (F-199). 3-draw payloads are about twice as large, and the UI's Stage 2 budget had fallen from 20,000 to 16,000 (`_bounded_env_int`). Most other budgets were hardcoded small. |
| **Papers and legs hit** | ORCH-755 all legs; ORCH-756 phase 1 (RAG prose 24/69, audit 14/15); ORCH-747 grid at 16,000 (9 of 10 variants) |
| **Evidence** | `ORCH-755-RESULT.md` § 4, `evidence/orch755_trace_stats.txt`; `ORCH-756-RESULT.md` § 3; `ORCH-757-RESULT.md` §§ 1-2, § 12; `ORCH-747-RESULT.md` § 3 |
| **Status** | **fixed** `23753144` (ORCH-757, D-105 § 2: every call site requests the configured budget, clamped to the smallest failover cap) plus `.env` 128,000 (extraction 64,000), D-105 § 1. ORCH-759: 1 empty in 309 attempts |
| **User workaround** | Operator: keep `LLM_MAX_TOKENS` and the stage variables at 128,000 (extraction 64,000). |
| **Planned fix** | none (residual is KI-055) |

### KI-055 — Reasoning still exhausts 128,000 tokens on some calls; 20-26 minute calls end in nothing

| | |
|---|---|
| **Symptom** | One call can take 20-26 minutes and end empty or with a provider error, costing the whole wait. ORCH-767: Stage 2 empty 6 of 34, three calls over 20 minutes, one provider error. |
| **Reason codes** | `finish_reason=length` with empty content; provider `finish_reason=error` (trace rows) |
| **Cause** | With no small cap, a reasoning model may reason to 128k. No per-call timeout exists in the client. |
| **Papers and legs hit** | ORCH-756 chlorophyll Stage 2 (21 and 26 min, one `error` after 1,585 s); ORCH-759p audit (~1,246 s, empty); ORCH-767 (Stage 2 1,216 s empty; audit 1,410 s empty; audit 1,277 s `error`) |
| **Evidence** | `ORCH-756-RESULT.md` § 4, `evidence/orch756_leg_stats.txt`; `evidence/orch759p_leg_stats.txt`; `ORCH-767-RESULT.md` "LLM health", `evidence/orch767_trace_stats.txt`, `orch767_leg_stats.txt` |
| **Status** | **open** (lowering Stage 2's budget was ruled out by `main`, 2026-10-21) |
| **User workaround** | Operator: allow 10,800 s per leg; re-run a leg that failed on an empty completion. |
| **Planned fix** | none planned. Sketch: a per-call timeout in the client, or disable reasoning on short-answer calls (F-199's measured remedy) |

### KI-056 — The extraction budget is capped for a fallback that is never called

| | |
|---|---|
| **Symptom** | Stage-1 empty-length attempts at 64,000 cost 300-950 s each, about 8 a night (ORCH-755). |
| **Reason codes** | rung-3 refusal `budget_indeterminate` (`extraction_ladder.SKIP_BUDGET_INDETERMINATE`) |
| **Cause** | Extraction stays at 64,000 because the fallback model caps output at 65,536; but rung 3 is unreachable (no `LegDeadline` active, P2-04) and the client retries the same model on empty replies. Every attempt went to `deepseek/deepseek-v4-flash`. |
| **Papers and legs hit** | ORCH-755 (8 of 111 Stage-1 attempts); ORCH-759 Wnt/Ca2+ (1) |
| **Evidence** | `ORCH-758-RESULT.md` § 3, § 4 D9 |
| **Status** | **open** (product question, ORCH-758 § 9 item 4) |
| **User workaround** | None. |
| **Planned fix** | none planned. Options: raise extraction to 128k for deepseek, or make the fallback reachable first |

### KI-057 — `.env` overrides the shell and the code default (draw count)

| | |
|---|---|
| **Symptom** | A shell `T2PW_EXTRACTION_DRAWS=1` is ignored when `.env` says 3; 24 tests failed in the primary checkout. |
| **Reason codes** | none |
| **Cause** | `llm/client.py` runs `load_dotenv(override=True)`, so `.env` wins over the environment. |
| **Papers and legs hit** | test suite in the primary checkout (`test_early_failure_replay.py`); every operator run |
| **Evidence** | `ORCH-754-RESULT.md` § 3.2 finding 4; `ORCH-754a-RESULT.md` § 1.8, § 10.2 N7 |
| **Status** | tests **fixed** `adc49c40` (ORCH-754a pins N=1 in the suite). The override itself is unchanged; subprocess children of tests are not pinned (**open**, N7) |
| **User workaround** | Operator: to change the draw count, run from a worktree with its own `.env` (as ORCH-756 did), not by exporting the variable. |
| **Planned fix** | none planned |

### KI-058 — The replay corpus cannot measure Stage-3 or Stage-6 fixes

| | |
|---|---|
| **Symptom** | A merged Stage-6 fix (ORCH-759e) changes no corpus verdict; a leg it fixes still fails on the corpus. |
| **Reason codes** | none |
| **Cause** | The no-LLM checkpoint is `final_mapped.json`, which is downstream of normalization and mapping. Ten corpus failures are LLM-originating and have no checkpoint. Pre-freeze legs need the PathBank DB. The corpus inputs are untracked files in the primary checkout and two agent worktrees. |
| **Papers and legs hit** | ORCH-759 chlorophyll (still `reaction_enzyme_must_be_protein_complex` on the corpus) |
| **Evidence** | `ORCH-768-RESULT.md` § 6, § 8 F-4 and F-6, § 11 items 1, 2, 5 |
| **Status** | **open** (by construction of the corpus) |
| **User workaround** | None. Operator: confirm Stage-3/Stage-6 fixes in a live run. |
| **Planned fix** | ORCH-774 (live confirmation). Sketch: record UniProt, UniParc, NCBI and PathBank answers per leg (ORCH-768 § 11 item 1 (a)) |

### KI-059 — Per-draw Stage-1 payloads are not kept

| | |
|---|---|
| **Symptom** | A merge question cannot be re-analysed from a leg; draws had to be rebuilt from side files. |
| **Reason codes** | none |
| **Cause** | Only response hashes and the pooled payload are kept; the raw pooled chunk payload under `tmp/stage1_raw…` is overwritten by the next leg; a wholly failed chunk ensemble writes no side file. |
| **Papers and legs hit** | ORCH-755 chlorophyll (D5 counterfactual key-level only), ORCH-759 menaquinone (rebuilt) |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D10, § 8; `ORCH-763-RESULT.md` § 2, § 8 item 1, § 9 Q6 |
| **Status** | **open** |
| **User workaround** | None. |
| **Planned fix** | none planned. Sketch: write the N draw payloads into the side-file directory (about 15-60 KB per leg, no LLM impact) |

### KI-060 — Operator-facing reports misstate the cause

| | |
|---|---|
| **Symptom** | "Chunk 3 failed to produce valid JSON" for valid empty JSON (KI-027); a no-reaction-nodes crash filed as `crash` with Stage-2 counts (KI-020); "No merge was flagged" read as "no merge ran"; `NO DELIVERABLE` while PWMLs sat on disk (F-196). |
| **Reason codes** | driver kind `crash`; message "Chunk `<n>` failed to produce valid JSON." |
| **Cause** | Messages and counters written for one failure shape reused for others; the driver's counts line read the pre-quarantine payload. |
| **Papers and legs hit** | ORCH-755 WNT legs; ORCH-767 WNT PCP; ORCH-755 chlorophyll |
| **Evidence** | `ORCH-758-RESULT.md` § 4 D3, § 6 C3; `ORCH-769c-RESULT.md` § 1 item 4, § 8 F-2; `FINDINGS.md` F-196 |
| **Status** | counts **fixed** `ec800370` (ORCH-769c fix B). The others are **open** |
| **User workaround** | Operator: count `.pwml` files on disk and read the leg's own reports, not the runner's summary line. |
| **Planned fix** | ORCH-772 (the text record's coverage and omission lines become the user's source of truth) |

---

## G. Operations

### KI-061 — Forced shutdown on 2026-10-25: several agents' Python jobs exhausted memory

| | |
|---|---|
| **Symptom** | The machine stopped responding and the product owner forced a shutdown (first recorded as "machine crash #2", 11:32-11:38). |
| **Reason codes** | none (infrastructure) |
| **Cause** | Likely several concurrent agents each running Python jobs that load large caches (`data/enrichment_cache.json` is 39 MB), pushing the machine into swapping. After the reboot, with no sprint work running, about 17.7 of 31.2 GB RAM was already in use by other applications (12 logical CPUs). The System log shows no BugCheck and no resource-exhaustion event. |
| **Papers and legs hit** | no leg; it interrupted the D-113 follow-up work (ORCH-769a/769d, ORCH-770) |
| **Evidence** | `LEDGER.md` rows "2026-10-25 11:53 — machine crash #2" and "Correction to the row above (2026-10-25)" (last two rows, commit `4befcb33`) |
| **Status** | **open** (cause likely, not measured). New operating rules (LEDGER, 2026-10-25): exactly one sprint Python job at a time machine-wide, every job takes the heavy lock; at most 2 subagents alive at once; a job starts only with at least 6 GB RAM free; an orphan sweep after every job; no `python -` stdin scripts |
| **User workaround** | Operator: follow the rules above; check free RAM before a run. |
| **Planned fix** | none planned. Sketch: a pre-job free-memory check inside `bounded_run.py` |

### KI-062 — Unexplained machine crashes and the stale heavy lock they leave

| | |
|---|---|
| **Symptom** | A crash kills the running leg; its G11 report is reserved but never written; the heavy lock stays held by a dead PID. |
| **Reason codes** | none (infrastructure) |
| **Cause** | Not diagnosed. 2026-10-16: interrupted ORCH-753 (gold cohort, no record) and ORCH-752's ledger row. 2026-10-22: during ORCH-756 Ara (wrapper PID 1441772 died with the machine). |
| **Papers and legs hit** | ORCH-756 Ara (report `04`, run dir `2026-10-22_0947` kept, re-run as report `05`) |
| **Evidence** | `ORCH-756-RESULT.md` § 5 item 3, holder sample `C:/t/bt-orch755/stale_holder_ORCH-756_2026-10-22.json`; `LEDGER.md` ORCH-753 row (2026-10-16); `ORCH-754-RESULT.md` § 1 |
| **Status** | **open** |
| **User workaround** | Operator: clear the stale lock by the documented procedure (two identical holder samples, PID proven dead) and re-run the leg with a fresh run directory. |
| **Planned fix** | none planned |

### KI-063 — The overnight summary was 8 hours late (ORCH-767)

| | |
|---|---|
| **Symptom** | The morning summary the product owner asked for appeared at about 13:50 instead of the morning. |
| **Reason codes** | `night.log` `summary_exit=2` |
| **Cause** | The night driver called `orch767_morning_summary.py`, which did not exist yet; the orchestrator, meaning to write it overnight, was blocked about 15 h because a subagent's own Agent dispatch blocks its tool round until the child finishes, even with `run_in_background: true`, and the ORCH-766 implementer ran about 15 h. |
| **Papers and legs hit** | ORCH-767 |
| **Evidence** | `ORCH-767-RESULT.md` "Incident"; `runs_papers/ORCH-767/MORNING_SUMMARY.md` (run scratch, not committed) |
| **Status** | **fixed** by a process rule recorded in `ORCH-767-RESULT.md` "Incident" (under D-112 § 4; the rule itself is not in `DECISIONS.md`): every runner writes its own end-of-run summary. ORCH-770's runner does (`b16a45e7`: `evidence/orch770_night.sh`, `orch770_morning_summary.py`); not yet run |
| **User workaround** | None. |
| **Planned fix** | none |

### KI-064 — Unattended agent sessions stall

| | |
|---|---|
| **Symptom** | Hours of idle time with nothing running: about 11 h in ORCH-756 before phase 2; ORCH-770 did not start on 2026-10-24; agent commands hang until moved to the background. |
| **Reason codes** | none (infrastructure) |
| **Cause** | An unattended permission prompt (most likely, ORCH-756); a usage limit hit by the sub-orchestrator before ORCH-770's 18:45 start; an empty `python -` heredoc waiting on stdin (recorded as a slip in ORCH-754d, -756, -759b, -759c, -759e, -762 and -768). |
| **Papers and legs hit** | ORCH-756 (idle 2026-10-21 ~22:55 → 2026-10-22 09:46); ORCH-770 (not run on 2026-10-24) |
| **Evidence** | `ORCH-756-RESULT.md` § 5 items 1 and 4; `LEDGER.md` row "2026-10-25 11:32 — D-113 recorded"; `ORCH-759b-RESULT.md` § 8 F4; `ORCH-768-RESULT.md` § 9 |
| **Status** | **open**; the `python -` pattern is now forbidden by the 2026-10-25 operating rules |
| **User workaround** | Operator: watch an unattended agent's transcript modification time, not its output file; never pipe into `python -`. |
| **Planned fix** | none planned |

---

## H. Added after ORCH-771

### KI-065 — A written PWML that draws no process (heme, PMC12856317)

| | |
|---|---|
| **Symptom** | The leg writes `pathway.review_required.pwml`, but the file's `<reactions/>`, `<transports/>` and `<interactions/>` are empty. The text record shows PWML coverage 0 of 2 (0 of 3 full count) and lists all three processes as omitted. |
| **Reason codes** | none of its own: the three omission reasons below, and release `review_required` |
| **Cause** | Each of the three processes in the text record is left out for its own catalogued reason. (1) The only reaction, "ALAS2-catalyzed ALA formation" (glycine + succinyl-CoA → aminolevulinic acid), carries the compound cofactor pyridoxal 5'-phosphate in its modifier slot. It is removed before the freeze, `non_protein_actor_in_modifiers` (KI-004; cause KI-005), and its enzyme wrapper `ALAS2 complex` leaves with it (`degree_zero_after_quarantine`). (2) "heme inhibition of ALAS2" names the bare protein `ALAS2`, so it is withheld, `bare_protein_endpoint` (KI-009). The leg's Stage 6 ran on a tree from 2026-08-22, before interaction endpoints were wrapped (ORCH-754b, `c711d859`), and the corpus replays from that frozen post-Stage-6 payload (KI-058). (3) "ALAS2 cofactor binding" (ALAS2 -cofactor-> PLP) is withheld, `untyped_interaction` (KI-008). Nothing is left to draw. The export still writes the file, because D-114 AMENDMENT 2 § 3 rules that a zero-process PWML is not declined. |
| **Papers and legs hit** | `runs_verify/2026-08-22_2017` PMC12856317 on the replay corpus at `main` (`00eb12d3`): the only zero-process PWML among the 56 corpus legs that write one. At its own tree the leg wrote no PWML: the final gate reported "Unknown protein/modifier reference" (KI-004, before ORCH-769b). |
| **Evidence** | `ORCH-772-RESULT.md` § 8 F-2; `evidence/orch772/REVIEW-r0.md` §§ 3.2-3.3; `ORCH-772b-RESULT.md` § 4 (the replay's PWML has 0 `<reaction>`, 0 `<transport>`, 0 `<interaction>`; its payload has `protein_complexes: []` and the endpoint `ALAS2`) |
| **Status** | **product decision** D-114 AMENDMENT 2 § 3: exported as `review_required`, with 0 % coverage shown and every process listed in the text record. **Open:** the KI-005 cofactor rule. A live run on current Stage 6 would wrap the `ALAS2` endpoint (KI-009 fix), which is **not measurable offline**. |
| **User workaround** | Build the pathway from the text record. Re-create the ALAS2 reaction with ALAS2 as a one-protein complex (UniProt id) and PLP as a participant (KI-005), then draw "heme inhibits ALAS2" to that complex (KI-009). The cofactor binding has no PathWhiz type (KI-008). |
| **Planned fix** | none planned. Sketch: KI-005's re-filing of a `role: cofactor` compound as a participant would keep the reaction and its wrapper. |

---

## Appendix — Reason code → KI id

Sorted by code. **Match** says how ORCH-772 matches it: `exact` (the whole string), `prefix` (the part before
the first `<`), or `suffix` (the part after the last `>`). Human messages are in the second table; match their
fixed words, because the names inside vary. Where one code has several causes, the first KI id is the one to
label an omission with; the ids in brackets explain the cause. A code marked "not an omission" is a flag or
status, listed so that it is never mistaken for one. Codes are verified against `src/t2pw` at `4befcb33`
(Grep); `file.function` names the module under `src/t2pw/` and the enclosing function or constant.

| reason code | match | KI id | where it is set |
|---|---|---|---|
| `<bucket>_missing_name` (e.g. `protein_missing_name`) | suffix `_missing_name` | KI-015 | `strict_quarantine._entity_representability`; `ir.validate_required_pwml_contract` |
| `accession_claimed_across_incompatible_entity_kinds` | exact | KI-017 | `identity_admission.RULE_ACCESSION_KIND_CONFLICT`; `map_ids._admit_identities` |
| `actor_schema_not_canonical` | exact | KI-023 | `stage_contracts._validate_canonical_actor_rows`; `payload_models._normalized_actor_issues` |
| `ambiguous_entity_type:<bucket>+<bucket>` (seen: `ambiguous_entity_type:nucleic_acids+proteins`) | prefix `ambiguous_entity_type:` | KI-011 | `strict_quarantine._essential_participant_verdict`, `_actor_verdict` |
| `ambiguous_protein_endpoint` | exact | KI-010 | `ir.interaction_pwml_verdict` |
| `bare_protein_endpoint` | exact | KI-009 | `ir.interaction_pwml_verdict` |
| `budget_exhausted` | exact | KI-028 | `deadline.BUDGET_EXHAUSTED` |
| `budget_indeterminate` | exact | KI-056 (also seen in KI-027) | `extraction_ladder.SKIP_BUDGET_INDETERMINATE` |
| `class_term_never_narrowed_to_specific_molecule` | exact | KI-029 | `compound_resolution._db_match_rule_decision`; `map_ids._refuse_class_term_identity` |
| `closure_not_converged:<n>` | prefix `closure_not_converged:` | KI-016 | `strict_quarantine.quarantine_and_close` |
| `complex_named_source_entity_wrapper` (a `generation_reason`, not an omission) | exact | KI-012 | `process_normalizer.relocate_complex_named_proteins` |
| `component_protein_unresolved` | exact | KI-015 | `ir.validate_required_pwml_contract`, `ir.build_pwml_ir` |
| `confidence_below_floor:<x><<y>` | prefix `confidence_below_floor:` | KI-015 | `strict_quarantine._weak_evidence_verdict` |
| `degree_zero_export:<n>` | prefix `degree_zero_export:` | KI-016 (cause: KI-013) | `strict_quarantine.quarantine_and_close` |
| `diagnostic_only` (release status) | exact | KI-016 (also KI-018, KI-026) | `release_status.DIAGNOSTIC_ONLY` |
| `duplicate_locked_reaction` | exact | KI-014 | `process_normalizer.dedupe_processes`; `stage_one_ensemble.merge_on_mapped_identity` |
| `entity_type_overlap:<n>` | prefix `entity_type_overlap:` | KI-011 (also KI-016) | `strict_quarantine.quarantine_and_close` |
| `explicit_out_of_scope_verdict` | exact | KI-015 | `strict_quarantine` admission loop (`_scope_is_out_of_scope`) |
| `final_gate_report_missing` | exact | KI-016 | `gate_reports.REASON_FINAL_REPORT_MISSING`, returned by `gate_reports.gate_verdict` |
| `final_pre_export_gate_lifecycle` (stage label) | exact | KI-016 | `gate_reports._fail_closed` |
| `final_pre_export_stage3_gates` (stage label) | exact | read the gate error: KI-004, KI-012, KI-013 | `streamlit_app.freeze_canonical_payload` |
| `gate.generated_protein_complex_<...>` (64-character slug that embeds the names) | prefix `gate.generated_protein_complex_` | KI-012 | `driver._slug_code` of the "missing positive stoichiometry" message |
| `gate.protein_<name>_is_missing_a_uniprot_or_drugbank_identifier` (slug embeds the name) | prefix `gate.protein_` and suffix `_is_missing_a_uniprot_or_drugbank_identifier` | KI-001 | `driver._slug_code` of the Stage-3 gate message |
| `gate.protein_has_degree_after_normalization` | exact | KI-013 | `driver._slug_code` |
| `gate.unknown_protein_modifier_reference` | exact | KI-004 (KI-005 for a compound cofactor) | `driver._slug_code` |
| `gate.unknown_transporter_reference` | exact | KI-004 | `driver._slug_code` |
| `generated_complex_component_missing_external_identity` | exact | KI-001 | `ir.validate_required_pwml_contract` |
| `generated_complex_component_missing_species` | exact | KI-003 | `ir.validate_required_pwml_contract` |
| `generated_wrapper_reason_invalid` (informational) | exact | KI-012 | `payload_models._generated_wrapper_issues` |
| `identity_refused_review_required` (`db_status`) | exact | KI-029 | `db_resolver.IDENTITY_REFUSED_STATUS` |
| `implausible_name_match` (mapping issue) | exact | KI-001 (cause on menaquinone: KI-002) | `map_ids._NAME_GATE_ISSUE` |
| `in_process_deadline` | exact | KI-028 | `leg_trace.SOURCE_IN_PROCESS_DEADLINE` |
| `interaction_withheld_from_pwml` (IR report warning) | exact; then label by the row's `reason` | KI-008 | `ir.build_pwml_ir` |
| `malformed_participant_in_<slot>` | prefix `malformed_participant_in_` | KI-015 | `strict_quarantine._essential_participant_verdict` |
| `malformed_placeholder:<claim>` | prefix `malformed_placeholder:` | KI-015 | `strict_quarantine._entity_representability` |
| `minimum_core:requested_core_coverage_below_minimum:<x><<y>` (review reason) | prefix `minimum_core:` | KI-038 | `strict_quarantine` review reasons |
| `missing_<slot>` | prefix `missing_` (quarantine `reason` field only) | KI-015 | `strict_quarantine._essential_participant_verdict` |
| `nested_location_merged` (merge flag) | exact | KI-043 | `stage_one_ensemble.NESTED_LOCATION_FLAG` |
| `no_defensible_reaction_support` | exact | KI-022 | `reaction_support.reaction_support_issue`; `stage_contracts.validate_pre_export` |
| `no_drawable_process` | exact | KI-021 | **not in the code** at `4befcb33` (D-113 § 3 orchestrator reading) |
| `no_reactions` (driver kind) | exact | KI-021 | `driver.KIND_NO_REACTIONS`, `driver._classify` |
| `no_shared_meaningful_token` (name-gate reason) | exact | KI-006 (CPO's accession stripped), KI-029 (DMAPP refused at Stage 6) | `map_ids._name_gate_verdict` |
| `non_protein_actor_in_<slot>` (`_enzymes`, `_modifiers`, `_transporters`) | prefix `non_protein_actor_in_` | KI-004 (KI-005 when the actor is a compound cofactor) | `strict_quarantine._actor_verdict` |
| `operation_timeout` | exact | KI-028 | `pipeline/deadline.py` |
| `outer_parent_kill` | exact | KI-028 | `leg_trace.SOURCE_OUTER_PARENT_KILL` |
| `parenthetical_definition_bridge` (merge flag) | exact | KI-041 | **not on the sprint tip**: unmerged branch `orch765/parenthetical-bridge` |
| `process_row_not_an_object` | exact | KI-015 | `strict_quarantine` admission loop |
| `process_row_vanished_during_closure` | exact | KI-015 | `strict_quarantine` closure revalidation |
| `protein_complex_component_cycle` | exact | KI-015 | `strict_quarantine._entity_representability` |
| `protein_complex_component_omitted:<component>:<reason>` (ORCH-773: a complex left out because a component it resolves to was omitted per row) | prefix `protein_complex_component_omitted:`; then label by `<reason>`, the component's own cause (seen: KI-001) | KI-015 (the sibling `protein_complex_component_unrepresentable` row; no closer entry) | `strict_quarantine._apply_per_row_failures` |
| `protein_complex_component_unnamed` | exact | KI-015 | `strict_quarantine._entity_representability` |
| `protein_complex_component_unrepresentable:<name>:<reason>` | prefix `protein_complex_component_unrepresentable:`; then label by `<reason>` (e.g. KI-001) | KI-015 | `strict_quarantine._entity_representability` |
| `protein_complex_component_unresolved:<name>` | prefix `protein_complex_component_unresolved:` | KI-015 | `strict_quarantine._entity_representability` |
| `protein_complex_missing_components` | exact | KI-015 | `strict_quarantine._entity_representability`; `ir.validate_required_pwml_contract` |
| `protein_complex_missing_species` | exact | KI-003 | `strict_quarantine._entity_representability`; `ir.validate_required_pwml_contract` |
| `protein_endpoint_missing_external_identity` | exact | KI-009 (cause: KI-001; RyhB: KI-051) | `ir.interaction_pwml_verdict` |
| `protein_endpoint_missing_species` | exact | KI-009 (cause: KI-003) | `ir.interaction_pwml_verdict` |
| `protein_endpoint_missing_species_and_external_identity` | exact | KI-009 | `ir.interaction_pwml_verdict` |
| `protein_missing_external_identity` | exact | KI-001 (cause KI-017 or KI-018 where diagnosed) | `strict_quarantine._entity_representability`; `ir.validate_required_pwml_contract` |
| `protein_missing_species` | exact | KI-003 | `strict_quarantine._entity_representability`; `ir.validate_required_pwml_contract` |
| `reaction_coupled_transport_has_no_exportable_ir_representation` | exact | KI-015 | `strict_quarantine.RCT_NOT_EXPORTABLE_REASON` |
| `reaction_enzyme_complex_wrapper_skipped_invalid_component` (mapping action) | exact | KI-006 | `map_ids._rewrite_reaction_protein_enzymes_to_complexes` |
| `reaction_enzyme_must_be_protein_complex` | exact | KI-006 | `ir.build_pwml_ir`, `ir.validate_required_pwml_contract`; `stage_contracts.validate_pre_export` |
| `reaction_enzyme_proteins_dropped_no_external_id` (counter) | exact | KI-007 | `map_ids._rewrite_reaction_protein_enzymes_to_complexes` |
| `refinement_review_crash` (corpus stage label) | exact | KI-020 | `evidence/orch768_replay_corpus.py` (harness, not `src/`) |
| `requested_core_coverage_below_minimum:<x><<y>` | prefix `requested_core_coverage_below_minimum` | KI-038 | `release_status.COVERAGE_REASON_BELOW_MINIMUM` |
| `scope_conflict` (status) | exact | KI-025 (KI-026 when the conflict is the organism) | `driver._STATUS_SCOPE_CONFLICT` = `rag.eligibility.OUTCOME_SCOPE_CONFLICT` |
| `serialization_would_require_invention` (release reason) | exact | KI-016 (also KI-018) | `release_status.classify_release_status` |
| `species_mismatch` (identity verdict) | exact | KI-002 | `map_ids.verify_real_protein_identity` |
| `species_missing_classification` | exact | KI-019 | `ir.validate_required_pwml_contract` |
| `species_missing_taxonomy` | exact | KI-019 | `ir.validate_required_pwml_contract` |
| `species_not_found:<species>` | prefix `species_not_found:` | KI-002 | `map_ids.PathBankDbResolver` (`find_biological_state`, `_map_protein_exact_name_species`) |
| `stage0_scope_conflict_stopped_the_run_before_serialization` | exact | KI-025 (KI-026 for an organism) | `driver.REASON_STAGE0_SCOPE_CONFLICT`; `release_status.SCOPE_GUARD_STOP_REASON` |
| `stage1_empty_chunk` (boundary record) | exact | KI-027 | `pipeline.BOUNDARY_STAGE1_EMPTY_CHUNK`, `pipeline._record_empty_chunk` |
| `typo_snapped` (merge flag) | exact | KI-039 | `stage_one_ensemble.TYPO_FLAG` |
| `unaccounted_locked_reactions:<n>` | prefix `unaccounted_locked_reactions:` | KI-016 | `strict_quarantine.quarantine_and_close` |
| `undeclared_entity_in_<slot>` | prefix `undeclared_entity_in_` | KI-015 | `strict_quarantine._essential_participant_verdict` |
| `undefined_state_treated_as_absent` (merge flag) | exact | KI-039 | `stage_one_ensemble.UNDEFINED_STATE_FLAG` |
| `unexportable_entity:<n>` | prefix `unexportable_entity:` | KI-016 (cause: KI-001, KI-017, KI-018) | `strict_quarantine.quarantine_and_close` |
| `unnamed_actor_in_<slot>` | prefix `unnamed_actor_in_` | KI-015 | `strict_quarantine._actor_verdict` |
| `unresolved_actor_in_<slot>` | prefix `unresolved_actor_in_` | KI-015 | `strict_quarantine._actor_verdict` |
| `untyped_interaction` | exact | KI-008 | `ir.interaction_pwml_verdict` |
| `valid_json_zero_processes` (attempt outcome) | exact | KI-027 | `extraction_diagnostics.OUTCOME_ZERO_PROCESSES` |

**Human messages** (match by their fixed words; the names inside vary):

| message | KI id | where |
|---|---|---|
| "Chunk `<n>` failed to produce valid JSON." | KI-027 (KI-060) | `pipeline.run_stage_one_with_chunking` |
| "Composite entity '`<name>`' in /entities/compounds has no protein-like left component; unsupported." | KI-024 | `process_normalizer.normalize_composites` |
| "Composite token '`<text>`' at `<pointer>` has no protein-like left component; ..." | KI-024 | `process_normalizer._rewrite_token` |
| "DraftGraph has no reaction nodes — nothing to render." | KI-020 | removed by ORCH-769c; at `e3d5b65f` in `draft_graph_render.render_draft_graph_to_png_bytes` |
| "extraction did not finish inside the time budget" (also "audit and DB mapping ...", "PWML export did not finish inside the time budget") | KI-028 | `driver._drive` |
| "AppTest script run timed out after 3600.0(s)" (leg `RESULT.txt` DETAIL; Streamlit's message) | KI-028 | Streamlit AppTest, reached through `driver.run_one` |
| "LLM returned an empty completion for `<stage>` (model `<m>`, finish_reason=length, ...)" | KI-054 (KI-055 at 128k) | `llm/client.chat_detailed` |
| "Generated protein complex '`<c>`' component '`<p>`' is missing positive stoichiometry." | KI-012 | `process_normalizer.run_strict_post_normalization_gates` |
| "Generated protein complex '`<c>`' component protein '`<p>`' is missing a UniProt or DrugBank identifier." | KI-001 | `process_normalizer.run_strict_post_normalization_gates` |
| "No refinement graph image is available." | KI-020 | `streamlit_app.py:2824` |
| "no UniProt or DrugBank identifier; other enzymes remain" | KI-007 | `map_ids._rewrite_reaction_protein_enzymes_to_complexes` |
| "Protein '`<name>`' is missing a UniProt or DrugBank identifier." | KI-001 | `process_normalizer.run_strict_post_normalization_gates` |
| "Protein has degree 0 after normalization: `<name>`" | KI-013 | `process_normalizer.run_strict_post_normalization_gates` |
| "Located protein is isolated in connectivity graph: `<name>`" (ORCH-773 omits the protein; closest entry KI-013, the same unattached-protein family) | KI-013 | `process_normalizer.run_strict_post_normalization_gates` |
| "PWML export stopped by the quarantine boundary: `<reasons>`" | KI-016 | `streamlit_app.run_pwml_export` |
| "PWML export stopped by validation-only pre-export Stage 3 revalidation; Stage 8 did not repair the payload." | KI-023 | `streamlit_app.run_pwml_export` |
| "PWML required-field gate failed." | KI-006 or KI-019 (read the IR codes) | `streamlit_app.run_pwml_export` |
| "Unknown protein/modifier reference: `<name>`" | KI-004 (KI-005) | `process_normalizer.run_strict_post_normalization_gates` |
| "Unknown transporter reference: `<name>`" | KI-004 | `process_normalizer.run_strict_post_normalization_gates` |
| "UniProt `<acc>`, from strain `<strain>`" (borrowed-identity flag) | KI-033 | `app/reaction_evidence` (`borrowed_protein_identities`) |
