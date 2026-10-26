# ORCH-767 — overnight run, morning summary

Written by the runner at the end of the run (`evidence/orch770_morning_summary.py`). Legs: `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs`.

## Per paper

PWML coverage = processes drawn in the PWML / processes in the text record (D-114; target 75%), from each leg's directory by `t2pw.app.reaction_evidence`. Headline first: the interactions PathWhiz cannot carry (KI-008) are left out (D-114 AMENDMENT 2); the full count follows. A leg with no PWML names the known issue of its recorded cause. "would import" leaves out the reactions PathWhiz drops on import: those the written PWML exports with no enzyme and `spontaneous=false` (column "dropped on import", counted in the XML).

| paper | PWML | PWML coverage | dropped on import | release | reactions / interactions | min | code |
|---|---|---|---|---|---|---|---|
| enterobactin | **no** | 0/21 (0%); full count 0/21 (0%); no PWML: KI-012 | - | review_required | 5 / 8 | 64.0 | `d6849c9e` |
| chlorophyll | **no** | 0/61 (0%); full count 0/61 (0%); no PWML: KI-004 | - | review_required | 27 / 5 | 74.2 | `d6849c9e` |
| sour_taste_v2 | **yes** | 7/12 (58%); full count 7/12 (58%); would import 7/12 (58%) | 0 | review_required | 0 / 0 | 49.0 | `d6849c9e` |
| wnt_pcp | **no** | 0/66 (0%); full count 0/66 (0%); no PWML: unclassified | - | review_required | 0 / 18 | 103.0 | `d6849c9e` |
| wnt_ca | **no** | 0/48 (0%); full count 0/48 (0%); no PWML: KI-016 | - | diagnostic_only | 4 / 33 | 109.2 | `d6849c9e` |
| menaquinone | **yes** | 14/14 (100%); full count 14/14 (100%); would import 14/14 (100%) | 0 | review_required | 11 / 1 | 55.5 | `d6849c9e` |

## Omitted from the PWML, per paper (kind, name, known issue)

### enterobactin: 21 processes, 0 actors left out

By KI: KI-012 17, KI-001 4

- interaction Fur repression of ent gene clusters: KI-001
- interaction Fur repression of ent gene clusters: KI-001
- interaction Fur binding to Fe2+: KI-001
- interaction Fur binding to ent gene cluster promoters: KI-001
- reaction EntC (isochorismate synthase): KI-012
- reaction EntB isochorismatase: KI-012
- reaction EntA (2,3-dihydro-2,3-dihydroxybenzoate dehydrogenase): KI-012
- reaction EntE (2,3-dihydroxybenzoyl-AMP ligase): KI-012
- reaction EntF enterobactin synthase reaction: KI-012
- transport Enterobactin secretion: KI-012
- transport Ferric enterobactin import: KI-012
- transport Enterobactin secretion: KI-012
- transport Ferric enterobactin import: KI-012
- interaction 2,3-DHB competitive inhibition of EntB: KI-012
- interaction 2,3-DHB binding to EntE: KI-012
- interaction EntC-EntB channeling: KI-012
- interaction EntB-EntA interaction: KI-012
- interaction EntB-EntE interaction: KI-012
- interaction EntB-EntF interaction: KI-012
- interaction EntD modification of EntB: KI-012
- interaction EntD modification of EntF: KI-012

### chlorophyll: 61 processes, 0 actors left out

By KI: KI-004 32, KI-011 29

- interaction HY5 activates CHLH: KI-011
- interaction PIF3 inhibits CHLH: KI-011
- interaction PIF3 inhibits GUN4: KI-011
- interaction PIF5 inhibits CHLH: KI-011
- interaction PIF3 binds to CHLH promoter: KI-011
- interaction GLK1 activates CHLH: KI-011
- interaction TOC1 inhibits CHLH: KI-011
- interaction FHY3 activates ALAD1: KI-011
- interaction RVE1 activates PORA: KI-011
- interaction SCL27 inhibits PORC: KI-011
- interaction FLU binds to GluTR1: KI-011
- interaction HY5 activates PORC: KI-011
- interaction COP1 inhibits HY5: KI-011
- interaction EIN3 activates PORA: KI-011
- interaction GLK1 activates GUN4: KI-011
- interaction GLK1 activates CAO: KI-011
- interaction FHY3 activates GUN4: KI-011
- interaction FHY3 activates CHLH: KI-011
- interaction ABI4 activates PORA: KI-011
- interaction GUN1 activates ABI4: KI-011
- interaction GUN1 downregulates GLK1 expression: KI-011
- interaction Flagellin 22 induces FC1 expression: KI-011
- interaction Flagellin 22 induces HEMA2 expression: KI-011
- interaction PIF4 inhibits CHLH: KI-011
- interaction PIF4 inhibits PORC: KI-011
- interaction PIF4 inhibits CAO: KI-011
- interaction EIL1 activates PORA: KI-011
- interaction FLU inhibits GluTR1: KI-011
- interaction CHLD is a subunit of Mg-chelatase: KI-011
- reaction GluTR reaction: KI-004
- reaction GSAT reaction: KI-004
- reaction ALA dehydratase reaction: KI-004
- reaction PBG deaminase reaction: KI-004
- reaction Urogen III synthase reaction: KI-004
- reaction Urogen III decarboxylase reaction: KI-004
- reaction Coprogen III oxidase reaction: KI-004
- reaction Protogen IX oxidase reaction: KI-004
- reaction Mg-chelatase reaction: KI-004
- reaction Mg-Proto IX methyltransferase reaction: KI-004
- reaction Mg-Proto ME cyclase reaction: KI-004
- reaction Pchlide oxidoreductase reaction: KI-004
- reaction DV-Chlide reductase reaction: KI-004
- reaction Geranylgeranyl reductase reaction: KI-004
- reaction Chl synthase reaction: KI-004
- reaction Chlide a oxygenase reaction (Chlide a to Chlide b): KI-004
- reaction Chlide a oxygenase reaction (Chl a to Chl b): KI-004
- reaction GluTR (HEMA1) reaction: KI-004
- reaction ALAD (HEMB1) reaction: KI-004
- reaction PORB reaction: KI-004
- reaction Chl synthase reaction (phytol): KI-004
- reaction Glu-tRNAGlu reduction to GSA: KI-004
- reaction Coprogen III to Protogen IX: KI-004
- reaction Protogen IX to Proto IX: KI-004
- reaction Mg-chelatase reaction: KI-004
- reaction singlet oxygen generation from Pchlide: KI-004
- reaction ferrochelatase reaction: KI-004
- interaction Mg-Proto IX represses LHCB1.1: KI-004
- interaction Mg-Proto IX represses RBCS: KI-004
- interaction Mg-protoporphyrin IX inhibits LHCB1.1: KI-004
- interaction Mg-protoporphyrin IX inhibits RBCS: KI-004
- interaction CHLI1 is a subunit of Mg-chelatase: KI-004

### sour_taste_v2: 5 processes, 0 actors left out

By KI: KI-001 5

- transport K+ efflux via Kir2.1: KI-001
- interaction H+ blocks Kir2.1 K+ channels: KI-001
- interaction 5-HT activates 5-HT3 receptor: KI-001
- interaction 5-HT activates 5-HT3 receptor: KI-001
- interaction ATP activates P2X receptor: KI-001

### wnt_pcp: 66 processes, 0 actors left out

By KI: KI-015 39, unclassified 18, KI-001 9

- reaction JNK phosphorylates c-Jun: KI-015
- reaction Prickle ubiquitination by Smurf: KI-001
- reaction CapZIP phosphorylation by JNK: KI-001
- reaction MRLC activation by ROCK: KI-015
- reaction Phosphorylation of Dvl by Frizzled: KI-001
- reaction Phosphorylation of MRLC by ROCK: KI-015
- reaction Dvl-DAAM complex activates RhoA: KI-001
- reaction DAAM activates Profilin: KI-015
- reaction Rac1 activates JNK: KI-015
- reaction RhoA activates ROCK: KI-015
- reaction RhoA activates DIA1: KI-015
- transport c-Jun transport to nucleus: KI-015
- transport c-Jun translocation to nucleus: KI-015
- interaction Par6 interacts with Dvl: KI-015
- interaction Smurf binds Par6: KI-015
- interaction Wnt5a binds Frizzled: KI-015
- interaction Dvl binds Par6: KI-015
- interaction Smurf binds Par6: KI-015
- interaction Par6 interacts with Dvl: KI-015
- interaction Wnt5a activates TAZ: KI-001
- interaction Smurf ubiquitinates Prickle: KI-015
- interaction Dvl binds Par6: KI-015
- interaction Wnt binds Frizzled: KI-001
- interaction Wnt binds Vangl2: KI-001
- interaction Wnt binds Celsr1: KI-001
- interaction Frizzled binds PTK7: KI-015
- interaction Frizzled binds MUSK: KI-015
- interaction Frizzled binds ROR1: KI-015
- interaction Frizzled binds ROR2: KI-015
- interaction Frizzled binds RYK: KI-015
- interaction Frizzled binds Syndecan: KI-015
- interaction Frizzled binds Glypican: KI-015
- interaction Par6 is part of Dvl-Par6-Smurf complex: KI-015
- interaction Smurf is part of Dvl-Par6-Smurf complex: KI-015
- interaction Frizzled (Fz) interacts PTK7: KI-015
- interaction Frizzled (Fz) interacts MUSK: KI-015
- interaction Frizzled (Fz) interacts ROR1: KI-015
- interaction Frizzled (Fz) interacts ROR2: KI-015
- interaction Frizzled (Fz) interacts RYK: KI-015
- interaction Frizzled (Fz) interacts Syndecan: KI-015
- interaction Frizzled (Fz) interacts Glypican: KI-015
- interaction PTK7 binds Frizzled (Fz): KI-015
- interaction MUSK binds Frizzled (Fz): KI-015
- interaction ROR1 binds Frizzled (Fz): KI-015
- interaction ROR2 binds Frizzled (Fz): KI-015
- interaction RYK binds Frizzled (Fz): KI-015
- interaction Syndecan binds Frizzled (Fz): KI-001
- interaction Glypican binds Frizzled (Fz): KI-015
- interaction Dvl recruits Inversin: unclassified
- interaction Dvl associates with DAAM: unclassified
- interaction Dvl binds Inversin: unclassified
- interaction Dvl binds DAAM: unclassified
- interaction Wnt5a binds Celsr1: unclassified
- interaction Wnt5a binds Vangl2: unclassified
- interaction Wnt5a activates YAP: unclassified
- interaction Dvl recruits Invs: unclassified
- interaction Vangl2 dissociates Dvl-Prickle-Intu complex: unclassified
- interaction Intu binds Dvl: unclassified
- interaction Dvl is part of Dvl-Prickle-Intu complex: unclassified
- interaction Prickle is part of Dvl-Prickle-Intu complex: unclassified
- interaction Intu is part of Dvl-Prickle-Intu complex: unclassified
- interaction Dvl is part of Dvl-Par6-Smurf complex: unclassified
- interaction Dvl is part of Dvl-Inversin complex: unclassified
- interaction Inversin is part of Dvl-Inversin complex: unclassified
- interaction Dvl is part of Dvl-DAAM complex: unclassified
- interaction DAAM is part of Dvl-DAAM complex: unclassified

### wnt_ca: 48 processes, 0 actors left out

By KI: KI-016 48

- reaction PLC cleavage of PIP2: KI-016
- reaction NLK phosphorylates TCF4: KI-016
- reaction calcineurin dephosphorylates NF-ATp: KI-016
- reaction CAMKII autophosphorylation: KI-016
- transport Ca2+ release from ER: KI-016
- transport SOCE via Orai: KI-016
- transport SERCA Ca2+ pump: KI-016
- transport IP3-induced Ca2+ release from ER: KI-016
- transport SERCA Ca2+ pump: KI-016
- transport IP3-induced Ca2+ release from ER: KI-016
- transport Store-operated Ca2+ entry: KI-016
- transport SERCA Ca2+ pump: KI-016
- transport IP3-induced Ca2+ release from ER: KI-016
- transport IP3-induced Ca2+ release: KI-016
- transport SERCA pumps Ca2+ into ER: KI-016
- interaction Wnt5a binds Fzd2: KI-016
- interaction DAG activates PKC: KI-016
- interaction Ca2+ activates PKC: KI-016
- interaction PKC activates Cdc42: KI-016
- interaction IP3 binds InsP3R: KI-016
- interaction Ca2+ activates calcineurin: KI-016
- interaction Ca2+ activates CAMKII: KI-016
- interaction CAMKII activates TAK1: KI-016
- interaction NLK inhibits β-catenin/TCF complex: KI-016
- interaction STIM1/2 activates Orai: KI-016
- interaction Frizzled2 activates G protein: KI-016
- interaction G protein activates PLC: KI-016
- interaction Calcineurin activates NFAT: KI-016
- interaction TAK1 activates NLK: KI-016
- interaction Ca2+ binds calmodulin: KI-016
- interaction CAM activates CAMKII: KI-016
- interaction DAG activates PKC: KI-016
- interaction calmodulin activates CAMKII: KI-016
- interaction CAMKII activates TAK1: KI-016
- interaction Ca2+ activates CAMKII: KI-016
- interaction Ca2+ activates calcineurin: KI-016
- interaction TAK1 activates NLK: KI-016
- interaction Wnt5a-Frizzled2 binding: KI-016
- interaction Frizzled2 activates phospholipase C: KI-016
- interaction Ca2+ activates PKC: KI-016
- interaction PKC activates Cdc42: KI-016
- interaction Ca2+ activates CaMKII: KI-016
- interaction IP3-InsP3R binding: KI-016
- interaction STIM1/2-Orai activation: KI-016
- interaction β-catenin component of β-catenin/TCF complex: KI-016
- interaction TCF component of β-catenin/TCF complex: KI-016
- interaction calcineurin A component of calcineurin complex: KI-016
- interaction calcineurin B component of calcineurin complex: KI-016

### menaquinone: 0 processes, 0 actors left out

By KI: none



## Gold accuracy (eval-v2 score_leg)

| paper / level | ORCH-767 |
|---|---|
| enterobactin stage1 | P 1.000 R 1.000 F1 1.000 (5/5) |
| enterobactin final | P 1.000 R 1.000 F1 1.000 (5/5) |
| enterobactin pwml | not written |
| menaquinone stage1 | P 1.000 R 1.000 F1 1.000 (9/9) |
| menaquinone final | P 0.909 R 1.000 F1 0.952 (9/9) |
| menaquinone pwml | P 0.909 R 1.000 F1 0.952 (9/9) |

## Failures, one line each (from each leg's RESULT.txt)

- **enterobactin:** post-pipeline validation failed: 1 blocking issue(s) at final_pre_export_stage3_gates [gate.generated_protein_complex_enterobactin_synthase_complex_componen]
- **chlorophyll:** post-pipeline validation failed: 9 blocking issue(s) at final_pre_export_stage3_gates [gate.unknown_protein_modifier_reference]
- **wnt_pcp:** the app never rendered the "refinement_generate_pwml" button ('refinement_generate_pwml'), so the review step that unlocks PWML export was not reached [(none reported)]
- **wnt_ca:** post-pipeline validation failed: 1 blocking issue(s) at final_pre_export_gate_lifecycle [final_gate_report_missing] [gate.gate_lifecycle]

## Paths

Review copies: `C:\t\orch774s\ms767\REVIEW/<paper>/strict/` (reaction_evidence.txt, RESULT.txt, stage1_ensemble_report.txt, the PWML where written); index `REVIEW/INDEX.txt`.

## LLM health

```
rag_prose_extraction: 0/104 empty | Stage 1 extraction: 1/51 empty | gap resolver: 0/36 empty | Stage 2 inference: 6/34 empty | preprocessor: 0/23 empty | audit loop: 1/16 empty | chat: 0/9 empty
```
```
    LONG/ERROR  Stage 2 inference attempt 1: finish_reason=length chars=0 ~1216 s
    LONG/ERROR  audit loop attempt 1: finish_reason=length chars=0 ~1410 s
    LONG/ERROR  audit loop attempt 2: finish_reason=error chars=0 ~1277 s
TOTAL model attempts 273, long (>= 1200 s) 3, finish_reason=error 1
```
