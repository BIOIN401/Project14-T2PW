# E2E-2 biology audit (PMID 32185015 / PMC7059786, sour taste, *Mus musculus*, strict)

**Verdict.** 0 reactions is correct for this paper, and `review_required` is the right state. The output is a fragmentary but mostly true channel skeleton. There is one real biological loss (SCN2A), one false anchor (PKD2L1), and an explanation record that is wrong in places. The web page tells the user to read warnings that say nothing about why the run needs review.

## Evidence limits

- **`out/` is gitignored, not committed.** `.gitignore:11` matches it and `git ls-files` returns 0 files there. E2E-2.md implies it is part of the evidence, but only `job_result.json`, `jobstore_state.json`, `tmp_artifacts/**` and the logs are tracked.
- Below, **[C]** means committed and **[L]** means local only, in `out/`.
- The [L] copy looks faithful:
  - `final_mapped.json` is 207766 bytes, which is the bridge size 203002 plus 4764 newlines (CRLF).
  - The PWML sha256 matches the record (`e129b789…`).
- **Paper text.** Quotes come from `source_text_index.normalized`, so they are lowercased and stripped of punctuation. The committed `tmp_artifacts/rag_round_00/checkpoint_graph.json` [C] holds an identical string. I confirmed this by comparison.
- **Canonical check.** No artifact has a literal `quarantine_ok` key. `out/quarantine_report.json` has `"ok": true` and strict stage, so I treated `final_mapped.json` as canonical.
- **No gold case exists.** The committed corpus row has `"gold": null` (`docs/pwml_recovery_sprint/evidence/orch799/corpus_tip.json`). No gold file names this paper in a repo-wide search that excluded `runs_*`, `evidence/` and caches. There is no `relevance_note` or `export_rationale` to cite, so `gold_data_defect` cannot apply and everything is judged against the contract directly.
- E2E-2 [C] `rag_round_00/quarantine/*` shows 17 quarantined rows and OTOP1 "unmatched". That is a **pre-mapping** checkpoint (`protein_missing_species`). Do not read it as the outcome.

## 1. What the run produced

**The 0-reactions claim is correct.** It holds in `job_result.json` [C] `counts`, `out/pwml_qa.json` [L], and at Stage 1, where `stage1_ensemble_report.json` [C] gives `/processes/reactions rows [0, 0, 0]`. All 10 earlier legs on this paper also had 0 reactions (`runs_papers/*/REVIEW/index.jsonl`, untracked).

| stage | what happened | source |
|---|---|---|
| Stage 1 (3 draws) | 11 proteins, 6 compounds, 6 transports, 6 interactions | `audit_iteration_summary.json` [L] |
| Stage 2 merge | 6 transports, 16 interactions (10 added, of which 7 are `SAME_AS` rows) | `merged_payload.json` [L], `reaction_summary.txt` [C] |
| Normalization + audit | 5 interactions left. `SAME_AS` rows gone; `PKD2L1 expressed_in Type III taste cell` gone | quarantine admissions list exactly 5 [L] |
| Post-mapping | SCN2A moved to `quarantined_proteins` | `final_mapped.json` [L] |
| Strict quarantine | 10 admitted, 1 out: `SNARE proteins facilitate 5-HT release`, `protein_missing_external_identity` | `quarantine_report.json` [L] |
| Closure | PKD2L1, `voltage-gated Ca++ channels`, SNARE dropped as degree zero | `removed_entity_report.json` [L] |
| Actor quarantine | both `voltage-gated Ca++ channels` transporter actors stripped from Ca++ influx (`declared_role_is_not_a_transporter_claim`) | `quarantined_process_claims` [L] |
| PWML IR | `Ca++ triggers 5-HT release` withheld: `untyped_interaction` | `pwml_ir_report.json` [L] |

**Final content.** 6 transports and 3 PWML interactions (4 in `final_mapped.json`).

- The 6 transports are only **3 distinct events**: H+ via OTOP1 ×3, 5-HT release ×2, Ca++ influx ×1.
- The "6 proteins" are OTOP1 (Q80VM9), Kir2.1 (P35561), P2rx1/2/3, and a placeholder `Unknown` whose record species is *Arabidopsis thaliana*.
- All process lineages are `paper_extraction`, except the Ca++ triggers row, which has no lineage. No `rag_provenance` appears anywhere, and `rag_admission_report.json` [L] has `considered: 0`.

**Correct restraint worth noting.**

- A best-effort Xenopus accession for the Ca++ channel (`A0A6I8RK79`, `gap_iteration_summary.json` [L]) did not reach the final payload.
- ATP is not drawn as released from Type III cells. The paper says "the source of atp for transmission of sour information from type iii cells remains enigmatic".

## 2. Is 0 reactions right?

**Yes. No enzymatic reactions were lost.** The sour section is entirely channels and signalling:

> "transduction of sour involves permeation of h through an apical ion channel … otop1 … the entering h ions not only directly depolarize the taste cells but also block kir2 1 k channels … the resulting depolarization triggers voltage gated na channels scn2a 15 to generate action potentials that activate voltage gated ca channels triggering the release of synaptic vesicles"

A search of the full text for enzyme terms found only the Type II PLC cascade (excluded scope) and the `car4` marker. Coverage of the paper's actual chain:

| step in paper | in PWML? |
|---|---|
| H+ through OTOP1 | yes, drawn 3 times |
| H+ blocks Kir2.1 | yes |
| depolarization → SCN2A → action potentials | **no** |
| VGCC → Ca++ influx | transport present, **no channel attached** |
| Ca++ → vesicle release | in canonical JSON, **withheld from PWML** |
| SNARE release machinery | quarantined (no identity) |
| 5-HT → 5-HT3 receptor | yes, placeholder-backed |
| ATP → P2X | yes, family complex |

The real shortfall is missing or unattached channel steps, not reactions.

## 3. PKD2L1 and SCN2A

**PKD2L1 is in the text and was extracted. Its absence is correct.**

- Stage 1 extracted it in 3/3 draws [C], with the row `expressed_in Type III taste cell`.
- Audit removed that row. The rationale in `audit_iteration_summary.json` [L] is "Removed interaction referencing cell type as entity". It is the only row with that target. The patch itself is not preserved, because that report elides further keys.
- Quarantine closure then dropped the protein as degree zero.
- The paper says it is a marker, not a participant: "in 2011 horio et al showed that pkd2l1 itself was not necessary for transduction of protons" and "using pkd2l1 as a molecular identifier for sour responsive taste cells".
- Contract §2 forbids such neighbours as participants ("no assay reporters as pathway members").
- **The defect is the anchor itself.** Stage 0 put PKD2L1 in `key_proteins` (`coverage_summary.json` [L]), so a correct omission is reported as a failed check.

**SCN2A is in the text and is a real step. It was lost at Stage 1.**

- It was extracted as an entity with a location in 3/3 draws [C], with evidence "voltage-gated Na+ channels (SCN2A)".
- **No process in any draw references it.** It is absent from every `/processes/*` bucket in the ensemble report [C], and `qa_report.json` [C] flags it `orphan_nodes … degree 0`.
- It was then dropped as `not_referenced_by_any_surviving_process`, with verified accession B1AWN6 preserved in `quarantined_proteins` [L]. That drop is correct behaviour given the input.
- Stage 1 did write the equivalent row for calcium, so an equivalent sodium row is a representable step. The paper does not literally state Na+ entry, so such a row would rest on the channel's name.

## 4. Is `review_required` right, and would the page mislead?

**The state is right.** Contract §4 and §7 fit: a valid PWML with named uncertainties and `strict_acceptance_eligible=false`. Removing the false PKD2L1 anchor would not change it, because SCN2A and the omission cap still apply.

One caveat is for the product owner. §7 asks for a fragment that is "internally connected". By my reading of the process rows, the canonical graph has 3 components and the PWML has 4:

- {H+, OTOP1, Kir2.1}
- {Ca++}
- {5-HT, 5-HT3R}
- {ATP, P2X}

Ca++ joins 5-HT only through the withheld row. The paper's actual links (depolarization, action potentials) are not entities. `quarantined_disconnected: 0` does not settle this, because at `strict_quarantine.py:2602-2636` that state covers rows lost during closure, not graph connectivity.

**The page does not claim confirmation, so §4's prohibition is met. It does mislead by omission** (`src/t2pw/web/views.py:199,218-221`, `templates/result.html`):

- It says "Read the warnings below". For this run the only warnings are two `entity_missing_mapping_meta @ /entities/subcellular_locations/6/...` lines, which are unrelated to why review is needed.
- `missing_anchors`, `semantic_failed_checks`, `completeness`, `expansion_blocked_reason` and the omitted process are never rendered. A grep of `src/t2pw/web` finds no use of them outside the curator queue and persistence.
- The user has to open `coverage_summary.json` under a disclosure to learn anything biological. `run_outcome.json` is not offered at all.
- The Attempts table shows "Passed" and the page gives no counts, so "no reactions, and these channels are absent" is never stated.
- Surfacing these fields is web-layer work with no biology change, for the orchestrator to scope.

## 5. Classification

Each finding has: **evidence** (above unless noted), **stage**, **cost**, and confidence. There are no gold citations, for the reason given above. No code is proposed.

| # | Finding | Class | Confidence |
|---|---|---|---|
| F1 | 0 reactions | **expected behaviour** | high |
| F2 | PKD2L1 reported as missing anchor | **gate defect** (Stage-0 anchor input); omission itself correct | high |
| F3 | SCN2A step missing | **undetermined** between depth shortfall and contract violation | medium |
| F4 | `expansion_blocked_reason` and `retrieval_attempts` misreport | **product_contract_violation** (§7 record) | high |
| F5 | completeness 0.8 overstated | **gate defect** (coverage measure) | high |
| F6 | P2X complex includes P2rx1 | **policy_disagreement** (D-117 LOCKED vs §2) | high |
| F7 | 5-HT3 receptor is `Unknown`/Arabidopsis-backed | **standing policy_disagreement** (§13); no fix proposed | n/a |
| F8 | Ca++→5-HT in canonical JSON, not PWML | **product_contract_violation** as §5 is written | medium |
| F9 | Duplicate OTOP1 transports; invented or wrong compartments | **product_contract_violation** (§2 locations), low severity | medium |
| F10 | Connectivity vs §7 wording | **policy question for the PO** | medium |
| F11 | Page does not show why review is needed | **web display gap**, not biology | high |

**F3, what would settle it.** Static reading shows gap detection cannot target an anchor: `rag/retrieve.py:656` `detect_gaps` emits five gap types, all reaction, metabolite, enzyme or compartment. `requested_core`, `missing_anchor` and `unmatched_terms` do not appear in `retrieve.py`, and the last two do not appear in `rag_orchestration.py` or `extraction.py`. Contract §2 steps 3-4 ask for retrieval on missing anchors. What I cannot tell is whether RAG ran here:

- `RAG_ENABLED` was not recorded and `.env` was deleted.
- The only checkpoint [C] is pre-round (`rounds_completed: 0`).
- `considered: 0` fits RAG off, no gaps, or nothing retrieved.
- A persisted RAG loop record with stop reason, or a rerun with it captured, would settle this. Stage-1 output also varies between runs.

**F4 detail.** The string says PKD2L1 and SCN2A are blocked by "candidate processes withheld by strict admission (quarantined_unmapped_entity:1); admitting them would require unsupported biology". The one withheld process is SNARE, unrelated to either anchor and stated in the paper ("snare mechanisms to effect release of synaptic vesicles"). It was withheld for missing identity. The string is hardcoded at `pipeline/strict_quarantine.py:3942`. `retrieval_attempts: 0` is by its own comment "NOT a retrieval-round counter" (`:4162`). §7 requires retrieval attempts and why expansion failed, and neither is truthfully recorded. Cost: a change under `pipeline/`, out of this track, for the PO.

**F5 detail.** `voltage-gated calcium channels` counts as matched only because the process *name* contains it: core term `ca++influxviavoltagegatedcalciumchannels`, substring rule at `strict_quarantine.py:964`. No such entity or actor is in the PWML. Excluding it gives 0.7, and also excluding the false PKD2L1 anchor gives 7/9. Neither crosses the 0.5 threshold, so the status is unchanged. Only the reported score is wrong. `subprocess_coverage_ratio` 0.11 is a lexical undercount in the other direction, for example "Kir2.1 channel block" is uncovered despite the admitted H+ inhibition row. It is diagnostic only.

**F6 detail.** The rule is "first 3 of 7 reviewed same-taxon hits, not the paper", and the source is recorded honestly. The body text names only "P2X receptors". P2X2/P2X3 appear only in a reference title. The PWML still shows a 1:1:1 P2rx1/2/3 complex. P2rx1 has no support in this paper, against §2 "correct complexes and components". Cost of a PO change: an unknown number of other family complexes.

**F8 detail.** Canonical has 4 interactions and the PWML has 3, after the freeze. It is disclosed as `omitted_from_pwml:1_processes`. The corpus has a `by_design_withheld` instrument, but I found no ruling covering `untyped_interaction`, which is why confidence is medium. A ruling would reclassify it.

**F9 detail.** One paper sentence became three OTOP1 transports with different endpoints (`taste bud lumen`, `extracellular space`, `saliva`), because draws were unioned without merging: `/processes/transports rows [2, 3, 1]`, merged 6. "lumen" occurs 0 times in the paper. One 5-HT row transports 5-HT into `afferent nerve fiber`. The paper says only that 5-HT acts "on neural 5-ht 3 receptors", and the other 5-HT row ends in extracellular space. All carry `paper_explicit: explicit`, so lineage overstates the location support.

## What I could not determine

- Whether RAG ran, or what it queried.
- What the curator did. The E2E-2 record already notes its report did not survive.
- The exact audit patch that removed the PKD2L1 row.
- Whether the Stage-1 miss on SCN2A recurs systematically. I did not trace the prior legs' content beyond counts.
- Biology beyond this review article's own claims. I did not consult primary papers.

## Files

- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/job_result.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/out/final_mapped.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/out/coverage_summary.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/out/quarantine_report.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/out/removed_entity_report.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/out/pwml_ir_report.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/tmp_artifacts/stage1_raw_extraction.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/tmp_artifacts/extraction_diagnostics/run-3b9c541c09aa/stage1_ensemble_report.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/webserver/evidence/E2E-2/tmp_artifacts/rag_round_00/checkpoint_graph.json`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src/t2pw/pipeline/strict_quarantine.py`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src/t2pw/rag/retrieve.py`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src/t2pw/web/views.py`
- `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src/t2pw/web/templates/result.html`
