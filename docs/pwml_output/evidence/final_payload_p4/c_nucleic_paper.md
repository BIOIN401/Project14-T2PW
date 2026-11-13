# Task C: nucleic-acid DNA/RNA type read from the paper (D-W20c)

Script: `c_nucleic_measure.py` (raw numbers `c_nucleic_measure.json`; run `PYTHONPATH=C:/t/fp/src python c_nucleic_measure.py OUT.json`).
Input: the 152 replayed `final_mapped.json` payloads of `C:/t/ev-cm/runlist.json` (filter `q`, same index as `replay.py`), each with
its paper text at `<run dir>/../01_source_text.txt`. The id fill does not touch nucleic-acid rows, so the raw payloads are the right input.
Tests: see the fix-round reports `tests_final/m_focus.*` (191 passed) and `m_golden.*` (38 passed), surviving processes 0.

**Fix round (after the opus review rejected HEAD cdcac27a):** a name no longer matches inside a longer name when joined by `.` `-` `/` plus a letter/digit (`LHCB1` in `LHCB1.1`, `RBCS` in `RBCS-1A`), and a type is resolved only when the 2-token window rule AND the whole-sentence rule agree on it (the intersection). The numbers below are the re-measure with both changes (`c_nucleic_measure.json` regenerated; the previous default-rule numbers are in the `window_only_*` totals).

## Counts (152 payloads)
| | rows |
|---|---:|
| runs with any nucleic-acid row | 6 (all 6 have the paper text on disk) |
| nucleic-acid rows | 39 |
| has a database id (never typed) | 0 |
| settled by name already (`nucleotide_type_for`) | 4 |
| **lack a settled type** | **35** |
| resolved from the paper (default rule = window AND whole sentence agree) | **10** (DNA 10, RNA 0) |
| unresolved, mentioned but no cue next to a mention | 11 |
| unresolved, never mentioned (name or synonym not in the text) | 7 |
| unresolved, conflicting (window hit contradicted by the sentence) | 6 |
| unresolved, protein-like name (`CRP-binding site`, "binding") | 1 |

Before the fix round the window rule alone resolved 16 (DNA 13, RNA 3), 12 no cue, 6 never mentioned, 0 conflicting. Of the 10 now, 6 have the cue inside the row name itself (`P BAD promoter`, `PORC promoter`, ...): the paper confirms that the name occurs, but the
type is name-borne. 4 rest on a cue outside the name (`P BAD`, `araBAD operon`, `RBCS`, `PhANGs`).

What the two changes did to the 16: 6 rows went from resolved to conflicting (`araO2`, `araI2`, `CHLH promoter`, `LHCB1.1`, `RyhB` in two runs), no resolved row changed type. The name-boundary change moved one row (`I1-I2 regulatory elements`) from no cue to never mentioned; `RBCS` is still resolved by its own `RBCS promoter` mentions.

Sensitivity (same rows): the window rule alone gives 16 resolved; whole-sentence cues alone (the PO wording "any sentence with both kinds = unresolved") give 18 resolved, 8 conflicting, 1 no cue. The default is now the intersection of the two = 10.

## The 33 harness messages (replay idx 7, chlorophyll PMC5130987)
The messages come from 12 nucleic-acid entities of one file (12 "type can't be blank" + 12 "not included in the list" + 7 + 7 location/visualization
messages). Entities: **4 of 12 resolve now** (PORC promoter, PORA promoter, RBCS, PhANGs: all DNA), 8 stay unset
(`CHLH promoter`, `LHCB1.1`: the window cue is contradicted by the sentence, conflicting; `ALAD1/HEMB1`, `miR171c`, `CHL27`: mentioned, no adjacent cue; `HEMA1 promoter`, `ALAD1/HEMB1 promoter`, `PORB promoter`: never mentioned verbatim in the text).
Expected effect: about 8 of the 24 type messages go away (was 12 before the fix round). NOT re-imported into the harness (not run), so the real message count after is not established.

## Rules and why
1. **Mention** = row name or any `synonyms` entry, case-insensitive, whole token (no letter/digit on either side), spaces/hyphens interchangeable. Names under 2 characters are ignored.
1b. **Name boundary**: a mention may not be joined to a further letter/digit by `.` `-` `/` on either side (`LHCB1` is not a mention of `LHCB1.1`, `RBCS` not of `RBCS-1A`).
2. **Cue window** = the 2 word tokens before and after the mention, same sentence. **A type is taken only if the whole-sentence rule gives the same type** (intersection). Reason: a head noun next to the name ("X mRNA", "X promoter region", "plasmid X") is a claim about X; a cue elsewhere in the sentence usually belongs to a neighbour
   ("the CHLH promoter and miR171c transcript"). Whole-sentence is available as `window=None` and is the strict PO wording; it resolves more rows but 8 conflicting, mostly from neighbours' cues, and it hands `miR171c` the cue of "the target SCL transcripts" (right answer by luck).
3. **Cues**: RNA = rna mrna mirna sirna shrna lncrna rrna trna srna ncrna snrna snorna transcript(s). DNA = dna cdna gdna gene(s) promoter(s) locus loci plasmid(s) genomic allele(s) knockout enhancer(s). "transcription of" / "expression of" are deliberately not cues: said of a gene as often as of its RNA.
4. **Conflict**: a mention whose window holds both kinds, or cue-bearing mentions that disagree => unresolved. **No majority vote**: a symbol used as gene in one place and as transcript in another is the ambiguity this must not paper over. A mention with no cue is neutral; one cue-bearing mention is enough if the others are neutral.
5. **Absent evidence** (no text, not mentioned, no cue) => unresolved, counted in the fill report (`no_paper_text`, `not_mentioned`, `no_cue`) and the writer still counts `nucleic_acids_type_unset`.
6. **Guards kept**: rows with `pathwhiz_id`/`pathbank_nucleic_acid_id`/`pw_nucleic_acid_id` are skipped; names with a protein-like token are skipped outright (I did not add the "explicit about the entity itself" exception: 1 such row exists in 152 payloads and it is not a case of explicit evidence).

## Evidence sample (resolved; quote is the stored `final_payload_fill` quote, cue is the stored cue)
| run idx | entity | type | cue | quote |
|---|---|---|---|---|
| 0 | araI2 | DNA | promoter | Similarly, araI2 promoter variants exhibited increased basal activity. |
| 0 | araO2 | DNA | promoter | This included all four single nucleotides in araO2 , araO2 combined promoter (P39-48), an araO2 deletion (Δ araO2 ), and three in araI2 ... |
| 0 | araBAD operon | DNA | promoter | An operator at-280 base pairs that is required for repression of araBAD operon promoter ... |
| 0 | P C promoter | DNA | promoter | As the AraC concentration increases, expression is downregulated by binding to araO1 located in the P C promoter. |
| 7 | PORA promoter | DNA | promoter | In addition, RVE1 upregulates PORA but not PORB by directly binding to the PORA promoter ( Xu et al., 2015 ). |
| 7 | PORC promoter | DNA | promoter | ... inhibit its DNA binding to the PORC promoter, which suggests an involvement of gibberellin signaling ... |
| 7 | RBCS | DNA | promoter | However, loss of ABI4 did not affect the RBCS promoter activity even when chloroplasts were disrupted ... |
| 7 | PhANGs | DNA | genes | ... photosynthesis-associated nuclear genes (PhANGs) ( Kobayashi et al., 2012b ). |
| 84, 105 | RyhB | RNA | srna | (B) Post-transcriptional repression of enterobactin synthesis by the RyhB sRNA (small RNA) under high-iron conditions. |
| 7 | LHCB1.1 | RNA | transcripts | HY5 is required for regulating the circadian oscillation of the LHCB1.1 transcripts ( Andronis et al., 2008 ). |

## Doubtful (judgement)
* (fix round) **LHCB1.1 is now conflicting** and stays unset, as this section below argued it should. The text below describes the pre-fix behaviour.
* **LHCB1.1 (RNA)**: weakest. Other mentions say "the CAB locus consisting of LHCB1.1 ..." and "LHCB1.1 and LHCB1.3 promoters" (gene sense) but those cues sit more than 2 tokens away, so the rule sees one RNA mention and no conflict. Whole-sentence rule would call it conflicting. Probably should stay unresolved; this is the known ceiling of a token window (a coordinated list hides the head noun).
* **araO2, araI2, RBCS** (DNA by "promoter"): the type is right for a regulatory element or a gene, but the cue describes the promoter of the thing named, not the thing itself ("RBCS promoter activity"). Same for any bare gene symbol: DNA is the sound type for the row, still an inference from adjacency.
* **RyhB (RNA)** is correct (a small RNA) but is now left unset: the whole-sentence rule calls it conflicting and the intersection follows the stricter rule. That is a loss of a correct answer, accepted for the rule the reviewer asked for.
* **miR171c** not resolved: "miR" is a microRNA name prefix, but that is a naming convention, not a cue in this rule. A name-pattern rule would settle it; it belongs to `nucleotide_type_for`, not this task.

## NOT established
* No harness re-import after this change; the 33 message count is an estimate (see above). The fix round did not re-run the replay or the harness either.
* No live run: in a live `run_pathway` run the text is `driver._paper_text(paper)` (set on `state` only with the flag on); not exercised end to end, the replay used the stored `01_source_text.txt`. The Streamlit app path does not set the text, so there the rows stay unset (counted as `no_paper_text`).
* Only 6 of 152 payloads have nucleic acids; precision beyond ~16 rows is untested. Name-only runs have no paper, so they never resolve here.
* No hand check of the 12 "no cue" rows beyond the 4 inspected (ALAD1/HEMB1, miR171c, CHL27, LHCB1.1).
