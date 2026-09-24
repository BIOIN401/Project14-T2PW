# PILOT-003 audit: pathway mining, jasmonate biosynthesis, *Arabidopsis thaliana*

Read-only evaluation of `C:\t\mining-pilot-runs3\pilot-003-jasmonate-ath` (code `C:/t/mining-pilot3 @ f9556288065e630260a9ff1106d0266787d04b9c`, ORCH-743 iteration 2, pre-registered at ledger commit `36230ba6`). Nothing was re-run and no model was called. Every number is read from a named artifact field by [`build_audit3.py`](build_audit3.py), with the narrative in [`md3.py`](md3.py). It ran through the pilot-3 `bounded_run.py` with `PYTHONPATH=C:/t/mining-pilot3/src` and `PYTHONDONTWRITEBYTECODE=1`, and left 0 surviving processes. The frozen code was imported only for pure functions: the title-organism detector, `full_text_matches_title`, the species marker tests, and `frontier.generate` (replayed on `state.json` with the frontier cap lifted). The machine-readable twin is [`pilot_003_audit.json`](pilot_003_audit.json). The comparisons use pilot-001 (`C:/t/mining-pilot/docs/pathway_mining/pilot/001/run`) and pilot-002 (`C:/t/mining-pilot-runs2/`).

**Headline.** `stopped_at_limit` / `max_new_reactions_per_round_reached` (new reactions in round 1: 35 >= limit 25). The run did **1 round** in 1115.319 s at an estimated $0.035558. It discovered 64 papers, included and processed 4, and made 16 extraction calls, all `ok`, which used 18 provider requests. 38 claims became 37 identities: **0 primary** reactions, 35 candidates and 2 `incomplete` claims.

**Key findings.**

1. **Batching fixed the empty completions, with little headroom left.** All 16 calls returned text on their first ladder rung. Only 2 provider requests came back empty, both in `_0008`; the client retried them internally and the third request answered. But the Marchantia review's batches still used 93%, 97%, 84% of the 12,000-token budget (0005, 0006, 0007), and `_0008`'s two empty requests hit it exactly.
2. **Four answers show almost no reasoning. The provider is not recorded.** Calls 0001, 0003, 0008, 0014 answered with 122-209 output tokens (`_0001`, `_0003` and `_0014` in 4.5-8.3 s). That is about 20-40 tokens beyond the JSON itself, and is consistent with a provider or mode that did not reason. `provider` is empty on all 16 records, and `actual_model` only echoes the requested slug, so the artifacts cannot say who answered. Every one of those answers has no enzyme, gene or species field and no participant citation.
3. **No primary reaction.** The pilot-002 primary (13-LOX, alpha-linolenic acid -> 13-HPOT in *A. thaliana*) came back 3 times from the same paper, now read in full text under its correct id `PMC:PMC8508710`. But the model filled no species, enzyme or gene on any of the three. Two were labelled `multiple_species`: with no species field, classification falls back to the paper's screen metadata, *A. thaliana* + *Homo sapiens*. The third is `inferred_only` because its passage says "as inferred from the dose-response curves".
4. **The stop: 35 new nodes, and none could be admitted.** `max_new_reactions_per_round` counts admitted **and** quarantined reactions new to the graph: 0 + 35 >= 25. 23 came from the Marchantia review (the same paper pilot-001 read), 8 from GH3.10 in-vitro assays, 3 from the 13-LOX paper and 1 from pepper. By name-level reading they are about 28 distinct conversions. The check runs only before a round, so round 1 finished and round 2 never started. The frontier held 60 items, the cap, and 57 were never asked. `frontier.generate` produced 194, and the cap cut **137** of them without recording it (F4 reproduces).
5. **Paper identity (IT2-2) works live.** All 28 NCBI ids are consistent: 19 resolved to a PMCID, 9 kept as PMIDs, and every URL is the paper's own PubMed page. Both included NCBI papers' full texts open with their own title and PMID. All 3 NCBI ids that can be checked against EuropePMC's id for the same article agree. There are 0 duplicate titles (pilot-002 had 6) and 0 full-text refusals (none was needed). 20 titles seen in both pilots changed id; the pilot-002 ids were cited-reference PMCIDs.
6. **Species states are still set by keywords (F3 / A3).** 16 of 17 `inferred_only` labels come from the substring 'ortholog' in the review's passages: its table header "Arabidopsis Ortholog", "lacks JAR1 orthologs", "orthology analysis". That includes the canonical *A. thaliana* OPR3 step, OPDA -> OPC-8:0, where the model cited *A. thaliana*. The pepper introduction is again `in_vitro_or_cell_free` because of an incidental sentence about other work.
7. **Participant citations (IT2-N1) track answer quality.** 29 of 35 candidates carry evidence ids on every participant. 6 carry none, and every one of those came from a call that filled no enzyme or species field either. No field was dropped, and `verbatim_cited_span` was never used.

## 1. Counts, with sources

| metric | value | source field | note |
|---|---|---|---|
| papers_discovered | `64` | summary papers.discovered=64; usage.papers_discovered=64; papers.jsonl rows=64 | by source {'europepmc': 36, 'ncbi': 28}; distinct normalized titles 64 |
| papers_screened | `64` | summary papers.screened=64 |  |
| papers_included | `4` | summary papers.included=4; papers.jsonl empty exclusion_reasons | by route {'mining_rescue route 2': 1, 'production': 3}; by source {'ncbi': 2, 'europepmc': 2} |
| papers_excluded | `60` | summary papers.excluded |  |
| papers_processed | `4` | papers.jsonl processed_round>=0; usage.papers_processed=4 |  |
| claims_extracted | `38` | usage.claims_extracted=38; rounds/round_001/claims_extracted.jsonl rows=38 | claims.jsonl stored=37 (= summary claims.extracted=37, the STORED count); distinct identities=37; latest status {'candidate': 34, 'hypothesis_only': 1, 'incomplete': 2} |
| primary_reactions | `0` | primary_graph.json reaction_count; summary graph.primary_reactions=0 |  |
| candidate_reactions | `35` | candidate_subgraphs.json reactions over count=21 components; summary graph.candidate_reactions=35 | decision status {'hypothesis_only': 1, 'candidate': 34}; the 2 `incomplete` claims (no substrate) have no reaction node |
| rejected_by_reason | `{}` | summary claims.rejected_by_reason | rejections.jsonl 38 rows (every non-admitted claim): {'candidate': 35, 'hypothesis_only': 1, 'incomplete': 2} |
| duplicates_merged | `1` | reaction history 'duplicate_merged'; claims extracted - distinct identities = 1 | ['dn-cis-OPDA -> dn-iso-OPDA'], across two batches of the same paper |
| species_states_reaction_nodes | `{"inferred_only": 17, "in_vitro_or_cell_free": 15, "multiple_species": 2, "species_not_stated": 1}` | candidate_subgraphs.json reactions[].species_state (== summary graph.species_states) | per extracted claim (38): {'multiple_species': 2, 'inferred_only': 18, 'in_vitro_or_cell_free': 15, 'species_not_stated': 3} |
| budget_stops | `"max_new_reactions_per_round_reached"` | run_manifest stop_reason; stop_detail='new reactions in round 1: 35 >= limit 25' | the only limit reached; see section 7 |
| model_calls | `{"ok": 16}` | model_calls.jsonl status (16 logical calls = 16 ladder rungs = usage.llm_calls) | provider requests 18 (model_raw provider_requests); log: 2 'retrying' lines (['length 1/3', 'length 2/3']), 0 'giving up' |
| uncited_field_drops | `0` | state.json errors; rounds/round_001/errors.jsonl rows=0 |  |
| fields_kept_by_attribution | `{"compartment:model_citation": 18, "direction:model_citation": 24, "enzyme:model_citation": 24, "gene:model_citation": 14, "species:model_citation": 30}` | claims_extracted.jsonl field_attribution (38 claims) | stored claims: {'compartment:model_citation': 18, 'direction:model_citation': 23, 'enzyme:model_citation': 24, 'gene:model_citation': 14, 'species:model_citation': 29}; verbatim_cited_span = 0 |
| passages | `{"extracted": 73, "retrieved": 73}` | summary passages; evidence_spans.jsonl rows=73 |  |

## 2. Every reaction (0 primary, 35 candidates) and the 2 incomplete claims

The seed connection is the auditor's name-level reading against the canonical route (lipase -> 13-LOX -> AOS -> AOC -> OPR3 -> beta-oxidation -> JA -> JAR1), used only as a yardstick. 'Marker' is the phrase in the cited passages that set the species state, found by the frozen species tests. 'Participants' counts participants that carry their own evidence ids.

| # | reaction id | equation | enzyme (attribution) | species state / value (attribution) | marker that set the state | paper | call(s) | participants cited | seed connection | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `rx_3a229b45b4b5cfa65ddc` | 16:1(n9) -> 16:2(n6) | MpFAD6 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `hypothesis_only` |
| 2 | `rx_c8eec14062ae5544f820` | dn-iso-OPDA -> dn-OPDA-Glu + dn-OPDA-Gln + dn-OPDA-His | MpGH3A (model_citation) | `in_vitro_or_cell_free` / Marchantia polymorpha (model_citation) | in vitro, recombinant enzyme assay | `PMC:PMC13364333` | 0006 | 4/4 | parallel C16 branch | `candidate` |
| 3 | `rx_cbf86a95dc4f74026267` | dn-cis-OPDA -> OPC-6:0 | OPR3 (model_citation) | `inferred_only` / Arabidopsis thaliana (model_citation) | ortholog | `PMC:PMC13364333` | 0006 | 2/2 | parallel C16 branch | `candidate` |
| 4 | `rx_f6e5d9f747dbe8e43554` | dn-cis-OPDA -> dn-iso-OPDA | none | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | parallel C16 branch | `candidate` |
| 5 | `rx_d86306a16865c7a08cec` | Δ4-dn-cis-OPDA -> Δ4-dn-iso-OPDA | none | `in_vitro_or_cell_free` / Marchantia polymorpha (model_citation) | in vitro, recombinant enzyme assay | `PMC:PMC13364333` | 0006 | 2/2 | parallel C16 branch | `candidate` |
| 6 | `rx_8ffac9975e3a977a75f0` | allene oxide -> OPDA | MpAOC (model_citation) | `in_vitro_or_cell_free` / Marchantia polymorpha (model_citation) | in vitro, recombinant enzyme assay | `PMC:PMC13364333` | 0006 | 2/2 | DIRECT (makes seed 12-OPDA) | `candidate` |
| 7 | `rx_e818daf35c555fd843da` | allene oxide -> (9S,13S)-dn-cis-OPDA | MpAOC (model_citation) | `in_vitro_or_cell_free` / Marchantia polymorpha (model_citation) | in vitro, recombinant enzyme assay | `PMC:PMC13364333` | 0006 | 2/2 | parallel C16 branch | `candidate` |
| 8 | `rx_60e130db9d1a0aa23dce` | allene oxide -> (9S,13S)-OPDA | MpAOC (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | DIRECT (makes seed 12-OPDA) | `candidate` |
| 9 | `rx_d438a2fdb9dede27f743` | C18-allene oxide -> cis-OPDA | PpAOC1 (model_citation) | `in_vitro_or_cell_free` / Physcomitrium patens (model_citation) | in vitro | `PMC:PMC13364333` | 0005 | 2/2 | DIRECT (makes seed 12-OPDA) | `candidate` |
| 10 | `rx_2889b990927302948b2e` | ALA (18:3n3) -> 13-HPOT | none | `inferred_only` / (empty) | ortholog, inferred from | `PMC:PMC13364333` | 0008 | 0/2 | upstream, 2 steps | `candidate` |
| 11 | `rx_41c97e790d2e17c19477` | LeA -> 13-HPOT | none | `inferred_only` / (empty) | inferred from | `PMC:PMC8508710` | 0002 | 0/2 | upstream, 2 steps | `candidate` |
| 12 | `rx_05534dc3edb77ff33a5e` | jasmonic acid -> JA-Gln | none | `in_vitro_or_cell_free` / Arabidopsis thaliana (model_citation) | in vitro | `PMC:PMC12096059` | 0009 | 2/2 | downstream side branch | `candidate` |
| 13 | `rx_2f53a30ba9d7528b9b3c` | jasmonic acid -> JA-Asn | none | `in_vitro_or_cell_free` / Arabidopsis thaliana (model_citation) | in vitro | `PMC:PMC12096059` | 0009 | 2/2 | downstream side branch | `candidate` |
| 14 | `rx_742cac9b32cba9f3bff8` | jasmonic acid -> jasmonoyl-isoleucine (JA-Ile) | GH3.10 (model_citation) | `in_vitro_or_cell_free` / Arabidopsis thaliana (model_citation) | in vitro | `PMC:PMC12096059` | 0009 | 2/2 | downstream, 4 steps | `candidate` |
| 15 | `rx_edc1ad1fa1676f1685fd` | jasmonic acid -> JA-Glu | none | `in_vitro_or_cell_free` / Arabidopsis thaliana (model_citation) | in vitro | `PMC:PMC12096059` | 0009 | 2/2 | downstream side branch | `candidate` |
| 16 | `rx_354a031228c7b0e2dc3c` | LA (18:2n6) -> GLA (18:3n6) | MpDES6 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `candidate` |
| 17 | `rx_78d3551bbaba2ff2d5f0` | ALA (18:3n3) -> SDA (18:4n3) | MpDES6 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `candidate` |
| 18 | `rx_9dc809e3d64d7a6dbcf8` | LA (18:2n6) -> ALA (18:3n3) | MpFAD7 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | upstream precursor | `candidate` |
| 19 | `rx_9ebcd41f3b14a7c49564` | DGLA (20:3n6) -> ARA (20:4n6) | MpDES5 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `candidate` |
| 20 | `rx_a723aaa413ec6bc29766` | SDA (18:4n3) -> ETA (20:4n3) | MpELO1 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `candidate` |
| 21 | `rx_d0557d1ccb1153bdae8a` | GLA (18:3n6) -> DGLA (20:3n6) | MpELO1 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `candidate` |
| 22 | `rx_f7233e1139027ba46f53` | ETA (20:4n3) -> EPA (20:5n3) | MpDES5 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `candidate` |
| 23 | `rx_0d8d43b8211e3ff44724` | ARA (20:4n6) -> EPA (20:5n3) | MpFAD3 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `candidate` |
| 24 | `rx_6570fd856f1200fa7a05` | α-linolenic acid (LeA) -> 13(S)-hydroperoxy-9(Z),11(E),15(Z)-octadecatrienoic acid (13-HPOT) + 12-oxo-phytodienoic acid (12-OPDA) | none | `multiple_species` / (empty) | paper metadata | `PMC:PMC8508710` | 0003 | 0/3 | collapsed cascade to the seed | `candidate` |
| 25 | `rx_ce92a2647a229cc6528e` | α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT) | none | `multiple_species` / (empty) | paper metadata | `PMC:PMC8508710` | 0001 | 0/2 | upstream, 2 steps | `candidate` |
| 26 | `rx_eba8e16b5409af0ff75a` | galactolipids -> α-linolenic acid | none | `in_vitro_or_cell_free` / (empty) | in vitro | `PMC:PMC12787732` | 0013 | 0/2 | upstream, 3 steps | `candidate` |
| 27 | `rx_3216468cdedfd13f06ef` | 12-hydroxy-JA -> 12-hydroxy-JA conjugate | GH3.10 (model_citation) | `in_vitro_or_cell_free` / Arabidopsis thaliana (model_citation) | in vitro | `PMC:PMC12096059` | 0009 | 2/2 | downstream, turnover branch | `candidate` |
| 28 | `rx_642b4a34498c459fb3f2` | C20-allene oxide derived from 12-HPETE -> 11-oxo-prostatrienoic acid (11-OPTA) | PpAOC2 (model_citation) | `in_vitro_or_cell_free` / Physcomitrium patens (model_citation) | in vitro | `PMC:PMC13364333` | 0005 | 2/2 | C20 branch, not the route | `candidate` |
| 29 | `rx_30cb458fb147c3de7897` | 16:0 -> 16:1n9 | MpFAD5 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | off the route | `candidate` |
| 30 | `rx_9a6326c17c7be6ac5412` | 12-hydroxy jasmonate -> 12-hydroxy jasmonoyl-isoleucine | GH3.10 (model_citation) | `in_vitro_or_cell_free` / Arabidopsis thaliana (model_citation) | in vitro | `PMC:PMC12096059` | 0010 | 2/2 | downstream, turnover branch | `candidate` |
| 31 | `rx_4764f832e88bebda27cb` | JA + Ile + ATP -> JA-Ile | GH3.10 (model_citation) | `in_vitro_or_cell_free` / Arabidopsis thaliana (model_citation) | in vitro, purified enzyme | `PMC:PMC12096059` | 0011 | 4/4 | downstream, 4 steps | `candidate` |
| 32 | `rx_90a150f373302278fb2f` | 12-OH-JA + Ile + ATP -> 12-OH-JA-Ile | GH3.10 (model_citation) | `in_vitro_or_cell_free` / Arabidopsis thaliana (model_citation) | in vitro, purified enzyme | `PMC:PMC12096059` | 0011 | 4/4 | downstream, turnover branch | `candidate` |
| 33 | `rx_3dc94e3847d88fa54c12` | HDA (16:2n6) -> HTA (16:3n3) | MpFAD7 (model_citation) | `inferred_only` / Marchantia polymorpha (model_citation) | ortholog | `PMC:PMC13364333` | 0005 | 2/2 | upstream C16 precursor | `candidate` |
| 34 | `rx_14eb3a43a495d7ce851f` | OPDA -> OPC-8:0 | OPR3 (model_citation) | `inferred_only` / Arabidopsis thaliana (model_citation) | ortholog | `PMC:PMC13364333` | 0006 | 2/2 | DIRECT (consumes seed 12-OPDA) | `candidate` |
| 35 | `rx_eab7ca60441693867cca` | HDA (16:2n6) -> HTA (16:3n3) | none | `species_not_stated` / (empty) | - | `PMC:PMC13364333` | 0007 | 0/2 | upstream C16 precursor | `candidate` |

Incomplete claims (no reaction node; decision `incomplete`):

- `cl_3acaf053a4653cf7` from `mc_001_extract_claims_0014` (`PMC:PMC12787732`): "? -> JA". Reason: no substrates supported by the evidence; retained and reconsiderable if a later round supplies them
- `cl_1311193317769ac2` from `mc_001_extract_claims_0014` (`PMC:PMC12787732`): "? -> JA-Ile". Reason: no substrates supported by the evidence; retained and reconsiderable if a later round supplies them

### Per-reaction detail

The excerpt is the sentence of the first cited span that names the most participants (verbatim; the frozen `_SENTENCE` splitter was used). Where 0 names matched, the model's wording is not in the span, and the full span is in Appendix A.

#### 1. `rx_3a229b45b4b5cfa65ddc`: 16:1(n9) -> 16:2(n6)

- **Graph:** component `cmp_00119fa52993`, `hypothesis_only`. Decision `hypothesis_only` (`dc_b3fcd36dbdf97b70`): "the source presents this as a proposal or model, not an observed result; retained as a labelled hypothesis, never as an established reaction"
- **Enzyme:** MpFAD6 (`model_citation`). **Gene:** Mp4g24020. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_e78ca4abb54a3fc8; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > Yes (CC) a MpFAD6 (FAD10) Mp4g24020 FAD6 Chloroplast (predicted) Δ 12 -desaturation of 16:1(n9) → 16:2(n6) (HDA); obligate step toward HTA Inferred by pathway logic; no functional study reported.
- **Participants:** substrate 16:1(n9): ev_20a7facf5fd73dd9; product 16:2(n6): ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: none

#### 2. `rx_c8eec14062ae5544f820`: dn-iso-OPDA -> dn-OPDA-Glu + dn-OPDA-Gln + dn-OPDA-His

- **Graph:** component `cmp_14e7f709c875`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_d420dae5be0d4ec2`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpGH3A (`model_citation`). **Gene:** none. **Species:** `in_vitro_or_cell_free`, Marchantia polymorpha (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro, recombinant enzyme assay):
  > 15 S -LOX activity on EPA/ARA; secondary 13-LOX on C 18 -PUFAs Biochemical: recombinant enzyme assays [ 29 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_f42d3135f85ff21e; calls mc_001_extract_claims_0006; spans ev_3b1f8607f4b8b229
- **Excerpt** (`ev_3b1f8607f4b8b229`, 1 name(s) matched):
  > No cis -to- iso isomerase Unknown None identified Unknown Conversion of dn- cis -OPDA → dn- iso -OPDA (and Δ 4 -dn- cis - → Δ 4 -dn- iso -OPDA) Enzymatic activity in crude extracts [ 12 ]; no candidate gene identified No MpGH3A Mp6g07600 JAR1/GH3.11 Unknown Amino acid conjugation of dn- iso -OPDA → dn-OPDA-Glu/-Gln/-His (deactivating step) Genetic: Mp gh3a abolishes conjugates, elevates dn-OPDAs, enhances defense [ 15 ].
- **Participants:** substrate dn-iso-OPDA: ev_3b1f8607f4b8b229; product dn-OPDA-Glu: ev_3b1f8607f4b8b229; product dn-OPDA-Gln: ev_3b1f8607f4b8b229; product dn-OPDA-His: ev_3b1f8607f4b8b229
- **Seed connection:** parallel C16 branch: dn-iso-OPDA amino-acid conjugates (MpGH3A). Graph: product_to_substrate via 'dn-iso-OPDA' with `rx_f6e5d9f747dbe8e43554`

#### 3. `rx_cbf86a95dc4f74026267`: dn-cis-OPDA -> OPC-6:0

- **Graph:** component `cmp_35d0b739cdec`, `species:inferred_only`. Decision `candidate` (`dc_15f6786d5cd23d91`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** OPR3 (`model_citation`). **Gene:** none. **Species:** `inferred_only`, Arabidopsis thaliana (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Marchantia lacks JAR1 orthologs and ligand activation is achieved via a cis -to- iso isomerization by a putative isomerase.
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_5338b4fc07a47cea; calls mc_001_extract_claims_0006; spans ev_c98d84c7d84855c1
- **Excerpt** (`ev_c98d84c7d84855c1`, 1 name(s) matched):
  > In Arabidopsis, OPDA and dn- cis -OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0, respectively.
- **Participants:** substrate dn-cis-OPDA: ev_c98d84c7d84855c1; product OPC-6:0: ev_c98d84c7d84855c1
- **Seed connection:** parallel C16 branch: OPR3 on dn-cis-OPDA -> OPC-6:0. Graph: shared_substrate via 'dn-cis-OPDA' with `rx_f6e5d9f747dbe8e43554`

#### 4. `rx_f6e5d9f747dbe8e43554`: dn-cis-OPDA -> dn-iso-OPDA

- **Graph:** component `cmp_35d0b739cdec`, `species:inferred_only`. Decision `candidate` (`dc_f694a09fe1009050`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Taken together with the absence of functional OPR3 and JAR1 orthologs in all bryophytes examined to date, and the phylogenetically independent evolution of the bryophyte CYP74 clade, a coherent picture emerges: the shared scaffold of the jasmonate pathway (lipolytic n3-PUFA release, LOX oxygenation, AOS/AOC processing, and COI1/JAZ/MYC perception) was already established before the divergence of bryophyte and vascular plant lineages, while the specific chemistry of ligand activation and inactivation evolved independently within each lineage.
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_832888dad708cf8e; calls mc_001_extract_claims_0005; spans ev_a82b0ad4450a0bd2, ev_98c3197eb40f9f1e, ev_3b1f8607f4b8b229, ev_c98d84c7d84855c1
- **Excerpt:** no sentence of `ev_a82b0ad4450a0bd2` names a participant as written (full span in Appendix A).
- **Participants:** substrate dn-cis-OPDA: ev_a82b0ad4450a0bd2, ev_98c3197eb40f9f1e; product dn-iso-OPDA: ev_a82b0ad4450a0bd2, ev_98c3197eb40f9f1e
- **Seed connection:** parallel C16 branch: dn-cis-OPDA isomerisation. Graph: product_to_substrate via 'dn-iso-OPDA' with `rx_c8eec14062ae5544f820`; shared_substrate via 'dn-cis-OPDA' with `rx_cbf86a95dc4f74026267`

#### 5. `rx_d86306a16865c7a08cec`: Δ4-dn-cis-OPDA -> Δ4-dn-iso-OPDA

- **Graph:** component `cmp_5782c1e049fd`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_2737da12f881cafd`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `in_vitro_or_cell_free`, Marchantia polymorpha (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro, recombinant enzyme assay):
  > 15 S -LOX activity on EPA/ARA; secondary 13-LOX on C 18 -PUFAs Biochemical: recombinant enzyme assays [ 29 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_381c10905474b7bf; calls mc_001_extract_claims_0006; spans ev_3b1f8607f4b8b229
- **Excerpt:** no sentence of `ev_3b1f8607f4b8b229` names a participant as written (full span in Appendix A).
- **Participants:** substrate Δ4-dn-cis-OPDA: ev_3b1f8607f4b8b229; product Δ4-dn-iso-OPDA: ev_3b1f8607f4b8b229
- **Seed connection:** parallel C16 branch: Delta4-dn-OPDA isomerisation. Graph: none

#### 6. `rx_8ffac9975e3a977a75f0`: allene oxide -> OPDA

- **Graph:** component `cmp_5c1f4c272b39`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_bf9796ae848c97a8`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpAOC (`model_citation`). **Gene:** none. **Species:** `in_vitro_or_cell_free`, Marchantia polymorpha (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro, recombinant enzyme assay):
  > 15 S -LOX activity on EPA/ARA; secondary 13-LOX on C 18 -PUFAs Biochemical: recombinant enzyme assays [ 29 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_87cb48915297c615; calls mc_001_extract_claims_0006; spans ev_3b1f8607f4b8b229
- **Excerpt** (`ev_3b1f8607f4b8b229`, 2 name(s) matched):
  > Yes (CC) c MpAOC (DIR17) Mp7g06220 AOC1-4 Chloroplast (confirmed) Stereospecific ring closure of allene oxide → (9 S ,13 S )-dn- cis -OPDA or OPDA.
- **Participants:** substrate allene oxide: ev_3b1f8607f4b8b229; product OPDA: ev_3b1f8607f4b8b229
- **Seed connection:** DIRECT (makes seed 12-OPDA): AOC step again, MpAOC, other passage. Graph: product_to_substrate via 'OPDA' with `rx_14eb3a43a495d7ce851f`; shared_substrate via 'allene oxide' with `rx_60e130db9d1a0aa23dce`; shared_substrate via 'allene oxide' with `rx_e818daf35c555fd843da`

#### 7. `rx_e818daf35c555fd843da`: allene oxide -> (9S,13S)-dn-cis-OPDA

- **Graph:** component `cmp_5c1f4c272b39`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_a0beefdd5278c683`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpAOC (`model_citation`). **Gene:** none. **Species:** `in_vitro_or_cell_free`, Marchantia polymorpha (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro, recombinant enzyme assay):
  > 15 S -LOX activity on EPA/ARA; secondary 13-LOX on C 18 -PUFAs Biochemical: recombinant enzyme assays [ 29 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_8ee4bd53fcfb1b9c; calls mc_001_extract_claims_0006; spans ev_3b1f8607f4b8b229
- **Excerpt** (`ev_3b1f8607f4b8b229`, 1 name(s) matched):
  > Yes (CC) c MpAOC (DIR17) Mp7g06220 AOC1-4 Chloroplast (confirmed) Stereospecific ring closure of allene oxide → (9 S ,13 S )-dn- cis -OPDA or OPDA.
- **Participants:** substrate allene oxide: ev_3b1f8607f4b8b229; product (9S,13S)-dn-cis-OPDA: ev_3b1f8607f4b8b229
- **Seed connection:** parallel C16 branch: AOC step to dn-cis-OPDA (bryophyte route). Graph: shared_substrate via 'allene oxide' with `rx_60e130db9d1a0aa23dce`; shared_substrate via 'allene oxide' with `rx_8ffac9975e3a977a75f0`

#### 8. `rx_60e130db9d1a0aa23dce`: allene oxide -> (9S,13S)-OPDA

- **Graph:** component `cmp_6a218c162b75`, `species:inferred_only`. Decision `candidate` (`dc_876b56d7573bc8a1`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpAOC (`model_citation`). **Gene:** Mp7g06220. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_2865e3d8c80b6688; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9, ev_a82b0ad4450a0bd2
- **Excerpt** (`ev_20a7facf5fd73dd9`, 1 name(s) matched):
  > Yes (CC) c MpAOC (DIR17) Mp7g06220 AOC1-4 Chloroplast (confirmed) Stereospecific ring closure of allene oxide → (9 S ,13 S )-dn- cis -OPDA or OPDA.
- **Participants:** substrate allene oxide: ev_20a7facf5fd73dd9, ev_a82b0ad4450a0bd2; product (9S,13S)-OPDA: ev_20a7facf5fd73dd9, ev_a82b0ad4450a0bd2
- **Seed connection:** DIRECT (makes seed 12-OPDA): AOC step, Marchantia MpAOC. Graph: shared_substrate via 'allene oxide' with `rx_8ffac9975e3a977a75f0`; shared_substrate via 'allene oxide' with `rx_e818daf35c555fd843da`

#### 9. `rx_d438a2fdb9dede27f743`: C18-allene oxide -> cis-OPDA

- **Graph:** component `cmp_7e72499580bf`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_370870dda2253578`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** PpAOC1 (`model_citation`). **Gene:** none. **Species:** `in_vitro_or_cell_free`, Physcomitrium patens (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > Recombinant MpAOC produces the naturally occurring (9 S ,13 S )-OPDA in a coupled in vitro assay with PpAOS1 (used as the AOS reaction partner) and 13-HPOT as substrate, and subcellular localization studies using MpAOC-GFP fusion proteins confirmed chloroplast targeting, mirroring the plastid-localized AOC enzymes of angiosperms [ 65 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_1ac4d9ef538231c7; calls mc_001_extract_claims_0005; spans ev_a82b0ad4450a0bd2
- **Excerpt:** no sentence of `ev_a82b0ad4450a0bd2` names a participant as written (full span in Appendix A).
- **Participants:** substrate C18-allene oxide: ev_a82b0ad4450a0bd2; product cis-OPDA: ev_a82b0ad4450a0bd2
- **Seed connection:** DIRECT (makes seed 12-OPDA): AOC step, moss PpAOC1 (cis-OPDA = 12-OPDA). Graph: none

#### 10. `rx_2889b990927302948b2e`: ALA (18:3n3) -> 13-HPOT

- **Graph:** component `cmp_8117f67d1867`, `species:inferred_only`. Decision `candidate` (`dc_765bfeffc2d0b28b`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog, inferred from); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `inferred_only`, (empty). Reason: "species support is an inference, not an observation (ortholog, inferred from); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog, inferred from):
  > LOX protein sequences were selected based on a prior orthology analysis in which LOX homologs belonging to the same orthogroup as M. polymorpha LOX proteins were identified using OrthoFinder [ 52 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_c230261f1437d392; calls mc_001_extract_claims_0008; spans ev_6f4dc3ffa5db1e0c, ev_74d8b3d259ca310a
- **Excerpt** (`ev_6f4dc3ffa5db1e0c`, 1 name(s) matched):
  > characterized MpLOX7 as a multifunctional enzyme with an unusually broad substrate tolerance, accepting both C 18 (LA, ALA) and C 20 (ARA, EPA) PUFAs at comparable efficiency, with ALA showing the highest k cat value [ 49 ].
- **Participants:** substrate ALA (18:3n3): uncited; product 13-HPOT: uncited
- **Seed connection:** upstream, 2 steps: 13-LOX step from the Marchantia review. Graph: product_to_substrate via 'ALA (18:3n3)' with `rx_9dc809e3d64d7a6dbcf8`; shared_product via '13-HPOT' with `rx_41c97e790d2e17c19477`; shared_substrate via 'ALA (18:3n3)' with `rx_78d3551bbaba2ff2d5f0`

#### 11. `rx_41c97e790d2e17c19477`: LeA -> 13-HPOT

- **Graph:** component `cmp_8117f67d1867`, `species:inferred_only`. Decision `candidate` (`dc_e1dc4113b7483e3f`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (inferred from); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `inferred_only`, (empty). Reason: "species support is an inference, not an observation (inferred from); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (inferred from):
  > Due to the effects of pH, reliable IC 50 values could not be derived for LOX2 and LOX6; however, as inferred from the dose-response curves and data shown in Table 3 , these IC 50 values likely range between 280 and 338 µM.
- **Paper:** `PMC:PMC8508710`, "Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana"; claims cl_7a9937b3db2c395f; calls mc_001_extract_claims_0002; spans ev_4dcc372e77b84976, ev_d0a63371734346ce, ev_bdc3296114db3865, ev_baf1514923696f5c
- **Excerpt** (`ev_4dcc372e77b84976`, 1 name(s) matched):
  > The reliability was assessed with the criterion that an increase in A 234nm and, ideally, oxygen consumption occurred via the injection of plant tissue extract (0.5–2 µg/µL protein in 50 mM Tris-HCl or K-P i , pH 6.6–8.0) in LeA (tested at 4–100 µM).
- **Participants:** substrate LeA: uncited; product 13-HPOT: uncited
- **Seed connection:** upstream, 2 steps: 13-LOX step again (full text of the same 13-LOX paper). Graph: shared_product via '13-HPOT' with `rx_2889b990927302948b2e`

#### 12. `rx_05534dc3edb77ff33a5e`: jasmonic acid -> JA-Gln

- **Graph:** component `cmp_9abd34f16007`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_0d91bcbacfd875fa`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `in_vitro_or_cell_free`, Arabidopsis thaliana (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > The substrate range of recombinant GH3.10 and related GH3 enzymes was studied using non-targeted ex vivo metabolomics with flower and leaf extracts of A. thaliana and in vitro enzyme assays.
- **Paper:** `PMC:PMC12096059`, "Arabidopsis GH3.10 conjugates jasmonates"; claims cl_5fc57dcfa32aaf35; calls mc_001_extract_claims_0009; spans ev_7d2787cbb14b4905
- **Excerpt** (`ev_7d2787cbb14b4905`, 1 name(s) matched):
  > Additional JA conjugates found in wounded leaves (JA-Gln) or formed in flowers upon MeJA treatment in the absence of JA-Ile (JA-Gln, JA-Asn, JA-Glu) were identified.
- **Participants:** substrate jasmonic acid: ev_7d2787cbb14b4905; product JA-Gln: ev_7d2787cbb14b4905
- **Seed connection:** downstream side branch: JA -> JA-Gln conjugate. Graph: shared_substrate via 'jasmonic acid' with `rx_2f53a30ba9d7528b9b3c`; shared_substrate via 'jasmonic acid' with `rx_742cac9b32cba9f3bff8`; shared_substrate via 'jasmonic acid' with `rx_edc1ad1fa1676f1685fd`

#### 13. `rx_2f53a30ba9d7528b9b3c`: jasmonic acid -> JA-Asn

- **Graph:** component `cmp_9abd34f16007`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_c63ec4d923d5ff9c`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `in_vitro_or_cell_free`, Arabidopsis thaliana (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > The substrate range of recombinant GH3.10 and related GH3 enzymes was studied using non-targeted ex vivo metabolomics with flower and leaf extracts of A. thaliana and in vitro enzyme assays.
- **Paper:** `PMC:PMC12096059`, "Arabidopsis GH3.10 conjugates jasmonates"; claims cl_2d06a621ba1c2fac; calls mc_001_extract_claims_0009; spans ev_7d2787cbb14b4905
- **Excerpt** (`ev_7d2787cbb14b4905`, 1 name(s) matched):
  > Additional JA conjugates found in wounded leaves (JA-Gln) or formed in flowers upon MeJA treatment in the absence of JA-Ile (JA-Gln, JA-Asn, JA-Glu) were identified.
- **Participants:** substrate jasmonic acid: ev_7d2787cbb14b4905; product JA-Asn: ev_7d2787cbb14b4905
- **Seed connection:** downstream side branch: JA -> JA-Asn conjugate. Graph: shared_substrate via 'jasmonic acid' with `rx_05534dc3edb77ff33a5e`; shared_substrate via 'jasmonic acid' with `rx_742cac9b32cba9f3bff8`; shared_substrate via 'jasmonic acid' with `rx_edc1ad1fa1676f1685fd`

#### 14. `rx_742cac9b32cba9f3bff8`: jasmonic acid -> jasmonoyl-isoleucine (JA-Ile)

- **Graph:** component `cmp_9abd34f16007`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_65f7d35c0b538491`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** GH3.10 (`model_citation`). **Gene:** GH3.10. **Species:** `in_vitro_or_cell_free`, Arabidopsis thaliana (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > The substrate range of recombinant GH3.10 and related GH3 enzymes was studied using non-targeted ex vivo metabolomics with flower and leaf extracts of A. thaliana and in vitro enzyme assays.
- **Paper:** `PMC:PMC12096059`, "Arabidopsis GH3.10 conjugates jasmonates"; claims cl_b89be46f92d4c4e1; calls mc_001_extract_claims_0009; spans ev_7d2787cbb14b4905
- **Excerpt** (`ev_7d2787cbb14b4905`, 2 name(s) matched):
  > In angiosperms, the canonical bioactive jasmonate is jasmonoyl-isoleucine (JA-Ile), which is formed in Arabidopsis thaliana by JAR1 and GH3.10.
- **Participants:** substrate jasmonic acid: ev_7d2787cbb14b4905; product jasmonoyl-isoleucine (JA-Ile): ev_7d2787cbb14b4905
- **Seed connection:** downstream, 4 steps: JA -> JA-Ile by GH3.10 (JAR1-type conjugation). Graph: shared_substrate via 'jasmonic acid' with `rx_05534dc3edb77ff33a5e`; shared_substrate via 'jasmonic acid' with `rx_2f53a30ba9d7528b9b3c`; shared_substrate via 'jasmonic acid' with `rx_edc1ad1fa1676f1685fd`

#### 15. `rx_edc1ad1fa1676f1685fd`: jasmonic acid -> JA-Glu

- **Graph:** component `cmp_9abd34f16007`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_3bd46224dd217a31`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `in_vitro_or_cell_free`, Arabidopsis thaliana (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > The substrate range of recombinant GH3.10 and related GH3 enzymes was studied using non-targeted ex vivo metabolomics with flower and leaf extracts of A. thaliana and in vitro enzyme assays.
- **Paper:** `PMC:PMC12096059`, "Arabidopsis GH3.10 conjugates jasmonates"; claims cl_9ba999bda7a136f3; calls mc_001_extract_claims_0009; spans ev_7d2787cbb14b4905
- **Excerpt** (`ev_7d2787cbb14b4905`, 1 name(s) matched):
  > Additional JA conjugates found in wounded leaves (JA-Gln) or formed in flowers upon MeJA treatment in the absence of JA-Ile (JA-Gln, JA-Asn, JA-Glu) were identified.
- **Participants:** substrate jasmonic acid: ev_7d2787cbb14b4905; product JA-Glu: ev_7d2787cbb14b4905
- **Seed connection:** downstream side branch: JA -> JA-Glu conjugate. Graph: shared_substrate via 'jasmonic acid' with `rx_05534dc3edb77ff33a5e`; shared_substrate via 'jasmonic acid' with `rx_2f53a30ba9d7528b9b3c`; shared_substrate via 'jasmonic acid' with `rx_742cac9b32cba9f3bff8`

#### 16. `rx_354a031228c7b0e2dc3c`: LA (18:2n6) -> GLA (18:3n6)

- **Graph:** component `cmp_a8c6dac117f6`, `species:inferred_only`. Decision `candidate` (`dc_ed190fab57b8b1cc`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpDES6 (`model_citation`). **Gene:** Mp6g17330. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_9fa8df5b9e2bb0a4; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > Yes (CC) b MpDES6 Mp6g17330 None (bryophyte-specific) ER (predicted) Δ 6 -desaturation: LA (18:2n6) → GLA (18:3n6); ALA → SDA (18:4n3); entry into LCPUFA elongation Biochemical: heterologous expression in yeast/plants [ 85 , 30 ].
- **Participants:** substrate LA (18:2n6): ev_20a7facf5fd73dd9; product GLA (18:3n6): ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: product_to_substrate via 'GLA (18:3n6)' with `rx_d0557d1ccb1153bdae8a`; shared_substrate via 'LA (18:2n6)' with `rx_9dc809e3d64d7a6dbcf8`

#### 17. `rx_78d3551bbaba2ff2d5f0`: ALA (18:3n3) -> SDA (18:4n3)

- **Graph:** component `cmp_a8c6dac117f6`, `species:inferred_only`. Decision `candidate` (`dc_88006510d6c16c90`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpDES6 (`model_citation`). **Gene:** Mp6g17330. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_7c0b9007ba1f8ec8; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 3 name(s) matched):
  > Yes (CC) b MpDES6 Mp6g17330 None (bryophyte-specific) ER (predicted) Δ 6 -desaturation: LA (18:2n6) → GLA (18:3n6); ALA → SDA (18:4n3); entry into LCPUFA elongation Biochemical: heterologous expression in yeast/plants [ 85 , 30 ].
- **Participants:** substrate ALA (18:3n3): ev_20a7facf5fd73dd9; product SDA (18:4n3): ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: product_to_substrate via 'ALA (18:3n3)' with `rx_9dc809e3d64d7a6dbcf8`; product_to_substrate via 'SDA (18:4n3)' with `rx_a723aaa413ec6bc29766`; shared_substrate via 'ALA (18:3n3)' with `rx_2889b990927302948b2e`

#### 18. `rx_9dc809e3d64d7a6dbcf8`: LA (18:2n6) -> ALA (18:3n3)

- **Graph:** component `cmp_a8c6dac117f6`, `species:inferred_only`. Decision `candidate` (`dc_d9f2427cd80f1c61`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpFAD7 (`model_citation`). **Gene:** Mp5g20700. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_559992a1bf688e15; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > No MpFAD7 (FAD11) Mp5g20700 FAD7/FAD8 Chloroplast (confirmed) n3-desaturation of HDA (16:2n6) → HTA (16:3n3) and LA (18:2n6) → ALA (18:3n3) Genetic: Mp fad7 depleted in HTA/ALA and canonical jasmonates [ 14 ].
- **Participants:** substrate LA (18:2n6): ev_20a7facf5fd73dd9; product ALA (18:3n3): ev_20a7facf5fd73dd9
- **Seed connection:** upstream precursor: makes ALA, the 13-LOX substrate (FAD7 desaturation). Graph: product_to_substrate via 'ALA (18:3n3)' with `rx_2889b990927302948b2e`; product_to_substrate via 'ALA (18:3n3)' with `rx_78d3551bbaba2ff2d5f0`; shared_substrate via 'LA (18:2n6)' with `rx_354a031228c7b0e2dc3c`

#### 19. `rx_9ebcd41f3b14a7c49564`: DGLA (20:3n6) -> ARA (20:4n6)

- **Graph:** component `cmp_a8c6dac117f6`, `species:inferred_only`. Decision `candidate` (`dc_98280da1bf261280`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpDES5 (`model_citation`). **Gene:** Mp3g13250. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_28107dbbe1153e90; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > No MpDES5 Mp3g13250 None (bryophyte-specific) ER (predicted) Δ 5 -desaturation: DGLA (20:3n6) → ARA (20:4n6); ETA (20:4n3) → EPA (20:5n3) Biochemical: heterologous expression in yeast/plants [ 85 , 30 ].
- **Participants:** substrate DGLA (20:3n6): ev_20a7facf5fd73dd9; product ARA (20:4n6): ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: product_to_substrate via 'ARA (20:4n6)' with `rx_0d8d43b8211e3ff44724`; product_to_substrate via 'DGLA (20:3n6)' with `rx_d0557d1ccb1153bdae8a`

#### 20. `rx_a723aaa413ec6bc29766`: SDA (18:4n3) -> ETA (20:4n3)

- **Graph:** component `cmp_a8c6dac117f6`, `species:inferred_only`. Decision `candidate` (`dc_cd03f211ac0c3095`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpELO1 (`model_citation`). **Gene:** Mp6g02350. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_06de0d61ec6ebd01; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > Yes (CC) a MpELO1 Mp6g02350 None (bryophyte-specific) ER (predicted) Δ 6 -elongation: GLA (18:3n6) → DGLA (20:3n6); SDA (18:4n3) → ETA (20:4n3) Biochemical: heterologous expression in yeast/plants [ 85 , 30 ].
- **Participants:** substrate SDA (18:4n3): ev_20a7facf5fd73dd9; product ETA (20:4n3): ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: product_to_substrate via 'ETA (20:4n3)' with `rx_f7233e1139027ba46f53`; product_to_substrate via 'SDA (18:4n3)' with `rx_78d3551bbaba2ff2d5f0`

#### 21. `rx_d0557d1ccb1153bdae8a`: GLA (18:3n6) -> DGLA (20:3n6)

- **Graph:** component `cmp_a8c6dac117f6`, `species:inferred_only`. Decision `candidate` (`dc_17423efeb1294d3d`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpELO1 (`model_citation`). **Gene:** Mp6g02350. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_183ebc7ca093105f; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > Yes (CC) a MpELO1 Mp6g02350 None (bryophyte-specific) ER (predicted) Δ 6 -elongation: GLA (18:3n6) → DGLA (20:3n6); SDA (18:4n3) → ETA (20:4n3) Biochemical: heterologous expression in yeast/plants [ 85 , 30 ].
- **Participants:** substrate GLA (18:3n6): ev_20a7facf5fd73dd9; product DGLA (20:3n6): ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: product_to_substrate via 'DGLA (20:3n6)' with `rx_9ebcd41f3b14a7c49564`; product_to_substrate via 'GLA (18:3n6)' with `rx_354a031228c7b0e2dc3c`

#### 22. `rx_f7233e1139027ba46f53`: ETA (20:4n3) -> EPA (20:5n3)

- **Graph:** component `cmp_a8c6dac117f6`, `species:inferred_only`. Decision `candidate` (`dc_4fb6a97bf3cd0f66`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpDES5 (`model_citation`). **Gene:** Mp3g13250. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_a453942d6bd35571; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > No MpDES5 Mp3g13250 None (bryophyte-specific) ER (predicted) Δ 5 -desaturation: DGLA (20:3n6) → ARA (20:4n6); ETA (20:4n3) → EPA (20:5n3) Biochemical: heterologous expression in yeast/plants [ 85 , 30 ].
- **Participants:** substrate ETA (20:4n3): ev_20a7facf5fd73dd9; product EPA (20:5n3): ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: product_to_substrate via 'ETA (20:4n3)' with `rx_a723aaa413ec6bc29766`; shared_product via 'EPA (20:5n3)' with `rx_0d8d43b8211e3ff44724`

#### 23. `rx_0d8d43b8211e3ff44724`: ARA (20:4n6) -> EPA (20:5n3)

- **Graph:** component `cmp_af4bce6bf0a5`, `species:inferred_only`. Decision `candidate` (`dc_8c3c9049252e70bc`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpFAD3 (`model_citation`). **Gene:** Mp3g17690. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_b5e2f8082d97ede6; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > Yes (CC) b MpFAD3 (FAD8) Mp3g17690 FAD3 ER (confirmed) n3-desaturation of ARA (20:4n6) → EPA (20:5n3) Genetic: Mp fad3 depleted in EPA and abolishes C 20 -OPDA and Δ 4 -dn- iso -OPDA [ 14 ].
- **Participants:** substrate ARA (20:4n6): ev_20a7facf5fd73dd9; product EPA (20:5n3): ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: product_to_substrate via 'ARA (20:4n6)' with `rx_9ebcd41f3b14a7c49564`; shared_product via 'EPA (20:5n3)' with `rx_f7233e1139027ba46f53`

#### 24. `rx_6570fd856f1200fa7a05`: α-linolenic acid (LeA) -> 13(S)-hydroperoxy-9(Z),11(E),15(Z)-octadecatrienoic acid (13-HPOT) + 12-oxo-phytodienoic acid (12-OPDA)

- **Graph:** component `cmp_bc76b9cc43fd`, `species:multiple_species`. Decision `candidate` (`dc_75e5ce6c1cb99631`): "species state 'multiple_species' is not target-species-supported (the evidence names more than one species including the target; which one the reaction belongs to is not established); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `multiple_species`, (empty). Reason: "the evidence names more than one species including the target; which one the reaction belongs to is not established"
- **Paper:** `PMC:PMC8508710`, "Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana"; claims cl_fcc1f538bbd36817; calls mc_001_extract_claims_0003; spans ev_6915f534ca2dfd74, ev_bb4c9802d1637278
- **Excerpt:** no sentence of `ev_6915f534ca2dfd74` names a participant as written (full span in Appendix A).
- **Participants:** substrate α-linolenic acid (LeA): uncited; product 13(S)-hydroperoxy-9(Z),11(E),15(Z)-octadecatrienoic acid (13-HPOT): uncited; product 12-oxo-phytodienoic acid (12-OPDA): uncited
- **Seed connection:** collapsed cascade to the seed: LOX + AOS + AOC written as one reaction with products 13-HPOT and 12-OPDA; not a single step. Graph: shared_substrate via 'α-linolenic acid (LeA)' with `rx_ce92a2647a229cc6528e`

#### 25. `rx_ce92a2647a229cc6528e`: α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT)

- **Graph:** component `cmp_bc76b9cc43fd`, `species:multiple_species`. Decision `candidate` (`dc_bbb8a44662643c96`): "species state 'multiple_species' is not target-species-supported (the evidence names more than one species including the target; which one the reaction belongs to is not established); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `multiple_species`, (empty). Reason: "the evidence names more than one species including the target; which one the reaction belongs to is not established"
- **Paper:** `PMC:PMC8508710`, "Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana"; claims cl_a32b77c9538b6f39; calls mc_001_extract_claims_0001; spans ev_5863efff93736770, ev_06e5eb88be1f278b
- **Excerpt** (`ev_5863efff93736770`, 4 name(s) matched):
  > 13-lipoxygenases (13-LOX) catalyze the dioxygenation of various polyunsaturated fatty acids (PUFAs), of which α-linolenic acid (LeA) is converted to 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT), the precursor for the prostaglandin-like plant hormones cis-(+)-12-oxophytodienoic acid (12-OPDA) and methyl jasmonate (MJ).
- **Participants:** substrate α-linolenic acid (LeA): uncited; product 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT): uncited
- **Seed connection:** upstream, 2 steps: 13-LOX step (makes the AOS substrate 13-HPOT); the pilot-002 primary reaction's chemistry. Graph: shared_substrate via 'α-linolenic acid (LeA)' with `rx_6570fd856f1200fa7a05`

#### 26. `rx_eba8e16b5409af0ff75a`: galactolipids -> α-linolenic acid

- **Graph:** component `cmp_c312598eda0c`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_bda1df2829af43d8`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `in_vitro_or_cell_free`, (empty). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > Using phosphovariants, in vitro studies confirmed that the unphosphorylatable AtLOX2 had over a 100 times increase in V max compared to the phosphomimic, which likely reflects blocking of the substrate from the catalytic site [ 42 ].
- **Paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves"; claims cl_70223dd072f56b56; calls mc_001_extract_claims_0013; spans ev_f7c56801adf6792d
- **Excerpt** (`ev_f7c56801adf6792d`, 1 name(s) matched):
  > After wounding or recognition of a necrotrophic pathogen, jasmonate biosynthesis starts in the chloroplast, where α -linolenic acid, a substrate of lipoxygenases (LOXs), is released from the galactolipids of the chloroplast membrane by a galactolipase [ 10 ].
- **Participants:** substrate galactolipids: uncited; product α-linolenic acid: uncited
- **Seed connection:** upstream, 3 steps: lipase release of alpha-linolenic acid from galactolipids. Graph: none

#### 27. `rx_3216468cdedfd13f06ef`: 12-hydroxy-JA -> 12-hydroxy-JA conjugate

- **Graph:** component `cmp_cfd9264eb103`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_d967f890e8be5f6e`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** GH3.10 (`model_citation`). **Gene:** GH3.10. **Species:** `in_vitro_or_cell_free`, Arabidopsis thaliana (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > The substrate range of recombinant GH3.10 and related GH3 enzymes was studied using non-targeted ex vivo metabolomics with flower and leaf extracts of A. thaliana and in vitro enzyme assays.
- **Paper:** `PMC:PMC12096059`, "Arabidopsis GH3.10 conjugates jasmonates"; claims cl_31a6adf4b22136d2; calls mc_001_extract_claims_0009; spans ev_7d2787cbb14b4905
- **Excerpt** (`ev_7d2787cbb14b4905`, 1 name(s) matched):
  > While 12-hydroxy-JA was identified as the preferred substrate of GH3.10, no other recombinant GH3 enzymes tested were capable of JA-Ile formation.
- **Participants:** substrate 12-hydroxy-JA: ev_7d2787cbb14b4905; product 12-hydroxy-JA conjugate: ev_7d2787cbb14b4905
- **Seed connection:** downstream, turnover branch: 12-OH-JA -> an unnamed conjugate. Graph: none

#### 28. `rx_642b4a34498c459fb3f2`: C20-allene oxide derived from 12-HPETE -> 11-oxo-prostatrienoic acid (11-OPTA)

- **Graph:** component `cmp_d6aa0a82726e`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_48edcd74bac614ba`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** PpAOC2 (`model_citation`). **Gene:** none. **Species:** `in_vitro_or_cell_free`, Physcomitrium patens (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > Recombinant MpAOC produces the naturally occurring (9 S ,13 S )-OPDA in a coupled in vitro assay with PpAOS1 (used as the AOS reaction partner) and 13-HPOT as substrate, and subcellular localization studies using MpAOC-GFP fusion proteins confirmed chloroplast targeting, mirroring the plastid-localized AOC enzymes of angiosperms [ 65 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_0e664d8a3907d58b; calls mc_001_extract_claims_0005; spans ev_a82b0ad4450a0bd2
- **Excerpt** (`ev_a82b0ad4450a0bd2`, 2 name(s) matched):
  > PpAOC2, by contrast, accepts both the C 18 -derived allene oxide and the allene oxide from 12-HPETE (a C 20 ARA-derived substrate), producing the novel cyclopentenone 11-oxo-prostatrienoic acid (11-OPTA) [ 62 ].
- **Participants:** substrate C20-allene oxide derived from 12-HPETE: ev_a82b0ad4450a0bd2; product 11-oxo-prostatrienoic acid (11-OPTA): ev_a82b0ad4450a0bd2
- **Seed connection:** C20 branch, not the route: AOC-type cyclisation of a C20 allene oxide (PpAOC2). Graph: none

#### 29. `rx_30cb458fb147c3de7897`: 16:0 -> 16:1n9

- **Graph:** component `cmp_db257e922c58`, `species:inferred_only`. Decision `candidate` (`dc_0fbbb11e15c99ffd`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpFAD5 (`model_citation`). **Gene:** Mp3g10660. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_5c72ac211f2bf29a; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 2 name(s) matched):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Participants:** substrate 16:0: ev_20a7facf5fd73dd9; product 16:1n9: ev_20a7facf5fd73dd9
- **Seed connection:** off the route: fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route. Graph: none

#### 30. `rx_9a6326c17c7be6ac5412`: 12-hydroxy jasmonate -> 12-hydroxy jasmonoyl-isoleucine

- **Graph:** component `cmp_dd8384a54e53`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_e57fcf7565b5587b`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** GH3.10 (`model_citation`). **Gene:** none. **Species:** `in_vitro_or_cell_free`, Arabidopsis thaliana (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro):
  > The substrate range of recombinant GH3.10 and related GH3 enzymes was studied using non‐targeted ex vivo metabolomics with flower and leaf extracts of A. thaliana and in vitro enzyme assays.
- **Paper:** `PMC:PMC12096059`, "Arabidopsis GH3.10 conjugates jasmonates"; claims cl_dee44fedcb6c01eb; calls mc_001_extract_claims_0010; spans ev_1f2c624a181ba5ef, ev_d216d8e2a43cb218
- **Excerpt:** no sentence of `ev_1f2c624a181ba5ef` names a participant as written (full span in Appendix A).
- **Participants:** substrate 12-hydroxy jasmonate: ev_1f2c624a181ba5ef, ev_d216d8e2a43cb218; product 12-hydroxy jasmonoyl-isoleucine: ev_1f2c624a181ba5ef
- **Seed connection:** downstream, turnover branch: 12-OH-JA -> 12-OH-JA-Ile (GH3.10). Graph: none

#### 31. `rx_4764f832e88bebda27cb`: JA + Ile + ATP -> JA-Ile

- **Graph:** component `cmp_dea944368da1`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_d68ef968338a30ad`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** GH3.10 (`model_citation`). **Gene:** none. **Species:** `in_vitro_or_cell_free`, Arabidopsis thaliana (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro, purified enzyme):
  > For the determination of relative enzyme activity, GH3.10 and JAR1 were purified in parallel and tested for enzyme activity by in vitro incubations with 4,5‐didehydro JA (4,5‐ddh‐JA), JA, or 12‐OH‐JA with ATP and Ile.
- **Paper:** `PMC:PMC12096059`, "Arabidopsis GH3.10 conjugates jasmonates"; claims cl_150891597632f0c5; calls mc_001_extract_claims_0011; spans ev_ae0d92db16547d2c, ev_c7a1b8c0ccdfe997, ev_6893aa984a651e1f
- **Excerpt** (`ev_ae0d92db16547d2c`, 1 name(s) matched):
  > References Acosta I.F. , Przybyl M. ( 2019 ) Jasmonate signaling during Arabidopsis stamen maturation .
- **Participants:** substrate JA: ev_6893aa984a651e1f, ev_c7a1b8c0ccdfe997; substrate Ile: ev_6893aa984a651e1f, ev_c7a1b8c0ccdfe997; substrate ATP: ev_6893aa984a651e1f, ev_c7a1b8c0ccdfe997; product JA-Ile: ev_6893aa984a651e1f
- **Seed connection:** downstream, 4 steps: the same JA -> JA-Ile step written with Ile and ATP. Graph: shared_substrate via 'Ile' with `rx_90a150f373302278fb2f`

#### 32. `rx_90a150f373302278fb2f`: 12-OH-JA + Ile + ATP -> 12-OH-JA-Ile

- **Graph:** component `cmp_dea944368da1`, `species:in_vitro_or_cell_free`. Decision `candidate` (`dc_0036e04561c9f3d3`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** GH3.10 (`model_citation`). **Gene:** none. **Species:** `in_vitro_or_cell_free`, Arabidopsis thaliana (`model_citation`). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Marker sentence** (in vitro, purified enzyme):
  > For the determination of relative enzyme activity, GH3.10 and JAR1 were purified in parallel and tested for enzyme activity by in vitro incubations with 4,5‐didehydro JA (4,5‐ddh‐JA), JA, or 12‐OH‐JA with ATP and Ile.
- **Paper:** `PMC:PMC12096059`, "Arabidopsis GH3.10 conjugates jasmonates"; claims cl_061c152eb99e1656; calls mc_001_extract_claims_0011; spans ev_ae0d92db16547d2c, ev_c7a1b8c0ccdfe997, ev_6893aa984a651e1f
- **Excerpt:** no sentence of `ev_ae0d92db16547d2c` names a participant as written (full span in Appendix A).
- **Participants:** substrate 12-OH-JA: ev_6893aa984a651e1f, ev_c7a1b8c0ccdfe997; substrate Ile: ev_6893aa984a651e1f, ev_c7a1b8c0ccdfe997; substrate ATP: ev_6893aa984a651e1f, ev_c7a1b8c0ccdfe997; product 12-OH-JA-Ile: ev_6893aa984a651e1f
- **Seed connection:** downstream, turnover branch: the same 12-OH-JA -> 12-OH-JA-Ile step written with Ile and ATP. Graph: shared_substrate via 'Ile' with `rx_4764f832e88bebda27cb`

#### 33. `rx_3dc94e3847d88fa54c12`: HDA (16:2n6) -> HTA (16:3n3)

- **Graph:** component `cmp_e4fbeae15dea`, `species:inferred_only`. Decision `candidate` (`dc_65679103eb217412`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** MpFAD7 (`model_citation`). **Gene:** Mp5g20700. **Species:** `inferred_only`, Marchantia polymorpha (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ].
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_5592a5f6c48d97b2; calls mc_001_extract_claims_0005; spans ev_20a7facf5fd73dd9
- **Excerpt** (`ev_20a7facf5fd73dd9`, 4 name(s) matched):
  > No MpFAD7 (FAD11) Mp5g20700 FAD7/FAD8 Chloroplast (confirmed) n3-desaturation of HDA (16:2n6) → HTA (16:3n3) and LA (18:2n6) → ALA (18:3n3) Genetic: Mp fad7 depleted in HTA/ALA and canonical jasmonates [ 14 ].
- **Participants:** substrate HDA (16:2n6): ev_20a7facf5fd73dd9; product HTA (16:3n3): ev_20a7facf5fd73dd9
- **Seed connection:** upstream C16 precursor: makes HTA, the C16 precursor of the dn-OPDA branch. Graph: shared_substrate via 'HDA (16:2n6)' with `rx_eab7ca60441693867cca`

#### 34. `rx_14eb3a43a495d7ce851f`: OPDA -> OPC-8:0

- **Graph:** component `cmp_f63e6cc91e7b`, `species:inferred_only`. Decision `candidate` (`dc_70cf3e592aca502b`): "species state 'inferred_only' is not target-species-supported (species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** OPR3 (`model_citation`). **Gene:** none. **Species:** `inferred_only`, Arabidopsis thaliana (`model_citation`). Reason: "species support is an inference, not an observation (ortholog); an ortholog or analogy never transfers a reaction between species"
- **Marker sentence** (ortholog):
  > Marchantia lacks JAR1 orthologs and ligand activation is achieved via a cis -to- iso isomerization by a putative isomerase.
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_0bc93d248b7f31db; calls mc_001_extract_claims_0006; spans ev_c98d84c7d84855c1
- **Excerpt** (`ev_c98d84c7d84855c1`, 2 name(s) matched):
  > In Arabidopsis, OPDA and dn- cis -OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0, respectively.
- **Participants:** substrate OPDA: ev_c98d84c7d84855c1; product OPC-8:0: ev_c98d84c7d84855c1
- **Seed connection:** DIRECT (consumes seed 12-OPDA): OPR3 reduces OPDA to OPC-8:0, the canonical Arabidopsis step. Graph: product_to_substrate via 'OPDA' with `rx_8ffac9975e3a977a75f0`

#### 35. `rx_eab7ca60441693867cca`: HDA (16:2n6) -> HTA (16:3n3)

- **Graph:** component `cmp_fe6e08218f26`, `species:species_not_stated`. Decision `candidate` (`dc_41c34e2cf2c77116`): "species state 'species_not_stated' is not target-species-supported (no species is stated anywhere in the supporting evidence); retained as a labelled lead, not as an established target-species reaction"
- **Enzyme:** none. **Gene:** none. **Species:** `species_not_stated`, (empty). Reason: "no species is stated anywhere in the supporting evidence"
- **Paper:** `PMC:PMC13364333`, "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond"; claims cl_bbe6e30f3c568977; calls mc_001_extract_claims_0007; spans ev_ba1b08899518f20e
- **Excerpt** (`ev_ba1b08899518f20e`, 3 name(s) matched):
  > Note that MpFAD7 and MpFAD5 (see Section 2.1 ) are both plastidial enzymes but act at distinct positions on the desaturation cascade; MpFAD5 is a Δ 7 -desaturase introducing the first double bond into 16:0, while MpFAD7 is the terminal n3-FAD converting HDA (16:2n6) and LA (18:2n6) to HTA and ALA, respectively.
- **Participants:** substrate HDA (16:2n6): uncited; product HTA (16:3n3): uncited
- **Seed connection:** upstream C16 precursor: HDA -> HTA again, no species. Graph: shared_substrate via 'HDA (16:2n6)' with `rx_3dc94e3847d88fa54c12`

## 3. Every included paper and its route

| paper id | PMID | title | source | route | recorded reason | title organisms (frozen detector) | text | calls / passages / claims |
|---|---|---|---|---|---|---|---|---|
| `PMC:PMC8508710` | 34638573 | Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana. | ncbi | mining_rescue route 2 | mining_rescue: names 'a. thaliana' and '12-OPDA' in one sentence -- production screen: off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | none | full text 88009 chars; opens with its title: True | 4 / 20 / 3 |
| `PMC:PMC13364333` | - | Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond. | europepmc | production | score 5.0 >= threshold 2.0; anchored on jasmonate biosynthesis, jasmonate biosynthetic pathway, jasmonate biosynthetic (local: precursors) | none | full text 98172 chars; opens with its title: True | 4 / 20 / 24 |
| `PMC:PMC12096059` | 40095511 | Arabidopsis GH3.10 conjugates jasmonates. | ncbi | production | score 4.75 >= threshold 2.0; anchored on jasmonate biosynthesis (local: knockout) | none | full text 50054 chars; opens with its title: True | 4 / 16 / 8 |
| `PMC:PMC12787732` | - | Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves. | europepmc | production | score 4.0 >= threshold 2.0; anchored on jasmonate biosynthesis (local: catalyzes) | none | full text 88532 chars; opens with its title: True | 4 / 17 / 3 |

Routes: 3 production and 1 route-2 rescue (`PMC:PMC8508710`, the same 13-LOX paper and the same matched term `12-OPDA` as in pilot-002). No route-1 rescue fired, and no grape or other host paper was included. All 4 are full texts; pilot-002 read 5 abstract-only papers. The evidence type is `unknown` for all 4, including the review.

## 4. Attribution methods and participants (IT2-N1)

Fields kept, by attribution (38 extracted claims): `{"compartment:model_citation": 18, "direction:model_citation": 24, "enzyme:model_citation": 24, "gene:model_citation": 14, "species:model_citation": 30}`. There were **0 uncited-field drops**, and `verbatim_cited_span` was used **0** times, so no N-1 row pairing needs checking. Every filled field in the raw completions came with its own citation list: `{"compartment:own list present": 18, "enzyme:own list present": 24, "gene:own list present": 14, "species:own list present": 30}`.

Per call (raw completion): claims, participants, participants with their own `evidence_ids`, and which name fields were filled.

| call | paper | claims | participants cited / total | fields filled |
|---|---|---|---|---|
| `mc_001_extract_claims_0001` | `PMC:PMC8508710` | 1 | 0 / 2 | - |
| `mc_001_extract_claims_0002` | `PMC:PMC8508710` | 1 | 0 / 2 | - |
| `mc_001_extract_claims_0003` | `PMC:PMC8508710` | 1 | 0 / 3 | - |
| `mc_001_extract_claims_0004` | `PMC:PMC8508710` | 0 | 0 / 0 | - |
| `mc_001_extract_claims_0005` | `PMC:PMC13364333` | 15 | 30 / 30 | compartment, enzyme, gene, species |
| `mc_001_extract_claims_0006` | `PMC:PMC13364333` | 7 | 16 / 16 | compartment, enzyme, species |
| `mc_001_extract_claims_0007` | `PMC:PMC13364333` | 1 | 0 / 2 | - |
| `mc_001_extract_claims_0008` | `PMC:PMC13364333` | 1 | 0 / 2 | - |
| `mc_001_extract_claims_0009` | `PMC:PMC12096059` | 5 | 10 / 10 | enzyme, gene, species |
| `mc_001_extract_claims_0010` | `PMC:PMC12096059` | 1 | 2 / 2 | enzyme, species |
| `mc_001_extract_claims_0011` | `PMC:PMC12096059` | 2 | 8 / 8 | enzyme, species |
| `mc_001_extract_claims_0012` | `PMC:PMC12096059` | 0 | 0 / 0 | - |
| `mc_001_extract_claims_0013` | `PMC:PMC12787732` | 1 | 0 / 2 | - |
| `mc_001_extract_claims_0014` | `PMC:PMC12787732` | 2 | 0 / 2 | - |
| `mc_001_extract_claims_0015` | `PMC:PMC12787732` | 0 | 0 / 0 | - |
| `mc_001_extract_claims_0016` | `PMC:PMC12787732` | 0 | 0 / 0 | - |

Per candidate: 29 of 35 have every participant cited; 6 have none: `rx_2889b990927302948b2e`, `rx_41c97e790d2e17c19477`, `rx_6570fd856f1200fa7a05`, `rx_ce92a2647a229cc6528e`, `rx_eba8e16b5409af0ff75a`, `rx_eab7ca60441693867cca`. Those six came from calls _0001, _0002, _0003, _0007, _0008 and _0013, and the two incomplete claims (from _0014) have no participant citations either. `build_claims` never drops an uncited participant; the claim-level `evidence_ids` still name the span. So IT2-N1 holds whenever the model answered in full, and fails silently when it did not.

## 5. Model calls and batching (a)

There were 16 logical calls, 1 ladder rung each, and 18 provider requests. Calls per paper: `{"PMC:PMC8508710": 4, "PMC:PMC13364333": 4, "PMC:PMC12096059": 4, "PMC:PMC12787732": 4}`. The batch size was 5 passages (a paper's last batch holds the remainder: 1 and 2). `max_tokens` was 12000. In the table, 'answer out' is the output tokens of the request that answered; for `_0008` that is the total less 2 x 12,000 for its two empty requests. 'Est. reasoning' subtracts the JSON length / 3.5 chars per token. It is an estimate, because the provider reports only the total.

| call | paper | batch | passages | chars sent | requests (finish/content) | input tok | output tok | answer out | % of 12k | response chars | est. reasoning | ~0 reasoning | latency s | claims |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mc_001_extract_claims_0001` | `PMC:PMC8508710` | 1 | 5 | 20617 | stop/445ch | 6309 | 156 | 156 | 1% | 445 | 29 | **yes** | 8.3 | 1 |
| `mc_001_extract_claims_0002` | `PMC:PMC8508710` | 2 | 5 | 23086 | stop/492ch | 6831 | 5471 | 5471 | 46% | 492 | 5330 | no | 102.1 | 1 |
| `mc_001_extract_claims_0003` | `PMC:PMC8508710` | 3 | 5 | 17528 | stop/616ch | 5719 | 209 | 209 | 2% | 616 | 33 | **yes** | 4.8 | 1 |
| `mc_001_extract_claims_0004` | `PMC:PMC8508710` | 4 | 5 | 12560 | stop/13ch | 4701 | 1068 | 1068 | 9% | 13 | 1064 | no | 11.4 | 0 |
| `mc_001_extract_claims_0005` | `PMC:PMC13364333` | 1 | 5 | 24561 | stop/10793ch | 7023 | 11196 | 11196 | 93% | 10793 | 8112 | no | 100.1 | 15 |
| `mc_001_extract_claims_0006` | `PMC:PMC13364333` | 2 | 5 | 22012 | stop/5091ch | 6632 | 11650 | 11650 | 97% | 5091 | 10195 | no | 106.7 | 7 |
| `mc_001_extract_claims_0007` | `PMC:PMC13364333` | 3 | 5 | 23876 | stop/270ch | 6387 | 10136 | 10136 | 84% | 270 | 10059 | no | 152.2 | 1 |
| `mc_001_extract_claims_0008` | `PMC:PMC13364333` | 4 | 5 | 22473 | length/0ch, length/0ch, stop/294ch | 19382 | 24122 | 122 | 1% | 294 | 38 | **yes** | 313.4 | 1 |
| `mc_001_extract_claims_0009` | `PMC:PMC12096059` | 1 | 5 | 18873 | stop/2762ch | 6868 | 895 | 895 | 8% | 2762 | 106 | no | 11.5 | 5 |
| `mc_001_extract_claims_0010` | `PMC:PMC12096059` | 2 | 5 | 23955 | stop/556ch | 7633 | 4247 | 4247 | 35% | 556 | 4088 | no | 68.5 | 1 |
| `mc_001_extract_claims_0011` | `PMC:PMC12096059` | 3 | 5 | 18999 | stop/2132ch | 6944 | 6730 | 6730 | 56% | 2132 | 6121 | no | 64.0 | 2 |
| `mc_001_extract_claims_0012` | `PMC:PMC12096059` | 4 | 1 | 724 | stop/14ch | 783 | 461 | 461 | 4% | 14 | 457 | no | 8.6 | 0 |
| `mc_001_extract_claims_0013` | `PMC:PMC12787732` | 1 | 5 | 20186 | stop/275ch | 5519 | 5879 | 5879 | 49% | 275 | 5800 | no | 101.7 | 1 |
| `mc_001_extract_claims_0014` | `PMC:PMC12787732` | 2 | 5 | 23333 | stop/474ch | 6442 | 154 | 154 | 1% | 474 | 19 | **yes** | 4.5 | 2 |
| `mc_001_extract_claims_0015` | `PMC:PMC12787732` | 3 | 5 | 12960 | stop/14ch | 3895 | 779 | 779 | 6% | 14 | 775 | no | 15.7 | 0 |
| `mc_001_extract_claims_0016` | `PMC:PMC12787732` | 4 | 2 | 7654 | stop/13ch | 3105 | 649 | 649 | 5% | 13 | 645 | no | 7.6 | 0 |

**Near the 12k budget.**
- `mc_001_extract_claims_0005` used 93%.
- `mc_001_extract_claims_0006` used 97%.
- `mc_001_extract_claims_0007` used 84%.
- `_0008` spent the whole budget twice: two requests ended `length` with 0 characters. That is 2 empty requests in the run (pilot-002 had 21).

All four are 5-passage batches of the Marchantia review. For that paper, one 12k call per 5 passages sits at the edge.

**Provider identity.** It is not recorded. `provider` is `""` on all 16 records and is never set: `ModelCallRecord.provider` exists, but `modelclient` does not fill it. `actual_model` is the resolved *requested* slug from `chat_detailed` diagnostics, not the response's model field. The per-request `provider_requests` rows carry no provider either.

**'Stop' answers that look non-reasoning.** Each of these ended `finish_reason=stop`:

- `mc_001_extract_claims_0001`: 156 output tokens for 445 chars (est. reasoning 29), 8.3 s. Raw text: `{ "claims": [ { "evidence_ids": ["ev_5863efff93736770", "ev_06e5eb88be1f278b"], "hypothesis": false, "products": [ { "name": "13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT)", "stoichiometry": "1" } ], "substrates":`
- `mc_001_extract_claims_0003`: 209 output tokens for 616 chars (est. reasoning 33), 4.8 s. Raw text: `{ "claims": [ { "evidence_ids": [ "ev_6915f534ca2dfd74", "ev_bb4c9802d1637278" ], "hypothesis": false, "products": [ { "name": "13(S)-hydroperoxy-9(Z),11(E),15(Z)-octadecatrienoic acid (13-HPOT)", "stoichiometry": "has_stoich`
- `mc_001_extract_claims_0008`: 122 output tokens for 294 chars (est. reasoning 38), 313.4 s. Raw text: `{ "claims": [ { "evidence_ids": ["ev_6f4dc3ffa5db1e0c", "ev_74d8b3d259ca310a"], "hypothesis": false, "products": [{"name": "13-HPOT", "stoichiometry": "1"}], "substrates": [{"name": "ALA (18:3n3)", "stoichiometry": "1"}], "tissue": "chloroplast" } ] }`
- `mc_001_extract_claims_0014`: 154 output tokens for 474 chars (est. reasoning 19), 4.5 s. Raw text: `{ "claims": [ { "evidence_ids": ["ev_b48f934b61eb21a4"], "hypothesis": false, "products": [{"name": "JA", "stoichiometry": "products_evidence_ids_placeholder"}], "substrates": [], "tissue": "leaf" }, { "evidence_ids": ["ev_b48f934b61eb21a4"],`

The other twelve spent from about 100 (`_0009`, 11.5 s, borderline, yet it filled every field) to about 10,200 estimated reasoning tokens. Even `{"claims":[]}` answers cost 461-1,068 output tokens. So these four look like a provider or mode that did not reason; with no provider recorded, that is an inference from the token counts and latency. What they produced:
- `_0001` and `_0003`: bare 13-LOX claims. `_0003` also writes the LOX -> AOS -> AOC cascade as one reaction.
- `_0014`: two claims with no substrate.
- `_0008`: ALA -> 13-HPOT with nothing else.

All four have no enzyme or species field and no participant citations. Two also wrote schema-shaped placeholders into `stoichiometry`:
- `"has_stoichiometry"` on all 3 participants of `_0003`;
- `"products_evidence_ids_placeholder"` on both products of `_0014`.

All 5 are stored as is in `claims.jsonl`, because nothing validates a stoichiometry value. A thin answer is `ok` to the client, so no rung is retried.

## 6. Search

| # | query | attempt | window | results | new discovered (by source) | new included | cumulative |
|---|---|---|---|---|---|---|---|
| 1 | "12-OPDA" AND "Arabidopsis thaliana" | 1 | 25 | 25 | 25 {"europepmc": 15, "ncbi": 10} | PMC:PMC8508710 | 25 |
| 2 | "AOS" AND "Arabidopsis thaliana" | 1 | 25 | 25 | 21 {"ncbi": 11, "europepmc": 10} | PMC:PMC13364333, PMC:PMC12096059 | 46 |
| 3 | "jasmonate biosynthesis" AND "Arabidopsis thaliana" | 1 | 25 | 25 | 18 {"europepmc": 11, "ncbi": 7} | PMC:PMC12787732 | 64 |

There was one round of 3 seed queries, each a full 25-result window, and 64 papers. `max_papers_discovered` (150) was never approached, so the IT2-3 budget-cut rule was not exercised; no item is exhausted. `cache_hit` is true on 0 rows. The round metrics say `papers_discovered` 4` against 64 in usage (pilot-001 A5 / F8).

## 7. Why the run stopped after round 1 (b)

**Counting.** Every claim goes through `_admit`. A claim whose identity is new to the graph adds its reaction to `added_reaction_ids` if admitted, and to `added_candidate_reaction_ids` if quarantined:

`src/t2pw/mining/controller.py:691-697`
```python
        if existing is None:
            if outcome.admitted:
                delta.added_reaction_ids.append(stored_reaction.reaction_id)
            else:
                delta.added_candidate_reaction_ids.append(stored_reaction.reaction_id)
        else:
            delta.updated_reaction_ids.append(stored_reaction.reaction_id)
```

When the round closes, their sum becomes `last_round_new_reactions`:

`src/t2pw/mining/controller.py:552-572`
```python
        self.ledger.record_round(
            useful=delta.useful,
            new_reactions=len(delta.added_reaction_ids) + len(delta.added_candidate_reaction_ids),
        )
        self.state.errors.extend(round_errors)

        next_frontier = frontier_mod.generate(
            graph=self.state.graph,
            store=self.state.store,
            delta=delta,
            config=self.config,
            round_index=index,
            previous=self.state.frontier,
            exhausted_ids=set(self.state.exhausted_frontier),
        )
        self.state.frontier = frontier_mod.merge_frontier(
            [i for i in self.state.frontier if not i.exhausted],
            next_frontier,
            cap=self.config.budgets.max_frontier_size,
        )

```
`src/t2pw/mining/budget.py:155-159`
```python
    def record_round(self, *, useful: bool, new_reactions: int = 0) -> None:
        self.usage.rounds += 1
        self.usage.last_round_new_reactions = max(0, int(new_reactions))
        self.usage.unproductive_streak = 0 if useful else self.usage.unproductive_streak + 1
        self.refresh()
```

In pilot-003, `graph_delta.json` round 1 has `added_reaction_ids` 0, `added_candidate_reaction_ids` 35 and `updated_reaction_ids` 1 (the dn-cis-OPDA isomerisation, found in two batches). So `last_round_new_reactions` = 35. By paper: `{"PMC:PMC8508710": 3, "PMC:PMC13364333": 23, "PMC:PMC12096059": 8, "PMC:PMC12787732": 1}`. By species state: `{"multiple_species": 2, "inferred_only": 17, "in_vitro_or_cell_free": 15, "species_not_stated": 1}`. Not one of the 35 could have been admitted.

**Stopping.** The limit is tested only in `check_before_round`, the gate at the top of the round loop:

`src/t2pw/mining/controller.py:319-333`
```python
            while True:
                blocked = self.ledger.check_before_round(
                    primary_reactions=len(self.state.graph.primary_reactions),
                    candidate_reactions=len(self.state.graph.candidate_reactions),
                    candidate_components=len(self.state.graph.candidate_components),
                )
                if blocked:
                    return self._finish(STATE_STOPPED)
                if not self.state.frontier:
                    self.ledger.stop(
                        STOP_FRONTIER_EXHAUSTED, "no unexhausted frontier items remain"
                    )
                    return self._finish(STATE_STOPPED)
                self._round_index += 1
                self._run_round(self._round_index)
```
`src/t2pw/mining/budget.py:193-225` and `src/t2pw/mining/budget.py:187-191`
```python
    def check_before_round(
        self,
        *,
        primary_reactions: int = 0,
        candidate_reactions: int = 0,
        candidate_components: int = 0,
    ) -> Optional[str]:
        """The stop reason that forbids starting another round, or ``None``."""
        self.refresh()
        if self._cancelled:
            self.stop(STOP_CANCELLED, "cancellation requested")
            return STOP_CANCELLED
        b = self.budgets
        for reason in (
            self._limit(STOP_MAX_ROUNDS, self.usage.rounds, b.max_rounds, "rounds"),
            self._limit(STOP_MAX_WALL_CLOCK, self.elapsed(), b.max_wall_clock_seconds, "seconds"),
            self._limit(STOP_MAX_LLM_CALLS, self.usage.llm_calls, b.max_llm_calls, "llm calls"),
            self._limit(STOP_MAX_INPUT_TOKENS, self.usage.input_tokens, b.max_input_tokens, "input tokens"),
            self._limit(STOP_MAX_OUTPUT_TOKENS, self.usage.output_tokens, b.max_output_tokens, "output tokens"),
            self._limit(STOP_MAX_COST, self.usage.estimated_cost_usd, b.max_cost_usd, "estimated usd"),
            self._limit(STOP_MAX_PAPERS_PROCESSED, self.usage.papers_processed, b.max_papers_processed, "papers processed"),
            self._limit(STOP_MAX_PRIMARY_REACTIONS, primary_reactions, b.max_primary_reactions, "primary reactions"),
            self._limit(STOP_MAX_CANDIDATE_REACTIONS, candidate_reactions, b.max_candidate_reactions, "candidate reactions"),
            self._limit(STOP_MAX_CANDIDATE_COMPONENTS, candidate_components, b.max_candidate_components, "candidate components"),
            self._limit(
                STOP_MAX_NEW_REACTIONS_PER_ROUND,
                self.usage.last_round_new_reactions,
                b.max_new_reactions_per_round,
                f"new reactions in round {self.usage.rounds}",
            ),
        ):
            if reason:
                return reason

    def _limit(self, reason: str, used: Any, cap: Any, label: str) -> Optional[str]:
        if cap and used >= cap:
            self.stop(reason, f"{label}: {used} >= limit {cap}")
            return reason
        return None
```

After round 1, every earlier limit in that list is under its cap: 1 round of 8, 16 calls of 200, 4 papers of 40, 0 primary, 35 candidate reactions of 300, 21 components of 40. Then 35 >= 25 records `max_new_reactions_per_round_reached` with "new reactions in round 1: 35 >= limit 25". Round 2 never starts. Nothing inside a round checks the limit, so round 1 ran to the end: that is the documented behaviour (`limits_and_stopping.md`: 'That round is complete and kept; the next one does not start'). But the counter counts *lexical* identities, quarantined non-target leads included. By the auditor's reading, 35 nodes are 28 distinct conversions:

- 3 nodes are 13-LOX.
- 3 are AOC to OPDA.
- 2 are JA -> JA-Ile.
- 2 are 12-OH-JA -> 12-OH-JA-Ile.
- 2 are HDA -> HTA.

That is still over 25. One bryophyte review alone gives 23. Any run whose first round reads a dense review of another organism will stop here. It is the same design question pilot-001 raised as NB-1: stop, or throttle?

**What was on the frontier and never searched.** `frontier.jsonl` holds 60 items, `{"bridge_search": 19, "seed_expansion": 3, "downstream_expansion": 27, "upstream_expansion": 11}`:

- The 3 seeds were asked once each, attempt 1, in round 1. Their attempt-2 formulations (plus the topic) and wider windows never ran.
- The other 57 were never asked: 19 bridges, 27 downstream and 11 upstream items.

The frontier is not the whole story. Replaying the frozen `frontier.generate` on `state.json` with the cap lifted gives **194** items, `{"downstream_expansion": 27, "upstream_expansion": 19, "missing_species_evidence": 33, "branch_reaction": 53, "enzyme_reactions": 13, "missing_direction": 12, "missing_compartment": 17, "cofactor_investigation": 1, "bridge_search (recorded)": 19}`. The 60-item cap kept everything at priority >= 6.5 and silently dropped **137**: `{"upstream_expansion": 8, "missing_species_evidence": 33, "branch_reaction": 53, "enzyme_reactions": 13, "missing_direction": 12, "missing_compartment": 17, "cofactor_investigation": 1}`. The dropped items include:

- all 33 `missing_species_evidence` items, the only kind that asks for target-species evidence for a held lead;
- all 13 `enzyme_reactions` items (AOS, AOC, OPR3, GH3.10 ...).

Validation: the recorded non-bridge items are exactly the 38 top-ranked non-bridge items of the replay, the ones that fit after the 3 seeds and 19 bridges (True). Bridges cannot be replayed exactly: a bridge anchors on the first name in component order, which depends on memory order (pilot-001 A10, still present). This process's replay produced 19 bridges, and 11 recorded bridge targets do not appear in it. Separate replay processes even produced different bridge counts: anchor collisions follow set order. Pilot-001's F4 therefore reproduces at twice its size: 62 cut then, 137 now. No stop reason or artifact names the cut.

## 8. Identity (c): is IT2-2 working live?

NCBI records: `{"total": 28, "pmcid_resolved": 19, "pmid_only": 9, "id_consistent": 28, "url_is_pubmed_of_own_pmid": 28, "with_doi": 0}`.
- Every NCBI `paper_id` equals `PMC:<pmcid>` when EuropePMC returned a PMCID, and `PMID:<pmid>` otherwise.
- Every URL is `https://pubmed.ncbi.nlm.nih.gov/<own pmid>/`.
- No NCBI record carries a DOI, so the parser's cited-reference DOI is gone too.

| check | result |
|---|---|
| `PMC:PMC8508710` (PMID 34638573): full text opens with its own title / PMID in the first 600 chars | True / True. Head: "pmc Int J Mol Sci Int J Mol Sci 808 ijms 101092791 ijms International Journal of Molecular Sciences 1422-0067 Multidisciplinary Digital Publishing Institute (MDPI) PMC850..." |
| `PMC:PMC13364333` (PMID -): full text opens with its own title / PMID in the first 600 chars | True / n/a. Head: "2909 plants Plants Plants (Basel) Multidisciplinary Digital Publishing Institute (MDPI) PMC13364333 13364333 13364333 42452298 10.3390/plants15132102 Jasmonate Biosynthes..." |
| `PMC:PMC12096059` (PMID 40095511): full text opens with its own title / PMID in the first 600 chars | True / True. Head: "pmc Plant Biol (Stuttg) Plant Biol (Stuttg) 379 blackwellopen 101148926 PLB Plant Biology (Stuttgart, Germany) 1435-8603 1438-8677 pmc-is-collection-domain yes pmc-collec..." |
| `PMC:PMC12787732` (PMID -): full text opens with its own title / PMID in the first 600 chars | True / n/a. Head: "2909 plants Plants Plants (Basel) Multidisciplinary Digital Publishing Institute (MDPI) PMC12787732 12787732 12787732 41514989 10.3390/plants15010045 Transcription, Alter..." |
| NCBI id vs EuropePMC's own id for the same article (pilot-002): Genetic Transformation of Triticum dicoccum and Triticum aestivum with Genes of Jasmonate Biosynthes | PMC:PMC11478715 vs PMC:PMC11478715: **agree** |
| NCBI id vs EuropePMC's own id for the same article (pilot-002): Size of chloroplasts in Arabidopsis mesophyll cells affects jasmonate biosynthesis. | PMC:PMC13358708 vs PMC:PMC13358708: **agree** |
| NCBI id vs EuropePMC's own id for the same article (pilot-002): Arabidopsis Transcriptomics Reveals the Role of Lipoxygenase2 (AtLOX2) in Wound-Induced Responses. | PMC:PMC11173247 vs PMC:PMC11173247: **agree** |
| 'full text refused' notes | 0 (none; both NCBI full texts matched) |
| duplicate titles among the 64 papers | 0 (pilot-002: 6) |

The same title under different ids in pilot-002 and pilot-003 happens 20 times; 43 titles have the same id in both. In every change, pilot-002's side is an NCBI record carrying a cited reference's PMCID:

| title | pilot-002 id (source) | pilot-003 id (source) |
|---|---|---|
| Maize OPR2 and LOX10 Mediate Defense against Fall Armyworm and Western Corn Rootworm by Tissue-Specific Regula | `PMC:PMC9505089` (ncbi) | `PMC:PMC10530937` (ncbi) |
| The AtMYB60 transcription factor regulates stomatal opening by modulating oxylipin synthesis in guard cells. | `PMC:PMC4970199` (ncbi) | `PMC:PMC8752683` (ncbi) |
| Multi-omics analysis of early leaf development in Arabidopsis thaliana. | `PMC:PMC4558650` (ncbi) | `PMC:PMC8085607` (ncbi) |
| The In Vitro Interaction of 12-Oxophytodienoic Acid and Related Conjugated Carbonyl Compounds with Thiol Antio | `PMC:PMC4911631` (ncbi) | `PMC:PMC8003295` (ncbi) |
| Drought stress modulates oxylipin signature by eliciting 12-OPDA as a potent regulator of stomatal aperture. | `PMC:PMC3938610` (ncbi) | `PMC:PMC4091591` (ncbi) |
| Functional convergence of oxylipin and abscisic acid pathways controls stomatal closure in response to drought | `PMC:PMC2039767` (ncbi) | `PMID:24429214` (ncbi) |
| The plant endophytic fungus Cyanodermella asteris produces the phytohormone jasmonic acid. | `PMC:PMC6164985` (ncbi) | `PMC:PMC13063583` (europepmc) |
| NAD&lt;sup&gt;+&lt;/sup&gt; deficiency primes defense metabolism via &lt;sup&gt;1&lt;/sup&gt;O&lt;sub&gt;2&lt; | `PMC:PMC6628814` (ncbi) | `PMC:PMC11300881` (europepmc) |
| Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana. | `PMC:PMC411572` (ncbi) | `PMC:PMC8508710` (ncbi) |
| Characterization of wound-induced electrical signals and reactive oxygen species in chickpea (Cicer arietinum) | `PMC:PMC8397809` (ncbi) | `PMC:PMC12584604` (ncbi) |
| GhAOS integrates circadian rhythm and jasmonic acid signaling to regulate high temperature stress responses in | `PMC:PMC8252431` (ncbi) | `PMC:PMC12506428` (ncbi) |
| Screening of MYB1R1 interaction with LDOX promoter to regulate anthocyanin biosynthesis in peaches. | `PMC:PMC4651307` (ncbi) | `PMC:PMC12422263` (ncbi) |
| An extragenic second-site mutation in the jar1-1 mutant suppresses the response to photoperiod stress independ | `PMC:PMC9806556` (ncbi) | `PMC:PMC12206676` (ncbi) |
| Hydrogen-Sulfide-Mediated PpAOS3-JA Module Provides Insight into Salt Stress Resistance in Peach. | `PMC:PMC3129672` (ncbi) | `PMC:PMC12114830` (ncbi) |
| CORONATINE INSENSITIVE 1-mediated repression of immunity-related genes in Arabidopsis roots is lifted upon inf | `PMC:PMC523327` (ncbi) | `PMC:PMC12116175` (ncbi) |
| Transcription factor ABF3 modulates salinity stress-enhanced jasmonate signaling in Arabidopsis. | `PMC:PMC3145709` (ncbi) | `PMC:PMC11726054` (ncbi) |
| Endogenous Hormone Levels and Transcriptomic Analysis Reveal the Mechanisms of Bulbil Initiation in Pinellia t | `PMC:PMC9133075` (ncbi) | `PMC:PMC11173086` (ncbi) |
| Arabidopsis GH3.10 conjugates jasmonates. | `PMC:PMC4793799` (ncbi) | `PMC:PMC12096059` (ncbi) |
| Interaction with CYP20-3 limits OPDA flux into jasmonate biosynthesis in wounded leaves of arabidopsis grown a | `PMC:PMC2716216` (ncbi) | `PMC:PMC13096701` (europepmc) |
| Rapid local and systemic jasmonate signalling drives the initiation and establishment of plant systemic immuni | `PMC:PMC3855844` (ncbi) | `PMC:PMC12830360` (ncbi) |

Three of these articles (NAD+ deficiency, CYP20-3 / OPDA flux, Cyanodermella) appeared twice in pilot-002. In pilot-003 they appear once, under EuropePMC's id. IT2-2 behaves as documented. One caveat: a PMCID that EuropePMC resolves is trusted as given. The 19 resolved ids can only be spot-checked; the 5 that can be checked (2 by full text, 3 against EuropePMC's own records) all agree.

## 9. Three-way side by side (e)

All three columns are computed by one function from each run directory.

| | pilot-001 | pilot-002 | pilot-003 |
|---|---|---|---|
| code | 23e1ae83 | ecf6b9a0 | f9556288 |
| stop_reason | frontier_exhausted | max_papers_discovered_reached | max_new_reactions_per_round_reached |
| rounds | 2 | 3 | 1 |
| elapsed s | 303.704 | 2039.18 | 1115.336 |
| queries (rows) | 63 | 58 | 3 |
| papers discovered (distinct titles) | 21 (21) | 150 (144) | 64 (64) |
| NCBI papers discovered | 0 | 83 | 28 |
| included: production / route 1 / route 2 | 1: 1 / 0 / 0 | 11: 5 / 5 / 1 | 4: 3 / 0 / 1 |
| processed | 1 | 10 | 4 |
| passages | 20 | 80 | 73 |
| logical model calls (by status) | 1 {"ok": 1} | 10 {"empty_completion": 2, "ok": 8} | 16 {"ok": 16} |
| ladder rungs (usage.llm_calls) | 1 | 15 | 16 |
| max passages per call / max_tokens | 20 / 4000 | 20 / 4000 | 5 / 12000 |
| claims extracted -> identities | 21 -> 19 | 16 -> 11 | 38 -> 37 |
| primary / candidate reactions / components | 0 / 19 / 14 | 1 / 10 / 3 | 0 / 35 / 21 |
| species states (nodes) | {"in_vitro_or_cell_free": 1, "inferred_only": 10, "species_not_stated": 8} | {"in_vitro_or_cell_free": 9, "species_not_stated": 1, "target_species_supported": 1} | {"in_vitro_or_cell_free": 15, "inferred_only": 17, "multiple_species": 2, "species_not_stated": 1} |
| uncited field drops | 59 | 0 | 0 |
| fields: model_citation / verbatim_cited_span | 0 / 0 | 53 / 0 | 110 / 0 |
| input / output tokens | 22843 / 3629 | 302829 / 95417 | 104173 / 83802 |
| est. cost USD | 0.003373 | 0.058908 | 0.035558 |
| evidence type of included | {"unknown": 1} | {"unknown": 11} | {"unknown": 4} |
| provider requests | 1 | 30 | 18 |
| empty requests (finish_reason=length) | 0 | 21 | 2 |
| frontier cut silently by the 60 cap | 62 | 0 (44 and 19, under the cap) | 137 |
| wrong NCBI ids / duplicate titles | - (no NCBI) / 0 | 7+ proven / 6 | 0 of 5 checked / 0 |

## 10. What still reproduces (f), and new anomalies

| id | finding | pilot-003 |
|---|---|---|
| F1 | every enzyme/gene/species dropped as uncited | **Fixed (since pilot-002).** 0 drops; all 110 kept fields `model_citation`. |
| F2 | round 2 found nothing: OR query, top-10, NCBI never reached, one-shot exhaustion | **Not exercised.** Only round 1 ran (3 seed queries, 64 papers, 28 from NCBI). |
| F3 | species labels by fall-through / keyword | **Reproduces.** 16 of 17 `inferred_only` labels come from the substring 'ortholog' ('Arabidopsis Ortholog' table header, 'lacks JAR1 orthologs', 'orthology analysis'), including the A. thaliana OPR3 step. 13-LOX is labelled from 'as inferred from the dose-response curves'. The pepper intro gives `in_vitro`. Two `multiple_species` come from paper metadata. |
| F4 | frontier cap silently cut 62 items | **Reproduces, larger.** 137 of 194 cut, unrecorded; every `missing_species_evidence` and `enzyme_reactions` item is gone. |
| F5 | screen missed key Arabidopsis enzymology | Not evaluated beyond round 1. Route 2 again brought in the 13-LOX paper. |
| F6 | review typed `unknown`, DOI empty | **Reproduces.** 4/4 included `unknown`, including the Marchantia review; `doi` is empty on 4/4. |
| F7 | report.md misstates numbers | **Reproduces.** Claims extracted: 37 across 37 distinct reaction identities / Primary graph: 0 reactions, 26 chemical connections, 67 entities / Model calls: 16 (38 extracted; the primary graph is empty; 18 provider requests). It also says 'generated but never asked' of the 3 seeds that were asked. |
| F8 | cache_hit never set; round provenance included-only; relevance 0.0; order-dependent bridges | **Reproduces.** `cache_hit` false 3/3; round `papers_discovered` 4 vs 64; relevance {0.0: 21}; bridge anchors differ on replay (11 of 19 in this build's process; the count varies between processes). |
| F9 | CLI does not load .env | **Still present; not exercised.** The launcher was used again, and `flags.py` is unchanged. |
| F10 | cli.md flag order | **Partly fixed.** `cli.md:109` still puts `--enable-pathway-mining` after `preflight`. |
| A1 | NCBI ids from the cited reference | **Fixed, verified live** (section 8). |
| A2 | reasoning exhausts max_tokens; the ladder cannot help | **Fixed in effect.** 16/16 calls `ok` on rung 1. 2 empty 12k requests remain (`_0008`), and the review's batches use 84-97% of the budget. |
| A3 | species states from keyword tests | **Reproduces** (F3 row). |
| A4 | abstract-only papers indexed twice | **Not exercised.** All 4 papers are full text; 0 pairs of spans share identical text. |
| A5 | phantom round after a mid-round stop | **Not exercised.** The stop came from `check_before_round`; `check_before_round` still ignores discovered papers and an existing stop reason. |
| A6 | discovery budget exhausts items on cut windows | **Fixed in code (IT2-3); not exercised** (64 of 150). |
| A7 | paper retrieval is a store-wide top-60 filtered per paper | **Still present.** The two late papers got 16 and 17 passages, not 20 (`retrieval.py` unchanged). |
| A8 | grape false rescue | **Fixed in code** (`_EXTRA_HOSTS` gains grape, vitis); no route-1 rescue fired. |
| A9 | lexical entity split | **Reproduces, worse.** 13-HPOT under 4 names, OPDA under 3 (OPDA, (9S,13S)-OPDA, cis-OPDA). Most of the 21 components are split this way; 35 nodes are about 28 conversions. |
| A10 | query-id collisions | **Not exercised** (3 queries). |
| A11 | nested parenthetical kept in query terms | **Not exercised.** |
| A12 | report.md misstatements | **Reproduces** (F7 row). |
| A13 | lineage / round-file observability | **Partly fixed.** `model_raw` now has fraction, passages, chars and per-request rows, and `attempt_log` is appended per rung. Still: no provider, `input_frontier` written after the round (seeds show `attempts: 1`), `cache_hit` never set. |
| A14 | evidence type / DOI missing | **Reproduces** (F6 row). |
| A15 | participants uncited | **Partly fixed.** 29/35 candidates fully cited; 6 have none, all from thin answers (section 4). |
| A16 | doc drift | **Reproduces.** `artifacts.md` still says `mining-extract/1.0.0`; `cli.md:109` flag order. |
| A17 | the same conversion as several nodes | **Reproduces, worse.** 5 groups, 7 extra nodes, and these count toward the new-reactions stop. |

### New in pilot-003

| id | severity | finding | evidence |
|---|---|---|---|
| B1 | high | The per-round new-reaction cap counts quarantined, non-admissible leads, so one off-species review ends the run after round 1 | 35 new nodes, 0 admissible (23 Marchantia, 8 in-vitro GH3.10, 3 13-LOX with no species, 1 pepper) >= 25. Round 2 and every missing_species_evidence question were never reached. Documented behaviour, but it makes a first-round review decisive (pilot-001 NB-1). |
| B2 | high | Thin, non-reasoning answers pass as `ok` and lose every name field | 4 answers at 122-209 output tokens with no enzyme/species/participant citations, among them two of the four 13-LOX batches (the paper that gave pilot-002's only primary reaction). Two wrote placeholder strings into `stoichiometry` ("has_stoichiometry", "products_evidence_ids_placeholder"), stored as is in `claims.jsonl`. Nothing records the provider, so these answers cannot be attributed or excluded. |
| B3 | medium | Species from paper metadata when the model leaves species empty | `rx_ce92a2647a229cc6528e` and `rx_6570fd856f1200fa7a05`: `multiple_species` from the screen's observed organisms (A. thaliana + Homo sapiens, from 'human LOX homologs'), not from any cited sentence (`extraction.py` `paper_observed_organisms`). |
| B4 | medium | A collapsed cascade recorded as one reaction node | `rx_6570fd856f1200fa7a05`: α-linolenic acid (LeA) -> 13-HPOT + 12-OPDA (LOX, AOS and AOC in one node), from a thin answer (`_0003`). |
| B5 | low | Output budget headroom for dense reviews is thin | Batches of the review at 93%, 97% and 84% of 12k; one batch needed 3 requests (2 empty at 12k), 313 s. A denser 5-passage batch would fail the same way pilot-002 did. |

## Appendix A: excerpt spans, verbatim

### `ev_1f2c624a181ba5ef`

```text
pmc Plant Biol (Stuttg) Plant Biol (Stuttg) 379 blackwellopen 101148926 PLB Plant Biology (Stuttgart, Germany) 1435-8603 1438-8677 pmc-is-collection-domain yes pmc-collection-title Wiley Open Access Collection PMC12096059 PMC12096059.1 12096059 12096059 40095511 10.1111/plb.70001 PLB70001 PlaBio-2024-12-0629-RP.R1 1 Research Article Research Article Arabidopsis GH3 .10 conjugates jasmonates Arabidopsis GH3.10 catalyses 12‐hydroxy jasmonoyl‐isoleucine formation Ni, Klein, Hossbach, Feussner, Hornung, Herrfurth, Hamberg & Feussner Ni B. 1 Klein M. 1 Hossbach B. 1 Feussner K. 1 Hornung E. 1 Herrfurth C. 1 2 Hamberg M. 3 Feussner I. https://orcid.org/0000-0002-9888-7003 1 2 4 ifeussn@uni-goettingen.de 1 Department of Plant Biochemistry, Albrecht‐von‐Haller‐Institute University of Goettingen Goettingen Germany 2 Service Unit for Metabolomics and Lipidomics, Goettingen Center for Molecular Biosciences (GZMB) University of Goettingen Goettingen Germany 3 Division of Physiological Chemistry II, Department of Medical Biochemistry and Biophysics Karolinska Institutet Stockholm Sweden 4 Department of Plant Biochemistry, Goettingen Center for Molecular Biosciences (GZMB) University of Goettingen Goettingen Germany * Correspondence I. Feussner, Department of Plant Biochemistry, Albrecht‐von‐Haller‐Institute, University of Goettingen, D‐37077 Goettingen, Germany. E‐mail: ifeussn@uni-goettingen.de 17 3 2025 6 2025 27 4 489026 10.1111/plb.v27.4 476 491 03 12 2024 21 1 2025 22 05 2025 23 05 2025 23 05 2025 © 2025 The Author(s). Plant Biology published by John Wiley & Sons Ltd on behalf of German Society for Plant Sciences, Royal Botanical Society of the Netherlands. https://creativecommons.org/licenses/by/4.0/ This is an open access article under the terms of the http://creativecommons.org/licenses/by/4.0/ License, which permits use, distribution and reproduction in any medium, provided the original work is properly cited. Abstract Jasmonates regulate plant development and defence. In angiosperms, the canonical bioactive jasmonate is jasmonoyl‐isoleucine (JA‐Ile), which is formed in Arabidopsis thaliana by JAR1 and GH3.10. In contrast to other jasmonate biosynthesis or perception mutants, however, gh3.10 jar1 knockout lines are still fertile. Therefore we investigated whether further jasmonates and GH3 enzymes contribute to regulation of fertility. Jasmonate levels were analysed by liquid chromatography–mass spectrometry. The substrate range of recombinant GH3.10 and related GH3 enzymes was studied using non‐targeted ex vivo metabolomics with flower and leaf extracts of A. thaliana and in vitro enzyme assays. Jasmonate application experiments were performed to study their potential bioactivity. In flowers and wounded leaves of gh3.10 jar1 knockout lines JA‐Ile was below the detection limit. While 12‐hydroxy‐JA was identified as the preferred substrate of GH3.10, no other recombinant GH3 enzymes tested were capable of JA‐Ile formation. Additional JA conjugates found in wounded leaves (JA‐Gln) or formed in flowers upon MeJA treatment in the absence of JA‐Ile (JA‐Gln, JA‐Asn, JA‐Glu) were identified. The aos gh3.10 jar1 was introduced as a novel tool to test for the bioactivity of JA‐Gln to regulate fertility. This study found JAR1 and GH3.10 are the only contributors to JA‐Ile biosynthesis in Arabidopsis and identified a number of JA conjugates as potential bioactive jasmonates acting in the absence of JA‐Ile. However, their contribution in regulating fertility is yet to be conclusively determined. 12‐hydroxy jasmonate is directly converted into 12‐hydroxy jasmonoyl‐isoleucine and fertile jasmonoyl‐isoleucine deficient Arabidopsis mutants point towards the presence of other bioactive jasmonates in flowers. GH3 enzymes jasmonate biosynthesis jasmonate catabolism oxylipins phytohormones seed formation wound response Deutsche Forschungsgemeinschaft 10.13039/501100001659 GRK 2172 PRoTECT INST 186/1434‐1 INST 186/822‐1 ZUK 45/2010 pmc-status-qastatus 0 pmc-status-live yes pmc-status-embargo no pmc-status-released yes pmc-prop-open-access yes pmc-prop-olf no pmc-prop-manuscript no pmc-prop-legally-suppressed no pmc-prop-has-pdf yes pmc-prop-has-supplement yes pmc-prop-pdf-only no pmc-prop-suppress-copyright no pmc-prop-is-real-version no pmc-prop-is-scanned-article no pmc-prop-preprint no pmc-prop-in-epmc yes pmc-license-ref CC BY source-schema-version-number 2.0 cover-date June 2025 details-of-publishers-convertor Converter:WILEY_ML3GV2_TO_JATSPMC version:6.5.6 mode:remove_FC converted:22.05.2025 Editor: F. Loreto
```

### `ev_20a7facf5fd73dd9`

```text
results from dn- cis -OPDA/OPDA β-oxidation via the OPR1/2 route, functioning as a degradation pathway rather than a hormonal biosynthetic pathway on route to JA-Ile. The failure of exogenous JA to elicit any detectable growth inhibition in Marchantia , P. patens , or the hornwort Anthoceros agrestis confirms that JA is physiologically inert in these species [ 17 ]. The independent emergence of OPR3 in vascular plants correlates with the appearance of a JA-Ile-based signaling system, and the peroxisomal channeling of OPDA and dn- cis -OPDA toward JA by OPR3 may have been a prerequisite for (or co-evolved alongside) this metabolic transition during land plant evolution. 7. Summary Table, Current Gaps and Open Questions in Bryophytes’ Jasmonate Biosynthesis Jasmonate biosynthesis in bryophytes represents a fascinating evolutionary snapshot of a receptor-coupled oxylipin signaling system in a state prior to the emergence of the canonical angiosperm JA-Ile pathway. The pathway shares its fundamental enzymatic architecture, i.e., lipolytic n3-PUFA release, LOX-mediated oxygenation, AOS cyclization initiation, AOC stereospecific ring closure, but is distinguished from the angiosperm equivalent in several aspects: (i) additional n3-LCPUFA input from EPA via ER-localized MpFAD3; (ii) inverted relative importance of the hexadecanoid versus octadecanoid pathways, with HTA (16:3n3) providing the dominant dn- iso -OPDA precursor; (iii) a phylogenetically independent evolution of CYP74 AOSs with expanded substrate range; (iv) evidence for a predicted double-bond isomerase deactivating the RES capacity of the canonical cyclopentenone dn- cis -OPDA; (v) the absence of functional OPR3 and JAR1, defining the metabolic endpoint as dn- iso -OPDA rather than JA-Ile; and (vi) the deactivating (rather than activating) role of GH3-mediated amino acid conjugation. Table 1 summarizes all aspects of jasmonate biosynthesis in bryophytes discussed here. Table 1 Summary of characterized and uncharacterized enzymes potentially involved in jasmonate biosynthesis in Marchantia polymorpha . Enzyme # Gene ID Arabidopsis Ortholog Subcellular Localization Reaction Catalyzed Key Evidence Mutant * MpFAD5 Mp3g10660 FAD5 Chloroplast (predicted) Δ 7 -desaturation of 16:0 → 16:1n9 Genetic: Mp fad5 nearly abolishes dn-OPDA after wounding [ 13 ]. Yes (CC) a MpFAD6 (FAD10) Mp4g24020 FAD6 Chloroplast (predicted) Δ 12 -desaturation of 16:1(n9) → 16:2(n6) (HDA); obligate step toward HTA Inferred by pathway logic; no functional study reported. No MpFAD7 (FAD11) Mp5g20700 FAD7/FAD8 Chloroplast (confirmed) n3-desaturation of HDA (16:2n6) → HTA (16:3n3) and LA (18:2n6) → ALA (18:3n3) Genetic: Mp fad7 depleted in HTA/ALA and canonical jasmonates [ 14 ]. Yes (CC) b MpFAD3 (FAD8) Mp3g17690 FAD3 ER (confirmed) n3-desaturation of ARA (20:4n6) → EPA (20:5n3) Genetic: Mp fad3 depleted in EPA and abolishes C 20 -OPDA and Δ 4 -dn- iso -OPDA [ 14 ]. Yes (CC) b MpDES6 Mp6g17330 None (bryophyte-specific) ER (predicted) Δ 6 -desaturation: LA (18:2n6) → GLA (18:3n6); ALA → SDA (18:4n3); entry into LCPUFA elongation Biochemical: heterologous expression in yeast/plants [ 85 , 30 ]. Genetic: Mp des6 depleted in EPA and abolishes C 20 -OPDA and Δ 4 -dn- iso -OPDA [ 11 ]. Yes (CC) a MpELO1 Mp6g02350 None (bryophyte-specific) ER (predicted) Δ 6 -elongation: GLA (18:3n6) → DGLA (20:3n6); SDA (18:4n3) → ETA (20:4n3) Biochemical: heterologous expression in yeast/plants [ 85 , 30 ]. No MpDES5 Mp3g13250 None (bryophyte-specific) ER (predicted) Δ 5 -desaturation: DGLA (20:3n6) → ARA (20:4n6); ETA (20:4n3) → EPA (20:5n3) Biochemical: heterologous expression in yeast/plants [ 85 , 30 ]. Genetic: Mp des5 depleted in EPA. Abolishes C 20 -OPDA but not Δ 4 -dn- iso -OPDA [ 11 ]. Yes (CC) MpPATATIN Mp5g21760 PLA1 (DAD1) Unknown Putative lipase releasing n3-PUFAs from membrane glycerolipids Transcriptomic: wound-induced, COI1-dependent [ 11 , 13 ] No MpLOX1 Mp2g00660 None (liverwort-specific clade) Unknown 11 S /15 S -LOX activity on EPA/ARA; secondary 13-LOX on C 18 -PUFAs Biochemical: recombinant enzyme assays [ 29 ]. No MpLOX2 Mp2g23380 None (liverwort-specific clade) Unknown 15 S -LOX activity on EPA/ARA; secondary 13-LOX on C 18 -PUFAs Biochemical: recombinant enzyme assays [ 29 ]. No MpLOX3 Mp3g06610 None (liverwort-specific clade) Unknown 15 S -LOX activity on EPA/ARA; secondary 13-LOX on C 18 -PUFAs Biochemical: recombinant enzyme assays [ 29 ]. No MpLOX7 Mp2g12250 Non-seed-plant LOX clade Unknown C 6 aldehyde ( n -hexanal, Z -3-hexenal) formation via HPL-independent mechanism Biochemical: recombinant enzyme assays [ 49 ]. No MpAOS1 Mp3g21350 CYP74 (independent clade) Cytosol (confirmed) AOS activity on 13-HPOT, 15-HPEPE, and 15-HPETE; likely participates in C 20 -jasmonate pathway Genetic: Mp aos1 aos2 double mutant abolishes OPDA; single Mpaos1 shows elevated OPDA [ 33 , 60 ]. Yes (CC) c MpAOS2 Mp5g16260 CYP74 (independent clade) Chloroplast (confirmed) AOS activity; primary preference for 13-HPOT and 15-HPEPE; likely main contributor to canonical dn-OPDA Genetic: see MpAOS1; Mp aos2 single mutant retains OPDA [ 33 , 60 ]. Yes (CC) c MpAOC (DIR17) Mp7g06220 AOC1-4 Chloroplast (confirmed) Stereospecific ring closure of allene oxide → (9 S ,13 S )-dn- cis -OPDA or OPDA. Biochemical +
```

### `ev_3b1f8607f4b8b229`

```text
15 S -LOX activity on EPA/ARA; secondary 13-LOX on C 18 -PUFAs Biochemical: recombinant enzyme assays [ 29 ]. No MpLOX7 Mp2g12250 Non-seed-plant LOX clade Unknown C 6 aldehyde ( n -hexanal, Z -3-hexenal) formation via HPL-independent mechanism Biochemical: recombinant enzyme assays [ 49 ]. No MpAOS1 Mp3g21350 CYP74 (independent clade) Cytosol (confirmed) AOS activity on 13-HPOT, 15-HPEPE, and 15-HPETE; likely participates in C 20 -jasmonate pathway Genetic: Mp aos1 aos2 double mutant abolishes OPDA; single Mpaos1 shows elevated OPDA [ 33 , 60 ]. Yes (CC) c MpAOS2 Mp5g16260 CYP74 (independent clade) Chloroplast (confirmed) AOS activity; primary preference for 13-HPOT and 15-HPEPE; likely main contributor to canonical dn-OPDA Genetic: see MpAOS1; Mp aos2 single mutant retains OPDA [ 33 , 60 ]. Yes (CC) c MpAOC (DIR17) Mp7g06220 AOC1-4 Chloroplast (confirmed) Stereospecific ring closure of allene oxide → (9 S ,13 S )-dn- cis -OPDA or OPDA. Biochemical + overexpression: elevated OPDA, growth suppression; C 20 -allene oxide activity untested [ 65 ]. No cis -to- iso isomerase Unknown None identified Unknown Conversion of dn- cis -OPDA → dn- iso -OPDA (and Δ 4 -dn- cis - → Δ 4 -dn- iso -OPDA) Enzymatic activity in crude extracts [ 12 ]; no candidate gene identified No MpGH3A Mp6g07600 JAR1/GH3.11 Unknown Amino acid conjugation of dn- iso -OPDA → dn-OPDA-Glu/-Gln/-His (deactivating step) Genetic: Mp gh3a abolishes conjugates, elevates dn-OPDAs, enhances defense [ 15 ]. Yes (CC) MpILR1 Mp1g20090 IAR3/ILL6 (auxin hydrolase) Unknown Amidohydrolase; possible regeneration of free dn- iso -OPDA from aa-conjugates In vitro activity on some auxin-aa conjugates; dn-OPDA conjugate hydrolysis untested [ 74 ]. No MpCYP94s Mp2g10330 CYP94B/C (hydroxylases) Unknown Putative ω-/ω-1 hydroxylation of dn- iso -OPDA and Δ 4 -dn- iso -OPDA Transcriptomic: wound-induced, COI1-dependent [ 11 ]; no biochemical data No MpOPR1/MpOPR2 Mp4g01180/Mp6g05600 OPR1/OPR2 Unknown Reduction of 4,5-ddh-JA; residual JA biosynthesis No genetic or biochemical characterization; OPR3 activity absent [ 16 , 17 ]. No # names in brackets indicate the actual name in https://marchantia.info/ . * CC, CRISPR-Cas9-generated mutant. A superscript letter indicates availability of a double mutant of this enzyme and another enzyme sharing the same superscript letter. Despite the rapid progress of the last decade, several major gaps in understanding remain around jasmonate biosynthesis in bryophytes. The identity of the lipase(s) releasing HTA, ALA, and EPA from membrane glycerolipids in Marchantia remains uncharacterized (see Section 2.4 for full
```

### `ev_4dcc372e77b84976`

```text
13,600 L −1 ·mol −1 ·cm −1 (KC4 microplate reader, Biotec, 25 °C) for determination of the thiol/protein ratio with background correction (150 µL 40 mM K-P i , pH 8.0, 10 µL DTNB 2 mM, 1.5 µL ligand stock or EtOH). 4.10. Ex Vivo Evaluation of 12-OPDA as a LOX Inhibitor To evaluate 12-OPDA as a LOX inhibitor under native conditions, we employed pea roots as a plant LOX source. The reliability was assessed with the criterion that an increase in A 234nm and, ideally, oxygen consumption occurred via the injection of plant tissue extract (0.5–2 µg/µL protein in 50 mM Tris-HCl or K-P i , pH 6.6–8.0) in LeA (tested at 4–100 µM). Furthermore, boiling of plant tissue extract (5 min, 95 °C) resulted in decreased activities, determined as described in the following. Pea seeds ( Pisum sativum “Kleine Rheinländerin”) were placed in darkness for 4 days on moist paper. The emerged roots were cut, frozen in liquid N 2 , and stored at −80 °C. Frozen tissue (~50 mg/mL) was pulverized in a pre-chilled mortar and immediately ground with 50 mM Tris-HCl, pH 7.2. After centrifugation (14,000 rpm, 4 °C), the clear supernatant was used as a protein source (pea protein). The immunological detection of 13-LOX was performed, based on [ 81 ]. Pea proteins (8 µg) were separated via 7.5% SDS-PAGE and subsequently transferred (30 min, 2.0 mA/cm 2 ) with Towbin transfer buffer onto nitrocellulose membrane, with semi-dry blotter. After 1 h blocking with blocking solution (1% ( w / v ) non-fat dry milk in TBST (150 mM NaCl, 50 mM Tris-HCl, pH 7.5, 0.05% tween 20)) and overnight incubation (4 °C) with LOX2 (At3g45140) serum (Agrisera, AS07 258, 1:25,000 in blocking solution), the blot was washed with TBST and incubated (2 h, 25 °C) with horseradish peroxidase-coupled anti-rabbit IgG as second antibody (1:20,000 in blocking solution) for the subsequent luminescence detection of immunodecorated LOX on X-ray films (30 s incubation). The effect of 12-OPDA on pea root LOX activity was studied by incubating 60 µL (0.6 µg/µL protein) with 2 µL 30 mM 12-OPDA or EtOH (control) for 45 min at 35 °C. The mixture was injected in 1.8 mL of 20 µM LeA (in 50 mM Tris-HCl, pH 7.2) and recorded for oxygen consumption. 5. Conclusions This report provides novel insights into the regulatory effects of physiologically relevant compounds on 13-LOX isoforms from A. thaliana . We confirmed that all 13-LOXs carry chloroplast signal peptides and are potential sources of 12-OPDA. Our hypothesis of LOX inhibition by 12-OPDA was challenged via two methods, revealing that among recombinantly expressed proteins, LOX3 and LOX4 were the most sensitive to inhibition by 12-OPDA. Thus, feedback modulation of these oxygenases by 12-OPDA might be relevant in vivo, to decrease the potential accumulation of potentially toxic LOX products under conditions of stress. In addition, we found that LOX3 and LOX2 were most sensitive toward inhibition by MJ. Insight into selective interactions is also provided for divalent metals; of which, the highly toxic cadmium metal ion activated LOX2 and inactivated LOX3, LOX4 and LOX6. Furthermore, we suggest similarities between A. thaliana 13-LOXs and human LOXs, known to be involved in inflammation and diseases. A
```

### `ev_5863efff93736770`

```text
13-lipoxygenases (13-LOX) catalyze the dioxygenation of various polyunsaturated fatty acids (PUFAs), of which α-linolenic acid (LeA) is converted to 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT), the precursor for the prostaglandin-like plant hormones cis-(+)-12-oxophytodienoic acid (12-OPDA) and methyl jasmonate (MJ). This study aimed for characterizing the four annotated A. thaliana 13-LOX enzymes (LOX2, LOX3, LOX4, and LOX6) focusing on synthesis of 12-OPDA and 4Z,7Z,10Z)-12-[[-(1S,5S)-4-oxo-5-(2Z)-pent-2-en-1yl] cyclopent-2-en-1yl] dodeca-4,7,10-trienoic acid (OCPD). In addition, we performed interaction studies of 13-LOXs with ions and molecules to advance our understanding of 13-LOX. Cell imaging indicated plastid targeting of fluorescent proteins fused to 13-LOXs-N-terminal extensions, supporting the prediction of 13-LOX localization to plastids. The apparent maximal velocity (Vmax&nbsp;app) values for LOX-catalyzed LeA oxidation were highest for LOX4 (128 nmol·s-1·mg protein-1), with a Km value of 5.8 µM. A. thaliana 13-LOXs, in cascade with 12-OPDA pathway enzymes, synthesized 12-OPDA and OCPD from LeA and docosahexaenoic acid, previously shown only for LOX6. The activities of the four isoforms were differently affected by physiologically relevant chemicals, such as Mg2+, Ca2+, Cu2+ and Cd2+, and by 12-OPDA and MJ. As demonstrated for LOX4, 12-OPDA inhibited enzymatic LeA hydroperoxidation, with half-maximal enzyme inhibition at 48 µM. Biochemical interactions, such as the sensitivity of LOX toward thiol-reactive agents belonging to cyclopentenone prostaglandins, are suggested to occur in human LOX homologs. Furthermore, we conclude that 13-LOXs are isoforms with rather specific functional and regulatory enzymatic features.
```

### `ev_6915f534ca2dfd74`

```text
~5-fold when cleared lysates were purified via metal affinity chromatography. This is demonstrated via SDS-PAGE ( Figure S2 ) and activity analysis (activity analysis is shown for LOX6 in Figure S1 ). Further attempts to purify 13-LOX were not performed due to our observation of unexpected elution positions during gel-filtration (see Figure S2D ) and the instability of 13-LOX [ 12 ]. We thus decided to utilize LOX3, LOX4 and LOX6 eluates and LOX2-cleared lysate for subsequent 13-LOX analysis, if not mentioned otherwise. To analyze the role of 13-LOXs as putative Fe-LOXs (see Section 2.1 ), we analyzed their iron contents using the ferrozine assay (see Section 4.9 ), revealing Fe/LOX ratios (mol/mol, means±SD of n ≥ 4) of 0.48 ± 0.1 (LOX2), 0.36 ± 0.1 (LOX3), 0.40 ± 0.1 (LOX4) and 0.27 ± 0.06 (LOX6). The detected iron loads above 20% clearly assign 13-LOXs as Fe-LOXs [ 29 , 30 ], as suggested in Section 2.1 . As chloroplast pH-values range between 7.0 in the dark and >8.0 in the light, we recorded the activities of 13-LOX with their optimal substrate LeA between pH 6.6 and 8.6, to detect the pH-optima of 13-LOX. All members showed high activity around neutral pH and low activity at pH 8.6 ( Figure S2C ). LOX2 was most sensitive and LOX4 was least sensitive toward pH changes. At pH 6.6, LOX3 and LOX6 activity were lowered by ~30%, in contrast to LOX2 and LOX4, when compared to their optimal pH values at pH 6.6 (3.9 ± 0.4 nmol·s −1 ·mg −1 ) and pH 7.2 (62.3 ± 0.4 nmol·s −1 ·mg −1 ). The apparent affinities of 13-LOX enzymes toward LeA were highest for LOX6 (K m app = 1.2 ± 0.4 µM) and lowest for LOX2 (K m app = 26.3 ± 2 µM). The respective velocities of product formation were highest for LOX3 and LOX4 and lowest for LOX2 and LOX6, the latter two had apparent V max -values~45-fold lower than that of LOX4 (128 ± 18 nmol·s −1 ·mg −1 ), which is partly due to the different degrees of purity. The purities and iron loads below 100% were not considered in the activity calculation or related properties studied herein. The enzymatic parameters of LOX isoforms, with LA as substrate, could only be determined for LOX2 (for more details, see Figure 3 and Figure S3 ). Relative to LeA, LOX2 was unusually effective in LA- (79%) and ARA-oxygenation (50%) ( Table 1 ). Regarding LOX2 oxygenates LA at ω-6 (13-HPOD) [ 12 ], as revealed from RP-HPLC analysis ( Figure S4A ), 15-(S)-hydroperoxy-5(Z),8(Z),11(Z),13(E)-eicosatetraenoic acid (15-HPETE) was the oxylipin predominantly formed by LOX2 from ARA and displayed retention times (Rt) of 12.5 min (Rt 15-HPETE 12.5 min). As shown spectrophotometrically, all three LOX2-derived peroxides were metabolized by AOS ( Figure S4B ), which is supportive to the turnover of ω-6 oxygenated PUFAs by AOS of rice and Zea mays [ 31 , 32 ]. We intentionally tested whether the 22:6 (ω-3) PUFA DHA is a substrate of 13-LOX. As shown in Table 1 , 13-LOX enzymes dioxygenated DHA, with rates of ~7 to 56% relative to LeA, of which LOX2 was the most active enzyme. In line with the previous data on SLOX and LOX6 [ 33 ], a dominant oxylipin derivative of DHA, after incubation of DHA with LOX2, fractioned at Rt ~13 min. The MS analysis of this fraction revealed the presence of HPDH, as indicated by the major signal appearing at m / z of 383.2 due to the association of HPDH (MW = 360.5 g·mol −1 ) with sodium [M+Na] + . The molecular ion was subjugated to an MS 2 experiment and the formation of the fragment ion m / z 297 confirmed the position of hydroperoxidation at ω-6, due to the loss of an 86 Da fragment (C 5 H 10 O), in accordance with the findings of [ 33 ]. This hypothesis was further endorsed by the confirmation of the sum formulas of the molecular ion and the m / z 297.2 fragment ion ( Figure S5 and Table S2 ). The specificity of LOX3 and LOX4 was rather poor for producing 17-HPDH (see Figure S5 ). We observed that 13-LOXs were highly sensitive toward PUFAs or their oxidation products, whereas LOX2 was most robust. As analyzed for their favored substrates, more than 50% inhibition occurred when the enzymes (0.22 µg/µL) were pre-incubated with 29 µM LA (LOX6) or 118 µM LA (LOX3 and LOX4). The effect of LA and LeA on 13-LOX inhibition is depicted in Figure S6 , revealing similar trends when pre-incubated with LeA. 2.3. Arabidopsis 13-LOX Isoforms Mediate the Synthesis of 12-OPDA and OCPD 13-HPOT is the substrate for the subsequent synthesis of 12-OPDA. As shown previously for LOX6 [
```

### `ev_6f4dc3ffa5db1e0c`

```text
reflects the adaptation of MpLOXs to the liverwort LCPUFA pool [ 29 ]. 3.2. MpLOX7 Accepts C 18 -PUFA Substrates and Contributes to C 6 Aldehyde Formation Beyond MpLOX1-3, Tawfik et al. characterized MpLOX7 as a multifunctional enzyme with an unusually broad substrate tolerance, accepting both C 18 (LA, ALA) and C 20 (ARA, EPA) PUFAs at comparable efficiency, with ALA showing the highest k cat value [ 49 ]. This study placed MpLOX7 in a distinct clade of non-seed plant LOXs that encompasses sequences from Physcomitrium and Selaginella [ 49 ]. Its most remarkable property is the capacity to form C 6 aldehydes ( n -hexanal from ARA and Z -3-hexenal from ALA) as by-products during catalysis, through cleavage of allylic C–C bonds in the hydroperoxide intermediates. This LOX-mediated C 6 aldehyde formation does not require HPL activity, which M. polymorpha lacks entirely at the genomic level [ 18 ], and represents an unprecedented enzymatic mechanism for green leaf volatile production in liverworts. This observation could partially explain why M. polymorpha can produce C 6 volatiles like n -hexanal and Z -3-hexenal despite lacking a canonical HPL gene, offering an alternative evolutionary solution to volatile defense signaling [ 49 ]. Whether MpLOX7 is the only Marchantia LOX with HPL-like activity and whether it can contribute to jasmonate biosynthesis remain to be elucidated. 3.3. Uncharacterized Marchantia LOXs and Jasmonate Biosynthesis Despite significant recent progress in characterizing MpLOX1–3 and MpLOX7, the specific MpLOX enzyme(s) that catalyze the committed oxygenation step in dn-OPDAs and Δ 4 -dn- iso -OPDA biosynthesis in planta remain unidentified—one of the most pressing biochemical gap in the entire bryophyte jasmonate pathway. In Figure 3 , we present a phylogenetic analysis for LOX enzymes of representative plant lineages and green algae. Marchantia LOXs can be divided into three distinct groups ( Figure 3 , I, II and III). In group I, most MpLOXs possess a chloroplast transit peptide (CTP) and the only enzyme characterized from this clade, MpLOX3, has a predominant 15-LOX activity (see Section 3.1 ). Group II, including LOX6/7/8, is phylogenetically more distant to seed-plant LOXs than group I and four out of these seven LOXs present a CTP. Three characterized MpLOXs, namely MpLOX1/2/7, fall in group II with MpLOX1/2 and MpLOX7 forming two distinct clades and having diverse substrate preferences (see Section 3.1 and Section 3.2 ). The third group, group III, is composed of MpLOX5 and MpLOX9, which are clearly placed among seed-plant 13-LOXs, several of which are involved in jasmonate biosynthesis in their respective species. A priori, MpLOX5 and MpLOX9 appear as strong candidates for jasmonate biosynthesis; however, since Marchantia produces jasmonates from plastidial C 16 and C 18 PUFAs and from ER-produced C 20 LCPUFAs (requiring 11-, 13- and 15-LOX activity respectively), it is difficult to prioritize LOX candidates based on subcellular localization or substrate preference alone. Nevertheless, cross-stress transcriptomic atlases [ 51 ] and positive-feedback wound and jasmonate inducibility data represent important complementary criteria for narrowing the candidate LOX list, alongside CRISPR-Cas9 disruption of prioritized candidates. Figure 3 Evolutionary diversification of lipoxygenase (LOX) proteins across Viridiplantae. LOX protein sequences were selected based on a prior orthology analysis in which LOX homologs belonging to the same orthogroup as M. polymorpha LOX proteins were identified using OrthoFinder [ 52 ]. Protein sequences were aligned using MAFFT [ 53 ], and poorly aligned regions were removed with trimAl [ 54 ] using the automated trimming option. Maximum-likelihood phylogenetic inference was performed using IQ-TREE3 [ 55 ] with automatic model selection using ModelFinder [ 56 ], and branch support was assessed using 1000 ultrafast bootstrap replicates. Bootstrap support values are represented by proportional node symbols, with larger circles indicating stronger support. The first letters of protein names designate plant species as follows: At, Arabidopsis thaliana (dicots, tracheophytes); CHLRE Chlamydomonas reinhardtii (green algae, chlorophytes); kfl, Klebsormidium nitens (green algae, charophytes); Mp, Marchantia polymorpha (liverworts, bryophytes); Pp, Physcomitrium patens (mosses, bryophytes); Zm, Zea mays (monocots, tracheophytes). Taxa are color-coded according to major phylogenetic groups: bryophytes (green), spermatophytes (black), charophytes (orange), and chlorophytes (red). The phylogeny highlights lineage-specific diversification of LOX proteins across green plants, with notable expansion and diversification among bryophyte homologs. Roman numbers I, II, and III highlight Marchantia ’s LOXs clades. Characterized MpLOXs are highlighted in blue. Strong MpLOX candidates for jasmonate biosynthesis are highlighted in yellow. The color of small circles indicates the predicted subcellular location of the LOX protein: green for plastids, blue for cytosolic, orange for tonoplast, and gray for unknown. Circles outlined in black indicate reported experimental evidence. An asterisk (*) next to ZmLOXs indicates the opposite regiospecificity 9-LOX vs. 13-LOX. Regiospecificity and substrate preference inferred from phylogenetic position should always be interpreted with caution. 4. Allene Oxide Synthases in Bryophytes 4.1. MpAOS1 and MpAOS2: Two Functionally Diverged CYP74 Enzymes of
```

### `ev_7d2787cbb14b4905`

```text
Jasmonates regulate plant development and defence. In angiosperms, the canonical bioactive jasmonate is jasmonoyl-isoleucine (JA-Ile), which is formed in Arabidopsis thaliana by JAR1 and GH3.10. In contrast to other jasmonate biosynthesis or perception mutants, however, gh3.10 jar1 knockout lines are still fertile. Therefore we investigated whether further jasmonates and GH3 enzymes contribute to regulation of fertility. Jasmonate levels were analysed by liquid chromatography-mass spectrometry. The substrate range of recombinant GH3.10 and related GH3 enzymes was studied using non-targeted ex vivo metabolomics with flower and leaf extracts of A. thaliana and in vitro enzyme assays. Jasmonate application experiments were performed to study their potential bioactivity. In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit. While 12-hydroxy-JA was identified as the preferred substrate of GH3.10, no other recombinant GH3 enzymes tested were capable of JA-Ile formation. Additional JA conjugates found in wounded leaves (JA-Gln) or formed in flowers upon MeJA treatment in the absence of JA-Ile (JA-Gln, JA-Asn, JA-Glu) were identified. The aos gh3.10 jar1 was introduced as a novel tool to test for the bioactivity of JA-Gln to regulate fertility. This study found JAR1 and GH3.10 are the only contributors to JA-Ile biosynthesis in Arabidopsis and identified a number of JA conjugates as potential bioactive jasmonates acting in the absence of JA-Ile. However, their contribution in regulating fertility is yet to be conclusively determined.
```

### `ev_a82b0ad4450a0bd2`

```text
low gene redundancy of the jasmonate pathway in this liverwort genome. Recombinant MpAOC produces the naturally occurring (9 S ,13 S )-OPDA in a coupled in vitro assay with PpAOS1 (used as the AOS reaction partner) and 13-HPOT as substrate, and subcellular localization studies using MpAOC-GFP fusion proteins confirmed chloroplast targeting, mirroring the plastid-localized AOC enzymes of angiosperms [ 65 ]. Mp AOC expression is strongly induced by wounding and moderately by exogenous OPDA, demonstrating conservation of positive feedback regulation of the jasmonate pathway at the AOC level [ 65 ]. Overexpression of MpAOC led to elevated endogenous OPDA levels and growth suppression, directly confirming that OPDA functions as a signaling molecule in the liverwort. Notably, however, the study quantified only OPDA and did not generate Mp aoc loss-of-function lines; characterizing the full jasmonate profile, including dn-OPDAs and the Δ 4 -dn- iso -OPDA, in both overexpression and knockout lines will be essential to assign MpAOC’s contribution to the complete spectrum of MpCOI1 ligand biosynthesis [ 65 ]. The currently untested ability of MpAOC to process C 20 -PUFA-derived allene oxide intermediates has direct implications for the Δ 4 -dn- iso -OPDA pathway. If MpAOC cannot cyclize the allene oxide derived from 15-HPEPE (the MpAOS1/2 product from EPA oxygenation), two possibilities arise: (i) an as-yet-undiscovered cyclase operates on C 20 -allene oxides in Marchantia , or (ii) the C 20 -OPDA is produced non-enzymatically, a process that would yield a mixture of stereoisomers rather than the pure (9 S ,13 S ) compound. The fact that C 20 -OPDA is detected as a single diastereoisomer in planta argues for a stereospecific biosynthesis [ 11 , 33 ]. The genome of P. patens encodes two AOC paralogs, PpAOC1 and PpAOC2, both localizing to chloroplasts despite lacking predictable N -terminal transit peptides. PpAOC1 shares the substrate specificity of AtAOC2, converting only C 18 -PUFA-derived allene oxides to cis -OPDA. PpAOC2, by contrast, accepts both the C 18 -derived allene oxide and the allene oxide from 12-HPETE (a C 20 ARA-derived substrate), producing the novel cyclopentenone 11-oxo-prostatrienoic acid (11-OPTA) [ 62 ]. Crystal structures of PpAOC1 (1.35 Å) and PpAOC2 (1.98 Å) are nearly identical in overall fold to AtAOC2 despite ~450 Myr divergence. However, analysis of the PpAOC2 structure revealed three cooperative features that explain its expanded specificity, (i) a modestly enlarged active site diameter, (ii) an elongated narrow cavity that accommodates the additional carbons of the C 20 substrate, and (iii) two less bulky residues at the active site entrance that make Arg-22 accessible for electrostatic stabilization of the C 20 substrate’s carboxylate head group [ 62 ]. Site-directed mutagenesis incorporating key PpAOC2 residues into PpAOC1 (F29I/F140V) was insufficient to transfer C 20 -allene oxide activity, indicating that substrate specificity is determined cooperatively by the full active site geometry rather than by any of the tested amino acids [ 62 , 66 ]. Targeted disruption of Pp AOC1 or Pp AOC2 in P. patens had no effect on vegetative growth under standard conditions but resulted in ~10-fold reduced capsule number, aberrant sporophyte morphology, and arrested sporogenesis at the post-meiotic tetrad stage, suggesting that AOC-dependent cyclopentenone biosynthesis is essential for moss reproduction [ 66 ]. Double knockouts were unobtainable in repeated attempts, a result most consistent with, though not proof of, an essential overlapping function between the two paralogs. Exogenous OPDA failed to complement the mutant phenotype, raising the possibility that the biologically active jasmonate in reproductive tissues is not OPDA itself but likely dn-OPDAs, 11-OPTA, or a currently undetected AOC-dependent derivative [ 66 ]. These mutant phenotypes, together with the established requirement for JA in angiosperm fertility, suggest that the AOC-dependent oxylipin signaling for sexual reproduction is evolutionarily ancient and predates the full elaboration of the JA-Ile system in vascular plants. The contrast between P. patens , where AOC disruption impairs sporophyte development, and Marchantia , where jasmonate deficiency does not affect fertility, reflects an important divergence in downstream signaling even within the bryophytes, likely arising from distinct rewiring of the jasmonate co-receptor system across liverwort and moss lineages [ 62 , 65 , 66 ]. 6. dn-OPDAs and OPDA Modifications: Isomerization, Conjugation to Amino Acids, Hydroxylation and β-Oxidation on Route to JA 6.1. The Cis-to-Iso Isomerization: Paradoxical Fate Divergence The cis -to- iso isomerization of dn-OPDA occupies a pivotal position in bryophyte jasmonate biology. While OPDA and dn- cis -OPDA are reactive electrophilic species (RES) through their α,β-unsaturated carbonyl group, enabling COI1-independent thermotolerance responses in streptophytes including M. polymorpha , K. nitens , and Arabidopsis, the iso forms lack the RES property due to their more stable tetrasubstituted olefin [ 67 ]. Thus, the conversion of dn- cis -OPDA to dn- iso -OPDA (also valid for the Δ 4 -variants) may stabilize the molecule against non-enzymatic Michael addition by nucleophiles
```

### `ev_ae0d92db16547d2c`

```text
available in the Supporting Information of this article. Metabolite fingerprinting data can be found in the Metabolights database under the ID: MTBLS9749. References Acosta I.F. , Przybyl M. ( 2019 ) Jasmonate signaling during Arabidopsis stamen maturation . Plant and Cell Physiology , 60 , 2648 – 2659 . 31651948 10.1093/pcp/pcz201 PMC6896695 Aoi Y. , Tanaka K. , Cook S.D. , Hayashi K.‐I. , Kasahara H. ( 2019 ) GH3 auxin‐amido synthetases alter the ratio of indole‐3‐acetic acid and phenylacetic acid in Arabidopsis . Plant and Cell Physiology , 61 , 596 – 605 . 10.1093/pcp/pcz223 PMC7065595 31808940 Baek M. , DiMaio F. , Anishchenko I. , Dauparas J. , Ovchinnikov S. , Lee G.R. , Wang J. , Cong Q. , Kinch L.N. , Schaeffer R.D. , Millán C. , Park H. , Adams C. , Glassman C.R. , DeGiovanni A. , Pereira J.H. , Rodrigues A.V. , van Dijk A.A. , Ebrecht A.C. , Opperman D.J. , Sagmeister T. , Buhlheller C. , Pavkov‐Keller T. , Rathinaswamy M.K. , Dalwadi U. , Yip C.K. , Burke J.E. , Garcia K.C. , Grishin N.V. , Adams P.D. , Read R.J. , Baker D. ( 2021 ) Accurate prediction of protein structures and interactions using a three‐track neural network . Science , 373 , 871 – 876 . 34282049 10.1126/science.abj8754 PMC7612213 Baltzis A. , Mansouri L. , Jin S. , Langer B.E. , Erb I. , Notredame C. ( 2022 ) Highly significant improvement of protein sequence alignments with AlphaFold2 . Bioinformatics , 38 , 5007 – 5011 . 36130276 10.1093/bioinformatics/btac625 PMC9665868 Benavente L.M. , Alonso J.M. ( 2006 ) Molecular mechanisms of ethylene signaling in Arabidopsis . Molecular BioSystems , 2 , 165 – 173 . 16880934 10.1039/b513874d Benedetti C.E. , Xie D.X. , Turner J.G. ( 1995 ) COI1 ‐dependent expression of an Arabidopsis vegetative storage protein in flowers and siliques and in response to coronatine or methyl jasmonate . Plant Physiology , 109 , 567 – 572 . 7480347 10.1104/pp.109.2.567 PMC157621 Browse J. ( 2009 ) Jasmonate passes muster: A receptor and targets for the defense hormone . Annual Review of Plant Biology , 60 , 183 – 205 . 10.1146/annurev.arplant.043008.092007 19025383 Bruckhoff V. , Haroth S. , Feussner K. , König S. , Brodhun F. , Feussner I. ( 2016 ) Functional characterization of CYP94‐genes and identification of a novel jasmonate catabolite in flowers . PLoS One , 11 , e0159875 . 27459369 10.1371/journal.pone.0159875 PMC4961372 Brunoni F. , Široká J. , Mik V. , Pospíšil T. , Kralová M. , Ament A. , Pernisová M. , Karady M. , Htitich M. , Ueda M. , Floková K. , Wasternack C. , Strnad M. , Novák O. ( 2023 ) Conjugation of cis ‐OPDA with amino acids is a conserved pathway affecting cis ‐OPDA homeostasis upon stress responses . bioRxiv: 2023.2007.2018.549545. Cai Q. , Yuan Z. , Chen M. , Yin C. , Luo Z. , Zhao X. , Liang W. , Hu J. , Zhang D. ( 2014 ) Jasmonic acid regulates spikelet development in rice . Nature Communications , 5 , 3476 . 10.1038/ncomms4476 24647160 Casanova‐Sáez R. , Mateo‐Bonmatí E. , Šimura J. , Pěnčík A. , Novák O. , Staswick P. , Ljung K. ( 2022 ) Inactivation of the entire Arabidopsis group II GH3s confers tolerance to salinity and water deficit . New Phytologist , 235 , 263 – 275 . 35322877 10.1111/nph.18114 PMC9322293 Chandler J. ( 2009 ) Auxin as compère in plant hormone crosstalk . Planta , 231 , 1 – 12 . 19888599 10.1007/s00425-009-1036-x Chandler J.W. ( 2011 ) The hormonal regulation of flower development . Journal of Plant Growth Regulation , 30 , 242 – 254 . Chini A. , Monte I. , Zamarreño A.M. , Hamberg M. , Lassueur S. , Reymond P. , Weiss S. , Stintzi A. , Schaller A. , Porzel A. , García‐Mina J.M. , Solano R. ( 2018 ) An OPR3‐independent pathway uses 4,5‐didehydrojasmonate for jasmonate synthesis . Nature Chemical Biology , 14 , 171 – 178 . 29291349 10.1038/nchembio.2540 Delfin J.C. , Kanno Y. , Seo M. , Kitaoka N. , Matsuura H. , Tohge T. , Shimizu T. ( 2022 ) AtGH3.10 is another jasmonic acid‐amido synthetase in Arabidopsis thaliana . The Plant Journal , 110 , 1082 – 1096 . 35247019 10.1111/tpj.15724 Devoto A. , Nieto‐Rostro M. , Xie D. , Ellis C. , Harmston R. , Patrick E. , Davis J. , Sherratt L. , Coleman M. , Turner J.G. ( 2002 ) COl1 links jasmonate signalling and fertility to the SCF ubiquitin‐ligase complex in Arabidopsis . The Plant Journal , 32 , 457 – 466 . 12445118 10.1046/j.1365-313x.2002.01432.x Di D.‐W. , Li G. , Sun L. , Wu J. , Wang M. , Kronzucker H.J. , Fang S.
```

### `ev_ba1b08899518f20e`

```text
only in trace amounts in Marchantia or fully absent like JA-Ile; compounds in green dn- iso -OPDA, C 20 -OPDA and Δ 4 -dn- iso -OPDA are Marchantia -specific jasmonates. An asterisk (*) denotes enzymes that have not been fully characterized in Marchantia ; the curved orange arrow indicates a proposed lipid transport between organelles. DES, Desaturase Enzyme; FAB2, (Stearoyl-ACP Desaturase); FAD, Fatty Acid Desaturase; START2, Steroidogenic Acute Regulatory Protein-Related Lipid Transfer Domain Protein 2. Enzymatic reactions occur on the acyl-esterified forms of fatty acids, primarily esterified to MGDG (chloroplast pathway) or PC (ER pathway). The liverwort M. polymorpha encodes two n3-FADs, MpFAD3, a microsomal (ER-localized) enzyme equivalent to Arabidopsis AtFAD3, and MpFAD7, a chloroplast-localized enzyme equivalent to AtFAD7/FAD8 [ 14 , 27 ]. Note that MpFAD7 and MpFAD5 (see Section 2.1 ) are both plastidial enzymes but act at distinct positions on the desaturation cascade; MpFAD5 is a Δ 7 -desaturase introducing the first double bond into 16:0, while MpFAD7 is the terminal n3-FAD converting HDA (16:2n6) and LA (18:2n6) to HTA and ALA, respectively. Cold treatment induces a 3- to 6-fold increase in MpFAD3 transcript levels [ 27 ]. Since MpFAD3 catalyzes the conversion of ARA (20:4n6) to EPA (20:5n3) in the ER, this transcriptional upregulation translates into a corresponding increase in EPA content under cold conditions [ 27 ]. The cold inducibility of MpFAD3 parallels the cold-induced transcriptional regulation of AtFAD3 in Arabidopsis; however, this regulatory mechanism contributes to EPA (20:5n3) production, a source of Marchantia -specific jasmonates (see Section 2.3 below). By contrast, MpFAD7 is only mildly cold-inducible and its disruption has a dramatic effect on canonical (C 16 /C 18 ) jasmonate production [ 14 , 27 ]. Recent CRISPR-Cas9-based genetic dissection in Marchantia confirmed that MpFAD7 is a plastidial n3-FAD controlling HTA (16:3n3) and ALA (18:3n3) pools that feed the hexadecanoid and octadecanoid jasmonate biosynthetic pathways, respectively. Mutants lacking MpFAD7 are severely depleted in HTA and ALA, and correspondingly impaired in canonical dn-OPDAs and OPDA jasmonate production upon wounding. The ER-localized MpFAD3, in contrast, contributes to the n3-long chain PUFA (n3-LCPUFA) pool specifically by converting arachidonic acid (ARA, 20:4n6) to EPA (20:5n3), and its disruption significantly reduces C 20 -OPDA (a C 20 OPDA analog derived from EPA) and Δ 4 -dn- iso -OPDA (a dn-OPDA analog derived from C 20 -OPDA) production without affecting canonical C 16 /C 18 -derived jasmonates [ 14 ]. The Mp fad3fad7 double mutant hyperaccumulates omega-6 (n6) PUFAs and is a fully jasmonate-impaired line, a crucial tool for understanding jasmonate signaling in an evolutionary context [ 14 ]. 2.3. The C 20 -LCPUFA Pathway: Eicosanoids in Bryophytes A defining feature that sets bryophyte fatty acid metabolism apart from that of angiosperms is the capacity to synthesize LCPUFAs with chain lengths of C 20 and beyond [ 28 ]. In Marchantia , significant quantities of ARA (20:4n6) and EPA (20:5n3) are present, reaching up to 2.2 and 2.6 mg/g dry weight, respectively, in some accessions [ 28 , 29 ]. These LCPUFAs are absent from most angiosperm tissues (apart from very low levels in specialized cells and singular species), making the liverwort lipid composition resemble that of marine algae and lower eukaryotes [ 13 , 25 , 28 ]. The biosynthetic route from C 18 -PUFAs to LCPUFAs in Marchantia was delineated by Kajikawa et al. through heterologous expression and gain-of-function transgenic studies. Linoleic acid (LA, 18:2n6) and ALA (18:3n3) are first desaturated by MpDES6 (Δ 6 -desaturase) to γ-linolenic acid (GLA, 18:3n6) and stearidonic acid (SDA, 18:4n3), respectively. These are then elongated by MpELO1 (Δ 6 -elongase) to dihomo-γ-linolenic acid (DGLA, 20:3n6) and eicosatetraenoic acid (ETA, 20:4n3). Final Δ 5 -desaturation by MpDES5 converts DGLA to AA and ETA to EPA [ 30 ]. Thus, EPA biosynthesis in Marchantia proceeds through the “Δ 6 -pathway”, a metabolic route shared with many lower eukaryotes, algae, and fungi but distinct from the de novo LCPUFA biosynthesis in marine organisms [ 31 ]. This pathway requires all three enzymatic activities (MpDES6, MpELO1, MpDES5) to function coordinately [ 11 ]. Angiosperms lack functional Δ 6 -desaturase/elongase and Δ 5 -desaturase activities, which explains why they cannot proceed beyond C 18 -PUFAs under normal conditions [ 30 , 31 , 32 ]. This metabolic incapacity in flowering plants is consistent with their ancient evolutionary divergence from nonvascular plants in LCPUFA biosynthesis, though whether it represents a primary evolutionary absence rather than a secondary loss has not been definitively resolved by genomic analyses alone [ 30 , 31 , 32 ]. The expression of bryophyte LCPUFA genes in transgenic Arabidopsis was demonstrated to be sufficient for de novo ARA and EPA production in planta , confirming the ancestral nature of this pathway [ 30 ]. The downstream consequence that EPA serves as an additional
```

### `ev_c98d84c7d84855c1`

```text
7 , 8 , 17 , 21 , 22 , 23 , 24 ], and the present manuscript does not aim to duplicate that coverage. Instead, this review makes four specific contributions. First, we provide a step-by-step mechanistic account of the complete bryophyte jasmonate biosynthetic pathway, integrating the recently characterized C 20 -LCPUFA branch as an evolutionary novelty with no counterpart in angiosperms. Second, we discuss the cis -to- iso isomerization as a key ligand-activation step in Marchantia , recognizing the identification of the responsible enzyme as a critical unresolved gap in the pathway. Third, we highlight the deactivating role of MpGH3A-mediated amino acid conjugation in bryophytes—a functional inversion relative to the activating JA-Ile conjugation in angiosperms—with significant implications for understanding the evolutionary trajectory of jasmonate signaling. Fourth, we present an integrated comparative framework that traces mechanistic parallels and divergences between bryophytes and angiosperms at each biosynthetic step and systematically identifies the experimental priorities most likely to resolve outstanding questions. Collectively, this synthesis is intended to serve as a reference resource for researchers working on jasmonate evolution, oxylipin metabolism, and the biology of early diverging land plants. Figure 1 Comparative jasmonate biosynthetic pathways in Arabidopsis thaliana (At) and Marchantia polymorpha (Mp). Jasmonate PUFA precursors are synthesized in two distinct membrane compartments, the chloroplast pathway (top-left, green panel) and the ER (top-right, gray panel). ER-produced long-chain PUFAs (LCPUFAs) are proposed as Marchantia -specific jasmonate precursors. Subsequent processing happens in the peroxisome (center, red panel) and cytosol, before perception by a conserved co-receptor module occurs in the nucleus (bottom, purple panel). The orange curved arrow indicates a proposed translocation of EPA-containing lipids from the ER to the chloroplast mediated by MpSTART2 [ 25 ]. In Arabidopsis, OPDA and dn- cis -OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0, respectively. Rounds of β-oxidation yield (3 R ,7 S )-JA, which can epimerize to the more thermodynamically stable (3 R ,7 R )-JA. In Marchantia , the β-oxidation of jasmonates also occurs but appears to be context-dependent. Marchantia lacks JAR1 orthologs and ligand activation is achieved via a cis -to- iso isomerization by a putative isomerase. In the nucleus of Arabidopsis cells, JA-Ile bridges AtCOI1 and an AtJAZ to form a co-receptor complex that triggers JAZ degradation and de-repression of JRGs. This module is conserved in Marchantia , where MpCOI1 and the single MpJAZ are activated by dn- iso -OPDA or Δ 4 -dn- iso -OPDA as cognate ligands. Green, blue and red asterisks distinguish JA-Ile, dn- iso -OPDA and Δ 4 -dn- iso -OPDA ligands, respectively. ACX, Acyl-CoA Oxidase; ALA, α-linolenic acid; AOC, Allene Oxide Cyclase; AOS, Allene Oxide Synthase; COI1, Coronatine Insensitive 1; CTS, COMATOSE; ddh, didehydro; DAD1, Defective in Anther Dehiscence 1; DGDG, digalactosyldiacylglycerol; EPA, eicosapentaenoic acid; ER, endoplasmic reticulum; 10,11-EHT, 10,11-epoxy-hexadecatrienoic acid; 12,13-EOT, 12,13-epoxy-octadecatrienoic acid; 14,15-EEPE, 14,15-epoxy-eicosapentaenoic acid; 11-HPHT, 11-hydroperoxy-( Z , E , Z )-7,9,13-hexadecatrienoic acid; 15-HPEPE, 15-hydroperoxy-( Z , Z , Z , E , Z )-5,8,11,13,17-eicosapentaenoic acid. HTA, hexadecatrienoic acid; JA, jasmonic acid; JAR1, Jasmonate-Resistant 1; JASSY, chloroplast outer-membrane OPDA exporter; JAT1/2, Jasmonate Transporters 1 and 2; JAZ, Jasmonate ZIM-Domain; JA-Ile, jasmonoyl-L-isoleucine; JRGs, jasmonate-responsive genes; KAT, l-3-Ketoacyl-CoA Thiolase; LOX, Lipoxygenase; MGDG, monogalactosyldiacylglycerol; MFP, Multifunctional Protein; OPC, 3-oxo-2-(2′-pentenyl)-cyclopentane-acetic acid; OPDA, 12-oxo-phytodienoic acid; OPR, 12-Oxophytodienoate Reductase; PLA, Phospholipase; PC, phosphatidylcholine; START2, Steroidogenic Acute Regulatory Protein-related Lipid Transfer Domain Protein 2. ‘?’ denotes steps/enzymes proposed by sequence homology or biochemical inference but not yet experimentally confirmed. The orange straight arrows indicate translocation events. 2. Omega-3 PUFA Precursors: Fatty Acid Desaturation and the Unique Lipid Profile of Bryophytes 2.1. The Role of MpFAD5, and HTA as the Primary PUFA Precursor for dn-iso-OPDA Production A specific and biochemically important Δ 7 -desaturase, MpFAD5, was identified as the key enzyme responsible for the biosynthesis of HTA in Marchantia . Stable isotope feeding experiments using deuterium-labeled HTA (d6-16:3n3) confirmed the rapid conversion of HTA-derived intermediates into dn-OPDA isomers without passing through OPDA [ 9 ]. Genetic knockout of Mp FAD5 nearly abolishes dn-OPDA accumulation after mechanical wounding [ 13 ]. MpFAD5 is orthologous to AtFAD5, an Arabidopsis enzyme known to synthesize 16:1n9 (Δ 7 -palmitoleic acid) attached to the sn -2 position of monogalactosyldiacylglycerols (MGDG) in the plastids [ 26 ]. MpFAD5 functions as an n9-FAD acting on palmitic acid (16:0) to introduce the first unsaturation that ultimately leads to HTA via the evolutionarily conserved and concerted action of MpFAD5, MpFAD6 (a Δ 12 -desaturase; see Section 7 for its uncharacterized role), and FAD7 ( Marchantia encodes a single n3-FAD ortholog in the chloroplast, in contrast to the two paralogs FAD7/8 of Arabidopsis) acting on MGDG lipids [ 13 , 26 ]. Genetic studies employing Mp fad5 mutant plants confirmed that HTA, not OPDA β-oxidation, is the dominant source (>90%) of dn-OPDAs in Marchantia [ 13 ]. Importantly, while
```

### `ev_f7c56801adf6792d`

```text
troduction Jasmonates are a family of oxylipin signaling molecules that regulate plant development processes, including flowering time and senescence [ 1 , 2 , 3 ], and stress responses [ 4 , 5 , 6 ]. In response to mechanical damage, such as that incurred by chewing insect herbivory, jasmonate signaling gives rise to approximately 95% of the wound-induced protein synthesis that leads to the production of protective plant specialized metabolites and morphological barriers [ 7 , 8 ]. Given the importance of this pathway in plant resistance to pathogens and insect herbivores, understanding its regulation is key and may lead to novel strategies to enhance crop productivity [ 9 ]. After wounding or recognition of a necrotrophic pathogen, jasmonate biosynthesis starts in the chloroplast, where α -linolenic acid, a substrate of lipoxygenases (LOXs), is released from the galactolipids of the chloroplast membrane by a galactolipase [ 10 ]. The fatty acid undergoes oxidation, catalyzed by a member of the 13 S -LOX class of lipoxygenases, producing (13 S )-hydroperoxy-octadecatrienoic acid (13-HPOT) [ 11 ]. At this point in the pathway, different oxylipins may be produced. The reaction that converts 13-HPOT to (12 S -13 S )-epoxy-hydroperoxy-octadecatrienoic acid ((12 S -13 S )-EOT) is catalyzed by allene oxide synthase (AOS) and commits the pathway towards jasmonate biosynthesis [ 12 ]. (12 S -13 S )-EOT is reconfigured by allene oxide cyclase (AOC) into the more stable (9 S ,13 S )-12- oxo -phytodienoic acid (OPDA), a bioactive jasmonate [ 13 , 14 , 15 ]. OPDA is transported from the chloroplast into the cytosol and then into the peroxisome [ 16 , 17 ]. In this organelle, jasmonate biosynthesis continues with the reduction of OPDA by OPDA reductase 3 (OPR3) [ 18 , 19 , 20 ]. After esterification to CoA [ 21 ], OPC-8 undergoes β -oxidation by the fatty acid β -oxidation machinery composed of acyl-CoA oxidase (ACX), multifunctional protein (MFP), and 3-ketoacyl CoA thiolase (KAT) [ 22 , 23 , 24 , 25 , 26 , 27 ]. The product, jasmonoyl-CoA, is transformed into jasmonic acid (JA), which moves into the cytosol [ 28 , 29 , 30 ]. There, it may be converted into its biologically active form, jasmonoyl-isoleucine (JA-Ile), by jasmonate resistant 1 (JAR1) [ 31 ]. After entering the nucleus [ 32 ], JA-Ile binds to the Skp1/Cullin1/F-box protein coronatine-insensitive 1 (SCF CO1 ) complex, bridging the ubiquitin E3 ligase with its target repressor jasmonate ZIM-domain (JAZ) proteins, leading to its degradation through the 26 S -proteasome, thus freeing MYC2, MYC3, or MYC4 transcription factors from repression and allowing JA-Ile-responsive gene expression to proceed [ 32 , 33 , 34 , 35 , 36 ]. Targets of these transcription factors include genes that encode enzymes in the jasmonate biosynthetic pathway, leading to feedforward regulation [ 37 ]. In unwounded leaves, the basal JA level in Arabidopsis thaliana (arabidopsis) is less than 30 pmol/gm fresh mass [ 38 ]. After wounding, JA levels increase within seconds after mechanical damage, reaching levels that are over 10-fold higher than those of unwounded leaves within 2 min. This response is too rapid to be attributed to the increase in the expression of genes encoding enzymes in the jasmonate biosynthesis pathway. This indicates that the pathway must be constitutively present and activated upon damage [ 10 , 39 , 40 ]. Thivierge et al. [ 41 ] found that AtLOX2, which catalyzes an early step in jasmonate biosynthesis, is constitutively phosphorylated on Serine 600 (Ser 600 ) and dephosphorylated in response to wounding. Using phosphovariants, in vitro studies confirmed that the unphosphorylatable AtLOX2 had over a 100 times increase in V max compared to the phosphomimic, which likely reflects blocking of the substrate from the catalytic site [ 42 ]. Therefore, wounding stress is proposed to lead to the dephosphorylation of AtLOX2, contributing to the rapid increase in JA-Ile biosynthesis. In arabidopsis AtLOX2, the phosphosite, where the Ser 600 is shown in red, is flanked by the amino acid sequence ARQ S LVNG [ 42 ]. The sequence ARQ S LVNG is relatively conserved among plant 13 S -LOXs; however, there are variations at the phosphosite Ser, where Ala or Ile occupy this position, generating non-phosphorylatable variants. We chose pepper ( Capsicum annuum ) as our study organism since pepper has four 13 S -LOXs [ 43 ]; CaLOX2 and CaLOX8 have a Ser at the phosphosite, whereas CaLOX6 and CaLOX7 have an unphosphorylatable Ala at this site, suggesting that these proteins may be regulated at the post-translational or transcriptional levels, respectively (Note: Pepper LOXs have been color-coded; those with an Ala at the phosphosite ( CaLOX6 and CaLOX7 ) in orange and those with a Ser at this site ( CaLOX2 and CaLOX8 ) in blue). Alternative splicing
```

## Appendix B: files

| file | what |
|---|---|
| `build_audit3.py` | computes every number and writes both outputs |
| `md3.py` | this document's narrative |
| `pilot_003_audit.json` | machine-readable twin, including the 137 cut frontier items and the quoted code |
| `explore3.py`, `peek3.py`, `fix3.py` | scratch readers and an abandoned text patch (it aborted before writing); not needed to rebuild |
| `g11/*.json` | `bounded_run.py` reports; all show 0 survivors |
