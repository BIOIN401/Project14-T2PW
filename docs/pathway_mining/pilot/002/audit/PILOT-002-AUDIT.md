# PILOT-002 audit: pathway mining, jasmonate biosynthesis, *Arabidopsis thaliana*

Read-only evaluation of `C:\t\mining-pilot-runs2\pilot-002-jasmonate-ath` (code `C:/t/mining-pilot2 @ ecf6b9a02feb8904015fb4e249cc3a3e1a8cdbed`, prompt `mining-extract/1.1.0`), for ORCH-743. Nothing was re-run and no model was called. Every number below is read from a named artifact field by [`build_audit.py`](build_audit.py) (narrative in [`md_text.py`](md_text.py)), run through `bounded_run.py` with `PYTHONPATH=C:/t/mining-pilot2/src` and `PYTHONDONTWRITEBYTECODE=1`. The frozen tree is imported only for pure functions: the extraction prompt builder (with a capturing stub client, to measure the exact input), the title organism detector `Acquirer._other_organisms_in_title`, and the PubMed XML parser on a synthetic record. Machine-readable twin: [`pilot_002_audit.json`](pilot_002_audit.json).

**Headline.** `stopped_at_limit` / `max_papers_discovered_reached` (papers discovered: 150 >= limit 150), 3 rounds (the third asked nothing), 2039.163 s, est. $0.058908. 150 papers discovered (144 distinct articles), 11 included, 10 processed. 16 claims -> 11 reaction nodes: **1 primary** (`target_species_supported`) and 10 candidates.

**Key findings.**

1. **The one primary reaction is real biology on a mislabelled paper.** It is the 13-LOX step, alpha-linolenic acid -> 13-HPOT, catalysed by the four *A. thaliana* 13-LOXs (LOX2/3/4/6). Its species is **model-cited** (`species_evidence_ids` present in the raw completion; attribution `model_citation`), not recovered by the fallback. But its two cited spans are the same 1,775-character abstract indexed twice (`abstract#0` and `body#0`), and the paper id it cites, `PMC:PMC411572` with URL `doi.org/10.1073/pnas.76.9.4350`, is not this 2021 study: that DOI is PNAS volume 76 (1979) (A1, A4).
2. **NCBI records carry another article's PMCID.** `rag/acquire.py:536` reads `.//ArticleIdList/ArticleId` over the whole PubMed record, reference list included, so the last cited reference's `pmc` and `doi` win. The frozen parser does exactly that on a synthetic record (PMC411572 returned for a record whose own PMCID is PMC999001). In this run 7 NCBI ids are provably wrong. For 2 included papers the full-text fetch then pulled the wrong article, and extraction read it as the included paper. The NAD+ paper was read as a cotton-bollworm effector paper, and 'Arabidopsis GH3.10 conjugates jasmonates' as a 2016 wound-amidohydrolase paper. 6 articles were discovered twice under two ids. This is production code that pilot-001 never reached (NCBI was new in item 3).
3. **Empty completions are output-budget exhaustion.** 2 calls failed outright. They were the two 20-passage inputs, about 22.9k and 17.6k input tokens per request. All 9 provider requests of each ended `finish_reason=length` with **zero** content and exactly 4000 output tokens (= `max_tokens`). The hidden reasoning spent the whole budget before any JSON. The shrink ladder (60%, 35% + tight schema) did not help. Those two calls cost $0.04568 of $0.058908 (78%) and 1183.5 s of 2039.163 s. Pilot-001 extracted from **the same 20 passages** of PMC13364333 under prompt 1.0.0 in one request with 3,629 output tokens.
4. **Citation repair (item 2) worked.** 0 uncited-field drops (pilot-001: 59 of 59). 53 kept fields are all `model_citation`; `verbatim_cited_span` was never used, so bio watch item N-1 has nothing to adjudicate.
5. **Species labels are still keyword artefacts (F3, new form).** All 9 pepper-paper reactions are `in_vitro_or_cell_free`. The model cited *Capsicum annuum* for 8 of them. The label came from one incidental sentence in the introduction, 'in vitro studies confirmed that the unphosphorylatable AtLOX2 ...', which describes other work. Conversely the primary's abstract describes enzyme kinetics and a reconstituted cascade, but matches no in-vitro marker. Nothing wrong was admitted, but the labels and the primary's state need a bio-auditor ruling (A3).
6. **`max_papers_discovered=150` is now the binding stop, and it starves the search.** 3 round-1 queries brought in 66 papers. By query 18 of 58 the run had 149. From query 15 on, each query's window was cut to the remaining budget (11, then 1), so **40 queries asked for a single result**. 21 of the 25 items exhausted in round 2 were exhausted on windows the budget had cut. 15 of 44 round-2 items were never asked. Round 3 then started and asked nothing (A5, A6).
7. **Round 2 processed 5 papers and got 0 claims.** One was a wheat transcriptome paper (empty completion, then `{ "claims": [] }` at 60%). Four were abstract-only rescues. Two of those are on-topic (the AOC family of *A. thaliana*; AtLOX2 kinetics), but their abstracts name no complete substrate -> product conversion. The model returned `{"claims": []}`, which the prompt calls a correct answer.
8. **The rescue (item 4) did its job once.** Route 2 re-admitted the paper that gave the only primary reaction (matched term `12-OPDA`). Route 1 re-admitted 5 papers. One of them is a grape paper where *A. thaliana* is only an ectopic-expression host, a false rescue inside the documented ceiling. It was never processed (A8).

## 1. Pre-registered counts

| metric | value | source field | note |
|---|---|---|---|
| papers_discovered | `150` | summary.json papers.discovered=150; usage.papers_discovered=150; papers.jsonl rows=150 | by source: {'europepmc': 67, 'ncbi': 83} |
| papers_screened | `150` | summary.json papers.screened=150; papers.jsonl rows with screening_outcome set |  |
| papers_included | `11` | summary.json papers.included=11; papers.jsonl rows with empty exclusion_reasons | by route (papers.jsonl screening_outcome + inclusion_reasons): {'mining_rescue route 2': 1, 'production': 5, 'mining_rescue route 1': 5}; by source: {'ncbi': 8, 'europepmc': 3} |
| papers_excluded | `139` | summary.json papers.excluded |  |
| papers_processed | `10` | papers.jsonl processed_round>=0; usage.papers_processed=10; summary papers.processed=10 | the 11th included paper, PMC:PMC5104190, was never processed: its notes say 'round 2: no passages retrieved for this query' |
| claims_extracted | `16` | usage.claims_extracted=16; rounds/*/claims_extracted.jsonl rows=16 | claims.jsonl holds 11 stored claims (= summary.json claims.extracted=11, which is the STORED count, not the extracted one); distinct identities=11 |
| primary_reactions | `1` | primary_graph.json reaction_count; summary graph.primary_reactions=1 | component cmp_98656adf98c1 |
| candidate_reactions | `10` | candidate_subgraphs.json sum(len(components[].reactions)) over count=3 components; summary graph.candidate_reactions=10 | decision status of the 10: {'hypothesis_only': 1, 'candidate': 9} |
| rejected_by_reason | `{}` | summary.json claims.rejected_by_reason ({} = no claim has status 'rejected') | rounds/*/rejections.jsonl holds 15 rows (every NON-ADMITTED claim, controller.py _admit): statuses {'candidate': 13, 'hypothesis_only': 2} |
| duplicates_merged | `5` | reaction history events 'duplicate_merged' in primary_graph.json + candidate_subgraphs.json; usage.claims_extracted - distinct identities = 5 | all 5 are the model re-emitting a claim inside call mc_001_extract_claims_0004 (same paper, same span ev_f7c56801adf6792d): {'jasmonoyl-isoleucine (JA-Ile) -> JA-Ile-bound SCF COI1 complex': 1, 'OPC-8 -> jasmonoyl-CoA': 2, 'jasmonic acid (JA) -> jasmonoyl-isoleucine (JA-Ile)': 1, 'jasmonoyl-CoA -> jasmonic acid (JA)': 1} |
| species_states_reaction_nodes | `{"target_species_supported": 1, "in_vitro_or_cell_free": 9, "species_not_stated": 1}` | candidate_subgraphs.json + primary_graph.json reactions[].species_state (== summary.json graph.species_states) | per extracted claim (16): {'target_species_supported': 1, 'in_vitro_or_cell_free': 14, 'species_not_stated': 1} |
| budget_stops | `"max_papers_discovered_reached"` | run_manifest.json stop_reason / stop_detail='papers discovered: 150 >= limit 150' / completion_state=stopped_at_limit | the only limit reached. It was recorded during round 2; round 3 then started and ran 0 queries (see anomaly A5) |
| model_calls_by_status | `{"ok": 8, "empty_completion": 2}` | model_calls.jsonl status (10 logical calls); summary.json model_calls.by_status | ladder rungs (model_raw.jsonl rows, = usage.llm_calls): 15, by parse_outcome {'ok': 8, 'empty_completion': 7}; provider HTTP requests: 30 (+1 preflight probe) |
| empty_completion_log_lines | `7` | pilot-002.log lines 'giving up and returning the empty string' | one line per ladder rung whose 3 provider requests all came back empty: 3 rungs of mc_001_..._0002 + 3 of _0003 + 1 of mc_002_..._0006 = 7. model_calls.jsonl shows only 2 failed calls because _0006 recovered on rung 2 (status ok). Other log retries: ['finish_reason=error attempt 1/3'] |
| uncited_field_drops | `{"total": 0}` | state.json errors (0 rows); rounds/*/errors.jsonl rows=0 | pilot-001 dropped 59 of 59 |
| fields_kept_by_attribution | `{"compartment:model_citation": 9, "direction:model_citation": 8, "enzyme:model_citation": 8, "gene:model_citation": 2, "species:model_citation": 9}` | claims.jsonl field_attribution (11 stored claims) | over the 16 extracted claims: {'compartment:model_citation': 14, 'direction:model_citation': 13, 'enzyme:model_citation': 10, 'gene:model_citation': 2, 'species:model_citation': 14}; verbatim_cited_span = 0 |
| passages | `{"extracted": 80, "retrieved": 80}` | summary.json passages; evidence_spans.jsonl rows=80 |  |

Included papers by route: **mining_rescue route 1** 5, **mining_rescue route 2** 1, **production** 5.

**Model calls, three ways.** `model_calls.jsonl` has 10 logical calls ({'ok': 8, 'empty_completion': 2}). `usage.llm_calls` = 15 counts shrink-ladder rungs, one row each in `model_raw.jsonl`; that is the budgeted 'LLM calls'. Every rung goes to `t2pw.llm.client.chat_detailed`. It re-sends an identical request up to 3 times on an empty 200, which gives **30 provider requests** (+1 preflight probe). The log has 7 'giving up' lines, one per rung whose 3 requests all came back empty: 3 rungs of `mc_001_extract_claims_0002`, 3 of `_0003` and 1 of `mc_002_extract_claims_0006`. The status file shows only 2 failures because `_0006` recovered on rung 2 and is recorded `ok`. The log's one `finish_reason=error` line is `_0009`'s first provider request; its second succeeded.

Budgets (used / limit):

| budget | used | limit |
|---|---|---|
| rounds | 3 | 8 |
| papers discovered | 150 | 150 |
| papers processed | 10 | 40 |
| LLM calls (ladder rungs) | 15 | 200 |
| input tokens | 302829 | 2000000 |
| output tokens | 95417 | 400000 |
| est. cost USD | 0.058908 | 5.0 |
| wall clock s | 2039.163 | 3600.0 |
| frontier size (start of round 2 / 3) | 44 / 19 | 60 |
| unproductive streak | 2 | 2 |

## 2. Every reaction (1 primary + 10 candidates)

The seed connection is the auditor's name-level reading against the canonical route (13-LOX -> AOS -> AOC -> OPR3 -> beta-oxidation -> JA -> JAR1), used only as a yardstick. 'Graph' lists the run's own chemical connections. No entity is named `12-OPDA`; the seed metabolite exists only as `(9S,13S)-12-oxo-phytodienoic acid (OPDA)` (`en_b81b6bcf76945143`), and AOS as the enzyme entity `allene oxide synthase (AOS)`.

| # | reaction id | graph | equation (as recorded) | enzyme (attribution) | species state / value | paper | seed connection | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | `rx_98656adf98c1fca1946a` | primary `cmp_98656adf98c1` | α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | 13-lipoxygenase (13-LOX) (model_citation) | `target_species_supported` / Arabidopsis thaliana | `PMC:PMC411572` | upstream, 2 steps | `accepted` |
| 2 | `rx_85236be236d5f1e8a0f1` | candidate `cmp_2c50f30771a0` | jasmonoyl-isoleucine (JA-Ile) -> JA-Ile-bound SCF COI1 complex | none | `in_vitro_or_cell_free` / Capsicum annuum | `PMC:PMC12787732` | not a metabolic step | `hypothesis_only` |
| 3 | `rx_312345b466fea5ca097c` | candidate `cmp_5d3abd31040c` | OPC-8 -> jasmonoyl-CoA | acyl-CoA oxidase (ACX), multifunctional protein (MFP), 3-ketoacyl-CoA thiolase (KAT) (model_citation) | `in_vitro_or_cell_free` / Capsicum annuum | `PMC:PMC12787732` | downstream, 2 steps | `candidate` |
| 4 | `rx_355aa30ff85f270c9a58` | candidate `cmp_5d3abd31040c` | jasmonic acid (JA) -> jasmonoyl-isoleucine (JA-Ile) | jasmonate resistant 1 (JAR1) (model_citation) | `in_vitro_or_cell_free` / Capsicum annuum | `PMC:PMC12787732` | downstream, 4 steps | `candidate` |
| 5 | `rx_86b33c076d39e423f2fa` | candidate `cmp_5d3abd31040c` | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) -> (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT) | allene oxide synthase (AOS) (model_citation) | `in_vitro_or_cell_free` / Capsicum annuum | `PMC:PMC12787732` | DIRECT (seed enzyme AOS) | `candidate` |
| 6 | `rx_afb91c7037581c9fadc2` | candidate `cmp_5d3abd31040c` | (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT) -> (9S,13S)-12-oxo-phytodienoic acid (OPDA) | allene oxide cyclase (AOC) (model_citation) | `in_vitro_or_cell_free` / Capsicum annuum | `PMC:PMC12787732` | DIRECT (makes seed 12-OPDA) | `candidate` |
| 7 | `rx_b30980df6dd1f29b4504` | candidate `cmp_5d3abd31040c` | α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) | 13S-lipoxygenase (13S-LOX) (model_citation) | `in_vitro_or_cell_free` / (empty) | `PMC:PMC12787732` | upstream, 2 steps | `candidate` |
| 8 | `rx_c1d050d4171b110ad2d4` | candidate `cmp_5d3abd31040c` | α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) | none | `in_vitro_or_cell_free` / Capsicum annuum | `PMC:PMC12787732` | upstream, 2 steps | `candidate` |
| 9 | `rx_da81fbd8f5b34dc5ec32` | candidate `cmp_5d3abd31040c` | jasmonoyl-CoA -> jasmonic acid (JA) | none | `in_vitro_or_cell_free` / Capsicum annuum | `PMC:PMC12787732` | downstream, 3 steps | `candidate` |
| 10 | `rx_f8664da4761606d51db8` | candidate `cmp_5d3abd31040c` | (9S,13S)-12-oxo-phytodienoic acid (OPDA) -> OPC-8 | OPDA reductase 3 (OPR3) (model_citation) | `in_vitro_or_cell_free` / Capsicum annuum | `PMC:PMC12787732` | DIRECT (consumes seed 12-OPDA) | `candidate` |
| 11 | `rx_febac937577910d44f45` | candidate `cmp_ce149ce26fcb` | α-linolenic acid -> 13-hydroperoxy-α-linolenic acid | 13-lipoxygenase (LOX2) (model_citation) | `species_not_stated` / (empty) | `PMC:PMC6628814` | upstream, 2 steps | `candidate` |

### The primary reaction in full

- **Reaction:** `rx_98656adf98c1fca1946a` in component `cmp_98656adf98c1` (the primary component; 0 chemical connections). Equation: α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT).
- **Enzyme:** 13-lipoxygenase (13-LOX) (`model_citation`); gene LOX2, LOX3, LOX4, LOX6 (`model_citation`). Direction `unspecified`, no compartment extracted, though the abstract states plastid targeting.
- **Species:** `target_species_supported`, value "Arabidopsis thaliana", attribution **`model_citation`**. The raw completion (`rounds/round_001/model_raw.jsonl`, `mc_001_extract_claims_0001`, attempt 1) carries `"species": "Arabidopsis thaliana"` and `"species_evidence_ids": ["ev_b58e2cb64e2f1d87", "ev_ba6fb9f5cdb8ce58"]`. So the model cited it: this is not the `verbatim_cited_span` fallback. Reason recorded: "target species 'arabidopsis thaliana' explicitly supported by 2 evidence span(s)".
- **The span that backs the species:** `ev_b58e2cb64e2f1d87` (`abstract#0`, chars 0-1775) and `ev_ba6fb9f5cdb8ce58` (`body#0`, chars 0-1775). Identical text: **True**. The paper has no full text, so its abstract was indexed as both abstract and body. The 'two evidence spans' are one passage. The sentences that tie *A. thaliana* to the enzymes, verbatim:
  > This study aimed for characterizing the four annotated A. thaliana 13-LOX enzymes (LOX2, LOX3, LOX4, and LOX6) focusing on synthesis of 12-OPDA and 4Z,7Z,10Z)-12-[[-(1S,5S)-4-oxo-5-(2Z)-pent-2-en-1yl] cyclopent-2-en-1yl] dodeca-4,7,10-trienoic acid (OCPD).
  > A. thaliana 13-LOXs, in cascade with 12-OPDA pathway enzymes, synthesized 12-OPDA and OCPD from LeA and docosahexaenoic acid, previously shown only for LOX6.

  The span never writes "Arabidopsis thaliana" in full; the model expanded "A. thaliana", and `normalize_species` folds both.
- **Source paper:** `PMC:PMC411572`, "Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana". The recorded id and URL belong to another article (A1). The title and abstract are internally consistent: both come from the PubMed record.
- **Seed connection:** upstream, 2 steps: 13-LOX step: makes 13-HPOT, the AOS substrate; 12-OPDA is AOS then AOC away. It is **not connected** in the graph. Its product '13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT)' and the candidate AOS reaction's substrate '(13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)' are two lexical entities (A9).
- **Decision:** `accepted` (`dc_33a7aaedfc8d6eaf`, evidence level `single_paper`): "paper-supported, target-species reaction with single paper; no external database lookup was required or performed".
- **Auditor's note on the species state (not a verdict).** The abstract reports kinetics ("Vmax app ... Km") and says the 13-LOXs "in cascade with 12-OPDA pathway enzymes, synthesized 12-OPDA". `species_policy.md` defines `in_vitro_or_cell_free` as a "purified-enzyme" or "reconstituted system". No `_IN_VITRO_MARKERS` phrase occurs, so the keyword test passed it. The biology is textbook-correct for *A. thaliana*. Whether recombinant characterisation of the target's own enzymes counts as target-species evidence is a `policy_disagreement` question for `pwml-bio-auditor`.

### Per-reaction detail

#### 1. `rx_98656adf98c1fca1946a`: α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT)

- **Graph:** primary, component `cmp_98656adf98c1`; duplicate merges: 0; hypothesis: False
- **Enzyme(s):** 13-lipoxygenase (13-LOX) (`model_citation`). **Gene:** LOX2, LOX3, LOX4, LOX6 (`model_citation`)
- **Species:** `target_species_supported`; value Arabidopsis thaliana (`model_citation`, cites ev_b58e2cb64e2f1d87, ev_ba6fb9f5cdb8ce58). Reason: "target species 'arabidopsis thaliana' explicitly supported by 2 evidence span(s)"
- **Source paper:** `PMC:PMC411572`, "Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana". Claim(s) cl_9ac4b6db39b411c0 from `mc_001_extract_claims_0001`
- **Span `ev_b58e2cb64e2f1d87`** (abstract, `abstract#0`, chars 0-1775; identical text to ev_ba6fb9f5cdb8ce58): verbatim excerpt, full text in Appendix A:
  > 13-lipoxygenases (13-LOX) catalyze the dioxygenation of various polyunsaturated fatty acids (PUFAs), of which α-linolenic acid (LeA) is converted to 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT), the precursor for the prostaglandin-like plant hormones cis-(+)-12-oxophytodienoic acid (12-OPDA) and methyl jasmonate (MJ).
  > This study aimed for characterizing the four annotated A. thaliana 13-LOX enzymes (LOX2, LOX3, LOX4, and LOX6) focusing on synthesis of 12-OPDA and 4Z,7Z,10Z)-12-[[-(1S,5S)-4-oxo-5-(2Z)-pent-2-en-1yl] cyclopent-2-en-1yl] dodeca-4,7,10-trienoic acid (OCPD).
  > A. thaliana 13-LOXs, in cascade with 12-OPDA pathway enzymes, synthesized 12-OPDA and OCPD from LeA and docosahexaenoic acid, previously shown only for LOX6.
- **Span `ev_ba6fb9f5cdb8ce58`** (body, `body#0`, chars 0-1775; identical text to ev_b58e2cb64e2f1d87)
- **Seed connection:** upstream, 2 steps: 13-LOX step: makes 13-HPOT, the AOS substrate; 12-OPDA is AOS then AOC away. Graph: none
- **Decision:** `accepted` (`dc_33a7aaedfc8d6eaf`, `single_paper`): "paper-supported, target-species reaction with single paper; no external database lookup was required or performed"

#### 2. `rx_85236be236d5f1e8a0f1`: jasmonoyl-isoleucine (JA-Ile) -> JA-Ile-bound SCF COI1 complex

- **Graph:** candidate, component `cmp_2c50f30771a0`, quarantine reason `hypothesis_only`; duplicate merges: 1; hypothesis: True
- **Enzyme(s):** none. **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value Capsicum annuum (`model_citation`, cites ev_f7c56801adf6792d). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_d168265641d5797d from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > After entering the nucleus [ 32 ], JA-Ile binds to the Skp1/Cullin1/F-box protein coronatine-insensitive 1 (SCF CO1 ) complex, bridging the ubiquitin E3 ligase with its target repressor jasmonate ZIM-domain (JAZ) proteins, leading to its degradation through the 26 S -proteasome, thus freeing MYC2, MYC3, or MYC4 transcription factors from repression and allowing JA-Ile-responsive gene expression to proceed [ 32 , 33 , 34 , 35 , 36 ].
- **Seed connection:** not a metabolic step: JA-Ile binding the SCF-COI1 receptor complex is perception, not a conversion; held as hypothesis_only. Graph: product_to_substrate via 'jasmonoyl-isoleucine (JA-Ile)' with `rx_355aa30ff85f270c9a58`
- **Decision:** `hypothesis_only` (`dc_8c4ff0d260c0f16d`, `single_paper`): "the source presents this as a proposal or model, not an observed result; retained as a labelled hypothesis, never as an established reaction"

#### 3. `rx_312345b466fea5ca097c`: OPC-8 -> jasmonoyl-CoA

- **Graph:** candidate, component `cmp_5d3abd31040c`, quarantine reason `species:in_vitro_or_cell_free`; duplicate merges: 2; hypothesis: False
- **Enzyme(s):** acyl-CoA oxidase (ACX), multifunctional protein (MFP), 3-ketoacyl-CoA thiolase (KAT) (`model_citation`). **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value Capsicum annuum (`model_citation`, cites ev_f7c56801adf6792d). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_d1c4e947dc6fe086 from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > After esterification to CoA [ 21 ], OPC-8 undergoes β -oxidation by the fatty acid β -oxidation machinery composed of acyl-CoA oxidase (ACX), multifunctional protein (MFP), and 3-ketoacyl CoA thiolase (KAT) [ 22 , 23 , 24 , 25 , 26 , 27 ].
- **Seed connection:** downstream, 2 steps: OPC-8 -> JA-CoA; the span collapses CoA activation and three beta-oxidation cycles into one step. Graph: product_to_substrate via 'OPC-8' with `rx_f8664da4761606d51db8`; product_to_substrate via 'jasmonoyl-CoA' with `rx_da81fbd8f5b34dc5ec32`
- **Decision:** `candidate` (`dc_4b793bd723426aaf`, `single_paper`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"

#### 4. `rx_355aa30ff85f270c9a58`: jasmonic acid (JA) -> jasmonoyl-isoleucine (JA-Ile)

- **Graph:** candidate, component `cmp_5d3abd31040c`, quarantine reason `species:in_vitro_or_cell_free`; duplicate merges: 1; hypothesis: False
- **Enzyme(s):** jasmonate resistant 1 (JAR1) (`model_citation`). **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value Capsicum annuum (`model_citation`, cites ev_f7c56801adf6792d). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_87b979b4ddda7787 from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > There, it may be converted into its biologically active form, jasmonoyl-isoleucine (JA-Ile), by jasmonate resistant 1 (JAR1) [ 31 ].
- **Seed connection:** downstream, 4 steps: JAR1 conjugates JA to JA-Ile. Graph: product_to_substrate via 'jasmonic acid (JA)' with `rx_da81fbd8f5b34dc5ec32`; product_to_substrate via 'jasmonoyl-isoleucine (JA-Ile)' with `rx_85236be236d5f1e8a0f1`
- **Decision:** `candidate` (`dc_96c52f468f24da44`, `single_paper`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"

#### 5. `rx_86b33c076d39e423f2fa`: (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) -> (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)

- **Graph:** candidate, component `cmp_5d3abd31040c`, quarantine reason `species:in_vitro_or_cell_free`; duplicate merges: 0; hypothesis: False
- **Enzyme(s):** allene oxide synthase (AOS) (`model_citation`). **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value Capsicum annuum (`model_citation`, cites ev_f7c56801adf6792d). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_9a3a35f18bb35153 from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > The reaction that converts 13-HPOT to (12 S -13 S )-epoxy-hydroperoxy-octadecatrienoic acid ((12 S -13 S )-EOT) is catalyzed by allene oxide synthase (AOS) and commits the pathway towards jasmonate biosynthesis [ 12 ].
- **Seed connection:** DIRECT (seed enzyme AOS): the AOS step 13-HPOT -> 12,13-EOT, catalysed by the seed enzyme. Graph: product_to_substrate via '(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)' with `rx_afb91c7037581c9fadc2`; product_to_substrate via '(13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)' with `rx_b30980df6dd1f29b4504`; product_to_substrate via '(13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)' with `rx_c1d050d4171b110ad2d4`
- **Decision:** `candidate` (`dc_6cd2f91dd0e4e154`, `single_paper`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"

#### 6. `rx_afb91c7037581c9fadc2`: (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT) -> (9S,13S)-12-oxo-phytodienoic acid (OPDA)

- **Graph:** candidate, component `cmp_5d3abd31040c`, quarantine reason `species:in_vitro_or_cell_free`; duplicate merges: 0; hypothesis: False
- **Enzyme(s):** allene oxide cyclase (AOC) (`model_citation`). **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value Capsicum annuum (`model_citation`, cites ev_f7c56801adf6792d). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_bbf451c081405dad from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > (12 S -13 S )-EOT is reconfigured by allene oxide cyclase (AOC) into the more stable (9 S ,13 S )-12- oxo -phytodienoic acid (OPDA), a bioactive jasmonate [ 13 , 14 , 15 ].
- **Seed connection:** DIRECT (makes seed 12-OPDA): the AOC step; its product (9S,13S)-OPDA is the seed 12-OPDA. Graph: product_to_substrate via '(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)' with `rx_86b33c076d39e423f2fa`; product_to_substrate via '(9S,13S)-12-oxo-phytodienoic acid (OPDA)' with `rx_f8664da4761606d51db8`
- **Decision:** `candidate` (`dc_f8762ee4106b9520`, `single_paper`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"

#### 7. `rx_b30980df6dd1f29b4504`: α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)

- **Graph:** candidate, component `cmp_5d3abd31040c`, quarantine reason `species:in_vitro_or_cell_free`; duplicate merges: 0; hypothesis: False
- **Enzyme(s):** 13S-lipoxygenase (13S-LOX) (`model_citation`). **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value (empty). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_50aa0e7c0c06a243 from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > After wounding or recognition of a necrotrophic pathogen, jasmonate biosynthesis starts in the chloroplast, where α -linolenic acid, a substrate of lipoxygenases (LOXs), is released from the galactolipids of the chloroplast membrane by a galactolipase [ 10 ].
  > The fatty acid undergoes oxidation, catalyzed by a member of the 13 S -LOX class of lipoxygenases, producing (13 S )-hydroperoxy-octadecatrienoic acid (13-HPOT) [ 11 ].
- **Seed connection:** upstream, 2 steps: same 13-LOX step as the primary reaction (pepper introduction, species left empty). Graph: product_to_substrate via '(13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)' with `rx_86b33c076d39e423f2fa`; shared_substrate via 'α-linolenic acid' with `rx_c1d050d4171b110ad2d4`; shared_substrate via 'α-linolenic acid' with `rx_febac937577910d44f45`
- **Decision:** `candidate` (`dc_f8ff12d0b2da0699`, `single_paper`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"

#### 8. `rx_c1d050d4171b110ad2d4`: α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)

- **Graph:** candidate, component `cmp_5d3abd31040c`, quarantine reason `species:in_vitro_or_cell_free`; duplicate merges: 0; hypothesis: False
- **Enzyme(s):** none. **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value Capsicum annuum (`model_citation`, cites ev_f7c56801adf6792d). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_1edde52e869b2e7d from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > After wounding or recognition of a necrotrophic pathogen, jasmonate biosynthesis starts in the chloroplast, where α -linolenic acid, a substrate of lipoxygenases (LOXs), is released from the galactolipids of the chloroplast membrane by a galactolipase [ 10 ].
  > The fatty acid undergoes oxidation, catalyzed by a member of the 13 S -LOX class of lipoxygenases, producing (13 S )-hydroperoxy-octadecatrienoic acid (13-HPOT) [ 11 ].
- **Seed connection:** upstream, 2 steps: same 13-LOX step again, same span, species Capsicum annuum: a second node for the same sentence. Graph: product_to_substrate via '(13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)' with `rx_86b33c076d39e423f2fa`; shared_substrate via 'α-linolenic acid' with `rx_b30980df6dd1f29b4504`; shared_substrate via 'α-linolenic acid' with `rx_febac937577910d44f45`
- **Decision:** `candidate` (`dc_ee61992b2b43c5ae`, `single_paper`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"

#### 9. `rx_da81fbd8f5b34dc5ec32`: jasmonoyl-CoA -> jasmonic acid (JA)

- **Graph:** candidate, component `cmp_5d3abd31040c`, quarantine reason `species:in_vitro_or_cell_free`; duplicate merges: 1; hypothesis: False
- **Enzyme(s):** none. **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value Capsicum annuum (`model_citation`, cites ev_f7c56801adf6792d). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_7695a7e80bc79328 from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > The product, jasmonoyl-CoA, is transformed into jasmonic acid (JA), which moves into the cytosol [ 28 , 29 , 30 ].
- **Seed connection:** downstream, 3 steps: JA-CoA -> JA (thioesterase step; no enzyme stated). Graph: product_to_substrate via 'jasmonic acid (JA)' with `rx_355aa30ff85f270c9a58`; product_to_substrate via 'jasmonoyl-CoA' with `rx_312345b466fea5ca097c`
- **Decision:** `candidate` (`dc_6197c7c4346ed6e3`, `single_paper`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"

#### 10. `rx_f8664da4761606d51db8`: (9S,13S)-12-oxo-phytodienoic acid (OPDA) -> OPC-8

- **Graph:** candidate, component `cmp_5d3abd31040c`, quarantine reason `species:in_vitro_or_cell_free`; duplicate merges: 0; hypothesis: False
- **Enzyme(s):** OPDA reductase 3 (OPR3) (`model_citation`). **Gene:** none
- **Species:** `in_vitro_or_cell_free`; value Capsicum annuum (`model_citation`, cites ev_f7c56801adf6792d). Reason: "the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction"
- **Source paper:** `PMC:PMC12787732`, "Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves". Claim(s) cl_ea067e387e3f42ac from `mc_001_extract_claims_0004`
- **Span `ev_f7c56801adf6792d`** (introduction, `introduction#0`, chars 3467-8329): verbatim excerpt, full text in Appendix A:
  > In this organelle, jasmonate biosynthesis continues with the reduction of OPDA by OPDA reductase 3 (OPR3) [ 18 , 19 , 20 ].
- **Seed connection:** DIRECT (consumes seed 12-OPDA): OPR3 reduces the seed 12-OPDA to OPC-8 (downstream). Graph: product_to_substrate via '(9S,13S)-12-oxo-phytodienoic acid (OPDA)' with `rx_afb91c7037581c9fadc2`; product_to_substrate via 'OPC-8' with `rx_312345b466fea5ca097c`
- **Decision:** `candidate` (`dc_b512c63d1ac266b2`, `single_paper`): "species state 'in_vitro_or_cell_free' is not target-species-supported (the supporting passage describes an in vitro / cell-free system, so no organism performed the reaction); retained as a labelled lead, not as an established target-species reaction"

#### 11. `rx_febac937577910d44f45`: α-linolenic acid -> 13-hydroperoxy-α-linolenic acid

- **Graph:** candidate, component `cmp_ce149ce26fcb`, quarantine reason `species:species_not_stated`; duplicate merges: 0; hypothesis: False
- **Enzyme(s):** 13-lipoxygenase (LOX2) (`model_citation`). **Gene:** LOX2 (`model_citation`)
- **Species:** `species_not_stated`; value (empty). Reason: "no species is stated anywhere in the supporting evidence"
- **Source paper:** `PMC:PMC6628814`, "NAD+ deficiency primes defense metabolism via 1O2-escalated jasmonate biosynthesis in plants". Claim(s) cl_f349e4bfd16fa738 from `mc_001_extract_claims_0005`
- **Span `ev_49c948bc587a2016`** (abstract, `abstract#0`, chars None-None): verbatim excerpt, full text in Appendix A:
  > The elevated defense in qs-2 is resulted from activated jasmonate biosynthesis, critically hydroperoxidation of α-linolenic acid by the 13-lipoxygenase (namely LOX2), which is escalated via the burst of chloroplastic ROS-singlet oxygen (1O2).
- **Seed connection:** upstream, 2 steps: the same 13-LOX step (LOX2) from the NAD+ paper's abstract. Graph: shared_substrate via 'α-linolenic acid' with `rx_b30980df6dd1f29b4504`; shared_substrate via 'α-linolenic acid' with `rx_c1d050d4171b110ad2d4`
- **Decision:** `candidate` (`dc_7cbf5810f3f97827`, `single_paper`): "species state 'species_not_stated' is not target-species-supported (no species is stated anywhere in the supporting evidence); retained as a labelled lead, not as an established target-species reaction"

**Why the 9 pepper reactions are `in_vitro_or_cell_free`.** All cite the one introduction span `ev_f7c56801adf6792d`. The model cited *Capsicum annuum* for 8 of them, from:
> We chose pepper ( Capsicum annuum ) as our study organism since pepper has four 13 S -LOXs [ 43 ]; CaLOX2 and CaLOX8 have a Ser at the phosphosite, whereas CaLOX6 and CaLOX7 have an unphosphorylatable Ala at this site, suggesting that these proteins may be regulated at the post-translational or transcriptional levels, respectively (Note: Pepper LOXs have been color-coded; those with an Ala at the phosphosite ( CaLOX6 and CaLOX7 ) in orange and those with a Ser at this site ( CaLOX2 and CaLOX8 ) in blue).

The in-vitro label comes from this sentence about other work, which is test 2 in `classify_species` and runs before the species comparison:
> Using phosphovariants, in vitro studies confirmed that the unphosphorylatable AtLOX2 had over a 100 times increase in V max compared to the phosphomimic, which likely reflects blocking of the substrate from the catalytic site [ 42 ].

The passage itself is a literature summary of the pathway, largely Arabidopsis work, and not an in-vitro experiment. Holding every one back is the right outcome. The honest label for the 8 with a cited species is `non_target_species` (*Capsicum annuum*). The one with no species is closer to `species_not_stated`.

## 3. Every included paper

Title organisms are the frozen `Acquirer._other_organisms_in_title`, replayed read-only on the recorded title. 'Text used' is what extraction read.

| # | paper id | title | source | how included | matched term / recorded reason | title organisms (non-target) | text used | passages / call / claims |
|---|---|---|---|---|---|---|---|---|
| 1 | `PMC:PMC411572` | Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana. | ncbi | mining_rescue route 2 | mining_rescue: names 'a. thaliana' and '12-OPDA' in one sentence -- production screen: off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | none | abstract only | 2 / mc_001_extract_claims_0001 / 1 |
| 2 | `PMC:PMC13364333` | Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from &lt;i&gt;Marchantia polymorpha&lt;/i&gt; and Beyond. | europepmc | production | score 5.0 >= threshold 2.0; anchored on jasmonate biosynthesis, jasmonate biosynthetic pathway, jasmonate biosynthetic (local: precursors) | none | full text 98172 chars | 20 / mc_001_extract_claims_0002 / 0 |
| 3 | `PMC:PMC4793799` | Arabidopsis GH3.10 conjugates jasmonates. | ncbi | production | score 4.75 >= threshold 2.0; anchored on jasmonate biosynthesis (local: knockout) | none | full text 64756 chars **of a different article** ("Hormone crosstalk in wound stress response: wound-inducible amidohydrolases can simultaneo...") | 20 / mc_001_extract_claims_0003 / 0 |
| 4 | `PMC:PMC12787732` | Transcription, Alternative Splicing, and Post-Translational Regulation of CaLOXs in the Dynamic Regulation of Jasmonate Levels in Wounded Pepper Leaves. | europepmc | production | score 4.0 >= threshold 2.0; anchored on jasmonate biosynthesis (local: catalyzes) | none | full text 88532 chars | 16 / mc_001_extract_claims_0004 / 14 |
| 5 | `PMC:PMC6628814` | NAD+ deficiency primes defense metabolism via 1O2-escalated jasmonate biosynthesis in plants. | ncbi | production | score 4.5 >= threshold 2.0; anchored on jasmonate biosynthesis (local: cofactor) | none | full text 47121 chars **of a different article** ("An effector from cotton bollworm oral secretion impairs host plant defense signaling Chen ...") | 8 / mc_001_extract_claims_0005 / 1 |
| 6 | `PMC:PMC5382222` | Identification of Candidate Genes and Biosynthesis Pathways Related to Fertility Conversion by Wheat KTM3315A Transcriptome Profiling. | europepmc | production | score 2.75 >= threshold 2.0; anchored on jasmonate biosynthesis (local: conversion of) | wheat | full text 49516 chars | 6 / mc_002_extract_claims_0006 / 0 |
| 7 | `PMC:PMC1955694` | Cesium Inhibits Plant Growth through Jasmonate Signaling in Arabidopsis thaliana. | ncbi | mining_rescue route 1 | mining_rescue: Arabidopsis thaliana named; score 3.0 >= 3.0 with the pathway-anchor rule off -- production screen: context_only: 'jasmonate biosynthesis, jasmonate pathway' appears in the title/abstract but no strong reaction/enzyme evidence is local to it | none | abstract only | 2 / mc_002_extract_claims_0007 / 0 |
| 8 | `PMID:12777050` | Jasmonate biosynthesis and the allene oxide cyclase family of Arabidopsis thaliana. | ncbi | mining_rescue route 1 | mining_rescue: Arabidopsis thaliana named; score 4.0 >= 3.0 with the pathway-anchor rule off -- production screen: omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: up regulated) | none | abstract only | 2 / mc_002_extract_claims_0008 / 0 |
| 9 | `PMC:PMC3000969` | Pseudophosphorylation of Arabidopsis jasmonate biosynthesis enzyme lipoxygenase 2 via mutation of Ser600 inhibits enzyme activity. | ncbi | mining_rescue route 1 | mining_rescue: Arabidopsis thaliana named; score 4.0 >= 3.0 with the pathway-anchor rule off -- production screen: context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it | none | abstract only | 2 / mc_002_extract_claims_0009 / 0 |
| 10 | `PMC:PMC5104190` | GRAS-domain transcription factor PAT1 regulates jasmonic acid biosynthesis in grape cold stress response. | ncbi | mining_rescue route 1 | mining_rescue: Arabidopsis thaliana named; score 3.25 >= 3.0 with the pathway-anchor rule off -- production screen: omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptome) | none | abstract only | 0 / - / 0 |
| 11 | `PMC:PMC5338657` | Glutathione and Its Biosynthetic Intermediates Alleviate Cesium Stress in Arabidopsis. | ncbi | mining_rescue route 1 | mining_rescue: Arabidopsis thaliana named; score 4.0 >= 3.0 with the pathway-anchor rule off -- production screen: context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it | none | abstract only | 2 / mc_002_extract_claims_0010 / 0 |

**Route 2** (`PMC:PMC411572`): matched species form `a. thaliana` and term `12-OPDA` (a seed). The sentence, re-found by splitting title + abstract with the frozen `_SENTENCE`:
> This study aimed for characterizing the four annotated A. thaliana 13-LOX enzymes (LOX2, LOX3, LOX4, and LOX6) focusing on synthesis of 12-OPDA and 4Z,7Z,10Z)-12-[[-(1S,5S)-4-oxo-5-(2Z)-pent-2-en-1yl] cyclopent-2-en-1yl] dodeca-4,7,10-trienoic acid (OCPD).

The production screen had called it `off_topic` (score 3.0). The seed enzyme `AOS` can never be a route-2 term: it is 3 characters, under `_MIN_TERM_CHARS = 5`. So round 1 had one usable rescue term, `12-OPDA`.

**Route 1** (5 papers, all round 2, all abstract-only NCBI records):
- `PMC:PMC1955694` score 3.0: Cesium Inhibits Plant Growth through Jasmonate Signaling in Arabidopsis thaliana. Claims: 0; processed round 2.
- `PMID:12777050` score 4.0: Jasmonate biosynthesis and the allene oxide cyclase family of Arabidopsis thaliana. Claims: 0; processed round 2.
- `PMC:PMC3000969` score 4.0: Pseudophosphorylation of Arabidopsis jasmonate biosynthesis enzyme lipoxygenase 2 via mutation of Ser600 inhibits enzyme activity. Claims: 0; processed round 2.
- `PMC:PMC5104190` score 3.25: GRAS-domain transcription factor PAT1 regulates jasmonic acid biosynthesis in grape cold stress response. Claims: 0; processed round -1. The abstract contains the foreign-host word "ectopic"; the guard reads the title only.
- `PMC:PMC5338657` score 4.0: Glutathione and Its Biosynthetic Intermediates Alleviate Cesium Stress in Arabidopsis. Claims: 0; processed round 2.

`PMC:PMC5104190` is a false rescue. It is about grape (*Vitis amurensis*) PAT1, and its abstract says "ectopic expression of VaPAT1 enhances cold tolerance in Arabidopsis". The title word 'grape' is unknown to both organism detectors and to `_EXTRA_HOSTS`, the documented ceiling. It was never processed: retrieval returned no passage for it (A7).

**Production inclusions worth noting.** `PMC:PMC5382222` is a wheat transcriptome paper. The title detector finds 'wheat' in it, but that check applies only to route 1; the production screen passed it (score 2.75). `PMC:PMC13364333` is the Marchantia review pilot-001 read. It still has evidence type `unknown` (F6).

### NCBI identity evidence (A1)

NCBI supplied 83 of 150 discovered papers (50 with a PMC id) and 8 of 11 included. The parser reads the PMCID and DOI from any `ArticleIdList` in the record, and the last one wins (`rag/acquire.py:536-543`). A synthetic record run through the frozen `_parse_pubmed_article` (own ids {"pmid": "111", "pmc": "PMC999001", "doi": "10.9999/own"}, one cited reference {"pmc": "PMC411572", "doi": "10.1073/pnas.76.9.4350"}) returns `{"id": "PMC411572", "source_uri": "https://doi.org/10.1073/pnas.76.9.4350", "title": "Own title"}`.

| evidence | NCBI paper id | what the record says | what the id actually is |
|---|---|---|---|
| full text fetched for the recorded PMCID | `PMC:PMC4793799` | "Arabidopsis GH3.10 conjugates jasmonates." (2025) | "Hormone crosstalk in wound stress response: wound-inducible amidohydrolases can simultaneously regulate jasmon..." (header of the fetched full text) |
| full text fetched for the recorded PMCID | `PMC:PMC6628814` | "NAD+ deficiency primes defense metabolism via 1O2-escalated jasmonate biosynthesis in plants." (2024) | "An effector from cotton bollworm oral secretion impairs host plant defense signaling Chen Chun-Yu a b 1 Liu Ya..." (header of the fetched full text) |
| same article also found via EuropePMC | `PMC:PMC7013545` (included: False) | "Genetic Transformation of Triticum dicoccum and Triticum aestivum with Genes of Jasmonate Biosynthes" | EuropePMC gives it `PMC:PMC11478715` (included: False) |
| same article also found via EuropePMC | `PMC:PMC6164985` (included: False) | "The plant endophytic fungus Cyanodermella asteris produces the phytohormone jasmonic acid." | EuropePMC gives it `PMC:PMC13063583` (included: False) |
| same article also found via EuropePMC | `PMC:PMC6628814` (included: True) | "NAD&lt;sup&gt;+&lt;/sup&gt; deficiency primes defense metabolism via &lt;sup&gt;1&lt;/sup&gt;O&lt;su" | EuropePMC gives it `PMC:PMC11300881` (included: False) |
| same article also found via EuropePMC | `PMC:PMC2716216` (included: False) | "Interaction with CYP20-3 limits OPDA flux into jasmonate biosynthesis in wounded leaves of arabidops" | EuropePMC gives it `PMC:PMC13096701` (included: False) |
| same article also found via EuropePMC | `PMC:PMC3694706` (included: False) | "Size of chloroplasts in Arabidopsis mesophyll cells affects jasmonate biosynthesis." | EuropePMC gives it `PMC:PMC13358708` (included: False) |
| same article also found via EuropePMC | `PMC:PMC8265181` (included: False) | "Arabidopsis Transcriptomics Reveals the Role of Lipoxygenase2 (AtLOX2) in Wound-Induced Responses." | EuropePMC gives it `PMC:PMC11173247` (included: False) |
| DOI inconsistent with the record | `PMC:PMC411572` | "Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana." (year 2021) | URL `https://doi.org/10.1073/pnas.76.9.4350`: PNAS volume 76, i.e. 1979 |

Consequences in this run:
- The only primary reaction cites the wrong paper id.
- Two extraction calls read another article's full text under the included paper's title (`PAPER:` header). The NAD+ paper's one claim came from its own PubMed abstract (`ev_49c948bc587a2016`, null offsets because that abstract is not a substring of the fetched text). Its other 7 passages were the cotton-bollworm paper.
- The NAD+ article was screened twice. The EuropePMC copy was excluded (score 3.0, `omics_only`); the NCBI copy was included (4.5).
- 150 'discovered' papers are 144 distinct articles.

### Recall check: excluded titles that name the pathway

| paper id | title | source | score | exclusion | title names Arabidopsis |
|---|---|---|---|---|---|
| `PMC:PMC12865809` | Plant oxylipins: adaptation to environmental stresses and impact on mycotoxin contamination. | europepmc | 0.0 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC7013545` | Genetic Transformation of Triticum dicoccum and Triticum aestivum with Genes of Jasmonate Biosynthesis Pathway Affects Growth and Productivity Characteristics. | ncbi | 2.5 | context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reactio | False |
| `PMC:PMC9505089` | Maize OPR2 and LOX10 Mediate Defense against Fall Armyworm and Western Corn Rootworm by Tissue-Specific Regulation of Jasmonic Acid and Ketol Metabolism. | ncbi | -0.75 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC13493544` | Phyto-Oxylipins as Key Regulators of Plant Resilience to Abiotic and Biotic Stress. | europepmc | -0.75 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC4970199` | The AtMYB60 transcription factor regulates stomatal opening by modulating oxylipin synthesis in guard cells. | ncbi | -1.5 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC11478715` | Genetic Transformation of <i>Triticum dicoccum</i> and <i>Triticum aestivum</i> with Genes of Jasmonate Biosynthesis Pathway Affects Growth and Productivity Characteristics. | europepmc | 2.5 | context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reactio | False |
| `PMC:PMC4911631` | The In Vitro Interaction of 12-Oxophytodienoic Acid and Related Conjugated Carbonyl Compounds with Thiol Antioxidants. | ncbi | 0.25 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMID:31834385` | 12-Oxophytodienoic Acid Reductase 3 (OPR3) Functions as NADPH-Dependent α,β-Ketoalkene Reductase in Detoxification and Monodehydroascorbate Reductase in Redox Homeostasis. | ncbi | 1.5 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC3938610` | Drought stress modulates oxylipin signature by eliciting 12-OPDA as a potent regulator of stomatal aperture. | ncbi | -0.75 | omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC2039767` | Functional convergence of oxylipin and abscisic acid pathways controls stomatal closure in response to drought. | ncbi | 0.0 | omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC11300881` | NAD&lt;sup&gt;+&lt;/sup&gt; deficiency primes defense metabolism via &lt;sup&gt;1&lt;/sup&gt;O&lt;sub&gt;2&lt;/sub&gt;-escalated jasmonate biosynthesis in plants. | europepmc | 3.0 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | False |
| `PMC:PMC13468208` | Maize Lipoxygenase &lt;i&gt;ZmLOX2&lt;/i&gt; Modulates Jasmonate-Associated Antioxidant Defense and ROS Scavenging Under Osmotic Stress in Transgenic Arabidopsis. | europepmc | 2.0 | omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | True |
| `PMC:PMC9806556` | An extragenic second-site mutation in the jar1-1 mutant suppresses the response to photoperiod stress independent of jasmonic acid. | ncbi | 0.5 | context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reactio | False |
| `PMC:PMC13096701` | Interaction with CYP20-3 limits OPDA flux into jasmonate biosynthesis in wounded leaves of arabidopsis grown at eCO&lt;sub&gt;2&lt;/sub&gt;. | europepmc | 2.25 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | True |
| `PMC:PMC3694706` | Size of chloroplasts in Arabidopsis mesophyll cells affects jasmonate biosynthesis. | ncbi | 1.5 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | True |
| `PMC:PMC2716216` | Interaction with CYP20-3 limits OPDA flux into jasmonate biosynthesis in wounded leaves of arabidopsis grown at eCO2. | ncbi | 2.25 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | True |
| `PMID:40578603` | Effects of the Ib-miR168-3p-IbKTN80 module on jasmonate biosynthesis in the systemic leaves of sweet potato after wounding. | ncbi | 1.5 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | False |
| `PMC:PMC13373823` | Weaponizing nutrition: plants use a double strategy to fight herbivory, converting nutritionally essential fatty acids into defensive oxylipin signals. | europepmc | 2.5 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | False |
| `PMC:PMC8265181` | Arabidopsis Transcriptomics Reveals the Role of Lipoxygenase2 (AtLOX2) in Wound-Induced Responses. | ncbi | 0.5 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | True |
| `PMC:PMC10135161` | Oxylipins and Reactive Carbonyls as Regulators of the Plant Redox and Reactive Oxygen Species Network under Stress. | europepmc | -0.75 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC10132601` | Oxylipin biosynthetic gene families of Cannabis sativa. | europepmc | 0.5 | omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC6320827` | Overexpression of Arabidopsis <i>OPR3</i> in Hexaploid Wheat (<i>Triticum aestivum</i> L.) Alters Plant Development and Freezing Tolerance. | europepmc | 2.5 | omics_only: 'jasmonate biosynthesis, jasmonate pathway' appears in the title/abstract but | True |
| `PMC:PMC5538288` | An allene oxide and 12-oxophytodienoic acid are key intermediates in jasmonic acid biosynthesis by <i>Fusarium oxysporum</i>. | europepmc | 2.0 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMC:PMC13358708` | Size of chloroplasts in Arabidopsis mesophyll cells affects jasmonate biosynthesis. | europepmc | 1.5 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | True |
| `PMID:25752924` | Activated expression of AtEDT1/HDG11 promotes lateral root formation in Arabidopsis mutant edt1 by upregulating jasmonate biosynthesis. | ncbi | 0.75 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | True |
| `PMC:PMC1693950` | The NAC-like gene ANTHER INDEHISCENCE FACTOR acts as a repressor that controls anther dehiscence by regulating genes in the jasmonate biosynthesis pathway in Arabidopsis. | ncbi | 1.25 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | True |
| `PMC:PMC1803729` | Salicylate-mediated suppression of jasmonate-responsive gene expression in Arabidopsis is targeted downstream of the jasmonate biosynthesis pathway. | ncbi | 2.0 | context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reactio | True |
| `PMC:PMC11173247` | Arabidopsis Transcriptomics Reveals the Role of Lipoxygenase2 (AtLOX2) in Wound-Induced Responses. | europepmc | 0.5 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | True |
| `PMC:PMC13254203` | Genomic contraction of the &lt;i&gt;LOX&lt;/i&gt; gene family limits jasmonic acid biosynthesis and contributes to delayed flower bud opening in honeysuckle (&lt;i&gt;Lonicera japonica&lt;/i&gt;). | europepmc | 3.0 | omics_only: 'jasmonate biosynthetic' appears in the title/abstract but no strong reaction/ | False |
| `PMID:38522214` | pH change accompanying long-distance electrical signal controls systemic jasmonate biosynthesis. | ncbi | 2.0 | context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reactio | False |
| `PMC:PMC13238047` | Lipoxygenase 2 (LOX2) enhances cold tolerance through modulating jasmonate biosynthesis in Nicotiana tabacum. | europepmc | 1.0 | omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/ | False |
| `PMC:PMC12709120` | Genome-wide identification reveals unprecedented expansion and stress-responsive expression patterns of the lipoxygenase gene family in soybean (&lt;i&gt;Glycine max&lt;/i&gt; L.). | europepmc | 0.25 | omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |
| `PMID:31974860` | Light promotes jasmonate biosynthesis to regulate photomorphogenesis in Arabidopsis. | ncbi | 0.75 | context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reactio | True |
| `PMID:31481228` | Calcium-mediation of jasmonate biosynthesis and signaling in plants. | ncbi | 0.5 | context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reactio | False |
| `PMC:PMC5379949` | JASSY, a chloroplast outer membrane protein required for jasmonate biosynthesis. | ncbi | 1.5 | context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reactio | False |
| `PMC:PMC4844422` | Activity Regulation by Heteromerization of Arabidopsis Allene Oxide Cyclase Family Members. | europepmc | 0.25 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | True |
| `PMC:PMC3177274` | OPDA-Ile - a new JA-Ile-independent signal? | ncbi | -1.5 | off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract | False |

Most of these are rightly out: wrong organism, transgenic hosts, reviews, signalling, or transcriptomics. A curator would want three for *A. thaliana* jasmonate biosynthesis:

- `PMC:PMC4844422` 'Activity Regulation by Heteromerization of Arabidopsis Allene Oxide Cyclase Family Members': AOC enzymology, `off_topic` at 0.25. Route 1 never takes `off_topic`, and route 2 had no matching run term: 'allene oxide cyclase' is neither a seed nor a `target_species_supported` reaction name.
- The CYP20-3 / OPDA-flux paper (2.25), a documented borderline case.
- OPR3, `PMID:31834385` (`off_topic` at 1.5). Its title does not name the species, so route 2 cannot reach it.

The AtLOX2 transcriptomics paper is borderline.

## 4. `verbatim_cited_span` enzyme / gene values (bio watch item N-1)

**None.** No extracted claim (of 16) and no stored claim (of 11) has a field attributed `verbatim_cited_span` (`field_attribution`, both files). There is no row pairing to check. The raw lineage shows why. Every filled field in the 3 productive completions came with its own list: {"compartment:own list present": 14, "direction:own list present": 13, "enzyme:own list present": 9, "gene:own list present": 2, "species:own list present": 14}. The fallback only fires when a list is missing.

Participants are the exception. Prompt 1.1.0 asks every substrate and product for its own `evidence_ids`, but 30 of 32 raw participants have none; only the NAD+ claim's two do. `build_claims` never drops a participant, and the claim-level `evidence_ids` still carry the span, so nothing was lost. `evidence_policy.md` nonetheless describes participants as 'each participant with its own evidence ids' (A15).

## 5. Empty completions

Input sizes are exact: the frozen `extract_claims` rebuilt each call's system and user messages from `evidence_spans.jsonl` into a capturing stub. The shrink ladder truncates the user string to 100% / 60% / 35%, with the tight schema on rung 3 (`modelclient.py:50, 271-280`). `input_tokens` in `model_calls.jsonl` is the provider's total over every request of the call. The per-request token figure splits that total in proportion to characters sent; for single-request calls it is exact.

| call | round | paper | passages | user chars (rung 1) | est. input tok / request (rung 1) | rungs: fraction -> outcome (requests) | provider requests | input tok | output tok | output tok / request | response chars | claims | latency s | cost USD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mc_001_extract_claims_0001` | 1 | `PMC:PMC411572` | 2 | 4072 | 2355 | 1.0 full -> stop/ok (1) | 1 | 2355 | 2661 | 2661.0 | 774 | 1 | 22.7 | 0.001034 |
| `mc_001_extract_claims_0002` | 1 | `PMC:PMC13364333` | 20 | 91626 | 22903 | 1.0 full -> length/empty_completion (3); 0.6 full -> length/empty_completion (3); 0.35 tight -> length/empty_completion (3) | 9 | 135891 | 36000 | 4000.0 | 0 | 0 | 890.1 | 0.024389 |
| `mc_001_extract_claims_0003` | 1 | `PMC:PMC4793799` | 20 | 63586 | 17577 | 1.0 full -> length/empty_completion (3); 0.6 full -> length/empty_completion (3); 0.35 tight -> length/empty_completion (3) | 9 | 104913 | 36000 | 4000.0 | 0 | 0 | 293.4 | 0.021291 |
| `mc_001_extract_claims_0004` | 1 | `PMC:PMC12787732` | 16 | 57613 | 15000 | 1.0 full -> stop/ok (1) | 1 | 15000 | 3031 | 3031.0 | 9537 | 14 | 98.4 | 0.002409 |
| `mc_001_extract_claims_0005` | 1 | `PMC:PMC6628814` | 8 | 32820 | 8705 | 1.0 full -> stop/ok (1) | 1 | 8705 | 2050 | 2050.0 | 778 | 1 | 60.5 | 0.001485 |
| `mc_002_extract_claims_0006` | 2 | `PMC:PMC5382222` | 6 | 26112 | 7611 | 1.0 full -> length/empty_completion (3); 0.6 full -> stop/ok (1) | 4 | 27664 | 12497 | 3124.2 | 16 | 0 | 177.5 | 0.006515 |
| `mc_002_extract_claims_0007` | 2 | `PMC:PMC1955694` | 2 | 2752 | 1166 | 1.0 full -> stop/ok (1) | 1 | 1166 | 6 | 6.0 | 14 | 0 | 4.7 | 0.000118 |
| `mc_002_extract_claims_0008` | 2 | `PMID:12777050` | 2 | 4012 | 1468 | 1.0 full -> stop/ok (1) | 1 | 1468 | 940 | 940.0 | 14 | 0 | 28.8 | 0.000429 |
| `mc_002_extract_claims_0009` | 2 | `PMC:PMC3000969` | 2 | 4201 | 1907 | 1.0 full -> stop/ok (2) | 2 | 3814 | 1621 | 810.5 | 14 | 0 | 318.3 | 0.000868 |
| `mc_002_extract_claims_0010` | 2 | `PMC:PMC5338657` | 2 | 3629 | 1853 | 1.0 full -> stop/ok (1) | 1 | 1853 | 611 | 611.0 | 14 | 0 | 6.1 | 0.000369 |

**What the raw lineage shows.** For each failed rung, `rounds/round_001/model_raw.jsonl` holds a row like:

```json
{"attempt": 1, "call_id": "mc_001_extract_claims_0002", "call_status": "empty_completion", "error": "empty completion (finish_reason=length)", "finish_reason": "length", "model": "deepseek/deepseek-v4-flash", "paper_id": "PMC:PMC13364333", "parse_outcome": "empty_completion", "purpose": "extract_claims", "raw_text": "", "response_hash": "", "round_index": 1}
```

There are 7 such rows: rungs 1-3 of `_0002` and `_0003`, and rung 1 of `_0006`. Every one has `raw_text: ""`. Not one character of JSON came back, so this is not truncated JSON. `_0006` rung 2 returned `{ "claims": [] }` (16 chars) and the four round-2 abstract calls returned `{"claims": []}`.

**Classification: output-budget exhaustion. The reasoning spent `max_tokens` before any content.**
- `_0002` and `_0003` each billed exactly 36000 and 36000 output tokens over 9 requests: 4,000 per request, the configured `max_tokens`. Each ended `length` with an empty `content`. A request that had begun emitting JSON would have returned a truncated, non-empty string (`truncated_completion`), and none did. The model config sets `reasoning: ""`, so no reasoning budget or effort is passed. Only visible `content` is read (`chat_detailed`).
- `_0006`: 3 x 4,000 on rung 1, then 497 output tokens for 16 characters on rung 2. Successful calls show the same reasoning overhead. `_0008` spent 940 output tokens and `_0010` 611 to say `{"claims": []}` (14 chars). `_0001` spent 2,661 on 774 chars.
- There is no provider error and no content filter: every empty `finish_reason` is `length`. The one `error` belongs to `_0009` and was retried successfully.

**Is it concentrated on large inputs?** Partly, but input size is not the cause.
- The two calls that failed outright are the two largest inputs: 20 passages each, about 22,903 and 17,577 tokens per request.
- Their 35% rungs, about 8,410 and 6,583 tokens, still failed. `_0004` succeeded at 15,000 tokens (16 passages, 14 claims), and `_0005` at 8,705.
- `_0006` failed at about 7,611 tokens.
- The ladder shrinks the *input*, but the budget went on reasoning and required output. Rung 3's tight schema does not cap the reasoning either. Truncating the user string keeps its first passages, which are the dense ones.

**Shrink ladder.** Three calls climbed it: `_0002` and `_0003` (all 3 rungs, all empty) and `_0006` (rung 2 succeeded). **No call produced a claim after shrinking.** `_0006`'s rung-2 answer was `{ "claims": [] }` for a wheat transcriptome paper.

**Pilot-001 comparison (n = 1, and the stack is non-deterministic at temperature 0).** Pilot-001 sent the **same 20 evidence ids** of PMC13364333 (set-equal: True) under prompt `mining-extract/1.0.0`. It got `stop` in one request: 22,843 in, 3,629 out, 10,814 chars, 21 claims. That run was already at 91% of the 4,000-token cap. Prompt 1.1.0 asks for a citation list per field, which adds output and bookkeeping for a 20-reaction review. Every one of 9 requests then ran out.

**Cost of the failure.** `_0002` + `_0003`: $0.04568 (78% of the run's estimate) and 1183.5 s of provider latency (58% of wall clock). `_0002` alone took 890 s. The two most pathway-dense papers of round 1 contributed nothing.

## 6. Search

| round | queries | new papers discovered (by source) | included | processed |
|---|---|---|---|---|
| 1 | 3 | 66 {"europepmc": 36, "ncbi": 30} | 5 | 5 |
| 2 | 55 | 84 {"europepmc": 31, "ncbi": 53} | 6 | 5 |
| 3 | 0 | 0 {} | 0 | 0 |

`queries.jsonl` has 58 rows and 50 distinct query ids (A10). `cache_hit` is true on 0 rows (A13).

**Per frontier item** (attempt 1 = the item's own terms AND the species; attempt 2 = plus the topic; 'wider window' = attempt 1's text again over 2 x 25 when the topic is already a term):

| round | kind | target | attempts | windows | results | new discovered | new included | outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | seed_expansion | 12-OPDA | attempt 1 | [25] | [25] | 25 | PMC:PMC411572 | open |
| 2 | seed_expansion | 12-OPDA | attempt 1, attempt 2 | [25, 25] | [25, 25] | 18 | PMC:PMC5382222 | open |
| 1 | seed_expansion | AOS | attempt 1 | [25] | [25] | 21 | PMC:PMC13364333, PMC:PMC4793799 | open |
| 2 | seed_expansion | AOS | attempt 1, attempt 2 | [25, 25] | [25, 25] | 11 | PMC:PMC1955694, PMID:12777050 | open |
| 1 | seed_expansion | jasmonate biosynthesis | attempt 1 | [25] | [25] | 20 | PMC:PMC12787732, PMC:PMC6628814 | open |
| 2 | seed_expansion | jasmonate biosynthesis | attempt 1, wider window 50 | [25, 50] | [25, 50] | 19 | PMC:PMC3000969, PMC:PMC5104190, PMC:PMC5338657 | open |
| 2 | bridge_search | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) \| 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | attempt 1 | [25] | [0] | 0 | - | zero results (exhausted at once) |
| 2 | bridge_search | JA-Ile-bound SCF COI1 complex \| 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | attempt 1 | [25] | [0] | 0 | - | zero results (exhausted at once) |
| 2 | bridge_search | α-linolenic acid \| 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | attempt 1, attempt 2 | [25, 25] | [2, 1] | 1 | - | attempts found no new included paper |
| 2 | downstream_expansion | 13-hydroperoxy-α-linolenic acid | attempt 1, attempt 2 | [25, 11] | [25, 11] | 24 | - | attempts found no new included paper |
| 2 | downstream_expansion | 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | attempt 1, attempt 2 | [11, 11] | [2, 1] | 0 | - | every attempt's window was cut to 11/11 by the remaining discovery budget |
| 2 | downstream_expansion | JA-Ile-bound SCF COI1 complex | attempt 1, attempt 2 | [11, 1] | [11, 1] | 10 | - | every attempt's window was cut to 11/1 by the remaining discovery budget |
| 2 | upstream_expansion | α-linolenic acid | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | upstream_expansion | α-linolenic acid (LeA) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | missing_species_evidence | (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT) -> (9S,13S)-12-oxo-phytodienoic acid (OPDA) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | missing_species_evidence | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) -> (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | missing_species_evidence | (9S,13S)-12-oxo-phytodienoic acid (OPDA) -> OPC-8 | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | missing_species_evidence | jasmonic acid (JA) -> jasmonoyl-isoleucine (JA-Ile) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | missing_species_evidence | jasmonoyl-CoA -> jasmonic acid (JA) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | missing_species_evidence | OPC-8 -> jasmonoyl-CoA | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | missing_species_evidence | α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | missing_species_evidence | α-linolenic acid -> 13-hydroperoxy-α-linolenic acid | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | (9S,13S)-12-oxo-phytodienoic acid (OPDA) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | 13-hydroperoxy-α-linolenic acid | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | JA-Ile-bound SCF COI1 complex | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | jasmonic acid (JA) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | jasmonoyl-CoA | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | jasmonoyl-isoleucine (JA-Ile) | attempt 1, attempt 2 | [1, 1] | [1, 1] | 0 | - | every attempt's window was cut to 1/1 by the remaining discovery budget |
| 2 | branch_reaction | OPC-8 | attempt 1 | [1] | [1] | 1 | - | open |

Round-2 exhaustions: {"zero results": 2, "attempts found no new included paper": 2, "window cut by discovery budget": 21}. Round 2 had 44 items and 29 were asked. The 15 never asked are: branch_reaction: α-linolenic acid; branch_reaction: α-linolenic acid (LeA); enzyme_reactions: 13-lipoxygenase (13-LOX); enzyme_reactions: 13-lipoxygenase (LOX2); enzyme_reactions: 13S-lipoxygenase (13S-LOX); enzyme_reactions: acyl-CoA oxidase (ACX), multifunctional protein (MFP), 3-ketoacyl-CoA thiolase (KAT); enzyme_reactions: allene oxide cyclase (AOC); enzyme_reactions: allene oxide synthase (AOS); enzyme_reactions: jasmonate resistant 1 (JAR1); enzyme_reactions: OPDA reductase 3 (OPR3); missing_direction: α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT); missing_direction: α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT); missing_direction: α-linolenic acid -> 13-hydroperoxy-α-linolenic acid; missing_compartment: α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT); missing_compartment: α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT). Three of them are the enzyme items for AOS, AOC and OPR3.

**Per query** (window = what the controller asked for: `min(25, remaining)`, or `min(50, remaining)` for a wider window; 'filled' = results reached the window, so the true hit count is unknown):

| # | round | kind | target | attempt | window | results | filled | new discovered (by source) | new included | cumulative | query text |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | seed_expansion | 12-OPDA | 1 | 25 | 25 | yes | 25 {"europepmc": 15, "ncbi": 10} | PMC:PMC411572 | 25 | "12-OPDA" AND "Arabidopsis thaliana" |
| 2 | 1 | seed_expansion | AOS | 1 | 25 | 25 | yes | 21 {"ncbi": 11, "europepmc": 10} | PMC:PMC13364333, PMC:PMC4793799 | 46 | "AOS" AND "Arabidopsis thaliana" |
| 3 | 1 | seed_expansion | jasmonate biosynthesis | 1 | 25 | 25 | yes | 20 {"europepmc": 11, "ncbi": 9} | PMC:PMC12787732, PMC:PMC6628814 | 66 | "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 4 | 2 | bridge_search | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) \| 13-S-hyd | 1 | 25 | 0 | no | 0  | - | 66 | "(13S)-hydroperoxy-octadecatrienoic acid" AND "13-S-hydroperoxyoctadeca-9,11,15-trienoic acid" AND "Arabidopsis thaliana" |
| 5 | 2 | seed_expansion | 12-OPDA | 1 | 25 | 25 | yes | 0  | - | 66 | "12-OPDA" AND "Arabidopsis thaliana" |
| 6 | 2 | seed_expansion | 12-OPDA | 2 | 25 | 25 | yes | 18 {"europepmc": 18} | PMC:PMC5382222 | 84 | "12-OPDA" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 7 | 2 | seed_expansion | AOS | 1 | 25 | 25 | yes | 0  | - | 84 | "AOS" AND "Arabidopsis thaliana" |
| 8 | 2 | seed_expansion | AOS | 2 | 25 | 25 | yes | 11 {"europepmc": 4, "ncbi": 7} | PMC:PMC1955694, PMID:12777050 | 95 | "AOS" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 9 | 2 | seed_expansion | jasmonate biosynthesis | 1 | 25 | 25 | yes | 0  | - | 95 | "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 10 | 2 | seed_expansion | jasmonate biosynthesis | 2 (wider) | 50 | 50 | yes | 19 {"ncbi": 13, "europepmc": 6} | PMC:PMC3000969, PMC:PMC5104190, PMC:PMC5338657 | 114 | "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 11 | 2 | bridge_search | JA-Ile-bound SCF COI1 complex \| 13-S-hydroperoxyoctadeca-9,1 | 1 | 25 | 0 | no | 0  | - | 114 | "JA-Ile-bound SCF COI1 complex" AND "13-S-hydroperoxyoctadeca-9,11,15-trienoic acid" AND "Arabidopsis thaliana" |
| 12 | 2 | bridge_search | α-linolenic acid \| 13-S-hydroperoxyoctadeca-9,11,15-trienoic | 1 | 25 | 2 | no | 1 {"europepmc": 1} | - | 115 | "α-linolenic acid" AND "13-S-hydroperoxyoctadeca-9,11,15-trienoic acid" AND "Arabidopsis thaliana" |
| 13 | 2 | bridge_search | α-linolenic acid \| 13-S-hydroperoxyoctadeca-9,11,15-trienoic | 2 | 25 | 1 | no | 0  | - | 115 | "α-linolenic acid" AND "13-S-hydroperoxyoctadeca-9,11,15-trienoic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 14 | 2 | downstream_expansion | 13-hydroperoxy-α-linolenic acid | 1 | 25 | 25 | yes | 24 {"ncbi": 23, "europepmc": 1} | - | 139 | "13-hydroperoxy-α-linolenic acid" AND "Arabidopsis thaliana" |
| 15 | 2 | downstream_expansion | 13-hydroperoxy-α-linolenic acid | 2 | 11 | 11 | yes | 0  | - | 139 | "13-hydroperoxy-α-linolenic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 16 | 2 | downstream_expansion | 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | 1 | 11 | 2 | no | 0  | - | 139 | "13-S-hydroperoxyoctadeca-9,11,15-trienoic acid" AND "Arabidopsis thaliana" |
| 17 | 2 | downstream_expansion | 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | 2 | 11 | 1 | no | 0  | - | 139 | "13-S-hydroperoxyoctadeca-9,11,15-trienoic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 18 | 2 | downstream_expansion | JA-Ile-bound SCF COI1 complex | 1 | 11 | 11 | yes | 10 {"ncbi": 10} | - | 149 | "JA-Ile-bound SCF COI1 complex" AND "Arabidopsis thaliana" |
| 19 | 2 | downstream_expansion | JA-Ile-bound SCF COI1 complex | 2 | 1 | 1 | yes | 0  | - | 149 | "JA-Ile-bound SCF COI1 complex" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 20 | 2 | upstream_expansion | α-linolenic acid | 1 | 1 | 1 | yes | 0  | - | 149 | "α-linolenic acid" AND "Arabidopsis thaliana" |
| 21 | 2 | upstream_expansion | α-linolenic acid | 2 | 1 | 1 | yes | 0  | - | 149 | "α-linolenic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 22 | 2 | upstream_expansion | α-linolenic acid (LeA) | 1 | 1 | 1 | yes | 0  | - | 149 | "α-linolenic acid" AND "Arabidopsis thaliana" |
| 23 | 2 | upstream_expansion | α-linolenic acid (LeA) | 2 | 1 | 1 | yes | 0  | - | 149 | "α-linolenic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 24 | 2 | missing_species_evidence | (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S) | 1 | 1 | 1 | yes | 0  | - | 149 | "(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)" AND "(9S,13S)-12-oxo-phytodienoic acid" AND "Arabidopsis thaliana" |
| 25 | 2 | missing_species_evidence | (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S) | 2 | 1 | 1 | yes | 0  | - | 149 | "(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)" AND "(9S,13S)-12-oxo-phytodienoic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 26 | 2 | missing_species_evidence | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) -> (12S,13 | 1 | 1 | 1 | yes | 0  | - | 149 | "(13S)-hydroperoxy-octadecatrienoic acid" AND "(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)" AND "Arabidopsis thaliana" |
| 27 | 2 | missing_species_evidence | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) -> (12S,13 | 2 | 1 | 1 | yes | 0  | - | 149 | "(13S)-hydroperoxy-octadecatrienoic acid" AND "(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 28 | 2 | missing_species_evidence | (9S,13S)-12-oxo-phytodienoic acid (OPDA) -> OPC-8 | 1 | 1 | 1 | yes | 0  | - | 149 | "(9S,13S)-12-oxo-phytodienoic acid" AND "OPC-8" AND "Arabidopsis thaliana" |
| 29 | 2 | missing_species_evidence | (9S,13S)-12-oxo-phytodienoic acid (OPDA) -> OPC-8 | 2 | 1 | 1 | yes | 0  | - | 149 | "(9S,13S)-12-oxo-phytodienoic acid" AND "OPC-8" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 30 | 2 | missing_species_evidence | jasmonic acid (JA) -> jasmonoyl-isoleucine (JA-Ile) | 1 | 1 | 1 | yes | 0  | - | 149 | "jasmonic acid" AND "jasmonoyl-isoleucine" AND "Arabidopsis thaliana" |
| 31 | 2 | missing_species_evidence | jasmonic acid (JA) -> jasmonoyl-isoleucine (JA-Ile) | 2 | 1 | 1 | yes | 0  | - | 149 | "jasmonic acid" AND "jasmonoyl-isoleucine" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 32 | 2 | missing_species_evidence | jasmonoyl-CoA -> jasmonic acid (JA) | 1 | 1 | 1 | yes | 0  | - | 149 | "jasmonoyl-CoA" AND "jasmonic acid" AND "Arabidopsis thaliana" |
| 33 | 2 | missing_species_evidence | jasmonoyl-CoA -> jasmonic acid (JA) | 2 | 1 | 1 | yes | 0  | - | 149 | "jasmonoyl-CoA" AND "jasmonic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 34 | 2 | missing_species_evidence | OPC-8 -> jasmonoyl-CoA | 1 | 1 | 1 | yes | 0  | - | 149 | "OPC-8" AND "jasmonoyl-CoA" AND "Arabidopsis thaliana" |
| 35 | 2 | missing_species_evidence | OPC-8 -> jasmonoyl-CoA | 2 | 1 | 1 | yes | 0  | - | 149 | "OPC-8" AND "jasmonoyl-CoA" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 36 | 2 | missing_species_evidence | α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid | 1 | 1 | 1 | yes | 0  | - | 149 | "α-linolenic acid" AND "(13S)-hydroperoxy-octadecatrienoic acid" AND "Arabidopsis thaliana" |
| 37 | 2 | missing_species_evidence | α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid | 2 | 1 | 1 | yes | 0  | - | 149 | "α-linolenic acid" AND "(13S)-hydroperoxy-octadecatrienoic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 38 | 2 | missing_species_evidence | α-linolenic acid -> 13-hydroperoxy-α-linolenic acid | 1 | 1 | 1 | yes | 0  | - | 149 | "α-linolenic acid" AND "13-hydroperoxy-α-linolenic acid" AND "Arabidopsis thaliana" |
| 39 | 2 | missing_species_evidence | α-linolenic acid -> 13-hydroperoxy-α-linolenic acid | 2 | 1 | 1 | yes | 0  | - | 149 | "α-linolenic acid" AND "13-hydroperoxy-α-linolenic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 40 | 2 | branch_reaction | (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S) | 1 | 1 | 1 | yes | 0  | - | 149 | "(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)" AND "Arabidopsis thaliana" |
| 41 | 2 | branch_reaction | (12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S) | 2 | 1 | 1 | yes | 0  | - | 149 | "(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 42 | 2 | branch_reaction | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) | 1 | 1 | 1 | yes | 0  | - | 149 | "(13S)-hydroperoxy-octadecatrienoic acid" AND "Arabidopsis thaliana" |
| 43 | 2 | branch_reaction | (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) | 2 | 1 | 1 | yes | 0  | - | 149 | "(13S)-hydroperoxy-octadecatrienoic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 44 | 2 | branch_reaction | (9S,13S)-12-oxo-phytodienoic acid (OPDA) | 1 | 1 | 1 | yes | 0  | - | 149 | "(9S,13S)-12-oxo-phytodienoic acid" AND "Arabidopsis thaliana" |
| 45 | 2 | branch_reaction | (9S,13S)-12-oxo-phytodienoic acid (OPDA) | 2 | 1 | 1 | yes | 0  | - | 149 | "(9S,13S)-12-oxo-phytodienoic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 46 | 2 | branch_reaction | 13-hydroperoxy-α-linolenic acid | 1 | 1 | 1 | yes | 0  | - | 149 | "13-hydroperoxy-α-linolenic acid" AND "Arabidopsis thaliana" |
| 47 | 2 | branch_reaction | 13-hydroperoxy-α-linolenic acid | 2 | 1 | 1 | yes | 0  | - | 149 | "13-hydroperoxy-α-linolenic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 48 | 2 | branch_reaction | 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | 1 | 1 | 1 | yes | 0  | - | 149 | "13-S-hydroperoxyoctadeca-9,11,15-trienoic acid" AND "Arabidopsis thaliana" |
| 49 | 2 | branch_reaction | 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | 2 | 1 | 1 | yes | 0  | - | 149 | "13-S-hydroperoxyoctadeca-9,11,15-trienoic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 50 | 2 | branch_reaction | JA-Ile-bound SCF COI1 complex | 1 | 1 | 1 | yes | 0  | - | 149 | "JA-Ile-bound SCF COI1 complex" AND "Arabidopsis thaliana" |
| 51 | 2 | branch_reaction | JA-Ile-bound SCF COI1 complex | 2 | 1 | 1 | yes | 0  | - | 149 | "JA-Ile-bound SCF COI1 complex" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 52 | 2 | branch_reaction | jasmonic acid (JA) | 1 | 1 | 1 | yes | 0  | - | 149 | "jasmonic acid" AND "Arabidopsis thaliana" |
| 53 | 2 | branch_reaction | jasmonic acid (JA) | 2 | 1 | 1 | yes | 0  | - | 149 | "jasmonic acid" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 54 | 2 | branch_reaction | jasmonoyl-CoA | 1 | 1 | 1 | yes | 0  | - | 149 | "jasmonoyl-CoA" AND "Arabidopsis thaliana" |
| 55 | 2 | branch_reaction | jasmonoyl-CoA | 2 | 1 | 1 | yes | 0  | - | 149 | "jasmonoyl-CoA" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 56 | 2 | branch_reaction | jasmonoyl-isoleucine (JA-Ile) | 1 | 1 | 1 | yes | 0  | - | 149 | "jasmonoyl-isoleucine" AND "Arabidopsis thaliana" |
| 57 | 2 | branch_reaction | jasmonoyl-isoleucine (JA-Ile) | 2 | 1 | 1 | yes | 0  | - | 149 | "jasmonoyl-isoleucine" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana" |
| 58 | 2 | branch_reaction | OPC-8 | 1 | 1 | 1 | yes | 1 {"europepmc": 1} | - | 150 | "OPC-8" AND "Arabidopsis thaliana" |

**What stopped the run.** The frontier did not run out: 19 items were open at the stop, 15 of them never asked. The binding limit was `max_papers_discovered`.
- Round 1's 3 seed queries filled their 25-result windows and brought in 66 papers.
- Round 2's first 15 queries took the count to 139. Query 18 (executed 20:10:49, 149 papers) left one slot.
- Every later query ran with a window of 1 (40 queries in 61 s). Each returned the single top hit, already seen, until query 58 found paper 150 at 20:11:52, 2037 s after the start.
- `check_papers_discovered` then recorded the stop. Round 3 started anyway, broke on its first item and closed as an unproductive round (A5).

So yes, 150 is now the binding stop. In effect it was reached at query 18, about 1,974 s in. Most of the elapsed time before that was the two failed extraction calls, not search.

**What the new search did right (item 3).** AND formulation: 2 of the 3 seed items gained a second formulation, and the topic seed a wider window. NCBI is reached: 83 of 150 papers, 8 of 11 included. Every query records its attempt. Round 2 discovered 84 new papers and included 6. Pilot-001 found 9 new and included 0. The bridge queries are unanswerable by construction here: they AND two synonyms of the same compound, so 2 of 3 returned zero (A9).

## 7. Side by side with pilot-001

Both columns are computed by the same function from each run directory (`pilot/001/run/` for pilot-001).

| | pilot-001 | pilot-002 |
|---|---|---|
| code | 23e1ae83 | ecf6b9a0 |
| prompt_version | mining-extract/1.0.0 | mining-extract/1.1.0 |
| stop_reason | frontier_exhausted | max_papers_discovered_reached |
| rounds | 2 | 3 |
| elapsed_s | 303.704 | 2039.18 |
| queries (rows) | 63 | 58 |
| papers discovered | 21 | 150 |
| papers screened | 21 | 150 |
| papers included | 1 | 11 |
| by production / route 1 / route 2 | 1 / 0 / 0 | 5 / 5 / 1 |
| papers processed | 1 | 10 |
| passages | 20 | 80 |
| claims extracted | 21 | 16 |
| distinct identities | 19 | 11 |
| duplicates merged | 2 | 5 |
| primary reactions | 0 | 1 |
| candidate reactions | 19 | 10 |
| candidate components | 14 | 3 |
| rejected (status) | {} | {} |
| species states (nodes) | {"in_vitro_or_cell_free": 1, "inferred_only": 10, "species_not_stated": 8} | {"in_vitro_or_cell_free": 9, "species_not_stated": 1, "target_species_supported": 1} |
| model calls (logical, by status) | {"ok": 1} | {"empty_completion": 2, "ok": 8} |
| LLM calls (ladder rungs, budgeted) | 1 | 15 |
| input / output tokens | 22843 / 3629 | 302829 / 95417 |
| est. cost USD | 0.003373 | 0.058908 |
| uncited field drops | 59 | 0 |
| fields kept: model_citation / verbatim_cited_span | 0 / 0 (field_attribution did not exist; 0 fields kept) | 53 / 0 |
| evidence type of included | {"unknown": 1} | {"unknown": 11} |
| empty rungs / 'giving up' log lines / provider requests | 0 / 0 / 1 | 7 / 7 / 30 |
| distinct articles among discovered | 21 | 144 |
| NCBI papers discovered | 0 (never reached, F2) | 83 |

## 8. Anomalies

### Pilot-001 findings F1-F10: do they still reproduce?

| id | pilot-001 finding | pilot-002 |
|---|---|---|
| F1 | every enzyme/gene/species dropped as uncited (59/59) | **Resolved.** 0 drops; 53 fields kept, all `model_citation`; the raw lineage shows every filled field with its own list. The fallback was never needed. |
| F2 | round 2 found nothing new: OR'd query, top-10 cap with NCBI never reached, one-shot exhaustion | **Mostly resolved, one new failure mode.** AND queries, 25-result windows, NCBI reached (83 papers), 2 attempts plus a wider window; round 2 found 84 new papers, 6 included. But the discovery budget cut windows to 1, so 21 items were exhausted on starved windows (A6). NCBI brought the identity defect (A1). |
| F3 | species labels by fall-through / wrong reasons | **Reproduces in a new form.** 9 pepper reactions `in_vitro_or_cell_free` from an incidental sentence, although for 8 of them the model cited *Capsicum annuum*; the primary's kinetic evidence passes the in-vitro test (A3). |
| F4 | frontier cap silently cut 62 items | **Not exercised.** The round-2 frontier was 44 and the round-3 frontier 19, both under the cap of 60. No cut. |
| F5 | screen missed AtLOX2 enzymology, 12-OPDA metabolites, DAD1 | **Partly resolved.** The rescue re-admitted 6 (route 1: 5, route 2: 1), and route 2 produced the only primary reaction. Still out: Arabidopsis AOC heteromerization `PMC:PMC4844422` (`off_topic` 0.25), the CYP20-3 / OPDA-flux paper (2.25, documented borderline) and OPR3 `PMID:31834385` (`off_topic` 1.5). One route-1 false rescue: grape (A8). |
| F6 | review labelled `unknown`; PMID and DOI empty | **Reproduces.** All 11 included are `unknown`, including the PMC13364333 review; every included record has `doi: ""`, and NCBI records carry a wrong PMCID and URL (A1). |
| F7 | report.md misstates claims extracted and primary-graph connections/entities | **Reproduces, plus two more.** 'Claims extracted: 11' (16); 'Primary graph: 1 reactions, 11 chemical connections, 20 entities' (the primary has 0 and 2); 'Model calls: 15' (10 calls, 30 requests); 'generated but never asked' lists 4 items that were asked (A12). |
| F8 | cache_hit never set; screened-out papers missing from round provenance; retries semantics; relevance 0.0; abstract span with null offsets / HTML | **Reproduces.** `cache_hit` false on 58/58; round metrics `papers_discovered` 5 and 6 (usage 66 and 84), with `papers_considered` and `new_paper_ids` included-only; `retries` is the max inner attempt count; all 3 candidate components have relevance 0.0; `ev_49c948bc587a2016` has null offsets; titles keep HTML entities (`&lt;i&gt;`). |
| F9 | mining CLI does not load .env | **Still present; not exercised.** The launcher was used again. `flags.mining_enabled` returns before `ensure_dotenv_loaded()` whenever the CLI flag supplies an override (`flags.py:55-62`), and nothing else in `t2pw.mining` loads .env. |
| F10 | cli.md shows the flag after the subcommand | **Mostly fixed.** `cli.md:4, 11, 20` are right; the `preflight` example at `cli.md:109` still puts `--enable-pathway-mining` after the subcommand. |

### New anomalies

| id | severity | finding | evidence | code / doc |
|---|---|---|---|---|
| A1 | high | NCBI records take their PMCID and DOI from the last cited reference, not the article | Frozen parser on a synthetic record returns PMC411572, the reference's id. In this run: 6 articles discovered twice, and each NCBI copy's PMCID differs from EuropePMC's. Of the 8 included NCBI papers, 2 full-text fetches returned text, and both were another article, which extraction then read under the included title (the other 6 returned no text). The only primary reaction cites `PMC:PMC411572` / `10.1073/pnas.76.9.4350` (PNAS 1979) for a 2021 abstract. 150 discovered = 144 articles. The NAD+ article was excluded via EuropePMC and included via NCBI. Class: `product_contract_violation` (provenance). Shared production code, latent until NCBI was reached | `src/t2pw/rag/acquire.py:536-543` (`findall('.//ArticleIdList/ArticleId')`); `mining/acquisition.py:528-` `Acquirer.acquire` (dedup by `paper_id` only) |
| A2 | high | Extraction of dense passages fails: reasoning exhausts max_tokens, and the ladder cannot fix it | 2 calls, 18 requests, each exactly 4,000 output tokens with empty content; 35% rungs about 8.4k / 6.6k tokens failed while 15k succeeded elsewhere. $0.04568 and 1183.5 s for 0 claims. Same passages succeeded under prompt 1.0.0 in pilot-001. The inner client re-sends identical requests 3 times per rung: 30 provider requests against a budgeted 'llm_calls' of 15 | `mining/config.py:148, 158` (`max_tokens` 4000, `reasoning` ""); `modelclient.py:50, 271-280`; `llm/client.py:682-756` |
| A3 | medium | Species states from keyword tests contradict the cited evidence | Pepper introduction -> `in_vitro_or_cell_free` for 9 reactions because of 'in vitro studies confirmed ...' about AtLOX2 phosphomutants, although for 8 of them the model cited *Capsicum annuum*, which would be `non_target_species`. The primary's abstract (Vmax/Km, 'in cascade with 12-OPDA pathway enzymes') is purified-enzyme work that no marker catches. Outcome safe; labels wrong; the primary's state is a policy question | `mining/species.py:108-115, 276-281`; `species_policy.md` state table |
| A4 | medium | An abstract-only paper is indexed twice, so one passage counts as two spans | `ev_b58e2cb64e2f1d87` (abstract#0) and `ev_ba6fb9f5cdb8ce58` (body#0) have identical text. The same happens for every abstract-only paper: 2 identical passages each for all 5 abstract-only papers processed (1 in round 1, 4 in round 2). The primary's reason says 'explicitly supported by 2 evidence span(s)'. The model is also shown the same text twice | `mining/retrieval.py:170-183` (candidate built with `abstract` = the record abstract and `full_text` = the same abstract) |
| A5 | medium | A phantom round after the stop reason is recorded | `max_papers_discovered_reached` was recorded inside round 2, right after query 58. `check_before_round` checks neither discovered papers nor an existing `stop_reason`, so round 3 started with 19 items, broke on the first and closed as a completed, unproductive round: `rounds_completed=3`, unproductive streak 2. Had discovery not been recorded first, this empty round would itself have stopped the run as `converged_no_useful_additions`, naming the wrong cause. The controller docstring says 'stop at the first limit reached' | `mining/budget.py:193-235`; `mining/controller.py:319-331, 383` |
| A6 | medium | The discovery budget silently starves and exhausts frontier items | Windows are `min(25, remaining)`. After 149 papers, 40 queries asked for 1 result and 21 items were marked exhausted after two 1-result windows, the same look as a real search that found nothing. `limits_and_stopping.md` promises exhaustion only 'after every attempt is tried without a new paper', which is literally true | `mining/controller.py:403-408, 443-446` |
| A7 | medium | Paper-scoped retrieval is a store-wide top-60 filtered to one paper | `PMC:PMC5104190` got 0 passages: included, never processed (11 included / 10 processed). Passage counts per paper (2-20) depend on what else is in the store, not on `max_passages_per_paper=20` | `mining/retrieval.py:227-238` (`store.query(top_k=max(k*3, k))` then filter by `paper_ids`) |
| A8 | medium | Route-1 false rescue: target species only an ectopic-expression host | `PMC:PMC5104190` (grape PAT1): the abstract says 'ectopic expression of VaPAT1 enhances cold tolerance in Arabidopsis'; 'grape' is unknown to the title detectors; the foreign-host guard reads the title only. Inside the documented ceiling (`evidence_policy.md`, `_EXTRA_HOSTS`); no effect because it was never processed | `mining/acquisition.py:62-70, 426-448, 487` |
| A9 | medium | Lexical entity split disconnects the primary reaction and makes bridges unanswerable | '13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT)' and '(13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)' are two entities, as are 'α-linolenic acid (LeA)' and 'α-linolenic acid'. The primary has 0 connections. The bridge query ANDs the two synonyms (0 results, exhausted at once), and two upstream items ask the identical question. Documented as a v1 limitation (`known_limitations.md`); the AND bridge formulation is what makes it a guaranteed 0 | `mining/normalize.py` `comparison_key`; `mining/queries.py:84-108` (bridge requires both sides) |
| A10 | low | Query ids collide across frontier items | `query_id = hash(text, round)`: 8 ids appear twice in `queries.jsonl`, with different items and different result counts (e.g. `qr_faf2adeaf3e7810c`: 25 results for one item, 1 for the other). `discovered_via_query_id` alone is ambiguous | `mining/queries.py:184`; `mining/ids.py:82-83` |
| A11 | low | A nested trailing parenthetical is not stripped from a query term | '(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)' went verbatim into 4 queries. `limits_and_stopping.md`: 'A trailing parenthetical note ... is dropped from a term' | `mining/queries.py:77` (`\s+\([^()]*\)$`) |
| A12 | medium | report.md misstates numbers (F7 plus two) | 'Claims extracted: **11**' (16 extracted; 11 is the stored count). 'Primary graph: **1** reactions, **11** chemical connections, **20** entities' (whole graph; the primary has 0 connections and 2 entities). 'Model calls: **15**' (10 logical calls, 15 rungs, 30 requests). 'These questions were generated but never asked' lists the 3 seeds (asked 3 times each) and OPC-8 (asked once) | `mining/report.py:149, 159-160, 275, 299, 417` |
| A13 | low | Observability gaps in the lineage and round files | `model_raw.jsonl` carries no shrink fraction or chars sent; `model_calls.jsonl` `attempt_log` is overwritten per rung (3 of `_0002`'s 9 requests survive); `retries` is a max, not a count; `round.json` `input_frontier` is serialised after the round (round-1 seeds show `attempts: 1`); `cache_hit` false on 58/58; round metrics `papers_discovered` counts included papers only | `mining/modelclient.py:291-300, 371-374`; `mining/controller.py:552-566` |
| A14 | low | Evidence type and identifiers still missing | All 11 included papers `unknown`, including a review; `doi` empty on all 11 (only `url` carries it) | `mining/acquisition.py` `classify_evidence_type` / `record_from_candidate` |
| A15 | low | Participants are uncited, despite prompt rule 1 and `evidence_policy.md` | 30 of 32 raw substrate/product entries have no `evidence_ids`; `build_claims` never drops a participant | `mining/extraction.py:46-53, 237-259`; `evidence_policy.md:83` |
| A16 | low | Documentation drift | `artifacts.md` gives `PROMPT_VERSION` as `mining-extract/1.0.0` (code: 1.1.0); `cli.md:109` flag order (F10); `ids.py` docstring and `resume_and_cancellation.md` say no id uses a counter, but `call_id` uses a per-process ordinal (a resume would reuse `mc_..._0001`; not exercised) | `docs/pathway_mining/artifacts.md`; `mining/ids.py:1-6, 94-95`; `mining/modelclient.py:242-243` |
| A17 | info | One completion repeated its own claims, and the same step became two nodes | `_0004` emitted 14 claims for 9 identities: 5 in-call duplicates, merged correctly. 'α-linolenic acid -> 13-HPOT' from the same sentence became two candidate nodes (`rx_b30980df6dd1f29b4504` with no species, `rx_c1d050d4171b110ad2d4` Capsicum) because species is part of identity | `mining/extraction.py` identity; `mining/claims.py` |

## Appendix A: cited evidence spans, verbatim

Exact `text` from `evidence_spans.jsonl` for every span a reaction cites.

### `ev_b58e2cb64e2f1d87`: `PMC:PMC411572`, abstract `abstract#0`, chars 0-1775

```text
13-lipoxygenases (13-LOX) catalyze the dioxygenation of various polyunsaturated fatty acids (PUFAs), of which α-linolenic acid (LeA) is converted to 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT), the precursor for the prostaglandin-like plant hormones cis-(+)-12-oxophytodienoic acid (12-OPDA) and methyl jasmonate (MJ). This study aimed for characterizing the four annotated A. thaliana 13-LOX enzymes (LOX2, LOX3, LOX4, and LOX6) focusing on synthesis of 12-OPDA and 4Z,7Z,10Z)-12-[[-(1S,5S)-4-oxo-5-(2Z)-pent-2-en-1yl] cyclopent-2-en-1yl] dodeca-4,7,10-trienoic acid (OCPD). In addition, we performed interaction studies of 13-LOXs with ions and molecules to advance our understanding of 13-LOX. Cell imaging indicated plastid targeting of fluorescent proteins fused to 13-LOXs-N-terminal extensions, supporting the prediction of 13-LOX localization to plastids. The apparent maximal velocity (Vmax&nbsp;app) values for LOX-catalyzed LeA oxidation were highest for LOX4 (128 nmol·s-1·mg protein-1), with a Km value of 5.8 µM. A. thaliana 13-LOXs, in cascade with 12-OPDA pathway enzymes, synthesized 12-OPDA and OCPD from LeA and docosahexaenoic acid, previously shown only for LOX6. The activities of the four isoforms were differently affected by physiologically relevant chemicals, such as Mg2+, Ca2+, Cu2+ and Cd2+, and by 12-OPDA and MJ. As demonstrated for LOX4, 12-OPDA inhibited enzymatic LeA hydroperoxidation, with half-maximal enzyme inhibition at 48 µM. Biochemical interactions, such as the sensitivity of LOX toward thiol-reactive agents belonging to cyclopentenone prostaglandins, are suggested to occur in human LOX homologs. Furthermore, we conclude that 13-LOXs are isoforms with rather specific functional and regulatory enzymatic features.
```

### `ev_ba6fb9f5cdb8ce58`: `PMC:PMC411572`, body `body#0`, chars 0-1775

```text
13-lipoxygenases (13-LOX) catalyze the dioxygenation of various polyunsaturated fatty acids (PUFAs), of which α-linolenic acid (LeA) is converted to 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT), the precursor for the prostaglandin-like plant hormones cis-(+)-12-oxophytodienoic acid (12-OPDA) and methyl jasmonate (MJ). This study aimed for characterizing the four annotated A. thaliana 13-LOX enzymes (LOX2, LOX3, LOX4, and LOX6) focusing on synthesis of 12-OPDA and 4Z,7Z,10Z)-12-[[-(1S,5S)-4-oxo-5-(2Z)-pent-2-en-1yl] cyclopent-2-en-1yl] dodeca-4,7,10-trienoic acid (OCPD). In addition, we performed interaction studies of 13-LOXs with ions and molecules to advance our understanding of 13-LOX. Cell imaging indicated plastid targeting of fluorescent proteins fused to 13-LOXs-N-terminal extensions, supporting the prediction of 13-LOX localization to plastids. The apparent maximal velocity (Vmax&nbsp;app) values for LOX-catalyzed LeA oxidation were highest for LOX4 (128 nmol·s-1·mg protein-1), with a Km value of 5.8 µM. A. thaliana 13-LOXs, in cascade with 12-OPDA pathway enzymes, synthesized 12-OPDA and OCPD from LeA and docosahexaenoic acid, previously shown only for LOX6. The activities of the four isoforms were differently affected by physiologically relevant chemicals, such as Mg2+, Ca2+, Cu2+ and Cd2+, and by 12-OPDA and MJ. As demonstrated for LOX4, 12-OPDA inhibited enzymatic LeA hydroperoxidation, with half-maximal enzyme inhibition at 48 µM. Biochemical interactions, such as the sensitivity of LOX toward thiol-reactive agents belonging to cyclopentenone prostaglandins, are suggested to occur in human LOX homologs. Furthermore, we conclude that 13-LOXs are isoforms with rather specific functional and regulatory enzymatic features.
```

### `ev_f7c56801adf6792d`: `PMC:PMC12787732`, introduction `introduction#0`, chars 3467-8329

```text
troduction Jasmonates are a family of oxylipin signaling molecules that regulate plant development processes, including flowering time and senescence [ 1 , 2 , 3 ], and stress responses [ 4 , 5 , 6 ]. In response to mechanical damage, such as that incurred by chewing insect herbivory, jasmonate signaling gives rise to approximately 95% of the wound-induced protein synthesis that leads to the production of protective plant specialized metabolites and morphological barriers [ 7 , 8 ]. Given the importance of this pathway in plant resistance to pathogens and insect herbivores, understanding its regulation is key and may lead to novel strategies to enhance crop productivity [ 9 ]. After wounding or recognition of a necrotrophic pathogen, jasmonate biosynthesis starts in the chloroplast, where α -linolenic acid, a substrate of lipoxygenases (LOXs), is released from the galactolipids of the chloroplast membrane by a galactolipase [ 10 ]. The fatty acid undergoes oxidation, catalyzed by a member of the 13 S -LOX class of lipoxygenases, producing (13 S )-hydroperoxy-octadecatrienoic acid (13-HPOT) [ 11 ]. At this point in the pathway, different oxylipins may be produced. The reaction that converts 13-HPOT to (12 S -13 S )-epoxy-hydroperoxy-octadecatrienoic acid ((12 S -13 S )-EOT) is catalyzed by allene oxide synthase (AOS) and commits the pathway towards jasmonate biosynthesis [ 12 ]. (12 S -13 S )-EOT is reconfigured by allene oxide cyclase (AOC) into the more stable (9 S ,13 S )-12- oxo -phytodienoic acid (OPDA), a bioactive jasmonate [ 13 , 14 , 15 ]. OPDA is transported from the chloroplast into the cytosol and then into the peroxisome [ 16 , 17 ]. In this organelle, jasmonate biosynthesis continues with the reduction of OPDA by OPDA reductase 3 (OPR3) [ 18 , 19 , 20 ]. After esterification to CoA [ 21 ], OPC-8 undergoes β -oxidation by the fatty acid β -oxidation machinery composed of acyl-CoA oxidase (ACX), multifunctional protein (MFP), and 3-ketoacyl CoA thiolase (KAT) [ 22 , 23 , 24 , 25 , 26 , 27 ]. The product, jasmonoyl-CoA, is transformed into jasmonic acid (JA), which moves into the cytosol [ 28 , 29 , 30 ]. There, it may be converted into its biologically active form, jasmonoyl-isoleucine (JA-Ile), by jasmonate resistant 1 (JAR1) [ 31 ]. After entering the nucleus [ 32 ], JA-Ile binds to the Skp1/Cullin1/F-box protein coronatine-insensitive 1 (SCF CO1 ) complex, bridging the ubiquitin E3 ligase with its target repressor jasmonate ZIM-domain (JAZ) proteins, leading to its degradation through the 26 S -proteasome, thus freeing MYC2, MYC3, or MYC4 transcription factors from repression and allowing JA-Ile-responsive gene expression to proceed [ 32 , 33 , 34 , 35 , 36 ]. Targets of these transcription factors include genes that encode enzymes in the jasmonate biosynthetic pathway, leading to feedforward regulation [ 37 ]. In unwounded leaves, the basal JA level in Arabidopsis thaliana (arabidopsis) is less than 30 pmol/gm fresh mass [ 38 ]. After wounding, JA levels increase within seconds after mechanical damage, reaching levels that are over 10-fold higher than those of unwounded leaves within 2 min. This response is too rapid to be attributed to the increase in the expression of genes encoding enzymes in the jasmonate biosynthesis pathway. This indicates that the pathway must be constitutively present and activated upon damage [ 10 , 39 , 40 ]. Thivierge et al. [ 41 ] found that AtLOX2, which catalyzes an early step in jasmonate biosynthesis, is constitutively phosphorylated on Serine 600 (Ser 600 ) and dephosphorylated in response to wounding. Using phosphovariants, in vitro studies confirmed that the unphosphorylatable AtLOX2 had over a 100 times increase in V max compared to the phosphomimic, which likely reflects blocking of the substrate from the catalytic site [ 42 ]. Therefore, wounding stress is proposed to lead to the dephosphorylation of AtLOX2, contributing to the rapid increase in JA-Ile biosynthesis. In arabidopsis AtLOX2, the phosphosite, where the Ser 600 is shown in red, is flanked by the amino acid sequence ARQ S LVNG [ 42 ]. The sequence ARQ S LVNG is relatively conserved among plant 13 S -LOXs; however, there are variations at the phosphosite Ser, where Ala or Ile occupy this position, generating non-phosphorylatable variants. We chose pepper ( Capsicum annuum ) as our study organism since pepper has four 13 S -LOXs [ 43 ]; CaLOX2 and CaLOX8 have a Ser at the phosphosite, whereas CaLOX6 and CaLOX7 have an unphosphorylatable Ala at this site, suggesting that these proteins may be regulated at the post-translational or transcriptional levels, respectively (Note: Pepper LOXs have been color-coded; those with an Ala at the phosphosite ( CaLOX6 and CaLOX7 ) in orange and those with a Ser at this site ( CaLOX2 and CaLOX8 ) in blue). Alternative splicing
```

### `ev_49c948bc587a2016`: `PMC:PMC6628814`, abstract `abstract#0`, chars None-None

```text
Nicotinamide adenine dinucleotide (NAD+) is a redox cofactor and signal central to cell metabolisms. Disrupting NAD homeostasis in plant alters growth and stress resistance, yet the underlying mechanisms remain largely unknown. Here, by combining genetics with multi-omics, we discover that NAD+ deficiency in qs-2 caused by mutation in NAD+ biosynthesis gene-Quinolinate Synthase retards growth but induces biosynthesis of defense compounds, notably aliphatic glucosinolates that confer insect resistance. The elevated defense in qs-2 is resulted from activated jasmonate biosynthesis, critically hydroperoxidation of α-linolenic acid by the 13-lipoxygenase (namely LOX2), which is escalated via the burst of chloroplastic ROS-singlet oxygen (1O2). The NAD+ deficiency-mediated JA induction and defense priming sequence in plants is recapitulated upon insect infestation, suggesting such defense mechanism operates in plant stress response. Hence, NAD homeostasis is a pivotal metabolic checkpoint that may be manipulated to navigate plant growth and defense metabolism for stress acclimation.
```

## Appendix B: files

| file | what |
|---|---|
| `build_audit.py` | computes every number and writes both outputs |
| `md_text.py` | this document's narrative; reads the computed result |
| `pilot_002_audit.json` | the machine-readable twin (all tables above, uncut) |
| `explore.py`, `peek.py`, `peek2.py`, `fix_pointers.py` | scratch readers and a one-off text patch used while writing; not needed to rebuild |
| `g11/*.json` | `bounded_run.py` reports for every script run; all show 0 survivors |

Rebuild (from `C:/t/mining-pilot-runs2/audit`):

```bash
PYTHONPATH=C:/t/mining-pilot2/src PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 \
  <venv>/python.exe C:/t/mining-pilot2/docs/pwml_recovery_sprint/evidence/bounded_run.py --timeout 300 \
  --label audit-build --json g11/build_audit.g11.json -- <venv>/python.exe build_audit.py
```
