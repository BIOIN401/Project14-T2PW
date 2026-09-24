# Pathway-mining pilot 001 — result

**Verdict for the mining lane (`ORCH-742a`): BLOCKED for evaluation.** The feature ran end to end
against live services and did nothing unsafe. It invented no reaction, and the species guardrail kept
every wrong-species reaction out of the target graph. But the run **produced no Arabidopsis
evidence at all**, and two product defects (below) mean more runs of the same code would mostly
repeat that. Per the pre-registration, this single run is the result. **It was not re-run.**

## What ran

| | |
|---|---|
| pre-registration | `LEDGER.md`, `ORCH-742a` amendment, committed at `08db8199` **before** any model call |
| code | `experiment/pathway-mining` @ `23e1ae83` (both stability fixes; reviewer `APPROVE_WITH_FINDINGS`), frozen detached worktree |
| inputs | *Arabidopsis thaliana* · topic `jasmonate biosynthesis` · seed metabolite `12-OPDA` · seed enzyme `AOS` · no seed papers |
| model / embeddings | `deepseek/deepseek-v4-flash`, temperature 0 · LM Studio `text-embedding-nomic-embed-text-v1.5` (live; `PATHWAY_MINING_REQUIRE_EMBEDDINGS=true`) |
| budgets | branch defaults, none hit |
| process | `bounded_run.py`, 304.38 s, exit 5 (`stopped_at_limit`), **FINAL SURVIVING COUNT 0**, no heavy lock. The mining process's working set was about 104 MB during the run |
| cost | 1 extraction call · 22,843 in / 3,629 out tokens · **est. $0.0034** (an estimate, not a bill). The preflight probes (one in `g11/002`, one inside the run) are not counted in `costs.json` |

### One deviation, recorded

The first live preflight (`g11/001`) failed on `model.provider: LLM_PROVIDER is unset`. **The mining CLI
never loads the project `.env`.** Its preflight reads `LLM_PROVIDER` / `OPENROUTER_API_KEY` before anything
populates them. The Streamlit entry path works only because the app imports `t2pw.config` first. No
mining code was changed. The run went through [`launch_mining.py`](launch_mining.py), which calls the
project's own `t2pw.config.ensure_dotenv_loaded()` and then the unmodified CLI. That gives the run the
same environment the Streamlit path has. Preflight then passed every check (`g11/002`). **That is the
first live confirmation of the 64-token probe fix.** This happened before round 1, so it is outside the
one-run rule.

## Counts (pre-registered)

| | value | source |
|---|---|---|
| papers discovered / screened | 21 / 21 | `summary.json` papers |
| papers included / processed | 1 / 1 | PMC13364333, a *Marchantia polymorpha* **review** (typed `unknown` by the run) |
| papers excluded by the screen | 20 | reasons in `papers.jsonl`; see the audit |
| claims extracted → distinct reactions | 21 → 19 | 2 same-span MpAOS1/MpAOS2 claims merged once the enzyme field was dropped |
| **admitted to the primary (A. thaliana) graph** | **0** | `primary_graph.json` |
| candidate reactions / components | 19 / 14 | `candidate_subgraphs.json` |
| rejected claims | 0 | nothing rejected; all 19 are held back as candidates |
| species states | 10 `inferred_only` · 8 `species_not_stated` · 1 `in_vitro_or_cell_free` · **0** `non_target_species` | see finding F3 |
| budget stops | none | stop reason `frontier_exhausted` after round 2 |
| empty / failed model calls | 0 | `model_calls.jsonl` |
| uncited fields dropped | **59 of 59** asserted enzyme (19), gene (19) and species (21) values | `rounds/round_001/errors.jsonl` |

## Every reaction

No reaction was admitted, so this is every candidate. The equations are as recorded. **Bio verdicts are
from the independent `pwml-bio-auditor`**, adjudicated against the source paper's text. All 19 are faithful
to the paper; none is `UNSUPPORTED`. Supporting quotes, span ids and admission reasons for every row are
in [`audit/PILOT-001-AUDIT.md`](audit/PILOT-001-AUDIT.md).

| # | reaction id | equation (as recorded) | species state | connects to seed | bio verdict |
|---|---|---|---|---|---|
| 1 | `rx_cfcaf34abd9bd9b01f69` | 16:0 (palmitic acid) -> 16:1n9 (Δ7-palmitoleic acid) | inferred_only | no direct link (fatty-acid desaturation upstream) | SUPPORTED_BUT_OFF_PATHWAY |
| 2 | `rx_ba120ec5dd1e2c7f160f` | 16:1n9 -> 16:2n6 (HDA) | inferred_only | no direct link | SUPPORTED_BUT_OFF_PATHWAY |
| 3 | `rx_b67ae97a4f1d4149c0ef` | HDA (16:2n6) -> HTA (16:3n3) | inferred_only | upstream precursor: produces HTA (C16 precursor of dn-OPDA) | SUPPORTED_BUT_OFF_PATHWAY |
| 4 | `rx_c52713f0275d843084ce` | LA (18:2n6) -> ALA (18:3n3) | inferred_only | upstream precursor: produces ALA (13-LOX substrate) | SUPPORTED_BUT_OFF_PATHWAY |
| 5 | `rx_b77605ae2b74a25a76c9` | ARA (20:4n6) -> EPA (20:5n3) | species_not_stated | C20 PUFA feeding the C20 oxylipin branch | SUPPORTED_BUT_OFF_PATHWAY |
| 6 | `rx_521de0c1ef1ce3980ffb` | LA (18:2n6) -> GLA (18:3n6) | species_not_stated | no direct link | SUPPORTED_BUT_OFF_PATHWAY |
| 7 | `rx_7e7a4e2597d21a8daf41` | ALA (18:3n3) -> SDA (18:4n3) | species_not_stated | consumes the precursor ALA and leads away from 12-OPDA (toward C20 PUFAs) | SUPPORTED_BUT_OFF_PATHWAY |
| 8 | `rx_271d9d29572551c0995a` | GLA (18:3n6) -> DGLA (20:3n6) | species_not_stated | no direct link | SUPPORTED_BUT_OFF_PATHWAY |
| 9 | `rx_5bc5aa72c50bf073d7e1` | SDA (18:4n3) -> ETA (20:4n3) | species_not_stated | no direct link | SUPPORTED_BUT_OFF_PATHWAY |
| 10 | `rx_33c56fd5b50b116df0ad` | DGLA (20:3n6) -> ARA (20:4n6) | species_not_stated | C20 PUFA precursor (ARA) of the C20 oxylipin branch | SUPPORTED_BUT_OFF_PATHWAY |
| 11 | `rx_d2d57c7e2eb099611fa5` | ETA (20:4n3) -> EPA (20:5n3) | species_not_stated | C20 PUFA precursor (EPA) of the C20 oxylipin branch | SUPPORTED_BUT_OFF_PATHWAY |
| 12 | `rx_31e360d39a47d50a837a` | 13-HPOT -> allene oxide (12,13-EOT) | inferred_only | DIRECT: canonical AOS step | SUPPORTED_WRONG_SPECIES |
| 13 | `rx_afa85544d1f3505a7571` | 15-HPEPE -> allene oxide (14,15-EEPE) | inferred_only | C20 oxylipin branch, AOS-type chemistry, not the Arabidopsis 12-OPDA route | SUPPORTED_WRONG_SPECIES |
| 14 | `rx_04b5a990fe0223013188` | 15-HPETE -> allene oxide | inferred_only | C20 oxylipin branch, AOS-type chemistry | SUPPORTED_WRONG_SPECIES |
| 15 | `rx_c63294bd6f7cd7898284` | allene oxide (12,13-EOT) -> (9S,13S)-OPDA | in_vitro_or_cell_free | DIRECT: produces (9S,13S)-OPDA = the seed 12-OPDA (AOC step) | SUPPORTED_WRONG_SPECIES |
| 16 | `rx_dbf98b00d2a61c12dc9a` | allene oxide (from 11-HPHT) -> (9S,13S)-dn-cis-OPDA | inferred_only | C16 dn-OPDA branch parallel to the 12-OPDA route | SUPPORTED_WRONG_SPECIES |
| 17 | `rx_3b1de071bdca890db17e` | dn-cis-OPDA -> dn-iso-OPDA | inferred_only | C16 dn-OPDA branch (bryophyte bioactive jasmonate) | SUPPORTED_WRONG_SPECIES |
| 18 | `rx_30597429065d268ba67a` | Δ4-dn-cis-OPDA -> Δ4-dn-iso-OPDA | species_not_stated | C16 dn-OPDA branch | SUPPORTED_WRONG_SPECIES |
| 19 | `rx_dd0c3c9fb247ccd3b5df` | dn-iso-OPDA -> dn-OPDA-Glu + dn-OPDA-Gln + dn-OPDA-His | inferred_only | C16 dn-OPDA branch (conjugates) | SUPPORTED_WRONG_SPECIES |

Tally: **0** supported-and-relevant · **11** supported but off the jasmonate route (fatty-acid
desaturation and elongation; 7 of them belong to the C20 LCPUFA branch, which the paper says angiosperms
lack) · **8** supported but the wrong species (*Marchantia* or in-vitro evidence). Only rows 12 (AOS) and
15 (AOC → 12-OPDA) touch the canonical route. Nothing downstream of 12-OPDA was found.

## Biological plausibility — `pwml-bio-auditor`: **NOT_PLAUSIBLE**

This is judged as an *A. thaliana* jasmonate expansion from these seeds. The empty primary graph is the
**correct** outcome given what the run read, and no reaction was fabricated. Measured against the canonical
Arabidopsis route (lipase → 13-LOX → AOS → AOC → OPR3 → β-oxidation → JAR1), used only as a yardstick, the
run produced Arabidopsis evidence for **0** steps. The one paper it read states several Arabidopsis steps
explicitly (13-LOX, OPR3, JAR1), and none of them was extracted.

## Findings (registered, NOT fixed — none is chartered)

| id | finding | class | where |
|---|---|---|---|
| F1 | **Every enzyme, gene and species field was dropped as uncited (59 of 59).** Every dropped value does appear in a passage the model was shown, and the participant citations were valid. So the model failed to cite, and the check enforced the contract correctly. The consequence: **no claim could ever reach `target_species_supported`**, whatever the paper said. We cannot tell *how* the model failed, because only the response hash is stored (`model_calls.jsonl`) | candidate `product_contract_violation`; needs lineage first | `mining/extraction.py:174-183, 271-285, 296-306` |
| F2 | **Round 2 found nothing new, although plenty exists.** It ran 60 queries with 600 result slots, which held only 20 unique papers; 9 were new, and the screen excluded all 9. Three causes. (1) Query formulation ORs the frontier target with the topic and generic words, so the item's own term is optional. (2) The top-10 cap has no paging, and NCBI was never reached. (3) One attempt with no newly included paper marks an item exhausted. `frontier_exhausted` is therefore literally true and misleading | `product_contract_violation` (retrieval) | `queries.py:108-118`, `rag/acquire.py:371-402, 887-914`, `controller.py:393, 406-409` |
| F3 | Species labels are reached by fall-through. Because species was always dropped (F1), no reaction became `non_target_species`. 10 were labelled `inferred_only`, triggered by unrelated phrases like "Arabidopsis Ortholog" in a table header. 8 were labelled `species_not_stated` with the reason "no species is stated anywhere in the supporting evidence", although their cited spans name *Marchantia*. The **outcome** (nothing admitted) was right; the labels and reasons are wrong | `product_contract_violation` (labelling) | `mining/species.py:283-287` |
| F4 | The frontier cap silently cut 62 of 121 generated round-1 items. They were never asked, never recorded, and no stop reason names the cap | observability | `frontier.py:335-336, 353-356` |
| F5 | The screen missed **PMC9947334** (primary recombinant AtLOX2 enzymology), excluded as `context_only`. PMC12280011 (12-OPDA metabolites in *A. thaliana*) and PMID 35404431 (DAD1, the initiating lipase) were also missed. Several exclusion reasons are keyword artefacts | `policy_disagreement` / screen recall | `t2pw.rag.eligibility.screen_candidate` |
| F6 | The included paper is a review ("This review is organized to follow the pathway sequentially"; EuropePMC pubType `review-article`), but its `evidence_type` is `unknown` and its PMID and DOI are empty | `product_contract_violation` (evidence policy) | `evidence_policy.md` `review_article_only` |
| F7 | `report.md` misstates numbers. It says "Claims extracted: 19" (21 were extracted), and its "Primary graph" line prints whole-graph counts (10 connections, 30 entities) for an empty primary graph. The errors list is truncated without a marker (59 → 50 → 40) | report defect | `report.py:149, 159-160, 194, 298-299, 409` |
| F8 | Smaller issues. `queries.jsonl` `cache_hit` is never set. Screened-out round-2 papers are missing from the round provenance. `retries=1` is recorded for a single attempt. A resume would anchor 7 bridge targets differently (order-dependent). One abstract span has null offsets and raw HTML. Every component has `relevance_score` 0.0 | observability / determinism | see audit anomalies A4, A5, A10, A11, A13, A14 |
| F9 | The mining CLI does not load `.env` (the deviation above) | defect | `mining/cli.py` / `preflight.py:98-99` |
| F10 | `docs/pathway_mining/cli.md` shows `--enable-pathway-mining` **after** `run`. It is a top-level flag and must come first | doc defect | `cli.md` |

Suggested order, if the product owner wants mining to be evaluable. First F1 lineage: persist the raw
completion so the citation failure can be attributed. Then F2, then F3. Nothing here justifies changing
production T2PW.

## Also noted: `max_new_reactions_per_round`

The approved fix makes this cap a **stop**: the run ends after a round that added at least 25 new
reactions. It is not a throttle within the round. That matches the brief (`check_before_round`) and the
docs, and this pilot never came near it (0 new primary reactions). Whether it *should* be a throttle is a
product decision; see reviewer finding NB-1.

## Files

| path | what |
|---|---|
| `run/` | the complete run directory: all 49 files, exactly as written |
| `g11/001-…-preflight.json`, `002-…-preflight-dotenv.json`, `003-mining-pilot-001.json` | wrapper reports; all show 0 survivors |
| `pilot-001.log` | console log (checked: no credentials) |
| `launch_mining.py` | the launcher described above |
| `audit/` | the compiler's full audit, its reproducible scripts and their wrapper reports |
| `bio-audit/` | the bio-auditor's notes (abstracts, evidence text, EuropePMC pubType check) |
| `audit_workflow_result.json` | both agents' structured returns, verbatim |
| `acquire_cache.sha256` | sha256 of the 62 acquire-cache files. They are **not committed**: a cache, holding the fetched paper text |
