# 08 Filling missing steps from databases of known pathways

Read-only assessment for ruling D-W17. No `src/` change, no LLM call, no pipeline run, no test. Queries were made by me on
2026-11-10 and 11-11 with User-Agent `t2pw-assessment/0.1 (research)`, no email, at 1 request/s or slower, no KEGG call. Raw
responses are cached outside the repo (`C:/t/ev-kpdb/raw/`, not committed). Scripts and result tables: `docs/pwml_output/evidence/08/`.
Labels: **M** measured by me, **L** from a document or web page I read, **G** guess or my reading.

## 0. NOT ESTABLISHED

1. **PathBank was not measured at all.** `pathbank.org` did not answer from this machine (connection failed, retried several
   times over two days), so its site terms, download page and any API were not read. Its licence below comes from the NAR 2024
   paper, not from its own site. No documented API exists in that paper. MySQL was off limits by the brief.
2. **MetaCyc/BioCyc was not queried.** Its data files and APIs need a paid subscription (s1). EcoCyc is free but under a licence
   I did not apply for.
3. **The Rhea licence text was read second-hand.** `rhea-db.org/help/license-disclaimer` returned HTTP 403 to the fetcher; the
   "CC BY 4.0" statement is from a search summary of that page (L, not verbatim).
4. **The measured recall is on 24 gold reactions in 7 papers, 6 of them bacterial.** Gold also lists 3 reactions for PMC12657337
   but that paper has no run (27 loaded in all; assessment 04 says 23; I did not reconcile the difference). Only 35 reaction
   slots were "missed" in 335 (run x reaction) slots, and 27 of the 35 are one reaction (DHB to DHB-AMP). Effective n is small.
5. **"Extra steps" were measured against gold only**, not against everything a paper says. Only the pathway sizes below.
6. **My automatic name matcher undercounts.** It uses exact names plus a hand alias list. Reactome uses abbreviations (dALA,
   SUCC-CoA), WikiPathways uses free labels, and our payload uses Greek letters and parenthetical abbreviations. I adjudicated
   the Rhea misses by hand (s3.2); I did not for Reactome and WikiPathways. Their numbers are floors.
7. **Whether the owner wants database steps inside the PWML** is open (Open questions). Nothing here shows a database step is
   biologically right for the paper's tissue, condition or pathway variant. Only that it exists in the database.
8. **Circularity is not removed by anything I ran.** Gold comes from PathBank-curated PathWhiz pathways. Section 5 gives a
   protocol; I did not execute it.
9. TCA name mode: Reactome step lists came back partly unresolved (s4), so I cannot give a complete step list for any database.

## 1. Candidate table

| | PathBank | Reactome | Rhea | WikiPathways | MetaCyc/BioCyc |
|---|---|---|---|---|---|
| Licence (L) | CC 4.0 suite with BY and NC ("released under the Creative Commons (CC) 4.0 License Suite", BY and NC, NAR 52:D654, read 2026-11-10; site not reachable) | **CC0** (https://reactome.org/license, read 2026-11-10: "licensed under the Creative Commons Public Domain Dedication (CC0)", commercial use allowed "without asking for permission", attribution "encouraged but not required"). Illustrations CC BY 4.0 | **CC BY 4.0** (https://rhea-db.org/help/license-disclaimer, via search summary 2026-11-10; credit required; "intended only for research, educational and informational purposes") | **CC0** (https://www.wikipathways.org/terms.html, read 2026-11-10: "WikiPathways content is covered by the Creative Commons CC0 Waiver"). Own terms ask for attribution as courtesy, "not a legally-binding code" | **Paid** (https://biocyc.org/download.shtml, read 2026-11-10: "Access to the BioCyc data files requires a paid BioCyc subscription costing at least $5,000"; "Access to BioCyc data via APIs requires a paid BioCyc subscription (access to the EcoCyc data is free)") |
| Access (M/L) | Bulk download: CSV/TSV, BioPAX, SBGN, SBML, PWML (L). No API documented (L). Website unreachable (M) | Public REST ContentService, no key, no stated rate limit; worked at 1 req/s (M). Bulk downloads exist (L) | Public REST/TSV at `www.rhea-db.org/rhea?query=...`, no key (M). SPARQL and bulk files (L) | SPARQL `sparql.wikipathways.org` worked (M). JSON files and GPML downloads; old web service retired (L, page notice) | API and data files subscription only. Not queried |
| Commercial/academic | NC: a free academic server is probably fine, commercial use is not (G, needs a written answer) | none | none (credit) | none | commercial and academic both pay for data; academic free for Pathway Tools software only |
| Coverage (L unless M) | 6,951 primary and 605,359 total pathways, 10 species: human, cow, rat, mouse, fly, worm, Arabidopsis, yeast, E. coli, P. aeruginosa | Curated human pathways inferred to a few eukaryote species. No bacteria returned for E. coli (M). Pathway search by name works (M: "TCA cycle" gives R-HSA-71403) | Reaction level only: no pathways, no organism field. Search by name, EC, ChEBI works (M) | Many species, mostly eukaryote; E. coli present but sparse (an "Acetaminophen synthesis" E. coli pathway matched chorismate; M). Title search works (M) | Strong on bacteria and plants (G, from literature) |
| By name / by entities | by name: yes (site search). By entity: yes on site (L); not tested | name: yes. entities: free-text search, needs a second fetch per hit to read participants (M) | name, EC, ChEBI id: yes. Best fit for a pair of entities | name: SPARQL title filter. entities: SPARQL on node labels (M) | not tested |
| Data shape | Reactions with participants, ids, compartments in PWML/SBML; closest to the T2PW payload (L) | reaction input/output with synonyms, ChEBI via cross-reference, compartment, catalysts, species (M) | equation with ChEBI-named participants, EC, UniProt enzyme count, direction flag. No compartment, no species (M) | nodes with labels and external ids, edges typed Conversion/Interaction, no stoichiometry or compartment in what I queried (M) | rich (L) |
| Verdict | Not usable until the site, terms and an access route are confirmed; strongest species fit for E. coli; **circular with the gold** | **Yes for eukaryote papers and name mode** | **Yes as the per-step source for any organism** | **Yes as second pathway source** | **No** (cost, API) |

KEGG: excluded (D-W14), no call made.

## 2. Data shape against the T2PW payload

T2PW needs, per reaction: input/output entities with names and ids, enzymes with UniProt, a location, and paper evidence (04, 06).

- **Rhea:** equation text with ChEBI names, EC number, and a count of UniProt entries. No location, no organism, no stoichiometry
  in the TSV columns I asked for. Direction: some entries are reverse of the gold (desmosterol to cholesterol is stored as
  cholesterol + NADP+ = desmosterol + NADPH, found only reversed). Some are lumped: RHEA:30571 is "3 DHB + 3 L-serine + 6 ATP =
  enterobactin", where the paper lists the intermediate steps.
- **Reactome:** one row per reaction with input, output, catalyst, compartment and a species name, all as readable names with
  a synonym list. Reactions are inferred to other species, so one reaction appears 6 to 10 times (HSA, MMU, RNO, CFA, BTA, SSC...).
  A hit for E. coli would be a human-ortholog step, which D-106 and D-108 forbid as an id and which must stay `inferred`.
- **WikiPathways:** label-to-label edges. No stoichiometry or compartment retrieved. Matching is on free text labels.
- **PathBank:** same model as PathWhiz, so cheapest to turn into payload rows, but unmeasured.

None of these is PWML. A mapper from a database step to the payload is a build, about the size of the existing `gap_resolver` (G).

## 3. Measured: what the databases could supply

### 3.1 Method (M)

Gold: 27 reactions in 8 papers loaded with the repo's own loader (`s1_gold.py`). For the 24 reactions with runs I counted, per
reaction, the run-slots in which the final payload (`final_mapped.json`) did not contain it, from the existing
`evidence/01/gold_recall.jsonl` (79 canonical runs, 335 slots). Those are the "missed" slots, 35 in all. Then for each gold
reaction I queried each public API with the gold substrate and product (`s2_query.py`) and counted a reaction as supplied when a
database reaction has every gold input on one side and every gold output on the other. Species of the paper: E. coli (4
papers), Listeria, Lactococcus (menaquinone), human (2 papers: heme, cholesterol).

### 3.2 Results (M, floors for Reactome and WikiPathways)

| source | gold reactions found (of 24) | missed slots covered (of 35) | species-correct hits |
|---|---|---|---|
| Rhea, automatic matcher | 14 | 4 | n/a, no species |
| **Rhea, after hand check** (`manual_adjudication.json`) | **23** (1 not checked: SEPHCHC to SHCHC) | **35** (the lumped enterobactin entry and the Kdo2 form are loose matches) | n/a |
| Reactome | 4 (human heme ALAS, lanosterol, CYP51A1 step, desmosterol to cholesterol) | 1 (cholesterol 1 of 18) | 4 of 4, because both papers are human; **0 for the 18 bacterial reactions** (no bacterial entry) |
| WikiPathways | 3 (all in human cholesterol) | 1 | 3 of 3; 0 for bacterial reactions |

Read: nearly every missed gold step is a known biochemical reaction in Rhea (G: gold was curated by people who knew the same
literature, so this is not independent). The pathway databases help only where the organism is a eukaryote. Not one of the
18 bacterial gold reactions was matched by Reactome or WikiPathways in the paper's species.

The 35 missed slots are 10.4% of the 335 slots. Even a perfect fill adds at most about that much reaction recall on these papers
(G); the biggest single item is DHB to DHB-AMP in PMC12452463 (19 of 25 runs missed it). Rhea has it (RHEA:20229).

### 3.3 Extra steps a fill would add (M for sizes, G for judgement)

- **Rhea, asked for a pair already in the payload:** 0 extra by construction, one reaction per pair (up to 3 for succinate
  dehydrogenase). Risk is lumping and direction, not volume.
- **Reactome pathway level:** the pathway above the hits, "Cholesterol biosynthesis" R-HSA-191273, lists 44 reactions against 3
  gold reactions, so up to 41 steps neither the paper nor gold has. "Heme biosynthesis" R-HSA-189451 returned only partly resolved
  ids; no count.
- **WikiPathways pathway level (canonical runs, below):** "Synthesis of ceramides and 1-deoxyceramides" has 109 metabolite edges
  against 16 delivered unique rows.

### 3.4 Three canonical runs on common pathways (no gold; M)

| run | pathway, organism | delivered unique rows | Rhea (literal names) | Reactome | WikiPathways |
|---|---|---|---|---|---|
| PMC5130987 | chlorophyll biosynthesis, Arabidopsis | 22 | 4 confirmed | none returned | only "Chlorophyll b degradation", 6 edges, 0 matched. **Wrong pathway by title** |
| PMC12376012 | sphingolipid/ceramide, human | 16 | 0 (names with abbreviations) | R-HSA-1660661 "Sphingolipid de novo biosynthesis" exists; steps not compared | 6 pathways; best "Synthesis of ceramides..." 109 edges, 16 matched by exact name; "Sphingolipid pathway" 81 edges, 3 matched |
| PMC11487621 | CoA biosynthesis, B. subtilis | 11 | 0 (alpha/beta characters in names) | none | none |

The zeros for Rhea are a naming failure (G), not an absence: e.g. our `α-ketoisovalerate` is not found by string, the ChEBI
name is `3-methyl-2-oxobutanoate`. So **matching must be by ChEBI id, after D-W12 id assignment**, not by name.

## 4. Name mode: "TCA cycle" (M)

| source | Homo sapiens | Escherichia coli |
|---|---|---|
| Reactome | R-HSA-71403 "Citric acid cycle (TCA cycle)", plus R-HSA-9854311 (maturation and regulation). Only 14 events resolved; 13 reactions, mostly enzyme assembly and maturation (SDHA binds SDHB, frataxin transfers Fe2+...). I could not get the core cycle steps in one call | no E. coli entry (search returned eukaryote items) |
| WikiPathways | 5 pathways, e.g. WP78 "TCA cycle (aka Krebs or citric acid cycle)" with 222 metabolite edges (more than 8 steps: it includes connected metabolites), WP2453, WP4236 | none |
| Rhea | no pathway concept. The 8 enzyme EC numbers (2.3.3.1, 4.2.1.3, 1.1.1.42, 1.2.4.2, 6.2.1.5, 1.3.5.1, 4.2.1.2, 1.1.1.37) each return 1 reaction (3 for succinate dehydrogenase) | same reactions: organism-free |
| PathBank | not queried (site unreachable); paper lists human and E. coli among 10 species | same |
| MetaCyc | not queried | not queried |

Reading (G): name mode needs a pathway source that lists the step order and organism. Reactome and WikiPathways give that for
human; for E. coli only PathBank or MetaCyc/EcoCyc would. Rhea can supply the chemistry of each step once a step list exists, but
it cannot say which steps make "the TCA cycle". The step list for the common textbook cycle could also come from the RAG paper
route of D-W16.

## 5. Risks

1. **Circular evaluation (high).** Gold = PathBank-curated reactions. A PathBank fill scored against that gold recovers the
   gold by construction. Reactome, Rhea and WikiPathways are independent of the PathBank record, but their curators and the gold
   curators read the same literature, so some correlation remains (G).
2. **Provenance.** A database step is not paper evidence. It must carry `db` with database, accession, retrieval date and match
   type (CLEAN-MODEL-REQUIREMENTS s1.1), must never raise the paper-support count, and must sit in a separate layer after the
   freeze (D-034, D-036, requirements rule 2).
3. **Species mismatch (D-106, D-108).** Reactome's non-human reactions are orthology inferences; Rhea has no species. A step
   found in the wrong species may only be `inferred`, never an id on a protein.
4. **Context.** Tissue-specific and disease pathways: WikiPathways returned "TCA cycle in senescence", "Krebs cycle disorders"
   and the title match "Chlorophyll b degradation" for a biosynthesis paper. Title match is not pathway match.
5. **Lumping and direction.** RHEA:30571 collapses the enterobactin steps; one gold step was only found reversed. Fills must not
   replace a more detailed paper row (04: the gold excludes protein-bound states).
6. **Volume.** Pathway-level fill adds tens of steps (44 reactions vs 3 gold in cholesterol biosynthesis), which conflicts with
   "smaller defensible beats larger contaminated" (PRODUCT_CONTRACT).
7. **Licences.** PathBank NC and MetaCyc payment are the only blockers; the other three allow a public server (L).
8. **Availability.** Public APIs are third-party; cache results and tolerate failure. PathBank was down for me for two days.

## 6. Recommendation

1. **Rhea as the per-step source, for every organism.** Use it only to fill a step whose substrate and product are both already
   entities in the payload with ChEBI ids (so it adds no new entity). Match by ChEBI id, check direction and lumping, record
   `db: Rhea RHEA:nnnn, retrieved <date>, match = ChEBI pair`. It covered all missed gold steps that I checked (s3.2). Cost: a
   ChEBI-pair lookup and a label, small.
2. **Reactome and WikiPathways, as pathway scaffolds, for eukaryote papers and for name mode.** Pick the pathway by the paper's
   organism and the entity overlap (not by title alone), then propose only steps whose participants overlap entities the paper
   gave, and list the rest as suggestions without writing them. Species must match; non-matching species are `inferred`, off by default.
3. **PathBank: do not build until** the NC terms are confirmed in writing and an access route exists; then it is the only
   scaffold source for E. coli in name mode, among the free ones. MetaCyc: no.
4. **Where in the pipeline:** after D-W12 id assignment and the completeness check, as a separate fill layer. It does not
   modify the frozen payload. The same function serves name mode: input is a pathway name, output is a scaffold from
   Reactome/WikiPathways plus Rhea chemistry; RAG (D-W16) fills what the databases lack.
5. **Labels:** `db` provenance with database, accession, retrieval date, match type; counted separately from paper support;
   shown in the web UI as "from database, not from your paper".
6. **Evaluation:** (a) *Hold-out ablation:* delete k gold-matched reactions from a frozen run, fill from each source, score
   recovery and the number of extras, per source. (b) *Source exclusion:* score a PathBank-derived fill only on pathways
   whose PathBank record is removed from the source; score Reactome/Rhea/WikiPathways separately and report PathBank
   separately. (c) Count extras, label them correct/off-pathway/wrong by hand on a sample, never against gold alone. (d) Report
   per organism class; today's gold is 6 of 7 bacterial papers, so it cannot test eukaryote scaffolds beyond 2 papers.
   Cost: a day to build the harness, another to review the sample (G).

## Open questions for the product owner

1. **Do database-filled steps go into the PWML, or only into other formats and the web view?** A: into PWML, labelled (matches
   D-W12 goal; risk: PWML has no provenance field I know of, so labels may be lost; needs the exporter to be proven). B: other
   formats and the report only, PWML unchanged (safe, no PWML regression; loses the main benefit). C: PWML only when the
   step connects two paper entities (middle road).
2. **PathBank terms.** A: write to the PathBank team about NC for a free public server (days; unblocks E. coli scaffolds and PWML
   bulk download). B: skip PathBank, use Reactome/WikiPathways/Rhea (no wait; no bacterial pathway source). C: both, with
   PathBank off by default.
3. **Is Rhea chemistry without organism acceptable as a fill when the paper names the organism?** A: yes, labelled `db` (best
   recall; the reaction may not occur in that organism). B: only when UniProt entries in Rhea include the paper's organism (needs one extra
   call per reaction; fewer fills). C: no.
4. **May Reactome ortholog steps (human to rat, etc.) be offered for a different species?** A: no (D-106/108 consistent).
   B: offer as `inferred` suggestions only, never written.
5. **Size limit for pathway-level fill.** A: only steps between entities the paper already named (smallest). B: also one hop
   beyond. C: whole pathway for name mode only.
6. **Evaluation:** accept hold-out ablation plus a hand-labelled sample as the paper evidence, or do you want a fresh
   independent gold (cost: curator time)?
7. **MetaCyc/EcoCyc:** ignore, or request an EcoCyc licence for E. coli name mode (free but needs application)?
