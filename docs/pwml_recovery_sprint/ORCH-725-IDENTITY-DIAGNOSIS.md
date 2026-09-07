# ORCH-725 — Why unseen pathways failed to export. A read-only diagnosis.

**Read-only investigation, 2026-09-07.** No production code was modified. `src/`, gold, run
artifacts and `main` are untouched. Branch `sprint/pwml-recovery` @ `fe1c0847`; pilot run
`runs_verify/2026-09-06_1425` (ORCH-724, Stage-1/Stage-2 `max_tokens=16000` per `F-186` —
that configuration difference is noted wherever a number could be read against committed
defaults, and no finding below is attributed to it).

**Evidence produced by this pass** (all under `evidence/`, all bounded, all `FINAL SURVIVING
COUNT : 0`):

| artifact | what it settles |
|---|---|
| `orch725_identity_probe.py` / `.log` · `g11/ORCH-725/02-identity-probe-v2.json` | PART A: does the shipped payload still fail? PART B: what did the resolver refuse, and was the rival real? |
| `orch725_uniprot_coverage_probe.py` / `.log` · `g11/ORCH-725/03-uniprot-coverage.json` | 37 live UniProt reads: is the residual class database coverage or query construction? |

---

# 1. Executive summary

> ## The dominant blocker is **not** identity resolution. It is `F-147` — a superseded contract report that the batch driver still honours.

**Four of the six evaluable strict legs were failed on a report that describes a payload
that no longer existed at export time.** In every one of them the payload that actually
reached export passes the real production gates.

| | |
|---|---|
| evaluable strict legs (ORCH-724 § 5 denominator) | **6** |
| produced a PWML | 1 |
| **blocked by `F-147` (stale `audit_round` report)** | **4** |
| blocked by a genuine live gate failure | 1 (`PMC11405693`, and it is *not* an identity failure) |

**The proof, three independent ways, on all four legs:**

1. **The production predicates pass.** `protein_external_identity` and
   `protein_species_context` — the same functions `process_normalizer.py` calls — applied row
   by row to the shipped `final_mapped.json` return **0 objections** on all four blocked legs,
   identical to the control leg that exported.
2. **The pointers cannot be hosted.** The failing report cites `/entities/proteins/5`, `/13`,
   `/15` against shipped protein lists of length **2, 3 and 8**. A pointer the list cannot
   host is by itself proof the report describes a payload that no longer exists.
3. **Every downstream report is clean.** `post_audit_contract_report` `ok=True` ·
   `post_remap_contract_report` `ok=True` · `pre_export_runtime_schema_report` `ok=True` ·
   `final_pre_export_stage3_gates` **`ok: true`, `errors: []`**.

And **not one** of the 19 entities the failing reports name is still a bare protein in the
shipped payload. Eleven became Unknown-backed functional complexes; eight were removed at
`pre_export_strict_quarantine` as `degree_zero_after_quarantine` — four of them
(`D4H`, `T16H2`, `T3O`, `T3R`) carrying `had_external_identity: true`, i.e. correctly
resolved proteins removed for being unreferenced.

`PRODUCT_CONTRACT` § 1 names **both** halves of this among the outcomes that may never end a
run without a PWML: *"a missing or **stale** gate report"* and *"an **irrelevant degree-zero
entity**"*.

## Are the failures technical or genuinely ambiguous biology?

**Overwhelmingly technical.** Of 55 refused protein rows, **~38 % are recoverable identities
the resolver could have reached** (Type 3), ~22 % are entities with no database identifier by
construction (Type 1), and only ~9 % are genuinely ambiguous biology (Type 2). Full split in
§ 6.

## The one number that reframes the whole finding

**`F-185` as written — "entity identity resolution is incomplete … and the required-field
gate then blocks strict PWML export" — is half right and half wrong.** Identity resolution
*is* incomplete. But the required-field gate did **not** block these exports: the Unknown-
sentinel fallback had already satisfied it, and the gate said so. What blocked them was the
driver reading a stale snapshot.

**Identity resolution is a *quality* axis in this system, not a *delivery* axis.** It decides
whether an enzyme ships with a real accession or with an honest placeholder. It does not
decide whether a PWML is produced.

---

# 2. Entity census

Source: `mapping_meta` on every entity of every leg that reached mapping, plus
`mapping_meta.identity_verdict.judged_candidate(s)` (retained under D-003 even after
`mapping_meta.candidates` is emptied at `map_ids.py:5350`), plus the pre-repair
`gate_fail_report.json` snapshots — which is where a protein rerouted into an Unknown wrapper
survives at all. Omitting that third source undercounts by 21 rows and loses `G10H`.

## By resolution outcome

| entity type | matched | novel / unresolved | ambiguous | Unknown-backed fallback | total |
|---|---:|---:|---:|---:|---:|
| proteins (incl. quarantined) | 39 | 34 | 1 | 4 | 78 |
| protein_complexes | 2 | 11 | 1 | 13 | 27 |
| compounds | 57 | 32 | 11 | — | 100 |
| species | 8 | 2 | — | 4 unresolved | 14 |
| subcellular_locations | — | — | — | 33 `not_applicable` | 33 |
| cell_types | — | — | — | 4 `not_applicable` | 4 |

## By organism group — proteins only, Unknown sentinel rows excluded

| organism group | resolved | total | rate |
|---|---:|---:|---:|
| human/mammal | 26 | 39 | **67 %** |
| plants | 8 | 15 | **53 %** |
| bacteria | 7 | 18 | **39 %** |
| fungi | 1 | 5 | **20 %** |

## By backend

| backend | role in the pilot | verdict |
|---|---|---|
| **UniProt REST** | every protein lookup | **not the problem.** 37/37 HTTP 200 in the live probe; 0 transport errors, 0 timeouts, 0 rate limits. In the run itself every recorded `queries_tried` returned a well-formed pool. |
| **PathBank local DB** | protein/complex/compound curated rows, and the `Unknown` sentinel (record 9659) | working. Supplies 18 of 39 protein matches and 48 of 57 compound matches. |
| ChEBI / KEGG / HMDB (`CompoundAPI`) | compound identifiers | working; 57 % compound match rate. |
| NCBI Taxonomy | not reached as a distinct backend | species resolution is done against the PathBank species table (`species_hydration`, `explicit_entity_species`), not an external taxonomy service. |

**Class A (external service failure) count for the entire pilot: zero.** No timeout, no rate
limit, no HTTP failure, no malformed response was recorded or reproduced.

## Papers blocked, by mechanism — the Pareto

Aggregated by **root mechanism**, never by issue code. `failures_by_code.txt` ranks by distinct
papers and every unresolved protein mints its own code, which is what scattered one mechanism
across 22 codes and ranked it below a one-off.

| mechanism | entities | papers | **strict exports blocked** |
|---|---:|---:|---:|
| **`F-147` — driver honours a superseded `audit_round` contract report** | n/a | **4** | **4** |
| margin rule rejects same-gene/same-organism duplicate records | 23 | 3 | 0 |
| no organism-matching record exists in UniProt (true coverage) | 12 | 4 | 0 |
| name-plausibility gate rejects a correct gene+organism match | 10 | 3 | 0 |
| wrong candidate selected / species comparator (strain parenthetical) | 5 | 2 | 0 |
| `reaction_enzyme_must_be_protein_complex` on a *resolved* protein | 1 | 1 | **1** |
| genuinely ambiguous identity | 2 | 2 | 0 |

**The key column is the last one.** Twenty-three margin rejections blocked **no** export,
because the Unknown fallback absorbed them. Four stale reports blocked four.

---

# 3. Backend / API diagnosis

**Which services work:** all of them. UniProt answered 37 of 37 live diagnostic requests with
HTTP 200 and zero transport errors. The PathBank local DB, the compound APIs and the species
table all returned usable results throughout the run.

**Which fail:** none.

**Which return no hits, and why:** 22 of the 37 live requests returned zero rows. Every one of
those zeros is attributable to the *query*, not to the service — and the probe separates the
two by asking each name three ways:

| case | production species-restricted query | control | diagnosis |
|---|---:|---:|---|
| `DltA` @ `Staphylococcus aureus (MRSA N315)` | **0** | 10, incl. **P99107 = *S. aureus* strain N315** | query construction |
| `DltC` @ same | **0** | 10, incl. *S. aureus* rows | query construction |
| `CrNPF2.9` @ *C. roseus* | **0** | 1 — `A0A1Q1PNY2`, gene `NPF2.9`, ***C. roseus*** | query construction |
| `CrGATA1` @ *C. roseus* | **0** | 1 — `A0A4Y5QBH9`, gene `GATA1`, ***C. roseus*** | query construction |
| `CrPIF1` @ *C. roseus* | **0** | hits | query construction |
| `CrTPT2`, `BIS1` @ *C. roseus* | 0 | 0 | **true coverage** |
| `NIT-9G`, `NIT-9E`, `NIT-1`, `NIT-12` @ *N. crassa* | 0 | 0 | **true coverage** |
| `ORMDL1`, `sphingomyelinase` @ *H. sapiens* | **hits** | — | the lookup succeeded; the failure is downstream |

**Two named query defects, both narrow and both demonstrated:**

1. **The organism string is passed verbatim, strain parenthetical and all.**
   `organism_name:"Staphylococcus aureus (MRSA N315)"` matches nothing in UniProt. The ladder
   then falls back to an unrestricted query whose top-ranked row is **human `P10515`** (it
   carries the gene alias `DLTA`), ships that, and the species check correctly rejects it —
   while `P0C397` (*S. aureus*, reviewed) and `P99107` (*S. aureus* **N315**, reviewed) sat in
   the same pool, unselected. The species gate did its job; the query and the ranking did not.
2. **The species-abbreviation prefix (`Cr…`) is not stripped for plants.** `CrNPF2.9`,
   `CrGATA1`, `CrPIF1` return nothing; `NPF2.9`, `GATA1`, `PIF1` restricted to *C. roseus*
   return the correct *C. roseus* records. The ladder already strips such prefixes for some
   names (`CrMYC2 → MYC2`, `CrWRKY1 → WRKY1` are in `queries_tried`) but the retry is not
   reached for these four.

**Which service is *not* the issue: all of them.** No API change, key, quota or endpoint
migration is implicated anywhere in this diagnosis.

---

# 4. Failure classes (§ 4 taxonomy), populated

| class | count | representative |
|---|---:|---|
| **A** external service failure | **0** | — none observed or reproduced |
| **B** database coverage — service worked, no such entity | 12 | `NIT-9G`/`NIT-9E` (protein *domains*, no accession by construction); `BIS1`, `CrTPT2` |
| **C** query generation | 8 | `DltA`/`DltC`/`DltD` strain parenthetical; `CrNPF2.9`/`CrGATA1`/`CrPIF1` prefix |
| **D** synonym / alias normalization | 4 | `sphingomyelinases`, `ceramidases`, `sphingosine-1-phosphate phosphatases` — the plural is queried literally |
| **E** species-context mismatch | 5 | judged candidate is the wrong organism while an organism-matching row sat in the same pool |
| **F** taxonomy resolution | 4 | `species_hydration` left *Arabidopsis thaliana*, *S. aureus*, *C. roseus* `unresolved`; `Catharanthus roseus` filed `novel / no_db_match` |
| **G** local PathBank indexing | 3 | `implausible_name_match` on PathBank rows carrying no accession (`ceramidase`, `sphingomyelinase`, `glucosylceramide synthase`) |
| **H** required-field policy | **0 blocking** | the gate the ORCH-724 report named. It was **satisfied** on all four legs — `final_pre_export_stage3_gates: ok` |
| **I** referential / complex linking | 2 | `PMC11405693` `AtCYP73A5` (a *resolved* protein) must reference a `protein_complex`; `PMC11172790` `/processes/interactions/5/entity_2 unknown entity: inner membrane state` |
| **J** intentional unresolved | 5 | `three-amino-acid product`, `SPT/ORMDL`, bare `ORMDL` |
| **—** margin rule (does not fit A–J; it is a *scoring* refusal of a correct answer) | 23 | `ORMDL1 → Q9P0S3` reviewed, rejected at margin 0.03 |

The last row is the point. **The largest single mechanism is not in the § 4 taxonomy at all,**
because the taxonomy assumes a refusal means the lookup failed. Here the lookup succeeded and
returned the right answer.

---

# 5. The dominant identity mechanism, stated exactly

`map_ids.py:5145-5175` computes a margin between the shipped candidate's score and the best
**rival**, and refuses below `_REAL_PROTEIN_MIN_MARGIN = 0.1`. The rival filter
(`:5144-5169`) excludes candidates of the wrong species, the wrong shape and the wrong name —
but **it does not exclude a candidate that is the same gene symbol in the same organism.**

UniProt routinely holds one reviewed Swiss-Prot entry plus several unreviewed TrEMBL records
for the same gene in the same organism. The scorer gives the reviewed entry **1.00** and the
TrEMBL duplicates **0.97** — a margin of **0.03**, below the threshold. Refused.

**Twelve rows — ten distinct entities — were refused this way while the judged candidate was a
reviewed Swiss-Prot entry with the exact gene symbol in the exact organism:**

| entity | leg(s) | organism | accession judged and discarded | margin |
|---|---|---|---|---:|
| `ORMDL1` | research + strict | *H. sapiens* | `Q9P0S3` ORM1-like protein 1 | 0.03 |
| `ORMDL2` | research + strict | *H. sapiens* | `Q53FV1` ORM1-like protein 2 | 0.03 |
| `ORMDL3` | research + strict | *H. sapiens* | `Q8N138` ORM1-like protein 3 | 0.03 |
| `Orm1` | research | *H. sapiens* | `P02763` Alpha-1-acid glycoprotein 1 | 0.03 |
| `Orm2` | research | *H. sapiens* | `P19652` Alpha-1-acid glycoprotein 2 | 0.03 |
| `dihydroceramide Δ4-desaturase 1` | research | *H. sapiens* | `O15121` | 0.03 |
| **`G10H`** | strict | *C. roseus* | **`Q8VWZ7` Geraniol 8-hydroxylase** | 0.03 |
| **`DAT`** | strict | *C. roseus* | **`Q9ZTK5`** | 0.03 |
| `PvdA` | research | *P. aeruginosa* | `Q51548` L-ornithine N(5)-monooxygenase | 0.05 |

*(`ORMDL1/2/3` appear on two legs each, hence 12 rows from 9 named entities; `PvdA`'s reviewed
`Q51548` is the tenth distinct case counting the strict-leg rows separately.)*

**A probable eleventh is not claimed, because the instrument cannot confirm it.** `NIT-3`
(*N. crassa*) had `P08619` **Nitrate reductase [NADPH]**, reviewed, gene `nit-3`, discarded at
margin 0.05 against `V5IKC3` — which is the same protein, same organism, unreviewed, but
carries **no gene symbol at all**, so the same-gene test cannot fire and the probe files it as
`GENUINE AMBIGUITY`. It is very likely a thirteenth redundant-rival row; it is reported here as
unconfirmed rather than counted.

A further **11** were refused with margin **0.00** where every rival was the *same gene in the
same species*, differing only by strain or isolate — the six `Pvd*` proteins of
*P. aeruginosa* being the clearest case (`PvdL`: PA14 `A0A0H2ZA98` vs PAO1 `Q9I157`, both
`pvdL`, both *P. aeruginosa*, both 0.80).

**Nothing about any of these is ambiguous.** The rival is the same protein.

---

# 6. Known name vs unknown identity — Type 1 / 2 / 3

| type | definition | entities | papers | strict exports it blocked |
|---|---|---:|---:|---:|
| **Type 3** — identity known, identifier exists, **our resolver failed** | margin rejects of reviewed entries (12) · strain parenthetical (3) · `Cr` prefix (3) · name gate vs exact gene+organism match (3) | **21** | 5 | **0** |
| **Type 1** — name and role known, **no identifier exists** | `NIT-9G`/`NIT-9E` (protein domains of the two-domain *nit-9* product) · family-level enzyme names (`ceramidase`, `sphingomyelinase`, `sphingosine kinase`, `sphingomyelin synthase 1`) · `BIS1`, `CrTPT2` | **12** | 3 | **0** |
| **Type 2** — identity itself ambiguous | `three-amino-acid product` · `SPT/ORMDL` (a complex written as a protein) · bare `ORMDL` · plural family terms | **5** | 2 | **0** |
| unclassified / other mechanisms | | 17 | | |

**The `Type 3` count of 21 against `0` exports blocked is the finding of this section.** A
tractable, real, recoverable identity gap exists — and fixing all of it would not, on this
cohort, have produced one additional PWML, because the Unknown fallback already carried those
entities past the gate.

## One correction to ORCH-724

ORCH-724 § 6 states that one of the two sub-causes is *"strain/construct labels extracted as
proteins (`NIT-9G` — the `strain_or_construct` forbidden class)"*. **That is not what
`NIT-9G` is.** The source text says, verbatim:

> *"In* N. crassa*, these steps are catalyzed by a two domain protein encoded by* nit-9*. Here,
> the **NIT-9G domain** adenylates MPT and the **NIT-9E domain** accepts the synthesized
> MPT-AMP …"*

They are **protein domains**, extracted correctly, and the paper attributes the two catalytic
steps to them exactly as the pipeline did. `strain_or_construct` is a **gold-set forbidden
class** (`bench/gold/pinned_v1.json`) with no production detector; it appears **zero** times in
the entire pilot run. The ORCH-724 claim was an inference, not a measurement.

The consequence is favourable and should be recorded: `NIT-9G`/`NIT-9E` are Type 1
(correct biology, no accession exists), **not** a forbidden extraction. The Moco pathway
`PMC7232280` recovered is legitimate.

---

# 7. Species bias — tested, and **refuted as stated**

ORCH-724 § 6 says the defect *"correlates with organism"* and that *"the identity layer does
not cover"* fungi and plants. The gradient in § 2 is real (human 67 % → fungi 20 %), but the
**mechanism** is organism-blind, and three measurements say so:

1. **The margin rule's worst absolute toll is on *Homo sapiens*.** Nine of the twelve
   reviewed-entry rejection rows are human (`ORMDL1/2/3` on two legs each, `Orm1`, `Orm2`,
   `dihydroceramide Δ4-desaturase 1`) — the most densely annotated organism in the corpus, and
   one of the three development organisms.
2. **Bacteria (39 %) score *below* plants (53 %).** *P. aeruginosa* — a bacterium, densely
   annotated — failed **6 of 6** `Pvd*` proteins on the margin rule. If the story were
   "development organisms resolve, exotic ones do not", this number would be inverted.
3. **The fungal cell is n = 5.** It cannot carry a claim, and the four `NIT-*` refusals in it
   are genuine coverage (§ 3), not bias — UniProt has no *N. crassa* record for a domain.

**What genuinely is organism-correlated is smaller and more specific:** the `Cr…`
species-abbreviation prefix convention in the plant literature, and the strain parenthetical in
the bacterial literature. Both are query-construction gaps, both are narrow, and neither is
about database coverage of non-model organisms.

**No hard-coded species restriction was found.** The species-restricted query is generated from
the entity's own organism field; there is no allowlist.

---

# 8. Sentinel / fallback feasibility

## The mechanism the question asks about already exists, already ships, and is already ruled on

`_apply_pathbank_unknown_enzyme_fallback` (`map_ids.py:7179`) is *"deliberately a Stage 6
structural fallback"*: for an enzyme that survived every resolution strategy unresolved, it
builds a **single-protein `protein_complex` carrying the functional enzyme name**, whose only
component is **PathBank record 9659, the `Unknown` protein**. It fired **13 times** in the
pilot. `pwml/writer.py:1568` has purpose-built serialization for it and cites PathWhiz's own
`reference/PW012926.pwml`, which carries several id-9659 locations — **this is PathWhiz's own
idiom, not an invention of ours.** It appears in already-shipped `pathway.pwml` files
(`runs_verify/2026-08-04_1754/papers/PMC12856317/strict/pathway.pwml`).

The eligibility rule is tight and worth stating, because it is what keeps it honest: the
fallback applies **only** to a reaction enzyme or a transport transporter, **only** when the
name is usable (not `unknown`/`hypothetical`/`putative`), **only** with recorded role evidence,
and **only** when the entity appears nowhere else in the payload
(`_has_disqualifying_reference`). It cannot be used to conceal an unidentified substrate.

## Feasibility table

| entity type | sentinel/fallback technically allowed? | importable? | biologically honest? | current policy permits? |
|---|---|---|---|---|
| reaction enzyme (protein) | **yes — implemented** (`pathbank_unknown_protein_fallback`) | **yes — proven in shipped PWML** | **yes**, when the functional name is real: it asserts a role, not an identity | **yes** — D-070 § O-1a rules the bare sentinel PathBank's legitimate representation, *not* a forged identity |
| transport transporter | **yes — implemented**, same seam | yes | yes | yes |
| protein complex | **yes** — the wrapper *is* a complex | yes | yes | yes |
| compound / metabolite | **no** such fallback exists | n/a | **no** — an unidentified metabolite is a chemistry claim, not a role claim | no, and it should stay no |
| species / taxonomy | `Unknown species` exists as an *absence* marker | yes | yes, as an absence | yes, but **C-098c is refused** and no path may reach `writer.py`'s `default_species_id` |
| subcellular compartment | `not_applicable` (33 rows) | yes | yes | yes |
| cell type | `not_applicable` (4 rows) | yes | yes | yes |
| biological state | auto-state creation exists | yes | yes | yes |

## The species-on-the-sentinel question, answered

The `Unknown` rows carry `species: Arabidopsis thaliana`, `taxonomy_id: 3702` in a
*Neurospora*, a *Catharanthus* and a human paper. **This is already adjudicated and is not a
defect.** D-070 § O-1a: *"Arabidopsis thaliana is that record's own species. On a row whose
entire content is 'this is PathBank record 9659', the Arabidopsis is a true fact about the
record, not a false mapping of an entity."* The rows additionally self-declare
`cross_species_placeholder: true` and `target_organism: "Neurospora crassa"`, and every one
carries `review_required: true` with the uncertainty stated in words.

**D-074 does not restrict this.** D-074 scopes when the **benchmark scorer**
(`bench/semantic.py:1418`) tolerates an Unknown-backed row against `PMC12444477`'s gold. It is
a scoring rule, not an export rule. Nothing in it forbids the exporter from emitting the
sentinel, and this pass did not reopen or modify it.

## Which F-185 failures are analogous to the existing Unknown handling, and which are not

- **Analogous, and already handled:** the 11 unresolved reaction enzymes that became wrappers
  (`NIT-7A/7B/9G/9E`, `G10H`, `STR`, `SGD`, `ceramidase`, `sphingomyelinase`,
  `sphingosine kinase`, `sphingomyelin synthase 1`, `dihydroceramide Δ4-desaturase 1`).
  Name and role known, identifier absent — precisely the case the sentinel is for.
- **Not analogous:** the 21 **Type 3** entities. For these the identifier *does* exist and we
  simply failed to reach it. Routing them to the sentinel is not a fallback, it is a
  **downgrade** — it replaces a recoverable real accession with a placeholder. That is the
  quality cost `F-185` actually imposes.
- **Not analogous, and must not be:** `three-amino-acid product`. An unidentified *product* is
  not an identity gap, it is missing chemistry. No sentinel path reaches compounds, correctly.

**Would expanding Unknown handling help?** No — and this is the sharpest negative result of
the pass. Expanding it would recover **zero** additional exports, because it was never the
binding constraint. It would also move Type-3 entities *away* from their real accessions.

---

# 9. Could PWML be valid without DB-exact IDs?

**It already is, and it already imports.** The single PWML the pilot produced is
`PMC12071552/strict/pathway.review_required.pwml` (graph valid, completeness 0.4). Earlier runs
shipped `pathway.pwml` files carrying record 9659. The writer emits a distinct
protein-location per sentinel usage specifically so that multiple Unknown-backed complexes do
not collapse onto one shared node — a defect that was found and fixed, which is itself evidence
the path is exercised, not theoretical.

Risks, per fallback, for the record:

| fallback | risk |
|---|---|
| Unknown-backed functional complex (**in use**) | reduced searchability; a reader must consult `functional_enzyme_name`; the record's Arabidopsis species travels with it (declared, ruled acceptable) |
| create-by-name protein with no identifier | **breaks import** — the required-field gate exists precisely to prevent it; would also mint duplicate PathBank entities |
| synthetic local ID | **false identity.** Rejected: it would claim a database record that does not exist |
| omit the optional identifier | not available — the identifier is required, not optional, for a protein |
| species-level fallback | **refused already** (C-098c, D-070). No path may end at `default_species_id` |

---

# 10. Top three candidate fixes

Ranked by **papers recovered**, which is the unit that matters.

## Fix 1 — Stop the driver honouring superseded `audit_round` contract reports (`F-147`)

- **Mechanism.** `batch/driver.py::_blocking_reports` (`:1026-1056`) scans every top-level
  `*_contract_report` and fails the run on any carrying errors. **It never reads the phase
  stamp** — while `streamlit_app.py:4055-4060` documents in terms that the `audit_round`
  report is *"not a verdict about what shipped"*. The gate channel immediately below
  (`:2506-2521`) **already had exactly this fix applied**, for exactly this reason: *"A truthy
  stale dict with no artifact able to countermand it meant no repair could ever produce a
  PASS."* The contract channel was not brought along.
- **Seam.** `_blocking_reports`, plus whatever `gate_reports.gate_verdict` already does for the
  gate channel — the precedent and the fail-closed discipline exist and would be followed.
- **Papers recovered on this cohort: 4** — `PMC7232280` (Moco biosynthesis, 5 reactions, one
  connected component, 100 %), `PMC12376012` (sphingolipid metabolism, **15 reactions**, one
  component, 100 %), `PMC8510960` (MIA biosynthesis, 5 reactions, 3 components, 56 %),
  `PMC11172790` (**0 reactions** — would export a near-empty pathway and is not a real gain).
  **Honest count: 3 substantive.**
- **Implementation size:** small — one scan predicate.
- **Biological risk: this is the one that needs the product owner, not the engineer.**
  F-147 is registered `HIGH` and **deliberately not chartered**, because on the *development*
  cohort lifting it would have converted two contract-correct no-export outcomes into two
  **gold-forbidden** exports (`PMC12452463`'s `enterobactin synthase complex` and `RyhB`
  interactions; `PMC12180156`'s `protoporphyrin IX`, a named hallucination test). **That
  precondition is unchanged and this pass did not clear it.** What is new is only that the
  same defect now demonstrably costs four unseen papers, and that on those four the shipped
  payloads carry **0 rows with no provenance** and pass every live gate. Whether that changes
  the balance is a product decision. **A fix would have to land with the gates that would then
  block the two development legs on their real problems**, exactly as F-147 states.
- **Regression risk:** the two development legs would need those gates, or they would export
  contaminated content. This is the whole of the risk and it is not small.
- **Production unfreeze required: yes.**

## Fix 2 — Exclude same-gene/same-organism duplicates from the margin rival set

- **Mechanism.** In the rival loop (`map_ids.py:5144-5169`), skip a candidate that shares the
  judged candidate's gene symbol **and** organism. Optionally also prefer a reviewed entry over
  unreviewed rows of the same gene. Both are refinements of a filter that already excludes
  wrong-species and wrong-name rivals; neither weakens a biological gate — a rival that *is*
  the shipped protein was never evidence of ambiguity.
- **Seam.** One filter clause in `_identity_verdict`'s rival loop. Genuinely small.
- **Papers recovered: 0.** **Entities upgraded from placeholder to a real reviewed accession:
  ~12–23, across 3 papers**, including `G10H → Q8VWZ7` and `DAT → Q9ZTK5` (the two flagship
  *C. roseus* enzymes) and `ORMDL1/2/3`.
- **Biological risk: low, and it is the *safe* direction** — it replaces `Unknown` placeholders
  with verified Swiss-Prot identities. It does not admit any new entity to the pathway.
- **PathWhiz import risk: reduces it** (real accessions instead of repeated record 9659).
- **Regression risk: real and must be measured.** The margin rule is a biological gate and
  loosening any gate is merge-rule-6 territory. The change must be shown not to admit a
  *different* protein anywhere — the `judged_candidates` are retained on every historical
  verdict, so a replay over the archived corpus can measure this exactly, with no new run.
- **Production unfreeze required: yes.**

## Fix 3 — Normalize the organism string and the species-abbreviation prefix before querying

- **Mechanism.** (a) strip a strain parenthetical from the organism before it enters
  `organism_name:"…"` and before the species comparator; (b) extend the existing
  species-prefix retry so `CrNPF2.9 → NPF2.9` is reached as `CrMYC2 → MYC2` already is.
- **Seam.** The query builder and `_candidate_species_verdict` in `map_ids.py`.
- **Papers recovered: 0. Entities upgraded: ~6**, including `DltA → P99107` (*S. aureus*
  **N315**, the exact requested strain) on the one leg that already exports, and three
  *C. roseus* regulators that were dropped as degree-zero anyway.
- **Biological risk: low**, but (a) touches the species comparator, which is a gate that
  demonstrably caught a wrong human identity in this very pilot. It must be shown to still
  refuse `P10515`.
- **Regression risk: moderate** — the species comparator is load-bearing.
- **Production unfreeze required: yes.**

### Explicitly *not* recommended

- **Expanding the Unknown sentinel.** Zero papers recovered; it was never the constraint.
- **Relaxing the required-field gate.** It did not block anything. § 3 class H is zero.
- **Anything about token budgets.** `F-186` is real but no finding here depends on it: every
  mechanism above is a lookup, scoring or reporting property, and the two JSON failures remain
  unattributed as `D-097` requires.

---

# 11. Recommendation

> **Is there one small, biologically honest identity-resolution improvement likely to
> substantially increase PWML generation?**

## **NO — for identity resolution. The limitation is genuine and should be documented.**

Identity resolution has a real, measurable, tractable gap — **21 Type-3 entities the resolver
could have reached and did not.** Fixing all of it, by the pilot's own evidence, would have
produced **zero additional PWML files.** The Unknown-sentinel fallback already carries
unresolved enzymes past the required-field gate, and it does so honestly and with PathWhiz's
own idiom. Fixes 2 and 3 are worth doing for **identity quality**, and they would materially
improve the *content* of exports — real Swiss-Prot accessions instead of placeholders for
`G10H`, `DAT`, `ORMDL1/2/3` — but they are not completeness fixes and should not be sold as
such.

## **But the question behind the question has a different answer.**

**The completeness bottleneck is `F-147`, and it is one predicate.** Four of six evaluable
strict legs were failed on a report the app itself documents as not being a verdict, naming
entities that do not ship, with pointers the shipped payload cannot host, while every live gate
said `ok`. `PRODUCT_CONTRACT` § 1 names a stale gate report an unacceptable terminal blocker,
and F-147 is already classed `product_contract_violation`.

**This is not a recommendation to fix it.** F-147's documented precondition — that lifting it
on the development cohort would produce two gold-forbidden exports — is **unchanged**, and this
pass produced no evidence bearing on it. Fixing the reporting seam before the upstream content
is stopped is the mistake F-147 itself names.

**What this pass changes is the price.** F-147 was recorded when it cost two development legs
that *should* not export. It now demonstrably costs **three substantive unseen pathways that
appear to have nothing wrong with them** — 25 reactions in total, one connected component at
100 % on two of the three, zero rows without provenance, and every live gate passing.

**The decision the product owner now has is a genuine three-way one:**

1. **Document and ship.** Report the limitation honestly. The manuscript's completeness number
   stays 1 of 6, and the limitation section says *"a superseded validation report blocks
   export on payloads that pass every live gate; registered as F-147, deliberately not fixed
   under the sprint's stopping rule."* Costs nothing, risks nothing, and is defensible.
2. **Fix F-147 together with the gates its own text requires**, then re-run the four legs as a
   separately-labelled diagnostic. This is not small: it is the fix F-147 says must land with
   the upstream gates, and it reopens engineering.
3. **Take Fix 2 alone**, as a quality improvement with no completeness claim attached. It is
   the smallest, safest and most self-contained of the three, it is measurable offline against
   the retained `judged_candidates` with no new run, and it upgrades the two flagship
   *C. roseus* enzymes and the three human `ORMDL`s from placeholder to verified Swiss-Prot.

**My reading, offered as evidence and not as a decision:** option 1 is correct for the
manuscript, and option 3 is the only one of the three that could be done within the sprint's
stopping rule without reopening the F-147 precondition. **Option 2 should not be taken to
increase a completeness number before submission.**

---

## What was NOT established, and must not be inferred from this report

- **That the four F-147 legs would export *good* biology.** They pass every technical gate and
  carry full provenance. **No human has reviewed their content**, and no gold exists for these
  papers. `PILOT-MANUAL-REVIEW.md` remains the blocking item.
- **That `PMC11172790` is a recovery.** Its shipped payload has **0 reactions**. It would
  export a near-empty pathway and should not be counted.
- **That fixing F-147 is safe.** The development-cohort precondition is untested here.
- **That any of this is caused by the `F-186` token-budget difference.** Nothing above depends
  on generation length.

---

*Read-only diagnostic pass. No production code, gold data, run artifact or `main` was modified.*
