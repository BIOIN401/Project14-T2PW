# Priority 1 — every remaining false real identifier, classified by mechanism

**Author:** Lead Orchestrator · **Date:** 2026-08-27 · **Integration tip at measurement:** `f7b282c`
**Corpus:** `runs_verify/2026-08-24_1428` (T-106, 10 papers / 20 legs, fully committed)
**Gold set:** `src/t2pw/bench/gold/pinned_v1.json`, version `2026-08-01.1`

Every number below was produced offline from committed artifacts, through
`bounded_run.py`, with no live paper run and no LLM-backed command. G11 task `ORCH-092`,
reports `01`–`09`. Every job reported `FINAL SURVIVING COUNT : 0` and `cleanup : success`.

---

## 0. What "false real identifier" actually means

`bench/semantic.py :: _check_id_conflicts` increments `false_real` on exactly two conditions,
and the first is the one that fires here: **the gold set declares the name is not a distinct
identity (`forbidden_identifiers`), and the payload has given it an external accession
anyway.** So Priority 1 is not "an entity we think is unsupported". It is a per-case,
gold-declared list, and the gold set already carries a `kind` on every entry.

That matters for this card: **the mechanism taxonomy the sprint asked me to invent already
exists in the gold data.** I have used the gold's own `kind` rather than inventing a parallel
vocabulary.

## 1. The eight, itemized

Reproduced the acceptance report's count of 8 exactly, by calling the shipped scorer against
each committed `final_mapped.json` (`ORCH-092/01-p1-itemize.json`).

| # | Paper / mode | Entity | Bucket | Identifiers it carries | Gold `kind` | Mechanism | Earliest unsafe seam | Minimum missing signal |
|---|---|---|---|---|---|---|---|---|
| 1 | PMC12096016 / research | `NADH` | compounds | drugbank, hmdb, kegg, chebi, pubchem | `placeholder_product` | unsupported cofactor inferred from standard chemistry — a coupled-assay LDH reporter species read as a pathway metabolite | entity admission: the leg's own `cofactor_policy_advisory` demoted it and the demotion is **advisory only** | none new — the existing verdict is simply not binding |
| 2 | PMC12096016 / research | `NAD+` | compounds | hmdb, kegg, chebi, pubchem | `placeholder_product` | as above | as above | as above |
| 3 | PMC12782028 / research | `LIPA` | proteins | pathbank_protein_id + uniprot | `heading_or_prose` | prose/enrichment gene-set membership converted into a pathway protein (a *degradative* lysosomal lipase, directionally opposite to the requested biosynthesis) | Stage-1 extraction admits a gene-list row with no role; `protein_export_policy` flags 34 unused proteins and quarantines 0 | entity-level **role** and **scope**: "named in an enrichment list" vs "participant in the requested pathway" |
| 4 | PMC12782028 / research | `LBR` | proteins | uniprot | `heading_or_prose` | as above (nuclear-envelope protein, no catalytic role stated) | as above | as above |
| 5 | PMC12782028 / research | `SREBF1` | proteins | uniprot | `regulator_as_metabolite` | regulator mistaken for a reaction participant — a transcription factor from a Reactome membership set | as above | as above |
| 6 | PMC12782028 / research | `SREBF2` | proteins | uniprot | `regulator_as_metabolite` | as above | as above | as above |
| 7 | PMC12856317 / strict | `Pyridoxal 5'-phosphate` | compounds | (real compound accessions) | `cofactor_as_protein` | valid entity assigned an unsupported pathway role — the ALAS2 cofactor, never a substrate, never a product | Stage-1 extraction admits it as a compound row; the leg's `hallucination_gate` fires on `hemin` in the same leg and has no rule for this | entity-level **role**: cofactor-of vs participant-in |
| 8 | PMC12856317 / research | `Pyridoxal 5'-phosphate` | compounds | (real compound accessions) | `cofactor_as_protein` | as above | as above | as above |

Three mechanism classes, three papers, **seven of the eight on research legs**; only #7 is on a
strict leg.

## 2. What the provenance carrier actually records — and why every broad predicate failed

Census over all **150 entity rows** in the 20 legs (`ORCH-092/03-p1-census.json`):

| `paper_extraction` lineage record | rows |
|---|---|
| `origin=paper_stated`, `support=unsupported`, `paper_explicit=explicit`, `sources=[]`, `review_required=false` | **128** |
| same but `paper_explicit=not_evaluated`, `review_required=true` ("not read verbatim") | 2 |
| no `paper_extraction` record at all | 20 |

**The extraction-stage carrier is constant.** Its `reason` string is literally *"present in the
Stage-1 extraction payload when it reached the Stage-1 to Stage-2 boundary"* on all 128. Its
`sources` list is empty on all 128. `support: "unsupported"` here means *"no support has been
established"*, **not** *"this is unsupported biology"* — and it says that about ATP, pyruvate,
enterobactin, EntA–EntF and every legitimate protein in the corpus equally.

Then `identifier_mapping` appends `origin=database_grounded, support=direct`. That `direct` is
support **from the database**, not from the paper. This is exactly the failure mode the product
contract names: *standard biochemical knowledge silently becoming paper-stated evidence.* The
carrier does not lie — it simply has no field in which the difference could be written.

**This is why every non-vacuous formulation in C-084 stripped legitimate biology.** Any
predicate over the existing entity fields partitions 128 identical records: it fires on all of
them or on none of them.

## 3. Candidate predicates, measured rather than argued

Measured over every row carrying a real external identifier in all 20 legs
(`ORCH-092/07-p1-rulemeasure.json`, `09-p1-textpresence.json`). Participation is by name
identity against `inputs`, `outputs`, `enzymes[].entity`, `modifiers[].entity`.

| Candidate | Catches | Collateral on legitimate rows | Verdict |
|---|---|---|---|
| **R1** entity participates in no retained process | 6 / 8 | **48** | **REJECTED** |
| **R2** entity was flagged by the leg's own `entity_admission_report` | 2 / 8 | 0 *as literally matched* — see below | **REJECTED** |
| **R3** R1 **and** R2 | 0 / 8 | 0 | **VACUOUS** |
| **R4** the paper text does not print the entity name | 1 / 8 | 4 | **REJECTED** |

**R1 — rejected.** Its 48 collateral rows are precisely the biology the contract protects:
`EntA EntB EntC EntD EntE EntF` (enterobactin biosynthesis), `LpxA LpxB LpxC LpxD LpxH LpxK
WaaA FabZ MsbA` (the requested lipid A pathway), `HMGCR HMGCS1 MVK MVD IDI1 SQLE FDFT1 FDPS
LSS CYP51A1 EBP NSDHL HSD17B7 MSMO1 ACAT2` (the requested cholesterol pathway), `ALAS1 ALAS2`,
`ferrochelatase`, `heme`, `ferric enterobactin`. This is a C-084-class formulation and is
rejected on the same grounds. *Caveat recorded honestly:* part of that 48 is a naming artifact —
`EntC` is a non-participant only because the reaction names its enzyme `"Isochorismate
synthase"`. That makes 48 an upper bound on what R1 strips, but it does not rescue the
predicate: the composition alone disqualifies it, and a rule whose behaviour depends on whether
Stage 1 wrote the gene symbol or the enzyme name is not a biological gate.

**R2 — rejected, and its apparent zero collateral is an artifact I am not quoting as safety.**
The advisory in PMC12096016/research demoted four compounds under
`currency_not_subject_of_the_requested_pathway`: `NADH`, `NAD+`, **`ATP`**, **`PPi`**. The
measured collateral was 0 only because the admission record says `"ATP"` while the payload row
is named `"Adenosine triphosphate"` (and `"PPi"` vs `"Pyrophosphate"`) — a name-normalization
miss, not an absence of collateral. Implemented correctly with alias-aware matching, R2 strips
ATP and pyrophosphate, both carrying five real accessions each, both legitimate. That is the
explicitly forbidden outcome.

**R4 — rejected, and it settles the obvious next idea.** Seven of the eight false identities
**are printed in the paper**; four legitimate id-carrying rows are not. Presence in the source
text has sensitivity 1/8 and non-zero collateral. `SREBF1`, `LIPA`, `LBR` appear because the
paper prints an enrichment gene list; `Pyridoxal 5'-phosphate` appears because the paper
correctly states that it is ALAS2's cofactor. **The paper really does say these words. It does
not say they are steps of the requested pathway.** The distinction is not presence. It is
**role and scope**.

## 4. Ruling

Of the three questions section 7 posed, the answer is the third.

1. *One safe production seam explains multiple cases?* No. Every seam wide enough to reach more
   than two of the eight also removes contract-protected biology, measured above.
2. *Disjoint narrow corrections?* Available only as per-paper name lists, which the charter
   forbids and which would not generalize beyond the T-106 fixtures.
3. ***The representation lacks the information required to distinguish supported from
   unsupported biology.* Confirmed by measurement.**

**No Priority-1 card is dispatched.** Forcing one would mean shipping a guard that removes ATP,
pyruvate, enterobactin, NAD(H) or valid proteins to reduce a benchmark count, which merge rule 6
and the product contract both forbid.

## 5. The missing distinction, and the smallest field that would carry it

**The asymmetry is exact and it is visible in one payload.** In `final_mapped.json`, a *process*
row carries:

```
"evidence": "The pathway begins with the conversion of chorismate to isochorismate,
             catalyzed by isochorismate synthase (EntC)",
"scope_membership": "core",
"enzymes": [{"entity": "...", "role": "catalyst", "evidence": "...", "confidence": 1.0}]
```

An *entity* row carries `name`, `class`, `confidence`, `provenance: "extracted"`, and a
`provenance_lineage` whose extraction record is the constant string in §2. **Processes carry a
verbatim span and a declared scope. Entities carry neither.** Enzyme *role* records inside a
process even carry `role` and their own `evidence` — so the schema for what is missing already
exists in the same file, one level down.

The enforcement substrate also already exists: `final_mapped.json` carries
`source_text_index.normalized`, the **full normalized paper text** (43–54 KB per leg), plus
`entity_admission_report` with working `hallucination_gate` and `cofactor_policy_advisory`
phases. Nothing needs to be fetched. What is missing is the *link* from an entity row to the
place in that text that supports it, and the *scope* that link was read under.

| Question | Answer |
|---|---|
| **Which biological distinction is missing** | Whether an entity is a **participant of the requested pathway**, as against merely *named in the paper* — as a cofactor, an assay reagent, an enrichment-list member, a regulator, or a competing branch. All five are legitimately present in the source text; none is a pathway step. |
| **Why existing fields cannot represent it** | The only entity-level evidence field is `provenance_lineage`, whose extraction record is byte-identical on 128 of 150 rows and whose `sources` list is empty on every one of them. It has no span, no role and no scope. `support: "direct"` after mapping means *the database matched*, which is the very inference that must not be laundered into paper evidence. |
| **Smallest additional field** | On each entity row, the two fields process rows already have: `evidence` — the verbatim span from the paper that admitted this entity — and `scope_membership`, from a closed vocabulary (`core`, `cofactor`, `regulator`, `assay_reagent`, `context`, `enrichment_membership`). Optionally `role`, which enzyme records already carry. |
| **Which stage populates it** | **Stage 1 extraction** — the same stage that already emits `evidence` and `scope_membership` on process rows and `role`/`evidence` on enzyme records. This is an extension of an existing contract, not a new one. |
| **Which seam enforces it** | The **entity admission gate** (`entity_admission_report`), before `identifier_mapping` promotes a row to `database_grounded` — i.e. before release, never at PWML export. Its `cofactor_policy_advisory` becomes binding *for rows whose declared scope is not `core`*, which is what makes it safe: explicitly supported ATP declared `core` survives; the same ATP declared `cofactor` in a leg that did not request it does not get a released identity. |
| **What makes the proposal non-vacuous** | (a) A row with `scope_membership` outside `core` may not be released carrying a real identifier — asserted against a fixture, and the test must fail when the guard is neutralized. (b) A row with `scope_membership: core` whose `evidence` span occurs in `source_text_index.normalized` **must keep** its identifiers — the preservation half, which is the half C-084 lacked. (c) A row whose declared `evidence` span does **not** occur in `source_text_index.normalized` is `review_required`, never silently dropped. (d) A corpus census asserting the new field is populated on ≥ some floor of rows, so the guard cannot be satisfied by Stage 1 emitting nothing. |

## 6. Consequence for T-107

Priority 1 is an **absolute** acceptance priority and it is **guaranteed to fail** on the next
run. Nothing in C-086…C-090 touches any of the three mechanisms above, and no safe general
correction is implementable against the current representation. **T-107 therefore remains
NO-GO.** The smallest next action that would change that is a product-owner ruling on the
Stage-1 schema extension in §5 — a schema decision, not a coding card, and outside the
orchestrator's authority to take.

## 7. Reproduction

```
export T2PW_OFFLINE_CURATOR=1
export PYTHONPATH=<tree>/src
<py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py next --task ORCH-092 --label <label>
<py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label <label> --timeout 300 \
     --json <allocated> -- <py> <probe>.py <tree>
```

| G11 report | What it measured |
|---|---|
| `ORCH-092/01-p1-itemize.json` | the eight, by paper/mode/entity/identifiers/gold-kind |
| `ORCH-092/02-p1-rows.json` | full entity rows and lineage for all eight |
| `ORCH-092/03-p1-census.json` | the 150-row lineage census of §2 |
| `ORCH-092/04-p1-participation.json` | first participation pass — **superseded**, its reader mis-read `enzymes`/`modifiers`; nothing from it is quoted |
| `ORCH-092/05-p1-schema.json` | the process-vs-entity schema asymmetry of §5 |
| `ORCH-092/06-p1-sourceindex.json` | `source_text_index`, `entity_admission_report`, `protein_export_policy` |
| `ORCH-092/07-p1-rulemeasure.json` | R1 / R2 / R3 catch and collateral, corrected reader |
| `ORCH-092/08-p1-atp.json` | ATP and PPi carrying real accessions under other names |
| `ORCH-092/09-p1-textpresence.json` | R4 — 7 of 8 are printed in the paper |
