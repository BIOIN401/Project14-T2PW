# Priority 1 — every remaining false real identifier, classified by mechanism

**Author:** Lead Orchestrator · **Date:** 2026-08-27 · **Integration tip at measurement:** `ea1d51e`
**Corpus:** the T-106 run `runs_verify/2026-08-24_1428` (10 papers / 20 legs) for the itemization;
**all 92 committed `final_mapped.json` artifacts (1487 entity rows)** for every corpus-wide claim.
**Denominator corrected 2026-08-27:** an earlier revision said 78, which was the `runs_verify/`
subtree only and omitted the 14 legs under `runs/`. Every corpus-wide measurement below was re-run
over all 92; **no conclusion changed.**
**Gold set:** `src/t2pw/bench/gold/pinned_v1.json`, version `2026-08-01.1`

Every number below was produced offline from committed artifacts, through `bounded_run.py`, with no
live paper run and no LLM-backed command. G11 task `ORCH-092`, reports `01`–`13`, zero
non-compliant. Every job reported `FINAL SURVIVING COUNT : 0` and `cleanup : success`.

**This document does not open the Priority-1 question. It extends F-127 and F-128** (`LEDGER.md:3930`,
`:4259`) with measurement those findings asserted but did not quantify, and it corrects one thing
about the shape of the problem.

---

## 0. What "false real identifier" actually means

`bench/semantic.py :: _check_id_conflicts` increments `false_real` on two conditions, and the one
that fires here is: **the gold set declares the name is not a distinct identity
(`forbidden_identifiers`), and the payload has given it an external accession anyway.** Priority 1
is therefore a per-case, gold-declared list — not "entities we suspect".

**The mechanism taxonomy this card was asked to invent already exists in the gold data**: every
`forbidden_identifiers` entry carries a `kind`. I have used the gold's own vocabulary rather than a
parallel one.

## 1. The count is 6, not 8 — and this matters

T-106 measured **8**. Two of those eight are **already fixed**: `Pyridoxal 5'-phosphate` on
PMC12856317, both modes. C-081's PASS C (`map_ids.py:8578-8620`,
`RULE_COFACTOR_ROLE_UNUSED`) withholds exactly that class, and **C-081 merged as `b869780` on 2026-08-25 11:23, one day
after the T-106 run was committed (`efca465`, 2026-08-24)**. Replaying the committed payloads
through the shipped predicate confirms it: `cofactor_participation` returns
`status="unsupported", reason="cofactor_role_used_by_no_reaction", evaluated=4` on both PLP rows
(`ORCH-092/10-p1-passc.json`).

F-127 already states the live figure as **six**. This document confirms it by replay rather than by
inference, and the itemization below is of the six.

| # | Paper / mode | Entity | Bucket | Gold `kind` | Mechanism | Earliest unsafe seam | Minimum missing signal |
|---|---|---|---|---|---|---|---|
| 1 | PMC12096016 / research | `NADH` | compounds | `placeholder_product` | coupled-assay LDH **reporter species** read as a pathway metabolite | Stage-1 extraction admits it; PASS C correctly abstains because a retained reaction **does** use it | whether a reaction is *the paper's assay readout* rather than *pathway chemistry* — a property of the reaction, not the entity |
| 2 | PMC12096016 / research | `NAD+` | compounds | `placeholder_product` | as above | as above | as above |
| 3 | PMC12782028 / research | `LIPA` | proteins | `heading_or_prose` | enrichment gene-set membership converted into a pathway protein (a *degradative* lysosomal lipase, directionally opposite to the requested biosynthesis) | Stage-1 admits a gene-list row with no role; `protein_export_policy` flags **34** unused proteins and quarantines **0** | entity-level **role** and **scope** |
| 4 | PMC12782028 / research | `LBR` | proteins | `heading_or_prose` | as above (nuclear-envelope protein, no catalytic role stated) | as above | as above |
| 5 | PMC12782028 / research | `SREBF1` | proteins | `regulator_as_metabolite` | regulator mistaken for a reaction participant — a transcription factor from a Reactome membership set | as above | as above |
| 6 | PMC12782028 / research | `SREBF2` | proteins | `regulator_as_metabolite` | as above | as above | as above |

Two mechanism classes, two papers, **all six on research legs**. After C-081 there is **no
strict-leg false identifier in the T-106 corpus at all.**

**Every one of the six is outside PASS C's reach by construction**, and for two different reasons:
1 and 2 because the cofactor-participation rule *correctly* abstains — a reaction genuinely uses
them; 3–6 because they are proteins and the rule is about declared cofactors.

## 2. What the provenance carrier records — F-127, quantified

Census over all 150 entity rows in the 20 T-106 legs (`ORCH-092/03-p1-census.json`):

| `paper_extraction` lineage record | rows |
|---|---|
| `origin=paper_stated`, `support=unsupported`, `paper_explicit=explicit`, `sources=[]`, `review_required=false` | **128** |
| same but `paper_explicit=not_evaluated`, `review_required=true` ("not read verbatim") | 2 |
| no `paper_extraction` record at all | 20 |

**The extraction-stage carrier is constant.** Its `reason` is literally *"present in the Stage-1
extraction payload when it reached the Stage-1 to Stage-2 boundary"* on all 128; `sources` is empty
on all 128. `support: "unsupported"` means *"no support has been established"*, **not** *"this is
unsupported biology"* — and it says exactly that about ATP, pyruvate, enterobactin, EntA–EntF and
every legitimate protein in the corpus.

`identifier_mapping` then appends `origin=database_grounded, support=direct`. That `direct` is
support **from the database**, not from the paper — the precise failure the contract names as
*standard biochemical knowledge silently becoming paper-stated evidence*.

**This is why every non-vacuous formulation in C-084 stripped legitimate biology.** Any predicate
over the existing entity fields partitions 128 identical records: all or none. This is F-127's claim,
now with a number on it.

## 3. Candidate predicates, measured rather than argued

Over every row carrying a real external identifier (`ORCH-092/07`, `09`). Participation is by name
identity against `inputs`, `outputs`, `enzymes[].entity`, `modifiers[].entity`.

| Candidate | Catches | Collateral on legitimate rows | Verdict |
|---|---|---|---|
| **R1** entity participates in no retained process | 6 / 8 | **48** | **REJECTED** |
| **R2** entity flagged by the leg's own `entity_admission_report` | 2 / 8 | 0 *as literally matched* — see below | **REJECTED** |
| **R3** R1 **and** R2 | 0 / 8 | 0 | **VACUOUS** |
| **R4** the paper text does not print the entity name | 1 / 8 | 4 | **REJECTED** |

*(Denominators are the T-106 eight, so R1–R4 are comparable with the acceptance report.)*

**R1 — rejected.** Its 48 collateral rows are precisely the protected biology: `EntA`–`EntF`,
`LpxA LpxB LpxC LpxD LpxH LpxK WaaA FabZ MsbA`, `HMGCR HMGCS1 MVK MVD IDI1 SQLE FDFT1 FDPS LSS
CYP51A1 EBP NSDHL HSD17B7 MSMO1 ACAT2`, `ALAS1 ALAS2`, `ferrochelatase`, `heme`,
`ferric enterobactin`. A C-084-class formulation, rejected on the same grounds. *Honest caveat:*
part of the 48 is a naming artifact — `EntC` counts as a non-participant only because its reaction
names the enzyme `"Isochorismate synthase"`. That makes 48 an upper bound, and does not rescue the
predicate: a rule whose behaviour turns on whether Stage 1 wrote the gene symbol or the enzyme name
is not a biological gate.

**R2 — rejected, and its apparent zero collateral is an artifact I am not quoting as safety.** The
`cofactor_policy_advisory` in PMC12096016/research demoted four compounds under
`currency_not_subject_of_the_requested_pathway`: `NADH`, `NAD+`, **`ATP`**, **`PPi`**. Collateral
measured 0 only because the advisory says `"ATP"` while the payload row is `"Adenosine
triphosphate"` (and `"PPi"` vs `"Pyrophosphate"`) — a name-normalization miss, not an absence of
collateral. Matched correctly it strips ATP and pyrophosphate, both legitimate, five real accessions
each. That is the explicitly forbidden outcome.

**R4 — rejected, and it settles the obvious next idea.** **Seven of the eight false identities are
printed in the paper**; four legitimate id-carrying rows are not. `SREBF1`, `LIPA`, `LBR` appear
because the paper prints an enrichment gene list; `Pyridoxal 5'-phosphate` appears because the paper
correctly states it is ALAS2's cofactor. **The paper really does say these words. It does not say
they are steps of the requested pathway.** The discriminator is not presence — it is role and scope.

## 4. One thing that IS actionable: PASS C's reach depends on a non-deterministic draw

PASS C is gated by `identity_admission.declares_cofactor_role(row)`
(`identity_admission.py:697-700`), which is exactly `row["class"] == "cofactor"`. That label is a
Stage-1 draw, and **it is not stable for the same molecule** (`ORCH-092/17-all92-drawvar.json`,
92 artifacts):

| molecule | `class='cofactor'` | `class='compound'` |
|---|---|---|
| `NADH` | 9 | **3** |
| `NAD+` | 8 | **3** |
| `ATP` | 6 | 2 |
| `Adenosine triphosphate` | 4 | **6** |
| `Pyridoxal 5'-phosphate` | 10 | 0 |
| `PPi` | 11 | 2 |

**A product-endorsed refusal silently does not run about a quarter of the time, on the identity of a
label rather than on biology.** That is a defect in the gate, not a policy choice.

**The rule itself is sound and discriminating** — not a blanket cofactor ban. Corpus-wide, on
id-carrying declared cofactors, `cofactor_participation` returns **supported 30, unsupported 18**.

### The measured candidate

**Reach B** — evaluate the *unchanged* predicate for a compound row whose `class` is `cofactor`
**or** whose name is in the pipeline's own 29-name ubiquitous-cofactor hub list
(`rag.synthesize.COFACTOR_NAMES`, the same set `bench.semantic._connected_core` already uses).
Eligibility widens; the rule does not move.

| | rows withheld | newly withheld | legitimate rows lost |
|---|---|---|---|
| Reach A (today) | 18 | — | — |
| **Reach B** | **19** | **1** | **0** |

The single newly withheld row is `2026-08-22_2147/PMC12096016/research :: NADH`, **gold-forbidden,
kind `placeholder_product`** — a true positive. Reach B **keeps** `NADH` ×10, `NAD+` ×10, `ATP` ×8
(plus `Adenosine triphosphate` ×4), `Pyridoxal 5'-phosphate` ×4, `AMP` ×2, `PPi` ×2, `CoA-SH` ×2 —
every one because the predicate says a reaction uses it. **ATP is never withheld anywhere in the corpus.** (`ORCH-092/18-all92-reach.json`)

### Why I am not chartering it

Three reasons, and the third is decisive.

1. **It moves the T-106 count by zero.** In that run NADH and NAD+ *are* used by a retained
   reaction, so the predicate correctly abstains. Reach B catches a T-105-run row. Priority 1 stays
   at 6.
2. It reaches none of the six live cases (§1).
3. **It is in direct conflict with an unresolved product ruling.** F-128 (`LEDGER.md:4259`) measured
   that **12 of PASS C's 18 refusals already violate D-069**, and C-091 was chartered for D-069
   compliance and *explicitly not to merge*. Widening the reach of a mechanism the product owner has
   already found to be **over**-refusing is a policy decision, not an implementation one. Taking it
   unilaterally would be the orchestrator ruling on a live product conflict.

**Recorded as the smallest actionable Priority-1 step available, contingent on the D-069 ruling.**

## 5. Ruling

Of the three questions § 7 posed, the answer is the third.

1. *One safe production seam explains multiple cases?* **No.** Every seam wide enough to reach more
   than two also removes contract-protected biology, measured in § 3.
2. *Disjoint narrow corrections?* **One exists** (§ 4), it is zero-collateral, and it reaches **none
   of the six** and is blocked by D-069. Everything else available is a per-paper name list, which
   the charter forbids.
3. ***The representation lacks the information required to distinguish supported from unsupported
   biology.* Confirmed by measurement, and sharper than before:** for cases 3–6 the missing thing is
   an entity-level role/scope; for cases 1–2 it is not an entity property at all — it is whether the
   *reaction* is the paper's assay readout rather than pathway chemistry.

**No Priority-1 card is dispatched.** Forcing one would ship a guard that removes ATP, pyruvate,
enterobactin, NAD(H) or valid proteins to reduce a benchmark count — merge rule 6 and the product
contract both forbid it.

## 6. The missing distinction, and the smallest field that would carry it

The asymmetry is exact and visible in one payload. A **process** row carries:

```
"evidence": "The pathway begins with the conversion of chorismate to isochorismate,
             catalyzed by isochorismate synthase (EntC)",
"scope_membership": "core",
"enzymes": [{"entity": "...", "role": "catalyst", "evidence": "...", "confidence": 1.0}]
```

An **entity** row carries `name`, `class`, `confidence`, `provenance: "extracted"`, and the constant
lineage record of § 2. **Processes carry a verbatim span and a declared scope. Entities carry
neither** — though the schema for what is missing already exists one level down, on enzyme role
records, in the same file.

The enforcement substrate also already exists: `final_mapped.json` carries
`source_text_index.normalized`, the **full normalized paper text** (43–54 KB per leg), plus
`entity_admission_report` with working `hallucination_gate` and `cofactor_policy_advisory` phases.
Nothing needs fetching. What is missing is the *link* from an entity row into that text, and the
*scope* the link was read under.

| Question | Answer |
|---|---|
| **Which biological distinction is missing** | Whether an entity is a **participant of the requested pathway**, as against merely *named in the paper* — as a cofactor, an assay reagent, an enrichment-list member, a regulator, or a competing branch. All five are legitimately present in the source text; none is a pathway step. For cases 1–2 the same question is asked of a *reaction*: is this the paper's assay readout or its chemistry? |
| **Why existing fields cannot represent it** | The only entity-level evidence field is `provenance_lineage`, byte-identical on 128 of 150 rows with an empty `sources` list on every one. No span, no role, no scope. `support: "direct"` after mapping means *the database matched*, which is the very inference that must not be laundered into paper evidence. |
| **Smallest additional field** | On each entity row, the two fields process rows already have: `evidence` — the verbatim span that admitted this entity — and `scope_membership` from a closed vocabulary (`core`, `cofactor`, `regulator`, `assay_reagent`, `context`, `enrichment_membership`). Optionally `role`, which enzyme records already carry. On each **reaction** row, extend the existing `scope_membership` vocabulary with `assay_readout`. |
| **Which stage populates it** | **Stage 1 extraction** — the same stage that already emits `evidence` and `scope_membership` on process rows and `role`/`evidence` on enzyme records. An extension of an existing contract, not a new one. |
| **Which seam enforces it** | The **entity admission gate** (`_admit_identities`, `map_ids.py:8402`), before `identifier_mapping` promotes a row to `database_grounded` — before release, never at PWML export, and it is where PASS A/B/C already live. |
| **What makes it non-vacuous** | (a) A row with `scope_membership` outside `core` may not be released carrying a real identifier — and the test must **fail when the guard is neutralized**. (b) A row with `scope_membership: core` whose `evidence` span occurs in `source_text_index.normalized` **must keep** its identifiers — the preservation half, which is the half C-084 lacked. (c) A row whose declared span does **not** occur in the source index is `review_required`, never silently dropped. (d) A corpus census asserting the field is populated above a floor, so the guard cannot be satisfied by Stage 1 emitting nothing. |

## 7. Consequence for T-107

Priority 1 is **absolute** and **guaranteed to fail** at 6. Nothing in C-086…C-090 touches any of
the two remaining mechanisms, the one available correction reaches none of the six and is blocked by
D-069, and no safe general correction is implementable against the current representation.
**T-107 remains NO-GO.**

The verdict changes only on a product-owner ruling — either accepting F-127's measurement limitation
with a stated non-zero Priority-1 floor, or ruling F-128/D-069 so that a participant-provenance
representation can be chartered. Both are schema and policy decisions outside the orchestrator's
authority.

## 8. Reproduction

```
export T2PW_OFFLINE_CURATOR=1
export PYTHONPATH=<tree>/src
<py> docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py next --task ORCH-092 --label <label>
<py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label <label> --timeout 300 \
     --json <allocated> -- <py> <probe>.py <tree>
```

| G11 report | What it measured |
|---|---|
| `01-p1-itemize.json` | the eight, by paper/mode/entity/identifiers/gold-kind |
| `02-p1-rows.json` | full entity rows and lineage for all eight |
| `03-p1-census.json` | the 150-row lineage census of § 2 |
| `04-p1-participation.json` | first participation pass — **superseded**, its reader mis-read `enzymes`/`modifiers`; nothing from it is quoted |
| `05-p1-schema.json` | the process-vs-entity schema asymmetry of § 6 |
| `06-p1-sourceindex.json` | `source_text_index`, `entity_admission_report`, `protein_export_policy` |
| `07-p1-rulemeasure.json` | R1 / R2 / R3 catch and collateral, corrected reader |
| `08-p1-atp.json` | ATP and PPi carrying real accessions under other names |
| `09-p1-textpresence.json` | R4 — 7 of 8 are printed in the paper |
| `10-p1-passc.json` | PASS C's verdict on the four legs — PLP `unsupported`, NADH/NAD+ `not_evaluated` |
| `11-p1-drawvariance.json` | the `class` label is not stable for the same molecule, 78 artifacts |
| `12-p1-reach.json` | Reach A vs Reach B — **superseded**, its paper index was off by one |
| `13-p1-reach-corrected.json` | Reach B: +1 true positive, 0 collateral, ATP never withheld |
