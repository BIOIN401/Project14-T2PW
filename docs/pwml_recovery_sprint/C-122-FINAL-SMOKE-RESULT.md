# `C-122` — merged, and the final reliability smoke. **RELIABILITY IS NOT COMPLETE.**

**2026-09-10.** `C-122` merged `--no-ff` at `04cc1908` after three review rounds and one blocking
correction. Production **RE-FROZEN**; `git diff 24dd4342 HEAD -- src/` is exactly one file,
`src/t2pw/mapping/map_ids.py`. `main` untouched. Protected `streamlit_app.py` unchanged at
`47e4fafa…`. Final smoke: `runs_smoke/2026-09-10_1135`, twelve fresh papers, one strict run each,
no favourable reruns, 2 h 57 m. Heavy lock `C-122` acquired and released,
`FINAL SURVIVING COUNT : 0`, `cleanup : success`.

---

# 1. THE DECISION

## `RELIABILITY PHASE IS NOT COMPLETE`

§ 23 permits closure at **approximately 80 % practical yield**, *or* when the remaining misses are
**isolated and stochastic rather than one repeated deterministic defect**. **Neither condition
holds**, and it is not close.

| | |
|---|---:|
| papers attempted | **12** |
| **meaningful cores** (`core_accepted ≥ 1`, F-179 not `no_defensible_core`) | **6** |
| **PWML files** | **4** |
| **overall yield** | **4 / 12 = 33 %** |
| serialization yield (cores that became files) | 4 / 6 = 67 % |
| structurally import-ready | **4 / 4** |
| dangling transport references (`F-195`) | **0** |

**Three mechanisms repeat across independent papers**, where § 23 asks for none:
`identity_resolution` ×2 · `f179` ×2 · `scope_or_guard_refusal` ×2.

The smoke also surfaced `F-202`, a spurious second species in 10 of 17 exported PWMLs. **It was
adjudicated the same day and is SENTINEL-ONLY** — every affected row is the PathBank `Unknown`
placeholder or a wrapper around one, and no resolved protein is mislabelled. It is not a defect and
it does not bear on this decision. § 5 carries the retraction; my original framing of it as a
correctness defect was wrong.

**This is a long way from 80 % and I am not going to present it as anything else.** The honest
readings that soften it are in § 4, and none of them reaches the bar.

---

# 2. The cohort, and why its number is conservative

Twelve **fresh** papers, zero overlap with the 75 PMC ids reachable from gold, any `runs*/` tree or
any `topics*.txt`. No control leg: `ORCH-734`'s sentinel existed for `F-192`, which `C-121` fixed and
live-validated. Manifest frozen and committed **before** execution:
[`topics_c122_final_smoke.txt`](../../topics_c122_final_smoke.txt). Selection tool:
[`evidence/c122_cohort_candidates.py`](evidence/c122_cohort_candidates.py), 90 full texts fetched,
62 passing candidates recorded; the tool scores and refuses, a human chose.

**Organism mix was stated as a downward bias before the run, not after it.** Bacterial 4 / plant 4 /
fungal 3 / mammalian 1, against `ORCH-734`'s 4 / 3 / 4 / 1 — heavier on fungal, lighter on
mammalian, because recent open-access literature titled *"biosynthesis"* is dominated by microbial
and plant natural products. `ORCH-725` measured resolution by organism group at human 67 %, plants
53 %, bacteria 39 %, fungi 20 %. **This mix therefore understates yield relative to `ORCH-734`, and
4/12 must NOT be compared like-for-like against its 7/12.** Two different cohorts, two denominators,
never merged.

---

# 3. Result table

Scored by [`evidence/orch734_cohort_report.py`](evidence/orch734_cohort_report.py), the same
instrument that was validated against `ORCH-732` before that cohort finished, so it cannot have been
tuned to these results. F-179 verdicts are **recomputed** from each leg's own `final_mapped.json`.

| Paper | Organism | Pathway | Rxns | Core | PWML | Bytes | Disposition | Failure |
|---|---|---|---:|---:|---|---:|---|---|
| `PMC12707518` | *S. suis* | branched-chain amino acid | 4 | 4 | **YES** | 63,894 | `review_required` | |
| `PMC13084691` | *A. membranaceus* | calycosin | 6 | 6 | **YES** | 67,552 | `review_required` | |
| `PMC13474689` | *M. oleifera* | glucomoringin | 7 | 7 | **YES** | 60,767 | `review_required` | |
| `PMC12914822` | *A. aculeatus* | *ent*-acu-dioxomorpholine A | 3 | 3 | **YES** | 35,078 | **`release_ready`** | |
| `PMC13089919` | *R. microsporus* | phosphatidylcholine | 5 | 5 | NO | — | `diagnostic_only` | `identity_resolution` — `ChoC`, `Cho2` |
| `PMC13488460` | *B. burgdorferi* | mevalonate | 1 | 1 | NO | — | `fail` | `identity_resolution` — `species_missing_taxonomy` |
| `PMC13438895` | *M. tuberculosis* | methionine | 4 | 1 | NO | — | `fail` | `f179` — `no_defensible_reaction_support` |
| `PMC13405594` | *M. musculus* | steroid | 1 | 5 | NO | — | `fail` | `f179` — `no_defensible_reaction_support` |
| `PMC13123502` | *P. polyphylla* | steroidal saponin | 0 | 0 | NO | — | `diagnostic_only` | `scope_conflict` — § 4.1 |
| `PMC13474940` | *F. proliferatum* | fumonisin | 0 | 0 | NO | — | `diagnostic_only` | `scope_conflict` — § 4.1 |
| `PMC13314799` | *B. brevis* | tyrocidine | 4 | 0 | NO | — | `fail` | `quarantine_core_coverage` — post-remap boundary |
| `PMC13398565` | *(none)* | camptothecin | 0 | 0 | NO | — | `fail` | `stage1_provider_delivery` — invalid JSON |

Per-leg wall times 6 m 56 s to 23 m 16 s. No leg hit the 3,600 s ceiling — **zero timeouts**, which
is itself a change from `ORCH-734`'s one.

---

# 4. The failures, classified before they are counted

## 4.1 `scope_conflict` ×2 — **partly my manifest, and partly a guard that refuses a generalization**

```
PMC13123502  requested 'steroidal saponin biosynthesis'
             Stage 0 read 'steroidal saponin (polyphyllin) biosynthesis in Paris polyphylla,
                           focusing on UGT-mediated 3-O-glucosylation'
PMC13474940  requested 'fumonisin biosynthesis'
             Stage 0 read 'fumonisin B1 biosynthesis'
```

**These are not the `ORCH-734` case and must not be filed with it.** There the operator asked for a
genuinely different pathway (*"steroidal glycoalkaloid"* against a paper about *"potato
solanidanes"*) and Stage 0 was right to refuse. **Here both requested scopes are strict
generalizations of what Stage 0 itself read.** *"fumonisin biosynthesis"* versus *"fumonisin B1
biosynthesis"* is the same pathway in the same paper, refused on a substring.

My share is real: I took each scope from the paper's **abstract**, while Stage 0 reads the **full
text** and produces a richer phrasing. Writing *"fumonisin B1 biosynthesis"* would have passed.

**But two independent papers failed the same way on a specificity mismatch, and that meets the
definition of a repeated deterministic mechanism that destroys otherwise-viable papers.** Both legs
died at Stage 0 with **0 reactions** — nothing about either paper was ever judged.

**Neither was re-run.** Changing the input after seeing the result is the favourable rerun the
charter forbids, and it would launder an operator error into a success. They stay in the
denominator with the cause named. Registered as **`F-203`**.

## 4.2 `f179` ×2 — **the safeguard working, and counted as a miss anyway**

`PMC13438895` (*M. tuberculosis* methionine) and `PMC13405594` (*M. musculus* steroid) both died at
`no_defensible_reaction_support`. Both are papers whose evidence is **transposon-sequencing and
knockout phenotype**, not reaction chemistry — exactly the shape that cannot support reaction-level
export. `PMC13405594` was flagged in the frozen manifest, before the run, as the thinnest paper in
the cohort and included deliberately rather than dropped.

**Per `PRODUCT_CONTRACT` § 1 a refusal to fabricate is a product success, not a shortfall.** These
two are counted as no-PWML because the deliverable is a PWML, but **they are not defects and no fix
is implied.**

## 4.3 `identity_resolution` ×2 — the standing class, and one is species-level

`PMC13089919` (*R. microsporus*): `ChoC` and `Cho2` carry no UniProt or DrugBank identifier. **5
reactions, 5 accepted core, F-179 sound** — a complete pathway lost at the identity gate alone.

`PMC13488460` (*B. burgdorferi*): `species_missing_taxonomy`, `species_missing_classification`. Not
a protein failure at all — **the organism itself is not in the local PathBank species table.** A
different gap from the protein-coverage one, and new to the record.

## 4.4 The two singletons

`PMC13314799` — 4 reactions extracted, **0 accepted core**, stopped at the post-remap stage
boundary with no issue codes. Opaque; not diagnosed here.
`PMC13398565` — *"Chunk 1 failed to produce valid JSON"*. The `F-193`/`F-199` provider-degeneracy
class, which `C-122` was never chartered to touch.

## 4.5 The framing that softens it, offered as context and NOT as a better headline

Excluding only my two scope strings, the system was asked ten fair questions and answered four with
a file. Excluding the two F-179 refusals as well — which are the gate behaving correctly — it
answered four of eight. **The reported number is 4 of 12.**

---

# 5. `F-202` — a WRONG ORGANISM is stamped into 10 of 17 exported PWMLs, and every prior spot check missed it

> ## ⚠ SECTION 5 IS RETRACTED ON ITS CENTRAL CLAIM. `C-123`, 2026-09-10.
>
> **`F-202` is SENTINEL-ONLY. No resolved protein carries a false organism, and this section's
> heading and its "wrong organism on a pathway enzyme" framing are wrong.**
>
> `AauA`/`AauB`/`AauC` are **sentinel-backed wrapper complexes**, not resolved enzymes. I classified
> them by name. An audit across 34 legs found **129** sentinel or wrapper rows and **0** resolved
> rows mislabelled (`evidence/c123_sentinel_audit.py`, `g11/C-123/01`).
>
> **The counts below are correct; the interpretation is not.** `F-202` does not outrank `F-195`, no
> production change is warranted, and the `release_ready` question is closed — the disposition
> logic was operating on placeholder rows. The remaining question is cosmetic and unchartered:
> whether the sentinel should carry a plant species at all.


The `release_ready` file `PMC12914822` declares two species:

```xml
<species><id>1386</id><name>Aspergillus aculeatus</name><taxonomy-id>5053</taxonomy-id></species>
<species><id>4</id><name>Arabidopsis thaliana</name><taxonomy-id>3702</taxonomy-id>
         <classification nil="true"/></species>
```

**and all three of its pathway enzymes are stamped with the plant**, not the fungus:

```
protein_complexes  AauA  species "Arabidopsis thaliana"  pathbank_species_id 4
protein_complexes  AauB  species "Arabidopsis thaliana"  pathbank_species_id 4
protein_complexes  AauC  species "Arabidopsis thaliana"  pathbank_species_id 4
proteins           Unknown  species "Arabidopsis thaliana"  pathbank_species_id 4
```

`pathbank_species_id: 4` is *Arabidopsis thaliana* and it is behaving as a **fallback species for
entities whose own species did not resolve.**

## It is PRE-EXISTING, not a `C-122` regression — measured, not assumed

| | |
|---|---:|
| import-set PWMLs carrying a spurious second species | **10 of 17** |
| of which generated **before** `C-122` | **6** |

Affected across three independent earlier cohorts: `ORCH-730` (*N. crassa*), `ORCH-732`
(*A. aeolicus*, *E. coli*), `ORCH-734` (*H. pylori*, *S. tubercidicus*, and `PMC11016064` which
carries **three** — *H. sapiens*, *M. musculus* **and** *A. thaliana*). `C-122` touched no species
resolution and this shape predates it.

## Why nobody saw it

`ORCH-734` § 8's biological spot check concluded *"The organism and taxonomy id are correct in every
case."* **That was true and insufficient** — it verified the *primary* species and never asked
whether a *second* one had been added. A check that looks only at the head of a list cannot see
what was appended to it.

> **This is more serious than `F-195`.** A dangling transport reference breaks an import. A wrong
> organism on a pathway enzyme **imports cleanly and is biologically false**, and in
> `PMC12914822` it does so in the highest-confidence disposition the system can assign.

## `release_ready` — a question, not a conclusion

`PMC12914822` is the **first `release_ready` PWML of the sprint**; `ORCH-734` § 1 recorded that every
one of its seven was `review_required`. Whether `C-122`'s identity changes contributed to this leg
clearing the release bar **with plant-stamped fungal enzymes is not established here**, and
separating it would need a base re-run this charter does not authorize. **Recorded as an open
question.** The file is committed as `C122_PMC12914822.WRONG-SPECIES.pwml` and **is not recommended
for import** except as a demonstration of `F-202`.

**Not fixed.** `C-122` is merged and re-frozen, and § 25 forbids widening it.

---

# 6. PathWhiz import inventory — 17 files, hash-pinned

All in [`pathwhiz_review/IMPORT-SET/`](pathwhiz_review/IMPORT-SET/), `sha256sum -c` all `OK`,
every copy `cmp`-verified byte-identical to its source run tree, which is preserved unchanged.

## Import first — the four new files, all structurally ready

| file | pathway / organism | bytes | rxn | import | caveat |
|---|---|---:|---:|---|---|
| `C122_PMC13474689.pwml` | glucomoringin, *M. oleifera* | 60,767 | 7 | **READY** | `F-202` |
| `C122_PMC13084691.pwml` | calycosin, *A. membranaceus* | 67,552 | 6 | **READY** | `F-202` |
| `C122_PMC12707518.pwml` | BCAA, *S. suis* | 63,894 | 4 | **READY** | `F-202` |
| `C122_PMC12914822.WRONG-SPECIES.pwml` | *ent*-acu-dioxomorpholine A, *A. aculeatus* | 35,078 | 3 | **READY** | **`F-202` on the enzymes themselves; `release_ready`** |

**Zero `F-195` dangling references in all four** — the first cohort with none.

**What to inspect after import:** whether the pathway renders and is biologically recognisable;
whether coordinates and edges generate; **and specifically whether the spurious *Arabidopsis
thaliana* species appears in the PathWhiz UI**, because that decides whether `F-202` is a cosmetic
export artifact or a real data-integrity defect.

Then the nine previously-recommended files, then the three `F-195` files **last** — § 22 stands
unchanged: if PathWhiz accepts and renders them, `F-195` is a checker limitation; if it rejects
them, `F-195` becomes a real blocker.

---

# 7. What `C-122` actually delivered

**Merged, reviewed over three rounds, and its blocking round-1 defect caught before it shipped.**

| claim | status |
|---|---|
| 4 LLM alias calls → 1 for a repeated `(name, organism)` | **measured** |
| a REVIEWED Swiss-Prot identity no longer rejected at margin 0.03 by its own gene's unreviewed records | **measured** (G9, on values) |
| transport failures no longer cached as permanent negatives | **measured** |
| `organism_id`/`taxonomy_id` plumbed to the collapse veto | **measured** |
| **identity-quality yield in production** | **UNMEASURED, and no number is claimed** |
| the archived unresolved proteins | **NOT recovered, and cannot be** — 11 of 14 verified against live UniProt to have no record for the requested organism |

`PMC13089919`'s `ChoC`/`Cho2` is a *new* instance of the identity class on a *new* organism, so the
class is not closed. **`C-122` did not measurably raise PWML yield in this cohort**, and the
comparison that would test it does not exist, because no cohort was run twice.

---

# 8. Next — the product owner's call, and I am not making it

Reliability does not close on this evidence. What would move it:

1. **Import the four files by hand.** That is the step this phase has been waiting on and the only
   thing that decides both `F-195` and `F-202`.
2. **`F-203`** (scope guard refuses a generalization) — 2 papers, deterministic, cheap. The obvious
   candidate for one narrow card, and the only one of the three repeats that is a genuine defect
   rather than a safeguard working or a coverage gap.
3. **`F-202`** — 10 of 17 files, three independent cohorts, wrong organism in exported biology. The
   most serious finding of this run.
4. `F-199` remains the standing charter candidate for the provider-delivery class that killed
   `PMC13398565`.

**RAG v2 does not begin.** § 23's bar is not met, and `RAG-V2-HANDOFF-REQUIREMENTS.md` § 6 is
explicit that closure is what licenses it.

---

# 9. Process

* Twelve legs in one bounded job, heavy lock `C-122` acquired and released,
  `FINAL SURVIVING COUNT : 0`, `cleanup : success`. Reports `g11/C-122/60`–`62`.
* Merge gate 10: **SMOKE 508** on the merged tree, zero survivors (`g11/C-122/50`).
* Independent review: **3 rounds**, `CORRECTION REQUIRED` → `APPROVE WITH FINDINGS` ×2, every fix
  reproduced by the reviewer rather than taken from the report.
* No leg re-run. No cohort manipulated. No scope string edited after seeing a result.
* `main` untouched. No cache or run directory committed. No worktree pruned.
