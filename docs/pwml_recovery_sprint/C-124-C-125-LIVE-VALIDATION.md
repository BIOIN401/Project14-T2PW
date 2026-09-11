# `C-124` / `C-125` post-merge live validation — both fixes confirmed in production

**2026-09-10.** Three strict legs, one draw each, no favourable reruns.
`runs_validation/2026-09-10_1809`, 1 h 08 m. Heavy lock `C-123` acquired and released,
`FINAL SURVIVING COUNT : 0`, `cleanup : success`. Report `evidence/g11/C-123/60-live-validation.json`.

**The scope strings were copied verbatim from `topics_c122_final_smoke.txt` and mechanically
diffed against it before the run.** That is the entire point: `C-125` claims a request may be more
general than what Stage 0 reads, so editing the strings to match Stage 0 would have tested nothing.

**Success was defined before execution**, in the committed manifest: for the two `C-125` legs,
*passing Stage 0*; for the `C-124` leg, *a resolved species*. A PWML was explicitly declared a bonus
and not the claim.

---

# 1. Result

| paper | card | before | after | wall |
|---|---|---|---|---|
| `PMC13488460` *B. burgdorferi* | `C-124` | `species_missing_taxonomy` + `species_missing_classification`, no PWML | **PASS — `pathway.review_required.pwml`, 19,818 B** | 14 m 00 s |
| `PMC13474940` *F. proliferatum* | `C-125` | `scope_conflict` at Stage 0, **0 reactions** | **passed Stage 0**, ran to `pwml_export`, refused on `no_defensible_reaction_support` | 14 m 39 s |
| `PMC13123502` *P. polyphylla* | `C-125` | `scope_conflict` at Stage 0, **0 reactions** | **passed Stage 0**, ran 39 minutes to the stage-3 gate, refused on 4 unresolved UGTs | 39 m 01 s |

**All three fixes did exactly what they claimed. Neither `C-125` leg produced a PWML, and that was
never the claim** — both papers are now *judged* where before they were never looked at.

---

# 2. `C-124` — the species was created, and the provenance is auditable

```
Borrelia burgdorferi    taxon 139     Prokaryote   pathbank_id None
  taxonomy_backfill: source "ncbi_db_synonym"
                     taxonomy_id "139"   classification "Prokaryote"
                     resolved_as "Borreliella burgdorferi"
                     donors ["Borreliella burgdorferi (strain ATCC 35210 / DSM 4680 / CIP 102532 / B31)"]
Borreliella burgdorferi taxon 224326  Prokaryote   pathbank_id 10659
```

Every safety property holds, measurably:

* **The display name is the paper's own spelling.** `Borrelia burgdorferi` is not rewritten.
* **The taxon is SPECIES rank (139), not the donor's STRAIN rank (224326).** This is the property
  review said mattered most — copying the donor's id would have shipped a strain-rank taxon for a
  species-rank name. `_binomial_from_organism` plus an NCBI-only answer is what produces 139.
* **`resolved_as` and `donors` record exactly which name was asked and which row asserted it**, so a
  curator can audit the substitution rather than having to trust it.
* **No PathBank id was invented** — the created species correctly carries none.

The third species row is the `Unknown` sentinel (`pathbank_species_id 4`, `classification` null),
which `C-123` established is a technical placeholder carrying no biology. It is untouched.

---

# 3. `C-125` — both papers are now judged, and both fail on something else

Neither leg records `scope_conflict`. `PMC13123502` ran **39 minutes** and reached
`final_pre_export_stage3_gates`, i.e. the paper was fully extracted, inferred, audited and mapped —
work that at `C-122` never began.

Their new failures are **not** `C-125`'s and are not defects of it:

* `PMC13474940` → `no_defensible_reaction_support`. That is `F-179` refusing to export chemistry the
  paper does not support. **Per `PRODUCT_CONTRACT` § 1 a refusal to fabricate is a product success.**
* `PMC13123502` → `UGT91BP2`, `UGT703R1`, `UGT703R2`, `UGT703R3` carry no UniProt or DrugBank
  identifier.

---

# 4. The remaining repeat is DATABASE COVERAGE, not a resolver defect

Identity resolution keeps appearing as an *outcome*. Its *cause* was measured against live UniProt,
each protein queried under its own organism:

| protein | organism | UniProt hits |
|---|---|---:|
| `UGT91BP2` | *Paris polyphylla* | **0** |
| `UGT703R1` | *Paris polyphylla* | **0** |
| `UGT703R2` | *Paris polyphylla* | **0** |
| `UGT703R3` | *Paris polyphylla* | **0** |
| `ChoC` | *Rhizopus microsporus* | **0** |
| `Cho2` | *Rhizopus microsporus* | 1 |

Five of six have **no record at all** for their own organism. These are enzymes characterised *by the
paper being read* — that is what a pathway-elucidation paper is for — so no resolver can find them,
and admitting a near-name from another organism is precisely what the species safeguard exists to
refuse.

This matches the `C-122` pre-charter measurement exactly: **11 of 14 archived unresolved proteins
were also coverage.** Across both measurements, **16 of 20** unresolved proteins have no database
record for their organism.

> **So no repeated technical PWML-generation blocker remains that any code change can fix.** What
> repeats is the limit of what public databases contain about newly described enzymes, and the
> correct behaviour on that limit is the one the pipeline already has: ship the pathway with an
> honest placeholder, or refuse, rather than invent an identifier.

---

# 5. Process

Both merges `--no-ff`: `C-124` at `ffbb7305`, `C-125` at `6d4dffde`. Combined-tree gate:
**711 passed** across SMOKE's 22 files plus both new suites, `C-122`, `C-120` and `map_ids`, zero
survivors. Independent review: `C-124` two rounds, `C-125` three, every probe and flip table re-run
by the reviewer rather than taken from the implementer's report.

No leg was re-run for a better draw. No scope string was edited after seeing a result. No cache or
run directory committed. `main` untouched.
