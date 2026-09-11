# `F-147` is live: four legs were failed on a report describing a payload that no longer existed

**2026-09-10.** Read-only. Triggered by the product owner asking where PathWhiz's requirements for
unresolved proteins are documented, and pointing out that such proteins should be put into a complex.
**They were right, the mechanism works, and finding that is what exposed this.**

> ## This corrects my own answer of an hour ago.
> I reported that no repeated technical PWML-generation blocker remained that code could fix, and
> that the identity failures were database coverage. **The coverage fact is true and is not the
> reason those legs failed.** Four legs across three cohorts were failed on a superseded
> `post_normalization` report while their **final** stage-3 gate passed clean with zero errors.

---

# 1. What PathWhiz actually requires, and what we already do about it

[`docs/pathwhiz_requirements.md`](../pathwhiz_requirements.md) § *Protein* / *Protein complex*,
corroborated by a direct PathWhiz UI observation recorded in `change_log.md`:

* **New Protein** requires `Name`, `Species`, and **either** a UniProt **or** a DrugBank ID.
  Gene name, EC number and sequence never substitute.
* **New Protein Complex** requires `Name`, `Species`, and ≥1 member protein with stoichiometry.
* A generated complex needs **no** PathWhiz complex ID, provided each component protein has
  species + UniProt/DrugBank.
* A reaction's enzyme **must be a `protein_complex`, never a bare protein**.

So a protein with no accession cannot be created in PathWhiz directly. The project's answer is
[`docs/protein_export_policy.md`](../protein_export_policy.md) § B, **the Unknown-backed functional
complex**: the functional name is preserved on a generated `protein_complex` whose only component is
**PathBank's real Unknown protein, row 9659, UniProt `Unknown`** — which satisfies the New Protein
form legitimately, because it is an existing database row rather than an invented identifier. The
reaction points at the functional complex name, never at the literal `Unknown`, and the row is
marked `identity_status: "placeholder"` and never counted as a real mapping.

**That is exactly the product owner's recollection, and it is implemented and working.**

---

# 2. It fired correctly on every leg examined

| leg | unresolved proteins | shipped as |
|---|---|---|
| `PMC13123502` | `UGT91BP2`, `UGT703R1`, `UGT703R2`, `UGT703R3` | four Unknown-backed complexes, names preserved |
| `PMC13089919` | `ChoC`, `Cho2` | two Unknown-backed complexes |
| `PMC13184244` | `UGT1`, `β-GD1`, `MATE1` | Unknown-backed complexes + `A622 complex`, `BBLa complex` |
| `PMC4471609` | `DmaW`, `EasF`, `EasE`, `EasC`, `EasD`, `EasA`, `EasG`, `EasH` | eight Unknown-backed complexes |

In each, `entities.proteins` ships **exactly one row** — the sentinel itself:

```
name 'Unknown'   uniprot 'Unknown'   pathbank_protein_id 9659
```

Every component list is `['Unknown']`. The policy behaved precisely as written.

---

# 3. Why the legs failed anyway — the three proofs

**Proof 1 — the final gate passed.** `final_stage3_gate_report.json` on all four:
`ok: true`, `errors: []`.

**Proof 2 — the failing report cites pointers the shipped payload cannot host.**

| leg | report cites | shipped `entities.proteins` length | unhostable |
|---|---|---:|---|
| `PMC13123502` | `/entities/proteins/0..3` | 1 | indices 1–3 |
| `PMC13089919` | `/entities/proteins/0..1` | 1 | index 1 |

A pointer a list cannot host is by itself proof the report describes a payload that no longer exists.

**Proof 3 — every downstream report is clean.** `post_audit_contract_report` `ok=True` ·
`post_remap_contract_report` `ok=True` · `pre_export_runtime_schema_report` `ok=True`.
**Only `post_normalization_contract_report` — an early stage — is false, and that is what the driver
honoured.**

This is `F-147` exactly, as diagnosed in
[`ORCH-725-IDENTITY-DIAGNOSIS.md`](ORCH-725-IDENTITY-DIAGNOSIS.md) § 1, which measured it as the
**dominant** blocker of that pilot — 4 of 6 evaluable legs — and recorded the same signature,
including *"not one of the 19 entities the failing reports name is still a bare protein in the
shipped payload. Eleven became Unknown-backed functional complexes."*

---

# 4. The census, stated precisely

Across four recent cohorts, legs that **failed on the stale report** while their final gate was clean:

| cohort | leg | issue codes the driver reported |
|---|---|---|
| `ORCH-734` | `PMC13184244` | `gate.protein_ugt1_…`, `gate.protein_gd1_…`, `gate.protein_mate1_…` |
| `C-122` final smoke | `PMC13089919` | `gate.protein_choc_…`, `gate.protein_cho2_…` |
| `C-125` validation | `PMC13123502` | `gate.protein_ugt91bp2_…` ×4 |
| `C-121` live validation | `PMC4471609` | `gate.protein_dmaw_…` ×8 |

**Four papers, three independent cohorts, one deterministic mechanism.**

Other failures in the same cohorts are **not** this and are correctly classified: `no_biological_states`
(`PMC11961743` ×2), `no_defensible_reaction_support` (`PMC13405594`, `PMC13438895`, `PMC13474940` —
`F-179` working), `scope_conflict` (now fixed by `C-125`), `species_missing_taxonomy` (now fixed by
`C-124`).

## What is NOT claimed

**Not that these four would each have produced a PWML.** A clean final gate and clean downstream
reports mean they had a real chance at export, not a guarantee — remaining serialization steps could
still refuse. **No PWML count is claimed and none should be quoted.**

---

# 5. What this means

**The identity-coverage finding stands and is unchanged**: 16 of 20 unresolved proteins genuinely
have no database record for their organism, and no resolver can find them. **What is wrong is my
conclusion that this is why the legs failed.** It is not. The placeholder policy absorbed exactly
that situation, correctly, and then a stale report discarded the result.

**So the answer to "is a repeated technical blocker left?" is YES, and it is `F-147`** — repeated
across four papers and three cohorts, deterministic, and unlike database coverage it is fixable in
code. By the product owner's own § 20 rule, a repeated major deterministic PWML blocker is the one
thing that may reopen reliability.

**Nothing is chartered here.** The scope of a fix — whether the driver should read the final gate
rather than an early contract report, or whether the early report should be superseded when a later
one passes — is a product-owner decision and needs its own narrow card with a base-failing proof.
