# Rulings — ORCH-717 wave close

Product-owner rulings received this wave, and the Lead's resolution of each, **with the measurement
that supports it**. Every number here I took myself at integration tip `6c20879` through
`bounded_run.py`; none is copied from a report.

Companion to `DECISION-PACKET-ORCH717.md`, which stated the questions. **The packet is preserved
unchanged.** This file records what was decided and what remains open.

---

## Q1 — negative-control scoring · **RULED BY PRODUCT OWNER** · instrument card required

### The ruling as received

A case explicitly designated by gold as `context_only` **or an equivalent negative control** must
**pass its semantic expectation** when all three hold:

1. the pipeline releases **no pathway reactions**;
2. it provides the **required rejection or empty-pathway reason**;
3. the empty result is **not** caused by a timeout, crash, missing artifact, or infrastructure
   failure.

It must **not** be scored as a normal positive pathway requiring reaction production. It is reported
under an explicit status — **`PASS_NEGATIVE_CONTROL`** — or the existing schema's closest explicit
equivalent.

Binding constraints carried with the ruling:

- preserve the **raw outcome** and the **rejection reason**;
- **do not manufacture PWML**;
- **do not reward an empty result caused by execution failure**;
- **do not change the official T-107 verdict**;
- test **positive controls, true negative controls, timeouts, missing artifacts, and accidental
  empty outputs separately**.

### This supersedes the packet's option A, and does not conflict with its "never PASS"

The packet recommended option A (`DECLINED (expected)`) and rejected a bare `PASS`, on the ground
that a bare `PASS` would make *"produced nothing"* indistinguishable from *"produced the right
thing"*. **`PASS_NEGATIVE_CONTROL` is an explicit, distinguishable token, so the packet's objection
does not reach it.** The ruling and the packet's reasoning agree.

### Can the current scorer represent this? — **PARTLY. A card is required.**

Measured against shipped code:

**What already exists, and must be reused rather than rebuilt:**

| Already present | Where |
|---|---|
| `case.is_negative_control` — *"graded on one thing only: did the pipeline decline to invent a pathway?"* | `bench/goldset.py:505` |
| `_empty_is_correct(case)` — **already the ruling's own predicate**, covering the negative control *and* any `context_only` case with no minimum core | `bench/acceptance.py:1530` |
| Negative controls already excluded from the extraction-blocker ranking, with `PMC13231680` named in the comment | `bench/acceptance.py:1575` |
| The renderer already tags `[NEGATIVE CONTROL]` | `bench/render.py:298` |

**What is missing, and is why a card is needed:**

1. **There is no explicit status token.** The leg's status still reads as a failure. The scorer knows
   empty was correct and says so in the *blocker ranking* only.
2. **`RESULT: FAIL` is emitted somewhere that cannot know any of this.**
   `batch/runner.py:717` computes

   ```python
   verdict = "PASS" if status == _STATUS_PASS else "FAIL"
   ```

   inside `result_text(row, paper=...)`, which receives a **manifest row and a paper dict — not a
   `GoldCase`**. The seam that prints the misleading token **has no access to the gold set at all**,
   so it cannot currently distinguish a negative control from a failure even in principle.
3. **Nothing distinguishes "empty because it declined" from "empty because it timed out".** The
   ruling requires exactly that separation, and **F-148 is the standing evidence that a timed-out
   leg preserves the stop reason and little else.** These two interact and the card must say so.

**Ruling: the current scorer CANNOT represent this correctly. Chartered as C-110.**

---

## Q3 — PathBank compound ids · **RULED BY THE LEAD: NOT a real accession. No code change.**

Q3 asked whether a `pathbank_compound_id` is a real accession for Priority 1, and the packet held
Q2's prediction open until it was answered.

### The measurement — `evidence/orch717_q3_pathbank.py`, G11 `ORCH-717/18-q3.pathbank2.json`

```
_external_ids WITH    pathbank_compound_id -> {drugbank, hmdb, kegg, chebi, pubchem}
_external_ids WITHOUT pathbank_compound_id -> {drugbank, hmdb, kegg, chebi, pubchem}
recognised namespaces      : ['chebi', 'drugbank', 'hmdb', 'kegg', 'pubchem']
pathbank recognised?       : False
identical with vs without? : True

bool(ids) WITH pathbank : True     bool(ids) WITHOUT pathbank : True
-> the predicate is UNCHANGED by the PathBank question

data/pathwhiz_id_db.json compounds.by_id rows : 55
id space sample : ['4', '5', '6', '7', '8', '9', '10', '414']
every id is a bare integer : True
```

### Three findings, in the order that decides the question

**1. `_external_ids` recognises no PathBank namespace today.** It reads exactly
`uniprot, drugbank, hmdb, kegg, chebi, pubchem`. So the status quo already *is* "PathBank ids are not
accessions".

**2. Q3 cannot move Q2's arithmetic, and the packet's coupling dissolves.** The affected row carries
**five recognised accessions** — `hmdb`, `kegg`, `chebi`, `pubchem`, `drugbank` — with the PathBank
id removed **entirely**. The Priority-1 `false_real` branch is guarded by `if ids:`, which asks
whether the row carries *any* recognised accession, not a particular one. **`bool(ids)` is `True`
either way.** The packet's five-of-nine count was right, and it is the five that decide it.

**3. A conditional-acceptance policy is not implementable here, and the guardrails forbid the only
available mechanism.** The product-owner guardrails require that a PathBank id be accepted *only*
when backed by a **real resolved row matching the candidate identity rather than merely sharing a
name fragment**, and that **arbitrary numeric IDs are never accepted**. But:

- the entire local id space is **bare small integers** over a **55-row** table — every id in it *is*
  an arbitrary numeric id;
- the committed exporter record
  (`evidence/probe_exporter_identity_mutation_2026-08-06.md`) shows these ids are produced with
  `db_status = matched_offline_name_index` and `chosen_rule = legacy_pathwhiz_id_unverified` — an
  **offline name-index match carrying an unverified legacy id**, which is precisely the
  "merely sharing a name fragment" case the guardrail excludes.

**So the conservative option is the status quo, and it satisfies every guardrail without a line of
code:**

| Guardrail | Status quo |
|---|---|
| accepted only when backed by a real resolved row | satisfied — none is accepted |
| resolved row must match identity, not a name fragment | satisfied — the only available mechanism *is* name-index matching, so nothing is accepted through it |
| arbitrary numeric IDs never accepted | satisfied — the whole id space is bare integers |
| fabricated/unresolved identifiers remain findings | unchanged (see residual below) |
| gold-forbidden identifiers remain forbidden | unchanged |
| exported forbidden identifier earns no positive coverage credit | unchanged |
| raw and contract-adjusted separately visible | unchanged |
| does not make Priority 1 pass by hiding a regression | nothing is hidden; nothing moves |
| A/B on the exact affected population | done, above — the predicate is **UNCHANGED** |
| unrelated papers and entity types do not move | nothing moves |

**Ruling: `pathbank_compound_id` is not a real accession for Priority 1. `_external_ids` is correct
as shipped. No card, no code change.** This is the most conservative outcome available and it
requires no production change, which is why it is preferred over any policy that would have to be
invented.

### One residual, REGISTERED not fixed

A row carrying **only** a PathBank id would read as *bare* to `_external_ids`, and D-074's sentinel
tolerance turns on bareness. **It has no live exposure today** — `acceptance.py`'s own docstring
records that the tolerance branch is *"UNREACHABLE TODAY, AND THAT IS THE RULING'S SHAPE, NOT A
BUG"*, because its one call site sits inside `if ids:`. **Registered as a property to re-check if
that condition is ever loosened. Not chartered, and loosening it to make it fire is refused.**

---

## Q2 — F-150 · half 1 unblocked by Q3; half 2 remains a product-owner question

### Q3 no longer constrains it

The packet said the Lead *"does not treat Q2's prediction as final until [Q3] is resolved"*. **It is
resolved and it does not touch the arithmetic** — finding 2 above. Q2 half 1 therefore stands on its
own merits and goes to Wave 4's conditional authority, where it needs an **independent reviewer** and
the **four-step A/B**, not a further ruling here.

### Both halves re-verified by me this wave

`evidence/orch717_f150_verify.py`, G11 `ORCH-717/16-f150.reverify2.json`:

```
forbidden_match('5-aminolevulinic acid'    ) -> '5-aminolevulinic acid'
forbidden_match('delta-aminolevulinic acid') -> None
forbidden_match('δ-aminolevulinic acid'    ) -> None
forbidden_identifiers[0].aliases : ['ALA', 'porphobilinogen', 'protoporphyrin IX',
                                    'succinyl-CoA', 'coproporphyrinogen III',
                                    'uroporphyrinogen III']

supported_reactions_complete   TRUE=0  FALSE=10  MISSING=0
max_retained_reactions set on exactly two cases: PMC12180156 (2), PMC13231680 (0)
```

**Half 1 confirmed:** the `delta` / `δ` spellings are absent from the alias list, so the run's worst
false accession was never counted. The gold author already used the delta spelling elsewhere **in
the same case** (`acceptable_enzymes[1].aliases`), which makes it an oversight, not a policy.

**Half 2 confirmed:** Priority 2 is evaluable only through `max_retained_reactions`, set on exactly
two cases, **both negative controls**.

> **Priority 2 = 1 is a real number and it is not a measure of how much invented chemistry T-107
> produced. Any report quoting it must carry that limit.**

### The one exact product-owner question that remains open

> **Should `supported_reactions_complete` be set on any gold case — and if so, which?**
>
> It is the only change that would let Priority 2 measure anything on a paper that is not a negative
> control. **It is a product decision about what the benchmark MEANS on every future run, not a data
> correction**, and no measurement this wave can settle it. **Not chartered. Not implemented.**

**Nothing else in Q2 is blocked on this.** Half 1 proceeds under Wave 4's rules; half 2 waits.

---

## Unchanged and protected

- **T-107** — not rerun, not rescored, not reinterpreted. **`NOT ACCEPTED` stands.**
- **F-147** — registered, deliberately **not chartered**. The earliest unsafe seam is **Stage-1
  extraction**, not the driver; a downstream-only fix would flip two legs to PASS that would then
  export gold-forbidden content. **Merge rule 6.**
- **`placeholder_backed_proteins` / `Unknown`-backed export** — `PRODUCT_CONTRACT` § 13 standing
  disagreement. **Escalate only. Untouched.**
- **The gold file** — unmodified at the time of writing.
