# F-062 — measured disposition at integration `e616846`

Prepared by the Lead Orchestrator, 2026-08-21. Every claim below is labelled
**verified** (read from source or executed today), **stale record**, or **inference**.

---

## 1. The mechanism is byte-identical at tip — VERIFIED

`src/t2pw/pipeline/strict_quarantine.py`, read today at `e616846`:

```
2251:    coverage_reasons = [
2254:    structural_reasons: List[str] = []
2256:        structural_reasons.append(f"entity_type_overlap:{len(overlaps)}")
2258:        structural_reasons.append(f"degree_zero_export:{len(degree_zero)}")
2260:        structural_reasons.append(f"unexportable_entity:{len(unexportable)}")
2262:        structural_reasons.append(f"unaccounted_locked_reactions:{unaccounted_locks}")
2264:        structural_reasons.append(f"closure_not_converged:{int(max_iterations)}")
2267:    defensible_core = bool(verdict is not None and verdict.has_surviving_core)
2272:    review_reasons: List[str] = coverage_reasons if defensible_core else []
2273-2275: refusal_reasons = ([] if defensible_core else list(coverage_reasons)) + structural_reasons
```

`structural_reasons` is appended to `refusal_reasons` **unconditionally**, regardless of
`defensible_core`. That is exactly what F-062 described at `FINDINGS.md:1487`. The seam has
**not** been touched by any PACK 10 card. C-067's own merge message says so in terms: *"The
routing seam is byte-identical here."*

**So F-062's reading of the mechanism was correct and remains correct.**

## 2. F-062's proposed REMEDY is superseded — VERIFIED

F-081 (`FINDINGS.md:2851-2854`) supersedes it explicitly: *"F-062's mechanism is correctly
read and its reading of merge rule 7 is right in spirit. Its proposed fix — routing
structural reasons into `review_reasons` — is wrong, and this finding records why. F-062 is
not withdrawn; its remedy direction is."*

Two independent reasons, both verified from source:

1. **It would ship a review instruction to delete a connected enzyme.** Routing
   `degree_zero_export` to review makes `ok = true` and emits
   `pathway.review_required.pwml` carrying a review reason asserting no connectivity about
   the catalyst of a surviving reaction. Under merge rule 6 that is weakening a gate to
   increase production on a signal the module's own registry contradicts.
2. **THE SECOND SEAM.** `classify_release_status` encodes the same refusal independently —
   `strict_quarantine.py:2342-2345` computes `strict_gates_passed` separately from
   `refusal_reasons`, and `release_status.py:492-497` checks it **above** the coverage
   branch. Flipping `ok` alone yields `ok: true` with `release.status: diagnostic_only`, and
   because `ok` is the PWML production switch (`app/streamlit_app.py:4717` returns early
   when `not quarantine_result.ok`), that would **ship a final PWML on a `diagnostic_only`
   run** — breaching `PRODUCT_CONTRACT.md:343`.

**F-062's "one-line fix at `:2230-2233`" is therefore refused on evidence, not on policy.**

## 3. The correct repair was delivered by C-067 — VERIFIED

C-067 (merged `bb6bb6d`, closing F-081 and F-083) implemented the repair one layer down:
`_degree_zero_exports` now resolves names through `_entity_name_norms` — name UNION synonyms
— the same way `_build_registry`, `validate_registry_references` and `_prune_entities`
already did. Verified in current source at `strict_quarantine.py:1905, 1925, 1953, 1969`,
with the `exempt` construction given the same treatment.

The consequence, per F-081: on a converged run `degree_zero_export` is **empty by
construction**, so a leg that previously died on it reaches the coverage branch **on its own
merits** — which is what merge rule 7 asks for — **without touching the routing policy and
without relaxing any gate.**

## 4. The other four structural reasons are adjudicated `keep_refusing` — VERIFIED

An independent read-only biological adjudication ruled all four keep refusing, each for its
own reason (`FINDINGS.md`, F-081 "The other four reasons"):

| reason | why routing to review would be wrong |
|---|---|
| `entity_type_overlap` | references bind by `setdefault` to whichever bucket sorts first — an arbitrary, unrecorded guess. Fails `PRODUCT_CONTRACT.md:189-197`'s *"representable without guessing"*. |
| `unexportable_entity` | every member of the failure set requires inventing a fact, `placeholder_claims_real_identity` (a forged accession) most sharply. Named verbatim in the locked text. |
| `unaccounted_locked_reactions` | `locked_reactions_found` is a bare count (`pipeline/pipeline.py:1064`) retaining no id list, so a reviewer is handed *"3 locked reaction(s)"* and cannot find them. Nothing actionable. |
| `closure_not_converged` | a non-converged run is a mid-reduction snapshot, not a fixpoint, so `defensible_core` itself is unreliable — the seam's own precondition for routing to review is not established. |

**So the unconditional append at `:2273-2275` is CORRECT behaviour for the four reasons that
remain, and the one reason for which it was wrong has had its trigger removed at the
detector.**

## 5. Classification, against the four options

Using the takeover prompt's own taxonomy:

* **defect corrected** — no, not at the routing seam; the seam is unchanged.
* **original trigger removed** — **YES, for `degree_zero_export`**, by C-067. This is the
  operative one.
* **proposed remedy superseded** — **YES**, by F-081, on evidence.
* **different residual seam still present** — **YES but adjudicated correct**: the
  unconditional append survives and is the right behaviour for the remaining four reasons.

**Verdict: F-062 requires NO code card.** Its mechanism was real, its remedy was wrong, and
the correct repair has already merged. Writing an F-062 card now would re-open a seam an
independent biological adjudication ruled should not move.

## 6. What CANNOT be verified offline, and it is the honest limit

**Whether the two T-100 legs now actually reach `review_required` cannot be established
without a live re-run.** F-081 records the reason and it was verified rather than assumed:

> *"The quarantine input payload is not persisted"* — `admitted_payload_hash` on the report
> is `sha256:b22521ec9dfc4088`, while recomputing over the committed `final_mapped.json`
> gives `sha256:7e22a4662dbe2f61` and over `merged_payload.json` gives
> `sha256:a88b67690be2da81`. **Neither committed file is the payload quarantine judged.**

This is instrumentation gap 1 of 3 at `FINDINGS.md:1580`, and it is **UNOWNED**.

So the confirming measurement is a milestone, not a card: **T-104**. That is the correct
place for it — T-104's acceptance row already requires PMC12452463 to reach the
contractually required status.

**Residual risk, stated plainly:** F-081 holds its own core claim at **MEDIUM**, not HIGH,
and names what would overturn it — *"If the flagged row's synonym set is disjoint from
`keep_norms`, the theorem is wrong and there is a third divergence not yet found."* T-104 is
the run that would expose that. It should be triaged with this possibility explicitly in
scope rather than assumed away.

## 7. The dependency chain to T-104, exactly

```
Decision 1 (C-010 reading ratified)
  -> PRODUCT_CONTRACT.md:341 binds today
  -> T-104's acceptance row becomes quotable
  -> T-104 runs
  -> EITHER F-062 confirmed closed by C-067 (expected)
     OR a third divergence surfaces and is registered as a new finding
```

**Decision 1 is the only remaining blocker on this chain.** It is a product-owner
ratification, not an engineering question — the evidence is assembled at F-080.

## 8. One loose thread, registered not fixed

F-081 corrects a committed test's comment: `tests/test_strict_quarantine_release_seam.py:264-267`
claims `entity_type_overlap` cannot fire without emptying the graph. F-081 shows it is **not
true in general** — a protein `X` exempt as a component of a surviving complex, plus a
compound `X` referenced by a surviving reaction, gives an overlap with both rows surviving.

Verified present in current source (the comment block is at `:260-276` today). It is a
**comment**, not an assertion, so nothing is mis-gated by it. **Priority LOW.** Recorded here
so it is not lost; it does not justify a card of its own and should ride along with the next
card that touches that file.
