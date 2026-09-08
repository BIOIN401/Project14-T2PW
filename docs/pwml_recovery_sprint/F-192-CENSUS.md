# `F-192` census — read-only, over 144 archived legs

**2026-09-08.** Nothing was re-run, no LLM was called, no network touched, no run directory
written. Tool: `evidence/f192_census.py`. Cleanup report:
`evidence/g11/C-120/45-f192-census-v4.json`, `FINAL SURVIVING COUNT : 0`.

**Populations are never summed.** Each run family is reported separately.

| family | legs |
|---|---:|
| `runs_verify` | 121 |
| `runs` | 14 |
| `runs_smoke` | 5 |
| `runs_validation` | 2 |
| `runs_validation/c120` (this card's validation) | 2 |
| **total scanned** | **144** |

144 is the exact number of `final_mapped.json` files in the tree, so the scan is complete.

---

# THE HEADLINE, STATED AGAINST THE CRITERION THAT WAS SET

> **It does not repeat. One occurrence in 144 legs — and zero in the 142 legs that predate the
> `C-120` validation run.**

The stated rule for chartering was *"if it repeats across multiple legs, fix `F-192` next."*
**It does not repeat, so on the evidence that criterion is not met.** The recommendation in § 6
follows from that and not from any judgement about whether the proposed invariant is good — it is.

---

# 1. How often does `ensure_autostates` create a state that later disappears?

**Routinely — in about a fifth of all legs — and it is almost always harmless.**

| | legs | share |
|---|---:|---:|
| any biological state removed | 44 | 31 % |
| **`__auto_state__` specifically removed** | **32** | **22 %** |
| of those, a real state remained (defect masked) | **31** | 97 % of removals |
| of those, the leg ended with **zero** states | **1** | 3 % of removals |

`__auto_state__` removal by family: `runs_verify` 25 · `runs_smoke` 4 · `runs_validation` 1 ·
`c120` 2.

**`state_unreferenced_after_quarantine` is the ONLY biological-state removal reason anywhere in
the corpus.** There is no competing mechanism to disentangle it from.

The removal is normally *correct*: Stage 1 supplied real states, the location rows reference those,
and the unreferenced auto-state is swept as designed. The sweep is not the defect.

---

# 2. How many papers / legs does it block?

**One leg, one paper.**

| | |
|---|---|
| legs failing on `no_biological_states` | **1** |
| distinct papers | **1** — `PMC9544450` |
| family | `runs_validation/c120`, run `2026-09-08_1240` |
| legs ending with zero biological states | **1** (the same leg) |
| the same paper, ORCH-732, one day earlier | **exported a PWML, 39,686 B** |

**Zero instances in the 142 legs that predate this run.**

---

# 3. Would it otherwise have had a viable reaction core?

**Yes — and that is what makes the single instance worth recording rather than dismissing.**

| paper | mode | reactions | `gate_errors` | `blocking_issues` | outcome |
|---|---|---:|---:|---:|---|
| `PMC9544450` | strict | **5** | **0** | **0** | FAIL, 11 required-field errors |

Identity was clean on that leg: species *Escherichia coli* → `562` / `Prokaryote`; `MenD` →
`P17109` and `MenH` → `P37355`, both through the **pre-existing** `exact_symbol_identity` rescue.
The `C-120` rung never fired on it. **A biologically sound, fully identified 5-reaction pathway was
lost in its entirety at the last gate.** The failure mode is total, not partial.

---

# 4. Is it always the same ordering?

**There is only one instance, so no "always" can be claimed. More usefully: the ordering alone is
NOT sufficient — the census measures three near-misses that share most of the chain and survive.**

The at-risk population is legs whose **Stage 1 emitted no biological state of its own**:

| | legs |
|---|---:|
| Stage-1 payload readable | 144 |
| **Stage 1 emitted zero biological states** | **13** |
| of those, survived with ≥1 final state | **12** |
| of those, ended with zero | **1** |

Survivors' final state counts: `2`×7, `1`×2, `3`, `4`, `7` — i.e. states were supplied downstream.

### The 13 at-risk legs

| family | paper | mode | final states | auto-state removed | `audit_repair` rows | rx | status |
|---|---|---|---:|---|---|---:|---|
| `runs` | `PMC13231680` | strict | 2 | no | no | 1 | fail |
| `runs_verify` | `PMC13231680` | research | 2 | no | no | 1 | pass |
| `runs_verify` | `PMC12452463` | strict | 7 | no | no | 5 | fail |
| `runs_verify` | `PMC12452463` | research | 4 | no | no | 4 | pass |
| `runs_verify` | `PMC13231680` | strict | 2 | no | no | 1 | pass |
| `runs_verify` | `PMC13231680` | strict | 1 | no | **yes** | 1 | fail |
| `runs_verify` | `PMC12782028` | research | 1 | no | no | 3 | pass |
| `runs_verify` | `PMC12096016` | strict | 3 | no | no | 7 | fail |
| `runs_verify` | `PMC13231680` | research | 2 | no | no | 1 | pass |
| `runs_verify` | `PMC12452463` | strict | 2 | **yes** | no | 4 | fail |
| `runs_verify` | `PMC12180156` | strict | 2 | no | no | 1 | pass |
| `c120` | `PMC10031235` | strict | 2 | **yes** | no | 3 | pass |
| **`c120`** | **`PMC9544450`** | **strict** | **0** | **yes** | **yes** | **5** | **FAIL** |

### The conjunction, read off that table

Three conditions must hold together, and **each on its own is survivable**:

1. Stage 1 emits **no** biological state — 13 legs, 12 survive.
2. `__auto_state__` is removed as unreferenced — 32 legs, 31 survive.
3. The element-location rows were written by **`audit_repair`** — 2 of the 13 at-risk legs, and
   **one of those two survived** (`PMC13231680`, which kept its auto-state because it was still
   referenced).

**All three co-occur exactly once in 144 legs.** So the mechanism in `F-192` is correct as
described, but the *sufficient* condition is the conjunction, not the ordering, and the census
cannot show the conjunction is stable — n = 1.

---

# 5. One entity type or several?

**Several — every visible element-location bucket present.**

| bucket | orphaned rows |
|---|---:|
| `compound_locations` | 8 |
| `protein_locations` | 2 |

Both buckets that exist in the payload were hit. Nothing suggests the defect is type-specific: the
auto-state assignment loop in `ensure_autostates` iterates `["compound_locations",
"protein_locations"]` uniformly, so any bucket whose rows are rewritten later loses the assignment.

---

# 6. Recommendation

**Do not charter `F-192` on this evidence.** The bar set for it was repetition across multiple
legs; the measurement is **1 in 144**, and **0 in everything that predates the run where it was
found**. Chartering a production change on a single occurrence is the "fix five unrelated edge
cases" failure the sprint has repeatedly refused, and production is re-frozen at `045447c8` under
`D-098`.

**What would change that judgement**, in order of how cheaply it can be obtained:

1. **A second independent instance.** Any future leg failing on `no_biological_states` should be
   treated as the confirming observation and charter immediately. It is now cheap to detect —
   re-run `evidence/f192_census.py`.
2. **A product-owner decision on severity over frequency.** The one observed instance destroyed a
   complete, correctly identified 5-reaction pathway with zero gate errors. If the product owner
   weights "a perfect leg silently yields nothing" above frequency, that is a legitimate basis to
   charter — but it is a *product* call, not something the data decides.

**If it is chartered, the proposed invariant is the right one and should be adopted as worded:**

> after any audit/remap mutation that can rewrite element-location rows, required auto-generated
> biological states must be re-established before quarantine / final export validation.

It changes no threshold, relaxes no gate, and adds no new state type. Two cautions for whoever
writes it:

* **It belongs upstream of the freeze.** Merge rule 8 forbids an exporter repairing biology after
  the canonical graph is frozen, so the re-establishment must happen in `process_normalizer.py`
  before quarantine — not in `pwml/ir.py` and not in the writer.
* **The regression surface is the 31 masked legs.** Re-establishing the auto-state must not cause
  it to *survive* where it is currently swept correctly, or those legs gain a spurious state and
  the graph hash moves. The A/B to run is: do all 31 masked legs and all 12 surviving at-risk legs
  end with an identical `biological_states` list before and after? Anything else is a regression.

---

# 7. A defect in this census, disclosed

**The first version of this census reported `blocked = 0` and would have closed `F-192` as
"never happens".** Its leg discovery used a fixed-depth glob, `*/papers/*/*`, which silently missed
`runs_validation/c120/<stamp>/papers/…` — one extra directory level, and the only tree in the whole
corpus containing the defect. It scanned 142 legs and reported the count without noticing that 144
`final_mapped.json` files exist.

Corrected to match on shape (`…/papers/<paper>/<mode>`) rather than depth, and the leg count is now
reconciled against an independent `find` count. Recorded because a census that undercounts in
exactly the region of interest is worse than no census, and because the same fixed-depth assumption
is present in other sprint tooling that has never been checked against a nested run family.
