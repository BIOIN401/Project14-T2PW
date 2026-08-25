# Decision bundle — post-T-106

**Prepared by the Lead Orchestrator, 2026-08-25**, at integration `03f60b0` on `sprint/pwml-recovery`.

Three items need the product owner. **None of them is authored into `DECISIONS.md` by me** — the
orchestrator does not write locked decisions. Each section below gives the current behaviour, the
options, a recommendation, and, where the ruling is really just recording a rule already in use,
**exact text the product owner can paste**.

Item 1 is a genuine ruling and is the **final T-107 blocker**. Items 2 and 3 are control-plane
tidying that block nothing; they are here so all three can be taken in one sitting.

---

## Item 1 — F-107 / D-062: what happens to a defensible core that a scope guard stopped

**Status: BLOCKS T-107.** Everything else in the acceptance set is either fixed, in flight, or
measured non-blocking. This is the one that is not.

### Current behaviour, measured on T-106

Six legs across three papers end `scope_conflict` at Stage 0 because the batch requested
*Bacillus subtilis* and the papers are *E. coli*, *L. lactis* and *L. monocytogenes*. All three are
**deliberate gold organism traps** — each `relevance_note` says so in capitals and each case lists
`Bacillus subtilis` in `forbidden_organisms`.

All six now carry a real release record (C-077's work), and it reads:

```
status               : diagnostic_only
pipeline_executed    : true
strict_gates_passed  : false
semantic_evaluation  : not_evaluated
reasons              : ['stage0_scope_conflict_stopped_the_run_before_serialization',
                        'strict_technical_gates_blocked_export']
```

`requested_scope` sits beside `observed_context`, so the mismatch is auditable. No PWML is written;
the run stops before audit, DB mapping, freeze and export.

### Why a ruling is needed

**D-062 (LOCKED) says the pathway is preserved "as `review_required`, carrying the OBSERVED
organism".** That state is **not constructible at that seam.** The C-077 reviewer established this
two independent ways: by reading `classify_release_status` (one `elif` chain whose second arm pins
`diagnostic_only` whenever the strict gates did not pass, with all five `REVIEW_REQUIRED` sites
below it), and by a **196,800-combination sweep** of the documented input surface with
`strict_gates_passed=False` pinned — `diagnostic_only` 196,800 times, `review_required` zero, against
a control arm at `True` returning 24.

Reaching `review_required` would have required fabricating a gate result for gates that never ran.
C-077 declined to and escalated. **That was correct and must not be charged against it.**

`PRODUCT_CONTRACT` § 4 also defines `review_required` as *"Valid, useful PWML **produced**"*, and no
PWML exists here. So **both** candidate states are inexact, and § 4 simply **has no state for
"a defensible core was extracted but never serialized because a scope guard correctly stopped the
run."** D-062 assumed one existed.

The core really is defensible. `PMC12421875/strict` reached a connected core of **9** against the
gold's own floor of **7**, with **8/8** enzyme and **10/10** metabolite recall, and every semantic
check `[ok]` — including `no_real_id_or_name_conflict`.

### And the half D-062 explicitly left open — this is what pins priority 5 at 0/4

D-062's closing section:

> The gold's `expected_export: strict_exportable` for `PMC12657337` and `PMC12421875` is **not**
> ratified by this ruling. Under D-062 the correct outcome for those two is `review_required`, so
> either the gold field or this ruling will need reconciling… **That reconciliation is a separate
> decision and is explicitly left open here.** Until it is taken, neither paper counts as a
> strict-export success and the strict denominator is unchanged.

So two of the four papers in priority 5's denominator are **forbidden by locked policy from ever
passing**, while the gold field that puts them in the denominator says they should. Priority 5
cannot be honestly scored until that contradiction is settled either way.

### The other two are correctly blocked — measured, not assumed

An earlier record of mine (`03f60b0`) claimed the ceiling was 1/4 and that `PMC12096016` hinged on a
single predicate. **That was wrong and is corrected in the LEDGER.** `release_status.py`'s caps are
independent by construction and each is guarded on `status == RELEASE_READY`, so **whichever fires
first hides the others** — a single entry in `semantic_failed_checks` does not mean a single blocker.

A counterfactual replay of the real `classify_release_status` over each leg's own recorded
`coverage_summary.json`, whose control arm reproduced **9 of 9** recorded statuses exactly, forced
the semantic verdict to `passed` on all four:

| paper | recorded | **semantics forced to `passed`** | reason it still fails |
|---|---|---|---|
| PMC12096016 | `review_required` | **`review_required`** | `requested_core_anchors_unmatched:NADH,ATP,MenD…,Fur…` (`:730`) |
| PMC12782028 | `review_required` | **`review_required`** | `requested_core_coverage_below_minimum:0.222<0.500` |
| PMC12421875 | `diagnostic_only` | **`diagnostic_only`** | `strict_technical_gates_blocked_export` |
| PMC12657337 | `diagnostic_only` | **`diagnostic_only`** | `strict_technical_gates_blocked_export` |

**Priority 5's ceiling under unchanged policy is 0/4.** Not one of the four moves even if every
semantic check passes. `PMC12096016` and `PMC12782028` are `correctly_blocked`; the actor check that
demoted `PMC12096016` is a **true positive** on a real biological error (see F-116), so weakening it
would be exactly the inversion this sprint exists to correct.

**Consequence: zero of the four are reachable by code that does not weaken a gate or manufacture bare
PWML. Every route to a non-zero priority 5 runs through a product ruling first.**

### The options

| | Option | What it means | Priority-5 effect | Risk |
|---|---|---|---|---|
| **A** | **Add a fourth state to `PRODUCT_CONTRACT` § 4** — e.g. `extracted_not_serialized` — and correct the gold's `expected_export` for the two trap papers to match D-062 | The record becomes accurate without inventing a gate result. The papers leave the strict denominator. | Denominator 4 → 2. The rate becomes honest rather than structurally impossible. | Lowest. No production code moves; no gate weakens. Requires a gold edit, openly recorded. |
| **B** | **Drive the run onward to serialization under the OBSERVED organism**, emitting `pathway.review_required.pwml` | Delivers D-062's literal wording. | Denominator unchanged at 4; still not strict successes, since D-062 forbids that. | **Highest.** Requires a second new ruling on what audit and DB mapping may do when the organism is known wrong, and it is the one shape that could accidentally reach the strict export D-062 forbids outright. **It also makes F-110 live**: `PMC12312563` carries `Mg2+` in both modes and would newly reach the name gate. |
| **C** | **Ratify the status quo** — `diagnostic_only` stands, and amend D-062's wording to match what the seam can support | Cheapest. Nothing moves. | 0/4 stands and is declared correct. | Leaves § 4's `diagnostic_only` gloss (*"recovery and retrieval could not establish a defensible pathway core"*) saying something untrue of these legs, which is the untruth C-077 was chartered to remove. |

### Recommendation: **A**

It is the only option that makes the record true without either fabricating a measurement or taking
a second, larger ruling. It costs no production code, weakens no gate, and it converts priority 5
from a metric that **cannot** be satisfied into one that can be honestly measured. Option B may
eventually be right, but it should be its own sequenced card with its own F-110 budget — not a
prerequisite for T-107.

**Safety note, unchanged under all three options:** none of them may raise the strict rate by
letting a mis-scoped paper export strictly. D-062 forbids that outright and this bundle does not
reopen it.

**Be clear about what A does and does not buy.** It takes the denominator from 4 to 2 and makes the
metric honest. It does **not** by itself produce a strict success: priority 5 would read **0/2**,
because `PMC12782028` is correctly blocked on coverage and `PMC12096016` is correctly blocked twice
over. A non-zero priority 5 additionally requires the **F-116** enterobactin-complex mapping to be
corrected *and* a separate look at cap 2's input quality (see the tension below). **T-107 must not be
authorised on an expectation that priority 5 passes.**

### The exact gold edit, if A is taken — prepared, NOT applied

In `src/t2pw/bench/gold/pinned_v1.json`, for **`PMC12421875`** and **`PMC12657337`** only:

```
  "expected_export": "strict_exportable"   ->   "expected_export": "partial_only"
```

and append to each `export_rationale`, leaving the existing text byte-identical:

> ` Reconciled with D-062 (LOCKED, 2026-08-22): a Stage-0 organism conflict whose reading is correct preserves the pathway for review under the OBSERVED organism and is never a strict export. The organism-labelling requirement above remains the scored property of whatever artifact this paper does produce.`

**Three things this edit must not do.** It must not touch `forbidden_organisms` — the trap must stay
exercisable. It must not touch `relevance_note` — the ORGANISM TRAP designation is the point of the
cases. It must not touch any topics file: D-062 forbids that in terms, and
`bench_acceptance.py --verify-plan` must keep returning `OK` with all ten `[pinned_override]`.

`PRODUCT_CONTRACT` §12 `:331-332` requires `expected_export` to be an explicit gold field precisely
so *"a silent default"* cannot move papers out of the strict denominator *"without anyone deciding
to."* This edit is the opposite of a silent default — it is an explicit decision, recorded, and
traceable to D-062. **I have not made it.** Silently editing gold to improve a rate is precisely what
this sprint exists to prevent.

### A live tension the ruling should also see

`PMC12096016`'s cap-2 demotion cites unmatched anchor **`MenD`** — which that paper's own gold
`export_rationale` says *"Export must **exclude** MenD."* The pipeline is being demoted for correctly
omitting something the gold forbids. `Fur` is a regulator; `ATP` and `NADH` are cofactors.

Cap 2's input is Stage 0's `key_compounds` / `key_proteins`, **not a curated core**, and that draw is
non-deterministic: the same paper's `missing_anchors` was `[ATP, NADH, NAD+, EntA, Fur]` at T-104,
`[ATP, EntD]` at T-105 and `[NADH, ATP, MenD, Fur]` at T-106 — coverage 0.706 / 0.857 / 0.765 across
three runs. The cap is a merged F-094 correction and is **not** reopened here, but the quality of its
input is a real and separate question currently costing a leg with 5/5 enzyme and 7/8 metabolite
recall.

---

## Item 2 — F-109: `pytest.ini` carries a setting `TEST_MATRIX` rule 10 refuses

**Blocks nothing.** Documentation-only. Every base proof this wave was pinned, so no measurement is
in doubt.

### The contradiction, both halves

`TEST_MATRIX.md:101-110`, rule 10:

> `pytest.ini` **must not** gain `pythonpath = src`: it was considered as a remedy for F-003 and is
> **refused**, because pytest *prepends* those entries, so it would sit ahead of the `PYTHONPATH`
> pin and **make every base-tree G9 proof silently measure the tip**.

`pytest.ini:1-8` has carried `pythonpath = src` since **C-070** (`5bc600e`), added for a real and
unrelated defect: 21 of 156 test files could not be collected individually.

Neither side is wrong on its own merits and neither author knew of the other. Not chargeable to
C-070 or C-079.

### Why it has not bitten

The standing mitigation already in force: **every base-tree measurement runs through
`pinned_pytest.py` with `--expect-tree` and a committed `--pin-verdict`**, which resolves `t2pw`
and refuses with **exit 98** if it lands in the wrong tree. `PYTHONPATH` is not evidence; the
resolved path written to a committed verdict is. That is what makes the C-070 setting survivable.

### Recommendation

**Keep `pythonpath = src`; amend rule 10 to record the refusal as superseded and name the pin as the
required mitigation.** Removing it would re-break individual-file collection for 21 test files to
solve a problem the pin already solves. Suggested amendment:

> `pytest.ini` carries `pythonpath = src` (C-070, `5bc600e`), which this rule previously refused.
> The refusal is **superseded, not forgotten**: the hazard it names is real — pytest prepends the
> entry ahead of `PYTHONPATH` — and is neutralised by the mandatory pin, not by the setting's
> absence. **Therefore: every base-tree measurement MUST run through `pinned_pytest.py` with
> `--expect-tree` and a committed `--pin-verdict`. An unpinned base run is not evidence.**

Also worth folding in here, from **F-114**: a `--basetemp` whose **parent** does not exist errors the
run outright (one measured instance errored 55 tests; creating the parent gave `339 passed`). That
is a second infrastructure mode that looks exactly like a large regression, and it belongs beside
the existing `PermissionError` note in § 0.

---

## Item 3 — F-113: the identity ruling that two merged cards rest on has no `D-xxx` entry

**Blocks nothing.** Pure control-plane gap.

The product-owner identity ruling of **2026-08-23** is quoted verbatim in `prompts/C-076.md` § 1 and
referenced throughout, but lives only in card prose, the LEDGER's C-076 row and `FINDINGS.md`. **Two
merged cards rest on it** — C-076 (`3b7a7b1`) and C-080 (`89aaced`) — and C-073 was corrected
against the D-035 clause it interprets.

### Exact text for the product owner to append

Recording a rule already in force. This does **not** reopen C-073, C-076 or C-080.

> ## D-064 — a shared accession within one kind is identity, not conflict · 2026-08-23 · LOCKED
>
> **Product-owner ruling, taken 2026-08-23. Recorded retroactively 2026-08-25** to close F-113: the
> ruling was already governing merged production code and belonged in the locked-decisions file.
> Recording it changes nothing and reopens nothing.
>
> The same UniProt accession may be shared by proven aliases of the same protein and by holo/apo
> states of the same underlying polypeptide. `EntE` and `enterobactin synthase` are the same protein
> identity. Holo-EntB and apo-EntB may share the underlying UniProt accession while remaining
> distinct pathway states. **Do not flag these as accession conflicts unless the entities are
> biologically unrelated or cross-kind. Update the scorer/gold classification rather than forcing
> the pipeline to invent different protein identities.**
>
> ### Implemented by
>
> C-076 (`3b7a7b1`, scorer + gold) and C-080 (`89aaced`, the production release gate, reading the
> same identity predicate). C-073 was corrected against the D-035 clause 3c this ruling interprets.
>
> ### A known and deliberate gap in the implementation
>
> The wording is *"biologically unrelated **or** cross-kind"*. **Both seams implement cross-kind
> only**, because neither has a biological-relatedness oracle. Two genuinely unrelated same-kind
> proteins fused onto one accession by a mapper bug are invisible to the scorer **and** to the
> production gate. That mirrors the pipeline's pre-existing blind spot and is what the C-076 charter
> directed, so it is not a deviation — but the "unrelated within one kind" half is **unmeasured
> corpus-wide** and is recorded here as a known gap, not an assumed non-issue.

---

## What is NOT in this bundle, and why

* **F-096** — in flight as C-081. A code defect, not a decision.
* **F-115** — in flight as C-082. A code defect, not a decision.
* **F-110** — **measured non-reachable on T-106.** Zero occurrences of `no_shared_meaningful_token`,
  `identity_refused_review_required` or `shipped_identity_name` across the whole run directory
  including `batch.log`; every ion- or formula-named row sits in a leg that stopped before DB
  mapping. No card is justified and no ruling is needed. **But note the coupling in Item 1 option
  B** — that option makes F-110 live.
* **F-092 defect 3** — charter written (`prompts/C-083.md`), not yet dispatched. Its two sibling
  claims are refuted and stay refuted.
* **F-116** — newly registered: `_rewrite_reaction_protein_enzymes_to_complexes`
  (`mapping/map_ids.py:8668`) resolved `EntE` onto a **superset** PathBank complex (3623:
  `EntB, EntD, EntF, EntE`), injecting three catalysts that do not perform EC 6.2.1.71, and collapsed
  two chemically distinct steps onto one actor. Implicated in **3 of the 6** strict legs that reached
  mapping, and the same generator produced PMC12452463's Stage-3 gate deaths. **This is a code
  defect, not a decision** — but the product owner may want to weigh one competing reading: the
  enterobactin synthase assembly line genuinely does perform both steps *in vivo*, so the attribution
  may be imprecise rather than chemically false. Correcting it strengthens biology; it does **not**
  unblock the leg (cap 2 fires anyway).
* **F-117** — newly registered, LOW: the actor gate cannot relate a DB canonical name to a bare gene
  symbol (`Lanosterol 14-alpha demethylase` vs a span naming only `CYP51A1`), a false positive on a
  one-component wrapper with exact identity. Changes no disposition today.
* **F-111 / F-112 / F-114** — tooling and staleness. F-114's one-line note is folded into Item 2.
