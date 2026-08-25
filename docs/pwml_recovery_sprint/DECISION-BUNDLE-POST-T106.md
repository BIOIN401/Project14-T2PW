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

---

# ADDENDUM — items surfaced by C-081, C-082 and C-085, added 2026-08-25

The three cards merged after this bundle was first written each surfaced a question the orchestrator
may not settle. They are collected here so all of them can be taken in one sitting.

---

## Item 4 — `supported_reactions_complete`: should any gold case set it?

**Raised by F-121. This is the other half of C-085, and C-085 was forbidden from touching it.**

### Current behaviour, after C-085

`supported_reactions_complete` is **absent from all ten** pinned gold cases, so it defaults `False`
everywhere. Before C-085 that silently produced `PASS 0` on an **absolute** priority. After C-085 it
correctly produces **`NOT EVALUATED`** — on 11 of 20 T-106 legs covering 6 papers.

**C-085 made the report honest. It did not make the measurement exist.** Priority 2 still cannot
detect a fabricated reaction on those six papers; it now says so out loud instead of claiming a pass.

Note one nuance C-085 preserved deliberately: a subset case in which **every** retained row matched a
quote-verified signature is a genuinely measured zero, and five T-106 legs still PASS on that basis.
The gap is only on legs with unattributed rows.

### The options

| | Option | What it means | Cost | Risk |
|---|---|---|---|---|
| **A** | **Leave it.** Priority 2 reports `NOT EVALUATED` wherever the signature set is a subset. | Honest, and already shipped. | none | The absolute gate stays unable to detect fabrication on 6 of 10 papers. We would *know* we cannot see, which is strictly better than the status quo ante — but we still cannot see. |
| **B** | **Author exhaustive signature sets** for some or all cases and set the flag `true`. | Priority 2 becomes a real measurement on those papers. | **High and manual** — `goldset.py:384`'s own comment says *"Only set this `True` after reading the whole paper and writing a signature for every reaction in it"*, and warns it is **incompatible with multi-paper RAG synthesis unless the run is seed-only.** | Setting it `true` without genuinely exhaustive signatures would convert every unattributed row into a reported fabrication — `semantic.py:700-704` records that this would have reported **227** fabricated reactions in a run that produced far fewer. **That is the worst outcome available and it is one keystroke away.** |
| **C** | **Set it only on the papers where an exhaustive set is tractable**, and leave the rest `NOT EVALUATED`. | Partial real coverage, honest elsewhere. | Moderate, per paper. | Needs a per-paper judgement about whether the signature set really is exhaustive. |

### Recommendation: **C, starting with the two negative controls, which already have ceilings**

`PMC13231680` and `PMC12180156` already carry `max_retained_reactions` and are the only two papers
where priority 2 has ever measured anything. They are also the shortest. Extending real coverage
outward from there is the low-risk path.

**Do not take B wholesale.** The 227-fabrication figure is the measured consequence of getting this
wrong, and the flag is a single boolean.

**This does not block T-107.** Priority 2 is now honest either way. It is the difference between
"we know we cannot see" and "we can see".

---

## Item 5 — F-123: a prefreeze declination does not demote the release status

**Raised by the C-082 review. Same family as Item 1 (F-107) and should be ruled in the same sitting.**

### Current behaviour, after C-082

An ambiguous species rename now declines instead of crashing the leg — `report["ok"] = False` plus
`report["review_required"]["species"] = "species_rename_declined:AMBIGUOUS_RENAME_TARGET"`.

**But `report["ok"] = False` does not demote anything.** Both consuming seams say so in terms:

> `writer.py:2731` — *"`prefreeze_report["ok"] is False` deliberately does NOT abort. D-029 (LOCKED)
> … Acting on it is the downstream seam's job."*
> `streamlit_app.py:4930` — *"D-029, as split by **D-040 §8**: this seam PERSISTS and SURFACES the
> verdict. It does not act on it — no branch here changes whether a PWML is produced."*

`classify_release_status` takes no `prefreeze_review_required` parameter at all.

### The observable change, stated plainly

**At base, a strict leg hitting this shape crashed, so it could never be `release_ready`. At the tip
it proceeds and *can* reach `release_ready` while carrying two organism rows the ladder wanted to
merge.**

That is **required** by merge rule 7 and **permitted** by D-035 clause 8 (graph intact, no invalid
PWML — both verified). But it means clause 8's *"must not become a successful export"* is enforced
only by the **other** gates, never by this channel.

### Why no card fixed it

Closing it requires `release_status.py` or `streamlit_app.py` — outside C-082's boundary, and one of
them is the forbidden product-owner file — and it would **reverse or extend D-040 §8, which is
LOCKED**. The reviewer's judgement, which I accept: the card did not stop short; the residual is
genuinely not constructible within that boundary.

### The options

| | Option | Effect |
|---|---|---|
| **A** | **Thread the prefreeze verdict into `classify_release_status`** so a declination demotes to `review_required`. | Closes it properly. Requires extending D-040 §8 and a card in `release_status.py`. |
| **B** | **Rule that the other gates are sufficient** and record that a declination is deliberately release-status-neutral. | Cheapest; makes the current behaviour intentional rather than incidental. |
| **C** | **Fold into Item 1's ruling** — F-107 is the same shape (a correct refusal that cannot reach the status it deserves), and one ruling could settle the general principle for both. |

### Recommendation: **C**, and take it with Item 1

Both are the same question wearing different clothes: *when a guard correctly refuses something, what
release status should the run carry?* Ruling them together avoids two half-answers. If they must be
split, **B** is a defensible interim for F-123 alone, because the biology is intact — the two
organism rows stay distinct, which is what the guard exists to protect.

---

## Item 6 — a scope note to ratify rather than inherit (C-081)

Not a decision that blocks anything, but it is policy-adjacent and it arrived inside a card rather
than a ruling.

C-081 refuses database identity to a row that declares `class:"cofactor"` and that **no reaction or
transport uses**. Its real class is *"cofactors the extractor never wired into a reaction"*, which is
broader than *"hallucinations"*.

**Cofactor binding — an `interaction` — is the canonical way papers state cofactor relationships**,
and C-081 rules that an interaction endpoint confers **no participant role**. On the committed corpus
every such row happens to be gold-forbidden, so measured collateral is **0 over 18 refusals across 89
artifacts**. But *"an interaction endpoint does not license a database identity"* is a judgement that
deserves an explicit ratification rather than silent inheritance from a card.

**Recommendation:** ratify as written. The measurement is strong (zero collateral, and a
schema-complete participant reader rescues none of the 18), the direction is conservative, and
`ATP` — which declares the same role ten times and is used every time — is untouched.

---

## Item 7 — two participant readers narrower than the schema they read (F-119, F-125)

**Not a decision — an owner is needed.** Recorded here because the two are the same defect shape in
two different absolute gates, and fixing them together is cheaper than twice.

| finding | reader | omits | measured exposure |
|---|---|---|---|
| **F-119** | `identity_admission._PARTICIPANT_NAME_KEYS` = `("entity","name","ref","id")` | the six `payload_models.py` keys `canonical.py:330` already uses | **0** today, but `{"protein": …}` is the dominant actor shape corpus-wide (**1,820** vs **615** for `entity`) |
| **F-125** | `semantic._orphaned_references` | `cargo`, `transporters`, `elements_with_states` — every slot a transport row has | **3 live invisible orphans** across 89 artifacts, including a leaked JSON pointer `/entities/proteins/0` in a name slot. **0 on T-106.** |

Both fail in directions that matter: F-119 would **strip a correct identifier**; F-125 **cannot see a
real referential-integrity violation** in a gate declared absolute. `canonical.py:330` already holds
the correct key list — the fix is reconciliation, not invention.
