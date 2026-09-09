# `RAG v2` — literature-expanded pathway reconstruction. HANDOFF REQUIREMENTS ONLY.

**2026-09-09, recorded under `D-099` § 20 while `C-121`'s live validation runs.** This file is a
**specification of what the next phase must preserve**, written now so the requirements are not
reconstructed later from memory. **Nothing here is authorized, chartered or implemented.** `RAG v2`
does not begin until reliability closes under the § 19 rule, and reliability cannot close until the
product owner has imported real PWML files by hand.

**Do not treat this file as a charter.** It has no ownership boundary, no gates and no branch. A
`RAG v2` card must be written against a live product-owner decision, not against this.

---

# 1. The three modes, and the only one that exists today

| mode | what it does | status |
|---|---|---|
| **A — Conservative** | paper-grounded reconstruction: only what the uploaded paper supports | **this is current production.** Everything the sprint has built and frozen |
| **B — Literature-expanded** | the uploaded paper is a **seed**; external literature may complete missing reactions **when supported by reaction-level evidence** | not built |
| **C — Pathway mining** | iterative frontier expansion — seed, find gaps, retrieve adjacent literature, extract evidence-supported reactions, add with provenance, re-expand, stop on a boundary | not built |

**Mode A must remain reachable and must remain the default.** Whatever B and C become, a user who
uploads one paper and wants only what that paper says must still get exactly that. A mode that
cannot be turned off is not a mode.

# 2. Per-reaction provenance is the load-bearing requirement

Every reaction in a mode-B or mode-C pathway must carry, and never lose:

* **target-paper vs external-literature origin** — a boolean a curator can filter on, not a free-text note;
* **the source paper** it came from, identified well enough to open;
* **the evidence quote or span** that supports it, in the source's own words;
* **confidence / support status**, on the same scale `F-179` already uses.

**A reaction whose provenance cannot be rendered must not be exportable.** The sprint already has
the carrier for this — `MASTER_PLAN.md` § 2 records the provenance carrier and row-level lineage as
**already implemented and tested**. `RAG v2` extends it; it does not rebuild it. Read that section
before writing a line.

# 3. `F-179`-style anti-invention guarantees get HARDER, not softer

`F-179` exists because the pipeline once fabricated `glycine → heme`. Mode B and mode C **increase
the fabrication surface by design**: they add reactions the uploaded paper never mentioned.

Therefore:

* The refusal `glycine → heme` must remain refused, in every mode. It is the sprint's standing
  negative control and it does not get a literature-expanded exemption.
* **"An external paper says so" is not reaction-level evidence.** A retrieved abstract asserting a
  pathway exists does not support a specific reaction with specific participants and direction.
  The bar that admits a reaction from external literature must be at least the bar that admits one
  from the target paper — and probably higher, because the operator has not read it.
* **Every relaxation needs a negative control that fails.** A mode-B admission rule with no test
  that refuses something is not a rule.

# 4. Mode C needs stop conditions written BEFORE it is built

Frontier expansion without a boundary is a crawler. The stopping rule must be decided as a product
decision, not discovered in a runaway run, and must cover at minimum:

* **depth** — how many expansion rounds from the seed;
* **relevance** — how far a retrieved reaction may drift from the requested pathway before it is
  refused, and who decides;
* **confidence floor** — the support level below which a reaction is not added at all;
* **cost and wall-clock ceiling** — this phase issues LLM calls per frontier node and the sprint has
  already measured legs at ~33 minutes each. An unbounded mine is an unbounded bill;
* **convergence** — what "no new frontier" means, and what happens when it is never reached.

The existing `RAG` gap detector, admission gate and stop policy are **already implemented**
(`MASTER_PLAN.md` § 2). Mode C's stop conditions should be expressed in terms of those, not as a
parallel mechanism.

# 5. Denominators must not be merged, and this is a repeated sprint failure

A mode-B or mode-C pathway is **not comparable** to a mode-A pathway. Its reaction count includes
reactions the paper never contained. Any statement of the form *"reconstruction improved from N to
M reactions"* that crosses modes is meaningless.

The sprint has already been burned here repeatedly: `ORCH-732`'s 3/6, `ORCH-734`'s 7/12, the
`C-120` validation and the `F-192` census all carry explicit "never merge these denominators"
warnings, and `F-196` is a runner that mis-tallied its own cohort. **Mode-B and mode-C results need
their own dataset identity from the first run**, not a retrofitted one.

# 6. What reliability closure does and does not license

`D-099` § 19 permits declaring **RELIABILITY PHASE COMPLETE** when `C-121` fixes `F-192`,
representative clean PWMLs import and render, and no new repeated deterministic yield blocker
appears. It explicitly does **not** require 100 % output, and it does **not** reopen for an isolated
identity miss, a rare provider failure, one missing reaction, a minor cofactor omission, a duplicate
enzyme or a minor topology issue.

**Closure licenses starting `RAG v2`. It does not license relaxing mode A.** Every gate, threshold
and refusal frozen during this sprint stays frozen unless a `RAG v2` card unfreezes it explicitly,
with its own product-owner decision and its own re-freeze.

# 7. Known defects `RAG v2` inherits, none fixed

| id | what it is | why it matters to `RAG v2` |
|---|---|---|
| **`F-195`** | `transport-compound-visualization` can reference a `compound-location-id` the document never declares — 2 of 3 transport-carrying PWMLs, invisible to every gate | mode B and C add transports across compartments. If this is a real importability defect, expansion multiplies it |
| **`F-196`** | the batch tally reports `N NO DELIVERABLE` for legs that DID produce one | a mode-C run has more legs, so a mis-tallying runner is more expensive to disbelieve |
| **`F-197`** | a Stage-0 `scope_conflict` is recorded at stage `stage1`, so a stage-keyed classifier misattributes it | mode C's relevance boundary is a scope decision. Misattributed scope failures will corrupt its diagnostics |
| **`F-187`** | untracked single-disk artifacts | partially addressed: the eleven PWML deliverables are now committed and hash-pinned under `pathwhiz_review/IMPORT-SET/`. Run trees are still untracked |
| **`F-186`** | the pilot ran with env-driven token budgets that differ from the committed literals | any `RAG v2` measurement must record the budget it actually ran with, not the one in `src/` |
| **the bucket asymmetry** | the `C-121` guard scans four `element_locations` buckets; `ensure_autostates` assigns two. Latent — no archived leg exercises it | mode B and C introduce entity kinds the corpus has never carried. This gap becomes reachable exactly when nucleic-acid or element-collection rows start appearing |

# 8. The measurement discipline this sprint paid for — carry it forward

Four counts in `C-121`'s cycle alone were wrong on first telling, and each was caught only because
somebody re-derived a number instead of quoting it:

* the `F-192` census undercounted with a fixed-depth glob and disclosed it;
* `ORCH-732` read a refusal as an overwrite;
* `REV-121` measured the right predicate on the wrong object;
* a latent population of 4 and 10 was reported as 3 and 7 — by the reviewer, then passed on unchecked
  by the orchestrator, and corrected by the implementer.

**The rule that falls out of this: a corpus count carries its measurement conditions or it is not a
measurement.** Which object, which entry point, which mode, which subset. A bare number in a
docstring or a report is unfalsifiable by the next reader, and this phase will generate far more of
them than mode A ever did.
