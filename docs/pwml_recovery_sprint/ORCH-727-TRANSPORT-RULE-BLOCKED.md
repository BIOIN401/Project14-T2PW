# ORCH-727 — the authorized wave is BLOCKED before implementation. No production change was made.

**Read-only, 2026-09-07.** Branch `sprint/pwml-recovery` @ `a6516ad8`. No `src/` edit, no
worktree created, no implementer dispatched, no merge. `main` untouched.

---

# VERDICT

> ## `DO NOT IMPLEMENT.` The authorized transport-integrity rule cannot produce the required proof set — and neither can any structural variant of it.

The wave was authorized on the premise that a general transport-integrity invariant would
block `PMC12452463` while letting `PMC7232280` and `PMC8510960` through. **Measured against
the archived payloads, no such invariant exists.** The rule as specified refuses *nothing*;
every variant that refuses the bad transport also refuses the two the wave exists to recover.

Implementing F-147 without a working companion gate would merge exactly the unsafe outcome
`F-147` was registered to prevent. So nothing was implemented and nothing was dispatched.

---

# 1. The rule as authorized refuses nothing

> *"a transport may not survive … when it has: no transporter; **and** no defensible
> source/destination locations."*

`Enterobactin secretion` — the transport this rule exists to catch — **has both locations**:

```json
{ "name": "Enterobactin secretion",
  "cargo": "enterobactin",
  "from_biological_state": "Escherichia coli cytosolic state",
  "to_biological_state":   "Escherichia coli extracellular state",
  "evidence": "Enterobactin is secreted by bacteria into the extracellular environment" }
```

The conjunction is therefore false and the transport survives. **Required proof 1 fails on
the rule as written.**

# 2. Every structural variant fails too

Applied to all nine archived transports across the five legs:

| variant | `PMC12452463` blocked? | `PMC7232280` survives? | `PMC8510960` survives? | control survives? | usable? |
|---|---|---|---|---|---|
| **A** — no transporter **AND** no locations *(as authorized)* | **NO** | yes | yes | yes | **no — refuses nothing** |
| **B** — no transporter (alone) | yes | **NO** | **NO** | yes | **no — kills both recoveries** |
| **C** — no transporter **OR** incomplete locations | yes | **NO** | **NO** | yes | **no — same** |
| **D** — no transporter **AND** not both locations | **NO** | yes | **NO** (`__auto_state__`) | yes | **no — worst of both** |

The reason is visible in one table. These are the four transports the rule must separate:

| leg | transport | transporter | from | to |
|---|---|---|---|---|
| **must block** | `Enterobactin secretion` | none | `E. coli cytosolic state` | `E. coli extracellular state` |
| must export | `cPMP export` (`PMC7232280`) | none | `mitochondrial state` | `cytoplasmic state` |
| must export | `Strictosidine export` (`PMC8510960`) | `[]` | `vacuole` | `cytosol` |
| must export | `Catharanthine secretion` (`PMC8510960`) | `[]` | `__auto_state__` | `leaf surface` |

**`Enterobactin secretion` and `cPMP export` are structurally identical** — no transporter,
both locations named, cargo present, paper-stated provenance. No predicate over transporters
or locations can separate them.

# 3. The provenance reading fails as well

The next hypothesis was the `F-179` shape extended to transports: refuse a transport whose
evidence span is not locatable in the paper. Measured by folding each evidence string against
the archived `01_source_text.txt`:

| leg | transport | evidence verbatim in source? |
|---|---|---|
| **must block** | `Enterobactin secretion` | **YES — 9/9 words, 100 %** |
| must export | `cPMP export` | yes — 9/9 |
| must export | `Strictosidine export` | yes — 11/11 |
| must export | `Catharanthine secretion` | yes — 12/12 |
| control | `D-Alanine transport` | yes — 10/10 |

*"Enterobactin is secreted by bacteria into the extracellular environment"* **is in the
paper, verbatim.** The gold's objection — *"Export of enterobactin from the cytoplasm is
never described at all"* — is a **curatorial judgment** that a general statement about
bacteria does not license emitting a cytoplasmic efflux *process* for this pathway. It is not
a claim that the sentence is absent, and no provenance rule can reconstruct it.

# 4. Two further invariants tested, both fail

**Cargo graph-connectivity** — "a transport's cargo must be produced or consumed by a
retained reaction":

| leg | cargo | produced | consumed |
|---|---|---|---|
| **must block** | `enterobactin` | **yes** | **yes** — fully connected |
| must export | `catharanthine` | no | yes |
| control | `d-alanine`, `d-serine` | **no** | yes |

**Inverted.** The transport that must be blocked is the best-connected one; the control's two
transports carry cargo no reaction produces.

**Complex-component structure** — aimed at the other `PMC12452463` objection, the
`enterobactin synthase complex` enzyme:

| complex | components |
|---|---|
| **`enterobactin synthase complex`** *(gold: forbidden_identifier)* | 1 — `['Unknown']` |
| `NIT-7A`, `NIT-7B`, `NIT-9G`, `NIT-9E` *(must export)* | 1 each — `['Unknown']` |
| `G10H`, `STR`, `SGD` *(must export)* | 1 each — `['Unknown']` |

**Byte-for-byte the same shape.** The forbidden complex is structurally indistinguishable
from the seven Unknown-backed wrappers the two recoverable legs depend on. A rule refusing
one refuses all eight.

# 5. Why this is a real result and not a failure to try

The four objections `F-147` names for `PMC12452463` are **curatorial content judgments made
by a human against a specific paper**, recorded in that paper's gold file:

- *"A complex name explicitly denoting three proteins"* — a judgment about `enterobactin
  synthase complex` as a name, not about its structure, which is identical to `G10H`'s.
- *"A small RNA, not a protein and never an enzyme"* — an entity-class fact about `RyhB`
  that no field in the payload carries.
- *"Export … is never described at all"* — a judgment about what a verbatim sentence licenses.
- `unknown_backed_proteins_acceptable: false` — a per-paper gold flag, and `D-070 § O-1a`
  rules the bare sentinel legitimate in general.

**None of these is a structural or provenance invariant.** They are exactly the class of
finding a benchmark's gold set exists to encode and a deterministic gate cannot. Building a
general rule that reproduced them would require either keying it to this paper — which the
authorization explicitly forbids and which merge rule 6 forbids — or refusing the Unknown
sentinel wholesale, which contradicts `D-070 § O-1a` and would block both recoveries.

`F-147`'s own text anticipated this: *"the fix must land together with the gates that would
then block these legs on their real problems."* **On this evidence those gates do not exist
in deterministic form.**

# 6. What was NOT done, deliberately

- No `src/` file was edited. No worktree. No branch. No implementer dispatched. No review
  requested. No merge. No push of production content. `main` untouched.
- The F-147 half was **not** implemented alone. It is correct on its own terms, and ORCH-726
  already proved it recovers 2 substantive PWMLs — but merging it without a working companion
  gate exports `PMC12452463` with all four gold objections intact, which is the precise
  outcome the authorization was structured to avoid.

# 7. The options, for the product owner

1. **Stop here.** Do not merge F-147. The pilot's strict-PWML count stays 1 of 6 and the
   manuscript's limitation section records F-147 honestly, with ORCH-726's measurement that
   it costs 2 substantive papers and buys refusal of 1 contaminated one. **Recommended** —
   it is the only option that needs no further engineering and no new risk.
2. **Merge F-147 alone, and accept that `PMC12452463` becomes exportable.** Defensible only
   if the product owner rules that paper's four objections are gold-specific curation rather
   than product defects — which would contradict `PRODUCT_CONTRACT § 13`'s *"Never strict
   success"* ruling for it. This is a product decision and I am not making it.
3. **Merge F-147 plus a `review_required`-only downgrade for any pathway containing an
   Unknown-backed enzyme.** This does not refuse `PMC12452463`; it marks it. It would also
   mark `PMC7232280` and `PMC8510960`, which already export as `review_required` anyway. It
   is honest and general, but it is a **different, larger** change than authorized and it
   does not satisfy required proof 1 as written.

---

*No production code, gold data, run artifact or `main` was modified. Nothing was merged or
pushed to production. This document and its measurement script are the only outputs.*
