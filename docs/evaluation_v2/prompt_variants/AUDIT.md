# Audit of the Stage-1 extraction prompt (`src/t2pw/llm/prompts/pwml_system.txt`)

Scope: the extraction prompt in both states, the code that consumes its output, and the
audit prompt. Nothing under `src/`, `tests/`, `docs/` or `scripts/` was modified; no
pipeline, model or network call was made.

## Citation convention

The prompt has uncommitted edits, so line numbers are ambiguous unless the state is named.

| Tag | State | sha256 (first 16) | Bytes | Lines |
|---|---|---|---|---|
| `(C)` | committed, `HEAD` | `1b3666a993f601f1` | 34,733 | 540 |
| `(W)` | current working tree | `0a1adddfe72655cf` | 37,044 | 562 |

Today's edit inserts 5 lines after `(C):114` and 17 lines after `(C):209`. The mapping is
therefore: `(C)` lines 1-114 unchanged; 115-209 shift `+5`; 210+ shift `+22`.
Code citations are `path:line` at the working-tree revision.

---

## 1. What the current prompt gets right

These are load-bearing and every variant preserves them.

1. **Evidence discipline is already strong and already enforced twice.** A verbatim quote
   of <=25 words is demanded per relationship/object `(W):76`, again per reaction
   `(W):82`, and the measured v02 run produced an evidence quote on all 28 rows with
   nothing invented. `research_report.py:789` prints `evidence: NONE RECORDED — treat as
   unsupported` for a row without one, so the discipline is visible to a reviewer rather
   than merely requested.
2. **The actor-type rules match the cleaner exactly.** `(W):95` and the COFACTOR RULE
   `(W):97-104` restrict `modifiers[]` to protein / protein_complex; `_clean_enzymes`
   enforces the same two values (`pipeline.py:1870`) and drops compound/cofactor/ion/
   small_molecule/metabolite rows (`pipeline.py:1871, 1891-1893`). A prompt rule and a
   cleaning rule agreeing is rare here and worth not disturbing.
3. **The legacy-key tolerance is correctly documented.** `(W):104` tells the model that
   `{"protein_complex": "X"}` in `enzymes[]` is acceptable, which is precisely the shape
   `_clean_enzymes` whitelists (`pipeline.py:1895-1896`). Without this the most common
   model shape, a bare `{"name": ...}`, is silently deleted (`pipeline.py:1900-1902`).
4. **Empty-row suppression matches the cleaner.** `(W):78-79, 81` forbid placeholder and
   participant-less rows; `_clean_processes` drops exactly those
   (`pipeline.py:2213-2228`), and `clean_stage_one_with_report` would otherwise report them
   as a cleaning loss (`pipeline.py:2536-2545`).
5. **Composite-name splitting is anticipated on both sides.** `(W):83-85` forbids `A + B`
   names; `_split_composite_token` splits on `+` and ` and ` anyway
   (`pipeline.py:1668-1673`), so prompt and code fail the same way rather than differently.
6. **Non-fabrication is stated where it counts.** `(W):77` (omit what is absent),
   `(W):109` (unknown subunits get `confidence < 1.0`, not invented names), the
   checklist's steps 3-5 `(W):425-431`, and `(W):557` (partial evidence stays partial).
   The working tree's resolution rule repeats it in its own terms at `(W):222-225`
   ("Finer than stated is a fabrication; coarser than stated is a loss").
7. **Today's resolution edit works.** It is the reason the v02 run produced DNA-site
   entities at all: `tmp/draft_graph.json` carries `nucleic_acid` rows for `A site`,
   `B site`, `CAP site`, `Pc promoter`, `PBAD promoter`, `araC gene`, `araBAD operon`,
   `ara regulatory region`, against zero under the committed prompt. Site-level
   extraction needs **no** schema change: see SCHEMA_CHECK.md §4.
8. **The scope/organism single-selection rules** `(W):135-162` are the only thing stopping
   a multi-pathway paper producing a merged graph, and `scope_membership` is now genuinely
   wired (`_carry_scope_membership`, `pipeline.py:2083-2159`, feeding
   `filter_out_of_scope_reactions`, `pipeline.py:288-368`).

---

## 2. Defects

Ordered by consequence. "Measured" means observed in the committed run artifacts
(`tmp/draft_graph.json`, `tmp/reaction_summary.txt`, `tmp/qa_report.json`,
`tmp/extraction_diagnostics/run-c5e6c739092d/`), which are the v02 araC run.

### D1 — Five fields the prompt mandates per process row are deleted before any reader sees them

The schema demands `class`, `confidence`, `provenance`, `source_refs` on every reaction
`(W):312-316`, transport `(W):354-357`, reaction-coupled transport `(W):336-339` and
interaction `(W):368-371`, plus `spontaneous: false` on every reaction `(W):93`.

`_clean_processes` rebuilds every process row from a key whitelist and **none of those five
keys is in it**:

* reactions keep `inputs, outputs, locked_reaction_id, name, enzymes, biological_state,
  evidence, inference, scope_membership` + RAG/lineage carriers
  (`pipeline.py:2229-2265`);
* interactions keep `entity_1, entity_2, name, relationship, biological_state, evidence,
  inference` + carriers (`pipeline.py:2383-2404`);
* transports `pipeline.py:2275-2312`; reaction-coupled transports `pipeline.py:2333-2359`.

Two consequences, both real:

1. **A review flag that cannot fire.** `stage_one_boundary._paper_entry` decides whether a
   row is `_PAPER_STATED` or `_PAPER_STATED_UNVERIFIED` by testing
   `row.get("provenance")` against `("inferred", "enriched")`
   (`stage_one_boundary.py:311`, values at `:212`). It runs on the **cleaned** payload
   (`clean_stage_one` → `settle_stage_one`). For process rows that key is already gone, so
   a reaction the model itself marked `provenance: "inferred"` is attributed as
   paper-stated with `review_required=False`. The docstring at
   `stage_one_boundary.py:299-303` asserts the opposite — "the process cleaners carry
   `provenance`/`inference` through" — which is true of `inference`
   (`pipeline.py:2251-2253, 2401-2403`) and false of `provenance`. This is the sprint's
   merge gate 7 ("preserve incomplete-but-correct pathways as `review_required`") being
   unenforceable on process rows.
2. **`relationship` is the single carrier of interaction type at export.**
   `pwml/ir.py:2291` reads `interaction_type or relationship or class or "interaction"`,
   and `pwml/writer.py:1419` writes it as the `interaction-type` attribute. Since `class`
   never survives cleaning, the fallback chain collapses to `relationship` alone. That is
   why the relationship vocabulary is worth a variant of its own (v09): it is not a
   cosmetic field, it is the only thing that types the exported interaction.

`confidence` and `provenance` **are** kept on actor sub-rows (`pipeline.py:1927-1932`) and
on entity rows, which are copied key-for-key (`pipeline.py:1749`). So the loss is specific
to the four process buckets.

### D2 — `compartment_canonical` is mandatory in the prompt and deleted by the cleaner; `compartments[]` is read by nothing

`(W):132` ("Every biological_state that has a subcellular_location MUST also set
compartment_canonical"), COMPARTMENT RULE Step 2 `(W):175-177`, schema field `(W):274`.

`_clean_biological_states` whitelists exactly `name, species, cell_type, tissue,
subcellular_location, evidence` (`pipeline.py:1772`). `compartment_canonical` is dropped.
The readers that want it are downstream of that: `payload_models.py:272`,
`process_normalizer.py:3422-3428` (builds a `biological_state → compartment_canonical`
lookup), `pwml/ir.py:312, 1678, 2789`, `pwml/writer.py:603-604`. `gap_resolver.py:728`
writes it back later, which is the repair path for a value Stage 1 was told to supply.

The top-level `compartments[]` array `(W):121-127`, mandated by COMPARTMENT RULE Step 1
`(W):167-173`, whose `canonical_name` vocabulary is fixed at `(W):171-172`,, is read by **nothing** in the core
pipeline: a grep for readers of `payload["compartments"]` finds only `rag/ingest.py:484`
and `sbml/examples.py:157, 312, 430`, which read a differently-shaped SBML document.
`PayloadModel.compartments` exists (`payload_models.py:466`) so it validates, and is then
ignored.

This is the defect with the largest effect on the **four metabolic papers**: compartment
assignment is most of the value of a chlorophyll- or plasmalogen-biosynthesis extraction,
the prompt spends ~20 lines `(W):166-195` demanding it, and Stage 1 delivers the half the
cleaner keeps. No prompt variant can fix it (the fix is in `_clean_biological_states`), so
no variant touches the COMPARTMENT RULE — but it must not be read as a prompt failure when
the A/B shows weak compartment data.

### D3 — The prompt's own worked example emits two of the exact values the prompt forbids

`(C):453` and `(C):458` — unchanged at `(W):475` and `(W):480` — are:

```
        "entity_1": "[SmallMolecule]", "relationship": "inhibits", "entity_2": "[EnzymeY]",
        "entity_1": "[OtherMolecule]", "relationship": "activates", "entity_2": "[EnzymeY]",
```

The enumeration at `(C):114` / `(W):114` allows only `activation | inhibition | binding |
interaction`. Today's edit added a rule `(W):115-119` that names `"activates"` and
`"represses"` as mistakes **by example** — and left Example 2, five hundred lines later,
demonstrating `"inhibits"` and `"activates"` in the field. Note also that both rows pair the
forbidden `relationship` with a *correct* `class` on the next line, so the example actively
teaches that the two fields take different vocabularies.

This is the most plausible mechanism for the measured vocabulary drift under the committed
prompt ("activates", "represses", "binds"): a concrete filled-in example outweighs a prose
prohibition for most models. **Today's edit did not fix it**, which is why v09 tests
construction-plus-example-repair against v02's prohibition-only approach.

### D4 — Alias rows: out of contract at Stage 1, legitimate at Stage 2, caught by neither cleanup

4 of 28 measured rows are identity declarations (`tmp/reaction_summary.txt` INTERACTION
13-16): `cAMP`/`cyclic AMP`, `CAP`/`catabolite gene-activator protein`, `PBAD`/`PBAD
promoter`, `Pc`/`Pc promoter`, all with `relationship: SAME_AS`.

* `SAME_AS` is **not** among the four values `(W):114` and is not mentioned anywhere in the
  extraction prompt, so these rows were already outside the stated contract. The rule added
  today `(W):115-119` prohibits verb forms and `promoter_of`; it says nothing about
  identity rows, so it does not reach them.
* `SAME_AS` is nonetheless a real downstream mechanism: `process_normalizer.py:1949-1951`
  recognises `sameas`/`same_as`, `canonicalize_same_as_aliases` (`:2332`) consumes it, and
  `pwml_infer_system.txt:144-153` explicitly instructs **Stage 2** to emit it for alias
  bridging. So it must be forbidden at Stage 1 without being broken at Stage 2 — v04 does
  exactly that, and nothing in the variant set touches the inference prompt.
* The curator's existing cleanup does not catch these. `pathway_curator_system.txt:19-21`
  removes a `SAME_AS` row "whose entity_1 and entity_2 are identical strings"; all four
  measured rows have *different* strings, which is the whole point of an alias.
* The paper's own wording that produced them is an abbreviations list:
  `"Abbreviations: CAP, catabolite gene-activator protein; cAMP, cyclic AMP"`. That is
  front-matter, not a result.

The brief's suggested fix — route them to an entity's `synonyms` — is now forbidden, see
D11 and SCHEMA_CHECK.md §5.

### D5 — "no coarser" was never "not *also* coarser"

The resolution rule `(W):210-225` instructs "record every process at the resolution the
text states, and no coarser", and at `(W):221-225` guards against manufacturing a finer
row. Nothing forbids emitting the same claim a second and third time at coarser grain.

Measured, from `tmp/reaction_summary.txt`:

| Claim | Rows |
|---|---|
| CAP activates the araC transcription unit | #9 `CAP → Pc promoter`, #19 `CAP → araC gene` |
| araC represses its own promoter | #11 `araC → Pc promoter`, #22 `araC → araC gene` |
| araC occupies the regulatory DNA | #1 `→ A site`, #25 `→ Pc promoter`, #35 `→ ara regulatory region` |
| CAP activates araBAD | #8 `→ PBAD promoter`, #20 `→ araBAD operon` |

Three rows (#34, #35, #36) target `ara regulatory region`, the coarsest available target.
That entity is the paper's own name for the whole 147-bp segment containing every site it
maps — so a row pointing at it discards the result the paper exists to report.

Count inflation is the visible symptom; the real cost is that a reviewer reading
`research_report.py:764-792` cannot tell which row is the finding and which are echoes,
because all of them carry a true verbatim quote.

### D6 — One protein, two registered names, by construction

`catabolite gene-activator protein` is a `protein` node and `CAP` is a separate node in
`tmp/draft_graph.json`; `tmp/reaction_summary.txt` #3 uses the long form as source while
#14 relates the two. The prompt has no rule selecting one surface form:

* `(W):75` dedupes "identical strings within each entity list" — two spellings are not
  identical strings;
* `_normalize_name` folds case and strips non-alphanumerics (`pipeline.py:1651-1653`), so
  `CAP` and `catabolite gene-activator protein` normalise to different keys and are two
  entities at every stage;
* `(W):91` resolves *named gene vs generic enzyme class*, a different problem from
  *abbreviation vs expansion*;
* `(W):72-73` ("No normalization... do not expand") arguably *encourages* keeping both.

The cost is downstream matching: `filter_unresolvable_reactions` resolves participants
through `_normalize_name` (`pipeline.py:909-910, 861-865`), and `process_normalizer`
builds its name index the same way (`:626-633`) — so a row written with the form that is
not the registry's loses its participant. v05 isolates this.

### D7 — A code defect that gets *worse* as the prompt gets better (prompt cannot fix it)

`build_draft_graph` builds its kind-resolution sets from **three** buckets only —
`proteins`, `protein_complexes`, `compounds` (`draft_graph.py:221-229`). `nucleic_acids`
and `element_collections` are absent. `_resolve_actor_kind` (`draft_graph.py:115-131`)
therefore cannot recognise a DNA element and falls through to its default:

```python
    # heuristic: complex-style names contain "/" or ":"
    if "/" in n or ":" in n:
        return "protein_complex"
    return "compound"
```

So every DNA site referenced by an interaction is typed `compound`, and
`_add_entity_node` mints a **second** node for it under that kind
(`draft_graph.py:241-261`, keyed `f"{kind}:{name}"`). Measured in `tmp/draft_graph.json`:
8 `nucleic_acid` nodes, each with degree 0, plus 12 `compound` nodes carrying labels such
as `A site`, `CAP site`, `Pc promoter`, `araC gene`, `ara regulatory region`. `tmp/qa_report.json`
reports precisely this as `orphan_nodes: 8` and `duplicate_species: 8` with reason
`normalized label collision`.

This qualifies the brief's premise. Site-level extraction fits the **payload** schema
exactly (SCHEMA_CHECK.md §4), but `draft_graph` mis-types it, so the variants that produce
more site-level rows (v03, v08) will also produce more orphan nucleic-acid nodes and more
collision flags. Those flags are an artifact of `draft_graph.py:221-229`, not of the
prompt, and must not be scored against a variant. The fix is three lines loading
`nucleic_acids` and `element_collections` into the same sets; it is outside this task's
boundary.

### D8 — Interactions pass through every gate the reactions have

`filter_out_of_scope_reactions` reads `processes.reactions` only
(`pipeline.py:343-361`); `filter_unresolvable_reactions` likewise. `scope_membership` is
demanded on reactions only (`(W):313`), and `_carry_scope_membership` is deliberately
reaction-only (`pipeline.py:2256-2259`). The post-extraction contract asks an interaction
for nothing but two endpoints (`stage_contracts.py:732-749`), and `_clean_processes` drops
an interaction only when an endpoint is blank (`pipeline.py:2378-2382`).

So an off-pathway, duplicated or coarse interaction has **no** downstream filter. That is
why all 28 rows shipped, and why interaction precision has to be achieved at extraction
time — the thing v03, v04 and v05 each test. Adding a scope field to interactions would be
a schema change and is out of bounds.

### D9 — Not a defect, but the constraint v06 has to respect

`(W):81` and `(W):401` require a reaction to have inputs **and** outputs; `_clean_processes`
agrees and drops the row otherwise (`pipeline.py:2213-2228`); but the stage contract
accepts input **or** output (`stage_contracts.py:696-700`). Prompt and cleaner are the
binding pair. A transcription reaction must therefore name both ends in the text or it
will be deleted silently — which is exactly how the measured committed-prompt run produced
"one of them chemically empty (araBAD operon → araBAD operon)": identical strings on both
sides survive `_dedupe_preserve_order` per side and pass the `inputs and outputs` test.
v06 forbids that shape explicitly.

### D10 — Today's edit already broke six prompt line-number citations, and the brief's premise about them needs correcting

The brief states that `stage_one_boundary.py` "validates provenance and other fields
AGAINST specific prompt line numbers". It does not. It matches on **values**
(`_PROVENANCE_NOT_READ = ("inferred", "enriched")`, `stage_one_boundary.py:212`) and on
**key names** (`_MARKS_NOT_READ = ("inference", "rag_provenance")`, `:217`). The line
numbers appear only in comments and docstrings, so no variant can change that module's
behaviour by moving a line. What a variant *can* do is make the comments wrong — and
today's edit already has:

| Citation site | Cites | `(C)` content | Now at `(W)` | Status |
|---|---|---|---|---|
| `stage_one_boundary.py:205` | `:116` | provenance values | `(W):121` | **broken** — `(W):116` is mid-sentence of the new relationship rule |
| `stage_one_boundary.py:215` | `:109`, `:401` | subunit note; "provenance inferred for the complex row" | `(W):109`, `(W):423` | `:109` ok, `:401` **broken** |
| `stage_one_boundary.py:299` | `:109`, `:116`, `:401` | as above | — | two of three **broken** |
| `tests/test_stage_one_boundary_lineage.py:177` | `:109/:116/:401` | as above | — | two of three **broken** |
| `tests/test_scope_membership_filter_end_to_end.py:177` | `:291` | `scope_membership` schema line | `(W):313` | **broken** |
| `docs/prd_kg_triples_middle_layer.md:69` | `:453` | "every named enzyme/catalyst in modifiers" | `(W):475` | **broken** |
| `docs/research_mode.md:436` | `:117` | `source_refs` | `(W):122` | **broken** |
| `pipeline.py:2086`, `tests/test_scope_membership_filter_end_to_end.py:7` | `:6`, `:15-16` | scope label rules | unchanged | ok |

All eight derived variants shift these further (each inserts 8-33 lines). The durable fix
is to cite by quoted text rather than by line, which is how `pipeline.py:2086` partly does
it. Recorded as a finding against today's edit, per the coordinator's request.

### D11 — `synonyms` is read by the code and never mentioned by the prompt

`process_normalizer._entity_name_norms` folds `row.get("synonyms")` into the name index
(`:626-633`, used at `:733, 761, 3780, 3900, 4337`); `apply_audit_patch` treats `/name`
and `/synonyms/<i>` as the two registry-weight leaves (`:991-1004`, rationale at
`:885-889`); the curator prompt routes a shorthand there as its lowest-risk fix
(`pathway_curator_system.txt:15-18`); `enrich_entities` populates it from external sources
(`:1389-1394, 1462-1469`). `_clean_entities` copies entity rows key-for-key
(`pipeline.py:1749`), so a `synonyms` array emitted at Stage 1 **would** survive cleaning.

This is the brief's "silent gap" and the natural destination for the D4 alias rows. Adding
it to the schema block is a new key under `entities`, which the tightened constraint
rejects. It is therefore **not** exercised by any variant; v04 drops alias rows and v05
prevents the second name from being registered at all. If the constraint is ever relaxed,
this is the first change to make.

### D12 — Minor dead weight

* Four of the nine whitelisted entity buckets — `cell_types`, `species`, `tissues`,
  `subcellular_locations` — never become graph nodes (`draft_graph.py:269-275` maps five
  buckets). They do feed `biological_states` and enrichment, so this is cost, not waste.
* An entity bucket outside the whitelist is dropped whole (`pipeline.py:1693-1703`, counted
  row-by-row at `:1710-1718`). The prompt lists exactly the nine whitelisted buckets
  `(W):249-264`, so this is currently aligned — it is a trap for any future schema edit.
* The TASK list `(W):54-68` enumerates 20 PWML list names (`bound_visualizations`,
  `vacuous_*`, `edges`, ...) that the schema block then mostly nests under
  `visualizations`, which `(W):226-228` tells the model to leave empty. ~15 lines of
  prompt describing output that is correctly always empty.

---

## 3. The audit prompt (`pathway_curator_system.txt`)

Also edited today: a new §5 (`:77-108`) plus one line in §Rules (`:118-119`).

**What is right.** §5 is well fenced for a stage that can add rows: a verbatim quote from
the payload is mandatory and a paraphrase is explicitly refused (`:88-89`); participants
must already exist in the entity list at the resolution the evidence uses (`:90-94`);
`relationship` is restricted to the same four values with `inhibition` named for
repression/blocking/competition (`:95-96`); an unreadable sign is a reason not to propose
rather than to guess (`:96-97`); duplicates **including cross-resolution restatements** are
skipped (`:100-102`); structural facts are excluded (`:103-104`); and a same-entity or
state-pointing row gets a warning rather than a `remove` patch (`:106-108`). The §Rules
amendment (`:118-119`) scopes the new permission narrowly to interactions.

**Three observations worth recording.**

1. **§5's quote rule is the only real gate on an interaction add.** The acceptance policy
   gives an `add` a threshold of 0.75 (`apply_audit_patch.py:175-176`);
   `_is_major_topology_path` does match `/processes/interactions/-` (`:150-156`) but its
   stricter threshold applies only when `enforce_major_topology_threshold=True`
   (`:2633`); `_is_core_semantics_path` deliberately excludes interactions (`:159-167`);
   and `_ACTOR_ROLE_PATH_RE` (`:1419-1423`) covers only reaction/transport actor
   containers, so the unevidenced-actor-role guard never inspects an interaction add.
   The wording at `:88-89` is doing the work alone. It is good wording; it is also
   load-bearing in a way the file does not say.
2. **§5's "add the nucleic_acids entity first, in the same patch set" (`:92-94`) depends on
   a specific applier behaviour** — `_pending_entity_name_norms` / the referential-integrity
   guard (`apply_audit_patch.py:1012, 1028+`) — which evaluates names introduced by earlier
   ops in the same batch. The instruction is correct only because that exists.
3. **§5 agrees with v03 already.** Its dedupe clause (`:100-102`) says to skip "a row that
   merely restates one at a different resolution, unless the evidence states both the
   binding and its consequence" — the same rule v03 adds at extraction time, with the same
   exception. That is a consistency argument for v03 and against leaving the duplication
   to the curator: §5 can only skip rows it is *proposing*, not deduplicate rows Stage 1
   already shipped.

**Does any extraction variant require a matching curator change? No.** Specifically:

| Variant | Curator interaction | Change needed |
|---|---|---|
| v03 resolution precedence | already stated at `:100-102` | none |
| v04 alias ban | `:19-21` removes only identical-string `SAME_AS`; `:103-104` excludes structural facts. v04 prevents the rows upstream, so the gap in `:19-21` (D4) stops mattering for new runs | none (and the `:19-21` gap is worth fixing on its own merits, independently of this A/B) |
| v05 one canonical name | `:14-18` prefers adding a shorthand to `synonyms` over editing a reference — complementary, and still correct if Stage 1 emits one name | none |
| v06 transcription as reaction | `:118` forbids the curator from adding reactions, so a transcription reaction Stage 1 misses can never be recovered. Consistent, but it means v06's recall is entirely a Stage-1 property | none |
| v07 transcription excluded | consistent with `:118` and with §5 routing regulation to interactions | none |
| v08 regulatory example | §5 is the audit-side analogue of the same pattern | none |
| v09 vocabulary by construction | `:95` already names the four values; v09 is strictly more specific and does not contradict it | none |
| v10 self-check | no overlap | none |

One asymmetry worth noting rather than fixing: §5 instructs the curator to emit `class`,
`confidence`, `provenance` and `source_refs` on the interaction it adds (`:84-85`). Per
D1 those four keys are stripped from interaction rows by `_clean_processes` — but the
curator runs *after* Stage-1 cleaning, so for a curator-added row they survive. The two
stages therefore ship interaction rows with different key sets. That is pre-existing and
is not something a prompt variant should try to equalise.
