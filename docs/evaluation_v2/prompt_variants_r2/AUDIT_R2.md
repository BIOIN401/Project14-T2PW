# AUDIT_R2 — what round 1 established, what it could not separate, and where I disagree

Scope: round 1's record and artifacts, the two certified gold cases, the drafted paper gold,
and the code that consumes the prompt. Nothing under `src/`, `tests/`, `docs/` or `scripts/`
was modified; no git command was run; no model, pipeline or network call was made. Every
file written by this task is under
`C:/Users/Angad/AppData/Local/Temp/claude/c--Users-Angad-Desktop-SummerBIOIN-Project14-T2PW/67c7df6d-ac23-48b4-aec9-f17acab096a8/scratchpad/prompt_variants_r2/`.

**Base for round 2.** Every variant is derived from
`docs/evaluation_v2/prompt_variants/v03_resolution_precedence.txt`
(38,443 bytes, sha256 `bc716123774c3d8b…`), per the brief. Each derivation is one anchored
insertion, or in `r2_11` one three-line replacement; `r2_16` is four insertions plus that
replacement. The `OUTPUT JSON SCHEMA` block is never an anchor and never inside a
replacement, so the frozen shape cannot drift by construction. See `SCHEMA_CHECK_R2.md`.

Citations are `path:line` at the current working tree. Prompt citations are
`v03:<line>` against the base file above.

---

## 1. What round 1 settled, and that round 2 keeps without retesting

1. **Relationship vocabulary: solved by prohibition.** The committed prompt put 13 of 13
   values outside the allowed four; every variant carrying the prohibition produced zero
   violations (`ORCH-747-RESULT.md:37-54`). The prohibition at `v03:115-119` is present,
   byte-identical, in all 16 round-2 files, and no variant spends itself on it.
   The reason it matters is mechanical, and I re-derived it: `_clean_processes` rebuilds an
   interaction row from `entity_1`, `entity_2`, `name`, `relationship`, `biological_state`,
   `evidence`, `inference` and the RAG carriers only (`pipeline.py:2374-2405`), so `class`
   never survives; the exporter reads
   `interaction_type or relationship or class` (`pwml/ir.py:2293`), maps only `activation`
   and `inhibition` (`ir.py:29`, `:2297`) and writes the result as `interaction-type`
   (`pwml/writer.py:1502`). `relationship` is the sole surviving carrier. Confirmed, kept.
2. **Alias rows and split naming are post-Stage-1 defects.** Stage 1 produced zero of each
   in the measured run and in all ten draws (`ORCH-747-RESULT.md:177-184`). No round-2
   variant instructs the extractor about either.
3. **The grid's budget finding is infrastructure, not prompt.** 16,000 output tokens
   starves this model and returns empty bodies with `finish_reason=length`
   (`ORCH-747-RESULT.md:66-97`). Any round-2 run must use the 64,000 the code defaults to
   (`scripts/prompt_ab.py:_app_extraction_max_tokens`, mirroring `streamlit_app.py`), or it
   measures the budget again rather than the prompts.

---

## 2. What round 1 could not separate, and why

Round 1's conclusion is honest about the margin — "the margin over the other new variants
is one row" (`ORCH-747-RESULT.md:14-15`) — but the reasons are worth stating precisely,
because they are design constraints on round 2, not bad luck.

**2.1 Three of the grid's five columns had no variance.** In the table at
`ORCH-747-RESULT.md:37-48`: `rows with no evidence` is 0 for 10 of 10; `bad vocabulary` is 0
for 9 of 10 (and the tenth is the control); `reactions` is 0 for 9 of 10. A column that is
at the floor for every treatment cannot order treatments. The grid had **one** column with
real variance — interaction count, 11 to 23 — plus `DNA entities`, which is partly a
restatement of it.

**2.2 That one column is not a quantity the evaluator scores.** `reactions.score_leg`
scores `processes.reactions` and nothing else (`docs/evaluation_v2/README.md` § 3, "Scored
bucket"); interactions are "recorded but not scored" and the current default is recall-only
counts with no precision or F1 (`ORCH-747-RESULT.md:248-250`). So the round-1 grid ranked
variants on a number the instrument does not convert into precision, recall or F1.

**2.3 One draw per variant, against a known non-determinism.** `ORCH-747-RESULT.md:33`
records one draw each, and `CLAUDE.md` states as a standing constraint that identical legs
give materially different Stage-1 draws at temperature 0. `make_config.py`'s own closing
comment says to use `--repeat 3` before trusting any difference. An 11-vs-16 interaction
count across single draws is not a result.

**2.4 The one paper's gold is not certified.** `gold_paper_pw701720.json` carries
`supported_reactions_complete: false`, and `report.evaluate` / `run_live` skip an uncertified
case, so reaction P/R/F1 is defined only over the certified cases
(`docs/evaluation_v2/README.md` § 1, § 10). PMC319880 therefore cannot contribute a scored
number at all — in round 1 or round 2.

**Consequence.** Round 1 established that nine of ten variants do not break evidence
discipline or vocabulary. It did not, and structurally could not, establish which is best.
I treat `v03` as the base because the brief instructs it and because its rule is additive
and defensible, not because the grid showed it winning.

---

## 3. Where I disagree with round 1's conclusions

### D-R2-1 — "v03 is near-inert on the metabolic papers, which is the point" is the defect, not the design

`VARIANTS.md:90-97` defends v03 as near-inert on metabolic papers: "there is no coarser
restatement of a conversion to suppress". That is true of v03's rule **as written**, and
that is the problem. The rule's match test is explicitly regulatory: two rows restate one
claim when "the regulator, the sign and the consequence are the same and only the grain of
the target differs" (`v03:229-233`). A metabolic paper has the identical defect in a
different shape, and the test does not fire on it:

| paper | the coarse restatement the text itself states | gold id |
|---|---|---|
| PMC12421875 | "MK biosynthesis starts from chorismate and proceeds through a seven-enzyme pathway … to produce DHNA" | `summary_statements` S1 |
| PMC12096016 | "2,3-DHB is produced from chorismate by sequential reactions catalyzed by three enzymes" | `summary_statements` S1 (and S2) |

A row carrying chorismate → DHNA has two real sides, a verbatim quote and a true claim, and
is never a true positive: the matcher classes it `paper_stated_summary`
(`docs/evaluation_v2/README.md` § 3). So on the two certified papers the lumped sentence is a
guaranteed precision leak that v03 does not address at all. `r2_08` is the generalization of
v03's own idea to the metabolic shape, and in my view it is the single most defensible new
rule in the set.

**Round 1's restatement metric has the same blind spot.** It counted "pairs that share a
source and a relationship where one target's name contains the other's"
(`ORCH-747-RESULT.md:59-62`). That detector cannot see a lumped metabolic summary, because
the summary's endpoints are *different species*, not a containing name. So n=1 was not only
small; it was the wrong instrument for the papers round 2 is scored on.

### D-R2-2 — "zero reactions is arguably correct" on PMC319880 is contradicted by round 1's own artifact

`ORCH-747-RESULT.md:27-28` frames v06/v07 as "opposite answers to a product question", with
v07 asserting that zero is right. But `docs/evaluation_v2/pathbank_gold/gold_paper_pw701720.json`
— drafted in the same task, from the authoritative `r3` text, with every quote verified —
books **one in-scope reaction**: `R1`, RNA polymerase, inputs `araC` / `araC gene` /
`araC DNA`, output `araC transcript`, enzyme `RNA polymerase`. It additionally books the
lumped araC gene → araC protein statement as `summary_statements` S1 and the araBAD → enzymes
statement as `off_pathway_reactions` O1.

The gold is not certified, so this does not settle the product question — but it does mean
the correct target on this paper is most likely **1 reaction, and that reaction is a
transcription step**, not zero. A variant that forbids transcription outright (v07) would be
scored as a miss by the very artifact round 1 produced, and a variant that suppresses
reactions to look clean on this paper destroys 5 and 9 reactions on the other two. **I
therefore wrote no round-2 variant that legislates the transcription question in either
direction.** D-open-2 is a product decision, and until it is made, any rule touching it
makes its variant unscoreable rather than informative. Where my variants must mention the
shape — `r2_06`'s two-sided gate — the wording is deliberately chosen so that a transcription
claim stated with two named sides still passes ("a gene … is not the product of a conversion
*unless the text states the conversion that produces it*"), while the real round-1 defect,
`araBAD operon → araBAD operon`, fails. That one line separates the two cases without
deciding the policy.

### D-R2-3 — the Example-2 contradiction was never measured on its own

`ORCH-747-RESULT.md:185-188` (F-747-6) identifies the prompt's worked Example 2 emitting
`"inhibits"` and `"activates"`, the exact values its enumeration forbids, and says "v09
repairs it". But v09 is two hunks — a full rewrite of the vocabulary rule *and* the example
repair — described in `VARIANTS.md:373` as "one idea, two places". It is not one idea: a
prohibition replaced by a construction procedure is a different instrument from fixing a
counterexample. Round 1's conclusion that "deriving the value from `class` added nothing on
top" (`ORCH-747-RESULT.md:52-53`) is therefore a statement about the *bundle*. `r2_11`
isolates the 241-byte example repair, which is the cheapest candidate in the set and the one
whose effect is most likely to generalize, because Example 2 is a metabolic example
(allosteric inhibition of an enzyme) and both certified papers state exactly that pattern —
2,3-DHB competitively inhibiting apo-EntB, DHNA allosterically inhibiting MenD.

### D-R2-4 — the self-check's danger was correctly spotted and the remedy mis-stated

`VARIANTS.md:482-489` notes that v10's participant check can be destructive: a model may
delete a correct reaction rather than register its participant. The remedy it names is the
closing instruction "omit the row when it cannot be fixed without invention", and it
concedes that this is "right for check 7 and wrong for check 6". A checklist whose own
author identifies one item as mis-instructed should be fixed, not shipped and measured.
`r2_14` carries the check with the repair written into it: *add the entity row — the
participant is already stated by the row you are checking, so registering it invents
nothing*. More generally, every self-check in this set is written to be two-directional
(it can find a missing row as well as a spurious one), because a suppression-only check is
the generalization trap in miniature: it looks like precision on the regulatory paper and
costs recall on the metabolic ones.

### D-R2-5 — minor: round 1's cost accounting understates the measurement problem

`ORCH-747-RESULT.md:270-273` books the grid as costing "about three times what it should
have" because of the failed attempts. The larger waste was spending 10 variants × 1 draw on
a paper that cannot produce a scored number (§ 2.2, § 2.4). Round 2's three-paper design
fixes that and is the reason the brief's generalization trap is now detectable at all.

---

## 4. The JSON contract, re-derived from the code

Only the parts that bound a prompt variant. Full detail in `SCHEMA_CHECK_R2.md`.

* The prompt file is read at `pipeline.py:272` and reaches the model unaltered through
  `_run_json_stage` (`:3608`); the user prompt is assembled at `_build_extraction_prompt`
  (`:4311`) and carries `<pathway_scope>`, the pathway-context header, optional
  `<user_task_context>`, and on rung 3 a narrowed text plus `<extraction_scope>`. Nothing
  rewrites the system prompt, so a variant is the whole treatment.
* **Dropped before any reader sees them**, so no variant may be judged on them:
  `class`, `confidence`, `provenance`, `source_refs`, `spontaneous` on every process row
  (`pipeline.py:2229-2265` reactions, `:2374-2405` interactions);
  `compartment_canonical` on every biological state (`:1772`). `modifiers` is merged into
  `enzymes` at `:2239`, so the model may emit either name.
* **Survives and has a mechanical effect:** `scope_membership`, carried by
  `_carry_scope_membership` (`:2083`) and consumed by `filter_out_of_scope_reactions`
  (`:288-368`), which removes a reaction only on an explicit `out_of_scope` verdict. This is
  the only precision lever in the set with enforcement behind it rather than hope, which is
  why `r2_09` matters more than its size suggests.
* **The merge-gate-7 hole is still open:** `stage_one_boundary._paper_entry` tests
  `row.get("provenance")` against `("inferred", "enriched")` (`:212`, used at `:311`) on the
  *cleaned* payload, where that key is already gone for process rows. No prompt variant can
  close it.
* `draft_graph`'s mis-typing of DNA entities **is fixed** in this tree: `ENTITY_KIND_MAP`
  (`draft_graph.py:119-125`) is now the single source for both node creation and
  `_resolve_actor_kind` (`:128`, registry built at `:231-235`). Round 1's warning that
  site-level variants inflate orphan counts therefore no longer applies.

---

## 5. What the three papers actually demand — the trap, made concrete

| | PMC319880 (regulatory) | PMC12096016 (metabolic, certified) | PMC12421875 (metabolic, certified) |
|---|---|---|---|
| in-scope reactions | 1, a transcription step (gold not certified) | 5 | 9 |
| off-pathway reactions the paper also states | 1 (lumped araBAD expression) | 4 (competing MenD branch, LDH coupled assay, TolC secretion, Fe chelation) | 3 (biosensor Fe reduction in *L. plantarum*, ferrozine colorimetry, DHNA efflux) |
| lumped summaries that must not be rows | 1 | 2 | 1 |
| regulation the paper states, which belongs in `interactions[]` | the entire result | 2,3-DHB inhibits apo-EntB (the paper's title claim), Fur at the ent promoters | DHNA allosterically inhibits MenD |
| the recall difficulty | content is occupancy at named DNA elements | R3's substrate is named only through the enzyme's systematic name; R4's product only through "adenylated" plus the ligase name | 5 of 9 steps are stated only through letters A–G assigned to intermediates and the rate laws of equations (5)–(11); one step's verb is wrong while its pair is right |

Read across that table and the trap is arithmetic. A rule that suppresses reactions scores
0 wrong on paper 1 and loses up to 14 true positives on papers 2 and 3. A rule that chases
reactions books up to 8 off-pathway rows and 4 summary rows as claimed extraction, which is
precision loss on exactly the two papers where precision is measured. **And both metabolic
papers state regulation prominently** — one of them in its title — so a prompt that routes
"metabolic paper → reactions only" loses the claim the paper exists to make.

That is why the two highest-value directions in this set are not the router at all: they are
**shape gating** (a claim lands in the bucket its own grammar fits: `r2_06`) and
**recall of indirectly stated conversions** (`r2_07`), both of which are kind-neutral by
construction.

---

## 6. How round 2 has to be scored to separate anything

1. **Three draws per variant per paper, minimum**, per `make_config.py`'s own note and
   `CLAUDE.md`. 16 variants × 3 papers × 3 draws is 144 calls; if that is too much, cut
   variants, not draws — § 2.3 is why.
2. **Score the two certified papers with the existing instrument**: `matcher.py` strict
   equality, gold aliases only, congeners never credited, and the row classes of
   `README.md` § 3. Report precision, recall, F1 **and** the off-pathway / summary /
   partial counts separately, because a prompt can buy recall with summary rows and that is
   invisible in F1 alone.
3. **Score PMC319880 as a precision-side control**, never as F1: count (a) reactions whose
   two sides are not chemically different species, (b) rows with no evidence quote, (c)
   reactions with no counterpart in the drafted paper gold, (d) interactions outside the
   four allowed relationship values. Zero on (a), (b) and (d) is the pass bar.
4. **Report the three papers side by side per variant, never summed.** A single mean across
   paper types is exactly the number that hides the failure this round exists to expose.
5. **Run everything under the bounded wrapper** with zero surviving processes, and at the
   64,000 output budget (§ 1.3).

---

## 7. What I deliberately did not touch

* The transcription question (D-R2-2). No variant includes a v06- or v07-style rule.
* `synonyms`, `compartments[]`, and any field the cleaner drops: unreachable from a prompt.
* The alias-row and canonical-name problems: round 1 proved them to be post-Stage-1
  (§ 1.2).
* `pathway_curator_system.txt` and `pwml_infer_system.txt`: out of scope, and round 1's
  compatibility matrix (`AUDIT.md` § 3) holds unchanged for every round-2 variant, since
  none of them widens what Stage 1 may emit.
