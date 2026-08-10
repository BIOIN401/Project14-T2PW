# Change Log

Every entry answers: what was the error, why did it appear, and how does the
fix stay consistent with the intended pipeline design.

---

## Degree zero is answered against the pre-prune process snapshot (2026-08-10, branch `agent/p01-stale-index`)

**Files changed:** `src/t2pw/pipeline/strict_quarantine.py`,
`tests/test_strict_quarantine.py`, `tests/test_strict_quarantine_real_artifact_replay.py`

`quarantine_and_close` compacted every process bucket in
`_drop_quarantined_processes`, then asked `_degree_zero_exports` a question
phrased in the ORIGINAL admission indices: out-of-range ones were skipped in
silence and shifted ones resolved to the wrong row, so entities reached only by a
skipped row read as degree-zero and a complete, correctly declared pathway
refused to export — the stale positional index `PRODUCT_CONTRACT.md` § 1 names as
a blocker that must never end a run without a PWML. The buckets are now
deep-copied into an immutable snapshot immediately before the compaction, and
`_degree_zero_exports` takes its REFERENCE set from that snapshot while still
reading entity rows from the post-drop payload. `_surviving_processes` grew
`strict=`, raising the new `StrictQuarantineInvariantError` instead of skipping;
it defaults to False because `_revalidate_surviving_processes` is *supposed* to
meet a vanished row and records `process_row_vanished_during_closure`.

No gate was weakened: six of 32 archived legs stop being refused for a reason
that was never true, no leg gains a degree-zero entity, `PMC12856317/research`
stays refused on `unexportable_entity:1`, and `PMC12452463` clearing this
boundary is not strict success — its gold `export_rationale` records the route as
chemically broken, so its outcome is `review_required`. Pinned per leg in
`EXPECTED_P01_DELTAS`. **Re-measured full-stack across the 39 cached legs** of the
frozen cohort in `tests/data/baseline_cohort_manifest.json`, unchanged:

| stage | result |
|---|---|
| quarantine | 28 admitted, 11 refused |
| Stage 3 after quarantine | **28 / 28 pass** |
| required-field contract | 9 / 28 pass |
| IR build + validation | 9 / 9 of those reaching it |
| **fully exportable** | **9 / 28** |

That is the whole delta: one cohort leg,
`runs/2026-08-02_2130/papers/PMC12096016/strict`, moves from refused to admitted
and clears every later stage (39 legs; 28 admitted / 11 refused; 28/28 Stage 3;
9/28 required contract; 9/9 IR; 9/28 exportable; by leg 19/19/4, by row 27/27/4).
The residual codes below do not move; that leg contributes none.

---

## A decision is bound to its inputs, not only its payload; research fails open through the whole boundary (2026-07-31, branch `research-mode`)

**Files changed:** `src/t2pw/pipeline/strict_quarantine.py`,
`src/t2pw/app/streamlit_app.py`,
`tests/test_strict_quarantine_versioning.py`,
`tests/test_streamlit_quarantine_boundary.py`,
`tests/test_strict_quarantine_real_artifact_replay.py`,
`tests/test_strict_quarantine.py`

Correction of the entry below, which closed two thirds of each of its own gaps.

**1 — Reuse compared the payload and ignored the rules.** The previous entry bound
a decision to `resulting_payload_hash` and stopped there. `export_mode`,
`strict_db` and the thresholds were *recorded* on the report and never compared,
and a field nobody consults is not a check. The consequence is not subtle: a
research decision quarantines nothing and hands back the candidate unchanged, so
its resulting hash is the candidate's hash — feed that candidate to a strict
export and the payload half matches perfectly, and every unmapped process the
strict run exists to stop ships under a report that never judged it strictly.

`canonical_decision_inputs` now names every control that can change a verdict —
policy version, export mode, `strict_db`, the canonical requested core (explicit
argument and the context-derived terms, normalized), confidence floor, both core
thresholds, the iteration cap — and `decision_input_hash` fingerprints it.
`decision_matches` requires **both** halves; `decision_matches_payload` is gone
rather than left as a footgun that looks sufficient. The context contributes its
derived core terms rather than its raw bytes: a moved paper title cannot change a
verdict, and voiding a good decision over one is a re-run the reviewer waits
through for nothing.

`QUARANTINE_POLICY_VERSION` is in the hash, so changing this module's rules
invalidates every stored decision instead of letting an old report authorize an
export judged by new logic.

**2 — Mode and policy changes void the run; they are not re-evaluated.** After a
strict boundary the payload in the exporter's hand is an *already reduced* graph.
Re-judging that under research rules would report a research decision over
material a strict pass had already removed — annotating a graph the reviewer never
saw, and truthfully claiming nothing was quarantined. So the controls are locked
for the pipeline run: a payload that moved is re-quarantined as before, but
controls that moved return `quarantine_decision_controls_changed:<field>` and
require a new run. History is keyed on the full decision identifier
(`<admitted payload hash>.<decision input hash>`), so the strict and research
decisions over one payload cannot overwrite each other — they share the payload
hash exactly and reach opposite conclusions.

**3 — Research mode failed open at quarantine and closed one line later.** The
previous entry recorded this as a pre-existing limit and left it. It was still a
blocked run: annotate-only quarantine in front of a mode-blind Stage 3 gate
delivers nothing for the exact pathway the mode exists for, because a novel enzyme
with no accession fails that gate for the same reason quarantine flagged it, and
`refinement_working_json` was never initialized. Stage 3 still **runs** in research
mode and its findings are kept in full — `final_stage3_gate_report` unchanged,
`research_stage3_review_flags`, `research_stage3_failed_open` — but they annotate
instead of blocking, and refinement review opens on the byte-for-byte candidate.
The UI says the gates did *not* pass and never says "ready for review". PathWhiz
mode is untouched and still fail-closed.

**4 — The lifecycle test tested itself.** It called `clear_quarantine_artifacts`
by hand and asserted the call worked. The reset is now also wired into the start
of the audit/mapping run, and the test drives two runs in one session through the
button: the second dies in mapping, *before* the boundary, so the first run's
session keys and artifacts being gone can only mean they were cleared ahead of it.
`quarantine_history/` survives, and the first decision is checked against
`decision_matches` for the second payload directly.

**5 — The full-stack baseline is pinned, not narrated.** Same measurement as
below (39 legs; 27 admitted / 12 refused; 27/27 Stage 3; 8/27 required contract;
8/8 IR; 8/27 exportable; by leg 19/19/4, by row 27/27/4) now asserted as an exact
equality in `test_the_full_stack_baseline_is_exactly_what_was_reported`, with leg
counts and row counts kept as separate assertions. Re-measure and update both the
test and this log together, and the two cannot disagree. *(Numbers re-measured
2026-08-05 over the frozen cohort — see the note under the table below.)*

**Also removed:** `assert ... == candidate or True`, which was tautologically
true. The intended assertion — the research candidate equals the mapped payload it
came from, byte for byte — is now made.

**Tests:** decision-versioning 25 · quarantine unit suites 150 passed / 8 skipped ·
boundary AppTests 21 · real-artifact replay 97.

---

## One decision per payload version; research mode fails open; Stage-3 recovery is not exportability (2026-07-31, branch `research-mode`)

**Files changed:** `src/t2pw/pipeline/strict_quarantine.py`,
`src/t2pw/app/streamlit_app.py`,
`tests/test_strict_quarantine_versioning.py` (new),
`tests/test_streamlit_quarantine_boundary.py`,
`tests/test_strict_quarantine_real_artifact_replay.py`

Correction of the entry below.

**1 — A decision outlived the payload it was made about.** The boundary runs
before refinement review; the reviewer can then delete the requested core in one
click, and `apply_grounding` rewrites entity identifiers inside the exporter. The
stored report was reused on the strength of *existing*, so either edit shipped a
graph nothing had admitted. Every report now carries `admitted_payload_hash` (what
it judged) and `resulting_payload_hash` (what it produced), and reuse is gated on
the latter — the exporter holds the reduced graph, so comparing against the
admitted hash would miss on every run where anything was actually quarantined.
Mismatch re-runs the boundary and writes a new decision; the superseded set is
archived under `quarantine_history/<hash>/` rather than overwritten, because each
is the only record of why that version was admitted. A report with no hash never
matches: it cannot prove what it judged.

**2 — Research mode was being handed a smaller strict graph.** Research mode
exists for a novel pathway whose enzymes have no accessions yet, so every one of
them is `quarantined_unmapped_entity` by construction — a destructive quarantine
there deletes exactly what the mode was built to keep. It now runs every decision
and applies none: the candidate comes back byte for byte, `ok` is True regardless
of coverage, and the findings surface as review flags with the strict verdict
recorded under `research_mode.would_have_refused`. Fail-open, never fail-silent.

**3 — Lifecycle.** `reset_quarantine_state` clears the session keys *and* the
on-disk artifacts when a new pipeline run begins. `st.session_state` survives a
rerun and `outputs/quarantine_report.json` survives the whole session, so without
it a second run that died before the boundary would render the first run's
coverage summary for a different pathway.

**4 — The export is now exercised, not assumed.** A test that stops at refinement
review proves the run was not blocked; it does not prove a file comes out. The
AppTest suite now clicks **Generate PWML** and asserts Stage 3, the required-field
gate, `build_pwml_ir`, `validate_pwml_ir`, `ok is True`, and non-empty XML on
disk — with the quarantined process absent from the bytes.

**5 — Stage-3 recovery is not strict exportability, and the previous entry
conflated them.** That entry reported "remaining downstream gate failures: none"
on the strength of Stage 3 alone. Measured full-stack across the 39 cached legs:

| stage | result |
|---|---|
| quarantine | 27 admitted, 12 refused |
| Stage 3 after quarantine | **27 / 27 pass** |
| required-field contract | 8 / 27 pass |
| IR build + validation | 8 / 8 of those reaching it |
| **fully exportable** | **8 / 27** |

> **Re-measured and frozen, 2026-08-05 (branch `agent/h01-baseline-manifest`).**
> The table above originally read 23 legs / 18 admitted / 1 exportable. It went
> stale because the cohort was discovered by globbing `runs/`: the 2026-08-02_2130
> batch added 16 legs and nobody re-measured, so the pinned totals and the actual
> population had disagreed since 2026-08-04. Worse, every milestone benchmark
> archives a run directory, so the population — and therefore the merge gate —
> was being redefined by the sprint it was supposed to be guarding.
>
> The cohort is now an explicit, version-controlled manifest,
> `tests/data/baseline_cohort_manifest.json`: 39 legs, each verified present,
> tracked and within the payload bound before freezing. The replay harness reads
> only that file, a missing entry fails loudly rather than shrinking the cohort,
> and a leg enters or leaves only through a reviewed edit. **No pipeline
> behaviour changed** — nothing under `src/` was touched — so the movement from
> 1/18 to 8/27 exportable is entirely the 16 previously unmeasured legs, not a
> change in any leg's verdict. This is also *not* C-010's per-leg delta
> allowlist (`docs/pwml_recovery_sprint/BASELINE.md` § 6), which is a different
> population spanning `runs/` and `runs_verify/`; the two must not be conflated.

The 19 failures are one class, and quarantine is the wrong place to fix it:
`species_missing_classification` (19 legs, 27 rows), `species_missing_taxonomy`
(19 legs, 27 rows), `no_biological_states` (4 legs, 4 rows). The gate wants a numeric
taxonomy id and a Prokaryote/Eukaryote classification on every species row because
a species with no reference-DB identity is created fresh in Rails; the archived
payloads carry species rows with neither. Inventing them inside quarantine would
be fabricating database identity, so this is recorded as **the next pipeline
defect** — species-metadata resolution — rather than repaired here.
`test_stage_three_recovery_is_not_strict_exportability` pins both halves and will
fail if the gap closes upstream, so the claim cannot drift from the measurement.

**Also recorded, not fixed.** The app's post-mapping Stage 3 revalidation is
mode-blind and always has been (unchanged at HEAD). A research run whose enzyme
has no accession is still stopped there, before refinement review opens — the
quarantine boundary passes it, the gate two lines later does not. That is a
pre-existing gap in the app's gate cadence with its own blast radius;
`test_research_mode_keeps_the_unmapped_candidate_and_does_not_block` asserts what
quarantine controls and states the limit explicitly rather than hiding it.

---

## The quarantine boundary was unreachable in production, and four rules disagreed with the schemas (2026-07-31, branch `research-mode`)

**Files changed:** `src/t2pw/pipeline/strict_quarantine.py`,
`src/t2pw/app/streamlit_app.py`, `src/t2pw/curation/apply_audit_patch.py`,
`src/t2pw/curation/gap_resolver.py`, `src/t2pw/curation/pathway_curator.py`,
`src/t2pw/pipeline/focused_repair.py`,
`tests/test_streamlit_quarantine_boundary.py` (new),
`tests/test_strict_quarantine_contract_alignment.py` (new),
`tests/test_strict_quarantine_locks_and_scope.py` (new),
`tests/test_strict_quarantine_real_artifact_replay.py` (new),
`tests/fixtures/strict_failures/cases.json`

Correction of the entry below. Every defect here is one the unit tests could not
see, and each was found by driving the real app or replaying real run artifacts.

**1 — Quarantine ran where production never reaches.** It lived inside
`run_pwml_export`. The app stops one step earlier: the post-mapping Stage 3
revalidation (`streamlit_app.py:5422`) sets `refinement_gate_errors` and never
initializes refinement review, so `_generate_pwml_from_refinement_working_json`
returns on those errors and the exporter is not called. A payload that failed
Stage 3 — the exact case quarantine exists for — reached it never. There is now
one boundary, `run_quarantine_boundary`, immediately before that check; the
reduced payload becomes `refinement_working_json`, so the reviewer and the
exporter see the same graph, and `run_pwml_export` carries the decision rather
than making a second one against a different payload.

**2 — The requested core came from the payload, which does not carry one.**
Stage 6 rebuilds rows from field whitelists that include neither `metadata` nor
`pathway_context`, so discovery found nothing, `requested_core_declared` came back
False, and coverage degraded to the regime where an unrelated survivor passes — on
every production run. `pathway_context` is now an explicit argument, threaded from
session state, and `coverage_summary.json` records it verbatim alongside a
`requested_core_source` saying which input produced the terms. No archived leg in
`runs/` carries `key_compounds` anywhere, which is the evidence that payload-only
discovery was never going to work; `test_no_archived_leg_carries_stage_zero_context`
pins that.

**3 — Three schema readers were narrower than production.** Transport cargo was
read from `cargo` alone, so every transport written with `transport_elements` —
the shape `ir.py:1780` reads *first* — was quarantined as having no participants.
Interaction endpoints were read from `entity_1`/`entity_2` alone, so `left`/`right`
and `source`/`target` rows, both of which `validate_registry_references:4188`
resolves, were quarantined the same way. Both are now read exactly as production
reads them, with positive controls per spelling. Separately,
`reaction_coupled_transports` were being *accepted*: `build_pwml_ir:1934-1946`
builds every RCT with `left`, `right` and `enzymes` hard-coded to `[]` and never
fills them, and `validate_pwml_ir:3018-3022` then raises three errors for exactly
that. There is no exportable representation of one today, so they are now refused
explicitly, and a test asserts the IR limitation so the refusal cannot outlive it.

**4 — Representability disagreed with the contract in both directions.** Missing
protein name and species, missing complex name and species, and generated-wrapper
component species were not checked, so a row could be admitted and fail
`validate_required_pwml_contract` afterwards. In the other direction, component
resolution was treated as blocking for *every* complex, while the contract makes it
an error only for a generated wrapper — which would have quarantined every reaction
catalysed by a hand-declared multi-subunit complex. Nameless rows now leave in
closure (the contract walks every declared row, so an unreferenced one still fails
`compound_missing_name`), and a surviving row the contract will reject is reported
as `unexportable_entities` and refused rather than silently deleted. All of these
assert through the *real* gates, not internal state.

**5 — A quarantined locked reaction vanished while the report said it exported.**
`locked_reaction_filter_report` is what the Stage 3 gate reads to decide whether
every lock is accounted for. It is now recomputed from active plus quarantined lock
ids, in the same shape and by the same rule `dedupe_processes:4061-4085` uses, and
the quarantined ones are appended to `quarantined_locked_reactions` with the
original row. An unaccounted lock refuses the export with its own reason.

**6 — The applied-patch log asserted edits that were undone.** On a batch rollback
the accepted ops stayed in `applied_patch_log`, each stamped `accepted`, while the
payload was byte-identical to the input. They now move to the rejected log stamped
`rolled_back` and the applied log for that batch is empty. Every consumer that
reports "how many patches landed" — streamlit candidate scoring (two sites plus the
round summary), `pathway_curator`, `gap_resolver`, `focused_repair`, and the CLI —
now reads `committed_change_count()`, which prefers `transaction.applied_count` and
falls back to `summary.accepted_count` only for reports written before
`transaction` existed. `summary` keeps its three-key shape.

**7 — Refusals were indistinguishable.** `minimum_core`, `entity_type_overlap`,
`degree_zero_export`, `unexportable_entity`, `unaccounted_locked_reactions` and
`closure_not_converged` are now separate named reasons on `refusal_reasons`;
`ok` is derived from that list, so an invariant cannot fail silently. The
quarantine report and artifact paths ride on every later Stage 3, required-contract
and IR failure return.

**Found by replaying real legs.** A clean pathway of four interactions and no
reactions (`runs/2026-07-28_2122` PMC12624714/strict) was refused: with no declared
core every accepted process falls to AUXILIARY unless it is a reaction, so
`core_accepted` was zero. Undeclared means relevance is unjudgeable and the only
rule left is "not empty", so the minimum now counts all accepted processes in that
regime and `core_accepted` only when a core was actually declared. Across 23 cached
legs: 15 recover, 5 are refused (all genuinely empty after quarantine), 3 were
already clean, 0 still fail, 0 clean legs refused.

**Also corrected.** `quarantined_disconnected` is documented as what it is — a
backstop that stays at zero while every removal is reference-driven, kept so a
future non-reference-driven removal leaves with a reason instead of putting a
dangling reference in front of the gate. It is tested by simulating exactly that
removal. The synthetic coupled-transport chain that previously demonstrated the
state is gone.

---

## Pre-export quarantine and graph closure; curation patches commit as a set (2026-07-31, branch `research-mode`)

**Files changed:** `src/t2pw/pipeline/strict_quarantine.py` (new),
`src/t2pw/curation/apply_audit_patch.py`, `src/t2pw/app/streamlit_app.py`,
`tests/test_strict_quarantine.py` (new), `tests/test_curation_patch_transaction.py` (new),
`tests/test_strict_failure_replay.py` (new),
`tests/fixtures/strict_failures/cases.json` (new)

**Error / symptom.** Strict export was all-or-nothing. One over-generated peripheral
reaction naming a participant no entity bucket declared failed the Stage 3 registry
gate for the whole pathway, and the nine good reactions beside it were never written
(AGENT_INSTRUCTIONS issue 5a, jasmonate biosynthesis: 21 errors, 48 warnings). The
recovery repeatedly reached for — delete the offending *participant* and re-run — is
worse than the failure: it exports a reaction the paper does not contain, silently.

**Root cause.** There was no stage between "the payload is final" and "the gates
judge it" at which a single unexportable row could be set aside as a unit. Every
existing removal path (`filter_unresolvable_reactions`, `prune_disconnected_proteins`)
handles one shape and leaves the graph in whatever state that shape's removal
produced; nothing closed the graph afterwards, and nothing checked that what
survived was still the pathway that was requested.

**Fix — `strict_quarantine.py`.** Seven explicit admission states
(`core_accepted`, `auxiliary_accepted`, and five `quarantined_*` reasons), then
closure to a fixpoint. Four rules make it safe in front of the gates:

1. *The unit is the process, never the participant.* An essential participant that
   cannot be represented takes the whole process, and the original row is retained
   verbatim in `quarantine_report.json`.
2. *Closure only removes what nothing references,* so it can never strand a
   reference it did not already have. It iterates because a coupled transport's
   `reaction` name resolves at admission against the payload as it arrived, and
   only closure can see that its target has since left — a chain that unwinds one
   link per round.
3. *A smaller graph still has to be the requested pathway.* Coverage is measured
   against `core_accepted` processes only, so an empty graph and a graph of
   unrelated survivors both fail rather than passing because the invalid material
   is gone. Removing everything is not a way to succeed.
4. *Nothing is deleted without a record:* `quarantine_report.json`,
   `removed_entity_report.json`, `graph_closure_iterations.json`,
   `coverage_summary.json`, all four always written.

Correctly formed Unknown-backed functional complexes are explicitly valid and need
no special case: the component IS the PathBank `Unknown` sentinel, which
`is_pathbank_unknown_protein` accepts, so the generic component walk admits the
complex. A placeholder that *forges* an accession is still quarantined.

**Fix — transactional curation patches.** Two holes the per-op guard could not see:

* `_REGISTRY_ENTITY_BUCKETS` was three buckets while `validate_registry_references`
  had grown to four. Coverage that does not count `nucleic_acids` cannot register a
  loss in it, so every nucleic-acid removal came back with an empty `lost` set and
  was waved through — including rows that reactions name as inputs
  (`pmrHFIJKLM operon`, PMC13278307).
* The removal look-ahead is a promise, and nothing confirmed it was kept. The
  guard credits any list of strings an `/entities` op would introduce without
  checking which leaf it lands on, so an add to `aliases` — or one that fails to
  apply at all — authorised a removal it could not compensate for. There is now a
  commit point: if the finished payload has a reference orphaned by coverage the
  batch removed and did not restore, the whole operation set rolls back and
  `report["transaction"]` says why.

The rollback is scoped to *lost coverage*, not to every dangling name. A reference
that dangles because a process op introduced it belongs to the Stage 3 registry
gate; rolling back over it here would refuse the reactions-array replacements this
module has always applied. `report["summary"]` keeps its exact three-key shape.

**Pipeline consistency.** No gate was relaxed. Quarantine runs *before* both gate
stacks — see the 2026-07-31 entry above for where that is in production — and both
stacks then run unchanged on less input.

A payload with nothing quarantined is **not** necessarily unchanged, and an earlier
draft of this entry claimed it was. Closure removes what nothing references, and it
does so whether or not any process was quarantined: an unreferenced compound, a
degree-zero protein, a nameless row, an `element_locations` entry whose entity is
gone. Those removals are the point — a degree-zero protein is a strict-gate failure
on its own — and they are all recorded in `removed_entity_report.json`. What holds
is narrower and still worth having: **quarantine never removes a process that both
gate stacks would have accepted, and never edits a surviving one.**

`out_of_scope` mirrors `filter_out_of_scope_reactions` exactly (only an explicit
verdict removes anything); weak-evidence quarantine is off unless the caller passes
a confidence floor, and never fires on an absent `confidence`.

**Replay.** `tests/fixtures/strict_failures/cases.json` stores eight compact
strict-failure shapes over one base pathway that passes strict export on its own.
Five now produce a smaller valid strict graph; two are correctly refused (empty
graph, unrelated survivors); one — the Unknown-backed complex — is untouched and
already passed. The eighth, a metabolite split at its colon and filed as a protein
complex, exposed a gap the current gates do not cover: they forbid a
proteins/protein_complexes overlap and nothing else, so a compound and a complex
sharing a name goes unremarked and the IR resolves the reference to whichever was
registered first. Quarantine treats an ambiguous name as a broken reference.

---

## Three residual repair-integrity gaps: the guards were half-guards (2026-07-30, branch `research-mode`)

Follow-up to the entry below. Each of these guards checked one direction of a two-directional
property, and the unchecked direction was in every case the *easier* failure mode to produce.

**Error 1 — the JSON content guard only prevented invention, not deletion.** It tested that
the repaired literal stream was an ordered SUBSEQUENCE of the malformed input's. A
subsequence cannot contain an invented token, cannot reorder and cannot duplicate — but it
can be *shorter*. So a repair could answer a trailing comma by returning one of two complete
reactions, or drop a middle entity row, and pass. Silently, from an extraction that was
correct about them.

*Fix.* Equality, not containment. The repaired stream must be exactly the input's complete
literals, in order, optionally minus the final one — and only when the document was cut off
mid-value AND that literal is the key the value belonged to (`_Scan.droppable_tail`). That is
the one case `_JSON_REPAIR_SYSTEM` asks the model to handle by deleting. A trailing comma, a
missing comma, a code fence and a missing closing delimiter all leave every literal intact, so
none of them authorizes deleting anything; the scanner sets `truncated` only for an
unterminated string or a bare literal running to EOF. Refusals now report `divergence`
(`added_or_reordered` vs `removed`) and a sample of what went missing, computed with
`difflib.SequenceMatcher` rather than a greedy forward scan — payload literals repeat
constantly (`name` is a key on every row), and a greedy scan matched a dropped compound's
`name` against the next reaction's and reported the wrong region.

**Error 2 — the row-repair guard walked the ORIGINAL's non-empty fields.** That is the wrong
place to stand to see an addition. Four things passed: adding an immutable field the row never
had, populating one that was present but empty, deleting one whose value was empty, and
introducing any brand-new top-level key. A fabricated `provenance` or `source_refs` is worse
than a missing one, because it is citable.

*Fix.* Exact shape over the union of both sides' keys. Outside the fields the contract named,
every key must be present if it was present, absent if it was absent, and equal if it was
equal — empty values included. Distinct reasons for `_added` / `_removed` / `_altered` and for
`immutable_` / `unrelated_`. "Named by the contract" is now read from the contract itself:
`_errors_by_row` carries `required_any` through (`_validate_reaction_structure` emits
`["inputs", "outputs"]`), so a participants error no longer licenses writing an `organism`.
Errors carrying no hint still fall back to every scientific field, because the alternative is a
repair that cannot repair. The `products` → `outputs` migration is preserved: atom loss is
still checked over the *union* of the named fields.

**Error 3 — numeric normalization went through `float`.** It bought `1e3 == 1000.0` and gave
away exactness. Every integer above 2**53 collapses, so `9007199254740993` and
`9007199254740992` were the same value to the guard — a taxonomy id, a PathBank row id or a
PubChem CID altered in its 16th digit compared equal and passed. Long decimals rounded the
same way.

*Fix.* `Decimal` in a 200-digit context, normalized and rendered fixed-point, which keeps
`1e3 == 1000.0` while distinguishing both cases. Floats are routed through `str` first
(`Decimal(0.1)` is the binary expansion and would never match the `0.1` in the source text),
and `-0` is canonicalized to `0` since JSON does not distinguish them. The guard's side of the
comparison parses with `parse_exact` (`parse_float=Decimal`) so a high-precision decimal is
not flattened before it is compared — without that, an *honest* repair of one would have been
refused. The payload the pipeline receives is still parsed normally: `Decimal` is not JSON
round-trippable and must not leak downstream.

**Verification.** Three new probes, run separately: JSON-repair deletion (24), row field
exactness (31), Decimal normalization (30). Adversarial Prompt 3 suites: 20 / 30 / 29 / 17.
Prompt 2 regressions: 554. Full suite: 1613 passed, up from 1528, with all four
previously-recovering early-failure fixtures still recovering.

---

## Five integrity gaps in the recovery path: instructions were standing in for enforcement (2026-07-30, branch `research-mode`)

Follow-up to the entry below. The recovery machinery landed and worked; these are the
places where it *asked* for a property rather than *enforcing* one.

**Error 1 — a model instruction is not enforcement.** `_JSON_REPAIR_SYSTEM` told the repair
model, in absolute terms, not to add, remove, rename, reorder or reword anything. Nothing
checked. A model that ignored it returned perfectly well-formed JSON containing a reaction
nobody extracted, and from that point on the invention was indistinguishable from extracted
biology — it had a name, participants, and flowed through cleaning, mapping and export like
any other. The repair pass was the only stage in the pipeline that could manufacture a
pathway out of a syntax error.

*Fix.* `json_repair_preserves_content` requires every complete key and scalar of the
repaired document to appear in the malformed input **in the same relative order**. One
ordered-subsequence test rejects all four attacks: invention and substitution contribute a
token the input cannot supply, reordering supplies the right tokens in the wrong order, and
duplication needs more occurrences than exist. Deletion stays legal because the input is
malformed *because* something was cut off, and the prompt asks for the incomplete pair to be
dropped — so the scanner ignores unterminated strings and any bare literal running to EOF.
Refusal is its own outcome, `semantic_guard_failed`, not a flavour of `invalid_json`: the
model produced valid JSON, so nothing about the syntax explains the rejection and the two
have opposite fixes. It is not retried, and the deterministic prefix salvage still runs. The
stored `stage1_invalid_json_localized_repair` fixture had to change: its repaired reply also
added an `evidence` field, which reads as harmless and is exactly what the guard now refuses
— a repairer that can add a field can add a reaction.

**Error 2 — grounding was one-sided, so repair could delete facts.** `evidence_supports`
asked whether *added* values were carried by the row's evidence and said nothing about
*removed* ones. So the cheapest way to satisfy a structural contract was to delete until the
row stopped failing, and every such repair passed the grounding check trivially because
nothing was added. Deleting an extracted input is worse than refusing the repair: the
compound was in the paper, the extractor found it, and no error anywhere said it went
missing.

*Fix.* `preserves_original_values` runs **before** grounding (deletion is the more damaging
failure, so it should be the reported one when a row fails both). Fields the contract did not
name must be byte-identical — including `name`, `evidence`, `provenance`, `source_refs`,
`source_papers`, `rag_provenance`, `scope_membership`, `locked_reaction_id`,
`preservation_status`, `confidence`, `inference`. Across the fields it *did* name, every
non-empty atom the original carried must still be present. Checked over the union of those
fields rather than field by field, because moving a participant from `products` to `outputs`
is the normalization a structural repair is *for*.

**Error 3 — reconstruction guessed what kind of actor it saw.** Actor roles mapped statically
to `entities.proteins`. But "enzyme" spans two buckets that are not interchangeable:
`proteins` is one gene product with a UniProt identity, `protein_complexes` is an assembly
with components and stoichiometry, and they hit different identity gates and different
PathBank tables. Filing a complex as a protein asserts it is a single gene product — a
biological claim, made by the one function whose whole premise is that it makes none.

*Fix.* Actor roles read the type the payload states (`protein_complex` / `protein` keys, or
`entity` plus a `protein`/`protein_complex` `entity_type`). An actor that states no usable
type is **skipped** with reason `actor_entity_type_unknown` and its name, so the gap is
reported rather than resolved by guesswork. Static compound reconstruction for reaction
inputs/outputs and transport cargo is untouched: there the role *is* the type.

**Error 4 — one recorder for the whole process.** Streamlit serves every browser session from
one process on its own ScriptRunner thread. A module-level recorder was shared across all of
them: session B's `activate` would discard session A's boundaries mid-run and both would
write the same filenames, so the artifacts became a blend of two papers with nothing saying
so — worse than no artifact, because it looks authoritative.

*Fix.* The recorder is a `ContextVar`, read and written per execution context; a new thread
starts from the default rather than inheriting, so nothing a worker records can land in
another run's artifacts. `activate(unique=True)` — which production passes — gives each
invocation its own subdirectory, and the resolved path is surfaced in the failure message.
The batch session-state hand-off is unchanged.

**Error 5 — bounding was per-entry-point.** `record_boundary`'s `**extra` keys were stored
verbatim, and `record_cleaning` / `record_mapped_failure` / `record_iteration` each stored
`dict(report)` untouched. The bound held for the shapes somebody had thought of, which is not
a bound.

*Fix.* One recursive sanitizer, `bound()`, at every entry point: `BULK_KEYS` become a
`{chars, hash, preview}` descriptor at any depth, mappings and sequences are capped in width,
strings are *clipped* rather than described (an `incomplete_reason` must stay readable), and
anything past `MAX_BOUND_DEPTH` collapses to a descriptor. The depth cap is 6, not
`failure_detail`'s 4, because a cleaning report's `discarded_sample` entries sit at depth 4
and censoring them would make the artifact useless to defend against a case that never occurs
in it.

**Verification.** Four adversarial groups, run separately: JSON-repair content invention (20),
row-repair deletion and provenance removal (30), actor entity type (29), and
isolation + recursive bounding (17), the last including a two-thread barrier regression and a
150 KB value nested three levels deep through every entry point at once. Prompt 3 targeted:
181. Prompt 2 regression (eligibility, identity, protein export, batch, contracts): 496. Full
suite: 1528 passed, up from 1432, with all four previously-recovering early-failure fixtures
still recovering.

---

## Early-failure diagnosability and bounded localized recovery (2026-07-29, branch `research-mode`)

**Error.** A Stage-0/Stage-1 death produced one sentence and nothing on disk. Both legs lost
in `runs/2026-07-28_0919` carry `files: []` in the manifest and have only `RESULT.txt` under
their run directories, which is why the postmortem in `llm/client.py` had to *infer* the
cause from the shape of the failure rather than read it. Five different faults all surfaced
downstream as the same message, "Payload must include a processes object":

1. the provider returned an empty completion;
2. it returned text that is not JSON;
3. it returned valid JSON declaring no processes;
4. it returned processes that *cleaning* then discarded row by row;
5. cleaning kept rows and the stage contract rejected the result.

Each of those has a different fix, and none of them was distinguishable from the others.
Recovery had the mirror problem: the only response to any of them was to re-issue the entire
extraction prompt — the largest prompt in the run — which returns a fresh sample rather than
the same content with the fault corrected.

**Why it appeared.** The facts that separate the five are produced deep in the call stack
(inside `chat()`'s retry loop, inside `_clean_processes`'s per-row `continue`) and consumed
at the top, by the app and the batch driver. Nothing carried them across, and every artifact
hand-off in the app sat *downstream* of the `st.stop()` that a failure takes — so the runs
that most needed evidence were exactly the runs that wrote none.

**Fix.**

*Diagnostics* (`pipeline/extraction_diagnostics.py`). One recorder per run, reached through
a module-level `current()` so recording is never conditional on a parameter nobody threaded
through. Every `record_*` call flushes to disk as it is made, so a stage that raises has
already written its evidence by the time the exception unwinds; no `finally` block has to be
correct. Each boundary record carries model, `finish_reason`, attempts, raw response status,
raw entity/process counts when parseable, cleaned counts, discarded-row counts by reason, a
capped sample of discarded names and pointers, the payload hash, and the stage and boundary
names. Everything is bounded at capture: counts, hashes and clipped previews only — never a
repeated evidence blob, because payload `evidence` fields reach six figures of characters and
these files are rewritten on every attempt. Artifacts: `stage0_attempts.json`,
`extraction_boundary_report.json`, `cleaning_report.json`, `mapped_failure_snapshot.json`
(only once mapping has run, so its absence stays meaningful), `audit_iteration_summary.json`
and `gap_iteration_summary.json`. `batch/driver.py` carries them into the leg from
`_add_common_artifacts`, which the Stage-1 failure branch already invoked.

*content_filter is durable* (`llm/client.py`). Moderation is a deterministic function of the
prompt, so re-sending an identical prompt gets an identical verdict. Both retry loops now
stop on the first empty `content_filter` reply and report `terminal_reason="content_filter"`
instead of spending `LLM_MAX_RETRIES` draws to arrive at the same empty string. A
`content_filter` reply that carries *text* is unaffected — a partial answer is an answer.

*Localized repair* (`pipeline/localized_repair.py`). Invalid JSON is shown to the model as
its own broken text plus the parser's error, with no source text in the prompt and therefore
nothing to re-extract. Contract-invalid rows are sent one row at a time with their exact
errors and their own evidence passage; valid rows are neither sent nor touched, returned
pointers that were not requested are discarded, and a repaired row whose scientific fields
are not present in the supplied evidence is refused and the original kept. Attempts are
capped and `content_filter` ends the sequence.

*Salvage no longer hides a lost payload.* `_extract_json_from_text` is a prefix scan: on a
reply whose syntax error lands inside `processes` it returns the entities that came before it
and drops every reaction — measured on the shape stored in
`tests/fixtures/early_failures/cases.json`. A salvage that lost the processes is now offered
to localized repair first, and the salvaged object is still used if repair fails.

*Deterministic registry reconstruction.* A participant named by a *valid* process row but
absent from the entity registry gets an unresolved shell carrying `name`, `provenance`,
`resolution_status` and its source pointer — no identifier, no EC number, no organism, no
function. This closes the silent loss where `filter_unresolvable_reactions` deletes correct,
evidenced reactions for a bookkeeping gap. Every identity gate downstream still refuses the
shell; the incompleteness is made explicit rather than laundered away.

*Incomplete is a reportable outcome* (`pipeline/stage_one_boundary.py`). When reconstruction
and repair are both spent and the contract still refuses the payload, `settle_stage_one`
returns the payload as far as it got with `ok=False` and a reason naming what blocked it,
what was already tried, and — when cleaning is the real cause — that the rows existed before
cleaning removed them. Nothing invents a reaction to satisfy `entities_required` or
`processes_required`.

**Consistency with the design.** `chat()` keeps its historical signature and returns exactly
the text it always did; `chat_detailed()` is the same call with the diagnostics attached.
`clean_stage_one()` is unchanged and delegates to `clean_stage_one_with_report()`, so its
dozens of call sites see no behavioural difference. Recording is additive everywhere and can
never raise: a diagnostics writer that can kill the run it is diagnosing is worse than none.

**Verification.** `tests/test_extraction_diagnostics.py`, `tests/test_localized_repair.py`,
`tests/test_stage_one_boundary.py`, `tests/test_early_failure_replay.py` (six stored
early-failure shapes replayed offline through the real Stage-0 → Stage-1 → boundary path),
plus new cases in `tests/test_llm_client_empty_completion_retry.py` and
`tests/test_batch_driver.py`. Full suite: 1432 passed, up from a 1322-passing baseline, with
no pre-existing test changed except where the seam it patched moved.

---

## Six corrections to the paper-eligibility gate: the false claim could still get back in (2026-07-29, branch `research-mode`)

Follow-up to the entry below; full rules in `docs/paper_eligibility.md`.

**Error.** The first pass at the gate removed the request-stamping bug at the source but
left six ways for its consequences to survive, four of which were demonstrable on the real
2026-07-28_2122 data.

1. *A legacy acquisition cache could reintroduce the false organism.* Cache files written
   before the split have no schema version and every candidate row's `organism` is the
   organism the search asked for. `_as_batch_paper` promoted `candidate.organism` into
   `observed_organisms` unconditionally, so reading one cached row put
   `"Escherichia coli"` back on the Fournier's-gangrene case report as an *observation*.
2. *`apply_stage0_observation` had no caller outside its tests.* The rule that Stage 0 may
   observe but not overwrite was implemented and verified in isolation and enforced
   nowhere.
3. *Species matching was too loose in one direction and unrepresented in the other.*
   `Escherichia` (bare genus) inferred `Escherichia coli`, and same-genus/different-species
   was silently folded into `match`, so `E. coli` vs `E. fergusonii` and `B. subtilis` vs
   `B. cereus` scored a full organism bonus with no signal that they were not the requested
   species.
4. *"Anchored" was a document-level test.* A pathway alias anywhere plus a generic word
   ("mechanism", "inhibition", "flux") anywhere else counted as mechanistic evidence. With
   abstracts, that admitted all five real cholesterol false-positive shapes — `SQLE`,
   `CYP51A1` and `DHCR24` named among differentially expressed hits from a proteomic or
   transcriptomic screen, and `PMC12993329`/`PMC12705669` naming the molecule with no local
   reaction at all.
5. *The fixed `want * 3` over-fetch outlived its calibration.* It was sized for a fetcher
   that took every hit with retrievable full text. Against a gate that rejects 21 of 28,
   one 3x pull under-delivers every topic, silently.
6. *The screening input was not persisted.* A rejection in `skipped.json` could not be
   reproduced offline, and the dry-run tool had no abstract to work from, so it could only
   ever screen titles.

**Why.** Each is the same class of mistake: a rule enforced at one seam and assumed at the
others. The split was made at the fetchers but not at the cache reader; the immutability
rule was written but not called; "same organism" was tested for equality-or-containment
rather than by species; locality was named in the design but implemented as a whole-document
keyword scan; and the funnel's sizing constant was never revisited when the funnel gained a
filter.

**Fix.**

* *Cache schema.* `CACHE_SCHEMA_VERSION = 2`. `migrate_cached_payload` runs on read and
  demotes a legacy stamped `organism` to `requested_organism`, clearing the observed side
  and keeping everything else, so an existing offline cache stays usable rather than being
  discarded. `search_candidates` reports `cache_schema_migrated`;
  `CandidatePaper.from_dict` applies the same demotion for any other reader; detection is
  on key presence, not value. The unconditional `candidate.organism` promotion is gone —
  `_as_batch_paper` now reads every observed field from the decision, which also makes
  `BatchPaper.observed_organisms`, `organism_match` and `eligibility["observed_organisms"]`
  unable to disagree (`BatchPaper.scope_disagreements()` asserts it,
  `eligibility_summary` reports it).
* *Stage-0 boundary.* `driver._reconcile_stage0_scope` runs right after stage 1 succeeds —
  the first point where the app's `session_state["pathway_context"]` and the plan's request
  both exist. It records the observation always, and on a contradiction writes a
  `scope_conflict.json` artifact, adds the `scope_conflict` issue code, and stops the run
  with `status="scope_conflict"`, which `report._norm_status` folds out of the failure tally.
  The request is never rewritten. Configurable via
  `RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS`.
* *Species precision.* `organism_match` gains `genus_level`. Exact binomials and
  strain-qualified forms are `match`; same genus/different species and bare-genus mentions
  are `genus_level` — permitted, but scored at 0.25 instead of 1.0, warned about, and
  flagged for review. The lexicon holds species-level aliases only; genera drive a binomial
  scan that can observe a species the lexicon has never seen.
* *Local evidence.* Anchoring is judged per pathway mention, in its sentence or a
  12-token window. `pathway_head` never anchors; the framing vocabulary
  (`mechanism`/`inhibition`/`flux`/bare `-ase`) is excluded from the strong set, as are
  `biosynthesis`/`biosynthetic` (which sit inside the aliases and let them self-certify);
  and in a document that announces itself as a screen, a pathway-specific term anchors only
  if the pathway is named in the title. Every decision carries a `classification`:
  `mechanistic` / `context_only` / `omics_only` / `off_topic`.
* *Filling the count.* `fetch_papers` escalates the search per topic until the requested
  count is filled, the source runs dry, or `eligibility_candidate_ceiling` candidates have
  been examined, and reports the funnel (requested / examined / eligible / ineligible /
  duplicate / no_full_text / accepted, per topic and overall, with every short topic and
  its stop reason) through a `stats` out-param that the runner logs and stores at
  `plan["eligibility"]["acquisition"]`.
* *Persisted input.* `screening_input` stores the title, the abstract bounded to 4000
  chars, and the SHA-256 of the full abstract, so a `skipped.json` rejection replays
  offline. The dry-run tool prefers the plan's stored abstract automatically, falls back to
  a bounded lead window recovered from `01_source_text.txt` for legacy plans, and marks
  anything short of a publisher abstract `provisional`.

**Measured on the real 2026-07-28_2122 plan** (17 labelled papers: the 6 known junk papers
and the 5 cholesterol false-positive shapes as negatives, the 5 genuine Lpx/Men/PPOX/Ent
reaction papers plus the enterobactin review as positives): with cached abstracts,
precision 1.00, recall 1.00, 7 of 28 accepted. Title-only: precision 1.00, recall 0.33,
with all four misses flagged for manual review — the contextual rule cannot confirm
mechanism from a title, which is what `provisional` has always meant.

**Pipeline consistency:** every correction tightens or instruments an existing seam rather
than adding one. The gate still sits entirely in the acquisition layer, `eligibility` still
imports nothing but `t2pw.config`, and the Stage-0 boundary reads the app's own session
state without touching the app.

---

## Only plausibly-mechanistic papers enter the pipeline; requested scope stops being stamped as observed (2026-07-29, branch `research-mode`)

Full rules in `docs/paper_eligibility.md`; this entry is the error/why/fix record.

**Error.** Two separate failures with one shared consequence. Run 2026-07-28_2122 planned 28
papers and ran each of them twice. Six of the 28 were papers no pathway extractor could ever
have succeeded on: a Fournier's-gangrene case report (`PMC12971581`, fetched for *lipid A
biosynthesis in E. coli*), a river-resistome surveillance study (`PMC13139079`), two poultry/turkey
ESBL virulence surveys (`PMC12649316`, `PMC12737783`, fetched for *enterobactin biosynthesis*), a
COVID-19 lncRNA comorbidity study (`PMC12797059`) and a gene-set-evolution tool (`PMC12898691`).
Each cost a full-text download plus two app runs, and then arrived in the morning triage filed as
an *extraction failure* — so debugging went into the extractor for papers that were never about
the requested pathway. Separately, every `CandidatePaper` in the plan carried
`organism: "Escherichia coli"` (or `Homo sapiens`, …) whether or not the paper had anything to do
with that organism — including the Fournier's case report, which is *Ochrobactrum anthropi*.

**Why.**

1. *Retrievable full text was the only admission criterion.* `batch/fetch.fetch_papers` took every
   candidate a topic query returned, fetched its text, and planned it. There was no cheap
   metadata gate anywhere before the expensive work.
2. *Requested metadata was written into observed fields.* Every acquire fetcher passed
   `organism=organism` — the organism the *search asked for* — onto the candidate, and
   `_as_batch_paper` did `candidate.organism or spec.organism`. Downstream, a stamped organism is
   indistinguishable from a reported one, so the single organism check that did exist
   (`select._organism_score`) was pinned at 1.0 and could never fire. "We asked for E. coli" and
   "this paper is about E. coli" were the same field.
3. *`ineligible_*` was an unrecognised status.* `report._norm_status` mapped anything it did not
   recognise to `STATUS_UNKNOWN`, and `_is_failure` counts `STATUS_UNKNOWN` as a failure — so even
   an explicitly screened-out paper would have been scored as a defect.

**Fix.**

* New `src/t2pw/rag/eligibility.py`: a deterministic, offline title/abstract scorer (fixed
  lexicons, word-boundary regexes, arithmetic — no network, no LLM, no clock). Positive evidence:
  pathway aliases, expected enzyme/metabolite terms, reaction/mechanism language, organism/taxonomy
  match, reconstruction and enzyme-characterization language. Negative evidence: incompatible
  organism, clinical case report, epidemiology/prevalence survey, animal-virulence survey for an
  unrelated pathway, software-only, pathway named only in background, no mechanistic pathway terms.
  Eight explicit outcomes. Thresholds are `RAG_ELIGIBILITY_*` config, defaulted in `RAG_DEFAULTS`.
* Eligibility requires a **pathway anchor** — a pathway-name alias or a pathway-specific
  enzyme/metabolite. The bare head compound scores but does not anchor: naming "cholesterol" is not
  evidence a paper is about cholesterol *biosynthesis*, and treating it as an anchor admitted four
  cholesterol-signalling and cholesterol-in-cancer papers.
* `RequestedScope` is a frozen dataclass and `apply_stage0_observation` returns it untouched, with
  Stage 0's reading landing in `ObservedContext` (`observed_pathways` / `observed_organisms` /
  `aliases` / `ambiguities`). A strong Stage-0 contradiction yields `conflicts` for the caller to
  act on; it can no longer re-point the batch at whatever a paper turned out to be about.
* `fetch_papers` runs the gate **before** `fetch_full_text`, so a rejected paper costs one keyword
  scan. `CandidatePaper` and `BatchPaper` gained `requested_pathway` / `requested_organism` /
  `observed_pathways` / `observed_organisms` / `organism_match`; `organism` now means only "what the
  paper reports" and no code path backfills it from the request. Pinned ids bypass the score
  (`pinned_override`) but still record every mismatch as a warning.
* Screened papers become `skipped.json` records with their eligibility report and get no paper
  folder and no manifest row, so they cannot reach the triage. Belt and braces:
  `report.STATUS_INELIGIBLE` folds the `ineligible_*` spellings out of `_is_failure` and into
  `incomplete`, and the summary counts them on their own line.
* `plan.json` persists the per-paper eligibility report and a top-level block with the thresholds
  the run was screened with, the per-outcome tally and the manual-inspection list.
* `scripts/eligibility_dry_run.py` re-screens a stored plan offline and reports what would be
  accepted or rejected, marking title-only verdicts provisional.

**Pipeline consistency:** the gate sits entirely in the acquisition layer, before Stage 0 —
it changes which papers reach the pipeline, never what the pipeline does with one. The dependency
arrow still points RAG → core only (`eligibility` imports nothing but `t2pw.config`), and the
requested/observed split makes the existing `select._organism_score` check meaningful for the
first time rather than replacing it.

---

## One authoritative protein export policy: verify real identities, place the rest honestly (2026-07-29, branch `research-mode`)

Full policy in `docs/protein_export_policy.md`; this entry is the error/why/fix record.

**Error.** Two opposite failures shipped from the same seam. Run 2026-07-28_0919 exported
`PhoP` as NAD+, `pmrHFIJKLM operon` as the lactose operon repressor and `mcr genes` as the human
mineralocorticoid receptor, every one of them with `resolution.status == "matched"` and zero gate
errors. Run 2026-07-28_2122 failed all 16 strict legs, 7 of them at the post-pipeline gates,
largely on enzymes the papers state clearly and no database has ever heard of. Between them,
`entities.proteins` was accumulating cofactors: 8 of that night's 27 distinct post-gate issue
codes are `coenzyme A (CoA)` or `succinyl-coenzyme A (ScoA)` filed as a protein.

**Why.** Three independent causes, each with the same shape — a check that measured the wrong
thing and reported success.

1. *No identity check on a real accession.* `map_payload` resolved an ambiguous candidate list by
   taking `next(c for c in candidates if c.get("uniprot"))` — literally list order, recorded as
   `chosen_rule: "ambiguous_first_candidate"` and `status: "matched"`. Every other real-ID route
   (cached hit, pre-existing id from the first mapping pass, second-pass promotion from row
   metadata, the Phase-2 UniProt fallback, `resolve_mapping_gaps`) had at most the name gate, and
   two had nothing at all.
2. *`PROTEIN_LIKE_RE` matches `enzyme`, which is inside `coenzyme`.* `_is_protein_like` therefore
   answered yes for `coenzyme A (CoA)` even when the payload's own compound registry said
   otherwise, and `promote_catalysts` moved it into `entities.proteins`, where it can never
   acquire a UniProt id and the identity gate fires forever.
3. *The prune/gate contradiction.* `prune_disconnected_proteins` and
   `drop_process_orphan_proteins` spared a degree-0 protein that carried an external identifier;
   `run_strict_post_normalization_gates` then rejected it for being degree-0. Sparing it never
   saved it — four of PMC12444477's strict-leg errors existed for no other reason.

**Fix.** One authority, `map_ids.verify_real_protein_identity`, with six recorded rungs — entity
type, species/taxon, name/alias/gene, identifier resolution, minimum score (0.5), margin over
rivals (0.1) — applied at *every* route that can write a real accession onto a protein row.
`ambiguous_first_candidate` is deleted: a candidate list is resolved on evidence or not at all,
and a lone survivor with sufficient margin is `ambiguous_verified_single_candidate`.

An actor that fails verification but has a usable functional name and direct role evidence is not
dropped — it goes to the strengthened `_apply_pathbank_unknown_enzyme_fallback`, which preserves
the functional name on a generated `protein_complex` backed by PathBank Unknown 9659, stamps
`identity_status="placeholder"` with the evidence and the real reason mapping failed, and is
counted separately from `verified_real_proteins` so a placeholder can never be read as a real
UniProt match. An actor with *no* role evidence — an inhibitor parked in `enzymes`, a cofactor in
the actor list — has its process claim quarantined instead; wrapping an unsupported claim would
manufacture an enzyme the paper never stated.

`entity_identity.compound_name_block_rule` fixes the promotion, keyed off the payload's own
compound registry first and a carrier-moiety name shape second, both deferring to an enzyme
head-noun check so `succinyl-CoA` is a metabolite and `succinyl-CoA synthetase` is not. Strict
mode now quarantines an unused protein whether or not it is mapped (into
`payload["quarantined_proteins"]`, with what it carried); research mode retains and flags it, and
both passes run in both modes so the research census exists at all.

**Measured, offline, on the compact fixtures.** `tests/fixtures/baseline_2026_07_28` gate census
is byte-identical before and after except `disconnected_mapped_protein`, which goes from 1 gate
error ("Protein has degree 0 after normalization: FabA") to 0 with FabA quarantined and its
verified identity recorded. On the cofactor payload: 5 gate errors before, 1 after — the four
`coenzyme A`/`succinyl-coenzyme A` errors are gone because the rows never leave
`entities.compounds`, and the one that remains is the genuine missing identity of `ALAS`. On the
ambiguous payload: `O34362` (*Bacillus subtilis*) shipped before for a *Camellia sinensis*
pathway as `status: matched`; after, it is refused and the actor points at a functional complex
named `MenD` backed by the sentinel.

### Correction, same day: the ladder was failing *open*

The first cut of the ladder treated "nothing to judge on" as a pass at four
separate rungs, which meant an accession with no candidate row, no organism, no
resolved name and no score verified cleanly — the exact shape of `mcr genes` ->
P08235. It now fails closed with `identity_evidence_missing` and routes to the
placeholder, so nothing is dropped and nothing is claimed. Four further
corrections landed with it:

* **Species.** Blanket same-genus agreement passed `Escherichia coli` for
  `Escherichia fergusonii` and `Bacillus subtilis` for `Bacillus cereus`.
  Agreement now requires a taxonomy id, the same binomial, or a strain
  qualification of it at a word boundary. Genus-level acceptance survives only
  when the *request* was genus-level and is recorded as `genus_level`, never as
  an exact species match.
* **PathBank.** The blanket name exemption is gone: it fired before any other
  evidence was weighed and returned `skip`, which a fail-closed ladder cannot
  distinguish from a pass, so curated provenance alone was shipping accessions.
  A PathBank row now proves itself on a name token, an exact gene symbol or an
  audited alias; it keeps only the *score* waiver, because that provider supplies
  no score. Cost, accepted and recorded: three matches of the `LpxL` -> P0ACV0
  shape (entity is the modern symbol, PathBank stores `htrB`) were right and now
  become placeholders.
* **Score attribution.** `_candidate_score` took `max(candidate, result)`, so on
  the second mapping pass a weak shipped accession borrowed the confidence of the
  *different* accession the resolver had just chosen. Result-level confidence is
  now read only when `result.mapped_ids` identifies the same candidate.
* **Role support.** Canonical membership in `enzymes`/`transporters` still
  authorizes the placeholder, but it is recorded as
  `role_basis: canonical_actor_membership` with a separate
  `direct_evidence_present` flag, and the raw evidence text is replaced by a
  bounded `evidence_digest` (200-char excerpt + sha256 + length). Calling
  collection membership "evidence", or copying a 139,576-character flattened
  corpus into `mapping_meta`, were both the same dishonesty in miniature.

Ten test fixtures across six files were given the candidate rows a real resolver
returns (organism, name, score) rather than the bare `{"uniprot": "..."}` they
carried; under a fail-closed ladder those stubs were asserting that unverifiable
evidence verifies. One pre-existing flake was fixed on the way:
`test_query_ladder_never_issues_a_bare_operon_query` reaches
`map_protein_uniprot`'s third tier, which calls the LLM over the network, and its
answer sometimes contained the very `operon` token the test forbids.

**Contracts deliberately changed**, each because it pinned the defect rather than the design:
`test_prune_disconnected_proteins_keeps_identified_proteins` (mapped orphans are now quarantined,
not spared for the gate to reject), `test_normalize_process_payload_returns_gate_failure_in_report`
(the orphan is resolved before the gate), and
`test_strict_mode_still_enforces_connectivity_on_the_same_payload` (same reversal, renamed). The
shared best-effort fixture in `test_stage2_mapping_boundary.py` moved 0.42 -> 0.62 so it stays in
the band those tests are about — below the 0.78 low-confidence line, above the 0.5 identity
floor — with the other side of the floor pinned separately.

---

## Pre-run readiness: the scope filter proved live end to end, and the night's arithmetic (2026-07-28, branch `research-mode`)

Written in the last hour before the overnight corpus run, and deliberately narrow. **No new
production fix landed with this entry.** It does three things: it re-proves, by execution, the
out-of-scope regression that the entry below closed — at the coordinates the code has *now*,
which are not the ones that entry cites; it states plainly the half of that problem the fix does
**not** touch; and it records what a read-only readiness audit found about whether the night is
worth starting at all. The verdict on the night is **go, with a `--limit` and two manual steps**
(§C, §D). The one thing that would have wasted the night silently — a corpus that cannot fit in
it — is arithmetic, not a bug, and is written down here so nobody rediscovers it at 3 a.m.

### A. The out-of-scope reaction filter: a regression from 2026-07-14, closed, re-verified, and still only half the problem

**Error.** The 2026-07-14 entry *Tighten default reaction scope and wire the out-of-scope
reaction filter* (this file) shipped `filter_out_of_scope_reactions` (`pipeline.py:247`), wired
between Stage 1 and Stage 2 (`streamlit_app.py:3617`), and it **never removed a reaction in its
life**. Measured over everything this project has ever delivered: **21 payload files under
`runs/` — 9 `stage1_payload.json`, 9 `merged_payload.json`, 3 `final_mapped.json`, 178 reactions
in total — and `scope_membership` present on exactly zero of them.** Including the reference run:
`PMC12444477/strict` carries 9 Stage-1 and 27 merged reactions, none of them labelled.

**Why.** `_clean_processes` (`pipeline.py:1929`) rebuilds every reaction from a key allowlist that
did not name `scope_membership`, and both Stage-1 branches hand the orchestrator `clean_stage_one`
output (`pipeline.py:2273`, `:2312` / `:2321`, `:2327`). The label was erased between the model
writing it and the filter reading it, and the old `rxn.get("scope_membership", "core")` then
reported `core` for the entire corpus. This is the same failure shape as the two fixes in the
entry below — a guard that is green in its own tests and inert in production because the stage
*before* it removes the thing it reads — and it is the oldest instance of it we have found.

**Fix** (landed with the entry below; its §A carries the design rationale, this is the
verification). `_carry_scope_membership` (`pipeline.py:1850`, body at `:1923-1925`) carries a
non-empty string label through the rebuild, called from the reaction branch of `_clean_processes`
at `:1982`. The filter's comparison (`pipeline.py:315`) is now
`isinstance(scope, str) and scope.strip().casefold() == "out_of_scope"`, which is what
`reaction_lock_manifest._scope_membership` (`:59-66`) does at `:186` and `:228`.

**Measured today, by execution**, on one raw Stage-1-shaped payload of six reactions labelled
`core` / `out_of_scope` / `OUT_OF_SCOPE` / `"  out_of_scope  "` / absent / `17`, run through the
real `clean_stage_one` before the filter sees it:

| path | labels the filter sees | it removes |
| --- | --- | --- |
| raw payload, no cleaning | all six as the model wrote them | `glucose phosphorylation`, `shouty`, `padded` |
| `clean_stage_one` with the carrier neutralised (i.e. the pre-fix allowlist) | `<ABSENT>` ×6; key union `('evidence', 'inputs', 'name', 'outputs')` | **nothing** |
| `clean_stage_one` as it ships now | `core`, `out_of_scope`, `OUT_OF_SCOPE`, `out_of_scope`, `<ABSENT>`, `<ABSENT>`; key union gains `scope_membership` | `glucose phosphorylation`, `shouty`, `padded` |

The unlabelled row and the `17` row survive all three passes: absent, empty and non-string mean
**keep**, deliberately (the entry below argues why; the short version is that a missing label is
evidence nobody classified the reaction, not evidence it is off-pathway). The label is stored
verbatim and case-folded only by the readers, so `OUT_OF_SCOPE` is removed by the filter *and*
refused a lock by the manifest instead of one of each.

**Effect on everything already on disk: exactly none**, and that is the honest headline. All 21
delivered payload files, all 178 reactions, zero labelled, zero removed by the now-live filter.
The change is **forward-only** — it can bite only on Stage-1 output produced after it landed,
where the model actually writes the label — so no delivered artifact changes and no previously
passing paper becomes a failing one.

**What it does not cover, plainly: cross-paper RAG imports, which remain open and need design.**
The filter runs once, between Stage 1 and Stage 2. Everything RAG imports arrives later, at the
S3 merge, and is therefore invisible to it. Measured on the reference run: `merged_payload.json`
holds 27 reactions and only **9** of them carry a `locked_reaction_id`, which is exactly the 9
reactions of its `stage1_payload.json` (all 9 were locked). The other 18 rows were not in the
payload the filter saw. Worse, a RAG row could not carry a verdict even if the filter ran
again: `rag/synthesize.py:1133` asserts `set(row) <= _ALLOWED_ROW_KEYS`, and that frozenset
(`:1136-1138`) is `{name, inputs, outputs, enzymes, entity, entity_type, role, source_refs}` plus
`RAG_ADDITIVE_KEYS`, with no scope field in it. So reactions 10–26 of the lipid A payload —
phospholipid biosynthesis, a different pathway — are untouched by this fix and stay in the debt
list below. **Seed-paper Stage-1 reactions only.** Anyone reading "the out-of-scope filter works
now" as "off-pathway reactions no longer ship" would be wrong twice over: unlabelled off-pathway
reactions still ship by design, and imported ones are never offered to the filter at all.

**Tests.** `tests/test_scope_membership_filter_end_to_end.py` (7 tests) is the first coverage that
puts **real `clean_stage_one` output** through the filter rather than a hand-built payload that
already carries the label — the trap that let this survive from 2026-07-14 to today:
`test_filter_removes_a_labelled_reaction_from_real_clean_stage_one_output`,
`test_label_survives_the_chunked_merge_path_too`,
`test_an_unlabelled_reaction_is_kept_because_absent_is_not_out_of_scope`,
`test_a_non_string_label_is_dropped_and_the_reaction_is_kept`,
`test_filter_and_lock_manifest_agree_on_case_and_whitespace_variants`,
`test_out_of_scope_reactions_no_longer_ship_unlocked`,
`test_unlabelled_rows_gain_no_key_and_other_process_buckets_are_untouched`. Re-run today together
with the two files that previously claimed to cover this area
(`tests/test_pipeline_cleanup.py`, `tests/test_streamlit_stage2_orchestration.py`): **26 passed in
1.20s**. The round that landed the fix measured the suite at **1067 → 1074**; it collects **1089**
as this is written, because other work landed in parallel today, so 1089 is not this fix's number
and is recorded only so the next reader knows which baseline they are looking at.

### B. Where the entry below had already drifted from the code

The entry immediately below was saved at 20:27, `pipeline.py` was last edited at 20:24 and
`rag/synthesize.py` at 20:05, and several of its citations describe a revision that no longer
exists. Corrected in place rather than left to rot, since a changelog nobody can navigate by is
the thing this file exists to prevent:

- `_clean_processes` `:1926` → **`:1929`**; the carrier call `:1979` → **`:1982`**;
  `clean_stage_one` call sites `:2270`, `:2309`/`:2318`, `:2324` → **`:2273`, `:2312`/`:2321`,
  `:2327`**; `write_stage1_lock_artifacts` `:2269`/`:2315` → **`:2272`/`:2318`**. Three lines of
  drift, in the one paragraph a reader would use to check the claim.
- `reaction_lock_manifest._scope_membership` is read at `:186` **and `:228`** (the debt round said
  `:229`), and the manifest writes the model's own spelling at `:257`.
- `_ALLOWED_ROW_KEYS` is asserted at `rag/synthesize.py:1133` and defined at **`:1136-1138`**. The
  debt entry below says `:1122-1125` and a working note from the same round said `:1212-1215`;
  both were true of different revisions of that file today. Corrected wherever it is cited.
- **The two corpus counts are both right and were measuring different sets.** 18 files / 130
  reactions is the Stage-1-plus-merged set; 21 files / 178 reactions is that set plus the three
  `final_mapped.json`. Zero reactions carry `scope_membership` under either reading. Both are now
  spelled out where they appear so they stop looking like a contradiction.

Nothing in the substance of that entry was wrong. Every behavioural claim of its §A re-verified
by execution here; its §B merge measurements (the 279 → 220 entity-row table) were **not** re-run
this pass and are recorded on that round's authority, not on this one's.

### C. The night does not fit in a night — plan for `--limit`, not for a finished corpus

Measured from `runs/2026-07-28_0919/manifest.jsonl` plus the one leg missing from it (§D), six legs
actually executed between 09:19:08 and 11:49:05:

| leg | seconds | outcome |
| --- | --- | --- |
| PMC12444477 strict | 2988.18 | pass |
| PMC12444477 research | 2777.82 | fail (contract, post-pipeline) |
| PMC13278307 strict | 2098.21 | pass |
| PMC13231680 research | 936.70 | pass — not in the manifest, see §D |
| PMC13278307 research | 137.14 | fail at the Stage-1 boundary |
| PMC13231680 strict | 54.70 | fail at the Stage-1 boundary |

The four legs that ran the whole pipeline mean **2200.2s**; the three big ones mean **2621.4s**;
all six mean **1498.8s**. Those six sum to 8992.75s inside an elapsed 8997s, so **~4 seconds of
that 2.5-hour run was not model time** — there is nothing to tune outside the LLM calls.

Against a 10-hour budget that is **13.7 legs** at the big-leg rate, **16.4** at the
full-pipeline rate, and **24** if the night is lucky enough to include cheap failures. The
now-concurrent enrichment (`enrich_entities.py`: 139.7s serial → 29.1s concurrent, 4.81×, 110.7s
saved on one leg) moves those to 14.3 / 17.2 / 25.9 — a **4–7% cut of a 37–50 minute leg**, and it
was measured with all 68 lookups cold, so against the warm cache it will be less. That fix is
worth having on its own axis; it is not the night's bottleneck and must not be sold as throughput.

`topics.txt` asks for 6+5+6+5+6 = **28 papers × 2 modes = 56 legs**, and a resume of
`2026-07-28_0919` owes **51** (the run's own last log line says so). One night therefore covers
roughly **a quarter to a half of the ceiling**. The honest figure is softer than that ratio makes
it sound, because `topics.txt` records its own observed full-text yield as "roughly half" — so the
realistic corpus is nearer 14 papers / 28 legs, i.e. **two nights, not four**. Either way it is
not one night. Recommendation: cap the night with `--limit` rather than letting the deadline
truncate the work mid-corpus. `--limit 7` (7 papers = 14 legs) sits exactly on the measured
capacity — it fits inside 10h at the full-pipeline mean of 2200.2s (16.4 legs) and overruns it by
about a leg at the big-leg mean of 2621.4s (13.7), which is the right side to err on, since the
deadline truncates cleanly and a truncated plan resumes.

Two deadline facts that decide when to walk away from the keyboard: the budget is checked **before
each leg** (`runner.py:2037-2042`), and the per-leg `--timeout` is an hour, so a leg starting at
9h59m runs to 10h59m — **budget 11 hours of wall clock for a 10-hour night**. And `started_clock`
is set on entry to `_run_batch` (`runner.py:1960`), so the deadline is **per launch, not per run**:
every relaunch of a resumed run gets a fresh 10 hours.

### D. A leg that finished, passed, and is not in the manifest

`runs/2026-07-28_0919/papers/PMC13231680__mechanistic-insights-into-phthalylsulfacetamide/research/`
holds a complete artifact set written at 11:49 — `RESULT.txt` reading `RESULT: PASS (with
warnings)`, `stage=research_report`, 936.7s, 2 citations, plus `merged_payload.json`,
`research_pathway_report.txt`, `research_pathway_citations.json`,
`research_pathway_elements.csv`, `review_flags.json`. `batch.log` shows that leg starting at
11:33:27 and the parent taking Ctrl+C at 11:49:05 — the same minute the artifacts were written.
The child ran to completion and wrote everything it owes; the parent died before
`append_manifest` (`runner.py:2096`) recorded it, i.e. somewhere in the window that opens when the
child exits and closes at that write, with `parse_child_output` (`runner.py:2066`) in between.
Consequences, all live right now:

- a resume re-runs that leg and pays its ~937s again, overwriting good artifacts;
- `SUMMARY.txt` says `papers attempted : 3` and `manifest rows : 5`, when six legs ran and
  **three** of them passed. Every count derived from the manifest understates the run by one
  pass — and specifically by that run's only research-mode pass, so `research relaxed : 0 pass
  / 2 fail` in the summary is wrong in the one direction that matters when triaging research
  mode as a defect class.

Not fixed — the window is inherent to "run child, then record", and closing it properly means
writing a provisional row before the child starts and reconciling after, which touches the
child→parent protocol the whole night depends on. Recorded in the debt list. The operator-side
workaround for tonight is to accept the re-run, or copy that row into `manifest.jsonl` by hand.

### E. The preflight, re-run

`runner.run_preflight()` returns **0 in 1.39s** under `.venv\Scripts\python.exe` with no problems
and no warnings, including the `t2pw.llm.client` entry added to `CHILD_IMPORTS` in the entry below
(`runner.py:1274-1279`) — so the key is present and well-formed, and the three import-time
`RuntimeError`s in `client.py` (`:38`, `:40`, `:63`) are all cleared before the night starts. Two
limits worth stating precisely rather than trusting: the probe proves the module *imports*, which
covers the key's presence and shape but not that `openrouter.ai` answers, that the key is unexpired
and funded, or that the account can serve the model; and the model id itself is resolved lazily, so
a missing `OPENROUTER_MODEL` raises at `client.py:261` on the **first call of the night**, not in
the preflight. It is set (`google/gemma-4-26b-a4b-it`), so that is a limit of the check, not a
finding against tonight.

---

## Pre-run close-out: two half-landed fixes wired, one blind spot in the preflight (2026-07-28, branch `research-mode`)

Written while preparing the next overnight corpus run. The entry below it closed the
debt round; this one closes what that round left in the state where the code was right
and **nothing called it that way**. Two of the Known Debt bullets further down described
defects that no longer exist and have been rewritten.

The theme is worth naming, because it happened twice in one day and once before that on
2026-07-14: a guard that takes an optional argument, or reads a field that something
upstream strips, is *green in its own tests and inert in production*. Neither of the two
fixes below could have been caught by a unit test of the module it lives in — one needed
the caller, the other needed the stage that runs before it.

### A. The out-of-scope reaction filter can fire, for the first time since 2026-07-14

**Error.** `filter_out_of_scope_reactions` (`pipeline.py:247`) is wired and is called
between Stage 1 and Stage 2 (`streamlit_app.py:3617`). It had never removed a reaction.

**Why.** `pwml_system.txt` (`:6`, `:15-16`) requires the model to label every reaction
`core | anaplerotic | cataplerotic | auxiliary | out_of_scope` and promises out-of-scope
ones "are removed from the payload before downstream stages run". Both Stage-1 branches
hand the orchestrator `clean_stage_one` output (`pipeline.py:2273`, `:2312`/`:2321`,
`:2327`), and `_clean_processes` (`:1929`) rebuilds every reaction from a key allowlist
that never named `scope_membership`. The label was erased before the filter could read
it, and the old `rxn.get("scope_membership", "core")` then reported `core` for
everything. The 2026-07-14 fix was wired on the wrong side of the allowlist and has been
inert since the day it landed.

The erased label was not only a dead filter. `write_stage1_lock_artifacts` runs on the
**raw** payload (`pipeline.py:2272` / `:2318`), *before* `clean_stage_one`, so
`reaction_lock_manifest` (`:186`, `:228`) does see the label: it refuses to list an
out-of-scope reaction and refuses to stamp it with a `locked_reaction_id`. Out-of-scope
reactions therefore shipped in the payload as **unlocked** ones — kept by the payload,
disowned by the lock manifest.

**Fix.** `_carry_scope_membership` (`pipeline.py:1850`), called from the reaction branch
of `_clean_processes` (`:1982`), carries a non-empty string label through the rebuild,
stripped and **verbatim** — never case-folded, because the manifest records the model's
own spelling in its own `scope_membership` field (`reaction_lock_manifest.py:257`) and
the two artifacts must not disagree about what the model said. Case-insensitivity moved
into the readers instead: `filter_out_of_scope_reactions` now tests
`isinstance(scope, str) and scope.strip().casefold() == "out_of_scope"`, matching
`reaction_lock_manifest._scope_membership` (`:59-66`). Without that, `"OUT_OF_SCOPE"`
would be refused a lock by the manifest and kept by the filter — the same
payload/manifest split, in a different spelling. Reactions only: transports,
reaction-coupled transports and interactions are not labelled by the prompt and the
filter never reads them. The key is appended last, so an unlabelled reaction keeps its
key order byte-for-byte.

**An absent label means KEEP**, and that is a decision, not an oversight. A missing
label is evidence that nobody classified the reaction, not evidence that it is
off-pathway, and three real sources produce unlabelled reactions: every payload written
before this fix; any model that ignores the instruction; and every reaction added
*after* this call runs — Stage-2 additions and RAG imports come from prompts and
adapters that cannot emit the field at all (`rag/synthesize.py`'s `_ALLOWED_ROW_KEYS`).
Keeping makes the failure mode "an out-of-scope reaction survives", which every
downstream gate still inspects, instead of "an in-scope reaction vanishes", which
nothing downstream can detect.

**The corpus before/after the debt item demanded cannot be produced retroactively.**
Measured over every delivered Stage-1 and merged artifact under `runs/` — 18 files,
130 reactions, or 21 files and 178 reactions if the three `final_mapped.json` are counted
too — `scope_membership` is present on **zero** of them, so the filter would
remove nothing from any of them and the record of what the model actually labelled was
destroyed by the defect before anything was written. The check the debt item asked for
is therefore only obtainable from a run made *after* this fix. See the debt list for
why that run will not record it either.

### B. The RAG merge no longer re-imports the seed paper's own entities

**Error.** `runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict/merged_payload.json`
delivers `entities.proteins` with 31 rows for 22 distinct normalized names — all nine
seed enzymes doubled — and `entities.compounds` with 56 rows for 43.

**Why.** RAG synthesis rebuilds an entity row for every participant of every reaction it
resolved, and it resolves the **seed's** reactions too (the seed paper is itself indexed,
so its own claims can corroborate). `merge_additions` dedupes with `_extend_unique`
(`pipeline.py:2584`), whose signature is `json.dumps(row, sort_keys=True)`, so
`{"name": "LpxA", "class": "protein", "confidence": 1.0, ...}` and
`{"name": "LpxA", "rag_provenance": {"source_id": "seed_paper"}, ...}` are two different
signatures for one protein and both survive. The names are byte-identical, which is
exactly why the 2026-07-23 synonym resolver — which collapses *synonyms* — correctly
never touched them.

**Fix, part 1 (the guard).** `conform_rag_additions_for_merge` (`rag/conform.py:200`)
takes the payload the envelope is about to be merged into and drops any entity row whose
name that base already registers. Identity is the registry gate's own
(`process_normalizer._normalize` / `_entity_name_norms`, imported rather than
re-implemented) over exactly the buckets `validate_registry_references` unions, so
"already registered" means what the gate means by it. The set grows as rows are
accepted, so one pass also dedupes the envelope against itself. Deliberately **not** in
synthesis: `synthesize_with_report` returns a standalone payload whose entity buckets
must cover its own reactions and whose seed rows carry the citations the pre-merge
report reads — suppressing them there strands references and deletes the corroboration
the seed is indexed to provide (`tests/test_rag_synthesize.py` pins both).

**Fix, part 2 (the wiring), which is the half that makes it real.** The guard landed
with `streamlit_app.py` still calling `conform_rag_additions_for_merge(rag_result.payload)`
— one argument, `seed_payload` defaulting to `None`, which is exactly the pre-fix
behaviour. Every unit test passed the base explicitly, so the whole suite was green over
a production path that had not changed at all. The call site now passes `final_payload`
(`streamlit_app.py:3741`) — Stage 1 **plus Stage 2**, the very object handed to
`merge_additions` on the next line, not the Stage-1 seed synthesis was given. That
distinction is load-bearing: `lipid IV_A precursor`, `Kdo-lipid A precursor` and
`tetra-acylated disaccharide intermediate` are Stage-1 reaction participants that Stage 1
never registered and Stage 2 did, invisible to synthesis and duplicated all the same.
`tests/test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform`
now pins the call site by AST, asserting the name given to `conform` is the same name
given to `merge_additions` as its base.

**Measured**, by splitting each delivered `merged_payload.json` back into base (rows with
no `rag_provenance`) and RAG side and re-running `conform` → `merge_additions` both ways,
after first asserting the old call shape reproduces the delivered file row-for-row:

| payload | rows before | rows after |
| --- | --- | --- |
| PMC12444477 strict | 88 (proteins 31, compounds 56, complexes 1) | 66 (22 / 43 / 1) |
| PMC12444477 research | 89 | 68 |
| PMC13278307 strict | 35 | 31 |
| PMC12312563 research / strict | 22 / 17 | 17 / 14 |
| PMC13231680 research / strict / research | 8 / 9 / 11 | 7 / 8 / 9 |
| **all 8 payloads with a RAG side** | **279** | **220** |

Zero duplicate normalized names remain, no reaction is lost on any payload (10, 7, 3, 4,
24, 27, 4, 14 before and after), and `validate_registry_references` returns the identical
error set on all eight — the fix introduces no "unknown entity" error anywhere.

**A coordinate note, because two different counts of the same file are in circulation.**
The reference payload's compounds are "41 distinct" under an alnum-only lowercase
normalizer and **43** under `process_normalizer._normalize`, which is the one this guard
uses. The two extra pairs `_normalize` keeps apart are `lipid IV A` / `lipid IV_A` and
`sn -glycerol 3-phosphate` / `sn-glycerol 3-phosphate` — spacing and underscore variants
of the same species. They stay as two rows each. That is deliberate: this guard
under-merges rather than over-merges, and collapsing spelling variants is the synonym
resolver's job, not its.

### C. The preflight was blind to the LLM backend

**Error.** `check_preflight` returns ok on an interpreter with no usable
`OPENROUTER_API_KEY`, and the night then dies 56 times after the fetch — the exact shape
the preflight was built to stop.

**Why.** `CHILD_IMPORTS` was derived from `driver.py`'s deferred imports, and
`t2pw.llm.client` is not one of them. Verified by execution: importing all four listed
modules in a fresh interpreter leaves `t2pw.llm.client` absent from `sys.modules`. It is
also the module in this codebase that raises at **import** time — `client.py:38` (no
key), `:41` (a key not starting with `sk-or-`), `:63` (`LLM_PROVIDER` neither `local`
nor `openrouter`) — while every stage of every leg calls `chat()`.

**Fix.** One entry added to `CHILD_IMPORTS` (`runner.py`). The probe catches
`BaseException`, so a `RuntimeError` at import is reported to the operator exactly like
a missing module, and importing the client does no network I/O — it constructs an
`OpenAI` object and stops. Cost: the probe goes from 0.72s to **1.26s**, once per night.
`tests/test_batch_preflight.py::test_the_llm_backend_is_probed_even_though_nothing_else_reaches_it`
asserts it by name rather than by coverage, so a refactor that incidentally imports the
client cannot delete the operator's reason for trusting the check.

**What a green preflight still does not promise:** an `openrouter.ai` that is reachable,
a key that is not expired or revoked, credit that is not exhausted, or a model id the
account can serve. All four need a live one-token call, which is slower and not free and
belongs behind its own flag, not inside a 1.26s import probe.

### D. Resume semantics documented, not changed

`completed_pairs` (`runner.py:404`) builds its done-set from any manifest row and never
looks at `status`, so a **failed** leg is as done as a passing one and a resume will not
revisit it. `runs/2026-07-28_0919` ended with five rows, three of them `fail`; relaunching
it starts at leg 6 with 51 pending and never touches those three again. This matters
right now, because §E of the entry below (the empty-completion retry) was written for two
of those exact legs and a resume will not exercise it on them.

Deliberately left as it is. The alternative — retry anything that failed — makes a
deterministically failing paper (a clinical case report with no pathway in it) cost the
night its full timeout on every relaunch. `docs/batch_runner.md` §8 now says so
explicitly, with the two ways to force a retry: `--fresh`, or delete the failed rows from
`manifest.jsonl`.

---

## Clearing the debt from `runs/2026-07-28_0919` (2026-07-28, branch `research-mode`)

The entry below this one closed four defects and then listed **twelve** items of known
debt found in the same run. This round closes six of those twelve outright — the
referential cascade on entity removal, the Stage-0 Case-C clobber, empty-but-successful
LLM completions, the missing batch preflight, serial enrichment, and the cross-check
that entry left explicitly outstanding. It closes half of two more: entity-type
confusion in one of its two directions, and per-reaction provenance, which is the
precondition for any work on scope creep. And it refutes the premise of a third:
`src/t2pw/stoich/` was nominated to catch reversed reactions and structurally cannot.

The cross-check produced the worst news here. **The out-of-scope reaction filter
shipped on 2026-07-14 is wired, is called, and cannot fire.** It has been inert since
the day it landed. That was written up as a regression in the debt section below, which
has been rewritten so that nothing fixed in this entry is still listed as open and
nothing found this round is missing from it. (It was closed later the same day — see §A
of the pre-run close-out entry above — so the debt bullet is gone and what replaced it
is the fact that a live filter's removals are not recorded anywhere.)

One coordinate note that applies throughout: `streamlit_app.py` grew 133 lines near the
top this round (§A), so orchestrator line numbers quoted in earlier entries and in the
cross-check sit about 107 lines lower now — the Stage-1 call is at `:3590`, the
out-of-scope filter at `:3617`, the RAG gate at `:3638`. Numbers below are current.

### A. Stage 0's retry could overwrite a correct refusal — and did

**Error.** PMC13278307 (*An Overview of Mobile Colistin Resistance (mcr) Genes*) ran
strict with the extraction-focus box set to "lipid A biosynthesis in Escherichia coli"
and came back **PASS in 2098.21s with 14 reactions and 0 gate errors** — every one of
them an mcr / PEtN / L-Ara4N lipid A *modification* step, and not one reaction of the
lipid A biosynthesis pathway it was asked for.

**Why.** Two defects in the same six-line block, both load-bearing. First, the retry
fired on a *correct* refusal: Stage 0 fails closed to an empty context, so
`streamlit_app.py` retried on a bounded head whenever `not _has_usable_context(...)` —
but the Stage-0 prompt's Case C (a multi-example review with no target selected)
**mandates** those same fields be blank, so a correct Case C is indistinguishable from
a failed Stage 0 by that test. With `_PREPROCESS_RETRY_CHARS` at 20,000
(`streamlit_app.py:204`) and this paper's source text at 50,377 chars, the retry fired
every time. Second, `pathway_context = preprocess(head)` adopted the second draw
unconditionally. When the first draw was a Case C and the second was blank, the
overwrite dropped `document_type` — the one field
`is_ambiguous_multi_example_review_context` requires (`preprocessor.py:317-328`) —
which disarms the refusal gate and lets an unguided extraction run over a review
describing six unrelated examples. Because the batch driver fills the focus box, such a
run no longer aborts loudly; it reports PASS.

**Fix.** The block moved into `_run_stage_zero_with_retry` (`streamlit_app.py:300`)
with an explicit ordering, `_stage0_context_rank` (`:268`): rank **2** is a well-formed
Case C, rank **1** a usable context, rank **0** neither. Case C outranks a usable
context on purpose — if one draw says "multi-example review, no target" and another
names a pathway, the safe reading is the refusal, and a second draw must never be able
to disarm a refusal gate. A deliberate Case C now skips the retry entirely (it is
deterministic; only the user naming a target helps), text at or below the retry bound
skips it too, and a retry that does run is kept in a local and adopted **only on a
strict rank improvement**. A genuinely failed Stage 0 can still be rescued; it can no
longer be degraded.

**On the `_EMPTY_CONTEXT` revert.** The 2026-07-25 entry (§E, this file) implemented
and then **reverted** adding `document_type`/`scope_status` to `_EMPTY_CONTEXT`. That
revert does not condemn this fix and the two do not collide: `document_type` only ever
arrives from a parsed model reply (`{**_EMPTY_CONTEXT, **result}`,
`preprocessor.py:139`), while every failure path returns `dict(_EMPTY_CONTEXT)`
(`preprocessor.py:165`, `:187`) carrying no `document_type` at all. Those are precisely
the draws that must rank 0, so the reverted shape supplies exactly the discrimination
the rank function needs. Adding the key back would give a crashed Stage 0 an *empty*
`document_type`, which the ambiguity predicate still rejects — it tests for the literal
value `multi_example_review`. Nothing here asks for the revert to be undone.

**Not re-measured.** No batch leg was re-run — that costs an LLM night — so the effect
is pinned by `tests/test_stage0_case_c_retry_clobber.py` (8 tests), not by a repeat of
PMC13278307.

### B. A type gate: a row's bucket now has to agree with what its name says

**Error.** The same PMC13278307 strict leg passed with `ok=true`, 0 errors, 37 warnings
and a 200,266-byte PWML while shipping **six entity rows whose declared type
contradicts their own names**: `pmrHFIJKLM operon` (an LPS-modification DNA operon) in
`entities.compounds`, and `arnBCADTEF operon`, `pmrCAB operon`, `mcr genes`,
`arnBCADTEF`, `pmrHFIJKLM` in `entities.proteins`. After mapping those protein rows
carry `mapped_ids.uniprot` **P30843** (×3), **P03023** (*Lactose operon repressor*) and
**P08235** (the human mineralocorticoid receptor) — accessions obtained by asking
UniProt about DNA.

**Why.** The origin is one line of RAG synthesis, `rag/synthesize.py:1054`
(`is_protein = key in enzyme_names`): the bucket is chosen by **grammatical role** —
appeared in some reaction's `enzymes[]` means protein, appeared anywhere else means
compound — and no type judgement is made anywhere. That is why the same DNA landed in
two different buckets of one payload. Nothing downstream noticed because no validator
we own has ever asked what *kind* of thing a name denotes: `validate_post_mapping`
checks that `mapping_meta`/`resolution`/`status` exist, and
`run_strict_post_normalization_gates` checks that a protein has a species and an
accession — all satisfied by a DNA operon filed as a compound. The one report that
mentions these rows, `strict/pwml_ir_report.json`, has the failure inverted: it flags
`pmrHFIJKLM operon` with `compound_db_resolution_failed`, complaining about the row
that harmlessly failed to resolve while saying nothing about the rows that resolved
confidently to the wrong molecule. The bucket is not a label: it selects the database
(`map_ids._DIRECTLY_MAPPED_ENTITY_BUCKETS`) and the PWML section a row is exported in.

**Fix.** `enforce_entity_type_consistency` (`process_normalizer.py:4746`), called at
`:5018` between `relocate_complex_named_proteins` and the strict gates — after the
passes that decide what the final actor rows and complex components are, and before the
passes that judge a row by the bucket it is sitting in. Two rules
(`nucleic_acid_name_verdict`, `:739`): an explicit **terminal** nucleic-acid noun
(`NUCLEIC_ACID_TAIL_RE`, `:228`, anchored at the end so *Lactose operon repressor* —
the very entry this run mis-shipped — stays a protein), and bacterial gene-cluster
shorthand (`GENE_CLUSTER_SYMBOL_RE`, `:242`, `^[a-z]{3}[A-Z]{4,}$` with a
`pro/pre/apo/iso/sub/cis` denylist so `proBDNF` is not read as a cluster), which is
**advisory only** and never relocates. Both regexes were replayed over every entity row
of every payload artifact under `runs/` — 12 payload files, 209 distinct names, 275
compound rows — and match exactly those six names and nothing else. A relocated row is
stripped of the identifiers the wrong database produced (`_BUCKET_IDENTITY_FIELDS`);
that is 92e1192's identifier-falsification class applied at the relocation instead of
at the resolver. A row referenced as an enzyme, modifier or transporter, or as a
protein_complex component, is **flagged and left in place** — moving it would
manufacture `Unknown protein/modifier reference` out of a correction — and stamped with
`entity_type_gate` so the mapper, the reviewer and the report all see the disagreement.
`validate_registry_references` now also unions `entities.nucleic_acids`: the bucket has
always been a first-class payload member and `pwml/ir.py:1394` has always listed
`nucleic_acid` as a legal `reaction_member`, but the registry validator never knew it
existed, so relocating `pmrHFIJKLM operon` — a reaction *input* of this paper — would
otherwise have produced `/processes/reactions/3/inputs/0 unknown entity`.

**Measured**, by replaying the gate today over that leg's own artifacts.
`merged_payload.json`, the shape the `streamlit_app.py:2197` normalization sees: **1 relocated**
(`pmrHFIJKLM operon`, out of compounds, taking its `element_locations` row with it) and
**5 flagged** in place. `final_mapped.json`, the post-mapping shape the writer's second
normalization sees (`pwml/writer.py:2650`): **3 relocated** (`pmrHFIJKLM operon`,
`arnBCADTEF operon`, `pmrCAB operon`), **stripping `mapped_ids.uniprot` P30843 from the
latter two**, and 3 flagged (`mcr genes`, pinned as an actor; `arnBCADTEF` and
`pmrHFIJKLM`, advisory-only). The function's docstring says "five of the run's six rows
are pinned"; that is true of the pre-mapping pass only — post-mapping the complex
wrappers absorb the actor references and two more rows become movable.

**Half of the class is deliberately not fixed.** The mirror direction — `PhoP` and
`phosphorylated PhoP`, a DNA-binding response regulator, filed as *compounds* — is left
alone. Moving a compound into `proteins` at this point manufactures two new hard gate
errors per row (missing species/organism, missing UniProt or DrugBank identifier),
because Stage 2 mapping already ran at `streamlit_app.py:2065` and stamped identity
only on rows that were proteins then; turning a PASS into a FAIL is not a fix. The
falsified identifier is already stopped by 92e1192's name-plausibility gate (`PhoP` →
NAD, CHEBI:15846 / KEGG C00003 / CAS 53-84-9). The correct home for that half is the
pre-mapping seam, which this pass cannot reach.

### C. Deleting an entity could strand every reaction that names it

**Error.** Two shapes, both reproducible from artifacts on disk.
`runs/2026-07-27_1623` PMC13231680/strict recorded
`"/processes/reactions/2/inputs/0 unknown entity: phthalylsulfacetamide (PSA)"` while
that same leg's `merged_payload.json` still carries five near-duplicate compound rows
(`phthalylsulfacetamide` twice, `phthalylsulfacetamide (PSA)`, `PSA`, `sulfacetamide`):
normalization passed the registry gate with all five present, then a curation step
deleted the redundant parenthetical row as duplicate cleanup and left the reaction
input, which spells the name *with* the parenthetical, pointing at nothing. At scale,
`runs/2026-07-28_0919` PMC12444477/research: **24 of its 25 gate errors** are
`unknown_protein_modifier_reference` — e.g. `/processes/reactions/8/enzymes/7` →
`"acetyl-CoA carboxylase enzyme complex (comprising AccA, B, C, and D components)"` —
after burning 2778s.

**Why.** `_is_core_semantics_path` scopes only to `/processes/*`, so the
`_is_safe_core_remove` guard in `_should_accept` never looks at `/entities/...`. A
remove of `/entities/compounds/N` needed nothing beyond confidence ≥ 0.95, and
`audit_json_llm.py`'s prompt actively solicits exactly that — its patch policy lists
"duplicate cleanup" under *High confidence (≥0.95)*. The asymmetry was total: a
reaction cannot be deleted unless it is a provable no-op at ≥ 0.97, while the entities
that reaction points at could be deleted freely, with a bare remove and no cascade.

**Fix.** A referential-integrity guard in `apply_patch_with_policy`
(`apply_audit_patch.py:1011`), running **after** each op against the live before/after
pair rather than inside `_should_accept`, which is handed the stale pre-batch
`source_payload` — after the first entity removal every later index has shifted by one,
so a pre-application check would inspect the wrong row. Diffing registry coverage also
makes the guard shape-agnostic: one predicate catches a whole-row remove, a whole-bucket
remove, a `/name` remove and a shortening `replace`. It **refuses rather than
cascades**: rewriting the surviving references is a second guess stacked on the model's
first, and the PSA case is three spellings whose merge target is a judgement call about
whether the parenthetical is an alias or part of the name. The refusal lands verbatim
in `rejected_patch_log.json` behind the greppable
`REFERENTIAL_INTEGRITY_REASON_PREFIX` (`:853`), naming at most 5 orphaned references
before summarising, so the next audit round can propose the synonym-add repair the
prompt itself calls the lowest-risk fix. The guard borrows `process_normalizer`'s own
`_entity_name_norms` / `_normalize` rather than re-deriving identity, so it cannot
disagree with the Stage-3 gate that would later reject the payload — declared synonyms
count on both sides. A whole-batch look-ahead (`_pending_entity_name_norms`, `:995`)
keeps the two-op cofactor relocation the prompt asks for ("remove from `proteins`, add
to `compounds`") working, since the remove is only safe in light of an add that has not
run yet. **True duplicate cleanup is untouched**: the surviving row supplies the same
normalized name, the coverage diff comes back empty, and the guard returns after one
set difference without ever walking the processes block.

### D. A delivered reaction was not attributable to the paper it came from

**Error.** In PMC12444477/strict `merged_payload.json` the key union across all **27**
delivered reactions is exactly `('biological_state', 'enzymes', 'evidence', 'inputs',
'locked_reaction_id', 'name', 'outputs')` plus two repair keys on 3 of them — **zero
reactions carry `rag_provenance`**. In the same file **41 of 56 compounds and 18 of 31
proteins do** (35 pointing at PMC12898747, 2 at PMC11046580). The payload therefore
proves cross-paper import happened while making it impossible to attribute a single
delivered *reaction* to the paper it came from.

**Why.** `_clean_processes` rebuilds every process row from a key whitelist that named
no RAG carrier; `_clean_entities` copies an entity row key-for-key. Same payload, two
policies.

**Fix.** `_carry_rag_provenance` (`pipeline.py:1747`) copies the three **namespaced**
carriers `_RAG_ROW_CARRIER_KEYS` (`:1744`) — `rag_provenance`, `source_papers`,
`rag_confidence` — onto reactions, transports, reaction-coupled transports and
interactions. `evidence` is excluded because the cleaners flatten it to a string on
purpose and re-emitting the record list would hand back the shape that seam exists to
remove. `source_refs` is excluded because it is a *core* key that two pieces of
locked-reaction machinery read as an evidence fallback
(`reaction_preservation_validator._evidence_text:120`,
`reaction_lock_manifest._evidence_quote:74`), so introducing it on reaction rows could
move a locked reaction's preservation status — a separate change needing its own
before/after. The copy is appended last, so a non-RAG row's key order is byte-for-byte
what it was and a row with no carrier gains no key; reaction merge and dedup fingerprint
on inputs+outputs only, so reaction identity, ordering and count are untouched.

This is a **precondition, not a cure**. It does not remove one imported reaction. It
makes "how much of this pathway came from another paper?" answerable at all, which is
what any gap-relevance filter has to be evaluated against.

### E. An HTTP 200 with no text was returned as an answer

**Error.** Two legs of the reference run died on `"Payload must include a processes
object"` — PMC13278307/research after **137s**, PMC13231680/strict after **55s**. The
latter is the paper that extracted 3 reactions successfully the previous day, and whose
research leg in this very run passed with 4 reactions in 936.7s off the same source
text.

**Why.** Every retry loop in `llm/client.py` fired only on a **raised** exception. An
HTTP 200 whose `message.content` was `""` or `None` was handed back as a success by
`return (resp.choices[0].message.content or "").strip()`, and `finish_reason` was never
read anywhere in the file. The one layer that can see "the provider answered 200 and
sent nothing" was the one layer that called it an answer.

**Fix.** `_completion_is_empty` (`client.py:193`) and `_finish_reason` (`:165`), folded
into the **existing** loop, backoff and `LLM_MAX_RETRIES` budget rather than growing a
second retry mechanism. On budget exhaustion `chat()` returns `""` and
`chat_with_tools` returns the raw response — the pre-change contract, because the
preprocessor already turns `""` into status `empty_reply` and a new exception type would
change every call site at once. The tool-calling direction is the dangerous one and is
handled explicitly: `tools_were_sent = bool(include_tools and tools)` is captured from
the same expression that decided what went on the wire, so a `tool_calls`-only reply
with `content=None` — the normal shape of a function-calling turn — is never counted
empty and never re-issued.

**Claim downgraded on review.** An earlier draft of this fix's own comment asserted
each of those two deaths was caused by a *single* empty Stage-0 reply. Review checked
that against the artifacts and it does not hold, so `client.py` now carries the
correction rather than a plausible story attached to a real fix. Neither leg wrote any
artifact (`files: []` in both manifest rows; only `RESULT.txt` on disk), so **nothing in
that run records an empty completion** — the diagnosis was inferred from the shape of
the failure. And the run executed commit 12bc11b (`batch.log` 09:19:08 → 11:49:05;
92e1192 landed at 14:06:43), where Stage 0 was *already* drawn twice for any text over
20,000 chars — both legs' source texts are 50,377 and 61,997 chars — so the
single-reply story needs two consecutive empty completions, not one. What survives
unchanged is the hole itself, which is visible in the code independent of any leg: an
empty 200 was returned as a success, cost a caller a full retry round or a degraded
result, and was invisible to every counter and log line in the module. `finish_reason`
is now surfaced precisely because a provider hiccup (retry is right) and a
`content_filter` stop (retrying the identical prompt burns wall clock and cannot help)
were indistinguishable in the 2026-07-28 postmortem.

### F. A preflight, because one missing import burned a whole night

**Error.** `runs/2026-07-27_2135`, still on disk and still looking like a night that
ran: the parent started at 21:35:06, fetched full text for **28 papers by 21:35:49
(43s)**, then recorded **all 56 paper+mode legs as failures between 21:35:49 and
21:36:13 — 24 seconds for work that takes ten hours**. Every one of the 56 manifest
rows carries the identical `ModuleNotFoundError: No module named 'streamlit'` from
`from streamlit.testing.v1 import AppTest`, filed as `failure_kind=crash`. `SUMMARY.txt`
reported `strict 0 pass 28 fail | research 0 pass 28 fail` and opened its triage matrix
with `!! RESEARCH-MODE DEFECT !! papers affected: 28` — 28 pipeline defects that do not
exist — and the run left a `plan.json`, 28 paper folders, a cache snapshot and a 56-row
manifest behind it.

**Why.** `driver.py` **defers** the two imports that matter — `t2pw.rag.research_report`
(`driver.py:1101`) and `streamlit.testing.v1` (`driver.py:1199`) — so the parent's own
`import t2pw.batch.driver` proves nothing about either. The usual Windows cause is that
`.py` is associated with `C:\WINDOWS\py.exe`, which ignores an active virtualenv, so
`scripts\batch_run.py` and `.venv\Scripts\python.exe scripts\batch_run.py` are two
different interpreters and only one of them has the dependencies.

**Fix.** A preflight in `batch/runner.py` that proves the **child's** environment in the
parent, before anything is planned or fetched. `CHILD_IMPORTS` (`:1233`) names the four
modules a child needs *with the reason it needs each*, and a test AST-parses `driver.py`
and fails if any non-stdlib deferred import is missing from that list, so a future
"move the import into the function" refactor cannot silently reopen the hole.
`probe_imports` spawns **one** fresh child (`[sys.executable, "-c", ...]` under
`child_env()`) which reports sentinel-prefixed JSON back. A subprocess, not an
in-process `importlib` call: the parent's `sys.modules` is only a proxy for a child's
and it lies both ways — four test modules in this repo install a `MagicMock` as
`sys.modules["streamlit"]` (they must; importing `streamlit_app` executes a Streamlit
script) and pytest imports every test module during collection. The first, in-process
version of this check broke three pre-existing tests in `tests/test_batch_run.py` and
reported streamlit missing on a box running streamlit 1.58.0. The subprocess is also
strictly stronger: it catches a broken `PYTHONHOME`, a raising `sitecustomize`, an
interpreter that cannot start, and a module that `sys.exit()`s at import.
`EXIT_PREFLIGHT = 3` (`:1211`) because 2 was already double-booked twice over —
argparse exits 2 on any usage error, and `run_overnight.bat` uses 2 for a missing
virtualenv. The message is ASCII-only, 78 columns, names the module, why a child needs
it, the interpreter in use, the project venv, and a cure that is a **command**,
branching three ways (wrong interpreter / already in the venv / no venv at all). An
interpreter outside the project `.venv` warns but never blocks — a conda env or a
container is legitimate — and the warning is suppressed on the failure path, where
claiming the imports succeeded above a `PREFLIGHT FAILED` block would be false.
`scripts/batch_run.py` calls it on the parent path only, after `--status` and `--single`
return and immediately above `run_batch`, whose first acts are `mkdir(out_dir)`,
`new_run_dir()` and that 43-second fetch. `--status` stays pollable at exit 0; the child
never re-probes, because the parent already vouched and a disagreement would break the
one-row-per-pair contract.

**Two review findings, both fixed.** The probe's answer is now written as its **own
whole line** and located by scanning lines for the sentinel — the convention
`parse_child_output` (`runner.py:1049`) has always used — instead of
`stdout.split(sentinel, 1)[1]` plus `json.loads` of the remainder, which any stray byte
of child output would have turned into a refusal of a **healthy** night; a
seen-but-unparseable sentinel is now a distinct message from "the probe never started".
And `run_overnight.bat` gained its own branch for exit 3 ("STOPPED BEFORE STARTING …
NO run folder was created"), because the generic branch's "read `SUMMARY.txt` in the
newest `runs\` folder" would have pointed the morning operator at the **previous**
night's summary — the exact "a refused night looks like a night that happened"
confusion the preflight exists to prevent, arriving on the one surface a double-click
user reads. Exit codes 0/1/2/3 are now documented in `scripts/batch_run.py`'s module
docstring, in its `--help` epilog, and in `docs/batch_runner.md` §2d.

**Measured.** The same fault is now detected in **0.72s** cold, including interpreter
startup (review independently measured 0.67–0.69s steady state, 1.44s on a cold first
run), before anything is fetched or created: exit 3, one message, and **zero bytes
written** — no run directory, no `plan.json`, no manifest, not even an empty `--out`
folder, asserted by a test that spawns the real CLI with a stub `streamlit` package on
the child's `PYTHONPATH` raising the recorded `ModuleNotFoundError`. Cost on a healthy
night: one 0.72s subprocess against a ten-hour run.

### G. Enrichment made 68 external calls one at a time

**Error.** Roughly **22 of PMC12444477/strict's 49m48s** went to this module.
Reconstructed from cache mtimes and the LM Studio log: the leg ran 09:19:08 → 10:08:57,
RAG finished its last embedding at ~09:27:48, `id_mapping_cache.json` was written at
09:44:44 and the audit candidates at 09:45:13, and **from 09:45 until 10:07:37 the only
file the process wrote was `enrichment_cache.json`**.

**Why.** That leg's `final_mapped.json` carries 44 compounds and 22 proteins, reducing
to **68 distinct upstream lookups** — 21 UniProt, 17 ChEBI, 17 KEGG, 13 HMDB. (The
56/31 counts quoted elsewhere are `merged_payload.json`, before mapping dedupes;
enrichment only ever sees 44/22.) Nothing orders those 68 against each other: each is
an independent read of a different public database and the merge that consumes them is
a pure function of the fetched blobs. It was serial only because nobody had made it
otherwise.

**Fix.** `enrich_payload` now runs in three phases: **plan** (serial, no network,
returns the de-duplicated first-encounter-ordered list of cache *misses*), **prefetch**
(one bounded `ThreadPoolExecutor` per service — uniprot 4, chebi 3, hmdb 2, kegg 2, with
a per-lane pacing floor), and **merge** (the original serial loops, unchanged, taking
each blob out of the phase-2 dict instead of calling the network; a key missing from
that dict falls through to the original inline fetch, so a planner/merge divergence can
cost a request but never change a result). Four invariants make the output identical:
the cache is read and written **only** from the main thread in phase 3 and in the same
order, so `cache_hits`, `api_calls` and `calls.*` are unchanged; `report["entities"]` is
appended to only in phase 3; both phases go through the same two key-construction
helpers; and each worker thread gets its own `HttpClient`/`requests.Session` via
`_ClientPool`, since `requests.Session` is not documented thread-safe and sharing one
is the classic route to interleaved-response corruption. `max_workers=1` restores the
literal pre-change behaviour. `EnrichmentCache.save` now writes a sibling temp named
with pid **and** thread ident and `os.replace`s it with 3 retries; the previous
`write_text` truncated the real 25.9 MB file and then streamed into it, so a crash or a
Ctrl-C left a half-written document that `except Exception: pass` in `__init__`
silently reads as "no cache at all".

**Measured**, replaying that exact leg's `final_mapped.json` offline with the fetchers
stubbed at latencies probed against the real URLs: **serial 139.7s → concurrent 29.1s,
4.81×, 110.7s saved on one leg**, `api_calls=68` and `cache_hits=9` in both, and the
enriched payload, the enrichment report and the written cache file **byte-identical**
between `max_workers=1` and the default. Review reproduced this independently on the
real payload (139.7 / 29.1 / 4.80× / 110.7s, identical 47-row report ordering, 11
distinct worker threads = 4+3+2+2) and confirmed no stray `data/*.tmp-*` files survive.

**Two honest limits.** The `RLock` added to the cache accessors is **not load-bearing
today** — nothing mutates the cache off the main thread — and its own docstring says
so; it exists so the first caller who fills the cache from a worker thread fails loudly
rather than losing entries to a non-atomic `setdefault`-then-assign. And the ceiling is
ChEBI, which is *dead*: `getCompleteEntity` returned **HTTP 500 in 0.99s** on probe, all
230 cached ChEBI entries carry status `error`/`request_failed`, and 197 of 201 cached
HMDB entries are HTTP 403 because hmdb.ca refuses the `Project14-T2PW-IDMapper/1.0`
User-Agent. At ~4.8s per dead ChEBI id (3 attempts plus backoff) and three workers, 17
ids take ~29s — i.e. **the whole concurrent runtime is failed ChEBI calls**. This change
made a fast path to failure faster; fixing the endpoints is worth more than widening
any lane, and is now recorded as debt.

### H. The cross-check the last entry left outstanding, completed

That entry's final debt bullet asked whether three earlier entries already cover the
scope-creep and duplication debt, and said the check was outstanding. It is done, and
it changes the classification of two items.

**A regression, still open: the out-of-scope reaction filter (2026-07-14, this file).**
Both halves of that fix are still present — `src/t2pw/llm/prompts/pwml_system.txt:15-16` still
carries the strict core-only rule, `streamlit_app.py:3617` still calls
`filter_out_of_scope_reactions` between Stage 1 (`:3590`) and Stage 2, and the AST test
at `tests/test_streamlit_stage2_orchestration.py:807-844` still passes. **The filter
cannot remove anything.** The orchestrator receives Stage-1 output from
`run_stage_one_with_chunking`, which returns `clean_stage_one(...)` on both branches
(`pipeline.py:2060`, `:2069`, `:2114`), and `_clean_processes` (`pipeline.py:1738-1789`)
rebuilds every reaction from a key allowlist that **does not include
`scope_membership`**; the filter then reads `rxn.get("scope_membership", "core")`
(`pipeline.py:271`) and defaults every reaction to core. Proved by execution — raw
labels `['core', 'out_of_scope']` → filter removes `['off-pathway step']`; after
`clean_stage_one` both labels are `<ABSENT>` and the filter removes nothing — and
confirmed on real data: all 9 Stage-1 reactions and all 27 merged reactions of
PMC12444477 carry no `scope_membership` key at all. `git log -S 'entry["scope_membership"]'`
returns nothing, so the key was never in the allowlist: the 2026-07-14 fix was wired on
the wrong side of it and **has been inert since the day it landed**. It also creates an
asymmetry that entry's "Pipeline consistency" paragraph assumed impossible:
`reaction_lock_manifest._scope_membership` (`:59-60`, `:186`) reads the **raw**
pre-clean output (`write_stage1_lock_artifacts` runs at `pipeline.py:2058-2059`, before
`clean_stage_one` at `:2060`), so the manifest correctly refuses to lock an out-of-scope
reaction while the payload keeps it — out-of-scope reactions ship as **unlocked**
reactions. The tests missed it the same way 92e1192 §A's did: the two unit tests
(`tests/test_pipeline_cleanup.py:230-284`) hand the filter a hand-built payload that
already carries the label, and the AST test pins call ordering only. **This round did
not fix it.** §D opened that same allowlist for the RAG carriers, so adding
`scope_membership` is a one-line change — but it would put the label in front of the
filter for the first time ever, and every reaction the filter then removes is a reaction
that used to ship. That needs its own before/after over the corpus, not a rider on a
provenance fix; it is now the top item in the debt list with the measurement spelled out.

**Correct but on a different axis: the RAG scope/gap guardrails (2026-07-25).** That fix
is intact and reachable — `and rag_incomplete_flag` is gone (`streamlit_app.py:3638`
reads `if rag_config()["enabled"]:`), `_AUTO_TRIGGER_GAP_SOURCES = {"gate", "mapping"}`
is live at `rag/triage.py:76` and applied at `:138`, and its own prediction still holds:
`streamlit_app.py:3644` supplies `reports={"qa_graph": ...}` only, so `scope_clarity_score`
plus the explicit flag remain the effective auto-trigger. But that guardrail decides
**whether RAG starts**, not **what it imports**. The observed import enters at the S3
merge, downstream of the only filter call, and a RAG row cannot carry a scope label even
in principle: `_ALLOWED_ROW_KEYS` (`rag/synthesize.py:1136-1138`, asserted at `:1133`) is
`{name, inputs, outputs, enzymes, entity, entity_type, role, source_refs}` plus
`RAG_ADDITIVE_KEYS`, and `rag/conform.py` is a pure shape adapter with no relevance or
scope test. Re-running the existing filter after the merge would still keep all 17
imported reactions.

**Correct, but not the duplication we are seeing: the RAG synonym merge (2026-07-23).**
That fix is present and wired (`rag/synonyms.py`, `build_offline_synonym_resolver()`
called at `streamlit_app.py:462`, threaded in at `:469`, consumed at
`synthesize.py:1319` and `:1321`). It cannot explain the observed duplicates because
they are **not synonyms — they are byte-identical names**. Measured on PMC12444477/strict
`merged_payload.json`: 31 protein rows for 22 distinct normalized names, with **all nine
seed enzymes doubled** (lpxa, lpxc, lpxd, lpxh, lpxb, lpxk, waaa, lpxl, lpxm); 56
compound rows for 41 distinct names, 14 duplicated (lipidiva ×3, udpglcnac ×2, ump ×2,
kdo ×2, kdolipida ×2). Diffing the pairs shows the second copy is **the seed's own
entity re-imported through the RAG merge**, tagged `rag_provenance.source_id =
"seed_paper"`. `merge_additions` (`pipeline.py:1064-1071`) dedupes with `_extend_unique`
(`pipeline.py:2458-2472`), whose signature is `json.dumps(item, sort_keys=True)`, and the
RAG copy carries a different key set, so the signatures differ and the row is appended;
`_clean_entities` (`pipeline.py:1540-1553`) dedupes by normalized name but only *within*
one incoming list, never across base + additions. That is new debt, recorded below.

**Already regressed once before, and re-fixed by 92e1192: enzyme fabrication by
substring matching (2026-07-22, this file, `:1217-1222`).** That entry excluded
`_inject_name_based_modifiers` from the shared cleanup because running it over RAG
evidence attached every enzyme to every reaction (99 spurious
`reaction_enzyme_must_be_protein_complex` errors) and stated it "now reads string
evidence only". The guard survived; the shape did not. `rag/conform.py` later began
flattening the evidence list to a string upstream, so the `isinstance(value, str)` test
was defending against a shape that no longer arrived, and the same defect returned as
92e1192 §A. Recorded here so the next round knows this guard has failed once already and
that a shape-based guard needs a test that feeds it the shape production actually
produces.

### I. `src/t2pw/stoich/` will not be wired, and here is the evidence

The debt list claimed "a mass/atom balance check would catch this class" and pointed at
`src/t2pw/stoich/`. A feasibility study read the package, traced every caller and ran
its classifier offline against the real payloads. **Recommendation: leave it. Do not
wire it.** A reasoned decision not to build something belongs in this file, so:

**It is not a stoichiometry checker.** Four files, 851 lines (`templates.py` 81,
`classifier.py` 156, `agent.py` 612): 11 hand-written keyword templates naming cofactors,
a `classify_reaction` that substring-matches the reaction name, tiebreaks on a 10-name
cofactor set and otherwise falls back to an LLM, and a per-reaction OpenAI tool loop.
There is **no mass, atom or charge balance and no coefficient inference anywhere in
it**: `chebi_verify` returns `{found, chebi_id, canonical_name}` and deliberately not a
formula, `kegg_reaction_get` parses only the EQUATION line, `_parse_kegg_equation_side`
(`agent.py:108-120`) explicitly **strips** stoichiometric coefficients, and the single
mutator `apply_stoich_fix` (`agent.py:196-248`) only **appends** named compounds. It is
a cofactor-completion agent.

**It has never run.** Exactly one caller, `streamlit_app.py:2816-2820`, guarded by
`use_stoich_agent`, whose only source is the checkbox at `:4021` with `value=False`.
`driver.py` never sets `session_state["post_use_stoich_agent"]`, `grep -rn stoich
src/t2pw/batch/` returns nothing, and no `*stoich*` artifact exists anywhere under
`runs/`. Zero tests import it — 851 lines with no test coverage. Last touched by commit
e3f2c95 on 2026-05-28, before RAG, before the gates, before 92e1192. Three dead one-line
shims survive at `src/stoich_agent.py`, `src/stoich_classifier.py`,
`src/stoich_templates.py`.

**Run against the real data it would make things worse.** With the LLM stubbed, **38 of
41 reactions (92.7%)** across the two completed strict legs fall through both
deterministic passes to the LLM — and all four named defects are in that 38. Of the 3
that do get a deterministic class, **2 are wrong and backwards**: "dephosphorylation of
PGP to produce PG" classifies as `kinase_phosphorylation` at HIGH confidence (so no LLM
check) and demands +ATP/−ADP on a **phosphatase**, because "phosphorylat"
(`templates.py:4`) is a substring of "dephosphorylation"; and "Acetyl-CoA → malonyl-CoA"
classifies as `coa_transfer` and demands free CoA as a substrate of a **carboxylase**.
The same collisions hit ordinary names: "enoyl-CoA dehydration" and "3-hydroxyacyl-ACP
dehydratase reaction" both classify as **hydration** and demand H₂O as an *input* —
direction inverted on the FAS-II / β-oxidation step present in essentially every lipid
paper this pipeline processes — and "succinate dehydrogenase" classifies as
`nad_linked_dehydrogenase` when it is the textbook FAD-linked one.

**And it cannot catch the class it was nominated for.** A balance check is
**direction-symmetric** — A→B is unbalanced exactly when B→A is — so no mass or atom
balance can ever detect the reversed `Phosphoethanolamine → lipid A`. The package has no
swap, replace, delete or rename operation either, and "phospholipid" is a compound
*class* with no formula, so nothing an appender can add repairs it. Cost if wired: of
the 41 reactions, **30 (73%) must be skipped unevaluated** because at least one
participant carries no formula-capable id; of the 11 nominally evaluable, 3 contain a
demonstrably wrong molecule read from the row's own `mapping_meta` (PG →
"Uridine diphosphate glucose", CL → "Chloride ion", PE → "Phytocassane E") and 6 contain
a class term (lipid A, phospholipid, LPS, modified lipid A) with no single formula —
leaving **2 of 41 (4.9%)** balanceable on identities that are both resolvable and
correct. Practical yield in the reference run is **zero**: all 14 ChEBI
`getCompleteEntity` lookups for these compounds returned HTTP 500
(`cache_snapshot/enrichment_cache.json`, `retrieved_at` 2026-07-22). Deleting the
package is a separate call nobody made this round; it stays where it is, unwired, and
the debt bullet that nominated it is rewritten below.

**Verified.** Full suite green: **1067 passed** in 201.5s (baseline at 92e1192: **929**).
Seven new files carry 106 test functions — `tests/test_batch_preflight.py` (35),
`tests/test_enrich_entities_concurrency.py` (17),
`tests/test_apply_audit_patch_referential_integrity.py` (15),
`tests/test_entity_type_gate.py` (13),
`tests/test_llm_client_empty_completion_retry.py` (11),
`tests/test_stage0_case_c_retry_clobber.py` (8),
`tests/test_pipeline_reaction_rag_provenance.py` (7) — and the balance of the +138 is
parametrised cases plus tests added to existing files. **No pre-existing test was
rewritten to match changed behaviour.** The one time this round broke pre-existing tests
— three CLI tests in `tests/test_batch_run.py`, under a first, in-process version of the
preflight import check — the check was redesigned into a subprocess probe rather than
the tests being relaxed.

---

## Four defects that let a wrong pathway pass every gate (2026-07-28, branch `research-mode`)

The first corpus run to actually reach export, `runs/2026-07-28_0919`, put two
papers through strict mode. **Both PASSED with `gate_errors: 0`** and wrote
importable PWML. Both pathways are substantially wrong. Nothing in the stack
noticed, because every gate we own measures structural validity and none of them
measures whether the biology is true.

**Error.** On PMC12444477 (*The regulation of lipid A biosynthesis*, E. coli),
Stage 1 extracted **9 reactions with exactly one enzyme each** — the correct Raetz
pathway. The delivered payload had **27 reactions and 204 enzyme rows**, of which
**177 carry evidence of exactly 119 or 120 characters**. Reaction #14's evidence
is **139,576 characters: one 4,812-character passage repeated 29 times**. WaaA,
the Kdo transferase, is credited with `CDP-DAG + G3P -> PG`; a complex headed by
FtsH, the LpxC protease, is credited with cardiolipin and phosphatidylserine
synthesis. On PMC13278307 (the mcr colistin review) the run shipped UniProt
**P08235 — the human mineralocorticoid receptor — as an enzyme on 10 of 14
reactions**, P03023 (LacI) for the `pmrHFIJKLM` operon, and mapped the protein
PhoP to **NAD+**, all with resolution status `matched`. The same paper's research
leg **failed with 30 gate errors while every contract report said `ok=true`**.

**Why.** These are four independent mechanisms that compose. Evidence gets
amplified into a blob; a substring matcher reads that blob and manufactures
enzymes; the mapper answers every query with its best guess and calls it a match;
and the batch driver fails research runs on a report research mode does not treat
as authoritative. Each is small. Together they turn nine honest reactions into a
certified-passing artifact that is mostly false.

### A. A substring matcher manufactured 177 of 204 enzyme attachments

`_inject_name_based_modifiers` (`src/t2pw/pipeline/pipeline.py`) tests every
declared protein name against a reaction's evidence text. Its guard,
`_row_evidence`, checked only `isinstance(value, str)` — and its own docstring
claimed to defend against *list*-shaped evidence, a shape that no longer reaches
it because `rag/conform.py` flattens the list to a string upstream. So the guard
was a no-op against the only shape that mattered, and a 139,576-character blob
sailed through it containing nearly every protein name in the corpus. The
`[:120]` slices that produced the 119/120-character signature are the injector
storing a truncated prefix of that blob as the modifier's evidence; the 119-vs-120
split is a `.strip()` removing a trailing space.

**Fix.** A size bound, `MAX_INJECTOR_EVIDENCE_CHARS = 400`, now lives in the new
leaf module `src/t2pw/pipeline/enzyme_cues.py` together with the catalysis-cue
machinery, which was previously two private names inside the 4,500-line
`process_normalizer.py`. The threshold is not arbitrary: measured reaction
evidence lengths in the research payload are `[22..102]` for ten reactions and
then `4636, 5113, 5113, 37946, ... 418122` — a **45× empty gap around 400**, with
real model-emitted evidence below it and RAG blob above. Attachment now also
requires the actor to sit inside a catalysis cue window **and be the only actor
that qualifies**, matching the pre-existing `len(matches) != 1: continue` rule in
`process_normalizer.py`; the matched cue snippet is stored as evidence instead of
a truncated prefix. `attach_enzymes_from_reaction_evidence` got the same length
guard as a `continue`, never a truncation. `_attached_actor_names` now scans
`enzymes` as well as `modifiers` — the old check missed that Stage 1 writes its
catalyst to `enzymes`, so all nine correct seed enzymes were being re-injected as
duplicate modifiers carrying truncated evidence. `_clean_enzymes` now carries
`provenance` and `confidence` through, which is why every enzyme row in every
previous run had the key set `('evidence', 'protein')` and was untraceable.

`pwml/qa.py:69-72` already documents that an enzyme-less reaction is expected and
not an error, so `reactions_no_enzyme` rising above zero is the correct outcome,
not a regression.

**A test was hiding this.** `test_merge_additions_still_applies_the_name_heuristic`
was **vacuous**: its fixture already gave reaction 0 an `enzymes` entry and the
test cleared only `modifiers`, so it passed with the injector deleted outright.
The fixture now clears `enzymes` too, so the test actually exercises what it
claims to pin.

### B. The same passage was stored 29 times because one carrier was not deduped

`_reactions_from_bundle` (`src/t2pw/rag/synthesize.py`) runs once per gap bundle,
and each reaction it builds carries the whole chunk text. A chunk that is top-k
for N gaps therefore yields N identical rows — **the repeat count is the bundle
count**. `_merge_into` then folds them with `target.evidence.extend(...)` while
*unioning* provenance six lines above. `_attach_provenance` shows the same
asymmetry: papers go through `_dedupe_papers`, source refs through
`_dedupe_strs`, and evidence through nothing.

**Fix.** Evidence and `source_papers` are now deduped where they are merged, and
a `_dedupe_evidence` helper sits beside its two siblings. The key is explicitly
`(chunk_id, text)`: `dict.fromkeys` and `set()` are both unusable here because the
records are dicts, and dict equality would fail anyway since `_evidence_from_hit`
stores a per-retrieval `score` that differs between gaps for the same chunk.
`conform.py::_evidence_to_str` deduplicates at the flatten boundary as well, where
the elements *are* strings and order-preserving `dict.fromkeys` is safe.
`target.scores` is deliberately left accumulating — those are per-retrieval and
genuinely additive. The proposed "cap at N distinct passages" was rejected: it
would discard real provenance on a genuinely multi-source row.

Reaction evidence (2,716,278 chars) plus enzyme-row evidence (1,888,665 chars)
was 4.6 MB of the 4.70 MB payload.

### C. The mapper answered every query and called it a match

Wrong identifiers shipped with `resolution.status = "matched"`:

| entity | shipped | what it actually is |
|---|---|---|
| `PhoP` (a protein, routed through the compound mapper) | KEGG C00003, CAS 53-84-9, CHEBI:15846 | NAD+ |
| `pmrHFIJKLM` | UniProt P03023 | LacI, lactose operon repressor |
| `mcr genes` | UniProt P08235 | human mineralocorticoid receptor |

`pmrHFIJKLM` reached LacI by degrading through its query ladder to the literature
alias `operon` — `queries_tried` literally contains
`(protein_name:"operon" OR gene:"operon")`. Any name ending in "operon" was
exposed. For `mcr genes`, `mapping_meta.resolved_name` said *"Type IV
methyl-directed restriction enzyme EcoKMcrA"* while the shipped accession was
P08235, so the audit trail did not describe what shipped.

**Fix.** A name-plausibility gate in `src/t2pw/mapping/map_ids.py` rejects a match
whose resolved name shares no meaningful token with the query and routes it to the
**existing** `novel` status rather than inventing a new state — `novel` already
worked and was simply never used for a bad match. Token comparison is
case-insensitive, strips punctuation, and ignores generic biology words
(`protein`, `enzyme`, `gene`, `operon`, `complex`, `subunit`, `transferase`, …).
Bare generic aliases are no longer issued as standalone queries.
`tests/test_map_ids_name_gate.py` pins the five cases using the **real** UniProt
and PathBank responses from this run as fixtures: `MCR-1` →
*Phosphatidylethanolamine transferase Mcr-1* must keep passing, while `mcr genes`
→ *Mineralocorticoid receptor*, `pmrHFIJKLM` → *Lactose operon repressor* (both
name forms) and `PhoP` → *NAD* must all fail.

### D. Research mode was failed by a gate leg that does not know about modes

Research mode is fail-open by construction. In the failing run **every contract
report said `ok=true`, `still_blocking=0`, `research_blocked=[]`** — the relaxation
worked exactly as designed. The run was failed anyway by `batch/driver.py`, which
reads `gate_fail_report` and `final_stage3_gate_report` straight out of the
artifacts dict with **no mode check**, and does so *before* the
`if outcome.mode == MODE_RESEARCH` branch. This is the same class as the
nested-`runtime_schema_report` misread fixed on 2026-07-27 — whose own docstring
names this very paper — re-landed at a different seam.

Worse, research mode **manufactured 4 of its own 30 errors**:
`process_normalizer.py` deliberately skips `drop_process_orphan_proteins` and
`prune_disconnected_proteins` in research mode on the stated ground that "the gate
only flags", and then calls `run_strict_post_normalization_gates(...,
enforce_all_proteins_connected=True)` unconditionally — so the gate raised on
exactly the orphans the relaxation had chosen to keep.
`run_strict_post_normalization_gates` takes no mode parameter and none of its call
sites passed one, leaving it structurally incapable of relaxing.

**Fix.** Gate errors in research mode are recorded and surfaced through the review
flag and warning channels without failing the run; strict behaviour is unchanged.
The self-inflicted case is closed by making the enforcement argument follow the
same condition as the skip. `tests/test_batch_research_gate_fail_open.py` pins
that a research run with non-empty gate errors and `ok=true` contract reports does
not fail.

**Verified.** Full suite green: **929 passed**.

---

## Known Debt — found in `runs/2026-07-28_0919` (rewritten 2026-07-28 after the readiness pass)

Recorded so none of it is rediscovered as new. Six of the original twelve bullets were
closed by the debt round above and have been **removed** from this list — referential
cascade on entity removal, Stage-0 Case-C clobber, empty LLM completions, the batch
preflight, serial enrichment, and the outstanding cross-check. Two more were closed by
the pre-run close-out (the inert out-of-scope filter and the duplicate seed entities) and
are gone from here too; what that round could **not** settle is written below as its own
bullet, and the readiness pass at the top of this file re-verified the filter closure by
execution, closed nothing further, and **added three items** — the corpus that does not
fit a night, the finished leg the manifest lost, and what a green preflight still does
not prove. The rest are kept and amended where a fix changed their shape.

- **A newly-live filter can now silently delete reactions, and the night will not
  record it.** The out-of-scope filter fires as of the close-out entry above. Its
  removals are announced with `st.info` (`streamlit_app.py:3619`) and nothing else: the
  batch driver collects `at.error` / `at.exception` / `at.warning`
  (`driver.py:949-956`) and never `at.info`, and the parent keeps a child's stderr only
  on a timeout or a crash (`runner.py:2075`, `:2080` — the `_timeout_row` / `_crash_row`
  paths, re-verified), so a **passing** leg's log is discarded entirely.
  `tmp/reaction_lock_report.json` does carry the matching
  `out_of_scope_excluded_count`, but it lives at the project root, is overwritten by
  every leg, and is not among the artifacts copied into a run directory. Nor can the
  question be answered retroactively: all 130 reactions across the 18 delivered Stage-1
  and merged artifacts under `runs/` (178 across 21 files counting `final_mapped.json`)
  carry no `scope_membership` at all, because the defect stripped it before anything was
  written. **Two corrections to the shape of this item**, both from the readiness pass.
  First, it is cheaper than "a per-leg artifact": the app already stores the removal list
  in session state as `out_of_scope_removed_reactions` (`streamlit_app.py:3820`), and
  grepping `out_of_scope` across `src/t2pw/batch/driver.py` returns **nothing** — so it is
  one read in `_add_common_artifacts` (`driver.py:997`), not a change to the child→parent
  protocol.
  Second, the per-paper before/after is already recoverable from the artifacts the next
  run writes anyway: `stage1_payload.json` is the **pre**-filter payload
  (`streamlit_app.py:3819` stores the variable the filter did not consume, and
  `driver.py:1002` writes it) while `merged_payload.json` comes from the post-filter
  `final_payload`, so per paper the removals are the rows whose `scope_membership`
  case-folds to `out_of_scope` in the first and are absent from the second. Worth
  checking on the first two or three papers of the run rather than trusting the batch.
- **The corpus does not fit in one night, and nothing in the runner says so.**
  Measured over the six legs of `2026-07-28_0919` (§C of the readiness entry): the four
  full-pipeline legs mean **2200.2s**, the three big ones **2621.4s**, so a 10-hour
  budget buys **13.7–16.4 legs** (**24** only if cheap failures pad the count).
  `topics.txt` asks for 28 papers × 2 modes = **56 legs**, and its own note records
  observed full-text yield as roughly half, so the realistic corpus is ~28 legs — still
  about **two nights**. The runner enforces the deadline but never projects it: it does
  not compare `pending_pairs` × observed mean against the budget at startup, so an
  operator gets no warning that two thirds of the plan will be truncated. Until it does,
  cap the work with `--limit` (~7 papers) and budget **11h of wall clock for a 10h
  night**, because the deadline is checked *before* a leg (`runner.py:2037-2042`) and the
  per-leg timeout is an hour.
- **A finished, passing leg can be lost between the child exiting and the manifest
  write.** Live example on disk right now:
  `runs/2026-07-28_0919/papers/PMC13231680__mechanistic-insights-into-phthalylsulfacetamide/research/`
  holds a full artifact set and a `RESULT.txt` reading `PASS (with warnings)` at 936.7s,
  and there is **no manifest row for it** — the parent took Ctrl+C at 11:49:05 in the
  window between `parse_child_output` (`runner.py:2066`) and `append_manifest`
  (`runner.py:2096`). Consequences: `completed_pairs` does not know the leg happened, so
  a resume pays its ~937s again and overwrites good artifacts; and every count derived
  from the manifest understates the run (`SUMMARY.txt` says 3 papers attempted / 5
  manifest rows for 6 legs and 3 passes). Closing it properly means writing a
  provisional row before the child starts and reconciling after — a change to the
  child→parent protocol, hence deferred, hence written down.
- **A green preflight still does not prove the night can talk to a model.**
  `run_preflight()` passes in **1.39s** and now imports `t2pw.llm.client`
  (`runner.py:1274-1279`), which clears the three import-time faults (`client.py:38`,
  `:40`, `:63`). It cannot see an unreachable `openrouter.ai`, an expired or unfunded
  key, or a model the account may not serve — all of which need a live one-token call —
  and the model id is resolved lazily, so a missing `OPENROUTER_MODEL` would raise at
  `client.py:261` on the first call of the night rather than in the check. It is
  currently set (`google/gemma-4-26b-a4b-it`), so this is a limit of the instrument, not
  a live fault.
- **A pathway can contain none of the pathway it declares.** PMC13278307 ran with
  the focus box set to "lipid A biosynthesis in Escherichia coli" and produced
  **zero** lipid A backbone reactions — no LpxA/C/D/H/B/K, no WaaA, no LpxL/M —
  yet passed with `gate_errors: 0`. One *contributing* mechanism is now closed (the
  Stage-0 Case-C clobber, §A above, which let that review run unguided), but the
  gap itself is untouched: **nothing checks that the delivered pathway is the one
  that was asked for.** The focus box, the Stage-0 `pathway_name` and the delivered
  reaction set are never compared.
- **Scope creep by cross-paper import.** Reactions 10–26 of the lipid A payload
  are phospholipid biosynthesis, a different pathway; `rag_provenance` shows 35
  entities from PMC12898747 and 2 from PMC11046580. Per-reaction source is now
  recoverable (§D above preserves `rag_provenance` on process rows), so this is at
  last *measurable* — but no filter acts on it. Two constraints for whoever builds
  one: reaction-signature deduplication was tested against the real data and does
  **not** fix it (27/27 signatures distinct, still 27/27 with the enzyme component
  removed, 26 with a tight normalizer — these are imports, not name variants); and a
  RAG row cannot carry a scope verdict today at all, because `_ALLOWED_ROW_KEYS`
  (`rag/synthesize.py:1136-1138`, asserted at `:1133`) admits only
  `{name, inputs, outputs, enzymes, entity, entity_type, role, source_refs}` plus
  `RAG_ADDITIVE_KEYS` and `rag/conform.py` is a pure shape adapter with no relevance
  test. The import enters at the S3 merge, which is *downstream* of the only
  `filter_out_of_scope_reactions` call. The filter itself is no longer inert — the
  close-out entry above made it able to fire — and that changes **nothing** here: it
  still runs before Stage 2, so these reactions have not been created yet when it looks,
  and a RAG row could not carry a label for it to read anyway. The size of the gap is
  measurable directly: `PMC12444477/strict/merged_payload.json` delivers **27 reactions
  of which only 9 carry a `locked_reaction_id`**, i.e. two thirds of the delivered
  pathway entered downstream of every scope decision the pipeline makes. This is the
  largest open correctness item in the list and it needs a design, not a one-line
  allowlist change.
- **Entity-type confusion — the compound-side half is still open.** The DNA-named
  rows are handled (§B above relocates or flags them), but the mirror direction is
  not: `PhoP` and `phosphorylated PhoP`, a DNA-binding response regulator, remain in
  `entities.compounds`. It cannot be fixed at the normalizer — moving a compound into
  `proteins` after Stage 2 mapping manufactures two new hard gate errors per row
  (missing species/organism, missing UniProt/DrugBank) — so it belongs at the
  pre-mapping seam. Also note the gene-cluster shape rule
  (`GENE_CLUSTER_SYMBOL_RE`) is deliberately **advisory**: `arnBCADTEF` and
  `pmrHFIJKLM` are flagged and stay in `entities.proteins`.
- **Spelling variants of one species still ship as two entity rows.** What is left of
  the duplicate-entity bullet after the close-out entry closed the re-import half. The
  guard added there uses `process_normalizer._normalize` — the registry gate's own,
  deliberately lexical — so `lipid IV A` / `lipid IV_A` and `sn -glycerol 3-phosphate` /
  `sn-glycerol 3-phosphate` are two rows each in PMC12444477/strict, and the compound
  bucket lands at 43 rows rather than the 41 an alnum-only normalizer would give. This
  is the synonym resolver's job, not the merge guard's, and under-merging is the safe
  direction: `_normalize` preserves `+` and `:`, so `NAD+` never collapses into `NAD`.
- **Reaction duplication by prose restatement** — known and deliberately deferred by
  the 2026-07-23 entry (`:1113-1117`): unmapped placeholders "share no ID, so they
  correctly do not merge". Real in the same payload: rxn 12 "Phosphatidic acid (PA) →
  CDP-DAG" vs rxn 18 "PA to CDP-DAG conversion"; rxn 14 "CDP-DAG → PG" vs rxns 19, 20,
  21, 23 and 24. Left to a prose-extraction quality gate that does not exist yet.
- **Chemically impossible and reversed reactions ship — and stoichiometry is not the
  instrument.** A DNA operon and a protein each appear as reaction *substrates*;
  `Phosphoethanolamine -> lipid A` and `4-amino-4-deoxy-L-arabinose + PEtN -> lipid A`
  run backwards. The previous version of this bullet proposed `src/t2pw/stoich/`; §I
  above shows why that is wrong — the package implements no balance of any kind, a
  balance check is direction-symmetric and so can never detect a reversal, and only
  **2 of the run's 41 reactions (4.9%)** carry identities both resolvable and correct
  enough to balance. What this class actually needs is a *directionality* check
  (thermodynamic or reference-pathway based) plus the entity-type gate's substrate
  side, neither of which exists.
- **`focused_repair.run_focused_repair_passes` has no caller anywhere in `src/`.**
  Re-verified 2026-07-28: only its definition (`focused_repair.py:911`) and
  `tests/test_focused_repair.py` reference it. Its first pass is precisely the
  "reaction input with no matching declared entity" repair the referential-integrity
  guard (§C above) now *refuses* patches over. Wire it in or delete it.
- **The referential-integrity guard refuses; it does not repair.** §C above blocks a
  removal that would strand a reference, which leaves the near-duplicate rows
  (`phthalylsulfacetamide` ×2, `phthalylsulfacetamide (PSA)`, `PSA`, `sulfacetamide`)
  in the payload and defers the real repair — a synonym add, or the focused-repair pass
  above — to a later audit round that nothing currently forces to happen.
- **`clean_stage_one` turns emptiness into a structural violation.** Still true
  (`pipeline.py:2034`): it writes the `processes` key only `if processes:`, so an
  extraction that finds no complete reaction omits the key and trips the
  `processes_required` structural guard — converting an honest "no reactions found"
  into an abort research mode cannot fail open on.
- **Two of the four enrichment services are dead, and we now fail faster.** ChEBI
  `getCompleteEntity` returns HTTP 500 (probed 2026-07-28; all 230 cached ChEBI
  entries carry status `error`/`request_failed`), and hmdb.ca returns HTTP 403 to the
  `Project14-T2PW-IDMapper/1.0` User-Agent (197 of 201 cached HMDB entries). At ~4.8s
  per dead ChEBI id, those failures *are* the 29s the now-concurrent enrichment takes.
  Fixing the endpoints (or the User-Agent) is worth more than any further concurrency,
  and until then the compound half of enrichment contributes nothing.

---

## Unattended overnight batch runner (2026-07-27, branch `research-mode`)

Adds an unattended runner that fetches N papers from the literature and pushes
every one through the pipeline **twice** — `Strict PWML` and `Research (relaxed)`
— writing every artifact, a per-pair `RESULT.txt`, a `SUMMARY.txt` and a ranked
`failures_by_code.txt` into `runs/TIMESTAMP/`. New code:
`src/t2pw/batch/{fetch,driver,runner,report}.py`, CLI `scripts/batch_run.py`,
launcher `run_overnight.bat`, work list `topics.txt`. Full operator
documentation in [`docs/batch_runner.md`](batch_runner.md).

**Error.** Research mode was shipped but had never been run over a corpus, so
nobody knew which papers it breaks on or why. Doing that by hand is ten papers ×
two modes × three clicks each, spread over hours of LLM and PathBank latency —
which means it does not get done, and the two things worth knowing stay unknown:
which strict failures are just PathWhiz FORMAT rules (expected wear) and which
are real bugs.

**Why.** The pair of outcomes is the diagnostic, and only running *both* modes on
the *same* paper produces it. strict FAIL + research PASS is a FORMAT rule to
catalogue; strict FAIL + research FAIL is a real bug upstream of export; and
**any** research failure is a code defect, because research mode is fail-open by
construction and has no legitimate way to fail on real data. `report.py` encodes
exactly that as its triage classes and prints research defects first and loudest.

**Fix — it drives the real app; nothing was duplicated or refactored.**
`streamlit_app.py` is **unmodified**. It is not a thin view over the pipeline —
export-mode selection, the Stage-0 ambiguity refusal, the Stage-3 pre-export
revalidation, the RAG seam (S1/S2/S5) and the pre-merge payload the citation
report must be built from live in that file and nowhere else. A runner that
re-implemented that wiring would drift within a week, and an overnight run whose
failures do not reproduce in the browser is worse than no overnight run. So
`driver.py` drives the app through Streamlit's own headless harness,
`streamlit.testing.v1.AppTest`: it sets the same widgets a human sets and reads
the same `st.session_state` the app writes. The app is never imported, never
refactored, never copied — a change to it is picked up on the next batch run.
Paper acquisition is likewise not new code: `fetch.py` is a thin driver over
Stage R1 (`t2pw.rag.acquire`), so there is no second fetcher, no new HTTP code
and no new dependency.

Two structural decisions carry the design. Every paper+mode runs in a **child
process** (`scripts/batch_run.py --single ...`) because a thread cannot be killed
and `signal.SIGALRM` does not exist on Windows, so a hung LLM request or wedged
MySQL socket is unkillable in-process; the child is put in its own process group
and killed with `taskkill /F /T`. And the loop is **strictly sequential**, because
`data/id_mapping_cache.json`, `data/enrichment_cache.json` and `data/rag_index`
are shared mutable state with read-modify-write access and no locking; the two
JSON caches are snapshotted to `cache_snapshot/` so a bad night is revertible.

**An adversarial review found 11 defects before the runner was ever used for
real.** Ten were fixed and re-verified by execution; the eleventh (the AppTest
timeout) was still broken on re-check and has now been patched as far as it can
be. The three severe ones:

- **A completed paper was silently discarded and then marked done, so it was
  never retried.** The child prints its manifest row as one `ensure_ascii=False`
  JSON line, and the child's stdout is *always* a pipe — whose encoding is the
  ANSI codepage (cp1252 here), not UTF-8. One Greek beta in an entity name
  (`β-hydroxymyristoyl` is the canonical lipid A intermediate) raised
  `UnicodeEncodeError` **after** the artifacts and `RESULT.txt` were already on
  disk. The parent saw no result line, synthesized a crash row — and because a row
  now existed, `completed_pairs()` counted the pair as finished, so the passing
  run was lost for good. The encoding is now pinned in three independent places:
  `emit_outcome` writes explicit UTF-8 *bytes* to `sys.stdout.buffer` (falling
  back to `ensure_ascii=True` JSON, which no codec can refuse, if there is no
  buffer), `force_utf8_stdio()` runs first in `main()` before anything can print,
  and `child_env()` sets `PYTHONUTF8` / `PYTHONIOENCODING` while the parent
  decodes the pipes as UTF-8.
- **An unguarded `print` in `Logger` killed the whole night before paper one.**
  Setup logs each fetched paper's *title* — arbitrary journal text with Greek
  letters and em dashes. Printing it to a cp1252 console, or worse to a redirected
  log file or Task Scheduler's captured stream, raises `UnicodeEncodeError`; and
  `UnicodeEncodeError` is a `ValueError`, so the `except OSError` that guarded the
  file write did not catch it. The batch died during planning with zero papers
  run. `Logger.__call__` now cannot raise at all (verbatim → `_ascii_safe` →
  silent), and `plan.json` is written *before* any title is logged, so even a
  catastrophic log failure leaves a resumable run directory instead of an orphan
  that `find_resumable` refuses and every subsequent launch replaces with another
  empty one.
- **`AppTest.run(timeout=...)` is inoperative as a bound.** It requests a stop and
  then joins the script thread **unbounded**, raising the timeout only after the
  blocking call returns on its own — a report, not an interruption. Measured:
  `at.run(timeout=2.0)` against a 12-second block **raised the timeout but
  returned after 12.1 s**. So a wedged LLM or DB socket is bounded *only* by the
  parent's process-tree kill at `--timeout` (default 3600 s), plus the whole-night
  `--deadline` (default 10 h) that stops the loop before it is still running at
  lunchtime. This cannot be fully fixed from here, so it is documented rather than
  papered over: on the kill path the child dies **before** it can write its
  artifacts, so a killed pair genuinely produces nothing. The child is given the
  parent's timeout minus a 120 s grace precisely so a *nearly*-finished pair can
  land its files, and if it printed its row before the stopwatch expired the
  parent uses the child's own row instead of a fabricated failure.

The remaining eight were one family: **a bad night that looked clean.** A
swallowed `OSError` filed a missing `01_source_text.txt` as "the extractor found
nothing" — an infrastructure fault reported as a biology defect. `_relocate_files`
rebuilt each `files` entry and dropped its `error` key, so a run whose deliverable
never reached the disk was reported as a pass. The parent overwrote the child's
richer `RESULT.txt` with a reconstruction that had already lost that error.
`_element_texts` type-checked for `list`/`tuple` against AppTest's `ElementList`,
so a run that failed *loudly* read as one that said nothing. `find_resumable`
reached back past the newest run directory and had no age limit, so one stale
paper was re-run every night forever while nothing new was fetched. A manifest
whose last line lost its newline fused two rows on append, making the pair
invisible to resume and re-mangling it on every retry. And a research run that
produced no report, no citations and no tiers was summarised as `ALL GREEN` —
now a warning that travels into the manifest, prints as
`PASS (no research deliverable)`, and downgrades the night to
`PASSED WITH WARNINGS`.

**Tests.** Baseline before: **725 passing**; after: **872 passing** (+147 batch
tests across `tests/test_batch_{fetch,driver,report,run}.py`). No pre-existing
test was modified, and no file under `src/` outside the new `src/t2pw/batch/`
package was touched. `ruff check src tests scripts` reports the same **49**
pre-existing findings as before the change — no new lint.

---

## Research mode — relaxed export policy for novel pathways (2026-07-27, branch `research-mode`)

Adds a second export policy so RAG-synthesized *novel* pathways can be generated
and reviewed without fighting the PathWhiz importer. Strict PWML stays the
default and is unchanged. Full categorization in
[`docs/research_mode.md`](research_mode.md) (277 checks: 85 SKIP, 86 FLAG,
106 UNCHANGED).

**Error.** A genuinely novel enzyme from a recent paper has no UniProt
accession, is not wrapped in a synthetic `protein_complex`, and may carry a `:`
in its name. Today every one of those is fatal, and they are fatal at the same
severity as "this reaction has no inputs or outputs" — so a pathway that is
*biologically* fine cannot be looked at, while the checks that actually matter
are indistinguishable from formatting noise.

**Why.** The pipeline has exactly one audience today: the PathWhiz Rails
importer. Rules that exist only to satisfy it (required external DB identity,
the enzyme→`protein_complex` wrapper and its component rules, `+`/`:`
name-format rules, the pre-export required-field contract) are enforced with the
same `raise` as biology and provenance rules. There was no way to ask for "the
biology checks, but do not stop".

**Fix.** An explicit, user-selected export policy — not a RAG-derived one.

- **`src/t2pw/pipeline/export_mode.py`** (new) holds `ExportMode`, the FORMAT
  code set, the structural-guard set, and `relax_report`. `coerce_mode` resolves
  anything unrecognised to `"pathwhiz"`, so a typo degrades to strict rather
  than silently relaxing every gate. Unknown issue codes classify as *review*,
  not *skip* — necessary because `payload_models.py` emits its own copies of
  `species_required` / `generated_wrapper_missing_components` /
  `actor_schema_not_canonical`, so codes are not globally unique.
- **`stage_contracts.py`**: no validator body was parameterized. The new
  `run_stage_contract(validator, *args, mode=...)` calls the validator unchanged
  and re-severities the *report*. Strict mode is a plain passthrough, which is
  why strict behaviour is byte-for-byte identical by construction.
- **`process_normalizer.py`**: the mode rides on the shared `report` dict, which
  is already threaded through every pass and nested closure, so no pass gained a
  keyword. Research mode stops three **uncaught `ValueError`s** in composite
  materialization from killing the run, converts the **hidden**
  `assert actor_contract.get("ok") is True` into a recorded flag, skips
  `drop_process_orphan_proteins` / `prune_disconnected_proteins`, and makes
  `_record_non_protein_catalyst_drop` the single decision point for all five
  non-protein-catalyst drop sites. Every preserved row is recorded under a
  `research_mode_` action prefix.
- **`map_ids.py` / `entity_identity.py`**: `mode=` makes only *name handling*
  lenient (`lenient_names` stops reading `:` as complex syntax). Dropping the
  wrapper needed **no new mapping code** — `allow_complex_wrapper_creation`
  already existed with a skip branch; the orchestrator flips it at Stage 6.
- **`t2pw/rag/tiers.py`** (new) assigns evidence tiers A–D off-payload.
  Tier A excludes the two sites that stamp a *fake* grounded identity: the
  PathBank Unknown sentinel (`mapped_ids.uniprot == "Unknown"`,
  `map_ids.py:4218`) and `best_effort_fallback` (`map_ids.py:3822`). Distinct-
  paper counting is on `source_id`, and a review can never alone satisfy Tier B.
  The UniProt retry is read-only, opt-in, and never writes into mapping — so it
  cannot cause a wrong-organism accession to be stamped `mapped`.

**Deliberately not done.** Research mode emits **no PWML XML**, so
`pwml/ir.py`, `writer.py`, `validate.py` and `qa.py` are untouched — this also
retires the mis-serialization risk of writing a compound's id into a
`<protein-id>` element. RAG synthesis keeps validating its own output strictly.
No page numbers are fabricated: none exist in the RAG path and none are
derivable for acquired papers, so citations use `title (source_id) — section`
plus the verbatim passage.

**No biology or provenance check was weakened.** They all still run; they stop
aborting and become per-item review flags. Research mode is fail-open but never
fail-silent — every skipped FORMAT rule and every flagged violation keeps its
issue code and JSON pointer and is surfaced in the Review-flags panel.

**Separation invariant holds**:
`grep -rn "t2pw.rag" src/t2pw/pipeline src/t2pw/mapping src/t2pw/curation src/t2pw/pwml`
still returns nothing. Three core stage files gained a research-mode branch; it
keys on the export policy, never on RAG state, and is recorded as a sanctioned
exception in `docs/rag/03_separation_invariant.md`.

**Default-off byte identity**: with the policy at its `"pathwhiz"` default no
code path changes. Baseline before: **626 passing**; after: **711 passing**
(+85 research-mode tests), with every pre-existing test unmodified except
`tests/test_streamlit_stage2_orchestration.py`, whose AST-lifted namespace and
`normalize_process_payload` stub needed the new argument (the stub now asserts
the default run is strict). `ruff check src tests` reports the same 49
pre-existing findings as before the change — no new lint.

**Defects caught by an adversarial review of the tiering/report modules and
fixed before commit**, all of which would have misled a reviewer:

- **A quote could buy a Tier B.** `_tier_from_sources` added `+1` to the
  distinct-paper count for the *presence* of any unverified pointer — exactly
  the increment that crosses C into B. Since Stage 1 fills `source_refs` with
  verbatim quotes, one real paper plus one quoted sentence was reported as "2
  distinct sources". `distinct_paper_count` now counts only *identified* papers;
  Tier B still requires a second independent statement, and the reason string
  names the seed as the seed instead of inflating the count.
- **Tier A was unreachable on real data.** The pre-merge payload provably cannot
  carry an identifier (`synthesize._ALLOWED_ROW_KEYS` has no `mapped_ids`), so
  every genuinely UniProt-mapped enzyme was under-reported. `assign_tiers` gained
  `identity_source=` — the mapped payload, whose entity rows survive the merge —
  read for identifiers only. Both fake-identity sites are still refused through it.
- **Flags were mis-attributed by pointer prefix**: `/processes/reactions/1`
  swallowed every flag on `/processes/reactions/10..19`.
- **The UNSOURCED banner could never fire** (read `summary["tiers"]`, which does
  not exist, instead of `summary["tier_counts"]`), and the per-entity table
  rendered blank `label`/`sources` columns.
- **A research run stamped its report `pathwhiz`**, because the mode was not
  threaded into `build_citation_report`.

New tests: `test_research_mode_contracts.py` (format-vs-biology per mode,
structural guards, strict passthrough, `:` name leniency),
`test_research_mode_normalizer.py` (the three uncaught raises, the hidden
assert, the destructive passes), `test_research_mode_orchestration.py` (AST
guards that no `validate_post_*` call bypasses the mode wrapper and that Stage 6
drops wrapping only in research mode), `test_research_mode_tiers.py` (incl. the
Unknown-sentinel and best-effort Tier A regressions), and
`test_research_mode_report.py`.

## RAG contract-compliance fixes (2026-07-25, branch `rag-contract-fixes`)

Implements the approved fixes from `docs/contract_compliance_audit_2026-07-25.md`
for the regressions the RAG subsystem introduced against the PWML pipeline
contracts. Functional blockers **A** (wrapper regression), **B** (guardrails never
auto-run), and **C** (papers rejected) were implemented. Issue **E** was
implemented then **reverted** — see below. The prompt-injection hardening (issue
**D**) was **deliberately deferred** — it does not affect PWML correctness or RAG
function, is the most invasive (core + curation files), and none of these fixes
open a new injection surface (issue C routes user text through the
already-neutralizing `_format_user_task_context`).

An adversarial multi-lens review of the diff **caught a real defect the audit's own
§6.B eye-test missed**: narrowing the auto-trigger *gap kinds* was insufficient,
because the orchestrator feeds triage a `qa_graph` built on the **pre-mapping**
Stage-1 payload, where `unmapped_enzyme` (every enzyme is unmapped before Stage 6)
and the connectivity gaps (every pathway has open ends) are ubiquitous — so RAG
still auto-fired on essentially every pathway. The corrective fix (below, in B)
filters the auto-trigger to *reliable, post-resolution* gap sources; the effective
auto-trigger at the R0 seam is now `scope_clarity_score` plus the explicit flag.

No gate/contract was weakened; the separation invariant holds (no core stage
imports `t2pw.rag`); the `RAG_ENABLED` code default stays `False` (regression
firewall) and `RAG_ENABLED=false` behavior is byte-for-byte preserved (verified:
full suite green with `RAG_ENABLED=false`). Baseline before these fixes: 614 tests
passing; after: **626 passing** (+12 regression tests, incl. the pre-mapping
auto-trigger regression test).

### A. Protein-complex wrapper regression — stale `enzyme_complexes` cache

**Error.** With RAG-supplied enzymes, `validate_post_remap`
(`stage_contracts.py:226-273`) and the pre-export PWML gate (`pwml/ir.py:2246-2336`)
again raised `generated_wrapper_component_protein_unresolved` /
`generated_wrapper_component_missing_species` /
`generated_wrapper_component_missing_external_identity` — the "protein complex
wrapper bug is abck" recurrence (audit §2.A / §6.A). No source change reintroduced
it.

**Why.** `_resolve_complex_name` (`map_ids.py`) reused a cached `enzyme_complexes`
row verbatim on a hit, bypassing the DB re-resolution path and its guards.
`data/id_mapping_cache.json` held 19 legacy rows with top-level
`chosen_rule: "novel_enzyme_single_component_complex"`, `status: "unmapped"`, and
NO `generated` flag (they predate the flag). `is_generated_complex_wrapper`
correctly flags these as generated wrappers via its legacy `chosen_rule` fallback,
and `_merge_complex_resolution_into_row` copies `generated` only when present — so
every run re-flagged them as wrappers whose unmapped members failed the
wrapper-integrity gates.

**Fix (the data is bad, not the classifier).** (1) Purged all 19 stale
`enzyme_complexes` rows from `data/id_mapping_cache.json` (regenerable derived data;
other namespaces preserved byte-for-byte). (2) Added a durable read-side guard
`_is_reusable_complex_cache_row(row)` used at the cache read: a non-wrapper
(real DB-resolved) row is always reused; a generated-wrapper-shaped row is reused
only if it carries `generated: True`, has species context
(`protein_species_context`), and every component carries the exact UniProt/DrugBank
identity the gates require (`has_protein_external_identity` — matching the gate, not
gene-only; this was tightened from `_component_mapped_ids` after the adversarial
review flagged the gene-only divergence). Any other wrapper-shaped row is treated
as a cache miss and re-resolved, and the existing `cache.set(...)` self-heals the
stored entry — converting a downstream gate abort into an upstream self-heal. The
predicate is deliberately **conservative** (reuse only provably-good rows;
re-resolve everything else): because it checks the cache row's own component rather
than the declared payload protein the gates resolve to, it can only ever be
*stricter* than the gates, never looser — so it never lets a gate-failing wrapper be
reused, and re-resolution is idempotent/self-healing. Files:
`src/t2pw/mapping/map_ids.py`, `data/id_mapping_cache.json`,
`tests/test_map_ids.py` (+6 tests).

### B. RAG scope/gap guardrails were unreachable dead code

**Error.** With RAG enabled, the scope/gap auto-triage in `should_run_rag` could
never decide anything: the orchestrator only entered the RAG block when the UI
toggle was on (`if rag_config()["enabled"] and rag_incomplete_flag:`) and then
passed that flag as `user_flag`, which short-circuits `should_run_rag` — so RAG ran
iff the checkbox was on and never auto-started from the guardrails (audit §2.B /
§6.B).

**Why.** The gate double-guarded on the per-run toggle in addition to the deploy
switch, collapsing "auto-decide" into "user said so."

**Fix.** Dropped the `and rag_incomplete_flag` clause from the orchestrator gate
(`src/t2pw/app/streamlit_app.py`) → `if rag_config()["enabled"]:`. The
`rag_config()["enabled"]` guard is kept (RAG-off stays byte-for-byte today's
single-paper behavior); `user_flag=bool(rag_incomplete_flag)` is kept, so an ON
toggle still force-runs RAG while an OFF toggle lets `should_run_rag(user_flag=False)`
auto-decide from scope/gap signals. Toggle help text + gate comment rewritten to the
new "OFF = auto-decide" semantics.

**Load-bearing amendment (paired with the gate-open).** Opening the gate means a
toggle-OFF run calls `should_run_rag(user_flag=False)`, which must reliably tell an
incomplete pathway from a healthy one. The first attempt narrowed the auto-trigger
gap *kinds* to `dangling_reaction`/`unmapped_enzyme`/`missing_precursor`
(excluding the obviously-noisy `missing_compartment`/`orphan_metabolite`). The
adversarial review proved this insufficient: the orchestrator supplies triage a
`qa_graph` built on the **pre-mapping** Stage-1 payload, where `unmapped_enzyme`
(from `flags.missing_ids` — every enzyme is unmapped before Stage 6) and the
connectivity gaps `dangling_reaction`/`orphan_metabolite` (every pathway has an
entry substrate and terminal product = open ends) are all ubiquitous. RAG still
auto-fired on essentially every protein-containing pathway (reproduced:
`run=True, 'missing_precursor x3; unmapped_enzyme x17'` on a complete sample).

The **actual fix** is a *source* filter, not a kind filter: `_gap_signals`
(`src/t2pw/rag/triage.py`) now counts a gap toward the auto-trigger only when it
comes from a reliable **post-resolution** source (`_AUTO_TRIGGER_GAP_SOURCES =
{"gate", "mapping"}`), where "unmapped" means genuinely failed to resolve — not the
pre-mapping `qa_graph`/`payload` sources where the same signals are universal. The
kind narrowing is kept as a second filter. Gap *retrieval* once RAG is running is
untouched (it uses `retrieve.detect_gaps` directly on the full gap set). The
`scope_clarity_score < 0.5` trigger is unchanged and, since the orchestrator
currently supplies only a pre-mapping `qa_graph`, it (plus the explicit flag) is
the effective auto-trigger today; a reliable gap-based auto-start goes live the
moment a gate/mapping report is wired into the reports mapping. Tests:
`test_orphan_metabolite_report_...` / `..._missing_compartment_...` /
`..._dangling_reaction_qa_report_...` / `..._missing_precursor_qa_report_...` all
assert **no** auto-fire from a pre-mapping `qa_graph`;
`test_unmapped_enzyme_mapping_report_auto_triggers` asserts a `mapping`-sourced gap
still fires; and `test_pre_mapping_qa_report_does_not_auto_trigger` is the direct
regression test for the reproduced defect. Files:
`src/t2pw/app/streamlit_app.py`, `src/t2pw/rag/triage.py`,
`tests/test_rag_triage_orchestration.py`.

### E. `_EMPTY_CONTEXT` scope fields — implemented then REVERTED

Issue E (audit §2.E / §6.E) proposed adding `document_type`/`scope_status` to
`_EMPTY_CONTEXT` so a failed Stage-0 context is distinguishable from a clear one.
It was implemented (both keys, truthiness-guarded in `format_context_header`) and
then **reverted** after the adversarial review, for two reasons: (1) it fixes
nothing functional — triage's low-clarity trigger reads `scope_clarity_score`, which
§6.E deliberately does NOT add, so a failed preprocess still (correctly) does not
auto-fire RAG with or without these keys; and (2) it was the only change with a
byte-identity wrinkle — `completeness_audit._build_user_prompt` does
`json.dumps(preprocessor_context)` over the whole context, so on Stage-0-failure
paths the two new empty keys would appear in that core (non-RAG) audit prompt,
technically breaking the "RAG-off byte-for-byte" guarantee. Reverting keeps the
change surgical and preserves strict byte-identity. `src/t2pw/pipeline/preprocessor.py`
is therefore unchanged on this branch.

### C. Related papers were structurally un-selectable (RAG kept ~0 papers)

**Error.** RAG kept almost no candidate papers (audit §2.C / §6.C), due to two
compounding mechanisms in `src/t2pw/rag/select.py`.

**Why + fix.**
- **C1 — candidates re-preprocessed with no scope.** `_candidate_context` re-ran
  Stage-0 `preprocess` on each candidate abstract without scoping context, so a
  review abstract deterministically fell to Case C (`multi_example_review`, blank
  scope), found no matching example, and ate `_REVIEW_PENALTY_NO_MATCH = 1.0` →
  driven below the `_MIN_SCORE = 0.0` floor → dropped. Fix: added `user_task_context`
  kwarg to `_candidate_context`/`score_candidate`, forwarded to `preprocess`;
  `select()` derives `seed_focus` once (`selected_example` else `pathway_name`) and
  threads it per candidate. A review about the seed pathway now reaches Case B
  (matched example) → `_REVIEW_PENALTY_MATCH = 0.25` (rank-below) instead of a
  force-drop.
- **C2 — sparse seed collapsed scoring.** When the seed is novel/ambiguous (blank
  pathway/compounds/proteins + blank organism → `total_terms == 0`, `seed_org == ""`),
  it cannot prove a review off-topic. `score_candidate` now downgrades the 1.0
  no-match penalty to 0.25 in exactly that case
  (`seed_has_signal = total_terms > 0 or bool(seed_org)`), gating the penalty — not
  `_MIN_SCORE`, so zero-signal *primary* papers still drop. When the seed has signal
  (the norm) the full 1.0 penalty is preserved.
  C3 (`acquire.search_candidates` short-circuits to `[]` on a term-less seed) is left
  unchanged by design. Files: `src/t2pw/rag/select.py`, `tests/test_rag_select.py`
  (+3 tests).

---

## Known Debt — Runtime Contract Audit (2026-07-24, diagnostic; NOT a fix)

A read-only sweep ran the runtime payload schema
(`validate_payload_shape` / `validate_runtime_payload_contract`,
`src/t2pw/pipeline/payload_models.py`) in **enforce** mode against every real
on-disk payload artifact under `tmp/`, at every applicable boundary. Purpose:
surface the pipeline's latent contract violations all at once instead of
discovering them one production crash at a time. The wider pipeline runs this
schema in `report` mode (`RUNTIME_SCHEMA_MODE = "report"`,
`stage_contracts.py`), so these shapes are silently tolerated today.

**Headline:** the schema is in good shape. Modern artifacts validate clean at
enforce level for post_extraction (`stage1_raw_extraction.json`), post_mapping
(`stage2.mapped.json`) and post_normalization; all 13 RAG-generated
protein-complex wrappers in `final.mapped.json` pass the generated-wrapper
contract. **5 distinct violation classes** were found — **2 live, 3 stale**
(only in an older 2026-06-11 run; current producers appear to emit the correct
shape) — plus 1 in-progress class already fixed elsewhere this session.

**LIVE (worth a small, deliberate follow-up — not firefighting):**

1. **`reactions[].evidence` is a `List[Dict]`, not `str`.** 9 rows in the RAG
   multi-paper `tmp/final.mapped.json`. Producer: `src/t2pw/rag/synthesize.py`
   (reaction `evidence` built as a list of `{text,...}` provenance snippets).
   Boundary-independent (pydantic `PayloadModel`), so it nominally fires at all
   7 boundaries. **Fix direction: loosen schema** — structured multi-paper
   evidence is intentional and load-bearing; widen `evidence` on `ReactionModel`
   (and peer evidence-bearing models) to `str | List[Dict[str, Any]] | None`
   rather than flatten away source pointers. *Caveat: `final.mapped.json` is
   timestamped before this session's conform/merge fix (which coerces evidence
   list→string at the S3 boundary), so this may be a STALE artifact rather than
   a live leak — confirm against a fresh multi-paper run before acting.*

2. **Auto-injected `subcellular_locations` row has no `mapping_meta`.** Present
   in the newest `final.audited.json` / `final.json`. Producer:
   `src/t2pw/pipeline/process_normalizer.py::ensure_autostates` (~line 2650:
   `subcellular_locations.append({"name": auto_location_name})`). The mapping
   contract requires every named entity to declare how it was resolved.
   **Fix direction: tighten producer** — attach a synthetic `mapping_meta` with
   `resolution.status = "auto_default"` so the injected default compartment is
   self-describing.

**STALE (older 2026-06-11 run only; current producer looks correct — want a
regression test, not a code change):**

3. `actor_schema_not_canonical` on `reactions[].enzymes[*]` (30 rows) — raw
   `protein`/`protein_complex` actor keys not canonicalized to
   `entity`/`entity_type`. Modern enzymes carry the canonical shape; canonicalizer
   is `process_normalizer.py::normalize_process_actor_schema`.
4. `actor_schema_not_canonical` on `transports[].transporters[*]` (2 rows) — same
   root cause as #3.
5. `mapping_resolution_required` on `species[*].mapping_meta.resolution` (1 row) —
   older path wrote `mapping_meta.species_resolution` instead of `resolution`.
   Modern species validate clean.

**IN-PROGRESS:** `interactions[].name` missing — fixed this session (see the
differential-gate entry below). Not reproducible from current artifacts (no
on-disk payload has an unnamed interaction), so it remains latent w.r.t. these
fixtures.

**Coverage caveats — do NOT read this as full coverage:** only ~3 pipeline runs
on disk, all single-pathway; **no dedicated `post_remap` / `post_enrichment`
artifacts** (approximated via `final.json`); this sweep covers only the runtime
schema, **not** the stricter hand-written contracts in `stage_contracts.py`
(`validate_post_remap`, `validate_pre_export` PWML contract); and rare row types
(`nucleic_acids`, `element_collections`, `bounds`, `reaction_coupled_transports`,
`sub_pathways`, `tissues`, `cell_types`, all `visualizations`) were empty in every
artifact, so their row-level contracts are effectively untested. Full inventory:
`scratchpad/contract_debt_inventory.md`.

---

## Fixed

---

### FIXED — Stage 0 empty context silently disabled multi-paper RAG; now falls back to the user's scope

**Files changed:** `.env`, `src/t2pw/app/streamlit_app.py`,
`tests/test_stage0_scope_fallback.py` (new), `tests/test_preprocessor.py`,
`docs/change_log.md`

**Error / symptom:** On repeated runs the preprocessor (Stage 0) returned a valid
but empty context (status `ok`, no `pathway_name` / `likely_organism` / compounds
/ proteins). That silently gutted multi-paper RAG — `t2pw.rag.acquire.build_query`
returns `""` for an empty context, so 0 candidate papers were fetched ("RAG
complete: 1 reaction from 0 papers") — and left extraction unguided. Extraction
and inference (on `deepseek/deepseek-v4-flash`) were unaffected on the same runs.

**Why it appeared:** Two independent causes.

1. *Weak Stage-0 model.* `OPENROUTER_PREPROCESSOR_MODEL` was
   `google/gemma-4-26b-a4b-it`, a smaller model than the extraction/inference
   model. Forced to emit a JSON-object summary of a dense 14-page paper it kept
   returning the empty template. Not quota (extraction shared the account and
   worked) and not length (the existing retry on a 20k-char head also failed) —
   the model itself was the weak link. Switched
   `OPENROUTER_PREPROCESSOR_MODEL` to `deepseek/deepseek-v4-flash` (the working
   extraction model).

2. *No fallback for a scope the user already provided.* Stage 0 was the SOLE
   source of the pathway scope, with no fallback — even though the user routinely
   types it into the extraction-focus box (e.g. "menaquinone biosynthesis in
   Lactococcus lactis"). A single flaky preprocessor call could therefore nuke a
   scope the pipeline already had in hand.

**How the fix is consistent with the pipeline design:** New helper
`_seed_context_from_user_scope(ctx, user_scope)` in `streamlit_app.py`. When
Stage 0 returns no usable context AND the run is not a deterministic Case C
ambiguous review, it derives a `pathway_name` (and, from a trailing
`" in <organism>"` of ≤3 words, a `likely_organism`) from the user's focus-box
text and augments the context so `_has_usable_context` passes and RAG can build a
query. It returns an augmented COPY (all original keys preserved) or `None` when
there is no scope to seed from — in which case the original "Stage 0 failed"
warning still shows (reworded to point at the focus-box workaround). The seeded
scope is surfaced verbatim via `st.info` so a heuristic mis-parse is visible, not
silent. A long "in <phrase>" tail (e.g. "in vitro reconstitution …") is kept as
the pathway name rather than mistaken for an organism, so it can't zero out the
literature query.

**Tests:** `tests/test_stage0_scope_fallback.py` (5) covers scope-with-organism,
scope-without-organism, the long-tail guard, blank/None scope → `None`, and
"don't overwrite an existing organism." `tests/test_preprocessor.py`'s
transient-warning guard test was updated to assert the INVARIANT (the warning is
enclosed by the `_has_usable_context` + `is_ambiguous_multi_example_review_context`
guard) rather than a now-nested exact phrase — same protection, robust to the
fallback branch.

---

### FIXED — S3 RAG-merge schema gate false-positived on the seed's own tolerated shapes (nameless Stage-2 interactions)

**Files changed:** `src/t2pw/app/streamlit_app.py`,
`src/t2pw/pipeline/pipeline.py`, `tests/test_rag_differential_gate.py` (new),
`docs/change_log.md`

**Error / symptom:** A valid multi-paper RAG merge was rejected at seam S3 with a
"RAG-merge schema failure" `st.error`, keeping the single-paper extraction, even
though the merge itself was clean. The offending rows were
`processes.interactions` entries with no `name` field — which did NOT come from
RAG.

**Why it appeared:** The S3 gate was the ONLY caller running the post-extraction
runtime schema in `enforce` mode; the rest of the pipeline runs it in `report`
mode (`RUNTIME_SCHEMA_MODE = "report"` in `stage_contracts.py`). `InteractionModel`
(`payload_models.py`) inherits a REQUIRED `name` from `NamedRecordModel`, but
`clean_inference_output` → `_clean_processes` (`pipeline.py`) only attached a
`name` to an interaction when one was already present, so Stage-2 inference
emitted nameless interactions that flowed through single-paper runs untouched.
The seed therefore legitimately carried a shape the strict schema flags, and the
enforce-mode gate held the merged payload to a stricter bar than the seed it
merged into — a false positive on a pre-existing shape, not a RAG defect.

**How the fix is consistent with the pipeline design:** Two contained changes.

1. *Differential gate* (`streamlit_app.py`). Two module-level helpers,
   `_post_extraction_error_keys(payload)` and
   `_post_extraction_new_error_keys(seed, merged)`, run the same
   `validate_runtime_payload_contract(..., boundary="post_extraction",
   mode="enforce")` inside a `try/except StageContractError` and return the set of
   `(code, normalized_pointer)` error identities — array indices in the JSON
   pointer collapsed to `/*` (`re.sub(r"/\d+", "/*", pointer)`) so a row shifting
   position between the seed and the merge does not read as a new error. The gate
   snapshots the seed's error keys BEFORE reassigning `final_payload = merged` and
   rejects only when the merge INTRODUCES a NEW violation
   (`new_keys = merged_keys - seed_keys`). If `new_keys` is empty the merge is
   adopted with the existing `st.info`; if non-empty the existing `st.error` +
   `st.json(failure.report)` reject path runs and the single-paper extraction is
   kept. All prior messaging/flow is preserved.

2. *Root cause* (`pipeline.py`). In the `_clean_processes` interaction loop, an
   interaction with usable `entity_1`/`entity_2` endpoints but no `name` now gets
   a DETERMINISTIC synthesized name — `f"{e1} {relationship} {e2}"` when a
   relationship is present, else `f"{e1} - {e2}"` — so the payload is genuinely
   schema-clean at its source rather than merely tolerated. Interactions that
   already have a name are untouched; no interaction is dropped.

**Tests:** `tests/test_rag_differential_gate.py` proves (a) a merge whose only
violation also exists in the seed is adopted (empty new-key set), (b) a merge
with a brand-new violation is rejected, and (c) `clean_inference_output`
synthesizes the deterministic name (with and without a relationship) while
preserving an existing name. No existing test asserted nameless interactions, so
none needed weakening.

---

### FIXED — Multi-paper RAG: synthesized payload merged into the seed instead of replacing it (fixes post-pipeline 'NoneType' crash)

**Files changed:** `src/t2pw/app/streamlit_app.py`, `src/t2pw/rag/conform.py` (new),
`tests/test_rag_conform_merge.py` (new), `docs/change_log.md`

**Error / symptom:** With multi-paper RAG on, the run crashed in the
post-pipeline audit / DB-mapping path with `'NoneType' object is not
subscriptable`. Single-paper runs were unaffected.

**Why it appeared:** At seam S3 the RAG-synthesized payload REPLACED the rich
single-paper (Stage-1) payload wholesale. RAG's `to_payload`
(`src/t2pw/rag/synthesize.py`) emits a *thin, reaction-only* payload that is
MISSING the shape the downstream assumes: no `element_locations`, no per-reaction
`compartment` / `biological_state`, no per-entity species / `mapping_meta`, no
`protein_complexes`, `evidence` typed as a LIST instead of a string, and
participants sometimes `{"name", "stoichiometry"}` dicts. A single-paper payload
carries all of this, so the audit / mapping / IR path read fields RAG never
emitted and dereferenced `None`.

**How the fix is consistent with the pipeline design:** Instead of REPLACING the
seed, the seam now INJECTS RAG's new reactions / entities INTO the seed with the
SAME additive machinery Stage 2 uses — `merge_additions`
(`src/t2pw/pipeline/pipeline.py`). A new import-light helper,
`conform_rag_additions_for_merge` (`src/t2pw/rag/conform.py`), first conforms the
synthesized payload into an `{"additions": {...}}` envelope, coercing each row to
the core shape: `evidence` list → string (on reactions AND enzyme actors, matching
`ReactionModel.evidence: str` / `ActorModel.evidence: str`), reaction participants
→ plain name strings (the core representation the whole pipeline uses —
`clean_inference_output`, `draft_graph`, and `_reaction_io_key` all treat
participants as strings; stoichiometry is not carried on reaction participants
anywhere, so no new shape is invented), and the carried-forward scaffolding
buckets (species / subcellular_locations / cell_types / tissues) excluded so they
are not double-added. `element_locations` and per-reaction `compartment` are
deliberately NOT emitted — they are left for the existing `default_compartment="cell"`
/ compartment-backfill path to populate on the merged, seed-shaped payload,
exactly as for a single-paper run.

After the merge, the manifest-aware `apply_post_merge_cleanup` runs on the merged
result (with the locked-reaction manifest and quarantine output path) so a locked
Stage-1 reaction that synthesis mangled is quarantined — and the quarantine file
written — rather than silently dropped, preserving the behavior of the old
replace-path. The old "wipe protection" count guard is replaced by MERGE
semantics: if the merge added ≥ 1 new reaction the merged payload is adopted (with
an `st.info` of how many were merged / dropped-or-quarantined); if it added zero
new reactions the seed is kept unchanged with the existing "no additional usable
reactions" `st.warning`. Finally a schema gate at the RAG boundary
(`validate_runtime_payload_contract(..., boundary="post_extraction", mode="enforce")`)
validates the merged payload; a failure surfaces a clear "RAG-merge schema
failure" `st.error` and keeps the single-paper extraction rather than passing a
malformed payload downstream — it does not hard-crash the app.

---

### FIXED — Streamlit: multi-paper RAG phase ran with no UI feedback, so the app looked frozen

**Files changed:** `src/t2pw/app/streamlit_app.py`, `docs/change_log.md`

**Error / symptom:** After Stage 1 extraction the app appeared to "just stop" —
a blank screen with no output — and never reached the "Running Stage 2
inference…" spinner. LM Studio kept logging embedding requests, so the process
was clearly alive, and the run eventually completed ("working randomly"). The
symptom was purely missing progress: the app was not frozen, it was working.

**Why it appeared:** `maybe_run_rag` runs the entire RAG chain
(acquire → select → ingest → retrieve → **synthesize**) synchronously with zero
`st` output between Stage 1 and the Stage 2 spinner. With
`RAG_EXTRACT_REACTIONS` on, the synthesize step performs prose-reaction
extraction — one sequential LLM chat call per retrieved evidence passage
(`retrieve_top_k=8` × one gap per unlinked metabolite) — while LM Studio also
has to swap in the chat model. That is minutes of real work behind a blank UI.
No timeouts are configured, so nothing fails fast either. This is not a
Streamlit bug and is unrelated to the Stage 3/Stage 6 normalization fix above
(that code runs *after* Stage 2, which was never reached).

**How the fix is consistent with the pipeline design:** Added a single
`st.status` container inside `maybe_run_rag` that narrates each phase (papers
fetched / selected / passages indexed / gaps detected / per-gap retrieval / and
a per-passage counter for the slow prose-extraction step) and is marked
`complete`/`error` on exit. The prose extractor is wrapped so each call ticks the
status label — the exact step that previously left the screen blank. It is
best-effort UI only: it adds no control flow and cannot raise into the run,
preserving the "RAG must never break the core run" contract. A follow-up option
(not taken here) is to add generous per-call timeouts on the RAG network/LLM
calls so a genuinely stuck call aborts to Stage 2 instead of blocking.

---

### FIXED — Stage 3 gate: source enzyme whose name already ends in ' complex' left as a bare protein (QTRT1/QTRT2 complex)

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/mapping/map_ids.py`, `tests/test_process_normalizer.py`,
`docs/change_log.md`

**Error / symptom:** The strict Stage 3 gate
`run_strict_post_normalization_gates` aborted normalization with
`/entities/proteins/N -> Generated protein complex wrapper 'QTRT1/QTRT2 complex'
must be listed under protein_complexes, not proteins.` The offending row was an
ENZYME actor for the queuosine biosynthesis pathway whose *source* name already
ended in " complex" (a heterodimeric, slash-joined subunit complex), so it sat in
`entities.proteins` and tripped the gate's `endswith(" complex")` guard.

**Why it appeared:** This is a NEW member of the previously-fixed wrapper-leak bug
class, surfaced by RAG-assembled multi-paper pathways introducing enzyme actor
names that already end in "complex" and/or contain slash-joined subunits — inputs
the pre-RAG fixtures never exercised. Two independent defects combined:

1. In `map_ids.py`, the single-protein PathWhiz wrapper generator built the
   wrapper name as `f"{protein_name} complex"` at three sites. When
   `protein_name` already ended in " complex", this produced a DOUBLED
   `QTRT1/QTRT2 complex complex` wrapper and left the original complex-named
   entity behind in `entities.proteins`.
2. In `process_normalizer.py`, every existing repair guard only prevented a
   GENERATED wrapper (already in `protein_complexes`) from leaking back into
   `proteins`; each checks `_find_entity_row(complexes, actor_name)`. None fired
   for a bare inbound source protein like `QTRT1/QTRT2 complex`, because that name
   is not itself in `complexes` (only the doubled `QTRT1/QTRT2 complex complex`
   is). The inbound-source-name case was a genuine gap.

**How the fix is consistent with the pipeline design:** Two contained changes,
matching the existing defense-in-depth layering (root-cause guard at the source,
authoritative normalizer safety net downstream):

- *Root-cause guard (`map_ids.py`).* A new module-level helper
  `_wrapper_complex_name(protein_name)` names the wrapper without doubling
  " complex" when the source name already ends in it; it replaces the three
  `f"{protein_name} complex"` constructions. This stops NEW doubled wrappers from
  being produced or cached. No other mapping logic changes.
- *Normalizer fallback pass (`process_normalizer.py`).* A new pass
  `relocate_complex_named_proteins`, wired in after
  `normalize_process_actor_schema` and before
  `drop_unresolved_complex_component_proteins`, repairs both freshly-fed and
  already-cached doubled names. For each protein whose canonical name ends in
  " complex" (and is not a biochemical colon name) it: resolves or creates the
  target `protein_complex` (collapsing any mapping-doubled `... complex complex`
  wrapper in place), renames the surviving protein to its subunit form while
  preserving external identity (UniProt/DrugBank) and species, sets the complex's
  single component to reference that subunit carrying the external identity, and
  rewrites every actor / interaction / transport / protein-location / component
  reference norm-based (`doubled_name` -> `complex_name`), marking any actor that
  now resolves to the complex with `entity_type: "protein_complex"`. It reuses the
  existing helpers (`_entity_lists`, `_find_entity_row`, `_remove_entity`,
  `_dedupe_named_rows`, `_normalize`, `_is_biochemical_colon_name`,
  `protein_species_context`) and the `report["summary"]` /
  `report["actions"]` bookkeeping conventions, adding a
  `complex_named_proteins_relocated` counter. The normalizer pass is the
  authoritative net (it also repairs legacy cached doubles); the mapping guard
  simply stops producing new ones.

---

### 2026-07-23 — Stage 0 replies truncated mid-JSON were discarded whole

**Files changed:** `src/t2pw/pipeline/preprocessor.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_preprocessor.py`,
`docs/change_log.md`

**Error / symptom:** With the Stage 0 diagnostic in place, a run reported
`Stage 0 failed: returned unparseable JSON (1373 chars): { "document_type":
"multi_example_review", … "scope_status": "targeted", "pathway_name":
"Clostridioides difficile queuosine salvage route", …`. The model had produced
the **correct** answer — Case B fired and the scope was targeted — but the reply
was cut off mid-object, so the entire context was discarded, every pathway field
came back blank, and RAG built an empty literature query and fetched 0 papers.

**Root cause:** Two parts.

1. *Output budget too small.* `preprocess()` requested `max_tokens=500` while
   `chat()`'s own default is 800. The Stage 0 output contract is large — Case B
   must emit `selected_example`, up to ten `candidate_examples` (each with five
   sub-fields), `excluded_examples`, and every standard field — so 500 output
   tokens truncates an ordinary targeted reply (1373 chars ÷ 500 ≈ 2.7
   chars/token, a textbook cap hit).
2. *No salvage.* `_parse_json` handled code fences and trailing commas but had no
   repair for an unclosed object, so a reply whose tail was cut returned `None`
   even though its leading fields were complete and valid.

**Fix:** The Stage 0 output budget is raised to 2000 tokens (still a parameter so
callers can override; `chat()`'s default is untouched). `_parse_json` gains a
final repair pass that runs only after the existing attempts fail: one
left-to-right scan tracks string-literal state and backslash escapes so a `{`,
`}`, `[`, `]` or escaped quote *inside a value* can never be read as structure;
it records only provable element boundaries as cut candidates and tries them
newest-first with the still-open containers closed in reverse order. A bare token
running to end-of-input is deliberately never a boundary, so a truncated `0.95`
is dropped rather than silently parsed as `0.9`.

**Pipeline consistency:** Recovery is never silent — a repaired result carries
`recovered: True` in the `preprocess_status` diagnostic along with the original
raw length, a "some fields may be missing" detail, and a distinct Streamlit
warning, so a clean parse is always distinguishable from a salvaged one. A
property check over *every* truncation point of a representative Stage 0 object
(725 cut points) confirmed the repair never invents a key and never alters a
value: 683 points recovered a strict subset of the original, 42 returned nothing,
zero violations.

---

### 2026-07-23 — Stage 0 never received the user's scope context, and its failures were indistinguishable

**Files changed:** `src/t2pw/pipeline/preprocessor.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_preprocessor.py`,
`docs/change_log.md`

**Error / symptom:** On a multi-example review (the PNAS queuosine salvage
paper), Stage 0 reported `Ambiguous review scope: … no target example was
selected` with `candidate_examples` populated and every pathway field blank, so
RAG built an empty literature query and fetched 0 papers. Naming a target example
in the "Optional extraction focus / task context" box changed nothing. A later run
instead reported `Stage 0 (preprocessor) returned no usable context` with no
candidate examples at all — a different failure that looked identical in the UI.

**Root cause:** Two defects on the same surface.

1. *Scope context never reached Stage 0.* `preprocess_system.txt` branches on a
   `<user_task_context>` / `<pathway_scope>` block (Case B — "a specific example
   IS named"), but `preprocess()` had no such parameter and its user message
   carried only the document text. The app collected `user_task_context` and
   passed it to Stage 1 and later stages — never to Stage 0. Case B was therefore
   structurally unreachable, and every `multi_example_review` fell through to
   Case C, which *deliberately* blanks `pathway_name` / `key_compounds` /
   `key_proteins` rather than merge examples. No prompt wording could fix it.
2. *Failures were indistinguishable.* `preprocess()` fails closed: an API
   exception, a non-JSON reply, and a genuinely empty result all returned the same
   `_EMPTY_CONTEXT`, with only a `logger.warning` the UI never surfaces. The
   generic "usually a transient LLM failure" warning also fired on the deliberate
   Case C guardrail, where the blank fields are by design, the outcome is
   deterministic, and re-running never helps.

**Fix:** `preprocess()` takes an optional `user_task_context` and prepends it as
an escaped `<user_task_context>` block ahead of the document text; both call sites
forward it, **including the long-document retry** that would otherwise silently
drop the scope. The close-tag escape is replicated from
`pipeline._format_user_task_context` rather than imported, because `pipeline`
already imports `preprocessor` and importing back is a circular import. Every
result now carries a `preprocess_status` diagnostic (`ok` / `llm_error` /
`unparseable` / `empty_reply`, with a 200-char capped raw preview), written
*after* the model's own keys are merged so a model reply containing that key can
never masquerade as the real status; the Stage 0 warning reports it. The
transient-failure warning is suppressed on the ambiguous-review branch, which
raises its own error, and that error now tells the user to name a target example.

**Pipeline consistency:** With `user_task_context` absent, the Stage 0 messages
are byte-identical to before, so single-paper runs are unchanged. Case B is now
reachable exactly as the prompt already documented — verified against the PNAS
paper, where Stage 0 returns `scope_status: "targeted"` with the selected example
and organism populated instead of a blank context. `preprocess()` still returns
every `_EMPTY_CONTEXT` key on every path. Because the diagnostic carries an
untrusted raw model reply, the one seam that serialized the whole context into
another prompt (`completeness_audit`'s `json.dumps(preprocessor_context)`) now
receives a stripped copy, so a raw model reply can never re-enter an LLM prompt.

---

### 2026-07-23 — RAG under-merged cross-paper synonym duplicates

**Files changed:** `src/t2pw/rag/synonyms.py` (new), `src/t2pw/rag/synthesize.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_rag_synonym_merge.py` (new),
`docs/change_log.md`

**Error / symptom:** With RAG enabled, the synthesized pathway carried redundant
reactions and entity nodes that describe the same chemistry under different
names — e.g. the seed's `LpxA reaction` on `UDP-N-acetylglucosamine` plus a
retrieved twin on `UDP-GlcNAc` (the same molecule, ChEBI:16264) survived as two
separate reactions, and the two names appeared as two distinct species. A
12-reaction seed inflated toward ~21 reactions / ~50 species. This is the "RAG
creates entities in an unexpected manner" symptom that surfaced alongside the
locked-reaction gate failure (see the sibling entry below); the single-paper
pipeline never hit it because one paper names a compound consistently — synonym
reconciliation is inherently a multi-paper (RAG) concern.

**Root cause:** `synthesize` reconciles cross-paper synonyms only through the
core's small hand-curated `BIOCHEMICAL_ALIAS_MAP` (`canonical_name`). Any synonym
absent from that map never collapses, so `_Reaction.conflict_key` /
`signature` (reaction grouping) and `_build_entities` (entity grouping) — all
keyed on casefolded literal names — treated `UDP-GlcNAc` and
`UDP-N-acetylglucosamine` as different nodes and never merged the reactions built
on them. The project already resolves these synonyms elsewhere: the ID-mapping
cache (`data/id_mapping_cache.json`) records both names against the same external
IDs (`CHEBI:16264` / `C00043`), but nothing fed that knowledge into synthesis.

**Fix:** New `t2pw.rag.synonyms.build_offline_synonym_resolver` reads the existing
`id_mapping_cache.json` and builds an offline `normalized name -> stable-ID token`
index (first present of chebi/kegg/hmdb/pubchem/cas/pathbank id), so two names
sharing an external ID resolve to one namespaced grouping token. It is injected
into `synthesize_with_report` / `synthesize` as an optional `synonym_resolver`
(default `None` ⇒ prior behavior byte-for-byte, mirroring the `prose_extractor`
seam) and threaded into `conflict_key`, `signature`, and `_build_entities`.
Crucially it is **grouping-only**: it feeds the merge/dedup KEYS but never
rewrites an emitted reaction or entity name — survivors keep the merge winner's
real names (seed/locked rows win by evidence weight), so locked-reaction matching
on raw names is unaffected. The real resolver is wired at the single caller
(`streamlit_app.py`). Offline and deterministic (pure file reads, no network / no
live LLM); a `llm_fallback` extension hook exists for names the cache misses but
is default-off to preserve those guarantees.

**Pipeline consistency:** The default (`synonym_resolver=None`) path is unchanged,
so single-paper runs and the full suite behave exactly as before. Grouping-only
merging leaves the locked-reaction accounting intact — verified end-to-end that
all 12 locked reactions still export with `unaccounted_locked_reactions == 0`
with the resolver active. Scope is deliberately bounded: participants that are
unmapped placeholders (`status == "unmapped"`, e.g. `LpxA product`) share no ID,
so they correctly do **not** merge, and genuinely new cross-paper reactions are
untouched — collapsing placeholder restatements is left to a separate
prose-extraction quality gate.

---

### 2026-07-23 — Reversible reaction dropped a locked direction (accounting gate failure)

**Files changed:** `src/t2pw/rag/synthesize.py`,
`tests/test_rag_reversible_reaction_preservation.py` (new),
`tests/test_rag_synthesize.py`, `docs/change_log.md`

**Error / symptom:** With RAG enabled, the post-normalization hard gate aborted
the run with `Locked reaction accounting failed: 1 locked reaction(s) are neither
active nor quarantined` (raised in `run_strict_post_normalization_gates`, written
to `tmp/gate_fail_report.json`). The lock report held 12 locked reactions found,
11 exported, 0 quarantined. The missing one was `rxn_lock_002`
(`LpxA reverse reaction`) — the exact reverse of `rxn_lock_001`. Preservation
reports showed it already gone at the Stage-2 checkpoint (the RAG seam), not in
normalization or audit.

**Root cause:** `_Reaction.conflict_key` grouped reactions by the *unordered set
of all participants* (inputs and outputs merged). A reversible reaction given as
an explicit forward + reverse pair has the identical participant set — only the
input/output sides are swapped — so both directions keyed to the same group.
`_resolve_reactions` then saw two different `signature()`s, declared them
conflicting variants of one reaction, kept the higher-evidence-weight direction
and dropped the other as a mere "conflict alternative." The dropped direction
carried a locked-reaction id, so it later showed up as unaccounted at the gate.
Pre-RAG this could not happen: the core `dedupe_processes` keys sorted inputs and
sorted outputs as *separate* slots, so a forward/reverse pair produces two
distinct keys and both survive. The RAG resolver's merged-set key lost that
distinction.

**Fix:** `conflict_key` is now direction-aware — keyed on the
`(sorted input names, sorted output names)` pair, matching the core
`dedupe_processes` key that the single-paper pipeline already relied on. Opposite
directions key to distinct groups and both survive; same-direction disagreements
(stoichiometry / compartment / reversible flag) are still grouped and resolved by
evidence weight. The existing `test_conflict_resolved_by_evidence_weight`, which
had encoded the buggy opposite-direction collapse (`A -> B` vs `B -> A`), was
re-pointed at a genuine same-direction stoichiometry conflict so the weight-based
resolution path stays covered rather than deleted.

**Pipeline consistency:** Restores the pre-RAG behavior — both directions of a
reversible reaction preserved — inside the RAG synthesis path, without touching
any core stage module (RAG → core dependency arrow unchanged). Verified
end-to-end that the reverse reaction is re-tagged as `rxn_lock_002` and all 12
locked reactions account to `unaccounted_locked_reactions == 0`, clearing the
gate that raised the error.

---

### 2026-07-22 — RAG-synthesized payload bypassed post-merge hardening

**Files changed:** `src/t2pw/app/streamlit_app.py`,
`src/t2pw/pipeline/pipeline.py`, `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/mapping/map_ids.py`, `src/t2pw/rag/extract.py`,
`src/t2pw/rag/ingest.py`, `src/t2pw/rag/synthesize.py`,
`tests/test_rag_payload_gate_guardrails.py`

**Error / symptom:** With RAG enabled, the PWML required-field gate failed the
export with 6 errors: `reaction_missing_right_participants` on reactions 5 and
19 (named `R-3-hydroxymyristoyl-ACP -> ?` and `UDP-N-acetylglucosamine -> ?`),
and `reaction_missing_left_participants`, `reaction_missing_right_participants`
plus two `duplicate_reaction_enzyme_complex` on reaction 23,
`(3R)-hydroxymyristoyl acyl carrier protein dehydration`. Stage 1 had extracted
5 well-formed reactions; the payload reaching the gate held 25.

**Root cause:** Four defects, all on RAG-added rows.

1. *Ordering.* `merge_additions` ends by hardening the merged payload
   (`_normalize_reaction_actors` for actor dedup, `filter_unresolvable_reactions`
   for empty/unresolvable sides). The orchestrator then replaced that hardened
   payload wholesale with `rag_result.payload` on a bare reaction-count
   comparison, so no RAG row was ever hardened. The guards that previously fixed
   this error class were present and correct — they simply never ran.
2. *One-sided transcription.* The three RAG reaction builders dropped a candidate
   only when **both** sides were empty, so a passage naming a substrate but no
   product became a live reaction; the name fallback rendered it `"<name> -> ?"`.
3. *Catalyst promotion.* `promote_catalysts` deduped only against `modifiers`
   (never `enzymes`) and stripped protein-like tokens out of `inputs` without
   checking anything survived — which is what emptied reaction 23.
4. *Post-mapping collision.* Two distinct actor names can resolve to one PathWhiz
   complex during Stage 6 wrapper rewriting, and nothing deduped afterwards.

Reaction 23 existed at all because the reference list was mined as prose: the
back matter carried no section label, so it was absorbed into the preceding body
section, and cited title #53 became a "reaction". Reaction 24 is the same defect
with a luckier parse — it cleared the gate as bogus content.

**Fix:** Extracted `apply_post_merge_cleanup` in `pipeline.py` and applied it to
the adopted RAG payload *before* the count comparison, threading the lock
manifest so a mangled locked reaction is quarantined rather than deleted. The
RAG builders now require both sides. `promote_catalysts` dedupes against both
actor collections and refuses to hollow out a reaction, deferring the verdict to
`filter_unresolvable_reactions`. Stage 6 merges actors that collide after
renaming, preserving their evidence. Ingest labels `references` /
`acknowledgments` (matched on the last occurrence, tail-anchored) and never
chunks them, with a citation-density backstop in synthesis for papers whose
headers do not parse.

`_inject_name_based_modifiers` is deliberately excluded from the shared cleanup:
it substring-matches protein names against a row's evidence, and a RAG row's
evidence is a list of retrieved passages rather than one sentence. Running it
over that corpus attached every enzyme to every reaction — 99 spurious
`reaction_enzyme_must_be_protein_complex` errors. It now reads string evidence
only and stays in the merge path.

**Pipeline consistency:** Every payload entering the post-pipeline path is
hardened by the same function, whether it came from Stage 2 merging or from
multi-paper synthesis. Replaying the failing payload takes it from 6 errors to a
clean gate, removing 3 unusable reactions and keeping the other 22.

---

### 2026-07-21 — RAG prose→reaction extraction (closes the arrow-only limitation)

**Files changed:** `src/t2pw/rag/extract.py` (new), `src/t2pw/rag/synthesize.py`,
`src/t2pw/config.py`, `src/t2pw/app/streamlit_app.py`, `.env`,
`tests/test_rag_extract.py` (new), `docs/change_log.md`.

**What was the limitation:** the RAG deep-dive entry below flagged that evidence
reactions were transcribed **only** from arrow-style equations
(`synthesize._parse_reaction_line`, e.g. `caffeine + O2 -> theobromine`). Paper
*prose* — "NdmB catalyzes the N3-demethylation of theobromine, producing
7-methylxanthine" — has no arrow, so it parsed to nothing. Cross-paper stitching
therefore materialized almost entirely from structured DB records, not from the
fetched papers, so multi-paper synthesis stayed sparse even after the four live-run
defects were fixed. This is the missing piece that makes "novel pathway from
multiple papers" actually work on real literature.

**Why it appeared:** WP5 shipped with only the deterministic parser (correct but
narrow — it never fabricates, but only catches equations). An LLM extraction step
was always the intended follow-on; it was simply not part of the WP0–WP7 scope.

**What this introduces:** a new `t2pw.rag.extract` module — `extract_reactions_from_text(text,
*, chat_fn=None)` sends one retrieved passage to an LLM and returns the reactions
it **explicitly states** as clean reaction dicts (`name` / `inputs` / `outputs` /
`enzymes` / `reversible`), plus `make_prose_extractor(chat_fn=None)` returning the
`text -> [reaction]` callable the orchestrator wires. `synthesize_with_report` /
`synthesize` gain an **opt-in** `prose_extractor` keyword; when supplied,
`_reactions_from_bundle` runs it per paper chunk alongside the arrow parser and
converts each result to a provenance-bound `_Reaction` (reusing
`_participants_from_field` for canonicalization and `_is_invalid_species_token`
for junk rejection — the same discipline the arrow path uses). Extraction is
memoized per chunk (`_make_memoized_extractor`) and capped at `_EXTRACT_MAX_PASSAGES`
= 24 passages/run, so the two passes over the bundles (synthesis + unfilled-gap
detection) never double-call the model and cost is bounded. The app builds the
extractor from the shared client and passes it, gated on the new
`RAG_EXTRACT_REACTIONS` config flag (default on).

**How it stays consistent with the design:** it obeys the separation invariant
exactly. **Evidence-bound / no invention:** the system prompt forbids inference
and background knowledge — the model transcribes only what the passage states, or
returns `{"reactions": []}` — and every extracted reaction inherits the source
chunk's provenance, so the WP6 "no element without evidence" guarantee still holds
(a reaction the passage does not state is never produced). **RAG → core only, no
stage edits:** all code lives in `t2pw.rag`; the shared `t2pw.llm.client.chat` is
imported **lazily** (importing `extract` needs no LLM client / key / network) and
no stage module is touched. **Offline / opt-in / fails closed:** the default
`prose_extractor=None` is byte-for-byte today's arrow-only synthesis (every
pre-existing synthesize test passes unchanged); extraction runs only when the app
wires it, and every call fails closed — an empty passage, a missing endpoint, a
malformed response, or a model that rejects JSON mode all degrade to `[]`, so prose
extraction can only *add* reactions, never break synthesis. With `RAG_ENABLED=false`
(or `RAG_EXTRACT_REACTIONS=false`) nothing here runs.

**Verified:** 10 new tests in `tests/test_rag_extract.py` — prose→structured parse,
markdown-fence tolerance, the no-reaction case, the empty-text short-circuit (no
model call), fail-closed on a raising `chat_fn`, garbage output → `[]`,
participant-less reaction dropped, and two end-to-end synthesis integration tests
proving a prose bundle adds an evidence-bound NdmB reaction *with* the extractor and
adds nothing *without* it, plus per-chunk memoization (exactly one call). A live
ad-hoc call against the configured model correctly extracted
`theobromine -> 7-methylxanthine (NdmB)` from a plain sentence. Full suite:
515 → **525 passed, 0 failures**.

---

### 2026-07-21 — RAG deep-dive: four live-run defects that emptied or degraded multi-paper output

**Files changed:** `src/t2pw/app/streamlit_app.py`, `src/t2pw/rag/retrieve.py`,
`src/t2pw/rag/store.py`, `src/t2pw/rag/select.py`, `.env`,
`tests/test_rag_retrieve.py`, `tests/test_rag_foundation.py`,
`tests/test_rag_select.py`, `docs/change_log.md`.

**What was the error:** with `RAG_ENABLED=true` and the "unknown / incomplete"
box checked, a real seed pathway (caffeine degradation, *Pseudomonas putida*)
came back with a **completely empty** `processes.reactions` — the pathway summary
and final merged JSON showed only a species and a biological state. The RAG unit
suite was fully green (offline fixtures), so none of these surfaced until the
subsystem actually ran end-to-end against LM Studio + live literature APIs. Four
independent defects, all inside `t2pw.rag` + the app orchestrator.

*Defect 1 (the empty payload — seed reactions dropped).* `maybe_run_rag` passed
`synthesize_with_report(seed_payload, bundles, seed_context_text)` a **string**
(`format_context_header(...)`) as the seed context. But `synthesize`'s
`_seed_source_descriptor` only accepts a **dict** carrying a `source_id`, so it
returned `None`; `_seed_reactions` then treated every seed reaction as having "no
supporting evidence" and **omitted all of them** (the no-invention guardrail,
misfiring on the seed paper). Synthesis emitted zero reactions.

*Defect 2 (the wipe).* Seam S3 replaced `final_payload = rag_result.payload`
wholesale whenever `rag_result.synthesized` was truthy — so the empty synthesized
payload from Defect 1 *overwrote* the real Stage 1/2 extraction. (The seed's
`species` / `biological_states` survived only because WP5's
`_carry_forward_scaffolding` re-copies them; the reactions did not.)

*Defect 3 (silent retrieval death — embedding dim mismatch).* `MemoryVectorStore.query`
scored every candidate with `_cosine`, which returns `0.0` on a length mismatch.
A lexical-fallback vector (256-dim) cached while the embeddings endpoint was down
(the user's exact earlier state) then sits in `embeddings_cache.json` beside real
API vectors (768-dim); once LM Studio is up, query 768 vs cached-chunk 256 → every
score `0.0` → semantic retrieval silently returns arbitrary chunks.

*Defect 4 (lost gap symbols).* `retrieve._reaction_symbols` read
`reaction["inputs"]/["outputs"]` only as `str`, but the real payload carries
participants as dicts (`{"name": "caffeine"}`). So a dangling-reaction gap query
lost its exact substrate/product symbols — the lexical half of the hybrid scorer
had only the reaction name and enzyme to match on.

Plus one robustness hole (Defect 5): `select._candidate_context` called the reused
`preprocess` per candidate **unguarded**, inside `maybe_run_rag`'s single
try/except — so one flaky LLM call (rate limit / timeout) among up to
`RAG_ACQUIRE_MAX_PAPERS` candidates aborted the *entire* RAG run.

**Why it appeared:** every WP was built and verified with offline, in-memory
fixtures where the seed context was already a dict, embeddings were a single
consistent width, and `preprocess` was mocked. The string-vs-dict seam (D1), the
wholesale replace (D2), the cache-poisoning dim mismatch (D3), the str-only
participant read (D4), and the unguarded fan-out (D5) are all boundary conditions
that only exist in a live run, so the green unit suite never exercised them.

**How the fix stays consistent with the design:** every change lives in
`t2pw.rag` + the app orchestrator (`src/t2pw/app`); no pipeline stage module was
edited and the separation invariant (docs/rag/03_separation_invariant.md) holds.
*D1:* `maybe_run_rag` now builds a seed **source descriptor** dict
(`{"text": seed_context_text, "source": {"source_id": "seed_paper",
"source_title": <pathway name>, "source_type": "paper"}}`) and passes it to
synthesis — the uploaded paper is legitimately evidence for its own reactions
(exactly what `_seed_source_descriptor`'s docstring intends), so seed reactions
carry `rag_provenance` and survive. *D2:* the S3 adoption is now guarded — the
synthesized payload replaces `final_payload` only when it preserves at least the
seed's reaction count (`_n_reactions(synth) >= max(1, _n_reactions(final))`),
otherwise the single-paper extraction is kept and a `st.warning` explains why; an
evidence-starved synthesis can never again blank the pathway. *D3:*
`MemoryVectorStore.query` falls back to lexical overlap whenever the query/chunk
vector widths differ (or either is missing) instead of scoring a silent `0.0`.
*D4:* `_reaction_symbols` now also reads dict-shaped participants
(`name`/`entity`/`compound`/…). *D5:* `_candidate_context` wraps `preprocess` in a
per-candidate try/except, degrading one failure to an organism-only context
rather than sinking the run. `.env` keeps `RAG_ENABLED=true` (LM Studio embeddings
verified reachable) now that the wipe is guarded. All fixes are additive/guarding;
with `RAG_ENABLED=false` the single-paper path is byte-for-byte unchanged.

**Verified:** offline repros confirmed each defect and its fix (string seed
context → 0 reactions vs dict → all preserved; 256-vs-768 cosine `0.0` → lexical
`0.4`; dict participants → symbols now include `caffeine`/`theobromine`). Three
regression tests added — `test_dangling_reaction_gap_captures_dict_participant_symbols`
(D4), `test_memory_query_dim_mismatch_falls_back_to_lexical` (D3),
`test_select_survives_a_failing_preprocess` (D5). RAG suite: 88 → **91 passed**;
full suite **515 passed, 0 failures**.

**Known limitation (not a regression — flagged for follow-up):** evidence-derived
reactions are still only transcribed from **arrow-style equations**
(`synthesize._parse_reaction_line`), which paper *prose* rarely contains, so
cross-paper stitching materializes almost entirely from structured DB records, not
free text. Until an LLM prose→reaction extraction step is added, multi-paper
synthesis will stay sparse even with all four defects fixed.

---

### 2026-07-21 — RAG synthesis: stop emitting `" ; "`-joined pathway-metadata blobs as entities (+ core defense-in-depth)

**Files changed:** `src/t2pw/rag/synthesize.py`,
`src/t2pw/pipeline/process_normalizer.py`, `tests/test_rag_synthesize.py`,
`tests/test_process_normalizer.py`, `docs/rag/03_separation_invariant.md`,
`docs/change_log.md`.

**Separation-invariant note (read this first):** this entry contains a **deliberate,
user-authorized exception** to the separation invariant
(docs/rag/03_separation_invariant.md): CHANGE 2 edits a **core stage module**
(`process_normalizer.py`). Every prior RAG entry kept all logic inside `t2pw.rag`;
this one does not, by explicit decision, to add defense-in-depth against a malformed
name class regardless of who produces it. The primary fix (CHANGE 1) is still fully
inside `t2pw.rag`; CHANGE 2 is an additive, narrowly-gated guard that changes no
existing behavior (zero test regressions). See the doc's new "Sanctioned exceptions"
section.

**What was the error:** with RAG enabled, the mapped pathway failed the pre-export
Stage 3 revalidation with "Protein '<blob>' is missing a UniProt or DrugBank
identifier.", where `<blob>` was an entire pathway serialized with `" ; "` separators
— e.g. `"Pathway12926 ; Arabidopsis thaliana, Cell, Plant-Type Vacuole ; Arabidopsis
thaliana, Cell, Cytosol ; ... ; Water ; Hydrogen Ion ; Triglyceride ; ... ; Glycerol
3-phosphate transporter ; Water"` (the Arabidopsis glycerolipid pathway), plus a
sibling `"Pathway4 ; Homo sapiens ... ; Adenosine triphosphate complex"` (glutathione)
routed into `protein_complexes` by the NAME-BASED COMPLEX RULE. Neither pathway was the
uploaded seed (a caffeine-degradation paper) — they were **retrieved corpus entries**.

**Root cause:** RAG synthesis re-parsed retrieved *evidence* chunk text as reaction
chemistry, but for corpus (`source_type="pwml_example"`) and DB (`pathbank`/`kegg`)
hits that text is not a clean equation — it is a `" ; "`-joined **bag** of
pathway-id + species + compartments + compounds + reaction-patterns that `ingest.py`
builds for *lexical scoring* (`_corpus_text_for_file` / `_extract_pwml_text` /
`_reaction_record_text`, all `" ; ".join(...)`). Every such bag contains a reaction
arrow somewhere (from the SBML `reaction_patterns`), so `synthesize._reactions_from_bundle`
→ `_parse_reaction_line` treated the whole ~800-token window as one equation, and
`_parse_side` — which split participants **only** on `" + "` (`_PLUS_SPLIT_RE`), never
on `" ; "` — collapsed the entire pre-arrow text into a **single giant participant
name**. That name became a compound entity (and, ending in "complex", a protein
complex), reaching the gate as an unresolvable protein/complex. Confirmed by live
repro: `_corpus_text_for_file('reference/PW012926 (1).sbml')` → first window →
`_parse_reaction_line(...)` returned a reaction whose `inputs[0].name` was the full
blob; the entity was present in `tmp/draft_graph.json` (built pre-mapping/pre-audit),
proving synthesis — not the audit or mapping — was the origin.

**Why the earlier robustness fixes did not catch it:** two structural reasons, i.e.
this is a genuine case of RAG **not** plugging into the prior stages despite the intent.
(1) Synthesis builds entities with its **own** ad-hoc parser (`_parse_reaction_line` /
`_parse_side`), far weaker than Stage 1 LLM extraction + `process_normalizer`; the prior
composite-splitting / name-sanitization fixes live in the extraction+normalizer path and
were never in synthesis's code path. (2) `" ; "` is a **RAG-invented** join delimiter —
the core normalizer's composite splitter keys on `" + "` / `_has_plus_token` and the core
pipeline never produced `" ; "`-joined names, so even downstream the splitter did not
recognize the blob as a composite and it survived to the gate untouched. Underneath both:
`build_motif_entry`'s corpus text is *lexical-retrieval scaffolding* (a token bag), never
meant to be round-tripped back into structured reactions.

**Fix — CHANGE 1 (primary, inside `t2pw.rag`, `synthesize.py`):** (a)
`_reactions_from_bundle` now reads `chunk.source_type` defensively and only transcribes
chunks whose type is in `{"paper", ""}` (+None) — `pwml_example` corpus scaffolding and
`pathbank`/`kegg` metadata bags are never parsed into reactions. (b) `_parse_side` /
`_parse_reaction_line` are hardened even for the chunks still parsed: participant sides
split on `";"` as well as `" + "` (`_SIDE_SPLIT_RE`), and any token that is clearly not a
single chemical species is rejected (`_is_invalid_species_token`: `^Pathway\d`, a
`", Cell,"`/biological-state descriptor, or > 12 words / > 120 chars); enzymes pass the
same filter; a reaction left with no valid participants is discarded. (c) Genuine
equations are unaffected (`"theobromine + O2 -> 7-methylxanthine + formaldehyde"` still
parses to 2+2; charge notation `NAD+`/`H+` survives — the split requires surrounding
spaces).

**Fix — CHANGE 2 (defense-in-depth, core exception, `process_normalizer.py`):** a
conservative, narrow guard `_quarantine_pathway_metadata_blobs`, called at the top of
`normalize_composites`, drops a `compounds`/`proteins`/`protein_complexes` row **only**
when its name contains `" ; "` **and** matches the garbage signature (`^Pathway\d` OR
`", Cell,"` OR > 12 words), recording a `pathway_metadata_blob_quarantined` action. The
**narrow guard was chosen over a broad `";"`-split** deliberately: the existing `" + "`
path *materializes a protein complex* (wrong semantics for a metadata bag), and the
narrow guard is the zero-regression path — the full suite confirmed no broadening was
needed. Real single-entity names essentially never contain `" ; "` (composites use
`" + "`), so the false-positive surface is negligible.

**Verified:** original bug reproduced then confirmed clean (the corpus blob no longer
yields a `" ; "` participant — the chunk is skipped, and the parser guards shatter the
bag even if it is mislabeled). New tests: `test_corpus_pwml_chunk_never_emits_pathway_blob_entities`
and `test_genuine_paper_equation_chunk_still_parses_cleanly`
(`tests/test_rag_synthesize.py`), `test_pathway_metadata_blob_is_quarantined_by_normalizer`
(`tests/test_process_normalizer.py`). Full suite: **512 passed, 0 failures** (509
baseline + 3 new); **no existing test regressed**. `ruff check src/t2pw/rag
src/t2pw/pipeline/process_normalizer.py` clean, no new violations.

---

### 2026-07-21 — RAG synthesis: carry the seed's species scaffolding + stop the `provenance` key collision

**Files changed:** `src/t2pw/rag/synthesize.py`, `src/t2pw/rag/provenance.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_rag_synthesize.py`,
`tests/test_rag_provenance_gates.py`, `tests/test_rag_foundation.py`,
`docs/rag/03_separation_invariant.md`, `docs/rag/00_overview.md`,
`docs/rag/agents/wp0_foundation.md`, `docs/rag/agents/wp5_synthesis.md`,
`docs/change_log.md`.

**What was the error:** with RAG enabled, a caffeine-degradation seed paper aborted
at the Stage 2B mapping output → Stage 3 normalization input boundary
(`validate_post_mapping`) with one error — `species_required`, "Mapped payload must
include at least one species row." — accompanied by 63 `runtime_schema_type_error`
warnings, one per compound, all "Expected a string." at
`/entities/compounds/N/provenance`. Two independent defects, both introduced by the
RAG work packages, both fixable entirely inside `t2pw.rag` + the app orchestrator.

*Defect 1 (the abort).* WP7 wires the synthesized payload to **replace**
`final_payload` wholesale at seam S3 (`streamlit_app.py`: `final_payload =
rag_result.payload`). But WP5 synthesis rebuilds the payload **from reactions only**:
`_build_entities`/`to_payload` emit just `entities.compounds` and `entities.proteins`
and never read `seed_payload["entities"]`. So the seed's contextual scaffolding —
`species`, `subcellular_locations`, `cell_types`, `tissues`, and top-level
`biological_states` — was silently dropped. Stage 2B mapping then produced a payload
with zero species rows, and the post-mapping gate (which legitimately requires
`entities.species` to be a non-empty list) aborted.

*Defect 2 (the 63 warnings).* WP0 chose `provenance` as one of the four additive RAG
keys, and WP5's `_attach_provenance` wrote a **dict** there
(`row["provenance"] = dict(primary)`). But `provenance` is **not** a free additive
name: the core schema already owns it as a *string*
(`PayloadProvenance = Literal["extracted","inferred","curated","enriched"]`, present
on every entity/process via `PayloadCommonRecord`). RAG was therefore repurposing and
retyping a core-owned field — exactly what the separation invariant's
additive-metadata rule forbids — which the runtime shape validator flagged (in report
mode, so non-fatal) on every compound.

**Why it appeared:** WP5 was written and reviewed as an *additive-evidence* layer, and
its own brief (docs/rag/agents/wp5_synthesis.md) called `species`/`subcellular_locations`
"contextual scaffolding … never emitted by RAG synthesis." That was harmless while
synthesis output only *augmented* the seed, but WP7 later made it *replace* the seed
payload, at which point "never emitted" became "silently deleted." The `provenance`
collision was latent from WP0: the additive keys were never checked against core-owned
field names, and because the shape validator runs in report mode the collision only
ever surfaced as a warning, so it was never treated as a defect.

**How the fix stays consistent with the design:** both fixes stay entirely inside
`t2pw.rag` + the app orchestrator; no stage module was edited and the separation
invariant (docs/rag/03_separation_invariant.md) holds. *Defect 1:* a new
`_carry_forward_scaffolding(payload, seed_payload)` in `synthesize.py` deep-copies the
seed's non-reaction scaffolding buckets (`species`, `subcellular_locations`,
`cell_types`, `tissues` into `entities`; `biological_states` at payload top level) into
the synthesized payload — only when present and not already rebuilt, so the
evidence-built `compounds`/`proteins` are never clobbered — guarded against a non-dict
seed / missing entities, and run before the existing `validate_post_extraction`
self-check. These buckets are evidence-*exempt* (they are not reaction chemistry; see
`_EVIDENCE_ENTITY_BUCKETS` in `provenance.py`), so carrying them verbatim does not
violate the "no element without evidence" guarantee. *Defect 2:* the additive source
pointer is renamed `provenance` → `rag_provenance` everywhere it is emitted or read as
the RAG key (`RAG_ADDITIVE_KEYS`, `RagAdditiveMetadata`, `_has_resolvable_source`,
`_attach_provenance`, `strip_provenance`/`validate_provenance` via the tuple, and the
app's provenance viewer). The core `provenance` string field is left untouched — in
particular `_seed_row_provenance` still reads the seed's core `provenance` string — so
a RAG-off or RAG-unaware stage sees an unchanged core row, and the namespaced `rag_*`
key can never again shadow a core one. The additive keys stay optional/additive and
`strip_provenance` still removes exactly `RAG_ADDITIVE_KEYS`.

**Verified:** new/extended tests in `tests/test_rag_synthesize.py` prove the seed's
`species` (and the other scaffolding) is carried forward and satisfies the
`validate_post_mapping` species predicate, that synthesized rows carry `rag_provenance`
(a dict) and never a dict under the core `provenance` key, and that `strip_provenance`
removes it; a new end-to-end test
(`test_synthesized_payload_survives_real_stage2b_mapping`) drives a synthesized payload
through the **real** `map_ids.map_payload` and asserts the **real**
`validate_post_mapping` passes with no `species_required` error and the seed species row
survives mapping with its `mapping_meta` (external resolver calls mocked offline,
mirroring `tests/test_stage2_mapping_boundary.py`; the gate is never stubbed).
`tests/test_rag_provenance_gates.py` and `tests/test_rag_foundation.py` updated for the
renamed key. Full suite: **509 passed, 0 failures** (508 baseline + 1 new integration
test); ruff clean on `src/t2pw/rag` with zero new violations in `streamlit_app.py`
(its pre-existing 34 E402 sys.path-shim baseline is unchanged). With `RAG_ENABLED=false`
the payload carries neither `rag_provenance` nor the carried-forward path, so today's
single-paper pipeline is byte-for-byte unchanged.

---

### 2026-07-20 — RAG defaults: dedicated embeddings endpoint + toggle-on default

**Files changed:** `src/t2pw/config.py`, `src/t2pw/rag/embed.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_rag_foundation.py`, `.env`,
`docs/change_log.md`.

**What was the error:** the embedder (`embed.py`) reused the shared chat client
(`t2pw.llm.client._client`) for embeddings and ignored `RAG_EMBEDDING_PROVIDER`.
With `LLM_PROVIDER=openrouter` that meant embedding calls went to OpenRouter,
which has **no embeddings endpoint**, so every call failed and silently dropped
to the lexical fallback — "full embeddings" could never actually run. The RAG
toggle also defaulted off and the master flag defaulted off, so RAG was never the
active default.

**Why it appeared:** WP0 wired embeddings against the single shared client on the
assumption chat and embeddings share a host. They don't when chat is OpenRouter.

**How the fix stays consistent with the design:** it stays entirely inside
`t2pw.rag` + config + the S5 orchestrator (no stage-module edit; separation
invariant intact). `config.py` gains two optional, default-safe keys
(`RAG_EMBEDDING_BASE_URL`, `RAG_EMBEDDING_API_KEY`); `embed.py` builds a
dedicated OpenAI-compatible client pointed at that base_url when set (e.g. LM
Studio at `http://127.0.0.1:1234/v1`) and otherwise reuses the shared client
exactly as before — the lexical offline fallback is unchanged, so a missing/
unreachable endpoint still degrades gracefully. The app toggle now defaults ON
(RAG is the default) and, when turned OFF, still takes the byte-for-byte pre-RAG
single-paper path (the orchestration call is guarded on both `RAG_ENABLED` and
the toggle). `.env` enables RAG with the `memory` vector backend (full semantic
search over real embeddings, no chromadb dependency) and LM Studio embeddings.
Blank config reproduces WP0 behavior, so `RAG_ENABLED=false` remains today's
pipeline.

---

### 2026-07-20 — RAG orchestration, UI & triage (WP7): wire R0–R5 behind the flag, no logic in the app

**Files changed:** `src/t2pw/rag/triage.py`, `src/t2pw/app/streamlit_app.py`,
`tests/test_rag_triage_orchestration.py`, `docs/change_log.md`.

**What this introduces:** the final RAG work package — stage **R0** (triage) plus the
orchestrator wiring (seam **S5**) that ties R0–R5 together and exposes them, without
moving any logic into the app. `triage.py` gains `should_run_rag(context, user_flag,
reports=None) -> TriageDecision` (the *one* piece of RAG decision logic the invariant
permits outside the app): an explicit user flag always runs RAG; otherwise it
auto-triggers on a low Stage-0 `scope_clarity_score` (< 0.5) or, when the core's
read-only reports are supplied, on the WP4 gap signals (dangling reactions, orphan
metabolites, unmapped enzymes) — delegating gap classification to `retrieve.detect_gaps`
(lazy import, so `import t2pw.rag.triage` needs no chromadb); a clean, in-scope pathway
returns `run=False`. `streamlit_app.py` gains **wiring only**: a thin, importable
`maybe_run_rag(...)` helper that CALLS `acquire.search_candidates` / `fetch_full_text`
-> `select.select` -> `ingest.ingest` -> `retrieve.detect_gaps` / `retrieve_evidence` /
`format_retrieval_context` -> `synthesize.synthesize_with_report` -> `validate_provenance`
and passes their results between the seams; the UI (a "This pathway is unknown /
incomplete (enable multi-paper RAG)" checkbox, a fetched+selected papers panel from the
WP1/WP2 reports, and a provenance viewer showing source papers per reaction/entity from
the WP5/WP6 provenance). Evidence rides the **existing** seams: S1 (folded into
`user_task_context`), S2 (appended to the audit's `retrieval_context` via a new
defaulted `rag_evidence_context=""` param), and S3 (the synthesized standard `Payload`
handed to the post-pipeline path).

**Why it appeared:** WP0–WP6 built the RAG subsystem but nothing exposed it or decided
*when* it should run. WP7 is that trigger + orchestration + UI layer: the last step that
makes multi-paper RAG reachable from the app while keeping it invisible when off.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. **S5 = wiring only, no logic:**
the app contains no normalization/mapping/retrieval/synthesis logic — `maybe_run_rag`
and `render_rag_panels` only call `t2pw.rag` (and existing stage) functions and read
their returned values; the sole RAG *decision* logic lives in `t2pw.rag.triage`, not the
app. **RAG-off byte-identity (definition of done):** every RAG addition is guarded by
`if rag_config()["enabled"]` (and `maybe_run_rag` returns `None` before importing or
calling any chain function when disabled or when triage declines), so with the default
`RAG_ENABLED=false` the checkbox is not rendered, the orchestration block and UI panels
do not run, `user_task_context`/`retrieval_context`/`final_payload` are untouched, and
the app path is identical to pre-initiative `main` — the new `rag_evidence_context`
param defaults to `""` (no-op), proven by the extracted-function orchestration test
still passing unchanged. **No core stage edited:** the only non-rag file changed is
`src/t2pw/app/streamlit_app.py` (the orchestrator, in `src/t2pw/app`, not a stage dir);
`git status --porcelain` on `pipeline`/`pwml`/`curation`/`mapping`/`schema.py`/`sbml`
shows no modified files, and no stage module imports `t2pw.rag` (`grep -rn "t2pw.rag"`
over the stage dirs is empty). Tests are offline, deterministic, self-contained (the
guarded `openai` + a MagicMock `streamlit` stub let the app-helper import run alone):
they cover the triage cases and a guard/wiring test proving the RAG path is not entered
with `RAG_ENABLED` off (`acquire` is never called) yet *is* entered when enabled+flagged.
Baseline 494 + 11 new = 505 passed, 0 failures; zero new ruff (streamlit_app.py stays at
34, the added import carries a scoped `# noqa: E402` matching the file's pre-existing
sys.path-shim pattern).

---

### 2026-07-20 — RAG provenance & gates (WP6): evidence-bound validation + the gate tripwire

**Files changed:** `src/t2pw/rag/provenance.py`, `tests/test_rag_provenance_gates.py`,
`docs/change_log.md`.

**What this introduces:** stage R6 of the RAG subsystem — the layer that *proves*
the initiative's core promise, **no element without evidence**, and that a
synthesized payload survives the existing Stage 3/8 gates unmodified. `provenance.py`
(the WP0 additive-key stub) is extended with: `validate_provenance(payload) ->
ProvenanceReport` — a read-only check that every reaction (`processes.reactions`) and
every non-cofactor entity (`entities.{compounds,proteins,protein_complexes}`) carries
at least one resolvable `source_id`/`source_uri` (via any of the four additive
carriers or the core-typed `source_refs`), flagging any that do not; the
`ProvenanceReport` / `ProvenanceIssue` dataclasses that carry the result; and
`strip_provenance(payload) -> Payload` — a deep copy with every `RAG_ADDITIVE_KEYS`
key removed at any depth (input never mutated), i.e. the plain payload a RAG-unaware
or RAG-off stage sees. The cofactor exemption reuses WP5's `COFACTOR_NAMES` (lazy
import to avoid a circular dependency and to keep the module import-cheap — verified
`import t2pw.rag.provenance` still needs no chromadb).

**Why it appeared:** WP5 emits a synthesized payload with additive provenance, but
nothing yet *enforced* that every element is evidence-bound, nor *demonstrated* that
the additive keys pass the core gates untouched. WP6 is that enforcement-and-proof
step: `validate_provenance` is the guardrail that catches an unsourced (invented)
element, and the tests are the tripwire that catches anyone loosening a gate to push
RAG output through.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. **Gates called unmodified,
directly:** the tests import and call the **real** `run_strict_post_normalization_gates`
(Stage 3) and `validate_required_pwml_contract` (Stage 8) — no RAG-specific variant,
no fork, no special-case — and assert a good synthesized payload passes both, and that
the *same* gates pass on `strip_provenance(payload)` (provenance is purely additive and
ignored). **Never weaken a gate:** no stage-module file was edited (verified: `git
status --porcelain` on the stage dirs / `schema.py` / `sbml` shows no modified files;
`process_normalizer.py` and `pwml/ir.py` are byte-unchanged); the gate-ready fixture is
built *honestly* — where Stage 3/8 legitimately require a mapped payload (external DB
identities, `biological_states`, generated-complex wrappers that the core mapping stage
adds *after* seam S3, needing the offline-unavailable reference DB), the fixture supplies
that mapping result rather than stubbing the gate, and a separate test ties it back to
reality by asserting genuine (un-mapped) `synthesize` output already satisfies
`validate_provenance`. **Additive-only provenance:** `strip_provenance` removes exactly
the four `RAG_ADDITIVE_KEYS`; a test proves none survive and that both gates still pass;
the `RAG_ENABLED=false` path asserts the plain payload carries no provenance keys and the
gates behave identically. **RAG → core only:** all code lives in `t2pw.rag`; nothing in
any stage module imports `t2pw.rag` (verified: `grep -rn "t2pw.rag"
src/t2pw/{pipeline,pwml,mapping,curation,sbml}` is empty). Tests are offline,
deterministic, and self-contained (the guarded `openai` stub keeps them passing run
alone) — baseline 486 + 8 new = 494 passed, 0 failures; zero new ruff.

---

### 2026-07-20 — RAG synthesis (WP5): stitch + reconcile + resolve + provenance → one standard Payload

**Files changed:** `src/t2pw/rag/synthesize.py`, `tests/test_rag_synthesize.py`,
`docs/change_log.md`.

**What this introduces:** stage R5 of the RAG subsystem — the layer that merges the
seed extraction plus WP4's per-gap `EvidenceBundle`s into **one connected pathway**
and emits it as a **standard** `Payload` (the `TypedDict` shapes in `t2pw.schema`) at
seam **S3**. `synthesize.py` adds: `synthesize(seed_payload, evidence_bundles,
seed_context) -> Payload` (the seam-S3 entry point named in wp5_synthesis.md) and its
sibling `synthesize_with_report(...) -> SynthesisResult`, which returns the same
payload **plus** the reports that ride alongside it (`unresolved_gaps`, `conflicts`,
`stitched`, `contract_report`); and `to_payload(entities, reactions) -> Payload`,
which assembles only the core `entities`/`processes` buckets. The four synthesis
steps: (1) **stitch** — reactions stated in evidence chunks are transcribed and
connected so a product feeds the next reaction's input across papers; a dangling end
is closed *only* where a retrieved reaction supplies the missing metabolite
(cross-paper links are detected and recorded in `stitched`); (2) **reconcile
synonyms** — every name is canonicalized through the core `BIOCHEMICAL_ALIAS_MAP`
(imported **read-only** from `process_normalizer`, the same casefold-keyed lookup it
performs, reproduced without importing RAG into it); (3) **resolve conflicts** —
reactions grouped by their unordered participant set; when variants disagree on
direction / stoichiometry / compartment the highest evidence-weight variant wins and
the losers are recorded in `conflicts` (nothing dropped silently); (4) **attach
provenance** — every reaction and every non-cofactor entity carries the additive
provenance keys WP0 defined (`provenance` / `evidence` / `source_papers` /
`rag_confidence`, `RAG_ADDITIVE_KEYS`) plus a core-typed `source_refs: List[str]`
pointer. Enzymes are emitted as canonical Actor rows (`entity` / `entity_type` /
`role`). Merging is out of scope for gate *enforcement* (WP6) and orchestration/UI
(WP7).

**Why it appeared:** WP0–WP4 built the store, embedder, hybrid scorer, and gap
retrieval, but nothing yet *assembled* the retrieved evidence into a single
exportable pathway. WP5 is that assembly step — the place the "novel pathway" (a
novel *connection* of individually evidence-backed steps; docs/rag/00_overview.md) is
actually built. It reuses landed pieces rather than duplicating them: WP4's
`EvidenceBundle` / `Gap` and the store `Chunk` / `Retrieved` are consumed by **duck
typing** (only their attributes are read) so importing `synthesize` pulls neither the
retrieval/ingest stack nor chromadb; the core alias map is imported read-only; and
the output is checked with the core `validate_post_extraction` before return.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. **Standard Payload at S3:**
`to_payload` emits only the core `entities`/`processes` buckets — the shape Stage 2B
already consumes — and the output **passes `validate_post_extraction`** (the module
imports and calls it as a self-check, raising `StageContractError` on any structural
failure; it never edits or weakens that contract). **Additive-only provenance:** the
only extra keys are the four `RAG_ADDITIVE_KEYS` plus the core-owned `source_refs`;
all are optional and ignored by any stage that does not know them (a test strips
every one and the payload still passes `validate_post_extraction`) — no RAG-only
*required* key is added. Runtime shape validation runs in report mode
(`RUNTIME_SCHEMA_MODE="report"`), so the additive keys surface as non-fatal warnings,
never errors. **No invented chemistry:** every reaction and every non-cofactor entity
must carry ≥1 provenance pointer; an element with none is **omitted** and reported in
`unresolved_gaps` (a gap whose evidence bundle has no hits stays unfilled and is
surfaced — never fabricated). **No pre-running the core:** synthesis does not
normalize, map, or audit — it emits the payload and lets Stage 2B→8 run. **RAG →
core only:** all code lives in `t2pw.rag`; it imports `BIOCHEMICAL_ALIAS_MAP`,
`validate_post_extraction`, and `RAG_ADDITIVE_KEYS` from core, and nothing in any
stage module imports `t2pw.rag` (verified: `grep -rn "t2pw.rag"
src/t2pw/{pipeline,mapping,curation,pwml,sbml}` is empty). No stage-module file was
edited (verified: `git status --porcelain` on the stage dirs / `schema.py` / `sbml`
shows no modified stage files). Tests are offline, deterministic, and self-contained
(the guarded `openai` stub is only needed to build the WP4 fixtures) and pass run
alone.

---

### 2026-07-20 — RAG gap retrieval (WP4): detect gaps → query → retrieve evidence → format context

**Files changed:** `src/t2pw/rag/retrieve.py`, `tests/test_rag_retrieve.py`,
`docs/change_log.md`.

**What this introduces:** stage R4 of the RAG subsystem — the layer that turns the
core's read-only gap signals into gap-targeted evidence and renders it to a string
the existing prompts already accept. `retrieve.py` adds: `detect_gaps(payload,
reports) -> list[Gap]`, which reads `qa_graph` connectivity/degree output (both the
`generate_qa_report` `flags` shape and the CLI `dangling_nodes` /
`missing_links_suspected` / `orphan_components` shape), the Stage-3 strict gate
report's `errors` list (from `run_strict_post_normalization_gates`), and mapping
reports (entities with `status="unmapped"`), classifying each into one of
`dangling_reaction` / `orphan_metabolite` / `unmapped_enzyme` / `missing_precursor`
/ `missing_compartment` (reaction gaps enriched, read-only, with participant/enzyme
symbols from the payload); `query_for_gap(gap, seed_context) -> str`, a
natural-language ask plus the exact gene/compound symbols (so the hybrid scorer's
lexical half never loses an exact symbol); `retrieve_evidence(gap, store, *,
top_k=rag_config()["retrieve_top_k"]) -> EvidenceBundle`, which retrieves via the
WP3 `build_hybrid_scorer(store)` and keeps each hit's `source_id` / `source_uri`
provenance; and `format_retrieval_context(bundles) -> str`, which mirrors/wraps the
existing `t2pw.sbml.examples.build_retrieval_context` renderer and appends the
mandatory additive provenance line per hit. `Gap` and `EvidenceBundle` are defined
within `t2pw.rag`. Merging evidence into a final payload (WP5) and gate enforcement
(WP6) are out of scope.

**Why it appeared:** WP0–WP3 built the store, embedder, and hybrid scorer, but
nothing yet detected *which* pieces of a pathway are missing, formed queries for
them, or rendered the retrieved evidence for injection. WP4 is that missing step. It
reuses the landed pieces rather than duplicating them: the WP3
`build_hybrid_scorer` (never a second scorer), the WP0 `VectorStore` / `Retrieved` /
`Chunk`, `rag_config()` for `retrieve_top_k`, and — critically — it **wraps** the
existing renderer `t2pw.sbml.examples.build_retrieval_context` (feeding it a
synthetic single-entry index with a self-matching query so it always renders, then
swapping its `[Example i]` header for a gap-tagged `[Evidence i]` header) instead of
writing a second formatter. Offline-first holds end to end: import requires no
chromadb / network / LLM, and with the `memory` backend + a stubbed embedder the
lexical half still retrieves an exact symbol (e.g. `NdmA`).

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly, and this is the first WP to
touch the core seams. **Evidence rides only the EXISTING seam params:**
`format_retrieval_context` returns a plain **string** meant to be folded into the
already-present `pathway_context` / `user_task_context` params of
`run_extraction_pipeline` (S1) and passed to the already-present
`run_audit(..., retrieval_context=...)` param (S2) — no new parameter is added to,
and no body is edited in, `pipeline.py` or `run_audit`; the actual wiring is left to
WP7 (S5). **Reports are read-only (S4):** `detect_gaps` inspects the `qa_graph` /
gate / mapping artifacts and never writes back (a test deep-compares `payload` and
`reports` before/after and asserts they are unchanged). All new code lives in
`t2pw.rag`; the dependency arrow points RAG → core only — `retrieve.py` imports the
WP3 scorer, the WP0 store, `rag_config`, and `t2pw.sbml.examples`, and nothing in any
stage module imports `t2pw.rag` (verified: `grep -rn "t2pw.rag"
src/t2pw/{pipeline,mapping,curation,pwml,sbml}` is empty). No stage-module file was
edited (verified: `git status --porcelain` on the stage dirs / `schema.py` /
`sbml` shows no modified stage files). Tests use the `memory` backend with a stubbed
offline embedder — no chromadb, no network, no live LLM — and pass run alone.

---

### 2026-07-20 — RAG ingest & index (WP3): chunk → embed → vector store + hybrid scorer

**Files changed:** `src/t2pw/rag/ingest.py`, `tests/test_rag_ingest.py`,
`docs/change_log.md`.

**What this introduces:** stage R3 of the RAG subsystem — the layer that turns the
WP2-selected papers (plus structured DB reaction records and the existing on-disk
example corpus) into a populated, persisted `VectorStore`, and exposes the hybrid
retriever WP4 will call. `ingest.py` adds: `chunk_paper(candidate) -> list[Chunk]`
(section-aware splitting into abstract / introduction / methods / results /
discussion / figure-caption chunks of ~500–1000 tokens with overlap, each carrying
`source_id` / `source_uri` / `organism` provenance and a chunk `id` that is a
stable hash of `(source_id, section, offset)`); `chunk_db_reactions(records) ->
list[Chunk]` (one chunk per reaction, `source_type` `"pathbank"` / `"kegg"`);
`chunk_corpus(dir)` (one-or-more chunks per `reference/*.pwml` / `*.sbml` file,
tagged `source_type="pwml_example"`); `ingest(selection) -> IngestReport` (chunk →
embed via the WP0 `Embedder` → `upsert` → `persist`); and
`build_hybrid_scorer(store)`, the WP4-facing callable that blends the store's
semantic score with the lexical motif score at `0.7*semantic + 0.3*lexical`
(weights tunable). No gap detection, query formulation, or synthesis happens here —
that is WP4/WP5.

**Why it appeared:** WP0–WP2 built the store/embedder and produced a small, on-topic
set of papers, but nothing yet chunked, embedded, indexed, or retrieved them. WP3
is that missing middle. It deliberately **reuses** the landed pieces rather than
duplicating them: the WP0 `Chunk` / `VectorStore` / `get_vector_store` / `Embedder`
(the embedder's cache means an unchanged chunk is never re-embedded), the WP1
`CandidatePaper`, and — critically — it **wraps** the existing lexical layer
`t2pw.sbml.examples` (`parse_sbml` + `build_motif_entry` for corpus text extraction,
and `_score_entry` for the lexical half of the hybrid scorer) instead of writing a
second token-overlap scorer. Offline-first is preserved end to end: with no
embedding endpoint the embedder falls back to its deterministic lexical vectors, and
the hybrid scorer's lexical half guarantees an exact gene/compound symbol (e.g.
`NdmA`) is still retrieved when embeddings are unavailable. Re-ingesting an unchanged
paper is a no-op — stable chunk ids overwrite the same records and the embedding
cache reports zero new embeddings.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. All new code lives in
`t2pw.rag`; the dependency arrow points RAG → core only — `ingest.py` imports the
WP0 store/embedder, the WP1 `CandidatePaper`, and `t2pw.sbml.examples`, and nothing
in `t2pw.sbml` (or any stage module) imports `t2pw.rag` (verified: `grep -rn
"t2pw.rag" src/t2pw/{pipeline,mapping,curation,pwml,sbml}` is empty). The lexical
layer is **wrapped, not edited** — `src/t2pw/sbml/examples.py` is untouched (verified:
`git status --porcelain` on it is empty); no RAG logic was added to any stage module.
WP3 uses **no** core seam: it changes no pipeline behavior and only builds the store
and scorer that WP4 will consume. All configuration is read through `rag_config()`;
tests use the `memory` backend with a stubbed offline embedder — no chromadb, no
network, no live LLM — and pass run alone.

---

### 2026-07-20 — RAG selection (WP2): rank / dedupe / cap candidates for embedding

**Files changed:** `src/t2pw/rag/select.py`, `tests/test_rag_select.py`,
`docs/change_log.md`.

**What this introduces:** stage R2 of the RAG subsystem — the selection layer
that turns the WP1 candidate papers into the small, on-topic subset worth
embedding (WP3). `select.py` adds `score_candidate(candidate, seed_context) ->
SelectionScore`, which combines organism match, overlap of the candidate's
entities with the seed's `key_compounds` / `key_proteins` / `gap_terms` /
`pathway_name`, the preprocessor's `pathway_relevance_score`, and a penalty for a
`multi_example_review` whose examples do not match the seed. `select(candidates,
seed_context, *, max_papers=RAG_SELECT_MAX_PAPERS) -> Selection` scores every
candidate, ranks them deterministically (score desc, then paper id), dedupes by
PMCID/PMID/DOI/normalized-title (reusing `CandidatePaper.identity_keys`), caps at
`RAG_SELECT_MAX_PAPERS`, and returns the kept subset plus a `selection_report`
that gives one entry per candidate and an explicit reason for **every** drop
(duplicate, non-matching `multi_example_review` below the score floor, or below
the cap). No chunking, embedding, or retrieval happens here — that is WP3/WP4.

**Why it appeared:** WP1 over-fetches candidates from several literature APIs; if
all of them reached the (expensive) embedding step, unrelated review examples
would bleed into the corpus and the pathway synthesis would be polluted. WP2 is
the gate that stops that. It deliberately **reuses the existing preprocessor**
(`t2pw.pipeline.preprocessor.preprocess`, run per candidate on its abstract or a
truncated full text, plus `is_ambiguous_multi_example_review_context`) rather
than building a second classifier, and reuses the name-normalization / safe-access
helpers from `t2pw.mapping.map_ids`. The `multi_example_review` handling follows
`preprocess_system.txt` STEP 3 locality discipline: a review whose examples do
not match the seed is penalized so it is dropped, and one whose example *does*
match is example-scoped and ranked below on-topic primary research — never
ingested wholesale.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. All new code lives in
`t2pw.rag`; the dependency arrow points RAG → core only — `select.py` imports
from `t2pw.pipeline.preprocessor` and `t2pw.mapping.map_ids`, and nothing in
`t2pw.pipeline` (or any stage module) imports `t2pw.rag` (verified: `grep -rn
"t2pw.rag" src/t2pw/{pipeline,mapping,curation,pwml}` is empty). WP2 uses **no**
core seam: it changes no pipeline behavior and edits no stage module — it only
filters the WP1 `list[CandidatePaper]` down for WP3 to consume. Determinism is
structural: given a fixed `preprocess` output the module is pure arithmetic and
stable sorting, so a re-run yields the same ranking and report; tests mock
`preprocess` and never touch the network / LLM. All configuration is read through
`rag_config()` (`RAG_SELECT_MAX_PAPERS`); nothing is hardcoded.

---

### 2026-07-20 — RAG acquisition (WP1): candidate paper fetch + offline cache

**Files changed:** `src/t2pw/rag/acquire.py`, `tests/test_rag_acquire.py`,
`docs/change_log.md`.

**What this introduces:** stage R1 of the RAG subsystem — the acquisition layer
that turns a seed pathway context into candidate papers. `acquire.py` adds a
`CandidatePaper` dataclass (`id, source, title, abstract, organism, full_text,
source_uri, year`), `search_candidates(context, *, sources, max_papers)` which
builds organism-scoped queries from the seed context (`pathway_name`,
`likely_organism`, `key_compounds`, `key_proteins`, `gap_terms`) and fetches from
EuropePMC and NCBI eutils (with optional Crossref / Semantic Scholar / bioRxiv
sources behind the `sources` flag), and `fetch_full_text(candidate)` which
downloads `fullTextXML` and converts it to plain text. Candidates are deduped
against the seed and each other by PMCID/PMID/DOI/normalized-title, capped at
`RAG_ACQUIRE_MAX_PAPERS`, and cached on disk under
`data/rag_index/acquire_cache/` keyed by a query hash. No ranking, chunking, or
embedding happens here — that is WP2/WP3.

**Why it appeared:** the RAG initiative needs an evidence source before it can
select (WP2) or embed (WP3). WP1 is that source. It deliberately reuses the
existing HTTP plumbing in `t2pw.mapping.map_ids` (`HttpClient`,
`_europepmc_full_text`, the `_NCBI_EUTILS_BASE` / `_ncbi_eutils_params` /
`_ncbi_throttle` eutils helpers) rather than re-deriving URL, retry, or
rate-limit logic, so acquisition inherits the same session, backoff, and NCBI
throttle the core already tuned.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. All new code lives in
`t2pw.rag`; the dependency arrow points RAG → core only — `acquire.py` imports
from `t2pw.mapping.map_ids`, and nothing in `t2pw.mapping` (or any stage module)
imports `t2pw.rag` (verified: `grep -rn "t2pw.rag"
src/t2pw/{pipeline,mapping,curation,pwml}` is empty). WP1 uses **no** core seam:
it changes no pipeline behavior and edits no stage module — it only produces a
`list[CandidatePaper]` for WP2 to consume. Offline-first is honored structurally,
matching the `id_mapping_cache.json` precedent: every network fetch is fail-safe
(a missing network or API error contributes an empty list, never a raised
exception), and the per-query-hash disk cache means a re-run is served from cache
without re-hitting the network. All configuration is read through `rag_config()`;
nothing is hardcoded.

---

### 2026-07-20 — RAG subsystem foundation (WP0): package, vector store, config, provenance

**Files changed:** `src/t2pw/rag/__init__.py`, `src/t2pw/rag/store.py`,
`src/t2pw/rag/embed.py`, `src/t2pw/rag/provenance.py`,
`src/t2pw/rag/{acquire,select,ingest,retrieve,synthesize,triage}.py` (stubs),
`src/t2pw/config.py`, `requirements.txt`, `.gitignore`,
`tests/test_rag_foundation.py`, `docs/change_log.md`.

**What this introduces:** the shared scaffolding for the RAG initiative — a new,
optional `t2pw.rag` package with a `VectorStore` `Protocol` plus a `memory` and a
default `chroma` backend, an offline-capable embedding client with a JSON cache,
a `rag_config()` reader for every `RAG_*` variable, and `TypedDict` definitions
for the additive provenance keys (`provenance`, `evidence`, `source_papers`,
`rag_confidence`). No pipeline behavior changes: with `RAG_ENABLED` unset,
nothing here runs and no core module imports it.

**Why it appeared:** the RAG initiative (docs/rag/) needs a single foundation
every later work package (WP1–WP7) builds on. Landing it first, fully green and
inert, lets those packages depend on stable interfaces instead of re-deriving
them, and keeps the risky pieces (an optional heavy dependency, a network
embedder) isolated behind guards from day one.

**How it stays consistent with the separation design:** it obeys the separation
invariant (docs/rag/03_separation_invariant.md) exactly. All RAG code lives in
`t2pw.rag`; the dependency arrow points RAG → core only (verified: `grep -rn
"t2pw.rag" src/t2pw/{pipeline,mapping,curation,pwml}` is empty). WP0 touches no
seam except adding config: no stage module was edited, and the additive
provenance keys are optional `TypedDict`s that existing stages ignore —
`t2pw.schema` is referenced, never modified. The optional-dependency and
offline-first rules are honored structurally: `chromadb` is imported lazily and
guarded (importing the package never requires it), the embedder imports the LLM
client lazily and degrades to a deterministic lexical vector when no endpoint is
reachable, and the index lives in git-ignored `data/rag_index/` like the other
rebuildable caches. This mirrors the pipeline's own rule that logic spanning
stages belongs in the orchestrator, not inside a stage — here, RAG is that
independent subsystem.

---

### 2026-07-15 — An unstated complex-component stoichiometry is blank, not an error

**Files changed:** `src/t2pw/pipeline/entity_identity.py`,
`src/t2pw/pwml/ir.py`, `src/t2pw/pwml/writer.py`, `src/t2pw/pwml/qa.py`,
`src/t2pw/mapping/map_ids.py`, `src/t2pw/curation/audit_json_llm.py`,
`src/t2pw/curation/gap_resolver.py`, `tests/test_pwml_ir.py`,
`tests/test_audit_json_llm_payload.py`, `docs/change_log.md`.

**Error / symptom:** Export blocked at the Stage 8 strict gate with four
pointer-level errors on a caffeine-degradation run:
`/entities/protein_complexes/0/components/0 - Component[0] in complex 'NdmCDE
protein complex' is missing stoichiometry.` (likewise components 1–2 and
`Cdh`). This symptom class had already been "fixed" repeatedly —
`docs/pathwhiz_requirements.md` §4.2 records it as entry #6 of an 8-entry
circular-fix chain.

**Root cause:** two independent defects.

First, the requirement was never real. PathWhiz's own
`ProteinComplexProtein` declares
`validates :stoichiometry, allow_nil: true, numericality: {only_integer: true}`
over a nullable `protein_complex_proteins.stoichiometry` column, and
`lib/pwml_parser.rb` skips any node whose content is blank. An unstated
coefficient is valid PathWhiz. The pipeline was enforcing a constraint the
target system does not have — deliberately so, since `bound_elements` *does*
declare `presence: true`. Worse, `build_pwml_ir` *dropped* the offending
component, which manufactured an empty complex and violated the rule PathWhiz
actually has (`protein_complex_proteins, length: {minimum: 1}`).

Second, five enforcement points each owned a private copy of the rule and
disagreed: `map_ids._component_stoichiometry` defaulted to 1;
`ir._component_stoichiometry` and `writer._component_stoichiometry` returned
None; `build_pwml_ir`, `validate_required_pwml_contract` and
`validate_pwml_ir` each re-implemented the check; and `pwml/qa.py` rejected an
empty value outright. A fix landing in one copy left the others intact — the
re-divergence mechanism named in `docs/pathwhiz_requirements.md` §5 items 2–3.

The deadlock was then held in place by policy: Stage 4a's
`_defer_complex_stoichiometry_patches` refuses (correctly) to fabricate a
count and defers to audit, while the audit prompt instructed the model to
"leave unresolved stoichiometry as an error for review". A paper that never
states subunit counts — the normal case — could therefore never satisfy the
gate, and each run burned an enrichment round-trip on a patch that would be
discarded anyway.

**Fix:** one shared `component_stoichiometry` in `entity_identity.py` (the
module `map_ids` and `ir` already both import) returns an explicit count or
None; mapping, IR and the writer now delegate to it instead of re-deriving it.
An unstated count is left blank end-to-end rather than assumed: `map_ids`
omits the field, `build_pwml_ir` keeps the component and omits the key, the
writer emits `<stoichiometry nil="true" type="integer"/>` via the existing
`_append_scalar` nil path, and `qa.py` skips nil nodes exactly as its
neighbouring `hmdb-id` check already did. All three IR validators now warn
(`component_stoichiometry_unstated`) instead of erroring, the writer no longer
raises, deterministic audit reports an error only when evidence gives an exact
count it can act on, and `gap_resolver` no longer holds a complex open on an
unstated count. A stated value is still preserved verbatim, and nothing infers
one.

Separately, the biological-state gate required species *and* subcellular
location as hard errors; PathWhiz's `BiologicalState#has_at_least_one_component`
requires one of species/tissue/cell_type/subcellular_location. Only a fully
empty state is now fatal; individual gaps are warnings.

**Verified:** the previously blocked payload exports (`ok = True`, zero QA
errors) with `NdmCDE` retaining all three members at `nil="true"`, `Cdh`
retaining one, and `NdmA complex` still carrying its explicit `1`. Suite: 359
passed, 0 failed (baseline 358/1; the 55 errors are a pre-existing `tmp_path`
environment fault, unchanged).

**Pipeline consistency:** field ownership is unchanged — Stage 1 records
stated evidence, Stage 4 may still patch a count from explicit evidence, and
Stage 8 remains the export authority. What changed is that the rule now has a
single definition, and that definition matches PathWhiz ground truth rather
than a stricter invention. This is the §5 item 2/3 collapse applied to one
concrete field: unknown stays blank, as the Unknown-protein sentinel already
does for identity.

---

### 2026-07-15 — Wrap name-only protein complexes with the PathBank Unknown sentinel

**Files changed:** `src/t2pw/mapping/map_ids.py`, `src/t2pw/pwml/ir.py`,
`tests/test_pathbank_unknown_fallback.py`, `tests/test_pwml_ir.py`,
`docs/pipeline.md`, `docs/change_log.md`.

**Error / symptom:** Raw, uncaught `ValueError` in `writer.py`:
`PWML export failed: Protein complex 'oxoglutarate dehydrogenase complex' has
no protein_complex-proteins to export.` — a crash deep in
`_protein_complex_members` instead of a clean export-time error, for a
`protein_complexes[]` row that reached final PWML serialization with an empty
`components` list.

**Root cause:** the 2026-07-14 "NAME-BASED COMPLEX RULE" fix (above)
correctly routes any entity whose name contains "complex" into
`entities.protein_complexes[]` even when the source paper never enumerates
subunits, using `components: []` intentionally in that case. Nothing
downstream reliably backfills that for every row: Stage 4a's gap-resolver can
only fill components from evidence text that names real subunits, and Stage
6's existing `_apply_pathbank_unknown_enzyme_fallback` — the established
pattern for "genuinely unresolvable, must not block export" proteins — is
actor-driven; it only walks `processes.reactions[].enzymes[]` and
`processes.transports[].transporters[]` looking for an exact-name match, so a
`protein_complexes[]` row never tied to exactly one reaction/transport actor
sails through untouched. Separately, `validate_pwml_ir` — the gate the
writer and `run_pwml_export` both call before serialization — only logged
`protein_complex_missing_components` as a `warning`, so `ok` stayed `True`
and nothing stopped the payload before the writer's hard crash.

**Fix:** two changes. (1) Added
`_apply_pathbank_unknown_complex_fallback` in `map_ids.py`, structurally
parallel to `_apply_pathbank_unknown_enzyme_fallback` but entity-driven: it
scans `entities.protein_complexes[]` directly, and for any row whose
`components` is still empty after normal mapping/gap-resolution/the
actor-driven fallback have all run, and which has no real complex-level
PathBank ID (`pathbank_complex_id`/`pathbank_protein_complex_id` on the row
or its `mapping_meta`), attaches a single component built from PathBank's
`Unknown` sentinel protein (id `9659`, species *Arabidopsis thaliana*),
registers that protein in `entities.proteins`, and stamps `mapping_meta`
with `chosen_rule=pathbank_unknown_protein_fallback`,
`fallback_reason="complex_has_no_resolvable_components"`, and
`cross_species_placeholder=true`. It runs only when
`allow_complex_wrapper_creation=True`, wired in immediately after the
existing actor-driven fallback call in `map_payload`, and tracks a new
`complex_missing_components_unknown_fallbacks` summary counter. (2) In
`validate_pwml_ir` (`ir.py`) and the writer's own `_protein_complex_members`
(`writer.py`), a complex still at zero components now only fails cleanly if
it also has no real, confirmed complex-level PathBank identity
(`pathbank_complex_id`/`pathwhiz_id` on the IR record) — checked directly
against `reference/PW1.pwml`, a real prior PathBank export, which contains
two `<protein-complex>` records (e.g. "alanine aminotransferase (ALT)",
`pwp-id PW_P000036`) with a genuine identity and a self-closing, empty
`<protein_complex-proteins/>`. SPMDB does not itself require every complex to
have a member; only a complex with no real identity of its own (which, after
Fix (1), should only be a generated wrapper that somehow missed even the
Unknown-sentinel fallback) is required to. `validate_pwml_ir` now raises
`error` only in that case and `warning` otherwise (previously an unconditional
`error`, which would have wrongly blocked a real, confirmed complex whose
PathBank record legitimately lists no subunits); `_protein_complex_members`
takes a matching `allow_empty` flag driven by the same check, so it no longer
raises unconditionally either. `build_pwml_ir`'s own internal bookkeeping
warning and `validate_required_pwml_contract`'s already-correct
generated-complex-only strictness (a Stage 3 pre-remap check, which runs
before any complex has a real ID to check) were both left untouched.

**Pipeline consistency:** this stays entirely inside Stage 6's existing
wrapper-creation ownership (`map_ids.py`, gated by
`allow_complex_wrapper_creation`, the sole module allowed to create
wrappers) and Stage 8's validate-only role (`ir.py`'s `validate_pwml_ir`
gates export, it does not mutate the payload). It does not reach into Stage
3's pre-remap gate or Stage 4/4a, matching "stages are independent." This is
the closing case of the same PathBank-Unknown-sentinel fallback family as
the 2026-07-13 Stage 6 entry and the 2026-07-14 "Widen the Stage 6 PathBank
Unknown fallback to cover transporter-only proteins" entry above — those
made the sentinel reachable for every *actor role*; this makes it reachable
for every *entity*, regardless of whether an actor ever references it.

**Verification:** `tests/test_pathbank_unknown_fallback.py` gained
`test_orphan_named_complex_referenced_as_enzyme_gets_wrapped` (actor-driven
and entity-driven passes agree for a referenced complex),
`test_orphan_named_complex_not_referenced_anywhere_still_gets_wrapped` (the
case the old actor-driven-only fallback missed),
`test_stage2_wrapper_disabled_does_not_wrap_orphan_complex` (disabled when
`allow_complex_wrapper_creation=False`), and
`test_complex_with_real_pathbank_id_is_not_wrapped_despite_empty_components`
(a real DB identity is never overwritten), and
`test_real_pathbank_complex_with_no_listed_components_exports_with_empty_members`
(a hand-built payload shaped exactly like `reference/PW1.pwml`'s ALT complex —
real `pathbank_complex_id`, empty `components`, never touched by `map_payload`
at all — passes Stage 3, `validate_pwml_ir` (as a warning, `ok=True`), and the
writer, producing an actual empty `<protein_complex-proteins/>` element,
proving Fix (2)'s leniency holds independently of Fix (1)). `tests/test_pwml_ir.py`
gained `test_validate_pwml_ir_errors_on_protein_complex_missing_components`
(no real identity → still an error) and kept
`test_protein_complex_unresolved_component_is_exportable_with_warnings`
passing under the corrected, identity-based rule (a complex with a real
`pathbank_complex_id` whose one listed component fails to resolve still ends
up exportable with a warning, not an error, matching the ALT-complex
precedent — component-level identity resolution failures are a separate,
already-covered concern from "does this complex exist"). Full suite re-run:
414 passing.

---

### 2026-07-14 — Tighten default reaction scope and wire the out-of-scope reaction filter

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `src/t2pw/app/streamlit_app.py`,
`tests/test_pipeline_cleanup.py`, `tests/test_streamlit_stage2_orchestration.py`,
`docs/change_log.md`.

**Error / symptom:** For pathway-dense topics (e.g. the TCA cycle), extraction returned far
more reactions than the paper's actual core pathway — anaplerotic, cataplerotic, and
auxiliary reactions mentioned only as background context were included in the final PWML
output alongside the pathway's defining steps.

**Root cause:** Two compounding gaps. (1) `pathway_scope` is a real parameter on
`run_extraction_pipeline` that the prompt's `scope_membership` rule depends on, but no live
caller ever populates it. `pwml_system.txt`'s own scope rule only defined strict
`core`-only behavior for "no `pathway_scope` AND no `upstream_context`," leaving the actual
common case — no `pathway_scope`, `upstream_context` present, true on every live run —
undefined, so the model had no explicit instruction to stay tight and would reach for the
full core/anaplerotic/cataplerotic/auxiliary taxonomy. (2) `filter_out_of_scope_reactions()`,
whose entire job is to drop reactions tagged `scope_membership: out_of_scope`, existed fully
implemented in `pipeline.py` but was never called anywhere in the live orchestrator, so even
a reaction the model correctly tagged `out_of_scope` remained in the payload through export.

**Fix:** `pwml_system.txt`'s scope rule now applies strict `core`-only classification
whenever no explicit `<pathway_scope>` is supplied, regardless of whether `upstream_context`
is present — the anaplerotic/cataplerotic/auxiliary labels are reserved for when a
`<pathway_scope>` block explicitly requests that broader taxonomy. `streamlit_app.py` now
calls `filter_out_of_scope_reactions()` on the merged Stage 1 payload immediately after
Stage 1's structural contract validates (and after `write_stage1_lock_artifacts` has already
captured the raw output and lock manifest), and before that payload reaches Stage 2
inference.

**Pipeline consistency:** Verified against `reaction_lock_manifest.py`:
`build_locked_reaction_manifest()` already excludes `scope_membership == "out_of_scope"`
reactions from locking, so the lock/preservation contract
(`reaction_preservation_validator.py`) was already designed assuming these reactions get
removed from the payload — this fix completes that assumption rather than introducing a new
one. No locked reaction is affected. Filtering between Stage 1 and Stage 2 is orchestrator-owned
cross-stage logic per `docs/pipeline.md` ("logic spanning two stages belongs in the
orchestrator"); `filter_out_of_scope_reactions` itself, already Stage-1-owned, was untouched.

**Verification:** New tests in `tests/test_pipeline_cleanup.py` (filter behavior in
isolation) and `tests/test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference`
(AST-verified call ordering). Full suite: 408 passing.

---

### 2026-07-14 — Replace unreachable multi-example-review branches with defensive text

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`,
`src/t2pw/llm/prompts/pwml_infer_system.txt`, `docs/change_log.md`.

**Error / symptom:** Both the Stage 1 and Stage 2 prompts contained a documented branch
("Case A" / "Rule 1") instructing the model to proceed with extraction/inference when
`document_type == "multi_example_review"` and `selected_example` is empty, claiming "the
upstream pipeline gate has already approved this text." This is not true of production
behavior.

**Root cause:** `is_ambiguous_multi_example_review_context()` in `preprocessor.py`, called
from `run_extraction_pipeline` and `run_inference_pipeline` in `pipeline.py`, hard-aborts the
entire pipeline with `PipelineFailure` for exactly this condition, before either stage's LLM
is ever called. The prompt branches described a state the orchestrator guarantees can never
reach the model in normal operation — dead, misleading documentation that could confuse a
future prompt editor into thinking the fallback path is live and tested.

**Fix:** Both branches now state plainly that the production orchestrator does not invoke the
stage in this state, and specify defensive behavior (return an empty extraction/no additions
with a warning flag) only for the hypothetical case of a direct or manual call that bypasses
the orchestrator gate.

**Pipeline consistency:** Prompt-text-only change; no orchestrator or normalization logic
touched. This is documentation truth-alignment to the orchestrator's actual, already-correct
hard-stop behavior, not a behavior change.

**Verification:** No automated test covers LLM prompt text in this repo (confirmed via
`rg "pwml_infer_system" tests` — zero matches); verified by inspection against `pipeline.py`'s
actual gate logic.

---

### 2026-07-14 — Standardize the transporter actor schema in the Stage 1 extraction prompt

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/change_log.md`.

**Error / symptom:** `pwml_system.txt` showed three different shapes for
`transports[].transporters[]` actor rows within the same file: `{"protein_complex": "..."}`
in the formal OUTPUT JSON SCHEMA, and `{"protein": "..."}` in Example 3 — neither matching
the canonical `entity`/`entity_type`/`role` actor shape `docs/pipeline.md` documents for
every other actor list (`reactions[].enzymes`, `reactions[].modifiers`,
`interactions[].participants`).

**Root cause:** The same bug class as the 2026-07-07 fix to `normalize_process_actor_schema`
for `reactions[].enzymes` (that entry: "all enzyme actor dicts ... retained
`protein_complex` ... as the name field" because the prompt/schema was inconsistent) —
transporters had simply never been brought in line with the canonical actor shape.

**Fix:** Both occurrences in `pwml_system.txt` now use the canonical
`{"entity": "", "entity_type": "protein | protein_complex", "role": "transporter", ...}`
shape.

**Pipeline consistency:** Confirmed by direct code read that `normalize_process_actor_schema`'s
`_rewrite_actor_rows` helper (`process_normalizer.py:2509-2515`) already canonicalizes
`transports[].transporters[]` to `entity`/`entity_type` regardless of which legacy key Stage 1
emits (it reads `entity`/`protein`/`protein_complex`/`name` as fallback fields) — this fix
changes zero bytes of what reaches export; it only removes an internally-inconsistent prompt
that could lead a weaker model to pick the wrong key or omit `entity_type`/`role`.

**Verification:** Inspected `process_normalizer.py` directly to confirm the downstream
migration path already exists and is unaffected. No automated test covers LLM prompt text.

---

### 2026-07-14 — Make Stage 1 prompt examples schema-valid and consolidate complex-routing guidance

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/change_log.md`.

**Error / symptom:** Two related prompt-quality gaps in `pwml_system.txt`: (1) all four
MODIFIER EXAMPLES omitted required top-level reaction/interaction/transport fields
(`biological_state`, `class`, `scope_membership`, `confidence`, `provenance`,
`source_refs`) that the formal OUTPUT JSON SCHEMA requires — a weaker model imitating the
examples rather than the full schema block could emit incomplete objects. (2) protein-vs-
protein_complex routing guidance (NAME-BASED COMPLEX RULE, extraction-layer cofactor-rule
bullets) was scattered across roughly 450 lines of the file with no single, locally-visible
decision procedure — the same root cause named in the earlier 2026-07-14 "Stop LLM extraction
from routing 'X complex' entities into proteins[]" entry ("that guidance was buried among many
other extraction rules").

**Fix:** All four MODIFIER EXAMPLES (enzyme catalyst, regulator/catalyst with interactions,
transporter, protein complex catalyst) now include every field the formal schema requires, so
each is a complete, valid instance of its schema rather than an abbreviated illustration.
Added a new "PROTEIN-COMPLEX DECISION CHECKLIST" section immediately before the examples,
consolidating the existing scattered rules (name-contains-"complex" check, explicit
multi-subunit language check, components explicit-vs-unresolved handling, never infer a
complex for export-wrapper reasons) into one explicit, numbered procedure. The existing
scattered rules were left in place as reinforcement rather than removed, to avoid risking any
rule a downstream check implicitly depends on the model having seen phrased a specific way.

**Pipeline consistency:** Prompt-text-only; no schema, normalizer, or export code touched. The
checklist restates existing policy already enforced downstream by Stage 3's NAME-BASED
COMPLEX RULE gate and Stage 6's generated-wrapper contract — it does not introduce new policy.

**Verification:** No automated test covers LLM prompt text; verified by inspection that every
added field matches `t2pw/schema.py`'s existing TypedDict definitions exactly (no new fields
invented).

---

### 2026-07-14 — Remove the live Homo sapiens organism default from the Stage 2 inference prompt

**Files changed:** `src/t2pw/llm/prompts/pwml_infer_system.txt`, `docs/change_log.md`.

**Error / symptom:** `pwml_infer_system.txt` section E ("Biological state and location
linking") still instructed the model to default an unresolved organism to "Homo sapiens" —
directly contradicting Stage 1's own species rule ("if no organism can be confidently
selected, leave species empty — do not guess") and Stage 0's preprocessor, both of which
already forbid organism guessing. This is the same contradiction the 2026-07-08 change log
entry fixed for Stage 1's own BIOLOGICAL STATE RULE — that fix never reached this second copy
in the Stage 2 prompt.

**Root cause:** The 2026-07-08 fix ("Strengthen extraction scoping...") only touched
`pwml_system.txt` and added a species cross-reference note to `pwml_infer_system.txt`'s
modifier-linking section; the separate default-organism line in that same file's
biological-state/location section was never located or updated.

**Fix:** Replaced the Homo sapiens default with an explicit priority order (upstream-selected
organism → locally-evidenced organism → empty) and an explicit prohibition on defaulting,
noting that an unresolved species is a valid Stage 3 gate finding the Stage 4 audit loop is
designed to repair from real evidence.

**Pipeline consistency:** Prompt-text-only; matches the already-existing Stage 3 species gate
(added 2026-07-08, "Add protein species and external identity checks to Stage 3 gate") plus
the Stage 4 audit repair path, both unaffected. No code touched.

**Verification:** No automated test covers LLM prompt text; confirmed the pre-existing
uncommitted section C (alias/synonym bridging) addition to this same file — added by
concurrent, unrelated work on cofactor charge-notation canonicalization — was left untouched.

---

### 2026-07-14 — Retire dead prompt files and their dead loader code

**Files changed:** `src/t2pw/llm/prompts/extract_json.md` (deleted),
`src/t2pw/llm/prompts/repair_json.md` (deleted),
`src/t2pw/llm/prompts/enrichment_system.txt` (deleted),
`src/t2pw/curation/gap_resolver.py`, `docs/change_log.md`.

**Error / symptom:** Three prompt files sat in `src/t2pw/llm/prompts/` with no live reference
anywhere in the codebase: `extract_json.md` and `repair_json.md` were both 0 bytes and
unreferenced by filename anywhere in `src/`. `enrichment_system.txt` was a real, substantive
prompt (patch-based, non-agentic enrichment) loaded only by `_get_enrichment_system_prompt()`
in `gap_resolver.py`, a function that was itself never called anywhere — the live Stage 4a
enrichment path uses the separate, actually-wired `enrichment_agentic_system.txt` /
`_get_enrichment_agentic_system_prompt()`.

**Root cause:** Vestigial from an earlier prompt-per-stage design iteration; nothing removed
them when the live enrichment path moved to the agentic/tool-calling variant.

**Fix:** Deleted all three files after independent re-verification (fresh repo-wide grep, not
reliance on prior analysis) confirmed zero references. Removed the now-fully-dead
`_get_enrichment_system_prompt()` function and its module-level `_ENRICHMENT_SYSTEM_PROMPT`
global from `gap_resolver.py`. Left `src/t2pw/config.py` untouched: although it is 0 bytes,
`src/config.py` (a separate legacy shim) does `from t2pw.config import *`, so deleting it
would break that shim's import — this dependency was found during re-verification and the
file was correctly left in place.

**Pipeline consistency:** Removes dead code only; no live prompt, stage function, or call path
touched. `enrichment_agentic_system.txt` (the live Stage 4a prompt) was explicitly left
untouched and re-verified as the sole caller-reachable enrichment prompt.

**Verification:** Repo-wide grep for each deleted filename and the removed function/global
name returned zero remaining source references. `python -m py_compile
src/t2pw/curation/gap_resolver.py` succeeded.

---

### 2026-07-14 — Charge-notation-aware alias canonicalization; interaction registry coverage

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/llm/prompts/pwml_infer_system.txt`, `src/t2pw/llm/prompts/pathway_curator_system.txt`,
`src/t2pw/curation/audit_json_llm.py`, `tests/test_process_normalizer.py`,
`docs/pipeline.md`, `docs/change_log.md`.

**Error / symptom:** Stage 8 IR construction failed with 14 `Process member 'X' was
not found in entity registries.` errors (for `NAD`, `NADP`, `Ca2`) and 3
`Interaction must have exactly one left and one right member.` errors, on a TCA-cycle
paper run. `entities.compounds` correctly declared the redox-specific species
(`nad+`, `nadh`, `nadp+`, `nadph`), but 9 reactions referenced the bare, ambiguous
tokens `"NAD"`/`"NADP"`, and two interactions were self-referential `SAME_AS`
declarations (`entity_1 == entity_2 == "NAD"`/`"NADP"`, evidently a failed attempt to
declare "NAD" and "NAD+" as synonyms) plus one interaction (`Ca2+ activates IDH and
OGDH`) referenced calcium, which was never extracted as a compound at all.

**Root cause:** four compounding issues, none of which is a "missing entity" problem
for NAD/NADP:

1. `process_normalizer._normalize()` stripped the `+` character while
   `t2pw.pwml.ir._norm()` (Stage 8) preserved it. Stage 3/4's registry check
   (`validate_registry_references`, exposed to audit via `_stage3_validation_issues`)
   therefore treated `"NAD"` and `"nad+"` as the same name and never flagged the
   mismatch — the payload reached Stage 8 looking clean, where the stricter,
   charge-aware `_norm()` correctly rejected it with no repair path left.
2. `apply_biochemical_aliases` (step 1) already had the exact fix in
   `BIOCHEMICAL_ALIAS_MAP` (`"nad": "NAD+"`, `"nadp": "NADP+"`) and already rewrote
   reaction inputs/outputs, but never touched `processes.interactions[]` participants
   or `processes.transports[].cargo` — the same class of bare compound reference was
   simply never in scope for this pass.
3. Independently of (1)/(2), `_rewrite_token` and `_token_parts_for_aliasing` (used by
   step 11, `canonicalize_same_as_aliases`, which runs on every payload regardless of
   whether it contains a `SAME_AS` interaction) tested for a composite `"A + B"`
   token with a raw `"+" in text` check instead of the charge-aware `_has_plus_token`
   guard already used elsewhere (`normalize_composites`). This silently mangled any
   correctly-charged compound name — `"NAD+"` was split on `"+"` into `["NAD", ""]`,
   the empty part dropped, and the single remaining part rejoined with no `+` at all.
   This is the actual, deterministic reason `"NAD+"` never survived to Stage 8 even
   after `apply_biochemical_aliases` had just correctly produced it.
4. `validate_registry_references` never checked `processes.interactions[]` at all
   (only reactions and transports), so the genuinely-missing `Ca2+` compound, and the
   self-referential `SAME_AS` rows, had no path to becoming a Stage 3 gate failure —
   they were invisible to the audit loop and only surfaced as a hard Stage 8 abort.

**Fix:**
1. `_normalize()` now preserves `+`, matching `ir.py`'s `_norm()`.
2. `apply_biochemical_aliases` now also rewrites `processes.interactions[]`
   (`entity_1`/`entity_2`/`left`/`right`/`source`/`target`) and
   `processes.transports[].cargo`/`cargo_complex`.
3. `_rewrite_token` and `_token_parts_for_aliasing` now use `_has_plus_token` instead
   of a raw `"+" in text` check, so charge notation on compound names is never
   mistaken for a composite separator during alias canonicalization.
4. `canonicalize_same_as_aliases` now drops a `SAME_AS` interaction whose two sides
   normalize to the same name after rewriting (including a degenerate declaration
   that was self-referential to begin with) instead of carrying forward an inert
   self-interaction.
5. `validate_registry_references` now checks `processes.interactions[]` participants
   against the registry, mirroring the existing reactions/transports coverage.
6. `_entity_name_norms` now also recognizes an entity's declared `synonyms`, matching
   `ir.py`'s existing alias resolution, so a curator/audit-proposed synonym patch is
   honored by the gate and not only by export.
7. Prompt updates (defense in depth, not required for the deterministic fix above):
   the Stage 2A inference prompt now requires `SAME_AS` pairs to use two distinct
   literal strings and prefers the deterministic charge-form directly for known
   cofactors; the Stage 5 curator prompt now covers interaction participants (not
   just reaction inputs/outputs) and prefers an entity `synonyms` patch over editing
   a reaction/interaction reference when the entity's declared name is already
   correct; the Stage 4 audit system prompt gained equivalent "registry reference
   mismatch" guidance (synonym-patch first, new-entity only when genuinely absent,
   remove degenerate self-referential `SAME_AS` rows).

**Pipeline consistency:** all deterministic changes stay inside
`process_normalizer.py`, which owns Stage 3's alias canonicalization and gate. None
of them reach into Stage 6/8 or invent biology — they only make an existing,
already-tested deterministic mechanism (biochemical alias rewriting, same-as
canonicalization, registry validation) correctly cover interactions and correctly
preserve chemically-significant `+` notation, so a repairable naming issue is caught
and fed to Stage 4 audit (per "the gate is not a blocker before audit") instead of
surfacing as an unrepairable Stage 8 abort. The genuinely-missing `Ca2+` entity is
still left as an audit-owned gap — no stage invents the missing compound.

**Verification:** `tests/test_process_normalizer.py` gained
`test_apply_biochemical_aliases_rewrites_interaction_and_transport_participants`,
`test_validate_registry_references_flags_unknown_interaction_participant`,
`test_validate_registry_references_allows_known_interaction_participant`,
`test_validate_registry_references_recognizes_declared_synonyms`,
`test_canonicalize_same_as_aliases_preserves_charge_notation`,
`test_canonicalize_same_as_aliases_drops_noop_same_as_interaction`, and
`test_full_normalization_resolves_bare_cofactor_and_flags_missing_ion` (an
end-to-end reproduction of the exact TCA-cycle payload shape: bare "NAD" input
resolves to "NAD+", the self-referential alias interactions are dropped, and the
genuinely-missing Ca2+ reference correctly surfaces as a gate error). Verified
directly against the failing run's saved `tmp/final.mapped.json`: before the fix,
`validate_registry_references` raised on all 14 tokens exactly matching the reported
Stage 8 errors; after the fix, a reduced reproduction of the same reactions/
interactions normalizes to zero gate errors except the genuine Ca2+ gap. Full suite
re-run: 405 passing (398 previously existing + 7 new).

---

### 2026-07-14 — Prefer a confident Stage 6 DB complex match's components over stale extraction data

**Files changed:** `src/t2pw/mapping/map_ids.py`, `tests/test_map_ids.py`,
`docs/pipeline.md`, `docs/change_log.md`.

**Error / symptom:** Stage 8 export error `Component[0] in complex 'pyruvate
dehydrogenase complex' is missing stoichiometry.` (and `[1]`, `[2]`) — for a
complex that Stage 6 had, in the same run, successfully matched to a real
PathBank record. This surfaced immediately after the 2026-07-14 "NAME-BASED
COMPLEX RULE" fix (below) started correctly routing this entity into
`entities.protein_complexes` for the first time; it had never reached this
code path before because it used to be misfiled under `proteins[]` with no
`components` at all.

**Root cause:** two upstream stages compound into a gap Stage 6 didn't cover.
Stage 1 extracts `components` as plain subunit-name strings (its schema has no
concept of stoichiometry). Stage 4a's gap-resolver
(`gap_resolver.py:_resolve_declared_complex_components`) tries to attach a
protein identity to each subunit by name-matching against
`entities.proteins`; when a subunit was never separately extracted as its own
protein row (true here — the paper only names E1/E2/E3 as complex members),
that lookup fails, but the function *still* unconditionally upgrades the
plain string into a dict (`{"name": ...}`) and writes it back, flagging
`missing_stoichiometry` with `resolution_owner: "audit"`. Stage 4's
deterministic audit rule can only backfill stoichiometry from an *explicit*
per-subunit count stated in the evidence text (its precedent case: "three
NdmC, three NdmD..."); this paper's evidence never states a count, so nothing
fills it in. Stage 6 then DB-matches the complex to a real PathBank record
(`pathbank_complex_id`) whose components carry real, correct stoichiometry —
but the mapping loop in `map_payload` was hard-coded to keep
`complex_row`'s existing (by-then broken) components whenever they were
non-empty, discarding the DB match's authoritative data outright.

**Fix:** in `map_payload`'s per-complex loop, when the Stage 6 mapping result
has `status == "mapped"` — a confident match via direct
`pathbank_protein_complex_id`, name+species, or resolved-component-species —
the DB-hydrated `result["components"]` (already reconciled against local
`entities.proteins` earlier in the same loop iteration) now overwrites
`complex_row["components"]` outright. Every other outcome (`unmapped`,
`ambiguous`, `novel`, and the PathBank `Unknown`-sentinel fallback, all of
which carry a non-`"mapped"` status) is unaffected — extraction/gap-resolver
components are still preferred there, since there is no more-authoritative
DB version to prefer.

**Pipeline consistency:** this stays entirely inside `map_ids.py`, the sole
Stage 6 module. The two upstream gaps that let a stoichiometry-less dict
component reach Stage 6 in the first place (Stage 4a always promoting
strings to dicts without a stoichiometry fallback, and Stage 4's audit only
backfilling from explicit textual counts) were deliberately left untouched —
this fix does not reach into Stage 3, 4, or 4a, per "stages are independent."
It only changes which of two already-available component lists Stage 6
prefers when it has legitimate grounds (a confident DB identity) to prefer
one over the other.

**Verification:** `tests/test_map_ids.py` gained
`test_confident_db_complex_match_overrides_stale_extraction_components`
(a complex with plain-string, unresolvable components and a mocked confident
DB match ends up with the DB's stoichiometry-bearing components) and
`test_unconfident_db_complex_match_keeps_extraction_components` (an
ambiguous/unmapped result leaves the original extraction components
untouched — current behavior preserved). No existing test asserted the old
precedence (checked every test that mocks Stage 6 complex-matching). Full
suite re-run: 398 passing.

---

### 2026-07-14 — Stop LLM extraction from routing "X complex" entities into proteins[]

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/pipeline.md`,
`docs/change_log.md`.

**Error / symptom:** Stage 3 gate error `Generated protein complex wrapper
'pyruvate dehydrogenase complex' must be listed under protein_complexes, not
proteins.` for an entity that was never a pipeline-generated wrapper at all —
it is a real, well-known multi-subunit enzyme complex.

**Root cause:** the Stage 1 extraction prompt already told the LLM to use
`protein_complexes[]` when the source text "explicitly supports a complex,"
but that guidance was buried among many other extraction rules and had no
single unmissable rule tied to the literal entity name. The model extracted
an entity named "...complex" directly into `proteins[]`. The Stage 3 gate
that reports this (`process_normalizer.py:3766`, unchanged by this fix)
correctly detects any `proteins[]` row named "...complex" as suspicious, but
it is detection-only — there is no auto-repair step for this class of
misclassification (only pipeline-generated wrapper duplicates are guarded).

**Fix:** added one explicit, mandatory rule to the Stage 1 prompt: any entity
whose own name contains the word "complex" must be extracted under
`protein_complexes[]`, never `proteins[]`, even when the source text does not
enumerate subunits (in which case `components: []` and confidence `< 1.0` are
used, mirroring the existing "unknown subunit membership" convention already
in the same prompt).

**Pipeline consistency:** this is a Stage 1 (Extract) prompt change only —
the entity-type decision belongs at extraction, where `PayloadProtein` vs.
`PayloadProteinComplex` is first assigned. It does not touch Stage 3's gate
logic, Stage 6's remap logic, or any other stage's module, per the "stages
are independent" design principle.

**Verification:** no automated test covers LLM prompt text in this repo
(confirmed via `grep -rn "pwml_system.txt" tests/` — no matches); the fix was
verified by inspection against the file's existing rule conventions and by
re-running the full test suite to confirm nothing else references or
snapshots this file's content.

---

### 2026-07-14 — Widen the Stage 6 PathBank Unknown fallback to cover transporter-only proteins

**Files changed:** `src/t2pw/mapping/map_ids.py`,
`tests/test_pathbank_unknown_fallback.py`, `docs/pipeline.md`,
`docs/change_log.md`.

**Error / symptom:** Stage 3 gate error `Protein 'ABCG-116' is missing a
UniProt or DrugBank identifier.` for a protein that legitimately could not be
matched to a real identifier — the same situation the Stage 6 PathBank
`Unknown` sentinel fallback exists to handle for enzymes, but this protein's
only role in the payload was as a transporter, not a reaction catalyst.

**Root cause:** `_apply_pathbank_unknown_enzyme_fallback` in `map_ids.py` was
deliberately scoped to reaction enzymes only (see the 2026-07-13 Stage 6
entry above and `docs/pipeline.md` Stage 6 section). `_has_non_enzyme_reference`
disqualified any protein referenced outside a catalyst role — including as a
transporter — from ever receiving the fallback, so an unresolved
transporter-only protein was left with zero identifiers and no path to a
valid export state.

**Fix:** generalized the disqualification check to accept a caller-specified
"allowed role" (`enzyme` or `transporter`) instead of hard-coding "enzyme."
Added a second pass over `processes.transports[].transporters[]` that applies
the identical guards already used for enzymes (only after real mapping
strategies fail, never overrides a real mapping, reused/deduplicated Unknown
sentinel, excluded once any other disqualifying reference exists) and rewrites
a qualifying transporter entry to reference the generated Unknown-backed
complex the same way a qualifying reaction enzyme entry is rewritten. A
protein referenced as a transporter *and* anywhere else disqualifying (reaction
input/output, non-catalyst/non-transporter modifier, interaction, complex
component) remains excluded, matching the existing enzyme-side behavior. A
new `transporter_unknown_fallbacks` counter tracks this path separately from
the existing `reaction_enzyme_unknown_fallbacks` counter so no existing
caller's assertion on that counter changes meaning.

**Pipeline consistency:** this stays entirely inside `map_ids.py`, the sole
Stage 6 module allowed to call `map_payload(..., allow_complex_wrapper_creation=True)`.
It does not touch Stage 3's gate (which correctly just reports the symptom)
or any other stage. The fallback's core invariants are unchanged and merely
extended to a second, symmetric role — it still never applies to a protein
with any other kind of reference, and it is still the sole wrapper-creating
pass in the pipeline.

**Verification:** `tests/test_pathbank_unknown_fallback.py` gained
`test_unknown_fallback_wraps_transporter_only_protein` (transporter-only
unresolved protein gets wrapped and its transporter entry is rewritten to
`entity_type: "protein_complex"`) and
`test_unknown_fallback_excludes_transporter_referenced_elsewhere` (a
transporter also referenced via an interaction stays excluded, matching the
existing enzyme-side exclusion test). Full test suite re-run: all passing.

---

### 2026-07-14 — Retire the spontaneous-reaction flag; every reaction exports non-spontaneous

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`,
`src/t2pw/curation/audit_json_llm.py`, `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/pwml/ir.py`, `src/t2pw/pwml/qa.py`, `tests/test_pwml_ir.py`,
`tests/test_audit_json_llm_payload.py`, `docs/pipeline.md`,
`docs/change_log.md`.

**Error / symptom:** Stage 8 export error
`Reaction 'OPCL1-catalyzed CoA ligation of OPC-4:0' is marked spontaneous but
also has enzymes.` — Stage 1 extraction (or the Stage 4 audit's deterministic
enzyme-less rule) could mark a reaction `spontaneous: true` independently of
whether it also carried real enzyme references, and the Stage 8 semantic gate
(`ir.py`) rejected the combination outright with no repair path.

**Root cause:** the `spontaneous` field could be set from three independent
places (Stage 1 LLM extraction judgment, Stage 4's deterministic
enzyme-less-reaction rule, and manual/legacy payload data) with nothing
reconciling it against the reaction's actual enzyme list until the Stage 8
export gate, which only detects the conflict and aborts.

**Fix:** spontaneity is not modeled for now. Every source that could set
`spontaneous: true` was changed to never do so (Stage 1 prompt instruction,
Stage 4's deterministic audit rule removed), Stage 3's normalizer now forces
`spontaneous: false` on every reaction as its first step so the persisted
normalized payload is consistent, and Stage 8's IR builder hardcodes
`spontaneous: False` on export regardless of upstream payload content. The
now-unreachable Stage 8 mutual-exclusion check
(`spontaneous_reaction_has_enzymes`) was removed. The companion legacy XML QA
check in `qa.py` that required every non-spontaneous reaction to have an
enzyme was relaxed to match — an enzyme-less reaction is now expected, not an
error, since spontaneity can no longer be asserted to explain it.

**Pipeline consistency:** the enforcement point that matters for correctness
is Stage 8 (export), which now owns the invariant unconditionally rather than
validating an upstream assertion. Stage 1's prompt and Stage 4's audit rule
were also updated so the persisted payload stays consistent with what
actually exports, but Stage 8 does not depend on either of them having done
so correctly — it forces the value itself, per "broken stages must not
produce output" / each stage should not trust an earlier stage's optional
field to be correct.

**Verification:** `tests/test_pwml_ir.py::test_spontaneous_field_is_always_forced_false_on_export`
and `test_pre_export_and_qa_reject_duplicate_enzyme_complex_even_when_spontaneous_set`
(renamed/updated from the prior spontaneous-preserving tests) and
`tests/test_audit_json_llm_payload.py::test_deterministic_audit_does_not_mark_enzyme_less_reaction_spontaneous`
(renamed/updated) all pass, along with the full related test suite (96 tests
across `test_pwml_ir.py`, `test_audit_json_llm_payload.py`, and
`test_process_normalizer.py`).

---

### 2026-07-13 — Quarantine coarse reactions before orphan-protein validation

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`,
`tests/test_locked_noop_quarantine_policy.py`, `docs/pipeline.md`, and
`docs/change_log.md`.

**Error / symptom:** A new paper began with 21 locked source reactions. Stage 2
preserved all 21, but Stage 3 retained only nine and reported the other 12 as
missing. The 12 all used the same coarse compound label on both sides. The
`KlpD` claim, for example, was represented as `klebsazolicin -> klebsazolicin`.
After that row was silently removed, its unmapped `KlpD` protein remained and
the pre-export Stage 3 revalidation stopped on a missing UniProt/DrugBank ID.

**Root cause:** `dedupe_processes` equated identical/subset normalized labels
with a biochemical no-op and deleted the row without considering lock
accounting. It also ran after orphan cleanup, so an enzyme could become orphaned
only after the cleanup passes had finished. `essential` exempted self-loops from
classification, and distinct locked duplicates could be silently collapsed.
The Stage 6 PathBank `Unknown` fallback was correctly narrow: because the KlpD
reaction no longer survived, it had no valid reaction enzyme to process.

**Fix:** Stage 3 now classifies same-label and output-subset reactions before
the final orphan passes. Unsupported unlocked no-ops are dropped. Locked or
directly evidenced coarse reactions are removed from active processes and
written to the existing `quarantined_locked_reactions` ledger with stable reason
codes, the original reaction/provenance, source JSON pointer, and action. Neither
`locked` nor `essential` forces a biologically invalid equation into export.
Distinct locked duplicates retain one active representative and account for the
other lock in quarantine. The final orphan cleanup then removes unmapped
proteins used only by rejected reactions while retaining proteins referenced by
surviving reactions, interactions, transports, or complexes. Preservation
validation consequently treats the intended result as active plus quarantined,
with no silently missing locks. The strict Stage 3 gate now rejects a positive,
negative, or malformed `unaccounted_locked_reactions` value at
`/locked_reaction_filter_report/unaccounted_locked_reactions`; zero accounted
locks pass. After normalization and before audit, the live orchestrator also
rewrites canonical `tmp/quarantined_locked_reactions.json` from the normalized
ledger and returns it through post-pipeline JSON artifacts, so stale
pre-normalization content cannot mask quarantine results.

**Pipeline consistency:** Stage 3 owns deterministic reaction classification,
quarantine, and post-classification cleanup. Stage 4 may restore a quarantined
claim only from evidence that establishes distinct biochemical participants.
Stage 6 remains responsible for the narrow PathBank `Unknown` fallback only
after normal identity mapping fails for a confirmed catalyst on a valid
surviving reaction. Mapping failure does not remove that valid reaction, and the
fallback is not broadened to unrelated orphan proteins.

**Verification:** Focused regressions cover the KlpD-shaped locked/essential
self-loop, unlocked no-op removal, active-plus-quarantined preservation
accounting, duplicate locked IDs, post-quarantine orphan cleanup with all
surviving reference types, and Stage 6 fallback eligibility for an unresolved
enzyme on a valid distinct-participant reaction. Additional boundary tests cover
strict-gate accounting enforcement and post-normalization replacement of a
stale canonical quarantine artifact before audit.

---

### 2026-07-13 — Make Stage 8 validation-only and preserve Stage 6 enzyme wrappers

**Files changed:** `src/t2pw/app/streamlit_app.py`,
`src/t2pw/pipeline/process_normalizer.py`, `src/t2pw/pwml/ir.py`,
`tests/test_streamlit_stage8_export_contract.py`,
`tests/test_process_normalizer.py`, `tests/test_pwml_ir.py`,
`docs/pipeline.md`, and `docs/change_log.md`.

**Error / symptom:** PWML export of the saved final mapping stopped with
repeated bare-protein enzyme errors for actors such as `NdmA` and `NdmB`.
Inspection of `tmp/final.mapped.json` proved that Stage 6 had already done the
right work: it contained structurally valid generated single-protein PathWhiz
wrappers (`NdmA complex`, `NdmB complex`, and peers), and the reactions
referenced those wrappers.

**Root cause:** `run_pwml_export` reran the full Stage 3 normalizer after Stage
6. Reaction-evidence attachment compared exact actor names, did not recognize a
valid generated wrapper as equivalent to its sole member, and re-added bare
`NdmA`/`NdmB` actors beside their complex wrappers. Actor mirroring into both
`enzymes` and `modifiers`, together with a required-contract exemption, allowed
those bad actors through that gate; IR validation then emitted the repeated
bare-protein failures.

**Fix:** Stage 8 now performs validation only after optional grounding. It does
not rerun normalization, create autostates, attach/promote actors, map or infer
entities, or create wrappers. Evidence attachment is wrapper-aware and treats
only generated, structurally valid one-protein wrappers as equivalent to their
member, making supported reruns idempotent without conflating ordinary
biological complexes with proteins. The pre-export and direct-IR contracts now
reject bare catalytic proteins at their own actor pointers, accept one canonical
cross-field `enzymes`/`modifiers` mirror as a single logical enzyme, and still
reject duplicates within either field. IR construction likewise emits that
cross-field mirror only once.

**Pipeline consistency:** Stage 3 remains the sole full normalization stage,
and Stage 6 remains the sole owner of wrapper creation and reaction remapping.
Stage 8 validates and serializes the exact post-remap payload without repairing
it or inventing export structure.

**Verification:** With valid pathway metadata supplied, the exact saved
`tmp/final.mapped.json` now exports with `ok=true`, writes the output, and has
zero required-contract and IR-validation errors. The focused merged suite
passes 127 tests; the full suite passes 380 tests, and Ruff, compile, and diff
checks are green. A live UI click is the remaining manual check.

---

### 2026-07-13 — Repair named-complex resolution, audit convergence, and organism-aware locations

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/curation/gap_resolver.py`, `src/t2pw/curation/audit_json_llm.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_process_normalizer.py`,
`tests/test_gap_resolver.py`, `tests/test_gap_resolver_stage3_issues.py`,
`tests/test_gap_resolver_agent_tools.py`, `tests/test_audit_json_llm_payload.py`,
`tests/test_streamlit_stage2_orchestration.py`, `docs/pipeline.md`, and
`docs/change_log.md`.

**Error / symptom:** The `NdmCDE` paper run reached final remapping but stopped
at pre-export Stage 3 revalidation. Normalization duplicated the declared
complex as a bare protein, Gap Resolve skipped the complex as `issue_not_found`,
the audit loop stopped after one round despite gap progress, and bacterial
compounds received eukaryotic organelle locations.

**Root cause:** Protein synthesis did not consult the complex registry; the gap
executor indexed only proteins and compounds; convergence considered audit
patch counts but not gap-only changes/unresolved issues; audit had no
evidence-safe component-ratio repair; and location ranking used global frequency
without organism compatibility.

**Fix:** Stage 3 now preserves complex identity and gates protein/complex name
collisions. Gap Resolve executes complex issues, hydrates member identity from
declared mapped proteins, reports unsupported ratios to audit, treats a valid
novel complex ID as optional, and filters incompatible organelles. Stage 4 audit
adds exact component ratios only from unambiguous evidence (including the
`NdmCDE` `3/3/3` sentence) and canonicalizes positive legacy coefficients. The
orchestrator gates every changed settled payload, includes unresolved gap issues
in convergence, preserves loop bounds, and labels final failures as pre-export
Stage 3 revalidation.

**Pipeline consistency:** Stage 3 owns deterministic entity-type invariants;
Stage 4a owns targeted member/ID/location resolution; Stage 4 audit owns
evidence-dependent ratios; the orchestrator owns convergence; Stage 6 remaps;
and Stage 8 remains the export guard. No stage invents missing biology.

**Remaining validation:** A fresh live run completed the configured DB/LLM
stages and produced the final Stage 6 artifact. The separate Stage 8 regression
found on export is fixed above, and the exact saved artifact now passes
programmatic PWML export. Only a final manual export click in the live UI
remains.

---

### 2026-07-13 — Process-aware extraction contracts and visible boundary failures

**Files changed:** `src/t2pw/pipeline/stage_contracts.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_stage_contracts.py`,
`tests/test_streamlit_stage2_orchestration.py`, `docs/pipeline.md`, and
`docs/change_log.md`.

**Error / symptom:** Clicking **Run audit and DB mapping** could stop with only
"Every extracted process must include inputs, outputs, or cargo." Database
mapping and audit never ran, the structured issue pointer was hidden, and a
valid interaction could trigger the same reaction-oriented error.

**Root cause:** The post-extraction contract applied one reaction/transport
participant rule to every process bucket. The Streamlit handlers also caught
`StageContractError` as a generic exception and displayed only its summary
message, despite the exception carrying a structured report.

**Fix:** Post-extraction validation now dispatches bucket-specific structural
rules for reactions, transports, interactions, reaction-coupled transports,
and sub-pathways while keeping unknown additive buckets object-safe. Both live
post-pipeline handlers render contract failures separately with the exact
boundary, skipped stages, issue codes and JSON pointers, full report, and a
downloadable JSON report. A new run also clears stale successful artifacts
before executing.

**Pipeline consistency:** Genuine structural failures still abort before
mapping and audit, but valid process shapes are no longer rejected by another
process type's rule. The orchestrator exposes the contract state without
repairing or silently changing the payload.

---

### 2026-07-13 — Runtime payload reports and live Stage 2 mapping boundary

**Files changed:** `src/t2pw/pipeline/payload_models.py`,
`src/t2pw/pipeline/stage_contracts.py`, `src/t2pw/mapping/map_ids.py`,
`src/t2pw/app/streamlit_app.py`, `tests/test_payload_models.py`,
`tests/test_stage_contracts.py`, `tests/test_stage2_mapping_boundary.py`,
`tests/test_streamlit_stage2_orchestration.py`, `docs/pipeline.md`, and
`AGENT_INSTRUCTIONS.md`.

**Error / symptom:** The live Streamlit post-pipeline path normalized the
merged extraction/inference payload before any Stage 2 database mapping. Its
only mapping call was the post-curation Stage 6 remap, so the UI could not show
the exact mapped payload entering Stage 3 or distinguish early mapping misses
from later wrapper creation. TypedDict documentation also did not catch nested
runtime type errors in full payloads.

**Root cause:** The documented Stage 2 boundary had not been wired into the
orchestrator, and mapper compatibility behavior combined annotation, wrapper
creation, and structural cleanup. Boundary contracts covered selected
structural and semantic invariants but did not recursively validate known
container/value shapes at runtime.

**Fix:** Added non-mutating Pydantic runtime models with stable JSON-pointer
reports and report/enforce modes, integrated through the stage-contract
adapter. The live Stage 2B call now uses cache with wrapper creation and
structural cleanup explicitly disabled, requires object payload/report results,
and validates the nested `mapping_meta.resolution` shape before Stage 3. Stage
6 explicitly bypasses cache and enables wrappers/cleanup. The passes share one
database configuration but emit separate `stage2.mapped.json`,
`stage2_mapping_report.json`, `stage2_runtime_schema_report.json`,
`final.mapped.json`, and `mapping_report.json` UI artifacts. Runtime reports
are also exposed after enrichment and before export. Runtime validation remains
report-first by default and allows unknown additive metadata; it does not
replace the semantic PWML gate.

**Pipeline consistency:** Stage 2 owns annotation and mapping uncertainty,
Stage 3 receives that exact output and still owns normalization, Stage 4 keeps
its strict-gate-only repair cadence, Stage 6 remains the sole wrapper-creating
remap (including the PathBank `Unknown` fallback), and Stage 8 retains semantic
export authority. Malformed or failed Stage 2 results stop before Stage 3 and
cannot produce a successful mapped artifact.

---

### 2026-07-13 - Explicit PathBank Unknown fallback for unresolved enzymes

**Files changed:** `src/t2pw/mapping/map_ids.py`,
`src/t2pw/mapping/enrich_entities.py`, `src/t2pw/pipeline/entity_identity.py`,
`src/t2pw/pwml/ir.py`, `src/t2pw/pwml/writer.py`, `src/t2pw/schema.py`,
`tests/test_pathbank_unknown_fallback.py`, and `docs/pipeline.md`.

**Error / symptom:** A source-supported functional enzyme name could remain a
bare, unmapped protein after every protein identity strategy failed. PathWhiz
requires a protein-complex enzyme with a resolvable member, so export was
blocked even though PathBank provides a known `Unknown` protein sentinel.

**Root cause:** Stage 6 had no explicit, provenance-bearing route from a fully
unresolved enzyme to the known PathBank sentinel. Treating the sentinel's
`Unknown` UniProt text as a normal accession would also trigger an invalid
UniProt enrichment request.

**Fix:** After ordinary protein and complex mapping plus the API retry fail,
the wrapper-enabled Stage 6 pass may create or reuse one functional-name
complex backed by PathBank protein `9659` (`Unknown`, *Arabidopsis thaliana*,
species 4, taxon 3702). It records the target organism and
`cross_species_placeholder`, synchronizes catalyst-modifier mirrors, preserves
non-catalytic references, deduplicates reruns, and skips UniProt enrichment.
Stage 2 cannot activate it. PWML emits the reference-compatible
`protein-complex-protein` child and the sentinel's exact scalar identity.

**Pipeline consistency:** Real mappings always win. Mapping and wrapper
creation remain Stage 6 responsibilities; Stage 3 owns catalyst promotion and
contract checking; Stage 8 only serializes the explicit mapping.

---

### 2026-07-10 — Shared entity identity and enforceable Stage 2/3/6 contracts

**Files changed:** `src/t2pw/pipeline/entity_identity.py`,
`src/t2pw/mapping/map_ids.py`, `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/pipeline/stage_contracts.py`, `src/t2pw/schema.py`,
`tests/test_entity_identity_contracts.py`

**Error / symptom:** Mapping, normalization, and PWML code independently
decided whether an entity was protein-like, whether a protein had exportable
identity, and whether a complex was a generated wrapper. Stage 2 could also
create export wrappers even though Stage 6 owns that transformation, while
actor rows could leave Stage 3 without canonical `entity`/`entity_type` fields.

**Root cause:** Identity rules were copied across stage-specific modules, the
mapping API had no wrapper-creation control, and the Stage 3/6 exit guarantees
were documented but not asserted.

**Fix:** Added the dependency-light `entity_identity` module and switched
mapping and normalization to its shared routing, external-ID, species, and
generated-wrapper helpers. Added `allow_complex_wrapper_creation` to mapping,
canonicalized all supported process actor collections, typed `spontaneous` and
generated-wrapper fields in `schema.py`, extended the Stage 3 actor contract,
and added a Stage 6 generated-component identity contract.

**Pipeline consistency:** Entity identity is now neutral shared policy rather
than Stage 3 reaching backward into Stage 2. Stage 2 can map without structural
wrapper creation; Stage 6 alone may create wrappers and must validate them
before export.

---

### 2026-07-10 — Stage 8 fails invalid enzymes and serializes PathWhiz truthfully

**Files changed:** `src/t2pw/pwml/ir.py`, `src/t2pw/pwml/writer.py`,
`src/t2pw/pwml/qa.py`, `src/t2pw/pwml/to_pwml.py`,
`src/t2pw/pwml/legacy_validate.py`, `src/to_pwml.py`, `src/validate.py`,
`tests/test_pwml_ir.py`, `tests/test_pwml_writer.py`

**Error / symptom:** Stage 8 silently wrapped a bare protein enzyme, discarded
reaction spontaneity before serialization, assigned every protein the
pathway's first species, and could hide duplicate enzyme-complex assignments
that PathWhiz rejects. Dead standalone converter/validation paths also offered
an alternate exporter with incompatible behavior.

**Root cause:** Export attempted last-resort structural repair instead of
enforcing Stage 6 output, the IR reaction omitted `spontaneous`, protein
serialization ignored per-record species context, and duplicate targets were
silently collapsed before QA could report them.

**Fix:** Bare protein enzymes now fail the PWML contract without auto-wrapping;
the IR carries `spontaneous`; writer species IDs resolve from each protein with
the pathway species only as a true fallback; QA rejects spontaneous reactions
with enzymes and repeated enzyme-complex targets; and the confirmed-dead
converter/legacy validation modules and shims were removed. The CLI export
also writes and enforces its normalization gate before building IR.

**Pipeline consistency:** Stage 8 validates and serializes the Stage 6 payload
without inventing biology or concealing import errors. Obsolete alternate
export paths are removed so the IR-backed writer remains the authoritative
implementation.

---

### FIXED - Stage 8 PWML IR: direct protein enzyme wrapper lost source protein metadata

**Files changed:** `src/t2pw/pwml/ir.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
The PWML IR builder has a last-resort safety net that wraps direct protein enzyme actors
as generated single-protein protein_complex records. That wrapper check was running after
entity rows had been converted into IR records, but those records did not preserve `species`
context and `_protein_external_id` did not inspect nested `mapped_ids`. A valid protein
catalyst could therefore be reported as missing species/UniProt and remain as a bare protein
enzyme in the IR, triggering `reaction_enzyme_must_be_protein_complex`.

**Fix:**
IR entity records now preserve `species`, `taxonomy_id`, and `species_ref`, and the protein
external-ID helper reads UniProt/DrugBank IDs from `mapped_ids`, `ids`, and `mapping_meta`.
The existing direct-protein catalyst tests now verify that valid proteins are wrapped into
single-component protein_complex enzymes.

---

### FIXED - Stage 3 alias normalization: protein_complex component metadata was flattened

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
`canonicalize_same_as_aliases` rewrote `protein_complexes[].components` to plain strings.
That preserved the component name but discarded structured fields such as `stoichiometry`,
`mapped_ids`, `uniprot`, and `pathbank_protein_id`. Generated PathWhiz wrapper complexes
could then lose the exact data needed by Stage 3/Stage 8 contract checks.

**Fix:**
Component alias rewriting now preserves dict component rows and rewrites only the component
name-bearing field. The same pass also reads component names with `_component_name_from_row`
instead of stringifying dict components.

---

### FIXED - Stage 3 gate: generated protein_complex components missing stoichiometry

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
Generated single-protein PathWhiz wrapper complexes could pass Stage 3 with a component record
that resolved to a declared protein and had species/external identity, but omitted explicit
`stoichiometry`. Stage 8 and the SPMDB schema require structured protein_complex components
to carry positive stoichiometry, so the payload failed later during required PWML contract
validation.

**How the fix is consistent with the pipeline design:**
Stage 3 already hard-gates generated protein_complex component resolution, species, and external
identity. The fix adds the matching positive `stoichiometry` requirement to that same generated
component loop, so repair/audit sees the issue before export. A focused regression test covers
`NdmA complex` with a resolved `NdmA` component missing `stoichiometry`.

---

### FIXED — Stage 3 gate: `canonicalize_same_as_aliases` leaked protein_complex names into `entities.proteins`

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `canonicalize_same_as_aliases`

**Error / symptom:**
After Fix 1 (degree-0 exemption), the gate still reported 21 errors:
- `Generated protein complex wrapper 'NdmA complex' must be listed under protein_complexes, not proteins.`
- `Protein 'NdmA complex' is missing species/organism.`
- `Protein 'NdmA complex' is missing a UniProt or DrugBank identifier.`
Same pattern for `NdmB complex`, `NdmC complex`, `NdmCDE complex`, `NdmD complex`, `TmuM complex`, `caffeine dehydrogenase complex`.

**Why it appeared:**
`canonicalize_same_as_aliases` iterates over every reaction's `enzymes` and `modifiers` and calls
`_ensure_protein(actor_name, payload, rep)` for each actor. Stage 2/6 mapping had already rewritten
these reaction modifier references to point to generated complex wrappers (e.g. `NdmA complex`).
`_ensure_protein` checks whether `actor_name` is in `entities.proteins` but not whether it is in
`entities.protein_complexes`. So it unconditionally added `NdmA complex`, `NdmB complex`, etc. to
`entities.proteins`, even though they were already correctly placed in `entities.protein_complexes`.

**How the fix is consistent with the pipeline design:**
`_ensure_protein` is a safety net to guarantee that every reaction actor has a declared entity.
It should only fire for names that are not yet declared as *any* entity type. Since protein_complex
entries are real entity declarations, an actor that is already in `complexes` needs no fallback
protein row. The one-line guard `if _find_entity_row(complexes, actor_name) is not None: continue`
skips the `_ensure_protein` call for actors already declared as a complex. Stage 3 owns normalization;
this fix stays within `canonicalize_same_as_aliases` and touches no other stage.

---

### FIXED — Stage 3 gate: degree-0 check incorrectly flagged proteins that are complex components

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `run_strict_post_normalization_gates`

**Error / symptom:**
PWML export failed with "PWML export stopped by Stage 3 gate." The gate reported errors such as:
- `Protein has degree 0 after normalization: NdmA`
- `Protein has degree 0 after normalization: NdmB`
- `Protein has degree 0 after normalization: NdmC`
- `Protein has degree 0 after normalization: NdmD`

**Why it appeared:**
Stage 2/6 mapping wraps single-protein reaction enzymes in generated protein_complex records
(e.g. `NdmA complex`) and replaces the direct protein reference in the reaction modifier
with the complex name. This is correct — PathWhiz requires protein_complex as the enzyme
actor. The side effect is that `NdmA` is no longer referenced directly in any reaction;
its network connection flows through `NdmA complex`. `build_graph` does not add edges for
protein→complex component membership, so `NdmA` has degree 0 in the connectivity graph.
`prune_disconnected_proteins` (step 15) correctly keeps `NdmA` because it has a UniProt ID
(`_has_protein_identity` returns True). The `enforce_all_proteins_connected` check in step 17
then flagged it as a connectivity failure even though degree-0 is the expected and correct
state for a complex-component protein.

**How the fix is consistent with the pipeline design:**
`pipeline.md` states: "A protein survives all three passes if it has any of:
complex-component membership with external identity, a process reference, a non-zero
graph degree, or an external database ID." The gate check lacked this exemption.
The fix builds a set of protein name norms that appear as components in any declared
`protein_complexes[]` entry and skips those from the `enforce_all_proteins_connected`
error. No other gate check is changed. Stage 3 owns the gate; the fix lives entirely
within `run_strict_post_normalization_gates`.

---

## Fixed

---

### FIXED — Stage 3 gate: degree-0 check incorrectly flagged proteins that are complex components

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `run_strict_post_normalization_gates`

**Error / symptom:**
PWML export failed with "PWML export stopped by Stage 3 gate." Gate reported errors like `Protein has degree 0 after normalization: NdmA/B/C/D`.

**Why it appeared:**
Stage 2/6 mapping wraps single-protein reaction enzymes in generated `protein_complexes` entries (e.g. `NdmA complex`) and rewrites the reaction modifier reference to use the complex name — correct, because PathWhiz requires a protein_complex as the enzyme actor. The side effect is that `NdmA` no longer appears directly in any reaction; its network connection flows through `NdmA complex`. `build_graph` does not add edges for protein→complex component membership, so `NdmA` has degree 0. `prune_disconnected_proteins` (step 15) correctly kept it because `_has_protein_identity` returned True. The `enforce_all_proteins_connected` check in step 17 then flagged it as a connectivity failure even though degree-0 is the expected state for a complex-component protein.

**How the fix is consistent with the pipeline design:**
`pipeline.md` states: "A protein survives all three passes if it has any of: complex-component membership with external identity, a process reference, a non-zero graph degree, or an external database ID." The gate was missing this exemption. The fix builds `_complex_component_norms` from all `protein_complexes[].components` entries and skips those protein names from the `enforce_all_proteins_connected` error. No other gate check is changed. Stage 3 owns the gate; the fix lives entirely within `run_strict_post_normalization_gates`.

---

## Open Issues

Issues confirmed by running the pipeline. Ordered by pipeline stage. Each entry
records its current status, diagnosis, and planned fix; some older entries are
partially resolved and retain their remaining work here.

---

### IMPLEMENTED — LIVE RERUN VERIFIED: Stage 3/4a/pre-export `NdmCDE` repair

**Files involved:** `src/t2pw/pipeline/process_normalizer.py`,
`src/t2pw/curation/gap_resolver.py`, `src/t2pw/app/streamlit_app.py`, tests for
normalization, Stage 3 gap issues, orchestration convergence, and organism-aware
location selection.

**Implementation status (2026-07-13):** The stage-owned repair described below
is implemented with deterministic regression coverage. The fresh live run
completed mapping, normalization, audit/gap resolution, curation, and final
Stage 6 remapping while keeping `NdmCDE` out of the protein registry. Its saved
artifact now also passes the repaired programmatic PWML export; only the final
manual Streamlit export click remains.

**Observed progress:** The earlier failed run stopped safely at pre-export
Stage 3 revalidation and exposed two pointer-addressed errors for the synthetic
`/entities/proteins/5` row. The subsequent live run cleared that boundary and
produced a valid final remap. Its first PWML attempt exposed the independent
Stage 8 re-normalization bug documented above; replaying the same artifact after
the repair returns `ok=true` with no required-contract or IR-validation errors.

**Error / symptom:** `NdmCDE` is correctly declared under
`entities.protein_complexes`, but the final normalization pass adds another
bare `NdmCDE` row under `entities.proteins`. The gate rejects that new protein
because it lacks species/organism and UniProt/DrugBank identity. In the Stage 3
resolution report, the real `protein_complex:ndmcde` issue is detected with
missing component stoichiometry and unresolved component references, then its
execution is skipped with `reason="issue_not_found"`. Only one gap-resolution
round is recorded.

**Root cause:**

1. `normalize_composites` rewrites `element_locations.protein_locations` through
   `_rewrite_token`. `_ensure_protein` checks only the protein registry and does
   not first preserve a matching declared protein complex, so the location row
   causes a cross-bucket duplicate.
2. `run_gap_resolution` builds `entity_by_key` for proteins and compounds only,
   although `_collect_stage3_issues` also emits protein-complex issues. The
   planner can therefore request a complex repair that the executor cannot find.
3. The outer audit loop decides convergence from audit patch counts. It can stop
   when no audit patch was accepted even if Gap Resolve changed the payload or
   still has actionable issues. Fresh gate evaluation is also conditional on an
   accepted audit patch instead of any settled-payload change.
4. Location candidate ranking uses broad PathBank frequency without a strong
   organism-compatibility filter. It selected endoplasmic-reticulum membrane for
   two compounds in *Pseudomonas putida*.

**Planned fix:**

1. Stage 3 normalization will perform type-aware registry lookup, preserve
   declared complex references in location/process rows, and gate cross-bucket
   duplicate names. It will not perform ID lookup or invent component ratios.
2. Stage 4a will index protein-complex entities, join component names to declared
   mapped proteins, and write structured component references. Stoichiometry
   must come from source evidence or an accepted audit patch; an absent value
   remains an explicit issue.
3. Stage 4 orchestration will run the fresh strict gate after audit or gap-only
   changes and use payload progress plus remaining issues for convergence. Loop
   safety remains bounded by unchanged/repeated payload detection, timeout, and
   maximum rounds.
4. Stage 4a location resolution will use resolved organism/taxonomy compatibility
   before LLM selection and reject clearly impossible compartments.
5. Stage 6 remains a remapper of the settled payload. Stage 8 remains the hard
   export guard, with UI wording clarified to identify the pre-export Stage 3
   revalidation.
6. Regression tests will cover complex location references, complex issue
   execution, gap-only convergence, organism-compatible locations, and an
   end-to-end `NdmCDE` export boundary.

**Pipeline consistency:** Entity classification is a Stage 3 normalization
invariant; targeted DB/component/location repair belongs to Stage 4a; iteration
and convergence belong to the Stage 4 orchestrator; Stage 6 refreshes mappings;
Stage 8 validates exportability. No proposed stage silently takes over another
stage's semantic responsibility.

---

### PARTIALLY RESOLVED - Stage 2/6/8: Generated PathWhiz protein-complex wrappers leak into proteins and bypass Stage 3 blocking

**Files to change:** `src/t2pw/mapping/map_ids.py`, `src/t2pw/pipeline/process_normalizer.py`, `src/t2pw/app/streamlit_app.py`, `src/t2pw/pwml/ir.py`, tests covering mapping, normalization gates, and PWML export blocking.

**Current status (2026-07-13):** The live Stage 2 pass is now annotation-only,
Stage 6 is the sole wrapper-creating remap, generated wrappers carry explicit
provenance and component-integrity requirements, and unresolved pre-export
Stage 3 failures stop PWML generation. The remaining named-complex duplication
case is not a Stage 2 wrapper leak: it is caused by Stage 3 location-reference
normalization and is tracked in the open `NdmCDE` issue above.

**Error / symptom:**
PWML required-field validation reports errors such as:

- `Protein 'NdmA complex' is missing species/organism.`
- `Protein 'NdmA complex' is missing a UniProt or DrugBank identifier.`
- Same pattern for `NdmB complex`, `NdmC complex`, `xanthine oxidase complex`, `urate oxidase complex`, `allantoinase complex`, `urease complex`, `TmuM complex`, and `TM-HIU hydrolase complex`.

These errors are misleading because the generated `* complex` names should not
be protein rows at all. In PathWhiz, the member protein needs the UniProt or
DrugBank ID; a protein-complex record can be created from valid member proteins
and does not necessarily need a complex-level PathBank ID.

**PathWhiz behavior confirmed from UI:**

1. The `New Protein` form requires `Name`, `Species`, and either `UniProt ID`
   or `DrugBank ID`.
2. The `New Protein Complex` form requires `Name`, `Species`, and at least one
   member `Protein` with stoichiometry.
3. Therefore a generated single-protein wrapper such as `NdmA complex` is valid
   only as a `protein_complexes[]` row with component `NdmA`; `NdmA` must be a
   valid `proteins[]` row with species and UniProt/DrugBank identity.
4. The pipeline should not try to find or assign a UniProt ID for
   `NdmA complex`; UniProt belongs to `NdmA`.

**Root cause:**
There are two interacting issues:

1. `map_ids._rewrite_reaction_protein_enzymes_to_complexes` creates novel
   single-component wrappers named `f"{protein_name} complex"` when PathBank DB
   lookup cannot resolve a real complex. This is acceptable only if the wrapper
   stays under `entities.protein_complexes` and its member protein is already
   mapped.
2. The Streamlit PWML export path calls `normalize_process_payload`, receives a
   Stage 3 gate report, but then proceeds to `validate_required_pwml_contract`
   instead of stopping on unresolved Stage 3 gate failures. As a result, issues
   that Stage 3 can detect still reach the Stage 8 hard gate.

The mapping cache also contains stale/generated `enzyme_complexes` records for
the affected names, including entries with `status: "unmapped"` and
`chosen_rule: "novel_enzyme_single_component_complex"`. These cache rows show
where the `* complex` names are being synthesized.

**Planned fix:**

1. Stage 2/6 mapping (`map_ids.py`):
   - Keep generated single-protein wrappers under `entities.protein_complexes`
     only.
   - Mark generated wrappers with explicit metadata such as
     `generated: true` and
     `generation_reason: "single_protein_pathwhiz_wrapper"`.
   - Before creating a usable wrapper, require the base protein row to have
     species plus UniProt/DrugBank identity.
   - If the base protein is unmapped, do not create an apparently exportable
     complex. Record a mapping issue instead.
   - Do not add or preserve rows like `NdmA complex` under
     `entities.proteins`.
2. Stage 3 normalization/gate (`process_normalizer.py`):
   - Add a hard gate check that rejects `entities.proteins[]` rows whose names
     are generated-complex shaped (`* complex`) when they correspond to
     generated wrappers.
   - Add a generated-complex integrity check: species present, at least one
     component, and every component resolves to a declared protein with
     UniProt/DrugBank identity.
   - Preserve the current design: Stage 3 reports these as gate failures for
     audit/review; it should not silently reclassify biological entities unless
     the operation is a deterministic generated-wrapper cleanup.
3. Orchestrator (`streamlit_app.py`):
   - Before initializing refinement review and before PWML generation, inspect
     `normalize_process_payload(...)[1]["gate"]`.
   - If the gate is not OK, stop and surface the Stage 3 gate errors. Do not
     continue to the PWML required-field gate.
4. Stage 8 PWML IR (`pwml/ir.py`):
   - Treat generated protein complexes without a complex-level PathBank ID as
     valid only when their component proteins satisfy the protein identity
     contract.
   - Keep strict validation for ordinary protein rows: species plus
     UniProt/DrugBank remains required.
5. Tests:
   - Add a fixture where `NdmA` has species and UniProt and `NdmA complex` is a
     generated protein complex. This should pass the generated-complex contract.
   - Add a fixture where `NdmA complex` appears under `entities.proteins`. This
     should fail Stage 3 before export.
   - Add a fixture where `NdmA complex` is generated but component `NdmA` lacks
     UniProt/DrugBank. This should fail before export.

**Pipeline consistency:**
The protein-vs-complex distinction belongs at the mapping and normalization
boundary. Stage 2/6 owns creation of generated PathWhiz wrapper complexes.
Stage 3 owns deterministic gate checks that prevent invalid rows from reaching
review/export. Stage 8 owns final PWML contract enforcement. The orchestrator
must wire these stages so unresolved Stage 3 failures block PWML generation
rather than being rediscovered later as required-field errors.

---

### OPEN â€” Stage 2 (Map): Best-effort UniProt fallback assigns lowest-scored candidate when no threshold passes

**Files to change:** `src/t2pw/mapping/map_ids.py` (replace best_effort_fallback block), `src/t2pw/curation/audit_json_llm.py` (add audit hint for best_effort IDs)

**Error / symptom:**
Generic enzyme names such as "N-methyltransferase complex" and "N-methylnucleosidase complex" have no species-specific UniProt entry that clears the 0.78 confidence threshold. As a temporary workaround (added 2026-07-08), the mapper now accepts the top-ranked UniProt candidate regardless of score and marks it `best_effort: True`. This prevents Stage 3 gate failures for missing external identity, but the assigned accession may be incorrect â€” it is simply the highest-scoring candidate from a name search, not a verified match.

**Root cause:**
Generic descriptive names ("N-methyltransferase", "N-methylnucleosidase") return many UniProt hits with similar, low Jaccard scores. None is definitively the right protein, so no candidate clears the strict acceptance threshold. The right fix is sequence-based disambiguation (BLAST or UniProt sequence search) â€” find the actual protein sequence from the paper or a reference, BLAST it against UniProt, and accept the top hit by sequence identity. This requires the pipeline to carry or fetch protein sequences, which it currently does not do.

**Planned fix:**
1. For proteins that reach the best_effort_fallback path, attempt a NCBI eSearch + efetch to retrieve the candidate sequence by gene name + organism.
2. Submit the retrieved sequence to the UniProt BLAST API.
3. Accept the BLAST top hit (â‰¥40% identity, â‰¥60% coverage) as the confirmed accession and replace the best_effort ID.
4. Add an audit hint in `audit_json_llm.py` that flags any entity with `best_effort: True` in its mapping metadata so the audit LLM knows to verify or propose a correction.

**Pipeline consistency:**
Sequence fetching and BLAST belong in Stage 2 mapping or Stage 4a gap resolution â€” both own external ID lookup. No normalization or export logic would change. The `best_effort` flag in mapping metadata is the audit signal; the audit loop owns the decision to accept or replace the provisional ID.

---

### OPEN â€” Stage 4 (Audit): LLM connection failure prevents semantic repair

**Files to change:** Configuration / environment (not source code)

**Error / symptom:**
`curator_report.json` shows `"error": "chat_with_tools call failed after
retries. Last error: Connection error."` The round-1 audit report shows
`"enabled": false`. The 3 reactions with missing inputs (see next issue) survive
to the final payload unremediated because no LLM repair ran.

**Root cause:**
The OpenRouter API is not reachable from this environment. Either the API key
is missing/invalid, the endpoint is blocked, or there is a network configuration
issue. The audit stage correctly identifies there are errors to fix (3 reactions
with empty inputs) but cannot reach the LLM to generate repair patches.

**Planned fix:**
1. Verify `OPENROUTER_API_KEY` (or equivalent) is set in the environment.
2. Confirm the OpenRouter endpoint is reachable (`curl https://openrouter.ai/api/v1/models`).
3. If using a local model or alternative provider, update the provider config in
   `src/t2pw/curation/audit_json_llm.py`.
4. Add a clear error message in the Streamlit UI when LLM is disabled so the
   user knows semantic repair was skipped, not that the pipeline succeeded.

**Pipeline consistency:**
This is a configuration / infrastructure issue. No stage logic needs to change.
The pipeline correctly passes gate failures to audit; the repair just cannot run
without an LLM connection.

---

### OPEN â€” Stage 1 (Extract): Empty reaction inputs in beta-oxidation chain

**Files to change:** None yet â€” this is a Stage 4 (Audit) repair task once
LLM is connected (see connection issue above).

**Error / symptom:**
Reactions `beta_oxidation_OPC8`, `beta_oxidation_OPC6`, `beta_oxidation_OPC4`
(indices 10, 11, 13 in the final payload) each have `inputs: []` and
`outputs: ["jasmonic acid"]`. The audit deterministic check reports:
`"Reaction must include at least one input and one output."` for all three.

In the real pathway, each beta-oxidation cycle takes an OPC-CoA ester (OPC-8:0-CoA,
OPC-6:0-CoA, OPC-4:0-CoA) as input and produces the next shorter chain as
output. The LLM collapsed the entire beta-oxidation chain and wrote only the
final product (jasmonic acid) as each reaction's output.

**Root cause:**
Stage 1 (Extract) LLM did not capture the intermediate acyl-CoA compounds as
reaction-level inputs/outputs. Each reaction was extracted as "produces JA" with
the intermediate steps omitted. This is an expected limitation of the extraction
stage â€” the LLM summarised rather than enumerated each cycle.

**Planned fix:**
This is the correct input for Stage 4 (Audit). Once the LLM connection is
restored (see above), the audit should:
1. Receive the gate failure list which includes these 3 reactions.
2. Propose patches that add the correct OPC-CoA intermediate as `inputs` for
   each reaction and the shortened OPC-CoA as `outputs` (except the last cycle
   which produces jasmonic acid).
No normalization or schema change is required â€” this is a data completeness
issue that the audit loop is designed to repair.

**Pipeline consistency:**
The gate correctly fires for these reactions. The pipeline design says gate
failures feed audit, not abort. No stage logic change needed. This issue will
resolve once the LLM connection issue is fixed.

---

### OPEN â€” Stage 1 (Extract): OPC-8, OPC-6, OPC-4 misclassified as proteins

**Files to change:** None yet â€” audit repair task.

**Error / symptom:**
Entities `OPC-8`, `OPC-6`, and `OPC-4` appear in `entities.proteins`. The audit
warns: `"Protein has no location link; default compartment may be used."` for
all three. These are 3-oxo-2-(2'-pentenyl)-cyclopentane-acyl-CoA intermediates â€”
chemical compounds, not proteins.

**Root cause:**
Stage 1 (Extract) LLM placed these in `proteins` because they appear alongside
enzyme names in the paper text, and their abbreviated names (OPC-N) look like
protein/gene identifiers. The normalization stage does not reclassify entity
types.

**Planned fix:**
Once the LLM connection is restored, the audit should propose moving these
entities from `entities.proteins` to `entities.compounds`. An alternative
heuristic: Stage 3 normalization could flag entities whose names match known
patterns (CoA suffix, lipid chain nomenclature) as candidate compound
misclassifications and include them in the gate report for audit review. This
would not require a hard reclassification in normalization (which would be
cross-stage logic), just an additional audit hint.

**Pipeline consistency:**
Reclassification is a semantic operation and belongs in Stage 4 (Audit). If a
normalization hint is added, it goes into the gate report as an audit input â€”
not as a normalization mutation. This preserves the rule that Stage 3 does not
make semantic corrections.

---

### OPEN â€” Stage 2 (Map): DB unavailable â€” degraded mapping rates

**Files to change:** Configuration only.

**Error / symptom:**
`mapping_report.dbonly.json` shows `"db_available": false`. Protein mapping:
38.46% (5/13). Compound mapping: 9.52% (2/21). All 12 protein complexes skipped
(10/12 have gap issues â€” component proteins unmapped, so complex cannot map).

**Root cause:**
The PathBank database is not reachable from this environment
(`db_available: false`). Stage 2 falls back to API-only mapping, which has
lower coverage than the local DB. Compound mapping is especially degraded
because most common metabolites depend on the local compound DB.

**Planned fix:**
1. Configure the local PathBank DB (host, schema, credentials) in the Streamlit
   sidebar or environment variables.
2. If DB is intentionally absent, document that compound ID coverage will be
   low and PWML export will operate in non-strict mode.
3. The protein complex gap issues (10/12) are downstream of the protein mapping
   problem: once component proteins map, the complexes should map too. Fix the
   DB connection first and re-run before filing separate complex issues.

**Pipeline consistency:**
This is a configuration / infrastructure issue. Stage 2 mapping logic is correct.
No source changes needed. The pipeline.md note on Stage 2 already says mapping
returns no-hit when an entity is unmapped, which is recorded in `mapping_meta`.

---

### OPEN â€” Stage 6 (Enrich): Enrichment stage produces data no stage consumes

**Files to change:** Decision required before code change.

**Error / symptom:**
`run_enrichment` fetches synonyms, cross-references, and properties and writes
`entity["enrichment"]` onto each entity. No downstream stage (normalization,
audit, PWML IR, SBML) reads this field. The enrichment API call and cache write
happen on every pipeline run with no effect on output.

**Root cause:**
The enrichment stage was built but not wired into the PWML IR builder or any
other consumer. This was documented as a product decision pending in the
refactoring plan (Step 8).

**Planned fix â€” choose one:**
- **Option A (Use it):** Wire `entity["enrichment"]` into the PWML IR builder
  (`src/t2pw/pwml/ir.py`) so synonyms and cross-references appear in the
  exported pathway file. This adds value to the PWML output.
- **Option B (Remove it):** Delete `run_enrichment` and its call site in
  `streamlit_app.py`. Mapping already attaches database IDs; enrichment adds
  nothing until Option A is implemented. Removal simplifies the orchestrator and
  eliminates dead code.
Until the decision is made, the enrichment stage runs silently on every pipeline
execution consuming API quota and cache space.

**Pipeline consistency:**
Option A change lives in `t2pw/pwml/ir.py`. Option B removes
`t2pw/mapping/enrich_entities.py` call site from the orchestrator. Neither
option adds cross-stage logic.

---

## Template

```
### YYYY-MM-DD â€” <short description>

**Files changed:** `path/to/file.py` (lines Xâ€“Y)

**Error / symptom:**
What the user or test saw. Quote the error message if there is one.

**Root cause:**
Why the error appeared. Name the specific stage boundary violation, field
mismatch, or misplaced logic that caused it.

**Fix:**
What was changed and where.

**Pipeline consistency:**
Which stage owns this change. Confirm it does not add cross-stage logic and
does not expand any module's scope beyond its intended area (see File ownership
table in pipeline.md).
```

---

## Entries

### 2026-07-08 â€” Best-effort UniProt fallback for generic enzyme names that clear no confidence threshold

**Files changed:** `src/t2pw/mapping/map_ids.py` (lines ~3580â€“3590, `map_protein_uniprot`), `docs/change_log.md`

**Error / symptom:**
Proteins with generic names such as "N-methyltransferase complex" and "N-methylnucleosidase complex" failed the Stage 3 gate checks for missing UniProt/DrugBank identifiers. These names have no species-specific UniProt entry that scores above the 0.78 acceptance threshold â€” the correct protein cannot be confidently distinguished from many similarly-named candidates by name alone.

**Root cause:**
`map_protein_uniprot` returned `status: "unmapped", reason: "ambiguous"` whenever candidates existed but none cleared the strict threshold. The caller only writes a UniProt accession when `status == "mapped"`, so ambiguous results produced no ID on the protein entity. For genuinely generic enzyme names, no name-based scoring strategy can reliably pick the right candidate â€” the correct fix is sequence-based lookup (BLAST), but the pipeline does not currently carry protein sequences.

**Fix:**
In `map_protein_uniprot`, when `_accepted_uniprot_candidate_result` returns None but at least one candidate has a non-empty accession, return that top candidate with `status: "mapped"`, `chosen_rule: "best_effort_fallback"`, and `best_effort: True` instead of `status: "unmapped"`. The caller writes the accession, which clears the Stage 3 gate check. The `best_effort: True` flag is preserved in the mapping metadata as a signal for the audit loop that this ID was not confidently matched and should be reviewed.

**Pipeline consistency:**
Change is entirely within `map_protein_uniprot` in `t2pw.mapping.map_ids`, which owns Stage 2 and Stage 6 ID mapping. No normalization, audit, or export logic was changed. The fallback is a last resort â€” it only activates when all three normal acceptance paths (strong_unique, reviewed_unique, reviewed_exact_gene_match) fail and candidates exist. The proper long-term fix (BLAST-based sequence lookup) is documented as an open issue. See OPEN issue: "Stage 2 (Map): Best-effort UniProt fallback assigns lowest-scored candidate when no threshold passes."

---

### 2026-07-08 â€” Strengthen extraction scoping to single-pathway + single-organism; fix doc inconsistencies

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
Three issues found after adding the initial species scoping rule: (1) The BIOLOGICAL STATE RULE still instructed the LLM to default to *Homo sapiens* when no organism was available, directly contradicting the new scoping rule's instruction to leave species empty. (2) The scoping rule told the LLM to pick one organism but did not tell it to first pick one pathway â€” papers covering multiple pathways (e.g. caffeine biosynthesis and caffeine degradation in the same review) would still produce a merged multi-pathway extraction scoped to one organism. (3) `docs/pipeline.md` had a duplicate copy of the Step 17 gate description mislabelled as Step 15, and the file ownership table listed no prompt files.

**Root cause:**
(1) The BIOLOGICAL STATE RULE predated the scoping rule and was never updated to match. An LLM reading both rules encounters conflicting instructions for the no-organism case. (2) The prior scoping rule said "choose one primary biological scope" but did not make pathway selection an explicit first decision â€” organism selection was the only named decision. (3) The pipeline.md duplicate was a copy-paste artifact from a prior edit; the prompt files were always owned but never listed in the table.

**Fix:**
1. Changed the BIOLOGICAL STATE RULE fallback from `"use 'Homo sapiens' as the default"` to `"leave species empty â€” do not guess or default to any organism"`, removing the contradiction.
2. Expanded the scoping rule into two explicit sequential decisions: Decision 1 (select one pathway â€” the most central to the paper) followed by Decision 2 (select one organism for that pathway). The pathway decision now comes first and is the primary filter; organism selection applies within it.
3. Removed the duplicate Step 15 paragraph from `docs/pipeline.md` (lines 135â€“138, copy-paste of Step 17 description).
4. Added `pwml_system.txt` and `pwml_infer_system.txt` to the file ownership table in `docs/pipeline.md`.

**Pipeline consistency:**
All changes are in prompt text files and documentation. No Python source was modified. The scoping decision remains Stage 1's responsibility â€” it is an extraction-time filter that prevents mixed-pathway, mixed-species entity sets from entering Stage 2 and beyond.

---

### 2026-07-08 â€” Add single-organism scoping rules to Stage 1 extraction prompt

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `src/t2pw/llm/prompts/pwml_infer_system.txt` (cross-reference note only), `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
Proteins from multiple organisms present in a single paper (e.g. *Coffea arabica* biosynthesis enzymes and *Pseudomonas putida* degradation enzymes) were extracted together into the same pathway payload, resulting in mixed species assignments across entities. This caused Stage 3 gate failures for missing species/organism on proteins that inherited no clear organism context, and UniProt mapping failures at Stage 2 and Stage 6 because the wrong species was searched for each protein.

**Root cause:**
The Stage 1 extraction prompt (`pwml_system.txt`) had no rule requiring the LLM to select a single primary organism before extracting reactions. Papers that cover multiple organisms â€” comparative studies, combined biosynthesis-plus-degradation reviews â€” caused the LLM to emit proteins from all mentioned organisms, mixing species context across entities. The BIOLOGICAL STATE RULE required species on every biological_state but gave no guidance for choosing among competing organisms.

**Fix:**
Added two rule blocks to `pwml_system.txt` immediately after the BIOLOGICAL STATE RULE:

1. **Species and organism scoping rule** â€” instructs the LLM to select one primary organism/species/strain before extracting reactions, assign it to all proteins, enzymes, complexes, reactions, and biological states, exclude entities from other organisms unless explicitly requested, and emit an audit warning rather than mix species when no organism can be confidently selected.
2. **Protein/enzyme species rule** â€” requires every protein, enzyme, and protein complex to inherit the selected pathway species before identifier mapping, and prohibits emitting a protein entity without a species/organism assignment and sufficient identifier context.

Added a species constraint cross-reference note to the locality constraint block in `pwml_infer_system.txt`: the Stage 2 mandatory modifier repair pass is now explicitly instructed to apply the Stage 1 species scoping rule and skip modifier links for proteins from other organisms.

**Pipeline consistency:**
Change is entirely within prompt text files. No Python source was modified. Species scoping is an extraction-time decision that Stage 1 owns â€” the correct stage boundary. Selecting a single organism at Stage 1 prevents mixed-species entity sets from propagating to Stage 2 mapping (where wrong-species queries fail silently) and Stage 3 gate checks (where missing species generates unrepaired gate failures). Stage 2â€“8 behavior is otherwise unchanged.

---

### 2026-07-08 â€” Strip "complex" from UniProt name query variants

**Files changed:** `src/t2pw/mapping/map_ids.py` (line ~50, `_name_variants`), `docs/change_log.md`

**Error / symptom:**
Proteins with "complex" in their names â€” e.g. "xanthine oxidase complex", "NdmA complex", "IMP dehydrogenase complex", "TmuM complex" â€” consistently failed the Stage 3 gate checks added on 2026-07-08 for missing UniProt/DrugBank identifiers. These proteins are findable in UniProt under their base names ("Xanthine oxidase", "NdmA", etc.) but the pipeline assigned no accession to any of them.

**Root cause:**
`_name_variants` (Stage 2 and 6 mapping) already strips "protein" and "enzyme" from name query strings to normalize them for UniProt lookup, but did not strip "complex". UniProt never includes "complex" in individual protein entry names â€” that word is a complex-level descriptor. Querying for "xanthine oxidase complex" produced a Jaccard similarity of 2/3 â‰ˆ 0.667 against the correct UniProt entry "Xanthine oxidase". After scoring (`base_score = 0.35 Ã— 0.667 = 0.234`, plus organism and reviewed bonuses), the total landed at â‰ˆ 0.53 â€” 0.25 points below the 0.78 acceptance threshold â€” so no accession was accepted despite the correct entry being returned by UniProt's API.

**Fix:**
Added `"complex"` to the word-strip regex in `_name_variants`:
```
re.sub(r"\b(protein|enzyme|complex)\b", " ", base, flags=re.IGNORECASE)
```
Names like "xanthine oxidase complex" now generate "xanthine oxidase" as a search variant. That variant scores 1.0 (exact name match, base_score = 0.55) plus organism and reviewed bonuses, clearing the 0.78 threshold and producing a mapped accession.

**Pipeline consistency:**
Change is entirely within `t2pw.mapping.map_ids`, which owns Stage 2 and Stage 6 ID mapping. No normalization, audit, export, or orchestrator logic was touched. The change is a query normalization improvement consistent with the pre-existing "protein" and "enzyme" stripping.

---

### 2026-07-08 â€” Add protein species and external identity checks to Stage 3 gate

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (inside `run_strict_post_normalization_gates`), `docs/change_log.md`

**Error / symptom:**
Stage 8 (Export) hard-aborted with `validate_required_pwml_contract` failures for two checks â€” `protein_missing_species` and `protein_missing_external_identity` â€” with no opportunity for the audit loop to repair the affected proteins. The specific failure: proteins (and compounds misclassified into `entities.proteins`) that had no species/organism field and no UniProt or DrugBank ID would pass Stage 3 and Stage 4 unchanged, then cause an unrecoverable abort at pre-export contract validation.

**Root cause:**
Both checks existed only in `t2pw/pwml/ir.py` (lines 1880â€“1915) as part of the hard Stage 8 pre-export semantic contract. They were absent from `run_strict_post_normalization_gates` in `process_normalizer.py`, so Stage 4 (Audit) never received them as gate failures to repair. The audit loop correctly repairs what the gate reports; the gate simply never reported these two conditions.

**Fix:**
Added two new loops inside `run_strict_post_normalization_gates`, immediately after the existing forbidden-name check loop on `entities.proteins`. Each loop iterates `entities.proteins`, skips unnamed rows (already caught by a separate check), and calls `_add_error` when the condition is unmet:

1. **Species/organism check** â€” mirrors the `species` resolution chain from `ir.py`: tries `species`, `organism`, `taxonomy_id`, `species_id`, `pathbank_species_id`, `species_ref.pathbank_species_id`, `species_ref.name`, `mapping_meta.species`, `mapping_meta.species_id`.
2. **External identity check** â€” emits an error if none of `uniprot`, `uniprot_id`, `drugbank`, `drugbank_id` are present and non-empty.

Both checks use only `_safe_dict` and `_safe_list`, which are already defined in `process_normalizer.py`. No imports from `t2pw.pwml.ir` or any other stage module were added.

**Pipeline consistency:**
The fix lives entirely within `run_strict_post_normalization_gates` in `process_normalizer.py`, which owns Stage 3's gate. The gate's return type and `errors` list shape (`{"path": str, "reason": str}`) are unchanged. The `GateValidationError` raise path is untouched. By surfacing these two conditions as Stage 3 gate failures, Stage 4 now receives them in its repair context and can propose patches (species assignment, ID lookup via gap resolution) before Stage 8 runs. No cross-stage logic was introduced â€” `process_normalizer.py` mirrors the field-level logic without importing from `ir.py`.

---

### 2026-07-07 â€” Fix `normalize_process_actor_schema` to write `entity`/`entity_type` for enzyme actors

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (blocks 1c and legacy-enzyme view), `tests/test_process_normalizer.py` (updated assertions)

**Error / symptom:**
All enzyme actor dicts in `reactions[].enzymes` retained `protein_complex` (or
`protein`) as the name field after normalization completed. `e.get("entity")`
returned `""` for every enzyme. After a full pipeline run on the Arabidopsis
jasmonic acid pathway, all 30 enzyme actors used `protein_complex`, while all 30
modifier actors correctly used `entity/entity_type`.

**Root cause:**
`normalize_process_actor_schema` has two passes. Pass 1 (`_rewrite_actor_rows`)
resolves each actor name against the protein and complex registries and writes
the canonical name back to `protein_complex` or `protein` â€” NOT to `entity`.
The post-process loop migrated `modifiers[]` rows to `entity/entity_type` schema,
but had no equivalent migration for `enzymes[]`. Additionally, the "legacy view"
reconstruction block that rebuilds `reaction["enzymes"]` from `modifiers[]`
wrote `protein`/`protein_complex` keys rather than `entity`/`entity_type`,
leaving enzymes in legacy field format after the schema-normalization step.

**Fix:**
1. Added block **1c** in the post-process loop (after the modifier migration and
   the 1b entity_type correction): iterates `reaction["enzymes"]`, migrates each
   dict from `protein_complex`/`protein`/`name` to `entity`/`entity_type`,
   drops actors whose `entity_type` is in `dropped_enzyme_entity_types`, and
   writes the result back to `reaction["enzymes"]`.
2. Updated the legacy-enzyme view reconstruction (formerly writing
   `protein_complex`/`protein` keys) to use `entity`/`entity_type` instead,
   keeping `reaction["enzymes"]` in sync with `modifiers[]` in the canonical
   schema.
3. Updated six test assertions in `tests/test_process_normalizer.py` that were
   checking for the old `protein`/`protein_complex` keys; all now verify
   `entity` and `entity_type` and confirm legacy keys are absent.

**Pipeline consistency:**
Change is entirely within `normalize_process_actor_schema` in
`process_normalizer.py`, which owns actor schema enforcement for Stage 3. No
orchestrator, mapping, audit, or export logic was touched. After the fix, any
code that calls `actor.get("entity")` works correctly for both enzyme and
modifier actors without special-casing field names.

---

### 2026-07-07 â€” Wire drop_process_orphan_proteins into normalize_process_payload

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (line ~3586), `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
`drop_process_orphan_proteins` was defined and documented but never called inside
`normalize_process_payload`. Standalone subunit proteins (e.g. NdmC, NdmD) that
appear only as `protein_complex.components` entries and are never referenced in any
reaction, transport, or interaction would pass through all normalization steps and
reach the gate as orphans, generating audit issues that should have been pre-empted
by pruning.

**Root cause:**
A prior implementation pass added the function to the module but omitted the call
site from the pipeline sequence in `normalize_process_payload`. The change log
stated it was wired in, but the code did not reflect this. The gap was discovered
by reviewing the actual step sequence (lines 3584â€“3588) against the documented
17-step list.

**Fix:**
Added the call `drop_process_orphan_proteins(data, report=report)` and its
corresponding `_checkpoint("drop_process_orphan_proteins")` between
`drop_unresolved_complex_component_proteins` and `prune_disconnected_proteins`.
Updated `docs/pipeline.md` to reflect the 17-step sequence and document why steps
13â€“15 run in sequence (each catches a different class of orphan; a protein must
fail all three to be treated as an orphan by the gate).

**Pipeline consistency:**
Change is entirely within `normalize_process_payload` in `process_normalizer.py`,
which owns all normalization steps. No orchestrator, mapping, audit, or export
logic was touched. The three pruning steps remain independent functions â€” each with
a single responsibility â€” rather than being merged into one function that would be
harder to reason about when a specific class of orphan slips through.

---

### 2026-07-07 - Deterministic PWML compound IDs with optional DB resolver

**Files changed:** `src/t2pw/pwml/ir.py`, `docs/change_log.md`

**Error / symptom:**
PWML IR tests could pass or fail depending on which modules pytest collected in
the same process. A payload with explicit `pathbank_compound_id` values produced
`compound_db_resolution_failed` errors when a DB resolver was importable.

**Root cause:**
`_resolve_compound_rows` only accepted direct PathWhiz compound IDs as a fallback
when no DB resolver was available. If resolver construction succeeded, the same
rows were sent through live DB matching and could fail despite already carrying
the required export ID.

**Fix:**
Accepted explicit `pathbank_compound_id` / `pw_compound_id` / `pathwhiz_id`
values before attempting resolver lookup, while still recording the
`legacy_id_unverified` DB-resolution status in the IR report.

**Pipeline consistency:**
This stays inside `t2pw.pwml.ir`, which owns pre-export IR construction. It does
not move mapping logic into export; it only makes already-mapped payload IDs
deterministic regardless of optional DB resolver availability.

---

### 2026-07-07 - Streamlit uses canonical normalization and post-audit cache bypass

**Files changed:** `src/t2pw/app/streamlit_app.py`, `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
The Streamlit post-pipeline path still owned a hand-built normalization sequence
and returned immediately on post-normalization gate failures, preventing the
audit loop from repairing the semantic issues documented by the gate.

**Root cause:**
Normalization logic lived partly in the orchestrator, including evidence-based
enzyme attachment and explicit gate handling. Post-audit mapping also reused the
normal mapping cache even though audit patches can rename entities.

**Fix:**
Replaced the manual normalization block with `normalize_process_payload` and an
`on_checkpoint` callback that writes the existing probe files. Gate failures are
now written to `gate_fail_report.json`, passed into the audit context, and
reported as audit input rather than a stopped pipeline. The post-audit mapping
pass now calls `map_payload` in memory with `use_cache=False` and writes the
same mapped payload/report artifacts.

**Pipeline consistency:**
The orchestrator now wires stage functions and artifacts only. Normalization
behavior remains in `process_normalizer.py`; mapping/cache behavior remains in
`map_ids.py`; the UI no longer owns enzyme-attachment logic or a parallel
normalization pipeline.

---

### 2026-07-07 - Normalizer actor lookup, evidence enzymes, pruning, and gate reporting

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `src/t2pw/pipeline/qa_graph.py`, `tests/test_process_normalizer.py`, `docs/change_log.md`

**Error / symptom:**
Normalizer actor rows could still resolve a stale legacy `protein` or
`protein_complex` field before the canonical `entity` field. Enzyme mentions in
reaction evidence were wired in Streamlit as plain strings, disconnected
proteins were pruned without respecting mapped identity, and
`normalize_process_payload` could abort on gate failure before the audit loop
received the gate details.

**Root cause:**
Normalizer-owned actor interpretation and evidence wiring were split across the
orchestrator and normalization stage. Some compatibility reads still used
legacy field order. Protein pruning and gate handling also mixed pre-audit
cleanup with semantic rejection, which conflicts with the audit-loop contract in
`docs/pipeline.md`.

**Fix:**
Moved `_norm_text` and `attach_enzymes_from_reaction_evidence` into
`process_normalizer.py`, with cue-near-name matching and canonical actor dict
output. Updated normalizer and QA graph actor lookup to read `entity` before
legacy fields. Changed `prune_disconnected_proteins` to remove only degree-0
proteins with no external identity and record report details. Wired
`normalize_process_payload` to run the enzyme-evidence step, support checkpoint
callbacks, and return gate details in `report["gate"]` instead of aborting.

**Pipeline consistency:**
All deterministic cleanup remains in `process_normalizer.py`, and graph
connectivity interpretation remains in `qa_graph.py`. Streamlit does not gain
new normalization logic. Gate failures remain semantic audit input after
normalization, preserving the documented normalize-to-audit loop rather than
turning the normalizer into a pre-audit hard abort.

### 2026-07-07 - Add stage boundary contract validators

**Files changed:** `src/t2pw/pipeline/stage_contracts.py` (lines 1-273), `tests/test_stage_contracts.py` (lines 1-104), `docs/change_log.md`

**Error / symptom:**
Step 6 of the pipeline refactor needed a dedicated `stage_contracts` module so
stage boundary checks are explicit and testable. Without it, callers had no
single place to distinguish structural aborts from semantic gate failures that
must be sent to audit.

**Root cause:**
Boundary contract ownership was documented in `docs/pipeline.md`, but no module
implemented those boundaries. That made it easy to collapse pre-audit semantic
gate failures into hard abort behavior, which would bypass the audit loop that
is supposed to repair them.

**Fix:**
Added `StageContractError` plus validators for post-extraction, post-mapping,
post-normalization, post-audit, and pre-export boundaries. Structural
validators raise `StageContractError`; post-normalization returns semantic gate
failures as a report for audit; pre-export wraps
`validate_required_pwml_contract` failures in `StageContractError`. Added
focused unit tests for missing required boundary fields and PWML contract
wrapping.

**Pipeline consistency:**
The change lives entirely in `t2pw.pipeline.stage_contracts`, the module named
as the owner of stage boundary validation. It does not add normalization,
mapping, audit, UI, or PWML IR logic, and it keeps pre-audit semantic failures
as audit input instead of making them aborts.

---

### 2026-07-07 - Document pipeline payload schema types

**Files changed:** `src/t2pw/schema.py`, `docs/change_log.md`

**Error / symptom:**
Step 1 of the refactor needed `t2pw/schema.py` to document the JSON payload
contracts, but the module was empty.

**Root cause:**
The pipeline stages pass dictionaries whose expected shapes are documented in
the extraction prompts and `docs/pipeline.md`, while the schema ownership module
did not yet expose those contracts for type checkers or importers.

**Fix:**
Added `TypedDict` definitions for the payload, entity buckets, biological
states, locations, process rows, visualizations, mapping metadata, and inference
additions. `PayloadReactionActor` documents `entity` as the canonical actor name
while retaining backwards-compatible `protein` and `protein_complex` fields.

**Pipeline consistency:**
This change is type/documentation only and lives entirely in the schema module
that owns payload shapes. It does not add validation, normalization, UI logic,
or cross-stage behavior, so runtime pipeline behavior is unchanged.

---

### 2026-07-07 â€” Gate validation errors not shown in UI

**Files changed:** `src/t2pw/app/streamlit_app.py` (lines 2637â€“2651)

**Error / symptom:**
PWML export failed with "Hard-gate validation failed after normalization" but
the UI showed no detail about which specific checks failed, making the error
unactionable.

**Root cause:**
The `st.error()` call at the gate failure block only displayed the top-level
error string from `gate_fail_report`. The `errors` list inside the report
(which contains per-check path and reason) was never rendered.

**Fix:**
Expanded the gate failure display block to iterate `gate_fail_report["errors"]`
and show each entry as a formatted line (path + reason) inside an expander.

**Pipeline consistency:**
Change is entirely within the orchestrator's display logic. No stage function
was modified. The gate report structure is owned by `process_normalizer.py` and
was not changed â€” only the UI reading of it was corrected. This is a pure
orchestrator responsibility: surface what a stage reported.

---

### 2026-07-07 â€” Orphan proteins not pruned when not complex components

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (after line 1639)

**Error / symptom:**
Proteins appeared in `entities.proteins` with no reference in any reaction,
transport, or interaction, and no external database identity. These caused
the gate's `enforce_all_proteins_connected` check to fail.

**Root cause:**
`drop_unresolved_complex_component_proteins` (the existing pruning step) only
dropped proteins that appeared as components of a declared `protein_complex`
entity. Proteins that the LLM extracted standalone, with no complex membership
and no process reference, were never caught.

**Fix:**
Added `drop_process_orphan_proteins` to `process_normalizer.py`. It collects
all entity names referenced across reactions, transports, and interactions,
then drops any protein not in that set that also has no external identity
(`_has_protein_identity` returns False). Wired into `normalize_process_payload`
between the existing complex-component pruning step and `dedupe_processes`.

**Pipeline consistency:**
Change lives entirely within `process_normalizer.py`, which owns all
normalization steps. The new function follows the existing pattern: takes
payload + optional report dict, mutates payload in-place on the deep copy,
records dropped items in the report. No orchestrator or UI code was changed.
No new stage or module was created.

---

### 2026-07-07 - In-memory wrappers for mapping and audit stages

**Files changed:** `src/t2pw/mapping/map_ids.py`, `src/t2pw/curation/audit_json_llm.py`, `src/t2pw/curation/apply_audit_patch.py`, `tests/test_map_ids.py`, `tests/test_audit_json_llm_payload.py`, `tests/test_apply_audit_patch_lock_policy.py`

**Error / symptom:**
Later orchestration work needed to pass Step 7 payload objects between mapping, audit, and patch application without forcing every stage through temporary JSON files. The existing file wrappers also made post-audit remapping vulnerable to stale cache reads unless callers could bypass or invalidate cache entries.

**Root cause:**
The core mapping implementation lived inside the file-based `run_mapping` adapter, so object-level orchestration could not reuse it directly. Audit and patch application had similar file-wrapper boundaries even though their core logic was already mostly payload-based. Mapping cache control was implicit in the cache file rather than exposed at the stage boundary.

**Fix:**
Added `map_payload` as the object-in/object-out mapping entry point and changed `run_mapping` to call it before writing the same mapped JSON/report files. Added `use_cache` and `invalidate_cache_keys` support to the mapping cache path for post-audit remapping correctness. Added `audit_payload` and `apply_audit_patch_payload` wrappers that reuse existing audit and patch core logic without duplicating manifest discovery. Added focused tests for the new wrapper contracts.

**Pipeline consistency:**
Mapping cache and ID assignment changes stay in `t2pw.mapping.map_ids`, which owns Stage 2 and post-audit remapping. Audit planning stays in `t2pw.curation.audit_json_llm`, and patch policy stays in `t2pw.curation.apply_audit_patch`. Existing `run_*` functions remain file adapters, and no Streamlit or normalization logic was moved into these stages.

### 2026-07-10 — Live stage contracts and fresh audit gate cadence

**Files changed:** `src/t2pw/app/streamlit_app.py`

**Error / symptom:** Boundary validators existed only in tests, and audit rounds
continued from a stale pre-audit gate report after accepting a repair patch.

**Root cause:** The Streamlit orchestrator used ad hoc report inspection instead
of the stage-contract API and never reran the cheap strict gate inside the audit
loop. Its only live `map_payload` call is the post-curation remap, so there was
also no honest Stage 2 mapping boundary to validate.

**Fix:** Wired the live extraction, normalization, audit, remap, and pre-export
boundaries to their contract functions; marked the existing remap call as the
wrapper-creating Stage 6 pass; and reran only
`run_strict_post_normalization_gates` after every selected patch with accepted
operations. Fresh pointer-level failures are saved in the iteration and passed
to the next audit prompt. No synthetic Stage 2 call was introduced.

**Pipeline consistency:** Boundary coordination stays in the Streamlit
orchestrator. Normalization still runs once before audit and once at export;
the loop invokes only the Stage 3-owned strict gate.

---

### 2026-07-10 — Audit reuses Stage 3 validators and resolves enzyme-less reactions

**Files changed:** `src/t2pw/curation/audit_json_llm.py`,
`tests/test_audit_json_llm_payload.py`

**Error / symptom:** Audit maintained separate composite/registry failure
definitions and could not explicitly resolve an enzyme-less reaction as
spontaneous.

**Root cause:** Stage 4 had grown its own deterministic checks rather than
consuming Stage 3's validators, and its patch policy had no allowed
`spontaneous` operation.

**Fix:** Audit now calls `validate_no_composites` and
`validate_registry_references` for the shared failure definitions while keeping
patch construction in Stage 4. When both enzyme rows and catalyst modifiers
lack a real actor reference, deterministic audit emits a documented
`spontaneous=true` patch.

**Pipeline consistency:** Stage 3 remains the owner of normalized composite and
registry validity. Stage 4 owns repair planning and is one of the two stages
permitted to write `spontaneous`.

---

### 2026-07-10 — Extraction and audit spontaneity instructions

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`,
`src/t2pw/curation/audit_json_llm.py`

**Error / symptom:** The model had no explicit distinction between
source-supported spontaneous extraction and audit-time resolution of a missing
enzyme.

**Root cause:** Neither prompt stated which stages may set `spontaneous` or the
enzyme-present contradiction.

**Fix:** The extraction prompt permits `spontaneous=true` only from explicit
source text. The audit prompt permits it only after checking both enzymes and
catalyst modifiers and finding no real catalyst; both forbid the flag when an
enzyme exists.

**Pipeline consistency:** The prompts mirror field ownership: Stage 1 records
explicit evidence and Stage 4 may resolve the absence of a real enzyme.

---

### 2026-07-10 — Concrete per-stage payload contracts

**Files changed:** `docs/pipeline.md`

**Error / symptom:** The eight stages described behavior but not exact input and
output shapes, allowing field ownership and wrapper creation to drift.

**Root cause:** Documentation named broad stage responsibilities without tying
them to `schema.py` types, boundary validators, or failure effects.

**Fix:** Added an eight-stage contract table naming the concrete TypedDict
inputs/outputs, exit guarantees, validator ownership, audit cadence, and the
Stage 2/Stage 6 wrapper-creation distinction. It also records that the current
Streamlit path has no separate live Stage 2 `map_payload` call.

**Pipeline consistency:** This documents the existing stage architecture and
its enforceable boundaries without assigning implementation logic to docs or
inventing an orchestration pass.

---

### 2026-07-10 — Remove the legacy non-IR PWML writer fallback

**Files changed:** `src/t2pw/pwml/writer.py`, `src/pwml_writer.py`,
`tests/test_pwml_writer.py`

**Error / symptom:** The writer still contained a second raw-payload export
implementation alongside the IR-backed pipeline. That branch used legacy
defaults and process builders, so invoking a different entrypoint could produce
PWML with behavior inconsistent with the validated Stage 8 path.

**Root cause:** `_populate_sections` and nine associated entity, process, and
layout builders predated the mapped-JSON → IR → PWML architecture but remained
reachable through `load_extraction`, `run_writer`, a raw argparse surface, and
the top-level `pwml_writer.py` shim.

**Fix:** Removed the 1,701-line raw/non-IR builder branch, its loading and CLI
entrypoints, and the obsolete top-level shim. The module entrypoint now exposes
only `run_pwml_pipeline_export`; writer tests were converted to exercise the IR
path or removed where they covered only the deleted fallback.

**Pipeline consistency:** There is now one authoritative Stage 8 serialization
route. Every command-line export passes through normalization, IR construction,
contract validation, the deterministic IR writer, and PWML QA.
