# SPIKE-002 — Compound-resolution extraction scoping

**Status: COMPLETE, independently reviewed.** 2026-08-05.
Investigation only — no branch, no code, no file modified.

> **Amended 2026-08-05**, after the investigation closed, by a documentation-only closeout
> correction (finding D-4): three of this report's own line citations were imprecise and
> have been corrected in §1, §5 and §7. The corrections are listed at the end of §9. That
> amendment modified no source file and no other document. The "Working tree" and "G11"
> rows below describe the investigation as it ran, not this amendment.

| | |
|---|---|
| Base | `sprint/pwml-recovery` @ `2b786aa4af1ba14ac8a27f0a749eab8affae7a6b` |
| Blocks | C-040, C-050, C-051 only |
| Verdict | **`LIFT_WITH_ADAPTER`** |
| Review | **APPROVE_WITH_CORRECTIONS** (independent, read-only reviewer) |
| Working tree | unchanged; `git status --porcelain` identical before and after, both agents |
| G11 | 9 bounded jobs across investigation + review, **0 surviving owned processes** |

---

## 1. Verdict — `LIFT_WITH_ADAPTER`

`_resolve_compound_rows` and `_canonicalize_compound_offline` are **not** coupled to
`build_pwml_ir`'s internal row shapes. Their input requires no reshaping. Both are
module-level and parameter-closed; every row field they read is payload-level
(`name`, `raw_name`, `mapped_ids`, `mapping_meta`, flat id fields) reached through
`_first_nonempty` (`ir.py:74-92`) or `PathWhizCompoundResolver.resolve`
(`db_resolver.py:278-423`). `row["key"]` — the one field that exists only after
`_dedupe_named_rows` — is never read, only carried through `dict(row)` copies.

Measured, not inferred: `_resolve_compound_rows` runs correctly on raw
`final_mapped.json` compound rows and on a bare `{"name": ..., "chebi_id": ...}` dict.
`_canonicalize_compound_offline` mutates in place, returns `None`, is deterministic
across repeat calls, and does not mutate the caller's rows (two prior copies at
`ir.py:531` and `db_resolver.py:427`). `db_resolver.py` imports nothing from `ir.py`,
so extraction creates no cycle.

**The reviewer independently confirmed the verdict and does not support
`REQUIRES_RESHAPE`.** The sprint's pre-Wave-A0 sequence was therefore not halted.

## 2. The adapter is a three-part contract

The reviewer corrected the investigation here: the adapter has **three** parts, not two.
All three must be written into C-040's prompt.

1. **Report shape.** The `report` argument hard-indexes `_new_report()`'s nested shape.
   A caller passing `{}` raises `KeyError: 'compounds'`. Reachable raise sites are
   `ir.py:844` (legacy-id row) and **`ir.py:870`** (non-legacy row); `:884` is equally
   unguarded but unreachable because `:870` raises first. The module must seed
   **three** nested containers itself, not two.
2. **Rename propagation.** `apply_compound_db_resolution` (`db_resolver.py:442`) and
   `_canonicalize_compound_offline` (`ir.py:604`) rewrite `row["name"]`, while
   `processes.reactions[].inputs` in the canonical payload are plain name **strings**
   (`["glycine", "succinyl-CoA"]`). `build_pwml_ir` absorbs this only because
   `entity_by_name` indexes both `name` and `raw_name` (`ir.py:1138-1146`). **No
   pre-freeze consumer does.** Propagation must cover
   `processes.reactions[].inputs/outputs`, `transports`, `interactions` and
   `enzymes[].entity`.
3. **Row-set and idempotency contract.** Pre-freeze the module sees the un-pruned,
   un-deduped payload rows.

## 3. Functions to move — verified line ranges

| Function | Lines | Size | Note |
|---|---|---|---|
| `_normalize_compound_external_ids` | `ir.py:530-555` | 26 | called first at `:806` |
| `_compound_external_ids` | `ir.py:558-575` | 18 | also used by `_emit_canonicalization_preflight` at `:920`, which **stays** → must be re-imported by `ir.py` |
| `_canonicalize_compound_offline` | `ir.py:578-621` | 44 | |
| `_resolve_compound_rows` | `ir.py:797-897` | 101 | |

**189 lines moved.** All five line ranges in the SPIKE prompt verified exact, including
`build_pwml_ir` at `ir.py:966-2007` = 1042 lines. Stay in `db_resolver.py`, already
stage-neutral: `PathWhizCompoundResolver` (`:172-423`), `apply_compound_db_resolution`
(`:426-452`), `normalize_chebi_id` (`:19-23`).

**New module boundary** — `src/t2pw/pwml/compound_resolution.py`: the four functions
verbatim, a ~6-line hardening of the report contract, and private copies of the leaf
helpers (`ir.py:43-96, 183-193, 244-260`). Duplication is deliberate — `ir.py` must
import the new module, so importing back is a cycle. Pin the duplication with an
equality test. Stays in `ir.py`: `_emit_canonicalization_preflight` (`:900-963`, reads
IR structures not payload), `_canonicalize_species_offline` (`:710-794`, see R2),
`_entity_record` (`:437-449`).

**Recommended, C-040-owned:** a keyword-only `apply_canonical_name: bool = True` on both
moved entry points. The default preserves today's behaviour exactly and it lets C-050
land identifier attachment with zero rename risk.

## 4. C-040 size — `[S4]` breach, must be pre-split

Investigation estimated ≈600 changed lines. The reviewer's bottom-up count is **≈700**
(189 function bodies + ~70 leaf helpers — **9**, not 8, and `_add_issue` is not a leaf,
it mutates `report["ok"]` — + ~30 header/imports + 60–120 adapter + 189 deletion + ~20
shim + 150–250 tests).

`_SHARED_BLOCKS.md:64` stops an implementer at ~400 changed lines. **C-040 must be
pre-split by the Lead before dispatch** — module-extraction and adapter/propagation as
separate branches — or the implementer is obligated to stop and propose a split on
arrival. The conclusion is robust across both the 600 and 700 figures.

> **Superseded prospectively by D-019 (LOCKED).** The paragraph above records the sizing rule
> as it stood when this report was authored; its `_SHARED_BLOCKS.md:64` pointer is historical,
> not current. D-019 removed the universal `~400` threshold and the unconditional obligation
> to pre-split C-040 that followed from it. Sizing and splitting are now governed by declared
> card-specific budgets and D-019's independently-implementable-AND-validatable split rule.

## 5. Risks

**R1 — rename propagation, HIGH, currently unscoped in C-050.** Confirmed real and, per
the reviewer, **understated**. `_canonicalize_compound_offline` writes `aliases`
(`ir.py:610-612`) and **never** `synonyms`, while `_entity_name_norms`
(`process_normalizer.py:626-636`), which `strict_quarantine.py:79-84` imports
character-for-character, keys on `name` + `synonyms` only — and `_find_entity_row`
(`:639-646`) and `_remove_entity` (`:649-659`) key on `name` **alone**. `resolve_entity`
(`ir.py:1400-1433`) emits `unresolved_entity_reference` at **error** severity on a miss.
A pre-freeze rename that writes only `aliases` makes quarantine prune the renamed
compound and the referencing reaction breaks. **This is a silent reaction-dropper and a
G7 concern.** On the measured case the alias is not even written — `_norm("glycine") ==
_norm("Glycine")`, so the `:609` branch is skipped and only `raw_name` carries the
pre-rename name, which no pre-freeze consumer indexes.

**Hard acceptance criterion for C-050:** reaction count in `final_mapped.json` before
and after pre-freeze resolution must be identical, proven on the `PMC12856317` and
`PMC12452463` legs; and the extraction name must be written to `synonyms`.

**R2 — species canonicalization is the same violation and is owned by nobody, MEDIUM.**
`_canonicalize_species_offline` (`ir.py:710-794`, called `:1040-1045`) rewrites species
`name` post-freeze at `:753-757` and `:781-785`. `PRODUCT_CONTRACT.md` §5 lists organism
context as a must-remain-equivalent dimension. No row in `MASTER_PLAN.md` §9 names this
function. Not yet demonstrated firing (`name_canonicalization.species` is `[]` in all
three committed `pwml_ir_report.json`), so the gap is structural. Left as-is, T-102's
comparator will fail on species and the failure will be misattributed to C-050.

**R3 — reachability-dependent export, MEDIUM.** Resolved by C-050 by construction.

**R4 — `_dedupe_named_rows` runs before resolution today** (`ir.py:1100-1105`, then
`:1107`). Moving resolution pre-freeze is strictly better, not worse. Note only.

**Quarantine row-set delta, measured.** `_prune_entities` (`strict_quarantine.py:1250`)
drops 1 of 7 compounds on `runs_verify/2026-08-04_1207/papers/PMC12452463/strict`
(`ferric enterobactin`), so a pre-freeze caller resolves a strict superset. **The
difference is desirable** — quarantine and the Stage-3 gates then judge resolved
identities rather than extraction names. No `_norm` collisions exist among compound rows
in any committed `final_mapped.json`, so the dedupe half of the delta is theoretical
today.

## 6. The measured violation — corrected magnitude

**The sprint's own committed evidence overcounts by ~10×, and this is confirmed.**
`probe_exporter_identity_mutation.py:78-79` keys both sides on the raw `name`. The
exporter renames `glycine → Glycine`, so `canonical_rows.get("Glycine")` returns `{}`
and every field on that compound is reported as added.

Re-derived independently by the reviewer, pairing on `ir.raw_name` → canonical `name`,
on `runs_verify/2026-08-04_1647/papers/PMC12856317/strict`:

| | Committed evidence says | Actually measured |
|---|---|---|
| `mapped_ids` keys added post-freeze | 10 | **1** — heme `pubchem '3334'`, itself re-projected by `ir.py:539-551` from the canonical row's own top-level `pubchem_cid: '3334'` |
| Glycine's "nine external identifiers absent from the canonical payload" | 9 absent | **0 absent** — all nine are present verbatim in the canonical Glycine row's `mapped_ids`; only the `chebi` *value* differs, by prefix. The enumerated list also holds eight items while the prose says nine |
| `CHEBI:` prefix stripping | "three others" | **4 of 4**, in both top-level `chebi_id` and `mapped_ids.chebi` |

**The §5 violation is still real and still blocking.** It is violated by *kind*, not by
count. After `streamlit_app.py:3507-3508` froze and hashed the payload, the exporter
renames an entity with no provenance record, fabricates a `db_row` no upstream stage
produced, rewrites identifier values on 4/4 compounds, materializes
`pathwhiz_id`/`db_id`/`db_status`/`chosen_rule` on 4/4, and adds one `mapped_ids` key.
`PRODUCT_CONTRACT.md:149-153` and `:213-214` are both violated, and `ir.py:810-821` is
literally a resolution step with a live-DB branch running downstream of the freeze.
Classification: **`product_contract_violation`** — the one class that justifies code.

**F-1, high, new.** The rename in the committed 1647 run is **entirely unrecorded**.
`pwml_ir_report.json` contains `"name_canonicalization": {"species": []}` — no
`compounds` key at all — because `ir.py:609` gates the report entry on
`_norm(extraction) != _norm(canonical)` and `glycine`/`Glycine` normalize identically. A
biological entity was renamed after the canonical freeze with zero provenance: a §3
traceability failure on top of the §5 failure.

**F-2, high, prerequisite.** `probe_exporter_identity_mutation.py` **can never reach its
T-102 acceptance target** while it pairs on raw `name` — the metric is a function of
naming, not of mutation. It must be corrected to pair on `ir.raw_name or ir.name`
**before** it is used as an acceptance gate.

**F-3, medium.** Duplicating `_norm` adds a **fourth** name normalizer to a codebase that
already documents divergence between three (`strict_quarantine.py:73-78`). Require a
shared normalizer or an equality test pinning `compound_resolution._norm` to `ir._norm`.

**F-4, medium.** Moving resolution pre-freeze writes `db_row`, `db_status`,
`chosen_rule`, `db_match`, `pathwhiz_id`, `db_id` into `final_mapped.json`, changing
`canonical_payload_sha256` and possibly `canonical_graph_sha256`. C-040's dependency row
lists SPIKE-002 only; the C-013/C-030 projection interaction must be stated.

## 7. DB reachable — **YES**

`PathBankDbResolver.from_env()` (`map_ids.py:820-873`) constructed; `available() → True`;
`_ensure_connection() → True`; a live query returned in 0.31 s:
`{'id': 78, 'name': 'Glycine', 'hmdb_id': 'HMDB0000123', 'kegg_id': 'C00037',
'chebi_id': '15428', 'drugbank_id': 'DB00145', 'cas': '56-40-6'}`.

**What reachability changed.** On the 1647 leg, nothing — all four compounds carry
`pathbank_compound_id` and hit the legacy short-circuit at `ir.py:837-855` before the
resolver is consulted; their identifiers came from the **offline index**. Every compound
row in every committed `pwml_ir_report.json` records
`"chosen_rule": "legacy_pathwhiz_id_unverified"`, so the resolver path is unexercised in
the committed evidence and those rows are DB-independent. On the 1207 leg it **did**
change the outcome: `exact_short_name_or_synonym` and `ambiguous` verdicts require rows
returned from `compounds`; with the DB down those rows would instead be
`unmatched / db_resolver_unavailable`.

**The exported PWML is therefore a function of network state at export time**, which
independently falsifies §5's reload-and-re-export equivalence requirement. Moving
resolution pre-freeze fixes it by construction.

**Trap for C-050:** `available()` is `self._driver is not None` (`map_ids.py:875-876`) —
it tests whether `pymysql` imports, **not** whether the database answers. `ir.py:819`
treats it as a connectivity check, and `_emit_canonicalization_preflight` suppresses its
collision-risk warning whenever `db_available` is truthy (`:931`). On a host with
`pymysql` installed and the database unreachable, the preflight goes silent precisely
when it is most needed. Confirmed in the artifacts: `"available": true` with
`"preflight": null` on legs whose compounds were canonicalized by the offline index alone.

## 8. Schedule impact

**The Day-5 RC benchmark holds, conditionally.**

- **C-040 does not move** — mechanical extraction — **but must be pre-split** (§4).
- **C-050 must be re-scoped before dispatch.** Its register row reads
  `streamlit_app.py :: enrichment block above the seam`, which reads as "call the
  extracted function there." It is not. Dispatched as written it lands a silent
  reaction-dropper, T-103/T-104 regress, and Day 6 is consumed diagnosing it.
- **Mitigation preserving Day 5:** land C-050 in two commits behind
  `apply_canonical_name`. Commit 1 attaches identifiers pre-freeze with the rename off —
  eliminating 4 of the 5 measured violation classes at zero reference risk. Commit 2
  turns the rename on with reference propagation and its own regression test. If commit
  2 slips, commit 1 still ships.
- **C-051 becomes trivial** once C-050 lands: delete `ir.py:1106-1114`, replace with an
  assertion that every compound row already carries a resolution verdict, fail closed.
- **R2 needs an owner today**, or it surfaces during T-102 on Day 4 and is misattributed.

**Natural seam.** `build_pwml_ir` splits at identity/registry `:977-1263` (the
C-040/C-051 slice, containing the compound-resolution call at `:1106-1114`) ·
reference resolution and biological states `:1265-1505` · geometry, processes and
serialization prep `:1507-2007`. Cleanest single seams: `:1054` and `:1152`.

## 9. Prompt citation corrections

| Prompt citation | Actual | Status |
|---|---|---|
| `_resolve_compound_rows (:797-897)` | exact | ✅ |
| `_canonicalize_compound_offline (:578-621)` | exact | ✅ |
| caller at `ir.py:1107` | exact | ✅ |
| `build_pwml_ir` `:966-2007`, 1042 lines | exact | ✅ |
| "`_dedupe_named_rows` / lookup at `:1030`" | `:1030` is the **component** loop; the **compound** path dedupes at `:1100-1105` and discards the lookup (`rows, _ =`). The `lookup` bound at `:1030` is never read anywhere in the file — a dead local | ⚠ mis-citation. Corrects the premise; **the answer to Q2 is unchanged and is "no"** |
| `MASTER_PLAN.md:38-45` / probe docstring `:11-16` magnitude | see §6 | ❌ factually wrong, ~10× inflated |

**Corrections to this report's own citations, 2026-08-05 (source: independent closeout
review, finding D-4).** Three ranges in this report were imprecise and have been corrected
in place above. Each corrected range was re-read against the source at
`1c2dbee` before being written; the `def` line and the last line of the function were
confirmed in every case.

| Section | Previously written | Corrected to | What was wrong |
|---|---|---|---|
| §1 | `PathWhizCompoundResolver.resolve` `db_resolver.py:279-305` | `db_resolver.py:278-423` | start off by one (`def` is `:278`); end 118 lines short (function ends `:423`) |
| §5 / R1 | `resolve_entity` `ir.py:1400-1416` | `ir.py:1400-1433` | `:1400-1416` covers only the miss/error path; the function ends at `:1433` |
| §7 | `PathBankDbResolver.from_env()` `map_ids.py:819-873` | `map_ids.py:820-873` | off by one — `:819` is the `@classmethod` decorator, `def` is `:820` |

All three are citation-precision corrections: no verdict, risk, finding or measurement in
this report was restated or revised. The closeout review verified every other range in
this report as exact; none of those was touched.

## 10. Items requiring the product owner

None of these was resolved by any agent. `DECISIONS.md` § Open is untouched — O-1
remains the only open question.

1. **Is compound name canonicalization biology, and where does it belong?** Moving it
   pre-freeze changes `final_mapped.json`'s entity `name` and every process reference
   string. Not moving it leaves a §5 violation. Not doing it at all changes PathWhiz
   import behaviour — `ir.py:900-908` documents non-canonical names colliding on import
   as "the root bug". All three are product positions. **This is the decision
   C-040/C-050/C-051 actually rest on, and it is not in `DECISIONS.md`.**
2. **The species gap (R2) needs a branch or an explicit deferral.** Either add a C-04x,
   or rule that T-102's "diff must be EMPTY" is scoped to compounds only.
   > **Resolved by D-016 (LOCKED): the compound-only alternative is superseded and is no
   > longer available.** T-102 must verify **both** compound identity **and**
   > organism/species equivalence across canonical JSON, PWML **and** SBML. The species
   > gap is owned by **C-045** (`MASTER_PLAN` § 9), planning-only and not yet dispatchable.
3. **`MASTER_PLAN.md:38-45` and `probe_exporter_identity_mutation.py:11-16` are
   factually wrong and must be amended.** `MASTER_PLAN` is sprint authority; the sprint
   must not carry a 10×-inflated number into its own planning. The §5 violation is
   unaffected and remains blocking.

## 11. Evidence and process

Nine bounded jobs through `evidence/bounded_run.py` (five investigation, four review),
Windows Job Object with `KILL_ON_JOB_CLOSE` on every one, **final surviving owned
process count 0 across all nine**, cleanup success on all nine. Four pre-existing
`python.exe` processes were reported on every job and never killed. No pytest was
invoked — this is a read-only scoping spike.

`git status --porcelain` was captured before and after by both agents and was identical
in both cases; HEAD never moved from `2b786aa`. No repository file was created,
modified, staged, committed or restored. No D-014 protected file was opened for writing.
No `cache_snapshot/` directory was touched.

**One disclosed `[S8]` deviation.** The reviewer ran a single sub-second read-only
`python -c` one-liner (pretty-printing two committed `pwml_ir_report.json` files)
outside the wrapper, disclosed it rather than omitting it, and the finding it produced
was independently corroborated by a wrapped job. No orphan attributable to it; the two
`python.exe` processes observed afterwards both pre-date every job root PID.
