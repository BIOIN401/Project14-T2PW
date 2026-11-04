# ORCH-804: an empty state reference must not land in an arbitrary compartment (RESULT)

> **Status:** v2, the lead-requested refinement, is in § 8 and is the version up for review. §§ 3-5 record v1.

Branch `orch804/ir-empty-state`, cut from `sprint/pwml-recovery` @ `563d39db`. Implementer only; the lead
reviews and merges. Source: ORCH-801 audit finding F7 (`runs_papers/ORCH-801/ACCURACY_AUDIT.md` § 5.2),
hazard first recorded in `ORCH-788-RESULT.md` line 36.

## 1. Defect

`src/t2pw/pwml/ir.py`, `build_pwml_ir`, inner `resolve_state`: an empty `biological_state` on a reaction,
transport, interaction or element-location row returned `default_bs_key`, the key of the pathway's
**first** state, whatever that state was. In the rice salicylic acid leg (ORCH-801, PMC12408352, strict),
the PAL reaction has no state. Its IR (`pwml_ir.json`) has states `bs_1` "peroxisomal state" ...
`bs_7` `__auto_state__`, so PAL, its compounds and OsPAL06 were exported in the peroxisome. The paper gives
PAL no location. That breaks PRODUCT_CONTRACT § 1 (no invented location) and merge rule 8 (the exporter
repairs biology). The `required=True` argument was ignored: both branches returned the same key.

## 2. Where the neutral state comes from (upstream, read-only)

`process_normalizer.ensure_autostates` (called at `process_normalizer.py` ~3402 and ~5659) creates
`__auto_state__` with subcellular location `cell` (and the default species). It assigns that state to
stateless `compound_locations`/`protein_locations` rows and to transport endpoints. **It does not assign
reactions or interactions.** Later pruning removes any state that nothing references. So `__auto_state__`
survives only when some element-location row or transport uses it. It is **not** guaranteed to exist when
a reaction is stateless. See the census in § 4.3: 20 of the 30 replayed legs with an empty reaction or
interaction state have no `__auto_state__`.

## 3. Fix (code commit)

`ir.py`, 3 lines changed in `build_pwml_ir`:

```python
empty_bs_key = (state_by_name.get(_norm("__auto_state__")) or {}).get("key") or default_bs_key
def resolve_state(...):
    if not name:
        return empty_bs_key
```

- An empty reference goes to the existing `__auto_state__` (`cell`), which is the pipeline's own neutral
  placeholder.
- **No neutral state:** the old first-state fallback is kept unchanged. Justification: the PO's hard rule is
  that no leg that exports a PWML today may stop. The alternatives are (a) raise an IR error, which drops
  PWMLs, or (b) synthesise a `cell` state in the exporter, which is the exporter changing biology after the
  freeze (rule 8) and can add new `biological_state_missing_*` errors. Neither is admissible here. No new
  warning was added either: a warning moves every report digest for no behavioural gain. Options are in § 7.
- The no-op `required` branch was collapsed. `required` still drives the severity for an **unresolved
  named** reference (unchanged). An unresolved named reference still falls back to the first state. That is
  out of scope; see § 7 Q3.
- Transport visualizations still use `default_bs_key` (layout only, not a state assignment; untouched).
- **Duplicate same-name states** (audit: two Stage-1 passes; rice IR has `bs_1`/`bs_4` both "peroxisomal
  state"): no interaction with the fix. The empty branch looks up only `__auto_state__`. `state_by_name` keeps
  the last duplicate, as before. If `__auto_state__` itself were duplicated, the last copy would be used,
  which is harmless. Deduplication stays upstream and out of scope.

Commits on the branch:

- `6af0f704`: the code fix and the new test.
- `588ed5d4`: R3 re-pin in `tests/test_pwml_ir_duplicate_row_refusal.py`.
- `6fa08d27`: GOLDEN re-pin in `tests/test_compound_resolution_extraction.py`.
- A docs and evidence commit.

The two re-pins sit in their own commits under merge rule 4; see § 5.

## 4. Proofs (every job: bounded wrapper, heavy lock `ORCH-804`, `FINAL SURVIVING COUNT : 0`)

### 4.1 G9 focused test: `tests/test_orch804_ir_empty_state.py`

| test | kind | base `563d39db` | tip |
|---|---|---|---|
| `test_empty_reference_does_not_take_the_first_paper_state` (audit fixture: [peroxisomal state, `__auto_state__`/cell], PAL with no state) | **correction** | **FAILS**: `assert 'peroxisomal state' == '__auto_state__'` | passes |
| `test_named_reference_still_lands_in_its_state` | safety twin | passes | passes |
| `test_no_neutral_state_keeps_first_state_so_the_pwml_still_exports` | pins the chosen fallback | passes | passes |

Base `1 failed, 2 passed`: `g11/ORCH-804/02-base-focused.json`. The base tree was built from
`git archive 563d39db` at `C:/t/orch804/base`. Tip `3 passed`: `g11/ORCH-804/01-tip-focused.json`.

### 4.2 Corpus replay (no-loss)

| run | result | artifact |
|---|---|---|
| tip replay | 79/91 | `g11/ORCH-804/03`; `evidence/orch804/corpus_tip.{json,txt}` |
| base replay (`563d39db`) | 79/91 | `g11/ORCH-804/04`; `evidence/orch804/corpus_base.{json,txt}` |
| ORCH-799 record vs base | 79/91 -> 79/91, **91/91 identical** (the base arm reproduces the committed reference exactly) | `g11/ORCH-804/08`; `orch804/compare_799_vs_base.txt` |
| base vs tip | 79/91 -> 79/91; `passing legs turned failing: none`; `gold recall lost: none`; `still failing, at a different step: none`; 14 legs' records differ, 77/91 identical. "D-112 § 2 REJECT" only because D-112 requires a gain | `g11/ORCH-804/06`; `orch804/compare_base_vs_tip.txt` |
| ORCH-799 record vs tip | same 14 legs, same verdict lines | `g11/ORCH-804/05` |

### 4.3 Changed legs and their classification

`evidence/orch804/state_diff.py` (G11 `07`, output `orch804/state_diff_base_vs_tip.txt`) compares the base
and tip replay IRs per leg. It covers every process and process-visualization state key, and every
(entity, state) location as a set, because locations are positional and gain rows when a member moves.

On all 14 changed legs the **only** record keys that move are the PWML/IR files and `pwml_sha256` (plus
`pwml_matches_leg` on two). The canonical, quarantine, coverage, gate and release fields are **unchanged**.
**Every** moved process goes to `__auto_state__ [cell]`. **No entity is left without a location.**

| leg | rows moved -> `__auto_state__` (from) | class |
|---|---|---|
| ORCH-759/chlorophyll-strict | 5 interactions (plastid) | empty ref, arbitrary compartment -> neutral = correct |
| ORCH-759/wnt_pcp-strict | 11 interactions, 3 reactions, 1 transporter (plasma membrane) | correct |
| ORCH-759p/sour_taste_v2-strict | 3 transporter rows, OTOP1 (Type III cell state) | correct |
| ORCH-767/wnt_ca-strict | 8 interactions, 7 transporter rows (cytoplasm) | correct |
| runs_verify/08-18_1328/PMC12096016 | 1 reaction, 2 transporter rows (cytoplasm) | correct |
| runs_verify/08-18_1328/PMC12452463 | 2 reactions, 3 transporter rows (E. coli cytosol) | correct (single-compartment bacterium, see Q2) |
| runs_verify/08-21_2239/PMC12782028 | CYP51A1, DHCR24 reactions (ER) | correct |
| runs_verify/08-22_2147/PMC12452463 | FepA transporter row (extracellular) | correct |
| runs_verify/08-22_2147/PMC13231680 | 4 interactions (E. coli cytoplasm); IR only, this leg writes no PWML | correct |
| runs_verify/08-24_1428/PMC12444477 | MsbA transporter row (cytoplasm) | correct |
| runs_verify/08-27_1341/PMC12096016 | LDH reaction, 1 interaction (cytoplasm) | correct |
| runs_verify/08-28_1816/PMC12782028 | CYP51A1, DHCR24 reactions (ER) | correct |
| runs_verify/09-01_1612/PMC12096016 | TolC transporter row (cytoplasm) | correct |
| runs_verify/09-06_1425/PMC8510960 | G10H, catharanthine-vindoline coupling reactions (epidermal cell cytosol) | correct |

Transporter rows count because a transporter's own state goes through `resolve_state` too: an empty one
used to take the first state and now takes the neutral one. No row moved to anything other than
`__auto_state__`.

Exact per-row lines: `orch804/state_diff_base_vs_tip.txt`.

**Fallback census** (`orch804/fallback_census.py`, G11 `10`, output `orch804/fallback_census_tip.txt`):
30 replayed legs have an empty reaction/interaction state. 10 have `__auto_state__` and are fixed. **20 do
not**, and they keep the first-state behaviour. 17 of those have more than one state, so the hazard
persists there. See Q1.

### 4.4 SMOKE and affected tests

- SMOKE (22 files): **`511 passed in 62.42s`**, `g11/ORCH-804/11-smoke.json`.
- IR neighbours before the re-pin: `1 failed, 180 passed`.
  `test_new_acceptance_a_non_colliding_leg_is_byte_identical_to_the_base` failed
  (`g11/ORCH-804/12-ir-neighbours.json`). See § 5.
- After the re-pin, the same files plus `test_orch788_state_without_location.py`,
  `test_c121_autostate_lifecycle.py` and the new file: **`199 passed`** (`g11/ORCH-804/16`).
- SMOKE ran on the code commit. The later commits change only test constants and docs, so the code SMOKE exercised has not changed.
- Other digest-pinning IR/PWML test files: see § 4.5.

## 5. Pinned baseline moved (merge rule 4): R3 control leg

`tests/test_pwml_ir_duplicate_row_refusal.py` pins the IR and report sha256 of `build_pwml_ir` over
`runs/2026-07-27_1623/.../PMC12312563.../strict/final_mapped.json`. Reactions 5 and 6 of that leg have an
**empty** state. Measured by `orch804/r3_ir_delta.py`, which replays the test's own call sequence on each
tree (G11 `13` base, `14` tip, `15` diff; full pointer list in `orch804/r3_ir_delta.txt`):

- IR `14af62c13c45…` (base, equal to the old pin) -> `de16dd279c3a…`. Reactions 5 and 6 move from `bs_1`
  "Listeria monocytogenes cellular state" to `bs_2` `__auto_state__`. Their member locations,
  process/complex visualizations and four edge paths follow. Six new location rows appear in `bs_2`, so
  location keys renumber. The explicit element locations in `bs_1` are all still present.
- Report `082729a8b749…` -> `9cae4bc39039…`. The whole delta is `counts.locations` and
  `ir_validation.counts.locations` 15 -> 21. `errors`, `warnings` and `ok` are unchanged; the test's own
  assertions on them still pass.

**GOLDEN (`tests/test_compound_resolution_extraction.py`), 35 of 89 legs moved.** I re-pinned them with
pytest's own values (old -> new for each leg: `orch804/golden_repin_pairs.txt`). The delta was measured by
`orch804/orch804_ir_digest_delta.py`, which reuses `orch747_ir_digest_delta.py` (G11 `22`, output
`orch804/orch804_ir_digest_delta.{json,txt}`). Every pointer that moved is a reaction, interaction or
transporter `biological_state_key`, or something that follows from it: locations, edges, process and
complex visualizations, and location counts. `errors / ok / stops unchanged on every leg: True`, exit 0.
The tool prints the verdict line "delta confined to the interaction export". That wording is inherited
from the ORCH-747 tool. The check behind it uses ORCH-804's owned prefixes.

Caveat: the instrument measures one configuration, the DB-down config used by R3. GOLDEN hashes five.
ORCH-754b re-pinned GOLDEN the same way. The per-row direction of the moves (always to `__auto_state__`)
is shown on the corpus in § 4.3. The instrument checks only which paths moved, not the values they moved
to.

All constants were re-pinned, with the delta written next to them. They sit in their own commit, so the
lead can drop the commit and refuse the move. **Flag for the reviewer:** on this leg, the members of
reaction 6 also carry **explicit** element locations in `bs_1`. At tip those members are drawn twice: in
`bs_2` for the reaction and in `bs_1` from the explicit rows. Nothing is invented (`cell` is neutral), but
this is a visible change. See Q2.

### 4.5 Other digest-pinning test files

I checked the test files that pin IR/PWML digests and also call the exporter (grep for `build_pwml_ir`,
`run_pwml_export` and the like, combined with `sha256`, `DIGEST` or `golden`):

| run | result | artifact |
|---|---|---|
| tip, before re-pin: `test_c030_*`, `test_compound_resolution_extraction`, `test_orch768_replay_corpus`, `test_orch773_omit_per_row`, `test_prefreeze_compound_resolution`, `test_prefreeze_third_export_seam`, `test_pwml_interaction_protein_wrappers` | `4 failed, 196 passed` | `g11/ORCH-804/18` |
| same 7 files, **base** `563d39db` (a detached git worktree at `C:/t/orch804/basewt`; the tests call `git ls-files`, so an archive tree cannot run them) | `3 failed, 197 passed` | `g11/ORCH-804/20` |
| tip after the GOLDEN re-pin | `3 failed, 197 passed`: **the same 3 nodes as base** | `g11/ORCH-804/23` |

The 3 nodes that fail on base too are not caused by this patch:
`test_the_golden_covers_every_committed_leg_fixture`,
`test_the_repaired_coverage_and_digest_pins_themselves_go_red` (committed `runs_verify/2026-08-27..`
legs are neither in GOLDEN nor in EXCLUDED), and
`test_prefreeze_third_export_seam::test_run_pwml_export_runs_the_prefreeze_sequence_once_before_build_pwml_ir`.

The one new failure was `test_build_pwml_ir_matches_the_pre_extraction_golden`: 35 of 89 GOLDEN digests
moved (`g11/ORCH-804/21`, log `orch804/golden_before_repin.pytest.log`). See § 5.

`test_streamlit_quarantine_boundary.py` is an AppTest file and was **not run**. Batched with the others, it
hit the 570 s wrapper timeout (`g11/ORCH-804/17`, `exit code None`, 0 survivors), which is the known AppTest
stall. It belongs to chunk D, which the brief excludes.

## 6. Dead ends and corrections

1. **The first base replay was 0/91.** The `git archive` base tree had no `reference/` directory. Every
   export failed on `reference/PW000001.pwml`. Fixed by archiving `reference/`.
2. **The second base replay changed canonicals on 5 legs** (ATP vs Adenosine triphosphate,
   `pyridoxal 5-phosphate` vs `Pyridoxal 5'-phosphate`). I suspected the quarantine. It calls `build_pwml_ir`
   as its IR gate (`strict_quarantine._downstream_gate_errors`), so `resolve_state` can reach the quarantine.
   A rerun of base on those 5 legs was deterministic (G11 `09`), which ruled out nondeterminism. The real
   cause: the base tree also lacked `data/` (enrichment and id-mapping caches), so name resolution differed.
   After archiving `data/`, base reproduces the ORCH-799 record 91/91. With a correct base, **no canonical
   and no quarantine count moves**. The quarantine path is reachable in principle, but no corpus leg's
   outcome changes through it.
3. My first per-leg diff keyed locations by list index. That showed spurious "moves" into paper compartments,
   which were index shifts. Locations are now compared as an (entity, state) set.
4. Some small non-test Python ran outside the wrapper, recorded here for completeness. Two read-only
   one-line `python -c` inspections of committed JSON (the rice IR state list and the IR top-level keys).
   Five throw-away text-patch scripts under `C:/t/orch804/`. They edited `state_diff.py` and this doc, and `repin_golden.py` copied the 35 new GOLDEN values out of the committed pytest log. None
   imports `t2pw`, and none spawns a child process.
5. **Scratch not cleaned up (cleanup was denied by the permission classifier).** Still present:
   - The detached base worktree `C:/t/orch804/basewt` is still registered in `.git/worktrees`. Remove it
     with `git worktree remove --force C:/t/orch804/basewt`.
   - Two directory **junctions**, `C:/t/orch804/base/runs` and `C:/t/orch804/base/runs_verify`, point
     at this worktree's `runs/` and `runs_verify/`. **Remove them with `rmdir` (cmd), never with
     `rm -rf`.** A recursive delete through a junction deletes the worktree's fixtures.
6. While the heavy lock was held by ORCH-803, five of my wrapper calls were refused with exit 95 (none
   started a child). Each was retried after the lock was released.
7. The rice leg itself is not in the corpus. Its committed IR has `__auto_state__` (`bs_7`), so the fix
   applies to it. It was not replayed (no pipeline legs, per the brief).

## 7. Open for the PO

- **Q1. 20 of 30 legs with empty process states have no `__auto_state__`**, so this fix cannot reach them.
  `ensure_autostates` never assigns reactions or interactions, and pruning then drops the unused state.
  Options:
  - (a) Upstream: have `ensure_autostates` also give stateless reactions and interactions
    `__auto_state__`, before the freeze. About 10 lines in `process_normalizer.py`. It moves canonical
    hashes on about 20 corpus legs and needs its own corpus A/B. Risk: the quarantine's state-without-location
    handling (ORCH-788) sees new states.
  - (b) Exporter: synthesise a `cell` state when none exists. Arguably a rule-8 breach, and it can add
    `biological_state_missing_*` errors that drop PWMLs. Not recommended.
  - (c) Leave as is and record the remaining legs as a known limit.
- **Q2. Empty reaction state when all members have explicit locations in one paper state** (R3 leg). Neutral
  placement duplicates the member nodes. Inferring the reaction's state from its members' common explicit
  state would avoid the duplication. That is a new inference rule and needs a decision. Two corpus legs are
  single-compartment bacteria (E. coli cytosol -> cell). Biologically harmless either way.
- **Q3.** An **unresolved named** state reference still falls back to the first state (`resolve_state`'s
  last line). It raises an `unresolved_biological_state` issue (an error in strict mode), so it is visible.
  Same hazard class. Not changed here.
- **Q4.** Should an empty reference that takes the first-state fallback carry a report warning? Cost: every
  report digest that covers such a leg moves. Benefit: the invented placement becomes reviewable.

## 8. v2: the lead's refinement for the PO's "no duplicates" rule (this is the version up for review)

**§§ 3-5 describe v1 (empty -> `__auto_state__`) and are kept as the record.** v2 replaces them. v2
commits: `1e604b3a` (code and test), `f75bd2fb` (reverts the v1 R3 re-pin), `8b7d6bc6` (reverts the v1
GOLDEN re-pin), `c8f31a3e` (re-pins GOLDEN against the base values).

### 8.1 Duplicate measurement on v1 (requested first)

A duplicate here means one entity drawn in two or more biological states in the IR `locations`.
`evidence/orch804/dup_count.py`, G11 `24`, output `orch804/dup_count_v1.txt`, base vs v1 tip:

| leg | multi-state entities base -> v1 | new dups | dups gone |
|---|---|---|---|
| ORCH-759/chlorophyll | 36 -> 35 | 0 | 1 |
| ORCH-767/wnt_ca | 8 -> 6 | 0 | 2 |
| runs_verify/08-18_1328/PMC12096016 | 10 -> 12 | 3 (isochorismate, EntE, isochorismatase) | 1 |
| runs_verify/08-18_1328/PMC12452463 | 3 -> 7 | 4 (DHB, isochorismate, EntE, isochorismatase) | 0 |
| runs_verify/08-21_2239/PMC12782028 | 3 -> 4 | 2 (cholesterol, lanosterol) | 1 |
| runs_verify/08-27_1341/PMC12096016 | 1 -> 3 | 2 (NAD+, pyruvate) | 0 |
| runs_verify/08-28_1816/PMC12782028 | 5 -> 2 | 0 | 3 |
| runs_verify/09-01_1612/PMC12096016 | 2 -> 3 | 1 (TolC) | 0 |
| the other 6 changed legs | unchanged | 0 | 0 |

**v1 total: 12 new duplicate entities and 8 removed.** Every new duplicate pairs `__auto_state__` with the
paper state where element_locations already drew the member.

### 8.2 Rule (v2)

An empty process state (reaction, interaction, or transporter actor) resolves in this order:

1. **The one paper state that all its located participants share**, if they agree on exactly one.
   - Participants are: for a reaction, its inputs, outputs and `enzymes`; for an interaction, its two
     endpoints; for a transporter row, the transporter itself.
   - A participant's states are its `element_locations` rows, already resolved to IR state keys.
   - Participants with no location row are skipped.
   - `__auto_state__` rows are ignored, because they are a placeholder and not a paper location.
2. Otherwise `__auto_state__`, if present.
3. Otherwise the old first-state fallback. It never raises and never drops a PWML.

**Why this places rather than invents (rule 8).** The state comes only from the frozen payload's own
`element_locations`, which already draw those members in that state. The exporter adds no location and
no compartment. It only refuses to draw the members a second time somewhere else. When the members
disagree, or none is located, there is no shared fact to place by, so the rule goes neutral.

**Participant states were available.** The `element_locations` loop runs before the process loops, so
`build_pwml_ir` records them in `explicit_states` there, and `members_state` reads them through
`entity_by_name` without side effects. No new machinery: about 20 lines in `ir.py`.

Ceilings (not built):
- Reaction `modifiers` with a catalyst role are not counted as participants.
- An unlocated member of a placed reaction is drawn in the shared state of its co-participants. A
  stricter variant ("every participant must be located") is Q5 below.

**Dead end inside v2.** The first v2 counted `__auto_state__` rows as evidence. That left **4** new
duplicates on 08-18 PMC12096016 (isochorismate, isochorismatase) and 08-21_2239 PMC12782028
(cholesterol, lanosterol). That run's G11 `30` report and `dup_count_v2.txt` were overwritten by the final
run, so this paragraph is the only record of the 4. On those legs one member sat in a paper state and the
other in `__auto_state__` (placed there by `ensure_autostates` because it had no location). So the
intersection was empty and the reaction went neutral. Evidence: `orch804/explain_leg.py`, G11 `31`/`32`,
output `orch804/explain_remaining_dups.txt`. Ignoring placeholder rows fixed it.

### 8.3 Proofs (all bounded, heavy lock `ORCH-804`, 0 survivors)

- **Focused test, tip:** `4 passed` (G11 `25`).
- **Focused test, base `563d39db`:** `2 failed, 2 passed` (G11 `26`), with these failures:
  - `test_empty_reference_does_not_take_the_first_paper_state`: `assert 'peroxisomal state' == '__auto_state__'` (rice fixture, no member locations -> neutral).
  - `test_empty_reference_takes_the_state_its_located_members_share`: `assert 'cytosolic state' == 'peroxisomal state'` (new: both members in the peroxisome -> the reaction goes there and every entity is drawn once).
  - Both corrections fail on base and pass at tip (G9).

  G11 `25` is the final code (it was rerun after the placeholder fix, which overwrote the first-v2 report).
  G11 `26` ran the same test file against base.
- **Corpus, base vs v2 tip** (G11 `27`, `28`; `orch804/compare_base_vs_tip_v2.txt`): `pass rate base 79/91 -> tip 79/91`,
  `passing legs turned failing: none`, `gold recall lost: none`, `still failing, at a different step: none`.
  23 legs changed, 68/91 identical.
- **Duplicates, base vs v2 tip** (G11 `30`, `orch804/dup_count_v2.txt`): **`total new duplicate entities: 0`**.
  23 multi-state entities are removed across 17 legs.
- **Per-leg state diff** (G11 `29`, `orch804/state_diff_base_vs_tip_v2.txt`): on all 23 legs only the
  PWML/IR files and `pwml_sha256` move (plus `pwml_matches_leg` on 3). The canonical, quarantine,
  coverage, gates and release are unchanged. `entities with no location left at tip: []` on all 23.
- **Pins.** R3 is **byte-identical to base** under v2 (IR `14af62c1`, report unchanged; the test passes at
  the base pins in G11 `34`), so the v1 R3 re-pin was reverted. GOLDEN: 33 of 89 legs move. The
  instrument (G11 `35`, `orch804/orch804_ir_digest_delta_v2.txt`) shows only state keys and what follows
  from them; `errors / ok / stops unchanged on every leg: True`. Re-pinned with pytest's values
  (`golden_repin_pairs_v2.txt`, `golden_before_repin_v2.pytest.log`).
- **Affected tests after the re-pin** (14 files, G11 `36`): `3 failed, 397 passed`. The 3 failures are
  the same nodes that fail on base (§ 4.5).
- **SMOKE** (G11 `37`): **`511 passed in 59.01s`**.

### 8.4 Changed legs under v2 (23), by destination of the moved rows

| leg | rows: from (base first state) -> to | class |
|---|---|---|
| ORCH-755/enterobactin | 4 transporter rows: cytoplasmic -> E. coli extracellular (TolC/TonB's own location rows) | placed by members; dup removed (2) |
| ORCH-759/chlorophyll | 2 interactions: plastid -> `__auto_state__` | neutral; dup removed (1) |
| ORCH-759/menaquinone | 1 row: L. lactis cells -> intracellular state | placed by members |
| ORCH-759/wnt_pcp | 5 rows -> `__auto_state__`, 1 -> cytoplasmic (from plasma membrane) | neutral / placed; dup removed (1) |
| ORCH-759p/enterobactin | 1 transporter: cytoplasmic -> outer membrane | placed; dup removed (1) |
| ORCH-759p/sour_taste_v2 | 3 OTOP1 transporter rows: Type III cell state -> `__auto_state__` | neutral |
| ORCH-767/enterobactin | 4 rows: "cytoplasmic" -> "cytoplasmic state" (same-compartment duplicate states) | placed; dup removed (2) |
| ORCH-767/wnt_ca | 14 -> `__auto_state__`, 1 -> cytosolic state (from cytoplasmic) | neutral / placed; dup removed (2) |
| ORCH-767/wnt_pcp | 4 rows: cytoplasm -> plasma membrane | placed |
| runs_verify/08-18_1328/PMC12452463 | FepA -> outer membrane; 1 -> `__auto_state__` (from cytosol) | placed / neutral; dup removed (1) |
| runs_verify/08-21_2057/PMC12452463 | FepA: cytoplasmic -> outer membrane | placed; dup removed (1) |
| runs_verify/08-21_2239/PMC12096016 | TolC -> outer membrane, TonB -> inner-membrane AutoState (from cytoplasmic) | placed; dup removed (2) |
| runs_verify/08-21_2239/PMC12452463 | FepA: cytosolic -> outer membrane | placed; dup removed (1) |
| runs_verify/08-22_2147/PMC12452463 | FepA: extracellular -> outer membrane | placed; dup removed (1) |
| runs_verify/08-22_2147/PMC13231680 | 1 interaction: cytoplasm -> `__auto_state__` (IR only, no PWML) | neutral |
| runs_verify/08-24_1428/PMC12444477 | MsbA: cytoplasmic -> cytoplasmic leaflet of IM | placed; dup removed (1) |
| runs_verify/08-25_1216/PMC12444477 | 3 MsbA rows: cytoplasmic -> E. coli inner membrane | placed; dup removed (1) |
| runs_verify/08-28_1816/PMC12452463 | FepA: cytosolic -> outer membrane | placed; dup removed (1) |
| runs_verify/08-28_1816/PMC12782028 | CYP51A1, DHCR24: ER state -> ER-membrane AutoState (members' explicit rows) | placed; dup removed (3) |
| runs_verify/09-02_2052/PMC12444477 | MsbA: cytoplasmic -> IM cytoplasmic-leaflet state | placed; dup removed (1) |
| runs_verify/09-06_1425/PMC12071552 | 2 DltD rows: cytoplasmic -> membrane state | placed; dup removed (1) |
| runs_verify/09-06_1425/PMC12376012 | 2 reactions: ER state -> Golgi state | placed; dup removed (3) |
| runs_verify/09-06_1425/PMC8510960 | 1 -> `__auto_state__`, 1 -> leaf surface (from epidermal cell cytosol) | neutral / placed |

Read off `state_diff_base_vs_tip_v2.txt`. Every placed destination is a state the same entities already
had in `element_locations`. Several are biologically better than the arbitrary first state: FepA, TolC
and MsbA move from the cytosol to their membranes. Placement does not judge the quality of an
element_locations row. TolC/TonB go to "E. coli extracellular" in ORCH-755 because that is their only
paper location row.

Four v1-changed legs are identical to base under v2 (08-18 PMC12096016, 08-21_2239 PMC12782028,
08-27 PMC12096016, 09-01 PMC12096016). On those legs the located members share the state the first-state
fallback happened to choose.

### 8.5 Open for the PO (added by v2)

- **Q5. Unlocated participants.** v2 skips unlocated participants when it looks for the shared state.
  The stricter alternative ("every participant must be located, otherwise neutral") never draws an
  unlocated member in its co-participants' state, but some duplicates come back. That has not been
  measured. Cost: a one-line change plus a corpus rerun.
- **Q6.** Reaction `modifiers` with a catalyst/activator/inhibitor role are not counted as participants.
  Adding them would mean duplicating the role filter in the reaction loop. No corpus duplicate came from
  them.
- Q1 to Q4 still stand. Q2 (duplicate member nodes) is resolved by v2. For Q1, v2 now also places
  empty-state processes on legs with no `__auto_state__`, wherever their members share a state. I have
  not re-counted how many of the census's 20 legs this reaches.


## 9. Review outcome: REJECT, branch HELD and not merged (lead, 2026-11-04)

An independent `pwml-reviewer` reviewed v2 (`4ccaddc1`) and rejected it. The code change itself is small and well-behaved. The rejection rests on the evidence claim in § 8.3 and § 8.5.

- **"0 new duplicate entities" is false.** `evidence/orch804/dup_count.py` (lines 26-27, 35) defines a duplicate only as "the entity is multi-state at all". It diffs sets of entity names, so an entity that was already multi-state at base and gains another drawn state is invisible to it. The reviewer recounted by (entity, state) pairs over `C:/t/orch804/corpus_base` vs `C:/t/orch804/corpus_v2` (79 IRs):
  - Whole corpus: **489 -> 471** extra drawn copies, net -18.
  - **Two legs get worse:** `ORCH-759_menaquinone` 24 -> 25 (MenA complex 2 -> 3 states) and `runs_verify_2026-08-28_1816_PMC12782028` 5 -> 6 (lanosterol 2 -> 3).
  - `ORCH-767_wnt_ca` nets flat: Ca2+ 10 -> 11 and calcineurin 2 -> 3 are offset by removals.
- **Mechanism:** the Q5 skip-unlocated choice draws an unlocated member, for example `MenA complex` on menaquinone `rxn_16`, in its co-participants' state. That is inference, not placement, and it adds the node.
- **The target defect is not demonstrated.** The rice PAL leg (PMC12408352) is not in the corpus, and `compare_base_vs_tip_v2.txt` ends `D-112 § 2: REJECT`.
- **Lower findings:**
  - No stdout was committed for G11 25, 26 and 36.
  - Gap-resolver `AutoState_*` states are counted as paper states.
  - `members_state` takes the union across same-name rows of different types.
  - The empty-name path also covers `element_locations` and cargo endpoints. That is not observable today.
- **Lead decision:** the branch is **held**. The PO's standing rule is never to worsen text-to-PWML output and not to create duplicates; a proven +1 on two legs violates it even though the corpus total improves. The PO asked to wrap up rather than iterate.

**Open for the PO, if this is ever resumed:**
- (a) Accept the trade-off: net -18 duplicate copies corpus-wide against +1 on two legs.
- (b) Add a "never add a drawn (entity, state) pair" guard and re-measure. This may leave the rice case unfixed, because moving PAL off the peroxisome would draw trans-CA in a new state.
- (c) Fix the cause upstream instead: `ensure_autostates` should assign processes (Q1a).
- In every case: run the rice leg to show F7 fixed, and recount duplicates by (entity, state).
