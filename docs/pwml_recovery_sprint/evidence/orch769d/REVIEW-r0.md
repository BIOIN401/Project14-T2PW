# ORCH-769d review, round 0

Reviewer: independent pwml-reviewer (no edit tools on the branch). Date: 2026-10-25.
Branch `orch769d/audit-rebase`, base `3ef3fa9c`, tip `22a306a0` (code `2a791f69`, tests `96848d43` / `2b0ea179` / `c03ab59f`, docs `22a306a0`).
Review worktrees: `.claude/worktrees/rev769d-base` (detached `3ef3fa9c`, plus an untracked copy of the new test file for the G9 run) and `.claude/worktrees/rev769d-tip` (detached `22a306a0`). Scratch and scripts: `C:/t/rev769d/`.
All line numbers below are `src/t2pw/curation/apply_audit_patch.py` at the tip unless stated.

## VERDICT

**APPROVE WITH NOTES.** No blocking finding under the fixed severity standard. Nine notes (N1 to N9), two of them small hardening suggestions inside the boundary (N1, N2) that the orchestrator may route as a follow-up or as an optional round-1 polish. None is a regression against base in a realistic patch.

## BOUNDARY CHECK

* `git diff --name-status 3ef3fa9c..22a306a0`: modified `src/t2pw/curation/apply_audit_patch.py`, `tests/test_apply_audit_patch_referential_integrity.py`; added `tests/test_orch769d_audit_patch_rebase.py`; everything else added is under `docs/pwml_recovery_sprint/` (`ORCH-769d-RESULT.md`, `evidence/g11/ORCH-769d/`, `evidence/g11/pin/ORCH-769d/`, `evidence/orch769d/`). No LEDGER, TEST_MATRIX, FINDINGS or DECISIONS edit.
* Per commit: `96848d43` test file only; `2a791f69` the applier only; `2b0ea179` the referential-integrity test only; `c03ab59f` test file only; `22a306a0` docs/evidence only (54 files, all additions).
* Inside the applier the code change is confined to: one comment block (`:826-835`), the trailing comment of `_should_accept` (`:2648-2654`, comment only, no code), three new helpers and a constant (`:2658-2752`), and `apply_patch_with_policy` (`:2778-2858`).
* `src/t2pw/app/streamlit_app.py`: `git diff --quiet 3ef3fa9c..22a306a0 -- src/t2pw/app/streamlit_app.py` exit 0, byte-identical.
* Verdict: inside the boundary.

## TESTS I RAN (my own output; every run through `bounded_run.py --heavy-lock ORCH-769d-rev`, FINAL SURVIVING COUNT 0 on every report)

G11 reports in `rev769d-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-769d/`; piped stdout in `C:/t/rev769d/<SEQ>-<label>.out.txt`.

| G11 | tree | what | result |
|---|---|---|---|
| 18 | base | my 29 adversarial patches (`C:/t/rev769d/adv_rebase.py`) | base fails 15 of the 24 cases that carry a ruling-(a) expectation (wrong row written, neighbour deleted, or an op meant for a removed row applied elsewhere) |
| 19 | tip | same | all 24 expectations met; case 18 raises (N1) |
| 20 | base (worktree at `3ef3fa9c` + copied test file, `--expect-tree`) | `tests/test_orch769d_audit_patch_rebase.py` | **7 failed**, all AssertionError on payload content or reason text (pin `20-rev-g9-red-base.pin.json`: t2pw = base src, refused false, no foreign src) |
| 21 | tip | new file + the 13 applier files + `test_completeness_audit.py` | **660 passed** |
| 22 | tip | SMOKE, 22 files | **511 passed** |
| 23 / 24 | base / tip | `C:/t/rev769d/probe2.py`: cumulative coverage, relocation after remove, F-5 shape, leading-zero index, four no-shift lock-manifest patches | see N2, N3, N7; the four no-shift lock cases P5-P8 are byte-identical base vs tip (full payload + report, md5 per line) |
| 25-28 | tip / base | first two oracle runs | superseded: my oracle had two bugs of its own (it aliased op values that the applier mutates, see N8; and it did not strip lineage on both sides). Kept for the record |
| 29 / 30 | base / tip | implementer replay `evidence/orch769d/replay_wntca_audit_patch.py` on the ORCH-767 leg | identical to the implementer G11 10 / 11 result JSONs except the `t2pw` path field (`diff`, including `applier_sha256`). Base: refused `unexportable_entity:1`, `diagnostic_only`, reproduces the recording. Tip: ok, `review_required`, final gate PASS 0/9/13, 0 cross-row synonyms |
| 31 / 32 / 33 | base / tip / compare | implementer A/B corpus `ab_no_shift_corpus.py` | no_shift 3000/3000 identical; shift 307 differ, all 307 among the 1014 with an accepted base remove/insert. My per-case hash lists are byte-identical to the implementer `13-` and `14-...hashes.json` (ignoring the tree path) |
| 34 | tip | 14 caller files, gap-resolver files last | **283 passed** |
| 35 / 36 | tip / base | independent-oracle fuzz `C:/t/rev769d/oracle_fuzz3.py`, 8000 random patches | superseded by 37/38 |
| 37 / 38 | tip / base | `C:/t/rev769d/oracle_fuzz4.py`, 8000 random patches (4000 flat incl. wholesale row replace, 4000 with nested synonym arrays) | **tip: 0 target or verdict mismatches.** The only differences are the relative order of two newly inserted items inside one gap: 33 outer cases classified automatically, and the 1 remaining nested case, which I inspected by hand, is the same thing in a synonyms array (N5). **Base: 4584 mismatches**, so the oracle has teeth |

`g11_evidence.py check --task ORCH-769d` on my tip worktree: exit 0, 38 reports, 14 pytest reports all with a pin verdict, staging empty.

The oracle (`oracle_fuzz4.py`) does no index arithmetic: every pre-patch row and inner synonym carries an identity and a sort key, every op resolves its target by pre-patch index to identity, an op at a removed identity must be `removed_target`, an index outside the snapshot must be `Application failed`. It compares the final rows and the per-op accept/reject class. To keep it a pure ruling-(a) oracle it does not generate an out-of-snapshot index after a `-` append (that reading is N6).

## REGRESSION PROOF (fails on base, behaviourally)

G11 20, base `3ef3fa9c`, t2pw resolved to `rev769d-base/src/t2pw/__init__.py`:

```
E  AssertionError: assert {'A': '', 'C'...ritten for C'} == {'A': '', 'C'...r C', 'D': ''}      (note landed on D)
E  AssertionError: assert {'A': '', 'C'...dup', 'D': ''} == {'A': '', 'C': '', 'D': ''}         (note for the removed row landed on C)
E  AssertionError: assert ['A', 'D'] == ['A', 'C', 'D']                                         (repeated remove deleted C)
E  assert {'entities': ...} == {'entities': ...}   (committed with synonym X on B)
E  AssertionError: assert [('A', 'writt...itten for D')] == [...]   At index 2 diff: ('D', '') != ('new', '')
E  assert "'meropenem' at /processes/reactions/0/inputs/0" in "referential_integrity: remove on /entities/compounds/2 ... 'hydrolyzed meropenem' ..."
E  AssertionError: assert ['calcineurin B'] == ['phosphorylated CAMKII']
7 failed in 0.18s
```

Every failure is on payload content or on the guard reason, none on an import or a missing symbol (the file imports only `apply_patch_with_policy`; reason prefixes are literals). G9 holds. The tip run (G11 21) is green.

## INVARIANTS (cited diff lines)

1. **No op redirected to a row it was not written for; an op at a removed row is rejected and recorded.** `_rebase_op` (`:2690-2725`) walks each array-index token through the events of that array in order. A remove at `at` lowers indices above it (`:2718-2719`); an insert at `at` raises indices at or above it (`:2720-2721`). A token equal to a removed row returns a `removed_target:` reason (`:2712-2717`) unless it is the final insertion position of an add (`:2708`). Walking every intermediate coordinate system in order is correct because each event stores the live index at which it happened (`:2744`) and shifts are keyed by the pre-patch path of the array (`:2703`, `:2744`). The reason is applied after the policy (`:2812-2813`) and lands in `rejected` and in the rejected log (`:2815-2818`). Reproduced: cases 06, 08, 09, 11, 24 (G11 19); oracle G11 37 (2658 cases with a removed-target op, all classified correctly).
2. **The acceptance policy is unchanged.** `_should_accept` is still called with the op as written and `source_payload` (`:2804-2810`), exactly as at base. It is a pure function of those two arguments, so no verdict it returns can change. Only the lock validators (`:2795`, `:2823`), the application (`:2821`), the integrity guard (`:2837-2839`) and lineage (`:2850`) receive `live_op`; each of those judges the live payload, so handing it the live address is the rebase itself, not a policy change. Cases 22/23 show the lock pre-op now judges the locked reaction the op was written for (base wiped the inputs of the unlocked neighbour R2).
3. **No change for a patch with no index-shifting op.** `shifts` starts empty (`:2779`) and is only appended to for an accepted add/remove whose live parent is a list and whose leaf is a digit (`:2743-2745`); every other branch only deletes keys (`:2751-2752`). With `shifts` empty, `_rebase_op` returns the op object itself (`:2694-2695`), so `rebased == {}` (`:2793`), the lock pre-op condition is the base condition (`:2794`), and every record and log entry has the base keys in the base order (`:2798`, `:2814`, `:2855-2857`). Measured: A/B no_shift 3000/3000 identical (G11 33); P5-P8 with a lock manifest and log arrays identical (G11 23/24).
4. **Shift recording sees only committed ops.** `_record_index_shift` runs after the lock post-op and the integrity guard have passed, inside the try, before `working` is replaced (`:2851-2852`), against the pre-op payload. A refused or failed op records nothing: cases 13, 14, 28 (G11 19).
5. **No exporter or biology gate touched or weakened.** The diff is confined to the applier; no quarantine, gate or registry code changed. The integrity guard, the batch rollback and the lock validators are called unchanged, only with the correct address.
6. **Rejected ops are recorded.** Every early `continue` appends to `rejected` and `rejected_log_records` (`:2801-2803`, `:2815-2818`, `:2829-2831`, `:2842-2844`, `:2860-2862`). Case 29 shows the removed-target rejection in `rejected_patch_log.json`.

## DECISION COMPLIANCE

* **D-113, orchestrator ruling for 769d, option (a):** satisfied. Later ops address the pre-patch snapshot and are translated to the live payload after an accepted remove or insert; an op at a removed row is rejected, not redirected.
* **D-104 (LLM-bound text):** n/a, no prompt changed.
* **D-114:** n/a.
* No item from DECISIONS.md Open was resolved by the implementer. The `/-` defect and the rollback blast radius are left as open questions in RESULT 7, as they should be.

## FINDINGS

| # | severity | where | what | why it matters |
|---|---|---|---|---|
| N1 | NOTE (theoretical) | `:2792`, `:2709` | `_rebase_op` is called before the per-op `try` (`:2819`). Its `int(token)` raises for a token that `str.isdigit()` accepts but `int()` refuses (superscript or circled digits, e.g. U+00B2). Case 18 (G11 19): after an accepted remove in the same array, `apply_patch_with_policy` raises `ValueError` out of the call; base (G11 18, case 18) records `Application failed`. Case 19 (no shift) is identical on both trees. | A crash instead of a recorded rejection, but only for a pointer token no producer writes. Fix inside the boundary: treat a token whose `int()` fails as a non-index in `_rebase_op`, or move the call inside the `try`. |
| N2 | NOTE (theoretical, same as base) | `:2703`, `:2744` | Shift keys use raw tokens. A non-canonical index in a path prefix (`/entities/proteins/00/synonyms/1` after `remove /entities/proteins/0/synonyms/0`) misses the key and is applied unrebased, so it lands one synonym late, exactly as at base (P4, G11 23/24). The index of the array itself (`"01"`) is handled because it is compared as an int. | RFC 6901 forbids leading zeros and no producer emits them, so this is not a realistic redirect. Fix if wanted: key on canonical index tokens. |
| N3 | NOTE (confirms F-5) | `:2777`, `:2812-2813` | The look-ahead still credits a later op aimed at the removed row; that op is now rejected, so the removal orphans its name and the whole batch rolls back (G9 test; P3 at G11 24: base committed `X` as a synonym of `B`, tip rolled back). | See Q2 below. Correct under the ruling and fail-closed; the cost is the rest of the batch. |
| N4 | NOTE (wording) | `:2837-2843`, `:2802`, `:2817`, `:2860-2862` | Reasons produced after the rebase cite the live address (`referential_integrity: remove on /entities/compounds/0 ...` while `op.path` is `/entities/compounds/1`, P1; `Array index out of range: 8` for written `/9`, case 28). `applied_path` is on the report record and on the applied log, but not on rejected-log entries (case 29). | A reader of `rejected_patch_log.json` sees a reason path that matches no patch path and has no `applied_path` to reconcile it. Suggest adding `applied_path` to the rejected-log record too. |
| N5 | NOTE (informational) | `:2720-2721` | New items inserted into the same gap keep patch order: two inserts at one pre-patch position (case 16: tip `N1, N2`, base `N2, N1`), and two inserts at different pre-patch positions whose separating rows were removed (oracle G11 37: 33 outer cases plus 1 inner synonyms case). Existing rows are always in the right place with the right content. | Not a redirect; row order inside `entities.*` and synonym order carry no biology. Worth one sentence in the block comment. |
| N6 | NOTE (undocumented semantics) | `:2746-2748` | Appends record no event, so an index at or beyond the snapshot length addresses appended rows in append order, walked through later removes: `[add /x/-, remove /x/0, replace /x/<n>/note]` writes the appended row at the tip, where base fails the third op (found by oracle G11 27, cases 10/15/17/18). Indexed inserts are never addressable (case 17, RESULT 7.4). Coherent ("appended rows are numbered after the snapshot") and the same as base when no remove precedes. | No snapshot writer emits an index at or beyond n, so not realistic; but RESULT 7.4 should name the append case next to the indexed-insert case, so the product owner sees both readings. |
| N7 | NOTE (test coverage) | `tests/test_apply_audit_patch_referential_integrity.py:295-336` | The moved pin now expects the correct ruling-(a) result (spectator removed, `applied_path == /entities/compounds/0`, registry valid). Its old intent (the guard judges what the op actually costs on the live payload) is covered by `test_g9_the_integrity_guard_judges_the_rebased_row` for a single referenced row, but no test now pins the cumulative case: two removes of two duplicate rows, the first harmless, the second orphaning. The tip handles it correctly (P1, G11 24: the second remove is refused for `meropenem`; base refused the wrong row, `hydro`). | Suggest one small test with two duplicate rows; not required for merge. |
| N8 | NOTE (pre-existing, out of scope) | `_set_value` `:92-108`, `_record_audit_lineage` `:1226-1235` | The applier inserts the value object of the op itself into the payload, and lineage is then recorded on that object, so the patch list of the caller (and `report["accepted"][i]["op"]["value"]`) gains `provenance_lineage`. Seen in my oracle runs on both trees (G11 25/26 examples). | Not caused by this lane. A FINDINGS candidate: a patch list reused after application carries lineage it did not write. |
| N9 | NOTE (evidence scope) | `evidence/orch769d/ab_no_shift_corpus.py` | The A/B no-shift corpus runs without a lock manifest and without log paths, so the lock branch of severity item 2 rested on the by-construction argument only. I closed it with P5-P8 (G11 23/24, byte-identical). | Evidence gap only, now closed. |

**No BLOCKING finding.** Against the six blocking items: (1) no redirect in any realistic patch, N2 is a non-canonical pointer and N6 is an index outside the snapshot; (2) no-shift identity measured three ways; (3) `_should_accept` inputs are unchanged, every verdict difference goes through the rebase; (4) base-red is behavioural; (5) boundary holds, streamlit byte-identical; (6) every rejection is recorded.

## THE MOVED PIN

`test_index_drift_across_two_removals_is_judged_against_the_live_payload` (`2b0ea179`). Same input `[remove /entities/compounds/0, remove /entities/compounds/1]` over `[orphan-a, spectator, meropenem, hydrolyzed meropenem]`. In the snapshot, index 1 is `spectator`, unreferenced, so under ruling (a) the correct result is two accepted removals leaving `[meropenem, hydrolyzed meropenem]` with a valid registry. That is what the new expectation asserts, plus `applied_path == /entities/compounds/0` and `validate_registry_references(patched)`. The old expectation pinned the defect (the second op hit `meropenem`). The move is deliberate, documented (RESULT 6) and correct; the test count is unchanged. The guard intent is still tested for the single-row case by the new G9 test; the cumulative case is untested (N7).

## VIEWS ON RESULT 7 (not blocking)

* **Q1, the `/-` defect (F-2, F-3): leaving it unfixed here is right.** Any fix changes a no-shift patch, which this lane may not do, and the root is `_resolve_parent` auto-creating dicts for every missing key (`:67-71`), which also produces F-3. My preference for the follow-up lane is option (a), fail closed (`Application failed`, recorded): option (b) silently creates registry coverage (a synonym that can keep a row referenced, the very mechanism of KI-018), and option (c) keeps dict-valued `synonyms` flowing downstream, which the Wnt/Ca2+ leg already shows. Fix F-2 and F-3 together in `_resolve_parent`, with its own A/B.
* **Q2, whole-batch rollback (F-5): not worse for a user than base.** Base commits a synonym on the wrong row: a silent biology error that can keep an unidentified protein referenced, invisible in the output, which is KI-018 itself. Rollback is loud (`transaction.rolled_back`, reason, rejected log) and leaves the payload as the last gate accepted it, so the next audit round can retry. The trigger is narrow: a later op addressed to the very row being removed, a referenced name on that row, and that op being the only compensation. It did not occur on Wnt/Ca2+ (15/15 accepted at the tip). The smaller-blast-radius option is computable at removal time (exclude from the look-ahead every later op whose path passes through the row being removed, so the removal itself is refused per-op and the rest commits); recommend it as a follow-up, since it changes the integrity guard and belongs in its own lane.
* **Q3 (op into an array rewritten wholesale):** applied as written, same as base (case 15). Fine as a documented ceiling.
* **Q4 (address a newly inserted row by index):** inherent to (a) (case 17). Add the append variant from N6. The prompt clarification ("all paths address the payload as shown") is cheap and worth a D-104 item.
* **Q5 (live confirmation):** agreed, only a live leg measures Stage 6 and the gap resolver on the corrected payload.
* **Q6 (`applied_path` key):** no consumer in `src/` validates record keys; I grepped the callers (`gap_resolver.py:3341`, `pathway_curator.py:348`, `focused_repair.py:215`, `interactive_curator.py:504-520`, `streamlit_app.py:3869/3939`): they read `summary`, `rejected[*].index/reason` and `committed_change_count`, none indexes the patched payload with a recorded op path. Harmless.

## WHAT I COULD NOT VERIFY

* The real ORCH-767 audit patch, its op order and its confidences: not persisted by the leg. The Wnt/Ca2+ evidence is a reconstruction (I re-ran it and got the implementer result byte for byte); it is consistent with, but does not prove, the exact patch the auditor emitted.
* A live pipeline leg on the tip (no LLM authorised).
* Readers of `applied_path` outside `src/` (scripts, notebooks, external tooling).
* Chunk D and the AppTest files: not run (not required for this boundary; the applier is reached from the app only through `run_apply`).
* My base tree is a full git worktree at `3ef3fa9c` with the new test file copied in, not the `git archive` export the implementer used; the pin verdict shows `t2pw` resolved to the base `src`.
* That no LLM producer writes sequential (RFC 6902) indices. I checked the curator and gap-resolver accumulation (single snapshot shown, dedupe on `(op, path)`), the deterministic auditor (`audit_json_llm.py:668`, `:1262`, descending removes), and the prompts (no "RFC 6902" or order wording). The interactive curator prompt asks the model to remove an entity and update its references, which can produce multi-remove patches; ruling (a) decides how those are read.

## PROCESS RECORD (my own conduct)

* Every Python job ran through `bounded_run.py --heavy-lock ORCH-769d-rev`, one at a time; free RAM was at least 13,416,028 KB before each (`RC=... freeKB_before=` lines). Exit 95 happened once (G11 27 attempt, logged in `27-rev-oracle2-tip.out.txt.wait`) and was retried after the lock cleared. Orphan sweeps after each job (`C:/t/rev769d/sweep.ps1`) found no orphan from my wrapped jobs.
* **Incident, my error:** an early inspection command accidentally contained a bare `python -` (a stray token in a `cat`/`grep` line), which started `.venv/Scripts/python.exe -` waiting on stdin (PIDs 195392 and child 189168, started 14:49). It is not a sprint job and holds no lock, but it violates hard rule 4. My `taskkill //PID 195392 //T //F` was refused by the permission classifier. **It is still alive; the orchestrator should end it with `taskkill //PID 195392 //T //F`.** It runs no code (it is blocked on stdin) and has no effect on any result above.
* I wrote only under `C:/t/rev769d/` and in my two review worktrees (G11 reports 18-38 and pins 20, 21, 22, 34 in the tip worktree; the copied test file in the base worktree). Nothing committed; no worktree removed; the lock was never touched by hand.

## G11 REPORTS (tip worktree `docs/pwml_recovery_sprint/evidence/g11/ORCH-769d/`)

18-rev-adv-base, 19-rev-adv-tip, 20-rev-g9-red-base, 21-rev-g9-applier-tip, 22-rev-smoke-tip, 23-rev-probe2-base, 24-rev-probe2-tip, 25-rev-oracle-tip, 26-rev-oracle-base, 27-rev-oracle2-tip, 28-rev-oracle2-base, 29-rev-wntca-base, 30-rev-wntca-tip, 31-rev-ab-base, 32-rev-ab-tip, 33-rev-ab-compare, 34-rev-callers-tip, 35-rev-oracle3-tip, 36-rev-oracle3-base, 37-rev-oracle4-tip, 38-rev-oracle4-base (all `.json`). Pins: `g11/pin/ORCH-769d/20-`, `21-`, `22-`, `34-...pin.json`. Stdout: `C:/t/rev769d/<SEQ>-<label>.out.txt`. Scripts: `C:/t/rev769d/adv_rebase.py`, `probe2.py`, `oracle_fuzz.py`, `oracle_fuzz3.py`, `oracle_fuzz4.py`, `wait_run.sh`, `sweep.ps1`.
