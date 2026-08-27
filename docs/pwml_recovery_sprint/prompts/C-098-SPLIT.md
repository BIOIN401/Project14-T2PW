# C-098 SPLIT — two arms, chartered separately, with a fixed merge order

**Supersedes `C-098.md` § 5's "determine which seam".** The determination is made, it was made by
measurement, and the answer is **both**. `C-098.md` remains authoritative for everything else — the
required behaviour in § 4, the ownership prohibitions in § 6, and the process rules in § 8 all still
bind. This document only splits the work and fixes the order.

**Finding:** F-135. **Depends on C-094 (`53eaf24`) — hard.** Both arms key on the marker C-094
introduces.

---

## 1. Why both — accepted, and recorded so nobody re-litigates it

The implementer measured the three states rather than reasoning about them:

| | base `5475ebc` | gate arm only | gate + cap |
|---|---|---|---|
| Stage-3 gate `ok` | **false**, 1 blocking error | true, 1 `review_finding` | same |
| post-normalization contract `ok` | **false** | true | same |
| `run_pwml_export` | **refuses, no XML built** | proceeds | proceeds |
| release status | **`diagnostic_only`** | `release_ready`, eligible **true** | `review_required`, eligible false |
| PWML file | **none** | **bare `pathway.pwml`** | `pathway.review_required.pwml` |

**The cap arm alone is impossible.** At base the post-normalization contract is `ok: false` and
`streamlit_app.py:4716` returns `{"ok": False, "output_path": ""}` **before any XML is built**. A
classification-only fix has no bytes to name, so § 7.1's *"`review_required` **with PWML**"* is
unreachable from that seam. Measured, not inferred.

**The gate arm alone is worse than the status quo.** It produces the PWML but lands `release_ready`
with `strict_acceptance_eligible: true` and writes a **bare `pathway.pwml`**. Before C-094 the leg
shipped that file with a false-but-present organism; gate-arm-only ships it with **no** organism and
still claims release-ready. That violates `C-098.md` § 4 ("never silently on `release_ready`"), § 7.6,
and `PRODUCT_CONTRACT` § 4 ("must not be represented as fully confirmed").

**The 6-shape control is unchanged in all three states** — `release_ready` / `pathway.pwml`.

---

## 2. C-098a — the cap arm. **Merges first.**

**Branch:** `card/C-098a-cap`, from **`53eaf24`** (C-094's tip, including the rename commit).
**Worktree:** `C:/t/c098a`.

**You own:**
* `src/t2pw/pipeline/release_status.py` — `gate_review_finding_rules` and
  `cap_release_for_unresolved_placeholder_species`, mirroring the existing
  `cap_release_for_prefreeze_declination`.
* `src/t2pw/batch/driver.py` — **the single wiring line in `_frozen_release_record`, and nothing
  else.** It already receives `pwml_result` carrying `stage3_gate_report` on every branch, so no
  signature changes. If you find yourself changing a signature, stop and report.

**Its acceptance criterion is that it is an observable no-op.** You measured the cap inert across
seven report shapes — absent key, empty list, `None`, CLI path-string, wrong severity, non-mapping
entries, missing rule — all returning `()` and leaving `release_ready` untouched, monotone, not
mutating the caller's record. **Turn that into committed tests.** This arm must be provably incapable
of changing any current behaviour, because it lands before the arm that produces the findings it caps.

**Fix the doubled reason string you found.** `unknown_placeholder_species_unresolved:unknown_placeholder_species_unresolved`
comes from the release reason constant and the gate rule name being the same string. They must
differ, and the emitted reason must read as one fact rather than a stutter.

**G9 labelling:** this arm is **new capability** — an explicitly labelled new acceptance test, no
fabricated base failure. Say so. Do not manufacture a base-SHA failure for a no-op.

## 3. C-098b — the gate arm. **Merges second.**

**Branch:** `card/C-098b-gate`, from **C-098a's tip**.
**Worktree:** `C:/t/c098b`.

**You own:** `src/t2pw/pipeline/process_normalizer.py` — the `review_findings` channel,
`_add_review_finding`, and `_is_marked_unknown_placeholder_wrapper` gating **one branch** of the
species check.

**G9 labelling:** this arm is a **correction** and carries the real behavioural proof — the identical
fixture at base gives `gate ok=false / contract ok=false / verdict failed / diagnostic_only /
artifact name ""`, and at tip gives `review_required` with `pathway.review_required.pwml`. That is
definitively "blocked", not "never going to be blocked anyway", which is exactly the distinction
§ 7.8 demanded.

**The 6-shape control passes at base and tip** — keep that, and assert it by value.

---

## 4. Merge order, and why it is not negotiable

```
C-094 (53eaf24)  ->  C-098a (cap, inert)  ->  C-098b (gate)
```

The cap arm is inert alone; the gate arm is not. **The reverse order leaves the integration branch
shipping a bare `pathway.pwml` at `release_ready` for an entity with no organism** — the state § 1
calls worse than the status quo. All three merge back to back as one integration step, with focused
tests after each and SMOKE after the last; the branch is not handed on part-merged.

---

## 5. Reachability — answered, and one answer changed the design

Recorded because it is the merge-rule-6 evidence and a reviewer will ask for it.

1. **Can the marker exist without C-094's fallback?** No. One writer in `src/` —
   `map_ids._record_placeholder_wrapper_species` (`:6841`), called from exactly `:7460`, `:7691`,
   `:8008`. No fixture, cache or committed artifact contains either string. The issue *name*
   `protein_complex_missing_species` has six other emitters and **none uses this reason**, which is
   why the predicate keys on the **reason**, not the name.
2. **Can a row lose its species by another route and still reach the demotion?** No — and this is
   the answer that matters. **The 6-shape also carries `placeholder_record_species`**, because C-094
   writes it on every Unknown-backed wrapper. So condition 1 alone is *provably insufficient*, and
   only the issue separates "nothing underneath" from "resolved underneath". A predicate keyed on the
   marker block alone would have demoted the 6-shape too. **Keep all three conditions.**
3. **Is it forgeable from a replayed `final_mapped.json`?** Conditions 1–2 live in `mapping_meta` and
   are forgeable. Condition 3 is not: a component must resolve **through the gate's own registry
   maps** to a protein satisfying `is_pathbank_unknown_protein` — id 9659, name `Unknown`, uniprot
   `Unknown`, fallback rule. A payload satisfying all three **is** an Unknown-backed wrapper whoever
   wrote it, so demoting it is correct rather than a relaxation. **Residual, stated rather than
   denied:** a hand-authored full sentinel protein row plus both meta keys is indistinguishable from
   the real thing by construction. That is acceptable and is recorded, not hidden.

---

## 6. Process

`C-098.md` § 8 still binds, per arm: `--task C-098a` / `--task C-098b` — **lowercase suffix,
corrected 2026-08-27: the allocator rejects `C-098A`, and the implementer's guard caught the
`ValueError` before its text could become a `--json` path**; `--expect-tree` matching the
worktree, `PYTHONPATH=<tree>/src`, `T2PW_OFFLINE_CURATOR=1`, `--basetemp` under a pre-created parent,
`--pin-verdict` on every pytest run, G11 allocator output guarded against anything containing `rror`,
`FINAL SURVIVING COUNT : 0` and `cleanup : success` confirmed after every job, wrapper stdout saved
immediately, reports and pin verdicts committed with the branch.

**No pytest obligations are discharged yet** — the work so far is probe-only. Each arm needs its own
focused tests, its affected-file sweep, and SMOKE.

**Two environment facts, both learned the expensive way this wave:**

* **Exported `PATHBANK_DB_*` cannot hide the database.** `src/t2pw/llm/client.py:22` calls
  `load_dotenv(dotenv_path=ENV_PATH, override=True)` and re-applies `.env` over exported values for
  any test that transitively imports the LLM client. Only physically renaming `.env` works — with a
  `trap … EXIT` to restore it. A reviewer voided two of its own jobs discovering this.
* **An agent worktree may have no `.env` at all**, which makes a database-dependent delta meaningless
  in the opposite direction. Check which state your tree is in and say so.

**The Chunk D gate cannot go green in this environment** at base or tip with the database up: `core`
carries a pre-existing `test_pwml_writer` failure and `qb` carries F-136. Classify any red against
`LEDGER.md` § F-136 before calling it a regression.

## 7. Report per arm

The `C-098.md` § 9 list, plus: for C-098a, the inertness measurement as committed tests and the
reason-string fix; for C-098b, the base-vs-tip end-to-end behaviour and the 6-shape control by value.

**Do not merge, do not push, do not touch `main` or `sprint/pwml-recovery`.**
