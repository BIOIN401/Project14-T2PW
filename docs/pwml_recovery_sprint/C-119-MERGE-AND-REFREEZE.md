# C-119 — MERGED, and production is RE-FROZEN.

**Lead Orchestrator, 2026-09-07.** Authorized by the product owner on the evidence of
`ORCH-728-DISPOSITION-RULE.md`. Independently reviewed as `REV-119`, two rounds.

---

# 1. The freeze

| | |
|---|---|
| **NEW FROZEN PRODUCTION SHA** | **`c7a0663eef43a1329f6e5c5acfb1f6dc3f2e6591`** |
| previous frozen SHA | `c4a97f60` (`D-090`, expired with this card as `D-095` did with `C-118`) |
| merge | `--no-ff` of `card/C-119-serialization-floor` @ `ce319e88`, base `6746a8d3` |
| `src/` files changed | **exactly two** — `batch/driver.py`, `pipeline/release_status.py` |
| post-merge SMOKE (gate G10) | **508 passed, exit 0**, `FINAL SURVIVING COUNT : 0`, `cleanup : success` |
| `main` | **untouched** |
| `src/t2pw/app/streamlit_app.py` | unmodified and uncommitted; both `D-097` pins hold — CRLF `47e4fafa…`, content `251122389a2d29e8…` |

**Production is re-frozen at `c7a0663e`.** No further production change is authorized.

---

# 2. What changed, in one paragraph

A leg whose **final, live** gates all pass is no longer destroyed because a contract report
the app itself stamps `phase: audit_round` — and documents as *"not a verdict about what
shipped"* — still carried errors. Such a leg now serializes as **`review_required`**, never
`release_ready`, and the superseded finding is preserved rather than dropped. **No gate was
weakened.** The serialization floor — F-179, the final live Stage-3 gate,
`reaction_enzyme_must_be_protein_complex`, the connected-core floor — is untouched and still
refuses first.

---

# 3. The three seams as merged

1. **`driver.py::_blocking_reports`** — phase-aware. Only `phase: audit_round` is excluded,
   and only when a later boundary has actually spoken (`post_audit` / `post_remap` contract
   report, or a `final_pre_export` Stage-3 gate report). **Absence of a later boundary still
   blocks.** An artifact set whose contract reports carry no phase keeps the pre-change
   arithmetic, so every archived run keeps its recorded verdict. A report at any other phase,
   and a report with no phase at all, block exactly as before.
2. **Batch finalization** — the superseded set is derived **once**, at the seam that decides
   whether the leg lives, and threaded to the export path, so the filename and the manifest
   row cannot disagree. The finding rides **three independent channels**: a review reason on
   the release record; a warning that survives an absent or uninterpretable frozen record; and
   a neutral count written *before* the blocking decision, so a leg that is **still refused**
   keeps the fact.
3. **`release_status.py::classify_release_status`** — a sixth cap of the same shape as the
   five that already existed. Only `release_ready → review_required`; exactly one step; never
   `diagnostic_only`; never creates `release_ready`; defaults to `None` so every existing
   caller is byte-identical.

---

# 4. Review record — `REV-119`, two rounds

**Round 1: `CORRECTION`.** R1–R5 all answered safe, each against a cited line. Boundary
confirmed to two `src/` files. Merge rules 6, 7 and 8 satisfied, rule 8 tested adversarially.
G9 base-failure proof **independently verified** to fail at `6746a8d3` on *value* assertions
(`2==0`, `7==0`, `'fail'=='pass'`) with the module importing cleanly — not on symbol absence.
SMOKE 508 at tip and base; **C-119 moves SMOKE by 0**.

One **BLOCKING** finding: a fixture app never set `final_payload`, so `run_one` died at the
`no_reactions` guard and C-119's own obligation — *a superseded count must survive on a
refusing path* — was unproven by the branch. **Test-harness defect, not an implementation
defect**; the reviewer proved the implementation correct by supplying the missing two lines.

**Corrections applied, test-only, `src/` byte-identical throughout** (verified by the reviewer
against sha256 `a1a6c582…` / `4d57b507…`):
- Finding 1 — the two prescribed lines, mirroring the G9 file.
- Finding 2 — a false G9 cross-reference corrected, since G9 labelling honesty is merge gate 9.

**Round 2: `APPROVE WITH FINDINGS`.** The reviewer did not merely re-run the test green; it
traced the corrected test through monkeypatched production seams and established the assertion
is **load-bearing**: `_finalize_pwml_export` is never called, the refusal comes through
`verdict.source='final_report'` at `final_pre_export_stage3_gates`, the count is already in
`outcome.counts` at the instant of refusal, and `grep` finds exactly **one** production write
site for it.

## Findings carried, not fixed — all judged safe to carry

| # | finding | disposition |
|---|---|---|
| **3** | **Seam 1 is mode-blind, so research legs change too.** A research leg previously filed `fail`/`contract` on a superseded snapshot can now be filed `pass` with a research report. It cannot produce a PWML (research returns before the export seam), cannot reach `release_ready`, and cannot move the strict denominator. **The number of research legs affected was NOT measured** — `ORCH-728` § 3 counts strict legs only, and a reader must not infer zero. | recorded, unmeasured, stated here |
| 4 | `_artifact_set_is_phase_stamped` uses a different discriminator than `gate_reports.is_current_artifact_set`. A hand-assembled set can serialize with no live final Stage-3 gate report — bounded to `review_required`, never `release_ready`. Unreachable in production: `streamlit_app.py:4392` stamps unconditionally. **Hardening judged NOT required** and out of this card's boundary. | LEDGER candidate for a future authorized `driver.py` touch |
| 5 | `" audit_round "` also excludes, via `_text`'s strip. Harmless, undocumented. | recorded |
| 6 | `test_no_archived_leg_becomes_release_ready` is **vacuous** — every fixture is already `review_required`. The property *is* proven, by `test_the_cap_can_never_create_release_ready` and by the reviewer's P6/P7 probes. **Do not cite the vacuous test as the proof.** | recorded |
| 7 | `PMC8510960` is not driven end to end; its § 4 row is carried at the record/filename level plus the G9 arm. | recorded |
| 8 | A bare-string carrier demotes `release_ready`. Deliberate for JSON round-trip; monotone-safe. | recorded |

---

# 5. What this does NOT establish

- **The biological content of the newly-serializable PWMLs has not been reviewed by anyone.**
  Merging ships two unreviewed pathways as `review_required`. `PILOT-MANUAL-REVIEW.md` remains
  the blocking item and this card does not touch it.
- The pilot denominator (*"4 of 6 evaluable strict legs"*) was not re-derived by the reviewer.
- Fresh-run behaviour of `PMC12452463` is unknowable without a run; its two archives disagree
  and both behave correctly under the patch.
- Chunk D and the benchmark suites were not run.
- `tests/test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_and_the_cure`
  fails in `C:/t/c119` because that worktree has no `.venv`. **Environmental, fails identically
  at the base SHA, outside this card.**

---

# 6. `STOP ENGINEERING.`

No further optimization wave. The remaining work is not engineering:

1. **Generate the PWMLs** deterministically from the archived canonical payloads.
2. **Manually review them** — `PILOT-MANUAL-REVIEW.md`.
3. **Manuscript analysis.**

---

*`main` untouched. Protected artifacts and all 216 worktrees preserved. No cache committed.*
