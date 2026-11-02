# ORCH-799 — typed single-source stabilization pass · RESULT

Branch `orch799/typed-core`, cut from `sprint/pwml-recovery` @ `b25327c1`. Sub-orchestrated; the lead merges.
Goal: stop the recurring "two components decide the same thing over untyped dicts" bug class by
single-sourcing duplicated decisions in plain typed Python. Not a rewrite, no new features, no new
dependencies. Mid-task the product owner added a hard constraint: **text-to-PWML results must not get worse.**

## 1. What was done

| step | outcome | artifact |
|---|---|---|
| Research (sonnet, read-only) | 7 findings triaged at `b25327c1`, ranked, 3 lanes proposed | `ORCH-799-RESEARCH.md` |
| Lane selection | **ORCH-799a only.** 799b (F-209 leftover) and 799c (actor-schema twin) need rulings; see § 5 | this doc |
| ORCH-799a implement | `19d45f32` code, `77be8327` G11 evidence | see § 2 |
| Test run | SMOKE 511 = baseline; corpus per-leg identical; base arm passes | § 3 |
| Review (opus, read-only) | **APPROVE WITH NOTES**, nothing blocking | § 4 |

## 2. ORCH-799a — one "is this row referenced" predicate (F-208 row 1, quarantine half)

`process_normalizer.row_is_referenced(row, referenced_norms: AbstractSet[str]) -> bool`
= `bool((_entity_name_norms([row]) & referenced_norms) - {""})`, defined next to `_entity_name_norms`
(`process_normalizer.py:651`). The four inline copies in `strict_quarantine.py` now call it:
`_prune_entities` (~2246), `_degree_zero_exports` (~2888 and ~2904, the latter as two calls for
`referenced` and `exempt`), `quarantine_and_close` (~3654). The normalizer orphan pass and the final-gate
degree test are **not** touched (that is the behaviour half of F-208 row 1; § 5 Q1). `QUARANTINE_POLICY_VERSION`
not bumped (verdict-neutral).

**Equivalence argument** (implementer's, verified line by line by the reviewer): sites 2246 and 2904 already
filtered `""`. Sites 2888 and 3654 did not, but their set `referenced` comes from `_referenced_entity_norms`,
which does `found.discard("")` (`strict_quarantine.py:2184`); `exempt` comes from `_complex_component_norms`,
which only adds non-empty norms (~2205); `bucket_keep` = `referenced | component_norms` (3664-3666). So `""`
can never match at any site, and subtracting it changes nothing. `_normalize` *can* return `""` (e.g. name `-`),
which is why the reviewer traced every set rather than accepting the claim.

**Proof type:** pure refactor, so equivalence, not G9. New test `tests/test_orch799a_row_is_referenced.py` is
labelled NEW ACCEPTANCE TEST (no base failure claimed). Its site-level case was run on the base tree and passes
there too (§ 3).

## 3. Measurements (all through `bounded_run.py --heavy-lock`, 0 survivors on every job)

| gate | result | evidence |
|---|---|---|
| Focused (new test + 7 quarantine suites) | 330 passed (count from implementer's piped output; the report holds no stdout) | `evidence/g11/ORCH-799a/03-focused.json`, `pin/ORCH-799a/03-*` |
| SMOKE (22 files) | **511 passed** in 42.4 s = baseline 511 (`g11/ORCH-797/06`) | `evidence/g11/ORCH-799/02-smoke.json` (01 identical run, count not captured) |
| Corpus run at tip | 79/91 | `evidence/g11/ORCH-799/03-corpus-tip.json`; `evidence/orch799/corpus_tip.{json,txt}` |
| Corpus compare vs `orch797/corpus_tip.json` | 79/91 → 79/91, no leg broke, no gold recall lost, no failing leg changed step; replay records identical in **all 91 legs** (pass/fail, coverage, gold recall, verdict codes). Exit 1 / "D-112 § 2 REJECT" only because D-112 requires a gain, which a refactor cannot give | `04`, `05` (05 has the captured output) |
| Per-leg identity (orchestrator check) | `evidence/orch799/corpus_tip.json` is **byte-identical** to `evidence/orch797/corpus_tip.json` (`diff` empty, both 326954 bytes); the `.txt` differs only in the `t2pw resolved:` path line | `diff` of the two files |
| Base arm, equivalence | site-level case of the new test: 1 passed on `b25327c1` (PYTHONPATH=base/src, `--cwd` base). Base tree left clean | `evidence/g11/ORCH-799/06-base-arm.json`; test copy `evidence/orch799/test_orch799a_base_arm.py` |

The PO's no-regression constraint is therefore met in the strongest form available: the corpus replay output is
unchanged byte for byte.

## 4. Reviewer verdict: APPROVE WITH NOTES

1. Equivalence holds at all four sites (traced as in § 2).
2. Boundary clean: the code commit changes only the import + new function in `process_normalizer.py`, the import + four sites in `strict_quarantine.py`, and the new test.
3. Note: the new test's payload has `protein_complexes: []`, so sites 2888 and 3654 (`quarantine_and_close`) are not exercised by it; they are covered only by the existing quarantine suites in the focused run and SMOKE.
4. Note (G11 rule 9, `TEST_MATRIX.md:91`, "every job's --json report … is committed with the branch"): the implementer deleted a non-compliant ORCH-799a `02` report and an `01` pin and re-ran as `03`. The deleted `02` came from a job that actually ran, so the letter of rule 9 is breached; nothing committed points at a missing file. Recorded here (§ 6.4) as the explanation the reviewer asked for.
5. No gate weakened, `review_required` unaffected, no exporter touched.

## 5. Lead dispositions (2026-11-02, under the PO's no-regression instruction)

All four are **deferred — nothing to build** in this sprint pass. Options and costs are kept so the PO can act.

**Q1. F-208 row 1, normalizer half.** The orphan pass (`process_normalizer.py` ~2278) and the final-gate degree
test (~5110-5118) match on the primary name only; the quarantine (now via `row_is_referenced`) matches name ∪ synonyms.
Deferred because it is a gate change that could alter the export/refuse balance (merge rule 6; D-106).
- (a) leave as is — cost: the disagreement stays; a synonym-only reference can be kept by quarantine and dropped/refused by the gate.
- (b) synonym-aware gate (call `row_is_referenced`; gate needs synonym-aware edge resolution) — ~25-40 lines; changes which proteins are dropped/refused; needs a D-entry and a corpus A/B.
- (c) name-only quarantine — small, but narrows what the quarantine keeps; likely loses PWMLs.

**Q2. F-209 leftover.** `add .../synonyms/-` on a missing key creates a dict (`apply_audit_patch._resolve_parent:63`, `_set_value:84`).
Deferred because the curation stage is invisible to the corpus (F-210), so a no-loss result cannot be proven.
- (a) reject the op — ~10 lines; a patch that relied on it fails loudly.
- (b) create a list — ~10 lines; silently does what the author meant.
- (c) leave — cost: a malformed `synonyms` dict can reach downstream readers.

**Q3. Actor-schema twin check.** `actor_schema_not_canonical` is decided in `payload_models.py:664-697` and
`stage_contracts.py:792-815`; they differ on a non-dict actor (payload_models skips it, stage_contracts errors).
This is the F-207 class (raw Stage-1 actor shape). Deferred because unifying makes the runtime validator stricter,
which risks refusing pathways that export today.
- (a) single-source on the stricter `stage_contracts` rule — ~12 lines; first prove by corpus that no leg carries a non-dict actor.
- (b) single-source on the lenient rule — weakens a contract check; not recommended.
- (c) leave — cost: the two validators keep drifting.

**Q4. F-212, F-218, F-207.** Each needs a PO ruling or touches protected `streamlit_app.py`.
- F-212: Stage-6 `_name_gate_verdict` (`map_ids.py:6152`) vs `db_resolver` rule `exact_short_name_or_synonym` (`db_resolver.py:364`); ~30-50 lines; needs a D-104 AMENDMENT 2 ruling.
- F-218: `writer.py:1311` is the only `pwp-id` writer; it is a policy choice (what id to write), not drift; needs a ruling.
- F-207: fix site `streamlit_app.py:3620/3656`; needs a D-entry. F-214 is a new feature, not drift; out of scope.

## 6. Failures, dead ends and incidents

1. **Hang 1 — `cd` compound command.** The implementer's focused `bounded_run` call began `cd <worktree> && …`; a `cd` inside a compound command raises a permission prompt, which unattended hangs forever. It sat 25 min; no Python process ever started. The lead stopped it; it was resumed with the rule: never `cd`, absolute paths and `git -C` only, wrapper `--cwd` for the working directory.
2. **Hang 2 — heredoc write into another worktree.** The first test runner wrote its base-arm file into the base worktree with a heredoc; that raised a permission prompt and hung. The lead stopped it, wrote the file to the scratchpad, and resumed it; the base arm then ran from the scratchpad with `--cwd` base, so nothing was written to the base tree.
3. **Hang 3 / waste — duplicate dispatch.** While runner 1 was stopped, I dispatched a second runner for SMOKE count + base arm; it duplicated runner 1's finished work and the lead stopped it before it ran anything. Lesson: when a child is interrupted, ask the lead for its state before re-dispatching.
4. **Stray `python -`.** The implementer ran one `python -` (forbidden) by mistake; it stopped it, no orphan process.
5. **Deleted evidence.** The implementer deleted a non-compliant ORCH-799a `02` report and an `01` pin and re-ran as `03`. This breaches the letter of G11 rule 9 (see § 4 note 4). What was deleted is not recoverable; the `03` run is the compliant record of the same selection.
6. **Smoke count.** Report `01-smoke` ran green but its output was truncated by `tail` before the pytest summary, so `02` was run for the count. Both are committed.
7. **Research correction.** `FINDINGS.md` F-209 said OPEN; its main defect was fixed by `2a791f69` (ORCH-769d). Heading corrected in this commit; only the `/-` leftover stays open (Q2).

## 7. Not settled

- Q1-Q4 above (PO).
- No test exercises `quarantine_and_close`'s keep set on a synonym-only complex reference (reviewer note 3); a test-only follow-up could add one.
- G11 rule 9 breach in § 6.5: the PO/lead decide whether it needs anything beyond this record.

No `TEST_MATRIX.md` change: SMOKE stays 511 (the new test file is not a SMOKE file).
