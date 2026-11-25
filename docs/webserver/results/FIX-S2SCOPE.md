# FIX-S2SCOPE: Stage 2 re-adds a reaction Stage 1 ruled out of scope

Branch `fix/stage2-respect-scope`, cut from `main` `5bf9d552`. Biology-path fix approved by the PO
("whatever makes it best") under the standing rule: never worsen text-to-PWML results; if no-loss
cannot be shown, the fix stays off. Code commits `ec611e4d` (code and tests) and `fbf8cbd7` (review F3: fail open).

**Status after review R1 (REJECT of the no-loss claim): the flag must NOT be enabled as is.** With the
chlorophyll gold's heme step matched, the rule deletes a curated gold reaction in all 5 scored
chlorophyll runs (sections 3.1 and 11). The code is inert with the flag off. This record and the
evidence (`docs/webserver/evidence/FIX-S2SCOPE/`) are in separate docs-only commits.

## 1. Symptom

`docs/pwml_output/ACCURACY.md` section 15, "Finding, outside the lane". In the enterobactin server job
(`C:/t/t2pw-server-data/work/item-4`, PMID 40400396), Stage 1 extracted the paper's assay "LDH reaction:
pyruvate + NADH -> lactate + NAD" and marked it `scope_membership: out_of_scope` (stage1 row 5).
`run_stage_one` drops it (`filter_out_of_scope_reactions`) before Stage 2, but Stage 2 (`pwml_infer`)
re-adds the same reaction (no scope, no lineage, enzyme `"provenance": "inferred"`, confidence 0.95)
and `merge_additions` merges it. It survives audit, gap fill and curator into `final_mapped.json`
(as "Pyruvic acid + NADH -> lactate + NAD+") and the PWML. item-5 is the same job shape.

## 2. The fix

`src/t2pw/service/extraction.py` (where the Stage-1 rows, with their scope labels, and the Stage-2
merge meet: `run_stage_two` holds `stage_one_result.stage_one`, the post-boundary, post-verifier
Stage-1 payload that still carries the out-of-scope rows, and the Stage-2 merge statement).

- Flag `T2PW_STAGE2_RESPECT_SCOPE`, read per call (`stage2_respect_scope_enabled`, modelled on
  `reaction_verifier.enabled`). **Default off: nothing new runs**, the merge gets the same object.
- On: `drop_stage_two_scope_repeats(stage_two, stage_one, context, verifier_report)` runs just
  before the Stage-2 `merge_additions` statement. The merge statement itself is unchanged (C-060a
  lifts it by AST). `stage_two` is rebound to the filtered envelope for that one statement and
  restored after it, so `state["stage_two"]` keeps the model's own output.
- Only Stage-2 reactions are filtered. Stage-1 rows (in scope or not), Stage-2 entities, RAG (its
  merge is the second site, untouched), audit and everything later are untouched.
- Nothing dropped: the function returns `stage_two` itself, so the merge input is byte for byte the
  same. Something dropped: it returns `clean_inference_output(deepcopy(stage_two))` with those rows
  removed (the same cleaning `merge_additions` applies first, so matching happens on the rows the
  merge would actually merge: composite tokens split, strings only).
- The record goes to `state["stage2_scope_drops"]` (each drop: the Stage-2 row, the Stage-1 row and
  its index, the out-of-scope reason, the matched keys) and `batch/driver.py` writes it into the run
  folder as `stage2_scope_drops.json`, next to `reaction_verifier_report.json` (the same state-key
  route). It is written whenever the flag is on and Stage 2 ran (an empty `drops` list says the check
  ran and found nothing); absent with the flag off, so the artifact set is unchanged.

### 2.1 The matching rule, and why

A Stage-2 reaction is dropped when **its step signature equals that of a Stage-1 row the extractor
marked `out_of_scope`, and of no Stage-1 row in scope.**

Step signature (`step_signature`): `(substrates, products)`, each a set of `name_keys.reference_key`
(production's one reference key, the exporter's; no new normaliser), with currency metabolites left
out. Currency = `cofactor_policy.classify_entity(name, PathwayContext).verdict ==
cofactor_or_currency`, in the run's own Stage-0 context (`pathway_context_from_stage_zero`, the same
factory the merge gate uses). A row made only of currency keeps all its names.

- **Lactate is caught.** pyruvate + NADH -> lactate + NAD has signature ({pyruvate}, {lactate}).
  Stage 2's copy matches whatever it does with the currency (NAD vs NAD+, an extra H+, NADPH).
- **A real step sharing only NADH/NAD+ with an assay is kept.** EntA (DHB-diol + NAD+ -> DHB + NADH)
  has signature ({2,3-dihydro-2,3-dihydroxybenzoate}, {2,3-dihydroxybenzoate}): no overlap with the
  assay's. Currency never decides a match; the non-currency substrates AND products must both be the
  same sets.
- **Currency in the policy, not a new list.** `cofactor_policy` is curated (ATP/ADP/AMP, NAD(P)(H),
  FAD, H2O, H+, O2, CO2, H2O2, NH3, Pi, PPi) and context-aware: in a pathway that is about NAD,
  ATP etc. the policy calls them participants, so they count and the match gets stricter, never
  looser. Hubs (CoA, SAM, glutamate, 2-oxoglutarate) are not currency here, so they count too. The
  verifier's own `_currency` list (COFACTOR_NAMES plus extras incl. CoA and SAM) is broader, and so
  looser; it was not used.
- **Same direction only.** lactate + NAD+ -> pyruvate + NADH is not a repeat of the assay: a pathway
  can run the reverse of a reaction the paper uses elsewhere.
- **Stage 1 holding the step in scope too is no verdict.** If any in-scope Stage-1 row has the same
  signature, nothing is dropped (and `_merge_reactions` would otherwise fold the Stage-2 copy's
  enzymes into that row; dropping it would lose them).
- **Only the extractor's verdict counts; rows the reaction verifier omitted or folded are ignored.**
  The verifier marks every row it omits (summary, split piece) or folds (duplicate) as
  `out_of_scope`. Those are not scope verdicts, and the first replay showed Stage 2 restoring a gold
  step the verifier had wrongly omitted (section 3.2). A Stage-1 row is ignored when
  `reaction_verifier_report` lists its index with any action other than kept, repaired or flagged.
  Rows already out of scope before the verifier are not in its report, so they still count.

## 3. Offline replay (the no-loss proof)

Script `evidence/FIX-S2SCOPE/scripts/replay.py`, read-only over every stored run with both
`stage1_payload.json` and `merged_payload.json`: `C:/t/acc-runs/legs*/**/papers/*/strict/` and the
server work items (`C:/t/t2pw-server-data/work/item-*/attempt-1/out`). Run through the queue
(`gates`-style wrapper, 0 surviving processes).

Upper-bound construction: the raw Stage-2 output is not stored, so the candidate additions are every
merged row whose inputs/outputs (merge's own `_reaction_io_key`) match no in-scope Stage-1 row (this
includes any RAG rows and Stage-1 rows the merge rewrote). The filter
runs with a neutral context ("requested pathway": no subject escapes, so the most currency and the
loosest match), and again with the run's own scope for the enterobactin runs (same result).

Gold: (a) every dropped row scored alone with `accscore.score_payload` (which gold reactions it
matches), and (b) the run's `final_mapped.json` scored with and without the rows carrying a dropped
signature (gold lost = the difference; final rows are matched with the evaluator's `strict_key`,
because mapping renames "pyruvate" to "Pyruvic acid").

### 3.1 Final rule, heme matched (`replay_heme.json`): gold lost 5

Scored with `scripts/accscore_heme.py`, a copy of `C:/t/acc-runs/scripts/accscore.py` (the original
is unchanged) that adds "Protoheme" and "Protoheme IX" to Heme, and "Fe 2", "Fe2", "Fe(2+)" and
"ferrous iron" to Fe2+, in PW_R2146849 (ferrochelatase: Protoporphyrin IX + Fe2+ -> H+ + Heme,
CHEBI:17627; protoheme IX is heme b). The original scorer missed that gold step because "protoheme"
is not a gold alias and Stage-2 cleaning splits "Fe2+" into "Fe 2" (review F1).

runs scanned 74, with an out-of-scope Stage-1 row 23, drops 16, dropped rows matching gold 6, gold lost 5

| paper | run | Stage-2 row dropped | Stage-1 row (index) | gold matched by dropped row | final_mapped gold lost | precision before -> after |
|---|---|---|---|---|---|---|
| PMC5130987 | legs-trna/off2-PMC5130987/2026-11-24_2259 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (12) | ['PW_R2146849'] | ['PW_R2146849'] | 0.4375 -> 0.4 |
| PMC5130987 | legs-trna/on1-PMC5130987/2026-11-24_2051 | protoporphyrin IX -> protoheme | protoporphyrin IX -> protoheme (15) | [] | [] | 0.4118 -> 0.4375 |
| PMC5130987 | legs4g/base1-PMC5130987/2026-11-23_1538 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (8) | ['PW_R2146849'] | no final_mapped | None -> None |
| PMC12096016 | legs4g/base2-PMC12096016/2026-11-23_1906 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.5714 -> 0.6667 |
| PMC5130987 | legs4g/base2-PMC5130987/2026-11-23_2100 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (18) | ['PW_R2146849'] | ['PW_R2146849'] | 0.4444 -> 0.4118 |
| PMC5130987 | legs4g/base3-PMC5130987/2026-11-24_0229 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (16) | ['PW_R2146849'] | ['PW_R2146849'] | 0.5 -> 0.4667 |
| PMC12096016 | legs4g/vt1-PMC12096016/2026-11-23_1322 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | no final_mapped | None -> None |
| PMC12096016 | legs4g/vt2-PMC12096016/2026-11-23_1919 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.5 -> 0.6 |
| PMC5130987 | legs4g/vt2-PMC5130987/2026-11-23_2211 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (19) | ['PW_R2146849'] | ['PW_R2146849'] | 0.3889 -> 0.3529 |
| PMC12096016 | legs4g/vt3-PMC12096016/2026-11-24_0016 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (9) | [] | [] | 0.8571 -> 1.0 |
| PMC5130987 | legs4g/vt3-PMC5130987/2026-11-24_1052 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (7) | ['PW_R2146849'] | ['PW_R2146849'] | 0.5 -> 0.4706 |
| PMC12096016 | legs5g/base1-PMC12096016/2026-11-24_1237 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.6667 -> 0.75 |
| PMC12096016 | legs5g/base3-PMC12096016/2026-11-24_1413 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.6667 -> 0.8 |
| PMC12096016 | legs5g/vt1-PMC12096016/2026-11-24_1250 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (8) | [] | [] | 0.75 -> 0.8571 |
| PMC12096016 | server item-4/attempt-1/out | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.75 -> 0.8571 |
| PMC12096016 | server item-5/attempt-1/out | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (6) | [] | [] | 0.8333 -> 1.0 |

The 9 LDH drops are still clean (no gold, precision up). 6 of the 7 ferrochelatase drops match gold
PW_R2146849 (the seventh has no Fe2+ and matches nothing). In all 5 runs with a `final_mapped.json`
that step is lost and precision falls. Log: `review1_replay_heme_oosdump.log`.

### 3.1a Same rule, original scorer (`replay.json`): WRONG, kept for the record

runs scanned 74, with an out-of-scope Stage-1 row 23, drops 16, dropped rows matching gold 0, gold lost 0

| paper | run | Stage-2 row dropped | Stage-1 row (index) | gold matched by dropped row | final_mapped gold lost | precision before -> after |
|---|---|---|---|---|---|---|
| PMC5130987 | legs-trna/off2-PMC5130987/2026-11-24_2259 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (12) | [] | [] | 0.375 -> 0.4 |
| PMC5130987 | legs-trna/on1-PMC5130987/2026-11-24_2051 | protoporphyrin IX -> protoheme | protoporphyrin IX -> protoheme (15) | [] | [] | 0.4118 -> 0.4375 |
| PMC5130987 | legs4g/base1-PMC5130987/2026-11-23_1538 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (8) | [] | no final_mapped | None -> None |
| PMC12096016 | legs4g/base2-PMC12096016/2026-11-23_1906 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.5714 -> 0.6667 |
| PMC5130987 | legs4g/base2-PMC5130987/2026-11-23_2100 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (18) | [] | [] | 0.3889 -> 0.4118 |
| PMC5130987 | legs4g/base3-PMC5130987/2026-11-24_0229 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (16) | [] | [] | 0.4375 -> 0.4667 |
| PMC12096016 | legs4g/vt1-PMC12096016/2026-11-23_1322 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | no final_mapped | None -> None |
| PMC12096016 | legs4g/vt2-PMC12096016/2026-11-23_1919 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.5 -> 0.6 |
| PMC5130987 | legs4g/vt2-PMC5130987/2026-11-23_2211 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (19) | [] | [] | 0.3333 -> 0.3529 |
| PMC12096016 | legs4g/vt3-PMC12096016/2026-11-24_0016 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (9) | [] | [] | 0.8571 -> 1.0 |
| PMC5130987 | legs4g/vt3-PMC5130987/2026-11-24_1052 | protoporphyrin IX + Fe 2 -> protoheme | protoporphyrin IX + Fe 2 -> protoheme (7) | [] | [] | 0.4444 -> 0.4706 |
| PMC12096016 | legs5g/base1-PMC12096016/2026-11-24_1237 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.6667 -> 0.75 |
| PMC12096016 | legs5g/base3-PMC12096016/2026-11-24_1413 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.6667 -> 0.8 |
| PMC12096016 | legs5g/vt1-PMC12096016/2026-11-24_1250 | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (8) | [] | [] | 0.75 -> 0.8571 |
| PMC12096016 | server item-4/attempt-1/out | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (5) | [] | [] | 0.75 -> 0.8571 |
| PMC12096016 | server item-5/attempt-1/out | pyruvate + NADH -> lactate + NAD | pyruvate + NADH -> lactate + NAD (6) | [] | [] | 0.8333 -> 1.0 |

Every drop is the LDH assay (enterobactin, 9 runs, incl. server items 4 and 5) or ferrochelatase's protoporphyrin IX + Fe2+ -> protoheme (chlorophyll paper PMC5130987, 7 runs). **This table's "gold lost 0" is wrong:** the heme step IS in the chlorophyll gold (PW_R2146849) and this scorer could not match it (section 3.1). 74 runs includes the other session's legs-trna legs and this task's two A/B legs, all read-only. Log: `replay_final.log`.

### 3.2 First rule, without the verifier exclusion (`replay_rule1_no_verifier_exclusion.json`, dead end)

70 runs scanned, 19 with an out-of-scope Stage-1 row, 20 drops, **1 dropped row matching gold**, gold
lost 0 (original scorer; the same ferrochelatase drops are gold, section 3.1). The five extra drops were all verifier omissions:

| run | Stage-2 row | verifier action on the Stage-1 row | gold |
|---|---|---|---|
| legs4g vt3-PMC12096016 | 2,3-dihydroxybenzoyl-AMP + L-serine -> enterobactin | omitted, "summary" | **R5** |
| legs4 vt1-PMC12312563 | 2-oxoglutarate -> intermediate I; intermediate I + isochorismate -> intermediate II; intermediate II -> SEPHCHC | split pieces | none (precision 0.25 -> 0.5) |
| legs5g vt3-PMC12312563 | 2-oxoglutarate -> intermediate I; intermediate I + isochorismate -> intermediate II | split pieces | none (precision 0.33 -> 0.5) |

The R5 case: the verifier omitted the true EntF step as a "summary" and kept the wrong row
"2,3-DHB + L-serine -> enterobactin" (flagged not_stated); Stage 2 restored the true step. Gold was not
lost in that run only because the kept wrong row also matched R5 in `final_mapped`. Under the hard
rule that is not a no-loss proof, so verifier omissions were excluded (section 2.1). Cost of the
exclusion: the generic-intermediate split pieces Stage 2 re-adds after the verifier merged them are
no longer dropped (a precision gain given up on PMC12312563).

## 4. Tests

`tests/test_fix_s2scope.py` (11 tests; the 11th, fail-open, is section 11): no out-of-scope rows -> nothing dropped and the same object
back; the lactate shape dropped and recorded (NAD+ vs NAD, extra H+); a real step sharing only
NADH/NAD+ with the assay kept (EntA, and pyruvate + NADH -> malate + NAD+); a step Stage 1 also holds
in scope kept; the reverse direction kept; a verifier-omitted or -folded row ignored, and the same row
respected without the verifier record; the real `merge_additions` on the filtered envelope has no
lactate row; `run_stage_two` with the flag off merges exactly what Stage 2 returned and never calls the
filter; with the flag on, the merge gets the filtered rows and `state["stage_two"]` stays raw; the
driver writes `stage2_scope_drops.json` only when the state key is present.

## 5. Gates (code `ec611e4d`, one queue hold, `evidence/FIX-S2SCOPE/gates.log`)

Every test file importing `service/extraction.py` or `batch/driver.py`, plus the C-060a AST test,
minus the files suite2 runs; AppTest files one per process:

| group | result |
|---|---|
| plain (25 files incl. `test_fix_s2scope.py`, `test_c060a_requested_pathway_wiring.py`, `test_reaction_verifier.py`) | 669 passed |
| 15 AppTest files, one process each | all passed except `test_batch_preflight.py`: 1 failed, 32 passed, 4 skipped |
| suite2: SMOKE | 511 passed |
| golden-seam batch / c011 / stage8 / quarantine (none edited) | 2 / 8 / 4 / 24 passed |
| equivalence run_pathway / app-vs-service / default | 8 / 14 / 6 passed |
| vision | 370 passed, 1 skipped |
| web | 83 passed |

Wrapper: 477 s, 0 surviving processes. The `test_batch_preflight.py` failure is environmental and
pre-existing: `test_the_message_names_the_problem_the_interpreter_and_the_cure` asserts "this project
ships a .venv", and a worktree has none. It fails identically on `C:/t/fix-noname` (`767c6e66`,
without this change) (`evidence/FIX-S2SCOPE/preflight_env_failure.log`).

## 6. Live A/B (Gemma)

Enterobactin, PMID 40400396 / PMC12096016, `T2PW_LLM_MODEL=google/gemma-4-31b-it`,
`T2PW_REACTION_VERIFIER=1` in both arms, other lane flags unset, code `ec611e4d`, through the queue
with the heavy lock (`evidence/FIX-S2SCOPE/scripts/leg.sh`, `driver.sh`; batch default engine, i.e.
`service.pathway.run_pathway`, the web worker's run function). Scored with `accscore.py`
(`scripts/score_ab.py`, `ab_score.json`).

| leg | flag | status | PWML | gold | rows | precision | Stage 1 marked lactate out of scope | lactate row in final_mapped | `stage2_scope_drops.json` | wrapper | survivors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off1 | off | pass | yes | 5/5 | 8 | 0.750 | no | **yes** ("Pyruvic acid + NADH -> lactate + NAD+", an FP) | absent (flag off) | 691 s | 0 |
| on1 | on | pass | yes | 5/5 | 8 | **0.875** | yes (row 8) | **no** | 1 drop: pyruvate + NADH -> lactate + NAD, matched ({pyruvate}, {lactate}) | 588 s | 0 |

**Incomplete: 1 + 1 of the planned 3 + 3 legs.** Before off2 the driver measured 4.8 GB free RAM
(kernel process objects 152,105, up from 130,729 at off1: the kernel zombie-process leak), under its
5 GB floor, created `C:/t/acc-runs/PAUSE`, logged PAUSED_LOW_RAM and stopped
(`ab/mem.log`, `ab/driver.out`). The other live session's driver stopped on low RAM at the same time.
I removed the PAUSE file my driver had created (logged in `ab/mem.log`). The driver is resume-safe:
after a reboot, re-running `driver.sh` skips off1/on1 (`.end` files) and runs off2, on2, off3, on3.

Read: the flag removes the lactate row where Stage 1 marked it out of scope, with no gold change
(5/5 both arms) and one false positive fewer. In off1 Stage 1 did not extract the LDH row at all
(its only out-of-scope row is a verifier fold), so Stage 2's lactate row there has nothing to match:
the flag catches the re-addition only when Stage 1 ruled on it, as designed. The orphan "lactate"
compound stays in on1's `final_mapped.json` entities but is not in its PWML (Q1).

## 7. Web server jobs

Yes, `stage2_scope_drops.json` lands in a web job's `out/` folder with no further change. The web
worker's run function is `service.pathway.run_pathway` (`web/worker.py:70`, `DEFAULT_RUN_FN`). It runs
`run_extraction` with `at = SimpleNamespace(session_state=state)` (`service/pathway.py:243`) and collects
artifacts with the same `driver._add_common_artifacts(at, ...)` (`service/pathway.py:296-384`), which is
where the new four lines are. `driver_bridge._to_result` then writes `outcome.artifacts` into `out_dir`
(`runner.write_artifacts`, `driver_bridge.py:243`). That is the route `reaction_verifier_report.json`
takes into the server job folders (item-4 has it). The A/B legs ran the batch default engine, which is
this same `run_pathway`, and on1's run folder has the file. The worker's child inherits the server
environment, so the flag is switched on by setting `T2PW_STAGE2_RESPECT_SCOPE=1` where
`T2PW_REACTION_VERIFIER=1` is set today.

## 8. Dead ends

- First rule counted every `out_of_scope` Stage-1 row, including the verifier's omissions: one
  dropped row was a gold step (section 3.2). Fixed by ignoring verifier-removed rows.
- First test draft expected "NAD+" to survive Stage-2 cleaning: `_split_composite_token` splits on
  "+", so Stage-2 "NAD+" reaches the merge as "NAD" (existing behaviour, not changed; currency is
  ignored by the rule anyway). The real-merge test also needed an entity registry, because
  `filter_unresolvable_reactions` removes rows whose sides name no entity.
- The first replay's final_mapped check matched rows by `reference_key`, which missed the lactate row
  there ("Pyruvic acid" after mapping); the replay now uses the evaluator's `strict_key` for that one
  check (replay-side only).

## 9. Open questions

- **Q1 Entities.** Only the reaction is dropped. The entities Stage 2 added with it (lactate) stay,
  as the Stage-1 out-of-scope filter already leaves the entities of the rows it removes. In on1 the
  orphan "lactate" is in `final_mapped.json` but not in the PWML. Options: leave it (no cost); drop
  Stage-2 entities named only by dropped rows (small, but touches entity merging).
- **Q2 Verifier split pieces.** With the verifier on, Stage 2 can re-add the generic-intermediate
  split pieces the verifier merged into a whole step (PMC12312563, two runs). Dropping them would
  have raised precision there, but the same mechanism restored a gold step the verifier wrongly
  omitted as a summary. Options: keep as is (safe); drop only re-added rows of verifier `split`
  omissions (needs its own no-loss replay).
- **Q3 A/B completion.** 2 of 6 legs ran before low RAM stopped the driver. Moot until the rule is
  settled (section 11.5).
- **Q4 Narrowing (review F2).** No structured field separates an assay from a branch (section 11.3).
  Options: (a) keep the flag off and accept the LDH row (status quo, no cost); (b) add an `assay` scope
  label to the Stage-1 prompt and drop only Stage-2 repeats of those rows (a prompt change; needs its
  own replay and A/B); (c) the `biological_state`-absent correlate (8 of 9 LDH drops, 0 gold in this
  replay, but no basis for generalising); (d) apply the rule only when an explicit `<pathway_scope>` is
  given, where the prompt does not fold branches into `out_of_scope` (needs a replay over scoped runs;
  none is stored).

## 10. Recommendation

**Do not enable `T2PW_STAGE2_RESPECT_SCOPE` as is.** It removes the LDH assay cleanly, but it also
removes the ferrochelatase step, a curated gold reaction of the chlorophyll pathway, in every
chlorophyll run where Stage 2 re-added it (5 of 5 scored runs lose PW_R2146849). That breaks the
no-loss rule. The code is safe to merge default-off (inert, fails open). Section 11.3 covers whether a
narrower rule exists.

## 11. Review R1 (REJECT of the no-loss claim) and response

- **F1 (blocking).** The 7 protoporphyrin IX + Fe2+ -> protoheme drops are the gold step PW_R2146849;
  accscore missed it ("protoheme" is not an alias; "Fe2+" is cleaned to "Fe 2"). Confirmed: section 3.1,
  gold lost 5 with heme matched. The earlier "gold lost 0" and the recommendation to enable are withdrawn.
- **F2 (blocking).** `pwml_system.txt:15` tells Stage 1 to mark biochemically adjacent and branch steps
  `out_of_scope` when no `<pathway_scope>` is given, and Stage 2 currently restores them. So
  `out_of_scope` covers both "assay in the Methods" (LDH) and "adjacent branch" (ferrochelatase), and the
  rule cannot tell them apart. Agreed; section 11.3 looks for a field that can.
- **F3 (should-fix).** Done in `fbf8cbd7`: `drop_stage_two_scope_repeats` wraps the check; any
  exception keeps `stage_two` unchanged, logs a warning and records `error` in the report. Test:
  `test_an_error_in_the_check_keeps_every_stage_two_row`. Gates: section 11.4.

### 11.3 Is there a structured signal that separates an assay from a branch? (data only)

`scripts/oosdump.py` dumped every `out_of_scope` Stage-1 row in the 74 stored runs (`oos_rows.json`,
38 rows: 16 removed by the reaction verifier, 22 marked by the extractor). After cleaning, a Stage-1 row
carries only `inputs`, `outputs`, `name`, `enzymes`, `biological_state`, `evidence`, `scope_membership`
and `provenance_lineage` (plus `inference` or `locked_reaction_id` when set). The extractor-marked rows,
grouped (`oos_rows_summary.txt`):

| paper | step (Stage-1 name) | rows | scope_membership | biological_state | enzyme provenance / confidence | lineage (stage / origin / explicit / support) | locked id | kind |
|---|---|---|---|---|---|---|---|---|
| PMC12096016 | LDH reaction: pyruvate + NADH -> lactate + NAD | 8 | out_of_scope | **absent** | extracted / 1.0 | paper_extraction / paper_stated / explicit / unsupported | none | assay |
| PMC12096016 | LDH reaction (same) | 2 | out_of_scope | cytosolic state | extracted / 1.0 | same | none | assay |
| PMC5130987 | ferrochelatase reaction: protoporphyrin IX (+ Fe 2) -> protoheme | 6 | out_of_scope | plastid state | extracted / 1.0 | same | none | branch (gold) |
| PMC5130987 | Proto IX to protoheme | 3 | out_of_scope | plastid state | extracted / 1.0 | same | none | branch (gold) |
| PMC12312563 | MenD step (2-oxoglutarate + isochorismate -> SEPHCHC; 2-oxoglutarate -> intermediate I) | 2 | out_of_scope | cytosolic state | extracted / 1.0 | audit_repair / audit_modified / explicit / direct | none | pathway step |
| PMC7059786 | ATP release from Type II cells: ATP -> ATP | 1 | out_of_scope | Type II cell state | none | paper_extraction / paper_stated / explicit / unsupported | none | other |

**No clean structured signal.** `scope_membership` has one value for both (`out_of_scope`; the prompt
has no "assay" label). Enzyme provenance and confidence, lineage and the absent lock id are identical
for the assay and the branch. The only field that differs is `biological_state`: absent on 8 of 10
LDH rows, present on every branch row. That is a correlate, not a signal: 2 of 10 LDH rows have
"cytosolic state", all 10 come from one assay in one paper, and nothing in the prompt makes an absent
state mean "measurement". For the record (`drops_by_biological_state.txt`), a rule narrowed to "the
Stage-1 row has no biological_state" would drop 8 of the 9 LDH rows and none of the ferrochelatase rows
in this replay (gold lost 0 with heme matched). With 10 assay rows from one paper that is no evidence it
generalises, so I do not propose it. A structured way to tell an assay apart would have to be new: a
distinct label in the Stage-1 prompt (for example `assay`, which is a prompt change), or the source
section of the evidence, which the cleaned Stage-1 row does not keep.

### 11.4 Gates on `fbf8cbd7`

One queue hold, 460 s, 0 surviving processes (`gates2.log`). Plain group 670 passed (669 + the
fail-open test); the 15 AppTest files as before (all pass except the environmental
`test_batch_preflight.py` .venv assertion, section 5); SMOKE 511; golden-seam 2/8/4/24 (none edited);
equivalence 8/14/6; vision 370 + 1 skipped; web 83.

### 11.5 Live legs

Not resumed, per the coordinator: they decide nothing until the rule is settled. 2 of 6 ran (section 6).
