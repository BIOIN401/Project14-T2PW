# FIX-TRNA: the audit counts a nucleic acid as declared (Bug B), flag-gated

Branch `fix/audit-nucleic-acid-declared`, cut from `main` 5bf9d552. PO-approved, flag-gated, default off.
Code commits `69e9984c` (change and tests) and `66b8ca26` (review F5 tests). This record is a separate
docs-only commit. Evidence: `docs/webserver/evidence/FIX-TRNA/`.

STATUS (2026-11-25, closed): **merged to main `3e098725`, default off, and NOT enabled on the server.** The
decisive replay (section 5) shows the flag does not fix the crash it was written for: flag on fails with the
same `AMBIGUOUS_REFERENCE`. The live A/B (section 6, 6 of 6 legs) shows no harm and no effect: no leg
declared the tRNA as a nucleic acid. The real cause is in prefreeze (section 5), a biology module, so it is
an open question for the PO (section 8).

## 1. Symptom (from FIX-NONAME section 6)

Chlorophyll leg `legs4g/base1-PMC5130987` (flags off) crashed in `t2pw.pwml.prefreeze_resolution`
with `AMBIGUOUS_REFERENCE: the name 'tRNA Glu' is also carried by ['nucleic_acids#0']`. The merged
payload declares `tRNA Glu` only as a nucleic acid; reaction 20 ("Glu to Glu-tRNA Glu") uses it as an
input. The deterministic audit counted a reference as declared only if it matched a compound, element
collection, protein or complex, so it added a duplicate compound row `tRNA Glu`, and prefreeze could not
redirect the name.

## 2. Change

`src/t2pw/curation/audit_json_llm.py`, `_deterministic_audit`: with `T2PW_AUDIT_NA_DECLARED` in
`1`/`true`/`yes` (read per call, same parsing as `T2PW_REACTION_VERIFIER`, default off), the set of
declared compound-like names also includes `names["nucleic_acids"]`. One `import os`, one `if`, one union.

Scope, stated plainly because it is wider than the one add op: `compound_norms` is built from that set
and feeds three checks, not one:

1. the "Reaction input/output does not exist in entities" error (reactions),
2. the "Transport cargo ... is not in entity registries" error,
3. the "add missing compound referenced by a process" op, plus its "Added missing compound" warning.

With the flag on, a name declared as a nucleic acid satisfies all three. With the flag off nothing changes.
Not touched: prompts, the LLM audit step, prefreeze, any other file.

Tests, `tests/test_fix_trna_audit_na_declared.py` (9 items): flag off reproduces the duplicate; `1`/`true`/`YES`
add none; an undeclared input and an undeclared output are still added; `""`/`0`/`off` are off; the
transport-cargo path (flag off adds the duplicate and reports the cargo missing; flag on does neither);
flag on removes the "does not exist" error and the "Added missing compound" warning for the reaction input.
(The earlier "unset vs `0`" test ran the same code twice and proved nothing; dropped, review F5.)

## 3. Offline replay (no-loss evidence)

`replay.py` runs `audit_payload(use_llm=False)` with the flag off then on over every stored merged payload:
`C:/t/acc-runs/legs*/**/merged_payload.json` (66) and the server work items (3), 69 in all. Read-only.
`compare2.py` diffs them (`compare2.txt`).

| measure | result |
|---|---|
| payloads scanned | 69 |
| flag-off output vs the original code (main tree), patch and report | identical on 69 of 69 |
| payloads whose audit result differs, flag off vs on | **1** (`legs4g/base1-PMC5130987`) |
| differences that change a mapped compound into an unmapped nucleic acid | **0** |

The one difference:

| item | flag off | flag on |
|---|---|---|
| op | `add /entities/compounds/-` `{"name": "tRNA Glu"}` | none |
| error | `/processes/reactions/20/inputs`: "Reaction input 'tRNA Glu' does not exist in entities." | none |
| warning | "Added missing compound 'tRNA Glu' referenced in process definitions." | none |
| summary | errors 8, patches 2, warnings 47 | errors 7, patches 1, warnings 46 |
| participant | reaction 20 "Glu to Glu-tRNA Glu", input `tRNA Glu` | same participant, now declared |

Before the flag, this payload ends in the `AMBIGUOUS_REFERENCE` crash (no PWML, section 1), not in a
mapped compound. So the replay finds no case where today's output has a DB-mapped compound that the flag
would turn into an unmapped nucleic acid. It cannot say what strict mode does with the unmapped nucleic acid
afterwards; section 5 and the live A/B are for that.

Note on the script (review F4). The first `compare.py` read `report["issues"]`, which does not exist; the
issues are in `report["errors"]` and `report["warnings"]`. It is kept as `compare_SUPERSEDED.*` and
replaced by `compare2.py`. The first run's "byte-identical 0 of 69" was not a change in behaviour: the full
result strings differ on every payload only through `report.llm.provider` (`local` in the worktree
process, `openrouter` in the main-tree process, from the environment). With `report.llm` removed the two
are identical on 69 of 69 (`ident2.py`).

## 4. Gates

| check | result | log |
|---|---|---|
| new test file | 7 passed (at `69e9984c`); 15 passed with `test_audit_json_llm_payload.py` at `66b8ca26` (9 new, 6 old) | `gate-cover.log`, `gate-unit-f5.log` |
| 14 test files importing `audit_json_llm`, group | 291 passed, 1 failed | `gate-cover.log` |
| the 1 failure, `test_c075_source_support_armed::test_corpus_the_armed_pass_...` | **fails the same way on the base tree** (this change stashed: 40 passed, 1 failed): `EXPECTED_CATCHES` lacks `('PMC12096016','strict',"Pantetheine 4'-phosphate")`, an artifact committed by an earlier merge. Not caused by this change; not touched | `gate-c075-base.log` |
| `suite2.sh` on `69e9984c`, clean tree | SMOKE 511; golden-seam 2/8/4/24 (no test edited); equivalence 8/14/6; vision 370 + 1 skip; web 83: all passed | `gate-suite2.log` |
| surviving processes | 0 in every wrapper report | logs |

The test-only commit `66b8ca26` changes no `src` file, so the suite2 result stands for the code.

## 5. Decisive post-pipeline replay (review request)

**DONE, the flag fails it.** `postreplay.py` replays the stored `legs4g/base1-PMC5130987` merged payload through
`run_post_pipeline` (offline curator, no LLM audit, no LLM gap resolver, strict DB), flag off then on, one lock
hold, 0 survivors (`postreplay2.log`, `postreplay-off.json`, `postreplay-on.json`, `post-off.log`, `post-on.log`).

| arm | result |
|---|---|
| off | `AMBIGUOUS_REFERENCE: the name 'tRNA Glu' is also carried by ['nucleic_acids#0']; a reference to it cannot be unambiguously redirected to 'tRNA(Glu)'`; no PWML (reproduces section 1) |
| on | **the same error**, same place; no PWML |

Where it fires: `postpipeline.py:1473` -> `final_payload.prefreeze_with_final_fill` ->
`prefreeze_resolution.run_prefreeze_resolution` -> `resolve_compounds_prefreeze` (:479) ->
`_reject_ambiguous_renames` (:894). That is after the audit loop (`postpipeline.py` ~:960-1008), which runs with
`use_llm=False` too, so the flag-on deterministic audit did run. With the duplicate compound row gone, prefreeze
still redirects the reaction-20 reference `tRNA Glu` to the mapped name `tRNA(Glu)` and refuses because the
nucleic acid carries the old name. **So section 1's diagnosis was incomplete:** the audit's duplicate add is not
what blocks the export; the rename-vs-nucleic-acid conflict in `pwml/prefreeze_resolution.py` is.

The first attempt (`postreplay.log`) hit a 40-minute cap per arm with no output. Stack dumps every 10 minutes
(`faulthandler`, added in the rerun) showed it was not hung: Stage-2 protein mapping fetches Europe PMC full texts
(`map_ids._europepmc_full_text`). Each arm took about 55 minutes.

## 6. Live A/B (chlorophyll PMC5130987, Gemma, verifier on in both arms)

Driver `drivertrna.sh`, legs `legtrna.sh`, interleaved off, on, off, on, off, on, through the queue.
Scores from `score_legs.py` (the accscore chlorophyll gold, 12 paper-stated steps). **6 of 6 legs done**
(`legs_score6.json`).

| leg | arm | result | reactions | recall | precision | gold matched | tRNA rows |
|---|---|---|---|---|---|---|---|
| off1 | off | FAIL at `post_pipeline`: `AMBIGUOUS_RENAME_TARGET` on `magnesiumprotoporphyrinmonomethylester` | 24 | n/a | n/a | n/a | compounds `Glu-tRNA Glu`, `tRNA Glu` |
| on1 | on | PASS, PWML | 18 | 0.583 | 0.412 | 7 | compounds `tRNA(Glu)`, `L-Glutamyl-tRNA` |
| off2 | off | PASS, PWML | 16 | 0.500 | 0.375 | 6 | compound `L-Glutamyl-tRNA` |
| on2 | on | PASS, PWML | | 0.583 | 0.368 | | compounds `tRNA(Glu)`, `L-Glutamyl-tRNA` |
| off3 | off | PASS, PWML | | 0.583 | 0.368 | | compounds `L-Glutamyl-tRNA`, `tRNA(Glu)` |
| on3 | on | PASS, PWML | | 0.500 | 0.353 | | compound `L-Glutamyl-tRNA` |

Means over passing legs: recall off 0.542 (2 legs), on 0.556 (3); precision off 0.372, on 0.378. No recall or
PWML loss with the flag on. But in none of the six legs is `tRNA Glu` a nucleic acid, so the flag's code path was
never reached live: the A/B shows no harm, not a benefit. `off1` crashed on an unrelated refusal (see 7.2).
(The legs ran after the PO reboot of 2026-11-25; on2/off3/on3 started at 9.6, 8.9, 5.1 GB free.)

## 7. Dead ends and findings

1. A bare `python -` was run by mistake early on and stopped at once; no stray process was left.
2. **Second prefreeze refusal, not fixed here.** `off1` stopped on `AMBIGUOUS_RENAME_TARGET`: two compound
   names canonicalize to `magnesiumprotoporphyrinmonomethylester` and the rename would merge them. Same
   family as Bug B (D-034/D-035 fail-closed), different cause. Flag-off, the flag cannot touch it.
3. The live A/B hit the machine's kernel-process-object leak: free RAM fell 8.7, 8.2, 6.3, 4.7 GB across three
   legs (objects 90k to 151k). The driver stopped itself at the 5 GB guard, as designed.
4. `out/`, `outputs/`, `tmp/` tracked files were dirtied by the live legs in the worktree; not committed.

## 8. Open questions (product owner)

1. **Orchestrator decision (2026-11-25): flag stays OFF on the server.** It is harmless (sections 3 and 6) but does
   not fix the crash (section 5), so enabling it buys nothing. It stays merged, default off. Remove it later if
   the prefreeze fix (next item) makes it redundant.
1a. **PO, biology:** the real fix is in `pwml/prefreeze_resolution.py` (`_reject_ambiguous_renames`): when the
   name a reference carries belongs to a nucleic acid, the rename to a mapped compound name should either
   not apply to that reference or resolve to the nucleic acid. Options: (a) a new flag-gated task on prefreeze,
   proven by this same replay (flag on must export reaction 20) plus the 74-payload no-loss replay, about one
   implementer + one reviewer cycle; (b) leave it: 1 of 62 stored runs hits it. Recommendation: (a), bundled
   with item 2, since both are prefreeze fail-closed refusals on chlorophyll.
2. `AMBIGUOUS_RENAME_TARGET` (7.2): worth its own task? Cost: a new read of `prefreeze_resolution`.
3. The pre-existing `test_c075` corpus failure on the base tree needs an owner.
