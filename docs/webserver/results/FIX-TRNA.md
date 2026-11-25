# FIX-TRNA: the audit counts a nucleic acid as declared (Bug B), flag-gated

Branch `fix/audit-nucleic-acid-declared`, cut from `main` 5bf9d552. PO-approved, flag-gated, default off.
Code commits `69e9984c` (change and tests) and `66b8ca26` (review F5 tests). This record is a separate
docs-only commit. Evidence: `docs/webserver/evidence/FIX-TRNA/`.

STATUS: change, tests, offline replay and gates are done. **Section 5 (decisive post-pipeline replay) and
the live A/B (section 6, 3 of 6 legs so far) are still open**, see the status lines there. This doc is
updated in place as they land.

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

**OPEN.** Replay the stored `legs4g/base1-PMC5130987` merged payload through post-pipeline, offline curator,
flag off and on: flag off must reproduce `AMBIGUOUS_REFERENCE`, flag on must export reaction 20 with
`tRNA Glu` as a nucleic-acid participant. Waiting for RAM (4.7 GB free, guard 5 GB).

## 6. Live A/B (chlorophyll PMC5130987, Gemma, verifier on in both arms)

Driver `drivertrna.sh`, legs `legtrna.sh`, interleaved off, on, off, on, off, on, through the queue.
Scores from `score_legs.py` (the accscore chlorophyll gold, 12 paper-stated steps). **3 of 6 legs done.**

| leg | arm | result | reactions | recall | precision | gold matched | tRNA rows |
|---|---|---|---|---|---|---|---|
| off1 | off | FAIL at `post_pipeline`: `AMBIGUOUS_RENAME_TARGET` on `magnesiumprotoporphyrinmonomethylester` | 24 | n/a | n/a | n/a | compounds `Glu-tRNA Glu`, `tRNA Glu` |
| on1 | on | PASS, PWML | 18 | 0.583 | 0.412 | 7 | compounds `tRNA(Glu)`, `L-Glutamyl-tRNA` |
| off2 | off | PASS, PWML | 16 | 0.500 | 0.375 | 6 | compound `L-Glutamyl-tRNA` |
| on2, off3, on3 | | not run: driver stopped on the 5 GB RAM guard before `on2` | | | | | |

Reading so far: in none of the three legs is `tRNA Glu` a nucleic acid, so the flag's only effect
(section 3) was not exercised. Stage 1 differs run to run; this A/B cannot show the flag's effect unless a
leg declares the tRNA as a nucleic acid. `off1` crashed on a different, unrelated refusal (see 7.2).

## 7. Dead ends and findings

1. A bare `python -` was run by mistake early on and stopped at once; no stray process was left.
2. **Second prefreeze refusal, not fixed here.** `off1` stopped on `AMBIGUOUS_RENAME_TARGET`: two compound
   names canonicalize to `magnesiumprotoporphyrinmonomethylester` and the rename would merge them. Same
   family as Bug B (D-034/D-035 fail-closed), different cause. Flag-off, the flag cannot touch it.
3. The live A/B hit the machine's kernel-process-object leak: free RAM fell 8.7, 8.2, 6.3, 4.7 GB across three
   legs (objects 90k to 151k). The driver stopped itself at the 5 GB guard, as designed.
4. `out/`, `outputs/`, `tmp/` tracked files were dirtied by the live legs in the worktree; not committed.

## 8. Open questions (product owner)

1. Decision rule: enable only if flag-on loses no gold step and no PWML against flag-off and the replay shows
   no mapped-compound to unmapped-nucleic-acid change. Replay: met (0 changes). Live A/B: undecided until
   section 5 and the legs are in; the three legs so far cannot test it.
2. `AMBIGUOUS_RENAME_TARGET` (7.2): worth its own task? Cost: a new read of `prefreeze_resolution`.
3. The pre-existing `test_c075` corpus failure on the base tree needs an owner.
