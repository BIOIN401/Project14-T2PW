# Task B (D-W20b): transport states, before/after

Code: `src/t2pw/pipeline/process_normalizer.py` (`_declared_side_state`, new, ~3143; used in `ensure_autostates` ~3229 and ~3234). Commit 36f0b36c.
Script: `b_before_after.py` (old = `git show f66b3ef2:` module, new = working tree), raw: `b_before_after.json`.

Input: all 243 stored `merged_payload.json` under `runs_*` (the highest stored stage before the process normalizer; `final_mapped.json` is after it, so it cannot show the change). Each run through `normalize_process_payload` old vs new (research mode when the path has `/research`), whole output compared.

| measure | value |
|---|---|
| runs compared / errors | 243 / 0 |
| runs with any difference | 2 |
| transports changed | 2 |
| differences outside transport from/to state in the payload | 0 |
| report differences | 8, all gate connectivity counters of the OTOP1 run (nodes 49->50, edges 70->73, main component 45->46): one extra location edge exists now |

## Every changed transport
| run | transport | cargo | before (from / to) | after (from / to) |
|---|---|---|---|---|
| runs_papers/ORCH-759p/legs/sour_taste_v2-strict/2026-10-22_2233/papers/PMC7059786/strict | OTOP1-mediated H+ permeation (idx 4) | H+ | `__auto_state__` / `__auto_state__` | `__auto_state__` / `Type III taste cell state` |
| runs_verify/2026-08-18_1328/papers/PMC12452463/strict | enterobactin secretion (idx 1) | enterobactin | `__auto_state__` / `Escherichia coli extracellular` | `Escherichia coli cytosol` / `Escherichia coli extracellular` |

Both new values are states the row declared itself in `elements_with_states` (side left/right) and that exist in `biological_states`.

## Left = right transports remaining after the change (4, was 5 in the merged payloads)
| run | transport | cargo | state both sides | why untouched |
|---|---|---|---|---|
| runs_smoke/2026-09-08_1528 PMC6112128 | TubI-mediated TBN export | tubercidin | `__auto_state__` | no state declared anywhere on the row |
| runs_verify/2026-08-21_2057 PMC12096016 research | ferric-enterobactin import | ferric enterobactin | `__auto_state__` | same (new find; not in the P3 list) |
| runs_verify/2026-08-27_1341 PMC12096016 | enterobactin secretion | enterobactin | `__auto_state__` | same |
| runs_verify/2026-09-02_2052 PMC12096016 | enterobactin secretion | enterobactin | `E. coli cytoplasmic state` | model wrote the same explicit state twice; a correct twin exists; overriding explicit output is not safe |

## Tests (bounded wrapper, 0 surviving processes)
- New file + test_process_normalizer + c121 + the 4 golden-seam files (unedited): 106 passed.
- All 39 other test files that mention process_normalizer / ensure_autostates: 1081 passed, 8 skipped, 1 failed: `test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform`. It is an AST check of `streamlit_app.py` source (untouched by this commit); not caused by this change, but not re-run on f66b3ef2 to prove it was already failing.
- SMOKE not run (separate dispatch).
Reports: `tests_final/b_new.*`, `tests_final/b_rest.*`.

## Fix round (after the opus review of HEAD cdcac27a)
`_declared_side_state` now compares the declared state names with `_normalize` (was case-sensitive `_canonical`, so `Cytosol` and `cytosol` on one side disagreed and fell back to the placeholder), and `ensure_autostates` keeps `__auto_state__` for a side whose declared state equals the state the other side ends up with (it would create a new left == right transport). New tests in `tests/test_process_normalizer_transport_states.py`: equal-sides case, and case-insensitive agreement.

Re-run of `b_before_after.py` on the working tree (243 stored merged payloads, old = f66b3ef2): **unchanged**. 243 compared / 0 errors, 2 runs changed, the same 2 transports (OTOP1-mediated H+ permeation: `__auto_state__` / `Type III taste cell state`; enterobactin secretion: `Escherichia coli cytosol` / `Escherichia coli extracellular`), 0 differences outside transport states, the same 8 gate connectivity counter differences, left = right transports 5 old -> 4 new (the same 4 as above). The new equal-sides rule does not fire on any stored payload. Raw: `b_before_after.json`, `b_before_after.stdout.txt`.
