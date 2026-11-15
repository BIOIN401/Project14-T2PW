# B4: transports whose left and right biological state are equal

Verdict: **not a writer or IR defect. Origin is upstream (Stage 1 data, and `process_normalizer.ensure_autostates`, a forbidden directory). Documented only, no code change.**

## The cases (152-run replay, `final_payload_p2/replay/{base1,off5,on5}`, identical in all three)
Scan of every `transport-element` with equal `left-biological-state-id` and `right-biological-state-id`: 4 transport elements in 4 files (idx 10, 102, 138, 78). The triage counted 6 in 5 files (import-time counts, which include the second run mapping and the PathBank re-import); the replay shows 4 distinct ones. The 2 others were not located in the replay output.

| idx | run (source `final_mapped.json`) | transport | cargo | input from / to state |
|---|---|---|---|---|
| 10 | runs_papers/ORCH-759p/.../PMC7059786 | OTOP1-mediated H+ permeation | Hydrogen Ion | `__auto_state__` / `__auto_state__` |
| 138 | runs_smoke/2026-09-08_1528/.../PMC6112128 | TubI-mediated TBN export | tubercidin | `__auto_state__` / `__auto_state__` |
| 78 | runs_verify/2026-08-27_1341/.../PMC12096016 | enterobactin secretion | enterobactin | `__auto_state__` / `__auto_state__` |
| 102 | runs_verify/2026-09-02_2052/.../PMC12096016 | enterobactin secretion | enterobactin | `E. coli cytoplasmic state` / `E. coli cytoplasmic state` |

## Trace
1. `ir.py` (~2287-2310) copies `from_biological_state` / `to_biological_state` (or the per-element left/right fields) through `resolve_state` into `left_biological_state_key` / `right_biological_state_key`. It does not merge states, but `resolve_state` (`ir.py` ~1836) returns the default (first) biological state for an empty name, and for an unresolved name only after logging it, so two empty or unresolved sides collapse to the same default state: a second way to get left = right, introduced in the IR rather than copied from the input.
2. Three of four: Stage 1/2 left both states empty. `process_normalizer.py:3198-3207` (`ensure_autostates`) then fills **each** missing side with the single `__auto_state__`, so the transport leaves and arrives in the same state. The evidence text says the cargo is secreted to the extracellular space, so the intent was a real compartment change; the information was lost before the exporter.
3. One of four (idx 102): the model wrote the same explicit state twice, although the evidence says "secreted to the extracellular environment" and an `extracellular space` state exists in the payload. Stage 1 extraction error. The same IR also holds a correct twin transport of enterobactin (cytoplasm -> extracellular space), so withholding the degenerate one loses nothing in idx 102.

## Why no writer fix
- The only writer-side options are to drop the transport element or to guess a state; the first loses content (project rule: no regression in PWML output), the second invents biology. Neither is inside the allowed files.
- Real fix lives in `pipeline/process_normalizer.py` (do not give a transport's missing side the same auto state as the other side) or in the Stage 1 prompt (`llm/prompts/`); both are outside this track.

## Open question for the product owner
Withhold such a transport (like ORCH-747 withholds untyped interactions, with a side-file line), or fix upstream so the extracellular side is inferred from the evidence text. Cost: withholding removes 4 transports from 4 of 152 files (for idx 102 nothing is lost, the correct twin remains); the upstream fix is a biology change needing a D-xxx.
