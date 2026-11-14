# 03 Remap transport-transporter ids

File: `lib/pwml_parser.rb`. Patch: `03-transport-transporter-remap.patch` (+19 lines, 5 hunks, additions only).

## Problem
`add_content` remaps `reaction-enzyme-id` through `@reaction_enzyme_list` (filled by `store_reaction_enzyme_ids`) but has nothing for `transport-transporter-id`. A transporter visualization keeps the raw XML id, which matches no `transport_transporters` row, so its `transport_transporter` is nil. If the raw number happens to equal a real row id, it silently attaches to an unrelated transporter.

## Evidence
- Harness with the writer fixes on, before this patch: 19 `transport_transporter_visualization: Transport transporter can't be blank` messages in 12 files (`FINAL-PAYLOAD-P2.md` section 4).
- The same failure occurs on the real PathBank file PW176888.
- After this patch (`FINAL-PAYLOAD-P3.md` section 4): 3 messages in 3 files remain (cause not traced, probably the failing left=right transports). Transporter visualizations created: 0 to 16 (together with fix 02). Clean files went 29 to 43 of 59 across the whole P3 change set, not attributable to this fix alone.
- Code: the reaction analog is `store_reaction_enzyme_ids` (line 810 of the final file); the new `store_transport_transporter_ids` is at line 832.

## The fix
1. `@transport_transporter_list = {}` initialised in both places `@reaction_enzyme_list` is (constructor and per-pathway reset).
2. A `transport-transporter-id` branch in `add_content` that reads that map.
3. `store_transport_transporter_ids(node, transport)`: for each `transport-transporters/transport-transporter` it translates the XML `protein-complex-id` through `@protein_complex_list`, finds the row with `Transport#get_enzyme`, and records XML id to row id.
4. It is called after a `Transport` is found as existing or saved (`if klass == Transport`), the way the reaction enzyme store is called for `Reaction`.

## How to verify
Apply 01 to 03 and run the harness (`README.md`). Expected: the 19 "Transport transporter can't be blank" messages drop to 3 or fewer on the 59-file set, and PW176888 loses its transporter message. `ruby -c` gives `Syntax OK`.

## Risk
Medium-low. Additions only; a missing mapping now gives "can't be blank" rather than a wrong attachment. Known gaps, not fixed here: `reaction-coupled-transport` enzymes probably have the same gap (not checked); `get_enzyme` ignores the biological state, so two transporters with the same complex in different states of one transport could collide; the 3 residual messages are untraced.
