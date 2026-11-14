# 02 Remap left/right biological state ids on transports

File: `lib/pwml_parser.rb`, `PwmlParser#add_content`. Patch: `02-transport-biological-state-remap.patch` (+8 lines, two `elsif` branches).

## Problem
`add_content` remaps `biological-state-id` through `@biological_state_list` but has no branch for `left-biological-state-id` or `right-biological-state-id`. They fall to the generic final `else` and keep the raw XML id. `TransportElement` then looks up a biological state whose database id equals that XML id. In an empty database this fails. In production it does not fail loudly: the transport silently attaches to whatever biological state happens to have that id.

## Evidence
- Harness, empty database, before the fix: 37 of 37 transports lost. All 37 messages read "Left ... Right ... can't be blank", followed by 37 `Transport elements is too short` (`transport.rb:30`, `transport_elements length >= 1`) and 37 `transport_visualization` messages (`IMPORT-ERROR-TRIAGE.md` section 6A, `FINAL-PAYLOAD-P2.md` section 4 item 2).
- Lookup site: `transport_element.rb:15-16`. Parser site: the old generic `else` (line 585 of the 2026-11-11 file); the existing `biological-state-id` branch is at lines 491-496.
- Real PathBank file PW012926 has a transport with left 151 / right 223 and imported only because that database held states with those ids.
- After the fix (P3 harness, together with fix 03): transports created 31 of 37. The remaining transport-family messages have other causes (left equals right, 6; `FINAL-PAYLOAD-P3.md` section 4). The effect of fix 02 alone was not measured separately.

## The fix
Two `elsif` branches right after the `biological-state-id` branch (lines 498-505 of the final file), copied from it minus the debug log line: read `@biological_state_list[node.content]` and store it under the underscored key.

## How to verify
Apply 01 and 02, then run the harness (`README.md`) on the 59-file set. Expected: no "Left/Right ... can't be blank" messages, and transport elements are created. `ruby -c` gives `Syntax OK`. A PWML id with no entry in the map now yields nil (a "can't be blank" error) instead of a wrong attachment.

## Risk
Low. Same pattern as the neighbouring branch. The behaviour change is that a missing mapping becomes a loud validation error instead of a silent wrong attachment, which is the intent. It assumes biological states are imported before transports, as for the existing `biological-state-id` users.

## Branch
`t2pw/02-transport-state-remap`, commit `48ab7499`, on top of `t2pw/01-vcv-node-typo`. Local only, not pushed. Worktree C:/t/pwpr/02.
