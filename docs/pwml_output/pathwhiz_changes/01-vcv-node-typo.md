# 01 vcv_node typo in the vacuous-edge-visualization loop

File: `lib/pwml_parser.rb`. Patch: `01-vcv-node-typo.patch` (1 line changed).

## Problem
The loop over `vacuous-edge-visualizations/vacuous-edge-visualization` names its block variable `vev_node`, but the body passes `vcv_node` to `build_nested_visualization`. `vcv_node` is a leftover variable from an earlier loop, so every vacuous edge visualization is built from the wrong XML node (or the conversion stops, because exceptions in this method are swallowed).

## Evidence
- Line 291 of the 2026-11-11 file (line 293 of the final file): `v = self.build_nested_visualization(vcv_node, pathway_visualization)`, directly under `... .each do |vev_node|` (line 290 / 292).
- Found by reading the code (`FINAL-PAYLOAD-P2.md` section 4, item 1). **Not reproduced.** No harness run isolates this fix, so there is no before/after message count for it.

## The fix
Use the loop variable: `build_nested_visualization(vev_node, pathway_visualization)`.

## How to verify
Syntax: `ruby -c lib/pwml_parser.rb` gives `Syntax OK`. Behaviour: import a PWML that contains a `vacuous-edge-visualization` (harness command in `README.md`) and check that the pathway visualization gets its vacuous edge visualizations. Expected: no change in the harness message counts for the 59-file set unless a file carries vacuous edge visualizations.

## Risk
Very low. One identifier; the old value was always the wrong node. Does not touch id remapping.

## Branch
`t2pw/01-vcv-node-typo`, commit `17af3272`, from `test-prod-with-parser-fix` HEAD `6e11df0e`. Local only, not pushed. Worktree C:/t/pwpr/01.
