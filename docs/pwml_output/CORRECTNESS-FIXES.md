# Correctness fixes: results record (Phase B)

PO rulings D-W23 g, D-W24 (biology changes authorized on this branch only), D-W25/D-W26 (PathWhiz target = the
`t2pw/*` stack). Branch `pwml/correctness`, pushed, **not merged** into `webserver/refactor` or `main`.
Assessment that drove this work: `CORRECTNESS-ASSESSMENT.md`. Working records (lane JSON, reviews, logs, replay tables):
`C:/t/corr-assess/` (lanes/, reviews/, tests/, g3/), PathWhiz harness runs `C:/t/corr-pw/`.

Final head of the gates: `9fcc8d19`. Small evidence subset: `docs/pwml_output/evidence/correctness_fixes/`.

## 1. Lanes merged

Every lane: own branch and worktree, failing-first tests, independent `ws-reviewer` (opus/high) on the actual diff,
tests through the bounded wrapper with 0 survivors, merge into `pwml/correctness` only after approval.

| Lane | What | Ruling | Before -> after (97 stored-run replays unless noted) | Review |
|---|---|---|---|---|
| L1 | Web-diagram prototype v2 (`C:/t/ev-diagram-v2`, outside repo): crash fix (duplicate transporter pill), regulation hidden by default with count + toggle, enzyme beside each reaction, identical steps once with "xN in data", plain-word tags + legend, lane order follows flow, compartment-crossing captions, arrowheads, hover highlight; independent selfcheck | D-W23 g (2) | render errors 1 -> 0; crossings 1114 -> 444 (full view 616); enzyme links > 200 px 88 -> 6; selfcheck 0 failures, 2 negative proofs | APPROVE WITH NITS |
| L2 | Species id collision fixed (idx 134 Homo sapiens / Mus musculus both id 1); compartment labels too wide drawn as stacked lines | D-W23 g (1) | over-wide labels 17 -> 0; v1 byte-identical 96/97 (134 = the fix) | APPROVE WITH NITS |
| L3 | Layout v2: one location per (compound, compartment) along a chain; cycles broken in pathway order | D-W23 g (1) | drawn components 222 -> 135 (logical 109); compound locations 1099 -> 911; reading-order violations 51 -> 20 of 253; overlap/label checks 0 -> 0; v1 97/97 identical; content 0/97 changed | APPROVE (2 rounds) |
| L4 | Activators/inhibitors become Activation/Inhibition interactions, not enzymes; Stage-6 wrapper for every exportable type; Binding/Regulation/Catalysis/Modification/Unspecified emitted with PathWhiz templates; evidence on every interaction | D-W24 1, 4, 6; D-W26 | 17 enzyme entries moved, all re-appear as interactions; drawn interactions 23 -> 48 (stored payloads), 61 -> 170 (fresh Stage 6); nothing drawn before lost | APPROVE (2 rounds) |
| L5 | Export quarantine never deletes a reaction for its enzyme: bare enzyme wrapped, unwritable actor dropped + recorded, reaction kept, review_required | D-W24 3 | reactions 377 -> 381 (27, 29, 44, 116 gain one; 113 duplicate merged); idx 116 ceramide synthase reaction now drawn; no tag count drops | APPROVE WITH NITS (2 rounds) |
| L6 | Near-synonym compartments merged; duplicate chains and exact duplicate reactions written once; fail-closed on ambiguity | D-W24 5 | states 290 -> 233; payload reactions 424 -> 392; 0 loss problems (independently rechecked); flag off identical 97/97 | APPROVE WITH NITS (3 rounds) |
| L7 | Missing compartments: neighbour estimate with evidence; else kept in place and flagged compartment_not_stated (no silent fallback). LLM estimate and general-state fallback built but OFF | D-W24 2, 7a | 10 neighbour estimates (0 correct->wrong vs papers); only idx 79 moves (CYP51A1 -> ER); 75 unresolved flagged | APPROVE WITH NITS (2 rounds) |
| L7c | Stricter acceptance for the (off) LLM estimate; live check | D-W24 2 | live: 25 calls, 1/88 placed (better), 0 worse; LLM_ESTIMATE stays off | APPROVE WITH NITS |
| L8 | Evidence checks flag (never remove, never change release status); enzymes with no DB identity kept as placeholder (strict_db); Unknown placeholder asserts no species (PWML keeps species-id 4 only as PathBank 9659 lookup key) | D-W24 6, 7b, 7c | 165 elements flagged, 0 removed; 97/97 identical with flag on and off vs pre-lane head | APPROVE WITH NITS (2 rounds) |
| L9 | Stage-1 draws (DRAWS=3) run concurrently; merge unchanged; order-invariant | D-W24 8 | wall time PMC12421875 strict 1636 s -> 1375 s (live) | APPROVE (2 rounds) |
| L9b | Root cause + fix of the equivalence ReplayMiss flake introduced by L9 (cassette serves identical requests in arrival order) | - | test_equivalence_run_pathway 5/15 failing -> 20/20 + 5/5 passing | APPROVE WITH NITS |
| L10 | Content-neutral review nits (id guard for every id type, label length, QA warning, ChEBI currency ids, docstrings, SBML omission reaction ref) | - | NOT merged: coded but untested when the machine restarted; dropped for low usage budget. Uncommitted work stays in `C:/t/corr-L10` | - |

Post-merge suites on each merge result: SMOKE 511, golden-seam 2/8/4/24 (38), equivalence run_pathway 8 /
app_vs_service 14 / default_path 6 / second_fixture 8, 0 survivors.

## 2. Deferred, and why (no-regression rule)

- **LLM compartment estimate (L7 step 2)** and **general-cell-state fallback (L7 step 3)**: off. Step 3 would move 22
  correct-by-chance placements to a vaguer state; step 2's live gain is 1 of 88. Before either is switched on: re-run L6's
  merge on moved rows; adjective-synonym leak (N4); "no quote" counted as "not verbatim"; H+ cargo; enzyme aliases.
- **`no_defensible_reaction_support` removals** unchanged: 9 of 10 are on the contract section 13 negative-control papers.
- **Cytoplasm vs cytosol in eukaryotes**: kept apart (conservative; settles idx 3).
- **L6 guard gap**: 4 duplicate reactions (idx 6, 7, 17) stay unmerged because a complex name is also a protein synonym.
- **Reversibility / stoichiometry fields** (D-W24 7): assessed and proposed only (see section 4).
- **Evidence-check precision**: about half the flags of `enzyme_not_tied_to_step` and `evidence_names_no_participant` are
  false positives in the reviewer's sample; they only flag and never feed release status.

## 3. Open questions (for the PO or other tracks)

- L6 follow-up: write transports that become identical after the state merge once (idx 5, 9).
- Not done (usage budget): L10 content-neutral nits; web prototype review nits (merge key with stoichiometry/tissue/species, `build_page.py` getattr).
- PathWhiz track: bind the "Unknown" placeholder (PathBank 9659) without a species.
- Text record: rows flagged by L6 F3 / L5 dropped actors are not printed with their own advice yet.
- Equivalence fixtures: a future re-recording made with concurrent draws would store answers in finish order; record with sequential draws or fix the recorder first.
- Release-line failures not caused by this branch: test_c052 (1) and test_orch768 sour_taste_v2 (1) fail at the release merge base; diagnosed on the release line.
- idx 116 "ceramide synthase" grounded to TLCD3B (Q71RH2), not CERS1-6 (identity/mapping).

## 4. Reversibility and stoichiometry (assessment and proposal only)

Assessor R1, read-only (`C:/t/corr-assess/r1/assessment.json`).

- **Today:** the extraction prompt and payload carry no direction, reversibility or stoichiometry field (all 1544
  reaction members are bare strings). The IR writes every reaction `Right` and stoichiometry 1. PathWhiz stores direction
  as `>`, `<` or `<>` (no "unknown") and writes SBML `reversible=true` exactly for `<>`, so the all-`false` PathWhiz SBML
  comes from our output, not from PathWhiz. Our SBML already marks direction "not stated". The `stoich` module is a
  KEGG/ChEBI participant-repair agent (off by default; it should stay off), not a reader of paper stoichiometry.
- **Frequency (152 stored runs):** 13 of 621 reactions (about 2%) carry reversibility wording in their own evidence;
  about 7 state reversible (MenF, LpxA, PSAT, HFP isomerase, IlvE), 3 irreversible (MenD, LpxC). No reaction member has a
  non-1 stoichiometry; fewer than 10 papers print a coefficient.
- **Proposal:**
  1. `reversible: true|false|null` on the payload reaction, set only from an explicit paper quote checked against a closed
     vocabulary, written as `<>` to PWML/SBML when true. Needs a prompt change (PO ruling) and a live 3x re-run of the ~11
     affected papers; provably additive on stored payloads.
  2. Stoichiometry: defer (under 2% benefit, no gold measures it, highest invention risk).
  3. Now, no prompt change: record "direction and stoichiometry are defaults" in the IR info messages and the web/text
     report.

## 5. Gates (D-W24 9)

BEFORE arm = release base `12c99a00` (= `bc63f504`); AFTER arm = `9fcc8d19` (live legs: `c5734190`, which differs only
in the switched-off LLM estimate).

**97-replay A/B** (`gates/ab97.json`, `ab97_verify.json`): 97 exports in both arms (same idx set); 25 byte-identical, 72
changed. Count drops in 11 files are merged duplicates (L6): 0 compound/protein/complex names lost, 0 distinct reaction
signatures (substrates, products, direction) lost. Location drops are placeholder states folded into named states.

**PathWhiz import, plain t2pw/09 `3408bc2c` (no patch)** (`gates/pathwhiz_ab.json`, 26 sample screenshots per arm in
`C:/t/corr-pw/runs/{before09,after09}/screenshots`):

| Created / expected | BEFORE | AFTER |
|---|---|---|
| imports successful | 97/97 | 97/97 |
| reactions | 388/409 | 378/381 |
| compounds | 681/687 | 685/691 |
| proteins | 333/374 | 335/376 |
| complexes | 389/393 | 393/397 |
| transports | 66/74 | 65/74 |
| interactions | 18/23 | 44/49 |
| importer error messages | 84 | 82 |

Reactions created fall by 10 because duplicates are now written once (expected also falls 409 -> 381); PathWhiz now
creates 99% of written reactions against 95% before. Interactions created 18 -> 44. One side effect found: in idx 5 and 9
the L6 state merge leaves two identical transports, and PathWhiz imports one (transports 66 -> 65). No content is lost
(they are identical), but L6 should write identical transports once: open item.

**Live gold A/B, strict, DRAWS=3** (launcher; `g3/comparison.md`, `g3/scores/`):

| Paper | Arm | Export | Reactions | Gold matched | Enzymes |
|---|---|---|---|---|---|
| PMC12421875 menaquinone, run 1 | BASE / AFTER | review_required / review_required | 11 / 14 | 9/9 / 9/9 | 9/9 / 9/9 |
| PMC12421875 re-run | BASE / AFTER | review_required / release_ready | 13 / 13 | 9/9 / 9/9 | 9/9 / 9/9 |
| PMC12096016 enterobactin | BASE / AFTER | review_required / review_required | 6 / 7 | 5/5 / 5/5 | 5/5 / 5/5 |
| PMC12444477 lipid A (pinned_v1) | BASE / AFTER | refused (no PWML) / review_required | 28 in payload / 30 | n/a | n/a |

AFTER is never worse on gold coverage. Lipid A now exports (pinned_v1: false real identifiers 4 -> 1, orphaned
references 9 -> 0, missing supported reactions 1 -> 0). Enterobactin pinned_v1: false identifiers 4 -> 0, missing
supported 3 -> 2. The MenA/MenG name-variant duplicates appear in both arms depending on the draw (not an AFTER effect).
Research legs were skipped (strict pairs showed no problem).

**Web diagrams** (all 97 re-rendered with `C:/t/ev-diagram-v2` from the final IR to `C:/t/render-corr/web/`, manifest in
the `render-p4` shape, screenshots `C:/t/render-corr/web_shots/`): 0 render errors, 0 selfcheck failures; crossings 425
(old prototype 1114), enzyme links over 200 px 6 (was 88).

**Final suite on `9fcc8d19`** (`C:/t/corr-assess/tests/final/`): SMOKE 511 passed; golden-seam 2/8/4/24; equivalence
run_pathway 8 (twice), app_vs_service 14, default_path 6, second_fixture 8; 0 survivors.

**Correctness verdict per layer (after this branch):**

| Layer | Verdict |
|---|---|
| Extraction | Unchanged by design (no prompt change); gold coverage held or improved on all 3 gold papers with DRAWS=3; name-variant duplicates remain draw-dependent |
| Final payload | Duplicates merged with 0 loss; missing compartments estimated or flagged; evidence flagged; placeholders kept |
| PWML | Activators/inhibitors no longer enzymes; all 7 PathWhiz interaction types; no reaction deleted for its enzyme; species ids unique; chains drawn once, in pathway order |
| PathWhiz import (t2pw/09) | 97/97; reactions created 99% of written; interactions 18 -> 44; one duplicate-transport side effect (idx 5, 9) |
| SBML (ours) | Carries modifier roles, withheld actors, compartment source; still no layout |
| SBGN / SBML (PathWhiz) | Fixed PathWhiz-side by t2pw/06-08 (transports X -> Y, SBGN 107/107 per PATHWHIZ-FIXES) |
| Web diagram | Faithful (selfcheck), clearer (crossings -62%, enzyme links near their reaction, regulation behind a toggle) |

## 6. Live-run cost

| Run | Calls | Tokens | Cost |
|---|---|---|---|
| L7 live check (provider down) | 0 | 0 | $0 |
| L7c live check | 25 | ~303k prompt + 18k completion | ~$0.04 (client default rates) |
| G3 live gold legs, 8 strict legs (both arms) | 426 requests in total (42+51, 52+55, 37+45, 78+66) | not recorded in run artifacts (client stats are in-process) | not measured |
