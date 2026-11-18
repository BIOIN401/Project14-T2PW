# Vision programme: progress

Plan: `docs/vision/PLAN.md`. Branch `vision/main` from `origin/main` `a6c07e16`. Sub-orchestrator writes no code.

## Current state

- 2026-11-17: plan written. Wave 1 (V1 model, V2 sbgn, V3 diagram) dispatched. Heavy lock held by the Thursday demo
  (`vllm-wire`); free RAM about 1 GB at planning time, so wave 1 runs light unit tests only.
- 2026-11-17 late: V2 and V3 merged into `vision/main` (aedb2d15, 135cfb64). V1 approved with notes in review r3; fix r3
  (VD-11 cofactor rule) at `vision/model` cee6bae4, review r4 pending, then merge.
- RAM: the PC has a kernel memory leak (main orchestrator, 2026-11-17). Floors: 2 GB free for focused tests and small
  proofs, 4 GB for SMOKE, replays, Docker, live legs. I briefly lowered the light floor to 1 GB in `C:/t/vis-runs/RULES.md`;
  the main orchestrator reversed it and RULES.md is fixed. Two V1 runs ran under the 1 GB floor (model-unit-10, model-proof-4);
  both finished with 0 survivors.

### Queue state at the 2026-11-18 hold (restart pending; nothing re-queued until main says the showcase is done)

Merged in `vision/main` (pushed): V1, V2, V3, V5, V6 at 04596e6c. Gated work still owed, in this order:

1. **Post-merge suites on 04596e6c** (V6 merge). Snapshot worktree `C:/t/vis-suite-0459` exists. Command (Git bash, background):
   `bash C:/t/vis-runs/queue.sh vis-suite suite-0459 5400 C:/t/vis-runs/json/suite-0459.json 4 C:/t/vis-runs/logs/suite-0459.log -- "C:/Program Files/Git/bin/bash.exe" C:/t/vis-runs/suite.sh C:/t/vis-suite-0459 suite-0459`
2. **V7a checks** (`vision/checks` fdce8edd, agent a94df03a34785b42f, sonnet/high): unit tests (min 2 GB) then the 97-run proof (min 4 GB); exact commands in `C:/t/vis-checks/docs/vision/results/V7a.json` `commands_still_to_run`, run through `queue.sh`. Code is committed, nothing run.
3. **V4 extend-db** (`vision/extend-db` db7896c2, agent a909dd2a92c062ddd, opus/high): 12 focused tests passed (`logs/v4-unit-2.log`); live proof `v4-live-1` (min 4 GB) NEEDS THE PATHBANK DB, which is unreachable: do not run until main says the DB is back.
4. **V9 bench** (`vision/bench` 190ba7b0, agent a8577fd290255303b, sonnet/high): done, 13 tests passed, proof ok on the paper set; full-view gold for PW739181, PW495535, PW488575, PW766996 NEEDS THE DB (via `C:/t/corr-assess/scripts/live_launch.py`): hold. Review of the current diff can run now (static).
5. Not yet dispatched: V7b (attractiveness metrics + PathWhiz gallery, compose project `t2pw-vis`, Docker, min 4 GB), V8 extend-lit (after V4), V10 web (V5+V6 merged; main's rulings in the section below).

### Resume notes (after a restart)

1. Worktrees: `C:/t/vis-main` (vision/main), `C:/t/vis-model`, `C:/t/vis-sbgn`, `C:/t/vis-diagram`. Rules for agents:
   `C:/t/vis-runs/RULES.md`; gate script `C:/t/vis-runs/bw.sh`; briefs ready to send in `C:/t/vis-runs/briefs/`
   (V4, V5, V6, V9, SUITE). Logs `C:/t/vis-runs/logs/`.
2. Next: V1 review r4 verdict -> merge `vision/model` into `vision/main` -> post-merge suites for wave 1 (SUITE.txt brief,
   ws-test-runner haiku/medium, needs 4 GB free and the heavy lock free) -> push `vision/main`.
3. Then wave 2: V4 extend-db (opus/high), V5 scope (sonnet/high), V9 bench (sonnet/high), each from the new `vision/main`
   head (replace BASE in the brief), worktrees `C:/t/vis-<lane>`.

## Dispatch table

| when | lane | role | agent (model/effort) | branch / worktree | state | evidence / note |
|---|---|---|---|---|---|---|
| 2026-11-17 | plan | scout x2 + follow-up | ws-scout (sonnet/medium) | read-only, `C:/t/vis-main` | done | payload, exporters, lineage fields, harness, gold, env mapped |
| 2026-11-17 | V1 | implementer | general-purpose (opus/high) | `vision/model`, `C:/t/vis-model` | done 11699135 | 17 tests; identity 152/152 (DEFAULT and FULL == base) |
| 2026-11-17 | V1 | review r1 | ws-reviewer (opus/high) | same | CHANGES_REQUESTED | F1 reaction identity ignores state; F2 hops through currency metabolites (blocking); F3-F9 notes, all sent back |
| 2026-11-17 | V2 | implementer | general-purpose (sonnet/high) | `vision/sbgn`, `C:/t/vis-sbgn` | done 4b94d430 | 9 tests; 97/97 valid (XSD + libSBGN schematron); 498 process glyphs = 424 reactions + 74 transports |
| 2026-11-17 | V2 | review r1 | ws-reviewer (opus/high) | same | CHANGES_REQUESTED | F1 substring relation matching reverses signs ("deactivates" -> stimulation); F2 non-regulatory relations drawn as modulation (VD-8) |
| 2026-11-17 | V3 | implementer | general-purpose (sonnet/medium) | `vision/diagram`, `C:/t/vis-diagram` | done e2b83593 | 5 tests; SVG byte-identical to prototype 97/97, selfcheck 0 failures |
| 2026-11-17 | V3 | review r1 | ws-reviewer (sonnet/high) | same | CHANGES_REQUESTED | fixtures have no reaction or transport; bad font env crashes import |
| 2026-11-17 | V1 | fix r1 + review r2 | same implementer (opus/high); same reviewer (opus/high) | `vision/model` 44904bfc | CHANGES_REQUESTED | F1-F9 fixed (26 tests, 9 failing first); new R1: currency stripped before the reversed match merges hexokinase with G6Pase |
| 2026-11-17 | V1 | fix r2 + review r3 | same / same | `vision/model` 94a8a593 | APPROVE WITH NOTES | R1-R4 fixed, 28 tests; identity 152/152; N1/N2 cofactor-merge ruling VD-11 sent as fix r3 before merge |
| 2026-11-17 | V2 | fix r1 + review r2 | same implementer (sonnet/high); same reviewer (opus/high) | `vision/sbgn` 2ea34e25 | APPROVE WITH NITS | 39 tests; 97/97 valid; interactions drawn 91 -> 5 of 188 (VD-8: 86 not regulatory, 97 no process target, all counted) |
| 2026-11-17 | V2 | merge | sub-orchestrator | `vision/main` aedb2d15 | MERGED | post-merge suites queued (RAM 1.2 GB free, waiting for 4 GB) |
| 2026-11-17 | V3 | fix r1 + review r2 | same implementer (sonnet/medium); same reviewer (sonnet/high) | `vision/diagram` ff8be7c3 | APPROVE WITH NITS | 9 tests incl. reaction+transport fixture with pinned SVG hash |
| 2026-11-17 | V3 | merge | sub-orchestrator | `vision/main` 135cfb64 | MERGED | post-merge suites queued with V2 (new files only; one combined run, VD-12) |
| 2026-11-17 | V1 | fix r3 + review r4 | same / same | `vision/model` cee6bae4 | APPROVE WITH NOTES | VD-11 implemented; 31 tests; identity 152/152 |
| 2026-11-17 | V1 | merge | sub-orchestrator | `vision/main` 4260ea80 | MERGED | pushed; post-merge suites wait for the heavy lock |
| 2026-11-17 | - | PC restart | - | - | - | all agents stopped; lanes re-dispatched; partial V6 work kept in its worktree |
| 2026-11-17 | V4 | implementer | general-purpose (opus/high) | `vision/extend-db`, `C:/t/vis-extend-db` | running | code commit db7896c2; tests wait for the heavy lock (demo `llm-ab`) |
| 2026-11-17 | V5 | implementer | general-purpose (sonnet/high) | `vision/scope`, `C:/t/vis-scope` | fix r1 | static review (sonnet/high) CHANGES_REQUESTED: negation lookahead keeps the depth word |
| 2026-11-17 | V6 | implementer | general-purpose (sonnet/high) | `vision/generate`, `C:/t/vis-generate` | fix r1 | static review (opus/high) CHANGES_REQUESTED: proof never pins T2PW_FINAL_PAYLOAD; flags missing from the cache key. Proof now in two arms (flag on, flag 0), 97/97 each |
| 2026-11-17 | V9 | implementer | general-purpose (sonnet/high) | `vision/bench`, `C:/t/vis-bench` | running | tests wait for the heavy lock |
| 2026-11-17 | V5 | fix r1 + review r2 | same implementer (sonnet/high); same reviewer (sonnet/high) | `vision/scope` d6c2bbdf | APPROVE | 74 focused tests passed, 0 survivors (`logs/v5-4.log`, run by the sub-orchestrator's queue) |
| 2026-11-17 | V5 | merge | sub-orchestrator | `vision/main` e7fa005d | MERGED | condition: `scope.py` reads `vision/prompts/*.txt` at import; the pyproject package-data entry `"t2pw.vision" = ["prompts/*"]` comes with V6 (present on `vision/generate`). Confirm it is merged before any non-editable install or deploy |
| 2026-11-18 | wave 1 + V5 | post-merge suites | sub-orchestrator queue (`suite.sh`, one lock hold, holder vis-suite) | detached snapshot `C:/t/vis-suite-e7fa` at e7fa005d | PASS | SMOKE 511, golden 2/8/4/24, equivalence run_pathway 8 / app_vs_service 14 / default_path 6, vision 153; 0 survivors (`evidence/suites/suite-e7fa005d.txt`) |
| 2026-11-18 | V6 | gated run result | queue_v6.sh | `vision/generate` d93c0748 | partial | PWML 97/97 byte-identical in both arms (final payload on, and =0); 2 focused tests fail: paper-only PWML has 0 reactions vs 1 in SBML/SBGN/KGML (root cause being traced, pwml/ untouched); cache-key test collides with the conftest flag pin |
| 2026-11-18 | V6 | fix r2 + re-run + review r2 | same implementer (sonnet/high); reviewer (opus/high) | `vision/generate` 163a9279 | APPROVE WITH NOTES | root cause (b): strict quarantine drops the paper-only reaction on purpose; SBML/SBGN/KGML now built from the same quarantine output; SBML = clean_model `sbml_text`; re-run: PWML 97/97 identical, matrix 210/210 ok, 0 survivors. Fix r3 before merge: a PWML refusal yields no format at all (F1a) |
| 2026-11-18 | V6 | delta review + merge | ws-reviewer (sonnet/medium) APPROVE | `vision/main` 04596e6c | MERGED | post-merge suites queued, then stopped at the hold (queue state section) |
| 2026-11-18 | V4 | static review | ws-reviewer (opus/high) | `vision/extend-db` 14f3230c | CHANGES_REQUESTED | currency rename turns charged ATP into a hop hub; compartment-blind merge (VD-13); Reactome sets flattened into co-subunits; fix r1 code-only during hold |
| 2026-11-18 | V7a | implementer + static review | general-purpose (sonnet/high); ws-reviewer (sonnet/high) | `vision/checks` 705e62cc | CHANGES_REQUESTED | placements PWML lacks never compared; SBGN regulatory arcs ignored; fix r1 code-only during hold |
| 2026-11-18 | V9 | implementer + static review | general-purpose (sonnet/high); ws-reviewer (sonnet/high) | `vision/bench` 190ba7b0 | APPROVE WITH NITS | 13 tests passed; paper set PAPER_ONLY 155/165 recall; fix r1 (VD-14) code-only during hold; full-view gold needs the DB |
| 2026-11-18 | V4 | fix r1-r2 + static re-reviews | same implementer (opus/high); reviewer (opus/high) r2 CHANGES_REQUESTED, delta check (sonnet/medium) APPROVE | `vision/extend-db` cbc5f120 | APPROVED pending runs | 23 tests written; runs held: v4-unit-3, then v4-live-2 when the DB is back |
| 2026-11-18 | V7a | fix r1-r2 + static re-review | same implementer (sonnet/high); same reviewer (sonnet/high) APPROVE | `vision/checks` 8b8a4b8b | APPROVED pending runs | runs held: 19 tests, then the 97-run proof; reverse-direction SBGN check logged for V7b |
| 2026-11-18 | V9 | fix r1 | same implementer (sonnet/high) | `vision/bench` 09050671 | pending re-check + runs | runs held (4 commands in V9.json) |
| 2026-11-18 | V9 | fix r2 + static re-review | same implementer; reviewer (sonnet/high) APPROVE | `vision/bench` d5732418 | APPROVED pending runs | 27 tests written |
| 2026-11-18 | V7b | implementer + static review + fix | general-purpose (sonnet/high); ws-reviewer (sonnet/high) APPROVE | `vision/layout` ca28897f | APPROVED pending runs | layout metrics (L3 definitions) + PathWhiz gallery on compose project `t2pw-vis`, ports 3040/3347, source `C:/t/pwpr/09` |
| 2026-11-18 | V10 | implementer + static review | general-purpose (sonnet/high); ws-reviewer (opus/high) | `vision/web` 1a492f5d | CHANGES_REQUESTED -> fix r2 | VD-15: keyword-only rescope on request threads; public `generate(offline=...)`; child process, per-process lock; blocker: unbounded lock wait in sync handlers |
| 2026-11-18 | V6 merge | post-merge suites | sub-orchestrator chain (`chain1.sh`) | snapshot `C:/t/vis-suite-0459` at 04596e6c | PASS | SMOKE 511, golden 2/8/4/24, equivalence 8/14/6, vision 163; 0 survivors (`evidence/suites/suite-04596e6c.txt`) |
| 2026-11-18 | V7a, V9, V7b | released gated runs | chain1 | lane branches | ran | V7a: 18/19 tests (1 failure being fixed), 97-run proof ran; V9: 27 tests passed, paper-set proof ran, sk-or- 0; V7b: 16 tests passed, 97-run metrics ran |
| 2026-11-17 | V6 | gated run | sub-orchestrator queue (`C:/t/vis-runs/queue_v6.sh`) | holder vis-generate | waiting | needs the heavy lock free and 4 GB RAM (1.3-1.7 GB free: desktop apps, not a leak per main) |

## Decisions made by the sub-orchestrator (PO gave full autonomy; recommended option each time)

| id | decision | why |
|---|---|---|
| VD-1 | Big payload = frozen `final_mapped.json` as `base`, never edited, plus an `extension` overlay, `provenance`, `hops`, `sources` | an empty overlay is today's output by construction, so the no-regression proof is an equality check |
| VD-2 | `T2PW_EXTEND` defaults off | switching it on changes default output and needs its own A/B; not provable now |
| VD-3 | Scope = `{paper_only, max_hops}`; default = base + hop 1; "more detailed" = hops + 1 | one integer covers paper only / default / full and every re-scope step without re-running |
| VD-4 | No PathWhiz edit planned | D-W28 (b): outside-sourced elements look the same in PathWhiz; edits only if a lane finds a real limit |
| VD-5 | Own SBGN-ML exporter from the view (not PathWhiz's SBGN export) | VISION § 2.5: every format from the same view |
| VD-6 | Move the web-diagram prototype into `src/t2pw/vision/diagram/` | the web view and generation need it in the repo; the prototype is proven (0 selfcheck failures on 97) |
| VD-8 | SBGN interactions: stimulation/inhibition from a closed word list, modulation only for regulates/modulates/modifies/cofactor; binding, component_of, SAME_AS etc. are omitted and counted; entity-to-entity interactions with no process target are omitted and counted (SBGN PD needs a process target; no invented association process) | an arc must not assert regulation the paper did not state (contract § 1) |
| VD-9 | Diagram font: bundling an open font (DejaVu Sans) and `pyproject` package-data for `diagram.css` go to the generation/deployment lane | the 97/97 proof holds on Segoe UI only; the Linux server needs a fixed font |
| VD-10 | Paper-only view drops rows whose only lineage is `audit_modified` (7 reactions in 152 runs) | conservative; V7 can promote a row whose quote is verbatim in the paper |
| VD-11 | Reaction merge identity: forward match merges when either row lists no cofactors, cofactors are equal, or differ only by NAD/NADP variants; a reversed match merges only when both rows list cofactors and they mirror | a wrong merge loses a step and attaches a false database confirmation; a missed merge only shows a step twice |
| VD-12 | Wave-1 lanes add only new files, so SMOKE, golden-seam and equivalence run once on the combined merge result instead of after each lane | the Thursday demo has priority on the heavy lock and RAM; a per-lane run adds no information when no existing file changes |
| VD-13 | DB-extension merge across compartments: merge a database step into a paper step only when one side has no compartment or both fold to the same compartment (GO id, else a small explicit synonym table); otherwise keep it apart and count it; currency compounds keep their family name, never a source suffix; Reactome complexes are emitted without components unless their subunits contain no nested set | VD-11 principle (under-merge wins); V4 static review F1-F3 |
| VD-14 | Benchmark headline = macro average across cases (micro secondary); interaction matches tiered exact / weak / contradiction, only exact counts toward recall | 29 of 33 gold legs are one paper; relaxed matching was tuned on one run (V9 static review F1, F3) |
| VD-15 | Web (V10): re-scope on a request thread uses the keyword fallback only (no LLM call in a request); `generate()` gets a public `offline` keyword (default True = old behaviour), the serialised child passes offline=False; single uvicorn worker assumed; cache path versioned | main-orchestrator rulings (own process, production DB, no env swap) and app.py's "no LLM on a request path" |
| VD-7 | Per-job extension budget defaults: 3 curated pathways, 5 outside papers, 120 s | VISION § 7 open point; conservative until the PO sets numbers |

## What I saw in the screenshots (2026-11-18, gallery from V7b chain3: runs 7, 10, 14, 116, 126, 128)

- **Our renderer.** Run 7 (Arabidopsis, 23 reactions) reads well: lanes per compartment, main chain left to right,
  enzymes beside their steps. Run 116 (sphingolipids) is drawn very wide; at page width the labels are unreadable
  (aspect-ratio metric requested from V7b).
- **PathWhiz (local harness, t2pw/09).** Chains read left to right and imports succeed. Problems seen:
  (1) one compartment box per biological state, so the same location appears twice (ER x2 in 116, plastid x2 in 7);
  (2) in run 10, compartment boxes overlap, one box has no label, and long transport lines cross boxes;
  (3) compound structure images are wrong (5-HT drawn as a diamine, sphingomyelin as a steroid). A scout traced (3)
  to the harness, not our PWML: the harness gives new compounds sequential `pwc_id`s, and the viewer loads
  `moldb.wishartlab.com/molecules/<pwc_id>/image.png`, i.e. an unrelated production compound. Our PWML carries correct
  ids (serotonin HMDB0000259, ChEBI 28790) and no structure.
- **Grounding gap seen:** Ca++ in run 10 has no database id.
- Items (1) and (2) are layout issues in how PWML biological states become PathWhiz boxes. **Taken by the main
  orchestrator** as a separate correctness-track lane after Thursday (pwml/ layout change, gated by replay
  no-regression and a PathWhiz import A/B); not part of vision. V7b's metrics (duplicate_compartment_boxes,
  compartment_box_overlaps, unlabeled_compartment_boxes) are kept as that lane's baseline.
- **Grounding item (logged):** Ca++ in run 10 carries no database id.

## Rulings from the main orchestrator (2026-11-18) and carried constraints

- **V10 / server:** the server runs `generate` with production DB resolution; offline (no-DB) drops are acceptable for
  tests only. `generate()` runs serialised in its own process; no process-wide env swap in the server. No exporter-side
  hook. The quarantine step and the PWML export step must always see the same DB availability (V6 review F2).
- **V7 cross-format check item:** PWML writes canonical (mapped) names while SBML/SBGN/KGML write the stored payload
  names (V6 review F1b); the consistency check must compare by identity, and report name differences separately.
- **Deploy:** `vision/prompts/*.txt` and `diagram/fonts/*` ship only through the pyproject package-data entries added by V6.

## Deferrals

See `PLAN.md` § 6.

## Open questions

None blocking. VISION § 7 points are handled by VD-4 and VD-7 until the PO rules.
