# Vision programme: progress

Plan: `docs/vision/PLAN.md`. Branch `vision/main` from `origin/main` `a6c07e16`. Sub-orchestrator writes no code.

## Current state

- 2026-11-17: plan written. Wave 1 (V1 model, V2 sbgn, V3 diagram) dispatched. Heavy lock held by the Thursday demo
  (`vllm-wire`); free RAM about 1 GB at planning time, so wave 1 runs light unit tests only.

## Dispatch table

| when | lane | role | agent (model/effort) | branch / worktree | state | evidence / note |
|---|---|---|---|---|---|---|
| 2026-11-17 | plan | scout x2 + follow-up | ws-scout (sonnet/medium) | read-only, `C:/t/vis-main` | done | payload, exporters, lineage fields, harness, gold, env mapped |

## Decisions made by the sub-orchestrator (PO gave full autonomy; recommended option each time)

| id | decision | why |
|---|---|---|
| VD-1 | Big payload = frozen `final_mapped.json` as `base`, never edited, plus an `extension` overlay, `provenance`, `hops`, `sources` | an empty overlay is today's output by construction, so the no-regression proof is an equality check |
| VD-2 | `T2PW_EXTEND` defaults off | switching it on changes default output and needs its own A/B; not provable now |
| VD-3 | Scope = `{paper_only, max_hops}`; default = base + hop 1; "more detailed" = hops + 1 | one integer covers paper only / default / full and every re-scope step without re-running |
| VD-4 | No PathWhiz edit planned | D-W28 (b): outside-sourced elements look the same in PathWhiz; edits only if a lane finds a real limit |
| VD-5 | Own SBGN-ML exporter from the view (not PathWhiz's SBGN export) | VISION § 2.5: every format from the same view |
| VD-6 | Move the web-diagram prototype into `src/t2pw/vision/diagram/` | the web view and generation need it in the repo; the prototype is proven (0 selfcheck failures on 97) |
| VD-7 | Per-job extension budget defaults: 3 curated pathways, 5 outside papers, 120 s | VISION § 7 open point; conservative until the PO sets numbers |

## Deferrals

See `PLAN.md` § 6.

## Open questions

None blocking. VISION § 7 points are handled by VD-4 and VD-7 until the PO rules.
