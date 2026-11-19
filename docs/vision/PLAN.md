# Vision programme: plan

Mandate: `docs/VISION.md` §§ 2-4 and D-W28. Integration branch `vision/main`, cut from `origin/main` `a6c07e16`.
Lane branches `vision/<lane>` in worktrees `C:/t/vis-<lane>`. `vision/main` is pushed, never merged into `main`
or `webserver/refactor` by this programme (the main orchestrator does that after review).
Live record: `docs/vision/PROGRESS.md`. PathWhiz edits (if any): `docs/vision/PATHWHIZ-EDITS.md`.

## 1. What already exists (reused, not rebuilt)

| Need | Existing piece (paths under `src/t2pw/`) |
|---|---|
| Paper payload (the base of the big payload) | `final_mapped.json`, frozen by `service/postpipeline.py` `freeze_canonical_payload`; final-payload layer `service/final_payload*.py` |
| Per-element provenance | `provenance_lineage` entries (`pipeline/lineage.py`): `origin` = `paper_stated` / `inferred` / `rag_literature` / `database_grounded` / `audit_modified` / ..., with `stage`, `sources`, `reason`; `evidence` quote strings; `rag_provenance` (`rag/provenance.py`) |
| Typed model with value-level provenance | `clean_model/` (`build_from_payload`, `to_payload`, labels paper/db/inferred/default/unknown) |
| Exporters | PWML: `pwml/ir.py` `build_pwml_ir` + `pwml/writer.py` (layout v2); SBML: `sbml/json_to_sbml.py` (production) and `clean_model/export/sbml.py` (pure); KGML: `clean_model/export/kgml.py` |
| Other papers | `rag/acquire.py` (Europe PMC, NCBI, Crossref, S2, bioRxiv; full text), `rag/extract.py`, `rag/admission.py` |
| PathBank pathway content | `bench/eval_v2/pathbank_gold.py` `extract(conn, PW id)` (reactions, enzymes, sub-pathways); read-only DB user via `PATHBANK_DB_*` |
| Web diagram | prototype `C:/t/ev-diagram-v2` (`t2pw_diagrams.py` 1654 lines, `metrics.py`, `selfcheck.py`; input = PWML IR), outside the repo |
| Validators | `libsbml` installed in `.venv`; `lxml` installed (XSD + ISO schematron) |
| PathWhiz harness | `C:/t/pwi/t2pw/tools/pathwhiz_import/run_import.py` (compose project hard-coded `t2pw-pwi`), PathWhiz stack `t2pw/01..09` in `C:/t/pwpr/` |
| Replays and gold | `C:/t/corr-assess/scripts/dump_replay.py` (97 exporting runs of the 152), `bench/gold/pinned_v1.json`, `bench/eval_v2/paper_gold.py`, `pathbank_gold.py` |

Missing: Reactome and WikiPathways clients; a runtime PathBank pathway query; a big-payload container; a scope
layer; on-demand generation; an SBGN-ML exporter; the diagram renderer inside the repo; cross-format and
faithfulness checks; attractiveness metrics beyond crossings and enzyme distance.

## 2. Design (fixed here so lanes can run in parallel)

### 2.1 The big payload = the frozen paper payload plus an overlay

```
big_payload.json = {
  "schema": "t2pw.big_payload/1",
  "base": <final_mapped.json, byte-for-byte unchanged>,
  "extension": <rows in the same shape as base sections: entities.*, processes.*, element_locations.*, biological_states>,
  "provenance": { <element key>: [<provenance record>, ...] },   # for base AND extension rows
  "hops": { <element key>: <int> },                              # graph distance from the paper core
  "sources": { <source id>: {kind, pmid, pmcid, doi, title, db, accession, url, retrieved} }
}
```

- **The base is never edited.** Everything new lives in the overlay. A big payload with an empty overlay is the
  current output, by construction.
- **Element key**: a deterministic string, `<section>:<normalized name or content signature>` (process rows reference
  entities by name today). V1 defines it; it must be stable across reloads.
- **Provenance record**: `{kind: paper | other_paper | database | inferred | curation, source: <source id or null>,
  quote: <str or null>, stage: <lineage stage or extension stage>, note: <str>}`. Mapping from today's fields:
  `provenance_lineage.origin` `paper_stated` -> paper; `rag_literature` -> other_paper (sources -> citation);
  `database_grounded` -> database; `inferred` / `audit_modified` -> inferred / curation; a row with no lineage but an
  `evidence` quote -> paper (`note: "no lineage, evidence quote"`). An element may carry several records (a paper
  reaction also confirmed by PathBank).
- **Hops**: base rows are hop 0. An extension row is hop *n* if its shortest path to a hop-0 row through shared
  grounded entities is *n* steps.

### 2.2 Scope and views

```
ScopeSpec = { "paper_only": bool, "max_hops": int | null }
  paper only : paper_only=true                -> base rows with a `paper` record, plus the entities they reference
  default    : paper_only=false, max_hops=1   -> all base rows + extension rows with hop <= 1 ("immediate context")
  full       : paper_only=false, max_hops=null-> everything
view(big, scope) -> payload dict in the final_mapped schema (reference-closed: no dangling names)
```

- **Invariant (no regression):** `view(big, default)` on a big payload with an empty overlay returns a dict equal
  to `base`. With the extension flag off, every exporter sees exactly today's input.
- **Re-scoping** never re-runs the pipeline. "More detailed" = `max_hops + 1` (or `full`); "only what the paper
  says" = `paper_only`. The scope chooser (V5) is an LLM call with a deterministic fallback to `default`.

### 2.3 Generation on demand

`generate(big, scope, fmt)` builds the view once and hands the same dict to every exporter: PWML (existing
writer, layout v2), SBML, SBGN-ML (new), KGML, web diagram (SVG/HTML). Output is cached by
`(sha256(view), fmt)`. All formats come from one view, so a cross-format check (V7) can compare them.

### 2.4 Flags

- `T2PW_EXTEND` (default **off**): runs the extension stage after freeze and writes the overlay. Off = no network,
  no new artifact, pipeline output unchanged.
- The view and generation code are pure functions over stored files; they change nothing in the default pipeline.
  Switching `T2PW_EXTEND` on by default needs its own A/B (gold + 97 replays + faithfulness) and is not in this plan.

## 3. Lanes

| Lane | Branch | What | Depends on | Implementer (model/effort) | Review |
|---|---|---|---|---|---|
| V1 model | `vision/model` | `src/t2pw/vision/{payload,view}.py`: big-payload container, element keys, provenance mapping from `provenance_lineage`/`evidence`/`rag_provenance`, hops, `view()`, `ScopeSpec`; identity proof on all 152 stored payloads | - | general-purpose as implementer (opus/high): interface design, behaviour-critical | opus/high |
| V2 sbgn | `vision/sbgn` | `src/t2pw/vision/export_sbgn.py`: SBGN-ML 0.3 Process Description from a payload view (compartments, macromolecule/simple chemical/complex glyphs, process nodes, consumption/production/catalysis/stimulation/inhibition arcs; layout from the PWML IR coordinates when present); dev validation against the libSBGN XSD + schematron | - | general-purpose (sonnet/high) | opus/high |
| V3 diagram | `vision/diagram` | Move the `C:/t/ev-diagram-v2` renderer (`t2pw_diagrams.py`, `diagram.css`, `metrics.py`, `selfcheck.py`) into `src/t2pw/vision/diagram/`; entry `render_svg(ir) -> str`; selfcheck as a test; drop the patch/explore scripts | - | general-purpose (sonnet/medium): mostly a move | sonnet/high |
| V4 extend-db | `vision/extend-db` | `src/t2pw/vision/extend/{pathbank,reactome,wikipathways,merge}.py`: find candidate curated pathways (organism + overlap of grounded ids), fetch their steps (PathBank via the read-only DB, reusing `pathbank_gold` readers; Reactome ContentService REST; WikiPathways REST/GPML), convert to overlay rows with `database` provenance, merge by identity (never duplicate a base element; a match adds a provenance record), compute hops. Cached, budgeted, fail-open. Flag `T2PW_EXTEND` off | V1 | general-purpose (opus/high): biology-sensitive merge | opus/high |
| V5 scope | `vision/scope` | `src/t2pw/vision/scope.py`: `choose_scope(request_text)` and `rescope(current, message)` through the existing LLM client (new prompt file inside `vision/`, no edit to `llm/prompts/`), JSON answer validated to `ScopeSpec`, fallback `default` | V1 | general-purpose (sonnet/high) | opus/high |
| V6 generate | `vision/generate` | `src/t2pw/vision/generate.py`: `generate(big, scope, fmt)` for pwml / sbml / sbgn / kgml / svg; cache; CLI `python -m t2pw.vision.generate <run_dir> --scope --fmt`; proof that `generate(big0, default, pwml)` == today's export bytes on the 97 replays | V1, V2, V3 | general-purpose (sonnet/high) | opus/high |
| V7 checks | `vision/checks` | Development checks, `src/t2pw/vision/checks/`: libSBML validation; SBGN XSD + schematron; cross-format consistency (parse PWML, SBML, SBGN back; same reactions, entities, compartments); faithfulness (paper view: every element's quote is in the paper text; extended view: every database element re-found in its record, every other-paper element's quote in the cited text); attractiveness metrics (overlaps, crossings, reading-order violations, label fit, duplicated compounds, enzyme distance) on PWML coordinates and on our SVG; gallery script (PathWhiz harness under compose project `t2pw-vis` + our renderer) | V6 | general-purpose (sonnet/high) | opus/high |
| V8 extend-lit | `vision/extend-lit` | Other peer-reviewed papers for the extension: queries for the steps next to the paper core through `rag/acquire` (abstract or full text), evidence-bound extraction through `rag/extract` + `rag/admission`, `other_paper` provenance with citation and quote | V4 | general-purpose (opus/high) | opus/high |
| V9 bench | `vision/bench` | `docs/vision/BENCHMARK.md` + scorer: main set = paper-only view against paper gold (score only what the paper covers); small full-view set = full view against the curated pathway (PathBank gold via `pathbank_gold`); reuse `bench/eval_v2` | V1 | general-purpose (sonnet/high) | sonnet/high |
| V10 web | `vision/web` | Job page: view at a scope, a re-scope box ("make this more detailed"), download buttons that call `generate`. Thin routes over V5/V6 only | V5, V6 | general-purpose (sonnet/high) | opus/high |

Order: wave 1 = V1, V2, V3 (parallel, no heavy runs). Wave 2 = V4, V5, V9. Wave 3 = V6, V8. Wave 4 = V7, V10.
At most 3 implementers at once. The data model (V1) merges before anything that reads it.

## 4. Gates (every lane, on the merge result in `vision/main`)

1. Dependencies merged; diff inside the lane's file boundary (new files under `src/t2pw/vision/`, `tests/test_vision_*.py`,
   `docs/vision/`; anything else is named in the brief).
2. The lane's own tests, plus SMOKE 511, golden-seam 38 (2/8/4/24), and the 3 equivalence files each alone, through
   `bounded_run.py --heavy-lock`, `PYTHONPATH=<tree>/src`, `T2PW_OFFLINE_CURATOR=1`, fresh `--basetemp`, 0 survivors.
3. **No regression.** With `T2PW_EXTEND` off and the default scope: `view(big0, default) == base` on all 152 stored
   payloads, and PWML bytes from `generate` equal today's replay bytes on the 97 replays (from V6 on). No existing
   file under `pipeline/`, `pwml/`, `mapping/`, `rag/`, `curation/`, `llm/prompts/` is changed by V1-V7, V9, V10.
   Anything that cannot be shown not to be worse is deferred.
4. Independent review of the actual diff (opus/high unless marked), no edit role. Two rejections = escalate one tier.
5. Evidence committed under `docs/vision/evidence/<lane>/`; `PROGRESS.md` row; logs checked for `sk-or-`.

## 5. Machine and priority rules

- Until Thursday 2026-11-19 evening the Thursday demo (showcase agent, vLLM/Gemma A/B) has priority on the heavy lock
  and RAM. Wave 1 is code plus light unit tests; SMOKE and replays wait until the lock is free, one job at a time,
  never chained for long.
- At least 4 GB free RAM before Docker or a heavy run. No `python -`, no stdin heredocs, no `&` with
  run_in_background. Kill only our own PIDs and the `t2pw-vis` compose project.
- Live LLM runs only through a launcher that calls `dotenv.load_dotenv(<primary .env>, override=False)` (pattern:
  `C:/t/corr-assess/scripts/live_launch.py`). Never read or print `.env`.

## 6. Deferred (and why)

- **Extension on by default.** Needs an A/B showing no loss at the default scope; not in this plan.
- **PathWhiz edits.** D-W28 (b): outside-sourced elements look the same as paper elements in PathWhiz, so no change is
  needed for provenance. Any PathWhiz edit found necessary goes on `t2pw/10-*` from `t2pw/09` (no `t2pw/latest` on the
  remote yet) and is logged in `PATHWHIZ-EDITS.md`.
- **BioPAX.** VISION § 3 says "if we add BioPAX"; not added.
- **SBML/SBGN third-party viewer screenshots** (VISION § 3): after V7, if a viewer runs headless here.
- **Reversibility and stoichiometry** (CORRECTNESS-FIXES § 4): unchanged.
- **Production use of the validators**: development only (D-W28 c).
- **Limits per job** (outside papers per pathway, time and cost): V4/V8 take a budget parameter with conservative
  defaults (3 curated pathways, 5 outside papers, 120 s); the PO's numbers replace them when given.
