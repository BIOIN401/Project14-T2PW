DUPLICATION ASSESSMENT (read-only; nothing in the repo changed; no pytest, pipeline or LLM run)

NOT ESTABLISHED
- Which pooled draw produced each H+/OTOP1 transport. Per-row lineage only says `paper_extraction`; the per-draw payloads aren't committed. So I can attribute the pair to Stage-1 extraction, not to a draw.
- Whether the 3-draw pooling causes the surge in near-copy transports. It correlates (3-draw era vs earlier), but 3 of the 13 recent runs are the transport-heavy sour-taste paper, which confounds it.
- Effect of any merge on scores or verdicts. No re-score was run. Counts below are hypotheses.
- Why the writer emits same-state repeat locations (`writer.py` ~1595-1612 and ~2456-2493 is where I'd start). I didn't trace it to a line.
- Kind c is by static inspection only. Only the e2e5 PNG is committed; the rest is derived from payloads.

CORPUS
192 `final_mapped.json` scanned: runs_papers 13, runs_validation 11, runs_verify 129, runs_smoke 23, runs 14, e2e4 and e2e5 1 each. e2e3 and e2e3c have none.
- Excluded: no abandoned or duplicate copies existed (0 skipped).
- 152 have `quarantine_ok=true` (canonical). All headline numbers below use these only.
- 8 refused (pre-quarantine fallback) and 32 older runs with no quarantine report are tabulated separately.

THE EXAMPLE (e2e5)
- Both transports ("H+ influx into Type III cell", "H+ influx through OTOP1") are already in `stage1_payload.json`, with `provenance_lineage` stage `paper_extraction`.
- The extraction boundary report shows 3 draws, with 1, 1 and 2 transports.
- They are identical through merge and mapping. The only difference is location strings: the first has from=`__auto_state__` and to="Type III cell state"; the second has saliva to "cytosol of Type III cell".
- Nothing merged them because every merger keys on exact state strings:
  - Stage-1 pool: `stage_one_ensemble.py:567-569`
  - `dedupe_processes`: `process_normalizer.py:4252-4258` (the run's report shows `dedupe_removed_transports=0`)
  - The identity merge, `merge_on_mapped_identity` (`stage_one_ensemble.py:1292`), handles reactions only.
- The PWML has one H+ record (40034) and one OTOP1; both transports reference them. The duplicate tiles are a renderer property: `draft_graph_render.py:5-8` draws every process as its own mini-diagram, so participants repeat per tile.

IDENTITY RULE TESTED AGAINST GOLD
Five PathBank PWMLs in `reference/` (PW000001, PW012926, PW176888, PW687775, PW741182):
- Zero same-name or same-accession records.
- Compartment is a location row, not a record. 21 of 85 located entities sit in 2 or more states.
- PathWhiz deliberately redraws: Hydrogen Ion has one record with 3 drawings in one state (PW012926) and 2 in another (PW176888). Water, ATP, Cl- and Na+ do the same.
- Verdict: your rule is wrong for entities. Record identity is the molecule alone; compartment and drawings are location rows. It is right for processes, where compartments matter.
- Binding rulings: D-035 and F-043 say a shared accession is evidence, not proof, and that no merge happens without matching identifiers, no conflicting identifiers, and no conflicting attributes. D-034 and D-036 bar any silent merge after the freeze. D-104 §3c and D-109 allow reaction merges on database ID, with nested or undefined locations compatible and a review flag.

TABLE (canonical runs, n=152, 727 processes, 2297 entities)

| Kind | Cause | How common | Correct behaviour | Safe to fix? |
|---|---|---|---|---|
| a1 exact process | alias names, same accession (`dedupe_processes` is name-keyed) | 2 runs, 3 rows (0.4%) | merge | only via fixing kind b |
| a2 near-copy, other state label | transport merge keys are exact-state in 3 places (above), and the identity merge skips transports | 14 runs (9%), 40 rows (5.5%). Since 2026-10-20: 8 of 11 runs with transports, 17 of 36 transports. Before: 1 of 70 | merge when D-104/D-109 compatibility holds, flagged | NO. Changes pipeline logic and needs a D-entry |
| a3 same participants, other enzyme or transporter | same | 13 runs, 24 rows | often legitimately separate | no, don't touch |
| b same entity, 2 records | 0 same-name; 19 runs / 25 records share an accession (1.1%) | | merge only if D-035 proof holds | NO. Same-accession is unsafe, see below |
| c entity redrawn | renderer tiles per process (39% of tile slots in the canonical runs, 94.7% of them affected); PWML same-state redraws are 17.1% of location rows vs 6.9% in gold | | by design | no code change warranted; a layout question |
| d same molecule in several compartments | 35% of runs | | correct, keep | n/a |

Shared accession is NOT identity (observed):
- hordatine A and C: identical IDs, two different molecules.
- "compound 1" and "compound 2": identical IDs.
- holo-EntB shares apo-EntB's UniProt, and "enterobactin synthase" shares EntE's: a modified form and a complex vs its subunit.
- DAAM vs DAAM1.
- Only 5-HT/serotonin and CoA-SH/coenzyme A look like true aliases.

Hazards: isoforms, stereoisomers, class names, and wrong mapped IDs (as above).

PLAIN ANSWER ON H+
No pathway file models individual ions; a record is a pool of a species. Two H+ records in one pathway is never right, and none of the 152 canonical runs has one. PathWhiz itself draws H+ several times from one record, to keep each reaction readable. So "reuse the same H+" is already true in the data. What you see twice is two drawings, or two near-identical transports. A new record is justified only for a different molecule or form, which needs a different database identity.

WHAT MERGING WOULD CHANGE
- Scores: Priority 2 counts matched row pointers (`bench/semantic.py:720`) and R-D092-2 notes duplicates are rewarded, so merging can move true positives and precision.
- Verdict floors: on gold-case runs, merging near-copies leaves distinct reactions above the gold floor. PMC12421875 has 8 distinct against a floor of 7, so the margin is one.
- Evidence loss: a merge folds the loser's evidence into the survivor (D-104 §3(d)). Provability of no loss is not established, so under your hard rule this means defer.
- Standing positions: PMC12452463, PMC13231680 and PMC12180156 had no duplicate hits.

OPTIONS, in my order of preference
1. Do nothing in code and treat the PNG tiles as a presentation matter. Cost: zero. It changes no biology. Optional later: dedupe the diagram only, and leave the data alone.
2. Extend the existing D-104/D-109 compatibility rule to transports (cargo plus transporter set, undefined or nested state, review flag). Cost: a pipeline-logic change with a new D-entry, an N>1 draws gate as for reactions, and re-baselining the goldens. Risk: lost evidence and score movement. Changes pipeline logic, so it needs your decision.
3. Accession-keyed entity merge. I recommend against it: the over-merge examples above, and it conflicts with D-035/F-043.
4. First instrument the lineage: per-row draw id at pooling, so duplicates can be attributed to a stage before any merge is designed. No biology change.

Scripts and tables are in `C:/t/ev-dedup/`.
- Scripts: `scan_runs.py` (needs `pwml_dup.py` alongside), `summarize.py`, `era.py`, `floor.py`, `hplus.py`, `gold_redraw.py`.
- Output tables in `C:/t/ev-dedup/out/`: `runs.jsonl`, `summary.json`, `era_summary.json`.

CONFIDENCE
High on the trace and the gold structure. Medium on the frequency numbers: the 3-draw era is a small sample (13 runs). What would change my mind: a fresh 3-draw run set showing transport near-copies are rare, or a proven non-lossy merge with before/after scores.
