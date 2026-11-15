# Live runs per mode, final payload phase 5 (`T2PW_FINAL_PAYLOAD=1`, curator live)

Tree C:/t/fp, code HEAD `34cb102a` (code unchanged since `f4b0be9a`). Driver `drive.py`, wrapper `run1.sh`. Input: PMC12782028 / PMID 41522513, "cholesterol biosynthesis", Homo sapiens. Keys loaded by the pipeline itself; `.env` never read by agents or the orchestrator. The machine restarted during the work; the three runs and the first import were already complete, the import was redone by the orchestrator.

| mode | run | status | wall | PWML | fill provenance (`final_payload_fill`) | title / reference | PathWhiz import (harness, h04 = branch 04 tip + owner change) |
|---|---|---|---:|---|---|---|---|
| PathWhiz (strict) | `pathwhiz` | pass | 1424 s | 32,791 bytes, 3 reactions, 5 compounds, 3 proteins | 3 entries: cellosaurus_id for 2 cell lines (kept in the payload only), 1 DrugBank id | NCBI title, PMID 41522513 reference | success, errors {} (25.6 s), 1 reference created |
| research | `research` | pass (report mode: citation report, no PWML by design) | 2678 s | none (research export is a report, `pwml_path` null) | 6 entries: 2 cellosaurus ids, DrugBank, ChEBI, KEGG, CAS | NCBI title and PMID in the final payload | not applicable |
| name only | `nameonly` | FAIL `no_reactions` at stage 1 (337 s): the paper the name search picked gave a pathway with no reactions or transports; this stage runs before any final-payload code | 337 s | none | n/a | n/a | n/a |
| name only, retry | `nameonly_r2` | pass | 864 s | 8,343 bytes (a different paper was picked: PMID 42997905, vitamin D receptor and farnesyl diphosphate synthase in esophageal carcinoma cells; 1 state) | 1 entry: tissue ontology id (esophagus, BTO) | NCBI title and PMID 42997905 (name mode takes the fetched paper's title because a paper was confirmed) | success, errors {} (8.8 s) |

Checks on the two PWML files: species-only states 0, wildcard-bindable states 0; layout counters recorded in `*.analysis.json` (for example `cell_types_dropped_no_id` 3 in the PathWhiz run: the two cell lines and one more id-less cell type are dropped from the PWML only and stay in the payload). Both PWML files import with no error message.

Notes and limits
- The name-only mode is nondeterministic in which paper it fetches; one of two runs failed in stage 1 with no relation to the flag. Retried once, as briefed.
- Only one paper (PMC12782028) was used for two modes; name-only landed on an unrelated paper, so its PWML is small (1 state) and says little about quality.
- One live run per mode is a smoke test of the seam with the flag on, not a no-loss proof. The no-loss argument for the default flip rests on: flag-off byte identity (replay), the replay measurements of P4/P5, and these seam runs.
- Research mode produces no PWML, so the PWML-specific counters could not be checked there; the fill reached the final payload (6 entries).
- No text-format export other than the research report was produced in these runs.
