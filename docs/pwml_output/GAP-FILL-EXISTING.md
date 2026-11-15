# Existing gap-fill pieces (read-only map, 2026-11-10)

Input for D-W12 and D-W16 (final payload; gap filling from databases, RAG and related papers). Mapped by an Explore agent (sonnet/low) at `e7f91884`; not re-verified line by line. Paths relative to `src/t2pw/`.

**1. RAG today**
- RAG is not limited to the submitted paper. `rag/acquire.py` runs `search_candidates` (:800) against Europe PMC search (`_EUROPEPMC_SEARCH` :60, `_fetch_europepmc` :453), NCBI esearch (`_ncbi_esearch_pmids` :500), Crossref (:601), Semantic Scholar (:650) and bioRxiv (:696).
- `fetch_full_text` (:994) downloads full text via the core `_europepmc_full_text`.
- The queries come from the pathway context and the gap terms (`build_query` :388).
- It runs after Stage 1 extraction, from `service/extraction.py:770` (`maybe_run_rag`) and `:1014` (`run_rag_rounds`). Both are defined in `service/rag_orchestration.py` (:117, :951).
- `rag/triage.py` (`should_run_rag`) decides whether to run. Gap-based auto-start is deliberately restricted (comments at :56-74). RAG also runs when the user sets the flag.
- `rag/controller.py` is the bounded loop; `max_rounds` defaults to 1 (`rag/loop_policy.py:143`).
- `rag/retrieve.py` has `detect_gaps` (:656) and `retrieve_evidence` (:1113). `rag/synthesize.py` builds reactions, and `rag/admission.py` is the admission gate.
- RAG does add content. `maybe_run_rag` and `rag_round_merge` (:864) merge synthesized reactions and entities into the payload.
- "rag_prose_extraction" is an LLM stage name (`llm/client.py:350`, `rag/extract.py:144`). It turns a retrieved prose passage into structured reactions. It is evidence-bound and opt-in (`RAG_EXTRACT_REACTIONS`), and it fails closed.

**2. Paper finding and name mode**
- The finders are the `rag/acquire.py` functions above and `batch/fetch.py:536` (`fetch_papers`).
- I found no citation or PathBank-reference traversal in the files I opened.
- Name mode: `service/driver_bridge.py:160` calls `inputs.resolve_name`. That function (`service/inputs.py:95`) builds a spec from the name and calls `fetch_papers` for open-access full text. `NAME_MODE_COUNT = 1` (:74), so one paper is fetched.
- The name-mode text is therefore a retrieved paper's full text. It then goes through the same pipeline as a submitted paper.

**3. Completeness and gap detection**
- `curation/completeness_audit.py:303` (`run_final_completeness_audit`) is an LLM check of the final payload against the locked reaction manifest and the source text. It is non-mutating and triggers no retrieval.
- `curation/audit_json_llm.py`, `curation/apply_audit_patch.py` and `curation/pathway_curator.py` form an audit and patch loop. It is driven from `service/postpipeline.py` around :1130.
- `curation/gap_resolver.py` (`run_gap_resolution`, called at `service/postpipeline.py:1144`) fills missing species, locations, IDs and complex components. It uses the DB and an LLM, and does not fetch papers.
- `pipeline/qa_graph.py` finds orphan components, dangling nodes and entities that are present but unused. It is read-only.
- `rag/retrieve.py:656` `detect_gaps` finds dangling reactions, orphan metabolites and unmapped enzymes. It is the only detector that feeds retrieval.
- Stage 2 inference is `pipeline/pipeline.py:386` (`run_inference_pipeline`, prompt `pwml_infer_system.txt`). It uses no retrieval.

**4. Freeze and mapping**
- Order in `service/postpipeline.py`: `map_payload` (:1386) → `run_enrichment` (:1427) → `freeze_canonical_payload` (:379, called :1501).
- Freeze means quarantine, gates, hash and serialize. `CANONICAL_PAYLOAD_KEY` is at `pipeline/gate_reports.py:141`.
- `batch/driver.py:1801` (`_add_identity_artifacts`) writes `final_mapped.json` from that canonical payload. `merged_payload.json` is the pre-mapping version.
- RAG rounds run before all of this. They have their own scratch re-mapping (`rag_round_mapping_cache_path`, :664).
- Any new fill stage has to go before `map_payload`, or its output needs its own map and enrich pass.

**5. Provenance**
- Rows can carry `rag_provenance`, `evidence`, `source_papers` and `rag_confidence` (`rag/provenance.py`).
- `rag_provenance` holds `source_id`, `source_uri`, `source_type` and `chunk_id`. `source_type` also allows pathbank, kegg and pwml_example.
- `rag/graph_delta.py:217-226` reads `source_papers`/`source_paper` and `source_refs`.
- Yes, a row can carry a different paper's id, since `source_id` is per chunk.
- `pipeline/payload_models.py` has an `evidence: str | None` field on its row models.
- `pipeline/lineage.py` records provenance.
- I did not check how `source_id` is carried into the final PWML.

**6. Smallest build**
Wrap four existing pieces:
- Detection: `detect_gaps` plus `qa_graph`.
- Fetching other papers: `rag/acquire.search_candidates` and `fetch_full_text`.
- Passage-to-reaction conversion: `rag/retrieve.py`, `rag/extract.py` (`rag_prose_extraction`) and `rag/synthesize.py`.
- Merge and gating: `rag/admission.py`, `rag/graph_delta.py` and `rag_round_merge`.

For DB fills, use `gap_resolver` and `map_ids`. Place the stage before `map_payload`, and name mode feeds in unchanged through `resolve_name`.

Missing entirely:
- No single completeness verdict (a score or threshold) that decides the "large gaps, so go to other papers" case.
- No process-level "is this pathway complete" check against a reference. The `completeness_audit` check is source-text-only.
- No fill from a database of pathway steps. The DB code resolves IDs and locations, not missing reactions.
- No name-only route that skips paper fetch and builds from DB plus RAG alone.
