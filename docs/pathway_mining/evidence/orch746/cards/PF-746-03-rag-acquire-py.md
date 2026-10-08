# PF-746-03 — rag/acquire.py: the PubMed parser takes a cited reference's PMCID/DOI

Status: REGISTERED, not dispatched. Production is frozen (D-102); dispatch needs a product-owner decision.

## Defect

`_parse_pubmed_article` (`src/t2pw/rag/acquire.py:523-559`, at b4bb0b74) reads article ids by searching every descendant of the whole `PubmedArticle` node:

```python
for aid in node.findall(".//ArticleIdList/ArticleId"):
    ...
    if id_type == "doi":
        doi = text
    elif id_type == "pmc":
        pmcid = text if text.upper().startswith("PMC") else f"PMC{text}"
primary_id = pmcid or pmid or doi
```

(`acquire.py:536-543`)

`.//ArticleIdList` matches the record's own `ArticleIdList`. It also matches the `ArticleIdList` of every cited reference in the record's reference list.

The loop overwrites `doi` and `pmcid` on each match, so the last cited reference that has a `pmc` or `doi` id wins:
- The wrong PMCID becomes `CandidatePaper.id` (`:543`, `:553`).
- The wrong DOI becomes `source_uri` (`:546-547`).
- If the record has no PMCID of its own but a reference does, the reference's PMCID replaces the PMID (`:524`) as the id.

Downstream effects:
- `fetch_full_text` (`:994-1031`) resolves the id through `_resolve_europepmc_item`, which turns a `PMC…` id straight into a EuropePMC item (`:961-963`). A wrong PMCID therefore fetches another article's full text.
- `identity_keys` (`:161-170`) and `_dedupe` (`:781-794`) also key on this id.

No other production code reads `ArticleIdList`. `git grep ArticleIdList b4bb0b74` finds only `acquire.py:536` and the test fixture `tests/test_rag_acquire.py:78-96`.

## Observed impact

Observed in mining pilot-002, which used this parser for NCBI records. `src/t2pw/rag/acquire.py` is identical at pilot-002's code SHA `ecf6b9a0` and at b4bb0b74 (empty `git diff`).

- **Wrong ids.** 7 NCBI paper ids were provably wrong (PILOT-002-AUDIT.md:10). The pilot-003 result table records pilot-002 as "7 of 8" wrong NCBI ids (PILOT-003-RESULT.md:24).
- **Wrong full texts.** For 2 included papers the full-text fetch returned another article, and extraction read it under the included paper's title (PILOT-002-AUDIT.md:10, :268-269, :280):
  - the NAD+ paper got a cotton-bollworm effector paper;
  - "Arabidopsis GH3.10 conjugates jasmonates" got a 2016 wound-amidohydrolase paper.
- **Wrong id on the primary reaction.** The run's only primary reaction cites `PMC:PMC411572` with URL `doi.org/10.1073/pnas.76.9.4350`, a 1979 PNAS article (PILOT-002-AUDIT.md:9, :276, :279). PILOT-002-RESULT.md:33 identifies it as a cited reference (Towbin 1979) and gives the true article as PMC8508710. Pilot-003 records the same title under PMC8508710 (PILOT-003-AUDIT.md:501, :764).
- **Duplicates.** 6 articles were discovered twice under two ids (PILOT-002-AUDIT.md:10, table :270-275).
- **Reproduction.** The audit ran the frozen `_parse_pubmed_article` on a synthetic record (PILOT-002-AUDIT.md:264):
  - the record's own ids were pmid `111`, pmc `PMC999001` and doi `10.9999/own`;
  - it had one cited reference with pmc `PMC411572` and doi `10.1073/pnas.76.9.4350`;
  - the parser returned `id "PMC411572"` and `source_uri "https://doi.org/10.1073/pnas.76.9.4350"`.
- **Pilot-003, with the mining mitigation in place.**
  - The result table records 0 of 28 NCBI ids wrong (PILOT-003-RESULT.md:24).
  - The audit calls all 28 "consistent" (PILOT-003-AUDIT.md:13, :735). It also warns that the 19 PMCIDs resolved through EuropePMC are trusted as given and only spot-checked (:777).
  - 20 titles seen in both pilots changed id. The audit states that in every one, the pilot-002 side is an NCBI record carrying a cited reference's PMCID (:13, :752-775).
  - Inferred: 20 changed ids suggests the 7 "provably wrong" in pilot-002 was a lower bound.

Inferred, not observed: both production acquisition paths have the same defect. No production run was examined for this card.
- **Streamlit app.** `streamlit_app.py:475` calls `acquire.search_candidates(pathway_context, status=...)` with the default sources, which include `ncbi` (`acquire.py:69`). `streamlit_app.py:570` then calls `acquire.fetch_full_text(paper)` on each selected paper.
- **Batch runner.** `batch/fetch.py :: fetch_papers` defaults to `search_candidates` and `fetch_full_text` (`:587-588`). It passes `sources` only when one is given (`:720-721`) and fetches full text at `:606-615`. `batch/runner.py` calls it without `sources` (`:2146-2148`, `:2359`).

## Current mitigation in mining

At 82fc69a7, in `src/t2pw/mining/acquisition.py`:
- `PaperSource.search` sends the `ncbi` source to `_ncbi` instead of `search_candidates` (`:365-373`).
- `PaperSource._ncbi` (`:384-431`):
  - It still calls production `_fetch_ncbi` (`:406`) and `_parse_pubmed_article` (`:415`).
  - It re-reads each record's own `./MedlineCitation/PMID` (`:414`).
  - It overwrites the parsed id with the PMID (`:419`) and `source_uri` with `https://pubmed.ncbi.nlm.nih.gov/<pmid>/` (`:420`).
  - It then sets the id to EuropePMC's PMCID for that PMID, or keeps the PMID (`:430`, via `_pmcid_for`, `:459-462`).
- A second guard, `full_text_matches_title` (`:103-113`), runs at `:930-936`:
  - It refuses a full text whose opening does not carry the record's title.
  - It then uses the candidate's own `full_text` if there is one, otherwise the abstract (`:937-939`).
  - It is on for `PaperSource` (`:355`) and off for `OfflinePaperSource` (`:491`).

`docs/pathway_mining/known_limitations.md` at 82fc69a7 does not mention this defect.

## Proposed change

In `_parse_pubmed_article`, read ids only from the record's own id list. Change the path at `acquire.py:536`:

`node.findall(".//ArticleIdList/ArticleId")` → `node.findall("./PubmedData/ArticleIdList/ArticleId")`

`node` is a `PubmedArticle` element (`_fetch_ncbi`, `:589-590`). In the PubMed DTD, the record's own `ArticleIdList` is a direct child of `PubmedData`, and cited references sit under `PubmedData/ReferenceList/Reference`. The DTD itself was not read in this session; this is external knowledge. The hand-trimmed mining fixture `tests/fixtures/mining/pubmed_pmid34638573.xml` @ 82fc69a7 has the same layout. Nothing else changes.

Cache invalidation is out of scope (see Risks).

## Ownership

- `src/t2pw/rag/acquire.py :: _parse_pubmed_article`: line 536 only.
- `tests/test_rag_acquire.py`: one new test (the G9 test below). Existing tests are unchanged.

## Acceptance

- Case 1: a `PubmedArticle` whose own `PubmedData/ArticleIdList` has pmc `PMC999001` and doi `10.9999/own`, followed by a `ReferenceList` with a reference carrying pmc `PMC411572` and doi `10.1073/pnas.76.9.4350`. `search_candidates(..., sources=("ncbi",))` returns:
  - `id == "PMC999001"`;
  - `source_uri == "https://doi.org/10.9999/own"`.
- Case 2: the same record with no pmc or doi in its own list. The result is `id == "111"` (its PMID) and `source_uri == "https://pubmed.ncbi.nlm.nih.gov/111/"`.
- The existing tests `test_search_ncbi_parses_pubmed_xml` and `test_dedup_across_sources_by_identity` (`tests/test_rag_acquire.py:326`, `:345`) pass unchanged. Their fixture (`:66-100`) has no reference list, and its id lists sit directly under `PubmedData`.
- All of `tests/test_rag_acquire.py` passes.
- The diff touches only line 536 of `acquire.py` plus the new test.
- SMOKE (508) passes.

## G9 obligation

This corrects observable behaviour, so it needs a behavioural proof that fails at the base SHA.

New test in `tests/test_rag_acquire.py`, for example `test_ncbi_ids_ignore_reference_list`:
- Use the `clean_env` fixture (`:164-167`), as the existing NCBI tests do.
- Use a fake client in the style of `_FakeClient` (`:122-141`). It returns:
  - for esearch: `{"esearchresult": {"idlist": ["111"]}}`;
  - for efetch: XML with one `PubmedArticle` containing `MedlineCitation/PMID` 111, title "Own title", the record's own `PubmedData/ArticleIdList` (pubmed 111, pmc PMC999001, doi 10.9999/own), and **after it** `PubmedData/ReferenceList/Reference/ArticleIdList` (pmc PMC411572, doi 10.1073/pnas.76.9.4350).
  - The order matters. The last match wins at the base, so a reference list placed before the record's own id list would let case 1 pass at b4bb0b74.
- Call `search_candidates(_SEED_CONTEXT, sources=("ncbi",), client=..., cache_dir=<dir>)`. Assert `id == "PMC999001"` and `source_uri == "https://doi.org/10.9999/own"`.
- Second case: the same record whose own list holds only pubmed 111. Assert `id == "111"` and the PubMed URL.
- Give each case its own `cache_dir` (for example `tmp_path / "own"` and `tmp_path / "pmid_only"`) or pass `use_cache=False`.
  - Both cases have the same search cache key: same context, sources and cap (`acquire.py:841-847`).
  - Non-empty results are cached (`:919-927`).
  - With a shared cache dir, case 2 would return case 1's cached `PMC999001` and fail after the fix.

At b4bb0b74:
- Case 1 returns `PMC411572` and the PNAS DOI URL. The pilot-002 audit observed exactly this output from the same parser on the same synthetic ids (PILOT-002-AUDIT.md:264).
- Case 2 should return `PMC411572` and the PNAS DOI URL. That follows from reading `:536-551`, but it has not been run; the implementer must show it failing at the base.
- After the change, both cases pass.

Run it through the bounded wrapper with `--basetemp`.

## Risks and interactions

- **Production callers.** The parser is reached by `search_candidates` → `_fetch_ncbi` → `_parse_pubmed_article`, from two places:
  - the Streamlit RAG path (`streamlit_app.py:475`, `:570`);
  - the batch runner (`batch/runner.py:2359` → `batch/fetch.py :: fetch_papers`, `:587-588`, `:727`).

  NCBI candidate ids and URLs will change, which is the intent. Dedupe against EuropePMC copies and the seed keys on these ids (`acquire.py:754-794`), so duplicate-candidate counts can change. Batch topic counts can change too (inferred).
- **Cache.** The search cache key (`acquire.py:841-847`) has no parser or schema version. `migrate_cached_payload` (`:212-249`) only upgrades schema-1 rows. Search results cached before the fix keep the wrong NCBI ids until refreshed. Whether to invalidate them needs a separate decision (`refresh`, or a `CACHE_SCHEMA_VERSION` bump with a migration branch), and this card does not cover it.
- **SMOKE.** No SMOKE file exercises the parser (TEST_MATRIX.md:259-271 @ b4bb0b74). `test_rag_admission_production_path.py:332` and `test_rag_triage_orchestration.py:192` monkeypatch `search_candidates`. SMOKE will not catch a regression here. The focused file is `tests/test_rag_acquire.py`, which is not in SMOKE.
- **Mining.** `_ncbi` calls `_parse_pubmed_article` and then overwrites `id` and `source_uri`, so its output is unaffected.
  - The mining tests at 82fc69a7 that use the reference-list fixture assert mining's final ids, not the parser's raw output: `tests/test_mining_iteration2.py:279-292` and `tests/test_mining_evidence_hygiene.py:502-537`. They are expected to stay green (inferred; not run).
  - After the fix, the `_ncbi` docstring (`acquisition.py:387-391`) describes behaviour that no longer exists.
  - Removing the workaround is a mining decision, not part of this card.
- **Scope.** The parser also ignores `Article/ELocationID` DOIs (no `ELocationID` in `acquire.py`). That is unchanged here and not part of this card.

## References

- `src/t2pw/rag/acquire.py` @ b4bb0b74: `:69`, `:161-170`, `:212-249`, `:523-559`, `:562-598`, `:754-794`, `:841-847`, `:919-927`, `:954-991`, `:994-1031`
- `src/t2pw/app/streamlit_app.py` @ b4bb0b74: `:475`, `:570`
- `src/t2pw/batch/fetch.py` @ b4bb0b74: `:536-550`, `:587-588`, `:606-615`, `:720-727`; `src/t2pw/batch/runner.py` @ b4bb0b74: `:2146-2148`, `:2359`
- `tests/test_rag_acquire.py` @ b4bb0b74: `:66-100`, `:122-141`, `:164-167`, `:326-354`
- `src/t2pw/mining/acquisition.py` @ 82fc69a7: `:103-113`, `:355`, `:365-373`, `:384-431`, `:459-462`, `:491`, `:930-939`
- `tests/fixtures/mining/pubmed_pmid34638573.xml`, `tests/test_mining_iteration2.py:279-292`, `tests/test_mining_evidence_hygiene.py:502-537` @ 82fc69a7
- `docs/pathway_mining/pilot/002/audit/PILOT-002-AUDIT.md` (experiment/pathway-mining-pilot-002): `:3`, `:9`, `:10`, `:262-282`
- `docs/pathway_mining/pilot/002/PILOT-002-RESULT.md` (experiment/pathway-mining-pilot-002): `:33`, `:39`
- `docs/pathway_mining/pilot/003/PILOT-003-RESULT.md` (experiment/pathway-mining-pilot-003): `:24`
- `docs/pathway_mining/pilot/003/audit/PILOT-003-AUDIT.md` (experiment/pathway-mining-pilot-003): `:13`, `:501`, `:735-738`, `:752-777`
- `docs/pwml_recovery_sprint/ORCH-745-RESULT.md:168` (primary checkout)
- `C:/t/mining-handoff/MINING-OVERNIGHT-HANDOFF.md:98-99`
- `docs/pwml_recovery_sprint/TEST_MATRIX.md:259-271` @ b4bb0b74 (SMOKE file list)
