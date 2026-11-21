# WS-FIX-CONFORM review (ws-reviewer, opus), recorded by the orchestrator

```
TASK: fix-stale-conform-test  BRANCH: ws/fix-stale-conform-test  REVIEWED HEAD: 3ad49853 (base main e17b78ba)
VERDICT: APPROVE WITH NOTES
GATES: boundary ok · no-biology ok · behaviour-preserved n.a. (test-only; edit confirmed legitimate) · tests ok · evidence (F4) · secrets ok · result-doc (F4)
CHECKS:
  (1) Same invariant, at least as strong: reads every .py under src/t2pw (also attribute-access calls); exactly 1 call;
      base argument a plain name (or the seed_payload keyword); the merge call taking the envelope uses the same name as
      its base. Every lookup fails the test if nothing is found.
  (2) Seed swap caught: mut_check.py swaps SRC to a copy with (rag_payload, seed_graph); real tree passes, copy fails.
  (3) Call disappearing gives 0 and fails; a second caller gives 2 and fails.
  (4) Diff touches tests/test_rag_seed_entity_reimport.py only; the relevant files are identical between e17b78ba and
      webserver/refactor f89b3f78.
  (5) Production wiring satisfies the invariant: rag_orchestration.py:896 conform(rag_payload, graph); :899
      merge_site(graph, envelope, ...); run_rag_rounds (:1086) passes context.graph into both; merge_site is
      extraction.py:1003 _rag_merge_site(final_payload, rag_envelope, ctx) -> merge_additions(final_payload, ...),
      whose first parameter is base (pipeline.py:1171).
FINDINGS:
  F1 [note] test trusts merge_site's first argument is the merge base; does not check extraction.py:1010-1014.
  F2 [note] docstring says the C-060a lift finds the statement "in streamlit_app.py"; since R3d it is service/extraction.py.
  F3 [note] aliased import or getattr call would escape the count (same gap as the old test).
  F4 [note] RESULT doc and evidence must be committed under docs/webserver/ before merging.
  F5 [note] SMOKE must run on the merge result into webserver/refactor.
```
