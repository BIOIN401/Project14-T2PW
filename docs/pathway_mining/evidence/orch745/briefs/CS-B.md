# ORCH-745 change set B — reading the claim's own evidence (F-5, F-1, F-6, B3, N-5)

Worktree `C:/t/m745-b`, branch `orch745/b-species-reading`, scratch `C:/t/bt/m745-b/`, your name
`m745-b`. Follow `C:/t/bt/orch745/COMMON_RULES.md` exactly, with ONE change: your base is NOT
f9556288. Your branch starts at `11032e8d`, a provisional merge of change sets A (evidence hygiene:
reviews graded `review`, reference lists / front matter / publisher notes never become spans, one
passage one span, per-paper retrieval) and C (run control: per-round cap, frontier species reserve,
thin-answer re-ask, CLI .env). Mining suite at `11032e8d`: **334 passed**. G9 tests must still fail
behaviourally at `f9556288` (the pre-ORCH-745 code) and pass at your tip; in your manifest, also say
whether each node passes or fails at `11032e8d` (it should fail there too for F-1/F-5/F-6, since A and
C do not touch the species gate).

**The binding acceptance spec is `C:/t/bt/orch745/F1_BIO_ADJUDICATION.md`** (the pwml-bio-auditor's
adjudication of the orchestrator's first proposal `C:/t/bt/orch745/F1_PROPOSAL.md`). Read both, the
adjudication carefully: its section 1c is the corrected rule and its section 3 T-rows (T1, T1b, T2,
T3, T4–T11) are mandatory tests. Your implementation must produce exactly those states. The regexes
there are guidance; an equivalent mechanism is fine, but only the T-rows decide. Also read
`docs/pathway_mining/species_policy.md`, `evidence_policy.md`, and `src/t2pw/mining/{species,
extraction,claims,admission,models,controller,graph,frontier}.py` (controller's extraction loop and
`_admit`; frontier's section 7 consumes candidate/incomplete claims).

Live data: pilot-003 on `experiment/pathway-mining-pilot-003` (`docs/pathway_mining/pilot/003/run/`:
`claims.jsonl`, `evidence_spans.jsonl`, `papers.jsonl`, `rounds/round_001/model_raw.jsonl`) and
pilot-002 on `experiment/pathway-mining-pilot-002`; scratch copies of pilot-003 in
`C:/t/bt/orch745/p003/`. Change set A's fixture `tests/fixtures/mining/orch745a_pilot_texts.json`
already holds pilot-003's four included papers' full texts and all 73 spans with role labels — reuse
it. Build your F-1 acceptance tests by REPLAYING the recorded model payloads (`raw_text` in
`model_raw.jsonl`) through `extraction.build_claims` with the recorded spans, plus the counterfactual
claim edits the T-rows describe (e.g. T5: species filled and cited).

## Order of work (one commit per item where practical)

### 1. F-5 (IT2-N1) — participants are not citation-gated; tissue is not gated; placeholders stored

`extraction._participants` keeps any participant with any name; uncited participants are stored and a
reaction can be admitted with no evidence for its substrates or products. Tissue is never gated. The
thin pilot-003 answers stored `"stoichiometry": "has_stoichiometry"` / `"products_evidence_ids_placeholder"`.
Decision already made (implement it):
- A participant is kept only if it has a valid citation (an id the model was shown), OR its name occurs
  verbatim (case/whitespace-insensitive, never inside a longer token; hyphens/Greek normalised as in
  F-1) in a span the claim cites (any of its field or claim-level `evidence_ids`). Record how it was
  kept (`model_citation` / `verbatim_cited_span`), as `field_attribution` does for fields — e.g. a
  per-participant attribution field on `ReactionParticipant`.
- A dropped participant is recorded in the `dropped` list (it reaches the round's `errors`).
- A claim whose substrates AND products are all unsupported becomes `incomplete` (kept as a lead, never
  admitted, never merged into a supported identity — give it its own identity namespace; keep the
  asserted names visible for audit and for the frontier's "retrieve the missing participants" item).
  One side unsupported is already `incomplete` via admission gate 4.
- Reject placeholder echoes in any value: `has_*`, `*_placeholder`, `...`, an `ev_...` literal, empty
  JSON-ish strings. Stoichiometry is kept only if it is a plausible coefficient (number / fraction /
  `n`), otherwise blanked.
- Tissue: kept only if it occurs verbatim in a cited span (attribution `verbatim_cited_span`), else
  dropped like any uncited field; `admission._strip_unsupported` must cover tissue too.
G9 (fails at f9556288): a payload whose participants are uncited and absent from the cited spans
yields a claim with those participants (base) vs none / `incomplete` (tip); `has_stoichiometry` stored
(base) vs rejected (tip); an uncited tissue kept (base) vs dropped (tip).

### 2. F-1 — species markers on the claim's own supporting sentences (the main blocker)

Implement section 1c of the adjudication. In outline:
- Normalise text for matching (U+2010/2011/2012/2013 hyphens, whitespace, Greek letters ↔ names),
  split cited spans into sentences (reuse `acquisition._SENTENCE` or an equivalent that does not break
  after "A." / "et al." / "Fig."), and split sentences into clauses for positive credit.
- Supporting sentences: name a participant, the enzyme or the gene. NEGATIVE markers (INFERENCE, HOST,
  IN_VITRO — lists extended per W7) use loose matching over supporting sentences; POSITIVE credit
  (IN_PLANTA) uses the strict rules 1–6 of 1c (not a review; no hedge, citation, procedural verb,
  purpose clause; not a methods/introduction span; no non-target organism unless the target is named in
  the sentence; clause-level co-occurrence of an in-planta context with (a) substrate+product+conversion
  predicate and no application marker, or (b) loss-of-function + enzyme/gene + substrate/product and no
  no-effect phrase; strict compound-name matching, cofactors and free amino acids never count).
- Review route (rule 7): only for papers graded `review`; result `target_species_supported` with grade
  `review` and evidence level `review_article_only` — never in_planta for a review.
- Aggregation (rule 8): inference (incl. the model's `species_inferred=true`) > in vitro (no positive
  credit and an IN_VITRO sentence) > not stated > multiple > non-target > target; target requires a
  model-cited species span that names the target in a sentence free of INFERENCE/HOST/contrast markers
  (W13), plus positive credit or the review route; otherwise `species_ambiguous`. HOST-only with the
  target -> `species_ambiguous` with a host reason (W12).
- **B3:** when the model leaves species empty, the claim is `species_not_stated`; paper-level screen
  organisms are never used as the claim's species (today they produce `multiple_species` from
  "human LOX homologs").
- `build_claims` needs the paper's evidence type and each span's section: pass the paper's
  `evidence_type` from `controller` -> `extract_claims` -> `build_claims` (new keyword arguments with
  safe defaults). Spans already carry `section`. A paper's grade is settled at inclusion (change set A)
  before extraction; if your design can see a grade change after extraction, recompute.
- Record the assessment on the claim for audit (e.g. `species_support`: the grade — `in_planta` /
  `review` / `in_vitro` / `inference` / `host` / `none` — and the graded sentences with evidence id,
  sentence text, grade and the reason/marker), carry the grade onto the reaction node, and make sure it
  reaches `claims.jsonl`, `decisions.jsonl` and the reaction node in `primary_graph.json` /
  `candidate_subgraphs.json` (bump `ARTIFACT_SCHEMA_VERSION` per `artifacts.md` policy if you add
  persisted fields; it is `t2pw.mining/1.1.0` at your base).
- `species.py` keeps `classify_species` working for its existing callers/tests (extend it, or add the
  sentence-level classifier beside it and route `build_claims` through the new one). Existing species
  tests that encode the old whole-span behaviour may need deliberate updates: list every changed
  assertion and why (merge rule 4: a pinned expectation moved deliberately, with the exact delta).
- Tests: every T-row of adjudication section 3 as a test (T1–T3 are G9: they fail at f9556288;
  T1b, T4–T11 are `test_new_acceptance_745b_*` and must FAIL against a literal implementation of
  F1_PROPOSAL — you need not prove that, but do not weaken them), plus the recommended #33 row.
  Also a test that the full replay of pilot-003's 16 recorded payloads admits exactly #3, #14, #34 and
  nothing else (state per claim per the adjudication table, section 2).

### 3. F-6 (IT2-N2) — same-paper merge is last-writer-wins

`ClaimStore.register` copies the later batch's `species_state` over the earlier one when the same paper
yields the same identity, so the result depends on batch order. Decision: recompute on the combined
evidence. Union the two claims' graded supporting sentences / species evidence (same paper only) and
re-run the F-1 aggregation, so register(A then B) == register(B then A). Also fix `field_attribution`
merging (N3/N4: it keeps the first label while evidence ids are unioned — keep `model_citation` if
either side has it). G9 (fails at f9556288): batch A (in-planta sentence) and batch B (in-vitro only)
of the same paper, registered in both orders, give the same state (at base they differ).

### 4. N-5 — the prompt template pre-fills `"species_inferred": false` and `"hypothesis": false`

The JSON example in `extraction.SYSTEM_PROMPT` shows literal `false` values a model may copy.
Replace them with an explicit choice (e.g. `true|false`, or omit them from the example and describe
them in the rules) and bump `PROMPT_VERSION` to `mining-extract/1.2.0`; update `artifacts.md`.
New-acceptance test only (prompt text).

## Not yours
Do not touch `acquisition.py`, `retrieval.py` (change set A), `budget.py`, `frontier.py`,
`modelclient.py`, `cli.py` (change set C), except: `controller.py` may change only to pass the paper's
evidence type into extraction and to carry the species grade; `graph.py`/`models.py` only to carry the
new fields. Both A and C are under review in parallel and may receive small repair commits in their
own files; keep your diff out of those files so the merge stays clean.
