# ORCH-745 change set A — evidence hygiene (F-2, F-4, A4, A7, IT2-N3)

Worktree `C:/t/m745-a`, branch `orch745/a-evidence-hygiene` (from `f9556288`), scratch
`C:/t/bt/m745-a/`, your name `m745-a`. Follow `C:/t/bt/orch745/COMMON_RULES.md` exactly.

Why this set matters: a later change set (F-1) will make the species gate read each claim's own
supporting sentences instead of whole spans. That is only safe once (a) reviews are graded as
reviews, so a review's statement is admitted as `review_article_only` and never as primary, and
(b) reference-list entries and journal front matter can never be cited as evidence. Otherwise
review legends and bibliography lines become admitted "primary" evidence.

Read first: `docs/pathway_mining/evidence_policy.md`, `src/t2pw/mining/acquisition.py`,
`src/t2pw/mining/retrieval.py`, `src/t2pw/mining/controller.py` (`_register_paper`, the
retrieve/extract loop in `_run_round`), `src/t2pw/mining/models.py` (`PaperRecord`, `EvidenceSpan`),
and `t2pw.rag.ingest.chunk_paper` (read only).

Live data you must use for fixtures (read with `git show`, write fixtures under
`tests/fixtures/mining/`): pilot-003's run on branch `experiment/pathway-mining-pilot-003`:
`docs/pathway_mining/pilot/003/run/evidence_spans.jsonl` (73 spans, 4 papers),
`.../run/papers.jsonl`, `.../audit/PILOT-003-AUDIT.md`. Pilot-002's on
`experiment/pathway-mining-pilot-002` (`docs/pathway_mining/pilot/002/...`, incl.
`bio-audit/lox13_fulltext.txt`).

## F-2 — a review is never graded `review_article_only`

`acquisition.classify_evidence_type` only looks at a declared article type (which the rag
`CandidatePaper` never carries), source/URL, and `title + abstract[:400]`. The pilot-003
bryophyte review PMC13364333 ("Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from
Marchantia polymorpha and Beyond", MDPI Plants) was typed `unknown`; so was every paper.
Its abstract has no review marker; its introduction says "this review makes four specific
contributions". `admission.evidence_level` already maps `evidence_type == "review"` to
`review_article_only` — the defect is the classification.

Required:
1. **Declared type.** NCBI: the efetch XML that `PaperSource._ncbi` already parses carries
   `PublicationTypeList/PublicationType` ("Review", "Systematic Review", "Journal Article", ...).
   Put them on the candidate (e.g. `article_type`) so `record_from_candidate` uses them.
   EuropePMC: rag's search drops `pubTypeList`. Add a mining-side metadata lookup for each paper
   that is actually INCLUDED (not every discovered one): one EuropePMC REST `search` call with
   `resultType=core` by PMCID or PMID (reuse rag's `HttpClient` and `_EUROPEPMC_SEARCH` read-only,
   as `_pmcid_for` already does), reading `pubTypeList.pubType` (e.g. "review-article", "Review"),
   and also `doi` and `journalTitle` to fill empty `PaperRecord.doi`/`journal` (registered
   defect A14: DOI empty on every paper). Cache per run. It must go through the `PaperSource`
   seam so `OfflinePaperSource` (tests) makes no network call; a failed lookup is a recorded note,
   not an exception, and falls back to the text rules below.
2. **Text rules.** Recognise a review from self-referential statements in the title, the abstract
   and the paper's own full text (after F-4 removes the reference list): e.g. "this review",
   "in this review", "the present review", "this mini-review", "here we review", "we review",
   "this article reviews", "review article". NOT a reference to someone else's review:
   "for a review, see", "reviewed in [..]", "(reviewed by ...)", "recently reviewed". A review
   mislabelled primary is the policy violation; a primary mislabelled review only downgrades its
   grade, so prefer recall, but do not label a primary paper a review on a "reviewed in" citation.
3. Re-classify once the full text is known (classification currently happens at discovery time,
   before the text is fetched). Record which rule decided (`notes` or a field) so an auditor can
   see why a paper is a review.
4. `evidence_level(...)` must then grade a reaction supported only by that paper as
   `review_article_only`. Make sure that grade is visible on the decision and the reaction node
   (it already exists on `ClaimDecision.evidence_level`; check the reaction/graph/summary path
   and fix it if the grade is lost).

G9 obligations (fail at `f9556288`): the bryophyte review (title + abstract + a full-text excerpt
containing "this review makes four specific contributions", from the pilot-003 spans) classifies
as `review`; an NCBI record whose XML says `<PublicationType>Review</PublicationType>` gives a
`review` paper record; a single-paper reaction from that review gets `review_article_only`.
Also keep primaries primary: the GH3.10 paper (PMC12096059, "Research Article") and the 13-LOX
paper (PMC8508710) must NOT become reviews (they cite "reviewed in"-style references).

## F-4 — bibliography entries and front matter are cited as evidence

The chunk `section` labels are unreliable, so this must be content-based:
- Reference-list text was chunked under `results`: `ev_3d83d105fb156829`, `ev_8fe6dbfd0a0a1c2a`,
  `ev_af12d6f302acb59e`, `ev_43efbc138027f970`, `ev_63ff7e40627bb63e`, `ev_50fe83be4e4544a8`
  (GH3.10 paper), and `ev_ae0d92db16547d2c` is real text up to "...ID: MTBLS9749." followed by
  "References Acosta I.F. , Przybyl M. ( 2019 ) ..." (pilot-003 claims #31/#32 cite it).
- JATS front matter became `body#0`: `ev_06e5eb88be1f278b`, `ev_1507d8084844aa5c`,
  `ev_1f2c624a181ba5ef`, `ev_b91fc021966a2713` — journal ids, ISSNs, PMCIDs, DOI, authors,
  affiliations, emails, dates, licence, then sometimes "Abstract ..." text.
- Publisher boilerplate: `ev_8fb9b658835414af` ("methods, instructions or products referred to in
  the content.", 61 chars, labelled `methods`).

Required: none of that text can ever become an `EvidenceSpan` (so it can never be cited, never
count as species evidence, never consume the per-paper passage cap). Do it in `retrieval.py`
before `max_passages_per_paper` truncates the chunk list (today `chunks[:cap]` lets reference
chunks displace body chunks). Suggested shape (your call): strip the front-matter prefix up to
the abstract/introduction heading when it is metadata; drop a reference-list region (heading
"References" / "REFERENCES" / "Literature Cited" / "Bibliography" followed by citation-dense text)
and any residual chunk that is citation-dense (author-initial lists, "( 2019 )" years, DOIs,
8-digit PMIDs, "14 , 171 – 178" page ranges); split a chunk at the heading and keep the real text
before it. **Do not over-filter**: real passages must survive — e.g. `ev_8d7685e98100893f`
(results), `ev_e0a035eec4eb94c6` (methods: plant material), `ev_d56713597b017d75` (discussion),
`ev_c98d84c7d84855c1` and `ev_20a7facf5fd73dd9` (review introduction/table text),
`ev_00885b6d262b0c3c` (figure legend), every `abstract#0`. Check where figure/table text sits in
the flattened full text before cutting "everything after References". Record, per paper, how many
chunks/characters were excluded and why (`IndexedPaper` stats, surfaced in the index stats / round
artifacts) so an auditor can see what was removed.

G9: feeding pilot-003's reference and front-matter texts through the indexing path at base
yields evidence spans containing them; at your tip it yields none, while the listed real passages
are all still retrievable.

## A4 — one passage counted twice

An abstract-only paper is indexed as `abstract#0` and `body#0` with the same text, so one passage
counts as two spans. Drop a chunk whose whitespace-normalised text equals, or is contained in, a
chunk already kept for the same paper. G9: an abstract-only paper yields one span, not two.

## A7 — a paper can get 0 passages

`PassageIndex.retrieve(paper_ids=[...])` takes a store-wide top `k*3` and then filters by paper,
so a paper whose chunks rank below other papers' chunks gets nothing (pilot-002 PMC5104190).
Per-paper retrieval must rank that paper's own chunks. G9: with other papers' chunks ranking
higher, the requested paper still receives `min(k, its chunk count)` passages.

## IT2-N3 — one article counted as two papers

`PaperSource._pmcid_for` caches a FAILED EuropePMC lookup as "no PMCID" for the whole run, so the
same article can appear as `PMID:` (NCBI) and `PMC:` (EuropePMC) and count as two independent
papers (which would inflate `multiple_independent_papers`). Distinguish "EuropePMC answered, no
PMCID" (cache it) from "lookup failed" (do not cache; retry next time). If your F-2 metadata lookup
gives you a better primitive, use it for both. G9: a lookup that fails once and then succeeds
resolves the PMCID on the second call.

## Out of scope for you
Do not touch `species.py`, `extraction.py`, `claims.py`, `admission.py` gates, `budget.py`,
`frontier.py`, `cli.py`, `modelclient.py` (another change set owns them, in parallel). The one
exception: if the review grade is lost between `ClaimDecision.evidence_level` and the
reaction/graph/summary, you may make the minimal change in `admission.py` `build_reaction` /
`graph.py` to carry it — say so explicitly in your report.
