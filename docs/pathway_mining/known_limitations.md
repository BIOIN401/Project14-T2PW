# Known limitations

Version 1. Everything here is deliberate or measured, not aspirational.

## Deferred by design

* **No external biochemical database validation.** PathBank, PWC, HMDB, KEGG,
  ChEBI, PubChem and UniProt are not consulted. Every entity and reaction reads
  `{"status": "not_evaluated"}`. A metabolite named two different ways in two
  papers that share no lexical form will be two entities. Attaching a resolver
  is the next obvious piece of work and `t2pw.mining.resolver` is its seam.
* **No PWML.** See [`no_pwml.md`](no_pwml.md).

## Normalization is lexical and within-run

`normalize.comparison_key` folds case, whitespace, dashes, primes, greek letters
and a trailing charge. It will merge `NAD+` with `NAD` and `12-OPDA` with
`12 OPDA`, and it will **not** merge `12-OPDA` with `12-oxophytodienoic acid`
unless a paper states the equivalence and a caller records it through
`link_alias`. Every merge carries its reason, so an auditor can see the
difference between "same normalized key" and "declared alias".

Entity types are inferred from the model's declared type first and an enzyme
suffix heuristic second. A declared type always wins; a disagreement between two
papers about a type is left as information rather than "fixed".

## Species normalization is a short table

`species.py` carries ~30 common names and handles the abbreviated binomial and a
strain tail. An unusual organism normalizes to itself, which is correct but
means `Arabidopsis` and `A. thaliana` fold while some other genus abbreviation
may not. It is not a taxonomy service and never pretends to be.

## Conflict detection is attribute-level and literal

Only `enzyme`, `direction`, `reversible` and `compartment` are compared, and
only by normalized string equality. "Chloroplast" and "plastid" will read as a
conflict; "AOS" and "allene oxide synthase" will too, unless an alias link was
recorded. That errs toward showing a disagreement rather than hiding one, which
is the right direction, but it will over-report.

## Retrieval quality depends on the embeddings service

With no embedding endpoint, `Embedder` falls back to deterministic lexical
vectors and retrieval gets materially worse. That is visible, never silent —
`embedding_mode`, a preflight check, a summary warning and a report line — and
`PATHWAY_MINING_REQUIRE_EMBEDDINGS=true` turns it into a preflight failure. But
an offline run's thin results are a retrieval limitation, not a statement about
the literature.

## Model non-determinism

Identical legs give materially different draws at temperature 0 on this stack.
Two mining runs over the same corpus can produce different graphs. Re-run before
calling a single-run difference a regression. The **policy** is deterministic:
given the same claims, the same decisions and the same placement follow.

## Full-text availability

`rag.acquire.fetch_full_text` resolves a PMCID and fetches EuropePMC full text.
A paper with no open full text contributes its abstract only, which is recorded
(`full_text_chars`) but sharply limits what can be extracted. DOI-only and
URL-only references frequently resolve to an abstract.

## Index hygiene knows the pilots' layouts

What indexing keeps out ([`evidence_policy.md`](evidence_policy.md) § what text can be
evidence) was fit on 35 EuropePMC full texts: the seven pilots 002 and 003 fetched and the 28
committed under `runs/2026-07-27_*` at f9556288. It was then checked, read only, on the 200
unique EuropePMC full texts in the project's acquire cache (`data/rag_index/acquire_cache/
fulltext`), indexing each with no abstract and no passage cap at 6ceef736 and at the repair-2
tip. Text kept at 6ceef736 and not at the tip, on 8 of the 200: reference entries (4 texts),
ACS processing flags (4), a data-availability and acknowledgments paragraph before a
headingless list (PMC10055903, 945 characters) and a Supporting Information file list
(PMC13418202, 428 characters). No body prose. Front matter
is recognised by EuropePMC's processing flags only: a PDF upload's title page, or a source
without those flags, keeps its front matter. Publisher notes are recognised by three endings:
the two MDPI notes in the pilot texts and Frontiers' standard note. A reference list needs a
`References` / `Literature Cited` / `Bibliography` heading or, without one, three authors with
initials in a citation-dense window to be cut as a region; otherwise only a chunk that is
citation-dense as a whole is dropped, and a chunk that is only partly bibliography can keep
that part. The density bar (5 marks per 1,000 characters) separates every reference chunk (13
or more) from every body window (1.6 at most) in the 35 texts. A list ends at the next figure
or table label OR mention after its run of citation marks, as at 6ceef736 (repair 1's
label-only rule was reverted: it dropped body prose). So a mention can end a list early and
leave a fragment outside it (the pepper paper's "(Supplemental Table S3) , and the raw read
data ..." sentence), and a table whose "References" column holds author-year cells still opens
a false list that runs to the table's own mention: PMC8510960 loses that table's rows and the
text up to "(Table 1 )", exactly as at 6ceef736.
The front matter can hold an abstract, a significance statement or highlights; only the
record's abstract survives as `abstract#0` (the metadata lookup fills an empty one), so a
PNAS significance statement is not indexed. A record with no abstract takes the front
matter's "Abstract ..." passage instead, which needs that heading: PMC8508710's front matter
has none. Figure and table legends printed after a reference list that sits in the last 45 %
of the text are not indexed at all: rag's chunker drops everything after that heading before
mining sees the chunks. The headingless-list rule needs three authors with initials in its
first 1,500 characters and backs over at most 300 characters of the first entry's authors, so
a list opening with a corporate author (Cureus: "1. World Health Organization ...") can keep
its first entry. "Survive" in these rules means the text stays retrievable: a trimmed chunk
gets a new evidence id, and a figure snippet contained in a body chunk is deduplicated (A4).

## A review must say it is one

**Product-owner decision (A1-R1-UNKNOWN-WITHOUT-FAILED-LOOKUP, recorded per the ORCH-745 CS-A
repair-2 brief).** A paper graded `unknown` WITHOUT a failed lookup remains eligible for
primary-strength credit: an offline run or an upload (no lookup is made), a paper EuropePMC has
no record of (DOI- or URL-only), a bare PMCID with no title (the subject heading has nothing to
anchor to), a PubMed record declaring only "Journal Article", and subject headings the rule does
not read (e.g. "Systematic Review and Meta-Analysis", "Review Paper"). Only a lookup that failed
twice marks a paper `evidence_type_verified: false`, the case a gate may fail closed on.

Without a declared review type (NCBI, or EuropePMC's metadata lookup, which an offline run
never makes), a review is recognised only by its JATS subject heading (which needs the record's
title to anchor it, so a bare PMCID with no title has none), a title marker or a
self-referential sentence ("this review", "here we review", ...). A review that never describes itself in those words
stays `unknown`, and its reactions are graded `single_paper`. Two reviews, or a review and a
primary paper, supporting one reaction still grade `multiple_independent_papers`:
`admission.evidence_level` counts papers, not their types.

## Uploaded PDFs

Text extraction goes through the existing `t2pw.extraction.pdf_parser`. A
scanned PDF without OCR yields little; the failure is recorded in the paper's
`notes` rather than raised. Page-level locators are not currently propagated
from the PDF parser into evidence spans, so an uploaded PDF's spans carry a
section and paragraph but `page` stays null.

## Sequential by construction

One model call at a time, one paper indexed at a time. A large run is slow. No
parallelism is introduced around the shared caches, the vector store or the
budget ledger because the architecture provides no proven safe mechanism for it.

## Streamlit blocks during a run

The page runs the controller synchronously with a progress bar. There is no
background job runner, so the browser tab is occupied for the duration.
Cancelling mid-run from the UI is therefore limited to interrupting the process;
the CLI's `Ctrl-C` path and `controller.cancel()` are the supported routes, and
both leave a resumable run.

## Cost is an estimate

Priced from configured per-million-token rates, before any bill exists. It does
not reflect provider discounts, cached-token pricing or minimum charges, and it
says so in `costs.json`, in `report.md` and in the CLI status output.

## The provider is unpinned and unreported (production follow-up)

`t2pw.llm.client` is shared with production and frozen for this work, so mining can only
mitigate what it does:

* **Provider routing is unpinned.** `chat_detailed` sends no OpenRouter `provider`
  preference, so identical requests can be served by different providers. Pilot-003 had 7
  thin answers of 16 (no field cited; two with schema placeholders as values). Three took
  150-210 output tokens, four took 5k-24k, and all seven share one alphabetically sorted key
  set: the shape follows the provider route, not the amount of reasoning. Mining therefore
  detects it structurally, re-asks a thin answer once (at most one extra call per thin answer)
  and marks what it kept (`answer_quality`); it cannot choose the route. With no usage
  reported, the token counts are estimates from the visible text, so the structural rule
  decides alone; a future token rule needs an explicit per-attempt `usage_reported` flag.
* **The provider is never reported.** `CompletionDiagnostics` has no provider field, so
  `ModelCallRecord.provider` reads `not_reported_by_llm_client` on every live call.
* **The reasoning setting is never sent.** `MiningModelConfig.reasoning` is recorded but
  `chat_detailed` takes no reasoning parameter, so the provider default applies.
* **`response_hash` is partial.** It is set only for a reply with text, as the hash of the
  last request of a call; an empty or content-filtered reply, and every re-sent request in
  the diagnostics `attempt_log`, carries none.
* **Usage is cumulative.** Token counts come from a process-wide counter, so mining can
  measure each attempt but not each provider request inside it.

The fix belongs in `llm/client.py`: pin (or at least report) the provider, pass the
reasoning setting, and hash every response.

## Frontier heuristics are untuned

Twelve frontier kinds with hand-set priorities. They have been exercised against
a synthetic fixture, not tuned against real corpora. Expect the bridge and
conflict kinds to be the most valuable and the cofactor kind to be mostly noise.
The species reserve (a quarter of the slots, each at the earlier of its priority
position or the next fourth place) and the 0-3
on-route relevance score are equally hand-set: lexical name matching, no taxonomy,
and a carried item keeps the score it was generated with.

## No live evaluation

No live mining run has been executed against real literature as part of this
work — the whole test suite is fixture-driven and spends no API credit. The
system is *verified* (226 focused tests plus an eight-point integration test) but
not *evaluated*: nobody has yet judged whether the reactions it mines from real
papers are good ones. That is the obvious next step and it needs explicit cost
authorization.

## Not merged

`experiment/pathway-mining` is not merged anywhere and is not proposed for
merge. If it ever is, see the checklist at the end of
[`feature_flag.md`](feature_flag.md).
