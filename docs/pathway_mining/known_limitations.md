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
  preference, so identical requests can be served by different providers. Pilot-003's thin
  answers (150-210 output tokens in 4-8 s, schema placeholders as values) look like a route
  that did no reasoning. Mining re-asks a thin answer once and marks what it kept
  (`answer_quality`); it cannot choose the route.
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
