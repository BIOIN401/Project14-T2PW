# ORCH-745 CS-A — repair round 1 (of at most 2)

Reviews of your tip `6ceef736`: code review **APPROVE_WITH_FINDINGS** (no blocking); bio-audit
**SOUND_WITH_FINDINGS with 2 blocking findings in your boundary** (a third, A3 evidence_level for
review-only multi-paper support, is in `admission.py` and goes to change set B, not you).
Same rules as before (`C:/t/bt/orch745/COMMON_RULES.md`); same worktree `C:/t/m745-a`, same branch.
Add new commits on top of `6ceef736` (never rewrite history). New tests for corrections of shipped
behaviour must fail at `6ceef736` or at `f9556288` as appropriate — say which in the manifest
(append to `C:/t/bt/m745-a/g9_manifest.json` under a new key `repair1_nodes`).

## Blocking

**A1 — review recall without a declared type (bio-audit, product_contract_violation).** When the
declared type is missing (EuropePMC lookup failed — only a note — or the NCBI record lists only
"Journal Article"), the grade rests on title + self-reference rules. Some reviews say they are
reviews only in their JATS subject heading, which sits in the front-matter region that `body_text`
strips before grading. Also `_PRIMARY_TYPES` contains "journal article", so a review whose PubMed
record lists only "Journal Article" and whose lookup failed is graded `primary` with an
authoritative-looking basis. Required:
1. Read the JATS subject heading from the front-matter region BEFORE stripping and treat it as a
   declared type. Anchor it to the words just before the record title ("Review", "Reviews",
   "Review Article", "Mini-Review", "Minireview", "Systematic Review", "JBC Reviews") so CRediT
   "Writing – review & editing" and "Reviewed by:" never match.
2. Remove bare "journal article" from `_PRIMARY_TYPES` ("research-article"/"Research Article" still
   mark a primary). Live EuropePMC core returns e.g. `["research-article", "Journal Article"]` for
   GH3.10 and 13-LOX and `["review-article", "Review", "Journal Article"]` for the bryophyte review
   (orchestrator-verified live, see C:/t/bt/orch745/epmc/out.json) — make sure that joined string
   still grades primary / review correctly.
3. After a failed metadata lookup, retry it once before extraction; if it still fails, set a
   machine-readable marker on the record (e.g. `evidence_type_verified: false` /
   basis "type unverified: lookup failed") so the species gate can fail closed later. Keep it a note,
   never an exception.
Regression fixture (offline): the bio-auditor used committed EuropePMC texts from
`runs/2026-07-27_*/papers/*/01_source_text.txt` at f9556288 (read via `git show f9556288:<path>`);
pick the reviews that declare themselves only in the subject heading, copy the needed text into a
fixture, and test them as `OfflinePaperSource` europepmc candidates with no abstract and no declared
type -> `review`.

**A2 — a reference list with no heading leaks into the boundary chunk (bio-audit).** With no
References/Literature Cited/Bibliography heading (EuropePMC's flattening of ACS articles), only chunks
that are citation-dense as a whole are dropped; the chunk straddling prose and bibliography is kept
untrimmed, so bibliography entries become an evidence span. Required: when no headed region is
found, detect the list from citation density — e.g. the first ~1,500-character window in the back
part of the text whose density is at or above `_CITATION_DENSITY`; follow its run of citation marks
with the existing `_CITATION_GAP` logic, then to the next caption or the end, recorded as a
`reference_list` region so `_outside` trims the boundary chunk. Also add the ACS processing flags
("document-id-old-9", "production-flag-...", "document-id-alt-64") to `_FRONT_FLAG`. Regression
fixture: the committed PMC12624714 text (find it under `runs/2026-07-27_*/papers/` at f9556288),
indexed with `max_passages_per_paper` 500: no kept span may contain "10.3390/microorganisms8111716",
"Tambadou" or "PMC7692639", while its real prose stays retrievable.

## Non-blocking (fix them in this round; they are small)

- **A4 / R745A-2 — IT2-N3 duplicate when the PMC copy is included first.** After `_settle_included`
  in `acquire()` and `load_explicit()`, if `record.pmcid` (or the DOI) maps to a paper_id already
  included (in `seen_paper_ids`) that differs from `record.paper_id`, exclude the record with a reason
  like "same article as PMC:...". On resume, seed the aliases from each restored record's pmcid. Test
  the EuropePMC-first order (expect exactly 1 included paper).
- **A5 — the abstract is lost when the record has none.** In EuropePMC's layout the abstract lies inside
  the front-matter region; with an empty `record.abstract` and a failed/absent lookup nothing indexes
  it (for GH3.10 that loses the decisive F-1 sentence "In flowers and wounded leaves of gh3.10 jar1
  knockout lines JA-Ile was below the detection limit."). When `record.abstract` is empty, take the
  front matter's "Abstract ..." passage, up to the keyword/funding/flag block, as the abstract chunk.
  Test: an explicit PMC12096059 with no abstract and no lookup -> a span containing "wounded leaves of
  gh3.10 jar1 knockout lines" is retrievable, and none contains "pmc-prop-" or "ifeussn@".
- **R745A-4** — store the EuropePMC `abstractText` as plain text (strip HTML `<h4>`, `<i>`, `<sup>`; reuse
  your `_plain`), never raw.
- **A6** — `report.md` "Admitted reactions" table: add an evidence-level column
  (`reaction.evidence_level`). (`report.py` is yours for this.)
- **R745A-1** — widen `_SELF_REVIEW`: "minireview"; "(this|the present|the current)
  (article|paper|work|chapter|manuscript) (\w+ ){0,3}(reviews|review|overview)"; "this
  (article|paper|work) (provides|presents|offers|gives) an? (\w+ ){0,2}(review|overview)"; "aim of this
  (paper|article|work) is to review"; plus the reviewer's failing phrases ("This article provides a
  comprehensive review of", "In this minireview we discuss", "The aim of this paper is to review",
  "This work reviews", "We summarize and review", "A review of the literature ... is presented") as
  parametrized cases. Keep the "reviewed in / for a review, see" negatives.
- **R745A-5** — the declared-type check `'review' in t.lower()` also catches "reviewed-preprint" and
  "Peer Review": use a word match excluding "reviewed" and "peer review".
- **R745A-6** — anchor the reference-list end to a legend form (token followed by a title-case word or a
  period, not preceded by "Supplemental"/"("), or reword the doc to "the next figure or table label".
- **A7** — add `FIG\.?` and `Fig` (no period) to `_CAPTION`.
- **R745A-3** — docs: `known_limitations.md` one line: legends printed after a tail reference list are
  not indexed (rag drops the tail section); and say that "survive" means the text is retrievable
  (figure snippets contained in a body chunk are deduplicated under A4).

Run the full mining suite at your new tip and report the count. Final report: same format as before,
covering only this round (new commits, per-finding change, tests, manifest additions).
