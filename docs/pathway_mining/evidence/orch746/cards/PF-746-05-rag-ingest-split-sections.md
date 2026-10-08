# PF-746-05 — rag/ingest._split_sections: first-occurrence headers and the body fallback mislabel sections

Status: REGISTERED, not dispatched. Production is frozen (D-102); dispatch needs a product-owner decision.

## Defect

`src/t2pw/rag/ingest.py::_split_sections` (b4bb0b74:202-246) splits a flattened full text into `(label, text)` spans. Three behaviours give a span the wrong label.

1. **A label binds to the first mention of its keyword, in any case and anywhere in the text.** The patterns are bare words (`_HEADER_KEYWORDS`, :97-106: `r"\bresults\b"`, `r"\bdiscussion\b"`, `r"\bconclusions?\b"`, `r"\bmethods\b"`, …). They are searched in the casefolded text (:213-218):
   ```python
   if label in boundaries:
       continue
   match = re.search(pattern, lowered)
   if match:
       boundaries[label] = match.start()
   ```
   So a prose word ("results from", "the conclusion that", "full discussion)") starts a section. Everything up to the next boundary then carries that label. When a label has several patterns, the first pattern in table order that matches anywhere wins, even if another of its patterns matches earlier. The comment at :94-96 says the first-occurrence rule is there to stop *later* prose mentions from starting a section. Nothing stops the *first* mention from doing so.
2. **`body` fallback.** Text before the first boundary becomes `("body", head)` (:236-240). A document with no boundary becomes `[("body", text)]` (:232-233). So `body` means both "front matter / preamble" and "no header found".
3. **Offset drift.** This is not part of the ORCH-745 wording, but it sits in the same lines. Positions come from `lowered = text.casefold()` (:211) but are used to slice `text` (:238, :243). `casefold()` makes some characters longer ("ß" → "ss"), so every boundary after such a character lands that many characters late.

## Observed impact

`rag/ingest.py` is identical at f9556288 (pilot-003), 82fc69a7, 686ae134 and b4bb0b74 (`git diff --stat` between them is empty). Span evidence comes from `experiment/pathway-mining-pilot-003:docs/pathway_mining/pilot/003/run/evidence_spans.jsonl` (the line numbers below refer to this file). Full texts come from `82fc69a7:tests/fixtures/mining/orch745a_pilot_texts.json`.

**Observed (labels as stored by the run):**
- **PMC8508710 (13-LOX primary).**
  - `conclusion#0` (ev_564d922a7e09ebae, line 7, chars 25776-30369) starts "conclusion that 12-OPDA was either a non-competitive or mixed LOX4 inhibitor". That is Results prose.
  - The paper's printed "5. Conclusions" heading is at char 57628 of the fixture text. It falls inside `methods#4` (ev_4dcc372e77b84976, line 6, chars 55422-58652).
- **PMC13364333 (bryophyte review).**
  - `results#0` (ev_20a7facf5fd73dd9, line 23) starts "results from dn- cis -OPDA/OPDA β-oxidation".
  - `discussion#0` (ev_47b9cde252a0621f, line 30) starts "discussion). The patatin-like candidate".
  - `methods#0` (ev_8fb9b658835414af, line 40) is the 61-character MDPI disclaimer "methods, instructions or products referred to in the content.".
  - Chars 2892-57328 are all labelled `introduction`.
  - 12 of the review's 23 reactions in PILOT-003-AUDIT.md cite ev_20a7facf5fd73dd9.
- **Front matter labelled `body`.** For all four full-text papers, `body#0` is EuropePMC front matter (journal ids, ISSN, authors): lines 2, 26, 48 and 63. Reaction #25 cites ev_06e5eb88be1f278b and reaction #30 cites ev_1f2c624a181ba5ef (PILOT-003-AUDIT.md:375, :435).
- **Offset drift.**
  - The fixture text of PMC12787732 contains two "ß" characters in its front matter (chars 1330, 1459).
    - Its `body#0` (line 63) ends "…Collection date 2026 Jan. 1. In".
    - Its `introduction#0` (ev_f7c56801adf6792d, line 57) starts "troduction Jasmonates".
    - Its results, discussion, conclusion and methods spans start "sults", "scussion", "nclusions" and "terials and Methods" (lines 64, 61, 58, 73).
  - PMC8508710 contains "ß" at chars 45447 and 77724. The Acknowledgments boundary after the first one landed one character late: `methods#4` (line 6) ends "…diseases. A".
- **Pilot-002 A4.** An abstract-only paper was indexed as `abstract#0` and `body#0` with identical text (PILOT-002-AUDIT.md:82, :565). The `body` label comes from the no-header fallback. The duplication itself comes from mining passing the abstract as both `abstract` and `full_text` (the audit cites `mining/retrieval.py:170-183` at that run), which `chunk_paper` then emits twice. That duplication is not part of this card.
- **Same function, different mechanism (out of scope).** In GH3.10 (PMC12096059), the reference list is labelled `results`. ev_ae0d92db16547d2c (line 51) contains "References Acosta I.F. …". Reactions #31 and #32 cite it (AUDIT:446-448, :458). The only "references" mention is at char 24880, before the 55 % tail start (27529), so the back-matter rule (:220-229) did not bind.

**Inferred, not observed:**
- On the unmerged species branch (686ae134), the section label can withhold credit:
  - spans whose label matches method, material, introduction, background, discussion or conclusion earn no positive credit;
  - `results` and `body` are not blocked by this check.

  Pilot-003 ran before that gate existed (f9556288 species.py does not read the section), so no pilot credit decision was made on these labels. Applied to the labels above, the section check would:
  - withhold credit from PMC8508710's Results prose, because it is labelled `conclusion`;
  - not withhold credit from the review's tail passages, because they are labelled `results`;
  - not withhold credit from front matter, because it is labelled `body`.

  Reviewers recorded this exposure as csb_r1 N-2 and BIO-R1-9, CS-B-redteam N3, and csb_r2_final R3-N2.
- **Where else the label shows up.**
  - The mining extraction prompt shows the label to the model (82fc69a7 extraction.py:174-185).
  - Production prints it in evidence blocks (retrieve.py:1155, :1172), in report citations (research_report.py:199-213) and in the reaction-evidence view (app/reaction_evidence.py:162).
  - At b4bb0b74, no production gate grants or denies anything based on these labels. The only label-based filter is `_NON_EXTRACTABLE_SECTIONS` (:128, :359-363).

## Current mitigation in mining

**None for the labels themselves.** Mining (82fc69a7) states that the labels cannot be trusted (retrieval.py:52-57) and removes content instead:
- `text_regions` (:136-172) finds front matter, publisher notes and reference lists.
- `PassageIndex._keep_prose` (:392-428) drops or trims chunks inside those regions, drops citation-dense chunks, and drops duplicate chunks (A4).

Together these remove the four front-matter `body#0` chunks, the disclaimer labelled `methods`, and the GH3.10 bibliography (G9 test `tests/test_mining_evidence_hygiene.py:400-425`). Chunks that survive keep the `_split_sections` label (retrieval.py:365, :517). When `chunk_paper` returns no chunks, `_chunk`'s fallback (:450-466) indexes the whole document as `body`.

On the species branch (686ae134):
- `_NO_CREDIT_SECTION` is a search, not a prefix test (species.py:576-580, :935-937).
- `body` keeps credit by decision (species_policy.md:251-256).
- The exposure from a background sentence labelled `body` is recorded in known_limitations.md:186-190.

## Proposed change

Change `_split_sections` only:
1. **Bind a body header only where it looks like a heading.** That means the keyword printed in Title Case or UPPER CASE (never lower case), at one of these positions:
   - the start of the text;
   - after a line break;
   - after sentence-final punctuation;
   - after a section number ("2.", "3.3.").

   UPPER CASE may stand anywhere. For each label, the earliest heading-form match wins. No body header binds at or after a bound back-matter (`references` / `acknowledgments`) boundary.
2. **Search `text` itself, not `text.casefold()`,** so that positions index the string being sliced.
3. **Give the text before the first bound header its own label.** The product owner picks the name, for example `front`. `body` stays only for a document where no header binds. Do not add the new label to `_NON_EXTRACTABLE_SECTIONS`, so no text is newly withheld from production retrieval.

Not in this card: the back-matter tail threshold, the A4 duplicate, and a per-span confidence signal.

## Ownership

`src/t2pw/rag/ingest.py :: _split_sections` (b4bb0b74:202-246), plus the `_HEADER_KEYWORDS` table and its comment (:94-106).

New tests go in `tests/test_rag_ingest.py`.

Nothing else may change: not `chunk_paper`, not the back-matter constants (:108-128), and no file under `src/t2pw/mining`.

## Acceptance

- G9 tests A-E below pass.
- `_split_sections("Arabidopsis aos lines lacked 12-OPDA.")` still returns `[("body", …)]`. This is a regression guard, and it passes at both SHAs.
- These existing tests pass unchanged:
  - `tests/test_rag_ingest.py`, the whole file (section labels are pinned at :93-115, :121-144 and :304-312);
  - `tests/test_rag_payload_gate_guardrails.py:561-589`;
  - SMOKE (508), which runs `tests/test_rag_admission_production_path.py` through the real `ingest` → `chunk_paper`.
- On the four pilot-003 fixture texts:
  - no span starts with a lower-case keyword;
  - no span starts in the middle of a word;
  - the review has no `results`, `discussion` or `methods` span;
  - each of these printed headings starts its span: PMC8508710 "2. Results" (8910), "3. Discussion" (30537), "4. Materials and Methods" (39648) and "5. Conclusions" (57628); PMC12787732 "1. Introduction", "2. Results", "3. Discussion", "3.3. Conclusions" and "4. Materials and Methods"; PMC12096059 "INTRODUCTION" and "RESULTS"; PMC13364333 "1. Introduction" and "8. Conclusions and Perspectives".
- Run the mining suites at 82fc69a7 and 686ae134 against the changed ingest as an interaction check. Report any failures; do not fix them in this change.

## G9 obligation

This card claims to correct existing behaviour, so it needs behavioural tests that fail at b4bb0b74 and pass after the change. The b4bb0b74 source of `_split_sections` was extracted with `git show` and run on its own on each input below; the base output is quoted for each. Every test checks the labels and text that `_split_sections` / `chunk_paper` return. None of them relies on a symbol being absent.

- **A. Prose "conclusion" (shaped like PMC8508710).**
  - Input: `"1. Introduction Lipoxygenases are dioxygenases. 2. Results 2.1. LOX4 kinetics The Dixon plot supported the conclusion that 12-OPDA was a mixed LOX4 inhibitor. 3. Discussion 3.1. These data extend the route. 4. Materials and Methods 4.1. Data processing was done."`
  - Assert: the span containing "conclusion that 12-OPDA" is labelled `results`, and no span is labelled `conclusion`.
  - Base output: `('conclusion', 'conclusion that 12-OPDA …')`.
- **B. Verbs in a review (shaped like PMC13364333).**
  - Input: `"1. Introduction Jasmonates are ancient. JA in this phylum likely results from beta-oxidation (see Section 2.4 for full discussion). The patatin-like candidate is wound-induced. 8. Conclusions and Perspectives Jasmonate biosynthesis is ancient."`
  - Assert: there is no `results` or `discussion` span, and "results from" lies in the `introduction` span.
  - Base output: a `results` span starting "results from" and a `discussion` span starting "discussion).".
- **C. Offset drift (shaped like PMC12787732).**
  - Input: `"Hans Straße 1, Weg Straße 2. 1. Introduction Jasmonates are signals. 2. Results 2.1. Levels rose."`
  - Assert: the `introduction` span starts "Introduction" and the `results` span starts "Results".
  - Base output: spans starting "troduction …" and "sults …".
- **D. Front matter.**
  - Input: `"pmc Int J Mol Sci 1422-0067 Maynard Daniel. 1. Introduction Lipoxygenases are dioxygenases. 2. Results LOX2 was active."`
  - Assert: the first span is not labelled `body`.
  - Base output: `('body', 'pmc Int J Mol Sci …')`.
- **E. Keyword inside the bibliography.**
  - Input: about 40 sentences under "1. Introduction" and about 40 under "2. Results" (enough that the References heading falls after 55 % of the text), then `"References 1. Smith J. A method. Nat. Methods. 2017;14:587. 2. Doe J. Plant J. 2019."`
  - Assert: there is no `methods` span, and no `chunk_paper` chunk contains "2017;14:587".
  - Base output: `('methods', 'Methods. 2017;14:587. 2. Doe J. Plant J. 2019.')`, which is a bibliography tail that `chunk_paper` emits. The `references` span ends at "Nat.", so at the base no chunk contains the literal "Nat. Methods". That is why the assertion probes "2017;14:587".
  - This failure is shown on synthetic input only; no pilot showed it.

## Risks and interactions

- **Production callers.** `chunk_paper` (:359-365) is called by `ingest` (:598) and by eval-v2 live retrieval (`src/t2pw/bench/eval_v2/rag.py:190-201`). Mining calls it too (82fc69a7 retrieval.py:447). Because chunk boundaries move, eval-v2 retrieval numbers may move too. Labels flow to:
  - retrieve.py:1155 (as the entry's `compartments`) and :1172 (the provenance line);
  - research_report.py:199-213;
  - synthesize.py:416/:429/:466/:1497;
  - admission.py:2090/:3987 (lineage locator fallback);
  - app/reaction_evidence.py:162.
- **Chunk ids change.** `_chunk_id` (:174-177) hashes the section, so a relabelled chunk gets a new id. `ingest` only upserts (:621), and both stores upsert by id without deleting (store.py:186-205, :331-350). A store persisted before the change would therefore keep the old chunks next to the new ones (inferred, not tested).
- **SMOKE.** `test_rag_admission_production_path.py` is in SMOKE. Its fixture (:110-121) uses Title Case headings at line starts, so it is expected to be unaffected; the SMOKE run has to confirm this.
- **Mining species gate.** The change moves credit under the 686ae134 gate:
  - the review's tail passages lose `results` (not blocked) and become `introduction` (no credit);
  - PMC8508710's Results prose stops being `conclusion`;
  - a new pre-header label is not matched by `_NO_CREDIT_SECTION`, so it would not be blocked unless the gate is updated in step with this change.

  `tests/test_mining_species_repair1.py:363-369` pins that a span labelled `body` keeps credit. It builds the span with `section="body"` directly (`probe`, :52-55) and never calls `_split_sections`, so this change does not affect it. The premise it encodes still holds: an abstract-only paper is indexed whole as `body`, because a header-less text stays `body`.
- **Limits of the heading-form rule.**
  - A capitalised word at the start of a prose sentence ("Results from …") still binds.
  - A heading printed in lower case is missed.
  - A structured abstract repeated in the front matter ("Background … Results …") would still bind there, as it does today. This is inferred; it does not occur in the four pilot-003 texts.
  - A combined "Results and Discussion" heading becomes a single label. Today it splits into a two-word `results` span followed by `discussion` (BIO-R1-9(a)). The product owner decides which label the combined heading gets.
- **Draft replay.** The drafter ran a draft of rule 1, read-only, on the fixture texts. It found every heading listed under Acceptance and none of the prose mentions. But it bound `methods` at "Nat. Methods." inside the review's reference list (char 90189), which is why rule 1 includes the "not after back matter" clause.
- **Still open after this card.** The GH3.10 bibliography stays labelled `results` (back-matter tail rule). It is mitigated in mining only.

## References

- b4bb0b74: `src/t2pw/rag/ingest.py:94-128, 174-177, 202-246, 301-382, 598, 621`; `src/t2pw/rag/store.py:186-205, 331-350`; `src/t2pw/rag/retrieve.py:1155, 1172`; `src/t2pw/rag/research_report.py:199-213`; `src/t2pw/app/reaction_evidence.py:162`; `src/t2pw/bench/eval_v2/rag.py:190-201`; `tests/test_rag_ingest.py:55-144, 274-312`; `tests/test_rag_payload_gate_guardrails.py:561-589`; `tests/test_rag_admission_production_path.py:110-121`; `docs/pwml_recovery_sprint/TEST_MATRIX.md:259-271`
- 82fc69a7: `src/t2pw/mining/retrieval.py:48-57, 136-172, 344-428, 430-467, 517`; `src/t2pw/mining/extraction.py:168-185`; `tests/test_mining_evidence_hygiene.py:400-434`; `tests/fixtures/mining/orch745a_pilot_texts.json`; `docs/pathway_mining/evidence/orch745/reviews/csb_r1.txt:37-40, 75-78`; `CS-B-redteam.txt:47-50`; `csb_r2_final.txt:147-150`
- 686ae134: `src/t2pw/mining/species.py:576-580, 935-937`; `docs/pathway_mining/species_policy.md:251-256`; `docs/pathway_mining/known_limitations.md:186-190`; `tests/test_mining_species_repair1.py:52-55, 363-369`
- Pilots: `experiment/pathway-mining-pilot-003:docs/pathway_mining/pilot/003/run/evidence_spans.jsonl` (lines 2, 6, 7, 23, 26, 30, 40, 48, 51, 57, 58, 61, 63, 64, 73); `…/003/audit/PILOT-003-AUDIT.md:3, 375, 435, 446-448, 458`; `experiment/pathway-mining-pilot-002:docs/pathway_mining/pilot/002/audit/PILOT-002-AUDIT.md:82, 565`
- Scratch (not committed): `C:/t/bt/rev745b-r1-bio/runs/r5/out.txt` (per-paper section inventory: the review has 54436 chars of `introduction` and 61 of `methods`; GH3.10 has 29590 chars of `results`)
- `docs/pwml_recovery_sprint/ORCH-745-RESULT.md:177` (the registration line)
