BLOCKED: the species gate (F-1 and its follow-ups, change sets B + D) ended its last allowed repair round with two blocking review findings, so no approved SHA exists to pilot; no pilot was run and the definition of done (items 2–4) is not met.

# ORCH-745 — make pathway mining work: result

2026-10-07 · Lead Orchestrator. **Nothing merged into `sprint/pwml-recovery` or `main`. Nothing pushed. No live model
call was made** (the only network use was one free EuropePMC REST probe, 3 requests, to confirm a response shape).
Production is untouched: every change is inside `src/t2pw/mining/`, its tests and `docs/pathway_mining/`.

**Timeline.** The session was suspended for about 12 hours overnight: no work happened between roughly 00:30 and
13:30 on 2026-10-07 (one review workflow shows 12.9 h of wall time), so the morning deadline was missed. Work
resumed at 13:30 and ended at about 22:30.

## Bottom line

- **Six of the nine blocking problems are fixed, reviewed, approved and merged** into `experiment/pathway-mining`:
  F-2 review grading (#2), F-4 bibliography and front matter (#3), B1/F-7 per-round cap (#4), F-3 frontier species
  reserve (#5), B2 thin answers (#8) and F9 `.env` (#9). A4, A7, IT2-N3, A5, A14 and the `cli.md` / `artifacts.md` errors are fixed too.
  Mining suite **377 passed**, SMOKE **508** with mining off.
- **The species gate (F-1, F-5, F-6, B3, N-5) is built and tested but NOT approved.** It is on
  `orch745/b-species-reading` @ `686ae134`: mining suite 856, SMOKE 508.
  - Replaying pilot-003's 16 recorded model answers offline, it admits exactly the three reactions the
    bio-auditor said it should: GH3.10 JA → JA-Ile on the knockout sentence, in planta; and both OPR3 steps from the
    bryophyte review, graded `review_article_only`.
  - It keeps every in-vitro, ortholog and off-species row out, including pilot-002's recombinant 13-LOX step.
  - It went through change set B (2 repair rounds) and change set D (2 repair rounds), each with a code reviewer,
    a bio-auditor and an adversarial red team. The final review split:
    - code: APPROVE_WITH_FINDINGS, "fit for pilot: YES";
    - red team: UNSOUND, but "fit for pilot: YES" as a gate run under audit;
    - bio-auditor: UNSOUND, "fit for pilot: NO".
  - Two blocking findings remain. Both have reviewer-prototyped one-line fixes (suite stays 856), committed as
    records.
- **Why BLOCKED rather than one more round.** The brief allows at most 2 repair rounds per change set.
  - I already opened change set D once, for new defects the final change-set-B review found in the loss-of-function
    rule.
  - The two remaining findings are defects in change set D's own repairs. One of them sits in a rule my own brief
    prescribed from a reviewer prototype.
  - Opening a third change set for them would be evading the cap, so I stopped here. A weakened gate was never on
    the table.

## Final state

| item | value |
|---|---|
| `experiment/pathway-mining` | **`82fc69a7`** (code at `3ce64f28` = `f9556288` + change set A `7f5b94be` + change set C `beb6d70e`; then evidence and records commits) |
| tests at `3ce64f28` | mining suite **377 passed** (evidence `048`), SMOKE **508** with `T2PW_PATHWAY_MINING_ENABLED` unset (`049`), 0 survivors |
| species gate, unmerged | `orch745/b-species-reading` @ **`686ae134`** = the approved A + C plus change sets B and D. Mining suite **856** (`050`), SMOKE **508** (`051`) |
| pilot branches | none new: no pilot-004 was run, because no approved SHA contains the species gate |
| live spend | **$0** (no model call) |
| G9 base arms | `036`–`047`, all run by me in `C:/t/mining-rep-base`, pinned. Every node failed with an `AssertionError` at its base; 0 survivors |

## Pilots (extending 001–003)

| | pilot-001 | pilot-002 | pilot-003 | pilot-004 |
|---|---|---|---|---|
| code | `23e1ae83` | `ecf6b9a0` | `f9556288` | **not run** |
| papers found / read | 21 / 1 | 150 / 10 | 64 / 4 | — |
| target (A. thaliana) reactions | 0 | 1 (in vitro: should not have been admitted) | 0 | — |
| bio verdict | NOT_PLAUSIBLE | PLAUSIBLE_BUT_THIN | NOT_PLAUSIBLE | — |
| why | | | | no approved species gate (see above) |

**What the unmerged gate does on pilot-003's recorded answers** (offline replay; evidence by test
`tests/test_mining_species_reading.py::test_new_acceptance_745b_f1_the_full_pilot003_replay_admits_exactly_3_14_and_34`
at `686ae134`). This is not a pilot and does not satisfy
the definition of done.

| claim | reaction | state, grade | evidence level | credited sentence |
|---|---|---|---|---|
| #14 `cl_b89be46f92d4c4e1` | jasmonic acid → JA-Ile (GH3.10), PMC12096059 | `target_species_supported`, in planta | single_paper | "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit." |
| #34 `cl_0bc93d248b7f31db` | OPDA → OPC-8:0 (OPR3), PMC13364333 (review) | `target_species_supported`, review | review_article_only | "In Arabidopsis, OPDA and dn-cis-OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0, respectively." |
| #3 `cl_5338b4fc07a47cea` | dn-cis-OPDA → OPC-6:0 (OPR3), PMC13364333 (review) | `target_species_supported`, review | review_article_only | same sentence |

The other 34 claims stay out:
- the recombinant 13-LOX steps, pilot-002's included, are `in_vitro_or_cell_free`;
- the GH3.10 purified-enzyme rows are in vitro;
- the Marchantia and Physcomitrium rows are non-target, in vitro or inference;
- the JA-conjugate rows are `species_ambiguous` (occurrence is not conversion; that is a product-owner hinge).

The pwml-bio-auditor adjudicated these expectations before any code was written
(`evidence/orch745/briefs/F1_BIO_ADJUDICATION.md`).

## What changed, by problem number

| # | problem | change | where | status |
|---|---|---|---|---|
| 1 | F-1 species gate reads whole passages | The gate grades the claim's own supporting sentences. Positive in-planta credit now comes **only** from loss-of-function genetics: a plant mutant/knockout of the claim's own gene, plus a strictly named participant, plus a measured effect with the right polarity (product absent or down, substrate up). Reviews earn credit only through a strict review route graded `review_article_only`. In-vitro, ortholog/homology, transgenic-host, fed-precursor, hedged, negated, cited-background, methods/introduction/discussion and other-organism sentences earn nothing. | B, D | **not approved** (2 blockers) |
| 2 | F-2 reviews not graded | A paper's type is read from NCBI PublicationType, EuropePMC `pubTypeList` (one lookup per included paper), a JATS subject heading, the title and the paper's self-references. A review is graded `review` and its reactions `review_article_only` on the decision, node, views, summary and report. A failed lookup is retried once and then marked `evidence_type_verified: false`. | A | merged |
| 3 | F-4 bibliography / front matter cited | Reference lists (with or without a heading), JATS front matter, publisher notes and duplicate abstract spans never become evidence spans. Exclusions are recorded per paper. | A | merged |
| 4 | B1/F-7 per-round cap stops on junk | `max_new_reactions_per_round` counts only newly admitted reactions and is still a stop condition. Also A5: no phantom round after a recorded stop or a spent discovery budget. | C | merged |
| 5 | F-3 frontier cap starves species search | At least ⌈25 %⌉ of slots are reserved for `missing_species_evidence` items, ordered by on-route relevance. Each reserved item takes the earlier of its priority place or the next fourth place. Every cut item is recorded in `frontier_cut.jsonl` and `metrics.frontier_cut`. | C | merged |
| 6 | F-5 participants not citation-gated | A participant is kept only if it is cited or appears verbatim in a cited span; otherwise the claim is an `incomplete` lead in its own identity namespace. Tissue is gated and placeholder echoes are rejected. | B | not approved (merged together with #1) |
| 7 | F-6 last-writer-wins merge | A paper's batches are all registered before judging. The same-paper union is re-classified on the combined graded sentences, so the result is order-independent; legacy floors persist. | B | not approved |
| 8 | B2 thin answers | A structurally thin answer (no field cited, or a schema placeholder echoed) is re-asked once, as a counted call, and marked `answer_quality`. The provider is recorded, or `not_reported_by_llm_client`. | C | merged |
| 9 | F9 CLI never loads `.env` | `ensure_dotenv_loaded()` runs before run, resume and preflight. `cli.md` now puts the flag before the subcommand. | C | merged |

Also fixed:
- **merged:** A4 duplicate abstract span; A7 per-paper retrieval; IT2-N3 failed lookup cached and the PMC/PMID
  double count; A14 DOI/journal now filled; report evidence-level column; `artifacts.md` prompt version.
- **not approved:** B3 paper-metadata species; N-5 prompt pre-fill (prompt now `mining-extract/1.2.0`); support
  counting (reviews and non-supporting siblings never counted as independent support; an admitted node is never
  relabelled by a lead).

## Review verdicts

| change set | round | pwml-reviewer | pwml-bio-auditor | red team (bio-auditor) |
|---|---|---|---|---|
| A evidence hygiene | 0 / r1 / **r2 (final)** | APPROVE_WITH_FINDINGS / REQUEST_CHANGES / **APPROVE_WITH_FINDINGS** | SOUND_WITH_FINDINGS (2 blocking) / SOUND_WITH_FINDINGS / **SOUND_WITH_FINDINGS** | — |
| C run control | 0 / **r1 (final)** | REQUEST_CHANGES / **APPROVE_WITH_FINDINGS** | SOUND_WITH_FINDINGS / **SOUND_WITH_FINDINGS** | — |
| B species reading | 0 / r1 / r2 (cap) | REQUEST_CHANGES ×3 | SOUND_WITH_FINDINGS / UNSOUND / UNSOUND | UNSOUND ×3 |
| D loss-of-function rule | 0 / r1 / **r2 (cap)** | REQUEST_CHANGES / REQUEST_CHANGES / **APPROVE_WITH_FINDINGS (fit: YES)** | UNSOUND / UNSOUND / **UNSOUND (fit: NO)** | UNSOUND / UNSOUND / **UNSOUND (fit: YES, under audit)** |

Every review and brief is committed under `docs/pathway_mining/evidence/orch745/` on the experiment branch.

**The two open blockers**, both in `species.py` on `686ae134`:
- **RT-D-R2-1 / BIO-D-R2-1.** The repair-2 "closed relative clause" rule treats a clause as closed at any comma, so
  an inner comma fools it: "(Fig. 3A, B)", "(9S,13S)", or ", which, after wounding, …". The head participant then
  gets the opposite predicate. On modelled jasmonate prose this admits a product-up / substrate-down reading, or the
  reversed CYP94B3 reaction. This is a weakening relative to `fd7ea63a`. The rule is the red team's own round-1
  prototype, which my brief prescribed.
- **BIO-D-R2-2.** The negated-comparison and delay window still stops at an abbreviation full stop. "The opr3 mutant
  did not accumulate OPC-8:0 (Fig. 3A) until 6 h after wounding." is credited as an absence.

Neither route was found in the reviewers' real-corpus scans (about 2,500–3,200 loss-of-function sentences per
reviewer, from up to 358 texts). The red team found the punctuation shape twice, in non-plant prose, and never the
whole route. The probes are modelled prose.
Fixes: `evidence/orch745/prototypes/rev745d_r2_proto.py` (fail-closed parenthesis and opener handling) and
`rev745d_r2_proto_win.py` (window `[^;]{0,60}?`). With both active, the suite stays 856 and the replay is unchanged.

## G9 evidence (`docs/pathway_mining/evidence/`, README updated)

`036` C (12/12 fail at `f9556288`) · `037` A (16 cases) · `038` B (8) · `039`/`040` C repair (at `d01e0825`; F3-ORDER
passes at `f9556288` as a preservation proof) · `041` A repair 1 (21 at `6ceef736`) · `042` B repair 1 (71 at
`4f8d33a3`) · `043` A repair 2 (12 at `92ab5be9`) · `044` B repair 2 (118 at `fd53a301`) · `045` D (51 at
`fd7c737c`) · `046` D repair 1 (75 at `6454ef98`) · `047` D repair 2 (25 at `fd7ea63a`). Gates: `048`/`049` (merged
A+C), `050`/`051` (species branch).

## Not fixed (registered)

- **Species gate residuals.** All are documented in `known_limitations.md` on the species branch.
  - The rule is lexical.
  - Unlisted plant genera (10 of 10 unlisted binomials credited) and regulators written as the enzyme get through.
  - Double mutants, where the other gene explains the result, get through.
  - A background sentence about Arabidopsis inside another organism's paper gets through.
  - Precision rests on the paper-level organism screen: it held back 27 of 29 creditable loss-of-function sentences
    in the real corpus.
  - A participant's model citation is never checked against the span's text (BIO-745B-2).
- **Recall costs of the fail-closed design.** These are leads, not admissions:
  - rule 5(a) conversions ("In Arabidopsis leaves, OPDA is converted to OPC-8:0 by OPR3");
  - extract-based measurements, hyphenated alleles ("jar1-1"), "reduced but not abolished", verb-first review prose;
  - discussion and conclusion sentences.
- **From the handoff catalogue, still open:**
  - B4 cascades recorded as one node;
  - A9/A17 entity splits;
  - B5 output headroom;
  - the screen still misses At enzymology papers (Otto 2016 AOC, OPR3 and 12-OPDA papers);
  - N-10, N-9 (screen hosts), N-11;
  - the PMC13358708 residual;
  - N-1 enzyme verbatim recovery;
  - N1 zero-result exhausting on a network error;
  - per-batch lineage, bridge anchors, query-id collisions, `cache_hit`;
  - the remaining `report.md` count misstatements.
- **CS-A non-blocking items:**
  - `&lt;` followed by a letter is still read as a tag;
  - the restart loop is unpinned;
  - eLife peer-review text and the Wiley converter block are indexed.

## Production follow-ups (not touched; D-102)

- `rag/acquire.py:536`: the PubMed parser takes a cited reference's PMCID/DOI. Mining works around it.
- `llm/client.py`:
  - provider routing is unpinned and unreported;
  - the reasoning setting is never sent;
  - `response_hash` is partial;
  - usage is only cumulative.
- `rag/embed`: the lexical-fallback embedding is cached under the real model key.
- `rag/eligibility.py`: the organism lexicon holds two plants, so the paper-level screen cannot see most plant hosts.
  The species gate leans on that screen.
- `rag/ingest._split_sections`: first-occurrence headers and the `body` fallback mislabel sections.

## Decisions for the product owner to revisit

1. **Opening change set D past change set B's repair cap**, and **stopping instead of a third set**. To unblock:
   authorise one more narrow round with the two committed prototypes, or run the gate as is under audit (two of
   three final reviewers said fit for pilot).
2. **Rule 5(a) removed.** In-planta credit now comes only from loss-of-function genetics; discussion and conclusion
   sections earn no credit.
3. **Effect polarity.** Product down or absent, or substrate up. This narrows the adjudicated rule 5(b).
4. **Reviews never count toward independence.** Review plus one primary is `single_paper` with a corroboration note.
5. **An `unknown` paper without a failed lookup** keeps primary-strength credit (an interim position).
6. **Fed-precursor and occurrence evidence** (the JA-conjugate hinge) is held back.
7. **Whether a participant's model citation must be verified** against its span's text (BIO-745B-2).

## Worktrees left in place

| path | state |
|---|---|
| `C:/t/mining` | `experiment/pathway-mining` @ `82fc69a7`, clean |
| `C:/t/m745-a` | `orch745/a-evidence-hygiene` @ `7f5b94be` (merged) |
| `C:/t/m745-c` | `orch745/c-run-control` @ `beb6d70e` (merged) |
| `C:/t/m745-b` | `orch745/b-species-reading` @ `686ae134` (species gate, **unmerged**) |
| `C:/t/mining-rep-base` | detached @ `f9556288`, clean (G9 base arms) |

Scratch: `C:/t/bt/orch745/` (briefs, reviews, G9 run outputs), `C:/t/bt/m745-{a,b,c}/`, `C:/t/bt/rev745*/`. Nothing
was pruned.
