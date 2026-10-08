# ORCH-745 CS-A — repair round 2 (the LAST allowed round; keep it minimal and safe)

Re-review of your repair-1 tip `92ab5be9`: code **REQUEST_CHANGES** (2 blocking, both introduced by repair 1);
bio **SOUND_WITH_FINDINGS** (A1/A2 closed; its one blocking item is for change set B, not you). Full text:
`C:/t/bt/orch745/reviews/csa_r1.txt`. Same rules (`COMMON_RULES.md`), same worktree/branch; commits on top of
`92ab5be9`; tests proving the corrections must FAIL at `92ab5be9` and pass at your new tip — list them in
`C:/t/bt/m745-a/g9_manifest.json` under `repair2_nodes`.

## Blocking

**B1 — R745A-6 legend anchoring is a regression.** With `_CAPTION`'s new lookbehind `(?<![\w.(])` and label
lookahead `(?=\s*[.:]|\s+[A-Z])`, a mention like "(Table 1 )" no longer ends a reference list, so a FALSE headed
list (the `_REF_HEADING` matching a table's "References" column, with author-year cells making it citation-dense)
runs to the end of the text and drops real body prose: PMC8510960 (spans 11/46,568 chars -> 4/16,514) and
PMC12925505 in the project's cache `data/rag_index/acquire_cache/fulltext` (read-only, in the primary checkout
`C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/` — READ ONLY, never write there). The bio-auditor separately
showed that legends starting "(A)", "|", a dash, a lowercase gene symbol, "13-LOX", a Greek letter or "Δ" are now
swallowed. **Decision: restore the 6ceef736 list-end semantics** (the list ends at the next Figure/Fig/FIG/Table/
TABLE/Scheme label OR mention, as before), keeping only A7's added `FIG`/`Fig` (no period) forms. Accept the
harmless pepper fragment (PMC12787732, "(Supplemental Table S3)") as before and drop/adjust the test that pinned
the narrowed behaviour (document the delta). Regression tests: PMC8510960 / PMC12925505-style text (copy the
needed excerpt from the cache into a fixture; keep the fixture small) keeps its body prose; the bio-auditor's
legend forms after a list are retrievable; the A2 leak strings stay absent.

**B2 — `_plain` corrupts abstracts.** It unescapes and then strips everything between a "<" and the next ">", so
"(P &lt; 0.05) and OPDA rose &gt;2-fold in <i>jar1</i>" becomes "(P 2-fold in jar1 ." — a quote the paper never
printed, indexed as abstract#0. Strip only tag-shaped markup (e.g. `</?[A-Za-z][A-Za-z0-9]*\b[^<>]*>`) so a "<"
followed by a space or digit is never a tag; fix it in `_plain` for every caller. Test: an EuropePMC abstractText
with "P &lt; 0.05" and a later `<i>` tag keeps the comparison verbatim (fails at 92ab5be9).

## Non-blocking (only these, small)
- **N1:** a test for the `load_explicit` same-article skip (a PMCID then the same article's PMID -> one paper + the
  "same article" error).
- **N2 / A1-R1:** add "Review Essay", "REVIEW ARTICLE", "Minireviews" to the subject-heading alternatives.
- **N4:** `front_matter_abstract` must return a real abstract: require a minimum length (e.g. ≥ 200 chars) and at
  least one sentence; otherwise nothing.
- **N5 / docs:** state the calibration corpus honestly in `known_limitations.md` (the pilot texts + the cache
  corpus you checked), and record A1-R1-UNKNOWN-WITHOUT-FAILED-LOOKUP as a product-owner decision there: a paper
  graded `unknown` WITHOUT a failed lookup (offline/upload, no EuropePMC record, bare PMCID with no title, PubMed
  "Journal Article" only, unread subject headings) remains eligible for primary-strength credit.
Before your final report, re-run your region/span comparison on the cache corpus (read-only) at 6ceef736 vs your
new tip and report any text whose kept body prose shrinks.

Run the full mining suite at your new tip. Final report covering this round only.
