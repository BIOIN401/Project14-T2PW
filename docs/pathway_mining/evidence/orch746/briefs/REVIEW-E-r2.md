# ORCH-746 change set E — FINAL review (repair round 2, the last allowed round). Shared brief for all three reviewers

**What you review.** T2PW pathway-mining species gate, worktree `C:/t/m745-b`, branch `orch745/b-species-reading`.
- Repair-2 delta: `git -C C:/t/m745-b diff 01d436cf..f3df072f` (7 commits; `species.py`, `tests/test_mining_species_cs_e.py`
  (appended; one comment-only edit), `species_policy.md`, `known_limitations.md`, `testing.md`).
- Whole change set E: `686ae134..f3df072f`. `686ae134` is the species gate ORCH-745 could not approve; `5eca087b` is E's
  round-0 tip; `01d436cf` is repair 1's tip.

**History.** ORCH-745's final review left two blockers (an inner comma fooling `_excluded`; the negated-comparison window
stopping at an abbreviation). E round 0 (`5eca087b`) made an effect count only when several readings of the clause agree and
widened the window. Round-0 review (`C:/t/bt/orch746/reviews/cse_r0.txt`): three blocking routes (an appositive comma pair; a
negated delay with other absence forms; no-difference prose). Repair 1 (`01d436cf`) closed those with reviewer-prototyped
readings and vetoes. Round-1 review (`cse_r1_code.txt`, `cse_r1_redteam.txt`, `cse_r1_bio.txt` in the same folder): all three
reviewers blocked on the SAME two classes (a delay / time course; a no-difference result: non-significance or a result shared
with a control) phrased in ordinary words the lists missed, and all three said fit for pilot once their measured vetoes were
in. Repair 2 (`f3df072f`; brief `C:/t/bt/orch746/briefs/CS-E-repair2.md`) closes those classes by SHAPE, as the union of all
three reviewers' prototypes:
1. time course: two time or stage points in one sentence; a later presence after "but" / ";"; a union word list (`till`,
   `later`, `thereafter`, `subsequently`, `delay`, `only after/at/...`, `became detectable`, `by <n> <unit>`, `(during|within|in)
   the first`, early windows, `<n> <unit> only`, ...);
2. non-significance: the adjectives `statistical` / `significant` and the noun `significance` anywhere (the adverb
   "significantly" stays allowed), `trend`, `tendenc`, `(ns)`, `n.s`, p-value phrases, range / variation / variability /
   inconsistency / "in one of three experiments" phrases;
3. a result shared with a control: a wild-type / control genotype word (wild type, Col(-0), WT, Ws, Ler, control(s), parental,
   parent) in a sentence with no `than`; "any / all / each / either genotype(s)", "the same was true", `nor`;
4. concessive (sentence-initial while / whilst, whilst, even if, notwithstanding, regardless of, albeit);
5. `nowhere near`; 6. background wording in `_CITATION` (known to, shown previously, well documented / established, it is known);
7. "lower leaves / part / epidermis ..." (a position); 8. regulator wording in the marker tail and after the gene
   (`_MARKER_OF`, `_CONTAINER`); 9. a participant directly after the effect word joins the subject candidates (attributive
   "reduced X"), so a substrate-and-product pair is ambiguous.

**Evidence so far.**
- Implementer: full mining suite at `f3df072f` = **1127 passed** (856 at `686ae134`, 915 at `5eca087b`, 1005 at `01d436cf`), 0
  survivors. Probe matrix `C:/t/bt/m746-e/probe_matrix_r2.txt` (721 probes: every earlier probe and all three round-1 reviewers'
  batteries; scripts in `C:/t/bt/m746-e/`): TSS 495 / 419 / 283 / 197 (01d436cf + round-1 prototypes) / 163 at the tip;
  **0 rows credited at the tip that `01d436cf`, `5eca087b` or `686ae134` refuses**; 120 rows lose credit against `01d436cf`, of
  which 11 are correct biology (listed in `known_limitations.md`, e.g. "OPDA accumulated to high levels, but OPC-8:0 was not
  detected." lost to the later-presence shape; "undetectable in jar1 mutants but abundant in Col-0" lost to the control shape).
- Orchestrator's G9 base arms, pinned in `C:/t/mining-rep-base`: `052` (44/44 at `686ae134`), `053` (82/82 at `5eca087b`),
  **`054`: all 103 `test_cs_e_r2_746_*` rows fail with `AssertionError` at `01d436cf`**, 0 survivors, pin violations `[]`
  (`C:/t/mining/docs/pathway_mining/evidence/054-g9-746e-r2-at-01d436cf.json` / `.pin.json`).
- The implementer's own list of shapes still credited at the tip (each documented in `known_limitations.md` with what a sentence
  must avoid to slip past): a later presence written without "but" / ";" / a listed word ("..., and rose sharply by the next
  sampling point"); two time points split by a semicolon; non-significance with no listed word ("small and not reproducible");
  a shared result with no control word ("or in the complemented line"); the true subject more than two tokens after the effect
  word; a regulator gap word outside the lists ("a partner of AOS"); unlisted background wording ("an established JA-Ile-deficient
  line"); CW1 and the semicolon-in-reference comparison; RT-D-R2-2; "only marginally / slightly less" (a product-owner policy item).

**Binding.** Biology spec `C:/t/mining/docs/pathway_mining/evidence/orch745/briefs/F1_BIO_ADJUDICATION.md` (T1, T1b, T2-T11; the
full pilot-003 replay admits exactly #3, #14, #34: #14 in_planta on ev_7d2787cbb14b4905[7], #3/#34 review / review_article_only
on ev_c98d84c7d84855c1[12]). Product-owner rules: repairs only subtract credit; keep the fail-closed policy; fed-precursor and
occurrence evidence stays held back; **a thin pilot is acceptable, a biologically unsafe pilot is not.**

**SEVERITY STANDARD (unchanged).** 'blocking' = (a) a gate weakened relative to `686ae134` (or relative to `01d436cf` for this
round's delta), (b) a contract violation, or (c) a route to an admission against policy DEMONSTRATED end to end (build_claims ->
ClaimStore -> admission.judge / MiningController._admit gives accepted) with realistic plant-biochemistry prose. A contrived
paraphrase inside the residual class that `known_limitations.md` documents = non_blocking. For every blocking finding name the
clause and give the end-to-end evidence path, AND say honestly whether the prose is ordinary results prose in this literature
(how you know) or a constructed paraphrase inside a documented residual. For every non_blocking one say why. **This is the LAST
round: no repair follows. A blocking finding means change set E does not merge and ORCH-746 stops BLOCKED; be exact.** Also
answer **fit_for_pilot YES/NO with a reason**: the next step is ONE pre-registered live pilot (A. thaliana, "jasmonate
biosynthesis", seeds 12-OPDA + AOS) whose every admitted reaction is adjudicated by pwml-bio-auditor; its definition of done
requires zero admissions against policy.

**Process rules.** Read-only on every repository file. Write only under your fresh scratch dir `C:/t/bt/rev746e-r2-<role>/`
(never into any worktree, never into another `C:/t/bt` dir; every earlier reviewer's and the implementer's dirs
(`C:/t/bt/rev746e-r1-*`, `C:/t/bt/rev746e2-*`, `C:/t/bt/rev745d-r2-*`, `C:/t/bt/m746-e/`) are READ ONLY: copy what you need).
No recursive deletes. Every python/pytest via the bounded wrapper:
`cd C:/t/m745-b && PYTHONPATH=C:/t/m745-b/src C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py --label <l> --timeout <=400 --json C:/t/bt/rev746e-r2-<role>/runs/<unique>/report.json -- <py> -u <script, or docs/pwml_recovery_sprint/evidence/pinned_pytest.py --pin-verdict <path> -q -p no:cacheprovider --basetemp=<fresh dir whose parent exists> <tests>>`
(pinned_pytest takes NO `--`); set `PYTHONDONTWRITEBYTECODE=1`; Bash tool `timeout` 480000; never `&`; never `pytest -n`; check
`final_surviving_count` 0. For probes at earlier SHAs, use git archives in your scratch dir; never check anything out in a
worktree. No network, no LLM calls, no SMOKE.

**Your final message is your review, in this exact shape** (it is all the orchestrator sees):
```
VERDICT: <APPROVE | APPROVE_WITH_FINDINGS | REQUEST_CHANGES | REJECT | SOUND | SOUND_WITH_FINDINGS | UNSOUND>
FIT_FOR_PILOT: <YES | NO> — <reason, and what the pilot auditor must watch>
SUMMARY: <what you checked and ran, with run paths, and what holds>
FINDINGS:
- <ID> [blocking | non_blocking] (<clause and realism, or why not blocking>) <file:location>
  PROBLEM: ...
  EVIDENCE: <run paths; tip vs 01d436cf vs 686ae134 states>
  FIX: <subtract-only, or "none">
```
