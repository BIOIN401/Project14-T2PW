# Accuracy lane (acc/verifier): verifier, split/summary rule, VD-18, interaction typing

Branch `acc/verifier` (from main `fd814b8e`). Biology change: **merge waits for PO sign-off.**
Work dir (scripts, logs, answers, scores): `C:/t/acc-runs/`. Strategy it implements:
`C:/t/strategy/accuracy/STRATEGY.md`.

STATUS: replay, policy, VD-18, typing and gates done; live 3x3 A/B in progress (§ 8-9).

## 1. What was built

| piece | where | flag (default) |
|---|---|---|
| Stage-1 reaction verifier + repair | `src/t2pw/service/reaction_verifier.py`, hook in `service/extraction.py` next to `transport_sides` | `T2PW_REACTION_VERIFIER` (off) |
| split / summary rule | same module, `apply_verdicts` | `T2PW_VERIFIER_GRANULARITY` = `merge` (only read when the verifier is on) |
| interaction typing from after-effects | same module, `apply_types` | `T2PW_INTERACTION_TYPING` (off) |
| VD-18 span templates | `src/t2pw/rag/admission.py`, `_EXTRA_PROSE_PATTERNS` | none (a parser fix; see § 5) |
| tests | `tests/test_reaction_verifier.py`, `tests/test_rag_admission_vd18.py` | |

**Verifier contract.** One model call per 30 Stage-1 rows (the Stage-1 model,
`OPENROUTER_EXTRACTION_MODEL`), after the Stage-1 boundary. The model gives a verdict and a quote per row;
everything after that is deterministic:

* `incomplete`: a missing substrate/co-product is added only if the name occurs in the paper text
  (EntB: `isochorismate -> 2,3-diDHB` gets `pyruvate`).
* `vague`: a generic name is renamed only to a name that occurs in the paper (`activated 2,3-DHB` ->
  `2,3-dihydroxybenzoyl-AMP`).
* `reversed`: sides swapped only when the model's quote is verbatim in the paper.
* `split` (and, deterministically, any row through a generic intermediate such as `intermediate I`):
  see § 4.
* `summary`: omitted only when the steps it summarises are all present as rows (graph path of 2+ rows).
* `off_pathway`: a branch (source or sink of the drawn pathway) is omitted; a row other rows feed AND
  drain is kept and flagged.
* `not_stated`: **flag only** (§ 7, dead end 2).
* A row the verifier changed that became identical to another row is folded into it (enzymes kept).
  Duplicates it did not make are left alone.
* Nothing is deleted silently: an omitted row is marked `scope_membership: out_of_scope` (the existing
  out-of-scope filter removes it before Stage 2, and its lock id is dropped so the manifest agrees) and
  keeps a `verifier` record `{verdict, action, reason, model_quote}` in `stage1_payload.json`; the full
  report is `reaction_verifier_report.json` in the run's diagnostics dir. Every repair appends an
  `audit_repair / audit_modified` lineage entry sourced to the seed paper.
* Quotes are checked against the paper with case/punctuation folded; short parentheticals
  (citations like `( 19 )`) may be left out by the model; no ellipsis, no rewording.
* Any error fails open (payload unchanged, error recorded).

## 2. Evaluation set (frozen)

| paper | what | gold | scorer |
|---|---|---|---|
| PMC12096016 | enterobactin, E. coli | eval_v2, 5 reactions (certified) | `eval_v2.reactions.score_leg` TABLE 1 |
| PMC12421875 | menaquinone, L. lactis | eval_v2, 9 (certified) | same |
| PMC12312563 | MenD, L. monocytogenes | eval_v2, 1 (certified) | same |
| PMC12657337 | MK-7 module, E. coli | eval_v2, 3 (labelled, not certified) | same |
| PMC5130987 | chlorophyll, Arabidopsis | V9 `gold_PW766996.json`, 12 paper-stated of 14 (pheophytin `PW_R2146856` and phytylation `PW_R2146854` flagged not-in-paper at scoring time; gold file unchanged) | `vision.bench`, PAPER_ONLY; precision = match rate |
| PMC13231680 | negative control | pinned_v1 (must yield 0 reactions) | row count |

Lipid A (PMC12444477): **not added** — it has only 2 pinned_v1 signatures; a reaction gold needs a full
read of the paper (strategy estimate: 1 day). Open question § 10.

Topic lines: `C:/t/acc-runs/topics/`. Scorer: `C:/t/acc-runs/scripts/accscore.py`.
Interaction labels (hand-checked, 104 entity pairs across the five papers; no gold interactions exist —
PW766996 has 0): `C:/t/acc-runs/ix_labels.json`, scorer `ixscore.py`.

## 3. Offline replay on the 14 saved A/B outputs (+2 chlorophyll, +2 negative-control runs)

Verifier calls on the frozen `stage1_payload.json` of every saved leg; the decisions are then carried to
`final_mapped.json` by `locked_reaction_id` (projection: rows Stage 2/RAG/audit added are untouched).
Two independent model samples (`answers2`, `answers3`; deepseek-v4-flash, $0.09 and $0.11 for 18
payloads). Final projection, policy `merge`, summed over legs:

| paper (legs) | gold matches, base -> verifier | precision (TABLE 1), base -> verifier | rows | gold lost |
|---|---|---|---|---|
| PMC12096016 enterobactin (7) | 32 -> **34** | 0.783 -> **0.902** / 0.907 | 46 -> 41 / 43 | none |
| PMC12421875 menaquinone (4) | 36 -> 36 | 0.800 -> 0.816 | 50 -> 49 | none |
| PMC12312563 MenD (2) | 2 -> 2 | 0.25 -> 0.40 | 8 -> 5 | none |
| PMC12657337 MK-7 (1) | 3 -> 3 | 0.714 -> **1.0** | 7 -> 5 | none |
| PMC5130987 chlorophyll (2) | 16 -> 16 | 0.415 -> 0.415 | 41 -> 41 | none |
| PMC13231680 negative control (1) | 0 rows -> 1 row (unchanged) | | 1 -> 1 | |

(two numbers: the two model samples; one number: both samples agree.) Macro precision over the five gold
papers 0.592 -> 0.707. Recall gain: the EntB row `isochorismate -> 2,3-diDHB` gets `pyruvate`, which turns
two B-leg R2 misses into matches. At Stage 1 itself (no projection) enterobactin is 31 -> 34 and
precision 0.804 -> 0.974 / 0.952.

Where it does not reach: what Stage 2, RAG and the audit add after Stage 1 is not verified. The MenD
`intermediate I/II` rows of `B-PMC12312563` were written by the audit (`audit_modified`), the negative
control's `meropenem -> hydrolyzed meropenem` row and the enterobactin `ferric-enterobactin`/`lactate`
rows came in through RAG/Stage 2. Chlorophyll's remaining "FPs" are mostly real paper steps named
differently from PathBank or outside the PathBank record (Glu-tRNA -> GSA, MV-Chlide a, divinyl
intermediates), which a verifier against the paper correctly keeps.

Scripts: `ask.py` (model calls), `evalv.py` (policies + projection), `summ_evalv.py`;
answers `C:/t/acc-runs/answers2`, `answers3`; scores `json/evalvg2.json`, `json/evalvg3.json`.

## 4. Split and summary steps (PO ruling b): `merge` wins

All four options were applied to the same model answers (the policy is deterministic), on both samples:

| option | MenD intermediates (PMC12312563) | enterobactin | any gold lost |
|---|---|---|---|
| **flag-and-omit** (`omit`): every split/summary row omitted and listed with its quote | P 0.25 -> 0.40 | 33 matches, P 0.90 | **yes: EntF `DHB-AMP + L-serine -> enterobactin` (R5) in 1 of 7 legs, both samples** |
| **merge into the paper's real step** (`merge`) | P 0.25 -> 0.40 | 34, P 0.902 / 0.907 | no |
| **keep** | unchanged, P 0.25 | 34, P 0.902 / 0.907 | no |
| combined (`combo`: split merged, else omitted; summaries omitted) | P 0.40 | 33 | yes, R5 as above |

**Winner: `merge`** (now the default of `T2PW_VERIFIER_GRANULARITY`). Why:

* the model labels a real one-enzyme step as a "summary" (EntF's three condensation cycles), so
  any rule that drops a summary on the model's word alone loses a certified gold reaction; `merge` drops
  a summary only when the rows it summarises are all present (2+ row path from its input to its
  output), which is lossless by construction;
* the MenD chain (`2-oxoglutarate -> intermediate I -> intermediate II -> SEPHCHC`) is the paper's own
  mechanism, and the model said "ok" to it in the first sample. `merge` therefore treats any row through
  a generic intermediate (`intermediate I`, `an adduct`) as split whatever the model says, and joins the
  chain's own rows into the whole step (`2-oxoglutarate + isochorismate -> SEPHCHC + CO2`), which is then
  folded into the paper's MenD row when it exists. No new name is introduced by a join;
* `keep` leaves the made-up intermediates in the PWML.

The split rows stay visible: each joined/omitted row keeps its `verifier` record and quote in
`stage1_payload.json` and in `reaction_verifier_report.json`.

## 5. VD-18 admission span parser

Replay corpus: every rejected RAG candidate in 194 stored `rag_admission_report.json` files with a
`roles_unassignable` refusal, plus all 16 negative-control reports (`C:/t/acc-runs/vd18_list.txt`).
Each distinct candidate is re-checked with `validate_evidence_span` under main's and this branch's
`admission.py` in one process (`scripts/vd18.py`, report `json/vd18-2.json`).

| | |
|---|---|
| `roles_unassignable` refusals in the corpus | 7708 (344 distinct candidates) |
| ... where it was the only reason (ORCH-724 counted 294 of these on its 10 papers) | 6626 (227 distinct) |
| ... already read by main today (D-095's nominalized template) | 582 (8) |
| ... still refused on main | 6044 (219) |
| **now read by the new templates** | **908 (4 distinct)**, all true enterobactin steps: EntC `chorismate -> isochorismate` (R1, "is first converted to isochorismate by EntC"), EntE `DHB -> DHB-AMP` (R4, "activation of DHB to DHB-AMP by EntE") |
| new readings on the negative control | **0** |
| candidates main accepts that the branch refuses | 0 |
| new false admissions | **0** |

The three templates (`copula_adverb_converted_by`, `nominalized_of_to_by`,
`catalyzed_nominalization_of_to`) bind the catalyst only from an explicit `by`/`catalyzed by` and only
when it is a protein symbol (EntC, MenF, ALAS2). The first version took any catalyst phrase and read 8
candidates, three of them wrong to admit: acetate-overflow chemistry in a menaquinone paper ("acetyl-CoA is
first converted into acetyl-Pi by phosphotransacetylase", "... catalyzed by acetate kinase") and a span
cut mid-name ("by Ent"). The symbol rule removed exactly those.

Against the strategy's success line ("admits >= 50% of the true refused steps") this is **short**: 908 of
6044 sole refusals (15%). The large remaining shapes need their own templates and their own review:
lists of nominalizations without a substrate ("formation of DHB by EntB", 1039+258), "EntE activates DHB
by adenylation, forming DHB-AMP" (521), "ALAS synthesizes ALA from succinyl-CoA and glycine" (137),
"ferrochelatase inserts ferrous iron into protoporphyrin IX to yield heme" (111), relative-clause chains
("..., which is phosphorylated by LpxK to produce lipid IVA", 159).

## 6. Interaction typing (PO ruling c)

**Measured first.** No gold interactions exist (eval_v2 lists regulation as excluded; PW766996 has 0),
so every distinct interaction (by paper and entity pair) in the saved runs was hand-checked against the
paper: 104 labels, `C:/t/acc-runs/ix_labels.json` (a list where either type is defensible; cofactor,
component and chelation rows are not judged). Correct type = what the paper's stated after-effect
supports; `binding` only for a physical interaction with no stated effect.

On the 14 A/B outputs + 2 chlorophyll runs (final payloads), **current interaction-type accuracy is
0.746** (101/135 judged). Every error has the same shape: **the paper states the effect but the row says
`binding` (or `regulates`, `allosteric_inhibitor_of`)**, 34 of 34; there are **0 wrong signs**. Since
PathWhiz stores only Activation/Inhibition, those 34 rows are withheld from the PWML today.

**Rule built.** The model reads the after-effect (target level/activity down, pathway blocked, product
lower, or loss of the regulator raising them -> inhibition; the reverse -> activation) and must give the
quote. A row changes only when (a) it carries no sign today, (b) the new type is activation or
inhibition, (c) the quote is the paper's (a clause may be left out; a negation or loss word may not).
A signed row is never flipped: in the first sample the only flip the model proposed was wrong (it quoted
"PIF1 and PIF3 caused increased expression of HEMA1" without the "loss of function of" before it).

Replay on the Stage-1 payloads (184 judged rows): **0.821 -> 0.886 / 0.837**, wrong signs 0 -> 0.
Examples changed: Fur `binding -> inhibition` ("holo-Fur binds to the promoters of three ent gene
clusters, silencing gene expression"), TOC1/CHLH `binding -> inhibition` ("antiphasic ... direct
suppression"), SCL27/PORC, GUN4/Mg-chelatase `-> activation` ("assists this step by stabilizing"), GBP/GluTR1
`-> activation` ("inhibits degradation of GluTR1": a double negative read correctly). What remains: rows
the model itself answers `binding` (sample 2 answered `binding` twice as often as sample 1 on the same
rows).

## 7. Dead ends and corrections (kept on purpose)

1. **Token budget.** First replay: deepseek-v4-flash spent the 8000-token budget on reasoning and returned
   empty or cut JSON (`finish_reason=length`). Raised to 32000; stopped and re-ran (`answers` -> `answers2`).
2. **"not stated" deletes certified steps.** With omission on `not_stated` (when neither quote was verbatim),
   sample 3 omitted MenH/MenC/MenE/MenB/MenI rows of `A-PMC12421875` (5 certified gold reactions, R3-R7)
   because their extracted quotes were not verbatim and the model said the paper does not state them. Now
   flag-only. Cost: FPs the model correctly calls invented (e.g. `menaquinone -> prenyl diphosphate`) stay.
3. **Off-pathway on the model's word.** Same risk class; now only a branch (source/sink) is omitted, a
   chain link is kept and flagged.
4. **Folding pre-existing duplicates.** Folding every identical pair at Stage 1 dropped the copy that the
   later stages had enriched (projection lost chlorophyll `PW_R2146844`). Now only rows the verifier itself
   changed are folded.
5. **Summary on the model's word** (`omit`/`combo`): loses EntF R5, § 4.
6. **The model's "split" verdict alone**: it called the MenD intermediate chain "ok" in sample 1; the
   generic-intermediate rule replaced reliance on it.
7. **Exact-quote rule for interactions**: refused 30 supported re-typings (citations, a dropped clause);
   replaced by the gapped rule with a negation guard.
8. **VD-18 any-catalyst templates**: 3 wrong readings out of 8, § 5.
9. **Writing code through `python -c` in bash** put a literal NUL byte into the module once (caught,
   rewritten); scripts are files now.

## 8. Baseline (main fd814b8e, live, 3 runs x 6 papers, DRAWS=3)

## 8-9. Live arms (IN PROGRESS)

Baseline = main `fd814b8e` in `C:/t/acc-base`; verifier arm = `175dd1b9` in `C:/t/acc-ver` with
`T2PW_REACTION_VERIFIER=1 T2PW_INTERACTION_TYPING=1 T2PW_VERIFIER_GRANULARITY=merge`. Both DRAWS=3, strict,
deepseek-v4-flash (default), interleaved per paper by `C:/t/acc-runs/scripts/driver_all.sh` through
`queue.sh` (holder `acc`, 4 GB floor, 10-min gap after every leg). Both trees read the same cached full
texts (seeded from the primary checkout; `scripts/check_cache.py`).

Legs completed so far (2026-11-20):

| leg | outcome | gold (final) |
|---|---|---|
| base1-PMC13231680 | pass, 0 reactions (3 interactions) | negative control: 0 rows |
| ver1-PMC13231680 | fail `no_reactions` (Stage 1 returned nothing; verifier made no call) | 0 rows |
| base1-PMC12312563 | review_required | R1, P 0.25 (4 rows) |
| base1-PMC12657337 | **timeout 5400 s**, nothing produced (the same deepseek heavy tail the A/B saw) | - |
| ver1-PMC12096016 | **crash** in post-pipeline: `AMBIGUOUS_RENAME_TARGET: 2 distinct compound names canonicalize to '2pyrocatechuicacid'` | Stage 1: R1-R5 all matched |

The ver1 enterobactin crash: Stage 1 of that draw wrote both `2,3-dihydroxybenzoic acid` and
`2,3-dihydroxybenzoate` (the verifier report shows both names in the rows before it touched them; none
of the 7 A/B enterobactin legs ever had the `-ate` spelling). `pwml/prefreeze_resolution.py` refuses to
merge two spellings that map to one compound and the run ends without a PWML. The verifier did not add
the name, but it also did not fold the bogus row (`DHB + ATP -> DHB-ate`, which it "repaired" by adding
DHB-AMP). Open question § 10.

Lost to infrastructure (re-queued, not scored): 3 legs whose paper fetch failed (`no_full_text`: a failed
fetch is cached as `""` and then served from cache forever; fixed in the lane trees by
`scripts/fix_cache.py`), and 1 verifier leg killed by the PC reboot (`C:/t/acc-runs/partial/`).

**To finish:** let `driver_all.sh` run to `DRIVER_ALL_DONE`, run it once more (it redoes every leg without
an `.end`), then `score_legs.py C:/t/lane-acc json/live.json "legs/*"` and `ixscore.py` on the legs'
`final_mapped.json`; the gate is per gold reaction: matched in no fewer verifier runs than base runs.

## 10. Gates, cost, open questions

**Gates on `2fba9118` (code = `175dd1b9`), one hold, 0 surviving processes** (`C:/t/acc-runs/logs/gates1.log`):
SMOKE 511 passed; golden-seam batch 2, c011 8, stage8 4, quarantine 24 passed (none edited);
equivalence run 8, app 14, default 6; vision 357 + 1 skipped; web 83; lane tests (verifier, VD-18,
adversarial admission) 104 passed.

**97-run PWML proof, flags off**: 88/97 identical to the stored `replay_out/base_on` — and main
`fd814b8e` itself gives the same 88/97 (`logs/noreg-main.log`; the stored baseline predates the
import fixes). Lane vs main: **97/97 byte-identical** (same sha per run).

**Cost**: $0.61 spent of $15 (replays $0.21, live legs $0.40 incl. the killed one). Verifier per paper:
about $0.005-0.02 (two calls; chlorophyll ~$0.02). Live verifier legs so far $0.03-0.09 per paper.

**Open questions for the PO**
1. Merge the verifier default-off and enable it on the server only after the live 3x3 confirms "no gold
   lost", or wait for the live arm before merging at all?
2. Stage-1 only: Stage 2/RAG/audit additions (MenD intermediates written by the audit, lactate and
   ferric-enterobactin from RAG, the negative control's meropenem row) are not verified. A second pass on
   the final payload (before export) is the next step; it needs a rule for rows that carry mapped ids.
3. `not_stated` is flag-only. Accept that invented-looking rows stay when their quote is not verbatim,
   or allow omission when the row is also unconnected (a source/sink)?
4. `AMBIGUOUS_RENAME_TARGET` ends a run when Stage 1 spells one compound two ways (acid / -ate). Out of
   this lane (pwml/); should the verifier fold such rows, or should prefreeze merge the spellings?
5. The 0.746 interaction baseline counts binding-where-the-paper-states-an-effect as wrong. Agree with
   that standard (it is what makes them exportable as Activation/Inhibition)?
6. VD-18 reaches 15% of the refused true steps, not 50%. More shapes (§ 5) one by one, each with a replay?
7. Lipid A gold (1 day of reading) to widen the eval set?
8. Fetch cache stores a failed fetch as an empty text forever (`rag/acquire.py fetch_full_text`). Out of
   lane; worth a fix ticket.
