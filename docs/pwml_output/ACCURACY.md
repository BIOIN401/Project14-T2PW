# Accuracy lane (acc/verifier): verifier, split/summary rule, VD-18, interaction typing

Branch `acc/verifier` (from main `fd814b8e`). Biology change: **merge waits for PO sign-off.**
Work dir (scripts, logs, answers, scores): `C:/t/acc-runs/`. Strategy it implements:
`C:/t/strategy/accuracy/STRATEGY.md`.

STATUS: round 4 fixes committed (38780ed0), gates pass with flags off; bigger live run (BASE vs verifier+typing) in progress (§ 13).

## 1. What was built

**Flag defaults (review R1).** Every new behaviour is behind a flag that is OFF by default; with all
four unset the branch behaves like main on the code paths it touches (tests
`test_flags_off_stage_one_hook_is_main`, `test_flags_off_admission_is_main`, `test_flag_off_is_main`).

| piece | where | flag | default |
|---|---|---|---|
| Stage-1 reaction verifier + repair | `src/t2pw/service/reaction_verifier.py`, hook in `service/extraction.py` next to `transport_sides` | `T2PW_REACTION_VERIFIER` | **off** |
| interaction typing from after-effects | same module, `apply_types` | `T2PW_INTERACTION_TYPING` | **off** |
| split / summary rule | same module, `apply_verdicts` | `T2PW_VERIFIER_GRANULARITY` | `merge`, **read only when the verifier is on** |
| VD-18 span templates | `src/t2pw/rag/admission.py`, `_VD18_PROSE_PATTERNS`, walked via `_prose_patterns()` | `T2PW_VD18_SPAN_TEMPLATES` | **off** (R1 F1: it was ungated and default-on in 175dd1b9) |
| tests | `tests/test_reaction_verifier.py`, `tests/test_rag_admission_vd18.py` | | |

**Must fix before enabling any flag** (review R1; ALL ADDRESSED on branch `acc/verifier-r2`, see § 11. Still required before enabling: the live 3x3 on the fixed code, and PO sign-off. Original list kept for the record:
live 3x3 re-runs only after these):
1. F3: the summary drop is not lossless: `_reach` lets a co-product (CO2, AMP) stand in for the
   product. Every non-currency output must be reached from the non-currency inputs.
2. F4: `off_pathway` omits on the model's word with no quote, and `_bridges` is dominated by currency
   (the LDH assay row was kept, real endpoints are not protected). Quote + non-currency bridging, or flag only.
3. F5: "a signed row is never flipped" covers only words in the PWML table (`allosteric_inhibitor_of`,
   `inhibited_by` count as unsigned); a quote the paper prefixes with a negation/loss word in the same
   sentence ("Loss of function of PIF1 caused ...") passes as a substring. Change only empty/unsigned words;
   refuse such quotes.
4. F6: the first live typing leg made a false signed edge (EntC/EntB "interact" -> activation). And
   unsigned interaction words already export at main (D-W26: Binding/Unspecified/Regulation), so § 6's
   "withheld from the PWML" is wrong: a wrong re-typing is a new wrong assertion.
5. F7: split/merge and incomplete are loosely anchored: a model `merged_step` is not tied to the row;
   `_chain_steps` joins unrelated chains sharing "intermediate I"; `incomplete` only appends (DHB-ate row).
6. F8: the lock manifest is written before the hook, so it still lists verifier-omitted lock ids
   (the "lock id is dropped so the manifest agrees" line below is wrong); `reaction_verifier_report.json`
   is written to the repo `tmp/` diagnostics dir, not the run output dir.
7. F9: no per-run cap on verifier calls / wall time. F10: tests for the paths above.
8. VD-18: prove PWML no-loss with the flag on (admissions reaching export) before enabling.

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
  out-of-scope filter removes it before Stage 2; its lock id is dropped from the row, but the lock manifest, written earlier, still lists it: R1 F8) and
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
  output). This was claimed lossless by construction; review R1 F3 shows it is not (a co-product can satisfy the path), see "must fix before enabling";
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

The table above is the replay of `175dd1b9` (three templates, default on). **Review R1 (F1) rejected
that**: an ungated, default-on admission change with no PWML-level no-loss proof (the replay only re-runs
the span check; nothing measured what the admissions do downstream), and the symbol rule is a proxy: the
copula template still read other-pathway chemistry when the paper writes a symbol ("acetyl-Pi is then
converted to acetate by AckA"), and all-capitals non-proteins ("by NADH", "by HPLC") passed as catalysts.

Now (R1 fix): the templates are behind `T2PW_VD18_SPAN_TEMPLATES` (**default off**; unset = main's
template list, tested); the copula template is **removed**; the catalyst symbol is `Xxx[A-Z]` or an
all-capitals symbol **with a digit** (NADH/HPLC/NMR/ATP refused); negative tests cover the AckA copula
sentence, "by NADH", "measured by HPLC", "by NMR", a cut name, negation, F-179, wrong direction and wrong
catalyst. Remaining limit, stated plainly: a parser cannot judge pathway membership. "conversion of A to
B by AckA" is read with the flag on, exactly as main's D-095 template already reads "conversion of A to B,
catalyzed by AckA"; refusing off-pathway chemistry stays with the pathway and connection rules.
Replay with the flag on, two templates (`json/vd18-3.json`; 150 of the 194 report files were still on disk): **2 distinct candidates / 84 refusals now read, both the true EntE step `DHB -> DHB-AMP` (R4); 0 on the negative control; 0 that main accepts now refused.** The EntC copula span (805 refusals) is no longer read.

History: the first version took any catalyst phrase and read 8 candidates, three wrong to admit
(acetate-overflow chemistry "by phosphotransacetylase", "catalyzed by acetate kinase", and "by Ent").

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
(Correction, review R1 F6: unsigned words already export at main as Binding/Unspecified/Regulation, D-W26; these rows are exported with the wrong type, not withheld, and a wrong re-typing would be a new wrong assertion.)

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

## 8-9. Live arms (PAUSED after review R1; must re-run after the "must fix before enabling" items)

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

**Paused 2026-11-20** (`C:/t/acc-runs/PAUSE`; driver stopped while its next leg waited at the gate, no lock held). It ran 175dd1b9, so its legs do not count for the fixed code. **To re-run:** let `driver_all.sh` run to `DRIVER_ALL_DONE`, run it once more (it redoes every leg without
an `.end`), then `score_legs.py C:/t/lane-acc json/live.json "legs/*"` and `ixscore.py` on the legs'
`final_mapped.json`; the gate is per gold reaction: matched in no fewer verifier runs than base runs.

## 10. Gates, cost, open questions

**Gates after the R1 fixes, code `84d79e15`, every new flag unset, one queue hold, 0 surviving processes** (`C:/t/acc-runs/logs/gates2.log`): SMOKE 511 passed; golden-seam batch 2, c011 8, stage8 4, quarantine 24 passed (no golden test edited; the branch adds only two new test files); equivalence run 8, app 14, default 6; vision 357 + 1 skipped; web 83; lane 217 passed (verifier, VD-18, admission adversarial, relation recall, service extraction stage 1, transport sides).

Earlier gates on `2fba9118` (code `175dd1b9`, before the R1 fixes; `logs/gates1.log`): SMOKE 511,
golden 2/8/4/24, equivalence 8/14/6, vision 357+1 skipped, web 83, lane 104, 0 survivors.

**What the 97-run PWML proof does and does not show.** `proof_generate.py` loads each stored
`final_mapped.json` and calls the exporter. It never runs Stage 1 and never runs RAG admission, so it
exercises **none** of this branch's code; "97/97 byte-identical to main" (88/97 against the stored
`replay_out/base_on` for both main `fd814b8e` and the lane, that baseline predating the import fixes) only
shows the export path is untouched. It is not evidence for flags-off equivalence. That evidence is the
flags-off tests (§ 1) plus, for the verifier and typing, the hook being inert by inspection with the flags
unset (review R1).

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

## 11. Round 2 (branch `acc/verifier-r2`, from 3ca54d0a): every "must fix before enabling" item

Flags stay OFF by default; nothing below runs with them unset (tests in § 1 still hold, now against a
frozen copy of main). No live run this round (unit and offline only).

| finding | what changed (`service/reaction_verifier.py` unless noted) | test(s) |
|---|---|---|
| F3 summary drop | `summary_covered`: a summary is dropped only when the KEPT rows make every non-currency product from its non-currency substrates; currency (CO2, AMP, PPi, NAD, ATP...) neither stands in for a product nor carries the walk. A one-enzyme step (EntF) is dropped only if another kept row already makes enterobactin from its inputs, i.e. never lost. | `test_summary_dropped_only_when_kept_steps_make_its_product`, `test_summary_is_not_covered_through_a_co_product` (the R1 DHNA/CO2 scenario), `test_a_mislabelled_one_enzyme_step_is_never_dropped` (the R1 EntF/AMP scenario) |
| F4 off-pathway drop | [Round 3, R3 P6: now flag only, never dropped.] Was: omitted only when (a) the model's quote is verified in the paper, (b) the row is isolated: it shares no non-currency name with any other row, so it is never a first, middle or last step of what the other rows draw (currency never connects), (c) other rows remain. Otherwise flagged, never dropped. The LDH assay row (shares pyruvate) and ferric-enterobactin (shares enterobactin) are now flagged. | `test_off_pathway_isolated_row_with_a_verified_quote_is_omitted_and_reported`, `test_off_pathway_never_drops_the_first_or_last_step[0,3]`, `test_off_pathway_without_a_verified_quote_or_connected_by_a_real_name_is_only_flagged`, `test_off_pathway_never_empties_the_pathway` |
| F5 typing guard | `retypable`: only an empty word or an unsigned Binding/Unspecified/Regulation word may get a sign; any word carrying a sign stem (inhibit, activat, repress, induc, negative, positive, ...) is signed whatever table it is (not) in, so `allosteric_inhibitor_of`, `inhibited_by`, `activated_by`, `negatively_regulates` are never flipped. `quote_unqualified`: the quote must occur in a paper sentence with no negation/loss word ("loss", "mutant", "not", "knockout", ...) before it that the quote left out. | `test_a_signed_word_is_never_flipped[8 words]`, `test_a_quote_that_drops_a_leading_negation_is_refused` (the R1 PIF1 case) |
| F6 false edge from a contact | `states_effect`: a re-typing quote must state an effect (inhibit, activat, increas, decreas, reduc, degrad, stabiliz, ...), not only a contact. The live EntC/EntB "interact to form a dynamic channeling system" -> activation is now refused. (Record: unsigned words already export at main as Binding/Unspecified; corrected in § 6.) | `test_a_quote_that_states_no_effect_does_not_type_the_row`, `test_unsigned_row_gets_the_sign_its_after_effect_states` |
| F7a unanchored merged step | The chain's own rows are joined first (no new names). A model's whole step is used only for a row through a generic intermediate, and only if it keeps every real (non-generic, non-currency) name of that row. A model "split" on a real row never replaces it (flagged). | `test_model_whole_step_must_keep_the_rows_real_names`, `test_model_split_on_a_real_row_never_replaces_it` |
| F7b shared generic key | Rows are linked only through a generic name that exactly one row makes and exactly one row uses, and only when they share a catalyst (or one names none); a group is joined only if every intermediate it makes it also uses. Two mechanisms that both say "intermediate I" are never joined. | `test_two_unrelated_mechanisms_saying_intermediate_I_are_never_joined`, `test_split_chain_is_joined_from_its_own_rows_and_folded_into_the_paper_row`, `test_generic_chain_without_the_paper_row` |
| F7c append-only incomplete | `rename` is applied for `incomplete` too (a WRONG participant is replaced, not kept next to the right one); the prompt says so. Folding compares non-currency names, so a joined step without the paper row's CO2 folds into the paper row. A string `add_*` is one name, never its letters; one-letter names are refused. | `test_incomplete_may_correct_a_wrong_participant`, `test_incomplete_with_a_string_adds_the_name_not_its_letters` |
| F8 lock manifest | `prune_lock_manifest`: the omitted (and folded) rows' lock ids are removed from `tmp/locked_reaction_manifest.json` (Stage 1 wrote it before the hook), so manifest and payload agree. The hook passes `manifest_dir=PROJECT_ROOT/tmp` (`service/extraction.py`). | `test_hook_prunes_the_lock_manifest_and_hands_the_report_to_the_run` |
| F8 report location | The full report goes into the run state (`reaction_verifier_report`); `batch/driver.py` `_add_common_artifacts` writes it into the run's own output folder as `reaction_verifier_report.json` (only when present, so the artifact set is unchanged with the flags off). The diagnostics copy stays. | same test, `test_batch_driver_writes_the_report_only_when_present` |
| F9 cap | `Budget`: at most `MAX_CALLS` = 6 model calls and `MAX_SECONDS` = 600 s per run (each call's timeout is clipped to what is left, at most 300 s); rows past the budget get no verdict and stay untouched; the report counts `calls_skipped_over_budget`. | `test_budget_caps_calls_and_rows_past_it_stay_untouched` |
| F10 tests | all of the above | 63 lane tests (`logs/ut9.log`) |
| R2 N1 | `test_flags_off_admission_is_main` now compares the flag-off template list, pattern by pattern, with a FROZEN sha256 of main fd814b8e's list (14 templates, computed from C:/t/acc-base by `scripts/main_pattern_hash.py`), not with the branch's own tuple. | `test_flags_off_admission_is_main` |
| R2 N3 | `tests/test_reaction_verifier.py` no longer imports another test module's fixtures; it has its own minimal stand-ins for `run_stage_one`'s callees. | `test_flags_off_stage_one_hook_is_main` |
| R2 N4 | `rag/admission.py` `_SYMBOL_ACTOR`: an all-capitals symbol needs 3+ capitals and a digit (ALAS2, POR1, CHL27); CO2, NO2, SO4 are refused. | `test_flag_on_still_refuses` ("by CO2" x2, "by NO2") |

**VD-18 PWML-level no-loss replay, flag on** (`scripts/vd18_admit.py`, `json/vd18adm-1.json`,
`logs/vd18adm-1.log`). [Corrected in round 3, review R3 Q3: VD-18 does NOT change only the span check. It also reaches the typed-resolution proposals (`admit_candidates` second return, `propose_typed_resolutions` via `_precursor_resolution`) and the vision literature-extension span check; see § 12 for the extended replay.] If the gate accepts
the same candidates with the flag on, the accepted reactions are the same. For every stored run
that still has `rag_admission_report.json` + `stage1_payload.json` + `plan.json`, the replay rebuilds the
gate's input (seed = in-scope Stage-1 payload, gaps = `detect_gaps` on it, candidates = every accepted and
rejected record) and runs `admit_candidates` with the flag unset and set.

* 240 reports are on disk; **20 runs are replayable** (220 lack a stored Stage-1 payload): 8 enterobactin,
  4 menaquinone (L. lactis), 3 MenD, 1 MK-7, 2 lipid A, 2 negative control, plus their RAG candidates
  (13 to 205 per run).
* Fidelity: the flag-off replay accepts exactly what the stored report accepted on **20/20**.
* Flag on: accepted set identical on **20/20**; **0 new admissions, 0 lost**, 0 on the negative control.
  The one span VD-18 newly reads in these runs (B4 enterobactin, "activation of DHB to DHB-AMP by EntE")
  is still refused by the later admission rules, so it never reaches the payload.
* Limit: 20 runs, and VD-18 reads few spans in them; this shows no loss, not a gain.

Still required before enabling any flag: the live 3x3 (baseline vs flags on) re-run on this code, and PO
sign-off.

**Round-2 gates, code `5bb5ad77`, every new flag unset, one queue hold, 0 surviving processes** (`C:/t/acc-runs/logs/gates3.log`): SMOKE 511 passed; golden-seam batch 2, c011 8, stage8 4, quarantine 24 passed (none edited); equivalence run 8, app 14, default 6; vision 357 + 1 skipped; web 83; lane 302 passed (verifier, VD-18, admission adversarial, relation recall, service extraction stage 1, transport sides, batch driver). Focused lane run: 63 passed (`logs/ut9.log`). Spend this round: $0 (no model calls); lane total $0.61 of $15.

## 12. Round 3 (review R3), branch `acc/verifier-r2` (main e17b78ba merged in at a34309af)

Code `4d623f2a`. Flags still OFF by default. Every change is unreachable with them unset, and the gates
below ran with all four unset.

| item | change | test(s) |
|---|---|---|
| P1 typing | `states_effect` matches whole effect word forms (inhibit/inhibits/inhibited, promote/promotes, reduce/reduced, up-/down-regulate ...), not stems. So "promoter", "reductase", "reduction", "inducible", "raised" and "accumulation" do not count as effects. | `test_contact_or_naming_words_are_not_effects` (5), `test_effect_words_are_effects` (6), `test_binding_to_a_promoter_stays_unsigned` |
| P2 typing | `split_sentences` cuts only before a capital letter (or bracket), and never after an abbreviation (Fig., et al., e.g., spp. ...) or a single-letter genus initial ("E. coli"). A negation or loss word that the quote leaves out before it refuses the quote, even when the quote has its own negation. | `test_a_genus_initial_does_not_hide_the_negation_before_it` (the R3 "Deletion of fur in E. coli ..." case), `test_a_dropped_leading_loss_is_refused_even_when_the_quote_has_its_own_negation` |
| P3 verifier | `Budget.take()` returns a per-attempt timeout and an attempt count whose worst case, computed by the client (`worst_case_call_seconds`, which prices attempts, backoff and SDK retries), fits the time left. So `MAX_SECONDS` = 600 is now a wall-clock bound across retries. `llm/client.chat_detailed` gained an optional `max_retries` parameter; its default is still `LLM_MAX_RETRIES`, so nothing changes for other callers. | `test_budget_is_a_wall_clock_bound_across_retries` |
| P4 verifier | A rename or an added name never creates a second spelling of a compound the payload already names. Spellings are compared with case, punctuation and acid/-ate folded. If the payload already has a spelling, that spelling is used. A rename is refused when other rows keep the old name. This blocks the AMBIGUOUS_RENAME_TARGET shape. | `test_a_rename_never_leaves_two_spellings_of_one_compound` (the acid/-ate case, plus pyruvate → the payload's "pyruvic acid") |
| P5 | Folding requires that, on each side, one row's currency is a subset of the other's. NAD vs NADP and ATP vs GTP are never folded, matching the Stage-1 ensemble rule. | `test_rows_differing_in_cofactor_are_not_folded` |
| P6 | `off_pathway` only flags. It never drops a row, even an isolated one with a verified quote. | `test_off_pathway_only_flags_never_drops` (3), first/last step (2) |
| P7 | Typing reads `interaction_type` before `relationship`, as the exporter does, and writes both. | `test_the_typing_word_is_read_as_the_exporter_reads_it` |
| P8 | Recorded limit, not changed. `prune_lock_manifest` is a new read-modify-write of the shared `PROJECT_ROOT/tmp` manifest. It is safe only while the server runs one job at a time, which is the existing shared-tmp design. | |

**VD-18 extended replay** (`scripts/vd18_admit2.py`, `json/vd18adm-3.json`, `logs/vd18adm-3.log`). It
covers the paths R3 Q3 listed. On the 20 replayable runs it compares, with the flag off and on: (1) the
accepted set and the AdmissionReport's typed-resolution proposals from `admit_candidates`; (2)
`propose_typed_resolutions`, run for every gap against every distinct stored evidence span, since the
retrieved chunks themselves are not stored. Separately, (3) it reruns the vision literature-extension
span check (`validate_evidence_span` as `vision/extend/literature.check_reaction` calls it) on all 30
stored V8 literature claims.

| measure | result |
|---|---|
| flag-off replay reproduces the stored accepted sets | 20/20 runs |
| accepted sets identical | 20/20 |
| typed-resolution proposals identical | 20/20 (none produced in either arm: these runs have no typed gaps, so this path is untested rather than shown equal) |
| `propose_typed_resolutions` on the stored spans identical | 20/20 (also none produced) |
| literature-extension span check, 30 stored claims | 0 changed verdicts |
| negative control (2 runs) | 0 differences |
| gold-matching gains | **0** |

**VD-18 stays OFF.** The replay finds no loss, but also no gain, and the orchestrator's rule needs at
least one gold-matching gain. The typed-proposal path could not be tested on stored data.

**Gates, code `4d623f2a`, all four flags unset, one queue hold, 0 surviving processes**
(`logs/gates5.log`):

| suite | result |
|---|---|
| SMOKE | 511 passed |
| golden seams (batch, c011, stage8, quarantine) | 2 / 8 / 4 / 24 passed, none edited |
| equivalence (run, app, default) | 8 / 14 / 6 passed |
| vision | 370 passed, 1 skipped |
| web | 83 passed |
| lane | 320 passed |
| client (own process) | 35 passed |

The two `tests/test_llm_client_*` files fail when run in one process after the stage-1/batch tests. They
fail the same way on the main-era tree (`logs/ut14.log`: 3 failed, auth error from a leaked client), so
the gates run them in their own process. Lane unit run: 81 passed (`logs/ut11.log`).

### Live 3x3 (review R3 section 4): IN PROGRESS

Design:
- One tree, `C:/t/acc-ver` at `4d623f2a`. BASE = all flags off; VT = verifier + typing, merge, VD-18 off.
- Strict, DRAWS=3, deepseek-v4-flash, the same cached texts, interleaved, a 10-min gap and a PAUSE check
  between legs. Batch limit 9000 s, wrapper 9600 s.
- Driver: `C:/t/acc-runs/scripts/driver3.sh` (log `logs3_driver.out`). Output: `legs3/`, `logs3/`.
  Scorer: `scripts/score3.py`.

Papers:
- **PMC12096016 enterobactin**: the verifier changes it most (EntB pyruvate, vague names, a summary
  mislabel) and it has interactions.
- **PMC5130987 chlorophyll**: the typing test bed (60+ interactions, many binding-vs-effect cases) and the
  long-paper stress test for the 600 s cap.
- **PMC13231680 negative control**: the 0-reaction gate and the false-signed-edge risk on an unrelated
  paper.

MK-7 and MenD are out: MK-7 timed out twice before, and MenD's split chain is covered offline.

Legs done so far:
- base1-PMC13231680: pass, **1 reaction** ("lipid A → lipopolysaccharide", a Stage-1 false positive at
  baseline).
- vt1-PMC13231680: Stage 1 empty, fail `no_reactions`, report present, 0 verifier calls.
- base1-PMC12096016: done.

Results per criterion go here when the 18 legs finish.

## 13. Round 4 (review R4), branch `acc/verifier-r2`

Flags are still OFF by default; every change below is reachable only with a flag on.

| item | change | test(s) |
|---|---|---|
| B1 second spelling (synonyms) | A renamed or added name that is not already a payload spelling is refused when it contains an abbreviation token: 2+ capitals in a token of at most 8 characters (DHB, DHB-AMP, CoA, SAM). Spelling alone cannot tell "DHB-AMP" is "2,3-dihydroxybenzoyl-AMP", and two spellings of one compound end the run (AMBIGUOUS_RENAME_TARGET). The payload's own exact spelling is still accepted. | `test_an_abbreviated_synonym_is_never_written_next_to_the_full_name` (the R4 DHB-AMP rename and a "DHB" addition; written to fail on 4d623f2a) |
| N1 agent nouns | "inhibitor(s)", "inhibitory", "activator(s)" and "repressor(s)" state a sign again ("2,3-DHB is a competitive inhibitor of apo-EntB"). "promoter" and "reductase" stay excluded. | `test_effect_words_are_effects` (+4 sentences) |
| N2 fail-open rate | `scripts/score3.py` reports, per paper, VT runs with the report missing, with a recorded verifier error (fail open), and with budget skips. | scorer |

**Test-isolation ticket (open, outside the lane).** `tests/test_llm_client_request_timeout.py` and
`tests/test_llm_client_empty_completion_retry.py` fail (3 tests) when run in one pytest process after the
stage-1 / transport / batch-driver tests: a client patched or rebuilt by an earlier test leaks through, and
the timeout tests see a real 401. Old main `fd814b8e` fails the same way (`C:/t/acc-runs/logs/ut14.log`).
Every gate script of this lane runs the two files in their own process.

**Criteria changes from R4**
- Negative control: per run, reactions after the verifier ≤ before. Per paper, VT reaction count ≤ BASE.
  The offline application on BASE's negative-control Stage-1 payload adds no row and no new signed edge.
- Pre-existing Stage-1 finding, BASE arm: run base1-PMC13231680 extracted "lipid A → lipopolysaccharide"
  from the negative-control paper with every flag off.
- Hard gates added: no AMBIGUOUS_RENAME_TARGET caused by a verifier rename or addition. For typing: 0
  changed rows hand-labelled binding/skip, 0 wrong sign, 0 already-signed rows changed, and type accuracy
  ≥ BASE.

**Offline re-application of the round-4 rules to the stored model answers** (no model cost; 18 Stage-1
payloads, both samples `answers2` / `answers3`; `json/evalv_r4_2.json`, `json/evalv_r4_3.json`; final
projection, policy merge, summed over legs):

| paper | gold matches, base → verifier | precision | gold lost |
|---|---|---|---|
| PMC12096016 enterobactin (7 legs) | 32 → 33 / 34 | 0.783 → 0.878 / 0.907 | 0 |
| PMC12421875 menaquinone (4 legs; 9 certified rows each) | 36 → 36 | 0.800 → 0.816 | 0 |
| PMC12312563 MenD (2) | 2 → 2 | 0.25 → 0.40 | 0 |
| PMC12657337 MK-7 (1) | 3 → 3 | 0.714 → 0.714 (off-pathway now flags only) | 0 |
| PMC5130987 chlorophyll (2) | 16 → 16 | 0.415 → 0.400 / 0.415 | 0 |
| PMC13231680 negative control (1) | 1 row → 1 row | | |

Interaction typing on the same payloads (`json/ixr4_2`, `json/ixr4_3`, 184 judged rows): accuracy
0.821 → 0.870 / 0.837. In both samples: 0 wrong signs, 0 changed rows labelled binding/skip, 0
already-signed rows changed.

**Live 3x3 on 4d623f2a: paused and folded into the bigger run.** The orchestrator decided to measure the
round-4 code directly. driver3 stopped after 4 scored legs (kept as evidence, `json/score3.json`):

| leg | result |
|---|---|
| base1-PMC13231680 | 1 reaction (the lipid A finding above) |
| vt1-PMC13231680 | 0 reactions (Stage 1 empty), report present, 0 verifier calls |
| base1-PMC12096016 | 5/5 gold matched, P 0.857 |
| vt1-PMC12096016 | 5/5 gold matched, P 1.0; 2 verifier calls, 10.5 s; nothing omitted or folded; lock removals 0 = 0; no typing change; no error |

The run then sat blocked for hours on free RAM (1.5 GB of 31 GB: the kernel zombie-process leak, which
only a reboot clears).

**Round-4 gates, code `38780ed0`, all four flags unset, one queue hold, 0 surviving processes** (`C:/t/acc-runs/logs/gates6.log`): SMOKE 511 passed; golden-seam 2/8/4/24 passed (none edited); equivalence 8/14/6; vision 370 + 1 skipped; web 83; lane 325 passed; client files in their own process 35 passed. Lane unit run 86 passed (`logs/ut15.log`); the new B1/N1 tests fail on 4d623f2a (5 failed, `logs/ut16.log`).

**Bigger run (folds in the 3x3)**: `scripts/driver4.sh` on `C:/t/acc-ver` at `38780ed0`; BASE all flags off vs VT verifier + typing (merge, VD-18 off); 6 papers (negative control PMC13231680, MenD PMC12312563, enterobactin PMC12096016, sour taste PMC7059786 with its curated PW739181 interaction, menaquinone PMC12421875, chlorophyll PMC5130987) x 3 draws x 2 arms, interleaved, 10-min gaps, 9000 s legs; resume-safe after a reboot (skips scored legs, moves an interrupted leg aside, `logs4/inflight.txt`), kernel process objects and free RAM logged to `logs4/mem.log` before every leg, and it pauses itself (PAUSE) under 5 GB free. Scoring: `scripts/score4.py` per leg against the R4 criteria, plus the offline paired application on every BASE Stage-1 payload. Results go in a later section.

## 14. Bigger run on Gemma (`google/gemma-4-31b-it`, both arms, code `82bda3b6`), and two fixes it found

PO direction: users run Gemma, so the bigger run measures Gemma end to end (Stage 1 to verifier) in both
arms. `scripts/driver4g.sh` on `C:/t/acc-ver` at `82bda3b6`; BASE all flags off vs VT (verifier + typing,
merge, VD-18 off); 6 papers x 3 draws x 2 arms, interleaved, 5-min gaps, 9000 s legs. The deepseek
legs already run under driver4 are kept as evidence (`json/score4_deepseek.json`). Model confirmed
in every VT report (`model: google/gemma-4-31b-it`).

**Status: 35 of 36 legs scored. PAUSED_LOW_RAM before vt3-PMC5130987 (free 3.9 GB, 123k kernel
process objects, the zombie-process leak); `C:/t/acc-runs/PAUSE` exists; restart `driver4g.sh` after a
reboot and it runs only that leg.** Scores: `json/score4g.json`, table `scripts/summ4g.py`.

| paper | arm | PWML | gold matched (runs per id) | mean P | VT verifier: max calls / max s, errors | R4 verdict |
|---|---|---|---|---|---|---|
| PMC13231680 negative control | base | 2/3 (base1: no reactions) | - | - | | |
| | vt | 3/3 | - | - | 1 / 9.1 s, 0 | pass: 0 rows in every run, VT ≤ BASE |
| PMC12312563 MenD | base | 3/3 | R1 3 | 0.35 | | |
| | vt | 3/3 | R1 2 | 0.22 | 2 / 338 s, 1 fail-open | **fail** live (R1 2 < 3; see F-G1); lock check fails on vt3 (F-G1) |
| PMC12096016 enterobactin | base | 3/3 | R1 3, R2 1, R3 3, R5 3 | 0.64 | | |
| | vt | **1/3** | R1 2, R2 1, R3 2, R4 1, R5 2 | 0.68 | 2 / 15 s, 0 | **fail**: VT no-PWML 2 > BASE 0; EntF row omitted on vt3 (F-G2) |
| PMC7059786 sour taste | base | 3/3 | 0 rows; interaction 0/1 | - | | |
| | vt | 3/3 | 0 rows; interaction 0/1 | - | 2 / 5 s, 0 | pass (no change) |
| PMC12421875 menaquinone | base | 3/3 | R1-R9: 3,3,2,2,2,2,2,3,3 | 0.82 | | |
| | vt | 3/3 | identical | 0.81 | 2 / 18 s, 0 | pass |
| PMC5130987 chlorophyll | base | 2/3 (base1: AMBIGUOUS_REFERENCE "tRNA Glu", flags off) | 7 ids x 2 | 0.41 | | |
| | vt (2 of 3) | 2/2 | 7 ids, R2146840 once | 0.36 | 2 / 17 s, 0 | open (vt3 pending); the verifier changed no row in vt1/vt2 |

Every VT run: report present, at most 2 of 6 calls, at most 338 s of 600, no AMBIGUOUS_RENAME_TARGET, 0
typing changes. Fail-open rate: MenD 1/3, every other paper 0/3.

**Offline paired application** (`scripts/ask4g_run.sh`, Gemma as verifier, on all 18 BASE Stage-1
payloads; `answers4g/`, `json/evalv_g4_r4.json`, `json/evalv_g4_r5.json`; merge policy):

| paper | gold matches base → verifier (Stage 1, summed over 3 legs) | gold lost |
|---|---|---|
| enterobactin | 10 → 11 | 0 |
| MenD | 3 → 3 (P 0.63 → 1.00) | 0 |
| menaquinone | 22 → 22 | 0 |
| chlorophyll | 24 → 24 | 0 |
| negative control | 0 rows → 0 rows, no signed edge | |
| sour taste | 0 → 0 | 0 |

Typing replay on the same payloads: 0 rows changed (Gemma proposes no retypable change), so typing
accuracy equals BASE. One offline typing call failed with invalid JSON (base1-PMC12312563, 229 s).

**F-G1 (fixed in `84205bd5`): a failed typing call discarded the finished reaction step, but its lock
prune still ran.** vt3-PMC12312563: the typing reply was invalid JSON (`Invalid \uXXXX escape`, 335 s).
`verify_stage_one` failed open and returned the original payload, yet the hook pruned the 3 lock-manifest
entries of rows the reaction step had omitted (rows kept, entries gone). The reaction step would have
appended the whole MenD step (R1), which this Stage-1 draw lacked. Fix: fail-open per step; the payload
of the last finished step comes back, so `report["reactions"]` and the prune always match the payload.
Test `test_failed_typing_call_keeps_the_finished_reaction_step_and_its_lock_prune`. The scorer now
checks that every pruned lock id is really gone from the payload (`pruned_but_present`).

**F-G2 (fixed in `32620fde`): a summary was "covered" through a co-substrate and a row the model
called wrong.** vt3-PMC12096016: the real EntF step (DHB-AMP + L-serine → enterobactin, gold R5) was
called a summary and omitted, because L-serine reached enterobactin through "DHB + L-serine →
enterobactin", a row the verifier itself judged not_stated. Fix: each product must be reachable from
EACH substrate, and rows judged not_stated / off_pathway never count as coverage. Test
`test_a_co_substrate_or_a_wrong_row_never_covers_a_summary` (old rules drop the row, new keep it).
Offline re-application on every cached answer set (Gemma `answers4g`, deepseek `answers2`, `answers3`):
identical to the R4 rules, 0 gold lost. Lane unit run: 88 passed (`logs/ut-cover.log`).

**F-G3 (open, outside the lane): Gemma runs die downstream when "2,3-dihydroxybenzoyl-AMP" is in play.**
VT enterobactin produced no PWML twice. vt1: Stage 1 itself named the compound "2,3-dihydroxybenzoyl-AMP"
(the verifier changed no row) and post_audit stopped on an entity with an empty name. vt3: the verifier
renamed "activated 2,3-DHB" to "2,3-dihydroxybenzoyl-AMP" (vague repair, quote in the paper) and a
RAG-added EntF row used the same name; two unresolved registry shells with that name reached the
exporter (PWML_IR_DUPLICATE_NAMED_ROW). All 3 Gemma BASE runs used "activated 2,3-DHB" and passed.
Under deepseek, 4 of the 5 earlier legs with the full name produced PWML. So the trigger is the full name under Gemma's
downstream (gap resolver / RAG), and the verifier's rename can supply it. Options for the PO: (a) leave
renames on and fix the duplicate-shell path in the pipeline (biology code, outside this lane); (b) make
the vague rename flag-only under Gemma, giving up its precision gain; (c) refuse a rename whose new name
is already a registry name (cheap, untested live).

**Spend:** Gemma legs + offline asks $1.98 (`logs4g/*.usage.jsonl`, `logs/ask4g-*.usage.jsonl`).

### 14a. F-G3 option (c), implemented in `e72c2759` (orchestrator decision)

- A rename is refused when its target is already carried by another entity or row of the payload, by the
  spelling key or by `t2pw.pipeline.name_keys.reference_key`, the key the exporter's duplicate-named-row
  check groups on (`pwml.ir._norm`). Names the same pass already added count ("will exist").
- `_add_compound` (additions, merged whole steps) dedupes on `reference_key`, so the verifier never writes
  a second entity row on the exporter's key. Additions of a name that already exists are NOT refused (the
  row references the existing entity; accepted by the orchestrator), so the EntB pyruvate co-product
  repair stays.
- New test `test_a_rename_onto_another_entitys_name_is_refused` (the vt3-PMC12096016 shape, exact name and
  a reference-key variant, plus an addition that must not create a second entity) fails on `4eb9cd22`.
- **Three lane tests rewritten** because they asserted the now-banned rename onto an existing name:
  `test_incomplete_may_correct_a_wrong_participant` (now renames pyruvate to a name no entity carries),
  `test_vague_rename_then_fold_into_the_identical_row` (replaced by the new test above), and the last
  clause of `test_an_abbreviated_synonym_is_never_written_next_to_the_full_name` (now asserts refusal).
  With R4 B1 still in place, a rename target with an abbreviation token (AMP, CoA) is now always refused:
  the old exact-spelling path that bypassed B1 is gone.
- **Cost, offline re-application (final projection, merge, 0 gold lost everywhere):** enterobactin
  precision on deepseek answers 0.878 → 0.844 (`answers2`) and 0.907 → 0.867 (`answers3`), from losing
  the rename-then-fold of duplicate EntE rows; every other paper unchanged; Gemma `answers4g` unchanged.
  `json/evalv_fg3_{2,3,4g}.json`. Lane unit run 88 passed (`logs/ut-fg3b.log`); new tests on 4eb9cd22: 2
  failed (`logs/ut-fg3b-old.log`).

### 14b. Ticket (outside the lane, for the PO; not fixed here): Gemma runs abort at post_audit on a nameless entity row

**Correction to F-G3 above:** this is not tied to "2,3-dihydroxybenzoyl-AMP". The BASE arm (every flag
off) hit the same abort on the negative control, and the 14 table's "base1: no reactions" for
PMC13231680 is wrong: that run aborted the same way.

| run | arm | paper | failing step |
|---|---|---|---|
| vt1-PMC12096016 | VT (the verifier changed no row; it flagged rows 4 and 6 only) | enterobactin | post_audit |
| base1-PMC13231680 | BASE, flags off | negative control | post_audit |

- **Failing step:** `t2pw.pipeline.stage_contracts.validate_post_audit` (effect `abort`), error
  `entity_missing_name` "Every entity row must have a non-empty name." (stage_contracts.py:657), reported
  as "Audit and DB mapping stopped at the post_audit stage boundary". `mapped_failure_snapshot.json`
  records `failing_codes: ["entity_missing_name"]`.
- **Where the nameless row comes from:** not Stage 1 and not the verifier. `stage1_payload.json` and
  `merged_payload.json` of both runs have no entity with an empty name, so the row is created between
  merged_payload and the post_audit boundary (audit loop / gap resolver / DB mapping). The run folder
  does not persist the contract's `row_digest` or pointer, so the bucket and index are not recoverable
  from the artifacts: an observability gap worth fixing with the ticket.
- **Repro payloads:** `C:/t/acc-runs/legs4g/vt1-PMC12096016/2026-11-23_1322/papers/PMC12096016/strict/`
  and `C:/t/acc-runs/legs4g/base1-PMC13231680/2026-11-23_0014/papers/PMC13231680/strict/`
  (`stage1_payload.json`, `merged_payload.json`, `mapped_failure_snapshot.json`, `LEG_TRACE.jsonl`).
  vt1 also carries two compound rows named "2,3-dihydroxybenzoyl-AMP" in merged_payload (one
  `reconstructed_from_process_participant`, one `inferred`); whether that pair is related is open.
- **Frequency:** Gemma 2 of 35 legs (1 BASE, 1 VT; `legs4g`). Deepseek 0 of 19 legs (`legs` 5,
  `legs3` 4, `legs4` 10).

### 14c. After the reboot: gates8, the last Gemma leg, and the focused re-measure on the fixed code

**gates8, code `e72c2759` (HEAD `4d29c4bf`), all four flags unset, one queue hold, 0 surviving processes**
(`C:/t/acc-runs/logs/gates8.log`): SMOKE 511 passed; golden-seam 2/8/4/24 passed (the four golden files
are unchanged since `e17b78ba`); equivalence 8/14/6; vision 370 + 1 skipped; web 83; lane 327; client
files in their own process 35.

**Bigger run complete (36/36).** vt3-PMC5130987 on `82bda3b6`: PWML, 8 chlorophyll gold rows matched,
2 verifier calls, 17 s, nothing omitted, folded or repaired. Chlorophyll is now VT 3/3 PWML vs BASE 2/3
(base1 aborted with flags off), no verifier-caused change in any run.

**Focused re-measure on the fixed code** (`C:/t/acc-ver5` at `4d29c4bf`; `scripts/driver5g.sh`, same
Gemma setup, guard and 5-min gaps; enterobactin + MenD x 3 draws x 2 arms, interleaved; scored with
`scripts/score5g.py`, which adds the lock-consistency check; `json/score5g.json`):

| paper | arm | PWML | gold matched (runs per id) | mean P | VT verifier |
|---|---|---|---|---|---|
| PMC12096016 enterobactin | base | 3/3 | R1 3, R2 2, R3 3, R4 3, R5 3 | 0.730 | |
| | vt | **3/3** | **R1-R5 all 3/3** | **0.869** | ≤2 calls, ≤10 s, 0 errors |
| PMC12312563 MenD | base | 3/3 | R1 3 | 0.328 | |
| | vt | 3/3 | R1 3 | 0.328 | ≤2 calls, ≤8 s, 0 errors |

R4 criteria, every VT run: report present, model `google/gemma-4-31b-it`; fail-open rate 0/3 on both
papers; no budget skip; lock-manifest removals equal omitted + folded lock ids, and no pruned lock id is
still in the payload (`pruned_but_present` empty); no AMBIGUOUS_RENAME_TARGET; no rename at all (F-G3);
0 typing changes; VT no-PWML 0 = BASE 0. Gold matched VT ≥ BASE on both papers (enterobactin 15 vs 14,
MenD 3 vs 3).

The scorer's "omitted/folded row matching gold" flag fired once: vt1-PMC12096016 folded row 5
(isochorismate → 2,3-dihydro-2,3-dihydroxybenzoate + pyruvate, R2) into row 1, which the verifier had just
completed with the same pyruvate co-product (addition, not a rename). The two rows were identical, and R2
is matched in that run's final output, so no gold is lost. vt3-PMC12312563 omitted the three partial MenD
rows through generic intermediates and appended the whole step; R1 matched.

**Spend:** last Gemma leg $0.11, re-measure $0.51 (`logs5g/*.usage.jsonl`). Running total for the
lane since the $10 cap: about $3.42.

### 14d. Review R5 (APPROVE): what changes in the record

**R1 F7c withdrawn for renames onto existing names.** The live repair R1 F7c was written for (the EntE
row whose product Stage 1 wrote as the wrong compound, renamed onto the existing
"2,3-dihydroxybenzoyl-AMP") is now refused by F-G3, and also by R4 B1 (the AMP token). That row stays as
Stage 1 wrote it; the report keeps the model's verdict and proposed rename. A rename still applies onto a
name no other entity carries. The test docstring says so (`a6f6e6e0`).

**Fold criterion, refined by R5.** A fold (a row folded into an identical or currency-compatible row)
counts as a gold loss only when the gold reaction it matched is then unmatched in that run's output. The
one flagged fold (vt1-PMC12096016, row 5 into row 1, both R2) left R2 matched, so it is not a loss.

**Conditions for enabling the verifier (`T2PW_REACTION_VERIFIER=1`), from R5:**
- Only with the measured model, Gemma 4 31B (`google/gemma-4-31b-it`). The verifier uses the Stage-1
  model (`OPENROUTER_EXTRACTION_MODEL`), so a server model change means re-running the 5g-style
  re-measure first.
- Only on code at or after `e72c2759`, with `T2PW_VERIFIER_GRANULARITY` unset (merge), and
  `T2PW_INTERACTION_TYPING` and `T2PW_VD18_SPAN_TEMPLATES` off.
- The server keeps its single worker (`web/retry.py` "the single worker"): the lock manifest lives in a
  shared `tmp/`, so the verifier's manifest pruning (R3 P8) depends on one job at a time.
- `reaction_verifier_report.json` is kept in every job folder, and the first server jobs are
  spot-checked for omitted rows.
- PO sign-off: the hard rule is met on this evidence, but n = 3 per arm per paper is small and Gemma Stage
  1 is noisy (BASE enterobactin R2 was 1/3 in 4g and 2/3 in 5g).

**Interaction typing: merged OFF, not to be enabled on Gemma.** Gemma proposed no re-typing in 24 live
VT legs (18 bigger run, 6 re-measure) and none offline, so the R4 typing zeros hold only because nothing changed. On Gemma it costs an
extra call per job and one typing call ran 335 s and returned invalid JSON. Revisit only with a model
that actually proposes re-typings, with a live arm where changes occur and are scored against
`ix_labels.json`.

**VD-18:** off, unchanged.

**Additions to the §14b ticket (outside the lane, for the PO):**
- **Flags-off chlorophyll abort, AMBIGUOUS_REFERENCE "tRNA Glu".** base1-PMC5130987 (BASE, every flag
  off), `C:/t/acc-runs/legs4g/base1-PMC5130987/2026-11-23_1538/papers/PMC5130987/strict/`: post_pipeline
  conversion stopped in `t2pw.pwml.prefreeze_resolution` (PrefreezeResolutionError) with
  "AMBIGUOUS_REFERENCE: the name 'tRNA Glu' is also carried by ['nucleic_acids#0']; a reference to it
  cannot be unambiguously redirected to 'tRNA(Glu)'" (`mapped_failure_snapshot.json`, `RESULT.txt`).
  Gemma: 1 of 6 chlorophyll legs (BASE); the 3 VT chlorophyll legs passed. No deepseek chlorophyll leg
  ran live in this lane. Pre-existing, not the verifier.
- **Check: do verifier records on flagged rows reach downstream LLM stages?** A flagged row (summary,
  split, not_stated, off_pathway) carries a `verifier` record (verdict, reason, model_quote) in the
  payload, and the audit loop and curator see the payload. vt1-PMC12096016 aborted at post_audit with
  only such flags set (rows 4 and 6). The BASE occurrence on the negative control makes a causal link
  unlikely, but the ticket should confirm whether those records enter any downstream prompt, and if so,
  whether stripping them before Stage 2 changes the outcome.
