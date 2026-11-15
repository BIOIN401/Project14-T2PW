# 05 Run-to-run variation: how big must a gain be to beat noise?

Read-only assessment. No `src/` change, no LLM call, no pipeline run. Scripts: `docs/pwml_output/evidence/05/scripts/`; tables:
`docs/pwml_output/evidence/05/` (`var_summary_by_paper.csv`, `var_agg.csv`, `mde_table.csv`, `qa_graph_hashseed_test.json`); full
scratch `C:/t/ev-oa/accvar/`. Runtime variance is already covered by `RUNTIME-VARIANCE.md`; not repeated. Labels: **M** measured by
me, **L** log/doc evidence, **G** guess.

## 0. NOT ESTABLISHED

1. **Pure same-code repeatability.** No set of repeated runs of one commit exists. Repeats are legs from 2026-08-18 to 2026-11-10 on
   changing code. I restricted to the largest same-month era per group (tables "sameEra") and the results hardly change (median
   count CV 0.21 both ways for reactions), but era and noise cannot be fully separated. Noise estimates are therefore **upper
   bounds on same-commit noise** (G).
2. **How many Stage-1 draws each repeated run pooled.** D-104 pools 3 draws; earlier legs may have used fewer. All sets below are
   post-pooling, so a single-draw run is noisier than shown (L: `PAPER_FINDINGS.md` s5.3 gives +-0.25 F1 for one draw).
3. **PMC7059786 "7 runs in batch".** Only 5 canonical `final_mapped.json` exist for that paper (4 PathWhiz: two 21-22 Oct batch legs,
   e2e4, e2e5; 1 research). The other batch legs left no payload I could open. All 5 have 0 reactions.
4. **Biological correctness of the unstable items.** An item seen in 1 of 10 runs is unstable, not wrong. Section 3 of assessment 04
   deals with correctness.
5. **Effect of the qa_graph ordering on outputs.** I show it changes the Stage-2 feedback text. Whether the LLM then answers
   differently needs LLM calls, which this task forbids (G).
6. **Between-paper effects.** 7 papers, all bacterial or human metabolic or signalling, small pathways (mean 1-5 reactions). The SDs
   below do not transfer to 100-reaction papers.

## 1. Corpus (M)

152 canonical runs, 51 papers, 19 papers with at least 2 runs. I analysed the 12 (paper, mode) groups with n >= 4, 7 papers:

| group | n | eras |
|---|---|---|
| PMC12096016 PathWhiz / research | 10 / 10 | Aug 6+7, Sep 2+2, Oct 2+1 |
| PMC12452463 PathWhiz / research | 10 / 9 | Aug 8+7, Sep 2+2 |
| PMC12782028 PathWhiz / research | 8 / 8 | Aug 6+6, Sep 2+2 |
| PMC12856317 PathWhiz / research | 8 / 8 | Aug 6+6, Sep 2+2 |
| PMC12180156 PathWhiz / research | 6 / 5 | Aug 4+4, Sep 2+1 |
| PMC9544450 PathWhiz | 5 | Sep 5 |
| PMC7059786 PathWhiz | 4 | Oct 2, web (9-10 Nov) 2 |

PathWhiz-mode and research-mode groups are never mixed: in research mode `final_mapped.json` is before the strict quarantine, in
PathWhiz mode after it (01 s1).

**Matching keys (all normalised: casefold, whitespace collapsed).** Entities: (bucket, name). Reactions: the unordered pair of
(sorted input names, sorted output names), enzymes ignored. Transports: (cargo, sorted transporter names). Interactions:
(unordered endpoint pair, relationship), `SAME_AS` excluded. "Id-collapsed" variant (C): a name is replaced by its mapped id
(UniProt, HMDB, PathBank compound id, ChEBI) when it has one, so synonyms of one molecule collapse. Jaccard = |A and B| / |A or B|,
mean over all run pairs. STABLE = item in >= 80% of the group's runs; UNSTABLE otherwise. Stages: S1 = `stage1_payload.json` (pooled
draws), F = `final_mapped.json`.

## 2. Numbers (M)

### 2.1 Median over the 11 same-era groups (final payload F; S1 in brackets)

| kind | groups | mean Jaccard between two runs | share of a run's items that are STABLE | share of union seen in exactly 1 run | median CV of count |
|---|---|---|---|---|---|
| compounds | 11 | 0.62 (0.72) | 0.72 (0.73) | 0.29 (0.25) | 0.15 (0.18) |
| proteins | 11 | 0.75 (0.77) | 0.79 (0.80) | 0.17 (0.13) | 0.21 (0.20) |
| complexes | 8 | 0.72 | 0.79 | 0.25 | 0.34 |
| **reactions** | 11 | **0.38 (0.50)** | **0.46 (0.53)** | **0.50 (0.34)** | **0.21 (0.07)** |
| transports | 5 | 0.33 (0.33) | 0.00 | 0.50 | 1.07 |
| **interactions** | 9 | **0.25 (0.43)** | **0.00 (0.58)** | **0.70 (0.50)** | **0.64 (0.41)** |

Id-collapsing barely moves anything (reactions 0.38 to 0.40, compounds 0.62 to 0.68, proteins 0.75 to 0.75), so the instability is
not mainly synonym spelling. All-era versions are in `var_agg.csv` and differ by at most 0.09 (reactions 0.46 vs 0.38, interactions 0.17 vs 0.25).

Three readings (M for the numbers, G for the cause):
- **Same count, different content.** PMC12452463 research gives exactly 4 reactions in all 9 runs (SD 0) yet the 9 runs contain 14
  distinct reactions (Jaccard 0.28; 1 reaction in all runs). PMC12096016 PathWhiz: 5-7 reactions per run, 17 distinct in total,
  2 stable. A count-only metric would call these runs identical.
- **Downstream stages add instability to reactions.** Reaction count CV is 0.07 at Stage 1 and 0.21 at the final payload; Jaccard
  falls 0.50 to 0.38. Interactions fall 0.43 to 0.25. Stage 2, audit and curator are LLM stages (G), and quarantine removes
  different rows in different runs (01 s2.5).
- **Entities are the stable layer, processes are not.** Entity Jaccard 0.6-0.75 and 72-79% stable; reactions 0.4 and 46%;
  transports and interactions have no stable item at the 80% rule in the median group.

### 2.2 Per (paper, mode), final payload, all eras (M)

| paper, mode | n | reactions: mean, SD, range | reaction J (name / id) | reactions stable / distinct | compounds+proteins: mean, SD | entity J (ids) |
|---|---|---|---|---|---|---|
| PMC12096016 PathWhiz | 10 | 5.6, 0.84, 5-7 | 0.46 / 0.38 | 2 / 17 | 18.4, 2.41 | 0.74 |
| PMC12096016 research | 10 | 5.0, 1.15, 3-6 | 0.39 / 0.33 | 1 / 18 | 18.8, 4.02 | 0.67 |
| PMC12452463 PathWhiz | 10 | 3.4, 0.97, 2-5 | 0.26 / 0.26 | 1 / 17 | 11.0, 2.31 | 0.61 |
| PMC12452463 research | 9 | 4.0, 0.00, 4-4 | 0.28 / 0.41 | 1 / 11 | 12.7, 1.94 | 0.72 |
| PMC12782028 PathWhiz | 8 | 3.25, 0.46, 3-4 | 0.76 / 0.76 | 3 / 6 | 8.0, 0.93 | 0.85 |
| PMC12782028 research | 8 | 3.5, 0.53, 3-4 | 0.79 / 0.79 | 3 / 6 | 19.8, 7.07 | 0.73 |
| PMC12856317 PathWhiz | 8 | 1.25, 0.46, 1-2 | 0.76 / 0.76 | 1 / 3 | 6.4, 0.52 | 0.67 |
| PMC12856317 research | 8 | 1.12, 0.35, 1-2 | 0.88 / 0.88 | 1 / 2 | 6.8, 1.16 | 0.75 |
| PMC12180156 PathWhiz | 6 | 1.33, 0.52, 1-2 | 0.16 / 0.06 | 0 / 6 | 5.8, 1.47 | 0.55 |
| PMC12180156 research | 5 | 1.4, 0.89, 0-2 | 0.07 / 0.03 | 0 / 6 | 6.8, 0.84 | 0.62 |
| PMC9544450 PathWhiz | 5 | 4.6, 0.89, 4-6 | 0.74 / 0.74 | 4 / 6 | 10.0, 0.71 | 0.89 |
| PMC7059786 PathWhiz | 4 | 0 in every run | transports 0.36, interactions 0.17 | 0 stable transports (5 distinct) | 4.2, 0.50 | 0.44 |

The papers split into **repeatable** (PMC12782028, PMC12856317, PMC9544450: reaction Jaccard 0.74-0.88; reactions are a closed
small set) and **unrepeatable** (PMC12452463, PMC12180156, PMC12096016: 0.07-0.46; the paper states few clean reactions or many
variants, 01 s2.3). The PMC12782028 research proteins range 4-20 (SD 6.8): one run read the gene list as proteins and others did
not. PMC7059786: Stage 1 gives 7-11 proteins in every run, the final payload 1-7 (PathWhiz mode 1-3); the quarantine, not the
reader, causes the variation. Both web runs keep exactly the H+/OTOP1 transport and a serotonin release and nothing else
stable, and the five runs share no complete transport set.

### 2.3 Recall noise on gold (M, 79 gold-paper canonical runs, 7 papers, from `gold_recall.jsonl`)

Per-run gold reaction recall, pooled within-(paper, mode) SD: **0.133 at Stage 1, 0.113 at the final payload** (df 66). Per paper:
PMC12096016 0.92 +- 0.14 (n=10, range 0.6-1.0); PMC12452463 0.70 +- 0.11 (n=10, 0.67-1.0); PMC12782028 PathWhiz 0.92 +- 0.24 (n=8,
0.33-1.0); PMC12856317 (both modes) and PMC12782028 research sit at 1.0 with SD 0 (no headroom). This agrees with the 88-draw ORCH-747 result, pooled SD 0.135 and
+-0.25 F1 on a single draw (L: `PAPER_FINDINGS.md` s5.3). Union of runs reaches 23/23 gold reactions at Stage 1 (L: 01 s2.3), so
a missed reaction is usually random, not systematic, except enterobactin's one miss (ORCH-751).

## 3. Known instability: Stage 1 and the two qa_graph sites

**Stage 1 (L).** Identical prompts at temperature 0 give different draws (`PAPER_FINDINGS.md` s5.3, matched reactions per draw
9, 7, 4 on a 9-reaction gold). D-104 pools three draws to cut this. Pooling reduces but does not remove it (s2 above is post-pooling).

**`qa_graph.py:236` and `:554` (read and tested, M).**
- `:236` is `def connected_components(adj)`. It walks `adj.keys()` (dict insertion order) and, inside the BFS, `for v in adj[u]`
  where `adj[u]` is a `set` (`:22 add_edge`). The BFS order is arbitrary, but the **membership of every component is the same**
  whatever the order, so component sizes and the main-component size are deterministic. What varies is the *order of the list
  `comps`*, which follows the insertion order of `adj`.
- That insertion order is not fixed because `get_entities` (`:33-64`) returns `set`s of strings (`:36-44`) and callers add isolated
  entities by iterating them: `pipeline.py:618-627` in `build_qa_feedback`, `qa_graph.py:543-552` in the CLI. String hashing is
  randomised per process and nothing in the repo pins `PYTHONHASHSEED`, so the order differs between runs.
- `:554` is `comps = connected_components(adj)` inside `main()` (`:528`), the **command-line** QA tool
  (`python qa_graph.py extracted.json`). It is not called by the pipeline. `generate_qa_report` (`:262`) does not use
  `connected_components`.
- The product consumer is `pipeline.py:630` (`build_qa_feedback`): `sorted(comps, key=len, reverse=True)` is stable, so equal-size
  components keep the unstable order. `comps_sorted[1:]` become `orphan_components` and are cut with `[: max(1, hint_limit // 5)]`
  (`:673`); `missing_links[:hint_limit]` is also in set order. The result is fed back into **Stage 2 rounds 2 and later**
  (`pipeline.py:716`) and shown in the UI (`streamlit_app.py:3001`). `process_normalizer.py:5047` uses only `len`, so it is
  deterministic.
- **Test (M).** I called the real `build_qa_feedback` on 15 distinct `stage1_payload.json` files (papers PMC12096016, PMC9544450,
  PMC12452463, PMC12782028, PMC7059786) under 6 different `PYTHONHASHSEED` values: **8 of 15 payloads produced different output**
  (2 to 6 distinct outputs); the differing fields were always `orphan_components` and `missing_links_suspected` (order, and which
  orphan components survive the cut of 5). Table: `qa_graph_hashseed_test.json`. For 6 of the 8 payloads the orphan count (1-5) and missing-link
  count (2-13) are under the caps (5 and 25), so only the **order** of hints in the Stage-2 prompt changed. In the 2 payloads
  with 12 and 14 orphan components (cut to 5; 6 distinct outputs each) the surviving orphans can differ in *content* as well
  (not checked item by item). Whether a reordered hint list changes the LLM answer is not established (G). No other site was tested.

This is a **small contributor to noise** (it touches Stage-2 rounds >= 2 only, and only prompt order) but it is the one
deterministic source we can remove at no biological risk: iterate `sorted(entities[...])` in `build_qa_feedback` and sort ties
(`key=lambda c: (-len(c), min(c))`). Needs a D-entry only because it changes Stage-2 prompts byte-wise.

## 4. How large must a gain be?

Method (G for the assumptions, M for the inputs). Compare arm A (old) and arm B (new) on the **same K papers**, n runs per arm per
paper, with the paper effect cancelling. With pooled within-paper SD s_w, the standard error of the difference in arm means is
`SE = s_w * sqrt(2 / (n K))`; minimum detectable effect at 80% power, 5% two-sided is `MDE = 2.8 * SE`. Total runs per arm n K =
`2 (2.8 s_w / delta)^2` for a target delta. It ignores paper-by-arm interaction and between-run correlation in the same era, so it
is optimistic (G). Inputs, pooled within (paper, mode, era) groups with n >= 4 (M, `power.py`): entity count (compounds + proteins +
complexes) s_w 2.82 on mean 13.3 (CV 0.21); reaction count 0.76 on 3.16 (0.24); all process count 1.73 on 5.55 (0.31); interaction
count 1.67 on 2.39 (0.70); transport count 0.83 on 0.96 (0.87); gold recall per run 0.12 (absolute).

| design (K papers x n runs per arm) | total runs per arm | MDE gold recall (points) | MDE entity count | MDE reaction count | MDE all processes | MDE interactions |
|---|---|---|---|---|---|---|
| 1 x 3 | 3 | 27 | 48% | 55% | 71% | 160% |
| 7 x 3 | 21 | 10.4 | 18% | 21% | 27% | 60% |
| 10 x 3 | 30 | 8.7 | 15% | 17% | 23% | 51% |
| 7 x 5 | 35 | 8.0 | 14% | 16% | 21% | 47% |
| 10 x 5 | 50 | 6.7 | 12% | 14% | 17% | 39% |
| 20 x 3 | 60 | 6.1 | 11% | 12% | 16% | 36% |
| 20 x 5 | 100 | 4.7 | 8% | 10% | 13% | 28% |

(Full grid: `mde_table.csv`.) Read-offs, each from the formula:
- **+10 points of reaction recall** needs about 23 runs per arm (for example 8 papers x 3). **+5 points** needs 91 (for example 18
  papers x 5).
- **+20% entities** needs about 18 runs per arm; **+10% entities** needs about 71 (for example 14 x 5).
- **+10% reactions** (count) needs about 91 runs per arm; +20% needs 23.
- **Transports and interactions** have CV 0.7-0.9: even 20 x 5 only detects a +28% (interactions) change in count. A gain of +10%
  to +20% in these classes **cannot be told from noise** with any plausible budget. Use set-level comparison instead (below).
- One run per arm on one paper (the current practice on E2E papers) detects only a change of more than 80% in entity count or 47
  points of recall. Any smaller claim from one pair of runs is noise.
- Ceiling: PMC12856317 (both modes) and PMC12782028 research already score recall 1.0 with SD 0. They contribute no detectable gain and shrink the
  headroom; recall gains must be judged on papers with headroom (PMC12452463 0.70, PMC12096016 0.92, PMC12782028 PathWhiz 0.92).

## 5. WHAT TO DO (protocol recommendation for the plan)

1. **Separate three kinds of change and test each differently.**
   - *Deterministic downstream changes* (exporters, writer, quarantine rules, mapping rules, naming): do not use fresh LLM runs.
     Replay the **frozen** `stage1_payload.json` / `merged_payload.json` of existing runs through old and new code. The noise
     from s2 vanishes; the comparison is exact per payload, n=1 per payload is enough, and 100+ payloads are available (152 runs).
     Pin `PYTHONHASHSEED` in the harness (s3) so the replay itself is deterministic. Caveat: mapping reads the mutable caches
     (L: READINESS gap 6), so pin the cache blobs too.
   - *Changes that alter LLM input* (prompts, Stage-2 hints, RAG): use the K x n design below. Single draws are uninterpretable
     (L: s5.3 of PAPER_FINDINGS).
   - *Changes that add rows* (recovered entities, new row classes): evaluate on **precision first** (assessment 04), because added
     rows raise raw counts by construction; a count gain with no precision measurement is not a gain.
2. **Standard A/B protocol for LLM-affecting changes.** At least 10 papers with certified gold and headroom, 5 runs per arm,
   interleaved in time, same code commit within an arm, one model/provider setting, 3 pooled draws per run, the same pooled
   settings in both arms. Primary endpoint: gold reaction recall and precision (micro over runs), MDE about 7 points. Report with
   per-paper paired differences and a bootstrap over papers (not over runs). 50 runs per arm, 100 in total. At 21-74 minutes
   per run for a mid-sized paper (L: RUNTIME-VARIANCE s3) the batch is days of wall time; plan it as an overnight job on the
   existing runner, not an interactive loop.
3. **A cheaper screening tier.** 7 papers x 3 runs per arm (42 runs) detects only a >= 10-point recall change or a >= 18% entity
   change. Use it to reject clear regressions and to find big effects, never to claim a small gain.
4. **Judge processes by stable core, not by count.** For transports, interactions and reactions on unrepeatable papers, compare
   (a) the *stable core* (items in >= 80% of an arm's runs) and (b) per-item inclusion frequencies over the union, with a paired
   permutation test over items or a McNemar test per item pair. Counts alone would call two different reaction sets equal (s2.1).
5. **Report effect sizes as a multiple of the noise.** Always print the arm SD, the number of runs and the MDE next to every
   delta (use `mde_table.csv`). A gain below the MDE is "not distinguishable from noise", not "no gain".
6. **Remove the cheap deterministic noise.** Fix the set iteration in `build_qa_feedback` (s3) and log `PYTHONHASHSEED`/commit in
   each run record. Expected gain bound: removes prompt-order variation in Stage-2 rounds >= 2; effect size unknown (G), no
   biology change; needs a D-entry because Stage-2 prompt bytes change.
7. **Do not mix eras in one comparison.** Of the 152 canonical runs, 150 predate D-117 to D-120 (01 s0.2, 02 s0.1). A baseline
   for the paper must be re-run on the target commit; old runs are only good for variance estimates, as used here.

Questions for the owner: (1) May the evaluation budget be about 100 runs per LLM-affecting change (10 papers x 5 x 2 arms)? If not,
state the smallest gain worth claiming; below about +7 points of recall or +12% of entities it cannot be shown. (2) Three more
certified-complete gold papers with headroom would cut the MDE more than extra runs on the current three (also asked in 04 A3).
