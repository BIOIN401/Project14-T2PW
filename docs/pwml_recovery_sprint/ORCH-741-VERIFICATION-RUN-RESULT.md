# `ORCH-741` — ten-paper production verification run. **7 / 10 PWML.**

**2026-09-16 21:53 → 2026-09-17 01:52, 3 h 59 m.** Ten fresh papers, one strict leg each, no
favourable reruns, on the production tree frozen at `5f6a7aa3` (`D-102`). Zero timeouts, zero
surviving processes, heavy lock `ORCH-741` acquired and released. **No production code was
changed, proposed or dispatched.** `git diff 5f6a7aa3 HEAD -- src/` is empty and `main` is
untouched.

Manifest frozen and committed **before** execution at `cd73cdbf`:
[`topics_orch741_verify.txt`](../../topics_orch741_verify.txt). Run tree:
`runs_verify/2026-09-16_2152`. G11 reports `ORCH-741/01`–`08`.

---

# 1. THE NUMBER

| | |
|---|---:|
| papers attempted | **10** |
| **meaningful cores** (a leg that reached a reaction graph) | **9** |
| **PWML files on disk** | **7** |
| **overall yield** | **7 / 10 = 70 %** |
| serialization yield (cores that became files) | 7 / 9 = **78 %** |
| structurally **IMPORT READY** | **6 / 7** |
| `F-195` dangling reference | **1** |
| timeouts | 0 |

**Three misses, three DIFFERENT mechanisms, each `n = 1`. No mechanism repeats.**

## 1.1 What this number is not

**It must not be compared like-for-like against `C-122`'s 4 / 12.** Two different cohorts, two
denominators, never merged. Three reasons to discount the jump, all of them stated in the manifest
**before** the run, not invented after seeing 70 %:

1. **The organism mix was chosen easier and was disclosed as easier.** Bacterial 3 / plant 3 /
   fungal 2 / mammalian 2, against `C-122`'s 4 / 4 / 3 / 1. `ORCH-725` measured resolution at
   human 67 %, plants 53 %, bacteria 39 %, fungi 20 %. Lighter on fungi and heavier on mammals is
   a tailwind, and I took it knowingly.
2. **Ten papers is not a population rate, and the difference is not statistically significant.**
   Computed, not asserted (Clopper-Pearson, exact binomial):

   | cohort | yield | 95 % CI |
   |---|---:|---|
   | `ORCH-741` | 7 / 10 | **34.8 % – 93.3 %** |
   | `C-122` | 4 / 12 | **9.9 % – 65.1 %** |

   The intervals **overlap across 34.8 %–65.1 %**, and **Fisher's exact test on 7/10 vs 4/12 gives
   `p = 0.198`** — nowhere near significance. One correction to my own first draft of this
   section: I wrote that the `ORCH-741` interval *contains* `C-122`'s 33 %, and **it does not** —
   its lower bound is 34.8 %, just above. The overlap and the `p`-value are the real arguments and
   they are enough. **On these two samples I cannot establish that the system improved at all**,
   and I am not going to claim it did.
3. **Selection favoured papers that state explicit reaction chemistry.** That is the standing
   selection rule rather than a thumb on the scale, but it is a real property of this cohort:
   every pick names enzymes, substrates and products in its own abstract.

**Nothing merged between `C-122` and this run except `C-124`, `C-125` and `C-126`,** none of which
touch extraction or mapping breadth. So the honest reading is that this cohort was easier, not
that the system got better.

---

# 2. Per-paper result

`rxn` = reactions in the graph, `pwml` = reactions that reached the file.

| PMCID | organism — pathway | rxn | pwml | bytes | outcome |
|---|---|---:|---:|---:|---|
| `PMC13364408` | *A. graveolens* — furanocoumarin | 8 | 8 | 90,583 | **IMPORT READY** |
| `PMC12059215` | *M. musculus* — testosterone | 7 | 7 | 70,630 | **IMPORT READY** |
| `PMC13170813` | *H. vulgare* — hordatine | 5 | 5 | 56,634 | **IMPORT READY** |
| `PMC13486683` | *P. hybrida* — benzyl alcohol | 5 | 5 | 96,351 | `F-195` — import last |
| `PMC12369002` | *C. jejuni* — CDP-d-fucofuranose | 7 | 4 | 52,920 | **IMPORT READY** |
| `PMC12505894` | *C. albicans* — farnesol | 4 | 4 | 49,471 | **IMPORT READY** |
| `PMC12983300` | *P. polonicum* — verrucofortine | 3 | 2 | 27,126 | **IMPORT READY** |
| `PMC13418202` | *S. angustmyceticus* — benzohydroxamic acid | 2 | — | — | **crash**, § 4.1 |
| `PMC13534948` | *H. chejuensis* — prodiginine | 2 | — | — | **contract**, § 4.2 |
| `PMC12860957` | *M. musculus* — keratan sulfate | 0 | — | — | **Stage 1**, § 4.3 |

`PMC13364408` at 8 reactions and 34 edges is the **largest PWML the sprint has produced**, ahead of
`ORCH732_PMC12051542` (8 reactions, 80,518 B). `PMC13486683` carries **36 edges**, the most of any
file on disk.

## 2.1 The reaction drops are the quarantine working, not silent loss

Two files serialized fewer reactions than their graph held, and in both cases
`removed_entity_report.json` names every removal with a reason:

* `PMC12369002` 7 → 4. Removals are `unreferenced_after_quarantine` on compounds
  (`CDP-d-fucofuranose`, `CDP-6-deoxy-l-altrofuranose`, `NAD+`, …) — orphaned *after* their
  reactions were quarantined, not dropped on their own.
* `PMC12983300` 3 → 2. `L-Trp` and `L-Leu` unreferenced after quarantine; protein `VftPS`
  removed as `degree_zero_after_quarantine` with `had_external_identity: false`.

This is the documented strict pre-export policy preserving a correct subgraph rather than shipping
an unsupported one. **It is not a defect and no card is proposed for it.**

---

# 3. Coordinates and structure — every file placed and routed

Checked by `evidence/orch734_pathwhiz_import_check.py`, G11 report `ORCH-741/08`.

| PMCID | canvas | positioned | edges | edges with a path | species (taxon) |
|---|---|---:|---:|---:|---|
| `PMC12369002` | 3200 × 1460 | 18 | 18 | **18** | *Campylobacter jejuni* (197) |
| `PMC13364408` | 3200 × 3092 | 32 | 34 | **34** | *Apium graveolens* (4045) |
| `PMC13170813` | 3200 × 1460 | 19 | 19 | **19** | *Hordeum vulgare* (4513) |
| `PMC13486683` | 3200 × 3155 | 40 | 36 | **36** | *Petunia hybrida* (4102) |
| `PMC12983300` | 3200 × 1400 | 8 | 8 | **8** | *Penicillium polonicum* (60169) |
| `PMC12505894` | 3200 × 1986 | 19 | 15 | **15** | *Candida albicans* (5476) |
| `PMC12059215` | 3200 × 3330 | 26 | 26 | **26** | *Mus musculus* (10090) |

**`edges_with_path` equals `edges` on all seven** — every connector is routed, none is a bare
reference. Every primary species carries the correct NCBI taxonomy id, verified by hand against
each paper's own organism. Canvas heights scale with graph size (1,400 for the 2-reaction file,
3,330 for the 7-reaction one) rather than being a fixed frame.

**This is a structural check, not a visual one.** It establishes that coordinates exist, are
non-zero and are internally consistent. **Whether the layout is legible — overlap, crossing
connectors, label collisions — is the product owner's call in the PathWhiz UI and is NOT claimed
here.**

---

# 4. The three misses, classified

Per `CLAUDE.md`: a run failure does not by itself justify a code change. Each is classified before
any card could be proposed.

## 4.1 `PMC13418202` — `PREFREEZE_CONNECTIVITY_BROKEN`. **NEW: `F-204`.**

```
Post-pipeline conversion failed: PREFREEZE_CONNECTIVITY_BROKEN:
renaming changed the participant edge set; the payload was not modified
```

The leg reached a meaningful core — 2 reactions, 3 proteins, 6 compounds, scope and organism both
read correctly — then died at `post_pipeline`. The guard
(`src/t2pw/pwml/prefreeze_resolution.py:471`) compares a connectivity signature before and after a
staged compound rename and refuses to commit when they differ. **The guard itself is behaving
correctly and fail-closed; a rename that changes the participant edge set must not be committed.**

**Classification: `policy_disagreement` on the outcome, plus a genuine observability defect.**

* The *policy* question — whether a payload that fails this guard should be lost entirely or
  preserved as `review_required` with the rename declined — is for the product owner. `CLAUDE.md`
  merge rule 7 requires patches to preserve incomplete-but-correct pathways rather than dropping
  them; whether that principle reaches a pre-freeze rename conflict is **not mine to decide**.
* The *observability* half is not a judgement call. The exception carries `first_divergence` and
  `rename_map`, and **neither reaches the manifest row, the RESULT.txt, or any report file.** The
  row records `issue_codes: []`. The most informative part of the failure is discarded at the
  moment it is raised. **REGISTERED as `F-204`, NOT CHARTERED.**

**Not diagnosed further here.** Recovering the actual divergence needs a read-only replay of
`prefreeze_resolution` against the committed `merged_payload.json`, which I deliberately did not
run during the cohort and have not run since: a replay in the wrong mode is exactly what
fabricated a false finding once before in this sprint.

## 4.2 `PMC13534948` — `final_gate_report_missing`. **NEW: `F-205`.**

```
post-pipeline validation failed: 1 blocking issue(s) at
final_pre_export_gate_lifecycle [final_gate_report_missing]
issue_codes: ['gate.gate_lifecycle']
```

Also a meaningful core (2 reactions, 11 entities), also correct scope and organism. The leg wrote
`initial_stage3_gate_report.json` and `gate_fail_report.json` but **no `final_stage3_gate_report.json`**,
and the lifecycle check then refused it for the absence.

**Classification: candidate `product_contract_violation`, NOT established.** This sits in the exact
area `C-119` and `C-126` worked — the gate-report lifecycle and which phase is authoritative — and
the failure is the *absence* of the final report rather than a bad verdict in it. That makes it
adjacent to, but visibly not the same as, the `F-147` supersession seam `C-126` closed.
**REGISTERED as `F-205`, NOT CHARTERED.** It needs the bio-auditor and a look at whether the
initial gate's failure legitimately prevents the final gate from ever running.

## 4.3 `PMC12860957` — Stage 1 empty completion. **Known `F-199` class. No card.**

```
Extraction failed: Chunk 2 failed to produce valid JSON.
```

Died at `stage1` with zero reactions; never reached a biological gate. This is the delivery-failure
class `ORCH-739` measured and `ORCH-740` failed to fix with a budget change: the primary model
spends its completion budget on reasoning tokens and returns nothing. **Stochastic. Per the frozen
manifest it was NOT re-run**, and a leg that never arrives proves nothing either way.

`C-122` lost `PMC13398565` the same way. **This is the one mechanism that has now appeared in two
consecutive cohorts**, and it remains the standing argument for the `F-199` runtime card that
`ORCH-739` § 16 C left with the product owner.

---

# 5. Pre-existing findings, re-observed

## 5.1 `F-195` — **fourth instance**, on `PMC13486683`

Two dangling `compound-location-id` references (66, 82) out of 192 internal refs. **The signature is
identical in shape to the known instance** `ORCH730_PMC7232280.F195` (dangling
`compound-location-id` 30, 34), verified by running the same checker over both. Population is now 4
papers. **Not fixed, per the standing instruction; the live import decides.** The file is committed
as `ORCH741_PMC13486683.F195.pwml` and should be imported **last**.

## 5.2 `F-202` — present in 5 of 7, still in its **sentinel-only** form

*Arabidopsis thaliana* appears as a second species element in `PMC12369002`, `PMC13364408`,
`PMC13170813`, `PMC13486683` and `PMC12983300` — **exactly once each, referenced by no protein and
no complex.** `PMC12505894` and `PMC12059215` are clean. This matches the `C-122` § 5 adjudication
that `F-202` is a `PathBank Unknown` placeholder artefact and **not** a correctness defect, and it
is reported here as a re-observation rather than a new finding. **No resolved protein in any of the
seven is mislabelled.**

## 5.3 `release_ready` — **zero**

All seven are `review_required`. That is a success under the frozen criterion and is **not** a
regression: `C-122`'s single `release_ready` (`PMC12914822`) remains the only one of the sprint.

---

# 6. G11

Eight bounded jobs: `01`–`04` cohort searches, `05` a failed first fetch-verify kept on the record,
`07` the real fetch-verify, `06` the ten-leg run, `08` the import check. **Every one
`FINAL SURVIVING COUNT : 0` / `cleanup : success`.** The run job observed and terminated **23
descendants** and left none. Heavy lock free before and after. Four pre-existing processes (VS Code
language servers) reported on every job and **never killed**.

Integration smoke suite re-run on the frozen tree before the cohort: **508 passed in 61 s**, zero
survivors.

---

# 7. What I am NOT claiming

* **Not that the reliability phase is complete.** § 23 permits closure at ~80 % **or** when the
  remaining misses are isolated and stochastic rather than one repeated deterministic defect. 70 %
  is not 80 %. The misses **are** isolated — three mechanisms, `n = 1` each, where `C-122` had
  three mechanisms repeating twice each — but **two of the three are deterministic, not
  stochastic**, so the second condition is only half met. **That judgement is the product owner's
  and I am not pre-empting it.**
* **Not that the system improved.** § 1.1. The cohort was disclosed as easier before the run and
  the confidence interval swallows the difference.
* **Not that any file imports.** `IMPORT READY` means every property the checker tests is
  satisfied. The live PathWhiz UI import needs the product owner's account.
* **Not that the layouts are good.** § 3. Coordinates exist and are consistent; legibility is
  unexamined.

# 8. Next, for the product owner

1. **Import the six clean files by hand** and check the coordinates in the UI. They are hash-pinned
   under `pathwhiz_review/IMPORT-SET/` as `ORCH741_*`. Start with **`ORCH741_PMC13364408`** — the
   largest graph the sprint has built, 8 reactions and 34 edges, so it is where a layout problem
   would show first. Import **`ORCH741_PMC13486683.F195` last**; it is the `F-195` specimen.
2. **`F-204`** — the observability half is cheap and self-contained: persist `first_divergence` and
   `rename_map` onto the failure row. The preservation half is a policy call.
3. **`F-205`** — needs adjudication before anything else.
4. **`F-199`** still stands, now with an instance in two consecutive cohorts.
