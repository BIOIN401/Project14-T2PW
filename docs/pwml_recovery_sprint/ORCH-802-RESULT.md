# ORCH-802 RESULT: word-boundary actor match in `enzyme_cues.cue_near_name`

**Status: STOPPED at the no-loss classification.** This is not a merge candidate yet.

- The fix and its G9 proof are done: base fails, tip passes.
- The no-loss scan found four injected actor names that are lost and that name a real catalyst: `ALAS`, `HepPPS`, `MenA` (on EcMenA rows), and `EntE` on two rows that name it as a whole word (`EntD/EntE/EntF complex`), which are refused only because the cue is more than 80 characters from that mention. Every one of these rows keeps another catalyst: the paper's own name for the same enzyme, or the enterobactin synthase complex.
- Reviewer F3's merged-payload replay (3.4) found **0 additions** and 29 drops, all in classes already listed here.
- The task brief says that a dropped injection that looks like a real catalyst, plural and digit-suffix losses included, means stop and report.
- The corpus replay (step 4) was skipped on the orchestrator's instruction (see 3.2 and 4). SMOKE (step 5) was run afterwards: 511 passed.

| | |
|---|---|
| Branch | `orch802/enzyme-cue-boundary`, base `d2e68b87` |
| Code commit | `2718851e` |
| Files | `src/t2pw/pipeline/enzyme_cues.py` (+7/-1), `tests/test_orch802_enzyme_cue_boundary.py` (+140, new) |

## 1. Defect

`enzyme_cues.cue_near_name` (line 81 on base) ran
`re.finditer(re.escape(actor_name), evidence, flags=re.IGNORECASE)`.

That is a bare substring match. In ORCH-801, leg cdadc1 (PMC12184417), reaction R1 is `dCTP -> dUTP` and its sole catalyst is CDADC1. The scan found "DUT" inside "dUTP", with "catalyz" in the window. `pipeline._inject_name_based_modifiers` then injected DUT as a second catalyst. The bio-auditor classified this as a `product_contract_violation`; see `runs_papers/ORCH-801/ACCURACY_AUDIT.md` F1 in the primary checkout, which is untracked.

Two attachers route through this one function:
- `pipeline._inject_name_based_modifiers` (pipeline.py:2990)
- `process_normalizer.attach_enzymes_from_reaction_evidence` (process_normalizer.py:3731, through the `_cue_near_name` alias)

## 2. Fix

- The pattern is now `(?<!\w)` + `re.escape(name)` + `(?!\w)`, still with IGNORECASE.
- `\b` is not used, because it fails after a name that ends in `+`.
- Short symbols are not made case-sensitive; that option was out of scope.

### Other window users checked

**`pipeline._transporter_cue_near_name` (pipeline.py:3074): unchanged.**
- It already has a boundary rule, added by F-058 / C-058.
- That rule asserts `(?<![0-9A-Za-z])` / `(?![0-9A-Za-z])`, and only on a side whose edge character is alphanumeric.
- It does not have the substring defect.
- It differs from the new enzyme rule in two ways:
  - It treats ASCII letters and digits only as word characters. `\w` also counts `_` and Unicode letters.
  - After a trailing `+` it applies no assertion. So "Ca2+" matches inside "Ca2+ATPase" there, but would not under the enzyme rule.
- Unifying the two rules is out of scope.

**`apply_audit_patch.py:2172`: not affected.** It uses only the window width, `_ACTOR_CUE_WINDOW = 80`, and does its own anchored token matching.

## 3. Proofs

### 3.1 G9 focused test: base fails, tip passes

The test file is `tests/test_orch802_enzyme_cue_boundary.py` and has 14 tests.

**Correction proofs** (expected to fail on base):
- `test_dut_does_not_match_inside_dutp` uses the real haystack quoted in audit F1.
- `test_dut_not_injected_by_stage2_injector`
- `test_dut_not_attached_by_stage5_normalizer`
- `test_short_symbol_not_matched_inside_longer_gene_name` (CYP1 against CYP11A1)

**Positive controls** (expected to pass on both). These are preservation guards, not corrections:
- a real DUT mention, at the predicate and through the injector
- the name at the start of the text, and at the end
- punctuation after the name: `,`, `-catalyzed`, `)`
- `NAD+`
- `Na+/K+-ATPase`
- a multi-word complex name
- `UDP-glucose 4-epimerase` and `CYP11A1`
- a case-insensitive hit

**Base method.** The test inserts `<repo>/src` into `sys.path`, and `pytest.ini` sets `pythonpath = src`. So setting `PYTHONPATH` to a base export would still import the tip code. Instead:
1. `git archive d2e68b87 src tests/conftest.py pytest.ini` was extracted to `C:/t/orch802/base`.
2. The test file was copied into it.
3. The run used `--cwd C:/t/orch802/base`.

The orchestrator accepted this method.

| run | result | G11 report |
|---|---|---|
| base | `4 failed, 10 passed in 1.29s`. The four failures are exactly the correction proofs, e.g. `assert ['CDADC1', 'DUT'] == ['CDADC1']` | `evidence/g11/ORCH-802/01b-base-focused-output.json`. The first base run, `01-base-focused.json`, has the same exit code 1, but its stdout was cut to `tail` and the counts were lost, so it was re-run as 01b |
| tip | `14 passed in 1.46s` | `evidence/g11/ORCH-802/02-tip-focused.json` |

Both runs show `FINAL SURVIVING COUNT : 0`, cleanup success, and the heavy lock acquired and released.

### 3.2 Does the corpus replay exercise the cue matcher? No, by reading the code

`orch768_replay_corpus.py` starts at `final_mapped.json`, which comes after both attachers. The functions it replays are:
- `run_enrichment`
- prefreeze
- `freeze_canonical_payload`
- `initialize_refinement_review_state`
- `run_pwml_export`, which calls `build_pwml_ir` directly

None of these reaches `normalize_process_payload`. That is the only caller of `attach_enzymes_from_reaction_evidence`, and its own callers are:
- `streamlit_app.build_rag_round_stages`
- `streamlit_app.run_post_pipeline_sbml_artifacts`
- the writer CLI, `run_pwml_pipeline_export`

The Stage-2 injector runs at pipeline.py:1260, which is also upstream of the checkpoint.

So a corpus replay at tip can only show that the downstream path is untouched. It cannot measure this change. **It has not been run** (see Status).

### 3.3 No-loss scan

Files:
- script: `evidence/orch802/orch802_cue_noloss_scan.py`
- output: `evidence/orch802/noloss_scan.json`
- stdout: `evidence/orch802/noloss_scan_stdout.txt`

G11 reports:
- `03-noloss-scan.json`: this run crashed while printing, with `UnicodeEncodeError`; a snippet contains U+2032 and the console is cp1252. The JSON output had already been written. The script now reconfigures stdout to UTF-8.
- `03b-noloss-scan.json`: the clean re-run, exit 0, 0 survivors.

What was scanned:
- 489 committed `stage1_payload.json` / `merged_payload.json` / `final_mapped.json` files.
- 9 untracked ORCH-801 files, read only: the cdadc1, tb and `salicylic_acid-strict/2026-11-04_1534` legs.

Two measurements:
- **A. Predicate differential.** Old versus new `cue_near_name`, each loaded by file path from the base and tip trees, over every pair of a reaction haystack (evidence of 400 characters or fewer) and a declared actor in that payload.
  ```
  {"A_predicate": {"distinct_flips": 34, "new_hit_without_old_hit": 0, "new_hits": 1494, "old_hits": 1583, "pairs_evaluated": 20739}, "files_scanned": {"committed": 489, "extra_untracked": 9}, ...}
  ```
- **B. Production replay.** `_inject_name_based_modifiers`, then `attach_enzymes_from_reaction_evidence`, on every `stage1_payload.json`, with the old and then the new matcher patched into both modules.
  ```
  {"stage1_legs": 190, "reaction_rows_differing": 24}
  ```
  - All 24 differences are drops. **None is an addition.**
  - 23 drop `EntE` on an enterobactin-assembly row of PMC12096016 or PMC12452463.
  - 1 drops `DUT` on ORCH-801 cdadc1 R0. This replays the F1 defect at base and shows it gone at tip, on the real Stage-1 payload.

Per-row detail comes from `evidence/orch802/orch802_flip_detail.py`, with output in `flip_detail_nonEntE.txt` and `flip_detail_EntE.txt` (G11 reports `04` and `05`). **No flip row is left with zero other actors.** Every row keeps at least one catalyst that is not the flipped name; this was checked mechanically over both detail files.

#### Classification of all 34 distinct flips (occurrence counts in brackets)

| actor | occurs inside | rows (inferred on row) | class |
|---|---|---|---|
| `DUT` | "dUTP" | 3 (1) | **false substring**: F1 itself |
| `EntE` | "enterobactin" / "Enterobactin" | 17 distinct haystacks, 67 occurrences, 37 with EntE inferred on the row | **false substring**. This is the F-058 class: "ente" is a prefix of "enterobactin". Every row keeps `EntF` and/or the enterobactin synthase (complex) |
| `EntE` | whole-word "EntE" in "(EntD/EntE/EntF complex)", but the only cue ("catalyzed") is more than 80 characters away; the old matcher hit "Ent" in "enterobactin, catalyzed" | 2 distinct haystacks, 5 occurrences, 3 inferred (`runs_verify/2026-08-24_1203` PMC12452463 strict r3 and research r3; `noloss_scan.json` lines 77 and 546) | **real enzyme name, dropped** (reviewer F1). Each row keeps the enterobactin synthase complex |
| `MenA` | "deMethyl**mena**quinone" on BsUbiE / UbiE rows | 4 (1) | **false substring**. MenA is a prenyltransferase, not the methyltransferase. The fix removes a fabricated catalyst |
| `UbiE` | "Bs**UbiE**" | 3 (0) | predicate-only. Nothing was injected: `BsUbiE` or `UbiE` is already on each row |
| `MenD` | "**menD**HCEB operon" | 1 (1) | **ambiguous**. MenD is one of the six operon enzymes the sentence names collectively, on a RAG-summary row (`isochorismate -> 1,4-dihydroxy-2-naphthoate`, PMC12421875 research) that already carries `menDHCEB` |
| `ALAS` | "**ALAS**2-catalyzed" (digit suffix) | 3 (1) | **real enzyme name, redundant duplicate**. `ALAS2` stays on every row. On strict r1, the one row where the `ALAS` injection was realized, `ALAS2` is a Stage-2 row (provenance `inferred`, confidence 0.95), not an extraction (reviewer F2). On the other two rows it is extracted (confidence 1.0). `ALAS` is the family name of the same enzyme (PMC12856317, run `runs/2026-07-28_2122`) |
| `HepPPS` | "Bs**HepPPS**" (species prefix) | 1 (1) | **real enzyme name, redundant duplicate**. `BsHepPPS` is extracted and stays (PMC12657337 research, `runs/2026-07-28_2122`) |
| `MenA` | "Ec**MenA**" (species prefix) | 2 (2) | **real enzyme name, redundant duplicate**. `EcMenA` is extracted and stays (PMC12657337 strict `runs/2026-08-02_2130`, research `runs/2026-07-28_2122`) |

**EntE count reconciliation (reviewer F4).** The "38 + 12" figure in the previous version counted actor *entries* at confidence 0.9 only. The scan's `actor_attached_inferred_on_row` sum of 40 counts *rows*. Recounted over `flip_detail_EntE.txt` (72 occurrence blocks): 32 blocks have no inferred EntE, 27 have one in `enzymes` only, and 13 have one in both `enzymes` and `modifiers`. That gives 40 rows and 53 entries: 40 in enzymes (38 at 0.9, 2 at 0.85) and 13 in modifiers (12 at 0.9, 1 at 0.85). The split by class is 37 rows (false substring) plus 3 rows (whole-word, far cue).

The `ALAS`, `HepPPS`, `MenA` (EcMenA) and whole-word `EntE` rows are why this task stopped:
- Each realized injection has confidence 0.9 and evidence that is a cue window starting with the reaction name. That is the Stage-2 injector's fingerprint.
- None of them leaves its reaction without a catalyst. The same enzyme stays on the row under the paper's spelling.
- But the dropped name does identify the real catalyst. The brief says that this counts as a stop.

Replay B on Stage-1 payloads does not reproduce the `HepPPS` / `ALAS` / `MenA` flips, because those actors are declared after Stage 1. The merged-payload replay (3.4) does reproduce them with the current attachers: the old matcher re-injects each one and the new matcher does not. So they are losses the current pipeline would deliver on those inputs. Each is a duplicate of a catalyst that stays.

### 3.4 Merged-payload replay (reviewer F3)

Files:
- script: `evidence/orch802/orch802_merged_replay.py`
- output: `evidence/orch802/merged_replay.json`
- stdout: `evidence/orch802/merged_replay_stdout.txt`
- G11 report: `evidence/g11/ORCH-802/07-merged-replay.json` (exit 0, `FINAL SURVIVING COUNT : 0`)

Why: production runs the injector on the merged payload (pipeline.py:1260) and the normalizer at Stage 5, both with more declared entities than Stage 1. Under the exactly-one gates, removing a spurious second qualifier could newly *enable* an injection. Replay B (Stage 1 only) could not see that.

**Which rows are stripped before replay.** The rule comes from the code:
- The injector (pipeline.py:3016) writes a `modifiers` row: confidence 0.9, provenance `inferred`, evidence = the cue window.
- `apply_post_merge_cleanup` -> `_normalize_reaction_actors` -> `_clean_enzymes` (pipeline.py:815, 1858) folds that row into `enzymes` as `{protein|protein_complex, evidence, provenance, confidence}`. That is the shape stored in `merged_payload.json`.
- A row is stripped as an injector row only if its provenance is `inferred`, its confidence is 0.9, and its evidence equals **exactly** either the old matcher's window for that actor on that row, or `evidence[:120]` (the pre-hardening injector's slice).
- The normalizer (process_normalizer.py:3738) writes confidence 0.75 with the cue window as evidence. The same exact-evidence test applies.
- Stage-2 LLM `inferred` rows do not carry the exact window, so they stay.

Then both attachers ran on each stripped payload, once under the old matcher and once under the new, and the results were diffed.

```
{"actor_rows_without_carrier_not_stripped": 742, "attacher_rows_stripped": 61, "files": {"committed": 187, "extra_untracked": 3}, "legs_replayed": 178, ..., "reaction_rows_differing": 29, "rows_with_additions": 0, "rows_with_drops": 29, "stripped_by": {"injector": 61, "normalizer": 0}, "stripped_recreated_by_old_replay": 43, "task": "ORCH-802"}
```

- **Additions: 0.** No injection is newly enabled anywhere.
- **Drops: 29 rows.** Every one keeps at least one actor (`grep -c '"actors_new": []'` gives 0). By class:

| class | rows |
|---|---|
| `EntE`, false substring in "enterobactin". Includes 1 row where EntE is attached to the FepA transport row `FepA-mediated transport of ferric enterobactin` (`runs_verify/2026-08-04_1207` PMC12452463 strict r4), which is plainly a wrong catalyst | 20 |
| `EntE`, whole-word mention with a far cue (`runs_verify/2026-08-24_1203` PMC12452463 strict r3 and research r3) | 2 |
| `DUT` inside "dUTP" (ORCH-801 cdadc1 R0; F1) | 1 |
| `MenA` inside "demethylmenaquinone" on `UbiE reaction` (`runs/2026-08-02_2130` PMC12657337 research r4), a wrong catalyst | 1 |
| `MenD` on the menDHCEB operon sentence (ambiguous, Q2) | 1 |
| `ALAS` beside `ALAS2` (real name, duplicate) | 1 |
| `HepPPS` beside `BsHepPPS` (real name, duplicate) | 1 |
| `MenA` beside `EcMenA` (real name, duplicate) | 2 |

**Fidelity limits, stated:**
- The replay input is the *post-cleanup* merged payload. The injector really runs before `screen_additions` and `apply_post_merge_cleanup`.
- 18 of the 61 stripped rows are **not** re-created by the old-matcher replay; 43 of 61 are. Those 18 produce no diff under either matcher, so they do not affect the result. I did not trace why each was not re-created.
- 742 actor rows carry neither provenance nor confidence. These are payloads written before 2026-07-29, when `_clean_enzymes` still stripped both. Injector rows among them cannot be told apart from extractions, so none of them was stripped.

## 4. SMOKE, and the corpus replay (skipped)

**SMOKE at tip `2718851e`** (the 22-file block in TEST_MATRIX.md), run after the stop on the orchestrator's instruction:
- Result: `511 passed in 74.37s (0:01:14)`.
- Wrapper: exit code 0, `FINAL SURVIVING COUNT : 0`, cleanup success, heavy lock acquired and released.
- Report: `evidence/g11/ORCH-802/06-smoke.json`.

**Corpus replay: skipped on the orchestrator's instruction.** The argument in 3.2 is the reason: the replay starts at `final_mapped.json`, which is downstream of both cue attachers, and none of its replayed functions reaches `normalize_process_payload`. So it cannot exercise `cue_near_name`. The orchestrator accepted that argument as sufficient. No corpus output exists for this task.

## 5. Open for the PO

**Q1.** Is dropping an injected real-name catalyst a regression when the row keeps another catalyst? The cases:
- `ALAS` beside `ALAS2`
- `HepPPS` beside `BsHepPPS`
- `MenA` beside `EcMenA`
- `EntE` on the two `(EntD/EntE/EntF complex)` rows, where the whole-word mention is more than 80 characters from the cue and each row keeps the enterobactin synthase complex. The old matcher reached EntE there only through "enterobactin, catalyzed". Under the new matcher the name is still seen, but there is no cue within the window, which is the gate's normal behaviour.

| Option | What happens | Cost |
|---|---|---|
| (a) Not a regression | The duplicate is a second enzyme node for one enzyme; arguably a correctness defect of its own. Merge as is | The PWML loses the generic-name enzyme node on those rows, and its identity mapping might differ from the prefixed name's |
| (b) Loss | Keep matching when the flank is a species prefix (`Bs`/`Ec`) or an isoform digit. That reopens a lexical exception list | `CYP1` against `CYP11A1` is the same shape as `ALAS` against `ALAS2`, so a digit exception would re-admit a false class. Expect the red-team non-convergence pattern seen in ORCH-746 |
| (c) Defer | Hold ORCH-802 and leave the DUT fabrication live | F1 stays a live `product_contract_violation` |

**Q2.** Should `MenD` on the `menDHCEB` operon sentence count as a real attachment? It is one of six enzymes, named collectively on a summary row.

**Q3.** The enzyme rule (`\w`, and an assertion after `+`) and the transporter rule (ASCII letters and digits, edge-conditional) now differ. Should they be unified in a separate lane?

## 6. Dead ends and failures

- **Base via `PYTHONPATH` alone does not work.** The test file's `sys.path` insert and `pytest.ini` `pythonpath = src` resolve to the tip tree. This was found by reading the code before running, and replaced by the in-export copy (3.1).
- **First base run's counts were lost.** The `tail` cut the pytest summary. Re-run as 01b.
- **Scan run 03 crashed on a cp1252 console** (U+2032 in a snippet). Fixed with `sys.stdout.reconfigure`, re-run as 03b. The two runs produced the same JSON numbers.
- **The audit's suggested safety twin, "DUT hydrolyzes dUTP to dUMP", is not a valid control.** "hydrolyzes" is not in `ENZYME_EVIDENCE_CUE_RE`, so it would not attach on base either. Replaced by "dUTP hydrolysis catalyzed by DUT".
- **The `cue_attacher_shape` column in the detail files is `False` for every row.** Merged rows no longer carry `source_refs`. Attacher origin was therefore read from confidence 0.9 plus the cue-window evidence shape, not from that column.
