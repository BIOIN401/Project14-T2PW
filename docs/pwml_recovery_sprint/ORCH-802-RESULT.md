# ORCH-802 RESULT: word-boundary actor match in `enzyme_cues.cue_near_name`

**Status: STOPPED at the no-loss classification.** This is not a merge candidate yet.

- The fix and its G9 proof are done: base fails, tip passes.
- The no-loss scan found three injected actor names that are lost and that name a real catalyst: `ALAS`, `HepPPS` and `MenA`. Each one duplicates a catalyst that stays on the same row under the paper's own name.
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
| `EntE` | "enterobactin" / "Enterobactin" | 19 distinct haystacks, 72 occurrences (38 enzyme + 12 modifier inferred rows) | **false substring**. This is the F-058 class: "ente" is a prefix of "enterobactin". Every row keeps `EntF` and/or the enterobactin synthase (complex) |
| `MenA` | "deMethyl**mena**quinone" on BsUbiE / UbiE rows | 4 (1) | **false substring**. MenA is a prenyltransferase, not the methyltransferase. The fix removes a fabricated catalyst |
| `UbiE` | "Bs**UbiE**" | 3 (0) | predicate-only. Nothing was injected: `BsUbiE` or `UbiE` is already on each row |
| `MenD` | "**menD**HCEB operon" | 1 (1) | **ambiguous**. MenD is one of the six operon enzymes the sentence names collectively, on a RAG-summary row (`isochorismate -> 1,4-dihydroxy-2-naphthoate`, PMC12421875 research) that already carries `menDHCEB` |
| `ALAS` | "**ALAS**2-catalyzed" (digit suffix) | 3 (1) | **real enzyme name, redundant duplicate**. The row's own catalyst `ALAS2` is extracted and stays. `ALAS` is the family name of the same enzyme (PMC12856317, run `runs/2026-07-28_2122`) |
| `HepPPS` | "Bs**HepPPS**" (species prefix) | 1 (1) | **real enzyme name, redundant duplicate**. `BsHepPPS` is extracted and stays (PMC12657337 research, `runs/2026-07-28_2122`) |
| `MenA` | "Ec**MenA**" (species prefix) | 2 (2) | **real enzyme name, redundant duplicate**. `EcMenA` is extracted and stays (PMC12657337 strict `runs/2026-08-02_2130`, research `runs/2026-07-28_2122`) |

The `ALAS`, `HepPPS` and `MenA` (EcMenA) rows are why this task stopped:
- Each realized injection has confidence 0.9 and evidence that is a cue window starting with the reaction name. That is the Stage-2 injector's fingerprint.
- None of them leaves its reaction without a catalyst. The same enzyme stays on the row under the paper's spelling.
- But the dropped name does identify the real catalyst. The brief says that this counts as a stop.

These flips occur only in pre-August committed merged payloads. Replay B on the matching Stage-1 payloads does not reproduce them under either matcher; the `HepPPS` / `ALAS` / `MenA` actors are declared after Stage 1. So they are predicate-level losses. I did not show them as losses the current pipeline would deliver.

## 4. SMOKE, and the corpus replay (skipped)

**SMOKE at tip `2718851e`** (the 22-file block in TEST_MATRIX.md), run after the stop on the orchestrator's instruction:
- Result: `511 passed in 74.37s (0:01:14)`.
- Wrapper: exit code 0, `FINAL SURVIVING COUNT : 0`, cleanup success, heavy lock acquired and released.
- Report: `evidence/g11/ORCH-802/06-smoke.json`.

**Corpus replay: skipped on the orchestrator's instruction.** The argument in 3.2 is the reason: the replay starts at `final_mapped.json`, which is downstream of both cue attachers, and none of its replayed functions reaches `normalize_process_payload`. So it cannot exercise `cue_near_name`. The orchestrator accepted that argument as sufficient. No corpus output exists for this task.

## 5. Open for the PO

**Q1.** Is dropping an injected *alias* of a catalyst that is already attached a regression? The cases are `ALAS` beside `ALAS2`, `HepPPS` beside `BsHepPPS`, and `MenA` beside `EcMenA`.

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
