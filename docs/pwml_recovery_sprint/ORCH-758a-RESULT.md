# ORCH-758a — charge notation is never a composite separator (defect D2)

**Status:** implemented and gated on the lane branch. **Not reviewed, not merged.** An independent
reviewer judges the actual diff next.

| | |
|---|---|
| branch | `orch758a/ion-charge-plus` (worktree `.claude/worktrees/orch758a`) |
| base | `3582b0d7` (approved ORCH-757 tip) |
| test-first commit (base code) | `c2a88027` |
| code commit | `b043803a` |
| docs/evidence commit | the commit that adds this file |
| diagnosis | `ORCH-758-RESULT.md` § 4 D2 (orch758 worktree) |
| evidence leg (read-only) | `runs_papers/ORCH-755/legs/sour_taste-strict/2026-10-21_0139/papers/PMC7059786/strict/` |

No LLM call, no pipeline run, no prompt change, and no edit to `streamlit_app.py`. Every number
below names its G11 report under `evidence/g11/ORCH-758a/` and its log under `evidence/orch758a/`.

---

## 1. The defect

Stage 1 wrote the compound `"Ca++"` and the protein `"voltage-gated Ca ++ channels"` (the paper
prints `Ca ++`). `_has_plus_token` (`process_normalizer.py:355-360` at base) stripped one trailing
`+` or one trailing `<digits>+`, then treated any `+` that was left as an `A + B` separator:

- `Ca++` became `Ca+`, which still holds a `+`, so it read as a composite. Strict
  `normalize_composites` raised: `Composite entity 'Ca++' in /entities/compounds has no
  protein-like left component; unsupported.` This is the `gate_fail_report.json` text of the leg.
  Reproduced on the leg's own `merged_payload.json` at base src: `g11/ORCH-758a/11`.
- `Fe+++`, `Ca ++`, and every ion inside a name failed the same way. That covers
  `voltage-gated Ca ++ channels`, `Na+/K+-ATPase`, `Ca2+-ATPase` and `Mg2+/ATP`. A protein of that
  kind in `element_locations.protein_locations` raised in `_rewrite_token`: `Composite token
  'Na+/K+-ATPase' ... compound+compound composite materialization is not supported`
  (`orch758a/01-g9-red.log`).

The app catches the exception and keeps the un-normalized payload. `normalize_process_actor_schema`
therefore never ran, and the pre-export Stage-3 revalidation stopped the PWML on
`actor_schema_not_canonical`.

## 2. The diff

The change is in one function, `src/t2pw/pipeline/process_normalizer.py` (`+16 −5`). The old
two-step strip is replaced by one module regex, `_CHARGE_PLUS_RE`. That regex removes the charge
runs, and any `+` left over is a separator:

```python
_CHARGE_PLUS_RE = re.compile(
    r"\s*\d*\++$"
    r"|(?:\s?\+{2,}|(?<=[A-Za-z])\s?\d+\+|(?<=[A-Za-z0-9])\+(?=[/-]))(?=[\s/\-)\],])"
)

def _has_plus_token(value: str) -> bool:
    return "+" in _CHARGE_PLUS_RE.sub("", _canonical(value))
```

- **At the end of a name:** any run of `+`, with optional digits and one optional space before it.
  Covers `NAD+`, `H+`, `Ca2+`, `Ca++`, `Fe+++`, `Ca ++`, `Ca 2+`. The old code handled only the
  first three.
- **Inside a name**, followed by a space, `/`, `-`, `)`, `]` or `,`: a `++` run (`Ca ++ channels`),
  element digits then `+` (`Ca2+ channel`, `Mg2+/ATP`), or one `+` attached to the left before `/`
  or `-` (`Na+/K+-ATPase`, `H+-ATPase`).
- **Still separators:** a lone spaced `+` (`A + B`, `NADH + H+`, `Ca2+ + calmodulin`) and a `+`
  between two words (`glucose+fructose`, `EntB+EntE`).

The function name, `_split_composite`, mapping, identity, and every other stage are unchanged. No
entity is renamed: the fixed tests assert that `Ca++` and the channel keep their Stage-1 names.

| name | base `_has_plus_token` | tip |
|---|---|---|
| `Ca++`, `Fe+++`, `Ca ++` | True (crash) | **False** |
| `voltage-gated Ca ++ channels`, `voltage-gated Ca++ channels` | True (crash) | **False** |
| `Na+/K+-ATPase`, `Ca2+-ATPase`, `Mg2+/ATP` | True (crash) | **False** |
| `Ca2+`, `Ca 2+`, `NAD+`, `H+`, `Mg2+` | False | False |
| `NADH + H+`, `A + B`, `EntB + EntE`, `EntB+EntE`, `glucose+fructose`, `ATP + Ca2+`, `Ca2+ + calmodulin`, `Ca++ + calmodulin` | True | True |

## 3. Every caller checked

All of them are in `process_normalizer.py`; `grep` finds no caller outside it. Each one routes
through the single changed function, so the fix lands once:

| caller | what it does with a `True` | effect of the fix |
|---|---|---|
| `_rewrite_token` (`:1181`) | splits and materializes, or raises in strict mode | ion and channel names pass through as one token |
| `normalize_composites` complexes (`:1481`) | materializes, or raises | a complex named with an ion is no longer split on its charge |
| `normalize_composites` compounds (`:1525`) | materializes, or **raises** (the D2 crash) | `Ca++` is kept as a compound |
| `_token_parts_for_aliasing` (`:2320`) | adds the split parts as alias keys | no false alias keys such as `voltage-gated Ca` / `channels` |
| `canonicalize_same_as_aliases._rewrite_token` (`:2567`) | rewrites each part and rejoins with ` + ` | the name is rewritten whole |
| `validate_no_composites` (`:4309`, `:4317`, `:4325`) | reports `'+' token` | ion names are no longer reported |
| `compute_normalization_stats` (`:4446`, `:4455`, `:4462`) | counts `n_plus_tokens_remaining` | ion names are no longer counted |

`_split_composite` is **unchanged** and is only reached when `_has_plus_token` is True, so a real
composite splits exactly as at base (see § 7, F5). Sibling splitters in other stages were checked
and left alone because they are out of scope: `pipeline.py:_split_composite_token` (Stage-1
cleaning) and `curation/audit_json_llm.py:_split_composite_name` (§ 7, F6).

## 4. Proofs (G9)

This is a correction of observable behaviour (a crash), so the proof has to fail at base.

| run | tree | G11 | result |
|---|---|---|---|
| G9 RED | `c2a88027` (test only, base code) | `01-g9-red` | **13 failed, 15 passed.** All 13 failures are `test_g9_*`, and each is the defect's own `ValueError` (`Composite entity 'Ca++' …` ×4, `'Fe+++'`, `'Ca ++'`, `Composite token 'Na+/K+-ATPase'`, `'Ca2+-ATPase'`, and `Composite validation failed` ×5) |
| G9 GREEN | `b043803a` | `02-g9-green` | **28 passed** |

The test file is `tests/test_orch758a_charge_notation_not_composite.py`:

- `test_g9_strict_normalization_completes_with_charge_notation` has 7 cases. Each is a minimal
  strict payload: one ion moved by one channel, with the transporter in the Stage-1 shape
  `{"protein": …}`. At tip, normalization completes, nothing is renamed, no complex is minted, the
  transporter has `entity` and `entity_type == "protein"`, `validate_post_normalization` gives 0
  `actor_schema_not_canonical`, and `n_plus_tokens_remaining == 0`.
- `test_g9_validate_no_composites_accepts_charge_notation` has 5 cases.
- `test_g9_orch755_sour_taste_excerpt_normalizes_in_strict_mode` is the real-shape replay. Its
  fixture, `tests/fixtures/orch758a/sour_taste_PMC7059786_merged_excerpt.json` (18 KB), copies rows
  verbatim from the leg's `merged_payload.json` (sha256 `bec156e6…c2f`) using
  `evidence/orch758a/orch758a_make_fixture.py`. It keeps transports 0, 1, 3 and 5 (the three OTOP1
  H+ rows and the Ca++ channel row), plus their entities, states and locations. The test checks that
  the three OTOP1 rows enter in Stage-1 shape and come out canonical, and that there are 0
  `actor_schema_not_canonical`. The file name avoids `final_mapped.json`, because four corpus tests
  `rglob` for that name.
- `test_preserved_*` has 15 cases and passes at both base and tip. `NADH + H+`, `glucose+fructose`,
  `ATP + Ca2+`, `Ca2+ + calmodulin` and `Ca++ + calmodulin` still abort strict
  `normalize_composites`. `NADH + H+`, `A + B` and `EntB+EntE` still fail `validate_no_composites`.
  `Ca2+`, `Ca 2+`, `NAD+`, `H+` and `Mg2+` were already non-composite. `EntB + EntE` and
  `EntB+EntE` still materialize `EntB:EntE`.

**Whole-leg replay.** Strict `normalize_process_payload` followed by `validate_post_normalization`
on the full leg `merged_payload.json`, run by `evidence/orch758a/orch758a_replay.py`:

| G11 | src | result |
|---|---|---|
| `11-replay-merged-base` | base | `ValueError: Composite entity 'Ca++' …`, the leg's failure reproduced |
| `12-replay-merged-tip` | tip | completes; `n_plus_tokens_remaining 0`; **0 `actor_schema_not_canonical`**; every transporter row is `{entity, entity_type, protein}`. The gate lists 5 issues, all `Protein '…' is missing a UniProt or DrugBank identifier`. These are expected: `merged_payload.json` is the payload before Stage-2 mapping (F3) |
| `13-replay-fixture-tip` | tip | the committed excerpt completes; transporters canonical |

## 5. Gates

| # | gate | G11 | result |
|---|---|---|---|
| 1 | G9 RED / GREEN | `01` / `02` | 13 failed, 15 passed / 28 passed |
| 2 | affected suites: 52 files, 2 chunks (every `tests/*.py` matching `process_normalizer`, `stage_contracts`, `validate_no_composites`, `normalize_composites` or `composite`, plus `test_stage3_gate_report_lifecycle.py` and `test_strict_quarantine*.py`, minus the AppTest files run alone in gate 5) | `03-affected-1` / `04-affected-2` | 639 passed, 3 skipped / **760 passed, 8 skipped, 1 failed** |
| 2a | that 1 failure: `test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform` | `05-base-known-fail` (base `process_normalizer.py` checked out) | fails the same way at base (`StopIteration`). It is an AST check of `streamlit_app.py`, and it is listed in TEST_MATRIX § ORCH-754 "Known failures" |
| 3 | SMOKE (22 files) | `06-smoke` | **508 passed** |
| 4 | `tests/test_eval_v2_*.py` | `07-eval-v2` | 89 passed, 1 xfailed |
| 5 | `test_streamlit_quarantine_boundary.py` alone | `08` | 23 passed |
| 5 | `test_c052_prefreeze_report_at_the_streamlit_seams.py` alone | `09` | 13 passed |
| 5 | `test_c055_rag_loop_wiring.py` alone | `10` | 14 passed |

Every report shows `FINAL SURVIVING COUNT : 0`. `g11_evidence.py check --task ORCH-758a` reports
**0 non-compliant**, and all 10 pytest reports carry a committed pin verdict. Reports `05`, `11` and
`14` ran with base `process_normalizer.py` checked out on purpose, so the wrapper records
`worktree dirty=True` for them. It was restored with `git checkout HEAD --` straight after each run,
and every other report is `dirty=False`. Skip reasons were not captured, because `-q` was run
without `-rs`. Chunk D was not run; the orchestrator runs it.

## 6. Files changed

- `src/t2pw/pipeline/process_normalizer.py`: `_CHARGE_PLUS_RE` and `_has_plus_token` (code commit)
- `tests/test_orch758a_charge_notation_not_composite.py`: new (test-first commit)
- `tests/fixtures/orch758a/sour_taste_PMC7059786_merged_excerpt.json`: new (test-first commit)
- `docs/pwml_recovery_sprint/ORCH-758a-RESULT.md`, `evidence/g11/ORCH-758a/01-14`,
  `evidence/g11/pin/ORCH-758a/`, `evidence/orch758a/` (logs and three helper scripts): docs commit

## 7. Findings, failures and dead ends

- **F1. A fix for the trailing charge only was not enough. I believed it would be, and a replay
  disproved it.** My first patch stripped only a trailing `\s*\d*\++$`. It cleared `Ca++`. Strict
  normalization of the leg's `merged_payload.json` then raised one pass later, on the protein:
  `Composite token 'voltage-gated Ca ++ channels' at /element_locations/protein_locations/3/protein
  has no protein-like left component` (scratch replay, not committed). The paper's spaced ion also
  appears inside the channel name, so charge runs inside a name had to be handled too.
- **F2. `final_mapped.json` is not a RED fixture.** It no longer holds `Ca++`, because the audit
  loop dropped the Ca++ row and the channel. Strict normalization of it already passes the gate
  (scratch replay at tip: gate ok, 0 contract errors). `merged_payload.json` does hold the failing
  rows, so the fixture is cut from that file.
- **F3. The leg directory holds no Stage-2 mapped payload, which is the app's actual
  normalization input.** The replays use `merged_payload.json` (Stage 1 merged, unmapped). That is
  why the tip replay's gate lists missing UniProt/DrugBank IDs. Those 5 issues have no `code` key,
  and none is `actor_schema_not_canonical`.
- **F4. Pre-existing, not caused by this lane: a protein_locations name that differs from a
  registered protein mints a compound row with the protein's name.** In replay `12`, the
  unregistered spelling `voltage-gated Ca++ channels` (no space; it appears only in a
  protein_location and a SAME_AS interaction) leaves the compound
  `voltage-gated Ca ++ channels` next to the protein of the same name. I made a variant with no
  `+` anywhere (`orch758a_noplus_variant.py`: channel names become `voltage-gated calcium
  channel(s)`, and `Ca++` becomes `Ca2+`) and ran it at **base** src (`14-replay-noplus-variant-base`).
  It produces the same duplicate (`compounds … 'voltage-gated calcium channel'` and the protein of
  the same name), so this is base behaviour that only became reachable here. The path I believe is
  responsible is `_rewrite_token` → `_ensure_compound` for a non-protein-like name, followed by
  SAME_AS canonicalization. I have not traced it further. This lane does not touch it.
- **F5. Real composites still lose the charge of their members when split, as at base.**
  `_split_composite("NADH + H+")` gives `["NADH", "H"]`, and `"Ca2+ + calmodulin"` gives
  `["Ca2", "calmodulin"]`, because the split regex eats the charge `+`. It is unchanged here,
  because the brief requires real composites to split exactly as at base. See § 8, Q2.
- **F6. Out-of-stage siblings, not changed.** `pipeline.py:_split_composite_token` splits Stage-1
  reaction inputs and outputs on `\s*\+\s*|\s+and\s+`, so a reaction side holding `Ca++` would
  become `Ca` there. `audit_json_llm._split_composite_name` is the curation copy. Sour taste has 0
  reactions, so neither was on the D2 path.
- **F7. Known limitation of the lexical rule** (it is marked with a `ponytail:` comment in the
  code). A composite with uneven spacing whose left member ends in a digit, such as `CO2+ H2O`, now
  reads as one name. Two things are unchanged from base: a single `+` attached to a word and
  followed by a space (`K+ channel`, `NAD+ kinase`) still reads as a separator, and a well-spaced
  `X + Y` is still a composite.

## 8. Not settled: for the product owner

- **Q1. The app continues with the un-normalized payload after any normalization exception**
  (`streamlit_app.py:3524` keeps `normalized_input = stage2_payload`; the catch is at `:3655`).
  D2 shows the cost: a pathway that is correctly `review_required` reaches the exporter in
  Stage-1 shape, and the exporter's schema check stops the PWML. This lane removes one trigger,
  not the class. Options:
  (a) Abort the leg on any normalization exception, as the `normalize_composites` comment "Strict mode aborts the whole run here" already claims.
      This loses the reviewable artifacts.
  (b) Carry the exception into the release state as an explicit `review_required` reason, and
      skip PWML export with that reason stated.
  (c) Leave it as it is.
  `streamlit_app.py` is protected, so any change needs a D-xxx.
- **Q2. Should `_split_composite` keep member charges** (`NADH + H+` → `NADH`, `H+` rather than
  `H`)? That changes the identity of real composites, so it is a ruling, not a bug fix. Cost: it
  moves names at base for every composite with a charged member, and may move pinned baselines.
- **Q3. F4's duplicate compound/protein row.** It is pre-existing and was reached here because
  normalization now completes. The next live sour-taste leg will show whether it reaches the PWML.
- **Q4. F6's Stage-1 splitter** has the same charge blindness for reaction sides. It is a
  different stage and is not on the D2 path.

## 9. TEST_MATRIX delta (for the orchestrator to apply; I did not edit it)

Append at the end of the file:

```
## ORCH-758a (D2): test obligations — appended end-of-file by rule · 2026-10-21

New focused file: `tests/test_orch758a_charge_notation_not_composite.py` (28 tests, <1 s), with
fixture `tests/fixtures/orch758a/sour_taste_PMC7059786_merged_excerpt.json`. It is in no chunk
and not in SMOKE; its focused selection is the file itself. G9: 13 `test_g9_*` red at base
(`g11/ORCH-758a/01`), 28 green at tip (`02`).
SMOKE unchanged at 508 (`g11/ORCH-758a/06`). No pin moved.
Affected sweep, 52 files: 1399 passed, 11 skipped, 1 failed (`03`, `04`). The failure is
`test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform`,
already in § ORCH-754 "Known failures" and red at base (`05`).
```

## 10. Orchestrator record: review and merge (appended by the sub-orchestrator, 2026-10-22)

Chartered by the ORCH-756-759 sub-orchestrator from `ORCH-758-RESULT.md` § 4 D2 (blocks PWML production after the
budget fix; a defect, not a product choice), base `3582b0d7` (approved ORCH-757 tip).

**Review** (`pwml-reviewer`, Opus, extra-high, round 0; severity standard fixed in the prompt: a byte outside
scope; a composite regression on REAL data, measured by a corpus scan; a non-behavioural G9; an LLM-bound change; a
lane-caused gate failure; a weakened gate or dropped `review_required` pathway). **Verdict: APPROVE WITH NOTES.**
- G9 base `g11/ORCH-758a/01-rev-g9-base` 13 failed / 15 passed, each with the defect's own `ValueError`; tip `15` 28
  passed; SMOKE `18` 508; affected 8 files `19` 247 passed, 2 failed, both the known
  `test_compound_resolution_extraction.py` GOLDEN-coverage pair, red at base (`03`).
- **Corpus scan** (`02`/`16`/`17`): 8,088 files, 10,632 distinct `+` strings (fixtures, evaluation_v2, evidence,
  189 runs_verify payloads, ORCH-755 payloads): **12 names changed, all True→False, all charge notation** (`Ca++`,
  `voltage-gated Ca ++ channels`, `ferric iron (Fe3+)`, …); none False→True; no real composite changed.
- Fixture regeneration reproduced the committed fixture (`20`).
- Notes: (1) the charge rule is wider than its `ponytail:` comment: hypothetical `CO2+ H2O`, `H2O2+ O2`, `A ++ B` now
  read as one name; none occurs in the corpus (`04`/`21`); (2) `K+ channel`, `NAD+ kinase`, `H+ influx via OTOP1`
  still read as composites, as at base, only as process names `validate_no_composites` does not check; (3) the false
  "repair the Ca++ composite" errors no longer reach the Stage-4 audit LLM's context note, which removes the defect
  and may stop the audit dropping `Ca++` (F2); watch the next sour-taste leg; (4) the sibling splitters
  `pipeline.py:1689` and `audit_json_llm.py:230` still split `Ca++` into `Ca` (Q4).

**Merge:** `--no-ff` into `sprint/pwml-recovery` = **`e9c852a5`**, after Chunk D on `e4520a01` (`ORCH-757i/01`,
187/187) and the merged-tip gates (`ORCH-757i/34`, `67`-`74`); pushed with `main` at `f392d447`. Open questions
Q1-Q4 (§ 8) go to the product owner in the sub-orchestrator's final report.
