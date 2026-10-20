# ORCH-754d: production stops importing the evaluator (layering repair)

| | |
|---|---|
| Branch | `orch754/d-eval-layering` |
| Base | `adc49c40` (`sprint/pwml-recovery` with ORCH-754a/b/c merged) |
| Code commit | `e42dccfd` |
| Reports | `evidence/g11/ORCH-754d/01`–`19` (pins under `evidence/g11/pin/ORCH-754d/`) |
| Status | ready for review. All gates green, zero surviving processes, G11 check clean |

## 1. The defect and where it came from

`tests/test_eval_v2_metrics.py::test_no_production_module_imports_evaluation_v2` (`:377-384`)
fails at base. It reads every `.py` under `src/t2pw/` outside an `eval_v2` directory and
fails on any occurrence of the package name. The rule exists so that production never
depends on the evaluator that measures it: if both shared one normalizer, a bug in it would
act on both sides at once and the measurement could never see it.

Measured with `git grep` over the history:

| revision | files outside the evaluator naming it | evaluator import sites in `src/t2pw/pipeline` |
|---|---|---|
| `70d05e03^` | none | 0 |
| `70d05e03` (ORCH-750, N-draw ensembling) | `stage_one_ensemble.py` | 1 |
| `f9892cdf` | `stage_one_ensemble.py` | 1 |
| `adc49c40` (ORCH-754a merged) | `stage_one_ensemble.py`, `paper_lexicon.py` | 6 (5 + 1) |
| `e42dccfd` (this task) | none | 0 |

So ORCH-750 introduced it and ORCH-754a extended it. The imports were all function-local
(`strict_key`, `text_key`, `mentions`), plus two docstring mentions. Report `/01` shows the
test failing at base with exactly those two files as offenders.

## 2. The fix (`e42dccfd`)

| file | change |
|---|---|
| `src/t2pw/pipeline/name_keys.py` | **new.** Production's own copies of the evaluation matcher's `strict_key`, `term_keys` (used by `mentions`), `text_key` and `mentions`, plus the v1 bench's `canonical_text` / `normalize_name` and the `_GREEK` / `_DASHES` / `_TRAILING_NOISE` tables under them. Standard library only; it imports nothing from `t2pw`. The docstring says why the copy exists without writing the forbidden string. `__all__` exports only the three functions production uses. |
| `src/t2pw/pipeline/stage_one_ensemble.py` | One module-level import, `from t2pw.pipeline.name_keys import mentions, strict_key, text_key`. The five function-local evaluator imports are deleted. The module-docstring paragraph is reworded to say what the key is now, and why. |
| `src/t2pw/pipeline/paper_lexicon.py` | A module-level import of `strict_key` from the copy. The function-local import is deleted and the docstring reworded. |
| `tests/test_name_keys_match_evaluation_matcher.py` | **new**, labelled as a new acceptance test (§ 3.2). |

The evaluator's code and the layering test are untouched (`git diff adc49c40 e42dccfd --stat`
lists only the four files above).

Why `normalize_name` is copied as well, rather than imported from `t2pw.bench.goldset`: the
matcher's `strict_key` is built on it, so importing it would leave production depending on
the evaluation layer through the back door, sharing exactly the normalizer the rule protects.
It would also break the codebase's own precedent: `src/t2pw/curation/apply_audit_patch.py:1346`
says "t2pw.bench is the evaluation layer" and copies `normalize_name`'s folding rather than
importing it, and `tests/test_c105_actor_role_evidence.py` pins those two copies against each
other. This task follows the same pattern.

Why the imports are now module-level: the function-local form existed only to avoid importing
`t2pw.bench` at module load. `name_keys` is standard library only, so that hazard is gone.
`pipeline.py:59` imports `stage_one_ensemble` at module level, so loading `t2pw.pipeline` now
pulls in no `t2pw.bench` code at all, where before a call pulled in the evaluator lazily.

**How verbatim the copy is.** A scratch line comparison (`C:/t/bt-orch754d/srccmp.py`, not
committed) found every code line of `name_keys.py` identical to a line of `goldset.py` or
`matcher.py`. That includes the literal U+00A0 in `canonical_text`, a byte check worth making
because the authoring tool decoded an escape on the first write. The only lines without an
original are docstrings, the two "copied from" comments, the `typing` import, `__all__`, and
the tail of `mentions`' docstring, which now names "the evaluation matcher's `matches`" because
this module has no `matches`. The equality test (§ 3.2) is the behavioural proof; this was
only a source-level sanity check.

## 3. Proofs

### 3.1 G9: a correction of pre-existing observable behaviour

| | report | result |
|---|---|---|
| layering test at `adc49c40`, before any edit | `/01` | **1 failed**: offenders `src/t2pw/pipeline/paper_lexicon.py`, `src/t2pw/pipeline/stage_one_ensemble.py` |
| the same test at the tip content | `/07`, `/11` | passed |

### 3.2 The equality test: the alarm if the copies drift

`tests/test_name_keys_match_evaluation_matcher.py` is a **new acceptance test**, because
`name_keys` is a new module, and it claims no base failure. It has three tests:

1. **`strict_key` / `text_key` / `normalize_name` / `canonical_text` agree on more than
   2,000 probes.** The probes are:
   - every string in the committed cohort payloads: `tests/fixtures/orch754a/*_payload.json`
     (chlorophyll draw 1 and pooled), `evidence/orch752_chlorophyll_payloads/` (3 draws),
     and `evidence/orch754a_inputs/{PMC12096016_enterobactin,PMC12421875_menaquinone}/`
     (3 draws each). These give participant names, enzyme names, states and evidence quotes;
   - every line of the three committed papers, and each paper whole;
   - hand edge cases: `None`, `""`, NBSP, non-strings; Greek letters in both cases;
     NFKC cases (ligatures, full width, super- and subscripts); the seven unicode dashes;
     whitespace kinds; trailing `(protein)` / `(abbrev.)` notes and parentheticals; the
     `-ic acid` rule; pyrophosphate / diphosphate; locants, digits, case variants,
     `PPI` / `PPi`, `ß`, `İ`.

   Each fixture group is asserted non-empty, so a broken glob cannot make the test vacuous.
2. **`mentions` agrees** for every name of each payload (plus the edge names) against every
   quote of that payload and all three papers, and on 50 calls per payload with aliases.
   Each side keys the text with its own `text_key`, as production composes them. The test
   asserts that both outcomes, True and False, occurred.
3. **`name_keys.py` imports only** `__future__`, `re`, `typing` and `unicodedata` (an AST
   check).

**Does it fire?** `evidence/orch754d_equality_test_mutants.py` (report `/10`) loads ten
one-rule mutants of `name_keys.py` in memory and runs the three tests against each. The
source tree is never edited.

- **Eight mutants are killed:** the pyrophosphate rule dropped, `κ` removed from `_GREEK`,
  the dash map dropped, the `(abbrev.)` note kept, the `-ic acid` word boundary dropped,
  `text_key` unpadded, `mentions` not token-bounded, and `casefold` replaced by `lower`.
- **Two survive, and both are equivalent mutants:**
  - the NBSP branch of `canonical_text` removed: NFKC has already mapped U+00A0 to U+0020
    before that branch is tested;
  - `⁺` removed from the charge class: NFKC has already mapped U+207A to `+` before the regex
    runs.

  The script checks both facts rather than assuming them (finding F1). Verdict:
  `EVERY NON-EQUIVALENT MUTANT KILLED`, exit 0.

### 3.3 Behaviour is byte-identical

Each ORCH-754a proof ran at base (before any edit) and again at the tip content, through the
wrapper, with the same arguments ORCH-754a § 12.4 used (`/219`–`/222` there).

| proof | base | tip | result |
|---|---|---|---|
| gold replays and side files: `evidence/orch754a_ensemble_merge.py` | `/03` | `/16` | chlorophyll pooled **41 → 29**; enterobactin **4/5**, P 0.8; menaquinone **9/9**, P 1.0 |
| N=1 identity: `evidence/orch750_n1_base_identity_proof.py --rev f9892cdf --out evidence/orch754a_n1_identity_verdict.json` | `/04` | `/17` | `N=1 IDENTICAL to base` |
| LLM-bound goldens: `evidence/orch754a_llm_input_base_golden.py` | `/05` | `/18` | `sha256_lf` `d171a3b83f8c…` and `ab01b0ddaec5…`, unchanged |
| `map_payload` identity: `evidence/orch754a_map_payload_base_identity_proof.py` | `/06` | `/19` | `map_payload BASE-IDENTICAL` |

Two checks back the "unchanged" claim:

- **The committed outputs did not move.** Each script rewrites its committed outputs in place:
  `orch754a_ensemble_merge.{json,txt}`, the three `orch754a_*_side_file.txt` renders,
  `orch754a_n1_identity_verdict.json`, `orch754a_map_payload_base_identity_verdict.json`,
  and the four `tests/fixtures/orch754a/` payloads and goldens. After both the base runs and
  the tip runs, `git status --porcelain` listed none of them.
- **The stdout did not move.** The proof bodies captured from the wrapper logs (from the
  first output line to the cleanup banner) are byte-identical, base vs tip, for all four
  pairs. `cmp` on 112, 61, 18 and 114 lines reported IDENTICAL. The logs are
  `C:/t/bt-orch754d/{03..06,16..19}-*.log`.

## 4. Gates

Every gate ran on the exact content later committed as `e42dccfd`. Every code edit came before
report `/07`, and `git status` after the commit shows no modified tracked file. Each row is one
bounded run, with zero surviving processes in its cleanup report.

| # | gate | report | result |
|---|---|---|---|
| 1 | layering test at base / at tip | `/01` / `/07` | 1 failed / passed |
| 2 | new equality test (3 tests) | `/07` | 3 passed (with the layering test: 4 passed) |
| 2a | equality-test mutation check | `/10` | 8 killed, 2 equivalent (checked) |
| 3 | all `tests/test_eval_v2_*.py` (6 files) at base / at tip | `/02` / `/11` | **1 failed, 88 passed, 1 xfailed** / **89 passed, 1 xfailed** |
| 4 | `test_stage_one_ensemble.py` + `test_stage_one_paper_dictionary.py` + the new file, default env | `/12` | **125 passed** (122 + 3 new) |
| 4 | the same, `T2PW_EXTRACTION_DRAWS=3` in the child env | `/13` | **125 passed** |
| 5 | SMOKE, the 22 files of `TEST_MATRIX.md` § Commands, default env | `/14` | **508 passed** |
| 5 | SMOKE, `T2PW_EXTRACTION_DRAWS=3` | `/15` | **508 passed** |
| 6 | byte-identity proofs, base / tip | `/03`–`/06` / `/16`–`/19` | identical (§ 3.3) |
| G11 | `g11_evidence.py check --task ORCH-754d --require-pin --forbid-refused-pin --forbid-foreign-src --require-label-match` | -- | **19 artifacts, 0 non-compliant**, exit 0; 8 direct-pytest reports, all 8 pinned, 0 refused |

Every run used the bounded wrapper from the worktree root:
`PYTHONPATH=<worktree>/src PYTHONIOENCODING=utf-8 T2PW_OFFLINE_CURATOR=1`, `--heavy-lock ORCH-754d`,
and for pytest `pinned_pytest.py --pin-verdict <pin> -q --basetemp=C:/t/bt-orch754d/<report-stem>-<random>`.
For the `draws3` rows, `T2PW_EXTRACTION_DRAWS=3` was exported into the wrapper's env and
inherited by the child. The wrapper report does not record child env (neither did ORCH-754a's
`/215`, `/216`), so the label and this paragraph are the record. The helper that built these
commands is `C:/t/bt-orch754d/run.sh`.

The six AppTest files that must run alone were not run, because no gate listed them. The
touched modules' only importers in the code are `pipeline.py:59` and `map_ids.py:11121`
(`git grep`, excluding tests and docs), and both are exercised by SMOKE. Streamlit code
reaches the touched modules only through `pipeline.py`, and the only new import on that path
is the standard-library-only `name_keys`.

## 5. Findings and dead ends

- **F1. Two lines in the copied normalizer are unreachable.** Both are in the evaluator's
  original as well:
  - the U+00A0 branch of `canonical_text`;
  - `⁺` and `⁻` in `normalize_name`'s trailing-charge class.

  NFKC runs first and maps U+00A0 to a space, `⁺` to `+`, and `⁻` to U+2212, which the dash
  map then turns into `-`. No input can reach either line, which is why their mutants survive.
  This is harmless and not a defect of the fix. The evaluator was left alone as instructed,
  and the copies keep the lines so they stay verbatim.
- **F2. The proof scripts read committed inputs, not `C:/t/cohort/` or `C:/t/eval-fulltext/`.**
  `orch754a_ensemble_merge.py` takes no path argument. It reads `evidence/orch754a_inputs/` and
  `evidence/orch752_chlorophyll_payloads/`, and asserts each paper's sha256 against the
  `paper_sha256` in its draws' `index.json`. The dispatch named the `C:/t` directories, but the
  committed, hash-checked copies are what produced 29 / 4/5 / 9/9 at base and at tip.
- **F3. The mutation check took three runs.**
  - `/08` crashed: one mutant wrote a NUL byte into the module source (`SyntaxError`).
  - `/09` ran clean but reported two survivors.
  - I first took the survivors for a gap in the probe set. Checking NFKC showed they are
    equivalent mutants (F1), so `/10` checks that fact explicitly instead of adding probes
    that could never kill them.

  `/08` and `/09` are committed as the record. Their exit 1 is a script result, not a test
  failure of the change.
- **F4. Process-hygiene slip, contained.** One Bash call I made by mistake carried an empty
  `python -` heredoc, against the dispatch rule, and it blocked. I stopped it with TaskStop and
  confirmed it left no process. Two `.venv` `python.exe -` processes that are still running
  belong to another session, not this one: parent `bash.exe` 194096, created 2026-10-19
  13:59, running a `C:/t/bt-orch754r0/probe` command. I left them alone. No test or measurement
  ran in that call; every gate above ran through the wrapper.

## 6. Not settled

1. **Two copies must now be kept in step.** The equality test is the mechanism: any change to
   one normalizer fails it until the other is changed on purpose. The options, if the product
   owner wants a different trade:
   - (a) keep the copies, the current state. Cost: a deliberate two-file edit per rule change;
   - (b) make the evaluator import production's `name_keys`. Cost: inverts the stated rule. A
     production normalization bug would again be invisible to the measurement;
   - (c) put a neutral shared module outside both packages. Cost: the same blindness as (b),
     because both sides would still share one normalizer.

   (a) is the only option that honours the reason the layering test exists.
2. **TEST_MATRIX delta, for the orchestrator to apply** (not edited here):
   - the ORCH-754a focused selection grows by `tests/test_name_keys_match_evaluation_matcher.py`
     (3 tests), from 122 to **125**;
   - `tests/test_eval_v2_*.py` goes from 1 failed, 88 passed, 1 xfailed to **89 passed,
     1 xfailed, 0 failed**. The layering failure, the only failure there at `adc49c40`, is gone;
   - SMOKE stays **508**.
3. **The unreachable lines (F1) stay in the evaluator.** Removing them is the evaluator's
   change to make, and it would need the matching edit in `name_keys.py`.
