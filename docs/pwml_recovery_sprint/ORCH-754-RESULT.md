# ORCH-754 — duplicate merge, protein interactions, prompt v03, then main

**Status:** DONE. All four lanes reviewed and merged into `sprint/pwml-recovery`, integration gates green apart from pre-existing failures, `main` fast-forwarded and pushed (§ 4). Orchestrator record; each lane writes its own `ORCH-754{a,b,c,d}-RESULT.md`.
Decisions: **D-104** (product owner, 2026-10-19, in session). Base `f9892cdf`.

## 1. What the product owner asked for

After a machine crash, the product owner asked whether the pipeline is ready for running papers and
comparing the text output by hand. On the answer, they ruled (D-104):

- raise the extraction token budget to 64,000;
- commit everything, update `main` and push;
- merge the duplicate reactions the 3 draws produce, with the paper as the dictionary and database IDs
  as the merge key (long form when there is no ID);
- export protein interactions to PathWhiz where PathWhiz allows it, with the text output as the exact
  record;
- promote the measured prompt;
- run the tests, but not the pipeline.

## 2. Findings made while orchestrating

1. **The production prompt was not the measured prompt.** Production `pwml_system.txt` matched variant
   `v02_working_tree_resolution` exactly (0 differing lines). Every scored draw since ORCH-747 round 2
   used `r2_01_control_v03`, which adds one 18-line block (`ONE CLAIM, ONE RESOLUTION`). Paper runs on
   v02 would have used a prompt measured only in round 1, with one draw per paper. Method: `diff`
   of the CR-stripped production file against every file in `docs/evaluation_v2/prompt_variants*/`.
   → D-104 § 5, lane `c`.
2. **The extraction budget in `.env` was 16,000**, the value ORCH-747 § 6.2 showed fails on this model;
   every measurement ran at 64,000. Fixed in `.env` (untracked), D-104 § 1.
3. **`ORCH-753` was already taken**, by a gold-cohort selection the crash interrupted (12 source manifests
   in `docs/evaluation_v2/gold_cohort/`, no RESULT doc, no ledger row). This task is therefore ORCH-754.
   The code commit `70d05e03` was written before this was found, and its message names ORCH-753 as the
   reviewer; that is wrong and is corrected in the ledger, not by rewriting a pushed commit.
4. **The commit swept in the product owner's own edit to the protected `streamlit_app.py`**, which D-103
   § 2 had kept uncommitted: the UI's Stage-1/Stage-2 token boxes now default from `.env`
   (`_bounded_env_int`). The product owner had asked for everything to be committed; D-104 § 6 records
   it as superseding D-103 § 2. It matters for paper runs: the UI now picks up the 64,000 budget.
5. **`PAPER_FINDINGS.md` still claimed pooling reaches recall 1.000**, which ORCH-751 disproved
   (enterobactin: all three draws missed the same reaction). Amended in place on 2026-10-19; the
   original text is kept and marked.
6. **PathWhiz validates the interaction type.** `smpdb/app/models/data/interactions/interaction.rb:13`
   defines `INTERACTION_TYPES = %w(Activation Inhibition)` and `:32` validates inclusion. The importer
   saves with validations (`lib/pwml_parser.rb`, `.save` at `:768`, `:889`). Endpoints are polymorphic
   (`interaction_element.rb`: Compound, ElementCollection, NucleicAcid, ProteinComplex, Bound) and have no
   bare-protein type. So protein endpoints can export through a ProteinComplex wrapper, but `binds`,
   `modifies` and the like cannot export at all. → D-104 § 4, lane `b`.
7. **Run scratch was never ignored.** `runs_validation/`, `runs_smoke/` and `runs_verify/` (~690 MB,
   45 MB cache snapshots each) sit untracked in the primary checkout. They were not committed. Adding
   them to `.gitignore` is open (§ 5).

## 3. Lanes

| lane | scope | worktree / branch | state |
|---|---|---|---|
| `754a` | paper-as-dictionary duplicate merge, database-ID merge before the freeze, support counts | `orch754a` / `orch754/a-ensemble-merge` | dispatched |
| `754b` | protein interaction endpoints through the enzyme wrapper path; exact interaction record in text | `orch754b` / `orch754/b-interaction-wrappers` | dispatched |
| `754c` | production prompt v02 → `r2_01_control_v03` | `orch754c` / `orch754/c-prompt-v03` | dispatched |
| `754r0` | independent review of `70d05e03`, which had none | read-only | dispatched |

All builders were Opus 5.5 at extra-high effort, as the product owner asked. Each lane is reviewed
independently before merge.

### 3.1 `754c` review — APPROVE WITH NOTES

Production prompt equals the measured variant: the same git blob `4160a649…`. The new pin `27f5e873…`
is the hash the measured draws recorded, found on 42 of them. The base failure was reproduced
independently. SMOKE 508. Notes: stale lines in the lane's own RESULT and stale line-number comments
outside its boundary. None block the merge.

### 3.2 `754r0` review of `70d05e03` — **REJECT**, four blocking findings

The ORCH-747 graph and report work, the PWML interaction plumbing and the `streamlit_app.py` hunk
were found correct as far as checked. The four blocking findings, each with its routing:

| # | finding | routed to |
|---|---|---|
| 1 | **The ensemble merge builds chimeric reactions.** The ORCH-752 key ignores cofactors, state and enzymes. A probe merged NAD-IDH (IDH3) and NADP-IDH (IDH1) into one row carrying NAD+, NADP+, NADH and NADPH; IDH1 vanished with no review flag. This is live whenever draws > 1, which `.env` now sets | `754a`: merge only *compatible* rows; a cofactor, enzyme or state conflict never merges. **Withdraws** the orchestrator's brief instruction 7 ("cofactor-blind key stays"). That instruction was the orchestrator's, not the product owner's |
| 2 | **The live curator prompt lets an LLM add interactions with no code gate.** The prompt demands a verbatim quote, but the code never sends the curator the evidence text; adds at confidence 0.75 or above are accepted. No decision authorized it | **Product owner, 2026-10-19: revert.** `754c` restores the file to `c9265fb8` |
| 3 | **Two pinned IR digests fail at `70d05e03`.** The delta was documented but the re-pin never done (rule 4) | `754b`: a deliberate re-pin with the cumulative documented delta |
| 4 | **24 tests fail in the primary checkout** (`test_early_failure_replay.py`). `llm/client.py:22` runs `load_dotenv(override=True)`, so `.env`'s `T2PW_EXTRACTION_DRAWS=3` beats the code default, and even a shell setting. SMOKE did not catch it | `754a`: tests pin N=1 unless they opt in; affected chunks re-run with draws = 3 |

Two non-blocking findings were also routed to `754b`:
- `competes_with` was exported as a directed Inhibition, i.e. a symmetric relation given a direction after the freeze. It will be withheld from the PWML.
- The withheld-interaction record was written only on the CLI path. It must also reach the app path's text output.

**Process incident.** The reviewer left two idle `python -` processes (Windows PIDs 193876, 190584,
created 13:59). The permission check refused its `taskkill`, so the orchestrator did not kill them
on its behalf. Surfaced to the product owner.

### 3.3 `754b` review — **REJECT**, repair round 1

The biology gates hold. The re-pin is valid: R3 was reproduced exactly, and every moved pointer sits under interactions, report warnings or counts. The match to PathBank `PW687775` interaction 15843 holds.

| # | finding | resolution |
|---|---|---|
| 1 | The text can say "Inhibition, drawn" while the IR withholds the row. The text resolves endpoints by exact or lowercased name; the IR uses synonyms with protein first. On committed payloads, 2/336 rows disagree (PMC12452463 `entC`/`entF`) | `754b`: the text verdict comes from the IR's own resolution |
| 2 | `draft_graph.py` was edited outside the boundary | **Granted by the orchestrator**: the reviewer judged it necessary and minimal, with nodes and edges unchanged |
| 3 | The new lines changed live LLM input: `reaction_summary.txt` feeds the curator and the enrichment agent; +32%, instruction-like wording | **Product owner: human-facing outputs only**, D-104 AMENDMENT 2. Applied to `754a`'s support marks too |
| 4 | `ir.py:138` message is untrue for rows Stage 6 refused | `754b` |

Scratch worktrees left by `754b` (`C:/t/bt-orch754b/base-wt`) and by its reviewer (`C:/t/bt-orch754b-rev/{base,tip}-wt`) are registered with git. Removing them was refused to the agents; it is left to the product owner.

### 3.4 `754b` review rounds 2-3

Round 2 **REJECT** (`9da8358e`): a Stage-6 summary counter reached the interactive curator's Mapping Misses (24/95 replay legs), and the `.txt` printed "drawn" when no PWML was written. Round 3 **APPROVE WITH NOTES** (`219d0003`), judged against the fixed severity standard. Mapping Misses differ on 0/95 legs; base and tip LLM-bound text are identical on the same post-Stage-6 payload; the moved GOLDEN digest was reproduced; the no-PWML path was checked on a committed leg; the sign-table move was byte-identical. Notes: the `ambiguous_protein_endpoint` reason hides a missing identity in the human record, and the wrapped-leg count (55 vs 56) is not reconciled. **Merged at `c711d859`.**

### 3.5 `754a` reviews, and the merge

Review 1 **REJECT** (`989dee54`): case-folded abbreviations (`ACP` renamed acetyl phosphate), skipped extra-enzyme quote check, cofactor role grouped by chemical family. Product owner: D-104 AMENDMENTS 4 (typo narrowed and flagged; rule 3 at Stage 6 only) and 5 (one keyword argument in the protected file). Review 2 **REJECT** (`75ca32f3`): entry-level quotes lost, enzyme typo snaps unflagged. Review 3 **APPROVE WITH NOTES** (`9ea36a6b`). **Merged at `adc49c40`.** Chlorophyll pool 41 → 29; enterobactin 4/5 and menaquinone 9/9 unchanged; N=1 identical to base; `streamlit_app.py` +1 line.

### 3.6 `754d` — evaluation layering

Dispatched after integration found `test_no_production_module_imports_evaluation_v2` red (since `70d05e03`). Verbatim copies of the matcher functions in `pipeline/name_keys.py`, with an equality test. Review **APPROVE WITH NOTES**: the copies were found AST-identical, and a 92-output base-vs-tip A/B gave identical hashes. Note: the equality test is weaker than an AST identity check. **Merged at `ff3457fa`.**

## 4. Gates

| after | run | report | result |
|---|---|---|---|
| merge of `754c` (`ca852bb6`) | SMOKE, primary checkout (`.env` present, draws = 3) | `evidence/g11/ORCH-754/03-smoke-after-c.json`, pin `pin/ORCH-754/03-…`, log `evidence/orch754/03-smoke-after-c.log` | **508 passed** in 42.69 s, 0 survivors |
| merge of `754b` (`c711d859`) | SMOKE, primary checkout | `evidence/g11/ORCH-754/04-smoke-after-b.json`, pin `pin/ORCH-754/04-…`, log `evidence/orch754/04-smoke-after-b.log` | **508 passed** in 70.30 s, 0 survivors |

| merge of `754a` (`adc49c40`), all lanes in | union of every lane's focused, affected and export-chunk files plus curator, prompt-proof, reaction-evidence and `test_early_failure_replay` (82 files) | `g11/ORCH-754/05`, log `orch754/05-…` | 1931 passed, **8 failed**: 4 pre-existing at base (c075, c100 5b, rag_seed conform, plus the coverage-tripwire pair counted as 2), 3 primary-only environmental, see below |
| | SMOKE | `06` | **508 passed** |
| | AppTest files, each alone: reaction_evidence_view, app_hook, batch_preflight, c055, c052 | `07`-`10`, `12` | 11, 1, 37, 14, 13 passed |
| | `test_streamlit_quarantine_boundary` alone | `11` | 22 passed, 1 failed: primary-only environmental |
| | Chunk D (`chunk_d_gate.py`, `T2PW_OFFLINE_CURATOR=1`) | `13` outer + inner | core 159/160 and node15 failed; both are the primary-only environmental nodes |
| | env-matched A/B of the 4 primary-only nodes: base `f9892cdf` vs tip, each carrying the primary's `.env` and uncommitted caches | `46`-`53` | **all 4 fail at base and tip with identical messages**: the DB is reachable in the primary checkout, which these tests assume it is not. Not regressions |
| | `tests/test_eval_v2_*.py` | `54`; base `55` | 88 passed, **1 failed: `test_no_production_module_imports_evaluation_v2`**. Red since `70d05e03` (fails at base, `55`), extended by `754a`. **A real defect, found only here: no lane ran these files.** → `ORCH-754d` |

`g11_evidence.py check --task ORCH-754`: 53 artifacts, 0 non-compliant. The 32 without a pin are Chunk D's inner per-node runs.

| after | run | report | result |
|---|---|---|---|
| merge of `754d` (`ff3457fa`), final tip | SMOKE | `56` | **508 passed** |
| | all `test_eval_v2_*` + ensemble, dictionary, name-key and early-failure-replay files | `57` | **239 passed, 1 xfailed, 0 failed** |

**Not re-run after `754d`:** the 82-file union and Chunk D. `754d` touches only name-key imports, and its reviewer proved byte-identical behaviour across 92 outputs, so the union and Chunk D results on `adc49c40` stand.

Reports `01` and `02` were reserved and never written: `01` waited on the heavy lock, which `754b`'s SMOKE held, and `02` failed on a missing log directory. Their placeholders sit in the git-ignored `.staging/`; `g11_evidence.py check --task ORCH-754` exits 0.

## 4a. `main`

`main` was fast-forwarded to the `sprint/pwml-recovery` tip and both were pushed to `origin` (D-104 § 7). **The working checkout was NOT switched to `main`.** A parallel session (ORCH-760, pathway mining) became active during integration and commits ledger rows to `sprint/pwml-recovery` from this checkout (`c60c1755`). Switching the checkout would send its next commit to `main` without it noticing. The switch waits for the product owner.

## 5. Open for the product owner

1. **Ignore the run scratch.** Add `runs_*/`, `out/`, `outputs/`, `tmp/` to `.gitignore`. That costs nothing
   and stops a future "commit everything" sweeping in 690 MB. They might also be wanted as committed
   evidence, so this is your call.
2. **`ORCH-753`, the interrupted gold cohort.** Resume it, or drop it. Building gold for the cohort is
   what ORCH-751 § 5 called "the obvious next investment".
3. **Switch the checkout to `main`** once ORCH-760 is not committing from the primary checkout (§ 4a).
4. **The first live 3-draw run is the first end-to-end test** of the ensemble path. Rule 3, the database-ID merge, has only been exercised on synthetic payloads, because it needs a real Stage 6.
5. **ORCH-747's interaction lines in the LLM-fed summary**, which predate D-104 AMENDMENT 2, still reach the curator and the enrichment agent (AMENDMENT 2 § 3).
6. **Strengthen the normalizer drift alarm** to an AST identity check (`754d` review note).
7. **Four tests fail only in the primary checkout** because the database is reachable there (`TEST_MATRIX.md` § ORCH-754). Decide whether those tests should skip when a DB is configured.
8. **Cleanup the agents could not do** (refused by the permission system): scratch worktrees under `C:/t/bt-orch754b/`, `C:/t/bt-orch754b-rev/`, `C:/t/bt-orch754a-rev/`, and **`C:/t/bt-orch754/env-base` and `env-tip`, which hold copies of `.env`**; plus two idle `python -` processes (PIDs 193876, 190584) left by the `754r0` reviewer.
