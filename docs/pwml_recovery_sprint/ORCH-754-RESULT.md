# ORCH-754 — duplicate merge, protein interactions, prompt v03, then main

**Status:** IN PROGRESS. Orchestrator record; each lane writes its own `ORCH-754{a,b,c}-RESULT.md`.
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

## 4. Gates

| after | run | report | result |
|---|---|---|---|
| merge of `754c` (`ca852bb6`) | SMOKE, primary checkout (`.env` present, draws = 3) | `evidence/g11/ORCH-754/03-smoke-after-c.json`, pin `pin/ORCH-754/03-…`, log `evidence/orch754/03-smoke-after-c.log` | **508 passed** in 42.69 s, 0 survivors |

Reports `01` and `02` were reserved and never written: `01` waited on the heavy lock, which `754b`'s SMOKE held, and `02` failed on a missing log directory. Their placeholders sit in the git-ignored `.staging/`; `g11_evidence.py check --task ORCH-754` exits 0.

## 5. Open for the product owner

1. **Ignore the run scratch.** Add `runs_*/`, `out/`, `outputs/`, `tmp/` to `.gitignore`. That costs nothing
   and stops a future "commit everything" sweeping in 690 MB. They might also be wanted as committed
   evidence, so this is your call.
2. **`ORCH-753`, the interrupted gold cohort.** Resume it, or drop it. Building gold for the cohort is
   what ORCH-751 § 5 called "the obvious next investment".
