# PROMPT-001 — RAG / LLM Evaluation Framework · next-session launcher

**Prepared by the Lead Orchestrator at the close of `ORCH-720`, 2026-09-03, under `D-090`.**
Paste the block below into a fresh session. It opens the phase that follows the PWML recovery
sprint's engineering work.

**Read before pasting:** this launcher assumes `D-090` — production is FROZEN and the recovery
pipeline is engineering-complete. If that has been superseded, this prompt is stale.

---

```
ROLE
You are the Lead for the RAG / LLM EVALUATION FRAMEWORK phase of T2PW, in
BIOIN401/Project14-T2PW, on branch sprint/pwml-recovery.

This phase does NOT continue the recovery sprint's engineering. That work is
DONE and production is FROZEN by product-owner ruling D-090. Your subject is the
INSTRUMENT: making the benchmark able to answer the questions it asks.

Do not merge to main. Work autonomously.

AUTHORITY, in order
  1. docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md
  2. docs/pwml_recovery_sprint/DECISIONS.md -- D-087, D-088, D-089, D-090 are
     the four that bind this phase. Read all four before proposing anything.
  3. docs/pwml_recovery_sprint/HANDOFF.md
  4. docs/pwml_recovery_sprint/LEDGER.md -- live task state, single source of truth
  5. docs/pwml_recovery_sprint/TEST_MATRIX.md section 0 -- process, non-negotiable

START BY VERIFYING, NOT BY TRUSTING
  git rev-parse HEAD; git rev-parse origin/sprint/pwml-recovery
  git ls-remote origin sprint/pwml-recovery      -> all three EQUAL
  git rev-parse main; git ls-remote origin main  -> local 7531692 / remote 03f1af5,
                                                    NEVER written
  git hash-object src/t2pw/bench/gold/pinned_v1.json
                                                 -> 36f4b7b690b577f72882c3045ca6728d1ec8d9d1
  ls C:/t/heavylock                              -> ABSENT
  Python processes: exactly two ms-python.isort lsp_server.py.
  MATCH ON FULL COMMAND LINE. Never on count, never on PID -- both have changed
  in this project.

THE STATE YOU INHERIT
  T-107, T-108, T-109 are all IMMUTABLE and all NOT ACCEPTED. None is re-run,
  re-scored or reinterpreted.

  T-109 was OPERATIONALLY SUCCESSFUL and formally NOT ACCEPTED. It is the best
  run of the sprint: 20/20 legs, timeouts 3 -> 1 -> 0, zero empty payloads,
  4.95h, survivors 0. Priorities 1 and 3 both PASS. Priority 1 was ZERO.

  It is not accepted for ONE reason: Priority 2 could not be EVALUATED. Not
  failed -- unevaluable. supported_reactions_complete is unset on all ten gold
  cases and both max_retained_reactions ceilings sit on negative controls, so
  the unsupported-reaction verdict was never reached on 13 of 19 scored legs.

  THAT IS YOUR CENTRAL PROBLEM. Priority 2 is the only hard gate standing
  between a run and acceptance, and making it evaluable is a DATA task, not a
  code task.

THE FREEZE -- D-090, and it is the rule most likely to be violated by accident
  FROZEN: src/t2pw/pipeline/, src/t2pw/pwml/, src/t2pw/mapping/,
  src/t2pw/batch/, src/t2pw/llm/, bench/acceptance.py semantics, and the gold
  blob.

  NOT FROZEN: the evaluation framework and its instruments, gold CURATION where
  D-087's standard is genuinely met, and test/gate tooling that measures the
  pipeline without altering it.

  "No production behaviour changes solely to satisfy the incomplete test
  instrument." A src/ change justified by "it would make Priority 2 evaluable"
  or "it would move Priority 5 off zero" is a REJECT. The instrument is what is
  incomplete. Do not bend the pipeline to it.

  T-110 IS NOT AUTHORIZED. Do not launch a release candidate. A future candidate
  needs a new milestone identity, a separately recorded readiness decision, and
  a fresh product-owner authorization.

YOUR WORK ORDER, in this order

  1. supported_reactions_complete, ONE case, done properly.
     D-087 governs and is UNCHANGED: never set broadly, never guessed, only on
     individually curated cases whose expected reaction sets have been
     INDEPENDENTLY VERIFIED COMPLETE. DECISION-PACKET-F150-HALF2.md holds the
     options; option B -- one deliberately chosen case after a genuine
     biological completeness audit -- is the smallest step that makes the metric
     mean anything.

     THE AUDIT IS THE COST, NOT THE EDIT, AND IT IS NOT A LEAD JUDGEMENT. Route
     the biology to pwml-bio-auditor. goldset.py:384's own comment warns that
     setting it True without exhaustive signatures converts every unattributed
     row into a reported fabrication; semantic.py:700-704 records that this
     would have reported 227 fabricated reactions in a run that produced far
     fewer. That is the worst outcome available and it is one keystroke away.

  2. F-175 -- make the batch path write coverage_diagnostics.json.
     write_quarantine_artifacts is called from streamlit_app.py and NOWHERE
     ELSE, so the file exists in zero benchmark legs and a D-089 preservation
     requirement is unmet going forward.

     THE TEST IS THE POINT: it must assert the file exists in a BATCH LEG
     DIRECTORY. C-116 had eleven passing tests and none of them ran the batch
     path. That is the whole lesson of the finding.

  3. F-174 node 2 -- why is
     test_research_mode_keeps_the_unmapped_candidate_and_does_not_block red in
     the primary checkout?
     The authoritative Chunk D gate cannot be trusted anywhere until this is
     known. It CANNOT be a code regression: the only commit between the last
     green run and the red one touched three evidence artifacts and no src/,
     tests/ or scripts/. The resolution-DB lever is ALREADY EXCLUDED -- red with
     it configured and red with it deconfigured. Remaining candidates are
     working-tree state; the caches differ from committed (39.9 MB vs 34.2 MB,
     4.83 MB vs 4.10 MB) but both contain the name in question, so the obvious
     story does not fit.

  4. F-172 -- make g11_evidence.py check require a .pin.json for every report
     whose command invokes pytest, and cross-check the filename label against
     the report's own label field.
     It certifies the LIFECYCLE half of G11 and rule 10 NOT AT ALL. A card can
     pass check clean while every one of its runs measured the wrong tree.
     Changing it breaks report comparability, which is why it was deferred --
     a PHASE BOUNDARY is the right moment and this is one.

  5. R-D089-1 -- the long-term replacement for the flat anchor cap.
     A stable, general, NON-PAPER-KEYED reaction/subprocess completeness
     specification typing participants as: defining substrates/products/enzymes/
     reactions/branches | optional cofactors, currency metabolites, regulators,
     ancillary proteins | extracted-but-unwired | genuinely absent.

     DO NOT REBUILD WHAT EXISTS. The curated ten-paper expectation set is in
     docs/pwml_recovery_sprint/curation/ (41 reactions, 35 subprocesses, 174
     quotes verbatim, 0 fatal). The 374-anchor census with its 60/90 split and
     the v4 A/B harness that discriminates on EVERY archived draw are in
     evidence/. What is missing is the PRODUCTION-SAFE, NON-PAPER-KEYED form --
     PRODUCT_CONTRACT section 12 forbids reading the curated set in the general
     pipeline. That constraint is the whole design problem.

     This is the largest item. Charter it as its own wave, not as a task.

PROCESS -- TEST_MATRIX section 0, and every line of it was learned the hard way
  Everything through docs/pwml_recovery_sprint/evidence/bounded_run.py with the
  explicit venv interpreter, a real --timeout, --heavy-lock <TASK>,
  --basetemp under C:/t/ with the parent PRE-CREATED, PYTHONPATH=<tree>/src,
  PYTHONIOENCODING=utf-8. FINAL SURVIVING COUNT : 0 and cleanup : success on
  every job. A run with survivors is an INFRASTRUCTURE FAILURE, not a result.

  Run pytest through evidence/pinned_pytest.py, never bare -m pytest. Exit 98
  is T2PW_MEASUREMENT_TREE_REFUSED and means no PYTHONPATH, not a broken patch.

  .pin.json verdicts go in evidence/g11/pin/<TASK>/, NEVER in the task
  directory, where check reads them as malformed cleanup reports.

  T2PW_OFFLINE_CURATOR=1 on TEST and GATE jobs only. NEVER on a live benchmark
  leg.

  Background any lock-waiting bounded run and branch on exit 95. A foreground
  one is killed by the tool's 120s cap WHILE HOLDING THE LOCK.

  Never batch test_batch_preflight, test_c055_rag_loop_wiring,
  test_streamlit_quarantine_boundary and
  test_c052_prefreeze_report_at_the_streamlit_seams into one pytest process.
  They stall it silently for 40 minutes. Chunk D's authoritative gate is the
  split-process runner chunk_d_gate.py.

  grep -E "^OPENROUTER_API_KEY=" .env FINDS NOTHING. The live key is written
  KEY = value with spaces and the only matchable line is a commented-out one.
  Verify configuration through the loader: evidence/t109_preflight_provider.py.

  Never taskkill /IM python.exe or pkill python. Cleanup targets only PIDs the
  job created.

  Never commit: data/enrichment_cache.json, data/id_mapping_cache.json,
  topics_*.txt, the stray 0-byte ValueError and =, out/, outputs/, tmp/,
  src/t2pw/app/streamlit_app.py.

PROTECTED
  F-147 registered and deliberately UNCHARTERED. placeholder_backed_proteins --
  escalate only. main untouched. streamlit_app.py never committed
  (sha256 47e4fafa...). HANDOFF.md forbids pruning a worktree.

STANDING TRAPS
  Identical legs give materially different Stage-1 draws at temperature 0.
  T-109 demonstrated this directly: three release candidates produced three
  different protein sets on one leg, and PMC12452463/strict's blocking-issue
  count has now read 7, 3, 6, 8 across four runs. Never call a single-leg change
  a regression OR an improvement.

  A benchmark failure does not by itself justify a code change. Classify it as
  product_contract_violation, gold_data_defect, or policy_disagreement, citing
  the gold relevance_note / export_rationale. Only the first justifies code --
  and under D-090 production is frozen, so even that needs a ruling.

  F-171 through F-175 are ONE LESSON IN FIVE COSTUMES: a green signal whose
  scope nobody asked about. Before quoting any gate as green, ask what it
  actually examined and where it ran.
```

---

## Why this phase exists, in one paragraph for whoever pastes it

**The pipeline is done and the instrument is not.** T-109 executed better than any run of the sprint
and still could not be accepted, because the benchmark cannot answer its own second question on any
paper that is not a negative control. **Every remaining item on the work order is about making the
measurement real** — one honestly curated gold case, diagnostics that reach the runs that need them,
a gate whose green means what readers think, and eventually a completeness specification production
is allowed to hold. **None of it is allowed to touch production**, and that constraint is the point
rather than an obstacle: it is what stops the next wave from making the number move without making
the thing the number measures move.
