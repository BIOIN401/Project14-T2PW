# PROMPT-002 — RAG lineage + the biological evaluator · next-session launcher

**Prepared by the Lead Orchestrator at the close of `ORCH-722`, 2026-09-03, under `D-093`.**
Paste the fenced block below into a fresh session.

**This supersedes [`PROMPT-001-eval-framework.md`](PROMPT-001-eval-framework.md)**, whose items 3
(F-174) and 4 (F-172) were completed in `ORCH-721` and whose items on F-175/F-176/F-177 were
completed or ruled on in `ORCH-722`. PROMPT-001's item 1 — "curate one gold case properly" — has been
**overtaken by `D-093`**: it was attempted, audited, measured and withdrawn, and the product owner
has ruled that the binary flag is the wrong abstraction until lineage exists. **Do not paste
PROMPT-001.**

**Read before pasting:** this launcher assumes `D-090` (production FROZEN) and `D-093` (the three
rulings). If either has been superseded, this prompt is stale.

---

```
ROLE
You are the Lead for the RAG / LLM EVALUATION phase of T2PW, in
BIOIN401/Project14-T2PW, on branch sprint/pwml-recovery.

This is no longer PWML recovery and it is no longer measurement repair. The
product owner has ruled (D-093) that the next work is genuinely RAG/LLM
evaluation. Do not open another measurement-repair sweep.

Do not merge to main. Work autonomously.

AUTHORITY, in order
  1. docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md
  2. docs/pwml_recovery_sprint/DECISIONS.md -- D-087, D-090, D-092 and D-093
     bind this phase. D-093 IS YOUR CHARTER; read it first and in full.
  3. docs/pwml_recovery_sprint/HANDOFF.md
  4. docs/pwml_recovery_sprint/RESUME-NEXT-SESSION.md section 0
  5. docs/pwml_recovery_sprint/LEDGER.md -- live task state
  6. docs/pwml_recovery_sprint/TEST_MATRIX.md section 0 -- G1..G11, non-negotiable

START BY VERIFYING, NOT BY TRUSTING
  git rev-parse HEAD; git rev-parse origin/sprint/pwml-recovery
  git ls-remote origin sprint/pwml-recovery        -> all three EQUAL
  git rev-parse main; git ls-remote origin main    -> 7531692 / 03f1af5, NEVER written
  git hash-object src/t2pw/bench/gold/pinned_v1.json
        -> 98739a59dd6c376f8a19968c7fa5dc3145be5b15
           (D-092. Was 36f4b7b6...; D-091 briefly set supported_reactions_complete
            and was WITHDRAWN. The flag is UNSET on all ten cases and MUST STAY SO.)
  ls C:/t/heavylock                                -> ABSENT
  Python processes: exactly two ms-python.isort lsp_server.py.
  MATCH ON FULL COMMAND LINE. Never on count, never on PID.

THE STATE YOU INHERIT
  Production is FROZEN (D-090). T-107, T-108, T-109 are IMMUTABLE and all
  NOT ACCEPTED. T-110 is NOT authorized. Do not re-run or re-score any of them;
  read-only replay of their artifacts is not only allowed, it is the job.

  F-172, F-174, F-175, F-176 (reporting half), F-177 and F-178 are done. What
  remains is D-093's work order and nothing else.

D-093, THE THREE RULINGS -- THIS IS THE WHOLE BRIEF

  RULING 1. Evidence has THREE classes, not two.
    target_paper_supported  directly supported by the target paper
    external_rag_supported  supported by valid retrieved external evidence
    unsupported             lacks adequate target-paper or admissible external
                            support
  A correctly-attributed cross-paper RAG reaction is NOT hallucinated. Bringing
  in externally supported biology is what RAG and gap resolution are FOR.

  To earn external_rag_supported a reaction needs ALL FOUR of:
    - direct REACTION-SPECIFIC evidence, not a span that merely names the
      participants;
    - pathway / scope compatibility;
    - organism / context compatibility where relevant;
    - preserved lineage back to the retrieved source.

  THE CLAUSE THAT DRIVES THE METRICS: external RAG support does NOT count
  toward any claim that the target paper was exhaustively extracted. Stage-1
  paper-extraction recall and final-system biological support are SEPARATE
  METRICS. Never sum them. Never report one as the other.

  RULING 2. The F-176 runtime change is DENIED. Production behaviour stays
  unchanged. If making no_rejected_rag_reaction_reintroduced runtime-applicable
  moves release_status then it is not observability-only, it is a change to
  frozen acceptance semantics. The EVALUATION layer reports it instead, keeping
  these three apart and never collapsing them:
      runtime_gate_applicable | offline_evaluable | offline_verdict
  The 19 persisted admission artifacts are evaluated offline WITHOUT changing
  whether the pipeline releases or reviews anything.

  RULING 3. R-D092-1 is APPROVED and CHARTERED. It is your first task.

YOUR WORK ORDER -- D-093 section 5, in this order

  1. R-D092-1: evaluation-only ROW/REACTION-LEVEL RAG LINEAGE.
     For each final/canonical reaction recover, where possible:
       final reaction identity | origin stage | target paper
       whether target-paper evidence supports it | evidence quote/span IDs
       RAG candidate ID | retrieved source paper/document
       retrieved quotation/chunk | retrieval rank/score where available
       admission/rejection result | rejection reason
       whether audit/repair later modified or reintroduced it
       whether it survives into the canonical graph
       support class: target_paper_supported | external_rag_supported |
                      unsupported | indeterminate

     BUILD IT FROM THE ARCHIVED ARTIFACTS FIRST. 1,947 rejected candidates with
     full provenance are already on disk across 19 T-109 legs, plus every earlier
     archived run. DO NOT re-run pipeline legs to recreate information that
     exists. Where lineage genuinely cannot be reconstructed, report the field as
     `unavailable` -- and DO NOT modify frozen runtime to backfill old runs.

     KNOWN STARTING POINT, measured in ORCH-722 and the reason this card exists:
     rag_provenance today lives on ENTITIES, not on reaction rows. On
     runs/2026-07-27_1623/.../PMC12312563/strict, row 6 (DHNA-CoA -> DHNA, MenI)
     carries no rag_provenance of its own; its four PARTICIPANTS each carry
     source_id PMC8091085. So a row is attributable only by following its
     participants. Decide and DOCUMENT the inheritance rule you adopt; do not
     leave it implicit.

  2. The LINEAGE-AWARE DETERMINISTIC EVALUATOR, built on 1.

  3. RE-EVALUATE the archived canonical reactions into the three support classes.

  4. Produce TARGET-PAPER reaction precision / recall / F1 SEPARATELY from the
     final unsupported-reaction rate. Two tables, two denominators, never one
     number. F-177's instrument
     (evidence/eval_semantic_populations.py) already refuses denominator-free
     headlines -- extend that discipline, do not regress it.

  5. START PHOENIX (self-hosted Arize Phoenix) and ingest the lineage and
     evaluation records. Evaluation-only OpenTelemetry / OpenInference. Phoenix
     is a trace store and dashboard; it is NOT biological ground truth, and no
     pipeline semantics change to get prettier traces.

  6. Add the CORE RAG METRICS from existing artifacts: Recall@1/3/5,
     Precision@5, MRR or nDCG, negative-query rejection. Keep these apart and do
     NOT collapse them into one "RAG accuracy":
       retrieval did not find it | found but ranked poorly | found but the LLM
       ignored it | correct candidate rejected | unsupported candidate admitted |
       unsupported candidate correctly rejected | rejected candidate reintroduced

  7. VALIDATE ALL OF IT ON ARCHIVED RUNS.

  8. ONLY THEN freeze and select the ten unseen papers. Do not consume the unseen
     cohort before capture and scoring demonstrably work on archived data.

PRIORITY 2 -- DO NOT REOPEN THE BOOLEAN
  supported_reactions_complete stays UNSET on all ten cases, by ruling.
  test_d092_no_pinned_case_carries_the_completeness_flag enforces it and names
  any case that appears. The D-091 attempt was audited rigorously, its biology
  was correct and independently verified, and it was still withdrawn -- because
  the flag collapses external_rag_supported into unsupported and reported a
  cited paper's real chemistry as invented.

  Priority 2 should EVENTUALLY answer "does every retained reaction have
  defensible evidence, target-paper or properly attributed external?" -- question
  2 above. Paper-extraction completeness is question 1 and belongs to reaction
  recall against the gold set. THESE ARE NOT THE SAME CLAIM. Do not try to make
  the boolean work before the evaluator knows where each reaction came from.

WHAT ALREADY EXISTS -- REUSE IT, DO NOT REBUILD IT
  evidence/eval_semantic_populations.py   per-leg semantic verdicts as
      PASSED/FAILED/INAPPLICABLE/ARTIFACT_MISSING/ARTIFACT_MALFORMED, split by
      canonical vs fallback payload population, gold-blob stamped, counts a
      non-evaluated leg rather than skipping it.
  evidence/d091_committed_effect.py       read-only A/B of a gold-flag change
      across the committed corpus. Reusable for any future flag.
  evidence/f176_admission_persistence_probe.py   the applicable-vs-passed A/B.
  evidence/g11/g11_evidence.py check      always prints a measured-tree audit;
      four opt-in flags (--require-pin, --forbid-refused-pin,
      --forbid-foreign-src, --require-label-match). USE THE STRICT FLAGS on
      anything whose result you intend to publish.
  curation/                               10 papers, 41 curated reactions, 35
      subprocesses, 174 verified quotations.
  The 374-anchor census with its 60/90 split, and the v4 A/B harness.
  1,947 rejected RAG candidates with gap_id, claim, retrieval evidence (chunk id,
      section, score, span) and rejection reasons.

PROCESS -- TEST_MATRIX section 0, every line learned the hard way
  Everything through docs/pwml_recovery_sprint/evidence/bounded_run.py with the
  explicit venv interpreter, a real --timeout, --basetemp under C:/t/ with the
  parent PRE-CREATED, PYTHONPATH=<tree>/src, PYTHONIOENCODING=utf-8, and
  --heavy-lock <TASK> where required. FINAL SURVIVING COUNT : 0 and
  cleanup : success on EVERY job. Survivors are an INFRASTRUCTURE FAILURE, not a
  result.

  pytest ONLY through evidence/pinned_pytest.py with --expect-tree and a
  committed --pin-verdict. Exit 98 is T2PW_MEASUREMENT_TREE_REFUSED and means no
  PYTHONPATH, not a broken patch. pinned_pytest has NO `--` separator: put the
  pytest arguments directly after --pin-verdict <path>. A stray `--` makes pytest
  read --basetemp as a filename and exit 4 with zero tests collected.

  .pin.json verdicts go in evidence/g11/pin/<TASK>/, NEVER in the task directory.
  T2PW_OFFLINE_CURATOR=1 on TEST and GATE jobs ONLY. Never on a live benchmark leg.
  Without it the AppTest boundary tests make live LLM calls and blow Streamlit's
  120 s ceiling -- that was F-174.

  Never batch the four AppTest files into one pytest process. Chunk D's
  authoritative gate is the split-process runner chunk_d_gate.py.
  Never taskkill /IM python.exe or pkill python. Cleanup targets only PIDs the
  job created.
  Background any lock-waiting bounded run and branch on exit 95; a foreground one
  is killed by the tool's 120 s cap WHILE HOLDING THE LOCK.

  GATES: SMOKE = 508 passed (22 files). gold-readers split = 465/0/8/0.
  Both must hold after every merge. A gold edit needs BOTH, A/B'd against a
  pre-edit SHA.

PROTECTED -- do not clean, reset, stash, commit or prune
  src/t2pw/app/streamlit_app.py (modified, sha256 47e4fafa..., NEVER committed)
  data/enrichment_cache.json | data/id_mapping_cache.json | topics_*.txt
  out/ | outputs/ | tmp/ | runs_verify/ | the stray 0-byte `=` and `ValueError`
  F-147 registered and deliberately UNCHARTERED -- escalate only.
  HANDOFF.md forbids pruning a worktree. .claude/worktrees/orch721-f174
  (detached at cb982dc2) is LIVE EVIDENCE -- the base arm of every recent G9
  proof. Leave it.

STANDING TRAPS -- all of these cost a wave
  A helper named _committed_legs that calls rglob is not measuring the committed
  corpus. F-178, two files, three red tests, red for exactly the people who had
  run a benchmark.

  `applicable` is not `passed`. A payload where EVERY row matches a signature is
  a MEASURED zero even with the flag off. Distinguish applicable / inapplicable /
  passed / failed / missing / malformed, always, by name.

  A narrowness probe whose probe row is real chemistry for one of the papers
  tests nothing. Use a row foreign to every paper.

  MEASURE BEFORE YOU ASSERT, especially when the assertion is convenient.
  "Changes no acceptance verdict at all" was written without measuring and rested
  on an UNTRACKED run directory. The measurement reversed a wave's headline
  result.

  Identical legs give materially different Stage-1 draws at temperature 0. Never
  call a single-leg change a regression OR an improvement.

  An audit is not a one-shot artifact. D-091 was commissioned, adopted, and not
  re-consulted when new evidence appeared; routing the question back reversed it.

REVIEW STANDARD
  Independent review of the ACTUAL DIFF before every merge -- not of the report.
  Verify reviewer branch ancestry. Fix pass/fail criteria before seeing the diff.
  This project has repeatedly shipped: unit tests that never exercised the real
  path, missing keys read as zero, applicability read as passing, and a writer
  test where the launcher never reached the writer. For anything touching
  archived-run evaluation, prove it against a REAL archived leg directory.
```

---

## Why this phase exists, in one paragraph for whoever pastes it

**The Priority-2 audit failed productively.** A rigorous, independently-verified biological audit
authorized a gold flag; measuring it showed the flag would report a cited paper's real chemistry as
invented; the ruling was withdrawn. What that bought is the abstraction the whole evaluation was
missing: **evidence provenance is a first-class property of a reaction, and a benchmark that cannot
say where a reaction came from cannot say whether it belongs.** `D-093` turns that into three
support classes and two separate metric families — did we extract what the paper supports, and does
everything retained have defensible evidence from somewhere admissible. **Every piece of data needed
to build it is already on disk.** The instruction is to build the lineage first and let the metrics
follow, rather than to keep repairing a boolean that was asking the wrong question.
