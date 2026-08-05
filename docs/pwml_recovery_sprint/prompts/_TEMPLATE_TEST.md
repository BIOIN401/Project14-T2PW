# Milestone test prompt template — `T-1xx`

Run by the test-runner agent. It reports numbers and fixes nothing.

---

```
[S1] [S5] [S7] [S8]

ROLE
  Milestone validation. You run suites and benchmarks and report results.
  You do NOT fix code, edit tests, or propose patches.

PROCESS LIFECYCLE -- HARD RULE, read [S8] in full before running anything
  Every command in this prompt runs through the bounded foreground wrapper whose
  path INIT-001 recorded. No detached processes, no nohup, no untracked
  background jobs, no Start-Process without bounded waiting.
  ONE heavy job at a time. Never pytest -n auto. Never concurrent benchmarks.
  Cleanup targets ONLY PIDs this job created. Global kills are forbidden.
  A job is NOT complete when pytest prints its summary -- it is complete when the
  root exited, every owned descendant exited, cleanup verification passed, and
  the cleanup report was recorded.
  If ANY owned process survives cleanup: this is an INFRASTRUCTURE FAILURE.
  Stop dispatch, report PID + command line + start time + memory. Do not report
  the test as passed.

MILESTONE        <M1..M5>
INTEGRATION SHA  <sha>
LEGS             <explicit list>
EXPECTED CLOCK   <hours>
BASELINE         docs/pwml_recovery_sprint/BASELINE.md

DO
  1. Run exactly the listed legs. Never a full benchmark unless this is M5.
  2. Score:
       .venv/Scripts/python.exe scripts/bench_acceptance.py \
         --run-dir <run> --json <out>
  3. Diff EVERY number against BASELINE.md.
  4. For every leg that changed, classify:
       product_contract_violation | gold_data_defect | policy_disagreement
     citing the gold relevance_note / export_rationale. A benchmark failure does
     NOT by itself justify a code change.
  5. NONDETERMINISM. Before calling any single-leg change a regression, re-run
     that leg. Identical legs give materially different Stage-1 draws at
     temperature 0 in this repository. Report the variance you observed.

MILESTONE ACCEPTANCE
  M1  PMC12452463/strict and PMC12096016/strict pass the quarantine boundary.
      PMC12452463 -> review_required, NOT strict success  [TRAP-1].
  M2  No leg reports "produced nothing". identical_empty_response recorded
      wherever two draws share a response hash. budget_exhausted distinct from
      scientific failure.
  M3  Reload final_mapped.json and re-export with ALL resolvers disabled ->
      identical canonical_graph_sha256, AND biological equivalence proven by
      PARSING AND NORMALIZING the JSON, PWML and SBML graphs and comparing:
      reactions, reactants, products, direction, reversibility, stoichiometry,
      enzymes, modifiers, transports and cargo, entities and identifiers,
      complexes and components, locations, process->entity references, organism.
      Comparing one JSON hash to itself proves nothing and is NOT acceptable
      evidence.
  M4  Every RAG round re-entered normalization, mapping, gates, persistence and
      classification. A round that skipped any of the five is a FAIL regardless
      of what it retrieved.
  M5  Full acceptance matrix vs BASELINE.md. Every remaining failure explained
      and classified.

REPORT
  ## LEGS RUN | WALL CLOCK
  ## ACCEPTANCE MATRIX   metric | baseline | now | delta | verdict
  ## CHANGED LEGS        leg | before | after | classification | gold citation
  ## NONDETERMINISM      legs re-run, variance observed
  ## REMAINING FAILURES  leg | class | owner | needs code? yes/no + why
  ## ARTIFACTS WRITTEN   paths committed under evidence/
  ## CLEANUP REPORT      one row per job:
       root PID/process group | timeout | exit reason | exit code |
       descendants observed | descendants terminated | final surviving count |
       cleanup success/failure
       final surviving count MUST be 0 for every job, or this milestone is an
       infrastructure failure regardless of the test numbers.
```

---

## Scheduling

Milestone benchmarks are launched by the Lead Orchestrator only. An implementer or
reviewer that wants benchmark evidence requests it; it does not run one.

Measured leg times on `ORIGIN_SHA`: 1308 s and 1511 s. A full pinned run is ~7 h across
20 legs. The runner's own `DEFAULT_DEADLINE_HOURS = 10.0` confirms overnight sizing.

**Never use a full benchmark as a per-patch merge gate.**
