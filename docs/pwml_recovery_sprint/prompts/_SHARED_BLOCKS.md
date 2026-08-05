# Shared prompt blocks

Referenced by ID from every task prompt. Written once so ~90 prompts can be generated
without drift. When a prompt says `[S1]`, paste this block's text.

---

## S1 — Environment contract

```
ENVIRONMENT
  Repo         : BIOIN401/Project14-T2PW
  Integration  : sprint/pwml-recovery
  Python       : .venv/Scripts/python.exe  (3.13.6, Windows)
  Authority    : docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md
  Your row     : docs/pwml_recovery_sprint/MASTER_PLAN.md § 9 branch register

TESTS
  ALWAYS pass --basetemp=<unique dir>. Without it 83 tests error with
  PermissionError and you will report a false regression.
  NEVER run the full suite unchunked (~16 GB).
  Commands: docs/pwml_recovery_sprint/TEST_MATRIX.md

DO NOT
  - modify any file outside your OWNED list. A diff touching an unowned file is
    an automatic reject, even if the change is correct.
  - run a full pinned benchmark (7 h). Milestone runs are the orchestrator's.
  - commit a cache modification. data/enrichment_cache.json is 39 MB and
    tracked; .git is already 158 MB.
  - read other branches' prompts.
```

## S2 — Test commands

See `TEST_MATRIX.md` § Commands. Smoke = 457 tests ~40 s. Chunk D = 177 tests ~222 s.

## S3 — Non-negotiable product invariants

```
You MUST NOT, under any circumstance:
  1. Weaken a biological gate to increase PWML production.
  2. Delete an incomplete-but-correct pathway instead of exporting it as
     review_required.
  3. Let an exporter add, remove, resolve or reinterpret biological content
     after the canonical graph is frozen.
  4. Accept an identifier because its FORMAT is valid.
  5. Invent biology to satisfy a completeness target.
  6. Make a pinned baseline test pass by reverting behaviour. If a pinned
     baseline must move, move it deliberately, document the exact delta, and
     say so in your report.
  7. Collapse `not_evaluated` into `false`.
  8. Treat a lookup or network failure as biological rejection.
If your change appears to require any of these, STOP and report instead.
```

## S4 — Stop conditions

```
STOP and report WITHOUT committing if:
  - The change requires editing a file you do not own.
  - A test outside your focused set fails and you did not expect it.
  - Your dependency's merged behaviour differs from what this prompt assumes.
  - You would have to violate S3.
  - The work exceeds ~400 changed lines. Report a proposed split instead.
```

## S5 — Evidence standard

```
Claims in your report must be evidenced:
  "test passes"     -> paste the pytest summary line.
  "behaviour X"     -> paste the command and its output.
  "artifact says Y" -> cite file path + line or key.
Never assert a runtime behaviour from a static code path alone.
Never claim a benchmark number whose source run is not committed.
```

## S6 — Implementer report format

```
## BRANCH
## BASE SHA
## FILES CHANGED   (path :: function, one per line)
## WHAT CHANGED    (<= 10 bullets)
## TESTS ADDED     (name :: exact failure it would catch)
## TEST OUTPUT     (pasted summaries)
## INVARIANTS      (one line per S3 item touched, with evidence)
## OUT OF SCOPE    (what you noticed and did NOT do)
## RISK            (what a reviewer should look hardest at)
```

## S7 — Standing traps

```
TRAP-1  PMC12452463's gold export_rationale records the route as chemically
        BROKEN (EntA absent; nothing converts 2,3-dihydro-2,3-dihydroxybenzoate
        onward). After C-010 it passes quarantine, the final Stage-3 gate and
        the required-field gate. Its CORRECT outcome is review_required with
        strict_acceptance_eligible=false. It must NEVER count as strict success.
        Any agent optimizing toward "PMC12452463 passes strict" is chasing the
        wrong target.
TRAP-2  tests/test_strict_quarantine_real_artifact_replay.py:416 pins
        FULL_STACK_BASELINE by exact equality. C-010 changes it BY DESIGN.
        An agent that makes this test pass by reverting behaviour is rejected.
TRAP-3  placeholder_backed_proteins (21 in the pinned run) is a standing POLICY
        DISAGREEMENT between gold set and pipeline, not a defect. No agent may
        "fix" it. Escalate to the product owner.
TRAP-4  Do not rebuild what already exists. See MASTER_PLAN.md § 2:
        the UniProt accession fetch (enrich_entities.py:507 + EnrichmentCache),
        accession-keyed PathBank lookup (map_ids.py:1822), the provenance
        carrier (pipeline._carry_rag_provenance:1880), alias traversal
        (reference_repair.REFERENCE_FIELDS:149), semantic not_evaluated
        (SemanticReport.evaluated), the SBML canonical binding, the artifact
        replay harness, the RAG gap detector (retrieve.detect_gaps:656 +
        class Gap:220), the RAG claim schema (admission.RagReactionCandidate),
        or the RAG admission gate (admission.py, 3100 lines).
TRAP-5  data/enrichment_cache.json is 39 MB and TRACKED. No branch may commit a
        cache modification.
```
