# SPIKE-002 — Compound-resolution extraction scoping

**No code. No branch. Investigation only.** Blocks C-040, C-050, C-051 only.

---

```
[S1] [S5] [S8]

ROLE
  Scoping investigation. You produce a verdict and a plan, not a patch. You
  create no branch and modify no file.

THE QUESTION
  Can src/t2pw/pwml/ir.py :: _resolve_compound_rows (:797-897) and
  _canonicalize_compound_offline (:578-621) be LIFTED into a stage-neutral module
  callable BEFORE the canonical freeze -- or are they coupled to build_pwml_ir's
  internal row shapes such that lifting requires reshaping their input?

WHY IT MATTERS
  Measured on committed runs_verify/2026-08-04_1647/papers/PMC12856317/strict:
  final_mapped.json -> pwml_ir.json adds pathwhiz_id and db_id to all four
  compounds and gives Glycine NINE external identifiers absent from the canonical
  payload -- drugbank DB00145, hmdb HMDB0000123, kegg C00037, chebi 15428,
  pubchem 5257127, chemspider 730, cas 56-40-6, pathbank_compound_id 78 -- plus a
  CHEBI: prefix normalization on three others. Ten identifiers added post-freeze.

  Regenerate it yourself:
    .venv/Scripts/python.exe \
      docs/pwml_recovery_sprint/evidence/probe_exporter_identity_mutation.py

  PRODUCT_CONTRACT section 5 forbids exactly this. Moving the resolution is
  C-040/C-050/C-051; how hard it is decides whether the first release-candidate
  benchmark starts Day 5 or Day 6. This is the sprint's single largest schedule
  unknown.

INVESTIGATE
  1. What shape does _resolve_compound_rows require -- payload compound rows, or
     rows build_pwml_ir has already transformed? Trace its caller at ir.py:1107.
  2. Does it depend on build_pwml_ir locals -- the dedupe map, key assignment,
     _dedupe_named_rows / lookup at :1030?
  3. Does _canonicalize_compound_offline mutate rows in place in a way a
     pre-freeze caller could reproduce exactly?
  4. What does PathBankDbResolver.from_env() need that is unavailable pre-freeze?
  5. Would moving it change WHICH rows get resolved? Quarantine prunes entities,
     so a pre-freeze caller sees rows the exporter never would. Say whether that
     changes the resolved set, and whether that difference is desirable.
  6. build_pwml_ir is 1042 lines (:966-2007). Where is the natural seam?

CONSTRAINTS
  Read-only. Run the evidence probes if useful -- through the bounded wrapper per
  [S8], and note that probe_exporter_identity_mutation.py may attempt a PathBank
  connection via PathBankDbResolver.from_env(). Record whether the database was
  reachable, because a reachable-or-not database changing the exported identifiers
  is itself part of the finding.

DELIVER
  ## VERDICT           CLEAN_LIFT | LIFT_WITH_ADAPTER | REQUIRES_RESHAPE
  ## FUNCTIONS TO MOVE exact names and line ranges
  ## NEW MODULE BOUNDARY  what goes in pwml/compound_resolution.py, what stays
  ## C-040 SIZE       estimated changed lines
  ## RISK             named, if REQUIRES_RESHAPE, plus a proposed 3-way split
  ## SCHEDULE IMPACT  does the Day-5 RC benchmark still hold?
  ## EVIDENCE         commands run and their output, per [S5]
  ## DB REACHABLE     yes/no, and what it changed
```
