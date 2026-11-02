# ORCH-799 research — which duplicated decisions are still open at b25327c1

Read-only pass. Nothing was run; every status below comes from reading `src/` at `b25327c1`
(branch `orch799/typed-core`). Identity routing (`reference_key`, ORCH-797a/c) is out of scope
and not re-proposed.

All paths below are under `src/t2pw/`.

## 1. Status table

| Finding | Status at b25327c1 | Evidence |
|---|---|---|
| F-206 kind disagreement | **FIXED** (mapper side) | `9757234d` (ORCH-762 FX6); one kind per name `pipeline/strict_quarantine.py:1902 _keep_one_kind_per_name`; route still decided in `pipeline/entity_identity.py:420` from a caller-supplied set. Residual (admission does not restore a withheld accession) is an acknowledged non-goal. |
| F-207 strict normalization exception | **OPEN** | `app/streamlit_app.py:3620` (`normalize_process_payload`) inside a `try` whose `except Exception` is at `:3656`; fallback `normalized_input = stage2_payload` at `:3525`. Protected file: **record only**. |
| F-208 row 1 protein references, name vs synonym | **OPEN** | see § 2.1 |
| F-208 row 2 actor slots | FIXED `509fe185` | `_actor_verdict` strict_quarantine.py:1384 |
| F-208 row 3 closure keep set | FIXED `45aa9c28` | `quarantine_and_close` ~3638 |
| F-209 audit patch stale indices | **FIXED** (`2a791f69`, ORCH-769d, KI-018) | `curation/apply_audit_patch.py:2659 "Index rebasing"`, `_rebase_op :2690`, wired at `:2778-2800`. **Residual OPEN**: `/-` on a missing key creates a dict (`_set_value :84`, `_resolve_parent :63`); ORCH-769d-RESULT § 7 item 1 left it for its own lane. |
| F-212 two compound gates | **OPEN in general** | see § 2.3 |
| F-214 gene and protein under one bare symbol | **OPEN upstream** (quarantine side by design, D-106) | `strict_quarantine.py:1800 _split_gene_protein_names`, `:1902 _keep_one_kind_per_name`; audit guard `apply_audit_patch.py:861,882-903` pools buckets. |
| F-218 fabricated pwp-id | **OPEN, unadjudicated** | `pwml/writer.py:1311` |

So three of the seven the lane list named are already fixed in code (F-206, F-208 rows 2-3, F-209
main defect). Their FINDINGS headers should not be read as open work. F-209's FINDINGS header still says
"OPEN"; that is stale (the residual `/-` defect is what remains).

## 2. Per-finding analysis

### 2.1 F-208 row 1 — a protein is "referenced" by name in two places and by name-or-synonym in two others

Four sites decide "is this protein row reached by a surviving process":

| site | test | file:line |
|---|---|---|
| quarantine pruner | `_entity_name_norms([row]) & keep` (name UNION synonyms) | `pipeline/strict_quarantine.py:2208 _prune_entities`, test at `:2245-2247` |
| quarantine degree-zero check | name UNION synonyms | `strict_quarantine.py:2837 _degree_zero_exports`, `:2888, :2904` (aligned by C-067 `96557ef6`) |
| normalizer orphan pass | **name only**: `_normalize(_canonical(protein["name"])) in process_ref_norms` | `pipeline/process_normalizer.py:~2200-2300` (test at `:2278`) |
| final gate degree test | **name only**: `deg.get(node("protein", name)) == 0` over `build_graph` | `process_normalizer.py:~4511-4520` (prune pass) and `:5110-5118` (emits `DEGREE_ZERO_PROTEIN_MESSAGE`) |

The shared helper already exists: `process_normalizer._entity_name_norms :638`, which the quarantine
imports (`strict_quarantine.py:88`). What is missing is a single "is this row reached" predicate over a
referenced-norm set; the two normalizer sites each re-implement the membership test with the primary name.

**Minimal typed single-source fix.** Add one function next to `_entity_name_norms` (`process_normalizer.py:638`):

```
def row_is_referenced(row: Mapping[str, Any], referenced_norms: Set[str]) -> bool:
    return bool(_entity_name_norms([row]) & referenced_norms)
```

Call it from `_prune_entities :2247`, `_degree_zero_exports :2888/:2904`, and (changed) the orphan pass
`:2278`. The gate's `build_graph` degree test (`:5110`) works on graph nodes keyed by name, so it would
need a synonym-aware edge resolution, a larger change.

- Pure-equivalence part (quarantine sites): about 8 lines, no verdict change.
- Behaviour-changing part (normalizer orphan pass and gate): about 25-40 lines; changes which proteins are
  dropped or refused.
- **Product decision required.** ORCH-759a § 7 item 3 lists the three options ((a) leave, (b) synonym-aware
  gate, (c) name-only pruner). (b) is a gate change; D-106 says no gate change without a ruling. Do not build
  without a D-entry. The pure-equivalence half (route the two quarantine sites through the one predicate) needs
  none.
- **G9 proof (if (b) is ruled):** a payload with protein `P` (no accession shared with a survivor) whose
  synonym `Q` is the only name a surviving reaction uses. Base: pruner keeps `P`, gate refuses
  `Protein has degree 0` -> late refusal. Tip: row is either kept and gate passes, or removed early;
  behaviourally observable on the pre-export gate result. Fixture seed: `tests/fixtures/orch759a/enterobactin_final_mapped.excerpt.json`.

### 2.2 F-207 — strict normalization failure vs the export's schema check (record only)

Two consumers of one contract: `app/streamlit_app.py:3656` swallows the exception and ships the Stage-2 shape;
the export's validation-only revalidation then refuses on `actor_schema_not_canonical`. Fix site is the protected
file and needs a D-entry (ORCH-758a-RESULT § 8 Q1). **Not proposed here.**

Related, *not protected* duplicate found while reading: the actor-row canonical predicate ("dict with non-empty
`entity` and `entity_type`") is written twice, with the same error code `actor_schema_not_canonical`:

- `pipeline/payload_models.py:~680-697` (`isinstance(entity, str)` and `.strip()`; a non-dict actor is skipped)
- `pipeline/stage_contracts.py:~802-815` (`_text(...)`; a non-dict actor is **an error**)

They already differ on non-dict actors. Single-source: one `actor_row_is_canonical(actor) -> bool` in
`payload_models.py`, called by `stage_contracts.py`. About 12 lines. The non-dict divergence means this is
**not** a pure equivalence refactor: whichever rule becomes the source changes the other validator's verdict
on a non-dict actor. Needs a quick look at who calls each before choosing (not done here). Low priority for the
lane; flagged because it is the same class as F-207.

### 2.3 F-212 — Stage-6 name gate vs pre-freeze compound resolution

- Site A: `mapping/map_ids.py:6152 _name_gate_verdict` (used at `:6865`, `:7175`, `:7418`). Refuses PathBank 874
  for "prenyl diphosphate" on token overlap.
- Site B: `pwml/db_resolver.py:364` rule `exact_short_name_or_synonym` (0.70), admitted by
  `pwml/compound_resolution.py:354 EXACT_NAME_MATCH_RULES` and `_admit_db_identity`.
- The bridge that exists: `compound_resolution.py:~599-628` vetoes a stamp when the row's
  `mapping_meta.rejected_mapped_ids` already holds that namespace. The gap: `map_payload` writes
  `rejected_mapped_ids` only when it strips ids from a row (`map_ids.py:6340`, `:7399`, `:7444`), so a gate
  refusal on an unmapped result is invisible to B.
- Fix: persist every `reject` verdict into `rejected_mapped_ids`. That changes `mapping_meta`, which the audit LLM
  reads. **Needs a D-104 AMENDMENT 2 ruling first** (FINDINGS F-212). The alternative of B calling A's
  `_name_gate_verdict` directly is the right single-source, but B would then refuse synonym matches it admits
  today: a policy change. About 30-50 lines either way. **Not a lane candidate** until ruled.

### 2.4 F-214 — gene and protein under one bare symbol

Created upstream (Stage 1) and handled safely downstream in several places (`_split_gene_protein_names`,
`_keep_one_kind_per_name`, `_entity_type_overlaps :2676`). No two components decide differently now; the
remaining cost is a missing capability (naming gene rows distinctly), which is "not chartered". Skip: new
feature, not drift.

### 2.5 F-218 — `PW_P{id:06d}` pwp-ids

`pwml/writer.py:1311` is the only producer of `pwp-id` in `src/` (grep across `src/` found no other writer);
`pwml/ir.py:3371` only mentions it in a comment. There is no second decider, so this is not a duplication bug.
It is a value-policy question: write a blank, write a real PathBank `PW_P` accession when
`pathbank_complex_id`/`pathwhiz_id` is known, or keep fabricating. Colliding rows were observed, harm to PathWhiz
is not established (`evidence/orch761_pathbank_probe5.txt`; census timed out). **Needs a product decision**
before any code; no G9 proof can be written until the intended value is chosen. Tests touching pwp-id:
`tests/test_pwml_ir.py`, `tests/test_pathbank_unknown_fallback.py`.

### 2.6 F-209 residual — `/-` on a missing key creates a dict

`curation/apply_audit_patch.py:63 _resolve_parent`, `:84 _set_value`. The options (reject, create list, leave)
are in ORCH-769d-RESULT § 7 item 1; each is a behaviour change and "needs its own lane". About 10 lines for
option (a) (reject: raise inside `_set_value`, already turned into "Application failed" by the caller's `try`).
**Needs a ruling (a/b/c)**; no new D-entry beyond choosing, but the choice is the product owner's. G9 proof:
patch `add /entities/compounds/3/synonyms/-` on a compound with no `synonyms` key; base writes a dict into
`synonyms`, tip rejects it. Test file: `tests/test_apply_audit_patch_referential_integrity.py` neighbours,
plus `tests/test_apply_audit_patch_lineage.py`.

## 3. Ranking (recurrence x still open x small diff)

| rank | item | recurrence | open | diff | blocker |
|---|---|---|---|---|---|
| 1 | F-208 row 1, quarantine half: route `_prune_entities` and `_degree_zero_exports` through one `row_is_referenced` | high (4 sites, 3 legs hit it) | yes | ~8 lines | none (equivalence) |
| 2 | F-208 row 1, behaviour half: normalizer orphan pass and gate become synonym-aware | high | yes | 25-40 | PO ruling (D-106 gate change) |
| 3 | F-209 residual `/-` defect | medium (every multi-op patch) | yes | ~10 | PO ruling a/b/c |
| 4 | actor-row canonical predicate written twice (new, under F-207 class) | medium | yes | ~12 | non-dict divergence must be chosen |
| 5 | F-212 | medium | yes | 30-50 | D-104 AMENDMENT 2 |
| 6 | F-218 | low | yes | small | PO value ruling, no second decider |
| 7 | F-207 | high | yes | n/a | protected file, D-entry |
| 8 | F-214 | high | yes | large | new feature |
| - | F-206, F-208 rows 2-3, F-209 main | - | fixed | - | - |

## 4. Recommended lanes (at most three)

### ORCH-799a — one "is this row reached" predicate (pure equivalence + a locked-in contract test)

- Owns: `pipeline/process_normalizer.py :: _entity_name_norms` neighbourhood (add `row_is_referenced`),
  `pipeline/strict_quarantine.py :: _prune_entities :2245`, `_degree_zero_exports :2888,:2904`,
  `quarantine_and_close ~3655` (same membership expression). Does **not** touch the normalizer orphan pass or
  the gate.
- Proof: this is a refactor; no G9 base failure is claimed. Add an explicitly labelled **new acceptance test**: for
  random payloads, `_prune_entities`, `_degree_zero_exports` and the closure keep set agree on the same row, and a
  synonym-only reference keeps the row in all three. Property-style, seeded fixtures from
  `tests/fixtures/orch759a/`.
- Rerun: `tests/test_orch769b_actor_protein_registry.py`, `tests/test_locked_noop_quarantine_policy.py`,
  `tests/test_batch_research_gate_fail_open.py`, the ORCH-759a/c tests (`grep -l _degree_zero_exports tests`),
  and the 508-test smoke suite (gate 10).
- Do not bump `QUARANTINE_POLICY_VERSION` (verdict-neutral).

### ORCH-799b — audit patch `/-` on a missing key (after the product owner picks a/b/c)

- Owns: `curation/apply_audit_patch.py :: _resolve_parent :63, _set_value :84` only.
- Proof (G9, behavioural, fails on base): `add /entities/compounds/N/synonyms/-` on a row without `synonyms`.
  Base leaves `synonyms` a dict; tip rejects (option a) or writes `["x"]` (option b). Also assert that a
  multi-op patch with a remove still applies later ops at rebased indices (guards ORCH-769d).
- Rerun: `tests/test_apply_audit_patch_lineage.py`, `tests/test_apply_audit_patch_lock_policy.py`,
  `tests/test_apply_audit_patch_referential_integrity.py`, `tests/test_audit_json_llm_payload.py`,
  `tests/test_c105_actor_role_evidence.py`.
- Blocked on a ruling; list it so the product owner can pick.

### ORCH-799c — single actor-row canonical predicate (conditional)

- Owns: `pipeline/payload_models.py :: ~680-697`, `pipeline/stage_contracts.py :: ~802-815`; one new
  `actor_row_is_canonical` in `payload_models.py`.
- First task is a read: confirm which validator is called when and whether the non-dict-actor difference is
  reachable. If it is not reachable, this is a pure equivalence refactor; if it is, a ruling on which verdict
  wins is needed first and the lane is dropped.
- Proof if reachable: a payload with a non-dict actor produces `actor_schema_not_canonical` from one validator
  and not the other at base; both agree at tip.
- Rerun: `grep -l "actor_schema_not_canonical\|stage_contracts\|payload_models" tests` plus the smoke suite.

## 5. Questions for the product owner (not settled)

1. F-208 row 1: option (a) leave, (b) synonym-aware normalizer pass and gate, or (c) name-only pruner. Cost of
   (b): gate change under D-106, more proteins kept early. Cost of (c): can undo F-083 synonym-aware complex handling.
2. F-209 residual: (a) reject, (b) create the list, (c) leave. See ORCH-769d-RESULT § 7 item 1.
3. F-218: blank, real PathBank `PW_P` accession when known, or keep fabricating; needs a PathWhiz-side check that
   a collision does nothing harmful.
4. F-207 and F-212 stay with their existing rulings pending (protected file; D-104 AMENDMENT 2).

## 6. Not verified here

- The exact line of the F-208 final-gate degree test was read at `process_normalizer.py:5110-5118` and the orphan
  pass at `:2200-2300`, but the other callers of `build_graph` degree were not enumerated.
- The `actor_schema_not_canonical` pair was compared by eye in two excerpts; their callers were not traced.
- No test was run; reruns listed above are by grep of `tests/`, not by measured impact.
