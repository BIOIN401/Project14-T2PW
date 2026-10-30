# ORCH-797a RESULT: one reference key for every stage; tRNA is a nucleic acid (D-120 §§ 1, 4)

Branch `orch797a/fix` off `aefa5b30`. Code commit `effba332`. Not merged, not pushed.
**Merge rule 5 is waived by the PO (D-120 § 5): no reviewer round.**

**STATUS: NOT DONE. The code is written and committed, but none of it has been proven.** No test, replay or audit has
run. For the whole session (about 75 minutes), the machine's free physical RAM stayed between 0.3 and 1.1 GB, under
the helpers' 3 GB floor (`run.sh` exit 90 on `audit-r1`, then repeated polls). Committed memory was about 52 GB on a
32 GB machine. The pressure is outside this task: Docker/WSL, MATLAB, browsers. I killed nothing. No G11 report
exists for ORCH-797a, and `g11_evidence.py check` has not run.

## 1. Cause (from the ORCH-794 § 11 and ORCH-796 artifacts, as recorded there; not re-measured)

The stages that resolve references key a name differently:

| key | `tRNA^Glu` | `5-HT 3 receptor` | `NAD+` |
|---|---|---|---|
| `process_normalizer._normalize` / `reference_repair._normalize` (quarantine, admission registry) | `trnaglu` | `5ht 3 receptor` | `nad+` |
| `ir._norm` / `prefreeze_resolution._norm` / `compound_resolution._norm` / `reaction_evidence._ir_norm` | `trna glu` | `5ht 3 receptor` | `nad+` |
| `entity_admission._norm` | `trnaglu` | `5ht3receptor` | `nad` |
| `entity_identity._normalize` (= `map_ids._normalize_name`) | `trnaglu` | `5ht 3 receptor` | `nad` |

The quarantine admits `tRNAGlu` against the row `tRNA^Glu`, but the IR cannot resolve it, so the whole pathway is
refused (`unresolved_entity_reference`). This is how ORCH-796 pass-1 chlorophyll failed. ORCH-794b fixed only the
two-kind case.

## 2. Change (`effba332`)

| file | function | change |
|---|---|---|
| `pipeline/name_keys.py` | `reference_key` (NEW) | NFKC, unicode dashes, any Greek letter spelled out (`α`->alpha; Unicode `LAMDA`->lambda; accents ignored), casefold. Punctuation and whitespace are removed. A separator between two digits is kept as `,` (`1,12-x` != `11,2-x`). Digits, `+` and `:` stay identity-bearing. `normalize_name`/`strict_key` are untouched (twinned with the evaluator) |
| `pipeline/process_normalizer.py` `_normalize`, `pipeline/reference_repair.py` `_normalize`, `pipeline/entity_admission.py` `_norm`, `pwml/ir.py` `_norm`, `pwml/prefreeze_resolution.py` `_norm`, `pwml/compound_resolution.py` `_norm`, `app/reaction_evidence.py` `_ir_norm` | | each now returns `reference_key(value)`, so every caller uses the same key |
| `pipeline/entity_admission.py` | `_core` | the currency check folds with VOCABULARY's own fold (`[^a-z0-9]` removed), so `NAD+` stays currency |
| `pwml/compound_resolution.py` | short-abbreviation length gate | measured on the pre-D-120 fold, so no name crosses the 4-character gate (no gate moves) |
| `pipeline/entity_identity.py` | `_named_in` (new), used by `_is_explicit_complex_colon_name`, `route_entity_for_mapping`, `compound_name_block_rule` | a caller's pre-folded name set is matched under either fold. `map_ids` builds its sets with `_normalize_name` (the old fold), and `process_normalizer` now builds them with the reference key. `entity_identity._normalize` itself is NOT changed: it is twinned with `map_ids._normalize_name`, which is in ORCH-797b's lane |
| `pipeline/strict_quarantine.py` | `_spell_references_canonically` (new), `_spell_references_as` (generalised) | pre-freeze, before the registry: every process, location and complex-component reference that names an entity row only under the shared key is rewritten to that row's exact name. If a key is held by rows of two kinds, nothing is rewritten and admission omits each such row (KI-011). Within one kind, the first row wins, the same row the IR's `_dedupe_named_rows` keeps. Free-text fields are never touched. The per-decision ORCH-794b call is replaced by this one pass |
| same | `_paper_kind`, `_keep_one_kind_per_name`, `_split_gene_protein_names`, `_TRNA_WORDING` (new) | a tRNA (`tRNA^Glu`, `Glu-tRNAGlu`, `L-glutamyl-tRNA(Glu)`) declared in two kinds keeps the nucleic acid before any identity rule (rule `trna_is_nucleic_acid`). A tRNA nucleic acid that clashes with a protein is never renamed `<name> gene`; the protein row is dropped instead (`same_name_declared_as_another_kind:nucleic_acids`, KI-051) |
| `tests/test_orch797a_reference_key.py` | new | 4 G9 tests, 3 preservation tests, 3 labelled NEW ACCEPTANCE tests |

Per-row omission (brief item 3b). Once every stage shares the key, a reference that resolves at admission also
resolves in the IR, and the quarantine's existing per-row path omits anything else
(`test_preservation_a_reference_naming_no_row_is_omitted_per_row_never_the_pathway`). No new reason code was added.

**Left alone on purpose:**
- `pwml/db_resolver._name_key`. It matches payload names against PathBank DB names (compound identity), not a
  process reference against an entity row. It drops `+`, so changing it would split `NAD+` from PathBank's `NAD`.
- `map_ids._normalize_name` (out of lane).

## 3. Proof

| what | result | G11 |
|---|---|---|
| G9 base arm (`C:/t/orch797a/base`, test copied in) | **NOT RUN** (RAM) | none |
| G9 tip | **NOT RUN** | none |
| affected importers (110 non-AppTest files list: `C:/t/orch797a/affected.txt`) | **NOT RUN** | none |
| collision audit (`evidence/orch797a_audit.py`, brief item 5) | **NOT RUN** | none |
| six-leg tail replay base vs tip (`evidence/orch797a_replay.py`: the leg's own `final_mapped.json`, unmodified) | **NOT RUN** | none |
| `g11_evidence.py check --task ORCH-797a` | **NOT RUN** | none |

## 4. Pins expected to move (unmeasured)

These are predictions, not measurements:
- Every hash over IR/quarantine keys or over admitted payloads that contain a name changed by the new key (for
  example the `test_orch759c_quarantine_bucket_closure` chlorophyll fixture sha pins). The ORCH-794b note in
  `prefreeze_resolution` already said changing `_norm` "would move the graph hash".
- `test_strict_quarantine*` overlap-gate vehicles, wherever two spellings now merge.

## 5. Failures, dead ends, owned breaches

1. **Blocked on RAM** (above). `run.sh … audit-r1` exited 90 before any job started. No G11 number was allocated.
2. **Owned breach:** one `python -c "print(1)"` was run outside the wrappers, to check the interpreter. Nothing else
   ran through it.
3. **Dead end avoided (reasoned, not measured).** Routing `entity_identity._normalize` through the new key would
   break `map_ids`' membership sets, which are built with the old fold. That is why `_named_in` exists.
4. **Dead end avoided (reasoned, not measured).** A space-free key fed straight into `compound_resolution`'s length
   gate would make short names with spaces need corroboration, which stricter, and would lose IDs. That is why the gate
   measures the old fold.

## 6. KI delta

No new reason code. A tRNA decision reuses `same_name_declared_as_another_kind:nucleic_acids` (KI-051) with the new
`decided_by: trna_is_nucleic_acid`.

## 7. LLM-side

None found. No leg was read beyond the file listing.

## 8. NOT settled (for the PO / sub-orchestrator)

1. **Run the proof.** A job needs at least 3 GB free; at least 6 GB is safer. In order: G9 base, G9 tip, affected
   chunks, audit, replay, G11 check. Everything is scripted and committed. Cost: about 30-60 minutes of one builder.
2. **A tRNA declared ONLY as a compound** is not converted to a nucleic acid. Converting it means moving its location
   rows across buckets, after the kind was decided. Options:
   - (a) leave it. Every reference still resolves to the one row.
   - (b) convert it pre-freeze. Cost: one function, and a G9 test with location moves.
3. **`map_ids._normalize_name` / `entity_identity._normalize`** should move to `reference_key` together, in
   ORCH-797b's lane or after it merges, and then `_named_in` should be dropped.
4. **`+` stays identity-bearing**, as the IR has always had it. No collision audit has been run to prove that
   dropping it is safe.
5. **The respelling walk** rewrites any non-free-text string in a process or location row whose key equals an entity
   name, as ORCH-794b did for one name. Applied to every name, an unusual field (for example a location string that
   is also an entity name) could be rewritten. The audit (item 1) is the check for this.

## 9. Proposed LEDGER row

`| ORCH-797a | D-120 §§ 1, 4 one reference key; tRNA nucleic acid | orch797a/fix @ effba332 | code committed, UNPROVEN: RAM < 3 GB floor all session | G9/affected/replay/audit/G11 not run | ORCH-797a-RESULT.md |`
