# ORCH-787 RESULT — `PREFREEZE_DUPLICATE_CANONICAL_ROWS` on chlorophyll (protoheme / heme)

Branch `orch787/fix` off `22538dee`. **Status: code + test + replay written, NOT YET RUN.** No Python job
was started: free RAM stayed between 0.73 and 1.44 GB for ~45 minutes (floor 2 GB, brief § Machine rules;
no sprint Python process was running — the memory is held by desktop apps). Nothing below is measured
except where a file path says so.

## 1. Cause (read from code and saved artifacts, not run)

- ORCH-784 `chlorophyll-strict` stopped at `streamlit_app.py:4324` `run_prefreeze_resolution` →
  `prefreeze_resolution._reject_duplicate_canonical_rows` (whole-pathway raise). Leg:
  `.claude/worktrees/orch784run/runs_papers/ORCH-784/legs/chlorophyll-strict/2026-10-27_2027/papers/PMC5130987/strict/RESULT.txt`.
- The same paper's ORCH-781 leg `final_mapped.json` shows Stage 6 stamps `heme` with PathBank **1799**
  (`legacy_id_unverified`, name kept, synonyms `Heme; Haem; Protoheme; Heme B; Protoheme IX`). In ORCH-784
  a second row `protoheme` arrived without an id; the pre-freeze DB resolver matches it by PathBank synonym
  to 1799 "Heme" and renames it onto `heme`'s `_norm` key → two rows, one name → refusal. Same entity.
- **ORCH-782 is not the cause.** `strict_quarantine._split_gene_protein_names` runs inside the strict
  quarantine (`strict_quarantine.py:3220`, called from `freeze_canonical_payload`, `streamlit_app.py:4364`),
  i.e. after the failing seam, and touches gene/protein names only.

## 2. Change

`src/t2pw/pwml/prefreeze_resolution.py`
- `_reject_duplicate_canonical_rows(..., resolved)` now returns the groups `_proven_same_entity` proves to
  be ONE PathBank entity instead of raising for them; every other group raises exactly as before.
- `_proven_same_entity` (D-035 clause 3): every member resolves to one PathBank id; each canonicalized
  (renamed) member got that id from resolution (`db_row.id`); no namespace (pathbank, KEGG, ChEBI, HMDB,
  PubChem, InChIKey) holds two values across the group, read from pre- AND post-resolution rows (catches
  F-043 stamped stubs). Never a name test. `class` is not compared (`cofactor` is a role claim,
  `identity_admission.py:45`).
- `_consolidate_rows` (D-035 clause 5): survivor = first row that already held the name; spellings kept
  as synonyms; empty fields filled from the merged row; provenance lineage unioned;
  survivor carries `consolidated_rows: [{name, merged_into, db_id, reason}]`; report key
  `compounds.rows_consolidated`. Connectivity proof kept: the before-signature maps merged-row tokens to the
  survivor (and shifts later indices) and must still equal the after-signature.
`src/t2pw/app/reaction_evidence.py` `_omissions`: each `consolidated_rows` entry is listed under
"Listed, not counted" as a merged duplicate, KI-045.

## 3. Proof — to run (commands in the handback)

- Replay: `docs/pwml_recovery_sprint/evidence/orch787_replay.py` (`record` once online, read-only PathBank
  SELECTs → `evidence/orch787/pathbank_compound_queries.json`; `replay` offline via
  `orch768_replay_corpus.load_tree`). Input = `merged_payload.json` + the Stage-6 stamps ORCH-781 recorded
  (Stage 6 is not replayable, KI-058). Expected: base raises the leg's exact message; tip writes a PWML with
  `protoheme` listed. Output → `evidence/orch787_replay.txt`. **Not run.**
- G9: `tests/test_orch787_duplicate_rows_consolidated.py` — CORRECTION (base raises), NEW ACCEPTANCE (text
  record), CONTROL (conflicting KEGG still refuses). **Not run.**
- Expected deliberate pin move: `tests/test_prefreeze_compound_resolution.py`
  `test_c050g_*` "sn -glycerol 3-phosphate" pair (both CHEBI:15428, both offline-index id 78) clears
  clause 3 and will consolidate instead of refusing. Not yet edited — confirm by running first.

## 4. Dead ends
- Replaying `merged_payload.json` as saved would NOT reproduce the leg: without the Stage-6 stamp `heme`
  is also DB-renamed to "Heme" and the seam raises `AMBIGUOUS_RENAME_TARGET` instead. Hence the recorded
  stamp carry.
- `data/pathwhiz_id_db.json` has no heme entry: the rename is PathBank-DB only, so an offline replay needs
  recorded DB answers.

## 5. Open questions for the product owner
1. **Not-proven groups still refuse the whole pathway.** D-035 clause 7 (LOCKED) says genuinely distinct
   colliding compounds "must keep refusing" (PEtN-lipid A / modified Lipid A); the Lead's rule (D-114) says
   omit the losing row. Options: (a) keep refusal (today; costs those legs their PWML); (b) decline the
   rename for the arriving row so it stays distinct with its extracted name and no identity (D-028
   record-only shape; moves the C_canned golden); (c) omit the arriving row and the processes needing it
   (needs a new reason code in `PWML_KNOWN_ISSUES.md` and a pre-freeze record channel). Needs a D-xxx.
2. D-036 deferred the consolidation engine until a production payload cleared clause 3; this leg is that
   payload. Record the reopening in DECISIONS.
3. A new KI is needed for "two compound rows canonicalize to one PathBank entity" with reason code
   `duplicate_canonical_row_consolidated_same_db_entity`; KI-045 is used meanwhile (constant
   `reaction_evidence.COMPOUND_MERGE_KI`), outside the appendix table.
