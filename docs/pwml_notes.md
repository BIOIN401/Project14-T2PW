# PWML Requirements & Issue Log

This file is a running reference for what PathWhiz expects from generated `.pwml`
files, plus a tally of concrete issues we've found in generated output and the
fixes (planned or applied) for each. Add to both sections as we learn more —
don't delete old entries, just mark them resolved.

## Requirements (confirmed so far)

### Structure (enforced by `src/t2pw/pwml/validate.py`)
- Root: `<super-pathway-visualization>` with children in this order:
  `named-for-id, named-for-type, cached-name, cached-description, cached-subject,
  pw-id, pathway-visualization-contexts`.
- `pathway-visualization` section order matters (height, width, background-color,
  id, pathway, cell-types, species, subcellular-locations, tissues,
  biological-states, bounds, compounds, ..., reactions, ..., visualizations...).
  `validate.py --repair-out` can reorder/insert empty containers automatically.
- Every `<compound>` is expected to carry the full PathBank field set
  (`hmdb-id`, `cas`, `kegg-id`, ... `pwc-id`, `short-name`, `element-states`).
- Every `<biological-state>` is expected to carry `pwbs-id`.

### Biological states represent **compartments, not entities**
- A `biological-state` = one (species, subcellular_location, tissue, cell_type)
  combination. ALL compounds/proteins physically present in that compartment
  should point at the **same** biological-state id.
- Reaction/transport visualizations draw arrows between participants that share
  a biological-state. If each participant gets its **own** biological-state
  (e.g. one named after the compound itself), the reaction can never be rendered
  as connected — this is a fundamental breakage, not just a cosmetic issue.
- `pwml_system.txt` (LLM prompt) currently only requires
  `species` + `subcellular_location` per biological_state — it does **not**
  instruct the model to reuse/dedupe states that share the same
  (species, subcellular_location, tissue, cell_type). `process_normalizer.py`
  also has no dedupe pass for `biological_states` (compare `_dedupe_named_rows`,
  which IS applied to compounds/proteins/complexes but not biological_states).

### `pwc-id` / `pwbs-id` / `short-name` for brand-new entities
- For compounds/biological-states that already exist in PathBank's DB, these
  IDs come from the DB lookup (`_trusted_compound_pwc_id`,
  `_trusted_biological_state_pwbs_id` in `writer.py`).
- For genuinely new entities (custom compound IDs in the 20000+ range, novel
  biological states), there is no DB row, so these fields are simply omitted.
  `validate.py` flags this as `missing-field` against the reference signature.
  **Unconfirmed**: whether PathWhiz's importer actually requires these fields
  to be present (even empty/`nil="true"`) for new-entity creation, or whether
  omission is fine and PathWhiz assigns them on save. Need to check against an
  actual PathWhiz import error message next time this happens.

### `__auto_state__`
- `ensure_autostates()` in `process_normalizer.py` creates a fallback
  biological-state literally named `__auto_state__` (location "cell") to catch
  compounds/proteins with no assigned compartment. The literal string
  `__auto_state__` should never reach the final PWML — as of 2026-06-11,
  `_rename_auto_states()` in `src/t2pw/pwml/ir.py` renames it to
  `"<species> <subcellular_location>"` (or `"Unlocalized"` if neither is
  known) before the IR is handed to the writer. See Issue Log entry below.

### IDs must be unique within each section
- Each `<compound>`/`<protein>`/etc. `id` should appear exactly once in its
  section. Duplicate entries (same id, same data, listed twice) have shown up
  in generated output — likely from merging multiple extraction passes without
  a final dedupe-by-id step.

---

## Issue Log

### 2026-06-11 — Kutzneria/PntM pentalenolactone pathway (pasted PWML, not yet saved to a file)
- **Per-entity biological states (likely the main "won't import/render" cause)**:
  the pathway had separate biological-states:
  - id 6: "pentalenolactone F in Streptomyces cytosol"
  - id 7: "pentalenolactone in Streptomyces cytosol"
  - id 8: "PntM in Streptomyces cytosol"

  All three are (species=Streptomyces sp., subcellular_location=cytosol) — i.e.
  the same compartment — but were given distinct biological-state ids/names
  instead of being merged into one "Streptomyces sp. cytosol" state. Reaction
  50002 (PntM oxidizing pentalenolactone F → pentalenolactone, with
  ferredoxin/H+/O2/water) therefore has its participants split across states
  that share no common biological-state, so PathWhiz has nowhere to draw a
  connected reaction-visualization for it.
  - Root cause: no dedupe of `biological_states` by
    (species, subcellular_location, tissue, cell_type) in
    `process_normalizer.py`, and `pwml_system.txt` doesn't instruct the LLM to
    reuse an existing compartment state instead of inventing a new one per
    entity.
  - **Fix: applied (2026-06-11)**. Added `_merge_duplicate_biological_states()`
    in `src/t2pw/pwml/ir.py`, called right after `ir["biological_states"]` is
    built. It groups states by
    `(species_key, subcellular_location_key, cell_type_key, tissue_key)`,
    keeps the entry with the shortest name as the surviving state for each
    group, remaps `state_by_name` so every later lookup (compounds, proteins,
    reaction participants, etc.) resolves to the survivor, and emits a
    `duplicate_biological_state_merged` warning into the IR report for each
    merged state. States with an unknown/`(None, None, None, None)` compartment
    key, or groups of size 1, are left untouched. Did NOT touch
    `process_normalizer.py` (would break exact-match assertions in
    `tests/test_process_normalizer.py`).
    Still **not done**: `pwml_system.txt` has not been updated to instruct the
    LLM to reuse an existing compartment state — the merge fix is a downstream
    safety net, not a prompt-side fix.

- **Duplicate compound entry**: `Water` (id 1420) appears twice in
  `<compounds>`, once with full HMDB metadata both times — a verbatim
  duplicate.
  - Root cause: two differently-named compound entities (e.g. "Water" and
    "H2O") both resolve via DB lookup to the same `pathwhiz_id`
    (`pathbank_compound_id`), so the writer emits two `<compound>` elements
    with the same `<id>`.
  - **Fix: applied (2026-06-11)**. Added `_merge_duplicate_compound_entities()`
    in `src/t2pw/pwml/ir.py`, called right after `ir["entities"]["compounds"]`
    (and the `entity_by_key`/`entity_by_name` indexes) are built. It groups
    compound records by `pathwhiz_id` (records with no `pathwhiz_id`, i.e.
    novel compounds, are left alone), keeps the first record per group as the
    survivor, remaps `entity_by_key`/`entity_by_name` so all later references
    point at the survivor, and emits a `duplicate_compound_merged` warning per
    merge. Verified on `tmp/final.mapped.json`: 22 compounds in, 22 unique
    `<id>`s out (no merges needed for that dataset, but no regressions either).

- **`__auto_state__` leaked into output**: biological-state id 9 is named
  literally `__auto_state__`, with subcellular-location "cell" (GO:0005623)
  and species Kutzneria — distinct from the proper "Kutzneria cytosol" state
  (id 5). It has a `bound-visualization` row but (in the visible portion) no
  compounds/proteins placed in it — an empty, mislabeled row.
  - Root cause: `ensure_autostates()` fallback name is never replaced with a
    human-readable name, and/or the entities that triggered its creation
    should have been assigned to the existing "Kutzneria cytosol" state
    instead.
  - **Fix: applied (2026-06-11)**. Added `_rename_auto_states()` in
    `src/t2pw/pwml/ir.py`, called after `_merge_duplicate_biological_states()`.
    Any biological-state whose name normalizes to `__auto_state__` is renamed
    to `"<species name> <subcellular location name>"` (or just whichever of
    the two is known; `"Unlocalized"` if neither is known). Verified on
    `tmp/final.mapped.json`: the `__auto_state__` state (species=Pseudomonas
    fluorescens, location=cell) became `"Pseudomonas fluorescens cell"` in the
    final `<biological-states>` output. Did not address the deeper "should
    this entity have been assigned to an existing compartment state instead"
    question — that's still a `process_normalizer`/prompt-level improvement to
    consider separately.

- **Missing `pwc-id` / `short-name` / `pwbs-id`** on all novel
  compounds/biological-states (L-N5-OH-Ornithine, L-piperazate, pentalenolactone
  F, pentalenolactone, and biological-states 5/6/7/8/9). Consistent with the
  "new entity" case described in Requirements above — flagged by
  `validate.py` but status as a real PathWhiz blocker is unconfirmed.
  - Fix: **not yet applied** — still unconfirmed whether PathWhiz's importer
    requires these fields for new entities.

### 2026-06-11 — Duplicate enzyme/protein-complex visualizations (ObaG complex x8 instead of x4)
- A different LLM model's extraction listed the same catalyst (e.g. "ObaG
  complex") in **both** `reaction.enzymes` AND `reaction.modifiers` (with
  `role: "catalyst"` in both) for the same reaction. `ir.py` builds
  `enzyme_sources` by concatenating `raw["enzymes"]` + catalyst-role
  `modifiers` with no dedup, so each affected reaction got two identical
  enzyme entries for the same protein complex. Combined with the
  one-visualization-per-reaction-use enzyme-viz fix (commit 9388b37), this
  produced 2x the expected `<protein-complex-visualization>` entries (8
  instead of 4 for ObaG complex across 4 reactions).
  - **Fix: applied (2026-06-11)**. In `src/t2pw/pwml/ir.py`, after building
    `enzyme_sources` (concatenation of `enzymes` + catalyst `modifiers`),
    dedupe by `(normalized actor name, role)` before resolving entities and
    appending to `reaction["enzymes"]`. Verified on `tmp/final.mapped.json`:
    each of the 4 ObaG-catalyzed reactions now has exactly one `pc_1` enzyme
    entry, and the final PWML has exactly 4 `<protein-complex-visualization>`
    entries for `protein-complex-id 40000`.

### Reactions with no enzyme and not marked spontaneous
- `pwml_qa.py` treats "no `reaction-enzymes` AND not `spontaneous`" as a hard
  **error**: `Reaction '<name>' has no enzyme and is not spontaneous.`
- `writer.py` itself handles this fine structurally — it draws a "virtual"
  edge directly between the reactant/product locations (no enzyme box), so
  the PWML is still valid XML and importable. The QA error is a
  data-completeness flag, not a structural/import blocker.
- This came up for "Obafluorin assembly from AHNB and 2,3-DHBA" in
  `tmp/final.mapped.json`, which has `"enzyme": null` and no `spontaneous`
  field. **Not fixed** — requires either identifying the real catalyst from
  the source paper, or explicitly setting `"spontaneous": true` if the step
  is genuinely uncatalyzed. This is a per-pathway data decision, not a
  pipeline bug.

### 2026-06-11 — Verification run on `tmp/final.mapped.json`
- After the three fixes above, ran:
  `python scripts/run_pwml.py --in tmp/final.mapped.json --out-dir tmp/pwml_test --non-strict-db`
- `pwml_validation_issue_count` dropped from 25 (pre-fix, see 2026-06-09 entry
  below) to **0**.
- `pwml_qa_ok: false` remains, but the single QA error
  (`Reaction '<unnamed>' has no enzyme and is not spontaneous` — corresponds to
  reaction index 2, "Obafluorin assembly from AHNB and 2,3-DHBA") is a
  **pre-existing data gap** in `tmp/final.mapped.json` itself (that reaction
  has `enzyme: None` and `spontaneous: None` in the source JSON) — unrelated to
  the IR fixes and not a regression.
- `pytest tests/test_pwml_ir.py tests/test_pwml_writer.py -q` — 40 passed, 4
  pre-existing failures unchanged (verified via `git stash`/`git stash pop`
  that they fail identically without the new fixes too).

### 2026-06-09 — `outputs/pathway.pwml` (nosiheptide/obafluorin pathway)
- `validate.py` against `reference/PW000001.pwml` reports 25 issues, all of the
  same two kinds:
  - Missing `pwbs-id` on biological-states 4, 5, 6 (`cytosol_Pfluorescens`,
    `Streptomyces cytosol`, `__auto_state__`).
  - Missing `pwc-id` + `short-name` on every novel compound (ids 20000-20009:
    4-nitrophenylacetaldehyde, AHNB, obafluorin, L-Trp, 3-methyl-2-indolic acid,
    MIA-AMP, MIA-S-NosJ, indoyl-Cys/C4-methylene pentathiazolyl intermediates,
    nosiheptide, pentathiazolyl nosiheptide intermediate).
  - `__auto_state__` again present as biological-state 6.
  - No duplicate compound ids in this file (checked all 22 compounds).
- Same root causes as above — not yet fixed.
