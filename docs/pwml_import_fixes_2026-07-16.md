# PWML Import & Enzyme-Rendering Fixes — 2026-07-16

This document records every change made while debugging why a generated
Herbaspirillum pathway (a) rendered its enzymes wrong and (b) failed to upload
into the local PathWhiz (`smpdb`) instance with a `500`. It explains what each
change does, why it is needed, and why it does not affect anything else.

Two repositories are touched:

- **T2PW** (`Project14-T2PW`) — the PWML generator.
- **smpdb** (`../smpdb`) — the Rails PathWhiz app that imports PWML.

---

## 0. Why were we only seeing this now?

The upload failure needs **two conditions at once**:

1. **The target PathWhiz DB already contains the entity under a canonical
   name.** Base PathBank seed data provides low-id canonical compounds
   (e.g. compound `73 = "Glycolic acid"`, `721 = "NAD"`), and earlier uploads
   accumulate more rows (e.g. species `12220 = "Herbaspirillum huttiense IAM
   15032"`, created by a *previous* T2PW run).
2. **T2PW emits a *different* display name for that same entity** (e.g.
   `"glycolate"`, `"NAD+"`, `"Herbaspirillum huttiense"`).

When both hold, the importer's `find_by(name + all ids)` misses the existing
row (name differs), tries to `INSERT` a duplicate, and the insert is rejected
by a **unique-key** validation (`compounds.hmdb_id`, `species.taxonomy_id`).
The record is dropped, and everything that referenced it becomes `nil` →
protein-complexes lose their species → reactions lose their enzymes → the
whole pathway fails to save.

It stayed hidden until now because condition 2 **drifts between runs**:

- **Compound names** are only rewritten to the DB's canonical form when
  T2PW's resolver reaches a live resolution DB *during generation*. If a run
  is offline (`db_not_configured`) or the DB lacks the row, T2PW keeps the
  extraction/LLM name (`"glycolate"`). So whether names come out canonical
  depended on run-time DB connectivity — not deterministic across runs.
- **Species names** are produced by extraction/resolution and are not
  deterministic: one run emitted `"Herbaspirillum huttiense IAM 15032"` (which
  got stored in the DB on that upload), a later run emitted the shortened
  `"Herbaspirillum huttiense"`.

Early uploads therefore either ran with canonical names (matched cleanly) or
hit a DB that did not yet contain the colliding rows (so the emitted name was
created fresh). As pathways were regenerated and re-uploaded, the DB
accumulated canonical rows **and** T2PW's emitted names drifted — so the
mismatch became guaranteed.

Finally, it was **masked**: when the collision first happened, the importer's
own error-cleanup code crashed with `NoMethodError` (see §4) and returned an
opaque `500` instead of the real "name/taxonomy already taken" message — so
the true cause was invisible until traced by hand.

---

## 1. Enzyme label shows the complex name — T2PW

**File:** `src/t2pw/pwml/writer.py` (~line 1576, complex-member protein-location
creation) — `label-type` changed from `"subunit"` to `"protein"`.

**Why:** PathWhiz's canvas renderer
(`smpdb/app/assets/javascripts/backbone/models/visualization/protein_location.js.coffee`,
method `text`) draws `label_type == "subunit"` → the *protein's* name, and
`label_type == "protein"` → the *protein-complex's* name. Every enzyme in this
pipeline is a single-protein complex whose meaningful identifier is the complex
name (the `C785_RS…` locus tag), so nodes were showing the inner protein name
(`"Unknown"`, `"Uncharacterized protein"`, …) instead. Real PathWhiz uses this
same `protein` label-type for its Unknown-sentinel protein
(`reference/PW012926.pwml`).

**Doesn't affect anything else:** bare-protein enzymes (not wrapped in a
complex) are created on a *different* path (`ensure_protein_location`,
~line 1731) and intentionally keep `"subunit"` — for them there is no complex
name, and `"protein"` would render the literal `"Unknown"` fallback. Test
`tests/test_pwml_writer.py::test_writer_emits_visible_complex_and_reaction_enzyme_visualization`
was updated to expect `"protein"` for the complex-member case. Full detail in
`pwml_coordinate_mapping.md` §2G.

---

## 2. Unknown-sentinel enzymes get their own boxes — T2PW

**File:** `src/t2pw/pwml/writer.py` (~line 1558, inside the
`protein_complex_visualizations` build loop). When the component protein is the
PathBank **Unknown sentinel** (detected via
`is_pathbank_unknown_protein(...)`), a **fresh** protein-location is minted per
complex instead of reusing the shared `(protein_key, biological_state)` entry,
and it is not registered in the shared entity-state caches.

**Why:** the Unknown sentinel backs many unrelated generated enzyme complexes.
Deduping its protein-location by protein identity collapsed all of them onto
**one** box, so every enzyme edge converged on a single point and the layout's
"S" shape was destroyed. Real PathWhiz emits a separate protein-location per
usage (several id-9659 locations at distinct coordinates in
`reference/PW012926.pwml`).

**Doesn't affect anything else:** the fresh-location branch is gated strictly
on the Unknown sentinel; real proteins keep normal de-duplication (a genuinely
shared enzyme still gets one box). Verified end-to-end that N distinct
unresolved enzymes now produce N boxes at N distinct serpentine coordinates.
Full detail in `pwml_coordinate_mapping.md` §2F.

---

## 3. Emit canonical DB names when an entity resolves — T2PW

**Files:**
- **new** `src/t2pw/pwml/name_index.py` — `PathwhizNameIndex`, loads
  `data/pathwhiz_id_db.json` and maps a compound external id (hmdb/kegg/chebi/
  pubchem/drugbank) or a species taxonomy id to the canonical PathWhiz name.
- `src/t2pw/pwml/ir.py` — late canonicalization passes in `build_pwml_ir`: when
  a compound/species row has no live-DB canonical name but its ids hit a real
  DB row in the index, its `name` (and a minimal `db_row`) are set to the
  canonical value; provenance recorded under `report["name_canonicalization"]`.
- `src/t2pw/mapping/build_pathwhiz_id_db.py` — now also captures `<species>`
  rows so the index JSON can carry species names once regenerated.

**Why:** the writer already prefers `db_row.get("name")` over the extraction
name (`writer.py:1069`), but `db_row` is only populated when the resolver
reaches a live DB during generation. This adds an offline fallback so canonical
names are emitted even on offline runs — which is the actual §0 drift.

**Doesn't affect anything else:** the pass only fires when an id hits a real
row in the index; novel/unresolved entities (`"L-KDP"`, `"αKGSA"`) are left
untouched. A live-DB `db_row` name always wins over the index. If the index
file is missing or unreadable it silently disables (never breaks export).

> **IMPORTANT — this change is inert until its data exists.** The shipped
> `data/pathwhiz_id_db.json` currently has ~29 compounds and **no** species,
> and does **not** contain the failing cases (`HMDB0000115`, taxonomy `863372`).
> Until it is regenerated from the full target PathBank corpus, this mechanism
> does not fix real uploads on its own — which is why §4 (the importer fix) is
> the actual remedy. This layer is a correct, tested foundation for later.

---

## 4. Importer: match shared entities by unique id, and stop crashing on failure — smpdb

**File:** `../smpdb/lib/pwml_parser.rb`. This is the change that actually fixes
the upload, independent of any T2PW name drift.

### 4a. Resolve compounds/species by their unique external id
New helper `find_by_unique_external_id(klass, attribute_hash)` (just above
`store_object`) and its use inside `store_object`:

```ruby
existing = find_by_unique_external_id(klass, attribute_hash) || klass.find_by(attribute_hash)
```

- `Compound` → looked up by `hmdb_id` (a unique column) when present.
- `Species` → looked up by `taxonomy_id` (a unique column) when present.
- **Everything else, and any compound/species without that id → returns `nil`**,
  so it falls straight through to the original `klass.find_by(attribute_hash)`.

**Why:** `hmdb_id` and `taxonomy_id` uniquely identify the compound/species
regardless of display name. Matching on them first means a name-only
difference reuses the existing canonical row instead of attempting a duplicate
insert that dies on the unique key. It fixes **all** current and future
name-drift collisions in one place, with no dependency on a name data file.

**Doesn't affect anything else:** the new lookup runs *before* the existing one
and returns `nil` for every type except compounds/species that actually carry
the id. When it returns `nil`, behaviour is byte-identical to before. The only
behavioural change is that a formerly-**failing** collision now succeeds by
reusing the canonical row — and because the id is unique, that row is by
definition the same entity, so nothing that previously worked changes.

### 4b. Nil-guard the failed-import cleanup
In `parse`, the `else` branch that runs when `super_pathway_visualization.save`
returns false:

```ruby
pvc.pathway_visualization.try(:pathway).try(:destroy)   # was: .pathway.try(:destroy)
```

**Why:** on a failed save `pvc.pathway_visualization` can be `nil`; calling
`.pathway` on it raised `NoMethodError`, crashing the cleanup with a `500` and
hiding the real validation errors assigned two lines below
(`@result[:errors][:super_pathway_visualization] = ...`). This is why the true
cause was invisible (§0).

**Doesn't affect anything else:** this line only executes in the
already-failing branch. Successful uploads never reach it. It converts a crash
into the intended graceful cleanup + real error report.

---

## 5. How to apply the smpdb change

The Rails app loads `lib/pwml_parser.rb` at boot, so the container must be
restarted (a full rebuild is not required — the repo is bind-mounted at `/app`):

```bash
cd ../smpdb
docker compose restart app     # picks up the edited lib/pwml_parser.rb
```

Then re-upload the pathway. `ruby -c lib/pwml_parser.rb` reports **Syntax OK**.
No database migration is involved; no schema, model, or other importer path is
changed.

---

## 6. Change inventory (quick reference)

| # | Repo | File | Change | Risk |
|---|------|------|--------|------|
| 1 | T2PW | `src/t2pw/pwml/writer.py` ~1576 | complex-member `label-type` → `protein` | none (bare proteins unchanged) |
| 2 | T2PW | `src/t2pw/pwml/writer.py` ~1558 | fresh protein-location per Unknown-sentinel complex | none (gated on sentinel) |
| 3 | T2PW | `src/t2pw/pwml/name_index.py` (new), `ir.py`, `mapping/build_pathwhiz_id_db.py` | offline canonical-name index + IR passes | none (inert without index data) |
| 4a | smpdb | `lib/pwml_parser.rb` | match Compound/Species by unique id first | none (nil for all other cases) |
| 4b | smpdb | `lib/pwml_parser.rb` | nil-guard failed-import cleanup | none (failure path only) |

All T2PW test suites pass (`418 passed`); `lib/pwml_parser.rb` passes
`ruby -c`. Items 1, 2, 4a, 4b are effective immediately; item 3 needs its
index JSON regenerated from the target corpus to become effective.
