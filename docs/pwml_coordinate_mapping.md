# PWML Coordinate & Layout Mapping — Ground Truth and Known Issues

Investigation date: 2026-07-15. This document is dedicated to one topic:
whether T2PW's generated (x, y) coordinates, compartment boxes, and edges
will actually render correctly once imported into PathWhiz.

`docs/pathwhiz_requirements.md` §3 already contains a coordinate/layout audit
(three duplicate layout algorithms, no collision avoidance, canvas not
horizontally auto-scaled, fuzzy compartment-name matching, disconnected QA
renderer). That audit is still accurate and is summarized/cross-referenced
below rather than repeated. This document adds a layer that audit did not
have access to: **full ActiveRecord schema and model ground truth from the
live `smpdb` app** (`../smpdb/db/schema.rb`, `../smpdb/app/models/**`), not
just the standalone copy of the importer kept at `tools/ruby/pwml_parser.rb`.
That copy is older than the live one — it doesn't even contain the
negative-coordinate offset fix described below — and has never had the Rails
schema/models next to it, so several structural issues were invisible to the
earlier audit. They're the headline findings here.

Source files referenced (all under `../smpdb`, sibling to this repo):

- `lib/pwml_parser.rb` — the canonical PWML importer.
- `db/schema.rb` — real table columns.
- `app/models/visualization/bound_visualizations/bound_visualization.rb`
- `app/models/data/bounds/bound.rb`
- `app/models/visualization/membrane_visualization.rb`
- `app/models/visualization/edge.rb`
- `app/models/visualization/protein_complexes/protein_complex_visualization.rb`

---

## 1. Ground truth: how PathWhiz actually models pathway geometry

PathWhiz splits geometry into two unrelated mechanisms. T2PW currently
conflates them.

### 1a. Per-element (x, y) — real, stored, validated

`compound_locations`, `protein_locations`, `nucleic_acid_locations`,
`element_collection_locations` all have real `x`, `y`, `biological_state_id`
columns (`db/schema.rb:130-141` for `compound_locations`, similarly for the
others). This is the part T2PW's `ir.py`/`writer.py` gets right for reaction
participants: an entity's compartment membership is carried correctly as
`biological_state_id`, and the importer stores whatever `x`/`y` it's given
verbatim.

`protein_complex_visualizations` (`db/schema.rb:637-643`) has **no** `x`/`y`
columns at all. Its visual position is computed on read, not stored:
`ProteinComplexVisualization#x`/`#y` (`protein_complex_visualization.rb:117-134`)
take the **min over its constituent `protein_complex_protein_visualizations`'
`protein_location.x`/`.y`**. T2PW already does the compatible thing here — it
positions each subunit's `protein-location` and never tries to write an x/y
onto the complex visualization itself. This part is fine.

### 1b. Compartment/complex "boxes" — two different, non-interchangeable mechanisms

**`Bound` / `BoundVisualization` is a *molecular complex* concept, not a
spatial compartment.** `Bound` (`app/models/data/bounds/bound.rb:14-17`)
requires at least 2 `bound_elements` and validates `has_macromolecule` — it
represents something like "this compound is non-covalently bound to this
protein complex," the same category as `protein_complexes` but mixing
compound/nucleic-acid/protein-complex members. `bound_visualizations`
(`db/schema.rb:102-108`) has only `bound_id`, `biological_state_id`,
`pathway_visualization_id` — **no `x`, `y`, `width`, or `height` columns**.
Its `x`/`y` are computed the same way as protein complexes: min over its
member `bound_compound_visualizations` / `bound_protein_complex_visualizations`
/ etc. (`bound_visualization.rb:122-164`), which are themselves real join
tables (`bound_compound_visualizations`: `bound_visualization_id`,
`compound_location_id`, `edge_id` — `db/schema.rb:42-48`) that must be
populated for the box to mean anything. `bound_id` is a required, validated
field (`bound_visualization.rb:8`).

**The actual mechanism for drawing a compartment boundary/box on the canvas
is `membrane_visualizations`**, not `bound_visualizations`.
`membrane_visualizations` (`db/schema.rb:441-450`) stores a `path` (SVG path
string defining the boundary shape), a `visualization_template_id` (selects
plasma/nuclear/mitochondrial membrane style, etc.), and a `complete_membrane`
flag. `MembraneVisualization#contains?(x, y)` (`membrane_visualization.rb:90-102`)
is the actual point-in-compartment test PathWhiz uses, and `get_x`/`get_y`/
`get_width`/`get_height` (lines 34-88) derive the bounding box by parsing the
path — this is also what feeds SBGN compartment export
(`sbgn_instance`, lines 111-127).

**T2PW currently writes per-biological-state boxes as `bound-visualization`
elements with `x`/`y`/`width`/`height`** (`ir.py:1181-1193`, resized/repositioned
by `writer.py:1645-1690`) **and never emits any `membrane-visualization` at
all** (`writer.py:2397` initializes `self.section_items["membrane-visualizations"]
= []` and nothing ever appends to it). See §2A below for the consequence.

### 1c. Edges and membranes carry coordinates as baked SVG path strings, not x/y

`edges` (`db/schema.rb:256-265`) has a `path` column (e.g.
`"M11 12 C34 28 74 55 98 72"`) and **no `x`/`y` columns whatsoever** — same
for `membrane_visualizations`. Any endpoint coordinates are baked directly
into the path string at write time.

### 1d. The importer's negative-coordinate fix does not cover path-based records

`lib/pwml_parser.rb` normalizes negative coordinates before import:
`compute_coordinate_offset` (lines 778-803) scans `compound-locations`,
`protein-locations`, `element-collection-locations`, `nucleic-acid-locations`,
and `bound-visualizations` for the most-negative x/y, and
`apply_coordinate_offset` (lines 806-814) shifts every subsequently-imported
node's `:x`/`:y` attribute hash keys by that amount. **This offset is only
ever applied to hash keys literally named `:x`/`:y`.** Since `Edge` and
`MembraneVisualization` store geometry as a `path` string, not `:x`/`:y`
keys, **their baked-in coordinates are never shifted**. If any location in a
PWML file has a negative coordinate, every node shifts into positive canvas
space but every edge (and any membrane boundary) stays exactly where it was
— edges visibly detach from the nodes they're supposed to connect on import.
Also note `protein-complex-visualization` and other x/y-bearing nodes outside
the five scanned `location_xpaths` are shifted by `apply_coordinate_offset`
(it's applied generically to anything with `:x`/`:y`) but are **not** included
when computing the offset itself — so the computed offset can undershoot if
the most-negative coordinate lives on one of those excluded node types.

### 1e. The importer is fail-soft, per-record

`store_nested_visualization` (lines 817-878) and
`build_nested_visualization_recursive` (lines 925-997) catch save/validation
failures per node and record them into `@result[:errors]` without aborting
the overall import. A `bound-visualization` that fails validation (e.g.
missing required `bound_id`) does not fail the pathway import — it silently
fails to appear in the diagram, and the failure is visible only in the
import result's error hash, not in T2PW's own pipeline output. This matches
what `docs/pathwhiz_requirements.md` §3 already noted generically ("bad/missing
coordinate on one node causes that one visual element to silently drop") —
§2A below is the specific instance of that failure mode this document found.

---

## 2. Confirmed issues in T2PW against this ground truth

### 2A. [Critical] `bound-visualization` is being used for the wrong PWML concept

**Where:** `ir.py:1181-1193` builds one `bound_visualizations` entry per
`biological_state` with `x`, `y`, `width`, `height` meant to represent that
compartment's canvas region. `writer.py:1645-1690` resizes/repositions these
per actual reaction density.

**Why it's wrong:** per §1b, `BoundVisualization` has no `x`/`y`/`width`/
`height` columns (they'd be silently dropped by
`to_hash_shallow(node).slice(*attribute_list(klass))` on import even if
present), and every `BoundVisualization` requires a `bound_id` referencing a
real `Bound` (a molecular complex with ≥2 elements including a macromolecule)
— which T2PW never creates or references (no `bound_key`/`bound_id` field
appears anywhere in `ir.py`'s `bound_visualizations` construction). `Bound`
validates `bound_id` presence (`bound_visualization.rb:8`).

**Consequence:** every `<bound-visualization>` T2PW emits will fail
`BoundVisualization.new(...).save` on import (missing required `bound_id`)
and be recorded as a per-element import error (§1e) — it will not render.
The entire per-compartment region-resize/reposition system in `writer.py`
(§2B) is computing geometry for an XML element that PathWhiz's importer
rejects outright, independent of whether the computed geometry is even
correct.

**What compartments should use instead:** `membrane-visualization` entries
(a `path` describing the compartment boundary shape + a
`visualization_template_id` for the membrane style + `complete_membrane`).
T2PW currently never emits these (`writer.py:2397`, always empty) — there is
currently no code path that produces a real, importable compartment boundary
at all.

### 2B. [High] Per-compartment region reflow only covers reactions

**Where:** `writer.py`'s `state_region_by_key`/`_serpentine_reaction_position`
system (`writer.py:1645-1709`) reflows reaction participants and enzymes into
freshly-computed, region-correct locations via `add_reaction_member_location`
(`writer.py:1797-1835`) and `ensure_protein_location`
(`writer.py:1711-1738`).

**What's excluded:**
- **Transports** — `transport_visualization` construction
  (`writer.py:2351-2389`) does a plain `lookup("locations", member.get("location_key"))`,
  keeping whatever `ir.py`'s naive, compartment-blind `process_xy()`
  (`ir.py:1234-1245`) originally computed. Since `process_xy` positions
  everything by **global index across the combined reaction+transport list**,
  not by which `biological_state` it belongs to, and since compartment
  regions in `writer.py` are resized to fit reaction density
  (`_reaction_region_height`, `writer.py:631-641`), a transport's coordinates
  can end up outside its own compartment's region once compartments have
  uneven reaction counts — a very common case, since transports typically
  connect a reaction-heavy compartment to a reaction-light or reaction-free
  one (e.g. "extracellular space").
- **Reaction-coupled transports** — `ir.py:1595-1609` explicitly notes
  visualization is "left explicit" (unimplemented); no coordinates are
  produced for these at all.
- **Orphan entities** (from `element_locations` not touched by any
  reaction/transport) — placed by a static fallback grid at
  `y = height * 0.75 + ...` (`ir.py:1579-1593`) using the pre-reflow canvas
  height, never adjusted for the fact that `writer.py` can grow the canvas
  taller than that (`writer.py:1669-1673`) to fit reaction-dense
  compartments.

(This item stands regardless of §2A — even once compartment visualization is
fixed to use `membrane-visualization`, transports/RCTs/orphans still need
their own reflow pass to land inside the correct compartment boundary.)

### 2C. [High] Interactions get zero coordinates or visualization

**Where:** `ir.py:1547-1578` builds each `interaction` process record and
appends it to `ir["processes"]["interactions"]`, but — unlike the reaction
and transport loops immediately above it — never calls `ensure_location` for
its `left`/`right` entities and never appends anything to
`ir["process_visualizations"]`.

**Consequence:** interactions are exported with a valid data record
(entities, `interaction_type`) but no visual placement whatsoever. Nothing
catches this: the IR validator's visualization-consistency check
(`ir.py:2686-2725`) only validates visualizations that *exist* against their
process's bucket — it never asserts every process *has* a visualization. An
interaction-only pathway (e.g. "ATP inhibits phosphofructokinase" with no
reaction) will currently produce a PWML file where the interaction is present
in the data but invisible in the rendered diagram.

### 2D. [Medium] No compartment boundary shape is ever produced

Direct consequence of §2A: `membrane-visualizations`
(`writer.py:2397`) stays permanently empty. Even if every other coordinate in
the file were perfect, PathWhiz would render a multi-compartment pathway with
no visible compartment boundaries at all — just entities positioned at
different y-ranges with nothing outlining the regions between them.

### 2E. [Medium, latent] Negative coordinates would desync edges from nodes on import

Per §1d, if any T2PW-computed coordinate goes negative, the importer shifts
every location but not `Edge.path`. This has not been confirmed to currently
trigger — `process_xy`'s `margin_left = 380` (`ir.py:1237`) and the orphan
grid's `x = 40 + ...` (`ir.py:1591`) are both always positive by
construction, but `writer.py`'s serpentine layout
(`_serpentine_reaction_position`, `writer.py:644-669`) computes
`enzyme_cx = region.x + region.w - pad - rxn_step_x // 2 - col_in_row * rxn_step_x`
on alternating ("boustrophedon") rows, which is subtractive and has no floor
check — a narrow region (few reactions, e.g. the minimum-height regions
described in §2B) combined with multiple columns could plausibly drive this
negative. Worth a bounds-check test rather than assuming it can't happen.

### 2F. [Critical] Unknown-sentinel enzymes collapse onto one shared protein-location, destroying the serpentine layout

**Observed on:** `outputs/pathway (81).pwml` (Herbaspirillum huttiense
xylose/arabinose degradation, 10 reactions, one source paper). Screenshot
symptom: only **5 enzyme boxes render** for 10 reactions, and a fan of ~12
golden enzyme edges all converge on a single green "Unknown" ellipse in the
bottom-right corner (loc at x=2695, y=2445), cutting diagonally across the
whole canvas. The user's intended boustrophedon ("S") layout is intact for
compounds but visibly destroyed for enzymes.

**What is actually correct in this file:** the *compound* serpentine is fine.
`compound-locations` step left→right on row 1 (y≈255), right→left on row 2
(y≈931–985), left→right on row 3 (y≈1631–1685), then down to row 4 (y≈2275+)
— a proper S. The 4 enzymes that got real protein identity are also placed
correctly on the S: protein-location 91 (`C785_RS00860`, Q6MEV0) at
(355,345) row 1; 92 (`C785_RS13685`, A0A1G8J7Q2) at (1895,1045) row 2; 93
(`C785_RS13675`, A0A090Q0Z5) and 94 (`C785_RS20555`, Q00796) at row 3. So
the serpentine layout math (§2B) is **not** the culprit here.

**Root cause — a mapping-stage identity collapse, not a layout-math bug:**

1. Upstream, 6 of the 10 reaction enzymes (`C785_RS00855`, `C785_RS21245`,
   `C785_RS21250`, `C785_RS13680`, `C785_RS13710`, `C785_RS20550`) fail every
   normal protein-identity strategy and fall through to the **Stage-6 PathBank
   Unknown-sentinel fallback**, `_apply_pathbank_unknown_enzyme_fallback`
   (`map_ids.py:4861`). That fallback creates **exactly one** shared sentinel
   protein for the whole pathway: `_ensure_unknown_protein`
   (`map_ids.py:5042-5112`) memoizes it in `nonlocal unknown_row` and always
   reuses the single `_PATHBANK_UNKNOWN_PROTEIN_ID` row (name "Unknown",
   uniprot "Unknown", species = PathBank Unknown/Arabidopsis id 4). Each
   enzyme keeps its *functional name* on its own generated protein-complex
   (that's why complexes 40004–40009 have distinct names), but every one of
   those complexes lists the **same** Unknown protein (id 9659) as its sole
   component.

2. In the writer, protein-locations are keyed by `(protein_key,
   biological_state_key)` in `ensure_protein_location`
   (`writer.py:1711-1738`). Because all 6 complexes share one protein entity
   key, they all resolve to **one** protein-location (id 98). Their 6 distinct
   `protein-complex-visualization`s (95/97/100/102/106/108/114 in the file)
   therefore all point their single
   `protein_complex_protein_visualization` at protein-location 98.

3. `ProteinComplexVisualization#x/#y` on the PathWhiz/smpdb side is the *min
   over its members' `protein_location.x/.y`* (§1a,
   `protein_complex_visualization.rb:117-134`). With one shared member, all 6
   complex visualizations compute the **identical** position — wherever loc 98
   currently sits.

4. Worse, loc 98's position is **overwritten once per reaction that uses it**.
   The enzyme-placement loop (`writer.py:1765-1784`) walks reactions in id
   order and, for each, sets `loc["x"]/loc["y"]` from that reaction's
   serpentine slot (`_serpentine_reaction_position`). Since 6 reactions write
   the *same* loc 98, **last-writer-wins**: the final coordinate is whatever
   the last Unknown-enzyme reaction (50009, `C785_RS20550`, in the bottom row)
   computed — (2695, 2445), bottom-right. Every earlier write is discarded.

**Net effect:** the 4 real enzymes serpentine correctly; the 6 Unknown-
sentinel enzymes are silently merged into one box parked at the bottom-right,
and every one of their reaction→enzyme edges (which are still baked per
reaction from each compound's correct S-position) is drawn to that single
point — producing the cross-canvas fan and the "only 5 enzymes" appearance.

**Why this is distinct from §2A–§2E:** those items are about compartments,
transports, interactions, and negative coordinates. This one is a *node-
identity* problem: two layers each behave "correctly" in isolation (the
mapping fallback dedups to one sentinel by design; the writer dedups
protein-locations by entity key by design) but compose into a layout
collapse. Any pathway with ≥2 unresolved enzymes will show it; it gets worse
linearly with the number of Unknown-sentinel enzymes.

**Fix (implemented 2026-07-16 — "Approach A", sentinel-scoped):** the writer's
complex-member protein-location creation (`writer.py:1558` region, inside the
`protein_complex_visualizations` build loop) now detects the Unknown sentinel
via `is_pathbank_unknown_protein(raw_entity_by_key[protein_key])` and, when the
component protein **is** the sentinel, **mints a fresh protein-location per
complex-visualization** instead of reusing the shared
`(protein_key, biological_state)` entry — and does not register that location in
the `protein_location_by_entity_state` / `location_by_entity_state` caches, so
no later complex reuses it. Real proteins keep their normal dedup (a genuinely
shared enzyme still gets one box). This mirrors real PathWhiz, which emits
several distinct protein-locations all pointing at protein id 9659 at different
coordinates (`reference/PW012926.pwml`).

Result (verified end-to-end through `map_payload → normalize → build_pwml_ir →
DeterministicPwmlBuilder`): N distinct unresolved enzymes now yield N distinct
protein-locations at N distinct serpentine coordinates, each labeled
`label-type: protein` (§2G) so each shows its own complex name — instead of one
shared box carrying a single name with every enzyme edge converging on it.

Scope notes / not done: this is deliberately **sentinel-only** ("Approach A").
A real enzyme that legitimately catalyses two different reactions still shares
one box (one protein identity → one location); if per-reaction boxes are ever
wanted for real repeated enzymes too, that is a separate, broader change to the
dedup key. The alternative of minting a distinct sentinel *protein row* per
unresolved enzyme upstream in mapping was considered and not taken — keeping a
single sentinel entity (all pointing at the same PathBank Unknown DB id) is
correct for data identity and import; only the *visual* placement needed to
diverge, which is why the fix lives in the writer.

### 2G. [High] Enzyme nodes are labeled with the protein name, never the complex name

**Observed on:** same `outputs/pathway (81).pwml`. The enzyme ellipses in the
diagram read "Unknown", "Uncharacterized protein", "D-altronate dehydratase",
"CoG1028" — i.e. the **constituent protein's** name — never the
`C785_RS…`/`… complex` name that T2PW assigned to the enclosing
protein-complex (and that PathWhiz stores correctly, visible on each
complex's own detail page: PW_P040000–PW_P040009).

**Root cause — a label-type choice in the writer, not a data problem.** The
canvas label text is chosen entirely by the protein-location's `label_type`
(`smpdb/app/assets/javascripts/backbone/models/visualization/protein_location.js.coffee:128-144`,
method `text`):

- `label_type == "subunit"` → `@name()` → **`@protein.name()`** (the protein)
- `label_type == "protein"` → **`@protein_complex_visualizations.first().name()`** (the complex)
- `label_type == "gene"` → `protein.gene_name`; `"uniprot"` → `protein.uniprot_id`
- if the resolved text is empty, it hard-falls-back to the literal string
  `"Unknown"` (lines 141-142).

T2PW's `ensure_protein_location` (`writer.py:1711-1738`) writes
**`"label-type": "subunit"` unconditionally** on every enzyme protein-location
(`writer.py:1731`). So PathWhiz is explicitly told to label each enzyme node
with its constituent protein's name. For the §2F Unknown-sentinel enzymes that
protein's name is literally `"Unknown"`; for the resolved ones it is whatever
vague protein name mapping found (`Uncharacterized protein`, `CoG1028`, …) —
in **no** case the locus-tag complex name the user actually assigned.

Note the two other places that DO surface the complex name and are therefore
misleading as a sanity check: the complex **detail page** and the click-modal
(`getDisplayName`, `protein_location.rb:48-53`, which returns the complex name
when the location belongs to a complex). Only the on-canvas `text()` path is
governed by `label_type`, and only it is wrong.

**Relationship to §2F:** independent. §2F is *placement* (all sentinel
enzymes collapse to one location). §2G is *labeling* (the node shows the
protein name instead of the complex name) and affects **every** enzyme,
resolved or not — even a perfectly-mapped single-protein enzyme complex will
display its protein name rather than the assigned complex/locus-tag name.

**Fix (implemented 2026-07-16):** `writer.py`'s complex-member protein-location
creation (site at `writer.py:1576`, inside the
`protein_complex_visualizations` build loop) now emits **`label-type: protein`**
instead of `subunit`, so every protein that is a member of a complex is labeled
with the complex name (`protein_complex_visualization.name()` = the `C785_RS…`
locus-tag / enzyme name T2PW assigned) rather than the constituent protein's
name. This was a deliberate choice to prefer the complex name for **every**
complex, not just the Unknown-sentinel ones — the complex identifier is the
meaningful enzyme label in this pipeline whether or not the underlying protein
resolved. It also matches real PathWhiz's own convention: reference export
`reference/PW012926.pwml` uses `label-type: protein` for exactly the same
Unknown-sentinel protein id 9659.

The **bare-protein enzyme** path (`ensure_protein_location`, `writer.py:1731`,
reached only for `entity_type == "protein"` enzymes that are *not* wrapped in a
complex) intentionally **keeps `subunit`** — such a location has no
`protein_complex_visualization`, so `label_type == "protein"` would resolve to
empty and the renderer's hard fallback (line 141-142) would print the literal
`"Unknown"`. Test `test_writer_emits_visible_complex_and_reaction_enzyme_visualization`
(`tests/test_pwml_writer.py:709`) was updated to assert `protein` for the
complex-member case.

Note this is a *labeling* fix only; it does not resolve §2F (the sentinel
enzymes still collapse onto one shared location — they will now all read the
correct distinct complex name but still stack at one point until §2F is fixed).
**Not addressed:** the SBGN export path (`protein_complex_visualization.rb:214-234`
→ `protein_location.rb:221`) collapses a single-member complex to its lone
protein and labels via `protein.name` regardless of `label_type`, so SBGN still
shows the protein name; if complex-name labels matter for SBGN too, that path
needs its own change.

---

## 3. Already-known issues (cross-referenced, not duplicated)

From `docs/pathwhiz_requirements.md` §3 — still accurate, summarized here for
a single point of reference on this topic:

- Three independent, non-communicating layout algorithms exist for
  overlapping purposes (`ir.py` fixed-grid, `writer.py` region+serpentine —
  the only one production uses, and `sbml/add_pathwhiz_layout.py` for the
  legacy SBML path only), plus a third dead sibling
  (`_build_locations_and_visualizations`) kept alive only by tests.
- No collision/overlap avoidance anywhere in any of the three algorithms.
- Compartment classification for legacy SBML layout uses fuzzy substring
  matching against ~13 hardcoded strings.
- Canvas width never auto-scales (only height grows with content).
- `pwml/render.py` (the PWML-specific QA preview renderer) is not wired into
  `streamlit_app.py` or any pipeline module — there is no automatic visual
  sanity check for the primary PWML export path.
- No geometry QA checks exist anywhere (`pwml/qa.py` is purely structural).

---

## 4. Suggested fix priority (informational — not started)

1. **§2A** first — it's the reason the entire region-layout system in
   `writer.py` currently produces output the importer rejects. Replace
   `bound-visualization` compartment boxes with real
   `membrane-visualization` records (a rectangular `path` per compartment
   region is sufficient to start; `complete_membrane = true` so PathWhiz's
   `contains?` and SBGN export both work).
2. **§2C** — give interactions a location/edge/`process_visualizations`
   entry the same way reactions and transports get one; otherwise
   interaction-only pathways render as empty canvases.
3. **§2B** — extend the `writer.py` region reflow to transports and orphan
   entities once §2A defines what a "region" should mean geometrically.
4. **§2E** — add a bounds check/test asserting no emitted coordinate goes
   negative, given the importer will not correct edges to match if it does.
5. Everything in §3, per the existing prioritization in
   `docs/pathwhiz_requirements.md`.
