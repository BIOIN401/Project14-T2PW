# PathBank drawing conventions (mined 2026-11-19)

Source: the PathBank MySQL database, read-only, through the repo helper `t2pw.bench.eval_v2.pathbank_gold.connect()`.
No credentials printed. Scripts and outputs: `docs/pwml_output/evidence/draw_pathwhiz/db/`.

## Method

- **Sample** (`db2_sample.py`): for each of the 12 membrane templates, the 600 newest visualizations that use it
  (highest id first), then **de-duplicated**: a visualization whose set of membrane paths is identical to one already
  taken is a propagated copy and is skipped. Up to 25 distinct diagrams per template: **268 diagrams**, updated
  2026: 73, 2025: 101, 2024: 6, 2023: 45, 2019-2022: 43. Propagation is heavy: for Mitochondria Membrane only 14 of
  the 600 newest visualizations were distinct.
- **Weighting** (`db3_stats.py`): updated 2023 or later = 1.0, 2019-2022 = 0.5, older = 0.25. Medians below are
  weighted medians; counts are weighted counts.
- **Global counts** (`db4_global.py`): all 1,596,320 membrane rows by template (no de-duplication, so dominated by
  propagated copies). Used only to show which templates exist in practice.

## 1. Membrane template and colour per compartment

PathWhiz draws template 220 ("Simple Membrane") as a 6-unit black line and every other membrane template as a
**50-unit band** in the template's colour (`plasma_membrane.jst.eco`, `stroke-width="50"`).

| compartment | template id | name | colour | sample (weighted) | global rows | labels found next to it |
|---|---:|---|---|---:|---:|---|
| cell / plasma, inner, outer membrane | 1 | Plasma Membrane | `#4BC8F2` | 384 | 588,916 | cytosol, plasma membrane, extracellular space, periplasm |
| mitochondrion | 84 | Mitochondria Membrane | `#FF0000` | 99.5 | 848,792 | mitochondria, outer/inner mitochondrial membrane |
| nucleus | 93 | Nucleus Membrane | `#FEF935` | 99.5 | 805 | nucleus |
| chloroplast / plastid | 237 | Chloroplast Membrane | `#389F3C` | 77 | 231 | chloroplast, thylakoid |
| endoplasmic reticulum | 91 | Endoplasmic Reticulum Membrane | `#EE9189` | 76 | 106,072 | endoplasmic reticulum, sarcoplasmic reticulum |
| peroxisome (and, in bacteria, the peptidoglycan layer) | 94 | Peroxisome Membrane | `#DDB0FD` | 50 | 17,528 | peptidoglycan 30, periplasm 21, peroxisome 3 |
| vacuole, **vesicles** | 238 | Vacuole Membrane | `#7CC4B7` | 46.5 | 2,406 | synaptic vesicle 21, COPII vesicle, phagosome |
| plant cell wall | 304 | Cell Wall Membrane | `#ABF7B1` | 42 | 77 | cell wall (encloses a Plasma Membrane 40/42) |
| lysosome | 92 | Lysosome Membrane | `#AED02A` | 39.5 | 109 | lysosome |
| Golgi apparatus | 146 | Golgi Membrane | `#FF7000` | 33.5 | 86 | Golgi apparatus |
| other | 220 | Simple Membrane | `#000000` | 29 | 82 | GLUT4 storage vesicles, ad hoc |
| peptidoglycan | 291 | Peptidoglycan Membrane | `#DDB0FD` | 8 | 31,151 | periplasm, peptidoglycan |
| melanosome | 95 | Melanosome Membrane | `#92130A` | 8 | 65 | (used for lipoprotein particles) |

The extracellular space has **no outline**: "extracellular space" is a label next to the cell's Plasma Membrane
(57 weighted hits). Template 220 ("Simple Membrane"), which T2PW used for every box, is rare (29 of 1,072).

## 2. Nesting

Smallest closed membrane that encloses each closed membrane (weighted):

- Plasma Membrane at the top level 255; organelles inside a Plasma Membrane: Mitochondria 46, Nucleus 44.5, ER 41.5,
  Lysosome 30.5, Vacuole 23.5, Chloroplast 23, Golgi 22.5, Peroxisome 10.
- Plasma Membrane inside a Plasma Membrane 43 (centre-to-centre gap median **204**, p25 97, p75 342): the Gram-negative
  envelope, outer membrane around inner membrane with the periplasm between. Plasma Membrane inside a Peroxisome
  Membrane 35 and a Peptidoglycan Membrane 8: the bacterial wall layer.
- Plasma Membrane inside a Cell Wall Membrane 40 (plants).

## 3. Double membranes

The same template drawn twice, one inside the other (weighted pairs, centre-to-centre gap):

| template | pairs | gap median | p25 - p75 |
|---|---:|---:|---|
| Nucleus Membrane | 41 | **70** | 64 - 79 |
| Mitochondria Membrane | 25 | **74** | 64 - 94 |
| Chloroplast Membrane | 20 | **94** | 74 - 95 |
| ER Membrane | 6 | 0 (re-drawn on top, not a double) | |

## 4. Transports across membranes

328.5 weighted transport visualizations; 212.5 with a drawn transporter.

- The transporter sits **on** the membrane line: distance from its centre to the nearest membrane path, median **8**
  units (p25 3, p75 35.5).
- Its cargo lies on **both sides** of that membrane in 136.5 of 142.5 cases (96 %).
- The transporter lies between its two cargo nodes in 170.5 of 208.5 (82 %).
- Cargo to cargo distance median 615; vertical crossings 113.5, horizontal 95.
- Membrane the transporter sits on: Plasma 110.5, Mitochondria 31.5, Vacuole 25, Lysosome 16, ER 13.5.

## 5. Spacing, chain shape, cofactor stubs

- Substrate centre to product centre of one reaction: median **667** (p25 535, p75 821). Axis: horizontal 537.5,
  diagonal 452.5, vertical 242.5 (weighted): chains run mostly left to right, with turns.
- Enzyme centre to the substrate-product midpoint: median 49.5: the enzyme sits on the arrow.
- Content aspect (width:height of the node bounding box): median **1.4** (p25 1.0, p75 1.9). Canvas median 4000 x 3150.
  Density 2.6 nodes per million square units.
- Edges: 50 % straight (control points within 5 units of the chord), median length 185.
- Cofactor stubs: 37 % of reaction members use a small template (water, H+, ATP, NAD(P)H, CO2, CoA, Pi, PPi ...),
  median **208** from the enzyme centre: short side stubs, not chain nodes.
- Labels: "Simple Label - Large" (235) 1016.5 and "Simple Label - Small" (15) 976; scale mostly 1.6, 1.0, 1.9, 2.2.
- Drawable images (cell pictures): Mitochondria 98, Endoplasmic Reticulum 59.5, Nucleus 54.5, Bacteria 49.

## What T2PW applies (layout v2, `src/t2pw/pwml/layout.py`)

Applied: §1 template per compartment kind (location name first, else the state name), extracellular without outline;
§2 nesting (organelles inside the cell, the Gram-negative envelope outer membrane > periplasm > inner membrane >
cytoplasm, leaflets inside their membrane, the matrix inside the mitochondrion); §3 double lines for nucleus (70),
mitochondria (74) and plastids (94); §4 a transporter that serves transports only is placed on the membrane between
its cargo compartments (between the two lines of a double membrane); §5 wider spacing (HGAP 110 -> 160, VGAP 40 -> 60,
target aspect 2.0 -> 1.5) and fewer crossings (24 barycenter sweeps, the order with the fewest crossings kept).

Not applied: drawable images (they are content in PWML and the harness has no image files), open membrane paths,
peptidoglycan layer lines (no T2PW state names one), cofactor stub geometry (existing currency templates already used).
