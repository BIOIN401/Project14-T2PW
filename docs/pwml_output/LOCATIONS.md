# Locations: every element and transport where the paper says (lane loc/assign, 2026-11-19)

Branch `loc/assign`, cut from main `fd814b8e`. **Biology change: merging into main waits for PO sign-off.** Code commit
`15a5b425`, review round 1 `56dd2803` (**section 8 supersedes rows 1b, 1e, 2 and 4 of section 1 and the numbers of
sections 2-3**), PO rulings `0cec290a` (**section 9 supersedes section 8's fix 3 and table; it holds the current
numbers**). Plan and pilot: `C:/t/strategy/location/STRATEGY.md`. Rulings: D-W21 (c, e), D-W24 (2, 7a, 9), and the
PO's approval of inferred locations that are always flagged and never override a stated one.

## 1. What changed

| # | Change | Where |
|---|---|---|
| 1a | L7 compares a transport's ends **by place** (location key, position words, cell type, tissue), not by state name. | `final_payload_compartments.py` |
| 1b | L7 never fills an **export's destination** from neighbouring reactions (name says export / secretion / efflux / excretion / extrusion). | same |
| 1c | **Leaflet spellings** share one location key: the membrane (word before "membrane", or IM / OM) plus the side words. Inner and outer membrane never merge (D-W21 e). | `final_payload_merge.location_key` |
| 1d | States whose only evidence is the Stage-0 "Expected compartments: ..." hint are **flagged** (`evidence_is_stage0_hint`, review reason). Not moved. | `final_payload_locations.py` |
| 1e | **`__auto__ -> __auto__` root cause**: `ensure_autostates` put the one placeholder on both ends of a transport with no stated side. The destination now gets its own placeholder (`__auto_state__ (other side)`); both stay unplaced and flagged. | `process_normalizer.py` |
| 2 | **Bacterial organism default**: unplaced reactions and element rows go to the organism's cytoplasm (an existing plain cytoplasm state, else a new `<organism> cytoplasm`); an unplaced side of a transport whose **name** says export / import goes to cytoplasm or extracellular space. Flag `inferred: organism_default`. Eukaryotes get no default. | same |
| 3 | **UniProt prior: not enabled.** Agreement with paper-stated locations is 77.9 % (< 80 %). See section 4. | none |
| 4 | **LLM location pass** behind `T2PW_LOCATION_LLM` (off by default): gemini-3.8-flash, L7's prompt, gate and cache; closed vocabulary per organism (PathWhiz locations only) plus the run's own stated states; the verbatim quote must name the place and the item (the item's synonyms count, e.g. "DHNA"); only unplaced items, or the estimated side of a transport whose ends are one place; never a stated location; the two ends never become one place. Flag `inferred: llm_quote`. | same |
| 5 | After steps 2 and 4, **L6's state merge runs again** (it guards transport ends itself), so a placeholder or synonym state that no transport end keeps apart now merges (menaquinone `cytoplasmic state` into `intracellular`, both the cytoplasm of a bacterium). | same |
| 6 | A transport whose two ends are still one place is flagged `transport_ends_same_place`. | same |

The new step (`final_payload_locations.assign_locations`) runs in `maybe_apply_final_id_fill` after L7 and before L8,
offline unless the LLM flag is on, fails open, runs once (`final_payload_locations` marks it). A live payload that
already carries L6/L7 gets this step alone, which also repairs an older L7 neighbour fill of an export destination.

**Flags.** Every placed row carries `inferred` (`inferred_sides` on a transport), `compartment_source`
(`organism_default` | `llm_estimate`), a `compartment_estimate` entry with the reason (and the quote) and a
`location_inferred: ...` review reason. The payload's `final_payload_locations` record lists every inference, every
Stage-0-hint state and every same-place transport left; the text record prints them under **INFERRED LOCATIONS (n)**
(absent when empty, so other reports keep their bytes). L8 checks that every `organism_default` row carries a reason.
**In the PWML** an inferred place follows the D-W24 (2) estimated-compartment convention: the row sits in that state;
the flag lives in the payload, the text record and our SBML, not in PWML (which has no field for it).

## 2. Gates: offline replay, 152 q-runs + 8 showcase runs (160 payloads, 105 PWML exports)

Harness `C:/t/loc-work/replay.py` (adapted from the import-fixes `replay97.py` and the showcase replay; no network; the
3 live showcase runs get the location step alone). Census `C:/t/loc-work/census.py`; arm diff `compare.py`; PWML A/B
`C:/t/pwml-fix-work/ab_replay.py`. Base = main `fd814b8e`.

| metric | base | deterministic | + LLM pass |
|---|---|---|---|
| same-place transports | 9 (8 runs) | 3 | **1** (idx 102, both ends stated) |
| `__auto__ -> __auto__` transports | 4 | 2 (sour taste, eukaryote) | **0** |
| duplicate state groups | 5 | 2 | 2 (by design, see Q3) |
| leaflet states | 14 | 10 | 10 |
| bacterial unplaced reactions | 35 (11 runs) | **0**, all 35 flagged | 0 |
| bacterial unplaced element rows | 59 | **0**, all flagged | 0 |
| unplaced reactions (all) | 119 | 84 (all flagged) | 75 (all flagged) |
| unplaced transport sides (transports) | 19 | 15 | 11 |
| placeholder / empty states | 60 | 57 | 57 |
| Stage-0-hint states flagged | 0 of 4 | 3 of 3 (menaquinone's merged away) | 3 of 3 |
| reactions / transports / element rows in payload | 692 / 128 / 1979 | 692 / 128 / 1979 | 692 / 128 / 1979 |
| **stated-location changes** | | **0** | **0** |
| inferred: organism_default (rx / transport side / element) | | 35 / 9 / 59 | 35 / 3 / 59 |
| inferred: llm_quote (rx / transport side) | | | 9 / 12 |

PWML A/B (deterministic arm): 105/105 exported, 98 byte-identical, 7 changed (138, 18, 25, 62, 78, lipid A, menaquinone).
**Lost names 0, lost reaction signatures 0**; reactions 488, transports 96, interactions 79, compounds 817, proteins 473
unchanged. Every listed change:

- **6 transport signatures relabelled, none lost** (96 transports before and after): 138 TBN export `cell->cell` ->
  `cytoplasm->Extracellular Space`; 78 enterobactin secretion `cytoplasm->cytoplasm` -> `cytoplasm->Extracellular Space`;
  menaquinone DHNA export `cytoplasm->intracellular` -> `intracellular->Extracellular Space`; lipid A three leaflet
  transports now carry one spelling per leaflet.
- **138**: its 8 unplaced reactions were drawn in the extracellular box (the IR's row-0 default); now in a new
  `Streptomyces tubercidicus cytoplasm` (+1 location, +1 compound location, +3 labels). 18 and 62: `cell` -> `cytoplasm`
  (+1 location). 78: +1 compound location (enterobactin outside), +1 membrane.
- **Rows that fell are duplicate boxes collapsing**, checked per (element, place) with `locpairs.py`: menaquinone
  compound+protein location rows 32 -> 22 with the 20 distinct (element, place) pairs unchanged; lipid A 69 -> 67, 65
  pairs unchanged; lipid A states 11 -> 7 (leaflet spellings), menaquinone 3 -> 2, 25: 2 -> 1 (no default state needed).

LLM arm: section 3.

**Tests.** Focused (new `tests/test_final_payload_locations.py` 16 tests; L6, L7, normalizer, autostate, reaction
evidence, final-payload seam): 245 passed. `suite2.sh C:/t/lane-loc loc-s1`: SMOKE 511, golden batch 2 / c011 8 /
stage8 4 / quarantine 24, equivalence 8 / 14 / 6, vision 357 (+1 skipped), web 83: **all passed, 0 survivors**. No
golden-seam test edited. Edited test: `test_process_normalizer_transport_states.py` (3 assertions pinned the
`__auto__ -> __auto__` this lane removes; one test added).

**PathWhiz import: not run.** Docker was down on this machine all session (`docker info` fails); nothing to import into.

## 3. LLM pass (gemini-3.8-flash)

Live replay of the 160 payloads with `T2PW_LOCATION_LLM=1` (census in the table of section 2): **0 stated-location
changes**; payload reactions 692, transports 128, element rows 1979 unchanged; same-place transports 9 -> 1;
`__auto__ -> __auto__` 4 -> 0; unplaced reactions 84 -> 75 and unplaced transport sides 15 -> 11 beyond the
deterministic arm. PWML: 105/105 exported; 20 files changed.

**Caveat on the LLM-arm PWML A/B.** To give the client its key the harness loaded the app's `.env`, which also
configures the name-resolution DB; 8 files (5, 9, 16, 18, 122, 131, 132, 141) then show different compound *display
names* ("1,4-dihydroxy-2-naphthoic acid" -> "1,4-dihydroxy-2-naphthoate"), counted by `ab_replay.py` as 14 lost names
and 23 lost reaction signatures. None of those 8 files received an LLM inference (the deterministic arm, without the
`.env`, has 0 lost names in all of them), so the renames come from the resolver, not the pass; counts are unchanged in
all 8. The clean A/B (LLM arm replayed from the reply cache with no `.env`, `replay.py --llm-cached`) is scripted
but **did not run**: the heavy lock went to other lanes for over an hour and the queued job was stopped. Run it with
`C:/t/loc-work/arm_llmc.sh` before the PO turns the pass on (Q4).

Every accepted inference, with its quote (all pass the verbatim + place + item gate):

| run | item | side | placed in | quote |
|---|---|---|---|---|
| 10, sour taste | OTOP1-mediated H+ permeation (x2) | from | apical state | "permeation of H + through an apical ion channel" |
| 10, sour taste | OTOP1-mediated H+ permeation | to | Type III cell state | "entering H + ions not only directly depolarize the taste cells" |
| 134 | BBD reaction | | cytosolic state | "carnitine biosynthesis occurs initially in the mitochondria and subsequently in the cytosol" |
| 134 | Tha1 HTMLA reaction | | cytosolic state | "the first hit is the cytosolic serine hydroxymethyltransferase (SHMT1" |
| 136 | NicGH (β-GD1) reaction | | vacuolar state | "predicted to encode a vacuolar localised β -glucosidase ( β-GD1 , 884g0020 )" |
| 21, 26 | enterobactin secretion | to | E. coli extracellular space | "Enterobactin is produced in the cytoplasm and secreted to the extracellular environment" |
| 26 | ferric-enterobactin import | from | E. coli extracellular space | "Following secretion, enterobactin chelates extracellular ferric iron, ..." |
| 78 | enterobactin secretion | from / to | cytoplasm / extracellular | "Enterobactin is produced in the cytoplasm and secreted to the extracellular environment" |
| 22, 79, 85, 86, 97, 98 | DHCR24 reaction | | endoplasmic reticulum state | "This enzyme is mainly located in the endoplasmic reticulum, ..." |
| menaquinone | DHNA export | to | L. lactis extracellular state | "extracellular DHNA accumulation rate" |

The DHCR24 answers settle the 6 cases L7 had left in a first state that was right "by accident" (idx 86 / 98 in
CORRECTNESS-FIXES). Rejected answers: 10 (7 quote names none of the item's participants, 2 place not in the quote,
1 quote does not name the state's place); null answers (no basis in the paper) are not counted. Spend: 36 calls,
1.80 M chars in, 9.5 k chars out, **about $0.23** (estimated: the client reports no token usage; chars/4 tokens at an
assumed $0.50/M in, $3/M out), well inside the $5 budget, plus the 4-call showcase pilot (~$0.03).

**Gold sets and the 26-pathway sample.** The 3 gold-set live legs (menaquinone, enterobactin, lipid A) are in the
showcase replay, and the 26-pathway sample (`evidence/correctness/sample.json`) is a subset of the 152 replayed. Gold
A/B on fresh Stage-1 draws was not run (a live pipeline leg per gold paper; it does not fit this lane's lock time).

## 4. UniProt prior: measured, not enabled

`C:/t/loc-work/uniprot_agree.py` over the 152 runs: 540 proteins carry a cached UniProt subcellular location
(`enrichment.protein.uniprot.subcellular_locations`); 385 also have a paper-stated location. Generous class-level
agreement (any UniProt class equal to any stated class): **300 / 385 = 77.9 %** (79.9 % with the showcase copies).
Below the 80 % bar, so the prior is not built. The main disagreement is UniProt's generic "Membrane" against a
paper-stated cytoplasm (63 cases: peripheral membrane enzymes such as EntE). Excluding proteins whose only UniProt
location is the generic "Membrane", agreement is 295 / 321 = 91.9 % (Q2).

## 5. Open questions for the PO

- **Q1. idx 102 "enterobactin secretion"**: both ends are *stated* as `E. coli cytoplasmic state` (a second copy of the
  transport is stated correctly, cytoplasm -> extracellular). Fixing it would move a stated location, which the rule
  forbids; it is flagged `transport_ends_same_place`. Options: (a) keep flagged (default); (b) allow a same-place
  transport's destination to be replaced when the paper quote names another place (one row; changes a stated side).
- **Q2. UniProt prior without generic "Membrane"** (91.9 % on 321 proteins). Options: (a) not built (current); (b)
  build it for proteins whose UniProt location is specific, flagged `inferred: uniprot` (about 2 days; it would place
  some of the 155 proteins that have a cached UniProt location and no stated one).
- **Q3. Two duplicate state groups remain by design.** idx 3: "Type III cell cytoplasm state" and "Type III cell
  cytosolic state" (location field both "cytoplasm"; L6 keeps cytoplasm and cytosol apart in a eukaryote). idx 34:
  "E. coli biosynthetic / low-iron / high-iron state" (one place, three conditions). Options: (a) keep (default);
  (b) merge condition-named states of one place (loses the condition in the state name); (c) treat cytosol =
  cytoplasm in eukaryotes when the location field agrees.
- **Q4. LLM pass default.** It is off. Turning it on costs about $0.006 per run with unplaced items and placed 21 items
  on the replays with 0 stated changes. Options: (a) keep off until a fresh-draw gold A/B; (b) turn on now.
- **Q5. Eukaryote element rows** (148 unplaced) get no default and the LLM pass does not ask about elements (they are
  drawn with their reactions). Options: (a) leave; (b) place an unplaced element row in the state of the reactions
  it takes part in, flagged.

## 6. Dead ends and slips

- First LLM showcase pilot accepted 0 of 4 real answers: the participant gate missed abbreviations (DHNA) and dropped a
  currency cargo (H+). Fixed by passing the entity synonyms and the transport's own cargo to the gate.
- One stray `python -` (stdin) command was started by mistake while inspecting; it was stopped at once, no survivor.
- Vocabulary terms that are not PathWhiz locations (endosome, thylakoid, synaptic vesicle / cleft) were dropped from
  the closed list; a run that declares them still offers them as candidate states.

## 7. Evidence

`docs/pwml_output/evidence/locations/`: arm diffs (`compare-*.json`, totals and stated-location changes), PWML A/B
(`ab-*.json`), inference lists (`summary-*.json`), UniProt agreement counts, LLM cost. Scripts and full outputs:
`C:/t/loc-work/`.

## 8. Review round 1 (independent review: REJECT; fixes in `56dd2803`)

The review confirmed: stated locations are never overwritten; the 7 changed PWMLs were no-loss; the LLM path is off by
default, quote-checked and failure-isolated; the `.env` explanation of the 8 renames holds; repo hygiene is clean.
Fixed:

| # | finding | fix |
|---|---|---|
| 1 HIGH | The `ensure_autostates` "(other side)" placeholder never ran in the replay (it starts from stored payloads); two `cell -> cell` ends would still bind to one PathWhiz state; and two differently named unplaced ends were no longer flagged. | Hunk dropped; `process_normalizer.py` and `test_process_normalizer_transport_states.py` are byte-identical to base. Two unplaced ends now count as a possible same-place transport and are flagged. Row 1e of section 1 is withdrawn: the 4 -> 2 drop came from the location step. |
| 2 MED | L7's EXPORT rule removed correct, flagged neighbour fills in eukaryotes (citrate export out of the mitochondrion). | Rule dropped (row 1b withdrawn). The by-place comparison already covers DHNA; the location step's legacy repair keys on same-place ends only. New test: a eukaryote export keeps its neighbour fill. |
| 3 MED | The bacterial default fired in mixed-organism papers. | No default (reactions, element rows, transport sides) when any species row is not a prokaryote or a species row / state names another organism; those rows stay unplaced and get a `compartment_not_stated: no organism default (...)` review reason. Fires on 5 runs, all from stray *Arabidopsis thaliana* species rows: 76, 78, 138, lipid A, menaquinone. |
| 4 LOW-MED | The name-based transport default sends Gram-negative "LPS export" / Sec "protein export" to the extracellular space. | Name default only for a genus on a Gram-positive list and no periplasm / outer membrane state in the paper; Gram-negative or unknown: unplaced and flagged. Fired on no run this round (E. coli runs blocked; the Gram-positive runs had no open export side or were blocked by rule 3). |
| 5 LOW | L7's shared prompt and `max_tokens` were changed. | Both restored. The location pass has its own prompt (`llm/prompts/location_assign_system.txt`), prompt id `t2pw:location_assign:v1` and `max_tokens` 8000. L7's request sha and cache key for a fixed payload are identical on base and HEAD (`C:/t/loc-work/l7key.py`: `100cb0ba...` / key `471ce885...` on both). |

**Tests.** Focused 253 passed (new: 3 mixed-organism cases, Gram-negative and periplasm, eukaryote export fill kept,
two unplaced ends flagged, a chat call that raises leaves exactly the rules-only result with the error path-redacted,
L7's request unchanged). suite2 on `56dd2803` (clean tree, log `C:/t/vis-runs/logs/loc-suite2-r1.log`): SMOKE 511, golden batch 2 / c011 8 / stage8 4 / quarantine 24, equivalence 8 / 14 / 6, vision 357 (+1 skipped), web 83: all passed, 0 survivors; no golden-seam test edited.

**Replay on `56dd2803`** (152 q-runs + 8 showcase, same harness and census as section 2):

| metric | base | rules only | + LLM pass (from cache, no `.env`) |
|---|---|---|---|
| same-place transports | 9 | 9 (all flagged) | **2** (102 both stated; 138 auto->auto) |
| `__auto__ -> __auto__` transports | 4 | 4 (flagged) | 1 (138) |
| duplicate state groups | 5 | 3 | 2 |
| leaflet states | 14 | 10 | 10 |
| bacterial reactions with no place | 35 | 13 (all flagged; mixed-organism runs) | 13 (all flagged) |
| bacterial element rows with no place | 59 | 13 (flagged) | 13 (flagged) |
| unplaced reactions (all) | 119 | 97 (all flagged) | 89 (all flagged) |
| unplaced transport sides (transports) | 19 | 19 | 16 |
| placeholder / empty states | 60 | 60 | 58 |
| Stage-0-hint states flagged | 0 of 4 | 4 of 4 | 3 of 3 |
| reactions / transports / element rows (payload) | 692 / 128 / 1979 | same | same |
| **stated-location changes** | | **0** | **0** |
| inferred: organism_default (rx / side / element) | | 22 / 0 / 46 | 22 / 0 / 46 |
| inferred: llm_quote (rx / side) | | | 8 / 8 |
| PWML exports / byte-identical / changed | 105 | 105 / 101 / 4 | 105 / 94 / 11 |
| PWML lost names / reaction signatures | | **0 / 0** | **0 / 0** |
| PWML transport signatures relabelled (none lost; 96 transports) | | 3 (lipid A leaflets) | 5 (+ 78, menaquinone) |

Per-file no-loss check of every changed PWML (`locpairs.py`, element drawings per (name, place), merged spellings
mapped together); every name is kept, a "lost" pair is an element moving out of a default box:
- rules only: 18 (7 compounds, 2 proteins `cell` -> `cytoplasm`), 25 (7 pairs unchanged; 2 -> 1 states), 62 (3 pairs
  `cell` -> `cytoplasm`), lipid A (69 -> 67 rows, 65 pairs unchanged; leaflet states 11 -> 7).
- LLM arm adds: 10 and sour taste (13 pairs unchanged; OTOP1 H+ permeation to-side -> Type III cell), 78 (+ enterobactin
  outside), 79 (DHCR24, desmosterol, cholesterol -> ER), 134 (carnitine-pathway steps from the row-0 mitochondrion box to
  the cytosol), 136 (NicGH step from the row-0 cytosol box to the vacuole), menaquinone (32 -> 22 rows, 20 pairs
  unchanged).

**LLM arm.** A fresh live pass was needed (the new prompt id changes the cache key): 40 calls, about $0.26 (estimated:
chars/4 at an assumed $0.50/M in, $3/M out); session total about $0.52 of $5. The arm of record replays those replies
from the cache with no `.env` loaded, so no resolver renames: 0 lost names. 16 accepted inferences, all with a
verbatim, place-naming quote (DHCR24 -> ER in 22, 79, 85, 86, 97; BBD and Tha1 -> cytosol in 134; NicGH -> vacuole in
136; enterobactin secretion -> extracellular in 21, 26, 78; ferric-enterobactin import from extracellular in 26; DHNA
export -> extracellular in menaquinone; OTOP1 H+ permeation -> Type III cell in 10 and sour taste). The model answered
DHCR24 in 98 last round and not this round (same input, temperature 0).

**A chat call that raises.** Replay with the pass on and every model call raising (43 calls, fill cache without
location replies): 105/105 exports succeed, and every PWML and every payload (bar the step's own `llm` error report) is
byte-identical to the rules-only arm.

**PathWhiz import: not run.** `docker ps` cannot reach the Docker engine (Docker Desktop is down). Earlier in the round
it was briefly up running another lane's stack (`t2pw-pwdraw`), never this lane's `t2pw-loc`; per the instruction it was
not started.

**Open questions for the PO (undecided):** the mixed-organism default (5 runs with stray *Arabidopsis* species rows now
get no default, including menaquinone's DHNA export and 138's TBN export); Gram-negative name defaults (should "secretion"
in E. coli ever reach the extracellular space without a quote?); preferring the more specific name when merging states
(menaquinone keeps "intracellular" over "cytoplasmic state", lipid A keeps the first leaflet spelling); and a D-entry for
"inferred, always flagged, never overrides a stated location". Q1-Q5 of section 5 stand.

## 9. PO rulings (code `0cec290a`; supersedes section 8's fix 3 and Q1 of section 5)

**Ruling 1, mixed-organism papers.** PO, verbatim: "just use main organism".
The blanket skip of review round 1 (fix 3) is reverted. The bacterial default uses the paper's MAIN organism: the
organism hint, else the most frequent species among the states. It applies even when other organisms or stray species
rows are present. It stays flagged `organism_default`, and the estimate entry carries `organism` and a reason that names
the organism assumed and the others the paper lists ("main organism assumed: Escherichia coli (the paper also names
Arabidopsis thaliana)"). A eukaryote main organism still gets no default. The Gram rule for name-based transport sides
(fix 4) stands.

**Ruling 2, contradictory stated locations.** PO, verbatim: "whatever is most correct; if it states it in paper but then
in another part it says something else the LLM should be able to reason between the lines".
Scoped to the detected contradiction: a transport whose two STATED ends are the same place (run 102's "enterobactin
secretion", cytoplasm -> cytoplasm). With the LLM pass on, a separate call (own prompt
`llm/prompts/location_conflict_system.txt`, own cache id `t2pw:location_conflict:v1`, so the pass's cached replies stay
valid) gets the paper text and the conflict. A side is corrected only if the reply's quote is verbatim in the source
text and names both the item (cargo, transporter or process) and the new place, the place is a candidate state or in
the organism's closed vocabulary, and the two ends then differ. Flag `inferred: llm_corrected`; the estimate entry keeps
the original (`stated_was`, `stated_location_was`) and the quote. No quote, a failed check or a model error: no change,
and the row stays flagged `transport_ends_same_place`. Rules-only mode makes no call and changes nothing.

**Tests** (focused 260 passed): the mixed-organism cases now default to the main organism and name it; the organism hint
picks the main organism; a eukaryote main organism gets no default even beside a bacterium; rules-only leaves a stated
conflict; an accepted correction (stated value kept, text record line); a quote not in the paper and a quote that names
no new place change nothing; a raising conflict call changes nothing. L7's request sha and cache key are still identical
to base. suite2 on `0cec290a` (clean tree, log `C:/t/vis-runs/logs/loc-suite2-po.log`): SMOKE 511, golden batch 2 / c011 8 / stage8 4 / quarantine 24, equivalence 8 / 14 / 6, vision 357 (+1 skipped), web 83: all passed, 0 survivors; no golden-seam test edited.

**Replay on `0cec290a`** (152 q-runs + 8 showcase):

| metric | base | rules only | + LLM pass (from cache, no `.env`) |
|---|---|---|---|
| same-place transports | 9 | 7 (all flagged) | **0** |
| `__auto__ -> __auto__` transports | 4 | 3 (flagged) | 0 |
| duplicate state groups | 5 | 2 | 2 |
| leaflet states | 14 | 10 | 10 |
| bacterial reactions with no place | 35 | **0** (35 flagged `organism_default`) | 0 |
| bacterial element rows with no place | 59 | 0 (59 flagged) | 0 |
| unplaced reactions (all) | 119 | 84 (all flagged) | 76 (all flagged) |
| unplaced transport sides (transports) | 19 | 18 | 15 |
| placeholder / empty states | 60 | 59 | 57 |
| Stage-0-hint states flagged | 0 of 4 | 3 of 3 | 3 of 3 |
| reactions / transports / element rows (payload) | 692 / 128 / 1979 | same | same |
| **stated-location changes** | | **0** | **0** |
| **llm_corrected** (stated side replaced, quote-verified) | | 0 | **1** (run 102) |
| inferred: organism_default (rx / side / element) | | 35 / 3 / 59 | 35 / 2 / 59 |
| inferred: llm_quote (rx / side) | | | 8 / 8 |
| PWML exports / byte-identical / changed | 105 | 105 / 99 / 6 | 105 / 92 / 13 |
| PWML lost names / reaction signatures | | **0 / 0** | **0 / 0** |
| PWML transport signatures relabelled (none lost; 96 transports) | | 5 | 7 |

**llm_corrected, every case:**

| run | transport | side | stated (kept in `stated_was`) | corrected to | quote |
|---|---|---|---|---|---|
| 102 | enterobactin secretion | to | E. coli cytoplasmic state (cytoplasm) | extracellular space | "Enterobactin is produced in the cytoplasm and secreted to the extracellular environment" |

Per-file no-loss check of every changed PWML (`locpairs.py`): every name kept; a "lost" pair is an element leaving a
default box. Rules only: 138 (its 8 unplaced reactions leave the row-0 extracellular box for a new *S. tubercidicus*
cytoplasm; TBN export `cell -> cell` becomes cytoplasm -> extracellular), 18 and 62 (`cell` -> `cytoplasm`), 25 (7
pairs unchanged), lipid A (65 pairs unchanged; leaflet states 11 -> 7), menaquinone (DHNA export cytoplasm ->
intracellular becomes intracellular -> extracellular; 20 pairs unchanged, 32 -> 22 rows). The LLM arm adds 10 and sour
taste (13 pairs unchanged), 102 (16 pairs unchanged; the correction), 78 (+ enterobactin outside), 79 (DHCR24 -> ER),
134 (-> cytosol), 136 (-> vacuole). The 5 runs round 1 skipped for stray *Arabidopsis* rows (76, 78, 138, lipid A,
menaquinone) now get the main organism's default where they have unplaced rows, flagged with both organisms named.

**A model call that raises.** Every call raising (44: 43 main pass + 1 conflict; original fill cache): 105/105 exports
succeed; every PWML and every payload (bar the step's own `llm` / `llm_conflict` error reports) is byte-identical to the
rules-only arm.

**Spend.** Only the new conflict prompt needed a fresh call: 1 call, about $0.006. Session total about $0.53 of $5
(estimates: the client reports no token usage). The live arm loaded the app's `.env` for the key; the arm of record
replays from the reply cache without it (0 renames).

**PathWhiz import: not run.** No lock hold of this lane had Docker with a `t2pw-loc` stack; the `t2pw-pwdraw` stack
belongs to the draw lane and was not touched.

**Still open for the PO:** Gram-negative name defaults; preferring the more specific name when merging states; a
D-entry for "inferred, always flagged, never overrides a stated location (except a quote-verified `llm_corrected`
contradiction)"; Q2-Q5 of section 5.

## 10. PathWhiz import A/B (D-W24 (9)), PWMLs of `0cec290a`

Harness: `C:/t/loc-work/harness` (the import-fixes harness, compose project **`t2pw-loc`**, ports 3367 db / 3064 app;
PathWhiz `t2pw/latest` `63959982` cloned into `C:/t/loc-work/smpdb-run`, MolDB pointed at a closed local port as in
the showcase). One lock hold (`C:/t/loc-work/logs/loc-pw.log`, 706 s, exit 0, 0 survivors). Docker Desktop was started
for it and stopped afterwards (no other container up); the stack went down with its volumes kept. Clean database per
file: the import rake truncates every data table before each file. Base = main `fd814b8e` replay; rules = rules-only
arm; LLM = LLM-from-cache arm. Every file imported (`success` true in all 32 imports).

| file | arm | created counts that changed (base -> branch) | importer messages | verdict |
|---|---|---|---|---|
| 10 (sour taste, stored) | LLM | transports 6 -> 7, transporters 2 -> 3 | 10 -> 6 (OTOP1 same-state transport refusal gone) | better |
| 102 | LLM | same counts | 3 -> 0 (same-state "enterobactin secretion" refusal gone) | better |
| 134 | LLM | compound locations 16 -> 19, protein locations 11 -> 13 | 0 -> 0 | better |
| 136 | LLM | compound locations 16 -> 17 | 0 -> 0 | better |
| 138 | rules, LLM | transports 0 -> 1, compound locations 13 -> 14 | 3 -> 0 (TBN export no longer refused) | better |
| 18 | rules, LLM | same counts | 0 -> 0 | same |
| 25 | rules, LLM | **biological states 2 -> 1** | 0 -> 0 | **BLOCKER (count down)** |
| 62 | rules, LLM | same counts | 0 -> 0 | same |
| 78 | LLM | transports 0 -> 1, compound locations 17 -> 18 | 3 -> 0 (enterobactin secretion no longer refused) | better |
| 79 | LLM | compound locations 6 -> 7 | 0 -> 0 | better |
| lipid A | rules, LLM | **compound locations 47 -> 45**, label locations and membrane visualisations down | 29 -> 21 (two kinds of ontology message gone) | **BLOCKER (count down)** |
| menaquinone | rules, LLM | **transports 2 -> 1**, biological states 3 -> 2, compound locations 16 -> 13, protein locations 16 -> 9, element states 15 -> 8 (and their drawings) | 0 -> 0 | **BLOCKER (count down)** |
| sour taste (showcase) | LLM | transports 6 -> 7, transporters 2 -> 3 | 10 -> 6 | better |

**No new refusal and no new message in any file**; message counts only fall (25 fewer in the LLM arm). **Three files have a created count that goes
down: reported as blockers, not fixed** (per the brief). What each one is, from created vs the count in the PWML
(`C:/t/loc-work/pw_detail.py`):
- **menaquinone, transports 2 -> 1.** The PWML still holds 2 transports (expected 2). In base they were "DHNA diffusion"
  intracellular -> extracellular and "DHNA export" cytoplasm -> intracellular (one place at both ends in a bacterium).
  On the branch both are intracellular -> extracellular, i.e. the same cargo between the same two states, and PathWhiz
  stores that once. The other drops follow from merging the `cytoplasmic state` into the `intracellular` state (the
  cytoplasm of a bacterium): 20 distinct (element, place) pairs before and after; the drop is duplicate drawings.
  Options: (a) accept (no distinct transport lost); (b) write two transports with the same cargo and sides as one in the
  PWML; (c) keep the two states apart in this run. PO decision.
- **lipid A, compound locations 47 -> 45.** The PWML itself holds 45 (expected 45): two duplicate drawings collapse
  when the three spellings of each inner-membrane leaflet merge into one state (65 distinct pairs before and after).
  Fewer state boxes, so fewer labels and membrane drawings. Transports created 3 of 7 in both arms (unchanged).
- **25, biological states 2 -> 1.** Base placed the unplaced element rows in a second, default state; the branch puts
  them in the run's own cytoplasm state (organism default, flagged). Same 7 (element, place) pairs.

Evidence: `evidence/locations/pw_ab_rules.json`, `pw_ab_llm.json` (per file: every created count, messages gained and
gone) and the three harness `results*.jsonl`.
