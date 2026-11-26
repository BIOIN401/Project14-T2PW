# VIS-4: structures in compound boxes, a readable picture, PathWhiz edge rules

2026-11-26 (night of 11-25). Branch `vis/render-structures` from `webserver/refactor` `019c521f`, merged with
`webserver/refactor` `68f7c6e4` (VIS-1 merged) at `20db38d0`. PO request, verbatim: "have the new better rendering
done by tomorrow morning iterate until our renderings look like how they should". Builds on VIS-3
(`docs/webserver/results/VIS-3.md`); the server switch `T2PW_RESULT_RENDER=pathbank` is unchanged and still off.

## 1. What changed

| file | what |
|---|---|
| `src/t2pw/render/structures.py` (new, ~170 lines) | PubChem 2-D structure PNG per compound. Key `CHEBI:<n>` (PubChem's ChEBI cross-reference, `compound/xref/RegistryID/CHEBI:<n>/PNG`) when the compound has a ChEBI id, else `CID:<n>` (`compound/cid/<n>/PNG`). Only that public id leaves the machine. On-disk cache `$T2PW_STRUCTURE_CACHE` (default `~/.cache/t2pw/structures`): `<key>.png`, `<key>.miss` (PubChem 4xx: never asked again), `<key>.fail` (network error or timeout after one retry: asked again after an hour). 4 s per request, requests paced to 4/s across 4 threads (PubChem throttles above 5/s with 503), overall budget `structure_budget_s` (default `$T2PW_STRUCTURE_BUDGET_S` or 6 s; 0 = cache only). What does not arrive in time lands in the cache for the next render. Nothing raises; a box without a picture stays empty. stdlib only (urllib, threads), no new dependency |
| `src/t2pw/render/pathbank.py` | (a) structure picture in every structure box (>= 90x80), backdrop made transparent, cropped to the drawing, placed as PathBank does (`translate(25,40) scale(0.75)`: 20-unit side margins, below the header lines), never enlarged past 1.1x. (b) **Orphan edges skipped**: an edge no `edge-id` in the visualization refers to is not drawn (PathWhiz's viewer drops it; review F2 of VIS-1). (c) Crop to the drawn extent + 40 units. (d) `output_scale`: 2400 px target, floor 0.85 px/unit, cap 8000 px (see § 3). (e) Compound header as PathBank: font 20, wrapped at spaces and after hyphens, extra lines running into the body; shrinks only when one part is wider than the box. (f) Enzyme labels: font 15 kept down to 8, text uses 92 % of the ellipse width (PathBank's own wrap), labels wrap after hyphens before cutting words. `render_pwml(path, "png"|"svg")` unchanged; new keyword `structure_budget_s`; `max_px` default is now `None` = 8000 cap |
| `tests/test_render_structures.py` (new, 13 tests, 16 with parameters) | cache hit/miss/fail markers and retry window, one retry + pacing, budget bound with a stalled fetcher, budget 0 = cache only, `download` URL and HTTP-status mapping, ChEBI-first key, picture cropping/transparency, picture drawn in the box and no request for pills, empty box when offline, byte-identical warm render, scale rule, orphan edge skipped, header wrapping. **No network**: `download` and `urlopen` are replaced in an autouse fixture |
| `tests/test_render_pathbank.py` | autouse fixture: per-test empty cache, `download` raises `OSError`, `urlopen` fails the test; MINI/EDGES_ONLY fixtures now refer to their edges (else the orphan rule drops them); the size-cap test checks the cropped aspect |

No file outside the boundary changed: `git diff --stat 68f7c6e4 HEAD` (the integration head merged in) lists exactly these four files, +631 -35. Code commit `539a0940`, merge `20db38d0`.
`pyproject.toml` not needed: `structures.py` is a module of the existing package.

## 2. What changed visually (rounds r0..r4)

Rounds in `C:/t/vis-pw4-work/r0` (VIS-3 merged renderer, baseline) .. `r4` (final), each with a `pathbank.py.snapshot`.
Side-by-side page: `C:/t/vis-pw4-work/sbs-r4.html` (PathBank SVG via Edge | VIS-3 | VIS-4 for PW000001, PW000017,
PW000779, PW012911, PW127049; VIS-3 | VIS-4 for item-6 final2 and base, chlorophyll, lipid A, ceramide, coenzyme A,
menaquinone, furochromone, sour taste, enterobactin). Committed composites: `evidence/VIS-4/*.png`.

| round | change | what I saw |
|---|---|---|
| r1 | structures by CID, crop, scale floor 0.55, PathBank header | references now read like PathBank (PW000001: every box has its molecule). item-6: **every catalysing enzyme label scrambled** (orchestrator saw it too); 11 of 101 fetches failed |
| r2 | orphan edges skipped; fetch paced + 1 retry; floor 0.85; picture cap 1.1x | enzyme labels clean. **Wrong molecules**: isochorismate drew 5-methylfurtrethonium, lactate drew oxatomide, ceramide a steroid glycoside: our PWMLs carry wrong PubChem CIDs (§ 5) |
| r3 | ChEBI first | item-6 molecules right (chorismate, isochorismate, 2,3-DHB, pyruvate, lactate, L-serine, enterobactin). 0 failures, 12 misses (PubChem has no compound for that ChEBI id: classes such as ceramide, lipid A, LPS, cardiolipin, sphingomyelin; also chlorophyll a/b and two chlorophyllides) |
| r4 | wrap after hyphens (header and labels) | "2,3-dihydro-2,3-dihydroxybenzoate" at 20 instead of 10 units; final |

Before/after: `evidence/VIS-4/ours-ent-item6-final2.png`, `zoom-ent-item6-final2.png`, `ours-chlorophyll.png`,
`ours-ceramide.png`, `ours-lipid-a.png`, `sbs-PW000001.png`, `sbs-PW012911.png`.

**Fidelity checked against PathBank, now matching**: structure placement and size, header font and wrap (PW012911's
"3,8-divinyl / chlorophyllide a" and one-line "Oxoglutaric acid" reproduced), enzyme label size, pill positions and
colours, membrane bands, zoom wedges, chevrons at the template-specified ends, flat bars on inhibitions.
**Still different** (not fixed tonight): pills are flat colours, not PathBank's gradients; the 78x78 small-molecule
icons show text, PathBank shows a ball model; PubChem's drawing style shows explicit H on heteroatoms where PathBank's
moldb drawing writes "OH"; DejaVu Sans, not Verdana.

## 3. Readability on the result page

Font sizes are in PWML units; the old scale (2400 px longest side) put a 7658-wide canvas at 0.31 px/unit (enzyme
text ~5 px). Now the picture is cropped to what is drawn and scaled `max(2400/longest, 0.85)`, at most 1, capped at
8000 px: PathBank's 12-unit pill text >= 10 px, 15-unit enzyme text >= 12 px in the PNG. Sizes: item-6 3238x1742
(221 KB), chlorophyll ~6500 wide (527 KB), lipid A 5074x4330 (797 KB), PW127049 (largest reference) 6 MB.
(PathBank's own full PNG of PW000001 is 4400x4800.)

**The page still shrinks it.** `result.html:48` uses `c.figure(...)` (`macros/components.html:253`): a bare `<img>`,
`max-width: 100%` (`site.css:88`) in a 72rem (1152 px) container, no link. item-6 is therefore shown at ~0.36x
(enzyme text ~4 px). Needed (outside this task's boundary, not done): wrap the image in a link to itself
(`<a href="{{ src }}">`, opens the full-size PNG in the browser, which zooms natively), or link a `pathway.svg`
beside it (the renderer already produces SVG; text stays sharp). One-line template change + a design-system test.

## 4. Gates

All through `C:/t/vis-runs/queue.sh` holder `vis4` (bounded wrapper, heavy lock), Git bash by full path.

| gate | tree | result | log |
|---|---|---|---|
| focused: `test_render_pathbank.py` + `test_render_structures.py` | `20db38d0` (code + merged VIS-1) | **39 passed** | `evidence/VIS-4/logs/vis4-r4.log` |
| focused: `test_service_run_pathway.py` | same | **107 passed** | same |
| focused: all 11 `tests/test_web_*.py` | same | **239 passed** | same |
| `suite2.sh C:/t/vis-pw4 vis4`: SMOKE | `20db38d0` | **511 passed** | `evidence/VIS-4/logs/vis4-suite2.log` |
| golden seam batch / c011 / stage8 / quarantine (unedited) | same | **2 / 8 / 4 / 24 passed** | same |
| equivalence run_pathway / app-vs-service / default path | same | **8 / 14 / 6 passed** | same |
| vision | same | **370 passed, 1 skipped** | same |
| web (app, review reasons, browser fixes, design system) | same | **83 passed** | same |
| render batch r4: 5 references + 10 of ours (PNG), 2 SVG | same | 17 of 17 rendered, 0 failures | `vis4-r4.log` |

Every run `BW_EXIT=0`, **FINAL SURVIVING COUNT 0** (r0/r1, r2, r3, r4, suite2). The wrapper's "wrapper vs HEAD"
line names the primary checkout's HEAD, not this worktree; the tree under test is the `HEAD` line each script prints
(`20db38d0`, `git status --porcelain` empty at the start of suite2). Earlier focused runs: r2 on the pre-commit
tree, 3 failures (the network-in-tests slip, § 6); r3 38 passed. No existing golden or equivalence test edited.

**Render timing** (`timing.py`, item-6 final2 and the largest showcase lipid A, plus chlorophyll; each with a fresh
empty cache, then twice warm; inside the r4 job, so on the merged tree):

| PWML | structure keys | budget 0 (no fetch) | cold cache | warm | warm again | pictures |
|---|---|---|---|---|---|---|
| item-6 final2 | 7 | 0.18 s | 2.10 s | 0.21 s | 0.21 s | 7 |
| lipid A | 15 | 0.61 s | 4.77 s | 0.65 s | 0.72 s | 10 (5 ChEBI classes PubChem lacks) |
| chlorophyll | 11 | 0.50 s | 3.88 s | 0.53 s | 0.52 s | 9 |

Cold time is the pacing (0.25 s per request start) plus PubChem latency; it is bounded by the 6 s budget whatever the
network does. The largest reference, PW127049 (6 MB PNG, 5385x6206), renders in 2.9 s warm. Warm renders are
byte-identical (test `test_render_is_deterministic_with_a_warm_cache`).

## 5. Findings for other lanes (not changed here)

1. **Wrong PubChem CIDs in our PWMLs** (mapping, not render). item-6: isochorismate `pubchem-cid` 4141 =
   5-Methylfurtrethonium, lactate 4615 = Oxatomide (PubChem titles checked); ceramide's CID draws a steroid glycoside.
   Their ChEBI ids are right, hence ChEBI-first. Owner: `mapping/` (biology track); worth a check across the
   showcases, since the PWML exports these CIDs to PathWhiz.
2. **Orphan connectors** (VIS-1 review F2): template-83 edges from each enzyme through its own location, referenced
   by nothing. Dropped by the renderer now, as PathWhiz drops them; the layout could stop writing them.
   The same rule also hides **7 template-5 reaction edges** that nothing refers to, in VIS-1's final2 replays:
   `100.pwml` #38, `146.pwml` #35-36, `6.pwml` #124-127 (VIS-4 review F5; checked with `scripts/orphans_tpl.py`;
   none in the 8 showcases or item-6). PathWhiz drops them too, so the picture matches PathWhiz, but the reactions
   lose those lines: a writer defect for the layout lane.
3. **Red edge 2,3-dihydroxybenzoic acid -> EntB** on item-6: edge 162, template 148, `start_flat_arrow`, drawn by
   `interaction-protein-complex-visualization` of interaction 60, type **Inhibition**, Compound 40792
   (2,3-dihydroxybenzoic acid) -> ProteinComplex 40002 (isochorismatase, EntB). 148 is PathWhiz's editor default for
   Inhibition (`pwml/interaction_types.py` `PWML_INTERACTION_EDGE_TEMPLATES`: Activation 149 cyan, Inhibition 148
   red), so the colour is right and the writer is right; whether DHB inhibits EntB is extraction content (PO).
   The cyan EntD -> EntB edge is interaction 63, Activation, 149. VIS-3's open question on red/cyan is answered: not
   ours to fix in the writer.
4. Layout notes from the renders (VIS-1/VIS-2): transport edges to the extracellular boxes on item-6 are right-angle
   routes; some Golgi edges on ceramide start in empty space; lipid A label-locations stack two pills for one name;
   compartments with template 220 (black, 6 wide) are nearly invisible on the dark-blue background.

## 6. Dead ends and slips

- CID first (r1, r2): wrong molecules on our PWMLs (§ 5.1).
- CID fallback after a ChEBI miss: rejected; it would bring the wrong CID pictures back for ceramide-like entries.
  Cost: PathBank's PW012911 chlorophyll a / chlorophyllide a boxes stay empty (their ChEBI ids have no PubChem
  compound; their CIDs are right there).
- 4 unpaced threads (r1): 11 of 101 requests failed (PubChem 503 throttle; the same CIDs fetched fine one by one).
- `fetch` default bound at definition time: the tests' monkeypatch of `structures.download` did not apply, so the
  r2 focused run (3 failures) made real PubChem requests from tests (`reference/PW000001.pwml`). Fixed (looked up at
  call time) and both test files now also replace `urllib.request.urlopen`, so a test that reaches the network fails.
- I typed `python -I -` once by mistake (a stdin interpreter, against the brief's first rule); it hung on stdin and I
  stopped it within two minutes (TaskStop; no process left: `ps -W` showed only processes from before 20:46).
- An outer `timeout 600` around the first queue call: killed by me before `bw.sh` ever took the lock (the queue was
  waiting on another session's container), so no lock was stranded.

## 7. Open questions

1. **Result page link** (§ 3): a template change (orchestrator / ws-ui-builder): link the image to itself or add
   `pathway.svg`. Without it the improved picture is still shrunk to the column.
2. **Service budget**: `service/pathway.py` calls `render_pwml(pwml_path)` without a budget, so a cold render can spend
   up to 6 s (default) fetching; within the run's own deadline check (VIS-3 F3) but not tied to it. Options: (a) pass
   `structure_budget_s=min(6, deadline - time.time())` there (2 lines in `service/`), (b) leave the env default.
3. **Server outbound access**: the server needs HTTPS to `pubchem.ncbi.nlm.nih.gov` and a writable
   `T2PW_STRUCTURE_CACHE`; offline it renders empty boxes (as before). Only ChEBI ids / CIDs are sent.
4. **Attribution**: structure drawings are PubChem's (public domain, NCBI asks for acknowledgement); add
   "structures: PubChem" next to the PathBank art credit (VIS-3 open question 4).
5. Wrong CIDs (§ 5.1): for the biology track.

## 8. Review round 1 (ws-reviewer on `95d05949`: APPROVE WITH NOTES), fixed in `067cb575`

- **F1** `structures.key_for` / `_valid`: ids must be 1-10 ASCII digits (`re.fullmatch(r"[0-9]{1,10}")`). Before,
  `str.isdigit()` let "²" through to `int()` (ValueError: the whole render failed), turned "٣" into CHEBI:3 (a wrong
  molecule) and let a 5000-digit id raise. Now any of these is "no id" (a broken ChEBI id falls back to the CID);
  `key_for` never raises.
- **F2** `structures.png_ok`: the PNG's IHDR width and height must be <= 2000 px (PubChem's depictions are 300 px),
  checked before a body is cached and before `pathbank._picture` decodes it, so a small file can no longer decode to a
  multi-GB array. A body over the 2 MB read cap is `Unusable` and never cached as `.png`; `_get` re-checks whatever a
  fetcher returns.
- **F3** `download`: only 404, or a 400 whose body says `PUGREST.NotFound`, is a permanent `.miss`. A 403, any other
  status, or a 200 that is not a usable PNG (maintenance page) raises, and is cached as a retryable `.fail` (one retry,
  then an hour), so an IP block cannot mark every compound missing for good.
- **F5** record only: § 5.2 now names the 7 template-5 reaction edges the orphan rule also hides.
- **F6** `_header_lines`: line widths are measured (`matplotlib.textpath.TextToPath`, the font the picture uses)
  instead of 0.55 em per character. "dihydroxybenzoate" now drops to 19 units and stays inside the 190-unit body;
  "Oxoglutaric acid" and "3,8-divinyl / chlorophyllide a" still match PathBank's lines. Zoom:
  `evidence/VIS-4/zoom-ent-item6-final2-r5.png` (round r5).
- **F7** `tests/conftest.py` already existed: one autouse fixture sets `T2PW_STRUCTURE_BUDGET_S=0` (cache only), so
  no test anywhere fetches unless it passes a budget itself (6 lines; outside the original file boundary, added at
  the orchestrator's request). The two render test files keep their own `download` / `urlopen` guards.
- Tests added: F1 (8 parametrised bad ids + the CID fallback), F2 (IHDR check, no decode of an oversized header,
  oversized and over-cap bodies, nothing cached as `.png`), F3 (404 and 400 NotFound miss; 400 BadRequest, 403, 410,
  429, 503 and a non-PNG 200 retryable; `.fail` files on disk), F6 (every header line measured to fit).

Gates on `067cb575` (`evidence/VIS-4/logs/vis4-review.log`, holder `vis4`, tree clean at start): render tests
(`test_render_pathbank.py` + `test_render_structures.py`) **49 passed**; `test_service_run_pathway.py` **107 passed**;
render round r5 15 of 15; `BW_EXIT=0`, **FINAL SURVIVING COUNT 0**. SMOKE and the full gates: the orchestrator runs
them on the merge result.
