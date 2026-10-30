# ORCH-797b RESULT: alias identity widened (D-120 § 2) and family terms (D-120 § 3), NOT DONE (blocked on machine RAM)

**Branch:** `orch797b/fix` (base `aefa5b30`). **Status: draft code committed, nothing run.** No test, probe, recording or
replay ran, so there is no G9 proof, no affected-test result, no leg replay and no G11 report. The draft must not merge.
**Merge rule 5 is waived** (D-120 § 5): no reviewer round.

## 1. Why nothing ran

The rules require 3 GB of free RAM (`gate.sh` / `run.sh` exit 90 below it). For about 2 h 20 min (`C:/t/orch797b/waitram.sh`,
polling every 15 s) free physical memory stayed between 0.47 and 2.13 GB. The heavy lock was free the whole time, and no
sprint Python process was running. `Get-Counter` showed 57 GB committed (RAM plus page file), a 2.9 GB non-paged pool and
1.2 GB available. The visible processes are Chrome, VS Code, MATLAB and claude, and their working sets sum to about 11 GB, so
most of the memory is held outside user processes. This matches the "multi-day uptime" pattern recorded in memory
(`parallel-agents-froze-the-machine`). A reboot, or closing MATLAB and Chrome, should clear it. The very first job
(`probe-rows-base`) was refused with exit 90 at 0.68 GB free.

## 2. What the draft changes (`src/t2pw/mapping/map_ids.py` only; unproven)

| function | change |
|---|---|
| `_extract_uniprot_candidates` | Keeps UniProt's recommended, alternative and **short** protein names as `protein_names`. Before this, alternative names were scored and then discarded, and short names were never read. A whole-word match of the paper's name inside a protein name scores like an exact name match (0.55 base). Today c-Jun scores 0.47 < 0.5. |
| `_name_in_protein_names` (new), `_alias_identity_match` (new), `_spell_greek` (new) | The paper's name equals a protein name, or appears in one as consecutive whole words. Words split on space, comma, semicolon, slash and brackets, **not on hyphens**, so "c-Jun" matches "Proto-oncogene c-Jun" but not "C-Jun-amino-terminal kinase-interacting protein 1", and "Kir2.1" does not match "Kir2.10". A Greek letter is spelled out, not deleted. |
| `_name_gate_verdict` | New rescue, `uniprot_protein_name_identity`: the name is one of the entry's protein names, the species is `ok`, and the name is not a family name. Added only for Kir2.1, whose name shares no gate token with "Inward rectifier potassium channel 2". |
| `verify_real_protein_identity`, margin rung | A candidate that matches the name through its protein names now also contests the margin as a rival. |
| `_break_alias_tie_by_paper_wording` (new), called after FX4 at both UniProt sites (`_map_protein_with_strategy` and the Phase-2 fallback) | Applies when two or more **reviewed** genes in the species carry the name. A gene wins only when its *distinctive* protein-name words occur in the paper's own evidence sentences naming the protein and no other matching gene's words do. Distinctive means the word is in no other matching gene's names, and is not a stopword, a generic word or the name itself. The rivals are then set aside, as FX4 does, and the winner still faces the unchanged ladder. Otherwise the result is left as it was (margin refusal, KI-001) and tagged `alias_tie_unresolved`. **The primary-gene tie-break never decides alone** (ORCH-793 § 4). Never applied to a family name. |
| `_payload_evidence_quotes` (new), `_identity_context` | Every `evidence` string of the payload. Each protein's context carries the sentences that name it as a whole word. |
| `_family_key` | Greek letters are spelled out. Today "PKCδ" keys as "pkc" and is wrongly treated as the PKC family. Under the draft it keys as "pkcdelta". |
| `_FAMILY_TERMS`, `_FAMILY_QUERY_CORE`, `_FAMILY_GENE_MEMBERS` (new), `_PLURAL_FAMILY` (new), `_natural_key` (new), `_apply_family_complexes` › `_uniprot_family_members` | Adds InsP3R / IP3R / IP3 receptor / inositol 1,4,5-trisphosphate receptor (UniProt "InsP3 receptor family"), SERCA, calmodulin, Dvl / Dishevelled ("DSH family"), Smurf, Par6, MRLC, profilin and Wnt. SERCA (ATP2A1-3), Smurf (SMURF1/2) and MRLC (MYL9, MYL12A, MYL12B) have no UniProt family name, so each member gene is asked for with `gene_exact`, reviewed, same taxon. **Generic detection:** a plural of one symbol-like word ("Rabs") is converted only when UniProt's exact family-level match returns members; otherwise the row is untouched. Members sort naturally (WNT1, WNT2, WNT2B). DIA1 (mDia1 = DIAPH1) and GBP (GluTR-binding protein) are single proteins, so they are not added. |

The cached `api-v9` / `api-fallback-v2` UniProt answers carry no `protein_names`. The draft deliberately does **not** bump the key: the tracked
`data/id_mapping_cache.json` holds no `api-v9` entry, so a fresh run worktree re-asks UniProt anyway. Only the ORCH-796 run worktree's own
cache keeps refusing these rows until they are re-asked. See § 5.

Replay script, written but not run: `docs/pwml_recovery_sprint/evidence/orch797b_replay.py`. For every protein the leg left without a
UniProt id it removes the run's cached answers, records the same queries once (`--record`, live, UniProt only), and replays both arms
offline. It then runs Stage 2, normalisation, Stage 6, strict quarantine and the ORCH-768 tail (`replay_leg`, pre-freeze).

## 3. What remains (in order)

1. Run `C:/t/orch797b/probe_rows.py` (base): the unresolved rows per leg and their cached candidates.
2. Run `orch797b_replay.py --record` with the tip tree, then the base tree, writing `evidence/orch797x/orch797b_uniprot_responses.json`.
3. Write `tests/test_orch797b_alias_and_family.py` from the recorded answers.
   - G9 cases: c-Jun → P05412, Kir2.1 → P35561, NARF → Q8WVD3 when the RING-finger sentence is present, TAK1 → O43318 ("TGF-β-activated protein kinase 1 (TAK1)"), and InsP3R / SERCA / calmodulin / Dvl converted to families.
   - Preservations: NARF never Q9UHQ1, and omitted when there is no wording; `test_preservation_a_primary_gene_name_does_not_outrank_another_entrys_synonym` stays green; Wnt5a, Fzd2 and PKCδ are not families; "G protein" never takes EIF3G.
4. Run the base arm (copied into `C:/t/orch797b/base`), the tip arm, and the affected files: every test importing `map_ids` that the brief lists, plus `test_orch793_gene_name_identity`, `test_orch790_*`, `test_orch783_*`, `test_c122_*`, `test_map_ids*` and `test_orch762_*`.
5. Run the leg replays base vs tip (chlorophyll, wnt_ca, wnt_pcp, sour_taste_v2; enterobactin and menaquinone for no-loss), then audit every accession the tip ships that base did not.
6. Run `g11_evidence.py check --task ORCH-797b`.

## 4. Failures, dead ends, wrong theories

- **A rules breach, stopped:** one Bash call began with a stray `python - 2>/dev/null` (stdin Python) ahead of an edit. It was stopped with
  TaskStop. A process check afterwards showed no surviving `python.exe` other than the VS Code isort language servers. It ran no code and read nothing.
- **Considered and rejected:** "an exact primary-gene or recommended-name match wins" (D-120 § 2) applied on its own. It would ship NARF → Q9UHQ1, the
  wrong protein (ORCH-793 § 4). In the draft that rule never decides alone; only the paper's wording can separate two genes.
- **Expected outcomes, from reading the recorded UniProt entries (`evidence/orch793/uniprot_entries.json`) and the wnt_ca quotes; not measured:**
  - TAK1 separates. The quote "TGF-β-activated protein kinase 1 (TAK1)" gives MAP3K7 the words tgf, activated and kinase; NR2C2 gets none.
  - NARF separates to RNF138 through "NLK-associated RING finger protein (NARF)", if UniProt's RNF138 names include "RING finger".
  - PIF1 stays omitted if a chlorophyll quote happens to contain a CLF-only word ("leaf", "flowering").
  - DIA1 is a synonym of both DIAPH1 and CYB5R3 (to be checked), and the quotes hold no distinctive word, so it stays omitted.

## 5. Not settled (for the PO)

1. **Cache key.** (a) Leave it (the draft): only stale run caches miss the new names. (b) Bump `api-v9` → `api-v10` and `api-fallback-v2` → `v3`:
   every protein is re-asked once per run, and fixtures that replay `api-v9` keys (`tests/fixtures/orch762`, the ORCH-793 tests) need re-recording.
2. **PLCβ** under the Greek fix stops being "the PLC family". PLCβ is itself a subfamily (PLCB1-4), while PKCδ is one protein; a lexical rule
   cannot tell the two shapes apart. Options: add "PLCβ" as a family term, or accept the per-row omission. No ORCH-796 leg has a Greek-letter protein
   other than β-catenin.
3. Paper-wording words include verbs ("activated"). A coincidental verb could support the wrong gene; the uniqueness rule (the rival must have
   none) limits this but does not remove it. A `ponytail:`-class ceiling; the new-ids audit is the check.

## 6. KI delta / LLM-side

- KI delta: none yet. The draft adds diagnostic keys only (`alias_tie_break`, `alias_tie_unresolved`, gate reason
  `uniprot_protein_name_identity`). An unseparated alias row stays KI-001.
- LLM-side: none identified.

## 7. Proposed LEDGER row

`ORCH-797b | builder | orch797b/fix @ <tip> | BLOCKED: draft of D-120 §§ 2-3 committed, unrun (free RAM < 3 GB for 2 h 20 min; 57 GB committed) | no G9, no tests, no replay, no G11 | rule 5 waived (D-120 § 5) | resume at RESULT § 3 after a reboot`
