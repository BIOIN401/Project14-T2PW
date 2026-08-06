# Corrected exporter-identity probe — measured output, 2026-08-06

Task **H-005**, branch `agent/h05-probe-and-authority-corrections`, base
`6bd4c21a14b8180bb8b1357f878431a989b0a662`.
Producer: `probe_exporter_identity_mutation.py` (this directory), run through
`bounded_run.py`.

**Definitive runs of the committed probe:** `g11/H-005/11-probe-anchor.json` (measurement),
`09-selfcheck-anchor.json` (10/10), `10-lineage-failure-exit.json` (nonzero exit on
lineage failure).
**The defect being corrected:** `01-probe-prefix.json`.
**Superseded but kept on the record** — the first cut of the fix, which paired on
`raw_name` with a positional cross-check and produced the same five numbers:
`02-probe-corrected.json`, `03-probe-selfcheck.json`, `07-probe-final.json`,
`08-selfcheck-final.json`.
Interpreter `.venv/Scripts/python.exe` 3.13.6, Windows.
Leg: `runs_verify/2026-08-04_1647/papers/PMC12856317/strict`, committed.

## What was wrong with the previous number

The probe keyed **both sides on the raw `name`** (old `:78-79`). The exporter
renames `glycine → Glycine`, so `canonical_rows.get("Glycine")` returned `{}` and
every field on that compound was counted as added. **This is the historical record
of what was reported and is not deleted** — the pre-fix run is reproduced verbatim:

```
  Glycine
      identity fields changed : {'pathwhiz_id': (None, 78), 'db_id': (None, 78), 'db_status': (None, 'matched_offline_name_index'), 'chosen_rule': (None, 'legacy_pathwhiz_id_unverified'), 'pathbank_compound_id': (None, 78)}
      mapped_ids changed      : {'hmdb': (None, 'HMDB0000123'), 'cas': (None, '56-40-6'), 'chemspider': (None, '730'), 'kegg': (None, 'C00037'), 'pubchem': (None, '5257127'), 'chebi': (None, '15428'), 'biocyc': (None, 'GLY'), 'drugbank': (None, 'DB00145'), 'pathbank_compound_id': (None, '78')}
      *** 9 identifier(s) ADDED by the exporter: ['biocyc', 'cas', 'chebi', 'chemspider', 'drugbank', 'hmdb', 'kegg', 'pathbank_compound_id', 'pubchem']
...
  TOTAL identifiers added post-freeze: 10
```

All nine are in fact present in the canonical Glycine row's `mapped_ids`; only the
`chebi` **value** differs (`CHEBI:15428` → `15428`). The pairing bug also made the
probe **structurally unable to reach its own T-102 target of 0**, since the metric
grew with how faithfully the exporter recorded a rename.

## Corrected pairing — anchored on `mapping_meta.query.name`

The anchor is **`mapping_meta.query.name`**, written upstream of the freeze and carried
through the rename untouched. Measured: the canonical row named `glycine` and the IR row
named `Glycine` both carry `mapping_meta.query.name == "glycine"`. Matching is strict,
unique and **one-to-one**.

**`raw_name` is not used — not as anchor, not as fallback.** It is written by the code
under examination, so it cannot establish independent pre-canonical lineage. Position is
not used as a gate either: `PRODUCT_CONTRACT` § 5 permits ordering to differ, so a
positional disagreement is not evidence of anything.

> **Superseded 2026-08-06, same task.** The first cut of this fix paired on `raw_name`
> with a positional cross-check. It produced the same five numbers, but the anchor was
> author-controlled. Recorded so the change of anchor is visible, not silent.

Every lineage defect is loud, named, and **fails closed**: `LINEAGE_ANCHOR_MISSING`,
`AMBIGUOUS_LINEAGE_ANCHOR`, `UNMATCHED_IR_ROW`, `UNMATCHED_CANONICAL_ROW`. Anchor
presence is verified, never assumed — it is genuinely absent in production
(`runs/2026-08-02_2130/papers/PMC12856317/strict` has a protein row whose
`mapping_meta.query` is `null`).

**A lineage failure can never present as clean.** The probe still classifies the rows
that paired — suppressing them would be blind exactly when the exporter is dishonest —
but labels the counts a **LOWER BOUND**, prints `RESULT: UNDETERMINED` rather than
`RESULT: MEASURED`, and exits `2`.

`--selfcheck` (`09-selfcheck-anchor.json`): **10/10 passed**, covering both
`LINEAGE_ANCHOR_MISSING` sides, both `AMBIGUOUS_LINEAGE_ANCHOR` sides, `UNMATCHED_IR_ROW`,
`UNMATCHED_CANONICAL_ROW`, that a rename creating **no** `raw_name` still pairs via the
anchor, that a misleading `raw_name` does not affect pairing, that identical rows report
nothing and are `MEASURED`, and the critical negative:

```
  OK   clean paired row + unmatched row cannot pass   all_five_counts_zero=True exit=2 undetermined=True measured=False
```

`--demo-lineage-failure` (`10-lineage-failure-exit.json`) makes the same guarantee visible
as a **process** exit: `exit_reason: nonzero`, `exit_code: 2`, all five counts 0.

## Corrected measurement — five categories, never summed

Anchor-paired run, `11-probe-anchor.json` (rows are anchor-sorted; per-row content is
identical to the earlier `raw_name`-paired run):

```
    pairing anchor: mapping_meta.query.name (pre-freeze); strict one-to-one;
    raw_name is NOT used, as anchor or fallback -- the exporter writes it
    4 canonical rows, 4 IR rows, 4 paired, 0 lineage problem(s)

  canonical 'heme'  ->  IR 'heme'
      2. mapped-ID changes            : 1
          - mapped_ids.pubchem ADDED '3334' [within-row re-projection of canonical pubchem_cid]
      4. prefix normalization         : 2   (mapped_ids.chebi and chebi_id, 'CHEBI:17627' -> '17627')
      5. identity materialization     : 4   (pathwhiz_id 1799, db_id 1799, db_status, chosen_rule)
  canonical 'aminolevulinic acid'  ->  IR 'aminolevulinic acid'
      4. prefix normalization         : 2   ('CHEBI:17549' -> '17549')
      5. identity materialization     : 4   (pathwhiz_id 894, db_id 894, db_status, chosen_rule)
  canonical 'glycine'  ->  IR 'Glycine'
      1. name changes                 : 1
          - 'glycine' -> 'Glycine' (raw_name='glycine'; no provenance record accompanies it)
      3. synthetic database rows      : 1
          - db_row {"id": 78, "name": "Glycine"} fabricated at export time
      4. prefix normalization         : 2   ('CHEBI:15428' -> '15428')
      5. identity materialization     : 4   (pathwhiz_id 78, db_id 78, db_status, chosen_rule)
  canonical 'succinyl-CoA'  ->  IR 'succinyl-CoA'
      4. prefix normalization         : 2   ('CHEBI:15380' -> '15380')
      5. identity materialization     : 4   (pathwhiz_id 808, db_id 808, db_status, chosen_rule)

  ---- FIVE CATEGORIES, REPORTED SEPARATELY, NEVER SUMMED ----
    1. name changes                 :   1 instance(s) across 1 of 4 paired compound row(s)
    2. mapped-ID changes            :   1 instance(s) across 1 of 4 paired compound row(s)
    3. synthetic database rows      :   1 instance(s) across 1 of 4 paired compound row(s)
    4. prefix normalization         :   8 instance(s) across 4 of 4 paired compound row(s)
    5. identity materialization     :  16 instance(s) across 4 of 4 paired compound row(s)

  RESULT: MEASURED -- all 4 row(s) paired one-to-one on mapping_meta.query.name.
```

Also observed and deliberately **not** counted in the five: an exporter-internal
handle `key = cmp_1..cmp_4` (structural, not a biological claim), and `enrichment`
present on every canonical row and absent from every IR row.

## Before / after

| | old (name-paired) | corrected |
|---|---|---|
| identifiers "added post-freeze" | **10** | **1** — heme `mapped_ids.pubchem='3334'`, a within-row re-projection of that row's own `pubchem_cid` |
| Glycine identifiers "added" | 9 | **0** — all nine are in the canonical row |
| name changes | not reported | **1** |
| synthetic `db_row` | not reported | **1** |
| prefix normalization | folded into the identifier diff | **8** across 4/4 rows |
| identity materialization | reported, uncounted | **16** across 4/4 rows |
| unmatched / ambiguous rows | silently became "added" | loud, named, `UNDETERMINED`, exit 2, never clean |
| pairing anchor | raw `name` (post-rename) | `mapping_meta.query.name` (pre-freeze), strict 1:1 |

## The violation is unchanged

`PRODUCT_CONTRACT` § 5 is violated **by kind, not by count**: a post-freeze entity
rename with no provenance record, a fabricated `db_row`, `CHEBI:` stripping on 4 of
4 compounds and identity materialization on 4 of 4. Only the magnitude was
overstated.

**Acceptance for C-050/C-051:** all five counts 0 **and** `RESULT: MEASURED`. That is the
same requirement as "the diff is empty", stated per category so none can be traded
against another.

**NOT a sufficient condition for T-102.** `DECISIONS.md` **D-016 (LOCKED)** rules that
T-102 equivalence is *not narrowed to compounds*: it must verify **both** compound
identity **and** organism/species equivalence, across canonical JSON, PWML **and** SBML
(`TEST_MATRIX.md:265` agrees). This probe measures compound rows in the canonical JSON
only. **Passing it does not satisfy T-102.** An earlier revision of this file said
"T-102 acceptance is now per category: all five must be 0"; that under-scoped T-102 and
is corrected here rather than removed. Two measured examples of the dimension this probe
does *not* cover, both on the same leg: `ALAS2` gains `pathwhiz_id 17`
(`finding_alas2_identity_placeholder_2026-08-06.md`), and the species row moves from
`entities.species` to a top-level `species` group, gaining `pathwhiz_id 1` and losing the
`mapping_meta` that carried its `taxonomy_backfill` provenance.

Measurement B of the probe is unaffected by the pairing defect and is unchanged.
