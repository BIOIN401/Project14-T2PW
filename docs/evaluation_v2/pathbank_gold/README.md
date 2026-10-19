# Paper gold for the PathBank lab pathways (ORCH-747)

Two golds per pathway, by the product owner's decision of 2026-10-13:

1. **the lab's curated PathBank record**, read straight out of the MySQL instance, and
2. **a paper gold**, drafted by a model from one pinned full text, every row carrying a
   verbatim quote, with a human still to certify it.

The second exists because the paper fed to the pipeline is not the source the curator
worked from, so the two can disagree legitimately. Measuring that disagreement is the
point, not an obstacle to work around.

**Nothing here is certified.** Every assembled case carries
`supported_reactions_complete: false`, so `report.evaluate` skips it and no precision,
recall or F1 is computed from it. `paper_gold.assemble` writes that flag `false` whatever
the draft claims: a model certifying its own exhaustiveness would be worthless.

## Files, for PW701720 (Ara operon Inactivation, Autoregulation of AraC)

| file | what it is |
|---|---|
| `PW701720__extract.json` | the curated pathway as stored: 1 reaction, 7 interactions, participants with their HMDB/KEGG/ChEBI/UniProt ids, compartments, the curator's cited references, `updated_at` and a content hash |
| `PW701720__draft_case.json` | the same reactions in gold shape with **empty quotes by construction**. Not gold, and `gold.load_supplement` refuses it. It is the scope hint handed to the drafter |
| `PMC319880__source.json` | the paper's manifest: citation, PDF and text hashes, why no machine-readable full text exists, and all three text-extraction revisions with the reason each was superseded |
| `PMC319880__draft_blind_final.json` | drafted with no sight of the curated record |
| `PMC319880__draft_hinted_final.json` | drafted with the curated record as a scope hint, and carrying the alignment verdicts |
| `PMC319880__blind_vs_hinted.json` | agreement between the two passes, and the contamination check |
| `gold_paper_pw701720.json` | the hinted draft assembled into a loadable supplement |
| `gold_paper_pw701720_blind.json` | the blind draft, same treatment, as an independent check |

## Commands

```bash
# 1. read a curated pathway (read-only; PW ids resolve via super_pathway_visualizations)
python -m t2pw.bench.eval_v2.pathbank_gold --out docs/evaluation_v2/pathbank_gold PW701720

# 2. a drafter writes a draft against the pinned text, following
#    src/t2pw/bench/eval_v2/prompts/paper_gold_drafter_v1.md

# 3. assemble and verify every quote with the repository's own loader
python -m t2pw.bench.eval_v2.paper_gold assemble \
  --draft docs/evaluation_v2/pathbank_gold/PMC319880__draft_hinted_final.json \
  --source docs/evaluation_v2/pathbank_gold/PMC319880__source.json \
  --out docs/evaluation_v2/pathbank_gold/gold_paper_pw701720.json \
  --fulltext-dir C:/t/eval-fulltext

# 4. is the hinted pass contaminated by the hint?
python -m t2pw.bench.eval_v2.paper_gold compare --blind ... --hinted ...

# 5. score a run against it WITHOUT touching the pinned four-case supplement
python -m t2pw.bench.eval_v2.report evaluate --out <dir> \
  --supplement docs/evaluation_v2/pathbank_gold/gold_paper_pw701720.json
```

The paper text itself is not committed, matching `t2pw.bench.eval_v2.fetch`. It lives at
`C:/t/eval-fulltext/PMC319880.txt`, and `gold.load_fulltext` refuses any text whose sha256
differs from the manifest, so a result can never be produced against the wrong revision.

## Open for the product owner

- **The curated record contradicts the paper on two edges.** `PW_I016132` says the araO1
  operator *activates* the araC promoter; the paper's central result is that occupancy of
  that site *represses* it by competing with RNA polymerase. The sign is a real claim, not
  a drawing convention: the same pathway uses `Inhibition` on `PW_I016136`. `PW_I016133`
  encodes the model the paper's Discussion argues against. Classify each as a
  `gold_data_defect`, a `policy_disagreement`, or a legitimate difference between this
  paper and the two Schleif reviews the curator cited.
- **`PW_I016134` is left at `not_stated`**, not `contradicted`: the paper is silent on CAP
  acting at araI rather than opposed. A product-owner call.
- **CAP is typed as a nucleic acid** (`nucleic_acids` 275) although it is a protein and
  PathBank holds `proteins` 10248, UniProt P0ACJ8. Recorded as an observation only.
- **This pathway has no metabolite conversions.** Its one reaction is araC gene to AraC
  protein, and the eval-v2 matcher scores two-sided metabolite conversions only. So the
  reaction metrics have almost nothing to measure here and the mechanism sits entirely in
  interactions, which are recorded but not scored. The scoring rule for signaling,
  physiological and disease pathways is still undecided.
- **Two spans of the paper remain unquotable**, where text lines cross figure regions on
  page 3. Both drafters left them out rather than stitching fragments, which was right:
  one of them carries a negation that a stitched fragment would invert. A human with the
  printed page can supply them.
