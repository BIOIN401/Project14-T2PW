# Curation brief — expected core reactions and major subprocesses for the ten benchmark papers

**Issued by the Lead Orchestrator, 2026-09-02, integration tip `7a1bb338`.**
**This is `HANDOFF.md` § 5.2a step 4, and D-088 clause 9's required replacement input.**

> **Until this dataset exists there is nothing to replace the entity-anchor cap WITH, and D-088
> clause 10 forbids removing the cap without a replacement.** This step is the long pole of the wave.

---

## 1. What you are producing, in one sentence

For each of the ten benchmark papers, a **curated statement of which reactions and which major
subprocesses a correct extraction of the requested pathway would have to recover from THAT PAPER**,
together with a typed classification of the participants those reactions involve.

---

## 2. The one rule that matters more than the schema

**Curate the biology in the paper. Do not curate toward an outcome.**

You are not told, and must not try to infer, which papers the pipeline currently passes or fails,
which entries would move a benchmark number, or what any gate does with your output. If you find
yourself reasoning about what result a choice would produce, stop and go back to the text.

This dataset's whole value is that it is an **independent** statement of what the paper contains. A
curation shaped by the answer it produces is worth less than no curation at all, because it would
launder a policy preference as biology. If you are genuinely unsure whether something is core, say so
in `uncertainties` and leave it out of the core list — an honest omission is recoverable, a
confident wrong inclusion is not.

**Every substantive claim carries a verbatim `quote` from the paper.** No quote, no entry. A quote
must be a contiguous span copied exactly from the full text, not a paraphrase and not stitched from
two places.

---

## 3. Your sources, in priority order

| # | Source | Path | Use |
|---|---|---|---|
| 1 | **The paper's full text** | `data/rag_index/acquire_cache/fulltext/<hash>.json`, field `full_text` | **Authoritative.** Every quote comes from here |
| 2 | The pinned gold case | `src/t2pw/bench/gold/pinned_v1.json`, the case with your `paper_id` | Corroboration and a sanity check. It is curated by the product owner and is good evidence — but it was written for a different purpose and is **not** a list of reactions |
| 3 | Archived Stage-0 drafts | `runs/*/papers/<id>/*/coverage_summary.json` and `runs_verify/...`, field `requested_context.main_subprocesses` | **Read these LAST and treat them as a hypothesis to check, never as an answer.** They are LLM output from up to eight independent draws and they disagree with each other. Their value is that a subprocess named in several independent draws is worth looking for in the text |

**Never derive an entry from what a pipeline run extracted.** Terms taken from what survived would
match whatever survived, which is not a test. `final_mapped.json`, `quarantine_report.json` and
`stage1_payload.json` are **off limits** for this task.

### Resolving your paper's full text

The cache filenames are hashes. Locate your paper by scanning for its `id`/`source`:

```python
import json, glob
for f in glob.glob("data/rag_index/acquire_cache/fulltext/*.json"):
    j = json.load(open(f, encoding="utf-8"))
    if "PMC12096016" in json.dumps({"id": j.get("id"), "source": j.get("source")}):
        text = j["full_text"]
```

Two cache entries exist per paper with identical byte counts; either is fine, but **record which
hash you read** in `source_cache_file`, and record `len(full_text)` in `source_chars`.

---

## 4. The distinctions the schema encodes — read these before writing anything

These four are the whole point of the dataset. Getting them right matters far more than completeness.

**(a) A core reaction.** A chemical transformation the paper asserts as part of the *requested*
pathway, with at least a substrate, a product, and normally a named catalyst. "The paper mentions
X" is not enough — the paper must assert the transformation.

**(b) A major subprocess.** A named stage, arm or branch of the pathway that groups one or more
reactions — *"the mevalonate pathway"*, *"lanosterol 14α-demethylation"*, *"NRPS assembly"*. A
subprocess is **major** if the paper treats it as a constituent stage of the requested pathway. A
subprocess may be one the paper names but does not itself detail chemically; record it anyway and
set `detailed_in_paper: false`, because a named-but-undetailed branch is still part of the pathway's
shape.

**(c) An IMPORTANT participant.** A participant is important when **any** of these holds — and you
must say which, in `reason`:

  - `defining_substrate_or_product` — the reaction is not that reaction without it;
  - `distinguishes_identity_or_direction` — without it you cannot tell which reaction this is, or
    which way it runs (an adenylation without ATP; a reduction without its reductant named as such);
  - `central_to_pathway_scope` — the paper's stated scope is about this molecule.

**(d) A SECONDARY participant.** An ordinary cofactor, currency metabolite, transcriptional
regulator, ancillary or accessory protein, water, or a proton — present in or around the chemistry
but not required to identify the reaction.

**The same molecule can be important in one reaction and secondary in another, and that is not a
contradiction.** ATP is a defining substrate of an adenylation step and a currency metabolite three
reactions later. When this happens, record it in **both** lists with the reaction ids that make each
true, and note it in `uncertainties`. Do not force a single global verdict on a molecule.

---

## 5. The schema — exact, and one JSON file per paper

Write `docs/pwml_recovery_sprint/curation/expected_core_<PAPER_ID>.json`:

```json
{
  "schema_version": 1,
  "paper_id": "PMC12096016",
  "requested_pathway": "enterobactin biosynthesis",
  "requested_organism": "Escherichia coli",
  "source_cache_file": "af4332adf5750b97dc7fbd2e01384141.json",
  "source_chars": 43667,
  "curated_by": "<your agent label>",
  "curation_date": "2026-09-02",

  "expected_core_reactions": [
    {
      "id": "R1",
      "description": "EntC converts chorismate to isochorismate",
      "substrates": ["chorismate"],
      "products": ["isochorismate"],
      "enzymes": ["EntC"],
      "important_participants": ["chorismate", "isochorismate"],
      "secondary_participants": [],
      "quote": "<verbatim contiguous span from full_text>",
      "confidence": "high",
      "rationale": "one sentence: why this is a core reaction OF THE REQUESTED PATHWAY"
    }
  ],

  "expected_major_subprocesses": [
    {
      "id": "S1",
      "name": "isochorismate formation",
      "aliases": [],
      "reaction_ids": ["R1"],
      "detailed_in_paper": true,
      "quote": "<verbatim span naming this stage>",
      "confidence": "high",
      "rationale": "one sentence"
    }
  ],

  "important_participants": [
    {
      "name": "ATP",
      "reason": "distinguishes_identity_or_direction",
      "reaction_ids": ["R4"],
      "quote": "<verbatim span>",
      "note": "adenylation is not identifiable as adenylation without it"
    }
  ],

  "secondary_participants": [
    {
      "name": "NADH",
      "class": "cofactor",
      "reaction_ids": ["R3"],
      "quote": "<verbatim span>",
      "note": "one sentence"
    }
  ],

  "out_of_scope": [
    {"name": "MenD", "reason": "menaquinone enzyme; the paper mentions it, it is not in the requested pathway"}
  ],

  "uncertainties": [
    "free text: anything you could not settle from the paper, and anything a reviewer should check"
  ]
}
```

Field rules:

- `confidence` is one of `high`, `medium`, `low`. **Use `low` freely.** A `low` entry that is honest
  is more useful than a `high` entry that is confident.
- `class` for a secondary participant is one of: `cofactor`, `currency_metabolite`, `regulator`,
  `ancillary_protein`, `water_or_proton`, `other`.
- `reason` for an important participant is one of the three exact strings in § 4(c).
- Every `quote` is verbatim and contiguous. **Verify each one appears in `full_text` by substring
  search before you write the file**, and say in `uncertainties` if any could not be verified.
- `out_of_scope` is where you put things the paper discusses that belong to a *different* pathway.
  It is a real and useful output: a paper that names a neighbouring pathway is a paper where an
  extraction can plausibly drift.

---

## 6. Also write one prose file

`docs/pwml_recovery_sprint/curation/CURATION-NOTES-<PAPER_ID>.md`, short, covering:

- what the requested pathway is in this paper and how much of it the paper actually delivers;
- which subprocesses the paper **names but does not detail** — call these out explicitly;
- anything that surprised you, any place the gold case and the paper appear to disagree, and any
  entry you nearly classified the other way.

**If the gold case and the paper disagree, say so plainly and do not resolve it.** You are not
authorised to change gold, and a disagreement you record is a finding; a disagreement you smooth over
is a defect you introduced.

---

## 7. Hard constraints

- **Write ONLY** under `docs/pwml_recovery_sprint/curation/`. Create the directory if needed.
- **Run no `git` command at all.** No add, no commit, no branch, no checkout, no stash. The Lead
  integrates.
- **Change no other file** — not gold, not source, not tests, not docs outside your directory.
- **Run no pipeline leg, no benchmark, no pytest, no LLM-backed command.** This task is reading and
  writing text. If you believe you need to run one, stop and report that instead.
- **Do not read** `final_mapped.json`, `quarantine_report.json`, `stage1_payload.json`, or any other
  record of what a run extracted. § 3 says why.
- Do not read `release_status.py` or `strict_quarantine.py`. What the code does with your output is
  not your input.

---

## 8. What to report back

A short report: the papers you covered, the count of core reactions and major subprocesses for each,
which entries you marked `low` confidence and why, every gold-vs-paper disagreement you found, and
anything you could not verify. **Report what you could not do as prominently as what you did.**
