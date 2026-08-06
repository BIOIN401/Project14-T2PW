# FINDING — ALAS2 ships as an `Unknown` placeholder while its accession is twice in the paper

**Status: UNRESOLVED FINDING. Not a fix, not a proposed fix, and it closes nothing —
in particular it does not close O-1.**
Recorded by H-005, 2026-08-06, branch `agent/h05-probe-and-authority-corrections`.
Surfaced while correcting the exporter-identity probe; escalated because it was
otherwise going to exist only in a chat report, which does not survive.
Nothing under `src/`, no gold-set entry, no pathway output and no production biology was
modified in recording it.

## The affected leg and artifacts

| | |
|---|---|
| Leg | `runs/2026-08-02_2130/papers/PMC12856317/strict` (committed; `RESULT.txt` → `RESULT: FAIL`, stage `post_pipeline`) |
| Artifact | `runs/2026-08-02_2130/papers/PMC12856317/strict/final_mapped.json` |
| Paper evidence | `runs/2026-08-02_2130/papers/PMC12856317/01_source_text.txt` (54 764 bytes) |
| Companion leg | `runs/2026-08-02_2130/papers/PMC12856317/research` — `RESULT: PASS (with warnings)` |

## What shipped

The sole protein row in the **strict** leg's canonical payload:

```json
{ "name": "Unknown",
  "uniprot_id": "Unknown",
  "mapped_ids": { "uniprot": "Unknown", "pathbank_protein_id": 9659 },
  "identity_status": "placeholder",
  "organism": "Arabidopsis thaliana",
  "mapping_meta": {
    "chosen_rule": "pathbank_unknown_protein_fallback",
    "confidence": 0.0,
    "fallback_used": true,
    "fallback_reason": "all_normal_protein_identity_strategies_failed",
    "cross_species_placeholder": true,
    "resolution": { "status": "fallback", "issue": "pathbank_unknown_sentinel" },
    "placeholder_target_organisms": ["Homo sapiens"] } }
```

The protein complex in the same payload is named `ALAS2 homodimer`, and its **only
component** is that placeholder:

```json
{ "name": "ALAS2 homodimer",
  "components": [ { "name": "Unknown", "stoichiometry": 1,
                    "pathbank_protein_id": 9659,
                    "mapped_ids": { "uniprot": "Unknown", "pathbank_protein_id": 9659 } } ] }
```

So a pathway whose subject is human ALAS2 shipped its enzyme as a literal string
`"Unknown"`, carried on a PathBank row for *Arabidopsis thaliana* (`species_id 4`,
`taxonomy_id "3702"`), with a `cross_species_placeholder` flag and confidence `0.0`.

The complex row is additionally **self-inconsistent on organism**: `organism` and
`species` say `"Arabidopsis thaliana"` with `pathbank_species_id 4`, while
`species_name`, `taxonomy_id "9606"` and `species_ref` on the same row say
*Homo sapiens*. It is marked `generated: true`,
`generation_reason: "single_protein_pathwhiz_wrapper"`. **This is the organism/species
dimension `D-016` places inside T-102**, which is why it is recorded here rather than
left in the compound probe's scope.

## The accession was in the supplied evidence — twice

`P22557` occurs **2 times** in `01_source_text.txt`, at byte offsets 30822 and 35197:

1. > "The amino acid sequences of human ( Homo sapiens ; P13196 , **P22557** ), bovine …"
2. > "Two copies of mature ALAS2 (residues 54–587, UniProt ID **P22557** ) and two copies of
   > heme b ligand were entered as the search query."

The second names the accession and the protein together in one sentence. `ALAS2` itself
occurs 97 times in the same file.

That the correct accession is `P22557` for human ALAS2 is independently corroborated by a
different committed leg: `runs_verify/2026-08-04_1647/papers/PMC12856317/strict/final_mapped.json`
carries `ALAS2` with `mapped_ids.uniprot = "P22557"` and `uniprot_id = "P22557"`, and its
shipped `pathway.pwml` contains `P22557`.

## The `research` leg fails differently, and also fails

`runs/2026-08-02_2130/papers/PMC12856317/research/final_mapped.json` keeps the name
`ALAS2` but ships `uniprot_id: null` and `mapped_ids: {}` — no identity at all, rather
than a wrong one. Its complex component is `{"name": "ALAS2", "stoichiometry": 2}`.
Same paper, same evidence, two different identity outcomes, neither correct.

## Why this is recorded and not acted on

* It is **not** the defect H-005 was dispatched to fix, and it is not in H-005's owned
  file set. No code was changed for it.
* **No biological gate may be weakened to make this resolve.** The placeholder is a
  refusal, and a refusal that the pipeline could not justify is the thing to examine —
  not something to route around by loosening admission.
* A fix must not be inferred from this record. Whether the failure is in accession
  extraction, in the identity ladder, or in the `pathbank_unknown_protein_fallback` rule
  preferring a cross-species PathBank row over no row at all, is **not established here**.
  `MASTER_PLAN.md` § 1.4 documents a *different* identity-ladder defect on a different
  leg; whether these share a mechanism is unverified.
* **`Unknown` is a fabricated identifier value.** `mapped_ids.uniprot = "Unknown"` is not
  a null and not a `not_evaluated`; it is a well-formed-looking string occupying an
  accession slot. Any consumer that checks only for presence will read it as an identity.
  That is worth a decision independently of whether `P22557` is ever recovered.

## Reproduction

```bash
<py> docs/pwml_recovery_sprint/evidence/bounded_run.py --label alas2 --timeout 300 \
  --json <allocated> -- <py> -c "import json; d=json.load(open('runs/2026-08-02_2130/papers/PMC12856317/strict/final_mapped.json',encoding='utf-8')); print(d['entities']['proteins']); print(d['entities']['protein_complexes'])"
```

Measured under `evidence/g11/H-005/12-alas2-finding.json`.
