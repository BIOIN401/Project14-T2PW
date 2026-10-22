# ORCH-759a fixtures

`enterobactin_final_mapped.excerpt.json` is an excerpt of the canonical payload of the ORCH-759
enterobactin 3-draw strict leg:

- source: `runs_papers/ORCH-759/legs/enterobactin-strict/2026-10-22_1037/papers/PMC12096016/strict/final_mapped.json`
  (primary checkout, uncommitted run scratch), file sha256
  `b2ab0c94ef7bef6899f5e312f8eb5c011932c2b3ec2bdecf8c5aaec37bb463f7`, canonical `payload_sha256`
  `e72d3a65d689269b1379cafd555c3dbac5e1db08fcda3c8c16701c5af4597d99` (the value its
  `final_stage3_gate_report.json` is bound to);
- cut by `docs/pwml_recovery_sprint/evidence/orch759a/make_fixture.py`, which removes every entity
  row's `enrichment` block (Stage 7 text, read by no gate) and the top-level `source_text_index`
  (a normalized copy of the paper; without it identity-admission pass A is `not_evaluated` and keeps
  every accession, the same outcome the leg recorded). Nothing else is changed;
- excerpt file sha256 `66c1d1ffe8791783ee7f1ffca5dc76afac648611723577365ed6623d69a86cfd`.

It fails the final pre-export gate exactly as the leg did: one error,
`/entities/proteins/7/name | Protein has degree 0 after normalization: enterobactin synthase`
(`tests/test_orch759a_duplicate_protein_fold.py::test_the_excerpt_carries_the_legs_defect`).
