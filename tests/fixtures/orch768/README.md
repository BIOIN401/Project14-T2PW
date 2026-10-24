# ORCH-768 replay-corpus fixture

Inputs of three ORCH-767 strict legs, gzipped, for `tests/test_orch768_replay_corpus.py` (the ORCH-768 new-capability
acceptance test, D-112 § 1). They are written by
`docs/pwml_recovery_sprint/evidence/orch768_replay_corpus.py fixture` (gzip `mtime=0`, so the bytes reproduce).

| leg | recorded verdict | why it is here |
|---|---|---|
| `ORCH-767/sour_taste_v2-strict/2026-10-24_0042` | PWML written (`pathway.review_required.pwml`) | the pass path; the replayed PWML must be byte-identical to the leg's (pinned by sha256 only, bytes not stored) |
| `ORCH-767/enterobactin-strict/2026-10-23_2224` | fails `final_pre_export_stage3_gates` (wrapper component) | the final-gate failure path, compared error by error |
| `ORCH-767/wnt_pcp-strict/2026-10-24_0131` | crashes opening refinement review (no reaction nodes) | the refinement-crash path |

`legs.json` is the corpus manifest's entry for each leg with paths made fixture-relative. Each `sha256` is the
**corpus manifest's** (`docs/pwml_recovery_sprint/evidence/orch768_corpus_manifest.json`), so a decompressed file
that matches it is the leg's own input, byte for byte. The test checks that.

The legs ran on `d6849c9e`, whose `src` tree equals `main`'s (`e3d5b65f`). A lane that changes one of these
verdicts on purpose moves `EXPECTED` in the test with its documented delta (merge rule 4). Size: about 190 KB.
