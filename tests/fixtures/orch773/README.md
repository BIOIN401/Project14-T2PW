# ORCH-773 fixtures

Each `<leg>.freeze_input.json.gz` is the exact `{"payload", "pathway_context"}` that the ORCH-768 replay corpus
(`docs/pwml_recovery_sprint/evidence/orch768_replay_corpus.py`, `replay_leg`) hands `freeze_canonical_payload`
on that corpus leg, at base `3ef3fa9c` (`src` tree `322d398d`).

* For a `canonical` leg this is the leg's own `final_mapped.json`. For a `pre_freeze` leg (ORCH-755 enterobactin,
  ORCH-767 Wnt/Ca2+) it is that file after the replay's `run_enrichment` (pinned end-of-leg cache) and
  `run_prefreeze_resolution`, offline.
* `pathway_context` is the leg's `quarantine_report.json` `coverage.requested_context`, as the harness passes it.
* Captured by `evidence/orch773/capture_freeze_inputs.py` (G11 `ORCH-773/03`), gzipped with mtime 0 by
  `evidence/orch773/make_fixtures.py` (G11 `ORCH-773/12`). The sha256 of each uncompressed file is asserted
  by `tests/test_orch773_omit_per_row.py` (`SHA256`).

| fixture | corpus leg | what it carries |
|---|---|---|
| `ORCH-755_enterobactin-strict_2026-10-20_2201` | ORCH-755 enterobactin (PMC12096016) | EntB without UniProt, a component of the declared complex `isochorismatase` (quarantine `unexportable_entity:1`) |
| `ORCH-767_wnt_ca-strict_2026-10-24_0314` | ORCH-767 Wnt/Ca2+ (PMC10425814) | `phosphorylated CAMKII` without identity, kept alive only by the synonym `calcineurin B`; two non-canonical transporters |
| `ORCH-759_enterobactin-strict_2026-10-22_1037` | ORCH-759 enterobactin | the stray row `enterobactin synthase`, degree 0 by the final gate |
| `runs_verify_2026-08-21_2057_PMC12452463` | runs_verify | a generated `enterobactin synthase complex` with no component stoichiometry, its reaction's only catalyst |
| `ORCH-755_sour_taste-strict_2026-10-21_0139` | ORCH-755 sour taste (PMC7059786) | three transports whose transporter is in Stage-1 shape (`actor_schema_not_canonical`) |
| `runs_verify_2026-08-21_2057_PMC12096016` | runs_verify enterobactin | the bare protein enzyme `lactate dehydrogenase` (`reaction_enzyme_must_be_protein_complex`) |
| `runs_verify_2026-08-25_1216_PMC12856317` | runs_verify heme | the complex `LONP1`, none of whose components the IR resolves (`protein_complex_missing_components`) |
| `runs_verify_2026-08-21_2239_PMC12180156` | runs_verify | no reaction with defensible support (`no_defensible_reaction_support`), a whole-pathway refusal |

The ORCH-767 enterobactin and sour-taste legs are read from `tests/fixtures/orch768/` instead.
