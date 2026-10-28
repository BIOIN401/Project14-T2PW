# ORCH-786: a family-complex member with no id must not stop the pathway (wnt_ca)

Branch `orch786/fix` off `22538dee`.
- Code commits: `3a90e354` (the member is dropped, then the Unknown fallback) and `1e342ce6` (D-118 § 1 slash-list support, plus tests).
- Status: built and tested. Not merged, not pushed.

## 1. Root cause (shown by replay)

ORCH-784 `wnt_ca-strict` stopped at `post_remap` with `generated_wrapper_component_missing_external_identity` on **STIM2**.
- The paper writes "STIM1/2" and never "STIM2".
- Identity admission pass A therefore withheld STIM2's accession. That happened *after* two things were built:
  - the ORCH-783 family complex `STIM1/2`, whose source-1 members STIM1 and STIM2 reuse the existing STIM2 row;
  - the single-protein wrapper `STIM2 complex`.
- Both complexes then held an id-less STIM2. The base replay names both errors: `protein_complexes/7/components/0` and `/8/components/1`.

## 2. Change

| file | function | change |
|---|---|---|
| `src/t2pw/mapping/identity_admission.py` | `identity_support` | New optional `listed_members` (member key -> list name). A candidate found there is `supported`, `matched="STIM2 (from 'STIM1/2')"`. This is D-118 § 1. |
| `src/t2pw/mapping/map_ids.py` | `_admit_identities` | Builds `listed_members` from every entity row whose name is a slash list (ORCH-783 `_paper_named_members`) **and** whose normalized name occurs in the source index as an exact token run. |
| `src/t2pw/mapping/map_ids.py` | `_drop_unidentified_family_members` (new); `map_payload` calls it after `_admit_identities` | A family member still without a UniProt/DrugBank id is dropped, with a KI-001 note on `family_complex.note`. If no member is left, the complex gets the Unknown member (D-070, D-117 § 2 step 4). |
| `tests/test_orch786_family_member_identity.py` | new | 4 tests |

The token-run check is used instead of `SourceIndex.contains` for a reason. `contains` skips the bare "2", so "STIM1" alone would vouch for STIM2. The first tip run caught this: the drop test failed with `['STIM1','STIM2'] == ['STIM1']`.

The max-3 and species-match rules are untouched. Pass B, pass C and the route keys are untouched.

## 3. Evidence (all runs used the bounded wrapper with `--heavy-lock ORCH-786`; 0 survivors in every run)

| report | tree | what | result |
|---|---|---|---|
| `evidence/g11/ORCH-786/01-g9-base.json` | `C:/t/orch786/base` (22538dee) | focused file | 4 failed. 3 fail on `StageContractError` (missing identity). The guard test fails because STIM2 has no id on base. |
| `02-tip-focused.json` | tip | focused file | 4 passed |
| `03-replay-base.json` / `04-replay-tip.json` | base / tip | offline replay (`evidence/orch786_replay.py`) | base: **STOP**, 2 errors on STIM2. tip: **PASS**; STIM1/2 = [STIM1, STIM2]; admission withheld nothing. Summary: `evidence/orch786_replay.txt` |
| `05-tip-affected.json` | tip | focused + `test_orch783_family_complex`, `test_map_ids_mapping_lineage`, `test_c073_identity_admission`, `test_c075_source_support_armed` | 131 passed, 1 failed. The failure is the pre-existing c075 `Pantetheine 4'-phosphate` failure, the same as ORCH-783 `05-base-c075`; "no longer caught: []", so D-118 changes no corpus catch. |
| `00-replay-probe-base.json` | base | replay without the STIM2 injection | post_remap PASS. Offline, STIM2 never gets an id, so the failure cannot reproduce. |

Counts come from the tee'd logs (`C:/t/orch786/*.log`), because the wrapper JSON reports carry no stdout.

G9 labels:
- G9 behavioural, fail on base: `test_g9_slash_list_supports_each_member`, `test_g9_unsupported_member_is_dropped`, `test_g9_no_identified_member_left_falls_back_to_unknown`.
- `test_new_acceptance_slash_list_does_not_support_unwritten_member` checks that STIM3 is not supported. It also fails on base, through its STIM2 half.

Replay limits:
- The audit and curator rounds are not run.
- The Stage-2 cache is copied from the orch784run run.
- STIM2's UniProt answer (Q9P246) is injected, because the leg ran Stage 6 with `use_cache=False` and saved nothing.

## 4. Dead ends

- A first replay fully offline: post_remap passes on base, so it cannot prove anything. This is why STIM2's answer is injected.
- Test resolutions with `candidates: []`: the name gate rejects them (`identity_evidence_missing`). Each test resolution now carries one candidate.
- Stamping the family route key on reused member rows: rejected as weakening the gate (rule 6). It was superseded by D-118.

## 5. Open questions for the product owner

1. **Unknown fallback refused.** `_ensure_unknown_protein` returns None, for example when a real protein row is named "Unknown". An emptied family complex then still stops `post_remap` (`generated_wrapper_missing_components`). Options: (a) accept, since it is very rare; (b) route the case through strict_quarantine per-row omission, which costs a post_remap change.
2. **Single-protein wrappers.** `STIM2 complex` failed for the same reason, and D-118 fixes this case. But a single-protein wrapper whose protein is withheld for any other reason (not named and not in a list) still stops `post_remap`. The drop rule covers family complexes only.
3. **The index has no slashes.** Running text "STIM1 2" also counts as the list "STIM1/2". This is marked `ponytail:` in `_admit_identities`. Closing it means storing raw text in the source index.
