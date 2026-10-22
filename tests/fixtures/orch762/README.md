# ORCH-762 fixtures

Read by `tests/orch762_harness.py` and `tests/test_orch762_protein_identity.py`. No test touches
the network; every outside answer is one of these files or a documented mock.

| file | what it is | how it was made |
|---|---|---|
| `menaquinone_kf147_merged.json` | ORCH-756 1-draw menaquinone leg `merged_payload.json` (Stage-2 input), source sha256 `57866e3e…dc51423` | species rows gain the `taxonomy_id` / `classification` the leg's own NCBI backfill wrote (`final_mapped.json` sha256 `ad1ec861…1b21`: 684738 and 1590, Prokaryote); nothing else changed |
| `menaquinone_lactis_species.json` | the same payload in the ORCH-755-night shape (organism read as "Lactococcus lactis", ORCH-761 § 4.2 / extract § B) | the nine `proteins[].organism` strings and the species row renamed, species taxonomy id 1358 |
| `enterobactin_orch755_merged.json` | ORCH-755 enterobactin leg `merged_payload.json` (the holo-EntB night), source sha256 `c7e2af1e…f740b3` | species row gains taxonomy id 562, Prokaryote |
| `enterobactin_orch755_resolutions.json` | the identity each protein row of that leg's `final_mapped.json` (sha256 `50c53e42…727f`) recorded, a withheld accession put back | `orch762_harness.recorded_resolutions` |
| `kf147_leg_cache_entries.json` | the ORCH-756 leg's own 18 mapping-cache rows for the nine Men proteins (`api-v8::…kf147…` best efforts, `db::…` `species_not_found`), verbatim; cache sha256 `de78cd56…3603` | |
| `uniprot_responses.json` | 111 UniProt REST answers (UniProtKB search, UniParc search and entry, UniProt taxonomy), keyed by URL + sorted query string, one answer per line | recorded ONCE live on 2026-10-22 by `docs/pwml_recovery_sprint/evidence/orch762/record_fixtures.py`, for the base (`493c1a10`) and the tip query shapes of every scenario; G11 reports `evidence/g11/ORCH-762/01-record-base.json`, `02-record-tip.json` |

The payload fixtures are cut by `docs/pwml_recovery_sprint/evidence/orch762/make_fixtures.py`
(it prints every source and output sha256). Mocked, not recorded: ChEBI / KEGG / HMDB (compound
lookups end `novel`), EuropePMC (an alias table the test passes: the leg's own `holo-EntB -> EntB`),
NCBI (backfill off; the taxonomy ids above are what it wrote), PathBank (no database; the
enterobactin protein rows are answered from the leg's recorded resolutions), and every model call.
UniProt answers are as of 2026-10-22; entries move, which is why they are recorded.
