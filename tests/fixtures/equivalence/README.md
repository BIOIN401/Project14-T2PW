# Equivalence fixture (R2.5)

`sour_taste_v2_strict.json.gz` is the record/replay cassette for `tests/test_equivalence_app_vs_service.py`:
one live strict run of the app on one paper, holding the LLM, HTTP and PathBank DB answers, the paper's full
text, and the run's recorded outputs. See `docs/webserver/results/R2.5.md`.

## Source text and attribution

The fixture contains the full text of this article, verbatim, plus copies of it inside the recorded LLM
prompts:

- **Authors:** Sue C. Kinnamon, Thomas E. Finger
- **Title:** Recent advances in taste transduction and signaling
- **Journal:** F1000Research, 2019
- **DOI:** [10.12688/f1000research.21099.1](https://doi.org/10.12688/f1000research.21099.1)
- **PMCID:** PMC7059786 (PMID 32185015)
- **Licence:** Creative Commons Attribution (CC BY), as reported by Europe PMC (`"license": "cc by"`). The raw
  response is in `docs/webserver/evidence/R2.5/licence-PMC7059786-europepmc.json`.

The text was fetched with `t2pw.batch.fetch.fetch_papers` and is unmodified. Its sha256 is in the fixture
(`case.full_text_sha256`).
