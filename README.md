# T2PW

T2PW extracts pathway descriptions into pathway JSON, enriches and validates the result, and exports PWML as the primary output. A legacy SBML path is still available for compatibility.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Configure the LLM provider in `.env`:

```text
LLM_PROVIDER=local
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=meta-llama-3.1-8b-instruct
```

For OpenRouter, set `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, and `OPENROUTER_MODEL`.

## Run The App

```powershell
streamlit run src/t2pw/app/streamlit_app.py
```

## CLI Paths

Legacy SBML export:

```powershell
python scripts/run.py --in final.json --out-dir outputs --no-llm-audit --no-sbml-overwatch
```

PWML export from mapped final JSON:

```powershell
python scripts/run_pwml.py --in final.mapped.json --out-dir outputs --non-strict-db
```

## Project Layout

- `src/t2pw/app/`: Streamlit application.
- `src/t2pw/pipeline/`: extraction, inference, graph QA, normalization, and orchestration.
- `src/t2pw/pwml/`: PWML IR, writer, validation, rendering, and QA.
- `src/t2pw/sbml/`: legacy SBML generation and rendering helpers.
- `src/t2pw/mapping/`: ID mapping, grounding, enrichment, and PathBank database helpers.
- `src/t2pw/curation/`: audit, patch application, gap resolution, and pathway curation.
- `src/t2pw/llm/`: LLM client and prompt files.
- `tools/ruby/`: Ruby parser utilities retained for reference and ad hoc tooling.
- `docs/`: setup, architecture, and pipeline notes.

## More Documentation

- [Setup](docs/setup.md)
- [Architecture](docs/architecture.md)
- [Pipeline](docs/pipeline.md)
