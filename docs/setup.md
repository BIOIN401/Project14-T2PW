# Setup

Create and activate a virtual environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
streamlit run src/t2pw/app/streamlit_app.py
```

The LLM client reads `.env` from the project root. Local LM Studio is the default:

```text
LLM_PROVIDER=local
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=meta-llama-3.1-8b-instruct
```

OpenRouter is also supported:

```text
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=...
```

Optional PathBank database-backed mapping uses:

```text
PATHBANK_ID_SOURCE=hybrid
PATHBANK_DB_HOST=...
PATHBANK_DB_PORT=3306
PATHBANK_DB_USER=...
PATHBANK_DB_PASSWORD=...
PATHBANK_DB_SCHEMA=pathbank
```
