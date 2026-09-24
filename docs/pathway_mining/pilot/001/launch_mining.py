"""ORCH-742a pilot launcher (outside the repo; no mining code is changed).

The mining CLI never loads the project .env, so its preflight reads LLM_PROVIDER
and OPENROUTER_API_KEY before anything has populated them. The Streamlit entry
path gets them because the app imports t2pw.config first. This launcher does the
same, with the project's own loader, then runs the unmodified CLI.
"""
import runpy
import sys

from t2pw.config import PROJECT_ROOT, ensure_dotenv_loaded

ensure_dotenv_loaded()
print(f"[launcher] project .env loaded from {PROJECT_ROOT / '.env'}", flush=True)
sys.argv = ["python -m t2pw.mining", *sys.argv[1:]]
runpy.run_module("t2pw.mining", run_name="__main__", alter_sys=True)
