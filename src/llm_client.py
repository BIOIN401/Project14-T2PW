import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError, AuthenticationError

# ---- Force load .env from project root (same folder as src) ----
# llm_client.py is inside src/, so parents[1] is the folder containing src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

PROVIDER = (os.getenv("LLM_PROVIDER", "local") or "local").strip().lower()


def _mask(s: str, keep: int = 8) -> str:
    if not s:
        return "None"
    return s[:keep] + "..."


if PROVIDER == "openrouter":
    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    model = (os.getenv("OPENROUTER_MODEL") or "").strip()

    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY not loaded. Looked for .env at: {ENV_PATH}")
    if not api_key.startswith("sk-or-"):
        raise RuntimeError(
            f"OPENROUTER_API_KEY looks wrong ({_mask(api_key)}). "
            "OpenRouter keys usually start with 'sk-or-'."
        )
    if not model:
        raise RuntimeError("Missing OPENROUTER_MODEL in .env")

    _client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PWML App"),
        },
    )
    _model = model

elif PROVIDER == "local":
    base_url = (os.getenv("LMSTUDIO_BASE_URL") or "http://127.0.0.1:1234/v1").strip()
    model = (os.getenv("LMSTUDIO_MODEL") or "meta-llama-3.1-8b-instruct").strip()

    _client = OpenAI(base_url=base_url, api_key="lm-studio")
    _model = model

else:
    raise RuntimeError("LLM_PROVIDER must be 'local' or 'openrouter'")


def chat(messages: List[Dict[str, Any]], temperature: float = 0.0, max_tokens: int = 800) -> str:
    kwargs: Dict[str, Any] = {
        "model": _model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "6"))
    base_sleep = float(os.getenv("LLM_RETRY_BASE_SLEEP", "1.0"))

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        except AuthenticationError as e:
            # DO NOT retry auth errors (401). This is a key/env issue.
            raise RuntimeError(
                f"Authentication failed (401). "
                f"Key loaded starts with: {_mask(os.getenv('OPENROUTER_API_KEY',''))}. "
                f"Checked .env at: {ENV_PATH}. "
                f"Original error: {e}"
            ) from e

        except RateLimitError as e:
            last_err = e
            sleep = min(base_sleep * (2 ** attempt), 20)
            time.sleep(sleep)

        except (APITimeoutError, APIError) as e:
            last_err = e
            sleep = min(base_sleep * (2 ** attempt), 20)
            time.sleep(sleep)

    raise RuntimeError(f"LLM failed after {max_retries} retries. Last error: {last_err}")