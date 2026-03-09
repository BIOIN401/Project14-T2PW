import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError, AuthenticationError, BadRequestError

# -----------------------------------------------------------------------------
# Load .env reliably (project root = folder containing "src")
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

PROVIDER = (os.getenv("LLM_PROVIDER", "local") or "local").strip().lower()


def _mask(s: str, keep: int = 8) -> str:
    if not s:
        return "None"
    return s[:keep] + "..."


# -----------------------------------------------------------------------------
# Client setup
# -----------------------------------------------------------------------------
if PROVIDER == "openrouter":
    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    model = (os.getenv("OPENROUTER_MODEL") or "").strip()

    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY not loaded. Looked for .env at: {ENV_PATH}")
    if not model:
        raise RuntimeError("Missing OPENROUTER_MODEL in .env")
    if not api_key.startswith("sk-or-"):
        raise RuntimeError(
            f"OPENROUTER_API_KEY looks wrong ({_mask(api_key)}). "
            "OpenRouter keys usually start with 'sk-or-'."
        )

    _client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PWML Multi-Stage Pipeline"),
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


# -----------------------------------------------------------------------------
# Chat function (NOW supports response_json=True)
# -----------------------------------------------------------------------------
def chat(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 800,
    response_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    response_json=True:
      - If json_schema is provided, requests structured JSON using the schema.
      - Otherwise requests a JSON object output.
    """

    kwargs: Dict[str, Any] = {
        "model": _model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # OpenAI-compatible response_format
    if response_json:
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

    # Retry tuning (env configurable)
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "8"))
    base_sleep = float(os.getenv("LLM_RETRY_BASE_SLEEP", "1.0"))
    max_sleep = float(os.getenv("LLM_RETRY_MAX_SLEEP", "20.0"))

    # Spacing helps chunk pipelines avoid bursts
    spacing = float(os.getenv("LLM_CALL_SPACING", "0.35"))

    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(**kwargs)
            time.sleep(spacing)
            return (resp.choices[0].message.content or "").strip()

        except AuthenticationError as e:
            raise RuntimeError(
                f"Authentication failed (401). Key starts with: {_mask(os.getenv('OPENROUTER_API_KEY',''))}. "
                f"Checked .env at: {ENV_PATH}. Original error: {e}"
            ) from e

        except BadRequestError as e:
            # If response_format isn't supported by a specific OpenRouter model, you'll see this.
            raise RuntimeError(
                "Bad request (400). This model/provider may not support response_format JSON. "
                "Try OPENROUTER_MODEL=openrouter/free or a different model. "
                f"Original error: {e}"
            ) from e

        except RateLimitError as e:
            last_err = e
            time.sleep(min(base_sleep * (2 ** attempt), max_sleep))

        except (APITimeoutError, APIError) as e:
            last_err = e
            time.sleep(min(base_sleep * (2 ** attempt), max_sleep))

    raise RuntimeError(f"LLM failed after {max_retries} retries. Last error: {last_err}")