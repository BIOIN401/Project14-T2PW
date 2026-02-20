import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_provider_raw = os.getenv("LLM_PROVIDER", "local")
PROVIDER = _provider_raw.split("#", 1)[0].strip().lower()

if PROVIDER == "openrouter":
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")
    if not model:
        raise RuntimeError("Missing OPENROUTER_MODEL in .env")
    _client = OpenAI(base_url=base_url, api_key=api_key)
    _model = model
elif PROVIDER == "local":
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    model = os.getenv("LMSTUDIO_MODEL", "meta-llama-3.1-8b-instruct")
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
    if PROVIDER == "openrouter":
        kwargs["extra_headers"] = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "pwml-pipeline"),
        }

    resp = _client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    return (content or "").strip()
