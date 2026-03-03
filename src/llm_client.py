import os
import time
import json
import hashlib
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_provider_raw = os.getenv("LLM_PROVIDER", "openrouter")
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


_request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "180"))
_max_retries = max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
_retry_base_seconds = float(os.getenv("LLM_RETRY_BASE_SECONDS", "1.2"))
_cache_enabled = os.getenv("LLM_ENABLE_CACHE", "1").strip().lower() not in {"0", "false", "no"}
_cache_max_items = max(8, int(os.getenv("LLM_CACHE_MAX_ITEMS", "256")))
_chat_cache: Dict[str, str] = {}


def _cache_key(
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    response_json: bool,
) -> str:
    payload = {
        "model": _model,
        "provider": PROVIDER,
        "temperature": round(float(temperature), 6),
        "max_tokens": int(max_tokens),
        "response_json": bool(response_json),
        "messages": messages,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "429",
        "rate limit",
        "timed out",
        "timeout",
        "503",
        "502",
        "gateway",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
    ]
    return any(marker in text for marker in markers)


def _cache_get(key: str) -> str:
    return _chat_cache.get(key, "")


def _cache_set(key: str, value: str) -> None:
    _chat_cache[key] = value
    while len(_chat_cache) > _cache_max_items:
        oldest = next(iter(_chat_cache))
        _chat_cache.pop(oldest, None)


def chat(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 800,
    *,
    response_json: bool = False,
    use_cache: bool = True,
) -> str:
    key = _cache_key(messages, temperature, max_tokens, response_json)
    if use_cache and _cache_enabled:
        cached = _cache_get(key)
        if cached:
            return cached

    kwargs: Dict[str, Any] = {
        "model": _model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": _request_timeout,
    }
    if PROVIDER == "openrouter":
        kwargs["extra_headers"] = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "pwml-pipeline"),
        }
    if response_json:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(1, _max_retries + 1):
        try:
            resp = _client.chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            if use_cache and _cache_enabled and content:
                _cache_set(key, content)
            return content
        except Exception as exc:  # noqa: BLE001
            text = str(exc).lower()
            if "response_format" in text or "json_object" in text:
                if "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    continue
            if attempt < _max_retries and _is_retryable_error(exc):
                time.sleep(_retry_base_seconds * attempt)
                continue
            raise

    return ""