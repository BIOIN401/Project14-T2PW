"""Per-run provider override and the ``vllm`` provider. Offline: a fake HTTP transport, no network.

``t2pw.llm.client`` picks its provider at import, so each test executes a FRESH copy of the
module from its file with the environment it wants (``load_dotenv`` is no-op'd: a developer's
real ``.env`` must not decide a test).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import httpx
import openai
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CLIENT_FILE = SRC / "t2pw" / "llm" / "client.py"
ENV_VARS = (
    "T2PW_LLM_PROVIDER", "LLM_PROVIDER", "VLLM_BASE_URL", "VLLM_MODEL", "VLLM_API_KEY",
    "VLLM_MAX_TOKENS", "LLM_MAX_TOKENS", "OPENROUTER_EXTRACTION_MAX_TOKENS",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "LMSTUDIO_BASE_URL", "LMSTUDIO_MODEL",
)


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    requests: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={
            "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{\"ok\": true}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    real_openai = openai.OpenAI

    def faked(**kwargs: Any):
        return real_openai(http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", faked)

    def _load(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        spec = importlib.util.spec_from_file_location("t2pw_llm_client_under_test", CLIENT_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, requests

    return _load


def test_unset_keeps_todays_provider(load) -> None:
    mod, _ = load()
    assert mod.PROVIDER == "local"
    mod, _ = load(LLM_PROVIDER="local", LMSTUDIO_MODEL="lm-x")
    assert (mod.PROVIDER, mod._model) == ("local", "lm-x")


def test_override_wins_over_llm_provider(load) -> None:
    mod, _ = load(LLM_PROVIDER="local", T2PW_LLM_PROVIDER="vllm", VLLM_MODEL="m")
    assert mod.PROVIDER == "vllm"


def test_blank_override_is_ignored(load) -> None:
    mod, _ = load(LLM_PROVIDER="local", T2PW_LLM_PROVIDER="")
    assert mod.PROVIDER == "local"


def test_process_env_survives_dotenv_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The reason for the snapshot: ``load_dotenv(override=True)`` would otherwise beat the process."""
    env_file = tmp_path / ".env"
    env_file.write_text("T2PW_LLM_PROVIDER=local\nVLLM_MODEL=from-dotenv\nLLM_PROVIDER=local\n")
    monkeypatch.setenv("T2PW_LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL", "from-process")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")  # so teardown undoes what dotenv writes
    monkeypatch.setattr("t2pw.paths.PROJECT_ROOT", tmp_path)
    spec = importlib.util.spec_from_file_location("t2pw_llm_client_dotenv", CLIENT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PROVIDER == "vllm"
    assert module._model == "from-process"


def test_vllm_requires_a_model(load) -> None:
    with pytest.raises(RuntimeError, match="VLLM_MODEL"):
        load(T2PW_LLM_PROVIDER="vllm")


def test_unknown_provider_still_raises(load) -> None:
    with pytest.raises(RuntimeError, match="vllm"):
        load(T2PW_LLM_PROVIDER="nope")


def test_vllm_defaults_and_request_shape(load) -> None:
    mod, requests = load(
        T2PW_LLM_PROVIDER="vllm", VLLM_MODEL="served-model",
        # OpenRouter leftovers from a real .env must not leak into a vLLM request.
        OPENROUTER_MODEL="vendor/other", OPENROUTER_EXTRACTION_MAX_TOKENS="64000",
        LLM_MAX_TOKENS="128000",
    )
    assert str(mod._client.base_url).rstrip("/") == "http://127.0.0.1:8000/v1"
    assert mod._client.api_key == "EMPTY"
    assert mod._client.max_retries == 0

    text = mod.chat(
        [{"role": "user", "content": "hi"}], response_json=True,
        model_env_var="OPENROUTER_EXTRACTION_MODEL",
    )
    assert text == "{\"ok\": true}"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://127.0.0.1:8000/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer EMPTY"
    assert "http-referer" not in request.headers and "x-title" not in request.headers
    body: Dict[str, Any] = json.loads(request.content)
    assert set(body) == {"model", "messages", "temperature", "max_tokens", "response_format"}
    assert body["model"] == "served-model"
    assert body["max_tokens"] == 16000
    assert body["response_format"] == {"type": "json_object"}


def test_vllm_env_overrides_and_budget(load) -> None:
    mod, requests = load(
        T2PW_LLM_PROVIDER="vllm", VLLM_MODEL="m", VLLM_BASE_URL="http://gpu:9000/v1",
        VLLM_API_KEY="secret", VLLM_MAX_TOKENS="90000", OPENROUTER_EXTRACTION_MAX_TOKENS="500",
    )
    mod.chat([{"role": "user", "content": "hi"}], stage_name="x")
    assert requests[0].url.host == "gpu" and requests[0].url.port == 9000
    assert requests[0].headers["authorization"] == "Bearer secret"
    body = json.loads(requests[0].content)
    assert body["max_tokens"] == 90000  # not clamped by OpenRouter's per-model table
    assert mod.configured_max_tokens("OPENROUTER_EXTRACTION_MAX_TOKENS") == 90000


def test_vllm_bad_budget_falls_back(load) -> None:
    mod, _ = load(T2PW_LLM_PROVIDER="vllm", VLLM_MODEL="m", VLLM_MAX_TOKENS="zero")
    assert mod.configured_max_tokens() == 16000


def test_vllm_timeout_matches_openrouter(load) -> None:
    mod, _ = load(T2PW_LLM_PROVIDER="vllm", VLLM_MODEL="m", LLM_REQUEST_TIMEOUT_SECONDS="42")
    assert mod._client.timeout == mod._timeout(42.0)


def test_openrouter_budget_unchanged(load) -> None:
    mod, _ = load(
        LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="sk-or-test", OPENROUTER_MODEL="vendor/x",
        OPENROUTER_EXTRACTION_MAX_TOKENS="1234",
    )
    assert mod.PROVIDER == "openrouter"
    assert mod.configured_max_tokens("OPENROUTER_EXTRACTION_MAX_TOKENS") == 1234
    assert mod._client.default_headers["X-Title"]


def test_run_model_override_unset_changes_nothing(load) -> None:
    mod, _ = load(
        LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="sk-or-test", OPENROUTER_MODEL="vendor/global",
        OPENROUTER_EXTRACTION_MODEL="vendor/stage",
    )
    assert mod._resolve_model(model_env_var="OPENROUTER_EXTRACTION_MODEL") == "vendor/stage"
    assert mod._resolve_model() == "vendor/global"
    assert mod._resolve_model(model_override="vendor/explicit") == "vendor/explicit"


def test_run_model_override_wins_on_openrouter(load) -> None:
    mod, requests = load(
        LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="sk-or-test", OPENROUTER_MODEL="vendor/global",
        OPENROUTER_EXTRACTION_MODEL="vendor/stage", T2PW_LLM_MODEL="google/gemma-x",
    )
    assert mod.PROVIDER == "openrouter"
    assert mod._resolve_model() == "google/gemma-x"
    assert mod._resolve_model(model_env_var="OPENROUTER_EXTRACTION_MODEL") == "google/gemma-x"
    assert mod._resolve_model(model_override="vendor/explicit") == "google/gemma-x"
    mod.chat([{"role": "user", "content": "hi"}], model_env_var="OPENROUTER_EXTRACTION_MODEL")
    assert json.loads(requests[0].content)["model"] == "google/gemma-x"


def test_run_model_override_wins_on_local_and_vllm(load) -> None:
    mod, _ = load(LMSTUDIO_MODEL="lm-x", T2PW_LLM_MODEL="run-model")
    assert mod._resolve_model() == "run-model"
    mod, _ = load(T2PW_LLM_PROVIDER="vllm", VLLM_MODEL="served", T2PW_LLM_MODEL="run-model")
    assert mod._resolve_model() == "run-model"
    # With the override set, VLLM_MODEL is no longer required.
    mod, _ = load(T2PW_LLM_PROVIDER="vllm", VLLM_MODEL="", T2PW_LLM_MODEL="run-model")
    assert mod._resolve_model() == "run-model"


def test_run_model_override_survives_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("T2PW_LLM_MODEL=from-dotenv\nLLM_PROVIDER=local\n")
    monkeypatch.setenv("T2PW_LLM_MODEL", "from-process")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")  # so teardown undoes what dotenv writes
    monkeypatch.setattr("t2pw.paths.PROJECT_ROOT", tmp_path)
    spec = importlib.util.spec_from_file_location("t2pw_llm_client_dotenv2", CLIENT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._resolve_model() == "from-process"


def test_ui_defaults_follow_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from t2pw.service import defaults

    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_EXTRACTION_MAX_TOKENS", "64000")
    assert defaults.extract_tokens() == 64000
    monkeypatch.setenv("T2PW_LLM_PROVIDER", "vllm")
    assert defaults.extract_tokens() == 16000 and defaults.infer_tokens() == 16000
    monkeypatch.setenv("VLLM_MAX_TOKENS", "8000")
    assert defaults.extract_tokens() == 8000
