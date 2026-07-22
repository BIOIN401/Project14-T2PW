"""Central configuration helpers for T2PW.

The most important job of this module is to make the *resolution database*
(the live PathBank/PathWhiz MySQL instance used to canonicalize compound and
species names) reliably configurable from the environment / ``.env`` file, so a
generation run does not silently degrade to non-canonical names because an
environment variable happened not to be loaded in the current import order.

Nothing in here hardcodes a host, user, or password: every value is read from
environment variables (optionally populated from the project ``.env`` file).
See ``docs/setup.md`` -> "Configuring the resolution DB for generation".
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from t2pw.paths import PROJECT_ROOT

# The single source of truth for which env vars configure the resolution DB.
# Keep this in sync with docs/setup.md.
RESOLUTION_DB_ENV = {
    "host": "PATHBANK_DB_HOST",
    "port": "PATHBANK_DB_PORT",
    "user": "PATHBANK_DB_USER",
    "password": "PATHBANK_DB_PASSWORD",
    "schema": "PATHBANK_DB_SCHEMA",
    "connect_timeout": "PATHBANK_DB_CONNECT_TIMEOUT",
    "read_timeout": "PATHBANK_DB_READ_TIMEOUT",
    "write_timeout": "PATHBANK_DB_WRITE_TIMEOUT",
}

_ENV_LOADED = False


def ensure_dotenv_loaded() -> None:
    """Load the project ``.env`` once, regardless of import order.

    ``PathBankDbResolver.from_env`` reads ``os.getenv`` for its connection
    settings. Those variables are only populated as a side effect of importing
    ``t2pw.llm.client`` today, so a generation path that resolves compounds
    without that import reports ``db_not_configured`` and silently falls back to
    offline (often non-canonical) names. Calling this first removes that
    import-order dependency. It never overrides values already present in the
    real environment.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv
    except Exception:  # noqa: BLE001 - dotenv is optional; env may be exported directly
        return
    env_path = PROJECT_ROOT / ".env"
    try:
        # override=False so an explicitly exported environment always wins.
        load_dotenv(dotenv_path=env_path, override=False)
    except Exception:  # noqa: BLE001 - a missing/broken .env must never break a run
        return


def resolution_db_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Return the resolution-DB connection settings from env/``.env``/overrides.

    Values in ``overrides`` (keyed like ``RESOLUTION_DB_ENV``) win over the
    environment. Empty/missing values are simply omitted so callers can apply
    their own defaults.
    """
    ensure_dotenv_loaded()
    over = overrides if isinstance(overrides, dict) else {}
    config: Dict[str, str] = {}
    for key, env_key in RESOLUTION_DB_ENV.items():
        value = over.get(key)
        if value is None or str(value).strip() == "":
            value = os.getenv(env_key, "")
        text = str(value or "").strip()
        if text:
            config[key] = text
    return config


def resolution_db_configured(overrides: Optional[Dict[str, Any]] = None) -> bool:
    """True when the minimum settings (host + user) needed to connect are present."""
    config = resolution_db_config(overrides)
    return bool(config.get("host") and config.get("user"))


# ---------------------------------------------------------------------------
# RAG subsystem configuration (WP0)
# ---------------------------------------------------------------------------
# The RAG subsystem (``t2pw.rag``) is an optional, opt-in evidence layer. Like
# the resolution DB above, every knob is read from the environment / ``.env``
# here rather than via scattered ``os.getenv`` calls, so behavior is stable
# regardless of import order. All values are default-safe: with nothing set,
# ``RAG_ENABLED`` is ``False`` and the core pipeline behaves exactly as today.
# The single source of truth for the RAG env var names; keep in sync with
# docs/rag/02_vector_store.md.
RAG_ENV = {
    "enabled": "RAG_ENABLED",
    "vector_backend": "RAG_VECTOR_BACKEND",
    "index_dir": "RAG_INDEX_DIR",
    "embedding_provider": "RAG_EMBEDDING_PROVIDER",
    "embedding_model": "RAG_EMBEDDING_MODEL",
    "embedding_base_url": "RAG_EMBEDDING_BASE_URL",
    "embedding_api_key": "RAG_EMBEDDING_API_KEY",
    "embedding_dim": "RAG_EMBEDDING_DIM",
    "acquire_max_papers": "RAG_ACQUIRE_MAX_PAPERS",
    "select_max_papers": "RAG_SELECT_MAX_PAPERS",
    "retrieve_top_k": "RAG_RETRIEVE_TOP_K",
    "extract_reactions": "RAG_EXTRACT_REACTIONS",
}

# Default-safe values used when an env var is unset or blank.
RAG_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "vector_backend": "chroma",
    "index_dir": "data/rag_index",
    "embedding_provider": "",  # blank -> falls back to LLM_PROVIDER at read time
    "embedding_model": "",
    # Dedicated embeddings endpoint. Blank -> the embedder reuses the shared LLM
    # client (t2pw.llm.client). Set these when embeddings live on a different
    # host than chat (e.g. OpenRouter serves chat but not embeddings, so point
    # these at LM Studio / OpenAI / another OpenAI-compatible embeddings server).
    "embedding_base_url": "",
    "embedding_api_key": "",
    "embedding_dim": 0,  # 0 == unset; validated on upsert only when > 0
    "acquire_max_papers": 20,
    "select_max_papers": 8,
    "retrieve_top_k": 8,
    # LLM prose→reaction extraction over retrieved passages (t2pw.rag.extract).
    # On by default when RAG runs; the app only wires it when this is true, and
    # every call fails closed, so it can add reactions but never break synthesis.
    "extract_reactions": True,
}

_TRUE_TOKENS = {"1", "true", "yes", "on", "y", "t"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    return text in _TRUE_TOKENS


def _as_int(value: Any, default: int) -> int:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def rag_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the RAG subsystem settings from env / ``.env`` / ``overrides``.

    Every value is default-safe (see ``RAG_DEFAULTS``); a fully unset
    environment yields ``enabled=False`` and today's behavior. ``overrides``
    (keyed like ``RAG_ENV``) win over the environment. ``embedding_provider``
    defaults to ``LLM_PROVIDER`` when left blank, mirroring the vector-store
    spec. Types are normalized: ``enabled`` is a ``bool``; the ``*_max_papers``,
    ``retrieve_top_k`` and ``embedding_dim`` values are ``int``s; everything else
    is a stripped ``str``.
    """
    ensure_dotenv_loaded()
    over = overrides if isinstance(overrides, dict) else {}

    def _raw(key: str) -> Any:
        value = over.get(key)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            value = os.getenv(RAG_ENV[key], "")
        return value

    config: Dict[str, Any] = {
        "enabled": _as_bool(_raw("enabled"), bool(RAG_DEFAULTS["enabled"])),
        "vector_backend": (str(_raw("vector_backend") or "").strip().lower()
                           or str(RAG_DEFAULTS["vector_backend"])),
        "index_dir": (str(_raw("index_dir") or "").strip()
                      or str(RAG_DEFAULTS["index_dir"])),
        "embedding_provider": str(_raw("embedding_provider") or "").strip(),
        "embedding_model": str(_raw("embedding_model") or "").strip(),
        "embedding_base_url": str(_raw("embedding_base_url") or "").strip(),
        "embedding_api_key": str(_raw("embedding_api_key") or "").strip(),
        "embedding_dim": _as_int(_raw("embedding_dim"), int(RAG_DEFAULTS["embedding_dim"])),
        "acquire_max_papers": _as_int(
            _raw("acquire_max_papers"), int(RAG_DEFAULTS["acquire_max_papers"])
        ),
        "select_max_papers": _as_int(
            _raw("select_max_papers"), int(RAG_DEFAULTS["select_max_papers"])
        ),
        "retrieve_top_k": _as_int(_raw("retrieve_top_k"), int(RAG_DEFAULTS["retrieve_top_k"])),
        "extract_reactions": _as_bool(
            _raw("extract_reactions"), bool(RAG_DEFAULTS["extract_reactions"])
        ),
    }

    if not config["embedding_provider"]:
        # Mirror the LLM client default so the embedder targets the same backend.
        config["embedding_provider"] = (os.getenv("LLM_PROVIDER", "local") or "local").strip().lower()

    return config


def rag_enabled(overrides: Optional[Dict[str, Any]] = None) -> bool:
    """True when the RAG subsystem master switch (``RAG_ENABLED``) is on."""
    return bool(rag_config(overrides)["enabled"])
