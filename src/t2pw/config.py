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
