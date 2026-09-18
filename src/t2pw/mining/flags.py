"""Master feature gate for the experimental pathway-mining subsystem.

Default OFF. With ``T2PW_PATHWAY_MINING_ENABLED`` unset, importing this module
is the only thing that happens: nothing here reads a cache, constructs an LLM
client, touches the network, or changes any existing pipeline behaviour.

Three activation paths, in precedence order:

1. an explicit per-invocation override (the CLI's ``--enable-pathway-mining``,
   or the Streamlit toggle) -- ``override=True``;
2. the environment / ``.env`` variable ``T2PW_PATHWAY_MINING_ENABLED``;
3. otherwise **disabled**.

Enabling the feature does not start a run. It only makes the mining entry
points reachable.
"""

from __future__ import annotations

import os
from typing import Optional

#: The single master gate. Nothing in ``t2pw.mining`` runs without it.
FEATURE_ENV_VAR = "T2PW_PATHWAY_MINING_ENABLED"

_TRUE_TOKENS = {"1", "true", "yes", "on", "enabled"}
_FALSE_TOKENS = {"0", "false", "no", "off", "disabled", ""}


class PathwayMiningDisabled(RuntimeError):
    """Raised when a mining entry point is reached with the feature disabled.

    Carries the exact instructions for enabling it so a caller never has to
    guess which of the three activation paths it was missing.
    """


def _as_bool(value: object, default: bool = False) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return default


def mining_enabled(override: Optional[bool] = None) -> bool:
    """Is pathway mining active for this invocation?

    ``override`` is the per-invocation switch. ``True`` enables the feature for
    this call regardless of the environment; ``False`` disables it regardless of
    the environment (so a UI toggle can turn it off again); ``None`` -- the
    default -- defers to ``T2PW_PATHWAY_MINING_ENABLED``.
    """
    if override is not None:
        return bool(override)
    # Load the project .env the same way every other T2PW config helper does, so
    # the flag behaves identically whether it is exported or written to .env.
    try:
        from t2pw.config import ensure_dotenv_loaded

        ensure_dotenv_loaded()
    except Exception:  # noqa: BLE001 - a missing/broken .env must never break the gate
        pass
    return _as_bool(os.getenv(FEATURE_ENV_VAR), False)


def require_mining_enabled(override: Optional[bool] = None) -> None:
    """Raise :class:`PathwayMiningDisabled` unless the feature is on.

    Every mining entry point calls this before doing any work. The error text is
    the activation documentation.
    """
    if mining_enabled(override):
        return
    raise PathwayMiningDisabled(
        "Pathway mining is disabled (experimental feature, default off). Enable it with "
        f"one of: (a) {FEATURE_ENV_VAR}=true in the environment or .env; "
        "(b) the CLI flag --enable-pathway-mining; "
        "(c) the 'Enable pathway mining' control in the Streamlit Pathway Mining page. "
        "Enabling the feature does not start a run."
    )


__all__ = [
    "FEATURE_ENV_VAR",
    "PathwayMiningDisabled",
    "mining_enabled",
    "require_mining_enabled",
]
