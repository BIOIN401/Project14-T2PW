"""The run settings, once: the Streamlit form's widget defaults and ``run_pathway``'s (R5).

Benchmark behaviour depends on the form's defaults, and a headless run has no form to
read them from (R3 series audit F2). So both read these, and the two cannot drift; the
environment-backed values are functions because the form evaluates them on every run.
``tests/test_service_run_pathway.py`` pins the form's widgets to these names.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# The "Run pipeline" form.
RUN_INFERENCE = True
ENABLE_CHUNKING = True
RAG_INCOMPLETE_FLAG = True  # the toggle's value when RAG_ENABLED is on; False otherwise
EXTRACT_ATTEMPTS = 2
INFER_ATTEMPTS = 2
INFER_ROUNDS = 2
TEMPERATURE = 0.0
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1600

# The "Run audit and DB mapping" panel.
USE_LLM_AUDIT = True
USE_SBML_OVERWATCH = False
USE_STOICH_AGENT = False
DEFAULT_COMPARTMENT = "cell"
MAPPING_CACHE_PATH = "data/id_mapping_cache.json"
AUDIT_MAX_ROUNDS = 4
AUDIT_TIMEOUT_SECONDS = 240
AUDIT_CANDIDATE_COUNT = 3
USE_EXAMPLE_RETRIEVAL = False
EXAMPLE_INDEX_PATH = "src/tmp/sbml_motif_index.json"
EXAMPLE_TOP_K = 3
USE_GAP_RESOLVER = True
USE_LLM_GAP_RESOLVER = True
GAP_RESOLVER_MAX_ITEMS = 80

ID_SOURCES = ("hybrid", "db", "api")


def bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read an integer UI default from the environment and keep it in range."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d.", name, raw, default)
        return default
    if value < minimum or value > maximum:
        clamped = min(max(value, minimum), maximum)
        logger.warning(
            "Clamping %s=%d to the supported UI range %d..%d.",
            name,
            value,
            minimum,
            maximum,
        )
        return clamped
    return value


def extract_tokens() -> int:
    return bounded_env_int("OPENROUTER_EXTRACTION_MAX_TOKENS", 64000, minimum=500, maximum=128000)


def infer_tokens() -> int:
    return bounded_env_int("OPENROUTER_INFERENCE_MAX_TOKENS", 64000, minimum=500, maximum=128000)


def id_source() -> str:
    """``PATHBANK_ID_SOURCE`` when it names a source, else the first one, as the form's selectbox."""
    value = (os.getenv("PATHBANK_ID_SOURCE", ID_SOURCES[0]) or ID_SOURCES[0]).strip().lower()
    return value if value in ID_SOURCES else ID_SOURCES[0]


def db_host() -> str:
    return os.getenv("PATHBANK_DB_HOST", "")


def db_port() -> int:
    return int(os.getenv("PATHBANK_DB_PORT", "3306") or "3306")


def db_user() -> str:
    return os.getenv("PATHBANK_DB_USER", "")


def db_schema() -> str:
    return os.getenv("PATHBANK_DB_SCHEMA", "pathbank")


def db_password() -> str:
    return os.getenv("PATHBANK_DB_PASSWORD", "")
