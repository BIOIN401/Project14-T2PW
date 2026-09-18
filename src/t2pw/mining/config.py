"""Configuration and budgets for an experimental pathway-mining run.

Everything here is default-safe and environment-overridable, and every resolved
value is written verbatim into the run manifest -- a run's limits are auditable
after the fact without re-reading anyone's shell.

Precedence for every field: explicit ``overrides`` (CLI flag / Streamlit
control / config file) > ``PATHWAY_MINING_*`` environment variable > the
conservative default below.

No secret is ever stored on a config object. The model client reads its own API
key from the environment; this module records only the model *slug*.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from t2pw.paths import PROJECT_ROOT

#: Bumped whenever the meaning of a persisted artifact field changes.
ARTIFACT_SCHEMA_VERSION = "t2pw.mining/1.0.0"

#: Bumped whenever an extraction prompt changes in a way that could change a claim.
PROMPT_VERSION = "mining-extract/1.0.0"

#: The default extraction and exploration model, per the feature brief.
DEFAULT_MINING_MODEL = "deepseek/deepseek-v4-flash"

#: Where run directories go unless the caller names somewhere else.
DEFAULT_RUNS_DIR = "runs_mining"


def _env(name: str) -> str:
    return str(os.getenv(name, "") or "").strip()


def _as_bool(value: Any, default: bool) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------
@dataclass
class MiningBudgets:
    """Hard limits. The controller stops at the first one reached and says which.

    Defaults are deliberately conservative: this is an exploratory feature and a
    runaway literature crawl is both expensive and useless. Every field is
    overridable and every resolved value lands in ``run_manifest.json``.
    """

    max_rounds: int = 8
    max_papers_discovered: int = 150
    max_papers_processed: int = 40
    max_new_reactions_per_round: int = 25
    max_passages_per_paper: int = 20
    max_primary_reactions: int = 200
    max_candidate_reactions: int = 300
    max_candidate_components: int = 40
    max_llm_calls: int = 200
    max_input_tokens: int = 2_000_000
    max_output_tokens: int = 400_000
    max_cost_usd: float = 5.0
    max_wall_clock_seconds: float = 3600.0
    max_frontier_size: int = 60
    #: Stop after this many consecutive rounds that added nothing useful.
    stop_after_unproductive_rounds: int = 2

    #: Price estimates used to price the run BEFORE any bill exists. These are
    #: configured estimates, never a billed amount -- ``costs.json`` labels them
    #: as such.
    price_input_per_mtok_usd: float = 0.10
    price_output_per_mtok_usd: float = 0.30

    @classmethod
    def from_env(cls, overrides: Optional[Mapping[str, Any]] = None) -> "MiningBudgets":
        over = dict(overrides or {})
        out: Dict[str, Any] = {}
        for f in fields(cls):
            env_name = f"PATHWAY_MINING_{f.name.upper()}"
            raw = over.get(f.name)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                raw = _env(env_name)
            default = f.default
            if isinstance(default, bool):
                out[f.name] = _as_bool(raw, default)
            elif isinstance(default, int):
                out[f.name] = max(0, _as_int(raw, default))
            else:
                out[f.name] = max(0.0, _as_float(raw, float(default)))
        return cls(**out)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class MiningModelConfig:
    """Which model answers, and how hard each call is allowed to try.

    ``model`` is the slug REQUESTED. What actually answered is recorded per call
    in ``model_calls.jsonl``; the two are kept separate because a silent
    provider substitution is exactly the thing that must be visible.
    """

    model: str = DEFAULT_MINING_MODEL
    temperature: float = 0.0
    max_tokens: int = 4000
    #: A narrower retry budget than the shared client's default: an extraction
    #: that needs eight attempts is a prompt problem, not a transient.
    max_attempts: int = 3
    request_timeout_seconds: float = 120.0
    #: An explicit, logged fallback. Empty means "no fallback" -- the run fails
    #: loudly rather than quietly answering with a different model.
    fallback_model: str = ""
    #: Ask for a JSON schema response when the provider supports it.
    use_structured_output: bool = True
    reasoning: str = ""
    prompt_version: str = PROMPT_VERSION

    @classmethod
    def from_env(cls, overrides: Optional[Mapping[str, Any]] = None) -> "MiningModelConfig":
        over = dict(overrides or {})

        def pick(name: str, env_name: str) -> Any:
            raw = over.get(name)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                raw = _env(env_name)
            return raw

        model = str(pick("model", "PATHWAY_MINING_MODEL") or "").strip() or DEFAULT_MINING_MODEL
        return cls(
            model=model,
            temperature=_as_float(pick("temperature", "PATHWAY_MINING_TEMPERATURE"), 0.0),
            max_tokens=_as_int(pick("max_tokens", "PATHWAY_MINING_MAX_TOKENS"), 4000),
            max_attempts=max(1, _as_int(pick("max_attempts", "PATHWAY_MINING_MAX_ATTEMPTS"), 3)),
            request_timeout_seconds=_as_float(
                pick("request_timeout_seconds", "PATHWAY_MINING_REQUEST_TIMEOUT"), 120.0
            ),
            fallback_model=str(pick("fallback_model", "PATHWAY_MINING_FALLBACK_MODEL") or "").strip(),
            use_structured_output=_as_bool(
                pick("use_structured_output", "PATHWAY_MINING_STRUCTURED_OUTPUT"), True
            ),
            reasoning=str(pick("reasoning", "PATHWAY_MINING_REASONING") or "").strip(),
            prompt_version=PROMPT_VERSION,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Starting inputs
# ---------------------------------------------------------------------------
#: Every starting-input kind version 1 accepts. Composable: a run may carry a
#: topic, a species, several seeds, several papers and an existing graph.
STARTING_INPUT_KINDS = (
    "topic",
    "seed_metabolite",
    "seed_enzyme",
    "seed_reaction",
    "paper",
    "pathway_graph",
)


@dataclass
class StartingInput:
    """One user-supplied starting point, kept verbatim alongside its parsing.

    ``raw`` is exactly what the user typed. ``value`` is the parsed/normalized
    form. Both survive into the manifest so a run can be reproduced from what
    was actually asked for, not from what the parser made of it.
    """

    kind: str
    raw: str
    value: str = ""
    #: For ``paper``: ``pmid`` | ``pmcid`` | ``doi`` | ``url`` | ``file``.
    id_type: str = ""
    #: For ``paper`` with ``id_type="file"``: sha256 of the uploaded bytes.
    file_sha256: str = ""
    file_name: str = ""
    #: For ``pathway_graph``: where the read-only starting graph came from.
    source_path: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# The run configuration
# ---------------------------------------------------------------------------
@dataclass
class MiningConfig:
    """Everything one mining run needs, resolved and recordable.

    ``target_species`` is mandatory and has no default. The species guardrail is
    the reason: a run with no declared target cannot decide whether a paper's
    evidence is on-target, and "whatever the papers say" is not a species.
    """

    target_species: str
    #: The normalized internal form. The supplied text is ALWAYS preserved too;
    #: normalization never replaces what the user wrote.
    target_species_normalized: str = ""
    topic: str = ""
    starting_inputs: List[StartingInput] = field(default_factory=list)

    budgets: MiningBudgets = field(default_factory=MiningBudgets)
    model: MiningModelConfig = field(default_factory=MiningModelConfig)

    #: Literature acquisition backends (paper DISCOVERY only -- never biochemical
    #: fact validation; see docs/pathway_mining/evidence_policy.md).
    acquisition_sources: List[str] = field(default_factory=lambda: ["europepmc", "ncbi"])
    #: Passages retrieved per generated query.
    retrieve_top_k: int = 8
    #: Vector backend for the run's own isolated index. ``memory`` keeps a run
    #: from writing into the production RAG index.
    vector_backend: str = "memory"
    #: Offline mode: never reach the network for papers. Fixtures and replays.
    offline: bool = False
    #: Screen acquired papers through ``t2pw.rag.eligibility`` before ingest.
    screen_papers: bool = True

    runs_dir: str = DEFAULT_RUNS_DIR
    run_id: str = ""
    #: Set by the CLI/UI when the feature was enabled per-invocation rather than
    #: by the environment. Recorded in the manifest.
    enabled_via: str = "environment"

    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        supplied = str(self.target_species or "").strip()
        if not supplied:
            raise ValueError(
                "target_species is required for a pathway-mining run: the species guardrail "
                "cannot classify evidence without a declared target."
            )
        self.target_species = supplied
        if not str(self.target_species_normalized or "").strip():
            from t2pw.mining.species import normalize_species

            self.target_species_normalized = normalize_species(supplied)

    # -- resolution ---------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        target_species: str,
        topic: str = "",
        starting_inputs: Optional[List[StartingInput]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "MiningConfig":
        """Resolve a config from explicit arguments + env, recording precedence."""
        over = dict(overrides or {})
        budgets = MiningBudgets.from_env(over.get("budgets"))
        model = MiningModelConfig.from_env(over.get("model"))

        sources_raw = over.get("acquisition_sources")
        if sources_raw is None or sources_raw == "":
            sources_raw = _env("PATHWAY_MINING_ACQUISITION_SOURCES")
        if isinstance(sources_raw, str):
            sources = [s.strip().lower() for s in sources_raw.split(",") if s.strip()]
        else:
            sources = [str(s).strip().lower() for s in (sources_raw or []) if str(s).strip()]
        if not sources:
            sources = ["europepmc", "ncbi"]

        def opt(name: str, env_name: str, default: Any) -> Any:
            raw = over.get(name)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                raw = _env(env_name)
            if isinstance(default, bool):
                return _as_bool(raw, default)
            if isinstance(default, int):
                return _as_int(raw, default)
            if isinstance(default, float):
                return _as_float(raw, default)
            return str(raw or "").strip() or default

        return cls(
            target_species=target_species,
            topic=str(topic or "").strip(),
            starting_inputs=list(starting_inputs or []),
            budgets=budgets,
            model=model,
            acquisition_sources=sources,
            retrieve_top_k=int(opt("retrieve_top_k", "PATHWAY_MINING_RETRIEVE_TOP_K", 8)),
            vector_backend=str(opt("vector_backend", "PATHWAY_MINING_VECTOR_BACKEND", "memory")),
            offline=bool(opt("offline", "PATHWAY_MINING_OFFLINE", False)),
            screen_papers=bool(opt("screen_papers", "PATHWAY_MINING_SCREEN_PAPERS", True)),
            runs_dir=str(opt("runs_dir", "PATHWAY_MINING_RUNS_DIR", DEFAULT_RUNS_DIR)),
            run_id=str(over.get("run_id") or ""),
            enabled_via=str(over.get("enabled_via") or "environment"),
        )

    # -- inputs -------------------------------------------------------------
    def inputs_of(self, kind: str) -> List[StartingInput]:
        return [i for i in self.starting_inputs if i.kind == kind]

    @property
    def seed_entities(self) -> List[StartingInput]:
        return [
            i
            for i in self.starting_inputs
            if i.kind in {"seed_metabolite", "seed_enzyme", "seed_reaction"}
        ]

    def resolve_runs_dir(self) -> Path:
        raw = Path(self.runs_dir)
        return raw if raw.is_absolute() else (PROJECT_ROOT / raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "target_species": self.target_species,
            "target_species_normalized": self.target_species_normalized,
            "topic": self.topic,
            "starting_inputs": [i.to_dict() for i in self.starting_inputs],
            "budgets": self.budgets.to_dict(),
            "model": self.model.to_dict(),
            "acquisition_sources": list(self.acquisition_sources),
            "retrieve_top_k": self.retrieve_top_k,
            "vector_backend": self.vector_backend,
            "offline": self.offline,
            "screen_papers": self.screen_papers,
            "runs_dir": self.runs_dir,
            "run_id": self.run_id,
            "enabled_via": self.enabled_via,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "DEFAULT_MINING_MODEL",
    "DEFAULT_RUNS_DIR",
    "STARTING_INPUT_KINDS",
    "MiningBudgets",
    "MiningModelConfig",
    "MiningConfig",
    "StartingInput",
]
