"""Visible service preflight, run before anything expensive starts.

A mining run costs tokens and wall clock. Discovering halfway through that the
embeddings endpoint is down, or that the output directory is read-only, wastes
both and -- worse -- produces a run whose thin results look like a finding about
the literature rather than an infrastructure failure.

Every check reports ``ok`` / ``warn`` / ``fail`` with a human-readable detail.
A ``fail`` blocks the run. No check prints a secret: an API key is reported as
present or absent, never echoed.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from t2pw.mining.config import MiningConfig, _as_int, _env
from t2pw.mining.flags import FEATURE_ENV_VAR, mining_enabled

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIPPED = "skipped"

#: Output-token budget for the one-word reachability probe. A reasoning model
#: (DeepSeek) can spend a smaller budget thinking and return empty content with
#: ``finish_reason=length``, failing a model that works. Override with
#: ``PATHWAY_MINING_PROBE_MAX_TOKENS``.
PROBE_MAX_TOKENS = 64


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == WARN]

    def add(self, name: str, status: str, detail: str = "", **data: Any) -> CheckResult:
        result = CheckResult(name=name, status=status, detail=detail, data=dict(data))
        self.checks.append(result)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "failures": [c.name for c in self.failures],
            "warnings": [c.name for c in self.warnings],
        }

    def render(self) -> str:
        symbol = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL", SKIPPED: "SKIP"}
        lines = [f"[{symbol.get(c.status, '?')}] {c.name}: {c.detail}" for c in self.checks]
        lines.append("")
        lines.append("PREFLIGHT PASSED" if self.ok else "PREFLIGHT FAILED")
        return "\n".join(lines)


def _check_model(
    report: PreflightReport, config: MiningConfig, *, probe: bool, injected: bool = False
) -> None:
    if injected:
        report.add(
            "model.configuration",
            OK,
            f"the model client was supplied by the caller (fixture, replay or embedding host); "
            f"requested slug {config.model.model!r} is recorded but the provider environment "
            "was not consulted",
            requested_model=config.model.model,
            injected=True,
        )
        report.add("model.reachable", SKIPPED, "caller-supplied client; no probe attempted")
        return
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    key_present = bool((os.getenv("OPENROUTER_API_KEY") or os.getenv("LOCAL_API_KEY") or "").strip())
    report.add(
        "model.configuration",
        OK,
        f"requested model {config.model.model!r} via provider {provider or 'unset'}; "
        f"API key {'present' if key_present else 'ABSENT'}; "
        f"fallback {'configured: ' + config.model.fallback_model if config.model.fallback_model else 'none (a failure will be loud)'}",
        requested_model=config.model.model,
        provider=provider,
        api_key_present=key_present,
    )
    if not provider:
        report.add("model.provider", FAIL, "LLM_PROVIDER is unset; the model client cannot be built")
        return
    if provider == "openrouter" and not key_present:
        report.add("model.credentials", FAIL, "OPENROUTER_API_KEY is not set")
        return
    if not probe:
        report.add("model.reachable", SKIPPED, "probe disabled (no live call attempted)")
        return
    probe_tokens = max(1, _as_int(_env("PATHWAY_MINING_PROBE_MAX_TOKENS"), PROBE_MAX_TOKENS))
    try:
        from t2pw.llm.client import chat_detailed

        result = chat_detailed(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0,
            max_tokens=probe_tokens,
            model_override=config.model.model,
            stage_name="mining:preflight",
            timeout=min(30.0, config.model.request_timeout_seconds),
        )
        text = str(getattr(result, "text", "") or "").strip()
        finish = str(getattr(getattr(result, "diagnostics", None), "finish_reason", "") or "")
        if text:
            report.add("model.reachable", OK, f"model answered ({len(text)} chars)")
        elif finish == "length":
            report.add(
                "model.reachable",
                FAIL,
                f"the {probe_tokens}-token preflight probe budget was exhausted before any "
                "answer text (finish_reason=length); raise PATHWAY_MINING_PROBE_MAX_TOKENS",
            )
        else:
            report.add(
                "model.reachable",
                FAIL,
                "model returned an empty completion on the preflight probe; "
                "starting a run now would attribute an outage to the literature",
            )
    except Exception as exc:  # noqa: BLE001
        report.add("model.reachable", FAIL, f"{type(exc).__name__}: {exc}")


def _check_embeddings(report: PreflightReport, *, probe: bool, required: bool) -> None:
    try:
        from t2pw.config import rag_config
        from t2pw.rag.embed import Embedder
    except Exception as exc:  # noqa: BLE001
        report.add("embeddings.import", FAIL if required else WARN, f"{type(exc).__name__}: {exc}")
        return

    cfg = rag_config()
    endpoint = str(cfg.get("embedding_base_url") or "").strip()
    model = str(cfg.get("embedding_model") or "").strip()
    report.add(
        "embeddings.configuration",
        OK if endpoint else WARN,
        f"endpoint {endpoint or 'UNSET'}; model {model or 'unset'}"
        + ("" if endpoint else " - retrieval will use the deterministic lexical fallback"),
        endpoint_configured=bool(endpoint),
        model=model,
    )
    if not probe:
        report.add("embeddings.reachable", SKIPPED, "probe disabled (no live call attempted)")
        return
    try:
        embedder = Embedder()
        embedder.embed_one("pathway mining preflight probe")
        stats = dict(getattr(embedder, "stats", {}) or {})
        if int(stats.get("api", 0) or 0):
            report.add("embeddings.reachable", OK, "embedding endpoint answered", **stats)
        else:
            status = FAIL if required else WARN
            report.add(
                "embeddings.reachable",
                status,
                "the embedding endpoint did not answer; the deterministic LEXICAL FALLBACK "
                "would be used. This is an infrastructure state, not a result about the "
                "literature. Set PATHWAY_MINING_REQUIRE_EMBEDDINGS=false to proceed anyway.",
                **stats,
            )
    except Exception as exc:  # noqa: BLE001
        report.add(
            "embeddings.reachable", FAIL if required else WARN, f"{type(exc).__name__}: {exc}"
        )


def _check_acquisition(report: PreflightReport, config: MiningConfig) -> None:
    if config.offline:
        report.add(
            "acquisition.sources",
            OK,
            "offline mode: no literature service will be contacted",
            offline=True,
        )
        return
    try:
        from t2pw.rag.acquire import _SOURCE_FETCHERS  # type: ignore

        known = set(_SOURCE_FETCHERS)
    except Exception:  # noqa: BLE001
        known = {"europepmc", "ncbi", "crossref", "semanticscholar", "biorxiv"}
    unknown = [s for s in config.acquisition_sources if s not in known]
    if unknown:
        report.add(
            "acquisition.sources",
            FAIL,
            f"unknown acquisition source(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}",
        )
        return
    ncbi_email = bool((os.getenv("NCBI_EMAIL") or "").strip())
    status = WARN if ("ncbi" in config.acquisition_sources and not ncbi_email) else OK
    report.add(
        "acquisition.sources",
        status,
        f"sources {', '.join(config.acquisition_sources)}"
        + ("" if ncbi_email or "ncbi" not in config.acquisition_sources
           else "; NCBI_EMAIL is unset, NCBI may throttle or refuse"),
        sources=list(config.acquisition_sources),
    )


def _check_outputs(report: PreflightReport, config: MiningConfig) -> None:
    runs_dir = config.resolve_runs_dir()
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        probe = runs_dir / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report.add("outputs.writable", OK, f"{runs_dir} is writable", path=str(runs_dir))
    except Exception as exc:  # noqa: BLE001
        report.add("outputs.writable", FAIL, f"{runs_dir}: {type(exc).__name__}: {exc}")


def _check_inputs(report: PreflightReport, config: MiningConfig) -> None:
    if not str(config.target_species or "").strip():
        report.add("inputs.species", FAIL, "target species is required and is empty")
    else:
        report.add(
            "inputs.species",
            OK,
            f"target species {config.target_species!r} (normalized {config.target_species_normalized!r})",
        )
    kinds = sorted({i.kind for i in config.starting_inputs})
    if not kinds and not config.topic:
        report.add(
            "inputs.starting",
            FAIL,
            "no topic, seed, paper or starting graph supplied; there is nothing to start from",
        )
    else:
        report.add(
            "inputs.starting",
            OK,
            f"topic={config.topic or '(none)'}; starting inputs: {', '.join(kinds) or '(none)'}",
            kinds=kinds,
        )


def run_preflight(
    config: MiningConfig,
    *,
    probe_model: bool = True,
    probe_embeddings: bool = True,
    require_embeddings: Optional[bool] = None,
    enabled_override: Optional[bool] = None,
    model_client_injected: bool = False,
) -> PreflightReport:
    """Check everything a run depends on, before the run starts."""
    report = PreflightReport()

    enabled = mining_enabled(enabled_override)
    report.add(
        "feature.flag",
        OK if enabled else FAIL,
        (
            f"pathway mining ENABLED ({config.enabled_via})"
            if enabled
            else f"pathway mining is disabled; set {FEATURE_ENV_VAR}=true or pass "
            "--enable-pathway-mining"
        ),
        env_var=FEATURE_ENV_VAR,
        enabled=enabled,
    )

    if require_embeddings is None:
        raw = (os.getenv("PATHWAY_MINING_REQUIRE_EMBEDDINGS") or "").strip().lower()
        require_embeddings = raw in {"1", "true", "yes", "on"}

    _check_inputs(report, config)
    _check_outputs(report, config)
    _check_model(
        report,
        config,
        probe=probe_model and not config.offline,
        injected=bool(model_client_injected),
    )
    _check_embeddings(report, probe=probe_embeddings, required=bool(require_embeddings))
    _check_acquisition(report, config)

    b = config.budgets
    report.add(
        "budgets.selected",
        OK,
        (
            f"rounds<={b.max_rounds}, papers processed<={b.max_papers_processed}, "
            f"llm calls<={b.max_llm_calls}, est. cost<=${b.max_cost_usd:g}, "
            f"wall clock<={b.max_wall_clock_seconds:g}s"
        ),
        **b.to_dict(),
    )
    report.add(
        "policy.database_validation",
        OK,
        "external biochemical database validation is DEFERRED in version 1; admission is on "
        "paper evidence alone and no identifier service will be queried",
    )
    report.add(
        "policy.pwml",
        OK,
        "this mode produces a research graph only; the PWML exporter is not invoked and no "
        "PWML file will be written",
    )
    return report


__all__ = ["OK", "WARN", "FAIL", "SKIPPED", "CheckResult", "PreflightReport", "run_preflight"]
