"""The run artifact directory: everything a mining run leaves behind.

One self-contained directory per run::

    runs_mining/<run_id>/
        run_manifest.json        config.json           preflight.json
        papers.jsonl             queries.jsonl         frontier.jsonl
        evidence_spans.jsonl     claims.jsonl          decisions.jsonl
        model_calls.jsonl        costs.json            conflicts.json
        primary_graph.json       candidate_subgraphs.json
        summary.json             report.md             checkpoint.json
        graph_views/reaction_network.json
        graph_views/provenance_network.json
        graph_views/*.dot
        rounds/round_001/...     rounds/round_002/...

Two rules shape this module:

* **Flush per round.** A run that is cancelled or killed keeps every completed
  round's artifacts, and its manifest says ``cancelled`` -- never ``completed``.
* **Rewrite, don't append.** Each snapshot rewrites its file from the in-memory
  state, so a resumed run cannot double-write a row, and a half-written append is
  never mistaken for data.

Nothing here writes a secret. The manifest records model *slugs* and whether a
key was present, never a key.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from t2pw.mining.config import (
    ARTIFACT_SCHEMA_VERSION,
    MiningConfig,
    PROMPT_VERSION,
)
from t2pw.mining.flags import FEATURE_ENV_VAR
from t2pw.mining.ids import run_id as make_run_id

STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_STOPPED = "stopped_at_limit"
STATE_CANCELLED = "cancelled"
STATE_FAILED = "failed"
STATE_PREFLIGHT_FAILED = "preflight_failed"

#: Completion states that are NOT a normal completion. A cancelled or
#: interrupted run must never look like one that finished.
INCOMPLETE_STATES = frozenset({STATE_RUNNING, STATE_CANCELLED, STATE_FAILED, STATE_PREFLIGHT_FAILED})


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - a missing git must never break a run
        return ""


def git_state(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Commit, branch and dirty state, best effort."""
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    status = _git("status", "--porcelain", cwd=cwd)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_file_count": len([line for line in status.splitlines() if line.strip()]),
    }


def environment_state() -> Dict[str, Any]:
    # importlib.metadata rather than module.__version__: reading __version__ is
    # deprecated on some of these and must not emit a warning during a run.
    from importlib import metadata

    versions: Dict[str, str] = {}
    for package in ("streamlit", "openai", "httpx", "jsonschema", "pandas", "lxml"):
        try:
            versions[package] = metadata.version(package)
        except Exception:  # noqa: BLE001
            versions[package] = "absent"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Iterable[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return path


def read_jsonl(path: Path) -> List[Any]:
    if not path.exists():
        return []
    out: List[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


@dataclass
class RunManifest:
    """The run's identity card. Written at start, updated at every flush."""

    run_id: str
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    prompt_version: str = PROMPT_VERSION
    started_at: str = field(default_factory=utcnow)
    updated_at: str = ""
    finished_at: str = ""
    completion_state: str = STATE_RUNNING
    stop_reason: str = ""
    stop_detail: str = ""
    git: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    rounds_completed: int = 0
    resumed_from_round: int = -1
    notes: List[str] = field(default_factory=list)

    @property
    def completed_normally(self) -> bool:
        return self.completion_state == STATE_COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "started_at": self.started_at,
            "updated_at": self.updated_at or utcnow(),
            "finished_at": self.finished_at,
            "completion_state": self.completion_state,
            "completed_normally": self.completed_normally,
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "git": dict(self.git),
            "environment": dict(self.environment),
            "feature_flags": dict(self.feature_flags),
            "config": dict(self.config),
            "rounds_completed": self.rounds_completed,
            "resumed_from_round": self.resumed_from_round,
            "notes": list(self.notes),
            "produces_pwml": False,
            "external_database_validation": "deferred_not_evaluated",
        }
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        payload = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


class RunDirectory:
    """Owns one run's directory and every write into it."""

    def __init__(self, root: Path, *, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rounds").mkdir(exist_ok=True)
        (self.root / "graph_views").mkdir(exist_ok=True)

    # -- construction -------------------------------------------------------
    @classmethod
    def create(cls, config: MiningConfig, *, run_id: str = "") -> "RunDirectory":
        rid = run_id or config.run_id or make_run_id(
            datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S"),
            config.target_species,
            config.topic,
        )
        return cls(config.resolve_runs_dir() / rid, run_id=rid)

    @classmethod
    def open(cls, path: Path) -> "RunDirectory":
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"no run directory at {root}")
        return cls(root, run_id=root.name)

    # -- paths --------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def round_dir(self, index: int) -> Path:
        directory = self.root / "rounds" / f"round_{int(index):03d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # -- writes -------------------------------------------------------------
    def write_manifest(self, manifest: RunManifest) -> Path:
        manifest.updated_at = utcnow()
        return write_json(self.path("run_manifest.json"), manifest.to_dict())

    def write_config(self, config: MiningConfig) -> Path:
        return write_json(self.path("config.json"), config.to_dict())

    def write_preflight(self, report: Any) -> Path:
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report or {})
        return write_json(self.path("preflight.json"), payload)

    def snapshot(
        self,
        *,
        papers: Sequence[Any],
        queries: Sequence[Any],
        frontier: Sequence[Any],
        evidence: Sequence[Any],
        claims: Sequence[Any],
        decisions: Sequence[Any],
        model_calls: Sequence[Any],
        conflicts: Sequence[Any],
        costs: Mapping[str, Any],
        primary_graph: Mapping[str, Any],
        candidate_subgraphs: Mapping[str, Any],
    ) -> None:
        """Rewrite every top-level artifact from current state."""
        def rows(items: Sequence[Any]) -> List[Any]:
            return [i.to_dict() if hasattr(i, "to_dict") else i for i in items]

        write_jsonl(self.path("papers.jsonl"), rows(papers))
        write_jsonl(self.path("queries.jsonl"), rows(queries))
        write_jsonl(self.path("frontier.jsonl"), rows(frontier))
        write_jsonl(self.path("evidence_spans.jsonl"), rows(evidence))
        write_jsonl(self.path("claims.jsonl"), rows(claims))
        write_jsonl(self.path("decisions.jsonl"), rows(decisions))
        write_jsonl(self.path("model_calls.jsonl"), rows(model_calls))
        write_json(self.path("conflicts.json"), {"conflicts": rows(conflicts)})
        write_json(self.path("costs.json"), dict(costs))
        write_json(self.path("primary_graph.json"), dict(primary_graph))
        write_json(self.path("candidate_subgraphs.json"), dict(candidate_subgraphs))

    def write_round(self, index: int, payload: Mapping[str, Any]) -> Path:
        directory = self.round_dir(index)
        for name in (
            "input_frontier",
            "queries",
            "papers_considered",
            "passages_retrieved",
            "claims_extracted",
            "decisions",
            "rejections",
            "graph_delta",
            "new_frontier",
            "metrics",
            "errors",
        ):
            if name in payload:
                value = payload[name]
                if isinstance(value, list):
                    write_jsonl(directory / f"{name}.jsonl", value)
                else:
                    write_json(directory / f"{name}.json", value)
        write_json(directory / "round.json", dict(payload))
        return directory

    def write_checkpoint(self, checkpoint: Any, *, round_index: int) -> Path:
        payload = checkpoint.to_dict() if hasattr(checkpoint, "to_dict") else dict(checkpoint)
        write_json(self.round_dir(round_index) / "checkpoint.json", payload)
        return write_json(self.path("checkpoint.json"), payload)

    def write_state(self, state: Mapping[str, Any]) -> Path:
        """The resumable state blob: graph, claims, papers, evidence, usage."""
        return write_json(self.path("state.json"), dict(state))

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        return write_json(self.path("summary.json"), dict(summary))

    def write_report(self, text: str) -> Path:
        path = self.path("report.md")
        path.write_text(str(text or ""), encoding="utf-8", newline="\n")
        return path

    def write_graph_view(self, name: str, payload: Any) -> Path:
        if isinstance(payload, str):
            path = self.path("graph_views", name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8", newline="\n")
            return path
        return write_json(self.path("graph_views", name), payload)

    # -- reads --------------------------------------------------------------
    def read_manifest(self) -> Optional[RunManifest]:
        data = read_json(self.path("run_manifest.json"))
        return RunManifest.from_dict(data) if isinstance(data, Mapping) else None

    def read_checkpoint(self) -> Optional[Dict[str, Any]]:
        data = read_json(self.path("checkpoint.json"))
        return dict(data) if isinstance(data, Mapping) else None

    def read_state(self) -> Optional[Dict[str, Any]]:
        data = read_json(self.path("state.json"))
        return dict(data) if isinstance(data, Mapping) else None

    def read_config(self) -> Optional[Dict[str, Any]]:
        data = read_json(self.path("config.json"))
        return dict(data) if isinstance(data, Mapping) else None

    def completed_rounds(self) -> List[int]:
        out: List[int] = []
        rounds = self.path("rounds")
        if not rounds.exists():
            return out
        for child in sorted(rounds.iterdir()):
            if child.is_dir() and (child / "checkpoint.json").exists():
                try:
                    out.append(int(child.name.split("_")[-1]))
                except ValueError:
                    continue
        return out


def build_manifest(config: MiningConfig, *, run_id: str, enabled_via: str) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        git=git_state(),
        environment=environment_state(),
        feature_flags={
            FEATURE_ENV_VAR: os.getenv(FEATURE_ENV_VAR, ""),
            "enabled_via": enabled_via,
            "offline": config.offline,
        },
        config=config.to_dict(),
    )


def list_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Every run directory under ``runs_dir``, newest first."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        manifest = read_json(child / "run_manifest.json")
        if not isinstance(manifest, Mapping):
            continue
        rows.append(
            {
                "run_id": child.name,
                "path": str(child),
                "started_at": manifest.get("started_at", ""),
                "completion_state": manifest.get("completion_state", ""),
                "stop_reason": manifest.get("stop_reason", ""),
                "rounds_completed": manifest.get("rounds_completed", 0),
                "species": (manifest.get("config") or {}).get("target_species", ""),
                "topic": (manifest.get("config") or {}).get("topic", ""),
            }
        )
    return rows


__all__ = [
    "STATE_RUNNING",
    "STATE_COMPLETED",
    "STATE_STOPPED",
    "STATE_CANCELLED",
    "STATE_FAILED",
    "STATE_PREFLIGHT_FAILED",
    "INCOMPLETE_STATES",
    "RunManifest",
    "RunDirectory",
    "build_manifest",
    "git_state",
    "environment_state",
    "list_runs",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
    "utcnow",
]
