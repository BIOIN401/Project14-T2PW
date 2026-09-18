"""Rebuild ``report.md``, ``summary.json`` and the graph views from artifacts.

Used by ``python -m t2pw.mining report <run>`` and by the Streamlit page when a
user opens a run that finished in another session. It reads only what is on
disk -- no model call, no network, no re-run -- so a report can always be
regenerated from a run directory alone, including one that was cancelled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from t2pw.mining import viz
from t2pw.mining.artifacts import RunDirectory, RunManifest
from t2pw.mining.budget import BudgetLedger, Usage
from t2pw.mining.config import MiningBudgets, MiningConfig, MiningModelConfig, StartingInput
from t2pw.mining.controller import MiningState
from t2pw.mining.report import build_report, build_summary


class _ReplayLedger(BudgetLedger):
    """A ledger reconstructed from a finished run's counters, for reporting only."""

    def __init__(self, budgets: MiningBudgets, usage: Dict[str, Any], stop_reason: str,
                 stop_detail: str, warnings: List[str]) -> None:
        super().__init__(budgets)
        self.usage = Usage(**{
            k: v for k, v in dict(usage or {}).items()
            if k in Usage.__dataclass_fields__  # type: ignore[attr-defined]
        })
        self.stop_reason = stop_reason
        self.stop_detail = stop_detail
        self.warnings = list(warnings or [])

    def refresh(self) -> None:  # the elapsed time is historical, not live
        return


def config_from_payload(payload: Dict[str, Any]) -> MiningConfig:
    """Rebuild a :class:`MiningConfig` from a run's ``config.json``."""
    data = dict(payload or {})
    return MiningConfig(
        target_species=str(data.get("target_species") or "unknown"),
        target_species_normalized=str(data.get("target_species_normalized") or ""),
        topic=str(data.get("topic") or ""),
        starting_inputs=[
            StartingInput(**{k: v for k, v in dict(i).items()
                             if k in StartingInput.__dataclass_fields__})  # type: ignore[attr-defined]
            for i in data.get("starting_inputs") or []
        ],
        budgets=MiningBudgets(**{
            k: v for k, v in (data.get("budgets") or {}).items()
            if k in MiningBudgets.__dataclass_fields__  # type: ignore[attr-defined]
        }),
        model=MiningModelConfig(**{
            k: v for k, v in (data.get("model") or {}).items()
            if k in MiningModelConfig.__dataclass_fields__  # type: ignore[attr-defined]
        }),
        acquisition_sources=list(data.get("acquisition_sources") or []),
        retrieve_top_k=int(data.get("retrieve_top_k") or 8),
        vector_backend=str(data.get("vector_backend") or "memory"),
        offline=bool(data.get("offline")),
        screen_papers=bool(data.get("screen_papers", True)),
        runs_dir=str(data.get("runs_dir") or "runs_mining"),
        run_id=str(data.get("run_id") or ""),
        enabled_via=str(data.get("enabled_via") or "environment"),
    )


def load_run(run_dir: RunDirectory) -> Dict[str, Any]:
    """Load a run's config, manifest, state and checkpoint from disk."""
    manifest = run_dir.read_manifest()
    if manifest is None:
        raise FileNotFoundError(f"no run_manifest.json under {run_dir.root}")
    state_payload = run_dir.read_state() or {}
    config = config_from_payload(run_dir.read_config() or manifest.config)
    checkpoint = run_dir.read_checkpoint() or {}
    state = MiningState.from_dict(state_payload)
    return {
        "manifest": manifest,
        "config": config,
        "state": state,
        "checkpoint": checkpoint,
    }


def regenerate(run_dir: RunDirectory) -> List[Path]:
    """Rewrite summary, report and both graph views. Returns the paths written."""
    loaded = load_run(run_dir)
    manifest: RunManifest = loaded["manifest"]
    config: MiningConfig = loaded["config"]
    state: MiningState = loaded["state"]
    checkpoint: Dict[str, Any] = loaded["checkpoint"]

    ledger = _ReplayLedger(
        config.budgets,
        checkpoint.get("usage") or {},
        manifest.stop_reason,
        manifest.stop_detail,
        [],
    )

    conflicts = dict(state.store.conflicts)
    reaction_view = viz.reaction_network(
        state.graph, papers=state.papers, conflicts=conflicts
    )
    provenance_view = viz.provenance_network(
        graph=state.graph,
        papers=state.papers,
        claims=state.store.claims,
        evidence=state.evidence,
        conflicts=conflicts,
    )
    table = viz.evidence_table(
        graph=state.graph,
        papers=state.papers,
        claims=state.store.claims,
        evidence=state.evidence,
        decisions=state.store.decisions(),
    )

    summary = build_summary(
        config=config,
        manifest=manifest,
        state=state,
        ledger=ledger,
        preflight=None,
        index_stats={},
        provenance_problems=[],
    )

    written = [
        run_dir.write_summary(summary),
        run_dir.write_report(build_report(summary, state=state, config=config)),
        run_dir.write_graph_view("reaction_network.json", reaction_view),
        run_dir.write_graph_view("provenance_network.json", provenance_view),
        run_dir.write_graph_view("reaction_network.dot", viz.reaction_network_dot(reaction_view)),
        run_dir.write_graph_view(
            "provenance_network.dot", viz.provenance_network_dot(provenance_view)
        ),
        run_dir.write_graph_view("evidence_table.json", table),
    ]
    return written


def load_views(run_dir: RunDirectory) -> Dict[str, Any]:
    """Read the two views and the evidence table, regenerating them if absent."""
    from t2pw.mining.artifacts import read_json

    reaction = read_json(run_dir.path("graph_views", "reaction_network.json"))
    provenance = read_json(run_dir.path("graph_views", "provenance_network.json"))
    table = read_json(run_dir.path("graph_views", "evidence_table.json"))
    if reaction is None or provenance is None:
        regenerate(run_dir)
        reaction = read_json(run_dir.path("graph_views", "reaction_network.json"))
        provenance = read_json(run_dir.path("graph_views", "provenance_network.json"))
        table = read_json(run_dir.path("graph_views", "evidence_table.json"))
    return {
        "reaction_network": reaction or {},
        "provenance_network": provenance or {},
        "evidence_table": table or [],
        "summary": read_json(run_dir.path("summary.json")) or {},
        "manifest": read_json(run_dir.path("run_manifest.json")) or {},
        "costs": read_json(run_dir.path("costs.json")) or {},
        "conflicts": read_json(run_dir.path("conflicts.json")) or {},
        "candidate_subgraphs": read_json(run_dir.path("candidate_subgraphs.json")) or {},
    }


__all__ = ["regenerate", "load_run", "load_views", "config_from_payload"]
