"""``summary.json`` and ``report.md``.

The report answers the questions a reviewer actually asks of an exploratory run:
what was read, what was believed, what was refused, what disagrees, and what it
cost. Two things it will never do: present a cancelled run as a completed one,
and imply that anything in the graph was checked against an external biochemical
database.

Typing is deliberately loose (``Any`` for the run state) to keep this module
importable from the controller without a cycle.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from t2pw.mining.admission import ADMITTED_STATUSES
from t2pw.mining.artifacts import STATE_COMPLETED
from t2pw.mining.config import MiningConfig
from t2pw.mining.models import (
    STATUS_ACCEPTED,
    STATUS_CANDIDATE,
    STATUS_CONFLICTING,
    STATUS_HYPOTHESIS_ONLY,
    STATUS_INCOMPLETE,
    STATUS_REJECTED,
)
from t2pw.mining.resolver import deferred_note
from t2pw.mining.species import SPECIES_TARGET_SUPPORTED


def build_summary(
    *,
    config: MiningConfig,
    manifest: Any,
    state: Any,
    ledger: Any,
    preflight: Optional[Any] = None,
    index_stats: Optional[Mapping[str, Any]] = None,
    provenance_problems: Sequence[str] = (),
) -> Dict[str, Any]:
    """Every number the report prints, computed once and persisted."""
    graph = state.graph
    store = state.store
    papers = dict(state.papers)

    included = [p for p in papers.values() if not p.exclusion_reasons]
    excluded = [p for p in papers.values() if p.exclusion_reasons]
    screened = [p for p in papers.values() if p.screening_outcome]
    processed = [p for p in papers.values() if p.processed_round >= 0]

    decisions = store.decisions()
    latest: Dict[str, Any] = {}
    for decision in decisions:
        latest[decision.claim_identity] = decision
    status_counts = Counter(d.status for d in latest.values())
    rejection_reasons = Counter(
        d.reason.split(";")[0].strip()[:120]
        for d in latest.values()
        if d.status == STATUS_REJECTED
    )

    primary = graph.primary_reactions
    candidates = graph.candidate_reactions
    multi_paper = [r for r in graph.reactions.values() if r.multi_paper_supported]

    species_counts = Counter(
        r.species_state or "unrecorded" for r in graph.reactions.values()
    )
    evidence_types = Counter(p.evidence_type for p in included)

    additions_by_round = [
        {
            "round": d.round_index,
            "added_reactions": len(d.added_reaction_ids),
            "added_candidates": len(d.added_candidate_reaction_ids),
            "promoted": len(d.promoted_reaction_ids),
            "updated": len(d.updated_reaction_ids),
            "new_conflicts": len(d.new_conflict_ids),
            "new_components": len(d.created_component_ids),
            "useful": d.useful,
            "graph_hash_after": d.graph_hash_after,
        }
        for d in state.deltas
    ]

    call_statuses = Counter(getattr(m, "status", "") or "unknown" for m in state.model_calls)
    failed_calls = [
        {
            "call_id": m.call_id,
            "round": m.round_index,
            "status": m.status,
            "finish_reason": m.finish_reason,
            "parse_error": m.parse_error,
            "schema_error": m.schema_error,
        }
        for m in state.model_calls
        if getattr(m, "status", "ok") != "ok"
    ]

    unresolved = []
    for conflict in store.conflicts.values():
        if not conflict.resolved:
            unresolved.append(
                {
                    "conflict_id": conflict.conflict_id,
                    "attribute": conflict.attribute,
                    "claim_identity": conflict.claim_identity,
                    "paper_ids": list(conflict.paper_ids),
                    "positions": [p.get("value") for p in conflict.positions],
                }
            )

    warnings = list(getattr(ledger, "warnings", []) or [])
    warnings.extend((index_stats or {}).get("warnings", []) or [])
    if preflight is not None:
        warnings.extend(f"preflight: {c.name}: {c.detail}" for c in preflight.warnings)

    return {
        "run_id": manifest.run_id,
        "schema_version": manifest.schema_version,
        "completion_state": manifest.completion_state,
        "completed_normally": manifest.completion_state == STATE_COMPLETED,
        "stop_reason": manifest.stop_reason,
        "stop_detail": manifest.stop_detail,
        "rounds_completed": manifest.rounds_completed,
        "target_species": config.target_species,
        "target_species_normalized": config.target_species_normalized,
        "topic": config.topic,
        "produces_pwml": False,
        "external_database_validation": "deferred_not_evaluated",
        "papers": {
            "discovered": len(papers),
            "screened": len(screened),
            "included": len(included),
            "excluded": len(excluded),
            "processed": len(processed),
            "by_evidence_type": dict(evidence_types),
            "excluded_reasons": dict(
                Counter(r for p in excluded for r in p.exclusion_reasons[:1])
            ),
        },
        "passages": {
            "retrieved": len(state.evidence),
            "extracted": ledger.usage.passages_extracted,
        },
        "claims": {
            "extracted": len(store.claims),
            "distinct_identities": len(store.by_identity),
            "by_latest_status": dict(status_counts),
            "rejected_by_reason": dict(rejection_reasons),
        },
        "graph": {
            "primary_reactions": len(primary),
            "candidate_reactions": len(candidates),
            "candidate_components": len(graph.candidate_components),
            "total_reactions": len(graph.reactions),
            "entities": len(graph.registry),
            "chemical_connections": len(graph.connections),
            "primary_component_id": graph.primary_component_id,
            "primary_selection_rationale": graph.primary_selection_rationale,
            "multi_paper_reactions": len(multi_paper),
            "species_states": dict(species_counts),
            "graph_hash": graph.graph_hash(),
            "promoted_components": [
                {
                    "component_id": c.component_id,
                    "created_round": c.created_round,
                    "promoted_round": c.promoted_round,
                    "bridge_reaction_ids": list(c.bridge_reaction_ids),
                    "bridge_paper_ids": list(c.bridge_paper_ids),
                }
                for c in graph.components.values()
                if c.promoted
            ],
        },
        "conflicts": {
            "total": len(store.conflicts),
            "unresolved": len(unresolved),
            "detail": unresolved,
        },
        "additions_by_round": additions_by_round,
        "usage": ledger.usage.to_dict(),
        "costs": ledger.costs_to_dict(),
        "model_calls": {
            "total": len(state.model_calls),
            "by_status": dict(call_statuses),
            "failed": failed_calls,
        },
        "retrieval": dict(index_stats or {}),
        "preflight": preflight.to_dict() if preflight is not None else None,
        "warnings": warnings,
        "errors": list(state.errors)[:50],
        "provenance_problems": list(provenance_problems),
    }


def _table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> List[str]:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def build_report(summary: Mapping[str, Any], *, state: Any, config: MiningConfig) -> str:
    """The human-readable run report."""
    graph = state.graph
    lines: List[str] = []
    add = lines.append

    add(f"# Pathway mining run `{summary['run_id']}`")
    add("")
    if not summary["completed_normally"]:
        add(
            f"> **This run did not complete normally.** State: "
            f"`{summary['completion_state']}`. Stop reason: `{summary['stop_reason']}`"
            f"{' - ' + summary['stop_detail'] if summary.get('stop_detail') else ''}."
        )
        add("> Every completed round's artifacts are present; nothing below is final.")
        add("")
    add(f"- **Target species:** {summary['target_species']} "
        f"(normalized `{summary['target_species_normalized']}`)")
    add(f"- **Topic:** {summary['topic'] or '(none)'}")
    add(f"- **Rounds completed:** {summary['rounds_completed']}")
    add(f"- **Stop reason:** `{summary['stop_reason'] or 'n/a'}`")
    add("")
    add("> **This run produced a research graph, not PWML.** The PWML exporter was not "
        "invoked, no PathWhiz identifier was required, and no PWML file was written.")
    add("")
    for note in deferred_note():
        add(f"> {note}")
    add("")

    add("## Literature")
    add("")
    p = summary["papers"]
    lines.extend(
        _table(
            [
                ["discovered", p["discovered"]],
                ["screened", p["screened"]],
                ["included", p["included"]],
                ["excluded", p["excluded"]],
                ["fully processed", p["processed"]],
            ],
            ["papers", "count"],
        )
    )
    add("")
    if p["by_evidence_type"]:
        add("Evidence type of included papers: "
            + ", ".join(f"`{k}` x{v}" for k, v in sorted(p["by_evidence_type"].items())))
        add("")
    if p["excluded_reasons"]:
        add("Exclusion reasons:")
        for reason, count in sorted(p["excluded_reasons"].items(), key=lambda kv: -kv[1]):
            add(f"- {count} x {reason}")
        add("")

    add("## Claims and decisions")
    add("")
    c = summary["claims"]
    add(f"- Claims extracted: **{c['extracted']}** across **{c['distinct_identities']}** "
        "distinct reaction identities")
    add(f"- Passages retrieved: **{summary['passages']['retrieved']}**")
    add("")
    if c["by_latest_status"]:
        add("Latest decision per reaction identity:")
        add("")
        for status in (
            STATUS_ACCEPTED, STATUS_CONFLICTING, STATUS_CANDIDATE,
            STATUS_INCOMPLETE, STATUS_HYPOTHESIS_ONLY, STATUS_REJECTED,
        ):
            if status in c["by_latest_status"]:
                add(f"- `{status}`: {c['by_latest_status'][status]}")
        add("")
    if c["rejected_by_reason"]:
        add("Rejections by reason:")
        for reason, count in sorted(c["rejected_by_reason"].items(), key=lambda kv: -kv[1]):
            add(f"- {count} x {reason}")
        add("")

    add("## Research graph")
    add("")
    g = summary["graph"]
    add(f"- Primary graph: **{g['primary_reactions']}** reactions, "
        f"**{g['chemical_connections']}** chemical connections, **{g['entities']}** entities")
    add(f"- Candidate subgraphs: **{g['candidate_components']}** components holding "
        f"**{g['candidate_reactions']}** reactions")
    add(f"- Reactions supported by more than one paper: **{g['multi_paper_reactions']}**")
    if g["primary_selection_rationale"]:
        add(f"- Primary component: `{g['primary_component_id']}` - "
            f"{g['primary_selection_rationale']}")
    add("")
    if g["species_states"]:
        add("Species state of every reaction node:")
        for state_name, count in sorted(g["species_states"].items()):
            marker = " (admitted to the target-species graph)" if state_name == SPECIES_TARGET_SUPPORTED else ""
            add(f"- `{state_name}`: {count}{marker}")
        add("")
    if g["promoted_components"]:
        add("Candidate components promoted after a chemical bridge was found:")
        for row in g["promoted_components"]:
            add(f"- `{row['component_id']}` discovered round {row['created_round']}, "
                f"promoted round {row['promoted_round']} "
                f"(bridge reactions: {', '.join(row['bridge_reaction_ids']) or 'n/a'})")
        add("")

    accepted = [
        r for r in sorted(graph.reactions.values(), key=lambda x: x.reaction_id)
        if r.decision_status in ADMITTED_STATUSES
    ]
    if accepted:
        add("### Admitted reactions")
        add("")
        add("| reaction | status | round | papers | enzyme | compartment |")
        add("|---|---|---|---|---|---|")
        for reaction in accepted[:100]:
            add(
                f"| {reaction.label()} | `{reaction.decision_status}` | "
                f"{reaction.discovery_round} | {reaction.independent_paper_count} | "
                f"{'; '.join(reaction.enzyme_names) or '-'} | {reaction.compartment or '-'} |"
            )
        if len(accepted) > 100:
            add(f"| ... {len(accepted) - 100} more | | | | | |")
        add("")

    add("## Conflicts")
    add("")
    conflicts = summary["conflicts"]
    if not conflicts["total"]:
        add("No conflicting evidence was recorded.")
    else:
        add(f"**{conflicts['unresolved']}** unresolved of **{conflicts['total']}** recorded. "
            "Both positions are retained; nothing was resolved by choosing a side.")
        add("")
        for row in conflicts["detail"][:30]:
            add(f"- `{row['attribute']}` on `{row['claim_identity']}`: "
                f"{', '.join(str(v) for v in row['positions'])} "
                f"(papers: {', '.join(row['paper_ids'])})")
    add("")

    add("## Growth by round")
    add("")
    add("| round | added | candidates | promoted | updated | conflicts | useful |")
    add("|---|---|---|---|---|---|---|")
    for row in summary["additions_by_round"]:
        add(
            f"| {row['round']} | {row['added_reactions']} | {row['added_candidates']} | "
            f"{row['promoted']} | {row['updated']} | {row['new_conflicts']} | "
            f"{'yes' if row['useful'] else 'no'} |"
        )
    add("")

    add("## Cost and usage")
    add("")
    costs = summary["costs"]
    usage = summary["usage"]
    add(f"- Model calls: **{costs['llm_calls']}**")
    add(f"- Input tokens: **{costs['input_tokens']:,}**; output tokens: "
        f"**{costs['output_tokens']:,}**; cached: **{costs['cached_tokens']:,}**")
    add(f"- Estimated cost: **${costs['estimated_cost_usd']:.6f}** "
        f"(limit ${costs['max_cost_usd']:g})")
    add(f"- Elapsed: **{usage['elapsed_seconds']:.1f}s**")
    add("")
    add(f"> {costs['note']}")
    add("")

    mc = summary["model_calls"]
    if mc["failed"]:
        add("### Model-call failures")
        add("")
        add("These calls produced no admissible claim. None of them added a reaction.")
        add("")
        for row in mc["failed"][:30]:
            add(f"- `{row['call_id']}` round {row['round']}: `{row['status']}` "
                f"(finish_reason `{row['finish_reason'] or 'n/a'}`) "
                f"{row['parse_error'] or row['schema_error'] or ''}".rstrip())
        add("")

    retrieval = summary.get("retrieval") or {}
    if retrieval:
        add("## Retrieval")
        add("")
        add(f"- Papers indexed: {retrieval.get('papers_indexed', 0)}; "
            f"chunks: {retrieval.get('chunks', 0)}")
        add(f"- Embedding mode: `{retrieval.get('embedding_mode', 'unknown')}`")
        add("")

    if summary["warnings"] or summary["errors"] or summary["provenance_problems"]:
        add("## Warnings, partial failures and unresolved questions")
        add("")
        for warning in summary["warnings"][:40]:
            add(f"- WARNING: {warning}")
        for problem in summary["provenance_problems"][:40]:
            add(f"- PROVENANCE: {problem}")
        for error in summary["errors"][:40]:
            add(f"- ERROR: {error.splitlines()[0][:300]}")
        add("")

    open_frontier = [i for i in state.frontier if not i.exhausted]
    if open_frontier:
        add("## Frontier left open at the stop")
        add("")
        add("These questions were generated but never asked. They are where a longer run "
            "would go next.")
        add("")
        for item in open_frontier[:30]:
            add(f"- `{item.kind}` -> {item.target} ({item.reason})")
        add("")

    return "\n".join(str(line) for line in lines) + "\n"


__all__ = ["build_summary", "build_report"]
