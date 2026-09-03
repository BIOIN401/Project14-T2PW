"""D-093 section 5.6 -- ingest lineage and evaluation records into self-hosted Phoenix.

EVALUATION-ONLY OpenTelemetry / OpenInference, exactly as D-093 words it. This module
reads archived artifacts and emits spans. It imports no production module, changes no
pipeline semantics, and produces no acceptance verdict.

**PHOENIX IS A TRACE STORE AND DASHBOARD. IT IS NOT BIOLOGICAL GROUND TRUTH.** Nothing
here may be cited as evidence that a reaction is correct: a span is a record of what the
pipeline did, and its ``support_class`` attribute is the R-D092-1 classifier's output,
carried across unchanged. No pipeline semantics were altered to produce prettier traces,
and if a trace looks wrong the answer is to fix the instrument, never the pipeline.

=============================================================================
WHY THIS RUNS IN AN ISOLATED INTERPRETER
=============================================================================

``arize-phoenix`` pulls a large dependency tree (SQLAlchemy, strawberry-graphql, pandas
and more). Every merge gate in this sprint -- SMOKE, the gold-readers split, Chunk D --
runs in the project ``.venv``, so installing an evaluation dashboard into it would put
a 508-test gate at the mercy of a transitive pin. **The eval venv is separate and the
project venv is never touched.** This module therefore runs under the eval interpreter
and is deliberately NOT importable from the project venv; the acceptance test for it
asserts only what can be checked without Phoenix installed.

=============================================================================
TRACE SHAPE, AND WHY IT IS SHAPED THAT WAY
=============================================================================

One TRACE per archived leg, so a reader opens a leg and sees its whole story:

  leg span (CHAIN)              run, population, payload file, target paper, and the
                                leg's two-table metrics as attributes
    reaction span (CHAIN)       one per canonical/fallback reaction: support class,
                                attribution tier, origin stages, resolved source ids,
                                whether audit modified it, whether it survives
      retrieval span (RETRIEVER)  present only when the chunk join recovered the
                                retrieved record. Carries the actual span text, the
                                retrieval score and the admission verdict as
                                OpenInference ``retrieval.documents`` -- the shape
                                Phoenix's RAG views are built to read.

**THE THREE F-176 FIELDS ARE CARRIED SEPARATELY AND NEVER COLLAPSED**, because D-093
section 2 DENIED making the runtime gate applicable and required the evaluation layer
to report it instead:

  ``runtime_gate_applicable``  what the frozen pipeline actually did -- always False
                               here, because the runtime change was denied and
                               production behaviour is unchanged.
  ``offline_evaluable``        whether the persisted artifacts let us evaluate it.
  ``offline_verdict``          what the offline evaluation concluded.

An ingestion that merged any two of these would re-create the exact confusion D-092
refused.

Usage (eval venv interpreter, NOT the project venv):
  python rd093_phoenix_ingest.py <repo-root> --lineage-json <records.json> \
      [--metrics-json <two_table.json>] [--limit N] [--no-server]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

#: Kept in sync with the R-D092-1 classifier's vocabulary by NAME, not by import:
#: this module runs under a different interpreter, so importing the classifier would
#: couple the eval venv to the project tree. The values are asserted against the
#: records file at load time by :func:`check_vocabulary`, so drift is caught loudly
#: rather than silently producing spans with unknown classes.
SUPPORT_CLASSES = ("target_paper_supported", "external_rag_supported",
                   "unsupported", "indeterminate")

#: D-093 section 2's three fields. Never collapsed into one.
RUNTIME_GATE_APPLICABLE = "eval.no_rejected_rag_reaction_reintroduced.runtime_gate_applicable"
OFFLINE_EVALUABLE = "eval.no_rejected_rag_reaction_reintroduced.offline_evaluable"
OFFLINE_VERDICT = "eval.no_rejected_rag_reaction_reintroduced.offline_verdict"


def check_vocabulary(records: Iterable[Dict[str, Any]]) -> List[str]:
    """Support classes in the records that this module does not know about."""

    seen = {r.get("support_class") for r in records}
    return sorted(c for c in seen if c and c not in SUPPORT_CLASSES)


def reintroduction_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The three F-176 fields for one reaction, kept apart.

    ``runtime_gate_applicable`` is hard-coded False and that is the point: D-093
    section 2 DENIED making the gate runtime-applicable, so production behaviour is
    unchanged and no evaluation may imply otherwise. The offline pair is what this
    layer contributes.

    A row is ``offline_evaluable`` only when the chunk join actually recovered an
    admission record for it -- otherwise there is nothing to evaluate and the verdict
    is ``unavailable``, never a pass. The verdict is ``rejected_candidate_reintroduced``
    when a row that survives into the payload traces back to a REJECTED
    reaction-specific admission record.
    """

    joined = rec.get("admission_result") not in (None, "unavailable")
    if not joined:
        return {RUNTIME_GATE_APPLICABLE: False, OFFLINE_EVALUABLE: False,
                OFFLINE_VERDICT: "unavailable"}
    reintroduced = (rec.get("admission_result") == "rejected"
                    and rec.get("chunk_join_reaction_specific") is True
                    and rec.get("survives_in_payload") is True)
    return {
        RUNTIME_GATE_APPLICABLE: False,
        OFFLINE_EVALUABLE: True,
        OFFLINE_VERDICT: ("rejected_candidate_reintroduced" if reintroduced
                          else "no_rejected_candidate_reintroduced"),
        # The scope is carried so a reader never mistakes a different run's verdict
        # for this run's: identical legs give materially different draws.
        "eval.chunk_join_scope": rec.get("chunk_join_scope", "unavailable"),
    }


def _scalar(v: Any) -> Any:
    """OTel attributes take scalars or homogeneous sequences; JSON-encode the rest."""

    if v is None:
        return "unavailable"
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, str) for x in v):
            return list(v)
        return json.dumps(v, ensure_ascii=False)[:4000]
    return json.dumps(v, ensure_ascii=False)[:4000]


def reaction_attributes(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Everything D-093 section 3 asks a lineage record to carry, as span attributes."""

    attrs: Dict[str, Any] = {
        "openinference.span.kind": "CHAIN",
        "t2pw.run": _scalar(rec.get("run")),
        "t2pw.leg_dir": _scalar(rec.get("leg_dir")),
        "t2pw.population": _scalar(rec.get("population")),
        "t2pw.payload_file": _scalar(rec.get("payload_file")),
        "t2pw.target_paper": _scalar(rec.get("target_paper")),
        "t2pw.reaction.row_index": _scalar(rec.get("row_index")),
        "t2pw.reaction.name": _scalar(rec.get("reaction_name")),
        "t2pw.reaction.inputs": _scalar(rec.get("inputs")),
        "t2pw.reaction.outputs": _scalar(rec.get("outputs")),
        "t2pw.reaction.enzymes": _scalar(rec.get("enzymes")),
        # attribution
        "t2pw.lineage.attribution_tier": _scalar(rec.get("attribution_tier")),
        "t2pw.lineage.origin_stages": _scalar(rec.get("origin_stages")),
        "t2pw.lineage.origins": _scalar(rec.get("origins")),
        "t2pw.lineage.source_ids": _scalar(rec.get("source_ids")),
        "t2pw.lineage.source_resolutions": _scalar(rec.get("source_resolutions")),
        "t2pw.lineage.retrieved_chunk_ids": _scalar(rec.get("retrieved_chunk_ids")),
        "t2pw.lineage.review_required": _scalar(rec.get("lineage_review_required")),
        # target-paper evidence: presence only, and labelled so, because the string
        # runs to 100k characters and silently carries EXTERNAL text.
        "t2pw.evidence.present_only": _scalar(rec.get("row_evidence_present")),
        "t2pw.evidence.chars": _scalar(rec.get("row_evidence_chars")),
        # admission / audit / survival
        "t2pw.admission.result": _scalar(rec.get("admission_result")),
        "t2pw.admission.rejection_reasons": _scalar(rec.get("rejection_reasons")),
        "t2pw.admission.requested_pathway_match": _scalar(rec.get("requested_pathway_match")),
        "t2pw.admission.organism_match": _scalar(rec.get("organism_match")),
        "t2pw.audit_modified": _scalar(rec.get("audit_modified")),
        "t2pw.survives_in_payload": _scalar(rec.get("survives_in_payload")),
        # the verdict, carried across unchanged
        "t2pw.support_class": _scalar(rec.get("support_class")),
        "t2pw.support_reason": _scalar(rec.get("support_reason")),
    }
    attrs.update(reintroduction_fields(rec))
    return attrs


def retrieval_attributes(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """OpenInference RETRIEVER attributes, or ``None`` when nothing was recovered.

    Returning ``None`` rather than an empty document list is deliberate: a retriever
    span with zero documents reads in Phoenix as "retrieval found nothing", which is a
    claim about the pipeline. "We could not reconstruct it from the archive" is a
    different fact and gets no span at all.
    """

    span_text = rec.get("retrieved_span")
    if not span_text or span_text == "unavailable":
        return None
    attrs: Dict[str, Any] = {
        "openinference.span.kind": "RETRIEVER",
        "retrieval.documents.0.document.id": _scalar(
            (rec.get("retrieved_chunk_ids") or ["unavailable"])[0]
            if isinstance(rec.get("retrieved_chunk_ids"), list) else "unavailable"),
        "retrieval.documents.0.document.content": _scalar(span_text)[:4000],
        "retrieval.documents.0.document.metadata": json.dumps({
            "source_paper": rec.get("retrieved_source_paper"),
            "section": rec.get("retrieval_section"),
            "gap_id": rec.get("rag_candidate_gap_id"),
            "admission_result": rec.get("admission_result"),
            "chunk_join_scope": rec.get("chunk_join_scope"),
            "reaction_specific": rec.get("chunk_join_reaction_specific"),
        }, ensure_ascii=False),
        "t2pw.retrieval.chunk_join_scope": _scalar(rec.get("chunk_join_scope")),
        "t2pw.retrieval.reaction_specific": _scalar(rec.get("chunk_join_reaction_specific")),
    }
    score = rec.get("retrieval_score")
    if isinstance(score, (int, float)):
        attrs["retrieval.documents.0.document.score"] = float(score)
    return attrs


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_root")
    ap.add_argument("--lineage-json", required=True,
                    help="the R-D092-1 records file (rd092_1_reaction_lineage.py --json)")
    ap.add_argument("--metrics-json", default=None,
                    help="the two-table metrics file, attached to each leg span")
    ap.add_argument("--limit", type=int, default=0, help="ingest at most N reactions")
    ap.add_argument("--endpoint", default="http://localhost:6006/v1/traces")
    ap.add_argument("--no-server", action="store_true",
                    help="do not launch Phoenix; export to --endpoint only")
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="keep the launched server up this long after ingest, so a "
                         "bounded job can be given a real timeout instead of running "
                         "a detached dashboard (G11: no untracked background jobs)")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    records = json.loads(Path(args.lineage_json).read_text(encoding="utf-8"))["records"]
    unknown = check_vocabulary(records)
    if unknown:
        raise SystemExit(f"unknown support classes in records, refusing to ingest: {unknown}")
    if args.limit:
        records = records[: args.limit]

    metrics_by_leg: Dict[str, Any] = {}
    if args.metrics_json:
        doc = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
        metrics_by_leg = {l["leg_dir"]: l["scores"] for l in doc.get("legs", [])}

    session = None
    if not args.no_server:
        import phoenix as px
        session = px.launch_app()
        print(f"phoenix ui: {session.url}")

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "t2pw-eval"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=args.endpoint)))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("t2pw.rd093")

    by_leg: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_leg.setdefault(rec["leg_dir"], []).append(rec)

    n_reactions = n_retrieval = 0
    for leg_dir, rows in by_leg.items():
        head = rows[0]
        with tracer.start_as_current_span("leg") as leg_span:
            leg_span.set_attribute("openinference.span.kind", "CHAIN")
            leg_span.set_attribute("t2pw.leg_dir", leg_dir)
            leg_span.set_attribute("t2pw.run", _scalar(head.get("run")))
            leg_span.set_attribute("t2pw.population", _scalar(head.get("population")))
            leg_span.set_attribute("t2pw.target_paper", _scalar(head.get("target_paper")))
            leg_span.set_attribute("t2pw.reactions", len(rows))
            scores = metrics_by_leg.get(leg_dir)
            if scores:
                # Both tables, with their denominators, never a single blended rate.
                t1 = scores["table1_paper_extraction"]
                t2 = scores["table2_final_support"]
                leg_span.set_attribute("t2pw.table1.recall", _scalar(t1.get("recall")))
                leg_span.set_attribute("t2pw.table1.recall_denominator",
                                       _scalar(t1.get("recall_denominator_verified_signatures")))
                leg_span.set_attribute("t2pw.table1.precision", _scalar(t1.get("precision")))
                leg_span.set_attribute(
                    "t2pw.table1.precision_denominator",
                    _scalar(t1.get("precision_denominator_rows_claimed_target_paper_supported")))
                leg_span.set_attribute("t2pw.table1.evaluable", _scalar(t1.get("evaluable")))
                leg_span.set_attribute("t2pw.table2.unsupported_rate",
                                       _scalar(t2.get("unsupported_rate")))
                leg_span.set_attribute("t2pw.table2.retained_rows_denominator",
                                       _scalar(t2.get("retained_rows")))

            for rec in rows:
                with tracer.start_as_current_span("reaction") as rx_span:
                    for k, v in reaction_attributes(rec).items():
                        rx_span.set_attribute(k, v)
                    n_reactions += 1
                    ra = retrieval_attributes(rec)
                    if ra is not None:
                        with tracer.start_as_current_span("rag_retrieval") as r_span:
                            for k, v in ra.items():
                                r_span.set_attribute(k, v)
                            n_retrieval += 1

    provider.force_flush()
    provider.shutdown()

    summary = {
        "instrument": "rd093_phoenix_ingest",
        "charter": "D-093 s.5.6",
        "evaluation_only": True,
        "phoenix_is_not_ground_truth": True,
        "legs_ingested": len(by_leg),
        "reaction_spans": n_reactions,
        "retrieval_spans": n_retrieval,
        "endpoint": args.endpoint,
        "ui": getattr(session, "url", None) if session else None,
    }
    print(json.dumps(summary, indent=1))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if session and args.hold_seconds > 0:
        import time
        time.sleep(args.hold_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
