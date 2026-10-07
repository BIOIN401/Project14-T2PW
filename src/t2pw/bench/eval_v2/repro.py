"""LLM reproducibility-metadata completeness over COMMITTED run artifacts (ORCH-742c).

TRACEABILITY FIRST, and read-only: this measures which fields the committed artifacts
record today. It adds no field, forces no provider or seed, and changes no production
semantics. Where a field is absent, the report names the exact file/function where a
future hook would go -- it does not add one.

Sources read: ``LEG_TRACE.jsonl`` (``model_attempt`` events written by
``t2pw.batch.leg_trace.record_model_attempt`` via ``t2pw.llm.client._publish_attempt``),
``audit_iteration_summary.json`` (per audit round), ``manifest.jsonl`` and ``plan.json``.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

#: field -> (where it would come from, future hook location if it is not recorded)
FIELDS: Dict[str, Dict[str, str]] = {
    "requested_model": {
        "recorded_as": "LEG_TRACE model_attempt.model (the model the client RESOLVED and sent)",
        "hook": "src/t2pw/llm/client.py::chat_detailed -- already passed to record_model_attempt",
    },
    "served_model": {
        "recorded_as": "not recorded (resp.model is never read)",
        "hook": "src/t2pw/llm/client.py::chat_detailed, after _client.chat.completions.create: resp.model",
    },
    "temperature": {
        "recorded_as": "audit_iteration_summary.json rounds[].temperature (audit stage only)",
        "hook": "src/t2pw/llm/client.py::chat_detailed kwargs['temperature'] -> "
                "src/t2pw/batch/leg_trace.py::record_model_attempt (no parameter today)",
    },
    "seed": {
        "recorded_as": "not recorded; no seed is sent (kwargs carry none)",
        "hook": "src/t2pw/llm/client.py::chat_detailed kwargs (no evaluation-only config surface exists)",
    },
    "provider_backend_identity": {
        "recorded_as": "not recorded (OpenRouter's response 'provider' is never read)",
        "hook": "src/t2pw/llm/client.py::chat_detailed: getattr(resp, 'provider', None)",
    },
    "finish_reason": {
        "recorded_as": "LEG_TRACE model_attempt.finish_reason",
        "hook": "",
    },
    "request_hash": {
        "recorded_as": "LEG_TRACE model_attempt.request_hash = client._hash(messages): a hash of the "
                       "MESSAGES only; model/temperature/max_tokens are not in it",
        "hook": "src/t2pw/llm/client.py::chat_detailed (CompletionDiagnostics(request_hash=_hash(messages)))",
    },
    "prompt_hash": {
        "recorded_as": "same value as request_hash (the only hash taken is over the messages)",
        "hook": "",
    },
    "response_hash": {
        "recorded_as": "LEG_TRACE model_attempt.response_hash -- the key exists but is written EMPTY: "
                       "CompletionDiagnostics.note() publishes the attempt row before chat_detailed "
                       "assigns diagnostics.response_hash (client.py, the `diagnostics.response_hash = "
                       "_hash(text)` line), so the published value is always ''",
        "hook": "src/t2pw/llm/client.py::_publish_attempt / CompletionDiagnostics.note ordering",
    },
    "token_usage": {
        "recorded_as": "not recorded per call; client._record_usage only accumulates process-wide "
                       "totals in memory (_token_stats)",
        "hook": "src/t2pw/llm/client.py::_record_usage -> record_model_attempt",
    },
    "retry_count": {
        "recorded_as": "derived: max model_attempt.attempt per (stage, request_hash)",
        "hook": "",
    },
}


def _git_files(paths: Sequence[str], name: str) -> List[str]:
    return [p for p in paths if p.endswith("/" + name)]


def completeness(repo_root: Path, paths: Sequence[str]) -> Dict[str, Any]:
    """Field-by-field presence over every committed model_attempt record."""

    root = Path(repo_root)
    traces = _git_files(paths, "LEG_TRACE.jsonl")
    legs_with_payload = {p.rpartition("/")[0] for p in paths
                         if p.endswith(("/final_mapped.json", "/merged_payload.json"))}
    attempts: List[Dict[str, Any]] = []
    for t in traces:
        for line in (root / t).read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "model_attempt":
                attempts.append(e)
    n = len(attempts)

    def present(key: str) -> int:
        return sum(1 for a in attempts if str(a.get(key, "")).strip())

    retries = collections.defaultdict(int)
    for a in attempts:
        k = (a.get("stage"), a.get("request_hash"))
        retries[k] = max(retries[k], int(a.get("attempt") or 0))
    audit_rounds = audit_with_temp = 0
    for p in _git_files(paths, "audit_iteration_summary.json"):
        try:
            doc = json.loads((root / p).read_text(encoding="utf-8"))
        except ValueError:
            continue
        for r in doc.get("rounds") or ():
            audit_rounds += 1
            audit_with_temp += int("temperature" in r)
    measured = {
        "requested_model": present("model"),
        "served_model": 0,
        "temperature": 0,
        "seed": present("seed"),
        "provider_backend_identity": present("provider"),
        "finish_reason": present("finish_reason"),
        "request_hash": present("request_hash"),
        "prompt_hash": present("request_hash"),
        "response_hash": present("response_hash"),
        "token_usage": present("usage") + present("prompt_tokens"),
        "retry_count": n,
    }
    table = []
    for field, info in FIELDS.items():
        k = measured[field]
        table.append({"field": field, "model_attempts_with_field": k, "model_attempts": n,
                      "completeness": round(k / n, 4) if n else None,
                      "status": "recorded" if n and k == n else "partial" if k else "not_recorded",
                      **info})
    return {
        "legs_with_payload": len(legs_with_payload),
        "legs_with_leg_trace": len(traces),
        "runs_with_leg_trace": sorted({t.split("/papers/")[0] for t in traces}),
        "model_attempt_records": n,
        "distinct_calls": len(retries),
        "calls_with_retries": sum(1 for v in retries.values() if v > 1),
        "audit_rounds": audit_rounds,
        "audit_rounds_with_temperature": audit_with_temp,
        "fields": table,
    }


__all__ = ["FIELDS", "completeness"]
