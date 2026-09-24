"""The model boundary for mining, with the empty/truncated pattern handled.

Everything crossing the provider boundary goes through
:meth:`MiningModelClient.call_json`, which returns a validated payload or an
explicit failure -- never a quietly repaired fragment.

The failure mode this exists for
--------------------------------
``finish_reason=length``, an empty 200, or JSON that stops mid-object. All three
used to look like "the model found nothing". Here each one:

* is recorded as a :class:`~t2pw.mining.models.ModelCallRecord` with its real
  reason, tokens and latency;
* triggers a retry with a **smaller passage** and, on the last attempt, a
  **tighter schema** -- not an identical re-send;
* if it never succeeds, yields ``payload=None`` and a status in
  :data:`~t2pw.mining.models.NON_ADMISSIBLE_EXTRACTION_STATUSES`, which
  :func:`t2pw.mining.admission.judge` rejects before any biological test.

A partial parse is kept **only** as a diagnostic, labelled
``partial_repair_diagnostic_only``, and can never become a reaction.

The underlying LLM client is imported lazily inside the call, so importing
``t2pw.mining`` never constructs a provider client or reads an API key.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from t2pw.mining.budget import BudgetLedger
from t2pw.mining.config import MiningModelConfig
from t2pw.mining.ids import call_id as make_call_id
from t2pw.mining.models import (
    EXTRACTION_EMPTY,
    EXTRACTION_OK,
    EXTRACTION_PARSE_ERROR,
    EXTRACTION_PARTIAL_REPAIR,
    EXTRACTION_SCHEMA_ERROR,
    EXTRACTION_TRUNCATED,
    ModelCallRecord,
)

#: The fraction of the passage kept on each successive retry.
_SHRINK_LADDER = (1.0, 0.6, 0.35)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

#: A provider error can quote a masked key prefix; the lineage file never keeps it.
_SECRETISH = re.compile(r"sk-[A-Za-z0-9_.\-]*")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _strip_fence(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text)
    return text.strip()


def looks_truncated(raw: str) -> bool:
    """Is this text a JSON object/array that simply stops?

    Bracket-balance, not a parser guess: an object that opened more braces than
    it closed was cut off, which is a different failure from malformed JSON and
    wants a different response (smaller passage, not a reworded prompt).
    """
    text = _strip_fence(raw)
    if not text:
        return False
    if not text.lstrip().startswith(("{", "[")):
        return False
    depth_curly = text.count("{") - text.count("}")
    depth_square = text.count("[") - text.count("]")
    return depth_curly > 0 or depth_square > 0


def parse_json_object(raw: str) -> Tuple[Optional[Any], str]:
    """Parse ``raw`` leniently. Returns ``(payload, error)``.

    Lenient means: strip a code fence and trim to the outermost brace pair. It
    does **not** mean repairing truncated JSON -- a cut-off object returns an
    error, and the caller records it as such.
    """
    text = _strip_fence(raw)
    if not text:
        return None, "empty response"
    try:
        return json.loads(text), ""
    except Exception as exc:  # noqa: BLE001
        first_error = str(exc)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1]), ""
            except Exception:  # noqa: BLE001
                continue
    return None, first_error


def validate_against_schema(payload: Any, schema: Optional[Dict[str, Any]]) -> str:
    """Return a schema error string, or ``""`` when the payload conforms.

    Uses ``jsonschema`` (a declared project dependency). If it is somehow absent
    the check degrades to a shape assertion rather than silently passing
    everything -- an unvalidated payload must never look like a validated one.
    """
    if not schema:
        return ""
    body = schema.get("schema") if isinstance(schema, dict) and "schema" in schema else schema
    try:
        import jsonschema  # type: ignore
    except Exception:  # noqa: BLE001
        if isinstance(body, dict) and body.get("type") == "object" and not isinstance(payload, dict):
            return "payload is not an object (jsonschema unavailable; shape check only)"
        return ""
    try:
        jsonschema.validate(instance=payload, schema=body)
    except Exception as exc:  # noqa: BLE001
        return str(exc).splitlines()[0][:400]
    return ""


@dataclass
class ModelCallOutcome:
    """What one logical model call produced."""

    record: ModelCallRecord
    payload: Optional[Any] = None
    status: str = EXTRACTION_OK
    raw_text: str = ""
    #: A partially parsed payload kept for diagnosis ONLY. Never admitted.
    partial: Optional[Any] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    #: One row per attempt: the raw completion text and how it was classified.
    raw_attempts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == EXTRACTION_OK and self.payload is not None

    def lineage_rows(self) -> List[Dict[str, Any]]:
        """``rounds/round_NNN/model_raw.jsonl`` rows, linked by ``call_id``.

        The raw completion is model output only: no request, no headers, no key.
        """
        base = {
            "call_id": self.record.call_id,
            "round_index": self.record.round_index,
            "purpose": self.record.purpose,
            "paper_id": self.record.paper_id,
            "call_status": self.status,
        }
        return [{**base, **row} for row in self.raw_attempts]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "status": self.status,
            "has_payload": self.payload is not None,
            "has_partial": self.partial is not None,
            "attempts": list(self.attempts),
        }


class MiningModelClient:
    """Bounded, logged, schema-validated access to the extraction model."""

    def __init__(
        self,
        config: MiningModelConfig,
        ledger: BudgetLedger,
        *,
        chat_fn: Optional[Callable[..., Any]] = None,
        token_stats_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self._chat_fn = chat_fn
        self._token_stats_fn = token_stats_fn
        self._clock = clock or time.monotonic
        self.records: List[ModelCallRecord] = []
        self._ordinal = 0

    # -- plumbing -----------------------------------------------------------
    def _chat(self) -> Callable[..., Any]:
        if self._chat_fn is not None:
            return self._chat_fn
        from t2pw.llm.client import chat_detailed  # lazy: constructs the provider client

        return chat_detailed

    def _token_stats(self) -> Dict[str, Any]:
        if self._token_stats_fn is not None:
            try:
                return dict(self._token_stats_fn() or {})
            except Exception:  # noqa: BLE001
                return {}
        try:
            from t2pw.llm.client import get_token_stats

            return dict(get_token_stats() or {})
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Crude fallback when the provider reports no usage. ~4 chars/token."""
        return max(1, len(str(text or "")) // 4)

    # -- the call -----------------------------------------------------------
    def call_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]] = None,
        tight_schema: Optional[Dict[str, Any]] = None,
        round_index: int = 0,
        paper_id: str = "",
        evidence_ids: Sequence[str] = (),
        max_tokens: Optional[int] = None,
    ) -> ModelCallOutcome:
        """One logical call, with the shrink-and-retry ladder applied.

        Returns an outcome whose ``status`` says exactly what happened. The
        budget is consulted before every attempt, so a run that has exhausted its
        token allowance stops here rather than inside the provider.
        """
        self._ordinal += 1
        cid = make_call_id(round_index, purpose or "call", self._ordinal)
        record = ModelCallRecord(
            call_id=cid,
            round_index=int(round_index),
            purpose=purpose,
            requested_model=self.config.model,
            prompt_version=self.config.prompt_version,
            reasoning=self.config.reasoning,
            paper_id=paper_id,
            evidence_ids=list(evidence_ids),
            started_at=_utcnow(),
        )
        outcome = ModelCallOutcome(record=record, status=EXTRACTION_EMPTY)

        blocked = self.ledger.check_before_call()
        if blocked:
            record.status = "budget_blocked"
            record.parse_error = f"not attempted: {blocked}"
            outcome.status = EXTRACTION_EMPTY
            self.records.append(record)
            return outcome

        models_to_try: List[Tuple[str, bool]] = [(self.config.model, False)]
        if self.config.fallback_model:
            models_to_try.append((self.config.fallback_model, True))

        last_error = ""
        for model_slug, is_fallback in models_to_try:
            for index, fraction in enumerate(_SHRINK_LADDER[: max(1, self.config.max_attempts)]):
                if self.ledger.check_before_call():
                    last_error = "budget exhausted mid-ladder"
                    break
                passage = user if fraction >= 1.0 else user[: max(200, int(len(user) * fraction))]
                use_schema = schema
                if index >= len(_SHRINK_LADDER) - 1 and tight_schema is not None:
                    use_schema = tight_schema
                schema_label = "tight" if use_schema is tight_schema and tight_schema is not None else (
                    "full" if use_schema is not None else "none")
                attempt = self._attempt(
                    record=record,
                    model_slug=model_slug,
                    is_fallback=is_fallback,
                    system=system,
                    user=passage,
                    schema=use_schema,
                    max_tokens=max_tokens,
                    attempt_index=index + 1,
                    shrink_fraction=fraction,
                    schema_label=schema_label,
                )
                outcome.attempts.append(attempt["log"])
                # One row per ladder rung, with what the rung sent and every
                # provider request it made (IT2-5: pilot-002 kept neither).
                outcome.raw_attempts.append(
                    {
                        "attempt": index + 1,
                        "model": model_slug,
                        "shrink_fraction": fraction,
                        "passages_sent": passage.count("[evidence_id: "),
                        "chars_sent": len(passage),
                        "schema": schema_label,
                        "provider_requests": attempt["provider_requests"],
                        "finish_reason": attempt["log"].get("finish_reason", ""),
                        "parse_outcome": attempt["log"].get("classification", ""),
                        "error": _SECRETISH.sub("[redacted]", str(attempt.get("error") or "")),
                        "response_hash": attempt["log"].get("response_hash", ""),
                        "raw_text": attempt["raw"],
                    }
                )
                if attempt["status"] == EXTRACTION_OK:
                    outcome.payload = attempt["payload"]
                    outcome.status = EXTRACTION_OK
                    outcome.raw_text = attempt["raw"]
                    record.status = "ok"
                    self.records.append(record)
                    return outcome
                outcome.status = attempt["status"]
                outcome.raw_text = attempt["raw"]
                if attempt.get("partial") is not None:
                    outcome.partial = attempt["partial"]
                last_error = attempt.get("error", "")
                if attempt["status"] == EXTRACTION_SCHEMA_ERROR and index + 1 >= self.config.max_attempts:
                    break
            if outcome.status == EXTRACTION_OK:
                break
            if is_fallback or not self.config.fallback_model:
                break
            record.fallback_used = True
            record.fallback_reason = (
                f"primary model {self.config.model!r} produced {outcome.status!r}; "
                f"falling back to {self.config.fallback_model!r} (explicit, configured, logged)"
            )

        record.status = outcome.status
        record.parse_error = record.parse_error or last_error
        self.records.append(record)
        return outcome

    def _attempt(
        self,
        *,
        record: ModelCallRecord,
        model_slug: str,
        is_fallback: bool,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]],
        max_tokens: Optional[int],
        attempt_index: int,
        shrink_fraction: float,
        schema_label: str = "",
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        before = self._token_stats()
        started = float(self._clock())
        raw = ""
        finish_reason = ""
        error = ""
        diagnostics: Any = None
        provider_requests: List[Dict[str, Any]] = []
        try:
            result = self._chat()(
                messages,
                temperature=self.config.temperature,
                max_tokens=int(max_tokens or self.config.max_tokens),
                response_json=True,
                json_schema=schema if (schema and self.config.use_structured_output) else None,
                model_override=model_slug,
                stage_name=f"mining:{record.purpose}",
                timeout=self.config.request_timeout_seconds,
            )
            raw = str(getattr(result, "text", result) or "")
            diagnostics = getattr(result, "diagnostics", None)
            if diagnostics is not None:
                finish_reason = str(getattr(diagnostics, "finish_reason", "") or "")
                record.request_hash = str(getattr(diagnostics, "request_hash", "") or "")
                record.response_hash = str(getattr(diagnostics, "response_hash", "") or "")
                record.retries = max(record.retries, int(getattr(diagnostics, "attempts", 0) or 0))
                attempt_log = getattr(diagnostics, "attempt_log", None)
                if isinstance(attempt_log, list):
                    # Appended per rung, never overwritten: every provider request survives.
                    provider_requests = [
                        {k: _SECRETISH.sub("[redacted]", v) if isinstance(v, str) else v
                         for k, v in (row.items() if isinstance(row, dict) else {"entry": str(row)}.items())}
                        for row in attempt_log
                    ]
                    record.attempt_log.extend(
                        {**row, "rung": attempt_index, "shrink_fraction": shrink_fraction}
                        for row in provider_requests
                    )
                actual = str(getattr(diagnostics, "model", "") or "")
                if actual:
                    record.actual_model = actual
        except Exception as exc:  # noqa: BLE001 - a provider error is a recorded outcome
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = int((float(self._clock()) - started) * 1000)
        record.latency_ms += latency_ms
        record.finish_reason = finish_reason or record.finish_reason
        record.actual_model = record.actual_model or model_slug
        record.fallback_used = record.fallback_used or is_fallback
        record.response_chars = len(raw)

        after = self._token_stats()
        input_tokens = max(0, int(after.get("prompt_tokens", 0) or 0) - int(before.get("prompt_tokens", 0) or 0))
        output_tokens = max(0, int(after.get("completion_tokens", 0) or 0) - int(before.get("completion_tokens", 0) or 0))
        if not input_tokens and not output_tokens:
            input_tokens = self._estimate_tokens(system) + self._estimate_tokens(user)
            output_tokens = self._estimate_tokens(raw)
        cost = self.ledger.record_call(input_tokens=input_tokens, output_tokens=output_tokens)
        record.input_tokens += input_tokens
        record.output_tokens += output_tokens
        record.estimated_cost_usd = round(record.estimated_cost_usd + cost, 6)

        log = {
            "attempt": attempt_index,
            "model": model_slug,
            "shrink_fraction": shrink_fraction,
            "chars_sent": len(user),
            "chars_received": len(raw),
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "schema": schema_label or ("full" if schema is not None else "none"),
            "response_hash": str(getattr(diagnostics, "response_hash", "") or ""),
        }

        if error:
            log["error"] = error
            log["classification"] = "provider_error"
            return {"status": EXTRACTION_PARSE_ERROR, "payload": None, "raw": raw,
                    "error": error, "log": log, "provider_requests": provider_requests}

        if not raw.strip():
            log["classification"] = "empty_completion"
            return {"status": EXTRACTION_EMPTY, "payload": None, "raw": raw,
                    "error": f"empty completion (finish_reason={finish_reason or 'unknown'})",
                    "log": log, "provider_requests": provider_requests}

        truncated = finish_reason == "length" or looks_truncated(raw)
        payload, parse_error = parse_json_object(raw)

        if payload is None and truncated:
            log["classification"] = "truncated_completion"
            record.parse_error = parse_error
            return {"status": EXTRACTION_TRUNCATED, "payload": None, "raw": raw,
                    "partial": None, "error": parse_error or "truncated JSON", "log": log,
                    "provider_requests": provider_requests}

        if payload is None:
            log["classification"] = "parse_error"
            record.parse_error = parse_error
            return {"status": EXTRACTION_PARSE_ERROR, "payload": None, "raw": raw,
                    "error": parse_error, "log": log,
                    "provider_requests": provider_requests}

        if truncated:
            # Parsed, but the provider says it stopped on length: the object is
            # syntactically whole only by luck. Kept as a diagnostic, never
            # admitted -- the next attempt gets a smaller passage.
            log["classification"] = "truncated_but_parseable"
            return {"status": EXTRACTION_PARTIAL_REPAIR, "payload": None, "raw": raw,
                    "partial": payload,
                    "error": "finish_reason=length; parsed payload kept as diagnostic only",
                    "log": log, "provider_requests": provider_requests}

        schema_error = validate_against_schema(payload, schema)
        if schema_error:
            log["classification"] = "schema_error"
            record.schema_error = schema_error
            return {"status": EXTRACTION_SCHEMA_ERROR, "payload": None, "raw": raw,
                    "partial": payload, "error": schema_error, "log": log,
                    "provider_requests": provider_requests}

        log["classification"] = "ok"
        return {"status": EXTRACTION_OK, "payload": payload, "raw": raw, "error": "", "log": log,
                "provider_requests": provider_requests}


__all__ = [
    "MiningModelClient",
    "ModelCallOutcome",
    "looks_truncated",
    "parse_json_object",
    "validate_against_schema",
]
