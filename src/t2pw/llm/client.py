import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError, AuthenticationError, BadRequestError
from t2pw.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load .env reliably (project root = folder containing "src")
# -----------------------------------------------------------------------------
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

PROVIDER = (os.getenv("LLM_PROVIDER", "local") or "local").strip().lower()


def _mask(s: str, keep: int = 8) -> str:
    if not s:
        return "None"
    return s[:keep] + "..."


# -----------------------------------------------------------------------------
# Client setup
# -----------------------------------------------------------------------------
if PROVIDER == "openrouter":
    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    model = (os.getenv("OPENROUTER_MODEL") or "").strip()

    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY not loaded. Looked for .env at: {ENV_PATH}")
    if not api_key.startswith("sk-or-"):
        raise RuntimeError(
            f"OPENROUTER_API_KEY looks wrong ({_mask(api_key)}). "
            "OpenRouter keys usually start with 'sk-or-'."
        )

    _client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PWML Multi-Stage Pipeline"),
        },
    )
    _model = model

elif PROVIDER == "local":
    base_url = (os.getenv("LMSTUDIO_BASE_URL") or "http://127.0.0.1:1234/v1").strip()
    model = (os.getenv("LMSTUDIO_MODEL") or "meta-llama-3.1-8b-instruct").strip()

    _client = OpenAI(base_url=base_url, api_key="lm-studio")
    _model = model

else:
    raise RuntimeError("LLM_PROVIDER must be 'local' or 'openrouter'")


# -----------------------------------------------------------------------------
# Token usage tracking
# -----------------------------------------------------------------------------
_token_stats: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

# Per-million-token prices (configurable via .env)
_COST_INPUT_PER_M = float(os.getenv("LLM_COST_INPUT_PER_M", "0.12"))
_COST_OUTPUT_PER_M = float(os.getenv("LLM_COST_OUTPUT_PER_M", "0.30"))


def _record_usage(resp: Any) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    _token_stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    _token_stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    _token_stats["calls"] += 1


def get_token_stats() -> Dict[str, Any]:
    """Return a snapshot of cumulative token usage and estimated cost."""
    p = _token_stats["prompt_tokens"]
    c = _token_stats["completion_tokens"]
    cost = (p / 1_000_000) * _COST_INPUT_PER_M + (c / 1_000_000) * _COST_OUTPUT_PER_M
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "api_calls": _token_stats["calls"],
        "estimated_cost_usd": round(cost, 6),
    }


def reset_token_stats() -> None:
    """Reset cumulative token counters (e.g. at the start of a new run)."""
    _token_stats["prompt_tokens"] = 0
    _token_stats["completion_tokens"] = 0
    _token_stats["calls"] = 0


# -----------------------------------------------------------------------------
# Empty-but-successful completions
#
# WHY THIS SECTION EXISTS. Until 2026-07-28 every retry loop in this module fired
# only on a RAISED exception (RateLimitError, APITimeoutError, APIError,
# json.JSONDecodeError). An HTTP 200 whose message.content was "" or None was
# handed straight back to the caller as a success by
# `return (resp.choices[0].message.content or "").strip()`, and finish_reason was
# never looked at anywhere in the file. So the ONE layer that can actually see
# "the provider answered 200 and sent nothing" was also the one layer that
# treated it as an answer, and every stage above it was left to rediscover the
# emptiness in whatever local vocabulary it happened to have.
#
# WHAT THE EVIDENCE DOES AND DOES NOT SAY. The batch in runs/2026-07-28_0919 lost
# two legs to the same abrupt message:
#
#   PMC13278307 / research   FAIL after  137s   "Payload must include a processes object"
#   PMC13231680 / strict     FAIL after   55s   "Payload must include a processes object"
#
# An earlier draft of this comment asserted those two deaths were each caused by
# a SINGLE empty Stage 0 reply. Review on 2026-07-28 checked that against the
# artifacts and it does not hold, so it is corrected here rather than left as a
# plausible story attached to a real fix:
#
#   * Neither leg wrote any artifact at all. Both manifest rows carry `files: []`
#     and the only thing on disk under their run directories is RESULT.txt, so
#     NOTHING in that run records an empty completion. The diagnosis was inferred
#     from the shape of the failure, not read out of a log.
#   * The run executed commit 12bc11b (batch.log: 09:19:08 -> 11:49:05 on
#     2026-07-28; the next commit, 92e1192, landed at 14:06:43). At 12bc11b,
#     streamlit_app.py already drew Stage 0 a SECOND time whenever
#     len(text) > _PREPROCESS_RETRY_CHARS (20,000). The two legs' 01_source_text
#     .txt files are 50,377 and 61,997 chars, so both were over that line and both
#     already got a second draw. The single-empty-reply story therefore needs two
#     consecutive empty completions, not one.
#   * Stage 1 and Stage 2 (pipeline._run_json_stage) do NOT survive an empty reply
#     silently either: "" fails json.loads, so an empty completion is already
#     retried there -- but as a whole extra round-trip of the full prompt, which
#     is where the wall clock goes on a leg that dies in 55s or 137s.
#
# WHAT IS STILL TRUE, AND WHY THE FIX STAYS. The hole itself is not in dispute and
# is visible in the code, independent of any particular leg: an empty 200 was
# returned as a success, cost a caller a full retry round or a degraded result,
# and was invisible to every counter and every log line in this module. Fixing it
# HERE is stage-agnostic on purpose -- it does not matter which stage draws the
# short straw, and the layer that can distinguish "empty because the provider
# hiccuped" from "empty because a content filter stopped it" (see _finish_reason)
# is this one. PMC13231680 is still the reason to believe the emptiness is
# transient rather than a property of the paper: the same paper extracted 3
# reactions the previous day, and in this very run its research leg passed with 4
# reactions in 936.7s while its strict leg died in 54.7s on the same source text.
#
# The fix deliberately does NOT introduce a second retry mechanism or a new
# exception type: an empty completion is folded into the existing loop, backoff
# and LLM_MAX_RETRIES budget, and once that budget is exhausted the empty string
# is returned exactly as before. Callers already cope with it - the preprocessor
# records status "empty_reply" and falls back to the empty context - and raising
# here would change the contract of every call site in the pipeline.
# -----------------------------------------------------------------------------
def _finish_reason(resp: Any) -> str:
    """
    Best-effort read of ``choices[0].finish_reason``, for the log line only.

    We surface it because "empty" has at least two causes that want opposite
    responses, and the 2026-07-28 postmortem could not tell them apart because
    nothing in this module ever recorded it:

      * a provider hiccup - finish_reason "stop" (or absent) with no text.
        Retrying is exactly right. Note that we cannot say this was the
        PMC13231680 case: that leg wrote no artifacts (`files: []` in the
        manifest) and this field was not recorded anywhere in the run, which is
        the whole reason it is surfaced from now on.
      * a moderation / refusal stop - finish_reason "content_filter". Retrying
        the identical prompt LLM_MAX_RETRIES times cannot help and just burns
        wall clock, so the operator needs to see this in the log to know that
        editing the prompt, not re-running the leg, is the fix.

    Never raises. A malformed or partial response object must not be able to kill
    a call by blowing up inside a logging helper.
    """
    try:
        choice = resp.choices[0]
    except Exception:  # noqa: BLE001 - diagnostics must never mask the real result
        return "unavailable"
    return str(getattr(choice, "finish_reason", None) or "unknown")


#: ``finish_reason`` spellings that mean "a moderation layer stopped this", not
#: "the provider hiccuped". Matched as substrings after case-folding because
#: OpenAI-compatible gateways are not consistent: OpenAI says ``content_filter``,
#: some proxies pass through ``content-filter``, and a few report ``safety``.
_CONTENT_FILTER_MARKERS = ("content_filter", "content-filter", "safety")


def _is_content_filter(reason: Any) -> bool:
    """Whether ``reason`` names a moderation stop.

    This is the one finish_reason for which retrying is *provably* useless:
    moderation is a deterministic function of the prompt, so re-sending the
    identical prompt gets the identical verdict. Every other empty completion is
    a transient worth another draw. See :func:`chat_detailed` for what is done
    with the answer.
    """

    token = str(reason or "").strip().casefold()
    return any(marker in token for marker in _CONTENT_FILTER_MARKERS)


#: Terminal reasons a completion call can carry back. ``""`` means it succeeded.
TERMINAL_CONTENT_FILTER = "content_filter"
TERMINAL_EMPTY_AFTER_RETRIES = "empty_after_retries"

#: Raw response statuses, one per observable provider behaviour.
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_CONTENT_FILTER = "content_filter"
STATUS_ERROR = "error"

#: Longest error/preview string kept on a diagnostics row. Diagnostics are
#: written on every attempt, so nothing unbounded may enter them.
_DIAG_CHARS = 240


def _diag_clip(value: Any, limit: int = _DIAG_CHARS) -> str:
    text = value if isinstance(value, str) else ("" if value is None else str(value))
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars elided)"


def _hash(value: Any) -> str:
    """Content fingerprint used to prove two attempts sent/received the same body.

    Deliberately duplicated from ``pipeline.extraction_diagnostics.payload_hash``
    rather than imported: ``t2pw.llm.client`` is imported *by* the pipeline and
    by standalone tools, and giving it a pipeline dependency for a four-line
    helper would invert that direction.
    """

    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.encode("utf-8", "replace")
    else:
        try:
            raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode(
                "utf-8", "replace"
            )
        except (TypeError, ValueError):  # pragma: no cover - default=str makes this rare
            raw = str(value).encode("utf-8", "replace")
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class CompletionDiagnostics:
    """What happened at one crossing of the provider boundary.

    Every field here is something the 2026-07-28 postmortem needed and could not
    get: which model actually answered (stage overrides make this non-obvious),
    what ``finish_reason`` it stopped on, how many attempts were spent, and
    whether the reply was empty because of a hiccup or a moderation stop.

    Bounded by construction -- counts, hashes and clipped previews only. The
    request and reply bodies are represented by :func:`_hash` so two attempts can
    be compared without either body being stored.
    """

    model: str = ""
    stage: str = ""
    attempts: int = 0
    finish_reason: str = ""
    response_status: str = ""
    terminal_reason: str = ""
    request_hash: str = ""
    response_hash: str = ""
    raw_chars: int = 0
    attempt_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "stage": self.stage,
            "attempts": self.attempts,
            "finish_reason": self.finish_reason,
            "response_status": self.response_status,
            "terminal_reason": self.terminal_reason,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "raw_chars": self.raw_chars,
            "attempt_log": list(self.attempt_log),
        }

    def note(
        self,
        *,
        attempt: int,
        status: str,
        finish_reason: str = "",
        content_chars: int = 0,
        error: str = "",
    ) -> None:
        row: Dict[str, Any] = {
            "attempt": int(attempt),
            "status": str(status),
            "content_chars": int(content_chars),
        }
        if finish_reason:
            row["finish_reason"] = str(finish_reason)
        if error:
            row["error"] = _diag_clip(error)
        self.attempt_log.append(row)
        self.attempts = int(attempt)
        if finish_reason:
            self.finish_reason = str(finish_reason)
        self.response_status = str(status)


@dataclass
class CompletionResult:
    """A completion plus why it looks the way it does.

    ``text`` is byte-for-byte what :func:`chat` has always returned, so a caller
    that only wants the string keeps today's contract; ``diagnostics`` is the
    addition.
    """

    text: str
    diagnostics: CompletionDiagnostics


def _completion_is_empty(resp: Any, *, tools_were_sent: bool) -> bool:
    """
    True when a 200 response carries no payload the caller can actually use.

    What counts as "payload" depends on what we asked the model for, and getting
    this backwards in the tool-calling direction would be a far worse regression
    than the bug being fixed here:

      * tools WERE sent -> an assistant message with ``tool_calls`` and
        ``content=None`` is the normal, correct shape of a function-calling turn;
        it is what the whole agentic loop in chat_with_tools is built to consume.
        Retrying such a reply would re-issue a round the model already answered
        and could double-execute tools. So tool_calls count as payload and the
        reply is NOT empty, however blank its content is.
      * tools were NOT sent -> chat(), and the forced final round of
        chat_with_tools (``_call_once(..., include_tools=False)``), both reduce
        the reply to ``(message.content or "").strip()`` before handing it back.
        Text is the only payload that can exist, so blank text means the caller
        gets "" and the call bought nothing.

    Whitespace-only content is treated as empty, matching both the ``.strip()``
    the callers already apply and the preprocessor's own empty-reply test, which
    feeds it "   \\n\\t " and expects status "empty_reply".

    Never raises: a response shaped unlike anything we recognise is reported as
    empty so it goes down the retry path rather than crashing the call.
    """
    try:
        message = resp.choices[0].message
    except Exception:  # noqa: BLE001 - unrecognisable shape -> treat as a transient
        return True

    if tools_were_sent and getattr(message, "tool_calls", None):
        return False

    return not (getattr(message, "content", None) or "").strip()


def _resolve_model(
    *,
    model_override: Optional[str] = None,
    model_env_var: Optional[str] = None,
) -> str:
    """
    Resolve the model for a call.

    Priority:
      1. explicit model_override
      2. stage-specific OpenRouter env var named by the caller
      3. OPENROUTER_MODEL
      4. provider default captured during client setup
    """
    override = (model_override or "").strip()
    if override:
        return override

    if PROVIDER == "openrouter":
        if model_env_var:
            stage_model = (os.getenv(model_env_var) or "").strip()
            if stage_model:
                return stage_model
        global_model = (os.getenv("OPENROUTER_MODEL") or "").strip()
        if global_model:
            return global_model

    if _model:
        return _model

    raise RuntimeError("Missing OPENROUTER_MODEL in .env")


# -----------------------------------------------------------------------------
# Chat function (NOW supports response_json=True)
# -----------------------------------------------------------------------------
def chat(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 800,
    response_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    model_env_var: Optional[str] = None,
    stage_name: Optional[str] = None,
) -> str:
    """
    response_json=True:
      - If json_schema is provided, requests structured JSON using the schema.
      - Otherwise requests a JSON object output.

    Returns the completion text. This is the historical signature and every
    existing call site keeps it; :func:`chat_detailed` is the same call with the
    diagnostics attached, for the boundaries that have to record why a reply
    looks the way it does.
    """
    return chat_detailed(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_json=response_json,
        json_schema=json_schema,
        model_override=model_override,
        model_env_var=model_env_var,
        stage_name=stage_name,
    ).text


def chat_detailed(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 800,
    response_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    model_env_var: Optional[str] = None,
    stage_name: Optional[str] = None,
) -> CompletionResult:
    """:func:`chat` plus a :class:`CompletionDiagnostics` describing the call.

    The text returned is identical to what ``chat`` returns for the same inputs;
    the only behavioural difference anywhere in this function relative to the
    pre-2026-07-29 code is the content-filter exit described below.

    CONTENT FILTER IS A DURABLE EXIT, NOT A TRANSIENT
    -------------------------------------------------
    An empty 200 whose ``finish_reason`` is ``content_filter`` is not a hiccup.
    Moderation is a deterministic function of the prompt, so LLM_MAX_RETRIES
    re-sends of the *identical* prompt buy exactly one thing: LLM_MAX_RETRIES
    times the latency, and then the same empty string. The loop therefore stops
    on the first one and reports ``terminal_reason="content_filter"``, which is
    the fact an operator needs -- editing the prompt is the fix, re-running the
    leg is not. Retrying a *different* prompt is still allowed and is what the
    localized-repair path does; what is forbidden is the blind identical retry.

    A ``content_filter`` reply that carries text is NOT affected: it is a partial
    answer, it is returned like any other non-empty completion, and the reason is
    recorded on the diagnostics.
    """
    resolved_model = _resolve_model(
        model_override=model_override,
        model_env_var=model_env_var,
    )
    logger.info("LLM model for %s: %s", stage_name or "chat", resolved_model)

    diagnostics = CompletionDiagnostics(
        model=resolved_model,
        stage=str(stage_name or "chat"),
        request_hash=_hash(messages),
    )

    kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # OpenAI-compatible response_format
    if response_json:
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

    # Retry tuning (env configurable)
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "8"))
    base_sleep = float(os.getenv("LLM_RETRY_BASE_SLEEP", "1.0"))
    max_sleep = float(os.getenv("LLM_RETRY_MAX_SLEEP", "20.0"))

    # Spacing helps chunk pipelines avoid bursts
    spacing = float(os.getenv("LLM_CALL_SPACING", "0.35"))

    last_err: Exception | None = None
    empty_attempts = 0

    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(**kwargs)
            _record_usage(resp)
            time.sleep(spacing)

            # An HTTP 200 with no text is a transient, not a result. See the
            # "Empty-but-successful completions" note above: this is the exact
            # shape that killed PMC13278307/research (137s) and
            # PMC13231680/strict (55s) in runs/2026-07-28_0919, and it is
            # deliberately handled *inside* this loop so it shares the same
            # backoff and the same LLM_MAX_RETRIES budget as every other
            # transient rather than growing a second retry mechanism.
            if _completion_is_empty(resp, tools_were_sent=False):
                empty_attempts += 1
                reason = _finish_reason(resp)
                last_err = RuntimeError(
                    f"empty completion from {resolved_model} (finish_reason={reason})"
                )
                if _is_content_filter(reason):
                    # Durable, not transient. Stop here rather than spending the
                    # rest of the budget re-asking a question already refused.
                    diagnostics.note(
                        attempt=attempt + 1,
                        status=STATUS_CONTENT_FILTER,
                        finish_reason=reason,
                        error=str(last_err),
                    )
                    diagnostics.terminal_reason = TERMINAL_CONTENT_FILTER
                    logger.error(
                        "LLM stopped %s on finish_reason=%s (model %s) with no content on "
                        "attempt %d/%d. This is a moderation stop, not a transient: the "
                        "identical prompt will be refused again, so no retry is spent. "
                        "Edit the prompt or the source passage rather than re-running.",
                        stage_name or "chat",
                        reason,
                        resolved_model,
                        attempt + 1,
                        max_retries,
                    )
                    return CompletionResult("", diagnostics)
                diagnostics.note(
                    attempt=attempt + 1,
                    status=STATUS_EMPTY,
                    finish_reason=reason,
                    error=str(last_err),
                )
                if attempt < max_retries - 1:
                    logger.warning(
                        "LLM returned an empty completion for %s (model %s, "
                        "finish_reason=%s) on attempt %d/%d; retrying as a transient.",
                        stage_name or "chat",
                        resolved_model,
                        reason,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(min(base_sleep * (2 ** attempt), max_sleep))
                    continue
                # Budget exhausted. Return the empty string exactly as this
                # function did before 2026-07-28 instead of raising: the
                # preprocessor turns "" into status "empty_reply" plus the empty
                # context, and a new exception type here would change the
                # contract of every call site at once.
                logger.error(
                    "LLM returned an empty completion for %s (model %s, "
                    "finish_reason=%s) on %d of %d attempts; giving up and returning "
                    "the empty string for the caller's own empty-reply handling.",
                    stage_name or "chat",
                    resolved_model,
                    reason,
                    empty_attempts,
                    max_retries,
                )
                diagnostics.terminal_reason = TERMINAL_EMPTY_AFTER_RETRIES
                return CompletionResult("", diagnostics)

            text = (resp.choices[0].message.content or "").strip()
            reason = _finish_reason(resp)
            diagnostics.note(
                attempt=attempt + 1,
                status=STATUS_OK,
                finish_reason=reason,
                content_chars=len(text),
            )
            diagnostics.response_hash = _hash(text)
            diagnostics.raw_chars = len(text)
            return CompletionResult(text, diagnostics)

        except AuthenticationError as e:
            raise RuntimeError(
                f"Authentication failed (401). Key starts with: {_mask(os.getenv('OPENROUTER_API_KEY',''))}. "
                f"Checked .env at: {ENV_PATH}. Original error: {e}"
            ) from e

        except BadRequestError as e:
            err_str = str(e).lower()
            json_format_error = (
                "response_format" in err_str
                or "json_object" in err_str
                or "json mode" in err_str
                or "json schema" in err_str
            )
            if json_format_error and "response_format" in kwargs:
                logger.warning(
                    "Model %s does not support response_format JSON; retrying without it.",
                    resolved_model,
                )
                kwargs.pop("response_format")
                last_err = e
                diagnostics.note(attempt=attempt + 1, status=STATUS_ERROR, error=str(e))
                continue
            raise RuntimeError(
                "Bad request (400). This model/provider may not support response_format JSON. "
                "Try OPENROUTER_MODEL=openrouter/free or a different model. "
                f"Original error: {e}"
            ) from e

        except RateLimitError as e:
            last_err = e
            diagnostics.note(attempt=attempt + 1, status=STATUS_ERROR, error=str(e))
            time.sleep(min(base_sleep * (2 ** attempt), max_sleep))

        except (APITimeoutError, APIError) as e:
            last_err = e
            diagnostics.note(attempt=attempt + 1, status=STATUS_ERROR, error=str(e))
            time.sleep(min(base_sleep * (2 ** attempt), max_sleep))

        except json.JSONDecodeError as e:
            # OpenAI SDK fails to parse a malformed HTTP response (e.g. local server
            # returned truncated or non-JSON body). Treat as transient and retry.
            last_err = e
            diagnostics.note(attempt=attempt + 1, status=STATUS_ERROR, error=str(e))
            time.sleep(min(base_sleep * (2 ** attempt), max_sleep))

    raise RuntimeError(f"LLM failed after {max_retries} retries. Last error: {last_err}")


# -----------------------------------------------------------------------------
# Agentic tool-calling loop
# -----------------------------------------------------------------------------
def chat_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_executor: Callable[[str, Dict[str, Any]], Any],
    temperature: float = 0.0,
    max_tokens: int = 1200,
    max_tool_rounds: int = 8,
    model_override: Optional[str] = None,
    model_env_var: Optional[str] = None,
    stage_name: Optional[str] = None,
) -> str:
    """
    Agentic tool-calling loop using the OpenAI function-calling protocol.

    Sends messages + tool definitions to the model.
    When the model returns tool_calls, executes them via tool_executor and
    feeds results back as role="tool" messages.
    Loops until the model returns content (final answer) or max_tool_rounds is hit.

    tool_executor(tool_name, tool_args_dict) -> Any
      Return value is JSON-serialised and sent back as the tool result.
    """
    spacing = float(os.getenv("LLM_CALL_SPACING", "0.35"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "8"))
    base_sleep = float(os.getenv("LLM_RETRY_BASE_SLEEP", "1.0"))
    max_sleep = float(os.getenv("LLM_RETRY_MAX_SLEEP", "20.0"))
    resolved_model = _resolve_model(
        model_override=model_override,
        model_env_var=model_env_var,
    )
    logger.info("LLM model for %s: %s", stage_name or "chat_with_tools", resolved_model)

    working_messages: List[Dict[str, Any]] = list(messages)

    def _call_once(msgs: List[Dict[str, Any]], include_tools: bool) -> Any:
        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Captured rather than re-derived: it is the single fact that decides
        # whether a content-less reply is a legitimate function-calling turn or
        # a dead 200. `include_tools and tools` is the real condition (a caller
        # can ask for tools and pass none), so the emptiness check must read the
        # same expression that decided what actually went on the wire.
        tools_were_sent = bool(include_tools and tools)
        if tools_were_sent:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        last_err: Optional[Exception] = None
        empty_attempts = 0
        for attempt in range(max_retries):
            try:
                resp = _client.chat.completions.create(**kwargs)
                _record_usage(resp)
                time.sleep(spacing)

                # Same 2026-07-28 hole as chat(), reached from both call sites
                # below: the tool round (include_tools=True) and the forced
                # final round (include_tools=False). `tools_were_sent` is what
                # keeps a valid tool_calls-only reply out of this branch -
                # retrying one of those would re-issue a round the model has
                # already answered.
                if _completion_is_empty(resp, tools_were_sent=tools_were_sent):
                    empty_attempts += 1
                    reason = _finish_reason(resp)
                    last_err = RuntimeError(
                        f"empty completion from {resolved_model} (finish_reason={reason})"
                    )
                    if _is_content_filter(reason):
                        # Same durable-exit rule as chat(): a moderation stop is
                        # a property of the prompt, and this loop would otherwise
                        # re-send that prompt LLM_MAX_RETRIES times for nothing.
                        # The response is handed back unchanged so the caller
                        # still reduces it to "" exactly as it always has.
                        logger.error(
                            "LLM stopped %s on finish_reason=%s (model %s, tools_sent=%s) "
                            "with no content on attempt %d/%d. Moderation stop: the "
                            "identical prompt will be refused again, so no retry is "
                            "spent. Edit the prompt rather than re-running.",
                            stage_name or "chat_with_tools",
                            reason,
                            resolved_model,
                            tools_were_sent,
                            attempt + 1,
                            max_retries,
                        )
                        return resp
                    if attempt < max_retries - 1:
                        logger.warning(
                            "LLM returned an empty completion for %s (model %s, "
                            "finish_reason=%s, tools_sent=%s) on attempt %d/%d; "
                            "retrying as a transient.",
                            stage_name or "chat_with_tools",
                            resolved_model,
                            reason,
                            tools_were_sent,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(min(base_sleep * (2 ** attempt), max_sleep))
                        continue
                    # Budget exhausted: hand the empty response back unchanged so
                    # the loop below still reduces it to "" the way it always
                    # has. Raising here would break every chat_with_tools caller.
                    logger.error(
                        "LLM returned an empty completion for %s (model %s, "
                        "finish_reason=%s, tools_sent=%s) on %d of %d attempts; "
                        "giving up and returning the empty response.",
                        stage_name or "chat_with_tools",
                        resolved_model,
                        reason,
                        tools_were_sent,
                        empty_attempts,
                        max_retries,
                    )
                    return resp

                return resp
            except AuthenticationError as e:
                raise RuntimeError(f"Authentication failed (401): {e}") from e
            except BadRequestError as e:
                raise RuntimeError(f"Bad request (400): {e}") from e
            except RateLimitError as e:
                last_err = e
                time.sleep(min(base_sleep * (2 ** attempt), max_sleep))
            except (APITimeoutError, APIError) as e:
                last_err = e
                time.sleep(min(base_sleep * (2 ** attempt), max_sleep))
            except json.JSONDecodeError as e:
                last_err = e
                time.sleep(min(base_sleep * (2 ** attempt), max_sleep))
        raise RuntimeError(f"chat_with_tools call failed after retries. Last error: {last_err}")

    for _ in range(max_tool_rounds):
        resp = _call_once(working_messages, include_tools=True)
        message = resp.choices[0].message

        if not message.tool_calls:
            return (message.content or "").strip()

        # Append the assistant turn (with tool_calls) to the conversation
        working_messages.append(message)

        # Execute each tool call and feed results back
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}
            try:
                result = tool_executor(tool_name, tool_args)
                result_str = (
                    json.dumps(result, ensure_ascii=False)
                    if not isinstance(result, str)
                    else result
                )
            except Exception as exc:  # noqa: BLE001
                result_str = json.dumps({"error": str(exc)}, ensure_ascii=False)
            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                }
            )

    # Max rounds reached — force a final text response without tools
    final_resp = _call_once(working_messages, include_tools=False)
    return (final_resp.choices[0].message.content or "").strip()
