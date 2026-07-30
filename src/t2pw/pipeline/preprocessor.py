import json
import logging
import re
from typing import Any, Dict, Optional

from t2pw.llm.client import chat_detailed
from t2pw.paths import PROMPTS_DIR
from t2pw.pipeline.extraction_diagnostics import (
    BOUNDARY_STAGE0_PREPROCESS,
    OUTCOME_EMPTY_COMPLETION,
    OUTCOME_INVALID_JSON,
    OUTCOME_OK,
    current as current_diagnostics,
    payload_hash,
)

logger = logging.getLogger(__name__)

_EMPTY_CONTEXT: Dict[str, Any] = {
    "pathway_name": "",
    "likely_organism": "",
    "key_compounds": [],
    "key_proteins": [],
    "likely_compartments": [],
    "main_subprocesses": [],
    "relevant_sections": [],
    "pathway_relevance_score": 0.0,
}

#: Top-level key on every :func:`preprocess` result holding *why* Stage 0
#: produced what it produced.  Without it an API error, a malformed reply and a
#: genuinely empty result are indistinguishable to callers and in the UI.
#:
#: The value is always a dict::
#:
#:     {
#:       "status": "ok" | "llm_error" | "unparseable" | "empty_reply",
#:       "recovered": <bool>,    # ok path only; True == repaired truncated reply
#:       "detail": "<one-line human-readable cause>",
#:       "raw_len": <int>,       # reply paths only (unparseable / empty_reply /
#:                               # recovered ok)
#:       "raw_preview": "<str>", # reply paths only, capped at _RAW_PREVIEW_LIMIT
#:     }
#:
#: ``recovered`` is always present on the ``ok`` path and is ``False`` for a
#: clean parse.  ``True`` means the model reply was cut off mid-JSON (a
#: ``max_tokens`` truncation) and was structurally repaired, so the context is
#: real but may be *incomplete* — callers and the UI must be able to tell that
#: apart from a fully clean Stage 0 run.
#:
#: It is written *after* the model's own keys are merged in, so a model reply
#: that happens to contain this key can never masquerade as the real status.
PREPROCESS_STATUS_KEY = "preprocess_status"

#: Hard cap (in characters, marker included) on the raw-reply preview.
_RAW_PREVIEW_LIMIT = 200

#: Most-recent candidate cut points the truncation repair will try, newest
#: first.  Bounds the worst case on a very long reply; the usable cut is always
#: near the truncation point, so the cap never costs a real recovery.
_REPAIR_MAX_CANDIDATES = 400


def _format_user_task_context(user_task_context: Optional[str]) -> str:
    """
    Format optional user scoping context for the Stage 0 prompt.

    Source of truth: ``t2pw.pipeline.pipeline._format_user_task_context``.  It is
    replicated here rather than imported because ``pipeline`` already imports
    this module (``from t2pw.pipeline.preprocessor import ...``), so importing it
    back would create a circular import.  Keep the two in sync.

    The context is untrusted text; neutralize matching close-tags so user text
    cannot break out of the intended block in the prompt.
    """
    if not user_task_context or not user_task_context.strip():
        return ""
    safe_context = user_task_context.strip().replace("</user_task_context>", "<\\/user_task_context>")
    return f"<user_task_context>\n{safe_context}\n</user_task_context>"


def preprocess(
    text: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    user_task_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lightweight preprocessing pass: sends raw text to the LLM and returns a
    structured biological context summary.

    ``user_task_context`` is the optional user-supplied extraction focus.  When
    non-blank it is prepended to the user message as a ``<user_task_context>``
    block so the Stage 0 system prompt can reach its "specific example named"
    branch (Case B); without it every multi-example review deterministically
    falls through to the ambiguous Case C.  When it is None/blank the messages
    are byte-identical to the no-context form.

    ``max_tokens`` defaults to 2000 because the Stage 0 output contract is
    large: Case B of ``preprocess_system.txt`` asks for ``selected_example``
    plus up to ten fully-described ``candidate_examples`` plus every standard
    field.  The previous default of 500 truncated real replies mid-JSON, which
    threw away a perfectly correct answer.  It stays a parameter so callers can
    tighten it.

    The returned dict always has all keys from _EMPTY_CONTEXT.  If the LLM
    fails or returns unparseable output, the empty context is returned so
    callers never need to handle None.

    It additionally always carries :data:`PREPROCESS_STATUS_KEY`, recording why
    the result looks the way it does (``ok`` / ``llm_error`` / ``unparseable`` /
    ``empty_reply``) so an empty context is never ambiguous.  On the ``ok`` path
    it also carries ``recovered``: ``True`` means the reply arrived truncated
    and was structurally repaired, so fields may be missing.  That key is set
    last, after the model's own keys are merged, so it cannot be clobbered.
    """
    system_prompt = (PROMPTS_DIR / "preprocess_system.txt").read_text(encoding="utf-8")
    task_context_block = _format_user_task_context(user_task_context)
    user_prompt = (
        (f"{task_context_block}\n\n" if task_context_block else "")
        + "Analyze the following text and return the structured context summary JSON.\n\n"
        "<<<\n"
        f"{text.strip()}\n"
        ">>>"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    context: Dict[str, Any]
    status: Dict[str, Any]
    # Populated on every path, including the exception one, and folded into the
    # status dict below. Stage 0 is the only stage that calls the provider once
    # with no parse-retry wrapper, so if these facts are not captured here they
    # are not captured anywhere.
    call_diagnostics: Dict[str, Any] = {
        "model": "",
        "finish_reason": "",
        "attempts": 0,
        "response_status": "",
        "terminal_reason": "",
    }
    try:
        completion = chat_detailed(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_json=True,
            model_env_var="OPENROUTER_PREPROCESSOR_MODEL",
            stage_name="preprocessor",
        )
        raw = completion.text
        call_diagnostics = completion.diagnostics.to_dict()
        result, recovered = _parse_json_reply(raw)
        raw_text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
        if isinstance(result, dict):
            context = {**_EMPTY_CONTEXT, **result}
            if recovered:
                # Never silent: a repaired reply is a *partial* reply.
                logger.warning(
                    "Preprocessor reply was truncated (%d chars); recovered %d field(s) "
                    "by repairing the JSON.",
                    len(raw_text),
                    len(result),
                )
                status = {
                    "status": "ok",
                    "recovered": True,
                    "detail": (
                        f"The model reply ({len(raw_text)} chars) was truncated mid-JSON "
                        "and was repaired; some fields may be missing."
                    ),
                    "raw_len": len(raw_text),
                    "raw_preview": _preview(raw_text),
                }
            else:
                status = {
                    "status": "ok",
                    "recovered": False,
                    "detail": "Stage 0 returned a JSON object.",
                }
        else:
            context = dict(_EMPTY_CONTEXT)
            if not raw_text.strip():
                # Distinct from `unparseable`: there was nothing to parse at all.
                logger.warning("Preprocessor returned an empty reply; using empty context.")
                status = {
                    "status": "empty_reply",
                    "detail": f"The model returned an empty reply ({len(raw_text)} chars).",
                    "raw_len": len(raw_text),
                    "raw_preview": "",
                }
            else:
                logger.warning("Preprocessor returned non-dict JSON; using empty context.")
                status = {
                    "status": "unparseable",
                    "detail": (
                        f"The model reply ({len(raw_text)} chars) was not a JSON object."
                    ),
                    "raw_len": len(raw_text),
                    "raw_preview": _preview(raw_text),
                }
    except Exception as exc:
        logger.warning("Preprocessor call failed: %s", exc)
        context = dict(_EMPTY_CONTEXT)
        status = {
            "status": "llm_error",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    # Set last, so a model-supplied key of the same name cannot shadow the
    # real status.  Every path above starts from _EMPTY_CONTEXT, so the
    # "all _EMPTY_CONTEXT keys are present" invariant holds unconditionally.
    #
    # The provider-boundary facts ride *inside* the status dict rather than at
    # the top level of the context, because the context is passed to prompt
    # builders and merged into payloads; a diagnostic key loose at that level
    # would eventually be read as pathway data.  ``describe_preprocess_status``
    # and the batch driver both already reach through ``preprocess_status``.
    status.update(call_diagnostics)
    context[PREPROCESS_STATUS_KEY] = status
    _record_stage_zero_boundary(status, context)
    return context


def _record_stage_zero_boundary(status: Dict[str, Any], context: Dict[str, Any]) -> None:
    """File this Stage-0 draw with the run's diagnostics recorder.

    Every draw is recorded, including the successful ones: ``stage0_attempts
    .json`` exists to answer "what did Stage 0 actually do", and an artifact that
    only appears on failures cannot answer that for the run where Stage 0
    succeeded but returned a context too thin to guide extraction.

    Never raises. Stage 0 already fails closed to an empty context; a diagnostics
    bug must not be able to turn that into an exception the callers do not expect.
    """

    try:
        diagnostics = current_diagnostics()
        name = str(status.get("status") or "unknown")
        if name == "empty_reply":
            outcome = OUTCOME_EMPTY_COMPLETION
        elif name == "unparseable":
            outcome = OUTCOME_INVALID_JSON
        elif name == "llm_error":
            outcome = "provider_error"
        else:
            outcome = OUTCOME_OK

        attempt = {
            "boundary": BOUNDARY_STAGE0_PREPROCESS,
            "stage": "stage_0",
            "status": name,
            "outcome": outcome,
            "recovered": bool(status.get("recovered", False)),
            "detail": status.get("detail", ""),
            "model": status.get("model", ""),
            "finish_reason": status.get("finish_reason", ""),
            "attempts": status.get("attempts", 0),
            "response_status": status.get("response_status", ""),
            "terminal_reason": status.get("terminal_reason", ""),
            "raw_chars": status.get("raw_len", 0),
            "raw_preview": status.get("raw_preview", ""),
            # Hashing the *context* rather than the reply is deliberate: the two
            # Stage-0 draws (full text, then bounded head) send different prompts
            # and get different replies, so reply hashes cannot be compared, but
            # identical context hashes prove the retry bought nothing.
            "context_hash": payload_hash(
                {key: value for key, value in context.items() if key != PREPROCESS_STATUS_KEY}
            ),
            # Stage 0 emits a context, not a payload, so the counts that matter
            # here are its search anchors -- the fields _has_usable_context reads
            # to decide whether extraction is guided and whether RAG can build a
            # query. A Stage-0 reply that parsed but named nothing is a distinct
            # failure from one that never parsed, and only these say which.
            "context_counts": _context_counts(context),
        }
        diagnostics.record_stage0_attempt(attempt)
    except Exception:  # noqa: BLE001 - diagnostics must never break Stage 0
        logger.debug("Stage 0 diagnostics could not be recorded", exc_info=True)


def _context_counts(context: Dict[str, Any]) -> Dict[str, int]:
    """How many anchors this Stage-0 context actually carries.

    Counts only -- never the names themselves. A context legitimately holds ten
    fully-described ``candidate_examples`` (Case B of ``preprocess_system.txt``),
    and copying those into an artifact rewritten on every attempt is exactly the
    repeated-blob problem the diagnostics contract forbids.
    """

    counts: Dict[str, int] = {}
    for key in (
        "key_compounds",
        "key_proteins",
        "likely_compartments",
        "main_subprocesses",
        "relevant_sections",
        "candidate_examples",
    ):
        value = context.get(key)
        counts[key] = len(value) if isinstance(value, (list, tuple)) else 0
    for key in ("pathway_name", "likely_organism"):
        counts[f"{key}_chars"] = len(str(context.get(key) or "").strip())
    return counts


def describe_preprocess_status(ctx: Optional[Dict[str, Any]]) -> str:
    """
    Render :data:`PREPROCESS_STATUS_KEY` as a single human-readable clause for
    the UI, e.g. ``"llm_error - ConnectionError: timed out"`` or
    ``"returned unparseable JSON (1234 chars): {oops..."``.

    Returns ``"unknown (no status recorded)"`` when the key is missing, which
    only happens for contexts that did not come from :func:`preprocess`.
    """
    status = ctx.get(PREPROCESS_STATUS_KEY) if isinstance(ctx, dict) else None
    if not isinstance(status, dict):
        return "unknown (no status recorded)"

    name = _clean_scalar(status.get("status")) or "unknown"
    detail = _clean_scalar(status.get("detail"))
    if name == "unparseable":
        preview = _clean_scalar(status.get("raw_preview"))
        return f"returned unparseable JSON ({status.get('raw_len')} chars): {preview}"
    if status.get("recovered"):
        # A repaired truncation must never read like a clean success.
        return f"{name} (recovered from a truncated reply) - {detail or 'reply was repaired'}"
    if detail:
        return f"{name} - {detail}"
    return name


def preprocess_was_recovered(ctx: Optional[Dict[str, Any]]) -> bool:
    """
    True when this context only exists because a truncated Stage 0 reply was
    repaired.  The parsed fields are genuine, but the reply was cut off, so
    fields the model had not written yet are simply absent.
    """
    status = ctx.get(PREPROCESS_STATUS_KEY) if isinstance(ctx, dict) else None
    if not isinstance(status, dict):
        return False
    return bool(status.get("recovered"))


def strip_preprocess_status(ctx: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Return ``ctx`` without :data:`PREPROCESS_STATUS_KEY`.

    Use this at any call site that serializes the whole context into an LLM
    prompt (e.g. the final completeness audit): the diagnostic block can carry a
    preview of an untrusted raw model reply, which has no business being
    replayed into another prompt.
    """
    if not isinstance(ctx, dict) or PREPROCESS_STATUS_KEY not in ctx:
        return ctx
    return {key: value for key, value in ctx.items() if key != PREPROCESS_STATUS_KEY}


def format_context_header(ctx: Optional[Dict[str, Any]]) -> str:
    """
    Render a pathway context dict as a compact plaintext header to prepend
    to extraction/inference prompts.  Returns "" if ctx is empty or None.
    """
    if not ctx or not isinstance(ctx, dict):
        return ""

    pathway = ctx.get("pathway_name", "").strip()
    organism = ctx.get("likely_organism", "").strip()
    compounds = ctx.get("key_compounds") or []
    proteins = ctx.get("key_proteins") or []
    compartments = ctx.get("likely_compartments") or []
    document_type = _clean_scalar(ctx.get("document_type"))
    context_type = _clean_scalar(ctx.get("context_type"))
    scope_status = _clean_scalar(ctx.get("scope_status"))
    selected_example = _clean_scalar(ctx.get("selected_example"))
    warning = _clean_scalar(ctx.get("warning"))
    candidate_example_names = _candidate_example_names(ctx.get("candidate_examples"))
    has_scope_clarity_score = "scope_clarity_score" in ctx
    scope_clarity_score = ctx.get("scope_clarity_score")
    scope_fields = [
        document_type,
        context_type,
        scope_status,
        selected_example,
        warning,
        candidate_example_names,
    ]
    if has_scope_clarity_score and scope_clarity_score is not None:
        scope_fields.append(str(scope_clarity_score))

    # Only emit the header if there is at least one meaningful field.
    if not any([pathway, organism, compounds, proteins, compartments, *scope_fields]):
        return ""

    lines = ["PATHWAY CONTEXT (from preprocessor):"]
    if document_type:
        lines.append(f"document_type: {document_type}")
    if context_type:
        lines.append(f"context_type: {context_type}")
    if scope_status:
        lines.append(f"scope_status: {scope_status}")
    if has_scope_clarity_score and scope_clarity_score is not None:
        lines.append(f"scope_clarity_score: {scope_clarity_score}")
    if "selected_example" in ctx:
        lines.append(f"selected_example: {selected_example}")
    if candidate_example_names:
        lines.append(f"candidate_examples: {', '.join(candidate_example_names)}")
    if warning:
        lines.append(f"warning: {warning}")
    if pathway:
        lines.append(f"Pathway: {pathway}")
    if organism:
        lines.append(f"Organism: {organism}")
    if compounds:
        lines.append(f"Key compounds: {', '.join(str(c) for c in compounds)}")
    if proteins:
        lines.append(f"Key proteins: {', '.join(str(p) for p in proteins)}")
    if compartments:
        lines.append(f"Expected compartments: {', '.join(str(c) for c in compartments)}")

    return "\n".join(lines)


def is_ambiguous_multi_example_review_context(ctx: Optional[Dict[str, Any]]) -> bool:
    """
    Return True when Stage 0 identified a multi-example review but did not
    select a single target example. Downstream extraction must stop in this
    state to avoid building a merged pathway from unrelated examples.
    """
    if not isinstance(ctx, dict):
        return False
    document_type = _clean_scalar(ctx.get("document_type")).casefold()
    if document_type != "multi_example_review":
        return False
    return _is_empty_value(ctx.get("selected_example"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preview(text: str, limit: int = _RAW_PREVIEW_LIMIT) -> str:
    """
    Collapse whitespace and hard-cap ``text`` at ``limit`` characters (the
    truncation marker counts toward the cap, so ``len(result) <= limit`` always).
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(limit - 3, 0)] + "..."


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _candidate_example_names(candidate_examples: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(candidate_examples, list):
        return names
    for item in candidate_examples:
        if isinstance(item, dict):
            name = _clean_scalar(item.get("name"))
        else:
            name = _clean_scalar(item)
        if name:
            names.append(name)
    return names

def _parse_json_reply(raw: str) -> tuple[Optional[Any], bool]:
    """
    Parse a Stage 0 reply, returning ``(value, recovered)``.

    ``recovered`` is True only when the reply could not be parsed as-is and was
    salvaged by :func:`_repair_truncated_json` — i.e. the model was cut off
    mid-JSON.  Order is deliberate: code-fence stripping, then the raw parse,
    then the trailing-comma parse, and only then the truncation repair.
    """
    text = (raw or "").strip()
    if not text:
        return None, False

    # Strip common code-fence markers without dropping content.
    text = text.replace("```json", "```").replace("```", "")

    start = text.find("{")
    if start == -1:
        return None, False

    # Try the text from the first '{' to the end.
    candidate = text[start:]
    try:
        return json.loads(candidate), False
    except json.JSONDecodeError:
        pass

    # Try stripping trailing commas and re-parsing.
    cleaned = _strip_trailing_commas(candidate)
    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        pass

    # Last resort: the reply is very likely a `max_tokens` truncation.  Salvage
    # the complete prefix rather than discarding a correct answer.
    repaired = _repair_truncated_json(candidate)
    if repaired is not None:
        return repaired, True
    return None, False


def _parse_json(raw: str) -> Optional[Any]:
    """Backwards-compatible view of :func:`_parse_json_reply` (value only)."""
    value, _recovered = _parse_json_reply(raw)
    return value


def _strip_trailing_commas(text: str) -> str:
    previous = None
    cleaned = text
    while cleaned != previous:
        previous = cleaned
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _repair_truncated_json(text: str) -> Optional[Any]:
    """
    Salvage a JSON object that was cut off mid-write (``max_tokens``).

    Strategy: scan once, tracking string state so a brace/bracket that merely
    *appears inside a string value* is never mistaken for structure, and record
    every position that is provably an element boundary together with the stack
    of containers open at that position.  Then, newest boundary first, cut the
    text there, append the closers for the still-open containers in reverse
    order of opening, and try ``json.loads``.  The first candidate that parses
    wins; if none do, return ``None`` exactly as before.

    Only three boundary kinds are recorded, and each one is a place where the
    prefix is a whole number of complete elements:

    * a ``,`` inside a container (cut *before* the comma),
    * the position just after a ``}``/``]`` that closed a nested container,
    * the position just after a closing string quote.

    A bare token (number / ``true`` / ``false`` / ``null``) that runs to the end
    of the text is deliberately *not* a boundary: ``123`` truncated to ``12``
    would parse fine and silently yield the wrong value.
    """
    # (cut_index, open_container_stack_at_that_point)
    candidates: list[tuple[int, str]] = []
    stack: list[str] = []
    in_string = False
    escaped = False

    def closers() -> str:
        return "".join("}" if opener == "{" else "]" for opener in reversed(stack))

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                # This character is consumed by the preceding backslash: it is
                # never structural, not even `"` / `{` / `}`.
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if stack:
                    candidates.append((index + 1, closers()))
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                # Unbalanced close: nothing sane left to salvage.
                break
            opener = stack.pop()
            if (opener == "{") != (char == "}"):
                # Mismatched pair: the text is malformed, not truncated.
                return None
            if stack:
                candidates.append((index + 1, closers()))
        elif char == "," and stack:
            # Cut *before* the comma, dropping the (possibly partial) element
            # that followed it.
            candidates.append((index, closers()))

    for cut, closing in reversed(candidates[-_REPAIR_MAX_CANDIDATES:]):
        try:
            return json.loads(text[:cut] + closing)
        except json.JSONDecodeError:
            continue
    return None
