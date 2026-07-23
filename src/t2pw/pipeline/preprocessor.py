import json
import logging
import re
from typing import Any, Dict, Optional

from t2pw.llm.client import chat
from t2pw.paths import PROMPTS_DIR

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
    max_tokens: int = 500,
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

    The returned dict always has all keys from _EMPTY_CONTEXT.  If the LLM
    fails or returns unparseable output, the empty context is returned so
    callers never need to handle None.
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

    try:
        raw = chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_json=True,
            model_env_var="OPENROUTER_PREPROCESSOR_MODEL",
            stage_name="preprocessor",
        )
        result = _parse_json(raw)
        if isinstance(result, dict):
            return {**_EMPTY_CONTEXT, **result}
        logger.warning("Preprocessor returned non-dict JSON; using empty context.")
    except Exception as exc:
        logger.warning("Preprocessor call failed: %s", exc)

    return dict(_EMPTY_CONTEXT)


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

def _parse_json(raw: str) -> Optional[Any]:
    text = (raw or "").strip()
    if not text:
        return None

    # Strip common code-fence markers without dropping content.
    text = text.replace("```json", "```").replace("```", "")

    start = text.find("{")
    if start == -1:
        return None

    # Try the text from the first '{' to the end.
    candidate = text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Try stripping trailing commas and re-parsing.
    cleaned = _strip_trailing_commas(candidate)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _strip_trailing_commas(text: str) -> str:
    previous = None
    cleaned = text
    while cleaned != previous:
        previous = cleaned
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned
