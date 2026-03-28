import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from llm_client import chat

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
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


def preprocess(
    text: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> Dict[str, Any]:
    """
    Lightweight preprocessing pass: sends raw text to the LLM and returns a
    structured biological context summary.

    The returned dict always has all keys from _EMPTY_CONTEXT.  If the LLM
    fails or returns unparseable output, the empty context is returned so
    callers never need to handle None.
    """
    system_prompt = (PROMPTS_DIR / "preprocess_system.txt").read_text(encoding="utf-8")
    user_prompt = (
        "Analyze the following text and return the structured context summary JSON.\n\n"
        "<<<\n"
        f"{text.strip()}\n"
        ">>>"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = chat(messages, temperature=temperature, max_tokens=max_tokens, response_json=True)
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

    # Only emit the header if there is at least one meaningful field.
    if not any([pathway, organism, compounds, proteins, compartments]):
        return ""

    lines = ["PATHWAY CONTEXT (from preprocessor):"]
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
