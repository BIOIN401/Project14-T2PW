from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import AddReactionOp, PatchPlan


ALLOWED_OPS_BASE = [
    "delete_species",
    "merge_reactions",
    "rename_compartment",
    "delete_compartment",
    "move_species_compartment",
    "rename_species",
]


def _parse_json_maybe(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("LLM returned empty response.")
    if text.startswith("```"):
        text = text.replace("```json", "```").replace("```", "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        parsed = json.loads(text[left : right + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM output was not valid JSON object.")


def _resolve_chat_fn(llm_client: Any):
    if callable(llm_client):
        return llm_client
    fn = getattr(llm_client, "chat", None)
    if callable(fn):
        return fn
    raise ValueError("Invalid llm_client passed to converter. Expected module/object with callable chat().")


def request_patch_plan(
    *,
    llm_client: Any,
    input_model_summary: Dict[str, Any],
    example_summaries: List[Dict[str, Any]],
    allow_add_reaction: bool,
    max_tokens: int = 2400,
) -> Tuple[PatchPlan, Dict[str, Any]]:
    chat_fn = _resolve_chat_fn(llm_client)
    allowed_ops = list(ALLOWED_OPS_BASE)
    if allow_add_reaction:
        allowed_ops.append("add_reaction")
    prompt = build_user_prompt(
        input_model_summary=input_model_summary,
        example_summaries=example_summaries,
        allow_add_reaction=allow_add_reaction,
        allowed_ops=allowed_ops,
    )
    raw = chat_fn(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=int(max_tokens),
        response_json=True,
    )
    parsed = _parse_json_maybe(raw)
    try:
        plan = PatchPlan.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(f"LLM PatchPlan schema validation failed: {exc}") from exc
    if not allow_add_reaction:
        for op in plan.ops:
            if isinstance(op, AddReactionOp):
                raise ValueError("LLM plan included add_reaction while allow_add_reaction=false.")
    return plan, {"raw": raw, "parsed": parsed}

