from __future__ import annotations

import json
from typing import Any, Dict, List

SYSTEM_PROMPT = (
    "You are a patch planner for SBML models intended for PathWhiz import. "
    "Output ONLY valid JSON matching the PatchPlan schema. Do NOT output SBML. "
    "Prefer minimal edits. Primary goals: remove disconnected species, merge duplicate reactions, "
    "and normalize obvious compartment redundancies. Only propose operations from the allowed list. "
    "If unsure, add a warning instead of changing."
)


USER_PROMPT_TEMPLATE = """You will be given:
- A summary of the SBML model we want to improve.
- Summaries of example SBML files known to import into PathWhiz.

Task:
Produce a PatchPlan JSON that makes the model more PathWhiz-friendly by:
- eliminating degree-0 species (unused species)
- merging duplicate reactions with identical reactants/products/compartment
- reducing redundant compartments when examples suggest a simpler pattern
- optionally moving mislocalized species if examples strongly imply they belong elsewhere

Do NOT add new reactions unless allow_add_reaction=true.

Constraints:
- Allowed ops: {allowed_ops}
- If allow_add_reaction=false, do NOT include add_reaction ops.
- Every op must include a short reason.

INPUT_MODEL_SUMMARY:
{input_model_summary}

EXAMPLE_SUMMARIES:
{example_summaries}

allow_add_reaction = {allow_add_reaction}

Return ONLY PatchPlan JSON.
"""


def build_user_prompt(
    input_model_summary: Dict[str, Any],
    example_summaries: List[Dict[str, Any]],
    *,
    allow_add_reaction: bool,
    allowed_ops: List[str],
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        allowed_ops=", ".join(allowed_ops),
        input_model_summary=json.dumps(input_model_summary, ensure_ascii=False, separators=(",", ":")),
        example_summaries=json.dumps(example_summaries, ensure_ascii=False, separators=(",", ":")),
        allow_add_reaction="true" if allow_add_reaction else "false",
    )

