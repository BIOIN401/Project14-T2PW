"""promptfoo custom provider: one Stage-1 extraction draw under a given system prompt.

promptfoo compares PROVIDERS side by side, so each prompt variant is registered as its own
provider with its own ``config.system_prompt``. The system prompt therefore never passes
through promptfoo's Nunjucks templating, which matters: these prompts are ~35,000
characters of JSON-shaped text, and a stray ``{{`` or ``{%`` in one of them would fail to
render or, worse, render differently from the file on disk and make the comparison a lie.

The call mirrors ``pipeline.run_extraction_pipeline`` through the same private stage runner,
so retries and JSON repair behave as in production. The only difference between providers is
the system prompt, which is the point. If that function's arguments change, update this file
with it; ``scripts/prompt_ab.py`` carries the same mirror and the same warning.

What promptfoo shows in its table is the readable summary a human would read in the app.
The full payload is written beside the config so a difference can be chased into the JSON.

Config per provider::

    - id: file://provider.py
      label: v01-committed
      config:
        system_prompt: ../../../<path to the variant>.txt   # relative to THIS file
        out_dir: C:/t/promptfoo-out/pw701720                # optional

Test vars: ``text_path`` (required), ``pathway``, ``organism``, ``scope``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _app_extraction_max_tokens() -> int:
    """The output budget the APP gives Stage 1, resolved the same way it does.

    A harness that under-budgets the model does not measure the prompt, it measures the
    budget. At 12000 the longer variants came back EMPTY with ``finish_reason=length``:
    the model spent the whole allowance before emitting any content, six of ten rows
    errored, and it looked like a prompt failure. streamlit_app.py reads
    OPENROUTER_EXTRACTION_MAX_TOKENS with a 64000 default; this file reads the same.
    """

    import os

    raw = (os.getenv("OPENROUTER_EXTRACTION_MAX_TOKENS") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 64000
    return value if 500 <= value <= 128000 else 64000


def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (HERE / p).resolve()


def _extract(system_prompt: str, paper_text: str, *, pathway: str, organism: str,
             scope: str, temperature: float, max_tokens: int,
             max_attempts: int) -> Dict[str, Any]:
    from t2pw.pipeline.pipeline import (_build_extraction_prompt, _run_json_stage,
                                        is_ambiguous_multi_example_review_context)

    context: Dict[str, Any] = {}
    if pathway:
        context["name"] = pathway
    if organism:
        context["organism"] = organism
    if is_ambiguous_multi_example_review_context(context or None):
        raise RuntimeError("the pathway context reads as an ambiguous multi-example review")

    payload, _logs = _run_json_stage(
        stage_name="extraction",
        model_env_var="OPENROUTER_EXTRACTION_MODEL",
        system_prompt=system_prompt,
        build_user_prompt=lambda prev_output, last_error: _build_extraction_prompt(
            paper_text, prev_output, last_error,
            pathway_context=context or None,
            pathway_scope=scope or None,
            user_task_context=None,
        ),
        max_attempts=max_attempts,
        temperature=temperature,
        max_tokens=max_tokens,
        retry_on_empty_payload=True,
    )
    return payload


def _summary(payload: Dict[str, Any]) -> str:
    try:
        from t2pw.pipeline.draft_graph import build_draft_graph
        from t2pw.pipeline.reaction_summary import generate_reaction_summary
        try:
            from t2pw.pipeline.qa_graph import generate_qa_report
        except Exception:  # noqa: BLE001
            generate_qa_report = None  # type: ignore[assignment]
        graph = build_draft_graph(payload)
        qa: Dict[str, Any] = {}
        if generate_qa_report is not None:
            try:
                qa = generate_qa_report(graph)
            except Exception:  # noqa: BLE001
                qa = {}
        return generate_reaction_summary(graph, qa)
    except Exception:  # noqa: BLE001 - a broken render must never lose the payload
        return "could not render a summary:\n" + traceback.format_exc()


def _counts(payload: Dict[str, Any]) -> Dict[str, int]:
    processes = payload.get("processes") or {}
    entities = payload.get("entities") or {}
    out = {k: len(v) for k, v in processes.items() if isinstance(v, list)}
    for key in ("compounds", "proteins", "protein_complexes", "nucleic_acids"):
        out[key] = len(entities.get(key) or [])
    rows = [r for r in (processes.get("interactions") or []) if isinstance(r, dict)]
    out["interactions_without_evidence"] = sum(
        1 for r in rows if not str(r.get("evidence") or "").strip())
    out["relationship_vocabulary_violations"] = sum(
        1 for r in rows
        if str(r.get("relationship") or "").strip().lower()
        not in {"activation", "inhibition", "binding", "interaction", ""})
    return out


def call_api(prompt: str, options: Optional[Dict[str, Any]] = None,
             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """promptfoo entry point. Returns ``{"output": ...}`` or ``{"error": ...}``."""

    options = options or {}
    context = context or {}
    config = options.get("config") or {}
    variables = context.get("vars") or {}

    try:
        from t2pw.config import ensure_dotenv_loaded
        ensure_dotenv_loaded()   # not the app: without this LLM_PROVIDER is unset

        prompt_path = config.get("system_prompt")
        if not prompt_path:
            return {"error": "provider config needs system_prompt"}
        system_file = _resolve(str(prompt_path))
        if not system_file.is_file():
            return {"error": f"no such system prompt: {system_file}"}
        system_prompt = system_file.read_text(encoding="utf-8")

        text_path = variables.get("text_path")
        if not text_path:
            return {"error": "test vars need text_path"}
        paper = _resolve(str(text_path))
        if not paper.is_file():
            return {"error": f"no such paper text: {paper}"}
        paper_text = paper.read_text(encoding="utf-8")

        label = str(config.get("label") or system_file.stem)
        payload = _extract(
            system_prompt, paper_text,
            pathway=str(variables.get("pathway") or ""),
            organism=str(variables.get("organism") or ""),
            scope=str(variables.get("scope") or ""),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens") or _app_extraction_max_tokens()),
            max_attempts=int(config.get("max_attempts", 2)),
        )

        summary = _summary(payload)
        counts = _counts(payload)

        out_dir = config.get("out_dir")
        if out_dir:
            d = Path(str(out_dir))
            d.mkdir(parents=True, exist_ok=True)
            stem = f"{label}__{paper.stem}"
            (d / f"{stem}__payload.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            (d / f"{stem}__summary.txt").write_text(summary, encoding="utf-8")

        header = (f"variant: {label}\n"
                  f"system prompt: {system_file.name} "
                  f"({len(system_prompt)} chars, "
                  f"sha256 {hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()[:16]})\n"
                  f"counts: {json.dumps(counts)}\n"
                  + "-" * 72 + "\n")
        return {
            "output": header + summary,
            "metadata": {"label": label, "counts": counts,
                         "system_prompt_sha256":
                             hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                         "paper_sha256": hashlib.sha256(paper.read_bytes()).hexdigest()},
        }
    except BaseException as exc:  # noqa: BLE001 - one provider must not kill the grid
        return {"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}
