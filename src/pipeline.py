import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_client import chat
from preprocessor import format_context_header
from qa_graph import build_graph, connected_components, degrees, get_entities

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
logger = logging.getLogger(__name__)


class PipelineFailure(RuntimeError):
    """Raised when a stage cannot produce valid JSON within the allotted attempts."""

    def __init__(self, stage: str, message: str, attempts: List[Dict[str, Any]]):
        super().__init__(message)
        self.stage = stage
        self.attempts = attempts


AttemptLog = Dict[str, Any]
AttemptLogs = List[AttemptLog]


def run_extraction_pipeline(
    input_text: str,
    *,
    pathway_context: Optional[Dict[str, Any]] = None,
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 12000,
) -> Tuple[Dict[str, Any], AttemptLogs]:
    """
    Stage 1: strict extraction. Automatically retries with self-repair instructions if JSON parsing fails.
    """
    return _run_json_stage(
        stage_name="extraction",
        system_prompt=(PROMPTS_DIR / "pwml_system.txt").read_text(encoding="utf-8"),
        build_user_prompt=lambda prev_output, last_error: _build_extraction_prompt(
            input_text, prev_output, last_error, pathway_context=pathway_context
        ),
        max_attempts=max_attempts,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_inference_pipeline(
    input_text: str,
    stage_one: Dict[str, Any],
    *,
    pathway_context: Optional[Dict[str, Any]] = None,
    qa_feedback: Optional[Dict[str, Any]] = None,
    chunk_section: Optional[str] = None,
    chunk_relevance_score: Optional[float] = None,
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 10000,
) -> Tuple[Dict[str, Any], AttemptLogs]:
    """
    Stage 2: inference/enrichment pass. Uses Stage-1 output as context and retries if JSON is invalid.
    """
    stage_one_str = json.dumps(stage_one, indent=2, ensure_ascii=False)
    return _run_json_stage(
        stage_name="inference",
        system_prompt=(PROMPTS_DIR / "pwml_infer_system.txt").read_text(encoding="utf-8"),
        build_user_prompt=lambda prev_output, last_error: _build_inference_prompt(
            input_text,
            stage_one_str,
            prev_output,
            last_error,
            qa_feedback,
            pathway_context=pathway_context,
            chunk_section=chunk_section,
            chunk_relevance_score=chunk_relevance_score,
        ),
        max_attempts=max_attempts,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_stage_two_with_chunking(
    input_text: str,
    stage_one: Dict[str, Any],
    chunk_details: Optional[List[Dict[str, Any]]] = None,
    *,
    pathway_context: Optional[Dict[str, Any]] = None,
    qa_feedback: Optional[Dict[str, Any]] = None,
    enable_chunking: bool,
    chunk_word_limit: int = 8000,
    chunk_overlap: int = 1200,
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 10000,
    compact_stage_one: bool = True,
    retry_on_failure: bool = True,
    retry_max_tokens: Optional[int] = None,
    retry_compact_stage_one: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optionally chunk Stage-2 inference by reusing Stage-1 chunk outputs.
    Returns merged inference additions plus per-chunk details.
    """
    chunks: List[Dict[str, Any]] = []

    if enable_chunking and chunk_details and len(chunk_details) > 1:
        chunks = chunk_details
    else:
        words = input_text.split()
        use_chunks = enable_chunking and len(words) > chunk_word_limit
        if use_chunks:
            chunks = chunk_text(input_text, chunk_word_limit, chunk_overlap)
        else:
            chunks = [
                {
                    "chunk_id": 1,
                    "start_word": 0,
                    "end_word": len(words),
                    "text": input_text,
                }
            ]

    chunk_results: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []

    for chunk in chunks:
        chunk_stage_one = chunk.get("output") if isinstance(chunk, dict) else None
        if not isinstance(chunk_stage_one, dict):
            chunk_stage_one = stage_one

        if compact_stage_one:
            chunk_stage_one = _compact_stage_one_for_inference(chunk_stage_one)

        chunk_section = chunk.get("section")
        chunk_relevance_score = chunk.get("relevance_score")

        try:
            parsed, attempts = run_inference_pipeline(
                chunk["text"],
                chunk_stage_one,
                pathway_context=pathway_context,
                qa_feedback=qa_feedback,
                chunk_section=chunk_section,
                chunk_relevance_score=chunk_relevance_score,
                max_attempts=max_attempts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except PipelineFailure as failure:
            logger.warning(
                "Stage-2 inference failed for chunk %s (%s-%s).",
                chunk.get("chunk_id"),
                chunk.get("start_word"),
                chunk.get("end_word"),
            )
            logger.debug("Stage-2 failure details: %s", failure.attempts)

            if retry_on_failure:
                compact_stage_one = (
                    _compact_stage_one_for_inference(chunk_stage_one)
                    if retry_compact_stage_one
                    else chunk_stage_one
                )
                retry_tokens = (
                    retry_max_tokens
                    if retry_max_tokens is not None
                    else _default_retry_tokens(max_tokens, failure.attempts)
                )
                try:
                    retry_parsed, retry_attempts = run_inference_pipeline(
                        chunk["text"],
                        compact_stage_one,
                        pathway_context=pathway_context,
                        qa_feedback=qa_feedback,
                        chunk_section=chunk_section,
                        chunk_relevance_score=chunk_relevance_score,
                        max_attempts=max_attempts,
                        temperature=temperature,
                        max_tokens=retry_tokens,
                    )
                    attempts = _tag_attempts(failure.attempts, "initial")
                    attempts.extend(_tag_attempts(retry_attempts, "retry"))
                    parsed = retry_parsed
                except PipelineFailure as retry_failure:
                    attempts = _tag_attempts(failure.attempts, "initial")
                    attempts.extend(_tag_attempts(retry_failure.attempts, "retry"))
                    last_error, raw_preview, raw_length, raw_tail = _summarize_failure(attempts)
                    message = (
                        f"Chunk {chunk.get('chunk_id')} failed to produce valid JSON after retry. "
                        f"Last error: {last_error}. Raw length: {raw_length}. "
                        f"Raw preview: {raw_preview}. Raw tail: {raw_tail}"
                    )
                    raise PipelineFailure(
                        stage=f"inference chunk {chunk.get('chunk_id')}",
                        message=message,
                        attempts=attempts,
                    ) from retry_failure
            else:
                last_error, raw_preview, raw_length, raw_tail = _summarize_failure(failure.attempts)
                message = (
                    f"Chunk {chunk.get('chunk_id')} failed to produce valid JSON. "
                    f"Last error: {last_error}. Raw length: {raw_length}. "
                    f"Raw preview: {raw_preview}. Raw tail: {raw_tail}"
                )
                raise PipelineFailure(
                    stage=f"inference chunk {chunk.get('chunk_id')}",
                    message=message,
                    attempts=failure.attempts,
                ) from failure

        parsed = clean_inference_output(parsed)
        chunk_entry = {**chunk, "stage_one": chunk_stage_one, "output": parsed, "attempts": attempts}
        chunk_results.append(chunk_entry)
        outputs.append(parsed)

    merged = merge_inference_outputs(outputs)
    return merged, chunk_results


def _default_retry_tokens(max_tokens: int, attempts: AttemptLogs) -> int:
    """
    Pick retry token budget based on observed failure mode.
    - If JSON appears truncated, increase token budget.
    - Otherwise keep a conservative smaller retry.
    """
    if _looks_truncated_json_failure(attempts):
        return min(24000, max(max_tokens + 800, int(max_tokens * 1.5)))
    return max(200, int(max_tokens * 0.6))


def _looks_truncated_json_failure(attempts: AttemptLogs) -> bool:
    if not attempts:
        return False
    last_error, _, _, _ = _summarize_failure(attempts)
    error = (last_error or "").lower()
    return (
        "unterminated string" in error
        or "expecting value" in error
        or "unexpected end" in error
        or "eof" in error
    )


def build_qa_feedback(payload: Dict[str, Any], *, hint_limit: int = 25) -> Dict[str, Any]:
    """
    Build deterministic graph-QA hints that can be fed back into Stage 2.
    """
    adj, meta = build_graph(payload)
    entities = get_entities(payload)

    for compound_name in entities["compounds"]:
        adj.setdefault(f"compound:{compound_name}", set())
    for protein_name in entities["proteins"]:
        adj.setdefault(f"protein:{protein_name}", set())
    for nucleic_acid_name in entities["nucleic_acids"]:
        adj.setdefault(f"nucleic_acid:{nucleic_acid_name}", set())
    for element_collection_name in entities["element_collections"]:
        adj.setdefault(f"element_collection:{element_collection_name}", set())
    for protein_complex_name in entities["protein_complexes"]:
        adj.setdefault(f"protein_complex:{protein_complex_name}", set())

    comps = connected_components(adj)
    comps_sorted = sorted(comps, key=lambda comp: len(comp), reverse=True)
    deg = degrees(adj)

    orphan_components: List[Dict[str, Any]] = []
    for comp in comps_sorted[1:]:
        orphan_components.append(
            {
                "size": len(comp),
                "nodes": sorted(comp)[:hint_limit],
            }
        )

    dangling_nodes = [
        {"node": node_name, "degree": degree}
        for node_name, degree in sorted(deg.items(), key=lambda pair: (pair[1], pair[0]))
        if degree <= 1
    ][:hint_limit]

    missing_links: List[Dict[str, Any]] = []
    for kind, names in [
        ("compound", entities["compounds"]),
        ("protein", entities["proteins"]),
        ("nucleic_acid", entities["nucleic_acids"]),
        ("element_collection", entities["element_collections"]),
        ("protein_complex", entities["protein_complexes"]),
    ]:
        for name in names:
            node_name = f"{kind}:{name}"
            if deg.get(node_name, 0) == 0:
                missing_links.append(
                    {
                        "node": node_name,
                        "hint": f"{kind} exists but is disconnected from processes/locations",
                    }
                )

    return {
        "meta": meta,
        "n_nodes": len(adj),
        "n_edges": sum(len(v) for v in adj.values()) // 2,
        "n_components": len(comps_sorted),
        "main_component_size": len(comps_sorted[0]) if comps_sorted else 0,
        "orphan_components": orphan_components[: max(1, hint_limit // 5)],
        "dangling_nodes": dangling_nodes,
        "missing_links_suspected": missing_links[:hint_limit],
    }


def run_stage_two_with_feedback_loop(
    input_text: str,
    stage_one: Dict[str, Any],
    chunk_details: Optional[List[Dict[str, Any]]] = None,
    *,
    pathway_context: Optional[Dict[str, Any]] = None,
    qa_rounds: int = 2,
    enable_chunking: bool,
    chunk_word_limit: int = 8000,
    chunk_overlap: int = 1200,
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 10000,
    compact_stage_one: bool = True,
    retry_on_failure: bool = True,
    retry_max_tokens: Optional[int] = None,
    retry_compact_stage_one: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run Stage 2 one or more times, feeding graph-QA hints into later rounds.
    Returns merged additions, flattened per-chunk details, and per-round summaries.
    """
    total_rounds = max(1, int(qa_rounds))
    base_stage_one = deepcopy(stage_one)
    working_stage_one = deepcopy(stage_one)
    all_outputs: List[Dict[str, Any]] = []
    all_chunk_results: List[Dict[str, Any]] = []
    round_summaries: List[Dict[str, Any]] = []

    last_signature = ""
    for round_index in range(1, total_rounds + 1):
        qa_feedback = build_qa_feedback(working_stage_one) if round_index > 1 else {}

        output, chunk_results = run_stage_two_with_chunking(
            input_text,
            working_stage_one,
            chunk_details=chunk_details if round_index == 1 else None,
            pathway_context=pathway_context,
            qa_feedback=qa_feedback,
            enable_chunking=enable_chunking,
            chunk_word_limit=chunk_word_limit,
            chunk_overlap=chunk_overlap,
            max_attempts=max_attempts,
            temperature=temperature,
            max_tokens=max_tokens,
            compact_stage_one=compact_stage_one,
            retry_on_failure=retry_on_failure,
            retry_max_tokens=retry_max_tokens,
            retry_compact_stage_one=retry_compact_stage_one,
        )

        tagged_chunks: List[Dict[str, Any]] = []
        for chunk in chunk_results:
            tagged = dict(chunk)
            tagged["qa_round"] = round_index
            tagged_chunks.append(tagged)
        all_chunk_results.extend(tagged_chunks)

        all_outputs.append(output)
        merged_additions = merge_inference_outputs(all_outputs)
        merged_payload = merge_additions(base_stage_one, merged_additions)
        signature = json.dumps(merged_additions, sort_keys=True)

        round_summaries.append(
            {
                "qa_round": round_index,
                "chunk_count": len(chunk_results),
                "used_feedback": bool(qa_feedback),
                "feedback_missing_links": len(qa_feedback.get("missing_links_suspected", []))
                if isinstance(qa_feedback, dict)
                else 0,
                "feedback_dangling_nodes": len(qa_feedback.get("dangling_nodes", []))
                if isinstance(qa_feedback, dict)
                else 0,
            }
        )

        if signature == last_signature:
            break
        last_signature = signature
        working_stage_one = merged_payload

    return merge_inference_outputs(all_outputs), all_chunk_results, round_summaries


def merge_additions(
    base: Dict[str, Any],
    inference_additions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge Stage-2 additions into a deep copy of the Stage-1 JSON.
    Deduplication is signature-based (JSON string) to avoid exact duplicates.
    """
    merged = deepcopy(base)
    inference_additions = clean_inference_output(inference_additions or {})
    additions = (inference_additions or {}).get("additions", {})

    entities_add = additions.get("entities", {})
    if isinstance(entities_add, dict):
        merged.setdefault("entities", {})
        for key, items in entities_add.items():
            if not isinstance(items, list):
                continue
            merged["entities"].setdefault(key, [])
            _extend_unique(merged["entities"][key], items)

    processes_add = additions.get("processes", {})
    if isinstance(processes_add, dict):
        merged.setdefault("processes", {})
        for key, items in processes_add.items():
            if not isinstance(items, list):
                continue
            merged["processes"].setdefault(key, [])
            if key == "reactions":
                _merge_reactions(merged["processes"][key], items)
            else:
                _extend_unique(merged["processes"][key], items)

    states_add = additions.get("biological_states", [])
    if isinstance(states_add, list):
        merged.setdefault("biological_states", [])
        _extend_unique(merged["biological_states"], states_add)

    locations_add = additions.get("element_locations", {})
    if isinstance(locations_add, dict):
        merged.setdefault("element_locations", {})
        for key, items in locations_add.items():
            if not isinstance(items, list):
                continue
            merged["element_locations"].setdefault(key, [])
            _extend_unique(merged["element_locations"][key], items)

    _inject_name_based_modifiers(merged)

    return merged


def merge_inference_outputs(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple Stage-2 inference outputs into a single additions payload.
    """
    merged_additions: Dict[str, Any] = {}
    intended_effect: Optional[str] = None
    expected_changes: List[str] = []

    for payload in outputs:
        if not isinstance(payload, dict):
            continue

        payload = clean_inference_output(payload)

        additions = payload.get("additions")
        if isinstance(additions, dict):
            _merge_dict_in_place(merged_additions, additions)

        qa_hints = payload.get("qa_hints")
        if isinstance(qa_hints, dict):
            effect = qa_hints.get("intended_effect")
            if not intended_effect and isinstance(effect, str) and effect.strip():
                intended_effect = effect

            changes = qa_hints.get("expected_changes")
            if isinstance(changes, list):
                for item in changes:
                    if isinstance(item, str) and item not in expected_changes:
                        expected_changes.append(item)

    result: Dict[str, Any] = {"additions": merged_additions}
    if intended_effect or expected_changes:
        qa_hints: Dict[str, Any] = {}
        if intended_effect:
            qa_hints["intended_effect"] = intended_effect
        if expected_changes:
            qa_hints["expected_changes"] = expected_changes
        result["qa_hints"] = qa_hints

    return result


def _tag_attempts(attempts: AttemptLogs, phase: str) -> AttemptLogs:
    tagged: AttemptLogs = []
    for entry in attempts:
        tagged_entry = dict(entry)
        tagged_entry["phase"] = phase
        tagged.append(tagged_entry)
    return tagged


def _summarize_failure(attempts: AttemptLogs, preview_chars: int = 500) -> Tuple[str, str, int, str]:
    last_error = "Unknown error"
    raw_preview = ""
    raw_length = 0
    raw_tail = ""
    for entry in reversed(attempts):
        if entry.get("error"):
            last_error = str(entry.get("error") or last_error)
            raw = str(entry.get("raw") or "")
            raw_length = len(raw)
            raw_preview = raw[:preview_chars].replace("\n", " ").strip()
            raw_tail = raw[-preview_chars:].replace("\n", " ").strip()
            break
    return last_error, raw_preview, raw_length, raw_tail


def _extract_json_from_text(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None

    # Remove common code fence markers without dropping JSON content.
    text = text.replace("```json", "```")
    text = text.replace("```", "")

    obj_start = text.find("{")
    if obj_start == -1:
        return None

    obj_end = _find_matching_brace(text, obj_start)
    if obj_end is not None:
        candidate = text[obj_start : obj_end + 1]
        candidate = _strip_trailing_commas(candidate)
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    candidate = text[obj_start:]
    repaired = _auto_close_json(candidate)
    if repaired is not None:
        repaired = _strip_trailing_commas(repaired)
        try:
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    salvaged = _salvage_truncated_json(candidate)
    if salvaged is not None:
        return salvaged

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None

    return None


def _find_matching_brace(text: str, start: int) -> Optional[int]:
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _scan_json_prefix(text: str) -> Tuple[bool, List[str], int]:
    """
    Scan JSON-like text and return:
    (in_string, open_stack, last_safe_index_outside_string)
    """
    stack: List[str] = []
    in_string = False
    escape = False
    last_safe = -1

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "{[":
            stack.append(ch)
            last_safe = i
        elif ch in "}]":
            if stack:
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()
            last_safe = i
        elif not ch.isspace():
            last_safe = i

    return in_string, stack, last_safe


def _find_last_safe_cut(text: str) -> Optional[int]:
    """
    Find a fallback cut position outside strings, preferring commas, then object/array starts.
    """
    in_string = False
    escape = False
    last_comma = -1
    last_open = -1

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == ",":
            last_comma = i
        elif ch in "{[":
            last_open = i

    if last_comma >= 0:
        return last_comma
    if last_open >= 0:
        return last_open + 1
    return None


def _salvage_truncated_json(text: str, max_steps: int = 25) -> Optional[Dict[str, Any]]:
    """
    Repeatedly trim tail to last safe delimiter and try to auto-close/parse.
    Useful when output is truncated mid-field/mid-string.
    """
    working = text
    for _ in range(max_steps):
        repaired = _auto_close_json(working)
        if repaired is not None:
            repaired = _strip_trailing_commas(repaired)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        cut = _find_last_safe_cut(working)
        if cut is None or cut <= 1:
            break
        working = working[:cut]

    return None


def _auto_close_json(text: str) -> Optional[str]:
    in_string, stack, last_safe = _scan_json_prefix(text)

    # If generation cut off inside a string, trim to the last safe position and close braces.
    if in_string:
        if last_safe < 0:
            return None
        text = text[: last_safe + 1]
        text = _strip_trailing_commas(text)
        in_string, stack, _ = _scan_json_prefix(text)
        if in_string:
            return None

    if not stack:
        return text

    closers = {"{": "}", "[": "]"}
    suffix = "".join(closers[ch] for ch in reversed(stack))
    return text + suffix


def _strip_trailing_commas(text: str) -> str:
    previous = None
    cleaned = text
    while cleaned != previous:
        previous = cleaned
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _compact_stage_one_for_inference(stage_one: Dict[str, Any]) -> Dict[str, Any]:
    return _strip_empty_and_evidence(stage_one)


def _strip_empty_and_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "evidence":
                continue
            cleaned = _strip_empty_and_evidence(item)
            if _is_empty_value(cleaned):
                continue
            compact[key] = cleaned
        return compact
    if isinstance(value, list):
        items = [_strip_empty_and_evidence(item) for item in value]
        return [item for item in items if not _is_empty_value(item)]
    return value


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize_name(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for value in values:
        norm = _normalize_name(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(value)
    return out


def _split_composite_token(value: str) -> List[str]:
    text = (value or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*\+\s*|\s+and\s+", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part and part.strip()]


def _clean_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entities, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    entity_keys = [
        "cell_types",
        "species",
        "tissues",
        "subcellular_locations",
        "compounds",
        "element_collections",
        "nucleic_acids",
        "proteins",
        "protein_complexes",
    ]

    for key in entity_keys:
        items = _safe_list(entities.get(key, []))
        cleaned_items: List[Dict[str, Any]] = []
        seen_names: set = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip()
            if not name:
                continue
            norm_name = _normalize_name(name)
            if not norm_name or norm_name in seen_names:
                continue
            seen_names.add(norm_name)
            cleaned_item = {k: v for k, v in item.items() if not _is_empty_value(v)}
            cleaned_item["name"] = name
            for list_key in ("components", "cofactors", "modifications"):
                if isinstance(cleaned_item.get(list_key), list):
                    cleaned_item[list_key] = [
                        v for v in cleaned_item[list_key] if isinstance(v, str) and v.strip()
                    ]
                    if not cleaned_item[list_key]:
                        cleaned_item.pop(list_key, None)
            cleaned_items.append(cleaned_item)

        if cleaned_items:
            cleaned[key] = cleaned_items

    return cleaned


def _clean_biological_states(states: List[Any]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for item in _safe_list(states):
        if not isinstance(item, dict):
            continue
        trimmed: Dict[str, Any] = {}
        for key in ["name", "species", "cell_type", "tissue", "subcellular_location", "evidence"]:
            value = item.get(key)
            if isinstance(value, str):
                value = value.strip()
            if not _is_empty_value(value):
                trimmed[key] = value
        if any(trimmed.get(k) for k in ["name", "species", "cell_type", "tissue", "subcellular_location"]):
            cleaned.append(trimmed)
    return cleaned


def _clean_element_locations(locations: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(locations, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, entity_key in [
        ("compound_locations", "compound"),
        ("element_collection_locations", "element_collection"),
        ("nucleic_acid_locations", "nucleic_acid"),
        ("protein_locations", "protein"),
    ]:
        items: List[Dict[str, Any]] = []
        for item in _safe_list(locations.get(key, [])):
            if not isinstance(item, dict):
                continue
            entity = (item.get(entity_key) or "").strip()
            if not entity:
                continue
            entry: Dict[str, Any] = {entity_key: entity}
            biological_state = (item.get("biological_state") or "").strip()
            if biological_state:
                entry["biological_state"] = biological_state
            evidence = (item.get("evidence") or "").strip()
            if evidence:
                entry["evidence"] = evidence
            items.append(entry)
        if items:
            cleaned[key] = items
    return cleaned


def _clean_enzymes(enzymes: Any) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for item in _safe_list(enzymes):
        if not isinstance(item, dict):
            continue
        entry: Dict[str, Any] = {}
        protein_complex = (item.get("protein_complex") or "").strip()
        if protein_complex:
            entry["protein_complex"] = protein_complex
        evidence = (item.get("evidence") or "").strip()
        if evidence:
            entry["evidence"] = evidence
        inference = item.get("inference")
        if inference and not _is_empty_value(inference):
            entry["inference"] = inference
        if entry:
            cleaned.append(entry)
    return cleaned


def _clean_elements_with_states(items: Any) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for item in _safe_list(items):
        if not isinstance(item, dict):
            continue
        element = (item.get("element") or "").strip()
        if not element:
            continue
        entry: Dict[str, Any] = {"element": element}
        side = (item.get("side") or "").strip()
        if side:
            entry["side"] = side
        biological_state = (item.get("biological_state") or "").strip()
        if biological_state:
            entry["biological_state"] = biological_state
        evidence = (item.get("evidence") or "").strip()
        if evidence:
            entry["evidence"] = evidence
        cleaned.append(entry)
    return cleaned


def _clean_processes(processes: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(processes, dict):
        return {}
    cleaned: Dict[str, Any] = {}

    reactions_out: List[Dict[str, Any]] = []
    for item in _safe_list(processes.get("reactions", [])):
        if not isinstance(item, dict):
            continue
        inputs: List[str] = []
        for value in _safe_list(item.get("inputs")):
            if not isinstance(value, str) or not value.strip():
                continue
            expanded = _split_composite_token(value)
            inputs.extend(expanded or [value.strip()])
        outputs: List[str] = []
        for value in _safe_list(item.get("outputs")):
            if not isinstance(value, str) or not value.strip():
                continue
            expanded = _split_composite_token(value)
            outputs.extend(expanded or [value.strip()])
        inputs = _dedupe_preserve_order(inputs)
        outputs = _dedupe_preserve_order(outputs)
        if not inputs or not outputs:
            continue
        entry: Dict[str, Any] = {"inputs": inputs, "outputs": outputs}
        name = (item.get("name") or "").strip()
        if name:
            entry["name"] = name
        enzymes = _clean_enzymes(item.get("enzymes"))
        if enzymes:
            entry["enzymes"] = enzymes
        biological_state = (item.get("biological_state") or "").strip()
        if biological_state:
            entry["biological_state"] = biological_state
        evidence = (item.get("evidence") or "").strip()
        if evidence:
            entry["evidence"] = evidence
        inference = item.get("inference")
        if inference and not _is_empty_value(inference):
            entry["inference"] = inference
        reactions_out.append(entry)

    if reactions_out:
        cleaned["reactions"] = reactions_out

    transports_out: List[Dict[str, Any]] = []
    for item in _safe_list(processes.get("transports", [])):
        if not isinstance(item, dict):
            continue
        entry: Dict[str, Any] = {}
        name = (item.get("name") or "").strip()
        if name:
            entry["name"] = name
        cargo = (item.get("cargo") or "").strip()
        if cargo:
            entry["cargo"] = cargo
        from_state = (item.get("from_biological_state") or "").strip()
        if from_state:
            entry["from_biological_state"] = from_state
        to_state = (item.get("to_biological_state") or "").strip()
        if to_state:
            entry["to_biological_state"] = to_state
        transporters = _clean_enzymes(item.get("transporters"))
        if transporters:
            entry["transporters"] = transporters
        elements = _clean_elements_with_states(item.get("elements_with_states"))
        if elements:
            entry["elements_with_states"] = elements
        evidence = (item.get("evidence") or "").strip()
        if evidence:
            entry["evidence"] = evidence
        inference = item.get("inference")
        if inference and not _is_empty_value(inference):
            entry["inference"] = inference
        if any(k in entry for k in ["cargo", "transporters", "elements_with_states"]):
            transports_out.append(entry)

    if transports_out:
        cleaned["transports"] = transports_out

    rct_out: List[Dict[str, Any]] = []
    for item in _safe_list(processes.get("reaction_coupled_transports", [])):
        if not isinstance(item, dict):
            continue
        entry: Dict[str, Any] = {}
        name = (item.get("name") or "").strip()
        if name:
            entry["name"] = name
        reaction = (item.get("reaction") or "").strip()
        if reaction:
            entry["reaction"] = reaction
        transport = (item.get("transport") or "").strip()
        if transport:
            entry["transport"] = transport
        enzymes = _clean_enzymes(item.get("enzymes"))
        if enzymes:
            entry["enzymes"] = enzymes
        elements = _clean_elements_with_states(item.get("elements_with_states"))
        if elements:
            entry["elements_with_states"] = elements
        evidence = (item.get("evidence") or "").strip()
        if evidence:
            entry["evidence"] = evidence
        inference = item.get("inference")
        if inference and not _is_empty_value(inference):
            entry["inference"] = inference
        if any(k in entry for k in ["reaction", "transport", "elements_with_states"]):
            rct_out.append(entry)

    if rct_out:
        cleaned["reaction_coupled_transports"] = rct_out

    interactions_out: List[Dict[str, Any]] = []
    for item in _safe_list(processes.get("interactions", [])):
        if not isinstance(item, dict):
            continue
        e1 = (item.get("entity_1") or "").strip()
        e2 = (item.get("entity_2") or "").strip()
        if not e1 or not e2:
            continue
        entry: Dict[str, Any] = {"entity_1": e1, "entity_2": e2}
        name = (item.get("name") or "").strip()
        if name:
            entry["name"] = name
        relationship = (item.get("relationship") or "").strip()
        if relationship:
            entry["relationship"] = relationship
        biological_state = (item.get("biological_state") or "").strip()
        if biological_state:
            entry["biological_state"] = biological_state
        evidence = (item.get("evidence") or "").strip()
        if evidence:
            entry["evidence"] = evidence
        inference = item.get("inference")
        if inference and not _is_empty_value(inference):
            entry["inference"] = inference
        interactions_out.append(entry)

    if interactions_out:
        cleaned["interactions"] = interactions_out

    return cleaned


def clean_stage_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cleaned: Dict[str, Any] = {}

    entities = _clean_entities(payload.get("entities", {}))
    if entities:
        cleaned["entities"] = entities

    biological_states = _clean_biological_states(_safe_list(payload.get("biological_states", [])))
    if biological_states:
        cleaned["biological_states"] = biological_states

    element_locations = _clean_element_locations(payload.get("element_locations", {}))
    if element_locations:
        cleaned["element_locations"] = element_locations

    processes = _clean_processes(payload.get("processes", {}))
    if processes:
        cleaned["processes"] = processes

    return cleaned


def clean_inference_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"additions": {}}

    additions: Any = payload.get("additions")
    if additions is None:
        additions = {}
        if isinstance(payload.get("entities"), dict):
            additions["entities"] = payload.get("entities")
        if isinstance(payload.get("processes"), dict):
            additions["processes"] = payload.get("processes")
        if isinstance(payload.get("biological_states"), list):
            additions["biological_states"] = payload.get("biological_states")
        if isinstance(payload.get("element_locations"), dict):
            additions["element_locations"] = payload.get("element_locations")

    if not isinstance(additions, dict):
        additions = {}

    entities = additions.get("entities")
    if isinstance(entities, dict) and "processes" in entities:
        misplaced = entities.pop("processes")
        if isinstance(misplaced, dict):
            additions.setdefault("processes", {})
            if isinstance(additions.get("processes"), dict):
                _merge_dict_in_place(additions["processes"], misplaced)

    cleaned_additions: Dict[str, Any] = {}
    cleaned_entities = _clean_entities(additions.get("entities", {}))
    if cleaned_entities:
        cleaned_additions["entities"] = cleaned_entities
    cleaned_processes = _clean_processes(additions.get("processes", {}))
    if cleaned_processes:
        cleaned_additions["processes"] = cleaned_processes
    cleaned_states = _clean_biological_states(_safe_list(additions.get("biological_states", [])))
    if cleaned_states:
        cleaned_additions["biological_states"] = cleaned_states
    cleaned_locations = _clean_element_locations(additions.get("element_locations", {}))
    if cleaned_locations:
        cleaned_additions["element_locations"] = cleaned_locations

    result: Dict[str, Any] = {"additions": cleaned_additions}
    qa_hints = payload.get("qa_hints")
    if isinstance(qa_hints, dict):
        qa_clean: Dict[str, Any] = {}
        intended = qa_hints.get("intended_effect")
        if isinstance(intended, str) and intended.strip():
            qa_clean["intended_effect"] = intended.strip()
        changes = qa_hints.get("expected_changes")
        if isinstance(changes, list):
            cleaned_changes = [c for c in changes if isinstance(c, str) and c.strip()]
            if cleaned_changes:
                qa_clean["expected_changes"] = cleaned_changes
        if qa_clean:
            result["qa_hints"] = qa_clean

    return result


def merge_stage_one_outputs(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple Stage-1 extraction payloads (e.g., chunked runs) into a single dict.
    """
    merged: Dict[str, Any] = {}
    for payload in outputs:
        if isinstance(payload, dict):
            _merge_dict_in_place(merged, payload)
    return merged


def run_stage_one_with_chunking(
    input_text: str,
    *,
    pathway_context: Optional[Dict[str, Any]] = None,
    enable_chunking: bool,
    chunk_word_limit: int = 8000,
    chunk_overlap: int = 1200,
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 12000,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optionally chunk the input text before running Stage 1 extraction. Returns the merged JSON
    plus per-chunk details (inputs, outputs, attempts) for inspection.
    """
    words = input_text.split()
    use_chunks = enable_chunking and len(words) > chunk_word_limit

    if not use_chunks:
        output, attempts = run_extraction_pipeline(
            input_text,
            pathway_context=pathway_context,
            max_attempts=max_attempts,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output = clean_stage_one(output)
        chunk_meta = {
            "chunk_id": 1,
            "start_word": 0,
            "end_word": len(words),
            "text": input_text,
            "output": output,
            "attempts": attempts,
        }
        return output, [chunk_meta]

    chunks = chunk_text(input_text, chunk_word_limit, chunk_overlap)
    chunk_results: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []

    for chunk in chunks:
        try:
            parsed, attempts = run_extraction_pipeline(
                chunk["text"],
                pathway_context=pathway_context,
                max_attempts=max_attempts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except PipelineFailure as failure:
            raise PipelineFailure(
                stage=f"extraction chunk {chunk['chunk_id']}",
                message=f"Chunk {chunk['chunk_id']} failed to produce valid JSON.",
                attempts=failure.attempts,
            ) from failure

        parsed = clean_stage_one(parsed)
        chunk_entry = {**chunk, "output": parsed, "attempts": attempts}
        chunk_results.append(chunk_entry)
        outputs.append(parsed)

    merged = merge_stage_one_outputs(outputs)
    merged = clean_stage_one(merged)
    return merged, chunk_results


def _inject_name_based_modifiers(merged: Dict[str, Any]) -> None:
    """
    Post-processing pass: for every protein and protein_complex in entities, check
    whether the name appears in a reaction name/evidence or a transport name/evidence.
    - Reactions: inject as a catalyst modifier if missing.
    - Transports: inject as a transporter entry (protein_complex field) if missing.
    Catches cases where Stage-1 and Stage-2 omit these links.
    """
    def _sl(x: Any) -> list:
        return x if isinstance(x, list) else []

    entities = merged.get("entities") or {}

    # Build actor list: (name, entity_type) for proteins and protein_complexes.
    # Check complexes first so they take priority over subunits with overlapping names.
    actors: List[tuple] = []
    for row in _sl(entities.get("protein_complexes", [])):
        if isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"].strip():
            actors.append((row["name"].strip(), "protein_complex"))
    for row in _sl(entities.get("proteins", [])):
        if isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"].strip():
            actors.append((row["name"].strip(), "protein"))

    processes = merged.get("processes") or {}
    reactions = _sl(processes.get("reactions", []))
    transports = _sl(processes.get("transports", []))

    for pname, entity_type in actors:
        pname_lower = pname.lower()

        # --- Reactions: inject missing catalyst modifiers ---
        for reaction in reactions:
            if not isinstance(reaction, dict):
                continue
            rname = (reaction.get("name") or "").lower()
            revidence = (reaction.get("evidence") or "").lower()
            if pname_lower not in rname and pname_lower not in revidence:
                continue
            existing_modifiers = _sl(reaction.get("modifiers", []))
            already_present = any(
                isinstance(m, dict) and (m.get("entity") or "").strip().lower() == pname_lower
                for m in existing_modifiers
            )
            if already_present:
                continue
            reaction.setdefault("modifiers", [])
            reaction["modifiers"].append({
                "entity": pname,
                "entity_type": entity_type,
                "role": "catalyst",
                "evidence": (reaction.get("evidence") or "")[:120],
                "confidence": 0.9,
                "provenance": "inferred",
                "source_refs": [(reaction.get("evidence") or "")[:120]],
            })

        # --- Transports: inject missing transporter protein_complex entries ---
        for transport in transports:
            if not isinstance(transport, dict):
                continue
            tname = (transport.get("name") or "").lower()
            tevidence = (transport.get("evidence") or "").lower()
            if pname_lower not in tname and pname_lower not in tevidence:
                continue
            existing_transporters = _sl(transport.get("transporters", []))
            already_present = any(
                isinstance(t, dict) and (
                    (t.get("protein_complex") or "").strip().lower() == pname_lower
                    or (t.get("protein") or "").strip().lower() == pname_lower
                )
                for t in existing_transporters
            )
            if already_present:
                continue
            # Patch the first transporter entry that is missing a protein_complex,
            # or append a new one if all existing entries already have one.
            patched = False
            for t in existing_transporters:
                if isinstance(t, dict) and not t.get("protein_complex") and not t.get("protein"):
                    t["protein_complex"] = pname
                    patched = True
                    break
            if not patched:
                transport.setdefault("transporters", [])
                transport["transporters"].append({
                    "protein_complex": pname,
                    "evidence": (transport.get("evidence") or "")[:120],
                    "confidence": 0.9,
                    "provenance": "inferred",
                    "source_refs": [(transport.get("evidence") or "")[:120]],
                })


def _reaction_io_key(r: Any) -> frozenset:
    """Fingerprint a reaction by its sorted inputs+outputs for deduplication."""
    if not isinstance(r, dict):
        return frozenset()
    inputs = sorted(str(x).strip().lower() for x in (r.get("inputs") or []) if x)
    outputs = sorted(str(x).strip().lower() for x in (r.get("outputs") or []) if x)
    return frozenset([("inputs", tuple(inputs)), ("outputs", tuple(outputs))])


def _merge_reactions(target: List[Any], new_items: List[Any]) -> None:
    """
    Merge Stage-2 reaction additions into target.
    - If a new reaction's inputs+outputs match an existing reaction, patch its
      modifiers[] with any new entries (avoiding duplicates) instead of appending
      a duplicate reaction.
    - If no matching reaction exists, append it normally.
    """
    target_keys = {_reaction_io_key(r): i for i, r in enumerate(target)}
    seen_signatures = {json.dumps(r, sort_keys=True) for r in target}

    for new_r in new_items:
        if not isinstance(new_r, dict):
            continue
        key = _reaction_io_key(new_r)
        if key and key in target_keys:
            # Patch modifiers into the existing reaction
            existing = target[target_keys[key]]
            new_modifiers = new_r.get("modifiers")
            if isinstance(new_modifiers, list) and new_modifiers:
                existing.setdefault("modifiers", [])
                _extend_unique(existing["modifiers"], new_modifiers)
        else:
            # Genuinely new reaction — append if not an exact duplicate
            sig = json.dumps(new_r, sort_keys=True)
            if sig not in seen_signatures:
                target.append(new_r)
                seen_signatures.add(sig)
                if key:
                    target_keys[key] = len(target) - 1


def _extend_unique(target: List[Any], new_items: List[Any]) -> None:
    """
    Append only novel entries into target, using JSON serialization as a dedupe signature.
    """
    seen = {json.dumps(item, sort_keys=True) for item in target}
    for item in new_items:
        try:
            signature = json.dumps(item, sort_keys=True)
        except TypeError:
            # Fall back to repr when item contains non-serializable objects (unlikely for LLM output)
            signature = repr(item)
        if signature in seen:
            continue
        target.append(item)
        seen.add(signature)


# ---------------------------------------------------------------------------
# Section-aware chunking helpers
# ---------------------------------------------------------------------------

# Matches a line that is solely a recognised academic section header, with an
# optional leading numeric prefix (e.g. "2.", "3.1 ").
_SECTION_HEADER_RE = re.compile(
    r'^[ \t]*(?:\d[\d.]*\.?\s+)?'
    r'(abstract|introduction|background|'
    r'materials?\s+and\s+methods?|methods?\s+and\s+materials?|'
    r'experimental\s+procedures?|methods?|'
    r'results?(?:\s+and\s+discussion)?|'
    r'discussion(?:\s+and\s+conclusions?)?|'
    r'conclusions?|summary|'
    r'supplementary(?:\s+\w+)*|supplemental(?:\s+\w+)*|supporting\s+information|'
    r'references?|bibliography|'
    r'acknowledgements?|acknowledgments?)'
    r'[ \t]*$',
    re.IGNORECASE,
)

_SECTION_RELEVANCE_MAP: Dict[str, float] = {
    "results": 0.9,
    "results and discussion": 0.9,
    "discussion": 0.85,
    "discussion and conclusions": 0.85,
    "conclusion": 0.75,
    "summary": 0.75,
    "methods": 0.7,
    "abstract": 0.6,
    "preamble": 0.5,
    "supplementary": 0.5,
    "supporting information": 0.5,
    "introduction": 0.4,
    "background": 0.4,
    "references": 0.1,
    "bibliography": 0.1,
    "acknowledgements": 0.05,
    "acknowledgments": 0.05,
}


def _get_section_relevance(section_label: str) -> float:
    """Return a relevance score 0–1 for a normalised section label."""
    name = section_label.strip().lower()
    if name in _SECTION_RELEVANCE_MAP:
        return _SECTION_RELEVANCE_MAP[name]
    for key, score in _SECTION_RELEVANCE_MAP.items():
        if key in name or name.startswith(key):
            return score
    return 0.5


def _normalize_section_label(raw: str) -> str:
    name = raw.strip().lower()
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'^\d[\d.]*\.?\s+', '', name).strip()
    return name


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split *text* into (section_label, section_text) pairs.
    Text that precedes the first recognised header is labelled 'preamble'.
    Figure captions stay in their parent section naturally because only
    section-header lines trigger a split.
    """
    sections: List[Tuple[str, str]] = []
    current_label = "preamble"
    current_lines: List[str] = []

    for line in text.split('\n'):
        m = _SECTION_HEADER_RE.match(line)
        if m:
            body = '\n'.join(current_lines).strip()
            if body:
                sections.append((current_label, body))
            current_label = _normalize_section_label(m.group(1))
            current_lines = []
        else:
            current_lines.append(line)

    body = '\n'.join(current_lines).strip()
    if body:
        sections.append((current_label, body))

    return sections if sections else [("unknown", text.strip())]


def _split_sentences(text: str) -> List[str]:
    """Split text at sentence boundaries (.!? followed by whitespace + capital)."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(\[])', text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_section_text(
    section_text: str,
    section_label: str,
    relevance_score: float,
    chunk_word_limit: int,
    overlap_words: int,
    start_chunk_id: int,
    start_word_offset: int,
) -> List[Dict[str, Any]]:
    """Chunk one section into sentence-boundary-respecting pieces."""
    sentences = _split_sentences(section_text)
    if not sentences:
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_id = start_chunk_id
    sent_idx = 0
    word_offset = start_word_offset

    while sent_idx < len(sentences):
        accumulated_words = 0
        chunk_sents: List[str] = []
        idx = sent_idx

        while idx < len(sentences):
            sent_words = len(sentences[idx].split())
            if accumulated_words + sent_words > chunk_word_limit and chunk_sents:
                break
            chunk_sents.append(sentences[idx])
            accumulated_words += sent_words
            idx += 1

        if not chunk_sents:
            # Single sentence exceeds the limit — include it whole.
            chunk_sents = [sentences[sent_idx]]
            accumulated_words = len(sentences[sent_idx].split())
            idx = sent_idx + 1

        chunk_str = " ".join(chunk_sents)
        wc = len(chunk_str.split())
        chunks.append(
            {
                "chunk_id": chunk_id,
                "section": section_label,
                "relevance_score": relevance_score,
                "text": chunk_str,
                "word_count": wc,
                "start_word": word_offset,
                "end_word": word_offset + wc,
            }
        )
        chunk_id += 1
        word_offset += wc

        if idx >= len(sentences):
            break

        # Walk backwards from idx to cover ~overlap_words for the next chunk.
        overlap_accumulated = 0
        new_sent_idx = idx
        for back in range(idx - 1, sent_idx - 1, -1):
            w = len(sentences[back].split())
            if overlap_accumulated + w > overlap_words:
                break
            overlap_accumulated += w
            new_sent_idx = back

        # Always advance by at least one sentence to avoid infinite loops.
        sent_idx = max(new_sent_idx, sent_idx + 1)

    return chunks


def chunk_text(text: str, chunk_word_limit: int, overlap_words: int) -> List[Dict[str, Any]]:
    """
    Split *text* into section-aware, sentence-boundary-respecting chunks.

    Each chunk dict contains:
      chunk_id        — processing order (high-relevance sections first)
      section         — normalised section label ("results", "methods", …)
      relevance_score — float 0–1; higher = more biologically relevant
      text            — chunk text (never split mid-sentence)
      word_count      — approximate word count
      start_word      — approximate word offset in the original text
      end_word        — approximate end word offset

    Ordering: Results/Discussion chunks come first; References/
    Acknowledgements chunks come last.  chunk_word_limit and overlap_words
    are honoured within each section.
    """
    sections = _split_into_sections(text)

    all_chunks: List[Dict[str, Any]] = []
    word_offset = 0
    tmp_id = 1

    for section_label, section_text in sections:
        relevance = _get_section_relevance(section_label)
        sec_chunks = _chunk_section_text(
            section_text,
            section_label,
            relevance,
            max(int(chunk_word_limit), 1),
            max(0, int(overlap_words)),
            start_chunk_id=tmp_id,
            start_word_offset=word_offset,
        )
        all_chunks.extend(sec_chunks)
        tmp_id += len(sec_chunks)
        word_offset += len(section_text.split())

    if not all_chunks:
        words = text.split()
        return [
            {
                "chunk_id": 1,
                "section": "unknown",
                "relevance_score": 0.5,
                "text": text,
                "word_count": len(words),
                "start_word": 0,
                "end_word": len(words),
            }
        ]

    # Sort: high-relevance first; within the same score, preserve original order.
    all_chunks.sort(key=lambda c: (-c["relevance_score"], c["chunk_id"]))

    # Re-number chunk_ids in processing order.
    for new_id, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = new_id

    return all_chunks


def _merge_dict_in_place(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            dest = target.get(key)
            if not isinstance(dest, dict):
                target[key] = deepcopy(value)
            else:
                _merge_dict_in_place(dest, value)
        elif isinstance(value, list):
            dest_list = target.setdefault(key, [])
            _extend_unique(dest_list, value)
        else:
            if key not in target or target[key] in ("", None):
                target[key] = value


def _run_json_stage(
    *,
    stage_name: str,
    system_prompt: str,
    build_user_prompt: Callable[[Optional[str], Optional[str]], str],
    max_attempts: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[Dict[str, Any], AttemptLogs]:
    attempts: AttemptLogs = []
    prev_output: Optional[str] = None
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        user_prompt = build_user_prompt(prev_output, last_error)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = chat(messages, temperature=temperature, max_tokens=max_tokens, response_json=True)
        log_entry: AttemptLog = {"attempt": attempt, "raw": raw, "error": None}

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise json.JSONDecodeError(
                    f"Expected JSON object, got {type(parsed).__name__}", raw, 0
                )
            attempts.append(log_entry)
            return parsed, attempts
        except json.JSONDecodeError as exc:
            extracted = _extract_json_from_text(raw)
            if extracted is not None:
                log_entry["note"] = "salvaged_json"
                attempts.append(log_entry)
                return extracted, attempts
            error_msg = f"{exc.__class__.__name__}: {exc}"
            log_entry["error"] = error_msg
            attempts.append(log_entry)
            prev_output = raw
            last_error = error_msg

    raise PipelineFailure(
        stage_name,
        f"{stage_name.title()} stage failed to produce valid JSON after {max_attempts} attempts.",
        attempts,
    )


def _build_extraction_prompt(
    input_text: str,
    prev_output: Optional[str],
    last_error: Optional[str],
    *,
    pathway_context: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = []

    context_header = format_context_header(pathway_context)
    if context_header:
        prompt.extend([context_header, ""])

    prompt.extend(
        [
            "Extract PWML-structured JSON strictly according to the schema.",
            "Return ONLY the JSON object.",
            "Pathway description:",
            "<<<",
            input_text.strip(),
            ">>>",
        ]
    )

    if prev_output and last_error:
        prompt.extend(
            [
                "",
                "Your previous attempt returned invalid JSON.",
                f"Parse error: {last_error}",
                "Here is the invalid output. Fix it while keeping evidence quotes verbatim and following all instructions.",
                "<<<",
                prev_output,
                ">>>",
            ]
        )

    return "\n".join(prompt)


def _build_inference_prompt(
    input_text: str,
    stage_one_json: str,
    prev_output: Optional[str],
    last_error: Optional[str],
    qa_feedback: Optional[Dict[str, Any]],
    *,
    pathway_context: Optional[Dict[str, Any]] = None,
    chunk_section: Optional[str] = None,
    chunk_relevance_score: Optional[float] = None,
) -> str:
    prompt = []

    context_header = format_context_header(pathway_context)
    if context_header:
        prompt.extend([context_header, ""])

    if chunk_section or chunk_relevance_score is not None:
        section_str = chunk_section or "unknown"
        score_str = f"{chunk_relevance_score:.2f}" if chunk_relevance_score is not None else "n/a"
        prompt.extend([f"CHUNK CONTEXT: Section={section_str}, Relevance={score_str}", ""])

    prompt.extend(
        [
            "Use the original description and Stage-1 strict JSON to propose conservative PWML additions.",
            "Return ONLY the additions JSON per the inference schema.",
            "",
            "Original description:",
            "<<<",
            input_text.strip(),
            ">>>",
            "",
            "Stage-1 JSON:",
            "<<<",
            stage_one_json,
            ">>>",
        ]
    )

    if qa_feedback:
        qa_json = json.dumps(qa_feedback, indent=2, ensure_ascii=False)
        prompt.extend(
            [
                "",
                "Graph QA feedback (use only as repair hints, stay conservative):",
                "<<<",
                qa_json,
                ">>>",
                "Prioritize reconnecting disconnected entities by adding supported reactions, locations, or state links.",
            ]
        )

    if prev_output and last_error:
        prompt.extend(
            [
                "",
                "Your previous inference output was invalid JSON.",
                f"Parse error: {last_error}",
                "Invalid output (revise into valid JSON without commentary):",
                "<<<",
                prev_output,
                ">>>",
            ]
        )

    return "\n".join(prompt)
