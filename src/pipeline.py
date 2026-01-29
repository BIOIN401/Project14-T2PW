import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_client import chat

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
EXTRACT_SYSTEM = (PROMPTS_DIR / "pwml_system.txt").read_text(encoding="utf-8")
INFER_SYSTEM = (PROMPTS_DIR / "pwml_infer_system.txt").read_text(encoding="utf-8")
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
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> Tuple[Dict[str, Any], AttemptLogs]:
    """
    Stage 1: strict extraction. Automatically retries with self-repair instructions if JSON parsing fails.
    """
    return _run_json_stage(
        stage_name="extraction",
        system_prompt=EXTRACT_SYSTEM,
        build_user_prompt=lambda prev_output, last_error: _build_extraction_prompt(
            input_text, prev_output, last_error
        ),
        max_attempts=max_attempts,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_inference_pipeline(
    input_text: str,
    stage_one: Dict[str, Any],
    *,
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> Tuple[Dict[str, Any], AttemptLogs]:
    """
    Stage 2: inference/enrichment pass. Uses Stage-1 output as context and retries if JSON is invalid.
    """
    stage_one_str = json.dumps(stage_one, indent=2, ensure_ascii=False)
    return _run_json_stage(
        stage_name="inference",
        system_prompt=INFER_SYSTEM,
        build_user_prompt=lambda prev_output, last_error: _build_inference_prompt(
            input_text, stage_one_str, prev_output, last_error
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
    enable_chunking: bool,
    chunk_word_limit: int = 1500,
    chunk_overlap: int = 200,
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 2000,
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

        try:
            parsed, attempts = run_inference_pipeline(
                chunk["text"],
                chunk_stage_one,
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
                    else max(200, int(max_tokens * 0.6))
                )
                try:
                    retry_parsed, retry_attempts = run_inference_pipeline(
                        chunk["text"],
                        compact_stage_one,
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
                    last_error, raw_preview, raw_length = _summarize_failure(attempts)
                    message = (
                        f"Chunk {chunk.get('chunk_id')} failed to produce valid JSON after retry. "
                        f"Last error: {last_error}. Raw length: {raw_length}. Raw preview: {raw_preview}"
                    )
                    raise PipelineFailure(
                        stage=f"inference chunk {chunk.get('chunk_id')}",
                        message=message,
                        attempts=attempts,
                    ) from retry_failure
            else:
                last_error, raw_preview, raw_length = _summarize_failure(failure.attempts)
                message = (
                    f"Chunk {chunk.get('chunk_id')} failed to produce valid JSON. "
                    f"Last error: {last_error}. Raw length: {raw_length}. Raw preview: {raw_preview}"
                )
                raise PipelineFailure(
                    stage=f"inference chunk {chunk.get('chunk_id')}",
                    message=message,
                    attempts=failure.attempts,
                ) from failure

        chunk_entry = {**chunk, "stage_one": chunk_stage_one, "output": parsed, "attempts": attempts}
        chunk_results.append(chunk_entry)
        outputs.append(parsed)

    merged = merge_inference_outputs(outputs)
    return merged, chunk_results


def merge_additions(
    base: Dict[str, Any],
    inference_additions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge Stage-2 additions into a deep copy of the Stage-1 JSON.
    Deduplication is signature-based (JSON string) to avoid exact duplicates.
    """
    merged = deepcopy(base)
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
            _extend_unique(merged["processes"][key], items)

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


def _summarize_failure(attempts: AttemptLogs, preview_chars: int = 500) -> Tuple[str, str, int]:
    last_error = "Unknown error"
    raw_preview = ""
    raw_length = 0
    for entry in reversed(attempts):
        if entry.get("error"):
            last_error = str(entry.get("error") or last_error)
            raw = str(entry.get("raw") or "")
            raw_length = len(raw)
            raw_preview = raw[:preview_chars].replace("\n", " ").strip()
            break
    return last_error, raw_preview, raw_length


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
    if obj_end is None:
        return None

    candidate = text[obj_start : obj_end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

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
    enable_chunking: bool,
    chunk_word_limit: int = 1500,
    chunk_overlap: int = 200,
    max_attempts: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 2000,
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
            max_attempts=max_attempts,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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

        chunk_entry = {**chunk, "output": parsed, "attempts": attempts}
        chunk_results.append(chunk_entry)
        outputs.append(parsed)

    merged = merge_stage_one_outputs(outputs)
    return merged, chunk_results


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


def chunk_text(text: str, chunk_word_limit: int, overlap_words: int) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks measured in approximate word counts.
    """
    words = text.split()
    if not words:
        return []

    chunk_size = max(int(chunk_word_limit), 1)
    overlap = max(0, min(int(overlap_words), chunk_size - 1))

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_id = 1

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_value = " ".join(chunk_words)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_word": start,
                "end_word": end,
                "text": chunk_text_value,
            }
        )
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
        chunk_id += 1

    return chunks


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

        raw = chat(messages, temperature=temperature, max_tokens=max_tokens)
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
) -> str:
    prompt = [
        "Extract PWML-structured JSON strictly according to the schema.",
        "Return ONLY the JSON object.",
        "Pathway description:",
        "<<<",
        input_text.strip(),
        ">>>",
    ]

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
) -> str:
    prompt = [
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
