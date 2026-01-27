import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_client import chat

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
EXTRACT_SYSTEM = (PROMPTS_DIR / "pwml_system.txt").read_text(encoding="utf-8")
INFER_SYSTEM = (PROMPTS_DIR / "pwml_infer_system.txt").read_text(encoding="utf-8")


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
            attempts.append(log_entry)
            return parsed, attempts
        except json.JSONDecodeError as exc:
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
