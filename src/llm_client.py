import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError, AuthenticationError, BadRequestError

# -----------------------------------------------------------------------------
# Load .env reliably (project root = folder containing "src")
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    if not model:
        raise RuntimeError("Missing OPENROUTER_MODEL in .env")
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
# Chat function (NOW supports response_json=True)
# -----------------------------------------------------------------------------
def chat(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 800,
    response_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    response_json=True:
      - If json_schema is provided, requests structured JSON using the schema.
      - Otherwise requests a JSON object output.
    """

    kwargs: Dict[str, Any] = {
        "model": _model,
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

    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(**kwargs)
            time.sleep(spacing)
            return (resp.choices[0].message.content or "").strip()

        except AuthenticationError as e:
            raise RuntimeError(
                f"Authentication failed (401). Key starts with: {_mask(os.getenv('OPENROUTER_API_KEY',''))}. "
                f"Checked .env at: {ENV_PATH}. Original error: {e}"
            ) from e

        except BadRequestError as e:
            # If response_format isn't supported by a specific OpenRouter model, you'll see this.
            raise RuntimeError(
                "Bad request (400). This model/provider may not support response_format JSON. "
                "Try OPENROUTER_MODEL=openrouter/free or a different model. "
                f"Original error: {e}"
            ) from e

        except RateLimitError as e:
            last_err = e
            time.sleep(min(base_sleep * (2 ** attempt), max_sleep))

        except (APITimeoutError, APIError) as e:
            last_err = e
            time.sleep(min(base_sleep * (2 ** attempt), max_sleep))

        except json.JSONDecodeError as e:
            # OpenAI SDK fails to parse a malformed HTTP response (e.g. local server
            # returned truncated or non-JSON body). Treat as transient and retry.
            last_err = e
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

    working_messages: List[Dict[str, Any]] = list(messages)

    def _call_once(msgs: List[Dict[str, Any]], include_tools: bool) -> Any:
        kwargs: Dict[str, Any] = {
            "model": _model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if include_tools and tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = _client.chat.completions.create(**kwargs)
                time.sleep(spacing)
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