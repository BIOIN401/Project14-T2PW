from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

from t2pw.curation.audit_json_llm import audit_payload  # noqa: E402


def test_audit_payload_returns_report_and_patch_without_llm() -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "ATP + ADP"}],
            "proteins": [{"name": "Kinase"}],
        },
        "processes": {"reactions": []},
    }

    result = audit_payload(payload, use_llm=False)

    assert result["report"]["llm"]["enabled"] is False
    assert result["report"]["summary"]["patch_count"] == len(result["patch"])
    assert any(op["path"] == "/entities/compounds/0" for op in result["patch"])
