from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_: None)
            )

    class _FakeOpenAIError(Exception):
        pass

    fake_openai.OpenAI = _FakeOpenAI
    fake_openai.RateLimitError = _FakeOpenAIError
    fake_openai.APIError = _FakeOpenAIError
    fake_openai.APITimeoutError = _FakeOpenAIError
    fake_openai.AuthenticationError = _FakeOpenAIError
    fake_openai.BadRequestError = _FakeOpenAIError
    sys.modules["openai"] = fake_openai

from t2pw.pipeline.pipeline import filter_unresolvable_reactions  # noqa: E402


def test_filter_unresolvable_reactions_removes_ghosts_and_keeps_partial_matches() -> None:
    payload = {
        "entities": {
            "compounds": [
                {"name": "hexadecatrienoic acid (16:3)"},
                {"name": "OPC-8:0-CoA"},
                {"name": "OPC-6:0-CoA"},
            ],
            "proteins": [{"name": "12-oxophytodienoate reductase"}],
            "protein_complexes": [
                {
                    "name": "hexadecatrienoic acid (16:3)",
                    "components": ["hexadecatrienoic acid (16", "3)"],
                },
                {
                    "name": "valid enzyme complex",
                    "components": ["12-oxophytodienoate reductase"],
                },
            ],
            "nucleic_acids": [],
            "element_collections": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "good reaction",
                    "inputs": ["OPC-8:0-CoA", "ATP"],
                    "outputs": ["OPC-6:0-CoA", "ADP"],
                },
                {
                    "name": "bad reaction",
                    "inputs": [],
                    "outputs": ["some product"],
                },
                {
                    "name": "ghost reaction",
                    "inputs": ["nonexistent_fragment_A"],
                    "outputs": ["nonexistent_fragment_B"],
                },
            ]
        },
    }

    filtered, removed = filter_unresolvable_reactions(payload)

    reactions = filtered["processes"]["reactions"]
    reaction_names = [reaction["name"] for reaction in reactions]
    complex_names = [
        protein_complex["name"]
        for protein_complex in filtered["entities"]["protein_complexes"]
    ]

    assert reaction_names == ["good reaction"]
    assert removed == ["bad reaction", "ghost reaction"]
    assert "hexadecatrienoic acid (16:3)" not in complex_names
    assert complex_names == ["valid enzyme complex"]
