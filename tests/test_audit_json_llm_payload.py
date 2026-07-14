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
from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402


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


def test_deterministic_audit_does_not_mark_enzyme_less_reaction_spontaneous() -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [],
        },
        "processes": {
            "reactions": [
                {"name": "A to B", "inputs": ["A"], "outputs": ["B"], "enzymes": []}
            ]
        },
    }

    result = audit_payload(payload, use_llm=False)

    assert not any(
        op["path"] == "/processes/reactions/0/spontaneous"
        for op in result["patch"]
    )


def test_deterministic_audit_does_not_mark_catalyzed_reaction_spontaneous() -> None:
    payload = {
        "entities": {
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [{"name": "Enzyme"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "A to B",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "modifiers": [
                        {"entity": "Enzyme", "entity_type": "protein", "role": "catalyst"}
                    ],
                }
            ]
        },
    }

    result = audit_payload(payload, use_llm=False)

    assert not any(
        op["path"] == "/processes/reactions/0/spontaneous"
        for op in result["patch"]
    )


def test_deterministic_audit_patches_explicit_protein_complex_component_counts() -> None:
    evidence = "NdmCDE consists of three NdmC, three NdmD, and three NdmE subunits"
    payload = {
        "entities": {
            "compounds": [],
            "proteins": [{"name": name} for name in ("NdmC", "NdmD", "NdmE")],
            "protein_complexes": [
                {
                    "name": "NdmCDE",
                    "evidence": evidence,
                    "components": [
                        {"name": "NdmC", "resolution_owner": "audit"},
                        {"name": "NdmD", "resolution_owner": "audit"},
                        {"name": "NdmE", "resolution_owner": "audit"},
                    ],
                }
            ],
        },
        "processes": {"reactions": []},
    }

    result = audit_payload(payload, use_llm=False)

    expected_paths = {
        f"/entities/protein_complexes/0/components/{idx}/stoichiometry"
        for idx in range(3)
    }
    stoichiometry_errors = {
        error["path"]
        for error in result["report"]["errors"]
        if "missing positive stoichiometry" in error["reason"]
    }
    stoichiometry_patches = [
        operation for operation in result["patch"] if operation["path"] in expected_paths
    ]
    assert stoichiometry_errors == expected_paths
    assert {operation["path"] for operation in stoichiometry_patches} == expected_paths
    assert all(operation["op"] == "add" for operation in stoichiometry_patches)
    assert all(operation["value"] == 3 for operation in stoichiometry_patches)
    assert all(operation["evidence"] == evidence for operation in stoichiometry_patches)

    patched, apply_report = apply_patch_with_policy(payload, stoichiometry_patches)

    assert apply_report["summary"]["accepted_count"] == 3
    assert [
        component["stoichiometry"]
        for component in patched["entities"]["protein_complexes"][0]["components"]
    ] == [3, 3, 3]


def test_deterministic_audit_does_not_infer_missing_or_ambiguous_component_counts() -> None:
    evidence_cases = [
        "NdmC, NdmD, and NdmE are subunits of NdmCDE.",
        "NdmCDE contains about three NdmC, two or three NdmD, and NdmE subunits.",
    ]
    for evidence in evidence_cases:
        payload = {
            "entities": {
                "compounds": [],
                "proteins": [{"name": name} for name in ("NdmC", "NdmD", "NdmE")],
                "protein_complexes": [
                    {
                        "name": "NdmCDE",
                        "evidence": evidence,
                        "components": [{"name": name} for name in ("NdmC", "NdmD", "NdmE")],
                    }
                ],
            },
            "processes": {"reactions": []},
        }

        result = audit_payload(payload, use_llm=False)

        errors = [
            error
            for error in result["report"]["errors"]
            if "missing positive stoichiometry" in error["reason"]
        ]
        assert [error["path"] for error in errors] == [
            f"/entities/protein_complexes/0/components/{idx}/stoichiometry"
            for idx in range(3)
        ]
        assert not any("/stoichiometry" in operation["path"] for operation in result["patch"])


def test_deterministic_audit_canonicalizes_positive_legacy_component_coefficient() -> None:
    payload = {
        "entities": {
            "compounds": [],
            "proteins": [{"name": "NdmE"}],
            "protein_complexes": [
                {
                    "name": "NdmCDE",
                    "components": [{"name": "NdmE", "coefficient": "3"}],
                }
            ],
        },
        "processes": {"reactions": []},
    }

    result = audit_payload(payload, use_llm=False)

    stoichiometry_patch = next(
        operation
        for operation in result["patch"]
        if operation["path"] == "/entities/protein_complexes/0/components/0/stoichiometry"
    )
    assert stoichiometry_patch["op"] == "add"
    assert stoichiometry_patch["value"] == 3
    assert "explicit legacy coefficient 3" in stoichiometry_patch["evidence"]

    patched, apply_report = apply_patch_with_policy(payload, [stoichiometry_patch])

    component = patched["entities"]["protein_complexes"][0]["components"][0]
    assert apply_report["summary"]["accepted_count"] == 1
    assert component == {"name": "NdmE", "coefficient": "3", "stoichiometry": 3}
