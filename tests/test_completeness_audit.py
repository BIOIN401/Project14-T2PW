"""Tests for t2pw.curation.completeness_audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from t2pw.curation.completeness_audit import (
    AUDIT_REPORT_FILENAME,
    _build_user_prompt,
    _compact_reactions,
    _parse_llm_response,
    _resolve_completeness_model,
    _truncate_source_text,
    run_final_completeness_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_manifest() -> List[Dict[str, Any]]:
    return [
        {
            "locked_reaction_id": "rxn_lock_001",
            "name": "Glucose phosphorylation",
            "inputs": ["Glucose"],
            "outputs": ["Glucose-6-phosphate"],
            "modifiers_or_enzymes": ["Hexokinase"],
            "evidence_quote": "Hexokinase catalyzes the phosphorylation of glucose.",
            "scope_membership": "core",
            "locked": True,
            "status": "valid",
        },
        {
            "locked_reaction_id": "rxn_lock_002",
            "name": "G6P isomerization",
            "inputs": ["Glucose-6-phosphate"],
            "outputs": ["Fructose-6-phosphate"],
            "modifiers_or_enzymes": ["Phosphoglucose isomerase"],
            "evidence_quote": "Isomerization of G6P to F6P.",
            "scope_membership": "core",
            "locked": True,
            "status": "valid",
        },
    ]


@pytest.fixture()
def sample_pathway_json() -> Dict[str, Any]:
    return {
        "metadata": {"pathway_name": "Glycolysis"},
        "entities": {
            "compounds": [
                {"name": "Glucose"},
                {"name": "Glucose-6-phosphate"},
                {"name": "Fructose-6-phosphate"},
            ],
            "proteins": [
                {"name": "Hexokinase"},
                {"name": "Phosphoglucose isomerase"},
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "Glucose phosphorylation",
                    "inputs": ["Glucose"],
                    "outputs": ["Glucose-6-phosphate"],
                    "locked_reaction_id": "rxn_lock_001",
                },
                {
                    "name": "G6P isomerization",
                    "inputs": ["Glucose-6-phosphate"],
                    "outputs": ["Fructose-6-phosphate"],
                    "locked_reaction_id": "rxn_lock_002",
                },
            ],
        },
    }


@pytest.fixture()
def sample_source_text() -> str:
    return (
        "Glycolysis is a metabolic pathway that converts glucose into pyruvate. "
        "The first step is the phosphorylation of glucose by hexokinase to form "
        "glucose-6-phosphate. Then phosphoglucose isomerase converts G6P to "
        "fructose-6-phosphate."
    )


@pytest.fixture()
def mock_llm_response() -> str:
    return json.dumps(
        {
            "locked_reaction_count": 2,
            "preserved_count": 2,
            "missing_locked_reactions": [],
            "changed_locked_reactions": [],
            "quarantined_locked_reactions": [],
            "possible_missed_in_scope_reactions": [],
            "possible_out_of_scope_reactions": [],
            "overall_completeness": "complete",
            "confidence": 0.95,
        }
    )


# ---------------------------------------------------------------------------
# Test: disabled audit returns minimal skip report
# ---------------------------------------------------------------------------
class TestDisabledAudit:
    def test_returns_skip_report(self, sample_pathway_json: Dict[str, Any]) -> None:
        report = run_final_completeness_audit(
            source_text="some text",
            final_pathway_json=sample_pathway_json,
            enabled=False,
        )
        assert report == {"enabled": False, "skipped": True}

    def test_disabled_does_not_call_llm(self, sample_pathway_json: Dict[str, Any]) -> None:
        with patch("t2pw.llm.client.chat") as mock_chat:
            run_final_completeness_audit(
                source_text="some text",
                final_pathway_json=sample_pathway_json,
                enabled=False,
            )
            mock_chat.assert_not_called()

    def test_disabled_writes_report_when_output_path_given(
        self, sample_pathway_json: Dict[str, Any], tmp_path: Path
    ) -> None:
        run_final_completeness_audit(
            source_text="some text",
            final_pathway_json=sample_pathway_json,
            enabled=False,
            output_path=tmp_path,
        )
        report_file = tmp_path / AUDIT_REPORT_FILENAME
        assert report_file.exists()
        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert data == {"enabled": False, "skipped": True}


# ---------------------------------------------------------------------------
# Test: missing manifest returns appropriate skip response
# ---------------------------------------------------------------------------
class TestMissingManifest:
    def test_no_manifest_returns_skip(self, sample_pathway_json: Dict[str, Any]) -> None:
        report = run_final_completeness_audit(
            source_text="some text",
            final_pathway_json=sample_pathway_json,
            enabled=True,
        )
        assert report["enabled"] is True
        assert report["skipped"] is True
        assert report["reason"] == "no_locked_manifest"

    def test_nonexistent_path_returns_skip(
        self, sample_pathway_json: Dict[str, Any], tmp_path: Path
    ) -> None:
        report = run_final_completeness_audit(
            source_text="some text",
            final_pathway_json=sample_pathway_json,
            locked_manifest_path=tmp_path / "does_not_exist.json",
            enabled=True,
        )
        assert report["skipped"] is True
        assert report["reason"] == "no_locked_manifest"

    def test_missing_manifest_does_not_call_llm(
        self, sample_pathway_json: Dict[str, Any]
    ) -> None:
        with patch("t2pw.llm.client.chat") as mock_chat:
            run_final_completeness_audit(
                source_text="some text",
                final_pathway_json=sample_pathway_json,
                enabled=True,
            )
            mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Test: LLM prompt construction
# ---------------------------------------------------------------------------
class TestPromptConstruction:
    def test_prompt_contains_source_text(
        self,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        prompt = _build_user_prompt(
            source_text=sample_source_text,
            preprocessor_context=None,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            final_preservation_report=None,
            quarantined_reactions=None,
        )
        assert "## Source Text" in prompt
        assert "Glycolysis is a metabolic pathway" in prompt

    def test_prompt_contains_manifest(
        self,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        prompt = _build_user_prompt(
            source_text=sample_source_text,
            preprocessor_context=None,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            final_preservation_report=None,
            quarantined_reactions=None,
        )
        assert "## Locked Reaction Manifest" in prompt
        assert "rxn_lock_001" in prompt
        assert "rxn_lock_002" in prompt

    def test_prompt_contains_compact_reactions(
        self,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        prompt = _build_user_prompt(
            source_text=sample_source_text,
            preprocessor_context=None,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            final_preservation_report=None,
            quarantined_reactions=None,
        )
        assert "## Final Pathway Reactions (compact)" in prompt
        assert "Glucose phosphorylation" in prompt

    def test_prompt_includes_preprocessor_context(
        self,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        context = {"pathway_type": "metabolic", "organism": "Homo sapiens"}
        prompt = _build_user_prompt(
            source_text=sample_source_text,
            preprocessor_context=context,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            final_preservation_report=None,
            quarantined_reactions=None,
        )
        assert "## Preprocessor Context Summary" in prompt
        assert "Homo sapiens" in prompt

    def test_prompt_includes_preservation_report(
        self,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        pres_report = {"preserved_count": 2, "missing_count": 0}
        prompt = _build_user_prompt(
            source_text=sample_source_text,
            preprocessor_context=None,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            final_preservation_report=pres_report,
            quarantined_reactions=None,
        )
        assert "## Preservation Report" in prompt
        assert "preserved_count" in prompt

    def test_prompt_includes_quarantined_reactions(
        self,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        quarantined = [{"name": "Some bad reaction", "reason": "out_of_scope"}]
        prompt = _build_user_prompt(
            source_text=sample_source_text,
            preprocessor_context=None,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            final_preservation_report=None,
            quarantined_reactions=quarantined,
        )
        assert "## Quarantined Reactions" in prompt
        assert "Some bad reaction" in prompt

    def test_prompt_truncates_long_source(
        self,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        long_text = " ".join(["word"] * 10000)
        prompt = _build_user_prompt(
            source_text=long_text,
            preprocessor_context=None,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            final_preservation_report=None,
            quarantined_reactions=None,
        )
        assert "[... truncated at 8000 words ...]" in prompt

    @patch("t2pw.llm.client.chat")
    def test_chat_called_with_correct_messages(
        self,
        mock_chat: MagicMock,
        mock_llm_response: str,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        mock_chat.return_value = mock_llm_response
        run_final_completeness_audit(
            source_text=sample_source_text,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            enabled=True,
            temperature=0.1,
            max_tokens=5000,
        )
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "completeness auditor" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "## Source Text" in messages[1]["content"]
        # Check keyword arguments
        assert call_args[1]["temperature"] == 0.1
        assert call_args[1]["max_tokens"] == 5000
        assert call_args[1]["response_json"] is True
        assert call_args[1]["stage_name"] == "final_completeness_audit"


# ---------------------------------------------------------------------------
# Test: output file writing
# ---------------------------------------------------------------------------
class TestOutputWriting:
    @patch("t2pw.llm.client.chat")
    def test_writes_report_to_file(
        self,
        mock_chat: MagicMock,
        mock_llm_response: str,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
        tmp_path: Path,
    ) -> None:
        mock_chat.return_value = mock_llm_response
        report = run_final_completeness_audit(
            source_text=sample_source_text,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            output_path=tmp_path,
            enabled=True,
        )
        report_file = tmp_path / AUDIT_REPORT_FILENAME
        assert report_file.exists()
        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert data["locked_reaction_count"] == 2
        assert data["overall_completeness"] == "complete"
        assert data["enabled"] is True
        assert data["llm_ok"] is True

    def test_no_file_without_output_path(
        self, sample_pathway_json: Dict[str, Any], tmp_path: Path
    ) -> None:
        # Disabled audit, no output_path -- no file should be written
        run_final_completeness_audit(
            source_text="some text",
            final_pathway_json=sample_pathway_json,
            enabled=False,
            # output_path intentionally omitted
        )
        # The tmp_path should be empty
        assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Test: LLM error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    @patch("t2pw.llm.client.chat")
    def test_llm_failure_returns_error_report(
        self,
        mock_chat: MagicMock,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        mock_chat.side_effect = RuntimeError("LLM connection timeout")
        report = run_final_completeness_audit(
            source_text=sample_source_text,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            enabled=True,
        )
        assert report["enabled"] is True
        assert report["llm_ok"] is False
        assert "LLM connection timeout" in report["error"]

    @patch("t2pw.llm.client.chat")
    def test_malformed_json_returns_error_report(
        self,
        mock_chat: MagicMock,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
    ) -> None:
        mock_chat.return_value = "not valid json {{"
        report = run_final_completeness_audit(
            source_text=sample_source_text,
            locked_manifest=sample_manifest,
            final_pathway_json=sample_pathway_json,
            enabled=True,
        )
        assert report["enabled"] is True
        assert report["llm_ok"] is False
        assert "error" in report


# ---------------------------------------------------------------------------
# Test: manifest loading from file
# ---------------------------------------------------------------------------
class TestManifestLoading:
    def test_manifest_from_path(
        self,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
        mock_llm_response: str,
        tmp_path: Path,
    ) -> None:
        manifest_path = tmp_path / "locked_reaction_manifest.json"
        manifest_path.write_text(
            json.dumps(sample_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with patch("t2pw.llm.client.chat") as mock_chat:
            mock_chat.return_value = mock_llm_response
            report = run_final_completeness_audit(
                source_text=sample_source_text,
                locked_manifest_path=manifest_path,
                final_pathway_json=sample_pathway_json,
                enabled=True,
            )
        assert report.get("llm_ok") is True
        assert report.get("locked_reaction_count") == 2

    def test_manifest_inline_takes_precedence(
        self,
        sample_source_text: str,
        sample_manifest: List[Dict[str, Any]],
        sample_pathway_json: Dict[str, Any],
        mock_llm_response: str,
        tmp_path: Path,
    ) -> None:
        """When both path and inline manifest are provided, inline wins."""
        manifest_path = tmp_path / "locked_reaction_manifest.json"
        # Write a different manifest to disk
        manifest_path.write_text(
            json.dumps([{"locked_reaction_id": "rxn_lock_999", "name": "Decoy"}]),
            encoding="utf-8",
        )
        with patch("t2pw.llm.client.chat") as mock_chat:
            mock_chat.return_value = mock_llm_response
            report = run_final_completeness_audit(
                source_text=sample_source_text,
                locked_manifest_path=manifest_path,
                locked_manifest=sample_manifest,  # inline takes precedence
                final_pathway_json=sample_pathway_json,
                enabled=True,
            )
        # Should have used the inline manifest (2 reactions), not the decoy
        mock_chat.assert_called_once()
        call_messages = mock_chat.call_args[0][0]
        user_msg = call_messages[1]["content"]
        assert "rxn_lock_001" in user_msg
        assert "rxn_lock_002" in user_msg


# ---------------------------------------------------------------------------
# Test: model resolution
# ---------------------------------------------------------------------------
class TestModelResolution:
    def test_final_completeness_model_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_FINAL_COMPLETENESS_MODEL", "owl-alpha")
        monkeypatch.delenv("OPENROUTER_EXTRACTION_MODEL", raising=False)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        assert _resolve_completeness_model() == "owl-alpha"

    def test_fallback_to_extraction_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_FINAL_COMPLETENESS_MODEL", raising=False)
        monkeypatch.setenv("OPENROUTER_EXTRACTION_MODEL", "gpt-4")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        assert _resolve_completeness_model() == "gpt-4"

    def test_fallback_to_openrouter_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_FINAL_COMPLETENESS_MODEL", raising=False)
        monkeypatch.delenv("OPENROUTER_EXTRACTION_MODEL", raising=False)
        monkeypatch.setenv("OPENROUTER_MODEL", "claude-3")
        assert _resolve_completeness_model() == "claude-3"

    def test_returns_none_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_FINAL_COMPLETENESS_MODEL", raising=False)
        monkeypatch.delenv("OPENROUTER_EXTRACTION_MODEL", raising=False)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        assert _resolve_completeness_model() is None


# ---------------------------------------------------------------------------
# Test: helper utilities
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_truncate_short_text(self) -> None:
        text = "short text"
        assert _truncate_source_text(text) == text

    def test_truncate_long_text(self) -> None:
        text = " ".join(["word"] * 10000)
        result = _truncate_source_text(text)
        assert "[... truncated at 8000 words ...]" in result
        # Should have exactly 8000 words before the truncation marker
        lines = result.split("\n\n")
        words_before = lines[0].split()
        assert len(words_before) == 8000

    def test_compact_reactions(self, sample_pathway_json: Dict[str, Any]) -> None:
        compact = _compact_reactions(sample_pathway_json)
        assert len(compact) == 2
        assert compact[0]["name"] == "Glucose phosphorylation"
        assert compact[0]["inputs"] == ["Glucose"]
        assert compact[0]["outputs"] == ["Glucose-6-phosphate"]
        assert compact[0]["locked_reaction_id"] == "rxn_lock_001"

    def test_compact_reactions_handles_dict_inputs(self) -> None:
        payload = {
            "processes": {
                "reactions": [
                    {
                        "name": "Test",
                        "inputs": [{"name": "A"}, {"entity": "B"}],
                        "outputs": ["C"],
                    }
                ]
            }
        }
        compact = _compact_reactions(payload)
        assert compact[0]["inputs"] == ["A", "B"]

    def test_compact_reactions_empty_payload(self) -> None:
        assert _compact_reactions({}) == []
        assert _compact_reactions({"processes": {}}) == []

    def test_parse_llm_response_valid(self) -> None:
        raw = json.dumps({
            "locked_reaction_count": 5,
            "preserved_count": 4,
            "missing_locked_reactions": [{"name": "X", "evidence": "...", "confidence": 0.8, "reason": "not found"}],
            "changed_locked_reactions": [],
            "quarantined_locked_reactions": [],
            "possible_missed_in_scope_reactions": [],
            "possible_out_of_scope_reactions": [],
            "overall_completeness": "mostly_complete",
            "confidence": 0.85,
        })
        report = _parse_llm_response(raw)
        assert report["locked_reaction_count"] == 5
        assert report["overall_completeness"] == "mostly_complete"
        assert report["confidence"] == 0.85

    def test_parse_llm_response_with_code_fences(self) -> None:
        raw = '```json\n{"locked_reaction_count": 1, "preserved_count": 1, "overall_completeness": "complete", "confidence": 0.9}\n```'
        report = _parse_llm_response(raw)
        assert report["locked_reaction_count"] == 1

    def test_parse_llm_response_normalizes_invalid_completeness(self) -> None:
        raw = json.dumps({
            "locked_reaction_count": 1,
            "preserved_count": 1,
            "overall_completeness": "bogus_value",
            "confidence": 0.5,
        })
        report = _parse_llm_response(raw)
        assert report["overall_completeness"] == "incomplete"

    def test_parse_llm_response_ensures_list_fields(self) -> None:
        raw = json.dumps({
            "locked_reaction_count": 1,
            "preserved_count": 1,
            "overall_completeness": "complete",
            "confidence": 0.9,
            "missing_locked_reactions": "not a list",
        })
        report = _parse_llm_response(raw)
        assert report["missing_locked_reactions"] == []
