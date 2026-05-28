from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.preprocessor import (  # noqa: E402
    format_context_header,
    is_ambiguous_multi_example_review_context,
)
from t2pw.pipeline.pipeline import (  # noqa: E402
    PipelineFailure,
    run_extraction_pipeline,
    run_inference_pipeline,
)


def test_format_context_header_includes_scope_metadata_without_pathway_fields() -> None:
    header = format_context_header(
        {
            "document_type": "multi_example_review",
            "context_type": "review",
            "scope_status": "ambiguous",
            "scope_clarity_score": 0.2,
            "selected_example": "",
            "candidate_examples": [
                {"name": "Example A", "organism": "Species one"},
                {"name": "Example B"},
                {"organism": "missing name"},
            ],
            "warning": "No specific example was selected.",
        }
    )

    assert "document_type: multi_example_review" in header
    assert "context_type: review" in header
    assert "scope_status: ambiguous" in header
    assert "scope_clarity_score: 0.2" in header
    assert "selected_example: " in header
    assert "candidate_examples: Example A, Example B" in header
    assert "warning: No specific example was selected." in header


def test_is_ambiguous_multi_example_review_context_requires_blank_selected_example() -> None:
    assert is_ambiguous_multi_example_review_context(
        {"document_type": "multi_example_review", "selected_example": ""}
    )
    assert is_ambiguous_multi_example_review_context(
        {"document_type": "multi_example_review", "selected_example": "   "}
    )
    assert is_ambiguous_multi_example_review_context(
        {"document_type": "multi_example_review", "selected_example": []}
    )
    assert not is_ambiguous_multi_example_review_context(
        {"document_type": "multi_example_review", "selected_example": "Example A"}
    )
    assert not is_ambiguous_multi_example_review_context(
        {"document_type": "primary_research", "selected_example": ""}
    )
    assert not is_ambiguous_multi_example_review_context(None)


def test_run_extraction_pipeline_rejects_ambiguous_review_before_llm(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("LLM stage should not run for ambiguous multi-example reviews")

    monkeypatch.setattr("t2pw.pipeline.pipeline._run_json_stage", fail_if_called)

    try:
        run_extraction_pipeline(
            "review text",
            pathway_context={
                "document_type": "multi_example_review",
                "selected_example": "",
                "candidate_examples": [{"name": "Example A"}],
            },
        )
    except PipelineFailure as exc:
        assert exc.stage == "ambiguous_review_scope"
        assert "ambiguous_review_scope" in exc.attempts[0]["raw"]
        assert "Example A" in exc.attempts[0]["raw"]
    else:
        raise AssertionError("Expected PipelineFailure for ambiguous review scope")


def test_run_inference_pipeline_rejects_ambiguous_review_before_llm(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("LLM stage should not run for ambiguous multi-example reviews")

    monkeypatch.setattr("t2pw.pipeline.pipeline._run_json_stage", fail_if_called)

    try:
        run_inference_pipeline(
            "review text",
            {},
            pathway_context={
                "document_type": "multi_example_review",
                "selected_example": "",
                "candidate_examples": [{"name": "Example B"}],
            },
        )
    except PipelineFailure as exc:
        assert exc.stage == "ambiguous_review_scope"
        assert "ambiguous_review_scope" in exc.attempts[0]["raw"]
        assert "Example B" in exc.attempts[0]["raw"]
    else:
        raise AssertionError("Expected PipelineFailure for ambiguous review scope")
