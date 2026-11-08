"""The "Generate PWML" step, moved out of ``streamlit_app.py`` (R4a).

:func:`run_export` snapshots the reviewed payload and exports it. The caller supplies
the ``state`` mapping (the app passes ``st.session_state``), the ``reporter`` (the app
passes ``StreamlitReporter(st)``) and the folders. The button, its try/except, the
``pwml_export_result`` write and the success/error text stay in the app.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, MutableMapping

# Module imports, so these are looked up at call time as the app script's per-run
# ``from ... import`` did: a patch or spy of ``postpipeline.run_pwml_export`` on its
# source module reaches the call (tests/equivalence/harness.py).
from t2pw.pipeline import export_mode as _export_mode
from t2pw.service import postpipeline as _postpipeline
from t2pw.service.reporter import Reporter


def quarantine_session_state(*, state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """The stored boundary decision, for callers downstream of it."""

    return {
        "quarantine_report": _postpipeline._safe_dict(state.get("quarantine_report")),
        "quarantine_artifacts": _postpipeline._safe_dict(state.get("quarantine_artifacts")),
    }


def _pwml_reference_path(project_root: Path) -> Path:
    ref_candidates = [
        project_root / "reference" / "PW000001.pwml",
        project_root / "reference" / "PW012926.pwml",
    ]
    return next((path for path in ref_candidates if path.exists()), ref_candidates[0])


def _write_reviewed_payload_snapshot(
    working_json: Dict[str, Any], work_dir: str | Path, *, state: MutableMapping[str, Any]
) -> Path:
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    reviewed_json_path = work_path / "final.mapped.json"
    reviewed_json_path.write_text(
        json.dumps(working_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    post_artifacts = state.get("post_pipeline_artifacts")
    if isinstance(post_artifacts, dict):
        reviewed_payload = deepcopy(working_json)
        post_artifacts["final_mapped_db"] = reviewed_payload
        post_artifacts["final_export_input"] = deepcopy(reviewed_payload)
        post_artifacts["reviewed_final_mapped_path"] = str(reviewed_json_path)
        state["post_pipeline_artifacts"] = post_artifacts

    return reviewed_json_path


def run_export(
    *,
    state: MutableMapping[str, Any],
    reporter: Reporter,
    work_dir: str | Path,
    project_root: Path,
) -> Dict[str, Any]:
    """Export the reviewed payload in ``state``; returns the export's result dict.

    The two refusals (no reviewed payload, Stage 3 gate errors) come back as failure
    dicts. An exception from the export propagates: the caller builds the failure dict.
    """

    with reporter.spinner("Building PWML from reviewed pathway JSON..."):
        working_json = state.get("refinement_working_json")
        if not isinstance(working_json, dict) or not working_json:
            return {
                "ok": False,
                "error": "No reviewed mapped pathway is available for PWML export.",
                "counts": {},
                "issues": 0,
                "output_path": "",
                "qa": {},
                "grounding_report": {},
            }

        refinement_gate_errors = _postpipeline._safe_list(state.get("refinement_gate_errors"))
        if refinement_gate_errors:
            return {
                "ok": False,
                "error": (
                    "PWML export stopped by pre-export Stage 3 revalidation. "
                    "Resolve the refinement gate errors first."
                ),
                "counts": {},
                "issues": len(refinement_gate_errors),
                "output_path": "",
                "qa": {},
                "grounding_report": {},
                "stage3_gate_errors": refinement_gate_errors,
                # Carried even here. This return is reached when the boundary refused
                # OR when Stage 3 failed after it, and the artifacts are the only
                # record of which -- dropping them makes the two indistinguishable.
                **quarantine_session_state(state=state),
            }

        _write_reviewed_payload_snapshot(working_json, work_dir, state=state)
        state["refinement_pwml_ready"] = True

        grounding_dict = None
        if bool(state.get("pwml_grounding", False)):
            grounding_dict = state.get("pwml_grounding_dict")

        return _postpipeline.run_pwml_export(
            working_json,
            pathway_name=str(state.get("pwml_name") or "Generated Pathway"),
            pathway_description=str(state.get("pwml_description") or ""),
            pathway_subject=str(state.get("pwml_subject") or "Metabolic"),
            project_root=project_root,
            ref_path=_pwml_reference_path(project_root),
            vis_width=int(state.get("pwml_width") or 3200),
            vis_height=int(state.get("pwml_height") or 1400),
            background_color=str(state.get("pwml_bg") or "#FFFFFF"),
            grounding_dict=grounding_dict,
            strict_db=bool(state.get("pwml_strict_db", True)),
            # The boundary already ran, upstream of the Stage 3 check that gated this
            # payload into review. Hand its decisions over rather than making new ones.
            pathway_context=_postpipeline._safe_dict(state.get("pathway_context")),
            export_mode=_export_mode.coerce_mode(state.get("export_mode")),
            **quarantine_session_state(state=state),
        )
