from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import streamlit as st

from .converter import MODE_DETERMINISTIC, MODE_LLM, convert_sbml_for_pathwhiz


def _metrics_table(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = ["isolated_species_count", "duplicate_reactions_count", "compartments_count"]
    rows: List[Dict[str, int]] = []
    for key in keys:
        rows.append(
            {
                "metric": key,
                "before": int(before.get(key, 0)),
                "after": int(after.get(key, 0)),
            }
        )
    return rows


def render_pathwhiz_converter_section(llm_client) -> None:
    st.subheader("PathWhiz Converter")

    input_sbml = st.file_uploader(
        "Upload your SBML (.xml)",
        type=["xml", "sbml"],
        accept_multiple_files=False,
        key="pathwhiz_converter_input_sbml",
    )
    example_files = st.file_uploader(
        "Upload PathWhiz-working example SBML files (.xml)",
        type=["xml", "sbml"],
        accept_multiple_files=True,
        key="pathwhiz_converter_examples",
    )
    mode = st.radio(
        "Mode",
        options=[MODE_DETERMINISTIC, MODE_LLM],
        index=0,
        key="pathwhiz_converter_mode",
    )

    allow_add_reaction = False
    if mode == MODE_LLM:
        allow_add_reaction = st.checkbox(
            "Allow adding missing bridge steps (proteolysis/import)",
            value=False,
            key="pathwhiz_converter_allow_bridge",
        )

    if st.button("Convert", key="pathwhiz_converter_convert_btn"):
        if input_sbml is None:
            st.error("Upload your SBML (.xml) first.")
            return
        if mode == MODE_LLM and not example_files:
            st.error("LLM mode requires one or more PathWhiz-working example SBML files.")
            return

        source_bytes = input_sbml.getvalue()
        examples: List[Tuple[str, bytes]] = []
        for upload in example_files or []:
            examples.append((upload.name, upload.getvalue()))

        try:
            with st.spinner("Converting SBML for PathWhiz compatibility..."):
                result = convert_sbml_for_pathwhiz(
                    input_sbml_bytes=source_bytes,
                    mode=mode,
                    llm_client=llm_client,
                    example_sbml_files=examples,
                    allow_add_reaction=bool(allow_add_reaction),
                    max_llm_rounds=2,
                )
            st.session_state["pathwhiz_converter_result"] = result
            st.success("PathWhiz conversion completed.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"PathWhiz conversion failed: {exc}")
            return

    result = st.session_state.get("pathwhiz_converter_result")
    if not isinstance(result, dict):
        return

    report = result.get("report", {})
    before = report.get("counts_before", {})
    after = report.get("counts_after", {})
    st.write("Metrics before/after", _metrics_table(before, after))
    st.write(
        {
            "disconnected_nodes_count": int(after.get("isolated_species_count", 0)),
            "validation_has_errors": bool(report.get("validation", {}).get("final", {}).get("has_errors", False)),
        }
    )
    st.write("Validation summary", report.get("validation", {}).get("final", {}))

    applied_ops = report.get("applied_ops", [])
    st.write("Changes (first 50 ops)", applied_ops[:50])
    if len(applied_ops) > 50:
        st.caption(f"Showing 50 of {len(applied_ops)} applied ops.")

    st.download_button(
        "Download converted_pathwhiz.xml",
        data=result.get("converted_sbml_bytes", b""),
        file_name="converted_pathwhiz.xml",
        mime="application/xml",
        key="dl_pathwhiz_converted_sbml",
    )
    st.download_button(
        "Download patch_plan.json",
        data=json.dumps(result.get("patch_plan", {}), indent=2),
        file_name="patch_plan.json",
        mime="application/json",
        key="dl_pathwhiz_patch_plan",
    )
    st.download_button(
        "Download converter_report.json",
        data=json.dumps(result.get("report", {}), indent=2),
        file_name="converter_report.json",
        mime="application/json",
        key="dl_pathwhiz_converter_report",
    )
