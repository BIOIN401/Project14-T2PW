"""Streamlit "Reaction Evidence" section (ORCH-742b). Presentation only.

Everything shown comes from :mod:`t2pw.app.reaction_evidence`, which reads the
run's existing artifacts. Nothing here edits the pathway, and nothing calls a
model or the network. The .txt download is ``render_text`` of the same report
the section displays.

Which run: the directory the app's quarantine boundary wrote this session's
artifacts to (``st.session_state["quarantine_artifacts"]``), or any directory the
reviewer pastes, such as a batch leg ``runs_verify/<run>/papers/<PMCID>/strict``.
The app never writes ``final_mapped.json`` for an interactive run, so for the
session's own directory only, the canonical payload is taken from session state
under the same key order the batch driver uses to write that file, and the report
names that source.

The section never emits ``st.error`` / ``st.warning`` and never raises: the batch
driver drives this page through AppTest and folds every error, warning and
exception into the leg's ``detail`` and failure classification
(``batch.driver._collect_app_text``). A review panel must not change a leg's
outcome, so its notices are ``st.info`` / coloured markdown only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

from t2pw.app.reaction_evidence import (
    DUPLICATE_WARNING,
    FINAL_MAPPED_FILENAME,
    JSON_ARTIFACTS,
    NONE_RECORDED,
    NOT_RECORDED,
    PWML_IR_FILENAME,
    build_report,
    coverage_lines,
    header_lines,
    interaction_lines,
    interactions_title,
    load_run_artifacts,
    quarantined_lines,
    reaction_lines,
    render_text,
    with_note,
)
from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY
from t2pw.pipeline.strict_quarantine import COVERAGE_REPORT_FILENAME

logger = logging.getLogger(__name__)

DIR_KEY = "reaction_evidence_dir"
DOWNLOAD_LABEL = "Download Reaction Evidence (.txt)"
NO_RUN_MESSAGE = (
    "No run loaded. Complete a pipeline run above, or paste a run artifact directory "
    "(for a batch leg: runs_verify/<run>/papers/<PMCID>/strict)."
)


def _session_run_dir() -> str:
    produced = st.session_state.get("quarantine_artifacts")
    path = produced.get(COVERAGE_REPORT_FILENAME) if isinstance(produced, dict) else None
    return str(Path(path).parent) if path else ""


def _session_payload() -> Tuple[str, Optional[Dict[str, Any]]]:
    # Same order as batch.driver._add_identity_artifacts writes final_mapped.json.
    artifacts = st.session_state.get("post_pipeline_artifacts")
    for key in (CANONICAL_PAYLOAD_KEY, "final_mapped_db", "final_mapped"):
        value = artifacts.get(key) if isinstance(artifacts, dict) else None
        if isinstance(value, dict) and value:
            return key, value
    return "", None


def _session_ir() -> Optional[Dict[str, Any]]:
    # The IR this session's PWML export built (``streamlit_app`` stores the export
    # result here). Read as it stands: the verdicts are the exporter's, not this page's.
    result = st.session_state.get("pwml_export_result")
    ir = result.get("pwml_ir") if isinstance(result, dict) else None
    return ir if isinstance(ir, dict) and ir else None


def _block(lines) -> None:
    st.code("\n".join(lines), language=None, wrap_lines=True)


def render_reaction_evidence() -> None:
    st.divider()
    st.subheader("Reaction Evidence")
    st.caption(
        "Read-only review of what the run's artifacts say the pathway contains. "
        "Nothing here edits the pathway or calls a model. Missing fields read 'Not recorded'."
    )
    try:
        _render_body()
    except Exception as exc:  # noqa: BLE001 -- see the module docstring
        logger.debug("reaction evidence could not be rendered", exc_info=True)
        st.info(f"Reaction Evidence could not be rendered: {exc}")


def _render_body() -> None:
    session_dir = _session_run_dir()
    # The session directory is part of the key: a keyed widget keeps its old value
    # when its default changes, so a page opened before the run would stay empty.
    directory = st.text_input("Run artifact directory", value=session_dir, key=f"{DIR_KEY}:{session_dir}").strip()
    if not directory:
        st.info(NO_RUN_MESSAGE)
        return
    if not Path(directory).is_dir():
        st.info(f"Not a directory: {directory}")
        return

    artifacts = load_run_artifacts(directory)
    source = FINAL_MAPPED_FILENAME
    if FINAL_MAPPED_FILENAME not in artifacts and directory == session_dir:
        key, payload = _session_payload()
        if payload is not None:
            artifacts[FINAL_MAPPED_FILENAME] = payload
            source = f"session post_pipeline_artifacts[{key}]"
    ir_source = PWML_IR_FILENAME
    if PWML_IR_FILENAME not in artifacts and directory == session_dir and _session_ir() is not None:
        artifacts[PWML_IR_FILENAME] = _session_ir()
        ir_source = "session pwml_export_result[pwml_ir]"
    if not artifacts:
        st.info(f"None of {', '.join(JSON_ARTIFACTS)} was found in {directory}.")
        return

    report = build_report(artifacts, payload_source=source, ir_source=ir_source)
    st.download_button(
        DOWNLOAD_LABEL,
        data=render_text(report).encode("utf-8"),
        file_name="reaction_evidence.txt",
        mime="text/plain",
        key="reaction_evidence_download",
    )
    columns = st.columns(4)
    columns[0].metric("Coverage", report["coverage_ratio"])
    columns[1].metric("Minimum core satisfied", report["minimum_core_satisfied"])
    columns[2].metric("Surviving processes", report["surviving_processes"])
    columns[3].metric("Quarantined processes", report["quarantined_processes"])
    _block(header_lines(report) + coverage_lines(report))

    heading = with_note(f"Surviving reactions ({len(report['reactions'])})", report["surviving_note"])
    st.markdown(f"**{heading}**")
    if not report["reactions"]:
        missing = report["payload_source"] == NOT_RECORDED
        st.info(f"Surviving reactions: {NOT_RECORDED + ' (no final_mapped.json)' if missing else NONE_RECORDED}")
    for reaction in report["reactions"]:
        if reaction["duplicate_of"]:
            others = ", ".join(str(n) for n in reaction["duplicate_of"])
            st.markdown(
                f":orange[**Reaction {reaction['number']}: {DUPLICATE_WARNING}** "
                f"(same substrates and products as reaction {others})]"
            )
        with st.expander(f"Reaction {reaction['number']}: {reaction['equation']}"):
            _block(reaction_lines(reaction))

    with st.expander(interactions_title(report)):
        if not report["interactions"]:
            st.write(NOT_RECORDED if report["payload_source"] == NOT_RECORDED else NONE_RECORDED)
        for item in report["interactions"]:
            _block(interaction_lines(item))

    quarantined = report["quarantined"]
    label = NOT_RECORDED if quarantined is None else str(len(quarantined))
    with st.expander(with_note(f"Quarantined / rejected candidate processes ({label})", report["quarantined_note"])):
        _block(report["quarantine_sources"])
        if not quarantined:
            st.write(NOT_RECORDED if quarantined is None else NONE_RECORDED)
        for number, row in enumerate(quarantined or [], 1):
            _block(quarantined_lines(number, row))
        if report["removed_entities"]:
            _block(["Removed entities:", *(f"- {e}" for e in report["removed_entities"])])
