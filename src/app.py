import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import streamlit as st
from lxml import etree

from grounding import apply_grounding
from pipeline import (
    PipelineFailure,
    build_qa_feedback,
    merge_additions,
    run_stage_two_with_feedback_loop,
    run_stage_one_with_chunking,
)
from pwml_validate import discover_structure_signature, repair_tree, validate_generated_tree
from pwml_writer import DeterministicPwmlBuilder
from qa_graph import build_graph, connected_components

st.set_page_config(page_title="PWML Multi-Stage Pipeline", layout="wide")
st.title("PWML Extraction -> Inference Pipeline (LM Studio)")


def render_attempts(label: str, attempts: List[Dict[str, Any]]) -> None:
    with st.expander(label, expanded=False):
        for log in attempts:
            status = "success" if not log.get("error") else "retry"
            phase = log.get("phase")
            phase_label = f" ({phase})" if phase else ""
            note = log.get("note")
            note_label = f" [{note}]" if note else ""
            st.markdown(f"**Attempt {log['attempt']}{phase_label}** - {status}{note_label}")
            st.code(log["raw"], language="json")
            if log.get("error"):
                st.caption(log["error"])


def graph_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    adj, meta = build_graph(payload)
    comps = connected_components(adj)
    return {
        **meta,
        "n_nodes": len(adj),
        "n_components": len(comps),
        "main_component_size": max((len(c) for c in comps), default=0),
    }


def qa_summary_line(payload: Dict[str, Any]) -> str:
    stats = graph_summary(payload)
    orphans = max(stats["n_components"] - 1, 0)
    return (
        f"Components: {stats['n_components']} | "
        f"Main size: {stats['main_component_size']} | "
        f"Orphans: {orphans}"
    )


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.exists():
        return candidate
    project_root = Path(__file__).resolve().parent.parent
    rooted = project_root / path_text
    if rooted.exists():
        return rooted
    return candidate


with st.form("pwml_pipeline"):
    text = st.text_area("Paste pathway description:", height=220)
    run_inference = st.checkbox(
        "Run inference/enrichment stage",
        value=True,
        help="Stage 1 always runs. Disable when you only want strict extraction.",
    )

    enable_chunking = st.checkbox(
        "Enable automatic chunking for long inputs",
        value=True,
        help="When enabled, Stage 1 splits long inputs into overlapping chunks before extraction.",
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    extract_attempts = col_a.number_input(
        "Stage 1 auto-repair attempts",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
    )
    infer_attempts = col_b.number_input(
        "Stage 2 auto-repair attempts",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
    )
    temperature = col_c.slider(
        "LLM temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Applied to both stages.",
    )
    infer_rounds = col_d.number_input(
        "Stage 2 QA rounds",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Round 1 is normal inference. Additional rounds include graph QA feedback hints.",
    )

    col_tokens_1, col_tokens_2 = st.columns(2)
    extract_tokens = col_tokens_1.number_input(
        "Stage 1 max tokens",
        min_value=500,
        max_value=4000,
        value=2000,
        step=100,
    )
    infer_tokens = col_tokens_2.number_input(
        "Stage 2 max tokens",
        min_value=500,
        max_value=4000,
        value=2000,
        step=100,
    )

    chunk_cols = st.columns(2)
    chunk_size = chunk_cols[0].number_input(
        "Chunk size (approx. words)",
        min_value=200,
        max_value=3000,
        value=1200,
        step=100,
    )
    chunk_overlap = chunk_cols[1].number_input(
        "Chunk overlap (words)",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
    )

    submit = st.form_submit_button("Run pipeline")

if submit:
    if not text.strip():
        st.warning("Paste some pathway text first.")
        st.stop()

    # Stage 1: strict extraction with auto-repair
    st.subheader("Stage 1 - Strict extraction")
    try:
        with st.spinner("Running Stage 1 extraction..."):
            stage_one, chunk_details = run_stage_one_with_chunking(
                text,
                enable_chunking=enable_chunking,
                chunk_word_limit=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                max_attempts=int(extract_attempts),
                temperature=temperature,
                max_tokens=int(extract_tokens),
            )
    except PipelineFailure as failure:
        st.error(f"Extraction failed: {failure}")
        render_attempts("Stage 1 attempts", failure.attempts)
        st.stop()

    st.json(stage_one)
    st.caption(f"Stage 1 QA: {qa_summary_line(stage_one)}")
    st.download_button(
        "Download Stage 1 JSON",
        json.dumps(stage_one, indent=2),
        file_name="stage1_extract.json",
        mime="application/json",
    )

    chunk_count = len(chunk_details)
    if chunk_count > 1:
        st.info(f"Chunked input into {chunk_count} slices (~{int(chunk_size)} words, overlap {int(chunk_overlap)}).")

    for chunk in chunk_details:
        chunk_label = f"Chunk {chunk['chunk_id']} - words {chunk['start_word']}-{chunk['end_word']}"
        with st.expander(chunk_label, expanded=False):
            preview = chunk["text"][:400]
            if len(chunk["text"]) > 400:
                preview += "..."
            st.caption(preview)
            st.markdown("**Chunk output JSON**")
            st.json(chunk["output"])
        render_attempts(f"{chunk_label} attempts", chunk["attempts"])

    final_payload = stage_one
    qa_hints = None
    stage_two_chunks: List[Dict[str, Any]] = []

    # Stage 2: inference/enrichment + auto-repair
    if run_inference:
        st.subheader("Stage 2 - Inference / enrichment")
        try:
            with st.spinner("Running Stage 2 inference..."):
                stage_two, stage_two_chunks, stage_two_rounds = run_stage_two_with_feedback_loop(
                    text,
                    stage_one,
                    chunk_details=chunk_details,
                    qa_rounds=int(infer_rounds),
                    enable_chunking=enable_chunking,
                    chunk_word_limit=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    max_attempts=int(infer_attempts),
                    temperature=temperature,
                    max_tokens=int(infer_tokens),
                )
        except PipelineFailure as failure:
            st.error(f"Inference stage failed: {failure}")
            label = f"{failure.stage} attempts" if getattr(failure, "stage", None) else "Stage 2 attempts"
            render_attempts(label, failure.attempts)
            st.stop()

        st.json(stage_two)
        if stage_two_rounds:
            st.write("Stage 2 QA rounds", stage_two_rounds)
        chunk_count = len(stage_two_chunks)
        if chunk_count > 1:
            st.info(
                f"Chunked inference into {chunk_count} slices (~{int(chunk_size)} words, overlap {int(chunk_overlap)})."
            )

        for chunk in stage_two_chunks:
            chunk_label = (
                f"Round {chunk.get('qa_round', 1)} - "
                f"Chunk {chunk['chunk_id']} - words {chunk['start_word']}-{chunk['end_word']}"
            )
            with st.expander(chunk_label, expanded=False):
                preview = chunk["text"][:400]
                if len(chunk["text"]) > 400:
                    preview += "..."
                st.caption(preview)
                st.markdown("**Chunk additions JSON**")
                st.json(chunk["output"])
            render_attempts(f"{chunk_label} attempts", chunk["attempts"])
        qa_hints = stage_two.get("qa_hints", {})
        final_payload = merge_additions(stage_one, stage_two)

        st.download_button(
            "Download Stage 2 additions",
            json.dumps(stage_two, indent=2),
            file_name="stage2_additions.json",
            mime="application/json",
        )

        if qa_hints:
            st.write("QA hints", qa_hints)

    st.subheader("Final merged output")
    st.json(final_payload)
    st.caption(f"Final QA: {qa_summary_line(final_payload)}")
    st.download_button(
        "Download merged JSON",
        json.dumps(final_payload, indent=2),
        file_name="pwml_pipeline_output.json",
        mime="application/json",
    )

    st.subheader("PWML export")
    pwml_col_a, pwml_col_b = st.columns(2)
    ref_path_text = pwml_col_a.text_input("Reference PWML path", value="PW012926.pwml")
    grounding_enabled = pwml_col_b.checkbox("Apply deterministic grounding before PWML export", value=False)

    grounding_path_text = ""
    if grounding_enabled:
        grounding_path_text = st.text_input("Grounding dictionary path", value="grounding_dictionary.example.json")

    meta_col_1, meta_col_2, meta_col_3 = st.columns(3)
    pw_id = meta_col_1.text_input("pw-id", value="PW000XYZ")
    named_for_id = meta_col_2.number_input("named-for-id", min_value=1, max_value=99999999, value=123, step=1)
    subject = meta_col_3.text_input("subject", value="Metabolic")

    naming_col_1, naming_col_2 = st.columns(2)
    pathway_name = naming_col_1.text_input("Pathway name", value="Generated Pathway")
    pathway_description = naming_col_2.text_input("Pathway description", value="")

    vis_col_1, vis_col_2, vis_col_3 = st.columns(3)
    vis_width = vis_col_1.number_input("Visualization width", min_value=200, max_value=10000, value=3200, step=100)
    vis_height = vis_col_2.number_input("Visualization height", min_value=200, max_value=10000, value=1400, step=100)
    background_color = vis_col_3.text_input("Background color", value="#FFFFFF")

    if st.button("Generate PWML from final JSON"):
        payload_for_writer = final_payload
        grounding_report: Dict[str, Any] = {}

        if grounding_enabled:
            try:
                grounding_path = resolve_path(grounding_path_text)
                grounding_dict = json.loads(grounding_path.read_text(encoding="utf-8"))
                if not isinstance(grounding_dict, dict):
                    raise ValueError("Grounding dictionary must be a JSON object.")
                payload_for_writer, grounding_report = apply_grounding(payload_for_writer, grounding_dict)
                st.write("Grounding report", grounding_report)
                st.download_button(
                    "Download grounded JSON",
                    json.dumps(payload_for_writer, indent=2),
                    file_name="grounded_output.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"Grounding failed: {exc}")
                st.stop()

        try:
            reference_path = resolve_path(ref_path_text)
            signature = discover_structure_signature(reference_path)
            args = SimpleNamespace(
                named_for_id=int(named_for_id),
                name=pathway_name,
                description=pathway_description,
                subject=subject,
                pw_id=pw_id,
                height=int(vis_height),
                width=int(vis_width),
                background_color=background_color,
            )
            builder = DeterministicPwmlBuilder(payload_for_writer, signature, args)
            build = builder.build()
            tree = etree.ElementTree(build.root)
            repaired = repair_tree(tree, signature)
            report = validate_generated_tree(repaired, signature)
            xml_bytes = etree.tostring(
                repaired.getroot(), encoding="utf-8", xml_declaration=True, pretty_print=True
            )

            st.write("PWML generation summary", build.counts)
            st.write("Dummy geometry generated", build.geometry_generated)
            st.write("PWML structural validation", {"ok": report["ok"], "issue_count": report["issue_count"]})
            if not report["ok"]:
                st.json(report["issues"])

            st.download_button(
                "Download PWML",
                data=xml_bytes,
                file_name="out.pwml",
                mime="application/xml",
            )
            st.download_button(
                "Download PWML validation report",
                data=json.dumps(report, indent=2),
                file_name="writer_validation_report.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"PWML generation failed: {exc}")

    st.subheader("Connectivity snapshot")
    stats = graph_summary(final_payload)
    st.write(stats)
    if run_inference:
        st.write("Connectivity repair hints used for later rounds", build_qa_feedback(final_payload))
