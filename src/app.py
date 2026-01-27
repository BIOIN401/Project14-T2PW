import json
from typing import Any, Dict, List

import streamlit as st

from pipeline import (
    PipelineFailure,
    merge_additions,
    run_inference_pipeline,
    run_stage_one_with_chunking,
)
from qa_graph import build_graph, connected_components

st.set_page_config(page_title="PWML Multi-Stage Pipeline", layout="wide")
st.title("PWML Extraction → Inference Pipeline (LM Studio)")


def render_attempts(label: str, attempts: List[Dict[str, Any]]) -> None:
    with st.expander(label, expanded=False):
        for log in attempts:
            status = "✅ success" if not log.get("error") else "⚠️ retry"
            st.markdown(f"**Attempt {log['attempt']}** — {status}")
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


with st.form("pwml_pipeline"):
    text = st.text_area("Paste pathway description:", height=220)
    run_inference = st.checkbox(
        "Run inference/enrichment stage",
        value=True,
        help="Stage 1 always runs. Disable when you only want strict extraction.",
    )

    enable_chunking = st.checkbox(
        "Enable automatic chunking for long inputs",
        value=False,
        help="When enabled, Stage 1 splits long inputs into overlapping chunks before extraction.",
    )

    col_a, col_b, col_c = st.columns(3)
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
        disabled=not enable_chunking,
    )
    chunk_overlap = chunk_cols[1].number_input(
        "Chunk overlap (words)",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        disabled=not enable_chunking,
    )

    submit = st.form_submit_button("Run pipeline")

if submit:
    if not text.strip():
        st.warning("Paste some pathway text first.")
        st.stop()

    # Stage 1: strict extraction with auto-repair
    st.subheader("Stage 1 · Strict extraction")
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
        chunk_label = f"Chunk {chunk['chunk_id']} · words {chunk['start_word']}–{chunk['end_word']}"
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
    stage_two_attempts: List[Dict[str, Any]] = []

    # Stage 2: inference/enrichment + auto-repair
    if run_inference:
        st.subheader("Stage 2 · Inference / enrichment")
        try:
            with st.spinner("Running Stage 2 inference..."):
                stage_two, stage_two_attempts = run_inference_pipeline(
                    text,
                    stage_one,
                    max_attempts=int(infer_attempts),
                    temperature=temperature,
                    max_tokens=int(infer_tokens),
                )
        except PipelineFailure as failure:
            st.error(f"Inference stage failed: {failure}")
            render_attempts("Stage 2 attempts", failure.attempts)
            st.stop()

        st.json(stage_two)
        render_attempts("Stage 2 attempts", stage_two_attempts)
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
    st.download_button(
        "Download merged JSON",
        json.dumps(final_payload, indent=2),
        file_name="pwml_pipeline_output.json",
        mime="application/json",
    )

    st.subheader("Connectivity snapshot")
    stats = graph_summary(final_payload)
    st.write(stats)
