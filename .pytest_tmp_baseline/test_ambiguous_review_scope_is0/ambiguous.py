import streamlit as st
from types import SimpleNamespace

st.radio("Export mode", ["Strict PWML", "Research (relaxed)"], key="export_mode_radio")
st.radio("Input mode", ["Paste text", "Upload PDF", "Text + PDF"], key="input_mode_radio")
with st.form("pwml_pipeline"):
    st.text_area("Paste pathway description:", key="pathway_text_0")
    user_task_context = st.text_area(
        "Optional extraction focus / task context",
        height=100,
        help="Use this to tell the model what pathway or scope you want extracted.",
    )
    st.session_state["observed_focus"] = user_task_context
    submitted = st.form_submit_button("Run pipeline")

if submitted:
    st.session_state["pipeline_ready"] = False
    st.session_state["pipeline_error"] = {
        "status": "ambiguous_review_scope",
        "message": "multi_example_review detected with no selected_example.",
        "candidate_examples": ["menaquinone biosynthesis", "heme biosynthesis"],
    }
    st.error("Ambiguous review scope: name the example you want, then re-run.")
    st.stop()
