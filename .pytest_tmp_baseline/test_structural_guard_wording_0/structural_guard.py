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
    st.error("Extraction boundary failed: Payload must include a processes object.")
    st.warning('Stage 0 failed: empty_reply - The model returned an empty reply (0 chars). Re-run (usually a transient LLM failure), or type the pathway/organism into the focus box so the pipeline can proceed without the preprocessor.')
    st.stop()
