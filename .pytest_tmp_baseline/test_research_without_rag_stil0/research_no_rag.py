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
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {'entities': {'compounds': [{'name': 'menaquinone'}], 'proteins': [{'name': 'MenG'}]}, 'processes': {'reactions': [{'name': 'menaquinone demethylation'}]}}
    st.session_state["final_payload"] = {'entities': {'compounds': [{'name': 'menaquinone'}, {'name': 'demethylmenaquinone'}], 'proteins': [{'name': 'MenG'}]}, 'processes': {'reactions': [{'name': 'menaquinone demethylation', 'inputs': ['menaquinone'], 'outputs': ['demethylmenaquinone']}]}}

if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        st.session_state["post_pipeline_artifacts"] = {'gate_failed': False, 'normalization_gate_failed': False, 'gate_fail_report': {}, 'export_mode': 'research', 'research_review_flags': [{'code': 'reaction_enzyme_unresolved', 'pointer': '/processes/reactions/0/enzymes/0', 'message': 'enzyme has no external identity', 'research_category': 'biology'}], 'research_skipped_format_rules': [], 'research_normalization_actions': [], 'final_mapped_db': {'entities': {'compounds': [{'name': 'menaquinone'}, {'name': 'demethylmenaquinone'}], 'proteins': [{'name': 'MenG'}]}, 'processes': {'reactions': [{'name': 'menaquinone demethylation', 'inputs': ['menaquinone'], 'outputs': ['demethylmenaquinone']}]}}, 'final_mapped': {'entities': {'compounds': [{'name': 'menaquinone'}, {'name': 'demethylmenaquinone'}], 'proteins': [{'name': 'MenG'}]}, 'processes': {'reactions': [{'name': 'menaquinone demethylation', 'inputs': ['menaquinone'], 'outputs': ['demethylmenaquinone']}]}}, 'mapping_report': {'summary': {'mapped': 3}}, 'post_extraction_contract_report': {'ok': True, 'errors': [], 'warnings': []}}

