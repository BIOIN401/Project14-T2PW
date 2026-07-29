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

    st.session_state["rag_result"] = SimpleNamespace(
        payload={'metadata': {'pathway_name': 'Menaquinone biosynthesis'}, 'entities': {'compounds': [{'name': 'menaquinone', 'rag_provenance': {'source_id': 'PMC1', 'source_title': 'Menaquinone study', 'source_type': 'paper', 'source_uri': 'https://example.org/PMC1', 'section': 'results', 'chunk_id': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}}], 'proteins': [{'name': 'MenG', 'rag_provenance': {'source_id': 'PMC1', 'source_title': 'Menaquinone study', 'source_type': 'paper', 'source_uri': 'https://example.org/PMC1', 'section': 'results', 'chunk_id': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}}]}, 'processes': {'reactions': [{'name': 'menaquinone demethylation', 'inputs': ['menaquinone'], 'outputs': ['demethylmenaquinone'], 'rag_provenance': {'source_id': 'PMC1', 'source_title': 'Menaquinone study', 'source_type': 'paper', 'source_uri': 'https://example.org/PMC1', 'section': 'results', 'chunk_id': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}, 'source_refs': ['PMC1'], 'source_papers': [{'source_id': 'PMC1', 'title': 'Menaquinone study', 'uri': 'https://example.org/PMC1', 'source_type': 'paper'}], 'rag_confidence': 0.83}]}},
        candidates=[],
        selection=None,
        evidence_context="PMC1: MenG demethylates menaquinone.",
    )

if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        st.session_state["post_pipeline_artifacts"] = {'gate_failed': False, 'normalization_gate_failed': False, 'gate_fail_report': {}, 'export_mode': 'research', 'research_review_flags': [], 'research_skipped_format_rules': [], 'research_normalization_actions': [], 'final_mapped_db': {'entities': {'compounds': [{'name': 'menaquinone'}, {'name': 'demethylmenaquinone'}], 'proteins': [{'name': 'MenG'}]}, 'processes': {'reactions': [{'name': 'menaquinone demethylation', 'inputs': ['menaquinone'], 'outputs': ['demethylmenaquinone']}]}}, 'final_mapped': {'entities': {'compounds': [{'name': 'menaquinone'}, {'name': 'demethylmenaquinone'}], 'proteins': [{'name': 'MenG'}]}, 'processes': {'reactions': [{'name': 'menaquinone demethylation', 'inputs': ['menaquinone'], 'outputs': ['demethylmenaquinone']}]}}, 'mapping_report': {'summary': {'mapped': 3}}, 'post_extraction_contract_report': {'ok': True, 'errors': [], 'warnings': []}}

