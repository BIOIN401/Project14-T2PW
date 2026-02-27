import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import streamlit as st
from lxml import etree

from apply_audit_patch import run_apply
from audit_json_llm import run_audit
from grounding import apply_grounding
from json_to_sbml import build_sbml
from map_ids import run_mapping
from sbml_overwatch import run_sbml_overwatch
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


def run_post_pipeline_sbml_artifacts(
    final_payload: Dict[str, Any],
    *,
    use_llm_audit: bool,
    use_sbml_overwatch: bool,
    default_compartment: str,
    mapping_cache_path: str,
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    cache_path = Path(mapping_cache_path)
    if not cache_path.is_absolute():
        cache_path = project_root / mapping_cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        input_json = tmp / "final.json"
        audit_report_path = tmp / "audit_report.json"
        audit_patch_path = tmp / "audit_patch.json"
        apply_report_path = tmp / "audit_apply_report.json"
        audited_json = tmp / "final.audited.json"
        mapped_json = tmp / "final.mapped.json"
        mapping_report_path = tmp / "mapping_report.json"
        sbml_path = tmp / "pathway.sbml"
        sbml_report_json_path = tmp / "sbml_validation_report.json"
        sbml_report_txt_path = tmp / "sbml_validation_report.txt"
        sbml_overwatch_path = tmp / "sbml_overwatch_report.json"

        input_json.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        run_audit(
            input_json,
            audit_report_path,
            audit_patch_path,
            use_llm=use_llm_audit,
            llm_temperature=0.0,
            llm_max_tokens=3600,
        )
        run_apply(
            input_json,
            audit_patch_path,
            audited_json,
            audit_report_path=audit_report_path,
            apply_report_path=apply_report_path,
        )
        mapping_report = run_mapping(
            audited_json,
            mapped_json,
            mapping_report_path,
            cache_path=cache_path,
        )
        sbml_build_report = build_sbml(
            mapped_json,
            sbml_path,
            sbml_report_json_path,
            sbml_report_txt_path,
            default_compartment_name=default_compartment,
        )
        sbml_overwatch_report: Dict[str, Any] = {}
        if use_sbml_overwatch:
            sbml_overwatch_report = run_sbml_overwatch(
                mapped_json,
                sbml_path,
                sbml_report_json_path,
                sbml_overwatch_path,
                use_llm=True,
                llm_max_tokens=3000,
            )

        return {
            "audit_report": json.loads(audit_report_path.read_text(encoding="utf-8")),
            "audit_patch": json.loads(audit_patch_path.read_text(encoding="utf-8")),
            "audit_apply_report": json.loads(apply_report_path.read_text(encoding="utf-8")),
            "final_audited": json.loads(audited_json.read_text(encoding="utf-8")),
            "final_mapped": json.loads(mapped_json.read_text(encoding="utf-8")),
            "mapping_report": mapping_report,
            "sbml_report_json": json.loads(sbml_report_json_path.read_text(encoding="utf-8")),
            "sbml_report_txt": sbml_report_txt_path.read_text(encoding="utf-8"),
            "sbml_overwatch_report": sbml_overwatch_report,
            "sbml_xml_bytes": sbml_path.read_bytes(),
            "sbml_build_report": sbml_build_report,
            "mapping_cache_path": str(cache_path),
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
        value=1,
        step=1,
        help="Round 1 is normal inference. Additional rounds include graph QA feedback hints.",
    )

    col_tokens_1, col_tokens_2 = st.columns(2)
    extract_tokens = col_tokens_1.number_input(
        "Stage 1 max tokens",
        min_value=500,
        max_value=128000,
        value=24000,
        step=100,
    )
    infer_tokens = col_tokens_2.number_input(
        "Stage 2 max tokens",
        min_value=500,
        max_value=128000,
        value=20000,
        step=100,
    )

    chunk_cols = st.columns(2)
    chunk_size = chunk_cols[0].number_input(
        "Chunk size (approx. words)",
        min_value=200,
        max_value=60000,
        value=10000,
        step=100,
    )
    chunk_overlap = chunk_cols[1].number_input(
        "Chunk overlap (words)",
        min_value=0,
        max_value=20000,
        value=1600,
        step=100,
    )
    st.caption("Runtime scales with: chunks x Stage 2 QA rounds x retry attempts.")

    submit = st.form_submit_button("Run pipeline")

if submit:
    if not text.strip():
        st.warning("Paste some pathway text first.")
        st.stop()

    # Stage 1: strict extraction with auto-repair
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

    final_payload = stage_one
    stage_two = None
    stage_two_chunks: List[Dict[str, Any]] = []
    stage_two_rounds: List[Dict[str, Any]] = []
    qa_hints = None

    # Stage 2: inference/enrichment + auto-repair
    if run_inference:
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

        qa_hints = stage_two.get("qa_hints", {}) if isinstance(stage_two, dict) else {}
        final_payload = merge_additions(stage_one, stage_two if isinstance(stage_two, dict) else {})

    st.session_state["pipeline_ready"] = True
    st.session_state["run_inference_enabled"] = bool(run_inference)
    st.session_state["stage_one"] = stage_one
    st.session_state["chunk_details"] = chunk_details
    st.session_state["stage_two"] = stage_two
    st.session_state["stage_two_chunks"] = stage_two_chunks
    st.session_state["stage_two_rounds"] = stage_two_rounds
    st.session_state["qa_hints"] = qa_hints
    st.session_state["final_payload"] = final_payload
    st.session_state.pop("post_pipeline_artifacts", None)

if st.session_state.get("pipeline_ready"):
    run_inference_from_state = bool(st.session_state.get("run_inference_enabled", False))
    stage_one = st.session_state.get("stage_one", {})
    chunk_details = st.session_state.get("chunk_details", [])
    stage_two = st.session_state.get("stage_two")
    stage_two_chunks = st.session_state.get("stage_two_chunks", [])
    stage_two_rounds = st.session_state.get("stage_two_rounds", [])
    qa_hints = st.session_state.get("qa_hints")
    final_payload = st.session_state.get("final_payload", {})

    st.subheader("Stage 1 - Strict extraction")
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

    if run_inference_from_state and isinstance(stage_two, dict):
        st.subheader("Stage 2 - Inference / enrichment")
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

    st.subheader("Post-pipeline SBML export")
    post_col_a, post_col_b = st.columns(2)
    use_llm_audit = post_col_a.checkbox(
        "Use LLM in audit stage",
        value=True,
        help="Disabling runs deterministic audit rules only.",
        key="post_use_llm_audit",
    )
    use_sbml_overwatch = post_col_a.checkbox(
        "Use SBML semantic overwatch",
        value=True,
        help="Runs deterministic + LLM semantic review on generated SBML.",
        key="post_use_sbml_overwatch",
    )
    default_compartment = post_col_b.text_input(
        "Default compartment",
        value="cell",
        key="post_default_compartment",
        help="Used when location/state is missing.",
    )
    mapping_cache_text = st.text_input(
        "ID mapping cache path",
        value="id_mapping_cache.json",
        key="post_mapping_cache",
        help="Cache file for UniProt/compound mapping lookups.",
    )

    if st.button("Run post-pipeline SBML conversion"):
        try:
            with st.spinner("Running audit, patching, ID mapping, and SBML build..."):
                artifacts = run_post_pipeline_sbml_artifacts(
                    final_payload,
                    use_llm_audit=bool(use_llm_audit),
                    use_sbml_overwatch=bool(use_sbml_overwatch),
                    default_compartment=(default_compartment or "cell").strip() or "cell",
                    mapping_cache_path=mapping_cache_text.strip() or "id_mapping_cache.json",
                )
            st.session_state["post_pipeline_artifacts"] = artifacts
            st.success("Post-pipeline conversion completed.")
        except Exception as exc:
            st.error(f"Post-pipeline conversion failed: {exc}")

    post_artifacts = st.session_state.get("post_pipeline_artifacts")
    if isinstance(post_artifacts, dict):
        audit_summary = post_artifacts.get("audit_report", {}).get("summary", {})
        mapping_summary = post_artifacts.get("mapping_report", {}).get("summary", {})
        sbml_summary = post_artifacts.get("sbml_report_json", {}).get("counts", {})
        sbml_validation = post_artifacts.get("sbml_report_json", {}).get("validation", {})
        sbml_overwatch_summary = post_artifacts.get("sbml_overwatch_report", {}).get("summary", {})

        st.write(
            {
                "audit": audit_summary,
                "mapping": mapping_summary,
                "sbml_counts": sbml_summary,
                "sbml_validation_has_errors": sbml_validation.get("has_errors"),
                "sbml_overwatch": sbml_overwatch_summary,
                "mapping_cache_path": post_artifacts.get("mapping_cache_path"),
            }
        )

        st.download_button(
            "Download audit_report.json",
            json.dumps(post_artifacts["audit_report"], indent=2),
            file_name="audit_report.json",
            mime="application/json",
            key="dl_audit_report",
        )
        st.download_button(
            "Download audit_patch.json",
            json.dumps(post_artifacts["audit_patch"], indent=2),
            file_name="audit_patch.json",
            mime="application/json",
            key="dl_audit_patch",
        )
        st.download_button(
            "Download audit_apply_report.json",
            json.dumps(post_artifacts["audit_apply_report"], indent=2),
            file_name="audit_apply_report.json",
            mime="application/json",
            key="dl_audit_apply",
        )
        st.download_button(
            "Download final.audited.json",
            json.dumps(post_artifacts["final_audited"], indent=2),
            file_name="final.audited.json",
            mime="application/json",
            key="dl_final_audited",
        )
        st.download_button(
            "Download final.mapped.json",
            json.dumps(post_artifacts["final_mapped"], indent=2),
            file_name="final.mapped.json",
            mime="application/json",
            key="dl_final_mapped",
        )
        st.download_button(
            "Download mapping_report.json",
            json.dumps(post_artifacts["mapping_report"], indent=2),
            file_name="mapping_report.json",
            mime="application/json",
            key="dl_mapping_report",
        )
        st.download_button(
            "Download pathway.sbml",
            post_artifacts["sbml_xml_bytes"],
            file_name="pathway.sbml",
            mime="application/xml",
            key="dl_pathway_sbml",
        )
        st.download_button(
            "Download sbml_validation_report.json",
            json.dumps(post_artifacts["sbml_report_json"], indent=2),
            file_name="sbml_validation_report.json",
            mime="application/json",
            key="dl_sbml_json",
        )
        st.download_button(
            "Download sbml_validation_report.txt",
            post_artifacts["sbml_report_txt"],
            file_name="sbml_validation_report.txt",
            mime="text/plain",
            key="dl_sbml_txt",
        )
        if post_artifacts.get("sbml_overwatch_report"):
            st.download_button(
                "Download sbml_overwatch_report.json",
                json.dumps(post_artifacts["sbml_overwatch_report"], indent=2),
                file_name="sbml_overwatch_report.json",
                mime="application/json",
                key="dl_sbml_overwatch",
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
    if run_inference_from_state:
        st.write("Connectivity repair hints used for later rounds", build_qa_feedback(final_payload))
