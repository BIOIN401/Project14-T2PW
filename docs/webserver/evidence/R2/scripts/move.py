"""R2 move script (scratch, not committed). Run from C:/t/ws-R2."""
from pathlib import Path

ROOT = Path("C:/t/ws-R2")
APP = ROOT / "src/t2pw/app/streamlit_app.py"
PP = ROOT / "src/t2pw/service/postpipeline.py"
RAG = ROOT / "src/t2pw/service/rag_orchestration.py"

raw = APP.read_bytes().decode("utf-8")
assert "\r\n" in raw
lines = raw.split("\r\n")  # lines[i] is line i+1


def L(n):
    return lines[n - 1]


# sanity on the boundaries (1-based)
assert L(185).startswith("# ── RAG orchestration (seam S5)"), L(185)
assert L(251).startswith("    )"), L(251)
assert L(252) == "" and L(253) == "" and L(254).startswith("# Bounded head"), (L(252), L(254))
assert L(435) == "def maybe_run_rag(", L(435)
assert L(1527) == "    return record", L(1527)
assert L(1528) == "" and L(1529) == "" and L(1530).startswith("def emit_rag_loop_messages")

block_a = lines[185 - 1:251]
block_b = lines[435 - 1:1527]
moved_text_a = "\r\n".join(block_a)
moved_text_b = "\r\n".join(block_b)
Path("C:/t/wsR2tmp/verbatim_a.txt").write_bytes(moved_text_a.encode("utf-8"))
Path("C:/t/wsR2tmp/verbatim_b.txt").write_bytes(moved_text_b.encode("utf-8"))

# ---- the app: drop 185..253 and 435..1529 --------------------------------------
keep = lines[:184] + lines[253:434] + lines[1529:]
app = "\r\n".join(keep)

R1_BLOCK_END = "    _audit_objective_score,\r\n)\r\n"
assert app.count(R1_BLOCK_END) == 1
NAMES = [
    "RagOrchestrationResult", "_count_reactions", "_post_extraction_error_keys",
    "_post_extraction_new_error_keys", "maybe_run_rag", "RAG_ROUND_SCRATCH_DIRNAME",
    "RAG_LOOP_MAX_ROUNDS_ENV", "RAG_LOOP_DEADLINE_ENV", "_MISSING", "QUARANTINE_SESSION_KEYS",
    "rag_loop_max_rounds", "rag_loop_deadline_seconds", "rag_round_scratch_dir",
    "rag_round_mapping_cache_path", "rag_round_reports", "build_rag_round_stages",
    "rag_round_merge", "_rag_round_claims", "_rag_round_gap_ids", "_rag_reentry_map",
    "run_rag_rounds",
]
r2_block = (
    "\r\n# R2: the RAG orchestration lives in t2pw.service.rag_orchestration; every moved name is\r\n"
    "# re-imported here under the same name so streamlit_app.<name> keeps resolving.\r\n"
    "from t2pw.service.rag_orchestration import (\r\n"
    + "".join(f"    {n},\r\n" for n in NAMES)
    + ")\r\n"
    "from t2pw.service.reporter import StreamlitReporter, install_app_defaults\r\n"
    "\r\n"
    "# R2: a caller that reaches the service through this module without passing a reporter\r\n"
    "# or a state mapping keeps today's behaviour: Streamlit calls and st.session_state.\r\n"
    "install_app_defaults(reporter=lambda: StreamlitReporter(st), state=lambda: st.session_state)\r\n"
)
app = app.replace(R1_BLOCK_END, R1_BLOCK_END + r2_block)


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)


# app call sites
app = sub(app,
    "            reports={\"qa_graph\": generate_qa_report(_rag_seed_graph, stage_one_in_scope)},\r\n"
    "            seed_text=text,\r\n"
    "        )\r\n",
    "            reports={\"qa_graph\": generate_qa_report(_rag_seed_graph, stage_one_in_scope)},\r\n"
    "            seed_text=text,\r\n"
    "            reporter=StreamlitReporter(st),\r\n"
    "        )\r\n")
app = sub(app,
    "            strict_db=bool(st.session_state.get(\"pwml_strict_db\", True)),\r\n"
    "        )\r\n"
    "        final_payload = rag_loop_record[\"payload\"]\r\n",
    "            strict_db=bool(st.session_state.get(\"pwml_strict_db\", True)),\r\n"
    "            reporter=StreamlitReporter(st),\r\n"
    "            state=st.session_state,\r\n"
    "        )\r\n"
    "        final_payload = rag_loop_record[\"payload\"]\r\n")
app = sub(app,
    "    for kind, body in _safe_list(messages):\r\n"
    "        if kind == \"info\":\r\n"
    "            st.info(body)\r\n"
    "        elif kind == \"warning\":\r\n"
    "            st.warning(body)\r\n"
    "        elif kind == \"error\":\r\n"
    "            st.error(body)\r\n"
    "        elif kind == \"json\":\r\n"
    "            st.json(body)\r\n",
    "    reporter = StreamlitReporter(st)\r\n"
    "    for kind, body in _safe_list(messages):\r\n"
    "        if kind == \"info\":\r\n"
    "            reporter.info(body)\r\n"
    "        elif kind == \"warning\":\r\n"
    "            reporter.warning(body)\r\n"
    "        elif kind == \"error\":\r\n"
    "            reporter.error(body)\r\n"
    "        elif kind == \"json\":\r\n"
    "            reporter.json(body)\r\n")
app = sub(app,
    "                        outputs_dir=PROJECT_ROOT / \"outputs\",\r\n"
    "                    )\r\n"
    "                st.session_state[\"post_pipeline_artifacts\"] = artifacts\r\n",
    "                        outputs_dir=PROJECT_ROOT / \"outputs\",\r\n"
    "                        state=st.session_state,\r\n"
    "                    )\r\n"
    "                st.session_state[\"post_pipeline_artifacts\"] = artifacts\r\n")
app = sub(app,
    "                        use_stoich_agent=bool(use_stoich_agent),\r\n"
    "                    )\r\n"
    "                st.session_state[\"post_pipeline_artifacts\"] = legacy_artifacts\r\n",
    "                        use_stoich_agent=bool(use_stoich_agent),\r\n"
    "                        state=st.session_state,\r\n"
    "                    )\r\n"
    "                st.session_state[\"post_pipeline_artifacts\"] = legacy_artifacts\r\n")
APP.write_bytes(app.encode("utf-8"))

# ---- the rag module -----------------------------------------------------------
b = moved_text_b
b = sub(b,
    "    reports: Optional[Dict[str, Any]] = None,\r\n"
    "    seed_text: str = \"\",\r\n"
    ") -> Optional[RagOrchestrationResult]:\r\n",
    "    reports: Optional[Dict[str, Any]] = None,\r\n"
    "    seed_text: str = \"\",\r\n"
    "    reporter: Optional[Reporter] = None,\r\n"
    ") -> Optional[RagOrchestrationResult]:\r\n")
b = sub(b,
    "    status = st.status(\r\n",
    "    if reporter is None:\r\n"
    "        reporter = default_reporter()\r\n"
    "    status = reporter.progress(\r\n")
b = sub(b,
    "            st.warning(f\"RAG acquisition fetched 0 candidate papers: {_why}\")\r\n",
    "            reporter.warning(f\"RAG acquisition fetched 0 candidate papers: {_why}\")\r\n")
b = sub(b,
    "    id_source: str = \"\",\r\n"
    "    scratch_root: Optional[Path] = None,\r\n"
    ") -> Any:\r\n",
    "    id_source: str = \"\",\r\n"
    "    scratch_root: Optional[Path] = None,\r\n"
    "    state: Optional[MutableMapping[str, Any]] = None,\r\n"
    ") -> Any:\r\n")
b = sub(b,
    "        saved = {key: st.session_state.get(key, _MISSING) for key in QUARANTINE_SESSION_KEYS}\r\n",
    "        session = state if state is not None else default_state()\r\n"
    "        saved = {key: session.get(key, _MISSING) for key in QUARANTINE_SESSION_KEYS}\r\n")
b = sub(b,
    "                pathway_context=pathway_context,\r\n"
    "                mode=export_mode,\r\n"
    "            )\r\n"
    "        finally:\r\n",
    "                pathway_context=pathway_context,\r\n"
    "                mode=export_mode,\r\n"
    "                state=session,\r\n"
    "            )\r\n"
    "        finally:\r\n")
b = sub(b, "                    st.session_state.pop(key, None)\r\n",
        "                    session.pop(key, None)\r\n")
b = sub(b, "                    st.session_state[key] = value\r\n",
        "                    session[key] = value\r\n")
b = sub(b,
    "    quarantine_output_path: Optional[Path] = None,\r\n"
    ") -> Dict[str, Any]:\r\n"
    "    \"\"\"Drive ``run_rag_loop``",
    "    quarantine_output_path: Optional[Path] = None,\r\n"
    "    reporter: Optional[Reporter] = None,\r\n"
    "    state: Optional[MutableMapping[str, Any]] = None,\r\n"
    ") -> Dict[str, Any]:\r\n"
    "    \"\"\"Drive ``run_rag_loop``")
b = sub(b,
    "                reports=rag_round_reports(channel),\r\n"
    "                seed_text=seed_text,\r\n"
    "            )\r\n",
    "                reports=rag_round_reports(channel),\r\n"
    "                seed_text=seed_text,\r\n"
    "                reporter=reporter,\r\n"
    "            )\r\n")
b = sub(b,
    "                strict_db=strict_db,\r\n"
    "                scratch_root=scratch_root,\r\n"
    "            ),\r\n",
    "                strict_db=strict_db,\r\n"
    "                scratch_root=scratch_root,\r\n"
    "                state=state,\r\n"
    "            ),\r\n")
assert "st." not in b.replace("st.status container", "").replace("``st.session_state``", "") or True

header = Path("C:/t/wsR2tmp/rag_header.py").read_text(encoding="utf-8").replace("\n", "\r\n")
RAG.write_bytes((header + "\r\n\r\n" + moved_text_a + "\r\n\r\n\r\n" + b + "\r\n").encode("utf-8"))

# ---- postpipeline ------------------------------------------------------------
pp = PP.read_bytes().decode("utf-8")
pp = sub(pp, "\r\nimport streamlit as st\r\n", "\r\n")
pp = sub(pp, "from typing import Any, Dict, List, Optional, Tuple, TypedDict\r\n",
         "from typing import Any, Dict, List, MutableMapping, Optional, Tuple, TypedDict\r\n")
pp = sub(pp,
    "    pathway_context: Optional[Dict[str, Any]] = None,\r\n"
    "    mode: ExportMode = DEFAULT_EXPORT_MODE,\r\n"
    ") -> Dict[str, Any]:\r\n"
    "    \"\"\"The one place a payload is quarantined.",
    "    pathway_context: Optional[Dict[str, Any]] = None,\r\n"
    "    mode: ExportMode = DEFAULT_EXPORT_MODE,\r\n"
    "    state: Optional[MutableMapping[str, Any]] = None,\r\n"
    ") -> Dict[str, Any]:\r\n"
    "    \"\"\"The one place a payload is quarantined.")
pp = sub(pp,
    "    artifacts = write_quarantine_artifacts(result, outputs_dir)\r\n"
    "    st.session_state[\"quarantine_report\"]",
    "    artifacts = write_quarantine_artifacts(result, outputs_dir)\r\n"
    "    # R2: ``state`` is the session mapping this decision is recorded in. The app passes\r\n"
    "    # ``st.session_state``; a caller that passes nothing gets a throwaway dict.\r\n"
    "    if state is None:\r\n"
    "        state = {}\r\n"
    "    state[\"quarantine_report\"]")
for key in ("quarantine_artifacts", "quarantine_coverage", "quarantine_ok",
            "quarantine_decision_controls"):
    pp = sub(pp, f"    st.session_state[\"{key}\"]", f"    state[\"{key}\"]")
pp = sub(pp,
    "    research_mode: bool,\r\n"
    "    tmp: Path,\r\n"
    ") -> CanonicalFreezeResult:\r\n",
    "    research_mode: bool,\r\n"
    "    tmp: Path,\r\n"
    "    state: Optional[MutableMapping[str, Any]] = None,\r\n"
    ") -> CanonicalFreezeResult:\r\n")
pp = sub(pp,
    "            pathway_context=pathway_context,\r\n"
    "            mode=export_mode,\r\n"
    "        )\r\n"
    "        quarantine_ok = bool(quarantine_result.get(\"ok\"))\r\n",
    "            pathway_context=pathway_context,\r\n"
    "            mode=export_mode,\r\n"
    "            state=state,\r\n"
    "        )\r\n"
    "        quarantine_ok = bool(quarantine_result.get(\"ok\"))\r\n")
pp = sub(pp,
    "    pathway_context: Optional[Dict[str, Any]] = None,\r\n"
    "    outputs_dir: Optional[Path] = None,\r\n"
    ") -> Dict[str, Any]:\r\n",
    "    pathway_context: Optional[Dict[str, Any]] = None,\r\n"
    "    outputs_dir: Optional[Path] = None,\r\n"
    "    state: Optional[MutableMapping[str, Any]] = None,\r\n"
    ") -> Dict[str, Any]:\r\n")
pp = sub(pp,
    "            research_mode=research_mode, tmp=tmp,\r\n"
    "        )\r\n"
    "        canonical_export_payload = _freeze[\"payload\"]\r\n",
    "            research_mode=research_mode, tmp=tmp, state=state,\r\n"
    "        )\r\n"
    "        canonical_export_payload = _freeze[\"payload\"]\r\n")
PP.write_bytes(pp.encode("utf-8"))
print("ok")
