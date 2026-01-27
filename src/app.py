import json
from pathlib import Path

import streamlit as st

from llm_client import chat
from qa_graph import build_graph, connected_components, degrees

BASE_DIR = Path(__file__).resolve().parent
SYSTEM = (BASE_DIR / "prompts" / "pwml_system.txt").read_text(encoding="utf-8")

st.set_page_config(page_title="PWML Extractor", layout="wide")
st.title("PWML Extractor (LM Studio)")

text = st.text_area("Paste pathway text:", height=220)

col1, col2 = st.columns(2)
run = col1.button("Run extraction")
show_raw = col2.checkbox("Show raw model output", value=True)


def one_shot_extract(input_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Extract PWML-structured JSON from this text:\n<<<\n{input_text}\n>>>"}
    ]
    return chat(messages, temperature=0, max_tokens=2000)


if run:
    if not text.strip():
        st.warning("Paste some text first.")
        st.stop()

    with st.spinner("Extracting..."):
        raw = one_shot_extract(text)

    if show_raw:
        st.subheader("Raw model output")
        st.code(raw, language="json")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"JSON parse failed: {e}")
        st.stop()

    st.subheader("Parsed JSON")
    st.json(obj)

    # Optional QA if you refactor qa_graph into callable funcs
    # adj, meta = build_graph(obj)
    # comps = connected_components(adj)
    # deg = degrees(adj)
    # st.subheader("Graph QA")
    # st.write({"n_components": len(comps), "main_component_size": max((len(c) for c in comps), default=0)})
