import streamlit as st
from src.llm_client import chat

st.set_page_config(page_title="LLM Demo", layout="centered")
st.title("LM Studio Chat")

# Keep chat history across interaction
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Show existing chat (skip showing system message)
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box
user_text = st.chat_input("Type your message...")

if user_text:
    # Add user msg
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Call your local LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = chat(st.session_state.messages, temperature=0.2, max_tokens=800)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})