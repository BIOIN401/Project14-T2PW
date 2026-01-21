# llm_client.py
# Minimal LM Studio (OpenAI-compatible) client wrapper.

import os
from openai import OpenAI

# Defaults work out-of-the-box for LM Studio Local Server.
# override the env variables, we can remove those later if we see fit 
BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LMSTUDIO_MODEL", "meta-llama-3.1-8b-instruct")

# LM Studio doesn't require a real API key, but the OpenAI client expects one.
client = OpenAI(base_url=BASE_URL, api_key="lm-studio")


def chat(messages, temperature: float = 0.0, max_tokens: int = 800) -> str:
    """
    Send chat messages to the locally hosted LM Studio model and return raw text.
    messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = resp.choices[0].message.content
    return (content or "").strip()
