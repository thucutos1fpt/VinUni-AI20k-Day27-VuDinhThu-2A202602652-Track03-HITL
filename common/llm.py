"""LLM factory. Returns an OpenAI-compatible chat model wired to Gemini."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set — add a Gemini API key to .env")
    return ChatOpenAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
        temperature=temperature,
    )
