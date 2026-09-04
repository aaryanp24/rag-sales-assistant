"""
Single entry point for LLM calls so the rest of the app doesn't care which
free provider is behind it. Set LLM_PROVIDER=groq or gemini in .env.
"""
from __future__ import annotations
from config import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GEMINI_API_KEY, GEMINI_MODEL


def generate(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    if LLM_PROVIDER == "groq":
        return _generate_groq(system_prompt, user_prompt, temperature)
    if LLM_PROVIDER == "gemini":
        return _generate_gemini(system_prompt, user_prompt, temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _generate_groq(system_prompt: str, user_prompt: str, temperature: float) -> str:
    from groq import Groq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def _generate_gemini(system_prompt: str, user_prompt: str, temperature: float) -> str:
    import google.generativeai as genai
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
    resp = model.generate_content(user_prompt, generation_config={"temperature": temperature})
    return resp.text
