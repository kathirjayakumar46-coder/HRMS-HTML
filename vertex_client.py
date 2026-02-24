# vertex_client.py

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Generator
from google import genai


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")


# ─────────────────────────────
# VALIDATION
# ─────────────────────────────
if not API_KEY or API_KEY.strip() == "":
    raise ValueError(
        "\n❌ GOOGLE_API_KEY missing.\n"
        "Create a .env file and add:\n"
        "GOOGLE_API_KEY=your_key_here\n"
    )


# ─────────────────────────────
# CLIENT INIT
# ─────────────────────────────
client = genai.Client(api_key=API_KEY)


# ─────────────────────────────
# STREAM GENERATION
# ─────────────────────────────
def stream_generate(prompt: str) -> Generator[str, None, None]:
    """
    Streams Gemini output tokens.
    Used for SSE streaming APIs.
    """

    try:
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=prompt
        )

        for chunk in stream:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"[LLM ERROR] {str(e)}"


# ─────────────────────────────
# NORMAL GENERATION
# ─────────────────────────────
def generate(prompt: str) -> str:
    """
    Non-stream version.
    Useful for testing or batch jobs.
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text or ""

    except Exception as e:
        return f"[LLM ERROR] {str(e)}"


# ─────────────────────────────
# CONNECTION TEST
# ─────────────────────────────
def test_connection() -> bool:
    """
    Checks if API + model works.
    Call during startup if needed.
    """

    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents="Reply with OK"
        )

        return bool(res.text)

    except Exception:
        return False