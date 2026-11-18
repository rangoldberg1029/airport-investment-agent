"""Speech-to-text for voice questions: Whisper on Groq (free tier). Needs GROQ_API_KEY."""

import io
import os

from agent.llm import GROQ_BASE_URL

TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
PROMPT = ("US airports and IATA codes, e.g. LAX, SNA, SFO, BOS, ANC, JFK; New England; "
          "load factor; congestion; terminal expansion.")   # vocabulary hint for Whisper


def voice_available() -> bool:
    """Voice input works only with a Groq key (Gemini's OpenAI-compatible endpoint has no Whisper)."""
    return bool(os.getenv("GROQ_API_KEY"))


def transcribe(audio_bytes: bytes, filename: str = "question.wav") -> str:
    """Return the transcript of a short recording. Raises on any provider error (caller shows a notice)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url=GROQ_BASE_URL, max_retries=1)
    audio = io.BytesIO(audio_bytes)
    audio.name = filename
    result = client.audio.transcriptions.create(model=TRANSCRIPTION_MODEL, file=audio, prompt=PROMPT,
                                                language="en", response_format="text")
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return text.strip()
