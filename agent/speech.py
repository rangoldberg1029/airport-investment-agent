"""Speech-to-text for voice questions: Groq Whisper via the same OpenAI-compatible key (free tier)."""

import io
import os

TRANSCRIPTION_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
PROMPT = ("US airports and IATA codes, e.g. LAX, SNA, SFO, BOS, ANC, JFK; New England; "
          "load factor; congestion; terminal expansion.")   # vocabulary hint for Whisper


def transcribe(audio_bytes: bytes, filename: str = "question.wav") -> str:
    """Return the transcript of a short recording. Raises on any provider error (caller shows a notice)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("LLM_API_KEY"),
                    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"), max_retries=1)
    audio = io.BytesIO(audio_bytes)
    audio.name = filename
    result = client.audio.transcriptions.create(model=TRANSCRIPTION_MODEL, file=audio, prompt=PROMPT,
                                                language="en", response_format="text")
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return text.strip()
