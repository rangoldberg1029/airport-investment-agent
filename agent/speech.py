"""Speech-to-text for voice questions: Whisper on Groq (free tier). Needs GROQ_API_KEY."""

import io
import os

from agent.llm import GROQ_BASE_URL

TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
# Vocabulary hint for Whisper: domain terms and airport codes it would otherwise mishear.
PROMPT = ("Airport investment questions: unmet demand, load factor, congestion, New England, terminal expansion, "
          "long-haul, capacity, delays, ranking, growth. Airport codes: LAX, SNA, SFO, BOS, ANC, JFK, ORD, ATL, "
          "DEN, SEA, BDL, PWM, PVD, BTV, MHT, HVN, OAK, SJC.")


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
