"""Smoke test: verify the configured free LLM providers (Gemini and, if GROQ_API_KEY is set, Groq) support
tool calling.

Usage:  python scripts/check_llm.py
Reads GEMINI_API_KEY / GEMINI_MODEL / GROQ_API_KEY / GROQ_MODEL from .env (see .env.example) and sends one chat
request per provider with a single dummy tool (get_time). Passes if each model returns a tool call.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from agent.llm import OpenAIChat, providers_from_env  # noqa: E402

load_dotenv()

TOOLS = [{"type": "function", "function": {
    "name": "get_time",
    "description": "Return the current time in a given IANA timezone.",
    "parameters": {"type": "object",
                   "properties": {"timezone": {"type": "string", "description": "e.g. America/New_York"}},
                   "required": ["timezone"]},
}}]
QUESTION = [{"role": "user", "content": "What time is it in New York? Use the tool."}]

failed = False
for role, provider in zip(("Primary", "Failover"), providers_from_env()):
    if provider is None:
        print(f"{role}: not configured")
        continue
    response = OpenAIChat(provider).complete(QUESTION, TOOLS)     # this provider alone
    print(f"{role}: {provider.name} {provider.model}")
    if response.tool_calls:
        call = response.tool_calls[0]
        print(f"  OK - tool call: {call.name}({json.loads(call.arguments)})")
    else:
        print(f"  FAIL - no tool call. Model replied: {response.content!r}")
        failed = True
sys.exit(1 if failed else 0)
