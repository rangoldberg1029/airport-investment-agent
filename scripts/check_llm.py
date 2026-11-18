"""Smoke test: verify the configured free LLM provider supports tool calling.

Usage:  python scripts/check_llm.py
Reads LLM_API_KEY, LLM_BASE_URL, LLM_MODEL from .env and sends one chat request
with a single dummy tool (get_time). Passes if the model returns a tool call.
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

if not api_key:
    sys.exit("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")

client = OpenAI(api_key=api_key, base_url=base_url)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Return the current time in a given IANA timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "e.g. America/New_York"}
                },
                "required": ["timezone"],
            },
        },
    }
]

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What time is it in New York? Use the tool."}],
    tools=tools,
)

message = response.choices[0].message
print(f"Provider: {base_url}\nModel:    {model}")

if message.tool_calls:
    call = message.tool_calls[0]
    print(f"OK - tool call: {call.function.name}({json.loads(call.function.arguments)})")
else:
    print(f"FAIL - no tool call. Model replied: {message.content!r}")
    sys.exit(1)
