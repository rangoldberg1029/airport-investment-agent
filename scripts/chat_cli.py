"""Terminal chat with the agent, for quick manual testing.

Run:  python scripts/chat_cli.py      (needs .env with GEMINI_API_KEY; commands: /reset, /quit)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from agent.agent import Agent, summarize_calls  # noqa: E402
from agent.llm import OpenAIChat  # noqa: E402
from data.repository import load_frames  # noqa: E402

if __name__ == "__main__":
    load_dotenv()
    agent = Agent(OpenAIChat.from_env(), load_frames())
    print("Airport investment agent - ask a question (/reset, /quit)")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in ("/quit", "/exit"):
            break
        if question == "/reset":
            agent.reset()
            print("(conversation cleared)")
            continue
        if not question:
            continue
        result = agent.ask(question)
        print(f"\nagent> {result.text}")
        for line in summarize_calls(result.tool_calls):
            print(f"  [tool] {line}")
        for n in result.untraceable_numbers:
            print(f"  [WARNING untraceable number] {n['value']}: ...{n['context']}...")
        print(f"  [{result.llm_calls} LLM calls, {result.total_tokens:,} tokens]")
